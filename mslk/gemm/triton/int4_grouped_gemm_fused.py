# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

"""
Native fused BF16xINT4 grouped GEMM for ROCm/AMD GPUs.

A single kernel handles all G groups, iterating over groups in an inner
tl.range loop (following _mslk_grouped_gemm pattern).  This eliminates
the G×Python-wrapper overhead of the loop-based implementation.

Unlike the rowwise kernel, activations X are loaded directly from the
interleaved [M_total, K] layout using stride-2 column access, avoiding
the pre-split strided copies done in the Python wrapper.

Weight layout:
  WQ           : [G, N, K//2]        int8 — packed INT4
  w_scale_group: [G, num_groups, N]  float32 or bfloat16
  w_zero_group : [G, num_groups, N]  float32 or bfloat16
  M_sizes      : [G]                 int64 — rows per group

Constraint: 2 * BLOCK_K must divide group_size.
"""

from typing import List

import torch
import triton  # @manual
import triton.language as tl  # @manual
from triton import Config  # @manual


# ---------------------------------------------------------------------------
# Autotuning configs
# ---------------------------------------------------------------------------


def _get_grouped_configs() -> List[Config]:
    configs = []
    for bm in [32, 64, 128]:
        for bn in [32, 64, 128, 256]:
            for bk in [32, 64, 128]:
                for nw in [4, 8]:
                    for ns in [1, 2]:
                        for gsm in [4, 8]:
                            configs.append(
                                Config(
                                    {
                                        "BLOCK_M": bm,
                                        "BLOCK_N": bn,
                                        "BLOCK_K": bk,
                                        "GROUP_SIZE_M": gsm,
                                    },
                                    num_warps=nw,
                                    num_stages=ns,
                                )
                            )
    return configs


def _prune_grouped_configs(configs, named_args, **kwargs):
    all_args = {**named_args, **kwargs}
    group_size = all_args["group_size"]
    N = all_args["N"]
    K2 = all_args["K2"]
    pruned = []
    for c in configs:
        bm = c.kwargs["BLOCK_M"]
        bn = c.kwargs["BLOCK_N"]
        bk = c.kwargs["BLOCK_K"]
        nw = c.num_warps
        ns = c.num_stages
        if group_size % (2 * bk) != 0:
            continue
        if K2 % bk != 0:
            continue
        if bn > max(N, 32):
            continue
        tile_elems = bm * bn
        if tile_elems <= 2048 and nw == 8:
            continue
        if tile_elems >= 16384 and nw == 4:
            continue
        if tile_elems >= 16384 and ns >= 3:
            continue
        pruned.append(c)
    return pruned


# ---------------------------------------------------------------------------
# Core Triton kernel
# ---------------------------------------------------------------------------


@triton.autotune(
    configs=_get_grouped_configs(),
    key=["G", "M_TOTAL", "N", "K2", "group_size"],
    prune_configs_by={"early_config_prune": _prune_grouped_configs},
)
@triton.heuristics(
    {
        "EVEN_K": lambda args: args["K2"] % args["BLOCK_K"] == 0,
    }
)
@triton.jit
def _bf16i4_grouped_kernel(
    X_ptr,  # [M_total, K]         bfloat16 — interleaved even/odd K columns
    WQ_ptr,  # [G, N, K//2]        int8 packed (lo nibble = even K, hi = odd K)
    Y_ptr,  # [M_total, N]         bfloat16
    scale_ptr,  # [G, num_groups, N]
    zero_ptr,  # [G, num_groups, N]
    m_sizes_ptr,  # [G]            int64 — rows per group
    M_TOTAL,
    N,
    K2,  # K // 2
    group_size,
    stride_xm,  # X row stride (= K)
    stride_wg,  # WQ group stride (= N * K2)
    stride_wn,  # WQ row stride (= K2)
    stride_wk,  # WQ col stride (= 1)
    stride_ym,  # Y row stride (= N)
    stride_sg,  # scale quant-group stride (= N)
    stride_sng,  # scale expert stride (= num_groups * N)
    stride_sn,  # scale col stride (= 1)
    G: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    EVEN_K: tl.constexpr,
) -> None:
    """
    Fused grouped BF16xINT4 GEMM.

    The wrapper pre-computes cumulative row starts (m_starts) and cumulative
    tile counts (tile_cumsums) so each CTA can find its group with a simple
    O(G) scan — no nested prefix-sum loop inside the kernel.

    Activations are loaded from the interleaved [M_total, K] layout with
    stride-2 column offsets (even col k2 → X column 2*k2, odd → 2*k2+1).
    """
    pid = tl.program_id(0)
    num_n_tiles = tl.cdiv(N, BLOCK_N)

    # Find which group this CTA belongs to with a single O(G) forward scan.
    # We accumulate the running tile count and stop when pid < running total.
    g = 0
    g_tile_start = tl.zeros([], dtype=tl.int64)
    M_start = tl.zeros([], dtype=tl.int64)
    m_size = tl.load(m_sizes_ptr).to(tl.int64)  # default: group 0

    for gg in tl.range(G):
        ms = tl.load(m_sizes_ptr + gg).to(tl.int64)
        n_tiles_g = tl.cdiv(ms, BLOCK_M) * num_n_tiles
        next_tile_end = g_tile_start + n_tiles_g
        if pid >= next_tile_end:
            g = gg + 1
            g_tile_start = next_tile_end
            M_start = M_start + ms
            if gg + 1 < G:
                m_size = tl.load(m_sizes_ptr + gg + 1).to(tl.int64)

    # Local tile index within this group
    local_pid = pid - g_tile_start
    num_m_tiles = tl.cdiv(m_size, BLOCK_M)
    pid_m = local_pid % num_m_tiles
    pid_n = local_pid // num_m_tiles

    # Offsets
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_m_abs = M_start + offs_m
    offs_n_g = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    m_mask = offs_m < m_size
    n_mask = offs_n_g < N

    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    W_g = WQ_ptr + g.to(tl.int64) * stride_wg
    S_g = scale_ptr + g.to(tl.int64) * stride_sng
    Z_g = zero_ptr + g.to(tl.int64) * stride_sng

    for k2_idx in tl.range(0, tl.cdiv(K2, BLOCK_K)):
        k2_start = k2_idx * BLOCK_K
        offs_k2 = k2_start + tl.arange(0, BLOCK_K)

        # X[m, 2*k] = even activations, X[m, 2*k+1] = odd activations
        x_even_offs = offs_m_abs[:, None] * stride_xm + offs_k2[None, :] * 2
        x_odd_offs = x_even_offs + 1

        if EVEN_K:
            xm_mask = m_mask[:, None]
            x_even = tl.load(X_ptr + x_even_offs, mask=xm_mask, other=0.0).to(
                tl.bfloat16
            )
            x_odd = tl.load(X_ptr + x_odd_offs, mask=xm_mask, other=0.0).to(tl.bfloat16)
            w_q = tl.load(
                W_g + offs_n_g[:, None] * stride_wn + offs_k2[None, :] * stride_wk,
                mask=n_mask[:, None],
                other=0,
            ).to(tl.int32)
        else:
            k_mask = offs_k2[None, :] < K2
            xmk_mask = m_mask[:, None] & k_mask
            x_even = tl.load(X_ptr + x_even_offs, mask=xmk_mask, other=0.0).to(
                tl.bfloat16
            )
            x_odd = tl.load(X_ptr + x_odd_offs, mask=xmk_mask, other=0.0).to(
                tl.bfloat16
            )
            w_q = tl.load(
                W_g + offs_n_g[:, None] * stride_wn + offs_k2[None, :] * stride_wk,
                mask=n_mask[:, None] & k_mask,
                other=0,
            ).to(tl.int32)

        group_idx = (k2_start * 2) // group_size
        s = tl.load(
            S_g + group_idx * stride_sg + offs_n_g * stride_sn,
            mask=n_mask,
            other=0.0,
        ).to(tl.float32)
        z = tl.load(
            Z_g + group_idx * stride_sg + offs_n_g * stride_sn,
            mask=n_mask,
            other=0.0,
        ).to(tl.float32)

        w_lo = w_q & 0x0F
        w_hi = (w_q >> 4) & 0x0F
        s_col = s[:, None]
        z_col = z[:, None]
        w_lo_dq = ((w_lo ^ 8) - 8).to(tl.float32) * s_col + z_col
        w_hi_dq = ((w_hi ^ 8) - 8).to(tl.float32) * s_col + z_col

        x_fused = tl.cat(x_even, x_odd, dim=1)
        w_fused = tl.cat(w_lo_dq.to(tl.bfloat16), w_hi_dq.to(tl.bfloat16), dim=1)
        acc = tl.dot(x_fused, tl.trans(w_fused), acc, out_dtype=tl.float32)

    y_ptrs = Y_ptr + offs_m_abs[:, None] * stride_ym + offs_n_g[None, :] * 1
    out_mask = m_mask[:, None] & n_mask[None, :]
    tl.store(y_ptrs, acc.to(tl.bfloat16), mask=out_mask)


# ---------------------------------------------------------------------------
# Python wrapper
# ---------------------------------------------------------------------------


def matmul_bf16i4_rowwise_grouped_fused(
    X: torch.Tensor,
    WQ: torch.Tensor,
    w_scale_group: torch.Tensor,
    w_zero_group: torch.Tensor,
    M_sizes: torch.Tensor,
) -> torch.Tensor:
    """
    Fused grouped BF16xINT4 GEMM — single kernel launch for all G groups.

    Args:
        X             : [M_total, K]          bfloat16 activations
        WQ            : [G, N, K//2]          int8 packed weights per group
        w_scale_group : [G, num_groups, N]    per-group scales
        w_zero_group  : [G, num_groups, N]    per-group zero points
        M_sizes       : [G]                   rows per group (int64, on device)

    Returns:
        Y : [M_total, N]  bfloat16
    """
    G = WQ.shape[0]
    M_total = X.shape[0]
    N = WQ.shape[1]
    K = X.shape[1]
    K2 = K // 2
    num_groups = w_scale_group.shape[1]
    group_size = K // num_groups

    assert X.dtype == torch.bfloat16, "X must be bfloat16"
    assert WQ.dtype == torch.int8, "WQ must be int8"
    assert X.is_contiguous(), "X must be contiguous"
    assert group_size % 64 == 0, f"group_size={group_size} must be divisible by 64"

    # M_sizes must be on device and int64 for pointer arithmetic in kernel
    if M_sizes.dtype != torch.int64:
        M_sizes = M_sizes.to(torch.int64)
    if not M_sizes.is_cuda:
        M_sizes = M_sizes.to(X.device)

    # Scale/zero: ensure float32
    if w_scale_group.dtype != torch.float32:
        w_scale_group = w_scale_group.to(torch.float32)
    if w_zero_group.dtype != torch.float32:
        w_zero_group = w_zero_group.to(torch.float32)

    Y = torch.empty((M_total, N), dtype=torch.bfloat16, device=X.device)

    m_sizes_cpu = M_sizes.tolist()

    def grid(meta):
        t = sum(
            triton.cdiv(m, meta["BLOCK_M"]) * triton.cdiv(N, meta["BLOCK_N"])
            for m in m_sizes_cpu
        )
        return (t,)

    _bf16i4_grouped_kernel[grid](
        X,
        WQ,
        Y,
        w_scale_group,
        w_zero_group,
        M_sizes,
        M_TOTAL=M_total,
        N=N,
        K2=K2,
        group_size=group_size,
        stride_xm=X.stride(0),
        stride_wg=WQ.stride(0),
        stride_wn=WQ.stride(1),
        stride_wk=WQ.stride(2),
        stride_ym=Y.stride(0),
        stride_sg=w_scale_group.stride(1),
        stride_sng=w_scale_group.stride(0),
        stride_sn=w_scale_group.stride(2),
        G=G,
    )

    return Y
