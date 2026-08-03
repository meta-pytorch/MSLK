# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

"""
Native fused BF16xINT4 grouped GEMM for ROCm/AMD GPUs.

A single kernel handles all G groups via a 3D grid (G, M-tiles, N-tiles).
Over-provisioned M-tiles that exceed their group's row count early-exit,
following the pattern in fp8_groupwise_grouped_gemm.py.  No device-to-host
sync is needed, so the wrapper captures cleanly inside a CUDA graph.

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

import inspect
from typing import List

import torch
import triton  # @manual
import triton.language as tl  # @manual
from triton import Config  # @manual

_TL_CAT_HAS_DIM = "dim" in inspect.signature(tl.cat).parameters


# ---------------------------------------------------------------------------
# Autotuning configs
# ---------------------------------------------------------------------------


def _num_warps_for_tile(bm: int, bn: int) -> List[int]:
    tile = bm * bn
    if tile <= 2048:
        return [4]
    elif tile >= 16384:
        return [8]
    else:
        return [4, 8]


def _get_grouped_configs() -> List[Config]:
    configs = []
    for bm in [32, 64, 128]:
        for bn in [32, 64, 128, 256]:
            for bk in [32, 64, 128]:
                for nw in _num_warps_for_tile(bm, bn):
                    configs.append(
                        Config(
                            {
                                "BLOCK_M": bm,
                                "BLOCK_N": bn,
                                "BLOCK_K": bk,
                            },
                            num_warps=nw,
                            num_stages=2,
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
        bn = c.kwargs["BLOCK_N"]
        bk = c.kwargs["BLOCK_K"]
        if group_size % (2 * bk) != 0:
            continue
        if K2 % bk != 0:
            continue
        if bn > max(N, 32):
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
    m_starts_ptr,  # [G]           int64 — cumulative row offsets per group
    M_TOTAL,
    G,
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
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    EVEN_K: tl.constexpr,
    FUSE_DOT: tl.constexpr,
) -> None:
    """
    Fused grouped BF16xINT4 GEMM.

    Grid: (G, ceil(M_total / BLOCK_M), ceil(N / BLOCK_N)).
    Tiles whose pid_m falls outside their group's M are discarded early.

    Activations are loaded from the interleaved [M_total, K] layout with
    stride-2 column offsets (even col k2 -> X column 2*k2, odd -> 2*k2+1).

    Constraint: 2 * BLOCK_K must divide group_size.
    """
    pid_g = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_n = tl.program_id(2)

    m_size = tl.load(m_sizes_ptr + pid_g).to(tl.int64)
    M_start = tl.load(m_starts_ptr + pid_g).to(tl.int64)

    if pid_m * BLOCK_M >= m_size:
        return

    # Offsets
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_m_abs = M_start + offs_m
    offs_n_g = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    m_mask = offs_m < m_size
    n_mask = offs_n_g < N

    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    W_g = WQ_ptr + pid_g.to(tl.int64) * stride_wg
    S_g = scale_ptr + pid_g.to(tl.int64) * stride_sng
    Z_g = zero_ptr + pid_g.to(tl.int64) * stride_sng

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

        if FUSE_DOT:
            x_fused = tl.cat(x_even, x_odd, dim=1)
            w_fused = tl.cat(w_lo_dq.to(tl.bfloat16), w_hi_dq.to(tl.bfloat16), dim=1)
            acc = tl.dot(x_fused, tl.trans(w_fused), acc, out_dtype=tl.float32)
        else:
            acc = tl.dot(
                x_even,
                tl.trans(w_lo_dq.to(tl.bfloat16)),
                acc,
                out_dtype=tl.float32,
            )
            acc = tl.dot(
                x_odd,
                tl.trans(w_hi_dq.to(tl.bfloat16)),
                acc,
                out_dtype=tl.float32,
            )

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

    CUDA-graph safe: no device-to-host sync.  The grid is over-provisioned
    using M_total (a host int from tensor metadata) instead of max(M_sizes);
    tiles that exceed their group's row count early-exit in the kernel.

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
    assert M_sizes.dtype == torch.int64, f"M_sizes must be int64, got {M_sizes.dtype}"
    assert M_sizes.is_cuda, "M_sizes must be on the GPU"
    assert w_scale_group.dtype == torch.float32, (
        f"w_scale_group must be float32, got {w_scale_group.dtype}"
    )
    assert w_zero_group.dtype == torch.float32, (
        f"w_zero_group must be float32, got {w_zero_group.dtype}"
    )
    assert w_scale_group.is_contiguous(), "w_scale_group must be contiguous"
    assert w_zero_group.is_contiguous(), "w_zero_group must be contiguous"

    Y = torch.empty((M_total, N), dtype=torch.bfloat16, device=X.device)

    # Derive M_starts (cumulative row offsets) on the GPU — no device-to-host
    # sync, so this captures cleanly inside a CUDA graph.
    M_starts = M_sizes.cumsum(0) - M_sizes

    # The M dimension of the grid is bounded by M_total (a host int already in
    # hand) rather than max group size, avoiding a .item() sync.  Tiles that
    # fall outside their group's M are discarded early in the kernel, so the
    # extra launches for uneven groups are cheap no-ops.
    grid = lambda meta: (  # noqa: E731
        G,
        triton.cdiv(M_total, meta["BLOCK_M"]),
        triton.cdiv(N, meta["BLOCK_N"]),
    )

    _bf16i4_grouped_kernel[grid](
        X,
        WQ,
        Y,
        w_scale_group,
        w_zero_group,
        M_sizes,
        M_starts,
        M_TOTAL=M_total,
        G=G,
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
        FUSE_DOT=_TL_CAT_HAS_DIM,
    )

    return Y
