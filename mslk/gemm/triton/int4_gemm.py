# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

"""
BF16 x INT4 weight-only GEMM for ROCm/AMD GPUs.

Weight layout (matches the CUDA/CUTLASS rowwise path):
  W  : [N, K//2]   int8  — two INT4 nibbles per byte:
                            lo nibble (bits 0-3) = W[n, 2k]
                            hi nibble (bits 4-7) = W[n, 2k+1]
  w_scale_group : [num_groups, N]  float32 or bfloat16
  w_zero_group  : [num_groups, N]  float32 or bfloat16
                  where num_groups = K // group_size

Dequantisation formula (matches int4_row_quantize + pack_int4 in gemm_test.py):
  w_float = w_signed * scale + zero
  where w_signed is the signed int4 two's-complement decoded from the nibble:
    signed = (nibble XOR 8) - 8
  nibble ∈ [0,7]  → signed ∈ [0,7]   (positive; bit3 of nibble is clear)
  nibble ∈ [8,15] → signed ∈ [-8,-1] (negative; bit3 of nibble is set)

Activation pre-split strategy (avoids LDS bank conflicts):
  The packed INT4 format stores even-K and odd-K weight values interleaved in
  each byte.  Separating them inside the Triton kernel requires a cross-lane
  data movement that ROCm Triton lowers through LDS, causing severe bank
  conflicts.  Instead, the Python wrapper splits the activation matrix X into
  two contiguous [M, K//2] tensors (even and odd K columns) before kernel
  launch using optimised PyTorch/HIP strided copies.  The kernel concatenates
  x_even and x_odd along K via tl.cat and issues a single fused tl.dot against
  the concatenated dequantised weights — one LDS staging pass instead of two.
"""

from typing import List

import torch
import triton  # @manual
import triton.language as tl  # @manual
from triton import Config  # @manual

# ---------------------------------------------------------------------------
# Autotuning configs
# ---------------------------------------------------------------------------


_MAX_SPLIT_K = 8  # largest SPLIT_K value in the config sweep


def _get_configs() -> List[Config]:
    configs = []
    for bm in [32, 64, 128, 256]:
        for bn in [32, 64, 128, 256]:
            for bk in [32, 64, 128]:
                for nw in [4, 8]:
                    for ns in [1, 2]:
                        for gsm in [4, 8]:
                            for sk in [1, 2, 4, 8]:
                                configs.append(
                                    Config(
                                        {
                                            "BLOCK_M": bm,
                                            "BLOCK_N": bn,
                                            "BLOCK_K": bk,
                                            "GROUP_SIZE_M": gsm,
                                            "SPLIT_K": sk,
                                        },
                                        num_warps=nw,
                                        num_stages=ns,
                                    )
                                )
    return configs


def _prune_configs(configs, named_args, **kwargs):
    group_size = named_args["group_size"]
    M = named_args["M"]
    N = named_args["N"]
    K2 = named_args["K2"]
    pruned = []
    for c in configs:
        bm = c.kwargs["BLOCK_M"]
        bn = c.kwargs["BLOCK_N"]
        bk = c.kwargs["BLOCK_K"]
        sk = c.kwargs.get("SPLIT_K", 1)
        nw = c.num_warps
        ns = c.num_stages
        # 2*BLOCK_K must divide group_size (dequant alignment)
        if group_size % (2 * bk) != 0:
            continue
        # K2 must be cleanly divisible by BLOCK_K * SPLIT_K
        if K2 % (bk * sk) != 0:
            continue
        # SPLIT_K > 1 only helps small M (otherwise adds atomic overhead for no gain)
        if sk > 1 and M >= 512:
            continue
        # Skip tiles much larger than the problem dimension
        if bm > max(M, 32) or bn > max(N, 32):
            continue
        # Small tiles don't need 8 warps; large tiles don't benefit from 4
        tile_elems = bm * bn
        if tile_elems <= 2048 and nw == 8:
            continue
        if tile_elems >= 16384 and nw == 4:
            continue
        # Large tiles with many stages cause register spills
        if tile_elems >= 16384 and ns >= 3:
            continue
        pruned.append(c)
    return pruned


# ---------------------------------------------------------------------------
# Core Triton kernel
# ---------------------------------------------------------------------------


@triton.autotune(
    configs=_get_configs(),
    key=["M", "N", "K2", "group_size"],
    prune_configs_by={"early_config_prune": _prune_configs},
)
@triton.heuristics(
    {
        "EVEN_K": lambda args: args["K2"] % (args["BLOCK_K"] * args["SPLIT_K"]) == 0,
        "EVEN_MN": lambda args: (args["M"] % args["BLOCK_M"] == 0)
        and (args["N"] % args["BLOCK_N"] == 0),
        "GRID_MN": lambda args: triton.cdiv(args["M"], args["BLOCK_M"])
        * triton.cdiv(args["N"], args["BLOCK_N"]),
    }
)
@triton.jit
def _bf16i4_rowwise_kernel(
    X_even_ptr,  # [M, K//2]  bfloat16 — even K columns of activations
    X_odd_ptr,  # [M, K//2]  bfloat16 — odd  K columns of activations
    W_ptr,  # [N, K//2]  int8 packed (lo nibble = even K, hi = odd K)
    Y_ptr,  # [M, N]     bfloat16
    scale_ptr,  # [num_groups, N]
    zero_ptr,  # [num_groups, N]
    M,
    N,
    K2,  # K // 2
    group_size,  # original group_size over full K
    stride_xm,
    stride_xk,  # strides for x_even / x_odd (same shape [M, K2])
    stride_wn,
    stride_wk,
    stride_yz,  # stride of SPLIT_K dimension in workspace [SPLIT_K, M, N]
    stride_ym,
    stride_yn,
    stride_sg,
    stride_sn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,  # tile size over K2; inner-dot K for each tl.dot
    GROUP_SIZE_M: tl.constexpr,
    SPLIT_K: tl.constexpr,
    EVEN_K: tl.constexpr,
    EVEN_MN: tl.constexpr,
    GRID_MN: tl.constexpr,
) -> None:
    """
    Computes Y[M, N] = X[M, K] @ dequant(W)[K, N].

    X has been pre-split into x_even [M, K//2] (even K columns) and
    x_odd [M, K//2] (odd K columns) by the Python wrapper.  Each kernel
    iteration concatenates x_even and x_odd along K and issues a single
    fused tl.dot against the concatenated dequantised weights.

    Constraint: 2 * BLOCK_K must divide group_size.
    """
    NUM_XCDS: tl.constexpr = 8
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)

    # XCD remapping: spread PIDs evenly across 8 chiplets
    pids_per_xcd = (GRID_MN + NUM_XCDS - 1) // NUM_XCDS
    xcd = pid % NUM_XCDS
    local_pid = pid // NUM_XCDS
    tall_xcds = GRID_MN % NUM_XCDS
    tall_xcds = NUM_XCDS if tall_xcds == 0 else tall_xcds
    if xcd < tall_xcds:
        pid = xcd * pids_per_xcd + local_pid
    else:
        pid = (
            tall_xcds * pids_per_xcd
            + (xcd - tall_xcds) * (pids_per_xcd - 1)
            + local_pid
        )

    # Grouped PID → (pid_m, pid_n) with L2-friendly M-grouping
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    tl.assume(stride_xm > 0)
    tl.assume(stride_xk > 0)
    tl.assume(stride_wn > 0)
    tl.assume(stride_wk > 0)
    tl.assume(stride_ym > 0)
    tl.assume(stride_yn > 0)
    tl.assume(stride_sg > 0)
    tl.assume(stride_sn > 0)

    if EVEN_MN:
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    else:
        offs_m = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
        offs_n = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N

    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    pid_z = tl.program_id(1)
    k2_per_split = tl.cdiv(K2, SPLIT_K)
    k2_slice_start = pid_z * k2_per_split
    k2_slice_end = tl.minimum(k2_slice_start + k2_per_split, K2)
    num_iters = tl.cdiv(k2_slice_end - k2_slice_start, BLOCK_K)

    # ---- prologue: issue loads for iteration 0 ----
    offs_k2 = k2_slice_start + tl.arange(0, BLOCK_K)
    if EVEN_MN and EVEN_K:
        x_even = tl.load(
            X_even_ptr + offs_m[:, None] * stride_xm + offs_k2[None, :] * stride_xk,
        ).to(tl.bfloat16)
        x_odd = tl.load(
            X_odd_ptr + offs_m[:, None] * stride_xm + offs_k2[None, :] * stride_xk,
        ).to(tl.bfloat16)
        w_q = tl.load(
            W_ptr + offs_n[:, None] * stride_wn + offs_k2[None, :] * stride_wk,
        ).to(tl.int32)
    elif EVEN_MN:
        k_mask = offs_k2[None, :] < K2
        x_even = tl.load(
            X_even_ptr + offs_m[:, None] * stride_xm + offs_k2[None, :] * stride_xk,
            mask=k_mask,
            other=0.0,
        ).to(tl.bfloat16)
        x_odd = tl.load(
            X_odd_ptr + offs_m[:, None] * stride_xm + offs_k2[None, :] * stride_xk,
            mask=k_mask,
            other=0.0,
        ).to(tl.bfloat16)
        w_q = tl.load(
            W_ptr + offs_n[:, None] * stride_wn + offs_k2[None, :] * stride_wk,
            mask=k_mask,
            other=0,
        ).to(tl.int32)
    elif EVEN_K:
        xk_mask = offs_m[:, None] < M
        wk_mask = offs_n[:, None] < N
        x_even = tl.load(
            X_even_ptr + offs_m[:, None] * stride_xm + offs_k2[None, :] * stride_xk,
            mask=xk_mask,
            other=0.0,
        ).to(tl.bfloat16)
        x_odd = tl.load(
            X_odd_ptr + offs_m[:, None] * stride_xm + offs_k2[None, :] * stride_xk,
            mask=xk_mask,
            other=0.0,
        ).to(tl.bfloat16)
        w_q = tl.load(
            W_ptr + offs_n[:, None] * stride_wn + offs_k2[None, :] * stride_wk,
            mask=wk_mask,
            other=0,
        ).to(tl.int32)
    else:
        xk_mask = (offs_m[:, None] < M) & (offs_k2[None, :] < K2)
        wk_mask = (offs_n[:, None] < N) & (offs_k2[None, :] < K2)
        x_even = tl.load(
            X_even_ptr + offs_m[:, None] * stride_xm + offs_k2[None, :] * stride_xk,
            mask=xk_mask,
            other=0.0,
        ).to(tl.bfloat16)
        x_odd = tl.load(
            X_odd_ptr + offs_m[:, None] * stride_xm + offs_k2[None, :] * stride_xk,
            mask=xk_mask,
            other=0.0,
        ).to(tl.bfloat16)
        w_q = tl.load(
            W_ptr + offs_n[:, None] * stride_wn + offs_k2[None, :] * stride_wk,
            mask=wk_mask,
            other=0,
        ).to(tl.int32)
    group_idx = (k2_slice_start * 2) // group_size
    if EVEN_MN:
        s = tl.load(
            scale_ptr + group_idx * stride_sg + offs_n * stride_sn,
        ).to(tl.float32)
        z = tl.load(
            zero_ptr + group_idx * stride_sg + offs_n * stride_sn,
        ).to(tl.float32)
    else:
        s = tl.load(
            scale_ptr + group_idx * stride_sg + offs_n * stride_sn,
            mask=offs_n < N,
        ).to(tl.float32)
        z = tl.load(
            zero_ptr + group_idx * stride_sg + offs_n * stride_sn,
            mask=offs_n < N,
        ).to(tl.float32)

    # ---- main loop: compute iteration i, prefetch iteration i+1 ----
    # The prefetch always runs (even on the last iteration where results are
    # discarded) to avoid Triton SSA scoping issues with conditional defines.
    for k2_idx in tl.range(0, num_iters):
        k2_start = k2_slice_start + k2_idx * BLOCK_K

        # Unpack and dequant current tile (uses already-loaded data)
        w_lo = w_q & 0x0F
        w_hi = (w_q >> 4) & 0x0F
        s_col = s[:, None]
        z_col = z[:, None]
        w_lo_dq = ((w_lo ^ 8) - 8).to(tl.float32) * s_col + z_col
        w_hi_dq = ((w_hi ^ 8) - 8).to(tl.float32) * s_col + z_col

        # Prefetch next iteration (clamp to last valid tile on final iteration)
        next_k2_start = tl.minimum(k2_start + BLOCK_K, k2_slice_end - BLOCK_K)
        next_offs_k2 = next_k2_start + tl.arange(0, BLOCK_K)
        if EVEN_MN and EVEN_K:
            next_x_even = tl.load(
                X_even_ptr
                + offs_m[:, None] * stride_xm
                + next_offs_k2[None, :] * stride_xk,
            ).to(tl.bfloat16)
            next_x_odd = tl.load(
                X_odd_ptr
                + offs_m[:, None] * stride_xm
                + next_offs_k2[None, :] * stride_xk,
            ).to(tl.bfloat16)
            next_w_q = tl.load(
                W_ptr + offs_n[:, None] * stride_wn + next_offs_k2[None, :] * stride_wk,
            ).to(tl.int32)
        elif EVEN_MN:
            next_k_mask = next_offs_k2[None, :] < K2
            next_x_even = tl.load(
                X_even_ptr
                + offs_m[:, None] * stride_xm
                + next_offs_k2[None, :] * stride_xk,
                mask=next_k_mask,
                other=0.0,
            ).to(tl.bfloat16)
            next_x_odd = tl.load(
                X_odd_ptr
                + offs_m[:, None] * stride_xm
                + next_offs_k2[None, :] * stride_xk,
                mask=next_k_mask,
                other=0.0,
            ).to(tl.bfloat16)
            next_w_q = tl.load(
                W_ptr + offs_n[:, None] * stride_wn + next_offs_k2[None, :] * stride_wk,
                mask=next_k_mask,
                other=0,
            ).to(tl.int32)
        elif EVEN_K:
            next_xk_mask = offs_m[:, None] < M
            next_wk_mask = offs_n[:, None] < N
            next_x_even = tl.load(
                X_even_ptr
                + offs_m[:, None] * stride_xm
                + next_offs_k2[None, :] * stride_xk,
                mask=next_xk_mask,
                other=0.0,
            ).to(tl.bfloat16)
            next_x_odd = tl.load(
                X_odd_ptr
                + offs_m[:, None] * stride_xm
                + next_offs_k2[None, :] * stride_xk,
                mask=next_xk_mask,
                other=0.0,
            ).to(tl.bfloat16)
            next_w_q = tl.load(
                W_ptr + offs_n[:, None] * stride_wn + next_offs_k2[None, :] * stride_wk,
                mask=next_wk_mask,
                other=0,
            ).to(tl.int32)
        else:
            next_xk_mask = (offs_m[:, None] < M) & (next_offs_k2[None, :] < K2)
            next_wk_mask = (offs_n[:, None] < N) & (next_offs_k2[None, :] < K2)
            next_x_even = tl.load(
                X_even_ptr
                + offs_m[:, None] * stride_xm
                + next_offs_k2[None, :] * stride_xk,
                mask=next_xk_mask,
                other=0.0,
            ).to(tl.bfloat16)
            next_x_odd = tl.load(
                X_odd_ptr
                + offs_m[:, None] * stride_xm
                + next_offs_k2[None, :] * stride_xk,
                mask=next_xk_mask,
                other=0.0,
            ).to(tl.bfloat16)
            next_w_q = tl.load(
                W_ptr + offs_n[:, None] * stride_wn + next_offs_k2[None, :] * stride_wk,
                mask=next_wk_mask,
                other=0,
            ).to(tl.int32)
        next_group_idx = (next_k2_start * 2) // group_size
        if EVEN_MN:
            next_s = tl.load(
                scale_ptr + next_group_idx * stride_sg + offs_n * stride_sn,
            ).to(tl.float32)
            next_z = tl.load(
                zero_ptr + next_group_idx * stride_sg + offs_n * stride_sn,
            ).to(tl.float32)
        else:
            next_s = tl.load(
                scale_ptr + next_group_idx * stride_sg + offs_n * stride_sn,
                mask=offs_n < N,
            ).to(tl.float32)
            next_z = tl.load(
                zero_ptr + next_group_idx * stride_sg + offs_n * stride_sn,
                mask=offs_n < N,
            ).to(tl.float32)

        x_fused = tl.cat(x_even, x_odd, dim=1)
        w_fused = tl.cat(w_lo_dq.to(tl.bfloat16), w_hi_dq.to(tl.bfloat16), dim=1)
        acc = tl.dot(x_fused, tl.trans(w_fused), acc, out_dtype=tl.float32)

        # Rotate buffers
        x_even = next_x_even
        x_odd = next_x_odd
        w_q = next_w_q
        s = next_s
        z = next_z

    # ---- store output ----
    # Write partial sum for this K-slice into workspace[pid_z, m, n].
    # When SPLIT_K=1, pid_z=0 and this is a direct store to [M, N].
    # When SPLIT_K>1, the caller launches _bf16i4_splitk_reduce to sum slices.
    y_ptrs = (
        Y_ptr
        + pid_z * stride_yz
        + offs_m[:, None] * stride_ym
        + offs_n[None, :] * stride_yn
    )
    if EVEN_MN:
        tl.store(y_ptrs, acc)
    else:
        tl.store(y_ptrs, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


# ---------------------------------------------------------------------------
# SPLIT_K reduction kernel
# ---------------------------------------------------------------------------


@triton.jit
def _bf16i4_splitk_reduce(
    workspace_ptr,  # [SPLIT_K, M, N] float32
    Y_ptr,  # [M, N] bfloat16
    M,
    N,
    SPLIT_K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    for k in tl.range(0, SPLIT_K):
        acc += tl.load(
            workspace_ptr + k * M * N + offs_m[:, None] * N + offs_n[None, :],
            mask=mask,
        )
    tl.store(
        Y_ptr + offs_m[:, None] * N + offs_n[None, :], acc.to(tl.bfloat16), mask=mask
    )


# ---------------------------------------------------------------------------
# Python wrappers
# ---------------------------------------------------------------------------


def matmul_bf16i4_rowwise(
    X: torch.Tensor,
    W: torch.Tensor,
    w_scale_group: torch.Tensor,
    w_zero_group: torch.Tensor,
) -> torch.Tensor:
    """
    BF16 activation x INT4 weight GEMM with per-group rowwise dequantisation.

    Args:
        X             : [..., K]          bfloat16 activations
        W             : [N, K//2]         int8 (2 INT4 per byte, lo=even K, hi=odd K)
        w_scale_group : [num_groups, N]   float32 or bfloat16 per-group scales
        w_zero_group  : [num_groups, N]   float32 or bfloat16 per-group zero points

    Returns:
        Y : [..., N]  bfloat16
    """
    leading = X.shape[:-1]
    K = X.shape[-1]
    M = X.numel() // K
    N = W.shape[0]
    num_groups = w_scale_group.shape[0]
    K2 = K // 2

    assert W.shape == (N, K2), f"W must be [N, K//2], got {W.shape}"
    assert w_scale_group.shape == (
        num_groups,
        N,
    ), (
        f"w_scale_group must be [num_groups, N]={num_groups, N}, got {w_scale_group.shape}"
    )
    assert w_zero_group.shape == (
        num_groups,
        N,
    ), f"w_zero_group must be [num_groups, N]={num_groups, N}, got {w_zero_group.shape}"
    group_size = K // num_groups
    assert group_size % 64 == 0, (
        f"group_size={group_size} must be divisible by 64 (2 * BLOCK_K_min=32); "
        "common values are 64, 128, 256."
    )

    X_2d = X.reshape(M, K)

    # Split activations into even/odd K columns outside the kernel.
    # .contiguous() issues an optimised HIP strided copy, which is far cheaper
    # than the LDS bank-conflict path the kernel would take doing this internally.
    x_even = X_2d[:, 0::2].contiguous()  # [M, K//2]
    x_odd = X_2d[:, 1::2].contiguous()  # [M, K//2]

    def grid(meta):
        return (
            triton.cdiv(M, meta["BLOCK_M"]) * triton.cdiv(N, meta["BLOCK_N"]),
            meta["SPLIT_K"],
        )

    # Workspace: [MAX_SPLIT_K, M, N] float32. Each K-slice stores its partial sum at
    # workspace[pid_z]. Allocated with empty (no zeroing) since the reduction kernel
    # only reads slices [0:split_k] — unwritten slices beyond split_k are never touched.
    workspace = torch.empty((_MAX_SPLIT_K, M, N), dtype=torch.float32, device=X.device)

    _bf16i4_rowwise_kernel[grid](
        x_even,
        x_odd,
        W,
        workspace,
        w_scale_group,
        w_zero_group,
        M,
        N,
        K2,
        group_size,
        x_even.stride(0),
        x_even.stride(1),
        W.stride(0),
        W.stride(1),
        M * N,
        N,
        1,
        w_scale_group.stride(0),
        w_scale_group.stride(1),
    )

    split_k = _bf16i4_rowwise_kernel.best_config.kwargs["SPLIT_K"]

    if split_k == 1:
        return workspace[0].to(torch.bfloat16).reshape(*leading, N)

    # SPLIT_K > 1: sum partial sums across K-slices and cast to bf16.
    Y_bf16 = torch.empty((M, N), dtype=torch.bfloat16, device=X.device)
    reduce_grid = (triton.cdiv(M, 32), triton.cdiv(N, 32))
    _bf16i4_splitk_reduce[reduce_grid](
        workspace,
        Y_bf16,
        M,
        N,
        SPLIT_K=split_k,
        BLOCK_M=32,
        BLOCK_N=32,
    )
    return Y_bf16.reshape(*leading, N)


def matmul_bf16i4_rowwise_batched(
    X: torch.Tensor,
    W: torch.Tensor,
    w_scale: torch.Tensor,
    w_zp: torch.Tensor,
) -> torch.Tensor:
    """
    Batched BF16 x INT4 GEMM (prototype: loops over the batch dimension).

    Args:
        X       : [B, M, K]              bfloat16
        W       : [B, N, K//2]           int8
        w_scale : [B * num_groups, N]    float32 or bfloat16
        w_zp    : [B * num_groups, N]    float32 or bfloat16

    Returns:
        Y : [B, M, N]  bfloat16
    """
    B, M, K = X.shape
    _, N, _ = W.shape
    num_groups_total = w_scale.shape[0]
    assert num_groups_total % B == 0, (
        f"w_scale.shape[0]={num_groups_total} must be divisible by B={B}"
    )
    num_groups = num_groups_total // B

    outs = []
    for b in range(B):
        scale_b = w_scale[b * num_groups : (b + 1) * num_groups]
        zp_b = w_zp[b * num_groups : (b + 1) * num_groups]
        outs.append(matmul_bf16i4_rowwise(X[b], W[b], scale_b, zp_b))
    return torch.stack(outs, dim=0)


# ---------------------------------------------------------------------------
# Register as ROCm implementations of mslk:: torch ops
# ---------------------------------------------------------------------------
# On ROCm, mslk::bf16i4bf16_rowwise and mslk::bf16i4bf16_rowwise_batched are
# schema-only (no C++ CUTLASS impl). Importing this module registers the Triton
# kernels above as the CUDA-dispatch implementations via torch.library.impl,
# making torch.ops.mslk.bf16i4bf16_rowwise(...) call into Triton on AMD GPUs.

if torch.version.hip is not None and hasattr(torch.ops, "mslk"):
    if hasattr(torch.ops.mslk, "bf16i4bf16_rowwise"):

        @torch.library.impl("mslk::bf16i4bf16_rowwise", "CUDA")
        def _bf16i4bf16_rowwise_rocm(
            X: torch.Tensor,
            W: torch.Tensor,
            w_scale_group: torch.Tensor,
            w_zero_group: torch.Tensor,
        ) -> torch.Tensor:
            return matmul_bf16i4_rowwise(X, W, w_scale_group, w_zero_group)

    if hasattr(torch.ops.mslk, "bf16i4bf16_rowwise_batched"):

        @torch.library.impl("mslk::bf16i4bf16_rowwise_batched", "CUDA")
        def _bf16i4bf16_rowwise_batched_rocm(
            X: torch.Tensor,
            W: torch.Tensor,
            w_scale: torch.Tensor,
            w_zp: torch.Tensor,
        ) -> torch.Tensor:
            return matmul_bf16i4_rowwise_batched(X, W, w_scale, w_zp)

    # bf16i4bf16_shuffled: on ROCm the CUTLASS shuffle layout does not exist,
    # so the shuffled ops route through the same rowwise kernels.
    if hasattr(torch.ops.mslk, "bf16i4bf16_shuffled"):

        @torch.library.impl("mslk::bf16i4bf16_shuffled", "CUDA")
        def _bf16i4bf16_shuffled_rocm(
            X: torch.Tensor,
            W: torch.Tensor,
            w_scale_group: torch.Tensor,
            w_zero_group: torch.Tensor,
        ) -> torch.Tensor:
            return matmul_bf16i4_rowwise(X, W, w_scale_group, w_zero_group)

    if hasattr(torch.ops.mslk, "bf16i4bf16_shuffled_batched"):

        @torch.library.impl("mslk::bf16i4bf16_shuffled_batched", "CUDA")
        def _bf16i4bf16_shuffled_batched_rocm(
            X: torch.Tensor,
            W: torch.Tensor,
            w_scale: torch.Tensor,
            w_zp: torch.Tensor,
        ) -> torch.Tensor:
            return matmul_bf16i4_rowwise_batched(X, W, w_scale, w_zp)

    # preshuffle_i4: CUTLASS SM90 pre-processing; identity on ROCm.
    if hasattr(torch.ops.mslk, "preshuffle_i4"):

        @torch.library.impl("mslk::preshuffle_i4", "CUDA")
        def _preshuffle_i4_rocm(
            WQ: torch.Tensor,
            w_scale: torch.Tensor,
        ) -> tuple:
            return WQ, w_scale
