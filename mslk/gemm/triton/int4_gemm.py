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
"""

from typing import List

import torch
import triton  # @manual
import triton.language as tl  # @manual
from triton import Config  # @manual


# ---------------------------------------------------------------------------
# Autotuning configs
# ---------------------------------------------------------------------------

def _get_configs() -> List[Config]:
    configs = []
    for bm in [16, 32, 64, 128]:
        for bn in [64, 128, 256]:
            for bk in [64, 128]:
                for nw in [4, 8]:
                    for ns in [2, 3]:
                        configs.append(
                            Config(
                                {"BLOCK_M": bm, "BLOCK_N": bn, "BLOCK_K": bk},
                                num_warps=nw,
                                num_stages=ns,
                            )
                        )
    return configs


# ---------------------------------------------------------------------------
# Core Triton kernel
# ---------------------------------------------------------------------------

@triton.autotune(configs=_get_configs(), key=["M", "N", "K", "group_size"])
@triton.jit
def _bf16i4_rowwise_kernel(
    X_ptr,
    W_ptr,
    Y_ptr,
    scale_ptr,
    zero_ptr,
    M,
    N,
    K,
    group_size,
    stride_xm,
    stride_xk,
    stride_wn,
    stride_wk,  # stride over packed bytes, not elements
    stride_ym,
    stride_yn,
    stride_sg,  # stride over groups (= N for contiguous [num_groups, N])
    stride_sn,  # stride over N columns (= 1 for contiguous)
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
) -> None:
    """
    Computes Y[M, N] = X[M, K] @ dequant(W)[K, N]  (i.e. X @ W.T after dequant).

    Each program instance handles a [BLOCK_M, BLOCK_N] output tile.
    The K loop iterates over BLOCK_K activation elements at a time (= BLOCK_K//2 packed bytes).

    Constraint: BLOCK_K must divide group_size (both are powers-of-2 in practice).
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    for k_idx in tl.range(0, tl.cdiv(K, BLOCK_K)):
        k_start = k_idx * BLOCK_K

        # ---- activation tile [BLOCK_M, BLOCK_K] ----
        offs_k = k_start + tl.arange(0, BLOCK_K)
        x_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        x = tl.load(
            X_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk,
            mask=x_mask,
            other=0.0,
        ).to(tl.bfloat16)

        # ---- packed weight tile [BLOCK_N, BLOCK_K//2] ----
        offs_k2 = k_start // 2 + tl.arange(0, BLOCK_K // 2)
        w_mask = (offs_n[:, None] < N) & (offs_k2[None, :] < K // 2)
        w_q = tl.load(
            W_ptr + offs_n[:, None] * stride_wn + offs_k2[None, :] * stride_wk,
            mask=w_mask,
            other=0,
        ).to(tl.int32)

        # ---- unpack nibbles ----
        # lo nibble (bits 0-3) -> even K positions (W[n, 2i])
        # hi nibble (bits 4-7) -> odd K positions  (W[n, 2i+1])
        # Values are 4-bit two's-complement stored in pack_int4 convention:
        #   signed int4 in [-8, 7] is stored as its 4 LSBs (unsigned nibble [0,15]).
        #   Positive values [0,7] map directly; negative [-8,-1] map to nibbles [8,15].
        w_lo = w_q & 0x0F            # [BLOCK_N, BLOCK_K//2], unsigned nibble [0,15]
        w_hi = (w_q >> 4) & 0x0F    # [BLOCK_N, BLOCK_K//2], unsigned nibble [0,15]

        # ---- per-group scale and zero ----
        group_idx = k_start // group_size
        s = tl.load(
            scale_ptr + group_idx * stride_sg + offs_n * stride_sn,
            mask=offs_n < N,
        ).to(tl.float32)  # [BLOCK_N]
        z = tl.load(
            zero_ptr + group_idx * stride_sg + offs_n * stride_sn,
            mask=offs_n < N,
        ).to(tl.float32)  # [BLOCK_N]

        # ---- dequantise: w_float = w_signed * scale + zero ----
        # Recover signed int4 from unsigned nibble via two's-complement:
        #   signed = (nibble XOR 8) - 8
        #   nibble 0..7  -> signed  0..7  (positive range, bit3 clear)
        #   nibble 8..15 -> signed -8..-1 (negative range, bit3 set)
        # Then: w_float = signed * scale + zero
        #              = ((nibble ^ 8) - 8) * scale + zero
        s_col = s[:, None]  # [BLOCK_N, 1]
        z_col = z[:, None]  # [BLOCK_N, 1]
        w_lo_s = (w_lo ^ 8) - 8  # signed int4, [BLOCK_N, BLOCK_K//2]
        w_hi_s = (w_hi ^ 8) - 8
        w_lo_dq = w_lo_s.to(tl.float32) * s_col + z_col  # [BLOCK_N, BLOCK_K//2]
        w_hi_dq = w_hi_s.to(tl.float32) * s_col + z_col  # [BLOCK_N, BLOCK_K//2]

        # ---- interleave lo/hi into [BLOCK_N, BLOCK_K] ----
        # w_full[n, 2i]   = lo_dq[n, i]  (even K)
        # w_full[n, 2i+1] = hi_dq[n, i]  (odd  K)
        #
        # Reshape each to [BLOCK_N, BLOCK_K//2, 1] then broadcast-select
        # along a new trailing axis of size 2, then reshape to [BLOCK_N, BLOCK_K].
        w_lo_3d = tl.reshape(w_lo_dq, [BLOCK_N, BLOCK_K // 2, 1])
        w_hi_3d = tl.reshape(w_hi_dq, [BLOCK_N, BLOCK_K // 2, 1])
        # selector[0] == True -> take lo, selector[1] == True -> take hi
        selector = tl.arange(0, 2)[None, None, :] == 0  # [1, 1, 2]
        w_interleaved = tl.where(selector, w_lo_3d, w_hi_3d)  # [BLOCK_N, BLOCK_K//2, 2]
        w_full = tl.reshape(w_interleaved, [BLOCK_N, BLOCK_K]).to(tl.bfloat16)

        # ---- GEMM accumulate: [BLOCK_M, BLOCK_K] @ [BLOCK_K, BLOCK_N] ----
        acc = tl.dot(x, tl.trans(w_full), acc, out_dtype=tl.float32)

    # ---- store output ----
    y_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(
        Y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn,
        acc.to(tl.bfloat16),
        mask=y_mask,
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

    assert W.shape == (N, K // 2), f"W must be [N, K//2], got {W.shape}"
    assert w_scale_group.shape == (num_groups, N), (
        f"w_scale_group must be [num_groups, N]={num_groups, N}, got {w_scale_group.shape}"
    )
    assert w_zero_group.shape == (num_groups, N), (
        f"w_zero_group must be [num_groups, N]={num_groups, N}, got {w_zero_group.shape}"
    )
    group_size = K // num_groups
    assert group_size % 64 == 0, (
        f"group_size={group_size} must be divisible by 64 (BLOCK_K min); "
        "common values are 64, 128, 256."
    )

    X_2d = X.reshape(M, K).contiguous()
    Y = torch.empty((M, N), dtype=torch.bfloat16, device=X.device)

    grid = lambda meta: (  # noqa: E731
        triton.cdiv(M, meta["BLOCK_M"]),
        triton.cdiv(N, meta["BLOCK_N"]),
    )
    _bf16i4_rowwise_kernel[grid](
        X_2d, W, Y,
        w_scale_group, w_zero_group,
        M, N, K,
        group_size,
        X_2d.stride(0), X_2d.stride(1),
        W.stride(0), W.stride(1),
        Y.stride(0), Y.stride(1),
        w_scale_group.stride(0), w_scale_group.stride(1),
    )
    return Y.reshape(*leading, N)


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
