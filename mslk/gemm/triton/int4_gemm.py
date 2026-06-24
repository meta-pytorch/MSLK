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
  launch using optimised PyTorch/HIP strided copies.  The kernel then issues
  two tl.dot calls — one against the lo nibbles, one against the hi nibbles —
  with fully coalesced loads and no LDS involvement.
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
            # BLOCK_K is the tile size over the K2 = K//2 dimension, which is
            # also the inner-dot K for each tl.dot call.  Must divide
            # group_size // 2 (so 2*BLOCK_K divides group_size).  With the
            # common group_size=128, BLOCK_K <= 64.
            for bk in [32, 64]:
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


@triton.autotune(configs=_get_configs(), key=["M", "N", "K2", "group_size"])
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
    stride_ym,
    stride_yn,
    stride_sg,
    stride_sn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,  # tile size over K2; inner-dot K for each tl.dot
) -> None:
    """
    Computes Y[M, N] = X[M, K] @ dequant(W)[K, N].

    X has been pre-split into x_even [M, K//2] (even K columns) and
    x_odd [M, K//2] (odd K columns) by the Python wrapper.  Each kernel
    iteration loads BLOCK_K elements from each and issues two tl.dot calls
    against the unpacked lo/hi weight halves — no interleave, no LDS.

    Constraint: 2 * BLOCK_K must divide group_size.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    for k2_idx in tl.range(0, tl.cdiv(K2, BLOCK_K)):
        k2_start = k2_idx * BLOCK_K
        offs_k2 = k2_start + tl.arange(0, BLOCK_K)

        # ---- activation tiles [BLOCK_M, BLOCK_K] — fully coalesced ----
        xk_mask = (offs_m[:, None] < M) & (offs_k2[None, :] < K2)
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

        # ---- packed weight tile [BLOCK_N, BLOCK_K] ----
        wk_mask = (offs_n[:, None] < N) & (offs_k2[None, :] < K2)
        w_q = tl.load(
            W_ptr + offs_n[:, None] * stride_wn + offs_k2[None, :] * stride_wk,
            mask=wk_mask,
            other=0,
        ).to(tl.int32)

        # ---- unpack nibbles ----
        # lo nibble (bits 0-3) -> even K weight values
        # hi nibble (bits 4-7) -> odd  K weight values
        w_lo = w_q & 0x0F  # [BLOCK_N, BLOCK_K]
        w_hi = (w_q >> 4) & 0x0F  # [BLOCK_N, BLOCK_K]

        # ---- per-group scale and zero ----
        # k2_start indexes K//2; the corresponding full-K position is 2*k2_start.
        group_idx = (k2_start * 2) // group_size
        s = tl.load(
            scale_ptr + group_idx * stride_sg + offs_n * stride_sn,
            mask=offs_n < N,
        ).to(tl.float32)
        z = tl.load(
            zero_ptr + group_idx * stride_sg + offs_n * stride_sn,
            mask=offs_n < N,
        ).to(tl.float32)
        s_col = s[:, None]
        z_col = z[:, None]

        # ---- dequantise: signed = (nibble ^ 8) - 8, w_float = signed * scale + zero ----
        w_lo_dq = ((w_lo ^ 8) - 8).to(tl.float32) * s_col + z_col  # [BLOCK_N, BLOCK_K]
        w_hi_dq = ((w_hi ^ 8) - 8).to(tl.float32) * s_col + z_col  # [BLOCK_N, BLOCK_K]

        # ---- two dot products, no interleave, no LDS ----
        # x_even [BLOCK_M, BLOCK_K] @ w_lo_dq.T [BLOCK_K, BLOCK_N]  (even K)
        # x_odd  [BLOCK_M, BLOCK_K] @ w_hi_dq.T [BLOCK_K, BLOCK_N]  (odd  K)
        acc = tl.dot(
            x_even, tl.trans(w_lo_dq.to(tl.bfloat16)), acc, out_dtype=tl.float32
        )
        acc = tl.dot(
            x_odd, tl.trans(w_hi_dq.to(tl.bfloat16)), acc, out_dtype=tl.float32
        )

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
    K2 = K // 2

    assert W.shape == (N, K2), f"W must be [N, K//2], got {W.shape}"
    assert w_scale_group.shape == (
        num_groups,
        N,
    ), f"w_scale_group must be [num_groups, N]={num_groups, N}, got {w_scale_group.shape}"
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

    Y = torch.empty((M, N), dtype=torch.bfloat16, device=X.device)

    grid = lambda meta: (  # noqa: E731
        triton.cdiv(M, meta["BLOCK_M"]),
        triton.cdiv(N, meta["BLOCK_N"]),
    )
    _bf16i4_rowwise_kernel[grid](
        x_even,
        x_odd,
        W,
        Y,
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
        Y.stride(0),
        Y.stride(1),
        w_scale_group.stride(0),
        w_scale_group.stride(1),
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
    assert (
        num_groups_total % B == 0
    ), f"w_scale.shape[0]={num_groups_total} must be divisible by B={B}"
    num_groups = num_groups_total // B

    outs = []
    for b in range(B):
        scale_b = w_scale[b * num_groups : (b + 1) * num_groups]
        zp_b = w_zp[b * num_groups : (b + 1) * num_groups]
        outs.append(matmul_bf16i4_rowwise(X[b], W[b], scale_b, zp_b))
    return torch.stack(outs, dim=0)


# ---------------------------------------------------------------------------
# BF16 preshuffle / unshuffle helpers
# ---------------------------------------------------------------------------
#
# CUTLASS preshuffle_i4 (BF16 variant) reorders INT4 values within each
# group of 8 along K using the permutation [0,2,4,6,1,3,5,7].  The CUDA
# implementation uses cutlass::reorder_tensor; here we replicate it with
# vectorised bitwise ops so it works on ROCm without CUTLASS headers.
#
# Input layout (rowwise): each byte packs (lo=even_K, hi=odd_K).
#   Bytes [b0, b1, b2, b3] hold nibbles [n0,n1,n2,n3,n4,n5,n6,n7].
# After preshuffle [0,2,4,6,1,3,5,7]:
#   out_byte0 = lo(n0) | hi(n2)  — lo nibbles of b0 and b1
#   out_byte1 = lo(n4) | hi(n6)  — lo nibbles of b2 and b3
#   out_byte2 = lo(n1) | hi(n3)  — hi nibbles of b0 and b1
#   out_byte3 = lo(n5) | hi(n7)  — hi nibbles of b2 and b3
# This separates even-K (lo) nibbles into bytes 0-1 and odd-K (hi)
# nibbles into bytes 2-3, which is a pure bitwise rearrangement.


def _bf16_preshuffle_i4(WQ: torch.Tensor) -> torch.Tensor:
    N, K2 = WQ.shape
    assert (K2 * 2) % 8 == 0, f"K={K2 * 2} must be divisible by 8 for preshuffle"
    b = WQ.view(torch.uint8).view(N, K2 // 4, 4).to(torch.int32)
    lo = b & 0x0F
    hi = (b >> 4) & 0x0F
    out = torch.stack(
        [
            lo[:, :, 0] | (lo[:, :, 1] << 4),
            lo[:, :, 2] | (lo[:, :, 3] << 4),
            hi[:, :, 0] | (hi[:, :, 1] << 4),
            hi[:, :, 2] | (hi[:, :, 3] << 4),
        ],
        dim=-1,
    )
    return out.to(torch.uint8).view(torch.int8).view(N, K2).contiguous()


@triton.jit
def _unshuffle_i4_kernel(
    src_ptr,
    dst_ptr,
    num_elements,
    BLOCK: tl.constexpr,
) -> None:
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < num_elements

    # Load 4 bytes as int32 — one group of 8 INT4 values in shuffled layout
    val = tl.load(src_ptr + offs, mask=mask).to(tl.int32)

    # Shuffled layout: lo nibbles in bytes 0-1, hi nibbles in bytes 2-3
    # Extract each byte
    b0 = val & 0xFF
    b1 = (val >> 8) & 0xFF
    b2 = (val >> 16) & 0xFF
    b3 = (val >> 24) & 0xFF

    # Reconstruct original rowwise bytes: byte_i = lo_nibble_i | (hi_nibble_i << 4)
    out0 = (b0 & 0x0F) | ((b2 & 0x0F) << 4)
    out1 = ((b0 >> 4) & 0x0F) | ((b2 >> 4) << 4)
    out2 = (b1 & 0x0F) | ((b3 & 0x0F) << 4)
    out3 = ((b1 >> 4) & 0x0F) | ((b3 >> 4) << 4)

    result = out0 | (out1 << 8) | (out2 << 16) | (out3 << 24)
    tl.store(dst_ptr + offs, result.to(tl.int32), mask=mask)


def _bf16_unshuffle_i4(WQ_shuffled: torch.Tensor) -> torch.Tensor:
    N, K2 = WQ_shuffled.shape
    assert (K2 * 2) % 8 == 0, f"K={K2 * 2} must be divisible by 8 for unshuffle"
    src = WQ_shuffled.view(torch.int32).contiguous()
    dst = torch.empty_like(src)
    num_elements = src.numel()
    BLOCK = 1024
    grid = (triton.cdiv(num_elements, BLOCK),)
    _unshuffle_i4_kernel[grid](src, dst, num_elements, BLOCK)
    return dst.view(torch.int8).view(N, K2)


def preshuffle_i4_bf16(
    WQ: torch.Tensor,
    w_scale: torch.Tensor,
) -> "tuple[torch.Tensor, torch.Tensor]":
    WQ_shuffled = _bf16_preshuffle_i4(WQ)
    return WQ_shuffled, w_scale


# ---------------------------------------------------------------------------
# Shuffled wrappers — unshuffle then delegate to rowwise kernel
# ---------------------------------------------------------------------------


def matmul_bf16i4_shuffled(
    X: torch.Tensor,
    W_shuffled: torch.Tensor,
    w_scale_group: torch.Tensor,
    w_zero_group: torch.Tensor,
) -> torch.Tensor:
    W = _bf16_unshuffle_i4(W_shuffled)
    return matmul_bf16i4_rowwise(X, W, w_scale_group, w_zero_group)


def matmul_bf16i4_shuffled_grouped(
    X: torch.Tensor,
    WQ: torch.Tensor,
    w_scale_group: torch.Tensor,
    w_zero_group: torch.Tensor,
    M_sizes: torch.Tensor,
) -> torch.Tensor:
    G = M_sizes.shape[0]
    N = WQ.shape[1]
    K2 = WQ.shape[2]
    M = X.shape[0]

    W_unshuffled = torch.stack([_bf16_unshuffle_i4(WQ[g]) for g in range(G)])

    m_list = M_sizes.cpu().tolist()
    x_groups = torch.split(X, m_list, dim=0)

    outs = []
    for g in range(G):
        if m_list[g] == 0:
            outs.append(X.new_empty((0, N)))
            continue
        num_groups = w_scale_group.shape[1]
        outs.append(
            matmul_bf16i4_rowwise(
                x_groups[g],
                W_unshuffled[g],
                w_scale_group[g],
                w_zero_group[g],
            )
        )
    return torch.cat(outs, dim=0)


def matmul_bf16i4_shuffled_batched(
    X: torch.Tensor,
    WQ: torch.Tensor,
    w_scale: torch.Tensor,
    w_zp: torch.Tensor,
) -> torch.Tensor:
    B = X.shape[0]
    W_unshuffled = torch.stack([_bf16_unshuffle_i4(WQ[b]) for b in range(B)])
    return matmul_bf16i4_rowwise_batched(X, W_unshuffled, w_scale, w_zp)


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

    if hasattr(torch.ops.mslk, "bf16i4bf16_shuffled"):

        @torch.library.impl("mslk::bf16i4bf16_shuffled", "CUDA")
        def _bf16i4bf16_shuffled_rocm(
            X: torch.Tensor,
            W: torch.Tensor,
            w_scale_group: torch.Tensor,
            w_zero_group: torch.Tensor,
        ) -> torch.Tensor:
            return matmul_bf16i4_shuffled(X, W, w_scale_group, w_zero_group)

    if hasattr(torch.ops.mslk, "bf16i4bf16_shuffled_grouped"):

        @torch.library.impl("mslk::bf16i4bf16_shuffled_grouped", "CUDA")
        def _bf16i4bf16_shuffled_grouped_rocm(
            X: torch.Tensor,
            WQ: torch.Tensor,
            w_scale_group: torch.Tensor,
            w_zero_group: torch.Tensor,
            M_sizes: torch.Tensor,
        ) -> torch.Tensor:
            return matmul_bf16i4_shuffled_grouped(
                X, WQ, w_scale_group, w_zero_group, M_sizes
            )

    if hasattr(torch.ops.mslk, "bf16i4bf16_shuffled_batched"):

        @torch.library.impl("mslk::bf16i4bf16_shuffled_batched", "CUDA")
        def _bf16i4bf16_shuffled_batched_rocm(
            X: torch.Tensor,
            WQ: torch.Tensor,
            w_scale: torch.Tensor,
            w_zp: torch.Tensor,
        ) -> torch.Tensor:
            return matmul_bf16i4_shuffled_batched(X, WQ, w_scale, w_zp)

    if hasattr(torch.ops.mslk, "preshuffle_i4"):

        @torch.library.impl("mslk::preshuffle_i4", "CUDA")
        def _preshuffle_i4_rocm(
            WQ: torch.Tensor,
            w_scale: torch.Tensor,
        ) -> "tuple[torch.Tensor, torch.Tensor]":
            if w_scale.dtype == torch.bfloat16:
                return preshuffle_i4_bf16(WQ, w_scale)
            raise NotImplementedError(
                f"ROCm preshuffle_i4 only supports BF16 scales, got {w_scale.dtype}"
            )
