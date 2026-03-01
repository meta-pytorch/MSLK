# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

"""Fused MXFP8 quantization with blocked scale layout.

Provides `triton_quantize_mxfp8`, which quantizes bf16/fp32 tensors to MXFP8
(Float8E4M3FN data + Float8E8M0FNU per-32-element block scales) and writes
the scales directly in the MMA atom-tiled blocked layout expected by
Blackwell blockscale GEMM kernels.
"""

from typing import Tuple

import torch
import triton  # @manual
from triton import language as tl


@triton.jit
def _kernel_quantize_mxfp8_blocked(
    A,
    out,
    scale,
    M,
    K,
    stride_am,
    stride_ak,
    GROUPS_PER_ROW,
    GROUP_SIZE: tl.constexpr,
    SCALE_K: tl.constexpr,
    M_ROUNDED: tl.constexpr,
    BLOCK_GROUPS: tl.constexpr,
) -> None:
    """Quantize a tensor to MXFP8 and write scales in blocked layout.

    Each program processes BLOCK_GROUPS groups of GROUP_SIZE elements,
    computing E8M0 scales and writing them directly in the MMA atom-tiled
    blocked layout. The iteration space covers the full padded scale tensor
    (M_ROUNDED * SCALE_K) so that padding elements are zeroed, allowing
    the scale tensor to be allocated with torch.empty instead of torch.zeros.

    Args:
        A: [M, K] Input tensor (bf16/fp32), may be non-contiguous.
        out: [M, K] Output FP8 tensor (always contiguous).
        scale: [M_ROUNDED * SCALE_K] Output blocked scale tensor (uint8).
        M: Number of rows.
        K: Number of columns.
        stride_am: Stride of A along the row (M) dimension.
        stride_ak: Stride of A along the column (K) dimension.
        GROUPS_PER_ROW: K // GROUP_SIZE, number of scale groups per row.
        GROUP_SIZE: Elements per scale group (32).
        SCALE_K: Rounded GROUPS_PER_ROW (rounded up to multiple of 4).
        M_ROUNDED: M rounded up to multiple of 128.
        BLOCK_GROUPS: Number of groups processed per program.
    """
    FP8_MAX: tl.constexpr = 448.0  # type: ignore[Incompatible variable type]
    E8M0_BIAS: tl.constexpr = 127  # type: ignore[Incompatible variable type]

    pid = tl.program_id(0)
    TOTAL_PADDED_GROUPS: tl.constexpr = M_ROUNDED * SCALE_K

    # Each program handles BLOCK_GROUPS consecutive groups over the full
    # padded space (M_ROUNDED * SCALE_K) so padding is explicitly zeroed.
    group_ids = tl.arange(0, BLOCK_GROUPS) + pid * BLOCK_GROUPS
    group_mask = group_ids < TOTAL_PADDED_GROUPS

    # Map flat group index to (row, col) using SCALE_K (constexpr) as divisor.
    group_rows = group_ids // SCALE_K
    group_cols = group_ids % SCALE_K

    # Mask for groups that correspond to real data (not padding).
    data_mask = (group_rows < M) & (group_cols < GROUPS_PER_ROW)

    # ---- Load input data: [BLOCK_GROUPS, GROUP_SIZE] ----
    elem_range = tl.arange(0, GROUP_SIZE)
    # Input offsets use strides to support non-contiguous tensors (e.g. .t()).
    a_offsets = (
        group_rows[:, None] * stride_am
        + (group_cols[:, None] * GROUP_SIZE + elem_range[None, :]) * stride_ak
    )
    all_mask = data_mask[:, None]

    a = tl.load(A + a_offsets, mask=all_mask, other=0).to(tl.float32)

    # Output offsets are always contiguous (out is freshly allocated).
    out_base = group_rows * K + group_cols * GROUP_SIZE
    out_offsets = out_base[:, None] + elem_range[None, :]

    # ---- E8M0 scale computation (RCEIL semantics via bitwise extraction) ----
    group_max = tl.max(tl.abs(a), axis=1)

    descale = group_max / FP8_MAX
    # Extract IEEE 754 float32 exponent and mantissa via bitwise ops.
    # For positive float32 x = 2^(e-127) * (1 + m/2^23):
    #   ceil(log2(x)) = (e - 127) + (1 if m != 0 else 0)
    # So biased E8M0 exp = e + (1 if m != 0 else 0), clamped to [0, 254].
    descale_bits = descale.to(tl.uint32, bitcast=True)
    float_exp = (descale_bits >> 23) & 0xFF
    has_mantissa = (descale_bits & 0x7FFFFF) != 0
    # For normals: biased_exp = float_exp + has_mantissa.
    # For denormals/zero (float_exp == 0): biased_exp = 0.
    biased_exp = tl.where(
        float_exp > 0,
        (float_exp + tl.where(has_mantissa, 1, 0)).to(tl.uint8),
        tl.zeros_like(float_exp).to(tl.uint8),
    )
    # Clamp to max valid E8M0 (254); 255 is NaN.
    biased_exp = tl.minimum(biased_exp, 254)

    scale_fp = tl.where(
        biased_exp == 0,
        tl.full(biased_exp.shape, 1.0, dtype=tl.float32),
        tl.exp2((E8M0_BIAS - biased_exp.to(tl.float32))),
    )

    # ---- Scale, clamp, and cast to FP8 ----
    scaled = a * scale_fp[:, None]
    scaled = tl.clamp(scaled, -FP8_MAX, FP8_MAX)
    tl.store(out + out_offsets, scaled.to(tl.float8e4nv), mask=all_mask)

    # ---- Write scale to blocked layout ----
    # Decompose (row, col) directly into blocked layout position.
    # The blocked layout is: view(n_row_blocks, 4, 32, n_col_blocks, 4)
    #                        .permute(0, 3, 2, 1, 4).flatten()
    row_superblock = group_rows // 128
    row_sub_group = (group_rows % 128) // 32
    row_within = group_rows % 32
    col_block = group_cols // 4
    col_within = group_cols % 4

    blocked_idx = (
        row_superblock * (SCALE_K * 128)
        + col_block * 512
        + row_within * 16
        + row_sub_group * 4
        + col_within
    )
    # Write real scales for data groups, zero for padding groups.
    scale_val = tl.where(data_mask, biased_exp, tl.zeros_like(biased_exp))
    tl.store(scale + blocked_idx, scale_val, mask=group_mask)


def _select_quantize_config(
    num_groups: int,
) -> Tuple[int, int, int]:
    """Select (BLOCK_GROUPS, num_warps, num_stages) based on problem size.

    Targets sufficient programs to saturate GPU SMs while keeping each
    program's work large enough to amortize overhead.
    """
    if num_groups <= 2048:
        # Small problems (e.g. M=8, K=5120 -> 1280 groups).
        # Use small blocks for more programs and low warps to reduce overhead.
        return 8, 2, 1
    # Default: good balance for medium to very large problems.
    return 128, 4, 2


def triton_quantize_mxfp8(
    data: torch.Tensor,
    block_size: int = 32,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantize to MXFP8 and produce scales directly in blocked layout.

    Fuses to_mxfp8 quantization with scale rearrangement into MMA atom-tiled
    blocked layout, eliminating the separate rearrangement pass.

    Args:
        data: Input tensor (bf16 or fp32), last dim divisible by block_size.
        block_size: Elements per scale group (default 32).

    Returns:
        (scale_blocked, data_fp8):
            scale_blocked: 1D flat tensor of E8M0 scales in blocked layout
                (uint8 viewed as float8_e8m0fnu).
            data_fp8: Quantized data in float8_e4m3fn, same shape as input.
    """
    orig_shape = data.shape
    assert data.ndim >= 1, f"data.ndim needs to be >= 1, but got {data.ndim}."
    # Compute logical 2D shape and strides without copying.
    # For 2D non-contiguous inputs (e.g. .t()), we pass strides directly
    # to the kernel. For higher-dim tensors, we attempt reshape which may
    # copy only if the leading dims aren't contiguous among themselves.
    K = data.shape[-1]
    M = data.numel() // K
    if data.ndim <= 2:
        stride_am = data.stride(-2) if data.ndim == 2 else K
        stride_ak = data.stride(-1)
    else:
        # For ndim >= 3, reshape to 2D. This is a no-op if leading dims are
        # already contiguous, otherwise it copies (same as original behavior).
        data = data.reshape(-1, K)
        stride_am = data.stride(0)
        stride_ak = data.stride(1)
    device = data.device

    assert K % block_size == 0, (
        f"Last dim must be divisible by block_size={block_size}, but got K={K}."
    )
    assert data.dtype in (torch.bfloat16, torch.float32), (
        f"Input dtype must be bf16 or fp32, but got {data.dtype}."
    )

    out = torch.empty((M, K), device=device, dtype=torch.float8_e4m3fn)

    def round_up(x: int, y: int) -> int:
        return (x + y - 1) // y * y

    scale_K = K // block_size
    rounded_M = round_up(M, 128)
    rounded_K = round_up(scale_K, 4)

    scale = torch.empty(rounded_M * rounded_K, device=device, dtype=torch.uint8)

    groups_per_row = scale_K
    num_padded_groups = rounded_M * rounded_K

    block_groups, num_warps, num_stages = _select_quantize_config(num_padded_groups)
    grid = (triton.cdiv(num_padded_groups, block_groups),)

    # pyre-ignore[28]: num_warps and num_stages are valid jit arguments.
    _kernel_quantize_mxfp8_blocked[grid](
        data,
        out,
        scale,
        M=M,
        K=K,
        stride_am=stride_am,
        stride_ak=stride_ak,
        GROUPS_PER_ROW=groups_per_row,
        GROUP_SIZE=block_size,
        SCALE_K=rounded_K,
        M_ROUNDED=rounded_M,
        BLOCK_GROUPS=block_groups,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    return out.view(orig_shape), scale.view(torch.float8_e8m0fnu)
