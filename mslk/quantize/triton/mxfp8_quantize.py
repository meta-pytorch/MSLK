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

import math
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
    GROUPS_PER_ROW,
    GROUP_SIZE: tl.constexpr,
    SCALE_K: tl.constexpr,
    BLOCK_GROUPS: tl.constexpr,
) -> None:
    """Quantize a tensor to MXFP8 and write scales in blocked layout.

    Each program processes BLOCK_GROUPS groups of GROUP_SIZE elements,
    computing E8M0 scales and writing them directly in the MMA atom-tiled
    blocked layout.

    Args:
        A: [M, K] Input tensor (bf16/fp32), flattened to 1D.
        out: [M, K] Output FP8 tensor, same shape as A.
        scale: [rounded_M * SCALE_K] Output blocked scale tensor (uint8).
        M: Number of rows.
        K: Number of columns.
        GROUPS_PER_ROW: K // GROUP_SIZE, number of scale groups per row.
        GROUP_SIZE: Elements per scale group (32).
        SCALE_K: Rounded GROUPS_PER_ROW (rounded up to multiple of 4).
        BLOCK_GROUPS: Number of groups processed per program.
    """
    FP8_MAX: tl.constexpr = 448.0  # type: ignore[Incompatible variable type]
    E8M0_BIAS: tl.constexpr = 127  # type: ignore[Incompatible variable type]

    pid = tl.program_id(0)
    NUM_GROUPS = M * GROUPS_PER_ROW

    # Each program handles BLOCK_GROUPS consecutive groups.
    group_ids = tl.arange(0, BLOCK_GROUPS) + pid * BLOCK_GROUPS
    group_mask = group_ids < NUM_GROUPS

    # Map flat group index to (row, col) in the group grid.
    group_rows = group_ids // GROUPS_PER_ROW
    group_cols = group_ids % GROUPS_PER_ROW

    # ---- Load input data: [BLOCK_GROUPS, GROUP_SIZE] ----
    base_offsets = group_rows * K + group_cols * GROUP_SIZE
    elem_range = tl.arange(0, GROUP_SIZE)
    all_offsets = base_offsets[:, None] + elem_range[None, :]
    all_mask = group_mask[:, None]

    a = tl.load(A + all_offsets, mask=all_mask, other=0).to(tl.float32)

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
    tl.store(out + all_offsets, scaled.to(tl.float8e4nv), mask=all_mask)

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
    tl.store(scale + blocked_idx, biased_exp, mask=group_mask)


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
    other_dims = 1 if data.ndim == 1 else -1
    data = data.reshape(other_dims, data.shape[-1])
    M, K = data.shape
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

    scale = torch.zeros(rounded_M * rounded_K, device=device, dtype=torch.uint8)

    groups_per_row = scale_K
    num_groups = M * groups_per_row

    block_groups, num_warps, num_stages = _select_quantize_config(num_groups)
    grid = (math.ceil(num_groups / block_groups),)

    # pyre-ignore[28]: num_warps and num_stages are valid jit arguments.
    _kernel_quantize_mxfp8_blocked[grid](
        data,
        out,
        scale,
        M=M,
        K=K,
        GROUPS_PER_ROW=groups_per_row,
        GROUP_SIZE=block_size,
        SCALE_K=rounded_K,
        BLOCK_GROUPS=block_groups,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    return scale.view(torch.float8_e8m0fnu), out.view(orig_shape)
