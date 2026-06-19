# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

from __future__ import annotations

from enum import IntEnum
from typing import Tuple

import torch
import triton  # @manual
from triton import language as tl  # @manual

FP4_E2M1_MAX = 6.0
FP8_E4M3_MAX = torch.finfo(torch.float8_e4m3fn).max  # 448
# exponent and mantissa bits of `torch.float4_e2m1fn_x2`
FP4_EBITS, FP4_MBITS = 2, 1


class RoundingMode(IntEnum):
    """Rounding options for quantization."""

    nearest = 0
    floor = 1
    even = 2
    stochastic = 3
    ceil = 4


def get_mx4_exp_bias(ebits):
    """Helper function to get the proper exponent bias for specified mx4 format.

    Args:
        ebits: The number of exponent bits in quantized format.

    Returns:
        The exponent bias for the specified mx4 format.
    """
    if ebits == 2:
        return 1
    elif ebits == 3:
        return 3
    else:
        raise NotImplementedError(f"MX4 with ebits={ebits} not supported.")


@triton.jit
def _floor_log2(x):
    """Helper function to efficiently compute floor(log2(x))

    Args:
        x (Tensor): FP32 Input tensor to operate on.

    Returns:
        Tensor: Floor of log2(x).
    """
    # Helpful bit constants.
    FP32_EXP_MASK: tl.constexpr = 0x7F800000  # type: ignore[Incompatible variable type]
    FP32_EXP_OFFSET: tl.constexpr = 23  # type: ignore[Incompatible variable type]
    FP32_EXP_BIAS: tl.constexpr = 127  # type: ignore[Incompatible variable type]

    # View x as an integer and extract its exponent.
    x = x.to(tl.int32, bitcast=True) & FP32_EXP_MASK
    # Shift exponent down to bottom bits.
    x = x >> FP32_EXP_OFFSET
    # Remove FP32 exponent bias and return.
    return (x - FP32_EXP_BIAS).to(tl.float32)


@triton.jit
def _compute_exp(
    group_max,
    rounding_mode,
    rand_bits,
    MBITS: tl.constexpr,
):
    """Compute shared exponent of group using specified rounding mode.

    Args:
        group_max (Tensor): Group of values to compute exponent of.
        rounding_mode (int or RoundingMode): Which rounding mode to use.
        rand_bits (int): Random integer values used for stochastic rounding.
        mbits (int): Number of mantissa bits in target mx4 format.

    Returns:
        Tensor: Shared exponent of group.
    """
    # Define some helpful constants.
    MBITS_FP32: tl.constexpr = 23  # type: ignore[Incompatible variable type]
    M_ROUND: tl.constexpr = (1 << (MBITS_FP32 - MBITS - 1)) - 1  # type: ignore[Incompatible variable type]
    RAND_MASK: tl.constexpr = (1 << (MBITS_FP32 - MBITS)) - 1  # type: ignore[Incompatible variable type]

    # Nearest rounding mode.
    if rounding_mode == 0:
        return tl.floor(tl.log2(group_max) + 0.5)
    # Floor rounding mode. This can be done with fast bit ops.
    if rounding_mode == 1:
        return _floor_log2(group_max)
    # Even pre-rounding mode.
    elif rounding_mode == 2:
        # Add fixed rounding to the mantissa bits of the input to round during truncation.
        group_max = group_max.to(tl.int32, bitcast=True) + M_ROUND
        # Then perform floor rounding of log.
        return _floor_log2(group_max)
    # Stochastic rounding mode.
    elif rounding_mode == 3:
        # Use random bits to add noise to mantissa that would otherwise
        # be rounded away.
        group_max = group_max.to(tl.int32, bitcast=True) + (RAND_MASK & rand_bits)
        # Now compute log and truncate.
        return _floor_log2(group_max)
    else:
        return tl.ceil(tl.log2(group_max))


def _to_blocked(x: torch.Tensor) -> torch.Tensor:
    """Converts a tensor to the blocked layout.
    Args:
        x (torch.Tensor): The input tensor in non-blocked layout.
    Returns:
        torch.Tensor: The output tensor in the blocked layout.
    """

    def ceil_div(a: int, b: int) -> int:
        return (a + b - 1) // b

    rows, cols = x.shape
    n_row_blocks = ceil_div(rows, 128)
    n_col_blocks = ceil_div(cols, 4)

    # Calculate the padded shape
    padded_rows = n_row_blocks * 128
    padded_cols = n_col_blocks * 4

    padded = x
    if (rows, cols) != (padded_rows, padded_cols):
        padded = torch.zeros(
            (padded_rows, padded_cols),
            device=x.device,
            dtype=x.dtype,
        )
        padded[:rows, :cols] = x

    # Rearrange the blocks
    rearranged = (
        padded.view(n_row_blocks, 4, 32, n_col_blocks, 4)
        .permute(0, 3, 2, 1, 4)
        .reshape(-1, 32, 16)
    )

    return rearranged.flatten()


def _from_blocked(x: torch.Tensor, original_shape: Tuple[int, int]) -> torch.Tensor:
    """Converts a tensor from the blocked layout back to standard layout.
    Args:
        x (torch.Tensor): The input tensor in blocked layout (flattened).
        original_shape (Tuple[int, int]): The original shape (rows, cols) before blocking.
    Returns:
        torch.Tensor: The output tensor in the standard layout.
    """

    def ceil_div(a: int, b: int) -> int:
        return (a + b - 1) // b

    original_rows, original_cols = original_shape
    n_row_blocks = ceil_div(original_rows, 128)
    n_col_blocks = ceil_div(original_cols, 4)

    # Calculate the padded shape
    padded_rows = n_row_blocks * 128
    padded_cols = n_col_blocks * 4

    # From flat back to (n_row_blocks, n_col_blocks, 32, 4, 4)
    rearranged = x.view(n_row_blocks, n_col_blocks, 32, 4, 4)

    # Reverse: (n_row_blocks, n_col_blocks, 32, 4, 4) -> (n_row_blocks, 4, 32, n_col_blocks, 4)
    padded = rearranged.permute(0, 3, 2, 1, 4).reshape(padded_rows, padded_cols)

    # Remove padding to get back to original shape
    if (original_rows, original_cols) != (padded_rows, padded_cols):
        return padded[:original_rows, :original_cols].contiguous()
    else:
        return padded.contiguous()


@triton.jit
def _fp32_to_e8m0(
    unscale,
    mbits: tl.constexpr,
    scale_round_mode: tl.constexpr,
):
    E8M0_EXPONENT_BIAS: tl.constexpr = 127  # type: ignore[Incompatible variable type]
    sign = tl.where(unscale < 0, -1.0, 1.0)
    abs_tensor = tl.abs(unscale)

    # MBITS_F32 = 23
    if scale_round_mode == "even":
        val_to_add = (1 << (23 - mbits - 1)) - 1
    elif scale_round_mode == "ceil":
        val_to_add = (1 << 23) - 1
    else:
        val_to_add = 0

    mask_exponent = ((1 << (8 + 1)) - 1) << 23
    mask_mantissa = (1 << 23) - 1

    fp32_bits = tl.extra.cuda.libdevice.float_as_int(abs_tensor)
    fp32_bits_exp = (fp32_bits + val_to_add) & mask_exponent
    exponent = (fp32_bits_exp >> 23) & 0xFF

    if scale_round_mode == "nv_round":
        mantissa = fp32_bits & mask_mantissa
        is_denormal = (exponent == 0) & (mantissa != 0)
        is_normal = ~is_denormal
        condition1 = is_normal & (exponent < 254) & (mantissa > 0)
        condition2 = is_denormal & (mantissa / (2**23) > 0.5)

        exponent = tl.where(condition1 | condition2, exponent + 1, exponent)

    exponent = exponent.to(tl.float32)
    e8m0_values = sign * tl.exp2(exponent - E8M0_EXPONENT_BIAS)

    unscale = e8m0_values
    # In case unscale=0 (scale will be inf), or unscale=inf or nan, we set the scale to 1.0
    unscale_invalid_mask = (
        (e8m0_values == 0)
        | (e8m0_values == float("inf"))
        | (e8m0_values == float("nan"))
    )
    unscale = tl.where(unscale_invalid_mask, 1.0, unscale)

    return unscale


def unsigned_fp32_to_e8m0(
    tensor: torch.Tensor, mbits: tl.constexpr, scale_round_mode: tl.constexpr
) -> torch.Tensor:
    E8M0_EXPONENT_BIAS: tl.constexpr = 127  # type: ignore[Incompatible variable type]

    # MBITS_F32 = 23
    if scale_round_mode == "even":
        val_to_add = (1 << (23 - mbits - 1)) - 1
    elif scale_round_mode == "ceil":
        val_to_add = (1 << 23) - 1
    else:
        val_to_add = 0

    mask_exponent = ((1 << (8 + 1)) - 1) << 23
    mask_mantissa = (1 << 23) - 1

    fp32_bits = tensor.view(torch.int32)
    fp32_bits_exp = (fp32_bits + val_to_add) & mask_exponent
    exponent = (fp32_bits_exp >> 23) & 0xFF

    if scale_round_mode == "nv_round":
        mantissa = fp32_bits & mask_mantissa
        is_denormal = (exponent == 0) & (mantissa != 0)
        is_normal = ~is_denormal
        condition1 = is_normal & (exponent < 254) & (mantissa > 0)
        condition2 = is_denormal & (mantissa / (2**23) > 0.5)

        exponent = torch.where(condition1 | condition2, exponent + 1, exponent)

    exponent = exponent.to(torch.float32)
    e8m0_values = torch.pow(2.0, exponent.float() - E8M0_EXPONENT_BIAS)
    unscale = e8m0_values

    unscale = e8m0_values

    return unscale


def cal_global_scale_mx4_as_nvfp4(x: torch.Tensor):
    """
    To use native nvfp4 to mimic mx4, we need to calculate the global scale in the following way
    global_scale = pow-of-2-floor(448.0 / fp32_to_e8m0(global_amax / 4.0, even_rounding_mode))
                 = 256.0 / fp32_to_e8m0(global_amax / 4.0, even_rounding_mode))
    """
    global_amax = torch.amax(torch.abs(x)).to(torch.float32)
    global_amax_in_mx4_range = unsigned_fp32_to_e8m0(
        global_amax / 4.0,
        # pyre-ignore[6]
        mbits=1,
        # pyre-ignore[6]
        scale_round_mode="even",
    )
    # pyre-ignore[58]
    global_scale = 256.0 / global_amax_in_mx4_range

    return global_scale


@triton.jit
def nvfp4_scale_swizzle(offs_m):
    """
    Produces scale offsets swizzled according to the blackwell 128x4 scale layout.
    Each 128x4 layout can be viewed as 32 4x4 layouts, each of which we'll refer to below as a sub_layout.

    The returned offsets assume a 128x4 layout starting at 0. offs_m could be a subset of rows within a 128x4 layout.
    """
    # Offset of the 4x4 sub_layout within the 128x4 layout
    sub_layout_idx = offs_m % 32
    sub_layout_stride = 16
    sub_layout_off = sub_layout_idx * sub_layout_stride
    # Which row within the 4x4 sub_layout
    sub_layout_row = offs_m // 32
    # Offsets of the elements within 4x4 sub_layout
    elems = tl.arange(0, 4)[None, :]
    sub_layout_elem_offs = sub_layout_row * 4 + elems

    scale_offs = sub_layout_off + sub_layout_elem_offs

    return scale_offs


@triton.jit
def convert_fp32_to_fp4_packed(x_pairs):
    """Convert FP32 pairs to packed FP4 format.

    This function takes tensor where consecutive values along the last dimension
    are packed together into single bytes.

    Args:
        x_pairs: [Tensor, Tensor] both w/ shapes [..., 1] where zipped last dimension contains
                interleaved pairs of FP32 values to be packed together.

    Returns:
        Packed tensor with shape [...] (last dimension removed) where each
        element is an int8 containing 2 FP4 values:
        - First value of pair → low nibble (bits 0-3)
        - Second value of pair → high nibble (bits 4-7)

    Example:
        Input:  [128, 32, 2] containing FP32 pairs
        Output: [128, 32] containing packed FP4 bytes

    """

    x_fp4x2 = tl.inline_asm_elementwise(
        asm="""
        {
        .reg .b8 byte0, byte1, byte2, byte3;
        cvt.rn.satfinite.e2m1x2.f32 byte0, $5, $1;
        cvt.rn.satfinite.e2m1x2.f32 byte1, $6, $2;
        cvt.rn.satfinite.e2m1x2.f32 byte2, $7, $3;
        cvt.rn.satfinite.e2m1x2.f32 byte3, $8, $4;
        mov.b32 $0, {byte0, byte1, byte2, byte3};
        }
        """,
        constraints=("=r,r,r,r,r,r,r,r,r"),
        args=x_pairs,
        dtype=tl.uint8,
        is_pure=True,
        pack=4,
    )

    return x_fp4x2


@triton.jit
def _e2m1_round_to_even(x):
    """Round *non-negative* x to nearest E2M1 value with ties-to-even.

    Returns the rounded value (one of {0, 0.5, 1, 1.5, 2, 3, 4, 6}).
    """
    # Boundary 0: 0.25 — even index → tie rounds DOWN (to 0.0)
    r = tl.where(x < 0.25, 0.0, tl.where(x == 0.25, 0.0, 0.0))
    # Boundary 1: 0.75 — odd index → tie rounds UP (to 1.0)
    r = tl.where(x > 0.25, 0.5, r)
    r = tl.where(x >= 0.75, 1.0, r)
    # Boundary 2: 1.25 — even index → tie rounds DOWN (to 1.0)
    r = tl.where(x > 1.25, 1.5, r)
    # Boundary 3: 1.75 — odd index → tie rounds UP (to 2.0)
    r = tl.where(x >= 1.75, 2.0, r)
    # Boundary 4: 2.5 — even index → tie rounds DOWN (to 2.0)
    r = tl.where(x > 2.5, 3.0, r)
    # Boundary 5: 3.5 — odd index → tie rounds UP (to 4.0)
    r = tl.where(x >= 3.5, 4.0, r)
    # Boundary 6: 5.0 — even index → tie rounds DOWN (to 4.0)
    r = tl.where(x > 5.0, 6.0, r)
    return r
