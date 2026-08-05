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
