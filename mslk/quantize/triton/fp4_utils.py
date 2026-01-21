# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import math

import torch
from mslk.quantize.triton.fp4_quantize import _from_blocked, FP4_E2M1_MAX, FP8_E4M3_MAX


def fp4_to_float(x: torch.Tensor) -> torch.Tensor:
    """Convert FP4 values packed in uint8 to float32.

    Args:
        x: Tensor of uint8 values, each containing two FP4 values.

    Returns:
        Tensor of float32 values with shape (..., 2*x.shape[-1]).
    """
    # Start by unpacking the FP4 values into separate integers.
    low_mx4 = torch.bitwise_and(x, 0xF)
    high_mx4 = torch.bitwise_and(x >> 4, 0xF)
    comb_shape = x.shape[:-1] + (x.shape[-1] * 2,)
    x_comb = (
        torch.stack([low_mx4, high_mx4], dim=0)
        .view(2, -1)
        .t()
        .contiguous()
        .to(torch.int32)
    )
    # Map to float with a lookup table.
    E2M1_LUT = torch.tensor(
        [0, 0.5, 1, 1.5, 2, 3, 4, 6, -0, -0.5, -1, -1.5, -2, -3, -4, -6],
        dtype=torch.float32,
        device=x.device,
    )
    return torch.index_select(E2M1_LUT, 0, x_comb.view(-1)).view(comb_shape)


def scale_nvfp4(
    x: torch.Tensor,
    scale: torch.Tensor,
    global_scale: torch.Tensor,
    group_size: int = 16,
) -> torch.Tensor:
    """Apply NVFP4 scaling to dequantized values.

    Args:
        x: Dequantized float tensor.
        scale: Per-block scales in fp8 format.
        global_scale: Global scale factor.
        group_size: Number of elements per scale group.

    Returns:
        Scaled tensor.
    """
    # NVFP4 uses a trick where global scales are folded into the scales
    # but not x itself. Those scales are normally removed in the epilogue.
    # Here, we manually get the scaling for x by removing global components.
    true_scale = scale.view(torch.float8_e4m3fn).to(torch.float) / global_scale
    # Now we can reverse scaling of x.
    num_groups = x.shape[-1] // group_size
    scaled_x = (
        x.view(-1, num_groups, group_size)
        * true_scale.view(x.shape[0], -1)[:, :num_groups, None]
    )
    return scaled_x.view(x.shape)


def global_scale_nvfp4(
    x: torch.Tensor,
) -> torch.Tensor:
    """Compute the global scale for NVFP4 quantization.

    Args:
        x: Input tensor in bfloat16 format.

    Returns:
        Global scale factor as a float32 tensor.
    """
    assert x.dtype == torch.bfloat16
    amax = torch.amax(torch.abs(x)).to(torch.float32)
    global_scale = (FP8_E4M3_MAX * FP4_E2M1_MAX) / amax
    return global_scale


def dequantize_nvfp4(
    input_quantized: torch.Tensor,
    scale: torch.Tensor,
    global_scale: torch.Tensor,
    group_size: int = 16,
) -> torch.Tensor:
    """Dequantize NVFP4 quantized tensor back to bfloat16.

    Args:
        input_quantized: Quantized tensor in uint8 format (two FP4 values per byte).
        scale: Per-block scales in blocked format.
        global_scale: Global scale factor.
        group_size: Number of elements per scale group.

    Returns:
        Dequantized tensor in bfloat16 format.
    """
    M = input_quantized.shape[0]
    # Two FP4 values are packed into one uint8.
    N = input_quantized.shape[1] * 2
    # Convert blocked scale format back to (M, num_groups) layout.
    scale = _from_blocked(scale, (M, math.ceil(N / group_size)))
    input_quantized_float = fp4_to_float(input_quantized)
    return scale_nvfp4(input_quantized_float, scale, global_scale, group_size).to(
        torch.bfloat16
    )
