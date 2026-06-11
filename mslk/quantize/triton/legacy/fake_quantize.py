# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

from __future__ import annotations

from typing import Optional, Tuple

import torch
import triton  # @manual
from mslk.quantize.triton.legacy.primitives import _e2m1_round_to_even
from triton import language as tl  # @manual


@triton.jit
def _fake_quantize_nvfp4_kernel(
    x_ptr,
    y_ptr,
    global_amax_ptr,
    numel,
    BLOCK_SIZE: tl.constexpr,
    BLOCKS_PER_PROGRAM: tl.constexpr,
):
    """Per-block fake quantize matching C++ kernel."""
    pid = tl.program_id(0)
    FP8_E4M3_MAX = 448.0
    FLT_MIN: tl.constexpr = (  # type: ignore[Incompatible variable type]
        1.175494350822287508e-38  # C FLT_MIN
    )

    global_amax = tl.load(global_amax_ptr)

    block_start = pid * BLOCKS_PER_PROGRAM * BLOCK_SIZE
    for block_i in range(BLOCKS_PER_PROGRAM):
        base = block_start + block_i * BLOCK_SIZE
        offsets = base + tl.arange(0, BLOCK_SIZE)
        mask = offsets < numel

        # Load BF16 → FP32
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

        # Block-local amax
        block_amax = tl.max(tl.abs(x))

        # --- compute_scale_with_global (double-precision intermediates) ---
        local_amax_invalid = (block_amax == 0.0) | (
            block_amax != block_amax
        )  # nan check
        global_amax_invalid = (global_amax == 0.0) | (
            global_amax != global_amax
        )  # nan check
        # isinf: fp32 inf is 0x7f800000
        local_amax_invalid = local_amax_invalid | (tl.abs(block_amax) > 3.4e38)
        global_amax_invalid = global_amax_invalid | (tl.abs(global_amax) > 3.4e38)
        any_invalid = local_amax_invalid | global_amax_invalid

        local_unscale = tl.div_rn(block_amax, 6.0)
        ratio_f32 = tl.div_rn(6.0, global_amax)
        two_level_scale = tl.cast(FP8_E4M3_MAX, tl.float64) * ratio_f32.to(tl.float64)
        temp_f64 = local_unscale.to(tl.float64) * two_level_scale
        temp_f32 = temp_f64.to(tl.float32)
        # E4M3 quantize the scale
        temp_e4m3 = temp_f32.to(tl.float8e4nv).to(tl.float32)
        local_unscale_q = temp_e4m3.to(tl.float64) / two_level_scale
        # Final scale and inv_scale
        scale = (tl.cast(1.0, tl.float64) / (local_unscale_q + FLT_MIN)).to(tl.float32)
        inv_scale = local_unscale_q.to(tl.float32)

        # Handle invalid amax
        scale = tl.where(any_invalid, 1.0, scale)
        inv_scale = tl.where(any_invalid, 1.0, inv_scale)

        # --- Per-element quantize ---
        sign = tl.where(x < 0.0, -1.0, 1.0)
        # For x == 0, sign doesn't matter since result will be 0
        x_abs = tl.abs(x)
        x_scaled = x_abs * scale

        # Saturate inf/nan to max
        is_inf_or_nan = (x_abs != x_abs) | (x_abs > 3.4e38)
        x_scaled = tl.where(is_inf_or_nan, 6.0, x_scaled)

        # E2M1 round-to-even
        quantized = _e2m1_round_to_even(x_scaled)

        # Unscale and apply sign
        result = sign * quantized * inv_scale

        # Store as BF16
        tl.store(y_ptr + offsets, result.to(tl.bfloat16), mask=mask)


def triton_fake_quantize_nvfp4_per_tensor(
    input: torch.Tensor,
    static_scales: Optional[torch.Tensor] = None,
    scale_ub: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Simulates NVFP4 quantization noise by quantizing BF16 input to E2M1 (4-bit FP)
    and immediately dequantizing back to BF16.  Bitwise identical to the C++ kernel.

    Args:
        input: BF16 tensor with dim >= 2.
        static_scales: Optional pre-computed global amax (float32 scalar on CUDA).
        scale_ub: Optional upper bound on the global amax.

    Returns:
        (quantized_input, scales)
    """
    assert input.numel() != 0
    assert input.dim() >= 2
    assert input.dtype == torch.bfloat16
    if scale_ub is not None and scale_ub.numel() != 1:
        raise AssertionError(
            f"scale_ub must be a scalar (numel == 1); got shape {tuple(scale_ub.shape)}"
        )

    output = torch.empty_like(input)

    if static_scales is not None:
        scales = static_scales
    else:
        scales = torch.empty(1, dtype=torch.float32, device=input.device)
        amax = input.float().abs().max()

        if scale_ub is not None:
            amax = torch.min(amax, scale_ub.squeeze())
        scales.fill_(amax.item())

    # Launch Triton kernel
    numel = input.numel()
    BLOCK_SIZE = 16
    BLOCKS_PER_PROGRAM = 4
    elements_per_program = BLOCK_SIZE * BLOCKS_PER_PROGRAM
    grid = (triton.cdiv(numel, elements_per_program),)

    _fake_quantize_nvfp4_kernel[grid](
        input.reshape(-1),
        output.reshape(-1),
        scales,
        numel,
        # pyre-ignore[6]
        BLOCK_SIZE=BLOCK_SIZE,
        # pyre-ignore[6]
        BLOCKS_PER_PROGRAM=BLOCKS_PER_PROGRAM,
    )

    return output, scales
