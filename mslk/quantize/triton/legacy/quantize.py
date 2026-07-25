# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

from __future__ import annotations

from typing import Tuple

import torch
import triton  # @manual
from mslk.quantize.triton.legacy.primitives import (
    _fp32_to_e8m0,
    convert_fp32_to_fp4_packed,
    nvfp4_scale_swizzle,
)
from triton import language as tl  # @manual


def triton_quantize_nvfp4(
    x: torch.Tensor,
    global_scale: torch.Tensor | None,
    use_e8m0_scale: bool = False,
    use_precise_math: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantize a tensor to NVFP4 format.

    Args:
        x (torch.Tensor): Input tensor to be quantized.
        global_scale (torch.Tensor | None): Per-tensor scale for two-level quantization.
            If None, the global scale is not applied (treated as 1.0).
        use_e8m0_scale (bool): Whether to use E8M0 for quantization. If True will mimic mx4's e8m0 scaling factor in nvfp4's fp8 local scale.
        use_precise_math (bool): Whether to use precise math for quantization.
            If disabled the kernel would multiply by the reciprocal instead of dividing when computing scales.
            In practice this is **often** bitwise accurate as the scales are converted to FP8 right after.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Quantized tensor and scales tensor in swizzled layout.
    """
    # reshape to 2d
    orig_leading_dims, orig_N = x.shape[:-2], x.shape[-1]
    x = x.reshape(-1, orig_N)

    M, N = x.shape
    assert N % 16 == 0, "N must be divisible by 16 for NVFP4 quantization"

    # Calculate blocks needed
    num_scales = N // 16
    n_row_blocks = triton.cdiv(M, 128)
    n_col_blocks = triton.cdiv(num_scales, 4)
    padded_rows = n_row_blocks * 128
    padded_cols = n_col_blocks * 4

    xq = x.new_empty(M, N // 2, dtype=torch.uint8)
    scales = x.new_empty(padded_rows, padded_cols, dtype=torch.float8_e4m3fn)

    # For small M use lower M_PER_BLOCK to reduce wasted FP32 math
    M_PER_BLOCK = min(triton.next_power_of_2(M), 128)
    # We don't support multiple 128x4 layouts per block
    assert M_PER_BLOCK <= 128

    # If we are not aligned to M_PER_BLOCK * 64, use a mask
    USE_MASK = M % M_PER_BLOCK != 0 or N % 64 != 0

    grid = (triton.cdiv(N, 64), triton.cdiv(M, M_PER_BLOCK))
    # If M_PER_BLOCK is not 128 launch an extra set of blocks along M to handle zeroing scales.
    # This is needed as otherwise the kernel would not visit those scales, and the spec requires padded scales to be zero.
    if M_PER_BLOCK != 128:
        grid = (grid[0], grid[1] + 1)

    use_global_scale = global_scale is not None
    if not use_global_scale:
        # Pass a dummy pointer; the kernel won't load from it.
        global_scale = x.new_empty(())

    # Use int64 indexing when pointer offsets can exceed INT32_MAX
    use_int64_indexing = M * N > 2**31 - 1

    triton_quantize_nvfp4_kernel[grid](
        x,
        global_scale,
        xq,
        scales,
        x.stride(0),
        x.stride(1),
        M,
        N,
        # pyre-ignore[6]
        M_PER_BLOCK=M_PER_BLOCK,
        # pyre-ignore[6]
        USE_MASK=USE_MASK,
        # pyre-ignore[6]
        USE_E8M0_SCALE=use_e8m0_scale,
        # pyre-ignore[6]
        USE_PRECISE_MATH=use_precise_math,
        # pyre-ignore[6]
        USE_GLOBAL_SCALE=use_global_scale,
        # pyre-ignore[6]
        USE_INT64_INDEXING=use_int64_indexing,
    )

    # reshape back to original shape
    scales = scales.view(*orig_leading_dims, -1, padded_cols)
    xq = xq.view(*orig_leading_dims, -1, N // 2)

    return xq.view(torch.float4_e2m1fn_x2), scales


@triton.jit
def triton_quantize_nvfp4_kernel(
    x_ptr,
    global_scale_ptr,
    q_ptr,
    s_ptr,
    stride_xm,
    stride_xn,
    M,
    N,
    M_PER_BLOCK: tl.constexpr,
    USE_MASK: tl.constexpr,
    USE_E8M0_SCALE: tl.constexpr,
    USE_PRECISE_MATH: tl.constexpr,
    USE_GLOBAL_SCALE: tl.constexpr,
    USE_INT64_INDEXING: tl.constexpr,
):
    E4M3_EPS = 1.5258789e-05
    FP8_E4M3_MAX = 448.0
    FP4_E2M1_MAX = 6.0
    INV_FP4_E2M1_MAX = 1.0 / 6.0

    NUM_ELEM_PER_LAYOUT = 128 * 4
    NUM_N_BLOCKS = tl.cdiv(N, 64)

    pid_m = tl.program_id(1)
    pid_n = tl.program_id(0)

    # Special blocks that zeros out tail M scales if M_PER_BLOCK != 128
    # Technically this is a data race as we zero out scales another block has also zero'd out.
    # Since we write the same value, it shouldn't be an issue.
    if M_PER_BLOCK != 128 and pid_m * M_PER_BLOCK >= M:
        # This is only used (and supported) when M < 128.
        tl.device_assert(pid_m == 1, "pid_m != 1 when M_PER_BLOCK != 128")

        # Offset of the 128x4 layout
        layout_off = pid_n * NUM_ELEM_PER_LAYOUT
        offs_m = tl.arange(0, 128)[:, None]
        scale_offs = layout_off + nvfp4_scale_swizzle(offs_m)

        oob_mask = (offs_m >= M) & tl.full((4,), True, dtype=tl.int1)[None, :]
        zero_scales = tl.full([128, 4], 0, dtype=tl.float8e4nv)
        tl.store(s_ptr + scale_offs, zero_scales, mask=oob_mask)
        return

    offs_m = pid_m * M_PER_BLOCK + tl.arange(0, M_PER_BLOCK)[:, None]
    offs_n = pid_n * 64 + tl.arange(0, 64)[None, :]
    if USE_INT64_INDEXING:
        offs_m = offs_m.to(tl.int64)
        offs_n = offs_n.to(tl.int64)

    if USE_MASK:
        mask = (offs_m < M) & (offs_n < N)
        other = 0.0
    else:
        mask = None
        other = None

    if USE_GLOBAL_SCALE:
        global_scale = tl.load(global_scale_ptr)  # Scalar
    else:
        global_scale = 1.0

    load_offsets = offs_m * stride_xm + offs_n * stride_xn
    x = tl.load(x_ptr + load_offsets, mask=mask, other=other)  # [M_PER_BLOCK, 64]
    x_blocks = x.to(tl.float32).reshape(M_PER_BLOCK, 4, 16)  # [M_PER_BLOCK, 4, 16]

    # Block-wise max
    block_amax = tl.max(tl.abs(x_blocks), axis=2)  # [M_PER_BLOCK, 4]

    # To avoid expensive per-element tl.div_rn we can multiply by the reciprocal.
    # This could introduce ~1ULP differnce. However as the scales are casted
    # from FP32 to FP4 right after, for most FP32 values this is equivalent still.
    # We gate this to USE_PRECISE_MATH=False.
    if USE_E8M0_SCALE:
        if USE_PRECISE_MATH:
            scales = tl.div_rn(block_amax, 4.0) * global_scale
        else:
            scales = block_amax * 0.25 * global_scale
        scales = _fp32_to_e8m0(scales, mbits=1, scale_round_mode="even")
    else:
        if USE_PRECISE_MATH:
            scales = tl.div_rn(block_amax, FP4_E2M1_MAX) * global_scale
        else:
            scales = block_amax * INV_FP4_E2M1_MAX * global_scale
        scales = tl.clamp(scales, E4M3_EPS, FP8_E4M3_MAX)

    scales = scales.to(tl.float8e4nv)  # [M_PER_BLOCK, 4]

    # Apply combined scale to data
    total_scale = tl.div_rn(
        global_scale, scales.to(tl.float32)[:, :, None]
    )  # [M_PER_BLOCK, 4, 1]
    x_blocks = x_blocks * total_scale  # [M_PER_BLOCK, 4, 16]

    if USE_MASK:
        scale_offs_n = pid_n * 4 + tl.arange(0, 4)[None, :]
        scale_mask = (offs_m < M) & (scale_offs_n < (N // 16))

        # Mask out scales to 0 if we are not aligned to M_PER_BLOCK x 64
        scales = tl.where(
            scale_mask,
            scales,
            0.0,
        )

    offs_m = (pid_m * M_PER_BLOCK % 128) + tl.arange(0, M_PER_BLOCK)[:, None]
    # Offset of the 128x4 layout
    layout_off = (
        (pid_m * M_PER_BLOCK) // 128
    ) * NUM_N_BLOCKS * NUM_ELEM_PER_LAYOUT + pid_n * NUM_ELEM_PER_LAYOUT
    scale_offs = layout_off + nvfp4_scale_swizzle(offs_m)
    tl.store(
        s_ptr + scale_offs,
        scales,
    )

    # Convert to FP4
    x_fp4x2 = convert_fp32_to_fp4_packed(x_blocks.reshape(M_PER_BLOCK, 32, 2).split())
    offs_m = pid_m * M_PER_BLOCK + tl.arange(0, M_PER_BLOCK)[:, None]
    offs_n = pid_n * 32 + tl.arange(0, 32)[None, :]
    if USE_MASK:
        mask = (offs_m < M) & (offs_n < N // 2)
    else:
        mask = None

    if USE_INT64_INDEXING:
        offs_m = offs_m.to(tl.int64)
        offs_n = offs_n.to(tl.int64)

    store_offsets = offs_m * (N // 2) + offs_n
    tl.store(q_ptr + store_offsets, x_fp4x2, mask=mask)
