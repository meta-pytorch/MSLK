# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

from __future__ import annotations

import math
from typing import Optional, Tuple, Union

import torch
import triton  # @manual
from mslk.quantize.triton.legacy.primitives import (
    convert_fp32_to_fp4_packed,
    get_mx4_exp_bias,
    nvfp4_scale_swizzle,
    RoundingMode,
)
from triton import language as tl  # @manual


@triton.jit
def _nvfp4_quantize_stacked_kernel(
    x_ptr,
    global_scale_ptr,
    q_ptr,
    s_ptr,
    m_sizes_ptr,
    stride_xm,
    stride_xn,
    M,
    N,
    NUM_GROUPS: tl.constexpr,
    PREFIX_NUM: tl.constexpr,
    BSEARCH_ITERS: tl.constexpr,
    M_PER_BLOCK: tl.constexpr = 128,  # type: ignore[Incompatible variable type]
):
    """Quantize concatenated MoE activations to NVFP4 with per-expert global scales.

    Uses a 2D tiled grid (cdiv(N, 64), cdiv(M, M_PER_BLOCK)). Each block
    processes an [M_PER_BLOCK, 64] tile. Per-expert global scales are loaded
    from a [num_segments] tensor. Cumsums and binary search are computed
    redundantly in every thread block to eliminate host-side GPU operations.

    Args:
        x_ptr: Input tensor pointer [M, N] (bf16/fp16).
        global_scale_ptr: Per-expert global scales [num_segments] (fp32).
        q_ptr: Output quantized data [M, N//2] (uint8).
        s_ptr: Output scale buffer [padded_total_M, N//16] (uint8).
        m_sizes_ptr: Rows per expert segment [num_segments] (int64).
        stride_xm: Row stride of input.
        stride_xn: Column stride of input.
        M: Total number of rows.
        N: Number of columns.
        NUM_GROUPS: Number of expert segments.
        PREFIX_NUM: next_power_of_2(NUM_GROUPS) for tl.cumsum.
        M_PER_BLOCK: Rows per block (constexpr, power of 2, <= 128, autotuned).
    """
    tl.static_assert(M_PER_BLOCK <= 128)
    E4M3_EPS: tl.constexpr = 1.5258789e-05  # type: ignore[Incompatible variable type]
    FP8_E4M3_MAX = 448.0  # type: ignore[Incompatible variable type]
    FP4_E2M1_MAX: tl.constexpr = 6  # type: ignore[Incompatible variable type]

    NUM_ELEM_PER_LAYOUT: tl.constexpr = (  # type: ignore[Incompatible variable type]
        128 * 4
    )
    NUM_N_BLOCKS = tl.cdiv(N, 64)

    pid_m = tl.program_id(1)
    pid_n = tl.program_id(0)

    # ---- Step 1: Compute cumsums and binary search for segment ownership ----
    seg_offs = tl.arange(0, PREFIX_NUM)
    seg_mask = seg_offs < NUM_GROUPS
    m_vals = tl.load(m_sizes_ptr + seg_offs, mask=seg_mask, other=0)
    cumsum_inc = tl.cumsum(m_vals, axis=0)
    padded_vals = ((m_vals + 127) // 128) * 128
    padded_cumsum_inc = tl.cumsum(padded_vals, axis=0)
    # Exclusive cumsums: [0, m_0, m_0+m_1, ...]
    cumsum_exc = cumsum_inc - m_vals
    padded_cumsum_exc = padded_cumsum_inc - padded_vals

    rows = pid_m * M_PER_BLOCK + tl.arange(0, M_PER_BLOCK)  # [M_PER_BLOCK]

    # Binary search: find largest seg_idx such that cumsum_exc[seg_idx] <= row
    lo = tl.zeros([M_PER_BLOCK], dtype=tl.int32)
    hi = tl.full([M_PER_BLOCK], NUM_GROUPS - 1, dtype=tl.int32)

    # Optimized by KernelAgent: Use ceil(log2(NUM_GROUPS)) iterations
    for _ in range(BSEARCH_ITERS):
        mid = (lo + hi + 1) // 2
        mid_val = tl.gather(cumsum_exc, mid, 0)
        lo = tl.where(mid_val <= rows, mid, lo)
        hi = tl.where(mid_val <= rows, hi, mid - 1)
    seg_idx = lo  # [M_PER_BLOCK]

    # ---- Step 2: Load per-expert global scale for each row ----
    global_scales = tl.load(global_scale_ptr + seg_offs, mask=seg_mask, other=1.0)
    row_scale = tl.gather(global_scales, seg_idx, 0)  # [M_PER_BLOCK]

    # ---- Step 3: Load tile [M_PER_BLOCK, 64] ----
    offs_m = pid_m * M_PER_BLOCK + tl.arange(0, M_PER_BLOCK)[:, None]
    offs_n = pid_n * 64 + tl.arange(0, 64)[None, :]
    mask = (offs_m < M) & (offs_n < N)

    x = tl.load(
        x_ptr + offs_m * stride_xm + offs_n * stride_xn, mask=mask, other=0.0
    )  # [M_PER_BLOCK, 64]

    # ---- Step 4: Quantize ----
    x_blocks = x.to(tl.float32).reshape(M_PER_BLOCK, 4, 16)  # [M_PER_BLOCK, 4, 16]

    # Block-wise max
    block_amax = tl.max(tl.abs(x_blocks), axis=2)  # [M_PER_BLOCK, 4]

    scales = tl.div_rn(block_amax, FP4_E2M1_MAX) * row_scale[:, None]
    scales = tl.clamp(scales, E4M3_EPS, FP8_E4M3_MAX)
    scales = scales.to(tl.float8e4nv)  # [M_PER_BLOCK, 4]

    # Apply combined scale to data
    total_scale = tl.div_rn(
        row_scale[:, None, None], scales.to(tl.float32)[:, :, None]
    )  # [M_PER_BLOCK, 4, 1]
    x_blocks = x_blocks * total_scale  # [M_PER_BLOCK, 4, 16]

    scale_offs_n = pid_n * 4 + tl.arange(0, 4)[None, :]
    scale_mask = (offs_m < M) & (scale_offs_n < (N // 16))
    scales = tl.where(scale_mask, scales, 0.0)

    # ---- Step 5: Write FP4 data (flat, no segment awareness) ----
    x_fp4x2 = convert_fp32_to_fp4_packed(x_blocks.reshape(M_PER_BLOCK, 32, 2).split())
    fp4_offs_m = pid_m * M_PER_BLOCK + tl.arange(0, M_PER_BLOCK)[:, None]
    fp4_offs_n = pid_n * 32 + tl.arange(0, 32)[None, :]
    fp4_mask = (fp4_offs_m < M) & (fp4_offs_n < N // 2)
    tl.store(q_ptr + fp4_offs_m * (N // 2) + fp4_offs_n, x_fp4x2, mask=fp4_mask)

    # ---- Step 6: Compute padded row ----
    seg_cumsum = tl.gather(cumsum_exc, seg_idx, 0)
    seg_padded_cumsum = tl.gather(padded_cumsum_exc, seg_idx, 0)
    padded_r = seg_padded_cumsum + (rows.to(tl.int64) - seg_cumsum)  # [M_PER_BLOCK]

    # ---- Step 7: Write scales to padded+swizzled layout ----
    padded_r_2d = padded_r[:, None]  # [M_PER_BLOCK, 1]
    layout_off = (
        padded_r_2d // 128
    ) * NUM_N_BLOCKS * NUM_ELEM_PER_LAYOUT + pid_n * NUM_ELEM_PER_LAYOUT
    offs_m_in_layout = (padded_r_2d % 128).to(tl.int32)
    scale_offs = layout_off + nvfp4_scale_swizzle(offs_m_in_layout)

    # Mask out rows beyond M and scale columns beyond N//16
    scale_store_offs_n = pid_n * 4 + tl.arange(0, 4)[None, :]
    store_mask = (rows[:, None] < M) & (scale_store_offs_n < (N // 16))
    tl.store(s_ptr + scale_offs, scales.to(tl.uint8, bitcast=True), mask=store_mask)


@triton.jit
def _nvfp4_quantize_stacked_token_scale_fused_row_kernel(
    x_ptr,
    q_ptr,
    s_ptr,
    token_scale_inv_ptr,
    m_sizes_ptr,
    stride_xm,
    stride_xn,
    M,
    N,
    NUM_GROUPS: tl.constexpr,
    PREFIX_NUM: tl.constexpr,
    BLOCK_K: tl.constexpr,
    SCALE_BLOCKS: tl.constexpr,
    USE_FULL_ROW_QUANT: tl.constexpr,
    CHUNK_SIZE: tl.constexpr,
    CHUNK_SCALE_BLOCKS: tl.constexpr,
):
    E4M3_EPS: tl.constexpr = 1.5258789e-05  # type: ignore[Incompatible variable type]
    FP8_E4M3_MAX = 448.0  # type: ignore[Incompatible variable type]
    FP4_E2M1_MAX: tl.constexpr = 6  # type: ignore[Incompatible variable type]
    MIN_NORMAL: tl.constexpr = 1.0e-8  # type: ignore[Incompatible variable type]

    NUM_ELEM_PER_LAYOUT: tl.constexpr = 128 * 4  # type: ignore[Incompatible variable type]
    NUM_N_BLOCKS = tl.cdiv(N, 64)

    row = tl.program_id(0)
    reduce_offs_n = tl.arange(0, BLOCK_K)
    reduce_mask = (row < M) & (reduce_offs_n < N)
    x = tl.load(
        x_ptr + row * stride_xm + reduce_offs_n * stride_xn,
        mask=reduce_mask,
        other=0.0,
    ).to(tl.float32)

    row_amax = tl.max(tl.abs(x), axis=0)
    token_scale_inv = tl.maximum(row_amax, MIN_NORMAL) / (448.0 * 6.0)
    row_scale = tl.div_rn(1.0, token_scale_inv)
    tl.store(token_scale_inv_ptr + row, token_scale_inv, mask=row < M)

    seg_offs = tl.arange(0, PREFIX_NUM)
    seg_mask = seg_offs < NUM_GROUPS
    m_vals = tl.load(m_sizes_ptr + seg_offs, mask=seg_mask, other=0)
    cumsum_inc = tl.cumsum(m_vals, axis=0)
    padded_vals = ((m_vals + 127) // 128) * 128
    padded_cumsum_inc = tl.cumsum(padded_vals, axis=0)
    cumsum_exc = cumsum_inc - m_vals
    padded_cumsum_exc = padded_cumsum_inc - padded_vals

    seg_idx = tl.max(tl.where(cumsum_exc <= row, seg_offs, 0), axis=0)
    seg_cumsum = tl.max(tl.where(seg_offs == seg_idx, cumsum_exc, 0), axis=0)
    seg_padded_cumsum = tl.max(
        tl.where(seg_offs == seg_idx, padded_cumsum_exc, 0), axis=0
    )
    padded_r = seg_padded_cumsum + (row.to(tl.int64) - seg_cumsum)

    if USE_FULL_ROW_QUANT:
        x_blocks = x.reshape(SCALE_BLOCKS, 16)
        block_amax = tl.max(tl.abs(x_blocks), axis=1)
        scales = tl.div_rn(block_amax, FP4_E2M1_MAX) * row_scale
        scales = tl.clamp(scales, E4M3_EPS, FP8_E4M3_MAX)
        scales_fp8 = scales.to(tl.float8e4nv)

        total_scale = tl.div_rn(row_scale, scales_fp8.to(tl.float32)[:, None])
        x_blocks = x_blocks * total_scale

        x_fp4x2 = convert_fp32_to_fp4_packed(x_blocks.reshape(BLOCK_K // 2, 2).split())
        fp4_offs_n = tl.arange(0, BLOCK_K // 2)
        fp4_mask = (row < M) & (fp4_offs_n < N // 2)
        tl.store(q_ptr + row * (N // 2) + fp4_offs_n, x_fp4x2, mask=fp4_mask)

        scale_col = tl.arange(0, SCALE_BLOCKS)
        col_block = scale_col // 4
        col_in_block = scale_col % 4
        offs_m_in_layout = (padded_r % 128).to(tl.int32)
        sub_layout_idx = offs_m_in_layout % 32
        sub_layout_row = offs_m_in_layout // 32
        scale_offs = (
            (padded_r // 128) * NUM_N_BLOCKS * NUM_ELEM_PER_LAYOUT
            + col_block * NUM_ELEM_PER_LAYOUT
            + sub_layout_idx * 16
            + sub_layout_row * 4
            + col_in_block
        )
        scale_mask = (row < M) & (scale_col < (N // 16))
        tl.store(
            s_ptr + scale_offs,
            scales_fp8.to(tl.uint8, bitcast=True),
            mask=scale_mask,
        )
    else:
        chunk_offs = tl.arange(0, CHUNK_SIZE)
        scale_offs_in_chunk = tl.arange(0, CHUNK_SCALE_BLOCKS)
        fp4_offs_in_chunk = tl.arange(0, CHUNK_SIZE // 2)

        for k_start in range(0, N, CHUNK_SIZE):
            chunk_offs_n = k_start + chunk_offs
            chunk_mask = (row < M) & (chunk_offs_n < N)
            x_chunk = tl.load(
                x_ptr + row * stride_xm + chunk_offs_n * stride_xn,
                mask=chunk_mask,
                other=0.0,
            ).to(tl.float32)
            x_blocks = x_chunk.reshape(CHUNK_SCALE_BLOCKS, 16)

            block_amax = tl.max(tl.abs(x_blocks), axis=1)
            scales = tl.div_rn(block_amax, FP4_E2M1_MAX) * row_scale
            scales = tl.clamp(scales, E4M3_EPS, FP8_E4M3_MAX)
            scales_fp8 = scales.to(tl.float8e4nv)

            total_scale = tl.div_rn(row_scale, scales_fp8.to(tl.float32)[:, None])
            x_blocks = x_blocks * total_scale

            x_fp4x2 = convert_fp32_to_fp4_packed(
                x_blocks.reshape(CHUNK_SIZE // 2, 2).split()
            )
            fp4_offs_n = k_start // 2 + fp4_offs_in_chunk
            fp4_mask = (row < M) & (fp4_offs_n < N // 2)
            tl.store(
                q_ptr + row * (N // 2) + fp4_offs_n,
                x_fp4x2,
                mask=fp4_mask,
            )

            scale_col = k_start // 16 + scale_offs_in_chunk
            col_block = scale_col // 4
            col_in_block = scale_col % 4
            offs_m_in_layout = (padded_r % 128).to(tl.int32)
            sub_layout_idx = offs_m_in_layout % 32
            sub_layout_row = offs_m_in_layout // 32
            scale_offs = (
                (padded_r // 128) * NUM_N_BLOCKS * NUM_ELEM_PER_LAYOUT
                + col_block * NUM_ELEM_PER_LAYOUT
                + sub_layout_idx * 16
                + sub_layout_row * 4
                + col_in_block
            )
            scale_mask = (row < M) & (scale_col < (N // 16))
            tl.store(
                s_ptr + scale_offs,
                scales_fp8.to(tl.uint8, bitcast=True),
                mask=scale_mask,
            )


def nvfp4_quantize_stacked(
    m_sizes: torch.Tensor,
    input: torch.Tensor,
    global_scale: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantize concatenated MoE activations to NVFP4 with per-expert global scales.

    Uses per-expert global scales for quantization math, with per-segment
    padded scales for the downstream GEMM. Cumsums are computed inside the
    Triton kernel to avoid GPU launch overhead.

    Args:
        m_sizes: [num_segments] int64 tensor of rows per expert.
        input: [M, K] concatenated activation tensor (bf16/fp16).
        global_scale: [num_segments] fp32 per-expert global scales
            (e.g. from calculate_group_max).

    Returns:
        Tuple of (xq, scale):
            xq: [M, K//2] uint8 packed FP4 data.
            scale: Flattened uint8 scale buffer (padded+swizzled).
    """
    assert input.ndim >= 1, f"input.ndim needs to be >= 1, but got {input.ndim}."
    input = input.reshape(-1, input.shape[-1])
    M, K = input.shape
    num_segments = m_sizes.shape[0]
    device = input.device

    assert K % 16 == 0, f"K must be divisible by 16, but got {K}."
    assert input.dtype in (
        torch.float16,
        torch.bfloat16,
    ), f"input.dtype needs to be fp16 or bf16 but got {input.dtype}."

    # Upper-bound on padded total rows avoids GPU->host sync (.item()) so
    # this function is safe to use inside CUDA-graph capture.  Each of the
    # num_segments segments can add at most 127 padding rows.
    padded_total_M_ub = M + num_segments * 127

    # Allocate outputs
    xq = torch.empty((M, K // 2), device=device, dtype=torch.uint8)

    num_scales_per_row = K // 16
    n_col_blocks = triton.cdiv(num_scales_per_row, 4)
    padded_cols = n_col_blocks * 4
    scale = torch.empty(
        (padded_total_M_ub, padded_cols), device=device, dtype=torch.uint8
    )

    grid = lambda META: (  # noqa E731
        triton.cdiv(K, 64),
        triton.cdiv(M, META["M_PER_BLOCK"]),
    )

    # Optimized by KernelAgent: use larger tiles and software pipelining
    # for M >= 256 where there are enough blocks to saturate SMs.
    # For small M, keep defaults to avoid SM underutilization.
    if M >= 256:
        m_per_block = 128
        n_stages = 3
    else:
        m_per_block = 64
        n_stages = 1

    # pyre-ignore[28]
    _nvfp4_quantize_stacked_kernel[grid](
        input,
        global_scale,
        xq,
        scale,
        m_sizes,
        input.stride(0),
        input.stride(1),
        M,
        K,
        # pyre-ignore[6]
        NUM_GROUPS=num_segments,
        PREFIX_NUM=triton.next_power_of_2(num_segments),
        BSEARCH_ITERS=max(1, (num_segments - 1).bit_length()),
        M_PER_BLOCK=m_per_block,  # pyre-ignore[6]
        num_stages=n_stages,
    )

    return xq.view(torch.float4_e2m1fn_x2), scale.view(torch.float8_e4m3fn)


def nvfp4_quantize_stacked_with_token_scale(
    m_sizes: torch.Tensor,
    input: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantize concatenated MoE activations with reciprocal per-token scales.

    Returns:
        Tuple of (xq, scale, token_scale_inv):
            xq: [M, K//2] packed FP4 data.
            scale: Flattened uint8 scale buffer in padded+swizzled CUTLASS layout.
            token_scale_inv: [M] fp32 reciprocal per-token scales.
    """
    assert input.ndim >= 1, f"input.ndim needs to be >= 1, but got {input.ndim}."
    input = input.reshape(-1, input.shape[-1])
    M, K = input.shape
    num_segments = m_sizes.shape[0]
    device = input.device

    assert K % 16 == 0, f"K must be divisible by 16, but got {K}."
    assert input.dtype in (
        torch.float16,
        torch.bfloat16,
    ), f"input.dtype needs to be fp16 or bf16 but got {input.dtype}."

    padded_total_M_ub = M + num_segments * 127
    xq = torch.empty((M, K // 2), device=device, dtype=torch.uint8)
    x_token_scale_inv = torch.empty((M,), device=device, dtype=torch.float32)

    num_scales_per_row = K // 16
    n_col_blocks = triton.cdiv(num_scales_per_row, 4)
    padded_cols = n_col_blocks * 4
    scale = torch.empty(
        (padded_total_M_ub, padded_cols), device=device, dtype=torch.uint8
    )

    block_k = triton.next_power_of_2(K)
    # pyre-ignore[28]
    _nvfp4_quantize_stacked_token_scale_fused_row_kernel[(M,)](
        input,
        xq,
        scale,
        x_token_scale_inv,
        m_sizes,
        input.stride(0),
        input.stride(1),
        M,
        K,
        NUM_GROUPS=num_segments,  # pyre-ignore[6]
        PREFIX_NUM=triton.next_power_of_2(num_segments),
        BLOCK_K=block_k,  # pyre-ignore[6]
        SCALE_BLOCKS=block_k // 16,  # pyre-ignore[6]
        USE_FULL_ROW_QUANT=block_k == K and K <= 8192,
        CHUNK_SIZE=2048,
        CHUNK_SCALE_BLOCKS=2048 // 16,
        num_warps=8,
    )

    return (
        xq.view(torch.float4_e2m1fn_x2),
        scale.view(torch.float8_e4m3fn),
        x_token_scale_inv,
    )


@triton.jit
def _mega_fp4_quantize_kernel(
    m_sizes_ptr,  # [num_segments] input sizes
    starting_row_after_padding_ptr,  # [num_segments + 1] output: padded cumsum
    search_size,
    search_padded_power: tl.constexpr,
    A,
    input_global_scale_tensor,
    out,
    scale,
    num_segments,
    prefix_num: tl.constexpr,
    rand_bits,
    M,
    K,
    GROUPS_PER_ROW,
    GROUPS_PER_THREAD,
    ROW_PADDING,
    EPS: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    EBITS: tl.constexpr,
    MBITS: tl.constexpr,
    ROUNDING_MODE: tl.constexpr,
    STOCHASTIC_CASTING: tl.constexpr,
    FP4_EXP_BIAS: tl.constexpr,
    GROUP_LOAD: tl.constexpr,
    USE_INT64: tl.constexpr,
    SCALE_K: tl.constexpr,
) -> None:
    """
    computed cumulative sum and padded cumulative sum. All blocks will do this
    in order to ensure that the changes are visible to all blocks without global synchronization
    """
    pid = tl.program_id(0)

    offs = tl.arange(0, prefix_num)
    mask = offs < num_segments

    # Load m_sizes
    m_sizes = tl.load(m_sizes_ptr + offs, mask=mask, other=0)

    # Compute inclusive cumsum
    cumsum = tl.cumsum(m_sizes, axis=0)

    # padded cumsum
    padded = ((m_sizes + 128 - 1) // 128) * 128
    # Compute inclusive cumsum
    padded_cumsum = tl.cumsum(padded, axis=0)

    if pid == 0:
        # Store at indices 1 through num_segments
        tl.store(
            starting_row_after_padding_ptr + offs + 1 + (num_segments + 1) * pid,
            padded_cumsum,
            mask=mask,
        )

        # Set first element to zero
        tl.store(
            starting_row_after_padding_ptr + offs + (num_segments + 1) * pid,
            tl.zeros([1], dtype=cumsum.dtype),
            mask=(offs == 0),
        )
    cumsum_shift = cumsum
    cumsum = cumsum - m_sizes
    padded_cumsum = padded_cumsum - padded

    """
    begin quantization
    """

    # Define Constant Expressions.
    BF16_MIN_NORMAL: tl.constexpr = 2 ** (-126)  # type: ignore[Incompatible variable type]

    # For very large inputs, we need to use int64 indexes. This is slower but necessary.
    if USE_INT64:
        pid = pid.to(tl.int64)
        M = tl.cast(M, tl.int64)
        K = tl.cast(K, tl.int64)
        GROUPS_PER_THREAD = tl.cast(GROUPS_PER_THREAD, tl.int64)

    # Boundaries for writing to output tensor.
    NUM_GROUPS = M * GROUPS_PER_ROW
    OUTPUT_CHUNK_SIZE = (GROUPS_PER_THREAD * GROUP_SIZE) // 8
    SCALE_CHUNK_SIZE = GROUPS_PER_THREAD
    OUTPUT_SIZE = (GROUP_SIZE * NUM_GROUPS) // 8
    SCALE_SIZE = NUM_GROUPS

    # Find starting offsets for this thread. These are calculated before adjusting for padding.
    input_start = pid * (GROUPS_PER_THREAD * GROUP_SIZE)
    output_start = pid * OUTPUT_CHUNK_SIZE
    exp_start = pid * SCALE_CHUNK_SIZE
    # Initiate offset ranges used in kernel.
    input_offset = tl.arange(0, GROUP_LOAD * GROUP_SIZE) + input_start
    output_offset = tl.arange(0, GROUP_LOAD * (GROUP_SIZE // 8)) + output_start

    # We need to shift output offsets to make space for shared exponent storage.
    # Now create offsets for writing the shared exponent.
    # make sure to add offset from padding
    exp_offset = tl.arange(0, GROUP_LOAD) + exp_start

    row_idx = exp_offset // GROUPS_PER_ROW

    init_offset_exp = exp_start // GROUPS_PER_ROW

    # binary search and store the indices of the tensors
    elements_to_search = tl.arange(0, search_padded_power) + init_offset_exp
    left = tl.zeros([search_padded_power], dtype=tl.int32)
    right = tl.zeros([search_padded_power], dtype=tl.int32) + num_segments
    search_guard = (tl.arange(0, search_padded_power) < search_size) & (
        elements_to_search < M
    )
    for _ in range(32):  # log2(num_segments) iterations
        mid = tl.where(search_guard, (left + right) // 2, 0)

        # Get cumsum value at mid position
        # Since we need cumsum[0] = 0, cumsum[1] = m_sizes[0], etc.
        mid_val = tl.gather(cumsum_shift, mid, 0)

        cond = mid_val <= elements_to_search
        left = tl.where(cond, mid + 1, left)
        right = tl.where(~cond, mid, right)

    tensor_idx_guard = (
        (row_idx < M)
        & (exp_offset < (SCALE_CHUNK_SIZE * (pid + 1)))
        & (exp_offset < SCALE_SIZE)
    )
    tensor_idx = tl.gather(
        left, tl.where(tensor_idx_guard, row_idx - init_offset_exp, 0), 0
    )

    tensor_offset = (
        tl.gather(padded_cumsum, tl.where(tensor_idx_guard, tensor_idx, 0), 0)
        * GROUPS_PER_ROW
    )

    inner_idx = (
        row_idx - tl.gather(cumsum, tl.where(tensor_idx_guard, tensor_idx, 0), 0)
    ) * GROUPS_PER_ROW

    actual_scale_offset = tensor_offset + inner_idx + exp_offset % GROUPS_PER_ROW

    # Load and process blocks of values for this chunk.
    for _k in range(0, tl.cdiv(GROUPS_PER_THREAD, GROUP_LOAD)):
        # We need to make some adjustments to allow for padding.
        pad_mask = (input_offset % (GROUPS_PER_ROW * GROUP_SIZE)) < K
        if ROW_PADDING != 0:
            # Shift the input to account for padding.
            padded_input_offset = (
                input_offset
                - (input_offset // (GROUPS_PER_ROW * GROUP_SIZE)) * ROW_PADDING
            )
        # When there's no padding we can simplify indexing.
        else:
            padded_input_offset = input_offset

        # Load a block of values.
        a = tl.load(
            A + padded_input_offset,
            # Mask values out of range for both the main array and this chunk. Also pad if needed.
            mask=(padded_input_offset < (M * K))
            & (padded_input_offset < ((pid + 1) * GROUPS_PER_THREAD * GROUP_SIZE))
            & pad_mask,
            other=0,
        )

        # View the block in terms of groups.
        a_groups = tl.reshape(a, [GROUP_LOAD, GROUP_SIZE])
        # Compute the shared exponent of each group.
        group_max = tl.max(tl.abs(a_groups), axis=1).to(tl.float32)

        # load scaling factor
        input_global_scale = tl.load(
            input_global_scale_tensor + tensor_idx, mask=tensor_idx_guard
        )
        # Next we scale A in preparation for quantization.
        scale_ = (group_max / 6.0 * input_global_scale).to(tl.float8e4nv)
        # Prevent infinite values in log.
        group_max = tl.where(group_max == 0, BF16_MIN_NORMAL, group_max)

        # Apply scale_ to input. We do this by broadcasting scale.
        # scaled_a = a * global_scale (fp32) / local_scale (fp8)
        scaled_a = tl.reshape(a, [GROUP_LOAD, GROUP_SIZE]) * tl.reshape(
            input_global_scale / scale_, [GROUP_LOAD, 1]
        )
        # Reshape back to a flat array.
        scaled_a = tl.reshape(scaled_a, [GROUP_LOAD * GROUP_SIZE])
        # split them into 8 arrays with in a round robin fashion
        # element 0 -> array 0, element 1 -> array 1, element 2 -> array 2 and so on
        temp_l, temp_r = tl.split(
            tl.reshape(scaled_a, [(GROUP_LOAD * GROUP_SIZE) // 2, 2])
        )  # 0, 2, 4, 6, 8 || 1, 3, 5, 7, 9
        t_one, t_two = tl.split(
            tl.reshape(temp_l, [(GROUP_LOAD * GROUP_SIZE) // 4, 2])
        )  # 0 4 8 || 2, 6, 10
        t_three, t_four = tl.split(
            tl.reshape(temp_r, [(GROUP_LOAD * GROUP_SIZE) // 4, 2])
        )  # 1, 5, 9 || 3, 7, 11

        f_one, f_two = tl.split(
            tl.reshape(t_one, [(GROUP_LOAD * GROUP_SIZE) // 8, 2])
        )  # 0, 8 || 4, 12
        f_three, f_four = tl.split(
            tl.reshape(t_two, [(GROUP_LOAD * GROUP_SIZE) // 8, 2])
        )  # 2, 10 || 6, 14
        f_five, f_six = tl.split(
            tl.reshape(t_three, [(GROUP_LOAD * GROUP_SIZE) // 8, 2])
        )  # 1, 9 || 5, 13
        f_seven, f_eight = tl.split(
            tl.reshape(t_four, [(GROUP_LOAD * GROUP_SIZE) // 8, 2])
        )  # 3, 11 || 7, 15
        packed_result = tl.inline_asm_elementwise(
            asm="""
            {
                .reg .b8 byte0;
                .reg .b8 byte1;
                .reg .b8 byte2;
                .reg .b8 byte3;
                cvt.rn.satfinite.e2m1x2.f32  byte0, $2, $1;
                cvt.rn.satfinite.e2m1x2.f32  byte1, $4, $3;
                cvt.rn.satfinite.e2m1x2.f32  byte2, $6, $5;
                cvt.rn.satfinite.e2m1x2.f32  byte3, $8, $7;
                mov.b32 $0, {byte0, byte1, byte2, byte3};

            }
            """,
            constraints="=r,f, f, f, f, f, f, f, f",
            args=[f_one, f_five, f_three, f_seven, f_two, f_six, f_four, f_eight],
            dtype=tl.int32,
            is_pure=True,
            pack=1,
        )

        n_col_blocks = SCALE_K // 4
        first_dim = actual_scale_offset // (512 * n_col_blocks)
        second_dim = (actual_scale_offset % (512 * n_col_blocks)) // (
            128 * n_col_blocks
        )
        third_dim = (actual_scale_offset % (128 * n_col_blocks)) // (4 * n_col_blocks)
        fourth_dim = (actual_scale_offset % (4 * n_col_blocks)) // 4
        fifth_dim = actual_scale_offset % 4
        actual_scale_offset_permute = (
            first_dim * (512 * n_col_blocks)
            + fourth_dim * (512)
            + third_dim * (16)
            + second_dim * (4)
            + fifth_dim
        )

        tl.store(
            scale + actual_scale_offset_permute,
            scale_.to(tl.uint8, bitcast=True),
            # Prevent writing outside this chunk or the main array.
            mask=(row_idx < M)
            & (exp_offset < (SCALE_CHUNK_SIZE * (pid + 1)))
            & (exp_offset < SCALE_SIZE),
        )
        # Write out packed values to output tensor.
        tl.store(
            out + output_offset,
            packed_result,
            # Prevent writing outside this chunk or the main array.
            mask=(output_offset < OUTPUT_SIZE)
            & (output_offset < (OUTPUT_CHUNK_SIZE * (pid + 1))),
        )

        # Update offsets so we work on the next block.
        input_offset += GROUP_LOAD * GROUP_SIZE

        # since next group might go across the tensor boundary into a new tensor, recalculate offset
        exp_offset += GROUP_LOAD
        row_idx = exp_offset // GROUPS_PER_ROW

        tensor_idx_guard = (
            (row_idx < M)
            & (exp_offset < (SCALE_CHUNK_SIZE * (pid + 1)))
            & (exp_offset < SCALE_SIZE)
        )
        tensor_idx = tl.gather(
            left, tl.where(tensor_idx_guard, row_idx - init_offset_exp, 0), 0
        )

        tensor_offset = (
            tl.gather(padded_cumsum, tl.where(tensor_idx_guard, tensor_idx, 0), 0)
            * GROUPS_PER_ROW
        )

        inner_idx = (
            row_idx - tl.gather(cumsum, tl.where(tensor_idx_guard, tensor_idx, 0), 0)
        ) * GROUPS_PER_ROW

        actual_scale_offset = tensor_offset + inner_idx + exp_offset % GROUPS_PER_ROW

        output_offset += GROUP_LOAD * GROUP_SIZE // 8


@triton.jit
def _mega_fp4_quantize_kernel_with_tensor_idx(
    m_sizes_ptr,  # [num_segments] input sizes
    starting_row_after_padding_ptr,  # [num_segments + 1] output: padded cumsum
    A,
    input_global_scale_tensor,
    out,
    scale,
    tensor_idx_ptr,
    num_segments,
    prefix_num: tl.constexpr,
    rand_bits,
    M,
    K,
    GROUPS_PER_ROW,
    GROUPS_PER_THREAD,
    ROW_PADDING,
    EPS: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    EBITS: tl.constexpr,
    MBITS: tl.constexpr,
    ROUNDING_MODE: tl.constexpr,
    STOCHASTIC_CASTING: tl.constexpr,
    FP4_EXP_BIAS: tl.constexpr,
    GROUP_LOAD: tl.constexpr,
    USE_INT64: tl.constexpr,
    SCALE_K: tl.constexpr,
) -> None:
    """
    computed cumulative sum and padded cumulative sum. All blocks will do this
    in order to ensure that the changes are visible to all blocks without global synchronization
    """
    pid = tl.program_id(0)

    offs = tl.arange(0, prefix_num)
    mask = offs < num_segments

    # Load m_sizes
    m_sizes = tl.load(m_sizes_ptr + offs, mask=mask, other=0)

    # Compute inclusive cumsum
    cumsum = tl.cumsum(m_sizes, axis=0)

    # padded cumsum
    padded = ((m_sizes + 128 - 1) // 128) * 128
    # Compute inclusive cumsum
    padded_cumsum = tl.cumsum(padded, axis=0)

    if pid == 0:
        # Store at indices 1 through num_segments
        tl.store(
            starting_row_after_padding_ptr + offs + 1 + (num_segments + 1) * pid,
            padded_cumsum,
            mask=mask,
        )

        # Set first element to zero
        tl.store(
            starting_row_after_padding_ptr + offs + (num_segments + 1) * pid,
            tl.zeros([1], dtype=cumsum.dtype),
            mask=(offs == 0),
        )
    cumsum = cumsum - m_sizes
    padded_cumsum = padded_cumsum - padded

    """
    begin quantization
    """

    # Define Constant Expressions.
    BF16_MIN_NORMAL: tl.constexpr = 2 ** (-126)  # type: ignore[Incompatible variable type]

    # For very large inputs, we need to use int64 indexes. This is slower but necessary.
    if USE_INT64:
        pid = pid.to(tl.int64)
        M = tl.cast(M, tl.int64)
        K = tl.cast(K, tl.int64)
        GROUPS_PER_THREAD = tl.cast(GROUPS_PER_THREAD, tl.int64)

    # Boundaries for writing to output tensor.
    NUM_GROUPS = M * GROUPS_PER_ROW
    OUTPUT_CHUNK_SIZE = (GROUPS_PER_THREAD * GROUP_SIZE) // 8
    SCALE_CHUNK_SIZE = GROUPS_PER_THREAD
    OUTPUT_SIZE = (GROUP_SIZE * NUM_GROUPS) // 8
    SCALE_SIZE = NUM_GROUPS

    # Find starting offsets for this thread. These are calculated before adjusting for padding.
    input_start = pid * (GROUPS_PER_THREAD * GROUP_SIZE)
    output_start = pid * OUTPUT_CHUNK_SIZE
    exp_start = pid * SCALE_CHUNK_SIZE
    # Initiate offset ranges used in kernel.
    input_offset = tl.arange(0, GROUP_LOAD * GROUP_SIZE) + input_start
    output_offset = tl.arange(0, GROUP_LOAD * (GROUP_SIZE // 8)) + output_start

    # We need to shift output offsets to make space for shared exponent storage.
    # Now create offsets for writing the shared exponent.
    # make sure to add offset from padding
    exp_offset = tl.arange(0, GROUP_LOAD) + exp_start

    row_idx = exp_offset // GROUPS_PER_ROW

    tensor_idx_guard = (
        (row_idx < M)
        & (exp_offset < (SCALE_CHUNK_SIZE * (pid + 1)))
        & (exp_offset < SCALE_SIZE)
    )
    # tensor_idx = tl.gather(
    #     left, tl.where(tensor_idx_guard, row_idx - init_offset_exp, 0), 0
    # )
    tensor_idx = tl.load(tensor_idx_ptr + row_idx, mask=tensor_idx_guard, other=0)

    tensor_offset = tl.gather(padded_cumsum, tensor_idx, 0) * GROUPS_PER_ROW

    inner_idx = (row_idx - tl.gather(cumsum, tensor_idx, 0)) * GROUPS_PER_ROW

    actual_scale_offset = tensor_offset + inner_idx + exp_offset % GROUPS_PER_ROW

    # Load and process blocks of values for this chunk.
    for _k in range(0, tl.cdiv(GROUPS_PER_THREAD, GROUP_LOAD)):
        # We need to make some adjustments to allow for padding.
        pad_mask = (input_offset % (GROUPS_PER_ROW * GROUP_SIZE)) < K
        if ROW_PADDING != 0:
            # Shift the input to account for padding.
            padded_input_offset = (
                input_offset
                - (input_offset // (GROUPS_PER_ROW * GROUP_SIZE)) * ROW_PADDING
            )
        # When there's no padding we can simplify indexing.
        else:
            padded_input_offset = input_offset

        # Load a block of values.
        a = tl.load(
            A + padded_input_offset,
            # Mask values out of range for both the main array and this chunk. Also pad if needed.
            mask=(padded_input_offset < (M * K))
            & (padded_input_offset < ((pid + 1) * GROUPS_PER_THREAD * GROUP_SIZE))
            & pad_mask,
            other=0,
        )

        # View the block in terms of groups.
        a_groups = tl.reshape(a, [GROUP_LOAD, GROUP_SIZE])
        # Compute the shared exponent of each group.
        group_max = tl.max(tl.abs(a_groups), axis=1).to(tl.float32)

        # load scaling factor
        input_global_scale = tl.load(
            input_global_scale_tensor + tensor_idx, mask=tensor_idx_guard
        )
        # Next we scale A in preparation for quantization.
        scale_ = (group_max / 6.0 * input_global_scale).to(tl.float8e4nv)
        # Prevent infinite values in log.
        group_max = tl.where(group_max == 0, BF16_MIN_NORMAL, group_max)

        # Apply scale_ to input. We do this by broadcasting scale.
        # scaled_a = a * global_scale (fp32) / local_scale (fp8)
        scaled_a = tl.reshape(a, [GROUP_LOAD, GROUP_SIZE]) * tl.reshape(
            input_global_scale / scale_, [GROUP_LOAD, 1]
        )
        # Reshape back to a flat array.
        scaled_a = tl.reshape(scaled_a, [GROUP_LOAD * GROUP_SIZE])
        # split them into 8 arrays with in a round robin fashion
        # element 0 -> array 0, element 1 -> array 1, element 2 -> array 2 and so on
        temp_l, temp_r = tl.split(
            tl.reshape(scaled_a, [(GROUP_LOAD * GROUP_SIZE) // 2, 2])
        )  # 0, 2, 4, 6, 8 || 1, 3, 5, 7, 9
        t_one, t_two = tl.split(
            tl.reshape(temp_l, [(GROUP_LOAD * GROUP_SIZE) // 4, 2])
        )  # 0 4 8 || 2, 6, 10
        t_three, t_four = tl.split(
            tl.reshape(temp_r, [(GROUP_LOAD * GROUP_SIZE) // 4, 2])
        )  # 1, 5, 9 || 3, 7, 11

        f_one, f_two = tl.split(
            tl.reshape(t_one, [(GROUP_LOAD * GROUP_SIZE) // 8, 2])
        )  # 0, 8 || 4, 12
        f_three, f_four = tl.split(
            tl.reshape(t_two, [(GROUP_LOAD * GROUP_SIZE) // 8, 2])
        )  # 2, 10 || 6, 14
        f_five, f_six = tl.split(
            tl.reshape(t_three, [(GROUP_LOAD * GROUP_SIZE) // 8, 2])
        )  # 1, 9 || 5, 13
        f_seven, f_eight = tl.split(
            tl.reshape(t_four, [(GROUP_LOAD * GROUP_SIZE) // 8, 2])
        )  # 3, 11 || 7, 15
        packed_result = tl.inline_asm_elementwise(
            asm="""
            {
                .reg .b8 byte0;
                .reg .b8 byte1;
                .reg .b8 byte2;
                .reg .b8 byte3;
                cvt.rn.satfinite.e2m1x2.f32  byte0, $2, $1;
                cvt.rn.satfinite.e2m1x2.f32  byte1, $4, $3;
                cvt.rn.satfinite.e2m1x2.f32  byte2, $6, $5;
                cvt.rn.satfinite.e2m1x2.f32  byte3, $8, $7;
                mov.b32 $0, {byte0, byte1, byte2, byte3};

            }
            """,
            constraints="=r,f, f, f, f, f, f, f, f",
            args=[f_one, f_five, f_three, f_seven, f_two, f_six, f_four, f_eight],
            dtype=tl.int32,
            is_pure=True,
            pack=1,
        )

        n_col_blocks = SCALE_K // 4
        first_dim = actual_scale_offset // (512 * n_col_blocks)
        second_dim = (actual_scale_offset % (512 * n_col_blocks)) // (
            128 * n_col_blocks
        )
        third_dim = (actual_scale_offset % (128 * n_col_blocks)) // (4 * n_col_blocks)
        fourth_dim = (actual_scale_offset % (4 * n_col_blocks)) // 4
        fifth_dim = actual_scale_offset % 4
        actual_scale_offset_permute = (
            first_dim * (512 * n_col_blocks)
            + fourth_dim * (512)
            + third_dim * (16)
            + second_dim * (4)
            + fifth_dim
        )

        tl.store(
            scale + actual_scale_offset_permute,
            scale_.to(tl.uint8, bitcast=True),
            # Prevent writing outside this chunk or the main array.
            mask=(row_idx < M)
            & (exp_offset < (SCALE_CHUNK_SIZE * (pid + 1)))
            & (exp_offset < SCALE_SIZE),
        )
        # Write out packed values to output tensor.
        tl.store(
            out + output_offset,
            packed_result,
            # Prevent writing outside this chunk or the main array.
            mask=(output_offset < OUTPUT_SIZE)
            & (output_offset < (OUTPUT_CHUNK_SIZE * (pid + 1))),
        )

        # Update offsets so we work on the next block.
        input_offset += GROUP_LOAD * GROUP_SIZE

        # since next group might go across the tensor boundary into a new tensor, recalculate offset
        exp_offset += GROUP_LOAD
        row_idx = exp_offset // GROUPS_PER_ROW

        tensor_idx_guard = (
            (row_idx < M)
            & (exp_offset < (SCALE_CHUNK_SIZE * (pid + 1)))
            & (exp_offset < SCALE_SIZE)
        )
        tensor_idx = tl.load(tensor_idx_ptr + row_idx, mask=tensor_idx_guard, other=0)

        tensor_offset = tl.gather(padded_cumsum, tensor_idx, 0) * GROUPS_PER_ROW

        inner_idx = (row_idx - tl.gather(cumsum, tensor_idx, 0)) * GROUPS_PER_ROW

        actual_scale_offset = tensor_offset + inner_idx + exp_offset % GROUPS_PER_ROW

        output_offset += GROUP_LOAD * GROUP_SIZE // 8


def mega_fp4_quantize_kernel(
    m_sizes: torch.Tensor,
    input: torch.Tensor,
    input_global_scale: torch.Tensor,
    optional_tensor_idx: Optional[torch.Tensor] = None,
    group_size: int = 16,
    ebits: int = 2,
    mbits: int = 1,
    rounding_mode: Union[RoundingMode, int] = RoundingMode.ceil,
    stochastic_casting: bool = False,
    EPS: float = 1e-5,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    orig_shape = input.shape
    assert input.ndim >= 1, f"input.ndim needs to be >= 1, but got {input.ndim}."
    other_dims = 1 if input.ndim == 1 else -1
    input = input.reshape(other_dims, input.shape[-1])
    M, K = input.shape
    block_size = group_size
    device = input.device

    assert K % block_size == 0, f"last dim has to be multiple of 16, but got {K}."
    assert input.dtype in (
        torch.float16,
        torch.bfloat16,
    ), f"input.dtype needs to be fp16 or bf16 but got {input.dtype}."

    # Two fp4 values will be packed into an uint8.
    out = torch.empty((M, K // 8), device=device, dtype=torch.uint32)

    # We use the rounded values to store the swizzled values. Due to the
    # requirement of the Tensor Core, the minimum tile is 128x4 for the scales.
    # So, we first pad the scales to multiples of 128 and 4. Then, the scales
    # (in float8_e4m3fn) int8. More:
    # https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-mma-scale-factor-b-layout-4x
    def round_up(x: int, y: int) -> int:
        return (x + y - 1) // y * y

    rounded_M = round_up(M + m_sizes.shape[0] * 128, 128)
    scale_K = K // block_size
    rounded_K = round_up(scale_K, 4)
    scale = torch.empty((rounded_M, rounded_K), device=device, dtype=torch.uint8)

    # In this kernel, we want each row to be divisible by group_size.
    # If the rows are not, then we will pad them. Find the number of
    # groups per row after padding.
    groups_per_row = math.ceil(K / group_size)
    num_groups = M * groups_per_row
    # Find how many groups each thread should process. We do this
    # by assuming that it is good to distribute work evenly over threads.
    num_threads = min(1024, math.ceil(math.sqrt(input.numel())))
    # Data is loaded in chunks of GROUP_LOAD elements, so theres no reason
    # to ever fewer groups per thread than it.
    GROUP_LOAD = 128
    groups_per_thread = max(math.ceil(num_groups / num_threads), GROUP_LOAD)
    # Determine how much padding, if any is needed for each row.
    if K % group_size != 0:
        padding = group_size - (K % group_size)
    else:
        padding = 0
    # If using stochastic rounding, create random noise for each group.
    # We use the same random bits as seeds when doing stochastic downcasting.
    if rounding_mode == RoundingMode.stochastic or stochastic_casting:
        # Each group will need a seed.
        rand_bits = torch.randint(
            low=0,
            high=2**31 - 1,
            size=(num_groups,),
            dtype=torch.int32,
            device=input.device,
        )
    else:
        rand_bits = None

    # Check if we need to use int64 for indexing.
    use_int64 = num_threads * groups_per_thread * group_size > 2**31 - 1

    device = m_sizes.device
    dtype = m_sizes.dtype
    num_segments = m_sizes.shape[0]
    max_row_per_thread = math.ceil(groups_per_thread / groups_per_row)

    # Allocate outputs
    starting_row_after_padding = torch.empty(
        (num_segments + 1), dtype=dtype, device=device
    )

    search_size = max_row_per_thread + 3
    search_padded_power = triton.next_power_of_2(search_size)

    # Invoke triton quantization kernel over rows.
    grid = (num_threads,)

    # Single block handles everything
    if optional_tensor_idx is None:
        _mega_fp4_quantize_kernel[grid](
            m_sizes,
            starting_row_after_padding,
            search_size,
            search_padded_power,
            input,
            input_global_scale,
            out,
            scale,
            num_segments=num_segments,
            prefix_num=triton.next_power_of_2(num_segments),
            rand_bits=rand_bits,
            M=M,
            K=K,
            GROUPS_PER_ROW=groups_per_row,
            GROUPS_PER_THREAD=groups_per_thread,
            ROW_PADDING=padding,
            # pyre-ignore[6]
            EPS=EPS,
            # pyre-ignore[6]
            GROUP_SIZE=group_size,
            # pyre-ignore[6]
            EBITS=ebits,
            # pyre-ignore[6]
            MBITS=mbits,
            # pyre-ignore[6]
            ROUNDING_MODE=rounding_mode,
            # pyre-ignore[6]
            STOCHASTIC_CASTING=stochastic_casting,
            FP4_EXP_BIAS=get_mx4_exp_bias(ebits),
            # pyre-ignore[6]
            GROUP_LOAD=GROUP_LOAD,
            # pyre-ignore[6]
            USE_INT64=use_int64,
            # pyre-ignore[6]
            SCALE_K=rounded_K,
        )
    else:
        _mega_fp4_quantize_kernel_with_tensor_idx[grid](
            m_sizes,
            starting_row_after_padding,
            input,
            input_global_scale,
            out,
            scale,
            optional_tensor_idx,
            num_segments=num_segments,
            prefix_num=triton.next_power_of_2(num_segments),
            rand_bits=rand_bits,
            M=M,
            K=K,
            GROUPS_PER_ROW=groups_per_row,
            GROUPS_PER_THREAD=groups_per_thread,
            ROW_PADDING=padding,
            # pyre-ignore[6]
            EPS=EPS,
            # pyre-ignore[6]
            GROUP_SIZE=group_size,
            # pyre-ignore[6]
            EBITS=ebits,
            # pyre-ignore[6]
            MBITS=mbits,
            # pyre-ignore[6]
            ROUNDING_MODE=rounding_mode,
            # pyre-ignore[6]
            STOCHASTIC_CASTING=stochastic_casting,
            FP4_EXP_BIAS=get_mx4_exp_bias(ebits),
            # pyre-ignore[6]
            GROUP_LOAD=GROUP_LOAD,
            # pyre-ignore[6]
            USE_INT64=use_int64,
            # pyre-ignore[6]
            SCALE_K=rounded_K,
        )
    scale = scale.flatten()
    return (
        out.view(list(orig_shape[:-1]) + [-1]).view(torch.uint8),
        scale,
        starting_row_after_padding,
    )
