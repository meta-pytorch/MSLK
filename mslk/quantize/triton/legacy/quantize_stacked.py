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
    convert_fp32_to_fp4_packed,
    nvfp4_scale_swizzle,
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
        # pyrefly: ignore [bad-assignment]
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
        # pyrefly: ignore [bad-argument-type]
        BSEARCH_ITERS=max(1, (num_segments - 1).bit_length()),
        M_PER_BLOCK=m_per_block,  # pyre-ignore[6]
        # pyrefly: ignore [unexpected-keyword]
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
        # pyrefly: ignore [bad-argument-type]
        USE_FULL_ROW_QUANT=block_k == K and K <= 8192,
        # pyrefly: ignore [bad-argument-type]
        CHUNK_SIZE=2048,
        # pyrefly: ignore [bad-argument-type]
        CHUNK_SCALE_BLOCKS=2048 // 16,
        # pyrefly: ignore [unexpected-keyword]
        num_warps=8,
    )

    return (
        xq.view(torch.float4_e2m1fn_x2),
        scale.view(torch.float8_e4m3fn),
        x_token_scale_inv,
    )
