# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

from __future__ import annotations

import math
from typing import Tuple

import torch
import triton  # @manual
from triton import language as tl  # @manual


@triton.jit
def fused_single_block_kernel(
    m_sizes_ptr,  # [num_segments] input sizes
    size_cumulative_ptr,  # [num_segments + 1] cumulative size sum
    starting_row_after_padding_ptr,  # [num_segments + 1] output: padded cumsum
    belong_indices_ptr,  # [N] output: segment index
    row_within_tensor_ptr,  # [N] output: position within segment
    num_segments: tl.constexpr,
    N: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    prefix_num: tl.constexpr,
):
    pid = tl.program_id(0)
    NUM_BLOCKS = tl.num_programs(0)

    offs = tl.arange(0, prefix_num)
    mask = offs < num_segments

    # Load m_sizes
    m_sizes = tl.load(m_sizes_ptr + offs, mask=mask, other=0)

    # Compute inclusive cumsum
    cumsum = tl.cumsum(m_sizes, axis=0)

    # Store cumsum at indices 1 through N
    tl.store(
        size_cumulative_ptr + offs + 1 + (num_segments + 1) * pid, cumsum, mask=mask
    )

    # Set first element to zero
    tl.store(
        size_cumulative_ptr + offs + (num_segments + 1) * pid,
        tl.zeros([1], dtype=cumsum.dtype),
        mask=(offs == 0),
    )

    if pid == 0:
        # Part 1: Compute padded cumsum (only first block does this)
        offs = tl.arange(0, prefix_num)
        mask = offs < num_segments

        # Load m_sizes
        m_sizes = tl.load(m_sizes_ptr + offs, mask=mask, other=0)

        # Compute padded sizes
        padded_sizes = ((m_sizes + 128 - 1) // 128) * 128

        # Compute inclusive cumsum
        cumsum = tl.cumsum(padded_sizes, axis=0)

        # Store at indices 1 through num_segments
        tl.store(starting_row_after_padding_ptr + offs + 1, cumsum, mask=mask)

        # Set first element to zero
        tl.store(
            starting_row_after_padding_ptr + offs,
            tl.zeros([1], dtype=cumsum.dtype),
            mask=(offs == 0),
        )
    tl.debug_barrier()
    # Part 2: Segmented arange - process N elements in chunks
    new_offs = tl.arange(0, BLOCK_SIZE) + BLOCK_SIZE * pid
    for start in range(0, N, BLOCK_SIZE * NUM_BLOCKS):
        row_idx = start + new_offs
        row_mask = row_idx < N

        # Binary search using the cumsum_regular we computed
        left = tl.zeros([BLOCK_SIZE], dtype=tl.int32)
        right = tl.zeros([BLOCK_SIZE], dtype=tl.int32) + num_segments

        for _ in range(64):  # log2(num_segments) iterations
            mid = (left + right) // 2

            # Get cumsum value at mid position
            # Since we need cumsum[0] = 0, cumsum[1] = m_sizes[0], etc.
            mid_val = tl.load(
                size_cumulative_ptr + mid + (num_segments + 1) * pid,
                mask=row_mask,
                other=0,
            )

            cond = mid_val <= row_idx
            left = tl.where(cond, mid + 1, left)
            right = tl.where(~cond, mid, right)

        belong_idx = left - 1
        tl.store(belong_indices_ptr + row_idx, belong_idx, mask=row_mask)

        # Compute row_within_tensor
        segment_start = tl.load(
            size_cumulative_ptr + (num_segments + 1) * pid + belong_idx,
            mask=row_mask,
            other=0,
        )
        row_within = row_idx - segment_start
        tl.store(row_within_tensor_ptr + row_idx, row_within, mask=row_mask)


def fused_single_block_cumsum_and_segmented_arange(m_sizes, N):
    device = m_sizes.device
    dtype = m_sizes.dtype
    num_segments = m_sizes.shape[0]
    NUM_BLOCKS = 256
    # cumulative size for m_sizes
    size_cumulative = torch.empty(
        (num_segments + 1) * NUM_BLOCKS, dtype=dtype, device=device
    )

    # Allocate outputs
    starting_row_after_padding = torch.empty(
        num_segments + 1, dtype=dtype, device=device
    )
    belong_indices = torch.empty(N, dtype=dtype, device=device)
    row_within_tensor = torch.empty(N, dtype=dtype, device=device)

    # Single block handles everything
    BLOCK_SIZE = 512
    fused_single_block_kernel[(NUM_BLOCKS,)](
        m_sizes,
        size_cumulative,
        starting_row_after_padding,
        belong_indices,
        row_within_tensor,
        num_segments=num_segments,
        N=N,
        BLOCK_SIZE=BLOCK_SIZE,
        prefix_num=triton.next_power_of_2(num_segments),
    )

    return starting_row_after_padding, belong_indices, row_within_tensor


@triton.jit
def fused_padding_cumsum_and_segmented_arange_kernel(
    m_sizes_ptr,  # [num_segments] input sizes
    starting_row_after_padding_ptr,  # [num_segments + 1] output: padded cumsum
    size_cumulative_ptr,  # [num_segments + 1] input: regular cumsum
    belong_indices_ptr,  # [N] output: segment index
    row_within_tensor_ptr,  # [N] output: position within segment
    num_segments: tl.constexpr,
    N: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    prefix_num: tl.constexpr,
):
    pid = tl.program_id(0)

    # Part 1: Compute padded cumsum (only first block does this)
    if pid == 0:
        offs = tl.arange(0, prefix_num)
        mask = offs < num_segments

        # Load m_sizes
        m_sizes = tl.load(m_sizes_ptr + offs, mask=mask, other=0)

        # Compute padded sizes
        padded_sizes = ((m_sizes + 128 - 1) // 128) * 128

        # Compute inclusive cumsum
        cumsum = tl.cumsum(padded_sizes, axis=0)

        # Store at indices 1 through num_segments
        tl.store(starting_row_after_padding_ptr + offs + 1, cumsum, mask=mask)

        # Set first element to zero
        first_elem_mask = offs == 0
        tl.store(
            starting_row_after_padding_ptr + offs,
            tl.zeros([prefix_num], dtype=cumsum.dtype),
            mask=first_elem_mask,
        )

    # Part 2: Segmented arange (all blocks do this)
    offs = tl.arange(0, BLOCK_SIZE)
    row_idx = pid * BLOCK_SIZE + offs
    mask = row_idx < N

    # Binary search using the regular cumsum
    left = tl.zeros([BLOCK_SIZE], dtype=tl.int32)
    right = tl.zeros([BLOCK_SIZE], dtype=tl.int32) + num_segments

    for _ in range(32):  # 32 iterations for binary search
        mid = (left + right) // 2
        mid_val = tl.load(size_cumulative_ptr + mid, mask=mask, other=0)
        cond = mid_val <= row_idx
        left = tl.where(cond, mid + 1, left)
        right = tl.where(cond, right, mid)

    belong_idx = left - 1
    tl.store(belong_indices_ptr + row_idx, belong_idx, mask=mask)

    # Compute row_within_tensor
    segment_start = tl.load(size_cumulative_ptr + belong_idx, mask=mask, other=0)
    row_within = row_idx - segment_start
    tl.store(row_within_tensor_ptr + row_idx, row_within, mask=mask)


@triton.jit
def cumsum_kernel(
    m_sizes_ptr,
    size_cumulative_ptr,
    N: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < N

    # Load m_sizes
    m_sizes = tl.load(m_sizes_ptr + offs, mask=mask, other=0)

    # Compute inclusive cumsum
    cumsum = tl.cumsum(m_sizes, axis=0)

    # Store cumsum at indices 1 through N
    tl.store(size_cumulative_ptr + offs + 1, cumsum, mask=mask)

    # Set first element to zero
    first_elem_mask = offs == 0
    tl.store(
        size_cumulative_ptr + offs,
        tl.zeros([BLOCK_SIZE], dtype=cumsum.dtype),
        mask=first_elem_mask,
    )


def nvfp4_fused_padding_cumsum_and_segmented_arange(m_sizes, N):
    device = m_sizes.device
    dtype = m_sizes.dtype
    num_segments = m_sizes.shape[0]

    # First compute regular cumsum (needed for segmented arange)
    size_cumulative = nvfp4_triton_cumsum(m_sizes)

    # Allocate outputs
    starting_row_after_padding = torch.empty(
        num_segments + 1, dtype=dtype, device=device
    )
    belong_indices = torch.empty(N, dtype=dtype, device=device)
    row_within_tensor = torch.empty(N, dtype=dtype, device=device)

    BLOCK_SIZE = 256
    # Need enough blocks to cover N, but at least 1 for the padding cumsum
    grid = (max(1, triton.cdiv(N, BLOCK_SIZE)),)
    prefix_num = triton.next_power_of_2(num_segments)
    fused_padding_cumsum_and_segmented_arange_kernel[grid](
        m_sizes,
        starting_row_after_padding,
        size_cumulative,
        belong_indices,
        row_within_tensor,
        num_segments=num_segments,
        N=N,
        BLOCK_SIZE=BLOCK_SIZE,
        prefix_num=prefix_num,
    )

    return starting_row_after_padding, belong_indices, row_within_tensor


def nvfp4_triton_cumsum(m_sizes):
    device = m_sizes.device
    dtype = m_sizes.dtype
    N = m_sizes.shape[0]

    size_cumulative = torch.empty(N + 1, dtype=dtype, device=device)

    BLOCK_SIZE = triton.next_power_of_2(N)
    grid = (1,)  # single-block kernel

    cumsum_kernel[grid](
        m_sizes,
        size_cumulative,
        N=N,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return size_cumulative


@triton.jit
def _calculate_group_max(
    A,
    m_sizes_ptr,
    out,
    tensor_idx_ptr,
    M,
    K,
    num_segments,
    prefix_num: tl.constexpr,
    search_size,
    search_padded_power: tl.constexpr,
    ELEMENTS_PER_THREAD,
    ROW_PADDING,
    GROUP_LOAD: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    USE_INT64: tl.constexpr,
) -> None:
    pid = tl.program_id(0)

    """
    compute cumulative sum to determine the tensor index of the writes
    """

    offs = tl.arange(0, prefix_num)
    mask = offs < num_segments

    # Load m_sizes
    m_sizes = tl.load(m_sizes_ptr + offs, mask=mask, other=0)

    # Compute inclusive cumsum
    cumsum_shift = tl.cumsum(m_sizes, axis=0)

    """
    begin binary search and max calculation
    """

    # Define Constant Expressions.
    MIN_NORMAL: tl.constexpr = 1e-8  # type: ignore[Incompatible variable type]

    # For very large inputs, we need to use int64 indexes. This is slower but necessary.
    if USE_INT64:
        pid = pid.to(tl.int64)
        M = tl.cast(M, tl.int64)
        K = tl.cast(K, tl.int64)
        ELEMENTS_PER_THREAD = tl.cast(ELEMENTS_PER_THREAD, tl.int64)

    GROUPS_PER_THREAD = tl.cdiv(ELEMENTS_PER_THREAD, K)
    GROUPS_PER_ROW = tl.cdiv(K, GROUP_SIZE)

    # Find starting offsets for this thread. These are calculated before adjusting for padding.
    input_start = pid * GROUPS_PER_THREAD * GROUP_SIZE
    # Initiate offset ranges used in kernel.
    input_offset = tl.arange(0, GROUP_LOAD * GROUP_SIZE) + input_start

    group_idx = pid * GROUPS_PER_THREAD + tl.arange(0, GROUP_LOAD)
    init_offset_row = pid * GROUPS_PER_THREAD // GROUPS_PER_ROW
    row_idx = group_idx // GROUPS_PER_ROW

    # binary search and store the indices of the tensors
    elements_to_search = tl.arange(0, search_padded_power) + init_offset_row
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

    tensor_idx_guard = (row_idx < M) & (group_idx < (GROUPS_PER_THREAD * (pid + 1)))
    tensor_idx = tl.gather(
        left, tl.where(tensor_idx_guard, row_idx - init_offset_row, 0), 0
    )
    tl.store(tensor_idx_ptr + row_idx, tensor_idx, mask=tensor_idx_guard)

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
        group_max = tl.max(tl.abs(a_groups), axis=1)

        global_scale = (
            448.0
            * 6.0
            / (tl.where(group_max == 0, MIN_NORMAL, group_max).to(tl.float32))
        )
        tl.atomic_min(
            out + tensor_idx,
            global_scale,
            # Prevent writing outside this chunk or the main array.
            mask=(group_idx < (GROUPS_PER_THREAD * (pid + 1))) & (row_idx < M),
        )

        # Update offsets so we work on the next block.
        input_offset += GROUP_LOAD * GROUP_SIZE

        # since next group might go across the tensor boundary into a new tensor, recalculate offset
        group_idx += GROUP_LOAD
        row_idx = group_idx // GROUPS_PER_ROW
        tensor_idx_guard = (row_idx < M) & (group_idx < (GROUPS_PER_THREAD * (pid + 1)))
        tensor_idx = tl.gather(
            left, tl.where(tensor_idx_guard, row_idx - init_offset_row, 0), 0
        )
        tl.store(tensor_idx_ptr + row_idx, tensor_idx, mask=tensor_idx_guard)


def calculate_group_max(
    input: torch.Tensor,
    m_sizes: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    assert input.ndim >= 1, f"input.ndim needs to be >= 1, but got {input.ndim}."
    other_dims = 1 if input.ndim == 1 else -1
    input = input.reshape(other_dims, input.shape[-1])
    M, K = input.shape
    device = input.device

    assert input.dtype in (
        torch.float16,
        torch.bfloat16,
    ), f"input.dtype needs to be fp16 or bf16 but got {input.dtype}."

    # float32 type for global max.
    out = torch.full(
        (m_sizes.numel(),),
        torch.finfo(torch.float32).max,
        device=device,
        dtype=torch.float32,
    )
    tensor_idx = torch.empty(M, device=device, dtype=torch.int64)

    # Find how many groups each thread should process. We do this
    # by assuming that it is good to distribute work evenly over threads.
    num_threads = min(1024, math.ceil(math.sqrt(input.numel())))
    # try loading at least 1 row at a time to speed up computation
    group_size = triton.next_power_of_2(K)
    # only works for group load = 1 if K is not a power of 2
    GROUP_LOAD = 1
    elements_per_thread = max(
        (math.ceil(input.numel() / num_threads) + group_size)
        // group_size
        * group_size,
        GROUP_LOAD * group_size,
    )
    # Determine how much padding, if any is needed for each row.
    if K % group_size != 0:
        padding = group_size - (K % group_size)
    else:
        padding = 0

    # Check if we need to use int64 for indexing.
    use_int64 = num_threads * elements_per_thread > 2**31 - 1

    max_row_per_thread = math.ceil(elements_per_thread / K)

    search_size = max_row_per_thread + 3
    search_padded_power = triton.next_power_of_2(search_size)

    # Invoke triton quantization kernel over rows.
    grid = (num_threads,)

    # Single block handles everything
    _calculate_group_max[grid](
        input,
        m_sizes,
        out,
        tensor_idx,
        M=M,
        K=K,
        num_segments=m_sizes.numel(),
        prefix_num=triton.next_power_of_2(m_sizes.numel()),
        search_size=search_size,
        search_padded_power=search_padded_power,
        ELEMENTS_PER_THREAD=elements_per_thread,
        ROW_PADDING=padding,
        # pyre-ignore[6]
        GROUP_LOAD=GROUP_LOAD,
        GROUP_SIZE=group_size,
        # pyre-ignore[6]
        USE_INT64=use_int64,
    )
    return out, tensor_idx
