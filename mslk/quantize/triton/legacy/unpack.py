# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import triton  # @manual
from triton import language as tl  # @manual


@triton.jit
def _mega_fp4_pack_kernel(
    A,
    input_global_scale_tensor,
    out,
    M,
    K,
    GROUPS_PER_ROW,
    GROUPS_PER_THREAD,
    ROW_PADDING,
    GROUP_SIZE: tl.constexpr,
    GROUP_LOAD: tl.constexpr,
    USE_INT64: tl.constexpr,
) -> None:
    pid = tl.program_id(0)

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

    # load scaling factor
    input_global_scale = tl.load(input_global_scale_tensor)

    # Find starting offsets for this thread. These are calculated before adjusting for padding.
    SCALE_SHIFT = OUTPUT_SIZE * 4
    input_start = pid * (GROUPS_PER_THREAD * GROUP_SIZE)
    output_start = pid * OUTPUT_CHUNK_SIZE
    exp_start = pid * SCALE_CHUNK_SIZE + SCALE_SHIFT
    # Initiate offset ranges used in kernel.
    input_offset = tl.arange(0, GROUP_LOAD * GROUP_SIZE) + input_start
    output_offset = tl.arange(0, GROUP_LOAD * (GROUP_SIZE // 8)) + output_start

    # We need to shift output offsets to make space for shared exponent storage.
    # Now create offsets for writing the shared exponent.
    # make sure to add offset from padding
    exp_offset = tl.arange(0, GROUP_LOAD) + exp_start

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

        tl.store(
            out + exp_offset,
            scale_.to(tl.uint8, bitcast=True),
            # Prevent writing outside this chunk or the main array.
            mask=(exp_offset < (SCALE_CHUNK_SIZE * (pid + 1) + SCALE_SHIFT))
            & (exp_offset < SCALE_SIZE + SCALE_SHIFT),
        )
        # Write out packed values to output tensor.
        ptr_int32 = out.to(tl.pointer_type(tl.int32))
        tl.store(
            ptr_int32 + output_offset,
            packed_result,
            # Prevent writing outside this chunk or the main array.
            mask=(output_offset < OUTPUT_SIZE)
            & (output_offset < (OUTPUT_CHUNK_SIZE * (pid + 1))),
        )

        # Update offsets so we work on the next block.
        input_offset += GROUP_LOAD * GROUP_SIZE

        # since next group might go across the tensor boundary into a new tensor, recalculate offset
        exp_offset += GROUP_LOAD
        output_offset += GROUP_LOAD * GROUP_SIZE // 8


@triton.jit
def _mega_fp4_pack_kernel_per_tensor(
    m_sizes_ptr,
    search_size,
    search_padded_power: tl.constexpr,
    A,
    input_global_scale_tensor,
    out,
    M,
    K,
    GROUPS_PER_ROW,
    GROUPS_PER_THREAD,
    num_segments,
    prefix_num: tl.constexpr,
    ROW_PADDING,
    GROUP_SIZE: tl.constexpr,
    GROUP_LOAD: tl.constexpr,
    USE_INT64: tl.constexpr,
) -> None:
    pid = tl.program_id(0)

    offs = tl.arange(0, prefix_num)
    mask = offs < num_segments

    # Load m_sizes
    m_sizes = tl.load(m_sizes_ptr + offs, mask=mask, other=0)

    # Compute inclusive cumsum
    cumsum_shift = tl.cumsum(m_sizes, axis=0)

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
    SCALE_SHIFT = OUTPUT_SIZE * 4
    input_start = pid * (GROUPS_PER_THREAD * GROUP_SIZE)
    output_start = pid * OUTPUT_CHUNK_SIZE
    exp_start = pid * SCALE_CHUNK_SIZE + SCALE_SHIFT
    # Initiate offset ranges used in kernel.
    input_offset = tl.arange(0, GROUP_LOAD * GROUP_SIZE) + input_start
    output_offset = tl.arange(0, GROUP_LOAD * (GROUP_SIZE // 8)) + output_start

    # We need to shift output offsets to make space for shared exponent storage.
    # Now create offsets for writing the shared exponent.
    # make sure to add offset from padding
    exp_offset = tl.arange(0, GROUP_LOAD) + exp_start

    # begin binary search
    row_idx = (exp_offset - SCALE_SHIFT) // GROUPS_PER_ROW
    init_offset_exp = (exp_start - SCALE_SHIFT) // GROUPS_PER_ROW

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
        & (exp_offset < (SCALE_CHUNK_SIZE * (pid + 1) + SCALE_SHIFT))
        & (exp_offset < (SCALE_SIZE + SCALE_SHIFT))
    )
    tensor_idx = tl.gather(
        left, tl.where(tensor_idx_guard, row_idx - init_offset_exp, 0), 0
    )

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

        tl.store(
            out + exp_offset,
            scale_.to(tl.uint8, bitcast=True),
            # Prevent writing outside this chunk or the main array.
            mask=(exp_offset < (SCALE_CHUNK_SIZE * (pid + 1) + SCALE_SHIFT))
            & (exp_offset < (SCALE_SIZE + SCALE_SHIFT)),
        )
        # Write out packed values to output tensor.
        ptr_int32 = out.to(tl.pointer_type(tl.int32))
        tl.store(
            ptr_int32 + output_offset,
            packed_result,
            # Prevent writing outside this chunk or the main array.
            mask=(output_offset < OUTPUT_SIZE)
            & (output_offset < (OUTPUT_CHUNK_SIZE * (pid + 1))),
        )

        # Update offsets so we work on the next block.
        input_offset += GROUP_LOAD * GROUP_SIZE

        # since next group might go across the tensor boundary into a new tensor, recalculate offset
        exp_offset += GROUP_LOAD
        row_idx = (exp_offset - SCALE_SHIFT) // GROUPS_PER_ROW

        tensor_idx_guard = (
            (row_idx < M)
            & (exp_offset < (SCALE_CHUNK_SIZE * (pid + 1) + SCALE_SHIFT))
            & (exp_offset < (SCALE_SIZE + SCALE_SHIFT))
        )
        tensor_idx = tl.gather(
            left, tl.where(tensor_idx_guard, row_idx - init_offset_exp, 0), 0
        )
        output_offset += GROUP_LOAD * GROUP_SIZE // 8


def mega_fp4_pack(
    input: torch.Tensor,
    input_global_scale: torch.Tensor,
    group_size: int = 16,
    per_tensor: bool = False,
    m_sizes: Optional[torch.Tensor] = None,
) -> torch.Tensor:
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
    out = torch.empty(
        M * K // 2 + (M * K // group_size), device=device, dtype=torch.uint8
    )

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

    # Check if we need to use int64 for indexing.
    use_int64 = num_threads * groups_per_thread * group_size > 2**31 - 1

    # Invoke triton quantization kernel over rows.
    grid = (num_threads,)

    # Single block handles everything
    if not per_tensor:
        _mega_fp4_pack_kernel[grid](
            input,
            input_global_scale,
            out,
            M=M,
            K=K,
            GROUPS_PER_ROW=groups_per_row,
            GROUPS_PER_THREAD=groups_per_thread,
            ROW_PADDING=padding,
            # pyre-ignore[6]
            GROUP_SIZE=group_size,
            # pyre-ignore[6]
            GROUP_LOAD=GROUP_LOAD,
            # pyre-ignore[6]
            USE_INT64=use_int64,
        )
    else:
        assert m_sizes is not None, "m_sizes must be provided if per_tensor is true."
        num_segments = input_global_scale.shape[0]
        max_row_per_thread = math.ceil(groups_per_thread / groups_per_row)
        search_size = max_row_per_thread + 3
        search_padded_power = triton.next_power_of_2(search_size)
        _mega_fp4_pack_kernel_per_tensor[grid](
            m_sizes,
            search_size,
            search_padded_power,
            input,
            input_global_scale,
            out,
            M=M,
            K=K,
            GROUPS_PER_ROW=groups_per_row,
            GROUPS_PER_THREAD=groups_per_thread,
            num_segments=num_segments,
            prefix_num=triton.next_power_of_2(num_segments),
            ROW_PADDING=padding,
            # pyre-ignore[6]
            GROUP_SIZE=group_size,
            # pyre-ignore[6]
            GROUP_LOAD=GROUP_LOAD,
            # pyre-ignore[6]
            USE_INT64=use_int64,
        )
    return out.view(list(orig_shape[:-1]) + [-1])


@triton.jit
def _mega_fp4_unpack_kernel(
    m_sizes_ptr,  # [num_segments] input sizes
    starting_row_after_padding_ptr,  # [num_segments + 1] output: padded cumsum
    search_size,
    search_padded_power: tl.constexpr,
    A,
    out,
    scale,
    num_segments,
    prefix_num: tl.constexpr,
    M,
    K,
    GROUPS_PER_ROW,
    GROUPS_PER_THREAD,
    ROW_PADDING,
    GROUP_SIZE: tl.constexpr,
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

    # For very large inputs, we need to use int64 indexes. This is slower but necessary.
    if USE_INT64:
        pid = pid.to(tl.int64)
        M = tl.cast(M, tl.int64)
        K = tl.cast(K, tl.int64)
        GROUPS_PER_THREAD = tl.cast(GROUPS_PER_THREAD, tl.int64)

    # Boundaries for writing to output tensor.
    # need to have an adjusted groups per row so that the search and permutation
    # uses the size of the original input rather than the packed input
    OUTPUT_CHUNK_SIZE = GROUPS_PER_THREAD * GROUP_SIZE
    SCALE_CHUNK_SIZE = GROUPS_PER_THREAD
    OUTPUT_SIZE = GROUP_SIZE * M * GROUPS_PER_ROW
    SCALE_SIZE = OUTPUT_SIZE * 2 // GROUP_SIZE
    ADJUSTED_GROUPS_PER_ROW = GROUPS_PER_ROW * 2

    # Find starting offsets for this thread. These are calculated before adjusting for padding.
    input_start = pid * (GROUPS_PER_THREAD * GROUP_SIZE)
    output_start = pid * OUTPUT_CHUNK_SIZE
    exp_start = pid * SCALE_CHUNK_SIZE
    # Initiate offset ranges used in kernel.
    input_offset = tl.arange(0, GROUP_LOAD * GROUP_SIZE) + input_start
    output_offset = tl.arange(0, GROUP_LOAD * GROUP_SIZE) + output_start

    # We need to shift output offsets to make space for shared exponent storage.
    # Now create offsets for writing the shared exponent.
    # make sure to add offset from padding
    # shift to account for start of exp in input
    exp_offset = tl.arange(0, GROUP_LOAD) + exp_start

    # when calculating row idx, need to account for the fact that the original input has length 2 * K,
    # which means it would have TWICE the number of groups_per_row
    # THESE CALCULATIONS ARE DONE WITH RESPECT TO THE INPUT SIZE BEFORE PACKING, WHICH IS TWICE
    row_idx = exp_offset // ADJUSTED_GROUPS_PER_ROW

    init_offset_exp = exp_start // ADJUSTED_GROUPS_PER_ROW

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
        * ADJUSTED_GROUPS_PER_ROW
    )

    inner_idx = (
        row_idx - tl.gather(cumsum, tl.where(tensor_idx_guard, tensor_idx, 0), 0)
    ) * ADJUSTED_GROUPS_PER_ROW

    actual_scale_offset = (
        tensor_offset + inner_idx + exp_offset % ADJUSTED_GROUPS_PER_ROW
    )

    # Load and process blocks of values for this chunk.
    for _k in range(0, tl.cdiv(GROUPS_PER_THREAD, GROUP_LOAD)):
        # Load a block of values.
        scaled_a = tl.load(
            A + input_offset,
            # Mask values out of range for both the main array and this chunk. Also pad if needed.
            mask=(input_offset < (M * K))
            & (input_offset < ((pid + 1) * GROUPS_PER_THREAD * GROUP_SIZE)),
            other=0,
        )

        # load the scales corresponding to scaled_a
        scale_ = tl.load(
            A + OUTPUT_SIZE + exp_offset,
            mask=(row_idx < M)
            & (exp_offset < (SCALE_CHUNK_SIZE * (pid + 1)))
            & (exp_offset < SCALE_SIZE),
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
            scale_,
            # Prevent writing outside this chunk or the main array.
            mask=(row_idx < M)
            & (exp_offset < (SCALE_CHUNK_SIZE * (pid + 1)))
            & (exp_offset < SCALE_SIZE),
        )

        # Write out packed values to output tensor.
        tl.store(
            out + output_offset,
            scaled_a,
            # Prevent writing outside this chunk or the main array.
            mask=(output_offset < OUTPUT_SIZE)
            & (output_offset < (OUTPUT_CHUNK_SIZE * (pid + 1))),
        )

        # Update offsets so we work on the next block.
        input_offset += GROUP_LOAD * GROUP_SIZE

        # since next group might go across the tensor boundary into a new tensor, recalculate offset
        exp_offset += GROUP_LOAD
        row_idx = exp_offset // ADJUSTED_GROUPS_PER_ROW

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
            * ADJUSTED_GROUPS_PER_ROW
        )

        inner_idx = (
            row_idx - tl.gather(cumsum, tl.where(tensor_idx_guard, tensor_idx, 0), 0)
        ) * ADJUSTED_GROUPS_PER_ROW

        actual_scale_offset = (
            tensor_offset + inner_idx + exp_offset % ADJUSTED_GROUPS_PER_ROW
        )

        output_offset += GROUP_LOAD * GROUP_SIZE


def mega_fp4_unpack(
    m_sizes: torch.Tensor,
    input: torch.Tensor,
    group_size: int = 16,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    orig_shape = input.shape
    assert input.ndim >= 1, f"input.ndim needs to be >= 1, but got {input.ndim}."
    other_dims = 1 if input.ndim == 1 else -1
    input = input.reshape(other_dims, input.shape[-1])
    M, original_K = input.shape
    device = input.device

    assert input.dtype in (torch.uint8,), (
        f"input.dtype needs to be fp16 or bf16 but got {input.dtype}."
    )

    # returns the size of the original k dimension before packing
    def decompose_K(k: int):
        assert (k * group_size) % (group_size // 2 + 1) == 0
        return (k * group_size) // (group_size // 2 + 1)

    K = decompose_K(original_K) // 2
    # Two fp4 values will be packed into an uint8.
    out = torch.empty((M, K), device=device, dtype=torch.uint8)

    # We use the rounded values to store the swizzled values. Due to the
    # requirement of the Tensor Core, the minimum tile is 128x4 for the scales.
    # So, we first pad the scales to multiples of 128 and 4. Then, the scales
    # (in float8_e4m3fn) int8. More:
    # https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-mma-scale-factor-b-layout-4x
    def round_up(x: int, y: int) -> int:
        return (x + y - 1) // y * y

    rounded_M = round_up(M + m_sizes.shape[0] * 128, 128)
    rounded_K = original_K - K
    scale = torch.empty((rounded_M, rounded_K), device=device, dtype=torch.uint8)

    # In this kernel, we want each row to be divisible by group_size.
    # If the rows are not, then we will pad them. Find the number of
    # groups per row after padding.
    groups_per_row = math.ceil(K / group_size)
    num_groups = M * groups_per_row
    # Find how many groups each thread should process. We do this
    # by assuming that it is good to distribute work evenly over threads.
    num_threads = min(1024, math.ceil(math.sqrt(input.numel() - (original_K - K) * M)))
    # Data is loaded in chunks of GROUP_LOAD elements, so theres no reason
    # to ever fewer groups per thread than it.
    GROUP_LOAD = 256
    groups_per_thread = max(2 * math.ceil(num_groups / num_threads), GROUP_LOAD)
    # Determine how much padding, if any is needed for each row.
    if K % group_size != 0:
        padding = group_size - (K % group_size)
    else:
        padding = 0

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
    _mega_fp4_unpack_kernel[grid](
        m_sizes,
        starting_row_after_padding,
        search_size,
        search_padded_power,
        input,
        out,
        scale,
        num_segments=num_segments,
        prefix_num=triton.next_power_of_2(num_segments),
        M=M,
        K=K,
        GROUPS_PER_ROW=groups_per_row,
        GROUPS_PER_THREAD=groups_per_thread,
        ROW_PADDING=padding,
        # pyre-ignore[6]
        GROUP_SIZE=group_size,
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
