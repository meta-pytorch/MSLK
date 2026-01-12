# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import math
from enum import IntEnum
from typing import Optional, Tuple, Union

import torch
import triton  # @manual
from triton import language as tl

try:
    from triton.language.extra.libdevice import rsqrt as tl_rsqrt
except ImportError:
    try:
        from triton.language.extra.cuda.libdevice import rsqrt as tl_rsqrt
    except ImportError:
        from triton.language.math import rsqrt as tl_rsqrt

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


@triton.jit
def _kernel_quantize_mx4_unpack(
    A,
    out,
    scale,
    rand_bits,
    M,
    K,
    GROUPS_PER_ROW,
    GROUPS_PER_THREAD,
    ROW_PADDING,
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
    """Quantize a 1D float tensor into a packed MX4 tensor.

    Args:
        A (Tensor): [M] float tensor to be quantized.
        out (Tensor): [M / 2] output containing packed mx4 values.
        scale (Tensor): [M / GROUP_SIZE] containing mx4 shared exponents.
        rand_bits (Optional Tensor): [M, K / 2] random integers used for stochastic rounding.
        M (int): Number of input rows.
        K (int): Number of input columns.
        GROUPS_PER_ROW (int): Number of groups in each row of the input.
        GROUPS_PER_THREAD (int): Number of groups to process per thread.
        ROW_PADDING (int): Number of elements of padding to insert into each row.
        GROUP_SIZE (int): Size of chunks that use the same shared exponent.
        EBITS (int): Number of exponent bits in target mx4 format.
        MBITS (int): Number of mantissa bits in target mx4 format.
        ROUNDING_MODE (int): Which rounding method to use when calculating shared exponent.
        STOCHASTIC_CASTING (bool): Whether to use stochastic rounding when downcasting.
        FP4_EXP_BIAS (int): Exponent bias of target mx4 format.
        GROUP_LOAD (int): Number of groups to process simultaneously.
        USE_INT64 (bool): Whether to use int64 for indexing. This is needed for large tensors.
    """
    # Define Constant Expressions.
    FP16_EXP_BIAS: tl.constexpr = 127  # type: ignore[Incompatible variable type]
    BF16_MIN_NORMAL: tl.constexpr = 2 ** (-126)  # type: ignore[Incompatible variable type]

    # Get the current thread number.
    pid = tl.program_id(0)
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
    exp_offset = tl.arange(0, GROUP_LOAD) + exp_start
    # We need to shift output offsets to make space for shared exponent storage.
    # Now create offsets for writing the shared exponent.
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
        # When theres no padding we can simplify indexing.
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

        # Scaling step
        ##############

        # View the block in terms of groups.
        a_groups = tl.reshape(a, [GROUP_LOAD, GROUP_SIZE]).to(tl.float32)
        # Compute the shared exponent of each group.
        group_max = tl.max(tl.abs(a_groups), axis=1)
        # Prevent infinite values in log.
        group_max = tl.where(group_max == 0, BF16_MIN_NORMAL, group_max)
        # Load relevant random values if doing stochastic rounding
        # or stochastic casting.
        group_rand_bits = None
        # Compute shared exponent using specified rounding mode.
        group_exp = _compute_exp(group_max, ROUNDING_MODE, group_rand_bits, MBITS)
        # Subtract largest exponent in target datatype and remove bias.
        group_exp = group_exp - EBITS
        # Make sure exponent is in valid range.
        group_exp = tl.clamp(group_exp, -127, 125)

        # Next we scale A in preparation for quantization.
        # TODO: We convert to float16 rather than bf16 due to numerical accuracy, but we might need to consider fp32
        scale_ = tl.exp2(group_exp.to(tl.float64)).to(tl.float32)
        # Apply scale_ to input. We do this by broadcasting scale.
        scaled_a = tl.reshape(a, [GROUP_LOAD, GROUP_SIZE]) / tl.reshape(
            scale_, [GROUP_LOAD, 1]
        )
        # Reshape back to a flat array.
        scaled_a = tl.reshape(scaled_a, [GROUP_LOAD * GROUP_SIZE])

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
        first_dim = exp_offset // (512 * n_col_blocks)
        second_dim = (exp_offset % (512 * n_col_blocks)) // (128 * n_col_blocks)
        third_dim = (exp_offset % (128 * n_col_blocks)) // (4 * n_col_blocks)
        fourth_dim = (exp_offset % (4 * n_col_blocks)) // 4
        fifth_dim = exp_offset % 4
        actual_offset = (
            first_dim * (512 * n_col_blocks)
            + fourth_dim * (512)
            + third_dim * (16)
            + second_dim * (4)
            + fifth_dim
        )
        # We're done with group_exp now so we can write it out.
        tl.store(
            scale + actual_offset,
            (group_exp + FP16_EXP_BIAS).to(tl.int8),
            # Prevent writing outside this chunk or the main array.
            mask=(exp_offset < SCALE_SIZE)
            & (exp_offset < (SCALE_CHUNK_SIZE * (pid + 1))),
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
        exp_offset += GROUP_LOAD
        output_offset += GROUP_LOAD * GROUP_SIZE // 8


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


def triton_quantize_mx4_unpack(
    input: torch.Tensor,
    group_size: int = 32,
    ebits: int = 2,
    mbits: int = 1,
    rounding_mode: Union[RoundingMode, int] = RoundingMode.ceil,
    stochastic_casting: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize a tensor to mx4 format using efficient triton kernels.

    Args:
        a (Tensor): [M] higher precision input tensor.
        group_size (int): Size of chunks that will use the same shared exponent.
        ebits (int): Number of bits to use for exponent in target mx4 format.
        mbits (int): Number of bits to use for mantissa in target mx4 format.
        rounding_mode (Union[RoundingMode, int]): Which type of rounding to use
        when calculating shared exponent. Defaults to pre-rounding to nearest even int.
        stochastic_casting (bool): Whether to use stochastic casting.

    Returns:
        torch.Tensor: [M / 2] mx4 scaled tensor packed into in8
        torch.Tensor: [M / group_size] mx4 shared exponents into int8

        eg.
        Input with shape [1, 8192] will be quantized to [1, 4096 + 512] as
        each value contain two elements packed into an int8 and
        there are 32 elements per group.
    """

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

    rounded_M = round_up(M, 128)
    scale_K = K // block_size
    rounded_K = round_up(scale_K, 4)
    scale = torch.empty((rounded_M, rounded_K), device=device, dtype=torch.int8)

    # In this kernel, we want each row to be divisible by group_size.
    # If the rows are not, then we will pad them. Find the number of
    # groups per row after padding.
    groups_per_row = math.ceil(K / group_size)
    num_groups = M * groups_per_row
    # Find how many groups each thread should process. We do this
    # by assuming that it is good to distribute work evenly over threads.
    num_threads = math.ceil(math.sqrt(input.numel()))
    # Data is loaded in chunks of GROUP_LOAD elements, so theres no reason
    # to ever fewer groups per thread than it.
    GROUP_LOAD = 64
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
    # Invoke triton quantization kernel over rows.

    grid = (num_threads,)
    _kernel_quantize_mx4_unpack[grid](
        input,
        out,
        scale,
        rand_bits=rand_bits,
        M=M,
        K=K,
        GROUPS_PER_ROW=groups_per_row,
        GROUPS_PER_THREAD=groups_per_thread,
        ROW_PADDING=padding,
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
    return out.view(list(orig_shape[:-1]) + [-1]).view(torch.uint8), scale


@triton.jit
def _kernel_silu_quantize_mx4_unpack(
    A,
    B,
    out,
    scale,
    rand_bits,
    M,
    K,
    GROUPS_PER_ROW,
    GROUPS_PER_THREAD,
    ROW_PADDING,
    GROUP_SIZE: tl.constexpr,
    EBITS: tl.constexpr,
    MBITS: tl.constexpr,
    ROUNDING_MODE: tl.constexpr,
    STOCHASTIC_CASTING: tl.constexpr,
    FP4_EXP_BIAS: tl.constexpr,
    GROUP_LOAD: tl.constexpr,
    USE_INT64: tl.constexpr,
) -> None:
    """Quantize a 1D float tensor into a packed MX4 tensor.

    Args:
        A (Tensor): [M] float tensor to be quantized.
        out (Tensor): [M / 2] output containing packed mx4 values.
        scale (Tensor): [M / GROUP_SIZE] containing mx4 shared exponents.
        rand_bits (Optional Tensor): [M, K / 2] random integers used for stochastic rounding.
        M (int): Number of input rows.
        K (int): Number of input columns.
        GROUPS_PER_ROW (int): Number of groups in each row of the input.
        GROUPS_PER_THREAD (int): Number of groups to process per thread.
        ROW_PADDING (int): Number of elements of padding to insert into each row.
        GROUP_SIZE (int): Size of chunks that use the same shared exponent.
        EBITS (int): Number of exponent bits in target mx4 format.
        MBITS (int): Number of mantissa bits in target mx4 format.
        ROUNDING_MODE (int): Which rounding method to use when calculating shared exponent.
        STOCHASTIC_CASTING (bool): Whether to use stochastic rounding when downcasting.
        FP4_EXP_BIAS (int): Exponent bias of target mx4 format.
        GROUP_LOAD (int): Number of groups to process simultaneously.
        USE_INT64 (bool): Whether to use int64 for indexing. This is needed for large tensors.
    """
    # Define Constant Expressions.
    FP16_EXP_MASK: tl.constexpr = 0x7F80  # type: ignore[Incompatible variable type]
    FP16_EXP_OFFSET: tl.constexpr = 7  # type: ignore[Incompatible variable type]
    FP16_EXP_BIAS: tl.constexpr = 127  # type: ignore[Incompatible variable type]
    FP16_SIGN_OFFSET: tl.constexpr = 15  # type: ignore[Incompatible variable type]
    SIGN_MASK: tl.constexpr = 0x1  # type: ignore[Incompatible variable type]
    FP16_MANTISSA_MASK: tl.constexpr = 0x007F  # type: ignore[Incompatible variable type]
    # FP4 has 2 mantissa bits, one explicit one implicit.
    MBITS_IMPLICIT: tl.constexpr = MBITS + 1  # type: ignore[Incompatible variable type]
    MAX_FP16_MANTISSA_BITS: tl.constexpr = 8  # type: ignore[Incompatible variable type]
    IMPLIED_1_BIT: tl.constexpr = 1 << 7  # type: ignore[Incompatible variable type]
    BF16_MIN_NORMAL: tl.constexpr = 2 ** (-126)  # type: ignore[Incompatible variable type]
    MANTISSA_OVERFLOW_THRESHOLD: tl.constexpr = (1 << MBITS_IMPLICIT) - 1  # type: ignore[Incompatible variable type]
    EXPONENT_OVERFLOW_THRESHOLD: tl.constexpr = (1 << EBITS) - 1  # type: ignore[Incompatible variable type]
    IMPLICIT_1_MASK = (1 << (MBITS_IMPLICIT - 1)) - 1
    RAND_MASK: tl.constexpr = (1 << (FP16_EXP_OFFSET - MBITS)) - 1  # type: ignore[Incompatible variable type]

    # Get the current thread number.
    pid = tl.program_id(0)
    # For very large inputs, we need to use int64 indexes. This is slower but necessary.
    if USE_INT64:
        pid = pid.to(tl.int64)
        M = tl.cast(M, tl.int64)
        K = tl.cast(K, tl.int64)
        GROUPS_PER_THREAD = tl.cast(GROUPS_PER_THREAD, tl.int64)

    # Boundaries for writing to output tensor.
    NUM_GROUPS = M * GROUPS_PER_ROW
    OUTPUT_CHUNK_SIZE = (GROUPS_PER_THREAD * GROUP_SIZE) // 2
    SCALE_CHUNK_SIZE = GROUPS_PER_THREAD
    OUTPUT_SIZE = (GROUP_SIZE * NUM_GROUPS) // 2
    SCALE_SIZE = NUM_GROUPS

    # Find starting offsets for this thread. These are calculated before adjusting for padding.
    input_start = pid * (GROUPS_PER_THREAD * GROUP_SIZE)
    output_start = pid * OUTPUT_CHUNK_SIZE
    exp_start = pid * SCALE_CHUNK_SIZE
    # Initiate offset ranges used in kernel.
    input_offset = tl.arange(0, GROUP_LOAD * GROUP_SIZE) + input_start
    output_offset = tl.arange(0, GROUP_LOAD * (GROUP_SIZE // 2)) + output_start
    # Stochastic rounding loads chunks of random values.
    if ROUNDING_MODE == 3:
        rand_bits_offset = tl.arange(0, GROUP_LOAD) + pid * GROUPS_PER_THREAD
    # Ceil rounding uses single values as a seed.
    else:
        rand_bits_offset = pid * GROUPS_PER_THREAD
    # We need to shift output offsets to make space for shared exponent storage.
    # Now create offsets for writing the shared exponent.
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
        # When theres no padding we can simplify indexing.
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
        b = tl.load(
            B + padded_input_offset,
            # Mask values out of range for both the main array and this chunk. Also pad if needed.
            mask=(padded_input_offset < (M * K))
            & (padded_input_offset < ((pid + 1) * GROUPS_PER_THREAD * GROUP_SIZE))
            & pad_mask,
            other=0,
        )

        # compute silu
        a_upcast = a.to(tl.float32)
        b_upcast = b.to(tl.float32)
        a = (a_upcast * tl.sigmoid(a_upcast) * b_upcast).to(tl.bfloat16)

        # Scaling step
        ##############

        # View the block in terms of groups.
        a_groups = tl.reshape(a, [GROUP_LOAD, GROUP_SIZE])
        # Compute the shared exponent of each group.
        group_max = tl.max(tl.abs(a_groups), axis=1)
        # Prevent infinite values in log.
        group_max = tl.where(group_max == 0, BF16_MIN_NORMAL, group_max)
        # Load relevant random values if doing stochastic rounding
        # or stochastic casting.
        group_rand_bits = None
        if (ROUNDING_MODE) == 3 or STOCHASTIC_CASTING:
            group_rand_bits = tl.load(
                rand_bits + rand_bits_offset,
                mask=rand_bits_offset < K // GROUP_SIZE,
                other=0,
            )
            rand_bits_offset += GROUP_LOAD
        # Compute shared exponent using specified rounding mode.
        group_exp = _compute_exp(group_max, ROUNDING_MODE, group_rand_bits, MBITS)
        # Subtract largest exponent in target datatype and remove bias.
        group_exp = group_exp - EBITS
        # Make sure exponent is in valid range.
        group_exp = tl.clamp(group_exp, -127, 125)

        # Next we scale A in preparation for quantization.
        scale_ = tl.exp2(group_exp.to(tl.float64)).to(tl.float16)
        # Apply scale_ to input. We do this by broadcasting scale.
        scaled_a = (
            tl.reshape(a, [GROUP_LOAD, GROUP_SIZE])
            / tl.reshape(scale_, [GROUP_LOAD, 1])
        ).to(tl.bfloat16)
        # Reshape back to a flat array.
        scaled_a = tl.reshape(scaled_a, [GROUP_LOAD * GROUP_SIZE])

        # We're done with group_exp now so we can write it out.
        # We readd fp16_exp_bias for compatibility with cuda dequant.
        tl.store(
            scale + exp_offset,
            (group_exp + FP16_EXP_BIAS).to(tl.int8),
            # Prevent writing outside this chunk or the main array.
            mask=(exp_offset < SCALE_SIZE)
            & (exp_offset < (SCALE_CHUNK_SIZE * (pid + 1))),
        )

        # Quantization step
        ###################

        # During quantization, we're going to be doing a lot of bitwise operations.
        # This is easier to work with in int32.
        scaled_a = scaled_a.to(tl.int16, bitcast=True)

        # When doing stochastic downcasting, generate random values for this block
        # and apply it to the mantissa.
        if STOCHASTIC_CASTING:
            # We're going to generate 4 blocks at once so we only need
            # one fourth of the input offsets.
            # Start by splitting down to half of offsets.
            philox_4x_offset = tl.split(
                tl.reshape(
                    input_offset,
                    [GROUP_LOAD * GROUP_SIZE // 2, 2],
                    can_reorder=True,
                )
            )
            # Split down to fourth.
            philox_4x_offset = tl.split(
                tl.reshape(
                    philox_4x_offset,
                    [GROUP_LOAD * GROUP_SIZE // 4, 2],
                    can_reorder=True,
                )
            )
            # Generate 4 blocks of random bits for this block.
            a_4x, b_4x, c_4x, d_4x = tl.randint4x(
                group_rand_bits, philox_4x_offset, n_rounds=7
            )
            # Combine the 4 blocks into a single chunk of random values.
            # This needs to be done incrementally.
            stochastic_round_bits = tl.join(tl.join(a_4x, b_4x), tl.join(c_4x, d_4x))
            # Flatten back to simple array.
            stochastic_round_bits = tl.reshape(
                stochastic_round_bits, [GROUP_LOAD * GROUP_SIZE]
            ).to(tl.int16, bitcast=True)

            # Mask off mantissa bits of random value and add to mantissa.
            scaled_a = scaled_a + (stochastic_round_bits & RAND_MASK)

        # Extract sign bit of value.
        sign_bit = (scaled_a >> FP16_SIGN_OFFSET) & SIGN_MASK

        # Extract exponent.
        biased_exp = (scaled_a & FP16_EXP_MASK) >> FP16_EXP_OFFSET

        # Extract mantissa.
        trailing_mantissa = scaled_a & FP16_MANTISSA_MASK

        # Adjust exponent bias for FP4.
        new_biased_exp = biased_exp - FP16_EXP_BIAS + FP4_EXP_BIAS

        # Compute difference between ideal exponent and what fp4 can represent.
        exp_diff = tl.where(new_biased_exp <= 0, 1 - new_biased_exp, 0)

        # Clip this difference to maximum number of fp32 mantissa bits.
        exp_diff = tl.minimum(exp_diff, MAX_FP16_MANTISSA_BITS)

        # Now we round our fp32 mantissa down to fp4.
        is_subnorm = biased_exp == 0
        # Add implied 1 bit to normal values.
        mantissa = tl.where(
            is_subnorm, trailing_mantissa, trailing_mantissa + IMPLIED_1_BIT
        )
        # Compute base number of bits corresponding to the mantissa, smaller for subnorms
        # since implied one is included in exp_diff.
        fp16_sig_bits = tl.where(is_subnorm, 7, 8).to(tl.int32)
        # Now we're ready to shift down to target bitwidth (with an extra bit for rounding).
        mantissa = mantissa >> (fp16_sig_bits + exp_diff - MBITS_IMPLICIT - 1)
        # Perform rounding by adding 1 and shifting down.
        mantissa = (mantissa + 1) >> 1

        # Check for overflow and adjust exponent accordingly.
        overflow = mantissa > MANTISSA_OVERFLOW_THRESHOLD
        # Allow subnorms to overflow into normals, otherwise shift away overflow.
        mantissa = tl.where(overflow and (not is_subnorm), mantissa >> 1, mantissa)
        # Special case where a value is subnormal and has a large mantissa, overflow it.
        new_biased_exp = tl.where(
            (new_biased_exp <= 0) and (mantissa == 2), 1, new_biased_exp
        )
        # Remove implicit 1.
        mantissa = mantissa & IMPLICIT_1_MASK
        # Add overflow to exponent.
        new_biased_exp = tl.where(overflow, new_biased_exp + 1, new_biased_exp)
        # If exp overflows, set mantissa to maximum value (equivalent to clamping).
        mantissa = tl.where(new_biased_exp > EXPONENT_OVERFLOW_THRESHOLD, 1, mantissa)

        # Construct FP4 value from components.
        new_biased_exp = tl.maximum(
            tl.minimum(new_biased_exp, EXPONENT_OVERFLOW_THRESHOLD), 0
        )

        mx4_value = (new_biased_exp << (MBITS_IMPLICIT - 1)) | mantissa
        mx4_value = (sign_bit << (EBITS + MBITS)) | mx4_value

        # Extract low and high bits from values.
        low_mx4, high_mx4 = tl.split(
            tl.reshape(mx4_value, [(GROUP_LOAD * GROUP_SIZE) // 2, 2])
        )
        # Shift mx4 values together so they are packed into int8.
        packed_mx4 = ((high_mx4 << 4) | (low_mx4)).to(tl.int8)

        # Write out packed values to output tensor.
        tl.store(
            out + output_offset,
            packed_mx4,
            # Prevent writing outside this chunk or the main array.
            mask=(output_offset < OUTPUT_SIZE)
            & (output_offset < (OUTPUT_CHUNK_SIZE * (pid + 1))),
        )

        # Update offsets so we work on the next block.
        input_offset += GROUP_LOAD * GROUP_SIZE
        exp_offset += GROUP_LOAD
        output_offset += GROUP_LOAD * GROUP_SIZE // 2


def triton_silu_quantize_mx4_unpack(
    a: torch.Tensor,
    b: torch.Tensor,
    group_size: int = 32,
    ebits: int = 2,
    mbits: int = 1,
    rounding_mode: Union[RoundingMode, int] = RoundingMode.ceil,
    stochastic_casting: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize a tensor to mx4 format using efficient triton kernels.

    Args:
        a (Tensor): [M] higher precision input tensor.
        group_size (int): Size of chunks that will use the same shared exponent.
        ebits (int): Number of bits to use for exponent in target mx4 format.
        mbits (int): Number of bits to use for mantissa in target mx4 format.
        rounding_mode (Union[RoundingMode, int]): Which type of rounding to use
        when calculating shared exponent. Defaults to pre-rounding to nearest even int.
        stochastic_casting (bool): Whether to use stochastic casting.

    Returns:
        torch.Tensor: [M / 2] mx4 scaled tensor packed into in8
        torch.Tensor: [M / group_size] mx4 shared exponents into int8

        eg.
        Input with shape [1, 8192] will be quantized to [1, 4096 + 256] as
        each value contain two elements packed into an int8 and
        there are 32 groups in each row.
    """
    # If given an empty shape, return an empty tensor.
    if a.numel() == 0:
        return torch.empty(a.shape, device=a.device, dtype=torch.uint8), torch.empty(
            a.shape, device=a.device, dtype=torch.uint8
        )
    # Make sure input is continuous in memory.
    assert a.is_contiguous(), "Inputs to mx4 quantize must be contiguous in memory."

    orig_shape = a.shape
    # For simplicity, view input as a 2D array.
    a = a.view(-1, a.shape[-1])
    # Extract rows and columns.
    M, K = a.shape
    # In this kernel, we want each row to be divisible by group_size.
    # If the rows are not, then we will pad them. Find the number of
    # groups per row after padding.
    groups_per_row = math.ceil(K / group_size)
    num_groups = M * groups_per_row
    # Find how many groups each thread should process. We do this
    # by assuming that it is good to distribute work evenly over threads.
    num_threads = math.ceil(math.sqrt(a.numel()))
    # Data is loaded in chunks of GROUP_LOAD elements, so theres no reason
    # to ever fewer groups per thread than it.
    GROUP_LOAD = 64
    groups_per_thread = max(math.ceil(num_groups / num_threads), GROUP_LOAD)
    # Determine how much padding, if any is needed for each row.
    if K % group_size != 0:
        padding = group_size - (K % group_size)
    else:
        padding = 0

    # Create output tensor.
    out_elems = (num_groups * group_size) // 2
    scale_elems = num_groups
    out = torch.empty([out_elems], device=a.device, dtype=torch.uint8)
    scale = torch.empty([scale_elems], device=a.device, dtype=torch.uint8)

    # If using stochastic rounding, create random noise for each group.
    # We use the same random bits as seeds when doing stochastic downcasting.
    if rounding_mode == RoundingMode.stochastic or stochastic_casting:
        # Each group will need a seed.
        rand_bits = torch.randint(
            low=0,
            high=2**31 - 1,
            size=(num_groups,),
            dtype=torch.int32,
            device=a.device,
        )
    else:
        rand_bits = None

    # Check if we need to use int64 for indexing.
    use_int64 = num_threads * groups_per_thread * group_size > 2**31 - 1
    # Invoke triton quantization kernel over rows.
    grid = (num_threads,)
    _kernel_silu_quantize_mx4_unpack[grid](
        a,
        b,
        out,
        scale,
        rand_bits=rand_bits,
        M=M,
        K=K,
        GROUPS_PER_ROW=groups_per_row,
        GROUPS_PER_THREAD=groups_per_thread,
        ROW_PADDING=padding,
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
    )
    scale = scale.view(torch.float8_e8m0fnu)
    scale = scale.view(orig_shape[0], -1)
    scale = _to_blocked(scale)

    return out.view(list(orig_shape[:-1]) + [-1]), scale


@triton.jit
def _kernel_rms_quantize_mx4_unpack(
    A,
    B,
    out,
    scale,
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
) -> None:
    """Quantize a 1D float tensor into a packed MX4 tensor.

    Args:
        A (Tensor): [M] float tensor to be quantized.
        out (Tensor): [M / 2] output containing packed mx4 values.
        scale (Tensor): [M / GROUP_SIZE] containing mx4 shared exponents.
        rand_bits (Optional Tensor): [M, K / 2] random integers used for stochastic rounding.
        M (int): Number of input rows.
        K (int): Number of input columns.
        GROUPS_PER_ROW (int): Number of groups in each row of the input.
        GROUPS_PER_THREAD (int): Number of groups to process per thread.
        ROW_PADDING (int): Number of elements of padding to insert into each row.
        GROUP_SIZE (int): Size of chunks that use the same shared exponent.
        EBITS (int): Number of exponent bits in target mx4 format.
        MBITS (int): Number of mantissa bits in target mx4 format.
        ROUNDING_MODE (int): Which rounding method to use when calculating shared exponent.
        STOCHASTIC_CASTING (bool): Whether to use stochastic rounding when downcasting.
        FP4_EXP_BIAS (int): Exponent bias of target mx4 format.
        GROUP_LOAD (int): Number of groups to process simultaneously.
        USE_INT64 (bool): Whether to use int64 for indexing. This is needed for large tensors.
    """
    # Define Constant Expressions.
    FP16_EXP_MASK: tl.constexpr = 0x7F80  # type: ignore[Incompatible variable type]
    FP16_EXP_OFFSET: tl.constexpr = 7  # type: ignore[Incompatible variable type]
    FP16_EXP_BIAS: tl.constexpr = 127  # type: ignore[Incompatible variable type]
    FP16_SIGN_OFFSET: tl.constexpr = 15  # type: ignore[Incompatible variable type]
    SIGN_MASK: tl.constexpr = 0x1  # type: ignore[Incompatible variable type]
    FP16_MANTISSA_MASK: tl.constexpr = 0x007F  # type: ignore[Incompatible variable type]
    # FP4 has 2 mantissa bits, one explicit one implicit.
    MBITS_IMPLICIT: tl.constexpr = MBITS + 1  # type: ignore[Incompatible variable type]
    MAX_FP16_MANTISSA_BITS: tl.constexpr = 8  # type: ignore[Incompatible variable type]
    IMPLIED_1_BIT: tl.constexpr = 1 << 7  # type: ignore[Incompatible variable type]
    BF16_MIN_NORMAL: tl.constexpr = 2 ** (-126)  # type: ignore[Incompatible variable type]
    MANTISSA_OVERFLOW_THRESHOLD: tl.constexpr = (1 << MBITS_IMPLICIT) - 1  # type: ignore[Incompatible variable type]
    EXPONENT_OVERFLOW_THRESHOLD: tl.constexpr = (1 << EBITS) - 1  # type: ignore[Incompatible variable type]
    IMPLICIT_1_MASK = (1 << (MBITS_IMPLICIT - 1)) - 1
    RAND_MASK: tl.constexpr = (1 << (FP16_EXP_OFFSET - MBITS)) - 1  # type: ignore[Incompatible variable type]

    # Get the current thread number.
    pid = tl.program_id(0)
    # For very large inputs, we need to use int64 indexes. This is slower but necessary.
    if USE_INT64:
        pid = pid.to(tl.int64)
        M = tl.cast(M, tl.int64)
        K = tl.cast(K, tl.int64)
        GROUPS_PER_THREAD = tl.cast(GROUPS_PER_THREAD, tl.int64)

    # Boundaries for writing to output tensor.
    NUM_GROUPS = M * GROUPS_PER_ROW
    OUTPUT_CHUNK_SIZE = (GROUPS_PER_THREAD * GROUP_SIZE) // 2
    SCALE_CHUNK_SIZE = GROUPS_PER_THREAD
    OUTPUT_SIZE = (GROUP_SIZE * NUM_GROUPS) // 2
    SCALE_SIZE = NUM_GROUPS

    # Find starting offsets for this thread. These are calculated before adjusting for padding.
    input_start = pid * (GROUPS_PER_THREAD * GROUP_SIZE)
    output_start = pid * OUTPUT_CHUNK_SIZE
    exp_start = pid * SCALE_CHUNK_SIZE
    # Initiate offset ranges used in kernel.
    input_offset = tl.arange(0, GROUP_LOAD * GROUP_SIZE) + input_start
    output_offset = tl.arange(0, GROUP_LOAD * (GROUP_SIZE // 2)) + output_start
    # Stochastic rounding loads chunks of random values.
    if ROUNDING_MODE == 3:
        rand_bits_offset = tl.arange(0, GROUP_LOAD) + pid * GROUPS_PER_THREAD
    # Ceil rounding uses single values as a seed.
    else:
        rand_bits_offset = pid * GROUPS_PER_THREAD
    # We need to shift output offsets to make space for shared exponent storage.
    # Now create offsets for writing the shared exponent.
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
        # When theres no padding we can simplify indexing.
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
        b = tl.load(
            B + padded_input_offset,
            # Mask values out of range for both the main array and this chunk. Also pad if needed.
            mask=(padded_input_offset < (M * K))
            & (padded_input_offset < ((pid + 1) * GROUPS_PER_THREAD * GROUP_SIZE))
            & pad_mask,
            other=0,
        )

        # compute rms

        a_segment = a.reshape(GROUP_LOAD, GROUP_SIZE).to(tl.float32)
        group_inv = tl_rsqrt(tl.sum(a_segment * a_segment, axis=1) / GROUP_SIZE + EPS)
        a = (
            (
                a_segment
                * group_inv.expand_dims(axis=1)
                * b.reshape(GROUP_LOAD, GROUP_SIZE).to(tl.float32)
            )
            .to(tl.bfloat16)
            .reshape(GROUP_LOAD * GROUP_SIZE)
        )

        # Scaling step
        ##############

        # View the block in terms of groups.
        a_groups = tl.reshape(a, [GROUP_LOAD, GROUP_SIZE])
        # Compute the shared exponent of each group.
        group_max = tl.max(tl.abs(a_groups), axis=1)
        # Prevent infinite values in log.
        group_max = tl.where(group_max == 0, BF16_MIN_NORMAL, group_max)
        # Load relevant random values if doing stochastic rounding
        # or stochastic casting.
        group_rand_bits = None
        if (ROUNDING_MODE) == 3 or STOCHASTIC_CASTING:
            group_rand_bits = tl.load(
                rand_bits + rand_bits_offset,
                mask=rand_bits_offset < K // GROUP_SIZE,
                other=0,
            )
            rand_bits_offset += GROUP_LOAD
        # Compute shared exponent using specified rounding mode.
        group_exp = _compute_exp(group_max, ROUNDING_MODE, group_rand_bits, MBITS)
        # Subtract largest exponent in target datatype and remove bias.
        group_exp = group_exp - EBITS
        # Make sure exponent is in valid range.
        group_exp = tl.clamp(group_exp, -127, 125)

        # Next we scale A in preparation for quantization.
        scale_ = tl.exp2(group_exp.to(tl.float64)).to(tl.float16)
        # Apply scale_ to input. We do this by broadcasting scale.
        scaled_a = (
            tl.reshape(a, [GROUP_LOAD, GROUP_SIZE])
            / tl.reshape(scale_, [GROUP_LOAD, 1])
        ).to(tl.bfloat16)
        # Reshape back to a flat array.
        scaled_a = tl.reshape(scaled_a, [GROUP_LOAD * GROUP_SIZE])

        # We're done with group_exp now so we can write it out.
        # We readd fp16_exp_bias for compatibility with cuda dequant.
        tl.store(
            scale + exp_offset,
            (group_exp + FP16_EXP_BIAS).to(tl.int8),
            # Prevent writing outside this chunk or the main array.
            mask=(exp_offset < SCALE_SIZE)
            & (exp_offset < (SCALE_CHUNK_SIZE * (pid + 1))),
        )

        # Quantization step
        ###################

        # During quantization, we're going to be doing a lot of bitwise operations.
        # This is easier to work with in int32.
        scaled_a = scaled_a.to(tl.int16, bitcast=True)

        # When doing stochastic downcasting, generate random values for this block
        # and apply it to the mantissa.
        if STOCHASTIC_CASTING:
            # We're going to generate 4 blocks at once so we only need
            # one fourth of the input offsets.
            # Start by splitting down to half of offsets.
            philox_4x_offset = tl.split(
                tl.reshape(
                    input_offset,
                    [GROUP_LOAD * GROUP_SIZE // 2, 2],
                    can_reorder=True,
                )
            )
            # Split down to fourth.
            philox_4x_offset = tl.split(
                tl.reshape(
                    philox_4x_offset,
                    [GROUP_LOAD * GROUP_SIZE // 4, 2],
                    can_reorder=True,
                )
            )
            # Generate 4 blocks of random bits for this block.
            a_4x, b_4x, c_4x, d_4x = tl.randint4x(
                group_rand_bits, philox_4x_offset, n_rounds=7
            )
            # Combine the 4 blocks into a single chunk of random values.
            # This needs to be done incrementally.
            stochastic_round_bits = tl.join(tl.join(a_4x, b_4x), tl.join(c_4x, d_4x))
            # Flatten back to simple array.
            stochastic_round_bits = tl.reshape(
                stochastic_round_bits, [GROUP_LOAD * GROUP_SIZE]
            ).to(tl.int16, bitcast=True)

            # Mask off mantissa bits of random value and add to mantissa.
            scaled_a = scaled_a + (stochastic_round_bits & RAND_MASK)

        # Extract sign bit of value.
        sign_bit = (scaled_a >> FP16_SIGN_OFFSET) & SIGN_MASK

        # Extract exponent.
        biased_exp = (scaled_a & FP16_EXP_MASK) >> FP16_EXP_OFFSET

        # Extract mantissa.
        trailing_mantissa = scaled_a & FP16_MANTISSA_MASK

        # Adjust exponent bias for FP4.
        new_biased_exp = biased_exp - FP16_EXP_BIAS + FP4_EXP_BIAS

        # Compute difference between ideal exponent and what fp4 can represent.
        exp_diff = tl.where(new_biased_exp <= 0, 1 - new_biased_exp, 0)

        # Clip this difference to maximum number of fp32 mantissa bits.
        exp_diff = tl.minimum(exp_diff, MAX_FP16_MANTISSA_BITS)

        # Now we round our fp32 mantissa down to fp4.
        is_subnorm = biased_exp == 0
        # Add implied 1 bit to normal values.
        mantissa = tl.where(
            is_subnorm, trailing_mantissa, trailing_mantissa + IMPLIED_1_BIT
        )
        # Compute base number of bits corresponding to the mantissa, smaller for subnorms
        # since implied one is included in exp_diff.
        fp16_sig_bits = tl.where(is_subnorm, 7, 8).to(tl.int32)
        # Now we're ready to shift down to target bitwidth (with an extra bit for rounding).
        mantissa = mantissa >> (fp16_sig_bits + exp_diff - MBITS_IMPLICIT - 1)
        # Perform rounding by adding 1 and shifting down.
        mantissa = (mantissa + 1) >> 1

        # Check for overflow and adjust exponent accordingly.
        overflow = mantissa > MANTISSA_OVERFLOW_THRESHOLD
        # Allow subnorms to overflow into normals, otherwise shift away overflow.
        mantissa = tl.where(overflow and (not is_subnorm), mantissa >> 1, mantissa)
        # Special case where a value is subnormal and has a large mantissa, overflow it.
        new_biased_exp = tl.where(
            (new_biased_exp <= 0) and (mantissa == 2), 1, new_biased_exp
        )
        # Remove implicit 1.
        mantissa = mantissa & IMPLICIT_1_MASK
        # Add overflow to exponent.
        new_biased_exp = tl.where(overflow, new_biased_exp + 1, new_biased_exp)
        # If exp overflows, set mantissa to maximum value (equivalent to clamping).
        mantissa = tl.where(new_biased_exp > EXPONENT_OVERFLOW_THRESHOLD, 1, mantissa)

        # Construct FP4 value from components.
        new_biased_exp = tl.maximum(
            tl.minimum(new_biased_exp, EXPONENT_OVERFLOW_THRESHOLD), 0
        )

        mx4_value = (new_biased_exp << (MBITS_IMPLICIT - 1)) | mantissa
        mx4_value = (sign_bit << (EBITS + MBITS)) | mx4_value

        # Extract low and high bits from values.
        low_mx4, high_mx4 = tl.split(
            tl.reshape(mx4_value, [(GROUP_LOAD * GROUP_SIZE) // 2, 2])
        )
        # Shift mx4 values together so they are packed into int8.
        packed_mx4 = ((high_mx4 << 4) | (low_mx4)).to(tl.int8)

        # Write out packed values to output tensor.
        tl.store(
            out + output_offset,
            packed_mx4,
            # Prevent writing outside this chunk or the main array.
            mask=(output_offset < OUTPUT_SIZE)
            & (output_offset < (OUTPUT_CHUNK_SIZE * (pid + 1))),
        )

        # Update offsets so we work on the next block.
        input_offset += GROUP_LOAD * GROUP_SIZE
        exp_offset += GROUP_LOAD
        output_offset += GROUP_LOAD * GROUP_SIZE // 2


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


def triton_rms_quantize_mx4_unpack(
    a: torch.Tensor,
    b: torch.Tensor,
    group_size: int = 32,
    ebits: int = 2,
    mbits: int = 1,
    rounding_mode: Union[RoundingMode, int] = RoundingMode.ceil,
    stochastic_casting: bool = False,
    EPS: float = 1e-5,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize a tensor to mx4 format using efficient triton kernels.

    Args:
        a (Tensor): [M] higher precision input tensor.
        group_size (int): Size of chunks that will use the same shared exponent.
        ebits (int): Number of bits to use for exponent in target mx4 format.
        mbits (int): Number of bits to use for mantissa in target mx4 format.
        rounding_mode (Union[RoundingMode, int]): Which type of rounding to use
        when calculating shared exponent. Defaults to pre-rounding to nearest even int.
        stochastic_casting (bool): Whether to use stochastic casting.

    Returns:
        torch.Tensor: [M / 2] mx4 scaled tensor packed into in8
        torch.Tensor: [M / group_size] mx4 shared exponents into int8

        eg.
        Input with shape [1, 8192] will be quantized to [1, 4096 + 256] as
        each value contain two elements packed into an int8 and
        there are 32 groups in each row.
    """
    # If given an empty shape, return an empty tensor.
    if a.numel() == 0:
        return torch.empty(a.shape, device=a.device, dtype=torch.uint8), torch.empty(
            a.shape, device=a.device, dtype=torch.uint8
        )
    # Make sure input is continuous in memory.
    assert a.is_contiguous(), "Inputs to mx4 quantize must be contiguous in memory."

    orig_shape = a.shape
    # For simplicity, view input as a 2D array.
    a = a.view(-1, a.shape[-1])
    # Extract rows and columns.
    M, K = a.shape
    # In this kernel, we want each row to be divisible by group_size.
    # If the rows are not, then we will pad them. Find the number of
    # groups per row after padding.
    groups_per_row = math.ceil(K / group_size)
    num_groups = M * groups_per_row
    # Find how many groups each thread should process. We do this
    # by assuming that it is good to distribute work evenly over threads.
    num_threads = math.ceil(math.sqrt(a.numel()))
    # Data is loaded in chunks of GROUP_LOAD elements, so theres no reason
    # to ever fewer groups per thread than it.
    GROUP_LOAD = 64
    groups_per_thread = max(math.ceil(num_groups / num_threads), GROUP_LOAD)
    # Determine how much padding, if any is needed for each row.
    if K % group_size != 0:
        padding = group_size - (K % group_size)
    else:
        padding = 0

    # Create output tensor.
    out_elems = (num_groups * group_size) // 2
    scale_elems = num_groups
    out = torch.empty([out_elems], device=a.device, dtype=torch.uint8)
    scale = torch.empty([scale_elems], device=a.device, dtype=torch.uint8)

    # If using stochastic rounding, create random noise for each group.
    # We use the same random bits as seeds when doing stochastic downcasting.
    if rounding_mode == RoundingMode.stochastic or stochastic_casting:
        # Each group will need a seed.
        rand_bits = torch.randint(
            low=0,
            high=2**31 - 1,
            size=(num_groups,),
            dtype=torch.int32,
            device=a.device,
        )
    else:
        rand_bits = None

    # Check if we need to use int64 for indexing.
    use_int64 = num_threads * groups_per_thread * group_size > 2**31 - 1
    # Invoke triton quantization kernel over rows.
    grid = (num_threads,)
    _kernel_rms_quantize_mx4_unpack[grid](
        a,
        b,
        out,
        scale,
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
    )
    scale = scale.view(torch.float8_e8m0fnu)
    scale = scale.view(orig_shape[0], -1)
    scale = _to_blocked(scale)

    return out.view(list(orig_shape[:-1]) + [-1]), scale


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


# DO NOT USE, WILL BE REMOVED SOON!!!
@triton.jit
def _kernel_nvfp4_quantize(
    A,
    input_global_scale_tensor,
    out,
    scale,
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
    USE_E8M0_SCALE: tl.constexpr,
) -> None:
    """Quantize a 1D float tensor into a packed MX4 tensor.

    Args:
        A (Tensor): [M] float tensor to be quantized.
        out (Tensor): [M / 2] output containing packed mx4 values.
        scale (Tensor): [M / GROUP_SIZE] containing mx4 shared exponents.
        rand_bits (Optional Tensor): [M, K / 2] random integers used for stochastic rounding.
        M (int): Number of input rows.
        K (int): Number of input columns.
        GROUPS_PER_ROW (int): Number of groups in each row of the input.
        GROUPS_PER_THREAD (int): Number of groups to process per thread.
        ROW_PADDING (int): Number of elements of padding to insert into each row.
        GROUP_SIZE (int): Size of chunks that use the same shared exponent.
        EBITS (int): Number of exponent bits in target mx4 format.
        MBITS (int): Number of mantissa bits in target mx4 format.
        ROUNDING_MODE (int): Which rounding method to use when calculating shared exponent.
        STOCHASTIC_CASTING (bool): Whether to use stochastic rounding when downcasting.
        FP4_EXP_BIAS (int): Exponent bias of target mx4 format.
        GROUP_LOAD (int): Number of groups to process simultaneously.
        USE_INT64 (bool): Whether to use int64 for indexing. This is needed for large tensors.
        USE_E8M0_SCALE (bool): Whether to use E8M0 for quantization
            (set to True when we want to mimic mx4's e8m0 scaling factor in nvfp4's fp8 local scale)
    """
    # Define Constant Expressions.
    BF16_MIN_NORMAL: tl.constexpr = 2 ** (-126)  # type: ignore[Incompatible variable type]

    # Get the current thread number.
    pid = tl.program_id(0)
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
    input_start = pid * (GROUPS_PER_THREAD * GROUP_SIZE)
    output_start = pid * OUTPUT_CHUNK_SIZE
    exp_start = pid * SCALE_CHUNK_SIZE
    # Initiate offset ranges used in kernel.
    input_offset = tl.arange(0, GROUP_LOAD * GROUP_SIZE) + input_start
    output_offset = tl.arange(0, GROUP_LOAD * (GROUP_SIZE // 8)) + output_start

    # We need to shift output offsets to make space for shared exponent storage.
    # Now create offsets for writing the shared exponent.
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
        # When theres no padding we can simplify indexing.
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
        if USE_E8M0_SCALE:
            scale_fp32 = group_max / 4.0 * input_global_scale
            scale_fp32 = _fp32_to_e8m0(scale_fp32, mbits=1, scale_round_mode="even")
        else:
            scale_fp32 = group_max / 6.0 * input_global_scale
        scale_ = scale_fp32.to(tl.float8e4nv)
        # Prevent infinite values in log.
        group_max = tl.where(group_max == 0, BF16_MIN_NORMAL, group_max)

        # Apply scale_ to input. We do this by broadcasting scale.
        # scaled_a = a * global_scale (fp32) / local_scale (fp8)
        scaled_a = tl.reshape(a, [GROUP_LOAD, GROUP_SIZE]) * tl.reshape(
            input_global_scale / scale_, [GROUP_LOAD, 1]
        )
        # Reshape back to a flat array.
        scaled_a = tl.reshape(scaled_a, [GROUP_LOAD * GROUP_SIZE])

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
        first_dim = exp_offset // (512 * n_col_blocks)
        second_dim = (exp_offset % (512 * n_col_blocks)) // (128 * n_col_blocks)
        third_dim = (exp_offset % (128 * n_col_blocks)) // (4 * n_col_blocks)
        fourth_dim = (exp_offset % (4 * n_col_blocks)) // 4
        fifth_dim = exp_offset % 4
        actual_offset = (
            first_dim * (512 * n_col_blocks)
            + fourth_dim * (512)
            + third_dim * (16)
            + second_dim * (4)
            + fifth_dim
        )
        tl.store(
            scale + actual_offset,
            scale_.to(tl.uint8, bitcast=True),
            # Prevent writing outside this chunk or the main array.
            mask=(exp_offset < SCALE_SIZE)
            & (exp_offset < (SCALE_CHUNK_SIZE * (pid + 1))),
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
        exp_offset += GROUP_LOAD
        output_offset += GROUP_LOAD * GROUP_SIZE // 8


@triton.jit
def _kernel_nvfp4_quantize_silu(
    A,
    B,
    input_global_scale_tensor,
    out,
    scale,
    rand_bits,
    M,
    K,
    GROUPS_PER_ROW,
    GROUPS_PER_THREAD,
    ROW_PADDING,
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
    """Quantize a 1D float tensor into a packed MX4 tensor.

    Args:
        A (Tensor): [M] float tensor to be quantized.
        out (Tensor): [M / 2] output containing packed mx4 values.
        scale (Tensor): [M / GROUP_SIZE] containing mx4 shared exponents.
        rand_bits (Optional Tensor): [M, K / 2] random integers used for stochastic rounding.
        M (int): Number of input rows.
        K (int): Number of input columns.
        GROUPS_PER_ROW (int): Number of groups in each row of the input.
        GROUPS_PER_THREAD (int): Number of groups to process per thread.
        ROW_PADDING (int): Number of elements of padding to insert into each row.
        GROUP_SIZE (int): Size of chunks that use the same shared exponent.
        EBITS (int): Number of exponent bits in target mx4 format.
        MBITS (int): Number of mantissa bits in target mx4 format.
        ROUNDING_MODE (int): Which rounding method to use when calculating shared exponent.
        STOCHASTIC_CASTING (bool): Whether to use stochastic rounding when downcasting.
        FP4_EXP_BIAS (int): Exponent bias of target mx4 format.
        GROUP_LOAD (int): Number of groups to process simultaneously.
        USE_INT64 (bool): Whether to use int64 for indexing. This is needed for large tensors.
    """
    # Define Constant Expressions.
    BF16_MIN_NORMAL: tl.constexpr = 2 ** (-126)  # type: ignore[Incompatible variable type]

    # Get the current thread number.
    pid = tl.program_id(0)
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
    input_start = pid * (GROUPS_PER_THREAD * GROUP_SIZE)
    output_start = pid * OUTPUT_CHUNK_SIZE
    exp_start = pid * SCALE_CHUNK_SIZE
    # Initiate offset ranges used in kernel.
    input_offset = tl.arange(0, GROUP_LOAD * GROUP_SIZE) + input_start
    output_offset = tl.arange(0, GROUP_LOAD * (GROUP_SIZE // 8)) + output_start

    # We need to shift output offsets to make space for shared exponent storage.
    # Now create offsets for writing the shared exponent.
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
        # When theres no padding we can simplify indexing.
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
        b = tl.load(
            B + padded_input_offset,
            # Mask values out of range for both the main array and this chunk. Also pad if needed.
            mask=(padded_input_offset < (M * K))
            & (padded_input_offset < ((pid + 1) * GROUPS_PER_THREAD * GROUP_SIZE))
            & pad_mask,
            other=0,
        )

        # compute silu
        a_upcast = a.to(tl.float32)
        b_upcast = b.to(tl.float32)
        a = a_upcast * tl.sigmoid(a_upcast) * b_upcast
        a = a.to(tl.bfloat16).to(tl.float32)

        # View the block in terms of groups.
        a_groups = tl.reshape(a, [GROUP_LOAD, GROUP_SIZE])
        # Compute the shared exponent of each group.
        group_max = tl.max(tl.abs(a_groups), axis=1)

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

        # We're done with group_exp now so we can write it out.
        # We readd fp16_exp_bias for compatibility with cuda dequant.
        n_col_blocks = SCALE_K // 4
        first_dim = exp_offset // (512 * n_col_blocks)
        second_dim = (exp_offset % (512 * n_col_blocks)) // (128 * n_col_blocks)
        third_dim = (exp_offset % (128 * n_col_blocks)) // (4 * n_col_blocks)
        fourth_dim = (exp_offset % (4 * n_col_blocks)) // 4
        fifth_dim = exp_offset % 4
        actual_offset = (
            first_dim * (512 * n_col_blocks)
            + fourth_dim * (512)
            + third_dim * (16)
            + second_dim * (4)
            + fifth_dim
        )
        tl.store(
            scale + actual_offset,
            scale_.to(tl.uint8, bitcast=True),
            # Prevent writing outside this chunk or the main array.
            mask=(exp_offset < SCALE_SIZE)
            & (exp_offset < (SCALE_CHUNK_SIZE * (pid + 1))),
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
        exp_offset += GROUP_LOAD
        output_offset += GROUP_LOAD * GROUP_SIZE // 8


def triton_scale_nvfp4_quant_silu(
    input: torch.Tensor,
    weight: torch.Tensor,
    input_global_scale: torch.Tensor,
    group_size: int = 16,
    ebits: int = 2,
    mbits: int = 1,
    rounding_mode: Union[RoundingMode, int] = RoundingMode.ceil,
    stochastic_casting: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize a tensor to nvfp4 format using efficient triton kernels.

    Args:
        a (Tensor): [M] higher precision input tensor.
        group_size (int): Size of chunks that will use the same shared exponent.
        ebits (int): Number of bits to use for exponent in target nvfp4 format.
        mbits (int): Number of bits to use for mantissa in target nvfp4 format.
        rounding_mode (Union[RoundingMode, int]): Which type of rounding to use
        when calculating shared exponent. Defaults to pre-rounding to nearest even int.
        stochastic_casting (bool): Whether to use stochastic casting.

    Returns:
        torch.Tensor: [M / 2] nvfp4 scaled tensor packed into int8
        torch.Tensor: [M / group_size] nvfp4 shared exponents into int8

        eg.
        Input with shape [1, 8192] will be quantized to [1, 4096 + 512] as
        each value contain two elements packed into an int8 and
        there are 16 elements per group.
    """

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

    rounded_M = round_up(M, 128)
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
    num_threads = math.ceil(math.sqrt(input.numel()))
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
    # Invoke triton quantization kernel over rows.

    grid = (num_threads,)
    _kernel_nvfp4_quantize_silu[grid](
        input,
        weight,
        input_global_scale,
        out,
        scale,
        rand_bits=rand_bits,
        M=M,
        K=K,
        GROUPS_PER_ROW=groups_per_row,
        GROUPS_PER_THREAD=groups_per_thread,
        ROW_PADDING=padding,
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
    return out.view(list(orig_shape[:-1]) + [-1]).view(torch.uint8), scale


@triton.jit
def _kernel_nvfp4_quantize_rms(
    A,
    B,
    input_global_scale_tensor,
    out,
    scale,
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
    """Quantize a 1D float tensor into a packed MX4 tensor.

    Args:
        A (Tensor): [M] float tensor to be quantized.
        out (Tensor): [M / 2] output containing packed mx4 values.
        scale (Tensor): [M / GROUP_SIZE] containing mx4 shared exponents.
        rand_bits (Optional Tensor): [M, K / 2] random integers used for stochastic rounding.
        M (int): Number of input rows.
        K (int): Number of input columns.
        GROUPS_PER_ROW (int): Number of groups in each row of the input.
        GROUPS_PER_THREAD (int): Number of groups to process per thread.
        ROW_PADDING (int): Number of elements of padding to insert into each row.
        GROUP_SIZE (int): Size of chunks that use the same shared exponent.
        EBITS (int): Number of exponent bits in target mx4 format.
        MBITS (int): Number of mantissa bits in target mx4 format.
        ROUNDING_MODE (int): Which rounding method to use when calculating shared exponent.
        STOCHASTIC_CASTING (bool): Whether to use stochastic rounding when downcasting.
        FP4_EXP_BIAS (int): Exponent bias of target mx4 format.
        GROUP_LOAD (int): Number of groups to process simultaneously.
        USE_INT64 (bool): Whether to use int64 for indexing. This is needed for large tensors.
    """
    # Define Constant Expressions.
    BF16_MIN_NORMAL: tl.constexpr = 2 ** (-126)  # type: ignore[Incompatible variable type]

    # Get the current thread number.
    pid = tl.program_id(0)
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
    input_start = pid * (GROUPS_PER_THREAD * GROUP_SIZE)
    output_start = pid * OUTPUT_CHUNK_SIZE
    exp_start = pid * SCALE_CHUNK_SIZE
    # Initiate offset ranges used in kernel.
    input_offset = tl.arange(0, GROUP_LOAD * GROUP_SIZE) + input_start
    output_offset = tl.arange(0, GROUP_LOAD * (GROUP_SIZE // 8)) + output_start

    # We need to shift output offsets to make space for shared exponent storage.
    # Now create offsets for writing the shared exponent.
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
        b = tl.load(
            B + padded_input_offset,
            # Mask values out of range for both the main array and this chunk. Also pad if needed.
            mask=(padded_input_offset < (M * K))
            & (padded_input_offset < ((pid + 1) * GROUPS_PER_THREAD * GROUP_SIZE))
            & pad_mask,
            other=0,
        )

        # compute rms

        a_segment = a.reshape(GROUP_LOAD, GROUP_SIZE).to(tl.float32)
        group_inv = tl_rsqrt(tl.sum(a_segment * a_segment, axis=1) / GROUP_SIZE + EPS)
        a = (
            (
                a_segment
                * group_inv.expand_dims(axis=1)
                * b.reshape(GROUP_LOAD, GROUP_SIZE).to(tl.float32)
            )
            .to(tl.bfloat16)
            .reshape(GROUP_LOAD * GROUP_SIZE)
            .to(tl.float32)
        )

        # View the block in terms of groups.
        a_groups = tl.reshape(a, [GROUP_LOAD, GROUP_SIZE])
        # Compute the shared exponent of each group.
        group_max = tl.max(tl.abs(a_groups), axis=1)

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

        # Further optimization could be done here to reduce the number of
        # instructions by using a single split.
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

        # We're done with group_exp now so we can write it out.
        # We readd fp16_exp_bias for compatibility with cuda dequant.
        n_col_blocks = SCALE_K // 4
        first_dim = exp_offset // (512 * n_col_blocks)
        second_dim = (exp_offset % (512 * n_col_blocks)) // (128 * n_col_blocks)
        third_dim = (exp_offset % (128 * n_col_blocks)) // (4 * n_col_blocks)
        fourth_dim = (exp_offset % (4 * n_col_blocks)) // 4
        fifth_dim = exp_offset % 4
        actual_offset = (
            first_dim * (512 * n_col_blocks)
            + fourth_dim * (512)
            + third_dim * (16)
            + second_dim * (4)
            + fifth_dim
        )
        tl.store(
            scale + actual_offset,
            scale_.to(tl.uint8, bitcast=True),
            # Prevent writing outside this chunk or the main array.
            mask=(exp_offset < SCALE_SIZE)
            & (exp_offset < (SCALE_CHUNK_SIZE * (pid + 1))),
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
        exp_offset += GROUP_LOAD
        output_offset += GROUP_LOAD * GROUP_SIZE // 8


def triton_scale_nvfp4_quant_rms(
    input: torch.Tensor,
    weight: torch.Tensor,
    input_global_scale: torch.Tensor,
    group_size: int = 16,
    ebits: int = 2,
    mbits: int = 1,
    rounding_mode: Union[RoundingMode, int] = RoundingMode.ceil,
    stochastic_casting: bool = False,
    EPS: float = 1e-5,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize a tensor to nvfp4 format using efficient triton kernels.

    Args:
        a (Tensor): [M] higher precision input tensor.
        group_size (int): Size of chunks that will use the same shared exponent.
        ebits (int): Number of bits to use for exponent in target nvfp4 format.
        mbits (int): Number of bits to use for mantissa in target nvfp4 format.
        rounding_mode (Union[RoundingMode, int]): Which type of rounding to use
        when calculating shared exponent. Defaults to pre-rounding to nearest even int.
        stochastic_casting (bool): Whether to use stochastic casting.

    Returns:
        torch.Tensor: [M / 2] nvfp4 scaled tensor packed into int8
        torch.Tensor: [M / group_size] nvfp4 shared exponents into int8

        eg.
        Input with shape [1, 8192] will be quantized to [1, 4096 + 512] as
        each value contain two elements packed into an int8 and
        there are 16 elements per group.
    """

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

    rounded_M = round_up(M, 128)
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
    num_threads = math.ceil(math.sqrt(input.numel()))
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
    # Invoke triton quantization kernel over rows.

    grid = (num_threads,)
    _kernel_nvfp4_quantize_rms[grid](
        input,
        weight,
        input_global_scale,
        out,
        scale,
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
    return out.view(list(orig_shape[:-1]) + [-1]).view(torch.uint8), scale


@triton.jit
def _kernel_nvfp4_quantize_stacked(
    A,
    input_global_scale_tensor,
    out,
    scale,
    belong_indices,
    starting_row_after_padding,
    row_within_tensor,
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
    """Quantize a 1D float tensor into a packed MX4 tensor.

    Args:
        A (Tensor): [M] float tensor to be quantized.
        out (Tensor): [M / 2] output containing packed mx4 values.
        scale (Tensor): [M / GROUP_SIZE] containing mx4 shared exponents.
        rand_bits (Optional Tensor): [M, K / 2] random integers used for stochastic rounding.
        M (int): Number of input rows.
        K (int): Number of input columns.
        GROUPS_PER_ROW (int): Number of groups in each row of the input.
        GROUPS_PER_THREAD (int): Number of groups to process per thread.
        ROW_PADDING (int): Number of elements of padding to insert into each row.
        GROUP_SIZE (int): Size of chunks that use the same shared exponent.
        EBITS (int): Number of exponent bits in target mx4 format.
        MBITS (int): Number of mantissa bits in target mx4 format.
        ROUNDING_MODE (int): Which rounding method to use when calculating shared exponent.
        STOCHASTIC_CASTING (bool): Whether to use stochastic rounding when downcasting.
        FP4_EXP_BIAS (int): Exponent bias of target mx4 format.
        GROUP_LOAD (int): Number of groups to process simultaneously.
        USE_INT64 (bool): Whether to use int64 for indexing. This is needed for large tensors.
    """
    # Define Constant Expressions.
    BF16_MIN_NORMAL: tl.constexpr = 2 ** (-126)  # type: ignore[Incompatible variable type]

    # Get the current thread number.
    pid = tl.program_id(0)
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
    tensor_idx = tl.load(
        belong_indices + row_idx,
        mask=(row_idx < M),
    )
    tensor_offset = (
        tl.load(starting_row_after_padding + tensor_idx, mask=(row_idx < M))
        * GROUPS_PER_ROW
    )
    inner_idx = (
        tl.load(
            row_within_tensor + row_idx,
            mask=(row_idx < M),
        )
        * GROUPS_PER_ROW
    )
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
        tensor_idx = tl.load(
            belong_indices + row_idx,
            mask=(row_idx < M),
        )
        tensor_offset = (
            tl.load(starting_row_after_padding + tensor_idx, mask=(row_idx < M))
            * GROUPS_PER_ROW
        )
        inner_idx = (
            tl.load(
                row_within_tensor + row_idx,
                mask=(row_idx < M),
            )
            * GROUPS_PER_ROW
        )
        actual_scale_offset = tensor_offset + inner_idx + exp_offset % GROUPS_PER_ROW

        output_offset += GROUP_LOAD * GROUP_SIZE // 8


def triton_nvfp4_quant_stacked(
    input: torch.Tensor,
    input_global_scale: torch.Tensor,
    belong_indices: torch.Tensor,
    starting_row_after_padding: torch.Tensor,
    row_within_tensor: torch.Tensor,
    group_size: int = 16,
    ebits: int = 2,
    mbits: int = 1,
    rounding_mode: Union[RoundingMode, int] = RoundingMode.ceil,
    stochastic_casting: bool = False,
    EPS: float = 1e-5,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize a tensor to nvfp4 format using efficient triton kernels.

    Args:
        a (Tensor): [M] higher precision input tensor.
        group_size (int): Size of chunks that will use the same shared exponent.
        ebits (int): Number of bits to use for exponent in target nvfp4 format.
        mbits (int): Number of bits to use for mantissa in target nvfp4 format.
        rounding_mode (Union[RoundingMode, int]): Which type of rounding to use
        when calculating shared exponent. Defaults to pre-rounding to nearest even int.
        stochastic_casting (bool): Whether to use stochastic casting.

    Returns:
        torch.Tensor: [M / 2] nvfp4 scaled tensor packed into int8
        torch.Tensor: [M / group_size] nvfp4 shared exponents into int8

        eg.
        Input with shape [1, 8192] will be quantized to [1, 4096 + 512] as
        each value contain two elements packed into an int8 and
        there are 32 elements per group.
    """

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

    rounded_M = round_up(M + (starting_row_after_padding.numel() - 1) * 128, 128)
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
    num_threads = math.ceil(math.sqrt(input.numel()))
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
    # Invoke triton quantization kernel over rows.

    grid = (num_threads,)
    _kernel_nvfp4_quantize_stacked[grid](
        input,
        input_global_scale,
        out,
        scale,
        belong_indices,
        starting_row_after_padding,
        row_within_tensor,
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
    return out.view(list(orig_shape[:-1]) + [-1]).view(torch.uint8), scale


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
def _kernel_nvfp4_quantize_stacked_silu(
    A,
    B,
    input_global_scale_tensor,
    out,
    scale,
    belong_indices,
    starting_row_after_padding,
    row_within_tensor,
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
    """Quantize a 1D float tensor into a packed MX4 tensor.

    Args:
        A (Tensor): [M] float tensor to be quantized.
        out (Tensor): [M / 2] output containing packed mx4 values.
        scale (Tensor): [M / GROUP_SIZE] containing mx4 shared exponents.
        rand_bits (Optional Tensor): [M, K / 2] random integers used for stochastic rounding.
        M (int): Number of input rows.
        K (int): Number of input columns.
        GROUPS_PER_ROW (int): Number of groups in each row of the input.
        GROUPS_PER_THREAD (int): Number of groups to process per thread.
        ROW_PADDING (int): Number of elements of padding to insert into each row.
        GROUP_SIZE (int): Size of chunks that use the same shared exponent.
        EBITS (int): Number of exponent bits in target mx4 format.
        MBITS (int): Number of mantissa bits in target mx4 format.
        ROUNDING_MODE (int): Which rounding method to use when calculating shared exponent.
        STOCHASTIC_CASTING (bool): Whether to use stochastic rounding when downcasting.
        FP4_EXP_BIAS (int): Exponent bias of target mx4 format.
        GROUP_LOAD (int): Number of groups to process simultaneously.
        USE_INT64 (bool): Whether to use int64 for indexing. This is needed for large tensors.
    """
    # Define Constant Expressions.
    BF16_MIN_NORMAL: tl.constexpr = 2 ** (-126)  # type: ignore[Incompatible variable type]

    # Get the current thread number.
    pid = tl.program_id(0)
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
    tensor_idx = tl.load(
        belong_indices + row_idx,
        mask=(row_idx < M),
    )
    tensor_offset = (
        tl.load(starting_row_after_padding + tensor_idx, mask=(row_idx < M))
        * GROUPS_PER_ROW
    )
    inner_idx = (
        tl.load(
            row_within_tensor + row_idx,
            mask=(row_idx < M),
        )
        * GROUPS_PER_ROW
    )
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
        b = tl.load(
            B + padded_input_offset,
            # Mask values out of range for both the main array and this chunk. Also pad if needed.
            mask=(padded_input_offset < (M * K))
            & (padded_input_offset < ((pid + 1) * GROUPS_PER_THREAD * GROUP_SIZE))
            & pad_mask,
            other=0,
        )

        # compute silu
        a_upcast = a.to(tl.float32)
        b_upcast = b.to(tl.float32)
        a = a_upcast * tl.sigmoid(a_upcast) * b_upcast
        a = a.to(tl.bfloat16).to(tl.float32)

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
        tensor_idx = tl.load(
            belong_indices + row_idx,
            mask=(row_idx < M),
        )
        tensor_offset = (
            tl.load(starting_row_after_padding + tensor_idx, mask=(row_idx < M))
            * GROUPS_PER_ROW
        )
        inner_idx = (
            tl.load(
                row_within_tensor + row_idx,
                mask=(row_idx < M),
            )
            * GROUPS_PER_ROW
        )
        actual_scale_offset = tensor_offset + inner_idx + exp_offset % GROUPS_PER_ROW

        output_offset += GROUP_LOAD * GROUP_SIZE // 8


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


def triton_nvfp4_quant_stacked_silu(
    input: torch.Tensor,
    weight: torch.Tensor,
    input_global_scale: torch.Tensor,
    belong_indices: torch.Tensor,
    starting_row_after_padding: torch.Tensor,
    row_within_tensor: torch.Tensor,
    group_size: int = 16,
    ebits: int = 2,
    mbits: int = 1,
    rounding_mode: Union[RoundingMode, int] = RoundingMode.ceil,
    stochastic_casting: bool = False,
    EPS: float = 1e-5,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize a tensor to nvfp4 format using efficient triton kernels.

    Args:
        a (Tensor): [M] higher precision input tensor.
        group_size (int): Size of chunks that will use the same shared exponent.
        ebits (int): Number of bits to use for exponent in target nvfp4 format.
        mbits (int): Number of bits to use for mantissa in target nvfp4 format.
        rounding_mode (Union[RoundingMode, int]): Which type of rounding to use
        when calculating shared exponent. Defaults to pre-rounding to nearest even int.
        stochastic_casting (bool): Whether to use stochastic casting.

    Returns:
        torch.Tensor: [M / 2] nvfp4 scaled tensor packed into int8
        torch.Tensor: [M / group_size] nvfp4 shared exponents into int8

        eg.
        Input with shape [1, 8192] will be quantized to [1, 4096 + 512] as
        each value contain two elements packed into an int8 and
        there are 32 elements per group.
    """

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

    rounded_M = round_up(M + (starting_row_after_padding.numel() - 1) * 128, 128)
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
    num_threads = math.ceil(math.sqrt(input.numel()))
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
    # Invoke triton quantization kernel over rows.

    grid = (num_threads,)
    _kernel_nvfp4_quantize_stacked_silu[grid](
        input,
        weight,
        input_global_scale,
        out,
        scale,
        belong_indices,
        starting_row_after_padding,
        row_within_tensor,
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
    return out.view(list(orig_shape[:-1]) + [-1]).view(torch.uint8), scale


@triton.jit
def _kernel_nvfp4_quantize_stacked_rms(
    A,
    B,
    input_global_scale_tensor,
    out,
    scale,
    belong_indices,
    starting_row_after_padding,
    row_within_tensor,
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
    """Quantize a 1D float tensor into a packed MX4 tensor.

    Args:
        A (Tensor): [M] float tensor to be quantized.
        out (Tensor): [M / 2] output containing packed mx4 values.
        scale (Tensor): [M / GROUP_SIZE] containing mx4 shared exponents.
        rand_bits (Optional Tensor): [M, K / 2] random integers used for stochastic rounding.
        M (int): Number of input rows.
        K (int): Number of input columns.
        GROUPS_PER_ROW (int): Number of groups in each row of the input.
        GROUPS_PER_THREAD (int): Number of groups to process per thread.
        ROW_PADDING (int): Number of elements of padding to insert into each row.
        GROUP_SIZE (int): Size of chunks that use the same shared exponent.
        EBITS (int): Number of exponent bits in target mx4 format.
        MBITS (int): Number of mantissa bits in target mx4 format.
        ROUNDING_MODE (int): Which rounding method to use when calculating shared exponent.
        STOCHASTIC_CASTING (bool): Whether to use stochastic rounding when downcasting.
        FP4_EXP_BIAS (int): Exponent bias of target mx4 format.
        GROUP_LOAD (int): Number of groups to process simultaneously.
        USE_INT64 (bool): Whether to use int64 for indexing. This is needed for large tensors.
    """
    # Define Constant Expressions.
    BF16_MIN_NORMAL: tl.constexpr = 2 ** (-126)  # type: ignore[Incompatible variable type]

    # Get the current thread number.
    pid = tl.program_id(0)
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
    tensor_idx = tl.load(
        belong_indices + row_idx,
        mask=(row_idx < M),
    )
    tensor_offset = (
        tl.load(starting_row_after_padding + tensor_idx, mask=(row_idx < M))
        * GROUPS_PER_ROW
    )
    inner_idx = (
        tl.load(
            row_within_tensor + row_idx,
            mask=(row_idx < M),
        )
        * GROUPS_PER_ROW
    )
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
        b = tl.load(
            B + padded_input_offset,
            # Mask values out of range for both the main array and this chunk. Also pad if needed.
            mask=(padded_input_offset < (M * K))
            & (padded_input_offset < ((pid + 1) * GROUPS_PER_THREAD * GROUP_SIZE))
            & pad_mask,
            other=0,
        )

        # compute rms
        a_segment = a.reshape(GROUP_LOAD, GROUP_SIZE).to(tl.float32)
        group_inv = tl_rsqrt(tl.sum(a_segment * a_segment, axis=1) / GROUP_SIZE + EPS)
        a = (
            (
                a_segment
                * group_inv.expand_dims(axis=1)
                * b.reshape(GROUP_LOAD, GROUP_SIZE).to(tl.float32)
            )
            .to(tl.bfloat16)
            .reshape(GROUP_LOAD * GROUP_SIZE)
            .to(tl.float32)
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
        tensor_idx = tl.load(
            belong_indices + row_idx,
            mask=(row_idx < M),
        )
        tensor_offset = (
            tl.load(starting_row_after_padding + tensor_idx, mask=(row_idx < M))
            * GROUPS_PER_ROW
        )
        inner_idx = (
            tl.load(
                row_within_tensor + row_idx,
                mask=(row_idx < M),
            )
            * GROUPS_PER_ROW
        )
        actual_scale_offset = tensor_offset + inner_idx + exp_offset % GROUPS_PER_ROW

        output_offset += GROUP_LOAD * GROUP_SIZE // 8


def triton_nvfp4_quant_stacked_rms(
    input: torch.Tensor,
    weight: torch.Tensor,
    input_global_scale: torch.Tensor,
    belong_indices: torch.Tensor,
    starting_row_after_padding: torch.Tensor,
    row_within_tensor: torch.Tensor,
    group_size: int = 16,
    ebits: int = 2,
    mbits: int = 1,
    rounding_mode: Union[RoundingMode, int] = RoundingMode.ceil,
    stochastic_casting: bool = False,
    EPS: float = 1e-5,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize a tensor to nvfp4 format using efficient triton kernels.

    Args:
        a (Tensor): [M] higher precision input tensor.
        group_size (int): Size of chunks that will use the same shared exponent.
        ebits (int): Number of bits to use for exponent in target nvfp4 format.
        mbits (int): Number of bits to use for mantissa in target nvfp4 format.
        rounding_mode (Union[RoundingMode, int]): Which type of rounding to use
        when calculating shared exponent. Defaults to pre-rounding to nearest even int.
        stochastic_casting (bool): Whether to use stochastic casting.

    Returns:
        torch.Tensor: [M / 2] nvfp4 scaled tensor packed into int8
        torch.Tensor: [M / group_size] nvfp4 shared exponents into int8

        eg.
        Input with shape [1, 8192] will be quantized to [1, 4096 + 512] as
        each value contain two elements packed into an int8 and
        there are 32 elements per group.
    """

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

    rounded_M = round_up(M + (starting_row_after_padding.numel() - 1) * 128, 128)
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
    num_threads = math.ceil(math.sqrt(input.numel()))
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
    # Invoke triton quantization kernel over rows.

    grid = (num_threads,)
    _kernel_nvfp4_quantize_stacked_rms[grid](
        input,
        weight,
        input_global_scale,
        out,
        scale,
        belong_indices,
        starting_row_after_padding,
        row_within_tensor,
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
    return out.view(list(orig_shape[:-1]) + [-1]).view(torch.uint8), scale


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


def get_nvfp4_global_scales_naive(
    xs: list[torch.Tensor], ws: list[torch.Tensor]
) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
    """
    Get global scales for each tensor in xs and ws.
    This is done "naively" (not efficiently with a kernel). This function is used in unit tests or debugging.
    """
    global_scales = []
    x_global_scales = []
    w_global_scales = []

    for x, w in zip(xs, ws):
        # pyre-ignore
        x_global_scale: torch.Tensor = (448.0 * 6.0) / torch.amax(
            torch.abs(x.flatten()), dim=-1
        ).to(torch.float32)
        # pyre-ignore
        w_global_scale: torch.Tensor = (448.0 * 6.0) / torch.amax(
            torch.abs(w.flatten()), dim=-1
        ).to(torch.float32)
        # pyre-ignore
        global_scale: torch.Tensor = 1 / (x_global_scale * w_global_scale)

        global_scales.append(global_scale)
        x_global_scales.append(x_global_scale)
        w_global_scales.append(w_global_scale)

    return global_scales, x_global_scales, w_global_scales


def quantize_nvfp4_naive(
    xs: list[torch.Tensor], global_scales: list[torch.Tensor]
) -> tuple[
    list[torch.Tensor],
    list[torch.Tensor],
]:
    """
    Quantize A to NVFP4 format.
    This is done "naively" using a kernel for each group. This function is largely used in unit tests or debugging.
    """
    xqs, x_scales = zip(
        *(
            triton_quantize_nvfp4(x, global_scale)
            for x, global_scale in zip(xs, global_scales)
        )
    )

    return xqs, x_scales


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


def triton_quantize_nvfp4(
    x: torch.Tensor,
    global_scale: torch.Tensor,
    use_e8m0_scale: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantize a tensor to NVFP4 format.

    Args:
        x (torch.Tensor): Input tensor to be quantized.
        global_scale (torch.Tensor): Per-tensor scale for two-level quantization.
        use_e8m0_scale (bool): Whether to use E8M0 for quantization. If True will mimic mx4's e8m0 scaling factor in nvfp4's fp8 local scale.

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
):
    E4M3_EPS = 1.5258789e-05

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
    if USE_MASK:
        mask = (offs_m < M) & (offs_n < N)
        other = 0.0
    else:
        mask = None
        other = None

    global_scale = tl.load(global_scale_ptr)  # Scalar

    x = tl.load(
        x_ptr + offs_m * stride_xm + offs_n * stride_xn, mask=mask, other=other
    )  # [M_PER_BLOCK, 64]
    x_blocks = x.to(tl.float32).reshape(M_PER_BLOCK, 4, 16)  # [M_PER_BLOCK, 4, 16]

    # Block-wise max
    block_amax = tl.max(tl.abs(x_blocks), axis=2)  # [M_PER_BLOCK, 4]

    if USE_E8M0_SCALE:
        scales = tl.div_rn(block_amax, 4.0) * global_scale
        scales = _fp32_to_e8m0(scales, mbits=1, scale_round_mode="even")
    else:
        scales = tl.div_rn(block_amax, FP4_E2M1_MAX) * global_scale
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
    tl.store(q_ptr + offs_m * (N // 2) + offs_n, x_fp4x2, mask=mask)


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
