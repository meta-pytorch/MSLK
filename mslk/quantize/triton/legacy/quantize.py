# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

from __future__ import annotations

import math
from typing import Tuple, Union

import torch
import triton  # @manual
from mslk.quantize.triton.legacy.primitives import (
    _compute_exp,
    _fp32_to_e8m0,
    convert_fp32_to_fp4_packed,
    get_mx4_exp_bias,
    nvfp4_scale_swizzle,
    RoundingMode,
)
from mslk.utils.device import is_gfx950, is_rocm
from triton import language as tl  # @manual


# ---------------------------------------------------------------------------
# gfx950 (CDNA4) hardware FP4 conversion
# ---------------------------------------------------------------------------
# V_CVT_SCALEF32_PK_FP4_F32 is a VOP3 instruction introduced in gfx950 (CDNA4,
# MI350/MI355X). It converts two F32 values to a packed byte containing two
# FP4 E2M1 nibbles using hardware block-scaling.
#
# AMD CDNA4 ISA Reference, section 6.7.1 Packed Convert — pseudo-code:
#   scale    = 32'U(exponent(S2.f32))   # extract biased exponent -> E8M0 uint8
#   tmp0     = f32_to_fp4_scale(S0.f32, scale.u8)   # S0 -> low  nibble [3:0]
#   tmp1     = f32_to_fp4_scale(S1.f32, scale.u8)   # S1 -> high nibble [7:4]
#   dstbyte  = OPSEL[3:2] * 8           # selects which byte [0..3] in VDST
#   VDST[dstbyte+7:dstbyte] = {tmp1, tmp0}
#   # other destination bits are preserved
#
# Scale operand is the GCN inline constant 1.0 (biased_exp=127 -> E8M0=127
# -> effective scale 2^(127-127)=1.0). No extra scaling is applied because
# the quantization kernel's group-scaling step already places inputs in
# [-6, 6] (the E2M1 representable range).
#
# Rounding: round-toward-nearest-even (RNE / banker's rounding), per ISA.
#   This differs from the pure-Triton path (_fp32_to_e2m1_nibble) which uses
#   nearest-ties-away-from-zero, so results may differ at exact midpoints
#   (e.g. ±0.25, ±0.75, ±1.25, ±1.75, ±2.5, ±3.5, ±5.0).
#
# OPSEL defaults to 0 when omitted, so the result is placed in bits [7:0].
# Nibble ordering (S0→low nibble [3:0], S1→high nibble [7:4]) matches the
# AMD VOP3 2-src packed-byte convention.
_GFX950_CVT_PK_FP4_F32 = tl.constexpr("v_cvt_scalef32_pk_fp4_f32 $0, $1, $2, 1.0")


@triton.jit
def _fp32_to_e2m1_nibble(v):
    """Encode one float32 element-wise as a 4-bit E2M1 nibble (int32, 0–15).

    Used by the pure-Triton fallback path (_fp32x8_to_e2m1_packed_rocm) for
    gfx942 (MI300X) and other non-gfx950 ROCm devices.  gfx950 uses the
    hardware instruction V_CVT_SCALEF32_PK_FP4_F32 instead (see
    _fp32x8_to_e2m1_packed_gfx950).

    E2M1 format: s | e1 | e0 | m0.
    Representable magnitudes: {0, 0.5, 1, 1.5, 2, 3, 4, 6}.
    Rounding: nearest-ties-away-from-zero (matches PTX ``cvt.rn`` for
    positive values).  This differs from the gfx950 hardware path which uses
    round-toward-nearest-even (RNE); results diverge only at exact midpoints
    (±0.25, ±0.75, ±1.25, ±1.75, ±2.5, ±3.5, ±5.0).

    Note on implementation: E2M1 levels are non-uniformly spaced, so a single
    IEEE 754 bit-manipulation trick (add rounding offset + shift) does NOT
    correctly handle all midpoints — specifically the 0.5↔1.0 boundary where
    the required offset differs from the 1.0↔1.5 boundary.  The tl.where chain
    is therefore the correct branchless GPU implementation: each tl.where lowers
    to a v_cmp + v_cndmask_b32 pair on CDNA, which is uniform across the warp.
    """
    sign = tl.where(v < 0.0, 1, 0).to(tl.int32)
    ax = tl.abs(v)
    # Clamp to the representable range [0, 6].
    ax = tl.minimum(ax, 6.0)
    # Midpoints between consecutive E2M1 levels; values *at* a midpoint
    # round up (nearest, ties-away-from-zero — matches PTX rn for positives).
    mag = tl.where(
        ax < 0.25,
        0,
        tl.where(
            ax < 0.75,
            1,
            tl.where(
                ax < 1.25,
                2,
                tl.where(
                    ax < 1.75,
                    3,
                    tl.where(
                        ax < 2.5, 4, tl.where(ax < 3.5, 5, tl.where(ax < 5.0, 6, 7))
                    ),
                ),
            ),
        ),
    )
    return ((sign << 3) | mag).to(tl.int32)


@triton.jit
def _fp32x8_to_e2m1_packed_rocm(
    f_one, f_five, f_three, f_seven, f_two, f_six, f_four, f_eight
):
    """Pack 8 float32 values into one int32 containing 8 E2M1 nibbles.

    Pure-Triton replacement for the PTX ``cvt.rn.satfinite.e2m1x2.f32``
    inline-asm block.  Argument order matches the PTX version:

        byte0 = pack(f_one,   f_five)   # low nibble = f_one, high = f_five
        byte1 = pack(f_three, f_seven)
        byte2 = pack(f_two,   f_six)
        byte3 = pack(f_four,  f_eight)
        result = byte0 | (byte1 << 8) | (byte2 << 16) | (byte3 << 24)
    """
    n0 = _fp32_to_e2m1_nibble(f_one)
    n1 = _fp32_to_e2m1_nibble(f_five)
    n2 = _fp32_to_e2m1_nibble(f_three)
    n3 = _fp32_to_e2m1_nibble(f_seven)
    n4 = _fp32_to_e2m1_nibble(f_two)
    n5 = _fp32_to_e2m1_nibble(f_six)
    n6 = _fp32_to_e2m1_nibble(f_four)
    n7 = _fp32_to_e2m1_nibble(f_eight)

    byte0 = n0 | (n1 << 4)
    byte1 = n2 | (n3 << 4)
    byte2 = n4 | (n5 << 4)
    byte3 = n6 | (n7 << 4)

    return byte0 | (byte1 << 8) | (byte2 << 16) | (byte3 << 24)


@triton.jit
def _fp32x8_to_e2m1_packed_gfx950(
    f_one, f_five, f_three, f_seven, f_two, f_six, f_four, f_eight
):
    """gfx950-native FP4 packing using V_CVT_SCALEF32_PK_FP4_F32.

    Replaces ~56 VALU compare+select ops (7 tl.where × 8 values) with
    4 hardware conversion instructions.  Only valid on gfx950 (CDNA4,
    MI350/MI355X); gfx942 (MI300X) does not have this instruction.

    ISA semantics (AMD CDNA4 §6.7.1, see module comment above):
        scale    = exponent(S2) as E8M0 uint8  [S2=1.0 → E8M0=127 → ×1.0]
        dst[3:0] = f32_to_fp4_scale(S0, scale)  # S0 → low  nibble
        dst[7:4] = f32_to_fp4_scale(S1, scale)  # S1 → high nibble
    Rounding: round-toward-nearest-even (RNE), per ISA.
    OPSEL defaults to 0: result lands in bits [7:0] of the destination VGPR.

    Constraints use ``v`` (VGPR) — the only valid class on AMDGPU LLVM;
    ``f`` (x87 float) and ``r`` (PTX register) are NVIDIA-only.

    Argument order matches _fp32x8_to_e2m1_packed_rocm so callers need no change:
        byte0 = CVT(f_one  [low nibble], f_five  [high nibble])
        byte1 = CVT(f_three[low nibble], f_seven [high nibble])
        byte2 = CVT(f_two  [low nibble], f_six   [high nibble])
        byte3 = CVT(f_four [low nibble], f_eight [high nibble])
        result = byte0 | (byte1<<8) | (byte2<<16) | (byte3<<24)
    """
    b0 = tl.inline_asm_elementwise(
        asm=_GFX950_CVT_PK_FP4_F32,
        constraints="=v,v,v",
        args=[f_one, f_five],
        dtype=tl.int32,
        is_pure=True,
        pack=1,
    )
    b1 = tl.inline_asm_elementwise(
        asm=_GFX950_CVT_PK_FP4_F32,
        constraints="=v,v,v",
        args=[f_three, f_seven],
        dtype=tl.int32,
        is_pure=True,
        pack=1,
    )
    b2 = tl.inline_asm_elementwise(
        asm=_GFX950_CVT_PK_FP4_F32,
        constraints="=v,v,v",
        args=[f_two, f_six],
        dtype=tl.int32,
        is_pure=True,
        pack=1,
    )
    b3 = tl.inline_asm_elementwise(
        asm=_GFX950_CVT_PK_FP4_F32,
        constraints="=v,v,v",
        args=[f_four, f_eight],
        dtype=tl.int32,
        is_pure=True,
        pack=1,
    )
    # Each bN has the packed FP4 byte in bits [7:0]; shift and OR into int32.
    return (b0 & 0xFF) | ((b1 & 0xFF) << 8) | ((b2 & 0xFF) << 16) | ((b3 & 0xFF) << 24)


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
    IS_ROCM: tl.constexpr,
    IS_GFX950: tl.constexpr,
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
        if IS_GFX950:
            # gfx950 (CDNA4): hardware V_CVT_SCALEF32_PK_FP4_F32 instruction.
            # 4 instructions replace ~56 compare+select ops of the fallback.
            packed_result = _fp32x8_to_e2m1_packed_gfx950(
                f_one, f_five, f_three, f_seven, f_two, f_six, f_four, f_eight
            )
        elif IS_ROCM:
            # gfx942 (CDNA3) or other ROCm: no FP4 hardware instruction.
            packed_result = _fp32x8_to_e2m1_packed_rocm(
                f_one, f_five, f_three, f_seven, f_two, f_six, f_four, f_eight
            )
        else:
            # NVIDIA: PTX hardware instruction (SM90a+ / Hopper+).
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

        row = exp_offset // GROUPS_PER_ROW
        col = exp_offset % GROUPS_PER_ROW
        padded_exp_offset = row * SCALE_K + col

        n_col_blocks = SCALE_K // 4
        first_dim = padded_exp_offset // (512 * n_col_blocks)
        second_dim = (padded_exp_offset % (512 * n_col_blocks)) // (128 * n_col_blocks)
        third_dim = (padded_exp_offset % (128 * n_col_blocks)) // (4 * n_col_blocks)
        fourth_dim = (padded_exp_offset % (4 * n_col_blocks)) // 4
        fifth_dim = padded_exp_offset % 4
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
    # E8M0 scale byte 127 is 2^0, the neutral scale for rounded tail slots.
    scale = torch.full(
        (rounded_M, rounded_K),
        127,
        device=device,
        dtype=torch.int8,
    )

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
        # pyre-ignore[6]
        IS_ROCM=is_rocm(),
        # pyre-ignore[6]
        IS_GFX950=is_gfx950(),
    )

    scale = scale.flatten()
    return out.view(list(orig_shape[:-1]) + [-1]).view(torch.uint8), scale


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
