# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

"""
FP4 packing primitives for converting float values to packed FP4 E2M1 format.

Device-aware packing:
- NVIDIA (SM100+): Blackwell PTX ``cvt.rn.satfinite.e2m1x2.f32``.
- AMD gfx950 (CDNA4 / MI350X): native ``v_cvt_scalef32_pk_fp4_f32``.
- Other ROCm (gfx942, ...): pure-Triton ``_fp32_to_e2m1_nibble`` fallback.

All three backends emit the identical packed layout: for each value pair
``(t0, t1)`` (from ``x_blocks.reshape(..., 32, 2).split()``), ``t0`` goes in the
low nibble [3:0] and ``t1`` in the high nibble [7:4] of the output byte.
"""

import triton  # @manual
from triton import language as tl  # @manual


# gfx950 (CDNA4) hardware FP4 conversion: two f32 -> one packed E2M1 byte.
# ``$1`` -> low nibble [3:0], ``$2`` -> high nibble [7:4]; scale operand 1.0
# (E8M0=127, no extra scaling — the kernel already group-scaled into [-6, 6]).
# Rounding: round-to-nearest-even (RNE), per AMD CDNA4 ISA.
_GFX950_CVT_PK_FP4_F32 = tl.constexpr("v_cvt_scalef32_pk_fp4_f32 $0, $1, $2, 1.0")


@triton.jit
def _fp32_to_e2m1_nibble(v):
    """Encode one float32 element as a 4-bit E2M1 nibble (int32, 0-15).

    Pure-Triton fallback for non-gfx950 ROCm (gfx942, etc.). Representable
    magnitudes: {0, 0.5, 1, 1.5, 2, 3, 4, 6}; rounding is round-half-up
    (ties always round toward the larger magnitude). This diverges from gfx950
    hardware (which uses RNE) at tie values 0.25, 1.25, 2.5, and 5.0. E2M1
    levels are non-uniformly spaced, so the branchless ``tl.where`` chain
    (each lowers to v_cmp + v_cndmask on CDNA) is the correct GPU
    implementation.
    """
    sign = tl.where(v < 0.0, 1, 0).to(tl.int32)
    ax = tl.abs(v)
    # Clamp to the representable range [0, 6].
    ax = tl.minimum(ax, 6.0)
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
def convert_fp32_to_fp4_packed(
    x_pairs,
    IS_GFX950: tl.constexpr,
    IS_ROCM: tl.constexpr,
):
    """Convert FP32 value pairs to packed FP4 E2M1 format.

    Args:
        x_pairs: 2-tuple ``(lo, hi)`` of fp32 tensors (from ``.split()`` on a
            ``[..., 2]`` trailing dim). ``lo`` -> low nibble, ``hi`` -> high
            nibble of each output byte.
        IS_GFX950: compile-time; use the gfx950 native conversion instruction.
        IS_ROCM: compile-time; use the pure-Triton fallback (non-gfx950 ROCm).
            Ignored when ``IS_GFX950`` is True.

    Returns:
        Packed uint8 tensor; each byte holds two E2M1 values.

    Default (both False) is the NVIDIA Blackwell PTX path (SM100+).
    """
    if IS_GFX950:
        lo, hi = x_pairs
        # v_cvt writes the packed byte into bits [7:0] of a 32-bit VGPR, so the
        # inline-asm output must be int32 (a uint8 output can't be allocated to a
        # `=v` register); mask the low byte and narrow to uint8.
        packed = tl.inline_asm_elementwise(
            asm=_GFX950_CVT_PK_FP4_F32,
            constraints="=v,v,v",
            args=[lo, hi],
            dtype=tl.int32,
            is_pure=True,
            pack=1,
        )
        # pyre-ignore[16,58]: tl.inline_asm_elementwise is typed as a tuple by
        # pyre; at runtime (dtype=int32) this is a tl.tensor supporting & / .to().
        return (packed & 0xFF).to(tl.uint8)
    elif IS_ROCM:
        lo, hi = x_pairs
        return (_fp32_to_e2m1_nibble(lo) | (_fp32_to_e2m1_nibble(hi) << 4)).to(tl.uint8)
    else:
        # NVIDIA Blackwell (SM100+): pairs -> packed FP4 via PTX. ``pack=4``
        # processes 4 pairs per invocation ($1-$4 = lo lanes, $5-$8 = hi lanes),
        # assembling 4 bytes into a uint32 via mov.b32. Round-to-nearest-even,
        # saturation to [-6, 6].
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
