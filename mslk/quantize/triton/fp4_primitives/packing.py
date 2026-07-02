# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

"""
FP4 packing primitives for converting float values to packed FP4 E2M1 format.

Blackwell PTX hardware-accelerated packing.
"""

import triton  # @manual
from triton import language as tl  # @manual


@triton.jit
def convert_fp32_to_fp4_packed(x_pairs):
    """Convert FP32 value pairs to packed FP4 E2M1 format using PTX inline assembly.

    Uses the Blackwell ``cvt.rn.satfinite.e2m1x2.f32`` PTX instruction to convert
    pairs of FP32 values into packed FP4 bytes. Each output byte contains two
    E2M1 values:

    - Low nibble (bits 0-3): first value of the pair
    - High nibble (bits 4-7): second value of the pair

    The PTX instruction handles:
    - Round-to-nearest-even rounding
    - Saturation to the FP4 E2M1 range [-6.0, 6.0]
    - Proper encoding of E2M1 special values

    Implementation detail: The ``pack=4`` parameter means 4 pairs are processed
    per inline asm invocation, producing 4 bytes that are assembled into a single
    uint32 via ``mov.b32``.

    Args:
        x_pairs: List of 8 float32 tensors representing 4 pairs:
            [pair0_lo, pair1_lo, pair2_lo, pair3_lo,
             pair0_hi, pair1_hi, pair2_hi, pair3_hi]

    Returns:
        Packed uint8 tensor where each byte contains two E2M1 values.

    Note:
        Requires Blackwell (SM 100+) GPU. Will fail on older architectures.
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
