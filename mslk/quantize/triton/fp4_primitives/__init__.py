# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

"""
fp4_primitives — Reusable building blocks for FP4 (E2M1) quantization kernels.

Submodules
----------
- constants: FP4/FP8 numeric limits, format parameters, RoundingMode enum.
- scale: Triton JIT helpers for computing MX4 quantization scales.
- packing: Triton JIT helpers for FP32→FP4 conversion and packing.
- layout: Blackwell 128×4 scale layout (blocked offset + segment map).
"""

from mslk.quantize.triton.fp4_primitives.constants import (  # noqa: F401
    BF16_MIN_NORMAL,
    E8M0_EXPONENT_BIAS,
    RoundingMode,
)
from mslk.quantize.triton.fp4_primitives.layout import (  # noqa: F401
    blocked_scale_offset,
    stacked_segment_map,
)
from mslk.quantize.triton.fp4_primitives.packing import (  # noqa: F401
    convert_fp32_to_fp4_packed,
)
from mslk.quantize.triton.fp4_primitives.scale import (  # noqa: F401
    mx4_scale_normalize_encode,
)
