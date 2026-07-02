# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

"""
Constants and enums for FP4 (E2M1) quantization.

Defines numeric limits, format parameters, and rounding modes used throughout
the MX4 quantization pipeline.
"""

from enum import IntEnum

from triton import language as tl  # @manual=//triton:triton


# =============================================================================
# E8M0 / BF16 Constants
# =============================================================================
#
# These are annotated ``tl.constexpr`` at module scope so they remain valid Triton
# kernel constants in OSS: OSS Triton rejects a plain (non-constexpr) Python global
# referenced inside a ``@triton.jit`` kernel (triton-lang/triton#3762). The
# annotation is a no-op for host/Python use (the value is the bare float/int).
E8M0_EXPONENT_BIAS: tl.constexpr = 127  # type: ignore[Incompatible variable type]
"""Exponent bias for the E8M0 scale format (same as IEEE 754 FP32 exponent bias)."""

BF16_MIN_NORMAL: tl.constexpr = 2 ** (-126)  # type: ignore[Incompatible variable type]
"""Minimum positive normal BF16 value; zero-guard in MX4 scale computation."""


class RoundingMode(IntEnum):
    """Rounding options for quantization.

    Controls how shared exponents are rounded during quantization. Each mode
    trades off between accuracy and hardware cost:

    - ``nearest``: Round to nearest integer (0.5 rounds up). Simple and fast.
    - ``floor``: Always round down. Uses fast FP32 bit manipulation.
    - ``even``: Round to nearest even integer (banker's rounding). Reduces bias.
    - ``stochastic``: Add random noise before truncation. Unbiased in expectation.
    - ``ceil``: Always round up. Most conservative (largest exponent).
    """

    nearest = 0
    floor = 1
    even = 2
    stochastic = 3
    ceil = 4
