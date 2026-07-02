# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

"""
Scale computation primitives for MX4 quantization.

Triton JIT helpers for the shared MX4 exponent, per-group scale + normalize,
and biased E8M0 encode.
"""

import triton  # @manual
from mslk.quantize.triton.fp4_primitives.constants import (
    BF16_MIN_NORMAL,
    E8M0_EXPONENT_BIAS,
)
from triton import language as tl  # @manual


# =============================================================================
# Shared Exponent Helpers
# =============================================================================


@triton.jit
def _floor_log2(x):
    """Efficiently compute floor(log2(x)) using FP32 bit manipulation.

    Instead of computing ``floor(log2(x))`` via expensive math operations, this
    extracts the exponent field directly from the IEEE 754 FP32 bit representation:

        FP32 layout:  [1 sign][8 exponent][23 mantissa]
        exponent_bits = (bitcast_to_int(x) & 0x7F800000) >> 23
        floor_log2    = exponent_bits - 127  (remove FP32 bias)

    This is exact for powers of 2, and equivalent to floor(log2(x)) for all
    positive normal floats (because the mantissa bits only add a fractional part
    to the true log2).

    Args:
        x: FP32 input tensor. Must be positive (undefined for x <= 0).

    Returns:
        floor(log2(x)) as float32 tensor.
    """
    FP32_EXP_MASK: tl.constexpr = 0x7F800000  # type: ignore[Incompatible variable type]
    FP32_EXP_OFFSET: tl.constexpr = 23  # type: ignore[Incompatible variable type]
    FP32_EXP_BIAS: tl.constexpr = 127  # type: ignore[Incompatible variable type]

    x = x.to(tl.int32, bitcast=True) & FP32_EXP_MASK
    x = x >> FP32_EXP_OFFSET
    return (x - FP32_EXP_BIAS).to(tl.float32)


@triton.jit
def _round_log2_via_bits(x, val_to_add: tl.constexpr):
    """Compute a rounded ``log2(x)`` via FP32 bit manipulation.

    Generalizes :func:`_floor_log2` by adding ``val_to_add`` to the FP32 bit
    representation **before** extracting the exponent. Choice of
    ``val_to_add`` selects the rounding mode:

      - ``val_to_add = 0``               → ``floor(log2(x))`` (== :func:`_floor_log2`).
      - ``val_to_add = (1 << 23) - 1``   → round-up (ceil): equals
        ``ceil(log2(x))`` for positive normals. The carry happens iff
        the mantissa is non-zero, i.e. iff ``x`` is not an exact power
        of two.

    Note that this helper computes the *mathematically exact*
    ``ceil(log2(x))`` for positive normal FP32 inputs, which can differ
    from ``tl.ceil(tl.log2(x))`` at exact powers of two when the
    underlying device intrinsic ``log2`` returns a value slightly above
    the true integer (a known ULP-level imprecision in CUDA libm).

    Args:
        x: Positive FP32 tensor (undefined for ``x <= 0`` and for
            inf/NaN — same domain as :func:`_floor_log2`).
        val_to_add: Compile-time integer constant selecting the rounding
            mode (see above).

    Returns:
        Rounded ``log2(x)`` as a float32 tensor.
    """
    FP32_EXP_MASK: tl.constexpr = 0x7F800000  # type: ignore[Incompatible variable type]
    FP32_EXP_OFFSET: tl.constexpr = 23  # type: ignore[Incompatible variable type]
    FP32_EXP_BIAS: tl.constexpr = 127  # type: ignore[Incompatible variable type]

    bits = x.to(tl.int32, bitcast=True) + val_to_add
    bits = (bits & FP32_EXP_MASK) >> FP32_EXP_OFFSET
    return (bits - FP32_EXP_BIAS).to(tl.float32)


@triton.jit
def _compute_exp(
    group_max,
    rounding_mode,
    rand_bits,
    MBITS: tl.constexpr,
):
    """Compute the shared MX4 exponent for a group using the specified rounding mode.

    The shared exponent determines the scale for a group of values. Different
    rounding modes produce different exponents:

    - ``nearest`` (0): Round log2(group_max) to nearest integer.
    - ``floor`` (1): Use fast bit-manipulation floor(log2).
    - ``even`` (2): Pre-round mantissa bits, then floor(log2). Reduces bias.
    - ``stochastic`` (3): Add random noise to mantissa, then floor(log2).
    - ``ceil`` (4): Always round up log2(group_max).

    Args:
        group_max: Per-group maximum absolute values (float32 tensor).
        rounding_mode: Which rounding mode to use (int or RoundingMode value).
        rand_bits: Random integer values used for stochastic rounding (ignored for
            other modes).
        MBITS: Number of mantissa bits in the target MX4 format (compile-time constant).

    Returns:
        Shared exponent for each group as float32 tensor.
    """
    MBITS_FP32: tl.constexpr = 23  # type: ignore[Incompatible variable type]
    M_ROUND: tl.constexpr = (1 << (MBITS_FP32 - MBITS - 1)) - 1  # type: ignore
    RAND_MASK: tl.constexpr = (1 << (MBITS_FP32 - MBITS)) - 1  # type: ignore
    CEIL_ROUND: tl.constexpr = (1 << MBITS_FP32) - 1  # type: ignore

    if rounding_mode == 0:
        # nearest (round-half-up of log2). The "correct" round-half-up
        # transition is at x = 2^k * sqrt(2), which does not correspond
        # to a clean bit-trick threshold, so we keep the legacy
        # ``floor(log2(x) + 0.5)`` formulation here.
        return tl.floor(tl.log2(group_max) + 0.5)
    if rounding_mode == 1:
        return _floor_log2(group_max)
    elif rounding_mode == 2:
        group_max = group_max.to(tl.int32, bitcast=True) + M_ROUND
        return _floor_log2(group_max)
    elif rounding_mode == 3:
        group_max = group_max.to(tl.int32, bitcast=True) + (RAND_MASK & rand_bits)
        return _floor_log2(group_max)
    else:
        # ceil: round-up via bit-extract.
        return _round_log2_via_bits(group_max, CEIL_ROUND)


# =============================================================================
# MX4 Scale Computation
# =============================================================================


@triton.jit
def encode_mx4_exponent(group_exp):
    """Encode an integer-valued group exponent for biased E8M0 storage.

    Adds the E8M0 exponent bias (127) and casts to int8 for storage. This is
    the inverse of decoding: ``original_exp = stored_value - 127``.

    Args:
        group_exp: Integer-valued shared exponent (float32 tensor), typically
            in the range [-127, 125] after clamping.

    Returns:
        Biased exponent as int8 tensor, with values in [0, 252].
    """
    return (group_exp + E8M0_EXPONENT_BIAS).to(tl.int8)


@triton.jit
def mx4_scale_normalize_encode(
    x_blocks,  # [M_PER_BLOCK, NUM_GROUPS, GROUP_SIZE] fp32 tile (pre-amax reshape)
    block_amax,  # [M_PER_BLOCK, NUM_GROUPS] per-group abs-max
    pid_m,  # tl.program_id(1) (runtime)
    pid_n,  # tl.program_id(0) (runtime)
    seed,  # int64 seed for tl.randint (only consulted when STOCHASTIC=True)
    N,  # number of columns (runtime int)
    M_PER_BLOCK: tl.constexpr,  # rows per program
    NUM_GROUPS: tl.constexpr,  # scale groups per 64-col tile (TILE_N // GROUP_SIZE)
    GROUP_SIZE: tl.constexpr,  # elements per scale group (32 for MX4)
    ROUNDING_MODE: tl.constexpr,  # RoundingMode enum for the shared exponent
    EBITS: tl.constexpr,  # exponent bits in target FP4 format (2 for E2M1)
    MBITS: tl.constexpr,  # mantissa bits in target FP4 format (1 for E2M1)
    STOCHASTIC: tl.constexpr,  # True → software stochastic rounding
):
    """Shared MX4 scale + normalize + (optional stochastic) + E8M0 encode.

    Given the fp32 tile ``x_blocks`` and its per-group ``block_amax``, this:
      1. zero-guards amax with ``BF16_MIN_NORMAL``;
      2. computes the shared exponent via ``_compute_exp`` (per-group stochastic
         rand bits when ``STOCHASTIC``), subtracts ``EBITS``, clamps to [-127, 125];
      3. builds the fp32 scale via an **fp64** ``exp2`` intermediate and divides
         ``x_blocks`` by it;
      4. optionally adds per-element mantissa-LSB noise for stochastic rounding of
         the FP32→FP4 cast (decorrelated stream via ``seed ^ 0x5A5A5A5A``);
      5. encodes the exponent as biased E8M0 int8.

    The stochastic group/element offsets use the **physical** grid row
    ``pid_m * M_PER_BLOCK``.

    Returns ``(x_blocks, encoded_scales)`` where ``x_blocks`` is the normalized
    (and stochastically perturbed) tile and ``encoded_scales`` is int8 E8M0.
    """
    safe_amax = tl.where(block_amax == 0, BF16_MIN_NORMAL, block_amax)

    # Stochastic rounding (OCP MX v1.0): per-group rand bits feed the
    # ``rounding_mode == 3`` branch of ``_compute_exp``. When
    # ``STOCHASTIC=False``, we pass scalar ``0`` so the non-stochastic
    # path is byte-identical to the pre-stochastic kernel.
    if STOCHASTIC:
        groups_per_row = N // GROUP_SIZE
        group_offsets = (pid_m * M_PER_BLOCK + tl.arange(0, M_PER_BLOCK))[
            :, None
        ] * groups_per_row + (pid_n * NUM_GROUPS + tl.arange(0, NUM_GROUPS))[None, :]
        group_rand_bits = tl.randint(seed, group_offsets, n_rounds=7)
        group_exp = _compute_exp(safe_amax, ROUNDING_MODE, group_rand_bits, MBITS)
    else:
        group_exp = _compute_exp(safe_amax, ROUNDING_MODE, 0, MBITS)
    group_exp = group_exp - EBITS
    group_exp = tl.clamp(group_exp, -127, 125)

    # Use float64 intermediate for exp2 precision
    scale_float = tl.exp2(group_exp.to(tl.float64)).to(tl.float32)

    # Normalize input by dividing by scale
    x_blocks = x_blocks / scale_float[:, :, None]

    # Per-element stochastic rounding on the FP32→FP4 cast. Add uniform noise in
    # the discarded mantissa bits before the ``cvt.rn.satfinite.e2m1x2.f32`` PTX
    # cast — the noise crosses the half-ulp boundary with the correct probability,
    # giving unbiased stochastic rounding. The element stream is decorrelated from
    # the per-group stream via a constexpr XOR of the seed
    # (``ELEM_SEED_XOR = 0x5A5A5A5A``). The noise mask ``(1 << (23 - MBITS)) - 1``
    # covers the bits strictly below the rn-tiebreak position for E2M1 (MBITS=1);
    # we subtract half the mask so the noise is symmetric about 0 (the FP32 sign
    # bit is untouched).
    if STOCHASTIC:
        ELEM_SEED_XOR: tl.constexpr = 0x5A5A5A5A  # type: ignore
        elem_offsets = (
            (pid_m * M_PER_BLOCK + tl.arange(0, M_PER_BLOCK))[:, None, None] * N
            + (pid_n * NUM_GROUPS + tl.arange(0, NUM_GROUPS))[None, :, None]
            * GROUP_SIZE
            + tl.arange(0, GROUP_SIZE)[None, None, :]
        )
        per_elem_rand_bits = tl.randint(seed ^ ELEM_SEED_XOR, elem_offsets, n_rounds=7)
        PER_ELEM_RAND_MASK: tl.constexpr = (1 << (23 - MBITS)) - 1  # type: ignore
        x_bits = x_blocks.to(tl.int32, bitcast=True)
        x_bits = (
            x_bits
            + (per_elem_rand_bits & PER_ELEM_RAND_MASK)
            - (PER_ELEM_RAND_MASK >> 1)
        )
        x_blocks = x_bits.to(tl.float32, bitcast=True)

    # Encode as biased E8M0 int8
    encoded_scales = encode_mx4_exponent(group_exp)
    return x_blocks, encoded_scales
