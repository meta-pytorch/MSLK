# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from enum import Enum


class Regime(Enum):
    R1_LARGE_COMPUTE = "R1"
    R2_MEDIUM = "R2"
    R3_SKINNY_DECODE = "R3"
    R4_SMALL_TAIL = "R4"
    R5_GROUPED_MOE = "R5"


class CeilingType(Enum):
    MFMA_PEAK = "mfma_peak"
    HBM_BW = "hbm_bw"
    EMPIRICAL_BEST = "empirical_best"


# Target bands per regime: (low%, high%)
_TARGET_BANDS: dict[Regime, tuple[float, float]] = {
    Regime.R1_LARGE_COMPUTE: (85.0, 92.0),
    Regime.R2_MEDIUM: (75.0, 85.0),
    Regime.R3_SKINNY_DECODE: (70.0, 85.0),
    Regime.R4_SMALL_TAIL: (95.0, 100.0),
    Regime.R5_GROUPED_MOE: (90.0, 100.0),
}

_CEILING_TYPES: dict[Regime, CeilingType] = {
    Regime.R1_LARGE_COMPUTE: CeilingType.MFMA_PEAK,
    Regime.R2_MEDIUM: CeilingType.MFMA_PEAK,
    Regime.R3_SKINNY_DECODE: CeilingType.HBM_BW,
    Regime.R4_SMALL_TAIL: CeilingType.EMPIRICAL_BEST,
    Regime.R5_GROUPED_MOE: CeilingType.EMPIRICAL_BEST,
}


@dataclass(frozen=True)
class RegimeClassification:
    regime: Regime
    ceiling_type: CeilingType
    target_low: float
    target_high: float
    rationale: str


def classify_regime(
    M: int,
    N: int,
    K: int,
    is_grouped: bool = False,
    num_groups: int | None = None,
) -> RegimeClassification:
    """Classify a GEMM shape into regime R1–R5.

    Priority order: R5 (grouped), R4 (small N/K), R1 (large), R3 (skinny),
    then R2 (medium).
    """
    if is_grouped and num_groups is not None and num_groups > 1:
        regime = Regime.R5_GROUPED_MOE
        rationale = f"Grouped GEMM with {num_groups} groups"
    elif N < 1024 or K < 1024:
        regime = Regime.R4_SMALL_TAIL
        rationale = f"Small {'N' if N < 1024 else 'K'} dimension ({min(N, K)})"
    elif M >= 4096 and N >= 4096 and K >= 4096:
        regime = Regime.R1_LARGE_COMPUTE
        rationale = f"Large compute-bound (M={M})"
    elif M <= 128:
        regime = Regime.R3_SKINNY_DECODE
        rationale = f"Skinny M={M}, memory-bound (decode-like)"
    elif M >= 512:
        regime = Regime.R2_MEDIUM
        rationale = f"Medium M={M}, compute with tile-quant sensitivity"
    else:
        # M in (128, 512) — treat as R2 but note transition zone
        regime = Regime.R2_MEDIUM
        rationale = f"M={M} in transition zone (128–512), treated as medium"

    low, high = _TARGET_BANDS[regime]
    return RegimeClassification(
        regime=regime,
        ceiling_type=_CEILING_TYPES[regime],
        target_low=low,
        target_high=high,
        rationale=rationale,
    )


def classify_regime_batch(
    shapes: list[tuple[int, int, int]],
    is_grouped: bool = False,
    num_groups: int | None = None,
) -> list[RegimeClassification]:
    return [classify_regime(M, N, K, is_grouped, num_groups) for M, N, K in shapes]
