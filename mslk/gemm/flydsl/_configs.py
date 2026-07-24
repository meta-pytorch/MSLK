# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""Default FlyDSL preshuffle GEMM tile configurations for gfx950."""

from dataclasses import dataclass


@dataclass
class KernelConfig:
    tile_m: int
    tile_n: int
    tile_k: int
    lds_stage: int
    use_cshuffle_epilog: int = 0
    use_async_copy: int = 0
    waves_per_eu: int = 0
    xcd_swizzle: int = 0


# Default configs for gfx950 heuristic selection.
# Keys are negative IDs (convention from aiter tuning infrastructure).
DEFAULT_CONFIGS_GFX950: dict[int, KernelConfig] = {
    -1: KernelConfig(128, 256, 256, 2, 0, 0, 2, 0),
    -2: KernelConfig(16, 64, 512, 2, 0, 0, 2, 0),
    -3: KernelConfig(32, 64, 512, 2, 0, 0, 2, 0),
    -4: KernelConfig(128, 128, 128, 2, 0, 0, 2, 0),
}


def select_default_config(m: int, n: int, k: int) -> KernelConfig:
    """Select a default FlyDSL tile config based on shape heuristics."""
    configs = DEFAULT_CONFIGS_GFX950
    fits = [c for c in configs.values() if n % c.tile_n == 0 and k % c.tile_k == 0]
    if not fits:
        raise RuntimeError(
            f"No FlyDSL preshuffle config fits shape ({m}, {n}, {k}). "
            f"N must be divisible by tile_n and K by tile_k."
        )
    want_tm = min(256, max(16, 1 << (m - 1).bit_length())) if m > 0 else 16
    return min(fits, key=lambda c: (abs(c.tile_m - want_tm), -c.tile_n, -c.tile_k))
