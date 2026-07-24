# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""Shared FlyDSL helpers used by both the JIT and AOT paths.

FlyDSL is a ROCm-only backend, required on ROCm builds and absent
elsewhere. These helpers detect availability and locate the bundled cache;
they are kernel-agnostic and independent of whether kernels are compiled
just-in-time or ahead-of-time.
"""

import functools
import importlib.util
import os

_INSTALL_HINT: str = (
    "FlyDSL is required for this kernel but is not available. "
    "Install it with `pip install flydsl`."
)

# Bundled AOT cache shipped inside the package (populated at build time).
# Shared across all FlyDSL kernel categories (gemm, moe, ...), so it lives
# under the dedicated mslk.flydsl package rather than under any kernel domain.
_BUNDLED_AOT_CACHE: str = os.path.join(os.path.dirname(__file__), "aot_artifacts")


@functools.lru_cache(maxsize=None)
def is_flydsl_available() -> bool:
    """True when FlyDSL is importable and supports the current GPU arch.

    FlyDSL only ships kernels for the architectures in its
    ``SMEM_CAPACITY_MAP``; importing kernel modules on other archs fails
    during config registration, so those archs are reported unavailable.
    """
    if importlib.util.find_spec("flydsl") is None:
        return False
    try:
        from flydsl.runtime.device import get_rocm_arch  # pyre-ignore[21]
        from flydsl.utils.smem_allocator import SMEM_CAPACITY_MAP  # pyre-ignore[21]

        return get_rocm_arch() in SMEM_CAPACITY_MAP
    except Exception:
        return False


def require_flydsl() -> None:
    """Raise ``RuntimeError`` with an install hint when FlyDSL is unavailable."""
    if not is_flydsl_available():
        raise RuntimeError(_INSTALL_HINT)
