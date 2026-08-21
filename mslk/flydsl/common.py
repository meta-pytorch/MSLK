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

# Minimum FlyDSL release the MSLK kernels are written against. Keep in sync
# with ``ci/flydsl_version.txt`` (the version setup.py pins for OSS installs).
# Older releases are missing APIs the kernels use (e.g. ``flydsl.expr.Pointer``
# and ``flydsl.compiler.protocol.extract_to_ir_values``) and fail at compile
# time rather than at import, so callers must gate on the version explicitly.
MIN_FLYDSL_VERSION: str = "0.2.4"

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


@functools.lru_cache(maxsize=None)
def flydsl_version() -> str | None:
    """Installed FlyDSL version, or ``None`` when it cannot be determined."""
    try:
        import flydsl  # pyre-ignore[21]

        return flydsl.__version__
    except Exception:
        return None


def _version_tuple(version: str) -> tuple[int, ...]:
    """Leading numeric components of a version, e.g. ``"0.2.4rc1" -> (0, 2, 4)``."""
    parts = []
    for part in version.split("."):
        digits = ""
        for ch in part:
            if not ch.isdigit():
                break
            digits += ch
        if not digits:
            break
        parts.append(int(digits))
    return tuple(parts)


def is_flydsl_version_at_least(minimum: str = MIN_FLYDSL_VERSION) -> bool:
    """True when FlyDSL is available and at least ``minimum``.

    ``is_flydsl_available()`` only checks importability, which is not enough:
    the kernels resolve most FlyDSL APIs lazily at compile time, so an
    out-of-date FlyDSL imports cleanly and then fails deep inside kernel
    compilation. Callers that need those APIs should gate on this instead.
    """
    if not is_flydsl_available():
        return False
    current = flydsl_version()
    if current is None:
        return False
    return _version_tuple(current) >= _version_tuple(minimum)


# Conservative default for architectures missing from FlyDSL's capacity table.
_LDS_CAPACITY_FALLBACK_BYTES = 64 * 1024


def lds_capacity_bytes(arch=None):
    """LDS bytes available to one workgroup on ``arch``.

    This is arch-dependent and the difference is large: CDNA3 (gfx942/MI300) has
    64 KiB while CDNA4 (gfx950/MI350) has 160 KiB. Sourced from FlyDSL's
    SMEM_CAPACITY_MAP so the limit stays in sync with the compiler that enforces
    it; unknown architectures fall back to the conservative 64 KiB.
    """
    try:
        from flydsl.utils.smem_allocator import SMEM_CAPACITY_MAP
    except Exception:
        return _LDS_CAPACITY_FALLBACK_BYTES
    return SMEM_CAPACITY_MAP.get(str(arch), _LDS_CAPACITY_FALLBACK_BYTES)
