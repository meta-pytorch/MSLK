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
import shutil
import sys
from pathlib import Path

_INSTALL_HINT: str = (
    "FlyDSL is required for this kernel but is not available. "
    "Install it with `pip install flydsl`."
)

# Minimum FlyDSL release the MSLK kernels are written against. This is a floor,
# not the pin: ``ci/flydsl_version.txt`` holds the exact version setup.py pins
# for OSS installs, which follows aiter's FlyDSL pin and may be newer. Raise
# this only when the kernels start using APIs from a newer release.
# Older releases are missing APIs the kernels use (e.g. ``flydsl.expr.Pointer``
# and ``flydsl.compiler.protocol.extract_to_ir_values``) and fail at compile
# time rather than at import, so callers must gate on the version explicitly.
MIN_FLYDSL_VERSION: str = "0.4.0"

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
    _prepare_flydsl_runtime_environment()
    try:
        from flydsl.runtime.device import (  # pyre-ignore[21]
            get_rocm_arch,
            SMEM_CAPACITY_MAP,
        )

        return get_rocm_arch() in SMEM_CAPACITY_MAP
    except Exception:
        return False


def _add_rocm_sysdeps_to_loader_path() -> None:
    """Expose the ROCm linker's private shared libraries to child processes."""
    rocm_home = os.environ.get("ROCM_HOME") or os.environ.get("ROCM_PATH")
    if rocm_home:
        root = Path(rocm_home).resolve()
    else:
        hipcc = shutil.which("hipcc")
        if hipcc is None:
            return
        root = Path(hipcc).resolve().parent.parent
        if root.name == "hip":
            root = root.parent
    sysdeps = root / "lib" / "rocm_sysdeps" / "lib"
    if not sysdeps.is_dir():
        return
    directory = str(sysdeps)
    current = os.environ.get("LD_LIBRARY_PATH", "")
    if directory not in current.split(":"):
        os.environ["LD_LIBRARY_PATH"] = (
            f"{current}:{directory}" if current else directory
        )


def _prepare_flydsl_runtime_environment() -> None:
    """Use the selected physical GPU and expose ROCm linker dependencies."""
    _add_rocm_sysdeps_to_loader_path()
    if os.environ.get("FLYDSL_GPU_ARCH") or os.environ.get("HSA_OVERRIDE_GFX_VERSION"):
        return
    # Torch is already imported by runtime callers. Do not import it while
    # building a GPU-free AOT cache or probing an optional backend.
    torch = sys.modules.get("torch")
    if torch is None or getattr(getattr(torch, "version", None), "hip", None) is None:
        return
    cuda = getattr(torch, "cuda", None)
    if cuda is None or not cuda.is_available():
        return
    try:
        arch = cuda.get_device_properties(cuda.current_device()).gcnArchName
    except RuntimeError:
        # A failing driver query must not escape the availability probe:
        # leave FLYDSL_GPU_ARCH unset so is_flydsl_available() still
        # returns False instead of raising to every caller.
        return
    # FlyDSL otherwise queries rocm_agent_enumerator and silently defaults to
    # gfx942 when the tool is missing or fails in a hermetic test sandbox.
    # An empty gcnArchName (unreported on some driver/ROCm combos) must not
    # become an empty override: FlyDSL trusts an explicit FLYDSL_GPU_ARCH
    # instead of enumerating, reintroducing the silent-wrong-arch failure.
    base = (arch or "").split(":", 1)[0].lower()
    if base:
        os.environ["FLYDSL_GPU_ARCH"] = base


def require_flydsl() -> None:
    """Require FlyDSL and make its ROCm linker dependencies discoverable."""
    _prepare_flydsl_runtime_environment()
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
        from flydsl.runtime.device import SMEM_CAPACITY_MAP
    except Exception:
        return _LDS_CAPACITY_FALLBACK_BYTES
    return SMEM_CAPACITY_MAP.get(str(arch), _LDS_CAPACITY_FALLBACK_BYTES)
