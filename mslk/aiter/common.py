# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""The single gate every aiter path in MSLK goes through.

aiter serves an op only when all of these hold:

- this is a ROCm build of PyTorch and aiter is installed;
- the installed aiter is exactly the build MSLK pins (``aiter_version.txt``),
  the only one its ops are tested against, unless ``MSLK_AITER_ALLOW_UNPINNED``
  is set;
- aiter is not switched off, globally or for the op (see below).

Within those limits, each op's tuning table decides which calls aiter serves.

Environment variables
---------------------

All are optional, and read once, on first use; tests that change them call
:func:`reset`.

``MSLK_DISABLE_AITER=1``
    Turns off every aiter path: all ops use MSLK's own kernels, as if aiter
    were not installed. For ruling aiter out while debugging, comparing
    performance with and without aiter, or working around a misbehaving aiter
    kernel without uninstalling it.

``MSLK_DISABLE_AITER_OPS=op1,op2``
    Turns off the aiter path of the listed ops only (MSLK op names, e.g.
    ``f4f4bf16``), which then always use MSLK's kernel; other ops still use
    aiter where their tables say so. For when one op's aiter path has a problem.

``MSLK_AITER_ALLOW_UNPINNED=1``
    Uses the installed aiter even when it is not the pinned build. By default
    such an aiter is not used, with one warning, because MSLK's tables and
    tests only vouch for the pinned build. For experiments, e.g. trying a newer
    aiter before moving the pin, or environments that build their own aiter;
    not for production.

The switches can only turn aiter off: neither makes aiter serve a call its
table does not give it. Tests and tuning scripts choose a backend with
:func:`mslk.aiter.routing.force_backend` instead.

Nothing here imports aiter; it is imported on the first call that uses it,
never by ``import mslk``.
"""

import functools
import importlib
import importlib.metadata
import importlib.resources
import importlib.util
import logging
import os
from types import ModuleType
from typing import FrozenSet, Optional, Tuple

import torch

logger: logging.Logger = logging.getLogger(__name__)

AITER_DIST: str = "amd-aiter"
DISABLE_ENV: str = "MSLK_DISABLE_AITER"
DISABLE_OPS_ENV: str = "MSLK_DISABLE_AITER_OPS"
ALLOW_UNPINNED_ENV: str = "MSLK_AITER_ALLOW_UNPINNED"


@functools.lru_cache(maxsize=1)
def pinned_version() -> str:
    """The aiter version MSLK is tested against."""
    return (
        importlib.resources.files("mslk.aiter")
        .joinpath("aiter_version.txt")
        .read_text(encoding="utf-8")
        .strip()
    )


def installed_version() -> Optional[str]:
    """The installed aiter's version, or ``None`` if aiter is not installed."""
    try:
        return importlib.metadata.version(AITER_DIST)
    except importlib.metadata.PackageNotFoundError:
        return None


@functools.lru_cache(maxsize=1)
def is_aiter_available() -> bool:
    """True when MSLK may use the installed aiter at all; decided once per process."""
    if torch.version.hip is None:
        return False
    installed = installed_version()
    if installed is None or importlib.util.find_spec("aiter") is None:
        return False
    if installed != pinned_version():
        if os.environ.get(ALLOW_UNPINNED_ENV, "0") == "1":
            logger.warning(
                "Using aiter %s, which is not the version MSLK is tested against "
                "(%s), because %s=1.",
                installed,
                pinned_version(),
                ALLOW_UNPINNED_ENV,
            )
            return True
        logger.warning(
            "aiter %s is installed, but MSLK is tested against aiter %s only, so "
            "it will not use aiter. Install the pinned version, or set %s=1 to "
            "use this one anyway.",
            installed,
            pinned_version(),
            ALLOW_UNPINNED_ENV,
        )
        return False
    return True


@functools.lru_cache(maxsize=1)
def _switches() -> Tuple[bool, FrozenSet[str]]:
    """(all of aiter off, ops with aiter off), read from the environment once."""
    disabled_ops = os.environ.get(DISABLE_OPS_ENV, "")
    return (
        os.environ.get(DISABLE_ENV, "0") == "1",
        frozenset(name.strip() for name in disabled_ops.split(",") if name.strip()),
    )


def is_aiter_enabled(op: str) -> bool:
    """True when aiter may serve ``op``: available and not switched off."""
    all_off, ops_off = _switches()
    if all_off or op in ops_off:
        return False
    return is_aiter_available()


def reset() -> None:
    """Read the environment and check the installed aiter again on next use.
    For tests that change the environment variables."""
    _switches.cache_clear()
    is_aiter_available.cache_clear()


@functools.lru_cache(maxsize=None)
def import_aiter_module(name: str) -> ModuleType:
    """Import an aiter module, e.g. ``"aiter.ops.triton.gemm.basic.gemm_afp4wfp4"``.

    Callers check :func:`is_aiter_enabled` first; this only guards against
    importing an aiter that MSLK has not accepted.
    """
    if not is_aiter_available():
        raise RuntimeError(
            f"aiter is not available to MSLK (installed: {installed_version()}, "
            f"pinned: {pinned_version()})"
        )
    return importlib.import_module(name)
