# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""Generic support for JIT-compiling FlyDSL kernels in MSLK.

FlyDSL is an optional ROCm-only backend (install with ``pip install
mslk[flydsl]``). Kernel modules use these helpers to detect availability,
fail with an actionable error when it is missing, and dispatch a FlyDSL
host launcher with torch tensors. The helpers are kernel-agnostic; layout
and op-specific logic stays in the kernel modules.
"""

import functools
import importlib.util
from typing import Any, Callable

_INSTALL_HINT: str = (
    "FlyDSL is required for this kernel but is not available. "
    "Install it with `pip install mslk[flydsl]`."
)


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
        from flydsl.runtime.device import get_rocm_arch
        from flydsl.utils.smem_allocator import SMEM_CAPACITY_MAP

        return get_rocm_arch() in SMEM_CAPACITY_MAP
    except Exception:
        return False


def require_flydsl() -> None:
    """Raise ``RuntimeError`` with an install hint when FlyDSL is unavailable."""
    if not is_flydsl_available():
        raise RuntimeError(_INSTALL_HINT)


def run_compiled(launcher: Callable[..., Any], *args: Any) -> None:
    """Dispatch a FlyDSL ``@flyc.jit`` host launcher with ``args``.

    The first call compiles and runs the kernel, caching the
    ``CompiledFunction`` on the launcher; later calls reuse it.
    """
    import flydsl.compiler as flyc

    cf = getattr(launcher, "_mslk_cf", None)
    if cf is None:
        launcher._mslk_cf = flyc.compile(launcher, *args)
    else:
        cf(*args)
