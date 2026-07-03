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
import os
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
        from flydsl.runtime.device import get_rocm_arch  # pyre-ignore[21]
        from flydsl.utils.smem_allocator import SMEM_CAPACITY_MAP  # pyre-ignore[21]

        return get_rocm_arch() in SMEM_CAPACITY_MAP
    except Exception:
        return False


def require_flydsl() -> None:
    """Raise ``RuntimeError`` with an install hint when FlyDSL is unavailable."""
    if not is_flydsl_available():
        raise RuntimeError(_INSTALL_HINT)


# Bundled AOT cache shipped inside the package (populated at build time).
# Shared across all FlyDSL kernel categories (gemm, moe, ...), so it lives
# alongside the FlyDSL support module rather than under any kernel domain.
_BUNDLED_AOT_CACHE: str = os.path.join(os.path.dirname(__file__), "flydsl_aot_cache")


def configure_runtime_cache() -> None:
    """Point FlyDSL at the bundled AOT cache, if present and not overridden.

    Cache hits skip the front-end compile; misses fall back to JIT. Set
    ``MSLK_FLYDSL_RUN_ONLY=1`` to forbid the JIT fallback (e.g. for CUDA
    graph capture), which makes FlyDSL raise on a cache miss instead.
    """
    if "FLYDSL_RUNTIME_CACHE_DIR" not in os.environ and os.path.isdir(
        _BUNDLED_AOT_CACHE
    ):
        os.environ["FLYDSL_RUNTIME_CACHE_DIR"] = _BUNDLED_AOT_CACHE
    if os.environ.get("MSLK_FLYDSL_RUN_ONLY", "0") == "1":
        os.environ["FLYDSL_RUNTIME_RUN_ONLY"] = "1"


def run_compiled(launcher: Callable[..., Any], *args: Any) -> None:
    """Dispatch a FlyDSL ``@flyc.jit`` host launcher with ``args``.

    The first call compiles and runs the kernel, caching the
    ``CompiledFunction`` on the launcher; later calls reuse it.
    """
    import flydsl.compiler as flyc  # pyre-ignore[21]

    cf = getattr(launcher, "_mslk_cf", None)
    if cf is None:
        launcher._mslk_cf = flyc.compile(launcher, *args)  # pyre-ignore[16]
    else:
        cf(*args)


configure_runtime_cache()
