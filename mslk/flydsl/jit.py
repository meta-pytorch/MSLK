# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""Just-in-time compilation support for FlyDSL kernels in MSLK.

Kernel modules use these helpers to dispatch a FlyDSL host launcher with
torch tensors, compiling on first use. On import this also points FlyDSL at
the bundled AOT cache so precompiled kernels are served without a front-end
compile, falling back to JIT on a cache miss. Availability detection lives
in :mod:`mslk.flydsl.common`.
"""

import os
from typing import Any, Callable

from mslk.flydsl.common import _BUNDLED_AOT_CACHE


def configure_runtime_cache() -> None:
    """Point FlyDSL at the bundled AOT cache, if present and not overridden.

    Cache hits skip the front-end compile; misses fall back to JIT. Set
    ``MSLK_FLYDSL_DISABLE_JIT=1`` to forbid the JIT fallback (e.g. for CUDA
    graph capture), which makes FlyDSL raise on a cache miss instead.
    """
    if "FLYDSL_RUNTIME_CACHE_DIR" not in os.environ and os.path.isdir(
        _BUNDLED_AOT_CACHE
    ):
        os.environ["FLYDSL_RUNTIME_CACHE_DIR"] = _BUNDLED_AOT_CACHE
    if os.environ.get("MSLK_FLYDSL_DISABLE_JIT", "0") == "1":
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
