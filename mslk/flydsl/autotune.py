# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

"""Opt-in tile autotuning for FlyDSL kernels.

Wraps FlyDSL's autotuner in the policy MSLK uses for every backend: tuning runs
only when ``MSLK_AUTOTUNE_ENABLE`` is set, and otherwise a fixed default config
is used with no benchmarking. This mirrors the CUTLASS and CK paths in
``csrc/gemm`` (see ``get_kernel_via_tuning`` vs ``get_kernel_via_heuristics``),
and it matters for two reasons: benchmarking on a cache miss would make CI pay
for a sweep, and it cannot happen inside a CUDA graph capture. With the env set,
a tuned config is benchmarked once per key and persisted to disk by FlyDSL, so
callers should warm the shapes they intend to capture before capturing them.

Kernel modules supply their own launch function, candidate configs, and cache
key; nothing here is specific to a kernel or an op.

    TILES = ({"tile_m": 64, "tile_n": 128}, {"tile_m": 128, "tile_n": 128})

    @tunable(
        configs=TILES,
        default={"tile_m": 128, "tile_n": 128},
        key=["m_bucket", "n"],
        prune=prune_by_divisibility({"tile_n": "n"}),
    )
    def launch(A, B, out, m_bucket, n, *, tile_m, tile_n):
        ...
        return out
"""

import functools
import inspect
import os
from typing import Any, Callable, Mapping, Sequence

AUTOTUNE_ENV = "MSLK_AUTOTUNE_ENABLE"


def autotune_enabled() -> bool:
    """True when tuning is opted into for this process."""
    return bool(os.environ.get(AUTOTUNE_ENV))


def next_pow2(x: int) -> int:
    """Smallest power of two >= x (x >= 1).

    Useful for bucketing a varying dimension into an autotune key so that nearby
    sizes share one tuned config, which bounds both the number of tuned entries
    and the set of shapes a caller must warm before graph capture.
    """
    if x <= 1:
        return 1
    return 1 << (int(x) - 1).bit_length()


def prune_by_divisibility(
    constraints: Mapping[str, str],
) -> Callable[..., list]:
    """Build a prune hook dropping configs that do not divide the problem.

    ``constraints`` maps a config key to the launch argument it must divide, e.g.
    ``{"tile_n": "n"}`` keeps only configs whose ``tile_n`` divides ``n``. If
    every config is pruned the full list is returned instead, leaving the failure
    to the kernel's own validation rather than producing an empty sweep.
    """

    def prune(configs, named_args, **_kwargs):
        kept = []
        for c in configs:
            ok = True
            for cfg_key, arg_name in constraints.items():
                extent = named_args.get(arg_name)
                value = c.kwargs.get(cfg_key)
                if extent is None or value is None:
                    continue
                if int(extent) % int(value) != 0:
                    ok = False
                    break
            if ok:
                kept.append(c)
        return kept or list(configs)

    return prune


def tunable(
    *,
    configs: Sequence[Mapping[str, Any]],
    default: Mapping[str, Any],
    key: Sequence[str],
    prune: Callable[..., list] | None = None,
    returns: str = "out",
):
    """Choose a launch function's tuning kwargs by autotuning, or by a default.

    The wrapped function takes the tuning parameters as keyword-only arguments
    and everything the cache key needs as ordinary arguments.

    ``returns`` names the output argument. FlyDSL's autotuner discards the tuned
    function's return value, so on the tuned path the wrapper hands back that
    argument instead, which requires the kernel to write into a caller-provided
    buffer.
    """

    def decorate(launch_fn):
        signature = inspect.signature(launch_fn)
        state: dict = {}

        def _tuner():
            tuner = state.get("tuner")
            if tuner is None:
                from flydsl.autotune import autotune, Config

                tuner = autotune(
                    configs=[Config(**dict(c)) for c in configs],
                    key=list(key),
                    prune_configs_by=prune,
                )(launch_fn)
                state["tuner"] = tuner
            return tuner

        @functools.wraps(launch_fn)
        def call(*args, **kwargs):
            if not autotune_enabled():
                return launch_fn(*args, **kwargs, **dict(default))
            _tuner()(*args, **kwargs)
            bound = signature.bind_partial(*args, **kwargs)
            bound.apply_defaults()
            try:
                return bound.arguments[returns]
            except KeyError:
                raise TypeError(
                    f"{launch_fn.__name__} has no argument {returns!r} to return "
                    "from the tuned path; pass returns= naming its output buffer"
                ) from None

        call.launch_untuned = launch_fn
        return call

    return decorate
