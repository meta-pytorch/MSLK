# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""Just-in-time compilation support for FlyDSL kernels in MSLK.

Kernel modules use these helpers to dispatch a FlyDSL host launcher with
torch tensors, compiling on first use. :func:`configure_runtime_cache` points
FlyDSL at the bundled AOT cache so precompiled kernels are served without a
front-end compile, falling back to JIT on a cache miss; the package
``__init__`` calls it, so it applies to every FlyDSL kernel rather than only
to those launched from here. Availability detection lives in
:mod:`mslk.flydsl.common`.
"""

import contextlib
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


def _launch_device(args: tuple[Any, ...]) -> Any:
    """The device of the first CUDA tensor or stream in ``args``, else the
    current device."""
    import torch

    for arg in args:
        if isinstance(arg, torch.Tensor) and arg.is_cuda:
            return arg.device
        if isinstance(arg, torch.cuda.Stream):
            return arg.device
    return torch.device("cuda", torch.cuda.current_device())


def _launcher_on(launcher: Any, index: int) -> Any:
    """``launcher`` itself on the first device it runs on, and a copy of it
    with caches of its own on any other.

    FlyDSL caches a compiled kernel on the ``@flyc.jit`` function, keyed by its
    arguments' signature but not by device, and the cached kernel's GPU
    modules are loaded into the device that was current when it was first
    run; HIP refuses to launch them on any other. One launcher therefore
    cannot serve two GPUs, so each further device gets its own, built from the
    same source. The kernel is compiled once per device, or read back from
    FlyDSL's disk cache.
    """
    owner = getattr(launcher, "_mslk_device", None)
    if owner is None:
        launcher._mslk_device = owner = index
    if owner == index:
        return launcher
    copies = getattr(launcher, "_mslk_device_copies", None)
    if copies is None:
        copies = launcher._mslk_device_copies = {}
    copy = copies.get(index)
    if copy is None:
        from flydsl.compiler.jit_function import JitFunction  # pyre-ignore[21]

        # The source as written, before FlyDSL rewrote it for tracing.
        copy = copies[index] = JitFunction(
            launcher._original_func, compile_hints=launcher.compile_hints
        )
    return copy


def run_compiled(launcher: Callable[..., Any], *args: Any, device: Any = None) -> None:
    """Dispatch a FlyDSL ``@flyc.jit`` host launcher with ``args``.

    The first call on a device compiles and runs the kernel, caching the
    ``CompiledFunction`` for that device; later calls on it reuse it. The
    kernel is compiled and launched with its device made current, which is
    where FlyDSL loads it; see :func:`_launcher_on` for why each device needs
    a launcher of its own.

    The device is ``device`` if given, else that of the first CUDA tensor or
    stream in ``args``, else the current one. Callers whose operands all go
    through :func:`ptr_arg` carry no tensor, so they pass ``device``.
    """
    import flydsl.compiler as flyc  # pyre-ignore[21]
    import torch

    if device is None:
        device = _launch_device(args)
    device = torch.device(device)
    current = torch.cuda.current_device()
    index = device.index if device.index is not None else current

    launcher = _launcher_on(launcher, index)
    # Callers usually have the device current already, and switching costs a
    # round trip to the runtime on every launch.
    with contextlib.nullcontext() if index == current else torch.cuda.device(index):
        cf = getattr(launcher, "_mslk_cf", None)
        if cf is None:
            launcher._mslk_cf = flyc.compile(launcher, *args)  # pyre-ignore[16]
        else:
            cf(*args)


def ptr_arg(tensor: Any) -> Any:
    """Wrap a tensor's data pointer as a FlyDSL pointer kernel argument.

    A launcher declaring ``fx.Pointer`` operands takes the raw address rather
    than the tensor, so its call sites pass every operand through here; one
    declaring ``fx.Tensor`` operands takes them directly and does not.

    A fake tensor has no storage to address and becomes a null pointer: meta
    tracing only has to get through the call, never to run it.
    """
    import flydsl.compiler as flyc  # pyre-ignore[21]
    import flydsl.expr as fx  # pyre-ignore[21]

    cls = type(tensor)
    if cls.__name__ == "FakeTensor" or "fake_tensor" in cls.__module__:
        return flyc.from_c_void_p(fx.Uint8, 0)
    return flyc.from_c_void_p(fx.Uint8, tensor.data_ptr())
