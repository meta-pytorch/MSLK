# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""Shared ``unittest`` skip decorators for gating GPU tests by device support.

Two kinds of gate, combined via ``unittest``'s AND-composition of stacked
decorators:

**Platform gates (strict)** pick the platform:

    @skipUnlessCuda()   # NVIDIA only
    @skipUnlessRocm()   # AMD only

**Refinements** narrow within a platform. Each is *transparent* on the other
platform (it never gates a device it doesn't speak for):

    @skipUnlessCudaCapability(min, max)   # NVIDIA compute capability
    @skipUnlessGfxArch(*archs)            # AMD gfx arch

A single-platform test stacks a platform gate with an optional refinement::

    @skipUnlessCuda()                 # CUDA, SM100+
    @skipUnlessCudaCapability(10)
    class MXFP4Tests(unittest.TestCase): ...

    @skipUnlessRocm()                 # ROCm, MI300
    @skipUnlessGfxArch("gfx942")
    class FooRocmTests(unittest.TestCase): ...

A dual-platform test (supported on NVIDIA *or* AMD) stacks the two refinements
instead. Because each is transparent on the other platform, AND-composition
yields the union — runs iff (CUDA & capability) or (ROCm & arch)::

    @skipUnlessGfxArch("gfx942")      # MI300, or
    @skipUnlessCudaCapability(9, 10)  # SM90 / SM100
    class FP8Tests(unittest.TestCase): ...

Skip messages are derived from the arguments; callers do not supply one.
"""

import unittest
from collections.abc import Callable
from typing import Any

from mslk.utils.device import (
    compute_capability_at_least,
    compute_capability_in,
    cuda_version_at_least,
    gfx_arch_in,
    is_cuda,
    is_rocm,
)


def skipUnlessCuda() -> Callable[..., Any]:
    """Skip the test unless running on any NVIDIA CUDA device.

    A strict platform gate. To narrow to a compute capability, stack with
    :func:`skipUnlessCudaCapability`.
    """
    return unittest.skipIf(not is_cuda(), "requires CUDA")


def skipUnlessRocm() -> Callable[..., Any]:
    """Skip the test unless running on any AMD ROCm device.

    A strict platform gate. To narrow to a gfx arch, stack with
    :func:`skipUnlessGfxArch`.
    """
    return unittest.skipIf(not is_rocm(), "requires ROCm")


def skipUnlessCudaCapability(
    major_min: int,
    major_max: int | None = None,
    *,
    minor_min: int = 0,
) -> Callable[..., Any]:
    """Narrow a CUDA test to a compute-capability range.

    Runs on CUDA only when the device compute-capability major is in
    ``[major_min, major_max]`` (``major_max=None`` means no upper bound, e.g.
    ``major_min=9`` is "SM90+"; ``major_min == major_max`` is an exact SM).

    ``minor_min`` adds a lower bound on the capability *minor*, so the gate
    compares the full ``(major, minor)`` pair — e.g. ``major_min=10,
    minor_min=3`` is "SM10.3+". It pairs naturally with no ``major_max``.

    Transparent on ROCm so it composes with :func:`skipUnlessGfxArch` into a
    dual-platform union. For a CUDA-only test, pair it with
    :func:`skipUnlessCuda` instead.
    """
    cuda_supported = (
        is_cuda()
        and compute_capability_in(major_min, major_max)
        and (minor_min == 0 or compute_capability_at_least(major_min, minor_min))
    )
    if minor_min:
        reason = f"requires CUDA SM{major_min}{minor_min}+"
    elif major_max is None:
        reason = f"requires CUDA SM{major_min}0+"
    elif major_min == major_max:
        reason = f"requires CUDA SM{major_min}0"
    else:
        reason = f"requires CUDA SM{major_min}0-SM{major_max}0"
    return unittest.skipIf(not (cuda_supported or is_rocm()), reason)


def skipUnlessCudaVersion(major_min: int) -> Callable[..., Any]:
    """Narrow a CUDA test to a minimum CUDA toolkit major version.

    Gates on the CUDA toolkit the binary was built against
    (``torch.version.cuda``), which is distinct from the device compute
    capability. Transparent on ROCm so it composes with the other refinements.
    """
    return unittest.skipIf(
        not (is_rocm() or (is_cuda() and cuda_version_at_least(major_min))),
        f"requires CUDA toolkit {major_min}+",
    )


def skipUnlessGfxArch(
    arch: str,
    *more_archs: str,
) -> Callable[..., Any]:
    """Narrow a ROCm test to one or more gfx archs.

    Runs on ROCm only when the device's gfx arch matches one of the given archs
    (e.g. ``"gfx942"`` for MI300, ``"gfx950"`` for MI350).

    Transparent on CUDA so it composes with :func:`skipUnlessCudaCapability`
    into a dual-platform union. For a ROCm-only test, pair it with
    :func:`skipUnlessRocm` instead.
    """
    archs = (arch, *more_archs)
    gfx_supported = is_rocm() and gfx_arch_in(archs)
    reason = f"requires ROCm arch {' or '.join(archs)}"
    return unittest.skipIf(not (gfx_supported or is_cuda()), reason)
