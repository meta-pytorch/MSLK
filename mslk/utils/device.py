# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""Centralized device-capability checks for CUDA (NVIDIA) and ROCm (AMD).

This is the single place to answer questions like "is this a ROCm build?",
"what is the CUDA compute capability?", or "is this gfx942/gfx950?". Library
code and tests should use these helpers instead of re-implementing the same
``torch.version.*`` / ``torch.cuda.*`` branching, which has historically been
duplicated across the codebase.

For test-only helpers built on top of these primitives (skip decorators and
feature-support flags), see ``mslk.testing.device``.
"""

import functools
import logging
import os
from collections.abc import Iterable

import torch

logger: logging.Logger = logging.getLogger(__name__)


def is_cuda() -> bool:
    """True when running on a CUDA (NVIDIA) build with an available device."""
    return torch.version.cuda is not None and torch.cuda.is_available()


def is_rocm() -> bool:
    """True when running on a ROCm (AMD) build with an available device."""
    return torch.version.hip is not None and torch.cuda.is_available()


def compute_capability_in(major_min: int, major_max: int | None = None) -> bool:
    """True if the device compute-capability major is in ``[major_min, major_max]``.

    ``major_max=None`` means no upper bound. Returns ``False`` when no GPU is
    available.
    """
    if not torch.cuda.is_available():
        return False
    major, _ = torch.cuda.get_device_capability()
    return major >= major_min and (major_max is None or major <= major_max)


def compute_capability_at_least(major_min: int, minor_min: int = 0) -> bool:
    """True if the device compute capability is at least ``(major_min, minor_min)``.

    Unlike :func:`compute_capability_in`, this compares the full
    ``(major, minor)`` pair, so it can express e.g. SM10.3+ (``(10, 3)``).
    Returns ``False`` when no GPU is available.
    """
    if not torch.cuda.is_available():
        return False
    return torch.cuda.get_device_capability() >= (major_min, minor_min)


def cuda_version_at_least(major_min: int) -> bool:
    """True on a CUDA build whose toolkit major version is at least ``major_min``.

    This checks the CUDA *toolkit* the binary was built against
    (``torch.version.cuda``), which is distinct from the device compute
    capability. Returns ``False`` on ROCm or CPU-only builds.
    """
    if torch.version.cuda is None:
        return False
    return int(torch.version.cuda.split(".")[0]) >= major_min


def get_gfx_arch_name() -> str:
    """Return the ROCm ``gcnArchName`` of the current device (e.g. ``gfx942``).

    Returns an empty string when no GPU is available or the arch cannot be
    determined.
    """
    if not torch.cuda.is_available():
        return ""
    try:
        # gcnArchName only exists on ROCm device-property objects; on a CUDA
        # build accessing it raises AttributeError.
        return torch.cuda.get_device_properties("cuda").gcnArchName
    except (RuntimeError, AssertionError, AttributeError):
        return ""


def gfx_arch_in(arch_list: Iterable[str]) -> bool:
    """True if the current ROCm device's gfx arch matches any entry in ``arch_list``."""
    gcn_arch_name = get_gfx_arch_name()
    return any(arch in gcn_arch_name for arch in arch_list)


def is_gfx942() -> bool:
    """True on AMD MI300X (CDNA3, gfx942)."""
    return is_rocm() and gfx_arch_in(["gfx942"])


def is_gfx950() -> bool:
    """True on AMD MI350 (CDNA4, gfx950)."""
    return is_rocm() and gfx_arch_in(["gfx950"])


# When set to "1" / "true" / "yes", forces MSLK to use AMD fnuz FP8
# (torch.float8_e4m3fnuz / tl.float8e4b8) even on gfx950 (MI350X), which
# natively uses OCP FP8 (torch.float8_e4m3fn).  Useful for debugging or when
# interoperating with code that always produces fnuz tensors.
_FORCE_FP8_FNUZ: bool = os.getenv("MSLK_ROCM_FORCE_FP8FNUZ_TYPE", "0").lower() in (
    "1",
    "true",
    "yes",
)


@functools.lru_cache
def supports_float8_fnuz(throw_on_hip_incompatibility: bool = True) -> bool:
    """Whether the current device uses the FP8 ``fnuz`` format (ROCm only).

    gfx942 (MI300) reports ``(9, 4)`` and gfx950 (MI350) reports ``(9, 5)``;
    both use the fnuz format. CUDA devices return ``False``.

    Args:
        throw_on_hip_incompatibility: When running on an unsupported ROCm arch,
            raise ``RuntimeError`` instead of logging and returning ``False``.
    """
    if not is_rocm():
        return False

    # gfx942 (MI300) reports (9, 4) and gfx950 (MI350) reports (9, 5).
    device_capability = torch.cuda.get_device_capability()
    if device_capability >= (9, 4):
        if _FORCE_FP8_FNUZ or device_capability == (9, 4):
            return True
        return False

    msg = f"Unsupported GPU arch: {get_gfx_arch_name()} for FP8"
    if throw_on_hip_incompatibility:
        raise RuntimeError(msg)
    logger.error(msg)
    return False
