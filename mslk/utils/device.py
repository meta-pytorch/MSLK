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
    if torch.cuda.get_device_capability() >= (9, 4):
        return True

    msg = f"Unsupported GPU arch: {get_gfx_arch_name()} for FP8"
    if throw_on_hip_incompatibility:
        raise RuntimeError(msg)
    logger.error(msg)
    return False
