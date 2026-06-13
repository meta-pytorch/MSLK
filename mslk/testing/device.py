# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest
from collections.abc import Callable
from typing import Any

from mslk.utils.device import compute_capability_in, is_cuda


def skip_unless_compute_capability(
    major_min: int,
    major_max: int | None = None,
    reason: str | None = None,
) -> Callable[..., Any]:
    """Skip the test unless on CUDA with compute-capability major in range.

    ``major_max=None`` means no upper bound (e.g. ``major_min=9`` is "SM90+").
    """
    if reason is None:
        bound = (
            f"SM{major_min}0+"
            if major_max is None
            else f"SM{major_min}0-SM{major_max}0"
        )
        reason = f"requires CUDA {bound}"
    return unittest.skipIf(
        not (is_cuda() and compute_capability_in(major_min, major_max)), reason
    )
