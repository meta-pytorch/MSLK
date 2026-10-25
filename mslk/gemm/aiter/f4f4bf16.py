# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""aiter backend of ``mslk::f4f4bf16`` (MXFP4 x MXFP4 -> BF16) on ROCm.

aiter's ``gemm_afp4wfp4`` computes what ``f4f4bf16`` computes on ROCm: FP4
values packed two per byte (low nibble first), one E8M0 scale per 32 elements
of K, BF16 output. Its outputs are bitwise identical to MSLK's Triton kernel's,
so the op's arguments go to it unchanged.

Its Gluon kernel (``backend="gluon"``) is not used: it is not aiter's default
on gfx950, and is 10-100x slower there on plain weights.

Which calls this backend serves is decided in ``mslk/gemm/dispatch/f4f4bf16.py``.
"""

from typing import Optional

import torch
from mslk.aiter.common import import_aiter_module


def f4f4bf16(
    XQ: torch.Tensor,
    WQ: torch.Tensor,
    x_scale: torch.Tensor,
    w_scale: torch.Tensor,
    output: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """``mslk::f4f4bf16`` computed by aiter's ``gemm_afp4wfp4``.

    Standard MXFP4 only (no global scale, 32-element scale blocks), like
    MSLK's own ROCm kernel.
    """
    gemm_afp4wfp4 = import_aiter_module(
        "aiter.ops.triton.gemm.basic.gemm_afp4wfp4"
    ).gemm_afp4wfp4
    # aiter reads these as raw bytes; the views are free and accept both uint8
    # and torch's FP4/E8M0 dtypes, as MSLK's kernel does.
    return gemm_afp4wfp4(
        XQ.view(torch.uint8),
        WQ.view(torch.uint8),
        x_scale.view(torch.uint8),
        w_scale.view(torch.uint8),
        dtype=torch.bfloat16,
        y=output,
    )
