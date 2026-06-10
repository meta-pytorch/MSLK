# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

from __future__ import annotations

import torch
from mslk.quantize.triton.legacy.quantize import triton_quantize_nvfp4


def get_nvfp4_global_scales_naive(
    xs: list[torch.Tensor], ws: list[torch.Tensor]
) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
    """
    Get global scales for each tensor in xs and ws.
    This is done "naively" (not efficiently with a kernel). This function is used in unit tests or debugging.
    """
    global_scales = []
    x_global_scales = []
    w_global_scales = []

    for x, w in zip(xs, ws):
        # pyre-ignore
        x_global_scale: torch.Tensor = (448.0 * 6.0) / torch.amax(
            torch.abs(x.flatten()), dim=-1
        ).to(torch.float32)
        # pyre-ignore
        w_global_scale: torch.Tensor = (448.0 * 6.0) / torch.amax(
            torch.abs(w.flatten()), dim=-1
        ).to(torch.float32)
        # pyre-ignore
        global_scale: torch.Tensor = 1 / (x_global_scale * w_global_scale)

        global_scales.append(global_scale)
        x_global_scales.append(x_global_scale)
        w_global_scales.append(w_global_scale)

    return global_scales, x_global_scales, w_global_scales


def quantize_nvfp4_naive(
    xs: list[torch.Tensor], global_scales: list[torch.Tensor]
) -> tuple[
    list[torch.Tensor],
    list[torch.Tensor],
]:
    """
    Quantize A to NVFP4 format.
    This is done "naively" using a kernel for each group. This function is largely used in unit tests or debugging.
    """
    xqs, x_scales = zip(
        *(
            triton_quantize_nvfp4(x, global_scale)
            for x, global_scale in zip(xs, global_scales)
        )
    )

    return xqs, x_scales
