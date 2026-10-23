# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import torch
from mslk.quantize.triton.legacy.fp4_utils import (  # noqa: F401
    dequantize_mx4,
    dequantize_nvfp4,
    fp4_to_float,
    global_scale_nvfp4,
    scale_nvfp4,
)


def dequantize_mx4_unswizzled(
    xq: torch.Tensor, scale: torch.Tensor, group_size: int = 32
) -> torch.Tensor:
    """Dequantize MXFP4 data whose E8M0 scales are in the plain ``[..., K / group_size]``
    layout (as ``triton_quantize_mx4_unpack`` returns them on ROCm), to float32.

    :func:`dequantize_mx4` instead takes CUTLASS's swizzled scale layout. The
    result is exact: every FP4 value times a power of two fits in float32.
    """
    values = fp4_to_float(xq.view(torch.uint8))
    scale = torch.exp2(scale.view(torch.uint8).float() - 127.0)
    return values * scale.repeat_interleave(group_size, dim=-1)[..., : values.shape[-1]]
