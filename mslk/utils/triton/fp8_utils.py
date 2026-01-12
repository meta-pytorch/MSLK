# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe
import functools
import logging
import os
from typing import Tuple

import torch
import triton.language as tl  # @manual
from triton.runtime.jit import reinterpret as tl_reinterpret, TensorWrapper  # @manual


running_on_github: bool = os.getenv("GITHUB_ENV") is not None


@functools.lru_cache
def supports_float8_fnuz(throw_on_hip_incompatibility: bool = True) -> bool:
    if torch.version.hip:
        device_capability = torch.cuda.get_device_capability()

        if device_capability < (9, 4):
            gpu_arch = torch.cuda.get_device_properties("cuda").gcnArchName
            msg = f"Unsupported GPU arch: {gpu_arch} for FP8"
            if throw_on_hip_incompatibility:
                raise RuntimeError(msg)
            else:
                logging.error(msg)
                return False

        elif device_capability == (9, 4):
            return True

    return False


def get_fp8_constants() -> Tuple[torch.dtype, tl.dtype, float, float]:
    """
    Helper function to get constant values for the current platform.

    Returns:
        pt_dtype (torch.dtype): The correct torch fp8 datatype.
        tl_dtype (tl.dtype): The correct triton fp8 datatype.
        max_fp8 (float): The maximum reprsentable value for the fp8 datatype.
        eps (float): Minimum clip value to prevent divide by zero.
    """
    if supports_float8_fnuz(throw_on_hip_incompatibility=(not running_on_github)):
        pt_fp8_dtype = torch.float8_e4m3fnuz
        tl_fp8_dtype = tl.float8e4b8
    else:
        pt_fp8_dtype = torch.float8_e4m3fn
        tl_fp8_dtype = tl.float8e4nv

    return pt_fp8_dtype, tl_fp8_dtype, torch.finfo(pt_fp8_dtype).max, 1e-12


def reinterpret_fp8_type(tensor: torch.Tensor, dtype: tl.dtype) -> TensorWrapper:
    """
    Converts tensor to triton fp8 type.

    Args:
        tensor (torch.Tensor): input tensor.
        dtype (tl.dtype): target triton dtype.

    Returns:
        triton.TensorWrapper: fp8 tensor.
    """
    return tl_reinterpret(tensor, dtype=dtype)
