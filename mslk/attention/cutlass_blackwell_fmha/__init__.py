# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

from mslk.utils.torch.library import load_library_buck

from . import cutlass_blackwell_fmha_custom_op  # noqa: F401
from .cutlass_blackwell_fmha_interface import cutlass_blackwell_fmha_func  # noqa: F401

load_library_buck(
    "//mslk/csrc/attention/cuda/cutlass_blackwell_fmha:blackwell_attention_ops_gpu"
)
