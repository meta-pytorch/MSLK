# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

from mslk.utils.torch.library import load_library_buck

from . import cutlass_blackwell_fmha_custom_op  # noqa: F401
from .cutlass_blackwell_fmha_interface import (  # noqa: F401
    _cutlass_blackwell_fmha_forward,
    cutlass_blackwell_fmha_decode_forward,
    cutlass_blackwell_fmha_func,
)

load_library_buck(
    "//mslk/csrc/attention/cuda/cutlass_blackwell_fmha:blackwell_attention_ops_gpu"
)

# Note: _cutlass_blackwell_fmha_forward is an internal function (indicated by leading underscore)
# that is exported here specifically for testing purposes. It allows tests to access the LSE
# (log-sum-exp) values returned by the forward pass without modifying the public API.
# Production code should use cutlass_blackwell_fmha_func instead.
__all__ = [
    "_cutlass_blackwell_fmha_forward",
    "cutlass_blackwell_fmha_decode_forward",
    "cutlass_blackwell_fmha_func",
]
