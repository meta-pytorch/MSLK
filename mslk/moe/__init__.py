# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import mslk  # noqa F401
import torch
from mslk.utils.torch.library import load_library_buck

load_library_buck("//mslk/csrc/moe:moe_ops")

index_shuffling = None
if torch.cuda.is_available():
    index_shuffling = torch.ops.mslk.index_shuffling  # noqa F401

from .activation import silu_mul, silu_mul_quant  # noqa F401
from .gather_scatter import (  # noqa F401
    gather_scale_dense_tokens,
    gather_scale_quant_dense_tokens,
    scatter_add_dense_tokens,
    scatter_add_padded_tokens,
)
from .shuffling import combine_shuffling, split_shuffling  # noqa F401
