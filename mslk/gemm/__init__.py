# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from mslk.utils.torch.library import load_library_buck

load_library_buck("//mslk/csrc/gemm:gemm_ops")

gemm_ops = [
    "//mslk/csrc/gemm/cutlass:cutlass_bf16bf16bf16_grouped_grad",
    "//mslk/csrc/gemm/cutlass:cutlass_bf16bf16bf16_grouped_wgrad",
]
for op in gemm_ops:
    load_library_buck(op)
