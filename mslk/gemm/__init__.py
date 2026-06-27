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

# Bypass set_python_module checks for internal; this check is disabled
# by default in OSS PyTorch.
import torch._utils_internal  # noqa: E402

# pyrefly: ignore [missing-attribute]
torch._utils_internal.REQUIRES_SET_PYTHON_MODULE = False

import torch  # noqa: E402

from . import _meta  # noqa: F401, E402

if torch.version.hip is not None:
    from .triton import grouped_gemm as _grouped_gemm, mx8mx4_gemm  # noqa: F401, E402
