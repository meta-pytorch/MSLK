# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from mslk.utils.torch.library import load_library_buck

load_library_buck("//mslk/csrc/conv:conv_ops")

# Bypass set_python_module checks for internal; this check is disabled
# by default in OSS PyTorch.
import torch._utils_internal  # noqa: E402

torch._utils_internal.REQUIRES_SET_PYTHON_MODULE = False

from . import _meta  # noqa: F401, E402
