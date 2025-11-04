# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import os
import subprocess
from typing import Tuple

import mslk
import torch

################################################################################
# Unit test skip attributes for environments
################################################################################

gpu_unavailable: Tuple[bool, str] = (
    not torch.cuda.is_available() or torch.cuda.device_count() == 0,
    "GPU is not available or no GPUs detected",
)

running_in_github: Tuple[bool, str] = (
    os.getenv("GITHUB_ENV") is not None,
    "Test fails or hangs when run in the GitHub runners",
)

running_in_oss: Tuple[bool, str] = (
    # pyre-ignore [16]
    getattr(mslk, "open_source", False),
    "Test is currently known to fail in OSS mode",
)

################################################################################
# Unit test skip attributes for platforms
################################################################################

running_on_arm: Tuple[bool, str] = (
    subprocess.run(["uname", "-m"], stdout=subprocess.PIPE)
    .stdout.decode("utf-8")
    .strip()
    == "aarch64",
    "Test is currently known to fail when running on ARM platform",
)

running_on_cuda: Tuple[bool, str] = (
    torch.cuda.is_available()
    and torch.cuda.device_count() > 0
    and torch.version.hip is not None,
    "Test currently doesn't work on the ROCm stack",
)

running_on_rocm: Tuple[bool, str] = (
    torch.cuda.is_available()
    and torch.cuda.device_count() > 0
    and torch.version.hip is not None,
    "Test currently doesn't work on the ROCm stack",
)
