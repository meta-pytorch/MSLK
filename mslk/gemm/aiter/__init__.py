# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""aiter backends for MSLK GEMM ops on ROCm.

Each module adapts one MSLK op to aiter's public op for the same computation:
it takes the op's arguments and calls aiter. Which calls an op's aiter backend
serves is decided by the op's dispatch module (``mslk/gemm/dispatch/``).
Importing these modules does not import aiter; see :mod:`mslk.aiter.common`.
"""
