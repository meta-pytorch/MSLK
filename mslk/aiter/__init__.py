# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""Support for serving MSLK ops from aiter (https://github.com/ROCm/aiter).

aiter is an optional, ROCm-only dependency: MSLK installs and works without it,
and every op keeps its own kernel as the fallback. MSLK is tested against one
exact aiter build, whose version is recorded in ``aiter_version.txt`` next to
this file and read by ``setup.py`` (the ``mslk[aiter]`` extra) and by CI.
"""
