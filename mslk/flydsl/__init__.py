# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""FlyDSL support for MSLK.

Importing any part of this package configures the FlyDSL runtime, so that a
kernel is served from the bundled AOT cache and honours the JIT switch
whichever launch helper it goes through. The call belongs here rather than
beside one of those helpers, so that the guarantee does not depend on which of
them a caller happens to use.
"""

from mslk.flydsl.jit import configure_runtime_cache

configure_runtime_cache()
