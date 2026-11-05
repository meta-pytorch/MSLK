# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""Which backend serves each GEMM op that has more than one, per call.

One module per op: it lists the op's backends, decides which of them can take
a call's arguments, and picks among those from the op's tuning table in
``tables/`` (see :mod:`mslk.aiter.routing`). The op's registered implementation
calls the module; the backends themselves live with their kind (``triton/``,
``aiter/``, ...). Importing these modules imports no backend.
"""
