# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""aiter backend of ``mslk::bf16bf16bf16_grouped_stacked`` on ROCm.

aiter's Triton ``gmm`` computes ``out[g] = X[g] @ W[g].T`` with groups packed
along M. ``W`` ([G, N, K], row-major) is passed as ``W.transpose(1, 2)``,
gmm's transposed layout 1, which unlike layout 2 is unambiguous when K == N.
``out`` is viewed as ``[total_M, N]`` and ``num_sms`` becomes ``grid_dim``.

Which calls this backend serves is decided in
``mslk/gemm/dispatch/bf16bf16bf16_grouped_stacked.py``.
"""

from typing import Optional

import torch
from mslk.aiter.common import import_aiter_module


def bf16bf16bf16_grouped_stacked(
    X: torch.Tensor,
    W: torch.Tensor,
    M_sizes: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    num_sms: Optional[int] = None,
) -> torch.Tensor:
    """``mslk::bf16bf16bf16_grouped_stacked`` computed by aiter's ``gmm``."""
    gmm = import_aiter_module("aiter.ops.triton.gemm.grouped.gmm").gmm
    total_M, N = X.shape[0], W.shape[1]
    return gmm(
        X,
        W.transpose(1, 2),
        M_sizes,
        preferred_element_type=torch.bfloat16,
        existing_out=None if out is None else out.view(total_M, N),
        grid_dim=num_sms,
    )
