# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""aiter backend of ``mslk::bf16bf16bf16_grouped_grad`` (dgrad) on ROCm.

aiter's Triton ``gmm`` computes ``out[g] = X[g] @ W[g].T``. ``W`` is the
``[G, N, K]`` view ``w.permute(0, 2, 1)`` of a contiguous ``[G, K, N]`` weight,
so ``W.permute(0, 2, 1)`` is gmm's non-transposed RHS. ``num_sms`` becomes
``grid_dim``.

Which calls this backend serves is decided in
``mslk/gemm/dispatch/bf16bf16bf16_grouped_grad.py``.
"""

from typing import Optional

import torch
from mslk.aiter.common import import_aiter_module


def bf16bf16bf16_grouped_grad(
    X: torch.Tensor,
    W: torch.Tensor,
    M_sizes: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    num_sms: Optional[int] = None,
) -> torch.Tensor:
    """``mslk::bf16bf16bf16_grouped_grad`` computed by aiter's ``gmm``."""
    gmm = import_aiter_module("aiter.ops.triton.gemm.grouped.gmm").gmm
    total_M, N = X.shape[0], W.shape[1]
    return gmm(
        X,
        W.permute(0, 2, 1),
        M_sizes,
        preferred_element_type=torch.bfloat16,
        existing_out=None if out is None else out.view(total_M, N),
        grid_dim=num_sms,
    )
