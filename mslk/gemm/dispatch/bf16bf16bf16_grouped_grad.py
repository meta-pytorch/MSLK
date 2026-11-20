# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Which backend serves ``mslk::bf16bf16bf16_grouped_grad`` (dgrad) on ROCm.

- ``mslk``: MSLK's Triton kernel (``grouped_gemm_dgrad`` in
  ``mslk/gemm/triton/grouped_gemm.py``), the default and the fallback.
- ``aiter``: aiter's Triton ``gmm``
  (``mslk/gemm/aiter/bf16bf16bf16_grouped_grad.py``), for the shapes
  ``tables/bf16bf16bf16_grouped_grad.csv`` assigns to it. That table is written
  by ``bench/aiter/tune_bf16bf16bf16_grouped_grad.py``.

aiter does not take calls with K == N (its weight layouts are ambiguous there)
or with other than the expected strides; those go to MSLK's kernel.
"""

from typing import Optional, Sequence, Tuple

import torch
from mslk.aiter.routing import MSLK, OpRouting

OP: str = "bf16bf16bf16_grouped_grad"
AITER: str = "aiter"


# Backends are imported on first use: importing mslk.gemm must not pull in
# Triton on CPU-only nodes, nor aiter anywhere.
def _mslk(
    X: torch.Tensor,
    W: torch.Tensor,
    M_sizes: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    num_sms: Optional[int] = None,
) -> torch.Tensor:
    from mslk.gemm.triton.grouped_gemm import grouped_gemm_dgrad

    return grouped_gemm_dgrad(X, W, M_sizes, out=out, num_sms=num_sms)


def _aiter(
    X: torch.Tensor,
    W: torch.Tensor,
    M_sizes: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    num_sms: Optional[int] = None,
) -> torch.Tensor:
    from mslk.gemm.aiter import bf16bf16bf16_grouped_grad as aiter_grad

    return aiter_grad.bf16bf16bf16_grouped_grad(X, W, M_sizes, out, num_sms)


def _aiter_accepts(
    X: torch.Tensor,
    W: torch.Tensor,
    M_sizes: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    num_sms: Optional[int] = None,
) -> bool:
    if X.dim() != 2 or W.dim() != 3 or M_sizes.dim() != 1:
        return False
    total_M, K = X.shape
    G, N, K_w = W.shape
    # aiter compares strides exactly; is_contiguous() ignores size-1 dims.
    return (
        total_M > 0
        and K == K_w
        and K != N
        and X.dtype == W.dtype == torch.bfloat16
        and M_sizes.shape == (G,)
        and M_sizes.dtype == torch.int64
        and M_sizes.is_contiguous()
        and X.stride() == (K, 1)
        and W.stride() == (K * N, 1, N)
        and (
            out is None
            or (
                out.dtype == torch.bfloat16
                and out.is_contiguous()
                and out.numel() == total_M * N
            )
        )
    )


def _key(
    X: torch.Tensor, W: torch.Tensor, *args: object, **kwargs: object
) -> Tuple[Sequence[object], int]:
    return tuple(W.shape), X.shape[0]


#: ``mslk::bf16bf16bf16_grouped_grad`` on ROCm: call it with the op's arguments.
ROUTING: OpRouting = OpRouting(
    OP,
    "mslk.gemm.dispatch",
    "tables/bf16bf16bf16_grouped_grad.csv",
    key_columns=("G", "N", "K"),
    bucket_column="M_max",
    backends={MSLK: _mslk, AITER: _aiter},
    key=_key,
    accepts={AITER: _aiter_accepts},
)
BACKENDS: Tuple[str, ...] = tuple(ROUTING.backends)
