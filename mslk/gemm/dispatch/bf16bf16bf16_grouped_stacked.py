# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Which backend serves ``mslk::bf16bf16bf16_grouped_stacked`` on ROCm.

- ``mslk``: MSLK's FlyDSL grouped kernel (``mslk/gemm/flydsl/bf16_grouped_gemm.py``),
  or CK where FlyDSL cannot serve the op; the default and the fallback.
- ``aiter``: aiter's Triton ``gmm``
  (``mslk/gemm/aiter/bf16bf16bf16_grouped_stacked.py``), for the shapes
  ``tables/bf16bf16bf16_grouped_stacked.csv`` assigns to it. That table is
  written by ``bench/aiter/tune_bf16bf16bf16_grouped_stacked.py``.
"""

import functools
import logging
from typing import Optional, Sequence, Tuple

import torch
from mslk.aiter.common import is_aiter_enabled
from mslk.aiter.routing import MSLK, OpRouting
from mslk.flydsl.autotune import autotune_enabled, AUTOTUNE_ENV

logger: logging.Logger = logging.getLogger(__name__)

OP: str = "bf16bf16bf16_grouped_stacked"
AITER: str = "aiter"


def warn_if_not_autotuned() -> None:
    """Warn once if the table routes calls while MSLK's FlyDSL kernel runs its
    default tile: the table was tuned against the autotuned kernel."""
    if not autotune_enabled() and is_aiter_enabled(OP):
        _warn_not_autotuned()


@functools.cache
def _warn_not_autotuned() -> None:
    logger.warning(
        "%s is routed by a table tuned with %s set, but it is not set, so MSLK's "
        "kernel runs its default tile; set %s=1 for the table's choices to hold.",
        OP,
        AUTOTUNE_ENV,
        AUTOTUNE_ENV,
    )


# Backends are imported on first use: importing mslk.gemm must not pull in
# FlyDSL or Triton on CPU-only nodes, nor aiter anywhere.
def _mslk(
    X: torch.Tensor,
    W: torch.Tensor,
    M_sizes: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    num_sms: Optional[int] = None,
) -> torch.Tensor:
    from mslk import gemm

    return gemm._bf16bf16bf16_grouped_stacked_impl(X.get_device())(
        X, W, M_sizes, out, num_sms
    )


def _aiter(
    X: torch.Tensor,
    W: torch.Tensor,
    M_sizes: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    num_sms: Optional[int] = None,
) -> torch.Tensor:
    from mslk.gemm.aiter import bf16bf16bf16_grouped_stacked as aiter_stacked

    return aiter_stacked.bf16bf16bf16_grouped_stacked(X, W, M_sizes, out, num_sms)


def _aiter_accepts(
    X: torch.Tensor,
    W: torch.Tensor,
    M_sizes: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    num_sms: Optional[int] = None,
) -> bool:
    # The op's registration has checked dtypes and shapes. aiter compares
    # strides exactly; is_contiguous() ignores size-1 dims.
    total_M, K = X.shape
    G, N, _ = W.shape
    return total_M > 0 and X.stride() == (K, 1) and W.stride() == (N * K, K, 1)


def _key(
    X: torch.Tensor, W: torch.Tensor, *args: object, **kwargs: object
) -> Tuple[Sequence[object], int]:
    return tuple(W.shape), X.shape[0]


#: ``mslk::bf16bf16bf16_grouped_stacked`` on ROCm: call it with the op's arguments.
ROUTING: OpRouting = OpRouting(
    OP,
    "mslk.gemm.dispatch",
    "tables/bf16bf16bf16_grouped_stacked.csv",
    key_columns=("G", "N", "K"),
    bucket_column="M_max",
    backends={MSLK: _mslk, AITER: _aiter},
    key=_key,
    accepts={AITER: _aiter_accepts},
)
BACKENDS: Tuple[str, ...] = tuple(ROUTING.backends)
