# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Which backend serves HSTU cross attention (``mslk::hstu_cross_fwd`` and
``mslk::hstu_cross_bwd``) on ROCm.

- ``mslk``: MSLK's own kernel, the default and the fallback.

Routings are called as ``ROUTING(q, **op_arguments_except_q)``: ``q`` comes
first because :class:`~mslk.aiter.routing.OpRouting` takes the device from the
first argument, and every other argument is passed by its schema name.

Backward backends return ``[dq, dk]`` when ``shared_kv`` (``dk`` holds dK+dV)
and ``[dq, dk, dv]`` otherwise.

Table key: head count, head dims, causal, softmax (1) or SiLU (0), shared K/V,
and the query length rounded up to a power of two (``max_q_len_p2``); bucket:
``max_kv_len`` (``max_seq_len``).
"""

from typing import Any, List, Optional, Sequence, Tuple

import torch
from mslk.aiter.routing import MSLK, OpRouting

OP: str = "hstu_cross"
KEY_COLUMNS: Tuple[str, ...] = (
    "H",
    "D",
    "DV",
    "causal",
    "softmax",
    "shared_kv",
    "max_q_len_p2",
)
BUCKET_COLUMN: str = "max_kv_len"
_TABLE = ("mslk.attention.dispatch", "tables/hstu_cross.csv")


def _mslk_fwd(q: torch.Tensor, **kwargs: Any) -> Tuple[torch.Tensor, torch.Tensor]:
    raise NotImplementedError(
        "MSLK has no cross HSTU forward kernel; no backend can serve this call"
    )


def _mslk_bwd(q: torch.Tensor, **kwargs: Any) -> List[torch.Tensor]:
    raise NotImplementedError(
        "MSLK has no cross HSTU backward kernel; no backend can serve this call"
    )


def _next_pow2(n: int) -> int:
    return 1 if n <= 1 else 1 << (n - 1).bit_length()


def key(
    q: torch.Tensor,
    *,
    k: torch.Tensor,
    v: Optional[torch.Tensor],
    max_seq_len: int,
    max_q_len: Optional[int],
    causal: bool,
    num_softmax_heads: int,
    shared_kv: bool,
    **_: Any,
) -> Tuple[Sequence[object], int]:
    """Table key and bucket. Shared by forward and backward, so both directions
    of a call route to the same backend (the backward consumes the forward's
    ``M``)."""
    value = k if v is None else v
    key_values: List[object] = [
        q.shape[1],
        q.shape[2],
        value.shape[2],
        int(causal),
        int(num_softmax_heads > 0),
        int(shared_kv),
        # The backward's max_q_len is optional; without it the call uses MSLK.
        _next_pow2(max_q_len) if max_q_len is not None else 0,
    ]
    return key_values, max_seq_len


FWD_ROUTING: OpRouting = OpRouting(
    OP,
    *_TABLE,
    key_columns=KEY_COLUMNS,
    bucket_column=BUCKET_COLUMN,
    backends={MSLK: _mslk_fwd},
    key=key,
)

BWD_ROUTING: OpRouting = OpRouting(
    OP,
    *_TABLE,
    key_columns=KEY_COLUMNS,
    bucket_column=BUCKET_COLUMN,
    backends={MSLK: _mslk_bwd},
    key=key,
)

BACKENDS: Tuple[str, ...] = tuple(FWD_ROUTING.backends)
