# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Which backend serves ragged HSTU self-attention (``mslk::hstu_ragged_fwd``
and ``mslk::hstu_ragged_bwd``) on ROCm.

- ``mslk``: MSLK's own kernel, the default and the fallback.

Routings are called as ``ROUTING(q, **op_arguments_except_q)``: ``q`` comes
first because :class:`~mslk.aiter.routing.OpRouting` takes the device from the
first argument, and every other argument is passed by its schema name.

Table key: head count, head dims, mask type, window (``max_attn_len``, 0 = no
window), softmax (1) or SiLU (0), shared K/V; bucket: ``max_seq_len`` (``N``).
Window values are matched exactly; windows the table does not list use MSLK.
"""

from typing import Any, List, Sequence, Tuple

import torch
from mslk.aiter.routing import MSLK, OpRouting

OP: str = "hstu_ragged"
KEY_COLUMNS: Tuple[str, ...] = (
    "H",
    "D",
    "DV",
    "mask",
    "max_attn_len",
    "softmax",
    "shared_kv",
)
BUCKET_COLUMN: str = "max_seq_len"
_TABLE = ("mslk.attention.dispatch", "tables/hstu_ragged.csv")


def _mslk_fwd(q: torch.Tensor, **kwargs: Any) -> Tuple[torch.Tensor, torch.Tensor]:
    raise NotImplementedError(
        "MSLK has no ragged HSTU forward kernel; no backend can serve this call"
    )


def _mslk_bwd(q: torch.Tensor, **kwargs: Any) -> None:
    raise NotImplementedError(
        "MSLK has no ragged HSTU backward kernel; no backend can serve this call"
    )


def key(
    q: torch.Tensor,
    *,
    v: torch.Tensor,
    N: int,
    invalid_attn_mask_type: str,
    max_attn_len: int,
    num_softmax_heads: int,
    shared_kv: bool,
    **_: Any,
) -> Tuple[Sequence[object], int]:
    """Table key and bucket. Shared by forward and backward, so both directions
    of a call route to the same backend (the backward consumes the forward's
    ``M``)."""
    key_values: List[object] = [
        q.shape[1],
        q.shape[2],
        v.shape[2],
        invalid_attn_mask_type,
        max_attn_len,
        int(num_softmax_heads > 0),
        int(shared_kv),
    ]
    return key_values, N


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
