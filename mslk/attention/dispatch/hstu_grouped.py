# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Which backend serves grouped ragged HSTU self-attention
(``mslk::hstu_grouped_fwd`` and ``mslk::hstu_grouped_bwd``) on ROCm.

- ``mslk``: MSLK's own kernel, the default and the fallback.

Routings are called as ``ROUTING(q, **op_arguments_except_q)``: ``q`` comes
first because :class:`~mslk.aiter.routing.OpRouting` takes the device from the
first argument, and every other argument is passed by its schema name.

Table key: head count, head dims, mask type, shared K/V; bucket:
``avg_seq_len``, the packed token count over the sequence count
(``T // (G * Z)``). The grouped API carries per-group maximum lengths and
windows only as device tensors, which routing cannot read without a host
synchronization, so the bucket uses the one length known on the host.
"""

from typing import Any, List, Sequence, Tuple

import torch
from mslk.aiter.routing import MSLK, OpRouting

OP: str = "hstu_grouped"
KEY_COLUMNS: Tuple[str, ...] = ("H", "D", "DV", "mask", "shared_kv")
BUCKET_COLUMN: str = "avg_seq_len"
_TABLE = ("mslk.attention.dispatch", "tables/hstu_grouped.csv")


def _mslk_fwd(q: torch.Tensor, **kwargs: Any) -> torch.Tensor:
    raise NotImplementedError(
        "MSLK has no grouped HSTU forward kernel; no backend can serve this call"
    )


def _mslk_bwd(q: torch.Tensor, **kwargs: Any) -> None:
    raise NotImplementedError(
        "MSLK has no grouped HSTU backward kernel; no backend can serve this call"
    )


def key(
    q: torch.Tensor,
    *,
    k: torch.Tensor,
    v: torch.Tensor,
    Z: int,
    group_offsets: torch.Tensor,
    invalid_attn_mask_type: str,
    **_: Any,
) -> Tuple[Sequence[object], int]:
    """Table key and bucket. Shared by forward and backward, so both directions
    of a call route to the same backend."""
    num_seqs = max(1, (group_offsets.shape[0] - 1) * Z)
    key_values: List[object] = [
        q.shape[1],
        q.shape[2],
        v.shape[2],
        invalid_attn_mask_type,
        int(v is k),
    ]
    return key_values, q.shape[0] // num_seqs


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
