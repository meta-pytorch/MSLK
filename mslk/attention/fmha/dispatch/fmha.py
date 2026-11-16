# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""OpRouting for FMHA: selects AITER for shapes where it is faster.

The tuning table (``tables/fmha.csv``) is written by
``bench/aiter/tune_fmha.py`` and maps (head_dim, dtype) x seqlen to either
``mslk`` (CK/FlyDSL via the existing priority list) or ``aiter``.

Integration: ``dispatch._dispatch_fw_priority_list`` calls
``should_use_aiter()`` and prepends ``aiter.FwOp`` when the table says so.
If AITER declines via ``not_supported_reasons``, it falls through to CK/FlyDSL.
"""

from typing import Sequence, Tuple

import torch
from mslk.aiter.routing import MSLK, OpRouting

OP: str = "fmha"
AITER: str = "aiter"


def _key(
    head_dim: int, dtype: torch.dtype, num_heads: int, *args: object, **kwargs: object
) -> Tuple[Sequence[object], int]:
    return (head_dim, str(dtype), num_heads), 0


_ROUTING: OpRouting = OpRouting(
    OP,
    "mslk.attention.fmha.dispatch",
    "tables/fmha.csv",
    key_columns=("head_dim", "dtype", "num_heads"),
    bucket_column="seqlen_max",
    backends={MSLK: lambda *a, **kw: None, AITER: lambda *a, **kw: None},
    key=_key,
)


def should_use_aiter(
    head_dim: int,
    dtype: torch.dtype,
    num_heads: int,
    seqlen: int,
    device: torch.device,
) -> bool:
    """Return True when the tuning table prefers AITER for this shape."""
    try:
        from mslk.aiter.common import is_aiter_available, is_aiter_enabled

        if not is_aiter_available() or not is_aiter_enabled("fmha"):
            return False
    except ImportError:
        return False

    try:
        name = _ROUTING.table.lookup(device, (head_dim, str(dtype), num_heads), seqlen)
        return name == AITER
    except Exception:
        return False
