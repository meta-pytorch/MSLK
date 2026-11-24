# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""OpRouting for FMHA: selects AITER for shapes where it is faster.

The tuning table (``tables/fmha.csv``) is written by
``bench/aiter/tune_fmha.py`` and maps (head_dim, dtype, num_heads, seqlen_kv)
x seqlen_q to either ``mslk`` (CK/FlyDSL via the existing priority list) or
``aiter``.
"""

import functools

import torch
from mslk.aiter.routing import ShapeTable

OP: str = "fmha"
AITER: str = "aiter"


@functools.lru_cache(maxsize=1)
def _load_table() -> ShapeTable:
    return ShapeTable.from_package(
        "mslk.attention.fmha.routing",
        "tables/fmha.csv",
        key_columns=("head_dim", "dtype", "num_heads", "seqlen_kv"),
        bucket_column="seqlen_q_max",
    )


def should_use_aiter(
    head_dim: int,
    dtype: "torch.dtype",
    num_heads: int,
    seqlen: int,
    seqlen_kv: int,
    device: "torch.device",
) -> bool:
    """Return True when the tuning table prefers AITER for this shape."""
    try:
        from mslk.aiter.common import is_aiter_available, is_aiter_enabled
    except ImportError:
        return False

    if not is_aiter_available() or not is_aiter_enabled("fmha"):
        return False

    try:
        from mslk.aiter.routing import choose_backend

        device_index = device.index if device.index is not None else 0
        name = choose_backend(
            OP,
            _load_table,
            device_index,
            (head_dim, str(dtype), num_heads, seqlen_kv),
            seqlen,
        )
        return name == AITER
    except ImportError:
        return False
