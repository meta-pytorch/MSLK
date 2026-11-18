# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Which backend serves ``paged_attention_decode_fp8`` on gfx950.

- ``mslk``: MSLK's FlyDSL kernel
  (``mslk/attention/flydsl/decode/pa_decode_fp8.py``), the default and the
  fallback.
- ``aiter``: aiter's FlyDSL ``pa_decode``
  (``mslk/attention/aiter/paged_attention_decode_fp8.py``), for the shapes
  ``tables/paged_attention_decode_fp8.csv`` assigns to it. That table is written
  by ``bench/aiter/tune_paged_attention_decode_fp8.py``.

The table key is the heads, head size, block size and the context the block
table can hold, rounded up to a power of two; the bucket is the batch size.
All come from tensor shapes, never from the context lengths on the GPU, so a
CUDA graph replays the backend it was captured with.
"""

from typing import Any, Optional, Sequence, Tuple

import torch
from mslk.aiter.routing import MSLK, OpRouting

OP: str = "paged_attention_decode_fp8"
AITER: str = "aiter"


# Backends are imported on first use: importing this module must not pull in
# FlyDSL or aiter.
def _mslk(*args: Any, **kwargs: Any) -> str:
    from mslk.attention.flydsl.decode.pa_decode_fp8_dispatch import (
        paged_attention_decode_fp8_flydsl,
    )

    return paged_attention_decode_fp8_flydsl(*args, **kwargs)


def _aiter(
    output: torch.Tensor,
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    context_lengths: torch.Tensor,
    block_tables: torch.Tensor,
    softmax_scale: float,
    key_scale: torch.Tensor,
    value_scale: torch.Tensor,
    *,
    stream: Optional[torch.cuda.Stream] = None,
    **mslk_scratch: Any,
) -> str:
    from mslk.attention.aiter import paged_attention_decode_fp8 as aiter_op

    return aiter_op.paged_attention_decode_fp8(
        output,
        query,
        key_cache,
        value_cache,
        context_lengths,
        block_tables,
        softmax_scale,
        key_scale,
        value_scale,
        stream,
    )


def _aiter_accepts(
    output: torch.Tensor,
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    context_lengths: torch.Tensor,
    block_tables: Optional[torch.Tensor],
    softmax_scale: float,
    key_scale: object,
    value_scale: object,
    *,
    stream: object = None,
    **mslk_scratch: Any,
) -> bool:
    # Calls aiter's pa_decode would reject, and calls unlike those the table
    # was tuned with (it measured bf16 queries and per-token scales only), go
    # to MSLK's kernel.
    if not (
        isinstance(block_tables, torch.Tensor)
        and isinstance(key_scale, torch.Tensor)
        and isinstance(value_scale, torch.Tensor)
        and query.dim() == 3
        and block_tables.dim() == 2
        and key_cache.dim() == value_cache.dim() == 5
    ):
        return False
    batch, _, head_size = query.shape
    num_blocks, kv_heads, _, block_size, _ = key_cache.shape
    tensors = (output, key_cache, value_cache, context_lengths, block_tables)
    return (
        output.shape == query.shape
        and output.dtype == query.dtype == torch.bfloat16
        and key_cache.dtype == value_cache.dtype == torch.float8_e4m3fn
        and context_lengths.dtype == block_tables.dtype == torch.int32
        and key_scale.dtype == value_scale.dtype == torch.float32
        and context_lengths.shape == (batch,)
        and block_tables.shape[0] == batch
        and value_cache.shape[0] <= num_blocks
        and key_scale.shape
        == value_scale.shape
        == (num_blocks, kv_heads, block_size, 1)
        and key_scale.stride() == value_scale.stride()
        and key_scale.stride(2) == 1
        and all(t.device == query.device for t in (*tensors, key_scale, value_scale))
        and query.stride(-1) == output.stride(-1) == 1
        and all(t.is_contiguous() for t in tensors[1:])
        # aiter's limits.
        and (head_size == 64 or head_size % 128 == 0 and head_size <= 1024)
        and 1 <= batch <= 4096
        and (stream is None or isinstance(stream, torch.cuda.Stream))
    )


def _key(
    output: torch.Tensor,
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    context_lengths: torch.Tensor,
    block_tables: Optional[torch.Tensor],
    *args: object,
    **kwargs: object,
) -> Tuple[Sequence[object], int]:
    block_size = key_cache.shape[-2]
    width = 0 if block_tables is None else block_tables.shape[-1]
    capacity = width * block_size
    context_max = max(256, 1 << (capacity - 1).bit_length())
    key = (query.shape[1], key_cache.shape[1], query.shape[-1], block_size, context_max)
    return key, context_lengths.shape[0]


#: ``paged_attention_decode_fp8`` on gfx950: call it with the op's arguments.
ROUTING: OpRouting = OpRouting(
    OP,
    "mslk.attention.dispatch",
    "tables/paged_attention_decode_fp8.csv",
    key_columns=("num_heads", "num_kv_heads", "head_size", "block_size", "context_max"),
    bucket_column="batch_max",
    backends={MSLK: _mslk, AITER: _aiter},
    key=_key,
    accepts={AITER: _aiter_accepts},
)
BACKENDS: Tuple[str, ...] = tuple(ROUTING.backends)
