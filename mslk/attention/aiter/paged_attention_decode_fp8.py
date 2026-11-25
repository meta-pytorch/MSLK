# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""aiter backend of ``paged_attention_decode_fp8`` (FP8 paged KV cache, one
query token per sequence) on gfx950.

aiter's FlyDSL ``pa_decode`` takes the op's cache layout as is: K as
[blocks, kv_heads, head_size / 16, block_size, 16], V transposed as
[blocks, kv_heads, block_size / 16, head_size, 16], per-token FP32 scales as
[blocks, kv_heads, block_size, 1]. It splits the work by the context lengths
on the GPU (``plan_pa_decode``, no host sync), so a call is CUDA-graph safe;
the plan and the kernel are compiled per batch size. It keeps its own plan
and scratch per call geometry (``_workspaces``), so the op's
``max_context_partition_num``, ``exp_sums``, ``max_logits`` and
``temporary_output`` are left unused.

aiter's ASM ``pa_fwd_asm`` is not used: its softmax scale is fixed.

Which calls this backend serves is decided in
``mslk/attention/dispatch/paged_attention_decode_fp8.py``.
"""

from typing import Any, Dict, Optional, Tuple

import torch
from mslk.aiter.common import import_aiter_module

# One work plan and scratch per call geometry, built outside graph capture
# with aiter's prepare_pa_decode_plan (its tuned work budget, if aiter has one
# for the shape) and refreshed in place on each call, as aiter recommends.
# Calls on one geometry share them, so they must not run concurrently on
# different streams.
_Workspace = Tuple[Any, Dict[str, torch.Tensor]]
_workspaces: Dict[Tuple[object, ...], _Workspace] = {}


def _workspace(
    pa_decode: Any,
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    context_lengths: torch.Tensor,
    block_tables: torch.Tensor,
    softmax_scale: float,
    key_scale: torch.Tensor,
    value_scale: torch.Tensor,
) -> _Workspace:
    kv_heads = key_cache.shape[1]
    key = (query.device, query.shape, query.dtype, kv_heads, key_cache.shape[3])
    workspace = _workspaces.get(key)
    if workspace is not None:
        plan, scratch = workspace
        pa_decode.plan_pa_decode(context_lengths, kv_heads, plan=plan)
        return workspace
    if torch.cuda.is_current_stream_capturing():
        # Plans can only be prepared outside capture; this call builds its own
        # and aiter allocates its scratch in the graph's pool.
        return pa_decode.plan_pa_decode(context_lengths, kv_heads), {}
    tuning = import_aiter_module("aiter.ops.flydsl.pa_decode_tuning")
    plan = tuning.prepare_pa_decode_plan(
        query,
        key_cache,
        value_cache,
        context_lengths,
        block_tables,
        softmax_scale,
        1,  # query_length
        compute_type=key_cache.dtype,
        key_scale=key_scale,
        value_scale=value_scale,
    )
    rows = (kv_heads, plan.capacity, query.shape[1] // kv_heads)
    scratch = {
        "max_logits": torch.empty(rows, dtype=torch.float32, device=query.device),
        "exp_sums": torch.empty(rows, dtype=torch.float32, device=query.device),
        "temporary_output": torch.empty(
            *rows, query.shape[2], dtype=query.dtype, device=query.device
        ),
    }
    _workspaces[key] = (plan, scratch)
    return plan, scratch


def paged_attention_decode_fp8(
    output: torch.Tensor,
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    context_lengths: torch.Tensor,
    block_tables: torch.Tensor,
    softmax_scale: float,
    key_scale: torch.Tensor,
    value_scale: torch.Tensor,
    stream: Optional[torch.cuda.Stream] = None,
) -> str:
    """``paged_attention_decode_fp8`` computed by aiter's FlyDSL ``pa_decode``;
    writes ``output``."""
    pa_decode = import_aiter_module("aiter.ops.flydsl.pa_decode")
    with torch.cuda.stream(stream or torch.cuda.current_stream(query.device)):
        plan, scratch = _workspace(
            pa_decode,
            query,
            key_cache,
            value_cache,
            context_lengths,
            block_tables,
            softmax_scale,
            key_scale,
            value_scale,
        )
        pa_decode.pa_decode(
            output,
            query,
            key_cache,
            value_cache,
            context_lengths,
            block_tables,
            softmax_scale,
            1,  # query_length
            compute_type=key_cache.dtype,
            key_scale=key_scale,
            value_scale=value_scale,
            work_plan=plan,
            **scratch,
        )
    return "aiter"
