# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""Public dispatcher for the FlyDSL native-fp8 paged-attention decode.

Native-fp8 paged KV with a symmetric per-token scale (vLLM/Gluon layout); distinct
from the Triton int32-packed asymmetric-scale path, hence a separate guarded entry.

Expected inputs (same CUDA device):
  * query       : [num_seqs, num_query_heads, head_size] bf16/f16.
  * key_cache   : [num_blocks, num_kv_heads, head_size // 16, block_size, 16] fp8.
  * value_cache : [num_blocks, num_kv_heads, block_size // 16, head_size, 16] fp8 (transposed).
  * key_scale/value_scale : per-token f32 [num_blocks, num_kv_heads, block_size, 1].
  * block_tables    : [num_seqs, max_blocks_per_seq] int32.
  * context_lengths : [num_seqs] int32.

Only block_size in {16, 64}; query_length must be 1 (decode); gfx950 only.
"""

from __future__ import annotations

from typing import Optional

import torch
from mslk.flydsl.common import is_flydsl_available, require_flydsl


def is_fp8_paged_decode_available() -> bool:
    """True when the FlyDSL native-fp8 paged decode can run on this arch (gfx950)."""
    if not is_flydsl_available():
        return False
    try:
        from flydsl.runtime.device import get_rocm_arch  # pyre-ignore[21]

        return get_rocm_arch().startswith("gfx950")
    except Exception:
        return False


def csr_to_block_tables(
    kv_page_indices: torch.Tensor,  # [total_pages] int32 — flat physical page ids
    kv_indptr: torch.Tensor,  # [num_seqs + 1] int32 — prefix sum of pages/seq
) -> torch.Tensor:
    """Convert ragged CSR paging (kv_page_indices/kv_indptr) into a dense padded
    block_tables[num_seqs, max_blocks_per_seq]. Rows right-padded with 0 (inert:
    the walk is bounded by context_lengths).
    """
    if kv_indptr.dtype != torch.int32:
        kv_indptr = kv_indptr.to(torch.int32)
    if kv_page_indices.dtype != torch.int32:
        kv_page_indices = kv_page_indices.to(torch.int32)
    dev = kv_page_indices.device
    indptr = kv_indptr.to(torch.long)
    num_seqs = indptr.numel() - 1
    counts = indptr[1:] - indptr[:-1]  # pages per sequence
    max_blocks = int(counts.max().item()) if num_seqs > 0 else 0
    max_blocks = max(max_blocks, 1)
    block_tables = torch.zeros((num_seqs, max_blocks), dtype=torch.int32, device=dev)
    for b in range(num_seqs):
        lo = int(indptr[b].item())
        hi = int(indptr[b + 1].item())
        n = hi - lo
        if n > 0:
            block_tables[b, :n] = kv_page_indices[lo:hi]
    return block_tables


def paged_attention_decode_fp8_csr(
    output: torch.Tensor,
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    context_lengths: torch.Tensor,
    kv_page_indices: torch.Tensor,  # [total_pages] int32
    kv_indptr: torch.Tensor,  # [num_seqs + 1] int32
    softmax_scale: float,
    key_scale: torch.Tensor,
    value_scale: torch.Tensor,
    *,
    max_context_partition_num: int = 0,
    exp_sums: Optional[torch.Tensor] = None,
    max_logits: Optional[torch.Tensor] = None,
    temporary_output: Optional[torch.Tensor] = None,
    stream: Optional[object] = None,
) -> str:
    """CSR-paging entry: converts ragged kv_page_indices/kv_indptr then dispatches to
    paged_attention_decode_fp8."""
    block_tables = csr_to_block_tables(kv_page_indices, kv_indptr)
    return paged_attention_decode_fp8(
        output,
        query,
        key_cache,
        value_cache,
        context_lengths,
        block_tables,
        softmax_scale,
        key_scale,
        value_scale,
        max_context_partition_num=max_context_partition_num,
        exp_sums=exp_sums,
        max_logits=max_logits,
        temporary_output=temporary_output,
        stream=stream,
    )


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
    *,
    max_context_partition_num: int = 0,
    exp_sums: Optional[torch.Tensor] = None,
    max_logits: Optional[torch.Tensor] = None,
    temporary_output: Optional[torch.Tensor] = None,
    stream: Optional[object] = None,
) -> str:
    """Run the FlyDSL native-fp8 paged decode (writes into `output`). Guarded wrapper
    around pa_decode_fp8.pa_decode_ps_launch; raises when FlyDSL/arch unavailable."""
    require_flydsl()
    if not is_fp8_paged_decode_available():
        raise RuntimeError(
            "FlyDSL native-fp8 paged decode requires gfx950 (CDNA4). "
            "For the int32-packed Triton fp8 format, use triton_splitk.FwOp instead."
        )
    from .pa_decode_fp8 import pa_decode_ps_launch

    return pa_decode_ps_launch(
        output,
        query,
        key_cache,
        value_cache,
        context_lengths,
        softmax_scale,
        key_scale=key_scale,
        value_scale=value_scale,
        block_tables=block_tables,
        max_context_partition_num=max_context_partition_num,
        exp_sums=exp_sums,
        max_logits=max_logits,
        temporary_output=temporary_output,
        stream=stream,
    )
