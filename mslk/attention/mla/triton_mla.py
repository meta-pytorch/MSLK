# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

"""
Triton MLA (Multi-head Latent Attention) forward op for ROCm.

DeepSeek-V3 style MLA after weight absorption: the K/V projection
matrices are absorbed into Q and output projections, turning multi-head
latent attention into an MQA-like op over a single latent KV head where
K and V share the same storage (V = kv_buffer[:, :, :kv_lora_rank]).

This module provides:
  - ``mla_decode_fwd``: split-K decode (qlen=1) for token generation
  - ``mla_prefill_fwd``: single-stage flash attention for prompt encoding

Both accept a paged ``kv_buffer`` with ``block_tables`` for cache indirection.

Kernel source: mslk/attention/mla/_triton/mla_kernels.py
Reference: ROCm/aimodels agent harness + aiter MLA decode rope kernel

Hardware target: AMD MI300X (gfx942) and MI350X (gfx950).
"""

from __future__ import annotations

import math
from typing import Optional

import torch
from mslk.attention.fmha._triton.available import is_triton_available

if is_triton_available():
    from ._triton.mla_kernels import mla_decode_forward, mla_prefill_forward


# ---------------------------------------------------------------------------
# MLA architectural constants (DeepSeek-V3, post weight-absorption)
# ---------------------------------------------------------------------------

MLA_NUM_HEADS = 128
MLA_NUM_KV_HEADS = 1
MLA_KV_LORA_RANK = 512
MLA_QK_ROPE_HEAD_DIM = 64
MLA_QK_HEAD_DIM = MLA_KV_LORA_RANK + MLA_QK_ROPE_HEAD_DIM  # 576
MLA_V_HEAD_DIM = MLA_KV_LORA_RANK  # 512


def _is_rocm() -> bool:
    return torch.version.hip is not None


def _detect_gfx_arch() -> str:
    """Return the GCN architecture name of the current device."""
    if not _is_rocm() or not torch.cuda.is_available():
        return ""
    return getattr(torch.cuda.get_device_properties(0), "gcnArchName", "")


_CACHED_GFX_ARCH = _detect_gfx_arch()


def _select_max_kv_splits(batch: int) -> int:
    """Choose max split-K count to balance GPU occupancy vs intermediate size."""
    num_cus = 304  # MI300X
    num_head_blks = MLA_NUM_HEADS // 64  # BLOCK_H=64 → 2 head blocks
    grid_no_split = num_head_blks * batch
    if grid_no_split >= num_cus * 2:
        return 1
    if grid_no_split * 2 >= num_cus:
        return 2
    if grid_no_split * 4 >= num_cus:
        return 4
    return 8


def _get_decode_tuning(arch: str) -> dict:
    """Return decode kernel tuning parameters for the given architecture."""
    if "gfx950" in arch:
        return {
            "max_kv_splits": 16,
            "block_h": 16,
            "num_warps": 8,
            "num_stages": 3,
        }
    # gfx942 (MI300X) — BH=64 maximizes KV reuse across heads,
    # stages=4 for deep pipelining
    return {
        "max_kv_splits": None,  # adaptive — set in mla_decode_fwd
        "block_h": 64,
        "num_warps": 4,
        "num_stages": 4,
    }


def _get_prefill_tuning(arch: str) -> dict:
    """Return prefill kernel tuning parameters for the given architecture."""
    if "gfx950" in arch:
        return {
            "tile_size": 16,
            "block_h": 64,
            "block_q": 1,
            "num_warps": 8,
            "num_stages": 1,
        }
    # gfx942 (MI300X) — BH=64 doubles KV reuse
    return {
        "tile_size": 16,
        "block_h": 64,
        "block_q": 1,
        "num_warps": 4,
        "num_stages": 1,
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def mla_decode_fwd(
    query: torch.Tensor,
    kv_buffer: torch.Tensor,
    block_tables: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    seqused_k: torch.Tensor,
    softmax_scale: Optional[float] = None,
) -> torch.Tensor:
    """MLA decode forward pass (split-K, qlen=1).

    Args:
        query: (batch, num_heads, qk_head_dim) — FP8 or BF16.
            After weight absorption: qk_head_dim = kv_lora_rank + qk_rope_head_dim.
        kv_buffer: (num_pages, page_size, num_kv_heads, qk_head_dim) — FP8 or BF16.
            Latent KV cache. K = kv_buffer[..., :qk_head_dim],
            V = kv_buffer[..., :kv_lora_rank]. K and V share storage.
        block_tables: (batch, max_blocks_per_seq) int32 — page table.
        cu_seqlens_q: (batch + 1,) int32 — cumulative query lengths.
        seqused_k: (batch,) int32 — number of KV tokens per sequence.
        softmax_scale: float — default 1/sqrt(qk_head_dim).

    Returns:
        (batch, num_heads, v_head_dim) BF16 attention output.
    """
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(query.shape[-1])

    tuning = _get_decode_tuning(_CACHED_GFX_ARCH)

    if tuning["max_kv_splits"] is None:
        batch = seqused_k.shape[0]
        tuning["max_kv_splits"] = _select_max_kv_splits(batch)

    if query.dtype in (torch.float8_e4m3fnuz, torch.float8_e5m2fnuz):
        tuning["num_stages"] = min(tuning["num_stages"], 1)

    return mla_decode_forward(
        query=query,
        kv_buffer=kv_buffer,
        block_tables=block_tables,
        cu_seqlens_q=cu_seqlens_q,
        seqused_k=seqused_k,
        softmax_scale=softmax_scale,
        **tuning,
    )


def mla_prefill_fwd(
    query: torch.Tensor,
    kv_buffer: torch.Tensor,
    block_tables: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    seqused_k: torch.Tensor,
    softmax_scale: Optional[float] = None,
) -> torch.Tensor:
    """MLA prefill forward pass (single-stage flash attention).

    Args:
        query: (total_tokens, num_heads, qk_head_dim) — FP8 or BF16.
        kv_buffer: (num_blocks, page_size, num_kv_heads, qk_head_dim) — FP8 or BF16.
        block_tables: (num_seqs, max_blocks_per_seq) int32.
        cu_seqlens_q: (num_seqs + 1,) int32 — cumulative query lengths.
        seqused_k: (num_seqs,) int32 — total KV length per sequence.
        softmax_scale: float — default 1/sqrt(qk_head_dim).

    Returns:
        (total_tokens, num_heads, v_head_dim) BF16 attention output.
    """
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(query.shape[-1])

    tuning = _get_prefill_tuning(_CACHED_GFX_ARCH)

    return mla_prefill_forward(
        query=query,
        kv_buffer=kv_buffer,
        block_tables=block_tables,
        cu_seqlens_q=cu_seqlens_q,
        seqused_k=seqused_k,
        softmax_scale=softmax_scale,
        **tuning,
    )
