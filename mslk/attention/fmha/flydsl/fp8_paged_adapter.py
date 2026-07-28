# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""On-the-fly adapter: dense f16/bf16 KV -> native-fp8 paged decode.

Bridges the dense padded KV cache (`[B, padding, G, Hkv, D]`) the CK decoder ops
pass to the fp8 kernel's native-fp8 paged cache by quantizing + paging per call.
Quant/repack cost is paid every call (no persistent fp8 cache); benchmarks should
account for it separately.

Only decode (`q_seqlen == 1`), head_dim % 16 == 0, gfx950.
"""

from __future__ import annotations

from typing import Tuple

import torch

_FP8_DTYPE = torch.float8_e4m3fn  # OCP e4m3fn — the correct gfx950 fp8 format.
_ELEMS_PER_VEC = 16  # 16 fp8 bytes per 128-bit vector (kernel's head-dim packing).


def _pertoken_quant_symmetric(
    x: torch.Tensor, fp8_dtype: torch.dtype = _FP8_DTYPE
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Symmetric per-token (last-dim) fp8 quant.  Returns (xq_fp8, scale_f32)."""
    fmax = torch.finfo(fp8_dtype).max
    amax = x.abs().amax(dim=-1, keepdim=True).clamp_min(1e-12).to(torch.float32)
    scale = amax / fmax
    xq = (x.to(torch.float32) / scale).clamp(-fmax, fmax).to(fp8_dtype)
    return xq, scale


def dense_kv_to_fp8_paged(
    key: torch.Tensor,    # [B, padding, G, Hkv, D] f16/bf16
    value: torch.Tensor,  # [B, padding, G, Hkv, D] f16/bf16
    block_size: int = 16,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantize + page a dense per-batch KV cache into the fp8 kernel's layout.

    Blocks are packed batch-major and contiguous, so `block_tables` is the identity.

    Returns (key_cache, value_cache_shuffled, key_scale, value_scale, block_tables):
      * key_cache          : [num_blocks, Hkv, D//16, block_size, 16] fp8
      * value_cache_shuffled : [num_blocks, Hkv, block_size//16, D, 16] fp8 (trans_v)
      * key_scale/value_scale: [num_blocks, Hkv, block_size, 1] f32 (per-token)
      * block_tables       : [B, blocks_per_seq] int32 (identity, contiguous)
    """
    B, padding, G, Hkv, D = key.shape
    assert D % _ELEMS_PER_VEC == 0, f"head_dim {D} must be a multiple of 16"
    assert padding % block_size == 0, (
        f"padding {padding} must be a multiple of block_size {block_size}"
    )
    # GQA: fold (B, G) into the batch/sequence axis (one "sequence" per KV head,
    # B*G folded sequences); the query path folds the same way so group g's query
    # heads pair with KV group g.
    from .pa_decode_fp8 import KV_COMPUTE_BLOCK

    BG = B * G
    dev = key.device

    # GOTCHA: the kernel reads KV_COMPUTE_BLOCK // block_size block-table entries per
    # partition; a context shorter than one partition would read block_tables/cache out
    # of bounds -> GPU fault. Pad each seq's block count up to one full partition (extra
    # tokens masked by context_lengths, never used) to keep every read in bounds.
    min_blocks_per_seq = KV_COMPUTE_BLOCK // block_size
    blocks_per_seq = max(padding // block_size, min_blocks_per_seq)
    padded = blocks_per_seq * block_size
    num_blocks = BG * blocks_per_seq

    # [B, padding, G, Hkv, D] -> [B*G, padding, Hkv, D] (group folded into batch/seq).
    kbg = key.permute(0, 2, 1, 3, 4).reshape(BG, padding, Hkv, D)
    vbg = value.permute(0, 2, 1, 3, 4).reshape(BG, padding, Hkv, D)
    if padded != padding:
        # GOTCHA: pad with ONES not zeros. An all-zero token quantizes to a ~0 scale and
        # the kernel can hit inf/NaN dequantizing it BEFORE context_lengths masks it out.
        pad_k = kbg.new_ones(BG, padded - padding, Hkv, D)
        kbg = torch.cat([kbg, pad_k], dim=1)
        vbg = torch.cat([vbg, pad_k], dim=1)
    # (B*G, padded) -> (num_blocks, block_size).
    k = kbg.reshape(num_blocks, block_size, Hkv, D).permute(0, 2, 1, 3).contiguous()
    v = vbg.reshape(num_blocks, block_size, Hkv, D).permute(0, 2, 1, 3).contiguous()

    # Per-token symmetric quant over D.
    qk, ks = _pertoken_quant_symmetric(k)  # qk [nb,Hkv,bs,D], ks [nb,Hkv,bs,1]
    qv, vs = _pertoken_quant_symmetric(v)

    # Key cache layout: [num_blocks, Hkv, D//16, block_size, 16].
    key_cache = (
        qk.view(num_blocks, Hkv, block_size, D // _ELEMS_PER_VEC, _ELEMS_PER_VEC)
        .permute(0, 1, 3, 2, 4)
        .contiguous()
    )
    # Value cache: first [num_blocks, Hkv, D, block_size], then trans_v shuffle to 5D.
    qv_t = qv.permute(0, 1, 3, 2).contiguous()  # [nb, Hkv, D, bs]
    value_cache = (
        qv_t.view(num_blocks, Hkv, D, block_size // _ELEMS_PER_VEC, _ELEMS_PER_VEC)
        .permute(0, 1, 3, 2, 4)
        .contiguous()
    )

    # Scales must be the kernel's [num_blocks, Hkv, block_size, 1] layout, contiguous
    # (strides (Hkv*bs, bs, 1, 1)).
    key_scale = ks.contiguous()
    value_scale = vs.contiguous()

    # Identity block_tables: folded seq bg (= b*G + g) owns blocks [bg*bps, (bg+1)*bps).
    block_tables = (
        torch.arange(num_blocks, dtype=torch.int32, device=dev)
        .view(BG, blocks_per_seq)
        .contiguous()
    )
    return key_cache, value_cache, key_scale, value_scale, block_tables


def fp8_paged_decode_from_dense(
    query: torch.Tensor,   # [B, q_seqlen, G, Hq, D] f16/bf16
    key: torch.Tensor,     # [B, padding, G, Hkv, D] f16/bf16
    value: torch.Tensor,   # [B, padding, G, Hkv, D] f16/bf16
    seq_positions: torch.Tensor,  # [B] int32 context lengths (or None)
    scale: float,
    *,
    block_size: int = 16,
) -> torch.Tensor:
    """Run fp8 paged decode against a dense f16/bf16 KV cache (quantized per call).

    Returns output shaped like the dense query heads: ``[B, q_seqlen, G, Hq, D]``.
    """
    from .pa_decode_fp8 import pa_decode_ps_launch

    B, q_seqlen, G, Hq, D = query.shape
    assert q_seqlen == 1, f"fp8 paged decode supports q_seqlen=1, got {q_seqlen}"
    _, padding, _, Hkv, _ = key.shape
    dev = query.device
    BG = B * G

    key_cache, value_cache, key_scale, value_scale, block_tables = dense_kv_to_fp8_paged(
        key, value, block_size=block_size
    )

    # GQA: fold (B, G) into the sequence axis (matching dense_kv_to_fp8_paged's B*G
    # paging); context_lengths must be replicated across the G groups per batch element.
    if seq_positions is None:
        context_lengths = torch.full((BG,), padding, dtype=torch.int32, device=dev)
    else:
        # seq_positions [B] -> [B, G] -> [B*G]
        context_lengths = (
            seq_positions.to(torch.int32).view(B, 1).expand(B, G).reshape(BG).contiguous()
        )

    # Kernel query layout [num_seqs=B*G, Hq, D]: fold group into the sequence axis to
    # pair with KV group g.
    q_flat = query.reshape(BG, Hq, D).contiguous()
    out = torch.zeros(BG, Hq, D, dtype=query.dtype, device=dev)

    pa_decode_ps_launch(
        out,
        q_flat,
        key_cache,
        value_cache,
        context_lengths,
        scale,
        key_scale=key_scale,
        value_scale=value_scale,
        block_tables=block_tables,
        max_context_partition_num=0,
    )
    # [B*G, Hq, D] -> [B, q_seqlen, G, Hq, D]
    return out.view(B, q_seqlen, G, Hq, D)
