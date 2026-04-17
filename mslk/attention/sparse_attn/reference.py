# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

"""Pure PyTorch reference implementation of NSA (Native Sparse Attention).

This serves as a correctness reference for the optimized FA4-based implementation.
All operations are done in PyTorch without custom kernels.

Reference: arxiv 2502.11089
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor


def nsa_forward_reference(
    Q: Tensor,  # (B, N, H, D)
    K: Tensor,  # (B, N, H_kv, D)
    V: Tensor,  # (B, N, H_kv, D)
    compress_block_size: int = 64,
    num_selected_blocks: int = 16,
    window_size: int = 512,
    W_k_compress: Tensor | None = None,  # (H_kv, D, D)
    W_v_compress: Tensor | None = None,  # (H_kv, D, D)
    gate_proj_weight: Tensor | None = None,  # (H, 3, D)
    causal: bool = True,
    softmax_scale: float | None = None,
    q_tile_size: int = 256,
) -> Tensor:
    """Pure PyTorch NSA forward pass.

    Combines three attention branches:
    1. Compressed attention: Attend over mean-pooled KV blocks with learned projection.
    2. Selected attention: Block-sparse attention over top-k important KV blocks.
    3. Sliding window attention: Local attention within a fixed window.

    All inputs/outputs use (B, N, H, D) layout.

    Returns:
        Output tensor of shape (B, N, H, D).
    """
    B, N, H, D = Q.shape
    H_kv = K.shape[2]
    assert H % H_kv == 0, "H must be divisible by H_kv for GQA"
    groups = H // H_kv

    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(D)

    # Transpose to (B, H, N, D) for matmul convenience
    q = Q.transpose(1, 2).float()  # (B, H, N, D)
    k = K.transpose(1, 2).float()  # (B, H_kv, N, D)
    v = V.transpose(1, 2).float()  # (B, H_kv, N, D)

    # Expand K, V for GQA: (B, H_kv, N, D) -> (B, H, N, D)
    if groups > 1:
        k = k.repeat_interleave(groups, dim=1)
        v = v.repeat_interleave(groups, dim=1)

    # --- Branch 1: Compressed Attention ---
    O_cmp = _compressed_attention(
        q, k, v, compress_block_size, W_k_compress, W_v_compress,
        groups, softmax_scale, causal, B, N, H, H_kv, D,
    )

    # --- Branch 2: Selected (Block-Sparse) Attention ---
    O_slc = _selected_attention(
        q, k, v, compress_block_size, num_selected_blocks,
        W_k_compress, groups, softmax_scale, causal, B, N, H, H_kv, D,
        q_tile_size=q_tile_size,
    )

    # --- Branch 3: Sliding Window Attention ---
    O_sld = _sliding_window_attention(
        q, k, v, window_size, softmax_scale, causal, B, N, H, D,
    )

    # --- Gate and Combine ---
    # All outputs are (B, H, N, D), transpose to (B, N, H, D)
    O_cmp = O_cmp.transpose(1, 2)
    O_slc = O_slc.transpose(1, 2)
    O_sld = O_sld.transpose(1, 2)

    if gate_proj_weight is not None:
        # gate_proj_weight: (H, 3, D)
        # Q: (B, N, H, D) -> compute per-head gates
        gates = torch.einsum("bnhd,hgd->bnhg", Q.float(), gate_proj_weight.float())
        gates = gates.sigmoid()  # (B, N, H, 3)
    else:
        gates = torch.ones(B, N, H, 3, device=Q.device, dtype=torch.float32) / 3.0

    O = (
        gates[..., 0:1] * O_cmp
        + gates[..., 1:2] * O_slc
        + gates[..., 2:3] * O_sld
    )

    return O.to(Q.dtype)


def _compressed_attention(
    q: Tensor, k: Tensor, v: Tensor,
    block_size: int,
    W_k_compress: Tensor | None, W_v_compress: Tensor | None,
    groups: int, softmax_scale: float, causal: bool,
    B: int, N: int, H: int, H_kv: int, D: int,
) -> Tensor:
    """Compressed attention: attend over mean-pooled KV blocks."""
    # Use un-expanded K, V for compression
    k_orig = k[:, ::groups]  # (B, H_kv, N, D)
    v_orig = v[:, ::groups]

    N_cmp = N // block_size
    # Reshape into blocks and mean-pool
    k_blocks = k_orig.reshape(B, H_kv, N_cmp, block_size, D)
    k_cmp = k_blocks.mean(dim=3)  # (B, H_kv, N_cmp, D)
    v_blocks = v_orig.reshape(B, H_kv, N_cmp, block_size, D)
    v_cmp = v_blocks.mean(dim=3)

    # Apply learned projection if provided
    if W_k_compress is not None:
        # W_k_compress: (H_kv, D, D)
        k_cmp = torch.einsum("bhnd,hde->bhne", k_cmp, W_k_compress.float())
    if W_v_compress is not None:
        v_cmp = torch.einsum("bhnd,hde->bhne", v_cmp, W_v_compress.float())

    # Expand for GQA
    if groups > 1:
        k_cmp = k_cmp.repeat_interleave(groups, dim=1)
        v_cmp = v_cmp.repeat_interleave(groups, dim=1)

    # Standard attention on compressed KV: (B, H, N, D) x (B, H, N_cmp, D)
    scores = torch.matmul(q, k_cmp.transpose(-2, -1)) * softmax_scale  # (B, H, N, N_cmp)

    if causal:
        # A query at position i can attend to compressed block j if the block
        # starts at or before position i: j * block_size <= i.
        q_pos = torch.arange(N, device=q.device).unsqueeze(1)  # (N, 1)
        kv_block_start = (torch.arange(N_cmp, device=q.device)) * block_size  # (N_cmp,)
        causal_mask = kv_block_start.unsqueeze(0) > q_pos  # (N, N_cmp)
        scores = scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float("-inf"))

    attn = torch.softmax(scores, dim=-1)
    out = torch.matmul(attn, v_cmp)  # (B, H, N, D)
    return out


def _selected_attention(
    q: Tensor, k: Tensor, v: Tensor,
    block_size: int, num_selected: int,
    W_k_compress: Tensor | None,
    groups: int, softmax_scale: float, causal: bool,
    B: int, N: int, H: int, H_kv: int, D: int,
    q_tile_size: int = 256,
) -> Tensor:
    """Selected block-sparse attention: attend to top-k important KV blocks.

    Aggregates block scores per q_tile_size (matching FA4's CTA tile) and
    selects blocks per query tile rather than per query block.
    """
    N_blocks = N // block_size
    N_q_tiles = N // q_tile_size

    # Score blocks using compressed keys
    k_orig = k[:, ::groups]  # (B, H_kv, N, D)
    k_blocks = k_orig.reshape(B, H_kv, N_blocks, block_size, D)
    k_cmp = k_blocks.mean(dim=3)  # (B, H_kv, N_blocks, D)
    if W_k_compress is not None:
        k_cmp = torch.einsum("bhnd,hde->bhne", k_cmp, W_k_compress.float())
    if groups > 1:
        k_cmp = k_cmp.repeat_interleave(groups, dim=1)

    # Compute block importance scores: (B, H, N, N_blocks)
    block_scores = torch.matmul(q, k_cmp.transpose(-2, -1)) * softmax_scale

    # Aggregate per query-tile (mean over queries in the same tile)
    block_scores_agg = block_scores.reshape(
        B, H, N_q_tiles, q_tile_size, N_blocks
    ).mean(dim=3)
    # (B, H, N_q_tiles, N_blocks)

    if causal:
        # Match score_and_select_blocks: block j is future if
        # j * block_size >= (tile_i + 1) * q_tile_size
        q_tile_end = (torch.arange(N_q_tiles, device=q.device) + 1) * q_tile_size
        kv_block_start = torch.arange(N_blocks, device=q.device) * block_size
        future_mask = kv_block_start.unsqueeze(0) >= q_tile_end.unsqueeze(1)
        block_scores_agg = block_scores_agg.masked_fill(
            future_mask.unsqueeze(0).unsqueeze(0), float("-inf")
        )

    # Select top-k blocks per query tile
    k_actual = min(num_selected, N_blocks)
    _, top_indices = block_scores_agg.topk(k_actual, dim=-1)  # (B, H, N_q_tiles, k)

    # Build block-sparse attention output
    out = torch.zeros(B, H, N, D, device=q.device, dtype=q.dtype)

    for b in range(B):
        for h in range(H):
            for qt in range(N_q_tiles):
                q_start = qt * q_tile_size
                q_end = (qt + 1) * q_tile_size

                # Gather selected KV blocks
                selected_k_list = []
                selected_v_list = []
                for idx in top_indices[b, h, qt]:
                    kv_start = idx.item() * block_size
                    kv_end = kv_start + block_size
                    selected_k_list.append(k[b, h, kv_start:kv_end])
                    selected_v_list.append(v[b, h, kv_start:kv_end])

                if not selected_k_list:
                    continue

                sel_k = torch.cat(selected_k_list, dim=0)  # (k*block_size, D)
                sel_v = torch.cat(selected_v_list, dim=0)

                # Compute attention for this query tile
                q_tile = q[b, h, q_start:q_end]  # (q_tile_size, D)
                scores = torch.matmul(q_tile, sel_k.T) * softmax_scale

                if causal:
                    q_positions = torch.arange(q_start, q_end, device=q.device).unsqueeze(1)
                    kv_positions = []
                    for idx in top_indices[b, h, qt]:
                        kv_s = idx.item() * block_size
                        kv_positions.append(
                            torch.arange(kv_s, kv_s + block_size, device=q.device)
                        )
                    kv_positions = torch.cat(kv_positions)
                    c_mask = kv_positions.unsqueeze(0) > q_positions
                    scores = scores.masked_fill(c_mask, float("-inf"))

                attn = torch.softmax(scores, dim=-1)
                out[b, h, q_start:q_end] = torch.matmul(attn, sel_v)

    return out


def _sliding_window_attention(
    q: Tensor, k: Tensor, v: Tensor,
    window_size: int, softmax_scale: float, causal: bool,
    B: int, N: int, H: int, D: int,
) -> Tensor:
    """Sliding window attention: each query attends to a local window of KV."""
    scores = torch.matmul(q, k.transpose(-2, -1)) * softmax_scale  # (B, H, N, N)

    # Build sliding window + causal mask
    q_pos = torch.arange(N, device=q.device).unsqueeze(1)
    kv_pos = torch.arange(N, device=q.device).unsqueeze(0)

    # Window mask: kv must be within [q_pos - window_size + 1, q_pos] (causal window)
    # or [q_pos - window_size//2, q_pos + window_size//2] (non-causal)
    if causal:
        mask = (kv_pos > q_pos) | (kv_pos < q_pos - window_size + 1)
    else:
        half_w = window_size // 2
        mask = (kv_pos > q_pos + half_w) | (kv_pos < q_pos - half_w)

    scores = scores.masked_fill(mask.unsqueeze(0).unsqueeze(0), float("-inf"))
    attn = torch.softmax(scores, dim=-1)
    out = torch.matmul(attn, v)
    return out
