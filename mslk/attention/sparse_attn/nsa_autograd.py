# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

"""NSA autograd function: wraps FA4-based forward + backward with activation checkpointing.

Provides NSAFunction(torch.autograd.Function) and a user-facing nsa() function
that supports training (backward pass).
"""

from __future__ import annotations

import math

import torch
from mslk.attention.sparse_attn.compress import (
    fused_compress_kv,
    fused_compress_kv_backward,
)
from mslk.attention.sparse_attn.gating import (
    compute_gates,
    fused_gating_backward,
    gate_and_combine,
)
from mslk.attention.sparse_attn.nsa_forward import (
    _fa4_bwd,
    _fa4_fwd,
    _make_compressed_causal_mask,
)
from mslk.attention.sparse_attn.select import fused_score_and_select_blocks
from mslk.attention.sparse_attn.sparsity_masks import build_fa4_block_sparse_tensors
from torch import Tensor


def _transpose_block_sparse_for_bwd(
    fwd_tensors,
    seqlen_q: int,
    seqlen_k: int,
    n_block_size: int,
):
    """Transpose forward block-sparse tensors for the backward pass.

    Forward: (B, H, n_q_tiles, max_kv_per_tile) — for each Q-tile, which KV-blocks
    Backward: (B, H, n_kv_blocks, n_q_tiles) — for each KV-block, which Q-tiles

    The backward kernel iterates over KV-blocks and needs to know which Q-tiles
    contribute gradients to each KV-block.
    """
    from mslk.attention.flash_attn.block_sparsity import BlockSparseTensorsTorch

    fwd_cnt = fwd_tensors.full_block_cnt  # (B, H, n_q_tiles)
    fwd_idx = fwd_tensors.full_block_idx  # (B, H, n_q_tiles, max_kv_per_tile)
    q_block_size, _ = fwd_tensors.block_size

    B, H, n_q_tiles = fwd_cnt.shape
    n_kv_blocks = (seqlen_k + n_block_size - 1) // n_block_size
    device = fwd_cnt.device

    # Build dense attendance: (B, H, n_q_tiles, n_kv_blocks)
    attendance = torch.zeros(
        B, H, n_q_tiles, n_kv_blocks, dtype=torch.bool, device=device
    )
    max_kv = fwd_idx.shape[3]
    idx_range = torch.arange(max_kv, device=device)
    valid_mask = idx_range < fwd_cnt.unsqueeze(-1)
    valid_indices = fwd_idx.long().clamp(0, n_kv_blocks - 1)
    attendance.scatter_(3, valid_indices, valid_mask)

    # Transpose and pack
    bwd_attendance = attendance.transpose(2, 3).contiguous()
    bwd_cnt = bwd_attendance.sum(dim=-1).to(torch.int32)
    _, sort_indices = bwd_attendance.int().sort(dim=-1, descending=True, stable=True)
    bwd_idx = sort_indices.to(torch.int32)

    bwd_mask_cnt = torch.zeros(B, H, n_kv_blocks, dtype=torch.int32, device=device)
    bwd_mask_idx = torch.zeros(
        B, H, n_kv_blocks, n_q_tiles, dtype=torch.int32, device=device
    )

    return BlockSparseTensorsTorch(
        mask_block_cnt=bwd_mask_cnt,
        mask_block_idx=bwd_mask_idx,
        full_block_cnt=bwd_cnt,
        full_block_idx=bwd_idx,
        block_size=(q_block_size, n_block_size),
    )


class NSAFunction(torch.autograd.Function):
    """Autograd function for NSA with activation checkpointing.

    Forward: runs the FA4-based NSA forward pass.
    Backward: recomputes FA4 branch outputs (activation checkpointing),
    then calls _flash_attn_bwd for each branch.

    Saved for backward (small tensors only):
        - Q, K, V (inputs, already in autograd graph)
        - K_cmp, V_cmp (compressed KV)
        - block_indices, sparse_tensors (discrete, small)
        - gates (B, N, H, 3)
        - Config scalars

    Recomputed during backward (activation checkpointing):
        - O_cmp, lse_cmp, O_slc, lse_slc, O_sld, lse_sld
    """

    @staticmethod
    def forward(
        ctx,
        Q: Tensor,
        K: Tensor,
        V: Tensor,
        W_k_compress: Tensor | None,
        W_v_compress: Tensor | None,
        gate_proj_weight: Tensor | None,
        compress_block_size: int,
        num_selected_blocks: int,
        window_size: int,
        causal: bool,
        softmax_scale: float,
        q_tile_size: int,
        n_block_size: int,
    ) -> Tensor:
        B, N, H, D = Q.shape

        # Step 1: Compress KV
        K_cmp, V_cmp = fused_compress_kv(
            K, V, compress_block_size, W_k_compress, W_v_compress
        )

        # Step 2: Score blocks and select top-k
        block_indices = fused_score_and_select_blocks(
            Q,
            K_cmp,
            num_selected_blocks,
            compress_block_size,
            causal=causal,
            q_tile_size=q_tile_size,
            softmax_scale=softmax_scale,
        )

        # Step 3: Build FA4 sparsity masks
        sparse_tensors = build_fa4_block_sparse_tensors(
            block_indices,
            compress_block_size,
            n_block_size=n_block_size,
            seqlen_k=N,
        )

        # Step 4: Three FA4 branches
        compressed_mask = (
            _make_compressed_causal_mask(compress_block_size) if causal else None
        )
        O_cmp, lse_cmp = _fa4_fwd(
            Q,
            K_cmp,
            V_cmp,
            causal=False,
            softmax_scale=softmax_scale,
            mask_mod=compressed_mask,
        )

        O_slc, lse_slc = _fa4_fwd(
            Q,
            K,
            V,
            causal=causal,
            softmax_scale=softmax_scale,
            block_sparse_tensors=sparse_tensors,
        )

        O_sld, lse_sld = _fa4_fwd(
            Q,
            K,
            V,
            causal=causal,
            softmax_scale=softmax_scale,
            window_size_left=window_size,
            window_size_right=0,
        )

        # Step 5: Compute gates and combine
        gates = compute_gates(Q, gate_proj_weight)
        O = gate_and_combine(O_cmp, O_slc, O_sld, gates)

        # Save for backward — small tensors + inputs only
        # (activation checkpointing: we do NOT save O_cmp, O_slc, O_sld, lse_*)
        ctx.save_for_backward(
            Q,
            K,
            V,
            K_cmp,
            V_cmp,
            gates,
            gate_proj_weight,
            W_k_compress,
            W_v_compress,
        )
        # Save block sparsity info (non-tensor or small)
        ctx.block_indices = block_indices
        ctx.sparse_tensors = sparse_tensors
        ctx.compress_block_size = compress_block_size
        ctx.num_selected_blocks = num_selected_blocks
        ctx.window_size = window_size
        ctx.causal = causal
        ctx.softmax_scale = softmax_scale
        ctx.q_tile_size = q_tile_size
        ctx.n_block_size = n_block_size

        return O

    @staticmethod
    def backward(ctx, dO: Tensor):
        (
            Q,
            K,
            V,
            K_cmp,
            V_cmp,
            gates,
            gate_proj_weight,
            W_k_compress,
            W_v_compress,
        ) = ctx.saved_tensors

        causal = ctx.causal
        softmax_scale = ctx.softmax_scale
        compress_block_size = ctx.compress_block_size
        window_size = ctx.window_size
        sparse_tensors = ctx.sparse_tensors

        # --- Backward: sequential per-branch to minimize peak memory ---
        # Without gate weights, dO_i = g_i * dO (no O_i needed).
        # With gate weights, we need all O_i for dgate — fall back to
        # recomputing all three upfront.

        compressed_mask = (
            _make_compressed_causal_mask(compress_block_size) if causal else None
        )

        dQ_gate = None
        dgate_proj_weight = None

        if gate_proj_weight is not None:
            # Gate weights present — need all O_i for gating backward.
            # Recompute all three forwards (existing behavior).
            O_cmp, lse_cmp = _fa4_fwd(
                Q,
                K_cmp,
                V_cmp,
                causal=False,
                softmax_scale=softmax_scale,
                mask_mod=compressed_mask,
            )
            O_slc, lse_slc = _fa4_fwd(
                Q,
                K,
                V,
                causal=causal,
                softmax_scale=softmax_scale,
                block_sparse_tensors=sparse_tensors,
            )
            O_sld, lse_sld = _fa4_fwd(
                Q,
                K,
                V,
                causal=causal,
                softmax_scale=softmax_scale,
                window_size_left=window_size,
                window_size_right=0,
            )

            dO_cmp, dO_slc, dO_sld, dQ_gate, dgate_proj_weight = fused_gating_backward(
                Q,
                dO,
                O_cmp,
                O_slc,
                O_sld,
                gates,
                gate_proj_weight,
            )
        else:
            # No gate weights — dO_i = g_i * dO, independent of O_i.
            # Process each branch sequentially to minimize peak memory.
            g = gates.float()
            dO_f = dO.float()
            dO_cmp = (g[..., 0:1] * dO_f).to(dO.dtype)
            dO_slc = (g[..., 1:2] * dO_f).to(dO.dtype)
            dO_sld = (g[..., 2:3] * dO_f).to(dO.dtype)
            del g, dO_f

            # Recompute and backward each branch one at a time.
            lse_cmp = lse_slc = lse_sld = None
            O_cmp = O_slc = O_sld = None

        # --- Branch 1: Compressed attention ---
        if O_cmp is None:
            O_cmp, lse_cmp = _fa4_fwd(
                Q,
                K_cmp,
                V_cmp,
                causal=False,
                softmax_scale=softmax_scale,
                mask_mod=compressed_mask,
            )
        dQ_cmp, dK_cmp, dV_cmp = _fa4_bwd(
            Q,
            K_cmp,
            V_cmp,
            O_cmp,
            dO_cmp,
            lse_cmp,
            softmax_scale=softmax_scale,
            causal=False,
            mask_mod=compressed_mask,
        )
        del O_cmp, lse_cmp, dO_cmp

        # --- Branch 2: Selected attention (block-sparse) ---
        if O_slc is None:
            O_slc, lse_slc = _fa4_fwd(
                Q,
                K,
                V,
                causal=causal,
                softmax_scale=softmax_scale,
                block_sparse_tensors=sparse_tensors,
            )
        bwd_sparse_tensors = _transpose_block_sparse_for_bwd(
            sparse_tensors,
            seqlen_q=Q.shape[1],
            seqlen_k=K.shape[1],
            n_block_size=ctx.n_block_size,
        )
        dQ_slc, dK_slc, dV_slc = _fa4_bwd(
            Q,
            K,
            V,
            O_slc,
            dO_slc,
            lse_slc,
            softmax_scale=softmax_scale,
            causal=causal,
            block_sparse_tensors=bwd_sparse_tensors,
        )
        del O_slc, lse_slc, dO_slc, bwd_sparse_tensors

        # --- Branch 3: Sliding window attention ---
        if O_sld is None:
            O_sld, lse_sld = _fa4_fwd(
                Q,
                K,
                V,
                causal=causal,
                softmax_scale=softmax_scale,
                window_size_left=window_size,
                window_size_right=0,
            )
        dQ_sld, dK_sld, dV_sld = _fa4_bwd(
            Q,
            K,
            V,
            O_sld,
            dO_sld,
            lse_sld,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size_left=window_size,
            window_size_right=0,
        )
        del O_sld, lse_sld, dO_sld

        # --- Stage 3: Compression backward ---
        # dK_cmp, dV_cmp are gradients w.r.t. compressed KV
        # CuteDSL kernel for mean-pool scatter, PyTorch for projection
        dK_compress, dV_compress, dW_k, dW_v = fused_compress_kv_backward(
            dK_cmp,
            dV_cmp,
            K,
            V,
            compress_block_size,
            W_k_compress,
            W_v_compress,
        )

        # --- Stage 4: Gradient accumulation ---
        dQ = dQ_cmp + dQ_slc + dQ_sld
        if dQ_gate is not None:
            dQ = dQ + dQ_gate

        dK = dK_slc + dK_sld + dK_compress
        dV = dV_slc + dV_sld + dV_compress

        # Return gradients matching forward() args order:
        # Q, K, V, W_k_compress, W_v_compress, gate_proj_weight, + 7 non-tensor args
        return (
            dQ,
            dK,
            dV,
            dW_k,
            dW_v,
            dgate_proj_weight,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


def nsa(
    Q: Tensor,  # (B, N, H, D) or (total_tokens, H, D) for varlen
    K: Tensor,  # (B, N, H_kv, D) or (total_tokens, H_kv, D)
    V: Tensor,  # (B, N, H_kv, D) or (total_tokens, H_kv, D)
    compress_block_size: int = 64,
    num_selected_blocks: int = 16,
    window_size: int = 512,
    W_k_compress: Tensor | None = None,  # (H_kv, D, D)
    W_v_compress: Tensor | None = None,  # (H_kv, D, D)
    gate_proj_weight: Tensor | None = None,  # (H, 3, D)
    causal: bool = True,
    softmax_scale: float | None = None,
    q_tile_size: int = 256,
    n_block_size: int = 128,
    cu_seqlens: Tensor | None = None,  # (batch_size + 1,) int32 for varlen
    max_seqlen: int | None = None,
) -> Tensor:
    """NSA with autograd support (forward + backward).

    Same interface as nsa_forward, but supports training via torch.autograd.
    Uses FA4 for both forward and backward passes, with activation checkpointing.
    Supports both fixed-length (4D) and variable-length (3D + cu_seqlens) inputs.

    Args:
        Q: Query tensor, (B, N, H, D) or (total_tokens, H, D) for varlen.
        K: Key tensor, (B, N, H_kv, D) or (total_tokens, H_kv, D).
        V: Value tensor, (B, N, H_kv, D) or (total_tokens, H_kv, D).
        compress_block_size: Number of KV positions per block for compression.
        num_selected_blocks: Number of KV blocks to select per query tile.
        window_size: Sliding window size (left window, causal).
        W_k_compress: Per-head key compression weights, shape (H_kv, D, D).
        W_v_compress: Per-head value compression weights, shape (H_kv, D, D).
        gate_proj_weight: Gate projection weights, shape (H, 3, D).
        causal: Whether to use causal masking.
        softmax_scale: Attention scaling factor (default: 1/sqrt(D)).
        q_tile_size: Query tile size for block selection.
        n_block_size: FA4 KV block size (128 for SM100).
        cu_seqlens: Cumulative sequence lengths for varlen, shape (B+1,) int32.
        max_seqlen: Maximum sequence length for varlen padding.

    Returns:
        Output tensor, same shape as Q.
    """
    is_varlen = cu_seqlens is not None

    if is_varlen:
        # Varlen: pad 3D → 4D, run fixed-length NSA autograd, unpad
        assert Q.dim() == 3, f"Varlen requires 3D input, got {Q.dim()}D"
        H = Q.shape[1]
        H_kv = K.shape[1]
        D_dim = Q.shape[2]
        batch_size = cu_seqlens.shape[0] - 1
        seqlens = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()

        if max_seqlen is None:
            max_seqlen = max(seqlens)
        pad_N = ((max_seqlen + q_tile_size - 1) // q_tile_size) * q_tile_size

        Q_pad = Q.new_zeros(batch_size, pad_N, H, D_dim)
        K_pad = K.new_zeros(batch_size, pad_N, H_kv, D_dim)
        V_pad = V.new_zeros(batch_size, pad_N, H_kv, D_dim)
        for i, slen in enumerate(seqlens):
            s = cu_seqlens[i].item()
            Q_pad[i, :slen] = Q[s : s + slen]
            K_pad[i, :slen] = K[s : s + slen]
            V_pad[i, :slen] = V[s : s + slen]

        # Enable gradients on padded tensors
        if Q.requires_grad:
            Q_pad.requires_grad_(True)
        if K.requires_grad:
            K_pad.requires_grad_(True)
        if V.requires_grad:
            V_pad.requires_grad_(True)

        O_pad = nsa(
            Q_pad, K_pad, V_pad,
            compress_block_size=compress_block_size,
            num_selected_blocks=num_selected_blocks,
            window_size=window_size,
            W_k_compress=W_k_compress,
            W_v_compress=W_v_compress,
            gate_proj_weight=gate_proj_weight,
            causal=causal,
            softmax_scale=softmax_scale,
            q_tile_size=q_tile_size,
            n_block_size=n_block_size,
        )

        # Unpad — use indexing that preserves autograd
        O_parts = []
        for i, slen in enumerate(seqlens):
            O_parts.append(O_pad[i, :slen])
        return torch.cat(O_parts, dim=0)

    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(Q.shape[-1])

    return NSAFunction.apply(
        Q,
        K,
        V,
        W_k_compress,
        W_v_compress,
        gate_proj_weight,
        compress_block_size,
        num_selected_blocks,
        window_size,
        causal,
        softmax_scale,
        q_tile_size,
        n_block_size,
    )
