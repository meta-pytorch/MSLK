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

        # --- Stage 1: Gating backward ---
        # O = g0*O_cmp + g1*O_slc + g2*O_sld
        # dO_i = gi * dO
        # dgi = (dO * Oi).sum(dim=-1)  (need Oi, so must recompute)

        # --- Recompute FA4 forward outputs (activation checkpointing) ---
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

        # --- Gating backward ---
        dQ_gate = None
        dgate_proj_weight = None
        if gate_proj_weight is not None:
            # Fused CuteDSL kernel for dO branches + dQ_gate,
            # PyTorch einsum for dW_gate (cross-row reduction)
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
            # No gate weights — uniform 1/3 gates, simple scaling
            g = gates.float()
            dO_f = dO.float()
            dO_cmp = (g[..., 0:1] * dO_f).to(dO.dtype)
            dO_slc = (g[..., 1:2] * dO_f).to(dO.dtype)
            dO_sld = (g[..., 2:3] * dO_f).to(dO.dtype)

        # --- Stage 2: Three attention backward passes (FA4 CuteDSL) ---

        # Branch 1: Compressed attention backward
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

        # Branch 2: Selected attention backward via gather/scatter (per-tile)
        # FA4 block-sparse backward is not yet supported on SM100, so we use
        # a PyTorch-based approach: for each query tile, gather selected K/V,
        # compute attention backward, then scatter dK/dV gradients back.
        # Processing one tile at a time avoids O(N_q_tiles * N * D) memory.
        # GQA is handled inside the loop to avoid expanding K/V to H heads.
        block_indices = ctx.block_indices
        q_tile_size = ctx.q_tile_size
        num_sel = ctx.num_selected_blocks
        block_size = compress_block_size

        B_size, N_size, H_size, D = Q.shape
        H_kv = K.shape[2]
        groups = H_size // H_kv

        N_blocks = N_size // block_size
        N_q_tiles = N_size // q_tile_size
        k_actual = min(num_sel, N_blocks)
        kv_len = k_actual * block_size

        # Expand block indices to position indices: (B, H, N_q_tiles, kv_len)
        offsets = torch.arange(block_size, device=Q.device)
        pos_indices = (block_indices.unsqueeze(-1) * block_size + offsets).reshape(
            B_size, H_size, N_q_tiles, kv_len
        )

        # Keep K, V at H_kv heads — no GQA expansion
        k_t = K.transpose(1, 2).float()  # (B, H_kv, N, D)
        v_t = V.transpose(1, 2).float()  # (B, H_kv, N, D)

        # Accumulate dQ at H heads, dK/dV at H_kv heads
        dQ_slc = torch.zeros(
            B_size, H_size, N_size, D, device=Q.device, dtype=torch.float32
        )
        dK_slc_full = torch.zeros(
            B_size, H_kv, N_size, D, device=Q.device, dtype=torch.float32
        )
        dV_slc_full = torch.zeros_like(dK_slc_full)

        q_t = Q.transpose(1, 2).float()  # (B, H, N, D)
        dO_slc_t = dO_slc.transpose(1, 2).float()

        for qt in range(N_q_tiles):
            q_start = qt * q_tile_size
            q_end = q_start + q_tile_size

            # Block indices for this tile: (B, H, kv_len)
            idx_qt = pos_indices[:, :, qt]

            # Gather K, V from H_kv heads, then expand for GQA
            # idx_qt is (B, H, kv_len) but K is (B, H_kv, N, D)
            # Map H→H_kv: idx_kv_qt = idx_qt[:, ::groups] (same indices per group)
            idx_kv_qt = idx_qt[:, ::groups]  # (B, H_kv, kv_len)
            gather_kv = idx_kv_qt.unsqueeze(-1).expand(B_size, H_kv, kv_len, D)
            sel_k_kv = torch.gather(k_t, 2, gather_kv)  # (B, H_kv, kv_len, D)
            sel_v_kv = torch.gather(v_t, 2, gather_kv)

            # Expand gathered K, V to H heads for attention
            if groups > 1:
                sel_k = sel_k_kv.repeat_interleave(groups, dim=1)  # (B, H, kv_len, D)
                sel_v = sel_v_kv.repeat_interleave(groups, dim=1)
            else:
                sel_k = sel_k_kv
                sel_v = sel_v_kv

            # Q and dO for this tile: (B, H, q_tile_size, D)
            q_tile = q_t[:, :, q_start:q_end]
            dO_tile = dO_slc_t[:, :, q_start:q_end]

            # Attention scores: (B, H, q_tile_size, kv_len)
            scores = torch.matmul(q_tile, sel_k.transpose(-2, -1)) * softmax_scale

            if causal:
                q_positions = torch.arange(q_start, q_end, device=Q.device).unsqueeze(1)
                kv_positions = idx_qt.unsqueeze(2)  # (B, H, 1, kv_len)
                c_mask = kv_positions > q_positions.reshape(1, 1, q_tile_size, 1)
                scores = scores.masked_fill(c_mask, float("-inf"))

            attn = torch.softmax(scores, dim=-1)

            # dV = attn^T @ dO
            dv_qt = torch.matmul(attn.transpose(-2, -1), dO_tile)  # (B, H, kv_len, D)

            # dP = dO @ V^T, dS = P * (dP - rowsum) * scale
            dp = torch.matmul(dO_tile, sel_v.transpose(-2, -1))
            rowsum = (dO_tile * torch.matmul(attn, sel_v)).sum(dim=-1, keepdim=True)
            ds = attn * (dp - rowsum) * softmax_scale

            # dQ = dS @ K
            dQ_slc[:, :, q_start:q_end] = torch.matmul(ds, sel_k)

            # dK = dS^T @ Q
            dk_qt = torch.matmul(ds.transpose(-2, -1), q_tile)  # (B, H, kv_len, D)

            # Reduce dK, dV from H heads to H_kv heads before scatter
            if groups > 1:
                dk_qt = dk_qt.reshape(B_size, H_kv, groups, kv_len, D).sum(dim=2)
                dv_qt = dv_qt.reshape(B_size, H_kv, groups, kv_len, D).sum(dim=2)

            # Scatter dK, dV back at H_kv resolution
            dK_slc_full.scatter_add_(2, gather_kv, dk_qt)
            dV_slc_full.scatter_add_(2, gather_kv, dv_qt)

        dQ_slc = dQ_slc.transpose(1, 2).to(Q.dtype)
        dK_slc = dK_slc_full.transpose(1, 2).to(K.dtype)
        dV_slc = dV_slc_full.transpose(1, 2).to(V.dtype)

        # Branch 3: Sliding window attention backward
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
    n_block_size: int = 128,
) -> Tensor:
    """NSA with autograd support (forward + backward).

    Same interface as nsa_forward, but supports training via torch.autograd.
    Uses FA4 for both forward and backward passes, with activation checkpointing
    to reduce memory usage (recomputes branch outputs during backward).

    Args:
        Q: Query tensor, shape (B, N, H, D).
        K: Key tensor, shape (B, N, H_kv, D).
        V: Value tensor, shape (B, N, H_kv, D).
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

    Returns:
        Output tensor, shape (B, N, H, D).
    """
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
