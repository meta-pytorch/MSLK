# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

"""NSA autograd function: wraps FA4-based forward + backward with activation checkpointing.

Provides NSAFunction(torch.autograd.Function) and a user-facing nsa() function
that supports training (backward pass).
"""

from __future__ import annotations

import math
from typing import Callable, Optional, Tuple

import torch
from mslk.attention.sparse_attn.compress import (
    compress_kv,
    fused_compress_kv,
    fused_compress_kv_backward,
)
from mslk.attention.sparse_attn.gating import (
    compute_gates,
    fused_gating_backward,
    gate_and_combine,
)
from mslk.attention.sparse_attn.nsa_forward import (
    _fa4_fwd,
    _make_compressed_causal_mask,
)
from mslk.attention.sparse_attn.select import fused_score_and_select_blocks
from mslk.attention.sparse_attn.sparsity_masks import build_fa4_block_sparse_tensors
from torch import Tensor


def _transpose_block_sparse_for_bwd(
    tensors,
    seqlen_q: int,
    seqlen_k: int,
    n_block_size: int,
    q_tile_size: int,
):
    """Transpose forward block sparse tensors to backward Q-direction format.

    Forward tensors indexed by (m=Q-tiles, n=KV-blocks).
    Backward tensors indexed by (m=KV-blocks, n=Q-tiles).
    """
    from mslk.attention.flash_attn.block_sparsity import BlockSparseTensorsTorch

    n_kv_blocks = (seqlen_k + n_block_size - 1) // n_block_size
    n_q_tiles = seqlen_q // q_tile_size

    B, H = tensors.full_block_cnt.shape[:2]
    device = tensors.full_block_cnt.device

    fwd_full_idx = tensors.full_block_idx  # (B, H, n_q_tiles, n_kv_blocks)
    fwd_full_cnt = tensors.full_block_cnt  # (B, H, n_q_tiles)

    # Build dense boolean attendance: (B, H, n_q_tiles, n_kv_blocks)
    attendance = torch.zeros(
        B, H, n_q_tiles, n_kv_blocks, dtype=torch.bool, device=device
    )
    for qt in range(n_q_tiles):
        cnt = fwd_full_cnt[:, :, qt]  # (B, H)
        for i in range(min(fwd_full_idx.shape[3], n_kv_blocks)):
            if (i < cnt).any():
                idx = fwd_full_idx[:, :, qt, i].long().clamp(0, n_kv_blocks - 1)
                # Set attendance[b, h, qt, idx[b,h]] = True where i < cnt[b,h]
                valid = i < cnt  # (B, H)
                for b in range(B):
                    for h in range(H):
                        if valid[b, h]:
                            attendance[b, h, qt, idx[b, h]] = True

    # Transpose: (B, H, n_q_tiles, n_kv_blocks) -> (B, H, n_kv_blocks, n_q_tiles)
    bwd_attendance = attendance.transpose(2, 3)

    # Convert to cnt/idx: for each KV block, list the Q tiles that attend to it
    bwd_full_cnt = bwd_attendance.sum(dim=-1).to(torch.int32)  # (B, H, n_kv_blocks)

    # Build index tensor: for each KV block, pack the attending Q tile indices
    bwd_full_idx = torch.zeros(
        B, H, n_kv_blocks, n_q_tiles, dtype=torch.int32, device=device
    )
    for b in range(B):
        for h in range(H):
            for kv in range(n_kv_blocks):
                qt_indices = bwd_attendance[b, h, kv].nonzero(as_tuple=True)[0]
                for i, qt in enumerate(qt_indices):
                    bwd_full_idx[b, h, kv, i] = qt.int()

    bwd_mask_cnt = torch.zeros(B, H, n_kv_blocks, dtype=torch.int32, device=device)
    bwd_mask_idx = torch.zeros(
        B, H, n_kv_blocks, n_q_tiles, dtype=torch.int32, device=device
    )

    return BlockSparseTensorsTorch(
        mask_block_cnt=bwd_mask_cnt,
        mask_block_idx=bwd_mask_idx,
        full_block_cnt=bwd_full_cnt,
        full_block_idx=bwd_full_idx,
        block_size=(q_tile_size, n_block_size),
    )


def _fa4_bwd(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    out: Tensor,
    dout: Tensor,
    lse: Tensor,
    softmax_scale: float | None = None,
    causal: bool = False,
    window_size_left: int | None = None,
    window_size_right: int | None = None,
    block_sparse_tensors=None,
    mask_mod: Callable | None = None,
    aux_tensors: list[Tensor] | None = None,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Call FA4's backward pass (low-level, supports block sparsity and mask_mod).

    All inputs are (B, N, H, D) layout.
    Returns (dq, dk, dv).
    """
    from mslk.fb.mslk.attention.flash_attn.interface import _flash_attn_bwd

    # Workaround: FA4 backward doesn't disable 2-CTA for block sparsity,
    # but does for mask_mod. Pass a trivial mask to force 1-CTA mode.
    if block_sparse_tensors is not None and mask_mod is None:
        import cutlass.cute as cute

        @cute.jit
        def _trivial_mask(batch_idx, head_idx, q_idx, kv_idx, seqlen_info, aux_tensors):
            return True

        mask_mod = _trivial_mask

    dq, dk, dv = _flash_attn_bwd(
        q,
        k,
        v,
        out,
        dout,
        lse,
        softmax_scale=softmax_scale,
        causal=causal,
        window_size_left=window_size_left,
        window_size_right=window_size_right,
        block_sparse_tensors=block_sparse_tensors,
        mask_mod=mask_mod,
        aux_tensors=aux_tensors,
    )
    return dq, dk, dv


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

        # --- Stage 2: Three FA4 backward passes ---
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

        # For the selected branch backward, use mask_mod to implement block
        # selection. This avoids FA4's block_sparse_tensors backward path which
        # has 2-CTA compatibility issues on SM100. The mask_mod checks if each
        # (q_idx, kv_idx) pair falls within a selected block, using the block
        # indices passed as aux_tensors.
        block_indices = ctx.block_indices
        q_tile_size = ctx.q_tile_size
        num_sel = ctx.num_selected_blocks

        import cutlass
        import cutlass.cute as cute

        @cute.jit
        def _selected_block_mask(
            batch_idx, head_idx, q_idx, kv_idx, seqlen_info, aux_tensors
        ):
            # aux_tensors[0] = block_indices: (B, H, N_q_tiles, k) int32
            indices = aux_tensors[0]
            q_tile = q_idx[0] // cutlass.const_expr(q_tile_size)
            kv_block = kv_idx[0] // cutlass.const_expr(compress_block_size)

            found = cutlass.Boolean(False)
            for i in cutlass.range_constexpr(num_sel):
                sel_block = indices[batch_idx[0], head_idx[0], q_tile, i]
                found = found | (sel_block == kv_block)
            if cutlass.const_expr(causal):
                found = found & (kv_idx <= q_idx)
            return found

        dQ_slc, dK_slc, dV_slc = _fa4_bwd(
            Q,
            K,
            V,
            O_slc,
            dO_slc,
            lse_slc,
            softmax_scale=softmax_scale,
            causal=False,  # causal handled in mask_mod
            mask_mod=_selected_block_mask,
            aux_tensors=[block_indices],
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
