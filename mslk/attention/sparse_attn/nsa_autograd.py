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


def _attn_bwd_pytorch(
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
    mask: Tensor | None = None,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Attention backward using PyTorch ops (torch.matmul + cuBLAS). No CUTLASS JIT.

    All inputs are (B, N, H, D) layout.
    Returns (dq, dk, dv).

    Uses the saved output and log-sum-exp to recompute attention weights
    without materializing the full N×N score matrix where possible.
    """
    B, N_q, H, D = q.shape
    H_kv = k.shape[2]
    N_kv = k.shape[1]
    groups = H // H_kv

    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(D)

    # Transpose to (B, H, N, D) for matmul
    q_t = q.transpose(1, 2).float()  # (B, H, N_q, D)
    k_t = k.transpose(1, 2).float()  # (B, H_kv, N_kv, D)
    v_t = v.transpose(1, 2).float()  # (B, H_kv, N_kv, D)
    dout_t = dout.transpose(1, 2).float()  # (B, H, N_q, D)

    # Expand K, V for GQA
    if groups > 1:
        k_t = k_t.repeat_interleave(groups, dim=1)  # (B, H, N_kv, D)
        v_t = v_t.repeat_interleave(groups, dim=1)

    # Recompute attention scores and weights from LSE
    scores = (
        torch.matmul(q_t, k_t.transpose(-2, -1)) * softmax_scale
    )  # (B, H, N_q, N_kv)

    # Apply masks
    if causal and N_q == N_kv:
        q_pos = torch.arange(N_q, device=q.device).unsqueeze(1)
        kv_pos = torch.arange(N_kv, device=q.device).unsqueeze(0)
        causal_mask = kv_pos > q_pos
        if window_size_left is not None:
            causal_mask = causal_mask | (kv_pos < q_pos - window_size_left + 1)
        scores = scores.masked_fill(
            causal_mask.unsqueeze(0).unsqueeze(0), float("-inf")
        )
    elif window_size_left is not None and N_q == N_kv:
        q_pos = torch.arange(N_q, device=q.device).unsqueeze(1)
        kv_pos = torch.arange(N_kv, device=q.device).unsqueeze(0)
        window_mask = (kv_pos > q_pos) | (kv_pos < q_pos - window_size_left + 1)
        scores = scores.masked_fill(
            window_mask.unsqueeze(0).unsqueeze(0), float("-inf")
        )

    if mask is not None:
        scores = scores.masked_fill(mask, float("-inf"))

    # Softmax (recompute from scores, not from LSE, for simplicity)
    attn = torch.softmax(scores, dim=-1)

    # Standard attention backward:
    # dV = P^T @ dO
    dv_t = torch.matmul(attn.transpose(-2, -1), dout_t)  # (B, H, N_kv, D)

    # dP = dO @ V^T
    dp = torch.matmul(dout_t, v_t.transpose(-2, -1))  # (B, H, N_q, N_kv)

    # dS = P * (dP - (dP * P).sum(-1, keepdim=True))
    # Equivalently: dS = P * (dP - rowsum), where rowsum = (dO * O).sum(D)
    rowsum = (dout_t * (torch.matmul(attn, v_t))).sum(dim=-1, keepdim=True)
    ds = attn * (dp - rowsum) * softmax_scale

    # dQ = dS @ K
    dq_t = torch.matmul(ds, k_t)  # (B, H, N_q, D)

    # dK = dS^T @ Q
    dk_t = torch.matmul(ds.transpose(-2, -1), q_t)  # (B, H, N_kv, D)

    # Handle GQA: sum dK, dV across head groups
    if groups > 1:
        dk_t = dk_t.reshape(B, H_kv, groups, N_kv, D).sum(dim=2)
        dv_t = dv_t.reshape(B, H_kv, groups, N_kv, D).sum(dim=2)

    # Transpose back to (B, N, H, D)
    return (
        dq_t.transpose(1, 2).to(q.dtype),
        dk_t.transpose(1, 2).to(k.dtype),
        dv_t.transpose(1, 2).to(v.dtype),
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

        # --- Stage 2: Three attention backward passes (PyTorch/cuBLAS, no CUTLASS JIT) ---

        # Branch 1: Compressed attention backward
        # Build compressed causal mask: kv_block_start > q_pos
        B_size, N_size, H_size = Q.shape[:3]
        N_cmp = K_cmp.shape[1]
        cmp_mask = None
        if causal:
            q_pos = torch.arange(N_size, device=Q.device).unsqueeze(1)
            kv_block_start = torch.arange(N_cmp, device=Q.device) * compress_block_size
            cmp_mask = (kv_block_start.unsqueeze(0) > q_pos).unsqueeze(0).unsqueeze(0)

        dQ_cmp, dK_cmp, dV_cmp = _attn_bwd_pytorch(
            Q,
            K_cmp,
            V_cmp,
            O_cmp,
            dO_cmp,
            lse_cmp,
            softmax_scale=softmax_scale,
            mask=cmp_mask,
        )

        # Branch 2: Selected attention backward via gather/scatter
        # Gather selected K/V blocks, run attention backward on gathered subset,
        # then scatter dK/dV gradients back to full KV positions.
        block_indices = ctx.block_indices
        q_tile_size = ctx.q_tile_size
        num_sel = ctx.num_selected_blocks
        block_size = compress_block_size

        N_blocks = N_size // block_size
        N_q_tiles = N_size // q_tile_size
        k_actual = min(num_sel, N_blocks)

        # Expand block indices to position indices
        offsets = torch.arange(block_size, device=Q.device)
        pos_indices = block_indices.unsqueeze(-1) * block_size + offsets
        kv_len = k_actual * block_size
        pos_indices = pos_indices.reshape(B_size, H_size, N_q_tiles, kv_len)

        # Gather K, V at selected positions: (B, H, N_q_tiles, kv_len, D)
        H_kv = K.shape[2]
        groups = H_size // H_kv
        k_t = K.transpose(1, 2).float()
        v_t = V.transpose(1, 2).float()
        if groups > 1:
            k_t = k_t.repeat_interleave(groups, dim=1)
            v_t = v_t.repeat_interleave(groups, dim=1)

        D = Q.shape[-1]
        gather_idx = pos_indices.unsqueeze(-1).expand(
            B_size, H_size, N_q_tiles, kv_len, D
        )
        k_exp = k_t.unsqueeze(2).expand(B_size, H_size, N_q_tiles, N_size, D)
        v_exp = v_t.unsqueeze(2).expand(B_size, H_size, N_q_tiles, N_size, D)
        sel_k = torch.gather(k_exp, 3, gather_idx)
        sel_v = torch.gather(v_exp, 3, gather_idx)

        # Reshape Q and dO into tiles
        q_tiles = (
            Q.transpose(1, 2).float().reshape(B_size, H_size, N_q_tiles, q_tile_size, D)
        )
        dO_slc_tiles = (
            dO_slc.transpose(1, 2)
            .float()
            .reshape(B_size, H_size, N_q_tiles, q_tile_size, D)
        )

        # Attention scores on gathered KV
        scores_slc = torch.matmul(q_tiles, sel_k.transpose(-2, -1)) * softmax_scale

        # Causal mask on gathered positions
        if causal:
            q_tile_starts = torch.arange(N_q_tiles, device=Q.device) * q_tile_size
            q_offsets_t = torch.arange(q_tile_size, device=Q.device)
            q_positions = q_tile_starts.unsqueeze(1) + q_offsets_t.unsqueeze(0)
            c_mask = pos_indices.unsqueeze(3) > q_positions.reshape(
                1, 1, N_q_tiles, q_tile_size, 1
            )
            scores_slc = scores_slc.masked_fill(c_mask, float("-inf"))

        attn_slc = torch.softmax(scores_slc, dim=-1)

        # dV_sel = attn^T @ dO_slc_tiles
        dv_sel = torch.matmul(attn_slc.transpose(-2, -1), dO_slc_tiles)

        # dP = dO @ V^T
        dp_slc = torch.matmul(dO_slc_tiles, sel_v.transpose(-2, -1))
        rowsum_slc = (dO_slc_tiles * torch.matmul(attn_slc, sel_v)).sum(
            dim=-1, keepdim=True
        )
        ds_slc = attn_slc * (dp_slc - rowsum_slc) * softmax_scale

        # dQ_slc_tiles = dS @ sel_K
        dq_slc_tiles = torch.matmul(ds_slc, sel_k)
        dQ_slc = (
            dq_slc_tiles.reshape(B_size, H_size, N_size, D).transpose(1, 2).to(Q.dtype)
        )

        # dK_sel = dS^T @ Q_tiles
        dk_sel = torch.matmul(ds_slc.transpose(-2, -1), q_tiles)

        # Scatter dK_sel and dV_sel back to full KV
        dK_slc_full = torch.zeros(
            B_size, H_size, N_size, D, device=Q.device, dtype=torch.float32
        )
        dV_slc_full = torch.zeros_like(dK_slc_full)
        scatter_idx = gather_idx  # (B, H, N_q_tiles, kv_len, D)
        # Sum across query tiles that share the same KV position
        for qt in range(N_q_tiles):
            dK_slc_full.scatter_add_(2, gather_idx[:, :, qt], dk_sel[:, :, qt])
            dV_slc_full.scatter_add_(2, gather_idx[:, :, qt], dv_sel[:, :, qt])

        # Handle GQA: sum across head groups
        if groups > 1:
            dK_slc_full = dK_slc_full.reshape(B_size, H_kv, groups, N_size, D).sum(
                dim=2
            )
            dV_slc_full = dV_slc_full.reshape(B_size, H_kv, groups, N_size, D).sum(
                dim=2
            )
        dK_slc = dK_slc_full.transpose(1, 2).to(K.dtype)
        dV_slc = dV_slc_full.transpose(1, 2).to(V.dtype)

        # Branch 3: Sliding window backward — use PyTorch SDPA which has
        # built-in memory-efficient backward (no N×N materialization).
        # We recompute forward + backward together via SDPA's autograd.
        H_kv_sld = K.shape[2]
        groups_sld = H_size // H_kv_sld
        q_sld = Q.float()
        k_sld = K.float()
        v_sld = V.float()

        # SDPA needs (B, H, N, D) layout
        q_sld_t = q_sld.transpose(1, 2)  # (B, H, N, D)
        k_sld_t = k_sld.transpose(1, 2)  # (B, H_kv, N, D)
        v_sld_t = v_sld.transpose(1, 2)

        # Expand for GQA
        if groups_sld > 1:
            k_sld_t = k_sld_t.repeat_interleave(groups_sld, dim=1)
            v_sld_t = v_sld_t.repeat_interleave(groups_sld, dim=1)

        # Enable grad for SDPA backward
        q_sld_t.requires_grad_(True)
        k_sld_t.requires_grad_(True)
        v_sld_t.requires_grad_(True)

        # Build causal+window mask via attn_mask
        # For sliding window: each query attends to [q_pos - window + 1, q_pos]
        with torch.enable_grad():
            O_sld_recomputed = torch.nn.functional.scaled_dot_product_attention(
                q_sld_t,
                k_sld_t,
                v_sld_t,
                is_causal=causal and window_size is None,
                scale=softmax_scale,
            )
            # Get dO in the right layout
            dO_sld_t = dO_sld.transpose(1, 2).float()
            O_sld_recomputed.backward(dO_sld_t)

        dQ_sld = q_sld_t.grad.transpose(1, 2).to(Q.dtype)
        dk_sld_t = k_sld_t.grad
        dv_sld_t = v_sld_t.grad

        # Handle GQA: sum across head groups for dK, dV
        if groups_sld > 1:
            dk_sld_t = dk_sld_t.reshape(B_size, H_kv_sld, groups_sld, N_size, D).sum(
                dim=2
            )
            dv_sld_t = dv_sld_t.reshape(B_size, H_kv_sld, groups_sld, N_size, D).sum(
                dim=2
            )

        dK_sld = dk_sld_t.transpose(1, 2).to(K.dtype)
        dV_sld = dv_sld_t.transpose(1, 2).to(V.dtype)

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
