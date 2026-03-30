# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

"""NSA autograd function: wraps FA4-based forward + backward with activation checkpointing.

Provides NSAFunction(torch.autograd.Function) and a user-facing nsa() function
that supports training (backward pass).

Supports both fixed-length (4D) and variable-length (3D + cu_seqlens) inputs.
For varlen, the selected and sliding window branches use native FA4 varlen
(no padding), while the compressed branch uses padded 4D (mask_mod blocks
varlen in FA4 backward).
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
from mslk.attention.sparse_attn.select import (
    fused_score_and_select_all,
    fused_score_and_select_blocks,
    select_compressed_blocks,
)
from mslk.attention.sparse_attn.sparsity_masks import (
    build_compressed_block_sparse_tensors,
    build_fa4_block_sparse_tensors,
)
from torch import Tensor


def _transpose_block_sparse_for_bwd(
    fwd_tensors,
    seqlen_q: int,
    seqlen_k: int,
    n_block_size: int,
    use_mask_blocks: bool = False,
):
    """Transpose forward block-sparse tensors for the backward pass.

    Forward: (B, H, n_q_tiles, k) — for each Q-tile, which KV-blocks
    Backward: (B, H, n_kv_blocks, max_q_per_kv) — for each KV-block, which Q-tiles

    The backward kernel iterates over KV-blocks and needs to know which Q-tiles
    contribute gradients to each KV-block.

    Uses a sparse inverted-index construction: O(total_entries) time and memory,
    no dense (n_q_tiles x n_kv_blocks) intermediate.

    Args:
        use_mask_blocks: If True, read from mask_block_cnt/idx instead of
            full_block_cnt/idx, and output as mask blocks. Used for the
            compressed branch where mask_mod is needed.
    """
    from mslk.attention.flash_attn.block_sparsity import BlockSparseTensorsTorch

    if use_mask_blocks:
        fwd_cnt = fwd_tensors.mask_block_cnt
        fwd_idx = fwd_tensors.mask_block_idx
    else:
        fwd_cnt = fwd_tensors.full_block_cnt  # (B, H, n_q_tiles)
        fwd_idx = fwd_tensors.full_block_idx  # (B, H, n_q_tiles, k)
    q_block_size, _ = fwd_tensors.block_size

    B, H, n_q_tiles = fwd_cnt.shape
    k = fwd_idx.shape[3]
    n_kv_blocks = (seqlen_k + n_block_size - 1) // n_block_size
    device = fwd_cnt.device

    # --- Sparse transpose: inverted index construction ---
    # Step 1: Count how many Q-tiles reference each KV-block
    # Build validity mask: (B, H, n_q_tiles, k) — which entries are valid
    idx_range = torch.arange(k, device=device)
    valid_mask = idx_range < fwd_cnt.unsqueeze(-1)  # (B, H, n_q_tiles, k)

    # Flatten valid KV-block indices and scatter-count
    kv_indices = fwd_idx.long().clamp(0, n_kv_blocks - 1)  # (B, H, n_q_tiles, k)

    # Count per KV-block: scatter_add ones at each kv_index
    bwd_cnt = torch.zeros(B, H, n_kv_blocks, dtype=torch.int32, device=device)
    ones = valid_mask.int()
    # Reshape for scatter_add: (B, H, n_q_tiles * k)
    flat_kv = kv_indices.reshape(B, H, -1)
    flat_ones = ones.reshape(B, H, -1)
    bwd_cnt.scatter_add_(2, flat_kv, flat_ones)

    max_q_per_kv = bwd_cnt.max().item()
    if max_q_per_kv == 0:
        max_q_per_kv = 1

    # Step 2: Build compact backward index
    # For each valid (q_tile, kv_block) pair, store q_tile at the next
    # available position for that kv_block in bwd_idx.
    bwd_idx = torch.zeros(
        B, H, n_kv_blocks, max_q_per_kv, dtype=torch.int32, device=device
    )

    # Use a simple per-batch-head Python loop. B*H is small (typically 1*32=32)
    # and k is small (typically 16), so this is fast.
    for b in range(B):
        for h in range(H):
            cnt_bh = fwd_cnt[b, h]  # (n_q_tiles,)
            idx_bh = fwd_idx[b, h]  # (n_q_tiles, k)
            write_offsets = torch.zeros(n_kv_blocks, dtype=torch.long, device=device)
            for t in range(n_q_tiles):
                n_valid = cnt_bh[t].item()
                for ki in range(min(n_valid, k)):
                    kv_blk = idx_bh[t, ki].item()
                    if 0 <= kv_blk < n_kv_blocks:
                        off = write_offsets[kv_blk].item()
                        if off < max_q_per_kv:
                            bwd_idx[b, h, kv_blk, off] = t
                            write_offsets[kv_blk] += 1

    # bwd_idx is already compact: (B, H, n_kv_blocks, max_q_per_kv)
    bwd_mask_cnt = torch.zeros(B, H, n_kv_blocks, dtype=torch.int32, device=device)
    bwd_mask_idx = torch.zeros(B, H, n_kv_blocks, 1, dtype=torch.int32, device=device)

    if use_mask_blocks:
        return BlockSparseTensorsTorch(
            mask_block_cnt=bwd_cnt,
            mask_block_idx=bwd_idx,
            full_block_cnt=bwd_mask_cnt,
            full_block_idx=bwd_mask_idx,
            block_size=(q_block_size, n_block_size),
        )

    return BlockSparseTensorsTorch(
        mask_block_cnt=bwd_mask_cnt,
        mask_block_idx=bwd_mask_idx,
        full_block_cnt=bwd_cnt,
        full_block_idx=bwd_idx,
        block_size=(q_block_size, n_block_size),
    )


def _pad_to_4d(
    src: Tensor,
    cu_seqlens: Tensor,
    seqlens: list[int],
    batch_size: int,
    pad_N: int,
) -> Tensor:
    """Pad a 3D varlen tensor (total_tokens, ...) to 4D (B, pad_N, ...)."""
    out = src.new_zeros(batch_size, pad_N, *src.shape[1:])
    for i, slen in enumerate(seqlens):
        s = cu_seqlens[i].item()
        out[i, :slen] = src[s : s + slen]
    return out


def _unpad_to_3d(
    src: Tensor,
    cu_seqlens: Tensor,
    seqlens: list[int],
    total_tokens: int,
) -> Tensor:
    """Unpad a 4D tensor (B, N, ...) to 3D (total_tokens, ...)."""
    out = src.new_zeros(total_tokens, *src.shape[2:])
    for i, slen in enumerate(seqlens):
        s = cu_seqlens[i].item()
        out[s : s + slen] = src[i, :slen]
    return out


class NSAFunction(torch.autograd.Function):
    """Autograd function for NSA with activation checkpointing.

    Forward: runs the FA4-based NSA forward pass.
    Backward: recomputes FA4 branch outputs (activation checkpointing),
    then calls _flash_attn_bwd for each branch.

    Supports varlen via cu_seqlens: selected + sliding window branches use
    native FA4 varlen (no padding), compressed branch uses padded 4D
    (mask_mod blocks varlen in FA4 backward).

    Saved for backward (small tensors only):
        - Q, K, V (inputs, already in autograd graph)
        - K_cmp, V_cmp (compressed KV, padded to max_N_cmp)
        - block_indices, sparse_tensors (discrete, small)
        - gates (..., H, 3)
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
        K_cmp: Tensor,
        V_cmp: Tensor,
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
        cu_seqlens: Tensor | None,
        max_seqlen: int | None,
        sparse_tensors,  # BlockSparseTensorsTorch
        cmp_sparse_tensors,  # BlockSparseTensorsTorch | None
    ) -> Tensor:
        is_varlen = cu_seqlens is not None

        if is_varlen:
            return _nsa_forward_varlen(
                ctx,
                Q,
                K,
                V,
                K_cmp,
                V_cmp,
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
                cu_seqlens,
                max_seqlen,
                sparse_tensors,
                cmp_sparse_tensors,
            )

        # Fixed-length 4D path (unchanged)
        B, N, H, D = Q.shape
        compressed_mask = (
            _make_compressed_causal_mask(compress_block_size) if causal else None
        )
        if cmp_sparse_tensors is not None:
            O_cmp, _ = _fa4_fwd(
                Q,
                K_cmp,
                V_cmp,
                causal=False,
                softmax_scale=softmax_scale,
                mask_mod=compressed_mask,
                block_sparse_tensors=cmp_sparse_tensors,
            )
        else:
            O_cmp, _ = _fa4_fwd(
                Q,
                K_cmp,
                V_cmp,
                causal=False,
                softmax_scale=softmax_scale,
                mask_mod=compressed_mask,
            )
        O_slc, _ = _fa4_fwd(
            Q,
            K,
            V,
            causal=causal,
            softmax_scale=softmax_scale,
            block_sparse_tensors=sparse_tensors,
        )
        O_sld, _ = _fa4_fwd(
            Q,
            K,
            V,
            causal=causal,
            softmax_scale=softmax_scale,
            window_size_left=window_size,
            window_size_right=0,
        )

        gates = compute_gates(Q, gate_proj_weight)
        O = gate_and_combine(O_cmp, O_slc, O_sld, gates)

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
        ctx.sparse_tensors = sparse_tensors
        ctx.cmp_sparse_tensors = cmp_sparse_tensors
        ctx.compress_block_size = compress_block_size
        ctx.window_size = window_size
        ctx.causal = causal
        ctx.softmax_scale = softmax_scale
        ctx.q_tile_size = q_tile_size
        ctx.n_block_size = n_block_size
        ctx.cu_seqlens = None
        ctx.max_seqlen = None
        ctx.varlen_meta = None

        return O

    @staticmethod
    def backward(ctx, dO: Tensor):
        is_varlen = ctx.cu_seqlens is not None
        if is_varlen:
            return _nsa_backward_varlen(ctx, dO)

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

        # Fixed-length 4D backward (unchanged from original)
        causal = ctx.causal
        softmax_scale = ctx.softmax_scale
        compress_block_size = ctx.compress_block_size
        window_size = ctx.window_size
        sparse_tensors = ctx.sparse_tensors
        cmp_sparse_tensors = ctx.cmp_sparse_tensors

        compressed_mask = (
            _make_compressed_causal_mask(compress_block_size) if causal else None
        )

        dQ_gate = None
        dgate_proj_weight = None

        if gate_proj_weight is not None:
            O_cmp, lse_cmp = _fa4_fwd(
                Q,
                K_cmp,
                V_cmp,
                causal=False,
                softmax_scale=softmax_scale,
                mask_mod=compressed_mask,
                block_sparse_tensors=cmp_sparse_tensors,
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
            g = gates.float()
            dO_f = dO.float()
            dO_cmp = (g[..., 0:1] * dO_f).to(dO.dtype)
            dO_slc = (g[..., 1:2] * dO_f).to(dO.dtype)
            dO_sld = (g[..., 2:3] * dO_f).to(dO.dtype)
            del g, dO_f
            lse_cmp = lse_slc = lse_sld = None
            O_cmp = O_slc = O_sld = None

        # Branch 1: Compressed
        if O_cmp is None:
            O_cmp, lse_cmp = _fa4_fwd(
                Q,
                K_cmp,
                V_cmp,
                causal=False,
                softmax_scale=softmax_scale,
                mask_mod=compressed_mask,
                block_sparse_tensors=cmp_sparse_tensors,
            )
        if cmp_sparse_tensors is not None:
            cmp_bwd_sparse = _transpose_block_sparse_for_bwd(
                cmp_sparse_tensors,
                seqlen_q=Q.shape[1],
                seqlen_k=K_cmp.shape[1],
                n_block_size=cmp_sparse_tensors.block_size[1],
                use_mask_blocks=True,
            )
        else:
            cmp_bwd_sparse = None
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
            block_sparse_tensors=cmp_bwd_sparse,
        )
        del O_cmp, lse_cmp, dO_cmp

        # Branch 2: Selected (block-sparse)
        if O_slc is None:
            O_slc, lse_slc = _fa4_fwd(
                Q,
                K,
                V,
                causal=causal,
                softmax_scale=softmax_scale,
                block_sparse_tensors=sparse_tensors,
            )
        bwd_sparse = _transpose_block_sparse_for_bwd(
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
            block_sparse_tensors=bwd_sparse,
        )
        del O_slc, lse_slc, dO_slc, bwd_sparse

        # Branch 3: Sliding window
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

        # Compression backward
        dK_compress, dV_compress, dW_k, dW_v = fused_compress_kv_backward(
            dK_cmp,
            dV_cmp,
            K,
            V,
            compress_block_size,
            W_k_compress,
            W_v_compress,
        )
        del dK_cmp, dV_cmp

        # Gradient accumulation — in-place to minimize peak memory
        dQ_cmp += dQ_slc
        del dQ_slc
        dQ_cmp += dQ_sld
        del dQ_sld
        if dQ_gate is not None:
            dQ_cmp += dQ_gate
            del dQ_gate
        dQ = dQ_cmp

        dK_slc += dK_sld
        del dK_sld
        dK_slc += dK_compress
        del dK_compress
        dK = dK_slc

        dV_slc += dV_sld
        del dV_sld
        dV_slc += dV_compress
        del dV_compress
        dV = dV_slc

        # Return gradients matching forward() args order:
        # Q, K, V, K_cmp, V_cmp, W_k, W_v, gate_proj_weight, + 11 non-tensor args
        return (
            dQ,
            dK,
            dV,
            None,
            None,  # K_cmp, V_cmp (not differentiable inputs)
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
            None,
            None,
            None,
            None,  # cmp_sparse_tensors
        )


def _nsa_forward_varlen(
    ctx,
    Q: Tensor,  # (total_tokens, H, D)
    K: Tensor,  # (total_tokens, H_kv, D)
    V: Tensor,  # (total_tokens, H_kv, D)
    K_cmp: Tensor,  # (B, max_N_cmp, H_kv, D)
    V_cmp: Tensor,  # (B, max_N_cmp, H_kv, D)
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
    cu_seqlens: Tensor,
    max_seqlen: int | None,
    sparse_tensors,
    cmp_sparse_tensors,
) -> Tensor:
    """Varlen forward: native varlen for selected + sliding, padded for compressed."""
    H = Q.shape[1]
    D = Q.shape[2]
    batch_size = cu_seqlens.shape[0] - 1
    seqlens = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
    if max_seqlen is None:
        max_seqlen = max(seqlens)
    pad_N = ((max_seqlen + q_tile_size - 1) // q_tile_size) * q_tile_size

    compressed_mask = (
        _make_compressed_causal_mask(compress_block_size) if causal else None
    )

    # Branch 1: Compressed attention (padded 4D — mask_mod blocks varlen)
    Q_pad = _pad_to_4d(Q, cu_seqlens, seqlens, batch_size, pad_N)
    if cmp_sparse_tensors is not None:
        O_cmp_pad, _ = _fa4_fwd(
            Q_pad,
            K_cmp,
            V_cmp,
            causal=False,
            softmax_scale=softmax_scale,
            mask_mod=compressed_mask,
            block_sparse_tensors=cmp_sparse_tensors,
        )
    else:
        O_cmp_pad, _ = _fa4_fwd(
            Q_pad,
            K_cmp,
            V_cmp,
            causal=False,
            softmax_scale=softmax_scale,
            mask_mod=compressed_mask,
        )
    O_cmp = _unpad_to_3d(O_cmp_pad, cu_seqlens, seqlens, Q.shape[0])
    del Q_pad, O_cmp_pad

    # Branch 2: Selected attention (native varlen + block sparse)
    O_slc, _ = _fa4_fwd(
        Q,
        K,
        V,
        causal=causal,
        softmax_scale=softmax_scale,
        block_sparse_tensors=sparse_tensors,
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_k=cu_seqlens,
        max_seqlen_q=pad_N,
        max_seqlen_k=pad_N,
    )

    # Branch 3: Sliding window (native varlen)
    O_sld, _ = _fa4_fwd(
        Q,
        K,
        V,
        causal=causal,
        softmax_scale=softmax_scale,
        window_size_left=window_size,
        window_size_right=0,
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_k=cu_seqlens,
        max_seqlen_q=max_seqlen,
        max_seqlen_k=max_seqlen,
    )

    # Gating (3D — Commit 1)
    gates = compute_gates(Q, gate_proj_weight)
    O = gate_and_combine(O_cmp, O_slc, O_sld, gates)

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
        cu_seqlens,
    )
    ctx.sparse_tensors = sparse_tensors
    ctx.cmp_sparse_tensors = cmp_sparse_tensors
    ctx.compress_block_size = compress_block_size
    ctx.window_size = window_size
    ctx.causal = causal
    ctx.softmax_scale = softmax_scale
    ctx.q_tile_size = q_tile_size
    ctx.n_block_size = n_block_size
    ctx.cu_seqlens = cu_seqlens
    ctx.max_seqlen = max_seqlen
    ctx.varlen_meta = (seqlens, batch_size, pad_N)

    return O


def _nsa_backward_varlen(ctx, dO: Tensor) -> tuple:
    """Varlen backward: native varlen for selected + sliding, padded for compressed."""
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
        cu_seqlens,
    ) = ctx.saved_tensors

    causal = ctx.causal
    softmax_scale = ctx.softmax_scale
    compress_block_size = ctx.compress_block_size
    window_size = ctx.window_size
    sparse_tensors = ctx.sparse_tensors
    cmp_sparse_tensors = ctx.cmp_sparse_tensors
    max_seqlen = ctx.max_seqlen
    seqlens, batch_size, pad_N = ctx.varlen_meta

    compressed_mask = (
        _make_compressed_causal_mask(compress_block_size) if causal else None
    )

    dQ_gate = None
    dgate_proj_weight = None

    if gate_proj_weight is not None:
        # Need all O_i for gating backward — recompute all three forwards
        Q_pad = _pad_to_4d(Q, cu_seqlens, seqlens, batch_size, pad_N)
        O_cmp_pad, _ = _fa4_fwd(
            Q_pad,
            K_cmp,
            V_cmp,
            causal=False,
            softmax_scale=softmax_scale,
            mask_mod=compressed_mask,
            block_sparse_tensors=cmp_sparse_tensors,
        )
        O_cmp = _unpad_to_3d(O_cmp_pad, cu_seqlens, seqlens, Q.shape[0])
        del O_cmp_pad

        O_slc, lse_slc = _fa4_fwd(
            Q,
            K,
            V,
            causal=causal,
            softmax_scale=softmax_scale,
            block_sparse_tensors=sparse_tensors,
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=cu_seqlens,
            max_seqlen_q=pad_N,
            max_seqlen_k=pad_N,
        )
        O_sld, lse_sld = _fa4_fwd(
            Q,
            K,
            V,
            causal=causal,
            softmax_scale=softmax_scale,
            window_size_left=window_size,
            window_size_right=0,
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=cu_seqlens,
            max_seqlen_q=max_seqlen,
            max_seqlen_k=max_seqlen,
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
        del O_cmp
    else:
        g = gates.float()
        dO_f = dO.float()
        dO_cmp = (g[..., 0:1] * dO_f).to(dO.dtype)
        dO_slc = (g[..., 1:2] * dO_f).to(dO.dtype)
        dO_sld = (g[..., 2:3] * dO_f).to(dO.dtype)
        del g, dO_f
        Q_pad = None
        lse_slc = lse_sld = None
        O_slc = O_sld = None

    # --- Branch 1: Compressed attention (padded 4D) ---
    if Q_pad is None:
        Q_pad = _pad_to_4d(Q, cu_seqlens, seqlens, batch_size, pad_N)
    O_cmp_pad, lse_cmp_pad = _fa4_fwd(
        Q_pad,
        K_cmp,
        V_cmp,
        causal=False,
        softmax_scale=softmax_scale,
        mask_mod=compressed_mask,
        block_sparse_tensors=cmp_sparse_tensors,
    )
    dO_cmp_pad = _pad_to_4d(dO_cmp, cu_seqlens, seqlens, batch_size, pad_N)
    if cmp_sparse_tensors is not None:
        cmp_bwd_sparse = _transpose_block_sparse_for_bwd(
            cmp_sparse_tensors,
            seqlen_q=pad_N,
            seqlen_k=K_cmp.shape[1],
            n_block_size=cmp_sparse_tensors.block_size[1],
            use_mask_blocks=True,
        )
    else:
        cmp_bwd_sparse = None
    dQ_cmp_pad, dK_cmp, dV_cmp = _fa4_bwd(
        Q_pad,
        K_cmp,
        V_cmp,
        O_cmp_pad,
        dO_cmp_pad,
        lse_cmp_pad,
        softmax_scale=softmax_scale,
        causal=False,
        mask_mod=compressed_mask,
        block_sparse_tensors=cmp_bwd_sparse,
    )
    dQ_cmp = _unpad_to_3d(dQ_cmp_pad, cu_seqlens, seqlens, Q.shape[0])
    del Q_pad, O_cmp_pad, lse_cmp_pad, dO_cmp_pad, dQ_cmp_pad, dO_cmp

    # --- Branch 2: Selected attention (native varlen + block sparse) ---
    if O_slc is None:
        O_slc, lse_slc = _fa4_fwd(
            Q,
            K,
            V,
            causal=causal,
            softmax_scale=softmax_scale,
            block_sparse_tensors=sparse_tensors,
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=cu_seqlens,
            max_seqlen_q=pad_N,
            max_seqlen_k=pad_N,
        )
    bwd_sparse = _transpose_block_sparse_for_bwd(
        sparse_tensors,
        seqlen_q=pad_N,
        seqlen_k=pad_N,
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
        block_sparse_tensors=bwd_sparse,
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_k=cu_seqlens,
        max_seqlen_q=pad_N,
        max_seqlen_k=pad_N,
    )
    del O_slc, lse_slc, dO_slc, bwd_sparse

    # --- Branch 3: Sliding window (native varlen) ---
    if O_sld is None:
        O_sld, lse_sld = _fa4_fwd(
            Q,
            K,
            V,
            causal=causal,
            softmax_scale=softmax_scale,
            window_size_left=window_size,
            window_size_right=0,
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=cu_seqlens,
            max_seqlen_q=max_seqlen,
            max_seqlen_k=max_seqlen,
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
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_k=cu_seqlens,
        max_seqlen_q=max_seqlen,
        max_seqlen_k=max_seqlen,
    )
    del O_sld, lse_sld, dO_sld

    # --- Compression backward (varlen: scatter to 3D directly) ---
    dK_compress, dV_compress, dW_k, dW_v = fused_compress_kv_backward(
        dK_cmp,
        dV_cmp,
        K,
        V,
        compress_block_size,
        W_k_compress,
        W_v_compress,
        cu_seqlens=cu_seqlens,
    )

    # --- Gradient accumulation (all 3D) — in-place to minimize peak memory ---
    dQ_cmp += dQ_slc
    del dQ_slc
    dQ_cmp += dQ_sld
    del dQ_sld
    if dQ_gate is not None:
        dQ_cmp += dQ_gate
        del dQ_gate
    dQ = dQ_cmp

    dK_slc += dK_sld
    del dK_sld
    dK_slc += dK_compress
    del dK_compress
    dK = dK_slc

    dV_slc += dV_sld
    del dV_sld
    dV_slc += dV_compress
    del dV_compress
    dV = dV_slc

    # Return gradients matching forward() args order
    return (
        dQ,
        dK,
        dV,
        None,
        None,  # K_cmp, V_cmp
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
        None,
        None,
        None,
        None,  # cmp_sparse_tensors
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
    num_cmp_selected_blocks: int | None = None,
) -> Tensor:
    """NSA with autograd support (forward + backward).

    Same interface as nsa_forward, but supports training via torch.autograd.
    Uses FA4 for both forward and backward passes, with activation checkpointing.
    Supports both fixed-length (4D) and variable-length (3D + cu_seqlens) inputs.

    For varlen, the selected and sliding window branches use native FA4 varlen
    (no padding on Q, K, V). Only the compressed branch uses padded Q for
    mask_mod compatibility. Compress/select operate on 3D input directly.

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
        max_seqlen: Maximum sequence length for varlen.
        num_cmp_selected_blocks: Number of FA4 KV blocks to select for the
            compressed branch. When set, the compressed branch uses block-sparse
            attention (sub-quadratic). When None (default), full compressed
            attention is used.

    Returns:
        Output tensor, same shape as Q.
    """
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(Q.shape[-1])

    is_varlen = cu_seqlens is not None

    if is_varlen:
        assert Q.dim() == 3, f"Varlen requires 3D input, got {Q.dim()}D"
        seqlens = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
        if max_seqlen is None:
            max_seqlen = max(seqlens)
        pad_N = ((max_seqlen + q_tile_size - 1) // q_tile_size) * q_tile_size

    # Step 1: Compress KV (varlen-aware — reads 3D directly, no Q/K/V padding)
    K_cmp, V_cmp = fused_compress_kv(
        K,
        V,
        compress_block_size,
        W_k_compress,
        W_v_compress,
        cu_seqlens=cu_seqlens,
        max_seqlen=max_seqlen if is_varlen else None,
    )

    # Step 2: Score blocks and select top-k (varlen-aware Q_mean)
    block_indices, cmp_block_indices = fused_score_and_select_all(
        Q,
        K_cmp,
        num_selected_blocks,
        compress_block_size,
        num_cmp_selected_blocks=num_cmp_selected_blocks,
        cmp_n_block_size=n_block_size,
        causal=causal,
        q_tile_size=q_tile_size,
        softmax_scale=softmax_scale,
        cu_seqlens=cu_seqlens,
        max_seqlen=max_seqlen if is_varlen else None,
    )

    # Step 3: Build FA4 sparsity masks
    seqlen_k = pad_N if is_varlen else Q.shape[1]
    sparse_tensors = build_fa4_block_sparse_tensors(
        block_indices,
        compress_block_size,
        n_block_size=n_block_size,
        seqlen_k=seqlen_k,
    )

    # Step 3b: Block-sparse compressed attention (optional)
    cmp_sparse_tensors = None
    if num_cmp_selected_blocks is not None:
        N_cmp = K_cmp.shape[1]
        cmp_n_block_size = n_block_size
        n_cmp_kv_blocks = (N_cmp + cmp_n_block_size - 1) // cmp_n_block_size
        if cmp_block_indices is not None:
            cmp_sparse_tensors = build_compressed_block_sparse_tensors(
                cmp_block_indices,
                n_kv_blocks=n_cmp_kv_blocks,
                q_tile_size=q_tile_size,
                cmp_n_block_size=cmp_n_block_size,
            )

    # Step 4: NSAFunction handles the three FA4 branches + gating
    return NSAFunction.apply(
        Q,
        K,
        V,
        K_cmp,
        V_cmp,
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
        cu_seqlens,
        max_seqlen,
        sparse_tensors,
        cmp_sparse_tensors,
    )
