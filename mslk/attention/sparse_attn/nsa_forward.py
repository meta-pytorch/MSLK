# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

"""End-to-end NSA (Native Sparse Attention) forward pass using FA4.

Orchestrates the three attention branches:
1. Compressed attention (FA4 on short KV sequence)
2. Selected attention (FA4 with block sparsity)
3. Sliding window attention (FA4 with window_size)

Then gates and combines the outputs.
"""

from __future__ import annotations

import math
from typing import Callable, Optional, Tuple

import torch
from mslk.attention.sparse_attn.compress import fused_compress_kv
from mslk.attention.sparse_attn.gating import fused_gate_and_combine
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


def _make_compressed_causal_mask(compress_block_size: int) -> Callable:
    """Create a CuteDSL mask_mod for compressed attention causal masking.

    Block j (at kv_idx=j) represents positions [j*block_size, (j+1)*block_size).
    Query at position q_idx can attend to block j iff j * block_size <= q_idx.
    """
    import cutlass.cute as cute

    @cute.jit
    def compressed_causal_mask(
        batch_idx, head_idx, q_idx, kv_idx, seqlen_info, aux_tensors
    ):
        return kv_idx * compress_block_size <= q_idx

    return compressed_causal_mask


def _fa4_fwd(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    causal: bool = False,
    softmax_scale: float | None = None,
    window_size_left: int | None = None,
    window_size_right: int | None = None,
    block_sparse_tensors=None,
    mask_mod: Callable | None = None,
    cu_seqlens_q: Tensor | None = None,
    cu_seqlens_k: Tensor | None = None,
    max_seqlen_q: int | None = None,
    max_seqlen_k: int | None = None,
) -> Tuple[Tensor, Tensor]:
    """Call FA4's forward pass.

    Supports block sparsity, mask_mod, and varlen (cu_seqlens).
    Inputs are (B, N, H, D) for fixed-length or (total_tokens, H, D) for varlen.
    Returns (output, lse).
    """
    from mslk.fb.mslk.attention.flash_attn.interface import _flash_attn_fwd

    # mask_mod is not compatible with pack_gqa=True in FA4, so disable it
    pack_gqa = False if mask_mod is not None else None

    out, lse = _flash_attn_fwd(
        q,
        k,
        v,
        softmax_scale=softmax_scale,
        causal=causal,
        window_size_left=window_size_left,
        window_size_right=window_size_right,
        block_sparse_tensors=block_sparse_tensors,
        mask_mod=mask_mod,
        pack_gqa=pack_gqa,
        return_lse=True,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
    )
    return out, lse


def _fa4_bwd(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    out: Tensor,
    dout: Tensor,
    lse: Tensor,
    causal: bool = False,
    softmax_scale: float | None = None,
    window_size_left: int | None = None,
    window_size_right: int | None = None,
    block_sparse_tensors=None,
    mask_mod: Callable | None = None,
    cu_seqlens_q: Tensor | None = None,
    cu_seqlens_k: Tensor | None = None,
    max_seqlen_q: int | None = None,
    max_seqlen_k: int | None = None,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Call FA4's backward pass.

    Supports block sparsity, mask_mod, and varlen (cu_seqlens).
    Returns (dq, dk, dv).
    """
    from mslk.fb.mslk.attention.flash_attn.interface import _flash_attn_bwd

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
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
    )
    return dq, dk, dv


def nsa_forward(
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
    """NSA forward pass using FA4 for all attention branches.

    Supports both fixed-length (4D) and variable-length (3D + cu_seqlens) inputs.
    For varlen, sequences are padded to max_seqlen internally (FA4 block_sparsity
    does not yet support varlen natively).

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
        cu_seqlens: Cumulative sequence lengths, shape (B+1,) int32 for varlen.
        max_seqlen: Maximum sequence length for varlen padding.
        num_cmp_selected_blocks: Number of FA4 KV blocks to select for the
            compressed branch. When set, the compressed branch uses block-sparse
            attention (only attending to the top-k most relevant blocks of
            compressed KV), making it O(N) instead of O(N^2/compress_block_size).
            When None (default), full compressed attention is used.

    Returns:
        Output tensor, same shape as Q.
    """
    is_varlen = cu_seqlens is not None

    if is_varlen:
        # Varlen: pad only for compress+select (need uniform blocks),
        # use native FA4 varlen (cu_seqlens) for the attention branches.
        assert Q.dim() == 3, f"Varlen requires 3D input, got {Q.dim()}D"
        H = Q.shape[1]
        H_kv = K.shape[1]
        D_dim = Q.shape[2]
        batch_size = cu_seqlens.shape[0] - 1
        seqlens = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()

        if max_seqlen is None:
            max_seqlen = max(seqlens)
        if softmax_scale is None:
            softmax_scale = 1.0 / math.sqrt(D_dim)
        pad_N = ((max_seqlen + q_tile_size - 1) // q_tile_size) * q_tile_size

        # Pad Q, K, V to 4D for compress + select (CuteDSL kernels need uniform N)
        Q_pad = Q.new_zeros(batch_size, pad_N, H, D_dim)
        K_pad = K.new_zeros(batch_size, pad_N, H_kv, D_dim)
        V_pad = V.new_zeros(batch_size, pad_N, H_kv, D_dim)
        for i, slen in enumerate(seqlens):
            s = cu_seqlens[i].item()
            Q_pad[i, :slen] = Q[s : s + slen]
            K_pad[i, :slen] = K[s : s + slen]
            V_pad[i, :slen] = V[s : s + slen]

        # Step 1: Compress KV (on padded 4D)
        K_cmp_pad, V_cmp_pad = fused_compress_kv(
            K_pad, V_pad, compress_block_size, W_k_compress, W_v_compress
        )
        N_cmp = pad_N // compress_block_size

        # Step 2: Score blocks and select top-k (on padded 4D)
        block_indices, cmp_block_indices = fused_score_and_select_all(
            Q_pad,
            K_cmp_pad,
            num_selected_blocks,
            compress_block_size,
            num_cmp_selected_blocks=num_cmp_selected_blocks,
            cmp_n_block_size=n_block_size,
            causal=causal,
            q_tile_size=q_tile_size,
            softmax_scale=softmax_scale,
        )

        # Step 3: Build FA4 sparsity masks (on padded 4D)
        sparse_tensors = build_fa4_block_sparse_tensors(
            block_indices,
            compress_block_size,
            n_block_size=n_block_size,
            seqlen_k=pad_N,
        )

        # Step 4: Three FA4 branches
        # Compressed branch uses padded 4D (mask_mod not supported with varlen in FA4).
        # Selected and sliding window use native varlen.

        # Branch 1: Compressed attention (padded 4D — mask_mod not supported with varlen)
        compressed_mask = (
            _make_compressed_causal_mask(compress_block_size) if causal else None
        )

        if num_cmp_selected_blocks is not None:
            N_cmp_pad = K_cmp_pad.shape[1]
            cmp_n_block_size = n_block_size
            n_cmp_kv_blocks = (N_cmp_pad + cmp_n_block_size - 1) // cmp_n_block_size
            if cmp_block_indices is not None:
                cmp_sparse = build_compressed_block_sparse_tensors(
                    cmp_block_indices,
                    n_kv_blocks=n_cmp_kv_blocks,
                    q_tile_size=q_tile_size,
                    cmp_n_block_size=cmp_n_block_size,
                )
                O_cmp_pad, _ = _fa4_fwd(
                    Q_pad,
                    K_cmp_pad,
                    V_cmp_pad,
                    causal=False,
                    softmax_scale=softmax_scale,
                    mask_mod=compressed_mask,
                    block_sparse_tensors=cmp_sparse,
                )
            else:
                O_cmp_pad, _ = _fa4_fwd(
                    Q_pad,
                    K_cmp_pad,
                    V_cmp_pad,
                    causal=False,
                    softmax_scale=softmax_scale,
                    mask_mod=compressed_mask,
                )
        else:
            O_cmp_pad, _ = _fa4_fwd(
                Q_pad,
                K_cmp_pad,
                V_cmp_pad,
                causal=False,
                softmax_scale=softmax_scale,
                mask_mod=compressed_mask,
            )
        # Unpad compressed output to 3D
        O_cmp = Q.new_zeros(Q.shape[0], H, D_dim)
        for i, slen in enumerate(seqlens):
            s = cu_seqlens[i].item()
            O_cmp[s : s + slen] = O_cmp_pad[i, :slen]

        # Branch 2: Selected attention (varlen Q/K/V + block sparse)
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

        # Branch 3: Sliding window (varlen Q/K/V)
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

        # Step 5: Gating (element-wise, works on 3D)
        if gate_proj_weight is not None:
            O = fused_gate_and_combine(Q, O_cmp, O_slc, O_sld, gate_proj_weight)
        else:
            O = (O_cmp + O_slc + O_sld) * (1.0 / 3.0)
        return O

    # Fixed-length path
    B, N, H, D = Q.shape

    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(D)

    # Step 1: Compress KV
    K_cmp, V_cmp = fused_compress_kv(
        K, V, compress_block_size, W_k_compress, W_v_compress
    )
    # K_cmp, V_cmp: (B, N_cmp, H_kv, D) where N_cmp = N // compress_block_size

    # Step 2: Score blocks and select top-k
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
    )
    # block_indices: (B, H, N_q_tiles, k) int32

    # Step 3: Build FA4 sparsity masks for selected attention
    sparse_tensors = build_fa4_block_sparse_tensors(
        block_indices,
        compress_block_size,
        n_block_size=n_block_size,
        seqlen_k=N,
    )

    # Step 4: Run three FA4 branches

    # Branch 1: Compressed attention — block-aware causal masking on short KV
    # Standard causal=True gives wrong masking for compressed KV since N_cmp << N:
    # it masks based on kv_j > q_i, but we need kv_j * compress_block_size > q_i.
    compressed_mask = (
        _make_compressed_causal_mask(compress_block_size) if causal else None
    )

    if num_cmp_selected_blocks is not None:
        # Block-sparse compressed attention: select top-k FA4 blocks of K_cmp
        N_cmp = K_cmp.shape[1]
        cmp_n_block_size = n_block_size
        n_cmp_kv_blocks = (N_cmp + cmp_n_block_size - 1) // cmp_n_block_size
        if cmp_block_indices is not None:
            cmp_sparse = build_compressed_block_sparse_tensors(
                cmp_block_indices,
                n_kv_blocks=n_cmp_kv_blocks,
                q_tile_size=q_tile_size,
                cmp_n_block_size=cmp_n_block_size,
            )
            O_cmp, lse_cmp = _fa4_fwd(
                Q,
                K_cmp,
                V_cmp,
                causal=False,
                softmax_scale=softmax_scale,
                mask_mod=compressed_mask,
                block_sparse_tensors=cmp_sparse,
            )
        else:
            O_cmp, lse_cmp = _fa4_fwd(
                Q,
                K_cmp,
                V_cmp,
                causal=False,
                softmax_scale=softmax_scale,
                mask_mod=compressed_mask,
            )
    else:
        O_cmp, lse_cmp = _fa4_fwd(
            Q,
            K_cmp,
            V_cmp,
            causal=False,
            softmax_scale=softmax_scale,
            mask_mod=compressed_mask,
        )

    # Branch 2: Selected attention — block-sparse attention on full KV
    O_slc, lse_slc = _fa4_fwd(
        Q,
        K,
        V,
        causal=causal,
        softmax_scale=softmax_scale,
        block_sparse_tensors=sparse_tensors,
    )

    # Branch 3: Sliding window attention
    O_sld, lse_sld = _fa4_fwd(
        Q,
        K,
        V,
        causal=causal,
        softmax_scale=softmax_scale,
        window_size_left=window_size,
        window_size_right=0,
    )

    # Step 5: Gate and combine branch outputs
    if gate_proj_weight is not None:
        O = fused_gate_and_combine(Q, O_cmp, O_slc, O_sld, gate_proj_weight)
    else:
        # No gate weights — uniform 1/3 gates. Simple average avoids launching
        # the CuteDSL gating kernel (saves ~0.3-1ms at small N).
        O = (O_cmp + O_slc + O_sld) * (1.0 / 3.0)

    return O
