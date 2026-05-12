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
from mslk.attention.sparse_attn.select import fused_score_and_select_blocks
from mslk.attention.sparse_attn.sparsity_masks import build_fa4_block_sparse_tensors
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
) -> Tuple[Tensor, Tensor]:
    """Call FA4's forward pass (low-level, supports block sparsity and mask_mod).

    Uses _flash_attn_fwd from the fb interface which supports block_sparse_tensors
    and mask_mod.

    All inputs are (B, N, H, D) layout.
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
) -> Tuple[Tensor, Tensor, Tensor]:
    """Call FA4's backward pass (low-level, supports block sparsity and mask_mod).

    Uses _flash_attn_bwd from the fb interface which supports block_sparse_tensors
    and mask_mod.

    All inputs are (B, N, H, D) layout.
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
    )
    return dq, dk, dv


def _fa4_fwd_simple(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    causal: bool = False,
    softmax_scale: float | None = None,
    window_size: tuple[int, int] | None = None,
) -> Tuple[Tensor, Tensor]:
    """Call FA4 via the autograd interface (for branches without block sparsity)."""
    from mslk.fb.mslk.attention.flash_attn.autograd_interface import flash_attn_func

    return flash_attn_func(
        q,
        k,
        v,
        softmax_scale=softmax_scale,
        causal=causal,
        window_size=window_size,
    )


def nsa_forward(
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
    """NSA forward pass using FA4 for all attention branches.

    Orchestrates:
    1. KV compression (mean-pool + optional projection)
    2. Block scoring and top-k selection
    3. Three FA4 calls: compressed, selected (block-sparse), sliding window
    4. Gating and combination

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
        q_tile_size: Query tile size for block selection (FA4 CTA tile: 2*128=256).
        n_block_size: FA4 KV block size (128 for SM100).

    Returns:
        Output tensor, shape (B, N, H, D).
    """
    B, N, H, D = Q.shape

    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(D)

    # Step 1: Compress KV
    K_cmp, V_cmp = fused_compress_kv(
        K, V, compress_block_size, W_k_compress, W_v_compress
    )
    # K_cmp, V_cmp: (B, N_cmp, H_kv, D) where N_cmp = N // compress_block_size

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
    O_sld, lse_sld = _fa4_fwd_simple(
        Q,
        K,
        V,
        causal=causal,
        softmax_scale=softmax_scale,
        window_size=(window_size, 0),  # (left, right=0 for causal)
    )

    # Step 5: Gate and combine branch outputs
    if gate_proj_weight is not None:
        O = fused_gate_and_combine(Q, O_cmp, O_slc, O_sld, gate_proj_weight)
    else:
        # No gate weights — uniform 1/3 gates. Simple average avoids launching
        # the CuteDSL gating kernel (saves ~0.3-1ms at small N).
        O = (O_cmp + O_slc + O_sld) * (1.0 / 3.0)

    return O
