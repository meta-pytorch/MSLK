# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

"""Block scoring and top-k selection for NSA."""

from __future__ import annotations

import math

import torch
from torch import Tensor


def score_and_select_blocks(
    Q: Tensor,  # (B, N, H, D)
    K_cmp: Tensor,  # (B, N_cmp, H_kv, D)
    num_selected_blocks: int,
    compress_block_size: int,
    causal: bool = True,
    q_tile_size: int = 256,
    softmax_scale: float | None = None,
) -> Tensor:
    """Score KV blocks by importance and select top-k for each query tile.

    Uses compressed keys to efficiently compute block importance scores.
    Each query tile (q_tile_size positions) independently selects its own
    top-k KV blocks.

    Processes query tiles in chunks to avoid materializing a full (B, H, N, N_cmp)
    score tensor, reducing peak memory from O(N * N_cmp) to
    O(chunk_tiles * q_tile_size * N_cmp).

    Args:
        Q: Query tensor, shape (B, N, H, D).
        K_cmp: Compressed key tensor, shape (B, N_cmp, H_kv, D).
        num_selected_blocks: Number of KV blocks to select per query tile.
        compress_block_size: Size of each KV block.
        causal: Whether to apply causal masking (prevent selecting future blocks).
        q_tile_size: Size of each query tile (should match FA4's CTA tile: 2 * m_block_size = 256).
        softmax_scale: Scaling factor for attention scores.

    Returns:
        block_indices: Selected KV block indices, shape (B, H, N_q_tiles, k).
            Each entry is an index into the compressed KV sequence.
    """
    B, N, H, D = Q.shape
    H_kv = K_cmp.shape[2]
    N_cmp = K_cmp.shape[1]
    groups = H // H_kv

    N_q_tiles = N // q_tile_size
    assert N % q_tile_size == 0, (
        f"Sequence length {N} must be divisible by q_tile_size {q_tile_size}"
    )

    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(D)

    k_actual = min(num_selected_blocks, N_cmp)

    # Expand K_cmp for GQA: (B, N_cmp, H_kv, D) -> (B, N_cmp, H, D)
    if groups > 1:
        K_cmp_expanded = K_cmp.repeat_interleave(groups, dim=2)
    else:
        K_cmp_expanded = K_cmp

    # Pre-transpose for efficient matmul: (B, H, D, N_cmp)
    K_cmp_t = K_cmp_expanded.permute(0, 2, 3, 1).float()

    block_indices = torch.empty(
        B, H, N_q_tiles, k_actual, dtype=torch.int32, device=Q.device
    )

    # Precompute causal mask data (tiny tensors)
    if causal:
        q_tile_end = (torch.arange(N_q_tiles, device=Q.device) + 1) * q_tile_size
        kv_block_start = torch.arange(N_cmp, device=Q.device) * compress_block_size

    # Process query tiles in chunks to bound peak memory.
    # Each chunk computes (B, H, chunk_tiles * q_tile_size, N_cmp) scores,
    # averages per tile, applies causal mask, and runs topk.
    chunk_tiles = min(16, N_q_tiles)
    for chunk_start in range(0, N_q_tiles, chunk_tiles):
        chunk_end = min(chunk_start + chunk_tiles, N_q_tiles)
        n_tiles = chunk_end - chunk_start
        q_start = chunk_start * q_tile_size
        q_end = chunk_end * q_tile_size

        # (B, H, n_tiles * q_tile_size, N_cmp) — bounded, not O(N²)
        scores_chunk = (
            torch.matmul(Q[:, q_start:q_end].permute(0, 2, 1, 3).float(), K_cmp_t)
            * softmax_scale
        )

        # Average per tile: (B, H, n_tiles, N_cmp)
        scores_agg = scores_chunk.reshape(B, H, n_tiles, q_tile_size, N_cmp).mean(dim=3)

        # Apply causal mask: block j is future if j * block_size >= (i+1) * q_tile_size
        if causal:
            future_mask = kv_block_start.unsqueeze(0) >= q_tile_end[
                chunk_start:chunk_end
            ].unsqueeze(1)  # (n_tiles, N_cmp)
            scores_agg.masked_fill_(
                future_mask.unsqueeze(0).unsqueeze(0), float("-inf")
            )

        # Select top-k blocks for this chunk
        topk_scores, chunk_idx = scores_agg.topk(k_actual, dim=-1)

        # Replace invalid selections (future blocks with -inf score) with block 0
        invalid = topk_scores == float("-inf")
        if invalid.any():
            chunk_idx = chunk_idx.masked_fill(invalid, 0)

        # Sort indices for better memory access patterns
        block_indices[:, :, chunk_start:chunk_end] = chunk_idx.sort(dim=-1).values.to(
            torch.int32
        )

    return block_indices


def _compute_q_mean(
    Q: Tensor,
    q_tile_size: int,
    softmax_scale: float,
    cu_seqlens: Tensor | None = None,
    max_seqlen: int | None = None,
) -> tuple[Tensor, int, int, int, int]:
    """Compute per-tile Q_mean for block scoring.

    Shared helper for both selected-branch and compressed-branch selectors.

    Args:
        Q: Query tensor, (B, N, H, D) or (total_tokens, H, D) for varlen.
        q_tile_size: Query tile size (typically 256).
        softmax_scale: Attention scaling factor.
        cu_seqlens: Cumulative sequence lengths for varlen.
        max_seqlen: Max sequence length for varlen.

    Returns:
        Q_mean: Scaled per-tile mean, shape (B, N_q_tiles, H, D) in float32.
        B: Batch size.
        N_q_tiles: Number of query tiles.
        H: Number of query heads.
        D: Head dimension.
    """
    is_varlen = cu_seqlens is not None

    if is_varlen:
        assert Q.dim() == 3, f"Varlen requires 3D Q, got {Q.dim()}D"
        H = Q.shape[1]
        D = Q.shape[2]
        batch_size = cu_seqlens.shape[0] - 1
        seqlens = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
        if max_seqlen is None:
            max_seqlen = max(seqlens)
        N_q_tiles = max_seqlen // q_tile_size
        B = batch_size

        Q_mean_parts = []
        for i, slen in enumerate(seqlens):
            s = cu_seqlens[i].item()
            n_tiles_i = slen // q_tile_size
            if n_tiles_i > 0:
                Q_mean_i = (
                    Q[s : s + n_tiles_i * q_tile_size]
                    .reshape(n_tiles_i, q_tile_size, H, D)
                    .mean(dim=1)
                    .float()
                )
            else:
                Q_mean_i = Q.new_zeros(0, H, D, dtype=torch.float32)
            Q_mean_parts.append(Q_mean_i)

        # Pad to (B, max_N_q_tiles, H, D)
        Q_mean = Q.new_zeros(B, N_q_tiles, H, D, dtype=torch.float32)
        for i, qm in enumerate(Q_mean_parts):
            if qm.shape[0] > 0:
                Q_mean[i, : qm.shape[0]] = qm

        Q_mean = Q_mean * softmax_scale
    else:
        B, N, H, D = Q.shape
        N_q_tiles = N // q_tile_size
        assert N % q_tile_size == 0, (
            f"Sequence length {N} must be divisible by q_tile_size {q_tile_size}"
        )
        Q_mean = (
            Q.reshape(B, N_q_tiles, q_tile_size, H, D).mean(dim=2).float()
            * softmax_scale
        )

    return Q_mean, B, N_q_tiles, H, D


def _score_and_topk(
    Q_mean: Tensor,  # (B, N_q_tiles, H, D) float32, already scaled
    K_cmp_t: Tensor,  # (B*H_kv, D, N_cmp) float32
    B: int,
    H: int,
    H_kv: int,
    N_q_tiles: int,
    N_cmp: int,
    k: int,
    causal: bool,
    compress_block_size: int,
    q_tile_size: int,
    chunk_tiles: int = 64,
    device: torch.device | None = None,
) -> Tensor:
    """Score Q_mean against K_cmp and select top-k compressed blocks.

    Uses GQA-aware bmm: Q groups are folded into the M dimension of the GEMM,
    avoiding K_cmp expansion from H_kv to H heads.

    Returns:
        block_indices: (B, H, N_q_tiles, k) int32.
    """
    groups = H // H_kv
    D = Q_mean.shape[-1]

    if device is None:
        device = Q_mean.device

    k_actual = min(k, N_cmp)

    # Causal mask data
    if causal:
        q_tile_end = (torch.arange(N_q_tiles, device=device) + 1) * q_tile_size
        kv_block_start = torch.arange(N_cmp, device=device) * compress_block_size

    block_indices = torch.empty(
        B, H, N_q_tiles, k_actual, dtype=torch.int32, device=device
    )

    for chunk_start in range(0, N_q_tiles, chunk_tiles):
        chunk_end = min(chunk_start + chunk_tiles, N_q_tiles)
        n_tiles = chunk_end - chunk_start

        # Q_mean chunk: (B, n_tiles, H, D) -> (B, n_tiles, H_kv, groups, D)
        # -> (B*H_kv, n_tiles*groups, D) for bmm
        q_chunk = Q_mean[:, chunk_start:chunk_end]
        q_grouped = q_chunk.reshape(B, n_tiles, H_kv, groups, D)
        q_bmm = q_grouped.permute(0, 2, 1, 3, 4).reshape(B * H_kv, n_tiles * groups, D)

        # bmm: (B*H_kv, n_tiles*groups, D) @ (B*H_kv, D, N_cmp)
        scores_flat = torch.bmm(q_bmm, K_cmp_t)

        # Reshape to (B, H, n_tiles, N_cmp)
        scores = (
            scores_flat.reshape(B, H_kv, n_tiles, groups, N_cmp)
            .permute(0, 2, 1, 3, 4)
            .reshape(B, n_tiles, H, N_cmp)
            .permute(0, 2, 1, 3)
        )

        # Causal mask
        if causal:
            future_mask = kv_block_start.unsqueeze(0) >= q_tile_end[
                chunk_start:chunk_end
            ].unsqueeze(1)
            scores.masked_fill_(future_mask.unsqueeze(0).unsqueeze(0), float("-inf"))

        # Top-k
        topk_scores, chunk_idx = scores.topk(k_actual, dim=-1)

        # Replace -inf selections with block 0
        invalid = topk_scores == float("-inf")
        if invalid.any():
            chunk_idx = chunk_idx.masked_fill(invalid, 0)

        # Sort ascending
        block_indices[:, :, chunk_start:chunk_end] = chunk_idx.sort(dim=-1).values.to(
            torch.int32
        )

    return block_indices


def fused_score_and_select_blocks(
    Q: Tensor,  # (B, N, H, D) or (total_tokens, H, D) for varlen
    K_cmp: Tensor,  # (B, N_cmp, H_kv, D)
    num_selected_blocks: int,
    compress_block_size: int,
    causal: bool = True,
    q_tile_size: int = 256,
    softmax_scale: float | None = None,
    cu_seqlens: Tensor | None = None,  # (B+1,) int32 for varlen
    max_seqlen: int | None = None,
) -> Tensor:
    """Fused block scoring and top-k selection using GEMM + topk.

    Uses the Q_mean optimization: mean(Q @ K) = mean(Q) @ K, reducing scoring
    from a full GEMM to a GEMV per query tile (256x fewer FLOPs).

    Q_mean is computed in PyTorch (single kernel), then cuBLAS GEMM computes
    scores, and torch.topk selects the top-k blocks. Chunked over Q tiles
    to bound peak memory.

    For varlen (3D + cu_seqlens): computes Q_mean per-sequence and pads to
    max_N_q_tiles.

    Args:
        Q: Query tensor, shape (B, N, H, D).
        K_cmp: Compressed key tensor, shape (B, N_cmp, H_kv, D).
        num_selected_blocks: Number of KV blocks to select per query tile.
        compress_block_size: Size of each KV block.
        causal: Whether to apply causal masking.
        q_tile_size: Size of each query tile.
        softmax_scale: Scaling factor for attention scores.

    Returns:
        block_indices: Selected KV block indices, shape (B, H, N_q_tiles, k).
    """
    H_kv = K_cmp.shape[2]
    N_cmp = K_cmp.shape[1]

    if softmax_scale is None:
        D = Q.shape[-1]
        softmax_scale = 1.0 / math.sqrt(D)

    Q_mean, B, N_q_tiles, H, D = _compute_q_mean(
        Q, q_tile_size, softmax_scale, cu_seqlens, max_seqlen
    )

    # K_cmp: (B, N_cmp, H_kv, D) -> (B*H_kv, D, N_cmp) for GQA-aware bmm
    K_cmp_t = K_cmp.permute(0, 2, 3, 1).reshape(B * H_kv, D, N_cmp).float()

    return _score_and_topk(
        Q_mean,
        K_cmp_t,
        B,
        H,
        H_kv,
        N_q_tiles,
        N_cmp,
        k=num_selected_blocks,
        causal=causal,
        compress_block_size=compress_block_size,
        q_tile_size=q_tile_size,
        device=Q.device,
    )


def fused_score_and_select_all(
    Q: Tensor,  # (B, N, H, D) or (total_tokens, H, D) for varlen
    K_cmp: Tensor,  # (B, N_cmp, H_kv, D)
    num_selected_blocks: int,
    compress_block_size: int,
    num_cmp_selected_blocks: int | None = None,
    cmp_n_block_size: int = 128,
    causal: bool = True,
    q_tile_size: int = 256,
    softmax_scale: float | None = None,
    cu_seqlens: Tensor | None = None,
    max_seqlen: int | None = None,
) -> tuple[Tensor, Tensor | None]:
    """Score once, select for both selected and compressed branches.

    Computes Q_mean and the Q_mean x K_cmp GEMM once, then derives:
    1. Selected-branch block indices (top-k over compressed tokens)
    2. Compressed-branch FA4 block indices (aggregate + top-k over FA4 blocks)

    This eliminates the duplicate GEMM that occurs when calling
    fused_score_and_select_blocks and select_compressed_blocks separately.

    Args:
        Q: Query tensor, (B, N, H, D) or (total_tokens, H, D) for varlen.
        K_cmp: Compressed key tensor, (B, N_cmp, H_kv, D).
        num_selected_blocks: Number of compressed blocks for selected branch.
        compress_block_size: Compression block size (e.g. 64).
        num_cmp_selected_blocks: Number of FA4 blocks for compressed branch.
            If None, only selected-branch indices are computed.
        cmp_n_block_size: FA4 KV block size for compressed branch (128).
        causal: Whether to apply causal masking.
        q_tile_size: Query tile size (256).
        softmax_scale: Attention scaling factor.
        cu_seqlens: Cumulative sequence lengths for varlen.
        max_seqlen: Max sequence length for varlen.

    Returns:
        block_indices: (B, H, N_q_tiles, k) int32 for selected branch.
        cmp_block_indices: (B, H, N_q_tiles, k_cmp) int32 for compressed
            branch, or None if num_cmp_selected_blocks is None.
    """
    H_kv = K_cmp.shape[2]
    N_cmp = K_cmp.shape[1]

    if softmax_scale is None:
        D = Q.shape[-1]
        softmax_scale = 1.0 / math.sqrt(D)

    # --- Shared: compute Q_mean once ---
    Q_mean, B, N_q_tiles, H, D = _compute_q_mean(
        Q, q_tile_size, softmax_scale, cu_seqlens, max_seqlen
    )

    # --- Shared: prepare K_cmp once ---
    K_cmp_t = K_cmp.permute(0, 2, 3, 1).reshape(B * H_kv, D, N_cmp).float()

    # --- Check if compressed branch selection is needed ---
    need_cmp = False
    if num_cmp_selected_blocks is not None:
        n_kv_blocks = N_cmp // cmp_n_block_size
        if n_kv_blocks == 0:
            n_kv_blocks = 1
        if n_kv_blocks > num_cmp_selected_blocks:
            need_cmp = True
            k_cmp_actual = min(num_cmp_selected_blocks, n_kv_blocks)

    if not need_cmp:
        # Only selected branch — delegate to existing function
        block_indices = _score_and_topk(
            Q_mean,
            K_cmp_t,
            B,
            H,
            H_kv,
            N_q_tiles,
            N_cmp,
            k=num_selected_blocks,
            causal=causal,
            compress_block_size=compress_block_size,
            q_tile_size=q_tile_size,
            device=Q.device,
        )
        return block_indices, None

    # --- Shared GEMM loop: score once, derive both outputs ---
    groups = H // H_kv
    k_sel_actual = min(num_selected_blocks, N_cmp)

    # Causal mask data for selected branch (per compressed token)
    if causal:
        q_tile_end = (torch.arange(N_q_tiles, device=Q.device) + 1) * q_tile_size
        kv_block_start_sel = torch.arange(N_cmp, device=Q.device) * compress_block_size
        kv_block_start_cmp = (
            torch.arange(n_kv_blocks, device=Q.device)
            * cmp_n_block_size
            * compress_block_size
        )

    block_indices = torch.empty(
        B, H, N_q_tiles, k_sel_actual, dtype=torch.int32, device=Q.device
    )
    cmp_block_indices = torch.empty(
        B, H, N_q_tiles, k_cmp_actual, dtype=torch.int32, device=Q.device
    )

    chunk_tiles = min(64, N_q_tiles)
    for chunk_start in range(0, N_q_tiles, chunk_tiles):
        chunk_end = min(chunk_start + chunk_tiles, N_q_tiles)
        n_tiles = chunk_end - chunk_start

        # --- SHARED GEMM: compute scores once ---
        q_chunk = Q_mean[:, chunk_start:chunk_end]
        q_grouped = q_chunk.reshape(B, n_tiles, H_kv, groups, D)
        q_bmm = q_grouped.permute(0, 2, 1, 3, 4).reshape(B * H_kv, n_tiles * groups, D)
        scores_flat = torch.bmm(q_bmm, K_cmp_t)

        # Reshape to (B, H, n_tiles, N_cmp)
        scores = (
            scores_flat.reshape(B, H_kv, n_tiles, groups, N_cmp)
            .permute(0, 2, 1, 3, 4)
            .reshape(B, n_tiles, H, N_cmp)
            .permute(0, 2, 1, 3)
        )

        # --- Selected branch: per-token causal mask + top-k ---
        sel_scores = scores.clone()
        if causal:
            future_mask = kv_block_start_sel.unsqueeze(0) >= q_tile_end[
                chunk_start:chunk_end
            ].unsqueeze(1)
            sel_scores.masked_fill_(
                future_mask.unsqueeze(0).unsqueeze(0), float("-inf")
            )

        topk_scores, chunk_idx = sel_scores.topk(k_sel_actual, dim=-1)
        invalid = topk_scores == float("-inf")
        if invalid.any():
            chunk_idx = chunk_idx.masked_fill(invalid, 0)
        block_indices[:, :, chunk_start:chunk_end] = chunk_idx.sort(dim=-1).values.to(
            torch.int32
        )

        # --- Compressed branch: aggregate per FA4 block + top-k ---
        if N_cmp % cmp_n_block_size == 0:
            block_scores = scores.reshape(
                B, H, n_tiles, n_kv_blocks, cmp_n_block_size
            ).mean(dim=-1)
        else:
            pad = cmp_n_block_size - (N_cmp % cmp_n_block_size)
            scores_padded = torch.nn.functional.pad(
                scores, (0, pad), value=float("-inf")
            )
            block_scores = scores_padded.reshape(
                B, H, n_tiles, n_kv_blocks, cmp_n_block_size
            ).mean(dim=-1)

        if causal:
            future_mask_cmp = kv_block_start_cmp.unsqueeze(0) >= q_tile_end[
                chunk_start:chunk_end
            ].unsqueeze(1)
            block_scores.masked_fill_(
                future_mask_cmp.unsqueeze(0).unsqueeze(0), float("-inf")
            )

        topk_scores_cmp, chunk_idx_cmp = block_scores.topk(k_cmp_actual, dim=-1)
        invalid_cmp = topk_scores_cmp == float("-inf")
        if invalid_cmp.any():
            chunk_idx_cmp = chunk_idx_cmp.masked_fill(invalid_cmp, 0)
        cmp_block_indices[:, :, chunk_start:chunk_end] = chunk_idx_cmp.sort(
            dim=-1
        ).values.to(torch.int32)

    return block_indices, cmp_block_indices
