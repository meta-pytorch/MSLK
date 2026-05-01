# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

"""Pure PyTorch reference implementation of NSA (Native Sparse Attention).

This serves as a correctness reference for the optimized FA4-based implementation.
All operations are done in PyTorch without custom kernels.

The forward pass is fully differentiable (all tensor ops, no Python loops that
break autograd), so PyTorch autograd provides the reference backward pass.
Use nsa_backward_reference() to compute gradients for validation.

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
    _block_indices: Tensor | None = None,
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
    # Use at least float32 for numerical stability, but preserve float64 for gradcheck
    compute_dtype = torch.float64 if Q.dtype == torch.float64 else torch.float32
    q = Q.transpose(1, 2).to(compute_dtype)  # (B, H, N, D)
    k = K.transpose(1, 2).to(compute_dtype)  # (B, H_kv, N, D)
    v = V.transpose(1, 2).to(compute_dtype)  # (B, H_kv, N, D)

    # Expand K, V for GQA: (B, H_kv, N, D) -> (B, H, N, D)
    if groups > 1:
        k = k.repeat_interleave(groups, dim=1)
        v = v.repeat_interleave(groups, dim=1)

    # --- Branch 1: Compressed Attention ---
    O_cmp = _compressed_attention(
        q,
        k,
        v,
        compress_block_size,
        W_k_compress,
        W_v_compress,
        groups,
        softmax_scale,
        causal,
        B,
        N,
        H,
        H_kv,
        D,
    )

    # --- Branch 2: Selected (Block-Sparse) Attention ---
    O_slc = _selected_attention(
        q,
        k,
        v,
        compress_block_size,
        num_selected_blocks,
        W_k_compress,
        groups,
        softmax_scale,
        causal,
        B,
        N,
        H,
        H_kv,
        D,
        q_tile_size=q_tile_size,
        _block_indices=_block_indices,
    )

    # --- Branch 3: Sliding Window Attention ---
    O_sld = _sliding_window_attention(
        q,
        k,
        v,
        window_size,
        softmax_scale,
        causal,
        B,
        N,
        H,
        D,
    )

    # --- Gate and Combine ---
    # All outputs are (B, H, N, D), transpose to (B, N, H, D)
    O_cmp = O_cmp.transpose(1, 2)
    O_slc = O_slc.transpose(1, 2)
    O_sld = O_sld.transpose(1, 2)

    if gate_proj_weight is not None:
        # gate_proj_weight: (H, 3, D)
        # Q: (B, N, H, D) -> compute per-head gates
        gates = torch.einsum(
            "bnhd,hgd->bnhg", Q.to(compute_dtype), gate_proj_weight.to(compute_dtype)
        )
        gates = gates.sigmoid()  # (B, N, H, 3)
    else:
        gates = torch.ones(B, N, H, 3, device=Q.device, dtype=compute_dtype) / 3.0

    O = gates[..., 0:1] * O_cmp + gates[..., 1:2] * O_slc + gates[..., 2:3] * O_sld

    return O.to(Q.dtype)


def _compressed_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    block_size: int,
    W_k_compress: Tensor | None,
    W_v_compress: Tensor | None,
    groups: int,
    softmax_scale: float,
    causal: bool,
    B: int,
    N: int,
    H: int,
    H_kv: int,
    D: int,
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
        k_cmp = torch.einsum("bhnd,hde->bhne", k_cmp, W_k_compress.to(k_cmp.dtype))
    if W_v_compress is not None:
        v_cmp = torch.einsum("bhnd,hde->bhne", v_cmp, W_v_compress.to(v_cmp.dtype))

    # Expand for GQA
    if groups > 1:
        k_cmp = k_cmp.repeat_interleave(groups, dim=1)
        v_cmp = v_cmp.repeat_interleave(groups, dim=1)

    # Standard attention on compressed KV: (B, H, N, D) x (B, H, N_cmp, D)
    scores = (
        torch.matmul(q, k_cmp.transpose(-2, -1)) * softmax_scale
    )  # (B, H, N, N_cmp)

    if causal:
        # A query at position i can attend to compressed block j if the block
        # starts at or before position i: j * block_size <= i.
        q_pos = torch.arange(N, device=q.device).unsqueeze(1)  # (N, 1)
        kv_block_start = (torch.arange(N_cmp, device=q.device)) * block_size  # (N_cmp,)
        causal_mask = kv_block_start.unsqueeze(0) > q_pos  # (N, N_cmp)
        scores = scores.masked_fill(
            causal_mask.unsqueeze(0).unsqueeze(0), float("-inf")
        )

    attn = torch.softmax(scores, dim=-1)
    out = torch.matmul(attn, v_cmp)  # (B, H, N, D)
    return out


def _selected_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    block_size: int,
    num_selected: int,
    W_k_compress: Tensor | None,
    groups: int,
    softmax_scale: float,
    causal: bool,
    B: int,
    N: int,
    H: int,
    H_kv: int,
    D: int,
    q_tile_size: int = 256,
    _block_indices: Tensor | None = None,
) -> Tensor:
    """Selected block-sparse attention: attend to top-k important KV blocks.

    Aggregates block scores per q_tile_size (matching FA4's CTA tile) and
    selects blocks per query tile rather than per query block.

    All operations use tensor ops (no Python loops) so autograd can
    differentiate through this function w.r.t. q, k, v.  The top-k block
    indices are non-differentiable (stop-gradient) by design.

    Args:
        _block_indices: If provided, use these pre-computed block indices
            instead of computing them from scores. Shape (B, H, N_q_tiles, k).
            Useful for freezing selection during gradcheck.
    """
    N_blocks = N // block_size
    N_q_tiles = N // q_tile_size
    k_actual = min(num_selected, N_blocks)

    if _block_indices is not None:
        top_indices = _block_indices
    else:
        # Score blocks using compressed keys
        k_orig = k[:, ::groups]  # (B, H_kv, N, D)
        k_blocks = k_orig.reshape(B, H_kv, N_blocks, block_size, D)
        k_cmp = k_blocks.mean(dim=3)  # (B, H_kv, N_blocks, D)
        if W_k_compress is not None:
            k_cmp = torch.einsum("bhnd,hde->bhne", k_cmp, W_k_compress.to(k_cmp.dtype))
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

        # Select top-k blocks per query tile (non-differentiable, stop-gradient)
        _, top_indices = block_scores_agg.topk(k_actual, dim=-1)  # (B, H, N_q_tiles, k)

    # Expand block indices to position-level indices for gather
    # Each block index i maps to positions [i*block_size, ..., (i+1)*block_size - 1]
    offsets = torch.arange(block_size, device=q.device)  # (block_size,)
    # (B, H, N_q_tiles, k, block_size)
    pos_indices = top_indices.unsqueeze(-1) * block_size + offsets
    kv_len = k_actual * block_size
    pos_indices = pos_indices.reshape(B, H, N_q_tiles, kv_len)

    # Gather K and V at selected positions using differentiable gather
    # k, v: (B, H, N, D) -> gather along N dimension per query tile
    gather_idx = pos_indices.unsqueeze(-1).expand(
        B, H, N_q_tiles, kv_len, D
    )  # (B, H, N_q_tiles, kv_len, D)
    k_exp = k.unsqueeze(2).expand(B, H, N_q_tiles, N, D)
    v_exp = v.unsqueeze(2).expand(B, H, N_q_tiles, N, D)
    sel_k = torch.gather(k_exp, 3, gather_idx)  # (B, H, N_q_tiles, kv_len, D)
    sel_v = torch.gather(v_exp, 3, gather_idx)

    # Reshape q into tiles and compute attention
    q_tiles = q.reshape(B, H, N_q_tiles, q_tile_size, D)
    scores = (
        torch.matmul(q_tiles, sel_k.transpose(-2, -1)) * softmax_scale
    )  # (B, H, N_q_tiles, q_tile_size, kv_len)

    if causal:
        # q positions per tile: (N_q_tiles, q_tile_size)
        q_tile_starts = torch.arange(N_q_tiles, device=q.device) * q_tile_size
        q_offsets = torch.arange(q_tile_size, device=q.device)
        q_positions = q_tile_starts.unsqueeze(1) + q_offsets.unsqueeze(0)
        # causal mask: kv_pos > q_pos
        # pos_indices: (B, H, N_q_tiles, kv_len) -> (B, H, N_q_tiles, 1, kv_len)
        # q_positions: (N_q_tiles, q_tile_size) -> (1, 1, N_q_tiles, q_tile_size, 1)
        c_mask = pos_indices.unsqueeze(3) > q_positions.reshape(
            1, 1, N_q_tiles, q_tile_size, 1
        )
        scores = scores.masked_fill(c_mask, float("-inf"))

    attn = torch.softmax(scores, dim=-1)
    out_tiles = torch.matmul(attn, sel_v)  # (B, H, N_q_tiles, q_tile_size, D)
    return out_tiles.reshape(B, H, N, D)


def _sliding_window_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    window_size: int,
    softmax_scale: float,
    causal: bool,
    B: int,
    N: int,
    H: int,
    D: int,
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


# ---------------------------------------------------------------------------
# Reference backward (via autograd on the differentiable forward)
# ---------------------------------------------------------------------------


def compute_block_indices_reference(
    Q: Tensor,  # (B, N, H, D)
    K: Tensor,  # (B, N, H_kv, D)
    compress_block_size: int = 64,
    num_selected_blocks: int = 16,
    W_k_compress: Tensor | None = None,
    causal: bool = True,
    softmax_scale: float | None = None,
    q_tile_size: int = 256,
) -> Tensor:
    """Pre-compute block selection indices (non-differentiable).

    Returns block indices that can be passed to nsa_forward_reference via
    _block_indices to freeze the selection during gradient computation.

    Returns:
        top_indices: (B, H, N_q_tiles, k) int64 tensor of selected block indices.
    """
    B, N, H, D = Q.shape
    H_kv = K.shape[2]
    groups = H // H_kv

    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(D)

    q = Q.transpose(1, 2).float()  # (B, H, N, D)
    k = K.transpose(1, 2).float()  # (B, H_kv, N, D)

    N_blocks = N // compress_block_size
    N_q_tiles = N // q_tile_size

    # K already has H_kv heads — no need to slice with ::groups
    k_blocks = k.reshape(B, H_kv, N_blocks, compress_block_size, D)
    k_cmp = k_blocks.mean(dim=3)
    if W_k_compress is not None:
        k_cmp = torch.einsum("bhnd,hde->bhne", k_cmp, W_k_compress.float())
    if groups > 1:
        k_cmp = k_cmp.repeat_interleave(groups, dim=1)

    block_scores = torch.matmul(q, k_cmp.transpose(-2, -1)) * softmax_scale
    block_scores_agg = block_scores.reshape(
        B, H, N_q_tiles, q_tile_size, N_blocks
    ).mean(dim=3)

    if causal:
        q_tile_end = (torch.arange(N_q_tiles, device=Q.device) + 1) * q_tile_size
        kv_block_start = torch.arange(N_blocks, device=Q.device) * compress_block_size
        future_mask = kv_block_start.unsqueeze(0) >= q_tile_end.unsqueeze(1)
        block_scores_agg = block_scores_agg.masked_fill(
            future_mask.unsqueeze(0).unsqueeze(0), float("-inf")
        )

    k_actual = min(num_selected_blocks, N_blocks)
    _, top_indices = block_scores_agg.topk(k_actual, dim=-1)
    return top_indices


def nsa_backward_reference(
    Q: Tensor,  # (B, N, H, D)
    K: Tensor,  # (B, N, H_kv, D)
    V: Tensor,  # (B, N, H_kv, D)
    dO: Tensor,  # (B, N, H, D)
    compress_block_size: int = 64,
    num_selected_blocks: int = 16,
    window_size: int = 512,
    W_k_compress: Tensor | None = None,  # (H_kv, D, D)
    W_v_compress: Tensor | None = None,  # (H_kv, D, D)
    gate_proj_weight: Tensor | None = None,  # (H, 3, D)
    causal: bool = True,
    softmax_scale: float | None = None,
    q_tile_size: int = 256,
    _block_indices: Tensor | None = None,
) -> dict[str, Tensor | None]:
    """Compute NSA gradients using autograd on the differentiable reference forward.

    Runs nsa_forward_reference with requires_grad=True inputs, then calls
    torch.autograd.grad to compute dQ, dK, dV, and optionally dW_k_compress,
    dW_v_compress, dgate_proj_weight.

    Args:
        Q, K, V: Input tensors (same as nsa_forward_reference).
        dO: Upstream gradient, shape (B, N, H, D).
        Other args: Same as nsa_forward_reference.

    Returns:
        Dictionary with keys: "dQ", "dK", "dV", "dW_k_compress",
        "dW_v_compress", "dgate_proj_weight".  Values are None for
        parameters that were not provided.
    """
    # Clone inputs and enable gradients
    Q_g = Q.detach().float().requires_grad_(True)
    K_g = K.detach().float().requires_grad_(True)
    V_g = V.detach().float().requires_grad_(True)

    params = []
    W_k_g = None
    if W_k_compress is not None:
        W_k_g = W_k_compress.detach().float().requires_grad_(True)
        params.append(W_k_g)
    W_v_g = None
    if W_v_compress is not None:
        W_v_g = W_v_compress.detach().float().requires_grad_(True)
        params.append(W_v_g)
    gate_g = None
    if gate_proj_weight is not None:
        gate_g = gate_proj_weight.detach().float().requires_grad_(True)
        params.append(gate_g)

    O = nsa_forward_reference(
        Q_g,
        K_g,
        V_g,
        compress_block_size=compress_block_size,
        num_selected_blocks=num_selected_blocks,
        window_size=window_size,
        W_k_compress=W_k_g,
        W_v_compress=W_v_g,
        gate_proj_weight=gate_g,
        causal=causal,
        softmax_scale=softmax_scale,
        q_tile_size=q_tile_size,
        _block_indices=_block_indices,
    )

    grad_inputs = [Q_g, K_g, V_g] + params
    grads = torch.autograd.grad(
        O,
        grad_inputs,
        grad_outputs=dO.float(),
        allow_unused=True,
    )

    result: dict[str, Tensor | None] = {
        "dQ": grads[0].to(Q.dtype),
        "dK": grads[1].to(K.dtype),
        "dV": grads[2].to(V.dtype),
        "dW_k_compress": None,
        "dW_v_compress": None,
        "dgate_proj_weight": None,
    }

    idx = 3
    if W_k_compress is not None:
        result["dW_k_compress"] = grads[idx].to(W_k_compress.dtype)
        idx += 1
    if W_v_compress is not None:
        result["dW_v_compress"] = grads[idx].to(W_v_compress.dtype)
        idx += 1
    if gate_proj_weight is not None:
        result["dgate_proj_weight"] = grads[idx].to(gate_proj_weight.dtype)

    return result
