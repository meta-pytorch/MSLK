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
from torch import Tensor


def nsa_forward_reference(
    Q: Tensor,  # (B, N, H, D)
    K: Tensor,  # (B, N, H_kv, D)
    V: Tensor,  # (B, N, H_kv, D)
    compress_block_size: int = 64,
    num_selected_blocks: int = 16,
    window_size: int = 512,
    W_k_compress: Tensor | None = None,
    W_v_compress: Tensor | None = None,
    gate_proj_weight: Tensor | None = None,
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
    """
    B, N, H, D = Q.shape
    H_kv = K.shape[2]
    assert H % H_kv == 0, "H must be divisible by H_kv for GQA"
    groups = H // H_kv

    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(D)

    # fp32 accumulation for numerical stability with bf16/fp16 inputs
    compute_dtype = torch.float64 if Q.dtype == torch.float64 else torch.float32
    q = Q.transpose(1, 2).to(compute_dtype)
    k = K.transpose(1, 2).to(compute_dtype)
    v = V.transpose(1, 2).to(compute_dtype)

    if groups > 1:
        k = k.repeat_interleave(groups, dim=1)
        v = v.repeat_interleave(groups, dim=1)

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

    O_cmp = O_cmp.transpose(1, 2)
    O_slc = O_slc.transpose(1, 2)
    O_sld = O_sld.transpose(1, 2)

    if gate_proj_weight is not None:
        gates = torch.einsum(
            "bnhd,hgd->bnhg", Q.to(compute_dtype), gate_proj_weight.to(compute_dtype)
        )
        gates = gates.sigmoid()
    else:
        gates = torch.ones(B, N, H, 3, device=Q.device, dtype=compute_dtype) / 3.0

    O = gates[..., 0:1] * O_cmp + gates[..., 1:2] * O_slc + gates[..., 2:3] * O_sld
    return O.to(Q.dtype)


def _compressed_attention(
    q,
    k,
    v,
    block_size,
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
):
    k_orig = k[:, ::groups]
    v_orig = v[:, ::groups]
    N_cmp = N // block_size
    k_blocks = k_orig.reshape(B, H_kv, N_cmp, block_size, D)
    k_cmp = k_blocks.mean(dim=3)
    v_blocks = v_orig.reshape(B, H_kv, N_cmp, block_size, D)
    v_cmp = v_blocks.mean(dim=3)

    if W_k_compress is not None:
        k_cmp = torch.einsum("bhnd,hde->bhne", k_cmp, W_k_compress.to(k_cmp.dtype))
    if W_v_compress is not None:
        v_cmp = torch.einsum("bhnd,hde->bhne", v_cmp, W_v_compress.to(v_cmp.dtype))
    if groups > 1:
        k_cmp = k_cmp.repeat_interleave(groups, dim=1)
        v_cmp = v_cmp.repeat_interleave(groups, dim=1)

    scores = torch.matmul(q, k_cmp.transpose(-2, -1)) * softmax_scale
    if causal:
        q_pos = torch.arange(N, device=q.device).unsqueeze(1)
        kv_block_start = torch.arange(N_cmp, device=q.device) * block_size
        causal_mask = kv_block_start.unsqueeze(0) > q_pos
        scores = scores.masked_fill(
            causal_mask.unsqueeze(0).unsqueeze(0), float("-inf")
        )
    attn = torch.softmax(scores, dim=-1)
    return torch.matmul(attn, v_cmp)


def _selected_attention(
    q,
    k,
    v,
    block_size,
    num_selected,
    W_k_compress,
    groups,
    softmax_scale,
    causal,
    B,
    N,
    H,
    H_kv,
    D,
    q_tile_size=256,
    _block_indices=None,
):
    N_blocks = N // block_size
    N_q_tiles = N // q_tile_size
    k_actual = min(num_selected, N_blocks)

    if _block_indices is not None:
        top_indices = _block_indices
    else:
        k_orig = k[:, ::groups]
        k_blocks = k_orig.reshape(B, H_kv, N_blocks, block_size, D)
        k_cmp = k_blocks.mean(dim=3)
        if W_k_compress is not None:
            k_cmp = torch.einsum("bhnd,hde->bhne", k_cmp, W_k_compress.to(k_cmp.dtype))
        if groups > 1:
            k_cmp = k_cmp.repeat_interleave(groups, dim=1)
        block_scores = torch.matmul(q, k_cmp.transpose(-2, -1)) * softmax_scale
        block_scores_agg = block_scores.reshape(
            B, H, N_q_tiles, q_tile_size, N_blocks
        ).mean(dim=3)
        if causal:
            q_tile_end = (torch.arange(N_q_tiles, device=q.device) + 1) * q_tile_size
            kv_block_start = torch.arange(N_blocks, device=q.device) * block_size
            future_mask = kv_block_start.unsqueeze(0) >= q_tile_end.unsqueeze(1)
            block_scores_agg = block_scores_agg.masked_fill(
                future_mask.unsqueeze(0).unsqueeze(0), float("-inf")
            )
        _, top_indices = block_scores_agg.topk(k_actual, dim=-1)

    offsets = torch.arange(block_size, device=q.device)
    pos_indices = top_indices.unsqueeze(-1) * block_size + offsets
    kv_len = k_actual * block_size
    pos_indices = pos_indices.reshape(B, H, N_q_tiles, kv_len)

    gather_idx = pos_indices.unsqueeze(-1).expand(B, H, N_q_tiles, kv_len, D)
    k_exp = k.unsqueeze(2).expand(B, H, N_q_tiles, N, D)
    v_exp = v.unsqueeze(2).expand(B, H, N_q_tiles, N, D)
    sel_k = torch.gather(k_exp, 3, gather_idx)
    sel_v = torch.gather(v_exp, 3, gather_idx)

    q_tiles = q.reshape(B, H, N_q_tiles, q_tile_size, D)
    scores = torch.matmul(q_tiles, sel_k.transpose(-2, -1)) * softmax_scale
    if causal:
        q_tile_starts = torch.arange(N_q_tiles, device=q.device) * q_tile_size
        q_offsets = torch.arange(q_tile_size, device=q.device)
        q_positions = q_tile_starts.unsqueeze(1) + q_offsets.unsqueeze(0)
        c_mask = pos_indices.unsqueeze(3) > q_positions.reshape(
            1, 1, N_q_tiles, q_tile_size, 1
        )
        scores = scores.masked_fill(c_mask, float("-inf"))
    attn = torch.softmax(scores, dim=-1)
    out_tiles = torch.matmul(attn, sel_v)
    return out_tiles.reshape(B, H, N, D)


def _sliding_window_attention(q, k, v, window_size, softmax_scale, causal, B, N, H, D):
    scores = torch.matmul(q, k.transpose(-2, -1)) * softmax_scale
    q_pos = torch.arange(N, device=q.device).unsqueeze(1)
    kv_pos = torch.arange(N, device=q.device).unsqueeze(0)
    if causal:
        mask = (kv_pos > q_pos) | (kv_pos < q_pos - window_size + 1)
    else:
        half_w = window_size // 2
        mask = (kv_pos > q_pos + half_w) | (kv_pos < q_pos - half_w)
    scores = scores.masked_fill(mask.unsqueeze(0).unsqueeze(0), float("-inf"))
    attn = torch.softmax(scores, dim=-1)
    return torch.matmul(attn, v)


def compute_block_indices_reference(
    Q,
    K,
    compress_block_size=64,
    num_selected_blocks=16,
    W_k_compress=None,
    causal=True,
    softmax_scale=None,
    q_tile_size=256,
):
    """Pre-compute block selection indices (non-differentiable).

    Returns block indices that can be passed to nsa_forward_reference via
    _block_indices to freeze the selection during gradient computation.
    """
    B, N, H, D = Q.shape
    H_kv = K.shape[2]
    groups = H // H_kv

    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(D)

    # fp32 for numerical stability
    q = Q.transpose(1, 2).float()
    k = K.transpose(1, 2).float()

    N_blocks = N // compress_block_size
    N_q_tiles = N // q_tile_size

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
    Q,
    K,
    V,
    dO,
    compress_block_size=64,
    num_selected_blocks=16,
    window_size=512,
    W_k_compress=None,
    W_v_compress=None,
    gate_proj_weight=None,
    causal=True,
    softmax_scale=None,
    q_tile_size=256,
    _block_indices=None,
):
    """Compute NSA gradients using autograd on the differentiable reference forward."""
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
        O, grad_inputs, grad_outputs=dO.float(), allow_unused=True
    )

    result = {
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
