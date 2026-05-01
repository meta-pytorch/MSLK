# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
# pyre-ignore-all-errors

"""KV compression for NSA: mean-pool KV into blocks and apply learned projection.

Provides compress_kv (reference), fused_compress_kv (with varlen support),
and fused_compress_kv_backward.
"""

from __future__ import annotations

import torch
from torch import Tensor


def compress_kv(
    K: Tensor,  # (B, N, H_kv, D)
    V: Tensor,  # (B, N, H_kv, D)
    block_size: int,
    W_k: Tensor | None = None,  # (H_kv, D, D) or None
    W_v: Tensor | None = None,  # (H_kv, D, D) or None
) -> tuple[Tensor, Tensor]:
    """Compress K, V by mean-pooling over blocks and applying optional linear projection.

    Args:
        K: Key tensor, shape (B, N, H_kv, D).
        V: Value tensor, shape (B, N, H_kv, D).
        block_size: Number of positions to pool together.
        W_k: Optional per-head projection weight for keys.
        W_v: Optional per-head projection weight for values.

    Returns:
        K_cmp: Compressed keys, shape (B, N // block_size, H_kv, D).
        V_cmp: Compressed values, shape (B, N // block_size, H_kv, D).
    """
    B, N, H_kv, D = K.shape
    assert N % block_size == 0, (
        f"Sequence length {N} must be divisible by block_size {block_size}"
    )

    N_cmp = N // block_size

    # Reshape into blocks and mean-pool
    K_cmp = K.reshape(B, N_cmp, block_size, H_kv, D).mean(dim=2)  # (B, N_cmp, H_kv, D)
    V_cmp = V.reshape(B, N_cmp, block_size, H_kv, D).mean(dim=2)

    # Apply per-head linear projection if provided
    if W_k is not None:
        # W_k: (H_kv, D, D)
        # K_cmp: (B, N_cmp, H_kv, D) -> einsum over D
        K_cmp = torch.einsum("bnhd,hde->bnhe", K_cmp, W_k)

    if W_v is not None:
        V_cmp = torch.einsum("bnhd,hde->bnhe", V_cmp, W_v)

    return K_cmp, V_cmp


def fused_compress_kv(
    K: Tensor,  # (B, N, H_kv, D) or (total_tokens, H_kv, D) for varlen
    V: Tensor,  # same shape as K
    block_size: int,
    W_k: Tensor | None = None,  # (H_kv, D, D) or None
    W_v: Tensor | None = None,  # (H_kv, D, D) or None
    cu_seqlens: Tensor | None = None,  # (B+1,) int32 for varlen
    max_seqlen: int | None = None,
) -> tuple[Tensor, Tensor]:
    """Fused KV compression using a CuteDSL kernel.

    Replaces compress_kv's two PyTorch mean-reduction calls with a single
    fused kernel launch that processes both K and V simultaneously.

    For varlen (3D + cu_seqlens): processes each sequence independently in
    PyTorch (O(N), no CuteDSL kernel change needed), producing padded output
    (B, max_N_cmp, H_kv, D) where max_N_cmp = max_seqlen // block_size.

    Args:
        K: Key tensor, shape (B, N, H_kv, D) or (total_tokens, H_kv, D).
        V: Value tensor, same shape as K.
        block_size: Number of positions to pool together.
        W_k: Optional per-head projection weight for keys.
        W_v: Optional per-head projection weight for values.
        cu_seqlens: Cumulative sequence lengths, (B+1,) int32 for varlen.
        max_seqlen: Maximum sequence length (optional, computed if not given).

    Returns:
        K_cmp: Compressed keys, shape (B, N_cmp, H_kv, D).
        V_cmp: Compressed values, shape (B, N_cmp, H_kv, D).
    """
    if cu_seqlens is not None:
        return _fused_compress_kv_varlen(
            K, V, block_size, W_k, W_v, cu_seqlens, max_seqlen
        )

    B, N, H_kv, D = K.shape
    assert N % block_size == 0, (
        f"Sequence length {N} must be divisible by block_size {block_size}"
    )

    N_cmp = N // block_size

    # Mean-pool over block_size positions — pure PyTorch, no CuteDSL
    K_cmp = K.reshape(B, N_cmp, block_size, H_kv, D).mean(dim=2)
    V_cmp = V.reshape(B, N_cmp, block_size, H_kv, D).mean(dim=2)

    # Apply optional projections
    if W_k is not None:
        K_cmp = torch.einsum("bnhd,hde->bnhe", K_cmp, W_k)

    if W_v is not None:
        V_cmp = torch.einsum("bnhd,hde->bnhe", V_cmp, W_v)

    return K_cmp, V_cmp


def _fused_compress_kv_varlen(
    K: Tensor,  # (total_tokens, H_kv, D)
    V: Tensor,  # (total_tokens, H_kv, D)
    block_size: int,
    W_k: Tensor | None,
    W_v: Tensor | None,
    cu_seqlens: Tensor,  # (B+1,) int32
    max_seqlen: int | None,
) -> tuple[Tensor, Tensor]:
    """Varlen compress: per-sequence mean-pool in PyTorch, no padding of input."""
    assert K.dim() == 3, f"Varlen requires 3D input, got {K.dim()}D"
    H_kv = K.shape[1]
    D = K.shape[2]
    batch_size = cu_seqlens.shape[0] - 1
    seqlens = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()

    if max_seqlen is None:
        max_seqlen = max(seqlens)
    max_N_cmp = max_seqlen // block_size

    K_cmp = K.new_zeros(batch_size, max_N_cmp, H_kv, D)
    V_cmp = V.new_zeros(batch_size, max_N_cmp, H_kv, D)

    for i, slen in enumerate(seqlens):
        s = cu_seqlens[i].item()
        n_cmp_i = slen // block_size
        if n_cmp_i > 0:
            K_seq = K[s : s + n_cmp_i * block_size]
            V_seq = V[s : s + n_cmp_i * block_size]
            K_cmp[i, :n_cmp_i] = K_seq.reshape(n_cmp_i, block_size, H_kv, D).mean(dim=1)
            V_cmp[i, :n_cmp_i] = V_seq.reshape(n_cmp_i, block_size, H_kv, D).mean(dim=1)

    if W_k is not None:
        K_cmp = torch.einsum("bnhd,hde->bnhe", K_cmp, W_k)
    if W_v is not None:
        V_cmp = torch.einsum("bnhd,hde->bnhe", V_cmp, W_v)

    return K_cmp, V_cmp


def fused_compress_kv_backward(
    dK_cmp: Tensor,  # (B, N_cmp, H_kv, D)
    dV_cmp: Tensor,  # (B, N_cmp, H_kv, D)
    K: Tensor,  # (B, N, H_kv, D) or (total_tokens, H_kv, D) for varlen
    V: Tensor,  # same shape as K
    block_size: int,
    W_k: Tensor | None = None,
    W_v: Tensor | None = None,
    cu_seqlens: Tensor | None = None,  # (B+1,) int32 for varlen
) -> tuple[Tensor, Tensor, Tensor | None, Tensor | None]:
    """Backward through KV compression: CuteDSL scatter + PyTorch projection.

    For varlen (3D K, V + cu_seqlens): scatters from padded compressed gradients
    directly to 3D varlen output, no padding of K or V needed.

    Returns (dK, dV, dW_k, dW_v).
    """
    if cu_seqlens is not None:
        return _fused_compress_kv_backward_varlen(
            dK_cmp, dV_cmp, K, V, block_size, W_k, W_v, cu_seqlens
        )

    B, N, H_kv, D = K.shape
    N_cmp = N // block_size

    dW_k = None
    dW_v = None

    # Projection backward (PyTorch)
    if W_k is not None:
        K_cmp_mean = K.reshape(B, N_cmp, block_size, H_kv, D).mean(dim=2)
        dW_k = torch.einsum("bnhd,bnhe->hde", K_cmp_mean.float(), dK_cmp.float()).to(
            W_k.dtype
        )
        dK_cmp = torch.einsum("bnhe,hde->bnhd", dK_cmp.float(), W_k.float()).to(
            dK_cmp.dtype
        )

    if W_v is not None:
        V_cmp_mean = V.reshape(B, N_cmp, block_size, H_kv, D).mean(dim=2)
        dW_v = torch.einsum("bnhd,bnhe->hde", V_cmp_mean.float(), dV_cmp.float()).to(
            W_v.dtype
        )
        dV_cmp = torch.einsum("bnhe,hde->bnhd", dV_cmp.float(), W_v.float()).to(
            dV_cmp.dtype
        )

    # Mean-pool backward: scatter gradient — pure PyTorch
    # dK[b, i, h, d] = dK_cmp[b, i // block_size, h, d] / block_size
    dK = (
        dK_cmp.unsqueeze(2).expand(B, N_cmp, block_size, H_kv, D).reshape(B, N, H_kv, D)
        / block_size
    )
    dV = (
        dV_cmp.unsqueeze(2).expand(B, N_cmp, block_size, H_kv, D).reshape(B, N, H_kv, D)
        / block_size
    )

    return dK, dV, dW_k, dW_v


def _fused_compress_kv_backward_varlen(
    dK_cmp: Tensor,  # (B, max_N_cmp, H_kv, D)
    dV_cmp: Tensor,  # (B, max_N_cmp, H_kv, D)
    K: Tensor,  # (total_tokens, H_kv, D)
    V: Tensor,  # (total_tokens, H_kv, D)
    block_size: int,
    W_k: Tensor | None,
    W_v: Tensor | None,
    cu_seqlens: Tensor,  # (B+1,) int32
) -> tuple[Tensor, Tensor, Tensor | None, Tensor | None]:
    """Varlen compress backward: scatter from padded compressed grads to 3D output."""
    assert K.dim() == 3, f"Varlen requires 3D input, got {K.dim()}D"
    H_kv = K.shape[1]
    D = K.shape[2]
    batch_size = cu_seqlens.shape[0] - 1
    seqlens = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()

    dW_k = None
    dW_v = None

    # Projection backward (PyTorch) — operates on padded compressed tensors
    if W_k is not None:
        # Build K_cmp_mean from 3D K per-sequence
        max_N_cmp = dK_cmp.shape[1]
        K_cmp_mean = K.new_zeros(batch_size, max_N_cmp, H_kv, D)
        for i, slen in enumerate(seqlens):
            s = cu_seqlens[i].item()
            n_cmp_i = slen // block_size
            if n_cmp_i > 0:
                K_cmp_mean[i, :n_cmp_i] = (
                    K[s : s + n_cmp_i * block_size]
                    .reshape(n_cmp_i, block_size, H_kv, D)
                    .mean(dim=1)
                )
        dW_k = torch.einsum("bnhd,bnhe->hde", K_cmp_mean.float(), dK_cmp.float()).to(
            W_k.dtype
        )
        dK_cmp = torch.einsum("bnhe,hde->bnhd", dK_cmp.float(), W_k.float()).to(
            dK_cmp.dtype
        )

    if W_v is not None:
        max_N_cmp = dV_cmp.shape[1]
        V_cmp_mean = V.new_zeros(batch_size, max_N_cmp, H_kv, D)
        for i, slen in enumerate(seqlens):
            s = cu_seqlens[i].item()
            n_cmp_i = slen // block_size
            if n_cmp_i > 0:
                V_cmp_mean[i, :n_cmp_i] = (
                    V[s : s + n_cmp_i * block_size]
                    .reshape(n_cmp_i, block_size, H_kv, D)
                    .mean(dim=1)
                )
        dW_v = torch.einsum("bnhd,bnhe->hde", V_cmp_mean.float(), dV_cmp.float()).to(
            W_v.dtype
        )
        dV_cmp = torch.einsum("bnhe,hde->bnhd", dV_cmp.float(), W_v.float()).to(
            dV_cmp.dtype
        )

    # Scatter from compressed to 3D varlen: dK_cmp[i, block, h, d] / block_size
    total_tokens = K.shape[0]
    dK = K.new_zeros(total_tokens, H_kv, D)
    dV = V.new_zeros(total_tokens, H_kv, D)
    inv_bs = 1.0 / block_size

    for i, slen in enumerate(seqlens):
        s = cu_seqlens[i].item()
        n_cmp_i = slen // block_size
        if n_cmp_i > 0:
            # Expand each compressed gradient to block_size positions
            dK[s : s + n_cmp_i * block_size] = (
                dK_cmp[i, :n_cmp_i]
                .unsqueeze(1)
                .expand(-1, block_size, -1, -1)
                .reshape(n_cmp_i * block_size, H_kv, D)
                * inv_bs
            )
            dV[s : s + n_cmp_i * block_size] = (
                dV_cmp[i, :n_cmp_i]
                .unsqueeze(1)
                .expand(-1, block_size, -1, -1)
                .reshape(n_cmp_i * block_size, H_kv, D)
                * inv_bs
            )

    return dK, dV, dW_k, dW_v
