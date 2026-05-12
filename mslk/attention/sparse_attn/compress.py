# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

"""KV compression for NSA: mean-pool KV into blocks and apply learned projection.

Provides compress_kv (reference implementation).
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
    """Compress K, V by mean-pooling over blocks.

    Applies optional linear projection after pooling.

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

    K_cmp = K.reshape(B, N_cmp, block_size, H_kv, D).mean(dim=2)
    V_cmp = V.reshape(B, N_cmp, block_size, H_kv, D).mean(dim=2)

    if W_k is not None:
        K_cmp = torch.einsum("bnhd,hde->bnhe", K_cmp, W_k)
    if W_v is not None:
        V_cmp = torch.einsum("bnhd,hde->bnhe", V_cmp, W_v)

    return K_cmp, V_cmp
