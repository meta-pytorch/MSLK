# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

"""Output gating and combination for NSA's three attention branches."""

from __future__ import annotations

import torch
from torch import Tensor


def compute_gates(
    Q: Tensor,  # (B, N, H, D)
    gate_proj_weight: Tensor | None = None,  # (H, 3, D)
    chunk_size: int | None = None,
) -> Tensor:
    """Compute per-head sigmoid gates for the three NSA branches.

    Args:
        Q: Query tensor, shape (B, N, H, D).
        gate_proj_weight: Learned gate projection, shape (H, 3, D).
            If None, returns uniform gates (1/3 each).
        chunk_size: If set, process the sequence dimension in chunks to
            reduce peak float32 memory usage. Recommended for N > 32K.

    Returns:
        gates: Sigmoid gates, shape (B, N, H, 3).
    """
    if gate_proj_weight is None:
        B, N, H, D = Q.shape
        return torch.ones(B, N, H, 3, device=Q.device, dtype=Q.dtype) / 3.0

    if chunk_size is None:
        # Original path for short sequences
        gate_logits = torch.einsum("bnhd,hgd->bnhg", Q.float(), gate_proj_weight.float())
        return gate_logits.sigmoid().to(Q.dtype)

    # Chunked path: process sequence in chunks to bound float32 intermediates
    B, N, H, D = Q.shape
    gates = torch.empty(B, N, H, 3, device=Q.device, dtype=Q.dtype)
    w = gate_proj_weight.float()
    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        gate_logits = torch.einsum("bnhd,hgd->bnhg", Q[:, start:end].float(), w)
        gates[:, start:end] = gate_logits.sigmoid().to(Q.dtype)
    return gates


def gate_and_combine(
    O_cmp: Tensor,  # (B, N, H, D)
    O_slc: Tensor,  # (B, N, H, D)
    O_sld: Tensor,  # (B, N, H, D)
    gates: Tensor,  # (B, N, H, 3)
    chunk_size: int | None = None,
) -> Tensor:
    """Combine three attention branch outputs with sigmoid gates.

    Args:
        O_cmp: Compressed attention output, shape (B, N, H, D).
        O_slc: Selected attention output, shape (B, N, H, D).
        O_sld: Sliding window attention output, shape (B, N, H, D).
        gates: Sigmoid gates, shape (B, N, H, 3).
        chunk_size: If set, process the sequence dimension in chunks to
            reduce peak float32 memory usage. Recommended for N > 32K.

    Returns:
        Combined output, shape (B, N, H, D).
    """
    if chunk_size is None:
        # Original path for short sequences
        g = gates.float()
        return (
            g[..., 0:1] * O_cmp.float()
            + g[..., 1:2] * O_slc.float()
            + g[..., 2:3] * O_sld.float()
        ).to(O_cmp.dtype)

    # Chunked path: avoid materializing 3 full (B,N,H,D) float32 tensors at once
    B, N, H, D = O_cmp.shape
    out = torch.empty(B, N, H, D, dtype=O_cmp.dtype, device=O_cmp.device)
    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        g = gates[:, start:end].float()
        out[:, start:end] = (
            g[..., 0:1] * O_cmp[:, start:end].float()
            + g[..., 1:2] * O_slc[:, start:end].float()
            + g[..., 2:3] * O_sld[:, start:end].float()
        ).to(O_cmp.dtype)
    return out
