# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

"""Output gating and combination for NSA's three attention branches.

Provides compute_gates and gate_and_combine (reference implementations).
"""

from __future__ import annotations

import torch
from torch import Tensor


def compute_gates(
    Q: Tensor,  # (B, N, H, D)
    gate_proj_weight: Tensor | None = None,  # (H, 3, D)
    chunk_size: int | None = None,
) -> Tensor:
    """Compute per-head gating weights from queries.

    If gate_proj_weight is provided, computes sigmoid(Q @ W) per head per branch.
    Otherwise, returns uniform 1/3 gates.

    Args:
        Q: Query tensor, shape (B, N, H, D).
        gate_proj_weight: Gate projection weights, shape (H, 3, D).
        chunk_size: If set, process in chunks to limit peak memory.

    Returns:
        gates: Gating weights, shape (B, N, H, 3).
    """
    if gate_proj_weight is None:
        return torch.ones(*Q.shape[:-1], 3, device=Q.device, dtype=Q.dtype) / 3.0

    if chunk_size is not None:
        return _compute_gates_chunked(Q, gate_proj_weight, chunk_size)

    # fp32 accumulation for numerical stability
    gates = torch.einsum("bnhd,hgd->bnhg", Q.float(), gate_proj_weight.float())
    return gates.sigmoid().to(Q.dtype)


def _compute_gates_chunked(
    Q: Tensor, gate_proj_weight: Tensor, chunk_size: int
) -> Tensor:
    """Compute gates in chunks to limit peak memory."""
    B, N, H, D = Q.shape
    gates = torch.empty(B, N, H, 3, device=Q.device, dtype=Q.dtype)
    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        # fp32 accumulation for numerical stability
        g = torch.einsum(
            "bnhd,hgd->bnhg",
            Q[:, start:end].float(),
            gate_proj_weight.float(),
        )
        gates[:, start:end] = g.sigmoid().to(Q.dtype)
    return gates


def gate_and_combine(
    O_cmp: Tensor,  # (B, N, H, D)
    O_slc: Tensor,  # (B, N, H, D)
    O_sld: Tensor,  # (B, N, H, D)
    gates: Tensor,  # (B, N, H, 3)
    chunk_size: int | None = None,
) -> Tensor:
    """Combine branch outputs using gating weights.

    combined = g0 * O_cmp + g1 * O_slc + g2 * O_sld

    Args:
        O_cmp: Compressed branch output, shape (B, N, H, D).
        O_slc: Selected branch output, shape (B, N, H, D).
        O_sld: Sliding window branch output, shape (B, N, H, D).
        gates: Gating weights, shape (B, N, H, 3).
        chunk_size: If set, process in chunks to limit peak memory.

    Returns:
        Combined output, shape (B, N, H, D).
    """
    if chunk_size is not None:
        return _gate_and_combine_chunked(O_cmp, O_slc, O_sld, gates, chunk_size)

    # fp32 accumulation for numerical stability
    g = gates.float()
    combined = (
        g[..., 0:1] * O_cmp.float()
        + g[..., 1:2] * O_slc.float()
        + g[..., 2:3] * O_sld.float()
    )
    return combined.to(O_cmp.dtype)


def _gate_and_combine_chunked(
    O_cmp: Tensor,
    O_slc: Tensor,
    O_sld: Tensor,
    gates: Tensor,
    chunk_size: int,
) -> Tensor:
    """Chunked gating to limit peak memory (avoids 3 full fp32 intermediates)."""
    N = O_cmp.shape[1]
    out = torch.empty_like(O_cmp)
    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        g = gates[:, start:end].float()
        out[:, start:end] = (
            g[..., 0:1] * O_cmp[:, start:end].float()
            + g[..., 1:2] * O_slc[:, start:end].float()
            + g[..., 2:3] * O_sld[:, start:end].float()
        ).to(O_cmp.dtype)
    return out
