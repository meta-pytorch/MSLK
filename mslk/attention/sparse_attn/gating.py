# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
# pyre-ignore-all-errors

"""Output gating and combination for NSA's three attention branches.

Includes both PyTorch reference implementations (compute_gates, gate_and_combine)
and a fused CuteDSL kernel (fused_gate_and_combine) that computes gates from Q
and combines the three branch outputs in a single kernel launch.
"""

from __future__ import annotations

import math
from typing import Type

import torch
from torch import Tensor


# ---------------------------------------------------------------------------
# CuteDSL fused gating kernel
# ---------------------------------------------------------------------------

_fused_gating_compile_cache: dict = {}


def _make_fused_gating_kernel(cute_dtype: Type, D: int, has_gate_weight: bool):
    """Create and return a compiled CuteDSL kernel for fused gating.

    Defined as a factory function so CuteDSL imports happen lazily,
    keeping the module importable without GPU/CuteDSL availability.
    """
    import cutlass
    import cutlass.cute as cute

    WARPS_PER_BLOCK = 4
    THREADS_PER_BLOCK = WARPS_PER_BLOCK * 32
    elems_per_thread = D // 32

    class _Kernel:
        @cute.jit
        def __call__(self, mQ, mO_cmp, mO_slc, mO_sld, mW, mOut, H, stream):
            num_rows = mQ.shape[0]
            grid_x = cute.ceil_div(num_rows, WARPS_PER_BLOCK)
            self.kernel(mQ, mO_cmp, mO_slc, mO_sld, mW, mOut, H).launch(
                grid=[grid_x, 1, 1],
                block=[THREADS_PER_BLOCK, 1, 1],
                stream=stream,
            )

        @cute.kernel
        def kernel(self, mQ, mO_cmp, mO_slc, mO_sld, mW, mOut, H):
            lane_id = cute.arch.lane_idx()
            warp_id = cute.arch.warp_idx()
            bidx = cute.arch.block_idx()[0]

            row = cutlass.Int64(bidx * WARPS_PER_BLOCK + warp_id)
            num_rows = mQ.shape[0]

            if row < num_rows:
                ept = cutlass.const_expr(elems_per_thread)

                if cutlass.const_expr(has_gate_weight):
                    head_idx = row % H
                    g0 = cutlass.Float32(0.0)
                    g1 = cutlass.Float32(0.0)
                    g2 = cutlass.Float32(0.0)
                    for e in cutlass.range_constexpr(ept):
                        d = lane_id * ept + e
                        q_val = cutlass.Float32(mQ[row, d])
                        g0 += q_val * cutlass.Float32(mW[head_idx, d])
                        g1 += q_val * cutlass.Float32(mW[head_idx, D + d])
                        g2 += q_val * cutlass.Float32(mW[head_idx, 2 * D + d])

                    for i in cutlass.range_constexpr(5):
                        g0 += cute.arch.shuffle_sync_bfly(g0, offset=1 << i)
                        g1 += cute.arch.shuffle_sync_bfly(g1, offset=1 << i)
                        g2 += cute.arch.shuffle_sync_bfly(g2, offset=1 << i)

                    log2_e = cutlass.const_expr(math.log2(math.e))
                    g0 = 1.0 / (1.0 + cute.math.exp2(-g0 * log2_e))
                    g1 = 1.0 / (1.0 + cute.math.exp2(-g1 * log2_e))
                    g2 = 1.0 / (1.0 + cute.math.exp2(-g2 * log2_e))
                else:
                    g0 = cutlass.Float32(1.0 / 3.0)
                    g1 = cutlass.Float32(1.0 / 3.0)
                    g2 = cutlass.Float32(1.0 / 3.0)

                for e in cutlass.range_constexpr(ept):
                    d = lane_id * ept + e
                    o = (
                        g0 * cutlass.Float32(mO_cmp[row, d])
                        + g1 * cutlass.Float32(mO_slc[row, d])
                        + g2 * cutlass.Float32(mO_sld[row, d])
                    )
                    mOut[row, d] = cute_dtype(o)

    return _Kernel()


def fused_gate_and_combine(
    Q: Tensor,  # (B, N, H, D) or (T, H, D) for varlen
    O_cmp: Tensor,  # same shape as Q
    O_slc: Tensor,  # same shape as Q
    O_sld: Tensor,  # same shape as Q
    gate_proj_weight: Tensor | None = None,  # (H, 3, D)
) -> Tensor:
    """Gate computation and branch combination — pure PyTorch.

    Computes per-head sigmoid gates from Q and gate_proj_weight, then
    combines the three branch outputs with the gate weights.

    Args:
        Q: Query tensor, shape (B, N, H, D) or (T, H, D) for varlen.
        O_cmp: Compressed attention output, same shape as Q.
        O_slc: Selected attention output, same shape as Q.
        O_sld: Sliding window attention output, same shape as Q.
        gate_proj_weight: Gate projection weights, shape (H, 3, D).
            If None, uses uniform gates (1/3 each).

    Returns:
        Combined output, same shape as Q.
    """
    gates = compute_gates(Q, gate_proj_weight)
    return gate_and_combine(O_cmp, O_slc, O_sld, gates)


# ---------------------------------------------------------------------------
# CuteDSL fused gating backward kernel
# ---------------------------------------------------------------------------

_fused_gating_bwd_compile_cache: dict = {}


def _make_fused_gating_bwd_kernel(cute_dtype: Type, D: int):
    """Create CuteDSL kernel for fused gating backward.

    Computes per-row:
    - dO_cmp = g0 * dO, dO_slc = g1 * dO, dO_sld = g2 * dO
    - dot_i = sum_d(dO[d] * O_i[d]) for each branch i
    - d_logit_i = dot_i * gi * (1 - gi)
    - dQ_gate[d] = sum_i(d_logit_i * W_i[d])

    Does NOT compute dW_gate (requires cross-row reduction, done in PyTorch).
    """
    import cutlass
    import cutlass.cute as cute

    WARPS_PER_BLOCK = 4
    THREADS_PER_BLOCK = WARPS_PER_BLOCK * 32
    elems_per_thread = D // 32

    class _Kernel:
        @cute.jit
        def __call__(
            self,
            mQ,
            mdO,
            mO_cmp,
            mO_slc,
            mO_sld,
            mW,
            mGates,
            mdO_cmp,
            mdO_slc,
            mdO_sld,
            mdQ_gate,
            H,
            stream,
        ):
            num_rows = mQ.shape[0]
            grid_x = cute.ceil_div(num_rows, WARPS_PER_BLOCK)
            self.kernel(
                mQ,
                mdO,
                mO_cmp,
                mO_slc,
                mO_sld,
                mW,
                mGates,
                mdO_cmp,
                mdO_slc,
                mdO_sld,
                mdQ_gate,
                H,
            ).launch(
                grid=[grid_x, 1, 1],
                block=[THREADS_PER_BLOCK, 1, 1],
                stream=stream,
            )

        @cute.kernel
        def kernel(
            self,
            mQ,
            mdO,
            mO_cmp,
            mO_slc,
            mO_sld,
            mW,
            mGates,
            mdO_cmp,
            mdO_slc,
            mdO_sld,
            mdQ_gate,
            H,
        ):
            lane_id = cute.arch.lane_idx()
            warp_id = cute.arch.warp_idx()
            bidx = cute.arch.block_idx()[0]

            row = cutlass.Int64(bidx * WARPS_PER_BLOCK + warp_id)
            num_rows = mQ.shape[0]

            if row < num_rows:
                ept = cutlass.const_expr(elems_per_thread)
                head_idx = row % H

                # Read gates for this row
                g0 = cutlass.Float32(mGates[row, 0])
                g1 = cutlass.Float32(mGates[row, 1])
                g2 = cutlass.Float32(mGates[row, 2])

                # Phase 1: Compute dot products dO·O_i via warp reduction
                dot0 = cutlass.Float32(0.0)
                dot1 = cutlass.Float32(0.0)
                dot2 = cutlass.Float32(0.0)
                for e in cutlass.range_constexpr(ept):
                    d = lane_id * ept + e
                    dO_val = cutlass.Float32(mdO[row, d])
                    dot0 += dO_val * cutlass.Float32(mO_cmp[row, d])
                    dot1 += dO_val * cutlass.Float32(mO_slc[row, d])
                    dot2 += dO_val * cutlass.Float32(mO_sld[row, d])

                for i in cutlass.range_constexpr(5):
                    dot0 += cute.arch.shuffle_sync_bfly(dot0, offset=1 << i)
                    dot1 += cute.arch.shuffle_sync_bfly(dot1, offset=1 << i)
                    dot2 += cute.arch.shuffle_sync_bfly(dot2, offset=1 << i)

                # Phase 2: Sigmoid derivative and d_logit
                d_logit0 = dot0 * g0 * (1.0 - g0)
                d_logit1 = dot1 * g1 * (1.0 - g1)
                d_logit2 = dot2 * g2 * (1.0 - g2)

                # Phase 3: Write outputs per element
                for e in cutlass.range_constexpr(ept):
                    d = lane_id * ept + e
                    dO_val = cutlass.Float32(mdO[row, d])

                    # dO_i = gi * dO
                    mdO_cmp[row, d] = cute_dtype(g0 * dO_val)
                    mdO_slc[row, d] = cute_dtype(g1 * dO_val)
                    mdO_sld[row, d] = cute_dtype(g2 * dO_val)

                    # dQ_gate = sum_i(d_logit_i * W_i[d])
                    dq = (
                        d_logit0 * cutlass.Float32(mW[head_idx, d])
                        + d_logit1 * cutlass.Float32(mW[head_idx, D + d])
                        + d_logit2 * cutlass.Float32(mW[head_idx, 2 * D + d])
                    )
                    mdQ_gate[row, d] = cute_dtype(dq)

    return _Kernel()


def fused_gating_backward(
    Q: Tensor,  # (B, N, H, D) or (T, H, D) for varlen
    dO: Tensor,  # same shape as Q
    O_cmp: Tensor,  # same shape as Q
    O_slc: Tensor,  # same shape as Q
    O_sld: Tensor,  # same shape as Q
    gates: Tensor,  # (..., H, 3) matching leading dims of Q
    gate_proj_weight: Tensor,  # (H, 3, D)
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Gating backward — pure PyTorch.

    Computes dO_cmp, dO_slc, dO_sld (gradient routing through gates),
    dQ_gate (query gradient from gate computation), and dW_gate (weight gradient).

    Args:
        Q: Query tensor, shape (B, N, H, D) or (T, H, D) for varlen.
        dO: Upstream gradient, same shape as Q.
        O_cmp, O_slc, O_sld: Branch outputs, same shape as Q.
        gates: Sigmoid gate values, (..., H, 3) matching leading dims of Q.
        gate_proj_weight: Gate weights, shape (H, 3, D).

    Returns:
        (dO_cmp, dO_slc, dO_sld, dQ_gate, dW_gate)
    """
    H = Q.shape[-2]
    D = Q.shape[-1]

    # dO_i = g_i * dO (gradient routing)
    g = gates.float()
    dO_f = dO.float()
    dO_cmp = (g[..., 0:1] * dO_f).to(dO.dtype)
    dO_slc = (g[..., 1:2] * dO_f).to(dO.dtype)
    dO_sld = (g[..., 2:3] * dO_f).to(dO.dtype)

    # Sigmoid derivative: d_logit_i = (dO · O_i) * g_i * (1 - g_i)
    O_branches = torch.stack([O_cmp.float(), O_slc.float(), O_sld.float()], dim=-1)
    dg = (dO_f.unsqueeze(-1) * O_branches).sum(dim=-2)  # (..., H, 3)
    d_logit = dg * g * (1 - g)

    # dQ_gate = d_logit @ W (query gradient from gate computation)
    dQ_gate = (
        torch.einsum(
            "mhg,hgd->mhd",
            d_logit.reshape(-1, H, 3),
            gate_proj_weight.float(),
        )
        .reshape(Q.shape)
        .to(Q.dtype)
    )

    # dW_gate = d_logit.T @ Q (weight gradient)
    dW_gate = torch.einsum(
        "mhg,mhd->hgd",
        d_logit.reshape(-1, H, 3),
        Q.float().reshape(-1, H, D),
    ).to(gate_proj_weight.dtype)

    return dO_cmp, dO_slc, dO_sld, dQ_gate, dW_gate


# ---------------------------------------------------------------------------
# PyTorch reference implementations (kept for testing and fallback)
# ---------------------------------------------------------------------------


def compute_gates(
    Q: Tensor,  # (B, N, H, D) or (T, H, D) for varlen
    gate_proj_weight: Tensor | None = None,  # (H, 3, D)
    chunk_size: int | None = None,
) -> Tensor:
    """Compute per-head sigmoid gates for the three NSA branches.

    Args:
        Q: Query tensor, shape (B, N, H, D) or (T, H, D) for varlen.
        gate_proj_weight: Learned gate projection, shape (H, 3, D).
            If None, returns uniform gates (1/3 each).
        chunk_size: If set, process the sequence dimension in chunks to
            reduce peak float32 memory usage. Recommended for N > 32K.
            Only supported for 4D input.

    Returns:
        gates: Sigmoid gates, shape (..., H, 3) matching leading dims of Q.
    """
    if gate_proj_weight is None:
        return torch.ones(*Q.shape[:-1], 3, device=Q.device, dtype=Q.dtype) / 3.0

    if chunk_size is None:
        # Reshape to (M, H, D) for einsum — works for both 3D and 4D
        orig_shape = Q.shape
        H, D = orig_shape[-2], orig_shape[-1]
        gate_logits = torch.einsum(
            "mhd,hgd->mhg",
            Q.float().reshape(-1, H, D),
            gate_proj_weight.float(),
        )
        return gate_logits.sigmoid().to(Q.dtype).reshape(*orig_shape[:-1], 3)

    # Chunked path: process sequence in chunks to bound float32 intermediates
    assert Q.dim() == 4, "Chunked gating only supported for 4D input"
    B, N, H, D = Q.shape
    gates = torch.empty(B, N, H, 3, device=Q.device, dtype=Q.dtype)
    w = gate_proj_weight.float()
    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        gate_logits = torch.einsum("bnhd,hgd->bnhg", Q[:, start:end].float(), w)
        gates[:, start:end] = gate_logits.sigmoid().to(Q.dtype)
    return gates


def gate_and_combine(
    O_cmp: Tensor,  # (B, N, H, D) or (T, H, D) for varlen
    O_slc: Tensor,  # same shape as O_cmp
    O_sld: Tensor,  # same shape as O_cmp
    gates: Tensor,  # (..., H, 3) matching leading dims
    chunk_size: int | None = None,
) -> Tensor:
    """Combine three attention branch outputs with sigmoid gates.

    Args:
        O_cmp: Compressed attention output, shape (B, N, H, D) or (T, H, D).
        O_slc: Selected attention output, same shape as O_cmp.
        O_sld: Sliding window attention output, same shape as O_cmp.
        gates: Sigmoid gates, (..., H, 3) matching leading dims.
        chunk_size: If set, process the sequence dimension in chunks to
            reduce peak float32 memory usage. Only supported for 4D input.

    Returns:
        Combined output, same shape as O_cmp.
    """
    if chunk_size is None:
        g = gates.float()
        return (
            g[..., 0:1] * O_cmp.float()
            + g[..., 1:2] * O_slc.float()
            + g[..., 2:3] * O_sld.float()
        ).to(O_cmp.dtype)

    # Chunked path: avoid materializing 3 full (B,N,H,D) float32 tensors at once
    assert O_cmp.dim() == 4, "Chunked gating only supported for 4D input"
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
