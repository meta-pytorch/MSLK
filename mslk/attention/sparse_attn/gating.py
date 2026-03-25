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
    Q: Tensor,  # (B, N, H, D)
    O_cmp: Tensor,  # (B, N, H, D)
    O_slc: Tensor,  # (B, N, H, D)
    O_sld: Tensor,  # (B, N, H, D)
    gate_proj_weight: Tensor | None = None,  # (H, 3, D)
) -> Tensor:
    """Fused gate computation and branch combination using a CuteDSL kernel.

    Replaces the two-step compute_gates() + gate_and_combine() with a single
    kernel launch. Eliminates the intermediate gates tensor and reduces memory
    traffic by fusing the dot-product gate computation with the weighted sum.

    Args:
        Q: Query tensor, shape (B, N, H, D).
        O_cmp: Compressed attention output, shape (B, N, H, D).
        O_slc: Selected attention output, shape (B, N, H, D).
        O_sld: Sliding window attention output, shape (B, N, H, D).
        gate_proj_weight: Gate projection weights, shape (H, 3, D).
            If None, uses uniform gates (1/3 each).

    Returns:
        Combined output, shape (B, N, H, D).
    """
    import cuda.bindings.driver as cuda
    import cutlass
    import cutlass.cute as cute
    from cutlass.cute.runtime import from_dlpack

    B, N, H, D = Q.shape
    assert D % 32 == 0, f"D={D} must be divisible by 32"

    out = torch.empty_like(O_cmp)
    has_gate_weight = gate_proj_weight is not None

    # Reshape to 2D: (B*N*H, D)
    M = B * N * H
    Q_2d = Q.reshape(M, D)
    O_cmp_2d = O_cmp.reshape(M, D)
    O_slc_2d = O_slc.reshape(M, D)
    O_sld_2d = O_sld.reshape(M, D)
    out_2d = out.reshape(M, D)

    # Gate weights: (H, 3, D) -> (H, 3*D)
    if has_gate_weight:
        W_2d = gate_proj_weight.reshape(H, 3 * D).contiguous()
    else:
        W_2d = torch.empty(1, 1, device=Q.device, dtype=Q.dtype)

    torch2cute_dtype = {
        torch.float16: cutlass.Float16,
        torch.bfloat16: cutlass.BFloat16,
        torch.float32: cutlass.Float32,
    }
    cute_dtype = torch2cute_dtype[Q.dtype]

    def to_cute(tensor):
        return from_dlpack(
            tensor.detach(), assumed_align=16
        ).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1))

    mQ = to_cute(Q_2d)
    mO_cmp = to_cute(O_cmp_2d)
    mO_slc = to_cute(O_slc_2d)
    mO_sld = to_cute(O_sld_2d)
    mW = to_cute(W_2d)
    mOut = to_cute(out_2d)

    current_stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

    compile_key = (cute_dtype, D, has_gate_weight)
    if compile_key not in _fused_gating_compile_cache:
        kernel_op = _make_fused_gating_kernel(cute_dtype, D, has_gate_weight)
        _fused_gating_compile_cache[compile_key] = cute.compile(
            kernel_op, mQ, mO_cmp, mO_slc, mO_sld, mW, mOut, H, current_stream
        )
    _fused_gating_compile_cache[compile_key](
        mQ, mO_cmp, mO_slc, mO_sld, mW, mOut, H, current_stream
    )

    return out


# ---------------------------------------------------------------------------
# PyTorch reference implementations (kept for testing and fallback)
# ---------------------------------------------------------------------------


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
        gate_logits = torch.einsum(
            "bnhd,hgd->bnhg", Q.float(), gate_proj_weight.float()
        )
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
