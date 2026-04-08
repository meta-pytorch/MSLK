# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
# pyre-ignore-all-errors

"""KV compression for NSA: mean-pool KV into blocks and apply learned projection.

Includes both a PyTorch reference (compress_kv) and a fused CuteDSL kernel
(fused_compress_kv) that processes both K and V in a single kernel launch.
"""

from __future__ import annotations

import math
from typing import Type

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


# ---------------------------------------------------------------------------
# CuteDSL fused compression kernel
# ---------------------------------------------------------------------------

_fused_compress_compile_cache: dict = {}


def _make_fused_compress_kernel(cute_dtype: Type, D: int, block_size: int):
    """Create and return a compiled CuteDSL kernel for fused KV compression.

    Each thread block handles one output element (b, j, h) — one compressed
    block for one KV head. It reads block_size input positions for both K and V,
    accumulates in float32, divides by block_size, and writes the mean.

    Grid: B * N_cmp * H_kv (one block per output row)
    Block: 128 threads
    Each thread handles elements d = tidx, tidx+128, ... of the D dimension.
    """
    import cutlass
    import cutlass.cute as cute
    from cutlass import const_expr, Float32

    THREADS_PER_BLOCK = 128
    D_PER_THREAD = math.ceil(D / THREADS_PER_BLOCK)

    class _Kernel:
        @cute.jit
        def __call__(self, mK, mV, mK_out, mV_out, N_cmp, H_kv, N, stream):
            grid_x = mK_out.shape[0]  # B * N_cmp * H_kv
            self.kernel(mK, mV, mK_out, mV_out, N_cmp, H_kv, N).launch(
                grid=[grid_x, 1, 1],
                block=[THREADS_PER_BLOCK, 1, 1],
                stream=stream,
            )

        @cute.kernel
        def kernel(self, mK, mV, mK_out, mV_out, N_cmp, H_kv, N):
            tidx = cute.arch.thread_idx()[0]
            bidx = cute.arch.block_idx()[0]

            # Decompose block index into (b, j, h)
            b = bidx // (N_cmp * H_kv)
            j = (bidx % (N_cmp * H_kv)) // H_kv
            h = bidx % H_kv

            inv_block_size = const_expr(1.0 / block_size)

            for e in cutlass.range_constexpr(D_PER_THREAD):
                d = tidx + const_expr(e * THREADS_PER_BLOCK)
                if d < const_expr(D):
                    acc_k = Float32(0.0)
                    acc_v = Float32(0.0)

                    for t in cutlass.range_constexpr(block_size):
                        input_row = cutlass.Int64(
                            b * N * H_kv
                            + (j * const_expr(block_size) + const_expr(t)) * H_kv
                            + h
                        )
                        acc_k += Float32(mK[input_row, d])
                        acc_v += Float32(mV[input_row, d])

                    output_row = cutlass.Int64(b * N_cmp * H_kv + j * H_kv + h)
                    mK_out[output_row, d] = cute_dtype(acc_k * inv_block_size)
                    mV_out[output_row, d] = cute_dtype(acc_v * inv_block_size)

    return _Kernel()


def fused_compress_kv(
    K: Tensor,  # (B, N, H_kv, D)
    V: Tensor,  # (B, N, H_kv, D)
    block_size: int,
    W_k: Tensor | None = None,  # (H_kv, D, D) or None
    W_v: Tensor | None = None,  # (H_kv, D, D) or None
) -> tuple[Tensor, Tensor]:
    """Fused KV compression using a CuteDSL kernel.

    Replaces compress_kv's two PyTorch mean-reduction calls with a single
    fused kernel launch that processes both K and V simultaneously.

    The optional W_k/W_v projections remain in PyTorch since they're rarely
    used and would require a separate, more complex kernel.

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
    import cuda.bindings.driver as cuda
    import cutlass
    import cutlass.cute as cute
    from cutlass.cute.runtime import from_dlpack

    B, N, H_kv, D = K.shape
    assert N % block_size == 0, (
        f"Sequence length {N} must be divisible by block_size {block_size}"
    )

    N_cmp = N // block_size

    # Allocate outputs
    K_cmp = torch.empty(B, N_cmp, H_kv, D, device=K.device, dtype=K.dtype)
    V_cmp = torch.empty(B, N_cmp, H_kv, D, device=V.device, dtype=V.dtype)

    # Reshape to 2D: (B*N*H_kv, D) and (B*N_cmp*H_kv, D)
    K_2d = K.reshape(B * N * H_kv, D).contiguous()
    V_2d = V.reshape(B * N * H_kv, D).contiguous()
    K_out_2d = K_cmp.reshape(B * N_cmp * H_kv, D)
    V_out_2d = V_cmp.reshape(B * N_cmp * H_kv, D)

    torch2cute_dtype = {
        torch.float16: cutlass.Float16,
        torch.bfloat16: cutlass.BFloat16,
        torch.float32: cutlass.Float32,
    }
    cute_dtype = torch2cute_dtype[K.dtype]

    def to_cute(tensor):
        return from_dlpack(
            tensor.detach(), assumed_align=16
        ).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1))

    mK = to_cute(K_2d)
    mV = to_cute(V_2d)
    mK_out = to_cute(K_out_2d)
    mV_out = to_cute(V_out_2d)

    current_stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

    compile_key = (cute_dtype, D, block_size)
    if compile_key not in _fused_compress_compile_cache:
        kernel_op = _make_fused_compress_kernel(cute_dtype, D, block_size)
        _fused_compress_compile_cache[compile_key] = cute.compile(
            kernel_op, mK, mV, mK_out, mV_out, N_cmp, H_kv, N, current_stream
        )
    _fused_compress_compile_cache[compile_key](
        mK, mV, mK_out, mV_out, N_cmp, H_kv, N, current_stream
    )

    # Apply optional projections in PyTorch (rare path)
    if W_k is not None:
        K_cmp = torch.einsum("bnhd,hde->bnhe", K_cmp, W_k)

    if W_v is not None:
        V_cmp = torch.einsum("bnhd,hde->bnhe", V_cmp, W_v)

    return K_cmp, V_cmp
