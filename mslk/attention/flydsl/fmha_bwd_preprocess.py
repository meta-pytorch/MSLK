# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""FMHA backward preprocess: D-vector kernel.

Computes D[row] = rowsum(dO[row,:] * O[row,:]) for each of n_rows rows.
Uses a multi-row-per-block strategy: each block processes ROWS_PER_BLOCK
rows in parallel, with threads_per_row threads per row doing a vec-load,
element-wise multiply, in-register reduce, and warp-shuffle reduce.
For D<=512, threads_per_row fits within a single warp so no LDS is needed.

Layout convention (matches ref_fmha_bwd_reference.py):
  dO, O  : [B*M*H, D]   int16 view of bf16/fp16 (contiguous)
  D_out  : [B*M*H, 1]   float32 — one scalar per row

Target: gfx950 (CDNA4, wave64).
"""

import math as _math

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import arith, range_constexpr
from flydsl.expr.vector import ReductionOp
from mslk.attention.flydsl.fmha_bwd_mfma import dtype_to_elem_type, WARP_SIZE

BLOCK_THREADS = 256  # threads per block


def compile_fmha_bwd_preprocess(*, D: int, dtype_str: str = "bf16"):
    """Compile the D-vector preprocess kernel.

    Args:
        D        : head dimension (must be multiple of 64)
        dtype_str: "bf16" or "f16"

    Returns:
        launch_fn(dO_2d, O_2d, D_out, n_rows, stream)
          dO_2d, O_2d : [n_rows, D]  int16 view of bf16/fp16  (2D, contiguous)
          D_out       : [n_rows, 1]  float32 (2D, one scalar per row)
    """
    assert D % WARP_SIZE == 0, f"D={D} must be a multiple of WARP_SIZE={WARP_SIZE}"

    elem_dtype = dtype_to_elem_type(dtype_str)
    VEC_WIDTH = 8  # 128-bit / 16-bit = 8 elements
    THREADS_PER_ROW = D // VEC_WIDTH  # 16 for D=128
    ROWS_PER_BLOCK = BLOCK_THREADS // THREADS_PER_ROW  # 16 for D=128

    assert THREADS_PER_ROW <= WARP_SIZE, (
        f"D={D} requires {THREADS_PER_ROW} threads/row > WARP_SIZE={WARP_SIZE}"
    )
    assert ROWS_PER_BLOCK >= 1, f"D={D} too large for BLOCK_THREADS={BLOCK_THREADS}"

    N_SHUFFLE_STEPS = int(_math.log2(THREADS_PER_ROW))

    @flyc.kernel(known_block_size=[BLOCK_THREADS, 1, 1])
    def d_vec_kernel(
        dO: fx.Tensor,  # [n_rows, D]  int16
        O: fx.Tensor,  # [n_rows, D]  int16
        D_out: fx.Tensor,  # [n_rows, 1]  float32
        n_rows: fx.Int32,
    ):
        from flydsl.expr import buffer_ops as _bops
        from flydsl.expr.typing import Vector as Vec

        bid = fx.block_idx.x
        tid = fx.thread_idx.x
        fm = arith.FastMathFlags.fast

        row_in_block = fx.Index(tid) // THREADS_PER_ROW
        col_thread = fx.Index(tid) % THREADS_PER_ROW
        row = fx.Index(bid) * ROWS_PER_BLOCK + row_in_block

        if row < fx.Index(n_rows):
            dO_rsrc = _bops.create_buffer_resource(dO)
            O_rsrc = _bops.create_buffer_resource(O)
            D_rsrc = _bops.create_buffer_resource(D_out)

            flat_elem = row * D + col_thread * VEC_WIDTH
            do_vec = _bops.buffer_load(
                dO_rsrc, flat_elem, vec_width=VEC_WIDTH, dtype=elem_dtype
            )
            o_vec = _bops.buffer_load(
                O_rsrc, flat_elem, vec_width=VEC_WIDTH, dtype=elem_dtype
            )

            do_f32 = Vec(do_vec).to(fx.Float32)
            o_f32 = Vec(o_vec).to(fx.Float32)
            prod = do_f32 * o_f32
            acc = prod.reduce(ReductionOp.ADD, fastmath=fm)

            # Warp shuffle reduce across THREADS_PER_ROW lanes
            w = acc
            for _sh in range_constexpr(N_SHUFFLE_STEPS):
                off = THREADS_PER_ROW // (2 << _sh)
                peer = w.shuffle_xor(off, WARP_SIZE)
                w = w.addf(peer, fastmath=fm)

            # Lane 0 of each row writes the result
            if col_thread == fx.Index(0):
                _bops.buffer_store(w, D_rsrc, row)

    @flyc.jit
    def launch_fn(
        dO: fx.Tensor,
        O: fx.Tensor,
        D_out: fx.Tensor,
        n_rows: fx.Int32,
        stream: fx.Stream,
    ):
        n_blocks = (fx.Index(n_rows) + ROWS_PER_BLOCK - 1) // ROWS_PER_BLOCK
        d_vec_kernel(dO, O, D_out, n_rows).launch(
            grid=(fx.Int32(n_blocks), 1, 1),
            block=(BLOCK_THREADS, 1, 1),
            stream=stream,
        )

    return launch_fn
