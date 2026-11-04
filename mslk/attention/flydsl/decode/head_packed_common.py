# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""Pieces shared by the head-packed gfx950 decode kernels (pa_decode_gfx950,
pa_decode_swa): a GQA group's (query token, head) pairs ride on the 16 rows of
a 16x16x32 MFMA, m = qtok * ratio + head."""

from typing import Any

import flydsl.expr as fx  # pyre-ignore[21]
from flydsl.expr import arith, range_constexpr, rocdl  # pyre-ignore[21]
from flydsl.expr.typing import T  # pyre-ignore[21]
from mslk.flydsl import buffer_ops, vector_ops as vector

MFMA_M = 16  # (query token, head) pairs packed on the QK MFMA M-axis
MFMA_N = 16  # tokens per QK sub-tile
MFMA_K_QK = 32  # head-dim elements per QK MFMA
TILE_N = 32  # tokens per streaming tile (= PV MFMA K-dim)
N_SUBTILE = TILE_N // MFMA_N  # QK sub-tiles per tile
LOG2E: float = 1.4426950408889634
FX_DTYPE = {"f32": fx.Float32, "f16": fx.Float16, "bf16": fx.BFloat16}


def mfma_16x16x32(kv_dtype_str: str) -> Any:  # pyre-ignore[3]
    return (
        rocdl.mfma_f32_16x16x32_bf16
        if kv_dtype_str == "bf16"
        else rocdl.mfma_f32_16x16x32_f16
    )


def load_q_frags(  # pyre-ignore[3]
    q_rsrc: Any,  # pyre-ignore[2]
    q_base_bh: Any,  # pyre-ignore[2]
    stride_qs: Any,  # pyre-ignore[2]
    tok_lane: Any,  # pyre-ignore[2]
    grp: Any,  # pyre-ignore[2]
    ratio: int,
    seqlen_q: int,
    head_dim: int,
    fx_kv: Any,  # pyre-ignore[2]
) -> list:
    """Loop-invariant QK A-fragments, q_frags[m_tile][k_group]: lane row
    tok_lane is M slot m = qtok * ratio + head of each tile, and q_base_bh is
    the offset of (request, first head of the group) plus that head.

    Rows whose query token is past seqlen_q load the last token instead (a
    harmless duplicate the epilogue discards).
    """
    t_pack = MFMA_M // ratio
    m_tiles = (seqlen_q + t_pack - 1) // t_pack
    q_tok_local = tok_lane // fx.Int32(ratio)
    q_frags = []
    for t in range_constexpr(m_tiles):
        q_tok = q_tok_local + fx.Int32(t * t_pack)
        q_tok = fx.Int32(
            arith.select(q_tok < fx.Int32(seqlen_q), q_tok, fx.Int32(seqlen_q - 1))
        )
        q_base = q_base_bh + q_tok * stride_qs
        frags = []
        for g in range_constexpr(head_dim // MFMA_K_QK):
            q_off = q_base + fx.Int32(g * MFMA_K_QK) + grp * fx.Int32(8)
            frags.append(
                buffer_ops.buffer_load(q_rsrc, q_off, vec_width=8, dtype=fx_kv)
            )
        q_frags.append(frags)
    return q_frags


def read_v_frag_tr(off_lo: Any, off_hi: Any, fx_kv: Any) -> Any:  # pyre-ignore[2,3]
    """PV B-fragment: 8 V values read with two ds_read_tr16_b64 (hardware
    transpose) at f16-element LDS offsets off_lo and off_hi."""
    v4h = T.vec(4, fx_kv.ir_type)
    halves = [
        rocdl.ds_read_tr16_b64(
            v4h,
            buffer_ops.create_llvm_ptr(fx.Int64(off * fx.Index(2)), address_space=3),
        ).result
        for off in (off_lo, off_hi)
    ]
    return vector.shuffle(halves[0], halves[1], [0, 1, 2, 3, 4, 5, 6, 7])


def finalize_smem(alloc: Any) -> None:  # pyre-ignore[2]
    """Emit the kernel's LDS global; call at the top of each launcher trace."""
    from flydsl._mlir import ir as _ir
    from flydsl.compiler.kernel_function import CompilationContext

    alloc.finalized = False
    ctx = CompilationContext.get_current()
    with _ir.InsertionPoint(ctx.gpu_module_body):
        alloc.finalize()
