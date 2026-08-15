# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""FMHA backward dQ convert: f32 accumulator -> output dtype (bf16/fp16).

A simple elementwise cast over the flat [B*M*H*D] dQ accumulator buffer.

Target: gfx950 (CDNA4, wave64).
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from mslk.attention.flydsl.fmha_bwd_mfma import dtype_to_elem_type

BLOCK_THREADS = 256
VEC_WIDTH = 4


def compile_fmha_bwd_convert_dq(*, dtype_str: str = "bf16"):
    """Compile the dQ f32->output-dtype convert kernel.

    Returns:
        launch_fn(dq_f32, dq_out, n_elems, stream)
          dq_f32  : [n_elems, 1] float32 input
          dq_out  : [n_elems, 1] output dtype (bf16/fp16)
          n_elems : total element count (B*M*H*D)
    """
    elem_dtype = dtype_to_elem_type(dtype_str)

    @flyc.kernel(known_block_size=[BLOCK_THREADS, 1, 1])
    def convert_dq_kernel(
        dq_f32: fx.Tensor,
        dq_out: fx.Tensor,
        n_elems: fx.Int32,
    ):
        from flydsl.expr import buffer_ops as _bops
        from flydsl.expr.typing import Vector as Vec

        bid = fx.block_idx.x
        tid = fx.thread_idx.x
        n_elems_idx = fx.Index(n_elems)

        src_rsrc = _bops.create_buffer_resource(dq_f32)
        dst_rsrc = _bops.create_buffer_resource(dq_out)

        base = (fx.Index(bid) * BLOCK_THREADS + fx.Index(tid)) * VEC_WIDTH
        if base < n_elems_idx:
            v = _bops.buffer_load(src_rsrc, base, vec_width=VEC_WIDTH, dtype=fx.Float32)
            v_out = Vec(v).to(elem_dtype)
            _bops.buffer_store(v_out.ir_value(), dst_rsrc, base)

    @flyc.jit
    def launch_fn(
        dq_f32: fx.Tensor,
        dq_out: fx.Tensor,
        n_elems: fx.Int32,
        stream: fx.Stream,
    ):
        n_blocks = (fx.Index(n_elems) + (BLOCK_THREADS * VEC_WIDTH) - 1) // (
            BLOCK_THREADS * VEC_WIDTH
        )
        convert_dq_kernel(dq_f32, dq_out, n_elems).launch(
            grid=(fx.Int32(n_blocks), 1, 1),
            block=(BLOCK_THREADS, 1, 1),
            stream=stream,
        )

    return launch_fn
