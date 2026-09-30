# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""FlyDSL split-K combine (reduce) kernel for paged-attention decode.

Inputs (from decode kernel):
  partial_out  : [B, G, max_parts, H_q, D]  f32   (UN-normalized numerator sum(p*v))
  partial_max  : [B, G, max_parts, H_q]     f32   (per-partition global_max)
  partial_sum  : [B, G, max_parts, H_q]     f32   (per-partition exp sum)
  out          : [B, G, H_q, D]             target dtype
Grid (B,G,H_q); Block (WARP_SIZE=64,1,1). Lane handles _CHUNKS=D//64 head-dim pos.

GOTCHA: partial_out is the un-normalized numerator, so combine each partition by
weight w only (NOT w*partial_sum) — the sum is already folded in.

Fast path (max_parts ≤ 64): lane l owns partition l; warp reduce for global
  max/sum; ds_bpermute broadcasts each partition's normalized weight.
Slow path (max_parts > 64): LDS-staged stats, each lane accumulates independently.
"""

from __future__ import annotations

import functools
from typing import Any, Dict, List, Tuple

import flydsl.compiler as flyc  # pyre-ignore[21]
import flydsl.expr as fx  # pyre-ignore[21]
import torch
from flydsl.expr import (  # pyre-ignore[21]
    arith,
    buffer_ops,
    const_expr,
    gpu,
    range_constexpr,
    rocdl,
    vector,
)
from flydsl.expr.typing import Int32, T  # pyre-ignore[21]
from flydsl.runtime.device import get_rocm_arch  # pyre-ignore[21]
from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr  # pyre-ignore[21]

from .utils import exp_f32, rcp_f32, WARP_SIZE, wave_reduce_max_f32, wave_reduce_sum_f32

_DTYPE_MAP = {
    "f32": (torch.float32, fx.Float32),
    "f16": (torch.float16, fx.Float16),
    "bf16": (torch.bfloat16, fx.BFloat16),
}


def _fx_dtype(dtype_str: str):  # pyre-ignore[3]
    return _DTYPE_MAP[dtype_str][1]


# ── Compiled reduce kernel ────────────────────────────────────────────────────


@functools.lru_cache(maxsize=256)
def _compile_reduce(
    head_size: int,
    max_parts: int,
    output_dtype_str: str,
    arch: str,
) -> Tuple[Any, Any]:  # pyre-ignore[3]
    _HEAD = head_size
    _MAX_PARTS = max_parts
    _FAST = _MAX_PARTS <= WARP_SIZE
    # Slow path stages stats in LDS one wave at a time; pad so every lane has a
    # slot and the staging loop needs no bound branch.
    _PAD_PARTS = ((_MAX_PARTS + WARP_SIZE - 1) // WARP_SIZE) * WARP_SIZE
    _OUT_FX = _fx_dtype(output_dtype_str)
    _CHUNKS = _HEAD // WARP_SIZE

    allocator = SmemAllocator(
        None,
        arch=arch,
        global_sym_name=f"pa_red_p{_MAX_PARTS}_h{_HEAD}_{output_dtype_str}",
    )
    if not _FAST:
        allocator.ptr = 2 * _PAD_PARTS * 4  # max + sum, f32 each

    @flyc.kernel(known_block_size=(WARP_SIZE, 1, 1))
    def _kernel(
        output_ptr: fx.Tensor,
        partial_out_ptr: fx.Tensor,
        partial_max_ptr: fx.Tensor,
        partial_sum_ptr: fx.Tensor,
        # partial_out strides: [B, G, SK, Hq, D]
        s_po_b: Int32,
        s_po_g: Int32,
        s_po_part: Int32,
        s_po_hq: Int32,
        # partial_max/sum strides: [B, G, SK, Hq] — Hq innermost (stride=1)
        s_pm_b: Int32,
        s_pm_g: Int32,
        s_pm_part: Int32,
        # output strides: [B, G, Hq, D]
        s_o_b: Int32,
        s_o_g: Int32,
        s_o_hq: Int32,
    ) -> None:
        lane = gpu.thread_idx.x  # 0..WARP_SIZE-1
        bid_b = gpu.block_idx.x
        bid_g = gpu.block_idx.y
        bid_hq = gpu.block_idx.z

        c_zero = arith.constant(0.0, type=T.f32)
        c_one = arith.constant(1.0, type=T.f32)
        c_neginf = arith.constant(float("-inf"), type=T.f32)
        c_gmax_floor = arith.constant(-3.0e38, type=T.f32)

        po_rsrc = buffer_ops.create_buffer_resource(partial_out_ptr, max_size=True)
        pm_rsrc = buffer_ops.create_buffer_resource(partial_max_ptr, max_size=True)
        ps_rsrc = buffer_ops.create_buffer_resource(partial_sum_ptr, max_size=True)
        out_rsrc = buffer_ops.create_buffer_resource(output_ptr, max_size=True)

        # hq has stride 1 in [B,G,SK,Hq]
        pm_base = bid_b * s_pm_b + bid_g * s_pm_g + bid_hq
        po_base_hq = bid_b * s_po_b + bid_g * s_po_g + bid_hq * s_po_hq
        o_base = bid_b * s_o_b + bid_g * s_o_g + bid_hq * s_o_hq

        if const_expr(_FAST):
            # Lane l owns partition l's statistics.
            c_mp = arith.constant(_MAX_PARTS, type=T.i32)
            active = lane < c_mp

            # Clamp the address the same way the slow path does. The load is
            # issued by all WARP_SIZE lanes and only masked afterwards, so with
            # _MAX_PARTS < WARP_SIZE the inactive lanes address up to
            # (WARP_SIZE - _MAX_PARTS) * s_pm_part past the end of pm/ps. Those
            # resources are max_size=True, so nothing bounds-checks them and the
            # read walks raw memory -- it faults only when the overrun happens to
            # clear the allocator's slab. Re-reading the last valid partition
            # costs nothing; `active` still discards the value.
            lane_ld = arith.select(
                arith.unwrap(active),
                arith.unwrap(lane),
                arith.constant(_MAX_PARTS - 1, type=T.i32),
            )
            pm_off = pm_base + fx.Int32(lane_ld) * s_pm_part
            p_max_r = buffer_ops.buffer_load(pm_rsrc, pm_off, vec_width=1, dtype=T.f32)
            p_sum_r = buffer_ops.buffer_load(ps_rsrc, pm_off, vec_width=1, dtype=T.f32)
            part_max = arith.select(active, p_max_r, c_neginf)
            part_sum = arith.select(active, p_sum_r, c_zero)

            gmax = arith.unwrap(wave_reduce_max_f32(fx.Float32(part_max)))
            # An empty request leaves every partition at -inf; keep gmax finite so
            # the weights come out 0 rather than exp(-inf - -inf) = NaN.
            gmax = arith.maximumf(gmax, c_gmax_floor)
            diff = arith.subf(part_max, gmax)
            w_f32 = arith.select(active, arith.unwrap(exp_f32(diff)), c_zero)
            gsum = arith.unwrap(
                wave_reduce_sum_f32(fx.Float32(arith.mulf(w_f32, part_sum)))
            )
            inv_sum = arith.unwrap(
                rcp_f32(fx.Float32(arith.select(gsum > c_zero, gsum, c_one)))
            )

            norm_w = arith.mulf(w_f32, inv_sum)
            norm_w32 = arith.bitcast(T.i32, norm_w)

            # Lane owns a contiguous _CHUNKS-wide slice, so one vec load replaces
            # _CHUNKS scalar loads per partition (coalescing preserved).
            base_hd = lane * fx.Int32(_CHUNKS)
            accs = [c_zero] * _CHUNKS
            for p in range_constexpr(_MAX_PARTS):
                src = arith.constant(p * 4, type=T.i32)
                wi32 = rocdl.ds_bpermute(T.i32, src, norm_w32)
                wf32 = arith.bitcast(T.f32, wi32)
                poff = po_base_hq + arith.constant(p, type=T.i32) * s_po_part
                vals = buffer_ops.buffer_load(
                    po_rsrc, poff + base_hd, vec_width=_CHUNKS, dtype=T.f32
                )
                if const_expr(_CHUNKS == 1):
                    accs[0] = arith.addf(accs[0], arith.mulf(vals, wf32))
                else:
                    for c in range_constexpr(_CHUNKS):
                        val = vector.extract(
                            vals, static_position=[c], dynamic_position=[]
                        )
                        accs[c] = arith.addf(accs[c], arith.mulf(val, wf32))

            for c in range_constexpr(_CHUNKS):
                out_val = _OUT_FX(arith.unwrap(fx.Float32(accs[c])))
                buffer_ops.buffer_store(
                    arith.unwrap(out_val), out_rsrc, o_base + base_hd + fx.Int32(c)
                )

        else:
            smem = allocator.get_base()
            lm_lds = SmemPtr(smem, 0, T.f32, shape=(_PAD_PARTS,)).get()
            ls_lds = SmemPtr(smem, _PAD_PARTS * 4, T.f32, shape=(_PAD_PARTS,)).get()

            # `p` is lane-dependent, so it cannot be a const_expr/arith.constant.
            #
            # The LDS arrays are padded to a whole number of waves, so every lane
            # has a slot to write and no branch is needed; the load index is clamped
            # instead, which duplicates a valid entry into the padding. The padded
            # slots are never read back: the accumulation loops below are
            # range_constexpr over _MAX_PARTS.
            for step in range_constexpr(_PAD_PARTS // WARP_SIZE):
                p = fx.Int32(step * WARP_SIZE) + lane
                p_ld = arith.select(
                    arith.unwrap(p < fx.Int32(_MAX_PARTS)),
                    arith.unwrap(p),
                    arith.constant(_MAX_PARTS - 1, type=T.i32),
                )
                pm_off = pm_base + fx.Int32(p_ld) * s_pm_part
                lm = buffer_ops.buffer_load(pm_rsrc, pm_off, vec_width=1, dtype=T.f32)
                ls = buffer_ops.buffer_load(ps_rsrc, pm_off, vec_width=1, dtype=T.f32)
                vector.store(
                    fx.Vector.from_elements([lm], dtype=fx.Float32),
                    lm_lds,
                    [fx.Index(p)],
                )
                vector.store(
                    fx.Vector.from_elements([ls], dtype=fx.Float32),
                    ls_lds,
                    [fx.Index(p)],
                )
            gpu.barrier()

            gmax = c_neginf
            for p in range_constexpr(_MAX_PARTS):
                v = fx.Vector.load(
                    T.vec(1, T.f32), lm_lds, [fx.Index(arith.constant(p, type=T.i32))]
                )[0]
                gmax = arith.maximumf(gmax, arith.unwrap(fx.Float32(v)))
            gmax = arith.maximumf(gmax, c_gmax_floor)

            gsum = c_zero
            accs = [c_zero] * _CHUNKS
            for p in range_constexpr(_MAX_PARTS):
                vm = fx.Vector.load(
                    T.vec(1, T.f32), lm_lds, [fx.Index(arith.constant(p, type=T.i32))]
                )[0]
                vs = fx.Vector.load(
                    T.vec(1, T.f32), ls_lds, [fx.Index(arith.constant(p, type=T.i32))]
                )[0]
                lm_v = arith.unwrap(fx.Float32(vm))
                ls_v = arith.unwrap(fx.Float32(vs))
                w = arith.unwrap(exp_f32(arith.subf(lm_v, gmax)))
                # partial_out is the UN-normalized numerator sum(p*v), so the
                # numerator weight is w alone; part_sum only enters the
                # denominator. (The fast path above does the same: it accumulates
                # vals * norm_w, never vals * part_sum.)
                gsum = arith.addf(gsum, arith.mulf(w, ls_v))
                poff = po_base_hq + arith.constant(p, type=T.i32) * s_po_part
                for c in range_constexpr(_CHUNKS):
                    hd = lane + fx.Int32(c * WARP_SIZE)
                    val = buffer_ops.buffer_load(
                        po_rsrc, poff + hd, vec_width=1, dtype=T.f32
                    )
                    accs[c] = arith.addf(accs[c], arith.mulf(val, w))

            safe = arith.select(gsum > c_zero, gsum, c_one)
            for c in range_constexpr(_CHUNKS):
                hd = lane + fx.Int32(c * WARP_SIZE)
                out_val = _OUT_FX(arith.unwrap(fx.Float32(arith.divf(accs[c], safe))))
                buffer_ops.buffer_store(arith.unwrap(out_val), out_rsrc, o_base + hd)

    return _kernel, allocator


def compile_pa_decode_reduce(
    *,
    head_size: int,
    max_parts: int,
    output_dtype_str: str = "f32",
    arch: str = "",
) -> Any:  # pyre-ignore[3]
    if not arch:
        arch = get_rocm_arch()
    kernel, _ = _compile_reduce(head_size, max_parts, output_dtype_str, arch)
    return kernel


# ── JIT launcher ─────────────────────────────────────────────────────────────


@functools.lru_cache(maxsize=256)
def _make_reduce_jit_launcher(
    head_size: int,
    max_parts: int,
    output_dtype_str: str,
    arch: str,
):  # pyre-ignore[3]
    kernel, _alloc = _compile_reduce(head_size, max_parts, output_dtype_str, arch)
    _fast = max_parts <= WARP_SIZE

    @flyc.jit
    def _launcher(
        output_ptr: fx.Tensor,
        partial_out_ptr: fx.Tensor,
        partial_max_ptr: fx.Tensor,
        partial_sum_ptr: fx.Tensor,
        s_po_b: Int32,
        s_po_g: Int32,
        s_po_part: Int32,
        s_po_hq: Int32,
        s_pm_b: Int32,
        s_pm_g: Int32,
        s_pm_part: Int32,
        s_o_b: Int32,
        s_o_g: Int32,
        s_o_hq: Int32,
        grid_b: Int32,
        grid_g: Int32,
        grid_hq: Int32,
        stream: fx.Stream = fx.Stream(None),
    ) -> None:
        from flydsl._mlir import ir as _ir  # pyre-ignore[21]
        from flydsl.compiler.kernel_function import (  # pyre-ignore[21]
            CompilationContext,
        )

        if not _fast:
            _alloc.finalized = False
            ctx = CompilationContext.get_current()
            with _ir.InsertionPoint(ctx.gpu_module_body):
                _alloc.finalize()

        kernel(
            output_ptr,
            partial_out_ptr,
            partial_max_ptr,
            partial_sum_ptr,
            s_po_b,
            s_po_g,
            s_po_part,
            s_po_hq,
            s_pm_b,
            s_pm_g,
            s_pm_part,
            s_o_b,
            s_o_g,
            s_o_hq,
        ).launch(grid=(grid_b, grid_g, grid_hq), block=(WARP_SIZE, 1, 1), stream=stream)

    return _launcher


# ── Host API ─────────────────────────────────────────────────────────────────


def pa_decode_reduce(
    partial_out: torch.Tensor,  # [B, G, max_parts, H_q, D]  f32
    partial_max: torch.Tensor,  # [B, G, max_parts, H_q]     f32
    partial_sum: torch.Tensor,  # [B, G, max_parts, H_q]     f32
    output: torch.Tensor,  # [B, G, H_q, D]             target dtype
    stream: object = None,
) -> None:
    """Combine split-K partitions into the final output (in-place).

    Pass the caller's stream so the reduce is captured on the same stream as the
    compute kernel under CUDA graphs; defaults to the current stream.
    """
    from mslk.flydsl.jit import run_compiled  # pyre-ignore[21]

    B, G, max_parts, H_q, D = partial_out.shape
    dtype_str = {torch.float32: "f32", torch.float16: "f16", torch.bfloat16: "bf16"}[
        output.dtype
    ]
    arch = get_rocm_arch()
    launcher = _make_reduce_jit_launcher(D, max_parts, dtype_str, arch)

    if stream is None:
        stream = torch.cuda.current_stream()

    po, pm, o = partial_out, partial_max, output
    run_compiled(
        launcher,
        output,
        partial_out,
        partial_max,
        partial_sum,
        po.stride(0),
        po.stride(1),
        po.stride(2),
        po.stride(3),
        pm.stride(0),
        pm.stride(1),
        pm.stride(2),
        o.stride(0),
        o.stride(1),
        o.stride(2),
        B,
        G,
        H_q,
        stream,
    )


# ── AOT interface ─────────────────────────────────────────────────────────────

AOT_ARCHS: List[str] = ["gfx942", "gfx950"]

AOT_CONFIGS: List[Dict[str, Any]] = [
    {"head_size": hs, "max_parts": mp, "output_dtype_str": dt}
    for hs in (64, 128, 256)
    # Matches pa_decode_dense._SPLIT_KS; 128 and 256 take the slow path.
    for mp in (1, 2, 4, 8, 16, 32, 64, 128, 256)
    for dt in ("f32", "f16", "bf16")
]


def compile_aot_config(config: Dict[str, Any], arch: str) -> None:
    compile_pa_decode_reduce(
        head_size=config["head_size"],
        max_parts=config["max_parts"],
        output_dtype_str=config["output_dtype_str"],
        arch=arch,
    )
