# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""Arch-generic CDNA/FlyDSL kernel-authoring primitives (wave64).

Scalar/vector math intrinsics, DPP cross-lane helpers, wave reductions, global
loads, and MFMA selection wrappers shared across FlyDSL kernels. Not attention
specific — see mslk.attention.fmha.flydsl.utils, which re-exports these.
"""

import flydsl.expr as fx  # pyre-ignore[21]
from flydsl._mlir import ir  # pyre-ignore[21]
from flydsl._mlir.dialects import (  # pyre-ignore[21]  # pyre-ignore[21]
    llvm,
    math as mlir_math,
)
from flydsl.expr import arith, buffer_ops, rocdl  # pyre-ignore[21]
from flydsl.expr.typing import T  # pyre-ignore[21]
from flydsl.runtime.device import get_rocm_arch  # pyre-ignore[21]
from flydsl.utils.smem_allocator import SMEM_CAPACITY_MAP  # pyre-ignore[21]


WARP_SIZE: int = 64  # CDNA wave64 (gfx942, gfx950)


def smem_bytes(arch=None) -> int:  # pyre-ignore[2]
    """LDS capacity in bytes for the given arch (from FlyDSL's known map)."""
    if arch is None:
        arch = get_rocm_arch()
    cap = SMEM_CAPACITY_MAP.get(arch)
    if cap is None:
        raise ValueError(f"Unsupported arch {arch!r}")
    return cap


# gfx942 (CDNA3/MI300): 64 KB LDS.  gfx950 (CDNA4/MI355): 160 KB LDS.
SMEM_BYTES_GFX942 = 65536
SMEM_BYTES_GFX950 = 163840


# ── Scalar / vector math intrinsics ─────────────────────────────────────────


def rcp_f32(value):  # pyre-ignore[2,3]
    """Reciprocal via `llvm.amdgcn.rcp.f32` (single instruction)."""
    return rocdl.rcp(T.f32, value)


def exp_f32(value):  # pyre-ignore[2,3]
    """Scalar `e^value` via mlir math.exp. Use (not exp2) to match CK natural-exp softmax."""
    raw = (
        arith.unwrap(value)
        if hasattr(value, "ir_value") or hasattr(value, "type")
        else value
    )
    return mlir_math.exp(raw)


def exp2_f32(value):  # pyre-ignore[2,3]
    """Scalar `2^value` via `llvm.amdgcn.exp2.f32` (single v_exp_f32). Used by the
    exp2-domain softmax in the MFMA decode kernels."""
    raw = arith.unwrap(value) if hasattr(value, "ir_value") else value
    return fx.Float32(
        llvm.call_intrinsic(ir.F32Type.get(), "llvm.amdgcn.exp2.f32", [raw], [], [])
    )


def maxnumf(a, b):  # pyre-ignore[2,3]
    """Non-NaN-propagating max — single `v_max_f32` instruction."""
    return type(a)(arith.maxnumf(arith.unwrap(a), arith.unwrap(b)))


def select_f32(cond, a, b):  # pyre-ignore[2,3]
    return arith.select(cond, arith.unwrap(a), arith.unwrap(b))


# ── DPP cross-lane helpers (wave64 CDNA only) ────────────────────────────────


def _dpp_xor_i32_raw(src_i32, offset: int):  # pyre-ignore[2,3]
    """Butterfly-XOR within a 16-lane DPP row (wave64), offsets 1,2,4,8 only.

    For offsets 16,32 use shuffle_xor/ds_swizzle. DPP control values from AMD ISA.
    """
    from flydsl._mlir.dialects import llvm as _llvm  # pyre-ignore[21]
    from flydsl._mlir.ir import IntegerType  # pyre-ignore[21]

    def _upd(src, old, ctrl, rmask, bmask):  # pyre-ignore[2,3]
        i1_ty = IntegerType.get_signless(1)
        bound_false = arith.constant(0, type=i1_ty)
        return _llvm.call_intrinsic(
            T.i32,
            "llvm.amdgcn.update.dpp.i32",
            [
                old,
                src,
                arith.unwrap(arith.constant(ctrl, type=T.i32)),
                arith.unwrap(arith.constant(rmask, type=T.i32)),
                arith.unwrap(arith.constant(bmask, type=T.i32)),
                bound_false,
            ],
            [],
            [],
        )

    if offset == 8:
        out = _upd(src_i32, src_i32, 280, 0xF, 0xC)
        out = _upd(src_i32, out, 264, 0xF, 0x3)
    elif offset == 4:
        out = _upd(src_i32, src_i32, 276, 0xF, 0xA)
        out = _upd(src_i32, out, 260, 0xF, 0x5)
    elif offset == 2:
        out = _upd(src_i32, src_i32, 78, 0xF, 0xF)
    elif offset == 1:
        out = _upd(src_i32, src_i32, 177, 0xF, 0xF)
    else:
        raise ValueError(f"dpp_xor only supports offsets 1,2,4,8; got {offset}")
    return out


def dpp_xor_f32(src, offset: int):  # pyre-ignore[2,3]
    """F32 butterfly-XOR within a 16-lane DPP row (wave64, offsets 1/2/4/8)."""
    from flydsl._mlir.dialects import arith as _arith_dialect  # pyre-ignore[21]

    raw = arith.unwrap(src) if hasattr(src, "ir_value") else src
    src_i32 = _arith_dialect.BitcastOp(T.i32, raw).result
    out_i32 = _dpp_xor_i32_raw(src_i32, offset)
    return fx.Float32(_arith_dialect.BitcastOp(T.f32, out_i32).result)


def wave_reduce_max_f32(val):  # pyre-ignore[2,3]
    """Full wave64 max reduction: DPP XOR (8,4,2,1) then shuffle_xor (32,16)."""
    for sh in (8, 4, 2, 1):
        val = maxnumf(val, dpp_xor_f32(val, sh))
    c_w = arith.constant(WARP_SIZE, type=T.i32)
    for sh in (32, 16):
        other = val.shuffle_xor(arith.constant(sh, type=T.i32), c_w)
        val = maxnumf(val, fx.Float32(other))
    return val


def wave_reduce_sum_f32(val):  # pyre-ignore[2,3]
    """Full wave64 warp-level sum reduction."""
    for sh in (8, 4, 2, 1):
        val = fx.Float32(
            arith.addf(arith.unwrap(val), arith.unwrap(dpp_xor_f32(val, sh)))
        )
    c_w = arith.constant(WARP_SIZE, type=T.i32)
    for sh in (32, 16):
        other = val.shuffle_xor(arith.constant(sh, type=T.i32), c_w)
        val = fx.Float32(arith.addf(arith.unwrap(val), arith.unwrap(fx.Float32(other))))
    return val


# ── Global pointer extraction ────────────────────────────────────────────────


def extract_global_ptr(tensor):  # pyre-ignore[2,3]
    """Extract a raw `!llvm.ptr<1>` from a FlyDSL tensor argument."""
    from flydsl._mlir.dialects import fly as _fly  # pyre-ignore[21]

    raw = (
        tensor.ir_value()
        if hasattr(tensor, "ir_value") and not isinstance(tensor, ir.Value)
        else tensor
    )
    ptr_type = ir.Type.parse("!llvm.ptr<1>")
    return _fly.extract_aligned_pointer_as_index(ptr_type, raw)


def global_load_f32(global_ptr, byte_offset_i64):  # pyre-ignore[2,3]
    """Load one f32 from a raw global pointer + byte offset."""
    ptr = buffer_ops.get_element_ptr(
        global_ptr, byte_offset=fx.Int64(byte_offset_i64), elem_type=T.i8
    )
    return llvm.LoadOp(T.f32, ptr, alignment=4).result


def global_load_f16x2(global_ptr, byte_offset_i64):  # pyre-ignore[2,3]
    """Load a packed pair of f16 values (32-bit aligned)."""
    ptr = buffer_ops.get_element_ptr(
        global_ptr, byte_offset=fx.Int64(byte_offset_i64), elem_type=T.i8
    )
    return llvm.LoadOp(T.i32, ptr, alignment=4).result


def global_load_i64x2(global_ptr, byte_offset_i64):  # pyre-ignore[2,3]
    """Load 128 bits (two i64) from a raw global pointer + byte offset."""
    ptr = buffer_ops.get_element_ptr(
        global_ptr, byte_offset=fx.Int64(byte_offset_i64), elem_type=T.i8
    )
    return llvm.LoadOp(T.i64x2, ptr, alignment=16).result


# ── MFMA selection helpers ───────────────────────────────────────────────────


def mfma_f32_16x16x16_f16(a, b, acc):  # pyre-ignore[2,3]
    """f16 × f16 → f32 MFMA (16×16×16)."""
    return rocdl.mfma_f32_16x16x16f16(T.f32x4, [a, b, acc, 0, 0, 0])


def mfma_f32_16x16x16_bf16(a, b, acc):  # pyre-ignore[2,3]
    """bf16 × bf16 → f32 MFMA (16×16×16); uses the 1k (accumulator) variant."""
    return rocdl.mfma_f32_16x16x16bf16_1k(T.f32x4, [a, b, acc, 0, 0, 0])


def mfma_f32_16x16x4_f32(a, b, acc):  # pyre-ignore[2,3]
    """f32 × f32 → f32 MFMA (16×16×4)."""
    return rocdl.mfma_f32_16x16x4f32(T.f32x4, [a, b, acc, 0, 0, 0])
