# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# SPDX-License-Identifier: Apache-2.0

"""Building blocks shared by the FP8 GEMM kernels built on the wide gfx950 MFMA.

Used by ``fp8_groupwise_wide_gemm`` and ``fp8_rowwise_gemm``: the MFMA geometry,
the LDS access path (direct-to-LDS transfers and swizzled fragment reads, kept
apart by alias scopes), the neutral-scale MFMA, and the ``permlane32_swap``
gather that widens the epilogue's stores. See ``fp8_groupwise_wide_gemm`` for
the reasoning behind each.

Fragment layouts for the wide MFMA:

  A operand   m = lane % 32,  k = (lane // 32) * 32 + byte
  B operand   n = lane % 32,  k = (lane // 32) * 32 + byte
  accumulator n = lane % 32,  m = 4 * (lane // 32) + 8 * (reg // 4) + reg % 4
"""

import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm, memref as memref_dialect
from flydsl.expr import arith, range_constexpr, rocdl
from flydsl.expr.typing import T, Vector
from flydsl.expr.utils.arith import _to_raw as _raw
from mslk.flydsl.kernels.mma.mfma_preshuffle_pipeline import swizzle_xor16

# MFMA geometry: one issue is a 32x32 output tile contracting 64 of K.
MFMA_M = 32
MFMA_K = 64
# Bytes each lane supplies per operand per issue (32 FP8 = 8 i32).
MFMA_OPERAND_BYTES = 32
# Accumulator registers per MFMA tile.
ACC_REGS = MFMA_M * MFMA_M // 64

WAVE = 64

# E8M0 exponent bias: a scale of 2^0, so the instruction's block scaling is the
# identity and the FP32 scales can be applied in software instead.
NEUTRAL_E8M0 = 0x7F7F7F7F

# Widest global load, and hence the LDS swizzle granularity.
LOAD_BYTES = 16

# The two alias scopes LDS accesses are tagged with: the direct-to-LDS
# transfers, and the fragment reads.
SCOPE_DMA, SCOPE_READS = 0, 1
_SCOPE_NAMES = ("dma", "reads")


class LdsAccess:
    """LDS addressing for one kernel's tile arena, built while tracing.

    ``prefix`` names the kernel's alias-scope domain. ``tile_k`` is the row
    width of the tiles in bytes, which the XOR16 swizzle runs over.
    """

    def __init__(self, lds, prefix: str, tile_k: int):
        domain = f'#llvm.alias_scope_domain<id = "{prefix}.lds">'
        self._scopes = [
            ir.Attribute.parse(
                f'#llvm.alias_scope<id = "{prefix}.{nm}", domain = {domain}>'
            )
            for nm in _SCOPE_NAMES
        ]
        self._tile_k = tile_k
        self._chunks = fx.Index(tile_k // LOAD_BYTES)
        self._ptr_ty = ir.Type.parse("!llvm.ptr<3>")
        self._addr_i32 = fx.Int32(
            arith.index_cast(
                T.i32, memref_dialect.extract_aligned_pointer_as_index(lds)
            )
        )
        self._addr_i64 = fx.Int64(
            arith.index_cast(
                T.i64, memref_dialect.extract_aligned_pointer_as_index(lds)
            )
        )
        self._v4i32 = ir.VectorType.get([4], fx.Int32.ir_type)

    def tag(self, op, mine: int) -> None:
        """Claim scope ``mine`` for ``op`` and disclaim the other one."""
        op = getattr(op, "owner", op)
        op.attributes["alias_scopes"] = ir.ArrayAttr.get([self._scopes[mine]])
        op.attributes["noalias_scopes"] = ir.ArrayAttr.get(
            [sc for i, sc in enumerate(self._scopes) if i != mine]
        )

    def swizzle(self, row, col):
        """Column ``col`` of ``row`` under the XOR16 swizzle."""
        return swizzle_xor16(row, col, self._chunks)

    def index(self, row, col, lds_base):
        """Byte offset of tile element ``(row, col)`` under the XOR16 swizzle."""
        return row * fx.Index(self._tile_k) + self.swizzle(row, col) + lds_base

    def ptr(self, byte_idx):
        """An LDS pointer to byte ``byte_idx`` of the arena."""
        return llvm.inttoptr(
            self._ptr_ty, (self._addr_i32 + fx.Int32(byte_idx)).ir_value()
        )

    def dma_tile(self, rsrc, num_loads, src_offset, lds_base, wave_id, threads):
        """Move a tile from global straight into LDS, never touching a VGPR.

        Lane L of a wave lands at a fixed ``lds_ptr + L * 16``, so LDS fills
        linearly and the XOR swizzle has to be applied to the source address,
        which ``src_offset(i)`` returns for the thread's ``i``-th chunk.
        """
        ptr = None
        for i in range_constexpr(num_loads):
            byte_off = src_offset(i)
            if i == 0:
                ptr = rocdl.readfirstlane(
                    T.i64,
                    self._addr_i64
                    + fx.Int64(lds_base)
                    + fx.Int64(wave_id * fx.Index(WAVE * LOAD_BYTES)),
                )
            else:
                ptr = ptr + fx.Int64(threads * LOAD_BYTES)
            self.tag(
                rocdl.raw_ptr_buffer_load_lds(
                    rsrc,
                    llvm.inttoptr(self._ptr_ty, ptr),
                    fx.Int32(LOAD_BYTES),
                    fx.Int32(byte_off),
                    fx.Int32(0),
                    fx.Int32(0),
                    1,
                ),
                SCOPE_DMA,
            )

    def read_frag(self, row, lds_base, ks, lane_hi):
        """Gather one lane's 32-byte operand fragment for MFMA step ``ks``.

        Issued as an LLVM load rather than a memref one so it can carry the
        alias metadata; ``vector.load`` has nowhere to put it.
        """
        halves = []
        for j in range_constexpr(MFMA_OPERAND_BYTES // LOAD_BYTES):
            col = (
                fx.Index(ks * MFMA_K)
                + lane_hi * fx.Index(32)
                + fx.Index(j * LOAD_BYTES)
            )
            ld = llvm.LoadOp(
                self._v4i32, self.ptr(self.index(row, col, lds_base)), alignment=16
            )
            self.tag(ld, SCOPE_READS)
            halves.append(Vector(ld.result))
        return Vector(halves[0]).shuffle(Vector(halves[1]), list(range(8))).ir_value()


def wide_mfma(v_acc_ty, a_op, b_op, c_in):
    """One ``v_mfma_scale_f32_32x32x64_f8f6f4`` with the neutral scale."""
    # Operands: a, b, c, cbsz, blgp, opsel_a, scale_a, opsel_b, scale_b.
    return rocdl.mfma_scale_f32_32x32x64_f8f6f4(
        v_acc_ty,
        [
            _raw(a_op),
            _raw(b_op),
            _raw(c_in),
            0,
            0,
            0,
            _raw(fx.Int32(NEUTRAL_E8M0)),
            0,
            _raw(fx.Int32(NEUTRAL_E8M0)),
        ],
    )


def swap_quad(even, odd):
    """Gather two column groups' 8-byte runs into one 16-byte store per lane.

    ``even`` and ``odd`` are a lane's ``(lo, hi)`` dwords of two adjacent column
    groups. ``permlane32_swap(a, b)`` gives low lanes ``a[L]`` and ``a[L+32]``
    and high lanes ``b[L-32]`` and ``b[L]``, so the low half-wave ends up with
    all eight columns of the even group and the high half with the odd one.
    Columns ascend with the address, so the dwords interleave: this lane's own
    pair, then its partner's.
    """
    pair_ty = ir.Type.parse("!llvm.struct<(i32, i32)>")
    sw_lo = rocdl.permlane32_swap(pair_ty, _raw(even[0]), _raw(odd[0]), False, False)
    sw_hi = rocdl.permlane32_swap(pair_ty, _raw(even[1]), _raw(odd[1]), False, False)
    return Vector.from_elements(
        [
            fx.Int32(llvm.extractvalue(T.i32, sw_lo, [0])),
            fx.Int32(llvm.extractvalue(T.i32, sw_hi, [0])),
            fx.Int32(llvm.extractvalue(T.i32, sw_lo, [1])),
            fx.Int32(llvm.extractvalue(T.i32, sw_hi, [1])),
        ],
        fx.Int32,
    )
