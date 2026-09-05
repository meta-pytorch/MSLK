# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# SPDX-License-Identifier: Apache-2.0

"""FP8 groupwise-scaled GEMM built on the wide gfx950 MFMA.

Computes ``D[M, N] = dequant(A) @ dequant(B).T`` for

  A        [M, K]           FP8 E4M3
  B        [N, K]           FP8 E4M3 (already transposed by the caller)
  scale_a  [K // 128, M]    FP32, one scale per (K-group, row)
  scale_b  [K // 128, N // 128] FP32, one scale per (K-group, N-group)

which is the contract of ``mslk::f8f8bf16_groupwise``.

The schedule:

  * ``v_mfma_scale_f32_32x32x64_f8f6f4`` with a neutral E8M0 scale, which makes
    the hardware block-scaling a no-op and lets the block-scaled instruction
    serve this GEMM. One issue covers K=64 of a 32x32 output tile.
  * A two-dimensional wave grid: ``waves_m x waves_n`` waves each own a
    ``(tile_m / waves_m) x (tile_n / waves_n)`` slab of the output tile. LDS read
    traffic per unit work is ``waves_m / tile_m + waves_n / tile_n``, minimised
    when the wave grid is proportioned like the tile, which a one-dimensional
    split cannot do.
  * Both operands staged through LDS, since every wave needs rows and columns
    that no single lane loads.
  * The dot runs against a zero accumulator and the block scales are folded in
    afterwards, because a scale changes every 128 elements of K while the
    accumulator has to persist across the whole contraction.

Fragment layouts for the wide MFMA:

  A operand   m = lane % 32,  k = (lane // 32) * 32 + byte
  B operand   n = lane % 32,  k = (lane // 32) * 32 + byte
  accumulator n = lane % 32,  m = 4 * (lane // 32) + 8 * (reg // 4) + reg % 4

so a lane's 16 accumulator values sit in one column, across four groups of four
consecutive rows.
"""

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm, math as math_dialect, memref as memref_dialect
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.expr import arith, buffer_ops, gpu, range_constexpr, rocdl
from flydsl.expr.typing import T, Vector
from flydsl.expr.utils.arith import _to_raw as _raw
from flydsl.runtime.device import get_rocm_arch as get_hip_arch
from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr
from mslk.flydsl.kernels.mma.mfma_preshuffle_pipeline import swizzle_xor16

# MFMA geometry: one issue is a 32x32 output tile contracting 64 of K.
MFMA_M = 32
MFMA_K = 64
# Bytes each lane supplies per operand per issue (32 FP8 = 8 i32).
MFMA_OPERAND_BYTES = 32
# Accumulator registers per MFMA tile.
ACC_REGS = MFMA_M * MFMA_M // 64

WAVE = 64

# Scale-block granularity, fixed by the op's contract.
SCALE_BLOCK = 128

# E8M0 exponent bias: a scale of 2^0, so the instruction's block scaling is the
# identity and the FP32 scales can be applied in software instead.
NEUTRAL_E8M0 = 0x7F7F7F7F

# Widest global load, and hence the LDS swizzle granularity.
LOAD_BYTES = 16

# LDS per workgroup on gfx950.
LDS_CAPACITY = 160 * 1024

# Alias scopes telling the backend that the direct-to-LDS transfers and the MFMA
# fragment reads never need a dependence inferred between them.
#
# `buffer_load_lds` writes LDS through an opaque pointer, so SIInsertWaitcnts
# cannot see which bytes it touches and conservatively drains every transfer
# with a vmcnt(0) before each ds_read -- which cancels exactly the overlap the
# pipeline exists to create. Read literally these two op classes *do* alias: the
# transfer issued in one iteration is consumed by the reads in the next. What
# the metadata asserts is not disjointness but that the dependence is carried
# explicitly, by the s_waitcnt and barrier at the top of the loop body.
#
# Because this suppresses the compiler's own analysis, that s_waitcnt becomes
# load-bearing for correctness rather than just for speed: too permissive a wait
# now reads LDS before the transfer lands, silently. The one-tile-deep pipeline
# keeps it at vmcnt(0), the maximally conservative value, so there is no count
# to get wrong.
_LDS_DOMAIN = '#llvm.alias_scope_domain<id = "gw_wide.lds">'
_SCOPE_NAMES = ("dma", "reads")
_SCOPE_DMA, _SCOPE_READS = range(len(_SCOPE_NAMES))

# LDS buffers per operand: one being filled by the next tile's transfer while the
# other is read by this tile's MFMAs. A deeper pipeline would need an exact
# vmcnt rather than the vmcnt(0) above, which the alias metadata makes
# load-bearing for correctness; see the K loop.
STAGES = 2


@functools.lru_cache(maxsize=64)
def compile_groupwise_wide_gemm(
    *,
    n: int,
    k: int,
    tile_m: int = 128,
    tile_n: int = 64,
    tile_k: int = 128,
    waves_m: int = 4,
    waves_n: int = 1,
    waves_per_eu: int | None = None,
):
    """Compile the kernel for one shape and tile config; return the launcher.

    ``waves_m``/``waves_n`` are the wave grid, so the block is
    ``waves_m * waves_n * 64`` threads. The grid is given as a shape rather than
    a wave count because its proportions drive LDS read traffic: reads per unit
    work are ``waves_m / tile_m + waves_n / tile_n``, which a grid proportioned
    like the tile minimises.
    """
    if tile_k != SCALE_BLOCK:
        raise ValueError(
            f"tile_k ({tile_k}) must equal the scale block ({SCALE_BLOCK}): the "
            "scales change every 128 elements of K, and a tile spanning more "
            "than one block would need a fold per sub-block"
        )
    if k % tile_k or n % tile_n:
        raise ValueError(
            f"n ({n}) and k ({k}) must divide by tile_n ({tile_n}) and tile_k "
            f"({tile_k}); this kernel does not mask a partial tile"
        )
    if n % SCALE_BLOCK:
        raise ValueError(f"n ({n}) must be a multiple of {SCALE_BLOCK}")
    if tile_m % (waves_m * MFMA_M) or tile_n % (waves_n * MFMA_M):
        raise ValueError(
            f"tile {tile_m}x{tile_n} does not divide into {waves_m}x{waves_n} "
            f"waves of {MFMA_M}x{MFMA_M} MFMA tiles"
        )
    if tile_k % MFMA_K:
        raise ValueError(f"tile_k ({tile_k}) must be a multiple of {MFMA_K}")
    if tile_n > SCALE_BLOCK:
        raise ValueError(
            f"tile_n ({tile_n}) must not exceed the scale block ({SCALE_BLOCK}), "
            "or a tile would span several B scales"
        )

    num_waves = waves_m * waves_n
    total_threads = num_waves * WAVE
    wave_tile_m = tile_m // waves_m
    wave_tile_n = tile_n // waves_n
    # MFMA tiles each wave owns, and hence its accumulator count.
    acc_m = wave_tile_m // MFMA_M
    acc_n = wave_tile_n // MFMA_M
    k_steps = tile_k // MFMA_K
    num_k_tiles = k // tile_k
    scale_n = n // SCALE_BLOCK

    # Each thread's share of a tile, in whole 16-byte loads.
    a_bytes_per_thread = tile_m * tile_k // total_threads
    b_bytes_per_thread = tile_n * tile_k // total_threads
    if a_bytes_per_thread % LOAD_BYTES or b_bytes_per_thread % LOAD_BYTES:
        raise ValueError(
            f"tile {tile_m}x{tile_n}x{tile_k} does not split into whole "
            f"{LOAD_BYTES}-byte loads across {total_threads} threads"
        )
    num_a_loads = a_bytes_per_thread // LOAD_BYTES
    num_b_loads = b_bytes_per_thread // LOAD_BYTES

    # 16-byte chunks per row, the modulus of the XOR swizzle.
    k_chunks = tile_k // LOAD_BYTES
    # 16-byte reads per operand fragment.
    frag_loads = MFMA_OPERAND_BYTES // LOAD_BYTES

    gpu_arch = get_hip_arch()
    if not str(gpu_arch).startswith("gfx95"):
        raise ValueError(f"the wide f8f6f4 MFMA is gfx950-only; arch is {gpu_arch}")

    allocator = SmemAllocator(None, arch=gpu_arch, global_sym_name="smem_gw_wide")
    lds_a_elems = tile_m * tile_k
    lds_b_elems = tile_n * tile_k
    # A tile is written into one buffer while the other is being read, which is
    # what lets a single barrier per K tile suffice.
    lds_bytes = STAGES * (lds_a_elems + lds_b_elems)
    if lds_bytes > LDS_CAPACITY:
        raise ValueError(
            f"tile {tile_m}x{tile_n}x{tile_k} double-buffered needs "
            f"{lds_bytes} B of LDS, over the {LDS_CAPACITY} B budget"
        )
    lds_off = allocator._align(allocator.ptr, 16)
    allocator.ptr = lds_off + lds_bytes

    # Everything that changes the emitted kernel has to reach the name, or two
    # configs would collide in the compile cache. The occupancy hint does, as
    # much as the tile and the wave grid do.
    _wpe = f"_wpe{int(waves_per_eu)}" if waves_per_eu else ""
    module_name = (
        f"gw_wide_n{n}_k{k}_t{tile_m}x{tile_n}x{tile_k}_w{waves_m}x{waves_n}{_wpe}"
    )

    # The AMDGPU default caps a workgroup at 256 threads; an eight-wave block
    # needs the larger bound declared up front.
    @flyc.kernel(name=module_name, known_block_size=[total_threads, 1, 1])
    def groupwise_wide_gemm_kernel(
        arg_d: fx.Tensor,
        arg_a: fx.Tensor,
        arg_b: fx.Tensor,
        arg_scale_a: fx.Tensor,
        arg_scale_b: fx.Tensor,
        i32_m: fx.Int32,
        i32_n: fx.Int32,
        i32_k: fx.Int32,
    ):
        # MLIR types need a live context, so they are built during tracing.
        v_acc_ty = Vector.make_type(ACC_REGS, fx.Float32)

        m_in = fx.Index(i32_m)
        n_in = fx.Index(i32_n)
        k_in = fx.Index(i32_k)

        tx = gpu.thread_id("x")
        by = gpu.block_id("x")  # N-block
        bx = gpu.block_id("y")  # M-tile

        bx_m = bx * fx.Index(tile_m)
        by_n = by * fx.Index(tile_n)

        # Wave grid: wave w owns rows [wm * wave_tile_m, +wave_tile_m) and
        # columns [wn * wave_tile_n, +wave_tile_n).
        wave_id = tx // fx.Index(WAVE)
        lane = tx % fx.Index(WAVE)
        wm = wave_id // fx.Index(waves_n)
        wn = wave_id % fx.Index(waves_n)
        # The wide MFMA splits a lane's role by half-wave: which 32 of K it
        # supplies, and which four of the 32 output rows it holds.
        lane_lo = lane % fx.Index(MFMA_M)
        lane_hi = lane // fx.Index(MFMA_M)

        wave_m0 = wm * fx.Index(wave_tile_m)
        wave_n0 = wn * fx.Index(wave_tile_n)

        a_rsrc = buffer_ops.create_buffer_resource(
            arg_a, max_size=False, num_records_bytes=m_in * k_in
        )
        b_rsrc = buffer_ops.create_buffer_resource(
            arg_b, max_size=False, num_records_bytes=n_in * k_in
        )
        d_rsrc = buffer_ops.create_buffer_resource(
            arg_d, max_size=False, num_records_bytes=m_in * n_in * fx.Index(2)
        )
        sa_rsrc = buffer_ops.create_buffer_resource(
            arg_scale_a,
            max_size=False,
            num_records_bytes=m_in * fx.Index(num_k_tiles * 4),
        )
        sb_rsrc = buffer_ops.create_buffer_resource(
            arg_scale_b,
            max_size=False,
            num_records_bytes=fx.Index(scale_n * num_k_tiles * 4),
        )

        base_ptr = allocator.get_base()
        # One arena: the A buffers, then the B buffers. Both operands are
        # row-major over K with the same stride, so one indexing helper serves
        # both.
        lds_a = SmemPtr(base_ptr, lds_off, T.f8, shape=(lds_bytes,)).get()

        def a_buf(slot):
            return slot * fx.Index(lds_a_elems)

        def b_buf(slot):
            return fx.Index(STAGES * lds_a_elems) + slot * fx.Index(lds_b_elems)

        c_chunks = fx.Index(k_chunks)

        # ---- staging coordinates -------------------------------------------
        # Load i of thread tx covers tile bytes [(i * threads + tx) * 16, +16),
        # so consecutive lanes walk a row and the reads coalesce.
        def stage_coords(num_loads):
            coords = []
            for i in range_constexpr(num_loads):
                linear = (fx.Index(i * total_threads) + tx) * fx.Index(LOAD_BYTES)
                coords.append((linear // fx.Index(tile_k), linear % fx.Index(tile_k)))
            return coords

        a_coords = stage_coords(num_a_loads)
        b_coords = stage_coords(num_b_loads)

        def lds_index(row, col, lds_base):
            """Byte offset of tile element ``(row, col)`` under the XOR16 swizzle.

            Row-major LDS would start every row in bank 0, so a fragment read --
            which is exactly "each lane takes a different row" -- would serialise
            32 ways. XORing the chunk index with the row spreads them out.
            """
            return row * fx.Index(tile_k) + swizzle_xor16(row, col, c_chunks) + lds_base

        _scopes = [
            ir.Attribute.parse(
                f'#llvm.alias_scope<id = "gw_wide.{nm}", domain = {_LDS_DOMAIN}>'
            )
            for nm in _SCOPE_NAMES
        ]

        def tag_alias(op, mine):
            """Claim scope ``mine`` for ``op`` and disclaim the other one."""
            op = getattr(op, "owner", op)
            op.attributes["alias_scopes"] = ir.ArrayAttr.get([_scopes[mine]])
            op.attributes["noalias_scopes"] = ir.ArrayAttr.get(
                [sc for i, sc in enumerate(_scopes) if i != mine]
            )

        lds_ptr_ty = ir.Type.parse("!llvm.ptr<3>")
        lds_addr_i32 = fx.Int32(
            arith.index_cast(
                T.i32, memref_dialect.extract_aligned_pointer_as_index(lds_a)
            )
        )
        lds_addr_base = fx.Int64(
            arith.index_cast(
                T.i64, memref_dialect.extract_aligned_pointer_as_index(lds_a)
            )
        )

        def dma_tile(rsrc, coords, row_base, kt, lds_base):
            """Move a tile from global straight into LDS, never touching a VGPR.

            The destination is not ours to choose: lane L of a wave lands at a
            fixed ``lds_ptr + L * 16``, so LDS fills linearly and the XOR swizzle
            moves to the *source* address instead. That is sound because the
            swizzle is an XOR and so its own inverse -- reading
            ``global(row, swizzle(row, col))`` into the linear slot for
            ``(row, col)`` leaves LDS holding what a swizzled store would have
            written, which is what the fragment reads expect.
            """
            ptr = None
            for i in range_constexpr(len(coords)):
                row, col = coords[i]
                byte_off = (
                    (row_base + row) * k_in
                    + kt * fx.Index(tile_k)
                    + swizzle_xor16(row, col, c_chunks)
                )
                if i == 0:
                    ptr = rocdl.readfirstlane(
                        T.i64,
                        lds_addr_base
                        + fx.Int64(lds_base)
                        + fx.Int64(wave_id * fx.Index(WAVE * LOAD_BYTES)),
                    )
                else:
                    ptr = ptr + fx.Int64(total_threads * LOAD_BYTES)
                tag_alias(
                    rocdl.raw_ptr_buffer_load_lds(
                        rsrc,
                        llvm.inttoptr(lds_ptr_ty, ptr),
                        fx.Int32(LOAD_BYTES),
                        fx.Int32(byte_off),
                        fx.Int32(0),
                        fx.Int32(0),
                        fx.Int32(1),
                    ),
                    _SCOPE_DMA,
                )

        def dma_stage(kt, slot):
            dma_tile(a_rsrc, a_coords, bx_m, kt, a_buf(slot))
            dma_tile(b_rsrc, b_coords, by_n, kt, b_buf(slot))

        # ---- fragment reads --------------------------------------------------
        v4i32 = ir.VectorType.get([4], fx.Int32.ir_type)

        def read_frag(row, lds_base, ks):
            """Gather one lane's 32-byte operand fragment for MFMA step ``ks``.

            Issued as an LLVM load rather than a memref one so it can carry the
            alias metadata; `vector.load` has nowhere to put it, and the
            attributes would be dropped on the way down.
            """
            halves = []
            for j in range_constexpr(frag_loads):
                col = (
                    fx.Index(ks * MFMA_K)
                    + lane_hi * fx.Index(32)
                    + fx.Index(j * LOAD_BYTES)
                )
                idx = lds_index(row, col, lds_base)
                ptr = llvm.inttoptr(
                    lds_ptr_ty, (lds_addr_i32 + fx.Int32(idx)).ir_value()
                )
                ld = llvm.LoadOp(v4i32, ptr, alignment=16)
                tag_alias(ld, _SCOPE_READS)
                halves.append(Vector(ld.result))
            return (
                Vector(halves[0]).shuffle(Vector(halves[1]), list(range(8))).ir_value()
            )

        def mfma(a_op, b_op, c_in):
            # The scale operands are the neutral exponent, so the instruction's
            # own block scaling is the identity and the FP32 scales are folded in
            # afterwards instead.
            return rocdl.mfma_scale_f32_32x32x64_f8f6f4(
                v_acc_ty,
                _raw(a_op),
                _raw(b_op),
                _raw(c_in),
                0,
                0,
                0,
                _raw(fx.Int32(NEUTRAL_E8M0)),
                0,
                _raw(fx.Int32(NEUTRAL_E8M0)),
            ).result

        zero_acc = Vector.from_elements(
            [fx.Float32(0.0) for _ in range_constexpr(ACC_REGS)], fx.Float32
        ).ir_value()

        def tile_product(a_base, b_base):
            """Contract one K tile into a fresh accumulator.

            The dot starts from zero rather than the running accumulator because
            the scales apply per K tile; folding them afterwards is what lets a
            single accumulator span the whole of K.
            """
            tiles = [zero_acc for _ in range_constexpr(acc_m * acc_n)]
            for ks in range_constexpr(k_steps):
                a_frags = []
                for ai in range_constexpr(acc_m):
                    row = wave_m0 + fx.Index(ai * MFMA_M) + lane_lo
                    a_frags.append(read_frag(row, a_base, ks))
                b_frags = []
                for aj in range_constexpr(acc_n):
                    row = wave_n0 + fx.Index(aj * MFMA_M) + lane_lo
                    b_frags.append(read_frag(row, b_base, ks))
                for ai in range_constexpr(acc_m):
                    for aj in range_constexpr(acc_n):
                        idx = ai * acc_n + aj
                        # Operands swapped on purpose. The hardware puts the
                        # second operand's index on lane % 32 and the first
                        # one's across the registers; feeding B first therefore
                        # transposes the result, so a lane ends up holding one
                        # output row and four runs of four adjacent columns.
                        # That buys a vectorised store, and it makes the block
                        # scale constant across a lane's whole accumulator.
                        tiles[idx] = mfma(b_frags[aj], a_frags[ai], tiles[idx])
            return tiles

        def load_scales(kt):
            """The combined scale for each of this lane's accumulators.

            With the transposed accumulator a lane owns exactly one output row
            per M subtile, so its A scale is a single value rather than one per
            register. Every column of the tile falls in one N scale block, so B
            contributes one scalar for the whole tile. The product is therefore
            constant across an accumulator, and the fold is 16 FMAs against a
            broadcast rather than 16 separate multipliers.
            """
            sb_idx = kt * fx.Index(scale_n) + (by_n // fx.Index(SCALE_BLOCK))
            sb = fx.Float32(
                buffer_ops.buffer_load(sb_rsrc, sb_idx, vec_width=1, dtype=T.f32)
            )
            out = []
            for ai in range_constexpr(acc_m):
                row = bx_m + wave_m0 + fx.Index(ai * MFMA_M) + lane_lo
                sa = fx.Float32(
                    buffer_ops.buffer_load(
                        sa_rsrc, kt * m_in + row, vec_width=1, dtype=T.f32
                    )
                )
                out.append(sa * sb)
            return out

        # ---- K loop ----------------------------------------------------------
        n_acc = acc_m * acc_n

        def split_state(state):
            vals = list(state) if isinstance(state, (list, tuple)) else [state]
            return [Vector(v, (ACC_REGS,), fx.Float32) for v in vals]

        def fold(cur, tiles, scales):
            out = []
            for ai in range_constexpr(acc_m):
                for aj in range_constexpr(acc_n):
                    i = ai * acc_n + aj
                    tile = Vector(tiles[i], (ACC_REGS,), fx.Float32)
                    # Emitted as an explicit fused multiply-add. Written as
                    # `acc + tile * scale` the two arith ops do not contract --
                    # contraction changes rounding, so the compiler will not do
                    # it unasked -- and the fold costs a separate v_pk_mul_f32
                    # and v_pk_add_f32 per pair instead of one v_pk_fma_f32.
                    vals = [
                        fx.Float32(
                            math_dialect.fma(
                                fx.Float32(tile[r]).ir_value(),
                                scales[ai].ir_value(),
                                fx.Float32(cur[i][r]).ir_value(),
                            )
                        )
                        for r in range_constexpr(ACC_REGS)
                    ]
                    out.append(Vector.from_elements(vals, fx.Float32))
            return out

        accs = [
            Vector(zero_acc, (ACC_REGS,), fx.Float32) for _ in range_constexpr(n_acc)
        ]

        # One tile deep: tile kt is read out of LDS while tile kt+1 is in flight
        # towards the other slot. Both operands go straight from global memory to
        # LDS, so the accumulators are the only value the loop carries.
        #
        # The order of the four steps in the body is forced, and each pairing
        # matters:
        #
        #   wait    publishes tile kt, whose transfer was issued last iteration.
        #           Nothing newer has been issued at this point, so vmcnt(0) here
        #           costs nothing and there is no count to get wrong -- which the
        #           alias metadata makes a correctness property, not just a
        #           performance one.
        #   barrier must follow the wait, or a wave crosses it with transfers
        #           still in flight and its partners read bytes it has not
        #           written; s_barrier orders execution, not memory. It also
        #           releases the slot the previous iteration finished reading,
        #           which the transfer below is about to overwrite.
        #   DMA     must follow the barrier, or it overwrites a slot the other
        #           waves are still reading.
        #   compute leaves that transfer in flight underneath this tile's MFMAs,
        #           which is the whole purpose of the arrangement.
        #
        # The scale loads sit with the fold that consumes them. Hoisting them
        # above the transfer, which would leave the fold waiting behind fewer
        # outstanding loads, measures slower at every tile tried.
        dma_stage(fx.Index(0), fx.Index(0))

        # The loop stops one tile short and the last is peeled below, so that the
        # transfer here is unconditional. Guarding it instead would put it in a
        # block of its own, where it can no longer be interleaved with the MFMAs
        # it is supposed to run underneath.
        if num_k_tiles > 1:
            for _kt, _st in fx.range(0, num_k_tiles - 1, 1, init=list(accs)):
                _cur = split_state(_st)
                _now = _kt % fx.Index(STAGES)
                _next_slot = (_kt + fx.Index(1)) % fx.Index(STAGES)

                rocdl.s_waitcnt(0)
                gpu.barrier()

                dma_stage(_kt + fx.Index(1), _next_slot)

                _tiles = tile_product(a_buf(_now), b_buf(_now))
                _res = yield fold(_cur, _tiles, load_scales(_kt))

            accs = split_state(_res)

        # The peeled tile. Its operands are already in flight from the last
        # iteration, and there is no successor to fetch, so this is the loop body
        # without the transfer.
        _last = num_k_tiles - 1
        _last_slot = _last % STAGES
        rocdl.s_waitcnt(0)
        gpu.barrier()
        accs = fold(
            accs,
            tile_product(a_buf(fx.Index(_last_slot)), b_buf(fx.Index(_last_slot))),
            load_scales(fx.Index(_last)),
        )

        # ---- epilogue --------------------------------------------------------
        # The transposed accumulator leaves a lane holding one output row and
        # four runs of four adjacent columns, with lanes L and L+32 holding
        # complementary halves of the same eight columns. Stored as-is that caps
        # the store at 8 bytes per lane, which fills only an eighth of a cache
        # line per transaction -- fine when the operands dominate, expensive when
        # the output does.
        #
        # `permlane32_swap(a, b)` gathers both half-wave halves of one operand
        # into one half-wave: low lanes receive a[L] and a[L+32], high lanes
        # receive b[L-32] and b[L]. Feeding it a pair of column groups therefore
        # gives the low half all eight columns of the even group and the high
        # half all eight of the odd one, both for the row the lane already owned.
        # Two swaps per pair of groups turn four 8-byte stores into two 16-byte
        # ones.
        pair_ty = ir.Type.parse("!llvm.struct<(i32, i32)>")

        def group_dwords(acc, g):
            """One column group's four values, as BF16 packed into two i32."""
            packed = Vector.from_elements(
                [
                    fx.BFloat16(arith.trunc_f(T.bf16, fx.Float32(acc[g * 4 + e])))
                    for e in range_constexpr(4)
                ],
                fx.BFloat16,
            ).bitcast(fx.Int32)
            return fx.Int32(packed[0]), fx.Int32(packed[1])

        for ai in range_constexpr(acc_m):
            row = bx_m + wave_m0 + fx.Index(ai * MFMA_M) + lane_lo
            for aj in range_constexpr(acc_n):
                acc = accs[ai * acc_n + aj]
                for pair in range_constexpr(ACC_REGS // 8):
                    even_lo, even_hi = group_dwords(acc, 2 * pair)
                    odd_lo, odd_hi = group_dwords(acc, 2 * pair + 1)
                    sw_lo = rocdl.permlane32_swap(
                        pair_ty, _raw(even_lo), _raw(odd_lo), False, False
                    )
                    sw_hi = rocdl.permlane32_swap(
                        pair_ty, _raw(even_hi), _raw(odd_hi), False, False
                    )
                    # Columns ascend with the address, so the dwords interleave:
                    # this lane's own pair, then its partner's.
                    quad = Vector.from_elements(
                        [
                            fx.Int32(llvm.extractvalue(T.i32, sw_lo, [0])),
                            fx.Int32(llvm.extractvalue(T.i32, sw_hi, [0])),
                            fx.Int32(llvm.extractvalue(T.i32, sw_lo, [1])),
                            fx.Int32(llvm.extractvalue(T.i32, sw_hi, [1])),
                        ],
                        fx.Int32,
                    )
                    # Low lanes take the even group, high lanes the odd one.
                    col = (
                        by_n
                        + wave_n0
                        + fx.Index(aj * MFMA_M)
                        + (fx.Index(2 * pair) + lane_hi) * fx.Index(8)
                    )
                    buffer_ops.buffer_store(
                        quad,
                        d_rsrc,
                        (row * n_in + col) * fx.Index(2),
                        offset_is_bytes=True,
                    )

    @flyc.jit
    def launch_groupwise_wide_gemm(
        arg_d: fx.Tensor,
        arg_a: fx.Tensor,
        arg_b: fx.Tensor,
        arg_scale_a: fx.Tensor,
        arg_scale_b: fx.Tensor,
        i32_m: fx.Int32,
        i32_n: fx.Int32,
        i32_k: fx.Int32,
        stream: fx.Stream,
    ):
        allocator.finalized = False
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            allocator.finalize()

        m_in = fx.Index(i32_m)
        gx = fx.Index(i32_n) // fx.Index(tile_n)
        gy = (m_in + fx.Index(tile_m - 1)) // fx.Index(tile_m)

        launcher = groupwise_wide_gemm_kernel(
            arg_d, arg_a, arg_b, arg_scale_a, arg_scale_b, i32_m, i32_n, i32_k
        )
        if waves_per_eu is not None and int(waves_per_eu) >= 1:
            for op in ctx.gpu_module_body.operations:
                if hasattr(op, "attributes") and op.OPERATION_NAME == "gpu.func":
                    op.attributes["rocdl.waves_per_eu"] = ir.IntegerAttr.get(
                        T.i32, int(waves_per_eu)
                    )
        launcher.launch(
            grid=(gx, gy, fx.Index(1)), block=(total_threads, 1, 1), stream=stream
        )

    return launch_groupwise_wide_gemm
