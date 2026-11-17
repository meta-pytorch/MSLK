# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# SPDX-License-Identifier: Apache-2.0

"""FP8 rowwise-scaled GEMM on plain (row-major) weights, for gfx942 and gfx950.

Computes, for each batch ``b`` of a ``[B, M, K] x [B, N, K]`` problem,

  D[b, m, n] = (sum_k A[b, m, k] * B[b, n, k]) * scale_a[b, m] * scale_b[b, n]

with A and B FP8 E4M3, the scales FP32, and D BF16 or FP16. This is the
contract of ``mslk::f8f8bf16_rowwise`` and its siblings, and the kernel takes
every shape those ops accept: M, N and K need not divide a tile, nor even a
load or a store.

The schedule follows ``fp8_groupwise_wide_gemm``: the wide
``v_mfma_scale_f32_32x32x64_f8f6f4`` with a neutral E8M0 scale, a
``waves_m x waves_n`` wave grid, both operands staged through LDS with an XOR
swizzle, and ``permlane32_swap`` to widen the stores. Rowwise scales do not
depend on K, so the accumulators run across the whole contraction and both
scales are applied once, in the epilogue.

gfx942 has neither that MFMA, nor ``permlane32_swap``, nor 16-byte transfers
straight into LDS. There each 64 of K is four ``v_mfma_f32_32x32x16_fp8_fp8``
issues, the operands are staged through registers, and stores are 8 bytes
wide. The 32x32x16 instruction lays its accumulator out exactly as the wide
one does, and a lane's 32-byte fragment of 64 of K splits into the four 8-byte
operands in order: each issue then pairs a different subset of K than the
instruction's own numbering says, but the same subset for A and B, which a dot
product does not distinguish. So the LDS layout, the fragment reads and the
epilogue are shared.

Shapes the tiles do not divide:

  * Every descriptor is based at its block's tile and sized to the rows (and,
    for B, the columns) the tile actually owns. A load past them reads zero and
    a store past them is dropped, so M and N tails need nothing more on the
    load side, and an operand larger than one descriptor can address (4 GiB)
    is never addressed through one.
  * A K tail is zeroed on load: a chunk past K is redirected out of range,
    which the hardware fills with zeros. Both operands must be zero there, as
    FP8 bytes past K can hold NaN encodings.
  * Where K is not a multiple of 16, a row does not start on a 16-byte
    boundary, so the operands are loaded at the widest width K allows and
    assembled in registers, then written to LDS, rather than moved straight to
    LDS.
  * Where N is not a multiple of 8 (of 4 on gfx942), a row of D does not
    start on a store boundary either, and each element is stored on its own.

Fragment layouts for the wide MFMA:

  A operand   m = lane % 32,  k = (lane // 32) * 32 + byte
  B operand   n = lane % 32,  k = (lane // 32) * 32 + byte

With B fed as the instruction's first operand the accumulator comes out
transposed: a lane holds one output row ``m = lane % 32`` and four runs of four
adjacent columns, ``n = 8 * (reg // 4) + 4 * (lane // 32) + reg % 4``.
"""

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.expr import arith, const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import T, Vector
from flydsl.expr.utils.arith import _to_raw as _raw
from flydsl.runtime.device import get_rocm_arch as get_hip_arch
from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr
from mslk.flydsl import buffer_ops
from mslk.flydsl.common import lds_capacity_bytes
from mslk.flydsl.kernels.common.kernels_common import default_f8_type
from mslk.flydsl.kernels.gemm.fp8_wide_mfma_common import (
    ACC_REGS,
    LdsAccess,
    LOAD_BYTES,
    MFMA_K,
    MFMA_M,
    MFMA_OPERAND_BYTES,
    swap_quad,
    WAVE,
    wide_mfma,
)

# Accumulator tiles one wave may own; see the check in the compiler.
MAX_ACC_TILES = 8

# A buffer offset past every descriptor's extent: a load there reads zero and a
# store is dropped. Each descriptor covers one tile, and the host keeps tiles
# below this.
OUT_OF_RANGE = 0x7FFFFFFF

# The widest count s_waitcnt can wait for on vmcnt.
VMCNT_MAX = 63


def _waitcnt_vm(count: int) -> int:
    """s_waitcnt immediate waiting for vmcnt <= count and every LDS access.

    On gfx9 vmcnt is split across bits [3:0] and [15:14]; expcnt, in [6:4], is
    left at its maximum, which waits for nothing.
    """
    return (count & 0xF) | ((count >> 4) << 14) | (0x7 << 4)


def _load_width(k: int) -> int:
    """Widest load, in bytes, that every row of a K-byte-wide operand starts on."""
    width = LOAD_BYTES
    while k % width:
        width //= 2
    return width


# Holds a whole tuning sweep's launchers for one shape, so a sweep compiles each
# config once.
@functools.lru_cache(maxsize=1024)
def compile_fp8_rowwise_gemm(
    *,
    n: int,
    k: int,
    tile_m: int = 128,
    tile_n: int = 128,
    tile_k: int = 128,
    waves_m: int = 2,
    waves_n: int = 2,
    stages: int = 2,
    waves_per_eu: int | None = None,
    out_dtype: str = "bf16",
):
    """Compile the kernel for one N, K and tile config; return the launcher.

    The block is ``waves_m * waves_n`` waves, and wave ``(wm, wn)`` owns a
    ``(tile_m / waves_m) x (tile_n / waves_n)`` slab of the output tile. M and
    the batch count are runtime arguments of the launcher.

    ``stages`` is the number of LDS buffers per operand, so ``stages - 1`` K
    tiles are in flight while one is consumed. With few rows a block's time is
    the latency of its weight loads, and more of them in flight is what hides
    it. Only the direct-to-LDS path pipelines past two.
    """
    if out_dtype not in ("bf16", "f16"):
        raise ValueError(f"out_dtype must be 'bf16' or 'f16', got {out_dtype!r}")
    if n <= 0 or k <= 0:
        raise ValueError(f"n ({n}) and k ({k}) must be positive")
    if tile_m % (waves_m * MFMA_M) or tile_n % (waves_n * MFMA_M):
        raise ValueError(
            f"tile {tile_m}x{tile_n} does not divide into {waves_m}x{waves_n} "
            f"waves of {MFMA_M}x{MFMA_M} MFMA tiles"
        )
    if tile_k % MFMA_K or (tile_k // LOAD_BYTES) & (tile_k // LOAD_BYTES - 1):
        raise ValueError(
            f"tile_k ({tile_k}) must be a multiple of {MFMA_K} and a power of two"
        )

    num_waves = waves_m * waves_n
    total_threads = num_waves * WAVE
    wave_tile_m = tile_m // waves_m
    wave_tile_n = tile_n // waves_n
    acc_m = wave_tile_m // MFMA_M
    acc_n = wave_tile_n // MFMA_M
    # Each accumulator tile is 16 registers a lane, held across all of K. Past
    # eight the accumulators alone take half the register file, and the
    # compiler spills.
    if acc_m * acc_n > MAX_ACC_TILES:
        raise ValueError(
            f"a {wave_tile_m}x{wave_tile_n} slab per wave needs {acc_m * acc_n} "
            f"accumulator tiles, more than {MAX_ACC_TILES}; use more waves"
        )
    k_steps = tile_k // MFMA_K
    num_k_tiles = -(-k // tile_k)

    # Each thread stages whole 16-byte chunks of each operand's tile.
    a_bytes_per_thread = tile_m * tile_k // total_threads
    b_bytes_per_thread = tile_n * tile_k // total_threads
    if a_bytes_per_thread % LOAD_BYTES or b_bytes_per_thread % LOAD_BYTES:
        raise ValueError(
            f"tile {tile_m}x{tile_n}x{tile_k} does not split into whole "
            f"{LOAD_BYTES}-byte chunks across {total_threads} threads"
        )
    num_a_loads = a_bytes_per_thread // LOAD_BYTES
    num_b_loads = b_bytes_per_thread // LOAD_BYTES

    # Every offset inside a descriptor is 32-bit and the out-of-range sentinel
    # has to stay past it.
    for what, nbytes in (
        ("A", tile_m * k),
        ("B", tile_n * k),
        ("D", tile_m * n * 2),
    ):
        if nbytes >= OUT_OF_RANGE:
            raise ValueError(
                f"one tile of {what} spans {nbytes} bytes, past what a buffer "
                f"descriptor addresses; use a smaller tile (n={n}, k={k})"
            )

    k_tail = k % tile_k != 0
    load_width = _load_width(k)
    gpu_arch = str(get_hip_arch())
    cdna4 = gpu_arch.startswith("gfx95")
    if not (cdna4 or gpu_arch.startswith("gfx942")):
        raise ValueError(f"supported on gfx942 and gfx950; arch is {gpu_arch}")
    # Direct-to-LDS moves whole 16-byte chunks, which only line up with the rows
    # when K is a multiple of 16, and gfx942 moves at most 4 bytes that way.
    use_dma = cdna4 and load_width == LOAD_BYTES
    # Stores are 16 bytes on gfx950 and 8 on gfx942, and a row of D starts on
    # such a boundary only when N is a multiple of 8, or of 4.
    store_cols = 8 if cdna4 else 4
    wide_store = n % store_cols == 0
    n_tail = n % tile_n != 0
    # A 16-byte read of four scales needs the same of a row of scale_b.
    wide_scale_b = n % 4 == 0

    if stages < 2 or (stages > 2 and not use_dma):
        raise ValueError(
            f"stages ({stages}) must be 2, or more where K is a multiple of 16"
        )
    # Transfers each thread issues per K tile, which is what the wait counts.
    loads_per_tile = num_a_loads + num_b_loads
    if (stages - 2) * loads_per_tile > VMCNT_MAX:
        raise ValueError(
            f"{stages} stages of {loads_per_tile} loads each leave more in flight "
            f"than s_waitcnt can count ({VMCNT_MAX})"
        )

    lds_a_elems = tile_m * tile_k
    lds_b_elems = tile_n * tile_k
    lds_bytes = stages * (lds_a_elems + lds_b_elems)
    capacity = lds_capacity_bytes(gpu_arch)
    if lds_bytes > capacity:
        raise ValueError(
            f"tile {tile_m}x{tile_n}x{tile_k} in {stages} stages needs {lds_bytes} B "
            f"of LDS, over the {capacity} B on {gpu_arch}"
        )
    allocator = SmemAllocator(None, arch=gpu_arch, global_sym_name="smem_rw_plain")
    lds_off = allocator._align(allocator.ptr, 16)
    allocator.ptr = lds_off + lds_bytes

    _wpe = f"_wpe{int(waves_per_eu)}" if waves_per_eu else ""
    _wpe += f"_s{stages}" if stages != 2 else ""
    module_name = (
        f"rw_plain_{out_dtype}_n{n}_k{k}_t{tile_m}x{tile_n}x{tile_k}"
        f"_w{waves_m}x{waves_n}{_wpe}"
    )

    @flyc.kernel(name=module_name, known_block_size=[total_threads, 1, 1])
    def fp8_rowwise_gemm_kernel(
        arg_d: fx.Tensor,
        arg_a: fx.Tensor,
        arg_b: fx.Tensor,
        arg_scale_a: fx.Tensor,
        arg_scale_b: fx.Tensor,
        i32_m: fx.Int32,
    ):
        v_acc_ty = Vector.make_type(ACC_REGS, fx.Float32)
        out_ty = fx.BFloat16 if out_dtype == "bf16" else fx.Float16
        out_ir = T.bf16 if out_dtype == "bf16" else T.f16

        m_in = fx.Index(i32_m)
        n_c = fx.Index(n)
        k_c = fx.Index(k)

        tx = gpu.thread_id("x")
        bn = gpu.block_id("x")
        bm = gpu.block_id("y")
        batch = gpu.block_id("z")

        bx_m = bm * fx.Index(tile_m)
        by_n = bn * fx.Index(tile_n)

        wave_id = tx // fx.Index(WAVE)
        lane = tx % fx.Index(WAVE)
        wm = wave_id // fx.Index(waves_n)
        wn = wave_id % fx.Index(waves_n)
        lane_lo = lane % fx.Index(MFMA_M)
        lane_hi = lane // fx.Index(MFMA_M)
        wave_m0 = wm * fx.Index(wave_tile_m)
        wave_n0 = wn * fx.Index(wave_tile_n)

        def clamp(left, cap):
            return fx.Index(arith.select(left < fx.Index(cap), left, fx.Index(cap)))

        # Rows of A and of B (columns of D) this block's tile actually owns.
        rows_a = clamp(m_in - bx_m, tile_m)
        rows_b = clamp(n_c - by_n, tile_n)
        # First row of the tile across the whole batch, for the 64-bit bases.
        row_a0 = batch * m_in + bx_m
        row_b0 = batch * n_c + by_n

        def tile_resource(arg, nbytes, base):
            return buffer_ops.create_buffer_resource(
                arg, max_size=False, num_records_bytes=nbytes, base_byte_offset=base
            )

        a_rsrc = tile_resource(arg_a, rows_a * k_c, row_a0 * k_c)
        b_rsrc = tile_resource(arg_b, rows_b * k_c, row_b0 * k_c)
        # D's descriptor covers whole rows, so a column past N stays inside it
        # and the epilogue masks it itself.
        d_rsrc = tile_resource(
            arg_d, rows_a * n_c * fx.Index(2), row_a0 * n_c * fx.Index(2)
        )
        sa_rsrc = tile_resource(arg_scale_a, rows_a * fx.Index(4), row_a0 * fx.Index(4))
        sb_rsrc = tile_resource(arg_scale_b, rows_b * fx.Index(4), row_b0 * fx.Index(4))

        base_ptr = allocator.get_base()
        lds = SmemPtr(base_ptr, lds_off, default_f8_type(), shape=(lds_bytes,)).get()

        def a_buf(slot):
            return slot * fx.Index(lds_a_elems)

        def b_buf(slot):
            return fx.Index(stages * lds_a_elems) + slot * fx.Index(lds_b_elems)

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

        # The direct-to-LDS transfers and the fragment reads carry alias scopes,
        # which let the backend overlap them; the s_waitcnt at the top of each
        # K tile is then what orders them, so its count has to be exact.
        lds_io = LdsAccess(lds, "rw_plain", tile_k)

        # ---- staging: global -> LDS ----------------------------------------
        def dma_tile(rsrc, coords, kt, lds_base):
            """Move a tile straight into LDS; a chunk past K is sent out of
            range and so lands as zeros."""

            def src_offset(i):
                row, col = coords[i]
                src_col = kt * fx.Index(tile_k) + lds_io.swizzle(row, col)
                byte_off = row * k_c + src_col
                if const_expr(k_tail):
                    byte_off = fx.Index(
                        arith.select(src_col < k_c, byte_off, fx.Index(OUT_OF_RANGE))
                    )
                return byte_off

            lds_io.dma_tile(
                rsrc, len(coords), src_offset, lds_base, wave_id, total_threads
            )

        # Loads of `load_width` bytes, as the element type of that width.
        piece_ty = {1: T.i8, 2: T.i16}.get(load_width, T.i32)
        piece_vec = {8: 2, 16: 4}.get(load_width, 1)
        piece_elem = {1: 1, 2: 2}.get(load_width, 4)
        pieces = LOAD_BYTES // load_width

        def load_chunks(rsrc, coords, kt):
            """Read a thread's chunks of a tile into registers, zero past K."""
            chunks = []
            for i in range_constexpr(len(coords)):
                row, col = coords[i]
                parts = []
                for j in range_constexpr(pieces):
                    kk = kt * fx.Index(tile_k) + col + fx.Index(j * load_width)
                    off = (row * k_c + kk) // fx.Index(piece_elem)
                    parts.append(
                        buffer_ops.buffer_load(
                            rsrc,
                            off,
                            vec_width=piece_vec,
                            dtype=piece_ty,
                            mask=(kk < k_c) if k_tail else None,
                        )
                    )
                if const_expr(pieces == 1):
                    chunks.append(Vector(parts[0]))
                else:
                    elems = parts
                    if const_expr(piece_vec > 1):
                        elems = [
                            Vector(p)[e]
                            for p in parts
                            for e in range_constexpr(piece_vec)
                        ]
                    raw_ty = {1: fx.Int8, 2: fx.Int16}.get(load_width, fx.Int32)
                    chunks.append(
                        Vector.from_elements(
                            [raw_ty(e) for e in elems], raw_ty
                        ).bitcast(fx.Int32)
                    )
            return chunks

        def store_chunks(chunks, coords, lds_base):
            for i in range_constexpr(len(coords)):
                row, col = coords[i]
                llvm.StoreOp(
                    _raw(chunks[i]),
                    lds_io.ptr(lds_io.index(row, col, lds_base)),
                    alignment=16,
                )

        def stage_issue(kt, slot):
            """Start moving tile ``kt``; return what `stage_finish` needs, if any."""
            pending = None
            if const_expr(use_dma):
                dma_tile(a_rsrc, a_coords, kt, a_buf(slot))
                dma_tile(b_rsrc, b_coords, kt, b_buf(slot))
            else:
                pending = (
                    load_chunks(a_rsrc, a_coords, kt),
                    load_chunks(b_rsrc, b_coords, kt),
                )
            return pending

        def stage_finish(pending, slot):
            if const_expr(not use_dma):
                store_chunks(pending[0], a_coords, a_buf(slot))
                store_chunks(pending[1], b_coords, b_buf(slot))

        # ---- fragment reads and MFMA ----------------------------------------
        def read_frag(row, lds_base, ks):
            return lds_io.read_frag(row, lds_base, ks, lane_hi)

        def mfma(a_op, b_op, c_in):
            """Contract 64 of K: one wide issue, or four 32x32x16 ones."""
            if const_expr(cdna4):
                acc = wide_mfma(v_acc_ty, a_op, b_op, c_in)
            else:
                # See the module docstring for why the slices pair up.
                acc = _raw(c_in)
                for j in range_constexpr(MFMA_OPERAND_BYTES // 8):
                    a_j = Vector(a_op).shuffle(Vector(a_op), [2 * j, 2 * j + 1])
                    b_j = Vector(b_op).shuffle(Vector(b_op), [2 * j, 2 * j + 1])
                    # Operands: a, b, c, cbsz, abid, blgp (no B permutation).
                    acc = rocdl.mfma_f32_32x32x16_fp8_fp8_(
                        v_acc_ty,
                        _raw(a_j.bitcast(fx.Int64)[0]),
                        _raw(b_j.bitcast(fx.Int64)[0]),
                        acc,
                        0,
                        0,
                        ir.Attribute.parse("#rocdl<mfma_perm_b none>"),
                    )
            return acc

        def tile_product(accs, a_base, b_base):
            """Contract one K tile into the running accumulators."""
            out = [a.ir_value() for a in accs]
            for ks in range_constexpr(k_steps):
                a_frags = [
                    read_frag(wave_m0 + fx.Index(ai * MFMA_M) + lane_lo, a_base, ks)
                    for ai in range_constexpr(acc_m)
                ]
                b_frags = [
                    read_frag(wave_n0 + fx.Index(aj * MFMA_M) + lane_lo, b_base, ks)
                    for aj in range_constexpr(acc_n)
                ]
                for ai in range_constexpr(acc_m):
                    for aj in range_constexpr(acc_n):
                        idx = ai * acc_n + aj
                        # B first, which transposes the accumulator; see the
                        # module docstring.
                        out[idx] = mfma(b_frags[aj], a_frags[ai], out[idx])
            return [Vector(o, (ACC_REGS,), fx.Float32) for o in out]

        # ---- K loop ----------------------------------------------------------
        def sync_tile(in_flight):
            """Wait until at most ``in_flight`` transfers are outstanding, then
            synchronise the block.

            The barrier is issued as inline assembly: the backend drains every
            outstanding transfer ahead of an ``s_barrier`` it can see, which
            would reduce the pipeline to one tile however deep it was built.
            The wait itself stays an intrinsic, so the backend still sees it.
            A one-wave block has nothing to synchronise.
            """
            rocdl.s_waitcnt(_waitcnt_vm(in_flight))
            if const_expr(num_waves > 1):
                llvm.inline_asm(
                    res=None,
                    operands_=[],
                    asm_string="s_barrier",
                    constraints="",
                    has_side_effects=True,
                )

        n_acc = acc_m * acc_n

        def split_state(state):
            vals = list(state) if isinstance(state, (list, tuple)) else [state]
            return [Vector(v, (ACC_REGS,), fx.Float32) for v in vals]

        zero_acc = Vector.from_elements(
            [fx.Float32(0.0) for _ in range_constexpr(ACC_REGS)], fx.Float32
        )
        accs = [zero_acc for _ in range_constexpr(n_acc)]

        # ``stages - 1`` tiles deep: tile kt is read out of LDS while the next
        # ``stages - 2`` are in flight and the one after is issued into the
        # slot tile kt-1 has just freed. Without the direct path the depth is
        # one, and the tile after is read into registers before tile kt's MFMAs
        # and written to LDS after them, which hides the same latency.
        #
        # Each tile starts with the same three steps as in
        # `fp8_groupwise_wide_gemm`: wait for tile kt to land, barrier, issue;
        # see `sync_tile` for why the barrier is not the intrinsic.
        # The wait leaves in flight exactly the transfers issued after tile
        # kt's, which the alias metadata makes a correctness property: one
        # count too many and the MFMAs read LDS before it is written. So the
        # loop issues on every iteration, and the last ``stages - 1`` tiles,
        # where there is nothing left to issue and the count falls, are
        # unrolled below it.
        depth = stages - 1
        for t in range_constexpr(min(depth, num_k_tiles)):
            stage_finish(stage_issue(fx.Index(t), fx.Index(t)), fx.Index(t))

        steady = num_k_tiles - depth
        if const_expr(steady > 0):
            for _kt, _st in fx.range(0, steady, 1, init=list(accs)):
                _cur = split_state(_st)
                _now = _kt % fx.Index(stages)
                _ahead = _kt + fx.Index(depth)
                _next_slot = _ahead % fx.Index(stages)

                sync_tile((depth - 1) * loads_per_tile)

                _pending = stage_issue(_ahead, _next_slot)
                _res_accs = tile_product(_cur, a_buf(_now), b_buf(_now))
                stage_finish(_pending, _next_slot)
                _res = yield _res_accs

            accs = split_state(_res)

        for t in range_constexpr(max(steady, 0), num_k_tiles):
            sync_tile((num_k_tiles - 1 - t) * loads_per_tile)
            slot = fx.Index(t % stages)
            accs = tile_product(accs, a_buf(slot), b_buf(slot))

        # ---- epilogue --------------------------------------------------------
        def scale_b_run(col):
            """scale_b for the four adjacent columns from local column ``col``."""
            if const_expr(wide_scale_b):
                run = Vector(
                    buffer_ops.buffer_load(sb_rsrc, col, vec_width=4, dtype=T.f32)
                )
            else:
                run = [
                    fx.Float32(
                        buffer_ops.buffer_load(
                            sb_rsrc, col + fx.Index(e), vec_width=1, dtype=T.f32
                        )
                    )
                    for e in range_constexpr(4)
                ]
            return run

        def col_valid(col):
            return (by_n + col) < n_c

        def store_wide(runs, row, col0):
            def dwords(g):
                packed = Vector.from_elements(runs[g], out_ty).bitcast(fx.Int32)
                return fx.Int32(packed[0]), fx.Int32(packed[1])

            for pair in range_constexpr(ACC_REGS // 8):
                # Two column groups gathered across the half-waves into one
                # 16-byte store per lane; see `fp8_wide_mfma_common.swap_quad`.
                quad = swap_quad(dwords(2 * pair), dwords(2 * pair + 1))
                # Low lanes take the even group, high lanes the odd one.
                col = col0 + (fx.Index(2 * pair) + lane_hi) * fx.Index(8)
                buffer_ops.buffer_store(
                    quad,
                    d_rsrc,
                    (row * n_c + by_n + col) * fx.Index(2),
                    mask=col_valid(col) if n_tail else None,
                    offset_is_bytes=True,
                )

        for ai in range_constexpr(acc_m):
            row = wave_m0 + fx.Index(ai * MFMA_M) + lane_lo
            sa = fx.Float32(
                buffer_ops.buffer_load(sa_rsrc, row, vec_width=1, dtype=T.f32)
            )
            for aj in range_constexpr(acc_n):
                acc = accs[ai * acc_n + aj]
                col0 = wave_n0 + fx.Index(aj * MFMA_M)
                # The scaled values of each run of four columns, in the output
                # type.
                runs = []
                for g in range_constexpr(ACC_REGS // 4):
                    col = col0 + fx.Index(8 * g) + lane_hi * fx.Index(4)
                    sb = scale_b_run(col)
                    runs.append(
                        [
                            out_ty(
                                arith.trunc_f(
                                    out_ir,
                                    fx.Float32(acc[4 * g + e]) * sa * fx.Float32(sb[e]),
                                )
                            )
                            for e in range_constexpr(4)
                        ]
                    )
                if const_expr(not wide_store):
                    for g in range_constexpr(ACC_REGS // 4):
                        col = col0 + fx.Index(8 * g) + lane_hi * fx.Index(4)
                        for e in range_constexpr(4):
                            c = col + fx.Index(e)
                            buffer_ops.buffer_store(
                                runs[g][e],
                                d_rsrc,
                                row * n_c + by_n + c,
                                mask=col_valid(c),
                            )
                elif const_expr(cdna4):
                    store_wide(runs, row, col0)
                else:
                    for g in range_constexpr(ACC_REGS // 4):
                        col = col0 + fx.Index(8 * g) + lane_hi * fx.Index(4)
                        buffer_ops.buffer_store(
                            Vector.from_elements(runs[g], out_ty).bitcast(fx.Int32),
                            d_rsrc,
                            (row * n_c + by_n + col) * fx.Index(2),
                            mask=col_valid(col) if n_tail else None,
                            offset_is_bytes=True,
                        )

    @flyc.jit
    def launch_fp8_rowwise_gemm(
        arg_d: fx.Tensor,
        arg_a: fx.Tensor,
        arg_b: fx.Tensor,
        arg_scale_a: fx.Tensor,
        arg_scale_b: fx.Tensor,
        i32_m: fx.Int32,
        i32_batch: fx.Int32,
        stream: fx.Stream,
    ):
        allocator.finalized = False
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            allocator.finalize()

        gx = fx.Index(-(-n // tile_n))
        gy = (fx.Index(i32_m) + fx.Index(tile_m - 1)) // fx.Index(tile_m)

        launcher = fp8_rowwise_gemm_kernel(
            arg_d, arg_a, arg_b, arg_scale_a, arg_scale_b, i32_m
        )
        if waves_per_eu is not None and int(waves_per_eu) >= 1:
            for op in ctx.gpu_module_body.operations:
                if hasattr(op, "attributes") and op.OPERATION_NAME == "gpu.func":
                    op.attributes["rocdl.waves_per_eu"] = ir.IntegerAttr.get(
                        T.i32, int(waves_per_eu)
                    )
        launcher.launch(
            grid=(gx, gy, fx.Index(i32_batch)),
            block=(total_threads, 1, 1),
            stream=stream,
        )

    return launch_fp8_rowwise_gemm
