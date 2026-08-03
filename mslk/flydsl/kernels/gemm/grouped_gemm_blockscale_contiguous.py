# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Contiguous Grouped FP8 GEMM kernel with block scaling.

Groups are concatenated along M with arbitrary (not tile-aligned) per-group row
counts, and the output is compact [M_total, N]. Each output M-tile belongs to
exactly one group, so a tile never spans a group boundary. The kernel resolves
the owning group for its M-tile from m_sizes, and rows at or beyond that group's
end are the partial-tile tail and are masked out of the store.

The `layout` argument selects how the group geometry is encoded; groups may also
occupy fixed per-group slabs rather than being packed. Only the resolution step
differs, so the loaders, the K loop and the epilogue are shared by all of them.

Scales are FP32 (software scaling) on all architectures.

Tensors:
  - A: [M_total, K] FP8 - concatenated rows from all groups
  - scale_a: FP32 per-token, per-128K scales, laid out as per-group blocks (as
    written by quantize_fp8_group with m_sizes): group g's block begins at
    m_start * scale_k, and within it element (local_m, k_block) sits at
    local_m + k_block * M_g. This is not a global [scale_k, M_total] transpose.
  - B: [num_groups, N, K] FP8 - one weight matrix per group, preshuffled
  - scale_b: [num_groups, scale_k, scale_n] FP32 - per-block scales
  - m_sizes: [num_groups] - the group geometry, whose encoding the `layout`
    argument selects: INT64 row counts, INT32 cumulative offsets, or unused
  - D: [M_total, N] BF16 - output

Block scaling granularity:
  - A: (1, 128) - per-token, per-128-K-elements
  - B: (128, 128) - per-128-N, per-128-K block
"""

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import scf
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.expr import (
    arith,
    buffer_ops,
    const_expr,
    gpu,
    range_constexpr,
    rocdl,
    vector,
)
from flydsl.expr.typing import T
from flydsl.runtime.device import get_rocm_arch as get_hip_arch
from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr
from mslk.flydsl.kernels.gemm.grouped_gemm_blockscale_common import (
    compute_compile_constants,
    compute_mfma_tiling,
    init_accumulators,
    make_a_tile_loaders,
    make_b_loader,
    make_b_tile_loaders,
    make_compute_tile,
    make_epilogue_writers,
    make_hot_loop_scheduler,
    make_k_tail_mask,
    make_kloop_plain,
    make_lds_b_loader,
    make_lds_loader,
    make_n_block_coords,
    make_pingpong_kloop,
    make_plain_b_tile,
    make_prefetch_scales,
    make_rowwise_scaler,
    out_mlir_for,
    setup_lds_allocation,
    setup_lds_allocation_plain,
    validate_lds_budget_plain,
    validate_lds_budget_preshuffle,
    validate_params,
)
from mslk.flydsl.kernels.mma.mfma_epilogues import mfma_epilog

# Supported encodings of the group geometry; see the ``layout`` argument below.
LAYOUTS = ("sizes", "offsets", "padded", "batched")


@functools.lru_cache(maxsize=128)
def compile_grouped_gemm_blockscale_contiguous(
    *,
    n: int,
    k: int,
    num_groups: int,
    tile_m: int = 128,
    tile_n: int = 128,
    tile_k: int = 128,
    scale_block_k: int = 128,
    scale_block_n: int = 128,
    out_dtype: str = "bf16",
    waves_per_eu: int | None = None,
    b_preshuffled: bool = True,
    blockscale: bool = False,
    layout: str = "sizes",
    k_padding: bool = False,
    n_padding: bool = False,
):
    """Compile grouped FP8 GEMM kernel and return the JIT launcher.

    Args:
        n: N dimension (output columns per group)
        k: K dimension (reduction dimension)
        num_groups: Number of groups (experts)
        tile_m: M tile size (default 128)
        tile_n: N tile size (default 128)
        tile_k: K tile size (default 128)
        scale_block_k: K-dimension scale block size (default 128)
        scale_block_n: N-dimension scale block size (default 128)
        out_dtype: Output data type ("bf16" or "f16")
        b_preshuffled: When True (default) B is expected pre-swizzled into the
            MFMA layout and loaded HBM->registers (no B LDS). When False, B is
            plain row-major [num_groups, N, K] and is staged HBM->LDS->registers
            like A. The two paths share the entire kernel body (tile-map group
            dispatch, scaling, wide-MFMA, CShuffle epilogue); only the B load
            stage and its LDS allocation differ.
        blockscale: Selects the scaling scheme, which sets the expected layout of
            scale_a / scale_b and where the scales are applied.
            False (default) is rowwise: scale_a is [M_total] and scale_b is
            [num_groups, N], one factor per row of A and per column of B, applied
            once in the epilogue. Tiles are then free of scale-block alignment,
            so tile_n may be smaller than scale_block_n.
            True is block scaling: scale_a is per-group [M_g, scale_k] blocks and
            scale_b is [num_groups, scale_k, scale_n], applied per scale block
            inside the K loop, which requires the tile to align to the blocks.
        layout: How the kernel learns which rows belong to which group. The
            encodings differ only in that resolution step; the loaders, the K
            loop and the epilogue are shared.
            "sizes" (default): rows of every group are packed into one [M_total,
            K] buffer and m_sizes is [num_groups] INT64 per-group row counts.
            "offsets": same packed buffer, but m_sizes is [num_groups] INT32
            cumulative row ends, i.e. the inclusive prefix sum of the sizes.
            "padded": each group owns a fixed slab of M_total/num_groups rows and
            m_sizes is [num_groups] INT64 counts of the rows per slab that hold
            real data; the rest is padding the epilogue must not write.
            "batched": fixed slabs as in "padded" but every row holds real
            data, so the row count is implied and m_sizes is never read. Rows
            still need masking, since a tile that overruns the slab would spill
            into the next group.

    Returns:
        JIT launcher function.
    """
    if layout not in LAYOUTS:
        raise ValueError(f"layout must be one of {LAYOUTS}, got {layout!r}")
    # Each group owns a fixed slab of rows, so the group is a grid axis and does
    # not have to be resolved from the row counts.
    slab_layout = layout in ("padded", "batched")
    # Only the batched layout implies its row counts; the rest read them from
    # the group metadata.
    reads_group_meta = layout != "batched"

    gpu_arch = get_hip_arch()
    # This FP8 kernel always uses the FP32 software-scaling path; the shared
    # helpers' hardware E8M0 microscaling path is not used here.
    _use_hw_scale = False
    # On gfx950 the SW path still uses the wide 16x16x128 MFMA with a neutral
    # E8M0 scale (no-op HW scaling) and applies FP32 scales in software; gfx942
    # lacks that instruction and falls back to the narrow 16x16x32 path.
    _is_gfx950 = str(gpu_arch).startswith("gfx95")

    _sym = "smem_grouped_gemm" if b_preshuffled else "smem_grouped_gemm_plain"
    allocator = SmemAllocator(None, arch=gpu_arch, global_sym_name=_sym)

    validate_params(
        n=n,
        k=k,
        tile_n=tile_n,
        tile_k=tile_k,
        blockscale=blockscale,
        k_padding=k_padding,
        n_padding=n_padding,
        scale_block_k=scale_block_k,
        scale_block_n=scale_block_n,
        out_dtype=out_dtype,
    )
    # Check the LDS budget before tracing: the compiler treats an overflow as a
    # hard error that kills the process, which an autotuner cannot skip. Capacity
    # is arch-dependent (64 KiB gfx942, 160 KiB gfx950).
    if b_preshuffled:
        # Preshuffled B goes HBM->registers; only A ping-pong / epilogue use LDS.
        validate_lds_budget_preshuffle(
            tile_m=tile_m, tile_n=tile_n, tile_k=tile_k, arch=gpu_arch
        )
    else:
        # Plain B needs its own LDS buffer alongside A.
        validate_lds_budget_plain(
            tile_m=tile_m, tile_n=tile_n, tile_k=tile_k, b_pingpong=False, arch=gpu_arch
        )
    out_mlir = out_mlir_for(out_dtype)

    _c = compute_compile_constants(
        n=n,
        k=k,
        tile_m=tile_m,
        tile_n=tile_n,
        tile_k=tile_k,
        scale_block_k=scale_block_k,
        scale_block_n=scale_block_n,
        k_padding=k_padding,
    )
    total_threads = _c.total_threads
    elem_bytes = _c.elem_bytes
    num_k_tiles = _c.num_k_tiles
    scale_k = _c.scale_k
    scale_n = _c.scale_n
    sb_per_tile = _c.sb_per_tile
    k_unroll = _c.k_unroll
    kpack_bytes = _c.kpack_bytes
    tile_k_bytes = _c.tile_k_bytes
    tile_k_dwords = _c.tile_k_dwords
    chunk_i32_a = _c.chunk_i32_a
    num_a_loads = _c.num_a_loads
    chunk_i32_b = _c.chunk_i32_b
    num_b_loads = _c.num_b_loads

    if b_preshuffled:
        lds_alloc_offset, lds_tile_elems = setup_lds_allocation(
            allocator=allocator,
            tile_m=tile_m,
            tile_k=tile_k,
            tile_n=tile_n,
            elem_bytes=elem_bytes,
        )
        lds_b_offset_elems = None
    else:
        lds_alloc_offset, lds_tile_elems, lds_b_offset_elems = (
            setup_lds_allocation_plain(
                allocator=allocator,
                tile_m=tile_m,
                tile_n=tile_n,
                tile_k=tile_k,
                elem_bytes=elem_bytes,
                b_pingpong=False,
            )
        )

    # Module name for caching
    _variant = "contiguous_pingpong" if b_preshuffled else "plain"
    _scaling = "blockscale" if blockscale else "rowwise"
    _kpad = "_kpad" if k_padding else ""
    _npad = "_npad" if n_padding else ""
    module_name = (
        f"grouped_gemm_{_scaling}_{layout}_{_variant}_{out_dtype}"
        f"_n{n}_k{k}_g{num_groups}"
        f"_t{tile_m}x{tile_n}x{tile_k}{_kpad}{_npad}"
    ).replace("-", "_")

    @flyc.kernel(name=module_name)
    def grouped_gemm_blockscale_contiguous_kernel(
        arg_d: fx.Tensor,
        arg_a: fx.Tensor,
        arg_b: fx.Tensor,
        arg_scale_a: fx.Tensor,
        arg_scale_b: fx.Tensor,
        arg_m_sizes: fx.Tensor,
        i32_m: fx.Int32,
        i32_n: fx.Int32,
        i32_k: fx.Int32,
        i32_num_groups: fx.Int32,
    ):
        # Convert runtime parameters to index type
        m_in = fx.Index(i32_m)
        n_in = fx.Index(i32_n)
        k_in = fx.Index(i32_k)
        num_groups_in = fx.Index(i32_num_groups)

        # Thread and block IDs
        tx = gpu.thread_id("x")
        by = gpu.block_id("x")  # N-block index
        bx = gpu.block_id("y")  # M-tile index (into the per-tile dispatch map)
        bz = gpu.block_id("z")  # group index; carries the group in the padded layout

        # N-block position; bx_m (global row base) is loaded from the tile map below.
        by_n = by * fx.Index(tile_n)

        # Wave/lane decomposition (256 threads = 4 waves x 64 lanes)
        layout_wave_lane = fx.make_layout((4, 64), stride=(64, 1))
        coord_wave_lane = fx.idx2crd(fx.Int32(tx), layout_wave_lane)
        wave_id = fx.get(coord_wave_lane, 0)
        lane_id = fx.get(coord_wave_lane, 1)

        # Lane decomposition for MFMA (lane_id -> lane_div_16, lane_mod_16)
        layout_lane16 = fx.make_layout((4, 16), stride=(16, 1))
        coord_lane16 = fx.idx2crd(fx.Int32(lane_id), layout_lane16)
        lane_div_16 = fx.get(coord_lane16, 0)
        lane_mod_16 = fx.get(coord_lane16, 1)

        # LDS setup: ping-pong A buffers (preshuffle) or A ping-pong + single B
        # buffer (plain). B LDS is only needed for the plain path.
        base_ptr = allocator.get_base()
        lds_a = SmemPtr(
            base_ptr, lds_alloc_offset, T.f8, shape=(2 * tile_m * tile_k,)
        ).get()
        lds_stride = tile_k
        layout_lds = fx.make_layout((tile_m, tile_k), stride=(lds_stride, 1))
        lds_base_pong = fx.Index(0)
        lds_base_ping = fx.Index(lds_tile_elems)

        if const_expr(not b_preshuffled):
            # Plain-B LDS buffer, placed just past the A ping-pong region.
            lds_b = SmemPtr(
                base_ptr,
                lds_alloc_offset,
                T.f8,
                shape=((lds_b_offset_elems + tile_n * tile_k),),
            ).get()
            layout_lds_b = fx.make_layout((tile_n, tile_k), stride=(tile_k, 1))
            lds_base_b = fx.Index(lds_b_offset_elems)

        # CShuffle epilogue LDS (aliased from same base, out-dtype element type)
        lds_out = SmemPtr(
            base_ptr, lds_alloc_offset, out_mlir(), shape=(tile_m * tile_n,)
        ).get()

        # Buffer resources
        a_nbytes = m_in * k_in
        a_rsrc = buffer_ops.create_buffer_resource(
            arg_a, max_size=False, num_records_bytes=a_nbytes
        )

        b_nbytes = num_groups_in * n_in * k_in
        b_rsrc = buffer_ops.create_buffer_resource(
            arg_b, max_size=False, num_records_bytes=b_nbytes
        )

        d_nbytes = m_in * n_in * fx.Index(2)  # bf16/f16 = 2 bytes
        d_rsrc = buffer_ops.create_buffer_resource(
            arg_d, max_size=False, num_records_bytes=d_nbytes
        )

        # Scale buffers — gfx950 HW E8M0 path consumes int8 (one byte/scale,
        # pre-packed on host); gfx942 SW path consumes f32.
        scale_byte_size = 1 if _use_hw_scale else 4

        if const_expr(blockscale):
            # scale_a: per-group [M_g, scale_k] blocks, scale_k values per row.
            sa_nbytes = fx.Index(scale_k) * m_in * fx.Index(scale_byte_size)
            # scale_b: [num_groups, scale_k, scale_n]
            sb_nbytes = num_groups_in * fx.Index(scale_n * scale_k * scale_byte_size)
        else:
            # scale_a: [M_total], one value per row.
            sa_nbytes = m_in * fx.Index(scale_byte_size)
            # scale_b: [num_groups, N], one value per column of each group.
            sb_nbytes = num_groups_in * n_in * fx.Index(scale_byte_size)
        sa_rsrc = buffer_ops.create_buffer_resource(
            arg_scale_a, max_size=False, num_records_bytes=sa_nbytes
        )
        sb_rsrc = buffer_ops.create_buffer_resource(
            arg_scale_b, max_size=False, num_records_bytes=sb_nbytes
        )

        # Group metadata is INT32 when it holds cumulative offsets and INT64
        # when it holds row counts. "batched" carries none, so no resource is
        # built for it and arg_m_sizes goes unread.
        if const_expr(reads_group_meta):
            meta_bytes = 4 if layout == "offsets" else 8
            ms_rsrc = buffer_ops.create_buffer_resource(
                arg_m_sizes,
                max_size=False,
                num_records_bytes=num_groups_in * fx.Index(meta_bytes),
            )

        def _i32(v):  # raw i32 constant (arith.* requires unwrapped MLIR values)
            return arith.constant(int(v), type=T.i32)

        bx_i32 = arith.index_cast(T.i32, bx)
        tile_m_c = _i32(tile_m)
        tile_m_bump = _i32(tile_m - 1)

        if const_expr(slab_layout):
            # Every group owns a fixed slab of expected_m rows, so the group is a
            # grid axis and needs no resolution.
            group_id_i32 = arith.index_cast(T.i32, bz)
            expected_m_i32 = arith.divui(
                arith.index_cast(T.i32, m_in), _i32(num_groups)
            )
            group_m_start_i32 = arith.muli(group_id_i32, expected_m_i32)
            group_m_size_i32 = expected_m_i32
            row_start_i32 = arith.addi(group_m_start_i32, arith.muli(bx_i32, tile_m_c))
            if const_expr(reads_group_meta):
                # m_sizes holds the count of rows in each slab that carry real
                # data; the rest is padding the epilogue must not write.
                valid_m = buffer_ops.buffer_load(
                    ms_rsrc, bz * 2, vec_width=1, dtype=T.i32
                )
                # A group can hold fewer valid rows than the grid has tiles for
                # it, so skip whole tiles that start past them.
                is_valid = arith.cmpi(
                    arith.CmpIPredicate.slt, arith.muli(bx_i32, tile_m_c), valid_m
                )
            else:
                # The slab is full, so every row of it carries data and the grid
                # is exactly ceil(slab / tile_m) tiles: no tile starts past the
                # slab, and the guard is dropped rather than emitted always-true.
                # The rows of the tile past the slab still carry nothing, though:
                # tile_m need not divide it, and the overrun lands on the next
                # group, so the epilogue still masks by row.
                valid_m = expected_m_i32
                is_valid = True
            row_limit_i32 = arith.addi(group_m_start_i32, valid_m)
        else:
            # Packed layout: groups are concatenated along M, so resolve which one
            # owns this flat M-tile id (bx) from m_sizes. Doing it here rather than
            # from a host-built dispatch map keeps the launch free of helper
            # kernels, which matters under CUDA-graph capture where each one is
            # replayed per call. num_groups is a compile-time constant, so the loop
            # unrolls to a few scalar ops. acc_m/acc_t are the running m_start and
            # tile_start prefixes; tiles beyond the real tile count (the grid extent
            # is an upper bound) match no group and stay marked -1.
            acc_m = _i32(0)  # cumulative rows before group g (m_starts[g])
            acc_t = _i32(0)  # cumulative tiles before group g (tile_starts[g])
            group_id_i32 = _i32(-1)
            row_start_i32 = _i32(0)
            row_limit_i32 = _i32(0)
            group_m_start_i32 = _i32(0)  # first global row of the owning group
            group_m_size_i32 = _i32(0)  # row count of the owning group
            for _g in range_constexpr(num_groups):
                if const_expr(layout == "offsets"):
                    # Cumulative int32 row ends. acc_m is already the running
                    # prefix, so a group's own row count is the step between them;
                    # decoding here costs a subtract and saves the caller a whole
                    # kernel launch to difference the offsets host-side.
                    m_g = arith.subi(
                        buffer_ops.buffer_load(ms_rsrc, _g, vec_width=1, dtype=T.i32),
                        acc_m,
                    )
                else:
                    # m_sizes is int64; read the low dword of element _g (index
                    # _g*2 in dwords). Row counts fit in int32, so the high dword
                    # is always zero and no host-side narrowing kernel is needed.
                    m_g = buffer_ops.buffer_load(
                        ms_rsrc, _g * 2, vec_width=1, dtype=T.i32
                    )
                tiles_g = arith.divui(arith.addi(m_g, tile_m_bump), tile_m_c)
                acc_t_next = arith.addi(acc_t, tiles_g)
                in_grp = arith.andi(
                    arith.cmpi(arith.CmpIPredicate.sge, bx_i32, acc_t),
                    arith.cmpi(arith.CmpIPredicate.slt, bx_i32, acc_t_next),
                )
                rs = arith.addi(acc_m, arith.muli(arith.subi(bx_i32, acc_t), tile_m_c))
                rl = arith.addi(acc_m, m_g)
                group_id_i32 = arith.select(in_grp, _i32(_g), group_id_i32)
                row_start_i32 = arith.select(in_grp, rs, row_start_i32)
                row_limit_i32 = arith.select(in_grp, rl, row_limit_i32)
                group_m_start_i32 = arith.select(in_grp, acc_m, group_m_start_i32)
                group_m_size_i32 = arith.select(in_grp, m_g, group_m_size_i32)
                acc_m = arith.addi(acc_m, m_g)
                acc_t = acc_t_next

            is_valid = arith.cmpi(arith.CmpIPredicate.sge, group_id_i32, _i32(0))

        # Early exit for surplus/no-op tiles.
        if is_valid:
            group_idx = fx.Index(group_id_i32)

            # Global row base of this tile and the exclusive row end of its group
            # (the group end masks the partial-tile tail in the epilogue store).
            bx_m = fx.Index(row_start_i32)

            _t = compute_mfma_tiling(tile_m=tile_m, tile_n=tile_n)
            m_repeat = _t.m_repeat
            n_per_wave = _t.n_per_wave
            num_acc_n = _t.num_acc_n

            acc_init, accs = init_accumulators(_t.num_accs)

            _nb = make_n_block_coords(
                wave_id=wave_id,
                by_n=by_n,
                group_idx=group_idx,
                num_groups_in=num_groups_in,
                n_in=n_in,
                k_in=k_in,
                lane_mod_16=lane_mod_16,
                kpack_bytes=kpack_bytes,
                elem_bytes=elem_bytes,
                scale_block_n=scale_block_n,
                scale_k=scale_k,
                n_per_wave=n_per_wave,
                num_acc_n=num_acc_n,
            )
            n_tile_base = _nb.n_tile_base
            n_block_for_scale = _nb.n_block_for_scale
            layout_b = _nb.layout_b
            n_blk_list = _nb.n_blk_list
            n_intra_list = _nb.n_intra_list
            c_scale_k = _nb.c_scale_k

            # Predicate for the partial final K tile; a no-op unless k_padding is
            # on and K stops mid-tile.
            k_tail_mask = make_k_tail_mask(
                k_padding=k_padding,
                num_k_tiles=num_k_tiles,
                k=k,
                tile_k=tile_k,
                k_in=k_in,
            )

            (
                prefetch_a_tile,
                store_a_tile_to_lds,
                a_row_local,
                a_col_local_i32,
                k_blocks16,
            ) = make_a_tile_loaders(
                a_rsrc=a_rsrc,
                lds_a=lds_a,
                layout_lds=layout_lds,
                bx_m=bx_m,
                tx=tx,
                tile_m=tile_m,
                tile_k=tile_k,
                tile_k_bytes=tile_k_bytes,
                tile_k_dwords=tile_k_dwords,
                chunk_i32_a=chunk_i32_a,
                num_a_loads=num_a_loads,
                total_threads=total_threads,
                elem_bytes=elem_bytes,
                k_in=k_in,
                k_tail_mask=k_tail_mask,
            )

            lds_load_packs_k64 = make_lds_loader(
                lds_a=lds_a,
                layout_lds=layout_lds,
                k_blocks16=k_blocks16,
            )

            # Base coordinates for A0 prefetch (mi=0, ku=0)
            row_a_lds_base = lane_mod_16  # mi=0
            col_offset_base_bytes = lane_div_16 * fx.Index(16)  # ku=0

            # ---- B load path: preshuffled (HBM->registers) vs plain (HBM->LDS->registers) ----
            if const_expr(b_preshuffled):
                load_b_tile = make_b_loader(
                    arg_b=arg_b,
                    b_rsrc=b_rsrc,
                    layout_b=layout_b,
                    n_blk_list=n_blk_list,
                    n_intra_list=n_intra_list,
                    lane_div_16=lane_div_16,
                    kpack_bytes=kpack_bytes,
                    elem_bytes=elem_bytes,
                    k_unroll=k_unroll,
                    num_acc_n=num_acc_n,
                )
            else:
                (
                    prefetch_b_tile,
                    store_b_tile_to_lds,
                    _b_row_local,
                    _b_col_local_i32,
                    k_blocks16_b,
                ) = make_b_tile_loaders(
                    b_rsrc=b_rsrc,
                    lds_b=lds_b,
                    layout_lds_b=layout_lds_b,
                    by_n=by_n,
                    group_idx=group_idx,
                    tx=tx,
                    tile_n=tile_n,
                    tile_k=tile_k,
                    tile_k_bytes=tile_k_bytes,
                    tile_k_dwords=tile_k_dwords,
                    chunk_i32_b=chunk_i32_b,
                    num_b_loads=num_b_loads,
                    total_threads=total_threads,
                    elem_bytes=elem_bytes,
                    n_in=n_in,
                    k_in=k_in,
                    k_tail_mask=k_tail_mask,
                    n_padding=n_padding,
                )
                lds_load_b_packs_k64 = make_lds_b_loader(
                    lds_b=lds_b,
                    layout_lds_b=layout_lds_b,
                    k_blocks16_b=k_blocks16_b,
                )
                load_b_tile_from_lds = make_plain_b_tile(
                    lds_load_b_packs_k64=lds_load_b_packs_k64,
                    lane_mod_16=lane_mod_16,
                    n_tile_base=n_tile_base,
                    col_offset_base_bytes=col_offset_base_bytes,
                    k_unroll=k_unroll,
                    num_acc_n=num_acc_n,
                )

            mfma_res_ty = T.f32x4

            ku_per_sb = scale_block_k // 64
            rocdl.sched_barrier(0)

            if const_expr(b_preshuffled):
                hot_loop_scheduler = make_hot_loop_scheduler(
                    _use_hw_scale=_use_hw_scale,
                    sb_per_tile=sb_per_tile,
                    m_repeat=m_repeat,
                    num_acc_n=num_acc_n,
                    k_unroll=k_unroll,
                    num_a_loads=num_a_loads,
                    ku_per_sb=ku_per_sb,
                )

            prefetch_scales = make_prefetch_scales(
                _use_hw_scale=_use_hw_scale,
                sa_rsrc=sa_rsrc,
                sb_rsrc=sb_rsrc,
                group_idx=group_idx,
                scale_n=scale_n,
                scale_k=scale_k,
                c_scale_k=c_scale_k,
                n_block_for_scale=n_block_for_scale,
                bx_m=bx_m,
                lane_mod_16=lane_mod_16,
                m_in=m_in,
                sb_per_tile=sb_per_tile,
                m_repeat=m_repeat,
                num_acc_n=num_acc_n,
            )

            compute_tile = make_compute_tile(
                _use_hw_scale=_use_hw_scale,
                _is_gfx950=_is_gfx950,
                lds_load_packs_k64=lds_load_packs_k64,
                sa_rsrc=sa_rsrc,
                sb_rsrc=sb_rsrc,
                group_idx=group_idx,
                scale_n=scale_n,
                scale_k=scale_k,
                c_scale_k=c_scale_k,
                n_block_for_scale=n_block_for_scale,
                bx_m=bx_m,
                lane_mod_16=lane_mod_16,
                lane_div_16=lane_div_16,
                m_in=m_in,
                sb_per_tile=sb_per_tile,
                m_repeat=m_repeat,
                num_acc_n=num_acc_n,
                ku_per_sb=ku_per_sb,
                col_offset_base_bytes=col_offset_base_bytes,
                mfma_res_ty=mfma_res_ty,
                acc_init=acc_init,
                group_m_start=fx.Index(group_m_start_i32),
                group_m_size=fx.Index(group_m_size_i32),
                blockscale=blockscale,
            )

            if const_expr(b_preshuffled):
                run_kloop = make_pingpong_kloop(
                    num_k_tiles=num_k_tiles,
                    tile_k=tile_k,
                    prefetch_a_tile=prefetch_a_tile,
                    store_a_tile_to_lds=store_a_tile_to_lds,
                    load_b_tile=load_b_tile,
                    prefetch_scales=prefetch_scales,
                    compute_tile=compute_tile,
                    hot_loop_scheduler=hot_loop_scheduler,
                    lds_load_packs_k64=lds_load_packs_k64,
                    lds_base_pong=lds_base_pong,
                    lds_base_ping=lds_base_ping,
                    row_a_lds_base=row_a_lds_base,
                    col_offset_base_bytes=col_offset_base_bytes,
                )
            else:
                run_kloop = make_kloop_plain(
                    num_k_tiles=num_k_tiles,
                    tile_k=tile_k,
                    prefetch_a_tile=prefetch_a_tile,
                    store_a_tile_to_lds=store_a_tile_to_lds,
                    prefetch_b_tile=prefetch_b_tile,
                    store_b_tile_to_lds=store_b_tile_to_lds,
                    load_b_tile_from_lds=load_b_tile_from_lds,
                    prefetch_scales=prefetch_scales,
                    compute_tile=compute_tile,
                    lds_base_pong=lds_base_pong,
                    lds_base_b=lds_base_b,
                )
            accs = run_kloop(accs)

            if const_expr(not blockscale):
                # Rowwise scales are constant along K, so the whole reduction is
                # scaled here in one pass rather than per K tile.
                accs = make_rowwise_scaler(
                    sa_rsrc=sa_rsrc,
                    sb_rsrc=sb_rsrc,
                    group_idx=group_idx,
                    n_in=n_in,
                    by_n=by_n,
                    n_tile_base=n_tile_base,
                    bx_m=bx_m,
                    lane_mod_16=lane_mod_16,
                    lane_div_16=lane_div_16,
                    m_repeat=m_repeat,
                    num_acc_n=num_acc_n,
                )(accs)

            # ===== Epilogue: CShuffle vectorized stores =====
            c_n = n_in
            e_vec = 4 if (tile_n % (32 * 4)) == 0 else 2

            write_row_to_lds, store_pair = make_epilogue_writers(
                accs=accs,
                d_rsrc=d_rsrc,
                out_mlir=out_mlir,
                e_vec=e_vec,
                c_n=c_n,
                n_padding=n_padding,
            )

            # Mask the partial-tile tail: skip stores for global rows at or beyond
            # the owning group's end. Returning (ctx, pred) lets the epilogue skip
            # the whole N-store loop for out-of-group rows.
            def precompute_row(*, row_local, row):
                row_i32 = arith.index_cast(T.i32, row)
                row_valid = arith.cmpi(arith.CmpIPredicate.ult, row_i32, row_limit_i32)
                return (None, row_valid)

            mfma_epilog(
                use_cshuffle=True,
                arith=arith,
                vector=vector,
                gpu=gpu,
                scf=scf,
                range_constexpr=range_constexpr,
                tile_m=tile_m,
                tile_n=tile_n,
                e_vec=e_vec,
                m_repeat=m_repeat,
                num_acc_n=num_acc_n,
                tx=tx,
                lane_div_16=lane_div_16,
                lane_mod_16=lane_mod_16,
                bx_m=bx_m,
                by_n=by_n,
                n_tile_base=n_tile_base,
                lds_out=lds_out,
                frag_elem_type=out_mlir(),
                write_row_to_lds=write_row_to_lds,
                precompute_row=precompute_row,
                store_pair=store_pair,
            )

    # ===== JIT Launcher =====
    @flyc.jit
    def launch_grouped_gemm_blockscale_contiguous(
        arg_d: fx.Tensor,
        arg_a: fx.Tensor,
        arg_b: fx.Tensor,
        arg_scale_a: fx.Tensor,
        arg_scale_b: fx.Tensor,
        arg_m_sizes: fx.Tensor,
        i32_m: fx.Int32,
        i32_n: fx.Int32,
        i32_k: fx.Int32,
        i32_num_groups: fx.Int32,
        i32_num_m_tiles: fx.Int32,
        stream: fx.Stream,
    ):
        allocator.finalized = False
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            allocator.finalize()

        # Grid dimensions. In the packed layout the M axis enumerates output
        # M-tiles across all groups; its extent is a host-known upper bound on the
        # tile count, and tiles past the real count match no group and exit early.
        # The padded layout instead gives each group its own z slice, so the M axis
        # only has to cover one group's slab and tiles past its valid rows exit.
        n_in = fx.Index(i32_n)
        gx = (
            (n_in + fx.Index(tile_n - 1)) // fx.Index(tile_n)
            if const_expr(n_padding)
            else n_in // fx.Index(tile_n)
        )  # N-blocks
        gy = fx.Index(i32_num_m_tiles)  # M-tiles
        gz = fx.Index(i32_num_groups) if const_expr(slab_layout) else fx.Index(1)

        launcher = grouped_gemm_blockscale_contiguous_kernel(
            arg_d,
            arg_a,
            arg_b,
            arg_scale_a,
            arg_scale_b,
            arg_m_sizes,
            i32_m,
            i32_n,
            i32_k,
            i32_num_groups,
        )
        if waves_per_eu is not None:
            _wpe = int(waves_per_eu)
            if _wpe >= 1:
                for op in ctx.gpu_module_body.operations:
                    if hasattr(op, "attributes") and op.OPERATION_NAME == "gpu.func":
                        op.attributes["rocdl.waves_per_eu"] = ir.IntegerAttr.get(
                            T.i32, _wpe
                        )
        launcher.launch(grid=(gx, gy, gz), block=(total_threads, 1, 1), stream=stream)

    return launch_grouped_gemm_blockscale_contiguous
