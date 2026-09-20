# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Grouped GEMM kernel.

The `in_dtype` argument selects the operand element type: "fp8", which the
kernel was written for and is still named after, or "bf16". It decides the MFMA
the K loop issues and the width of every operand address, and it fixes the
scaling scheme, since only a quantised operand carries scales. Everything else
-- the group resolution, the loaders, the LDS swizzle and the epilogue -- is
shared, the operand staging being expressed in bytes throughout.

The `layout` argument selects both which axis the groups divide and how their
geometry is encoded. Where the groups divide M they are concatenated with
arbitrary, not tile-aligned, row counts and the output is compact
[M_total, N]; each output M-tile then belongs to exactly one group, resolved
from m_sizes, and rows at or beyond that group's end are the partial-tile tail
and are masked out of the store. Groups may instead occupy fixed per-group
slabs, or divide N or K. Only the resolution step differs between them, so the
loaders, the K loop and the epilogue are shared.

Where there are scales they are FP32 (software scaling) on all architectures.

Tensors:
  - A: [M_total, K] - the rows of every group, packed or in per-group slabs
  - B: [num_groups, N, K] - one weight matrix per group, in the MFMA B
    layout when b_preshuffled; a single [total_N, K] or [N, total_K] matrix
    where the groups divide N or K
  - m_sizes: [num_groups] - the group geometry, whose encoding the `layout`
    argument selects: INT64 row counts, INT32 cumulative offsets, or unused
  - D: [M_total, N] BF16 - output, which is the flattened [G * M, N] where the
    groups divide K and each produces a whole output of its own

A and B carry `in_dtype` elements; the output is BF16 or FP16 whichever they
are.

BF16 operands carry their own exponent, so they have no scales: the scale
arguments go unread, nothing is folded in the K loop, and the epilogue writes
the accumulators straight out.

Rowwise scaling carries one FP32 scale per row of A and per column of B, both
constant along K, and applies them in the epilogue.

Block scaling carries FP32 scales at (1, 128) for A -- per-token, per-128-K --
and (128, 128) for B, as:
  - scale_a: laid out as per-group blocks (as written by quantize_fp8_group
    with m_sizes): group g's block begins at m_start * scale_k, and within it
    element (local_m, k_block) sits at local_m + k_block * M_g. This is not a
    global [scale_k, M_total] transpose.
  - scale_b: [num_groups, scale_k, scale_n]
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
from flydsl.expr.typing import T, Vector
from flydsl.runtime.device import get_rocm_arch as get_hip_arch
from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr
from mslk.flydsl.kernels.gemm.fp8_grouped_gemm_common import (
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
    make_pingpong_stages,
    make_plain_b_tile,
    make_prefetch_scales,
    make_rowwise_scaler,
    out_mlir_for,
    resolve_group_cols,
    resolve_group_k,
    resolve_group_rows,
    SCALING_BLOCK,
    SCALING_NONE,
    SCALING_ROW,
    setup_lds_allocation,
    setup_lds_allocation_plain,
    validate_lds_budget_plain,
    validate_lds_budget_preshuffle,
    validate_params,
)
from mslk.flydsl.kernels.mma.mfma_epilogues import mfma_epilog

# Supported encodings of the group geometry; see the ``layout`` argument below.
LAYOUTS = ("sizes", "offsets", "padded", "batched", "n_offsets", "k_offsets")

# Lanes the CShuffle epilogue lays across N, matching mfma_epilog's default.
CSHUFFLE_NLANE = 32


@functools.lru_cache(maxsize=128)
def compile_fp8_grouped_gemm(
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
    waves_m: int = 1,
    waves_n: int = 4,
    waves_per_eu: int | None = None,
    b_preshuffled: bool = True,
    in_dtype: str = "fp8",
    scaling: str = SCALING_ROW,
    layout: str = "sizes",
    k_padding: bool = False,
    n_padding: bool = False,
    group_based_b: bool = False,
    group_based_rows: bool = False,
    roll_k: bool = False,
):
    """Compile the grouped GEMM kernel and return the JIT launcher.

    Serves FP8 operands with block or rowwise scales and BF16 operands with
    none; ``in_dtype`` and ``scaling`` select between them.

    Args:
        n: N dimension (output columns per group)
        k: K dimension (reduction dimension)
        num_groups: Number of groups (experts)
        tile_m: M tile size (default 128)
        tile_n: N tile size (default 128)
        tile_k: K tile size (default 128)
        waves_m: Waves dividing the tile's rows (default 1).
        waves_n: Waves dividing the tile's columns (default 4). The block is
            waves_m * waves_n * 64 threads and each wave owns a
            (tile_m / waves_m) x (tile_n / waves_n) slab of the output tile.
            Each wave reads its whole slab's worth of both operands out of LDS,
            so LDS reads per unit of work go as
            waves_m / tile_m + waves_n / tile_n, which is minimised by a grid
            proportioned like the tile.
        scale_block_k: K-dimension scale block size (default 128)
        scale_block_n: N-dimension scale block size (default 128)
        out_dtype: Output data type ("bf16" or "f16")
        waves_per_eu: Minimum waves per EU to ask the register allocator for,
            or None to leave the choice to the compiler. Raising it caps the
            register budget, which trades spills against occupancy.
        k_padding: Emit the per-load K-range predicate so K need only reach a
            16-element load boundary rather than a whole tile. Forced on where
            the groups divide K, whose per-group end is a runtime value.
        n_padding: Emit the per-store column predicate so N need only reach an
            8-column store boundary rather than a whole tile. Forced on where
            the groups divide N, for the same reason.
        group_based_b: Address B from the owning group's first row rather than
            from the stack's, and bound the descriptor by that group. Required
            once the whole stack exceeds the 4 GiB a buffer descriptor
            addresses. The caller decides this from G * N * K, which is
            host-known.
        group_based_rows: Address A and D from the owning group's first row
            rather than from the operand's, and bound each descriptor by that
            group rather than by the whole operand. Required once A or D
            exceeds the 4 GiB a buffer descriptor addresses, past which the
            hardware reads zero and drops stores rather than faulting.
        b_preshuffled: When True (default) B is expected pre-swizzled into the
            MFMA layout and loaded HBM->registers (no B LDS). When False, B is
            plain row-major [num_groups, N, K] and is staged HBM->LDS->registers
            like A. The two paths share the entire kernel body (tile-map group
            dispatch, scaling, wide-MFMA, CShuffle epilogue); only the B load
            stage and its LDS allocation differ.
        in_dtype: Element type of A and B, "fp8" (default) or "bf16". It sets
            the MFMA the K loop issues and the width of every operand address,
            and it fixes `scaling`: fp8 is quantised and needs a scheme, bf16
            carries its own exponent and needs "none". A bf16 tile spans twice
            the LDS of the fp8 tile of the same shape, so the tile space the
            LDS budget admits is correspondingly smaller.
        scaling: Selects the scaling scheme, which sets the expected layout of
            scale_a / scale_b and where the scales are applied.
            "row" (default): scale_a is [M_total] and scale_b is [num_groups, N],
            one factor per row of A and per column of B, applied once in the
            epilogue. Tiles are then free of scale-block alignment, so tile_n may
            be smaller than scale_block_n.
            "block": scale_a is per-group [M_g, scale_k] blocks and scale_b is
            [num_groups, scale_k, scale_n], applied per scale block inside the K
            loop, which requires the tile to align to the blocks.
            "none": the operands carry no scales at all, as bf16 does. Nothing is
            folded in the K loop and nothing is applied in the epilogue, and the
            scale arguments are ignored.
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
            "n_offsets": the groups partition the output's COLUMNS rather than
            its rows. A is per-group [G, M, K] slabs as in "batched", B is one
            [total_N, K] matrix whose rows the groups divide, the output is
            [M, total_N] with the groups side by side, and m_sizes is
            [num_groups] INT32 cumulative column ends. Rowwise plain-B only.
            "k_offsets": the groups divide the CONTRACTION. A is [M, total_K]
            and B is [N, total_K], each group taking a column slice, and every
            group produces a full [M, N] so the output is [G, M, N] with nothing
            packed. m_sizes is [num_groups] INT32 cumulative K ends. The slice
            length is a runtime value, so this layout always rolls the K loop.
            Rowwise plain-B only.

        roll_k: Emit the K loop once as a real loop instead of unrolling it over
            every K tile. The unrolled form folds the tile index into every
            address and keeps the pipeline free of loop overhead, but the traced
            IR, the compile time and the final code all then grow linearly with
            K. Rolling holds them constant, at the cost of carrying the
            accumulators and the prefetched tiles as loop state. Both K loops
            can be rolled: the plain one a tile at a time, the ping-pong one a
            pair at a time so its buffer alternation stays compile-time inside
            the body. This is the kernel's mechanism and defaults off; the host
            dispatch sets the policy.

    Returns:
        JIT launcher function.
    """
    if layout not in LAYOUTS:
        raise ValueError(f"layout must be one of {LAYOUTS}, got {layout!r}")
    # Each group owns a fixed slab of rows and rides the grid's z axis, so it
    # does not have to be resolved from the row counts.
    slab_layout = layout in ("padded", "batched")
    # Groups divide the output's columns, so the group follows from the N-block
    # index and B is one matrix rather than a stack of per-group ones.
    n_grouped = layout == "n_offsets"
    # Groups divide the contraction: every group owns a K slice of both operands
    # and produces a whole output of its own.
    k_grouped = layout == "k_offsets"
    if k_grouped:
        if scaling != SCALING_ROW or b_preshuffled:
            raise ValueError(
                "layout 'k_offsets' supports only rowwise scaling with plain B: "
                "a scale block cannot straddle a group's K slice, and the "
                "preshuffled layout swizzles K across the whole matrix"
            )
        # A group's K length is only known on the device. The trip count is
        # therefore not a compile-time constant, so the loop cannot be
        # unrolled, and every tile is bounded by that runtime end rather than
        # by a compile-time last tile.
        roll_k = True
        k_padding = True
    if n_grouped:
        if scaling != SCALING_ROW or b_preshuffled:
            raise ValueError(
                "layout 'n_offsets' supports only rowwise scaling with plain B: "
                "ragged N would need per-group scale blocks, and the preshuffled "
                "layout swizzles N across the whole matrix"
            )
        # A group's column end is a runtime value, so unlike a compile-time N
        # remainder the tail mask cannot be elided; N only has to reach a store
        # boundary, which validate_params checks below.
        n_padding = True

    # The wave grid has to cut the tile into whole 16x16 MFMA tiles, since a
    # wave's accumulators are counted in them and cannot straddle a boundary.
    if waves_m < 1 or waves_n < 1:
        raise ValueError(f"wave grid must be positive, got {waves_m}x{waves_n}")
    if tile_m % (waves_m * 16) or tile_n % (waves_n * 16):
        raise ValueError(
            f"tile {tile_m}x{tile_n} does not divide into a {waves_m}x{waves_n} "
            "wave grid of whole 16x16 MFMA tiles"
        )
    num_waves = waves_m * waves_n

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
        scaling=scaling,
        in_dtype=in_dtype,
        k_padding=k_padding,
        n_padding=n_padding,
        scale_block_k=scale_block_k,
        scale_block_n=scale_block_n,
        out_dtype=out_dtype,
    )
    if in_dtype != "fp8" and b_preshuffled:
        # The preshuffled B layout, its HBM->register loader and the hot loop's
        # instruction-group counts are all written around a one-byte element.
        raise ValueError(
            f"in_dtype {in_dtype!r} is supported with plain B only; pass "
            "b_preshuffled=False"
        )

    _c = compute_compile_constants(
        n=n,
        k=k,
        tile_m=tile_m,
        tile_n=tile_n,
        tile_k=tile_k,
        scale_block_k=scale_block_k,
        scale_block_n=scale_block_n,
        k_padding=k_padding,
        in_dtype=in_dtype,
        scaling=scaling,
        num_waves=num_waves,
    )
    # Check the LDS budget before tracing: the compiler treats an overflow as a
    # hard error that kills the process, which an autotuner cannot skip. Capacity
    # is arch-dependent (64 KiB gfx942, 160 KiB gfx950).
    if b_preshuffled:
        # Preshuffled B goes HBM->registers; only A ping-pong / epilogue use LDS.
        validate_lds_budget_preshuffle(
            tile_m=tile_m,
            tile_n=tile_n,
            tile_k=tile_k,
            elem_bytes=_c.elem_bytes,
            arch=gpu_arch,
        )
    else:
        # Plain B needs its own LDS buffer alongside A.
        validate_lds_budget_plain(
            tile_m=tile_m,
            tile_n=tile_n,
            tile_k=tile_k,
            elem_bytes=_c.elem_bytes,
            b_pingpong=False,
            arch=gpu_arch,
        )
    out_mlir = out_mlir_for(out_dtype)

    total_threads = _c.total_threads
    elem_bytes = _c.elem_bytes
    num_k_tiles = _c.num_k_tiles
    scale_k = _c.scale_k
    scale_n = _c.scale_n
    sb_per_tile = _c.sb_per_tile
    ku_per_sb = _c.ku_per_sb
    k_unroll = _c.k_unroll
    kpack_bytes = _c.kpack_bytes
    tile_k_bytes = _c.tile_k_bytes
    tile_k_dwords = _c.tile_k_dwords
    chunk_i32_a = _c.chunk_i32_a
    num_a_loads = _c.num_a_loads
    chunk_i32_b = _c.chunk_i32_b
    num_b_loads = _c.num_b_loads

    if b_preshuffled:
        lds_alloc_offset, lds_tile_bytes = setup_lds_allocation(
            allocator=allocator,
            tile_m=tile_m,
            tile_k=tile_k,
            tile_n=tile_n,
            elem_bytes=elem_bytes,
        )
        lds_b_offset_bytes = None
    else:
        lds_alloc_offset, lds_tile_bytes, lds_b_offset_bytes = (
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
    _variant = "pingpong" if b_preshuffled else "plain"
    _scaling = scaling if scaling != SCALING_ROW else "rowwise"
    _kpad = "_kpad" if k_padding else ""
    _roll = "_rollk" if roll_k else ""
    _wpe = f"_wpe{int(waves_per_eu)}" if waves_per_eu else ""
    _npad = "_npad" if n_padding else ""
    # The wave grid changes the emitted kernel, so it has to reach the name or a
    # second grid over the same tile would collide in the JIT cache.
    _wg = f"_w{waves_m}x{waves_n}" if (waves_m, waves_n) != (1, 4) else ""
    # The row basing changes the emitted kernel, so it has to reach the name or
    # the two forms would collide in the JIT cache.
    _gbr = "_gbr" if group_based_rows else ""
    _gbb = "_gbb" if group_based_b else ""
    module_name = (
        f"grouped_gemm_{in_dtype}_{_scaling}_{layout}_{_variant}_{out_dtype}"
        f"_n{n}_k{k}_g{num_groups}"
        f"_t{tile_m}x{tile_n}x{tile_k}{_kpad}{_npad}{_roll}{_wpe}{_wg}{_gbr}{_gbb}"
    ).replace("-", "_")

    # The AMDGPU default caps a workgroup at 256 threads, so a grid of more than
    # four waves has to declare its size up front. Below that the bound is
    # already right, and declaring it anyway would perturb register allocation
    # on every existing configuration for no reason.
    _kernel_attrs = (
        {"known_block_size": [total_threads, 1, 1]} if total_threads > 256 else {}
    )

    @flyc.kernel(name=module_name, **_kernel_attrs)
    def fp8_grouped_gemm_kernel(
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

        # Wave/lane decomposition (total_threads = num_waves x 64 lanes)
        layout_wave_lane = fx.make_layout((num_waves, 64), stride=(64, 1))
        coord_wave_lane = fx.idx2crd(fx.Int32(tx), layout_wave_lane)
        wave_id = fx.get(coord_wave_lane, 0)
        lane_id = fx.get(coord_wave_lane, 1)

        # Wave grid: wave w owns rows [wm * tile_m / waves_m, +tile_m / waves_m)
        # and, via n_tile_base below, the matching slab of columns. A 1 x waves_n
        # grid leaves every wave spanning the whole of tile_m, and the row
        # derivations then take no offset at all.
        if const_expr(waves_m > 1):
            m_wave_base = (wave_id // fx.Index(waves_n)) * fx.Index(tile_m // waves_m)
        else:
            m_wave_base = None

        # Lane decomposition for MFMA (lane_id -> lane_div_16, lane_mod_16)
        layout_lane16 = fx.make_layout((4, 16), stride=(16, 1))
        coord_lane16 = fx.idx2crd(fx.Int32(lane_id), layout_lane16)
        lane_div_16 = fx.get(coord_lane16, 0)
        lane_mod_16 = fx.get(coord_lane16, 1)

        # LDS setup: ping-pong A buffers (preshuffle) or A ping-pong + single B
        # buffer (plain). B LDS is only needed for the plain path.
        # The A/B arena is addressed as raw bytes -- T.f8 is its byte element
        # type, not the dtype of the operands -- so its shapes, strides and
        # column coordinates are all byte quantities. That is what lets the
        # XOR16 swizzle and the LDS pack loads be shared across input dtypes.
        base_ptr = allocator.get_base()
        lds_a = SmemPtr(
            base_ptr, lds_alloc_offset, T.f8, shape=(2 * tile_m * tile_k_bytes,)
        ).get()
        lds_stride = tile_k_bytes
        layout_lds = fx.make_layout((tile_m, tile_k_bytes), stride=(lds_stride, 1))
        lds_base_pong = fx.Index(0)
        lds_base_ping = fx.Index(lds_tile_bytes)

        if const_expr(not b_preshuffled):
            # Plain-B LDS buffer, placed just past the A ping-pong region.
            lds_b = SmemPtr(
                base_ptr,
                lds_alloc_offset,
                T.f8,
                shape=((lds_b_offset_bytes + tile_n * tile_k_bytes),),
            ).get()
            layout_lds_b = fx.make_layout(
                (tile_n, tile_k_bytes), stride=(tile_k_bytes, 1)
            )
            lds_base_b = fx.Index(lds_b_offset_bytes)

        # CShuffle epilogue LDS (aliased from same base, out-dtype element type)
        lds_out = SmemPtr(
            base_ptr, lds_alloc_offset, out_mlir(), shape=(tile_m * tile_n,)
        ).get()

        # Buffer resources. These extents are byte counts, so the K extent of a
        # row has to be widened by the element size.
        k_bytes_in = k_in if elem_bytes == 1 else k_in * fx.Index(elem_bytes)

        def rebased_resource(arg, num_records_bytes, byte_offset=None):
            """Buffer descriptor over one slab of an operand, based at that slab.

            A descriptor addresses at most 4 GiB: both its num_records and the
            voffset a buffer_load adds are 32-bit. An out-of-range load reads
            zero and an out-of-range store is dropped, so an operand past that
            limit yields a wrong result rather than an error.

            Folding the group into the base address keeps the descriptor over a
            single group and does the group arithmetic in 64 bits. The offset
            goes to `base_byte_offset`, which applies to the descriptor's base
            pointer, so the group term never passes through a 32-bit quantity.
            """
            return buffer_ops.create_buffer_resource(
                arg,
                max_size=False,
                num_records_bytes=num_records_bytes,
                base_byte_offset=byte_offset,
            )

        # A's, B's and D's descriptors are all built further down, once the group
        # is resolved: each is based at this block's group rather than at the
        # whole operand, so that none of them has to address more than one
        # group's worth. See `rebased_resource`.
        b_nbytes = n_in * k_bytes_in
        # What one descriptor must cover when it is not based at a group.
        b_whole_nbytes = (
            b_nbytes if (n_grouped or k_grouped) else num_groups_in * b_nbytes
        )
        b_rsrc_whole = (
            None if group_based_b else rebased_resource(arg_b, b_whole_nbytes)
        )

        # The output is [M, total_N] when the groups divide N: m_in counts the
        # rows of every group's slab, but they share the output's rows.
        if const_expr(n_grouped):
            d_rows = m_in // num_groups_in
        elif const_expr(k_grouped):
            d_rows = num_groups_in * m_in
        else:
            d_rows = m_in
        d_row_bytes = n_in * fx.Index(2)  # bf16/f16 = 2 bytes

        # Scale buffers — gfx950 HW E8M0 path consumes int8 (one byte/scale,
        # pre-packed on host); gfx942 SW path consumes f32.
        scale_byte_size = 1 if _use_hw_scale else 4

        if const_expr(scaling == SCALING_NONE):
            # Unscaled operands: nothing reads these, and the caller has no
            # scales to hand over, so the resources cover no bytes at all.
            sa_nbytes = fx.Index(0)
            sb_nbytes = fx.Index(0)
        elif const_expr(scaling == SCALING_BLOCK):
            # scale_a: per-group [M_g, scale_k] blocks, scale_k values per row.
            sa_nbytes = fx.Index(scale_k) * m_in * fx.Index(scale_byte_size)
            # scale_b: [num_groups, scale_k, scale_n]
            sb_nbytes = num_groups_in * fx.Index(scale_n * scale_k * scale_byte_size)
        else:
            # scale_a: [M_total], one value per row, or one per row per group
            # where the groups divide K and each quantised its own slice.
            sa_rows = num_groups_in * m_in if const_expr(k_grouped) else m_in
            sa_nbytes = sa_rows * fx.Index(scale_byte_size)
            # scale_b: [num_groups, N], one value per column of each group, or
            # a flat [total_N] when the groups divide N.
            sb_cols = n_in if const_expr(n_grouped) else num_groups_in * n_in
            sb_nbytes = sb_cols * fx.Index(scale_byte_size)
        sa_rsrc = buffer_ops.create_buffer_resource(
            arg_scale_a, max_size=False, num_records_bytes=sa_nbytes
        )
        sb_rsrc = buffer_ops.create_buffer_resource(
            arg_scale_b, max_size=False, num_records_bytes=sb_nbytes
        )

        if const_expr(k_grouped):
            # The group is the grid axis, so only the ends of its K slice have
            # to be read. Group 0 starts at zero and is not read.
            _ks = resolve_group_k(
                arg_offsets=arg_m_sizes,
                num_groups_in=num_groups_in,
                slab_idx=bz,
            )
            k_start_i32 = _ks.k_start
            k_end_i32 = _ks.k_end
        if const_expr(n_grouped):
            _col = resolve_group_cols(
                arg_offsets=arg_m_sizes,
                num_groups_in=num_groups_in,
                n_block_idx=by,
                tile_n=tile_n,
                num_groups=num_groups,
            )
            _col_group_id = _col.group_id
        else:
            _col = None
            _col_group_id = None

        def _i32k(v):  # raw i32 constant
            return arith.constant(int(v), type=T.i32)

        if const_expr(k_grouped):
            # Every group spans the whole output, so there is no row geometry
            # to resolve: the group rides the z axis and the M-tile indexes the
            # output directly.
            group_id_i32 = arith.index_cast(T.i32, bz)
            group_m_start_i32 = _i32k(0)
            group_m_size_i32 = arith.index_cast(T.i32, m_in)
            row_start_i32 = arith.muli(arith.index_cast(T.i32, bx), _i32k(tile_m))
            row_limit_i32 = arith.index_cast(T.i32, m_in)
            # A group that contracts over nothing contributes nothing, and its
            # output slab is left to the caller, as CK leaves it.
            is_valid = arith.cmpi(arith.CmpIPredicate.slt, k_start_i32, k_end_i32)
        else:
            _grp = resolve_group_rows(
                arg_m_sizes=arg_m_sizes,
                num_groups_in=num_groups_in,
                m_in=m_in,
                m_tile_idx=bx,
                slab_idx=bz,
                tile_m=tile_m,
                num_groups=num_groups,
                layout=layout,
                group_id=_col_group_id,
            )
            group_id_i32 = _grp.group_id
            row_start_i32 = _grp.row_start
            row_limit_i32 = _grp.row_limit
            group_m_start_i32 = _grp.group_m_start
            group_m_size_i32 = _grp.group_m_size
            # Rows are always whole slabs when the groups divide N, so validity
            # is entirely a question of whether this N-block belongs to a group.
            is_valid = _col.is_valid if const_expr(n_grouped) else _grp.is_valid
        if const_expr(n_grouped):
            # Blocks are enumerated across the packed groups, so this block's
            # first column comes from the partition rather than from its index.
            by_n = fx.Index(_col.col_base)
            # Where the owning group's columns stop; the bound for both the B
            # row tail and the epilogue's column mask.
            n_bound = fx.Index(_col.col_limit)
            # scale_b is one flat [total_N] here, so the group is already in by_n.
            sb_group_off = fx.Index(0)
        elif const_expr(k_grouped):
            # B is a single [N, total_K] the groups slice by column, so the slice
            # is expressed as a K offset rather than a row base.
            # scale_b is still [G, N], one set of column scales per group.
            n_bound = None
            sb_group_off = None
        else:
            n_bound = None
            sb_group_off = None

        # Early exit for surplus/no-op tiles.
        if is_valid:
            group_idx = fx.Index(group_id_i32)

            # B is one [total_N, K] matrix when the groups divide N or K, and a
            # stack of num_groups [N, K] ones otherwise. Only the stack has a
            # group axis to fold into the base; the single matrix is already one
            # slab, which every block reads whole.
            if const_expr(group_based_b):
                b_rsrc = rebased_resource(
                    arg_b, b_nbytes, byte_offset=group_idx * b_nbytes
                )
            else:
                # One descriptor over the whole stack, so the group stays in the
                # coordinates rather than in the base. Built outside this
                # conditional, since it does not depend on the group.
                b_rsrc = b_rsrc_whole

            # Global row base of this tile and the exclusive row end of its group
            # (the group end masks the partial-tile tail in the epilogue store).
            bx_m = fx.Index(row_start_i32)
            # The output of an N-grouped GEMM is [M, total_N]: every group shares
            # its rows, so the epilogue indexes rows within the slab while A and
            # scale_a keep addressing the flattened [G * M, ...] operands.
            if const_expr(n_grouped):
                bx_m_out = bx * fx.Index(tile_m)
                row_limit_out_i32 = group_m_size_i32
            else:
                bx_m_out = bx_m
                row_limit_out_i32 = row_limit_i32
            # Where the groups divide K, A's [M, total_K] rows are shared by
            # every group while scale_a is [G, M], each group having quantised
            # its own slice. So A keeps the plain row and scale_a takes the
            # group's slab; the output does too, via d_group_off below.
            if const_expr(k_grouped):
                bx_m_scale = bx_m + fx.Index(group_id_i32) * m_in
                d_group_off = fx.Index(group_id_i32) * m_in * n_in
                k_base_div4 = fx.Index(k_start_i32) // fx.Index(4)
                k_bound = fx.Index(k_end_i32)
            else:
                bx_m_scale = bx_m
                d_group_off = None
                k_base_div4 = None
                k_bound = None

            if const_expr(group_based_rows):
                # A and D are addressed from this group's first row rather than
                # from the operand's, since a whole operand can exceed what one
                # descriptor reaches. Rows inside the loader and the epilogue
                # are then group-local, and the group's own extent bounds each
                # descriptor.
                a_group_rows = fx.Index(group_m_size_i32)
                a_row_base = fx.Index(group_m_start_i32)
                a_rsrc = rebased_resource(
                    arg_a,
                    a_group_rows * k_bytes_in,
                    byte_offset=a_row_base * k_bytes_in,
                )
                # Where the groups divide N the output rows are already
                # slab-local, so the group is in the column and there is no row
                # base to take.
                d_row_base = fx.Index(0) if const_expr(n_grouped) else a_row_base
                d_base_bytes = d_row_base * d_row_bytes
                if d_group_off is not None:
                    # The groups divide K, so each owns a whole [M, N] output.
                    d_base_bytes = d_base_bytes + d_group_off * fx.Index(2)
                    # Folded into the base, so the store must not add it again.
                    d_group_off = None
                d_group_rows = d_rows if const_expr(n_grouped) else a_group_rows
                d_rsrc = rebased_resource(
                    arg_d, d_group_rows * d_row_bytes, byte_offset=d_base_bytes
                )
                a_loader_row_base = a_row_base
            else:
                # One descriptor spans each whole operand, so the rows stay
                # global and nothing is subtracted.
                a_rsrc = rebased_resource(arg_a, m_in * k_bytes_in)
                d_rsrc = rebased_resource(arg_d, d_rows * d_row_bytes)
                d_row_base = None
                a_loader_row_base = None

            _t = compute_mfma_tiling(
                tile_m=tile_m, tile_n=tile_n, waves_m=waves_m, waves_n=waves_n
            )
            m_repeat = _t.m_repeat
            n_per_wave = _t.n_per_wave
            num_acc_n = _t.num_acc_n

            acc_init, accs = init_accumulators(_t.num_accs)

            _nb = make_n_block_coords(
                waves_n=waves_n,
                wave_id=wave_id,
                by_n=by_n,
                group_idx=group_idx,
                num_groups_in=num_groups_in,
                n_in=n_in,
                k_in=k_in,
                lane_mod_16=lane_mod_16,
                kpack_bytes=kpack_bytes,
                elem_bytes=elem_bytes,
                b_group_based=group_based_b,
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
                elem_bytes=elem_bytes,
                always=roll_k,
                k_bound=k_bound,
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
                # Group-local where a_rsrc is based at the group's first row;
                # the plain global row otherwise.
                bx_m=(bx_m if a_loader_row_base is None else bx_m - a_loader_row_base),
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
                k_base_div4=k_base_div4,
            )

            lds_load_packs_k64 = make_lds_loader(
                lds_a=lds_a,
                layout_lds=layout_lds,
                k_blocks16=k_blocks16,
            )

            # Base coordinates for A0 prefetch (mi=0, ku=0)
            row_a_lds_base = lane_mod_16  # mi=0
            if const_expr(m_wave_base is not None):
                row_a_lds_base = row_a_lds_base + m_wave_base
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
                    # The base carries the group when B is based per group, so
                    # the row offset must not carry it too.
                    # No group term where the base already carries it, and
                    # none where B has no group axis to carry: the layouts that
                    # divide N or K give every block one shared matrix.
                    b_group_off=(
                        None
                        if (group_based_b or n_grouped or k_grouped)
                        else group_idx * n_in * (k_bytes_in // fx.Index(4))
                    ),
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
                    n_bound=n_bound,
                    k_base_div4=k_base_div4,
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
                m_wave_base=m_wave_base,
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
                m_wave_base=m_wave_base,
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
                scaling=scaling,
                in_dtype=in_dtype,
            )

            if const_expr(b_preshuffled):
                pingpong_prologue, pingpong_pair = make_pingpong_stages(
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

            # The ping-pong state is nested (a B tile is k_unroll pairs of
            # num_acc_n packs); a loop carries a flat list, so convert both ways.
            # The prefetched scales are always None on this software-scaling
            # path, so they are rebuilt rather than carried.
            def _flatten_pp(st):
                b_tile, _sc, a0 = st
                flat = []
                for _p0, _p1 in b_tile:
                    flat.extend(_p0)
                    flat.extend(_p1)
                flat.extend(a0)
                return flat

            def _unflatten_pp(vals):
                vals = [fx.Int64(v) for v in vals]
                b_tile = []
                _i = 0
                for _ in range_constexpr(k_unroll):
                    _p0 = vals[_i : _i + num_acc_n]
                    _i += num_acc_n
                    _p1 = vals[_i : _i + num_acc_n]
                    _i += num_acc_n
                    b_tile.append((_p0, _p1))
                return (b_tile, None, tuple(vals[_i : _i + 2]))

            if const_expr(b_preshuffled and roll_k):
                # Rolled ping-pong loop. Two K tiles per iteration, which keeps
                # the ping/pong alternation compile-time inside the body while
                # the loop itself rolls; the pairs that have no successor to
                # prefetch are peeled off the end and walked unrolled, so the
                # rolled body needs no tail tests. The B tile and the first A
                # pack ride the loop, being produced one pair ahead of use.
                _st0 = pingpong_prologue()
                _pairs = max((num_k_tiles - 1) // 2, 0)
                _n_acc = len(accs)
                if const_expr(_pairs > 0):
                    for _it, _sv in fx.range(
                        0,
                        _pairs,
                        1,
                        init=list(accs) + _flatten_pp(_st0),
                    ):
                        _accs = [Vector(v) for v in _sv[:_n_acc]]
                        _cur = _unflatten_pp(_sv[_n_acc:])
                        _accs, _cur = pingpong_pair(_accs, _it * 2, _cur, steady=True)
                        _res = yield list(_accs) + _flatten_pp(_cur)
                    accs = [Vector(v) for v in _res[:_n_acc]]
                    _st0 = _unflatten_pp(_res[_n_acc:])
                for _kp in range_constexpr(2 * _pairs, num_k_tiles, 2):
                    accs, _st0 = pingpong_pair(accs, _kp, _st0)
            elif const_expr(b_preshuffled):
                _st0 = pingpong_prologue()
                for _kp in range_constexpr(0, num_k_tiles, 2):
                    accs, _st0 = pingpong_pair(accs, _kp, _st0)
            elif const_expr(roll_k):
                # Rolled K loop. The body is traced once, so it has to live here
                # rather than behind a helper: only the kernel function's own
                # source is AST-rewritten, and fx.range/yield is a rewritten
                # construct. The pipeline is the same as the unrolled form --
                # the next tile's global loads are issued before the current
                # tile computes -- which is why the prefetched registers ride
                # the loop next to the accumulators. The final tile is peeled,
                # having no successor to prefetch.
                _a_regs = prefetch_a_tile(0)
                _b_regs = prefetch_b_tile(0)
                _n_acc = len(accs)
                _n_a = len(_a_regs)
                if const_expr(k_grouped):
                    # ceil(K_g / tile_k), a runtime value. A bound of zero or
                    # less simply runs no iterations.
                    _kt_n = (
                        fx.Index(arith.subi(k_end_i32, k_start_i32))
                        + fx.Index(tile_k - 1)
                    ) // fx.Index(tile_k)
                    _main = _kt_n - fx.Index(1)
                    _run_main = True
                else:
                    _main = num_k_tiles - 1
                    _run_main = num_k_tiles > 1
                if _run_main:
                    for _kt, _st in fx.range(
                        0,
                        _main,
                        1,
                        init=list(accs) + list(_a_regs) + list(_b_regs),
                    ):
                        _accs = [Vector(v) for v in _st[:_n_acc]]
                        _a_cur = [Vector(v) for v in _st[_n_acc : _n_acc + _n_a]]
                        _b_cur = [Vector(v) for v in _st[_n_acc + _n_a :]]
                        store_a_tile_to_lds(_a_cur, lds_base_pong)
                        store_b_tile_to_lds(_b_cur, lds_base_b)
                        _scales = prefetch_scales(_kt)
                        gpu.barrier()
                        _a_nxt = prefetch_a_tile(_kt + 1)
                        _b_nxt = prefetch_b_tile(_kt + 1)
                        _b_tile = load_b_tile_from_lds(lds_base_b)
                        _accs = compute_tile(
                            _accs, _kt, lds_base_pong, _b_tile, _scales
                        )
                        gpu.barrier()
                        _res = yield list(_accs) + list(_a_nxt) + list(_b_nxt)
                    accs = [Vector(v) for v in _res[:_n_acc]]
                    _a_regs = [Vector(v) for v in _res[_n_acc : _n_acc + _n_a]]
                    _b_regs = [Vector(v) for v in _res[_n_acc + _n_a :]]
                # The peeled tile. Clamped so an empty group still names a
                # real tile; every one of its loads is masked off regardless.
                if const_expr(k_grouped):
                    _last = fx.Index(
                        arith.maxsi(
                            arith.index_cast(T.i32, _kt_n - fx.Index(1)),
                            _i32k(0),
                        )
                    )
                else:
                    _last = num_k_tiles - 1
                store_a_tile_to_lds(_a_regs, lds_base_pong)
                store_b_tile_to_lds(_b_regs, lds_base_b)
                _scales = prefetch_scales(_last)
                gpu.barrier()
                _b_tile = load_b_tile_from_lds(lds_base_b)
                accs = compute_tile(accs, _last, lds_base_pong, _b_tile, _scales)
                gpu.barrier()
            else:
                accs = run_kloop(accs)

            if const_expr(scaling == SCALING_ROW):
                # Rowwise scales are constant along K, so the whole reduction is
                # scaled here in one pass rather than per K tile.
                accs = make_rowwise_scaler(
                    m_wave_base=m_wave_base,
                    sa_rsrc=sa_rsrc,
                    sb_rsrc=sb_rsrc,
                    group_idx=group_idx,
                    n_in=n_in,
                    by_n=by_n,
                    n_tile_base=n_tile_base,
                    bx_m=bx_m_scale,
                    lane_mod_16=lane_mod_16,
                    lane_div_16=lane_div_16,
                    m_repeat=m_repeat,
                    num_acc_n=num_acc_n,
                    group_n_off=sb_group_off,
                )(accs)

            # ===== Epilogue: CShuffle vectorized stores =====
            c_n = n_in
            # The CShuffle epilogue lays CSHUFFLE_NLANE lanes across N, each
            # storing e_vec elements, and requires tile_n to divide by their
            # product. Take the wider store where it does.
            e_vec = 4 if (tile_n % (CSHUFFLE_NLANE * 4)) == 0 else 2

            write_row_to_lds, store_pair = make_epilogue_writers(
                accs=accs,
                d_rsrc=d_rsrc,
                out_mlir=out_mlir,
                e_vec=e_vec,
                c_n=c_n,
                n_padding=n_padding,
                n_bound=n_bound,
                d_group_off=d_group_off,
                d_row_base=d_row_base,
            )

            # Mask the partial-tile tail: skip stores for global rows at or beyond
            # the owning group's end. Returning (ctx, pred) lets the epilogue skip
            # the whole N-store loop for out-of-group rows.
            def precompute_row(*, row_local, row):
                row_i32 = arith.index_cast(T.i32, row)
                row_valid = arith.cmpi(
                    arith.CmpIPredicate.ult, row_i32, row_limit_out_i32
                )
                return (None, row_valid)

            mfma_epilog(
                use_cshuffle=True,
                m_wave_base=m_wave_base,
                arith=arith,
                vector=vector,
                gpu=gpu,
                scf=scf,
                range_constexpr=range_constexpr,
                tile_m=tile_m,
                tile_n=tile_n,
                e_vec=e_vec,
                block_size=total_threads,
                m_repeat=m_repeat,
                num_acc_n=num_acc_n,
                tx=tx,
                lane_div_16=lane_div_16,
                lane_mod_16=lane_mod_16,
                bx_m=bx_m_out,
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
    def launch_fp8_grouped_gemm(
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
        if const_expr(n_grouped):
            # One N-block enumeration across every group, with the same
            # host-known upper bound the packed M axis uses: each group wastes
            # at most one partial block, and surplus blocks match no group.
            gx = n_in // fx.Index(tile_n) + fx.Index(i32_num_groups)
        elif const_expr(n_padding):
            gx = (n_in + fx.Index(tile_n - 1)) // fx.Index(tile_n)
        else:
            gx = n_in // fx.Index(tile_n)
        gy = fx.Index(i32_num_m_tiles)  # M-tiles
        gz = (
            fx.Index(i32_num_groups)
            if const_expr(slab_layout or k_grouped)
            else fx.Index(1)
        )

        launcher = fp8_grouped_gemm_kernel(
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

    return launch_fp8_grouped_gemm
