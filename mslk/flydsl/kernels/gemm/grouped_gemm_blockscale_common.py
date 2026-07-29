# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Shared building blocks for the grouped FP8 blockscale GEMM kernels.

Used by the grouped_gemm_blockscale contiguous and masked kernels. Holds the
parts of the two kernels that are byte-identical (parameter validation,
compile-time scalar constants, helper closures) so they live in one place.

scale_b is indexed as [num_groups, scale_k, scale_n] (per-group, per-K-block,
per-N-block); scale_a is [scale_k, M] (transposed, per-token per-K-block).
"""

from collections import namedtuple

import flydsl.expr as fx
from flydsl._mlir.dialects import math as math_dialect
from flydsl.expr import arith, buffer_ops, gpu, range_constexpr, rocdl, vector
from flydsl.expr.arith import ArithValue
from flydsl.expr.typing import T, Vector
from mslk.flydsl.kernels.mma.mfma_preshuffle_pipeline import (
    crd2idx,
    lds_store_16b_xor16,
    load_b_pack_k32,
    make_preshuffle_b_layout,
    swizzle_xor16,
    tile_chunk_coord_i32,
)

CompileConstants = namedtuple(
    "CompileConstants",
    [
        "total_threads",
        "elem_bytes",
        "num_k_tiles",
        "scale_k",
        "scale_n",
        "sb_per_tile",
        "k_unroll",
        "kpack_bytes",
        "tile_k_bytes",
        "tile_k_dwords",
        "bytes_a_per_tile",
        "bytes_per_thread_a",
        "a_load_bytes",
        "chunk_i32_a",
        "num_a_loads",
        "chunk_i32_b",
        "num_b_loads",
    ],
)


def validate_params(
    *, n, k, tile_n, tile_k, scale_block_k, scale_block_n, out_dtype, blockscale=False
):
    """Validate the divisibility constraints and out_dtype choice shared by
    the grouped GEMM kernels.

    scale_block_k splits a K tile into the sub-blocks the MFMA schedule steps
    through, so tile_k must cover a whole number of them under either scaling
    scheme -- a tile_k below scale_block_k yields zero sub-blocks and a compute
    loop that never runs.

    scale_block_n only matters to block scaling, where a tile must not straddle
    a scale block. Rowwise scaling carries one scale per row of A and per column
    of B and applies it in the epilogue, so tile_n is free of that alignment and
    may be smaller than scale_block_n.
    """
    if k % tile_k != 0:
        raise ValueError(f"k ({k}) must be divisible by tile_k ({tile_k})")
    if n % tile_n != 0:
        raise ValueError(f"n ({n}) must be divisible by tile_n ({tile_n})")
    if tile_k % scale_block_k != 0:
        raise ValueError(
            f"tile_k ({tile_k}) must be divisible by scale_block_k ({scale_block_k})"
        )
    if blockscale and tile_n % scale_block_n != 0:
        raise ValueError(
            f"tile_n ({tile_n}) must be divisible by scale_block_n ({scale_block_n})"
        )
    if out_dtype not in ("bf16", "f16"):
        raise ValueError(f"out_dtype must be 'bf16' or 'f16', got {out_dtype!r}")


# Conservative default for architectures missing from FlyDSL's capacity table.
_LDS_CAPACITY_FALLBACK_BYTES = 64 * 1024


def lds_capacity_bytes(arch=None):
    """LDS bytes available to one workgroup on ``arch``.

    This is arch-dependent and the difference is large: CDNA3 (gfx942/MI300) has
    64 KiB while CDNA4 (gfx950/MI350) has 160 KiB. Sourced from FlyDSL's
    SMEM_CAPACITY_MAP so the limit stays in sync with the compiler that enforces
    it; unknown architectures fall back to the conservative 64 KiB.
    """
    try:
        from flydsl.utils.smem_allocator import SMEM_CAPACITY_MAP
    except Exception:
        return _LDS_CAPACITY_FALLBACK_BYTES
    return SMEM_CAPACITY_MAP.get(str(arch), _LDS_CAPACITY_FALLBACK_BYTES)


def _check_lds_budget(*, variant, total, detail, tile_m, tile_n, tile_k, arch):
    """Raise ValueError if ``total`` LDS bytes overflow the device capacity.

    Must run BEFORE the kernel is traced: the compiler backend reports an LDS
    overflow as a hard error that aborts the process, which an autotuner cannot
    catch and skip.
    """
    capacity = lds_capacity_bytes(arch)
    if total > capacity:
        raise ValueError(
            f"{variant} LDS budget {total} bytes exceeds {capacity} on "
            f"{arch or 'unknown arch'} ({detail}) for tile_m={tile_m} "
            f"tile_n={tile_n} tile_k={tile_k}. Reduce tile_m/tile_n/tile_k."
        )


def validate_lds_budget_preshuffle(*, tile_m, tile_n, tile_k, elem_bytes=1, arch=None):
    """Check the preshuffle-B kernel's LDS budget fits in one workgroup's LDS.

    B is loaded HBM->registers here, so only the ping-pong A tiles occupy LDS
    during the K-loop; the CShuffle epilogue output aliases that same arena.
    Budget = max(A ping-pong, epilogue out).
    """
    lds_a_bytes = 2 * tile_m * tile_k * elem_bytes  # ping-pong A
    lds_out_bytes = tile_m * tile_n * 2  # bf16/f16 epilogue output, aliases base
    _check_lds_budget(
        variant="preshuffle-B",
        total=max(lds_a_bytes, lds_out_bytes),
        detail=f"A ping-pong {lds_a_bytes}, epilogue {lds_out_bytes}",
        tile_m=tile_m,
        tile_n=tile_n,
        tile_k=tile_k,
        arch=arch,
    )


def validate_lds_budget_plain(
    *, tile_m, tile_n, tile_k, elem_bytes=1, b_pingpong=False, arch=None
):
    """Check the plain-B kernel's LDS budget fits in one workgroup's LDS.

    The plain-B kernel stages BOTH A (ping-pong) and B through LDS during the
    K-loop, so they coexist; the epilogue output aliases that same arena (it
    runs after the final barrier). Budget = max(K-loop staging, epilogue out).
    Raises ValueError if a tile config would overflow LDS.
    """
    lds_a_bytes = 2 * tile_m * tile_k * elem_bytes  # ping-pong A
    b_buffers = 2 if b_pingpong else 1
    lds_b_bytes = b_buffers * tile_n * tile_k * elem_bytes
    lds_out_bytes = tile_m * tile_n * 2  # bf16/f16 epilogue output, aliases base
    kloop_bytes = lds_a_bytes + lds_b_bytes
    _check_lds_budget(
        variant="plain-B",
        total=max(kloop_bytes, lds_out_bytes),
        detail=(
            f"A ping-pong {lds_a_bytes} + B {lds_b_bytes} = {kloop_bytes}, "
            f"epilogue {lds_out_bytes}, b_pingpong={b_pingpong}"
        ),
        tile_m=tile_m,
        tile_n=tile_n,
        tile_k=tile_k,
        arch=arch,
    )


def out_mlir_for(out_dtype):
    """Return a zero-arg callable that yields the MLIR element type for the
    chosen output dtype. Matches the original local `out_mlir` lambda exactly
    so MLIR emission is unchanged."""
    return lambda: T.bf16 if out_dtype == "bf16" else T.f16


def compute_compile_constants(
    *, n, k, tile_m, tile_n, tile_k, scale_block_k, scale_block_n
):
    """Compute the compile-time scalar constants shared by both kernels.

    Returns a `CompileConstants` namedtuple. Pure-Python — no MLIR ops emitted.
    """
    total_threads = 256
    elem_bytes = 1  # FP8
    num_k_tiles = k // tile_k
    scale_k = k // scale_block_k
    scale_n = n // scale_block_n
    sb_per_tile = tile_k // scale_block_k  # scale blocks per K-tile
    k_unroll = tile_k // 64  # K64-byte micro-steps (for K32 MFMA pairs)
    kpack_bytes = 16  # 16-byte packs for FP8

    tile_k_bytes = tile_k * elem_bytes
    tile_k_dwords = tile_k_bytes // 4
    bytes_a_per_tile = tile_m * tile_k * elem_bytes
    bytes_per_thread_a = bytes_a_per_tile // total_threads
    a_load_bytes = 16  # 16-byte loads (dwordx4)
    chunk_i32_a = a_load_bytes // 4  # 4 dwords per load
    num_a_loads = bytes_per_thread_a // a_load_bytes

    # Plain-B staging (non-preshuffle kernel): B tile is [tile_n, tile_k],
    # loaded HBM->LDS just like A but with N in place of M.
    bytes_b_per_tile = tile_n * tile_k * elem_bytes
    bytes_per_thread_b = bytes_b_per_tile // total_threads
    chunk_i32_b = a_load_bytes // 4  # same 16-byte dwordx4 load
    num_b_loads = bytes_per_thread_b // a_load_bytes

    return CompileConstants(
        total_threads=total_threads,
        elem_bytes=elem_bytes,
        num_k_tiles=num_k_tiles,
        scale_k=scale_k,
        scale_n=scale_n,
        sb_per_tile=sb_per_tile,
        k_unroll=k_unroll,
        kpack_bytes=kpack_bytes,
        tile_k_bytes=tile_k_bytes,
        tile_k_dwords=tile_k_dwords,
        bytes_a_per_tile=bytes_a_per_tile,
        bytes_per_thread_a=bytes_per_thread_a,
        a_load_bytes=a_load_bytes,
        chunk_i32_a=chunk_i32_a,
        num_a_loads=num_a_loads,
        chunk_i32_b=chunk_i32_b,
        num_b_loads=num_b_loads,
    )


def setup_lds_allocation(*, allocator, tile_m, tile_k, tile_n, elem_bytes):
    """Reserve LDS for ping-pong A tiles and the CShuffle epilogue output.

    The ping-pong A buffers and the FP16/BF16 epilogue output share the same
    LDS arena (alias), so we reserve the max of the two. Returns
    `(lds_alloc_offset, lds_tile_elems)` where `lds_tile_elems` is the
    A-element stride between the ping and pong halves.
    """
    lds_a_bytes = tile_m * tile_k * elem_bytes
    lds_pingpong_bytes = 2 * lds_a_bytes
    lds_out_bytes = tile_m * tile_n * 2  # bf16/f16 = 2 bytes per element
    lds_total_bytes = max(lds_pingpong_bytes, lds_out_bytes)
    lds_alloc_offset = allocator._align(allocator.ptr, 16)
    allocator.ptr = lds_alloc_offset + lds_total_bytes
    lds_tile_elems = tile_m * tile_k  # element offset between ping and pong
    return lds_alloc_offset, lds_tile_elems


def setup_lds_allocation_plain(
    *, allocator, tile_m, tile_n, tile_k, elem_bytes, b_pingpong=False
):
    """Reserve LDS for the plain-B kernel: ping-pong A + (single/ping-pong) B +
    aliased CShuffle epilogue output.

    Unlike the preshuffle kernel (which loads B straight to registers and needs
    no B LDS), plain B is staged HBM->LDS->registers alongside A, so A and B
    LDS coexist during the K-loop. The epilogue output aliases the whole arena
    (offset 0) since it runs after the final K-loop barrier.

    Returns `(lds_alloc_offset, lds_tile_elems, lds_b_offset_elems)` where:
      - `lds_alloc_offset` is the byte base of the arena (A ping half at 0),
      - `lds_tile_elems` is the A ping<->pong element stride (= tile_m*tile_k),
      - `lds_b_offset_elems` is the element offset (from arena base) to the B
        buffer, i.e. just past the A ping-pong region.
    """
    lds_a_elems = tile_m * tile_k
    lds_a_pingpong_elems = 2 * lds_a_elems
    b_buffers = 2 if b_pingpong else 1
    lds_b_elems = b_buffers * tile_n * tile_k
    kloop_elems = lds_a_pingpong_elems + lds_b_elems  # FP8: 1 byte/elem
    lds_out_bytes = tile_m * tile_n * 2
    lds_total_bytes = max(kloop_elems * elem_bytes, lds_out_bytes)
    lds_alloc_offset = allocator._align(allocator.ptr, 16)
    allocator.ptr = lds_alloc_offset + lds_total_bytes
    lds_tile_elems = lds_a_elems
    lds_b_offset_elems = lds_a_pingpong_elems
    return lds_alloc_offset, lds_tile_elems, lds_b_offset_elems


def make_a_tile_loaders(
    *,
    a_rsrc,
    lds_a,
    layout_lds,
    bx_m,
    tx,
    tile_m,
    tile_k,
    tile_k_bytes,
    tile_k_dwords,
    chunk_i32_a,
    num_a_loads,
    total_threads,
    elem_bytes,
    k_in,
    m_in=None,
    group_idx=None,
):
    """Build the prefetch + LDS-store closures for the A tile.

    Returns `(prefetch_a_tile, store_a_tile_to_lds, a_row_local,
    a_col_local_i32, k_blocks16)`. When `m_in` and `group_idx` are both
    None (contig path) no group offset is emitted; when both are provided
    (masked path), `group_idx * m_in * (k_in/4)` is added as the leading
    term inside `prefetch_a_tile`, exactly matching the original masked
    code so the resulting MLIR (and ISA) is byte-identical. `k_blocks16`
    is returned for reuse by the downstream LDS-load helper.
    """
    layout_a_tile_div4 = fx.make_layout(
        (tile_m, tile_k_dwords), stride=(tile_k_dwords, 1)
    )
    c_chunk_a = fx.Index(chunk_i32_a)
    tx_i32_base = tx * c_chunk_a
    _k_div4_factor = k_in // fx.Index(4)
    if m_in is not None and group_idx is not None:
        a_tile_offset_div4 = group_idx * m_in * _k_div4_factor  # 3D A Offset
    else:
        a_tile_offset_div4 = None
    k_blocks16 = arith.index(tile_k_bytes // 16)
    c4_bytes = fx.Index(4)

    a_row_local = []
    a_col_local_i32 = []
    for i in range_constexpr(num_a_loads):
        row_local, col_local_i32 = tile_chunk_coord_i32(
            arith,
            tx_i32_base=tx_i32_base,
            i=i,
            total_threads=total_threads,
            layout_tile_div4=layout_a_tile_div4,
            chunk_i32=chunk_i32_a,
        )
        a_row_local.append(row_local)
        a_col_local_i32.append(col_local_i32)

    def prefetch_a_tile(k_tile_idx_py):
        """Load A tile from global memory into VGPRs."""
        base_k_div4 = fx.Index(k_tile_idx_py * tile_k_dwords)
        parts = []
        for i in range_constexpr(num_a_loads):
            row_global = bx_m + a_row_local[i]
            if a_tile_offset_div4 is None:
                idx_i32 = row_global * _k_div4_factor + base_k_div4 + a_col_local_i32[i]
            else:
                idx_i32 = (
                    a_tile_offset_div4
                    + row_global * _k_div4_factor
                    + base_k_div4
                    + a_col_local_i32[i]
                )
            a_vec = buffer_ops.buffer_load(a_rsrc, idx_i32, vec_width=4, dtype=T.i32)
            parts.append(Vector(a_vec).bitcast(fx.Int32))
        return parts

    def store_a_tile_to_lds(a_parts, lds_base):
        """Write prefetched A tile from VGPRs into LDS with XOR16 swizzle."""
        for i in range_constexpr(num_a_loads):
            lds_store_16b_xor16(
                arith,
                vector,
                lds_memref=lds_a,
                vec16_ty=T.f8x16,
                layout_lds=layout_lds,
                row_local=a_row_local[i],
                col_local_i32=a_col_local_i32[i],
                tx_c4=c4_bytes,
                k_blocks16=k_blocks16,
                lds_base=lds_base,
                vec_part_i32x4=a_parts[i],
                elem_bytes=elem_bytes,
            )

    return (
        prefetch_a_tile,
        store_a_tile_to_lds,
        a_row_local,
        a_col_local_i32,
        k_blocks16,
    )


def make_b_tile_loaders(
    *,
    b_rsrc,
    lds_b,
    layout_lds_b,
    by_n,
    group_idx,
    tx,
    tile_n,
    tile_k,
    tile_k_bytes,
    tile_k_dwords,
    chunk_i32_b,
    num_b_loads,
    total_threads,
    elem_bytes,
    n_in,
    k_in,
):
    """Build the prefetch + LDS-store closures for a PLAIN (non-preshuffled)
    B tile [tile_n, tile_k].

    Mirror of `make_a_tile_loaders` with N in place of M. B is `[G, N, K]`
    row-major, so the per-tile global base adds the group offset
    `group_idx * n_in * (k_in/4)` (always present — B is always grouped) plus
    the N-tile base `by_n` (the block's N-block start). Coalesced 16-byte
    (dwordx4) loads via `tile_chunk_coord_i32`; LDS store uses the same XOR16
    swizzle as A. Returns `(prefetch_b_tile, store_b_tile_to_lds, b_row_local,
    b_col_local_i32, k_blocks16_b)`.
    """
    layout_b_tile_div4 = fx.make_layout(
        (tile_n, tile_k_dwords), stride=(tile_k_dwords, 1)
    )
    c_chunk_b = fx.Index(chunk_i32_b)
    tx_i32_base = tx * c_chunk_b
    _k_div4_factor = k_in // fx.Index(4)
    # B is [G, N, K]: leading offset selects this tile's group and N-block base.
    b_tile_offset_div4 = group_idx * n_in * _k_div4_factor
    k_blocks16_b = arith.index(tile_k_bytes // 16)
    c4_bytes = fx.Index(4)

    b_row_local = []
    b_col_local_i32 = []
    for i in range_constexpr(num_b_loads):
        row_local, col_local_i32 = tile_chunk_coord_i32(
            arith,
            tx_i32_base=tx_i32_base,
            i=i,
            total_threads=total_threads,
            layout_tile_div4=layout_b_tile_div4,
            chunk_i32=chunk_i32_b,
        )
        b_row_local.append(row_local)
        b_col_local_i32.append(col_local_i32)

    def prefetch_b_tile(k_tile_idx_py):
        """Load plain B tile from global memory into VGPRs (coalesced dwordx4)."""
        base_k_div4 = fx.Index(k_tile_idx_py * tile_k_dwords)
        parts = []
        for i in range_constexpr(num_b_loads):
            row_global = by_n + b_row_local[i]  # global N row
            idx_i32 = (
                b_tile_offset_div4
                + row_global * _k_div4_factor
                + base_k_div4
                + b_col_local_i32[i]
            )
            b_vec = buffer_ops.buffer_load(b_rsrc, idx_i32, vec_width=4, dtype=T.i32)
            parts.append(Vector(b_vec).bitcast(fx.Int32))
        return parts

    def store_b_tile_to_lds(b_parts, lds_base):
        """Write prefetched B tile from VGPRs into LDS with XOR16 swizzle."""
        for i in range_constexpr(num_b_loads):
            lds_store_16b_xor16(
                arith,
                vector,
                lds_memref=lds_b,
                vec16_ty=T.f8x16,
                layout_lds=layout_lds_b,
                row_local=b_row_local[i],
                col_local_i32=b_col_local_i32[i],
                tx_c4=c4_bytes,
                k_blocks16=k_blocks16_b,
                lds_base=lds_base,
                vec_part_i32x4=b_parts[i],
                elem_bytes=elem_bytes,
            )

    return (
        prefetch_b_tile,
        store_b_tile_to_lds,
        b_row_local,
        b_col_local_i32,
        k_blocks16_b,
    )


def make_lds_loader(*, lds_a, layout_lds, k_blocks16):
    """Build the LDS-side A K64 pack loader.

    Returns `lds_load_packs_k64(curr_row_a_lds, col_base_bytes, lds_base)`
    which loads 16B from LDS with the XOR16 swizzle and returns the two
    i64 halves.
    """

    def lds_load_packs_k64(curr_row_a_lds, col_base_bytes, lds_base):
        col_base_swz_bytes = swizzle_xor16(curr_row_a_lds, col_base_bytes, k_blocks16)
        idx_a16 = crd2idx((curr_row_a_lds, col_base_swz_bytes), layout_lds) + lds_base
        loaded_a16 = Vector.load(T.vec(16, T.f8), lds_a, [idx_a16])
        a_i64x2 = loaded_a16.bitcast(fx.Int64)
        return a_i64x2[0], a_i64x2[1]

    return lds_load_packs_k64


def make_lds_b_loader(*, lds_b, layout_lds_b, k_blocks16_b):
    """Build the LDS-side plain-B K64 pack loader (mirror of `make_lds_loader`).

    Returns `lds_load_b_packs_k64(row_n_lds, col_base_bytes, lds_base_b)` which
    loads 16B from the B LDS buffer with the same XOR16 swizzle A uses and
    returns the two i64 halves — the exact fragment form the MFMA consumes for
    the B operand.
    """

    def lds_load_b_packs_k64(row_n_lds, col_base_bytes, lds_base_b):
        col_base_swz_bytes = swizzle_xor16(row_n_lds, col_base_bytes, k_blocks16_b)
        idx_b16 = crd2idx((row_n_lds, col_base_swz_bytes), layout_lds_b) + lds_base_b
        loaded_b16 = Vector.load(T.vec(16, T.f8), lds_b, [idx_b16])
        b_i64x2 = loaded_b16.bitcast(fx.Int64)
        return b_i64x2[0], b_i64x2[1]

    return lds_load_b_packs_k64


def make_plain_b_tile(
    *,
    lds_load_b_packs_k64,
    lane_mod_16,
    n_tile_base,
    col_offset_base_bytes,
    k_unroll,
    num_acc_n,
):
    """Build the plain-B tile assembler that reads B from LDS (mirror of the
    preshuffle `make_b_loader`, but sourcing from LDS instead of HBM).

    Returns `load_b_tile_from_lds(lds_base_b)` producing the SAME structure the
    preshuffle path did — a list of length `k_unroll` where each entry is
    `(packs0[ni], packs1[ni])` (two i64 halves per K64 micro-step, per N-acc) —
    so `make_compute_tile` consumes it unchanged.

    N-row addressing must match what the MFMA B operand expects, i.e. the same
    N-column `make_n_block_coords` uses (common.py):
        col = by_n + n_tile_base + ni*16 + lane_mod_16
    where `n_tile_base = wave_mod_4 * n_per_wave` is this WAVE's N sub-range.
    Since the B LDS buffer holds the block's [tile_n, tile_k] tile (row 0 = the
    block's `by_n`), the LDS N-row for accumulator `ni` is the tile-LOCAL row
    `n_tile_base + ni*16 + lane_mod_16` (by_n is the tile base, already 0 in the
    LDS-local frame). Missing `n_tile_base` gives the wrong N per wave.

    K addressing mirrors A exactly: per-pack column base is
    `col_offset_base_bytes + ku*64` (`col_offset_base_bytes = lane_div_16*16`).
    """

    def load_b_tile_from_lds(lds_base_b):
        b_tile = []
        for ku in range_constexpr(k_unroll):
            col_base_bytes = col_offset_base_bytes + fx.Index(ku * 64)
            packs0 = []
            packs1 = []
            for ni in range_constexpr(num_acc_n):
                row_n_lds = n_tile_base + (ni * 16) + lane_mod_16
                b0, b1 = lds_load_b_packs_k64(row_n_lds, col_base_bytes, lds_base_b)
                packs0.append(b0)
                packs1.append(b1)
            b_tile.append((packs0, packs1))
        return b_tile

    return load_b_tile_from_lds


def make_b_loader(
    *,
    arg_b,
    b_rsrc,
    layout_b,
    n_blk_list,
    n_intra_list,
    lane_div_16,
    kpack_bytes,
    elem_bytes,
    k_unroll,
    num_acc_n,
):
    """Build the B-tile loader closure.

    Returns `load_b_tile(base_k)` which loads all B packs for one K-tile,
    returning a list of length `k_unroll` where each entry is
    `(packs_half0[ni], packs_half1[ni])` for one K64 micro-step.
    """

    def load_b_pack(base_k, ki_step, ni):
        return load_b_pack_k32(
            buffer_ops,
            arith,
            vector,
            arg_b=arg_b,
            b_rsrc=b_rsrc,
            layout_b=layout_b,
            base_k=base_k,
            ki_step=ki_step,
            n_blk=n_blk_list[ni],
            n_intra=n_intra_list[ni],
            lane_div_16=lane_div_16,
            elem_type=T.f8,
            kpack_bytes=kpack_bytes,
            elem_bytes=elem_bytes,
        )

    def load_b_tile(base_k):
        b_tile = []
        for ku in range_constexpr(k_unroll):
            packs0 = []
            packs1 = []
            for ni in range_constexpr(num_acc_n):
                ki0 = (ku * 2) + 0
                ki1 = (ku * 2) + 1
                b0 = load_b_pack(base_k, ki0, ni)
                b1 = load_b_pack(base_k, ki1, ni)
                packs0.append(b0)
                packs1.append(b1)
            b_tile.append((packs0, packs1))
        return b_tile

    return load_b_tile


def pack_i64x4_to_i32x8(x0, x1, x2, x3):
    """Pack four i64 values into a single i32x8 vector via i64x4 bitcast.

    Used to assemble the K=128 MFMA A/B operands on gfx950.
    """
    v4 = Vector.from_elements([x0, x1, x2, x3], fx.Int64)
    return v4.bitcast(fx.Int32)


def make_hot_loop_scheduler(
    *,
    _use_hw_scale,
    sb_per_tile,
    m_repeat,
    num_acc_n,
    k_unroll,
    num_a_loads,
    ku_per_sb,
):
    """Build the per-tile sched_group_barrier scheduler closure.

    Emits the dsrd / mfma / vmem_rd / dswr group barriers in the order
    matching the MoE stage-2 pattern. Returns a zero-arg closure to be
    invoked once per K-tile body inside the ping-pong loop.
    """

    def hot_loop_scheduler():
        mfma_group = num_acc_n
        if _use_hw_scale:
            total_mfma = sb_per_tile * m_repeat * mfma_group
        else:
            total_mfma = k_unroll * m_repeat * mfma_group * 2
        rocdl.sched_group_barrier(rocdl.mask_dsrd, ku_per_sb * m_repeat, 0)
        rocdl.sched_group_barrier(rocdl.mask_mfma, total_mfma, 1)
        rocdl.sched_group_barrier(rocdl.mask_vmem_rd, num_a_loads, 2)
        rocdl.sched_group_barrier(rocdl.mask_dswr, num_a_loads, 3)
        rocdl.sched_barrier(0)

    return hot_loop_scheduler


def make_prefetch_scales(
    *,
    _use_hw_scale,
    sa_rsrc,
    sb_rsrc,
    group_idx,
    scale_n,
    scale_k,
    c_scale_k,
    n_block_for_scale,
    bx_m,
    lane_mod_16,
    m_in,
    sb_per_tile,
    m_repeat,
    num_acc_n,
    sa_group_off=None,
):
    """Build the cross-tile E8M0 scale prefetch closure (gfx950 HW path).

    Returns `prefetch_scales(k_tile_idx_py)` that returns
    `(sa_pf, sb_pf)` — outer index = sb (sb_per_tile), inner =
    m_repeat / num_acc_n. Returns None on the gfx942 SW path (where
    scales are loaded inside compute_tile instead).

    `sa_group_off` is None for the contig path (no addition emitted)
    and `group_idx * c_scale_k * m_in` for the masked path. Using a
    Python `is None` guard keeps the contig MLIR identical to the
    pre-extraction code.
    """

    def prefetch_scales(k_tile_idx_py):
        if not _use_hw_scale:
            return None
        sa_pf = []
        sb_pf = []
        # scale_b layout is [num_groups, scale_k, scale_n].
        sb_group_offset = group_idx * fx.Index(scale_k * scale_n)
        for sb in range_constexpr(sb_per_tile):
            kb = fx.Index(k_tile_idx_py * sb_per_tile + sb)
            if sa_group_off is None:
                sa_base_pf = kb * m_in
            else:
                sa_base_pf = sa_group_off + kb * m_in

            sa_sb = []
            for mi in range_constexpr(m_repeat):
                sa_row = bx_m + (mi * 16) + lane_mod_16
                sa_idx = sa_base_pf + sa_row
                sa_i8 = buffer_ops.buffer_load(sa_rsrc, sa_idx, vec_width=1, dtype=T.i8)
                sa_e8m0 = ArithValue(sa_i8).extui(T.i32)
                sa_sb.append(sa_e8m0)
            sa_pf.append(sa_sb)

            sb_sb = []
            for ni in range_constexpr(num_acc_n):
                sb_idx = (
                    sb_group_offset + kb * fx.Index(scale_n) + n_block_for_scale[ni]
                )
                sb_i8 = buffer_ops.buffer_load(sb_rsrc, sb_idx, vec_width=1, dtype=T.i8)
                sb_i32 = ArithValue(sb_i8).extui(T.i32)
                sb_e8m0 = rocdl.readfirstlane(T.i32, sb_i32)
                sb_sb.append(sb_e8m0)
            sb_pf.append(sb_sb)
        return (sa_pf, sb_pf)

    return prefetch_scales


def make_compute_tile(
    *,
    _use_hw_scale,
    _is_gfx950=False,
    lds_load_packs_k64,
    sa_rsrc,
    sb_rsrc,
    group_idx,
    scale_n,
    scale_k,
    c_scale_k,
    n_block_for_scale,
    bx_m,
    lane_mod_16,
    lane_div_16,
    m_in,
    sb_per_tile,
    m_repeat,
    num_acc_n,
    ku_per_sb,
    col_offset_base_bytes,
    mfma_res_ty,
    acc_init,
    sa_group_off=None,
    group_m_start=None,
    group_m_size=None,
    blockscale=False,
):
    """Build the per-K-tile compute closure.

    Returns `compute_tile(accs_in, k_tile_idx_py, lds_base, b_tile_in,
    scales_pf, *, a0_prefetch=None)` which advances the accumulators by
    one K-tile of MFMA work. `scales_pf` is the prefetched scales for
    the gfx950 HW path; None for the gfx942 SW path (which loads scales
    locally inside the closure).

    `sa_group_off` is None for the contig path (no addition emitted)
    and `group_idx * c_scale_k * m_in` for the masked path; only the
    gfx942 SW path uses it.

    Where the scales are applied follows from how they vary. Block scaling has a
    distinct scale per (scale_block_k x scale_block_n) block, so each block's
    partial sum needs its own factor and the scaling happens here, per block,
    inside the K loop. Rowwise scaling has one scale per row of A and one per
    column of B, both constant along K, so they factor out of the reduction and
    the K loop accumulates unscaled, leaving a single scaling in the epilogue
    (see `make_rowwise_scaler`).
    """

    def compute_tile(
        accs_in, k_tile_idx_py, lds_base, b_tile_in, scales_pf, *, a0_prefetch=None
    ):
        current_accs = list(accs_in)

        for sb in range_constexpr(sb_per_tile):
            kb = fx.Index(k_tile_idx_py * sb_per_tile + sb)

            s_a_vecs = []
            s_b_vals = []
            if blockscale and not _use_hw_scale:
                if group_m_start is not None:
                    # quantize_fp8_group(m_sizes=...) stores scale_a as per-group
                    # blocks: group g starts at M_start*scale_k and holds element
                    # (local_m, k_g) at local_m + k_g*M_g. This is NOT a global
                    # [scale_k, TotalM] transpose -- the two only coincide when
                    # there is one group or one K-block.
                    # sa_idx adds row_global (= M_start + local_m) below, so fold
                    # the -M_start into the base: M_start*(scale_k-1) + k_g*M_g.
                    sa_base = group_m_start * fx.Index(scale_k - 1) + kb * group_m_size
                elif sa_group_off is None:
                    sa_base = kb * m_in
                else:
                    sa_base = sa_group_off + kb * m_in
                row_off_base = lane_div_16 * fx.Index(4)
                for mi in range_constexpr(m_repeat):
                    s_a_row = []
                    for ii in range_constexpr(4):
                        row_in_tile = (mi * 16) + row_off_base + fx.Index(ii)
                        row_global = bx_m + row_in_tile
                        sa_idx = sa_base + row_global
                        s_a_val = buffer_ops.buffer_load(
                            sa_rsrc, sa_idx, vec_width=1, dtype=T.f32
                        )
                        s_a_row.append(s_a_val)
                    s_a_vec4 = Vector.from_elements(s_a_row, fx.Float32)
                    s_a_vecs.append(s_a_vec4)

                # scale_b layout is [num_groups, scale_k, scale_n]:
                # element (g, kb, n_blk) at g*scale_k*scale_n + kb*scale_n + n_blk.
                sb_group_offset = group_idx * fx.Index(scale_k * scale_n)
                for ni in range_constexpr(num_acc_n):
                    sb_idx = (
                        sb_group_offset + kb * fx.Index(scale_n) + n_block_for_scale[ni]
                    )
                    s_b_val = buffer_ops.buffer_load(
                        sb_rsrc, sb_idx, vec_width=1, dtype=T.f32
                    )
                    s_b_val = rocdl.readfirstlane(T.f32, s_b_val)
                    s_b_vals.append(s_b_val)

            if _use_hw_scale:
                sa_pf, sb_pf = scales_pf
                sa_e8m0_list = sa_pf[sb]
                sb_e8m0_list = sb_pf[sb]

                ku0 = sb * ku_per_sb
                ku1 = ku0 + 1
                b0_packs0, b0_packs1 = b_tile_in[ku0]
                b1_packs0, b1_packs1 = b_tile_in[ku1]
                col_base0 = col_offset_base_bytes + fx.Index(ku0 * 64)
                col_base1 = col_offset_base_bytes + fx.Index(ku1 * 64)

                for mi in range_constexpr(m_repeat):
                    curr_row_a_lds = lane_mod_16 + (mi * 16)
                    if a0_prefetch is not None and sb == 0 and mi == 0:
                        a0, a1 = a0_prefetch
                    else:
                        a0, a1 = lds_load_packs_k64(curr_row_a_lds, col_base0, lds_base)
                    a2, a3 = lds_load_packs_k64(curr_row_a_lds, col_base1, lds_base)
                    a128 = pack_i64x4_to_i32x8(a0, a1, a2, a3)

                    for ni in range_constexpr(num_acc_n):
                        b128 = pack_i64x4_to_i32x8(
                            b0_packs0[ni],
                            b0_packs1[ni],
                            b1_packs0[ni],
                            b1_packs1[ni],
                        )
                        acc_idx = mi * num_acc_n + ni
                        current_accs[acc_idx] = rocdl.mfma_scale_f32_16x16x128_f8f6f4(
                            mfma_res_ty,
                            [
                                a128,
                                b128,
                                current_accs[acc_idx],
                                0,
                                0,
                                0,
                                sa_e8m0_list[mi],
                                0,
                                sb_e8m0_list[ni],
                            ],
                        )
            elif _is_gfx950:
                # gfx950: use the wide 16x16x128 MFMA with a neutral E8M0 scale
                # (0x7F7F7F7F = no-op hardware scaling), which avoids the
                # 4x-narrower 16x16x32 MFMA of the path below.
                #
                # Block scaling accumulates a whole scale block into block_accs
                # and folds it into current_accs with the FP32 scales once per
                # block, keeping the VALU scale cost off the per-K-step path.
                # Rowwise scaling has nothing to fold in per block, so the MFMAs
                # accumulate straight into current_accs.
                if blockscale:
                    combined_scales = []
                    for mi in range_constexpr(m_repeat):
                        mi_combined = []
                        for ni in range_constexpr(num_acc_n):
                            s_b_bc = Vector.filled(
                                (4,), fx.Float32(s_b_vals[ni]), fx.Float32
                            )
                            mi_combined.append(
                                ArithValue(s_a_vecs[mi]) * ArithValue(s_b_bc)
                            )
                        combined_scales.append(mi_combined)
                    mfma_accs = [acc_init] * (num_acc_n * m_repeat)
                else:
                    mfma_accs = current_accs
                ku0 = sb * ku_per_sb
                ku1 = ku0 + 1
                b0_packs0, b0_packs1 = b_tile_in[ku0]
                b1_packs0, b1_packs1 = b_tile_in[ku1]
                col_base0 = col_offset_base_bytes + fx.Index(ku0 * 64)
                col_base1 = col_offset_base_bytes + fx.Index(ku1 * 64)

                for mi in range_constexpr(m_repeat):
                    curr_row_a_lds = lane_mod_16 + (mi * 16)
                    if a0_prefetch is not None and sb == 0 and mi == 0:
                        a0, a1 = a0_prefetch
                    else:
                        a0, a1 = lds_load_packs_k64(curr_row_a_lds, col_base0, lds_base)
                    a2, a3 = lds_load_packs_k64(curr_row_a_lds, col_base1, lds_base)
                    a128 = pack_i64x4_to_i32x8(a0, a1, a2, a3)

                    for ni in range_constexpr(num_acc_n):
                        b128 = pack_i64x4_to_i32x8(
                            b0_packs0[ni], b0_packs1[ni], b1_packs0[ni], b1_packs1[ni]
                        )
                        acc_idx = mi * num_acc_n + ni
                        mfma_accs[acc_idx] = rocdl.mfma_scale_f32_16x16x128_f8f6f4(
                            mfma_res_ty,
                            [
                                a128,
                                b128,
                                mfma_accs[acc_idx],
                                0,
                                0,
                                0,
                                0x7F7F7F7F,
                                0,
                                0x7F7F7F7F,
                            ],
                        )

                if blockscale:
                    for mi in range_constexpr(m_repeat):
                        for ni in range_constexpr(num_acc_n):
                            acc_idx = mi * num_acc_n + ni
                            current_accs[acc_idx] = math_dialect.fma(
                                mfma_accs[acc_idx],
                                combined_scales[mi][ni],
                                current_accs[acc_idx],
                            )
            else:
                for ku_local in range_constexpr(ku_per_sb):
                    ku = sb * ku_per_sb + ku_local
                    k_offset_bytes = ku * 64
                    b_packs0, b_packs1 = b_tile_in[ku]

                    for mi in range_constexpr(m_repeat):
                        if (
                            a0_prefetch is not None
                            and sb == 0
                            and ku_local == 0
                            and mi == 0
                        ):
                            a0, a1 = a0_prefetch
                        else:
                            row_a_lds = lane_mod_16 + (mi * 16)
                            col_a_base_bytes = lane_div_16 * fx.Index(16) + fx.Index(
                                k_offset_bytes
                            )
                            a0, a1 = lds_load_packs_k64(
                                row_a_lds, col_a_base_bytes, lds_base
                            )

                        for ni in range_constexpr(num_acc_n):
                            acc_idx = mi * num_acc_n + ni
                            mfma_fn = rocdl.mfma_f32_16x16x32_fp8_fp8

                            if blockscale:
                                # This block's partial sum starts from zero so it
                                # can be scaled on its own before joining the
                                # running accumulator.
                                mfma_mid = mfma_fn(
                                    T.f32x4, [a0, b_packs0[ni], acc_init, 0, 0, 0]
                                )
                                mfma_result = mfma_fn(
                                    T.f32x4, [a1, b_packs1[ni], mfma_mid, 0, 0, 0]
                                )

                                s_a_v4 = s_a_vecs[mi]
                                s_b_bc = Vector.filled(
                                    (4,), fx.Float32(s_b_vals[ni]), fx.Float32
                                )
                                scaled = ArithValue(mfma_result) * ArithValue(s_a_v4)
                                current_accs[acc_idx] = math_dialect.fma(
                                    scaled, s_b_bc, current_accs[acc_idx]
                                )
                            else:
                                # Nothing to scale per block, so chain the MFMAs
                                # straight onto the running accumulator.
                                mfma_mid = mfma_fn(
                                    T.f32x4,
                                    [a0, b_packs0[ni], current_accs[acc_idx], 0, 0, 0],
                                )
                                current_accs[acc_idx] = mfma_fn(
                                    T.f32x4, [a1, b_packs1[ni], mfma_mid, 0, 0, 0]
                                )

        return current_accs

    return compute_tile


def make_rowwise_scaler(
    *,
    sa_rsrc,
    sb_rsrc,
    group_idx,
    n_in,
    by_n,
    n_tile_base,
    bx_m,
    lane_mod_16,
    lane_div_16,
    m_repeat,
    num_acc_n,
):
    """Build the closure that applies rowwise scales to the finished accumulators.

    Rowwise scaling multiplies each output element by ``scale_a[row]`` and
    ``scale_b[group, col]``, both constant along K, so this runs once after the
    K loop rather than inside it.

    The MFMA 16x16 C fragment holds four f32 per lane that share a column and
    differ by row, which is exactly how the two scales vary: scale_b contributes
    one value broadcast across the fragment, and scale_a contributes a vector of
    the fragment's four rows. Unlike the block-scale lookup, the column here
    includes ``lane_mod_16``, so scale_b is not wave-uniform and must not be
    routed through readfirstlane.

    Returns ``apply_rowwise_scales(accs)``.
    """

    def apply_rowwise_scales(accs):
        scaled = list(accs)

        # scale_a is [total_M]: one value per global row.
        row_off_base = lane_div_16 * fx.Index(4)
        s_a_vecs = []
        for mi in range_constexpr(m_repeat):
            s_a_row = []
            for ii in range_constexpr(4):
                row_global = bx_m + (mi * 16) + row_off_base + fx.Index(ii)
                s_a_row.append(
                    buffer_ops.buffer_load(
                        sa_rsrc, row_global, vec_width=1, dtype=T.f32
                    )
                )
            s_a_vecs.append(Vector.from_elements(s_a_row, fx.Float32))

        # scale_b is [num_groups, n]: element (g, col) at g*n + col.
        group_n_off = group_idx * n_in
        s_b_bcs = []
        for ni in range_constexpr(num_acc_n):
            col_global = group_n_off + by_n + n_tile_base + (ni * 16) + lane_mod_16
            s_b_val = buffer_ops.buffer_load(
                sb_rsrc, col_global, vec_width=1, dtype=T.f32
            )
            s_b_bcs.append(Vector.filled((4,), fx.Float32(s_b_val), fx.Float32))

        for mi in range_constexpr(m_repeat):
            for ni in range_constexpr(num_acc_n):
                acc_idx = mi * num_acc_n + ni
                v = ArithValue(scaled[acc_idx]) * ArithValue(s_a_vecs[mi])
                scaled[acc_idx] = v * ArithValue(s_b_bcs[ni])
        return scaled

    return apply_rowwise_scales


def make_kloop_plain(
    *,
    num_k_tiles,
    tile_k,
    prefetch_a_tile,
    store_a_tile_to_lds,
    prefetch_b_tile,
    store_b_tile_to_lds,
    load_b_tile_from_lds,
    prefetch_scales,
    compute_tile,
    lds_base_pong,
    lds_base_b,
):
    """Single-LDS-buffer K-loop for the plain-B kernel.

    Plain B is staged HBM->LDS->registers each K-tile, alongside A. A and B each
    get one LDS buffer rather than the ping-pong pair `make_pingpong_kloop` uses,
    which keeps the LDS budget within reach at wide tile_n but costs two barriers
    per K-tile: one after the stores, and one at the end of the iteration before
    the buffers are overwritten. Global-load latency is still overlapped, by
    issuing the next tile's HBM loads into registers before computing the current
    one. `lds_base_pong` is reused as the single A buffer base.
    """

    def run_kloop(accs):
        if num_k_tiles == 0:
            return accs

        a_regs = prefetch_a_tile(0)
        b_regs = prefetch_b_tile(0)
        for kt in range_constexpr(num_k_tiles):
            # Publish this tile's A/B (already in VGPRs) to LDS.
            store_a_tile_to_lds(a_regs, lds_base_pong)
            store_b_tile_to_lds(b_regs, lds_base_b)
            scales_pf = prefetch_scales(kt)
            gpu.barrier()

            # Issue next tile's HBM loads NOW so they overlap the compute below.
            if kt + 1 < num_k_tiles:
                a_regs = prefetch_a_tile(kt + 1)
                b_regs = prefetch_b_tile(kt + 1)

            # Read B fragment from LDS, compute the tile.
            b_tile = load_b_tile_from_lds(lds_base_b)
            accs = compute_tile(accs, kt, lds_base_pong, b_tile, scales_pf)
            # Barrier before next iter overwrites the shared A/B buffers.
            gpu.barrier()
        return accs

    return run_kloop


def make_pingpong_kloop(
    *,
    num_k_tiles,
    tile_k,
    prefetch_a_tile,
    store_a_tile_to_lds,
    load_b_tile,
    prefetch_scales,
    compute_tile,
    hot_loop_scheduler,
    lds_load_packs_k64,
    lds_base_pong,
    lds_base_ping,
    row_a_lds_base,
    col_offset_base_bytes,
):
    """Build the ping-pong K-loop driver.

    Returns `run_kloop(accs)` which advances `accs` through all
    K-tiles using the prologue + alternating ping/pong stages.
    Loop body is byte-identical between contig and masked, so this
    factory has no offset parameters.
    """

    def run_kloop(accs):
        # Prologue: prefetch first A tile into VGPRs, store to LDS, load B + scales
        a_regs0 = prefetch_a_tile(0)
        store_a_tile_to_lds(a_regs0, lds_base_pong)
        b_tile_pong = load_b_tile(fx.Index(0))
        scales_pong_pf = prefetch_scales(0)
        gpu.barrier()

        # Prefetch first A pack from pong (hides LDS latency behind upcoming VMEM)
        a0_prefetch_pong = lds_load_packs_k64(
            row_a_lds_base, col_offset_base_bytes, lds_base_pong
        )

        for k_pair in range_constexpr(0, num_k_tiles, 2):
            # Prefetch the next scales before the B-tile VMEM so the scale-load
            # latency hides behind it; then the A+B registers.
            if k_pair + 1 < num_k_tiles:
                scales_ping_pf = prefetch_scales(k_pair + 1)
                a_regs_ping = prefetch_a_tile(k_pair + 1)
                b_tile_ping = load_b_tile(fx.Index((k_pair + 1) * tile_k))

            # Compute current tile from pong LDS
            accs = compute_tile(
                accs,
                k_pair,
                lds_base_pong,
                b_tile_pong,
                scales_pong_pf,
                a0_prefetch=a0_prefetch_pong,
            )
            a0_prefetch_pong = None

            # Store next A to LDS (ds_write after compute, overlaps with trailing MFMAs)
            if k_pair + 1 < num_k_tiles:
                store_a_tile_to_lds(a_regs_ping, lds_base_ping)
                hot_loop_scheduler()
            gpu.barrier()

            if k_pair + 1 < num_k_tiles:
                # Prefetch first A pack from ping
                a0_prefetch_ping = lds_load_packs_k64(
                    row_a_lds_base, col_offset_base_bytes, lds_base_ping
                )

                # Prefetch next scales + A+B
                if k_pair + 2 < num_k_tiles:
                    scales_pong_pf = prefetch_scales(k_pair + 2)
                    a_regs_pong = prefetch_a_tile(k_pair + 2)
                    b_tile_pong = load_b_tile(fx.Index((k_pair + 2) * tile_k))

                # Compute current tile from ping LDS
                accs = compute_tile(
                    accs,
                    k_pair + 1,
                    lds_base_ping,
                    b_tile_ping,
                    scales_ping_pf,
                    a0_prefetch=a0_prefetch_ping,
                )
                a0_prefetch_ping = None

                # Store next A to LDS
                if k_pair + 2 < num_k_tiles:
                    store_a_tile_to_lds(a_regs_pong, lds_base_pong)
                    hot_loop_scheduler()
                gpu.barrier()

                # Prefetch first A pack from pong for next iteration
                if k_pair + 2 < num_k_tiles:
                    a0_prefetch_pong = lds_load_packs_k64(
                        row_a_lds_base, col_offset_base_bytes, lds_base_pong
                    )

        return accs

    return run_kloop


def make_epilogue_writers(
    *,
    accs,
    d_rsrc,
    out_mlir,
    e_vec,
    c_n,
    d_group_off=None,
):
    """Build the CShuffle-epilogue writer closures.

    Returns `(write_row_to_lds, store_pair)` to be passed to
    `mfma_epilog`. `d_group_off` is None for the contig path (no
    addition emitted) and `group_idx * m_in * n_in` for the masked
    path. Using a Python `is None` guard keeps the contig MLIR
    identical to the pre-extraction code.
    """

    def write_row_to_lds(
        *,
        mi,
        ii,
        row_in_tile,
        row,
        row_base_lds,
        col_base_local,
        num_acc_n,
        lds_out,
    ):
        for ni in range_constexpr(num_acc_n):
            col_local = col_base_local + (ni * 16)
            acc_idx = mi * num_acc_n + ni
            acc = accs[acc_idx]
            val = Vector(acc)[ii]
            v_out = arith.trunc_f(out_mlir(), val)
            lds_idx = row_base_lds + col_local
            v1 = Vector.from_elements([v_out])
            v1.store(lds_out, [lds_idx], alignment=2)

    def store_pair(*, row_local, row, row_ctx, col_pair0, col_g0, frag):
        if d_group_off is None:
            idx_out = row * c_n + col_g0
        else:
            idx_out = d_group_off + row * c_n + col_g0
        byte_off = idx_out * 2
        if e_vec == 4:
            frag_i32x2 = Vector(frag).bitcast(fx.Int32)
            buffer_ops.buffer_store(frag_i32x2, d_rsrc, byte_off, offset_is_bytes=True)
        else:
            frag_i32x1 = Vector(frag).bitcast(fx.Int32)
            frag_i32 = frag_i32x1[0]
            buffer_ops.buffer_store(frag_i32, d_rsrc, byte_off, offset_is_bytes=True)

    return write_row_to_lds, store_pair


MfmaTilingConstants = namedtuple(
    "MfmaTilingConstants",
    ["m_repeat", "num_waves", "n_per_wave", "num_acc_n", "num_accs"],
)


def compute_mfma_tiling(*, tile_m, tile_n):
    """Pure-Python derivation of the MFMA tiling constants.

    Returns an `MfmaTilingConstants` namedtuple with `m_repeat`,
    `num_waves`, `n_per_wave`, `num_acc_n`, `num_accs`. Emits no MLIR.
    """
    m_repeat = tile_m // 16  # 8 for tile_m=128
    num_waves = 4
    n_per_wave = tile_n // num_waves  # 32 for tile_n=128
    num_acc_n = n_per_wave // 16  # 2 for n_per_wave=32
    num_accs = m_repeat * num_acc_n
    return MfmaTilingConstants(
        m_repeat=m_repeat,
        num_waves=num_waves,
        n_per_wave=n_per_wave,
        num_acc_n=num_acc_n,
        num_accs=num_accs,
    )


def init_accumulators(num_accs):
    """Emit the FP32 zero-vector accumulator constant and replicate it
    for all `num_accs` MFMA result slots. Returns `(acc_init, accs)`."""
    acc_init = arith.constant_vector(0.0, T.f32x4)
    accs = [acc_init] * num_accs
    return acc_init, accs


NBlockCoords = namedtuple(
    "NBlockCoords",
    [
        "n_tile_base",
        "n_block_for_scale",
        "layout_b",
        "n_blk_list",
        "n_intra_list",
        "c_scale_k",
    ],
)


def make_n_block_coords(
    *,
    wave_id,
    by_n,
    group_idx,
    num_groups_in,
    n_in,
    k_in,
    lane_mod_16,
    kpack_bytes,
    elem_bytes,
    scale_block_n,
    scale_k,
    n_per_wave,
    num_acc_n,
):
    """Compute per-wave N-tile base, scale_b N-block indices, the
    preshuffle B layout, and the per-MFMA (n_blk, n_intra) coordinate
    lists for all groups concatenated along N.

    Byte-identical between contig and masked. Returns an `NBlockCoords`
    namedtuple matching the original local variable names so the caller
    can keep referring to them unchanged.
    """
    wave_mod_4 = wave_id % fx.Index(4)
    n_tile_base = wave_mod_4 * fx.Index(n_per_wave)

    c_scale_block_n = fx.Index(scale_block_n)
    c_scale_k = fx.Index(scale_k)
    n_block_for_scale = []
    for ni in range_constexpr(num_acc_n):
        col_base = by_n + n_tile_base + (ni * 16)
        n_blk = col_base // c_scale_block_n
        n_block_for_scale.append(n_blk)

    c_n_total = num_groups_in * n_in
    b_layout = make_preshuffle_b_layout(
        arith,
        c_n=c_n_total,
        c_k=k_in,
        kpack_bytes=kpack_bytes,
        elem_bytes=elem_bytes,
    )
    layout_b = b_layout.layout_b

    c_n0 = c_n_total // fx.Index(16)
    c_n0_i32 = arith.index_cast(T.i32, c_n0)
    layout_n_blk_intra = fx.make_layout((c_n0_i32, 16), stride=(16, 1))
    n_blk_list = []
    n_intra_list = []
    group_n_off = group_idx * n_in
    for ni in range_constexpr(num_acc_n):
        col_global = group_n_off + by_n + n_tile_base + (ni * 16) + lane_mod_16
        coord_ni = fx.idx2crd(fx.Int32(col_global), layout_n_blk_intra)
        n_blk_list.append(fx.get(coord_ni, 0))
        n_intra_list.append(fx.get(coord_ni, 1))

    return NBlockCoords(
        n_tile_base=n_tile_base,
        n_block_for_scale=n_block_for_scale,
        layout_b=layout_b,
        n_blk_list=n_blk_list,
        n_intra_list=n_intra_list,
        c_scale_k=c_scale_k,
    )
