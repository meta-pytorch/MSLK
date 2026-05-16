# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
# pyre-strict

"""Block score column-reduction utilities for NSA scoring phase.

During the fused scoring + compressed attention phase, the softmax warps
read QK scores from TMEM and reduce each column (KV block dimension) to
produce a single block importance score. These scores are accumulated
across all Q rows within a tile to produce per-block scores.

The column reduction reuses the TMEM read infrastructure from FA4's
softmax warps, adding a parallel reduction along the M dimension.
"""

from __future__ import annotations

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, const_expr
import cutlass.cute.nvgpu.tcgen05 as tcgen05

import mslk.fb.mslk.attention.flash_attn.utils as utils


@cute.jit
def reduce_s_to_block_scores(
    tSrS_t2r: cute.Tensor,  # Register fragment of S tile loaded from TMEM
    thr_tmem_load: cute.CopyAtom,
    thr_mma_qk: cute.core.ThrMma,
    m_block_size: int,
    n_block_size: int,
    block_scores: cute.Tensor,  # (n_blocks_in_tile,) accumulator in registers
    n_block_offset: Int32,  # Which compressed block this KV tile corresponds to
    n_blocks_per_tile: int,  # How many compressed blocks fit in one n_block_size tile
    is_first: bool,
) -> None:
    """Reduce S tile columns to block scores and accumulate.

    After QK GEMM produces S in TMEM, the softmax warps load it into
    registers. This function reduces each column of S (summing across
    the M dimension, i.e., all Q positions in the tile) to produce a
    single score per KV block.

    For compressed attention where compress_block_size >= n_block_size,
    each S tile column already corresponds to one compressed block, so
    we just sum the column. When compress_block_size < n_block_size,
    multiple compressed blocks share one S tile — we partition the
    columns accordingly.

    Args:
        tSrS_t2r: S scores in registers after TMEM load, shape depends on
            partition layout from thr_mma_qk.
        thr_tmem_load: TMEM load copy atom (for layout info).
        thr_mma_qk: QK MMA partition (for coordinate mapping).
        m_block_size: M tile size (e.g., 128).
        n_block_size: N tile size (e.g., 128).
        block_scores: Accumulator for block scores, shape (n_cmp,).
        n_block_offset: Starting compressed block index for this tile.
        n_blocks_per_tile: Number of compressed blocks per N tile.
        is_first: If True, initialize block_scores; otherwise accumulate.
    """
    # The S tile has been loaded into registers with layout from thr_mma_qk.
    # We need to reduce across the M dimension (rows) for each N position (column).
    #
    # tSrS_t2r has shape organized by the MMA partition. We interpret it as
    # (M_per_thread, N_per_thread) logically.
    #
    # For SM100 with 128-thread warp groups doing TMEM loads:
    # Each thread in the softmax warp group gets one row of the 128x128 S tile.
    # So each thread has n_block_size values (one full row).
    # The column reduction = warp-level reduction across threads (rows),
    # then store into block_scores.

    acc_S_mn = utils.make_acc_tensor_mn_view(tSrS_t2r)
    n_cols = cute.size(acc_S_mn, mode=[1])

    # Each thread contributes its row's values to the column sum.
    # We do a warp-level reduction for each column group.
    # Since each compressed block spans (compress_block_size / n_block_size * n_block_size)
    # columns within the tile, or if compress_block_size aligns with n_block_size,
    # just one column per block.

    # For the common case where n_blocks_per_tile == 1 (compress_block_size >= n_block_size),
    # sum all columns of this thread's row, then warp-reduce.
    if const_expr(n_blocks_per_tile == 1):
        row_sum = Float32(0.0)
        for c in cutlass.range(n_cols, unroll_full=True):
            row_sum += acc_S_mn[0, c].load()

        # Warp-level reduction: sum across all threads (rows in the tile)
        col_sum = utils.warp_reduce(row_sum, cute.arch.fadd, width=4)

        if is_first:
            block_scores[n_block_offset] = col_sum
        else:
            block_scores[n_block_offset] += col_sum
    else:
        # Multiple compressed blocks per N tile
        cols_per_block: int = const_expr(n_block_size // n_blocks_per_tile)
        for blk in cutlass.range_constexpr(n_blocks_per_tile):
            row_sum = Float32(0.0)
            for c in cutlass.range_constexpr(cols_per_block):
                col_idx: int = const_expr(blk * cols_per_block + c)
                row_sum += acc_S_mn[0, col_idx].load()

            col_sum = utils.warp_reduce(row_sum, cute.arch.fadd, width=4)

            global_blk_idx = n_block_offset + blk
            if is_first:
                block_scores[global_blk_idx] = col_sum
            else:
                block_scores[global_blk_idx] += col_sum


@cute.jit
def apply_causal_mask_to_block_scores(
    block_scores: cute.Tensor,  # (n_cmp,) block score accumulator
    n_cmp: Int32,
    m_block: Int32,  # Current Q tile index
    q_tile_size: int,
    compress_block_size: int,
) -> None:
    """Apply causal mask to block scores: future blocks get -inf.

    A compressed block j is "future" for query tile i if:
        j * compress_block_size >= (i + 1) * q_tile_size

    Args:
        block_scores: Block score accumulator, shape (n_cmp,).
        n_cmp: Total number of compressed blocks.
        m_block: Current Q tile index.
        q_tile_size: Size of each Q tile.
        compress_block_size: Size of each compressed block.
    """
    q_tile_end = (m_block + 1) * q_tile_size
    tidx = cute.arch.thread_idx()[0] % 128  # thread in correction warp group

    # Each thread checks its assigned blocks
    n_per_thread = (n_cmp + 127) // 128
    for i in cutlass.range(n_per_thread, unroll=1):
        blk_idx = tidx * n_per_thread + i
        if blk_idx < n_cmp:
            blk_start = blk_idx * compress_block_size
            if blk_start >= q_tile_end:
                block_scores[blk_idx] = -Float32.inf
