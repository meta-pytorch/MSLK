# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

"""Block scoring and top-k selection for NSA."""

from __future__ import annotations

import math
from typing import Type

import torch
from torch import Tensor


def score_and_select_blocks(
    Q: Tensor,  # (B, N, H, D)
    K_cmp: Tensor,  # (B, N_cmp, H_kv, D)
    num_selected_blocks: int,
    compress_block_size: int,
    causal: bool = True,
    q_tile_size: int = 256,
    softmax_scale: float | None = None,
) -> Tensor:
    """Score KV blocks by importance and select top-k for each query tile.

    Uses compressed keys to efficiently compute block importance scores.
    Each query tile (q_tile_size positions) independently selects its own
    top-k KV blocks.

    Processes query tiles in chunks to avoid materializing a full (B, H, N, N_cmp)
    score tensor, reducing peak memory from O(N * N_cmp) to
    O(chunk_tiles * q_tile_size * N_cmp).

    Args:
        Q: Query tensor, shape (B, N, H, D).
        K_cmp: Compressed key tensor, shape (B, N_cmp, H_kv, D).
        num_selected_blocks: Number of KV blocks to select per query tile.
        compress_block_size: Size of each KV block.
        causal: Whether to apply causal masking (prevent selecting future blocks).
        q_tile_size: Size of each query tile (should match FA4's CTA tile: 2 * m_block_size = 256).
        softmax_scale: Scaling factor for attention scores.

    Returns:
        block_indices: Selected KV block indices, shape (B, H, N_q_tiles, k).
            Each entry is an index into the compressed KV sequence.
    """
    B, N, H, D = Q.shape
    H_kv = K_cmp.shape[2]
    N_cmp = K_cmp.shape[1]
    groups = H // H_kv

    N_q_tiles = N // q_tile_size
    assert N % q_tile_size == 0, (
        f"Sequence length {N} must be divisible by q_tile_size {q_tile_size}"
    )

    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(D)

    k_actual = min(num_selected_blocks, N_cmp)

    # Expand K_cmp for GQA: (B, N_cmp, H_kv, D) -> (B, N_cmp, H, D)
    if groups > 1:
        K_cmp_expanded = K_cmp.repeat_interleave(groups, dim=2)
    else:
        K_cmp_expanded = K_cmp

    # Pre-transpose for efficient matmul: (B, H, D, N_cmp)
    K_cmp_t = K_cmp_expanded.permute(0, 2, 3, 1).float()

    block_indices = torch.empty(
        B, H, N_q_tiles, k_actual, dtype=torch.int32, device=Q.device
    )

    # Precompute causal mask data (tiny tensors)
    if causal:
        q_tile_end = (torch.arange(N_q_tiles, device=Q.device) + 1) * q_tile_size
        kv_block_start = torch.arange(N_cmp, device=Q.device) * compress_block_size

    # Process query tiles in chunks to bound peak memory.
    # Each chunk computes (B, H, chunk_tiles * q_tile_size, N_cmp) scores,
    # averages per tile, applies causal mask, and runs topk.
    chunk_tiles = min(16, N_q_tiles)
    for chunk_start in range(0, N_q_tiles, chunk_tiles):
        chunk_end = min(chunk_start + chunk_tiles, N_q_tiles)
        n_tiles = chunk_end - chunk_start
        q_start = chunk_start * q_tile_size
        q_end = chunk_end * q_tile_size

        # (B, H, n_tiles * q_tile_size, N_cmp) — bounded, not O(N²)
        scores_chunk = (
            torch.matmul(Q[:, q_start:q_end].permute(0, 2, 1, 3).float(), K_cmp_t)
            * softmax_scale
        )

        # Average per tile: (B, H, n_tiles, N_cmp)
        scores_agg = scores_chunk.reshape(B, H, n_tiles, q_tile_size, N_cmp).mean(dim=3)

        # Apply causal mask: block j is future if j * block_size >= (i+1) * q_tile_size
        if causal:
            future_mask = kv_block_start.unsqueeze(0) >= q_tile_end[
                chunk_start:chunk_end
            ].unsqueeze(1)  # (n_tiles, N_cmp)
            scores_agg.masked_fill_(
                future_mask.unsqueeze(0).unsqueeze(0), float("-inf")
            )

        # Select top-k blocks for this chunk
        topk_scores, chunk_idx = scores_agg.topk(k_actual, dim=-1)

        # Replace invalid selections (future blocks with -inf score) with block 0
        invalid = topk_scores == float("-inf")
        if invalid.any():
            chunk_idx = chunk_idx.masked_fill(invalid, 0)

        # Sort indices for better memory access patterns
        block_indices[:, :, chunk_start:chunk_end] = chunk_idx.sort(dim=-1).values.to(
            torch.int32
        )

    return block_indices


# ---------------------------------------------------------------------------
# CuteDSL fused scoring + top-k selection kernel
# ---------------------------------------------------------------------------

_fused_select_compile_cache: dict = {}

# Bucket sizes for max_n_per_thread to limit compilations
_N_PER_THREAD_BUCKETS = [4, 8, 16, 32, 64, 128]


def _bucket_n_per_thread(n: int) -> int:
    """Round up n to the next bucket size."""
    for b in _N_PER_THREAD_BUCKETS:
        if n <= b:
            return b
    return _N_PER_THREAD_BUCKETS[-1]


def _make_fused_select_kernel(
    cute_dtype: Type, D: int, k: int, max_n_per_thread: int, causal: bool
):
    """Create a compiled CuteDSL kernel for fused scoring + top-k selection.

    Each thread block handles one (batch, head, query_tile) triple.
    128 threads (4 warps) per block:
      Phase 1: Each thread scores ceil(N_cmp/128) KV blocks via dot product.
      Phase 2: parallel_topk finds the global top-k across all threads.
      Phase 3: Thread 0 replaces -inf entries with 0, sorts ascending, writes.

    The topk logic is inlined directly in the kernel to avoid CuteDSL
    preprocessor issues with 'break' inside range_constexpr in helper
    functions.
    """
    import cutlass
    import cutlass.cute as cute
    from cutlass import const_expr, Float32, Int32

    THREADS_PER_BLOCK = 128

    class _Kernel:
        @cute.jit
        def __call__(
            self,
            mQ_mean,
            mK_cmp,
            mOut,
            N_cmp,
            N_q_tiles,
            H,
            H_kv,
            compress_block_size,
            q_tile_size,
            stream,
        ):
            B_val = mQ_mean.shape[0] // (N_q_tiles * H)
            grid_x = B_val * H * N_q_tiles
            self.kernel(
                mQ_mean,
                mK_cmp,
                mOut,
                N_cmp,
                N_q_tiles,
                H,
                H_kv,
                compress_block_size,
                q_tile_size,
                B_val,
            ).launch(
                grid=[grid_x, 1, 1],
                block=[THREADS_PER_BLOCK, 1, 1],
                smem=const_expr(4 * k * 4 + 16 + 4 * k * 4 + 16),
                stream=stream,
            )

        @cute.kernel
        def kernel(
            self,
            mQ_mean,
            mK_cmp,
            mOut,
            N_cmp,
            N_q_tiles,
            H,
            H_kv,
            compress_block_size,
            q_tile_size,
            B_val,
        ):
            tidx = cute.arch.thread_idx()[0]
            bidx = cute.arch.block_idx()[0]

            # Decompose block index into (b, h, tile)
            b = bidx // (H * N_q_tiles)
            h = (bidx % (H * N_q_tiles)) // N_q_tiles
            tile = bidx % N_q_tiles

            # GQA: map query head to KV head
            h_kv = h // (H // H_kv)

            # Q_mean row index: b * N_q_tiles * H + tile * H + h
            q_row = b * N_q_tiles * H + tile * H + h

            # --- Phase 1: Score blocks ---
            n_per_thread = (N_cmp + Int32(THREADS_PER_BLOCK - 1)) // Int32(
                THREADS_PER_BLOCK
            )
            scores = cute.make_fragment(max_n_per_thread, Float32)
            scores.fill(-Float32.inf)

            # Causal: block j is future if
            # j * compress_block_size >= (tile+1) * q_tile_size
            if const_expr(causal):
                causal_limit = (tile + 1) * q_tile_size

            for i in cutlass.range(n_per_thread, unroll=1):
                j = tidx * n_per_thread + i
                if j < N_cmp:
                    # Causal check
                    skip = Int32(0)
                    if const_expr(causal):
                        if j * compress_block_size >= causal_limit:
                            skip = Int32(1)

                    if skip == 0:
                        # Dot product: sum_d Q_mean[q_row, d] * K_cmp[k_row, d]
                        k_row = b * N_cmp * H_kv + j * H_kv + h_kv
                        acc = Float32(0.0)
                        for d in cutlass.range_constexpr(D):
                            acc += Float32(mQ_mean[q_row, d]) * Float32(
                                mK_cmp[k_row, d]
                            )
                        scores[i] = acc

            # --- Phase 2: TopK ---
            @cute.struct
            class TopkSmem:
                vals: cute.struct.MemRange[Float32, const_expr(4 * k)]
                idxs: cute.struct.MemRange[Int32, const_expr(4 * k)]

            smem = cutlass.utils.SmemAllocator()
            storage = smem.allocate(TopkSmem, 16)
            smem_vals = storage.vals.get_tensor(cute.make_layout(const_expr(4 * k)))
            smem_idxs = storage.idxs.get_tensor(cute.make_layout(const_expr(4 * k)))

            # Step 2a: Per-thread insertion sort
            # Maintains top_vals in descending order (largest at index 0).
            # For each score, scan from bottom (k-1) to top (0). At each
            # position where val > top_vals[jj], shift top_vals[jj] down
            # and write val. Without break, val cascades up to its correct
            # position — earlier writes are overwritten by later shifts.
            top_vals = cute.make_fragment(k, Float32)
            top_idxs = cute.make_fragment(k, Int32)
            top_vals.fill(-Float32.inf)
            top_idxs.fill(Int32(0))

            for ii in cutlass.range(n_per_thread, unroll=1):
                val = scores[ii]
                idx = Int32(ii)
                for jj in cutlass.range_constexpr(k - 1, -1, -1):
                    if val > top_vals[jj]:
                        if const_expr(jj < k - 1):
                            top_vals[jj + 1] = top_vals[jj]
                            top_idxs[jj + 1] = top_idxs[jj]
                        top_vals[jj] = val
                        top_idxs[jj] = idx

            # Step 2b: Convert local indices to global
            lane_id = tidx % 32
            warp_id = tidx // 32
            for ii in cutlass.range_constexpr(k):
                top_idxs[ii] = lane_id * n_per_thread + top_idxs[ii]

            # Step 2c: Warp butterfly merge (5 rounds)
            for round_idx in cutlass.range_constexpr(5):
                distance: int = const_expr(1 << round_idx)
                p_vals = cute.make_fragment(k, Float32)
                p_idxs = cute.make_fragment(k, Int32)
                for ii in cutlass.range_constexpr(k):
                    p_vals[ii] = cute.arch.shuffle_sync_bfly(
                        top_vals[ii], offset=distance
                    )
                    p_idxs[ii] = cute.arch.shuffle_sync_bfly(
                        top_idxs[ii], offset=distance
                    )
                if (lane_id & distance) == 0:
                    # Merge top_vals/idxs with p_vals/p_idxs
                    m_vals = cute.make_fragment(k, Float32)
                    m_idxs = cute.make_fragment(k, Int32)
                    ia = Int32(0)
                    ib = Int32(0)
                    for oi in cutlass.range_constexpr(k):
                        take_a = (ia < k) & ((ib >= k) | (top_vals[ia] >= p_vals[ib]))
                        if take_a:
                            m_vals[oi] = top_vals[ia]
                            m_idxs[oi] = top_idxs[ia]
                            ia += 1
                        else:
                            m_vals[oi] = p_vals[ib]
                            m_idxs[oi] = p_idxs[ib]
                            ib += 1
                    for ii in cutlass.range_constexpr(k):
                        top_vals[ii] = m_vals[ii]
                        top_idxs[ii] = m_idxs[ii]

            # Step 2d: Cross-warp merge via shared memory
            n_per_warp = n_per_thread * 32

            if lane_id == 0:
                for ii in cutlass.range_constexpr(k):
                    smem_vals[warp_id * k + ii] = top_vals[ii]
                    smem_idxs[warp_id * k + ii] = top_idxs[ii] + warp_id * n_per_warp

            cute.arch.sync_threads()

            if tidx == 0:
                # Start with warp 0's data (already in top_vals/top_idxs)
                for w in cutlass.range_constexpr(1, 4):
                    o_vals = cute.make_fragment(k, Float32)
                    o_idxs = cute.make_fragment(k, Int32)
                    for ii in cutlass.range_constexpr(k):
                        o_vals[ii] = smem_vals[w * k + ii]
                        o_idxs[ii] = smem_idxs[w * k + ii]
                    # Merge
                    m_vals = cute.make_fragment(k, Float32)
                    m_idxs = cute.make_fragment(k, Int32)
                    ia = Int32(0)
                    ib = Int32(0)
                    for oi in cutlass.range_constexpr(k):
                        take_a = (ia < k) & ((ib >= k) | (top_vals[ia] >= o_vals[ib]))
                        if take_a:
                            m_vals[oi] = top_vals[ia]
                            m_idxs[oi] = top_idxs[ia]
                            ia += 1
                        else:
                            m_vals[oi] = o_vals[ib]
                            m_idxs[oi] = o_idxs[ib]
                            ib += 1
                    for ii in cutlass.range_constexpr(k):
                        top_vals[ii] = m_vals[ii]
                        top_idxs[ii] = m_idxs[ii]

                # Write merged result to smem for broadcast
                for ii in cutlass.range_constexpr(k):
                    smem_vals[ii] = top_vals[ii]
                    smem_idxs[ii] = top_idxs[ii]

            cute.arch.sync_threads()

            # All threads read final result
            result_idxs = cute.make_fragment(k, Int32)
            for ii in cutlass.range_constexpr(k):
                result_idxs[ii] = smem_idxs[ii]

            # --- Phase 3: Sort + write (thread 0) ---
            if tidx == 0:
                # Replace -inf-scored entries with index 0
                for ii in cutlass.range_constexpr(k):
                    if smem_vals[ii] == -Float32.inf:
                        result_idxs[ii] = Int32(0)

                # Insertion sort indices ascending
                for ii in cutlass.range_constexpr(1, k):
                    key = result_idxs[ii]
                    insert_pos = Int32(ii)
                    for j_scan in cutlass.range_constexpr(ii):
                        pos: int = const_expr(ii - 1 - j_scan)
                        if insert_pos == const_expr(pos + 1):
                            if result_idxs[pos] > key:
                                result_idxs[pos + 1] = result_idxs[pos]
                                insert_pos = Int32(pos)
                    result_idxs[insert_pos] = key

                # Write output
                out_row = b * H * N_q_tiles + h * N_q_tiles + tile
                for ii in cutlass.range_constexpr(k):
                    mOut[out_row, ii] = result_idxs[ii]

    return _Kernel()


def fused_score_and_select_blocks(
    Q: Tensor,  # (B, N, H, D)
    K_cmp: Tensor,  # (B, N_cmp, H_kv, D)
    num_selected_blocks: int,
    compress_block_size: int,
    causal: bool = True,
    q_tile_size: int = 256,
    softmax_scale: float | None = None,
) -> Tensor:
    """Fused block scoring and top-k selection using a CuteDSL kernel.

    Replaces score_and_select_blocks with a single kernel launch. Uses the
    Q_mean optimization: mean(Q @ K) = mean(Q) @ K, reducing scoring from a
    full GEMM to a GEMV per query tile (256x fewer FLOPs).

    The Q_mean computation is done in PyTorch (single kernel), then the
    CuteDSL kernel fuses dot products, causal masking, top-k selection, and
    sorting into one launch.

    Falls back to PyTorch reference for N_cmp > 128 * 128 = 16384 blocks.

    Args:
        Q: Query tensor, shape (B, N, H, D).
        K_cmp: Compressed key tensor, shape (B, N_cmp, H_kv, D).
        num_selected_blocks: Number of KV blocks to select per query tile.
        compress_block_size: Size of each KV block.
        causal: Whether to apply causal masking.
        q_tile_size: Size of each query tile.
        softmax_scale: Scaling factor for attention scores.

    Returns:
        block_indices: Selected KV block indices, shape (B, H, N_q_tiles, k).
    """
    import cuda.bindings.driver as cuda
    import cutlass
    import cutlass.cute as cute
    from cutlass.cute.runtime import from_dlpack

    B, N, H, D = Q.shape
    H_kv = K_cmp.shape[2]
    N_cmp = K_cmp.shape[1]

    N_q_tiles = N // q_tile_size
    assert N % q_tile_size == 0, (
        f"Sequence length {N} must be divisible by q_tile_size {q_tile_size}"
    )

    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(D)

    k_actual = min(num_selected_blocks, N_cmp)

    THREADS = 128
    raw_n_per_thread = (N_cmp + THREADS - 1) // THREADS
    max_n_per_thread = _bucket_n_per_thread(raw_n_per_thread)

    # Fall back to PyTorch reference for very large N_cmp
    if max_n_per_thread > _N_PER_THREAD_BUCKETS[-1]:
        return score_and_select_blocks(
            Q,
            K_cmp,
            num_selected_blocks,
            compress_block_size,
            causal=causal,
            q_tile_size=q_tile_size,
            softmax_scale=softmax_scale,
        )

    # --- Q_mean in PyTorch ---
    # Q: (B, N, H, D) -> (B, N_q_tiles, q_tile_size, H, D) -> mean over q_tile_size
    Q_mean = (
        Q.reshape(B, N_q_tiles, q_tile_size, H, D)
        .mean(dim=2)  # (B, N_q_tiles, H, D)
        .float()
        * softmax_scale
    )

    # Reshape Q_mean to 2D: row (b, tile, h) -> b * N_q_tiles * H + tile * H + h
    Q_mean_2d = Q_mean.reshape(B * N_q_tiles * H, D).contiguous()

    # Reshape K_cmp to 2D: row (b, j, h_kv) -> b * N_cmp * H_kv + j * H_kv + h_kv
    K_cmp_2d = K_cmp.reshape(B * N_cmp * H_kv, D).contiguous().float()

    # Output: (B * H * N_q_tiles, k_actual) int32
    out_2d = torch.empty(
        B * H * N_q_tiles, k_actual, dtype=torch.int32, device=Q.device
    )

    torch2cute_dtype = {
        torch.float16: cutlass.Float16,
        torch.bfloat16: cutlass.BFloat16,
        torch.float32: cutlass.Float32,
    }
    cute_dtype = torch2cute_dtype[Q_mean_2d.dtype]

    def to_cute(tensor):
        return from_dlpack(
            tensor.detach(), assumed_align=16
        ).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1))

    mQ_mean = to_cute(Q_mean_2d)
    mK_cmp = to_cute(K_cmp_2d)
    mOut = to_cute(out_2d)

    current_stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

    compile_key = (cute_dtype, D, k_actual, max_n_per_thread, causal)
    if compile_key not in _fused_select_compile_cache:
        kernel_op = _make_fused_select_kernel(
            cute_dtype, D, k_actual, max_n_per_thread, causal
        )
        _fused_select_compile_cache[compile_key] = cute.compile(
            kernel_op,
            mQ_mean,
            mK_cmp,
            mOut,
            N_cmp,
            N_q_tiles,
            H,
            H_kv,
            compress_block_size,
            q_tile_size,
            current_stream,
        )

    _fused_select_compile_cache[compile_key](
        mQ_mean,
        mK_cmp,
        mOut,
        N_cmp,
        N_q_tiles,
        H,
        H_kv,
        compress_block_size,
        q_tile_size,
        current_stream,
    )

    # Reshape output: (B * H * N_q_tiles, k) -> (B, H, N_q_tiles, k)
    block_indices = out_2d.reshape(B, H, N_q_tiles, k_actual)
    return block_indices
