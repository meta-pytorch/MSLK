# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
# pyre-strict

"""In-register parallel top-K selection for NSA block scoring.

Provides CuteDSL @cute.jit utilities for performing top-K selection entirely
in registers across the correction warp group (128 threads, 4 warps).

Algorithm:
1. Each thread holds N_cmp/128 block scores in registers.
2. Local insertion sort: each thread keeps its top-K values + indices.
3. Warp-level merge: 5 rounds of butterfly shuffle to merge across 32 threads.
4. Cross-warp merge: 4 warps → final K candidates via shared memory staging.

For 1M tokens with L=64: N_cmp=16384, each thread holds 128 scores.
With K_SEL=16, total ~240 comparisons per thread — negligible vs MMA time.
"""

from __future__ import annotations

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, const_expr


@cute.jit
def insertion_sort_topk(
    scores: cute.Tensor,  # Thread-local scores, shape (local_n,)
    num_local: Int32,
    k: int,
) -> tuple[cute.Tensor, cute.Tensor]:
    """Per-thread insertion sort to find top-K from local scores.

    Each thread independently sorts its local scores (descending) and
    retains the top-K values and their local indices.

    Args:
        scores: Thread-local block scores, shape (local_n,).
        num_local: Number of valid scores this thread holds.
        k: Number of top elements to retain.

    Returns:
        top_vals: Top-K values (descending), shape (k,).
        top_idxs: Corresponding local indices, shape (k,).
    """
    top_vals = cute.make_fragment(k, Float32)
    top_idxs = cute.make_fragment(k, Int32)

    # Initialize with -inf
    top_vals.fill(-Float32.inf)
    top_idxs.fill(Int32(0))

    # Insertion sort: for each score, insert into sorted top_vals if larger
    for i in cutlass.range(num_local, unroll=1):
        val = scores[i]
        idx = Int32(i)
        # Find insertion point (linear scan from the end)
        for j in cutlass.range_constexpr(k - 1, -1, -1):
            if val > top_vals[j]:
                if const_expr(j < k - 1):
                    top_vals[j + 1] = top_vals[j]
                    top_idxs[j + 1] = top_idxs[j]
                top_vals[j] = val
                top_idxs[j] = idx
                break

    return top_vals, top_idxs


@cute.jit
def merge_sorted_pairs(
    vals_a: cute.Tensor,  # (k,) descending
    idxs_a: cute.Tensor,  # (k,)
    vals_b: cute.Tensor,  # (k,) descending
    idxs_b: cute.Tensor,  # (k,)
    k: int,
) -> tuple[cute.Tensor, cute.Tensor]:
    """Merge two sorted top-K lists into one top-K list.

    Standard merge of two sorted (descending) sequences, keeping only
    the top-K elements from the merged result.

    Args:
        vals_a, idxs_a: First sorted top-K list.
        vals_b, idxs_b: Second sorted top-K list.
        k: Number of elements to keep.

    Returns:
        merged_vals: Merged top-K values (descending), shape (k,).
        merged_idxs: Corresponding indices, shape (k,).
    """
    merged_vals = cute.make_fragment(k, Float32)
    merged_idxs = cute.make_fragment(k, Int32)

    ia = Int32(0)
    ib = Int32(0)
    for out_idx in cutlass.range_constexpr(k):
        take_a = (ia < k) & ((ib >= k) | (vals_a[ia] >= vals_b[ib]))
        if take_a:
            merged_vals[out_idx] = vals_a[ia]
            merged_idxs[out_idx] = idxs_a[ia]
            ia += 1
        else:
            merged_vals[out_idx] = vals_b[ib]
            merged_idxs[out_idx] = idxs_b[ib]
            ib += 1

    return merged_vals, merged_idxs


@cute.jit
def warp_merge_topk(
    vals: cute.Tensor,  # (k,) per-thread top-K values
    idxs: cute.Tensor,  # (k,) per-thread top-K local indices
    thread_id_in_warp: Int32,
    k: int,
    n_per_thread: Int32,
) -> tuple[cute.Tensor, cute.Tensor]:
    """Warp-level butterfly merge of per-thread top-K lists.

    After this, thread 0 in each warp holds the warp-level top-K.
    Indices are converted from thread-local to warp-global during merging.

    5 rounds of shuffle: distance 1, 2, 4, 8, 16.

    Args:
        vals: Per-thread top-K values, shape (k,).
        idxs: Per-thread top-K local indices, shape (k,).
        thread_id_in_warp: Lane ID (0-31).
        k: Number of top elements.
        n_per_thread: Number of scores each thread originally held.

    Returns:
        vals: Merged top-K values (valid on lane 0).
        idxs: Merged top-K global indices (valid on lane 0).
    """
    # Convert local indices to global: global_idx = thread_id * n_per_thread + local_idx
    for i in cutlass.range_constexpr(k):
        idxs[i] = thread_id_in_warp * n_per_thread + idxs[i]

    # Butterfly reduction: 5 rounds
    for round_idx in cutlass.range_constexpr(5):
        distance: int = const_expr(1 << round_idx)
        # Exchange top-K with partner
        partner_vals = cute.make_fragment(k, Float32)
        partner_idxs = cute.make_fragment(k, Int32)
        for i in cutlass.range_constexpr(k):
            partner_vals[i] = cute.arch.shfl_xor(vals[i], distance)
            partner_idxs[i] = cute.arch.shfl_xor(idxs[i], distance)
        # Merge: the lower-ID thread in each pair does the merge
        if (thread_id_in_warp & distance) == 0:
            vals, idxs = merge_sorted_pairs(vals, idxs, partner_vals, partner_idxs, k)

    return vals, idxs


@cute.jit
def cross_warp_merge_topk(
    vals: cute.Tensor,  # (k,) — valid on lane 0 of each warp
    idxs: cute.Tensor,  # (k,) — valid on lane 0 of each warp
    thread_id_in_wg: Int32,  # 0..127 within correction warp group
    warp_id_in_wg: Int32,  # 0..3
    smem_vals: cute.Tensor,  # shared mem buffer for vals, shape (4, k)
    smem_idxs: cute.Tensor,  # shared mem buffer for idxs, shape (4, k)
    k: int,
    n_per_warp: Int32,
) -> tuple[cute.Tensor, cute.Tensor]:
    """Cross-warp merge: reduce 4 warp-level top-K lists to one.

    Lane 0 of each warp writes its top-K to shared memory. Then lane 0 of
    warp 0 performs a sequential merge of all 4 lists.

    After the merge, the result is broadcast back to all threads via smem.

    Args:
        vals: Warp-level top-K values (valid on lane 0).
        idxs: Warp-level top-K global indices (valid on lane 0).
        thread_id_in_wg: Thread index within correction warp group (0-127).
        warp_id_in_wg: Warp index within correction warp group (0-3).
        smem_vals: Shared memory for values exchange, shape (4, k).
        smem_idxs: Shared memory for indices exchange, shape (4, k).
        k: Number of top elements.
        n_per_warp: Scores per warp (for global index offset).

    Returns:
        vals: Final top-K values (broadcast to all threads).
        idxs: Final top-K global indices (broadcast to all threads).
    """
    lane_id = thread_id_in_wg % 32

    # Lane 0 of each warp writes to shared memory
    # Offset indices by warp_id * n_per_warp for cross-warp global indexing
    if lane_id == 0:
        for i in cutlass.range_constexpr(k):
            smem_vals[warp_id_in_wg * k + i] = vals[i]
            smem_idxs[warp_id_in_wg * k + i] = idxs[i] + warp_id_in_wg * n_per_warp

    cute.arch.sync_threads()

    # Lane 0 of warp 0 merges all 4 lists sequentially
    if thread_id_in_wg == 0:
        # Start with warp 0's list
        for w in cutlass.range_constexpr(1, 4):
            other_vals = cute.make_fragment(k, Float32)
            other_idxs = cute.make_fragment(k, Int32)
            for i in cutlass.range_constexpr(k):
                other_vals[i] = smem_vals[w * k + i]
                other_idxs[i] = smem_idxs[w * k + i]
            vals, idxs = merge_sorted_pairs(vals, idxs, other_vals, other_idxs, k)

        # Write merged result back to smem for broadcast
        for i in cutlass.range_constexpr(k):
            smem_vals[i] = vals[i]
            smem_idxs[i] = idxs[i]

    cute.arch.sync_threads()

    # All threads read the final result
    result_vals = cute.make_fragment(k, Float32)
    result_idxs = cute.make_fragment(k, Int32)
    for i in cutlass.range_constexpr(k):
        result_vals[i] = smem_vals[i]
        result_idxs[i] = smem_idxs[i]

    return result_vals, result_idxs


@cute.jit
def parallel_topk(
    scores: cute.Tensor,  # Thread-local scores, shape (local_n,)
    num_local: Int32,
    k: int,
    thread_id_in_wg: Int32,  # 0..127 within correction warp group
    smem_vals: cute.Tensor,  # shared mem buffer, shape (4 * k,)
    smem_idxs: cute.Tensor,  # shared mem buffer, shape (4 * k,)
    n_total: Int32,
) -> cute.Tensor:
    """Full parallel top-K pipeline: local sort → warp merge → cross-warp merge.

    Each of 128 threads holds num_local = N_cmp/128 scores. Returns the global
    top-K indices into the score array.

    Args:
        scores: Thread-local block scores, shape (local_n,).
        num_local: Number of valid scores per thread.
        k: Number of top blocks to select (K_SEL).
        thread_id_in_wg: Thread index within the correction warp group (0-127).
        smem_vals: Shared memory for cross-warp merge, shape (4 * k,).
        smem_idxs: Shared memory for cross-warp merge, shape (4 * k,).
        n_total: Total number of compressed blocks.

    Returns:
        top_indices: Top-K global block indices, shape (k,), sorted descending by score.
    """
    lane_id = thread_id_in_wg % 32
    warp_id_in_wg = thread_id_in_wg // 32
    n_per_thread = num_local
    n_per_warp = n_per_thread * 32

    # Step 1: Per-thread insertion sort
    vals, idxs = insertion_sort_topk(scores, num_local, k)

    # Step 2: Warp-level butterfly merge
    vals, idxs = warp_merge_topk(vals, idxs, lane_id, k, n_per_thread)

    # Step 3: Cross-warp merge (4 warps → 1)
    vals, idxs = cross_warp_merge_topk(
        vals, idxs, thread_id_in_wg, warp_id_in_wg,
        smem_vals, smem_idxs, k, n_per_warp,
    )

    return idxs
