# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# The kernel is adapted from AITER
# (https://github.com/ROCm/aiter,
# aiter/ops/flydsl/kernels/hstu/hstu_attention_fwd.py):
# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

# pyre-unsafe

"""FlyDSL ragged HSTU attention forward (gfx942 / gfx950).

Implements :func:`mslk.attention.hstu._torch_ref.ragged_hstu_attention_ref`:
SiLU heads ``out_i = scale_i * sum_j valid(i,j) * silu(alpha * q_i.k_j) * v_j``
and softmax heads (with the base-2 LSE ``M``), every T1 mask type, targets,
window, full attention and contextual prefix, and the three SiLU scales
(``1/N``, a 0-dim tensor, one per token).

Each feature that is present is compiled in; its value is a runtime argument.
SiLU and softmax heads run as separate launches over their head ranges.

Pipeline (one query tile per block): K goes global -> registers -> XOR-swizzled
LDS; V is register-prefetched under a counted vmcnt while GEMM1 runs and then
stored transposed (``[d, kv]``). GEMM1 computes ``S^T = K Q^T`` so the score
fragment is already GEMM2's operand; GEMM2 computes ``O^T = V^T P^T``, which
leaves each lane with one query row in both, so the softmax rescale and the
per-token scale stay lane-local. Tensor strides are i64, so packed offsets
past 2^31 elements address correctly.

A query tile visits only the KV tiles its rows can see (see
``RaggedMask.kv_range``); with a window and a contextual prefix it also visits
the contextual columns, which every row sees.
"""

import functools
from typing import Any, Dict, List, Optional, Tuple

import torch
from mslk.attention.flydsl.hstu.common import (
    arch_dma_params,
    bind_mfma_accs,
    grouped_loader,
    grouped_storer,
    LOG2E,
    MFMA_ELEMS_PER_LANE,
    MFMA_K,
    MFMA_LANE_K,
    MFMA_M,
    MFMA_N,
    mfma_params_for_dim,
    pack_mfma_frag,
    RaggedFeatures,
    RaggedMask,
    silu_gate,
    WARP_SIZE,
)
from mslk.flydsl.autotune import next_pow2, tunable

# Blocks are decoded group-major for L2 locality.
NUM_GRID_GROUPS = 8
SCALE_MODES = ("inv_n", "scalar", "per_token")
# Finite stand-in for -inf: fast-math assumes no infinities.
_NEG_BIG = -1.0e30


def validate_config(
    head_dim: int,
    hidden_dim: int,
    *,
    block_m: int,
    block_n: int,
    num_waves: int,
    waves_per_eu: int,
    arch: str,
) -> None:
    from mslk.flydsl.common import lds_capacity_bytes

    if not (arch.startswith("gfx942") or arch.startswith("gfx950")):
        raise ValueError(f"hstu ragged fwd supports gfx942 and gfx950, got {arch!r}")
    mfma_k, _ = mfma_params_for_dim(head_dim, arch)
    if head_dim <= 0 or head_dim % mfma_k != 0:
        raise ValueError(f"head_dim must be a multiple of {mfma_k}, got {head_dim}")
    if hidden_dim <= 0 or hidden_dim % MFMA_M != 0:
        raise ValueError(f"hidden_dim must be a multiple of {MFMA_M}, got {hidden_dim}")
    if num_waves <= 0 or block_m % (num_waves * MFMA_M) != 0:
        raise ValueError(f"block_m {block_m} must be a multiple of num_waves*16")
    if block_n <= 0 or block_n % MFMA_N != 0:
        raise ValueError(f"block_n {block_n} must be a multiple of {MFMA_N}")
    if waves_per_eu < 0:
        raise ValueError(f"waves_per_eu must be >= 0, got {waves_per_eu}")
    dma_elems, _, _ = arch_dma_params(arch)
    threads = num_waves * WARP_SIZE
    head_dim_k = (head_dim + 63) // 64 * 64
    for name, width in (("K", head_dim_k), ("V", hidden_dim)):
        if (block_n * width) % (threads * dma_elems) != 0:
            raise ValueError(f"{name} tile does not divide the DMA pass evenly")
        vec = 8 if (block_n * width) % (threads * 8) == 0 else dma_elems
        per_row = width // vec
        if threads % per_row != 0:
            raise ValueError(f"{name}: {threads} threads not divisible by {per_row}")
        rows = threads // per_row
        if block_n % rows != 0 and rows <= block_n:
            raise ValueError(f"{name}: {rows} rows per pass must divide {block_n}")
    lds = block_n * head_dim_k * 2 + hidden_dim * (block_n + 8) * 2
    if lds > lds_capacity_bytes(arch):
        raise ValueError(f"LDS tile {lds} B exceeds {lds_capacity_bytes(arch)} B")


@functools.lru_cache(maxsize=1024)
def compile_hstu_ragged_fwd(
    *,
    num_heads: int,
    head_begin: int,
    launch_heads: int,
    head_dim: int,
    hidden_dim: int,
    dtype_str: str,
    feats: RaggedFeatures,
    softmax: bool,
    scale_mode: str,
    block_m: int = 128,
    block_n: int = 32,
    num_waves: int = 4,
    waves_per_eu: int = 2,
    arch: str = "",
) -> Any:
    """Build the host launcher for one compile-time specialization."""
    import flydsl.compiler as flyc  # pyre-ignore[21]
    import flydsl.expr as fx  # pyre-ignore[21]
    from flydsl.expr import const_expr, gpu, range_constexpr, rocdl  # pyre-ignore
    from flydsl.expr.typing import Vector as Vec  # pyre-ignore[21]
    from flydsl.runtime.device import get_rocm_arch  # pyre-ignore[21]

    if not arch:
        arch = get_rocm_arch()
    feats.validate()
    validate_config(
        head_dim,
        hidden_dim,
        block_m=block_m,
        block_n=block_n,
        num_waves=num_waves,
        waves_per_eu=waves_per_eu,
        arch=arch,
    )
    if dtype_str not in ("f16", "bf16"):
        raise ValueError(f"dtype must be f16 or bf16, got {dtype_str!r}")
    if scale_mode not in SCALE_MODES:
        raise ValueError(f"scale_mode must be one of {SCALE_MODES}")
    if not 0 <= head_begin < head_begin + launch_heads <= num_heads:
        raise ValueError("head range out of bounds")

    BLOCK_M = block_m
    BLOCK_N = block_n
    BLOCK_THREADS = num_waves * WARP_SIZE
    ROWS_PER_WAVE = BLOCK_M // num_waves
    Q_SUBTILES = ROWS_PER_WAVE // MFMA_M
    KV_SUBTILES = BLOCK_N // MFMA_N
    MFMA_DIM_K, MFMA_DIM_LANE_K = mfma_params_for_dim(head_dim, arch)
    DMA_ELEMS, K_SWZ_ROWS, K_SWZ_SHIFT = arch_dma_params(arch)
    MFMA_DIM_LANE_K_LOG2 = MFMA_DIM_LANE_K.bit_length() - 1
    assert K_SWZ_SHIFT >= MFMA_DIM_LANE_K_LOG2

    elem_dtype = fx.Float16 if dtype_str == "f16" else fx.BFloat16
    is_lower = feats.mask_type == "lower_triangular"

    K_STEPS = head_dim // MFMA_DIM_K
    # The K swizzle has period 64, so the K row pitch rounds up to 64; padded
    # columns load as zeros against a zero Q operand.
    HEAD_DIM_K = (head_dim + 63) // 64 * 64
    K_COL_GUARD = head_dim < HEAD_DIM_K
    K_STEPS_K = HEAD_DIM_K // MFMA_DIM_K
    D_CHUNKS = hidden_dim // MFMA_M
    # V is stored transposed, [d, kv]; the +8 breaks LDS bank conflicts.
    V_T_STRIDE = BLOCK_N + 8

    def load_tiling(width):
        vec = 8 if (BLOCK_N * width) % (BLOCK_THREADS * 8) == 0 else DMA_ELEMS
        per_row = width // vec
        rows = BLOCK_THREADS // per_row
        return vec, per_row, rows, max(1, BLOCK_N // rows), rows > BLOCK_N

    VEC_K, THREADS_PER_ROW_K, ROWS_PER_BATCH_K, NUM_BATCHES_K, K_NEEDS_GUARD = (
        load_tiling(HEAD_DIM_K)
    )
    VEC_V, THREADS_PER_ROW_V, ROWS_PER_BATCH_V, NUM_BATCHES_V, V_NEEDS_GUARD = (
        load_tiling(hidden_dim)
    )
    assert VEC_K % MFMA_DIM_LANE_K == 0

    N_ACC = D_CHUNKS * Q_SUBTILES
    # Tiles strictly below the diagonal and before the targets are fully valid
    # under a plain causal mask, so they skip the per-element mask.
    CAUSAL_SPLIT = is_lower and not (feats.has_window or feats.has_contextual)
    CTX_LOOP = feats.has_window and feats.has_contextual

    @fx.struct
    class SharedStorage:
        k: fx.Array[elem_dtype, BLOCK_N * HEAD_DIM_K, 16]
        v: fx.Array[elem_dtype, hidden_dim * V_T_STRIDE, 16]

    @flyc.kernel(known_block_size=[BLOCK_THREADS, 1, 1])
    def hstu_ragged_fwd_kernel(
        q: fx.Tensor,
        k: fx.Tensor,
        v: fx.Tensor,
        seq_offsets: fx.Tensor,
        num_targets: fx.Tensor,
        scale: fx.Tensor,
        out: fx.Tensor,
        lse: fx.Tensor,
        num_q_tiles: fx.Int32,
        hz_per_group: fx.Int32,
        hz_total: fx.Int32,
        window: fx.Int32,
        contextual: fx.Int32,
        full: fx.Int32,
        c_alpha: fx.Float32,
        out_scale: fx.Float32,
    ) -> None:
        c_zero_dim_pack = fx.full(MFMA_DIM_LANE_K, 0.0, elem_dtype).ir_value()
        dim_mfma_acc, mfma_acc = bind_mfma_accs(
            elem_dtype, (MFMA_DIM_K, MFMA_DIM_LANE_K), (MFMA_K, MFMA_LANE_K)
        )

        tid = fx.Int32(gpu.thread_idx.x)
        wave_id = tid // fx.Int32(WARP_SIZE)
        lane = tid % fx.Int32(WARP_SIZE)
        lane_mod_16 = lane % fx.Int32(MFMA_N)
        lane_div_16 = lane // fx.Int32(MFMA_N)

        # ---- Group-major grid decode -> (batch, head, query tile) ----
        block_id = fx.Int32(gpu.block_idx.x)
        grid_group = block_id % fx.Int32(NUM_GRID_GROUPS)
        pos_in_group = block_id // fx.Int32(NUM_GRID_GROUPS)
        local_hz_idx = pos_in_group // num_q_tiles
        q_tile_idx = pos_in_group % num_q_tiles
        hz_idx = grid_group * hz_per_group + local_hz_idx
        # The last group is padded past hz_total; padding blocks get seq_len 0.
        block_valid = hz_idx < hz_total
        hz_idx = block_valid.select(hz_idx, fx.Int32(0))
        batch_idx = hz_idx // fx.Int32(launch_heads)
        head_idx = fx.Int32(head_begin) + hz_idx % fx.Int32(launch_heads)

        seq_start = fx.Int32(seq_offsets[batch_idx])
        seq_len = fx.Int32(seq_offsets[batch_idx + fx.Int32(1)]) - seq_start
        seq_len = block_valid.select(seq_len, fx.Int32(0))
        num_target = fx.Int32(0)
        if const_expr(feats.has_targets):
            num_target = fx.Int32(num_targets[batch_idx])
        mask = RaggedMask(feats, seq_len, num_target, window, contextual, full)

        q_load = grouped_loader(q, head_dim, MFMA_DIM_LANE_K)
        k_load = grouped_loader(k, head_dim, VEC_K)
        v_load = grouped_loader(v, hidden_dim, VEC_V)
        o_store = grouped_storer(out, hidden_dim, MFMA_ELEMS_PER_LANE)

        lds = fx.SharedAllocator().allocate(SharedStorage).peek()
        k_smem = lds.k.view(
            fx.make_layout(
                (BLOCK_N, HEAD_DIM_K // MFMA_DIM_LANE_K, MFMA_DIM_LANE_K),
                (HEAD_DIM_K, MFMA_DIM_LANE_K, 1),
            )
        )
        v_smem = lds.v.view(
            fx.make_layout(
                (hidden_dim, V_T_STRIDE // MFMA_LANE_K, MFMA_LANE_K),
                (V_T_STRIDE, MFMA_LANE_K, 1),
            )
        )

        def k_swz_grp(tile_row, col_grp):
            """K LDS swizzle in MFMA_DIM_LANE_K-group units."""
            return col_grp ^ (
                (tile_row & fx.Int32(K_SWZ_ROWS - 1))
                << fx.Int32(K_SWZ_SHIFT - MFMA_DIM_LANE_K_LOG2)
            )

        # ---- Q rows (lane-owned: query = lane % 16) and register Q packs ----
        q_wave_base = q_tile_idx * fx.Int32(BLOCK_M) + wave_id * fx.Int32(ROWS_PER_WAVE)
        q_rows = []
        q_in_bounds = []
        for qg in range_constexpr(Q_SUBTILES):
            row = q_wave_base + fx.Int32(qg * MFMA_M) + lane_mod_16
            q_rows.append(row)
            q_in_bounds.append(row < seq_len)
        q_row_ids = [mask.to_id(q_rows[qg]) for qg in range_constexpr(Q_SUBTILES)]

        q_packs = []
        for ks in range_constexpr(K_STEPS):
            q_col_grp = fx.Int32(ks * (MFMA_DIM_K // MFMA_DIM_LANE_K)) + lane_div_16
            per_qg = []
            for qg in range_constexpr(Q_SUBTILES):
                safe = q_in_bounds[qg].select(seq_start + q_rows[qg], seq_start)
                raw = q_load(fx.Int64(safe), head_idx, q_col_grp).ir_value()
                per_qg.append(q_in_bounds[qg].select(raw, c_zero_dim_pack))
            q_packs.append(per_qg)

        # ---- KV tile range of this query tile ----
        q_start = q_tile_idx * fx.Int32(BLOCK_M)
        active = q_start < seq_len
        q_last = (q_start + fx.Int32(BLOCK_M) < seq_len).select(
            q_start + fx.Int32(BLOCK_M - 1), seq_len - fx.Int32(1)
        )
        kv_low, kv_high = mask.kv_range(q_start, q_last)
        kv_high = active.select(kv_high, fx.Int32(0))
        n_tiles = (kv_high + fx.Int32(BLOCK_N - 1)) // fx.Int32(BLOCK_N)
        kv_tile_start = kv_low // fx.Int32(BLOCK_N)
        ctx_tiles = fx.Int32(0)
        if const_expr(CTX_LOOP):
            ctx_tiles = (contextual + fx.Int32(BLOCK_N - 1)) // fx.Int32(BLOCK_N)
            ctx_tiles = (ctx_tiles < kv_tile_start).select(ctx_tiles, kv_tile_start)
        unmasked_end = kv_tile_start
        if const_expr(CAUSAL_SPLIT):
            split_pos = q_start
            if const_expr(feats.has_targets):
                split_pos = (q_start < mask.max_id).select(q_start, mask.max_id)
            unmasked_end = split_pos // fx.Int32(BLOCK_N)
            unmasked_end = (unmasked_end > kv_tile_start).select(
                unmasked_end, kv_tile_start
            )
            unmasked_end = (unmasked_end < n_tiles).select(unmasked_end, n_tiles)

        # ---- K: coalesced global -> registers -> swizzled LDS ----
        k_load_row_in_batch = tid // fx.Int32(THREADS_PER_ROW_K)
        k_load_lane_in_row = tid % fx.Int32(THREADS_PER_ROW_K)
        k_load_col = k_load_lane_in_row * fx.Int32(VEC_K)
        k_load_col_grp = k_load_lane_in_row * fx.Int32(VEC_K // MFMA_DIM_LANE_K)

        def async_load_k_regs(kv_start, full_tile):
            vecs = []
            for b in range_constexpr(NUM_BATCHES_K):
                row = k_load_row_in_batch + fx.Int32(b * ROWS_PER_BATCH_K)
                tok = kv_start + row
                guard = None
                if const_expr(not full_tile):
                    guard = tok < seq_len
                if const_expr(K_NEEDS_GUARD):
                    rowg = row < fx.Int32(BLOCK_N)
                    guard = rowg if guard is None else (guard & rowg)
                if const_expr(K_COL_GUARD):
                    colg = k_load_col < fx.Int32(head_dim)
                    guard = colg if guard is None else (guard & colg)
                if const_expr(guard is None):
                    raw = k_load(
                        fx.Int64(seq_start + tok), head_idx, k_load_lane_in_row
                    ).ir_value()
                    vecs.append(raw)
                else:
                    safe = guard.select(seq_start + tok, seq_start)
                    raw = k_load(fx.Int64(safe), head_idx, k_load_lane_in_row)
                    vecs.append(
                        guard.select(
                            raw.ir_value(), fx.full(VEC_K, 0.0, elem_dtype).ir_value()
                        )
                    )
            return vecs

        def store_k_regs_to_lds(vecs):
            for b in range_constexpr(NUM_BATCHES_K):
                row = k_load_row_in_batch + fx.Int32(b * ROWS_PER_BATCH_K)
                for h in range_constexpr(VEC_K // MFMA_DIM_LANE_K):
                    half = Vec.from_elements(
                        [
                            Vec(vecs[b])[h * MFMA_DIM_LANE_K + j]
                            for j in range_constexpr(MFMA_DIM_LANE_K)
                        ],
                        elem_dtype,
                    )
                    col_grp = k_load_col_grp + fx.Int32(h)
                    k_smem[row, k_swz_grp(row, col_grp), None].store(half)

        # ---- V: register prefetch, waited on only before its LDS publish ----
        v_load_row_in_batch = tid // fx.Int32(THREADS_PER_ROW_V)
        v_load_lane_in_row = tid % fx.Int32(THREADS_PER_ROW_V)
        v_load_col = v_load_lane_in_row * fx.Int32(VEC_V)

        def async_load_v_regs(kv_start, full_tile):
            vecs = []
            for b in range_constexpr(NUM_BATCHES_V):
                row = v_load_row_in_batch + fx.Int32(b * ROWS_PER_BATCH_V)
                tok = kv_start + row
                guard = None
                if const_expr(not full_tile):
                    guard = tok < seq_len
                if const_expr(V_NEEDS_GUARD):
                    rowg = row < fx.Int32(BLOCK_N)
                    guard = rowg if guard is None else (guard & rowg)
                if const_expr(guard is None):
                    raw = v_load(
                        fx.Int64(seq_start + tok), head_idx, v_load_lane_in_row
                    ).ir_value()
                    vecs.append(raw)
                else:
                    safe = guard.select(seq_start + tok, seq_start)
                    raw = v_load(fx.Int64(safe), head_idx, v_load_lane_in_row)
                    vecs.append(
                        guard.select(
                            raw.ir_value(), fx.full(VEC_V, 0.0, elem_dtype).ir_value()
                        )
                    )
            return vecs

        def store_v_regs_to_lds(vecs):
            for b in range_constexpr(NUM_BATCHES_V):
                kv_row = v_load_row_in_batch + fx.Int32(b * ROWS_PER_BATCH_V)
                kv_grp = kv_row // fx.Int32(MFMA_LANE_K)
                kv_lane = kv_row % fx.Int32(MFMA_LANE_K)
                vv = Vec(vecs[b])
                for j in range_constexpr(VEC_V):
                    v_smem[v_load_col + fx.Int32(j), kv_grp, kv_lane] = vv[j]

        def read_k_packs(ng):
            local_k_row = fx.Int32(ng * MFMA_M) + lane_mod_16
            packs = []
            for ks in range_constexpr(K_STEPS_K):
                k_col_grp = fx.Int32(ks * (MFMA_DIM_K // MFMA_DIM_LANE_K)) + lane_div_16
                packs.append(
                    k_smem[local_k_row, k_swz_grp(local_k_row, k_col_grp), None].load()
                )
            return packs

        # ---- GEMM1: S^T = K Q^T. Lane holds query lane%16, keys 4*(lane/16)+i.
        def gemm1(k_packs_by_ng):
            s = [[None] * Q_SUBTILES for _ in range_constexpr(KV_SUBTILES)]
            for ng in range_constexpr(KV_SUBTILES):
                k_packs = k_packs_by_ng[ng]
                for qg in range_constexpr(Q_SUBTILES):
                    cur = fx.full(MFMA_ELEMS_PER_LANE, 0.0, fx.Float32).ir_value()
                    for ks in range_constexpr(K_STEPS_K):
                        q_op = q_packs[ks][qg] if ks < K_STEPS else c_zero_dim_pack
                        cur = dim_mfma_acc(k_packs[ks].ir_value(), q_op, cur)
                    s[ng][qg] = [
                        Vec(cur)[i] for i in range_constexpr(MFMA_ELEMS_PER_LANE)
                    ]
            return s

        def tile_keep(kv_start):
            keep = [[None] * Q_SUBTILES for _ in range_constexpr(KV_SUBTILES)]
            for ng in range_constexpr(KV_SUBTILES):
                kv_base = kv_start + fx.Int32(ng * MFMA_N) + lane_div_16 * fx.Int32(4)
                cols = [kv_base + fx.Int32(i) for i in range_constexpr(4)]
                col_ids = [mask.to_id(c) for c in cols]
                in_seq = [c < seq_len for c in cols]
                for qg in range_constexpr(Q_SUBTILES):
                    keep[ng][qg] = [
                        mask.keep(q_rows[qg], cols[i], q_row_ids[qg], col_ids[i])
                        & in_seq[i]
                        for i in range_constexpr(4)
                    ]
            return keep

        c_zero_f = fx.Float32(0.0)
        c_neg_big = fx.Float32(_NEG_BIG)

        def silu_tile(s, keep):
            p_packs = [[None] * Q_SUBTILES for _ in range_constexpr(KV_SUBTILES)]
            for ng in range_constexpr(KV_SUBTILES):
                for qg in range_constexpr(Q_SUBTILES):
                    vals = s[ng][qg]
                    if const_expr(keep is not None):
                        vals = [
                            keep[ng][qg][i].select(vals[i], c_zero_f)
                            for i in range_constexpr(4)
                        ]
                    p_packs[ng][qg] = pack_mfma_frag(
                        silu_gate(vals, c_alpha), elem_dtype, truncate=True
                    )
            return p_packs

        def softmax_tile(s, keep, o_acc, m_run, l_run):
            """Online softmax in the log2 domain; returns P packs and the
            rescaled accumulators and row statistics."""
            p_packs = [[None] * Q_SUBTILES for _ in range_constexpr(KV_SUBTILES)]
            for qg in range_constexpr(Q_SUBTILES):
                t = [
                    [s[ng][qg][i] * c_alpha for i in range_constexpr(4)]
                    for ng in range_constexpr(KV_SUBTILES)
                ]
                if const_expr(keep is not None):
                    t = [
                        [
                            keep[ng][qg][i].select(t[ng][i], c_neg_big)
                            for i in range_constexpr(4)
                        ]
                        for ng in range_constexpr(KV_SUBTILES)
                    ]
                tile_max = t[0][0]
                for ng in range_constexpr(KV_SUBTILES):
                    for i in range_constexpr(4):
                        tile_max = fx.max(tile_max, t[ng][i])
                # The four lanes sharing a query differ in lane bits 4 and 5.
                tile_max = fx.max(tile_max, tile_max.shuffle_xor(fx.Int32(16), 64))
                tile_max = fx.max(tile_max, tile_max.shuffle_xor(fx.Int32(32), 64))
                m_new = fx.max(m_run[qg], tile_max)
                corr = fx.exp2(m_run[qg] - m_new, fastmath="afn")
                row_sum = c_zero_f
                for ng in range_constexpr(KV_SUBTILES):
                    p = [
                        fx.exp2(t[ng][i] - m_new, fastmath="afn")
                        for i in range_constexpr(4)
                    ]
                    if const_expr(keep is not None):
                        p = [
                            keep[ng][qg][i].select(p[i], c_zero_f)
                            for i in range_constexpr(4)
                        ]
                    for i in range_constexpr(4):
                        row_sum = row_sum + p[i]
                    # Probabilities need round-to-nearest; truncation biases
                    # every term low.
                    p_packs[ng][qg] = pack_mfma_frag(p, elem_dtype, truncate=False)
                l_run[qg] = l_run[qg] * corr + row_sum
                m_run[qg] = m_new
                for c in range_constexpr(D_CHUNKS):
                    acc = Vec(o_acc[c * Q_SUBTILES + qg])
                    o_acc[c * Q_SUBTILES + qg] = Vec.from_elements(
                        [acc[e] * corr for e in range_constexpr(4)], fx.Float32
                    ).ir_value()
            return p_packs

        # ---- GEMM2: O^T += V^T P^T. Lane holds query lane%16, d 4*(lane/16)+e.
        def accum_o_tile(o_acc, p_packs):
            for c in range_constexpr(D_CHUNKS):
                d_col = fx.Int32(c * MFMA_M) + lane_mod_16
                v_packs = [
                    v_smem[
                        d_col,
                        fx.Int32(ng * (MFMA_N // MFMA_LANE_K)) + lane_div_16,
                        None,
                    ]
                    .load()
                    .ir_value()
                    for ng in range_constexpr(KV_SUBTILES)
                ]
                for qg in range_constexpr(Q_SUBTILES):
                    cur = o_acc[c * Q_SUBTILES + qg]
                    for ng in range_constexpr(KV_SUBTILES):
                        cur = mfma_acc(v_packs[ng], p_packs[ng][qg], cur)
                    o_acc[c * Q_SUBTILES + qg] = cur
            return o_acc

        def run_kv_tile(state, kv_start, apply_mask, full_tile):
            o_acc = list(state[:N_ACC])
            k_vecs = async_load_k_regs(kv_start, full_tile)
            v_vecs = async_load_v_regs(kv_start, full_tile)
            rocdl.s_waitcnt(vmcnt=NUM_BATCHES_V)
            store_k_regs_to_lds(k_vecs)
            gpu.barrier()
            k_packs = [read_k_packs(ng) for ng in range_constexpr(KV_SUBTILES)]
            s = gemm1(k_packs)
            keep = tile_keep(kv_start) if apply_mask else None
            if const_expr(softmax):
                m_run = [fx.Float32(x) for x in state[N_ACC : N_ACC + Q_SUBTILES]]
                l_run = [fx.Float32(x) for x in state[N_ACC + Q_SUBTILES :]]
                p_packs = softmax_tile(s, keep, o_acc, m_run, l_run)
            else:
                p_packs = silu_tile(s, keep)
            rocdl.s_waitcnt(vmcnt=0)
            store_v_regs_to_lds(v_vecs)
            rocdl.sched_group_barrier(rocdl.mask_dswr, NUM_BATCHES_V, 0)
            gpu.barrier()
            o_acc = accum_o_tile(o_acc, p_packs)
            if const_expr(softmax):
                return (
                    o_acc
                    + [x.ir_value() for x in m_run]
                    + [x.ir_value() for x in l_run]
                )
            return o_acc

        if active:
            c_zero_v4 = fx.full(MFMA_ELEMS_PER_LANE, 0.0, fx.Float32).ir_value()
            init = [c_zero_v4 for _ in range(N_ACC)]
            if const_expr(softmax):
                init = (
                    init
                    + [c_neg_big.ir_value() for _ in range(Q_SUBTILES)]
                    + [c_zero_f.ir_value() for _ in range(Q_SUBTILES)]
                )

            # Contextual columns (window + contextual only; else an empty range).
            results = init
            for kv_tile, it in range(
                fx.Int32(0), fx.Int32(ctx_tiles), fx.Int32(1), init=init
            ):  # ty: ignore
                st = list(it) if isinstance(it, (list, tuple)) else [it]
                st = run_kv_tile(st, fx.Int32(kv_tile) * fx.Int32(BLOCK_N), True, False)
                results = yield st

            # Unmasked causal prefix (CAUSAL_SPLIT only; else an empty range).
            results2 = results
            for kv_tile, it in range(
                fx.Int32(kv_tile_start),
                fx.Int32(unmasked_end),
                fx.Int32(1),
                init=results,
            ):  # ty: ignore
                st = list(it) if isinstance(it, (list, tuple)) else [it]
                st = run_kv_tile(st, fx.Int32(kv_tile) * fx.Int32(BLOCK_N), False, True)
                results2 = yield st

            # Masked remainder.
            results3 = results2
            for kv_tile, it in range(
                fx.Int32(unmasked_end), fx.Int32(n_tiles), fx.Int32(1), init=results2
            ):  # ty: ignore
                st = list(it) if isinstance(it, (list, tuple)) else [it]
                st = run_kv_tile(st, fx.Int32(kv_tile) * fx.Int32(BLOCK_N), True, False)
                results3 = yield st

            final = (
                list(results3) if isinstance(results3, (list, tuple)) else [results3]
            )
            for qg in range_constexpr(Q_SUBTILES):
                row = q_rows[qg]
                if q_in_bounds[qg]:
                    row64 = fx.Int64(seq_start + row)
                    if const_expr(softmax):
                        l_part = fx.Float32(final[N_ACC + Q_SUBTILES + qg])
                        l_part = l_part + l_part.shuffle_xor(fx.Int32(16), 64)
                        l_tot = l_part + l_part.shuffle_xor(fx.Int32(32), 64)
                        factor = fx.Float32(
                            fx.rocdl.rcp(fx.Float32.ir_type, l_tot.ir_value())
                        )
                        m_fin = fx.Float32(final[N_ACC + qg])
                        if lane_div_16 == fx.Int32(0):
                            lse[row64, head_idx] = m_fin + fx.Float32(
                                fx.log2(l_tot.ir_value(), fastmath="afn")
                            )
                    else:
                        factor = out_scale
                        if const_expr(scale_mode == "scalar"):
                            factor = factor * fx.Float32(scale[fx.Int32(0)])
                        if const_expr(scale_mode == "per_token"):
                            factor = factor * fx.Float32(scale[seq_start + row])
                    for c in range_constexpr(D_CHUNKS):
                        acc = Vec(final[c * Q_SUBTILES + qg])
                        vals = Vec.from_elements(
                            [
                                (acc[e] * factor).to(elem_dtype)
                                for e in range_constexpr(4)
                            ],
                            elem_dtype,
                        )
                        o_store(
                            row64,
                            head_idx,
                            fx.Int32(c * (MFMA_M // 4)) + lane_div_16,
                            vals,
                        )

    @flyc.jit
    def launch(
        num_q_tiles: fx.Int32,
        batch: fx.Int32,
        q: fx.Tensor,
        k: fx.Tensor,
        v: fx.Tensor,
        seq_offsets: fx.Tensor,
        num_targets: fx.Tensor,
        scale: fx.Tensor,
        out: fx.Tensor,
        lse: fx.Tensor,
        window: fx.Int32,
        contextual: fx.Int32,
        full: fx.Int32,
        c_alpha: fx.Float32,
        out_scale: fx.Float32,
        stream: fx.Stream,
    ) -> None:
        hz_total = batch * fx.Int32(launch_heads)
        hz_per_group = (hz_total + fx.Int32(NUM_GRID_GROUPS - 1)) // fx.Int32(
            NUM_GRID_GROUPS
        )
        grid_blocks = num_q_tiles * hz_per_group * fx.Int32(NUM_GRID_GROUPS)
        hstu_ragged_fwd_kernel(
            q,
            k,
            v,
            seq_offsets,
            num_targets,
            scale,
            out,
            lse,
            num_q_tiles,
            hz_per_group,
            hz_total,
            window,
            contextual,
            full,
            c_alpha,
            out_scale,
            value_attrs={
                "rocdl.waves_per_eu": waves_per_eu,
                "rocdl.flat_work_group_size": f"{BLOCK_THREADS},{BLOCK_THREADS}",
            },
        ).launch(grid=(grid_blocks, 1, 1), block=BLOCK_THREADS, smem=0, stream=stream)

    launch.compile_hints = {"fast_fp_math": True, "unsafe_fp_math": True}
    return launch


# ---- Host side ----------------------------------------------------------------


def default_config(head_dim: int, hidden_dim: int) -> Dict[str, int]:
    """Tile config used when autotuning is off."""
    dim = max(hidden_dim, (head_dim + 63) // 64 * 64)
    if dim >= 256:
        return {"block_m": 128, "block_n": 16, "num_waves": 4, "waves_per_eu": 0}
    if dim >= 128:
        return {"block_m": 128, "block_n": 64, "num_waves": 4, "waves_per_eu": 2}
    return {"block_m": 128, "block_n": 32, "num_waves": 4, "waves_per_eu": 2}


_TILES = tuple(
    {"block_m": bm, "block_n": bn, "num_waves": nw, "waves_per_eu": wpe}
    for bm in (64, 128)
    for bn in (16, 32, 64)
    for nw in (4,)
    for wpe in (0, 2)
)


def _launch(
    q,
    k,
    v,
    seq_offsets,
    num_targets,
    scale,
    out,
    lse,
    n_bucket,
    batch,
    spec,
    *,
    max_seq_len,
    window,
    contextual,
    full,
    c_alpha,
    out_scale,
    block_m,
    block_n,
    num_waves,
    waves_per_eu,
):
    """Compile (cached) and launch one head range. ``n_bucket``, ``batch`` and
    ``spec`` (the compile-time key) feed the autotune key."""
    from mslk.flydsl.jit import run_compiled

    launcher = compile_hstu_ragged_fwd(
        **dict(spec),
        block_m=block_m,
        block_n=block_n,
        num_waves=num_waves,
        waves_per_eu=waves_per_eu,
    )
    run_compiled(
        launcher,
        (max_seq_len + block_m - 1) // block_m,
        batch,
        q,
        k,
        v,
        seq_offsets,
        num_targets,
        scale,
        out,
        lse,
        window,
        contextual,
        full,
        c_alpha,
        out_scale,
        torch.cuda.current_stream(),
    )
    return out


@functools.lru_cache(maxsize=None)
def _tuned_launch(head_dim: int, hidden_dim: int):
    return tunable(
        configs=_TILES,
        default=default_config(head_dim, hidden_dim),
        key=["n_bucket", "batch", "spec"],
    )(_launch)


_DTYPE_STR = {torch.bfloat16: "bf16", torch.float16: "f16"}


def hstu_ragged_fwd(
    N: int,
    alpha: float,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    seq_offsets: torch.Tensor,
    invalid_attn_mask_type: str = "lower_triangular",
    num_targets: Optional[torch.Tensor] = None,
    attn_scale: Optional[torch.Tensor] = None,
    max_attn_len: int = 0,
    contextual_seq_len: int = 0,
    full_attn_size: int = 0,
    num_softmax_heads: int = 0,
    config: Optional[Dict[str, int]] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Ragged HSTU forward; same arguments and results as
    :func:`mslk.attention.hstu._torch_ref.ragged_hstu_attention_ref` (pass
    ``v=k`` for shared KV).

    ``N`` must be at least the longest sequence: rows past ``N`` are not
    computed. Nothing is synchronized to the host, so this is graph-safe once
    the specialization is compiled. ``config`` overrides the tile config.
    """
    from mslk.attention.hstu._torch_ref import _check_mask_options

    if q.ndim != 3 or k.ndim != 3 or v.ndim != 3:
        raise ValueError("q, k and v must be [tokens, heads, dim] tensors")
    if q.shape != k.shape or k.shape[:2] != v.shape[:2]:
        raise ValueError(
            f"q/k/v shapes disagree: {tuple(q.shape)}, {tuple(k.shape)}, "
            f"{tuple(v.shape)}"
        )
    if q.dtype not in _DTYPE_STR or k.dtype != q.dtype or v.dtype != q.dtype:
        raise ValueError(f"q/k/v must all be bf16 or f16, got {q.dtype}")
    _check_mask_options(
        invalid_attn_mask_type, max_attn_len, contextual_seq_len, full_attn_size
    )
    total, heads, head_dim = q.shape
    hidden_dim = v.shape[2]
    if not 0 <= num_softmax_heads <= heads:
        raise ValueError(f"num_softmax_heads must be in [0, {heads}]")
    batch = seq_offsets.numel() - 1
    if seq_offsets.ndim != 1 or batch < 1:
        raise ValueError("seq_offsets must be 1-D with at least 2 entries")
    if num_targets is not None and num_targets.shape != (batch,):
        raise ValueError(f"num_targets must have shape [{batch}]")

    dev = q.device
    out = torch.empty(total, heads, hidden_dim, dtype=q.dtype, device=dev)
    lse = torch.empty(total, num_softmax_heads, dtype=torch.float32, device=dev)
    if total == 0:
        return out, lse

    q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
    offsets = seq_offsets.to(torch.int32).contiguous()
    dummy = torch.zeros(1, dtype=torch.int32, device=dev)
    targets = dummy if num_targets is None else num_targets.to(torch.int32)
    has_window = max_attn_len > 0
    feats = RaggedFeatures(
        mask_type=invalid_attn_mask_type,
        has_targets=num_targets is not None,
        has_window=has_window,
        has_contextual=contextual_seq_len > 0,
        has_full=has_window and full_attn_size > 0,
    )
    if attn_scale is None:
        scale_mode, scale = "inv_n", torch.zeros(1, dtype=torch.float32, device=dev)
    elif attn_scale.ndim == 0:
        scale_mode, scale = "scalar", attn_scale.reshape(1).float()
    else:
        if attn_scale.shape != (total,):
            raise ValueError(f"attn_scale must be 0-dim or have shape [{total}]")
        scale_mode, scale = "per_token", attn_scale.float().contiguous()

    common = {
        "num_heads": heads,
        "head_dim": head_dim,
        "hidden_dim": hidden_dim,
        "dtype_str": _DTYPE_STR[q.dtype],
        "feats": feats,
    }
    launch = _tuned_launch(head_dim, hidden_dim)
    if config is not None:
        launch = functools.partial(_launch, **config)
    ranges = [(0, num_softmax_heads, True), (num_softmax_heads, heads, False)]
    for begin, end, softmax in ranges:
        if begin == end:
            continue
        spec = tuple(
            sorted(
                dict(
                    common,
                    head_begin=begin,
                    launch_heads=end - begin,
                    softmax=softmax,
                    scale_mode=scale_mode if not softmax else "inv_n",
                ).items()
            )
        )
        launch(
            q,
            k,
            v,
            offsets,
            targets,
            scale,
            out,
            lse if softmax else scale,
            next_pow2(N),
            batch,
            spec,
            max_seq_len=N,
            window=max_attn_len,
            contextual=contextual_seq_len,
            full=full_attn_size,
            c_alpha=alpha * (LOG2E if softmax else -LOG2E),
            out_scale=(1.0 / N if scale_mode == "inv_n" else 1.0) * (-1.0 / LOG2E),
        )
    return out, lse


# ---- AOT interface ------------------------------------------------------------

AOT_ARCHS: List[str] = ["gfx942", "gfx950"]

AOT_CONFIGS: List[Dict[str, Any]] = [
    {"head_dim": d, "dtype_str": "bf16", "has_targets": t}
    for d in (64, 128, 256)
    for t in (False, True)
]


def compile_aot_config(config: Dict[str, Any], arch: str) -> None:
    """Precompile the common SiLU causal specialization of one config."""
    d = config["head_dim"]
    compile_hstu_ragged_fwd(
        num_heads=4,
        head_begin=0,
        launch_heads=4,
        head_dim=d,
        hidden_dim=d,
        dtype_str=config["dtype_str"],
        feats=RaggedFeatures(
            "lower_triangular", config["has_targets"], False, False, False
        ),
        softmax=False,
        scale_mode="inv_n",
        arch=arch,
        **default_config(d, d),
    )
