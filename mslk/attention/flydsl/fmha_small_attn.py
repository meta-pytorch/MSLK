# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Small-attention FMHA kernels for large head dim (D>256) with small sequences.

Forward and backward for the regime where the entire Q/K/V/dO fit in gfx950's
160KB LDS.  One workgroup per (batch, head) loads everything into LDS, computes
the full score matrix in registers (accumulated over D in 16-element MFMA
chunks), and writes O/gradients back.

Target workload: B=1172, H=1, Lq=16, Lkv=25, D=512 (bf16) -- W3.

Gate: 256 < D <= 512, (2*Lq + 2*Lkv) * D * 2 <= 160*1024.
Arch: gfx950 only (mfma_f32_32x32x16_bf16, 160KB LDS).
"""

import math as _math

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import arith, const_expr, gpu, math as fly_math, range_constexpr, rocdl
from flydsl.expr.typing import Vector as Vec
from flydsl.expr.utils.arith import _to_raw as _raw
from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr
from mslk.attention.flydsl.fmha_bwd_mfma import (
    _grad_ds_unscaled,
    _LOG2E,
    _softmax_p,
    CAUSAL_BOTTOM_RIGHT,
    CAUSAL_NONE,
    CAUSAL_TOP_LEFT,
    dtype_to_elem_type,
    WARP_SIZE,
)
from mslk.flydsl import buffer_ops

_GFX950_LDS_BYTES = 160 * 1024


def small_attn_lds_bytes(MAX_LQ: int, MAX_LKV: int, D: int, backward: bool) -> int:
    """LDS footprint in bytes for the small-attention kernel."""
    bf16 = 2
    q_bytes = MAX_LQ * D * bf16
    k_bytes = MAX_LKV * D * bf16
    v_bytes = MAX_LKV * D * bf16
    do_bytes = MAX_LQ * D * bf16 if backward else 0
    return q_bytes + k_bytes + v_bytes + do_bytes


def small_attn_fits(D: int, Lq: int, Lkv: int) -> bool:
    """True when the entire Q+K+V+dO fits in gfx950 LDS for backward."""
    if D <= 256:
        return False
    return small_attn_lds_bytes(Lq, Lkv, D, backward=True) <= _GFX950_LDS_BYTES


def _pad32(n: int) -> int:
    return ((n + 31) // 32) * 32


def compile_fmha_small_attn_fwd(
    *,
    D: int = 512,
    dtype_str: str = "bf16",
    scale: float = None,
    causal_mode: int = CAUSAL_NONE,
    MAX_LQ: int = 32,
    MAX_LKV: int = 32,
    output_dtype_str: str = None,
):
    """Compile a small-attention forward kernel.

    One workgroup per (batch, head).  Loads Q [Lq, D] and K [Lkv, D] into LDS,
    computes S = QK^T in registers (accumulated over D in MFMA-K=16 chunks),
    applies softmax, loads V, computes O = P @ V, stores O and LSE.

    Args:
        D: head dimension (must be a multiple of 32, 256 < D <= 512)
        dtype_str: "bf16" or "f16"
        scale: softmax scale (default 1/sqrt(D))
        causal_mode: CAUSAL_NONE, CAUSAL_TOP_LEFT, or CAUSAL_BOTTOM_RIGHT
        MAX_LQ: compile-time upper bound on Lq (padded to multiple of 32)
        MAX_LKV: compile-time upper bound on Lkv (padded to multiple of 32)

    Returns:
        launch_fn(Q_i16, K_i16, V_i16, O_i16, LSE_f32,
                  B, Lq, Lkv, H, stream)
    """
    import math as _pm

    if scale is None:
        scale = 1.0 / _pm.sqrt(D)

    MAX_LQ = _pad32(MAX_LQ)
    MAX_LKV = _pad32(MAX_LKV)

    assert D % 32 == 0 and D > 256 and D <= 512
    assert small_attn_lds_bytes(MAX_LQ, MAX_LKV, D, backward=False) <= _GFX950_LDS_BYTES

    elem_dtype = dtype_to_elem_type(dtype_str)
    out_elem_dtype = (
        dtype_to_elem_type(output_dtype_str) if output_dtype_str else None
    )
    MFMA_K = 16
    MFMA_LK = 8  # lanes per K-dim half
    K_STEPS = D // MFMA_K
    fm = arith.FastMathFlags.fast

    BLOCK_SIZE = 256
    NUM_WAVES = BLOCK_SIZE // WARP_SIZE  # 4

    LDS_Q_STRIDE = D + 8
    LDS_K_STRIDE = D + 8

    LDS_Q_ELEMS = MAX_LQ * LDS_Q_STRIDE
    LDS_K_ELEMS = MAX_LKV * LDS_K_STRIDE
    LDS_V_ELEMS = MAX_LKV * LDS_K_STRIDE

    allocator = SmemAllocator(
        None, arch="gfx950", global_sym_name="fmha_small_attn_fwd_smem"
    )
    lds_q_off = allocator._align(allocator.ptr, 16)
    allocator.ptr = lds_q_off + LDS_Q_ELEMS * 2
    lds_k_off = allocator._align(allocator.ptr, 16)
    allocator.ptr = lds_k_off + LDS_K_ELEMS * 2
    lds_v_off = allocator._align(allocator.ptr, 16)
    allocator.ptr = lds_v_off + LDS_V_ELEMS * 2

    log2e_scale = _LOG2E * scale

    @flyc.kernel(known_block_size=[BLOCK_SIZE, 1, 1])
    def fmha_small_attn_fwd_kernel(
        Q: fx.Tensor,
        K: fx.Tensor,
        V: fx.Tensor,
        Out: fx.Tensor,
        LSE_out: fx.Tensor,
        seq_Q: fx.Int32,
        seq_KV: fx.Int32,
        n_heads: fx.Int32,
    ):
        v4elem_type = Vec.make_type(MFMA_LK // 2, elem_dtype)
        v16f32_type = Vec.make_type(16, fx.Float32)
        bid = fx.block_idx.x
        tid = fx.thread_idx.x

        n_heads_idx = fx.Index(n_heads)
        seq_Q_idx = fx.Index(seq_Q)
        seq_KV_idx = fx.Index(seq_KV)

        bid_idx = fx.Index(bid)
        batch_idx = bid_idx // n_heads_idx
        head_idx = bid_idx % n_heads_idx

        wave = fx.Index(tid // WARP_SIZE)
        lane = fx.Index(tid % WARP_SIZE)
        lane_mod_32 = fx.Index(lane % 32)
        lane_div_32 = fx.Index(lane // 32)

        base_ptr = allocator.get_base()
        lds_q = SmemPtr(
            base_ptr, lds_q_off, elem_dtype.ir_type, shape=(LDS_Q_ELEMS,)
        ).get()
        lds_k = SmemPtr(
            base_ptr, lds_k_off, elem_dtype.ir_type, shape=(LDS_K_ELEMS,)
        ).get()
        lds_v = SmemPtr(
            base_ptr, lds_v_off, elem_dtype.ir_type, shape=(LDS_V_ELEMS,)
        ).get()

        q_rsrc = buffer_ops.create_buffer_resource(Q)
        k_rsrc = buffer_ops.create_buffer_resource(K)
        v_rsrc = buffer_ops.create_buffer_resource(V)
        o_rsrc = buffer_ops.create_buffer_resource(Out)
        lse_rsrc = buffer_ops.create_buffer_resource(LSE_out)

        def _q_row(q_pos):
            return fx.Int32((batch_idx * seq_Q_idx + q_pos) * n_heads_idx + head_idx)

        def _kv_row(kv_pos):
            return fx.Int32((batch_idx * seq_KV_idx + kv_pos) * n_heads_idx + head_idx)

        def _load_vec(rsrc, row_i32, col_offset_idx):
            flat_elem = fx.Index(row_i32) * fx.Index(D) + col_offset_idx
            return buffer_ops.buffer_load(
                rsrc, flat_elem, vec_width=MFMA_LK, dtype=elem_dtype
            )

        # -- Cooperative load: Q -> LDS_Q, K -> LDS_K --
        VEC_COLS = D // MFMA_LK
        TOTAL_Q_ITEMS = MAX_LQ * VEC_COLS
        TOTAL_K_ITEMS = MAX_LKV * VEC_COLS
        ITEMS_PER_THREAD_Q = (TOTAL_Q_ITEMS + BLOCK_SIZE - 1) // BLOCK_SIZE
        ITEMS_PER_THREAD_K = (TOTAL_K_ITEMS + BLOCK_SIZE - 1) // BLOCK_SIZE

        tid_idx = fx.Index(tid)
        for it in range_constexpr(ITEMS_PER_THREAD_Q):
            item = tid_idx + fx.Index(it * BLOCK_SIZE)
            row_i = item // fx.Index(VEC_COLS)
            cv_i = item % fx.Index(VEC_COLS)
            q_valid = row_i < seq_Q_idx
            q_safe = q_valid.select(row_i, seq_Q_idx - fx.Index(1))
            q_row_g = _q_row(q_safe)
            col_off = cv_i * fx.Index(MFMA_LK)
            q_vec = _load_vec(q_rsrc, q_row_g, col_off)
            lds_base = row_i * LDS_Q_STRIDE + col_off
            Vec(q_vec).store(lds_q, [lds_base])

        for it in range_constexpr(ITEMS_PER_THREAD_K):
            item = tid_idx + fx.Index(it * BLOCK_SIZE)
            row_i = item // fx.Index(VEC_COLS)
            cv_i = item % fx.Index(VEC_COLS)
            k_valid = row_i < seq_KV_idx
            k_safe = k_valid.select(row_i, seq_KV_idx - fx.Index(1))
            kv_row_g = _kv_row(k_safe)
            col_off = cv_i * fx.Index(MFMA_LK)
            k_vec = _load_vec(k_rsrc, kv_row_g, col_off)
            lds_base = row_i * LDS_K_STRIDE + col_off
            Vec(k_vec).store(lds_k, [lds_base])

        gpu.barrier()

        # -- Compute S = Q @ K^T, one 32x32 MFMA tile at a time --
        # Wave assignment: each wave handles one 32-row sub-block of Q.
        # With MAX_LQ=32, only wave 0 has real work in the Q dimension.
        # With MAX_LKV=32, only one N-sub-tile.
        Q_SUBS = MAX_LQ // 32
        KV_SUBS = MAX_LKV // 32

        def mfma(a_pack, b_pack, c_acc):
            if const_expr(dtype_str == "bf16"):
                return rocdl.mfma_f32_32x32x16_bf16(
                    v16f32_type, [a_pack, b_pack, c_acc]
                )
            return rocdl.mfma_f32_32x32x16_f16(
                v16f32_type, [a_pack, b_pack, c_acc]
            )

        def _lds_load_q(q_sub, k_step):
            lds_row = fx.Index(q_sub * 32) + lane_mod_32
            lds_col_lo = fx.Index(k_step * MFMA_K) + lane_div_32 * MFMA_LK
            lds_col_hi = lds_col_lo + fx.Index(MFMA_LK // 2)
            lo = Vec.load(v4elem_type, lds_q, [lds_row * LDS_Q_STRIDE + lds_col_lo])
            hi = Vec.load(v4elem_type, lds_q, [lds_row * LDS_Q_STRIDE + lds_col_hi])
            return Vec(lo).shuffle(Vec(hi), list(range(MFMA_LK))).ir_value()

        def _lds_load_k(kv_sub, k_step):
            lds_row = fx.Index(kv_sub * 32) + lane_mod_32
            lds_col_lo = fx.Index(k_step * MFMA_K) + lane_div_32 * MFMA_LK
            lds_col_hi = lds_col_lo + fx.Index(MFMA_LK // 2)
            lo = Vec.load(v4elem_type, lds_k, [lds_row * LDS_K_STRIDE + lds_col_lo])
            hi = Vec.load(v4elem_type, lds_k, [lds_row * LDS_K_STRIDE + lds_col_hi])
            return Vec(lo).shuffle(Vec(hi), list(range(MFMA_LK))).ir_value()

        # S[q_sub, kv_sub] accumulators -- each is a v16f32 MFMA result
        s_accs = {}
        for qs in range_constexpr(Q_SUBS):
            for kvs in range_constexpr(KV_SUBS):
                s_accs[(qs, kvs)] = Vec.filled(16, 0.0, fx.Float32)

        # Accumulate S = Q @ K^T over D dimension
        for ks in range_constexpr(K_STEPS):
            q_packs = [_lds_load_q(qs, ks) for qs in range(Q_SUBS)]
            k_packs = [_lds_load_k(kvs, ks) for kvs in range(KV_SUBS)]
            for qs in range_constexpr(Q_SUBS):
                for kvs in range_constexpr(KV_SUBS):
                    s_accs[(qs, kvs)] = mfma(q_packs[qs], k_packs[kvs], s_accs[(qs, kvs)])

        # -- Softmax: scale, mask, row-max, exp2, row-sum, normalize --
        # MFMA 32x32x16 output layout: lane j owns n=j%32, r gives m position.
        # Softmax is per-element: scale S, apply mask, then cross-lane reduce
        # for row-max and row-sum.
        log2e_scale_cst = fx.Float32(log2e_scale)

        def _s_elem_scaled(qs, kvs, r):
            """Read S[qs,kvs][r], apply log2e*scale, apply validity/causal mask."""
            s_raw = Vec(s_accs[(qs, kvs)])[r]
            scaled_val = fx.Float32(
                arith.mulf(_raw(s_raw), _raw(log2e_scale_cst), fastmath=fm)
            )
            m_within = lane_div_32 * fx.Index(4) + fx.Index((r // 4) * 8 + (r % 4))
            n_within = lane_mod_32
            m_abs = fx.Index(qs * 32) + m_within
            n_abs = fx.Index(kvs * 32) + n_within
            valid = (m_abs < seq_Q_idx) & (n_abs < seq_KV_idx)
            if const_expr(causal_mode == CAUSAL_TOP_LEFT):
                valid = valid & (n_abs <= m_abs)
            elif const_expr(causal_mode == CAUSAL_BOTTOM_RIGHT):
                valid = valid & ((n_abs + seq_Q_idx) <= (m_abs + seq_KV_idx))
            return valid.select(scaled_val, fx.Float32(-1e30))

        # Step 1: Per-lane max across KV_SUBS, then cross-lane max (32 lanes)
        row_maxes = {}
        for qs in range_constexpr(Q_SUBS):
            for r in range_constexpr(16):
                rmax = _s_elem_scaled(qs, 0, r)
                for kvs in range_constexpr(1, KV_SUBS):
                    v = _s_elem_scaled(qs, kvs, r)
                    rmax = fx.Float32(arith.maximumf(_raw(rmax), _raw(v), fastmath=fm))
                for offset in [16, 8, 4, 2, 1]:
                    other = fx.Float32(
                        gpu.shuffle_xor(_raw(rmax), fx.Int32(offset), WARP_SIZE)
                    )
                    rmax = fx.Float32(
                        arith.maximumf(_raw(rmax), _raw(other), fastmath=fm)
                    )
                row_maxes[(qs, r)] = rmax

        # Step 2: exp2(S_scaled - max), row-sum, cross-lane sum, normalize -> P
        row_sums = {}
        p_vals = {}
        for qs in range_constexpr(Q_SUBS):
            for r in range_constexpr(16):
                rsum = fx.Float32(0.0)
                for kvs in range_constexpr(KV_SUBS):
                    s_scaled = _s_elem_scaled(qs, kvs, r)
                    diff = fx.Float32(
                        arith.subf(_raw(s_scaled), _raw(row_maxes[(qs, r)]), fastmath=fm)
                    )
                    p = fx.Float32(fly_math.exp2(diff, fastmath=fm))
                    p_vals[(qs, kvs, r)] = p
                    rsum = fx.Float32(arith.addf(_raw(rsum), _raw(p), fastmath=fm))
                for offset in [16, 8, 4, 2, 1]:
                    other = fx.Float32(
                        gpu.shuffle_xor(_raw(rsum), fx.Int32(offset), WARP_SIZE)
                    )
                    rsum = fx.Float32(arith.addf(_raw(rsum), _raw(other), fastmath=fm))
                row_sums[(qs, r)] = rsum

        # Normalize P = exp2(S-max) / sum
        for qs in range_constexpr(Q_SUBS):
            for kvs in range_constexpr(KV_SUBS):
                for r in range_constexpr(16):
                    p = p_vals[(qs, kvs, r)]
                    inv_sum = fx.Float32(arith.divf(
                        _raw(fx.Float32(1.0)), _raw(row_sums[(qs, r)]), fastmath=fm
                    ))
                    p_vals[(qs, kvs, r)] = fx.Float32(
                        arith.mulf(_raw(p), _raw(inv_sum), fastmath=fm)
                    )

        # Step 3: LSE = (max + log2(sum)) / log2(e)
        inv_log2e = fx.Float32(1.0 / _LOG2E)
        for qs in range_constexpr(Q_SUBS):
            for r in range_constexpr(16):
                log2_sum = fx.Float32(fly_math.log2(_raw(row_sums[(qs, r)]), fastmath=fm))
                lse_log2 = fx.Float32(
                    arith.addf(_raw(row_maxes[(qs, r)]), _raw(log2_sum), fastmath=fm)
                )
                lse_val = fx.Float32(
                    arith.mulf(_raw(lse_log2), _raw(inv_log2e), fastmath=fm)
                )
                # Store LSE: only one lane per m-row should store
                # m_abs = qs*32 + (lane//32)*4 + (r//4)*8 + (r%4)
                m_within = lane_div_32 * fx.Index(4) + fx.Index((r // 4) * 8 + (r % 4))
                m_abs = fx.Index(qs * 32) + m_within
                # Only lane%32 == 0 stores (one store per m-row)
                store_ok = (lane_mod_32 == fx.Index(0)) & (m_abs < seq_Q_idx)
                if store_ok:
                    lse_flat = fx.Int32(
                        (batch_idx * n_heads_idx + head_idx) * seq_Q_idx + m_abs
                    )
                    buffer_ops.buffer_store(
                        _raw(lse_val), lse_rsrc, fx.Index(lse_flat),
                    )

        # -- Load V into LDS in row-major [kv, D] layout --
        gpu.barrier()

        for it in range_constexpr(ITEMS_PER_THREAD_K):
            item = tid_idx + fx.Index(it * BLOCK_SIZE)
            row_i = item // fx.Index(VEC_COLS)
            cv_i = item % fx.Index(VEC_COLS)
            v_valid = row_i < seq_KV_idx
            v_safe = v_valid.select(row_i, seq_KV_idx - fx.Index(1))
            v_row_g = _kv_row(v_safe)
            col_off = cv_i * fx.Index(MFMA_LK)
            v_vec = _load_vec(v_rsrc, v_row_g, col_off)
            lds_base = row_i * LDS_K_STRIDE + col_off
            Vec(v_vec).store(lds_v, [lds_base])

        gpu.barrier()

        # -- Compute O = P @ V, accumulated over KV_SUBS and D chunks --
        # P is in registers: p_vals[(qs, kvs, r)] is f32 per lane.
        # V is in LDS: lds_v[kv_row, d_col].
        # O[m, d] = sum_n P[m, n] * V[n, d]
        #
        # For P @ V, we need P as the A-operand in an MFMA. But P is in the MFMA
        # output layout (scrambled m). Instead of doing a full MFMA for the PV
        # multiply, we write P to LDS in transposed layout [N, M_PAD], then use
        # it as an A-operand in MFMA(P^T_col, V_col) for each D chunk.
        #
        # Alternative: manually accumulate in registers. Since Lkv is small (<=32),
        # we can do a register-level multiply-accumulate: for each n in 0..Lkv-1,
        # broadcast P[m,n] and multiply with V[n, d].
        # For a 32x32 tile, each lane holds one n-value (lane%32 = n).
        # V[n, d]: load from LDS for each d-chunk.

        # Strategy: Store P to LDS in [N, MPAD] layout for MFMA-based PV,
        # or do a simple loop. Let's use LDS for P and MFMA.

        # Write P to LDS in transposed form [N, MPAD] where MPAD = MAX_LQ + 8.
        # Each lane writes its P values: lane%32 = n column, registers = m rows.
        # Transposed: lds_p[n, m] = P[m, n]

        LDS_NPAD = MAX_LKV + 8
        # Store P in row-major [M, NPAD] layout in lds_k (K is consumed).
        # A-operand reads P[m, n_chunk] contiguously along n.

        for qs in range_constexpr(Q_SUBS):
            for kvs in range_constexpr(KV_SUBS):
                for r in range_constexpr(16):
                    p_f32 = p_vals[(qs, kvs, r)]
                    n_within = lane_mod_32
                    m_within = lane_div_32 * fx.Index(4) + fx.Index((r // 4) * 8 + (r % 4))
                    n_abs = fx.Index(kvs * 32) + n_within
                    m_abs = fx.Index(qs * 32) + m_within
                    lds_idx = m_abs * LDS_NPAD + n_abs
                    Vec.from_elements([p_f32], fx.Float32).to(elem_dtype).store(
                        lds_k, [lds_idx]
                    )

        gpu.barrier()

        # O = P @ V via MFMA: C[m,n] += A[m,k] * B[k,n]
        # A = P[m, n_chunk], B = V[n_chunk, d_tile]
        # A-operand: lane j reads P[j%32, k_offset] contiguously along n in row-major
        def _lds_load_p_a(qs, kv_chunk, k_sub):
            m_lane = lane_mod_32 + fx.Index(qs * 32)
            k_base = fx.Index(kv_chunk * MFMA_K) + lane_div_32 * MFMA_LK
            k_lo = k_base
            k_hi = k_base + fx.Index(MFMA_LK // 2)
            lo = Vec.load(v4elem_type, lds_k, [m_lane * LDS_NPAD + k_lo])
            hi = Vec.load(v4elem_type, lds_k, [m_lane * LDS_NPAD + k_hi])
            return Vec(lo).shuffle(Vec(hi), list(range(MFMA_LK))).ir_value()

        def _lds_load_v_b(kvs, d_sub, kv_chunk):
            # B-operand: B[k, n] in MFMA layout -> free=j%32 -> n (output d_col),
            #   k_dim = j//32*LK + e -> k (kv row within chunk).
            # V is row-major [kv, D] in LDS. Elements at different kv rows
            # at the same d_col are strided by LDS_K_STRIDE. Load scalars and
            # build the v8 operand.
            d_col = lane_mod_32 + fx.Index(d_sub * 32)
            kv_base = fx.Index(kvs * 32 + kv_chunk * MFMA_K) + lane_div_32 * MFMA_LK
            v1_type = Vec.make_type(1, elem_dtype)
            elems = []
            for e in range_constexpr(MFMA_LK):
                kv_r = kv_base + fx.Index(e)
                lds_idx = kv_r * LDS_K_STRIDE + d_col
                v1 = Vec.load(v1_type, lds_v, [lds_idx])
                elems.append(Vec(v1)[0])
            return Vec.from_elements(elems, elem_dtype).ir_value()

        # D is split into 32-col output sub-tiles. Each wave handles some.
        D_SUBS = D // 32
        D_SUBS_PER_WAVE = -(-D_SUBS // NUM_WAVES)
        KV_K_STEPS = MAX_LKV // MFMA_K

        # O accumulators: one v16f32 per (qs, d_sub)
        # Distribute D-subs across waves
        for d_sub_local in range_constexpr(D_SUBS_PER_WAVE):
            d_sub = wave * D_SUBS_PER_WAVE + d_sub_local
            d_in_range = d_sub < fx.Index(D_SUBS)
            d_sub_safe = d_in_range.select(d_sub, fx.Index(0))

            for qs in range_constexpr(Q_SUBS):
                o_acc = Vec.filled(16, 0.0, fx.Float32)
                for kv_chunk in range_constexpr(KV_K_STEPS):
                    a_pack = _lds_load_p_a(qs, kv_chunk, 0)
                    b_pack = _lds_load_v_b(0, d_sub_safe, kv_chunk)
                    o_acc = mfma(a_pack, b_pack, o_acc)

                # Store O[m, d] to global memory
                for r in range_constexpr(16):
                    m_within = lane_div_32 * fx.Index(4) + fx.Index((r // 4) * 8 + (r % 4))
                    m_abs = fx.Index(qs * 32) + m_within
                    d_col = lane_mod_32 + d_sub_safe * fx.Index(32)
                    m_ok = (m_abs < seq_Q_idx) & d_in_range & (d_col < fx.Index(D))

                    if m_ok:
                        o_val = Vec(o_acc)[r]
                        o_row = _q_row(m_abs)
                        o_flat = fx.Index(o_row) * fx.Index(D) + d_col
                        if const_expr(out_elem_dtype is not None):
                            o_trunc = Vec.from_elements(
                                [o_val], fx.Float32
                            ).to(out_elem_dtype)
                            buffer_ops.buffer_store(
                                _raw(Vec(o_trunc)[0]), o_rsrc, o_flat,
                            )
                        else:
                            buffer_ops.buffer_store(
                                _raw(o_val), o_rsrc, o_flat,
                            )

    @flyc.jit
    def launch_fn(
        Q: fx.Tensor,
        K: fx.Tensor,
        V: fx.Tensor,
        Out: fx.Tensor,
        LSE_out: fx.Tensor,
        B: fx.Int32,
        Lq: fx.Int32,
        Lkv: fx.Int32,
        H: fx.Int32,
        stream: fx.Stream,
    ):
        from flydsl._mlir import ir
        from flydsl.compiler.kernel_function import CompilationContext

        allocator.finalized = False
        _ctx = CompilationContext.get_current()
        with ir.InsertionPoint(_ctx.gpu_module_body):
            allocator.finalize()

        grid_x = fx.Int32(fx.Index(B) * fx.Index(H))
        fmha_small_attn_fwd_kernel(
            Q, K, V, Out, LSE_out, Lq, Lkv, H,
        ).launch(
            grid=(grid_x, 1, 1),
            block=(BLOCK_SIZE, 1, 1),
            stream=stream,
        )

    return launch_fn


def compile_fmha_small_attn_bwd(
    *,
    D: int = 512,
    dtype_str: str = "bf16",
    scale: float = None,
    causal_mode: int = CAUSAL_NONE,
    MAX_LQ: int = 32,
    MAX_LKV: int = 32,
):
    """Compile a small-attention backward kernel.

    One workgroup per (batch, head).  Loads Q, K, V, dO into LDS, recomputes
    P from stored LSE, and computes dQ, dK, dV.

    Args:
        D: head dimension (must be a multiple of 32, 256 < D <= 512)
        dtype_str: "bf16" or "f16"
        scale: softmax scale (default 1/sqrt(D))
        causal_mode: CAUSAL_NONE, CAUSAL_TOP_LEFT, or CAUSAL_BOTTOM_RIGHT
        MAX_LQ: compile-time upper bound on Lq (padded to multiple of 32)
        MAX_LKV: compile-time upper bound on Lkv (padded to multiple of 32)

    Returns:
        launch_fn(Q_i16, K_i16, V_i16, dO_i16, dV_f32, dK_f32, dQ_f32,
                  LSE_f32, D_vec_f32, B, Lq, Lkv, H, stream)
    """
    import math as _pm

    if scale is None:
        scale = 1.0 / _pm.sqrt(D)

    MAX_LQ = _pad32(MAX_LQ)
    MAX_LKV = _pad32(MAX_LKV)

    assert D % 32 == 0 and D > 256 and D <= 512
    assert small_attn_lds_bytes(MAX_LQ, MAX_LKV, D, backward=True) <= _GFX950_LDS_BYTES

    elem_dtype = dtype_to_elem_type(dtype_str)
    MFMA_K = 16
    MFMA_LK = 8
    K_STEPS = D // MFMA_K
    fm = arith.FastMathFlags.fast

    BLOCK_SIZE = 256
    NUM_WAVES = BLOCK_SIZE // WARP_SIZE  # 4

    LDS_Q_STRIDE = D + 8
    LDS_K_STRIDE = D + 8
    LDS_DO_STRIDE = D + 8

    LDS_Q_ELEMS = MAX_LQ * LDS_Q_STRIDE
    LDS_K_ELEMS = MAX_LKV * LDS_K_STRIDE
    LDS_V_ELEMS = MAX_LKV * LDS_K_STRIDE
    LDS_DO_ELEMS = MAX_LQ * LDS_DO_STRIDE

    allocator = SmemAllocator(
        None, arch="gfx950", global_sym_name="fmha_small_attn_bwd_smem"
    )
    lds_q_off = allocator._align(allocator.ptr, 16)
    allocator.ptr = lds_q_off + LDS_Q_ELEMS * 2
    lds_k_off = allocator._align(allocator.ptr, 16)
    allocator.ptr = lds_k_off + LDS_K_ELEMS * 2
    lds_v_off = allocator._align(allocator.ptr, 16)
    allocator.ptr = lds_v_off + LDS_V_ELEMS * 2
    lds_do_off = allocator._align(allocator.ptr, 16)
    allocator.ptr = lds_do_off + LDS_DO_ELEMS * 2

    log2e_scale = _LOG2E * scale
    Q_SUBS = MAX_LQ // 32
    KV_SUBS = MAX_LKV // 32
    KV_K_STEPS = MAX_LKV // MFMA_K
    D_SUBS = D // 32
    D_SUBS_PER_WAVE = -(-D_SUBS // NUM_WAVES)
    LDS_MPAD = MAX_LQ + 8

    @flyc.kernel(known_block_size=[BLOCK_SIZE, 1, 1])
    def fmha_small_attn_bwd_kernel(
        Q: fx.Tensor,
        K: fx.Tensor,
        V: fx.Tensor,
        dO: fx.Tensor,
        dV_out: fx.Tensor,
        dK_out: fx.Tensor,
        dQ_out: fx.Tensor,
        LSE: fx.Tensor,
        D_vec: fx.Tensor,
        seq_Q: fx.Int32,
        seq_KV: fx.Int32,
        n_heads: fx.Int32,
    ):
        v4elem_type = Vec.make_type(MFMA_LK // 2, elem_dtype)
        v16f32_type = Vec.make_type(16, fx.Float32)
        bid = fx.block_idx.x
        tid = fx.thread_idx.x

        n_heads_idx = fx.Index(n_heads)
        seq_Q_idx = fx.Index(seq_Q)
        seq_KV_idx = fx.Index(seq_KV)

        bid_idx = fx.Index(bid)
        batch_idx = bid_idx // n_heads_idx
        head_idx = bid_idx % n_heads_idx

        wave = fx.Index(tid // WARP_SIZE)
        lane = fx.Index(tid % WARP_SIZE)
        lane_mod_32 = fx.Index(lane % 32)
        lane_div_32 = fx.Index(lane // 32)

        base_ptr = allocator.get_base()
        lds_q = SmemPtr(
            base_ptr, lds_q_off, elem_dtype.ir_type, shape=(LDS_Q_ELEMS,)
        ).get()
        lds_k = SmemPtr(
            base_ptr, lds_k_off, elem_dtype.ir_type, shape=(LDS_K_ELEMS,)
        ).get()
        lds_v = SmemPtr(
            base_ptr, lds_v_off, elem_dtype.ir_type, shape=(LDS_V_ELEMS,)
        ).get()
        lds_do = SmemPtr(
            base_ptr, lds_do_off, elem_dtype.ir_type, shape=(LDS_DO_ELEMS,)
        ).get()

        q_rsrc = buffer_ops.create_buffer_resource(Q)
        k_rsrc = buffer_ops.create_buffer_resource(K)
        v_rsrc = buffer_ops.create_buffer_resource(V)
        do_rsrc = buffer_ops.create_buffer_resource(dO)
        dv_rsrc = buffer_ops.create_buffer_resource(dV_out)
        dk_rsrc = buffer_ops.create_buffer_resource(dK_out)
        dq_rsrc = buffer_ops.create_buffer_resource(dQ_out)
        lse_rsrc = buffer_ops.create_buffer_resource(LSE)
        dvec_rsrc = buffer_ops.create_buffer_resource(D_vec)

        def _q_row(q_pos):
            return fx.Int32((batch_idx * seq_Q_idx + q_pos) * n_heads_idx + head_idx)

        def _kv_row(kv_pos):
            return fx.Int32((batch_idx * seq_KV_idx + kv_pos) * n_heads_idx + head_idx)

        def _load_vec(rsrc, row_i32, col_offset_idx):
            flat_elem = fx.Index(row_i32) * fx.Index(D) + col_offset_idx
            return buffer_ops.buffer_load(
                rsrc, flat_elem, vec_width=MFMA_LK, dtype=elem_dtype
            )

        def _lse_idx(q_pos):
            return fx.Int32(
                (batch_idx * n_heads_idx + head_idx) * seq_Q_idx + q_pos
            )

        def _dvec_idx(q_pos):
            return fx.Int32(
                (batch_idx * seq_Q_idx + q_pos) * n_heads_idx + head_idx
            )

        # -- Cooperative load: Q, K, V, dO -> LDS --
        VEC_COLS = D // MFMA_LK
        TOTAL_Q_ITEMS = MAX_LQ * VEC_COLS
        TOTAL_K_ITEMS = MAX_LKV * VEC_COLS
        ITEMS_PER_THREAD_Q = (TOTAL_Q_ITEMS + BLOCK_SIZE - 1) // BLOCK_SIZE
        ITEMS_PER_THREAD_K = (TOTAL_K_ITEMS + BLOCK_SIZE - 1) // BLOCK_SIZE

        tid_idx = fx.Index(tid)

        for it in range_constexpr(ITEMS_PER_THREAD_Q):
            item = tid_idx + fx.Index(it * BLOCK_SIZE)
            row_i = item // fx.Index(VEC_COLS)
            cv_i = item % fx.Index(VEC_COLS)
            q_valid = row_i < seq_Q_idx
            q_safe = q_valid.select(row_i, seq_Q_idx - fx.Index(1))
            q_row_g = _q_row(q_safe)
            col_off = cv_i * fx.Index(MFMA_LK)
            q_vec = _load_vec(q_rsrc, q_row_g, col_off)
            do_vec = _load_vec(do_rsrc, q_row_g, col_off)
            lds_base_q = row_i * LDS_Q_STRIDE + col_off
            lds_base_do = row_i * LDS_DO_STRIDE + col_off
            Vec(q_vec).store(lds_q, [lds_base_q])
            Vec(do_vec).store(lds_do, [lds_base_do])

        for it in range_constexpr(ITEMS_PER_THREAD_K):
            item = tid_idx + fx.Index(it * BLOCK_SIZE)
            row_i = item // fx.Index(VEC_COLS)
            cv_i = item % fx.Index(VEC_COLS)
            k_valid = row_i < seq_KV_idx
            k_safe = k_valid.select(row_i, seq_KV_idx - fx.Index(1))
            kv_row_g = _kv_row(k_safe)
            col_off = cv_i * fx.Index(MFMA_LK)
            k_vec = _load_vec(k_rsrc, kv_row_g, col_off)
            v_vec = _load_vec(v_rsrc, kv_row_g, col_off)
            lds_base = row_i * LDS_K_STRIDE + col_off
            Vec(k_vec).store(lds_k, [lds_base])
            Vec(v_vec).store(lds_v, [lds_base])

        gpu.barrier()

        # -- Recompute S = Q @ K^T, accumulated over D --
        def mfma(a_pack, b_pack, c_acc):
            if const_expr(dtype_str == "bf16"):
                return rocdl.mfma_f32_32x32x16_bf16(
                    v16f32_type, [a_pack, b_pack, c_acc]
                )
            return rocdl.mfma_f32_32x32x16_f16(
                v16f32_type, [a_pack, b_pack, c_acc]
            )

        def _lds_load_q_a(q_sub, k_step):
            lds_row = fx.Index(q_sub * 32) + lane_mod_32
            lds_col_lo = fx.Index(k_step * MFMA_K) + lane_div_32 * MFMA_LK
            lds_col_hi = lds_col_lo + fx.Index(MFMA_LK // 2)
            lo = Vec.load(v4elem_type, lds_q, [lds_row * LDS_Q_STRIDE + lds_col_lo])
            hi = Vec.load(v4elem_type, lds_q, [lds_row * LDS_Q_STRIDE + lds_col_hi])
            return Vec(lo).shuffle(Vec(hi), list(range(MFMA_LK))).ir_value()

        def _lds_load_k_b(kv_sub, k_step):
            lds_row = fx.Index(kv_sub * 32) + lane_mod_32
            lds_col_lo = fx.Index(k_step * MFMA_K) + lane_div_32 * MFMA_LK
            lds_col_hi = lds_col_lo + fx.Index(MFMA_LK // 2)
            lo = Vec.load(v4elem_type, lds_k, [lds_row * LDS_K_STRIDE + lds_col_lo])
            hi = Vec.load(v4elem_type, lds_k, [lds_row * LDS_K_STRIDE + lds_col_hi])
            return Vec(lo).shuffle(Vec(hi), list(range(MFMA_LK))).ir_value()

        def _lds_load_do_a(q_sub, k_step):
            lds_row = fx.Index(q_sub * 32) + lane_mod_32
            lds_col_lo = fx.Index(k_step * MFMA_K) + lane_div_32 * MFMA_LK
            lds_col_hi = lds_col_lo + fx.Index(MFMA_LK // 2)
            lo = Vec.load(v4elem_type, lds_do, [lds_row * LDS_DO_STRIDE + lds_col_lo])
            hi = Vec.load(v4elem_type, lds_do, [lds_row * LDS_DO_STRIDE + lds_col_hi])
            return Vec(lo).shuffle(Vec(hi), list(range(MFMA_LK))).ir_value()

        def _lds_load_v_b(kv_sub, k_step):
            lds_row = fx.Index(kv_sub * 32) + lane_mod_32
            lds_col_lo = fx.Index(k_step * MFMA_K) + lane_div_32 * MFMA_LK
            lds_col_hi = lds_col_lo + fx.Index(MFMA_LK // 2)
            lo = Vec.load(v4elem_type, lds_v, [lds_row * LDS_K_STRIDE + lds_col_lo])
            hi = Vec.load(v4elem_type, lds_v, [lds_row * LDS_K_STRIDE + lds_col_hi])
            return Vec(lo).shuffle(Vec(hi), list(range(MFMA_LK))).ir_value()

        # S accumulators
        s_accs = {}
        for qs in range_constexpr(Q_SUBS):
            for kvs in range_constexpr(KV_SUBS):
                s_accs[(qs, kvs)] = Vec.filled(16, 0.0, fx.Float32)

        for ks in range_constexpr(K_STEPS):
            q_packs = [_lds_load_q_a(qs, ks) for qs in range(Q_SUBS)]
            k_packs = [_lds_load_k_b(kvs, ks) for kvs in range(KV_SUBS)]
            for qs in range_constexpr(Q_SUBS):
                for kvs in range_constexpr(KV_SUBS):
                    s_accs[(qs, kvs)] = mfma(
                        q_packs[qs], k_packs[kvs], s_accs[(qs, kvs)]
                    )

        # -- Recompute P from S and stored LSE --
        log2e_scale_cst = fx.Float32(log2e_scale)
        neg_log2e = fx.Float32(-_LOG2E)

        p_vals = {}
        for qs in range_constexpr(Q_SUBS):
            for kvs in range_constexpr(KV_SUBS):
                for r in range_constexpr(16):
                    s_val = Vec(s_accs[(qs, kvs)])[r]
                    m_within = lane_div_32 * fx.Index(4) + fx.Index((r // 4) * 8 + (r % 4))
                    m_abs = fx.Index(qs * 32) + m_within
                    n_within = lane_mod_32
                    n_abs = fx.Index(kvs * 32) + n_within

                    m_valid = m_abs < seq_Q_idx
                    n_valid = n_abs < seq_KV_idx
                    valid = m_valid & n_valid

                    if const_expr(causal_mode == CAUSAL_TOP_LEFT):
                        valid = valid & (n_abs <= m_abs)
                    elif const_expr(causal_mode == CAUSAL_BOTTOM_RIGHT):
                        valid = valid & ((n_abs + seq_Q_idx) <= (m_abs + seq_KV_idx))

                    # Load LSE for this m-row
                    lse_flat = _lse_idx(m_abs)
                    lse_val = buffer_ops.buffer_load(
                        lse_rsrc, fx.Index(lse_flat), vec_width=1, dtype=fx.Float32
                    )
                    neg_log2e_lse = fx.Float32(
                        arith.mulf(_raw(neg_log2e), _raw(fx.Float32(lse_val)), fastmath=fm)
                    )

                    p_vals[(qs, kvs, r)] = _softmax_p(
                        s_val, neg_log2e_lse, log2e_scale_cst, valid, fm
                    )

        # -- Compute dP = dO @ V^T, accumulated over D --
        # dP[m, n] = sum_d dO[m, d] * V[n, d]
        dp_accs = {}
        for qs in range_constexpr(Q_SUBS):
            for kvs in range_constexpr(KV_SUBS):
                dp_accs[(qs, kvs)] = Vec.filled(16, 0.0, fx.Float32)

        for ks in range_constexpr(K_STEPS):
            do_packs = [_lds_load_do_a(qs, ks) for qs in range(Q_SUBS)]
            v_packs = [_lds_load_v_b(kvs, ks) for kvs in range(KV_SUBS)]
            for qs in range_constexpr(Q_SUBS):
                for kvs in range_constexpr(KV_SUBS):
                    dp_accs[(qs, kvs)] = mfma(
                        do_packs[qs], v_packs[kvs], dp_accs[(qs, kvs)]
                    )

        # -- Compute dS = P * (dP - D_vec), with validity mask --
        ds_vals = {}
        for qs in range_constexpr(Q_SUBS):
            for kvs in range_constexpr(KV_SUBS):
                for r in range_constexpr(16):
                    m_within = lane_div_32 * fx.Index(4) + fx.Index((r // 4) * 8 + (r % 4))
                    m_abs = fx.Index(qs * 32) + m_within
                    n_within = lane_mod_32
                    n_abs = fx.Index(kvs * 32) + n_within

                    m_valid = m_abs < seq_Q_idx
                    n_valid = n_abs < seq_KV_idx
                    valid = m_valid & n_valid

                    if const_expr(causal_mode == CAUSAL_TOP_LEFT):
                        valid = valid & (n_abs <= m_abs)
                    elif const_expr(causal_mode == CAUSAL_BOTTOM_RIGHT):
                        valid = valid & ((n_abs + seq_Q_idx) <= (m_abs + seq_KV_idx))

                    # Load D_vec for this m-row
                    dvec_flat = _dvec_idx(m_abs)
                    dm_val = fx.Float32(buffer_ops.buffer_load(
                        dvec_rsrc, fx.Index(dvec_flat), vec_width=1, dtype=fx.Float32
                    ))

                    dp_val = Vec(dp_accs[(qs, kvs)])[r]
                    p_val = p_vals[(qs, kvs, r)]
                    ds_vals[(qs, kvs, r)] = _grad_ds_unscaled(
                        p_val, dp_val, dm_val, valid, fm
                    )

        # -- Store dS to LDS for use as MFMA A-operand in dV, dK, dQ --
        # Also store P to LDS for dV computation.
        # We'll reuse lds_do for dS (since dO is consumed after dP computation)
        # and lds_q for P^T (since Q is consumed after S computation).
        # Actually Q and dO are still needed for dK and dQ. Let's use a different
        # approach: store dS and P in transposed form in scratch LDS areas.

        # For dV = P^T @ dO and dK = dS^T @ Q, we need P^T and dS^T as A-operands
        # and dO/Q as B-operands. P and dS are in registers.
        # Store them to LDS in transposed [N, MPAD] layout for A-operand reads.

        # We'll use LDS scratch after the Q/K/V/dO regions.
        # But we're tight on LDS. Let's do the computations sequentially:
        # Phase 1: dV computation, Phase 2: dK computation, Phase 3: dQ computation

        # For each phase, write the needed matrix to LDS, barrier, compute, barrier.

        # Phase 1: dV = P^T @ dO
        # Write P transposed [N, MPAD] to scratch LDS area (reuse some region)
        # We can't reuse Q/K/V/dO yet since we need them for dK/dQ.
        # However, for the small-attn case the LDS budget is generous.
        # Let's do a simpler approach: scalar register accumulation.

        # Since Lq and Lkv are very small (16, 25), we can afford to:
        # 1. Store all gradient intermediates in scratch LDS
        # 2. Have each wave read what it needs

        # Actually, the cleanest approach for small tiles:
        # Write dS and P to LDS in row-major [M, N] layout,
        # then read them transposed for dV/dK.

        # Use a scratch area starting after the main LDS allocations.
        # LDS budget: we used ~84KB for Q+K+V+dO. With 160KB budget, 76KB left.
        # P[M,N] = 32*32*2 = 2KB, dS[M,N] = 32*32*2 = 2KB. Plenty.

        # But we can't easily add more SmemAllocator regions dynamically.
        # Instead, let's store P and dS into registers grouped by output needs,
        # and do element-wise accumulation in registers for the small case.

        # For D=512, D_SUBS=16, distributed across 4 waves (4 each).
        # Each wave handles 4 D-subtiles of 32 columns for dV, dK, dQ.

        # dV[n, d] = sum_m P[m, n] * dO[m, d]  -- for each n, d
        # dK[n, d] = scale * sum_m dS[m, n] * Q[m, d]  -- for each n, d
        # dQ[m, d] = scale * sum_n dS[m, n] * K[n, d]  -- for each m, d

        # For MFMA-based computation, we need matrix operands.
        # dV: A = P^T[n, m_chunk], B = dO[m_chunk, d_tile] -> C[n, d_tile]
        # dK: A = dS^T[n, m_chunk], B = Q[m_chunk, d_tile] -> C[n, d_tile]
        # dQ: A = dS[m, n_chunk], B = K[n_chunk, d_tile] -> C[m, d_tile]

        # We need P^T and dS^T in LDS for A-operand reads.
        # Let's write them after a barrier.

        # Actually, for the backward we can proceed more carefully.
        # Since P and dS are small (MAX_LQ * MAX_LKV = 32*32 = 1024 elements),
        # we can convert them to bf16 and store to LDS.

        # Use a dedicated scratch region. We have headroom.
        # P and dS each need MAX_LQ * MAX_LKV * 2 bytes = 2KB each.
        # Store in transposed [N, MPAD] layout for A-operand reads in dV/dK.

        # For simplicity with the single-tile case (Q_SUBS=1, KV_SUBS=1):
        # P^T[n, m]: lane%32=n, register r maps to m -> already in correct layout
        # for a transposed store.

        # Write P^T to scratch in lds_q (we can reload Q from HBM if needed,
        # but actually Q is still in lds_q -- can't overwrite).
        # We DO need Q for dK. And dO for dV.

        # The safest approach: do dQ first (needs dS in regs + K in LDS),
        # then dV (needs P in regs + dO in LDS), then dK (needs dS in regs + Q in LDS).
        # All three need to write their transposed matrix to LDS to use as A-operand.
        # But we have LDS_V space that won't be needed after dP is computed.
        # After computing dP = dO @ V^T, V is consumed. Reuse lds_v for scratch.

        # Phase 1: dQ = scale * dS @ K (A=dS[m,n_chunk], B=K[n_chunk,d_tile])
        # dS is in registers. Write dS[m,n] to LDS scratch (lds_v) in row-major.
        # Then read as A-operand.

        # Write dS to lds_v in row-major [M, KV_PAD] layout
        DS_STRIDE = MAX_LKV + 8
        for qs in range_constexpr(Q_SUBS):
            for kvs in range_constexpr(KV_SUBS):
                for r in range_constexpr(16):
                    ds_val = ds_vals[(qs, kvs, r)]
                    m_within = lane_div_32 * fx.Index(4) + fx.Index((r // 4) * 8 + (r % 4))
                    m_abs = fx.Index(qs * 32) + m_within
                    n_within = lane_mod_32
                    n_abs = fx.Index(kvs * 32) + n_within
                    lds_idx = m_abs * DS_STRIDE + n_abs
                    Vec.from_elements([ds_val], fx.Float32).to(elem_dtype).store(
                        lds_v, [lds_idx]
                    )

        gpu.barrier()

        # dQ[m, d] = scale * sum_n dS[m, n] * K[n, d]
        # MFMA: A = dS[m, n_chunk] from lds_v, B = K[n, d_tile] from lds_k
        scale_cst = fx.Float32(scale)

        def _lds_load_ds_a(qs, kv_k_step):
            m_lane = fx.Index(qs * 32) + lane_mod_32
            k_base = fx.Index(kv_k_step * MFMA_K) + lane_div_32 * MFMA_LK
            k_lo = k_base
            k_hi = k_base + fx.Index(MFMA_LK // 2)
            lo = Vec.load(v4elem_type, lds_v, [m_lane * DS_STRIDE + k_lo])
            hi = Vec.load(v4elem_type, lds_v, [m_lane * DS_STRIDE + k_hi])
            return Vec(lo).shuffle(Vec(hi), list(range(MFMA_LK))).ir_value()

        def _lds_load_k_b_d(kvs, d_sub, kv_k_chunk):
            # B-operand for dQ: K[kv_row, d_col]
            # MFMA B: free=j%32 -> n (output d_col), k=j//32*LK+e -> kv row
            d_col = lane_mod_32 + fx.Index(d_sub * 32)
            kv_base = fx.Index(kvs * 32 + kv_k_chunk * MFMA_K) + lane_div_32 * MFMA_LK
            v1_type = Vec.make_type(1, elem_dtype)
            elems = []
            for e in range_constexpr(MFMA_LK):
                kv_r = kv_base + fx.Index(e)
                lds_idx = kv_r * LDS_K_STRIDE + d_col
                v1 = Vec.load(v1_type, lds_k, [lds_idx])
                elems.append(Vec(v1)[0])
            return Vec.from_elements(elems, elem_dtype).ir_value()

        # Compute dQ, distributed across waves by D-subtile
        for d_sub_local in range_constexpr(D_SUBS_PER_WAVE):
            d_sub = wave * D_SUBS_PER_WAVE + d_sub_local
            d_in_range = d_sub < fx.Index(D_SUBS)
            d_sub_safe = d_in_range.select(d_sub, fx.Index(0))

            for qs in range_constexpr(Q_SUBS):
                dq_acc = Vec.filled(16, 0.0, fx.Float32)
                for kv_k in range_constexpr(KV_K_STEPS):
                    a_pack = _lds_load_ds_a(qs, kv_k)
                    b_pack = _lds_load_k_b_d(0, d_sub_safe, kv_k)
                    dq_acc = mfma(a_pack, b_pack, dq_acc)

                # Store dQ (f32) to output
                for r in range_constexpr(16):
                    m_within = lane_div_32 * fx.Index(4) + fx.Index((r // 4) * 8 + (r % 4))
                    m_abs = fx.Index(qs * 32) + m_within
                    d_col = lane_mod_32 + d_sub_safe * fx.Index(32)
                    m_ok = (m_abs < seq_Q_idx) & d_in_range & (d_col < fx.Index(D))

                    if m_ok:
                        dq_val = fx.Float32(arith.mulf(
                            _raw(Vec(dq_acc)[r]), _raw(scale_cst), fastmath=fm
                        ))
                        q_row_out = _q_row(m_abs)
                        flat_dq = fx.Index(q_row_out) * fx.Index(D) + d_col
                        buffer_ops.buffer_store(
                            _raw(dq_val), dq_rsrc, flat_dq,
                        )

        # Phase 2: dV and dK
        # Need P^T and dS^T as A-operands.
        # Write P transposed [N, MPAD] and dS transposed [N, MPAD] to lds_v
        # (overwriting the row-major dS we just consumed).
        gpu.barrier()

        # Write P^T to first half of lds_v, dS^T to second half
        PT_OFF = 0
        DST_OFF = MAX_LKV * LDS_MPAD

        for qs in range_constexpr(Q_SUBS):
            for kvs in range_constexpr(KV_SUBS):
                for r in range_constexpr(16):
                    m_within = lane_div_32 * fx.Index(4) + fx.Index((r // 4) * 8 + (r % 4))
                    m_abs = fx.Index(qs * 32) + m_within
                    n_within = lane_mod_32
                    n_abs = fx.Index(kvs * 32) + n_within

                    # P^T[n, m] stored at PT_OFF + n_abs * MPAD + m_abs
                    # P^T[n, m] stored at PT_OFF + n_abs * MPAD + m_abs
                    p_f32 = p_vals[(qs, kvs, r)]
                    lds_idx_p = fx.Index(PT_OFF) + n_abs * LDS_MPAD + m_abs
                    Vec.from_elements([p_f32], fx.Float32).to(elem_dtype).store(
                        lds_v, [lds_idx_p]
                    )

                    # dS^T[n, m] stored at DST_OFF + n_abs * MPAD + m_abs
                    ds_f32 = ds_vals[(qs, kvs, r)]
                    lds_idx_ds = fx.Index(DST_OFF) + n_abs * LDS_MPAD + m_abs
                    Vec.from_elements([ds_f32], fx.Float32).to(elem_dtype).store(
                        lds_v, [lds_idx_ds]
                    )

        gpu.barrier()

        # dV[n, d] = sum_m P^T[n, m_chunk] * dO[m_chunk, d_tile]
        # dK[n, d] = scale * sum_m dS^T[n, m_chunk] * Q[m_chunk, d_tile]
        Q_K_STEPS = MAX_LQ // MFMA_K

        def _lds_load_pt_a(kvs, q_k_step):
            n_lane = fx.Index(kvs * 32) + lane_mod_32
            k_base = fx.Index(q_k_step * MFMA_K) + lane_div_32 * MFMA_LK
            k_lo = k_base
            k_hi = k_base + fx.Index(MFMA_LK // 2)
            base = fx.Index(PT_OFF)
            lo = Vec.load(v4elem_type, lds_v, [base + n_lane * LDS_MPAD + k_lo])
            hi = Vec.load(v4elem_type, lds_v, [base + n_lane * LDS_MPAD + k_hi])
            return Vec(lo).shuffle(Vec(hi), list(range(MFMA_LK))).ir_value()

        def _lds_load_dst_a(kvs, q_k_step):
            n_lane = fx.Index(kvs * 32) + lane_mod_32
            k_base = fx.Index(q_k_step * MFMA_K) + lane_div_32 * MFMA_LK
            k_lo = k_base
            k_hi = k_base + fx.Index(MFMA_LK // 2)
            base = fx.Index(DST_OFF)
            lo = Vec.load(v4elem_type, lds_v, [base + n_lane * LDS_MPAD + k_lo])
            hi = Vec.load(v4elem_type, lds_v, [base + n_lane * LDS_MPAD + k_hi])
            return Vec(lo).shuffle(Vec(hi), list(range(MFMA_LK))).ir_value()

        def _lds_load_do_b(qs, d_sub, q_k_chunk):
            # B-operand for dV: dO[q_row, d_col]
            # MFMA B: free=j%32 -> d_col, k=j//32*LK+e -> q_row
            d_col = lane_mod_32 + fx.Index(d_sub * 32)
            q_base = fx.Index(qs * 32 + q_k_chunk * MFMA_K) + lane_div_32 * MFMA_LK
            v1_type = Vec.make_type(1, elem_dtype)
            elems = []
            for e in range_constexpr(MFMA_LK):
                q_r = q_base + fx.Index(e)
                lds_idx = q_r * LDS_DO_STRIDE + d_col
                v1 = Vec.load(v1_type, lds_do, [lds_idx])
                elems.append(Vec(v1)[0])
            return Vec.from_elements(elems, elem_dtype).ir_value()

        def _lds_load_q_b(qs, d_sub, q_k_chunk):
            # B-operand for dK: Q[q_row, d_col]
            # MFMA B: free=j%32 -> d_col, k=j//32*LK+e -> q_row
            d_col = lane_mod_32 + fx.Index(d_sub * 32)
            q_base = fx.Index(qs * 32 + q_k_chunk * MFMA_K) + lane_div_32 * MFMA_LK
            v1_type = Vec.make_type(1, elem_dtype)
            elems = []
            for e in range_constexpr(MFMA_LK):
                q_r = q_base + fx.Index(e)
                lds_idx = q_r * LDS_Q_STRIDE + d_col
                v1 = Vec.load(v1_type, lds_q, [lds_idx])
                elems.append(Vec(v1)[0])
            return Vec.from_elements(elems, elem_dtype).ir_value()

        # Compute dV and dK, distributed across waves by D-subtile
        for d_sub_local in range_constexpr(D_SUBS_PER_WAVE):
            d_sub = wave * D_SUBS_PER_WAVE + d_sub_local
            d_in_range = d_sub < fx.Index(D_SUBS)
            d_sub_safe = d_in_range.select(d_sub, fx.Index(0))

            for kvs in range_constexpr(KV_SUBS):
                dv_acc = Vec.filled(16, 0.0, fx.Float32)
                dk_acc = Vec.filled(16, 0.0, fx.Float32)

                for q_k in range_constexpr(Q_K_STEPS):
                    pt_pack = _lds_load_pt_a(kvs, q_k)
                    dst_pack = _lds_load_dst_a(kvs, q_k)
                    do_pack = _lds_load_do_b(0, d_sub_safe, q_k)
                    q_pack = _lds_load_q_b(0, d_sub_safe, q_k)
                    dv_acc = mfma(pt_pack, do_pack, dv_acc)
                    dk_acc = mfma(dst_pack, q_pack, dk_acc)

                # Store dV (f32) and dK (f32 * scale) to output
                for r in range_constexpr(16):
                    n_within = lane_div_32 * fx.Index(4) + fx.Index((r // 4) * 8 + (r % 4))
                    n_abs = fx.Index(kvs * 32) + n_within
                    d_col = lane_mod_32 + d_sub_safe * fx.Index(32)
                    n_ok = (n_abs < seq_KV_idx) & d_in_range & (d_col < fx.Index(D))

                    if n_ok:
                        # dV: no scale factor
                        dv_val = Vec(dv_acc)[r]
                        kv_row_out = _kv_row(n_abs)
                        flat = fx.Index(kv_row_out) * fx.Index(D) + d_col
                        buffer_ops.buffer_store(
                            _raw(dv_val), dv_rsrc, flat,
                        )

                        # dK: multiply by scale
                        dk_val = fx.Float32(arith.mulf(
                            _raw(Vec(dk_acc)[r]), _raw(scale_cst), fastmath=fm
                        ))
                        buffer_ops.buffer_store(
                            _raw(dk_val), dk_rsrc, flat,
                        )

    @flyc.jit
    def launch_fn(
        Q: fx.Tensor,
        K: fx.Tensor,
        V: fx.Tensor,
        dO: fx.Tensor,
        dV: fx.Tensor,
        dK: fx.Tensor,
        dQ: fx.Tensor,
        LSE: fx.Tensor,
        D_vec: fx.Tensor,
        B: fx.Int32,
        Lq: fx.Int32,
        Lkv: fx.Int32,
        H: fx.Int32,
        stream: fx.Stream,
    ):
        from flydsl._mlir import ir
        from flydsl.compiler.kernel_function import CompilationContext

        allocator.finalized = False
        _ctx = CompilationContext.get_current()
        with ir.InsertionPoint(_ctx.gpu_module_body):
            allocator.finalize()

        grid_x = fx.Int32(fx.Index(B) * fx.Index(H))
        fmha_small_attn_bwd_kernel(
            Q, K, V, dO, dV, dK, dQ, LSE, D_vec,
            Lq, Lkv, H,
        ).launch(
            grid=(grid_x, 1, 1),
            block=(BLOCK_SIZE, 1, 1),
            stream=stream,
        )

    return launch_fn


# ---------------------------------------------------------------------------
# torch.library.custom_op: lean forward dispatch for the small-attention kernel.
# Caches the compiled kernel at module level so repeated calls skip the Python
# compile/cache-lookup path entirely.
# ---------------------------------------------------------------------------

import torch  # noqa: E402

_small_attn_fwd_compiled: dict = {}


@torch.library.custom_op("mslk::small_attn_fwd", mutates_args=("O_f32", "LSE"))
def small_attn_fwd(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    O_f32: torch.Tensor,
    LSE: torch.Tensor,
    B: int,
    Sq: int,
    Skv: int,
    H: int,
    D: int,
    scale: float,
    causal_mode: int,
) -> None:
    import flydsl.compiler as flyc

    MAX_LQ = _pad32(Sq)
    MAX_LKV = _pad32(Skv)
    dtype = Q.dtype
    dtype_str = "bf16" if dtype == torch.bfloat16 else "f16"
    stream = torch.cuda.current_stream(Q.device)

    fwd_key = (D, dtype_str, scale, causal_mode, MAX_LQ, MAX_LKV)
    compiled = _small_attn_fwd_compiled.get(fwd_key)
    if compiled is None:
        launch = compile_fmha_small_attn_fwd(
            D=D, dtype_str=dtype_str, scale=scale,
            causal_mode=causal_mode, MAX_LQ=MAX_LQ, MAX_LKV=MAX_LKV,
        )
        Q_i16 = Q.view(torch.int16)
        tmp_o = torch.empty_like(O_f32)
        tmp_lse = torch.empty_like(LSE)
        compiled = flyc.compile(launch, Q_i16, K.view(torch.int16), V.view(torch.int16),
                                tmp_o, tmp_lse, B, Sq, Skv, H, stream)
        _small_attn_fwd_compiled[fwd_key] = compiled

    compiled(Q.view(torch.int16), K.view(torch.int16), V.view(torch.int16),
             O_f32, LSE, B, Sq, Skv, H, stream)


@torch.library.register_fake("mslk::small_attn_fwd")
def _small_attn_fwd_fake(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    O_f32: torch.Tensor,
    LSE: torch.Tensor,
    B: int,
    Sq: int,
    Skv: int,
    H: int,
    D: int,
    scale: float,
    causal_mode: int,
) -> None:
    return None
