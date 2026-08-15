# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""FMHA backward kernels using 32x32 MFMA tiling (gfx942/CDNA3 fallback).

Provides dV/dK, dQ, and fused dQ+dV+dK backward passes for flash attention
using mfma_f32_32x32x{8,16}_{bf16,f16}. On gfx942 (CDNA3) the K8 MFMA
variant is selected automatically; on gfx950 (CDNA4) the native K16 path
and optional ds_read_tr16_b64 hardware-transpose loads are available.

MFMA 32x32x16 register layout (wave64):
  Lane j (0..63): j_mod = j%32, j_div = j//32
  INPUT operand[free, k]:  free = j_mod,  k = j_div*8 + e  (e = 0..7)
  OUTPUT C[m, n] for reg r (0..15):
    m = j_div*4 + (r//4)*8 + (r%4)   (A-operand free dim; varies with r)
    n = j_mod                        (B-operand free dim; fixed across r)
  The output row index m uses a scrambled mapping that differs from the
  input free-dim j_mod -- the GEMM1->GEMM2 bridge must reconcile this.
"""

import math as _math

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import (
    arith,
    buffer_ops,
    const_expr,
    gpu,
    math as fly_math,
    range_constexpr,
    rocdl,
)
from flydsl.expr.typing import Vector as Vec
from flydsl.expr.utils.arith import _to_raw as _raw, ArithValue
from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr


# Inlined (not imported from kernels.kernels_common / kernels.common.kernels_common):
# that module's path has moved across FlyDSL checkouts on different nodes, so this
# kernel is kept self-contained rather than depending on FlyDSL's internal layout.
def dtype_to_elem_type(dtype_str: str):
    """Map a dtype string to its FlyDSL numeric type ('f32', 'f16', 'bf16', 'fp8')."""
    if dtype_str == "f32":
        return fx.Float32
    if dtype_str == "f16":
        return fx.Float16
    if dtype_str == "bf16":
        return fx.BFloat16
    if dtype_str == "fp8":
        return fx.Float8E4M3FN
    raise ValueError(
        f"unsupported dtype: {dtype_str!r} (expected 'f32', 'f16', 'bf16', or 'fp8')"
    )


def get_warp_size() -> int:
    """Wavefront size for the CDNA (gfx9xx) targets this kernel supports (gfx942/gfx950)."""
    return 64


WARP_SIZE = get_warp_size()  # 64 on CDNA3/CDNA4
_LOG2E = _math.log2(_math.e)


def _softmax_p(s_val, neg_log2e_lse, log2e_scale, valid, fm):
    """P = exp2(log2e*scale*S - log2e*LSE), masked to 0 outside valid.

    log2e_scale and neg_log2e_lse are host/row-hoisted constants (log2e*scale,
    -log2e*lse) so the per-element cost collapses to one FMA + one exp2.
    """
    p_arg = fx.Float32(
        fly_math.fma(_raw(log2e_scale), _raw(s_val), _raw(neg_log2e_lse), fastmath=fm)
    )
    p_val = fx.Float32(fly_math.exp2(p_arg, fastmath=fm))
    return valid.select(p_val, fx.Float32(0.0))


def _grad_ds_unscaled(p_val, dp_val, dm_val, valid, fm):
    """dS' = P*(dP-D), masked to 0 outside valid; the `scale` factor is applied
    once to the finished dK/dQ accumulator instead of per-element here, since
    MFMA is linear in its A-operand: scale*(dS'@X) == (scale*dS')@X.
    """
    dp_sub = fx.Float32(arith.subf(_raw(dp_val), _raw(dm_val), fastmath=fm))
    ds_val = fx.Float32(arith.mulf(_raw(p_val), _raw(dp_sub), fastmath=fm))
    return valid.select(ds_val, fx.Float32(0.0))


def _ds_read_tr_v4(v4_type, lds_elem_idx, lds_byte_base):
    """gfx950 hardware-transpose LDS read (ds_read_b64_tr_b16), rocdl wrapper.

    Reads a 4x16 bf16 tile cooperatively across a 16-lane group and returns the
    transposed 16x4 (4 bf16 per lane). Pair two calls + shuffle for a v8 operand.
    """
    byte_i64 = fx.Int64(lds_elem_idx * 2 + lds_byte_base)
    ptr = buffer_ops.create_llvm_ptr(byte_i64, address_space=3)
    return rocdl.ds_read_tr16_b64(v4_type, ptr).result


def compile_fmha_bwd_dvdk_mfma(
    *,
    D: int = 64,
    dtype_str: str = "bf16",
    BLOCK_M: int = 64,
    BLOCK_N: int = 64,
    scale: float = None,
    use_trload: bool = False,
    use_pipeline: bool = False,
    gpu_arch: str = "gfx950",
    causal: bool = False,
    heads_per_kv: int = 1,
    varlen: bool = False,
):
    """Fused dV + dK backward kernel with MFMA.

    use_trload=True: GEMM2 B-operand (dO/Q) is read via the hardware LDS-transpose
    ds_read_b64_tr_b16 instead of the 8x scalar gather. Requires EVEN LDS_Q_STRIDE
    (odd stride breaks the tr 64-bit column alignment). The A-operand (P^T/dS^T) is
    re-ordered to the transpose's P8 k-permutation so contraction stays aligned.

    Both dV and dK grid over N-tiles and share S, dP, P, dS. Computing them in
    one kernel eliminates a redundant S/dP GEMM pass vs running dv+dk separately.
      S  = Q @ K^T,  dP = dO @ V^T          (GEMM1a/b, shared)
      P  = softmax(S);  dS = scale*P*(dP-Dm) (both stored to LDS)
      dV = P^T  @ dO                         (GEMM2a)
      dK = dS^T @ Q                          (GEMM2b)
    Neither output needs atomics (both N-tile-unique). dQ stays a separate kernel
    (grid over M-tiles; fusing it here would require atomic-add for correctness).

    GQA (heads_per_kv > 1): the grid is regrouped by KV-head
    (`B*Hkv*num_N_tiles` instead of `B*Hq*num_N_tiles`) and each block
    internally loops over its `heads_per_kv` Q-heads, accumulating all their
    P/dS contributions into ONE register accumulator before a single plain
    store. heads_per_kv==1 (default, non-GQA) degenerates to exactly the
    original per-head-unique behavior (loop trivially runs once).

    varlen: B collapses to 1 (all batches' Q/K/V physically concatenated along
    the M/N axis with no padding); grid_x is sized off `max_seqlen_k` (a host
    constant) instead of per-tensor `N`, so trip counts stay host-computed.
    Two new runtime `fx.Tensor` params, `seqstart_q`/`seqstart_k` (int32,
    shape `[B_logical+1]`, cumulative offsets), are buffer-loaded once per
    block to derive per-batch sequence lengths and row-offset bases.

    Returns:
        launch_fn(Q, K, V, dO, dV, dK, LSE, D_vec, B, M, N, H, n_M_tiles,
                  q_stride_m, kv_stride_n, stream)
          Q, K, V, dO : [B*seq*H, D] (row pitch may exceed H*D -- see
                        q_stride_m/kv_stride_n)
          dV, dK      : [B*N*Hkv*D, 1] float32 (always contiguous output;
                        Hkv = H // heads_per_kv)
          LSE, D_vec  : [B*H*M,1] / [B*M*H,1] float32
          q_stride_m  : Q/dO row pitch in ROW units (real_elem_stride(dim=1) // D);
                        H for contiguous BMHK.
          kv_stride_n : K/V row pitch in ROW units, same convention.
          H           : total Q head count (Hq); Hkv is derived as H //
                        heads_per_kv (compile-time), not passed separately.
          seqstart_q/seqstart_k : (varlen=True only) int32 [B_logical+1] cumulative
                        offset tensors; M/N become max_seqlen_q/max_seqlen_k.
    """
    import math as _pm

    if scale is None:
        scale = 1.0 / _pm.sqrt(D)
    assert BLOCK_N == 64, (
        f"dvdk requires BLOCK_N == 64 (wave-tiling fixed), got {BLOCK_N}"
    )
    assert D % 32 == 0, (
        f"dvdk requires D a multiple of 32 (wave D-subtile width), got D={D}"
    )
    assert D % 16 == 0

    # gfx950 (CDNA4) has native K16 MFMA; gfx942 (CDNA3) only has native K8 -- same
    # K_STEPS/MFMA_LK-parameterized dispatch as FlyDSL's own flash_attn_generic.py
    # forward kernel (mfma_acc()), not a "call K8 twice into one K16 slot" wrapper.
    USE_K16 = gpu_arch.startswith("gfx950")
    # ds_read_tr16_b64 (used by use_trload) is a gfx950(CDNA4)-only HW-transpose LDS
    # read (lib/Dialect/FlyROCDL/CDNA4/CopyAtom.cpp) -- unrelated to the MFMA K-width
    # gap but also unavailable on gfx942.
    assert not (use_trload and not USE_K16), (
        "use_trload requires gfx950 (ds_read_tr16_b64 is CDNA4-only)"
    )

    elem_dtype = dtype_to_elem_type(dtype_str)
    MFMA_K = 16 if USE_K16 else 8
    MFMA_LK = 8 if USE_K16 else 4
    K_STEPS = D // MFMA_K
    fm = arith.FastMathFlags.fast

    BLOCK_SIZE = 256
    NUM_WAVES = BLOCK_SIZE // WARP_SIZE  # 4
    WAVE_N_TILES = BLOCK_N // 32  # 2 (BLOCK_N=64 fixed)
    WAVES_PER_N_GROUP = NUM_WAVES // WAVE_N_TILES  # 2
    D_TOTAL_SUBS = D // 32  # 1,2,3,4,8 for D=32,64,96,128,256
    # D_SUBS_PER_WAVE = ceil(D_TOTAL_SUBS / WAVES_PER_N_GROUP): D=64/128/256 divide evenly
    # (1/2/4, unchanged from before). D=32/96 don't divide evenly across the 2 waves in a
    # D-group -- the last wave's nominal subtile range can run past D_TOTAL_SUBS (D=96: wave
    # group 0 covers real subtiles {0,1}, group 1 covers {2, <3-doesn't-exist>}; D=32: group 0
    # covers real subtile {0}, group 1's nominal {1} doesn't exist at all). Rather than a new
    # warp-partition per head-dim, this kernel keeps the existing wave-tiling
    # and lets excess waves compute a redundant/garbage out-of-range subtile that is simply
    # never stored (guarded by `wave_d_sub_i < D_TOTAL_SUBS` at the store site below) --
    # correct, not maximally efficient, acceptable since D=32/96 aren't the perf-critical shapes.
    D_SUBS_PER_WAVE = -(-D_TOTAL_SUBS // WAVES_PER_N_GROUP)
    # wave sequentially covers D_SUBS_PER_WAVE contiguous 32-col D-subtiles.

    # LDS layout:
    # Q/dO: [M, LDS_Q_STRIDE] row-major, padded stride for bank-conflict-free scatter.
    # P/dS: TRANSPOSED [N, LDS_MPAD] with padded stride for vectorized GEMM2 A-reads.
    # LSE, D_vec: [BLOCK_M] f32 scalars.
    # Bank analysis for Q/dO scatter (GEMM2): (m*LDS_Q_STRIDE+d)/2%32.
    # S=D+2=66: 16 consecutive m-rows map to 16 distinct banks — zero conflicts.
    # S=66 is 4-byte aligned (m*132%4=0), enabling ds_read_b64 (v4 f16) for GEMM1.
    LDS_MPAD = BLOCK_M + 8  # P/dS transposed stride padding
    # Q/dO row stride: baseline uses D+2 (odd for D=64) for bank-conflict-free scalar scatter;
    # trload needs EVEN stride (D+8) so ds_read_b64_tr keeps 64-bit column alignment.
    LDS_Q_STRIDE = (D + 8) if use_trload else (D + 2)
    LDS_Q_ELEMS = BLOCK_M * LDS_Q_STRIDE
    LDS_DO_ELEMS = BLOCK_M * LDS_Q_STRIDE
    LDS_DS_ELEMS = BLOCK_N * LDS_MPAD
    LDS_P_ELEMS = BLOCK_N * LDS_MPAD
    LDS_LSE_ELEMS = BLOCK_M
    LDS_DM_ELEMS = BLOCK_M

    allocator = SmemAllocator(
        None, arch=gpu_arch, global_sym_name="fmha_bwd_dvdk_mfma_smem"
    )
    lds_q_off = allocator._align(allocator.ptr, 16)
    allocator.ptr = lds_q_off + LDS_Q_ELEMS * 2
    lds_do_off = allocator._align(allocator.ptr, 16)
    allocator.ptr = lds_do_off + LDS_DO_ELEMS * 2
    lds_ds_off = allocator._align(allocator.ptr, 16)
    allocator.ptr = lds_ds_off + LDS_DS_ELEMS * 2
    lds_p_off = allocator._align(allocator.ptr, 16)
    allocator.ptr = lds_p_off + LDS_P_ELEMS * 2
    lds_lse_off = allocator._align(allocator.ptr, 16)
    allocator.ptr = lds_lse_off + LDS_LSE_ELEMS * 4
    lds_dm_off = allocator._align(allocator.ptr, 16)
    allocator.ptr = lds_dm_off + LDS_DM_ELEMS * 4

    @flyc.kernel(known_block_size=[BLOCK_SIZE, 1, 1])
    def fmha_bwd_dvdk_mfma_kernel(  # noqa: F811
        Q: fx.Tensor,
        K: fx.Tensor,
        V: fx.Tensor,
        dO: fx.Tensor,
        dV: fx.Tensor,
        dK: fx.Tensor,
        LSE: fx.Tensor,
        D_vec: fx.Tensor,
        seq_M: fx.Int32,
        seq_N: fx.Int32,
        n_heads: fx.Int32,
        n_M_tiles: fx.Int32,
        q_stride_m: fx.Int32,
        kv_stride_n: fx.Int32,
        do_stride_m: fx.Int32,
        seqstart_q: fx.Tensor,
        seqstart_k: fx.Tensor,
        total_m: fx.Int32,
    ):
        bid = fx.block_idx.x
        tid = fx.thread_idx.x

        n_heads_idx = fx.Index(n_heads)
        seq_M_idx = fx.Index(seq_M)
        seq_N_idx = fx.Index(seq_N)
        n_M_tiles_idx = fx.Index(n_M_tiles)
        q_stride_m_idx = fx.Index(q_stride_m)
        kv_stride_n_idx = fx.Index(kv_stride_n)
        do_stride_m_idx = fx.Index(do_stride_m)
        # varlen: LSE/D_vec are packed over the TOTAL (sum-over-batches)
        # M length, NOT seq_M_idx (which is max_seqlen_q here, used only for
        # grid/loop sizing) -- see the module docstring. Non-varlen: total_m == seq_M.
        total_m_idx = fx.Index(total_m)
        num_N_tiles = (seq_N_idx + BLOCK_N - 1) // BLOCK_N
        # GQA: grid is regrouped by KV-head (see
        # module docstring) -- n_kv_heads_idx = Hq // heads_per_kv (compile-time
        # constant divisor). heads_per_kv==1 (non-GQA) makes kv_head_idx ==
        # head_idx for the single q_head_in_group iteration below.
        n_kv_heads_idx = n_heads_idx // heads_per_kv

        bid_idx = fx.Index(bid)
        n_tile = bid_idx % num_N_tiles
        bh_idx = bid_idx // num_N_tiles
        batch_idx = bh_idx // n_kv_heads_idx
        kv_head_idx = bh_idx % n_kv_heads_idx

        n_start = n_tile * BLOCK_N

        # varlen: q_start/k_start (packed-M/N row-offset bases) and
        # this_seqlen_q/this_seqlen_k (per-batch REAL length, for masking) come
        # from a runtime seqstart lookup instead of a globally-uniform
        # batch_idx*seq_len_idx/seq_len_idx -- see module docstring. Non-varlen:
        # these reduce to exactly the original formulas (B_logical==1-style).
        if const_expr(varlen):
            from flydsl.expr import buffer_ops as _seq_bops

            seqstart_q_rsrc = _seq_bops.create_buffer_resource(seqstart_q)
            seqstart_k_rsrc = _seq_bops.create_buffer_resource(seqstart_k)

            def _seqstart_load(rsrc, idx):
                return fx.Index(
                    _seq_bops.buffer_load(
                        rsrc, fx.Index(idx), vec_width=1, dtype=fx.Int32
                    )
                )

            q_start = _seqstart_load(seqstart_q_rsrc, batch_idx)
            k_start = _seqstart_load(seqstart_k_rsrc, batch_idx)
            this_seqlen_q = _seqstart_load(seqstart_q_rsrc, batch_idx + 1) - q_start
            this_seqlen_k = _seqstart_load(seqstart_k_rsrc, batch_idx + 1) - k_start
        else:
            q_start = batch_idx * seq_M_idx
            k_start = batch_idx * seq_N_idx
            this_seqlen_q = seq_M_idx
            this_seqlen_k = seq_N_idx

        wave = fx.Index(tid // WARP_SIZE)
        lane = fx.Index(tid % WARP_SIZE)
        lane_mod_32 = fx.Index(lane % 32)
        lane_div_32 = fx.Index(lane // 32)
        wave_n_sub = fx.Index(wave // WAVES_PER_N_GROUP)
        wave_d_group = fx.Index(wave % WAVES_PER_N_GROUP)
        wave_d_sub_base = wave_d_group * D_SUBS_PER_WAVE

        dV_buf = fx.rocdl.make_buffer_tensor(dV)
        dK_buf = fx.rocdl.make_buffer_tensor(dK)
        LSE_buf = fx.rocdl.make_buffer_tensor(LSE)
        Dvec_buf = fx.rocdl.make_buffer_tensor(D_vec)

        copy_f32 = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.Float32)
        store_f32 = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.Float32)

        v8elem_type = Vec.make_type(MFMA_LK, elem_dtype)
        v16f32_type = Vec.make_type(16, fx.Float32)

        base_ptr = allocator.get_base()
        lds_q = SmemPtr(
            base_ptr, lds_q_off, elem_dtype.ir_type, shape=(LDS_Q_ELEMS,)
        ).get()
        lds_do = SmemPtr(
            base_ptr, lds_do_off, elem_dtype.ir_type, shape=(LDS_DO_ELEMS,)
        ).get()
        lds_ds = SmemPtr(
            base_ptr, lds_ds_off, elem_dtype.ir_type, shape=(LDS_DS_ELEMS,)
        ).get()
        lds_p = SmemPtr(
            base_ptr, lds_p_off, elem_dtype.ir_type, shape=(LDS_P_ELEMS,)
        ).get()
        lds_lse = SmemPtr(
            base_ptr, lds_lse_off, fx.Float32.ir_type, shape=(LDS_LSE_ELEMS,)
        ).get()
        lds_dm = SmemPtr(
            base_ptr, lds_dm_off, fx.Float32.ir_type, shape=(LDS_DM_ELEMS,)
        ).get()

        # Q/dO and K/V are possibly-non-contiguous user inputs (e.g. a packed-qkv
        # unbind view) -- row pitch is q_stride_m_idx/kv_stride_n_idx (in row
        # units, i.e. real_elem_stride // D), NOT necessarily n_heads_idx. dO can
        # have a DIFFERENT row pitch than Q (e.g. Q comes from a qkv-unbind view
        # but dO is a fresh contiguous torch.randn_like(out)) -- separate helper.
        # GQA: this block's Q-side rows vary across the
        # `heads_per_kv` group (see the q_head_in_group loop below), so head_idx
        # is now an explicit param rather than a fixed block-wide closure value.
        # varlen: q_start/k_start replace batch_idx*seq_len_idx as the row-offset
        # base -- q_pos/kv_pos are still batch-LOCAL positions (0..this_seqlen-1),
        # added to the packed-global start before multiplying by the row stride.
        def _q_row(q_pos, head_idx):
            return fx.Int32((q_start + q_pos) * q_stride_m_idx + head_idx)

        def _do_row(q_pos, head_idx):
            return fx.Int32((q_start + q_pos) * do_stride_m_idx + head_idx)

        # K/V are indexed by kv_head_idx (fixed for the whole block -- GQA's
        # grid-regroup-by-KV-head, see module docstring), NOT a per-q-head value.
        def _kv_row(kv_pos):
            return fx.Int32((k_start + kv_pos) * kv_stride_n_idx + kv_head_idx)

        # dV/dK are freshly-allocated contiguous outputs, shaped [B,N,Hkv,D] --
        # row pitch n_kv_heads_idx, indexed by kv_head_idx (one write per block).
        # Packed the SAME way as K/V's own N axis (varlen: k_start-relative).
        def _kv_row_out(kv_pos):
            return fx.Int32((k_start + kv_pos) * n_kv_heads_idx + kv_head_idx)

        # LSE layout is [B,H,M] (batch-major, non-varlen) vs packed [1,H,sum_M]
        # under varlen (B collapses to 1). Not unifiable via a single
        # q_start-relative formula because the head-axis stride differs:
        # seq_M_idx (non-varlen) vs total_m_idx (varlen, the FULL packed extent).
        def _lse_row(q_pos, head_idx):
            if const_expr(varlen):
                return fx.Int32(head_idx * total_m_idx + q_start + q_pos)
            return fx.Int32((batch_idx * n_heads_idx + head_idx) * seq_M_idx + q_pos)

        # D_vec is a freshly-allocated contiguous tensor (BMHK row-major) --
        # row pitch is always n_heads_idx regardless of varlen/non-varlen.
        def _dvec_row(q_pos, head_idx):
            return fx.Int32((q_start + q_pos) * n_heads_idx + head_idx)

        from flydsl.expr import buffer_ops as _bops

        q_rsrc = _bops.create_buffer_resource(Q)
        k_rsrc = _bops.create_buffer_resource(K)
        v_rsrc = _bops.create_buffer_resource(V)
        do_rsrc = _bops.create_buffer_resource(dO)

        def _load_global_vec_cv(rsrc, row_i32, col_offset_idx):
            flat_elem = fx.Index(row_i32) * fx.Index(D) + col_offset_idx
            return _bops.buffer_load(
                rsrc, flat_elem, vec_width=MFMA_LK, dtype=elem_dtype
            )

        def _load_f32_row(buf, row_idx):
            row_sl = fx.slice(buf, (row_idx, None))
            div_1 = fx.logical_divide(row_sl, fx.make_layout(1, 1))
            r = fx.make_rmem_tensor(1, fx.Float32)
            fx.copy_atom_call(copy_f32, fx.slice(div_1, (None, 0)), r)
            return fx.memref_load(r, 0)

        def _store_f32_row(buf, row_idx, val):
            row_sl = fx.slice(buf, (row_idx, None))
            div_1 = fx.logical_divide(row_sl, fx.make_layout(1, 1))
            r = fx.make_rmem_tensor(1, fx.Float32)
            fx.memref_store(val, r, 0)
            fx.copy_atom_call(store_f32, r, fx.slice(div_1, (None, 0)))

        v4elem_type = Vec.make_type(MFMA_LK // 2, elem_dtype)

        def _lds_load_pack_a(lds_arr, base_row_in_tile, k_step):
            # Q/dO stored with padded stride LDS_Q_STRIDE=66. Use 2×v4 loads (ds_read_b64,
            # 4-byte aligned) rather than 1 v8 (ds_read_b128, needs 16-byte alignment).
            lds_row = fx.Index(base_row_in_tile) + lane_mod_32
            lds_col_lo = fx.Index(k_step * MFMA_K) + lane_div_32 * MFMA_LK
            lds_col_hi = lds_col_lo + fx.Index(MFMA_LK // 2)
            lo = Vec.load(v4elem_type, lds_arr, [lds_row * LDS_Q_STRIDE + lds_col_lo])
            hi = Vec.load(v4elem_type, lds_arr, [lds_row * LDS_Q_STRIDE + lds_col_hi])
            return Vec(lo).shuffle(Vec(hi), list(range(MFMA_LK))).ir_value()

        def mfma(a_pack, b_pack, c_acc):
            if const_expr(dtype_str == "bf16"):
                if const_expr(USE_K16):
                    return rocdl.mfma_f32_32x32x16_bf16(
                        v16f32_type, [a_pack, b_pack, c_acc]
                    )
                a_pack = Vec(a_pack).bitcast(fx.Int16)
                b_pack = Vec(b_pack).bitcast(fx.Int16)
                return rocdl.mfma_f32_32x32x8bf16_1k(
                    v16f32_type, [a_pack, b_pack, c_acc]
                )
            if const_expr(USE_K16):
                return rocdl.mfma_f32_32x32x16_f16(v16f32_type, [a_pack, b_pack, c_acc])
            return rocdl.mfma_f32_32x32x8f16(v16f32_type, [a_pack, b_pack, c_acc])

        # ---- Pre-load K and V packs for this wave's N sub-tile ----
        # Bounds against this_seqlen_k (this batch's REAL length, varlen) rather
        # than seq_N_idx (max_seqlen_k, used only for grid/loop sizing) -- see
        # module docstring. Non-varlen: this_seqlen_k == seq_N_idx, unchanged.
        n_global_wave_base = n_start + wave_n_sub * 32
        n_row_abs_kv = n_global_wave_base + lane_mod_32
        n_valid_kv = n_row_abs_kv < this_seqlen_k
        n_safe_kv = n_valid_kv.select(n_row_abs_kv, this_seqlen_k - fx.Index(1))
        kv_row_g_pre = _kv_row(n_safe_kv)

        k_packs = []
        v_packs = []
        for ks in range_constexpr(K_STEPS):
            col_off = fx.Index(ks * MFMA_K) + lane_div_32 * MFMA_LK
            k_packs.append(_load_global_vec_cv(k_rsrc, kv_row_g_pre, col_off))
            v_packs.append(_load_global_vec_cv(v_rsrc, kv_row_g_pre, col_off))

        # One dk/dv accumulator PER D-subtile this wave sequentially owns
        # (D_SUBS_PER_WAVE == 1 for D==BLOCK_N==64, matching the original single-accumulator
        # behavior; >1 for D=128/256, where this wave loops over multiple 32-col D chunks).
        dk_inits = [Vec.filled(16, 0.0, fx.Float32) for _ in range(D_SUBS_PER_WAVE)]
        dv_inits = [Vec.filled(16, 0.0, fx.Float32) for _ in range(D_SUBS_PER_WAVE)]
        dummy_val = fx.Float32(0.0)
        init_st = dk_inits + dv_inits + [dummy_val]

        # GQA: dk_accs/dv_accs must accumulate across
        # ALL `heads_per_kv` Q-heads sharing this block's kv_head_idx (see module
        # docstring) -- reset ONCE (init_st) before the group, not per q-head.
        # The m_tile scf.for loop is re-entered once per q_head_in_group, each
        # time threading the running accumulator through as its `init`.
        # heads_per_kv==1 (non-GQA): this loop runs once, degenerating to the
        # original per-head-unique behavior.
        loop_results = init_st
        for q_head_in_group in range_constexpr(heads_per_kv):
            head_idx = kv_head_idx * heads_per_kv + q_head_in_group

            for m_tile, iter_args in range(
                fx.Index(0), n_M_tiles_idx, fx.Index(1), init=loop_results
            ):
                dk_accs = list(iter_args[0:D_SUBS_PER_WAVE])
                dv_accs = list(iter_args[D_SUBS_PER_WAVE : 2 * D_SUBS_PER_WAVE])
                m_start = m_tile * BLOCK_M

                # ---- Cooperative LDS load: Q and dO tiles ----
                VEC_COLS = D // MFMA_LK
                ROWS_PER_WAVE_LD = BLOCK_M // NUM_WAVES
                if use_pipeline:
                    # Lane-distributed cooperative load: the (row_off, cv) work items of
                    # this wave are spread across its 64 lanes (baseline had every lane
                    # redundantly issue ALL items -> 64x redundant global loads + same-
                    # address LDS stores). 32 rows * 8 cvs = 256 items / 64 lanes = 4/lane.
                    N_ITEMS_LD = ROWS_PER_WAVE_LD * VEC_COLS
                    ITEMS_PER_LANE = N_ITEMS_LD // WARP_SIZE
                    for it in range_constexpr(ITEMS_PER_LANE):
                        item = lane + fx.Index(it * WARP_SIZE)
                        row_off_i = item // fx.Index(VEC_COLS)
                        cv_i = item % fx.Index(VEC_COLS)
                        row_in_tile = wave * ROWS_PER_WAVE_LD + row_off_i
                        m_global_ld = m_start + row_in_tile
                        m_valid_ld = m_global_ld < this_seqlen_q
                        m_safe_ld = m_valid_ld.select(
                            m_global_ld, this_seqlen_q - fx.Index(1)
                        )
                        q_row_g = _q_row(m_safe_ld, head_idx)
                        do_row_g = _do_row(m_safe_ld, head_idx)
                        col_off_ld = cv_i * fx.Index(MFMA_LK)
                        q_vec = _load_global_vec_cv(q_rsrc, q_row_g, col_off_ld)
                        do_vec = _load_global_vec_cv(do_rsrc, do_row_g, col_off_ld)
                        lds_base = row_in_tile * LDS_Q_STRIDE + col_off_ld
                        Vec(q_vec).store(lds_q, [lds_base])
                        Vec(do_vec).store(lds_do, [lds_base])
                else:
                    for row_off in range_constexpr(ROWS_PER_WAVE_LD):
                        row_in_tile = wave * ROWS_PER_WAVE_LD + row_off
                        m_global_ld = m_start + row_in_tile
                        m_valid_ld = m_global_ld < this_seqlen_q
                        m_safe_ld = m_valid_ld.select(
                            m_global_ld, this_seqlen_q - fx.Index(1)
                        )
                        q_row_g = _q_row(m_safe_ld, head_idx)
                        do_row_g = _do_row(m_safe_ld, head_idx)
                        for cv in range_constexpr(VEC_COLS):
                            col_off_ld = fx.Index(cv * MFMA_LK)
                            q_vec = _load_global_vec_cv(q_rsrc, q_row_g, col_off_ld)
                            do_vec = _load_global_vec_cv(do_rsrc, do_row_g, col_off_ld)
                            lds_base = row_in_tile * LDS_Q_STRIDE + cv * MFMA_LK
                            Vec(q_vec).store(lds_q, [lds_base])
                            Vec(do_vec).store(lds_do, [lds_base])

                # ---- Cooperative LSE + D_vec tile stage ----
                tid_idx = fx.Index(tid)
                if tid_idx < fx.Index(BLOCK_M):
                    m_g_ls = m_start + tid_idx
                    m_ok_ls = m_g_ls < this_seqlen_q
                    m_sf_ls = m_ok_ls.select(m_g_ls, this_seqlen_q - fx.Index(1))
                    lse_g = _load_f32_row(LSE_buf, _lse_row(m_sf_ls, head_idx))
                    dm_g = _load_f32_row(Dvec_buf, _dvec_row(m_sf_ls, head_idx))
                    Vec.from_elements([lse_g], fx.Float32).store(lds_lse, [tid_idx])
                    Vec.from_elements([dm_g], fx.Float32).store(lds_dm, [tid_idx])

                gpu.barrier()

                log2e_scale_cst = fx.Float32(_LOG2E * scale)
                M_SUBTILES = BLOCK_M // 32
                for m_sub in range_constexpr(M_SUBTILES):
                    # ---- GEMM1a: S = Q @ K^T ; GEMM1b: dP = dO @ V^T ----
                    s_acc = Vec.filled(16, 0.0, fx.Float32)
                    dp_acc = Vec.filled(16, 0.0, fx.Float32)
                    for ks in range_constexpr(K_STEPS):
                        q_pack = _lds_load_pack_a(lds_q, m_sub * 32, ks)
                        do_pack = _lds_load_pack_a(lds_do, m_sub * 32, ks)
                        s_acc = mfma(q_pack, k_packs[ks], s_acc)
                        dp_acc = mfma(do_pack, v_packs[ks], dp_acc)

                    # ---- P (for dV) and dS' (for dK, scale deferred to store), both to LDS[m,n] ----
                    n_within = lane_mod_32
                    n_row_abs = n_within + wave_n_sub * 32 + n_start
                    n_ok = n_row_abs < this_seqlen_k
                    for r in range_constexpr(16):
                        m_within = lane_div_32 * 4 + ((r // 4) * 8 + (r % 4))
                        m_row_abs = m_within + (m_sub * 32) + m_start
                        m_valid = m_row_abs < this_seqlen_q
                        m_local_f = m_within + (m_sub * 32)
                        lse_val = Vec.load(
                            Vec.make_type(1, fx.Float32), lds_lse, [m_local_f]
                        )[0]
                        dm_val = Vec.load(
                            Vec.make_type(1, fx.Float32), lds_dm, [m_local_f]
                        )[0]
                        s_val = Vec(s_acc)[r]
                        dp_val = Vec(dp_acc)[r]
                        valid_mn = m_valid & n_ok
                        if const_expr(causal):
                            valid_mn = valid_mn & (n_row_abs <= m_row_abs)
                        neg_log2e_lse = fx.Float32(
                            arith.mulf(
                                _raw(lse_val), _raw(fx.Float32(-_LOG2E)), fastmath=fm
                            )
                        )
                        p_val = _softmax_p(
                            s_val, neg_log2e_lse, log2e_scale_cst, valid_mn, fm
                        )
                        ds_val = _grad_ds_unscaled(p_val, dp_val, dm_val, valid_mn, fm)
                        m_local = m_within + (m_sub * 32)
                        n_local = n_within + wave_n_sub * 32
                        Vec.from_elements([p_val], fx.Float32).to(elem_dtype).store(
                            lds_p, [n_local * LDS_MPAD + m_local]
                        )
                        Vec.from_elements([ds_val], fx.Float32).to(elem_dtype).store(
                            lds_ds, [n_local * LDS_MPAD + m_local]
                        )

                    gpu.barrier()

                    # ---- dV += P^T @ dO ; dK += dS^T @ Q ----
                    # A=P^T/dS^T[n,m]: free=n=lane%32, k=m; B=dO/Q[m,d]: free=d=lane%32, k=m.
                    # P/dS (the A-operand) do NOT depend on d, so they're loaded ONCE per ks and
                    # reused across every D-subtile this wave sequentially owns (D_SUBS_PER_WAVE
                    # loop below) — only the B-operand (dO/Q at a given d) changes per subtile.
                    MFMA_KS = 32 // MFMA_K
                    for ks in range_constexpr(MFMA_KS):
                        n_local = lane_mod_32 + wave_n_sub * 32
                        if use_trload:
                            # A-operand (P^T/dS^T): must hold the SAME m as the tr B-operand at
                            # each hardware slot (independent of which D-subtile is being read).
                            # B (tr) gives m = (m_sub*32+ks*16) + lane_div_32*4 + P8[e],
                            # P8={0,1,2,3,8,9,10,11}. So load P^T[n, base_a+{0,1,2,3}] ++
                            # P^T[n, base_a+{8,9,10,11}].
                            base_a = lane_div_32 * 4 + (m_sub * 32 + ks * MFMA_K)
                            p_lo = Vec.load(
                                v4elem_type, lds_p, [n_local * LDS_MPAD + base_a]
                            )
                            p_hi = Vec.load(
                                v4elem_type, lds_p, [n_local * LDS_MPAD + base_a + 8]
                            )
                            p_pack = (
                                Vec(p_lo)
                                .shuffle(Vec(p_hi), [0, 1, 2, 3, 4, 5, 6, 7])
                                .ir_value()
                            )
                            ds_lo = Vec.load(
                                v4elem_type, lds_ds, [n_local * LDS_MPAD + base_a]
                            )
                            ds_hi = Vec.load(
                                v4elem_type, lds_ds, [n_local * LDS_MPAD + base_a + 8]
                            )
                            ds_pack = (
                                Vec(ds_lo)
                                .shuffle(Vec(ds_hi), [0, 1, 2, 3, 4, 5, 6, 7])
                                .ir_value()
                            )
                            tr_k_group = (
                                lane_mod_32 % 16
                            ) // 4  # lane%16 //4 within 32-lane half
                            tr_col_sub = lane % 4
                            tr_col_half = lane_mod_32 // 16
                            m_base = (
                                m_sub * 32 + ks * MFMA_K + lane_div_32 * 4 + tr_k_group
                            )
                            for d_iter in range_constexpr(D_SUBS_PER_WAVE):
                                wave_d_sub_i_raw = wave_d_sub_base + d_iter
                                # D=32/96 (D_TOTAL_SUBS not evenly divisible by WAVES_PER_N_GROUP):
                                # the last wave-group's nominal D-subtile range can run past
                                # D_TOTAL_SUBS. Clamp the address to subtile 0 (always in-bounds)
                                # for out-of-range iterations; the result is discarded at the store
                                # site below via d_in_range, so the clamped value is never observed.
                                d_in_range = wave_d_sub_i_raw < fx.Index(D_TOTAL_SUBS)
                                wave_d_sub_i = d_in_range.select(
                                    wave_d_sub_i_raw, fx.Index(0)
                                )
                                # B-operand (dO/Q) via HW transpose. tr yields, per lane, contract
                                # m = P8 = {0,1,2,3,8,9,10,11} + lane_div_32*4 (relative to m_row
                                # base), free d = lane%32. Read row-major [m,d] with EVEN LDS_Q_STRIDE.
                                d_col = (
                                    wave_d_sub_i * 32
                                    + tr_col_half * 16
                                    + tr_col_sub * 4
                                )
                                lo = m_base * LDS_Q_STRIDE + d_col
                                hi = lo + 8 * LDS_Q_STRIDE
                                do_a = _ds_read_tr_v4(v4elem_type, lo, lds_do_off)
                                do_b = _ds_read_tr_v4(v4elem_type, hi, lds_do_off)
                                do_pack = (
                                    Vec(do_a)
                                    .shuffle(Vec(do_b), [0, 1, 2, 3, 4, 5, 6, 7])
                                    .ir_value()
                                )
                                q_a = _ds_read_tr_v4(v4elem_type, lo, lds_q_off)
                                q_b = _ds_read_tr_v4(v4elem_type, hi, lds_q_off)
                                q_pack = (
                                    Vec(q_a)
                                    .shuffle(Vec(q_b), [0, 1, 2, 3, 4, 5, 6, 7])
                                    .ir_value()
                                )
                                dv_accs[d_iter] = mfma(p_pack, do_pack, dv_accs[d_iter])
                                dk_accs[d_iter] = mfma(ds_pack, q_pack, dk_accs[d_iter])
                        else:
                            base_m = lane_div_32 * MFMA_LK + (m_sub * 32 + ks * MFMA_K)
                            p_pack = Vec.load(
                                v8elem_type, lds_p, [n_local * LDS_MPAD + base_m]
                            ).ir_value()
                            ds_pack = Vec.load(
                                v8elem_type, lds_ds, [n_local * LDS_MPAD + base_m]
                            ).ir_value()
                            for d_iter in range_constexpr(D_SUBS_PER_WAVE):
                                wave_d_sub_i_raw = wave_d_sub_base + d_iter
                                # See the use_trload branch above for why out-of-range D-subtiles
                                # (D=32/96) are clamped rather than skipped: the LDS address must
                                # stay in-bounds even though the result is discarded at store time.
                                d_in_range = wave_d_sub_i_raw < fx.Index(D_TOTAL_SUBS)
                                wave_d_sub_i = d_in_range.select(
                                    wave_d_sub_i_raw, fx.Index(0)
                                )
                                d_local = lane_mod_32 + wave_d_sub_i * 32
                                # B-operand (dO / Q): scatter load with padded stride LDS_Q_STRIDE=D+2
                                # for bank-conflict-free access (16 consecutive m-rows hit 16 distinct banks).
                                do_r = fx.make_rmem_tensor(MFMA_LK, elem_dtype)
                                q_r = fx.make_rmem_tensor(MFMA_LK, elem_dtype)
                                for e in range_constexpr(MFMA_LK):
                                    m_local = lane_div_32 * MFMA_LK + (
                                        m_sub * 32 + ks * MFMA_K + e
                                    )
                                    do_sc = Vec.load(
                                        Vec.make_type(1, elem_dtype),
                                        lds_do,
                                        [m_local * LDS_Q_STRIDE + d_local],
                                    )[0]
                                    q_sc = Vec.load(
                                        Vec.make_type(1, elem_dtype),
                                        lds_q,
                                        [m_local * LDS_Q_STRIDE + d_local],
                                    )[0]
                                    fx.memref_store(do_sc, do_r, e)
                                    fx.memref_store(q_sc, q_r, e)
                                dv_accs[d_iter] = mfma(
                                    p_pack, fx.memref_load_vec(do_r), dv_accs[d_iter]
                                )
                                dk_accs[d_iter] = mfma(
                                    ds_pack, fx.memref_load_vec(q_r), dk_accs[d_iter]
                                )

                    gpu.barrier()

                loop_results = yield dk_accs + dv_accs + [dummy_val]

        # ---- Store dV and dK ----  (same output decode: M=n_key varies with r, N=d fixed)
        # Loop-invariant `scale` (deferred from the per-element dS' epilogue above) is
        # applied once here: MFMA is linear in its A-operand, so scale*(dS'@Q) == (scale*dS')@Q.
        scale_cst = fx.Float32(scale)
        dk_finals = loop_results[0:D_SUBS_PER_WAVE]
        dv_finals = loop_results[D_SUBS_PER_WAVE : 2 * D_SUBS_PER_WAVE]
        for r in range_constexpr(16):
            n_within = lane_div_32 * 4 + ((r // 4) * 8 + (r % 4))
            n_row_abs = n_within + wave_n_sub * 32 + n_start
            n_ok = n_row_abs < this_seqlen_k
            n_safe = n_ok.select(n_row_abs, this_seqlen_k - fx.Index(1))
            kv_row_g = _kv_row_out(n_safe)
            for d_iter in range_constexpr(D_SUBS_PER_WAVE):
                wave_d_sub_i = wave_d_sub_base + d_iter
                d_col_abs = lane_mod_32 + wave_d_sub_i * 32
                # D=32/96: this wave's nominal D-subtile range can run past D (see
                # D_SUBS_PER_WAVE comment above) -- skip the store for out-of-range columns.
                d_ok = d_col_abs < fx.Index(D)
                flat_col = fx.Int32(fx.Index(kv_row_g) * fx.Index(D) + d_col_abs)
                if n_ok & d_ok:
                    dk_scaled = fx.Float32(
                        arith.mulf(
                            _raw(Vec(dk_finals[d_iter])[r]),
                            _raw(scale_cst),
                            fastmath=fm,
                        )
                    )
                    _store_f32_row(dV_buf, flat_col, Vec(dv_finals[d_iter])[r])
                    _store_f32_row(dK_buf, flat_col, dk_scaled)

    @flyc.jit
    def launch_fn(
        Q: fx.Tensor,
        K: fx.Tensor,
        V: fx.Tensor,
        dO: fx.Tensor,
        dV: fx.Tensor,
        dK: fx.Tensor,
        LSE: fx.Tensor,
        D_vec: fx.Tensor,
        B: fx.Int32,
        M: fx.Int32,
        N: fx.Int32,
        H: fx.Int32,
        n_M_tiles: fx.Int32,
        q_stride_m: fx.Int32,
        kv_stride_n: fx.Int32,
        do_stride_m: fx.Int32,
        seqstart_q: fx.Tensor,
        seqstart_k: fx.Tensor,
        total_m: fx.Int32,
        stream: fx.Stream,
    ):
        from flydsl._mlir import ir
        from flydsl.compiler.kernel_function import CompilationContext

        allocator.finalized = False
        _ctx = CompilationContext.get_current()
        with ir.InsertionPoint(_ctx.gpu_module_body):
            allocator.finalize()

        num_N_tiles = (fx.Index(N) + BLOCK_N - 1) // BLOCK_N
        # GQA: grid is regrouped by KV-head (see kernel
        # docstring) -- H // heads_per_kv KV-heads, not H (Hq) blocks per batch.
        n_kv_heads_idx = fx.Index(H) // heads_per_kv
        grid_x = fx.Int32(fx.Index(B) * n_kv_heads_idx * num_N_tiles)
        fmha_bwd_dvdk_mfma_kernel(
            Q,
            K,
            V,
            dO,
            dV,
            dK,
            LSE,
            D_vec,
            M,
            N,
            H,
            n_M_tiles,
            q_stride_m,
            kv_stride_n,
            do_stride_m,
            seqstart_q,
            seqstart_k,
            total_m,
        ).launch(
            grid=(grid_x, 1, 1),
            block=(BLOCK_SIZE, 1, 1),
            stream=stream,
        )

    return launch_fn


def compile_fmha_bwd_dq_mfma(
    *,
    D: int = 64,
    dtype_str: str = "bf16",
    BLOCK_M: int = 64,
    BLOCK_N: int = 64,
    scale: float = None,
    use_pipeline: bool = False,
    gpu_arch: str = "gfx950",
    causal: bool = False,
    heads_per_kv: int = 1,
    varlen: bool = False,
):
    """Standalone dQ backward kernel with MFMA. Grid over M-tiles, loop over N-tiles.

    dQ[m,d] = sum_n dS[m,n] * K[n,d],  dS = scale * P * (dP - D_vec[m])
      S  = Q @ K^T          (GEMM1a)
      dP = dO @ V^T         (GEMM1b)
      dS = scale*P*(dP-Dm)  (elementwise; store to LDS[m,n])
      dQ = dS @ K           (GEMM2; contract over n)

    Mirror of dK: Q/dO are the fixed register A-packs; K/V stream into LDS each
    N-tile (K used in two layouts -> must be LDS). Each block owns a full M-tile
    and accumulates over all N-tiles in registers -> no atomics.

    Returns:
        launch_fn(Q, K, V, dO, dQ, LSE, D_vec, B, M, N, H, n_N_tiles,
                  q_stride_m, kv_stride_n, do_stride_m, stream)
          dQ : [B*M*H*D, 1] float32 (always contiguous output)
          q_stride_m/kv_stride_n: row pitch in ROW units (real_elem_stride(dim=1)
            // D) for possibly-non-contiguous Q/dO and K/V inputs; H for
            contiguous BMHK.
          heads_per_kv (compile-time): GQA head ratio (H // Hkv, 1 if not GQA).
            K/V are indexed by kv_head_idx = head_idx // heads_per_kv; dQ output
            stays uniquely addressed by head_idx, so no accumulation change needed.
          varlen: mirrors compile_fmha_bwd_dvdk_mfma's `varlen` -- see that
            docstring for the full design.
    """
    import math as _pm

    if scale is None:
        scale = 1.0 / _pm.sqrt(D)
    assert BLOCK_M == 64, (
        f"dq requires BLOCK_M == 64 (wave-tiling fixed), got {BLOCK_M}"
    )
    assert D % 32 == 0, (
        f"dq requires D a multiple of 32 (wave D-subtile width), got D={D}"
    )
    assert D % 16 == 0

    # See compile_fmha_bwd_dvdk_mfma for the K16-vs-K8 dispatch rationale.
    USE_K16 = gpu_arch.startswith("gfx950")

    elem_dtype = dtype_to_elem_type(dtype_str)
    MFMA_K = 16 if USE_K16 else 8
    MFMA_LK = 8 if USE_K16 else 4
    K_STEPS = D // MFMA_K
    fm = arith.FastMathFlags.fast

    BLOCK_SIZE = 256
    NUM_WAVES = BLOCK_SIZE // WARP_SIZE  # 4
    WAVE_M_TILES = BLOCK_M // 32  # 2 (BLOCK_M=64 fixed)
    WAVES_PER_M_GROUP = NUM_WAVES // WAVE_M_TILES  # 2
    D_TOTAL_SUBS = D // 32  # 1,2,3,4,8 for D=32,64,96,128,256
    # ceil-div + out-of-range clamp/guard for D=32/96 (not evenly divisible by
    # WAVES_PER_M_GROUP); see compile_fmha_bwd_dvdk_mfma for the full rationale.
    D_SUBS_PER_WAVE = -(-D_TOTAL_SUBS // WAVES_PER_M_GROUP)
    # wave sequentially covers D_SUBS_PER_WAVE contiguous 32-col D-subtiles (mirrors dvdk's
    # generalization; see compile_fmha_bwd_dvdk_mfma for the full rationale).

    # LDS: K tile + V tile [BLOCK_N, D] + dS scratch [BLOCK_M, BLOCK_N].
    LDS_K_ELEMS = BLOCK_N * D
    LDS_V_ELEMS = BLOCK_N * D
    LDS_DS_ELEMS = BLOCK_M * BLOCK_N

    allocator = SmemAllocator(
        None, arch=gpu_arch, global_sym_name="fmha_bwd_dq_mfma_smem"
    )
    lds_k_off = allocator._align(allocator.ptr, 16)
    allocator.ptr = lds_k_off + LDS_K_ELEMS * 2
    lds_v_off = allocator._align(allocator.ptr, 16)
    allocator.ptr = lds_v_off + LDS_V_ELEMS * 2
    lds_ds_off = allocator._align(allocator.ptr, 16)
    allocator.ptr = lds_ds_off + LDS_DS_ELEMS * 2

    @flyc.kernel(known_block_size=[BLOCK_SIZE, 1, 1])
    def fmha_bwd_dq_mfma_kernel(  # noqa: F811
        Q: fx.Tensor,
        K: fx.Tensor,
        V: fx.Tensor,
        dO: fx.Tensor,
        dQ: fx.Tensor,
        LSE: fx.Tensor,
        D_vec: fx.Tensor,
        seq_M: fx.Int32,
        seq_N: fx.Int32,
        n_heads: fx.Int32,
        n_N_tiles: fx.Int32,
        q_stride_m: fx.Int32,
        kv_stride_n: fx.Int32,
        do_stride_m: fx.Int32,
        seqstart_q: fx.Tensor,
        seqstart_k: fx.Tensor,
        total_m: fx.Int32,
    ):
        bid = fx.block_idx.x
        tid = fx.thread_idx.x

        n_heads_idx = fx.Index(n_heads)
        seq_M_idx = fx.Index(seq_M)
        seq_N_idx = fx.Index(seq_N)
        n_N_tiles_idx = fx.Index(n_N_tiles)
        q_stride_m_idx = fx.Index(q_stride_m)
        kv_stride_n_idx = fx.Index(kv_stride_n)
        do_stride_m_idx = fx.Index(do_stride_m)
        # varlen: see compile_fmha_bwd_dvdk_mfma for total_m_idx's role
        # (LSE/D_vec packed row pitch, distinct from seq_M_idx == max_seqlen_q).
        total_m_idx = fx.Index(total_m)
        num_M_tiles = (seq_M_idx + BLOCK_M - 1) // BLOCK_M

        bid_idx = fx.Index(bid)
        m_tile = bid_idx % num_M_tiles
        bh_idx = bid_idx // num_M_tiles
        batch_idx = bh_idx // n_heads_idx
        head_idx = bh_idx % n_heads_idx
        # GQA: heads_per_kv is a compile-time constant
        # (like `causal`), not a runtime kernel arg -- Hq/Hkv are fixed at compile
        # time via flydsl.py's per-shape kernel cache key.
        kv_head_idx = head_idx // heads_per_kv

        m_start = m_tile * BLOCK_M

        # varlen: see compile_fmha_bwd_dvdk_mfma for the full design --
        # q_start/k_start (packed-row-offset bases) and this_seqlen_q/this_seqlen_k
        # (per-batch REAL length, for masking) replace the globally-uniform
        # batch_idx*seq_len_idx/seq_len_idx. Non-varlen: reduces to the original.
        if const_expr(varlen):
            from flydsl.expr import buffer_ops as _seq_bops

            seqstart_q_rsrc = _seq_bops.create_buffer_resource(seqstart_q)
            seqstart_k_rsrc = _seq_bops.create_buffer_resource(seqstart_k)

            def _seqstart_load(rsrc, idx):
                return fx.Index(
                    _seq_bops.buffer_load(
                        rsrc, fx.Index(idx), vec_width=1, dtype=fx.Int32
                    )
                )

            q_start = _seqstart_load(seqstart_q_rsrc, batch_idx)
            k_start = _seqstart_load(seqstart_k_rsrc, batch_idx)
            this_seqlen_q = _seqstart_load(seqstart_q_rsrc, batch_idx + 1) - q_start
            this_seqlen_k = _seqstart_load(seqstart_k_rsrc, batch_idx + 1) - k_start
        else:
            q_start = batch_idx * seq_M_idx
            k_start = batch_idx * seq_N_idx
            this_seqlen_q = seq_M_idx
            this_seqlen_k = seq_N_idx

        wave = fx.Index(tid // WARP_SIZE)
        lane = fx.Index(tid % WARP_SIZE)
        lane_mod_32 = fx.Index(lane % 32)
        lane_div_32 = fx.Index(lane // 32)
        wave_m_sub = fx.Index(wave // WAVES_PER_M_GROUP)
        wave_d_group = fx.Index(wave % WAVES_PER_M_GROUP)
        wave_d_sub_base = wave_d_group * D_SUBS_PER_WAVE

        dQ_buf = fx.rocdl.make_buffer_tensor(dQ)
        LSE_buf = fx.rocdl.make_buffer_tensor(LSE)
        Dvec_buf = fx.rocdl.make_buffer_tensor(D_vec)

        copy_f32 = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.Float32)
        store_f32 = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.Float32)

        v8elem_type = Vec.make_type(MFMA_LK, elem_dtype)
        v16f32_type = Vec.make_type(16, fx.Float32)

        base_ptr = allocator.get_base()
        lds_k = SmemPtr(
            base_ptr, lds_k_off, elem_dtype.ir_type, shape=(LDS_K_ELEMS,)
        ).get()
        lds_v = SmemPtr(
            base_ptr, lds_v_off, elem_dtype.ir_type, shape=(LDS_V_ELEMS,)
        ).get()
        lds_ds = SmemPtr(
            base_ptr, lds_ds_off, elem_dtype.ir_type, shape=(LDS_DS_ELEMS,)
        ).get()

        # Q/dO and K/V are possibly-non-contiguous user inputs (e.g. a packed-qkv
        # unbind view) -- row pitch is q_stride_m_idx/kv_stride_n_idx (in row
        # units, i.e. real_elem_stride // D), NOT necessarily n_heads_idx. dO can
        # have a DIFFERENT row pitch than Q -- separate helper. varlen: q_start/
        # k_start (packed-row-offset bases) replace batch_idx*seq_len_idx -- see
        # compile_fmha_bwd_dvdk_mfma for the full design.
        def _q_row(q_pos):
            return fx.Int32((q_start + q_pos) * q_stride_m_idx + head_idx)

        def _do_row(q_pos):
            return fx.Int32((q_start + q_pos) * do_stride_m_idx + head_idx)

        # GQA: K/V are indexed by kv_head_idx (= head_idx // heads_per_kv), not
        # head_idx -- multiple Q heads share one KV head. kv_stride_n_idx is K/V's
        # own row pitch (in row units), unaffected by GQA head-count.
        def _kv_row(kv_pos):
            return fx.Int32((k_start + kv_pos) * kv_stride_n_idx + kv_head_idx)

        # dQ is a freshly-allocated contiguous output -- always row pitch n_heads_idx.
        def _q_row_out(q_pos):
            return fx.Int32((q_start + q_pos) * n_heads_idx + head_idx)

        # LSE layout is [B,H,M] (non-varlen) vs packed [1,H,sum_M] (varlen).
        # Not unifiable via a single q_start-relative formula since the
        # head-axis stride differs (seq_M_idx vs total_m_idx).
        def _lse_row(q_pos):
            if const_expr(varlen):
                return fx.Int32(head_idx * total_m_idx + q_start + q_pos)
            return fx.Int32(bh_idx * seq_M_idx + q_pos)

        # D_vec is a freshly-allocated contiguous tensor -- always row pitch n_heads_idx.
        def _dvec_row(q_pos):
            return _q_row_out(q_pos)

        from flydsl.expr import buffer_ops as _bops

        q_rsrc = _bops.create_buffer_resource(Q)
        k_rsrc = _bops.create_buffer_resource(K)
        v_rsrc = _bops.create_buffer_resource(V)
        do_rsrc = _bops.create_buffer_resource(dO)

        def _load_global_vec_cv(rsrc, row_i32, col_offset_idx):
            flat_elem = fx.Index(row_i32) * fx.Index(D) + col_offset_idx
            return _bops.buffer_load(
                rsrc, flat_elem, vec_width=MFMA_LK, dtype=elem_dtype
            )

        def _load_f32_row(buf, row_idx):
            row_sl = fx.slice(buf, (row_idx, None))
            div_1 = fx.logical_divide(row_sl, fx.make_layout(1, 1))
            r = fx.make_rmem_tensor(1, fx.Float32)
            fx.copy_atom_call(copy_f32, fx.slice(div_1, (None, 0)), r)
            return fx.memref_load(r, 0)

        def _store_f32_row(buf, row_idx, val):
            row_sl = fx.slice(buf, (row_idx, None))
            div_1 = fx.logical_divide(row_sl, fx.make_layout(1, 1))
            r = fx.make_rmem_tensor(1, fx.Float32)
            fx.memref_store(val, r, 0)
            fx.copy_atom_call(store_f32, r, fx.slice(div_1, (None, 0)))

        def _lds_load_pack_a(lds_arr, base_row_in_tile, k_step):
            lds_row = fx.Index(base_row_in_tile) + lane_mod_32
            lds_col = fx.Index(k_step * MFMA_K) + lane_div_32 * MFMA_LK
            return Vec.load(v8elem_type, lds_arr, [lds_row * D + lds_col]).ir_value()

        def mfma(a_pack, b_pack, c_acc):
            if const_expr(dtype_str == "bf16"):
                if const_expr(USE_K16):
                    return rocdl.mfma_f32_32x32x16_bf16(
                        v16f32_type, [a_pack, b_pack, c_acc]
                    )
                a_pack = Vec(a_pack).bitcast(fx.Int16)
                b_pack = Vec(b_pack).bitcast(fx.Int16)
                return rocdl.mfma_f32_32x32x8bf16_1k(
                    v16f32_type, [a_pack, b_pack, c_acc]
                )
            if const_expr(USE_K16):
                return rocdl.mfma_f32_32x32x16_f16(v16f32_type, [a_pack, b_pack, c_acc])
            return rocdl.mfma_f32_32x32x8f16(v16f32_type, [a_pack, b_pack, c_acc])

        # ---- Pre-load Q, dO A-packs for this wave's M sub-tile (constant across N-loop) ----
        # A-operand of S=Q@K^T and dP=dO@V^T: free=m=wave_m_sub*32+lane%32, contract=d.
        m_wave_base = m_start + wave_m_sub * 32
        m_row_abs_q = m_wave_base + lane_mod_32
        m_valid_q = m_row_abs_q < this_seqlen_q
        m_safe_q = m_valid_q.select(m_row_abs_q, this_seqlen_q - fx.Index(1))
        q_row_g_pre = _q_row(m_safe_q)
        do_row_g_pre = _do_row(m_safe_q)

        q_packs = []
        do_packs = []
        for ks in range_constexpr(K_STEPS):
            col_off = fx.Index(ks * MFMA_K) + lane_div_32 * MFMA_LK
            q_packs.append(_load_global_vec_cv(q_rsrc, q_row_g_pre, col_off))
            do_packs.append(_load_global_vec_cv(do_rsrc, do_row_g_pre, col_off))

        # ---- Pre-load per-r LSE, D_vec, m-validity (m varies with r, const across N) ----
        lse_vals = []
        dvec_vals = []
        m_valids = []
        m_row_abss = []
        for r in range_constexpr(16):
            m_within = lane_div_32 * 4 + ((r // 4) * 8 + (r % 4))
            m_row_abs = m_within + wave_m_sub * 32 + m_start
            m_valid = m_row_abs < this_seqlen_q
            m_safe = m_valid.select(m_row_abs, this_seqlen_q - fx.Index(1))
            lse_vals.append(_load_f32_row(LSE_buf, _lse_row(m_safe)))
            dvec_vals.append(_load_f32_row(Dvec_buf, _dvec_row(m_safe)))
            m_valids.append(m_valid)
            m_row_abss.append(m_row_abs)

        log2e_scale_cst = fx.Float32(_LOG2E * scale)
        neg_log2e_lse_vals = [
            fx.Float32(
                arith.mulf(_raw(lse_vals[r]), _raw(fx.Float32(-_LOG2E)), fastmath=fm)
            )
            for r in range(16)
        ]

        # One dq accumulator PER D-subtile this wave sequentially owns (mirrors dvdk's
        # generalization; D_SUBS_PER_WAVE==1 for D==BLOCK_M==64, unchanged from before).
        dq_inits = [Vec.filled(16, 0.0, fx.Float32) for _ in range(D_SUBS_PER_WAVE)]
        dummy_val = fx.Float32(0.0)
        init_dq = dq_inits + [dummy_val]

        # causal skip-ahead (mirrors compile_fmha_bwd_dqdkdv_mfma's identical-in-kind
        # optimization, applied to the OPPOSITE loop bound since this kernel's grid is
        # M-tile-major, not N-tile-major): for top-left causal masking, an M-tile at
        # rows [m_start, m_start+BLOCK_M) can only attend to N-tiles whose rows reach
        # AT MOST m_start+BLOCK_M-1 -- i.e. n_tile <= (m_start+BLOCK_M-1)//BLOCK_N,
        # so the loop's END (not start) is capped. N-tiles beyond this are entirely
        # masked out (every (m,n) pair in them has n > m). Ending the loop there
        # instead of at n_N_tiles_idx skips those fully-masked N-tiles entirely,
        # cutting both wasted MFMA and redundant per-N-tile K/V HBM reloads roughly
        # in half for causal shapes -- same magnitude win as compile_fmha_bwd_dqdkdv_mfma's
        # original causal skip-ahead finding.
        if const_expr(causal):
            n_tile_end_raw = (m_start + fx.Index(BLOCK_M) - fx.Index(1)) // fx.Index(
                BLOCK_N
            ) + fx.Index(1)
            n_tile_end = (n_tile_end_raw < n_N_tiles_idx).select(
                n_tile_end_raw, n_N_tiles_idx
            )
        else:
            n_tile_end = n_N_tiles_idx

        # ---- Software-pipelined K/V prefetch: the global load for N-tile
        # (n_tile+1) is issued interleaved with the CURRENT N-tile's GEMM1 MFMA
        # stream, so its VMEM latency overlaps with compute instead of stalling at
        # the top of the NEXT iteration. Prefetched registers are threaded through
        # the n_tile scf.for loop as extra iter_args and stored to LDS at the START
        # of the iteration that consumes them.
        VEC_COLS = D // MFMA_LK
        ROWS_PER_WAVE_LD = BLOCK_N // NUM_WAVES
        N_ITEMS_LD = ROWS_PER_WAVE_LD * VEC_COLS
        ITEMS_PER_LANE = N_ITEMS_LD // WARP_SIZE
        if ITEMS_PER_LANE < 1:
            ITEMS_PER_LANE = 1

        def _load_kv_item(n_tile_idx, it):
            n_start_p = n_tile_idx * BLOCK_N
            item = lane + fx.Index(it * WARP_SIZE)
            item_ok = item < fx.Index(N_ITEMS_LD)
            item_s = item_ok.select(item, fx.Index(0))
            row_off_i = item_s // fx.Index(VEC_COLS)
            cv_i = item_s % fx.Index(VEC_COLS)
            row_in_tile = wave * ROWS_PER_WAVE_LD + row_off_i
            n_global_ld = n_start_p + row_in_tile
            n_valid_ld = n_global_ld < this_seqlen_k
            n_safe_ld = n_valid_ld.select(n_global_ld, this_seqlen_k - fx.Index(1))
            kv_row_g = _kv_row(n_safe_ld)
            col_off_ld = cv_i * fx.Index(MFMA_LK)
            return (
                _load_global_vec_cv(k_rsrc, kv_row_g, col_off_ld),
                _load_global_vec_cv(v_rsrc, kv_row_g, col_off_ld),
            )

        def _load_kv_regs(n_tile_idx):
            regs_k, regs_v = [], []
            for it in range_constexpr(ITEMS_PER_LANE):
                k_v, v_v = _load_kv_item(n_tile_idx, it)
                regs_k.append(k_v)
                regs_v.append(v_v)
            return regs_k, regs_v

        def _store_kv_regs_to_lds(regs_k, regs_v):
            for it in range_constexpr(ITEMS_PER_LANE):
                item = lane + fx.Index(it * WARP_SIZE)
                row_off_i = item // fx.Index(VEC_COLS)
                cv_i = item % fx.Index(VEC_COLS)
                row_in_tile = wave * ROWS_PER_WAVE_LD + row_off_i
                col_off_ld = cv_i * fx.Index(MFMA_LK)
                lds_base = row_in_tile * D + col_off_ld
                Vec(regs_k[it]).store(lds_k, [lds_base])
                Vec(regs_v[it]).store(lds_v, [lds_base])

        prologue_k, prologue_v = _load_kv_regs(fx.Index(0))
        entry_state = init_dq + prologue_k + prologue_v

        loop_results = init_dq
        for n_tile, iter_args in range(
            fx.Index(0), n_tile_end, fx.Index(1), init=entry_state
        ):
            dq_accs = list(iter_args[0:D_SUBS_PER_WAVE])
            cur_k = list(
                iter_args[D_SUBS_PER_WAVE + 1 : D_SUBS_PER_WAVE + 1 + ITEMS_PER_LANE]
            )
            cur_v = list(
                iter_args[
                    D_SUBS_PER_WAVE + 1 + ITEMS_PER_LANE : D_SUBS_PER_WAVE
                    + 1
                    + 2 * ITEMS_PER_LANE
                ]
            )
            n_start = n_tile * BLOCK_N

            # ---- Store this iteration's already-prefetched K/V into LDS ----
            _store_kv_regs_to_lds(cur_k, cur_v)

            gpu.barrier()

            _MFMA_MASK = 0x008
            _VMEM_MASK = 0x020
            next_k = [None] * ITEMS_PER_LANE
            next_v = [None] * ITEMS_PER_LANE
            _next_load_it = 0  # plain Python int; range_constexpr unrolls at
            # compile time, safe to mutate directly here
            # (mirrors dqdkdv's identical pattern/caveat).

            N_SUBTILES = BLOCK_N // 32
            for n_sub in range_constexpr(N_SUBTILES):
                # ---- GEMM1a: S = Q @ K^T ; GEMM1b: dP = dO @ V^T ----
                # A=Q/dO (free=m), B=K/V (free=n=n_sub*32+lane%32). Output C[m,n].
                s_acc = Vec.filled(16, 0.0, fx.Float32)
                dp_acc = Vec.filled(16, 0.0, fx.Float32)
                for ks in range_constexpr(K_STEPS):
                    k_pack = _lds_load_pack_a(lds_k, n_sub * 32, ks)
                    v_pack = _lds_load_pack_a(lds_v, n_sub * 32, ks)
                    s_acc = mfma(q_packs[ks], k_pack, s_acc)
                    dp_acc = mfma(do_packs[ks], v_pack, dp_acc)
                    rocdl.sched_group_barrier(_MFMA_MASK, 2, 0)
                    if const_expr(_next_load_it < ITEMS_PER_LANE):
                        k_v, v_v = _load_kv_item(n_tile + fx.Index(1), _next_load_it)
                        next_k[_next_load_it] = k_v
                        next_v[_next_load_it] = v_v
                        _next_load_it += 1
                    rocdl.sched_group_barrier(_VMEM_MASK, 1, 0)

                # Any remaining prefetch items (ITEMS_PER_LANE > N_SUBTILES*K_STEPS,
                # e.g. wide D) are issued right after this n_sub's GEMM1 -- still
                # overlaps this n_sub's epilogue/dQ-contraction compute below.
                if const_expr(n_sub == N_SUBTILES - 1):
                    for _pad_it in range_constexpr(ITEMS_PER_LANE):
                        if const_expr(_pad_it >= _next_load_it):
                            k_v, v_v = _load_kv_item(n_tile + fx.Index(1), _pad_it)
                            next_k[_pad_it] = k_v
                            next_v[_pad_it] = v_v

                # ---- dS' = P*(dP-D_vec[m]) ; store to LDS[m,n] (scale applied once at dQ store) ----
                n_within = lane_mod_32
                n_row_abs = n_within + n_sub * 32 + n_start
                n_ok = n_row_abs < this_seqlen_k
                for r in range_constexpr(16):
                    m_within = lane_div_32 * 4 + ((r // 4) * 8 + (r % 4))
                    dm_val = dvec_vals[r]
                    s_val = Vec(s_acc)[r]
                    dp_val = Vec(dp_acc)[r]
                    valid_mn = m_valids[r] & n_ok
                    if const_expr(causal):
                        valid_mn = valid_mn & (n_row_abs <= m_row_abss[r])
                    p_val = _softmax_p(
                        s_val, neg_log2e_lse_vals[r], log2e_scale_cst, valid_mn, fm
                    )
                    ds_val = _grad_ds_unscaled(p_val, dp_val, dm_val, valid_mn, fm)
                    m_local = m_within + wave_m_sub * 32
                    n_local = n_within + n_sub * 32
                    ds_vec = Vec.from_elements([ds_val], fx.Float32).to(elem_dtype)
                    ds_vec.store(lds_ds, [m_local * BLOCK_N + n_local])

                gpu.barrier()

                # ---- dQ += dS @ K  (contract over this n_sub's 32 key rows) ----
                # A=dS[m,n]: free=m=lane%32, k=n=ks*16+lane//32*8+e (d-independent, loaded
                # once per ks and reused across every D-subtile this wave owns). As e varies,
                # the LDS address is contiguous (n_local increments by 1) -- one vector load.
                # B=K[n,d] : free=d=lane%32, k=n=ks*16+lane//32*8+e
                for ks in range_constexpr(32 // MFMA_K):
                    m_local = lane_mod_32 + wave_m_sub * 32
                    base_n = lane_div_32 * MFMA_LK + (n_sub * 32 + ks * MFMA_K)
                    dst_pack = Vec.load(
                        v8elem_type, lds_ds, [m_local * BLOCK_N + base_n]
                    ).ir_value()
                    for d_iter in range_constexpr(D_SUBS_PER_WAVE):
                        wave_d_sub_i_raw = wave_d_sub_base + d_iter
                        # D=32/96 (D_TOTAL_SUBS not evenly divisible by WAVES_PER_M_GROUP):
                        # clamp out-of-range D-subtiles to subtile 0 to keep the LDS address
                        # in-bounds; the store site below discards this iteration's result.
                        d_in_range = wave_d_sub_i_raw < fx.Index(D_TOTAL_SUBS)
                        wave_d_sub_i = d_in_range.select(wave_d_sub_i_raw, fx.Index(0))
                        d_local = lane_mod_32 + wave_d_sub_i * 32
                        k_r = fx.make_rmem_tensor(MFMA_LK, elem_dtype)
                        for e in range_constexpr(MFMA_LK):
                            n_local = lane_div_32 * MFMA_LK + (
                                n_sub * 32 + ks * MFMA_K + e
                            )
                            k_sc = Vec.load(
                                Vec.make_type(1, elem_dtype),
                                lds_k,
                                [n_local * D + d_local],
                            )[0]
                            fx.memref_store(k_sc, k_r, e)
                        dq_accs[d_iter] = mfma(
                            dst_pack, fx.memref_load_vec(k_r), dq_accs[d_iter]
                        )

                gpu.barrier()

            loop_results = yield dq_accs + [dummy_val] + next_k + next_v

        # ---- Store dQ ----  output C[M,N]: M=m (varies with r), N=d (fixed)
        # Loop-invariant `scale` (deferred from the per-element dS' epilogue above) is
        # applied once here: MFMA is linear in its A-operand, so scale*(dS'@K) == (scale*dS')@K.
        scale_cst = fx.Float32(scale)
        dq_finals = loop_results[0:D_SUBS_PER_WAVE]
        for r in range_constexpr(16):
            m_within = lane_div_32 * 4 + ((r // 4) * 8 + (r % 4))
            m_row_abs = m_within + wave_m_sub * 32 + m_start
            m_ok = m_row_abs < this_seqlen_q
            m_safe = m_ok.select(m_row_abs, this_seqlen_q - fx.Index(1))
            q_row_g = _q_row_out(m_safe)
            for d_iter in range_constexpr(D_SUBS_PER_WAVE):
                wave_d_sub_i = wave_d_sub_base + d_iter
                d_col_abs = lane_mod_32 + wave_d_sub_i * 32
                # D=32/96: this wave's nominal D-subtile range can run past D -- skip the
                # store for out-of-range columns (see D_SUBS_PER_WAVE comment above).
                d_ok = d_col_abs < fx.Index(D)
                flat_dq = fx.Int32(fx.Index(q_row_g) * fx.Index(D) + d_col_abs)
                val_f32 = fx.Float32(
                    arith.mulf(
                        _raw(Vec(dq_finals[d_iter])[r]), _raw(scale_cst), fastmath=fm
                    )
                )
                if m_ok & d_ok:
                    _store_f32_row(dQ_buf, flat_dq, val_f32)

    @flyc.jit
    def launch_fn(
        Q: fx.Tensor,
        K: fx.Tensor,
        V: fx.Tensor,
        dO: fx.Tensor,
        dQ: fx.Tensor,
        LSE: fx.Tensor,
        D_vec: fx.Tensor,
        B: fx.Int32,
        M: fx.Int32,
        N: fx.Int32,
        H: fx.Int32,
        n_N_tiles: fx.Int32,
        q_stride_m: fx.Int32,
        kv_stride_n: fx.Int32,
        do_stride_m: fx.Int32,
        seqstart_q: fx.Tensor,
        seqstart_k: fx.Tensor,
        total_m: fx.Int32,
        stream: fx.Stream,
    ):
        from flydsl._mlir import ir
        from flydsl.compiler.kernel_function import CompilationContext

        allocator.finalized = False
        _ctx = CompilationContext.get_current()
        with ir.InsertionPoint(_ctx.gpu_module_body):
            allocator.finalize()

        num_M_tiles = (fx.Index(M) + BLOCK_M - 1) // BLOCK_M
        grid_x = fx.Int32(fx.Index(B) * fx.Index(H) * num_M_tiles)
        fmha_bwd_dq_mfma_kernel(
            Q,
            K,
            V,
            dO,
            dQ,
            LSE,
            D_vec,
            M,
            N,
            H,
            n_N_tiles,
            q_stride_m,
            kv_stride_n,
            do_stride_m,
            seqstart_q,
            seqstart_k,
            total_m,
        ).launch(
            grid=(grid_x, 1, 1),
            block=(BLOCK_SIZE, 1, 1),
            stream=stream,
        )

    return launch_fn


def compile_fmha_bwd_dqdkdv_mfma(
    *,
    D: int = 64,
    dtype_str: str = "bf16",
    BLOCK_M: int = 64,
    BLOCK_N: int = 64,
    scale: float = None,
    use_pipeline: bool = True,
    use_lds_reduce: bool = False,
    use_trload: bool = False,
    causal: bool = False,
    heads_per_kv: int = 1,
    varlen: bool = False,
    gpu_arch: str = "gfx950",
    M_SPLIT: int = 1,
    BLOCK_SIZE: int = 256,
):
    """Fused dQ + dV + dK backward kernel in one N-tile-gridded pass.

    Extends compile_fmha_bwd_dvdk_mfma with the dQ GEMM grafted in, so
    Q/K/V/dO are loaded once and S/dP/P/dS are computed once for all three
    gradients (the fusion win vs running dvdk + dq separately).

    dV/dK are N-tile-unique -> register-accumulated, plain store (no atomics).
    dQ[m,d] = sum_n dS[m,n]*K[n,d] is SHARED across N-tile blocks -> each block
    contributes a partial via atomic-add into an f32 dQ scratch. A separate
    convert kernel casts the f32 scratch to the output dtype.

    dQ's wave assignment is DECOUPLED from GEMM1/2's (wave_n_sub, wave_d_group)
    N-split: all NUM_WAVES waves cover the SAME m_sub rows and each owns its
    own D-subtile (DQ_D_SUBS_PER_WAVE contiguous 32-col slices), sweeping the
    ENTIRE n range internally via an n_sub_iter loop. Every wave therefore
    produces one COMPLETE dq_acc and fires its own atomic directly -- no
    cross-wave reduction step. `use_lds_reduce` is now a no-op (kept for
    call-site compatibility). D=64: 2 of 4 waves are idle during the dQ step.

    use_trload=True: GEMM2's dO/Q B-operand is read via hardware LDS-transpose
    ds_read_tr16_b64 instead of the 8x scalar gather (gfx950-only). Requires
    EVEN LDS_Q_STRIDE (D+8, vs D+2 scalar-scatter). Does NOT touch the dQ
    contraction's K-operand (lds_k) -- that stays scalar-gather.

    D generalization: BLOCK_N stays fixed; each wave sequentially owns
    D_SUBS_PER_WAVE contiguous 32-col D-subtiles. D=32/96 excess waves
    compute a redundant subtile that is never stored.

    causal: masks P/dS to 0 where n_row_abs > m_row_abs.

    GQA (heads_per_kv > 1): grid regrouped by KV-head, each block loops over
    its heads_per_kv Q-heads accumulating dV/dK across the group. dQ is
    Q-head-unique (each iteration atomic-adds independently).

    varlen: B collapses to 1; per-batch lengths from seqstart_q/seqstart_k.

    M_SPLIT (grid-occupancy fix for large BLOCK_N): M_SPLIT>1 splits each
    N-tile's M-sweep across M_SPLIT blocks (grid becomes
    B*Hkv*num_N_tiles*M_SPLIT). Each chunk is computed at runtime per n_tile
    (m_tile_start varies under causal). dV/dK store becomes atomic-add when
    M_SPLIT>1 (multiple blocks contribute partials). M_SPLIT=1 (default)
    uses plain stores, no atomics.

    Returns:
        launch_fn(Q, K, V, dO, dV, dK, dQ_f32, LSE, D_vec, B, M, N, H, n_M_tiles,
                  q_stride_m, kv_stride_n, do_stride_m, seqstart_q, seqstart_k,
                  total_m, stream)
          dQ_f32 : [B*M*H*D, 1] float32 scratch (zero before launch; convert after)
          dV, dK : [B*N*Hkv*D, 1] float32 (always contiguous; Hkv = H // heads_per_kv)
          q_stride_m/kv_stride_n/do_stride_m: row pitch in ROW units.
          seqstart_q/seqstart_k : (varlen only) int32 [B_logical+1] cumulative offsets.
    """
    import math as _pm
    import os as _os

    if scale is None:
        scale = 1.0 / _pm.sqrt(D)
    # BLOCK_N must divide evenly into 32-row wave-subtiles that themselves divide
    # evenly across NUM_WAVES (WAVE_N_TILES=BLOCK_N//32 must divide NUM_WAVES) --
    # i.e. BLOCK_N in {32, 64, 128}. At BLOCK_N=128: WAVES_PER_N_GROUP becomes 1,
    # so GEMM1/2 stop D-splitting waves within an N-group (every wave owns a
    # unique N-subtile) -- the ceil-div generalization (D_SUBS_PER_WAVE) handles
    # this without new code.
    assert BLOCK_N in (32, 64, 128), (
        f"requires BLOCK_N in (32,64,128) (wave-tiling), got {BLOCK_N}"
    )
    assert D % 32 == 0, f"requires D a multiple of 32 (wave D-subtile width), got D={D}"
    assert D % 16 == 0
    assert BLOCK_SIZE % WARP_SIZE == 0, (
        f"requires BLOCK_SIZE a multiple of WARP_SIZE={WARP_SIZE}, got {BLOCK_SIZE}"
    )
    assert (BLOCK_SIZE // WARP_SIZE) % (BLOCK_N // 32) == 0, (
        f"requires NUM_WAVES ({BLOCK_SIZE // WARP_SIZE}) a multiple of WAVE_N_TILES "
        f"({BLOCK_N // 32}) for GEMM1/2's wave-tiling, got BLOCK_SIZE={BLOCK_SIZE}, BLOCK_N={BLOCK_N}"
    )

    # ABLATION ONLY (perf profiling): skip the dQ atomic-add to isolate its cost.
    # Keeps the dQ GEMM so VALU/LDS are identical; dQ output is then wrong (do NOT
    # use for correctness). Set FMHA_ABLATE_DQ_ATOMIC=1.
    _ablate_atomic = _os.environ.get("FMHA_ABLATE_DQ_ATOMIC", "0") == "1"

    # gfx950 (CDNA4) has native K16 MFMA; gfx942 (CDNA3) only has native K8 -- same
    # K_STEPS/MFMA_LK-parameterized dispatch as compile_fmha_bwd_dvdk_mfma/dq_mfma.
    USE_K16 = gpu_arch.startswith("gfx950")
    # ds_read_tr16_b64 (used by use_trload) is a gfx950(CDNA4)-only HW-transpose LDS
    # read, unrelated to the MFMA K-width gap but also unavailable on gfx942 -- same
    # assert as compile_fmha_bwd_dvdk_mfma.
    assert not (use_trload and not USE_K16), (
        "use_trload requires gfx950 (ds_read_tr16_b64 is CDNA4-only)"
    )

    elem_dtype = dtype_to_elem_type(dtype_str)
    MFMA_K = 16 if USE_K16 else 8
    MFMA_LK = 8 if USE_K16 else 4
    K_STEPS = D // MFMA_K
    fm = arith.FastMathFlags.fast

    # BLOCK_SIZE (waves/block) is a compile-time kwarg (default 256/4-waves).
    # Increasing it (e.g. 512/8-waves) restores WAVES_PER_N_GROUP at wider
    # BLOCK_N without touching BLOCK_N itself (which separately controls Q/dO
    # HBM-reload volume).
    NUM_WAVES = BLOCK_SIZE // WARP_SIZE  # 4 at the default BLOCK_SIZE=256
    WAVE_N_TILES = BLOCK_N // 32  # 2 (BLOCK_N=64 fixed)
    WAVES_PER_N_GROUP = NUM_WAVES // WAVE_N_TILES  # 2
    D_TOTAL_SUBS = D // 32  # 1,2,3,4,8 for D=32,64,96,128,256
    # D_SUBS_PER_WAVE = ceil(D_TOTAL_SUBS / WAVES_PER_N_GROUP): D=64/128/256 divide evenly
    # (1/2/4). D=32/96 don't divide evenly across the 2 waves in a D-group -- the last
    # wave's nominal subtile range can run past D_TOTAL_SUBS; excess waves compute a
    # redundant/garbage out-of-range subtile that is simply never stored (guarded by
    # `wave_d_sub_i < D_TOTAL_SUBS` at the store site below) -- mirrors
    # compile_fmha_bwd_dvdk_mfma's identical wave-tiling generalization.
    D_SUBS_PER_WAVE = -(-D_TOTAL_SUBS // WAVES_PER_N_GROUP)
    # wave sequentially covers D_SUBS_PER_WAVE contiguous 32-col D-subtiles.

    # dQ-specific wave assignment (decoupled from GEMM1/2's N-split): splits the
    # OUTPUT (D) axis across all warps, each sweeping the FULL n range internally,
    # so no warp needs another's partial sum. This gives every wave a COMPLETE
    # dq_acc directly (no cross-wave reduction), at the cost of leaving
    # NUM_WAVES - ceil(D_TOTAL_SUBS/NUM_WAVES) waves idle during the dQ step
    # when D_TOTAL_SUBS < NUM_WAVES (e.g. D=64: 2 of 4 waves idle).
    DQ_D_SUBS_PER_WAVE = -(-D_TOTAL_SUBS // NUM_WAVES)
    # wave sequentially covers DQ_D_SUBS_PER_WAVE contiguous 32-col D-subtiles,
    # base = wave * DQ_D_SUBS_PER_WAVE (each wave's own unique D-range, no sharing).

    LDS_MPAD = BLOCK_M + 8
    # odd stride (D+2): bank-conflict-free scalar scatter (default). trload needs
    # EVEN stride (D+8) so ds_read_tr16_b64 keeps 64-bit column alignment -- same
    # tradeoff as compile_fmha_bwd_dvdk_mfma.
    LDS_Q_STRIDE = (D + 8) if use_trload else (D + 2)
    LDS_Q_ELEMS = BLOCK_M * LDS_Q_STRIDE
    LDS_DO_ELEMS = BLOCK_M * LDS_Q_STRIDE
    LDS_DS_ELEMS = BLOCK_N * LDS_MPAD
    LDS_P_ELEMS = BLOCK_N * LDS_MPAD
    # K in [n,d] layout for the dQ contraction. use_trload also transpose-loads
    # this buffer (dQ's B-operand) via ds_read_tr16_b64, which needs the same
    # EVEN-stride/anti-power-of-2-bank-conflict padding as LDS_Q_STRIDE above;
    # D alone is already even (D%32==0) but is a power of 2 for the common
    # D=64/128/256 shapes, so pad it the same way.
    LDS_K_STRIDE = (D + 8) if use_trload else D
    LDS_K_ELEMS = BLOCK_N * LDS_K_STRIDE
    LDS_LSE_ELEMS = BLOCK_M
    LDS_DM_ELEMS = BLOCK_M

    allocator = SmemAllocator(
        None, arch=gpu_arch, global_sym_name="fmha_bwd_dqdkdv_mfma_smem"
    )
    lds_q_off = allocator._align(allocator.ptr, 16)
    allocator.ptr = lds_q_off + LDS_Q_ELEMS * 2
    lds_do_off = allocator._align(allocator.ptr, 16)
    allocator.ptr = lds_do_off + LDS_DO_ELEMS * 2
    lds_ds_off = allocator._align(allocator.ptr, 16)
    allocator.ptr = lds_ds_off + LDS_DS_ELEMS * 2
    lds_p_off = allocator._align(allocator.ptr, 16)
    allocator.ptr = lds_p_off + LDS_P_ELEMS * 2
    lds_k_off = allocator._align(allocator.ptr, 16)
    allocator.ptr = lds_k_off + LDS_K_ELEMS * 2
    lds_lse_off = allocator._align(allocator.ptr, 16)
    allocator.ptr = lds_lse_off + LDS_LSE_ELEMS * 4
    lds_dm_off = allocator._align(allocator.ptr, 16)
    allocator.ptr = lds_dm_off + LDS_DM_ELEMS * 4

    @flyc.kernel(known_block_size=[BLOCK_SIZE, 1, 1])
    def fmha_bwd_dqdkdv_mfma_kernel(  # noqa: F811
        Q: fx.Tensor,
        K: fx.Tensor,
        V: fx.Tensor,
        dO: fx.Tensor,
        dV: fx.Tensor,
        dK: fx.Tensor,
        dQ_f32: fx.Tensor,
        LSE: fx.Tensor,
        D_vec: fx.Tensor,
        seq_M: fx.Int32,
        seq_N: fx.Int32,
        n_heads: fx.Int32,
        n_M_tiles: fx.Int32,
        q_stride_m: fx.Int32,
        kv_stride_n: fx.Int32,
        do_stride_m: fx.Int32,
        seqstart_q: fx.Tensor,
        seqstart_k: fx.Tensor,
        total_m: fx.Int32,
    ):
        bid = fx.block_idx.x
        tid = fx.thread_idx.x

        n_heads_idx = fx.Index(n_heads)
        seq_M_idx = fx.Index(seq_M)
        seq_N_idx = fx.Index(seq_N)
        n_M_tiles_idx = fx.Index(n_M_tiles)
        q_stride_m_idx = fx.Index(q_stride_m)
        kv_stride_n_idx = fx.Index(kv_stride_n)
        do_stride_m_idx = fx.Index(do_stride_m)
        # varlen: LSE/D_vec are packed over the TOTAL (sum-over-batches) M length,
        # NOT seq_M_idx (max_seqlen_q, used only for grid/loop sizing) -- see
        # module docstring. Non-varlen: total_m_idx == seq_M_idx.
        total_m_idx = fx.Index(total_m)
        num_N_tiles = (seq_N_idx + BLOCK_N - 1) // BLOCK_N
        # GQA: grid is regrouped by KV-head (see module docstring) --
        # n_kv_heads_idx blocks per batch, each looping over its heads_per_kv
        # Q-heads. heads_per_kv==1 makes kv_head_idx == head_idx below.
        n_kv_heads_idx = n_heads_idx // heads_per_kv

        bid_idx = fx.Index(bid)
        # M_SPLIT>1: grid gains an innermost M_SPLIT factor (mirrors how n_tile is
        # already innermost relative to bh_idx) -- m_split_idx is this block's
        # slice index within its N-tile's M range (see module docstring).
        if const_expr(M_SPLIT > 1):
            m_split_idx = bid_idx % fx.Index(M_SPLIT)
            n_bh_idx = bid_idx // fx.Index(M_SPLIT)
        else:
            m_split_idx = fx.Index(0)
            n_bh_idx = bid_idx
        n_tile = n_bh_idx % num_N_tiles
        bh_idx = n_bh_idx // num_N_tiles
        batch_idx = bh_idx // n_kv_heads_idx
        kv_head_idx = bh_idx % n_kv_heads_idx

        n_start = n_tile * BLOCK_N

        # varlen: q_start/k_start (packed-row-offset bases) and this_seqlen_q/
        # this_seqlen_k (per-batch REAL length, for masking) come from a runtime
        # seqstart lookup instead of a globally-uniform batch_idx*seq_len_idx --
        # see module docstring. Non-varlen: these reduce to the original formulas.
        if const_expr(varlen):
            from flydsl.expr import buffer_ops as _seq_bops

            seqstart_q_rsrc = _seq_bops.create_buffer_resource(seqstart_q)
            seqstart_k_rsrc = _seq_bops.create_buffer_resource(seqstart_k)

            def _seqstart_load(rsrc, idx):
                return fx.Index(
                    _seq_bops.buffer_load(
                        rsrc, fx.Index(idx), vec_width=1, dtype=fx.Int32
                    )
                )

            q_start = _seqstart_load(seqstart_q_rsrc, batch_idx)
            k_start = _seqstart_load(seqstart_k_rsrc, batch_idx)
            this_seqlen_q = _seqstart_load(seqstart_q_rsrc, batch_idx + 1) - q_start
            this_seqlen_k = _seqstart_load(seqstart_k_rsrc, batch_idx + 1) - k_start
        else:
            q_start = batch_idx * seq_M_idx
            k_start = batch_idx * seq_N_idx
            this_seqlen_q = seq_M_idx
            this_seqlen_k = seq_N_idx

        wave = fx.Index(tid // WARP_SIZE)
        lane = fx.Index(tid % WARP_SIZE)
        lane_mod_32 = fx.Index(lane % 32)
        lane_div_32 = fx.Index(lane // 32)
        wave_n_sub = fx.Index(wave // WAVES_PER_N_GROUP)
        wave_d_group = fx.Index(wave % WAVES_PER_N_GROUP)
        wave_d_sub_base = wave_d_group * D_SUBS_PER_WAVE

        dV_buf = fx.rocdl.make_buffer_tensor(dV)
        dK_buf = fx.rocdl.make_buffer_tensor(dK)
        LSE_buf = fx.rocdl.make_buffer_tensor(LSE)
        Dvec_buf = fx.rocdl.make_buffer_tensor(D_vec)

        copy_f32 = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.Float32)
        store_f32 = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.Float32)

        v8elem_type = Vec.make_type(MFMA_LK, elem_dtype)
        v16f32_type = Vec.make_type(16, fx.Float32)

        base_ptr = allocator.get_base()
        lds_q = SmemPtr(
            base_ptr, lds_q_off, elem_dtype.ir_type, shape=(LDS_Q_ELEMS,)
        ).get()
        lds_do = SmemPtr(
            base_ptr, lds_do_off, elem_dtype.ir_type, shape=(LDS_DO_ELEMS,)
        ).get()
        lds_ds = SmemPtr(
            base_ptr, lds_ds_off, elem_dtype.ir_type, shape=(LDS_DS_ELEMS,)
        ).get()
        lds_p = SmemPtr(
            base_ptr, lds_p_off, elem_dtype.ir_type, shape=(LDS_P_ELEMS,)
        ).get()
        lds_k = SmemPtr(
            base_ptr, lds_k_off, elem_dtype.ir_type, shape=(LDS_K_ELEMS,)
        ).get()
        lds_lse = SmemPtr(
            base_ptr, lds_lse_off, fx.Float32.ir_type, shape=(LDS_LSE_ELEMS,)
        ).get()
        lds_dm = SmemPtr(
            base_ptr, lds_dm_off, fx.Float32.ir_type, shape=(LDS_DM_ELEMS,)
        ).get()

        # Q/dO and K/V are possibly-non-contiguous user inputs (e.g. a packed-qkv
        # unbind view) -- row pitch is q_stride_m_idx/kv_stride_n_idx (in row
        # units), NOT necessarily n_heads_idx. dO can have a DIFFERENT row pitch
        # than Q -- separate helper (mirrors compile_fmha_bwd_dvdk_mfma).
        # GQA: Q-side rows (_q_row/_do_row/_lse_row/_dvec_row) vary across the
        # `heads_per_kv` group (see the q_head_in_group loop below), so head_idx
        # is now an explicit param. K/V (_kv_row) stay indexed by kv_head_idx,
        # fixed for the whole block.
        def _q_row(q_pos, head_idx):
            return fx.Int32((q_start + q_pos) * q_stride_m_idx + head_idx)

        def _do_row(q_pos, head_idx):
            return fx.Int32((q_start + q_pos) * do_stride_m_idx + head_idx)

        def _kv_row(kv_pos):
            return fx.Int32((k_start + kv_pos) * kv_stride_n_idx + kv_head_idx)

        # dV/dK are freshly-allocated contiguous outputs, shaped [B,N,Hkv,D] --
        # row pitch n_kv_heads_idx (NOT kv_stride_n_idx), indexed by kv_head_idx.
        # Packed the SAME way as K/V's own N axis (varlen: k_start-relative).
        def _kv_row_out(kv_pos):
            return fx.Int32((k_start + kv_pos) * n_kv_heads_idx + kv_head_idx)

        # dQ is a freshly-allocated contiguous output -- always row pitch
        # n_heads_idx (NOT q_stride_m_idx), indexed by the per-q-head head_idx.
        def _q_row_out(q_pos, head_idx):
            return fx.Int32((q_start + q_pos) * n_heads_idx + head_idx)

        # LSE layout is [B,H,M] (batch-major, non-varlen) vs packed [1,H,sum_M]
        # under varlen (B collapses to 1). Not unifiable via a single
        # q_start-relative formula since the head-axis stride differs.
        def _lse_row(q_pos, head_idx):
            if const_expr(varlen):
                return fx.Int32(head_idx * total_m_idx + q_start + q_pos)
            return fx.Int32((batch_idx * n_heads_idx + head_idx) * seq_M_idx + q_pos)

        # D_vec is a freshly-allocated contiguous tensor -- always row pitch
        # n_heads_idx regardless of Q's own stride; q_start already unifies both
        # cases the same way as _q_row_out above.
        def _dvec_row(q_pos, head_idx):
            return _q_row_out(q_pos, head_idx)

        from flydsl.expr import buffer_ops as _bops

        q_rsrc = _bops.create_buffer_resource(Q)
        k_rsrc = _bops.create_buffer_resource(K)
        v_rsrc = _bops.create_buffer_resource(V)
        do_rsrc = _bops.create_buffer_resource(dO)

        from flydsl._mlir import ir as _ir_d

        # Raw <4xi32> buffer resource for f32 atomics (raw_buffer_atomic_fadd wants
        # a vector<4xi32> rsrc, NOT the ptr<8> descriptor create_buffer_resource
        # returns). Build the descriptor manually (flash_attn_gfx950.py:741 recipe).
        from flydsl._mlir.dialects import fly as _fly_d, llvm as _llvm_d
        from flydsl.expr.typing import T as _T

        def _make_raw_f32_rsrc(tensor):
            base_ptr = _fly_d.extract_aligned_pointer_as_index(
                _ir_d.Type.parse("!llvm.ptr"), tensor
            )
            base_i64 = _llvm_d.PtrToIntOp(_T.i64, base_ptr).result
            lo = ArithValue(base_i64).trunci(_T.i32)
            hi = ArithValue(ArithValue(base_i64).shrui(fx.Int64(32))).trunci(_T.i32)
            return Vec.from_elements(
                [
                    lo,
                    hi,
                    _bops._create_i32_constant(0xFFFFFFFF),
                    _bops._create_i32_constant(_bops._get_buffer_flags()),
                ],
                fx.Int32,
            ).ir_value()

        dq_rsrc = _make_raw_f32_rsrc(dQ_f32)
        # M_SPLIT>1 only: dV/dK's final store becomes an atomic-add (multiple
        # blocks now contribute a partial sum per N-tile) -- see module docstring.
        if const_expr(M_SPLIT > 1):
            dv_rsrc = _make_raw_f32_rsrc(dV)
            dk_rsrc = _make_raw_f32_rsrc(dK)

        def _load_global_vec_cv(rsrc, row_i32, col_offset_idx):
            flat_elem = fx.Index(row_i32) * fx.Index(D) + col_offset_idx
            return _bops.buffer_load(
                rsrc, flat_elem, vec_width=MFMA_LK, dtype=elem_dtype
            )

        def _load_f32_row(buf, row_idx):
            row_sl = fx.slice(buf, (row_idx, None))
            div_1 = fx.logical_divide(row_sl, fx.make_layout(1, 1))
            r = fx.make_rmem_tensor(1, fx.Float32)
            fx.copy_atom_call(copy_f32, fx.slice(div_1, (None, 0)), r)
            return fx.memref_load(r, 0)

        def _store_f32_row(buf, row_idx, val):
            row_sl = fx.slice(buf, (row_idx, None))
            div_1 = fx.logical_divide(row_sl, fx.make_layout(1, 1))
            r = fx.make_rmem_tensor(1, fx.Float32)
            fx.memref_store(val, r, 0)
            fx.copy_atom_call(store_f32, r, fx.slice(div_1, (None, 0)))

        def _atomic_add_f32(rsrc, flat_elem, val_f32):
            # f32 atomic add into rsrc[flat_elem]; offset is in BYTES.
            rocdl.raw_buffer_atomic_fadd(
                _raw(val_f32),
                rsrc,
                _raw(fx.Int32(fx.Index(flat_elem) * 4)),
                _raw(fx.Int32(0)),
                _raw(fx.Int32(0)),
            )

        def _atomic_add_dq(flat_elem, val_f32):
            _atomic_add_f32(dq_rsrc, flat_elem, val_f32)

        v4elem_type = Vec.make_type(MFMA_LK // 2, elem_dtype)

        def _lds_load_pack_a(lds_arr, base_row_in_tile, k_step):
            lds_row = fx.Index(base_row_in_tile) + lane_mod_32
            lds_col_lo = fx.Index(k_step * MFMA_K) + lane_div_32 * MFMA_LK
            lds_col_hi = lds_col_lo + fx.Index(MFMA_LK // 2)
            lo = Vec.load(v4elem_type, lds_arr, [lds_row * LDS_Q_STRIDE + lds_col_lo])
            hi = Vec.load(v4elem_type, lds_arr, [lds_row * LDS_Q_STRIDE + lds_col_hi])
            return Vec(lo).shuffle(Vec(hi), list(range(MFMA_LK))).ir_value()

        def mfma(a_pack, b_pack, c_acc):
            if const_expr(dtype_str == "bf16"):
                if const_expr(USE_K16):
                    return rocdl.mfma_f32_32x32x16_bf16(
                        v16f32_type, [a_pack, b_pack, c_acc]
                    )
                a_pack = Vec(a_pack).bitcast(fx.Int16)
                b_pack = Vec(b_pack).bitcast(fx.Int16)
                return rocdl.mfma_f32_32x32x8bf16_1k(
                    v16f32_type, [a_pack, b_pack, c_acc]
                )
            if const_expr(USE_K16):
                return rocdl.mfma_f32_32x32x16_f16(v16f32_type, [a_pack, b_pack, c_acc])
            return rocdl.mfma_f32_32x32x8f16(v16f32_type, [a_pack, b_pack, c_acc])

        # ---- Pre-load K and V packs for this wave's N sub-tile (QK/dP layout) ----
        n_global_wave_base = n_start + wave_n_sub * 32
        n_row_abs_kv = n_global_wave_base + lane_mod_32
        n_valid_kv = n_row_abs_kv < this_seqlen_k
        n_safe_kv = n_valid_kv.select(n_row_abs_kv, this_seqlen_k - fx.Index(1))
        kv_row_g_pre = _kv_row(n_safe_kv)

        k_packs = []
        v_packs = []
        for ks in range_constexpr(K_STEPS):
            col_off = fx.Index(ks * MFMA_K) + lane_div_32 * MFMA_LK
            k_packs.append(_load_global_vec_cv(k_rsrc, kv_row_g_pre, col_off))
            v_packs.append(_load_global_vec_cv(v_rsrc, kv_row_g_pre, col_off))

        # ---- Cooperative LDS load: K tile [BLOCK_N, D] (once per block, for dQ) ----
        VEC_COLS_KV = D // MFMA_LK
        ROWS_PER_WAVE_KV = BLOCK_N // NUM_WAVES
        N_ITEMS_KV = ROWS_PER_WAVE_KV * VEC_COLS_KV
        ITEMS_PER_LANE_KV = N_ITEMS_KV // WARP_SIZE
        if ITEMS_PER_LANE_KV < 1:
            ITEMS_PER_LANE_KV = 1
        for it in range_constexpr(ITEMS_PER_LANE_KV):
            item = lane + fx.Index(it * WARP_SIZE)
            item_ok = item < fx.Index(N_ITEMS_KV)
            item_s = item_ok.select(item, fx.Index(0))
            row_off_i = item_s // fx.Index(VEC_COLS_KV)
            cv_i = item_s % fx.Index(VEC_COLS_KV)
            row_in_tile = wave * ROWS_PER_WAVE_KV + row_off_i
            n_global_ld = n_start + row_in_tile
            n_valid_ld = n_global_ld < this_seqlen_k
            n_safe_ld = n_valid_ld.select(n_global_ld, this_seqlen_k - fx.Index(1))
            kv_row_g = _kv_row(n_safe_ld)
            col_off_ld = cv_i * fx.Index(MFMA_LK)
            k_vec = _load_global_vec_cv(k_rsrc, kv_row_g, col_off_ld)
            Vec(k_vec).store(lds_k, [row_in_tile * LDS_K_STRIDE + col_off_ld])

        # One dk/dv accumulator PER D-subtile this wave sequentially owns (mirrors
        # compile_fmha_bwd_dvdk_mfma's generalization; D_SUBS_PER_WAVE==1 for D==64,
        # unchanged from before).
        dk_inits = [Vec.filled(16, 0.0, fx.Float32) for _ in range(D_SUBS_PER_WAVE)]
        dv_inits = [Vec.filled(16, 0.0, fx.Float32) for _ in range(D_SUBS_PER_WAVE)]
        dummy_val = fx.Float32(0.0)
        init_st = dk_inits + dv_inits + [dummy_val]

        # ---- Software-pipelined Q/dO prefetch: the global load for tile
        # (m_tile+1) is issued right after the current tile's barrier, so its
        # VMEM latency overlaps with the current tile's GEMM1/epilogue/GEMM2/dQ
        # compute instead of stalling at the top of the NEXT iteration.
        # Prefetched registers are threaded through the m_tile scf.for loop as
        # extra iter_args and stored to LDS at the START of the consuming iteration.
        VEC_COLS = D // MFMA_LK
        ROWS_PER_WAVE_LD = BLOCK_M // NUM_WAVES
        N_ITEMS_LD = ROWS_PER_WAVE_LD * VEC_COLS
        ITEMS_PER_LANE = N_ITEMS_LD // WARP_SIZE

        def _load_qdo_item(head_idx_p, m_tile_idx, it):
            m_start_p = m_tile_idx * BLOCK_M
            item = lane + fx.Index(it * WARP_SIZE)
            row_off_i = item // fx.Index(VEC_COLS)
            cv_i = item % fx.Index(VEC_COLS)
            row_in_tile = wave * ROWS_PER_WAVE_LD + row_off_i
            m_global_ld = m_start_p + row_in_tile
            m_valid_ld = m_global_ld < this_seqlen_q
            m_safe_ld = m_valid_ld.select(m_global_ld, this_seqlen_q - fx.Index(1))
            q_row_g = _q_row(m_safe_ld, head_idx_p)
            do_row_g = _do_row(m_safe_ld, head_idx_p)
            col_off_ld = cv_i * fx.Index(MFMA_LK)
            return (
                _load_global_vec_cv(q_rsrc, q_row_g, col_off_ld),
                _load_global_vec_cv(do_rsrc, do_row_g, col_off_ld),
            )

        def _load_qdo_regs(head_idx_p, m_tile_idx):
            regs_q, regs_do = [], []
            for it in range_constexpr(ITEMS_PER_LANE):
                q_v, do_v = _load_qdo_item(head_idx_p, m_tile_idx, it)
                regs_q.append(q_v)
                regs_do.append(do_v)
            return regs_q, regs_do

        def _store_qdo_regs_to_lds(regs_q, regs_do):
            for it in range_constexpr(ITEMS_PER_LANE):
                item = lane + fx.Index(it * WARP_SIZE)
                row_off_i = item // fx.Index(VEC_COLS)
                cv_i = item % fx.Index(VEC_COLS)
                row_in_tile = wave * ROWS_PER_WAVE_LD + row_off_i
                col_off_ld = cv_i * fx.Index(MFMA_LK)
                lds_base = row_in_tile * LDS_Q_STRIDE + col_off_ld
                Vec(regs_q[it]).store(lds_q, [lds_base])
                Vec(regs_do[it]).store(lds_do, [lds_base])

        # GQA: dk_accs/dv_accs accumulate across ALL heads_per_kv Q-heads sharing
        # this block's kv_head_idx (see module docstring) -- reset ONCE (init_st)
        # before the group, not per q-head. The m_tile scf.for loop is re-entered
        # once per q_head_in_group, threading the running accumulator through as
        # its `init`. heads_per_kv==1 (non-GQA): runs once, degenerates to the
        # original per-head-unique behavior.
        loop_results = init_st
        for q_head_in_group in range_constexpr(heads_per_kv):
            head_idx = kv_head_idx * heads_per_kv + q_head_in_group

            # Causal skip-ahead: for top-left causal masking, an N-tile at
            # n_tile can only be attended to by m_tile >= (n_tile*BLOCK_N)//BLOCK_M.
            # Starting the M-loop there skips fully-masked M-tiles, roughly
            # halving both wasted MFMA work and redundant Q/dO HBM reloads.
            m_tile_start_full = (
                (n_tile * BLOCK_N) // BLOCK_M if const_expr(causal) else fx.Index(0)
            )

            # M_SPLIT>1: divide this N-tile's (post-causal-skip-ahead) M range
            # [m_tile_start_full, n_M_tiles_idx) into M_SPLIT roughly-equal chunks,
            # computed AT RUNTIME per n_tile (m_tile_start_full varies by n_tile
            # under causal, so a fixed absolute chunk width would degenerate for
            # late n_tiles' already-small M ranges -- see module docstring).
            # m_split_idx==M_SPLIT-1 gets any remainder (chunk_end clamped to
            # n_M_tiles_idx). Blocks whose computed range is empty (chunk_end <=
            # m_tile_start) fall through to an empty scf.for -- a genuine no-op,
            # same as any other empty-range loop (mirrors how the existing loop
            # already tolerates m_tile_start reaching n_M_tiles_idx).
            if const_expr(M_SPLIT > 1):
                m_range_total = n_M_tiles_idx - m_tile_start_full
                m_chunk = (m_range_total + fx.Index(M_SPLIT) - fx.Index(1)) // fx.Index(
                    M_SPLIT
                )
                m_tile_start = m_tile_start_full + m_split_idx * m_chunk
                m_tile_end_raw = m_tile_start + m_chunk
                m_tile_end = (m_tile_end_raw < n_M_tiles_idx).select(
                    m_tile_end_raw, n_M_tiles_idx
                )
                m_tile_start = (m_tile_start < n_M_tiles_idx).select(
                    m_tile_start, n_M_tiles_idx
                )
            else:
                m_tile_start = m_tile_start_full
                m_tile_end = n_M_tiles_idx

            # Prologue: prefetch tile m_tile_start's Q/dO for THIS head (redone
            # per q_head_in_group since head_idx changes the Q/dO row address).
            # NOTE: if m_tile_start == m_tile_end (empty range, M_SPLIT>1 only),
            # this prefetch reads a tile that is never consumed by the loop below
            # (which won't execute) -- harmless (same over-fetch tolerance as the
            # existing last-M-tile prefetch, clamped in-bounds by _load_qdo_item).
            prologue_q, prologue_do = _load_qdo_regs(head_idx, m_tile_start)
            entry_state = (
                list(loop_results[0 : 2 * D_SUBS_PER_WAVE + 1])
                + prologue_q
                + prologue_do
            )

            for m_tile, iter_args in range(
                m_tile_start, m_tile_end, fx.Index(1), init=entry_state
            ):
                dk_accs = list(iter_args[0:D_SUBS_PER_WAVE])
                dv_accs = list(iter_args[D_SUBS_PER_WAVE : 2 * D_SUBS_PER_WAVE])
                cur_q = list(
                    iter_args[
                        2 * D_SUBS_PER_WAVE + 1 : 2 * D_SUBS_PER_WAVE
                        + 1
                        + ITEMS_PER_LANE
                    ]
                )
                cur_do = list(
                    iter_args[
                        2 * D_SUBS_PER_WAVE + 1 + ITEMS_PER_LANE : 2 * D_SUBS_PER_WAVE
                        + 1
                        + 2 * ITEMS_PER_LANE
                    ]
                )
                m_start = m_tile * BLOCK_M

                # ---- Store this iteration's already-prefetched Q/dO into LDS ----
                _store_qdo_regs_to_lds(cur_q, cur_do)

                # ---- Cooperative LSE + D_vec tile stage ----
                tid_idx = fx.Index(tid)
                if tid_idx < fx.Index(BLOCK_M):
                    m_g_ls = m_start + tid_idx
                    m_ok_ls = m_g_ls < this_seqlen_q
                    m_sf_ls = m_ok_ls.select(m_g_ls, this_seqlen_q - fx.Index(1))
                    lse_g = _load_f32_row(LSE_buf, _lse_row(m_sf_ls, head_idx))
                    dm_g = _load_f32_row(Dvec_buf, _dvec_row(m_sf_ls, head_idx))
                    Vec.from_elements([lse_g], fx.Float32).store(lds_lse, [tid_idx])
                    Vec.from_elements([dm_g], fx.Float32).store(lds_dm, [tid_idx])

                gpu.barrier()

                # ---- Issue next tile's Q/dO global loads INTERLEAVED with GEMM1's
                # MFMA stream: spread ITEMS_PER_LANE loads one-per-GEMM1-step,
                # each wrapped in sched_group_barrier(MFMA,2)+sched_group_barrier(VMEM,1)
                # pairs so the scheduler keeps loads adjacent to (and overlapping)
                # each step's 2 MFMAs (s_acc, dp_acc). Safe to over-fetch on the
                # final m_tile: the per-lane bounds check clamps to this_seqlen_q-1.
                _MFMA_MASK = 0x008
                _VMEM_MASK = 0x020
                next_q = [None] * ITEMS_PER_LANE
                next_do = [None] * ITEMS_PER_LANE
                _next_load_it = 0  # plain Python int; range_constexpr is a Python-level
                # unroll, so this is safe to mutate directly (a nested
                # closure over a mutable cell does NOT survive FlyDSL's
                # kernel-body tracing/re-execution).

                log2e_scale_cst = fx.Float32(_LOG2E * scale)
                scale_cst = fx.Float32(scale)
                M_SUBTILES = BLOCK_M // 32
                for m_sub in range_constexpr(M_SUBTILES):
                    # ---- GEMM1a: S = Q @ K^T ; GEMM1b: dP = dO @ V^T ----
                    s_acc = Vec.filled(16, 0.0, fx.Float32)
                    dp_acc = Vec.filled(16, 0.0, fx.Float32)
                    for ks in range_constexpr(K_STEPS):
                        q_pack = _lds_load_pack_a(lds_q, m_sub * 32, ks)
                        do_pack = _lds_load_pack_a(lds_do, m_sub * 32, ks)
                        s_acc = mfma(q_pack, k_packs[ks], s_acc)
                        dp_acc = mfma(do_pack, v_packs[ks], dp_acc)
                        rocdl.sched_group_barrier(_MFMA_MASK, 2, 0)
                        if const_expr(_next_load_it < ITEMS_PER_LANE):
                            q_v, do_v = _load_qdo_item(
                                head_idx, m_tile + fx.Index(1), _next_load_it
                            )
                            next_q[_next_load_it] = q_v
                            next_do[_next_load_it] = do_v
                            _next_load_it += 1
                        rocdl.sched_group_barrier(_VMEM_MASK, 1, 0)

                    # Any remaining prefetch items for this m_sub (ITEMS_PER_LANE >
                    # M_SUBTILES*K_STEPS, e.g. wide D) are issued right after
                    # GEMM1 -- still overlaps this m_sub's epilogue/GEMM2/dQ below.
                    # `const_expr()` marks this Python-int comparison as a
                    # compile-time constant so the AST rewriter unrolls it instead
                    # of lowering to a dynamic scf.if (which would require next_q/
                    # next_do -- plain Python lists -- to be MLIR-value loop state).
                    if const_expr(m_sub == M_SUBTILES - 1):
                        for _pad_it in range_constexpr(ITEMS_PER_LANE):
                            if const_expr(_pad_it >= _next_load_it):
                                q_v, do_v = _load_qdo_item(
                                    head_idx, m_tile + fx.Index(1), _pad_it
                                )
                                next_q[_pad_it] = q_v
                                next_do[_pad_it] = do_v

                    # ---- P (for dV) and dS' (for dK/dQ, scale deferred to store/atomic-add),
                    # both stored TRANSPOSED [n,m] ----
                    # m_within = lane_div_32*4 + (r//4)*8 + r%4: within each group of 4
                    # consecutive r (same r//4), m_within increments by 1 -- 4 CONTIGUOUS
                    # lds_lse/lds_dm addresses. Load each group as one v4 (instead of 4
                    # scalar ds_read) and index within the group by r%4; cuts LDS
                    # instruction count for this epilogue 4x (32 scalar reads -> 8 v4 reads).
                    v4f32_type = Vec.make_type(4, fx.Float32)
                    n_within = lane_mod_32
                    n_row_abs = n_within + wave_n_sub * 32 + n_start
                    n_ok = n_row_abs < this_seqlen_k
                    for r_group in range_constexpr(4):
                        m_group_base = lane_div_32 * 4 + r_group * 8 + (m_sub * 32)
                        lse_grp = Vec.load(v4f32_type, lds_lse, [m_group_base])
                        dm_grp = Vec.load(v4f32_type, lds_dm, [m_group_base])
                        for r_mod in range_constexpr(4):
                            r = r_group * 4 + r_mod
                            m_within = lane_div_32 * 4 + r_group * 8 + r_mod
                            m_row_abs = m_within + (m_sub * 32) + m_start
                            m_valid = m_row_abs < this_seqlen_q
                            lse_val = Vec(lse_grp)[r_mod]
                            dm_val = Vec(dm_grp)[r_mod]
                            s_val = Vec(s_acc)[r]
                            dp_val = Vec(dp_acc)[r]
                            valid_mn = m_valid & n_ok
                            if const_expr(causal):
                                valid_mn = valid_mn & (n_row_abs <= m_row_abs)
                            neg_log2e_lse = fx.Float32(
                                arith.mulf(
                                    _raw(lse_val),
                                    _raw(fx.Float32(-_LOG2E)),
                                    fastmath=fm,
                                )
                            )
                            p_val = _softmax_p(
                                s_val, neg_log2e_lse, log2e_scale_cst, valid_mn, fm
                            )
                            ds_val = _grad_ds_unscaled(
                                p_val, dp_val, dm_val, valid_mn, fm
                            )
                            m_local = m_within + (m_sub * 32)
                            n_local = n_within + wave_n_sub * 32
                            Vec.from_elements([p_val], fx.Float32).to(elem_dtype).store(
                                lds_p, [n_local * LDS_MPAD + m_local]
                            )
                            Vec.from_elements([ds_val], fx.Float32).to(
                                elem_dtype
                            ).store(lds_ds, [n_local * LDS_MPAD + m_local])

                    gpu.barrier()

                    # ---- dV += P^T @ dO ; dK += dS^T @ Q ----
                    # P/dS (the A-operand) do NOT depend on d, so they're loaded ONCE per ks
                    # and reused across every D-subtile this wave sequentially owns
                    # (D_SUBS_PER_WAVE loop below) -- only the B-operand (dO/Q at a given d)
                    # changes per subtile. Mirrors compile_fmha_bwd_dvdk_mfma's identical
                    # D-generalization and use_trload branch verbatim.
                    MFMA_KS = 32 // MFMA_K
                    for ks in range_constexpr(MFMA_KS):
                        n_local = lane_mod_32 + wave_n_sub * 32
                        if use_trload:
                            # A-operand (P^T/dS^T): must hold the SAME m as the tr B-operand at
                            # each hardware slot (independent of which D-subtile is being read).
                            # B (tr) gives m = (m_sub*32+ks*16) + lane_div_32*4 + P8[e],
                            # P8={0,1,2,3,8,9,10,11}. So load P^T[n, base_a+{0,1,2,3}] ++
                            # P^T[n, base_a+{8,9,10,11}].
                            base_a = lane_div_32 * 4 + (m_sub * 32 + ks * MFMA_K)
                            p_lo = Vec.load(
                                v4elem_type, lds_p, [n_local * LDS_MPAD + base_a]
                            )
                            p_hi = Vec.load(
                                v4elem_type, lds_p, [n_local * LDS_MPAD + base_a + 8]
                            )
                            p_pack = (
                                Vec(p_lo)
                                .shuffle(Vec(p_hi), [0, 1, 2, 3, 4, 5, 6, 7])
                                .ir_value()
                            )
                            ds_lo = Vec.load(
                                v4elem_type, lds_ds, [n_local * LDS_MPAD + base_a]
                            )
                            ds_hi = Vec.load(
                                v4elem_type, lds_ds, [n_local * LDS_MPAD + base_a + 8]
                            )
                            ds_pack = (
                                Vec(ds_lo)
                                .shuffle(Vec(ds_hi), [0, 1, 2, 3, 4, 5, 6, 7])
                                .ir_value()
                            )
                            tr_k_group = (
                                lane_mod_32 % 16
                            ) // 4  # lane%16 //4 within 32-lane half
                            tr_col_sub = lane % 4
                            tr_col_half = lane_mod_32 // 16
                            m_base = (
                                m_sub * 32 + ks * MFMA_K + lane_div_32 * 4 + tr_k_group
                            )
                            for d_iter in range_constexpr(D_SUBS_PER_WAVE):
                                wave_d_sub_i_raw = wave_d_sub_base + d_iter
                                # D=32/96 (D_TOTAL_SUBS not evenly divisible by WAVES_PER_N_GROUP):
                                # clamp out-of-range D-subtiles to subtile 0 (in-bounds); result
                                # discarded at the store site below via d_in_range.
                                d_in_range = wave_d_sub_i_raw < fx.Index(D_TOTAL_SUBS)
                                wave_d_sub_i = d_in_range.select(
                                    wave_d_sub_i_raw, fx.Index(0)
                                )
                                # B-operand (dO/Q) via HW transpose. tr yields, per lane, contract
                                # m = P8 = {0,1,2,3,8,9,10,11} + lane_div_32*4 (relative to m_row
                                # base), free d = lane%32. Read row-major [m,d] with EVEN LDS_Q_STRIDE.
                                d_col = (
                                    wave_d_sub_i * 32
                                    + tr_col_half * 16
                                    + tr_col_sub * 4
                                )
                                lo = m_base * LDS_Q_STRIDE + d_col
                                hi = lo + 8 * LDS_Q_STRIDE
                                do_a = _ds_read_tr_v4(v4elem_type, lo, lds_do_off)
                                do_b = _ds_read_tr_v4(v4elem_type, hi, lds_do_off)
                                do_pack = (
                                    Vec(do_a)
                                    .shuffle(Vec(do_b), [0, 1, 2, 3, 4, 5, 6, 7])
                                    .ir_value()
                                )
                                q_a = _ds_read_tr_v4(v4elem_type, lo, lds_q_off)
                                q_b = _ds_read_tr_v4(v4elem_type, hi, lds_q_off)
                                q_pack = (
                                    Vec(q_a)
                                    .shuffle(Vec(q_b), [0, 1, 2, 3, 4, 5, 6, 7])
                                    .ir_value()
                                )
                                dv_accs[d_iter] = mfma(p_pack, do_pack, dv_accs[d_iter])
                                dk_accs[d_iter] = mfma(ds_pack, q_pack, dk_accs[d_iter])
                        else:
                            base_m = lane_div_32 * MFMA_LK + (m_sub * 32 + ks * MFMA_K)
                            p_pack = Vec.load(
                                v8elem_type, lds_p, [n_local * LDS_MPAD + base_m]
                            ).ir_value()
                            ds_pack = Vec.load(
                                v8elem_type, lds_ds, [n_local * LDS_MPAD + base_m]
                            ).ir_value()
                            for d_iter in range_constexpr(D_SUBS_PER_WAVE):
                                wave_d_sub_i_raw = wave_d_sub_base + d_iter
                                # D=32/96 (D_TOTAL_SUBS not evenly divisible by WAVES_PER_N_GROUP):
                                # clamp out-of-range D-subtiles to subtile 0 to keep the LDS address
                                # in-bounds; the store site below discards this iteration's result.
                                d_in_range = wave_d_sub_i_raw < fx.Index(D_TOTAL_SUBS)
                                wave_d_sub_i = d_in_range.select(
                                    wave_d_sub_i_raw, fx.Index(0)
                                )
                                d_local = lane_mod_32 + wave_d_sub_i * 32
                                do_r = fx.make_rmem_tensor(MFMA_LK, elem_dtype)
                                q_r = fx.make_rmem_tensor(MFMA_LK, elem_dtype)
                                for e in range_constexpr(MFMA_LK):
                                    m_local = lane_div_32 * MFMA_LK + (
                                        m_sub * 32 + ks * MFMA_K + e
                                    )
                                    do_sc = Vec.load(
                                        Vec.make_type(1, elem_dtype),
                                        lds_do,
                                        [m_local * LDS_Q_STRIDE + d_local],
                                    )[0]
                                    q_sc = Vec.load(
                                        Vec.make_type(1, elem_dtype),
                                        lds_q,
                                        [m_local * LDS_Q_STRIDE + d_local],
                                    )[0]
                                    fx.memref_store(do_sc, do_r, e)
                                    fx.memref_store(q_sc, q_r, e)
                                dv_accs[d_iter] = mfma(
                                    p_pack, fx.memref_load_vec(do_r), dv_accs[d_iter]
                                )
                                dk_accs[d_iter] = mfma(
                                    ds_pack, fx.memref_load_vec(q_r), dk_accs[d_iter]
                                )

                    # ---- dQ = dS @ K (contract over the FULL n range, D-split across
                    # all waves): every wave owns a D-slice and sweeps the entire
                    # BLOCK_N range internally, producing one COMPLETE dq_acc with no
                    # cross-wave reduction. D=64: 2 of 4 waves are idle (out-of-range
                    # D-subtile, discarded via d_in_range below).
                    # A=dS[m,n]: free=m=lane%32, k=n. dS in lds_ds as [n,m] (n_local*MPAD+m).
                    # B=K[n,d] : free=d=lane%32, k=n. K in lds_k as [n,d] (n_local*LDS_K_STRIDE+d).
                    # Both already resident in LDS for the whole BLOCK_N range.
                    #
                    # use_trload=True: BOTH operands have their contract dim (n) as the LDS
                    # OUTER/strided axis (unlike GEMM2, where only the B-operand did) -- dS is
                    # [n,m] and K is [n,d], n outer in both -- so BOTH get the identical
                    # ds_read_tr16_b64 recipe GEMM2 uses for its B-operand (dvdk/dqdkdv's
                    # dO/Q), sharing one hardware-transposed contract-row base (n_base) so the
                    # two operands present the SAME per-lane n ordering by construction (no
                    # P8-pattern replication needed on either side, unlike GEMM2's direct-load
                    # A-operand which had to manually match the tr B-operand's m ordering).
                    for d_iter in range_constexpr(DQ_D_SUBS_PER_WAVE):
                        wave_d_sub_i_raw = wave * DQ_D_SUBS_PER_WAVE + d_iter
                        d_in_range = wave_d_sub_i_raw < fx.Index(D_TOTAL_SUBS)
                        wave_d_sub_i = d_in_range.select(wave_d_sub_i_raw, fx.Index(0))
                        dq_acc = Vec.filled(16, 0.0, fx.Float32)
                        for n_sub_iter in range_constexpr(WAVE_N_TILES):
                            for ks in range_constexpr(MFMA_KS):
                                if use_trload:
                                    tr_k_group = (lane_mod_32 % 16) // 4
                                    tr_col_sub = lane % 4
                                    tr_col_half = lane_mod_32 // 16
                                    n_base = (
                                        n_sub_iter * 32
                                        + ks * MFMA_K
                                        + lane_div_32 * 4
                                        + tr_k_group
                                    )
                                    # B-operand (K): free=d.
                                    d_col = (
                                        wave_d_sub_i * 32
                                        + tr_col_half * 16
                                        + tr_col_sub * 4
                                    )
                                    lo_k = n_base * LDS_K_STRIDE + d_col
                                    hi_k = lo_k + 8 * LDS_K_STRIDE
                                    k_a = _ds_read_tr_v4(v4elem_type, lo_k, lds_k_off)
                                    k_b = _ds_read_tr_v4(v4elem_type, hi_k, lds_k_off)
                                    k_pack = (
                                        Vec(k_a)
                                        .shuffle(Vec(k_b), [0, 1, 2, 3, 4, 5, 6, 7])
                                        .ir_value()
                                    )
                                    # A-operand (dS): free=m.
                                    m_col = (
                                        m_sub * 32 + tr_col_half * 16 + tr_col_sub * 4
                                    )
                                    lo_ds = n_base * LDS_MPAD + m_col
                                    hi_ds = lo_ds + 8 * LDS_MPAD
                                    ds_a = _ds_read_tr_v4(
                                        v4elem_type, lo_ds, lds_ds_off
                                    )
                                    ds_b = _ds_read_tr_v4(
                                        v4elem_type, hi_ds, lds_ds_off
                                    )
                                    ds_pack = (
                                        Vec(ds_a)
                                        .shuffle(Vec(ds_b), [0, 1, 2, 3, 4, 5, 6, 7])
                                        .ir_value()
                                    )
                                    dq_acc = mfma(ds_pack, k_pack, dq_acc)
                                else:
                                    ds_r = fx.make_rmem_tensor(MFMA_LK, elem_dtype)
                                    k_r = fx.make_rmem_tensor(MFMA_LK, elem_dtype)
                                    m_free = lane_mod_32 + (m_sub * 32)
                                    d_free = lane_mod_32 + wave_d_sub_i * 32
                                    for e in range_constexpr(MFMA_LK):
                                        n_local = lane_div_32 * MFMA_LK + (
                                            n_sub_iter * 32 + ks * MFMA_K + e
                                        )
                                        ds_sc = Vec.load(
                                            Vec.make_type(1, elem_dtype),
                                            lds_ds,
                                            [n_local * LDS_MPAD + m_free],
                                        )[0]
                                        k_sc = Vec.load(
                                            Vec.make_type(1, elem_dtype),
                                            lds_k,
                                            [n_local * LDS_K_STRIDE + d_free],
                                        )[0]
                                        fx.memref_store(ds_sc, ds_r, e)
                                        fx.memref_store(k_sc, k_r, e)
                                    dq_acc = mfma(
                                        fx.memref_load_vec(ds_r),
                                        fx.memref_load_vec(k_r),
                                        dq_acc,
                                    )

                        # ---- dQ store (no combine needed: dq_acc is already a complete
                        # sum over the full n range, computed entirely by this one wave).
                        # D=32/96 or NUM_WAVES>D_TOTAL_SUBS: skip the store for out-of-range
                        # D-subtiles (the dq_acc computed above is garbage but never observed).
                        d_col_abs_dq = lane_mod_32 + wave_d_sub_i * 32
                        if not _ablate_atomic:
                            for r in range_constexpr(16):
                                m_within = lane_div_32 * 4 + ((r // 4) * 8 + (r % 4))
                                m_row_abs = m_within + (m_sub * 32) + m_start
                                m_ok = (m_row_abs < this_seqlen_q) & d_in_range
                                if m_ok:
                                    q_row_g = _q_row_out(m_row_abs, head_idx)
                                    flat_dq = fx.Int32(
                                        fx.Index(q_row_g) * fx.Index(D) + d_col_abs_dq
                                    )
                                    dq_scaled = fx.Float32(
                                        arith.mulf(
                                            _raw(Vec(dq_acc)[r]),
                                            _raw(scale_cst),
                                            fastmath=fm,
                                        )
                                    )
                                    _atomic_add_dq(flat_dq, dq_scaled)

                    gpu.barrier()

                iter_args = yield dk_accs + dv_accs + [dummy_val] + next_q + next_do

            loop_results = iter_args[0 : 2 * D_SUBS_PER_WAVE + 1]

        # ---- Store dV and dK ----  (same output decode: M=n_key varies with r, N=d fixed)
        # Loop-invariant `scale` (deferred from the per-element dS' epilogue above) is
        # applied once here: MFMA is linear in its A-operand, so scale*(dS'@Q) == (scale*dS')@Q.
        store_scale_cst = fx.Float32(scale)
        dk_finals = loop_results[0:D_SUBS_PER_WAVE]
        dv_finals = loop_results[D_SUBS_PER_WAVE : 2 * D_SUBS_PER_WAVE]
        for r in range_constexpr(16):
            n_within = lane_div_32 * 4 + ((r // 4) * 8 + (r % 4))
            n_row_abs = n_within + wave_n_sub * 32 + n_start
            n_ok = n_row_abs < this_seqlen_k
            n_safe = n_ok.select(n_row_abs, this_seqlen_k - fx.Index(1))
            kv_row_g = _kv_row_out(n_safe)
            for d_iter in range_constexpr(D_SUBS_PER_WAVE):
                wave_d_sub_i = wave_d_sub_base + d_iter
                d_col_abs = lane_mod_32 + wave_d_sub_i * 32
                # D=32/96: this wave's nominal D-subtile range can run past D (see
                # D_SUBS_PER_WAVE comment above) -- skip the store for out-of-range columns.
                d_ok = d_col_abs < fx.Index(D)
                flat_col = fx.Int32(fx.Index(kv_row_g) * fx.Index(D) + d_col_abs)
                if n_ok & d_ok:
                    dk_scaled = fx.Float32(
                        arith.mulf(
                            _raw(Vec(dk_finals[d_iter])[r]),
                            _raw(store_scale_cst),
                            fastmath=fm,
                        )
                    )
                    if const_expr(M_SPLIT > 1):
                        # Multiple blocks (one per m_split_idx) contribute a
                        # PARTIAL dV/dK sum for this N-tile -- must combine via
                        # atomic-add instead of a single plain store.
                        _atomic_add_f32(dv_rsrc, flat_col, Vec(dv_finals[d_iter])[r])
                        _atomic_add_f32(dk_rsrc, flat_col, dk_scaled)
                    else:
                        _store_f32_row(dV_buf, flat_col, Vec(dv_finals[d_iter])[r])
                        _store_f32_row(dK_buf, flat_col, dk_scaled)

    @flyc.jit
    def launch_fn(
        Q: fx.Tensor,
        K: fx.Tensor,
        V: fx.Tensor,
        dO: fx.Tensor,
        dV: fx.Tensor,
        dK: fx.Tensor,
        dQ_f32: fx.Tensor,
        LSE: fx.Tensor,
        D_vec: fx.Tensor,
        B: fx.Int32,
        M: fx.Int32,
        N: fx.Int32,
        H: fx.Int32,
        n_M_tiles: fx.Int32,
        q_stride_m: fx.Int32,
        kv_stride_n: fx.Int32,
        do_stride_m: fx.Int32,
        seqstart_q: fx.Tensor,
        seqstart_k: fx.Tensor,
        total_m: fx.Int32,
        stream: fx.Stream,
    ):
        from flydsl._mlir import ir
        from flydsl.compiler.kernel_function import CompilationContext

        allocator.finalized = False
        _ctx = CompilationContext.get_current()
        with ir.InsertionPoint(_ctx.gpu_module_body):
            allocator.finalize()

        num_N_tiles = (fx.Index(N) + BLOCK_N - 1) // BLOCK_N
        # GQA: grid is regrouped by KV-head (H // heads_per_kv blocks per batch,
        # not H) -- see module docstring. varlen: B is already collapsed to 1 by
        # the caller (flydsl.py), so this is unchanged either way.
        n_kv_heads_idx = fx.Index(H) // heads_per_kv
        # M_SPLIT>1: grid gains an innermost M_SPLIT factor (see module docstring
        # and the kernel body's bid_idx decomposition).
        grid_x = fx.Int32(fx.Index(B) * n_kv_heads_idx * num_N_tiles * M_SPLIT)
        fmha_bwd_dqdkdv_mfma_kernel(
            Q,
            K,
            V,
            dO,
            dV,
            dK,
            dQ_f32,
            LSE,
            D_vec,
            M,
            N,
            H,
            n_M_tiles,
            q_stride_m,
            kv_stride_n,
            do_stride_m,
            seqstart_q,
            seqstart_k,
            total_m,
        ).launch(
            grid=(grid_x, 1, 1),
            block=(BLOCK_SIZE, 1, 1),
            stream=stream,
        )

    return launch_fn
