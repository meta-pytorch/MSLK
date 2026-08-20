# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""FlyDSL decode (gfx950) — head-packed MFMA + double-buffered wide V load.

Packs up to 16 query heads sharing a KV head onto the MFMA M-dim. One CTA = one
warp = one KV head's whole GQA group.
  QK: A=Q[head=M(16), k=head_dim], B=K[tok=N(16), k=head_dim] -> C[head, tok].
  Softmax: per-head max/sum reduce over low 4 lane bits via dpp_xor(1,2,4,8).
  PV: A=P[head=M, tok=K(32)], B=V[tok=K, d=N(16)] -> C[head, d].
MFMA reg<->matrix layout: 16x16x32 lane l reg e -> C[m=(l//16)*4+e, n=l%16].

V staged into LDS in [dpass][tok][16] transpose layout via wide vec8 loads, read
back with ds_read_tr16_b64. V HBM loads issued EARLY so latency overlaps QK+softmax
(intra-tile software pipeline).

gfx950 only; GQA ratio in [1,16] (else falls back to pa_decode_generic). Split-K
via pa_decode_reduce.
"""

from __future__ import annotations

import functools
from typing import Any

import flydsl.compiler as flyc  # pyre-ignore[21]
import flydsl.expr as fx  # pyre-ignore[21]
import torch
from flydsl._mlir.dialects import llvm as _llvm  # pyre-ignore[21]
from flydsl.expr import (  # pyre-ignore[21]
    arith,
    buffer_ops,
    const_expr,
    gpu,
    range_constexpr,
    rocdl,
    vector,
)
from flydsl.expr.typing import T  # pyre-ignore[21]
from flydsl.runtime.device import get_rocm_arch  # pyre-ignore[21]
from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr  # pyre-ignore[21]

from .pa_decode_reduce import pa_decode_reduce
from .utils import (
    dpp_xor_f32,
    exp2_f32 as _exp2_fast,
    maxnumf as _mxf,
    rcp_f32,
    smem_bytes,
    WARP_SIZE,
)

MFMA_M = 16  # heads packed on the QK MFMA M-axis
MFMA_N = 16  # tokens per QK sub-tile (N-axis)
MFMA_K_QK = 32  # QK MFMA K-dim (head-dim elements per call)
TILE_N = 32  # tokens per streaming tile (= PV MFMA K-dim)
N_SUBTILE = TILE_N // MFMA_N  # 2 QK sub-tiles per tile
BLOCK = WARP_SIZE  # one warp per CTA

_FX_DTYPE = {"f32": fx.Float32, "f16": fx.Float16, "bf16": fx.BFloat16}
LOG2E: float = 1.4426950408889634


@functools.lru_cache(maxsize=256)
def compile_pa_decode_gfx950(
    *,
    head_size: int,
    kv_dtype_str: str,
    output_dtype_str: str,
    split_k: int = 1,
    arch: str = "",
) -> Any:  # pyre-ignore[3]
    if not arch:
        arch = get_rocm_arch()
    assert head_size % MFMA_K_QK == 0
    assert head_size % MFMA_N == 0
    assert kv_dtype_str in ("f16", "bf16")
    assert arch.startswith("gfx950"), f"pa_decode_gfx950 requires gfx950, got {arch}"

    _HEAD = head_size
    _SK = split_k
    _SPLIT = _SK > 1
    _FX_KV = _FX_DTYPE[kv_dtype_str]
    _FX_OUT = _FX_DTYPE[output_dtype_str]
    _QK_GRP = _HEAD // MFMA_K_QK  # head-dim groups for QK (4 at D=128)
    _DN = _HEAD // MFMA_N  # d-passes for PV (8 at D=128)
    _mfma = (
        rocdl.mfma_f32_16x16x32_bf16
        if kv_dtype_str == "bf16"
        else rocdl.mfma_f32_16x16x32_f16
    )

    # LDS: P[MFMA_M, TILE_N] f32 + double-buffered V ([dpass][tok][16] transpose tiles).
    _NUM_DMA_V = (TILE_N * _HEAD // 8) // WARP_SIZE  # 16B (8 f16) chunks / 64 lanes
    _P_LDS = MFMA_M * TILE_N  # f32, P redistribution
    _V_LDS = TILE_N * _HEAD  # f16, one V tile (transpose layout)
    _P_BYTES = _P_LDS * 4
    _V_BYTES = _V_LDS * 2  # per buffer
    _LDS_TOTAL = _P_BYTES + 2 * _V_BYTES  # double-buffered V
    cap = smem_bytes(arch)
    if _LDS_TOTAL > cap:
        raise ValueError(f"LDS {_LDS_TOTAL}B > {arch!r} cap {cap}B")

    alloc = SmemAllocator(
        None,
        arch=arch,
        global_sym_name=f"pa_gfx950_h{_HEAD}_{kv_dtype_str}_sk{_SK}",
    )
    alloc.ptr = _LDS_TOTAL

    @flyc.kernel(known_block_size=(BLOCK, 1, 1))
    def pa_decode_gfx950_kernel(
        out_ptr: fx.Tensor,
        partial_max_ptr: fx.Tensor,
        partial_sum_ptr: fx.Tensor,
        q_ptr: fx.Tensor,
        k_ptr: fx.Tensor,
        v_ptr: fx.Tensor,
        seq_ptr: fx.Tensor,
        stride_qb: fx.Int32,
        stride_qg: fx.Int32,
        stride_qh: fx.Int32,
        stride_kb: fx.Int32,
        stride_km: fx.Int32,
        stride_kg: fx.Int32,
        stride_kh: fx.Int32,
        num_hq: fx.Int32,
        num_g: fx.Int32,
        kv_max: fx.Int32,
        num_hkv: fx.Int32,
        ratio: fx.Int32,
        softmax_scale: fx.Float32,
        split_total: fx.Int32,
    ) -> None:
        lane = gpu.thread_idx.x
        tok_lane = lane % fx.Int32(MFMA_N)  # 0..15  (N index / token within sub-tile)
        grp = lane // fx.Int32(MFMA_N)  # 0..3   (M-group / k-sub-group)

        # Grid: flat -> (split_idx, kv_head, g, b). One CTA per (b,g,kv_head[,split]).
        flat = fx.Int32(gpu.block_idx.x)
        if const_expr(_SPLIT):
            split_idx = flat % split_total
            rest = flat // split_total
        else:
            split_idx = fx.Int32(0)
            rest = flat
        hkv_abs = rest % num_hkv
        rest2 = rest // num_hkv
        g_idx = rest2 % num_g
        b_idx = rest2 // num_g
        hq_base = hkv_abs * ratio  # first query head sharing this KV head

        c_zero = arith.constant(0.0, type=T.f32)
        c_one = arith.constant(1.0, type=T.f32)
        c_neginf = arith.constant(float("-inf"), type=T.f32)
        zero_v4 = arith.constant_vector(0.0, T.vec(4, T.f32))
        zero_v8h = arith.constant_vector(0.0, T.vec(8, _FX_KV.ir_type))

        q_rsrc = buffer_ops.create_buffer_resource(q_ptr, max_size=True)
        k_rsrc = buffer_ops.create_buffer_resource(k_ptr, max_size=True)
        v_rsrc = buffer_ops.create_buffer_resource(v_ptr, max_size=True)
        out_rsrc = buffer_ops.create_buffer_resource(out_ptr, max_size=True)
        pm_rsrc = buffer_ops.create_buffer_resource(partial_max_ptr, max_size=True)
        ps_rsrc = buffer_ops.create_buffer_resource(partial_sum_ptr, max_size=True)
        seq_rsrc = buffer_ops.create_buffer_resource(seq_ptr, max_size=True)

        seq_len = buffer_ops.buffer_load(seq_rsrc, b_idx, vec_width=1, dtype=T.i32)
        t_full = arith.select(seq_len > fx.Int32(0), seq_len, kv_max)
        if const_expr(_SPLIT):
            chunk = (t_full + split_total - fx.Int32(1)) // split_total
            t_start = split_idx * chunk
            t_end_raw = (split_idx + fx.Int32(1)) * chunk
            t_end = arith.select(t_end_raw < t_full, t_end_raw, t_full)
        else:
            t_start = fx.Int32(0)
            t_end = t_full

        smem = alloc.get_base()
        p_lds = SmemPtr(smem, 0, T.f32, shape=(_P_LDS,)).get()
        lds_base = buffer_ops.extract_base_index(smem, address_space=3)  # f16 elem base
        v_lds_f16 = lds_base + fx.Index(
            _P_BYTES // 2
        )  # V tile (transpose) after P region

        # ── Pre-load Q (loop-invariant) ──
        # A-frag: lane l -> Q[head=tok_lane, k=grp*8+0..7]. head on M = tok_lane
        # (0..15); only heads < ratio meaningful.
        q_head = tok_lane
        q_base = b_idx * stride_qb + g_idx * stride_qg + (hq_base + q_head) * stride_qh
        q_frags = []
        for g in range_constexpr(_QK_GRP):
            q_off = q_base + fx.Int32(g * MFMA_K_QK) + grp * fx.Int32(8)
            q_frags.append(
                buffer_ops.buffer_load(q_rsrc, q_off, vec_width=8, dtype=_FX_KV)
            )

        kv_base = b_idx * stride_kb + g_idx * stride_kg + hkv_abs * stride_kh

        # Loop-carried state: per-head (reg e over 0..3) running max, running sum,
        # and PV accumulator (_DN d-passes x 4 regs).
        _N_ACC = _DN * 4
        _init = [c_neginf] * 4 + [c_zero] * 4 + [c_zero] * _N_ACC

        for _tile_i, state in range(
            fx.Index(t_start), fx.Index(t_end), arith.index(TILE_N), init=_init
        ):
            rmax = [fx.Float32(state[i]) for i in range(4)]
            rsum = [fx.Float32(state[4 + i]) for i in range(4)]
            acc = [fx.Float32(state[8 + i]) for i in range(_N_ACC)]
            tile_start = fx.Int32(arith.index_cast(T.i32, _tile_i))

            # ── Issue V HBM loads EARLY (into regs) so latency overlaps the
            # QK+softmax below; LDS transpose stores + barrier happen just before PV.
            _v8s = []
            _v8dst = []
            for _r in range_constexpr(_NUM_DMA_V):
                _lin = lane + fx.Int32(_r * WARP_SIZE)
                _tok = _lin % fx.Int32(TILE_N)
                _rest = _lin // fx.Int32(TILE_N)  # dpass*2 + half
                _dp = _rest // fx.Int32(2)
                _half = _rest % fx.Int32(2)
                _col = _dp * fx.Int32(16) + _half * fx.Int32(8)
                _v8s.append(
                    buffer_ops.buffer_load(
                        v_rsrc,
                        kv_base + (tile_start + _tok) * stride_km + _col,
                        vec_width=8,
                        dtype=_FX_KV,
                    )
                )
                _dst = (
                    v_lds_f16
                    + fx.Index(_dp) * fx.Index(TILE_N * 16)
                    + fx.Index(_tok) * fx.Index(16)
                    + fx.Index(_half) * fx.Index(8)
                )
                _v8dst.append(_dst)

            # ── QK: N_SUBTILE sub-tiles of 16 tokens ──
            # qk[st] reg e -> score[head=grp*4+e, tok=st*16+tok_lane]
            qk_st = []
            for st in range_constexpr(N_SUBTILE):
                acc_qk = zero_v4
                for g in range_constexpr(_QK_GRP):
                    k_tok = tile_start + fx.Int32(st * MFMA_N) + tok_lane
                    k_off = (
                        kv_base
                        + k_tok * stride_km
                        + fx.Int32(g * MFMA_K_QK)
                        + grp * fx.Int32(8)
                    )
                    k8 = buffer_ops.buffer_load(
                        k_rsrc, k_off, vec_width=8, dtype=_FX_KV
                    )
                    acc_qk = _mfma(T.vec(4, T.f32), [q_frags[g], k8, acc_qk, 0, 0, 0])
                qk_st.append(acc_qk)

            # ── Online softmax, per head (reg e); tile max via dpp over tok_lane ──
            new_max = []
            alpha = []
            for e in range_constexpr(4):
                loc = fx.Float32(c_neginf)
                for st in range_constexpr(N_SUBTILE):
                    s = fx.Float32(
                        vector.extract(
                            qk_st[st], static_position=[e], dynamic_position=[]
                        )
                    )
                    s = fx.Float32(
                        arith.mulf(arith.unwrap(s), arith.unwrap(softmax_scale))
                    )
                    # mask out-of-range tokens
                    tok_abs = tile_start + fx.Int32(st * MFMA_N) + tok_lane
                    ok = tok_abs < t_end
                    s = fx.Float32(
                        arith.select(arith.unwrap(ok), arith.unwrap(s), c_neginf)
                    )
                    loc = _mxf(loc, s)
                    qk_st[st] = vector.insert(
                        arith.unwrap(s),
                        qk_st[st],
                        static_position=[e],
                        dynamic_position=[],
                    )
                for sh in (1, 2, 4, 8):
                    loc = _mxf(loc, dpp_xor_f32(loc, sh))
                nm = _mxf(rmax[e], loc)
                new_max.append(nm)
                a = _exp2_fast(
                    fx.Float32(
                        arith.mulf(
                            arith.subf(arith.unwrap(rmax[e]), arith.unwrap(nm)),
                            arith.constant(LOG2E, type=T.f32),
                        )
                    )
                )
                alpha.append(a)

            # P = exp2((score - new_max)*log2e); write to LDS[head, tok]; accumulate sum.
            tile_sum = [fx.Float32(c_zero) for _ in range(4)]
            for e in range_constexpr(4):
                head = grp * fx.Int32(4) + fx.Int32(e)
                for st in range_constexpr(N_SUBTILE):
                    s = fx.Float32(
                        vector.extract(
                            qk_st[st], static_position=[e], dynamic_position=[]
                        )
                    )
                    p = _exp2_fast(
                        fx.Float32(
                            arith.mulf(
                                arith.subf(arith.unwrap(s), arith.unwrap(new_max[e])),
                                arith.constant(LOG2E, type=T.f32),
                            )
                        )
                    )
                    # masked lanes gave s=-inf -> p=0
                    p = fx.Float32(
                        arith.select(
                            arith.unwrap(new_max[e]) > c_neginf, arith.unwrap(p), c_zero
                        )
                    )
                    tile_sum[e] = fx.Float32(
                        arith.addf(arith.unwrap(tile_sum[e]), arith.unwrap(p))
                    )
                    tok = fx.Int32(st * MFMA_N) + tok_lane
                    vector.store(
                        fx.Vector.from_elements([arith.unwrap(p)], dtype=fx.Float32),
                        p_lds,
                        [fx.Index(head * fx.Int32(TILE_N) + tok)],
                    )
            for e in range_constexpr(4):
                for sh in (1, 2, 4, 8):
                    tile_sum[e] = fx.Float32(
                        arith.addf(
                            arith.unwrap(tile_sum[e]),
                            arith.unwrap(dpp_xor_f32(tile_sum[e], sh)),
                        )
                    )
                rsum[e] = fx.Float32(
                    arith.addf(
                        arith.mulf(arith.unwrap(alpha[e]), arith.unwrap(rsum[e])),
                        arith.unwrap(tile_sum[e]),
                    )
                )
                rmax[e] = new_max[e]

            # Write the (already-loaded) V vec8s into the LDS transpose layout; the
            # barrier below covers both P writes and these V writes before PV.
            for _r in range_constexpr(_NUM_DMA_V):
                _sp = buffer_ops.create_llvm_ptr(
                    fx.Int64(_v8dst[_r] * fx.Index(2)), address_space=3
                )
                _llvm.StoreOp(_v8s[_r], _sp, alignment=16)

            gpu.barrier()

            # ── PV: A=P[head,tok] (LDS), B=V[tok,d] -> C[head,d]; rescale acc by alpha ──
            for dpass in range_constexpr(_DN):
                for e in range_constexpr(4):
                    acc[dpass * 4 + e] = fx.Float32(
                        arith.mulf(
                            arith.unwrap(acc[dpass * 4 + e]), arith.unwrap(alpha[e])
                        )
                    )
            # A-frag P: lane l -> P[head = tok_lane, tok = grp*8 + 0..7]
            p_head = tok_lane
            p_vals = []
            for j in range_constexpr(8):
                pv = fx.Vector.load(
                    T.vec(1, T.f32),
                    p_lds,
                    [
                        fx.Index(
                            p_head * fx.Int32(TILE_N) + grp * fx.Int32(8) + fx.Int32(j)
                        )
                    ],
                )[0]
                p_vals.append(
                    arith.truncf(_FX_KV.ir_type, arith.unwrap(fx.Float32(pv)))
                )
            p_frag = zero_v8h
            for j in range_constexpr(8):
                p_frag = vector.insert(
                    p_vals[j], p_frag, static_position=[j], dynamic_position=[]
                )

            _v4h = T.vec(4, _FX_KV.ir_type)
            for dpass in range_constexpr(_DN):
                # B-frag V via two ds_read_tr16_b64 (128-bit HW transpose): group grp
                # owns toks grp*8..+7 -> V[tok, d=dpass*16+tok_lane] (2 wide reads vs 8).
                _GB = fx.Int32(dpass * (TILE_N * 16)) + (grp * fx.Int32(8)) * fx.Int32(
                    16
                )
                _off_lo = v_lds_f16 + fx.Index(_GB) + (fx.Index(tok_lane)) * fx.Index(4)
                _vlo = rocdl.ds_read_tr16_b64(
                    _v4h,
                    buffer_ops.create_llvm_ptr(
                        fx.Int64(_off_lo * fx.Index(2)), address_space=3
                    ),
                ).result
                _off_hi = _off_lo + fx.Index(4 * 16)
                _vhi = rocdl.ds_read_tr16_b64(
                    _v4h,
                    buffer_ops.create_llvm_ptr(
                        fx.Int64(_off_hi * fx.Index(2)), address_space=3
                    ),
                ).result
                v_frag = vector.shuffle(_vlo, _vhi, [0, 1, 2, 3, 4, 5, 6, 7])
                c_in = zero_v4
                for e in range_constexpr(4):
                    c_in = vector.insert(
                        arith.unwrap(acc[dpass * 4 + e]),
                        c_in,
                        static_position=[e],
                        dynamic_position=[],
                    )
                c_out = _mfma(T.vec(4, T.f32), [p_frag, v_frag, c_in, 0, 0, 0])
                for e in range_constexpr(4):
                    acc[dpass * 4 + e] = fx.Float32(
                        vector.extract(c_out, static_position=[e], dynamic_position=[])
                    )

            gpu.barrier()  # P_LDS reused next tile

            state_out = (
                [arith.unwrap(rmax[i]) for i in range(4)]
                + [arith.unwrap(rsum[i]) for i in range(4)]
                + [arith.unwrap(acc[i]) for i in range(_N_ACC)]
            )
            results = yield state_out

        f_max = [fx.Float32(results[i]) for i in range(4)]
        f_sum = [fx.Float32(results[4 + i]) for i in range(4)]
        f_acc = [fx.Float32(results[8 + i]) for i in range(_N_ACC)]

        # ── Epilogue: normalize + store per head ──
        # head=grp*4+e; d=dpass*16+tok_lane always < _HEAD, so no d guard.
        for e in range_constexpr(4):
            head = grp * fx.Int32(4) + fx.Int32(e)
            head_abs = hq_base + head
            safe_sum = fx.Float32(
                arith.select(
                    arith.unwrap(f_sum[e]) > c_zero, arith.unwrap(f_sum[e]), c_one
                )
            )
            inv = rcp_f32(safe_sum)
            if const_expr(_SPLIT):
                _pm_base = (
                    b_idx * (num_g * split_total * num_hq)
                    + g_idx * (split_total * num_hq)
                    + split_idx * num_hq
                    + head_abs
                )
                _po_base = _pm_base * fx.Int32(_HEAD)
                if (head < ratio) & (head_abs < num_hq):
                    for dpass in range_constexpr(_DN):
                        d = fx.Int32(dpass * MFMA_N) + tok_lane
                        buffer_ops.buffer_store(
                            arith.unwrap(f_acc[dpass * 4 + e]), out_rsrc, _po_base + d
                        )
                    if tok_lane == fx.Int32(0):
                        buffer_ops.buffer_store(
                            arith.unwrap(f_max[e]), pm_rsrc, _pm_base
                        )
                        buffer_ops.buffer_store(
                            arith.unwrap(f_sum[e]), ps_rsrc, _pm_base
                        )
            else:
                out_base = b_idx * stride_qb + g_idx * stride_qg + head_abs * stride_qh
                inv_raw = arith.unwrap(inv)
                if (head < ratio) & (head_abs < num_hq):
                    for dpass in range_constexpr(_DN):
                        d = fx.Int32(dpass * MFMA_N) + tok_lane
                        val = fx.Float32(
                            arith.mulf(arith.unwrap(f_acc[dpass * 4 + e]), inv_raw)
                        )
                        out_val = _FX_OUT(arith.unwrap(val))
                        buffer_ops.buffer_store(
                            arith.unwrap(out_val), out_rsrc, out_base + d
                        )

    return pa_decode_gfx950_kernel, alloc


@functools.lru_cache(maxsize=256)
def _make_gfx950_jit_launcher(head_size, kv_dtype_str, out_dtype_str, split_k):
    kernel, _alloc = compile_pa_decode_gfx950(
        head_size=head_size,
        kv_dtype_str=kv_dtype_str,
        output_dtype_str=out_dtype_str,
        split_k=split_k,
    )

    @flyc.jit
    def _launcher(
        out_ptr,
        pm_ptr,
        ps_ptr,
        q_ptr,
        k_ptr,
        v_ptr,
        seq_ptr,
        stride_qb,
        stride_qg,
        stride_qh,
        stride_kb,
        stride_km,
        stride_kg,
        stride_kh,
        num_hq,
        num_g,
        kv_max,
        num_hkv,
        ratio,
        scale,
        split_total,
        grid_x,
        stream: fx.Stream = fx.Stream(None),
    ):
        from flydsl._mlir import ir as _ir
        from flydsl.compiler.kernel_function import CompilationContext

        _alloc.finalized = False
        ctx = CompilationContext.get_current()
        with _ir.InsertionPoint(ctx.gpu_module_body):
            _alloc.finalize()
        kernel(
            out_ptr,
            pm_ptr,
            ps_ptr,
            q_ptr,
            k_ptr,
            v_ptr,
            seq_ptr,
            stride_qb,
            stride_qg,
            stride_qh,
            stride_kb,
            stride_km,
            stride_kg,
            stride_kh,
            num_hq,
            num_g,
            kv_max,
            num_hkv,
            ratio,
            scale,
            split_total,
        ).launch(grid=(grid_x, 1, 1), block=(BLOCK, 1, 1), stream=stream)

    return _launcher


def pa_decode_gfx950_launch(
    Q, K, V, seq_positions, softmax_scale, split_k=0, output_dtype=None
):
    """Head-packed MFMA decode. One CTA per KV head packs its GQA group onto
    the MFMA M-axis. Falls back to the generic kernel for ratio>16 or non-gfx950."""
    from mslk.flydsl.jit import run_compiled

    from .pa_decode_dense import auto_split_k_hp

    B, _, G, H_q, D = Q.shape
    _, KV_MAX, _, H_kv, _ = K.shape
    ratio = H_q // H_kv if H_kv > 0 else 0
    ok = (
        H_kv > 0
        and H_q % H_kv == 0
        and 1 <= ratio <= MFMA_M
        and get_rocm_arch().startswith("gfx950")
        and K.dtype in (torch.float16, torch.bfloat16)
        and D % MFMA_K_QK == 0
    )
    if not ok:
        from .pa_decode_generic import pa_decode_generic_launch

        return pa_decode_generic_launch(
            Q, K, V, seq_positions, softmax_scale, split_k, output_dtype
        )
    if output_dtype is None:
        output_dtype = Q.dtype
    kv_str = {torch.float16: "f16", torch.bfloat16: "bf16"}[K.dtype]
    out_str = {torch.float16: "f16", torch.bfloat16: "bf16", torch.float32: "f32"}[
        output_dtype
    ]
    if seq_positions is None:
        seq_positions = torch.full((B,), KV_MAX, dtype=torch.int32, device=Q.device)
    elif seq_positions.dtype != torch.int32:
        seq_positions = seq_positions.to(torch.int32)
    if split_k == 0:
        split_k = auto_split_k_hp(B, G, H_q, H_kv, KV_MAX)
    out = torch.empty((B, 1, G, H_q, D), dtype=output_dtype, device=Q.device)
    sq = Q.stride()
    sk2 = K.stride()
    dev = Q.device
    n_cta_base = B * G * H_kv
    # Thread the live stream into .launch so the kernel is captured under CUDA graphs
    # (a default-stream launch would capture empty).
    stream = torch.cuda.current_stream()
    if split_k == 1:
        dummy = torch.empty(0, dtype=torch.float32, device=dev)
        launcher = _make_gfx950_jit_launcher(D, kv_str, out_str, 1)
        run_compiled(
            launcher,
            out,
            dummy,
            dummy,
            Q,
            K,
            V,
            seq_positions,
            sq[0],
            sq[2],
            sq[3],
            sk2[0],
            sk2[1],
            sk2[2],
            sk2[3],
            H_q,
            G,
            KV_MAX,
            H_kv,
            ratio,
            softmax_scale,
            split_k,
            n_cta_base,
            stream,
        )
    else:
        po = torch.empty((B, G, split_k, H_q, D), dtype=torch.float32, device=dev)
        pm = torch.empty((B, G, split_k, H_q), dtype=torch.float32, device=dev)
        ps = torch.empty((B, G, split_k, H_q), dtype=torch.float32, device=dev)
        launcher = _make_gfx950_jit_launcher(D, kv_str, "f32", split_k)
        run_compiled(
            launcher,
            po,
            pm,
            ps,
            Q,
            K,
            V,
            seq_positions,
            sq[0],
            sq[2],
            sq[3],
            sk2[0],
            sk2[1],
            sk2[2],
            sk2[3],
            H_q,
            G,
            KV_MAX,
            H_kv,
            ratio,
            softmax_scale,
            split_k,
            n_cta_base * split_k,
            stream,
        )
        pa_decode_reduce(po, pm, ps, out.squeeze(1), stream=stream)
    return out
