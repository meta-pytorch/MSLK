# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""FlyDSL paged-attention decode (generic) — per-warp Q-head ownership with TLOOP.

Arch-generic fallback. Uses mfma_f32_16x16x32 (K=32). Each warp owns one Q head
(NUM_WARPS=4 heads/CTA), so softmax is intra-warp only — no pv_lds/ms_lds merge.
TLOOP: each warp covers all TILE_N=64 tokens per step via 4 sub-tiles of 16.

MFMA layout (mfma_f32_16x16x32_f16, wave64), lane l:
  A: vec<8,f16>/lane → A[row=l%16, k=(l//16)*8 : +8]
  B: vec<8,f16>/lane → B[col=l%16, k=(l//16)*8 : +8]
  C: vec<4,f32>/lane → C[(l//16)*4+elem, l%16]
Per warp: tok_qk = lane%16 (N-col/token), k_grp = lane//16 (0..3, D chunk).
"""

from __future__ import annotations

import functools
from typing import Any, Optional

import flydsl.compiler as flyc  # pyre-ignore[21]
import flydsl.expr as fx  # pyre-ignore[21]
import torch
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

NUM_WARPS = 4  # warps per CTA = Q heads per CTA
MFMA_N = 16  # tokens per MFMA call
MFMA_K = 32  # K-dim of mfma_f32_16x16x32_f16
TLOOP = NUM_WARPS  # sub-tiles per step (each warp covers NUM_WARPS×16 = 64 tokens)
TILE_N = TLOOP * MFMA_N  # 64 tokens per tile step
BLOCK = NUM_WARPS * WARP_SIZE  # 256 threads

_FX_DTYPE = {"f32": fx.Float32, "f16": fx.Float16, "bf16": fx.BFloat16}
LOG2E: float = 1.4426950408889634


@functools.lru_cache(maxsize=256)
def compile_pa_decode_generic(
    *,
    head_size: int,
    kv_dtype_str: str,
    output_dtype_str: str,
    split_k: int = 1,
    arch: str = "",
) -> Any:  # pyre-ignore[3]
    if not arch:
        arch = get_rocm_arch()

    assert head_size % MFMA_K == 0, f"head_size must be multiple of {MFMA_K}"
    assert kv_dtype_str in ("f16", "bf16")

    _HEAD = head_size
    _SK = split_k
    _SPLIT = _SK > 1
    _FX_KV = _FX_DTYPE[kv_dtype_str]
    _FX_OUT = _FX_DTYPE[output_dtype_str]
    # MFMA intrinsic MUST match KV operand dtype (bf16 fails the _f16 verifier).
    _mfma = (
        rocdl.mfma_f32_16x16x32_bf16
        if kv_dtype_str == "bf16"
        else rocdl.mfma_f32_16x16x32_f16
    )
    _QK_GROUPS = _HEAD // MFMA_K  # D/32 groups for Q·K
    _PV_GROUPS = _HEAD // MFMA_N  # D/16 groups for P·V

    # p_lds[NUM_WARPS*TILE_N]: P weights, [warp_id*TILE_N + td*MFMA_N + tok_qk].
    # No ms_lds/pv_lds — softmax and PV accum are intra-warp per Q head.
    _P_ELEMS = NUM_WARPS * TILE_N
    _LDS_TOTAL = _P_ELEMS * 4  # 1024 bytes

    cap = smem_bytes(arch)
    if _LDS_TOTAL > cap:
        raise ValueError(f"LDS {_LDS_TOTAL}B > {arch!r} cap {cap}B")

    alloc = SmemAllocator(
        None,
        arch=arch,
        global_sym_name=f"pa_generic_h{_HEAD}_{kv_dtype_str}_nw{NUM_WARPS}_sk{_SK}",
    )
    alloc.ptr = _LDS_TOTAL

    @flyc.kernel(known_block_size=(BLOCK, 1, 1))
    def pa_decode_generic_kernel(
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
        softmax_scale: fx.Float32,
        split_total: fx.Int32,
    ) -> None:
        tid = gpu.thread_idx.x
        warp_id = tid >> fx.Int32(6)
        lane = tid & fx.Int32(63)

        # tok_qk = lane%16 (N-col/token), k_grp = lane//16 (D chunk, 0..3)
        tok_qk = lane & fx.Int32(MFMA_N - 1)
        k_grp = lane >> fx.Int32(4)

        # Grid → (b, g, hq_block, split_idx)
        flat = fx.Int32(gpu.block_idx.x)
        if const_expr(_SPLIT):
            split_idx = flat % split_total
            rest = flat // split_total
        else:
            split_idx = fx.Int32(0)
            rest = flat

        n_hq_blocks = (num_hq + fx.Int32(NUM_WARPS - 1)) // fx.Int32(NUM_WARPS)
        hq_block = rest % n_hq_blocks
        rest2 = rest // n_hq_blocks
        g_idx = rest2 % num_g
        b_idx = rest2 // num_g

        # Each warp owns ONE Q head
        hq_abs = hq_block * fx.Int32(NUM_WARPS) + warp_id
        hkv_abs = hq_abs * num_hkv // num_hq

        q_base = b_idx * stride_qb + g_idx * stride_qg + hq_abs * stride_qh
        kv_base = b_idx * stride_kb + g_idx * stride_kg + hkv_abs * stride_kh

        c_zero = arith.constant(0.0, type=T.f32)
        c_one = arith.constant(1.0, type=T.f32)
        c_neginf = arith.constant(float("-inf"), type=T.f32)
        zero_v4 = arith.constant_vector(0.0, T.vec(4, T.f32))
        zero_v8h = arith.constant_vector(0.0, T.vec(8, _FX_KV.ir_type))

        seq_rsrc = buffer_ops.create_buffer_resource(seq_ptr, max_size=True)
        q_rsrc = buffer_ops.create_buffer_resource(q_ptr, max_size=True)
        k_rsrc = buffer_ops.create_buffer_resource(k_ptr, max_size=True)
        v_rsrc = buffer_ops.create_buffer_resource(v_ptr, max_size=True)
        out_rsrc = buffer_ops.create_buffer_resource(out_ptr, max_size=True)
        pm_rsrc = buffer_ops.create_buffer_resource(partial_max_ptr, max_size=True)
        ps_rsrc = buffer_ops.create_buffer_resource(partial_sum_ptr, max_size=True)

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
        p_lds = SmemPtr(smem, 0, T.f32, shape=(_P_ELEMS,)).get()

        # Pre-load Q A-frags: Q[hq_abs, g*MFMA_K + k_grp*8 : +8] as vec<8,f16>
        q_frags = []
        for g in range_constexpr(_QK_GROUPS):
            q_off = q_base + fx.Int32(g * MFMA_K) + k_grp * fx.Int32(8)
            q_frags.append(
                buffer_ops.buffer_load(q_rsrc, q_off, vec_width=8, dtype=_FX_KV)
            )

        # State: running max, running sum, then PV accum (_PV_GROUPS×4 C-elems/lane)
        _init_neg = arith.constant(float("-inf"), type=T.f32)
        _init_zer = arith.constant(0.0, type=T.f32)
        _N_PV = _PV_GROUPS * 4
        _init_state = [_init_neg, _init_zer] + [_init_zer] * _N_PV

        _t_s = fx.Index(t_start)
        _t_e = fx.Index(t_end)

        for _tile_i, state in range(_t_s, _t_e, arith.index(TILE_N), init=_init_state):
            running_max = fx.Float32(state[0])
            running_sum = fx.Float32(state[1])
            pv_scalars = [state[2 + i] for i in range(_N_PV)]

            tile_start = fx.Int32(arith.index_cast(T.i32, _tile_i))

            # ── QK: prefetch K for all TLOOP sub-tiles, then MFMA ─────────
            # sub-tile td covers tokens tile_start + td*MFMA_N + tok_qk
            k_frags_all = []
            for td in range_constexpr(TLOOP):
                tok_td = tile_start + fx.Int32(td * MFMA_N) + tok_qk
                k_frags_td = []
                for g in range_constexpr(_QK_GROUPS):
                    k_off = (
                        kv_base
                        + tok_td * stride_km
                        + fx.Int32(g * MFMA_K)
                        + k_grp * fx.Int32(8)
                    )
                    k_frags_td.append(
                        buffer_ops.buffer_load(k_rsrc, k_off, vec_width=8, dtype=_FX_KV)
                    )
                k_frags_all.append(k_frags_td)

            rocdl.sched_barrier(0)

            qk_vecs = []
            for td in range_constexpr(TLOOP):
                qk_acc = zero_v4
                for g in range_constexpr(_QK_GROUPS):
                    qk_acc = _mfma(
                        T.vec(4, T.f32),
                        [q_frags[g], k_frags_all[td][g], qk_acc, 0, 0, 0],
                    )
                qk_vecs.append(qk_acc)

            # ── Softmax (intra-warp, no LDS) ──────────────────────────────
            # QK scalar = C[elem=0, col=tok_qk] per sub-tile
            qk_vals = []
            for td in range_constexpr(TLOOP):
                tok_td = tile_start + fx.Int32(td * MFMA_N) + tok_qk
                qk_raw = vector.extract(
                    qk_vecs[td], static_position=[0], dynamic_position=[]
                )
                qk_sc = arith.mulf(qk_raw, arith.unwrap(softmax_scale))
                head_ok = hq_abs < num_hq
                tok_ok = tok_td < t_end
                in_range = arith.andi(arith.unwrap(head_ok), arith.unwrap(tok_ok))
                qk_vals.append(fx.Float32(arith.select(in_range, qk_sc, c_neginf)))

            # Intra-warp max across TLOOP*16 tokens (all in registers).
            tile_max = qk_vals[0]
            for td in range_constexpr(1, TLOOP):
                tile_max = _mxf(tile_max, qk_vals[td])
            # DPP butterfly over tok_qk within each 16-lane k_grp segment.
            for sh in (8, 4, 2, 1):
                tile_max = _mxf(tile_max, dpp_xor_f32(tile_max, sh))
            # NOTE: no cross-k_grp reduce needed. The MFMA already accumulates all
            # k_grps into C[0,tok_qk], so every k_grp holds the SAME full Q·K scalar.

            new_max = _mxf(running_max, tile_max)
            rescale = _exp2_fast(
                fx.Float32(
                    arith.mulf(
                        arith.subf(arith.unwrap(running_max), arith.unwrap(new_max)),
                        arith.constant(LOG2E, type=T.f32),
                    )
                )
            )

            # P values, normalized by new_max (standard online softmax).
            safe_max = fx.Float32(
                arith.select(
                    arith.unwrap(new_max) > c_neginf, arith.unwrap(new_max), c_zero
                )
            )
            p_vals = []
            intra_sum = fx.Float32(c_zero)
            for td in range_constexpr(TLOOP):
                tok_td = tile_start + fx.Int32(td * MFMA_N) + tok_qk
                head_ok = hq_abs < num_hq
                tok_ok = tok_td < t_end
                in_range = arith.andi(arith.unwrap(head_ok), arith.unwrap(tok_ok))
                p_c = _exp2_fast(
                    fx.Float32(
                        arith.mulf(
                            arith.subf(
                                arith.unwrap(qk_vals[td]), arith.unwrap(safe_max)
                            ),
                            arith.constant(LOG2E, type=T.f32),
                        )
                    )
                )
                p_c = fx.Float32(arith.select(in_range, arith.unwrap(p_c), c_zero))
                p_vals.append(p_c)
                intra_sum = fx.Float32(
                    arith.addf(arith.unwrap(intra_sum), arith.unwrap(p_c))
                )

            # DPP sum over tok_qk within 16-lane segment.
            for sh in (8, 4, 2, 1):
                intra_sum = fx.Float32(
                    arith.addf(
                        arith.unwrap(intra_sum),
                        arith.unwrap(dpp_xor_f32(intra_sum, sh)),
                    )
                )
            tile_sum = intra_sum

            new_sum = fx.Float32(
                arith.addf(
                    arith.mulf(arith.unwrap(rescale), arith.unwrap(running_sum)),
                    arith.unwrap(tile_sum),
                )
            )

            # Write P to p_lds[warp_id*TILE_N + td*MFMA_N + tok_qk].
            for td in range_constexpr(TLOOP):
                p_slot = fx.Index(
                    warp_id * fx.Int32(TILE_N) + fx.Int32(td * MFMA_N) + tok_qk
                )
                vector.store(
                    fx.Vector.from_elements(
                        [arith.unwrap(p_vals[td])], dtype=fx.Float32
                    ),
                    p_lds,
                    [p_slot],
                )
            gpu.barrier()

            # ── PV MFMA ──────────────────────────────────────────────────
            # 64 tokens, K=32 → 2 MFMA calls (halves 0..31, 32..63) per D-group.
            # P B-frag: p_lds[warp*64 + k_grp*8 + j].
            # V A-frag: V[tok=tile_start + half*32 + k_grp*8+j, d_out=g*MFMA_N+tok_qk].

            # Prefetch P frags for both halves
            p_half_base = [
                warp_id * fx.Int32(TILE_N),
                warp_id * fx.Int32(TILE_N) + fx.Int32(TILE_N // 2),
            ]
            p_frags = []
            for half in range_constexpr(2):
                p_frag = zero_v8h
                pbase = p_half_base[half] + k_grp * fx.Int32(8)
                for j in range_constexpr(8):
                    pf_j = fx.Vector.load(
                        T.vec(1, T.f32), p_lds, [fx.Index(pbase + fx.Int32(j))]
                    )[0]
                    p_f16 = arith.truncf(_FX_KV.ir_type, arith.unwrap(fx.Float32(pf_j)))
                    p_frag = vector.insert(
                        p_f16, p_frag, static_position=[j], dynamic_position=[]
                    )
                p_frags.append(p_frag)

            # Prefetch V for both halves across all _PV_GROUPS D-groups.
            v_pf = []  # [half][g][j]  half=0..1 (32 toks each), g, j=0..7
            for half in range_constexpr(2):
                vhalf = []
                for g in range_constexpr(_PV_GROUPS):
                    gvals = []
                    for j in range_constexpr(8):
                        tok_j = (
                            tile_start
                            + fx.Int32(half * (TILE_N // 2))
                            + k_grp * fx.Int32(8)
                            + fx.Int32(j)
                        )
                        d_out = fx.Int32(g * MFMA_N) + tok_qk
                        v_off = kv_base + tok_j * stride_km + d_out
                        v_val = buffer_ops.buffer_load(
                            v_rsrc, v_off, vec_width=1, dtype=_FX_KV
                        )
                        gvals.append(arith.unwrap(_FX_KV(v_val)))
                    vhalf.append(gvals)
                v_pf.append(vhalf)

            rocdl.sched_barrier(0)

            rescale_raw = arith.unwrap(rescale)
            new_pv_scalars = []
            for g in range_constexpr(_PV_GROUPS):
                c_acc = zero_v4
                for e in range_constexpr(4):
                    c_acc = vector.insert(
                        arith.mulf(pv_scalars[g * 4 + e], rescale_raw),
                        c_acc,
                        static_position=[e],
                        dynamic_position=[],
                    )

                for half in range_constexpr(2):
                    v_frag = zero_v8h
                    for j in range_constexpr(8):
                        v_frag = vector.insert(
                            v_pf[half][g][j],
                            v_frag,
                            static_position=[j],
                            dynamic_position=[],
                        )
                    c_acc = _mfma(
                        T.vec(4, T.f32), [v_frag, p_frags[half], c_acc, 0, 0, 0]
                    )

                for e in range_constexpr(4):
                    new_pv_scalars.append(
                        vector.extract(c_acc, static_position=[e], dynamic_position=[])
                    )

            pv_scalars = new_pv_scalars
            state_out = [arith.unwrap(new_max), arith.unwrap(new_sum)] + list(
                pv_scalars
            )
            results = yield state_out

        final_max = fx.Float32(results[0])
        final_sum = fx.Float32(results[1])
        final_pv_sc = [results[2 + i] for i in range(_N_PV)]

        safe_sum = fx.Float32(
            arith.select(
                arith.unwrap(final_sum) > c_zero, arith.unwrap(final_sum), c_one
            )
        )
        inv_sum = rcp_f32(safe_sum)
        out_base = b_idx * stride_qb + g_idx * stride_qg + hq_abs * stride_qh

        if const_expr(_SPLIT):
            _pm_base = (
                b_idx * (num_g * split_total * num_hq)
                + g_idx * (split_total * num_hq)
                + split_idx * num_hq
                + hq_abs
            )
            _po_base = _pm_base * fx.Int32(_HEAD)

        # Only tok_qk=0 lanes write; d_out = g*MFMA_N + k_grp*4 + elem (C layout).
        if tok_qk == fx.Int32(0):
            if hq_abs < num_hq:
                if const_expr(_SPLIT):
                    for g in range_constexpr(_PV_GROUPS):
                        for e in range_constexpr(4):
                            d_out = (
                                fx.Int32(g * MFMA_N) + k_grp * fx.Int32(4) + fx.Int32(e)
                            )
                            buffer_ops.buffer_store(
                                final_pv_sc[g * 4 + e], out_rsrc, _po_base + d_out
                            )
                else:
                    for g in range_constexpr(_PV_GROUPS):
                        for e in range_constexpr(4):
                            d_out = (
                                fx.Int32(g * MFMA_N) + k_grp * fx.Int32(4) + fx.Int32(e)
                            )
                            out_val = _FX_OUT(
                                arith.unwrap(
                                    fx.Float32(
                                        arith.mulf(
                                            final_pv_sc[g * 4 + e],
                                            arith.unwrap(inv_sum),
                                        )
                                    )
                                )
                            )
                            buffer_ops.buffer_store(
                                arith.unwrap(out_val), out_rsrc, out_base + d_out
                            )

        if const_expr(_SPLIT):
            if lane == fx.Int32(0):
                if hq_abs < num_hq:
                    buffer_ops.buffer_store(arith.unwrap(final_max), pm_rsrc, _pm_base)
                    buffer_ops.buffer_store(arith.unwrap(final_sum), ps_rsrc, _pm_base)

    return pa_decode_generic_kernel, alloc


@functools.lru_cache(maxsize=256)
def _make_generic_jit_launcher(
    head_size: int,
    kv_dtype_str: str,
    out_dtype_str: str,
    split_k: int,
) -> Any:  # pyre-ignore[3]
    kernel, _alloc = compile_pa_decode_generic(
        head_size=head_size,
        kv_dtype_str=kv_dtype_str,
        output_dtype_str=out_dtype_str,
        split_k=split_k,
    )

    @flyc.jit
    def _launcher(
        out_ptr: fx.Tensor,
        pm_ptr: fx.Tensor,
        ps_ptr: fx.Tensor,
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
        scale: fx.Float32,
        split_total: fx.Int32,
        grid_x: fx.Int32,
    ) -> None:
        from flydsl._mlir import ir as _ir  # pyre-ignore[21]
        from flydsl.compiler.kernel_function import (  # pyre-ignore[21]
            CompilationContext,
        )

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
            scale,
            split_total,
        ).launch(grid=(grid_x, 1, 1), block=(BLOCK, 1, 1))

    return _launcher


def pa_decode_generic_launch(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    seq_positions: Optional[torch.Tensor],
    softmax_scale: float,
    split_k: int = 0,
    output_dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """Decode: mfma_f32_16x16x32 + per-warp Q-head ownership + TLOOP."""
    from mslk.flydsl.jit import run_compiled  # pyre-ignore[21]

    from .pa_decode_dense import auto_split_k

    B, _, G, H_q, D = Q.shape
    _, KV_MAX, _, H_kv, _ = K.shape
    assert D % MFMA_K == 0, f"head_size must be multiple of {MFMA_K}"
    assert K.dtype in (torch.float16, torch.bfloat16)

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
        split_k = auto_split_k(B, G, H_q, KV_MAX)

    hq_blocks = (H_q + NUM_WARPS - 1) // NUM_WARPS
    out = torch.empty((B, 1, G, H_q, D), dtype=output_dtype, device=Q.device)
    sq = Q.stride()
    sk2 = K.stride()
    dev = Q.device

    if split_k == 1:
        dummy = torch.empty(0, dtype=torch.float32, device=dev)
        launcher = _make_generic_jit_launcher(D, kv_str, out_str, 1)
        grid_x = B * G * hq_blocks
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
            softmax_scale,
            split_k,
            grid_x,
        )
    else:
        po = torch.empty((B, G, split_k, H_q, D), dtype=torch.float32, device=dev)
        pm = torch.empty((B, G, split_k, H_q), dtype=torch.float32, device=dev)
        ps = torch.empty((B, G, split_k, H_q), dtype=torch.float32, device=dev)
        launcher = _make_generic_jit_launcher(D, kv_str, "f32", split_k)
        grid_x = B * G * hq_blocks * split_k
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
            softmax_scale,
            split_k,
            grid_x,
        )
        out_view = out.squeeze(1)
        pa_decode_reduce(po, pm, ps, out_view)

    return out
