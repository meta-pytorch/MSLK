# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""FlyDSL decode (gfx950 cooperative-DMA) — ds_read_tr16_b64 HW transpose.

Uses ds_read_tr16_b64 (gfx950+ HW LDS transpose) for PV to cut V LDS reads 8x
(2 reads per (dc, pks) step = 16/lane/tile vs 128 scalar reads).

MFMA: mfma_f32_32x32x16_f16 with A=V_T (from ds_read_tr16_b64), B=P.
  A[m=d_sub, k=tok] = V[tok, d=dc*32+d_sub] (HW transposed)
  B[n=d_sub2, k=tok] = P[tok] (broadcast to all n rows)
  C[m=d_sub, n=*] = PV[d=dc*32+d_sub] (all n cols equal for broadcast P)

Lane decomposition (matching flash_attn_generic.py):
  lane_div_32 = lane//32     -> tok half (lo/hi within pks step)
  tr_k_group  = (lane%16)//4 -> 0..3: K-row (tok) offset within 4-row group
  tr_col_sub  = lane%4       -> 0..3: 4-column (d) sub-group
  tr_col_half = (lane%32)//16-> 0/1: first/second 16-d half of DC chunk

V LDS: linear row-major (no swizzle — required by ds_read_tr16_b64). K/P LDS: as v3.
Output: C[e] at lane l -> d = dc*32 + ld32*4 + (e//4)*8 + (e%4).

gfx950 only (ds_read_tr16_b64 requires CDNA4).
"""

from __future__ import annotations

import functools
from typing import Any, Optional

import torch

import flydsl.compiler as flyc  # pyre-ignore[21]
import flydsl.expr as fx  # pyre-ignore[21]
from flydsl._mlir.dialects import llvm as _llvm  # pyre-ignore[21]
from flydsl.expr import (  # pyre-ignore[21]
    arith, buffer_ops, const_expr, gpu, range_constexpr, rocdl, vector,
)
from flydsl.expr.typing import T  # pyre-ignore[21]
from flydsl.runtime.device import get_rocm_arch  # pyre-ignore[21]
from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr  # pyre-ignore[21]

from .utils import dpp_xor_f32, exp2_f32 as _exp2_fast, maxnumf as _mxf, rcp_f32, smem_bytes, WARP_SIZE
from .pa_decode_reduce import pa_decode_reduce

NUM_WARPS    = 4
MFMA_N       = 16    # QK MFMA sub-tile (tokens per group, mfma_f32_16x16x32_f16)
MFMA_K_QK    = 32   # QK MFMA K-dim
TLOOP        = NUM_WARPS   # 4 sub-tiles per tile
TILE_N       = TLOOP * MFMA_N   # 64 tokens per tile
BLOCK        = NUM_WARPS * WARP_SIZE   # 256

DMA_BYTES    = 16    # bytes per lane per DMA call (raw_ptr_buffer_load_lds)

# PV: mfma_f32_32x32x16_f16 with ds_read_tr16_b64
DC_CHUNK     = 32    # d-values per DC pass (MFMA M=32); _D_CHUNKS = HEAD//DC_CHUNK
PV_K_STEP    = 16   # tokens per pks step (MFMA K=16)
K_SUB_N      = 32   # half TILE_N (lo vs hi token groups)
PV_K_STEPS   = TILE_N // PV_K_STEP  # 4 steps: pks=0..3

_FX_DTYPE = {"f32": fx.Float32, "f16": fx.Float16, "bf16": fx.BFloat16}
LOG2E: float = 1.4426950408889634


@functools.lru_cache(maxsize=256)
def compile_pa_decode_gfx950_coop(
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
    assert head_size % DC_CHUNK == 0
    assert kv_dtype_str in ("f16", "bf16")
    assert arch.startswith("gfx950"), f"pa_decode_gfx950_coop requires gfx950 (ds_read_tr16_b64), got {arch}"

    _HEAD       = head_size
    _SK         = split_k
    _SPLIT      = _SK > 1
    _FX_KV      = _FX_DTYPE[kv_dtype_str]
    _FX_OUT     = _FX_DTYPE[output_dtype_str]
    # MFMA intrinsics are dtype-specific: pick f16/bf16 to match KV operand dtype
    # (mismatched operand dtype fails MLIR verification).
    _mfma_qk = rocdl.mfma_f32_16x16x32_bf16 if kv_dtype_str == "bf16" else rocdl.mfma_f32_16x16x32_f16
    _mfma_pv = rocdl.mfma_f32_32x32x16_bf16 if kv_dtype_str == "bf16" else rocdl.mfma_f32_32x32x16_f16
    _QK_GROUPS  = _HEAD // MFMA_K_QK   # 4 for head=128
    _D_CHUNKS   = _HEAD // DC_CHUNK    # head-dim / 32 (=4 for head=128, 8 for head=256)
    # PV accumulator: _D_CHUNKS * 16 scalars per lane (v16f32 per DC chunk)
    _N_PV       = _D_CHUNKS * 16

    # LDS: K_LDS TILE_N x HEAD f16 (XOR swizzle) + V_LDS TILE_N x HEAD f16
    # (row-major, no swizzle) + P_LDS NUM_WARPS x TILE_N f32.
    _K_LDS_F16  = TILE_N * _HEAD
    _V_LDS_F16  = TILE_N * _HEAD
    _P_LDS_F32  = NUM_WARPS * TILE_N
    _LDS_TOTAL  = (_K_LDS_F16 + _V_LDS_F16) * 2 + _P_LDS_F32 * 4

    cap = smem_bytes(arch)
    if _LDS_TOTAL > cap:
        raise ValueError(f"LDS {_LDS_TOTAL}B > {arch!r} cap {cap}B")

    alloc = SmemAllocator(
        None, arch=arch,
        global_sym_name=f"pa_gfx950_coop_h{_HEAD}_{kv_dtype_str}_nw{NUM_WARPS}_sk{_SK}",
    )
    alloc.ptr = _LDS_TOTAL

    _DMA_BATCH     = BLOCK * DMA_BYTES
    _KV_TILE_BYTES = TILE_N * _HEAD * 2
    _NUM_DMA_KV    = _KV_TILE_BYTES // _DMA_BATCH   # 4 rounds
    _LANES_PER_ROW = _HEAD * 2 // DMA_BYTES          # 16
    _ROWS_PER_ROUND = _DMA_BATCH // (_HEAD * 2)      # 16

    # V LDS stride (row-major; ds_read_tr16_b64 needs no padding)
    _V_STRIDE = _HEAD   # f16 per row (tok)

    @flyc.kernel(known_block_size=(BLOCK, 1, 1))
    def pa_decode_gfx950_coop_kernel(
        out_ptr: fx.Tensor, partial_max_ptr: fx.Tensor, partial_sum_ptr: fx.Tensor,
        q_ptr: fx.Tensor, k_ptr: fx.Tensor, v_ptr: fx.Tensor, seq_ptr: fx.Tensor,
        stride_qb: fx.Int32, stride_qg: fx.Int32, stride_qh: fx.Int32,
        stride_kb: fx.Int32, stride_km: fx.Int32, stride_kg: fx.Int32, stride_kh: fx.Int32,
        num_hq: fx.Int32, num_g: fx.Int32, kv_max: fx.Int32, num_hkv: fx.Int32,
        softmax_scale: fx.Float32, split_total: fx.Int32,
    ) -> None:
        tid     = gpu.thread_idx.x
        warp_id = tid >> fx.Int32(6)
        lane    = tid & fx.Int32(63)

        # QK lane decomposition (mfma_f32_16x16x32_f16)
        tok_qk  = lane & fx.Int32(MFMA_N - 1)
        k_grp   = lane >> fx.Int32(4)

        # ds_read_tr16_b64 lane decomposition (PV mfma_f32_32x32x16_f16)
        lane_div_32  = lane >> fx.Int32(5)              # 0 or 1
        lane_mod_32  = lane & fx.Int32(31)
        tr_k_group   = (lane & fx.Int32(15)) >> fx.Int32(2)    # (lane%16)//4: 0..3
        tr_col_sub   = lane & fx.Int32(3)                       # lane%4: 0..3
        tr_col_half  = (lane & fx.Int32(31)) >> fx.Int32(4)    # (lane%32)//16: 0 or 1

        # Grid decode
        flat = fx.Int32(gpu.block_idx.x)
        if const_expr(_SPLIT):
            split_idx = flat % split_total
            rest      = flat // split_total
        else:
            split_idx = fx.Int32(0)
            rest      = flat

        n_hq_blocks = (num_hq + fx.Int32(NUM_WARPS - 1)) // fx.Int32(NUM_WARPS)
        hq_block    = rest % n_hq_blocks
        rest2       = rest // n_hq_blocks
        g_idx       = rest2 % num_g
        b_idx       = rest2 // num_g

        hq_abs  = hq_block * fx.Int32(NUM_WARPS) + warp_id
        hkv_abs = hq_abs * num_hkv // num_hq
        q_base  = b_idx * stride_qb + g_idx * stride_qg + hq_abs * stride_qh
        kv_base = b_idx * stride_kb + g_idx * stride_kg + hkv_abs * stride_kh

        c_zero   = arith.constant(0.0, type=T.f32)
        c_one    = arith.constant(1.0, type=T.f32)
        c_neginf = arith.constant(float('-inf'), type=T.f32)
        zero_v4  = arith.constant_vector(0.0, T.vec(4, T.f32))
        zero_v8h = arith.constant_vector(0.0, T.vec(8, _FX_KV.ir_type))
        zero_v16 = arith.constant_vector(0.0, T.vec(16, T.f32))

        seq_rsrc = buffer_ops.create_buffer_resource(seq_ptr,         max_size=True)
        q_rsrc   = buffer_ops.create_buffer_resource(q_ptr,           max_size=True)
        k_rsrc   = buffer_ops.create_buffer_resource(k_ptr,           max_size=True)
        v_rsrc   = buffer_ops.create_buffer_resource(v_ptr,           max_size=True)
        out_rsrc = buffer_ops.create_buffer_resource(out_ptr,         max_size=True)
        pm_rsrc  = buffer_ops.create_buffer_resource(partial_max_ptr, max_size=True)
        ps_rsrc  = buffer_ops.create_buffer_resource(partial_sum_ptr, max_size=True)

        seq_len = buffer_ops.buffer_load(seq_rsrc, b_idx, vec_width=1, dtype=T.i32)
        t_full  = arith.select(seq_len > fx.Int32(0), seq_len, kv_max)
        if const_expr(_SPLIT):
            chunk     = (t_full + split_total - fx.Int32(1)) // split_total
            t_start   = split_idx * chunk
            t_end_raw = (split_idx + fx.Int32(1)) * chunk
            t_end     = arith.select(t_end_raw < t_full, t_end_raw, t_full)
        else:
            t_start = fx.Int32(0)
            t_end   = t_full

        smem    = alloc.get_base()
        lds_base = buffer_ops.extract_base_index(smem, address_space=3)
        k_lds_base_bytes = lds_base
        v_lds_base_bytes = lds_base + fx.Index(_K_LDS_F16 * 2)
        p_lds = SmemPtr(smem, (_K_LDS_F16 + _V_LDS_F16) * 2, T.f32,
                        shape=(_P_LDS_F32,)).get()

        _wave_dma_offset = fx.Index(warp_id * fx.Int32(WARP_SIZE * DMA_BYTES))
        _dma_size = fx.Int32(DMA_BYTES)
        _dma_soff = fx.Int32(0)
        _dma_off  = fx.Int32(0)
        _dma_aux  = fx.Int32(1)

        # Pre-load Q
        q_frags = []
        for g in range_constexpr(_QK_GROUPS):
            q_off = q_base + fx.Int32(g * MFMA_K_QK) + k_grp * fx.Int32(8)
            q_frags.append(buffer_ops.buffer_load(q_rsrc, q_off, vec_width=8, dtype=_FX_KV))

        _init_neg = arith.constant(float('-inf'), type=T.f32)
        _init_zer = arith.constant(0.0,           type=T.f32)
        _init_state = [_init_neg, _init_zer] + [_init_zer] * _N_PV

        for _tile_i, state in range(fx.Index(t_start), fx.Index(t_end),
                                    arith.index(TILE_N), init=_init_state):
            running_max = fx.Float32(state[0])
            running_sum = fx.Float32(state[1])
            pv_scalars  = [state[2 + i] for i in range(_N_PV)]

            tile_start = fx.Int32(arith.index_cast(T.i32, _tile_i))

            # ── DMA K to LDS (linear, QK reads K linearly) ──
            for d in range_constexpr(_NUM_DMA_KV):
                row_in_tile = tid // fx.Index(_LANES_PER_ROW) + fx.Index(d * _ROWS_PER_ROUND)
                col_f16     = (tid % fx.Index(_LANES_PER_ROW)) * fx.Index(8)
                global_row  = tile_start + fx.Int32(row_in_tile)
                k_voffset   = (kv_base + global_row * stride_km + fx.Int32(col_f16)) * fx.Int32(2)
                k_lds_rb    = k_lds_base_bytes + _wave_dma_offset + fx.Index(d * _DMA_BATCH)
                rocdl.raw_ptr_buffer_load_lds(k_rsrc,
                    buffer_ops.create_llvm_ptr(rocdl.readfirstlane(fx.Int64.ir_type, fx.Int64(k_lds_rb)), address_space=3),
                    _dma_size, k_voffset, _dma_soff, _dma_off, _dma_aux)

            # ── DMA V to LDS (row-major; ds_read_tr16_b64 needs linear layout) ──
            for d in range_constexpr(_NUM_DMA_KV):
                row_in_tile = tid // fx.Index(_LANES_PER_ROW) + fx.Index(d * _ROWS_PER_ROUND)
                col_f16     = (tid % fx.Index(_LANES_PER_ROW)) * fx.Index(8)
                global_row  = tile_start + fx.Int32(row_in_tile)
                v_voffset   = (kv_base + global_row * stride_km + fx.Int32(col_f16)) * fx.Int32(2)
                v_lds_rb    = v_lds_base_bytes + _wave_dma_offset + fx.Index(d * _DMA_BATCH)
                rocdl.raw_ptr_buffer_load_lds(v_rsrc,
                    buffer_ops.create_llvm_ptr(rocdl.readfirstlane(fx.Int64.ir_type, fx.Int64(v_lds_rb)), address_space=3),
                    _dma_size, v_voffset, _dma_soff, _dma_off, _dma_aux)

            gpu.barrier()

            # ── QK (mfma_f32_16x16x32_f16) ──
            tile_max   = fx.Float32(c_neginf)
            qk_scalars = []
            for td in range_constexpr(TLOOP):
                k_tok  = fx.Int32(td * MFMA_N) + tok_qk
                k_v8s  = []
                for g in range_constexpr(_QK_GROUPS):
                    k_col  = fx.Int32(g * MFMA_K_QK) + k_grp * fx.Int32(8)
                    k_byte = k_lds_base_bytes + (fx.Index(k_tok) * fx.Index(_HEAD) + fx.Index(k_col)) * fx.Index(2)
                    k_ptr  = buffer_ops.create_llvm_ptr(fx.Int64(k_byte), address_space=3)
                    k_v8s.append(_llvm.LoadOp(T.vec(8, _FX_KV.ir_type), k_ptr, alignment=16).result)
                qk_acc = zero_v4
                for g in range_constexpr(_QK_GROUPS):
                    qk_acc = _mfma_qk(T.vec(4, T.f32), [q_frags[g], k_v8s[g], qk_acc, 0, 0, 0])
                tok_td   = tile_start + fx.Int32(td * MFMA_N) + tok_qk
                qk_raw   = vector.extract(qk_acc, static_position=[0], dynamic_position=[])
                qk_sc    = arith.mulf(qk_raw, arith.unwrap(softmax_scale))
                head_ok  = hq_abs < num_hq
                tok_ok   = tok_td < t_end
                in_range = arith.andi(arith.unwrap(head_ok), arith.unwrap(tok_ok))
                qk_val   = fx.Float32(arith.select(in_range, qk_sc, c_neginf))
                qk_scalars.append(qk_val)
                tile_max = _mxf(tile_max, qk_val)

            for sh in (8, 4, 2, 1):
                tile_max = _mxf(tile_max, dpp_xor_f32(tile_max, sh))

            new_max = _mxf(running_max, tile_max)
            rescale = _exp2_fast(fx.Float32(arith.mulf(
                arith.subf(arith.unwrap(running_max), arith.unwrap(new_max)),
                arith.constant(LOG2E, type=T.f32))))

            safe_max  = fx.Float32(arith.select(arith.unwrap(new_max) > c_neginf, arith.unwrap(new_max), c_zero))
            intra_sum = fx.Float32(c_zero)
            for td in range_constexpr(TLOOP):
                tok_td   = tile_start + fx.Int32(td * MFMA_N) + tok_qk
                head_ok  = hq_abs < num_hq
                tok_ok   = tok_td < t_end
                in_range = arith.andi(arith.unwrap(head_ok), arith.unwrap(tok_ok))
                p_c = _exp2_fast(fx.Float32(arith.mulf(
                    arith.subf(arith.unwrap(qk_scalars[td]), arith.unwrap(safe_max)),
                    arith.constant(LOG2E, type=T.f32))))
                p_c = fx.Float32(arith.select(in_range, arith.unwrap(p_c), c_zero))
                intra_sum = fx.Float32(arith.addf(arith.unwrap(intra_sum), arith.unwrap(p_c)))
                p_slot = fx.Index(warp_id * fx.Int32(TILE_N) + fx.Int32(td * MFMA_N) + tok_qk)
                vector.store(fx.Vector.from_elements([arith.unwrap(p_c)], dtype=fx.Float32), p_lds, [p_slot])

            for sh in (8, 4, 2, 1):
                intra_sum = fx.Float32(arith.addf(arith.unwrap(intra_sum), arith.unwrap(dpp_xor_f32(intra_sum, sh))))
            new_sum = fx.Float32(arith.addf(
                arith.mulf(arith.unwrap(rescale), arith.unwrap(running_sum)), arith.unwrap(intra_sum)))

            gpu.barrier()

            # ── PV: mfma_f32_32x32x16_f16 with A=V_T (ds_read_tr16_b64), B=P ──
            # ds_read_tr16_b64 lane addressing (matching flash_attn_generic.py):
            #   d_col  = dc*32 + tr_col_half*16 + tr_col_sub*4  (f16 idx, d-dim)
            #   k_row  = pks*16 + ld32*4 + tr_k_group           (f16 idx, tok-dim)
            #   lds_lo = v_lds_base + k_row*_V_STRIDE + d_col
            # v_lo=tr16(lds_lo) -> k=0..3; v_hi=tr16(lds_lo+8*_V_STRIDE) -> k=4..7;
            # A-frag = shuffle(v_lo, v_hi) -> v8f16. P B-frag mirrors tr16 tok order:
            # k=0..3 -> toks pks*16+ld32*4+j, k=4..7 -> +j+4 (gap at 4..7).

            v4f16_type = T.vec(4, _FX_KV.ir_type)

            rescale_raw = arith.unwrap(rescale)
            new_pv_scalars = []
            for dc in range_constexpr(_D_CHUNKS):
                c_acc = zero_v16
                for e in range_constexpr(16):
                    c_acc = vector.insert(arith.mulf(pv_scalars[dc * 16 + e], rescale_raw),
                                          c_acc, static_position=[e], dynamic_position=[])

                # d_col base for this DC chunk (per-lane via tr_col_half/tr_col_sub)
                d_col_base = fx.Index(dc * DC_CHUNK) + tr_col_half * fx.Index(16) + tr_col_sub * fx.Index(4)

                for pks in range_constexpr(PV_K_STEPS):
                    # k_row base for this pks step (per-lane via lane_div_32/tr_k_group)
                    k_row_base = fx.Index(pks * PV_K_STEP) + lane_div_32 * fx.Index(4) + tr_k_group

                    # V A-frag via ds_read_tr16_b64: two reads combine into v8f16
                    v_lds_lo_f16 = v_lds_base_bytes // fx.Index(2) + k_row_base * fx.Index(_V_STRIDE) + d_col_base
                    v_lds_lo_byte = v_lds_lo_f16 * fx.Index(2)
                    v_lds_hi_byte = v_lds_lo_byte + fx.Index(8 * _V_STRIDE * 2)  # +8 toks

                    lo_ptr = buffer_ops.create_llvm_ptr(fx.Int64(v_lds_lo_byte), address_space=3)
                    hi_ptr = buffer_ops.create_llvm_ptr(fx.Int64(v_lds_hi_byte), address_space=3)
                    v_lo_v4 = rocdl.ds_read_tr16_b64(v4f16_type, lo_ptr).result  # k=0..3
                    v_hi_v4 = rocdl.ds_read_tr16_b64(v4f16_type, hi_ptr).result  # k=4..7
                    # Combine into v8f16 A-frag: [lo[0..3], hi[0..3]]
                    v_frag = vector.shuffle(v_lo_v4, v_hi_v4, [0, 1, 2, 3, 4, 5, 6, 7])

                    # P B-frag must match V A-frag tok order: j=0..3 -> tok
                    # pks*16+ld32*4+j, j=4..7 -> +j+4 (V hi-read covers toks +{8..11}).
                    p_frag = zero_v8h
                    for j in range_constexpr(8):
                        tok_j = fx.Int32(pks * PV_K_STEP) + lane_div_32 * fx.Int32(4) + fx.Int32(j % 4) + fx.Int32((j // 4) * 8)
                        p_slot = warp_id * fx.Int32(TILE_N) + tok_j
                        pf = fx.Vector.load(T.vec(1, T.f32), p_lds, [fx.Index(p_slot)])[0]
                        p_f16 = arith.truncf(_FX_KV.ir_type, arith.unwrap(fx.Float32(pf)))
                        p_frag = vector.insert(p_f16, p_frag, static_position=[j], dynamic_position=[])

                    # PV MFMA: A=V_T (tr16), B=P (broadcast) -> C[m=d_sub, n=*]=PV[d]
                    c_acc = _mfma_pv(
                        T.vec(16, T.f32), [v_frag, p_frag, c_acc, 0, 0, 0])

                for e in range_constexpr(16):
                    new_pv_scalars.append(vector.extract(c_acc, static_position=[e], dynamic_position=[]))

            pv_scalars = new_pv_scalars
            state_out = [arith.unwrap(new_max), arith.unwrap(new_sum)] + list(pv_scalars)
            results = yield state_out

        final_max   = fx.Float32(results[0])
        final_sum   = fx.Float32(results[1])
        final_pv_sc = [results[2 + i] for i in range(_N_PV)]

        safe_sum = fx.Float32(arith.select(arith.unwrap(final_sum) > c_zero, arith.unwrap(final_sum), c_one))
        inv_sum  = rcp_f32(safe_sum)
        out_base = b_idx * stride_qb + g_idx * stride_qg + hq_abs * stride_qh

        if const_expr(_SPLIT):
            _pm_base = (b_idx * (num_g * split_total * num_hq)
                        + g_idx * (split_total * num_hq)
                        + split_idx * num_hq + hq_abs)
            _po_base = _pm_base * fx.Int32(_HEAD)

        # Output layout (empirical): C[e] at lane l -> d = dc*32 + ld32*4 + (e//4)*8 + (e%4).
        # ld32=0/1 partition the 32 d per DC chunk; all written once per lane.
        if hq_abs < num_hq:
            inv_raw = arith.unwrap(inv_sum)
            for dc in range_constexpr(_D_CHUNKS):
                for e in range_constexpr(16):
                    d_out = fx.Int32(dc * DC_CHUNK) + lane_div_32 * fx.Int32(4) + fx.Int32((e // 4) * 8 + (e % 4))
                    pv_val = final_pv_sc[dc * 16 + e]
                    if const_expr(_SPLIT):
                        buffer_ops.buffer_store(pv_val, out_rsrc, _po_base + d_out)
                    else:
                        out_val = _FX_OUT(arith.unwrap(fx.Float32(arith.mulf(pv_val, inv_raw))))
                        buffer_ops.buffer_store(arith.unwrap(out_val), out_rsrc, out_base + d_out)

        if const_expr(_SPLIT):
            if lane == fx.Int32(0):
                if hq_abs < num_hq:
                    buffer_ops.buffer_store(arith.unwrap(final_max), pm_rsrc, _pm_base)
                    buffer_ops.buffer_store(arith.unwrap(final_sum), ps_rsrc, _pm_base)

    return pa_decode_gfx950_coop_kernel, alloc


@functools.lru_cache(maxsize=256)
def _make_gfx950_coop_jit_launcher(head_size, kv_dtype_str, out_dtype_str, split_k):
    kernel, _alloc = compile_pa_decode_gfx950_coop(head_size=head_size, kv_dtype_str=kv_dtype_str,
                                           output_dtype_str=out_dtype_str, split_k=split_k)
    @flyc.jit
    def _launcher(out_ptr, pm_ptr, ps_ptr, q_ptr, k_ptr, v_ptr, seq_ptr,
                  stride_qb, stride_qg, stride_qh, stride_kb, stride_km, stride_kg, stride_kh,
                  num_hq, num_g, kv_max, num_hkv, scale, split_total, grid_x):
        from flydsl.compiler.kernel_function import CompilationContext
        from flydsl._mlir import ir as _ir
        _alloc.finalized = False
        ctx = CompilationContext.get_current()
        with _ir.InsertionPoint(ctx.gpu_module_body):
            _alloc.finalize()
        kernel(out_ptr, pm_ptr, ps_ptr, q_ptr, k_ptr, v_ptr, seq_ptr,
               stride_qb, stride_qg, stride_qh, stride_kb, stride_km, stride_kg, stride_kh,
               num_hq, num_g, kv_max, num_hkv, scale, split_total).launch(
                   grid=(grid_x, 1, 1), block=(BLOCK, 1, 1))
    return _launcher


def pa_decode_gfx950_coop_launch(Q, K, V, seq_positions, softmax_scale, split_k=0, output_dtype=None):
    """ds_read_tr16_b64 HW transpose for V reads — 8× fewer LDS instructions than scalar reads."""
    from mslk.flydsl.jit import run_compiled
    from flydsl.runtime.device import get_rocm_arch
    from .pa_decode_dense import auto_split_k_coop
    B,_,G,H_q,D = Q.shape; _,KV_MAX,_,H_kv,_ = K.shape
    # Requires (a) gfx950 (ds_read_tr16_b64 + raw_ptr_buffer_load_lds) and (b)
    # cooperative-DMA coherence: all NUM_WARPS warps share one K/V LDS tile, so
    # must map to the same KV head — valid only when GQA ratio H_q/H_kv is a
    # multiple of NUM_WARPS. Else fall back to pa_decode_generic (per-warp heads).
    _coop_ok = (H_q % H_kv == 0 and (H_q // H_kv) % NUM_WARPS == 0
                and H_q % NUM_WARPS == 0)
    if not _coop_ok or not get_rocm_arch().startswith("gfx950"):
        from .pa_decode_generic import pa_decode_generic_launch
        return pa_decode_generic_launch(Q, K, V, seq_positions, softmax_scale, split_k, output_dtype)
    assert D % MFMA_K_QK == 0 and D % DC_CHUNK == 0
    assert K.dtype in (torch.float16, torch.bfloat16)
    if output_dtype is None: output_dtype = Q.dtype
    kv_str = {torch.float16:"f16", torch.bfloat16:"bf16"}[K.dtype]
    out_str = {torch.float16:"f16", torch.bfloat16:"bf16", torch.float32:"f32"}[output_dtype]
    if seq_positions is None: seq_positions = torch.full((B,),KV_MAX,dtype=torch.int32,device=Q.device)
    elif seq_positions.dtype != torch.int32: seq_positions = seq_positions.to(torch.int32)
    if split_k == 0: split_k = auto_split_k_coop(B,G,H_q,KV_MAX)
    hq_blocks = (H_q + NUM_WARPS - 1) // NUM_WARPS
    out = torch.empty((B,1,G,H_q,D), dtype=output_dtype, device=Q.device)
    sq = Q.stride(); sk2 = K.stride(); dev = Q.device
    if split_k == 1:
        dummy = torch.empty(0,dtype=torch.float32,device=dev)
        launcher = _make_gfx950_coop_jit_launcher(D,kv_str,out_str,1)
        run_compiled(launcher,out,dummy,dummy,Q,K,V,seq_positions,
                     sq[0],sq[2],sq[3],sk2[0],sk2[1],sk2[2],sk2[3],
                     H_q,G,KV_MAX,H_kv,softmax_scale,split_k,B*G*hq_blocks)
    else:
        po = torch.empty((B,G,split_k,H_q,D),dtype=torch.float32,device=dev)
        pm = torch.empty((B,G,split_k,H_q),dtype=torch.float32,device=dev)
        ps = torch.empty((B,G,split_k,H_q),dtype=torch.float32,device=dev)
        launcher = _make_gfx950_coop_jit_launcher(D,kv_str,"f32",split_k)
        run_compiled(launcher,po,pm,ps,Q,K,V,seq_positions,
                     sq[0],sq[2],sq[3],sk2[0],sk2[1],sk2[2],sk2[3],
                     H_q,G,KV_MAX,H_kv,softmax_scale,split_k,B*G*hq_blocks*split_k)
        pa_decode_reduce(po,pm,ps,out.squeeze(1))
    return out
