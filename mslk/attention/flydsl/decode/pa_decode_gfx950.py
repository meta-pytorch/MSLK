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

# Query blocks longer than MFMA_M//ratio need several M-tiles, looped inside the
# KV loop so one K/V stream feeds them all. Each tile adds loop-carried
# accumulator, so the count is bounded by where the compiler starts spilling.
#
# Measured on gfx950 (rocprofv3 --kernel-trace, VGPR_Count + Scratch_Size):
#
#   tiles     1    2    3    4    5    6      7      8
#   D=64     52   76  104  132  140  168    188    212      all clean
#   D=128    88  120  148  184  220  256   256*   256*      * spills (280B, 916B)
#
# So D=64 is clean to 8 tiles and D=128 to 6. Spilling a bandwidth-bound decode
# kernel is self-defeating.
MAX_M_TILES_BY_HEAD_DIM = {64: 8, 128: 6}


def max_m_tiles(head_dim: int) -> int:
    """Largest M-tile count that does not spill at this head dim."""
    return MAX_M_TILES_BY_HEAD_DIM.get(head_dim, 2)


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
    paged: bool = False,
    page_size: int = 0,
    gqa_ratio: int = MFMA_M,
    seqlen_q: int = 1,
    kv_i64: bool = False,
) -> Any:  # pyre-ignore[3]
    if not arch:
        arch = get_rocm_arch()
    assert head_size % MFMA_K_QK == 0
    assert head_size % MFMA_N == 0
    assert kv_dtype_str in ("f16", "bf16")
    assert arch.startswith("gfx950"), f"pa_decode_gfx950 requires gfx950, got {arch}"
    if paged:
        # A TILE_N-aligned tile must never straddle a page, so the whole tile shares
        # one block-table entry and the lookup stays scalar (see the tile loop).
        assert page_size > 0 and page_size % TILE_N == 0, (
            f"paged pa_decode_gfx950 requires page_size % {TILE_N} == 0, got {page_size}"
        )

    _HEAD = head_size
    _SK = split_k
    _SPLIT = _SK > 1
    _PAGED = bool(paged)
    _PAGE_SIZE = int(page_size)
    # Paged caches past 4 GiB need 64-bit page offsets. That costs a descriptor
    # per tile (measured ~6% at D=128), so it is only compiled in when needed.
    _KV_I64 = _PAGED and bool(kv_i64)

    # M-axis packing. The 16 M slots hold (query-token, head) pairs laid out as
    # m = qtok * ratio + head, so _T_PACK query tokens ride on one tile and a
    # query block of _SQ needs _M_TILES of them. At ratio 16 this degenerates to
    # _T_PACK=1 / _M_TILES=1, i.e. the original heads-only mapping.
    _RATIO = int(gqa_ratio)
    _SQ = int(seqlen_q)
    assert 1 <= _RATIO <= MFMA_M, f"gqa_ratio must be in [1,{MFMA_M}], got {_RATIO}"
    assert _SQ == 1 or MFMA_M % _RATIO == 0, (
        f"gqa_ratio must divide MFMA_M={MFMA_M} so (qtok, head) tiles evenly "
        f"when seqlen_q > 1; got {_RATIO}"
    )
    assert _SQ >= 1
    # At a single query token any ratio <= MFMA_M fits one tile heads-only.
    _T_PACK = MFMA_M // _RATIO if MFMA_M % _RATIO == 0 else 1
    _M_TILES = (_SQ + _T_PACK - 1) // _T_PACK
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
    _P_LDS = MFMA_M * TILE_N  # f32, P redistribution (per M-tile)
    _V_LDS = TILE_N * _HEAD  # f16, one V tile (transpose layout)
    _P_BYTES = _M_TILES * _P_LDS * 4  # one P plane per M-tile
    _V_BYTES = _V_LDS * 2  # per buffer
    _LDS_TOTAL = _P_BYTES + 2 * _V_BYTES  # double-buffered V
    cap = smem_bytes(arch)
    if _LDS_TOTAL > cap:
        raise ValueError(f"LDS {_LDS_TOTAL}B > {arch!r} cap {cap}B")

    alloc = SmemAllocator(
        None,
        arch=arch,
        global_sym_name=(
            f"pa_gfx950_h{_HEAD}_{kv_dtype_str}_sk{_SK}"
            + (f"_pg{_PAGE_SIZE}" if _PAGED else "")
            + f"_r{_RATIO}q{_SQ}"
            + ("_i64" if _KV_I64 else "")
        ),
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
        bt_ptr: fx.Tensor,
        stride_qb: fx.Int32,
        stride_qs: fx.Int32,
        stride_qg: fx.Int32,
        stride_qh: fx.Int32,
        # Dense KV: (batch, token, group, kv_head) strides of [B, KV_MAX, G, H_kv, D].
        # Paged KV: stride_kb is the *page* stride and stride_km the token-within-page
        # stride of [num_pages, page_size, H_kv, D]; stride_kg is unused (pass 0).
        stride_kb: fx.Int32,
        stride_km: fx.Int32,
        stride_kg: fx.Int32,
        stride_kh: fx.Int32,
        stride_btb: fx.Int32,
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
        if const_expr(_PAGED):
            bt_rsrc = buffer_ops.create_buffer_resource(bt_ptr, max_size=False)

        seq_len = buffer_ops.buffer_load(seq_rsrc, b_idx, vec_width=1, dtype=T.i32)
        if const_expr(_PAGED):
            # Paged lengths are exact: an empty (e.g. padding) request, or a
            # query group whose shifted length went negative, attends to nothing
            # rather than walking stale block-table entries up to kv_max.
            t_full = arith.select(seq_len > fx.Int32(0), seq_len, fx.Int32(0))
        else:
            t_full = arith.select(seq_len > fx.Int32(0), seq_len, kv_max)
        if const_expr(_SPLIT):
            chunk = (t_full + split_total - fx.Int32(1)) // split_total
            # Round the chunk up to TILE_N so every tile_start stays TILE_N-aligned.
            # Paged addressing depends on it (a tile must not straddle a page); it is
            # harmless for dense. Trailing splits may end up empty, which the reduce
            # kernel already tolerates (it skips partitions with sum == 0).
            chunk = ((chunk + fx.Int32(TILE_N - 1)) // fx.Int32(TILE_N)) * fx.Int32(
                TILE_N
            )
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
        # A-frag: lane l -> Q[m=tok_lane, k=grp*8+0..7], where the M slot decodes
        # as m = qtok * ratio + head. Rows whose query token is past _SQ are
        # clamped to the last valid token here (a harmless duplicate load) and
        # discarded in the epilogue.
        q_head = tok_lane % fx.Int32(_RATIO)
        q_tok_local = tok_lane // fx.Int32(_RATIO)
        q_base_bh = (
            b_idx * stride_qb + g_idx * stride_qg + (hq_base + q_head) * stride_qh
        )
        q_frags = []  # q_frags[m_tile][g]
        for t in range_constexpr(_M_TILES):
            q_tok = q_tok_local + fx.Int32(t * _T_PACK)
            q_tok = fx.Int32(
                arith.select(
                    arith.unwrap(q_tok < fx.Int32(_SQ)),
                    arith.unwrap(q_tok),
                    arith.constant(_SQ - 1, type=T.i32),
                )
            )
            q_base = q_base_bh + q_tok * stride_qs
            frags = []
            for g in range_constexpr(_QK_GRP):
                q_off = q_base + fx.Int32(g * MFMA_K_QK) + grp * fx.Int32(8)
                frags.append(
                    buffer_ops.buffer_load(q_rsrc, q_off, vec_width=8, dtype=_FX_KV)
                )
            q_frags.append(frags)

        # Dense: the (b, g) origin is a fixed offset. Paged: (b, token) is resolved per
        # tile through the block table, so only the kv-head offset is loop-invariant.
        if const_expr(_PAGED):
            kv_base = hkv_abs * stride_kh
        else:
            kv_base = b_idx * stride_kb + g_idx * stride_kg + hkv_abs * stride_kh

        # Loop-carried state: per-head (reg e over 0..3) running max, running sum,
        # and PV accumulator (_DN d-passes x 4 regs).
        _N_ACC = _DN * 4
        _TILE_STATE = 8 + _N_ACC  # rmax[4] + rsum[4] + acc[_N_ACC], per M-tile
        _init = ([c_neginf] * 4 + [c_zero] * 4 + [c_zero] * _N_ACC) * _M_TILES

        for _tile_i, state in range(
            fx.Index(t_start), fx.Index(t_end), arith.index(TILE_N), init=_init
        ):
            rmax = [
                [fx.Float32(state[t * _TILE_STATE + i]) for i in range(4)]
                for t in range(_M_TILES)
            ]
            rsum = [
                [fx.Float32(state[t * _TILE_STATE + 4 + i]) for i in range(4)]
                for t in range(_M_TILES)
            ]
            acc = [
                [fx.Float32(state[t * _TILE_STATE + 8 + i]) for i in range(_N_ACC)]
                for t in range(_M_TILES)
            ]
            tile_start = fx.Int32(arith.index_cast(T.i32, _tile_i))

            # ── Paged: one block-table read per tile ──
            # tile_start is TILE_N-aligned and page_size % TILE_N == 0, so all TILE_N
            # tokens of this tile live in one page. The lookup is therefore uniform
            # across the warp (scalar) and hoisted here rather than per lane.
            if const_expr(_PAGED):
                _page = buffer_ops.buffer_load(
                    bt_rsrc,
                    b_idx * stride_btb + tile_start // fx.Int32(_PAGE_SIZE),
                    vec_width=1,
                    dtype=T.i32,
                )
                # Origin of this tile inside its page.
                tile_org = kv_base + (tile_start % fx.Int32(_PAGE_SIZE)) * stride_km
                if const_expr(_KV_I64):
                    # Fold the page into a per-tile descriptor base in 64 bits: as
                    # a 32-bit element offset, page * stride_kb wraps past 4 GiB.
                    _page_u = rocdl.readfirstlane(T.i32, arith.unwrap(fx.Int32(_page)))
                    _page_bytes = (
                        fx.Int64(fx.Int32(_page_u))
                        * fx.Int64(stride_kb)
                        * fx.Int64(2)  # f16/bf16
                    )
                    k_tile_rsrc = buffer_ops.create_buffer_resource(
                        k_ptr, max_size=True, base_byte_offset=_page_bytes
                    )
                    v_tile_rsrc = buffer_ops.create_buffer_resource(
                        v_ptr, max_size=True, base_byte_offset=_page_bytes
                    )
                else:
                    k_tile_rsrc, v_tile_rsrc = k_rsrc, v_rsrc
                    tile_org = tile_org + fx.Int32(_page) * stride_kb
            else:
                k_tile_rsrc, v_tile_rsrc = k_rsrc, v_rsrc
                tile_org = kv_base + tile_start * stride_km

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
                        v_tile_rsrc,
                        tile_org + _tok * stride_km + _col,
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
            # qk[t][st] reg e -> score[m=grp*4+e, tok=st*16+tok_lane] for M-tile t.
            # The K fragment is indexed by lane only, not by M, so it is loaded
            # once and fed to every M-tile: extra query tokens cost MFMA issue,
            # not memory traffic.
            qk_st = [[zero_v4 for _ in range(N_SUBTILE)] for _ in range(_M_TILES)]
            for st in range_constexpr(N_SUBTILE):
                for g in range_constexpr(_QK_GRP):
                    # Offset within the tile; tile_org already carries the page (paged)
                    # or the (b, g, tile_start) origin (dense).
                    k_tok_in_tile = fx.Int32(st * MFMA_N) + tok_lane
                    k_off = (
                        tile_org
                        + k_tok_in_tile * stride_km
                        + fx.Int32(g * MFMA_K_QK)
                        + grp * fx.Int32(8)
                    )
                    k8 = buffer_ops.buffer_load(
                        k_tile_rsrc, k_off, vec_width=8, dtype=_FX_KV
                    )
                    for t in range_constexpr(_M_TILES):
                        qk_st[t][st] = _mfma(
                            T.vec(4, T.f32),
                            [q_frags[t][g], k8, qk_st[t][st], 0, 0, 0],
                        )

            # ── Online softmax, per M row (reg e); tile max via dpp over tok_lane ──
            # The dpp_xor(1,2,4,8) reduction is over tok_lane (the N axis), which
            # packing does not touch: each M row still reduces over its tokens.
            new_max = [[] for _ in range(_M_TILES)]
            alpha = [[] for _ in range(_M_TILES)]
            for t in range_constexpr(_M_TILES):
                for e in range_constexpr(4):
                    # Bottom-right causal: query token i attends to
                    # [0, seqlen_kv - Sq + i + 1). At Sq=1 this is t_full, so
                    # end_m collapses to t_end and matches the unpacked kernel.
                    m_idx = grp * fx.Int32(4) + fx.Int32(e)
                    qtok_g = fx.Int32(t * _T_PACK) + m_idx // fx.Int32(_RATIO)
                    causal_end = t_full - fx.Int32(_SQ) + qtok_g + fx.Int32(1)
                    end_m = fx.Int32(
                        arith.select(
                            arith.unwrap(t_end < causal_end),
                            arith.unwrap(t_end),
                            arith.unwrap(causal_end),
                        )
                    )
                    loc = fx.Float32(c_neginf)
                    for st in range_constexpr(N_SUBTILE):
                        s = fx.Float32(
                            vector.extract(
                                qk_st[t][st], static_position=[e], dynamic_position=[]
                            )
                        )
                        s = fx.Float32(
                            arith.mulf(arith.unwrap(s), arith.unwrap(softmax_scale))
                        )
                        # mask tokens past this row's causal / split bound
                        tok_abs = tile_start + fx.Int32(st * MFMA_N) + tok_lane
                        ok = tok_abs < end_m
                        s = fx.Float32(
                            arith.select(arith.unwrap(ok), arith.unwrap(s), c_neginf)
                        )
                        loc = _mxf(loc, s)
                        qk_st[t][st] = vector.insert(
                            arith.unwrap(s),
                            qk_st[t][st],
                            static_position=[e],
                            dynamic_position=[],
                        )
                    for sh in (1, 2, 4, 8):
                        loc = _mxf(loc, dpp_xor_f32(loc, sh))
                    nm = _mxf(rmax[t][e], loc)
                    new_max[t].append(nm)
                    a = _exp2_fast(
                        fx.Float32(
                            arith.mulf(
                                arith.subf(arith.unwrap(rmax[t][e]), arith.unwrap(nm)),
                                arith.constant(LOG2E, type=T.f32),
                            )
                        )
                    )
                    alpha[t].append(a)

            # P = exp2((score - new_max)*log2e); write to LDS[m_tile][m, tok];
            # accumulate sum. Each M-tile owns its own _P_LDS plane.
            for t in range_constexpr(_M_TILES):
                tile_sum = [fx.Float32(c_zero) for _ in range(4)]
                for e in range_constexpr(4):
                    m_idx = grp * fx.Int32(4) + fx.Int32(e)
                    for st in range_constexpr(N_SUBTILE):
                        s = fx.Float32(
                            vector.extract(
                                qk_st[t][st], static_position=[e], dynamic_position=[]
                            )
                        )
                        p = _exp2_fast(
                            fx.Float32(
                                arith.mulf(
                                    arith.subf(
                                        arith.unwrap(s), arith.unwrap(new_max[t][e])
                                    ),
                                    arith.constant(LOG2E, type=T.f32),
                                )
                            )
                        )
                        # masked lanes gave s=-inf -> p=0
                        p = fx.Float32(
                            arith.select(
                                arith.unwrap(new_max[t][e]) > c_neginf,
                                arith.unwrap(p),
                                c_zero,
                            )
                        )
                        tile_sum[e] = fx.Float32(
                            arith.addf(arith.unwrap(tile_sum[e]), arith.unwrap(p))
                        )
                        tok = fx.Int32(st * MFMA_N) + tok_lane
                        vector.store(
                            fx.Vector.from_elements(
                                [arith.unwrap(p)], dtype=fx.Float32
                            ),
                            p_lds,
                            [
                                fx.Index(
                                    fx.Int32(t * _P_LDS)
                                    + m_idx * fx.Int32(TILE_N)
                                    + tok
                                )
                            ],
                        )
                for e in range_constexpr(4):
                    for sh in (1, 2, 4, 8):
                        tile_sum[e] = fx.Float32(
                            arith.addf(
                                arith.unwrap(tile_sum[e]),
                                arith.unwrap(dpp_xor_f32(tile_sum[e], sh)),
                            )
                        )
                    rsum[t][e] = fx.Float32(
                        arith.addf(
                            arith.mulf(
                                arith.unwrap(alpha[t][e]), arith.unwrap(rsum[t][e])
                            ),
                            arith.unwrap(tile_sum[e]),
                        )
                    )
                    rmax[t][e] = new_max[t][e]

            # Write the (already-loaded) V vec8s into the LDS transpose layout; the
            # barrier below covers both P writes and these V writes before PV.
            for _r in range_constexpr(_NUM_DMA_V):
                _sp = buffer_ops.create_llvm_ptr(
                    fx.Int64(_v8dst[_r] * fx.Index(2)), address_space=3
                )
                _llvm.StoreOp(_v8s[_r], _sp, alignment=16)

            gpu.barrier()

            # ── PV: A=P[m,tok] (LDS), B=V[tok,d] -> C[m,d]; rescale acc by alpha ──
            for t in range_constexpr(_M_TILES):
                for dpass in range_constexpr(_DN):
                    for e in range_constexpr(4):
                        acc[t][dpass * 4 + e] = fx.Float32(
                            arith.mulf(
                                arith.unwrap(acc[t][dpass * 4 + e]),
                                arith.unwrap(alpha[t][e]),
                            )
                        )
            # A-frag P: lane l -> P[t][m = tok_lane, tok = grp*8 + 0..7]
            p_frags = []
            for t in range_constexpr(_M_TILES):
                p_vals = []
                for j in range_constexpr(8):
                    pv = fx.Vector.load(
                        T.vec(1, T.f32),
                        p_lds,
                        [
                            fx.Index(
                                fx.Int32(t * _P_LDS)
                                + tok_lane * fx.Int32(TILE_N)
                                + grp * fx.Int32(8)
                                + fx.Int32(j)
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
                p_frags.append(p_frag)

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
                # v_frag is indexed by lane only, so one read feeds every M-tile.
                v_frag = vector.shuffle(_vlo, _vhi, [0, 1, 2, 3, 4, 5, 6, 7])
                for t in range_constexpr(_M_TILES):
                    c_in = zero_v4
                    for e in range_constexpr(4):
                        c_in = vector.insert(
                            arith.unwrap(acc[t][dpass * 4 + e]),
                            c_in,
                            static_position=[e],
                            dynamic_position=[],
                        )
                    c_out = _mfma(T.vec(4, T.f32), [p_frags[t], v_frag, c_in, 0, 0, 0])
                    for e in range_constexpr(4):
                        acc[t][dpass * 4 + e] = fx.Float32(
                            vector.extract(
                                c_out, static_position=[e], dynamic_position=[]
                            )
                        )

            gpu.barrier()  # P_LDS reused next tile

            # A comprehension, not a `for` statement: the AST rewriter turns bare
            # `for ... in range(...)` inside a kernel into a dynamic scf.for, which
            # cannot carry a Python list.
            state_out = [
                x
                for t in range(_M_TILES)
                for x in (
                    [arith.unwrap(rmax[t][i]) for i in range(4)]
                    + [arith.unwrap(rsum[t][i]) for i in range(4)]
                    + [arith.unwrap(acc[t][i]) for i in range(_N_ACC)]
                )
            ]
            results = yield state_out

        f_max = [
            [fx.Float32(results[t * _TILE_STATE + i]) for i in range(4)]
            for t in range(_M_TILES)
        ]
        f_sum = [
            [fx.Float32(results[t * _TILE_STATE + 4 + i]) for i in range(4)]
            for t in range(_M_TILES)
        ]
        f_acc = [
            [fx.Float32(results[t * _TILE_STATE + 8 + i]) for i in range(_N_ACC)]
            for t in range(_M_TILES)
        ]

        # ── Epilogue: normalize + store per (query token, head) ──
        # m = grp*4+e decodes as qtok = m // ratio, head = m % ratio; d =
        # dpass*16+tok_lane is always < _HEAD, so no d guard. `head < ratio` is
        # now structural, but qtok can overrun _SQ when _SQ < _M_TILES*_T_PACK,
        # so that is the guard.
        # Split-K partials fold the query axis into the head axis, giving
        # [B, G, SK, _SQ*H_q(, D)] -- pa_decode_reduce is generic over it.
        _nhq_eff = num_hq * fx.Int32(_SQ)
        for t in range_constexpr(_M_TILES):
            for e in range_constexpr(4):
                m_idx = grp * fx.Int32(4) + fx.Int32(e)
                head = m_idx % fx.Int32(_RATIO)
                qtok = fx.Int32(t * _T_PACK) + m_idx // fx.Int32(_RATIO)
                head_abs = hq_base + head
                row_ok = (qtok < fx.Int32(_SQ)) & (head_abs < num_hq)
                safe_sum = fx.Float32(
                    arith.select(
                        arith.unwrap(f_sum[t][e]) > c_zero,
                        arith.unwrap(f_sum[t][e]),
                        c_one,
                    )
                )
                inv = rcp_f32(safe_sum)
                if const_expr(_SPLIT):
                    _pm_base = (
                        b_idx * (num_g * split_total * _nhq_eff)
                        + g_idx * (split_total * _nhq_eff)
                        + split_idx * _nhq_eff
                        + qtok * num_hq
                        + head_abs
                    )
                    _po_base = _pm_base * fx.Int32(_HEAD)
                    if row_ok:
                        for dpass in range_constexpr(_DN):
                            d = fx.Int32(dpass * MFMA_N) + tok_lane
                            buffer_ops.buffer_store(
                                arith.unwrap(f_acc[t][dpass * 4 + e]),
                                out_rsrc,
                                _po_base + d,
                            )
                        if tok_lane == fx.Int32(0):
                            buffer_ops.buffer_store(
                                arith.unwrap(f_max[t][e]), pm_rsrc, _pm_base
                            )
                            buffer_ops.buffer_store(
                                arith.unwrap(f_sum[t][e]), ps_rsrc, _pm_base
                            )
                else:
                    out_base = (
                        b_idx * stride_qb
                        + qtok * stride_qs
                        + g_idx * stride_qg
                        + head_abs * stride_qh
                    )
                    inv_raw = arith.unwrap(inv)
                    if row_ok:
                        for dpass in range_constexpr(_DN):
                            d = fx.Int32(dpass * MFMA_N) + tok_lane
                            val = fx.Float32(
                                arith.mulf(
                                    arith.unwrap(f_acc[t][dpass * 4 + e]), inv_raw
                                )
                            )
                            out_val = _FX_OUT(arith.unwrap(val))
                            buffer_ops.buffer_store(
                                arith.unwrap(out_val), out_rsrc, out_base + d
                            )

    return pa_decode_gfx950_kernel, alloc


@functools.lru_cache(maxsize=256)
def _make_gfx950_jit_launcher(
    head_size,
    kv_dtype_str,
    out_dtype_str,
    split_k,
    paged=False,
    page_size=0,
    gqa_ratio=MFMA_M,
    seqlen_q=1,
    kv_i64=False,
):
    kernel, _alloc = compile_pa_decode_gfx950(
        head_size=head_size,
        kv_dtype_str=kv_dtype_str,
        output_dtype_str=out_dtype_str,
        split_k=split_k,
        paged=paged,
        page_size=page_size,
        gqa_ratio=gqa_ratio,
        seqlen_q=seqlen_q,
        kv_i64=kv_i64,
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
        bt_ptr,
        stride_qb,
        stride_qs,
        stride_qg,
        stride_qh,
        stride_kb,
        stride_km,
        stride_kg,
        stride_kh,
        stride_btb,
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
            bt_ptr,
            stride_qb,
            stride_qs,
            stride_qg,
            stride_qh,
            stride_kb,
            stride_km,
            stride_kg,
            stride_kh,
            stride_btb,
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
    Q,
    K,
    V,
    seq_positions,
    softmax_scale,
    split_k=0,
    output_dtype=None,
    block_table=None,
    page_size=0,
    kv_max=None,
):
    """Head-packed MFMA decode. One CTA per KV head packs its GQA group onto
    the MFMA M-axis. Falls back to the generic kernel for ratio>16 or non-gfx950.

    Dense mode (``block_table is None``): K/V are ``[B, KV_MAX, G, H_kv, D]``.

    Paged mode: K/V are ``[num_pages, page_size, H_kv, D]`` (MSLK "linear" layout)
    and ``block_table`` is ``[B, max_pages_per_seq]`` int32. ``kv_max`` must be
    supplied by the caller -- deriving it from ``seq_positions`` would be a
    device->host sync and is illegal under CUDA-graph capture.

    Q is made contiguous here: the single-split epilogue addresses the output
    with Q's strides, so a non-contiguous Q (a slice of a fused QKV, say) would
    write to the wrong place for every batch element after the first.
    """
    from mslk.flydsl.jit import run_compiled

    from .pa_decode_dense import auto_split_k_hp

    paged = block_table is not None
    Q = Q.contiguous()
    B, Sq, G, H_q, D = Q.shape
    if paged:
        H_kv = K.shape[2]
        if kv_max is None:
            raise ValueError("pa_decode_gfx950_launch: paged mode requires kv_max")
        KV_MAX = int(kv_max)
    else:
        _, KV_MAX, _, H_kv, _ = K.shape
    ratio = H_q // H_kv if H_kv > 0 else 0
    # M holds ratio*Sq (qtok, head) pairs in MFMA_M slots, so for Sq > 1 ratio
    # must divide MFMA_M for the tiling to be even, and Sq is capped by
    # max_m_tiles(D) tiles. A single query token fits any ratio <= MFMA_M.
    t_pack = MFMA_M // ratio if ratio and MFMA_M % ratio == 0 else 0
    ok = (
        H_kv > 0
        and H_q % H_kv == 0
        and 1 <= ratio <= MFMA_M
        and (Sq == 1 or 1 <= Sq <= t_pack * max_m_tiles(D))
        # Sq>1 folds (qtok, head) in the split-K partials, which only matches the
        # [B, Sq, G, H_q, D] output layout when G == 1. It also applies a
        # bottom-right causal bound per query token, which is the paged caller's
        # contract but not the dense decode op's -- so keep dense at Sq=1.
        and (Sq == 1 or (G == 1 and paged))
        and get_rocm_arch().startswith("gfx950")
        and K.dtype in (torch.float16, torch.bfloat16)
        and D % MFMA_K_QK == 0
        and (not paged or page_size % TILE_N == 0)
    )
    if not ok:
        if paged:
            # No generic paged fallback exists; the caller must keep its own path.
            raise ValueError(
                f"pa_decode_gfx950_launch: unsupported paged config "
                f"(ratio={ratio}, D={D}, dtype={K.dtype}, page_size={page_size})"
            )
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
    out = torch.empty((B, Sq, G, H_q, D), dtype=output_dtype, device=Q.device)
    sq = Q.stride()
    dev = Q.device
    # Split-K partials fold the query axis into the head axis, so the combine
    # kernel (which is generic over that axis) needs no changes.
    HQ_EFF = Sq * H_q
    ks = K.stride()
    if paged:
        # (page, token-in-page, unused, kv_head) — see the kernel signature comment.
        k_strides = (ks[0], ks[1], 0, ks[2])
        bt = (
            block_table
            if block_table.dtype == torch.int32
            else block_table.to(torch.int32)
        )
        bt_stride = bt.stride(0)
    else:
        k_strides = (ks[0], ks[1], ks[2], ks[3])
        bt = torch.empty(0, dtype=torch.int32, device=dev)
        bt_stride = 0
    n_cta_base = B * G * H_kv
    # Bytes addressed from the base pointer, so strided views of a larger cache
    # count what they can actually reach.
    kv_span = max(
        1 + sum((n - 1) * st for n, st in zip(t.shape, t.stride())) for t in (K, V)
    )
    kv_i64 = paged and kv_span * K.element_size() >= 2**32
    # Thread the live stream into .launch so the kernel is captured under CUDA graphs
    # (a default-stream launch would capture empty).
    stream = torch.cuda.current_stream()
    if split_k == 1:
        dummy = torch.empty(0, dtype=torch.float32, device=dev)
        launcher = _make_gfx950_jit_launcher(
            D, kv_str, out_str, 1, paged, page_size if paged else 0, ratio, Sq, kv_i64
        )
        run_compiled(
            launcher,
            out,
            dummy,
            dummy,
            Q,
            K,
            V,
            seq_positions,
            bt,
            sq[0],
            sq[1],
            sq[2],
            sq[3],
            k_strides[0],
            k_strides[1],
            k_strides[2],
            k_strides[3],
            bt_stride,
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
        po = torch.empty((B, G, split_k, HQ_EFF, D), dtype=torch.float32, device=dev)
        pm = torch.empty((B, G, split_k, HQ_EFF), dtype=torch.float32, device=dev)
        ps = torch.empty((B, G, split_k, HQ_EFF), dtype=torch.float32, device=dev)
        launcher = _make_gfx950_jit_launcher(
            D,
            kv_str,
            "f32",
            split_k,
            paged,
            page_size if paged else 0,
            ratio,
            Sq,
            kv_i64,
        )
        run_compiled(
            launcher,
            po,
            pm,
            ps,
            Q,
            K,
            V,
            seq_positions,
            bt,
            sq[0],
            sq[1],
            sq[2],
            sq[3],
            k_strides[0],
            k_strides[1],
            k_strides[2],
            k_strides[3],
            bt_stride,
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
        if Sq == 1:
            red_out = out.squeeze(1)  # [B, G, H_q, D]
        else:
            # The partials fold the query axis into the head axis as
            # qtok*H_q + head. `out` is [B, Sq, G, H_q, D], so that folded view
            # matches memory only when G == 1 -- enforced in the `ok` gate above.
            red_out = out.view(B, G, HQ_EFF, D)
        pa_decode_reduce(po, pm, ps, red_out, stream=stream)
    return out
