# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""FMHA backward (fused dQ + dV + dK) kernel for gfx950 (CDNA4), occupancy=1.

Uses the trload pipeline: Q/K/V/dO are loaded once and S/dP/P/dS are computed
once for all three gradients, using gfx950's hardware ds_read_tr16_b64 LDS
transpose in place of scalar-gather transposes.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import arith, const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import Vector as Vec
from flydsl.expr.utils.arith import _to_raw as _raw, ArithValue
from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr
from mslk.attention.flydsl.fmha_bwd_mfma import (
    _ds_read_tr_v4,
    _grad_ds_unscaled,
    _LOG2E,
    _softmax_p,
    dtype_to_elem_type,
    WARP_SIZE,
)

_GFX950_LDS_BYTES = 160 * 1024


def gfx950_lds_bytes(
    D: int, BLOCK_M: int, BLOCK_N: int, use_trload: bool, alias: bool = True
) -> int:
    """Static LDS footprint for the gfx950 dqdkdv buffer set (bytes).

    Group A = prologue K/V; Group B = per-m-tile Q/dO/dS/LSE/D. With register
    residency the two groups have disjoint lifetimes and are UNIONED onto the
    same LDS (footprint = max of stage groups); `alias=False` sums them instead
    (unused by any current caller; kept for the max-vs-sum contrast).
    """
    lds_mpad = BLOCK_M + 8
    q_stride = (D + 8) if use_trload else (D + 2)
    k_stride = (D + 8) if use_trload else D
    bf16 = 2
    f32 = 4
    group_a = BLOCK_N * k_stride * bf16 * 2  # lds_k + lds_v
    group_b = (
        BLOCK_M * q_stride * bf16 * 4  # lds_q + lds_do (double-buffered)
        + BLOCK_N * lds_mpad * bf16  # lds_ds
        + BLOCK_M * f32 * 2  # lds_lse + lds_dm
    )
    return max(group_a, group_b) if alias else (group_a + group_b)


def gfx950_trload_fits(D: int, BLOCK_M: int, BLOCK_N: int, gpu_arch: str) -> bool:
    return (
        gpu_arch.startswith("gfx950")
        and gfx950_lds_bytes(D, BLOCK_M, BLOCK_N, True, alias=True) <= _GFX950_LDS_BYTES
    )


def gfx950_tile_defaults(D: int, N: int, gpu_arch: str) -> tuple[int, int]:
    """Trload pipeline tile table, LDS-clipped for gfx950 buffers.

    Two tile shapes selected by seqlen_k: kM0=32, kN0=128 (small seqlen, N<384)
    and kM0=16, kN0=192 (large seqlen, N>=384 -- the shape that matters for real
    training workloads, e.g. N=2048).
    """
    is_gfx950 = gpu_arch.startswith("gfx950")
    if is_gfx950:
        if N >= 384 and D in (128, 256):
            block_m, block_n = 16, 192
        else:
            block_m = 32 if D in (64, 128, 256) else 64
            block_n = 128 if N >= 128 else 64
    else:
        block_m = 32 if D >= 128 or D >= 96 else 64
        block_n = 64
    if is_gfx950 and not gfx950_trload_fits(D, block_m, block_n, gpu_arch):
        if block_n > 64:
            block_n = 64
    return block_m, block_n


def compile_fmha_bwd_dqdkdv_mfma_gfx950(
    *,
    D: int = 64,
    dtype_str: str = "bf16",
    BLOCK_M: int = 32,
    BLOCK_N: int = 128,
    scale: float = None,
    use_pipeline: bool = True,  # unused, kept for API compat
    use_lds_reduce: bool = False,
    use_trload: bool | None = None,
    causal: bool = False,
    heads_per_kv: int = 1,
    varlen: bool = False,
    gpu_arch: str = "gfx950",
    deterministic: bool = False,
    ck_scope_dvdk: bool = False,
):
    """FUSED dQ + dV + dK in one N-tile-gridded kernel (gfx950/CDNA4 only),
    using the trload pipeline: Q/K/V/dO are loaded once and S/dP/P/dS are
    computed once for all three gradients.

    dV/dK are N-tile-unique for MHA -> register-accumulated, plain store.
    For GQA, the grid covers query heads; multiple query-head blocks contribute
    to the same compact KV-head dV/dK row, so those partials are atomic-added.
    dQ[m,d] = sum_n dS[m,n]*K[n,d] is SHARED across N-tile blocks -> each block
    contributes a partial via atomic-add into an f32 dQ scratch. A separate
    convert kernel casts the f32 scratch to the output dtype (same as the
    standalone dq path uses f32).

    dQ's wave assignment is DECOUPLED from GEMM1/2/3's N-split (GEMM4 warp
    layout 1x4x1): all NUM_WAVES waves cover the SAME m_sub rows and each owns
    its own D-subtile (GEMM4_DQ_D_SUBS_PER_WAVE contiguous 32-col slices, base =
    wave*GEMM4_DQ_D_SUBS_PER_WAVE), sweeping the ENTIRE n range (WAVE_N_TILES*32
    rows of lds_ds/lds_k, both already block-resident) internally via an extra
    n_sub_iter loop. Every wave therefore produces one COMPLETE,
    already-fully-summed dq_acc and fires its own atomic directly -- no
    cross-wave reduction step exists. `use_lds_reduce` is a no-op (kept as a
    parameter only for call-site/test compatibility): the LDS-combine-then-
    single-atomic step it used to gate no longer exists, since this design never
    produces a partial sum needing combine in the first place. When NUM_WAVES
    exceeds the D-subtile count, the excess waves idle during the dQ step
    (own an out-of-range D-subtile, discarded).

    KT/qt/dot/dQ-dS paths use ds_read_tr16_b64 (the gfx950 CDNA4 hardware LDS
    transpose) instead of a scalar gather. Requires EVEN LDS strides (D+8).
    GEMM1 Q/dO A-operands stay plain LDS vector loads (row-major, no transpose).

    causal (simple top-left-aligned only):
    masks P/dS to 0 where n_row_abs > m_row_abs, in addition to the seqlen bounds.

    GQA (heads_per_kv > 1): grid over QUERY heads (`B*H*num_N_tiles`), while
    K/V are addressed by `head_idx // heads_per_kv`. This preserves grid
    parallelism at BLOCK_N=128 (e.g. d128_gqa has 16*64 blocks, not 16*8).
    Since dV/dK are compact KV-head outputs shared by the heads_per_kv query
    heads, their per-query-head partials use atomic-add into the pre-zeroed f32
    dV/dK buffers. dQ remains query-head-unique and atomic-adds across N tiles.

    varlen (non-causal or causal BlockDiagonalMask, mirrors compile_fmha_bwd_dvdk_mfma):
    B collapses to 1 (Q/K/V/dO physically concatenated along M/N, no padding);
    grid_x is sized off max_seqlen_k (seq_N, a host constant) instead of
    per-tensor N. q_start/k_start (packed-row-offset bases) and
    this_seqlen_q/this_seqlen_k (per-batch REAL length) come from a runtime
    seqstart_q/seqstart_k lookup instead of the globally-uniform
    batch_idx*seq_M_idx/seq_N_idx. Two new runtime fx.Tensor params,
    seqstart_q/seqstart_k (int32, shape [B_logical+1]), plus total_m (LSE/D_vec's
    packed row pitch, distinct from seq_M == max_seqlen_q).

    Returns:
        launch_fn(Q, K, V, dO, dV, dK, dQ_f32, LSE, D_vec, B, M, N, H, n_M_tiles,
                  q_stride_m, kv_stride_n, do_stride_m, seqstart_q, seqstart_k,
                  total_m, stream)
          dQ_f32 : [B*M*H*D, 1] float32 scratch (zero it before launch; convert after)
          dV, dK : [B*N*Hkv*D, 1] float32 (always contiguous output; Hkv = H // heads_per_kv).
            Under ck_scope_dvdk=True (see that kwarg's definition below), dV/dK
            are instead PER QUERY HEAD, uncompacted: [B*N*H*D, 1] -- the caller
            must reduce over the heads_per_kv group to recover the true gradient.
          q_stride_m/kv_stride_n/do_stride_m: row pitch in ROW units
            (real_elem_stride(dim=1) // D) for possibly-non-contiguous Q/K,V/dO
            inputs; H (or Hkv for kv_stride_n) for contiguous BMHK.
          seqstart_q/seqstart_k : (varlen=True only) int32 [B_logical+1] cumulative
            offset tensors; M/N become max_seqlen_q/max_seqlen_k.
    """
    import math as _pm
    import os as _os

    if scale is None:
        scale = 1.0 / _pm.sqrt(D)
    # BLOCK_N must divide evenly into 32-row bands (the K/V prologue load's ceil-div
    # band-per-wave loop and GEMM4's WAVE_N_TILES both operate at 32-row granularity).
    # 128 and 192 are the two trload tile shapes (kN0=128 small-seqlen, kN0=192
    # large-seqlen/N>=384); BLOCK_N//32 need not divide NUM_WAVES (GEMM4's
    # WAVE_N_TILES is a per-wave internal loop count; GEMM1/2's CK_N_GROUPS and the
    # K/V prologue load's N_BANDS_PER_WAVE are both ceil-divs).
    assert BLOCK_N in (32, 64, 128, 192), (
        f"requires BLOCK_N in (32,64,128,192), got {BLOCK_N}"
    )
    assert D % 32 == 0, f"requires D a multiple of 32 (wave D-subtile width), got D={D}"
    assert D % 16 == 0
    # gfx950-only: this kernel now implements the trload path exclusively
    # (the gfx942 K8-MFMA fallback and the non-trload scalar-gather transpose were
    # removed -- gfx950_tile_defaults never selects a (D,N) that needs them on
    # gfx950 in production). Non-gfx950 callers should use fmha_bwd_mfma.py instead.
    assert gpu_arch.startswith("gfx950"), (
        f"fmha_bwd_mfma_gfx950 requires gfx950 (got {gpu_arch}); "
        "use compile_fmha_bwd_dqdkdv_mfma from fmha_bwd_mfma.py for other archs"
    )
    assert use_trload is not False, (
        "use_trload=False (scalar-gather transpose) was removed; "
        "the gfx950 trload path is now the only path"
    )
    use_trload = True

    # IGLP sched_group_barrier masks
    _MFMA_MASK = 0x008
    _VMEM_MASK = 0x020
    _LDS_READ_MASK = 0x100
    _LDS_WRITE_MASK = 0x200
    # Per-query-head dV/dK scope: writes dV/dK per query head (uncompacted [B,N,H,D]) instead
    # of the GQA-combined [B,N,Hkv,D]. The caller must reduce across the
    # heads_per_kv group to recover the true gradient (see flydsl.py BwOp.apply).
    _use_ck_scope_dvdk = (
        ck_scope_dvdk or _os.environ.get("FMHA_CK_LOGIC_CK_SCOPE_DVDK", "0") == "1"
    )

    # D==128 hardware-transpose paths: KT/dS/Q/dO use ds_read_tr16_b64 with
    # XOR-swizzled LDS layouts. Other D values fall back to scalar gather.
    _use_kt_tr = gpu_arch.startswith("gfx950") and D == 128
    _use_ck_kswz = _use_kt_tr
    _use_v_swz = gpu_arch.startswith("gfx950") and D == 128
    _use_ds_tr = gpu_arch.startswith("gfx950") and D == 128
    _use_ds_writer_wide = _use_ds_tr
    # Hand-emitted ds_write2st64_b64 for CK_N_GROUPS==3 (kn192); no-op at kn128.
    _use_ds_write2_asm = _use_ds_writer_wide
    # Swizzled lds_q/lds_do for bijective transpose reads (D==128 only).
    _use_qdo_swz2 = gpu_arch.startswith("gfx950") and D == 128

    # GQA atomic dV/dK store when multiple query heads share a KV head.
    _dvdk_atomic = heads_per_kv > 1
    lds_b = gfx950_lds_bytes(D, BLOCK_M, BLOCK_N, True, alias=True)
    assert lds_b <= _GFX950_LDS_BYTES, (
        f"use_trload LDS overflow: {lds_b}B > {_GFX950_LDS_BYTES}B at "
        f"D={D} BLOCK_M={BLOCK_M} BLOCK_N={BLOCK_N}"
    )
    elem_dtype = dtype_to_elem_type(dtype_str)
    MFMA_K = 16
    MFMA_LK = 8
    K_STEPS = D // MFMA_K
    fm = arith.FastMathFlags.fast

    BLOCK_SIZE = 256
    NUM_WAVES = BLOCK_SIZE // WARP_SIZE  # 4
    # Query-tile height (kM0): 16-row tile at D=128/256, 32-row at D=32/64. Each
    # 16-row tile is exactly ONE MFMA-M tile (kM=16 for both 16x16x32 and
    # 16x16x16 TransposeC MFMAs), so BLOCK_M decomposes as M_SUBTILES 32-row
    # subtiles, each split into M_HALVES 16-row MFMA-M tiles. kM0=16 (BLOCK_M=16)
    # runs the m_half body once (MIterPerWarp=1); kM0=32 runs it twice.
    # Invariant: M_SUBTILES * M_HALVES == BLOCK_M // 16 (total 16-row M-tiles).
    M_SUBTILES = max(1, BLOCK_M // 32)  # 16->1, 32->1, 64->2
    M_HALVES = (BLOCK_M // 16) if BLOCK_M < 64 else 2  # 16->1, 32->2, 64->2
    WAVE_N_TILES = BLOCK_N // 32  # 2 (BLOCK_N=64 fixed)
    # K/V block-entry HBM->LDS prologue: each wave covers ceil(N_BANDS_TOTAL/NUM_WAVES)
    # contiguous 32-row bands (band_idx = wave*N_BANDS_PER_WAVE + band_iter, guarded
    # < N_BANDS_TOTAL) instead of assuming exactly one band/wave -- generalizes past
    # BLOCK_N=128 (4 bands, 4 waves, 1 each -> loop runs once, byte-identical to before)
    # to BLOCK_N=192 (6 bands, 4 waves -> waves 0-2 cover 2 bands each, wave 3 idle on
    # its 2nd iter).
    N_BANDS_TOTAL = BLOCK_N // 32
    N_BANDS_PER_WAVE = -(-N_BANDS_TOTAL // NUM_WAVES)
    # GEMM1/3 TransposeC: MIterPerWarp=N/16 acc slices; NWarp splits D/16 bands.
    CK_N_ITERS = BLOCK_N // 16
    # NIterPerWarp = kN0/(NWarp*16): 4 waves cover 64 n-cols per group, so
    # BLOCK_N=128 needs 2 n-groups (each group = 4 waves * 16). GEMM0/epilogue loop
    # over these so the full BLOCK_N of S/P/dS is computed, not just n in [0,64).
    CK_N_GROUPS = -(-BLOCK_N // (NUM_WAVES * 16))
    # _use_ds_write2_asm only applies at the exact CK_N_GROUPS==3 shape its
    # fuse-last-2 pairing and n_grp1->n_grp2 address delta were derived from
    # and probe-validated against (kn192).
    _use_ds_write2_asm = _use_ds_write2_asm and CK_N_GROUPS == 3
    CK_D_ITERS_PER_WAVE = -(-(D // 16) // NUM_WAVES)
    DV_DK_ACCS_PER_WAVE = CK_N_ITERS * CK_D_ITERS_PER_WAVE
    DV_DK_ACC_LANES = 4
    # N-split GEMM2 layout (rm1=4): wave W owns NSPLIT_MITER interleaved 16-row n-bands
    # (n = mIter*64 + W*16 + lane%16, mIter in [0,NSPLIT_MITER)) and sweeps the FULL D
    # (NSPLIT_NITER = D//16 d-subtiles of 16). acc index = mIter*NSPLIT_NITER + nIter.
    # NSPLIT_MITER == CK_N_GROUPS (MIterPerWarp for GEMM1); the total per-wave acc count
    # NSPLIT_MITER*NSPLIT_NITER equals the D-split DV_DK_ACCS_PER_WAVE (invariant), so the
    # accumulator lists / iter_args threading are unchanged -- only the partition axis and
    # the acc index formula differ.
    NSPLIT_MITER = CK_N_GROUPS
    NSPLIT_NITER = D // 16

    # dQ-specific wave assignment (decoupled from GEMM1/2/3's N-split) --
    # Gemm4BlockWarps=1x4x1 splits the OUTPUT (D) axis across all 4 warps and
    # has each warp sweep the FULL kN0 range internally, so every wave gets a
    # COMPLETE dq_acc directly with no cross-wave reduction needed. Excess
    # waves idle during the dQ step when NUM_WAVES exceeds the D-subtile count.
    # GEMM4 16x16x32: Gemm4BlockWarps splits D in 16-col subtiles (NWarp=4 at D=64).
    GEMM4_D_TOTAL_SUBS = D // 16
    GEMM4_DQ_D_SUBS_PER_WAVE = -(-GEMM4_D_TOTAL_SUBS // NUM_WAVES)
    # GEMM4 dS-transpose residency: the dS A-operand transpose slice is read
    # ONCE per n_sub_iter step and reused against EVERY D-subtile a warp owns
    # (zero LDS cost). This hoists the dS-transpose read OUTSIDE the d_iter
    # loop (caching all WAVE_N_TILES slices once, same pattern as
    # kt_gemm4_regs's residency cache) instead of re-reading the identical LDS
    # data once per d_iter. D > 64 only: GEMM4_DQ_D_SUBS_PER_WAVE==1 at D<=64
    # under 4 waves, so there is no redundancy to eliminate there.
    _use_ds4_residency = GEMM4_DQ_D_SUBS_PER_WAVE > 1

    LDS_MPAD = BLOCK_M + 8
    # odd stride (D+2): bank-conflict-free scalar scatter (default). trload needs
    # EVEN stride (D+8) so ds_read_tr16_b64 keeps 64-bit column alignment -- same
    # tradeoff as compile_fmha_bwd_dvdk_mfma.
    LDS_Q_STRIDE = (D + 8) if use_trload else (D + 2)
    LDS_Q_ELEMS = BLOCK_M * LDS_Q_STRIDE
    LDS_DO_ELEMS = BLOCK_M * LDS_Q_STRIDE
    LDS_DS_ELEMS = BLOCK_N * LDS_MPAD
    # K in [n,d] layout for the dQ contraction. use_trload also transpose-loads
    # this buffer (dQ's B-operand) via ds_read_tr16_b64, which needs the same
    # EVEN-stride/anti-power-of-2-bank-conflict padding as LDS_Q_STRIDE above;
    # D alone is already even (D%32==0) but is a power of 2 for the common
    # D=64/128/256 shapes, so pad it the same way.
    # lds_k holds plain [n,d] K. The GEMM4 B-operand (Kt) is read directly from
    # lds_k via _kt_pack_gemm4*: the GEMM4 B-operand (Kt) is read directly from
    # lds_k with no separate transposed buffer -- the same k_lds view used by
    # the plain K read is transposed on the fly via ds_read_tr16_b64.
    LDS_K_STRIDE = (D + 8) if use_trload else D
    LDS_K_ELEMS = BLOCK_N * LDS_K_STRIDE
    LDS_V_STRIDE = LDS_K_STRIDE
    LDS_V_ELEMS = BLOCK_N * LDS_V_STRIDE
    LDS_LSE_ELEMS = BLOCK_M
    LDS_DM_ELEMS = BLOCK_M

    # LDS footprint = max(stage groups), NOT a flat sum: with register residency,
    # the prologue K/V/KT LDS (Group A) is dead by the time the per-m-tile
    # Q/dO/QT/dOT/P/dS buffers (Group B) are written, so the two groups are
    # UNIONED onto the same base (a handoff gpu.barrier separates their
    # lifetimes). This is what lets D=128 fit BLOCK_N=128.
    allocator = SmemAllocator(
        None, arch=gpu_arch, global_sym_name="fmha_bwd_dqdkdv_mfma_gfx950_smem"
    )
    _union_base = allocator._align(allocator.ptr, 16)

    # ---- Group B: per-m-tile buffers ----
    _pb = _union_base
    lds_q_off = allocator._align(_pb, 16)
    _pb = lds_q_off + LDS_Q_ELEMS * 2 * 2  # double-buffered Q
    lds_do_off = allocator._align(_pb, 16)
    _pb = lds_do_off + LDS_DO_ELEMS * 2 * 2  # double-buffered dO
    lds_ds_off = allocator._align(_pb, 16)
    _pb = lds_ds_off + LDS_DS_ELEMS * 2
    # LSE_PREFETCH double-buffers lds_lse/lds_dm the same way lds_q/lds_do
    # already are (a *2 factor), so the prefetched NEXT tile's LSE/D can be
    # staged into the other slot while the CURRENT tile still reads its own --
    # see above. Trivial LDS cost
    # (BLOCK_M*4 extra bytes each at both real tile shapes, confirmed against
    # the gfx950 160KB budget before implementing).
    _lse_dm_bufs = 2
    lds_lse_off = allocator._align(_pb, 16)
    _pb = lds_lse_off + LDS_LSE_ELEMS * 4 * _lse_dm_bufs
    lds_dm_off = allocator._align(_pb, 16)
    _pb = lds_dm_off + LDS_DM_ELEMS * 4 * _lse_dm_bufs
    _groupB_end = _pb

    # ---- Group A: prologue-only K/V (overlaid on Group B when aliasing) ----
    _pa = _union_base
    lds_k_off = allocator._align(_pa, 16)
    _pa = lds_k_off + LDS_K_ELEMS * 2
    lds_v_off = allocator._align(_pa, 16)
    _pa = lds_v_off + LDS_V_ELEMS * 2
    _groupA_end = _pa

    allocator.ptr = max(_groupB_end, _groupA_end)

    @flyc.kernel(known_block_size=[BLOCK_SIZE, 1, 1])
    def fmha_bwd_dqdkdv_mfma_gfx950_kernel(  # noqa: F811
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
        batch_size: fx.Int32,
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
        # dQ_f32's real element count, needed to bound its buffer descriptor for
        # the atomic-add OOB offset-shift trick (see above
        # below). total_m_idx is the sum-over-batches row count under varlen
        # (B already collapsed to 1 there, see module docstring) but is only the
        # PER-BATCH seq_M_idx under non-varlen -- mirrors _lse_row's own
        # varlen/non-varlen row-pitch split, needs an explicit batch_size factor
        # in the non-varlen case that _lse_row doesn't (LSE is [B,H,M], dQ_f32 is
        # flat [B*M*H*D]).
        if const_expr(varlen):
            dq_total_elems_idx = total_m_idx * n_heads_idx * fx.Index(D)
        else:
            dq_total_elems_idx = (
                fx.Index(batch_size) * seq_M_idx * n_heads_idx * fx.Index(D)
            )
        num_N_tiles = (seq_N_idx + BLOCK_N - 1) // BLOCK_N
        # Compact K/V and dK/dV tensors are indexed by KV heads; the grid is query
        # head based, but the output row pitch for dK/dV remains Hkv = H/heads_per_kv.
        n_kv_heads_idx = n_heads_idx // heads_per_kv
        bid_idx = fx.Index(bid)
        n_bh_idx = bid_idx
        n_tile = n_bh_idx % num_N_tiles
        bh_idx = n_bh_idx // num_N_tiles
        # Per-query-head grid: blockIdx.y ranges over QUERY heads (num_head_q),
        # with K/V looked up via head_idx // heads_per_kv. This gives
        # B*H*num_N_tiles blocks.
        batch_idx = bh_idx // n_heads_idx
        head_idx = bh_idx % n_heads_idx
        kv_head_idx = head_idx // fx.Index(heads_per_kv)

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
        lane_mod_16 = fx.Index(lane % 16)
        lane_div_16 = fx.Index(lane // 16)

        def _qdo_swz(col, row):
            # Identity placeholder: the K-write's XOR-swizzle formula
            # (_ck_q_swz_off) is used instead wherever it applies (D==128); this
            # plain pass-through covers the remaining D values.
            return col

        dV_buf = fx.rocdl.make_buffer_tensor(dV)
        dK_buf = fx.rocdl.make_buffer_tensor(dK)
        LSE_buf = fx.rocdl.make_buffer_tensor(LSE)
        Dvec_buf = fx.rocdl.make_buffer_tensor(D_vec)

        copy_f32 = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.Float32)
        store_f32 = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.Float32)

        v8elem_type = Vec.make_type(MFMA_LK, elem_dtype)
        v4f32_type = Vec.make_type(4, fx.Float32)

        base_ptr = allocator.get_base()
        lds_q = SmemPtr(
            base_ptr, lds_q_off, elem_dtype.ir_type, shape=(LDS_Q_ELEMS * 2,)
        ).get()
        lds_do = SmemPtr(
            base_ptr, lds_do_off, elem_dtype.ir_type, shape=(LDS_DO_ELEMS * 2,)
        ).get()
        lds_ds = SmemPtr(
            base_ptr, lds_ds_off, elem_dtype.ir_type, shape=(LDS_DS_ELEMS,)
        ).get()
        lds_k = SmemPtr(
            base_ptr, lds_k_off, elem_dtype.ir_type, shape=(LDS_K_ELEMS,)
        ).get()
        lds_v = SmemPtr(
            base_ptr, lds_v_off, elem_dtype.ir_type, shape=(LDS_V_ELEMS,)
        ).get()
        lds_lse = SmemPtr(
            base_ptr,
            lds_lse_off,
            fx.Float32.ir_type,
            shape=(LDS_LSE_ELEMS * _lse_dm_bufs,),
        ).get()
        lds_dm = SmemPtr(
            base_ptr,
            lds_dm_off,
            fx.Float32.ir_type,
            shape=(LDS_DM_ELEMS * _lse_dm_bufs,),
        ).get()

        # Q/dO and K/V are possibly-non-contiguous user inputs (e.g. a packed-qkv
        # unbind view) -- row pitch is q_stride_m_idx/kv_stride_n_idx (in row
        # units), NOT necessarily n_heads_idx. dO can have a DIFFERENT row pitch
        # than Q -- separate helper (mirrors compile_fmha_bwd_dvdk_mfma).
        # GQA: Q-side rows (_q_row/_do_row/_lse_row/_dvec_row) are indexed by the
        # per-block query head. K/V (_kv_row) stay indexed by kv_head_idx, derived
        # from head_idx // heads_per_kv.
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

        # ck_scope_dvdk-only: per-query-head scope, writing dV/dK per QUERY
        # head, uncompacted, into a [B,N,H,D]-sized buffer (row pitch
        # n_heads_idx, indexed by head_idx -- NOT the KV-head combine
        # _kv_row_out performs). The GQA reduction across the heads_per_kv
        # query heads sharing a KV head must happen OUTSIDE this kernel.
        def _kv_row_out_per_qhead(kv_pos):
            return fx.Int32((k_start + kv_pos) * n_heads_idx + head_idx)

        # dQ is a freshly-allocated contiguous output -- always row pitch
        # n_heads_idx (NOT q_stride_m_idx), indexed by the per-q-head head_idx.
        def _q_row_out(q_pos, head_idx):
            return fx.Int32((q_start + q_pos) * n_heads_idx + head_idx)

        # LSE layout is [B,H,M] (batch-major, non-varlen) vs packed [1,H,sum_M]
        # under varlen (B collapses to 1 -- packed LSE convention).
        # NOT unifiable via a single q_start-relative formula since the head-axis
        # stride differs: seq_M_idx (non-varlen) vs total_m_idx (varlen).
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

        def _make_raw_f32_rsrc(tensor, num_records_bytes=None):
            base_ptr = _fly_d.extract_aligned_pointer_as_index(
                _ir_d.Type.parse("!llvm.ptr"), tensor
            )
            base_i64 = _llvm_d.PtrToIntOp(_T.i64, base_ptr).result
            lo = ArithValue(base_i64).trunci(_T.i32)
            hi = ArithValue(ArithValue(base_i64).shrui(fx.Int64(32))).trunci(_T.i32)
            size_val = (
                _bops._create_i32_constant(0xFFFFFFFF)
                if num_records_bytes is None
                else _raw(num_records_bytes)
            )
            return Vec.from_elements(
                [
                    lo,
                    hi,
                    size_val,
                    _bops._create_i32_constant(_bops._get_buffer_flags()),
                ],
                fx.Int32,
            ).ir_value()

        dq_rsrc = _make_raw_f32_rsrc(dQ_f32)
        # Deterministic dQ uses plain buffer_store (collision-free per-n_tile slot),
        # which needs the ptr<8> descriptor rather than the raw <4xi32> atomic rsrc.
        if const_expr(deterministic):
            dq_store_rsrc = _bops.create_buffer_resource(dQ_f32)
        # dV/dK's final store is an atomic-add whenever multiple blocks contribute a
        # partial sum for the same KV-head N-tile: GQA (heads_per_kv>1, per-query-head
        # grid). MHA keeps the plain deterministic store.
        if const_expr(_dvdk_atomic):
            dv_rsrc = _make_raw_f32_rsrc(dV)
            dk_rsrc = _make_raw_f32_rsrc(dK)
        # Wide plain-store path (see above): needs the
        # ptr<8> ordinary buffer_store descriptor (buffer_ops.buffer_store), distinct
        # from the raw <4xi32> atomic rsrc built above for the GQA-combine path.
        if const_expr(not (_dvdk_atomic and not _use_ck_scope_dvdk)):
            dv_store_rsrc = _bops.create_buffer_resource(dV)
            dk_store_rsrc = _bops.create_buffer_resource(dK)

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

        def _store_f32_vec4(rsrc, flat_col_base, vals):
            # One v4f32 buffer_store covering 4 contiguous elements (flat_col_base..+3) instead of 4
            # scalar _store_f32_row calls. Caller guarantees all 4 are in-bounds
            # (the group-level d_ok/n_ok check already covers r=0..3 uniformly).
            v = Vec.from_elements(vals, fx.Float32)
            _bops.buffer_store(v.ir_value(), rsrc, flat_col_base)

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

        # Deterministic mode: the non-deterministic path atomic-adds each N-tile
        # block's dQ partial into a shared buffer (_atomic_add_dq); the
        # deterministic path instead stores each partial into its own per-split
        # slot, then a separate
        # reduce sums the splits in a fixed order (bit-reproducible). Our N-tile-
        # major grid makes n_tile the natural split index, so dQ_f32 becomes a
        # [num_N_tiles, B*M*H*D] workspace, TILE-MAJOR/ELEMENT-MINOR (n_tile
        # outermost, flat_elem innermost) -- flat_elem is consecutive across lanes
        # within a lane-group, so keeping it the fastest-varying axis preserves a
        # wide coalesced store; an element-major layout (flat_elem*num_N_tiles+
        # n_tile) would space adjacent lanes' writes num_N_tiles*4 bytes apart,
        # forcing a scalar/strided store that gets worse as num_N_tiles grows with
        # seqlen. Slots are collision-free (a given (batch,head,m,d) is owned by
        # exactly one block per n_tile) so no atomics; the caller zero-inits the
        # workspace (causal leaves some slots unwritten) and reduces
        # dq_acc.view(num_N_tiles, B*M*H*D).sum(0) to recover dQ.
        def _store_dq(flat_elem, val_f32, is_valid=None):
            if const_expr(deterministic):
                row_idx = fx.Int32(n_tile * dq_total_elems_idx + fx.Index(flat_elem))
                _bops.buffer_store(val_f32, dq_store_rsrc, row_idx)
            else:
                _atomic_add_dq(flat_elem, val_f32)

        v4elem_type = Vec.make_type(MFMA_LK // 2, elem_dtype)

        def mfma_gemm4(a_pack, b_pack, c_acc):
            """GEMM4: mfma_f32_16x16x32 (K=32 contract per k4 slice). gfx950 only."""
            if const_expr(dtype_str == "bf16"):
                return rocdl.mfma_f32_16x16x32_bf16(
                    v4f32_type, [a_pack, b_pack, c_acc, 0, 0, 0]
                )
            return rocdl.mfma_f32_16x16x32_f16(
                v4f32_type, [a_pack, b_pack, c_acc, 0, 0, 0]
            )

        K16_LK = 4

        def mfma_gemm16_tc(a_pack, b_pack, c_acc):
            """GEMM1/3: mfma_f32_16x16x16 TransposeC (K=16 per step).

            TransposeC convention: hardware MFMA takes (b,a) when the
            API-level call is (a,b).
            """
            if const_expr(dtype_str == "bf16"):
                a_i16 = Vec(a_pack).bitcast(fx.Int16)
                b_i16 = Vec(b_pack).bitcast(fx.Int16)
                return rocdl.mfma_f32_16x16x16bf16_1k(
                    v4f32_type, [b_i16, a_i16, c_acc, 0, 0, 0]
                )
            return rocdl.mfma_f32_16x16x16f16(
                v4f32_type, [b_pack, a_pack, c_acc, 0, 0, 0]
            )

        def _pack_qdo_ck_k32(m_sub, m_half, k_iter, buf_elems, lds_arr, stride):
            # Q/dO A-operand pack: m=lane%16+m_half*16; d=(lane//16)*8+e. The MFMA_LK
            # contiguous d elements are contiguous in LDS, so read them as ONE wide
            # ds_read instead of MFMA_LK scalar gathers.
            m_free = lane_mod_16 + m_sub * 32 + m_half * 16
            d_base = lane_div_16 * MFMA_LK + k_iter * 32
            if const_expr(_use_qdo_swz2):
                # Swizzled lds_q/lds_do: d_base is always 8-aligned, so the
                # wide v8 read stays contiguous under the XOR swizzle.
                return Vec.load(
                    v8elem_type, lds_arr, [buf_elems + _ck_q_swz_off(m_free, d_base)]
                ).ir_value()
            d_base_swz = _qdo_swz(d_base, m_free)
            return Vec.load(
                v8elem_type, lds_arr, [buf_elems + m_free * stride + d_base_swz]
            ).ir_value()

        def _pack_kv_ck_k32(k_iter, lds_arr, stride, row_base, swz=False):
            # K/V B-operand pack: row=lane%16+row_base (row_base=wave*16);
            # d=(lane//16)*8+e. swz=True: lds_arr uses the XOR-swizzled layout
            # (_ck_k_swz_off), bijective with the matching swizzled write.
            row_free = lane_mod_16 + row_base
            d_base = lane_div_16 * MFMA_LK + k_iter * 32
            # The MFMA_LK=8 contiguous d-elements (e=0..MFMA_LK-1) are contiguous in LDS
            # under both K's swizzle and V's plain layout (same contiguity property
            # _pack_qdo_ck_k32 relies on for Q/dO) -- read them as ONE wide
            # ds_read instead of MFMA_LK separate scalar gathers, each of which
            # forced its own lgkmcnt dependency.
            off = (
                _ck_k_swz_off(row_free, d_base)
                if const_expr(swz)
                else (row_free * stride + d_base)
            )
            return Vec.load(v8elem_type, lds_arr, [off]).ir_value()

        # ---- K/V block-entry HBM -> LDS -> registers (prologue).
        # Each wave covers N_BANDS_PER_WAVE contiguous 32-row bands (ceil-div)
        # instead of assuming exactly one -- generalizes past BLOCK_N=128 to
        # BLOCK_N=192 without changing behavior at BLOCK_N=128.
        for band_iter in range_constexpr(N_BANDS_PER_WAVE):
            band_idx = wave * N_BANDS_PER_WAVE + band_iter
            band_in_range = band_idx < fx.Index(N_BANDS_TOTAL)
            band_safe = band_in_range.select(band_idx, fx.Index(0))
            n_global_band_base = n_start + band_safe * 32
            n_row_abs_kv = n_global_band_base + lane_mod_32
            n_valid_kv = band_in_range & (n_row_abs_kv < this_seqlen_k)
            n_safe_kv = n_valid_kv.select(n_row_abs_kv, this_seqlen_k - fx.Index(1))
            kv_row_g_pre = _kv_row(n_safe_kv)
            n_local_k_row_band = band_safe * 32 + lane_mod_32

            if band_in_range:
                for ks in range_constexpr(K_STEPS):
                    col_off = fx.Index(ks * MFMA_K) + lane_div_32 * MFMA_LK
                    k_vec = _load_global_vec_cv(k_rsrc, kv_row_g_pre, col_off)
                    if const_expr(_use_ck_kswz):
                        # KPack-split + XOR write layout (0-residual both tiles):
                        #   off = (d//64)*(BLOCK_N*64) + n*64 + ((d%64)//8 ^ (n&7))*8 + (d%64)%8
                        # v8 store covers one 8-elem block, contiguous under the swizzle.
                        # Bijective with the reads (_ck_k_swz_off). Same formula below.
                        n_w = n_local_k_row_band
                        d_hi = col_off // fx.Index(64)
                        d_in = col_off % fx.Index(64)
                        blk = (d_in // fx.Index(8)) ^ (n_w & fx.Index(7))
                        ck_off = (
                            d_hi * fx.Index(BLOCK_N * 64)
                            + n_w * fx.Index(64)
                            + blk * fx.Index(8)
                            + (d_in % fx.Index(8))
                        )
                        Vec(k_vec).store(lds_k, [ck_off])
                    else:
                        Vec(k_vec).store(
                            lds_k, [n_local_k_row_band * LDS_K_STRIDE + col_off]
                        )
                    v_vec = _load_global_vec_cv(v_rsrc, kv_row_g_pre, col_off)
                    if const_expr(_use_v_swz):
                        # Swizzled lds_v write -- same XOR formula as K's swizzled write.
                        n_vw = n_local_k_row_band
                        d_vhi = col_off // fx.Index(64)
                        d_vin = col_off % fx.Index(64)
                        v_blk = (d_vin // fx.Index(8)) ^ (n_vw & fx.Index(7))
                        v_ck_off = (
                            d_vhi * fx.Index(BLOCK_N * 64)
                            + n_vw * fx.Index(64)
                            + v_blk * fx.Index(8)
                            + (d_vin % fx.Index(8))
                        )
                        Vec(v_vec).store(lds_v, [v_ck_off])
                    else:
                        Vec(v_vec).store(
                            lds_v, [n_local_k_row_band * LDS_V_STRIDE + col_off]
                        )

        gpu.barrier()

        def _kt_pack_gemm4_ck_k32(wave_d_sub_16, n_sub_iter):
            # GEMM4 Kt B-operand scalar pack: d=lane%16+wave_d*16; n=(lane//16)*8+e.
            k_r = fx.make_rmem_tensor(MFMA_LK, elem_dtype)
            d_local = lane_mod_16 + wave_d_sub_16 * 16
            for e in range_constexpr(MFMA_LK):
                n_local = lane_div_16 * MFMA_LK + (n_sub_iter * 32 + e)
                k_sc = Vec.load(
                    Vec.make_type(1, elem_dtype),
                    lds_k,
                    [n_local * LDS_K_STRIDE + d_local],
                )[0]
                fx.memref_store(k_sc, k_r, e)
            return fx.memref_load_vec(k_r)

        def _ck_k_swz_off(n, d):
            # Swizzled lds_k element offset (0-residual both tiles):
            #   off = (d//64)*(BLOCK_N*64) + n*64 + ((d%64)//8 ^ (n&7))*8 + (d%64)%8
            # ds_read_tr addresses must index THIS layout (not plain n*STRIDE+d).
            d_hi = d // fx.Index(64)
            d_in = d % fx.Index(64)
            blk = (d_in // fx.Index(8)) ^ (n & fx.Index(7))
            return (
                d_hi * fx.Index(BLOCK_N * 64)
                + n * fx.Index(64)
                + blk * fx.Index(8)
                + (d_in % fx.Index(8))
            )

        def _ck_q_swz_off(m, d):
            # Swizzled lds_q/lds_do element offset -- same XOR template as
            # _ck_k_swz_off, with BLOCK_M/m substituted for BLOCK_N/n.
            # Probe-verified: 0 mismatches at both tile shapes, both Q and dO.
            d_hi = d // fx.Index(64)
            d_in = d % fx.Index(64)
            blk = (d_in // fx.Index(8)) ^ (m & fx.Index(7))
            return (
                d_hi * fx.Index(BLOCK_M * 64)
                + m * fx.Index(64)
                + blk * fx.Index(8)
                + (d_in % fx.Index(8))
            )

        def _ck_ds_off(m, n):
            # Swizzled lds_ds element offset (0-residual both tiles). Splits
            # M1->(M1_0=2,M1_1=2), N1->(N1_0=2,N1_1=8), one XOR transform on
            # (M1_0,N1_0) -- structurally like _ck_k_swz_off's KPack-split+XOR
            # but a DIFFERENT split (do not conflate the two formulas).
            N0_LEN = BLOCK_N // 16
            M1_0, M1_1, N1_0, N1_1, M2 = 2, 2, 2, 8, 4
            stride_M2 = 1
            stride_N1 = M2
            stride_M1 = N1_0 * N1_1 * M2
            stride_N0 = M1_0 * M1_1 * N1_0 * N1_1 * M2
            stride_M0 = N0_LEN * stride_N0

            m0_idx = m // fx.Index(16)
            m_rem = m % fx.Index(16)
            m1_0_idx = m_rem // fx.Index(8)
            m_rem2 = m_rem % fx.Index(8)
            m1_1_idx = m_rem2 // fx.Index(4)
            m2_idx = m_rem2 % fx.Index(4)

            n0_idx = n // fx.Index(16)
            n_rem = n % fx.Index(16)
            n1_0_idx = n_rem // fx.Index(8)
            n1_1_idx = n_rem % fx.Index(8)

            n1_0_phys = n1_0_idx ^ m1_0_idx

            return (
                m0_idx * fx.Index(stride_M0)
                + n0_idx * fx.Index(stride_N0)
                + (m1_0_idx * fx.Index(M1_1) + m1_1_idx) * fx.Index(stride_M1)
                + (n1_0_phys * fx.Index(N1_1) + n1_1_idx) * fx.Index(stride_N1)
                + m2_idx * fx.Index(stride_M2)
            )

        # The dS write2 fusion's n_grp1->n_grp2 address delta, in
        # ds_write2st64_b64 ST64 units (each = 64 elements * 8 bytes = 512 bytes):
        # algebraically derived from _ck_ds_off's fixed constants (the n_grp step
        # is NUM_WAVES*16=64 -> n0_idx advances by 4 -> stride_N0=256 elements ->
        # 4*256=1024 elements = 2048 bytes = 4 ST64 units) and confirmed constant
        # across every (m, n) combination (probe_ds_write2_handasm.py's offset1=4
        # case; see above).
        _DS_WRITE2_OFFSET1_ST64 = 4

        def _ds_write2st64_b64_asm(
            lds_elem_idx, lds_byte_base, data0_i32x2, data1_i32x2, offset1
        ):
            """Hand-emit `ds_write2st64_b64 vaddr, vdata0, vdata1 offset0:0 offset1:{offset1}`.

            Fuses two adjacent-n_grp wide dS stores into ONE machine instruction,
            bypassing the compiler's non-firing automatic `ds_write2` combine -- mirrors
            `_ds_read_tr16_b64_imm` (flash_attn_gfx950.py:44-55)'s escape-hatch
            pattern: a raw i32 LDS byte address (ONE shared base register for both
            64-bit data operands, per the ISA's `CalcDsAddr(ADDR, 0, 0)` semantics),
            `~{memory}` clobber + has_side_effects=True to block reordering/
            elimination. `lds_elem_idx`/`lds_byte_base` compose exactly like
            `_ds_read_tr_v4`'s address arithmetic (element index * dtype width in
            bytes + the buffer's static byte base). Probe-validated bit-exact:
            test/attention/fmha/probe_ds_write2_handasm.py.
            """
            from flydsl._mlir import ir as _ir_w2
            from flydsl._mlir.dialects import llvm as _llvm_w2

            byte_off = lds_elem_idx * 2 + lds_byte_base
            addr_i32 = fx.Int32(byte_off)
            _llvm_w2.inline_asm(
                _ir_w2.Type.parse("!llvm.void"),
                [_raw(addr_i32), _raw(data0_i32x2), _raw(data1_i32x2)],
                f"ds_write2st64_b64 $0, $1, $2 offset0:0 offset1:{int(offset1)}\n",
                "v,v,v,~{memory}",
                has_side_effects=True,
            )

        def _dq_ds_pack_gemm4_tr(m_sub, m_half, n_sub_iter):
            # GEMM4 dS A-operand via ds_read_tr16_b64 on the swizzled lds_ds
            # (replaces the scalar gather, ~16.9M conflicts). Probe-verified to
            # reproduce the correct register contents at both tile shapes.
            #   m_arg  = (lane%4)*4 + 16*m_half + 32*m_sub
            #   n_base = lane//4  (lo, at n_sub_iter's window);  +16 more (hi)
            # Output (m,n) per (lane,e) matches the scalar path's KPart formula.
            m_arg = (lane % fx.Index(4)) * fx.Index(4) + fx.Index(
                16 * m_half + 32 * m_sub
            )
            n_base = lane // fx.Index(4)
            lo_ds = _ck_ds_off(m_arg, n_base + fx.Index(32 * n_sub_iter))
            hi_ds = _ck_ds_off(m_arg, n_base + fx.Index(16 + 32 * n_sub_iter))
            ds_a = _ds_read_tr_v4(v4elem_type, lo_ds, lds_ds_off)
            ds_b = _ds_read_tr_v4(v4elem_type, hi_ds, lds_ds_off)
            return Vec(ds_a).shuffle(Vec(ds_b), [0, 1, 2, 3, 4, 5, 6, 7]).ir_value()

        def _kt_pack_gemm4_tr(wave_d_sub_16, n_sub_iter):
            # GEMM4 Kt B-operand via ds_read_tr16_b64 on the swizzled lds_k (replaces
            # the scalar gather ~34.7M conflicts). Probe-verified at D=128/BLOCK_N=128:
            #   n_arg = (lane//4) + 32*n_sub          (lo);  + 16 more (hi)
            #   d_arg = wave*16 + (lane%4)*4 + 64*j1
            # j1 (= d_iter = which KPack half) is recovered from wave_d_sub_16. The
            # hardware transpose redistributes these coarse corners to the KPack
            # residency (d = wave*16+lane%16+64*j1; n = 16*n16+(lane//16)*4+r).
            j1 = wave_d_sub_16 % fx.Index(GEMM4_DQ_D_SUBS_PER_WAVE)
            n_base = lane // fx.Index(4)
            d_arg = (
                wave * fx.Index(16)
                + (lane % fx.Index(4)) * fx.Index(4)
                + j1 * fx.Index(64)
            )
            lo_k = _ck_k_swz_off(n_base + fx.Index(32 * n_sub_iter), d_arg)
            hi_k = _ck_k_swz_off(n_base + fx.Index(32 * n_sub_iter + 16), d_arg)
            k_a = _ds_read_tr_v4(v4elem_type, lo_k, lds_k_off)
            k_b = _ds_read_tr_v4(v4elem_type, hi_k, lds_k_off)
            return Vec(k_a).shuffle(Vec(k_b), [0, 1, 2, 3, 4, 5, 6, 7]).ir_value()

        def _kt_pack_gemm4(wave_d_sub_16, n_sub_iter, ks):
            if const_expr(_use_kt_tr):
                return _kt_pack_gemm4_tr(wave_d_sub_16, n_sub_iter)
            return _kt_pack_gemm4_ck_k32(wave_d_sub_16, n_sub_iter)

        # K/V register residency: read every GEMM1 K/V pack once here
        # (indexed [k_iter*CK_N_GROUPS + n_grp]) and hold across the M-sweep, rather
        # than re-reading from lds_k/lds_v inside each m_tile.
        k_regs_ck = []
        v_regs_ck = []
        for k_iter in range_constexpr(D // 32):
            for n_grp in range_constexpr(CK_N_GROUPS):
                kv_row_base = n_grp * fx.Index(NUM_WAVES * 16) + wave * fx.Index(16)
                k_regs_ck.append(
                    _pack_kv_ck_k32(
                        k_iter, lds_k, LDS_K_STRIDE, kv_row_base, swz=_use_ck_kswz
                    )
                )
                v_regs_ck.append(
                    _pack_kv_ck_k32(
                        k_iter, lds_v, LDS_V_STRIDE, kv_row_base, swz=_use_v_swz
                    )
                )

        # KT residency for GEMM4 dQ: preload the K^T B-operand packs once (indexed
        # [d_iter*WAVE_N_TILES + n_sub_iter]) so the dQ loop no longer re-reads lds_k.
        kt_gemm4_regs = []
        for d_iter in range_constexpr(GEMM4_DQ_D_SUBS_PER_WAVE):
            wave_d_sub_16_raw = wave * GEMM4_DQ_D_SUBS_PER_WAVE + d_iter
            d_in_range_g4r = wave_d_sub_16_raw < fx.Index(GEMM4_D_TOTAL_SUBS)
            wave_d_sub_16_r = d_in_range_g4r.select(wave_d_sub_16_raw, fx.Index(0))
            for n_sub_iter in range_constexpr(WAVE_N_TILES):
                kt_gemm4_regs.append(_kt_pack_gemm4(wave_d_sub_16_r, n_sub_iter, 0))

        # LDS-union handoff: every wave has now read all of K/V/KT out of the Group-A
        # LDS into registers, so it is safe for the m-loop prologue below to start
        # overwriting that same physical LDS with the Group-B Q/dO buffers.
        gpu.barrier()

        def _qdo_read_k16_tr(m_sub, ks, wave_d, buf_elems, lds_byte_off):
            # GEMM2 B-operand via ds_read_tr16_b64 (16-col D band wave_d).
            tr_k_group = lane_mod_16 // 4
            tr_col_sub = lane % 4
            tr_col_half = wave_d % 2
            wave_d_sub_i = wave_d // 2
            m_base = m_sub * 32 + ks * 16 + lane_div_16 * 4 + tr_k_group
            d_col = wave_d_sub_i * 32 + tr_col_half * 16 + tr_col_sub * 4
            if const_expr(_use_qdo_swz2):
                # Swizzled lds_q/lds_do: m_base/d_col address the XOR-swizzled
                # layout instead of plain row-major.
                lo = buf_elems + _ck_q_swz_off(m_base, d_col)
            else:
                d_col_swz = _qdo_swz(d_col, m_base)
                lo = buf_elems + m_base * LDS_Q_STRIDE + d_col_swz
            a = _ds_read_tr_v4(v4elem_type, lo, lds_byte_off)
            out = fx.make_rmem_tensor(K16_LK, elem_dtype)
            for e in range_constexpr(K16_LK):
                fx.memref_store(Vec(a)[e], out, e)
            return fx.memref_load_vec(out)

        def _dq_ds_pack_gemm4_ck_k32(m_sub, m_half, n_sub_iter):
            # GEMM4 dS A-operand scalar pack: one v8 per lane, K=32.
            # m=lane%16+m_half*16; n=(lane//16)*8+e within the k4 window.
            # Under _use_kt_tr (D==128), re-paired to match the KT-tr B-operand's
            # contraction order -- both A and B must place the same n at each
            # register e for the MFMA to contract correctly.
            m_free = lane_mod_16 + m_sub * 32 + m_half * 16
            ds_r = fx.make_rmem_tensor(MFMA_LK, elem_dtype)
            for e in range_constexpr(MFMA_LK):
                if const_expr(_use_kt_tr):
                    # KPart path: the KT-tr B-operand presents contraction (n) as
                    # n = 32*n_sub + 4*(lane//16) + (e%4) + (e//4)*16. The dS
                    # A-operand must place the SAME n at each register e.
                    n_local = (
                        n_sub_iter * 32 + lane_div_16 * 4 + (e % 4) + (e // 4) * 16
                    )
                else:
                    n_local = lane_div_16 * MFMA_LK + (n_sub_iter * 32 + e)
                ds_sc = Vec.load(
                    Vec.make_type(1, elem_dtype), lds_ds, [n_local * LDS_MPAD + m_free]
                )[0]
                fx.memref_store(ds_sc, ds_r, e)
            return fx.memref_load_vec(ds_r)

        def _dq_ds_pack_gemm4(m_sub, m_half, n_sub_iter, ks):
            if const_expr(_use_ds_tr):
                return _dq_ds_pack_gemm4_tr(m_sub, m_half, n_sub_iter)
            return _dq_ds_pack_gemm4_ck_k32(m_sub, m_half, n_sub_iter)

        # One dk/dv accumulator PER D-subtile this wave sequentially owns (mirrors
        # compile_fmha_bwd_dvdk_mfma's generalization; D_SUBS_PER_WAVE==1 for D==64,
        # unchanged from before).
        dk_inits = [
            Vec.filled(DV_DK_ACC_LANES, 0.0, fx.Float32)
            for _ in range(DV_DK_ACCS_PER_WAVE)
        ]
        dv_inits = [
            Vec.filled(DV_DK_ACC_LANES, 0.0, fx.Float32)
            for _ in range(DV_DK_ACCS_PER_WAVE)
        ]
        dummy_val = fx.Float32(0.0)
        init_st = dk_inits + dv_inits + [dummy_val]

        # ---- Software-pipelined Q/dO prefetch (prologue/epilogue pattern):
        # the global load for tile (m_tile+1) is
        # issued right after the current tile's barrier, so its VMEM latency
        # overlaps with the current tile's GEMM1/epilogue/GEMM2/dQ compute instead of
        # stalling at the top of the NEXT iteration with nothing to hide behind (ATT
        # profiling showed
        # this load-then-immediately-consume pattern was the single largest stall
        # source). The prefetched registers are threaded through the m_tile
        # scf.for loop as extra iter_args and stored to LDS at the START of the
        # iteration that consumes them.
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

        def _load_lse_item(head_idx_p, m_tile_idx):
            # Mirrors _load_qdo_item's shape: every lane computes a
            # CLAMPED row and issues the load (harmless over-fetch on
            # invalid lanes/rows, same tolerance _load_qdo_item already
            # relies on), so this can be called unconditionally without a
            # dynamic `if tid < BLOCK_M:` guard around the load itself --
            # only the STORE (below) needs the row-count-aware index.
            m_start_p = m_tile_idx * BLOCK_M
            tid_idx = fx.Index(tid)
            m_g_ls = m_start_p + tid_idx
            m_ok_ls = m_g_ls < this_seqlen_q
            m_sf_ls = m_ok_ls.select(m_g_ls, this_seqlen_q - fx.Index(1))
            return (
                _load_f32_row(LSE_buf, _lse_row(m_sf_ls, head_idx_p)),
                _load_f32_row(Dvec_buf, _dvec_row(m_sf_ls, head_idx_p)),
            )

        def _store_lse_regs_to_lds(lse_v, dm_v, buf_elem_off):
            tid_idx = fx.Index(tid)
            if tid_idx < fx.Index(BLOCK_M):
                lse_store_idx = buf_elem_off + tid_idx
                Vec.from_elements([lse_v], fx.Float32).store(lds_lse, [lse_store_idx])
                Vec.from_elements([dm_v], fx.Float32).store(lds_dm, [lse_store_idx])

        def _store_qdo_regs_to_lds(regs_q, regs_do, buf_elem_off):
            for it in range_constexpr(ITEMS_PER_LANE):
                item = lane + fx.Index(it * WARP_SIZE)
                row_off_i = item // fx.Index(VEC_COLS)
                cv_i = item % fx.Index(VEC_COLS)
                row_in_tile = wave * ROWS_PER_WAVE_LD + row_off_i
                col_off_ld = cv_i * fx.Index(MFMA_LK)
                if const_expr(_use_qdo_swz2):
                    # Swizzled lds_q/lds_do: col_off_ld is always 8-aligned,
                    # so the wide v8 store stays contiguous under the swizzle.
                    lds_base = buf_elem_off + _ck_q_swz_off(row_in_tile, col_off_ld)
                else:
                    col_off_swz = _qdo_swz(col_off_ld, row_in_tile)
                    lds_base = buf_elem_off + row_in_tile * LDS_Q_STRIDE + col_off_swz
                Vec(regs_q[it]).store(lds_q, [lds_base])
                Vec(regs_do[it]).store(lds_do, [lds_base])

        loop_results = init_st
        # Per-query-head grid: this block owns exactly ONE query head (head_idx).
        # The heads_per_kv Q-heads are distinct blocks, and their dV/dK partials
        # are combined via atomic-add at the store.
        for _q_head_once in range_constexpr(1):
            # causal skip-ahead: for top-left causal masking, an N-tile at n_tile
            # (rows [n_tile*BLOCK_N, n_tile*BLOCK_N+BLOCK_N)) can only be
            # attended to by m_tile if that tile's rows reach at least
            # n_tile*BLOCK_N -- i.e. m_tile >= (n_tile*BLOCK_N)//BLOCK_M.
            # M-tiles below this are entirely masked out (every (m,n) pair in
            # them has n > m). Starting the loop there instead of 0 skips those
            # fully-masked M-tiles entirely, roughly HALVING both wasted MFMA
            # work and (more importantly, per ATT profiling) redundant Q/dO
            # HBM reloads -- our grid is N-tile-major, so every block reloads
            # the full Q/dO range from HBM once per M-tile it visits; skipping
            # unneeded M-tiles directly cuts that HBM traffic, not just compute.
            m_tile_start_full = (
                (n_tile * BLOCK_N) // BLOCK_M if const_expr(causal) else fx.Index(0)
            )

            m_tile_start = m_tile_start_full
            m_tile_end = n_M_tiles_idx

            # Prologue: prefetch tile m_tile_start's Q/dO for this block's query head.
            # NOTE: if m_tile_start == m_tile_end (empty range, causal with the whole
            # M range masked out), this prefetch reads a tile that is never consumed
            # by the loop below (which won't execute) -- harmless (same over-fetch
            # tolerance as the existing last-M-tile prefetch, clamped in-bounds by
            # _load_qdo_item).
            prologue_buf = (m_tile_start & fx.Index(1)) * fx.Index(LDS_Q_ELEMS)
            prologue_q, prologue_do = _load_qdo_regs(head_idx, m_tile_start)
            _store_qdo_regs_to_lds(prologue_q, prologue_do, prologue_buf)
            # Same prologue treatment as Q/dO: seed the start tile's LSE/D
            # into its double-buffer slot before the hot loop begins, so
            # the first tile's epiB barrier also waits on data with lead
            # time instead of a load issued moments earlier.
            prologue_lse, prologue_dm = _load_lse_item(head_idx, m_tile_start)
            prologue_lse_buf = (m_tile_start & fx.Index(1)) * fx.Index(LDS_LSE_ELEMS)
            _store_lse_regs_to_lds(prologue_lse, prologue_dm, prologue_lse_buf)
            entry_state = list(loop_results[0 : 2 * DV_DK_ACCS_PER_WAVE + 1])

            # ---- Per-m_tile body, factored so the LAST m_tile can run as a peeled,
            # prefetch-free straight-line "drain" AFTER the hot loop -- mirroring
            # flash_attn_gfx950.py's prologue + reduced main loop + epilogue drain
            # (the FlyDSL idiom for a dynamic-trip-count software pipeline). A runtime
            # `if is_hot:` cannot express this because the prefetch subscript-assigns
            # the Python lists next_q/next_do, which an scf.if cannot carry as state.
            #   do_handoff (compile-time bool): the hot loop hands the NEXT tile's
            #     Q/dO off to LDS and prefetches it; the tail drain does neither, so
            #     with the loop stopping at m_tile_end-1 EVERY prefetched tile is
            #     actually consumed -- zero over-fetch.
            #   active (Python True in the hot loop, or a runtime i1 in the tail):
            #     ANDed via _am() into every output-validity mask, so when the whole
            #     range is EMPTY (causal seqlen_k>seqlen_q) the unconditional tail
            #     contributes nothing -- our analog of
            #     flash_attn's "pad the tile count, mask the extra tile".
            def _run_m_tile(
                m_tile,
                dk_accs,
                dv_accs,
                do_handoff,
                active,
                phase="all",
                s_in=None,
                dp_in=None,
            ):
                # phase gates the m-tile pipeline (M_SUBTILES==1 only; "all" keeps the
                # original serial behavior for the M_SUBTILES>1 fallback):
                #   "gemm1"  : GEMM1 -> s/dp registers + next-tile prefetch/handoff.
                #             returns (s_accs, dp_accs); does NOT touch lds_ds.
                #   "epiB"   : LSE load + softmax epilogue -> writes lds_ds(m_tile);
                #             consumes s_in/dp_in; ends barrier.
                #   "stageb" : residency packs + GEMM4(dQ) + GEMM2(dV/dK); reads
                #             lds_ds(m_tile). NO trailing barrier (driver adds it
                #             after the overlapped GEMM1 of the next tile).
                #   "all"    : original full serial body (unchanged).
                _do_gemm1 = const_expr(phase in ("all", "gemm1"))
                _do_epiB = const_expr(phase in ("all", "epiB"))
                _do_stageb = const_expr(phase in ("all", "stageb"))
                dk_accs = list(dk_accs)
                dv_accs = list(dv_accs)
                _sdp = [s_in, dp_in]  # gemm1 fills; epilogue reads

                def _am(cond):
                    # AND the runtime `active` mask in only when it is dynamic (tail
                    # drain over a maybe-empty range); a no-op in the hot loop where
                    # `active` is the Python constant True (const_expr-elided).
                    if const_expr(active is not True):
                        return cond & active
                    return cond

                m_start = m_tile * BLOCK_M
                cur_buf_elems = (m_tile & fx.Index(1)) * fx.Index(LDS_Q_ELEMS)
                cur_lse_buf = (m_tile & fx.Index(1)) * fx.Index(LDS_LSE_ELEMS)

                # ---- Cooperative LSE + D_vec tile stage ---- (epiB/all)
                # LSE/D for THIS tile was already loaded and stored to LDS by the
                # PREVIOUS tile's prefetch step below (or the prologue, for
                # m_tile_start) -- read it straight from the double-buffer
                # slot, no synchronous global load or cooperative write
                # here. The barrier stays (still gates the Q/dO handoff),
                # but no longer waits on a load issued zero cycles earlier.
                if const_expr(_do_epiB):
                    gpu.barrier()

                # ---- Issue next tile's Q/dO global loads INTERLEAVED with GEMM1's
                # MFMA stream (staged scheduling: explicitly alternates MFMA/memory
                # sched_group_barrier groups
                # so the hardware issues them concurrently instead of relying on
                # the default list scheduler). A single `sched_barrier(0)` fence
                # issued right after a bulk-load (tried first) did NOT move the
                # needle -- LLVM still serialized load-then-use because nothing
                # forced actual interleaving. Instead: spread the ITEMS_PER_LANE
                # loads one-per-GEMM1-(m_sub,ks)-step, each wrapped in
                # sched_group_barrier(MFMA_MASK,2,..)+sched_group_barrier(VMEM_MASK,1,..)
                # pairs so the scheduler keeps a load adjacent to (and thus
                # overlapping) that step's 2 MFMAs (s_acc, dp_acc) instead of
                # sinking all loads down to the end of the block. Safe to
                # over-fetch on the final m_tile: the per-lane bounds check clamps
                # to this_seqlen_q-1, in-bounds but unused.
                next_q = [None] * ITEMS_PER_LANE
                next_do = [None] * ITEMS_PER_LANE
                _next_load_it = 0  # plain Python int; range_constexpr is a Python-level
                # unroll, so this is safe to mutate directly (a nested
                # closure over a mutable cell does NOT survive FlyDSL's
                # kernel-body tracing/re-execution).
                # Piggyback the next tile's LSE/D load onto this SAME interleaved
                # prefetch mechanism, issued once (not per-ITEMS_PER_LANE item
                # like Q/dO -- LSE/D is one f32/lane, not a wide vector), at
                # the first available GEMM1 slot.
                next_lse = None
                next_dm = None
                _lse_issued = False

                log2e_scale_cst = fx.Float32(_LOG2E * scale)
                scale_cst = fx.Float32(scale)
                for m_sub in range_constexpr(M_SUBTILES):
                    # ---- GEMM1a: S = Q @ K^T ; GEMM1b: dP = dO @ V^T ---- (gemm1/all)
                    if const_expr(_do_gemm1):
                        # s_accs/dp_accs indexed [n_grp*M_HALVES + m_half]: NIterPerWarp
                        # gives CK_N_GROUPS n-groups (each = NUM_WAVES*16 cols), so
                        # BLOCK_N=128 fills all of S/dP, not just wave*16 in [0,64).
                        # M_HALVES 16-row MFMA-M tiles per m_sub (1 at kM0=16, 2 at kM0=32).
                        s_accs = [
                            Vec.filled(4, 0.0, fx.Float32)
                            for _ in range(M_HALVES * CK_N_GROUPS)
                        ]
                        dp_accs = [
                            Vec.filled(4, 0.0, fx.Float32)
                            for _ in range(M_HALVES * CK_N_GROUPS)
                        ]
                        # Stage (a): batch ALL Q A-operand LDS reads into registers
                        # BEFORE the MFMA stream. Each read is now one wide ds_read
                        # (see _pack_qdo_ck_k32); issuing them all up front lets the
                        # backend keep them in flight under a relaxed lgkmcnt and run
                        # the MFMAs below straight from registers, instead of draining
                        # LDS (lgkmcnt(0)) between every 1-2 MFMAs (the 7% MfmaUtil
                        # stall root cause). K/V B-operands are already register-resident.
                        # do_pack is NOT hoisted here -- dP is computed by
                        # _do_gemm1_dp_only() below instead, called later (after the
                        # epilogue's P-only pass), which hoists its own do_pack reads
                        # at that later point.
                        q_packs = [[None] * M_HALVES for _ in range_constexpr(D // 32)]
                        for k_iter in range_constexpr(D // 32):
                            for m_half in range_constexpr(M_HALVES):
                                q_packs[k_iter][m_half] = _pack_qdo_ck_k32(
                                    m_sub,
                                    m_half,
                                    k_iter,
                                    cur_buf_elems,
                                    lds_q,
                                    LDS_Q_STRIDE,
                                )

                        # dP-only loop, matching the stage order (S -> softmax/P ->
                        # dP -> dV -> dS -> dK) by running AFTER the epilogue's P-only
                        # pass instead of fused with S above. Writes into dp_accs (the
                        # SAME list object created above, mutated in place via index
                        # assignment -- no nonlocal needed) so the epilogue's dS-only
                        # pass reads it exactly as it does in the fused/unsplit form.
                        def _do_gemm1_dp_only():
                            _dp_do_packs = [
                                [None] * M_HALVES for _ in range_constexpr(D // 32)
                            ]
                            for k_iter in range_constexpr(D // 32):
                                for m_half in range_constexpr(M_HALVES):
                                    _dp_do_packs[k_iter][m_half] = _pack_qdo_ck_k32(
                                        m_sub,
                                        m_half,
                                        k_iter,
                                        cur_buf_elems,
                                        lds_do,
                                        LDS_Q_STRIDE,
                                    )
                            for k_iter in range_constexpr(D // 32):
                                for m_half in range_constexpr(M_HALVES):
                                    do_pack = _dp_do_packs[k_iter][m_half]
                                    for n_grp in range_constexpr(CK_N_GROUPS):
                                        ai = n_grp * M_HALVES + m_half
                                        kv_i = k_iter * CK_N_GROUPS + n_grp
                                        v_pack = v_regs_ck[kv_i]
                                        dp_accs[ai] = mfma_gemm4(
                                            do_pack, v_pack, dp_accs[ai]
                                        )
                                        rocdl.sched_group_barrier(_MFMA_MASK, 1, 0)

                        for k_iter in range_constexpr(D // 32):
                            for m_half in range_constexpr(M_HALVES):
                                q_pack = q_packs[k_iter][m_half]
                                for n_grp in range_constexpr(CK_N_GROUPS):
                                    ai = n_grp * M_HALVES + m_half
                                    kv_i = k_iter * CK_N_GROUPS + n_grp
                                    k_pack = k_regs_ck[kv_i]
                                    s_accs[ai] = mfma_gemm4(q_pack, k_pack, s_accs[ai])
                                    rocdl.sched_group_barrier(_MFMA_MASK, 1, 0)
                            rocdl.sched_group_barrier(_MFMA_MASK, 2, 0)
                            if const_expr(
                                do_handoff and _next_load_it < ITEMS_PER_LANE
                            ):
                                q_v, do_v = _load_qdo_item(
                                    head_idx, m_tile + fx.Index(1), _next_load_it
                                )
                                next_q[_next_load_it] = q_v
                                next_do[_next_load_it] = do_v
                                _next_load_it += 1
                            elif const_expr(do_handoff and not _lse_issued):
                                # Piggyback LSE/D onto the next available slot after
                                # Q/dO's own ITEMS_PER_LANE items are all issued (or
                                # immediately, if ITEMS_PER_LANE==0 -- not expected at
                                # the real tile shapes, but handled by the elif chain
                                # either way).
                                next_lse, next_dm = _load_lse_item(
                                    head_idx, m_tile + fx.Index(1)
                                )
                                _lse_issued = True
                            rocdl.sched_group_barrier(_VMEM_MASK, 1, 0)
                        # Any remaining prefetch items for this m_sub (ITEMS_PER_LANE >
                        # M_SUBTILES*K_STEPS, e.g. wide D) are issued right after
                        # GEMM1 -- still overlaps this m_sub's epilogue/GEMM2/dQ below.
                        # `const_expr()` marks this Python-int comparison as a
                        # compile-time constant so the AST rewriter unrolls it instead
                        # of lowering to a dynamic scf.if (which would require next_q/
                        # next_do -- plain Python lists -- to be MLIR-value loop state).
                        if const_expr(do_handoff and m_sub == M_SUBTILES - 1):
                            for _pad_it in range_constexpr(ITEMS_PER_LANE):
                                if const_expr(_pad_it >= _next_load_it):
                                    q_v, do_v = _load_qdo_item(
                                        head_idx, m_tile + fx.Index(1), _pad_it
                                    )
                                    next_q[_pad_it] = q_v
                                    next_do[_pad_it] = do_v
                            if const_expr(not _lse_issued):
                                # Fallback: no GEMM1 slot was available this m_sub
                                # (e.g. K_STEPS==0 at some D) -- issue plainly, still
                                # ahead of the store below.
                                next_lse, next_dm = _load_lse_item(
                                    head_idx, m_tile + fx.Index(1)
                                )
                                _lse_issued = True

                        rocdl.sched_barrier(0)

                        # Hand s/dp registers to the epilogue phase (pipeline: epiB is a
                        # separate call; serial "all": consumed a few lines below).
                        _sdp[0], _sdp[1] = s_accs, dp_accs

                    # ---- P (for dV) and dS' (for dK/dQ, scale deferred to store/atomic-add),
                    # both stored TRANSPOSED [n,m] ----
                    # m_within = lane_div_32*4 + (r//4)*8 + r%4: within each group of 4
                    # consecutive r (same r//4), m_within increments by 1 -- 4 CONTIGUOUS
                    # lds_lse/lds_dm addresses. Load each group as one v4 (instead of 4
                    # scalar ds_read) and index within the group by r%4; cuts LDS
                    # instruction count for this epilogue 4x (32 scalar reads -> 8 v4 reads).
                    if const_expr(_do_epiB and not _do_gemm1):
                        # epiB-only pipeline call: recover s/dp handed over by the
                        # preceding (separate) GEMM1 phase invocation.
                        s_accs, dp_accs = _sdp[0], _sdp[1]
                    if const_expr(_do_epiB):
                        v4f32_type = Vec.make_type(4, fx.Float32)
                        # N-split register-only handoff: per-lane P/dS tiles, one 4-wide pack
                        # per acc-index ai=n_grp*M_HALVES+m_half. GEMM2 (N-split) consumes
                        # P directly as its A-operand (Pᵀ) with no LDS round-trip (dS still
                        # routes through lds_ds, since it also feeds GEMM4/dQ). The (ai, r)
                        # order here IS the TransposeC A-operand layout because our
                        # GEMM0 K32 s_accs already place P at n=lane%16+wave*16+n_grp*64
                        # (fixed/lane), m=lane//16*4+r+m_half*16.
                        p_regs_ns = [
                            fx.make_rmem_tensor(4, elem_dtype)
                            for _ in range(M_HALVES * CK_N_GROUPS)
                        ]
                        ds_regs_ns = [
                            fx.make_rmem_tensor(4, elem_dtype)
                            for _ in range(M_HALVES * CK_N_GROUPS)
                        ]
                        # Loop n-groups so all of BLOCK_N (not just wave*16) is filled.
                        # lse_grp/dm_grp depend only on m_half (via lse_read_base below),
                        # not n_grp -- load each ONCE here, before the n_grp loop, instead
                        # of CK_N_GROUPS redundant re-reads per m_half.
                        lse_grps_hoisted, dm_grps_hoisted = [], []
                        for m_half in range_constexpr(M_HALVES):
                            m_group_base = lane_div_16 * 4 + m_half * 16 + (m_sub * 32)
                            lse_read_base = cur_lse_buf + m_group_base
                            lse_grps_hoisted.append(
                                Vec.load(v4f32_type, lds_lse, [lse_read_base])
                            )
                            dm_grps_hoisted.append(
                                Vec.load(v4f32_type, lds_dm, [lse_read_base])
                            )
                        # Run the body below twice, computing/storing P on pass 0 and
                        # dS on pass 1, instead of both in one fused pass. p_val/valid_mn
                        # are carried from pass 0 to pass 1 as SSA values via these
                        # constexpr dicts, so pass 1 recomputes no arithmetic and
                        # re-reads no LDS -- the emitted work is the same, only its
                        # order differs.
                        _epi_passes = 2
                        _epi_p_vals, _epi_valids = {}, {}
                        for _epi_pass in range_constexpr(_epi_passes):
                            _epi_do_p = _epi_pass == 0
                            _epi_do_ds = _epi_pass == 1
                            # pass 0 (P-only) just finished -- run dP's MFMA now, before
                            # pass 1 (dS-only) needs dp_accs. Stage order:
                            # S (GEMM1 above) -> softmax/P (pass 0) -> dP (here)
                            # -> dS (pass 1) -> dK.
                            if const_expr(_epi_pass == 1):
                                _do_gemm1_dp_only()
                            # Holds n_grp==1's (ds_wide_elems, ds_wide_base_off) per
                            # m_half until n_grp==2 arrives, so both can be fused into
                            # one ds_write2st64_b64 at the flush site below.
                            _ds_write2_pending = (
                                [None] * M_HALVES
                                if const_expr(_use_ds_write2_asm)
                                else None
                            )
                            for n_grp in range_constexpr(CK_N_GROUPS):
                                n_within = (
                                    lane_mod_16
                                    + wave * fx.Index(16)
                                    + n_grp * fx.Index(NUM_WAVES * 16)
                                )
                                n_row_abs = n_within + n_start
                                n_ok = n_row_abs < this_seqlen_k
                                for m_half in range_constexpr(M_HALVES):
                                    lse_grp = lse_grps_hoisted[m_half]
                                    dm_grp = dm_grps_hoisted[m_half]
                                    ai = n_grp * M_HALVES + m_half
                                    ds_wide_elems = (
                                        [] if const_expr(_use_ds_writer_wide) else None
                                    )
                                    ds_wide_base_off = None
                                    for r in range_constexpr(4):
                                        m_within = lane_div_16 * 4 + r + m_half * 16
                                        if const_expr(_epi_do_p):
                                            m_row_abs = (
                                                m_within + (m_sub * 32) + m_start
                                            )
                                            m_valid = m_row_abs < this_seqlen_q
                                            lse_val = Vec(lse_grp)[r]
                                            s_val = Vec(s_accs[ai])[r]
                                            valid_mn = _am(m_valid & n_ok)
                                            if const_expr(causal):
                                                valid_mn = valid_mn & (
                                                    n_row_abs <= m_row_abs
                                                )
                                            neg_log2e_lse = fx.Float32(
                                                arith.mulf(
                                                    _raw(lse_val),
                                                    _raw(fx.Float32(-_LOG2E)),
                                                    fastmath=fm,
                                                )
                                            )
                                            p_val = _softmax_p(
                                                s_val,
                                                neg_log2e_lse,
                                                log2e_scale_cst,
                                                valid_mn,
                                                fm,
                                            )
                                            _epi_p_vals[(n_grp, m_half, r)] = p_val
                                            _epi_valids[(n_grp, m_half, r)] = valid_mn
                                        else:
                                            p_val = _epi_p_vals[(n_grp, m_half, r)]
                                            valid_mn = _epi_valids[(n_grp, m_half, r)]
                                        if const_expr(_epi_do_ds):
                                            dm_val = Vec(dm_grp)[r]
                                            dp_val = Vec(dp_accs[ai])[r]
                                            ds_val = _grad_ds_unscaled(
                                                p_val, dp_val, dm_val, valid_mn, fm
                                            )
                                        # Register-only handoff for GEMM2 (dV/dK): stash P/dS
                                        # into per-(ai,r) rmem; the N-split GEMM2 reads these
                                        # instead of lds_p/lds_ds. BUT dS ALSO feeds GEMM4/dQ,
                                        # which still consumes it from lds_ds (dS->dQ
                                        # routes through LDS) -- so keep the lds_ds write;
                                        # only the lds_p write is dropped (P feeds only
                                        # dV/GEMM2).
                                        if const_expr(_epi_do_p):
                                            fx.memref_store(
                                                Vec.from_elements(
                                                    [p_val], fx.Float32
                                                ).to(elem_dtype)[0],
                                                p_regs_ns[ai],
                                                r,
                                            )
                                        if const_expr(_epi_do_ds):
                                            fx.memref_store(
                                                Vec.from_elements(
                                                    [ds_val], fx.Float32
                                                ).to(elem_dtype)[0],
                                                ds_regs_ns[ai],
                                                r,
                                            )
                                            m_local = m_within + (m_sub * 32)
                                            n_local = n_within
                                            # Default [n,m] m-contiguous store, UNLESS _use_ds_tr
                                            # (paired with the ds_read_tr A-operand read below --
                                            # _dq_ds_pack_gemm4_tr) -- then use the
                                            # swizzled layout (_ck_ds_off).
                                            if const_expr(_use_ds_writer_wide):
                                                # Buffer this group's 4 ds_val's (r=0..3,
                                                # contiguous _ck_ds_off addresses) and issue
                                                # ONE v4 store after the loop instead of 4
                                                # scalar ones.
                                                ds_wide_elems.append(ds_val)
                                                if const_expr(r == 0):
                                                    ds_wide_base_off = _ck_ds_off(
                                                        m_local, n_local
                                                    )
                                            else:
                                                ds_off = (
                                                    _ck_ds_off(m_local, n_local)
                                                    if const_expr(_use_ds_tr)
                                                    else (n_local * LDS_MPAD + m_local)
                                                )
                                                Vec.from_elements(
                                                    [ds_val], fx.Float32
                                                ).to(elem_dtype).store(lds_ds, [ds_off])
                                    # end for r in range_constexpr(4): flush the buffered wide dS
                                    # store, firing only on the branch that populated ds_wide_elems.
                                    if const_expr(_epi_do_ds and _use_ds_writer_wide):
                                        if const_expr(_use_ds_write2_asm):
                                            # Fused write2 pattern:
                                            # n_grp==0 stays a plain wide store, n_grp==1's
                                            # payload is held until n_grp==2 arrives, then both
                                            # are fused into ONE ds_write2st64_b64 hand-asm call
                                            # (n_grp==1's base address as ADDR/offset0, n_grp==2
                                            # at the fixed +4-ST64-unit delta as offset1 -- see
                                            # _ds_write2st64_b64_asm and _DS_WRITE2_OFFSET1_ST64).
                                            if const_expr(n_grp == 0):
                                                Vec.from_elements(
                                                    ds_wide_elems, fx.Float32
                                                ).to(elem_dtype).store(
                                                    lds_ds, [ds_wide_base_off]
                                                )
                                            elif const_expr(n_grp == 1):
                                                _ds_write2_pending[m_half] = (
                                                    ds_wide_elems,
                                                    ds_wide_base_off,
                                                )
                                            else:
                                                _prev_elems, _prev_base_off = (
                                                    _ds_write2_pending[m_half]
                                                )
                                                data0 = (
                                                    Vec.from_elements(
                                                        _prev_elems, fx.Float32
                                                    )
                                                    .to(elem_dtype)
                                                    .bitcast(fx.Int32)
                                                    .ir_value()
                                                )
                                                data1 = (
                                                    Vec.from_elements(
                                                        ds_wide_elems, fx.Float32
                                                    )
                                                    .to(elem_dtype)
                                                    .bitcast(fx.Int32)
                                                    .ir_value()
                                                )
                                                _ds_write2st64_b64_asm(
                                                    _prev_base_off,
                                                    lds_ds_off,
                                                    data0,
                                                    data1,
                                                    _DS_WRITE2_OFFSET1_ST64,
                                                )
                                            rocdl.sched_group_barrier(
                                                _LDS_WRITE_MASK, 1, 0
                                            )
                                        else:
                                            Vec.from_elements(
                                                ds_wide_elems, fx.Float32
                                            ).to(elem_dtype).store(
                                                lds_ds, [ds_wide_base_off]
                                            )
                                            rocdl.sched_group_barrier(
                                                _LDS_WRITE_MASK, 1, 0
                                            )

                        rocdl.sched_barrier(0)

                        gpu.barrier()

                    # Pipeline phase split: a gemm1-only or epiB-only call stops here
                    # (M_SUBTILES==1). "all"/"stageb" fall through to GEMM2/GEMM4 below.
                    if const_expr(not _do_stageb):
                        break

                    def _do_gemm4_dq():
                        # ---- dQ GEMM4: Gemm4BlockWarps=1x4x1, 16x16x32 with
                        # SGradRegSlice A-operand and register-resident K^T B. k4_loops =
                        # WAVE_N_TILES (= kN0/kK4): prefetch the next dS LDS slice while
                        # MFMA-ing the current one.
                        if const_expr(_use_ds4_residency):
                            # Read each n_sub_iter's dS-transpose slice ONCE per m_half
                            # (mirrors kt_gemm4_regs's existing residency-cache pattern)
                            # and reuse it across every d_iter this wave owns, instead of
                            # re-reading the IDENTICAL LDS data once per d_iter -- pure
                            # register reuse, no LDS re-read at all.
                            for m_half in range_constexpr(M_HALVES):
                                ds_gemm4_regs = []
                                for n_sub_iter in range_constexpr(WAVE_N_TILES):
                                    ds_gemm4_regs.append(
                                        _dq_ds_pack_gemm4(m_sub, m_half, n_sub_iter, 0)
                                    )

                                for d_iter in range_constexpr(GEMM4_DQ_D_SUBS_PER_WAVE):
                                    wave_d_sub_16_raw = (
                                        wave * GEMM4_DQ_D_SUBS_PER_WAVE + d_iter
                                    )
                                    d_in_range = wave_d_sub_16_raw < fx.Index(
                                        GEMM4_D_TOTAL_SUBS
                                    )
                                    wave_d_sub_16 = d_in_range.select(
                                        wave_d_sub_16_raw, fx.Index(0)
                                    )
                                    dq_acc = Vec.filled(4, 0.0, fx.Float32)
                                    for n_sub_iter in range_constexpr(WAVE_N_TILES):
                                        ds_pack = ds_gemm4_regs[n_sub_iter]
                                        b_pack = kt_gemm4_regs[
                                            d_iter * WAVE_N_TILES + n_sub_iter
                                        ]
                                        dq_acc = mfma_gemm4(ds_pack, b_pack, dq_acc)
                                        rocdl.sched_group_barrier(_MFMA_MASK, 1, 0)
                                        rocdl.sched_group_barrier(_LDS_READ_MASK, 1, 0)
                                    if const_expr(_use_kt_tr):
                                        d_col_abs_dq = (
                                            wave * fx.Index(16)
                                            + lane_mod_16
                                            + d_iter * fx.Index(64)
                                        )
                                    else:
                                        d_col_abs_dq = lane_mod_16 + wave_d_sub_16 * 16
                                    for r in range_constexpr(4):
                                        m_within = lane_div_16 * 4 + r
                                        m_row_abs = (
                                            m_within
                                            + m_half * 16
                                            + (m_sub * 32)
                                            + m_start
                                        )
                                        m_ok = _am(
                                            (m_row_abs < this_seqlen_q) & d_in_range
                                        )
                                        if m_ok:
                                            q_row_g = _q_row_out(m_row_abs, head_idx)
                                            flat_dq = fx.Int32(
                                                fx.Index(q_row_g) * fx.Index(D)
                                                + d_col_abs_dq
                                            )
                                            dq_scaled = fx.Float32(
                                                arith.mulf(
                                                    _raw(Vec(dq_acc)[r]),
                                                    _raw(scale_cst),
                                                    fastmath=fm,
                                                )
                                            )
                                            _store_dq(flat_dq, dq_scaled)
                        else:
                            for d_iter in range_constexpr(GEMM4_DQ_D_SUBS_PER_WAVE):
                                wave_d_sub_16_raw = (
                                    wave * GEMM4_DQ_D_SUBS_PER_WAVE + d_iter
                                )
                                d_in_range = wave_d_sub_16_raw < fx.Index(
                                    GEMM4_D_TOTAL_SUBS
                                )
                                wave_d_sub_16 = d_in_range.select(
                                    wave_d_sub_16_raw, fx.Index(0)
                                )
                                for m_half in range_constexpr(M_HALVES):
                                    dq_acc = Vec.filled(4, 0.0, fx.Float32)
                                    ds_pack = _dq_ds_pack_gemm4(m_sub, m_half, 0, 0)
                                    for n_sub_iter in range_constexpr(WAVE_N_TILES):
                                        if const_expr(n_sub_iter + 1 < WAVE_N_TILES):
                                            ds_pack_next = _dq_ds_pack_gemm4(
                                                m_sub, m_half, n_sub_iter + 1, 0
                                            )
                                        b_pack = kt_gemm4_regs[
                                            d_iter * WAVE_N_TILES + n_sub_iter
                                        ]
                                        dq_acc = mfma_gemm4(ds_pack, b_pack, dq_acc)
                                        if const_expr(n_sub_iter + 1 < WAVE_N_TILES):
                                            ds_pack = ds_pack_next
                                        rocdl.sched_group_barrier(_MFMA_MASK, 1, 0)
                                        rocdl.sched_group_barrier(_LDS_READ_MASK, 1, 0)
                                    if const_expr(_use_kt_tr):
                                        # KPart path: the KT-tr B-operand's free dim (output d) is
                                        # KPack layout d = wave*16 + (lane%16) + 64*j1, with
                                        # j1 = d_iter (which KPack half). The MFMA output column
                                        # (lane%16) therefore maps to this d, not the contiguous
                                        # wave_d_sub_16*16 the scalar path uses.
                                        d_col_abs_dq = (
                                            wave * fx.Index(16)
                                            + lane_mod_16
                                            + d_iter * fx.Index(64)
                                        )
                                    else:
                                        d_col_abs_dq = lane_mod_16 + wave_d_sub_16 * 16
                                    for r in range_constexpr(4):
                                        m_within = lane_div_16 * 4 + r
                                        m_row_abs = (
                                            m_within
                                            + m_half * 16
                                            + (m_sub * 32)
                                            + m_start
                                        )
                                        m_ok = _am(
                                            (m_row_abs < this_seqlen_q) & d_in_range
                                        )
                                        if m_ok:
                                            q_row_g = _q_row_out(m_row_abs, head_idx)
                                            flat_dq = fx.Int32(
                                                fx.Index(q_row_g) * fx.Index(D)
                                                + d_col_abs_dq
                                            )
                                            dq_scaled = fx.Float32(
                                                arith.mulf(
                                                    _raw(Vec(dq_acc)[r]),
                                                    _raw(scale_cst),
                                                    fastmath=fm,
                                                )
                                            )
                                            _store_dq(flat_dq, dq_scaled)

                    def _do_gemm2_dvdk():
                        # ---- dV += P^T @ dO ; dK += dS^T @ Q ----
                        # N-split (Gemm1BlockWarps rm1=4): each wave owns
                        # NSPLIT_MITER interleaved 16-row n-bands (= the n_grp bands it
                        # produced in GEMM0/epilogue) and sweeps the FULL D. A-operand
                        # (Pᵀ/dSᵀ) comes straight from the epilogue registers p_regs_ns/
                        # ds_regs_ns[ai] (ai = mIter*M_HALVES + ks) -- register-only, NO
                        # LDS round-trip. B-operand (dOᵀ/Qᵀ) spans full D: read
                        # every d-subtile straight off the plain lds_q/lds_do (all
                        # waves cooperatively wrote it; here each wave reads all D//16
                        # bands for its own n-bands). acc index = mIter*NSPLIT_NITER + d_full.
                        #
                        # Stage (a): batch ALL NSPLIT_NITER*M_HALVES q_pack/do_pack
                        # transpose reads into registers BEFORE the d_full loop issues
                        # any MFMA -- mirrors GEMM1's q_packs/do_packs hoist above.
                        # Without this, each d_full's q_pack/do_pack is read
                        # immediately before that d_full's first consuming MFMA, so
                        # every d_full iteration stalls on its own just-issued
                        # ds_read_tr16_b64.
                        qdo_tr_q_packs = [
                            [None] * M_HALVES for _ in range_constexpr(NSPLIT_NITER)
                        ]
                        qdo_tr_do_packs = [
                            [None] * M_HALVES for _ in range_constexpr(NSPLIT_NITER)
                        ]
                        for d_full in range_constexpr(NSPLIT_NITER):
                            d_in_range_ns = (d_full * 16) < fx.Index(D)
                            d_band = d_in_range_ns.select(fx.Index(d_full), fx.Index(0))
                            for ks in range_constexpr(M_HALVES):
                                qdo_tr_q_packs[d_full][ks] = _qdo_read_k16_tr(
                                    m_sub, ks, d_band, cur_buf_elems, lds_q_off
                                )
                                qdo_tr_do_packs[d_full][ks] = _qdo_read_k16_tr(
                                    m_sub, ks, d_band, cur_buf_elems, lds_do_off
                                )
                                rocdl.sched_group_barrier(_LDS_READ_MASK, 1, 0)
                        for d_full in range_constexpr(NSPLIT_NITER):
                            d_in_range_ns = (d_full * 16) < fx.Index(D)
                            d_band = d_in_range_ns.select(fx.Index(d_full), fx.Index(0))
                            for ks in range_constexpr(M_HALVES):
                                q_pack = qdo_tr_q_packs[d_full][ks]
                                do_pack = qdo_tr_do_packs[d_full][ks]
                                for mIter in range_constexpr(CK_N_GROUPS):
                                    ai = mIter * M_HALVES + ks
                                    acc_i = mIter * NSPLIT_NITER + d_full
                                    ds_pack = fx.memref_load_vec(ds_regs_ns[ai])
                                    p_pack = fx.memref_load_vec(p_regs_ns[ai])
                                    dv_accs[acc_i] = mfma_gemm16_tc(
                                        p_pack, do_pack, dv_accs[acc_i]
                                    )
                                    dk_accs[acc_i] = mfma_gemm16_tc(
                                        ds_pack, q_pack, dk_accs[acc_i]
                                    )
                                    rocdl.sched_group_barrier(_MFMA_MASK, 1, 0)
                                    rocdl.sched_group_barrier(_LDS_READ_MASK, 1, 0)

                    _do_gemm4_dq()
                    _do_gemm2_dvdk()
                    rocdl.sched_barrier(0)

                    # Serial "all" ends every tile with a barrier. In the pipeline the
                    # stage-B call intentionally omits it so the driver can slot the
                    # next tile's GEMM1 in before the single closing barrier.
                    if const_expr(phase == "all"):
                        gpu.barrier()

                if const_expr(_do_gemm1 and do_handoff):
                    nxt_buf_elems = ((m_tile + fx.Index(1)) & fx.Index(1)) * fx.Index(
                        LDS_Q_ELEMS
                    )
                    _store_qdo_regs_to_lds(next_q, next_do, nxt_buf_elems)
                    nxt_lse_buf = ((m_tile + fx.Index(1)) & fx.Index(1)) * fx.Index(
                        LDS_LSE_ELEMS
                    )
                    _store_lse_regs_to_lds(next_lse, next_dm, nxt_lse_buf)

                if const_expr(phase == "gemm1"):
                    return _sdp[0], _sdp[1]
                return dk_accs, dv_accs

            _BASE = 2 * DV_DK_ACCS_PER_WAVE + 1
            hot_end = m_tile_end - fx.Index(1)

            # ---- Serial hot loop over [m_tile_start, m_tile_end-1): every iteration
            # hands its successor to LDS, so the loop only walks tiles whose Q/dO is
            # already resident (prologue seeded m_tile_start). Peeled tail below
            # handles the final tile with no handoff. hot_end < m_tile_start
            # (empty/single-tile range) -> zero-trip loop, entry_state passes through.
            for m_tile, iter_args in range(
                m_tile_start, hot_end, fx.Index(1), init=entry_state
            ):
                dk_accs = list(iter_args[0:DV_DK_ACCS_PER_WAVE])
                dv_accs = list(iter_args[DV_DK_ACCS_PER_WAVE : 2 * DV_DK_ACCS_PER_WAVE])
                dk_accs, dv_accs = _run_m_tile(m_tile, dk_accs, dv_accs, True, True)
                iter_args = yield dk_accs + dv_accs + [dummy_val]

            hot_results = iter_args[0:_BASE]

            # ---- Peeled tail (flash_attn drain): the last m_tile, no prefetch
            # handoff. `tail_active` is False iff the range was empty
            # (m_tile_end==m_tile_start); then _am() masks every output to zero so
            # this straight-line call is a no-op, and tail_m is clamped to
            # m_tile_start (in-LDS, safe to read).
            tail_active = m_tile_end > m_tile_start
            tail_m = tail_active.select(m_tile_end - fx.Index(1), m_tile_start)
            tail_dk = list(hot_results[0:DV_DK_ACCS_PER_WAVE])
            tail_dv = list(hot_results[DV_DK_ACCS_PER_WAVE : 2 * DV_DK_ACCS_PER_WAVE])
            tail_dk, tail_dv = _run_m_tile(tail_m, tail_dk, tail_dv, False, tail_active)
            loop_results = tail_dk + tail_dv + [dummy_val]

        # ---- Store dV and dK ----
        store_scale_cst = fx.Float32(scale)
        dk_finals = loop_results[0:DV_DK_ACCS_PER_WAVE]
        dv_finals = loop_results[DV_DK_ACCS_PER_WAVE : 2 * DV_DK_ACCS_PER_WAVE]
        # N-split C-distribution (rm1=4): wave W owns interleaved n-bands
        # n = mIter*64 + wave*16 + lane%16 (mIter in [0,NSPLIT_MITER)); each spans
        # full D: d = d_full*16 + lane//16*4 + r. acc index = mIter*NSPLIT_NITER+d_full.
        for mIter in range_constexpr(NSPLIT_MITER):
            n_row_abs = (
                mIter * fx.Index(NUM_WAVES * 16)
                + wave * fx.Index(16)
                + lane_mod_16
                + n_start
            )
            n_ok = n_row_abs < this_seqlen_k
            n_safe = n_ok.select(n_row_abs, this_seqlen_k - fx.Index(1))
            kv_row_g = (
                _kv_row_out_per_qhead(n_safe)
                if const_expr(_use_ck_scope_dvdk)
                else _kv_row_out(n_safe)
            )
            for d_full in range_constexpr(NSPLIT_NITER):
                acc_i = mIter * NSPLIT_NITER + d_full
                d_col_abs0 = d_full * fx.Index(16) + lane_div_16 * 4
                d_ok0 = d_col_abs0 < fx.Index(D)
                flat_col0 = fx.Int32(fx.Index(kv_row_g) * fx.Index(D) + d_col_abs0)
                if const_expr(not (_dvdk_atomic and not _use_ck_scope_dvdk)):
                    if n_ok & d_ok0:
                        dk_vals = [
                            fx.Float32(
                                arith.mulf(
                                    _raw(Vec(dk_finals[acc_i])[r]),
                                    _raw(store_scale_cst),
                                    fastmath=fm,
                                )
                            )
                            for r in range(4)
                        ]
                        dv_vals = [Vec(dv_finals[acc_i])[r] for r in range(4)]
                        _store_f32_vec4(dk_store_rsrc, flat_col0, dk_vals)
                        _store_f32_vec4(dv_store_rsrc, flat_col0, dv_vals)
                else:
                    for r in range_constexpr(4):
                        d_col_abs = d_full * fx.Index(16) + lane_div_16 * 4 + r
                        d_ok = d_col_abs < fx.Index(D)
                        flat_col = fx.Int32(
                            fx.Index(kv_row_g) * fx.Index(D) + d_col_abs
                        )
                        if n_ok & d_ok:
                            dk_scaled = fx.Float32(
                                arith.mulf(
                                    _raw(Vec(dk_finals[acc_i])[r]),
                                    _raw(store_scale_cst),
                                    fastmath=fm,
                                )
                            )
                            if const_expr(_dvdk_atomic and not _use_ck_scope_dvdk):
                                _atomic_add_f32(dk_rsrc, flat_col, dk_scaled)
                                _atomic_add_f32(
                                    dv_rsrc, flat_col, Vec(dv_finals[acc_i])[r]
                                )
                            else:
                                _store_f32_row(dK_buf, flat_col, dk_scaled)
                                _store_f32_row(
                                    dV_buf, flat_col, Vec(dv_finals[acc_i])[r]
                                )

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
        # Per-query-head grid: B * H (QUERY heads) * num_N_tiles blocks. K/V are
        # shared within each heads_per_kv group (looked up via
        # head_idx//heads_per_kv in the kernel) and
        # dV/dK partials combined via atomic-add. varlen: B already collapsed to 1 by
        # the caller.
        grid_x = fx.Int32(fx.Index(B) * fx.Index(H) * num_N_tiles)
        fmha_bwd_dqdkdv_mfma_gfx950_kernel(
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
            B,
        ).launch(
            grid=(grid_x, 1, 1),
            block=(BLOCK_SIZE, 1, 1),
            stream=stream,
        )

    # Occupancy=1: exactly one wave per SIMD (waves_per_eu=1).
    launch_fn.compile_hints = {"waves_per_eu": 1}

    return launch_fn
