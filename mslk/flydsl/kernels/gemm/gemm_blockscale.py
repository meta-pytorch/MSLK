# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Plain (non-grouped) FP8 GEMM with symmetric 2D block scaling.

This is the kernel behind ``mslk::f8f8bf16_blockwise`` on ROCm. It replaces the
CK ``DeviceGemmMultiD_ABScale_Xdl_CShuffle_V3`` kernel and matches its exact
block-scale contract:

  A: [M, K] FP8      x_scale: [ceil(M/128), K//128] FP32  (ScaleBlockM=128,
                                                           ScaleBlockK=128,
                                                           M-outer / K-major)
  B: [N, K] FP8      w_scale: [N//128,      K//128] FP32  (ScaleBlockN=128,
                                                           ScaleBlockK=128,
                                                           N-outer / K-major)
  D: [M, N] BF16/FP16

  Y[m, n] = sum_k (A[m, k] * x_scale[m // 128, k // 128])
                * (B[n, k] * w_scale[n // 128, k // 128])

Unlike the per-token groupwise kernel (``fp8_grouped_gemm``),
the A scale here has ScaleBlockM=128, so one scalar covers a whole 16-row MFMA
fragment (all lanes/rows in the block share it): the kernel issues one scalar A
scale load per (M-block, K-block) instead of the groupwise path's per-token vec4
loads. All the tile loading / MFMA / epilogue machinery is shared with the
groupwise kernel via ``fp8_grouped_gemm_common``; only the scale indexing
and the (single-group) tile map differ, so this file adds a ``compute_tile`` and
the launcher and reuses everything else.

Two B layouts are supported (compile-time ``b_preshuffled``):
  * plain B ``[N, K]`` staged HBM->LDS->registers (matches CK -- no preshuffle);
  * B pre-swizzled into the MFMA layout, loaded HBM->registers (faster; for
    callers that shuffle the weight once and cache it).
"""

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import math as math_dialect, scf
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.expr import (
    arith,
    buffer_ops,
    const_expr,
    gpu,
    range_constexpr,
    rocdl,
    vector,
)
from flydsl.expr.arith import ArithValue
from flydsl.expr.typing import T, Vector
from flydsl.runtime.device import get_rocm_arch as get_hip_arch
from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr
from mslk.flydsl.kernels.gemm.fp8_grouped_gemm_common import (
    compute_compile_constants,
    compute_mfma_tiling,
    init_accumulators,
    make_a_tile_loaders,
    make_b_loader,
    make_b_tile_loaders,
    make_epilogue_writers,
    make_hot_loop_scheduler,
    make_kloop_plain,
    make_lds_b_loader,
    make_lds_loader,
    make_n_block_coords,
    make_pingpong_kloop,
    make_plain_b_tile,
    out_mlir_for,
    pack_i64x4_to_i32x8,
    setup_lds_allocation,
    setup_lds_allocation_plain,
    validate_lds_budget_plain,
    validate_lds_budget_preshuffle,
    validate_params,
)
from mslk.flydsl.kernels.mma.mfma_epilogues import mfma_epilog


def make_compute_tile_blockwise(
    *,
    _is_gfx950,
    sb_per_tile,
    ku_per_sb,
    scale_k,
    m_repeat,
    num_acc_n,
    acc_init,
    mfma_res_ty,
    m_blk_list,
    n_block_for_scale,
    sa_rsrc,
    sb_rsrc,
    lds_load_packs_k64,
    col_offset_base_bytes,
    lane_mod_16,
    lane_div_16,
):
    """Build the per-K-tile compute closure for the native 2D block-scale GEMM.

    Defined at module scope (NOT inside the ``@flyc.kernel`` body) so its plain
    Python ``if``/``else`` control flow runs during tracing instead of being
    AST-rewritten into scf branches -- the same structure the shared
    ``fp8_grouped_gemm_common`` helpers use.

    Returns ``compute_tile(accs_in, k_tile_idx_py, lds_base, b_tile_in,
    scales_pf, *, a0_prefetch=None)``. Scales are loaded here (software FP32
    path): one A scalar per (M-block, K-block) and one B scalar per (N-block,
    K-block), both uniform across the wave.
    """

    def compute_tile(
        accs_in, k_tile_idx_py, lds_base, b_tile_in, scales_pf, *, a0_prefetch=None
    ):
        current_accs = list(accs_in)

        for sb in range_constexpr(sb_per_tile):
            kb = fx.Index(k_tile_idx_py * sb_per_tile + sb)

            # One A scale scalar per M-block (broadcast to the 4-row acc), one B
            # scale scalar per N-block; both uniform across the wave.
            s_a_vecs = []
            for mi in range_constexpr(m_repeat):
                sa_idx = m_blk_list[mi] * fx.Index(scale_k) + kb
                sa = buffer_ops.buffer_load(sa_rsrc, sa_idx, vec_width=1, dtype=T.f32)
                sa = rocdl.readfirstlane(T.f32, sa)
                s_a_vecs.append(Vector.filled((4,), fx.Float32(sa), fx.Float32))
            s_b_vals = []
            for ni in range_constexpr(num_acc_n):
                sb_idx = n_block_for_scale[ni] * fx.Index(scale_k) + kb
                sbv = buffer_ops.buffer_load(sb_rsrc, sb_idx, vec_width=1, dtype=T.f32)
                sbv = rocdl.readfirstlane(T.f32, sbv)
                s_b_vals.append(sbv)

            if _is_gfx950:
                # Wide 16x16x128 MFMA with neutral E8M0 scale; accumulate the
                # whole scale block, then apply the FP32 scale in software.
                combined_scales = []
                for mi in range_constexpr(m_repeat):
                    mi_combined = []
                    for ni in range_constexpr(num_acc_n):
                        s_b_bc = Vector.filled(
                            (4,), fx.Float32(s_b_vals[ni]), fx.Float32
                        )
                        mi_combined.append(
                            ArithValue(s_a_vecs[mi]) * ArithValue(s_b_bc)
                        )
                    combined_scales.append(mi_combined)

                block_accs = [acc_init] * (num_acc_n * m_repeat)
                ku0 = sb * ku_per_sb
                ku1 = ku0 + 1
                b0_packs0, b0_packs1 = b_tile_in[ku0]
                b1_packs0, b1_packs1 = b_tile_in[ku1]
                col_base0 = col_offset_base_bytes + fx.Index(ku0 * 64)
                col_base1 = col_offset_base_bytes + fx.Index(ku1 * 64)

                for mi in range_constexpr(m_repeat):
                    curr_row_a_lds = lane_mod_16 + (mi * 16)
                    if a0_prefetch is not None and sb == 0 and mi == 0:
                        a0, a1 = a0_prefetch
                    else:
                        a0, a1 = lds_load_packs_k64(curr_row_a_lds, col_base0, lds_base)
                    a2, a3 = lds_load_packs_k64(curr_row_a_lds, col_base1, lds_base)
                    a128 = pack_i64x4_to_i32x8(a0, a1, a2, a3)

                    for ni in range_constexpr(num_acc_n):
                        b128 = pack_i64x4_to_i32x8(
                            b0_packs0[ni],
                            b0_packs1[ni],
                            b1_packs0[ni],
                            b1_packs1[ni],
                        )
                        acc_idx = mi * num_acc_n + ni
                        block_accs[acc_idx] = rocdl.mfma_scale_f32_16x16x128_f8f6f4(
                            mfma_res_ty,
                            [
                                a128,
                                b128,
                                block_accs[acc_idx],
                                0,
                                0,
                                0,
                                0x7F7F7F7F,
                                0,
                                0x7F7F7F7F,
                            ],
                        )

                for mi in range_constexpr(m_repeat):
                    for ni in range_constexpr(num_acc_n):
                        acc_idx = mi * num_acc_n + ni
                        current_accs[acc_idx] = math_dialect.fma(
                            block_accs[acc_idx],
                            combined_scales[mi][ni],
                            current_accs[acc_idx],
                        )
            else:
                # gfx942: narrow 16x16x32 MFMA pairs, FP32 scale per K-step.
                for ku_local in range_constexpr(ku_per_sb):
                    ku = sb * ku_per_sb + ku_local
                    k_offset_bytes = ku * 64
                    b_packs0, b_packs1 = b_tile_in[ku]

                    for mi in range_constexpr(m_repeat):
                        if (
                            a0_prefetch is not None
                            and sb == 0
                            and ku_local == 0
                            and mi == 0
                        ):
                            a0, a1 = a0_prefetch
                        else:
                            row_a_lds = lane_mod_16 + (mi * 16)
                            col_a_base_bytes = lane_div_16 * fx.Index(16) + fx.Index(
                                k_offset_bytes
                            )
                            a0, a1 = lds_load_packs_k64(
                                row_a_lds, col_a_base_bytes, lds_base
                            )

                        for ni in range_constexpr(num_acc_n):
                            acc_idx = mi * num_acc_n + ni
                            mfma_fn = rocdl.mfma_f32_16x16x32_fp8_fp8
                            mfma_mid = mfma_fn(
                                T.f32x4, [a0, b_packs0[ni], acc_init, 0, 0, 0]
                            )
                            mfma_result = mfma_fn(
                                T.f32x4, [a1, b_packs1[ni], mfma_mid, 0, 0, 0]
                            )
                            s_a_v4 = s_a_vecs[mi]
                            s_b_bc = Vector.filled(
                                (4,), fx.Float32(s_b_vals[ni]), fx.Float32
                            )
                            scaled = ArithValue(mfma_result) * ArithValue(s_a_v4)
                            current_accs[acc_idx] = math_dialect.fma(
                                scaled, s_b_bc, current_accs[acc_idx]
                            )

        return current_accs

    return compute_tile


@functools.lru_cache(maxsize=128)
def compile_fp8_blockwise_gemm(
    *,
    n: int,
    k: int,
    tile_m: int = 128,
    tile_n: int = 128,
    tile_k: int = 128,
    scale_block_m: int = 128,
    scale_block_n: int = 128,
    scale_block_k: int = 128,
    out_dtype: str = "bf16",
    waves_per_eu: int | None = None,
    b_preshuffled: bool = False,
):
    """Compile the plain FP8 blockwise GEMM kernel and return the JIT launcher.

    Runtime args of the launcher:
      ``(arg_d, arg_a, arg_b, arg_scale_a, arg_scale_b, i32_m, i32_n, i32_k,
         stream)``.
    """
    gpu_arch = get_hip_arch()
    _is_gfx950 = str(gpu_arch).startswith("gfx95")

    validate_params(
        n=n,
        k=k,
        tile_n=tile_n,
        tile_k=tile_k,
        scale_block_k=scale_block_k,
        scale_block_n=scale_block_n,
        out_dtype=out_dtype,
    )
    if scale_block_m % 16 != 0:
        raise ValueError(
            f"scale_block_m ({scale_block_m}) must be a multiple of the 16-row "
            "MFMA fragment so a fragment never straddles two M scale blocks."
        )
    # LDS budget check must run before tracing (an overflow is a hard compiler
    # error the autotuner cannot catch).
    if b_preshuffled:
        validate_lds_budget_preshuffle(
            tile_m=tile_m, tile_n=tile_n, tile_k=tile_k, arch=gpu_arch
        )
    else:
        validate_lds_budget_plain(
            tile_m=tile_m, tile_n=tile_n, tile_k=tile_k, b_pingpong=False, arch=gpu_arch
        )

    out_mlir = out_mlir_for(out_dtype)

    _c = compute_compile_constants(
        n=n,
        k=k,
        tile_m=tile_m,
        tile_n=tile_n,
        tile_k=tile_k,
        scale_block_k=scale_block_k,
        scale_block_n=scale_block_n,
    )
    total_threads = _c.total_threads
    elem_bytes = _c.elem_bytes
    num_k_tiles = _c.num_k_tiles
    scale_k = _c.scale_k
    scale_n = _c.scale_n
    sb_per_tile = _c.sb_per_tile
    k_unroll = _c.k_unroll
    kpack_bytes = _c.kpack_bytes
    tile_k_bytes = _c.tile_k_bytes
    tile_k_dwords = _c.tile_k_dwords
    chunk_i32_a = _c.chunk_i32_a
    num_a_loads = _c.num_a_loads
    chunk_i32_b = _c.chunk_i32_b
    num_b_loads = _c.num_b_loads

    if b_preshuffled:
        lds_alloc_offset, lds_tile_elems = setup_lds_allocation(
            allocator=(
                alloc := SmemAllocator(
                    None, arch=gpu_arch, global_sym_name="smem_blockwise_gemm"
                )
            ),
            tile_m=tile_m,
            tile_k=tile_k,
            tile_n=tile_n,
            elem_bytes=elem_bytes,
        )
        lds_b_offset_elems = None
    else:
        lds_alloc_offset, lds_tile_elems, lds_b_offset_elems = (
            setup_lds_allocation_plain(
                allocator=(
                    alloc := SmemAllocator(
                        None, arch=gpu_arch, global_sym_name="smem_blockwise_gemm_plain"
                    )
                ),
                tile_m=tile_m,
                tile_n=tile_n,
                tile_k=tile_k,
                elem_bytes=elem_bytes,
                b_pingpong=False,
            )
        )
    allocator = alloc

    _variant = "preshuffle" if b_preshuffled else "plain"
    module_name = (
        f"fp8_blockwise_gemm_{_variant}_{out_dtype}"
        f"_n{n}_k{k}_t{tile_m}x{tile_n}x{tile_k}"
    ).replace("-", "_")

    ku_per_sb = scale_block_k // 64

    @flyc.kernel(name=module_name)
    def blockwise_gemm_kernel(
        arg_d: fx.Tensor,
        arg_a: fx.Tensor,
        arg_b: fx.Tensor,
        arg_scale_a: fx.Tensor,
        arg_scale_b: fx.Tensor,
        i32_m: fx.Int32,
        i32_n: fx.Int32,
        i32_k: fx.Int32,
    ):
        m_in = fx.Index(i32_m)
        n_in = fx.Index(i32_n)
        k_in = fx.Index(i32_k)

        tx = gpu.thread_id("x")
        by = gpu.block_id("x")  # N-block index
        bx = gpu.block_id("y")  # M-tile index

        by_n = by * fx.Index(tile_n)
        bx_m = bx * fx.Index(tile_m)

        # Wave/lane decomposition (256 threads = 4 waves x 64 lanes).
        layout_wave_lane = fx.make_layout((4, 64), stride=(64, 1))
        coord_wave_lane = fx.idx2crd(fx.Int32(tx), layout_wave_lane)
        wave_id = fx.get(coord_wave_lane, 0)
        lane_id = fx.get(coord_wave_lane, 1)

        layout_lane16 = fx.make_layout((4, 16), stride=(16, 1))
        coord_lane16 = fx.idx2crd(fx.Int32(lane_id), layout_lane16)
        lane_div_16 = fx.get(coord_lane16, 0)
        lane_mod_16 = fx.get(coord_lane16, 1)

        # ---- LDS ----
        base_ptr = allocator.get_base()
        lds_a = SmemPtr(
            base_ptr, lds_alloc_offset, T.f8, shape=(2 * tile_m * tile_k,)
        ).get()
        layout_lds = fx.make_layout((tile_m, tile_k), stride=(tile_k, 1))
        lds_base_pong = fx.Index(0)
        lds_base_ping = fx.Index(lds_tile_elems)

        if const_expr(not b_preshuffled):
            lds_b = SmemPtr(
                base_ptr,
                lds_alloc_offset,
                T.f8,
                shape=((lds_b_offset_elems + tile_n * tile_k),),
            ).get()
            layout_lds_b = fx.make_layout((tile_n, tile_k), stride=(tile_k, 1))
            lds_base_b = fx.Index(lds_b_offset_elems)

        lds_out = SmemPtr(
            base_ptr, lds_alloc_offset, out_mlir(), shape=(tile_m * tile_n,)
        ).get()

        # ---- Buffer resources ----
        a_rsrc = buffer_ops.create_buffer_resource(
            arg_a, max_size=False, num_records_bytes=m_in * k_in
        )
        b_rsrc = buffer_ops.create_buffer_resource(
            arg_b, max_size=False, num_records_bytes=n_in * k_in
        )
        d_rsrc = buffer_ops.create_buffer_resource(
            arg_d, max_size=False, num_records_bytes=m_in * n_in * fx.Index(2)
        )
        # scale_a [ceil(M/scale_block_m), scale_k] at mb*scale_k+kb;
        # scale_b [scale_n, scale_k] at nb*scale_k+kb (both FP32).
        sa_rows = (m_in + fx.Index(scale_block_m - 1)) // fx.Index(scale_block_m)
        sa_rsrc = buffer_ops.create_buffer_resource(
            arg_scale_a,
            max_size=False,
            num_records_bytes=sa_rows * fx.Index(scale_k * 4),
        )
        sb_rsrc = buffer_ops.create_buffer_resource(
            arg_scale_b,
            max_size=False,
            num_records_bytes=fx.Index(scale_n * scale_k * 4),
        )

        _t = compute_mfma_tiling(tile_m=tile_m, tile_n=tile_n)
        m_repeat = _t.m_repeat
        n_per_wave = _t.n_per_wave
        num_acc_n = _t.num_acc_n

        acc_init, accs = init_accumulators(_t.num_accs)

        _nb = make_n_block_coords(
            wave_id=wave_id,
            by_n=by_n,
            group_idx=fx.Index(0),
            num_groups_in=fx.Index(1),
            n_in=n_in,
            k_in=k_in,
            lane_mod_16=lane_mod_16,
            kpack_bytes=kpack_bytes,
            elem_bytes=elem_bytes,
            scale_block_n=scale_block_n,
            scale_k=scale_k,
            n_per_wave=n_per_wave,
            num_acc_n=num_acc_n,
        )
        n_tile_base = _nb.n_tile_base
        n_block_for_scale = _nb.n_block_for_scale
        layout_b = _nb.layout_b
        n_blk_list = _nb.n_blk_list
        n_intra_list = _nb.n_intra_list

        (
            prefetch_a_tile,
            store_a_tile_to_lds,
            a_row_local,
            a_col_local_i32,
            k_blocks16,
        ) = make_a_tile_loaders(
            a_rsrc=a_rsrc,
            lds_a=lds_a,
            layout_lds=layout_lds,
            bx_m=bx_m,
            tx=tx,
            tile_m=tile_m,
            tile_k=tile_k,
            tile_k_bytes=tile_k_bytes,
            tile_k_dwords=tile_k_dwords,
            chunk_i32_a=chunk_i32_a,
            num_a_loads=num_a_loads,
            total_threads=total_threads,
            elem_bytes=elem_bytes,
            k_in=k_in,
        )

        lds_load_packs_k64 = make_lds_loader(
            lds_a=lds_a, layout_lds=layout_lds, k_blocks16=k_blocks16
        )

        row_a_lds_base = lane_mod_16
        col_offset_base_bytes = lane_div_16 * fx.Index(16)

        if const_expr(b_preshuffled):
            load_b_tile = make_b_loader(
                arg_b=arg_b,
                b_rsrc=b_rsrc,
                layout_b=layout_b,
                n_blk_list=n_blk_list,
                n_intra_list=n_intra_list,
                lane_div_16=lane_div_16,
                kpack_bytes=kpack_bytes,
                elem_bytes=elem_bytes,
                k_unroll=k_unroll,
                num_acc_n=num_acc_n,
            )
        else:
            (
                prefetch_b_tile,
                store_b_tile_to_lds,
                _b_row_local,
                _b_col_local_i32,
                k_blocks16_b,
            ) = make_b_tile_loaders(
                b_rsrc=b_rsrc,
                lds_b=lds_b,
                layout_lds_b=layout_lds_b,
                by_n=by_n,
                group_idx=fx.Index(0),
                tx=tx,
                tile_n=tile_n,
                tile_k=tile_k,
                tile_k_bytes=tile_k_bytes,
                tile_k_dwords=tile_k_dwords,
                chunk_i32_b=chunk_i32_b,
                num_b_loads=num_b_loads,
                total_threads=total_threads,
                elem_bytes=elem_bytes,
                n_in=n_in,
                k_in=k_in,
            )
            lds_load_b_packs_k64 = make_lds_b_loader(
                lds_b=lds_b, layout_lds_b=layout_lds_b, k_blocks16_b=k_blocks16_b
            )
            load_b_tile_from_lds = make_plain_b_tile(
                lds_load_b_packs_k64=lds_load_b_packs_k64,
                lane_mod_16=lane_mod_16,
                n_tile_base=n_tile_base,
                col_offset_base_bytes=col_offset_base_bytes,
                k_unroll=k_unroll,
                num_acc_n=num_acc_n,
            )

        mfma_res_ty = T.f32x4

        # Per-fragment M scale-block index: mi's 16 rows lie in one scale block
        # (16 | scale_block_m), so m_blk is a single scalar per mi.
        m_blk_list = []
        for mi in range_constexpr(m_repeat):
            m_row0 = bx_m + fx.Index(mi * 16)
            m_blk_list.append(m_row0 // fx.Index(scale_block_m))

        # compute_tile (native 2D block scaling): built by a module-level factory
        # so its Python control flow isn't AST-rewritten by @flyc.kernel.
        compute_tile = make_compute_tile_blockwise(
            _is_gfx950=_is_gfx950,
            sb_per_tile=sb_per_tile,
            ku_per_sb=ku_per_sb,
            scale_k=scale_k,
            m_repeat=m_repeat,
            num_acc_n=num_acc_n,
            acc_init=acc_init,
            mfma_res_ty=mfma_res_ty,
            m_blk_list=m_blk_list,
            n_block_for_scale=n_block_for_scale,
            sa_rsrc=sa_rsrc,
            sb_rsrc=sb_rsrc,
            lds_load_packs_k64=lds_load_packs_k64,
            col_offset_base_bytes=col_offset_base_bytes,
            lane_mod_16=lane_mod_16,
            lane_div_16=lane_div_16,
        )

        def prefetch_scales(_k_tile_idx_py):
            # Scales are loaded inside compute_tile (software FP32 path).
            return None

        rocdl.sched_barrier(0)

        if const_expr(b_preshuffled):
            hot_loop_scheduler = make_hot_loop_scheduler(
                _use_hw_scale=False,
                sb_per_tile=sb_per_tile,
                m_repeat=m_repeat,
                num_acc_n=num_acc_n,
                k_unroll=k_unroll,
                num_a_loads=num_a_loads,
                ku_per_sb=ku_per_sb,
            )
            run_kloop = make_pingpong_kloop(
                num_k_tiles=num_k_tiles,
                tile_k=tile_k,
                prefetch_a_tile=prefetch_a_tile,
                store_a_tile_to_lds=store_a_tile_to_lds,
                load_b_tile=load_b_tile,
                prefetch_scales=prefetch_scales,
                compute_tile=compute_tile,
                hot_loop_scheduler=hot_loop_scheduler,
                lds_load_packs_k64=lds_load_packs_k64,
                lds_base_pong=lds_base_pong,
                lds_base_ping=lds_base_ping,
                row_a_lds_base=row_a_lds_base,
                col_offset_base_bytes=col_offset_base_bytes,
            )
        else:
            run_kloop = make_kloop_plain(
                num_k_tiles=num_k_tiles,
                tile_k=tile_k,
                prefetch_a_tile=prefetch_a_tile,
                store_a_tile_to_lds=store_a_tile_to_lds,
                prefetch_b_tile=prefetch_b_tile,
                store_b_tile_to_lds=store_b_tile_to_lds,
                load_b_tile_from_lds=load_b_tile_from_lds,
                prefetch_scales=prefetch_scales,
                compute_tile=compute_tile,
                lds_base_pong=lds_base_pong,
                lds_base_b=lds_base_b,
            )
        accs = run_kloop(accs)

        # ---- Epilogue: CShuffle vectorized stores, mask partial-M tail ----
        c_n = n_in
        e_vec = 4 if (tile_n % (32 * 4)) == 0 else 2

        write_row_to_lds, store_pair = make_epilogue_writers(
            accs=accs, d_rsrc=d_rsrc, out_mlir=out_mlir, e_vec=e_vec, c_n=c_n
        )

        m_in_i32 = i32_m

        def precompute_row(*, row_local, row):
            row_i32 = arith.index_cast(T.i32, row)
            row_valid = arith.cmpi(arith.CmpIPredicate.ult, row_i32, m_in_i32)
            return (None, row_valid)

        mfma_epilog(
            use_cshuffle=True,
            arith=arith,
            vector=vector,
            gpu=gpu,
            scf=scf,
            range_constexpr=range_constexpr,
            tile_m=tile_m,
            tile_n=tile_n,
            e_vec=e_vec,
            m_repeat=m_repeat,
            num_acc_n=num_acc_n,
            tx=tx,
            lane_div_16=lane_div_16,
            lane_mod_16=lane_mod_16,
            bx_m=bx_m,
            by_n=by_n,
            n_tile_base=n_tile_base,
            lds_out=lds_out,
            frag_elem_type=out_mlir(),
            write_row_to_lds=write_row_to_lds,
            precompute_row=precompute_row,
            store_pair=store_pair,
        )

    @flyc.jit
    def launch_blockwise_gemm(
        arg_d: fx.Tensor,
        arg_a: fx.Tensor,
        arg_b: fx.Tensor,
        arg_scale_a: fx.Tensor,
        arg_scale_b: fx.Tensor,
        i32_m: fx.Int32,
        i32_n: fx.Int32,
        i32_k: fx.Int32,
        stream: fx.Stream,
    ):
        allocator.finalized = False
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            allocator.finalize()

        n_v = fx.Index(i32_n)
        m_v = fx.Index(i32_m)
        gx = n_v // fx.Index(tile_n)  # N-blocks
        gy = (m_v + fx.Index(tile_m - 1)) // fx.Index(tile_m)  # M-tiles

        launcher = blockwise_gemm_kernel(
            arg_d, arg_a, arg_b, arg_scale_a, arg_scale_b, i32_m, i32_n, i32_k
        )
        if waves_per_eu is not None:
            _wpe = int(waves_per_eu)
            if _wpe >= 1:
                for op in ctx.gpu_module_body.operations:
                    if hasattr(op, "attributes") and op.OPERATION_NAME == "gpu.func":
                        op.attributes["rocdl.waves_per_eu"] = ir.IntegerAttr.get(
                            T.i32, _wpe
                        )
        launcher.launch(grid=(gx, gy, 1), block=(total_threads, 1, 1), stream=stream)

    return launch_blockwise_gemm
