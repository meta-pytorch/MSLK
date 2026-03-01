# @nolint # fbcode
# pyre-ignore-all-errors
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# ===========================================================================
# BF16 Grouped GEMM (CuteDSL, Blackwell)
# ===========================================================================
# Reference: genai/msl/ops/kernels/cute/gemm/grouped_gemm.py
#
# Constraints:
# * Input dtype: bf16 (A and B must match)
# * Output dtype: bf16 or fp32
# * Contiguous dimension of each tensor must be at least 16-byte aligned
# * Batch-size per group must be 1
# * A=row-major (K-major), B=row-major (K-major), C=row-major (N-major)
# ===========================================================================

from inspect import isclass
from typing import Type, Union

import cuda.bindings.driver as cuda
import torch

import cutlass
import cutlass.cute as cute
from cutlass.cute.nvgpu import cpasync, tcgen05
import cutlass.torch as cutlass_torch
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
from cutlass.cute.runtime import from_dlpack

_GROUPED_FPROP: int = 0


class BF16GroupedGemmKernel:
    """Blackwell SM100 warp-specialized persistent grouped GEMM kernel for BF16."""

    def __init__(
        self,
        acc_dtype: type[cutlass.Numeric],
        use_2cta_instrs: bool,
        mma_tiler_mn: tuple[int, int],
        cluster_shape_mn: tuple[int, int],
        tensormap_update_mode: utils.TensorMapUpdateMode = utils.TensorMapUpdateMode.SMEM,
    ):
        self.acc_dtype: Type[cutlass.Numeric] = acc_dtype
        self.use_2cta_instrs = use_2cta_instrs
        self.cluster_shape_mn = cluster_shape_mn
        self.mma_tiler = (*mma_tiler_mn, 1)
        self.cta_group = (
            tcgen05.CtaGroup.TWO if use_2cta_instrs else tcgen05.CtaGroup.ONE
        )

        self.tensormap_update_mode = tensormap_update_mode
        self.delegate_tensormap_ab_init = (
            tensormap_update_mode == utils.TensorMapUpdateMode.SMEM
        )

        self.num_mcast_ctas_a = 1
        self.num_mcast_ctas_b = 1
        self.is_a_mcast = False
        self.is_b_mcast = False

        self.occupancy = 1
        self.epilog_warp_id = (0, 1, 2, 3)
        self.mma_warp_id = 4
        self.tma_warp_id = 5
        self.threads_per_cta = 32 * len(
            (self.mma_warp_id, self.tma_warp_id, *self.epilog_warp_id)
        )
        self.cta_sync_bar_id = 0
        self.epilog_sync_bar_id = 1
        self.tmem_ptr_sync_bar_id = 2
        self.tensormap_ab_init_bar_id = 4
        self.smem_capacity = utils.get_smem_capacity_in_bytes("sm_100")
        self.num_tma_load_bytes = 0

    def _setup_attributes(self):
        tiled_mma = sm100_utils.make_trivial_tiled_mma(
            self.a_dtype,
            self.a_major_mode,
            self.b_major_mode,
            self.acc_dtype,
            self.cta_group,
            self.mma_tiler[:2],
        )

        mma_inst_shape_k = cute.size(tiled_mma.shape_mnk, mode=[2])
        mma_inst_tile_k = 8
        self.mma_tiler = (
            self.mma_tiler[0],
            self.mma_tiler[1],
            mma_inst_shape_k * mma_inst_tile_k,
        )
        self.cta_tile_shape_mnk = (
            self.mma_tiler[0] // cute.size(tiled_mma.thr_id.shape),
            self.mma_tiler[1],
            self.mma_tiler[2],
        )
        self.cluster_tile_shape_mnk = tuple(
            x * y for x, y in zip(self.cta_tile_shape_mnk, (*self.cluster_shape_mn, 1))
        )

        self.cluster_layout_vmnk = cute.tiled_divide(
            cute.make_layout((*self.cluster_shape_mn, 1)),
            (tiled_mma.thr_id.shape,),
        )

        self.num_mcast_ctas_a = cute.size(self.cluster_layout_vmnk.shape[2])
        self.num_mcast_ctas_b = cute.size(self.cluster_layout_vmnk.shape[1])
        self.is_a_mcast = self.num_mcast_ctas_a > 1
        self.is_b_mcast = self.num_mcast_ctas_b > 1

        self.epi_tile = utils.compute_epilogue_tile_shape(
            self.cta_tile_shape_mnk,
            self.use_2cta_instrs,
            self.c_layout,
            self.c_dtype,
        )

        (
            self.num_acc_stage,
            self.num_ab_stage,
            self.num_epi_stage,
        ) = self._compute_stages(
            tiled_mma,
            self.mma_tiler,
            self.a_dtype,
            self.b_dtype,
            self.epi_tile,
            self.c_dtype,
            self.c_layout,
            self.smem_capacity,
            self.occupancy,
        )

        self.a_smem_layout_staged = sm100_utils.make_smem_layout_a(
            tiled_mma,
            self.mma_tiler,
            self.a_dtype,
            self.num_ab_stage,
        )
        self.b_smem_layout_staged = sm100_utils.make_smem_layout_b(
            tiled_mma,
            self.mma_tiler,
            self.b_dtype,
            self.num_ab_stage,
        )
        self.epi_smem_layout_staged = sm100_utils.make_smem_layout_epi(
            self.c_dtype,
            self.c_layout,
            self.epi_tile,
            self.num_epi_stage,
        )

        tensor_smem_bytes = self._get_tensor_smem_bytes(
            self.a_smem_layout_staged,
            self.a_dtype,
            self.b_smem_layout_staged,
            self.b_dtype,
            self.epi_smem_layout_staged,
            self.c_dtype,
        )
        mbar_smem_bytes = self._get_mbar_smem_bytes(
            num_acc_stage=self.num_acc_stage,
            num_ab_stage=self.num_ab_stage,
            num_epi_stage=self.num_epi_stage,
        )
        tensormap_smem_bytes = self._get_tensormap_smem_bytes(
            self.tensormap_update_mode
        )
        if (
            mbar_smem_bytes
            + tensormap_smem_bytes
            + BF16GroupedGemmKernel.tensor_memory_management_bytes
            > self.reserved_smem_bytes
        ):
            raise ValueError(
                f"smem consumption for mbar and tensormap {mbar_smem_bytes + tensormap_smem_bytes} exceeds the "
                f"reserved smem bytes {self.reserved_smem_bytes}"
            )

        self.num_tmem_alloc_cols = self._compute_num_tmem_alloc_cols(
            tiled_mma, self.mma_tiler, self.num_acc_stage
        )

    @cute.jit
    def __call__(
        self,
        tensor_a: cute.Tensor,
        tensor_b: cute.Tensor,
        tensor_c: cute.Tensor,
        group_count: cutlass.Constexpr[int],
        problem_shape_mnkl: cute.Tensor,
        strides_abc: cute.Tensor,
        tensor_address_abc: cute.Tensor,
        estimate_total_num_clusters: int,
        total_num_clusters: cute.Tensor,
        tensormap_cute_tensor: cute.Tensor,
        max_active_clusters: cutlass.Constexpr[int],
        output_accum: cutlass.Constexpr[bool],
        stream: cuda.CUstream,
        split_sizes: cute.Tensor,
        input_problem_sizes_gmnk: tuple[int, int, int, int],
        input_strides_abc: tuple[tuple[int, int], tuple[int, int], tuple[int, int]],
        input_block_sizes_mn: cutlass.Constexpr[tuple[int, int]],
    ):
        self.a_dtype = tensor_a.element_type
        self.b_dtype = tensor_b.element_type
        self.c_dtype = tensor_c.element_type
        self.a_major_mode = utils.LayoutEnum.from_tensor(tensor_a).mma_major_mode()
        self.b_major_mode = utils.LayoutEnum.from_tensor(tensor_b).mma_major_mode()
        self.c_layout = utils.LayoutEnum.from_tensor(tensor_c)
        if cutlass.const_expr(self.a_dtype != self.b_dtype):
            raise TypeError(f"Type mismatch: {self.a_dtype} != {self.b_dtype}")

        self._setup_attributes()

        tiled_mma = sm100_utils.make_trivial_tiled_mma(
            self.a_dtype,
            self.a_major_mode,
            self.b_major_mode,
            self.acc_dtype,
            self.cta_group,
            self.mma_tiler[:2],
        )
        atom_thr_size = cute.size(tiled_mma.thr_id.shape)

        # Setup TMA load for A
        a_op = sm100_utils.cluster_shape_to_tma_atom_A(
            self.cluster_shape_mn, tiled_mma.thr_id
        )
        a_smem_layout = cute.slice_(self.a_smem_layout_staged, (None, None, None, 0))
        tma_atom_a, tma_tensor_a = cute.nvgpu.make_tiled_tma_atom_A(
            a_op,
            tensor_a,
            a_smem_layout,
            self.mma_tiler,
            tiled_mma,
            self.cluster_layout_vmnk.shape,
        )

        # Setup TMA load for B
        b_op = sm100_utils.cluster_shape_to_tma_atom_B(
            self.cluster_shape_mn, tiled_mma.thr_id
        )
        b_smem_layout = cute.slice_(self.b_smem_layout_staged, (None, None, None, 0))
        tma_atom_b, tma_tensor_b = cute.nvgpu.make_tiled_tma_atom_B(
            b_op,
            tensor_b,
            b_smem_layout,
            self.mma_tiler,
            tiled_mma,
            self.cluster_layout_vmnk.shape,
        )

        a_copy_size = cute.size_in_bytes(self.a_dtype, a_smem_layout)
        b_copy_size = cute.size_in_bytes(self.b_dtype, b_smem_layout)
        self.num_tma_load_bytes = (a_copy_size + b_copy_size) * atom_thr_size

        # Setup TMA store for C
        tma_atom_c = None
        tma_tensor_c = None
        c_cta_v_layout = cute.composition(
            cute.make_identity_layout(tensor_c.shape), self.epi_tile
        )
        epi_smem_layout = cute.slice_(self.epi_smem_layout_staged, (None, None, 0))
        tma_atom_c, tma_tensor_c = cpasync.make_tiled_tma_atom(
            cpasync.CopyReduceBulkTensorTileS2GOp()
            if output_accum
            else cpasync.CopyBulkTensorTileS2GOp(),
            tensor_c,
            epi_smem_layout,
            c_cta_v_layout,
        )

        estimate_tile_sched_params = self._compute_tile_sched(
            estimate_total_num_clusters,
            self.cluster_shape_mn,
        )
        grid = self._compute_grid(
            estimate_tile_sched_params,
            max_active_clusters,
        )

        self.buffer_align_bytes = 1024
        self.size_tensormap_in_i64 = (
            0
            if self.tensormap_update_mode == utils.TensorMapUpdateMode.GMEM
            else BF16GroupedGemmKernel.num_tensormaps
            * BF16GroupedGemmKernel.bytes_per_tensormap
            // 8
        )

        @cute.struct
        class SharedStorage:
            tensormap_buffer: cute.struct.MemRange[
                cutlass.Int64, self.size_tensormap_in_i64
            ]
            ab_full_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_ab_stage]
            ab_empty_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_ab_stage]
            acc_full_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_acc_stage]
            acc_empty_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_acc_stage]
            tmem_dealloc_mbar_ptr: cutlass.Int64
            tmem_holding_buf: cutlass.Int32
            sC: cute.struct.Align[
                cute.struct.MemRange[
                    self.c_dtype,
                    cute.cosize(self.epi_smem_layout_staged.outer),
                ],
                self.buffer_align_bytes,
            ]
            sA: cute.struct.Align[
                cute.struct.MemRange[
                    self.a_dtype, cute.cosize(self.a_smem_layout_staged.outer)
                ],
                self.buffer_align_bytes,
            ]
            sB: cute.struct.Align[
                cute.struct.MemRange[
                    self.b_dtype, cute.cosize(self.b_smem_layout_staged.outer)
                ],
                self.buffer_align_bytes,
            ]

        self.shared_storage = SharedStorage

        # Launch prepare kernel then main kernel
        self.prepare_kernel(
            input_ptr_a=tensor_a.iterator,
            input_ptr_b=tensor_b.iterator,
            input_ptr_c=tensor_c.iterator,
            split_sizes=split_sizes,
            input_problem_sizes_gmnk=input_problem_sizes_gmnk,
            input_strides_abc=input_strides_abc,
            input_block_sizes_mn=input_block_sizes_mn,
            output_problem_sizes_mnkl=problem_shape_mnkl,
            output_strides_abc=strides_abc,
            output_ptrs_abc=tensor_address_abc,
            output_total_num_clusters=total_num_clusters,
        ).launch(
            grid=(group_count, 1, 1),
            block=(32, 1, 1),
            cluster=(1, 1, 1),
            stream=stream,
        )

        self.kernel(
            tiled_mma=tiled_mma,
            tma_atom_a=tma_atom_a,
            mA_mkl=tma_tensor_a,
            tma_atom_b=tma_atom_b,
            mB_nkl=tma_tensor_b,
            tma_atom_c=tma_atom_c,
            mC_mnl=tma_tensor_c,
            cluster_layout_vmnk=self.cluster_layout_vmnk,
            a_smem_layout_staged=self.a_smem_layout_staged,
            b_smem_layout_staged=self.b_smem_layout_staged,
            epi_smem_layout_staged=self.epi_smem_layout_staged,
            epi_tile=self.epi_tile,
            total_num_clusters=total_num_clusters,
            group_count=group_count,
            problem_sizes_mnkl=problem_shape_mnkl,
            strides_abc=strides_abc,
            ptrs_abc=tensor_address_abc,
            tensormaps=tensormap_cute_tensor,
        ).launch(
            grid=grid,
            block=[self.threads_per_cta, 1, 1],
            cluster=(*self.cluster_shape_mn, 1),
            stream=stream,
        )
        return

    @cute.kernel
    def kernel(
        self,
        tiled_mma: cute.TiledMma,
        tma_atom_a: cute.CopyAtom,
        mA_mkl: cute.Tensor,
        tma_atom_b: cute.CopyAtom,
        mB_nkl: cute.Tensor,
        tma_atom_c: cute.CopyAtom,
        mC_mnl: cute.Tensor,
        cluster_layout_vmnk: cute.Layout,
        a_smem_layout_staged: cute.ComposedLayout,
        b_smem_layout_staged: cute.ComposedLayout,
        epi_smem_layout_staged: Union[cute.Layout, cute.ComposedLayout],
        epi_tile: cute.Tile,
        total_num_clusters: cute.Tensor,
        group_count: cutlass.Constexpr[int],
        problem_sizes_mnkl: cute.Tensor,
        strides_abc: cute.Tensor,
        ptrs_abc: cute.Tensor,
        tensormaps: cute.Tensor,
    ):
        warp_idx = cute.arch.warp_idx()
        warp_idx = cute.arch.make_warp_uniform(warp_idx)

        if warp_idx == self.tma_warp_id:
            cpasync.prefetch_descriptor(tma_atom_a)
            cpasync.prefetch_descriptor(tma_atom_b)
            cpasync.prefetch_descriptor(tma_atom_c)

        use_2cta_instrs = cute.size(tiled_mma.thr_id.shape) == 2

        bid = cute.arch.block_idx()
        mma_tile_coord_v = bid[0] % cute.size(tiled_mma.thr_id.shape)
        is_leader_cta = mma_tile_coord_v == 0
        cta_rank_in_cluster = cute.arch.make_warp_uniform(
            cute.arch.block_idx_in_cluster()
        )
        block_in_cluster_coord_vmnk = cluster_layout_vmnk.get_flat_coord(
            cta_rank_in_cluster
        )
        tidx, _, _ = cute.arch.thread_idx()

        tile_sched_params = self._compute_tile_sched(
            total_num_clusters[0], self.cluster_shape_mn
        )

        smem = utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)

        tensormap_a_smem_ptr = None
        tensormap_b_smem_ptr = None
        tensormap_c_smem_ptr = None
        if cutlass.const_expr(
            self.tensormap_update_mode == utils.TensorMapUpdateMode.SMEM
        ):
            tensormap_smem_ptr = storage.tensormap_buffer.data_ptr()
            tensormap_a_smem_ptr = tensormap_smem_ptr
            tensormap_b_smem_ptr = (
                tensormap_a_smem_ptr + BF16GroupedGemmKernel.bytes_per_tensormap // 8
            )
            tensormap_c_smem_ptr = (
                tensormap_b_smem_ptr + BF16GroupedGemmKernel.bytes_per_tensormap // 8
            )
        ab_full_mbar_ptr = storage.ab_full_mbar_ptr.data_ptr()
        ab_empty_mbar_ptr = storage.ab_empty_mbar_ptr.data_ptr()
        acc_full_mbar_ptr = storage.acc_full_mbar_ptr.data_ptr()
        acc_empty_mbar_ptr = storage.acc_empty_mbar_ptr.data_ptr()
        tmem_dealloc_mbar_ptr = storage.tmem_dealloc_mbar_ptr
        tmem_holding_buf = storage.tmem_holding_buf

        if warp_idx == self.epilog_warp_id[0]:
            for k_stage in range(self.num_ab_stage):
                num_tma_producer = self.num_mcast_ctas_a + self.num_mcast_ctas_b - 1
                with cute.arch.elect_one():
                    cute.arch.mbarrier_init(ab_full_mbar_ptr + k_stage, 1)
                    cute.arch.mbarrier_init(
                        ab_empty_mbar_ptr + k_stage, num_tma_producer
                    )
        if warp_idx == self.mma_warp_id:
            for acc_stage in range(self.num_acc_stage):
                with cute.arch.elect_one():
                    cute.arch.mbarrier_init(acc_full_mbar_ptr + acc_stage, 1)
                    cute.arch.mbarrier_init(
                        acc_empty_mbar_ptr + acc_stage, 8 if use_2cta_instrs else 4
                    )
        if use_2cta_instrs:
            if warp_idx == self.tma_warp_id:
                num_tmem_dealloc_threads = 32
                with cute.arch.elect_one():
                    cute.arch.mbarrier_init(
                        tmem_dealloc_mbar_ptr, num_tmem_dealloc_threads
                    )
        cute.arch.mbarrier_init_fence()

        if cute.size(self.cluster_shape_mn) > 1:
            cute.arch.cluster_arrive_relaxed()

        sC = storage.sC.get_tensor(
            epi_smem_layout_staged.outer, swizzle=epi_smem_layout_staged.inner
        )
        sA = storage.sA.get_tensor(
            a_smem_layout_staged.outer, swizzle=a_smem_layout_staged.inner
        )
        sB = storage.sB.get_tensor(
            b_smem_layout_staged.outer, swizzle=b_smem_layout_staged.inner
        )

        a_full_mcast_mask = None
        b_full_mcast_mask = None
        ab_empty_mcast_mask = None
        if cutlass.const_expr(self.is_a_mcast or self.is_b_mcast or use_2cta_instrs):
            a_full_mcast_mask = cpasync.create_tma_multicast_mask(
                cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=2
            )
            b_full_mcast_mask = cpasync.create_tma_multicast_mask(
                cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=1
            )
            ab_empty_mcast_mask = a_full_mcast_mask | b_full_mcast_mask
        acc_full_mcast_mask = None
        if cutlass.const_expr(use_2cta_instrs):
            acc_full_mcast_mask = cute.make_layout_image_mask(
                cluster_layout_vmnk, block_in_cluster_coord_vmnk, mode=0
            )
            block_in_cluster_coord_vmnk_peer = (
                block_in_cluster_coord_vmnk[0] ^ 1,
                *block_in_cluster_coord_vmnk[1:],
            )
            a_full_mcast_mask_peer = cpasync.create_tma_multicast_mask(
                cluster_layout_vmnk, block_in_cluster_coord_vmnk_peer, mcast_mode=2
            )
            b_full_mcast_mask_peer = cpasync.create_tma_multicast_mask(
                cluster_layout_vmnk, block_in_cluster_coord_vmnk_peer, mcast_mode=1
            )
            ab_empty_mcast_mask = (
                a_full_mcast_mask_peer
                | b_full_mcast_mask_peer
                | cutlass.Int16(
                    0 if ab_empty_mcast_mask is None else ab_empty_mcast_mask
                )
            )

        gA_mkl = cute.local_tile(
            mA_mkl, cute.slice_(self.mma_tiler, (None, 0, None)), (None, None, None)
        )
        gB_nkl = cute.local_tile(
            mB_nkl, cute.slice_(self.mma_tiler, (0, None, None)), (None, None, None)
        )
        gC_mnl = cute.local_tile(
            mC_mnl, cute.slice_(self.mma_tiler, (None, None, 0)), (None, None, None)
        )

        thr_mma = tiled_mma.get_slice(mma_tile_coord_v)
        tCgA = thr_mma.partition_A(gA_mkl)
        tCgB = thr_mma.partition_B(gB_nkl)
        tCgC = thr_mma.partition_C(gC_mnl)

        a_cta_layout = cute.make_layout(
            cute.slice_(cluster_layout_vmnk, (0, 0, None, 0)).shape
        )
        tAsA, tAgA = cpasync.tma_partition(
            tma_atom_a,
            block_in_cluster_coord_vmnk[2],
            a_cta_layout,
            cute.group_modes(sA, 0, 3),
            cute.group_modes(tCgA, 0, 3),
        )
        b_cta_layout = cute.make_layout(
            cute.slice_(cluster_layout_vmnk, (0, None, 0, 0)).shape
        )
        tBsB, tBgB = cpasync.tma_partition(
            tma_atom_b,
            block_in_cluster_coord_vmnk[1],
            b_cta_layout,
            cute.group_modes(sB, 0, 3),
            cute.group_modes(tCgB, 0, 3),
        )

        tCrA = tiled_mma.make_fragment_A(sA)
        tCrB = tiled_mma.make_fragment_B(sB)
        acc_shape = tiled_mma.partition_shape_C(self.mma_tiler[:2])
        tCtAcc_fake = tiled_mma.make_fragment_C(
            cute.append(acc_shape, self.num_acc_stage)
        )

        if cute.size(self.cluster_shape_mn) > 1:
            cute.arch.cluster_wait()
        else:
            cute.arch.barrier(
                barrier_id=self.cta_sync_bar_id, number_of_threads=self.threads_per_cta
            )

        grid_dim = cute.arch.grid_dim()
        tensormap_workspace_idx = (
            bid[2] * grid_dim[1] * grid_dim[0] + bid[1] * grid_dim[0] + bid[0]
        )

        tensormap_manager = utils.TensorMapManager(
            self.tensormap_update_mode, BF16GroupedGemmKernel.bytes_per_tensormap
        )
        tensormap_a_ptr = tensormap_manager.get_tensormap_ptr(
            tensormaps[(tensormap_workspace_idx, 0, None)].iterator
        )
        tensormap_b_ptr = tensormap_manager.get_tensormap_ptr(
            tensormaps[(tensormap_workspace_idx, 1, None)].iterator
        )
        tensormap_c_ptr = tensormap_manager.get_tensormap_ptr(
            tensormaps[(tensormap_workspace_idx, 2, None)].iterator
        )
        if cutlass.const_expr(
            self.tensormap_update_mode == utils.TensorMapUpdateMode.SMEM
        ):
            tensormap_a_init_ptr = tensormap_a_smem_ptr
            tensormap_b_init_ptr = tensormap_b_smem_ptr
            tensormap_c_init_ptr = tensormap_c_smem_ptr
        else:
            tensormap_a_init_ptr = tensormap_a_ptr
            tensormap_b_init_ptr = tensormap_b_ptr
            tensormap_c_init_ptr = tensormap_c_ptr

        # -- TMA load warp --
        if warp_idx == self.tma_warp_id:
            if cutlass.const_expr(self.delegate_tensormap_ab_init == False):
                tensormap_manager.init_tensormap_from_atom(
                    tma_atom_a, tensormap_a_init_ptr, self.tma_warp_id
                )
                tensormap_manager.init_tensormap_from_atom(
                    tma_atom_b, tensormap_b_init_ptr, self.tma_warp_id
                )

            tile_sched = utils.StaticPersistentTileScheduler.create(
                tile_sched_params, bid, grid_dim
            )
            group_gemm_ts_helper = utils.GroupedGemmTileSchedulerHelper(
                group_count,
                tile_sched_params,
                self.cluster_tile_shape_mnk,
                utils.create_initial_search_state(),
            )
            tensormap_init_done = cutlass.Boolean(False)
            total_k_tile_cnt = cutlass.Int32(0)
            last_group_idx = cutlass.Int32(-1)
            work_tile = tile_sched.initial_work_tile_info()
            while work_tile.is_valid_tile:
                cur_tile_coord = work_tile.tile_idx
                grouped_gemm_cta_tile_info = group_gemm_ts_helper.delinearize_z(
                    cur_tile_coord,
                    problem_sizes_mnkl,
                )
                cur_k_tile_cnt = grouped_gemm_cta_tile_info.cta_tile_count_k
                cur_group_idx = grouped_gemm_cta_tile_info.group_idx
                is_group_changed = cur_group_idx != last_group_idx
                if is_group_changed:
                    real_tensor_a = self.make_tensor_for_tensormap_update(
                        cur_group_idx,
                        self.a_dtype,
                        (
                            grouped_gemm_cta_tile_info.problem_shape_m,
                            grouped_gemm_cta_tile_info.problem_shape_n,
                            grouped_gemm_cta_tile_info.problem_shape_k,
                        ),
                        strides_abc,
                        ptrs_abc,
                        0,
                    )
                    real_tensor_b = self.make_tensor_for_tensormap_update(
                        cur_group_idx,
                        self.b_dtype,
                        (
                            grouped_gemm_cta_tile_info.problem_shape_m,
                            grouped_gemm_cta_tile_info.problem_shape_n,
                            grouped_gemm_cta_tile_info.problem_shape_k,
                        ),
                        strides_abc,
                        ptrs_abc,
                        1,
                    )
                    if tensormap_init_done == False:
                        if cutlass.const_expr(self.delegate_tensormap_ab_init):
                            cute.arch.barrier(
                                barrier_id=self.tensormap_ab_init_bar_id,
                                number_of_threads=64,
                            )
                        tensormap_manager.fence_tensormap_initialization()
                        tensormap_init_done = True

                    tensormap_manager.update_tensormap(
                        (real_tensor_a, real_tensor_b),
                        (tma_atom_a, tma_atom_b),
                        (tensormap_a_ptr, tensormap_b_ptr),
                        self.tma_warp_id,
                        (tensormap_a_smem_ptr, tensormap_b_smem_ptr),
                    )

                mma_tile_coord_mnl = (
                    grouped_gemm_cta_tile_info.cta_tile_idx_m
                    // cute.size(tiled_mma.thr_id.shape),
                    grouped_gemm_cta_tile_info.cta_tile_idx_n,
                    0,
                )

                tAgA_slice = tAgA[
                    (None, mma_tile_coord_mnl[0], None, mma_tile_coord_mnl[2])
                ]
                tBgB_slice = tBgB[
                    (None, mma_tile_coord_mnl[1], None, mma_tile_coord_mnl[2])
                ]

                num_prev_k_blk = total_k_tile_cnt
                total_k_tile_cnt += cur_k_tile_cnt

                tma_wr_k_tile = cutlass.Int32(0)
                smem_wr_buffer = (num_prev_k_blk + tma_wr_k_tile) % self.num_ab_stage
                tma_wr_ab_empty_phase = (
                    num_prev_k_blk + tma_wr_k_tile
                ) // self.num_ab_stage % 2 ^ 1
                peek_ab_empty_status = cute.arch.mbarrier_conditional_try_wait(
                    tma_wr_k_tile < cur_k_tile_cnt,
                    ab_empty_mbar_ptr + smem_wr_buffer,
                    tma_wr_ab_empty_phase,
                )
                if is_group_changed:
                    tensormap_manager.fence_tensormap_update(tensormap_a_ptr)
                    tensormap_manager.fence_tensormap_update(tensormap_b_ptr)

                for k_tile in cutlass.range(0, cur_k_tile_cnt, 1, unroll=1):
                    tma_wr_k_tile_next = tma_wr_k_tile + 1
                    smem_wr_buffer_next = (
                        num_prev_k_blk + tma_wr_k_tile_next
                    ) % self.num_ab_stage
                    tma_wr_ab_empty_phase_next = (
                        tma_wr_ab_empty_phase ^ 1
                        if smem_wr_buffer_next == 0
                        else tma_wr_ab_empty_phase
                    )

                    smem_full_mbar_ptr = ab_full_mbar_ptr + smem_wr_buffer

                    if peek_ab_empty_status == 0:
                        cute.arch.mbarrier_wait(
                            ab_empty_mbar_ptr + smem_wr_buffer, tma_wr_ab_empty_phase
                        )

                    if is_leader_cta:
                        with cute.arch.elect_one():
                            cute.arch.mbarrier_arrive_and_expect_tx(
                                smem_full_mbar_ptr, self.num_tma_load_bytes
                            )

                    cute.copy(
                        tma_atom_a,
                        tAgA_slice[(None, tma_wr_k_tile)],
                        tAsA[(None, smem_wr_buffer)],
                        tma_bar_ptr=smem_full_mbar_ptr,
                        mcast_mask=a_full_mcast_mask,
                        tma_desc_ptr=tensormap_manager.get_tensormap_ptr(
                            tensormap_a_ptr,
                            cute.AddressSpace.generic,
                        ),
                    )
                    cute.copy(
                        tma_atom_b,
                        tBgB_slice[(None, tma_wr_k_tile)],
                        tBsB[(None, smem_wr_buffer)],
                        tma_bar_ptr=smem_full_mbar_ptr,
                        mcast_mask=b_full_mcast_mask,
                        tma_desc_ptr=tensormap_manager.get_tensormap_ptr(
                            tensormap_b_ptr,
                            cute.AddressSpace.generic,
                        ),
                    )

                    peek_ab_empty_status = cute.arch.mbarrier_conditional_try_wait(
                        tma_wr_k_tile_next < cur_k_tile_cnt,
                        ab_empty_mbar_ptr + smem_wr_buffer_next,
                        tma_wr_ab_empty_phase_next,
                    )

                    tma_wr_k_tile = tma_wr_k_tile_next
                    smem_wr_buffer = smem_wr_buffer_next
                    tma_wr_ab_empty_phase = tma_wr_ab_empty_phase_next

                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()
                last_group_idx = cur_group_idx

        # -- MMA warp --
        if warp_idx == self.mma_warp_id:
            if cutlass.const_expr(self.delegate_tensormap_ab_init):
                tensormap_manager.init_tensormap_from_atom(
                    tma_atom_a, tensormap_a_init_ptr, self.mma_warp_id
                )
                tensormap_manager.init_tensormap_from_atom(
                    tma_atom_b, tensormap_b_init_ptr, self.mma_warp_id
                )
                cute.arch.barrier(
                    barrier_id=self.tensormap_ab_init_bar_id, number_of_threads=64
                )
            tmem_ptr_read_threads = 32 * len((self.mma_warp_id, *self.epilog_warp_id))
            cute.arch.barrier(
                barrier_id=self.tmem_ptr_sync_bar_id,
                number_of_threads=tmem_ptr_read_threads,
            )

            tmem_ptr = cute.arch.retrieve_tmem_ptr(
                self.acc_dtype,
                alignment=16,
                ptr_to_buffer_holding_addr=tmem_holding_buf,
            )
            tCtAcc_base = cute.make_tensor(tmem_ptr, tCtAcc_fake.layout)

            tile_sched = utils.StaticPersistentTileScheduler.create(
                tile_sched_params, bid, grid_dim
            )
            group_gemm_ts_helper = utils.GroupedGemmTileSchedulerHelper(
                group_count,
                tile_sched_params,
                self.cluster_tile_shape_mnk,
                utils.create_initial_search_state(),
            )

            work_tile = tile_sched.initial_work_tile_info()
            total_k_tile_cnt = cutlass.Int32(0)
            while work_tile.is_valid_tile:
                cur_tile_coord = work_tile.tile_idx
                (
                    cur_k_tile_cnt,
                    cur_group_idx,
                ) = group_gemm_ts_helper.search_cluster_tile_count_k(
                    cur_tile_coord,
                    problem_sizes_mnkl,
                )
                acc_buf_idx = tile_sched.num_tiles_executed % self.num_acc_stage
                tCtAcc = tCtAcc_base[(None, None, None, acc_buf_idx)]

                num_prev_k_blk = total_k_tile_cnt
                total_k_tile_cnt += cur_k_tile_cnt

                mma_rd_k_tile = cutlass.Int32(0)
                smem_rd_buffer = (num_prev_k_blk + mma_rd_k_tile) % self.num_ab_stage
                need_check_rd_buffer_full = (
                    mma_rd_k_tile < cur_k_tile_cnt and is_leader_cta
                )
                mma_rd_ab_full_phase = (
                    (num_prev_k_blk + mma_rd_k_tile) // self.num_ab_stage % 2
                )
                peek_ab_full_status = cute.arch.mbarrier_conditional_try_wait(
                    need_check_rd_buffer_full,
                    ab_full_mbar_ptr + smem_rd_buffer,
                    mma_rd_ab_full_phase,
                )

                if is_leader_cta:
                    acc_empty_phase = (
                        tile_sched.num_tiles_executed // self.num_acc_stage % 2 ^ 1
                    )
                    cute.arch.mbarrier_wait(
                        acc_empty_mbar_ptr + acc_buf_idx, acc_empty_phase
                    )

                tiled_mma.set(tcgen05.Field.ACCUMULATE, False)

                for k_tile in range(cur_k_tile_cnt):
                    mma_rd_k_tile_next = cutlass.Int32(k_tile + 1)
                    smem_rd_buffer_next = (
                        num_prev_k_blk + mma_rd_k_tile_next
                    ) % self.num_ab_stage
                    mma_rd_ab_full_phase_next = (
                        mma_rd_ab_full_phase ^ 1
                        if smem_rd_buffer_next == 0
                        else mma_rd_ab_full_phase
                    )
                    if is_leader_cta:
                        if peek_ab_full_status == 0:
                            cute.arch.mbarrier_wait(
                                ab_full_mbar_ptr + smem_rd_buffer, mma_rd_ab_full_phase
                            )

                        num_kblocks = cute.size(tCrA, mode=[2])
                        for kblock_idx in cutlass.range(num_kblocks, unroll_full=True):
                            kblock_coord = (None, None, kblock_idx, smem_rd_buffer)

                            cute.gemm(
                                tiled_mma,
                                tCtAcc,
                                tCrA[kblock_coord],
                                tCrB[kblock_coord],
                                tCtAcc,
                            )
                            tiled_mma.set(tcgen05.Field.ACCUMULATE, True)

                        with cute.arch.elect_one():
                            tcgen05.commit(
                                ab_empty_mbar_ptr + smem_rd_buffer,
                                ab_empty_mcast_mask,
                                self.cta_group,
                            )

                    peek_ab_full_status = cute.arch.mbarrier_conditional_try_wait(
                        mma_rd_k_tile_next < cur_k_tile_cnt and is_leader_cta,
                        ab_full_mbar_ptr + smem_rd_buffer_next,
                        mma_rd_ab_full_phase_next,
                    )

                    mma_rd_k_tile = mma_rd_k_tile_next
                    smem_rd_buffer = smem_rd_buffer_next
                    mma_rd_ab_full_phase = mma_rd_ab_full_phase_next

                if is_leader_cta:
                    with cute.arch.elect_one():
                        tcgen05.commit(
                            acc_full_mbar_ptr + acc_buf_idx,
                            acc_full_mcast_mask,
                            self.cta_group,
                        )

                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()

        # -- Epilogue warps --
        if warp_idx < self.mma_warp_id:
            tensormap_manager.init_tensormap_from_atom(
                tma_atom_c,
                tensormap_c_init_ptr,
                self.epilog_warp_id[0],
            )
            if warp_idx == self.epilog_warp_id[0]:
                cute.arch.alloc_tmem(
                    self.num_tmem_alloc_cols,
                    tmem_holding_buf,
                    is_two_cta=use_2cta_instrs,
                )

            tmem_ptr_read_threads = 32 * len((self.mma_warp_id, *self.epilog_warp_id))
            cute.arch.barrier(
                barrier_id=self.tmem_ptr_sync_bar_id,
                number_of_threads=tmem_ptr_read_threads,
            )

            tmem_ptr = cute.arch.retrieve_tmem_ptr(
                self.acc_dtype,
                alignment=16,
                ptr_to_buffer_holding_addr=tmem_holding_buf,
            )
            tCtAcc_base = cute.make_tensor(tmem_ptr, tCtAcc_fake.layout)

            epi_tidx = tidx
            (
                tiled_copy_t2r,
                tTR_tAcc_base,
                tTR_rAcc,
            ) = self.epilog_tmem_copy_and_partition(
                epi_tidx, tCtAcc_base, tCgC, epi_tile, use_2cta_instrs
            )

            tTR_rC = cute.make_fragment(tTR_rAcc.shape, self.c_dtype)
            tiled_copy_r2s, tRS_rC, tRS_sC = self.epilog_smem_copy_and_partition(
                tiled_copy_t2r, tTR_rC, epi_tidx, sC
            )
            (
                tma_atom_c,
                bSG_sC,
                bSG_gC_partitioned,
            ) = self.epilog_gmem_copy_and_partition(tma_atom_c, tCgC, epi_tile, sC)

            tile_sched = utils.StaticPersistentTileScheduler.create(
                tile_sched_params, bid, grid_dim
            )
            group_gemm_ts_helper = utils.GroupedGemmTileSchedulerHelper(
                group_count,
                tile_sched_params,
                self.cluster_tile_shape_mnk,
                utils.create_initial_search_state(),
            )

            work_tile = tile_sched.initial_work_tile_info()
            tensormap_manager.fence_tensormap_initialization()
            total_k_tile_cnt = cutlass.Int32(0)
            last_group_idx = cutlass.Int32(-1)
            while work_tile.is_valid_tile:
                cur_tile_coord = work_tile.tile_idx
                grouped_gemm_cta_tile_info = group_gemm_ts_helper.delinearize_z(
                    cur_tile_coord,
                    problem_sizes_mnkl,
                )
                cur_group_idx = grouped_gemm_cta_tile_info.group_idx
                is_group_changed = cur_group_idx != last_group_idx
                if is_group_changed:
                    real_tensor_c = self.make_tensor_for_tensormap_update(
                        cur_group_idx,
                        self.c_dtype,
                        (
                            grouped_gemm_cta_tile_info.problem_shape_m,
                            grouped_gemm_cta_tile_info.problem_shape_n,
                            grouped_gemm_cta_tile_info.problem_shape_k,
                        ),
                        strides_abc,
                        ptrs_abc,
                        2,
                    )
                    tensormap_manager.update_tensormap(
                        ((real_tensor_c),),
                        ((tma_atom_c),),
                        ((tensormap_c_ptr),),
                        self.epilog_warp_id[0],
                        (tensormap_c_smem_ptr,),
                    )

                mma_tile_coord_mnl = (
                    grouped_gemm_cta_tile_info.cta_tile_idx_m
                    // cute.size(tiled_mma.thr_id.shape),
                    grouped_gemm_cta_tile_info.cta_tile_idx_n,
                    0,
                )
                cur_k_tile_cnt = grouped_gemm_cta_tile_info.cta_tile_count_k
                total_k_tile_cnt += cur_k_tile_cnt

                bSG_gC = bSG_gC_partitioned[
                    (
                        None,
                        None,
                        None,
                        *mma_tile_coord_mnl,
                    )
                ]

                acc_buf_idx = tile_sched.num_tiles_executed % self.num_acc_stage
                tTR_tAcc = tTR_tAcc_base[(None, None, None, None, None, acc_buf_idx)]

                acc_full_phase = tile_sched.num_tiles_executed // self.num_acc_stage % 2
                cute.arch.mbarrier_wait(acc_full_mbar_ptr + acc_buf_idx, acc_full_phase)

                tTR_tAcc = cute.group_modes(tTR_tAcc, 3, cute.rank(tTR_tAcc))
                bSG_gC = cute.group_modes(bSG_gC, 1, cute.rank(bSG_gC))
                if is_group_changed:
                    if warp_idx == self.epilog_warp_id[0]:
                        tensormap_manager.fence_tensormap_update(tensormap_c_ptr)

                subtile_cnt = cute.size(tTR_tAcc.shape, mode=[3])
                num_prev_subtiles = tile_sched.num_tiles_executed * subtile_cnt
                for subtile_idx in range(subtile_cnt):
                    tTR_tAcc_mn = tTR_tAcc[(None, None, None, subtile_idx)]
                    cute.copy(tiled_copy_t2r, tTR_tAcc_mn, tTR_rAcc)

                    acc_vec = tiled_copy_r2s.retile(tTR_rAcc).load()
                    if cur_k_tile_cnt == 0:
                        acc_vec = cute.zeros_like(acc_vec)
                    tRS_rC.store(acc_vec.to(self.c_dtype))
                    epi_buffer = (num_prev_subtiles + subtile_idx) % self.num_epi_stage
                    cute.copy(
                        tiled_copy_r2s,
                        tRS_rC,
                        tRS_sC[(None, None, None, epi_buffer)],
                    )
                    cute.arch.fence_proxy(
                        cute.arch.ProxyKind.async_shared,
                        space=cute.arch.SharedSpace.shared_cta,
                    )
                    epilog_threads = 32 * len(self.epilog_warp_id)
                    cute.arch.barrier(
                        barrier_id=self.epilog_sync_bar_id,
                        number_of_threads=epilog_threads,
                    )
                    if warp_idx == self.epilog_warp_id[0]:
                        cute.copy(
                            tma_atom_c,
                            bSG_sC[(None, epi_buffer)],
                            bSG_gC[(None, subtile_idx)],
                            tma_desc_ptr=tensormap_manager.get_tensormap_ptr(
                                tensormap_c_ptr,
                                cute.AddressSpace.generic,
                            ),
                        )
                        cute.arch.cp_async_bulk_commit_group()
                        cute.arch.cp_async_bulk_wait_group(
                            self.num_epi_stage - 1, read=True
                        )
                    cute.arch.barrier(
                        barrier_id=self.epilog_sync_bar_id,
                        number_of_threads=epilog_threads,
                    )
                with cute.arch.elect_one():
                    cute.arch.mbarrier_arrive(
                        acc_empty_mbar_ptr + acc_buf_idx,
                        cta_rank_in_cluster // 2 * 2 if use_2cta_instrs else None,
                    )

                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()
                last_group_idx = cur_group_idx

            if warp_idx == self.epilog_warp_id[0]:
                cute.arch.relinquish_tmem_alloc_permit(is_two_cta=use_2cta_instrs)
            epilog_threads = 32 * len(self.epilog_warp_id)
            cute.arch.barrier(
                barrier_id=self.epilog_sync_bar_id, number_of_threads=epilog_threads
            )
            if warp_idx == self.epilog_warp_id[0]:
                if use_2cta_instrs:
                    cute.arch.mbarrier_arrive(
                        tmem_dealloc_mbar_ptr, cta_rank_in_cluster ^ 1
                    )
                    cute.arch.mbarrier_wait(tmem_dealloc_mbar_ptr, 0)
                cute.arch.dealloc_tmem(
                    tmem_ptr, self.num_tmem_alloc_cols, is_two_cta=use_2cta_instrs
                )

            if warp_idx == self.epilog_warp_id[0]:
                cute.arch.mbarrier_wait(
                    (ab_empty_mbar_ptr + ((total_k_tile_cnt - 1) % self.num_ab_stage)),
                    (((total_k_tile_cnt - 1) // self.num_ab_stage) % 2),
                )

    @cute.jit
    def make_tensor_for_tensormap_update(
        self,
        group_idx: cutlass.Int32,
        dtype: Type[cutlass.Numeric],
        problem_shape_mnk: tuple[cutlass.Int32, cutlass.Int32, cutlass.Int32],
        strides_abc: cute.Tensor,
        tensor_address_abc: cute.Tensor,
        tensor_index: int,
    ):
        ptr_i64 = tensor_address_abc[(group_idx, tensor_index)]
        if cutlass.const_expr(
            not isclass(dtype) or not issubclass(dtype, cutlass.Numeric)
        ):
            raise TypeError(
                f"dtype must be a type of cutlass.Numeric, got {type(dtype)}"
            )
        tensor_gmem_ptr = cute.make_ptr(
            dtype, ptr_i64, cute.AddressSpace.gmem, assumed_align=16
        )

        strides_tensor_gmem = strides_abc[(group_idx, tensor_index, None)]
        strides_tensor_reg = cute.make_fragment(
            cute.make_layout(2),
            strides_abc.element_type,
        )
        cute.autovec_copy(strides_tensor_gmem, strides_tensor_reg)
        stride_mn = strides_tensor_reg[0]
        stride_k = strides_tensor_reg[1]
        c1 = cutlass.Int32(1)
        c0 = cutlass.Int32(0)

        if cutlass.const_expr(tensor_index == 0):  # tensor A
            m = problem_shape_mnk[0]
            k = problem_shape_mnk[2]
            return cute.make_tensor(
                tensor_gmem_ptr,
                cute.make_layout((m, k, c1), stride=(stride_mn, stride_k, c0)),
            )
        elif cutlass.const_expr(tensor_index == 1):  # tensor B
            n = problem_shape_mnk[1]
            k = problem_shape_mnk[2]
            return cute.make_tensor(
                tensor_gmem_ptr,
                cute.make_layout((n, k, c1), stride=(stride_mn, stride_k, c0)),
            )
        else:  # tensor C
            m = problem_shape_mnk[0]
            n = problem_shape_mnk[1]
            return cute.make_tensor(
                tensor_gmem_ptr,
                cute.make_layout((m, n, c1), stride=(stride_mn, stride_k, c0)),
            )

    def epilog_tmem_copy_and_partition(
        self,
        tidx: cutlass.Int32,
        tAcc: cute.Tensor,
        gC_mnl: cute.Tensor,
        epi_tile: cute.Tile,
        use_2cta_instrs: Union[cutlass.Boolean, bool],
    ) -> tuple[cute.TiledCopy, cute.Tensor, cute.Tensor]:
        copy_atom_t2r = sm100_utils.get_tmem_load_op(
            self.cta_tile_shape_mnk,
            self.c_layout,
            self.c_dtype,
            self.acc_dtype,
            epi_tile,
            use_2cta_instrs,
        )
        tAcc_epi = cute.flat_divide(
            tAcc[((None, None), 0, 0, None)],
            epi_tile,
        )
        tiled_copy_t2r = tcgen05.make_tmem_copy(
            copy_atom_t2r, tAcc_epi[(None, None, 0, 0, 0)]
        )

        thr_copy_t2r = tiled_copy_t2r.get_slice(tidx)
        tTR_tAcc = thr_copy_t2r.partition_S(tAcc_epi)

        gC_mnl_epi = cute.flat_divide(
            gC_mnl[((None, None), 0, 0, None, None, None)], epi_tile
        )
        tTR_gC = thr_copy_t2r.partition_D(gC_mnl_epi)
        tTR_rAcc = cute.make_fragment(
            tTR_gC[(None, None, None, 0, 0, 0, 0, 0)].shape, self.acc_dtype
        )
        return tiled_copy_t2r, tTR_tAcc, tTR_rAcc

    def epilog_smem_copy_and_partition(
        self,
        tiled_copy_t2r: cute.TiledCopy,
        tTR_rC: cute.Tensor,
        tidx: cutlass.Int32,
        sC: cute.Tensor,
    ) -> tuple[cute.TiledCopy, cute.Tensor, cute.Tensor]:
        copy_atom_r2s = sm100_utils.get_smem_store_op(
            self.c_layout, self.c_dtype, self.acc_dtype, tiled_copy_t2r
        )
        tiled_copy_r2s = cute.make_tiled_copy_D(copy_atom_r2s, tiled_copy_t2r)
        thr_copy_r2s = tiled_copy_r2s.get_slice(tidx)
        tRS_sC = thr_copy_r2s.partition_D(sC)
        tRS_rC = tiled_copy_r2s.retile(tTR_rC)
        return tiled_copy_r2s, tRS_rC, tRS_sC

    def epilog_gmem_copy_and_partition(
        self,
        tma_atom_c: cute.CopyAtom,
        gC_mnl: cute.Tensor,
        epi_tile: cute.Tile,
        sC: cute.Tensor,
    ) -> tuple[cute.CopyAtom, cute.Tensor, cute.Tensor]:
        gC_epi = cute.flat_divide(
            gC_mnl[((None, None), 0, 0, None, None, None)], epi_tile
        )
        sC_for_tma_partition = cute.group_modes(sC, 0, 2)
        gC_for_tma_partition = cute.group_modes(gC_epi, 0, 2)
        bSG_sC, bSG_gC = cpasync.tma_partition(
            tma_atom_c,
            0,
            cute.make_layout(1),
            sC_for_tma_partition,
            gC_for_tma_partition,
        )
        return tma_atom_c, bSG_sC, bSG_gC

    @staticmethod
    def _compute_stages(
        tiled_mma: cute.TiledMma,
        mma_tiler_mnk: tuple[int, int, int],
        a_dtype: type[cutlass.Numeric],
        b_dtype: type[cutlass.Numeric],
        epi_tile: cute.Tile,
        c_dtype: type[cutlass.Numeric],
        c_layout: utils.LayoutEnum,
        smem_capacity: int,
        occupancy: int,
    ) -> tuple[int, int, int]:
        num_acc_stage = 2
        num_epi_stage = 2

        a_smem_layout_stage_one = sm100_utils.make_smem_layout_a(
            tiled_mma, mma_tiler_mnk, a_dtype, 1
        )
        b_smem_layout_staged_one = sm100_utils.make_smem_layout_b(
            tiled_mma, mma_tiler_mnk, b_dtype, 1
        )
        epi_smem_layout_staged_one = sm100_utils.make_smem_layout_epi(
            c_dtype, c_layout, epi_tile, 1
        )
        ab_bytes_per_stage = cute.size_in_bytes(
            a_dtype, a_smem_layout_stage_one
        ) + cute.size_in_bytes(b_dtype, b_smem_layout_staged_one)

        epi_bytes_per_stage = cute.size_in_bytes(c_dtype, epi_smem_layout_staged_one)
        epi_bytes = epi_bytes_per_stage * num_epi_stage

        num_ab_stage = (
            smem_capacity // occupancy
            - BF16GroupedGemmKernel.reserved_smem_bytes
            - epi_bytes
        ) // ab_bytes_per_stage

        remaining_smem = (
            smem_capacity
            - occupancy * ab_bytes_per_stage * num_ab_stage
            - occupancy * (BF16GroupedGemmKernel.reserved_smem_bytes + epi_bytes)
        )
        num_epi_stage += remaining_smem // (occupancy * epi_bytes_per_stage)
        return num_acc_stage, num_ab_stage, num_epi_stage

    @staticmethod
    def _compute_tile_sched(
        total_num_clusters: int,
        cluster_shape_mn: tuple[int, int],
    ) -> utils.PersistentTileSchedulerParams:
        problem_shape_ntile_mnl = (
            cluster_shape_mn[0],
            cluster_shape_mn[1],
            cutlass.Int32(total_num_clusters),
        )
        tile_sched_params = utils.PersistentTileSchedulerParams(
            problem_shape_ntile_mnl, (*cluster_shape_mn, 1)
        )
        return tile_sched_params

    @staticmethod
    def _compute_grid(
        tile_sched_params: utils.PersistentTileSchedulerParams,
        max_active_clusters: cutlass.Constexpr[int],
    ) -> tuple[int, int, int]:
        grid = utils.StaticPersistentTileScheduler.get_grid_shape(
            tile_sched_params, max_active_clusters
        )
        return grid

    @staticmethod
    def _get_mbar_smem_bytes(**kwargs_stages: int) -> int:
        num_barriers_per_stage = 2
        num_bytes_per_barrier = 8
        mbar_smem_consumption = sum(
            [
                num_barriers_per_stage * num_bytes_per_barrier * stage
                for stage in kwargs_stages.values()
            ]
        )
        return mbar_smem_consumption

    @staticmethod
    def _get_tensormap_smem_bytes(
        tensormap_update_mode: utils.TensorMapUpdateMode,
    ) -> int:
        if tensormap_update_mode == utils.TensorMapUpdateMode.GMEM:
            return 0
        elif tensormap_update_mode == utils.TensorMapUpdateMode.SMEM:
            return (
                BF16GroupedGemmKernel.bytes_per_tensormap
                * BF16GroupedGemmKernel.num_tensormaps
            )
        else:
            raise ValueError(f"Invalid tensormap update mode: {tensormap_update_mode}")

    @staticmethod
    def _get_tensor_smem_bytes(
        a_smem_layout_staged: cute.Layout,
        a_dtype: Type[cutlass.Numeric],
        b_smem_layout_staged: cute.Layout,
        b_dtype: Type[cutlass.Numeric],
        epi_smem_layout_staged: cute.Layout,
        c_dtype: Type[cutlass.Numeric],
    ) -> int:
        ab_bytes = cute.size_in_bytes(
            a_dtype, a_smem_layout_staged
        ) + cute.size_in_bytes(b_dtype, b_smem_layout_staged)
        epi_bytes = cute.size_in_bytes(c_dtype, epi_smem_layout_staged)
        return ab_bytes + epi_bytes

    @staticmethod
    def _compute_num_tmem_alloc_cols(
        tiled_mma: cute.TiledMma,
        mma_tiler: tuple[int, int, int],
        num_acc_stage: int,
    ) -> int:
        acc_shape = tiled_mma.partition_shape_C(mma_tiler[:2])
        tCtAcc_fake = tiled_mma.make_fragment_C(cute.append(acc_shape, num_acc_stage))
        num_tmem_alloc_cols = utils.get_num_tmem_alloc_cols(tCtAcc_fake)
        return num_tmem_alloc_cols

    reserved_smem_bytes = 1024
    bytes_per_tensormap = 128
    num_tensormaps = 3
    tensor_memory_management_bytes = 12

    @cute.kernel
    def prepare_kernel(
        self,
        input_ptr_a: cute.Pointer,
        input_ptr_b: cute.Pointer,
        input_ptr_c: cute.Pointer,
        split_sizes: cute.Tensor,
        input_problem_sizes_gmnk: tuple[int, int, int, int],
        input_strides_abc: tuple[tuple[int, int], tuple[int, int], tuple[int, int]],
        input_block_sizes_mn: tuple[cutlass.Constexpr[int], cutlass.Constexpr[int]],
        output_problem_sizes_mnkl: cute.Tensor,
        output_strides_abc: cute.Tensor,
        output_ptrs_abc: cute.Tensor,
        output_total_num_clusters: cute.Tensor,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        bdim, _, _ = cute.arch.block_dim()

        if tidx == 0:
            num_groups = split_sizes.shape[0]

            G, M, N, K = input_problem_sizes_gmnk

            (
                (stride_am, stride_ak),
                (stride_bn, stride_bk),
                (stride_cm, stride_cn),
            ) = input_strides_abc

            block_size_m, block_size_n = input_block_sizes_mn

            off_a = cutlass.Int64(0)
            off_b = cutlass.Int64(0)
            off_c = cutlass.Int64(0)
            assert (
                input_ptr_a.dtype == cutlass.Float16
                or input_ptr_a.dtype == cutlass.BFloat16
            )
            assert (
                input_ptr_b.dtype == cutlass.Float16
                or input_ptr_b.dtype == cutlass.BFloat16
            )
            element_size_a = input_ptr_a.dtype.width // 8
            element_size_b = input_ptr_b.dtype.width // 8
            assert (
                input_ptr_c.dtype == cutlass.Float32
                or input_ptr_c.dtype == cutlass.BFloat16
                or input_ptr_c.dtype == cutlass.Float16
            )
            element_size_c = input_ptr_c.dtype.width // 8

            total_num_clusters = cutlass.Int32(0)

            for g in range(num_groups):
                if bidx == g:
                    output_ptrs_abc[g, 0] = input_ptr_a.toint() + off_a * element_size_a
                    output_ptrs_abc[g, 1] = input_ptr_b.toint() + off_b * element_size_b
                    output_ptrs_abc[g, 2] = input_ptr_c.toint() + off_c * element_size_c

                chunk_size = split_sizes[g].to(cutlass.Int32)
                m = chunk_size
                n = N
                k = K

                off_a += chunk_size * stride_am
                off_b += N * stride_bn
                off_c += chunk_size * stride_cm

                total_num_clusters += ((m + block_size_m - 1) // block_size_m) * (
                    (n + block_size_n - 1) // block_size_n
                )

                if bidx == g:
                    output_problem_sizes_mnkl[g, 0] = m
                    output_problem_sizes_mnkl[g, 1] = n
                    output_problem_sizes_mnkl[g, 2] = k
                    output_problem_sizes_mnkl[g, 3] = 1

                    output_strides_abc[g, 0, 0] = stride_am
                    output_strides_abc[g, 0, 1] = stride_ak
                    output_strides_abc[g, 1, 0] = stride_bn
                    output_strides_abc[g, 1, 1] = stride_bk
                    output_strides_abc[g, 2, 0] = stride_cm
                    output_strides_abc[g, 2, 1] = stride_cn

            if bidx == 0:
                output_total_num_clusters[0] = total_num_clusters


# ---------------------------------------------------------------------------
# Caches and hardware info for BF16 grouped GEMM
# ---------------------------------------------------------------------------
_GROUPED_HARDWARE_INFO: utils.HardwareInfo | None = None
_GROUPED_CACHE_KEY_TYPE = tuple[
    int,
    tuple[torch.dtype, ...],
    torch.dtype,
    tuple[int | None, int | None, int | None, int | None],
    tuple[int, int, int],
    bool,
]
_GROUPED_SCHEDULE_METADATA_CACHE: dict[_GROUPED_CACHE_KEY_TYPE, tuple[int, int]] = {}
_GROUPED_COMPILED_KERNEL_CACHE: dict[
    _GROUPED_CACHE_KEY_TYPE,
    cutlass.cutlass_dsl.cuda_jit_executor.CudaDialectJitCompiledFunction,
] = {}


def _launch_bf16_grouped_gemm(
    problem_sizes_gmnk: tuple[int, int, int, int],
    torch_tensors_abc: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    torch_split_sizes: torch.Tensor,
    output_accum: bool,
) -> None:
    """Internal launcher for BF16 grouped GEMM kernel."""
    mma_tiler_mn: tuple[int, int] = (256, 256)
    cluster_shape_mn: tuple[int, int] = (2, 1)
    use_2cta_instrs: bool = True
    acc_dtype: Type[cutlass.Numeric] = cutlass.Float32
    tensormap_update_mode: utils.TensorMapUpdateMode = utils.TensorMapUpdateMode.SMEM

    G: int
    M: int
    N: int
    K: int
    G, M, N, K = problem_sizes_gmnk

    problem_type = _GROUPED_FPROP
    shape_key: tuple[int | None, int | None, int | None, int | None] = (G, None, N, K)
    leading_dims: tuple[int, int, int] = (1, 1, 1)

    cute_tensors_abc: list[cute.Tensor] = [
        from_dlpack(
            x.unsqueeze(-1).detach(),
            assumed_align=16,
        ).mark_layout_dynamic(leading_dim=d)
        for x, d in zip(torch_tensors_abc, leading_dims)
    ]
    strides_abc: tuple[tuple[int, int], ...] = tuple(
        (x.stride(0), x.stride(1)) for x in torch_tensors_abc
    )
    block_sizes_mnk: tuple[int, int] = mma_tiler_mn

    device: torch.device = torch_tensors_abc[0].device

    torch_problem_sizes_mnkl: torch.Tensor = torch.empty(
        (G, 4), dtype=torch.int32, device=device
    )
    cute_problem_sizes_mnkl: cute.Tensor = from_dlpack(
        torch_problem_sizes_mnkl, assumed_align=16
    )

    torch_strides_abc: torch.Tensor = torch.empty(
        (G, 3, 2), dtype=torch.int32, device=device
    )
    cute_strides_abc: cute.Tensor = from_dlpack(torch_strides_abc, assumed_align=16)

    torch_ptrs_abc: torch.Tensor = torch.empty((G, 3), dtype=torch.int64, device=device)
    cute_ptrs_abc: cute.Tensor = from_dlpack(torch_ptrs_abc, assumed_align=16)

    torch_num_total_clusters: torch.Tensor = torch.empty(
        (1,), dtype=torch.int32, device=device
    )
    cute_num_total_clusters: cute.Tensor = from_dlpack(
        torch_num_total_clusters, assumed_align=16
    )

    cute_split_sizes: cute.Tensor = from_dlpack(torch_split_sizes, assumed_align=16)

    cache_key: _GROUPED_CACHE_KEY_TYPE = (
        problem_type,
        tuple(t.dtype for t in torch_tensors_abc),
        torch_split_sizes.dtype,
        shape_key,
        leading_dims,
        output_accum,
    )

    global _GROUPED_HARDWARE_INFO
    hardware_info: utils.HardwareInfo
    if _GROUPED_HARDWARE_INFO is None:
        hardware_info = utils.HardwareInfo()
        _GROUPED_HARDWARE_INFO = hardware_info
    else:
        hardware_info = _GROUPED_HARDWARE_INFO

    sm_count: int
    max_active_clusters: int
    if cache_key not in _GROUPED_SCHEDULE_METADATA_CACHE:
        ctx_result = cuda.cuCtxGetCurrent()
        current_ctx = ctx_result[1]
        need_ctx_pop = current_ctx is None or int(current_ctx) == 0
        if need_ctx_pop:
            cuda.cuCtxPushCurrent(hardware_info.context)
        try:
            sm_count = hardware_info.get_max_active_clusters(1)
            max_active_clusters = hardware_info.get_max_active_clusters(
                cluster_shape_mn[0] * cluster_shape_mn[1]
            )
        finally:
            if need_ctx_pop:
                cuda.cuCtxPopCurrent()
        _GROUPED_SCHEDULE_METADATA_CACHE[cache_key] = (sm_count, max_active_clusters)
    else:
        sm_count, max_active_clusters = _GROUPED_SCHEDULE_METADATA_CACHE[cache_key]

    num_tensormap_buffers: int = sm_count
    tensormap_shape: tuple[int, int, int] = (
        num_tensormap_buffers,
        BF16GroupedGemmKernel.num_tensormaps,
        BF16GroupedGemmKernel.bytes_per_tensormap // 8,
    )

    torch_tensor_of_tensormap: torch.Tensor = torch.empty(
        tensormap_shape, dtype=torch.int64, device=device
    )
    cute_tensor_of_tensormap: cute.Tensor = from_dlpack(
        torch_tensor_of_tensormap, assumed_align=16
    )

    def compute_cluster_tile_shape(
        mma_tiler_mn: tuple[int, int],
        cluster_shape_mn: tuple[int, int],
        use_2cta_instrs: bool,
    ) -> tuple[int, int]:
        cta_tile_shape_mn: list[int] = list(mma_tiler_mn)
        if use_2cta_instrs:
            cta_tile_shape_mn[0] = cta_tile_shape_mn[0] // 2
        return tuple(x * y for x, y in zip(cta_tile_shape_mn, cluster_shape_mn))

    cluster_tile_shape_mn: tuple[int, int] = compute_cluster_tile_shape(
        mma_tiler_mn, cluster_shape_mn, use_2cta_instrs,
    )

    def cdiv(x: int, y: int) -> int:
        return (x + y - 1) // y

    BLOCK_SIZE_M, BLOCK_SIZE_N = cluster_tile_shape_mn
    estimate_total_num_clusters: int = cdiv(M, BLOCK_SIZE_M) * cdiv(N, BLOCK_SIZE_N)

    current_stream: cuda.CUstream = cutlass_torch.current_stream()

    grouped_gemm: BF16GroupedGemmKernel = BF16GroupedGemmKernel(
        acc_dtype,
        use_2cta_instrs,
        mma_tiler_mn,
        cluster_shape_mn,
        tensormap_update_mode,
    )

    compiled_grouped_gemm: (
        cutlass.cutlass_dsl.cuda_jit_executor.CudaDialectJitCompiledFunction
    )
    if cache_key not in _GROUPED_COMPILED_KERNEL_CACHE:
        compiled_grouped_gemm = cute.compile(
            grouped_gemm,
            tensor_a=cute_tensors_abc[0],
            tensor_b=cute_tensors_abc[1],
            tensor_c=cute_tensors_abc[2],
            group_count=G,
            problem_shape_mnkl=cute_problem_sizes_mnkl,
            strides_abc=cute_strides_abc,
            tensor_address_abc=cute_ptrs_abc,
            estimate_total_num_clusters=estimate_total_num_clusters,
            total_num_clusters=cute_num_total_clusters,
            tensormap_cute_tensor=cute_tensor_of_tensormap,
            max_active_clusters=max_active_clusters,
            stream=current_stream,
            output_accum=output_accum,
            split_sizes=cute_split_sizes,
            input_problem_sizes_gmnk=problem_sizes_gmnk,
            input_strides_abc=strides_abc,
            input_block_sizes_mn=block_sizes_mnk,
        )
        _GROUPED_COMPILED_KERNEL_CACHE[cache_key] = compiled_grouped_gemm
    else:
        compiled_grouped_gemm = _GROUPED_COMPILED_KERNEL_CACHE[cache_key]

    compiled_grouped_gemm(
        tensor_a=cute_tensors_abc[0],
        tensor_b=cute_tensors_abc[1],
        tensor_c=cute_tensors_abc[2],
        problem_shape_mnkl=cute_problem_sizes_mnkl,
        strides_abc=cute_strides_abc,
        tensor_address_abc=cute_ptrs_abc,
        estimate_total_num_clusters=estimate_total_num_clusters,
        total_num_clusters=cute_num_total_clusters,
        tensormap_cute_tensor=cute_tensor_of_tensormap,
        stream=current_stream,
        split_sizes=cute_split_sizes,
        input_problem_sizes_gmnk=problem_sizes_gmnk,
        input_strides_abc=strides_abc,
    )


def bf16_grouped_gemm(
    x: torch.Tensor,  # [GM, K]
    w: torch.Tensor,  # [G, N, K]
    split_sizes: torch.Tensor,  # [G]
    *,
    y: torch.Tensor | None = None,  # [GM, N]
) -> torch.Tensor:
    """Perform BF16 grouped GEMM using CuteDSL kernel on Blackwell.

    Computes: y_g = x_g @ w_g.T for each group g, where x_g is a sub-block
    of x with ``split_sizes[g]`` rows.

    Args:
        x: Input tensor of shape (GM, K) in bfloat16.
        w: Weight tensor of shape (G, N, K) in bfloat16.
        split_sizes: 1-D int32 tensor of shape (G,) specifying the M-dimension
            size of each group.  sum(split_sizes) == GM.
        y: Optional pre-allocated output tensor of shape (GM, N) in bfloat16.

    Returns:
        Output tensor y of shape (GM, N) in bfloat16.
    """
    x_dtype = x.dtype
    w_dtype = w.dtype

    assert x_dtype == w_dtype
    assert x_dtype in (torch.float16, torch.bfloat16)

    y_dtype = y.dtype if y is not None else x_dtype
    assert y_dtype in (torch.float16, torch.bfloat16, torch.float32)

    x_shape = x.shape
    w_shape = w.shape

    GM, K = x_shape
    G, N, K_ = w_shape
    assert K == K_
    assert split_sizes.shape == (G,)

    y_shape = (GM, N) if y is None else y.shape
    assert y_shape == (GM, N)

    if y is None:
        y = torch.empty(y_shape, device=x.device, dtype=y_dtype)
    if GM == 0 or N == 0 or K == 0:
        return y

    w = w.reshape(G * N, K)
    x_stride = x.stride()
    w_stride = w.stride()
    y_stride = y.stride()

    assert x_stride[0] * x_dtype.itemsize % 128 == 0
    assert x_stride[1] == 1
    assert w_stride[0] * w_dtype.itemsize % 128 == 0
    assert w_stride[1] == 1
    assert split_sizes.is_contiguous()
    assert y_stride[0] * y_dtype.itemsize % 128 == 0
    assert y_stride[1] == 1

    _launch_bf16_grouped_gemm(
        problem_sizes_gmnk=(G, GM, N, K),
        torch_tensors_abc=(x, w, y),
        torch_split_sizes=split_sizes,
        output_accum=False,
    )

    return y
