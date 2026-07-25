# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import functools
import warnings
from typing import Optional

import torch
import triton
import triton.language as tl
from triton.runtime import driver  # @manual

try:
    # @manual=//triton:triton
    from triton.tools.tensor_descriptor import TensorDescriptor

    TMA_AVAILABLE = True
except ImportError:
    TMA_AVAILABLE = False
    pass


def _grouped_gemm_set_block_size_hook(nargs):
    BLOCK_M = nargs["BLOCK_SIZE_M"]
    BLOCK_N = nargs["BLOCK_SIZE_N"]
    BLOCK_K = nargs["BLOCK_SIZE_K"]
    if nargs["USE_TMA_LOAD"]:
        nargs["a_desc_ptr"].block_shape = [BLOCK_M, BLOCK_K]
        nargs["b_desc_ptr"].block_shape = [BLOCK_N, BLOCK_K]


_NV_CONFIGS = [
    triton.Config(
        {
            "BLOCK_SIZE_M": block_size_m,
            "BLOCK_SIZE_N": block_size_n,
            "BLOCK_SIZE_K": block_size_k,
            "NUM_CONSUMER_GROUPS": 1,
        },
        num_stages=num_stages,
        num_warps=num_warps,
        num_ctas=num_ctas,
        pre_hook=_grouped_gemm_set_block_size_hook,
    )
    for block_size_m in [64, 128]
    for block_size_n in [64, 128, 256]
    for block_size_k in [64, 128, 256]
    for num_stages in [3, 4]
    for num_warps in [4, 8]
    for num_ctas in [1]
]

if TMA_AVAILABLE:
    _NV_WS_CONFIGS = [
        triton.Config(
            {
                "BLOCK_SIZE_M": block_size_m,
                "BLOCK_SIZE_N": block_size_n,
                "BLOCK_SIZE_K": block_size_k,
                "NUM_CONSUMER_GROUPS": 1,
                "USE_TMA_STORE": use_tma_store,
            },
            num_stages=num_stages,
            num_warps=num_warps,
            num_ctas=num_ctas,
            pre_hook=_grouped_gemm_set_block_size_hook,
        )
        for block_size_m in [64, 128, 256]
        for block_size_n in [64, 128, 256]
        for block_size_k in [64, 128, 256]
        for num_stages in [2, 3, 4]
        for num_warps in [4, 8, 16]
        for num_ctas in [1]
        for use_tma_store in [False]
    ]
else:
    _NV_WS_CONFIGS = _NV_CONFIGS


_AMD_CONFIGS = [
    triton.Config(
        {
            "BLOCK_SIZE_M": block_size_m,
            "BLOCK_SIZE_N": block_size_n,
            "BLOCK_SIZE_K": block_size_k,
            "waves_per_eu": waves_per_cu,
            "matrix_instr_nonkdim": matrix_instr_nonkdim,
            "NUM_CONSUMER_GROUPS": 1,
        },
        num_stages=num_stages,
        num_warps=num_warps,
    )
    for block_size_m in [32, 64, 128]
    for block_size_n in [32, 64, 128, 256]
    for block_size_k in [128, 256]
    for num_stages in [1, 2]
    for num_warps, waves_per_cu in [(4, 1), (8, 2), (16, 4)]
    for matrix_instr_nonkdim in [16]
]


def early_config_prune(configs, named_args, dtsize=None, dtype=None, **kwargs):
    device = torch.cuda.current_device()
    # BLOCK_M, BLOCK_N, BLOCK_K, SPLIT_K, num_warps, num_stages
    if dtsize is None:
        dtsize = named_args["c_ptr"].element_size()
    if dtype is None:
        dtype = named_args["c_ptr"].dtype

    pruned_configs = []
    for config in configs:
        kw = config.kwargs
        (
            BLOCK_M,
            BLOCK_N,
            BLOCK_K,
            num_stages,
            use_tma_load_on_scales,
        ) = (
            kw["BLOCK_SIZE_M"],
            kw["BLOCK_SIZE_N"],
            kw["BLOCK_SIZE_K"],
            config.num_stages,
            kw.get("USE_TMA_LOAD_ON_SCALES", False),
        )
        G, M, N = (
            named_args["G"],
            named_args["M_BUCKET"],
            named_args["N"],
        )

        # 1. make sure we have enough smem
        max_shared_memory = driver.active.utils.get_device_properties(device)[
            "max_shared_mem"
        ]
        if torch.version.hip:
            required_shared_memory = BLOCK_N * BLOCK_K * num_stages * dtsize
        else:
            required_shared_memory = (BLOCK_M + BLOCK_N) * BLOCK_K * num_stages * dtsize
        if required_shared_memory > max_shared_memory:
            continue

        M_PER_GROUP = M // G
        MIN_M_TILES = 32 if torch.version.hip else 64
        # 2. make sure we don't load M tiles that are too big
        if BLOCK_M > MIN_M_TILES and BLOCK_M > (M_PER_GROUP * 2):
            continue
        # 3. make sure we don't load N tiles that are too small
        if BLOCK_M < 128 and BLOCK_M < (M_PER_GROUP // 2):
            continue

        num_sm = driver.active.utils.get_device_properties(device)[
            "multiprocessor_count"
        ]
        N_TILES = (N + BLOCK_N - 1) // BLOCK_N
        MIN_N_TILES = 32 if torch.version.hip else 64
        # 4. make sure we don't load N tiles that are too big
        if BLOCK_N > MIN_N_TILES and M * N_TILES < num_sm:
            continue
        # 5. make sure we don't load N tiles that are too small
        if BLOCK_N < 128 and M * N_TILES > 2 * num_sm:
            continue
        if dtsize >= 2:
            if use_tma_load_on_scales:
                continue
        pruned_configs.append(config)

    return pruned_configs


def early_config_prune_ws(configs, named_args, dtsize=None, dtype=None, **kwargs):
    device = torch.cuda.current_device()
    # BLOCK_M, BLOCK_N, BLOCK_K, SPLIT_K, num_warps, num_stages
    if dtsize is None:
        dtsize = named_args["c_ptr"].element_size()
    if dtype is None:
        dtype = named_args["c_ptr"].dtype

    pruned_configs = []
    for config in configs:
        kw = config.kwargs
        (
            BLOCK_M,
            BLOCK_N,
            BLOCK_K,
            num_stages,
            use_tma_load_on_scales,
        ) = (
            kw["BLOCK_SIZE_M"],
            kw["BLOCK_SIZE_N"],
            kw["BLOCK_SIZE_K"],
            config.num_stages,
            kw.get("USE_TMA_LOAD_ON_SCALES", False),
        )
        G, M, N = (
            named_args["G"],
            named_args["M_BUCKET"],
            named_args["N"],
        )

        # 1. make sure we have enough smem
        max_shared_memory = driver.active.utils.get_device_properties(device)[
            "max_shared_mem"
        ]
        if torch.version.hip:
            required_shared_memory = BLOCK_N * BLOCK_K * num_stages * dtsize
        else:
            required_shared_memory = (BLOCK_M + BLOCK_N) * BLOCK_K * num_stages * dtsize
        if required_shared_memory > max_shared_memory:
            continue

        M_PER_GROUP = M // G
        MIN_M_TILES = 32 if torch.version.hip else 64
        # 2. make sure we don't load M tiles that are too big
        if BLOCK_M > MIN_M_TILES and BLOCK_M > (M_PER_GROUP * 2):
            continue
        # 3. make sure we don't load N tiles that are too small
        if BLOCK_M < 128 and BLOCK_M < (M_PER_GROUP // 2):
            continue

        num_sm = driver.active.utils.get_device_properties(device)[
            "multiprocessor_count"
        ]
        N_TILES = (N + BLOCK_N - 1) // BLOCK_N
        MIN_N_TILES = 32 if torch.version.hip else 64
        # 4. make sure we don't load N tiles that are too big
        if BLOCK_N > MIN_N_TILES and M * N_TILES < num_sm:
            continue
        # 5. make sure we don't load N tiles that are too small
        if BLOCK_N < 128 and M * N_TILES > 2 * num_sm:
            continue

        if dtsize >= 2:
            if use_tma_load_on_scales:
                continue
        pruned_configs.append(config)

    return pruned_configs


@triton.autotune(
    configs=_AMD_CONFIGS if torch.version.hip else _NV_CONFIGS,
    key=["G", "M_BUCKET", "N", "K"],
    prune_configs_by={"early_config_prune": early_config_prune},
    restore_value=["c_ptr"],  # restore for scatter_add fusion
)
@triton.jit
def _mslk_grouped_gemm(
    a_desc_ptr,
    b_desc_ptr,
    c_ptr,
    scatter_add_indices,
    m_sizes,
    bias_ptr,
    token_weights_ptr,
    # problem sizes
    G: tl.constexpr,
    M_BUCKET,
    N: tl.constexpr,
    K: tl.constexpr,
    NUM_SMS: tl.constexpr,
    FUSE_SCATTER_ADD: tl.constexpr,
    USE_TMA_LOAD: tl.constexpr,
    USE_TMA_STORE: tl.constexpr,
    USE_FAST_ACCUM: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    HAS_TOKEN_WEIGHTS: tl.constexpr,
    # tile sizes
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    NUM_CONSUMER_GROUPS: tl.constexpr,
) -> None:
    tl.static_assert(
        not (FUSE_SCATTER_ADD and USE_TMA_STORE),
        "Cannot fuse scatter add with TMA store!",
    )

    tidx = tl.program_id(0)

    M_end_offset = 0
    M_end_offset = M_end_offset.to(tl.int64)  # pyre-ignore
    iterated_tiles = 0
    for g in tl.range(G):
        # Move across groups
        m_size = tl.load(m_sizes + g)

        if m_size > 0:
            M_start_offset = M_end_offset
            M_end_offset = M_start_offset + m_size
            N_start_offset = g.to(tl.int64) * N
            n_size = N

            num_m_tiles = tl.cdiv(m_size, BLOCK_SIZE_M)
            num_n_tiles = tl.cdiv(n_size, BLOCK_SIZE_N)
            num_tiles = num_m_tiles * num_n_tiles

            if USE_TMA_STORE:
                c_desc_ptr = tl.make_tensor_descriptor(
                    c_ptr + M_start_offset * N,
                    # pyrefly: ignore [bad-argument-type]
                    shape=[m_size, n_size],
                    # pyre-ignore
                    strides=[n_size, 1],
                    block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_N],
                )

            # Move across tiles
            while tidx >= iterated_tiles and tidx < iterated_tiles + num_tiles:
                gidx = tidx - iterated_tiles
                # Split M first and N second.
                tile_m_idx = gidx % num_m_tiles
                tile_n_idx = gidx // num_m_tiles

                accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

                if USE_TMA_LOAD:
                    tl.static_assert(K % BLOCK_SIZE_K == 0)
                    m_offset = (M_start_offset + tile_m_idx * BLOCK_SIZE_M).to(tl.int32)
                    n_offset = (N_start_offset + tile_n_idx * BLOCK_SIZE_N).to(tl.int32)
                    for k_offset in range(0, K, BLOCK_SIZE_K):
                        a = a_desc_ptr.load([m_offset, k_offset])
                        b = b_desc_ptr.load([n_offset, k_offset])
                        if USE_FAST_ACCUM:
                            accumulator = tl.dot(a, b.T, accumulator)
                        else:
                            accumulator += tl.dot(a, b.T)
                else:
                    offs_am = tile_m_idx * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
                    offs_bn = tile_n_idx * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
                    offs_k = tl.arange(0, BLOCK_SIZE_K)
                    a_ptrs = (
                        a_desc_ptr
                        + (M_start_offset + offs_am[:, None]) * K
                        + offs_k[None, :]
                    )
                    b_ptrs = (
                        b_desc_ptr
                        + (N_start_offset + offs_bn[:, None]) * K
                        + offs_k[None, :]
                    )
                    for k_offset in range(0, K, BLOCK_SIZE_K):
                        updated_k_offset = k_offset + offs_k
                        updated_k_offset_mask = updated_k_offset[None, :] < K  # type: ignore[16]
                        a = tl.load(
                            a_ptrs,
                            mask=((offs_am[:, None] < m_size) & updated_k_offset_mask),
                            other=0.0,
                        )
                        b = tl.load(
                            b_ptrs,
                            mask=((offs_bn[:, None] < n_size) & updated_k_offset_mask),
                            other=0.0,
                        )
                        accumulator += tl.dot(a, b.T)
                        a_ptrs += BLOCK_SIZE_K
                        b_ptrs += BLOCK_SIZE_K

                if HAS_BIAS:
                    offs_bn = tile_n_idx * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
                    bias_ptrs = bias_ptr + g.to(tl.int64) * N + offs_bn
                    bias = tl.load(bias_ptrs, mask=(offs_bn < n_size), other=0.0).to(
                        accumulator.dtype
                    )
                    accumulator = accumulator + bias[None, :]

                if HAS_TOKEN_WEIGHTS:
                    offs_am = tile_m_idx * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
                    tw_ptrs = token_weights_ptr + M_start_offset + offs_am
                    tw = tl.load(tw_ptrs, mask=(offs_am < m_size), other=1.0).to(
                        accumulator.dtype
                    )
                    accumulator = accumulator * tw[:, None]

                if USE_TMA_STORE:
                    m_offset = (tile_m_idx * BLOCK_SIZE_M).to(tl.int32)
                    n_offset = (tile_n_idx * BLOCK_SIZE_N).to(tl.int32)
                    # pyre-ignore
                    c_desc_ptr.store(
                        [m_offset, n_offset], accumulator.to(c_ptr.dtype.element_ty)
                    )
                elif FUSE_SCATTER_ADD:
                    offs_am = tile_m_idx * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
                    mask = offs_am < m_size
                    m_offsets = tl.load(
                        scatter_add_indices + M_start_offset + offs_am,
                        mask=mask,
                        cache_modifier=".ca",
                    )
                    offs_bn = tile_n_idx * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
                    c = accumulator.to(c_ptr.dtype.element_ty)
                    tl.atomic_add(
                        c_ptr + m_offsets[:, None] * N + offs_bn[None, :],
                        c,
                        mask=mask[:, None] and offs_bn[None, :] < n_size,
                        sem="relaxed",
                    )
                else:
                    offs_am = tile_m_idx * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
                    offs_bn = tile_n_idx * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
                    c = accumulator.to(c_ptr.dtype.element_ty)
                    tl.store(
                        c_ptr
                        + (M_start_offset + offs_am[:, None]) * N
                        + offs_bn[None, :],
                        c,
                        mask=offs_am[:, None] < m_size and offs_bn[None, :] < n_size,
                    )
                tidx += NUM_SMS

            iterated_tiles += num_tiles


# TODO(shikaili): Too much code duplication. Need to refactor.
@triton.autotune(
    configs=_NV_WS_CONFIGS,
    key=["G", "M_BUCKET", "N", "K"],
    prune_configs_by={"early_config_prune": early_config_prune_ws},
    restore_value=["c_ptr"],  # restore for scatter_add fusion
)
@triton.jit
def _mslk_grouped_gemm_ws(
    a_desc_ptr,
    b_desc_ptr,
    c_ptr,
    scatter_add_indices,
    m_sizes,
    bias_ptr,
    token_weights_ptr,
    # problem sizes
    G: tl.constexpr,
    M_BUCKET: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    NUM_SMS: tl.constexpr,
    FUSE_SCATTER_ADD: tl.constexpr,
    USE_TMA_LOAD: tl.constexpr,
    USE_FAST_ACCUM: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    HAS_TOKEN_WEIGHTS: tl.constexpr,
    # tile sizes
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    NUM_CONSUMER_GROUPS: tl.constexpr,
    USE_TMA_STORE: tl.constexpr,
) -> None:
    tl.static_assert(USE_TMA_LOAD, "Always use TMA load with warp specialziation!")
    tl.static_assert(
        not (FUSE_SCATTER_ADD and USE_TMA_STORE),
        "Cannot fuse scatter add with TMA store!",
    )

    tidx = tl.program_id(0)

    M_end_offset = 0
    M_end_offset = M_end_offset.to(tl.int64)  # pyre-ignore
    iterated_tiles = 0
    for g in tl.range(G):
        # Move across groups
        m_size = tl.load(m_sizes + g, cache_modifier=".ca")

        if m_size > 0:
            M_start_offset = M_end_offset
            M_end_offset = M_start_offset + m_size
            N_start_offset = g.to(tl.int64) * N

            num_m_tiles = tl.cdiv(m_size, BLOCK_SIZE_M)
            tl.static_assert(N % BLOCK_SIZE_N == 0, f"{N=} {BLOCK_SIZE_N=}")
            NUM_N_TILES: tl.constexpr = N // BLOCK_SIZE_N
            num_tiles = num_m_tiles * NUM_N_TILES

            if USE_TMA_STORE:
                c_desc_ptr = tl.make_tensor_descriptor(
                    c_ptr + M_start_offset * N,
                    # pyrefly: ignore [bad-argument-type]
                    shape=[m_size, N],
                    # pyre-ignore
                    strides=[N, 1],
                    block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_N],
                )

            # Move across tiles
            next_iterated_tiles = iterated_tiles + num_tiles
            if (tidx >= iterated_tiles) and (tidx < next_iterated_tiles):
                for i in range(tidx, next_iterated_tiles, NUM_SMS):
                    gidx = i - iterated_tiles
                    # Split M first and N second.
                    tile_m_idx = gidx % num_m_tiles
                    tile_n_idx = gidx // num_m_tiles

                    accumulator = tl.zeros(
                        (BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32
                    )
                    tl.static_assert(K % BLOCK_SIZE_K == 0)
                    m_offset = (M_start_offset + tile_m_idx * BLOCK_SIZE_M).to(tl.int32)
                    n_offset = (N_start_offset + tile_n_idx * BLOCK_SIZE_N).to(tl.int32)
                    for k_offset in range(0, K, BLOCK_SIZE_K):
                        a = a_desc_ptr.load([m_offset, k_offset])
                        b = b_desc_ptr.load([n_offset, k_offset])
                        if USE_FAST_ACCUM:
                            accumulator = tl.dot(a, b.T, accumulator)
                        else:
                            accumulator += tl.dot(a, b.T)

                    if HAS_BIAS:
                        offs_bn = tile_n_idx * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
                        bias_ptrs = bias_ptr + g.to(tl.int64) * N + offs_bn
                        bias = tl.load(bias_ptrs).to(accumulator.dtype)
                        accumulator = accumulator + bias[None, :]

                    if HAS_TOKEN_WEIGHTS:
                        offs_am = tile_m_idx * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
                        tw_ptrs = token_weights_ptr + M_start_offset + offs_am
                        tw = tl.load(tw_ptrs, mask=(offs_am < m_size), other=1.0).to(
                            accumulator.dtype
                        )
                        accumulator = accumulator * tw[:, None]

                    if USE_TMA_STORE:
                        m_offset = (tile_m_idx * BLOCK_SIZE_M).to(tl.int32)
                        n_offset = (tile_n_idx * BLOCK_SIZE_N).to(tl.int32)
                        # pyre-ignore
                        c_desc_ptr.store(
                            [m_offset, n_offset],
                            accumulator.to(c_ptr.dtype.element_ty),
                        )
                    elif FUSE_SCATTER_ADD:
                        offs_am = tile_m_idx * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
                        mask = offs_am < m_size
                        m_offsets = tl.load(
                            scatter_add_indices + M_start_offset + offs_am,
                            mask=mask,
                            cache_modifier=".ca",
                        )
                        offs_bn = tile_n_idx * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
                        c = accumulator.to(c_ptr.dtype.element_ty)
                        tl.atomic_add(
                            c_ptr + m_offsets[:, None] * N + offs_bn[None, :],
                            c,
                            mask=mask[:, None],
                            sem="relaxed",
                        )
                    else:
                        offs_am = tile_m_idx * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
                        offs_bn = tile_n_idx * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
                        c = accumulator.to(c_ptr.dtype.element_ty)
                        tl.store(
                            c_ptr
                            + (M_start_offset + offs_am[:, None]) * N
                            + offs_bn[None, :],
                            c,
                            mask=offs_am[:, None] < m_size,
                            cache_modifier=".cs",
                        )
                    tidx += NUM_SMS

            iterated_tiles += num_tiles


TT_FP8_DTYPE = tl.float8e4b8 if torch.version.hip else tl.float8e4nv


# TODO(shikaili): clean up redundant 'b_scale_desc_ptr' argument.
@triton.autotune(
    configs=_AMD_CONFIGS if torch.version.hip else _NV_CONFIGS,
    key=["G", "M_BUCKET", "N", "K"],
    prune_configs_by={
        "early_config_prune": functools.partial(
            early_config_prune, dtype=TT_FP8_DTYPE, dtsize=1
        )
    },
    restore_value=["c_ptr"],  # restore for scatter_add fusion
)
@triton.jit
def _mslk_grouped_gemm_fp8_rowwise(
    a_desc_ptr,
    a_scale_ptr,
    b_desc_ptr,
    b_scale_ptr,
    b_scale_desc_ptr,
    c_ptr,
    scatter_add_indices,
    m_sizes,
    # problem sizes
    G: tl.constexpr,
    M_BUCKET,
    N: tl.constexpr,
    K: tl.constexpr,
    NUM_SMS: tl.constexpr,
    FUSE_SCATTER_ADD: tl.constexpr,
    USE_TMA_LOAD: tl.constexpr,
    USE_TMA_STORE: tl.constexpr,
    USE_FAST_ACCUM: tl.constexpr,
    # tile sizes
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    NUM_CONSUMER_GROUPS: tl.constexpr,
) -> None:
    tl.static_assert(
        not (FUSE_SCATTER_ADD and USE_TMA_STORE),
        "Cannot fuse scatter add with TMA store!",
    )

    tidx = tl.program_id(0)

    M_end_offset = 0
    M_end_offset = M_end_offset.to(tl.int64)  # pyre-ignore
    iterated_tiles = 0
    for g in tl.range(G):
        # Move across groups
        m_size = tl.load(m_sizes + g)

        if m_size > 0:
            M_start_offset = M_end_offset
            M_end_offset = M_start_offset + m_size
            N_start_offset = g.to(tl.int64) * N
            n_size = N

            num_m_tiles = tl.cdiv(m_size, BLOCK_SIZE_M)
            num_n_tiles = tl.cdiv(n_size, BLOCK_SIZE_N)
            num_tiles = num_m_tiles * num_n_tiles

            if USE_TMA_STORE:
                c_desc_ptr = tl.make_tensor_descriptor(
                    c_ptr + M_start_offset * N,
                    # pyrefly: ignore [bad-argument-type]
                    shape=[m_size, n_size],
                    # pyre-ignore
                    strides=[n_size, 1],
                    block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_N],
                )

            # Move across tiles
            while tidx >= iterated_tiles and tidx < iterated_tiles + num_tiles:
                gidx = tidx - iterated_tiles
                # Split M first and N second.
                tile_m_idx = gidx % num_m_tiles
                tile_n_idx = gidx // num_m_tiles

                accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
                tl.static_assert(K % BLOCK_SIZE_K == 0)
                if USE_TMA_LOAD:
                    m_offset = (M_start_offset + tile_m_idx * BLOCK_SIZE_M).to(tl.int32)
                    n_offset = (N_start_offset + tile_n_idx * BLOCK_SIZE_N).to(tl.int32)
                    for k_offset in range(0, K, BLOCK_SIZE_K):
                        a = a_desc_ptr.load([m_offset, k_offset])
                        b = b_desc_ptr.load([n_offset, k_offset])
                        if USE_FAST_ACCUM:
                            accumulator = tl.dot(a, b.T, accumulator)
                        else:
                            accumulator += tl.dot(a, b.T)
                else:
                    offs_am = tile_m_idx * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
                    offs_bn = tile_n_idx * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
                    offs_k = tl.arange(0, BLOCK_SIZE_K)
                    a_ptrs = (
                        a_desc_ptr
                        + (M_start_offset + offs_am[:, None]) * K
                        + offs_k[None, :]
                    )
                    b_ptrs = (
                        b_desc_ptr
                        + (N_start_offset + offs_bn[:, None]) * K
                        + offs_k[None, :]
                    )
                    for _ in range(0, K, BLOCK_SIZE_K):
                        a = tl.load(a_ptrs, mask=offs_am[:, None] < m_size)
                        b = tl.load(b_ptrs, mask=offs_bn[:, None] < n_size)
                        accumulator += tl.dot(a, b.T)
                        a_ptrs += BLOCK_SIZE_K
                        b_ptrs += BLOCK_SIZE_K

                offs_am = tile_m_idx * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
                offs_bn = tile_n_idx * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
                a_scale = tl.load(
                    a_scale_ptr + M_start_offset + offs_am[:, None],
                    mask=offs_am[:, None] < m_size,
                )
                b_scale = tl.load(
                    b_scale_ptr + N_start_offset + offs_bn[None, :],
                    mask=offs_bn[None, :] < n_size,
                )
                c = accumulator.to(tl.float32) * a_scale * b_scale

                if USE_TMA_STORE:
                    m_offset = (tile_m_idx * BLOCK_SIZE_M).to(tl.int32)
                    n_offset = (tile_n_idx * BLOCK_SIZE_N).to(tl.int32)
                    # pyre-ignore
                    c_desc_ptr.store([m_offset, n_offset], c.to(c_ptr.dtype.element_ty))
                elif FUSE_SCATTER_ADD:
                    offs_am = tile_m_idx * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
                    mask = offs_am < m_size
                    m_offsets = tl.load(
                        scatter_add_indices + M_start_offset + offs_am,
                        mask=mask,
                        cache_modifier=".ca",
                    )
                    offs_bn = tile_n_idx * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
                    tl.atomic_add(
                        c_ptr + m_offsets[:, None] * N + offs_bn[None, :],
                        c.to(c_ptr.dtype.element_ty),
                        mask=mask[:, None] and offs_bn[None, :] < n_size,
                        sem="relaxed",
                    )
                else:
                    tl.store(
                        c_ptr
                        + (M_start_offset + offs_am[:, None]) * N
                        + offs_bn[None, :],
                        c,
                        mask=offs_am[:, None] < m_size and offs_bn[None, :] < n_size,
                    )
                tidx += NUM_SMS

            iterated_tiles += num_tiles


# TODO(shikaili): Too much code duplication. Need to refactor.
@triton.autotune(
    configs=_NV_WS_CONFIGS,
    key=["G", "M_BUCKET", "N", "K"],
    prune_configs_by={
        "early_config_prune": functools.partial(
            early_config_prune_ws, dtype=TT_FP8_DTYPE, dtsize=1
        )
    },
    restore_value=["c_ptr"],  # restore for scatter_add fusion
)
@triton.jit
def _mslk_grouped_gemm_fp8_rowwise_ws(
    a_desc_ptr,
    a_scale_ptr,
    b_desc_ptr,
    b_scale_ptr,
    c_ptr,
    scatter_add_indices,
    m_sizes,
    # problem sizes
    G: tl.constexpr,
    M_BUCKET: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    NUM_SMS: tl.constexpr,
    FUSE_SCATTER_ADD: tl.constexpr,
    USE_TMA_LOAD: tl.constexpr,
    USE_FAST_ACCUM: tl.constexpr,
    # tile sizes
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    NUM_CONSUMER_GROUPS: tl.constexpr,
    USE_TMA_STORE: tl.constexpr,
) -> None:
    tl.static_assert(USE_TMA_LOAD, "Always use TMA load with warp specialziation!")
    tl.static_assert(
        not (FUSE_SCATTER_ADD and USE_TMA_STORE),
        "Cannot fuse scatter add with TMA store!",
    )

    tidx = tl.program_id(0)

    M_end_offset = 0
    M_end_offset = M_end_offset.to(tl.int64)  # pyre-ignore
    iterated_tiles = 0
    for g in tl.range(G):
        # Move across groups
        m_size = tl.load(m_sizes + g, cache_modifier=".ca")

        if m_size > 0:
            M_start_offset = M_end_offset
            M_end_offset = M_start_offset + m_size
            N_start_offset = g.to(tl.int64) * N

            num_m_tiles = tl.cdiv(m_size, BLOCK_SIZE_M)
            tl.static_assert(N % BLOCK_SIZE_N == 0)
            NUM_N_TILES: tl.constexpr = N // BLOCK_SIZE_N
            num_tiles = num_m_tiles * NUM_N_TILES

            if USE_TMA_STORE:
                c_desc_ptr = tl.make_tensor_descriptor(
                    c_ptr + M_start_offset * N,
                    # pyrefly: ignore [bad-argument-type]
                    shape=[m_size, N],
                    # pyre-ignore
                    strides=[N, 1],
                    block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_N],
                )

            # Move across tiles
            next_iterated_tiles = iterated_tiles + num_tiles
            if (tidx >= iterated_tiles) and (tidx < next_iterated_tiles):
                for i in range(tidx, next_iterated_tiles, NUM_SMS):
                    gidx = i - iterated_tiles
                    # Split M first and N second.
                    tile_m_idx = gidx % num_m_tiles
                    tile_n_idx = gidx // num_m_tiles

                    accumulator = tl.zeros(
                        (BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32
                    )
                    tl.static_assert(K % BLOCK_SIZE_K == 0)

                    m_offset = (M_start_offset + tile_m_idx * BLOCK_SIZE_M).to(tl.int32)
                    n_offset = (N_start_offset + tile_n_idx * BLOCK_SIZE_N).to(tl.int32)
                    for k_offset in range(0, K, BLOCK_SIZE_K):
                        a = a_desc_ptr.load([m_offset, k_offset])
                        b = b_desc_ptr.load([n_offset, k_offset])
                        if USE_FAST_ACCUM:
                            accumulator = tl.dot(a, b.T, accumulator)
                        else:
                            accumulator += tl.dot(a, b.T)

                    offs_am = tile_m_idx * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
                    offs_bn = tile_n_idx * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
                    a_scale = tl.load(
                        a_scale_ptr + M_start_offset + offs_am[:, None],
                        mask=offs_am[:, None] < m_size,
                        cache_modifier=".ca",
                    )
                    b_scale = tl.load(
                        b_scale_ptr + N_start_offset + offs_bn[None, :],
                        cache_modifier=".ca",
                    )
                    c = accumulator.to(tl.float32) * a_scale * b_scale

                    if USE_TMA_STORE:
                        m_offset = (tile_m_idx * BLOCK_SIZE_M).to(tl.int32)
                        n_offset = (tile_n_idx * BLOCK_SIZE_N).to(tl.int32)
                        # pyre-ignore
                        c_desc_ptr.store(
                            [m_offset, n_offset], c.to(c_ptr.dtype.element_ty)
                        )
                    elif FUSE_SCATTER_ADD:
                        offs_am = tile_m_idx * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
                        mask = offs_am < m_size
                        m_offsets = tl.load(
                            scatter_add_indices + M_start_offset + offs_am,
                            mask=mask,
                            cache_modifier=".ca",
                        )
                        offs_bn = tile_n_idx * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
                        tl.atomic_add(
                            c_ptr + m_offsets[:, None] * N + offs_bn[None, :],
                            c,
                            mask=mask[:, None],
                            sem="relaxed",
                        )
                    else:
                        offs_am = tile_m_idx * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
                        offs_bn = tile_n_idx * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
                        tl.store(
                            c_ptr
                            + (M_start_offset + offs_am[:, None]) * N
                            + offs_bn[None, :],
                            c,
                            mask=offs_am[:, None] < m_size,
                            cache_modifier=".cs",
                        )
                    tidx += NUM_SMS

            iterated_tiles += num_tiles


warnings.simplefilter("once")


def _grouped_gemm(
    *,
    x: torch.Tensor,
    w: torch.Tensor,
    m_sizes: torch.Tensor,
    x_scale: Optional[torch.Tensor],
    w_scale: Optional[torch.Tensor],
    bias: Optional[torch.Tensor],
    token_weights: Optional[torch.Tensor],
    use_fast_accum: bool,
    use_warp_specialization: bool,
    output_tensor: Optional[torch.Tensor],
    scatter_add_indices: Optional[torch.Tensor],
) -> torch.Tensor:
    USE_TMA_LOAD = not torch.version.hip and TMA_AVAILABLE
    USE_TMA_STORE = False

    # TODO(shikaili): Check the readniess of WS on ROCm side in Meta's Triton.
    if use_warp_specialization and torch.version.hip:
        warnings.warn(
            "Warp specialization is disabled as it is not supported on ROCm.",
            stacklevel=2,
        )
        use_warp_specialization = False

    if use_warp_specialization:
        assert TMA_AVAILABLE, "TMA is not available"
        USE_TMA_STORE = True  # Tuning decision

    G = m_sizes.shape[0]

    assert x.is_contiguous()
    assert w.is_contiguous()
    assert m_sizes.is_contiguous()

    M, K = x.shape
    N = w.shape[0] // G
    assert K == w.shape[1]

    if K % 8 != 0 or N % 8 != 0:
        use_warp_specialization = False
        USE_TMA_LOAD = False
        USE_TMA_STORE = False
        warnings.warn(
            f"TMA load and warp specialization are disabled since K or N is not a multiple of 8: {K=}, {N=}.",
            stacklevel=2,
        )
        assert x_scale is None, (
            f"Quantisation is not supported yet when K or N is not a multiple of 8: {K=}, {N=}."
        )

        assert output_tensor is None, (
            f"Fused scatter add has large rounding error when K or N is not a multiple of 8: {K=}, {N=}."
        )

    HAS_BIAS = bias is not None
    if HAS_BIAS:
        assert bias is not None  # for type checker
        assert bias.is_contiguous(), "Bias must be contiguous"
        assert len(bias.shape) == 2, f"Bias must be 2D, got shape {bias.shape}"
        assert bias.shape[0] == G, f"Bias dim 0 must match G={G}, got {bias.shape[0]}"
        assert bias.shape[1] == N, f"Bias dim 1 must match N={N}, got {bias.shape[1]}"

    HAS_TOKEN_WEIGHTS = token_weights is not None
    if HAS_TOKEN_WEIGHTS:
        assert token_weights is not None  # for type checker
        assert token_weights.is_contiguous(), "token_weights must be contiguous"
        assert len(token_weights.shape) == 1, (
            f"token_weights must be 1D, got shape {token_weights.shape}"
        )
        assert token_weights.shape[0] == M, (
            f"token_weights dim 0 must match M={M}, got {token_weights.shape[0]}"
        )

    if output_tensor is None:
        FUSE_SCATTER_ADD = False
        assert scatter_add_indices is None
        y = torch.empty((M, N), device=x.device, dtype=torch.bfloat16)
    else:
        FUSE_SCATTER_ADD = True
        assert scatter_add_indices is not None
        assert scatter_add_indices.is_contiguous()
        assert scatter_add_indices.shape == (M,)
        y = output_tensor
    if M == 0 or N == 0:
        return y

    NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count

    # A dummy block value that will be overwritten in the pre_hook when we have the real block size
    dummy_block = [1, 1]

    if USE_TMA_LOAD:
        # pyre-ignore[6]: In call `TensorDescriptor.__init__`, for 2nd positional
        # argument, expected `List[int]` but got `Size`
        desc_x = TensorDescriptor(x, x.shape, x.stride(), dummy_block)
        # pyre-ignore[6]: In call `TensorDescriptor.__init__`, for 2nd positional
        # argument, expected `List[int]` but got `Size`
        desc_w = TensorDescriptor(w, w.shape, w.stride(), dummy_block)
    else:
        desc_x = x
        desc_w = w

    if USE_TMA_STORE:

        def alloc_fn(size: int, alignment: int, stream: Optional[int]):
            return torch.empty(size, device="cuda", dtype=torch.int8)

        triton.set_allocator(alloc_fn)

    def grid(META):
        return (NUM_SMS,)

    M_BUCKET_CAP = 16384
    M_BUCKET = min(triton.next_power_of_2(M), M_BUCKET_CAP)
    if x_scale is not None and w_scale is not None:
        assert x_scale.is_contiguous()
        assert w_scale.is_contiguous()
        fn = (
            _mslk_grouped_gemm_fp8_rowwise_ws
            if use_warp_specialization
            else _mslk_grouped_gemm_fp8_rowwise
        )
        if use_warp_specialization:
            args = (
                desc_x,
                x_scale,
                desc_w,
                w_scale,
                y,
                scatter_add_indices,
                m_sizes,
                G,
                M_BUCKET,
                N,
                K,
                NUM_SMS,
                FUSE_SCATTER_ADD,
                USE_TMA_LOAD,
                use_fast_accum,
            )
        else:
            args = (
                desc_x,
                x_scale,
                desc_w,
                w_scale,
                w_scale,  # b_scale_desc_ptr (unused, just passed for API compatibility)
                y,
                scatter_add_indices,
                m_sizes,
                G,
                M_BUCKET,
                N,
                K,
                NUM_SMS,
                FUSE_SCATTER_ADD,
                USE_TMA_LOAD,
                USE_TMA_STORE,
                use_fast_accum,
            )
        fn[grid](*args)
    else:
        assert x_scale is None
        assert w_scale is None
        fn = _mslk_grouped_gemm_ws if use_warp_specialization else _mslk_grouped_gemm
        args = (
            desc_x,
            desc_w,
            y,
            scatter_add_indices,
            m_sizes,
            bias if HAS_BIAS else None,
            token_weights if HAS_TOKEN_WEIGHTS else None,
            G,
            M_BUCKET,
            N,
            K,
            NUM_SMS,
            FUSE_SCATTER_ADD,
            USE_TMA_LOAD,
        )
        if use_warp_specialization:
            args += (use_fast_accum, HAS_BIAS, HAS_TOKEN_WEIGHTS)
        else:
            args += (USE_TMA_STORE, use_fast_accum, HAS_BIAS, HAS_TOKEN_WEIGHTS)
        fn[grid](*args)

    return y


def grouped_gemm(
    x: torch.Tensor,
    w: torch.Tensor,
    m_sizes: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    token_weights: Optional[torch.Tensor] = None,
    use_fast_accum: bool = True,
    *,
    _use_warp_specialization: bool = True,
    _output_tensor: Optional[torch.Tensor] = None,
    _scatter_add_indices: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Grouped GEMM with optional bias addition and per-token weight scaling.

    Performs: output = (x @ w.T + bias) * token_weights
    where operations are grouped by experts.

    Args:
        x: Input tensor [M, K] where M is total tokens across all experts
        w: Weight tensor [G * N, K] where G is number of experts
        m_sizes: Tensor [G] indicating number of tokens per expert
        bias: Optional bias tensor [G, N], one bias vector per expert
        token_weights: Optional per-token scaling weights [M] (e.g., router weights)
        use_fast_accum: Enable fast accumulation for better performance
        _use_warp_specialization: Flag for warp specialization
        _output_tensor: Optional pre-allocated output tensor for scatter-add
        _scatter_add_indices: Optional indices for scatter-add operation

    Returns:
        Output tensor [M, N]
    """
    return _grouped_gemm(
        x=x,
        w=w,
        m_sizes=m_sizes,
        x_scale=None,
        w_scale=None,
        bias=bias,
        token_weights=token_weights,
        use_fast_accum=use_fast_accum,
        use_warp_specialization=_use_warp_specialization,
        output_tensor=_output_tensor,
        scatter_add_indices=_scatter_add_indices,
    )


def grouped_gemm_fp8_rowwise(
    x: torch.Tensor,
    w: torch.Tensor,
    m_sizes: torch.Tensor,
    x_scale: torch.Tensor,
    w_scale: torch.Tensor,
    use_fast_accum: bool = True,
    *,
    _use_warp_specialization: bool = True,
    _output_tensor: Optional[torch.Tensor] = None,
    _scatter_add_indices: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    return _grouped_gemm(
        x=x,
        w=w,
        m_sizes=m_sizes,
        x_scale=x_scale,
        w_scale=w_scale,
        bias=None,
        token_weights=None,
        use_fast_accum=use_fast_accum,
        use_warp_specialization=_use_warp_specialization,
        output_tensor=_output_tensor,
        scatter_add_indices=_scatter_add_indices,
    )


# ---------------------------------------------------------------------------
# Triton port of CUTLASS bf16bf16bf16_grouped_grad / _wgrad ops.
#
# The CUTLASS kernels in csrc/gemm/cutlass/bf16bf16bf16_grouped_{grad,wgrad}.cu
# are CUDA-only (Sm90/Sm100 TMA + ptr-array grouped GEMM). The Triton kernels
# below reproduce the same numerics with the same external schema, and are
# registered as the ROCm dispatch of:
#   torch.ops.mslk.bf16bf16bf16_grouped_grad
#   torch.ops.mslk.bf16bf16bf16_grouped_wgrad
# (see _register_rocm_grad_ops below).
#
# Conventions (matching the .cu / test_grouped_gemm_{dgrad,wgrad}):
#   dgrad: X [total_M, K]            W [G, N, K] with W.stride(-2) == 1
#          out [total_M, N]          per group: out[g] = X[g] @ W[g].T
#          (caller typically passes w.permute(0, 2, 1) of a contiguous [G,K,N])
#   wgrad: X [total_M, N]            W [total_M, K] (here W is really `dy`)
#          out [G, N, K]             per group: out[g] = X[g].T @ W[g]
#          OUTPUT_ACCUM: pre-allocated fp32 out, accumulator is added in-place
# ---------------------------------------------------------------------------


_AMD_GRAD_CONFIGS = [
    triton.Config(
        {
            "BLOCK_SIZE_M": block_size_m,
            "BLOCK_SIZE_N": block_size_n,
            "BLOCK_SIZE_K": block_size_k,
            "waves_per_eu": waves_per_cu,
            "matrix_instr_nonkdim": 16,
        },
        num_stages=num_stages,
        num_warps=num_warps,
    )
    for block_size_m in [32, 64, 128]
    for block_size_n in [64, 128, 256]
    for block_size_k in [64, 128]
    for num_stages in [1, 2]
    for num_warps, waves_per_cu in [(4, 1), (8, 2)]
]


_AMD_WGRAD_CONFIGS = [
    triton.Config(
        {
            "BLOCK_SIZE_M": block_size_m,
            "BLOCK_SIZE_N": block_size_n,
            "BLOCK_SIZE_K": block_size_k,
            "waves_per_eu": waves_per_cu,
            "matrix_instr_nonkdim": 16,
        },
        num_stages=num_stages,
        num_warps=num_warps,
    )
    for block_size_m in [32, 64]
    for block_size_n in [32, 64, 128]
    for block_size_k in [32, 64, 128]
    for num_stages in [1, 2]
    for num_warps, waves_per_cu in [(4, 1), (8, 2)]
]


@triton.autotune(
    configs=_AMD_GRAD_CONFIGS,
    key=["G", "M_BUCKET", "N", "K"],
)
@triton.jit
def _mslk_grouped_gemm_dgrad(
    x_ptr,  # X [total_M, K] row-major
    w_ptr,  # W [G, N, K] with strides [N*K, 1, N] (i.e. K-major per group)
    c_ptr,  # output [total_M, N] row-major
    m_sizes,
    G: tl.constexpr,
    M_BUCKET,
    N: tl.constexpr,
    K: tl.constexpr,
    NUM_SMS: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
) -> None:
    tidx = tl.program_id(0)

    M_end_offset = 0
    M_end_offset = M_end_offset.to(tl.int64)  # pyre-ignore
    # Keep tile counters in int64. M_sizes is int64 (required by the op), so
    # num_tiles is int64; iterated_tiles must match to stay type-stable across
    # the `if m_size > 0` block (Triton forbids dtype changes across blocks).
    iterated_tiles = 0
    iterated_tiles = iterated_tiles.to(tl.int64)  # pyre-ignore
    for g in tl.range(G):
        m_size = tl.load(m_sizes + g).to(tl.int64)
        if m_size > 0:
            M_start_offset = M_end_offset
            M_end_offset = M_start_offset + m_size

            num_m_tiles = tl.cdiv(m_size, BLOCK_SIZE_M)
            num_n_tiles = tl.cdiv(N, BLOCK_SIZE_N)
            num_tiles = num_m_tiles * num_n_tiles

            while tidx >= iterated_tiles and tidx < iterated_tiles + num_tiles:
                gidx = tidx - iterated_tiles
                tile_m_idx = gidx % num_m_tiles
                tile_n_idx = gidx // num_m_tiles

                accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

                offs_am = tile_m_idx * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
                offs_bn = tile_n_idx * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
                offs_k = tl.arange(0, BLOCK_SIZE_K)

                # A: X[M_start + offs_am, offs_k] (row-major in K).
                a_ptrs = (
                    x_ptr + (M_start_offset + offs_am[:, None]) * K + offs_k[None, :]
                )
                # B: per-group view of W as a [K, N] row-major matrix.
                # W[g, n, k] lives at addr g*N*K + n + k*N; equivalently
                # B[k, n] = W[g, n, k] at addr g*N*K + offs_k*N + offs_bn.
                b_ptrs = (
                    w_ptr
                    + g.to(tl.int64) * N * K
                    + offs_k[:, None] * N
                    + offs_bn[None, :]
                )

                for k_offset in range(0, K, BLOCK_SIZE_K):
                    k_remaining = k_offset + offs_k
                    k_mask_row = k_remaining[None, :] < K
                    k_mask_col = k_remaining[:, None] < K
                    a = tl.load(
                        a_ptrs,
                        mask=(offs_am[:, None] < m_size) & k_mask_row,
                        other=0.0,
                    )
                    b = tl.load(
                        b_ptrs,
                        mask=k_mask_col & (offs_bn[None, :] < N),
                        other=0.0,
                    )
                    accumulator += tl.dot(a, b)
                    a_ptrs += BLOCK_SIZE_K
                    b_ptrs += BLOCK_SIZE_K * N

                c = accumulator.to(c_ptr.dtype.element_ty)
                tl.store(
                    c_ptr + (M_start_offset + offs_am[:, None]) * N + offs_bn[None, :],
                    c,
                    mask=(offs_am[:, None] < m_size) & (offs_bn[None, :] < N),
                )
                tidx += NUM_SMS

            iterated_tiles += num_tiles


@triton.autotune(
    configs=_AMD_WGRAD_CONFIGS,
    key=["G", "M_BUCKET", "N", "K", "OUTPUT_ACCUM"],
    # When OUTPUT_ACCUM is set the kernel does an in-place read-modify-write
    # (output += dW). The autotuner benchmarks each config by invoking the
    # kernel many times on the *same* buffer, which would otherwise compound
    # the accumulation and corrupt the user's output. restore_value snapshots
    # the buffer before each trial and restores it (including before the final
    # real launch), so accumulation happens exactly once.
    restore_value=["c_ptr"],
)
@triton.jit
def _mslk_grouped_gemm_wgrad(
    x_ptr,  # X [total_M, N] row-major
    w_ptr,  # W [total_M, K] row-major (a.k.a. dy)
    c_ptr,  # output [G, N, K] row-major (fp32 if OUTPUT_ACCUM else bf16/fp16)
    m_sizes,
    G: tl.constexpr,
    M_BUCKET,
    N: tl.constexpr,
    K: tl.constexpr,
    NUM_SMS: tl.constexpr,
    OUTPUT_ACCUM: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
) -> None:
    tidx = tl.program_id(0)

    M_end_offset = 0
    M_end_offset = M_end_offset.to(tl.int64)  # pyre-ignore
    # Keep tile counters in int64 for type-stability across the `if m_size > 0`
    # block (M_sizes is int64; see dgrad kernel note).
    iterated_tiles = 0
    iterated_tiles = iterated_tiles.to(tl.int64)  # pyre-ignore
    for g in tl.range(G):
        m_size = tl.load(m_sizes + g).to(tl.int64)
        if m_size > 0:
            M_start_offset = M_end_offset
            M_end_offset = M_start_offset + m_size

            num_n_tiles = tl.cdiv(N, BLOCK_SIZE_N)
            num_k_tiles = tl.cdiv(K, BLOCK_SIZE_K)
            num_tiles = num_n_tiles * num_k_tiles

            while tidx >= iterated_tiles and tidx < iterated_tiles + num_tiles:
                gidx = tidx - iterated_tiles
                tile_n_idx = gidx % num_n_tiles
                tile_k_idx = gidx // num_n_tiles

                accumulator = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_K), dtype=tl.float32)

                offs_am = tl.arange(0, BLOCK_SIZE_M)
                offs_an = tile_n_idx * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
                offs_bk = tile_k_idx * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)

                # X slice for this group/tile: [BM, BN]
                a_base = (
                    x_ptr + (M_start_offset + offs_am[:, None]) * N + offs_an[None, :]
                )
                # W slice for this group/tile: [BM, BK]
                b_base = (
                    w_ptr + (M_start_offset + offs_am[:, None]) * K + offs_bk[None, :]
                )

                an_mask = offs_an[None, :] < N
                bk_mask = offs_bk[None, :] < K

                for m_offset in range(0, m_size, BLOCK_SIZE_M):
                    m_mask = (m_offset + offs_am[:, None]) < m_size
                    a = tl.load(
                        a_base + m_offset * N,
                        mask=m_mask & an_mask,
                        other=0.0,
                    )
                    b = tl.load(
                        b_base + m_offset * K,
                        mask=m_mask & bk_mask,
                        other=0.0,
                    )
                    accumulator += tl.dot(tl.trans(a), b)

                out_ptrs = (
                    c_ptr
                    + g.to(tl.int64) * N * K
                    + offs_an[:, None] * K
                    + offs_bk[None, :]
                )
                store_mask = (offs_an[:, None] < N) & (offs_bk[None, :] < K)

                if OUTPUT_ACCUM:
                    prev = tl.load(out_ptrs, mask=store_mask, other=0.0)
                    tl.store(
                        out_ptrs,
                        accumulator + prev,
                        mask=store_mask,
                    )
                else:
                    tl.store(
                        out_ptrs,
                        accumulator.to(c_ptr.dtype.element_ty),
                        mask=store_mask,
                    )

                tidx += NUM_SMS

            iterated_tiles += num_tiles


_MIN_M_PER_GROUP = 8


def _assert_m_sizes_min(m_sizes: torch.Tensor, op_name: str) -> None:
    """Ensure every non-empty group has m_size >= _MIN_M_PER_GROUP.

    Matches the host-side guard requested for the Triton ports: groups of size
    0 are allowed (and skipped inside the kernel), but any non-empty group
    smaller than _MIN_M_PER_GROUP would land on an MFMA-tile edge case.
    """
    sizes = m_sizes.tolist()
    for g, s in enumerate(sizes):
        if 0 < s < _MIN_M_PER_GROUP:
            raise ValueError(
                f"{op_name} requires every non-empty group to have "
                f"m_size >= {_MIN_M_PER_GROUP}; got m_sizes[{g}] = {s}. "
                "Pad your group on the caller side (or round up to a "
                f"multiple of {_MIN_M_PER_GROUP})."
            )


def grouped_gemm_dgrad(
    x: torch.Tensor,
    w: torch.Tensor,
    m_sizes: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    num_sms: Optional[int] = None,
) -> torch.Tensor:
    """Triton port of mslk::bf16bf16bf16_grouped_grad.

    Args:
        x:        [total_M, K], row-major, bf16/fp16
        w:        [G, N, K] with w.stride(-2) == 1, bf16/fp16 (typically
                  the caller does w.permute(0, 2, 1) on a contiguous
                  [G, K, N] tensor)
        m_sizes:  [G], int64 — per-group token counts
        out:      Optional pre-allocated output buffer (numel >= total_M * N);
                  will be returned viewed as [total_M, N].
        num_sms:  Override the launch grid (defaults to all SMs).

    Returns:
        [total_M, N] tensor, dtype = x.dtype.
    """
    assert x.dim() == 2, f"X must be 2D, got shape {x.shape}"
    assert w.dim() == 3, f"W must be 3D, got shape {w.shape}"
    assert x.is_contiguous(), "X must be contiguous (row-major)"
    assert w.stride(-2) == 1, (
        "W must be column-major in the N (-2) dim; pass "
        "w_contig.permute(0, 2, 1) of a [G, K, N] contiguous tensor"
    )
    assert m_sizes.is_contiguous() and m_sizes.dtype == torch.int64, (
        "M_sizes must be a contiguous int64 tensor of shape [G]"
    )
    assert m_sizes.device == x.device, "M_sizes must be on the same device as X"
    assert x.dtype == w.dtype, f"X and W must share dtype, got {x.dtype} vs {w.dtype}"
    assert x.dtype in (torch.bfloat16, torch.float16), (
        f"grouped_gemm_dgrad supports bf16/fp16, got {x.dtype}"
    )

    total_M, K = x.shape
    G, N, K_w = w.shape
    assert K == K_w, f"X.shape[-1] ({K}) must match W.shape[-1] ({K_w})"
    assert m_sizes.shape == (G,), f"M_sizes shape {m_sizes.shape} != ({G},)"

    if out is None:
        y = torch.empty(total_M * N, device=x.device, dtype=x.dtype)
    else:
        assert out.device == x.device and out.dtype == x.dtype
        assert out.is_contiguous()
        assert out.numel() >= total_M * N, (
            f"Pre-allocated out buffer has {out.numel()} elements, need >= {total_M * N}"
        )
        y = out

    if total_M == 0:
        return y.view(total_M, N)

    _assert_m_sizes_min(m_sizes, "grouped_gemm_dgrad")

    if num_sms is None:
        num_sms = torch.cuda.get_device_properties(x.device).multi_processor_count
    NUM_SMS = int(num_sms)

    M_BUCKET_CAP = 16384
    M_BUCKET = min(triton.next_power_of_2(total_M), M_BUCKET_CAP)

    grid = (NUM_SMS,)
    _mslk_grouped_gemm_dgrad[grid](
        x,
        w,
        y,
        m_sizes,
        G,
        M_BUCKET,
        N,
        K,
        NUM_SMS,
    )

    return y.view(total_M, N)


def grouped_gemm_wgrad(
    x: torch.Tensor,
    w: torch.Tensor,
    m_sizes: torch.Tensor,
    output: Optional[torch.Tensor] = None,
    output_accum: bool = False,
    num_sms: Optional[int] = None,
) -> torch.Tensor:
    """Triton port of mslk::bf16bf16bf16_grouped_wgrad.

    Args:
        x:            [total_M, N], row-major, bf16/fp16
        w:            [total_M, K], row-major, bf16/fp16 (typically `dy`)
        m_sizes:      [G], int64 — per-group token counts
        output:       Optional pre-allocated output [G, N, K]. Required when
                      output_accum=True (dtype must be float32 in that case).
        output_accum: If True, accumulate into the pre-allocated fp32 output.
        num_sms:      Override the launch grid (defaults to all SMs).

    Returns:
        [G, N, K] tensor. dtype is fp32 if output_accum else x.dtype.
    """
    assert x.dim() == 2 and w.dim() == 2, (
        f"X and W must be 2D, got {x.shape} and {w.shape}"
    )
    assert x.is_contiguous() and w.is_contiguous(), "X and W must be contiguous"
    assert x.shape[0] == w.shape[0], (
        f"X and W must share dim 0 (total_M); got {x.shape[0]} vs {w.shape[0]}"
    )
    assert x.dtype == w.dtype, f"X and W must share dtype, got {x.dtype} vs {w.dtype}"
    assert x.dtype in (torch.bfloat16, torch.float16), (
        f"grouped_gemm_wgrad supports bf16/fp16, got {x.dtype}"
    )
    assert m_sizes.is_contiguous() and m_sizes.dtype == torch.int64, (
        "M_sizes must be a contiguous int64 tensor of shape [G]"
    )
    assert m_sizes.device == x.device, "M_sizes must be on the same device as X"

    total_M, N = x.shape
    _, K = w.shape
    G = m_sizes.shape[0]

    if output_accum:
        assert output is not None, (
            "Must provide a pre-allocated output tensor when output_accum=True"
        )
        assert output.dtype == torch.float32, (
            "Output tensor must be Float32 when output_accum=True"
        )
        y = output
    elif output is not None:
        assert output.dtype in (torch.bfloat16, torch.float16), (
            "Output tensor must be BFloat16 or Float16 when output_accum=False"
        )
        y = output
    else:
        y = torch.zeros((G, N, K), device=x.device, dtype=x.dtype)

    assert y.is_contiguous() and y.shape == (G, N, K), (
        f"Output must be contiguous [G, N, K]={(G, N, K)}, got shape {y.shape}"
    )

    if total_M == 0:
        return y

    _assert_m_sizes_min(m_sizes, "grouped_gemm_wgrad")

    if num_sms is None:
        num_sms = torch.cuda.get_device_properties(x.device).multi_processor_count
    NUM_SMS = int(num_sms)

    M_BUCKET_CAP = 16384
    M_BUCKET = min(triton.next_power_of_2(total_M), M_BUCKET_CAP)

    grid = (NUM_SMS,)
    _mslk_grouped_gemm_wgrad[grid](
        x,
        w,
        y,
        m_sizes,
        G,
        M_BUCKET,
        N,
        K,
        NUM_SMS,
        output_accum,
    )

    return y


# ---------------------------------------------------------------------------
# torch.library.impl registration — wire torch.ops.mslk.bf16bf16bf16_grouped_*
# to the Triton functions on ROCm. The C++ schemas are declared under
# #ifdef USE_ROCM in csrc/gemm/gemm_ops.cpp; this module is imported from
# mslk/gemm/__init__.py so registration happens after the .so is loaded.
# ---------------------------------------------------------------------------


def _register_rocm_grad_ops() -> None:
    import torch.library  # noqa: F401

    # PyTorch HIP builds surface the CUDA dispatch key for GPU kernels; keep
    # this on "CUDA" until PyTorch exposes a stable HIP/PrivateUse key.
    @torch.library.impl("mslk::bf16bf16bf16_grouped_grad", "CUDA")
    def _bf16_grouped_grad_impl(
        X: torch.Tensor,
        W: torch.Tensor,
        M_sizes: torch.Tensor,
        out: Optional[torch.Tensor] = None,
        num_sms: Optional[int] = None,
    ) -> torch.Tensor:
        return grouped_gemm_dgrad(X, W, M_sizes, out=out, num_sms=num_sms)

    @torch.library.impl("mslk::bf16bf16bf16_grouped_wgrad", "CUDA")
    def _bf16_grouped_wgrad_impl(
        X: torch.Tensor,
        W: torch.Tensor,
        M_sizes: torch.Tensor,
        output: Optional[torch.Tensor] = None,
        output_accum: bool = False,
        num_sms: Optional[int] = None,
    ) -> torch.Tensor:
        return grouped_gemm_wgrad(
            X, W, M_sizes, output=output, output_accum=output_accum, num_sms=num_sms
        )


# Register immediately on ROCm — the C++ library loads the schemas during
# import of mslk.gemm, so by the time user code imports from here the ops
# exist. If registration fails (e.g. schemas not yet present in restricted
# unit-test environments), defer to a later caller.
if torch.version.hip is not None:
    try:
        _register_rocm_grad_ops()
    except Exception as exc:
        warnings.warn(
            f"Deferring ROCm bf16bf16bf16_grouped_{{grad,wgrad}} registration: {exc}",
            stacklevel=2,
        )
