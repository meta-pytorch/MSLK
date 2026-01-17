# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe
import logging
import os
from typing import Dict, List, Optional, Tuple, Union

import torch
import triton  # @manual
import triton.language as tl  # @manual
from mslk.gemm.triton.matmul_perf_model import early_config_prune, estimate_matmul_time
from mslk.gemm.triton.utils import map_dtype_to_triton, TmaAutoTuneHelper
from mslk.utils.triton.fp8_utils import get_fp8_constants, reinterpret_fp8_type
from packaging import version
from torch._tensor import Tensor
from triton import Config  # @manual
from triton.runtime.jit import TensorWrapper  # @manual

logger: logging.Logger = logging.getLogger(__name__)

running_on_github: bool = os.getenv("GITHUB_ENV") is not None

try:
    # pyre-ignore[21]
    from triton.fb.compat import disable_bufferops  # @manual
except ModuleNotFoundError:
    # Ensure we can call disable_bufferops if compat is not included (e.g. opensource)
    # TODO(njriasan): Remove when we integrate triton.fb.compat into every Triton
    # version.
    from contextlib import contextmanager

    @contextmanager
    def disable_bufferops(_unused: bool):
        yield None


def init_to_zero(name):
    return lambda nargs: nargs[name].zero_()


def get_configs_io_bound() -> List[Config]:
    """
    Returns a list of configs for matmul that are IO bound.

    Returns:
        List[Config]: list of configs.
    """
    configs = []
    for num_stages in [2, 3, 4, 5, 6]:
        for block_m in [16, 32]:
            for block_k in [32, 64]:
                for block_n in [32, 64, 128, 256]:
                    num_warps = 2 if block_n <= 64 else 4
                    configs.append(
                        Config(
                            {
                                "BLOCK_M": block_m,
                                "BLOCK_N": block_n,
                                "BLOCK_K": block_k,
                                "SPLIT_K": 1,
                            },
                            num_stages=num_stages,
                            num_warps=num_warps,
                        )
                    )
                    # split_k
                    for split_k in []:  # Disabled [2, 4, 8, 16]:
                        configs.append(
                            Config(
                                {
                                    "BLOCK_M": block_m,
                                    "BLOCK_N": block_n,
                                    "BLOCK_K": block_k,
                                    "SPLIT_K": split_k,
                                },
                                num_stages=num_stages,
                                num_warps=num_warps,
                                pre_hook=init_to_zero("C"),
                            )
                        )
    return configs


def dummy_prune_configs(configs, named_args, **kwargs):
    M = named_args["M"]
    N = named_args["N"]
    K = named_args["K"]

    logger.info(f"{len(configs)=} {len(configs)=} for {M=} {N=} {K=}")
    return configs


MATMUL_CONFIGS: List[Config] = [
    # basic configs for compute-bound matmuls
    Config(
        {"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 32, "SPLIT_K": 1},
        num_stages=3,
        num_warps=8,
    ),
    Config(
        {"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 32, "SPLIT_K": 1},
        num_stages=3,
        num_warps=8,
    ),
    Config(
        {"BLOCK_M": 256, "BLOCK_N": 64, "BLOCK_K": 32, "SPLIT_K": 1},
        num_stages=4,
        num_warps=4,
    ),
    Config(
        {"BLOCK_M": 64, "BLOCK_N": 256, "BLOCK_K": 32, "SPLIT_K": 1},
        num_stages=4,
        num_warps=4,
    ),
    Config(
        {"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 128, "SPLIT_K": 1},
        num_stages=4,
        num_warps=4,
    ),
    Config(
        {"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 256, "SPLIT_K": 1},
        num_stages=4,
        num_warps=4,
    ),
    Config(
        {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32, "SPLIT_K": 1},
        num_stages=4,
        num_warps=4,
    ),
    Config(
        {"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 32, "SPLIT_K": 1},
        num_stages=4,
        num_warps=4,
    ),
    Config(
        {"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32, "SPLIT_K": 1},
        num_stages=4,
        num_warps=4,
    ),
    Config(
        {"BLOCK_M": 128, "BLOCK_N": 32, "BLOCK_K": 32, "SPLIT_K": 1},
        num_stages=4,
        num_warps=4,
    ),
    Config(
        {"BLOCK_M": 64, "BLOCK_N": 32, "BLOCK_K": 32, "SPLIT_K": 1},
        num_stages=5,
        num_warps=2,
    ),
    # good for int8
    Config(
        {"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 128, "SPLIT_K": 1},
        num_stages=3,
        num_warps=8,
    ),
    Config(
        {"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 128, "SPLIT_K": 1},
        num_stages=3,
        num_warps=8,
    ),
    Config(
        {"BLOCK_M": 256, "BLOCK_N": 64, "BLOCK_K": 128, "SPLIT_K": 1},
        num_stages=4,
        num_warps=4,
    ),
    Config(
        {"BLOCK_M": 64, "BLOCK_N": 256, "BLOCK_K": 128, "SPLIT_K": 1},
        num_stages=4,
        num_warps=4,
    ),
    Config(
        {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 128, "SPLIT_K": 1},
        num_stages=4,
        num_warps=4,
    ),
    Config(
        {"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 64, "SPLIT_K": 1},
        num_stages=4,
        num_warps=4,
    ),
    Config(
        {"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 64, "SPLIT_K": 1},
        num_stages=4,
        num_warps=4,
    ),
    Config(
        {"BLOCK_M": 128, "BLOCK_N": 32, "BLOCK_K": 64, "SPLIT_K": 1},
        num_stages=4,
        num_warps=4,
    ),
    Config(
        {"BLOCK_M": 64, "BLOCK_N": 32, "BLOCK_K": 64, "SPLIT_K": 1},
        num_stages=5,
        num_warps=2,
    ),
] + get_configs_io_bound()


@triton.autotune(
    configs=MATMUL_CONFIGS,
    prune_configs_by={
        "early_config_prune": dummy_prune_configs,
    },
    key=[
        "m_key",
        "n_key",
        "k_key",
    ],
)
@triton.jit
def _kernel_matmul_fp8_row(
    A_ptr,
    B_ptr,
    C_ptr,
    M,
    N,
    K,
    m_key,
    n_key,
    k_key,
    A_scale,
    B_scale,
    Bias,
    stride_am,
    stride_ak,
    stride_bn,
    stride_bk,
    stride_cm,
    stride_cn,
    dot_out_dtype: tl.constexpr,
    allow_tf32: tl.constexpr,
    fp8_fast_accum: tl.constexpr,
    skip_scaling_a: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    SPLIT_K: tl.constexpr,
    USE_BIAS: tl.constexpr,
    AB_DTYPE: tl.constexpr,
    NUM_SMS: tl.constexpr,
) -> None:
    """Matmul kernel of [M, K] @ [N, K] with row-wise scales

    performs swizzled matmul in [BLOCK_M, BLOCK_K] with [BLOCK_K, BLOCK_N] tiles.

    Args:
        A (TensorWrapper): [M, K] input tensor.
        B (TensorWrapper): [N, K] input tensor.
        C (TensorWrapper): [M, N] output tensor.
        M (int): M dimension of input tensor.
        N (int): N dimension of input tensor.
        K (int): K dimension of input tensor.
        m_key (int): Autotuning key for M dimension of input tensor.
        n_key (int): Autotuning key for N dimension of input tensor.
        k_key (int): Autotuning key for K dimension of input tensor.
        A_scale (TensorWrapper): [M] reciprocal scale tensor per row. A * A_scale = original A.
        B_scale (TensorWrapper): [N] reciprocal scale tensor per row. B * B_scale = original B.
        Bias (tensorWrapper): [N] Optional bias tensor.
        stride_am (int): Stride of M dimension of A.
        stride_ak (int): Stride of K dimension of A.
        stride_bn (int): Stride of N dimension of B.
        stride_bk (int): Stride of K dimension of B.
        stride_cm (int): Stride of M dimension of C.
        stride_cn (int): Stride of N dimension of C.
        dot_out_dtype (torch.dtype): Output type of tensor core.
        allow_tf32 (bool): Whether to use TF32 for tensor core.
        fp8_fast_accum (bool): Whether to use fast accumulation for tensor core.
        BLOCK_M (int): Block size for M dimension.
        BLOCK_N (int): Block size for N dimension.
        BLOCK_K (int): Block size for K dimension.
        GROUP_M (int): Number of groups for M dimension swizzle.
        SPLIT_K (int): Number of SM's to launch per row.
        USE_BIAS (bool): Whether to use bias.
        EVEN_K (bool): Whether K is evenly divisible by BLOCK_K * SPLIT_K.
        AB_DTYPE (bool): Whether to cast A and B to C.dtype before tensor core.
    """
    # Matrix multiplication.
    start_pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    k_tiles = tl.cdiv(K, BLOCK_K)
    num_tiles = num_pid_m * num_pid_n

    offs_k_for_mask = tl.arange(0, BLOCK_K)

    num_pid_in_group = GROUP_M * num_pid_n

    acc_dtype = tl.float32 if allow_tf32 else dot_out_dtype

    # Outer loop over tiles assigned to this SM
    for tile_id in range(start_pid, num_tiles, NUM_SMS):
        group_id = tile_id // num_pid_in_group
        first_pid_m = group_id * GROUP_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
        # pyre-ignore[58]: `%` is not supported for operand types `int` and `tl.core.constexpr`.
        pid_m = first_pid_m + (tile_id % group_size_m)
        pid_n = (tile_id % num_pid_in_group) // group_size_m

        start_m = pid_m * BLOCK_M
        start_n = pid_n * BLOCK_N
        offs_am = start_m + tl.arange(0, BLOCK_M)
        offs_bn = start_n + tl.arange(0, BLOCK_N)
        offs_am = tl.where(offs_am < M, offs_am, 0)
        offs_bn = tl.where(offs_bn < N, offs_bn, 0)
        offs_am = tl.max_contiguous(tl.multiple_of(offs_am, BLOCK_M), BLOCK_M)
        offs_bn = tl.max_contiguous(tl.multiple_of(offs_bn, BLOCK_N), BLOCK_N)

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=acc_dtype)

        # Inner loop over K dimension
        for ki in range(0, k_tiles):
            offs_k = ki * BLOCK_K + tl.arange(0, BLOCK_K)
            A = A_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
            B = B_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

            a = tl.load(A, mask=offs_k_for_mask[None, :] < K - ki * BLOCK_K, other=0.0)
            b = tl.load(B, mask=offs_k_for_mask[:, None] < K - ki * BLOCK_K, other=0.0)
            acc = tl.dot(a, b, acc, out_dtype=acc_dtype, allow_tf32=allow_tf32)

        # rematerialize rm and rn to save registers
        rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

        # Invert scaling.
        b_scale = tl.load(B_scale + rn, mask=rn < N)
        if skip_scaling_a:
            acc *= b_scale[None, :]
        else:
            a_scale = tl.load(A_scale + rm, mask=rm < M)
            # pyre-ignore[16]: Undefined attribute [16]: `float`
            # has no attribute `__getitem__`.
            scale = a_scale[:, None] * b_scale[None, :]
            acc *= scale

        # Load and add bias if specified.
        if USE_BIAS:
            bias = tl.load(Bias + rn, mask=rn < N)
            acc += bias[None, :]

        acc = acc.to(C_ptr.dtype.element_ty)
        C = C_ptr + (rm[:, None] * stride_cm + rn[None, :] * stride_cn)
        mask = (rm < M)[:, None] & (rn < N)[None, :]
        # Handles write-back with reduction-splitting
        tl.store(C, acc, mask=mask)


@triton.autotune(
    configs=MATMUL_CONFIGS
    + [
        Config(
            {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 128, "SPLIT_K": 1},
            num_stages=3,
            num_warps=8,
        ),
    ],
    key=[
        "m_key",
        "n_key",
        "k_key",
    ],
)
@triton.heuristics(
    {
        "EVEN_K": lambda args: args["K"] % (args["BLOCK_K"] * args["SPLIT_K"]) == 0,
    }
)
@triton.jit
def _kernel_matmul_fp8_row_no_fast_acc(
    A_ptr,
    B_ptr,
    C_ptr,
    M,
    N,
    K,
    m_key,
    n_key,
    k_key,
    A_scale,
    B_scale,
    Bias,
    stride_am,
    stride_ak,
    stride_bn,
    stride_bk,
    stride_cm,
    stride_cn,
    dot_out_dtype: tl.constexpr,
    allow_tf32: tl.constexpr,
    fp8_fast_accum: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    SPLIT_K: tl.constexpr,
    EVEN_K: tl.constexpr,
    USE_BIAS: tl.constexpr,
    AB_DTYPE: tl.constexpr,
    NUM_SMS: tl.constexpr,
) -> None:
    """Matmul kernel of [M, K] @ [N, K] with row-wise scales

    performs swizzled matmul in [BLOCK_M, BLOCK_K] with [BLOCK_K, BLOCK_N] tiles.

    Args:
        A (TensorWrapper): [M, K] input tensor.
        B (TensorWrapper): [N, K] input tensor.
        C (TensorWrapper): [M, N] output tensor.
        M (int): M dimension of input tensor.
        N (int): N dimension of input tensor.
        K (int): K dimension of input tensor.
        m_key (int): Autotuning key for M dimension of input tensor.
        n_key (int): Autotuning key for N dimension of input tensor.
        k_key (int): Autotuning key for K dimension of input tensor.
        A_scale (TensorWrapper): [M] reciprocal scale tensor per row. A * A_scale = original A
        B_scale (TensorWrapper): [N] reciprocal scale tensor per row. B * B_scale = original B
        Bias (TensorWrapper): [N] Optional bias tensor.
        stride_am (int): Stride of M dimension of A.
        stride_ak (int): Stride of K dimension of A.
        stride_bn (int): Stride of N dimension of B.
        stride_bk (int): Stride of K dimension of B.
        stride_cm (int): Stride of M dimension of C.
        stride_cn (int): Stride of N dimension of C.
        dot_out_dtype (torch.dtype): Output type of tensor core.
        allow_tf32 (bool): Whether to use TF32 for tensor core.
        fp8_fast_accum (bool): Whether to use fast accumulation for tensor core.
        BLOCK_M (int): Block size for M dimension.
        BLOCK_N (int): Block size for N dimension.
        BLOCK_K (int): Block size for K dimension.
        GROUP_M (int): Number of groups for M dimension swizzle.
        SPLIT_K (int): Number of SM's to launch per row.
        EVEN_K (bool): Whether K is evenly divisible by BLOCK_K * SPLIT_K.
        USE_BIAS(bool): Whether to use bias.
        AB_DTYPE (bool): Whether to cast A and B to C.dtype before tensor core.
    """
    # Matrix multiplication.

    start_pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    k_tiles = tl.cdiv(K, BLOCK_K)
    num_tiles = num_pid_m * num_pid_n

    tiles_per_SM = num_tiles // NUM_SMS
    if start_pid < num_tiles % NUM_SMS:
        tiles_per_SM += 1

    tile_id = start_pid - NUM_SMS
    ki = -1

    offs_k_for_mask = tl.arange(0, BLOCK_K)

    num_pid_in_group = GROUP_M * num_pid_n

    pid_m = 0
    pid_n = 0
    offs_am = tl.arange(0, BLOCK_M)
    offs_bn = tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=dot_out_dtype)

    for _ in range(0, k_tiles * tiles_per_SM):
        ki = tl.where(ki == k_tiles - 1, 0, ki + 1)
        if ki == 0:
            tile_id += NUM_SMS
            group_id = tile_id // num_pid_in_group
            first_pid_m = group_id * GROUP_M
            group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
            pid_m = first_pid_m + (tile_id % group_size_m)
            pid_n = (tile_id % num_pid_in_group) // group_size_m

            start_m = pid_m * BLOCK_M
            start_n = pid_n * BLOCK_N
            offs_am = start_m + tl.arange(0, BLOCK_M)
            offs_bn = start_n + tl.arange(0, BLOCK_N)
            offs_am = tl.where(offs_am < M, offs_am, 0)
            offs_bn = tl.where(offs_bn < N, offs_bn, 0)
            offs_am = tl.max_contiguous(tl.multiple_of(offs_am, BLOCK_M), BLOCK_M)
            offs_bn = tl.max_contiguous(tl.multiple_of(offs_bn, BLOCK_N), BLOCK_N)
        offs_k = ki * BLOCK_K + tl.arange(0, BLOCK_K)
        A = A_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
        B = B_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

        a = tl.load(A, mask=offs_k_for_mask[None, :] < K - ki * BLOCK_K, other=0.0)
        b = tl.load(B, mask=offs_k_for_mask[:, None] < K - ki * BLOCK_K, other=0.0)
        acc += tl.dot(a, b, out_dtype=dot_out_dtype, allow_tf32=allow_tf32)

        if ki == k_tiles - 1:
            # rematerialize rm and rn to save registers
            rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
            rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

            # Invert scaling.
            a_scale = tl.load(A_scale + rm, mask=rm < M)
            b_scale = tl.load(B_scale + rn, mask=rn < N)
            # pyre-ignore[16]: Undefined attribute [16]: `float` has no attribute `__getitem__`.
            scale = a_scale[:, None] * b_scale[None, :]
            acc *= scale

            # Load and add bias if specified.
            if USE_BIAS:
                bias = tl.load(Bias + rn, mask=rn < N)
                acc += bias[None, :]

            acc = acc.to(C_ptr.dtype.element_ty)
            C = C_ptr + (rm[:, None] * stride_cm + rn[None, :] * stride_cn)
            mask = (rm < M)[:, None] & (rn < N)[None, :]
            # Handles write-back with reduction-splitting
            tl.store(C, acc, mask=mask)
            acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=dot_out_dtype)


@triton.autotune(
    configs=MATMUL_CONFIGS,
    key=[
        "m_key",
        "n_key",
        "k_key",
    ],
)
@triton.heuristics(
    {
        "EVEN_K": lambda args: args["K"] % (args["BLOCK_K"] * args["SPLIT_K"]) == 0,
    }
)
@triton.jit
def _kernel_matmul_fp8_row_imprecise_acc(
    A,
    B,
    C,
    M,
    N,
    K,
    m_key,
    n_key,
    k_key,
    A_scale,
    B_scale,
    Bias,
    stride_am,
    stride_ak,
    stride_bn,
    stride_bk,
    stride_cm,
    stride_cn,
    dot_out_dtype: tl.constexpr,
    allow_tf32: tl.constexpr,
    fp8_fast_accum: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    SPLIT_K: tl.constexpr,
    EVEN_K: tl.constexpr,
    USE_BIAS: tl.constexpr,
    AB_DTYPE: tl.constexpr,
) -> None:
    """Matmul kernel of [M, K] @ [N, K] with row-wise scales

    performs swizzled matmul in [BLOCK_M, BLOCK_K] with [BLOCK_K, BLOCK_N] tiles.

    Args:
        A (TensorWrapper): [M, K] input tensor.
        B (TensorWrapper): [N, K] input tensor.
        C (TensorWrapper): [M, N] output tensor.
        M (int): M dimension of input tensor.
        N (int): N dimension of input tensor.
        K (int): K dimension of input tensor.
        m_key (int): Autotuning key for M dimension of input tensor.
        n_key (int): Autotuning key for N dimension of input tensor.
        k_key (int): Autotuning key for K dimension of input tensor.
        A_scale (TensorWrapper): [M] reciprocal scale tensor per row. A * A_scale = original A
        B_scale (TensorWrapper): [N] reciprocal scale tensor per row. B * B_scale = original B
        Bias (TensorWrapper): [N] Optional bias tensor.
        stride_am (int): Stride of M dimension of A.
        stride_ak (int): Stride of K dimension of A.
        stride_bn (int): Stride of N dimension of B.
        stride_bk (int): Stride of K dimension of B.
        stride_cm (int): Stride of M dimension of C.
        stride_cn (int): Stride of N dimension of C.
        dot_out_dtype (torch.dtype): Output type of tensor core.
        allow_tf32 (bool): Whether to use TF32 for tensor core.
        fp8_fast_accum (bool): Whether to use fast accumulation for tensor core.
        BLOCK_M (int): Block size for M dimension.
        BLOCK_N (int): Block size for N dimension.
        BLOCK_K (int): Block size for K dimension.
        GROUP_M (int): Number of groups for M dimension swizzle.
        SPLIT_K (int): Number of SM's to launch per row.
        EVEN_K (bool): Whether K is evenly divisible by BLOCK_K * SPLIT_K.
        USE_BIAS (bool): Whether to use bias.
        AB_DTYPE (bool): Whether to cast A and B to C.dtype before tensor core.
    """
    # Matrix multiplication.
    pid = tl.program_id(0)
    pid_z = tl.program_id(1)
    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)
    # Re-order program ID for better L2 performance (swizzle).
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // (group_size)
    # Do matrix multiplication.
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    ram = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M)
    rbn = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N)
    rk = pid_z * BLOCK_K + tl.arange(0, BLOCK_K)
    # Pointers.
    A = A + (ram[:, None] * stride_am + rk[None, :] * stride_ak)
    B = B + (rk[:, None] * stride_bk + rbn[None, :] * stride_bn)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=dot_out_dtype)

    for k in range(0, tl.cdiv(K, BLOCK_K * SPLIT_K)):
        if EVEN_K:
            a = tl.load(A)
            b = tl.load(B)
        else:
            k_remaining = K - k * (BLOCK_K * SPLIT_K)
            _0 = tl.zeros((1, 1), dtype=C.dtype.element_ty)
            a = tl.load(A, mask=rk[None, :] < k_remaining, other=_0)
            b = tl.load(B, mask=rk[:, None] < k_remaining, other=_0)
        if AB_DTYPE:
            a = a.to(C.dtype.element_ty)
            b = b.to(C.dtype.element_ty)
        if fp8_fast_accum:
            acc = tl.dot(
                a,
                b,
                acc,
                max_num_imprecise_acc=32,
                out_dtype=dot_out_dtype,
                allow_tf32=allow_tf32,
            )
        else:
            acc += tl.dot(a, b, out_dtype=dot_out_dtype, allow_tf32=allow_tf32)

        A += BLOCK_K * SPLIT_K * stride_ak
        B += BLOCK_K * SPLIT_K * stride_bk

    # rematerialize rm and rn to save registers
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Invert scaling.
    a_scale = tl.load(A_scale + rm, mask=rm < M)
    b_scale = tl.load(B_scale + rn, mask=rn < N)
    # pyre-ignore[16]: Undefined attribute [16]: `float` has no attribute `__getitem__`.
    scale = a_scale[:, None] * b_scale[None, :]
    acc *= scale

    # Apply bias.
    if USE_BIAS:
        bias = tl.load(Bias + rn, mask=rn < N)
        acc += bias[None, :]

    acc = acc.to(C.dtype.element_ty)
    C = C + (rm[:, None] * stride_cm + rn[None, :] * stride_cn)
    mask = (rm < M)[:, None] & (rn < N)[None, :]
    # Handles write-back with reduction-splitting
    if SPLIT_K == 1:
        tl.store(C, acc, mask=mask)
    else:
        tl.atomic_add(C, acc, mask=mask)


@triton.autotune(
    configs=[
        Config(
            {"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 128, "SPLIT_K": 1},
            num_stages=3,
            num_warps=8,
        ),
        Config(
            {"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 128, "SPLIT_K": 1},
            num_stages=3,
            num_warps=8,
        ),
        Config(
            {"BLOCK_M": 256, "BLOCK_N": 64, "BLOCK_K": 128, "SPLIT_K": 1},
            num_stages=4,
            num_warps=4,
        ),
        Config(
            {"BLOCK_M": 64, "BLOCK_N": 256, "BLOCK_K": 128, "SPLIT_K": 1},
            num_stages=4,
            num_warps=4,
        ),
        Config(
            {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 128, "SPLIT_K": 1},
            num_stages=4,
            num_warps=4,
        ),
        Config(
            {"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 64, "SPLIT_K": 1},
            num_stages=4,
            num_warps=4,
        ),
        Config(
            {"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 64, "SPLIT_K": 1},
            num_stages=4,
            num_warps=4,
        ),
        Config(
            {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 512, "SPLIT_K": 1},
            num_stages=3,
            num_warps=4,
        ),
    ],
    key=[
        "m_key",
        "n_key",
        "k_key",
    ],
    use_cuda_graph=True,
)
@triton.heuristics(
    {
        "EVEN_K": lambda args: args["K"] % (args["BLOCK_K"] * args["SPLIT_K"]) == 0,
    }
)
@triton.jit
def _kernel_matmul_fp8_row_tma_persistent(
    A_ptr,
    B_ptr,
    C_ptr,
    M,
    N,
    K,
    m_key,
    n_key,
    k_key,
    A_scale,
    B_scale,
    Bias,
    stride_am,
    stride_ak,
    stride_bn,
    stride_bk,
    stride_cm,
    stride_cn,
    dot_out_dtype: tl.constexpr,
    c_dtype: tl.constexpr,
    bias_dtype: tl.constexpr,
    allow_tf32: tl.constexpr,
    fp8_fast_accum: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    AB_DTYPE: tl.constexpr,
    SPLIT_K: tl.constexpr,
    EVEN_K: tl.constexpr,
    NUM_SMS: tl.constexpr,
    USE_BIAS: tl.constexpr,
) -> None:
    """Matmul kernel of [M, K] @ [N, K] with row-wise scales

    performs swizzled matmul in [BLOCK_M, BLOCK_K] with [BLOCK_K, BLOCK_N] tiles.

    Args:
        A (TensorWrapper): [M, K] input tensor.
        B (TensorWrapper): [N, K] input tensor.
        C (TensorWrapper): [M, N] output tensor.
        M (int): M dimension of input tensor.
        N (int): N dimension of input tensor.
        K (int): K dimension of input tensor.
        A_scale (TensorWrapper): [M] reciprocal scale tensor per row. A * A_scale = original A
        B_scale (TensorWrapper): [N] reciprocal scale tensor per row. B * B_scale = original B
        stride_am (int): Stride of M dimension of A.
        stride_ak (int): Stride of K dimension of A.
        stride_bn (int): Stride of N dimension of B.
        stride_bk (int): Stride of K dimension of B.
        stride_cm (int): Stride of M dimension of C.
        stride_cn (int): Stride of N dimension of C.
        dot_out_dtype (torch.dtype): Output type of tensor core.
        allow_tf32 (bool): Whether to use TF32 for tensor core.
        fp8_fast_accum (bool): Whether to use fast accumulation for tensor core.
        BLOCK_M (int): Block size for M dimension.
        BLOCK_N (int): Block size for N dimension.
        BLOCK_K (int): Block size for K dimension.
        GROUP_M (int): Number of groups for M dimension swizzle.
        SPLIT_K (int): Number of SM's to launch per row.
        EVEN_K (bool): Whether K is evenly divisible by BLOCK_K * SPLIT_K.
        AB_DTYPE (bool): Whether to cast A and B to C.dtype before tensor core.
    """
    # Matrix multiplication.
    start_pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    k_tiles = tl.cdiv(K, BLOCK_K)
    num_tiles = num_pid_m * num_pid_n

    tiles_per_SM = num_tiles // NUM_SMS
    if start_pid < num_tiles % NUM_SMS:
        tiles_per_SM += 1

    tile_id = start_pid - NUM_SMS
    ki = -1

    pid_m = 0
    pid_n = 0
    offs_am = 0
    offs_bn = 0

    num_pid_in_group = GROUP_M * num_pid_n

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=dot_out_dtype)

    dtype_fp8 = tl.float8e4nv
    scale_dtype = tl.float32

    for _ in range(0, k_tiles * tiles_per_SM):
        ki = tl.where(ki == k_tiles - 1, 0, ki + 1)
        if ki == 0:
            tile_id += NUM_SMS
            group_id = tile_id // num_pid_in_group
            first_pid_m = group_id * GROUP_M
            group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
            pid_m = first_pid_m + (tile_id % group_size_m)
            pid_n = (tile_id % num_pid_in_group) // group_size_m

            offs_am = pid_m * BLOCK_M
            offs_bn = pid_n * BLOCK_N
            offs_am = tl.multiple_of(offs_am, BLOCK_M)
            offs_bn = tl.multiple_of(offs_bn, BLOCK_N)

        offs_k = ki * BLOCK_K

        a = tl._experimental_descriptor_load(
            A_ptr, [offs_am, offs_k], [BLOCK_M, BLOCK_K], dtype_fp8
        )
        b = tl._experimental_descriptor_load(
            B_ptr, [offs_bn, offs_k], [BLOCK_N, BLOCK_K], dtype_fp8
        )

        if fp8_fast_accum:
            acc = tl.dot(a, b.T, acc, out_dtype=dot_out_dtype, allow_tf32=allow_tf32)
        else:
            acc += tl.dot(a, b.T, out_dtype=dot_out_dtype, allow_tf32=allow_tf32)

        if ki == k_tiles - 1:
            # rematerialize rm and rn to save registers

            # # Invert scaling.
            a_scale = tl._experimental_descriptor_load(
                A_scale, [offs_am], [BLOCK_M], scale_dtype
            )
            b_scale = tl._experimental_descriptor_load(
                B_scale, [offs_bn], [BLOCK_N], scale_dtype
            )
            # pyre-ignore[16]: Undefined attribute [16]: `float` has no attribute `__getitem__`.
            scale = a_scale[:, None] * b_scale[None, :]
            acc *= scale

            # Load and add bias if specified.
            if USE_BIAS:
                bias = tl._experimental_descriptor_load(
                    Bias, [offs_bn], [BLOCK_N], bias_dtype
                )
                acc += bias[None, :]

            acc = acc.to(c_dtype)
            tl._experimental_descriptor_store(C_ptr, acc, [offs_am, offs_bn])
            acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=dot_out_dtype)


has_warp_specialization = hasattr(tl, "async_task")


def make_autotuner_config(dictargs, **kwargs):
    # NOTE: Triton 3.4.x removed some keyword arguments from Config constructor;
    # however, fbcode uses 3.3.1, and so this shim is provided to support both
    # versions.
    #
    # https://github.com/triton-lang/triton/blob/v3.3.1/python/triton/runtime/autotuner.py#L275
    # https://github.com/triton-lang/triton/blame/release/3.4.x/python/triton/runtime/autotuner.py#L319
    if version.parse(triton.__version__) > version.parse("3.3.1"):
        for key in ["num_buffers_warp_spec", "num_consumer_groups"]:
            kwargs.pop(key, None)
    return Config(dictargs, **kwargs)


def get_ws_configs() -> List[Config]:
    if not has_warp_specialization:
        return []
    return [
        make_autotuner_config(
            {
                "BLOCK_M": 128,
                "BLOCK_N": 256,
                "BLOCK_K": 128,
                "SPLIT_K": 1,
                "NUM_CONSUMER_GROUPS": 2,
            },
            num_stages=3,
            num_warps=4,
            num_consumer_groups=2,
            num_buffers_warp_spec=3,
        ),
        make_autotuner_config(
            {
                "BLOCK_M": 128,
                "BLOCK_N": 128,
                "BLOCK_K": 128,
                "SPLIT_K": 1,
                "NUM_CONSUMER_GROUPS": 2,
            },
            num_stages=4,
            num_warps=4,
            num_consumer_groups=2,
            num_buffers_warp_spec=4,
        ),
        make_autotuner_config(
            {
                "BLOCK_M": 128,
                "BLOCK_N": 256,
                "BLOCK_K": 128,
                "SPLIT_K": 1,
                "NUM_CONSUMER_GROUPS": 1,
            },
            num_stages=3,
            num_warps=8,
            num_consumer_groups=0,
            num_buffers_warp_spec=3,
        ),
        make_autotuner_config(
            {
                "BLOCK_M": 64,
                "BLOCK_N": 64,
                "BLOCK_K": 512,
                "SPLIT_K": 1,
                "NUM_CONSUMER_GROUPS": 1,
            },
            num_stages=3,
            num_warps=4,
            num_consumer_groups=0,
            num_buffers_warp_spec=3,
        ),
    ]


@triton.autotune(
    configs=[
        Config(
            {
                "BLOCK_M": 128,
                "BLOCK_N": 256,
                "BLOCK_K": 128,
                "SPLIT_K": 1,
                "NUM_CONSUMER_GROUPS": 1,
            },
            num_stages=3,
            num_warps=8,
        ),
    ]
    + get_ws_configs(),
    key=[
        "m_key",
        "n_key",
        "k_key",
    ],
    use_cuda_graph=True,
)
@triton.heuristics(
    {
        "EVEN_K": lambda args: args["K"] % (args["BLOCK_K"] * args["SPLIT_K"]) == 0,
    }
)
@triton.jit
def _kernel_matmul_fp8_row_tma_persistent_ws_cooperative(
    A_ptr,
    B_ptr,
    C_ptr,
    M,
    N,
    K,
    m_key,
    n_key,
    k_key,
    A_scale,
    B_scale,
    Bias,
    stride_am,
    stride_ak,
    stride_bn,
    stride_bk,
    stride_cm,
    stride_cn,
    dot_out_dtype: tl.constexpr,
    c_dtype: tl.constexpr,
    bias_dtype: tl.constexpr,
    allow_tf32: tl.constexpr,
    fp8_fast_accum: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    AB_DTYPE: tl.constexpr,
    SPLIT_K: tl.constexpr,
    EVEN_K: tl.constexpr,
    NUM_SMS: tl.constexpr,
    USE_BIAS: tl.constexpr,
    NUM_CONSUMER_GROUPS: tl.constexpr,
) -> None:
    """Matmul kernel of [M, K] @ [N, K] with row-wise scales

    performs swizzled matmul in [BLOCK_M, BLOCK_K] with [BLOCK_K, BLOCK_N] tiles.

    Args:
        A (TensorWrapper): [M   , K] input tensor.
        B (TensorWrapper): [N, K] input tensor.
        C (TensorWrapper): [M, N] output tensor.
        M (int): M dimension of input tensor.
        N (int): N dimension of input tensor.
        K (int): K dimension of input tensor.
        A_scale (TensorWrapper): [M] reciprocal scale tensor per row. A * A_scale = original A
        B_scale (TensorWrapper): [N] reciprocal scale tensor per row. B * B_scale = original B
        stride_am (int): Stride of M dimension of A.
        stride_ak (int): Stride of K dimension of A.
        stride_bn (int): Stride of N dimension of B.
        stride_bk (int): Stride of K dimension of B.
        stride_cm (int): Stride of M dimension of C.
        stride_cn (int): Stride of N dimension of C.
        dot_out_dtype (torch.dtype): Output type of tensor core.
        allow_tf32 (bool): Whether to use TF32 for tensor core.
        fp8_fast_accum (bool): Whether to use fast accumulation for tensor core.
        BLOCK_M (int): Block size for M dimension.
        BLOCK_N (int): Block size for N dimension.
        BLOCK_K (int): Block size for K dimension.
        GROUP_M (int): Number of groups for M dimension swizzle.
        SPLIT_K (int): Number of SM's to launch per row.
        EVEN_K (bool): Whether K is evenly divisible by BLOCK_K * SPLIT_K.
        AB_DTYPE (bool): Whether to cast A and B to C.dtype before tensor core.
    """
    num_tiles = tl.cdiv(M, BLOCK_M) * tl.cdiv(N, BLOCK_N)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    dtype_fp8 = tl.float8e4nv
    for pid in range(tl.program_id(0), num_tiles, tl.num_programs(0)):
        num_pid_in_group = GROUP_M * num_pid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
        # pyre-ignore
        pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
        pid_n = (pid % num_pid_in_group) // group_size_m

        # ----------------------------------------------------------
        # Create pointers for the first blocks of A and B.
        # We will advance this pointer as we move in the K direction
        # and accumulate
        # `a_ptrs` is a block of [BLOCK_M, BLOCK_K] pointers
        # `b_ptrs` is a block of [BLOCK_K, BLOCK_N] pointers
        # See above `Pointer Arithmetic` section for details
        offs_am = pid_m * BLOCK_M
        offs_bn = pid_n * BLOCK_N
        offs_k = 0
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=dot_out_dtype)
        # pyre-ignore
        tl.assume(tl.cdiv(K, BLOCK_K) > 0)
        for _ in range(0, tl.cdiv(K, BLOCK_K)):
            # pyre-ignore
            with tl.async_task([0]):
                a = tl._experimental_descriptor_load(
                    A_ptr,
                    [offs_am, offs_k],
                    [BLOCK_M, BLOCK_K],
                    dtype_fp8,
                )
                b = tl._experimental_descriptor_load(
                    B_ptr, [offs_bn, offs_k], [BLOCK_N, BLOCK_K], dtype_fp8
                )

            if fp8_fast_accum:
                acc = tl.dot(
                    a, b.T, acc, out_dtype=dot_out_dtype, allow_tf32=allow_tf32
                )
            else:
                acc += tl.dot(a, b.T, out_dtype=dot_out_dtype, allow_tf32=allow_tf32)

            offs_k += BLOCK_K

        # pyre-ignore
        with tl.async_task([1, NUM_CONSUMER_GROUPS]):
            # Invert scaling.
            rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
            rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
            a_scale = tl.load(A_scale + rm, mask=rm < M)
            b_scale = tl.load(B_scale + rn, mask=rn < N)
            scale = a_scale[:, None] * b_scale[None, :]
            acc *= scale
            # Load and add bias if specified.
            if USE_BIAS:
                bias = tl._experimental_descriptor_load(
                    Bias, [offs_bn], [BLOCK_N], bias_dtype
                )
                acc += bias[None, :]
            acc = acc.to(c_dtype)
            tl._experimental_descriptor_store(C_ptr, acc, [offs_am, offs_bn])


def _is_eligible_for_skip_scaling(
    is_rowwise: bool,
    fp8_fast_accum: bool,
    imprecise_acc: bool,
    tma_persistent: bool,
    no_use_persistent: Optional[bool],
    use_warp_specialization: bool,
) -> bool:
    if not is_rowwise:
        return False

    return (
        fp8_fast_accum
        and not imprecise_acc
        and not tma_persistent
        and not no_use_persistent
        and not use_warp_specialization
    )


@torch._library.triton_op("triton::matmul_fp8_row", mutates_args=())
def matmul_fp8_row(
    a: torch.Tensor,
    b: torch.Tensor,
    a_scale: Optional[torch.Tensor],
    b_scale: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    dot_out_dtype: Optional[torch.dtype] = None,
    allow_tf32: bool = True,
    fp8_fast_accum: bool = True,
    imprecise_acc: bool = False,
    tma_persistent: bool = True,
    no_use_persistent: Optional[bool] = None,
    use_warp_specialization: bool = False,
) -> torch.Tensor:
    """
    Performs matmul on [M, K] and [N, K] fp8 matrices with row-wise scalings [M], [N].

    Args:
        a (torch.Tensor): [M, K] input tensor.
        b (torch.Tensor): [N, K] input tensor.
        a_scale (Optiona;[torch.Tensor]): [M] reciprocal scale tensor per row.
            A * a_scale = original A. Scaling will be skiped if a_scale is None.
        b_scale (torch.Tensor): [N] reciprocal scale tensor per row. B * b_scale = original B
        bias (torch.Tensor): [N] optional bias tensor to add to output if provided.
        dot_out_dtype (torch.dtype): Output type of tensor core.
        allow_tf32 (bool): Whether to use TF32 for tensor core.
        fp8_fast_accum (bool): Whether to use fast accumulation for tensor core.
        tma_persistent (bool): Whether to use TMA persistent kernel impl.

    Returns:
        torch.Tensor: [M, N] Output tensor a @ b / (a_scale[:, None] * b_scale[None, :])
    """
    if no_use_persistent is None:
        # Default True for AMD and False for Nvidia.
        if torch.version.hip is not None:
            no_use_persistent = True
        else:
            no_use_persistent = False
    # Get datatypes and constants to use.
    pt_fp8_dtype, _, _, _ = get_fp8_constants()
    # Handle 3D+ a shape
    a_shape = a.shape
    a = a.view(-1, a.size(-1))
    # View inputs into proper torch fp8 dtype.
    if torch.version.cuda:
        assert a.dtype in (torch.float8_e4m3fn, torch.float8_e5m2)
    elif torch.version.hip:
        if torch.cuda.get_device_capability() < (9, 5):
            assert a.dtype in (
                torch.float8_e4m3fnuz,
                torch.float8_e5m2fnuz,
            )
        else:
            assert a.dtype in (torch.float8_e4m3fn, torch.float8_e5m2)
    else:
        assert a.dtype in (torch.float8_e4m3fnuz, torch.float8_e5m2fnuz)
    assert b.dtype == pt_fp8_dtype
    M, N, K, m_key, n_key, k_key, c, c_dtype_triton, dot_out_dtype_triton, device = (
        prep_matmul(a, b, dot_out_dtype)
    )

    # Skip scaling (a_scale is None) can only be applied in certain cases.
    assert a_scale is not None or _is_eligible_for_skip_scaling(
        is_rowwise=True,
        fp8_fast_accum=fp8_fast_accum,
        imprecise_acc=imprecise_acc,
        tma_persistent=tma_persistent,
        no_use_persistent=no_use_persistent,
        use_warp_specialization=use_warp_specialization,
    )

    output_shape = a_shape[:-1] + (N,)
    # Handle tensor with empty inputs.
    if (M == 0) or (N == 0) or (K == 0):
        return torch.zeros(output_shape, device=device, dtype=torch.bfloat16)
    # launch kernel
    if a.device == torch.device("cpu"):
        logger.info(
            "FP8 Row-wise Triton kernel not supported on cpu, fallback to torch"
        )
        if a_scale is None:
            scale = b_scale[None, :]
        else:
            scale = a_scale[:, None] * b_scale[None, :]
        output = torch.matmul(a.to(torch.bfloat16), b.to(torch.bfloat16).T) * scale
        if bias is not None:
            output += bias[None, :]
        return output.to(c.dtype)

    def grid(META: Dict[str, int]) -> Tuple[int, int]:
        return (
            triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]),
            META["SPLIT_K"],
        )

    NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count

    def persistent_grid(META: Dict[str, int]) -> Tuple[int]:
        return (
            min(
                NUM_SMS,
                triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]),
            ),
        )

    if no_use_persistent:
        logger.debug("Using non-persistent kernel")
        with torch.cuda.device(a.device.index):
            torch._library.capture_triton(_kernel_matmul_fp8_row_non_persistent)[grid](
                a,
                b,
                c,
                M,
                N,
                K,
                m_key,
                n_key,
                k_key,
                a_scale,
                b_scale,
                bias,
                a.stride(0),
                a.stride(1),
                b.stride(0),
                b.stride(1),
                c.stride(0),
                c.stride(1),
                dot_out_dtype=dot_out_dtype_triton,
                allow_tf32=allow_tf32,
                fp8_fast_accum=fp8_fast_accum,
                # GROUP_M=8,
                USE_BIAS=bias is not None,
                AB_DTYPE=False,
            )
    elif use_warp_specialization:
        assert has_warp_specialization
        # used by TMA warp specialization kernel
        desc_helper = TmaAutoTuneHelper()
        desc_helper.init_tma_descriptor("a")
        desc_helper.init_tma_descriptor("b")
        desc_helper.init_tma_descriptor("c")
        desc_helper.init_tma_descriptor("a_scale")
        desc_helper.init_tma_descriptor("b_scale")
        desc_helper.init_tma_descriptor("bias")

        def persistent_grid_tma_ws(META: Dict[str, int]) -> Tuple[int]:
            nonlocal desc_helper  # noqa: F824
            assert a_scale is not None  # Type narrowing for Pyre
            desc_helper.fill_2d_tma_descriptor(
                "a",
                a.data_ptr(),
                M,
                K,
                META["BLOCK_M"] // META["NUM_CONSUMER_GROUPS"],
                META["BLOCK_K"],
                a.element_size(),
            )

            desc_helper.fill_2d_tma_descriptor(
                "b",
                b.data_ptr(),
                N,
                K,
                META["BLOCK_N"],
                META["BLOCK_K"],
                b.element_size(),
            )
            desc_helper.fill_2d_tma_descriptor(
                "c",
                c.data_ptr(),
                M,
                N,
                META["BLOCK_M"] // META["NUM_CONSUMER_GROUPS"],
                META["BLOCK_N"],
                c.element_size(),
            )
            desc_helper.fill_1d_tma_descriptor(
                "a_scale",
                a_scale.data_ptr(),
                M,
                META["BLOCK_M"] // META["NUM_CONSUMER_GROUPS"],
                a_scale.element_size(),
            )
            desc_helper.fill_1d_tma_descriptor(
                "b_scale",
                b_scale.data_ptr(),
                N,
                META["BLOCK_N"],
                b_scale.element_size(),
            )
            if bias is not None:
                desc_helper.fill_1d_tma_descriptor(
                    "bias",
                    bias.data_ptr(),
                    N,
                    META["BLOCK_N"],
                    bias.element_size(),
                )
            return (
                min(
                    NUM_SMS,
                    triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]),
                ),
            )

        desc_a = desc_helper.get_tma_descriptor_kernel_param("a")
        desc_b = desc_helper.get_tma_descriptor_kernel_param("b")
        desc_c = desc_helper.get_tma_descriptor_kernel_param("c")
        desc_a_scale = desc_helper.get_tma_descriptor_kernel_param("a_scale")
        desc_b_scale = desc_helper.get_tma_descriptor_kernel_param("b_scale")
        desc_bias = desc_helper.get_tma_descriptor_kernel_param("bias")

        bias_dtype_triton = None
        if bias is not None:
            bias_dtype_triton = map_dtype_to_triton(bias.dtype)

        # pyre-ignore
        torch._library.capture_triton(
            _kernel_matmul_fp8_row_tma_persistent_ws_cooperative
        )[persistent_grid_tma_ws](
            desc_a,
            desc_b,
            desc_c,
            M,
            N,
            K,
            m_key,
            n_key,
            k_key,
            a_scale,
            b_scale,
            desc_bias,
            a.stride(0),
            a.stride(1),
            b.stride(0),
            b.stride(1),
            c.stride(0),
            c.stride(1),
            dot_out_dtype=dot_out_dtype_triton,
            c_dtype=c_dtype_triton,
            bias_dtype=bias_dtype_triton,
            allow_tf32=allow_tf32,
            fp8_fast_accum=fp8_fast_accum,
            GROUP_M=8,
            AB_DTYPE=False,
            NUM_SMS=NUM_SMS,
            USE_BIAS=bias is not None,
        )
    elif tma_persistent:
        # used by TMA persistent kernel
        desc_helper = TmaAutoTuneHelper()
        desc_helper.init_tma_descriptor("a")
        desc_helper.init_tma_descriptor("b")
        desc_helper.init_tma_descriptor("c")
        desc_helper.init_tma_descriptor("a_scale")
        desc_helper.init_tma_descriptor("b_scale")
        desc_helper.init_tma_descriptor("bias")

        def persistent_grid_tma(META: Dict[str, int]) -> Tuple[int]:
            nonlocal desc_helper  # noqa: F824
            assert a_scale is not None  # Type narrowing for Pyre
            desc_helper.fill_2d_tma_descriptor(
                "a",
                a.data_ptr(),
                M,
                K,
                META["BLOCK_M"],
                META["BLOCK_K"],
                a.element_size(),
            )

            desc_helper.fill_2d_tma_descriptor(
                "b",
                b.data_ptr(),
                N,
                K,
                META["BLOCK_N"],
                META["BLOCK_K"],
                b.element_size(),
            )
            desc_helper.fill_2d_tma_descriptor(
                "c",
                c.data_ptr(),
                M,
                N,
                META["BLOCK_M"],
                META["BLOCK_N"],
                c.element_size(),
            )
            desc_helper.fill_1d_tma_descriptor(
                "a_scale",
                a_scale.data_ptr(),
                M,
                META["BLOCK_M"],
                a_scale.element_size(),
            )
            desc_helper.fill_1d_tma_descriptor(
                "b_scale",
                b_scale.data_ptr(),
                N,
                META["BLOCK_N"],
                b_scale.element_size(),
            )
            if bias is not None:
                desc_helper.fill_1d_tma_descriptor(
                    "bias",
                    bias.data_ptr(),
                    N,
                    META["BLOCK_N"],
                    bias.element_size(),
                )
            return (
                min(
                    NUM_SMS,
                    triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]),
                ),
            )

        desc_a = desc_helper.get_tma_descriptor_kernel_param("a")
        desc_b = desc_helper.get_tma_descriptor_kernel_param("b")
        desc_c = desc_helper.get_tma_descriptor_kernel_param("c")
        desc_a_scale = desc_helper.get_tma_descriptor_kernel_param("a_scale")
        desc_b_scale = desc_helper.get_tma_descriptor_kernel_param("b_scale")
        desc_bias = desc_helper.get_tma_descriptor_kernel_param("bias")

        bias_dtype_triton = None
        if bias is not None:
            bias_dtype_triton = map_dtype_to_triton(bias.dtype)

        # pyre-ignore
        torch._library.capture_triton(_kernel_matmul_fp8_row_tma_persistent)[
            persistent_grid_tma
        ](
            desc_a,
            desc_b,
            desc_c,
            M,
            N,
            K,
            m_key,
            n_key,
            k_key,
            desc_a_scale,
            desc_b_scale,
            desc_bias,
            a.stride(0),
            a.stride(1),
            b.stride(0),
            b.stride(1),
            c.stride(0),
            c.stride(1),
            dot_out_dtype=dot_out_dtype_triton,
            c_dtype=c_dtype_triton,
            bias_dtype=bias_dtype_triton,
            allow_tf32=allow_tf32,
            fp8_fast_accum=fp8_fast_accum,
            GROUP_M=8,
            AB_DTYPE=False,
            NUM_SMS=NUM_SMS,
            USE_BIAS=bias is not None,
        )
    elif imprecise_acc:
        with torch.cuda.device(a.device.index):
            torch._library.capture_triton(_kernel_matmul_fp8_row_imprecise_acc)[grid](
                a,
                b,
                c,
                M,
                N,
                K,
                m_key,
                n_key,
                k_key,
                a_scale,
                b_scale,
                bias,
                a.stride(0),
                a.stride(1),
                b.stride(0),
                b.stride(1),
                c.stride(0),
                c.stride(1),
                dot_out_dtype=dot_out_dtype_triton,
                allow_tf32=allow_tf32,
                fp8_fast_accum=fp8_fast_accum,
                GROUP_M=8,
                USE_BIAS=bias is not None,
                AB_DTYPE=False,
            )
    elif fp8_fast_accum:
        skip_scaling_a = a_scale is None
        torch._library.capture_triton(_kernel_matmul_fp8_row)[persistent_grid](
            a,
            b,
            c,
            M,
            N,
            K,
            m_key,
            n_key,
            k_key,
            a_scale,
            b_scale,
            bias,
            a.stride(0),
            a.stride(1),
            b.stride(0),
            b.stride(1),
            c.stride(0),
            c.stride(1),
            dot_out_dtype=dot_out_dtype_triton,
            allow_tf32=allow_tf32,
            fp8_fast_accum=fp8_fast_accum,
            skip_scaling_a=skip_scaling_a,
            GROUP_M=8,
            USE_BIAS=bias is not None,
            AB_DTYPE=False,
            NUM_SMS=NUM_SMS,
        )
    else:
        torch._library.capture_triton(_kernel_matmul_fp8_row_no_fast_acc)[
            persistent_grid
        ](
            a,
            b,
            c,
            M,
            N,
            K,
            m_key,
            n_key,
            k_key,
            a_scale,
            b_scale,
            bias,
            a.stride(0),
            a.stride(1),
            b.stride(0),
            b.stride(1),
            c.stride(0),
            c.stride(1),
            dot_out_dtype=dot_out_dtype_triton,
            allow_tf32=allow_tf32,
            fp8_fast_accum=fp8_fast_accum,
            GROUP_M=8,
            USE_BIAS=bias is not None,
            AB_DTYPE=False,
            NUM_SMS=NUM_SMS,
        )
    return c.view(output_shape)


@matmul_fp8_row.register_fake
def matmul_fp8_row_meta(
    a: torch.Tensor,
    b: torch.Tensor,
    a_scale: Optional[torch.Tensor],
    b_scale: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    dot_out_dtype: Optional[torch.dtype] = None,
    allow_tf32: bool = True,
    fp8_fast_accum: bool = True,
    imprecise_acc: bool = False,
    tma_persistent: bool = True,
    no_use_persistent: Optional[bool] = None,
    use_warp_specialization: bool = False,
) -> torch.Tensor:
    """Shape function for torch compile."""
    M, K = a.shape
    N, K = b.shape
    return torch.empty(
        (M, N),
        device=a.device,
        dtype=torch.bfloat16 if dot_out_dtype is None else dot_out_dtype,
    )


# pruned some unreasonable config
def prune_configs_block(configs, named_args, **kwargs):
    configs = early_config_prune(configs, named_args, **kwargs)
    scale_block_k = named_args["scale_block_k"]
    pruned_configs = []
    # Further rule out configs with scale_block_k is not a multiple of BLOCK_K
    for config in configs:
        kw = config.kwargs
        BLOCK_K = kw["BLOCK_K"]
        if scale_block_k % BLOCK_K != 0:
            continue
        pruned_configs.append(config)
    return pruned_configs


@triton.autotune(
    configs=MATMUL_CONFIGS,
    key=[
        "m_key",
        "n_key",
        "k_key",
    ],  # TODO caller side bin keys so similar shapes can use same triton.autotune.
    prune_configs_by={
        "early_config_prune": prune_configs_block,
        "perf_model": estimate_matmul_time,
        "top_k": 10,
    },
)
@triton.heuristics(
    {
        "EVEN_K": lambda args: args["K"] % (args["BLOCK_K"] * args["SPLIT_K"]) == 0,
    }
)
@triton.jit
def _kernel_matmul_fp8_block_fastacc(
    A,
    B,
    C,
    M,
    N,
    K,
    m_key,
    n_key,
    k_key,
    A_scale,
    B_scale,
    scale_block_m: tl.constexpr,
    scale_block_n: tl.constexpr,
    scale_block_k: tl.constexpr,
    stride_am,
    stride_ak,
    stride_bn,
    stride_bk,
    stride_cm,
    stride_cn,
    stride_scale_am,
    stride_scale_ak,
    stride_scale_bn,
    stride_scale_bk,
    dot_out_dtype: tl.constexpr,
    allow_tf32: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    SPLIT_K: tl.constexpr,
    EVEN_K: tl.constexpr,
    AB_DTYPE: tl.constexpr,
) -> None:
    """Matmul kernel of [M, K] @ [N, K] with block-wise scales

    Performs swizzled matmul in [BLOCK_M, BLOCK_K] with [BLOCK_K, BLOCK_N] tiles and
    A and B scaled by a scaling factor per [scale_block_m, scale_block_k] and
    [scale_block_n, scale_block_k] tiles
    respectively.

    Todo:
        * Support scale_block_{mnk} < BLOCK{MNK} for each dim.
    Args:
        A (TensorWrapper): [M, K] input tensor.
        B (TensorWrapper): [N, K] input tensor.
        C (TensorWrapper): [M, N] output tensor.
        M (int): M dimension of input tensor.
        N (int): N dimension of input tensor.
        K (int): K dimension of input tensor.
        m_key (int): Autotuning key for M dimension of input tensor.
        n_key (int): Autotuning key for N dimension of input tensor.
        k_key (int): Autotuning key for K dimension of input tensor.
        A_scale (TensorWrapper): [cdiv(M, scale_block_m), cdiv(K, scale_block_k)] reciprocal scale tensor per block. A * A_scale = original A
        B_scale (TensorWrapper): [cdiv(N, scale_block_n), cdiv(K, scale_block_k)] reciprocal scale tensor per block. B * B_scale = original B
        scale_block_m (int): Block size for M dimension of A_scale.
        scale_block_n (int): Block size for N dimension of B_scale.
        scale_block_k (int): Block size for K dimension of A_scale and B_scale.
        stride_am (int): Stride of M dimension of A.
        stride_ak (int): Stride of K dimension of A.
        stride_bn (int): Stride of N dimension of B.
        stride_bk (int): Stride of K dimension of B.
        stride_cm (int): Stride of M dimension of C.
        stride_cn (int): Stride of N dimension of C.
        stride_scale_am (int): Stride of M dimension of A_scale.
        stride_scale_ak (int): Stride of K dimension of A_scale.
        stride_scale_bn (int): Stride of N dimension of B_scale.
        stride_scale_bk (int): Stride of K dimension of B_scale.
        dot_out_dtype (torch.dtype): Output type of tensor core.
        allow_tf32 (bool): Whether to use TF32 for tensor core.
        fp8_fast_accum (bool): Whether to use fast accumulation for tensor core.
        BLOCK_M (int): Block size for M dimension.
        BLOCK_N (int): Block size for N dimension.
        BLOCK_K (int): Block size for K dimension.
        GROUP_M (int): Number of groups for M dimension swizzle.
        SPLIT_K (int): Number of SM's to launch per row.
        EVEN_K (bool): Whether K is evenly divisible by BLOCK_K * SPLIT_K.
        AB_DTYPE (bool): Whether to cast A and B to C.dtype before tensor core.
    """
    assert BLOCK_M < scale_block_m
    assert BLOCK_N < scale_block_n
    assert BLOCK_K < scale_block_k
    # matrix multiplication
    pid = tl.program_id(0)
    pid_z = tl.program_id(1)

    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)
    # re-order program ID for better L2 performance
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // (group_size)
    # do matrix multiplication
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    ram = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M)
    rbn = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N)
    rk = pid_z * BLOCK_K + tl.arange(0, BLOCK_K)
    # pointers
    A = A + (ram[:, None] * stride_am + rk[None, :] * stride_ak)
    B = B + (rk[:, None] * stride_bk + rbn[None, :] * stride_bn)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=dot_out_dtype)
    _0 = tl.zeros((1, 1), dtype=C.dtype.element_ty)
    scale_m = pid_m * BLOCK_M // scale_block_m
    scale_n = pid_n * BLOCK_N // scale_block_n
    k_multiple = scale_block_k // BLOCK_K

    for k in range(0, tl.cdiv(K, BLOCK_K * SPLIT_K)):
        k_remaining = K - k * (BLOCK_K * SPLIT_K)

        if EVEN_K:
            a = tl.load(A)
            b = tl.load(B)
        else:
            a = tl.load(A, mask=rk[None, :] < k_remaining, other=_0)
            b = tl.load(B, mask=rk[:, None] < k_remaining, other=_0)
        if AB_DTYPE:
            a = a.to(C.dtype.element_ty)
            b = b.to(C.dtype.element_ty)

        acc = tl.dot(a, b, acc, out_dtype=dot_out_dtype, allow_tf32=allow_tf32)

        A += BLOCK_K * SPLIT_K * stride_ak
        B += BLOCK_K * SPLIT_K * stride_bk

        # Some math to precompute on scalars, and apply once on matrix.
        # a + c/s = (as + c) / s
        # (((a_i-1 * s_i-1 + c_i-1) / s_i-1) * s_i + c_i) / s_i ... ) * s_k + c_k) * 1.0 / s_k
        # Simplifies to (a_i-1 + c) * (s_i+1/s_i)
        # And have s_k+1 be 1.
        # Scale_i = pid_i * BLOCK_I / scale_block_i
        pid_k = k * SPLIT_K + pid_z
        if ((pid_k + 1) % k_multiple == 0) or (k_remaining < BLOCK_K * SPLIT_K):
            # Note: Due to split_k access "pid_k" = k * SPLIT_K + pid_z
            # Access a_scale[pid_m, k * SPLIT_K + pid_z]
            # and b_scale[k * SPLIT_K + pid_z, pid_n]

            scale_k = pid_k // k_multiple
            scale_k_next = scale_k + 1
            a_scale = tl.load(
                A_scale + scale_m * stride_scale_am + scale_k * stride_scale_ak
            )
            b_scale = tl.load(
                B_scale + scale_n * stride_scale_bn + scale_k * stride_scale_bk
            )
            scale = a_scale * b_scale
            if k + 1 == tl.cdiv(K, BLOCK_K * SPLIT_K):
                scale_next_inv_scale = scale
            else:
                a_scale_next = tl.load(
                    A_scale + scale_m * stride_scale_am + scale_k_next * stride_scale_ak
                )
                b_scale_next = tl.load(
                    B_scale + scale_n * stride_scale_bn + scale_k_next * stride_scale_bk
                )
                scale_next = a_scale_next * b_scale_next
                scale_next_inv_scale = scale / scale_next
            acc *= scale_next_inv_scale

    # rematerialize rm and rn to save registers
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = acc.to(C.dtype.element_ty)
    c = C + (rm[:, None] * stride_cm + rn[None, :] * stride_cn)
    mask = (rm < M)[:, None] & (rn < N)[None, :]
    # handles write-back with reduction-splitting
    if SPLIT_K == 1:
        tl.store(c, acc, mask=mask)
    else:
        tl.atomic_add(c, acc, mask=mask)


@triton.autotune(
    configs=MATMUL_CONFIGS,
    key=[
        "m_key",
        "n_key",
        "k_key",
    ],  # TODO caller side bin keys so similar shapes can use same triton.autotune.
    prune_configs_by={
        "early_config_prune": early_config_prune,
        "perf_model": estimate_matmul_time,
        "top_k": 10,
    },
)
@triton.heuristics(
    {
        "EVEN_K": lambda args: args["K"] % (args["BLOCK_K"] * args["SPLIT_K"]) == 0,
    }
)
@triton.jit
def _kernel_matmul_fp8_block_slowacc(
    A,
    B,
    C,
    M,
    N,
    K,
    m_key,
    n_key,
    k_key,
    A_scale,
    B_scale,
    scale_block_m: tl.constexpr,
    scale_block_n: tl.constexpr,
    scale_block_k: tl.constexpr,
    stride_am,
    stride_ak,
    stride_bn,
    stride_bk,
    stride_cm,
    stride_cn,
    stride_scale_am,
    stride_scale_ak,
    stride_scale_bn,
    stride_scale_bk,
    dot_out_dtype: tl.constexpr,
    allow_tf32: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    SPLIT_K: tl.constexpr,
    EVEN_K: tl.constexpr,
    AB_DTYPE: tl.constexpr,
) -> None:
    """Matmul kernel of [M, K] @ [N, K] with block-wise scales

    Performs swizzled matmul in [BLOCK_M, BLOCK_K] with [BLOCK_K, BLOCK_N] tiles and
    A and B scaled by a scaling factor per [scale_block_m, scale_block_k] and
    [scale_block_n, scale_block_k] tiles
    respectively.

    Todo:
        * Support scale_block_{mnk} < BLOCK{MNK} for each dim.
    Args:
        A (TensorWrapper): [M, K] input tensor.
        B (TensorWrapper): [N, K] input tensor.
        C (TensorWrapper): [M, N] output tensor.
        M (int): M dimension of input tensor.
        N (int): N dimension of input tensor.
        K (int): K dimension of input tensor.
        m_key (int): Autotuning key for M dimension of input tensor.
        n_key (int): Autotuning key for N dimension of input tensor.
        k_key (int): Autotuning key for K dimension of input tensor.
        A_scale (TensorWrapper): [cdiv(M, scale_block_m), cdiv(K, scale_block_k)] reciprocal scale tensor per block. A * A_scale = original A
        B_scale (TensorWrapper): [cdiv(N, scale_block_n), cdiv(K, scale_block_k)] reciprocal scale tensor per block. B * B_scale = original B
        scale_block_m (int): Block size for M dimension of A_scale.
        scale_block_n (int): Block size for N dimension of B_scale.
        scale_block_k (int): Block size for K dimension of A_scale and B_scale.
        stride_am (int): Stride of M dimension of A.
        stride_ak (int): Stride of K dimension of A.
        stride_bn (int): Stride of N dimension of B.
        stride_bk (int): Stride of K dimension of B.
        stride_cm (int): Stride of M dimension of C.
        stride_cn (int): Stride of N dimension of C.
        stride_scale_am (int): Stride of M dimension of A_scale.
        stride_scale_ak (int): Stride of K dimension of A_scale.
        stride_scale_bn (int): Stride of N dimension of B_scale.
        stride_scale_bk (int): Stride of K dimension of B_scale.
        dot_out_dtype (torch.dtype): Output type of tensor core.
        allow_tf32 (bool): Whether to use TF32 for tensor core.
        fp8_fast_accum (bool): Whether to use fast accumulation for tensor core.
        BLOCK_M (int): Block size for M dimension.
        BLOCK_N (int): Block size for N dimension.
        BLOCK_K (int): Block size for K dimension.
        GROUP_M (int): Number of groups for M dimension swizzle.
        SPLIT_K (int): Number of SM's to launch per row.
        EVEN_K (bool): Whether K is evenly divisible by BLOCK_K * SPLIT_K.
        AB_DTYPE (bool): Whether to cast A and B to C.dtype before tensor core.
    """
    assert BLOCK_M < scale_block_m
    assert BLOCK_N < scale_block_n
    assert BLOCK_K < scale_block_k
    # matrix multiplication
    pid = tl.program_id(0)
    pid_z = tl.program_id(1)

    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)
    # re-order program ID for better L2 performance
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // (group_size)
    # do matrix multiplication
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    ram = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M)
    rbn = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N)
    rk = pid_z * BLOCK_K + tl.arange(0, BLOCK_K)
    # pointers
    A = A + (ram[:, None] * stride_am + rk[None, :] * stride_ak)
    B = B + (rk[:, None] * stride_bk + rbn[None, :] * stride_bn)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=dot_out_dtype)
    scale_m = pid_m * BLOCK_M // scale_block_m
    scale_n = pid_n * BLOCK_N // scale_block_n
    _0 = tl.zeros((1, 1), dtype=C.dtype.element_ty)

    for k in range(0, tl.cdiv(K, BLOCK_K * SPLIT_K)):
        # Note: Due to split_k access "pid_k" = k * SPLIT_K + pid_z
        # Access a_scale[pid_m, k * SPLIT_K + pid_z]
        # and b_scale[k * SPLIT_K + pid_z, pid_n]
        pid_k = k * SPLIT_K + pid_z
        scale_k = pid_k * BLOCK_K // scale_block_k
        a_scale = tl.load(
            A_scale + scale_m * stride_scale_am + scale_k * stride_scale_ak
        )
        b_scale = tl.load(
            B_scale + scale_n * stride_scale_bn + scale_k * stride_scale_bk
        )
        scale = a_scale * b_scale

        if EVEN_K:
            a = tl.load(A)
            b = tl.load(B)
        else:
            k_remaining = K - k * (BLOCK_K * SPLIT_K)

            a = tl.load(A, mask=rk[None, :] < k_remaining, other=_0)
            b = tl.load(B, mask=rk[:, None] < k_remaining, other=_0)
        if AB_DTYPE:
            a = a.to(C.dtype.element_ty)
            b = b.to(C.dtype.element_ty)

        acc += tl.dot(a, b, out_dtype=dot_out_dtype, allow_tf32=allow_tf32) * scale
        A += BLOCK_K * SPLIT_K * stride_ak
        B += BLOCK_K * SPLIT_K * stride_bk

    # rematerialize rm and rn to save registers
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = acc.to(C.dtype.element_ty)
    c = C + (rm[:, None] * stride_cm + rn[None, :] * stride_cn)
    mask = (rm < M)[:, None] & (rn < N)[None, :]
    # handles write-back with reduction-splitting
    if SPLIT_K == 1:
        tl.store(c, acc, mask=mask)
    else:
        tl.atomic_add(c, acc, mask=mask)


@torch.library.custom_op("triton::matmul_fp8_block", mutates_args=())
def matmul_fp8_block(
    a: torch.Tensor,
    b: torch.Tensor,
    a_scale: torch.Tensor,
    b_scale: torch.Tensor,
    scale_block_m: int = 256,
    scale_block_n: int = 256,
    scale_block_k: int = 256,
    dot_out_dtype: Optional[torch.dtype] = None,
    allow_tf32: bool = True,
    fp8_fast_accum: bool = True,
) -> Tensor:
    """Performs matmul on [M, K] and [N, K] fp8 matrices with block-wise scalings.

    Args:
        a (torch.Tensor): [M, K] input tensor.
        b (torch.Tensor): [N, K] input tensor.
        a_scale (torch.Tensor): [cdiv(M, scale_block_m), cdiv(K, scale_block_k)] reciprocal scale tensor per scale block. A * A_scale = original A
        b_scale (torch.Tensor): [cdiv(N, scale_block_n), cdiv(K, scale_block_k)] reciprocal scale tensor per scale block. B * B_scale = original B
        scale_block_m (int): Block size for M dimension of A_scale.
        scale_block_n (int): Block size for N dimension of B_scale.
        scale_block_k (int): Block size for K dimension of A_scale and B_scale.
        dot_out_dtype (torch.dtype): Output type of tensor core.
        allow_tf32 (bool): Whether to use TF32 for tensor core.
        fp8_fast_accum (bool): Whether to use fast accumulation for tensor core.

    Returns:
        Tensor: [M, N] output tensor, (a / a_scale) @ (b / b_scale)
    """
    # Get datatypes and constants to use.
    _, tl_fp8_dtype, _, _ = get_fp8_constants()
    # Handle 3D+ a shape
    a_shape = a.shape
    a = a.view(-1, a.size(-1))
    # View inputs into proper triton fp8 dtype.
    a_tl = reinterpret_fp8_type(a, tl_fp8_dtype)
    b_tl = reinterpret_fp8_type(b, tl_fp8_dtype)

    M, N, K, m_key, n_key, k_key, c, _, dot_out_dtype_triton, device = prep_matmul(
        a_tl, b_tl, dot_out_dtype
    )

    output_shape = a_shape[:-1] + (N,)
    # Handle case where inputs are empty.
    if (M == 0) or (N == 0) or (K == 0):
        return torch.zeros(output_shape, device=device, dtype=torch.bfloat16)

    # launch kernel
    assert device != torch.device("cpu"), (
        "Blockwise matmul not supported on cpu, please use row-wise instead."
    )

    if b.device != a.device:
        raise Exception("'b' must be on the same device as 'a'")
    if a_scale.device != a.device:
        raise Exception("'a_scale' must be on the same device as 'a'")
    if b_scale.device != a.device:
        raise Exception("'b_scale' must be on the same device as 'a'")

    # noqa: E731:
    def grid(META: Dict[str, int]) -> Tuple[int, int]:
        return (
            triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]),
            META["SPLIT_K"],
        )

    if fp8_fast_accum:
        with torch.cuda.device(a_tl.device.index):
            _kernel_matmul_fp8_block_fastacc[grid](
                a_tl,
                b_tl,
                c,
                M,
                N,
                K,
                m_key,
                n_key,
                k_key,
                a_scale,
                b_scale,
                scale_block_m,
                scale_block_n,
                scale_block_k,
                a.stride(0),
                a.stride(1),
                b.stride(0),
                b.stride(1),
                c.stride(0),
                c.stride(1),
                a_scale.stride(0),
                a_scale.stride(1),
                b_scale.stride(0),
                b_scale.stride(1),
                dot_out_dtype=dot_out_dtype_triton,
                allow_tf32=allow_tf32,
                GROUP_M=8,
                AB_DTYPE=False,
            )
    else:
        with torch.cuda.device(a_tl.device.index):
            _kernel_matmul_fp8_block_slowacc[grid](
                a_tl,
                b_tl,
                c,
                M,
                N,
                K,
                m_key,
                n_key,
                k_key,
                a_scale,
                b_scale,
                scale_block_m,
                scale_block_n,
                scale_block_k,
                a.stride(0),
                a.stride(1),
                b.stride(0),
                b.stride(1),
                c.stride(0),
                c.stride(1),
                a_scale.stride(0),
                a_scale.stride(1),
                b_scale.stride(0),
                b_scale.stride(1),
                dot_out_dtype=dot_out_dtype_triton,
                allow_tf32=allow_tf32,
                GROUP_M=8,
                AB_DTYPE=False,
            )
    return c.view(output_shape)


@matmul_fp8_block.register_fake
def matmul_fp8_block_meta(
    a: torch.Tensor,
    b: torch.Tensor,
    a_scale: torch.Tensor,
    b_scale: torch.Tensor,
    scale_block_m: int = 256,
    scale_block_n: int = 256,
    scale_block_k: int = 256,
    dot_out_dtype: Optional[torch.dtype] = None,
    allow_tf32: bool = True,
    fp8_fast_accum: bool = True,
) -> torch.Tensor:
    """Shape function for torch compile."""
    M, K = a.shape
    N, K = b.shape
    return torch.empty((M, N), device=a.device, dtype=torch.bfloat16)


def get_matmul_tune(M: int, N: int, K: int) -> Tuple[int, int, int]:
    """
    Generate a simplified matmul tune key for A @ B.T
    with [M, K] A and [N, K] B to reduce excessive autotuning.

    Args:
        M (int): Number of rows in A.
        N (int): Number of rows in B.
        K (int): Number of cols in A and cols in B.

    Returns:
        m_key (int): Autotuning key for M dim.
        n_key (int): Autotuning key for N dim.
        k_key (int): Autotuning key for K dim.

    TODO: Refine this. For now it's useful for LLM inference where N, K dims are fixed
          and M dim varies due to seq_len.
    """
    if M < 256:
        m_key = M
    else:
        m_key = 256 + M // 1024
    return m_key, N, K


def prep_matmul(
    a: Union[TensorWrapper, torch.Tensor],
    b: Union[TensorWrapper, torch.Tensor],
    dot_out_dtype: Optional[torch.dtype],
) -> Tuple[
    int, int, int, int, int, int, torch.Tensor, tl.dtype, tl.dtype, torch.device
]:
    """
    Shared bookkeeping for a @ b.T matmul.

    Args:
        a (torch.Tensor): [M, K] input tensor.
        b (torch.Tensor): [N, K] input tensor.
        dot_out_dtype (tl.dtype): Output type of tensor core.

    Returns:
        M (int): Number of rows in A.
        N (int): Number of rows in B.
        K (int): Number of cols in A and cols in B.
        m_key (int): Autotuning key for M dim.
        n_key (int): Autotuning key for N dim.
        k_key (int): Autotuning key for K dim.
        c (Tensor): [M, N] output tensor.
        c_dtype_triton (tl.dtype): Type of output tensor.
        dot_out_dtype (tl.dtype): Output type of tensor core.
        device (torch.device): Device of output tensor.
    """
    device = a.device

    # checks constraints
    assert a.shape[1] == b.shape[1], (
        f"incompatible dimensions, a: {a.shape}, b: {b.shape}"
    )
    M, K = a.shape
    N, _ = b.shape
    m_key, n_key, k_key = get_matmul_tune(M, N, K)

    # allocates output
    assert a.dtype in [
        torch.float8_e4m3fn,
        torch.float8_e5m2,
        torch.float8_e4m3fnuz,
        torch.float8_e5m2fnuz,
        tl.float8e4nv,
        tl.float8e4b15,
        tl.float8e5,
        tl.float8e4b8,
    ]
    assert b.dtype in [
        torch.float8_e4m3fn,
        torch.float8_e5m2,
        torch.float8_e4m3fnuz,
        torch.float8_e5m2fnuz,
        tl.float8e4nv,
        tl.float8e4b15,
        tl.float8e5,
        tl.float8e4b8,
    ]

    c_dtype, c_dtype_triton = (
        (torch.bfloat16, tl.bfloat16)
        if dot_out_dtype is None
        else (dot_out_dtype, map_dtype_to_triton(dot_out_dtype))
    )

    c = torch.empty((M, N), device=device, dtype=c_dtype)
    if dot_out_dtype is None:
        dot_out_dtype_triton = tl.float32
    else:
        assert isinstance(dot_out_dtype, torch.dtype), (
            f"dot_out_dtype type {type(dot_out_dtype)} must be a torch.dtype"
        )
        dot_out_dtype_triton = map_dtype_to_triton(dot_out_dtype)

    return M, N, K, m_key, n_key, k_key, c, c_dtype_triton, dot_out_dtype_triton, device


def need_split_k(SIZE_M, SIZE_N, SIZE_K):
    return (SIZE_M < 64 or SIZE_N < 64) and SIZE_K > 1024


# Force a failure instead of a warning when all configs are pruned.
# TODO: Determine a better approach for model level testing. We need
# to standardize our approach around prune_configs in general.
FORCE_FAILURE_ON_EMPTY_CONFIGS = False


def is_invalid_config(config, N, M, K, mfma, use_bias):
    """
    Contains all of the configuration checks for prune_configs
    that will result in an invalid result if select as the config.

    This is done to ensure that if no config is "optimal" for a given
    shape we don't accidentally select
    """
    BLOCK_SIZE_M = config.kwargs.get("BLOCK_M")
    BLOCK_SIZE_N = config.kwargs.get("BLOCK_N")
    BLOCK_SIZE_K = config.kwargs.get("BLOCK_K")
    SPLIT_K = config.kwargs.get("SPLIT_K")
    matrix_instr_nonkdim = config.kwargs.get("matrix_instr_nonkdim")
    if matrix_instr_nonkdim > mfma:
        return True
    if mfma == 4 and BLOCK_SIZE_K < 64:
        return True
    # some layouts could not work properly in case
    # number elements per thread is less 1
    if BLOCK_SIZE_M * BLOCK_SIZE_N < 64:
        return True
    if BLOCK_SIZE_M < matrix_instr_nonkdim or BLOCK_SIZE_N < matrix_instr_nonkdim:
        return True
    if M <= matrix_instr_nonkdim and BLOCK_SIZE_M != matrix_instr_nonkdim:
        return True
    if N <= matrix_instr_nonkdim and BLOCK_SIZE_N != matrix_instr_nonkdim:
        return True
    # split_k cannot be used if there is a bias
    if use_bias and SPLIT_K != 1:
        return True
    return False


# Configs adapted from https://github.com/ROCm/triton/blob/main_perf/python/perf-kernels/tools/tune_gemm/tune_gemm.py
def prune_configs(configs, named_args, **kwargs):
    pruned_configs = []
    M = named_args["M"]
    N = named_args["N"]
    K = named_args["K"]
    elemBytes_a = named_args["A"].element_size()
    elemBytes_b = named_args["B"].element_size()
    use_bias = kwargs["USE_BIAS"]

    if M < 32 or N < 32:
        mfma = 16
    else:
        mfma = 32

    for config in configs:
        BLOCK_SIZE_M = config.kwargs.get("BLOCK_M")
        BLOCK_SIZE_N = config.kwargs.get("BLOCK_N")
        BLOCK_SIZE_K = config.kwargs.get("BLOCK_K")
        SPLIT_K = config.kwargs.get("SPLIT_K")
        GROUP_M = config.kwargs.get("GROUP_M")
        if is_invalid_config(config, N, M, K, mfma, use_bias):
            continue
        # Skip BLOCK_SIZE that is too large compare to M/N
        # unless BLOCK_SIZE is already small enough
        if BLOCK_SIZE_M > M * 2 and BLOCK_SIZE_M != 16:
            continue
        if BLOCK_SIZE_N > N * 2 and BLOCK_SIZE_N != 16:
            continue
        # skip large split_k when not necessary
        if SPLIT_K != 1 and not need_split_k(M, N, K):
            continue
        # skip large GROUP_M
        if GROUP_M * BLOCK_SIZE_M >= M and GROUP_M != 1:
            continue
        # out of shared memory resource
        # TODO (zhanglx): This does not consider the LDS usage in the epilogue
        LDS = (
            BLOCK_SIZE_K * BLOCK_SIZE_M * elemBytes_a
            + BLOCK_SIZE_K * BLOCK_SIZE_N * elemBytes_b
        )
        if LDS > 65536:
            continue
        pruned_configs.append(config)

    print(f"{len(configs)=} {len(pruned_configs)=} for {M=} {N=} {K=}")
    if len(pruned_configs) == 0:
        if not FORCE_FAILURE_ON_EMPTY_CONFIGS:
            # Prune configs that can lead to incorrect results even if all configs are sub-optimal.
            candidate_configs = [
                c for c in configs if not is_invalid_config(c, N, M, K, mfma, use_bias)
            ]
            print(f"No configs left after pruning! {M=} {N=} {K=}")
            pruned_configs = candidate_configs[:10]
        if len(pruned_configs) == 0:
            raise RuntimeError(
                "No valid configs left after pruning! Consider autotuning further with TritonBench"
            )
    return pruned_configs


def get_full_non_persistent_tuning_space():
    configs = []

    block_mn_range = [16, 32, 64, 128, 256]
    block_k_range = [16, 32, 64, 128, 256]
    split_k_range = [1]
    num_warps_range = [1, 2, 4, 8]
    group_m_range = [1, 2, 4, 8, 16, 32]
    num_stage_range = [2]
    waves_per_eu_range = [0]
    matrix_instr_nonkdim_range = [16, 32]
    kpack_range = [1, 2]

    for block_m in block_mn_range:
        for block_n in block_mn_range:
            for block_k in block_k_range:
                for num_warps in num_warps_range:
                    for group_m in group_m_range:
                        for split_k in split_k_range:
                            for num_stages in num_stage_range:
                                for waves_per_eu in waves_per_eu_range:
                                    for (
                                        matrix_instr_nonkdim
                                    ) in matrix_instr_nonkdim_range:
                                        for kpack in kpack_range:
                                            configs.append(
                                                triton.Config(
                                                    {
                                                        "BLOCK_M": block_m,
                                                        "BLOCK_N": block_n,
                                                        "BLOCK_K": block_k,
                                                        "GROUP_M": group_m,
                                                        "SPLIT_K": split_k,
                                                        "waves_per_eu": waves_per_eu,
                                                        "matrix_instr_nonkdim": matrix_instr_nonkdim,
                                                        "kpack": kpack,
                                                    },
                                                    num_warps=num_warps,
                                                    num_stages=num_stages,
                                                )
                                            )
    return configs


MATMUL_CONFIGS_NON_PERSISTENT: List[Config] = get_full_non_persistent_tuning_space()
# (BLOCK_M, BLOCK_N, BLOCK_K, GROUP_M, SPLIT_K, waves_per_eu, matrix_instr_nonkdim, kpack, num_warps, num_stages)
_MATMUL_CONFIG_TUPLES_PINGPONG_4K_8K_16K = [
    (16, 16, 256, 1, 1, 8, 16, 2, 2, 2),
    (16, 16, 256, 1, 1, 0, 16, 2, 2, 2),
    (32, 64, 512, 1, 1, 2, 16, 2, 8, 2),
    (64, 64, 256, 1, 1, 2, 16, 2, 4, 2),
    (256, 256, 128, 32, 1, 2, 16, 1, 8, 2),
    (256, 256, 128, 2, 1, 0, 32, 2, 8, 2),
    (256, 256, 128, 1, 1, 0, 32, 2, 8, 2),
    (256, 256, 128, 2, 1, 0, 16, 1, 8, 2),
    (256, 256, 64, 2, 1, 2, 16, 1, 8, 2),
    (128, 256, 64, 2, 1, 2, 16, 1, 4, 2),
    (256, 128, 128, 4, 1, 0, 16, 1, 8, 2),
    (128, 128, 128, 1, 1, 2, 16, 2, 4, 2),
    (128, 128, 256, 1, 1, 2, 16, 2, 8, 2),
    (128, 128, 64, 4, 1, 2, 16, 2, 4, 2),
    (128, 128, 64, 1, 1, 2, 16, 2, 4, 2),
    (128, 64, 64, 4, 1, 0, 16, 2, 4, 2),
    (128, 64, 64, 1, 1, 0, 16, 2, 4, 2),
    (256, 128, 128, 1, 1, 2, 16, 1, 8, 2),
    (128, 256, 128, 2, 1, 2, 16, 2, 4, 1),
    (256, 128, 64, 2, 1, 2, 16, 1, 4, 2),
    (128, 128, 256, 2, 1, 0, 16, 2, 8, 2),
    (128, 64, 128, 2, 1, 2, 16, 2, 4, 2),
    (128, 128, 64, 2, 1, 0, 16, 1, 4, 2),
    (128, 128, 128, 1, 1, 2, 16, 1, 4, 2),
]


def _should_skip_config(block_k, matrix_instr_nonkdim):
    """Skip config if BLOCK_K=64 and matrix_instr_nonkdim=16 on GFX95+"""
    try:
        return (
            block_k == 64
            and matrix_instr_nonkdim == 16
            and torch.version.hip is not None
            and torch.cuda.get_device_capability() >= (9, 5)
        )
    except RuntimeError:
        # If no HIP GPUs are available, we can't check device capability
        # so we don't skip any configs
        return False


MATMUL_CONFIGS_NON_PERSISTENT_PINGPONG_4K_8K_16K = [
    triton.Config(
        {
            "BLOCK_M": block_m,
            "BLOCK_N": block_n,
            "BLOCK_K": block_k,
            "GROUP_M": group_m,
            "SPLIT_K": split_k,
            "waves_per_eu": waves_per_eu,
            "matrix_instr_nonkdim": matrix_instr_nonkdim,
            "kpack": kpack,
        },
        num_warps=num_warps,
        num_stages=num_stages,
    )
    for block_m, block_n, block_k, group_m, split_k, waves_per_eu, matrix_instr_nonkdim, kpack, num_warps, num_stages in _MATMUL_CONFIG_TUPLES_PINGPONG_4K_8K_16K
    if not _should_skip_config(block_k, matrix_instr_nonkdim)
]

# Set this to enable full autotuning for proper benchmarking.
# This should only be used when invoking the kernel through
# Triton directly (e.g. TritonBench)
#
# NOTE: This will SIGNIFICANTLY increase autotuning time, often
# taking hours. You should combine this with TRITON_PRINT_AUTOTUNING=1
# to extract and add the optimal autotuning configs to
# MATMUL_CONFIGS_NON_PERSISTENT_PINGPONG_4K_8K_16K.

FULL_NON_PERSISTENT_AUTOTUNING = False
USED_MATMUL_NON_PERSISTENT_CONFIGS = (
    MATMUL_CONFIGS_NON_PERSISTENT
    if FULL_NON_PERSISTENT_AUTOTUNING
    else MATMUL_CONFIGS_NON_PERSISTENT_PINGPONG_4K_8K_16K
)


@triton.autotune(
    configs=USED_MATMUL_NON_PERSISTENT_CONFIGS,
    key=["M", "N", "K"],
    prune_configs_by={
        "early_config_prune": prune_configs,
        "perf_model": None,
        "top_k": None,
    },
    use_cuda_graph=FULL_NON_PERSISTENT_AUTOTUNING,
)
@triton.heuristics(
    {
        "EVEN_K": lambda args: args["K"] % (args["BLOCK_K"] * args["SPLIT_K"]) == 0,
    }
)
@triton.jit
def _kernel_matmul_fp8_row_non_persistent(
    A,
    B,
    C,
    M,
    N,
    K,
    m_key,
    n_key,
    k_key,
    A_scale,
    B_scale,
    Bias,
    stride_am,
    stride_ak,
    stride_bn,
    stride_bk,
    stride_cm,
    stride_cn,
    dot_out_dtype: tl.constexpr,
    allow_tf32: tl.constexpr,
    fp8_fast_accum: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    SPLIT_K: tl.constexpr,
    EVEN_K: tl.constexpr,
    USE_BIAS: tl.constexpr,
    AB_DTYPE: tl.constexpr,
) -> None:
    """Matmul kernel of [M, K] @ [N, K] with row-wise scales

    performs swizzled matmul in [BLOCK_M, BLOCK_K] with [BLOCK_K, BLOCK_N] tiles.

    Args:
        A (TensorWrapper): [M, K] input tensor.
        B (TensorWrapper): [N, K] input tensor.
        C (TensorWrapper): [M, N] output tensor.
        M (int): M dimension of input tensor.
        N (int): N dimension of input tensor.
        K (int): K dimension of input tensor.
        m_key (int): Autotuning key for M dimension of input tensor.
        n_key (int): Autotuning key for N dimension of input tensor.
        k_key (int): Autotuning key for K dimension of input tensor.
        A_scale (TensorWrapper): [M] reciprocal scale tensor per row. A * A_scale = original A
        B_scale (TensorWrapper): [N] reciprocal scale tensor per row. B * B_scale = original B
        Bias (tensorWrapper): [N] Optional bias tensor.
        stride_am (int): Stride of M dimension of A.
        stride_ak (int): Stride of K dimension of A.
        stride_bn (int): Stride of N dimension of B.
        stride_bk (int): Stride of K dimension of B.
        stride_cm (int): Stride of M dimension of C.
        stride_cn (int): Stride of N dimension of C.
        dot_out_dtype (torch.dtype): Output type of tensor core.
        allow_tf32 (bool): Whether to use TF32 for tensor core.
        fp8_fast_accum (bool): Whether to use fast accumulation for tensor core.
        BLOCK_M (int): Block size for M dimension.
        BLOCK_N (int): Block size for N dimension.
        BLOCK_K (int): Block size for K dimension.
        GROUP_M (int): Number of groups for M dimension swizzle.
        SPLIT_K (int): Number of SM's to launch per row.
        EVEN_K (bool): Whether K is evenly divisible by BLOCK_K * SPLIT_K.
        USE_BIAS (bool): Whether to use bias.
        AB_DTYPE (bool): Whether to cast A and B to C.dtype before tensor core.
    """
    tl.assume(M >= 0)
    tl.assume(N >= 0)
    tl.assume(K >= 0)
    tl.assume(stride_am >= 0)
    tl.assume(stride_ak >= 0)
    tl.assume(stride_bn >= 0)
    tl.assume(stride_bk >= 0)
    tl.assume(stride_cm >= 0)
    tl.assume(stride_cn >= 0)
    # Matrix multiplication.
    pid = tl.program_id(0)
    pid_z = tl.program_id(1)
    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)
    # Re-order program ID for better L2 performance (swizzle).
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + ((pid % width) % group_size)
    pid_n = (pid % width) // (group_size)
    tl.assume(pid_m >= 0)
    tl.assume(pid_n >= 0)
    # Do matrix multiplication.
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    ram = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M)
    rbn = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N)
    rk = pid_z * BLOCK_K + tl.arange(0, BLOCK_K)
    # Pointers.
    A = A + (ram[:, None] * stride_am + rk[None, :] * stride_ak)
    B = B + (rk[:, None] * stride_bk + rbn[None, :] * stride_bn)
    acc_dtype = tl.float32 if allow_tf32 else dot_out_dtype
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=acc_dtype)

    for k in range(0, tl.cdiv(K, BLOCK_K * SPLIT_K)):
        if EVEN_K:
            a = tl.load(A)
            b = tl.load(B)
        else:
            k_remaining = K - k * (BLOCK_K * SPLIT_K)
            _0 = tl.zeros((1, 1), dtype=C.dtype.element_ty)
            a = tl.load(A, mask=rk[None, :] < k_remaining, other=_0)
            b = tl.load(B, mask=rk[:, None] < k_remaining, other=_0)
        if AB_DTYPE:
            a = a.to(C.dtype.element_ty)
            b = b.to(C.dtype.element_ty)
        if fp8_fast_accum:
            acc = tl.dot(a, b, acc, out_dtype=acc_dtype, allow_tf32=allow_tf32)
        else:
            acc += tl.dot(a, b, out_dtype=acc_dtype, allow_tf32=allow_tf32)

        A += BLOCK_K * SPLIT_K * stride_ak
        B += BLOCK_K * SPLIT_K * stride_bk

    # rematerialize rm and rn to save registers
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Invert scaling.
    a_scale = tl.load(A_scale + rm, mask=rm < M)
    b_scale = tl.load(B_scale + rn, mask=rn < N)
    # Invert vector, then multiply on matrix for speed.
    # pyre-ignore[16]: Undefined attribute [16]: `float` has no attribute `__getitem__`.
    scale = a_scale[:, None] * b_scale[None, :]
    acc *= scale

    # Load and add bias if specified.
    if USE_BIAS:
        bias = tl.load(Bias + rn, mask=rn < N)
        acc += bias[None, :]

    acc = acc.to(C.dtype.element_ty)
    C = C + (rm[:, None] * stride_cm + rn[None, :] * stride_cn)
    mask = (rm < M)[:, None] & (rn < N)[None, :]
    # Handles write-back with reduction-splitting
    if SPLIT_K == 1:
        tl.store(C, acc, mask=mask)
    else:
        tl.atomic_add(C, acc, mask=mask)


# This function is extracted from https://github.com/pytorch/ao/blob/v0.12.0/torchao/prototype/mx_formats/mx_tensor.py#L142
def to_mxfp8(
    data_hp: torch.Tensor,
    block_size: int = 32,
):
    assert data_hp.dtype in (
        torch.bfloat16,
        torch.float,
    ), f"{data_hp.dtype} is not supported yet"
    assert data_hp.shape[-1] % block_size == 0, (
        f"the last dimension of shape {data_hp.shape} must be divisible by block_size {block_size}"
    )
    assert data_hp.is_contiguous(), "unsupported"

    orig_shape = data_hp.shape
    data_hp = data_hp.reshape(
        *orig_shape[:-1], orig_shape[-1] // block_size, block_size
    )

    max_abs = torch.amax(torch.abs(data_hp), -1).unsqueeze(-1)

    data_hp = data_hp.to(torch.float32)
    max_abs = max_abs.to(torch.float32)

    F8E4M3_MAX = torch.finfo(torch.float8_e4m3fn).max  # 448.0
    max_pos = F8E4M3_MAX

    # RCEIL
    def _to_mx_rceil(
        data_hp: torch.Tensor,
        max_abs: torch.Tensor,
        max_pos: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        E8M0_EXPONENT_BIAS = 127
        descale = max_abs / max_pos
        exponent = torch.where(
            torch.isnan(descale),
            0xFF,  # Handle biased exponent for nan
            # NOTE: descale < (torch.finfo(torch.float32).smallest_normal / 2) is handled through clamping
            (
                torch.clamp(
                    torch.ceil(torch.log2(descale)),
                    min=-E8M0_EXPONENT_BIAS,
                    max=E8M0_EXPONENT_BIAS,
                )
                + E8M0_EXPONENT_BIAS
            ).to(torch.uint8),
        )

        descale_fp = torch.where(
            exponent == 0,
            1.0,
            torch.exp2(E8M0_EXPONENT_BIAS - exponent.to(torch.float32)),
        )

        # scale and saturated cast the data elements to max of target dtype
        data_lp = torch.clamp(data_hp * descale_fp, min=-1 * max_pos, max=max_pos)
        return exponent, data_lp

    scale_e8m0_biased, data_lp = _to_mx_rceil(data_hp, max_abs, max_pos)

    # cast to target dtype
    data_lp = data_lp.to(torch.float8_e4m3fn)
    # need to reshape at the end to help inductor fuse things
    data_lp = data_lp.reshape(orig_shape)

    scale_e8m0_biased = scale_e8m0_biased.view(torch.float8_e8m0fnu)
    scale_e8m0_biased = scale_e8m0_biased.squeeze(-1)
    return scale_e8m0_biased, data_lp
