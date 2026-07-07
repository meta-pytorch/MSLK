# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

"""
BF16 x INT4 grouped GEMM for ROCm/AMD GPUs.

Delegates to the fused grouped kernel in int4_grouped_gemm_fused.py, which
handles all G groups in a single launch with no device-to-host sync
(CUDA-graph safe).  On ROCm, bf16i4bf16_shuffled_grouped routes through
this path.

Weight layout:
  WQ          : [G, N, K//2]        int8
  w_scale_group: [G, num_groups, N]  float32 or bfloat16
  w_zero_group : [G, num_groups, N]  float32 or bfloat16
  M_sizes      : [G]                 int32 or int64 -- rows per group

Output: [M_total, N] bfloat16, where M_total = sum(M_sizes).
"""

import torch
from mslk.gemm.triton.int4_grouped_gemm_fused import matmul_bf16i4_rowwise_grouped_fused


def matmul_bf16i4_rowwise_grouped(
    X: torch.Tensor,
    WQ: torch.Tensor,
    w_scale_group: torch.Tensor,
    w_zero_group: torch.Tensor,
    M_sizes: torch.Tensor,
) -> torch.Tensor:
    """
    Grouped BF16 x INT4 GEMM — CUDA-graph safe.

    Args:
        X             : [M_total, K]          bfloat16 activations (rows packed)
        WQ            : [G, N, K//2]          int8 packed weights per group
        w_scale_group : [G, num_groups, N]    per-group scales
        w_zero_group  : [G, num_groups, N]    per-group zero points
        M_sizes       : [G]                   rows per group

    Returns:
        Y : [M_total, N]  bfloat16
    """
    return matmul_bf16i4_rowwise_grouped_fused(
        X, WQ, w_scale_group, w_zero_group, M_sizes
    )


# Register as ROCm implementation of mslk::bf16i4bf16_shuffled_grouped.
if torch.version.hip is not None and hasattr(torch.ops, "mslk"):
    if hasattr(torch.ops.mslk, "bf16i4bf16_shuffled_grouped"):

        @torch.library.impl("mslk::bf16i4bf16_shuffled_grouped", "CUDA")
        def _bf16i4bf16_shuffled_grouped_rocm(
            X: torch.Tensor,
            WQ: torch.Tensor,
            w_scale_group: torch.Tensor,
            w_zero_group: torch.Tensor,
            M_sizes: torch.Tensor,
        ) -> torch.Tensor:
            return matmul_bf16i4_rowwise_grouped(
                X, WQ, w_scale_group, w_zero_group, M_sizes
            )
