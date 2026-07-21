# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

"""
BF16 x INT4 grouped GEMM for ROCm/AMD GPUs.

Delegates to matmul_bf16i4_rowwise from int4_gemm.py for each group.
On ROCm, bf16i4bf16_shuffled_grouped routes through this path.

Weight layout:
  WQ          : [G, N, K//2]        int8
  w_scale_group: [G, num_groups, N]  float32 or bfloat16
  w_zero_group : [G, num_groups, N]  float32 or bfloat16
  M_sizes      : [G]                 int32 or int64 -- rows per group

Output: [M_total, N] bfloat16, where M_total = sum(M_sizes).
"""

import torch
from mslk.gemm.triton.int4_gemm import matmul_bf16i4_rowwise
from mslk.gemm.triton.int4_grouped_gemm_fused import matmul_bf16i4_rowwise_grouped_fused

# Fused kernel wins for small M-per-group (latency-bound regime) where Python
# loop overhead dominates.  For larger M the kernel is bandwidth-bound and the
# stride-2 X load in the fused path costs more than the Python overhead saved.
_FUSED_M_THRESHOLD = 256


def matmul_bf16i4_rowwise_grouped(
    X: torch.Tensor,
    WQ: torch.Tensor,
    w_scale_group: torch.Tensor,
    w_zero_group: torch.Tensor,
    M_sizes: torch.Tensor,
) -> torch.Tensor:
    """
    Grouped BF16 x INT4 GEMM.

    Routes to a native fused kernel for small M-per-group (eliminates
    Python-loop overhead) or a Python loop over the rowwise kernel for
    large M-per-group (avoids stride-2 X load bandwidth penalty).

    Args:
        X             : [M_total, K]          bfloat16 activations (rows packed)
        WQ            : [G, N, K//2]          int8 packed weights per group
        w_scale_group : [G, num_groups, N]    per-group scales
        w_zero_group  : [G, num_groups, N]    per-group zero points
        M_sizes       : [G]                   rows per group

    Returns:
        Y : [M_total, N]  bfloat16
    """
    G = WQ.shape[0]

    # Use the maximum group size to decide routing — M_sizes may be unequal
    # (e.g. MoE with variable token counts per expert), and the fused kernel's
    # stride-2 X load only regresses when the *largest* group is bandwidth-bound.
    m_max = int(M_sizes.max().item())

    if m_max < _FUSED_M_THRESHOLD:
        return matmul_bf16i4_rowwise_grouped_fused(
            X, WQ, w_scale_group, w_zero_group, M_sizes
        )

    m_sizes_cpu = M_sizes.tolist()
    x_splits = torch.split(X, m_sizes_cpu, dim=0)
    outs = []
    for g in range(G):
        outs.append(
            matmul_bf16i4_rowwise(
                x_splits[g],
                WQ[g],
                w_scale_group[g],
                w_zero_group[g],
            )
        )
    return torch.cat(outs, dim=0)


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
