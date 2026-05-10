# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Python-side Meta (shape inference) implementations for MOE ops.

These were previously registered in C++ (csrc/moe/moe_ops.cpp) but are moved
here to decouple Meta dispatch from the C++ ABI.

Each Meta registration is guarded by its own `hasattr(torch.ops.mslk, ...)`
check. The C++ schema may be missing for several reasons: the op is
platform-specific and was conditionally compiled out, the op lives in a
separate Buck target whose `.so` failed to load, or the consolidated
`mslk.so` failed to load silently in the consuming environment. In any of
those cases we want this module to still import cleanly rather than blowing
up with `RuntimeError: operator mslk::<name> does not exist`.
"""

from typing import Optional

import torch


# ---------------------------------------------------------------------------
# Common ops (defined for all platforms when the moe schema loads)
# ---------------------------------------------------------------------------

if hasattr(torch.ops.mslk, "index_shuffling"):

    @torch.library.register_fake("mslk::index_shuffling")
    def index_shuffling_meta(
        routing_scores: torch.Tensor,
        expert_index_start: Optional[int] = None,
        expert_index_end: Optional[int] = None,
        valid_token_count: Optional[torch.Tensor] = None,
        top_k: int = 1,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        T = routing_scores.shape[0]
        E = routing_scores.shape[1]
        device = routing_scores.device
        token_counts_per_expert = torch.empty(E + 2, dtype=torch.int32, device=device)
        expert_indices = torch.empty(T * top_k, dtype=torch.int32, device=device)
        token_indices = torch.empty(T * top_k, dtype=torch.int32, device=device)
        return token_counts_per_expert, expert_indices, token_indices


# ---------------------------------------------------------------------------
# CUDA-only ops (not defined on ROCm builds)
# ---------------------------------------------------------------------------
if hasattr(torch.ops.mslk, "scatter_add_along_first_dim"):

    @torch.library.register_fake("mslk::scatter_add_along_first_dim")
    def scatter_add_along_first_dim_meta(
        dst: torch.Tensor,
        src: torch.Tensor,
        index: torch.Tensor,
    ) -> None:
        pass
