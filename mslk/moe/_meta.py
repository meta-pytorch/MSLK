# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Python-side Meta (shape inference) implementations for MOE ops.

These were previously registered in C++ (csrc/moe/moe_ops.cpp) but are moved
here to decouple Meta dispatch from the C++ ABI.

Platform-specific ops have their C++ schema conditionally compiled, so we
guard their Meta registrations with a `hasattr(torch.ops.mslk, ...)` check on
a representative op. If the schema was not registered for this build, the
check is False and we skip the block.
"""

from typing import Optional

import torch


# ---------------------------------------------------------------------------
# Common ops (always defined, all platforms)
# ---------------------------------------------------------------------------


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
    opts_int = {"dtype": torch.int32, "device": routing_scores.device}
    token_counts_per_expert = torch.empty(E + 2, **opts_int)
    expert_indices = torch.empty(T * top_k, **opts_int)
    token_indices = torch.empty(T * top_k, **opts_int)
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
