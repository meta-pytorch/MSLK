# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Python-side Meta (shape inference) implementations for MOE ops.

These were previously registered in C++ (csrc/moe/moe_ops.cpp) but are moved
here to decouple Meta dispatch from the C++ ABI.
"""

from typing import Optional

import torch


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


try:

    @torch.library.register_fake("mslk::scatter_add_along_first_dim")
    def scatter_add_along_first_dim_meta(
        dst: torch.Tensor,
        src: torch.Tensor,
        index: torch.Tensor,
    ) -> None:
        pass

except Exception:
    # This op is only defined on CUDA (non-ROCm) builds
    pass
