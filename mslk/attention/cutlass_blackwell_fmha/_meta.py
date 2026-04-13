# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Meta (shape inference) implementations for Blackwell FMHA C++ ops.

These were previously registered in C++ (csrc/attention/cuda/cutlass_blackwell_fmha/)
but are moved here to decouple Meta dispatch from the C++ ABI.

Note: The wrapper ops (mslk::cutlass_blackwell_fmha_fwd/bwd) already have Python
Meta impls in cutlass_blackwell_fmha_custom_op.py. This file covers the lower-level
C++ ops (mslk::fmha_fwd, mslk::fmha_bwd, mslk::fmha_gen_fwd).

Platform-specific ops have their C++ schema conditionally compiled, so we
guard each Meta registration with its own `hasattr(torch.ops.mslk, ...)`
check. If the schema was not registered for this build, we skip that op.
"""

from typing import Optional

import torch


# ---------------------------------------------------------------------------
# Blackwell-only ops (require SM100+ native support)
# ---------------------------------------------------------------------------
if hasattr(torch.ops.mslk, "fmha_fwd"):

    @torch.library.register_fake("mslk::fmha_fwd")
    def fmha_fwd_meta(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        cu_seqlens_q: Optional[torch.Tensor] = None,
        cu_seqlens_k: Optional[torch.Tensor] = None,
        max_seq_len_q: Optional[int] = None,
        max_seq_len_k: Optional[int] = None,
        softmax_scale: Optional[float] = None,
        causal: bool = False,
        seqlen_kv: Optional[torch.Tensor] = None,
        page_table: Optional[torch.Tensor] = None,
        seqlen_k: Optional[int] = None,
        window_size_left: int = -1,
        window_size_right: int = -1,
        bottom_right: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        output = torch.empty_like(query)
        k_is_varlen = max_seq_len_q is not None
        if k_is_varlen:
            SQ = query.shape[0]
            H_Q = query.shape[1]
            B = 1
        else:
            SQ = query.shape[1]
            H_Q = query.shape[2]
            B = query.shape[0]
        logsumexp = torch.empty(
            (B, H_Q, SQ), dtype=torch.float32, device=query.device
        )
        return output, logsumexp


if hasattr(torch.ops.mslk, "fmha_bwd"):

    @torch.library.register_fake("mslk::fmha_bwd")
    def fmha_bwd_meta(
        dOutput: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        output: torch.Tensor,
        softmax_lse: torch.Tensor,
        cu_seqlens_q: Optional[torch.Tensor] = None,
        cu_seqlens_k: Optional[torch.Tensor] = None,
        max_seq_len_q: Optional[int] = None,
        max_seq_len_k: Optional[int] = None,
        softmax_scale: Optional[float] = None,
        causal: bool = False,
        window_size_left: int = -1,
        window_size_right: int = -1,
        bottom_right: bool = True,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            torch.empty_like(query),
            torch.empty_like(key),
            torch.empty_like(value),
        )


if hasattr(torch.ops.mslk, "fmha_gen_fwd"):

    @torch.library.register_fake("mslk::fmha_gen_fwd")
    def fmha_gen_fwd_meta(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        seqlen_kv: torch.Tensor,
        batch_idx: Optional[torch.Tensor] = None,
        kernel_type: int = 0,
        window_left: int = -1,
        window_right: int = -1,
        split_k_size: int = 1024,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        output = torch.empty_like(query)
        b = query.shape[0]
        h = query.shape[2]
        # For meta, create a dummy LSE with single split
        lse = torch.empty((b, 1, h), dtype=torch.float32, device=query.device)
        return output, lse
