# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""Layout utilities: dense KV <-> FlyDSL kernel API adaptation.

Kernel expects 5D BMGHK: Q [B, 1, G, H_q, D], K/V [B, KVMAX, G, H_kv, D]
(H_kv may = 1 for MQA), seq [B] int32.
"""

from typing import Optional, Tuple

import torch


def canonicalize_qkv_5d(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (Q, K, V) in [B, *, G, H, D] 5D form (4D BMHK promoted with G=1)."""
    if Q.ndim == 4:
        Q = Q.unsqueeze(2)
    if K.ndim == 4:
        K = K.unsqueeze(2)
    if V.ndim == 4:
        V = V.unsqueeze(2)

    assert Q.ndim == 5 and K.ndim == 5 and V.ndim == 5, (
        f"Expected 5D tensors after promotion; got Q={Q.shape}, K={K.shape}"
    )

    # Multiquery (stride-0 H) is handled by the kernel from .stride() directly.
    return Q, K, V


def normalize_seq_positions(
    seq_kv_lens: Optional[torch.Tensor],
    B: int,
    KV_MAX: int,
    device: torch.device,
) -> torch.Tensor:
    """Return a [B] int32 tensor of valid KV lengths.

    If ``seq_kv_lens`` is None, all entries are set to ``KV_MAX``.
    """
    if seq_kv_lens is None:
        return torch.full((B,), KV_MAX, dtype=torch.int32, device=device)
    if seq_kv_lens.dtype != torch.int32:
        seq_kv_lens = seq_kv_lens.to(torch.int32)
    if seq_kv_lens.device != device:
        seq_kv_lens = seq_kv_lens.to(device)
    return seq_kv_lens.contiguous()


def get_split_k_heuristic(B: int, H: int, Mk: int) -> int:
    """Mirror of flydsl_splitk.FwOp.get_split_k — used as default split count."""
    bh = max(B * H, 1)
    split_k = max(Mk, 1024) // bh
    max_chunk_size = 64 if Mk <= 512 and bh <= 64 else 128
    while split_k > 0 and Mk / split_k < max_chunk_size:
        split_k = split_k // 2
    split_k = min(split_k, 64)
    split_k = max(split_k, 1)
    return split_k
