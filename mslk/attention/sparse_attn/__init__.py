# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

from mslk.attention.sparse_attn.compress import compress_kv, fused_compress_kv
from mslk.attention.sparse_attn.gating import (
    compute_gates,
    fused_gate_and_combine,
    gate_and_combine,
)
from mslk.attention.sparse_attn.nsa_autograd import nsa
from mslk.attention.sparse_attn.nsa_forward import nsa_forward
from mslk.attention.sparse_attn.reference import (
    nsa_backward_reference,
    nsa_forward_reference,
)
from mslk.attention.sparse_attn.select import (
    fused_score_and_select_blocks,
    score_and_select_blocks,
    select_compressed_blocks,
)
from mslk.attention.sparse_attn.sparsity_masks import (
    build_compressed_block_sparse_tensors,
    build_fa4_block_sparse_tensors,
)

__all__ = [
    "nsa",
    "nsa_forward",
    "nsa_forward_reference",
    "nsa_backward_reference",
    "compress_kv",
    "fused_compress_kv",
    "score_and_select_blocks",
    "fused_score_and_select_blocks",
    "select_compressed_blocks",
    "build_fa4_block_sparse_tensors",
    "build_compressed_block_sparse_tensors",
    "compute_gates",
    "fused_gate_and_combine",
    "gate_and_combine",
]
