# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""paged_attention_decode_fp8 against an fp32 reference over the dequantized
cache: ragged lengths, padded block tables and CUDA graphs."""

import pytest
import torch
from mslk.attention.flydsl.decode.fp8_paged_cache import (
    dequantize_fp8_paged_cache,
    random_fp8_paged_cache,
)
from mslk.attention.flydsl.decode.pa_decode_fp8_dispatch import (
    is_fp8_paged_decode_available,
    paged_attention_decode_fp8,
)
from paged_reference import paged_reference

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not is_fp8_paged_decode_available(),
    reason="requires a gfx950 GPU with FlyDSL",
)

# The kernels also round the query and the softmax probabilities to FP8, which
# puts a few elements up to 20% of the output scale off. Outputs are O(1) (see
# random_fp8_paged_cache), so a wrong head, page or length still fails by several
# times this tolerance, except for a few missing tokens of a long sequence.
ATOL = RTOL = 0.3
# The rounding is unbiased, so per sequence the output's least-squares scale
# against the reference stays within 1%, and its relative L2 error within 5%.
# These catch K or V scale errors of a few percent, which ATOL lets through.
MAX_SCALE_ERROR = 0.025
MAX_RELATIVE_ERROR = 0.1


def make_inputs(num_heads, num_kv_heads, head_dim, block_size, lengths, dtype=None):
    """Arguments of the op on a random_fp8_paged_cache, and the dequantized
    cache for ``paged_reference``."""
    key_cache, value_cache, key_scale, value_scale, block_tables, context_lengths = (
        random_fp8_paged_cache(num_kv_heads, head_dim, block_size, lengths)
    )
    query = torch.randn(
        len(lengths), num_heads, head_dim, dtype=dtype or torch.bfloat16, device="cuda"
    )
    args = (
        torch.full_like(query, float("nan")),
        query,
        key_cache,
        value_cache,
        context_lengths,
        block_tables,
        head_dim**-0.5,
        key_scale,
        value_scale,
    )
    k, v = dequantize_fp8_paged_cache(key_cache, value_cache, key_scale, value_scale)
    return args, (k, v)


def assert_matches(out, expected):
    out = out.float().view(expected.shape)
    torch.testing.assert_close(out, expected, atol=ATOL, rtol=RTOL)
    out, expected = out.flatten(1), expected.flatten(1)
    norm = expected.norm(dim=1)
    out, expected, norm = out[norm > 0], expected[norm > 0], norm[norm > 0]
    scale = (out * expected).sum(1) / norm**2
    assert (scale - 1).abs().max() <= MAX_SCALE_ERROR, scale
    assert ((out - expected).norm(dim=1) / norm).max() <= MAX_RELATIVE_ERROR


def reference(args, kv):
    _, query, _, _, context_lengths, block_tables, *_ = args
    out = paged_reference(
        query.unsqueeze(1), *kv, block_tables, context_lengths, causal=False
    )
    return out.squeeze(1)


@pytest.mark.parametrize(
    "num_heads,num_kv_heads,head_dim,block_size,lengths,dtype",
    [
        (8, 1, 128, 16, [1], None),
        (32, 8, 128, 16, [1, 15, 17, 300], None),  # within and across pages
        (64, 8, 128, 64, [4096, 63, 1000], None),
        (16, 1, 128, 16, [32768, 5000], None),
        (16, 4, 64, 16, [257, 512, 0, 2047, 9], None),  # one empty sequence
        (8, 8, 256, 64, [3000, 1, 700, 2048], None),
        (32, 8, 128, 16, [(37 * b) % 2048 + 1 for b in range(64)], None),
        (32, 8, 128, 16, [1000, 7], torch.float16),
    ],
)
def test_matches_reference(
    num_heads, num_kv_heads, head_dim, block_size, lengths, dtype
):
    torch.manual_seed(0)
    args, kv = make_inputs(
        num_heads, num_kv_heads, head_dim, block_size, lengths, dtype=dtype
    )
    paged_attention_decode_fp8(*args)
    assert_matches(args[0], reference(args, kv))


def test_empty_block_table():
    """All sequences empty and a block table with no columns: zeros, no fault."""
    args, kv = make_inputs(8, 1, 128, 16, [0, 0])
    args = (*args[:5], args[5][:, :0], *args[6:])
    paged_attention_decode_fp8(*args)
    torch.testing.assert_close(args[0], torch.zeros_like(args[0]))


@pytest.mark.parametrize("index", [4, 5])  # context_lengths, block_tables
def test_rejects_int64_indices(index):
    args, _ = make_inputs(8, 1, 128, 16, [300, 17])
    args = (*args[:index], args[index].long(), *args[index + 1 :])
    with pytest.raises(TypeError, match="int32"):
        paged_attention_decode_fp8(*args)


@pytest.mark.parametrize("block_size", [16, 64])
def test_cuda_graph_replays_new_lengths_and_pages(block_size):
    """A captured call reads the lengths and block table at replay."""
    torch.manual_seed(0)
    num_heads, num_kv_heads, head_dim, parts = 32, 8, 128, 8
    args, kv = make_inputs(num_heads, num_kv_heads, head_dim, block_size, [4096] * 4)
    output, _, _, _, context_lengths, block_tables, *_ = args
    context_lengths.copy_(torch.tensor([1500, 3, 4096, 250]))
    scratch = (4, num_kv_heads, parts, num_heads // num_kv_heads)
    exp_sums = torch.empty(scratch, device="cuda")
    max_logits = torch.empty(scratch, device="cuda")
    temporary_output = torch.empty(
        *scratch, head_dim, dtype=torch.bfloat16, device="cuda"
    )

    def run():
        paged_attention_decode_fp8(
            *args,
            max_context_partition_num=parts,
            exp_sums=exp_sums,
            max_logits=max_logits,
            temporary_output=temporary_output,
        )

    run()  # compile before capturing
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()

    context_lengths.copy_(torch.tensor([1, 4096, 2000, 777]))
    block_tables.copy_(block_tables[:, torch.randperm(block_tables.shape[1])])
    output.fill_(float("nan"))
    graph.replay()
    assert_matches(output, reference(args, kv))
