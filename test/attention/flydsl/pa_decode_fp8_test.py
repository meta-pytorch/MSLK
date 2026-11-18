# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""paged_attention_decode_fp8 against an fp32 reference over the dequantized
cache: ragged lengths, padded block tables and CUDA graphs, through its routing
and with each backend forced."""

import contextlib

import pytest
import torch
from mslk.aiter.common import is_aiter_enabled
from mslk.aiter.routing import device_key, force_backend
from mslk.attention.dispatch.paged_attention_decode_fp8 import OP, ROUTING
from mslk.attention.flydsl.decode.fp8_paged_cache import (
    dequantize_fp8_paged_cache,
    random_fp8_paged_cache,
)
from mslk.attention.flydsl.decode.pa_decode_fp8_dispatch import (
    is_fp8_paged_decode_available,
    paged_attention_decode_fp8,
    paged_attention_decode_fp8_csr,
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


BACKENDS = ("default", "mslk", "aiter")

# (num_heads, num_kv_heads, head_dim, block_size, context_max, batch) shapes that
# the tuning table routes to each backend on gfx950, so that the routed path is
# covered on both.
_ROUTED_SHAPES = {
    "aiter": [(32, 8, 128, 16, 256, 256), (64, 8, 128, 64, 32768, 1)],
    "mslk": [(8, 1, 128, 16, 1024, 8), (4, 1, 128, 64, 256, 4)],
}


@contextlib.contextmanager
def backend_forced(backend):
    if backend == "aiter" and not is_aiter_enabled(OP):
        pytest.skip("requires the pinned aiter, enabled for the op")
    with (
        force_backend(OP, backend) if backend != "default" else contextlib.nullcontext()
    ):
        yield


def run_and_check(args, kv, backend):
    """Run the op on ``args`` and check its output; returns the backend that
    served it."""
    served_by = ROUTING.backend_for(*args)
    paged_attention_decode_fp8(*args)
    assert_matches(args[0], reference(args, kv))
    if backend != "default":
        assert served_by == backend
    return served_by


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize(
    "num_heads,num_kv_heads,head_dim,block_size,lengths",
    [
        (8, 1, 128, 16, [1]),
        (32, 8, 128, 16, [1, 15, 17, 300]),  # within and across pages
        (64, 8, 128, 64, [4096, 63, 1000]),
        (16, 1, 128, 16, [32768, 5000]),
        (16, 4, 64, 16, [257, 512, 0, 2047, 9]),  # one empty sequence
        (8, 8, 256, 64, [3000, 1, 700, 2048]),
        (32, 8, 128, 16, [(37 * b) % 2048 + 1 for b in range(64)]),
    ],
)
def test_matches_reference(
    num_heads, num_kv_heads, head_dim, block_size, lengths, backend
):
    torch.manual_seed(0)
    args, kv = make_inputs(num_heads, num_kv_heads, head_dim, block_size, lengths)
    with backend_forced(backend):
        run_and_check(args, kv, backend)


@pytest.mark.parametrize("backend", BACKENDS)
def test_empty_block_table(backend):
    """All sequences empty and a block table with no columns: zeros, no fault."""
    args, kv = make_inputs(8, 1, 128, 16, [0, 0])
    args = (*args[:5], args[5][:, :0], *args[6:])
    with backend_forced(backend):
        paged_attention_decode_fp8(*args)
    torch.testing.assert_close(args[0], torch.zeros_like(args[0]))


def test_broadcast_block_table():
    """A block table with row stride 0 (every sequence on the same pages)."""
    torch.manual_seed(0)
    args, kv = make_inputs(16, 4, 128, 16, [300, 120, 17, 300])
    args = (*args[:5], args[5][:1].expand(4, -1), *args[6:])
    paged_attention_decode_fp8(*args)
    assert_matches(args[0], reference(args, kv))


def test_padding_with_own_pages_ignores_unused_block_0():
    """Rows padded with their own first page must not read block 0, which no
    sequence uses here and which holds NaN."""
    torch.manual_seed(0)
    args, _ = make_inputs(16, 4, 128, 16, [300, 17, 64])
    out, query, k, v, lengths, tables, scale, k_scale, v_scale = args
    # Move block 0's data to the spare block, then poison block 0.
    used = set(tables[tables >= 0].tolist())
    spare = next(b for b in range(k.shape[0]) if b not in used)
    for t in (k, v, k_scale, v_scale):
        t[spare] = t[0]
    tables[tables == 0] = spare
    k_scale[0] = v_scale[0] = float("nan")
    tables[:] = torch.where(tables < 0, tables[:, :1], tables)
    paged_attention_decode_fp8(*args)
    kv = dequantize_fp8_paged_cache(k, v, k_scale, v_scale)
    assert_matches(out, reference(args, kv))


def test_csr_entry_takes_int64_lengths():
    torch.manual_seed(0)
    args, kv = make_inputs(16, 4, 128, 16, [300, 17, 64])
    out, query, k, v, lengths, tables, scale, k_scale, v_scale = args
    pages = [row[row >= 0] for row in tables]
    indptr = torch.tensor([0] + [len(p) for p in pages], device="cuda").cumsum(0)
    paged_attention_decode_fp8_csr(
        out,
        query,
        k,
        v,
        lengths.long(),
        torch.cat(pages),
        indptr,
        scale,
        k_scale,
        v_scale,
    )
    assert_matches(out, reference(args, kv))


def test_cache_builder_takes_no_sequences():
    *_, block_tables, context_lengths = random_fp8_paged_cache(1, 128, 16, [])
    assert block_tables.shape == (0, 1) and context_lengths.numel() == 0


@pytest.mark.parametrize("index", [4, 5])  # context_lengths, block_tables
def test_rejects_int64_indices(index):
    args, _ = make_inputs(8, 1, 128, 16, [300, 17])
    args = (*args[:index], args[index].long(), *args[index + 1 :])
    with pytest.raises(TypeError, match="int32"):
        paged_attention_decode_fp8(*args)


@pytest.mark.parametrize(
    "shape,expected",
    [(shape, b) for b, shapes in _ROUTED_SHAPES.items() for shape in shapes],
)
def test_routes_by_its_tuning_table(shape, expected):
    num_heads, num_kv_heads, head_dim, block_size, context_max, batch = shape
    torch.manual_seed(0)
    # Lengths spread over (context_max / 2, context_max], as in tuning; the
    # longest fills the block table.
    lengths = [context_max - context_max // 2 * b // batch for b in range(batch)]
    args, kv = make_inputs(num_heads, num_kv_heads, head_dim, block_size, lengths)
    served_by = run_and_check(args, kv, "default")
    gfx, cu_num = device_key(torch.cuda.current_device())
    covered = any(
        (r["gfx"], r["cu_num"]) == (gfx, str(cu_num)) for r in ROUTING.table.rows
    )
    if is_aiter_enabled(OP) and covered:
        assert served_by == expected, (
            "the tuning table no longer routes this shape to "
            f"{expected}; replace it in _ROUTED_SHAPES with one it does"
        )
    else:
        assert served_by == "mslk"


@pytest.mark.parametrize(
    "call",
    [
        "head_size_192",
        "f16_query",
        "uint8_cache",
        "larger_scales",
        "cpu_scales",
        "wider_table",
        "2d_output",
    ],
)
def test_aiter_leaves_calls_it_cannot_take_to_mslk(call):
    """Calls MSLK's kernel takes but aiter can't (or wasn't tuned for) go to
    MSLK's kernel even when aiter is forced."""
    torch.manual_seed(0)
    args, kv = make_inputs(
        16,
        4,
        192 if call == "head_size_192" else 128,
        16,
        [300, 17],
        dtype=torch.float16 if call == "f16_query" else None,
    )
    out, query, k, v, lengths, tables, scale, k_scale, v_scale = args
    if call == "uint8_cache":
        k, v = k.view(torch.uint8), v.view(torch.uint8)
    elif call == "larger_scales":  # of a larger block pool; only strides are used
        k_scale, v_scale = (
            torch.cat([s, torch.ones_like(s[:4])]) for s in (k_scale, v_scale)
        )
    elif call == "cpu_scales":
        k_scale, v_scale = k_scale.cpu(), v_scale.cpu()
    elif call == "wider_table":  # rows for a larger batch than this call's
        tables = torch.cat([tables, torch.full_like(tables, -1)])
    elif call == "2d_output":
        out = out.flatten(1)
    args = (out, query, k, v, lengths, tables, scale, k_scale, v_scale)
    with backend_forced("aiter"):
        assert run_and_check(args, kv, "default") == "mslk"


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("block_size", [16, 64])
def test_cuda_graph_replays_new_lengths_and_pages(block_size, backend):
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

    with backend_forced(backend):
        run()  # compile before capturing
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            run()

    context_lengths.copy_(torch.tensor([1, 4096, 2000, 777]))
    block_tables.copy_(block_tables[:, torch.randperm(block_tables.shape[1])])
    output.fill_(float("nan"))
    graph.replay()
    assert_matches(output, reference(args, kv))
