# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.
# pyre-unsafe

import pytest
import torch

from mslk.attention import fmha
from mslk.attention.fmha.split_blocks_fairinternal import (
    split_blocks_for_decoding,
    split_blocks_for_prefill,
)

from .utils import cuda_only

compute_capability = (0, 0)
if torch.cuda.is_available():
    compute_capability = torch.cuda.get_device_capability("cuda")
sm80_or_better_only = pytest.mark.skipif(
    torch.version.cuda is not None and compute_capability < (8, 0),
    reason="requires sm80+",
)


def test_split_blocks_for_decoding():
    torch.manual_seed(0)
    max_len_kv = 2048
    B = 64
    local_attention_len = 512
    seqlens = torch.randint(
        max_len_kv, size=(B,), device=torch._C._get_accelerator().type
    )
    seqlens[4] = 0
    attn_bias = fmha.attn_bias.BlockDiagonalCausalWithOffsetPaddedKeysMask.from_seqlens(
        q_seqlen=[1] * B, kv_seqlen=seqlens.tolist(), kv_padding=max_len_kv
    )
    chunked_bias = split_blocks_for_decoding(attn_bias, local_attention_len)
    assert chunked_bias.q_seqinfo.seqstart_py == list(range(B + 1))
    assert chunked_bias.k_seqinfo.seqlen.tolist() == chunked_bias.k_seqinfo.seqlen_py
    assert (
        chunked_bias.k_seqinfo.seqstart.tolist() == chunked_bias.k_seqinfo.seqstart_py
    )
    assert (chunked_bias.k_seqinfo.seqlen <= local_attention_len).all()
    assert (chunked_bias.k_seqinfo.seqstart >= attn_bias.k_seqinfo.seqstart).all()


def test_split_blocks_for_decoding_with_paged():
    torch.manual_seed(0)
    max_len_kv = 2048
    B = 16
    local_attention_len = 512
    seqlens = torch.randint(
        max_len_kv, size=(B,), device=torch._C._get_accelerator().type
    )
    # test the case with zero lengths:
    # - set last 3 seqlen to 0
    seqlens[-2] = seqlens[-1] = seqlens[0] = 0

    page_size = 256
    num_pages = 64

    page_table = torch.zeros((B, num_pages), dtype=torch.int32, device="cuda")

    attn_bias = fmha.attn_bias.BlockDiagonalCausalWithOffsetPaddedKeysMask.from_seqlens(
        q_seqlen=[1] * B, kv_seqlen=seqlens.tolist(), kv_padding=max_len_kv
    )
    chunked_bias = split_blocks_for_decoding(
        attn_bias, local_attention_len, block_tables=page_table, page_size=page_size
    )

    # for paged + gappy, the length is the last post, instead of the real length

    k_seqlen = chunked_bias.k_seqinfo.seqlen
    k_seqstart = chunked_bias.k_seqinfo.seqstart
    k_seqlen_py = chunked_bias.k_seqinfo.seqlen_py
    k_seqstart_py = chunked_bias.k_seqinfo.seqstart_py

    paged_gappy_length = k_seqlen - k_seqstart

    assert (paged_gappy_length <= local_attention_len).all()
    assert (paged_gappy_length >= 0).all()
    assert (k_seqstart >= 0).all()  # check the zero length case
    assert len(k_seqstart) == len(k_seqlen) == B
    assert k_seqlen.tolist() == k_seqlen_py
    assert k_seqstart.tolist() == k_seqstart_py


if not torch.version.hip and compute_capability >= (9, 0):
    decode_ops = [fmha.triton_splitk.FwOp, fmha.flash3.FwOp, fmha.flash3.FwOp_KVSplit]
else:
    decode_ops = [fmha.triton_splitk.FwOp]


@cuda_only
@sm80_or_better_only
@pytest.mark.parametrize(
    "spec_decoding", [False, True], ids=lambda x: "spec" if x else ""
)
@pytest.mark.parametrize("paged", [False, True], ids=lambda x: "paged" if x else "")
@pytest.mark.parametrize(
    "decoding_op",
    decode_ops,
    ids=lambda x: x.__name__,
)
def test_split_blocks_decoding_vs_prefill(spec_decoding, paged, decoding_op):
    """
    We should be able to use the prefill split-blocks algo for decoding, and get the same attention output.
    """
    torch.manual_seed(0)
    if torch.version.hip:
        op = fmha.ck.FwOp
    elif fmha.flash.FwOp.VARLEN_LSE_PACKED:
        # Modern enough flash attention
        op = fmha.flash.FwOp
    else:
        pytest.skip("We have op to test with")
        raise AssertionError

    AttnBias = (
        fmha.attn_bias.BlockDiagonalPaddedKeysMask
        if spec_decoding
        else fmha.attn_bias.BlockDiagonalCausalWithOffsetPaddedKeysMask
    )
    PagedAttnBias = (
        fmha.attn_bias.PagedBlockDiagonalPaddedKeysMask
        if spec_decoding
        else fmha.attn_bias.PagedBlockDiagonalCausalWithOffsetPaddedKeysMask
    )

    dtype = torch.bfloat16
    nheads_kv = 1
    nheads_q = 8
    seq_len_q = 5 if spec_decoding else 1
    head_dim = 128
    max_len_kv = 2048
    B = 64
    local_attention_len = 512
    seqlens = torch.randint(
        low=seq_len_q,
        high=max_len_kv,
        size=(B,),
        device=torch._C._get_accelerator().type,
    )

    if not spec_decoding:
        # Here we introduce corner cases when the query is close to the boundaries of two iRoPE chunks.
        # In particular, for spec decoding, when the query (aka draft) length > 1, special care
        # needs to be taken when the draft crosses the boundary - see the comment in
        # split_blocks_fairinternal.split_blocks_for_decoding_gpu_part.
        # Function split_blocks_for_prefill doesn't handle this case correctly,
        # so we skip these corner cases for spec decoding.
        seqlens[3] = local_attention_len * 2  # corner cases
        # seqlens[4] = local_attention_len * 2  + 3 # reproduces cross boundary case
        seqlens[2] = local_attention_len
    if spec_decoding:
        # Ensure no cross-boundary attention
        non_compliant_mask = (seqlens >= seq_len_q) & (
            seqlens % local_attention_len < seq_len_q
        )
        adjustment = (seqlens % local_attention_len) * non_compliant_mask
        seqlens -= adjustment
    attn_bias = AttnBias.from_seqlens(
        q_seqlen=[seq_len_q] * B, kv_seqlen=seqlens.tolist(), kv_padding=max_len_kv
    )
    if paged:
        page_size = 256
        block_tables = torch.arange(
            B * max_len_kv // page_size, device="cuda", dtype=torch.int32
        ).reshape(B, -1)
    else:
        page_size = None
        block_tables = None
    chunked_bias_decoding = split_blocks_for_decoding(
        attn_bias, local_attention_len, block_tables, page_size
    )
    chunked_bias_prefill = split_blocks_for_prefill(attn_bias, local_attention_len)
    prefill_attn_to_use = chunked_bias_prefill
    if paged:
        attn_batch_size = len(chunked_bias_prefill.k_seqinfo.seqlen)
        if attn_batch_size != block_tables.shape[0]:
            block_tables = block_tables.view(attn_batch_size, -1)
        prefill_attn_to_use = chunked_bias_prefill.make_paged(
            block_tables,
            page_size,
            paged_type=PagedAttnBias,
        )

    if type(chunked_bias_decoding) not in decoding_op.SUPPORTED_ATTN_BIAS_TYPES:
        pytest.skip(
            f"Decoding op {decoding_op.NAME} doesn't support bias {type(chunked_bias_decoding)}"
        )
    if type(prefill_attn_to_use) not in op.SUPPORTED_ATTN_BIAS_TYPES:
        pytest.skip(
            f"Prefill op {op.NAME} doesn't support bias {type(prefill_attn_to_use)}"
        )

    # The only difference between attention biases should be that the bias computed
    # using split_blocks_for_prefill contains elements with query len 0.
    decoding_q_lens = [b - a for a, b in chunked_bias_decoding.q_seqinfo.intervals()]
    prefill_q_lens = [b - a for a, b in prefill_attn_to_use.q_seqinfo.intervals()]
    assert [x for x in prefill_q_lens if x > 0] == decoding_q_lens
    if paged:
        # seqlen_py is the end of the sequence, not the length
        decoding_k_lens = [
            a - b
            for a, b in zip(
                chunked_bias_decoding.k_seqinfo.seqlen_py,
                chunked_bias_decoding.k_seqinfo.seqstart_py,
            )
        ]
    else:
        decoding_k_lens = chunked_bias_decoding.k_seqinfo.seqlen_py
    filtered_prefill_k_lens = [
        x
        for x, y in zip(prefill_attn_to_use.k_seqinfo.seqlen_py, prefill_q_lens)
        if y > 0
    ]
    assert decoding_k_lens == filtered_prefill_k_lens

    q = torch.randn(
        B,
        seq_len_q,
        nheads_q,
        head_dim,
        device=torch._C._get_accelerator().type,
        dtype=dtype,
    )
    k = torch.randn(
        B,
        max_len_kv,
        nheads_kv,
        head_dim,
        device=torch._C._get_accelerator().type,
        dtype=dtype,
    )
    v = torch.randn(
        B,
        max_len_kv,
        nheads_kv,
        head_dim,
        device=torch._C._get_accelerator().type,
        dtype=dtype,
    )

    xq = q.view(1, -1, nheads_q, head_dim)
    xk = k.view(1, -1, nheads_kv, head_dim).expand(1, -1, nheads_q, -1)
    xv = v.view(1, -1, nheads_kv, head_dim).expand(1, -1, nheads_q, -1)

    out_dec, lse_dec = fmha.memory_efficient_attention_forward_requires_grad(
        xq, xk, xv, chunked_bias_decoding, op=decoding_op
    )

    out_prefill, lse_prefill = fmha.memory_efficient_attention_forward_requires_grad(
        xq, xk, xv, prefill_attn_to_use, op=op
    )

    torch.testing.assert_close(out_dec, out_prefill, rtol=1e-4, atol=5e-3)
    torch.testing.assert_close(lse_dec, lse_prefill, rtol=1e-4, atol=1e-4)
