# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Dropout tests for the FlyDSL forward op.

flydsl_forward.FwOp reproduces CK's exact philox dropout: the op materializes the
mask via ``torch.ops.xformers._ck_rand_uniform`` (the same generator CK's fused
forward uses) and records the philox ``[seed, offset]`` in ``ctx.rng_state`` so
``ck.BwOp`` regenerates the identical mask. These tests mirror ``test_dropout_ck``
but are scoped to the op's supported subset: dense self-attention (q_len == kv_len),
f16/bf16, head_dim in {64, 128}.
"""

import pytest
import torch
from mslk.attention import fmha
from scipy.stats import binomtest  # type: ignore

from .test_mem_eff_attention import _vec_binom_test
from .utils import assert_allclose, ref_attention_for_test, rocm_only


def _drop_mask(batch_size, q_len, kv_len, p, device):
    # Op-independent: always CK's philox (what the op itself calls internally).
    dev = torch.device(device)
    dev_index = dev.index if dev.index is not None else 0
    rand_uniform = torch.ops.xformers._ck_rand_uniform(
        p, batch_size, 1, q_len, kv_len, dev_index
    )
    mask = (rand_uniform <= int((1.0 - p) * 255.0)).to(torch.float32)
    return mask.reshape(batch_size, q_len, kv_len)


@rocm_only
@pytest.mark.parametrize("attn_bias", [None, fmha.attn_bias.LowerTriangularMask()])
@pytest.mark.parametrize("seed", [42, 124])
@pytest.mark.parametrize("p", [0.3, 0.7])
@pytest.mark.parametrize("k_len", [64, 128])
@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("seq_len", [32, 256])
def test_dropout_flydsl(seq_len, batch_size, k_len, p, seed, attn_bias):
    op = fmha.flydsl_forward.FwOp
    device = "cuda"
    scale = 3
    dtype = torch.float16
    # Dense self-attention only (q_len == kv_len).
    q_len = kv_len = seq_len

    query = torch.randn((batch_size, q_len, k_len), device=device, dtype=dtype) * scale
    key = torch.randn((batch_size, kv_len, k_len), device=device, dtype=dtype) * scale
    value = torch.randn((batch_size, kv_len, k_len), device=device, dtype=dtype) * scale

    if not op.supports(fmha.Inputs(query, key, value, attn_bias, p, None)):
        pytest.skip(f"{op.NAME}: unsupported input")

    torch.manual_seed(seed)
    out = fmha.memory_efficient_attention(
        query, key, value, attn_bias, p, op=(op, None)
    )
    torch.manual_seed(seed)
    out2 = fmha.memory_efficient_attention(
        query, key, value, attn_bias, p, op=(op, None)
    )
    assert_allclose(out, out2, "dropout reproducibility")

    # Correctness: rebuild CK's philox mask and compare against a masked reference.
    torch.manual_seed(seed)
    mask = _drop_mask(batch_size, q_len, kv_len, p, device)
    ref = ref_attention_for_test(query, key, value, attn_bias, mask, p)
    assert_allclose(out.float(), ref, atol=2.8e-2)

    # Statistical: keep prob is the byte-quantized threshold (floor((1-p)*255)+1)/256,
    # not the naive 1-p (the gap dominates the binomial over ~1e8 samples).
    num_trials = 1000
    p_val_tol = 1e-6
    keep_prob = (int((1.0 - p) * 255.0) + 1) / 256.0
    masks = []
    for _ in range(num_trials):
        masks.append(_drop_mask(batch_size, q_len, kv_len, p, device).clone().cpu())
    masks = torch.stack(masks, dim=0)
    p_value = binomtest(int(masks.sum()), masks.numel(), p=keep_prob).pvalue
    assert p_value > p_val_tol, p_value
    # Per-element check allows a few spurious outliers (~N*1e-6 expected at this tol).
    masks = masks.sum(0).flatten()
    p_values = _vec_binom_test(masks.numpy(), num_trials, p=keep_prob)
    n_outliers = int((p_values <= p_val_tol).sum())
    assert n_outliers <= max(1, masks.numel() // 50000), (
        f"{n_outliers} per-element binomial outliers of {masks.numel()}"
    )


@rocm_only
@pytest.mark.parametrize("p", [0.0, 0.3, 0.7])
@pytest.mark.parametrize("k_len", [64, 128])
@pytest.mark.parametrize("seq_len", [32, 256])
def test_dropout_backward_flydsl(seq_len, k_len, p):
    # The FlyDSL forward has no native backward; with dropout it hands the philox
    # rng_state to ck.BwOp, which must regenerate the identical mask. Compare the
    # full fw+bw against CK end-to-end under the same seed.
    if not fmha.ck.BwOp.is_available():
        pytest.skip("ck.BwOp unavailable")
    flyF = fmha.flydsl_forward.FwOp
    device = "cuda"
    scale = 3
    dtype = torch.float16
    q_len = kv_len = seq_len

    def mk():
        torch.manual_seed(0)
        q = torch.randn((1, q_len, 1, k_len), device=device, dtype=dtype) * scale
        k = torch.randn((1, kv_len, 1, k_len), device=device, dtype=dtype) * scale
        v = torch.randn((1, kv_len, 1, k_len), device=device, dtype=dtype) * scale
        return q, k, v

    q, k, v = mk()
    inp = fmha.Inputs(query=q, key=k, value=v, attn_bias=None, p=p)
    if not flyF.supports(inp):
        pytest.skip(f"{flyF.NAME}: unsupported input")

    grad_out = torch.randn((1, q_len, 1, k_len), device=device, dtype=dtype)

    torch.manual_seed(77)
    out_f, ctx_f = flyF.apply(inp, needs_gradient=True)
    if p != 0.0:
        assert ctx_f.rng_state is not None
        assert ctx_f.rng_state.dtype == torch.int64
        assert tuple(ctx_f.rng_state.shape) == (2,)
        assert ctx_f.op_bw is fmha.ck.BwOp
    g_f = fmha.ck.BwOp.apply(ctx_f, inp, grad_out)

    torch.manual_seed(77)
    out_c, ctx_c = fmha.ck.FwOp.apply(inp, needs_gradient=True)
    g_c = fmha.ck.BwOp.apply(ctx_c, inp, grad_out)

    assert_allclose(out_f.float(), out_c.float(), "fwd", atol=2.8e-2)
    assert_allclose(g_f.dq.float(), g_c.dq.float(), "dq", atol=2.8e-2, rtol=2e-2)
    assert_allclose(g_f.dk.float(), g_c.dk.float(), "dk", atol=2.8e-2, rtol=2e-2)
    assert_allclose(g_f.dv.float(), g_c.dv.float(), "dv", atol=2.8e-2, rtol=2e-2)
