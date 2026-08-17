# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Dropout tests for the FlyDSL forward op.

flydsl.FwOp reproduces CK's exact philox dropout: the op materializes the
mask via ``torch.ops.xformers._ck_rand_uniform`` (the same generator CK's fused
forward uses) and records the philox ``[seed, offset]`` in ``ctx.rng_state`` so
``ck.BwOp`` regenerates the identical mask. These tests mirror ``test_dropout_ck``
but are scoped to the op's supported subset: dense self-attention (q_len == kv_len),
f16/bf16, multiple heads, head_dim in {64, 128}.
"""

import pytest
import torch
from mslk.attention import fmha
from mslk.attention.fmha.common import Context
from scipy.stats import binomtest  # type: ignore

from .test_mem_eff_attention import _vec_binom_test
from .utils import assert_allclose, ref_attention_for_test, rocm_only


def _drop_mask(batch_size, num_heads, q_len, kv_len, p, device):
    # Op-independent: always CK's per-head philox (what the op calls internally).
    dev = torch.device(device)
    dev_index = dev.index if dev.index is not None else 0
    rand_uniform = torch.ops.xformers._ck_rand_uniform(
        p, batch_size, num_heads, q_len, kv_len, dev_index
    )
    mask = (rand_uniform <= int((1.0 - p) * 255.0)).to(torch.float32)
    return mask.reshape(batch_size, num_heads, q_len, kv_len)


@rocm_only
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("attn_bias", [None, fmha.attn_bias.LowerTriangularMask()])
@pytest.mark.parametrize("seed", [42, 124])
@pytest.mark.parametrize("p", [0.3, 0.7])
@pytest.mark.parametrize("k_len", [64, 128])
@pytest.mark.parametrize("h_len", [1, 3])
@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("seq_len", [32, 256])
def test_dropout_flydsl(seq_len, batch_size, h_len, k_len, p, seed, attn_bias, dtype):
    op = fmha.flydsl.FwOp
    device = "cuda"
    scale = 3
    # Dense self-attention only (q_len == kv_len). BMHK so we exercise >1 head.
    q_len = kv_len = seq_len

    query = (
        torch.randn((batch_size, q_len, h_len, k_len), device=device, dtype=dtype)
        * scale
    )
    key = (
        torch.randn((batch_size, kv_len, h_len, k_len), device=device, dtype=dtype)
        * scale
    )
    value = (
        torch.randn((batch_size, kv_len, h_len, k_len), device=device, dtype=dtype)
        * scale
    )

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

    # Correctness: rebuild CK's per-head philox mask and compare against a masked
    # reference computed head-by-head (the 3-D ref applies drop_mask per head).
    torch.manual_seed(seed)
    mask = _drop_mask(batch_size, h_len, q_len, kv_len, p, device)
    ref = torch.stack(
        [
            ref_attention_for_test(
                query[:, :, h], key[:, :, h], value[:, :, h], attn_bias, mask[:, h], p
            )
            for h in range(h_len)
        ],
        dim=2,
    )
    # scale=3 produces large logits; large elements need an rtol (bf16 ~0.5% at
    # |x|~13), and heavy dropout at kv>=256 leaves a few near-zero outputs at the
    # f16/bf16 absolute floor (~0.04), so use a matching atol. rtol still governs the
    # signal elements, so a wrong dropout mask is still caught.
    assert_allclose(out.float(), ref, atol=5e-2, rtol=2e-2)

    # Statistical: keep prob is the byte-quantized threshold (floor((1-p)*255)+1)/256,
    # not the naive 1-p (the gap dominates the binomial over ~1e8 samples).
    num_trials = 1000
    p_val_tol = 1e-6
    keep_prob = (int((1.0 - p) * 255.0) + 1) / 256.0
    masks = []
    for _ in range(num_trials):
        masks.append(
            _drop_mask(batch_size, h_len, q_len, kv_len, p, device).clone().cpu()
        )
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
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("p", [0.0, 0.3, 0.7])
@pytest.mark.parametrize("k_len", [64, 128])
@pytest.mark.parametrize("h_len", [1, 3])
@pytest.mark.parametrize("seq_len", [32, 256])
def test_dropout_backward_flydsl(seq_len, h_len, k_len, p, dtype):
    # The FlyDSL forward has no native backward; with dropout it hands the philox
    # rng_state to ck.BwOp, which must regenerate the identical mask. Compare the
    # full fw+bw against CK end-to-end under the same seed.
    if not fmha.ck.BwOp.is_available():
        pytest.skip("ck.BwOp unavailable")
    flyF = fmha.flydsl.FwOp
    device = "cuda"
    scale = 3
    q_len = kv_len = seq_len

    def mk():
        torch.manual_seed(0)
        q = torch.randn((1, q_len, h_len, k_len), device=device, dtype=dtype) * scale
        k = torch.randn((1, kv_len, h_len, k_len), device=device, dtype=dtype) * scale
        v = torch.randn((1, kv_len, h_len, k_len), device=device, dtype=dtype) * scale
        return q, k, v

    q, k, v = mk()
    inp = fmha.Inputs(query=q, key=k, value=v, attn_bias=None, p=p)
    if not flyF.supports(inp):
        pytest.skip(f"{flyF.NAME}: unsupported input")

    grad_out = torch.randn((1, q_len, h_len, k_len), device=device, dtype=dtype)

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

    if dtype is torch.float16:
        # f16 (10-bit mantissa): flydslF+ck.BwOp matches ck end-to-end per element.
        # atol 5e-2 covers the handful of near-zero elements at the f16 floor for
        # kv>=256 (cancellation in the backward); rtol governs the signal.
        assert_allclose(out_f.float(), out_c.float(), "fwd", atol=5e-2, rtol=2e-2)
        assert_allclose(g_f.dq.float(), g_c.dq.float(), "dq", atol=5e-2, rtol=2e-2)
        assert_allclose(g_f.dk.float(), g_c.dk.float(), "dk", atol=5e-2, rtol=2e-2)
        assert_allclose(g_f.dv.float(), g_c.dv.float(), "dv", atol=5e-2, rtol=2e-2)
    else:
        # bf16 (8-bit mantissa): the attention backward is cancellation-dominated at
        # near-zero gradients (and outputs), where bf16 has large per-element error
        # regardless of kernel. Compare the aggregate relative L2 norm instead:
        # robust to those outliers, but a wrong dropout mask still blows the norm up
        # well past the thresholds.
        def _rel_l2(a, b):
            return (a - b).float().norm() / b.float().norm().clamp_min(1e-6)

        for name, ga, gb, tol in (
            ("fwd", out_f, out_c, 5e-2),
            ("dq", g_f.dq, g_c.dq, 1e-1),
            ("dk", g_f.dk, g_c.dk, 1e-1),
            ("dv", g_f.dv, g_c.dv, 1e-1),
        ):
            rel = _rel_l2(ga, gb)
            assert rel < tol, f"{name}: relative L2 {rel:.4f} exceeds {tol}"


@rocm_only
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("p", [0.0, 0.3])
@pytest.mark.parametrize("hq_len", [1, 3])
@pytest.mark.parametrize("g_len", [2])
@pytest.mark.parametrize("seq_len", [32, 128])
def test_dropout_backward_bmghk_flydsl(seq_len, g_len, hq_len, p, dtype):
    # BMGHK (5-D) dropout with gradients: the op folds G into H, runs the dense
    # dropout forward, then re-wraps the 5-D context. That re-wrap must PRESERVE the
    # philox rng_state (regression guard) so ck.BwOp can regenerate the 5-D mask
    # instead of raising. Compare the full fw+bw against ck end to end.
    if not fmha.ck.BwOp.is_available():
        pytest.skip("ck.BwOp unavailable")
    flyF = fmha.flydsl.FwOp
    device = "cuda"
    scale = 3
    k_len = 128
    q_len = kv_len = seq_len

    def mk():
        torch.manual_seed(0)
        qs = (1, q_len, g_len, hq_len, k_len)
        kvs = (1, kv_len, g_len, hq_len, k_len)
        return (
            torch.randn(qs, device=device, dtype=dtype) * scale,
            torch.randn(kvs, device=device, dtype=dtype) * scale,
            torch.randn(kvs, device=device, dtype=dtype) * scale,
        )

    q, k, v = mk()
    inp = fmha.Inputs(query=q, key=k, value=v, attn_bias=None, p=p)
    if not flyF.supports(inp):
        pytest.skip(f"{flyF.NAME}: unsupported input")

    grad_out = torch.randn((1, q_len, g_len, hq_len, k_len), device=device, dtype=dtype)

    torch.manual_seed(77)
    out_f, ctx_f = flyF.apply(inp, needs_gradient=True)
    assert out_f.shape == q.shape
    # LSE is unflattened back to the (G, Hq) head split.
    assert tuple(ctx_f.lse.shape) == (1, g_len, hq_len, q_len), ctx_f.lse.shape
    if p != 0.0:
        # The 5-D re-wrap must keep rng_state, else ck.BwOp gets None and raises.
        assert ctx_f.rng_state is not None
        assert ctx_f.op_bw is fmha.ck.BwOp

    # ck.BwOp runs on BMHK, so fold G into H exactly as the op's autograd wrapper
    # does and drive the backward with the preserved rng_state. This confirms
    # ck.BwOp can regenerate the 5-D dropout mask (the regression the fix guards)
    # rather than getting rng_state=None and raising.
    qf, kf, vf = q.flatten(2, 3), k.flatten(2, 3), v.flatten(2, 3)
    inp_flat = fmha.Inputs(query=qf, key=kf, value=vf, attn_bias=None, p=p)
    grad_flat = grad_out.flatten(2, 3)

    def _bwd(out_x, lse_x, rng_x):
        ctx = Context(
            lse=lse_x.flatten(1, 2),
            out=out_x.flatten(2, 3),
            op_bw=fmha.ck.BwOp,
            rng_state=rng_x,
        )
        return fmha.ck.BwOp.apply(ctx, inp_flat, grad_flat)

    g_f = _bwd(out_f, ctx_f.lse, ctx_f.rng_state)

    torch.manual_seed(77)
    out_c, ctx_c = fmha.ck.FwOp.apply(inp, needs_gradient=True)
    g_c = _bwd(out_c, ctx_c.lse, ctx_c.rng_state)

    if dtype is torch.float16:
        assert_allclose(out_f.float(), out_c.float(), "fwd", atol=5e-2, rtol=2e-2)
        assert_allclose(g_f.dq.float(), g_c.dq.float(), "dq", atol=5e-2, rtol=2e-2)
        assert_allclose(g_f.dk.float(), g_c.dk.float(), "dk", atol=5e-2, rtol=2e-2)
        assert_allclose(g_f.dv.float(), g_c.dv.float(), "dv", atol=5e-2, rtol=2e-2)
    else:

        def _rel_l2(a, b):
            return (a - b).float().norm() / b.float().norm().clamp_min(1e-6)

        for name, ga, gb, tol in (
            ("fwd", out_f, out_c, 5e-2),
            ("dq", g_f.dq, g_c.dq, 1e-1),
            ("dk", g_f.dk, g_c.dk, 1e-1),
            ("dv", g_f.dv, g_c.dv, 1e-1),
        ):
            rel = _rel_l2(ga, gb)
            assert rel < tol, f"{name}: relative L2 {rel:.4f} exceeds {tol}"
