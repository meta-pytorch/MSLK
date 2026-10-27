# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Dropout tests for the FlyDSL attention ops.

The reference mask is an independent Triton (tl.philox) implementation of the
definition in mslk.attention.flydsl.dropout_philox.
"""

import pytest
import torch
import triton
import triton.language as tl
from mslk.attention import fmha
from scipy.stats import binomtest  # type: ignore

from .test_mem_eff_attention import _vec_binom_test
from .utils import assert_allclose, ref_attention_for_test, rocm_only


@triton.jit
def _reference_keep_kernel(out_ptr, seed, total, M, N, threshold, BLOCK: tl.constexpr):
    # keep[b, h, m, n]: word n % 4 of Philox4x32-10(key=seed,
    # counter=(n // 4, m, b * H + h, 0)) >= threshold.
    idx = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    n = idx % N
    m = (idx // N) % M
    bh = idx // N // M
    c0 = (n // 4).to(tl.uint32)
    r0, r1, r2, r3 = tl.philox(
        seed, c0, m.to(tl.uint32), bh.to(tl.uint32), tl.zeros_like(c0)
    )
    word = n % 4
    rand = tl.where(word == 0, r0, tl.where(word == 1, r1, tl.where(word == 2, r2, r3)))
    keep = rand >= threshold.to(tl.uint32)
    tl.store(out_ptr + idx, keep.to(tl.uint8), mask=idx < total)


def _draw_seed():
    from mslk.attention.flydsl.dropout_philox import draw_dropout_seed

    return int(draw_dropout_seed(torch.device("cuda")).item())


def _drop_mask(seed, batch_size, num_heads, q_len, kv_len, p):
    keep = torch.empty(
        (batch_size, num_heads, q_len, kv_len), dtype=torch.uint8, device="cuda"
    )
    threshold = min(round(p * 2**32), 2**32 - 1)
    grid = (triton.cdiv(keep.numel(), 1024),)
    _reference_keep_kernel[grid](
        keep, seed, keep.numel(), q_len, kv_len, threshold, BLOCK=1024
    )
    return keep.to(torch.float32)


def _keep_probability(p):
    return 1.0 - min(round(p * 2**32), 2**32 - 1) / 2**32


def _is_gfx950():
    return "gfx950" in torch.cuda.get_device_properties(0).gcnArchName


@rocm_only
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("p", [0.2, 0.5])
@pytest.mark.parametrize("D", [64, 128])
def test_dropout_mask_bit_exact_flydsl(D, p, dtype):
    # With kv_len == D, V = I gives out = P' and dO = I gives dV = P'^T; P > 0
    # everywhere, so the nonzeros are exactly the kept elements.
    if not _is_gfx950():
        pytest.skip("flydsl dropout backward is gfx950-only")
    B, H, S = 2, 3, D
    torch.manual_seed(0)
    q = torch.randn((B, S, H, D), device="cuda", dtype=dtype).requires_grad_(True)
    k = torch.randn((B, S, H, D), device="cuda", dtype=dtype)
    eye = torch.eye(D, device="cuda", dtype=dtype)[None, :, None, :].expand(B, S, H, D)
    v = eye.contiguous().requires_grad_(True)
    rng = torch.cuda.get_rng_state()
    out = fmha.memory_efficient_attention(q, k, v, p=p)
    out.backward(eye.contiguous())
    torch.cuda.set_rng_state(rng)
    ref = _drop_mask(_draw_seed(), B, H, S, S, p).bool()
    fwd_keep = out.permute(0, 2, 1, 3) != 0  # [B, H, m, n]
    bwd_keep = v.grad.permute(0, 2, 3, 1) != 0  # dV[b, n, h, m] -> [B, H, m, n]
    assert torch.equal(fwd_keep, ref), f"forward: {int((fwd_keep != ref).sum())} bits"
    assert torch.equal(bwd_keep, ref), f"backward: {int((bwd_keep != ref).sum())} bits"


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

    # Reseeding reproduces the op's Philox seed, hence its mask.
    torch.manual_seed(seed)
    mask = _drop_mask(_draw_seed(), batch_size, h_len, q_len, kv_len, p)
    ref = torch.stack(
        [
            ref_attention_for_test(
                query[:, :, h], key[:, :, h], value[:, :, h], attn_bias, mask[:, h], p
            )
            for h in range(h_len)
        ],
        dim=2,
    )
    # A wrong mask diverges in thousands of elements; allow a few bf16 outliers.
    of = out.float()
    close = torch.isclose(of, ref.float(), atol=5e-2, rtol=2e-2)
    n_bad = int((~close).sum())
    assert n_bad <= max(8, of.numel() // 2000), (
        f"{n_bad}/{of.numel()} elements exceed tolerance (likely wrong dropout mask)"
    )

    # Statistical: one fresh seed per trial.
    num_trials = 1000
    p_val_tol = 1e-6
    keep_prob = _keep_probability(p)
    masks = []
    for _ in range(num_trials):
        masks.append(
            _drop_mask(_draw_seed(), batch_size, h_len, q_len, kv_len, p).cpu()
        )
    masks = torch.stack(masks, dim=0)
    p_value = binomtest(int(masks.sum()), masks.numel(), p=keep_prob).pvalue
    assert p_value > p_val_tol, p_value
    masks = masks.sum(0).flatten()
    p_values = _vec_binom_test(masks.numpy(), num_trials, p=keep_prob)
    n_outliers = int((p_values <= p_val_tol).sum())
    assert n_outliers <= max(1, masks.numel() // 50000), (
        f"{n_outliers} per-element binomial outliers of {masks.numel()}"
    )


@rocm_only
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("causal", [False, True], ids=["full", "causal"])
@pytest.mark.parametrize("p", [0.3, 0.7])
@pytest.mark.parametrize("k_len", [64, 128])
@pytest.mark.parametrize("seq_len", [32, 200])
def test_dropout_training_flydsl(seq_len, k_len, p, causal, dtype):
    if not _is_gfx950():
        pytest.skip("flydsl dropout backward is gfx950-only")
    device, B, H = "cuda", 2, 3
    torch.manual_seed(0)
    q, k, v, grad = (
        torch.randn((B, seq_len, H, k_len), device=device, dtype=dtype)
        for _ in range(4)
    )
    q, k, v = (t.requires_grad_(True) for t in (q, k, v))
    bias = fmha.attn_bias.LowerTriangularMask() if causal else None
    rng = torch.cuda.get_rng_state()
    out = fmha.memory_efficient_attention(q, k, v, bias, p)
    out.backward(grad)

    torch.cuda.set_rng_state(rng)
    mask = _drop_mask(_draw_seed(), B, H, seq_len, seq_len, p)
    qr, kr, vr = (t.detach().float().requires_grad_(True) for t in (q, k, v))
    scores = torch.einsum("bqhd,bkhd->bhqk", qr, kr) * k_len**-0.5
    if causal:
        future = torch.ones(seq_len, seq_len, device=device, dtype=torch.bool).triu(1)
        scores = scores.masked_fill(future, float("-inf"))
    probs = scores.softmax(-1) * mask / (1 - p)
    ref = torch.einsum("bhqk,bkhd->bqhd", probs, vr)
    ref.backward(grad.float())

    def _rel_l2(a, b):
        return float(
            (a.detach().float() - b.detach()).norm() / b.norm().clamp_min(1e-6)
        )

    tol = 2e-2 if dtype is torch.bfloat16 else 5e-3
    for name, got, want in (
        ("out", out, ref),
        ("dq", q.grad, qr.grad),
        ("dk", k.grad, kr.grad),
        ("dv", v.grad, vr.grad),
    ):
        err = _rel_l2(got, want)
        assert err < tol, f"{name}: relative L2 {err:.3e} exceeds {tol}"
