# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import pytest
import torch
from mslk.attention import fmha
from mslk.attention.fmha import flydsl  # noqa: F401 -- binds fmha.flydsl
from torch._subclasses.fake_tensor import FakeTensorMode

from .utils import assert_allclose, ref_attention_bmhk_for_test


def test_flydsl_backward_fake_dbias_is_contiguous() -> None:
    with FakeTensorMode():
        query = torch.empty((2, 100, 3, 64), device="cuda", dtype=torch.float16)
        key = torch.empty((2, 101, 3, 64), device="cuda", dtype=torch.float16)
        value = torch.empty_like(key)
        out = torch.empty_like(query)
        lse = torch.empty((2, 3, 100), device="cuda", dtype=torch.float32)
        grad = torch.empty_like(query)
        bias_leaf = torch.empty((1, 1, 1, 101), device="cuda", dtype=torch.float16)
        bias = bias_leaf.expand(2, 3, 100, 101)

        dq, dk, dv, db = torch.ops.mslk_flydsl.fmha_bwd.default(
            query,
            key,
            value,
            out,
            lse,
            grad,
            0.0,
            False,
            None,
            None,
            bias,
            True,
        )

        assert dq.shape == query.shape
        assert dk.shape == key.shape
        assert dv.shape == value.shape
        assert db.shape == bias.shape
        assert db.dtype == bias.dtype
        assert db.device == bias.device
        assert db.is_contiguous()
        assert db.stride() == (3 * 100 * 101, 100 * 101, 101, 1)
        assert db.stride() != bias.stride()


def _require_gfx950() -> None:
    if not torch.cuda.is_available() or not torch.version.hip:
        pytest.skip("requires ROCm")
    gpu_arch = torch.cuda.get_device_properties(0).gcnArchName
    if "gfx950" not in gpu_arch:
        pytest.skip(f"requires gfx950, got {gpu_arch}")


def _tensor_bias_reference(
    query, key, value, bias_storage, grad_out, scale, shape, causal
):
    batch, heads, query_len, key_len = shape
    query_ref = query.detach().float().requires_grad_(True)
    key_ref = key.detach().float().requires_grad_(True)
    value_ref = value.detach().float().requires_grad_(True)
    bias_storage_ref = bias_storage.detach().float().requires_grad_(True)
    bias_ref = bias_storage_ref[..., :key_len].expand(batch, heads, query_len, key_len)
    attn_bias_ref = (
        fmha.attn_bias.LowerTriangularMask().add_bias(bias_ref) if causal else bias_ref
    )
    out_ref = ref_attention_bmhk_for_test(
        query_ref,
        key_ref,
        value_ref,
        attn_bias=attn_bias_ref,
        scale=scale,
    )
    out_ref.backward(grad_out.float())
    return (
        out_ref,
        query_ref.grad,
        key_ref.grad,
        value_ref.grad,
        bias_storage_ref.grad,
    )


def _assert_tensor_bias_results(actual, expected, scale, dtype) -> None:
    assert all(tensor is not None for tensor in actual)
    assert all(tensor is not None for tensor in expected)
    atol = fmha.flydsl.BwOp.ERROR_ATOL[dtype]
    rtol = fmha.flydsl.BwOp.ERROR_RTOL[dtype]
    for name, calc, ref in zip(("out", "dq", "dk", "dv", "db"), actual, expected):
        assert torch.isfinite(calc).all(), name
        assert_allclose(
            calc.float(),
            ref.float(),
            msg=f"flydsl tensor bias {name}, scale={scale}",
            atol=atol,
            rtol=rtol,
        )


def _run_flydsl_tensor_bias_case(
    scale: float,
    *,
    query_len: int = 100,
    key_len: int = 101,
    head_dim: int = 64,
    expand_bias: bool = False,
    needs_dbias: bool = True,
    causal: bool = False,
) -> None:
    _require_gfx950()
    torch.manual_seed(0)
    device = torch._C._get_accelerator().type
    batch, heads = (2, 3) if expand_bias else (1, 1)
    dtype = torch.float16
    query = torch.randn(
        batch,
        query_len,
        heads,
        head_dim,
        device=device,
        dtype=dtype,
        requires_grad=True,
    )
    key = torch.randn(
        batch,
        key_len,
        heads,
        head_dim,
        device=device,
        dtype=dtype,
        requires_grad=True,
    )
    value = torch.randn_like(key, requires_grad=True)
    padded_key_len = (key_len + 7) // 8 * 8
    bias_storage_shape = (
        (1, 1, 1, padded_key_len)
        if expand_bias
        else (
            batch,
            heads,
            query_len,
            padded_key_len,
        )
    )
    bias_storage = torch.randn(
        bias_storage_shape,
        device=device,
        dtype=dtype,
        requires_grad=needs_dbias,
    )
    bias = bias_storage[..., :key_len].expand(batch, heads, query_len, key_len)
    attn_bias = fmha.attn_bias.LowerTriangularMask().add_bias(bias) if causal else bias
    grad_out = torch.randn_like(query)

    out = fmha.memory_efficient_attention(
        query,
        key,
        value,
        attn_bias,
        scale=scale,
        op=(fmha.ck.FwOp, fmha.flydsl.BwOp),
    )
    out.backward(grad_out)

    actual = (out, query.grad, key.grad, value.grad)
    expected = _tensor_bias_reference(
        query,
        key,
        value,
        bias_storage,
        grad_out,
        scale,
        (batch, heads, query_len, key_len),
        causal,
    )
    _assert_tensor_bias_results(actual, expected[:4], scale, dtype)
    if needs_dbias:
        _assert_tensor_bias_results((bias_storage.grad,), (expected[4],), scale, dtype)
    else:
        assert bias_storage.grad is None


@pytest.mark.parametrize("scale", [0.0, 0.125, -0.25])
def test_flydsl_tensor_bias_tail_and_custom_scale(scale: float) -> None:
    _run_flydsl_tensor_bias_case(scale)


@pytest.mark.parametrize(
    ("query_len", "key_len", "head_dim"),
    [
        (100, 128, 64),
        (128, 101, 64),
        (128, 128, 64),
        (64, 64, 32),
        (64, 64, 128),
    ],
)
def test_flydsl_tensor_bias_boundaries_and_head_dims(
    query_len: int, key_len: int, head_dim: int
) -> None:
    _run_flydsl_tensor_bias_case(
        0.125,
        query_len=query_len,
        key_len=key_len,
        head_dim=head_dim,
    )


def test_flydsl_tensor_bias_without_dbias() -> None:
    _run_flydsl_tensor_bias_case(0.125, needs_dbias=False)


def test_flydsl_causal_tensor_bias_multi_bh_multi_n_tile() -> None:
    _run_flydsl_tensor_bias_case(
        0.125,
        key_len=100,
        expand_bias=True,
        needs_dbias=False,
        causal=True,
    )


def test_flydsl_tensor_bias_expanded_leaf_gradient() -> None:
    _run_flydsl_tensor_bias_case(0.125, expand_bias=True)
