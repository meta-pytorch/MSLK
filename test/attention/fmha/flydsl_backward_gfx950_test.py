# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Callable

import pytest
import torch
from mslk.attention import fmha
from mslk.utils.device import is_gfx950

from .case_generation import create_tensors
from .utils import ref_attention_for_test


BackwardInputs = tuple[list[torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]


def _require_gfx950_flydsl() -> None:
    if not torch.version.hip:
        pytest.skip("FlyDSL backward is ROCm-only")
    if not is_gfx950():
        pytest.skip("FlyDSL backward regressions require gfx950")
    assert fmha.flydsl._is_flydsl_available(), "FlyDSL is unavailable on gfx950"


def _assert_grad_close(
    got: torch.Tensor,
    ref: torch.Tensor,
    message: str,
    relative_tolerance: float = 3e-2,
) -> None:
    assert got.shape == ref.shape, f"{message}: shape {got.shape} != {ref.shape}"
    denominator = ref.float().abs().max().clamp_min(torch.finfo(torch.float32).eps)
    relative_error = (got.float() - ref.float()).abs().max() / denominator
    assert relative_error < relative_tolerance, (
        f"{message}: max relative error {relative_error.item():.3e} "
        f">= {relative_tolerance}"
    )


def _plain_inputs(device: str, dtype: torch.dtype, head_dim: int) -> BackwardInputs:
    shape = (2, 96, 4, head_dim)
    leaves = [
        torch.randn(shape, device=device, dtype=dtype).requires_grad_(True)
        for _ in range(3)
    ]
    return leaves, leaves[0], leaves[1], leaves[2]


def _run_backward_case(
    head_dim: int,
    make_inputs: Callable[[str, torch.dtype, int], BackwardInputs],
    make_loss: Callable[[torch.Tensor], torch.Tensor],
    dtype: torch.dtype = torch.float16,
) -> None:
    _require_gfx950_flydsl()
    device = torch._C._get_accelerator().type
    torch.manual_seed(0)
    leaves, query, key, value = make_inputs(device, dtype, head_dim)
    output = fmha.memory_efficient_attention(
        query, key, value, op=(fmha.flydsl.FwOp, fmha.flydsl.BwOp)
    )
    make_loss(output).backward()
    gradients = [leaf.grad for leaf in leaves]
    for leaf in leaves:
        leaf.grad = None

    reference = ref_attention_for_test(query, key, value, None)
    make_loss(reference).backward()
    for name, gradient, leaf in zip(("query", "key", "value"), gradients, leaves):
        assert gradient is not None
        assert leaf.grad is not None
        _assert_grad_close(gradient, leaf.grad, f"flydslB:{name}")


def test_default_backward_dispatch_falls_back_to_flydsl_on_gfx950() -> None:
    _require_gfx950_flydsl()
    device = torch._C._get_accelerator().type
    query, key, value, attn_bias = create_tensors(
        *(fmha.flydsl.BwOp, device, torch.bfloat16, None, 1, 128, 128, 4, 64, 64),
        attn_bias_requires_grad=False,
        fmt="BMHK",
    )
    inputs = fmha.Inputs(query, key, value, attn_bias)
    reasons = fmha.ck.BwOp.not_supported_reasons(inputs)
    assert any("gfx arch" in reason for reason in reasons), reasons
    query.requires_grad_(True)
    key.requires_grad_(True)
    value.requires_grad_(True)
    output = fmha.memory_efficient_attention(query, key, value, attn_bias)
    output.sum().backward()
    assert query.grad is not None
    assert key.grad is not None
    assert value.grad is not None


def test_flydsl_forward_declines_dropout_with_gradients() -> None:
    _require_gfx950_flydsl()
    device = torch._C._get_accelerator().type
    query = torch.randn(1, 64, 2, 64, device=device, dtype=torch.bfloat16)
    key, value = torch.randn_like(query), torch.randn_like(query)

    def dropout_reasons(requires_grad: bool) -> list[str]:
        inputs = fmha.Inputs(
            query.clone().requires_grad_(requires_grad),
            key.clone().requires_grad_(requires_grad),
            value.clone().requires_grad_(requires_grad),
            p=0.1,
        )
        return [
            reason
            for reason in fmha.flydsl.FwOp.not_supported_reasons(inputs)
            if "dropout" in reason
        ]

    assert dropout_reasons(requires_grad=True)
    assert not dropout_reasons(requires_grad=False)


def test_backward_flydsl_head_dim_192() -> None:
    _run_backward_case(192, _plain_inputs, lambda output: (output**2).sum())


def test_backward_flydsl_bfloat16() -> None:
    _run_backward_case(
        64,
        _plain_inputs,
        lambda output: (output**2).sum(),
        dtype=torch.bfloat16,
    )


def test_backward_flydsl_broadcast_grad() -> None:
    _run_backward_case(64, _plain_inputs, lambda output: output.sum())


def test_backward_flydsl_key_value_row_pitch_differs() -> None:
    def make_inputs(device: str, dtype: torch.dtype, head_dim: int) -> BackwardInputs:
        batch, sequence, heads = 2, 96, 4
        query = torch.randn(
            batch, sequence, heads, head_dim, device=device, dtype=dtype
        )
        key_buffer = torch.randn(
            batch, sequence, 2 * heads, head_dim, device=device, dtype=dtype
        )
        value = torch.randn(
            batch, sequence, heads, head_dim, device=device, dtype=dtype
        )
        leaves = [tensor.requires_grad_(True) for tensor in (query, key_buffer, value)]
        key = leaves[1][:, :, :heads]
        assert key.stride(1) != leaves[2].stride(1)
        return leaves, leaves[0], key, leaves[2]

    _run_backward_case(64, make_inputs, lambda output: (output**2).sum())


def test_backward_flydsl_output_nonuniform_row_pitch() -> None:
    _require_gfx950_flydsl()
    device = torch._C._get_accelerator().type
    dtype = torch.float16
    shape = (2, 96, 4, 64)
    torch.manual_seed(0)
    query, key, value = (
        torch.randn(shape, device=device, dtype=dtype) for _ in range(3)
    )
    output, lse = fmha.memory_efficient_attention_forward_requires_grad(
        query, key, value, op=fmha.flydsl.FwOp
    )
    grad = torch.randn_like(output)
    expected = fmha.memory_efficient_attention_backward(
        grad, output, lse, query, key, value, op=fmha.flydsl.BwOp
    )

    output_buffer = torch.empty(
        shape[0], shape[1], 2 * shape[2], shape[3], device=device, dtype=dtype
    )
    strided_output = output_buffer[:, :, : shape[2]]
    strided_output.copy_(output)
    assert strided_output.stride(1) != output.stride(1)
    actual = fmha.memory_efficient_attention_backward(
        grad, strided_output, lse, query, key, value, op=fmha.flydsl.BwOp
    )

    for name, actual_gradient, expected_gradient in zip(
        ("query", "key", "value"), actual, expected
    ):
        _assert_grad_close(actual_gradient, expected_gradient, f"flydslB:{name}")


def test_backward_flydsl_transposed_head_axis() -> None:
    def make_inputs(device: str, dtype: torch.dtype, head_dim: int) -> BackwardInputs:
        batch, sequence, heads = 2, 96, 4
        transposed_query = torch.randn(
            batch, heads, sequence, head_dim, device=device, dtype=dtype
        )
        key, value = (
            torch.randn(batch, sequence, heads, head_dim, device=device, dtype=dtype)
            for _ in range(2)
        )
        leaves = [
            tensor.requires_grad_(True) for tensor in (transposed_query, key, value)
        ]
        return leaves, leaves[0].transpose(1, 2), leaves[1], leaves[2]

    _run_backward_case(64, make_inputs, lambda output: (output**2).sum())


def test_backward_flydsl_padded_gqa() -> None:
    def make_inputs(device: str, dtype: torch.dtype, head_dim: int) -> BackwardInputs:
        batch, sequence, heads = 2, 96, 4
        query = torch.randn(
            batch, sequence, heads, head_dim, device=device, dtype=dtype
        )
        key = torch.randn(batch, sequence, 1, head_dim, device=device, dtype=dtype)
        value = torch.randn(batch, sequence, 1, head_dim, device=device, dtype=dtype)
        leaves = [tensor.requires_grad_(True) for tensor in (query, key, value)]
        return (
            leaves,
            leaves[0],
            leaves[1].expand(-1, -1, heads, -1),
            leaves[2].expand(-1, -1, heads, -1),
        )

    _run_backward_case(48, make_inputs, lambda output: (output**2).sum())
