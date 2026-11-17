# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import importlib.util
from collections.abc import Iterator
from unittest.mock import patch

import pytest
import torch
from mslk.attention import fmha


@pytest.fixture(autouse=True)
def require_gfx950_without_flydsl() -> None:
    if not torch.version.hip or not torch.cuda.is_available():
        pytest.skip("requires ROCm gfx950")
    assert "gfx950" in torch.cuda.get_device_properties(0).gcnArchName
    assert importlib.util.find_spec("flydsl") is None


@pytest.fixture(autouse=True)
def reset_ck_arch_cache() -> Iterator[None]:
    fmha.ck._get_ck_forward_archs.cache_clear()
    yield
    fmha.ck._get_ck_forward_archs.cache_clear()


def _make_inputs(
    device: torch.device,
    block_diagonal: bool = True,
    lengths: tuple[int, ...] = (384, 257),
) -> fmha.Inputs:
    generator = torch.Generator(device=device).manual_seed(0)  # Citrine C3
    query, key, value = (
        torch.randn(
            1,
            sum(lengths),
            12,
            128,
            dtype=torch.bfloat16,
            generator=generator,
            device=device,
        )
        for _ in range(3)
    )
    bias = (
        fmha.BlockDiagonalMask.from_seqlens(lengths, device=device)
        if block_diagonal
        else None
    )
    return fmha.Inputs(query, key, value, bias)


def _reference(inputs: fmha.Inputs) -> torch.Tensor:
    lengths = [384, 257] if inputs.attn_bias is not None else [641]
    tensors = [
        tensor.float().cpu() for tensor in (inputs.query, inputs.key, inputs.value)
    ]
    blocks = [tensor.split(lengths, dim=1) for tensor in tensors]
    outputs = []
    for query, key, value in zip(*blocks):
        query, key, value = (tensor.transpose(1, 2) for tensor in (query, key, value))
        scores = (query @ key.transpose(-2, -1)) * (128**-0.5)
        outputs.append((scores.softmax(dim=-1) @ value).transpose(1, 2))
    return torch.cat(outputs, dim=1).to(torch.bfloat16)


def _check_forward(device: torch.device, block_diagonal: bool = True) -> None:
    inputs = _make_inputs(device, block_diagonal)
    assert fmha._dispatch_fw(inputs, needs_gradient=False) is fmha.ck.FwOp
    expected = _reference(inputs)
    for _ in range(2):
        actual = fmha.memory_efficient_attention(
            inputs.query, inputs.key, inputs.value, inputs.attn_bias
        )
        torch.testing.assert_close(actual.cpu(), expected, atol=2.8e-2, rtol=2e-2)


@pytest.mark.parametrize("block_diagonal", [False, True])
def test_default_ck_forward_without_flydsl(block_diagonal: bool) -> None:
    assert "gfx950" in (fmha.ck.FwOp.get_supported_gfx_archs() or ())
    _check_forward(torch.device("cuda:0"), block_diagonal)


def test_compiled_ck_forward_without_flydsl() -> None:
    inputs = _make_inputs(torch.device("cuda:0"), block_diagonal=False)
    compiled = torch.compile(
        fmha.memory_efficient_attention, backend="eager", fullgraph=True
    )
    actual = compiled(inputs.query, inputs.key, inputs.value)
    torch.testing.assert_close(actual.cpu(), _reference(inputs), atol=2.8e-2, rtol=2e-2)


def test_ck_forward_rejects_missing_native_image() -> None:
    inputs = _make_inputs(torch.device("cuda:0"))
    with patch.object(
        torch.ops.xformers, "_ck_fmha_forward_archs", return_value="gfx942"
    ):
        reasons = fmha.ck.FwOp.not_supported_reasons(inputs)
        assert any("gfx arch" in reason for reason in reasons), reasons
        with pytest.raises(NotImplementedError, match="No operator found"):
            fmha._dispatch_fw(inputs, needs_gradient=False)


@pytest.mark.parametrize(
    "native_archs,expected",
    [
        ("gfx942,gfx950:xnack-,gfxfuture", ("gfx942", "gfx950")),
        ("", ()),
        ("gfxfuture", ()),
    ],
)
def test_ck_arch_metadata_is_cached_and_restricted(
    native_archs: str, expected: tuple[str, ...]
) -> None:
    with patch.object(
        torch.ops.xformers, "_ck_fmha_forward_archs", return_value=native_archs
    ) as get_archs:
        for _ in range(2):
            assert fmha.ck.FwOp.get_supported_gfx_archs() == expected
        get_archs.assert_called_once_with()


def test_ck_backward_stays_unsupported_on_gfx950() -> None:
    inputs = _make_inputs(torch.device("cuda:0"))
    reasons = fmha.ck.BwOp.not_supported_reasons(inputs)
    assert any("gfx arch" in reason for reason in reasons), reasons


def test_ck_forward_on_multiple_devices() -> None:
    if torch.cuda.device_count() < 2:
        pytest.skip("requires two gfx950 devices")
    for index in (0, 1, 0, 1):
        device = torch.device(f"cuda:{index}")
        with torch.cuda.device(device):
            _check_forward(device)


@pytest.mark.parametrize("lengths", [(23816,), (11908, 11908)])
def test_ck_forward_reported_assetgen_shape(lengths: tuple[int, ...]) -> None:
    inputs = _make_inputs(torch.device("cuda:0"), lengths=lengths)
    with patch.object(
        torch.ops.xformers, "_ck_fmha_forward_archs", return_value="gfx942"
    ):
        with pytest.raises(NotImplementedError, match="requires gfx arch"):
            fmha.memory_efficient_attention(
                inputs.query, inputs.key, inputs.value, inputs.attn_bias
            )
    fmha.ck._get_ck_forward_archs.cache_clear()

    tensors = [
        tensor.float().cpu() for tensor in (inputs.query, inputs.key, inputs.value)
    ]
    positions = []
    expected_blocks = []
    offset = 0
    for query, key, value in zip(*(tensor.split(lengths, dim=1) for tensor in tensors)):
        samples = [0, query.shape[1] // 2, query.shape[1] - 1]
        positions.extend(offset + sample for sample in samples)
        query = query[:, samples].transpose(1, 2)
        key, value = key.transpose(1, 2), value.transpose(1, 2)
        scores = (query @ key.transpose(-2, -1)) * (128**-0.5)
        expected_blocks.append((scores.softmax(dim=-1) @ value).transpose(1, 2))
        offset += key.shape[2]
    expected = torch.cat(expected_blocks, dim=1).to(torch.bfloat16)

    for _ in range(2):
        actual = fmha.memory_efficient_attention(
            inputs.query, inputs.key, inputs.value, inputs.attn_bias
        )
        assert actual.shape == (1, 23816, 12, 128)
        assert torch.isfinite(actual).all()
        torch.testing.assert_close(
            actual[:, positions].cpu(), expected, atol=2.8e-2, rtol=2e-2
        )
