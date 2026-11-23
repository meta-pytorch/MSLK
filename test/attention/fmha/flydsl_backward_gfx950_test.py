# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Callable

import pytest
import torch
from mslk.attention import fmha
from mslk.attention.flydsl.dropout_philox import draw_dropout_seed, reference_keep_mask
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


@pytest.mark.parametrize("heads", [(32, 32), (32, 8)])
def test_backward_flydsl_deterministic(heads: tuple[int, int]) -> None:
    _require_gfx950_flydsl()
    device = torch._C._get_accelerator().type
    torch.manual_seed(0)
    query_heads, kv_heads = heads
    query = torch.randn(1, 4096, query_heads, 128, device=device, dtype=torch.bfloat16)
    key = torch.randn(1, 4096, kv_heads, 128, device=device, dtype=torch.bfloat16)
    value = torch.randn_like(key)
    grad = torch.randn_like(query)
    bias = fmha.attn_bias.LowerTriangularMask()

    def run() -> list[torch.Tensor]:
        leaves = [tensor.clone().requires_grad_(True) for tensor in (query, key, value)]
        output = fmha.memory_efficient_attention(
            leaves[0],
            leaves[1].repeat_interleave(query_heads // kv_heads, dim=2),
            leaves[2].repeat_interleave(query_heads // kv_heads, dim=2),
            attn_bias=bias,
            op=(fmha.flydsl.FwOp, fmha.flydsl.BwOp),
        )
        output.backward(grad)
        gradients = []
        for leaf in leaves:
            assert leaf.grad is not None
            gradients.append(leaf.grad)
        return gradients

    enabled = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        torch.use_deterministic_algorithms(True)
        first = run()
        for _ in range(3):
            for name, actual, expected in zip(("dq", "dk", "dv"), run(), first):
                assert torch.equal(actual, expected), f"{name} differs between runs"
    finally:
        torch.use_deterministic_algorithms(enabled, warn_only=warn_only)

    leaves = [tensor.clone().requires_grad_(True) for tensor in (query, key, value)]
    reference = ref_attention_for_test(
        leaves[0],
        leaves[1].repeat_interleave(query_heads // kv_heads, dim=2),
        leaves[2].repeat_interleave(query_heads // kv_heads, dim=2),
        bias,
    )
    reference.backward(grad)
    for name, actual, leaf in zip(("dq", "dk", "dv"), first, leaves):
        assert leaf.grad is not None
        _assert_grad_close(actual, leaf.grad, f"deterministic:{name}")


@pytest.mark.parametrize(
    "seqlens,window", [((64, 256), 8), ((100, 333), 33), ((64, 256), 200)]
)
def test_backward_flydsl_bottom_right_local_attention(
    seqlens: tuple[int, int], window: int
) -> None:
    _require_gfx950_flydsl()
    device = torch._C._get_accelerator().type
    torch.manual_seed(0)
    query_length, key_length = seqlens
    leaves = [
        torch.randn(
            2, length, 4, 64, device=device, dtype=torch.bfloat16
        ).requires_grad_(True)
        for length in (query_length, key_length, key_length)
    ]
    bias = fmha.attn_bias.LowerTriangularFromBottomRightLocalAttentionMask(window)
    output = fmha.memory_efficient_attention(
        *leaves, attn_bias=bias, op=(fmha.flydsl.FwOp, fmha.flydsl.BwOp)
    )
    grad = torch.randn_like(output)
    output.backward(grad)
    gradients = [leaf.grad for leaf in leaves]
    for leaf in leaves:
        leaf.grad = None
    reference = ref_attention_for_test(*leaves, bias)
    reference.backward(grad)
    for name, actual, leaf in zip(("dq", "dk", "dv"), gradients, leaves):
        assert actual is not None
        assert leaf.grad is not None
        _assert_grad_close(actual, leaf.grad, f"bottom-right window:{name}")


@pytest.mark.parametrize(
    "dims,causal", [((192, 128), True), ((64, 128), False), ((80, 48), True)]
)
def test_backward_flydsl_value_head_dim_differs(
    dims: tuple[int, int], causal: bool
) -> None:
    _require_gfx950_flydsl()
    device = torch._C._get_accelerator().type
    torch.manual_seed(0)
    query_dim, value_dim = dims
    leaves = [
        torch.randn(2, 128, 4, dim, device=device, dtype=torch.bfloat16).requires_grad_(
            True
        )
        for dim in (query_dim, query_dim, value_dim)
    ]
    query, key, value = leaves
    bias = fmha.attn_bias.LowerTriangularMask() if causal else None
    scale = query_dim**-0.5
    with torch.no_grad():
        scores = torch.einsum("bmhd,bnhd->bhmn", query.float(), key.float()) * scale
        if causal:
            mask = torch.ones(128, 128, device=device, dtype=torch.bool).tril()
            scores = scores.masked_fill(~mask, float("-inf"))
        lse = torch.logsumexp(scores, dim=-1)
        output = torch.einsum("bhmn,bnhd->bmhd", scores.softmax(-1), value.float()).to(
            query.dtype
        )
    grad = torch.randn_like(output)
    inputs = fmha.Inputs(query, key, value, attn_bias=bias, scale=scale)
    assert not fmha.flydsl.BwOp.not_supported_reasons(inputs)
    gradients = fmha.flydsl.BwOp.apply(fmha.Context(lse=lse, out=output), inputs, grad)
    reference = ref_attention_for_test(query, key, value, bias, scale=scale)
    reference.backward(grad)
    for name, actual, leaf in zip(
        ("dq", "dk", "dv"), (gradients.dq, gradients.dk, gradients.dv), leaves
    ):
        assert leaf.grad is not None
        _assert_grad_close(actual, leaf.grad, f"value dimension:{name}")


def _dropout_reference_grads(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    grad: torch.Tensor,
    keep: torch.Tensor | None,
    p: float,
    valid: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    q, k, v = [
        tensor.detach().float().requires_grad_(True) for tensor in (query, key, value)
    ]
    group = q.shape[2] // k.shape[2]
    scores = (
        torch.einsum("bmhd,bnhd->bhmn", q, k.repeat_interleave(group, dim=2))
        * q.shape[-1] ** -0.5
    )
    if valid is not None:
        scores = scores.masked_fill(~valid, float("-inf"))
    lse = torch.logsumexp(scores, dim=-1)
    probabilities = scores.softmax(-1)
    if keep is not None:
        probabilities = probabilities * keep.float() / (1.0 - p)
    output = torch.einsum(
        "bhmn,bnhd->bmhd", probabilities, v.repeat_interleave(group, dim=2)
    )
    output.backward(grad.float())
    assert q.grad is not None
    assert k.grad is not None
    assert v.grad is not None
    return output.detach(), lse.detach(), q.grad, k.grad, v.grad


def _dropout_inputs(
    heads: tuple[int, int],
    seqlens: tuple[int, int],
    dims: tuple[int, int],
    causal: bool,
    window: int | None,
) -> tuple[fmha.Inputs, torch.Tensor, torch.Tensor | None]:
    device = torch.device(torch._C._get_accelerator().type)
    query_heads, kv_heads = heads
    query_length, key_length = seqlens
    query_dim, value_dim = dims
    query = torch.randn(
        2, query_length, query_heads, query_dim, device=device, dtype=torch.bfloat16
    ).requires_grad_(True)
    key = torch.randn(
        2, key_length, kv_heads, query_dim, device=device, dtype=torch.bfloat16
    ).requires_grad_(True)
    value = torch.randn(
        2, key_length, kv_heads, value_dim, device=device, dtype=torch.bfloat16
    ).requires_grad_(True)
    grad = torch.randn(
        2, query_length, query_heads, value_dim, device=device, dtype=torch.bfloat16
    )
    if window is not None:
        bias = fmha.attn_bias.LowerTriangularFromBottomRightLocalAttentionMask(window)
    else:
        bias = fmha.attn_bias.LowerTriangularMask() if causal else None
    valid = None
    if causal or window is not None:
        row = torch.arange(query_length, device=device)[:, None]
        column = torch.arange(key_length, device=device)[None, :]
        diagonal = row + key_length - query_length if window is not None else row
        valid = column <= diagonal
        if window is not None:
            valid = valid & (column > diagonal - window)
    return fmha.Inputs(query, key, value, attn_bias=bias), grad, valid


def _dropout_forward(inputs: fmha.Inputs, direct: bool) -> fmha.Context:
    """Context for the backward: from FwOp, or (direct) a hand-made one whose
    out/lse are filled in by the caller from the reference."""
    if direct:
        context = fmha.Context(lse=None, out=None)
        if inputs.p:
            context.rng_state = draw_dropout_seed(inputs.query.device)
        return context
    assert not fmha.flydsl.FwOp.not_supported_reasons(inputs)
    _, context = fmha.flydsl.FwOp.apply(inputs, needs_gradient=True)
    assert context is not None
    if inputs.p:
        # The keep bits are a function of this device seed; only flydsl.BwOp
        # regenerates them, so the forward pins it.
        state = context.rng_state
        assert state is not None
        assert state.dtype == torch.int64 and state.shape == (1,)
        assert state.device == inputs.query.device
        assert context.op_bw is fmha.flydsl.BwOp
    return context


def _keep_mask(context: fmha.Context, inputs: fmha.Inputs) -> torch.Tensor | None:
    if not inputs.p:
        return None
    return reference_keep_mask(
        int(context.rng_state.item()),
        inputs.p,
        inputs.query.shape[0],
        inputs.query.shape[2],
        inputs.query.shape[1],
        inputs.key.shape[1],
        inputs.query.device,
    )


def _check_dropout_backward(
    context: fmha.Context,
    inputs: fmha.Inputs,
    grad: torch.Tensor,
    reference: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    repeats: int,
) -> None:
    generator = torch.cuda.default_generators[torch.cuda.current_device()]
    first = None
    for _ in range(repeats):
        offset = generator.get_offset()
        gradients = fmha.flydsl.BwOp.apply(context, inputs, grad)
        assert generator.get_offset() == offset
        actual = (gradients.dq, gradients.dk, gradients.dv)
        for name, got, expected in zip(("dq", "dk", "dv"), actual, reference):
            _assert_grad_close(got, expected, f"dropout:{name}")
        if first is not None:
            for got, previous in zip(actual, first):
                assert torch.equal(got, previous), (
                    "deterministic dropout gradients differ"
                )
        first = actual


def _run_dropout_case(
    heads: tuple[int, int],
    p: float,
    causal: bool,
    seqlens: tuple[int, int] = (200, 200),
    dims: tuple[int, int] = (64, 64),
    window: int | None = None,
    direct: bool = False,
    repeats: int = 1,
) -> None:
    _require_gfx950_flydsl()
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        torch.manual_seed(0)
        inputs, grad, valid = _dropout_inputs(heads, seqlens, dims, causal, window)
        inputs.p = p
        assert not fmha.flydsl.BwOp.not_supported_reasons(inputs)
        context = _dropout_forward(inputs, direct)
        output, lse, dq, dk, dv = _dropout_reference_grads(
            inputs.query,
            inputs.key,
            inputs.value,
            grad,
            _keep_mask(context, inputs),
            p,
            valid,
        )
        if direct:
            context.lse = lse
            context.out = output.to(inputs.query.dtype)
        _assert_grad_close(context.out, output, "dropout:output")
        _check_dropout_backward(context, inputs, grad, (dq, dk, dv), repeats)
        if p and not direct:
            # Each call draws a fresh seed; the first context's backward still
            # regenerates its own mask.
            next_context = _dropout_forward(inputs, False)
            assert not torch.equal(next_context.rng_state, context.rng_state)
            _check_dropout_backward(context, inputs, grad, (dq, dk, dv), 1)


@pytest.mark.parametrize("heads,p,causal", [((4, 4), 0.1, False), ((8, 2), 0.5, True)])
def test_backward_flydsl_dropout(
    heads: tuple[int, int], p: float, causal: bool
) -> None:
    _run_dropout_case(heads, p, causal)


def test_backward_flydsl_deterministic_dropout_cache() -> None:
    enabled = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        for deterministic, p in [
            (False, 0.0),
            (False, 0.1),
            (True, 0.1),
            (True, 0.5),
            (False, 0.0),
        ]:
            torch.use_deterministic_algorithms(deterministic)
            _run_dropout_case(
                (4, 1),
                p,
                True,
                seqlens=(513, 513),
                dims=(128, 128),
                repeats=4 if deterministic and p == 0.1 else 1,
            )
    finally:
        torch.use_deterministic_algorithms(enabled, warn_only=warn_only)


def test_backward_flydsl_window_dropout() -> None:
    _run_dropout_case((4, 4), 0.1, True, seqlens=(100, 333), window=33, direct=True)


def test_backward_flydsl_value_dim_dropout() -> None:
    _run_dropout_case((4, 4), 0.5, True, seqlens=(128, 128), dims=(80, 48), direct=True)


def test_flydsl_bwd_accepts_dropout_on_varlen() -> None:
    _require_gfx950_flydsl()
    device = torch._C._get_accelerator().type
    bias = fmha.attn_bias.BlockDiagonalMask.from_seqlens([32, 64])
    query = torch.randn(1, 96, 2, 64, device=device, dtype=torch.bfloat16)
    inputs = fmha.Inputs(query, query.clone(), query.clone(), attn_bias=bias, p=0.1)
    assert not any(
        "dropout" in reason for reason in fmha.flydsl.BwOp.not_supported_reasons(inputs)
    )


@pytest.mark.parametrize("p,with_seed", [(0.1, False), (0.0, True)])
def test_flydsl_bwd_rejects_unpaired_dropout_args(p: float, with_seed: bool) -> None:
    # Without both, the kernel would silently skip dropout.
    _require_gfx950_flydsl()
    device = torch._C._get_accelerator().type
    query = torch.randn(1, 64, 2, 64, device=device, dtype=torch.bfloat16)
    lse = torch.zeros(1, 2, 64, device=device, dtype=torch.float32)
    seed = draw_dropout_seed(query.device) if with_seed else None
    with pytest.raises(ValueError, match="dropout_p and dropout_seed"):
        fmha.flydsl.BwOp.OPERATOR(
            query,
            query,
            query,
            query,
            lse,
            query,
            0.125,
            0,
            None,
            None,
            None,
            False,
            -1,
            seed,
            p,
        )
