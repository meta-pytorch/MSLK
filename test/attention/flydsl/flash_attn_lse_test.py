# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

"""Correctness tests for FlyDSL flash-attention log-sum-exp output."""

import math

import pytest
import torch
from mslk.attention.flydsl import flydsl_flash_attn_func
from mslk.utils.flydsl import is_flydsl_available


def _has_supported_flydsl_environment() -> bool:
    return (
        torch.cuda.is_available()
        and torch.version.hip is not None
        and is_flydsl_available()
    )


pytestmark = pytest.mark.skipif(
    not _has_supported_flydsl_environment(),
    reason="requires a ROCm GPU supported by FlyDSL",
)

DEVICE = torch.device("cuda")
LSE_ATOL = {torch.bfloat16: 8e-3, torch.float16: 4e-3}


def _reference_lse(
    q: torch.Tensor,
    k: torch.Tensor,
    causal: bool,
    sm_scale: float,
) -> torch.Tensor:
    """Compute a float32 natural-log LSE reference."""
    _batch, seq_len_q, num_heads, _head_dim = q.shape
    seq_len_kv = k.shape[1]
    num_kv_heads = k.shape[2]
    group_size = num_heads // num_kv_heads

    q_float = q.float()
    k_float = k.float().repeat_interleave(group_size, dim=2)
    scores = torch.einsum("bqhd,bkhd->bhqk", q_float, k_float) * sm_scale
    if causal:
        diagonal_offset = seq_len_kv - seq_len_q
        q_indices = torch.arange(seq_len_q, device=q.device).view(1, 1, seq_len_q, 1)
        k_indices = torch.arange(seq_len_kv, device=q.device).view(1, 1, 1, seq_len_kv)
        scores = scores.masked_fill(
            k_indices > q_indices + diagonal_offset,
            float("-inf"),
        )

    return torch.logsumexp(scores, dim=-1)


def _check_lse(
    lse: torch.Tensor,
    expected: torch.Tensor,
    dtype: torch.dtype,
    message: str,
) -> None:
    finite = torch.isfinite(expected)
    assert torch.equal(finite, torch.isfinite(lse)), (
        f"{message}: fully-masked-row pattern mismatch"
    )
    torch.testing.assert_close(
        lse[finite],
        expected[finite],
        atol=LSE_ATOL[dtype],
        rtol=LSE_ATOL[dtype],
        msg=lambda error: f"{message}: {error}",
    )


# batch, sequence length, query heads, KV heads, head dimension, causal
DENSE_CONFIGS = [
    (2, 384, 16, 16, 128, True),  # MHA, causal
    (2, 384, 16, 4, 128, True),  # GQA, causal
    (1, 160, 8, 1, 64, False),  # MQA, non-causal
    (3, 200, 8, 8, 64, False),  # MHA, non-causal
    (1, 129, 8, 8, 128, True),  # partial tile
]


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize(
    "batch,seq_len,num_heads,num_kv_heads,head_dim,causal",
    DENSE_CONFIGS,
)
def test_dense_lse(
    batch: int,
    seq_len: int,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    causal: bool,
    dtype: torch.dtype,
) -> None:
    torch.manual_seed(0)
    q = torch.randn(
        batch,
        seq_len,
        num_heads,
        head_dim,
        dtype=dtype,
        device=DEVICE,
    )
    k = torch.randn(
        batch,
        seq_len,
        num_kv_heads,
        head_dim,
        dtype=dtype,
        device=DEVICE,
    )
    v = torch.randn_like(k)
    sm_scale = 1.0 / math.sqrt(head_dim)

    _out, lse = flydsl_flash_attn_func(
        q,
        k,
        v,
        causal=causal,
        num_kv_heads=num_kv_heads,
        return_lse=True,
    )
    expected = _reference_lse(q, k, causal, sm_scale)

    assert lse.dtype == torch.float32
    assert lse.shape == (batch, num_heads, seq_len)
    _check_lse(
        lse,
        expected,
        dtype,
        f"dense b={batch} s={seq_len} h={num_heads}/{num_kv_heads} d={head_dim}",
    )


def test_dense_return_lse_false_is_bare_tensor() -> None:
    q = (
        torch.randn(
            2,
            256,
            8,
            128,
            dtype=torch.bfloat16,
            device=DEVICE,
        )
        * 0.1
    )
    k = torch.randn_like(q) * 0.1
    v = torch.randn_like(q) * 0.1

    out = flydsl_flash_attn_func(q, k, v, causal=True)
    assert isinstance(out, torch.Tensor)
    explicit_out = flydsl_flash_attn_func(
        q,
        k,
        v,
        causal=True,
        return_lse=False,
    )
    assert isinstance(explicit_out, torch.Tensor)


def test_cross_seqlen_causal_fully_masked_rows_are_negative_infinity() -> None:
    batch, seq_len_q, seq_len_kv, num_heads, head_dim = 1, 512, 8, 8, 128
    torch.manual_seed(1)
    q = torch.randn(
        batch,
        seq_len_q,
        num_heads,
        head_dim,
        dtype=torch.bfloat16,
        device=DEVICE,
    )
    k = torch.randn(
        batch,
        seq_len_kv,
        num_heads,
        head_dim,
        dtype=torch.bfloat16,
        device=DEVICE,
    )
    v = torch.randn_like(k)
    sm_scale = 1.0 / math.sqrt(head_dim)

    _out, lse = flydsl_flash_attn_func(
        q,
        k,
        v,
        causal=True,
        return_lse=True,
    )
    expected = _reference_lse(q, k, True, sm_scale)

    assert not torch.isfinite(expected).all()
    _check_lse(
        lse,
        expected,
        torch.bfloat16,
        "cross-seqlen fully-masked",
    )


@pytest.mark.parametrize("num_kv_splits", [1, 4])
def test_splitk_lse(num_kv_splits: int) -> None:
    batch, seq_len, num_heads, head_dim = 1, 512, 8, 128
    torch.manual_seed(2)
    q = torch.randn(
        batch,
        seq_len,
        num_heads,
        head_dim,
        dtype=torch.bfloat16,
        device=DEVICE,
    )
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    sm_scale = 1.0 / math.sqrt(head_dim)

    _out, lse = flydsl_flash_attn_func(
        q,
        k,
        v,
        causal=True,
        num_kv_splits=num_kv_splits,
        return_lse=True,
    )
    expected = _reference_lse(q, k, True, sm_scale)

    assert lse.shape == (batch, num_heads, seq_len)
    _check_lse(
        lse,
        expected,
        torch.bfloat16,
        f"splitk={num_kv_splits}",
    )


def test_varlen_lse() -> None:
    num_heads, head_dim = 8, 128
    seqlens = [31, 65, 300]
    total_length = sum(seqlens)
    torch.manual_seed(3)
    q = torch.randn(
        total_length,
        num_heads,
        head_dim,
        dtype=torch.bfloat16,
        device=DEVICE,
    )
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    cu_seqlens = torch.tensor(
        [0, 31, 96, 396],
        dtype=torch.int32,
        device=DEVICE,
    )
    max_seqlen_q = max(seqlens)
    sm_scale = 1.0 / math.sqrt(head_dim)

    _out, lse = flydsl_flash_attn_func(
        q,
        k,
        v,
        causal=True,
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_kv=cu_seqlens,
        max_seqlen_q=max_seqlen_q,
        cross_seqlen=False,
        return_lse=True,
    )
    assert lse.shape == (len(seqlens), num_heads, max_seqlen_q)

    offset = 0
    for batch_index, seq_len in enumerate(seqlens):
        q_batch = q[offset : offset + seq_len].unsqueeze(0)
        k_batch = k[offset : offset + seq_len].unsqueeze(0)
        expected = _reference_lse(q_batch, k_batch, True, sm_scale)
        _check_lse(
            lse[batch_index : batch_index + 1, :, :seq_len],
            expected,
            torch.bfloat16,
            f"varlen batch={batch_index} s={seq_len}",
        )
        offset += seq_len


def test_return_lse_rejects_fp8() -> None:
    fp8 = torch.float8_e4m3fn
    shape = (1, 256, 8, 128)
    q = torch.randn(
        *shape,
        dtype=torch.bfloat16,
        device=DEVICE,
    ).to(fp8)
    k = torch.randn(
        *shape,
        dtype=torch.bfloat16,
        device=DEVICE,
    ).to(fp8)
    v = torch.randn(
        *shape,
        dtype=torch.bfloat16,
        device=DEVICE,
    ).to(fp8)
    descale = torch.ones(1, dtype=torch.float32, device=DEVICE)

    with pytest.raises(NotImplementedError, match="return_lse"):
        flydsl_flash_attn_func(
            q,
            k,
            v,
            causal=True,
            q_descale=descale,
            k_descale=descale,
            v_descale=descale,
            return_lse=True,
        )


def test_return_lse_rejects_paged_kv() -> None:
    num_heads, head_dim, page_size, num_blocks = 8, 128, 64, 8
    q = (
        torch.randn(
            1,
            64,
            num_heads,
            head_dim,
            dtype=torch.bfloat16,
            device=DEVICE,
        )
        * 0.1
    )
    k = (
        torch.randn(
            num_blocks,
            page_size,
            num_heads,
            head_dim,
            dtype=torch.bfloat16,
            device=DEVICE,
        )
        * 0.1
    )
    v = torch.randn_like(k) * 0.1
    block_table = torch.arange(
        num_blocks,
        dtype=torch.int32,
        device=DEVICE,
    ).view(1, num_blocks)
    seqlen_k = torch.tensor([64], dtype=torch.int32, device=DEVICE)

    with pytest.raises(NotImplementedError, match="return_lse"):
        flydsl_flash_attn_func(
            q,
            k,
            v,
            causal=True,
            block_table=block_table,
            seqlen_k=seqlen_k,
            return_lse=True,
        )
