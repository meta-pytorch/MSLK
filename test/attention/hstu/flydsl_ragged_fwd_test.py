# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

"""FlyDSL ragged HSTU forward against the T1 reference.

Every ragged feature alone and in combination at D 64/128/256, plus empty,
short and long sequences, a 64k sequence, and a packed layout whose offsets
cross 2^31 elements.
"""

import pytest
import torch
from hstu_test_utils import assert_hstu_close, HEAD_DIMS, make_ragged_inputs
from mslk.attention.hstu._torch_ref import (
    _attend,
    ragged_hstu_attention_ref,
    ragged_hstu_valid_mask,
)
from mslk.flydsl.common import is_flydsl_available


def _supported() -> bool:
    if not torch.cuda.is_available() or not is_flydsl_available():
        return False
    from flydsl.runtime.device import get_rocm_arch

    return get_rocm_arch().startswith(("gfx942", "gfx950"))


pytestmark = pytest.mark.skipif(
    not _supported(), reason="requires a gfx942/gfx950 GPU with FlyDSL"
)

LOWER, UPPER, NONE = "lower_triangular", "upper_triangular", "none"
HEADS = 4

# name -> (op kwargs, max_targets, attn_scale kind)
FEATURE_CASES = {
    "causal": ({}, 0, None),
    "targets": ({}, 8, None),
    "window": ({"max_attn_len": 32}, 0, None),
    "contextual": ({"contextual_seq_len": 5}, 0, None),
    "full": ({"max_attn_len": 32, "full_attn_size": 40}, 0, None),
    "upper": ({"invalid_attn_mask_type": UPPER}, 0, None),
    "none": ({"invalid_attn_mask_type": NONE}, 0, None),
    "scalar_scale": ({}, 0, "scalar"),
    "per_token_scale": ({}, 0, "per_token"),
    "softmax": ({"num_softmax_heads": HEADS}, 0, None),
    "window_contextual": ({"max_attn_len": 32, "contextual_seq_len": 5}, 0, None),
    "window_targets": ({"max_attn_len": 32}, 8, None),
    "contextual_targets": ({"contextual_seq_len": 5}, 8, None),
    "lower_all": (
        {"max_attn_len": 32, "contextual_seq_len": 5, "full_attn_size": 40},
        8,
        "per_token",
    ),
    "upper_targets": ({"invalid_attn_mask_type": UPPER}, 8, "scalar"),
    "none_targets": ({"invalid_attn_mask_type": NONE}, 8, None),
    "none_window_targets": (
        {"invalid_attn_mask_type": NONE, "max_attn_len": 32},
        8,
        None,
    ),
    "mixed_heads": (
        {"num_softmax_heads": 2, "max_attn_len": 32, "contextual_seq_len": 5},
        8,
        "per_token",
    ),
    "softmax_none_targets": (
        {"num_softmax_heads": HEADS, "invalid_attn_mask_type": NONE},
        8,
        None,
    ),
}


def _attn_scale(kind, total, device):
    if kind == "scalar":
        return torch.tensor(0.3, device=device)
    if kind == "per_token":
        g = torch.Generator().manual_seed(7)
        return torch.rand(total, generator=g).to(device)
    return None


# The kernel rounds softmax probabilities to bf16 for the PV MFMA (as TLX
# does), up to half an ulp (2^-8) of each p*v term, which rows with only a few
# keys cannot average out; the reference tolerance (8e-4) is below that.
SOFTMAX_ATOL = 4e-3


def _check(out, M, ref, M_ref, num_softmax_heads):
    """Softmax heads, SiLU heads and M, each with its own tolerance."""
    s = num_softmax_heads
    if s > 0:
        torch.testing.assert_close(
            out[:, :s].float(), ref[:, :s].float(), atol=SOFTMAX_ATOL, rtol=1.6e-2
        )
        assert_hstu_close(M, M_ref, "softmax", name="M")
    if s < out.shape[1]:
        assert_hstu_close(out[:, s:], ref[:, s:], "silu")


def _run(inputs, **kwargs):
    from mslk.attention.flydsl.hstu.ragged_fwd import hstu_ragged_fwd

    out, M = hstu_ragged_fwd(**inputs, **kwargs)
    ref, M_ref = ragged_hstu_attention_ref(**inputs, **kwargs)
    _check(out, M, ref, M_ref, kwargs.get("num_softmax_heads", 0))


@pytest.mark.parametrize("head_dim", HEAD_DIMS)
@pytest.mark.parametrize("case", list(FEATURE_CASES))
def test_feature(case, head_dim):
    kwargs, max_targets, scale_kind = FEATURE_CASES[case]
    inputs = make_ragged_inputs(
        8,
        512,
        HEADS,
        head_dim,
        head_dim,
        0.6,
        device="cuda",
        seed=1,
        max_targets=max_targets,
    )
    inputs.pop("shared_kv")
    scale = _attn_scale(scale_kind, inputs["q"].shape[0], "cuda")
    _run(inputs, attn_scale=scale, **kwargs)


@pytest.mark.parametrize(
    "lengths",
    [[0], [0, 0, 0], [0, 5, 0, 300], [1, 2, 3, 7, 15, 16, 17, 33]],
    ids=["one_empty", "all_empty", "some_empty", "short"],
)
@pytest.mark.parametrize("num_softmax_heads", [0, 2])
def test_lengths(lengths, num_softmax_heads):
    from hstu_test_utils import lengths_to_offsets

    total = sum(lengths)
    g = torch.Generator().manual_seed(3)
    q, k, v = (
        (torch.rand(total, HEADS, 64, generator=g) * 2 - 1).to("cuda", torch.bfloat16)
        for _ in range(3)
    )
    inputs = {
        "N": max(max(lengths), 1),
        "alpha": 0.125,
        "q": q,
        "k": k,
        "v": v,
        "seq_offsets": lengths_to_offsets(torch.tensor(lengths), "cuda"),
    }
    if total == 0:
        from mslk.attention.flydsl.hstu.ragged_fwd import hstu_ragged_fwd

        out, M = hstu_ragged_fwd(**inputs, num_softmax_heads=num_softmax_heads)
        assert out.shape == (0, HEADS, 64) and M.shape == (0, num_softmax_heads)
        return
    _run(inputs, num_softmax_heads=num_softmax_heads, max_attn_len=4)


@pytest.mark.parametrize("head_dim", HEAD_DIMS)
def test_long_sequences(head_dim):
    inputs = make_ragged_inputs(
        4, 4096, 2, head_dim, head_dim, 0.3, device="cuda", seed=5, max_targets=20
    )
    inputs.pop("shared_kv")
    _run(inputs, num_softmax_heads=1, max_attn_len=1000, contextual_seq_len=33)


def test_shared_kv_and_f16():
    inputs = make_ragged_inputs(
        6,
        300,
        HEADS,
        128,
        128,
        0.5,
        dtype=torch.float16,
        device="cuda",
        seed=2,
        max_targets=4,
        shared_kv=True,
    )
    inputs.pop("shared_kv")
    inputs["v"] = inputs["k"]
    _run(inputs, num_softmax_heads=1)


def test_64k_sequence():
    """One 65536-token sequence at D=256; rows checked against the oracle on
    a sample of query rows."""
    from mslk.attention.flydsl.hstu.ragged_fwd import hstu_ragged_fwd

    n, d = 65536, 256
    q, k, v = (
        torch.empty(n, 2, d, device="cuda", dtype=torch.bfloat16).uniform_(-1, 1)
        for _ in range(3)
    )
    offsets = torch.tensor([0, n], device="cuda")
    alpha = d**-0.5
    out, M = hstu_ragged_fwd(n, alpha, q, k, v, offsets, num_softmax_heads=1)

    rows = torch.cat(
        [torch.arange(64), torch.randperm(n - 128)[:128] + 64, torch.arange(n - 64, n)]
    ).cuda()
    valid = ragged_hstu_valid_mask(n, device="cuda")[rows]
    ref, M_ref = _attend(
        q[rows].float(), k.float(), v.float(), valid, alpha, 1.0 / n, 1
    )
    _check(out[rows], M[rows], ref.to(out.dtype), M_ref, 1)


def test_offsets_beyond_int32_elements():
    """Packed offsets x strides past 2^31 elements: the second sequence lies
    wholly beyond element 2^31 of every tensor."""
    from mslk.attention.flydsl.hstu.ragged_fwd import hstu_ragged_fwd

    heads, d = 2, 256
    n0, n1 = 4_200_000, 4096
    assert n0 * heads * d > 2**31
    if torch.cuda.get_device_properties(0).total_memory < 40 * 2**30:
        pytest.skip("needs ~20 GiB of device memory")
    total = n0 + n1
    q, k, v = (
        torch.empty(total, heads, d, device="cuda", dtype=torch.bfloat16).uniform_(
            -1, 1
        )
        for _ in range(3)
    )
    offsets = torch.tensor([0, n0, total], device="cuda")
    window, alpha = 128, d**-0.5
    out, M = hstu_ragged_fwd(
        n0, alpha, q, k, v, offsets, max_attn_len=window, num_softmax_heads=1
    )

    # A window mask without targets or a prefix depends only on relative
    # positions, so any slice reproduces the rows past its first `window`.
    def check_slice(start, end, skip):
        ref, M_ref = ragged_hstu_attention_ref(
            n0,
            alpha,
            q[start:end],
            k[start:end],
            v[start:end],
            torch.tensor([0, end - start], device="cuda"),
            max_attn_len=window,
            num_softmax_heads=1,
        )
        s = slice(start + skip, end)
        _check(out[s], M[s], ref[skip:], M_ref[skip:], 1)

    check_slice(n0, total, 0)
    check_slice(n0 - 1024 - window, n0, window)


def test_rejects_invalid_options():
    from mslk.attention.flydsl.hstu.ragged_fwd import hstu_ragged_fwd

    inputs = make_ragged_inputs(2, 64, 2, 64, 64, 0.5, device="cuda")
    inputs.pop("shared_kv")
    with pytest.raises(ValueError, match="lower_triangular"):
        hstu_ragged_fwd(**inputs, invalid_attn_mask_type=UPPER, contextual_seq_len=2)
    with pytest.raises(ValueError, match="bf16 or f16"):
        hstu_ragged_fwd(
            **{**inputs, "q": inputs["q"].float(), "k": inputs["k"].float()}
        )
