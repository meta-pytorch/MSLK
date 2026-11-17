# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

"""Tests of the HSTU attention oracle in ``mslk.attention.hstu._torch_ref``.

The mask cases are hand-checked grids of two sequences (lengths 8 and 5); each
is compared with the mask helper and, through the full op, with attention
computed directly from the grid. Rows are queries, columns keys, ``#`` valid.
"""

import math

import pytest
import torch
import torch.nn.functional as F
from hstu_test_utils import (
    assert_hstu_close,
    generate_sparse_seq_len,
    HEAD_DIMS,
    make_cross_inputs,
    make_grouped_inputs,
    make_ragged_inputs,
)
from mslk.attention.hstu._torch_ref import (
    cross_hstu_attention_ref,
    cross_hstu_valid_mask,
    grouped_ragged_hstu_attention_ref,
    ragged_hstu_attention_ref,
    ragged_hstu_valid_mask,
)

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def _grid(rows):
    return torch.tensor([[c == "#" for c in row] for row in rows])


def _offsets(lengths, device="cpu"):
    return torch.tensor([0] + list(lengths), device=device).cumsum(0)


def _rand(*shape, device="cpu", seed=0):
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(*shape, generator=generator, dtype=torch.float64).to(device)


def _direct_attention(
    q, k, v, offsets_q, offsets_kv, grids, alpha, scale, num_softmax_heads
):
    """Attention from per-sequence masks, independent of the module under test."""
    outs = []
    for b, valid in enumerate(grids):
        qb = q[offsets_q[b] : offsets_q[b + 1]]
        kb = k[offsets_kv[b] : offsets_kv[b + 1]]
        vb = v[offsets_kv[b] : offsets_kv[b + 1]]
        valid = valid.to(q.device)
        out = torch.zeros(
            qb.shape[0], qb.shape[1], vb.shape[2], dtype=q.dtype, device=q.device
        )
        for h in range(qb.shape[1]):
            s = alpha * qb[:, h] @ kb[:, h].T
            if h < num_softmax_heads:
                p = torch.softmax(
                    s.masked_fill(~valid, float("-inf")), dim=-1
                ).nan_to_num(0.0)
            else:
                p = F.silu(s) * scale * valid
            out[:, h] = p @ vb[:, h]
        outs.append(out)
    return torch.cat(outs)


# (options, num_targets, grid of the length-8 sequence, grid of the length-5 sequence)
MASK_CASES = [
    pytest.param(
        {},
        None,
        [
            "#.......",
            "##......",
            "###.....",
            "####....",
            "#####...",
            "######..",
            "#######.",
            "########",
        ],
        ["#....", "##...", "###..", "####.", "#####"],
        id="causal",
    ),
    pytest.param(
        {},
        (2, 2),
        [
            "#.......",
            "##......",
            "###.....",
            "####....",
            "#####...",
            "######..",
            "#######.",
            "######.#",
        ],
        ["#....", "##...", "###..", "####.", "###.#"],
        id="targets",
    ),
    pytest.param(
        {"max_attn_len": 2},
        None,
        [
            "#.......",
            "##......",
            "###.....",
            ".###....",
            "..###...",
            "...###..",
            "....###.",
            ".....###",
        ],
        ["#....", "##...", "###..", ".###.", "..###"],
        id="window",
    ),
    pytest.param(
        {"max_attn_len": 2, "full_attn_size": 2},
        None,
        [
            "#.......",
            "##......",
            "###.....",
            ".###....",
            "..###...",
            "...###..",
            "#######.",
            "########",
        ],
        ["#....", "##...", "###..", "####.", "#####"],
        id="window-full-attn",
    ),
    pytest.param(
        {"contextual_seq_len": 3},
        None,
        [
            "########",
            "########",
            "########",
            "####....",
            "#####...",
            "######..",
            "#######.",
            "########",
        ],
        ["#####", "#####", "#####", "####.", "#####"],
        id="contextual",
    ),
    pytest.param(
        {"contextual_seq_len": 3, "max_attn_len": 2},
        None,
        [
            "########",
            "########",
            "########",
            "####....",
            "#####...",
            "######..",
            "###.###.",
            "###..###",
        ],
        ["#####", "#####", "#####", "####.", "#####"],
        id="contextual-window",
    ),
    pytest.param(
        {"contextual_seq_len": 3},
        (2, 2),
        [
            "######..",
            "######..",
            "######..",
            "####....",
            "#####...",
            "######..",
            "#######.",
            "######.#",
        ],
        ["###..", "###..", "###..", "####.", "###.#"],
        id="contextual-targets",
    ),
    pytest.param(
        {"invalid_attn_mask_type": "none"},
        None,
        ["########"] * 8,
        ["#####"] * 5,
        id="none",
    ),
    pytest.param(
        {"invalid_attn_mask_type": "none", "max_attn_len": 1},
        None,
        [
            "##......",
            "###.....",
            ".###....",
            "..###...",
            "...###..",
            "....###.",
            ".....###",
            "......##",
        ],
        ["##...", "###..", ".###.", "..###", "...##"],
        id="none-window",
    ),
    pytest.param(
        {"invalid_attn_mask_type": "none"},
        (2, 2),
        [
            "######..",
            "######..",
            "######..",
            "######..",
            "######..",
            "######..",
            "#######.",
            "######.#",
        ],
        ["###..", "###..", "###..", "####.", "###.#"],
        id="none-targets",
    ),
    pytest.param(
        {"invalid_attn_mask_type": "upper_triangular"},
        None,
        [
            "########",
            ".#######",
            "..######",
            "...#####",
            "....####",
            ".....###",
            "......##",
            ".......#",
        ],
        ["#####", ".####", "..###", "...##", "....#"],
        id="upper",
    ),
    pytest.param(
        {"invalid_attn_mask_type": "upper_triangular"},
        (2, 2),
        [
            "########",
            ".#######",
            "..######",
            "...#####",
            "....####",
            ".....###",
            "......#.",
            ".......#",
        ],
        ["#####", ".####", "..###", "...#.", "....#"],
        id="upper-targets",
    ),
]


@pytest.mark.parametrize("options,targets,grid8,grid5", MASK_CASES)
def test_mask_helper_matches_hand_checked_grid(options, targets, grid8, grid5):
    mask_type = options.get("invalid_attn_mask_type", "lower_triangular")
    other = {k: v for k, v in options.items() if k != "invalid_attn_mask_type"}
    for length, grid, t in zip((8, 5), (grid8, grid5), targets or (None, None)):
        torch.testing.assert_close(
            ragged_hstu_valid_mask(length, mask_type, t, **other), _grid(grid)
        )


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("options,targets,grid8,grid5", MASK_CASES)
def test_ragged_op_matches_hand_checked_grid(device, options, targets, grid8, grid5):
    lengths, heads, dim, N, alpha = (8, 5), 2, 4, 8, 0.7
    offsets = _offsets(lengths, device)
    q, k, v = (_rand(13, heads, dim, device=device, seed=s) for s in range(3))
    num_targets = None if targets is None else torch.tensor(targets, device=device)
    out, M = ragged_hstu_attention_ref(
        N,
        alpha,
        q,
        k,
        v,
        offsets,
        num_targets=num_targets,
        num_softmax_heads=1,
        **options,
    )
    grids = [_grid(grid8), _grid(grid5)]
    expected = _direct_attention(
        q, k, v, offsets.tolist(), offsets.tolist(), grids, alpha, 1.0 / N, 1
    )
    torch.testing.assert_close(out, expected)
    assert M.shape == (13, 1) and M.dtype == torch.float32


def _two_sequence_case(device):
    # Tokens (q, k, v): sequence 0 is (1, 1, 1), (2, 3, 10); sequence 1 is (1, 2, 5).
    q = torch.tensor([1.0, 2.0, 1.0], dtype=torch.float64, device=device).view(3, 1, 1)
    k = torch.tensor([1.0, 3.0, 2.0], dtype=torch.float64, device=device).view(3, 1, 1)
    v = torch.tensor([1.0, 10.0, 5.0], dtype=torch.float64, device=device).view(3, 1, 1)
    return q, k, v, _offsets((2, 1), device)


def _silu(x):
    return x / (1.0 + math.exp(-x))


@pytest.mark.parametrize("device", DEVICES)
def test_silu_values_by_hand(device):
    q, k, v, offsets = _two_sequence_case(device)
    # alpha = 0.5: s00 = 0.5, s10 = 1, s11 = 3; sequence 1 has s = 1.
    expected = [
        _silu(0.5) * 1.0,
        _silu(1.0) * 1.0 + _silu(3.0) * 10.0,
        _silu(1.0) * 5.0,
    ]

    out, M = ragged_hstu_attention_ref(4, 0.5, q, k, v, offsets)
    torch.testing.assert_close(out.flatten().tolist(), [e / 4 for e in expected])
    assert M.shape == (3, 0)

    out, _ = ragged_hstu_attention_ref(
        4, 0.5, q, k, v, offsets, attn_scale=torch.tensor(0.1, device=device)
    )
    torch.testing.assert_close(out.flatten().tolist(), [e * 0.1 for e in expected])

    per_token = torch.tensor([0.1, 0.2, 0.3], device=device)
    out, _ = ragged_hstu_attention_ref(4, 0.5, q, k, v, offsets, attn_scale=per_token)
    torch.testing.assert_close(
        out.flatten().tolist(), [e * s for e, s in zip(expected, [0.1, 0.2, 0.3])]
    )


@pytest.mark.parametrize("device", DEVICES)
def test_softmax_values_and_lse_by_hand(device):
    q, k, v, offsets = _two_sequence_case(device)
    out, M = ragged_hstu_attention_ref(4, 0.5, q, k, v, offsets, num_softmax_heads=1)
    w1 = math.exp(3.0) / (math.exp(1.0) + math.exp(3.0))
    torch.testing.assert_close(
        out.flatten().tolist(), [1.0, (1 - w1) * 1.0 + w1 * 10.0, 5.0]
    )
    expected_lse = [
        0.5 / math.log(2),
        math.log2(math.exp(1.0) + math.exp(3.0)),
        1.0 / math.log(2),
    ]
    torch.testing.assert_close(M.flatten().tolist(), expected_lse)


@pytest.mark.parametrize("device", DEVICES)
def test_ragged_shared_kv(device):
    lengths, heads, dim = (8, 5), 2, 4
    offsets = _offsets(lengths, device)
    q = _rand(13, heads, dim, device=device, seed=0).requires_grad_()
    k = _rand(13, heads, dim, device=device, seed=1).requires_grad_()
    out, _ = ragged_hstu_attention_ref(
        8, 0.5, q, k, k, offsets, num_softmax_heads=1, shared_kv=True
    )
    dout = _rand(*out.shape, device=device, seed=2)
    dq, dk = torch.autograd.grad(out, (q, k), dout)

    k2, v2 = k.detach().clone().requires_grad_(), k.detach().clone().requires_grad_()
    out2, _ = ragged_hstu_attention_ref(8, 0.5, q, k2, v2, offsets, num_softmax_heads=1)
    dq2, dk2, dv2 = torch.autograd.grad(out2, (q, k2, v2), dout)
    torch.testing.assert_close(out, out2)
    torch.testing.assert_close(dq, dq2)
    torch.testing.assert_close(dk, dk2 + dv2)

    with pytest.raises(ValueError, match="v to be k"):
        ragged_hstu_attention_ref(8, 0.5, q, k, k.clone(), offsets, shared_kv=True)


def test_cross_mask_helper_matches_hand_checked_grid():
    # len_q=3, len_kv=5: query i sees key j <= i + 5 - uih_len_q.
    torch.testing.assert_close(
        cross_hstu_valid_mask(3, 5, 0, causal=True), _grid(["###..", "####.", "#####"])
    )
    torch.testing.assert_close(
        cross_hstu_valid_mask(3, 5, 1, causal=True), _grid(["####.", "#####", "#####"])
    )
    # len_q=4, len_kv=2, one target: the first query sees nothing.
    torch.testing.assert_close(
        cross_hstu_valid_mask(4, 2, 1, causal=True), _grid(["..", "#.", "##", "##"])
    )
    torch.testing.assert_close(
        cross_hstu_valid_mask(4, 2, 1, causal=False), _grid(["##"] * 4)
    )


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("causal", [False, True])
def test_cross_op_matches_hand_checked_grid(device, causal):
    lengths_q, lengths_kv, heads, dim, alpha = (3, 4), (5, 2), 2, 4, 0.7
    offsets_q, offsets_kv = _offsets(lengths_q, device), _offsets(lengths_kv, device)
    q = _rand(7, heads, dim, device=device, seed=0)
    k, v = (_rand(7, heads, dim, device=device, seed=s) for s in (1, 2))
    num_targets = torch.tensor([1, 1], device=device)
    out, M = cross_hstu_attention_ref(
        6,
        alpha,
        q,
        k,
        v,
        offsets_kv,
        offsets_q,
        4,
        None,
        1,
        num_softmax_heads=1,
        num_targets=num_targets,
        causal=causal,
    )
    grids = [_grid(["####.", "#####", "#####"]), _grid(["..", "#.", "##", "##"])]
    if not causal:
        grids = [_grid(["#####"] * 3), _grid(["##"] * 4)]
    expected = _direct_attention(
        q, k, v, offsets_q.tolist(), offsets_kv.tolist(), grids, alpha, 1.0 / 6, 1
    )
    torch.testing.assert_close(out, expected)
    # A query that sees no key outputs 0 and has an LSE of -inf.
    assert torch.isfinite(M).all().item() != causal
    if causal:
        assert M[3, 0].item() == float("-inf") and out[3].abs().max().item() == 0.0


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("truncate_method", ["keep_first", "keep_last"])
def test_cross_truncation_equals_slicing(device, truncate_method):
    lengths_q, lengths_kv, max_seq_len = (2, 3), (6, 3), 4
    q = _rand(5, 2, 4, device=device, seed=0)
    k = _rand(9, 2, 4, device=device, seed=1).requires_grad_()
    v = _rand(9, 2, 4, device=device, seed=2)
    offsets_q, offsets_kv = _offsets(lengths_q, device), _offsets(lengths_kv, device)
    out, _ = cross_hstu_attention_ref(
        max_seq_len,
        0.5,
        q,
        k,
        v,
        offsets_kv,
        offsets_q,
        3,
        None,
        1,
        num_softmax_heads=1,
        truncate_method=truncate_method,
    )
    kept = slice(0, 4) if truncate_method == "keep_first" else slice(2, 6)
    rows = list(range(9))[kept] + [6, 7, 8]
    out_sliced, _ = cross_hstu_attention_ref(
        max_seq_len,
        0.5,
        q,
        k[rows],
        v[rows],
        _offsets((4, 3), device),
        offsets_q,
        3,
        None,
        1,
        num_softmax_heads=1,
    )
    torch.testing.assert_close(out, out_sliced)
    (dk,) = torch.autograd.grad(out.sum(), k)
    dropped = sorted(set(range(9)) - set(rows))
    assert dk[dropped].abs().max().item() == 0.0


@pytest.mark.parametrize("device", DEVICES)
def test_cross_gqa_equals_repeated_kv_heads(device):
    q = _rand(5, 4, 4, device=device, seed=0)
    k, v = (_rand(9, 2, 4, device=device, seed=s) for s in (1, 2))
    offsets_q, offsets_kv = _offsets((2, 3), device), _offsets((6, 3), device)
    args = (offsets_kv, offsets_q, 3, None)
    out, M = cross_hstu_attention_ref(
        6, 0.5, q, k, v, *args, 2, num_softmax_heads=1, causal=True
    )
    k_rep, v_rep = k.repeat_interleave(2, dim=1), v.repeat_interleave(2, dim=1)
    out_rep, M_rep = cross_hstu_attention_ref(
        6, 0.5, q, k_rep, v_rep, *args, 1, num_softmax_heads=1, causal=True
    )
    torch.testing.assert_close(out, out_rep)
    torch.testing.assert_close(M, M_rep)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("scale_form", ["none", "per-group", "dynamic"])
def test_grouped_equals_loop_over_ragged(device, scale_form):
    max_seq_lens, group_size = [8, 6], 2
    inputs = make_grouped_inputs(
        group_size,
        max_seq_lens,
        2,
        8,
        8,
        0.7,
        dtype=torch.float64,
        device=device,
        seed=3,
        max_targets=2,
        requires_grad=True,
    )
    windows, contexts, full_sizes = [2, 0], [0, 2], [1, 0]
    total = inputs["q"].shape[0]
    attn_scale = {
        "none": None,
        "per-group": torch.tensor([0.3, 0.05], dtype=torch.float64, device=device),
        "dynamic": torch.linspace(0.05, 0.5, total, dtype=torch.float64, device=device),
    }[scale_form]
    out = grouped_ragged_hstu_attention_ref(
        **dict(
            inputs,
            attn_scale_tensor=attn_scale,
            max_attn_len_tensor=torch.tensor(windows, device=device),
            contextual_seq_len_tensor=torch.tensor(contexts, device=device),
            full_attn_size_tensor=torch.tensor(full_sizes, device=device),
        )
    )
    dout = _rand(*out.shape, device=device, seed=4)
    leaves = (inputs["q"], inputs["k"], inputs["v"])
    grads = torch.autograd.grad(out, leaves, dout)

    offsets = inputs["seq_offsets"].tolist()
    expected = []
    for g, n in enumerate(max_seq_lens):
        first, last = offsets[g * group_size], offsets[(g + 1) * group_size]
        local_offsets = (
            inputs["seq_offsets"][g * group_size : (g + 1) * group_size + 1] - first
        )
        if scale_form == "none":
            scale = None
        elif scale_form == "per-group":
            scale = attn_scale[g]
        else:
            scale = attn_scale[first:last]
        out_g, _ = ragged_hstu_attention_ref(
            n,
            inputs["alpha"],
            inputs["q"][first:last],
            inputs["k"][first:last],
            inputs["v"][first:last],
            local_offsets,
            num_targets=inputs["num_targets"][g * group_size : (g + 1) * group_size],
            attn_scale=scale,
            max_attn_len=windows[g],
            contextual_seq_len=contexts[g],
            full_attn_size=full_sizes[g],
        )
        expected.append(out_g)
    expected = torch.cat(expected)
    torch.testing.assert_close(out, expected)
    for got, want in zip(grads, torch.autograd.grad(expected, leaves, dout)):
        torch.testing.assert_close(got, want)


def test_gradcheck_all_ops():
    q, k, v = (_rand(13, 2, 3, seed=s).requires_grad_() for s in range(3))
    offsets, targets = _offsets((8, 5)), torch.tensor([2, 1])

    def ragged(q, k, v):
        return ragged_hstu_attention_ref(
            8,
            0.6,
            q,
            k,
            v,
            offsets,
            num_targets=targets,
            attn_scale=torch.tensor(0.2, dtype=torch.float64),
            max_attn_len=2,
            contextual_seq_len=2,
            full_attn_size=1,
            num_softmax_heads=1,
        )[0]

    def cross(q, k, v):
        return cross_hstu_attention_ref(
            6,
            0.6,
            q[:7],
            k,
            v,
            _offsets((6, 7)),
            _offsets((3, 4)),
            4,
            None,
            2,
            num_softmax_heads=1,
            truncate_method="keep_last",
            num_targets=torch.tensor([1, 1]),
            causal=True,
        )[0]

    def grouped(q, k, v):
        return grouped_ragged_hstu_attention_ref(
            1,
            q,
            k,
            v,
            offsets,
            offsets,
            targets,
            torch.tensor([8, 6]),
            torch.tensor([2, 0]),
            torch.tensor([0, 0]),
            torch.tensor([0, 2]),
            0.6,
            "lower_triangular",
            attn_scale_tensor=torch.tensor([0.3, 0.1], dtype=torch.float64),
        )

    for fn in (ragged, grouped):
        assert torch.autograd.gradcheck(fn, (q, k, v))
    k_cross, v_cross = (_rand(13, 1, 3, seed=s).requires_grad_() for s in (5, 6))
    assert torch.autograd.gradcheck(cross, (q, k_cross, v_cross))


def _ragged_args(**overrides):
    q = torch.zeros(13, 2, 4)
    args = {
        "N": 8,
        "alpha": 1.0,
        "q": q,
        "k": q,
        "v": q,
        "seq_offsets": _offsets((8, 5)),
    }
    args.update(overrides)
    return args


def _cross_args(**overrides):
    q, kv = torch.zeros(7, 2, 4), torch.zeros(9, 2, 4)
    args = {
        "max_seq_len": 8,
        "alpha": 1.0,
        "q": q,
        "k": kv,
        "v": kv,
        "seq_offsets": _offsets((6, 3)),
        "seq_offsets_q": _offsets((3, 4)),
        "max_q_len": 4,
        "attn_scale": None,
        "G": 1,
    }
    args.update(overrides)
    return args


def _grouped_args(**overrides):
    q = torch.zeros(13, 2, 4)
    zeros = torch.zeros(2, dtype=torch.int64)
    args = {
        "Z": 1,
        "q": q,
        "k": q,
        "v": q,
        "seq_offsets": _offsets((8, 5)),
        "group_offsets": _offsets((8, 5)),
        "num_targets": None,
        "max_seq_len_tensor": torch.tensor([8, 5]),
        "max_attn_len_tensor": zeros,
        "full_attn_size_tensor": zeros,
        "contextual_seq_len_tensor": zeros,
        "alpha": 1.0,
        "invalid_attn_mask_type": "lower_triangular",
    }
    args.update(overrides)
    return args


@pytest.mark.parametrize(
    "op,args,match",
    [
        (ragged_hstu_attention_ref, _ragged_args(N=6), "length <= N"),
        (
            ragged_hstu_attention_ref,
            _ragged_args(invalid_attn_mask_type="causal"),
            "invalid_attn_mask_type",
        ),
        (
            ragged_hstu_attention_ref,
            _ragged_args(invalid_attn_mask_type="none", contextual_seq_len=2),
            "require invalid_attn_mask_type",
        ),
        (
            ragged_hstu_attention_ref,
            _ragged_args(invalid_attn_mask_type="upper_triangular", full_attn_size=2),
            "require invalid_attn_mask_type",
        ),
        (
            ragged_hstu_attention_ref,
            _ragged_args(invalid_attn_mask_type="upper_triangular", max_attn_len=2),
            "upper_triangular",
        ),
        (
            ragged_hstu_attention_ref,
            _ragged_args(max_attn_len=-1),
            "max_attn_len must be >= 0",
        ),
        (
            ragged_hstu_attention_ref,
            _ragged_args(num_targets=torch.tensor([9, 0])),
            r"num_targets\[0\]",
        ),
        (
            ragged_hstu_attention_ref,
            _ragged_args(num_targets=torch.tensor([1])),
            "num_targets must have shape",
        ),
        (
            ragged_hstu_attention_ref,
            _ragged_args(attn_scale=torch.ones(5)),
            "attn_scale",
        ),
        (
            ragged_hstu_attention_ref,
            _ragged_args(num_softmax_heads=3),
            "num_softmax_heads",
        ),
        (
            ragged_hstu_attention_ref,
            _ragged_args(seq_offsets=torch.tensor([0, 8, 12])),
            "end at 13",
        ),
        (
            ragged_hstu_attention_ref,
            _ragged_args(seq_offsets=torch.tensor([0.0, 8.0, 13.0])),
            "integer",
        ),
        (
            ragged_hstu_attention_ref,
            _ragged_args(v=torch.zeros(13, 2, 4), shared_kv=True),
            "v to be k",
        ),
        (ragged_hstu_attention_ref, _ragged_args(k=torch.zeros(13, 2, 8)), "head dims"),
        (cross_hstu_attention_ref, _cross_args(G=3), "H // G"),
        (cross_hstu_attention_ref, _cross_args(max_q_len=3), "max_q_len"),
        (
            cross_hstu_attention_ref,
            _cross_args(truncate_method="keep_middle"),
            "truncate_method",
        ),
        (
            cross_hstu_attention_ref,
            _cross_args(seq_offsets=_offsets((6, 3, 0))),
            "same batch",
        ),
        (cross_hstu_attention_ref, _cross_args(attn_scale=torch.ones(9)), "attn_scale"),
        (
            cross_hstu_attention_ref,
            _cross_args(num_targets=torch.tensor([4, 0])),
            r"num_targets\[0\]",
        ),
        (
            cross_hstu_attention_ref,
            _cross_args(num_softmax_heads=-1),
            "num_softmax_heads",
        ),
        (cross_hstu_attention_ref, _cross_args(max_seq_len=0), "max_seq_len"),
        (
            cross_hstu_attention_ref,
            _cross_args(v=torch.zeros(9, 2, 4), shared_kv=True),
            "v to be k",
        ),
        (
            grouped_ragged_hstu_attention_ref,
            _grouped_args(max_seq_len_tensor=torch.tensor([8, 5, 5])),
            r"G \* Z",
        ),
        (grouped_ragged_hstu_attention_ref, _grouped_args(Z=2), r"G \* Z"),
        (
            grouped_ragged_hstu_attention_ref,
            _grouped_args(max_seq_len_tensor=torch.tensor([8, 4])),
            r"max_seq_len_tensor\[1\]",
        ),
        (
            grouped_ragged_hstu_attention_ref,
            _grouped_args(group_offsets=torch.tensor([0, 13])),
            "group_offsets must equal",
        ),
        (
            grouped_ragged_hstu_attention_ref,
            _grouped_args(max_attn_len_tensor=torch.tensor([1])),
            "one per group",
        ),
        (
            grouped_ragged_hstu_attention_ref,
            _grouped_args(full_attn_size_tensor=torch.tensor([1.0, 0.0])),
            "integer",
        ),
        (
            grouped_ragged_hstu_attention_ref,
            _grouped_args(attn_scale_tensor=torch.ones(3)),
            "attn_scale_tensor",
        ),
        (
            grouped_ragged_hstu_attention_ref,
            _grouped_args(attn_scale_tensor=torch.tensor(0.1)),
            "attn_scale_tensor",
        ),
        (
            grouped_ragged_hstu_attention_ref,
            _grouped_args(
                invalid_attn_mask_type="none",
                contextual_seq_len_tensor=torch.tensor([0, 1]),
            ),
            "require invalid_attn_mask_type",
        ),
    ],
)
def test_rejects_invalid_arguments(op, args, match):
    with pytest.raises(ValueError, match=match):
        op(**args)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
@pytest.mark.parametrize("head_dim", HEAD_DIMS)
def test_cpu_and_gpu_references_agree(head_dim):
    cpu = make_ragged_inputs(4, 64, 2, head_dim, head_dim, 0.6, seed=1, max_targets=4)
    gpu = {
        name: x.cuda() if isinstance(x, torch.Tensor) else x for name, x in cpu.items()
    }
    options = {
        "contextual_seq_len": 3,
        "max_attn_len": 16,
        "full_attn_size": 4,
        "num_softmax_heads": 1,
    }
    out_cpu, M_cpu = ragged_hstu_attention_ref(**cpu, **options)
    out_gpu, M_gpu = ragged_hstu_attention_ref(**gpu, **options)
    assert_hstu_close(out_gpu[:, :1].cpu(), out_cpu[:, :1], "softmax")
    assert_hstu_close(out_gpu[:, 1:].cpu(), out_cpu[:, 1:], "silu")
    torch.testing.assert_close(M_gpu.cpu(), M_cpu)

    cpu = make_cross_inputs(
        3, 64, 16, 4, 2, head_dim, head_dim, 0.6, seed=2, shared_kv=True
    )
    gpu = {
        name: x.cuda() if isinstance(x, torch.Tensor) else x for name, x in cpu.items()
    }
    gpu["v"] = gpu["k"]
    out_cpu, _ = cross_hstu_attention_ref(**cpu, truncate_method="keep_last")
    out_gpu, _ = cross_hstu_attention_ref(**gpu, truncate_method="keep_last")
    assert_hstu_close(out_gpu.cpu(), out_cpu, "silu")


@pytest.mark.parametrize("head_dim", HEAD_DIMS)
@pytest.mark.parametrize("shared_kv", [False, True])
def test_generators_are_seeded_and_shaped(head_dim, shared_kv):
    a = make_ragged_inputs(
        6, 32, 2, head_dim, head_dim, 0.5, seed=7, max_targets=3, shared_kv=shared_kv
    )
    b = make_ragged_inputs(
        6, 32, 2, head_dim, head_dim, 0.5, seed=7, max_targets=3, shared_kv=shared_kv
    )
    for name in ("q", "k", "v", "seq_offsets", "num_targets"):
        torch.testing.assert_close(a[name], b[name])
    total = int(a["seq_offsets"][-1])
    assert a["q"].shape == (total, 2, head_dim) and a["q"].dtype == torch.bfloat16
    assert (a["v"] is a["k"]) == shared_kv
    lengths = a["seq_offsets"].diff()
    assert (a["num_targets"] <= lengths.clamp(max=3)).all()

    cross = make_cross_inputs(
        5,
        32,
        8,
        4,
        2,
        head_dim,
        head_dim,
        0.5,
        seed=7,
        max_targets=2,
        shared_kv=shared_kv,
    )
    assert cross["k"].shape[1] == 2 and (cross["v"] is cross["k"]) == shared_kv
    assert cross["seq_offsets_q"].diff().max() <= 8
    cross_hstu_attention_ref(**cross, causal=True)


def test_sparse_lengths_follow_sparsity():
    generator = torch.Generator().manual_seed(0)
    for sparsity in (0.2, 0.5, 0.8):
        lengths = generate_sparse_seq_len(4096, 1000, sparsity, generator)
        assert lengths.min() >= 0 and lengths.max() <= 1000
        assert abs(lengths.float().mean().item() / 1000 - sparsity) < 0.02
    assert (generate_sparse_seq_len(3, 100, 1.0, generator) == 100).all()


def test_assert_hstu_close_tolerances():
    ref = torch.tensor([10.0, -1.0, 0.0])
    assert_hstu_close(ref + 0.07, ref, "silu")  # 0.07 < 8e-3 * max|ref| + 8e-3 * |ref|
    with pytest.raises(AssertionError, match="silu"):
        assert_hstu_close(ref + 0.2, ref, "silu")
    assert_hstu_close(torch.tensor([1e-4]), torch.tensor([0.0]), "softmax")
    with pytest.raises(AssertionError, match="softmax"):
        assert_hstu_close(torch.tensor([1e-2]), torch.tensor([0.0]), "softmax")
    with pytest.raises(ValueError, match="mode"):
        assert_hstu_close(ref, ref, "relu")
