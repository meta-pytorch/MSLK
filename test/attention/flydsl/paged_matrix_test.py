# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Correctness matrix for paged FlyDSL attention.

Every case goes through ``flydsl_flash_attn_func``, records which kernel served
it, and is compared against an fp32 reference. KV pages a request must not
read hold NaN, so a read past a request's keys or through a padding slot shows
up in the output. Each case runs twice, so state a split merge leaves behind
cannot leak into the next call.

``test_route_matrix`` covers the routes as the dispatcher picks them;
``test_forced_feature`` turns each generic-kernel feature on where the
heuristics would not; ``test_route_inputs`` runs every route with awkward
inputs (wide metadata dtypes, strided tensors, padding, graphs, streams).
Long contexts and large batches run with ``MSLK_FLYDSL_EXTENDED_TESTS=1``.
"""

import math
import os
from dataclasses import dataclass, field, replace

import pytest
import torch
from mslk.attention.flydsl import flash_attn_interface as fai, flydsl_flash_attn_func
from mslk.flydsl.common import is_flydsl_available
from paged_reference import paged_reference, run_on_busy_side_stream, spy

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not is_flydsl_available(),
    reason="requires a ROCm GPU with FlyDSL",
)

PAGE = 64
EXTENDED = os.environ.get("MSLK_FLYDSL_EXTENDED_TESTS", "0") == "1"


def _gfx950() -> bool:
    from flydsl.runtime.device import get_rocm_arch

    return get_rocm_arch().startswith("gfx950")


# ── cases ────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Case:
    """One paged call. ``window`` is the public (flash-attention) window_left;
    ``lengths`` names a per-request length pattern (see ``_lengths``)."""

    B: int
    Sq: int
    ctx: int
    D: int = 128
    H: int = 32
    HKV: int = 4
    window: int = -1
    causal: bool = True
    dtype: torch.dtype = torch.bfloat16
    varlen: bool = False
    lengths: str = "full"
    num_kv_splits: int = 0
    knobs: tuple = field(default_factory=tuple)

    @property
    def id(self) -> str:
        parts = [
            f"B{self.B}",
            f"q{self.Sq}",
            f"ctx{self.ctx}",
            f"d{self.D}",
            f"h{self.H}x{self.HKV}",
            "f16" if self.dtype == torch.float16 else "bf16",
        ]
        if self.window >= 0:
            parts.append(f"w{self.window}")
        if not self.causal:
            parts.append("noncausal")
        if self.varlen:
            parts.append("varlen")
        if self.lengths != "full":
            parts.append(self.lengths)
        if self.num_kv_splits:
            parts.append(f"s{self.num_kv_splits}")
        parts += [f"{k}={v}" for k, v in self.knobs]
        return "-".join(parts)


def _lengths(case: Case) -> list:
    """Per-request KV lengths as passed: ``full``; ``ragged`` cycling through
    the edges (empty, below the query block, one past and short of a page, a
    page exactly, half, one short of full); or ``oob`` with lengths the
    kernel must bound to [0, ctx] itself."""
    if case.lengths == "full":
        return [case.ctx] * case.B
    if case.lengths == "oob":
        edges = [-7, 2**30, case.ctx // 2, 0, case.ctx + 5]
        return [edges[b % len(edges)] for b in range(case.B)]
    edges = [
        case.ctx,
        0,
        max(case.Sq - 1, 1),
        PAGE - 1,
        PAGE,
        PAGE + 1,
        case.ctx // 2 + 3,
        case.ctx - 1,
    ]
    return [edges[b % len(edges)] for b in range(case.B)]


# Off-route knobs: (module attribute, value) pairs set for one case.
SWA_OFF = (("_PAGED_SWA_KERNEL", False),)
HP_OFF = (("_DISABLE_PAGED_DECODE_HP", True),)
PACK_OFF = (("_PAGED_GQA_PACK", False),)


# Each entry: (case, route the dispatcher must take). Routes: "swa"
# (pa_decode_swa), "hp" (head-packed decode), "light" (generic paged kernel),
# "dualwave" (gfx950 dualwave paged kernel).
ROUTE_CASES = [
    # pa_decode_swa: in-kernel split merge, reduce-kernel merge, no split,
    # D=64, two M-tiles, and full attention.
    (Case(16, 1, 4096, window=2047, lengths="ragged"), "swa"),
    (Case(4, 1, 4096, window=2047, lengths="ragged"), "swa"),
    (Case(64, 4, 4096, window=2047), "swa"),
    (Case(8, 4, 4096, D=64, window=2047, lengths="ragged"), "swa"),
    (Case(16, 4, 4096, window=2047, dtype=torch.float16), "swa"),
    (Case(1, 1, 32768, lengths="full"), "swa"),
    (Case(2, 4, 16384, D=64, lengths="ragged"), "swa"),
    (Case(16, 1, 4096, H=16, HKV=4, window=255), "swa"),
    (Case(16, 1, 4096, H=32, HKV=2, window=1023), "swa"),
    # Head-packed decode: MHA (packing declines), multi-token, f16, and the
    # query-grouped deep block.
    (Case(4, 1, 8192, H=8, HKV=8, lengths="ragged", knobs=SWA_OFF), "hp"),
    (Case(4, 2, 8192, D=64, H=8, HKV=8, knobs=SWA_OFF), "hp"),
    (Case(4, 1, 8192, H=8, HKV=8, dtype=torch.float16, knobs=SWA_OFF), "hp"),
    (Case(8, 16, 32768, knobs=SWA_OFF + PACK_OFF), "hp"),
    (Case(8, 4, 8192, D=64, lengths="ragged", knobs=SWA_OFF + PACK_OFF), "hp"),
    # Generic paged kernel: packed GQA decode and short blocks, windows,
    # uniform varlen, wide packed tiles, f16, explicit splits.
    (Case(4, 1, 32768, knobs=SWA_OFF + HP_OFF, lengths="ragged"), "light"),
    (Case(2, 16, 32768, D=64, knobs=SWA_OFF), "light"),
    (Case(2, 64, 8192, D=64, knobs=SWA_OFF), "light"),
    (Case(1, 13, 4096, D=64, knobs=SWA_OFF, lengths="ragged"), "light"),
    (Case(2, 16, 4096, window=255, knobs=SWA_OFF), "light"),
    (Case(1, 64, 4096, D=64, window=17, knobs=SWA_OFF), "light"),
    (Case(4, 16, 8192, varlen=True, knobs=SWA_OFF, lengths="ragged"), "light"),
    (Case(2, 4, 8192, varlen=True, window=511, knobs=SWA_OFF), "light"),
    (Case(2, 16, 8192, dtype=torch.float16, knobs=SWA_OFF), "light"),
    (Case(1, 1, 8192, num_kv_splits=4, knobs=SWA_OFF + HP_OFF), "light"),
    (Case(32, 1, 8192, D=64, knobs=SWA_OFF + HP_OFF), "light"),
    (Case(2, 128, 4096, D=64, knobs=SWA_OFF), "light"),
    # Lengths only the kernels bound (negative, past max_seqlen_kv).
    (Case(10, 1, 4096, window=2047, lengths="oob"), "swa"),
    (Case(3, 1, 16384, lengths="oob"), "swa"),
    (Case(5, 2, 8192, H=8, HKV=8, lengths="oob", knobs=SWA_OFF), "hp"),
    (Case(5, 4, 8192, lengths="oob", knobs=SWA_OFF), "light"),
    (Case(5, 16, 8192, varlen=True, lengths="oob", knobs=SWA_OFF), "light"),
    (Case(5, 384, 2048, causal=False, lengths="oob"), "dualwave"),
    # Split-K with per-request lengths has no dualwave form.
    (Case(2, 384, 2048, causal=False, num_kv_splits=2), "light"),
    # Dualwave: bf16 non-causal long query blocks without a window.
    (Case(2, 384, 2048, causal=False, knobs=SWA_OFF), "dualwave"),
    (Case(2, 512, 4096, D=64, causal=False, lengths="ragged"), "dualwave"),
]

EXTENDED_CASES = [
    (Case(1, 1, 262144, window=2047), "swa"),
    (Case(1, 1, 131072, D=64), "light"),
    (Case(1, 4, 524288, D=64, knobs=SWA_OFF), "light"),
    (Case(256, 1, 4096, window=2047, lengths="ragged"), "swa"),
    (Case(128, 4, 32768, knobs=SWA_OFF), "light"),
    (Case(512, 1, 2048, H=8, HKV=8, knobs=SWA_OFF), "hp"),
    (Case(64, 16, 65536, D=64, knobs=SWA_OFF), "light"),
]


# ── inputs, routes, checks ──────────────────────────────────────────────────


@dataclass
class Inputs:
    q: torch.Tensor  # dense [B, Sq, H, D]
    k: torch.Tensor
    v: torch.Tensor
    block_table: torch.Tensor
    seqlen_k: torch.Tensor  # as passed to the op
    ref_lengths: torch.Tensor  # what each request may see, for the reference
    starts: torch.Tensor = None  # gappy: first key of each request


def _i32(values) -> torch.Tensor:
    return torch.tensor(values, device="cuda", dtype=torch.int32)


def _make_inputs(
    case: Case, seed: int = 0, pad_table: bool = False, starts=None
) -> Inputs:
    """Random paged inputs with shuffled pages. Pages a request does not own,
    and spare pages no request owns, hold NaN; with ``pad_table`` the
    block-table slots past a request's pages are -1. ``starts`` makes the
    request's keys begin part-way into its row (gappy KV)."""
    torch.manual_seed(seed)
    raw = _lengths(case)
    starts = [0] * case.B if starts is None else starts
    lengths = [min(max(n, 0), case.ctx - s) for n, s in zip(raw, starts)]
    per_req = case.ctx // PAGE
    total = case.B * per_req + 4
    shape = (total, PAGE, case.HKV, case.D)
    k = torch.randn(shape, device="cuda", dtype=case.dtype)
    v = torch.randn(shape, device="cuda", dtype=case.dtype)
    order = torch.randperm(total, device="cuda", dtype=torch.int32)
    block_table = order[: case.B * per_req].view(case.B, per_req).clone()
    owned = torch.zeros(total, dtype=torch.bool, device="cuda")
    for b, (n, s) in enumerate(zip(lengths, starts)):
        first, end = s // PAGE, -(-(s + n) // PAGE) if n else s // PAGE
        owned[block_table[b, first:end].long()] = True
        if pad_table:
            block_table[b, end:] = -1
    k[~owned] = math.nan
    v[~owned] = math.nan
    # A peaky softmax, so a wrong key changes the output; sharper than this
    # puts bf16 rounding alone at the tolerance.
    scale = 2.0
    q = torch.randn(case.B, case.Sq, case.H, case.D, device="cuda", dtype=case.dtype)
    return Inputs(q * scale, k, v, block_table, _i32(raw), _i32(lengths), _i32(starts))


class Routes:
    """Records which paged kernel each call reached."""

    def __init__(self, monkeypatch):
        from mslk.attention.flydsl.decode import pa_decode_dense, pa_decode_swa

        self.calls = {
            "swa": spy(monkeypatch, pa_decode_swa, "pa_decode_swa_launch"),
            "hp": spy(monkeypatch, pa_decode_dense, "pa_decode_paged_launch"),
            "light": spy(monkeypatch, fai, "_build_paged_light"),
            "dualwave": spy(monkeypatch, fai, "_build_paged"),
        }
        self._seen = {k: 0 for k in self.calls}

    def take(self) -> str:
        """The single route used since the last ``take``."""
        new = [k for k, c in self.calls.items() if len(c) > self._seen[k]]
        self._seen = {k: len(c) for k, c in self.calls.items()}
        assert len(new) == 1, f"expected one route, saw {new}"
        return new[0]

    def light_traits(self):
        return self.calls["light"].results[-1].traits


def _call(case: Case, x: Inputs, **kw):
    """The paged call for ``case``; varlen cases pack Q (uniform lengths)."""
    args = dict(
        causal=case.causal,
        num_kv_heads=case.HKV,
        block_table=x.block_table,
        seqlen_k=x.seqlen_k,
        kv_cache_layout="linear",
        num_kv_splits=case.num_kv_splits,
        max_seqlen_kv=case.ctx,
        window_left=case.window,
    )
    q = x.q
    if case.varlen:
        q = q.reshape(-1, case.H, case.D)
        cu_q = torch.arange(
            0, (case.B + 1) * case.Sq, case.Sq, device="cuda", dtype=torch.int32
        )
        cu_kv = torch.nn.functional.pad(x.seqlen_k.cumsum(0), (1, 0)).to(torch.int32)
        args.update(
            cu_seqlens_q=cu_q,
            cu_seqlens_kv=cu_kv,
            max_seqlen_q=case.Sq,
            cross_seqlen=True,
        )
    args.update(kw)
    out = flydsl_flash_attn_func(q, x.k, x.v, **args)
    if isinstance(out, tuple):
        return out[0].reshape(case.B, case.Sq, case.H, case.D), out[1]
    return out.reshape(case.B, case.Sq, case.H, case.D)


def _reference(case: Case, x: Inputs) -> torch.Tensor:
    block_table = x.block_table.clamp(min=0)
    return paged_reference(
        x.q,
        x.k,
        x.v,
        block_table,
        x.ref_lengths,
        case.causal,
        case.window,
        starts=x.starts,
    )


def _check(case: Case, got: torch.Tensor, ref: torch.Tensor) -> None:
    tol = 1e-2 if case.dtype == torch.float16 else 2e-2
    # Pages a request does not own hold NaN, so this also catches stray reads.
    assert not got.isnan().any(), "NaN in the output"
    torch.testing.assert_close(got.float(), ref, atol=tol, rtol=tol)


def _apply(monkeypatch, case: Case) -> None:
    for name, value in case.knobs:
        monkeypatch.setattr(fai, name, value)


def _needs_gfx950(route: str) -> None:
    if route in ("swa", "hp", "dualwave") and not _gfx950():
        pytest.skip(f"the {route} route is gfx950-only")


# ── tests ────────────────────────────────────────────────────────────────────


def _route_params(cases):
    return [pytest.param(c, r, id=f"{r}-{c.id}") for c, r in cases]


@pytest.mark.parametrize(
    "case,route", _route_params(ROUTE_CASES + (EXTENDED_CASES if EXTENDED else []))
)
def test_route_matrix(monkeypatch, case, route):
    """The dispatcher sends the case to ``route``, the result matches the
    reference, and a second call gives the same answer."""
    _needs_gfx950(route)
    _apply(monkeypatch, case)
    routes = Routes(monkeypatch)
    x = _make_inputs(case)
    got = _call(case, x)
    assert routes.take() == route
    _check(case, got, _reference(case, x))
    again = _call(case, x)
    routes.take()
    torch.testing.assert_close(again, got, atol=0, rtol=0)


# Generic-kernel features forced on, each over shapes the feature admits. The
# value is the trait the build must show.
FORCED = {
    "k_prefetch": ("_paged_k_prefetch", "PAGED_K_PREFETCH"),
    "kv_dma": ("_paged_kv_dma", "ENABLE_DMA"),
    "narrow_tile": ("_paged_narrow_tile", None),
    # Page ids in registers replace the K prefetch's early lookup; the size
    # gate is the feature's precondition, so only the prefetch is turned off.
    "pid_regs": ("_paged_k_prefetch", "PAGED_PID_REGS"),
    "fused_combine": ("_paged_fused_combine", "FUSED_COMBINE"),
}
FORCED_CASES = {
    "k_prefetch": [
        Case(2, 1, 4096, D=64, lengths="ragged"),
        Case(2, 16, 4096, num_kv_splits=2),
        Case(2, 4, 4096, D=64, window=255),
        Case(4, 4, 4096, varlen=True, lengths="ragged"),
    ],
    "kv_dma": [
        Case(2, 16, 8192, D=64, lengths="ragged"),
        Case(2, 16, 8192, D=64, num_kv_splits=4),
    ],
    "narrow_tile": [
        Case(2, 1, 8192, D=64, lengths="ragged"),
        Case(2, 4, 8192, num_kv_splits=2),
        Case(2, 2, 4096, D=64, window=255),
    ],
    "pid_regs": [
        Case(2, 1, 4096, D=64, lengths="ragged"),
        Case(2, 16, 4096, num_kv_splits=4),
        Case(2, 4, 4096, D=64, window=511),
    ],
    "fused_combine": [
        Case(32, 1, 4096, D=64, num_kv_splits=4, lengths="ragged"),
        Case(16, 1, 4096, num_kv_splits=4),
        Case(32, 4, 4096, D=64, num_kv_splits=2, window=1023),
    ],
}


@pytest.mark.parametrize(
    "feature,case",
    [
        pytest.param(f, c, id=f"{f}-{c.id}")
        for f, cases in FORCED_CASES.items()
        for c in cases
    ],
)
def test_forced_feature(monkeypatch, feature, case):
    """Each generic-kernel feature, turned on where the heuristics would not
    pick it, still gives the reference answer, and the build really has it."""
    monkeypatch.setattr(fai, "_PAGED_SWA_KERNEL", False)
    monkeypatch.setattr(fai, "_DISABLE_PAGED_DECODE_HP", True)
    decide, trait = FORCED[feature]
    monkeypatch.setattr(fai, decide, lambda *a, **kw: feature != "pid_regs")
    if feature == "kv_dma":
        # DMA and the K prefetch exclude each other; keep the one under test.
        monkeypatch.setattr(fai, "_paged_k_prefetch", lambda *a, **kw: False)
    if feature == "fused_combine":
        # Keep the build's own exclusions: no LSE, enough counters.
        monkeypatch.setattr(
            fai,
            decide,
            lambda splitk, return_lse, wg, *rest: (
                splitk and not return_lse and wg <= 1 << 16
            ),
        )
    routes = Routes(monkeypatch)
    x = _make_inputs(case)
    got = _call(case, x)
    assert routes.take() == "light"
    traits = routes.light_traits()
    if trait is not None:
        assert getattr(traits, trait), f"{feature} was asked for but not built"
    else:
        assert traits.BLOCK_M == 32, "narrow tile was asked for but not built"
    _check(case, got, _reference(case, x))
    torch.testing.assert_close(_call(case, x), got, atol=0, rtol=0)


def test_wave_splitk_engages():
    """Packed single-token decode at D=128 over a long context builds wave
    split-K, and the result is right. One request: wider grids take the
    narrow tile instead."""
    case = Case(1, 1, 32768)
    with pytest.MonkeyPatch.context() as m:
        m.setattr(fai, "_PAGED_SWA_KERNEL", False)
        m.setattr(fai, "_DISABLE_PAGED_DECODE_HP", True)
        routes = Routes(m)
        x = _make_inputs(case)
        got = _call(case, x)
        assert routes.take() == "light"
        assert routes.light_traits().ENABLE_WAVE_SPLITK
    _check(case, got, _reference(case, x))


# Representative case per route for the input variants below.
ROUTE_REPRESENTATIVES = {
    "swa": Case(16, 1, 4096, window=2047, lengths="ragged"),
    "swa_reduce": Case(4, 1, 4096, window=2047, lengths="ragged"),
    "hp": Case(6, 2, 8192, H=8, HKV=8, lengths="ragged", knobs=SWA_OFF),
    "light": Case(4, 4, 8192, lengths="ragged", knobs=SWA_OFF),
    "light_varlen": Case(4, 16, 8192, varlen=True, lengths="ragged", knobs=SWA_OFF),
    "dualwave": Case(2, 384, 2048, causal=False, knobs=SWA_OFF),
}


def _route_name(rep: str) -> str:
    return {"swa_reduce": "swa", "light_varlen": "light"}.get(rep, rep)


@pytest.mark.parametrize(
    "variant",
    [
        "int64_lengths",
        "padded_table",
        "strided_q",
        "strided_kv",
        "out_guard",
        "explicit_stream",
        "graph_replay",
    ],
)
@pytest.mark.parametrize("rep", list(ROUTE_REPRESENTATIVES))
def test_route_inputs(monkeypatch, rep, variant):
    """Every route takes the awkward forms of its inputs."""
    route = _route_name(rep)
    _needs_gfx950(route)
    case = ROUTE_REPRESENTATIVES[rep]
    if variant == "strided_q" and case.varlen:
        pytest.skip("varlen Q is packed by the caller")
    _apply(monkeypatch, case)
    routes = Routes(monkeypatch)
    x = _make_inputs(case, pad_table=variant == "padded_table")
    ref = _reference(case, x)
    kw = {}
    if variant == "int64_lengths":
        x = replace(x, seqlen_k=x.seqlen_k.to(torch.int64))
    elif variant == "strided_q":
        # Q sliced out of a fused QKV projection.
        qkv = torch.zeros(
            case.B,
            case.Sq,
            case.H + 2 * case.HKV,
            case.D,
            device="cuda",
            dtype=case.dtype,
        )
        qkv[:, :, : case.H] = x.q
        x = replace(x, q=qkv[:, :, : case.H])
    elif variant == "strided_kv":
        # K and V as views into a cache that interleaves them per page.
        kv = torch.stack((x.k, x.v), dim=1)
        x = replace(x, k=kv[:, 0], v=kv[:, 1])
    if variant == "out_guard":
        guard = 1024
        shape = (case.B * case.Sq, case.H, case.D) if case.varlen else x.q.shape
        n = math.prod(shape)
        storage = torch.full((n + 2 * guard,), 7.0, device="cuda", dtype=case.dtype)
        kw["out"] = storage[guard:-guard].view(shape)
        got = _call(case, x, **kw)
        assert got.data_ptr() == kw["out"].data_ptr()
        assert (storage[:guard] == 7).all() and (storage[-guard:] == 7).all()
    elif variant == "explicit_stream":
        _call(case, x)  # compile, so the timed call only launches
        torch.cuda.synchronize()
        routes.take()
        got = run_on_busy_side_stream(lambda s: _call(case, x, stream=s))
    elif variant == "graph_replay":
        x = replace(x, seqlen_k=x.seqlen_k.clone())
        _call(case, x)
        routes.take()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            got = _call(case, x)
        x.seqlen_k.copy_(torch.flip(x.seqlen_k, (0,)))
        x.block_table.copy_(torch.flip(x.block_table, (0,)))
        x = replace(x, ref_lengths=torch.flip(x.ref_lengths, (0,)))
        graph.replay()
        torch.cuda.synchronize()
        ref = _reference(case, x)
    else:
        got = _call(case, x)
    assert routes.take() == route
    _check(case, got, ref)


@pytest.mark.parametrize("causal", [True, False])
def test_gappy_kv(monkeypatch, causal):
    """Requests whose keys start part-way into their block-table row
    (``kv_seqstart``), including a start that is not page-aligned."""
    case = Case(4, 4, 4096, causal=causal, varlen=True)
    starts = [0, 37, 1000, 4000]
    x = _make_inputs(case, starts=starts)
    routes = Routes(monkeypatch)
    cu_kv = torch.nn.functional.pad(x.ref_lengths.cumsum(0), (1, 0)).to(torch.int32)
    got = _call(
        case,
        x,
        seqlen_k=None,
        cu_seqlens_kv=cu_kv,
        kv_seqstart=x.starts,
        window_left=-1,
    )
    assert routes.take() == "light"
    _check(case, got, _reference(case, x))


def _reference_lse(case: Case, x: Inputs) -> torch.Tensor:
    """fp32 log-sum-exp [B, H, Sq] of the scaled, masked scores; -inf where a
    row keeps no key."""
    group = case.H // case.HKV
    out = torch.full((case.B, case.H, case.Sq), -math.inf, device="cuda")
    for b in range(case.B):
        n = int(x.ref_lengths[b])
        if n == 0:
            continue
        pages = x.block_table[b, : -(-n // PAGE)].long()
        kb = x.k[pages].reshape(-1, case.HKV, case.D)[:n].float()
        qb = x.q[b].float().reshape(case.Sq, case.HKV, group, case.D)
        scores = torch.einsum("tngd,knd->ngtk", qb, kb) / math.sqrt(case.D)
        q_pos = torch.arange(case.Sq, device="cuda") + n - case.Sq
        k_pos = torch.arange(n, device="cuda")
        keep = torch.ones(case.Sq, n, dtype=torch.bool, device="cuda")
        if case.causal:
            keep &= k_pos[None, :] <= q_pos[:, None]
        if case.window >= 0:
            keep &= k_pos[None, :] >= q_pos[:, None] - case.window
        lse = scores.masked_fill(~keep, -math.inf).logsumexp(-1)
        out[b] = lse.reshape(case.H, case.Sq)
    return out


@pytest.mark.parametrize(
    "case",
    [
        Case(4, 4, 4096, D=64, lengths="ragged", knobs=SWA_OFF),
        Case(2, 1, 8192, window=255, lengths="ragged", knobs=SWA_OFF),
        Case(1, 16, 8192, num_kv_splits=4, knobs=SWA_OFF),
    ],
    ids=lambda c: c.id,
)
def test_return_lse(monkeypatch, case):
    """The log-sum-exp the generic paged kernel returns, split or not."""
    _apply(monkeypatch, case)
    x = _make_inputs(case)
    got, lse = _call(case, x, return_lse=True)
    _check(case, got, _reference(case, x))
    ref = _reference_lse(case, x)
    assert torch.equal(lse.isinf(), ref.isinf())
    finite = ~ref.isinf()
    torch.testing.assert_close(lse[finite], ref[finite], atol=2e-2, rtol=2e-3)


def _vectorized(k: torch.Tensor, v: torch.Tensor):
    """Linear [pages, page, Hkv, D] K/V in the vectorized (aiter 5D) layout:
    K [pages, Hkv, D/kVS, page, kVS], V [pages, Hkv, page/kVS, D, kVS]."""
    pages, page, hkv, d = k.shape
    vs = 16 // k.element_size()
    k5 = k.view(pages, page, hkv, d // vs, vs).permute(0, 2, 3, 1, 4).contiguous()
    v5 = v.view(pages, page // vs, vs, hkv, d).permute(0, 3, 1, 4, 2).contiguous()
    return k5, v5


@pytest.mark.parametrize(
    "case",
    [
        Case(2, 4, 4096, D=64, lengths="ragged"),
        Case(2, 1, 8192, lengths="ragged"),
        Case(2, 16, 4096, num_kv_splits=2),
    ],
    ids=lambda c: c.id,
)
def test_vectorized_cache_layout(monkeypatch, case):
    """The vectorized KV layout gives the same answer as the linear one."""
    routes = Routes(monkeypatch)
    x = _make_inputs(case)
    k5, v5 = _vectorized(x.k, x.v)
    got = _call(case, replace(x, k=k5, v=v5), kv_cache_layout="vectorized")
    assert routes.take() == "light"
    _check(case, got, _reference(case, x))


ERRORS = [
    ("negative splits", dict(num_kv_splits=-1), ValueError, "nonnegative"),
    ("max_seqlen_kv past table", dict(max_seqlen_kv=4096 + 64), ValueError, None),
    ("negative max_seqlen_kv", dict(max_seqlen_kv=-1), ValueError, None),
    (
        "non-causal window",
        dict(causal=False, window_left=255),
        NotImplementedError,
        "causal",
    ),
]


@pytest.mark.parametrize("name,kw,exc,match", ERRORS, ids=[e[0] for e in ERRORS])
def test_rejects(name, kw, exc, match):
    case = Case(2, 1, 4096)
    x = _make_inputs(case)
    with pytest.raises(exc, match=match):
        _call(case, x, **kw)


def test_rejects_unsupported_cache_shapes():
    case = Case(2, 1, 4096)
    x = _make_inputs(case)
    with pytest.raises(ValueError, match="matching"):
        _call(case, replace(x, v=x.v[:, :, :2]))
    with pytest.raises(ValueError, match="head_dim"):
        _call(case, replace(x, k=x.k[..., :64], v=x.v[..., :64]))
    small = replace(
        x,
        k=x.k.reshape(-1, 32, case.HKV, case.D),
        v=x.v.reshape(-1, 32, case.HKV, case.D),
    )
    with pytest.raises(NotImplementedError, match="page_size"):
        _call(case, small, max_seqlen_kv=case.ctx // 2)


def test_empty_batch():
    case = Case(2, 1, 4096)
    x = _make_inputs(case)
    empty = replace(
        x, q=x.q[:0], block_table=x.block_table[:0], seqlen_k=x.seqlen_k[:0]
    )
    out = flydsl_flash_attn_func(
        empty.q,
        x.k,
        x.v,
        causal=True,
        num_kv_heads=case.HKV,
        block_table=empty.block_table,
        seqlen_k=empty.seqlen_k,
        kv_cache_layout="linear",
        max_seqlen_kv=case.ctx,
    )
    assert out.shape == empty.q.shape


# ── host-side sizing invariants (no GPU work) ────────────────────────────────


@pytest.mark.parametrize("window", [-1, 255, 2047])
@pytest.mark.parametrize("D", [64, 128])
def test_auto_split_invariants(window, D):
    """Auto split counts are at least 1, fit the workspace budget, and give
    every split at most the block-table window's pages wherever a count that
    fits the budget can."""
    budget = fai._paged_affordable_splits
    for B in (1, 2, 4, 16, 64, 256):
        for H, rows, q_len in ((4, 8, 1), (4, 32, 4), (4, 128, 16), (32, 1, 1)):
            for skv in (64, 4096, 65536, 1 << 20, 3 << 20):
                splits = fai._paged_auto_num_kv_splits(
                    B, H, rows, q_len, skv, D, PAGE, window, H == 4, 256
                )
                assert splits >= 1
                assert splits <= max(1, budget(B, H, rows, D))
                floor = -(-(-(-skv // PAGE)) // fai._PAGED_BT_LDS_SIZE)
                if floor <= budget(B, H, rows, D):
                    assert splits >= floor
