# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""FlyDSL-vs-CK forward FMHA sweep (correctness + latency).

Runs ``flydsl.FwOp`` and ``ck.FwOp`` on identical inputs across the
CK-supported bias-type matrix (built with the test suite's ``create_tensors`` so
the varlen / paged / gappy layouts are exactly what the tests use), timing both
ops and computing the relative error wherever both accept the case. This is the
reproducible source for the PR's parity + representative-latency tables: the
representative rows are simply filtered from the full sweep and re-printed at the
end.

Bias categories (the labels used in the representative tables):
    none causal window varlen_causal varlen_causal_br gappy_causal
    paged paged_gappy tensorbias

Examples::

    # Representative rows only (fast) -> reproduces the two tables:
    python -m mslk.bench.attn.flydsl_fwd_bench

    # Full sweep across head dims 64/96/128/256 (approaches the ~222-case sweep):
    python -m mslk.bench.attn.flydsl_fwd_bench --full --head-dims 64,96,128,256
"""

import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import click
import torch
from mslk.attention import fmha
from mslk.attention.fmha import flydsl


def _load_create_tensors():
    """Import the test suite's create_tensors (handles every bias layout)."""
    rel = os.path.join("test", "attention", "fmha", "case_generation.py")
    cands = [os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))]
    d = os.getcwd()
    for _ in range(8):
        cands.append(d)
        d = os.path.dirname(d)
    root = next((c for c in cands if os.path.exists(os.path.join(c, rel))), None)
    if root is None:
        raise SystemExit(
            "Could not locate test/attention/fmha/case_generation.py; run from the "
            "MSLK repo root."
        )
    sys.path.insert(0, os.path.join(root, "test", "attention"))
    try:
        from fmha.case_generation import create_tensors  # noqa: E402
    except Exception as e:  # pragma: no cover - dev tool
        raise SystemExit(f"Failed importing create_tensors ({e}); pytest installed?")
    return create_tensors


_ab = fmha.attn_bias
# label -> attn_bias type passed to create_tensors
BIAS_CASES: Dict[str, object] = {
    "none": type(None),
    "causal": _ab.LowerTriangularMask,
    "window": _ab.LowerTriangularFromBottomRightLocalAttentionMask,
    "varlen_causal": _ab.BlockDiagonalCausalMask,
    "varlen_causal_br": _ab.BlockDiagonalCausalFromBottomRightMask,
    "gappy_causal": _ab.BlockDiagonalCausalWithOffsetGappyKeysMask,
    "paged": _ab.PagedBlockDiagonalCausalWithOffsetPaddedKeysMask,
    "paged_gappy": _ab.PagedBlockDiagonalGappyKeysMask,
    "tensorbias": torch.Tensor,
}

# label -> (B, q_len, kv_len) canonical representative shape (matches the tables).
REPRESENTATIVE: Dict[str, Tuple[int, int, int]] = {
    "none": (1, 4096, 4096),
    "causal": (1, 4096, 4096),
    "window": (1, 2048, 2048),
    "varlen_causal": (8, 1024, 1024),
    "varlen_causal_br": (4, 512, 512),
    "gappy_causal": (4, 256, 512),
    "paged": (4, 256, 512),
    "paged_gappy": (4, 256, 512),
    "tensorbias": (1, 2048, 2048),
}

# Extra shapes added in --full mode (superset that widens the sweep).
FULL_EXTRA: Dict[str, List[Tuple[int, int, int]]] = {
    "none": [(1, 1024, 1024), (1, 2048, 2048), (2, 2048, 2048)],
    "causal": [(1, 1024, 1024), (1, 2048, 2048), (2, 2048, 2048)],
    "window": [(1, 1024, 1024), (1, 4096, 4096)],
    "varlen_causal": [(4, 512, 512), (2, 2048, 2048)],
    "varlen_causal_br": [(8, 1024, 1024)],
    "gappy_causal": [(4, 512, 1024), (8, 512, 512)],
    "paged": [(8, 512, 1024), (4, 512, 512)],
    "paged_gappy": [(8, 512, 1024), (4, 512, 512)],
    "tensorbias": [(1, 1024, 1024), (1, 4096, 4096)],
}

_DTYPE_MAP = {"bf16": torch.bfloat16, "f16": torch.float16}


@dataclass
class Row:
    dtype: str
    bias: str
    B: int
    q: int
    kv: int
    H: int
    D: int
    fly_us: Optional[float]
    ck_us: Optional[float]
    relerr: Optional[float]
    note: str = ""

    @property
    def ratio(self) -> float:
        if self.fly_us and self.ck_us:
            return self.fly_us / self.ck_us
        return float("nan")

    def line(self) -> str:
        shape = f"{self.B},{self.q},{self.kv},{self.D}"
        if self.fly_us is None:
            return (
                f"{self.dtype:>4} {self.bias:>17} {shape:>18} "
                f"{'--':>9} {'--':>9} {'--':>7}   {self.note}"
            )
        return (
            f"{self.dtype:>4} {self.bias:>17} {shape:>18} "
            f"{self.fly_us:>9.1f} {self.ck_us:>9.1f} {self.ratio:>7.2f}  "
            f"relerr={self.relerr:.4g}"
        )


_HEADER = (
    f"{'dt':>4} {'bias':>17} {'B,q,kv,D':>18} {'fly_us':>9} {'ck_us':>9} {'fly/ck':>7}"
)


def _time_loop(call, iters: int) -> float:
    torch.cuda.synchronize()
    st, en = torch.cuda.Event(True), torch.cuda.Event(True)
    st.record()
    for _ in range(iters):
        call()
    en.record()
    torch.cuda.synchronize()
    return st.elapsed_time(en) / iters * 1e3  # ms -> us


def _time_us(call, iters: int, warmup: int, use_graph: bool = True) -> float:
    """Kernel time per call. With use_graph, a CUDA graph replay isolates GPU work
    from flydsl's per-call Python dispatch/packing (CK has ~none, so a plain loop
    over-charges flydsl); falls back to a plain timed loop if capture is unsafe."""
    for _ in range(warmup):  # includes first-call JIT compile
        call()
    torch.cuda.synchronize()
    if use_graph:
        try:
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(3):
                    call()
            torch.cuda.current_stream().wait_stream(s)
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                call()
            torch.cuda.synchronize()
            st, en = torch.cuda.Event(True), torch.cuda.Event(True)
            st.record()
            for _ in range(iters):
                g.replay()
            en.record()
            torch.cuda.synchronize()
            return st.elapsed_time(en) / iters * 1e3
        except Exception:
            torch.cuda.synchronize()  # capture unsafe (host sync) -> plain loop
    return _time_loop(call, iters)


def _bench_case(
    create_tensors, dtype_str, bias, B, q, kv, H, D, iters, warmup, use_graph=True
) -> Row:
    dtype = _DTYPE_MAP[dtype_str]
    bt = BIAS_CASES[bias]
    try:
        qt, kt, vt, ab = create_tensors(
            fmha.ck.FwOp, "cuda", dtype, bt, B, q, kv, H, D, D, fmt="BMHK"
        )
    except Exception as e:
        return Row(dtype_str, bias, B, q, kv, H, D, None, None, None, f"build: {e}")

    inp = fmha.Inputs(query=qt, key=kt, value=vt, attn_bias=ab)
    if not flydsl.FwOp.supports(inp):
        return Row(
            dtype_str, bias, B, q, kv, H, D, None, None, None, "flydslF declined"
        )
    if not fmha.ck.FwOp.supports(inp):
        return Row(dtype_str, bias, B, q, kv, H, D, None, None, None, "ck declined")

    mea = fmha.memory_efficient_attention_forward
    out_fly = mea(qt, kt, vt, ab, op=flydsl.FwOp)
    out_ck = mea(qt, kt, vt, ab, op=fmha.ck.FwOp)
    denom = out_ck.float().norm().clamp_min(1e-6)
    relerr = float((out_fly.float() - out_ck.float()).norm() / denom)

    fly_us = _time_us(
        lambda: mea(qt, kt, vt, ab, op=flydsl.FwOp), iters, warmup, use_graph
    )
    ck_us = _time_us(
        lambda: mea(qt, kt, vt, ab, op=fmha.ck.FwOp), iters, warmup, use_graph
    )
    return Row(dtype_str, bias, B, q, kv, H, D, fly_us, ck_us, relerr)


def _print_representative(rows: List[Row], heads: int) -> None:
    want = {(b, *s) for b, s in REPRESENTATIVE.items()}
    for dt in ("bf16", "f16"):
        sel = [
            r
            for r in rows
            if r.dtype == dt
            and r.D == 128
            and r.H == heads
            and (r.bias, r.B, r.q, r.kv) in want
            and r.fly_us is not None
        ]
        if not sel:
            continue
        print(f"\nRepresentative {dt} results (H={heads}, D=128):")
        print(
            f"{'bias type':>17} {'shape (B,q,kv,D)':>18} {'FlyDSL us':>10} "
            f"{'CK us':>9} {'fly/ck':>7}"
        )
        order = list(REPRESENTATIVE)
        for r in sorted(sel, key=lambda x: order.index(x.bias)):
            print(
                f"{r.bias:>17} {r.B},{r.q},{r.kv},{r.D:>0}".ljust(37)
                + f"{r.fly_us:>10.1f} {r.ck_us:>9.1f} {r.ratio:>7.2f}"
            )


@click.command()
@click.option("--dtypes", default="bf16,f16", help="Comma list: bf16,f16.")
@click.option("--head-dims", "head_dims", default="128", help="Comma list of D.")
@click.option("--heads", default=16, type=int, help="Number of heads (H).")
@click.option("--full", is_flag=True, help="Add the FULL_EXTRA shapes (wider sweep).")
@click.option("--iters", default=50, type=int, help="Timed iterations per op.")
@click.option("--warmup", default=10, type=int, help="Warmup iters (incl. JIT).")
@click.option(
    "--cuda-graph/--no-cuda-graph",
    "use_graph",
    default=True,
    help="Time via CUDA-graph replay (kernel time, excludes per-call Python).",
)
def invoke_main(dtypes, head_dims, heads, full, iters, warmup, use_graph) -> None:
    if not flydsl.FwOp.is_available():
        raise SystemExit("flydslF unavailable on this device/arch")
    create_tensors = _load_create_tensors()
    dts = [d.strip() for d in dtypes.split(",") if d.strip()]
    dims = [int(d) for d in head_dims.split(",") if d.strip()]

    # Build the (bias, B, q, kv) work-list.
    shapes: List[Tuple[str, int, int, int]] = []
    for bias in BIAS_CASES:
        cand = [REPRESENTATIVE[bias]] + (FULL_EXTRA[bias] if full else [])
        for B, q, kv in cand:
            shapes.append((bias, B, q, kv))

    print(_HEADER)
    print("-" * len(_HEADER))
    rows: List[Row] = []
    n_ok = n_skip = 0
    worst = {"bf16": 0.0, "f16": 0.0}
    for dt in dts:
        for bias, B, q, kv in shapes:
            for D in dims:
                r = _bench_case(
                    create_tensors,
                    dt,
                    bias,
                    B,
                    q,
                    kv,
                    heads,
                    D,
                    iters,
                    warmup,
                    use_graph,
                )
                rows.append(r)
                print(r.line())
                if r.fly_us is None:
                    n_skip += 1
                else:
                    n_ok += 1
                    worst[dt] = max(worst.get(dt, 0.0), r.relerr or 0.0)

    _print_representative(rows, heads)
    print(f"\nHardware: {torch.cuda.get_device_name()}")
    print(
        f"Cases: {n_ok} compared, {n_skip} skipped (op declined). "
        f"Worst relerr bf16={worst.get('bf16', 0.0):.4g} f16={worst.get('f16', 0.0):.4g}"
    )


if __name__ == "__main__":
    invoke_main()
