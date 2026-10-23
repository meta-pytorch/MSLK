# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""The driver behind every op's aiter tuning script (``bench/aiter/tune_<op>.py``).

An op's tuning script subclasses :class:`OpTuner` and supplies only what is
specific to the op: its table's key columns, how to build inputs for a shape,
how to call the op, and its reference and tolerance. This
module does the rest, the same way for every op, modelled on aiter's own
tuners (``aiter/utility/base_tuner.py``):

- **Shapes** come from an untuned list (``-i``), a CSV whose columns are exactly
  the table's key columns, e.g. ``N,K,M_max``; lines starting with ``#`` are
  comments. Each row becomes one row of the op's tuning table (``-o``).
- **Incremental and resumable.** Shapes already in the table for this device
  are skipped (``--all`` re-tunes them). Results are merged into the table by
  key every ``--batch`` shapes, so an interrupted run loses at most one batch
  and is resumed by running it again. Rows for other devices are kept.
- **Multi-GPU.** ``--mp N`` tunes on the first N visible GPUs, one worker
  process each; every candidate for a shape runs on the same GPU. Restrict the
  GPUs with ``HIP_VISIBLE_DEVICES``.
- **Timing** is GPU time from CUDA-graph replays, after warm-up (so autotuning
  and compilation are excluded). Each replay runs the op on several copies of
  the inputs in turn, together larger than the last-level cache
  (``--rotate_mb``), so that every call reads its operands from memory, as in
  a model where other layers evict them; aiter's ``perftest`` does the same.
  Backends are timed in alternating order from shape to shape.
- **Correctness** is checked through the op itself (backend forced), against
  the op's reference and tolerance, which should be tight enough that a wrong
  or partial result cannot pass at any shape in the list. Every element must
  be within tolerance (NaN never is), and outputs are filled with NaN before
  each backend runs. The table also records the error (``err_ratio``: fraction
  of elements outside tolerance, as in aiter's tables, and ``max_abs_err``).
- **Choosing.** MSLK's kernel is the default. Another backend is chosen only if
  it is correct and at least ``--min_improvement_pct`` faster (default 3%,
  aiter's default for updating a tuned shape), so that timing noise does not
  flip shapes. A candidate that fails the check, or raises (e.g. a launch config
  that does not fit the GPU), is left out for that shape. MSLK's own kernel
  failing the check is a bug to fix, not a shape to route around: that row is
  not written, and the run exits with an error listing such shapes.
- **Buckets** are timed at their bound (M_max=128 at M=128; it serves
  64 < M <= 128, see :mod:`mslk.aiter.routing`), and checked for correctness
  at both of their ends.
- **Profile.** ``-o2`` appends every candidate's measurement, including errors,
  to a CSV.

Merging tables from separate runs::

    python -m mslk.aiter.tuning merge <table.csv> <part.csv> ...
"""

import abc
import argparse
import csv
import math
import multiprocessing as mp
import os
import sys
import traceback
import zlib
from concurrent.futures import as_completed, ProcessPoolExecutor
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from mslk.aiter.routing import device_key, force_backend, MSLK

Row = Dict[str, str]

DEVICE_COLUMNS: Tuple[str, ...] = ("gfx", "cu_num")


@dataclass
class Measurement:
    backend: str
    # GPU time per call in microseconds, or None if the candidate raised.
    us: Optional[float]
    correct: bool
    err_ratio: float
    max_abs_err: float
    error: Optional[str] = None


@dataclass(frozen=True)
class Settings:
    rotate_bytes: int
    warmup: int
    iters: int
    min_improvement_pct: float


@dataclass
class ShapeResult:
    shape: Row
    device: Row
    measurements: List[Measurement]
    backend: str
    flops: Optional[float] = None
    bytes_moved: Optional[float] = None


class OpTuner(abc.ABC):
    """What an op's tuning script supplies; see the module docstring."""

    #: The op's name, as used by :func:`mslk.aiter.routing.force_backend`.
    op: str
    #: Table key columns besides the device and the bucket, e.g. ("N", "K").
    key_columns: Tuple[str, ...]
    #: The bucketed key column, e.g. "M_max".
    bucket_column: str
    #: Candidate backends; MSLK's kernel first.
    backends: Tuple[str, ...]
    #: Tolerances of the correctness check.
    atol: float
    rtol: float
    #: Keys of :meth:`make_inputs` the op writes; filled with NaN before each
    #: backend's check.
    outputs: Tuple[str, ...] = ()

    @abc.abstractmethod
    def make_inputs(self, shape: Row) -> Dict[str, Any]:
        """Inputs for one call of the op on ``shape`` (a row of the untuned
        list), including any output buffer. Tensors in it are cloned to rotate
        through copies, so it should hold only what one call needs."""

    @abc.abstractmethod
    def run(self, inputs: Dict[str, Any]) -> torch.Tensor:
        """Call the op on ``inputs`` and return its output."""

    @abc.abstractmethod
    def reference(self, shape: Row, inputs: Dict[str, Any]) -> torch.Tensor:
        """What the op's unit test compares the output against."""

    def flops(self, shape: Row) -> Optional[float]:
        """FLOPs of one call, for the table's ``tflops`` column (optional)."""
        return None

    def bytes_moved(self, shape: Row) -> Optional[float]:
        """Bytes read and written by one call, for the ``bw`` column (optional)."""
        return None

    # The table's columns, in order.
    def table_columns(self) -> List[str]:
        return (
            list(DEVICE_COLUMNS)
            + list(self.key_columns)
            + [self.bucket_column, "backend", "us", "err_ratio", "max_abs_err"]
            + ["tflops", "bw"]
            + [f"{b}_us" for b in self.backends]
        )

    def row_key(self, row: Row) -> Tuple[str, ...]:
        return tuple(
            row[c] for c in DEVICE_COLUMNS + self.key_columns + (self.bucket_column,)
        )


# ---------------------------------------------------------------------------
# Choosing, and table files (no GPU needed)
# ---------------------------------------------------------------------------


def choose(measurements: Sequence[Measurement], min_improvement_pct: float) -> str:
    """The backend for one shape, by the rules in the module docstring.

    Assumes MSLK's kernel was correct; callers handle the case where it was not.
    """
    by_backend = {m.backend: m for m in measurements}
    mslk_us = by_backend[MSLK].us
    assert mslk_us is not None
    best = min(
        (m for m in measurements if m.correct and m.us is not None),
        key=lambda m: m.us,  # pyre-ignore[6]
    )
    best_us = best.us
    assert best_us is not None
    if best.backend != MSLK and mslk_us < best_us * (1 + min_improvement_pct / 100):
        return MSLK
    return best.backend


def read_csv_rows(path: str) -> List[Row]:
    """Rows of a CSV with a header; lines starting with ``#`` are comments."""
    if not os.path.exists(path):
        return []
    with open(path) as f:
        lines = [line for line in f if line.strip() and not line.startswith("#")]
    return [dict(r) for r in csv.DictReader(lines)]


def _sort_value(value: str) -> Tuple[int, float, str]:
    try:
        return (0, float(value), "")
    except ValueError:
        return (1, 0.0, value)


def merge_rows(
    old: Iterable[Row],
    new: Iterable[Row],
    key_columns: Sequence[str],
) -> List[Row]:
    """``old`` updated with ``new``: a new row replaces an old one with the same
    key, other old rows (e.g. other devices) are kept, and the result has one
    row per key, sorted by key."""
    merged: Dict[Tuple[str, ...], Row] = {}
    for row in list(old) + list(new):
        merged[tuple(row[c] for c in key_columns)] = row
    return sorted(
        merged.values(), key=lambda r: [_sort_value(r[c]) for c in key_columns]
    )


def write_csv_rows(path: str, rows: Sequence[Row], columns: Sequence[str]) -> None:
    """Write ``rows``; columns of existing rows that ``columns`` lacks are
    kept at the end, and missing cells are left empty."""
    all_columns = list(columns) + sorted({c for r in rows for c in r} - set(columns))
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=all_columns, restval="", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def shapes_to_tune(
    untuned: Sequence[Row],
    table: Sequence[Row],
    device: Row,
    tuner: OpTuner,
    retune_all: bool,
) -> List[Row]:
    """Untuned shapes without a row for ``device`` in ``table``, or all of
    them with ``retune_all``; duplicates removed, order kept."""
    shape_columns = tuner.key_columns + (tuner.bucket_column,)
    done = {
        tuple(r[c] for c in shape_columns)
        for r in table
        if all(r.get(c) == device[c] for c in DEVICE_COLUMNS)
    }
    seen = set()
    todo = []
    for shape in untuned:
        key = tuple(shape[c] for c in shape_columns)
        if key in seen or (key in done and not retune_all):
            continue
        seen.add(key)
        todo.append({c: shape[c] for c in shape_columns})
    return todo


def bucket_lower_edges(
    untuned: Sequence[Row], tuner: OpTuner
) -> Dict[Tuple[str, ...], int]:
    """For each untuned shape (its key columns and bucket), the smallest bucket
    value it serves: one more than the previous bucket for the same key, or 1."""
    bounds: Dict[Tuple[str, ...], List[int]] = {}
    for shape in untuned:
        key = tuple(shape[c] for c in tuner.key_columns)
        bounds.setdefault(key, []).append(int(shape[tuner.bucket_column]))
    edges = {}
    for key, values in bounds.items():
        values = sorted(set(values))
        for i, value in enumerate(values):
            edges[key + (str(value),)] = values[i - 1] + 1 if i else 1
    return edges


def table_row(tuner: OpTuner, result: ShapeResult) -> Row:
    by_backend = {m.backend: m for m in result.measurements}
    chosen = by_backend[result.backend]
    us = chosen.us
    assert us is not None
    row = {**result.device, **result.shape, "backend": result.backend}
    row["us"] = f"{us:.2f}"
    row["err_ratio"] = f"{chosen.err_ratio:.4g}"
    row["max_abs_err"] = f"{chosen.max_abs_err:.4g}"
    row["tflops"] = f"{result.flops / us / 1e6:.2f}" if result.flops else ""
    row["bw"] = f"{result.bytes_moved / us / 1e3:.2f}" if result.bytes_moved else ""
    for b in tuner.backends:
        t = by_backend[b].us if b in by_backend else None
        row[f"{b}_us"] = "" if t is None else f"{t:.2f}"
    return row


PROFILE_COLUMNS: Tuple[str, ...] = (
    "candidate",
    "us",
    "correct",
    "err_ratio",
    "max_abs_err",
    "error",
)


def profile_rows(result: ShapeResult) -> List[Row]:
    return [
        {
            **result.device,
            **result.shape,
            "candidate": m.backend,
            "us": "" if m.us is None else f"{m.us:.2f}",
            "correct": str(m.correct),
            "err_ratio": f"{m.err_ratio:.4g}",
            "max_abs_err": f"{m.max_abs_err:.4g}",
            "error": m.error or "",
        }
        for m in result.measurements
    ]


# ---------------------------------------------------------------------------
# Measuring (one GPU)
# ---------------------------------------------------------------------------


def _clone(value: Any) -> Any:
    return value.clone() if isinstance(value, torch.Tensor) else value


def _input_bytes(inputs: Dict[str, Any]) -> int:
    return sum(
        v.numel() * v.element_size()
        for v in inputs.values()
        if isinstance(v, torch.Tensor)
    )


def rotation_copies(input_bytes: int, rotate_bytes: int, max_copies: int = 64) -> int:
    """How many copies of the inputs to rotate through so that, together, they
    exceed ``rotate_bytes``."""
    if input_bytes <= 0:
        return 1
    return max(1, min(max_copies, math.ceil(rotate_bytes / input_bytes)))


def time_us(
    tuner: OpTuner,
    copies: Sequence[Dict[str, Any]],
    warmup: int,
    iters: int,
) -> float:
    """GPU microseconds per op call, from replaying a CUDA graph that calls
    the op once on each copy of the inputs."""
    for inputs in copies:  # autotune and compile before capturing
        tuner.run(inputs)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for inputs in copies:
            tuner.run(inputs)
    for _ in range(warmup):
        graph.replay()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        graph.replay()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1e3 / (iters * len(copies))


def _error_stats(
    tuner: OpTuner, out: torch.Tensor, ref: torch.Tensor
) -> Tuple[float, float]:
    err = (out.float() - ref.float()).abs()
    # Negated so that NaN or inf in the output counts as outside tolerance.
    outside = ~(err <= tuner.atol + tuner.rtol * ref.float().abs())
    return outside.float().mean().item(), err.max().item()


def _check(
    tuner: OpTuner, inputs: Dict[str, Any], ref: torch.Tensor
) -> Tuple[float, float]:
    for name in tuner.outputs:
        inputs[name].fill_(float("nan"))
    return _error_stats(tuner, tuner.run(inputs), ref)


def tune_shape(
    tuner: OpTuner,
    shape: Row,
    settings: Settings,
    order_flip: bool,
    lower_edge: Optional[int] = None,
) -> ShapeResult:
    """Check and time every backend on one shape, on the current GPU; with
    ``lower_edge``, also check correctness at that bucket value."""
    gfx, cu_num = device_key(torch.cuda.current_device())
    device = {"gfx": gfx, "cu_num": str(cu_num)}
    # Seeded by the shape, so a shape's data does not depend on which worker
    # tunes it or in what order.
    torch.manual_seed(zlib.crc32(repr(sorted(shape.items())).encode()))
    inputs = tuner.make_inputs(shape)
    ref = tuner.reference(shape, inputs)
    n_copies = rotation_copies(_input_bytes(inputs), settings.rotate_bytes)
    copies = [inputs] + [
        {k: _clone(v) for k, v in inputs.items()} for _ in range(n_copies - 1)
    ]
    checks = [(inputs, ref)]
    if lower_edge is not None and lower_edge != int(shape[tuner.bucket_column]):
        lower = {**shape, tuner.bucket_column: str(lower_edge)}
        lower_inputs = tuner.make_inputs(lower)
        checks.append((lower_inputs, tuner.reference(lower, lower_inputs)))
    measurements = []
    order = tuner.backends[::-1] if order_flip else tuner.backends
    for backend in order:
        with force_backend(tuner.op, backend):
            try:
                errors = [_check(tuner, i, r) for i, r in checks]
                err_ratio = max(e[0] for e in errors)
                max_err = max(e[1] for e in errors)
                correct = err_ratio == 0.0
                us = time_us(tuner, copies, settings.warmup, settings.iters)
                measurements.append(
                    Measurement(backend, us, correct, err_ratio, max_err)
                )
            except Exception as e:  # noqa: BLE001
                reason = f"{type(e).__name__}: {(str(e).splitlines() or [''])[0]}"
                measurements.append(
                    Measurement(backend, None, False, 1.0, math.inf, reason)
                )
        torch.cuda.synchronize()
    measurements.sort(key=lambda m: tuner.backends.index(m.backend))
    mslk_ok = next(m for m in measurements if m.backend == MSLK).correct
    return ShapeResult(
        shape=shape,
        device=device,
        measurements=measurements,
        # MSLK failing has no valid choice; the driver reports it instead.
        backend=(choose(measurements, settings.min_improvement_pct) if mslk_ok else ""),
        flops=tuner.flops(shape),
        bytes_moved=tuner.bytes_moved(shape),
    )


def _worker_init(gpu_queue: "mp.Queue[int]") -> None:
    torch.cuda.set_device(gpu_queue.get())


# ---------------------------------------------------------------------------
# The driver
# ---------------------------------------------------------------------------


def _describe(tuner: OpTuner, result: ShapeResult) -> str:
    shape = " ".join(f"{k}={v}" for k, v in result.shape.items())
    parts = []
    for m in result.measurements:
        if m.us is None:
            parts.append(f"{m.backend} failed ({m.error})")
        else:
            flag = "" if m.correct else f" INCORRECT (err_ratio {m.err_ratio:.3g})"
            parts.append(f"{m.backend} {m.us:.1f} us{flag}")
    return f"{shape} | " + ", ".join(parts) + f" -> {result.backend or 'none'}"


def build_parser(
    tuner: OpTuner, untune_file: str, tune_file: str
) -> argparse.ArgumentParser:
    # Flag names follow aiter's tuners (aiter/utility/base_tuner.py) where the
    # meaning is the same.
    parser = argparse.ArgumentParser(
        description=f"Tune which backend serves mslk::{tuner.op}, per shape."
    )
    parser.add_argument(
        "-i", "--untune_file", default=untune_file, help="shapes to tune"
    )
    parser.add_argument(
        "-o", "--tune_file", default=tune_file, help="tuning table to update"
    )
    parser.add_argument(
        "-o2", "--profile_file", default="", help="append every candidate's result here"
    )
    parser.add_argument(
        "--all", action="store_true", help="re-tune shapes already in the table"
    )
    parser.add_argument(
        "--mp", type=int, default=1, help="number of GPUs (worker processes)"
    )
    parser.add_argument(
        "--batch", type=int, default=20, help="write results every N shapes"
    )
    parser.add_argument(
        "--warmup", type=int, default=10, help="graph replays before timing"
    )
    parser.add_argument("--iters", type=int, default=100, help="timed graph replays")
    parser.add_argument(
        "--min_improvement_pct",
        type=float,
        default=3.0,
        help="speed-up over MSLK's kernel needed to choose another backend",
    )
    parser.add_argument(
        "--rotate_mb",
        type=int,
        default=512,
        help="input copies to rotate through, in MB; at least the last-level "
        "cache (MI350X: 256 MB)",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser


def main(tuner: OpTuner, untune_file: str, tune_file: str) -> None:
    """Run the tuning described in the module docstring for ``tuner``."""
    args = build_parser(tuner, untune_file, tune_file).parse_args()
    columns = tuner.table_columns()
    key_columns = DEVICE_COLUMNS + tuner.key_columns + (tuner.bucket_column,)
    gfx, cu_num = device_key(torch.cuda.current_device())
    device = {"gfx": gfx, "cu_num": str(cu_num)}

    table = read_csv_rows(args.tune_file)
    untuned = read_csv_rows(args.untune_file)
    todo = shapes_to_tune(untuned, table, device, tuner, args.all)
    edges = bucket_lower_edges(untuned, tuner)

    def lower_edge(shape: Row) -> int:
        return edges[
            tuple(shape[c] for c in tuner.key_columns + (tuner.bucket_column,))
        ]

    print(f"{len(todo)} shapes to tune on {gfx} ({cu_num} CUs), {args.mp} GPU(s)")
    if not todo:
        return

    pending_table: List[Row] = []
    pending_profile: List[Row] = []
    mslk_failures: List[str] = []

    def flush() -> None:
        nonlocal table
        if pending_table:
            table = merge_rows(table, pending_table, key_columns)
            write_csv_rows(args.tune_file, table, columns)
            pending_table.clear()
        if pending_profile and args.profile_file:
            old = read_csv_rows(args.profile_file)
            profile_columns = list(key_columns) + list(PROFILE_COLUMNS)
            write_csv_rows(args.profile_file, old + pending_profile, profile_columns)
            pending_profile.clear()

    def collect(result: ShapeResult) -> None:
        print(_describe(tuner, result), flush=True)
        pending_profile.extend(profile_rows(result))
        if result.backend:
            pending_table.append(table_row(tuner, result))
        else:
            mslk_failures.append(_describe(tuner, result))
        if len(pending_table) >= args.batch:
            flush()

    settings = Settings(
        rotate_bytes=args.rotate_mb << 20,
        warmup=args.warmup,
        iters=args.iters,
        min_improvement_pct=args.min_improvement_pct,
    )
    try:
        if args.mp <= 1:
            for i, shape in enumerate(todo):
                collect(
                    tune_shape(tuner, shape, settings, i % 2 == 1, lower_edge(shape))
                )
        else:
            ctx = mp.get_context("spawn")
            gpus = ctx.Queue()
            n_workers = min(args.mp, torch.cuda.device_count())
            for g in range(n_workers):
                gpus.put(g)
            pool = ProcessPoolExecutor(
                max_workers=n_workers,
                mp_context=ctx,
                initializer=_worker_init,
                initargs=(gpus,),
            )
            try:
                futures = [
                    pool.submit(
                        tune_shape,
                        tuner,
                        shape,
                        settings,
                        i % 2 == 1,
                        lower_edge(shape),
                    )
                    for i, shape in enumerate(todo)
                ]
                for future in as_completed(futures):
                    collect(future.result())
            finally:
                # On an error or Ctrl-C, drop the queued shapes instead of
                # waiting for all of them; only those already running finish.
                pool.shutdown(wait=False, cancel_futures=True)
    except BaseException:
        # Keep what was measured (e.g. a worker died on a GPU fault); running
        # the script again resumes from here.
        flush()
        traceback.print_exc()
        print("Interrupted; results so far are saved. Run again to resume.")
        sys.exit(1)
    flush()
    print(f"Table: {args.tune_file}")
    if mslk_failures:
        print(
            "MSLK's kernel failed the correctness check on these shapes (not written):"
        )
        print("\n".join(f"  {f}" for f in mslk_failures))
        sys.exit(1)


def merge_tables(out: str, parts: Sequence[str]) -> int:
    """Merge tables from separate runs into ``out``; for the same key, later
    files win. The key is every column before ``backend``."""
    rows = read_csv_rows(out)
    columns = list(rows[0]) if rows else []
    for part in parts:
        part_rows = read_csv_rows(part)
        if not part_rows:
            continue
        columns = columns or list(part_rows[0])
        key_columns = columns[: columns.index("backend")]
        rows = merge_rows(rows, part_rows, key_columns)
    write_csv_rows(out, rows, columns)
    return len(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge aiter tuning tables.")
    sub = parser.add_subparsers(dest="command", required=True)
    merge = sub.add_parser("merge", help="merge tables from separate runs into OUT")
    merge.add_argument("out")
    merge.add_argument("parts", nargs="+")
    args = parser.parse_args()
    n = merge_tables(args.out, args.parts)
    print(f"{n} rows -> {args.out}")
