# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tune which backend serves ``paged_attention_decode_fp8`` on gfx950, per shape.

Compares MSLK's FlyDSL kernel with aiter's FlyDSL ``pa_decode`` on every shape
in ``bench/aiter/untuned/paged_attention_decode_fp8.csv`` and updates
``mslk/attention/dispatch/tables/paged_attention_decode_fp8.csv``. The tuning
itself (incremental runs, multi-GPU, timing, choosing) is
:mod:`mslk.aiter.tuning`; this file holds only what is specific to the op.
Needs the pinned aiter and a gfx950 GPU.

Inputs: ``batch_max`` sequences on a ``random_fp8_paged_cache`` whose block
table is ``context_max`` tokens wide. The routing keys on that width, but
callers often size block tables for longer contexts than they hold, so each
shape is tuned twice, through the shared driver: with lengths spread evenly
over (context_max / 2, context_max] ("full") and over [1, context_max / 16]
("short"). A shape goes to aiter only by the rule in ``combine`` (see
``MIN_IMPROVEMENT_PCT`` and ``MAX_SHORT_*``); the committed table records both
runs' timings. The runs' own tables and logs go to ``--parts_dir`` (a
temporary directory by default), which also lets an interrupted run resume.

Correctness: the reference is attention over the dequantized cache in FP32,
checked as the op's test checks it: elementwise and, per sequence, in scale
and relative error (``TOLERANCE``, ``MAX_SCALE_ERROR``, ``MAX_RELATIVE_ERROR``).

Example, on four GPUs (run from the repository root):

    HIP_VISIBLE_DEVICES=0,1,2,3 python bench/aiter/tune_paged_attention_decode_fp8.py --mp 4

The driver's options apply to both runs, except ``-o2``: each run's checks and
timings are logged in ``--parts_dir``. ``-o`` names the combined table.
"""

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List

import torch
from mslk.aiter import tuning
from mslk.aiter.routing import MSLK
from mslk.aiter.tuning import OpTuner, Row
from mslk.attention.dispatch.paged_attention_decode_fp8 import AITER, BACKENDS, OP
from mslk.attention.flydsl.decode.fp8_paged_cache import (
    dequantize_fp8_paged_cache,
    random_fp8_paged_cache,
)
from mslk.attention.flydsl.decode.pa_decode_fp8 import (
    get_recommended_splits,
    KV_COMPUTE_BLOCK,
)
from mslk.attention.flydsl.decode.pa_decode_fp8_dispatch import (
    paged_attention_decode_fp8,
)

REPO = Path(__file__).resolve().parents[2]
UNTUNED = REPO / "bench" / "aiter" / "untuned" / "paged_attention_decode_fp8.csv"
TABLE = (
    REPO
    / "mslk"
    / "attention"
    / "dispatch"
    / "tables"
    / "paged_attention_decode_fp8.csv"
)
# Where the two runs keep their tables and logs, to combine and to resume.
PARTS_ENV = "MSLK_TUNE_PA_DECODE_PARTS"
PARTS = Path(
    os.environ.get(
        PARTS_ENV, Path(tempfile.gettempdir()) / "mslk_tune_paged_attention_decode_fp8"
    )
)
# The length profile of one run, set for each driver run (and its workers).
PROFILE_ENV = "MSLK_TUNE_PA_DECODE_LENGTHS"
PROFILES = ("full", "short")
# Which backend serves a shape. The key cannot see the context lengths, so a
# shape goes to aiter only if it is faster with full-length contexts by the
# driver's usual margin, and with short contexts slower by at most these.
MIN_IMPROVEMENT_PCT = 3.0
MAX_SHORT_LOSS_US = 5.0
MAX_SHORT_SLOWDOWN = 1.5
# The op test's checks (test/attention/flydsl/pa_decode_fp8_test.py): FP8
# rounding needs a wide elementwise tolerance, so each sequence must also match
# the reference in least-squares scale and relative L2 error. On the tuning
# shapes both kernels reach ~2% scale error (short sequences average out less
# rounding), against 1% in the test's cases, so the scale limit here is 4%:
# K or V scale errors of 5% still fail.
TOLERANCE = 0.3
MAX_SCALE_ERROR = 0.04
MAX_RELATIVE_ERROR = 0.1


def _lengths(shape: Row) -> List[int]:
    batch, context = int(shape["batch_max"]), int(shape["context_max"])
    if os.environ.get(PROFILE_ENV, "full") == "short":
        top = max(1, context // 16)
        return [top - (top - 1) * b // batch for b in range(batch)]
    return [context - context // 2 * b // batch for b in range(batch)]


# random_fp8_paged_cache's outputs, as keys of the inputs.
_CACHE_KEYS = (
    "key_cache",
    "value_cache",
    "key_scale",
    "value_scale",
    "block_tables",
    "context_lengths",
)


class PagedAttentionDecodeFP8Tuner(OpTuner):
    op = OP
    key_columns = (
        "num_heads",
        "num_kv_heads",
        "head_size",
        "block_size",
        "context_max",
    )
    bucket_column = "batch_max"
    backends = BACKENDS
    atol = rtol = TOLERANCE
    outputs = ("output",)

    def make_inputs(self, shape: Row) -> Dict[str, Any]:
        heads, kv_heads = int(shape["num_heads"]), int(shape["num_kv_heads"])
        head_size, block_size = int(shape["head_size"]), int(shape["block_size"])
        lengths = _lengths(shape)
        cache = random_fp8_paged_cache(
            kv_heads,
            head_size,
            block_size,
            lengths,
            width=int(shape["context_max"]) // block_size,
        )
        query = torch.randn(
            len(lengths), heads, head_size, dtype=torch.bfloat16, device="cuda"
        )
        # MSLK's kernel needs its scratch preallocated to be captured in a CUDA
        # graph; aiter's backend leaves it unused.
        parts = get_recommended_splits(
            len(lengths), kv_heads, split_kv_blocks=KV_COMPUTE_BLOCK // block_size
        )
        scratch = (len(lengths), kv_heads, parts, heads // kv_heads)
        inputs = {
            "output": torch.empty_like(query),
            "query": query,
            **dict(zip(_CACHE_KEYS, cache)),
            "max_context_partition_num": parts,
            "exp_sums": torch.empty(scratch, device="cuda"),
            "max_logits": torch.empty(scratch, device="cuda"),
            "temporary_output": torch.empty(
                *scratch, head_size, dtype=torch.bfloat16, device="cuda"
            ),
        }
        inputs["expected"] = self._expected(shape, inputs)
        return inputs

    def run(self, inputs: Dict[str, Any]) -> torch.Tensor:
        query = inputs["query"]
        paged_attention_decode_fp8(
            inputs["output"],
            query,
            inputs["key_cache"],
            inputs["value_cache"],
            inputs["context_lengths"],
            inputs["block_tables"],
            query.shape[-1] ** -0.5,
            inputs["key_scale"],
            inputs["value_scale"],
            max_context_partition_num=inputs["max_context_partition_num"],
            exp_sums=inputs["exp_sums"],
            max_logits=inputs["max_logits"],
            temporary_output=inputs["temporary_output"],
        )
        if torch.cuda.is_current_stream_capturing():
            return inputs["output"]  # timed: the statistics are for the check
        return torch.cat([inputs["output"].float().flatten(), self._stats(inputs)])

    def reference(self, shape: Row, inputs: Dict[str, Any]) -> torch.Tensor:
        """The expected output, then zeros where ``run`` appends each sequence's
        scale and relative errors, scaled so that the driver's elementwise
        ``atol`` enforces the op test's limits on them too."""
        expected = inputs["expected"]
        return torch.cat([expected.flatten(), expected.new_zeros(2 * len(expected))])

    def _stats(self, inputs: Dict[str, Any]) -> torch.Tensor:
        out = inputs["output"].float().flatten(1)
        expected = inputs["expected"].flatten(1)
        norm = expected.norm(dim=1)
        nonzero = norm > 0
        safe = torch.where(nonzero, norm, torch.ones_like(norm))
        scale = (out * expected).sum(1) / safe**2
        relative = (out - expected).norm(dim=1) / safe
        scale_error = torch.where(nonzero, scale - 1, out.norm(dim=1))
        relative_error = torch.where(nonzero, relative, out.norm(dim=1))
        return torch.cat(
            [
                scale_error * (TOLERANCE / MAX_SCALE_ERROR),
                relative_error * (TOLERANCE / MAX_RELATIVE_ERROR),
            ]
        )

    def _expected(self, shape: Row, inputs: Dict[str, Any]) -> torch.Tensor:
        query = inputs["query"]
        _, kv_heads, _, block_size, _ = inputs["key_cache"].shape
        head_size = query.shape[-1]
        out = torch.empty(query.shape, dtype=torch.float32, device="cuda")
        for b, n in enumerate(_lengths(shape)):
            blocks = inputs["block_tables"][b, : -(-n // block_size)].long()
            k, v = (
                x.reshape(-1, kv_heads, head_size)[:n]
                for x in dequantize_fp8_paged_cache(
                    *(inputs[key][blocks] for key in _CACHE_KEYS[:4])
                )
            )
            q = query[b].float().view(kv_heads, -1, head_size)
            scores = torch.einsum("hgd,thd->hgt", q, k) * head_size**-0.5
            out[b] = torch.einsum("hgt,thd->hgd", scores.softmax(-1), v).flatten(0, 1)
        return out

    def flops(self, shape: Row) -> float:
        tokens = sum(_lengths(shape))
        return 4.0 * tokens * int(shape["num_heads"]) * int(shape["head_size"])

    def bytes_moved(self, shape: Row) -> float:
        # FP8 K and V, one FP32 scale each per token; the queries and outputs
        # are negligible.
        tokens = sum(_lengths(shape))
        kv_heads, head_size = int(shape["num_kv_heads"]), int(shape["head_size"])
        return tokens * kv_heads * (2 * head_size + 8)


def _part(profile: str, kind: str = "table") -> str:
    suffix = "" if kind == "table" else ".log"
    return str(PARTS / f"paged_attention_decode_fp8.{profile}{suffix}.csv")


def combine(tuner: OpTuner, out: str) -> None:
    """Write ``out`` from the two runs: aiter serves a shape if it was correct
    in both, at least ``MIN_IMPROVEMENT_PCT`` faster with full-length
    contexts, and with short contexts at most ``MAX_SHORT_LOSS_US`` and
    ``MAX_SHORT_SLOWDOWN`` times slower. Timings are the full run's; the short
    run's are added."""
    full, short = (tuning.read_csv_rows(_part(p)) for p in PROFILES)
    # Each run's log has every candidate's check; the last entry wins.
    checks: Dict[tuple, Row] = {}
    for p in PROFILES:
        for r in tuning.read_csv_rows(_part(p, "log")):
            checks[(p, tuner.row_key(r), r["candidate"])] = r
    short_by_key = {tuner.row_key(r): r for r in short}
    rows = []
    for row in full:
        key = tuner.row_key(row)
        other = short_by_key.get(key)
        if other is None:
            continue  # not tuned in both runs yet
        correct = all(
            checks.get((p, key, AITER), {}).get("correct") == "True" for p in PROFILES
        )
        use_aiter = (
            correct
            and float(row["aiter_us"]) * (1 + MIN_IMPROVEMENT_PCT / 100)
            < float(row["mslk_us"])
            and float(other["aiter_us"]) - float(other["mslk_us"]) <= MAX_SHORT_LOSS_US
            and float(other["aiter_us"]) <= MAX_SHORT_SLOWDOWN * float(other["mslk_us"])
        )
        backend = AITER if use_aiter else MSLK
        errors = [checks.get((p, key, backend), {}) for p in PROFILES]
        rows.append(
            {
                **row,
                "backend": backend,
                "us": row[f"{backend}_us"],
                "err_ratio": max((e.get("err_ratio", "0") for e in errors), key=float),
                "max_abs_err": max(
                    (e.get("max_abs_err", "0") for e in errors), key=float
                ),
                **{f"short_{b}_us": other[f"{b}_us"] for b in tuner.backends},
            }
        )
    columns = tuner.table_columns() + [f"short_{b}_us" for b in tuner.backends]
    tuning.write_csv_rows(out, rows, columns)
    print(
        f"Table: {out} ({sum(r['backend'] == AITER for r in rows)} of {len(rows)} rows to aiter)"
    )


if __name__ == "__main__":
    tuner = PagedAttentionDecodeFP8Tuner()
    profile = os.environ.get(PROFILE_ENV)
    if profile:  # one run, through the shared driver
        tuning.main(tuner, untune_file=str(UNTUNED), tune_file=_part(profile))
    else:
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument("-o", "--tune_file", default=str(TABLE))
        parser.add_argument("--parts_dir", default=str(PARTS))
        args, driver_args = parser.parse_known_args()
        PARTS = Path(args.parts_dir)
        if {"-h", "--help"} & set(driver_args):
            print(__doc__)
            sys.exit(
                subprocess.run(
                    [sys.executable, __file__, "-h"],
                    env={**os.environ, PROFILE_ENV: "full"},
                ).returncode
            )
        for p in PROFILES:
            subprocess.run(
                [sys.executable, __file__, *driver_args, "-o2", _part(p, "log")],
                env={**os.environ, PROFILE_ENV: p, PARTS_ENV: args.parts_dir},
                check=True,
            )
        combine(tuner, args.tune_file)
