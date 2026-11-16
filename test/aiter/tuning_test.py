# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The parts of the aiter tuning driver that decide and keep results.

Choosing a backend, merging and resuming tables, and sizing the input rotation
are plain logic, tested here without a GPU or aiter.
"""

import io
import os
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from typing import Any, Dict
from unittest import mock

import torch
from mslk.aiter.routing import choose_backend, MSLK, ShapeTable
from mslk.aiter.tuning import (
    _error_stats,
    bucket_lower_edges,
    choose,
    main,
    Measurement,
    merge_rows,
    merge_tables,
    OpTuner,
    read_csv_rows,
    rotation_copies,
    Row,
    Settings,
    ShapeResult,
    shapes_to_tune,
    table_row,
    tune_shape,
    write_csv_rows,
)

GFX950 = {"gfx": "gfx950", "cu_num": "256"}
KEYS = ["gfx", "cu_num", "N", "K", "M_max"]


class _Tuner(OpTuner):
    op = "op"
    key_columns = ("N", "K")
    bucket_column = "M_max"
    backends = (MSLK, "aiter")
    atol = rtol = 1e-2

    def make_inputs(self, shape: Row) -> Dict[str, Any]:
        return {}

    def run(self, inputs: Dict[str, Any]) -> torch.Tensor:
        return torch.empty(0)

    def reference(self, shape: Row, inputs: Dict[str, Any]) -> torch.Tensor:
        return torch.empty(0)


def _m(backend: str, us: float | None, correct: bool = True) -> Measurement:
    return Measurement(backend, us, correct, 0.0 if correct else 0.5, 0.0)


def _row(n: str, m: str, backend: str, device: Row = GFX950) -> Row:
    return {**device, "N": n, "K": "512", "M_max": m, "backend": backend}


class ChooseTest(unittest.TestCase):
    def test_another_backend_needs_the_minimum_improvement(self) -> None:
        self.assertEqual(choose([_m(MSLK, 10.0), _m("aiter", 9.8)], 3.0), MSLK)
        self.assertEqual(choose([_m(MSLK, 10.0), _m("aiter", 9.5)], 3.0), "aiter")

    def test_mslk_wins_when_faster(self) -> None:
        self.assertEqual(choose([_m(MSLK, 5.0), _m("aiter", 9.0)], 3.0), MSLK)

    def test_incorrect_or_failed_candidates_are_never_chosen(self) -> None:
        self.assertEqual(
            choose([_m(MSLK, 10.0), _m("aiter", 1.0, correct=False)], 3.0), MSLK
        )
        failed = Measurement("aiter", None, False, 1.0, float("inf"), "OutOfResources")
        self.assertEqual(choose([_m(MSLK, 10.0), failed], 3.0), MSLK)


class TableFilesTest(unittest.TestCase):
    def test_merge_replaces_by_key_and_keeps_other_devices(self) -> None:
        other = {"gfx": "gfx942", "cu_num": "304"}
        old = [
            _row("8", "1", "aiter"),
            _row("8", "2", "aiter"),
            _row("8", "1", MSLK, other),
        ]
        new = [_row("8", "1", MSLK), _row("8", "1", MSLK)]
        merged = merge_rows(old, new, KEYS)
        self.assertEqual(len(merged), 3)
        by_key = {tuple(r[k] for k in KEYS): r["backend"] for r in merged}
        self.assertEqual(by_key[("gfx950", "256", "8", "512", "1")], MSLK)
        self.assertEqual(by_key[("gfx950", "256", "8", "512", "2")], "aiter")
        self.assertEqual(by_key[("gfx942", "304", "8", "512", "1")], MSLK)

    def test_rows_sort_numerically(self) -> None:
        merged = merge_rows([], [_row("8", "16", MSLK), _row("8", "2", MSLK)], KEYS)
        self.assertEqual([r["M_max"] for r in merged], ["2", "16"])

    def test_write_keeps_unknown_columns_and_merge_tables_dedups(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            table, part = os.path.join(tmp, "t.csv"), os.path.join(tmp, "p.csv")
            write_csv_rows(
                table, [{**_row("8", "1", "aiter"), "old_col": "x"}], KEYS + ["backend"]
            )
            write_csv_rows(
                part, [_row("8", "1", MSLK), _row("8", "2", MSLK)], KEYS + ["backend"]
            )
            self.assertEqual(merge_tables(table, [part]), 2)
            rows = read_csv_rows(table)
            self.assertEqual([r["backend"] for r in rows], [MSLK, MSLK])
            self.assertIn("old_col", rows[0])

    def test_merge_tables_rejects_mismatched_key_columns(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            table = os.path.join(tmp, "table.csv")
            missing = os.path.join(tmp, "missing.csv")
            extra = os.path.join(tmp, "extra.csv")
            original = [_row("8", "1", MSLK)]
            write_csv_rows(table, original, KEYS + ["backend"])
            write_csv_rows(
                missing,
                [{k: v for k, v in original[0].items() if k != "K"}],
                [c for c in KEYS if c != "K"] + ["backend"],
            )
            write_csv_rows(
                extra,
                [{**original[0], "tile": "32"}],
                KEYS + ["tile", "backend"],
            )

            for part in (missing, extra):
                with self.subTest(part=part):
                    with self.assertRaisesRegex(ValueError, "key columns"):
                        merge_tables(table, [part])
                    self.assertEqual(read_csv_rows(table), original)


class ResumeTest(unittest.TestCase):
    def test_only_shapes_missing_for_this_device_are_tuned(self) -> None:
        untuned = [{"N": "8", "K": "512", "M_max": m} for m in ("1", "2", "4", "2")]
        table = [
            _row("8", "1", MSLK),
            _row("8", "2", MSLK, {"gfx": "gfx942", "cu_num": "304"}),
        ]
        todo = shapes_to_tune(untuned, table, GFX950, _Tuner(), retune_all=False)
        self.assertEqual([s["M_max"] for s in todo], ["2", "4"])
        todo = shapes_to_tune(untuned, table, GFX950, _Tuner(), retune_all=True)
        self.assertEqual([s["M_max"] for s in todo], ["1", "2", "4"])


class RowFormatTest(unittest.TestCase):
    def test_failed_candidate_has_an_empty_time(self) -> None:
        failed = Measurement("aiter", None, False, 1.0, float("inf"), "OutOfResources")
        result = ShapeResult(
            shape={"N": "8", "K": "512", "M_max": "1"},
            device=GFX950,
            measurements=[_m(MSLK, 2.0), failed],
            backend=MSLK,
            flops=2.0 * 8 * 512,
        )
        row = table_row(_Tuner(), result)
        self.assertEqual(row["backend"], MSLK)
        self.assertEqual(row["mslk_us"], "2.00")
        self.assertEqual(row["aiter_us"], "")
        self.assertNotIn("inf", ",".join(row.values()))
        self.assertEqual(list(row), _Tuner().table_columns())


class ErrorStatsTest(unittest.TestCase):
    def test_nan_and_inf_count_as_outside_tolerance(self) -> None:
        ref = torch.ones(4)
        out = torch.tensor([1.0, float("nan"), float("inf"), 1.005])
        err_ratio, _ = _error_stats(_Tuner(), out, ref)
        self.assertEqual(err_ratio, 0.5)


class _TensorTuner(_Tuner):
    outputs = ("out",)

    def __init__(self, behavior: str = "correct", backend: str = "aiter") -> None:
        self.behavior = behavior
        self.backend = backend
        self.table = ShapeTable([], self.key_columns, self.bucket_column)

    def make_inputs(self, shape: Row) -> Dict[str, Any]:
        x = torch.arange(int(shape[self.bucket_column]), dtype=torch.float32)
        return {"x": x, "out": torch.empty_like(x)}

    def reference(self, shape: Row, inputs: Dict[str, Any]) -> torch.Tensor:
        return inputs["x"] + 1

    def run(self, inputs: Dict[str, Any]) -> torch.Tensor:
        backend = choose_backend(self.op, lambda: self.table, 0, (), 1)
        x, out = inputs["x"], inputs["out"]
        behavior = self.behavior if backend == self.backend else "correct"
        if behavior == "raise":
            raise RuntimeError("candidate launch failed")
        if behavior == "interrupt" and len(x) == 4:
            raise KeyboardInterrupt
        if behavior == "partial":
            out[:-1].copy_(x[:-1] + 1)
        elif behavior == "wrong_lower" and len(x) == 1:
            out.zero_()
        else:
            out.copy_(x + 1)
        return out


class TuneShapeTest(unittest.TestCase):
    def _run(self, tuner: _TensorTuner) -> ShapeResult:
        with (
            mock.patch("mslk.aiter.tuning.device_key", return_value=("gfx950", 256)),
            mock.patch("torch.cuda.current_device", return_value=0),
            mock.patch("torch.cuda.synchronize"),
            mock.patch("mslk.aiter.tuning.time_us", side_effect=[10.0, 1.0]),
        ):
            return tune_shape(
                tuner,
                {"N": "8", "K": "512", "M_max": "4"},
                Settings(64, 1, 1, 3.0),
                order_flip=False,
                lower_edge=1,
            )

    def test_partial_write_cannot_reuse_a_previous_output(self) -> None:
        result = self._run(_TensorTuner("partial"))
        self.assertEqual(result.backend, MSLK)
        self.assertFalse(result.measurements[1].correct)
        self.assertEqual(result.measurements[1].err_ratio, 1.0)

    def test_candidate_exception_is_recorded_and_mslk_is_chosen(self) -> None:
        result = self._run(_TensorTuner("raise"))
        candidate = result.measurements[1]
        self.assertEqual(result.backend, MSLK)
        self.assertFalse(candidate.correct)
        self.assertIsNone(candidate.us)
        assert candidate.error is not None
        self.assertIn("candidate launch failed", candidate.error)

    def test_fast_candidate_must_be_correct_at_the_lower_bucket_edge(self) -> None:
        result = self._run(_TensorTuner("wrong_lower"))
        self.assertEqual(result.backend, MSLK)
        self.assertFalse(result.measurements[1].correct)
        self.assertEqual(result.measurements[1].err_ratio, 1.0)

    def test_mslk_failure_has_no_valid_backend(self) -> None:
        result = self._run(_TensorTuner("partial", MSLK))
        self.assertEqual(result.backend, "")
        self.assertFalse(result.measurements[0].correct)
        self.assertTrue(result.measurements[1].correct)


class DriverTest(unittest.TestCase):
    def test_interruption_flushes_completed_rows_and_resume_skips_them(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            untuned, table = (
                os.path.join(tmp, "input.csv"),
                os.path.join(tmp, "table.csv"),
            )
            profile = os.path.join(tmp, "profile.csv")
            shapes = [{"N": "8", "K": "512", "M_max": m} for m in ("1", "4")]
            write_csv_rows(untuned, shapes, ["N", "K", "M_max"])
            with (
                mock.patch("sys.argv", ["tune", "-o2", profile]),
                mock.patch(
                    "mslk.aiter.tuning.device_key", return_value=("gfx950", 256)
                ),
                mock.patch("torch.cuda.current_device", return_value=0),
                mock.patch("torch.cuda.synchronize"),
                mock.patch("mslk.aiter.tuning.time_us", return_value=10.0),
                redirect_stdout(io.StringIO()),
                redirect_stderr(io.StringIO()),
            ):
                with self.assertRaises(SystemExit) as exit_context:
                    main(_TensorTuner("interrupt", MSLK), untuned, table)
            self.assertEqual(exit_context.exception.code, 1)
            rows = read_csv_rows(table)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["M_max"], "1")
            self.assertEqual(len(read_csv_rows(profile)), 2)
            self.assertEqual(
                shapes_to_tune(shapes, rows, GFX950, _Tuner(), False), shapes[1:]
            )


class BucketEdgeTest(unittest.TestCase):
    def test_lower_edge_is_one_past_the_previous_bucket_for_the_same_key(self) -> None:
        untuned = [
            {"N": "8", "K": "512", "M_max": m} for m in ("64", "1", "16", "16")
        ] + [{"N": "9", "K": "512", "M_max": "64"}]
        edges = bucket_lower_edges(untuned, _Tuner())
        self.assertEqual(edges[("8", "512", "1")], 1)
        self.assertEqual(edges[("8", "512", "16")], 2)
        self.assertEqual(edges[("8", "512", "64")], 17)
        self.assertEqual(edges[("9", "512", "64")], 1)


class RotationTest(unittest.TestCase):
    def test_copies_cover_the_cache(self) -> None:
        mb = 1 << 20
        self.assertEqual(rotation_copies(100 * mb, 512 * mb), 6)
        self.assertEqual(rotation_copies(900 * mb, 512 * mb), 1)
        self.assertEqual(rotation_copies(1024, 512 * mb), 64)  # capped


if __name__ == "__main__":
    unittest.main()
