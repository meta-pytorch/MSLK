# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import unittest
from unittest import mock

import torch
from mslk.attention.fmha import dispatch
from mslk.attention.fmha.common import Inputs


class _Op:
    NAME = "test_op"


def _inputs(query: torch.Tensor) -> Inputs:
    return Inputs(query=query, key=query, value=query)


class ReportDispatchTest(unittest.TestCase):
    def test_skipped_under_torch_compile(self) -> None:
        sample_factory = mock.Mock()
        scuba_factory = mock.Mock()

        def report(query: torch.Tensor) -> torch.Tensor:
            dispatch._report_dispatch("fw", _Op, _inputs(query))
            return query + 1

        with (
            mock.patch.object(dispatch, "Sample", sample_factory),
            mock.patch.object(dispatch, "ScubaData", scuba_factory),
            mock.patch.object(dispatch, "_usage_seen", set()),
        ):
            compiled_report = torch.compile(report, backend="eager", fullgraph=True)
            actual = compiled_report(torch.zeros(1))

        self.assertTrue(torch.equal(actual, torch.ones(1)))
        sample_factory.assert_not_called()
        scuba_factory.assert_not_called()

    def test_reported_in_eager(self) -> None:
        sample = mock.Mock()
        scuba = mock.MagicMock()
        scuba.__enter__.return_value = scuba
        sample_factory = mock.Mock(return_value=sample)
        scuba_factory = mock.Mock(return_value=scuba)

        with (
            mock.patch.object(dispatch, "Sample", sample_factory),
            mock.patch.object(dispatch, "ScubaData", scuba_factory),
            mock.patch.object(dispatch, "justknobs_check", return_value=True),
            mock.patch.object(dispatch, "_usage_seen", set()),
        ):
            dispatch._report_dispatch("fw", _Op, _inputs(torch.zeros(1)))

        sample_factory.assert_called_once_with()
        scuba_factory.assert_called_once_with("mslk_fmha_dispatch")
        scuba.addSample.assert_called_once_with(sample)
