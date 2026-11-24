# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from unittest import mock

import torch
from mslk.attention.fmha.common import Inputs
from mslk.attention.fmha.flydsl import BwOp

gpu_unavailable = not torch.cuda.is_available()


class FlyDSLBackwardArchitectureTest(unittest.TestCase):
    @unittest.skipIf(gpu_unavailable, "requires a GPU")
    def test_native_backward_declines_unimplemented_gpu_architecture(self) -> None:
        query = torch.empty((1, 16, 2, 64), device="cuda", dtype=torch.float16)
        inputs = Inputs(query=query, key=query, value=query)
        if torch.version.hip:
            arch = torch.cuda.get_device_properties(query.device).gcnArchName.split(
                ":", 1
            )[0]
            # Library availability covers all kernel families. The backward
            # MFMA kernel must retain its own instruction-architecture bound
            # when a provider adds support for other kernels on a new GPU.
            with mock.patch(
                "mslk.attention.fmha.flydsl._is_flydsl_available", return_value=True
            ):
                reasons = BwOp.not_supported_reasons(inputs)
            if arch in {"gfx942", "gfx950"}:
                self.assertFalse(reasons, reasons)
            else:
                self.assertTrue(
                    reasons,
                    "native MFMA backward admitted an unimplemented GPU architecture",
                )
        else:
            self.assertTrue(
                BwOp.not_supported_reasons(inputs),
                "HIP-only backward admitted a CUDA device",
            )
