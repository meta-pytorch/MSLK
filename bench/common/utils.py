# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import contextlib
import os
import tempfile
import uuid

import torch
from torch.profiler import profile, ProfilerActivity  # pyre-ignore


def profiler(
    enabled: bool,
    with_stack: bool = False,
    record_shapes: bool = False,
):
    """
    Returns a profiler context manager if enabled, otherwise a null context.

    When enabled, profiles CPU and CUDA activities.

    Args:
        enabled: Whether to enable profiling.
        with_stack: Whether to record stack traces.
        record_shapes: Whether to record tensor shapes.

    Returns:
        A context manager - either a torch profiler or nullcontext.
    """

    def _kineto_trace_handler(p: torch.profiler.profile) -> None:
        trace_filename = f"mslk_{os.getpid()}_{uuid.uuid4().hex}.json"

        if os.path.exists("/etc/fbwhoami"):
            trace_url = f"manifold://gpu_traces/tree/accelerator/{trace_filename}"
        else:
            trace_url = os.path.join(tempfile.gettempdir(), trace_filename)

        p.export_chrome_trace(trace_url)

    return (
        profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],  # pyre-ignore
            on_trace_ready=_kineto_trace_handler,
            with_stack=with_stack,
            record_shapes=record_shapes,
        )
        if enabled
        else contextlib.nullcontext()
    )
