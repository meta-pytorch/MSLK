# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import contextlib
import copy
import os
import tempfile
import uuid
from dataclasses import dataclass
from typing import Any, Callable

import torch
import triton  # @manual=//triton:triton
from torch.profiler import profile, ProfilerActivity  # pyre-ignore


@dataclass
class BenchOptions:
    """Common benchmark options used across all benchmark scripts.

    This dataclass encapsulates all configuration options for running benchmarks
    in the MSLK benchmark suite. It is used by gemm_bench, conv_bench, and
    quantize_bench to maintain consistent benchmarking behavior.

    Attributes:
        num_iters: Number of iterations to repeat each benchmark for averaging.
        cuda_graph: Whether to use CUDA graphs for benchmarking. CUDA graphs
            reduce kernel launch overhead and provide more accurate measurements
            for GPU-bound workloads.
        rotating_buffer: Whether to use a rotating buffer during benchmarking.
            This helps flush L2/L3 cache between iterations to get more realistic
            memory-bound performance measurements.
        rep_ms: Repetition time in milliseconds for triton.testing.do_bench.
            Controls how long each benchmark runs before measuring.
        trace: Whether to produce a performance trace of the benchmark using
            PyTorch profiler. Traces are saved to Manifold or temp directory.
        fast_accum: Whether to enable fast accumulation for FP8 implementations.
            This is only relevant for Hopper GPUs.
        torch_compile: Whether to use torch.compile for applicable operations.
    """

    num_iters: int = 1
    cuda_graph: bool = True
    rotating_buffer: bool = False
    rep_ms: int = 200
    trace: bool = False
    fast_accum: bool = False
    torch_compile: bool = False


def _do_bench(
    fn: Callable[[], Any],
    opts: BenchOptions,
) -> float:
    if opts.cuda_graph:
        with torch.cuda.stream(torch.cuda.Stream()):
            return triton.testing.do_bench_cudagraph(fn, rep=opts.rep_ms)
    else:
        return triton.testing.do_bench(fn, rep=opts.rep_ms)


def do_bench(
    fn: Callable[..., Any],
    args: tuple[Any, ...],
    opts: BenchOptions,
) -> float:
    """
    Benchmark a function using triton's benchmarking utilities.

    Args:
        fn: The function to benchmark.
        args: Tuple of arguments to pass to the function.
        opts: Benchmark options

    Returns:
        The runtime in milliseconds.
    """
    if not opts.rotating_buffer:
        return _do_bench(lambda: fn(*args), opts)

    # Calculate input size to determine how many copies we need.
    input_size_bytes = sum(
        t.element_size() * t.numel() for t in args if isinstance(t, torch.Tensor)
    )

    # Use a 50MB buffer, this may need to be variable in the future.
    rotating_buffer_size_bytes = 50 * 1024 * 1024
    # Make at least one copy of the inputs.
    copy_cnt = max(rotating_buffer_size_bytes // input_size_bytes, 1)

    args_list = [args]
    for _ in range(copy_cnt):
        args_list.append(copy.deepcopy(args))

    # We benchmark on a different stream, so a sync is required.
    torch.cuda.synchronize()

    def rotating_buffer_fn() -> None:
        for a in args_list:
            fn(*a)

    return _do_bench(rotating_buffer_fn, opts) / len(args_list)


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
