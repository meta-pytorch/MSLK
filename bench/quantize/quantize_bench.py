# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import itertools
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from typing import Callable, Optional

import click
import pandas as pd
import torch
import triton  # @manual=//triton:triton
from mslk.bench.common.utils import BenchOptions, profiler
from mslk.bench.quantize.quantize_ops import get_ops, QuantizeOpBase
from tabulate import tabulate

type ShapeFunction = Callable[[], list[tuple[int, int]]]

shape_registry: dict[str, ShapeFunction] = {}


def register_shapes(name: str) -> Callable[[ShapeFunction], ShapeFunction]:
    def decorator(
        shape_function: ShapeFunction,
    ) -> ShapeFunction:
        shape_registry[name] = shape_function
        return shape_function

    return decorator


@register_shapes("llm_eval")
def llm_eval() -> list[tuple[int, int]]:
    return [
        (1, 5120),
        (1024, 5120),
        (2000, 5120),
        (4096, 5120),
        (16384, 5120),
        (1024, 7168),
        (4096, 4096),
    ]


@register_shapes("decode_1024")
def decode_1024_shapes() -> list[tuple[int, int]]:
    return [
        (1, 1024),
        (1, 2048),
        (1, 4096),
        (1, 5120),
        (1, 6144),
        (1, 7168),
        (1, 8192),
    ]


@register_shapes("prefill_1024")
def prefill_1024_shapes() -> list[tuple[int, int]]:
    shapes = []
    for M in [2048, 4096, 8192, 16384]:
        shapes += [
            (M, 1024),
            (M, 2048),
            (M, 4096),
            (M, 5120),
            (M, 6144),
            (M, 7168),
            (M, 8192),
        ]
    return shapes


@dataclass
class Metrics:
    op: str
    M: int = 0
    K: int = 0
    sim: float = 0.0
    us: float = 0.0
    gbps: float = 0.0
    memory_bw_util: float = 0.0

    @staticmethod
    def header() -> str:
        header = f"{'OpName':<20} {'Problem Shape':<15} {'Sim':<10} {'Us':<10} {'GB/s':<10} {'Mem BW Util %':<10}"
        divider = "-" * len(header)
        return f"Quantize Bench\n{divider}\n{header}\n{divider}"

    def __str__(self) -> str:
        problem_shape = f"({self.M}, {self.K})"
        return f"{self.op:<20} {problem_shape:<15} {self.sim:<10.3f} {self.us:<10.3f} {self.gbps:<10.2f} {self.memory_bw_util:<10.2f}"

    def as_dict(self) -> dict[str, float]:
        return {
            "M": self.M,
            "K": self.K,
            f"{self.op}_sim": self.sim,
            f"{self.op}_us": self.us,
            f"{self.op}_gb/s": self.gbps,
            f"{self.op}_memory_bw_util": self.memory_bw_util,
        }


def get_problem_shapes(
    shapes: Optional[str],
    m: Optional[str],
    k: Optional[str],
    pair_mk: bool,
) -> list[tuple[int, int]]:
    if shapes:
        all_shapes = set()

        for shape in shapes.strip().split(","):
            if shape not in shape_registry:
                print(
                    f"Shape {shape} not found in shape registry. Valid shapes: {', '.join(shape_registry.keys())}."
                )
                sys.exit(1)
            all_shapes.update(shape_registry[shape]())

        return list(all_shapes)

    if m is None:
        raise Exception("M must be non-empty.")
    M = [int(m_val) for m_val in m.strip().split(",")]
    if k is None:
        raise Exception("K must be non-empty.")
    K = [int(k_val) for k_val in k.strip().split(",")]

    if pair_mk:
        if len(M) != len(K):
            raise Exception("M and K must be the same length in pair_MK mode.")
        return list(zip(M, K))
    else:
        return list(itertools.product(M, K))


def benchmark(
    quantize_ops: list[QuantizeOpBase],
    m: int,
    k: int,
    mem_bw_roofline_gbps: float,
    opts: BenchOptions,
) -> list[Metrics]:
    # Create input tensors.
    input = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)

    # Keep track of results.
    results = []
    # Benchmark each operator.
    for quantize_op in quantize_ops:
        metrics = Metrics(op=quantize_op.name, M=m, K=k)
        args = quantize_op.preprocess(input)
        quantized = quantize_op.quantize(input, *args)
        dequantized = quantize_op.dequantize(*quantized)
        metrics.sim = torch.mean(torch.pow(dequantized - input, 2)).item()

        for _ in range(opts.num_iters):
            with profiler(enabled=opts.trace, with_stack=True):
                ms_runtime = quantize_op.benchmark(
                    input,
                    args,
                    opts=opts,
                )

            input_bytes = input.numel() * input.element_size()
            output_bytes = sum(t.numel() * t.element_size() for t in quantized)
            total_size_bytes = input_bytes + output_bytes
            gbps = (total_size_bytes / 1e9) / (ms_runtime / 1e3)
            metrics.gbps += gbps
            metrics.us += ms_runtime * 1000
            metrics.memory_bw_util += (gbps / mem_bw_roofline_gbps) * 100

        metrics.us /= opts.num_iters
        metrics.gbps /= opts.num_iters
        metrics.memory_bw_util /= opts.num_iters

        results.append(metrics)

    return results


def collect_kernels_to_profile(kernels: Optional[list[str]]) -> list[QuantizeOpBase]:
    # Get existing quantization operators.
    quantize_ops = [op for op in get_ops() if op.supported]
    if kernels is None:
        return quantize_ops
    return [op for op in quantize_ops if op.name in kernels]


def print_kernels(kernels: Optional[list[str]]) -> None:
    data = sorted(
        [
            (op.name, "Yes" if op.cuda else "No", "Yes" if op.hip else "No")
            for op in get_ops()
        ]
    )
    print(tabulate(data, headers=["Name", "CUDA", "ROCm"], tablefmt="orgtbl"))


@click.command()
@click.option(
    "--output-dir",
    default="/tmp",
    help="Directory to save plots and csvs to",
)
@click.option(
    "--num-iters",
    default=1,
    type=int,
    help="Number of iterations to repeat each benchmark.",
)
@click.option(
    "--export-csv",
    is_flag=True,
    help="Export results to a CSV file.",
)
@click.option(
    "--kernels",
    default=None,
    help="Comma separated list of kernels to benchmark. Defaults to all kernels.",
)
@click.option(
    "--M",
    default=None,
    help="Comma separated list of M values to benchmark.",
)
@click.option(
    "--K",
    default=None,
    help="Comma separated list of K values to benchmark.",
)
@click.option(
    "--pair-MK",
    is_flag=True,
    help="If set, instead of benchmarking cartesian product of M * K, benchmark consecutive MK pairs together.",
)
@click.option(
    "--no-cuda-graph",
    is_flag=True,
    help="If set, do not use cuda graph for benchmarking.",
)
@click.option(
    "--no-rotating-buffer",
    is_flag=True,
    help="If set, do not use rotating buffer for benchmarking.",
)
@click.option(
    "--shapes",
    default=None,
    help=f"Specific model shapes to use, options: {', '.join(shape_registry.keys())}.",
)
@click.option(
    "--trace",
    is_flag=True,
    help="If set, produce a performance trace of the benchmark.",
)
def invoke_main(
    output_dir: str,
    num_iters: int,
    export_csv: bool,
    kernels: Optional[str],
    m: Optional[str],
    k: Optional[str],
    pair_mk: bool,
    no_cuda_graph: bool,
    no_rotating_buffer: bool,
    shapes: Optional[str],
    trace: bool,
) -> None:
    # If kernel filter is provided, parse it. Else, benchmark all kernels.
    all_kernels = kernels.strip().split(",") if kernels else None
    quantize_ops = collect_kernels_to_profile(all_kernels)

    if len(quantize_ops) == 0:
        print("No valid kernels to benchmark. Available kernels:")
        print_kernels(all_kernels)
        sys.exit(1)

    if num_iters < 1:
        print("Warning: Number of iterations must be at least 1.")
        num_iters = 1

    mem_bw_roofline_gbps = triton.testing.get_dram_gbps()
    MK = get_problem_shapes(shapes, m, k, pair_mk)

    opts = BenchOptions(
        num_iters=num_iters,
        cuda_graph=not no_cuda_graph,
        rotating_buffer=not no_rotating_buffer,
        trace=trace,
    )

    # Iterate over shapes and benchmark.
    benchmark_results = []
    csv = []
    for M, K in MK:
        quantize_measurements = benchmark(
            quantize_ops,
            M,
            K,
            mem_bw_roofline_gbps,
            opts,
        )
        benchmark_results.extend(quantize_measurements)
        csv_row = {}
        for metric in quantize_measurements:
            csv_row.update(metric.as_dict())
        csv.append(csv_row)

    print(Metrics.header())
    for metric in benchmark_results:
        print(metric)

    print("")
    print(f"Hardware: {torch.cuda.get_device_name()}")
    print(f"    Memory BW Roofline: {mem_bw_roofline_gbps} GB/s")

    print("")
    print("Benchmark Settings:")
    print(f"    CUDA graph: {opts.cuda_graph}")
    print(f"    Buffer rotation: {opts.rotating_buffer}")

    if export_csv:
        os.makedirs(output_dir, exist_ok=True)
        datetime_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_file = os.path.join(
            output_dir, f"quantize_ops_benchmark_{datetime_str}.csv"
        )
        # Export results to a CSV file.
        df = pd.DataFrame(csv)
        df.to_csv(csv_file, na_rep="NaN", index=False)
        print(f"CSV saved to {csv_file}")


if __name__ == "__main__":
    invoke_main()
