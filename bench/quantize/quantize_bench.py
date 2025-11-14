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
from typing import Any, Callable, Optional

import click

import pandas as pd
import torch

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
    op_name: str

    sim: float = 0.0
    us: float = 0.0
    gbps: float = 0.0

    def __str__(self) -> str:
        return (
            f"{self.op_name} sim: {self.sim:.3f}.\n"
            f"{self.op_name} us: {self.us:.3f}.\n"
            f"{self.op_name} GB/s: {self.gbps:.3f}."
        )


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
                    f"Shape {shape} not found in shape registry. Valid shapes: {", ".join(shape_registry.keys())}."
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
    use_cuda_graph: bool = True,
    num_iters: int = 1,
) -> dict[str, Any]:
    # Create input tensors.
    A = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)

    # Keep track of results.
    results: dict[str, Any] = {"M": m, "K": k}
    # Benchmark each operator.
    for quantize_op in quantize_ops:
        metrics = Metrics(op_name=quantize_op.name)
        quantized = quantize_op.quantize(A)
        dequantized = quantize_op.dequantize(*quantized)
        metrics.sim = torch.mean(torch.pow(dequantized - A, 2)).item()

        for _ in range(num_iters):
            ms_runtime = quantize_op.benchmark(
                A,
                use_cuda_graph=use_cuda_graph,
            )

            input_bytes = A.numel() * A.element_size()
            output_bytes = sum(t.numel() * t.element_size() for t in quantized)
            metrics.gbps += (input_bytes + output_bytes) / (ms_runtime / 1e3) / 1e9
            metrics.us += ms_runtime * 1000

        # Print out results for this op.
        metrics.us /= num_iters
        metrics.gbps /= num_iters
        print(f"Average metrics over {num_iters} iterations:")
        print(metrics)

        # Save results for this operator.
        results[f"{quantize_op.name}_us"] = metrics.us
        results[f"{quantize_op.name}_gb/s"] = metrics.gbps
        results[f"{quantize_op.name}_sim"] = metrics.sim

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
    "--shapes",
    default=None,
    help=f"Specific model shapes to use, options: {", ".join(shape_registry.keys())}.",
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
    shapes: Optional[str],
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

    MK = get_problem_shapes(shapes, m, k, pair_mk)
    # Iterate over shapes and benchmark.
    benchmark_results = []
    for M, K in MK:
        print(f"Benchmarking M={M}, K={K}.")
        quantize_measurements = benchmark(
            quantize_ops,
            M,
            K,
            not no_cuda_graph,
            num_iters,
        )
        benchmark_results.append(quantize_measurements)
    if export_csv:
        os.makedirs(output_dir, exist_ok=True)
        datetime_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_file = os.path.join(
            output_dir, f"quantize_ops_benchmark_{datetime_str}.csv"
        )
        print(f"CSV saved to {csv_file}")
        # Export results to a CSV file.
        df = pd.DataFrame(benchmark_results)
        df.to_csv(csv_file, na_rep="NaN", index=False)


if __name__ == "__main__":
    invoke_main()
