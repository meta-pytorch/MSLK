# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import itertools
import os
import sys

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Optional

import click

import matplotlib.pyplot as plt

import pandas as pd
import seaborn as sns
import torch
import triton  # @manual=//triton:triton

from mslk.bench.common import profiler
from mslk.bench.conv.conv_ops import ConvOpBase, get_conv_ops
from tabulate import tabulate


shape_registry = {}


def register_shapes(name):
    def decorator(op):
        shape_registry[name] = op
        return op

    return decorator


@register_shapes("default")
def default_shapes() -> (
    list[tuple[int, int, int, int, int, int, int, int, int, int, int, int]]
):
    """
    Default convolution shapes for benchmarking.

    Returns tuples of (N, D, H, W, C, K, T, R, S, pad, stride, dilation)
    where:
        N = batch size
        D, H, W = input spatial dimensions (depth, height, width)
        C = input channels
        K = output channels (filters)
        T, R, S = kernel spatial dimensions
        pad, stride, dilation = convolution parameters (applied uniformly to D, H, W)
    """
    shapes = []
    # Common batch sizes
    for N in [1, 4, 8, 16]:
        # Common configurations
        shapes += [
            # Small spatial dimensions, various channel sizes
            (N, 8, 8, 8, 64, 64, 3, 3, 3, 1, 1, 1),
            (N, 8, 8, 8, 64, 128, 3, 3, 3, 1, 1, 1),
            (N, 8, 8, 8, 128, 128, 3, 3, 3, 1, 1, 1),
            # Medium spatial dimensions
            (N, 16, 16, 16, 64, 64, 3, 3, 3, 1, 1, 1),
            (N, 16, 16, 16, 64, 128, 3, 3, 3, 1, 1, 1),
            # Larger spatial dimensions
            (N, 32, 32, 32, 32, 64, 3, 3, 3, 1, 1, 1),
            (N, 32, 32, 32, 64, 64, 3, 3, 3, 1, 1, 1),
            # 1x1x1 convolutions (common in ResNets)
            (N, 16, 16, 16, 64, 128, 1, 1, 1, 0, 1, 1),
            (N, 16, 16, 16, 128, 256, 1, 1, 1, 0, 1, 1),
        ]
    return shapes


@dataclass
class Metrics:
    op: str
    N: int = 0
    D: int = 0
    H: int = 0
    W: int = 0
    C: int = 0
    K: int = 0
    T: int = 0
    R: int = 0
    S: int = 0
    pad: int = 0
    stride: int = 0
    dilation: int = 0

    sim: float = 0.0
    ms: float = 0.0
    tflops: float = 0.0
    gbps: float = 0.0

    @staticmethod
    def header() -> str:
        header = (
            f"{'OpName':<20} {'(N,D,H,W,C,K,T,R,S,pad,stride,dilation)':<50} "
            f"{'Sim':<10} {'ms':<10} {'TFLOPS':<10} {'GB/s':<10}"
        )
        divider = "-" * len(header)
        return f"Conv Bench\n{divider}\n{header}\n{divider}"

    def __str__(self) -> str:
        problem_shape = (
            f"({self.N},{self.D},{self.H},{self.W},{self.C},{self.K},"
            f"{self.T},{self.R},{self.S},{self.pad},{self.stride},{self.dilation})"
        )
        return (
            f"{self.op:<20} {problem_shape:<50} "
            f"{self.sim:<10.3f} {self.ms:<10.3f} {self.tflops:<10.2f} {self.gbps:<10.2f}"
        )

    def as_dict(self) -> dict[str, float]:
        return {
            "N": self.N,
            "D": self.D,
            "H": self.H,
            "W": self.W,
            "C": self.C,
            "K": self.K,
            "T": self.T,
            "R": self.R,
            "S": self.S,
            "pad": self.pad,
            "stride": self.stride,
            "dilation": self.dilation,
            f"{self.op}_sim": self.sim,
            f"{self.op}_ms": self.ms,
            f"{self.op}_tflops": self.tflops,
            f"{self.op}_gb/s": self.gbps,
        }


def benchmark(
    conv_ops: list[ConvOpBase],
    n: int,
    d: int,
    h: int,
    w: int,
    c: int,
    k: int,
    t: int,
    r: int,
    s: int,
    pad: int,
    stride: int,
    dilation: int,
    bench_quantize: bool = False,
    use_cuda_graph: bool = True,
    trace: bool = False,
    num_iters: int = 1,
    torch_compile: bool = False,
) -> list[Metrics]:
    # Create input tensors in NCDHW format
    activation = torch.randn(n, c, d, h, w, device="cuda", dtype=torch.bfloat16)
    # Create filter tensors in KCTRS format
    filter = torch.randn(k, c, t, r, s, device="cuda", dtype=torch.bfloat16)

    # Convolution parameters (uniform for all dimensions)
    padding = [pad, pad, pad]
    stride_vec = [stride, stride, stride]
    dilation_vec = [dilation, dilation, dilation]

    # Compute baseline output for correctness checking using PyTorch
    # Convert to NCDHW for PyTorch
    out_ref = torch.nn.functional.conv3d(
        activation,
        filter,
        bias=None,
        stride=stride_vec,
        padding=padding,
        dilation=dilation_vec,
    )

    # Keep track of results.
    results = []

    # Benchmark each operator.
    for conv_op in conv_ops:
        print(
            f"Benchmarking {conv_op.name} with "
            f"(N={n}, D={d}, H={h}, W={w}, C={c}, K={k}, T={t}, R={r}, S={s}, "
            f"pad={pad}, stride={stride}, dilation={dilation})"
        )
        metrics = Metrics(
            op=conv_op.name,
            N=n,
            D=d,
            H=h,
            W=w,
            C=c,
            K=k,
            T=t,
            R=r,
            S=s,
            pad=pad,
            stride=stride,
            dilation=dilation,
        )
        if hasattr(conv_op, "torch_compile"):
            conv_op.torch_compile = torch_compile
        # Preprocess data if needed.
        preprocessed_args = conv_op.preprocess(
            activation, filter, padding, stride_vec, dilation_vec
        )
        # Get the quantized tensors for this operator.
        quantized_vals = conv_op.quantize(*preprocessed_args)
        # Compute the output given quantized values.
        output = conv_op.compute(*quantized_vals)
        # Compare the quantize op output to reference as a sanity check.
        metrics.sim = torch.mean(torch.pow(output - out_ref, 2)).item()

        # Compute output spatial dimensions
        z = 1 + (d + 2 * pad - ((t - 1) * dilation + 1)) // stride
        p = 1 + (h + 2 * pad - ((r - 1) * dilation + 1)) // stride
        q = 1 + (w + 2 * pad - ((s - 1) * dilation + 1)) // stride

        for _ in range(num_iters):
            # Now perform benchmark.
            if bench_quantize:
                # Benchmark both quantize and compute.
                with profiler(enabled=trace, with_stack=True):
                    ms_runtime = conv_op.benchmark(
                        *preprocessed_args,
                        bench_quantize=True,
                        use_cuda_graph=use_cuda_graph,
                    )
            else:
                with profiler(enabled=trace, with_stack=True):
                    ms_runtime = conv_op.benchmark(
                        *quantized_vals,
                        bench_quantize=False,
                        use_cuda_graph=use_cuda_graph,
                    )

            # Compute performance metrics
            # FLOPs for convolution: 2 * N * Z * P * Q * K * T * R * S * C
            flops = 2 * n * z * p * q * k * t * r * s * c
            metrics.tflops += flops / (ms_runtime / 1e3) / 1e12

            # Compute memory bandwidth
            # Input: N * D * H * W * C, Filter: K * T * R * S * C, Output: N * Z * P * Q * K
            input_size = n * d * h * w * c * quantized_vals[0].element_size()
            filter_size = k * t * r * s * c * quantized_vals[1].element_size()
            output_size = n * z * p * q * k * output.element_size()

            metrics.gbps += (
                (input_size + filter_size + output_size) / (ms_runtime / 1e3) / 1e9
            )
            metrics.ms += ms_runtime

        # Average metrics over iterations.
        metrics.ms /= num_iters
        metrics.tflops /= num_iters
        metrics.gbps /= num_iters

        results.append(metrics)

    return results


def plot_benchmark(results: list[dict[str, Any]], output_dir: str) -> None:
    """Create a barplot visualizing the TFLOPS of each kernel."""
    # Reprocess into new dataframe with proper graph format.
    data = []
    # Extract measurements for each shape.
    for impl in results:
        shape_str = f"N{impl['N']}_D{impl['D']}_H{impl['H']}_W{impl['W']}_C{impl['C']}_K{impl['K']}"
        # Iterate over keys to find tflops entries.
        for key in impl:
            if "tflops" in key:
                op_name = key.split("_tflops")[0]
                op_tflops = impl[key]
                data.append(
                    {"Shape": shape_str, "kernel": op_name, "TFLOPS": op_tflops}
                )

    # Create a barplot using seaborn.
    df = pd.DataFrame(data)
    plot = plt.figure()
    plt.xticks(rotation=30)
    plt.yscale("log")
    ax = sns.barplot(x="Shape", y="TFLOPS", hue="kernel", data=df)
    ax.tick_params(axis="x", labelsize=3)
    img_fn = os.path.join(output_dir, "conv_ops_benchmark.png")
    plot.savefig(img_fn, dpi=300)
    print(f"Plot saved to {img_fn}")


def collect_kernels_to_profile(kernels: Optional[list[str]]) -> list[ConvOpBase]:
    # Get existing convolution operators.
    conv_ops = [op for op in get_conv_ops() if op.supported]
    if kernels is None:
        return conv_ops
    return [op for op in conv_ops if op.name in kernels]


def print_kernels(kernels: Optional[list[str]]) -> list[ConvOpBase]:
    data = sorted(
        [
            (op.name, "Yes" if op.cuda else "No", "Yes" if op.hip else "No")
            for op in get_conv_ops()
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
    "--plot",
    is_flag=True,
    help="Create a plot of the benchmark measurements.",
)
@click.option(
    "--bench-quantize",
    is_flag=True,
    help="If set, include quantization cost in benchmark.",
)
@click.option(
    "--kernels",
    default=None,
    help="Comma separated list of kernels to benchmark. Defaults to all kernels.",
)
@click.option(
    "--N",
    default=None,
    help="Comma separated list of batch sizes to benchmark.",
)
@click.option(
    "--D",
    default=None,
    help="Comma separated list of depth values to benchmark.",
)
@click.option(
    "--H",
    default=None,
    help="Comma separated list of height values to benchmark.",
)
@click.option(
    "--W",
    default=None,
    help="Comma separated list of width values to benchmark.",
)
@click.option(
    "--C",
    default=None,
    help="Comma separated list of input channel values to benchmark.",
)
@click.option(
    "--K",
    default=None,
    help="Comma separated list of output channel (filter) values to benchmark.",
)
@click.option(
    "--T",
    default=None,
    help="Comma separated list of kernel depth values to benchmark.",
)
@click.option(
    "--R",
    default=None,
    help="Comma separated list of kernel height values to benchmark.",
)
@click.option(
    "--S",
    default=None,
    help="Comma separated list of kernel width values to benchmark.",
)
@click.option(
    "--pad",
    default=None,
    help="Comma separated list of padding values to benchmark.",
)
@click.option(
    "--stride",
    default=None,
    help="Comma separated list of stride values to benchmark.",
)
@click.option(
    "--dilation",
    default=None,
    help="Comma separated list of dilation values to benchmark.",
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
@click.option(
    "--trace",
    is_flag=True,
    help="If set, produce a performance trace of the benchmark.",
)
@click.option(
    "--torch-compile",
    is_flag=True,
    help="If set, torch.compile will be used for aten backed ops.",
)
def invoke_main(
    output_dir: str,
    num_iters: int,
    export_csv: bool,
    plot: bool,
    bench_quantize: bool,
    kernels: Optional[str],
    n: Optional[str],
    d: Optional[str],
    h: Optional[str],
    w: Optional[str],
    c: Optional[str],
    k: Optional[str],
    t: Optional[str],
    r: Optional[str],
    s: Optional[str],
    pad: Optional[str],
    stride: Optional[str],
    dilation: Optional[str],
    no_cuda_graph: bool,
    shapes: Optional[str],
    trace: bool,
    torch_compile: bool,
):
    # If kernel filter is provided, parse it. Else, benchmark all kernels.
    all_kernels = kernels.strip().split(",") if kernels else None
    conv_ops = collect_kernels_to_profile(all_kernels)

    if len(conv_ops) == 0:
        print("No valid kernels to benchmark. Available kernels:")
        print_kernels(all_kernels)
        sys.exit(1)

    if num_iters < 1:
        print("Warning: Number of iterations must be at least 1.")
        num_iters = 1

    # Enumerate shapes to benchmark.
    if shapes:
        if shapes not in shape_registry:
            print(
                f"Shape {shapes} not found in shape registry. Valid shapes: {", ".join(shape_registry.keys())}."
            )
            sys.exit(1)
        conv_shapes = shape_registry[shapes]()
    else:
        # Parse individual dimension parameters
        N = [int(n_val) for n_val in n.strip().split(",")] if n else [1, 4, 8]
        D = [int(d_val) for d_val in d.strip().split(",")] if d else [8, 16]
        H = [int(h_val) for h_val in h.strip().split(",")] if h else [8, 16]
        W = [int(w_val) for w_val in w.strip().split(",")] if w else [8, 16]
        C = [int(c_val) for c_val in c.strip().split(",")] if c else [64, 128]
        K = [int(k_val) for k_val in k.strip().split(",")] if k else [64, 128]
        T = [int(t_val) for t_val in t.strip().split(",")] if t else [3]
        R = [int(r_val) for r_val in r.strip().split(",")] if r else [3]
        S = [int(s_val) for s_val in s.strip().split(",")] if s else [3]
        Pad = [int(p_val) for p_val in pad.strip().split(",")] if pad else [1]
        Stride = (
            [int(st_val) for st_val in stride.strip().split(",")] if stride else [1]
        )
        Dilation = (
            [int(di_val) for di_val in dilation.strip().split(",")] if dilation else [1]
        )

        # Create all combinations
        conv_shapes = list(
            itertools.product(N, D, H, W, C, K, T, R, S, Pad, Stride, Dilation)
        )

    # Iterate over shapes and benchmark.
    benchmark_results = []
    csv = []
    for n, d, h, w, c, k, t, r, s, pad, stride, dilation in conv_shapes:
        conv_measurements = benchmark(
            conv_ops,
            n,
            d,
            h,
            w,
            c,
            k,
            t,
            r,
            s,
            pad,
            stride,
            dilation,
            bench_quantize,
            not no_cuda_graph,
            trace,
            num_iters,
            torch_compile,
        )
        benchmark_results.extend(conv_measurements)
        csv_row = {}
        for metric in conv_measurements:
            csv_row.update(metric.as_dict())
        csv.append(csv_row)

    print(Metrics.header())
    for metric in benchmark_results:
        print(metric)

    mem_bw_roofline_gbps = triton.testing.get_dram_gbps()

    print("")
    print(f"Hardware: {torch.cuda.get_device_name()}")
    print(f"    Memory BW Roofline: {mem_bw_roofline_gbps} GB/s")

    print("")
    print("Benchmark Settings:")
    print(f"    CUDA graph: {not no_cuda_graph}")
    print(f"    Bench quantize: {bench_quantize}")
    print(f"    Torch compile: {torch_compile}")

    if export_csv or plot:
        os.makedirs(output_dir, exist_ok=True)
    if export_csv:
        datetime_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_file = os.path.join(output_dir, f"conv_ops_benchmark_{datetime_str}.csv")
        # Export results to a CSV file.
        df = pd.DataFrame(csv)
        df.to_csv(csv_file, na_rep="NaN", index=False)
        print(f"CSV saved to {csv_file}")
    if plot:
        plot_benchmark(csv, output_dir)


if __name__ == "__main__":
    invoke_main()  # pragma: no cover
