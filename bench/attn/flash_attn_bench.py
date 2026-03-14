# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Flash Attention (FA4) benchmark.

Benchmarks flash_attn_func across sequence lengths and head configurations.
Reports wall-clock time and effective TFLOPS.
"""

import os
import sys
from dataclasses import dataclass
from datetime import datetime
from typing import Callable, Optional

import click
import torch
from mslk.bench.common.utils import BenchOptions, do_bench

type ShapeFunction = Callable[[], list[tuple[int, int, int, int, int]]]

shape_registry: dict[str, ShapeFunction] = {}


def register_shapes(name: str) -> Callable[[ShapeFunction], ShapeFunction]:
    def decorator(shape_function: ShapeFunction) -> ShapeFunction:
        shape_registry[name] = shape_function
        return shape_function

    return decorator


@register_shapes("llama4")
def llama4_shapes() -> list[tuple[int, int, int, int, int]]:
    """Llama 4 attention shapes: (B, N, H, H_kv, D)."""
    shapes = []
    for N in [1024, 2048, 4096, 8192, 16384, 32768, 65536]:
        shapes.append((2, N, 32, 8, 128))
    return shapes


@register_shapes("ldm")
def ldm_shapes() -> list[tuple[int, int, int, int, int]]:
    """Latent diffusion model attention shapes: (B, N, H, H_kv, D)."""
    shapes = []
    for N in [4096, 8192, 16384, 32768]:
        shapes.append((1, N, 24, 24, 128))
    return shapes


def compute_flops(B: int, H: int, N: int, D: int, window_size: int = 0) -> int:
    """Compute FLOPs for dense attention forward pass.

    FLOPs = 4 * B * H * N * N_eff * D
    where N_eff is the effective KV length (window_size if set, else N).
    The factor of 4 accounts for QK^T (2*B*H*N*N_eff*D) and PV (2*B*H*N*N_eff*D).
    """
    N_eff = min(N, window_size) if window_size > 0 else N
    return 4 * B * H * N * N_eff * D


@dataclass
class Metrics:
    B: int = 0
    N: int = 0
    H: int = 0
    H_kv: int = 0
    D: int = 0
    causal: bool = True
    window_size: int = 0
    ms: float = 0.0
    tflops: float = 0.0

    @staticmethod
    def header() -> str:
        header = (
            f"{'B':>4} {'N':>8} {'H':>4} {'H_kv':>5} {'D':>4} "
            f"{'Causal':>6} {'Window':>6} | {'Ms':>10} {'TFLOPS':>10}"
        )
        divider = "-" * len(header)
        return f"Flash Attention Bench\n{divider}\n{header}\n{divider}"

    def __str__(self) -> str:
        window_str = str(self.window_size) if self.window_size > 0 else "-"
        return (
            f"{self.B:>4} {self.N:>8} {self.H:>4} {self.H_kv:>5} {self.D:>4} "
            f"{'Y' if self.causal else 'N':>6} {window_str:>6} | "
            f"{self.ms:>10.3f} {self.tflops:>10.2f}"
        )

    def as_dict(self) -> dict[str, float]:
        return {
            "B": self.B,
            "N": self.N,
            "H": self.H,
            "H_kv": self.H_kv,
            "D": self.D,
            "causal": int(self.causal),
            "window_size": self.window_size,
            "ms": self.ms,
            "tflops": self.tflops,
        }


def benchmark(
    B: int,
    N: int,
    H: int,
    H_kv: int,
    D: int,
    causal: bool,
    window_size: int,
    opts: BenchOptions,
) -> Metrics:
    """Benchmark flash_attn_func for one configuration."""
    from mslk.attention.flash_attn import flash_attn_func

    dtype = torch.bfloat16
    Q = torch.randn(B, N, H, D, device="cuda", dtype=dtype)
    K = torch.randn(B, N, H_kv, D, device="cuda", dtype=dtype)
    V = torch.randn(B, N, H_kv, D, device="cuda", dtype=dtype)

    ws = (window_size, 0) if window_size > 0 else None

    def fn():
        flash_attn_func(Q, K, V, causal=causal, window_size=ws)

    ms = do_bench(fn, (), opts)

    flops = compute_flops(B, H, N, D, window_size)
    tflops = flops / (ms / 1e3) / 1e12

    return Metrics(
        B=B,
        N=N,
        H=H,
        H_kv=H_kv,
        D=D,
        causal=causal,
        window_size=window_size,
        ms=ms,
        tflops=tflops,
    )


@click.command()
@click.option(
    "--N",
    "n_str",
    default="1024,2048,4096,8192,16384,32768,65536",
    help="Comma-separated sequence lengths.",
)
@click.option("--B", "b", default=2, type=int, help="Batch size.")
@click.option("--H", "h", default=32, type=int, help="Number of query heads.")
@click.option("--H-kv", "h_kv", default=8, type=int, help="Number of KV heads.")
@click.option("--D", "d", default=128, type=int, help="Head dimension.")
@click.option("--causal/--no-causal", default=True, help="Causal masking.")
@click.option(
    "--window-size", default=0, type=int, help="Sliding window size (0 = no window)."
)
@click.option(
    "--num-iters", default=1, type=int, help="Number of benchmark iterations."
)
@click.option("--no-cuda-graph", is_flag=True, help="Disable CUDA graphs.")
@click.option("--export-csv", is_flag=True, help="Export results to CSV.")
@click.option("--output-dir", default="/tmp", help="Directory for output files.")
@click.option(
    "--shapes",
    default=None,
    help=f"Named shape preset: {', '.join(shape_registry.keys())}.",
)
@click.option(
    "--rep",
    default=200,
    type=int,
    help="Repetition time in ms for triton.testing.do_bench.",
)
def invoke_main(
    n_str: str,
    b: int,
    h: int,
    h_kv: int,
    d: int,
    causal: bool,
    window_size: int,
    num_iters: int,
    no_cuda_graph: bool,
    export_csv: bool,
    output_dir: str,
    shapes: Optional[str],
    rep: int,
) -> None:
    if num_iters < 1:
        print("Warning: Number of iterations must be at least 1.")
        num_iters = 1

    opts = BenchOptions(
        num_iters=num_iters,
        cuda_graph=not no_cuda_graph,
        rep_ms=rep,
    )

    # Build shape list.
    if shapes:
        if shapes not in shape_registry:
            valid_shapes = ", ".join(shape_registry.keys())
            print(f"Shape '{shapes}' not found. Valid: {valid_shapes}.")
            sys.exit(1)
        shape_list = shape_registry[shapes]()
    else:
        N_vals = [int(x) for x in n_str.strip().split(",")]
        shape_list = [(b, N, h, h_kv, d) for N in N_vals]

    # Run benchmarks.
    results: list[Metrics] = []
    csv_rows: list[dict] = []

    for B, N, H, H_kv, D in shape_list:
        print(f"Benchmarking B={B}, N={N}, H={H}, H_kv={H_kv}, D={D}")
        for _ in range(num_iters):
            m = benchmark(B, N, H, H_kv, D, causal, window_size, opts)
            results.append(m)
            csv_rows.append(m.as_dict())

    # Print results.
    print("")
    print(Metrics.header())
    for m in results:
        print(m)

    print("")
    print(f"Hardware: {torch.cuda.get_device_name()}")
    print("")
    print("Benchmark Settings:")
    print(f"    CUDA graph: {not no_cuda_graph}")
    print(f"    Causal: {causal}")
    if window_size > 0:
        print(f"    Window size: {window_size}")

    if export_csv:
        os.makedirs(output_dir, exist_ok=True)
        datetime_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_file = os.path.join(output_dir, f"flash_attn_bench_{datetime_str}.csv")
        import pandas as pd

        pd.DataFrame(csv_rows).to_csv(csv_file, index=False)
        print(f"CSV saved to {csv_file}")


if __name__ == "__main__":
    invoke_main()
