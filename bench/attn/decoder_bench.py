# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Paged-attention decode benchmark: FlyDSL vs CK vs Triton.

Benchmarks the fmha decode forward ops (see ``decoder_ops.py``) across decode shapes
and reports latency, achieved HBM bandwidth, and memory-bandwidth utilization. One
row per op per shape; compare rows to read relative speedups.

``--dtype {bf16,f16}`` runs the dense ops; ``--dtype fp8`` quantizes the KV cache and
runs the fp8 ops (FlyDSLFp8, TritonFp8) instead.

Usage:
    python bench/attn/decoder_bench.py --shapes decode_llm
    python bench/attn/decoder_bench.py --shapes sweep_kv --dtype fp8 --export-csv
    python bench/attn/decoder_bench.py --kernels FlyDSLDecode,CKDecode,TritonSplitK
"""

import os
import sys
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Optional

import click
import torch
import triton  # @manual=//triton:triton
from mslk.bench.attn.decoder_ops import DecodeOpBase, get_decode_ops
from mslk.bench.common.utils import BenchOptions, common_bench_options, profiler

ShapeList = list[tuple[int, int, int, int, int]]

shape_registry: dict[str, Callable[[], ShapeList]] = {}


def register_shapes(name: str) -> Callable[[Callable[[], ShapeList]], Callable[[], ShapeList]]:
    def decorator(fn: Callable[[], ShapeList]) -> Callable[[], ShapeList]:
        shape_registry[name] = fn
        return fn

    return decorator


@register_shapes("decode_llm")
def _shapes_decode_llm() -> ShapeList:
    """Common LLM decode shapes with various GQA ratios: (B, Hq, Hkv, kv_len, D)."""
    return [
        (1, 32, 8, 512, 128),
        (1, 32, 8, 2048, 128),
        (1, 32, 8, 4096, 128),
        (8, 32, 8, 2048, 128),
        (16, 32, 8, 2048, 128),
        (1, 64, 8, 2048, 128),
        (1, 64, 16, 2048, 128),
        (1, 128, 16, 2048, 128),
        (1, 32, 4, 2048, 256),
        (8, 32, 4, 2048, 256),
    ]


@register_shapes("sweep_kv")
def _shapes_sweep_kv() -> ShapeList:
    """Sweep KV sequence length."""
    return [(1, 32, 8, kv_len, 128) for kv_len in [128, 256, 512, 1024, 2048, 4096, 8192, 16384]]


@register_shapes("sweep_batch")
def _shapes_sweep_batch() -> ShapeList:
    """Sweep batch size."""
    return [(B, 32, 8, 2048, 128) for B in [1, 2, 4, 8, 16, 32, 64]]


def _bytes_read_write(
    B: int, Hq: int, Hkv: int, kv_seqlen: int, D: int, dtype: str
) -> int:
    """Approximate HBM traffic for one decode step (bytes): Q + K + V + output.

    Query/output are 16-bit; the KV cache is 1 byte under fp8, else 16-bit.
    """
    io_elem = 2  # bf16/f16 query + output
    kv_elem = 1 if dtype == "fp8" else 2
    q_read = B * Hq * D * io_elem
    kv_read = B * kv_seqlen * Hkv * D * kv_elem * 2  # K and V
    out_write = B * Hq * D * io_elem
    return q_read + kv_read + out_write


@dataclass
class Metrics:
    op: str
    B: int
    Hq: int
    Hkv: int
    kv_seqlen: int
    D: int
    dtype: str
    ms: float = 0.0
    gbps: float = 0.0
    mem_bw_util: float = 0.0

    @staticmethod
    def header() -> str:
        header = (
            f"{'OpName':<16} {'B':>4} {'Hq':>4} {'Hkv':>4} {'KV':>6} {'D':>4} "
            f"{'dtype':>6} | {'Ms':>10} {'GB/s':>10} {'Mem BW Util %':>14}"
        )
        divider = "-" * len(header)
        return f"Decoder Attention Bench\n{divider}\n{header}\n{divider}"

    def __str__(self) -> str:
        return (
            f"{self.op:<16} {self.B:>4} {self.Hq:>4} {self.Hkv:>4} {self.kv_seqlen:>6} "
            f"{self.D:>4} {self.dtype:>6} | {self.ms:>10.3f} {self.gbps:>10.2f} "
            f"{self.mem_bw_util:>14.2f}"
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "op": self.op,
            "B": self.B,
            "Hq": self.Hq,
            "Hkv": self.Hkv,
            "kv_seqlen": self.kv_seqlen,
            "D": self.D,
            "dtype": self.dtype,
            "ms": self.ms,
            "gbps": self.gbps,
            "mem_bw_util": self.mem_bw_util,
        }


def benchmark(
    ops: list[DecodeOpBase],
    B: int,
    Hq: int,
    Hkv: int,
    kv_seqlen: int,
    D: int,
    dtype: torch.dtype,
    dtype_str: str,
    mem_bw_roofline_gbps: float,
    opts: BenchOptions,
) -> list[Metrics]:
    """Benchmark every op for one decode shape."""
    nbytes = _bytes_read_write(B, Hq, Hkv, kv_seqlen, D, dtype_str)
    results: list[Metrics] = []
    for op in ops:
        shape_str = f"(B={B}, Hq={Hq}, Hkv={Hkv}, KV={kv_seqlen}, D={D})"
        print(f"Benchmarking {op.name} with {shape_str}")
        try:
            args = op.setup(B, Hq, Hkv, kv_seqlen, D, dtype)
            op.compute(*args)  # warmup / sanity
        except Exception as e:
            print(f"Decode op {op.name} failed to run due to error: {e}.")
            continue
        with profiler(enabled=opts.trace, with_stack=True):
            ms = op.benchmark(*args, opts=opts)
        gbps = nbytes / (ms / 1e3) / 1e9
        results.append(
            Metrics(
                op=op.name,
                B=B,
                Hq=Hq,
                Hkv=Hkv,
                kv_seqlen=kv_seqlen,
                D=D,
                dtype=dtype_str,
                ms=ms,
                gbps=gbps,
                mem_bw_util=(gbps / mem_bw_roofline_gbps) * 100,
            )
        )
    return results


def collect_ops(kernels: Optional[list[str]], dtype: str) -> list[DecodeOpBase]:
    ops = [
        op
        for op in get_decode_ops()
        if op.supported and dtype in op.supported_dtypes
    ]
    if kernels is None:
        return ops
    return [op for op in ops if op.name in kernels]


@click.command()
@common_bench_options(shape_registry)
@click.option(
    "--dtype",
    default="bf16",
    type=click.Choice(["bf16", "f16", "fp8"]),
    help="KV-cache dtype. fp8 quantizes the KV cache (bf16 query) and runs the fp8 ops.",
)
def invoke_main(
    output_dir: str,
    export_csv: bool,
    kernels: Optional[str],
    cuda_graph: bool,
    rotating_buffer: bool,
    shapes: Optional[str],
    trace: bool,
    rep_ms: int,
    dtype: str,
) -> None:
    # fp8 uses a bf16 query with a quantized KV cache (see the fp8 ops).
    torch_dtype = {
        "bf16": torch.bfloat16,
        "f16": torch.float16,
        "fp8": torch.bfloat16,
    }[dtype]

    kernel_filter = kernels.strip().split(",") if kernels else None
    ops = collect_ops(kernel_filter, dtype)
    if not ops:
        available = ", ".join(op.name for op in get_decode_ops())
        print(f"No matching supported ops. Available: {available}.")
        sys.exit(1)

    if shapes:
        if shapes not in shape_registry:
            print(
                f"Shape '{shapes}' not found. Valid: {', '.join(shape_registry.keys())}."
            )
            sys.exit(1)
        shape_list = shape_registry[shapes]()
    else:
        shape_list = shape_registry["decode_llm"]()

    opts = BenchOptions(
        cuda_graph=cuda_graph,
        rotating_buffer=rotating_buffer,
        rep_ms=rep_ms,
        trace=trace,
    )

    mem_bw_gbps = triton.testing.get_dram_gbps()
    results: list[Metrics] = []
    for B, Hq, Hkv, kv_seqlen, D in shape_list:
        results.extend(
            benchmark(
                ops, B, Hq, Hkv, kv_seqlen, D, torch_dtype, dtype, mem_bw_gbps, opts
            )
        )

    print("")
    print(Metrics.header())
    for m in results:
        print(m)

    print("")
    print(f"Hardware: {torch.cuda.get_device_name()}")
    print(f"    Memory BW: {mem_bw_gbps:.2f} GB/s")
    print("")
    print("Benchmark Settings:")
    print(f"    CUDA graph: {cuda_graph}")
    print(f"    Buffer rotation: {rotating_buffer}")
    print(f"    dtype: {dtype}")

    if export_csv:
        import pandas as pd

        os.makedirs(output_dir, exist_ok=True)
        datetime_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_file = os.path.join(output_dir, f"decoder_bench_{datetime_str}.csv")
        pd.DataFrame([m.as_dict() for m in results]).to_csv(csv_file, index=False)
        print(f"CSV saved to {csv_file}")


if __name__ == "__main__":
    invoke_main()  # pragma: no cover
