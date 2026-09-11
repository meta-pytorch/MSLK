# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import itertools
import os
import sys
from copy import copy
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional

import click
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import triton  # @manual=//triton:triton
from mslk.bench.common.telemetry import export_benchmark_to_scuba, is_scuba_available
from mslk.bench.common.utils import BenchOptions, common_bench_options, profiler
from mslk.bench.gemm.gemm_ops import ComputeDtype, GemmOpBase, GemmType, get_gemm_ops
from mslk.bench.roofline.capture import EmpiricalRoofline, load_empirical_roofline
from mslk.bench.roofline.regime import CeilingType, classify_regime
from mslk.bench.roofline.targets import load_target_matrix, lookup_target, TargetMatrix
from mslk.utils.device import get_gfx_arch_name
from tabulate import tabulate


# Compute theoretical roofline values in TFLOPS for GPU and dtype combinations.
# Keyed by device name (torch.cuda.get_device_name()) or GPU arch (gcnArchName).
COMPUTE_ROOFLINE_TFLOPS: dict[str, dict[ComputeDtype, float]] = {
    # NVIDIA
    "NVIDIA H100": {
        ComputeDtype.FP8: 1979.0,
        ComputeDtype.BF16: 989.0,
        ComputeDtype.TF32: 494.5,
        ComputeDtype.FP32: 67.0,  # non-tensorcore
    },
    "NVIDIA B200": {
        ComputeDtype.FP4: 9000.0,
        ComputeDtype.FP8: 4500.0,
        ComputeDtype.BF16: 2250.0,
        ComputeDtype.TF32: 1100.0,
        ComputeDtype.FP32: 75.0,  # non-tensorcore
    },
    "NVIDIA GB200": {
        ComputeDtype.FP4: 10000.0,
        ComputeDtype.FP8: 5000.0,
        ComputeDtype.BF16: 2500.0,
        ComputeDtype.TF32: 1250.0,
        ComputeDtype.FP32: 80.0,  # non-tensorcore
    },
    # AMD MI300X (gfx942, CDNA3, 304 CUs @ 2100 MHz, 750W)
    # Source: AMD Instinct MI300X data sheet
    "gfx942": {
        ComputeDtype.FP8: 2614.9,
        ComputeDtype.BF16: 1307.4,
        ComputeDtype.TF32: 653.7,
        ComputeDtype.FP32: 163.4,
    },
    # AMD MI350X (gfx950, CDNA4, 256 CUs @ 2200 MHz, 1000W)
    # Source: AMD Instinct MI350X data sheet
    "gfx950": {
        ComputeDtype.FP4: 9200.0,
        ComputeDtype.FP8: 4600.0,
        ComputeDtype.BF16: 2300.0,
        ComputeDtype.FP32: 144.2,
    },
}


_COMPUTE_DTYPE_TO_ROOFLINE_KEY: dict[ComputeDtype, str] = {
    ComputeDtype.FP4: "fp4",
    ComputeDtype.FP8: "fp8",
    ComputeDtype.BF16: "bf16",
    ComputeDtype.FP32: "fp32",
}

_empirical_roofline: EmpiricalRoofline | None = None
_target_matrix: TargetMatrix | None = None
_stat_iterations: int = 1
_cov_bound: float = 3.0
_cold_l2: bool = False


def set_empirical_roofline(roofline: EmpiricalRoofline) -> None:
    global _empirical_roofline
    _empirical_roofline = roofline


def set_target_matrix(matrix: TargetMatrix) -> None:
    global _target_matrix
    _target_matrix = matrix


def set_bench_options(
    stat_iterations: int = 1,
    cov_bound: float = 3.0,
    cold_l2: bool = False,
) -> None:
    global _stat_iterations, _cov_bound, _cold_l2
    _stat_iterations = stat_iterations
    _cov_bound = cov_bound
    _cold_l2 = cold_l2


def get_compute_roofline_tflops(compute_dtype: ComputeDtype) -> float | None:
    if _empirical_roofline is not None:
        dtype_key = _COMPUTE_DTYPE_TO_ROOFLINE_KEY.get(compute_dtype)
        if dtype_key and dtype_key in _empirical_roofline.mfma_peak_tflops:
            return _empirical_roofline.mfma_peak_tflops[dtype_key]
    # Fall back to datasheet: try device name first (NVIDIA)
    gpu_rooflines = COMPUTE_ROOFLINE_TFLOPS.get(torch.cuda.get_device_name())
    if gpu_rooflines is not None:
        return gpu_rooflines.get(compute_dtype)
    # Fall back to GPU arch (AMD — device name is generic "AMD Radeon Graphics")
    gcn = get_gfx_arch_name()
    if gcn:
        arch = gcn.split(":")[0]
        gpu_rooflines = COMPUTE_ROOFLINE_TFLOPS.get(arch)
        if gpu_rooflines is not None:
            return gpu_rooflines.get(compute_dtype)
    return None


def get_hbm_bw_gbps() -> float:
    if _empirical_roofline is not None and _empirical_roofline.hbm_bw_gbps > 0:
        return _empirical_roofline.hbm_bw_gbps
    return triton.testing.get_dram_gbps()


shape_registry = {}


def register_shapes(name):
    def decorator(op):
        shape_registry[name] = op
        return op

    return decorator


def generate_group_tensor(G, M):
    """
    Generate a tensor with G elements whose integer elements sum to A.

    Args:
        G (int): Number of elements in the tensor.
        M (int): Sum of the elements in the tensor.

    Returns:
        torch.Tensor: A tensor with G elements whose integer elements sum to M.
    """

    # First, we generate a random tensor with G elements
    random_tensor = torch.rand(G)
    # Then, we normalize this tensor so it sums up to 1
    normalized_tensor = random_tensor / random_tensor.sum()
    # Finally, we multiply this tensor by M and round to the nearest integer
    output_tensor = torch.round(normalized_tensor * M).to(torch.int64)
    # Adjust the last element to ensure the sum is exactly M
    output_tensor[-1] += max(0, M - output_tensor.sum())
    return output_tensor.tolist()


def set_amd_env_vars() -> None:
    print("Setting environment variables for AMD GPU performance")
    os.environ["DISABLE_ADDMM_HIP_LT"] = "0"
    os.environ["HIP_FORCE_DEV_KERNARG"] = "1"
    os.environ["PYTORCH_TUNABLEOP_VERBOSE"] = "0"
    os.environ["PYTORCH_TUNABLEOP_ENABLED"] = "1"
    os.environ["PYTORCH_TUNABLEOP_TUNING"] = "1"
    os.environ["PYTORCH_TUNABLEOP_FILENAME"] = "hipblas_tuning_pt_llama.csv"
    os.environ["PYTORCH_TUNABLEOP_MAX_TUNING_DURATION_MS"] = "30"
    os.environ["PYTORCH_TUNABLEOP_MAX_WARMUP_DURATION_MS"] = "30"


@register_shapes("llama3_70b")
def llama3_70b_shapes() -> list[tuple[int, int, int]]:
    shapes = []
    for M in [1, 16, 32, 64, 96, 128]:
        shapes += [
            (M, 1280, 8192),
            (M, 8192, 1024),
            (M, 7168, 8192),
            (M, 8192, 3584),
        ]
    return shapes


@register_shapes("autotune")
def autotune() -> list[tuple[int, int, int]]:
    shapes = []
    for M in [
        1,
        64,
        128,
        256,
        512,
        1024,
        2048,
        4096,
        8192,
        16384,
    ]:
        for N in range(1024, 16384 + 1, 1024):
            for K in range(1024, 16384 + 1, 1024):
                shapes.append((M, N, K))
    return shapes


@register_shapes("llama3_405b")
def llama3_405b_shapes() -> list[tuple[int, int, int]]:
    shapes = []
    for M in [1, 16, 32, 64, 96, 128]:
        shapes += [
            (M, 13312, 6656),
            (M, 13312, 16384),
            (M, 16384, 6656),
            (M, 16384, 16384),
        ]
    return shapes


@register_shapes("llama4")
def llama4_shapes() -> list[tuple[int, int, int]]:
    shapes = []
    for M in [1, 16, 32, 64, 96, 128]:
        shapes += [
            (M, 896, 5120),
            (M, 5120, 640),
            (M, 2048, 5120),
            (M, 5120, 1024),
        ]
    return shapes


@register_shapes("ldm")
def ldm_shapes() -> list[tuple[int, int, int]]:
    return [
        (1536, 3584, 3584),
        (8192, 9728, 3584),
        (8192, 3584, 9728),
        (8192, 3584, 3584),
        (4096, 3584, 3584),
        (768, 3584, 3584),
        (4096, 9728, 3584),
        (4096, 3584, 9728),
        (7200, 3584, 3584),
        (7200, 9728, 3584),
        (7200, 3584, 9728),
        (3600, 3584, 3584),
        (3600, 9728, 3584),
        (3600, 3584, 9728),
        (1536, 4096, 4096),
        (3600, 4096, 4096),
        (3600, 11008, 4096),
        (3600, 4096, 11008),
        (4096, 4096, 4096),
        (4096, 11008, 4096),
        (4096, 4096, 11008),
        (32768, 128, 8192),
        (32768, 8192, 1024),
        (32768, 8192, 3072),
        (32768, 3072, 8192),
        (32768, 1024, 8192),
    ]


class ShapeMode(Enum):
    REGULAR = "regular"  # (M, N, K)
    GROUPED = "grouped"  # G, (M, N, K)
    GROUPED_TOTAL_M = "grouped_total_m"  # G, (TotalM, N, K)
    GROUPED_TOTAL_K = "grouped_total_k"  # G, (M, N, TotalK)


@dataclass
class Metrics:
    op: str
    M: Any = 0
    N: Any = 0
    K: Any = 0
    groups: Optional[int] = None
    shape_mode: ShapeMode = ShapeMode.REGULAR

    sim: float = 0.0
    sqnr: float = 0.0
    ms: float = 0.0
    tflops: float = 0.0
    gbps: float = 0.0
    mem_bw_util: float = 0.0
    compute_util: float = 0.0
    extra_tags: dict[str, str] = field(default_factory=dict)
    extra_metrics: dict[str, float] = field(default_factory=dict)

    # Roofline / regime fields
    regime: str = ""
    ceiling_type: str = ""
    ceiling_value: float = 0.0
    target_pct: float = 0.0
    achieved_pct: float = 0.0
    target_pass: Optional[bool] = None
    cov_pct: float = 0.0
    n_iterations: int = 0
    median_ms: float = 0.0

    @staticmethod
    def header(shape_mode: ShapeMode = ShapeMode.REGULAR) -> str:
        is_grouped = shape_mode in (
            ShapeMode.GROUPED,
            ShapeMode.GROUPED_TOTAL_M,
            ShapeMode.GROUPED_TOTAL_K,
        )
        if shape_mode == ShapeMode.GROUPED_TOTAL_M:
            shape_col = "(TotalM, N, K)"
        elif shape_mode == ShapeMode.GROUPED_TOTAL_K:
            shape_col = "(M, N, TotalK)"
        else:
            shape_col = "(M, N, K)"

        group_col = f"{'G':<6}" if is_grouped else ""
        header = (
            f"{'OpName':<30} {group_col} {shape_col:<25} "
            f"{'Sim':<10} {'SQNR(dB)':<10} {'Ms':<10} {'TFLOPS':<10} "
            f"{'GB/s':<10} {'Mem BW Util %':<14} {'Compute Util %':<14} "
            f"{'Regime':<6} {'Achieved%':<10} {'Target%':<8} {'Pass':<5} {'CoV%':<6}"
        )
        divider = "-" * len(header)
        return f"GEMM Bench\n{divider}\n{header}\n{divider}"

    def __str__(self) -> str:
        is_grouped = self.shape_mode in (
            ShapeMode.GROUPED,
            ShapeMode.GROUPED_TOTAL_M,
            ShapeMode.GROUPED_TOTAL_K,
        )
        if self.shape_mode == ShapeMode.GROUPED_TOTAL_M:
            total_m = sum(self.M) if isinstance(self.M, list) else self.M
            shape = f"({total_m}, {self.N}, {self.K})"
        elif self.shape_mode == ShapeMode.GROUPED_TOTAL_K:
            total_k = sum(self.K) if isinstance(self.K, list) else self.K
            shape = f"({self.M}, {self.N}, {total_k})"
        else:
            shape = f"({self.M}, {self.N}, {self.K})"

        group_col = f"{self.groups:<6}" if is_grouped else ""
        compute_util_str = (
            f"{self.compute_util:<14.2f}" if self.compute_util > 0 else f"{'N/A':<14}"
        )
        sqnr_str = f"{self.sqnr:<10.2f}" if self.sqnr > 0 else f"{'N/A':<10}"
        regime_str = f"{self.regime:<6}" if self.regime else f"{'':6}"
        achieved_str = (
            f"{self.achieved_pct:<10.2f}" if self.achieved_pct > 0 else f"{'':10}"
        )
        target_str = f"{self.target_pct:<8.1f}" if self.target_pct > 0 else f"{'':8}"
        if self.target_pass is None:
            pass_str = f"{'':5}"
        else:
            pass_str = f"{'PASS' if self.target_pass else 'FAIL':<5}"
        cov_str = f"{self.cov_pct:<6.2f}" if self.cov_pct > 0 else f"{'':6}"
        return (
            f"{self.op:<30} {group_col} {shape:<25} "
            f"{self.sim:<10.3f} {sqnr_str} {self.ms:<10.3f} "
            f"{self.tflops:<10.2f} {self.gbps:<10.2f} "
            f"{self.mem_bw_util:<14.2f} {compute_util_str} "
            f"{regime_str} {achieved_str} {target_str} {pass_str} {cov_str}"
        )

    def as_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "M": self.M,
            "N": self.N,
            "K": self.K,
            f"{self.op}_sim": self.sim,
            f"{self.op}_sqnr": self.sqnr,
            f"{self.op}_ms": self.ms,
            f"{self.op}_tflops": self.tflops,
            f"{self.op}_gb/s": self.gbps,
            f"{self.op}_mem_bw_util": self.mem_bw_util,
            f"{self.op}_compute_util": self.compute_util,
            f"{self.op}_regime": self.regime,
            f"{self.op}_ceiling_type": self.ceiling_type,
            f"{self.op}_ceiling_value": self.ceiling_value,
            f"{self.op}_target_pct": self.target_pct,
            f"{self.op}_achieved_pct": self.achieved_pct,
            f"{self.op}_target_pass": self.target_pass,
            f"{self.op}_cov_pct": self.cov_pct,
            f"{self.op}_median_ms": self.median_ms,
        }
        if self.groups is not None:
            result["groups"] = self.groups
        return result


def _enrich_metrics_with_regime(
    metrics: Metrics,
    m: int,
    n: int,
    k: int,
    mem_bw_roofline_gbps: float,
    compute_roofline_tflops: float | None,
    is_grouped: bool = False,
    num_groups: int | None = None,
) -> None:
    """Add regime, ceiling, achieved percentage, and target result to metrics."""
    rc = classify_regime(m, n, k, is_grouped, num_groups)
    metrics.regime = rc.regime.value
    metrics.ceiling_type = rc.ceiling_type.value
    metrics.target_pct = rc.target_low

    target = None
    if _target_matrix is not None:
        target = lookup_target(_target_matrix, metrics.op, m, n, k)
        if target is not None:
            metrics.target_pct = target.target_pct

    if rc.ceiling_type == CeilingType.MFMA_PEAK and compute_roofline_tflops:
        metrics.ceiling_value = compute_roofline_tflops
        metrics.achieved_pct = (
            (metrics.tflops / compute_roofline_tflops) * 100
            if compute_roofline_tflops > 0
            else 0.0
        )
    elif rc.ceiling_type == CeilingType.HBM_BW:
        metrics.ceiling_value = mem_bw_roofline_gbps
        metrics.achieved_pct = (
            (metrics.gbps / mem_bw_roofline_gbps) * 100
            if mem_bw_roofline_gbps > 0
            else 0.0
        )
    elif target is not None and target.ceiling_value:
        metrics.ceiling_value = target.ceiling_value
        metrics.achieved_pct = metrics.tflops / target.ceiling_value * 100

    if metrics.achieved_pct > 0:
        metrics.target_pass = metrics.achieved_pct >= metrics.target_pct


def _get_bench_options(opts: BenchOptions) -> BenchOptions:
    if not _cold_l2:
        return opts
    bench_opts = copy(opts)
    bench_opts.rotating_buffer = True
    return bench_opts


def _record_statistics(metrics: Metrics, stat) -> float:
    metrics.cov_pct = stat.cov_pct
    metrics.median_ms = stat.median_ms
    metrics.n_iterations = stat.n
    if stat.cov_pct > _cov_bound:
        print(
            f"  WARNING: CoV {stat.cov_pct:.2f}% exceeds bound "
            f"{_cov_bound}% for {metrics.op} ({metrics.M},{metrics.N},{metrics.K})"
        )
    return stat.median_ms


def benchmark_grouped(
    gemm_ops: list[GemmOpBase],
    m: list[int],
    n: list[int],
    k: list[int],
    mem_bw_roofline_gbps: float,
    opts: BenchOptions,
    shape_mode: ShapeMode = ShapeMode.GROUPED,
) -> list[Metrics]:
    num_groups = len(m)
    # Create input tensors.
    A = []
    B = []
    for i in range(num_groups):
        A.append(torch.randn(m[i], k[i], device="cuda", dtype=torch.bfloat16))
        B.append(torch.randn(n[i], k[i], device="cuda", dtype=torch.bfloat16))
    # Compute baseline output for correctness checking.
    out_ref = []
    for i in range(num_groups):
        out_ref.append(torch.matmul(A[i], B[i].t()))
    # Keep track of results.
    # Only log all shapes in a group if they are unique.
    log_m = m[0] if len(np.unique(m)) == 1 else m
    log_n = n[0] if len(np.unique(n)) == 1 else n
    log_k = k[0] if len(np.unique(k)) == 1 else k
    results: list[Metrics] = []
    # Benchmark each operator.
    for gemm_op in gemm_ops:
        # Build progress message based on shape mode.
        if shape_mode == ShapeMode.GROUPED_TOTAL_M:
            total_m = sum(m)
            shape_str = f"(G={num_groups}, TotalM={total_m}, N={log_n}, K={log_k})"
        elif shape_mode == ShapeMode.GROUPED_TOTAL_K:
            total_k = sum(k)
            shape_str = f"(G={num_groups}, M={log_m}, N={log_n}, TotalK={total_k})"
        else:
            shape_str = f"(G={num_groups}, M={log_m}, N={log_n}, K={log_k})"
        print(f"Benchmarking {gemm_op.name} with {shape_str}")
        metrics = Metrics(
            op=gemm_op.name,
            M=log_m,
            N=log_n,
            K=log_k,
            groups=num_groups,
            shape_mode=shape_mode,
        )
        # Set fast accum mode if applicable.
        if hasattr(gemm_op, "fast_accum"):
            gemm_op.fast_accum = opts.fast_accum
        if hasattr(gemm_op, "torch_compile"):
            gemm_op.torch_compile = opts.torch_compile

        # Get compute roofline for this op's compute dtype.
        compute_roofline_tflops = get_compute_roofline_tflops(gemm_op.compute_dtype)

        try:
            # Get the quantized tensors for this operator.
            preprocessed_args = gemm_op.preprocess(A, B)
            quantized_vals = gemm_op.quantize(*preprocessed_args)
            # Compute the output given quantized values.
            output = gemm_op.compute(*quantized_vals)
        except Exception as e:
            print(f"GEMM op {gemm_op.name} failed to run due to error: {e}.")
            continue
        # Some kernels may pad output, just take the first m values of each row.
        if isinstance(output, torch.Tensor) and output.ndim == 2:
            # Output is stacked and needs to be split.
            output = torch.split(output, m, dim=0)
        else:
            # Otherwise output may be padded or require unbinding.
            output = [o[: m[i]] for i, o in enumerate(output)]
        # Compare the quantize op output to reference as a sanity check.
        signal_power = 0.0
        noise_power = 0.0
        for i in range(num_groups):
            if m[i] > 0:
                metrics.sim += float(
                    torch.mean(torch.pow(output[i] - out_ref[i], 2)).item()
                )
                signal_power += (out_ref[i].float() ** 2).sum().item()
                noise_power += (
                    ((out_ref[i].float() - output[i].float()) ** 2).sum().item()
                )
        if noise_power > 0:
            metrics.sqnr = float(10 * np.log10(signal_power / noise_power))
        bench_opts = _get_bench_options(opts)

        # Now perform benchmark.
        with profiler(enabled=opts.trace, with_stack=True):
            if _stat_iterations > 1:
                stat = gemm_op.benchmark_statistical(
                    *quantized_vals,
                    opts=bench_opts,
                    n_iterations=_stat_iterations,
                )
                ms_runtime = _record_statistics(metrics, stat)
            else:
                ms_runtime = gemm_op.benchmark(*quantized_vals, opts=bench_opts)

        for i in range(num_groups):
            output_multiplier = 2 if "fuse_scatter_add" in gemm_op.name else 1
            if m[i] > 0:
                tflops = 2 * m[i] * n[i] * k[i] / (ms_runtime / 1e3) / 1e12
                gbps = (
                    (
                        quantized_vals[0][i].numel()
                        * quantized_vals[0][i].element_size()
                        + quantized_vals[1][i].numel()
                        * quantized_vals[1][i].element_size()
                        + output_multiplier
                        * output[i].numel()
                        * output[i].element_size()
                    )
                    / (ms_runtime / 1e3)
                    / 1e9
                )
                metrics.gbps += gbps
                metrics.tflops += tflops
                metrics.mem_bw_util += (gbps / mem_bw_roofline_gbps) * 100
                if compute_roofline_tflops is not None:
                    metrics.compute_util += (tflops / compute_roofline_tflops) * 100
        metrics.ms = ms_runtime

        # Regime classification for grouped shapes (use first group's dimensions)
        rep_m = m[0] if isinstance(m, list) else m
        rep_n = n[0] if isinstance(n, list) else n
        rep_k = k[0] if isinstance(k, list) else k
        _enrich_metrics_with_regime(
            metrics,
            rep_m,
            rep_n,
            rep_k,
            mem_bw_roofline_gbps,
            compute_roofline_tflops,
            is_grouped=True,
            num_groups=num_groups,
        )

        results.append(metrics)

    return results


def benchmark(
    gemm_ops: list[GemmOpBase],
    m: int,
    n: int,
    k: int,
    mem_bw_roofline_gbps: float,
    opts: BenchOptions,
    shape_mode: ShapeMode = ShapeMode.REGULAR,
) -> list[Metrics]:
    # Create input tensors.
    A = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    B = torch.randn(n, k, device="cuda", dtype=torch.bfloat16)

    # Compute baseline output for correctness checking.
    out_ref = torch.matmul(A, torch.transpose(B, -2, -1))
    # Keep track of results.
    results: list[Metrics] = []
    # Benchmark each operator.
    for gemm_op in gemm_ops:
        shape_str = f"(M={m}, N={n}, K={k})"
        print(f"Benchmarking {gemm_op.name} with {shape_str}")
        metrics = Metrics(op=gemm_op.name, M=m, N=n, K=k, shape_mode=shape_mode)
        # Set fast accum mode if applicable.
        if hasattr(gemm_op, "fast_accum"):
            gemm_op.fast_accum = opts.fast_accum
        if hasattr(gemm_op, "torch_compile"):
            gemm_op.torch_compile = opts.torch_compile

        # Get compute roofline for this op's compute dtype.
        compute_roofline_tflops = get_compute_roofline_tflops(gemm_op.compute_dtype)

        try:
            # Preprocess data if needed.
            preprocessed_args = gemm_op.preprocess(A, B)
            # Get the quantized tensors for this operator.
            quantized_vals = gemm_op.quantize(*preprocessed_args)
            # Compute the output given quantized values.
            output = gemm_op.compute(*quantized_vals)
        except Exception as e:
            print(f"GEMM op {gemm_op.name} failed to run due to error: {e}.")
            continue
        # Compare the quantize op output to reference as a sanity check.
        # TODO(shikaili): This calculation is incorrect for scatter add fusion.
        metrics.sim = torch.mean(torch.pow(output - out_ref, 2)).item()
        signal_power = (out_ref.float() ** 2).sum()
        noise_power = ((out_ref.float() - output.float()) ** 2).sum()
        if noise_power > 0:
            metrics.sqnr = (10 * torch.log10(signal_power / noise_power)).item()

        bench_opts = _get_bench_options(opts)

        # Now perform benchmark.
        with profiler(enabled=opts.trace, with_stack=True):
            if _stat_iterations > 1:
                stat = gemm_op.benchmark_statistical(
                    *quantized_vals,
                    opts=bench_opts,
                    n_iterations=_stat_iterations,
                )
                ms_runtime = _record_statistics(metrics, stat)
            else:
                ms_runtime = gemm_op.benchmark(*quantized_vals, opts=bench_opts)

        metrics.ms = ms_runtime
        metrics.tflops = 2 * m * n * k / (ms_runtime / 1e3) / 1e12
        metrics.gbps = (
            (
                quantized_vals[0].numel() * quantized_vals[0].element_size()
                + quantized_vals[1].numel() * quantized_vals[1].element_size()
                + output.numel() * output.element_size()
            )
            / (ms_runtime / 1e3)
            / 1e9
        )
        metrics.mem_bw_util = (metrics.gbps / mem_bw_roofline_gbps) * 100
        if compute_roofline_tflops is not None:
            metrics.compute_util = (metrics.tflops / compute_roofline_tflops) * 100

        _enrich_metrics_with_regime(
            metrics, m, n, k, mem_bw_roofline_gbps, compute_roofline_tflops
        )

        results.append(metrics)

    return results


def plot_benchmark(results: list[Metrics], output_dir: str) -> None:
    """Create a barplot visualizing the TFLOPS of each kernel."""
    # Reprocess into new dataframe with proper graph format.
    data = []
    # Extract measurements for each shape.
    for metric in results:
        mnk = f"{metric.M}, {metric.N}, {metric.K}"
        data.append({"MNK": mnk, "kernel": metric.op, "TFLOPS": metric.tflops})

    # Create a barplot using seaborn.
    df = pd.DataFrame(data)
    plot = plt.figure()
    plt.xticks(rotation=30)
    plt.yscale("log")
    ax = sns.barplot(x="MNK", y="TFLOPS", hue="kernel", data=df)
    ax.tick_params(axis="x", labelsize=3)
    img_fn = os.path.join(output_dir, "gemm_ops_benchmark.png")
    plot.savefig(img_fn, dpi=300)
    print(f"Plot saved to {img_fn}")


def collect_kernels_to_profile(
    kernels: Optional[list[str]], is_grouped: bool
) -> list[GemmOpBase]:
    gemm_type = GemmType.GROUPED if is_grouped else GemmType.REGULAR
    gemm_ops = [
        op
        for op in get_gemm_ops()
        if op.supported and gemm_type in op.supported_gemm_types
    ]
    if kernels is None:
        return gemm_ops
    return [op for op in gemm_ops if op.name in kernels]


def print_kernels(kernels: Optional[list[str]]) -> list[GemmOpBase]:
    data = sorted(
        (
            op.name,
            ",".join(accelerator.name for accelerator in op.supported_accelerators),
        )
        for op in get_gemm_ops()
    )
    print(tabulate(data, headers=["Name", "Accelerators"], tablefmt="orgtbl"))


@click.command()
@common_bench_options(shape_registry)
@click.option(
    "--export-scuba",
    is_flag=True,
    hidden=True,
    help="Export results to a Scuba table (internal only).",
)
@click.option(
    "--plot",
    is_flag=True,
    help="Create a plot of the benchmark measurements.",
)
@click.option(
    "--enable-amd-env-vars",
    is_flag=True,
    help="Enable a set of environment variables for AMD GPU performance",
)
@click.option(
    "--M",
    default=None,
    help="Comma separated list of M values to benchmark.",
)
@click.option(
    "--N",
    default=None,
    help="Comma separated list of N values to benchmark",
)
@click.option(
    "--K",
    default=None,
    help="Comma separated list of K values to benchmark.",
)
@click.option(
    "--pair-NK",
    is_flag=True,
    help=(
        "If set, instead of benchmarking cartesian product of N * K, "
        "benchmark consecutive NK pairs together."
    ),
)
@click.option(
    "--grouped",
    is_flag=True,
    help="If set, do grouped gemm. In this mode, M, N, and K are interpreted "
    "as the size of groups. The length of each must be the same.",
)
@click.option(
    "--groups",
    default=None,
    help=(
        "If set with grouped mode, repeat MNK shapes this many times. "
        "Comma separated list of groups to benchmark"
    ),
)
@click.option(
    "--total-K",
    default=None,
    help="If set, adjusts the K values to sum to this number. "
    "This can help simulate real grouped workloads in backward wgrad. "
    "Comma separated list of total-K values to benchmark.",
)
@click.option(
    "--total-M",
    default=None,
    help="If set, adjusts the M values to sum to this number. "
    "This can help simulate real grouped workloads."
    "Comma separated list of total-M values to benchmark.",
)
@click.option(
    "--disable-fast-accum",
    is_flag=True,
    help="If set, disable fast accumulation for FP8 implementations.",
)
@click.option(
    "--torch-compile",
    is_flag=True,
    help="If set, torch.compile will be used for scaled_mm backed ops.",
)
@click.option(
    "--empirical-roofline",
    default=None,
    type=click.Path(exists=True),
    help="Path to empirical roofline JSON (replaces datasheet peaks).",
)
@click.option(
    "--cold-l2/--no-cold-l2",
    default=False,
    help="Force a rotating buffer for cold-L2 measurements.",
)
@click.option(
    "--target-matrix",
    "target_matrix_path",
    default=None,
    type=click.Path(exists=True),
    help="Path to target matrix YAML for pass/fail evaluation.",
)
@click.option(
    "--stat-iterations",
    default=1,
    type=int,
    help="Number of benchmark iterations for CoV. Use >=5 for reliable stats.",
)
@click.option(
    "--cov-bound",
    default=3.0,
    type=float,
    help="Maximum acceptable CoV %%. Warn if exceeded.",
)
def invoke_main(
    output_dir: str,
    export_csv: bool,
    export_scuba: bool,
    plot: bool,
    enable_amd_env_vars: bool,
    kernels: Optional[str],
    m: Optional[str],
    n: Optional[str],
    k: Optional[str],
    pair_nk: bool,
    grouped: bool,
    groups: Optional[str],
    total_k: Optional[str],
    total_m: Optional[str],
    cuda_graph: bool,
    rotating_buffer: bool,
    shapes: Optional[str],
    trace: bool,
    disable_fast_accum: bool,
    torch_compile: bool,
    rep_ms: int,
    empirical_roofline: Optional[str],
    cold_l2: bool,
    target_matrix_path: Optional[str],
    stat_iterations: int,
    cov_bound: float,
):
    # Load empirical roofline if provided
    if empirical_roofline:
        roofline = load_empirical_roofline(empirical_roofline)
        set_empirical_roofline(roofline)
        print(f"Using empirical roofline from {empirical_roofline}")
        print(f"  MFMA peaks: {roofline.mfma_peak_tflops}")
        print(f"  HBM BW: {roofline.hbm_bw_gbps:.1f} GB/s")

    if target_matrix_path:
        matrix = load_target_matrix(target_matrix_path)
        set_target_matrix(matrix)
        target_count = len(matrix.targets)
        print(
            f"Loaded target matrix from {target_matrix_path} ({target_count} targets)"
        )

    set_bench_options(
        stat_iterations=stat_iterations,
        cov_bound=cov_bound,
        cold_l2=cold_l2,
    )

    if enable_amd_env_vars:
        set_amd_env_vars()

    # Validate that total_m and total_k are mutually exclusive
    if total_m is not None and total_k is not None:
        raise ValueError(
            "total_m and total_k cannot be specified at the same time. "
            "Please provide only one of them."
        )

    if groups:
        grouped = True

    # If kernel filter is provided, parse it. Else, benchmark all kernels.
    all_kernels = kernels.strip().split(",") if kernels else None
    gemm_ops = collect_kernels_to_profile(all_kernels, grouped)

    if len(gemm_ops) == 0:
        print("No valid kernels to benchmark. Available kernels:")
        print_kernels(all_kernels)
        sys.exit(1)

    # Enumerate shapes to benchmark.
    if grouped and not groups:
        # In grouped mode, M, N, and K represent the groups of a single gemm.
        assert m is not None and n is not None and k is not None
        M = [int(m_val) for m_val in m.strip().split(",")]
        N = [int(n_val) for n_val in n.strip().split(",")]
        K = [int(k_val) for k_val in k.strip().split(",")]
        assert len(M) == len(N) == len(K), (
            "M, N, and K must be the same length in grouped mode."
        )

        # Note this is a single grouped gemm.
        MNK = [[M, N, K]]
    else:
        if shapes:
            if shapes not in shape_registry:
                print(
                    f"Shape {shapes} not found in shape registry. "
                    f"Valid shapes: {', '.join(shape_registry.keys())}."
                )
                sys.exit(1)
            MNK = shape_registry[shapes]()
        else:
            if m is None:
                M = [1, 4, 8, 16, 32, 64, 128, 2048, 4096, 8192, 16384]
            else:
                M = [int(m_val) for m_val in m.strip().split(",")]
            if n is None:
                N = [1280, 2304, 7168, 8192, 16384]
            else:
                N = [int(n_val) for n_val in n.strip().split(",")]
            if k is None:
                K = [1024, 3584, 8192, 16384]
            else:
                K = [int(k_val) for k_val in k.strip().split(",")]
            # List all shapes for simplicity.
            if pair_nk:
                if len(N) != len(K):
                    raise Exception("N and K must be the same length in pair_NK mode.")
                NK = zip(N, K)
                MNK = [(M, N, K) for (M, (N, K)) in itertools.product(M, NK)]
            else:
                MNK = list(itertools.product(M, N, K))
    # When groups is provided transform shapes into grouped format.
    if groups:
        groups_list = [int(g) for g in groups.strip().split(",")]
        if total_m:
            total_m_list = [int(tm) for tm in total_m.strip().split(",")]
            MNK = [
                [
                    generate_group_tensor(g, tm),
                    [n] * g,
                    [k] * g,
                ]
                for g in groups_list
                for tm in total_m_list
                for _, n, k in MNK
            ]
            shape_mode = ShapeMode.GROUPED_TOTAL_M
        elif total_k:
            total_k_list = [int(tk) for tk in total_k.strip().split(",")]
            MNK = [
                [
                    [m] * g,
                    [n] * g,
                    generate_group_tensor(g, tk),
                ]
                for g in groups_list
                for tk in total_k_list
                for m, n, _ in MNK
            ]
            shape_mode = ShapeMode.GROUPED_TOTAL_K
        else:
            MNK = [[[m] * g, [n] * g, [k] * g] for g in groups_list for m, n, k in MNK]
            shape_mode = ShapeMode.GROUPED
    elif grouped:
        shape_mode = ShapeMode.GROUPED
    else:
        shape_mode = ShapeMode.REGULAR

    # Iterate over shapes and benchmark.
    mem_bw_gbps = get_hbm_bw_gbps()
    benchmark_results: list[Metrics] = []
    csv: list[dict[str, Any]] = []
    benchmark_func = benchmark_grouped if grouped else benchmark

    opts = BenchOptions(
        cuda_graph=cuda_graph,
        rotating_buffer=rotating_buffer,
        rep_ms=rep_ms,
        trace=trace,
        fast_accum=not disable_fast_accum,
        torch_compile=torch_compile,
    )

    for m, n, k in MNK:
        shape_measurements = benchmark_func(
            gemm_ops,
            m,  # pyre-ignore[6]: Incompatible parameter type [6]
            n,  # pyre-ignore[6]: Incompatible parameter type [6]
            k,  # pyre-ignore[6]: Incompatible parameter type [6]
            mem_bw_gbps,
            opts,
            shape_mode,
        )
        benchmark_results.extend(shape_measurements)
        csv_row: dict[str, Any] = {}
        for metric in shape_measurements:
            csv_row.update(metric.as_dict())
        csv.append(csv_row)

    print("")
    print(Metrics.header(shape_mode))
    for metric in benchmark_results:
        print(metric)

    print("")
    print(f"Hardware: {torch.cuda.get_device_name()}")
    print(f"    Memory BW: {mem_bw_gbps:.2f} GB/s")
    roofline_src = "empirical" if _empirical_roofline is not None else "datasheet"
    print(f"    Roofline source: {roofline_src}")

    print("")
    print("Benchmark Settings:")
    print(f"    CUDA graph: {cuda_graph}")
    print(f"    Buffer rotation: {rotating_buffer}")
    print(f"    Fast accumulation: {not disable_fast_accum}")
    print(f"    Torch compile: {torch_compile}")
    if stat_iterations > 1:
        print(
            f"    Statistical iterations: {stat_iterations} (CoV bound: {cov_bound}%)"
        )
    if cold_l2:
        print("    Cold L2: enabled for R3 shapes")

    if export_csv or plot:
        os.makedirs(output_dir, exist_ok=True)
    if export_csv:
        datetime_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_file = os.path.join(output_dir, f"gemm_ops_benchmark_{datetime_str}.csv")
        # Export results to a CSV file.
        df = pd.DataFrame(csv)
        df.to_csv(csv_file, na_rep="NaN", index=False)
        print(f"CSV saved to {csv_file}")
    if export_scuba:
        if not is_scuba_available():
            print(
                "Warning: --export-scuba requires FB-internal build. "
                "Skipping Scuba export."
            )
        else:
            # Determine kernel category based on benchmark mode
            kernel_category = "Grouped GEMM Kernels" if grouped else "GEMM Kernels"
            # Set shape_mode in extra_tags for each sample
            for metric in benchmark_results:
                metric.extra_tags["shape_mode"] = metric.shape_mode.value
            export_benchmark_to_scuba(
                samples=benchmark_results,
                bench_type="gemm",
                kernel_category=kernel_category,
                mem_bw_roofline_gbps=mem_bw_gbps,
                cuda_graph_enabled=cuda_graph,
                fast_accum_enabled=not disable_fast_accum,
                torch_compile_enabled=torch_compile,
            )
    if plot:
        plot_benchmark(benchmark_results, output_dir)


if __name__ == "__main__":
    invoke_main()  # pragma: no cover
