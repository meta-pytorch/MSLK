#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Standalone FlyDSL preshuffle GEMM benchmark for WP-1a roofline profiling.

Bypasses the full gemm_ops registry (which requires the C++ .so) and directly
benchmarks the FlyDSL preshuffle kernel against empirical roofline ceilings.
"""

import os
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import click
import numpy as np
import pandas as pd
import torch
import triton
from mslk.bench.roofline.capture import EmpiricalRoofline, load_empirical_roofline
from mslk.bench.roofline.regime import CeilingType, classify_regime
from mslk.bench.roofline.targets import (
    load_target_matrix,
    lookup_target,
    save_target_matrix,
    TargetMatrix,
)
from mslk.utils.device import supports_float8_fnuz


def _quantize_fp8_row(x: torch.Tensor):
    fp8_dtype = torch.float8_e4m3fnuz if supports_float8_fnuz() else torch.float8_e4m3fn
    xmax = x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12)
    scale = xmax / torch.finfo(fp8_dtype).max
    xq = (x / scale).to(fp8_dtype)
    return xq, scale.squeeze(-1)


@dataclass
class BenchResult:
    kernel: str
    M: int
    N: int
    K: int
    ms: float
    median_ms: float
    tflops: float
    gbps: float
    regime: str
    ceiling_type: str
    ceiling_value: float
    achieved_pct: float
    target_pct: float
    target_pass: Optional[bool]
    cov_pct: float
    n_iters: int


def bench_flydsl_preshuffle(
    M: int,
    N: int,
    K: int,
    roofline: Optional[EmpiricalRoofline],
    target_matrix: Optional[TargetMatrix],
    n_iterations: int = 5,
    rep_ms: int = 200,
    device: int = 0,
) -> BenchResult:
    """Benchmark FlyDSL preshuffle GEMM for a single shape."""
    # Keep this optional kernel import out of report-only command startup.
    from mslk.gemm.flydsl.preshuffle_gemm import (
        flydsl_preshuffle,
        flydsl_preshuffle_gemm,
    )

    torch.cuda.set_device(device)

    A = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    B = torch.randn(N, K, device="cuda", dtype=torch.bfloat16)

    xq, x_scale = _quantize_fp8_row(A)
    wq, w_scale = _quantize_fp8_row(B)
    wq_shuffled = flydsl_preshuffle(wq)

    # Warmup
    for _ in range(3):
        flydsl_preshuffle_gemm(xq, wq_shuffled, x_scale, w_scale)
    torch.cuda.synchronize()

    # Benchmark N iterations
    timings = []
    for _ in range(n_iterations):
        ms = triton.testing.do_bench(
            lambda: flydsl_preshuffle_gemm(xq, wq_shuffled, x_scale, w_scale),
            rep=rep_ms,
        )
        timings.append(ms)

    arr = np.array(timings)
    median_ms = float(np.median(arr))
    mean_ms = float(np.mean(arr))
    std_ms = float(np.std(arr))
    cov_pct = (std_ms / mean_ms * 100) if mean_ms > 0 else 0.0

    tflops = 2 * M * N * K / (median_ms / 1e3) / 1e12
    input_bytes = xq.numel() + wq_shuffled.numel()  # FP8 = 1 byte each
    output_bytes = M * N * 2  # BF16 output = 2 bytes
    scale_bytes = (
        x_scale.numel() * x_scale.element_size()
        + w_scale.numel() * w_scale.element_size()
    )
    gbps = (input_bytes + output_bytes + scale_bytes) / (median_ms / 1e3) / 1e9

    # Regime classification
    rc = classify_regime(M, N, K)

    # Ceiling and achieved %
    ceiling_value = 0.0
    achieved_pct = 0.0
    if rc.ceiling_type == CeilingType.MFMA_PEAK:
        if roofline and "fp8" in roofline.mfma_peak_tflops:
            ceiling_value = roofline.mfma_peak_tflops["fp8"]
        else:
            ceiling_value = 4600.0  # datasheet fallback
        achieved_pct = (tflops / ceiling_value) * 100 if ceiling_value > 0 else 0
    elif rc.ceiling_type == CeilingType.HBM_BW:
        if roofline and roofline.hbm_bw_gbps > 0:
            ceiling_value = roofline.hbm_bw_gbps
        else:
            ceiling_value = triton.testing.get_dram_gbps()
        achieved_pct = (gbps / ceiling_value) * 100 if ceiling_value > 0 else 0

    # Target
    target_pct = rc.target_low
    target_pass = None
    if target_matrix:
        t = lookup_target(target_matrix, "FP8RowwisePreshuffleFlyDSL", M, N, K)
        if t:
            target_pct = t.target_pct
    if achieved_pct > 0:
        target_pass = achieved_pct >= target_pct

    return BenchResult(
        kernel="FP8RowwisePreshuffleFlyDSL",
        M=M,
        N=N,
        K=K,
        ms=mean_ms,
        median_ms=median_ms,
        tflops=tflops,
        gbps=gbps,
        regime=rc.regime.value,
        ceiling_type=rc.ceiling_type.value,
        ceiling_value=ceiling_value,
        achieved_pct=achieved_pct,
        target_pct=target_pct,
        target_pass=target_pass,
        cov_pct=cov_pct,
        n_iters=n_iterations,
    )


@click.command()
@click.option("--roofline-json", default=None, type=click.Path(exists=True))
@click.option("--target-template", default=None, type=click.Path(exists=True))
@click.option("--output-dir", default="/tmp/mslk_flydsl_bench")
@click.option("--M", "m_vals", default="1,16,128,1024,4096,8192")
@click.option("--N", "n_vals", default="8192")
@click.option("--K", "k_vals", default="8192")
@click.option("--stat-iterations", default=5, type=int)
@click.option("--device", default=0, type=int)
@click.option(
    "--shapes",
    default=None,
    help="Preset: llama3_70b, llama3_405b, roofline_validation",
)
def main(
    roofline_json,
    target_template,
    output_dir,
    m_vals,
    n_vals,
    k_vals,
    stat_iterations,
    device,
    shapes,
):
    os.makedirs(output_dir, exist_ok=True)

    roofline = load_empirical_roofline(roofline_json) if roofline_json else None
    target_matrix = None
    if target_template:
        target_matrix = load_target_matrix(target_template)

    # Build shape list
    if shapes == "llama3_70b":
        shape_list = []
        for m in [1, 16, 32, 64, 128]:
            shape_list += [(m, 1280, 8192), (m, 7168, 8192), (m, 8192, 3584)]
    elif shapes == "llama3_405b":
        shape_list = []
        for m in [1, 128]:
            shape_list += [(m, 13312, 6656), (m, 16384, 16384)]
    elif shapes == "roofline_validation":
        shape_list = [
            (8192, 8192, 8192),
            (4096, 4096, 4096),
            (1024, 8192, 8192),
            (128, 8192, 8192),
            (16, 8192, 8192),
            (1, 8192, 8192),
        ]
    else:
        Ms = [int(x) for x in m_vals.split(",")]
        Ns = [int(x) for x in n_vals.split(",")]
        Ks = [int(x) for x in k_vals.split(",")]
        shape_list = [(m, n, k) for m in Ms for n in Ns for k in Ks]

    if roofline:
        fp8_peak = roofline.mfma_peak_tflops.get("fp8", 0)
        print(
            f"Empirical roofline: FP8 MFMA={fp8_peak:.1f} TFLOPS, "
            f"HBM={roofline.hbm_bw_gbps:.1f} GB/s"
        )
    shape_count = len(shape_list)
    print(f"Benchmarking {shape_count} shapes with {stat_iterations} iterations each\n")

    # Header
    print(
        f"{'Shape':<25} {'Regime':<6} {'Ms':<10} {'TFLOPS':<10} {'GB/s':<10} "
        f"{'Ceiling':<15} {'Achieved%':<10} {'Target%':<9} {'Pass':<5} {'CoV%':<7}"
    )
    print("-" * 120)

    results = []
    for M, N, K in shape_list:
        try:
            r = bench_flydsl_preshuffle(
                M,
                N,
                K,
                roofline,
                target_matrix,
                n_iterations=stat_iterations,
                device=device,
            )
            pass_str = (
                "PASS" if r.target_pass else ("FAIL" if r.target_pass is False else "")
            )
            ceiling_str = f"{r.ceiling_type}={r.ceiling_value:.0f}"
            shape_str = f"({r.M},{r.N},{r.K})"
            print(
                f"{shape_str:<25} "
                f"{r.regime:<6} {r.median_ms:<10.3f} {r.tflops:<10.2f} {r.gbps:<10.2f} "
                f"{ceiling_str:<15} {r.achieved_pct:<10.2f} {r.target_pct:<9.1f} "
                f"{pass_str:<5} {r.cov_pct:<7.2f}"
            )
            results.append(r)
        except RuntimeError as error:
            print(f"({M},{N},{K}) FAILED: {error}")

    # Export CSV
    if results:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = os.path.join(output_dir, f"flydsl_bench_{timestamp}.csv")
        rows = [
            {
                "kernel": r.kernel,
                "M": r.M,
                "N": r.N,
                "K": r.K,
                "median_ms": r.median_ms,
                "tflops": r.tflops,
                "gbps": r.gbps,
                "regime": r.regime,
                "ceiling_type": r.ceiling_type,
                "ceiling_value": r.ceiling_value,
                "achieved_pct": r.achieved_pct,
                "target_pct": r.target_pct,
                "target_pass": r.target_pass,
                "cov_pct": r.cov_pct,
                "n_iters": r.n_iters,
            }
            for r in results
        ]
        pd.DataFrame(rows).to_csv(csv_path, index=False)
        print(f"\nCSV saved to {csv_path}")

        # Summary
        passed = sum(1 for r in results if r.target_pass is True)
        failed = sum(1 for r in results if r.target_pass is False)
        total = passed + failed
        print(f"\nSummary: {passed}/{total} targets met")

        # Update target matrix if provided
        if target_matrix:
            for r in results:
                t = lookup_target(target_matrix, r.kernel, r.M, r.N, r.K)
                if t:
                    t.baseline_achieved_pct = r.achieved_pct
                    t.ceiling_value = r.ceiling_value
            matrix_path = os.path.join(output_dir, f"target_matrix_{timestamp}.yaml")
            save_target_matrix(target_matrix, matrix_path)
            print(f"Updated target matrix saved to {matrix_path}")


if __name__ == "__main__":
    main()
