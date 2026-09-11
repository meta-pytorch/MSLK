#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""WP-1a orchestrator: capture roofline -> benchmark -> classify -> report."""

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import click
import pandas as pd
import torch
from mslk.bench.roofline.capture import (
    capture_empirical_roofline,
    EmpiricalRoofline,
    load_empirical_roofline,
)
from mslk.bench.roofline.targets import (
    load_target_matrix,
    save_target_matrix,
    TargetMatrix,
)


def _run_benchmarks(
    kernel_names: list[str],
    shapes: list[tuple[int, int, int]],
    roofline: EmpiricalRoofline | None,
    opts: dict,
) -> list[dict[str, Any]]:
    """Run GEMM benchmarks and return results as list of dicts."""
    from mslk.bench.common.utils import BenchOptions

    # Import here to avoid circular deps and heavy torch init at module level
    from mslk.bench.gemm.gemm_bench import (
        benchmark,
        collect_kernels_to_profile,
        get_hbm_bw_gbps,
        set_bench_options,
        set_empirical_roofline,
        ShapeMode,
    )

    if roofline:
        set_empirical_roofline(roofline)

    set_bench_options(
        stat_iterations=opts.get("stat_iterations", 5),
        cov_bound=opts.get("cov_bound", 3.0),
        cold_l2=opts.get("cold_l2", True),
    )

    bench_opts = BenchOptions(
        cuda_graph=opts.get("cuda_graph", True),
        rotating_buffer=opts.get("rotating_buffer", False),
        rep_ms=opts.get("rep_ms", 200),
    )

    gemm_ops = collect_kernels_to_profile(kernel_names, is_grouped=False)
    if not gemm_ops:
        print(f"WARNING: No matching kernels found for {kernel_names}")
        return []

    mem_bw_gbps = get_hbm_bw_gbps()
    all_results: list[dict[str, Any]] = []

    for m, n, k in shapes:
        metrics_list = benchmark(
            gemm_ops, m, n, k, mem_bw_gbps, bench_opts, ShapeMode.REGULAR
        )
        for met in metrics_list:
            all_results.append(met.as_dict())

    return all_results


def _generate_report(
    output_dir: str,
    roofline: EmpiricalRoofline | None,
    matrix: TargetMatrix,
    bench_results: list[dict],
    conditions: dict,
) -> None:
    """Generate WP-1a report artifacts."""
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    # Save target matrix YAML
    yaml_path = os.path.join(output_dir, f"target_matrix_{timestamp}.yaml")
    save_target_matrix(matrix, yaml_path)
    print(f"Target matrix saved to {yaml_path}")

    # Save target matrix CSV
    rows = []
    for t in matrix.targets:
        rows.append(
            {
                "kernel": t.kernel_name,
                "M": t.M,
                "N": t.N,
                "K": t.K,
                "regime": t.regime,
                "ceiling_type": t.ceiling_type,
                "ceiling_value": t.ceiling_value,
                "target_pct": t.target_pct,
                "baseline_achieved_pct": t.baseline_achieved_pct,
                "model_source": t.model_source,
                "pass": (
                    t.baseline_achieved_pct >= t.target_pct
                    if t.baseline_achieved_pct is not None
                    else None
                ),
                "notes": t.notes,
            }
        )
    csv_path = os.path.join(output_dir, f"target_matrix_{timestamp}.csv")
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"Target matrix CSV saved to {csv_path}")

    # Save benchmark results CSV
    if bench_results:
        bench_csv = os.path.join(output_dir, f"bench_results_{timestamp}.csv")
        pd.DataFrame(bench_results).to_csv(bench_csv, index=False)
        print(f"Benchmark results saved to {bench_csv}")

    # Save conditions
    cond_path = os.path.join(output_dir, f"measurement_conditions_{timestamp}.json")
    with open(cond_path, "w") as f:
        json.dump(conditions, f, indent=2)

    # Human-readable summary
    report_path = os.path.join(output_dir, f"wp1a_report_{timestamp}.txt")
    with open(report_path, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("MSLK WP-1a Roofline Report\n")
        f.write(f"Generated: {datetime.now(timezone.utc).isoformat()}\n")
        f.write("=" * 80 + "\n\n")

        if roofline:
            f.write("EMPIRICAL ROOFLINE\n")
            f.write(f"  GPU: {roofline.gpu_arch} ({roofline.gpu_name})\n")
            f.write(f"  HBM BW: {roofline.hbm_bw_gbps:.1f} GB/s\n")
            for dtype, tflops in roofline.mfma_peak_tflops.items():
                f.write(f"  MFMA {dtype}: {tflops:.1f} TFLOPS\n")
            f.write(f"  SCLK: {roofline.sclk_mhz} MHz, MCLK: {roofline.mclk_mhz} MHz\n")
            f.write(f"  Tool: {roofline.tool_version}\n\n")

        f.write("TARGET MATRIX\n")
        f.write(
            f"{'Kernel':<35} {'Shape':<25} {'Regime':<6} {'Ceiling':<15} "
            f"{'Target%':<9} {'Achieved%':<11} {'Pass':<5}\n"
        )
        f.write("-" * 110 + "\n")

        pass_count = 0
        fail_count = 0
        for t in matrix.targets:
            achieved = (
                f"{t.baseline_achieved_pct:.1f}"
                if t.baseline_achieved_pct is not None
                else "N/A"
            )
            if t.baseline_achieved_pct is not None:
                passed = t.baseline_achieved_pct >= t.target_pct
                pass_str = "PASS" if passed else "FAIL"
                if passed:
                    pass_count += 1
                else:
                    fail_count += 1
            else:
                pass_str = ""
            shape_str = f"({t.M},{t.N},{t.K})"
            f.write(
                f"{t.kernel_name:<35} {shape_str:<25} "
                f"{t.regime:<6} {t.ceiling_type:<15} {t.target_pct:<9.1f} "
                f"{achieved:<11} {pass_str:<5}\n"
            )

        f.write("\n")
        total = pass_count + fail_count
        if total > 0:
            pass_pct = pass_count / total * 100
            f.write(f"SUMMARY: {pass_count}/{total} targets met ({pass_pct:.0f}%)\n")
        else:
            f.write("SUMMARY: No results; run on hardware to fill baselines.\n")

    print(f"Report saved to {report_path}")


def _load_matrix(target_template: str | None) -> TargetMatrix:
    if target_template:
        return load_target_matrix(target_template)

    default_template = Path(__file__).parent / "templates" / "target_matrix.yaml"
    if not default_template.exists():
        return TargetMatrix()

    matrix = load_target_matrix(str(default_template))
    print(f"Loaded default template ({len(matrix.targets)} targets)")
    return matrix


def _apply_roofline(matrix: TargetMatrix, roofline: EmpiricalRoofline) -> None:
    matrix.gpu_arch = roofline.gpu_arch
    matrix.tool_version = roofline.tool_version
    for target in matrix.targets:
        if target.ceiling_type == "mfma_peak":
            target.ceiling_value = roofline.mfma_peak_tflops.get("fp8", 0.0)
        elif target.ceiling_type == "hbm_bw":
            target.ceiling_value = roofline.hbm_bw_gbps


def _select_shapes(
    shapes: str | None,
    matrix: TargetMatrix,
) -> list[tuple[int, int, int]]:
    from mslk.bench.gemm.gemm_bench import shape_registry

    if shapes:
        if shapes not in shape_registry:
            valid_shapes = ", ".join(shape_registry)
            raise click.BadParameter(
                f"unknown shape registry {shapes!r}; choose from {valid_shapes}",
                param_hint="--shapes",
            )
        return shape_registry[shapes]()

    return sorted({(target.M, target.N, target.K) for target in matrix.targets})


def _fill_baselines(matrix: TargetMatrix, bench_results: list[dict[str, Any]]) -> None:
    targets_by_shape = {
        (target.M, target.N, target.K, target.kernel_name): target
        for target in matrix.targets
    }
    for result in bench_results:
        m = result.get("M")
        n = result.get("N")
        k = result.get("K")
        if not isinstance(m, int) or not isinstance(n, int) or not isinstance(k, int):
            continue
        for key, value in result.items():
            if not key.endswith("_achieved_pct") or value is None:
                continue
            kernel_name = key.removesuffix("_achieved_pct")
            target = targets_by_shape.get((m, n, k, kernel_name))
            if target is not None:
                target.baseline_achieved_pct = value


@click.command()
@click.option(
    "--roofline-json",
    default=None,
    type=click.Path(exists=True),
    help="Pre-captured roofline JSON. Skips roofline capture if provided.",
)
@click.option(
    "--target-template",
    default=None,
    type=click.Path(exists=True),
    help="Target matrix YAML template. Uses default if not provided.",
)
@click.option("--output-dir", default="/tmp/mslk_wp1a", help="Output directory.")
@click.option("--kernels", default=None, help="Comma-separated kernel names.")
@click.option(
    "--shapes",
    default=None,
    help="Shape registry name (llama3_70b, llama3_405b, llama4, ldm).",
)
@click.option("--lock-sclk", default=None, type=int, help="Lock SCLK in MHz.")
@click.option("--lock-mclk", default=None, type=int, help="Lock MCLK in MHz.")
@click.option("--device", default=0, type=int, help="GPU device ID.")
@click.option("--stat-iterations", default=5, type=int, help="Iterations for CoV.")
@click.option("--cov-bound", default=3.0, type=float, help="Max acceptable CoV %%.")
@click.option(
    "--cold-l2/--no-cold-l2",
    default=True,
    help="Use a rotating buffer for cold-L2 measurements.",
)
@click.option(
    "--skip-benchmark", is_flag=True, help="Generate template only, skip benchmarks."
)
@click.option("--skip-roofline", is_flag=True, help="Skip roofline capture.")
@click.option(
    "--dtypes", default="fp8,bf16", help="Comma-separated dtypes for roofline capture."
)
def run_wp1a(
    roofline_json: Optional[str],
    target_template: Optional[str],
    output_dir: str,
    kernels: Optional[str],
    shapes: Optional[str],
    lock_sclk: Optional[int],
    lock_mclk: Optional[int],
    device: int,
    stat_iterations: int,
    cov_bound: float,
    cold_l2: bool,
    skip_benchmark: bool,
    skip_roofline: bool,
    dtypes: str,
) -> None:
    """Run the complete WP-1a pipeline."""
    os.makedirs(output_dir, exist_ok=True)

    # ---- Step 1: Roofline capture ----
    roofline: EmpiricalRoofline | None = None
    if roofline_json:
        roofline = load_empirical_roofline(roofline_json)
        print(f"Loaded roofline from {roofline_json}")
    elif not skip_roofline:
        print("Step 1: Capturing empirical roofline...")
        dtype_list = [d.strip() for d in dtypes.split(",")]
        roofline_path = os.path.join(output_dir, "roofline.json")
        roofline = capture_empirical_roofline(
            output_path=roofline_path,
            dtypes=dtype_list,
            lock_sclk_mhz=lock_sclk,
            lock_mclk_mhz=lock_mclk,
            device_id=device,
            output_dir=output_dir,
        )

    # ---- Step 2: Load or build target matrix ----
    matrix = _load_matrix(target_template)

    if roofline:
        _apply_roofline(matrix, roofline)

    # ---- Step 3: Run benchmarks ----
    bench_results = []
    if not skip_benchmark:
        print("\nStep 3: Running benchmarks...")

        bench_shapes = _select_shapes(shapes, matrix)

        # Determine kernels
        kernel_list = [k.strip() for k in kernels.split(",")] if kernels else None
        if kernel_list is None:
            kernel_list = list({t.kernel_name for t in matrix.targets})

        bench_results = _run_benchmarks(
            kernel_names=kernel_list,
            shapes=bench_shapes,
            roofline=roofline,
            opts={
                "stat_iterations": stat_iterations,
                "cov_bound": cov_bound,
                "cold_l2": cold_l2,
            },
        )

        _fill_baselines(matrix, bench_results)

    # ---- Step 4: Generate report ----
    conditions = {
        "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else "N/A",
        "sclk_mhz": lock_sclk,
        "mclk_mhz": lock_mclk,
        "cache_state": "cold_l2" if cold_l2 else "warm_l2",
        "stat_iterations": stat_iterations,
        "cov_bound_pct": cov_bound,
        "roofline_source": "empirical" if roofline else "datasheet",
    }

    print("\nStep 4: Generating report...")
    _generate_report(output_dir, roofline, matrix, bench_results, conditions)

    print(f"\nWP-1a pipeline complete. Artifacts in {output_dir}/")


if __name__ == "__main__":
    run_wp1a()
