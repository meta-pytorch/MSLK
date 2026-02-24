#!/usr/bin/env python3
# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.


"""
NCU Report Analyzer — extract key metrics from .ncu-rep files.

Usage:
    python3 analyze_ncu.py <report.ncu-rep> [--compare <other.ncu-rep>] [--metrics <m1,m2,...>] [--json] [--csv]

Examples:
    # Summary of all kernels
    python3 analyze_ncu.py profile.ncu-rep

    # Compare two reports side-by-side
    python3 analyze_ncu.py baseline.ncu-rep --compare optimized.ncu-rep

    # Extract specific metrics
    python3 analyze_ncu.py profile.ncu-rep --metrics gpu__time_duration.sum,sm__throughput.avg.pct_of_peak_sustained_elapsed

    # JSON output for further processing
    python3 analyze_ncu.py profile.ncu-rep --json

    # CSV output
    python3 analyze_ncu.py profile.ncu-rep --csv

    # List all available metrics in the report
    python3 analyze_ncu.py profile.ncu-rep --list-metrics

    # Show rule/bottleneck analysis
    python3 analyze_ncu.py profile.ncu-rep --rules
"""

import argparse
import json
import os
import sys

# Auto-detect NCU Python module path
NCU_PYTHON_PATHS = sorted(
    [
        os.path.join(d, "extras/python")
        for d in (
            os.path.join("/opt/nvidia/nsight-compute", v)
            for v in os.listdir("/opt/nvidia/nsight-compute/")
        )
        if os.path.isdir(os.path.join(d, "extras/python"))
    ],
    reverse=True,  # prefer newest version
)

for p in NCU_PYTHON_PATHS:
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    import ncu_report
except ImportError:
    print(
        "ERROR: Cannot import ncu_report. Ensure NVIDIA Nsight Compute is installed.",
        file=sys.stderr,
    )
    print(
        "Try: export PYTHONPATH=/opt/nvidia/nsight-compute/<version>/extras/python:$PYTHONPATH",
        file=sys.stderr,
    )
    sys.exit(1)

# Key performance metrics to extract by default
DEFAULT_METRICS = [
    # Timing
    "gpu__time_duration.sum",
    "gpu__time_duration.avg",
    # Compute
    "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "sm__warps_active.avg.pct_of_peak_sustained_active",
    # Memory
    "dram__throughput.avg.pct_of_peak_sustained_elapsed",
    "dram__bytes.sum",
    "dram__bytes_read.sum",
    "dram__bytes_write.sum",
    # Cache
    "l1tex__throughput.avg.pct_of_peak_sustained_elapsed",
    "lts__throughput.avg.pct_of_peak_sustained_elapsed",
    "lts__t_sector_hit_rate.pct",
    # Launch config
    "launch__grid_size",
    "launch__block_size",
    "launch__registers_per_thread",
    "launch__occupancy_limit_registers",
    "launch__waves_per_multiprocessor",
    # Instruction
    "smsp__inst_executed.sum",
    "sm__inst_executed_pipe_tensor.sum",
]


def safe_metric_value(action, metric_name):
    """Safely extract a metric value, returning None if not found."""
    m = action.metric_by_name(metric_name)
    if m is None:
        return None
    try:
        return m.value()
    except Exception:
        return None


def format_value(val, unit=""):
    """Format a metric value for display."""
    if val is None:
        return "N/A"
    if isinstance(val, float):
        if unit and "pct" in unit.lower():
            return f"{val:.2f}%"
        if abs(val) >= 1e9:
            return f"{val:.3e}"
        if abs(val) >= 1e3:
            return f"{val:,.1f}"
        return f"{val:.4f}"
    if isinstance(val, int):
        if abs(val) >= 1e9:
            return f"{val:,}"
        return str(val)
    return str(val)


def extract_kernel_data(action, metric_names):
    """Extract metrics from a single kernel action."""
    data = {"kernel_name": action.name()}
    for name in metric_names:
        val = safe_metric_value(action, name)
        data[name] = val
    return data


def extract_rule_results(action):
    """Extract rule/bottleneck analysis from an action."""
    results = []
    try:
        for rule_dict in action.rule_results_as_dicts():
            result = {
                "rule": rule_dict.get("name", ""),
                "section": rule_dict.get("section_identifier", ""),
            }
            msg = rule_dict.get("rule_message", {})
            if msg:
                result["title"] = msg.get("title", "")
                result["message"] = msg.get("message", "")
            speedup = rule_dict.get("speedup_estimation", {})
            if speedup and speedup.get("speedup", 0) > 0:
                result["estimated_speedup"] = f"{speedup['speedup']:.2f}x"
            focus = rule_dict.get("focus_metrics", [])
            if focus:
                result["focus_metrics"] = [
                    {"name": f["name"], "value": f"{f['value']:.2f}"} for f in focus
                ]
            results.append(result)
    except Exception:
        pass
    return results


def print_summary(report_path, metric_names, show_rules=False):
    """Print a summary table for all kernels in a report."""
    ctx = ncu_report.load_report(report_path)
    print(f"\n{'=' * 80}")
    print(f"NCU Report: {report_path}")
    print(f"{'=' * 80}")

    for ri in range(ctx.num_ranges()):
        rng = ctx.range_by_idx(ri)
        for ai in range(rng.num_actions()):
            action = rng.action_by_idx(ai)
            data = extract_kernel_data(action, metric_names)

            print(f"\n--- Kernel: {data['kernel_name']} ---")
            for name in metric_names:
                val = data.get(name)
                m = action.metric_by_name(name)
                unit = m.unit() if m else ""
                label = name.replace(".", " / ")
                print(f"  {label:60s} {format_value(val, unit):>15s} {unit}")

            if show_rules:
                rules = extract_rule_results(action)
                if rules:
                    print("\n  Bottleneck Analysis:")
                    for r in rules:
                        if r.get("message"):
                            print(f"    [{r.get('section', '')}] {r.get('title', '')}")
                            # Truncate long messages
                            msg = r["message"]
                            if len(msg) > 200:
                                msg = msg[:200] + "..."
                            print(f"      {msg}")
                            if r.get("estimated_speedup"):
                                print(
                                    f"      Estimated speedup: {r['estimated_speedup']}"
                                )


def print_comparison(path_a, path_b, metric_names):
    """Print side-by-side comparison of two reports."""
    ctx_a = ncu_report.load_report(path_a)
    ctx_b = ncu_report.load_report(path_b)

    print(f"\n{'=' * 100}")
    print("NCU Comparison")
    print(f"  A: {path_a}")
    print(f"  B: {path_b}")
    print(f"{'=' * 100}")

    range_a = ctx_a.range_by_idx(0)
    range_b = ctx_b.range_by_idx(0)

    # Compare kernel by kernel (by index)
    max_actions = max(range_a.num_actions(), range_b.num_actions())
    for ai in range(max_actions):
        action_a = range_a.action_by_idx(ai) if ai < range_a.num_actions() else None
        action_b = range_b.action_by_idx(ai) if ai < range_b.num_actions() else None

        name_a = action_a.name() if action_a else "N/A"
        name_b = action_b.name() if action_b else "N/A"

        print(f"\n--- Kernel A: {name_a}")
        print(f"--- Kernel B: {name_b}")
        print(f"  {'Metric':<55s} {'A':>12s} {'B':>12s} {'Delta':>10s}")
        print(f"  {'-' * 55} {'-' * 12} {'-' * 12} {'-' * 10}")

        for name in metric_names:
            val_a = safe_metric_value(action_a, name) if action_a else None
            val_b = safe_metric_value(action_b, name) if action_b else None

            m = (
                (action_a or action_b).metric_by_name(name)
                if (action_a or action_b)
                else None
            )
            unit = m.unit() if m else ""

            delta = ""
            if (
                isinstance(val_a, (int, float))
                and isinstance(val_b, (int, float))
                and val_a != 0
            ):
                pct = ((val_b - val_a) / abs(val_a)) * 100
                delta = f"{pct:+.1f}%"

            label = (
                name.split(".")[-2] + "." + name.split(".")[-1] if "." in name else name
            )
            if len(label) > 55:
                label = label[:52] + "..."
            print(
                f"  {label:<55s} {format_value(val_a, unit):>12s} {format_value(val_b, unit):>12s} {delta:>10s}"
            )


def print_json(report_path, metric_names):
    """Output metrics as JSON."""
    ctx = ncu_report.load_report(report_path)
    output = {"report": report_path, "kernels": []}

    for ri in range(ctx.num_ranges()):
        rng = ctx.range_by_idx(ri)
        for ai in range(rng.num_actions()):
            action = rng.action_by_idx(ai)
            data = extract_kernel_data(action, metric_names)
            output["kernels"].append(data)

    print(json.dumps(output, indent=2, default=str))


def print_csv(report_path, metric_names):
    """Output metrics as CSV."""
    ctx = ncu_report.load_report(report_path)
    header = ["kernel_name"] + metric_names
    print(",".join(header))

    for ri in range(ctx.num_ranges()):
        rng = ctx.range_by_idx(ri)
        for ai in range(rng.num_actions()):
            action = rng.action_by_idx(ai)
            data = extract_kernel_data(action, metric_names)
            row = [str(data.get(h, "")) for h in header]
            print(",".join(row))


def list_metrics(report_path):
    """List all available metrics in the report."""
    ctx = ncu_report.load_report(report_path)
    rng = ctx.range_by_idx(0)
    action = rng.action_by_idx(0)

    print(f"\nAvailable metrics in: {report_path}")
    print(f"Kernel: {action.name()}")
    print(f"Total metrics: {len(action)}\n")

    for name in sorted(action.metric_names()):
        m = action.metric_by_name(name)
        val = safe_metric_value(action, name)
        unit = m.unit() if m else ""
        desc = m.description() if m else ""
        if val is not None:
            print(
                f"  {name:<70s} = {format_value(val, unit):>15s} {unit:<10s} {desc[:50]}"
            )


def main():
    parser = argparse.ArgumentParser(description="Analyze NCU .ncu-rep report files")
    parser.add_argument("report", help="Path to .ncu-rep file")
    parser.add_argument("--compare", help="Second .ncu-rep file for comparison")
    parser.add_argument(
        "--metrics", help="Comma-separated list of metric names to extract"
    )
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parser.add_argument("--csv", action="store_true", help="Output as CSV")
    parser.add_argument(
        "--list-metrics", action="store_true", help="List all available metrics"
    )
    parser.add_argument(
        "--rules", action="store_true", help="Show rule/bottleneck analysis"
    )

    args = parser.parse_args()

    if args.metrics:
        metric_names = [m.strip() for m in args.metrics.split(",")]
    else:
        metric_names = DEFAULT_METRICS

    if args.list_metrics:
        list_metrics(args.report)
    elif args.compare:
        print_comparison(args.report, args.compare, metric_names)
    elif args.json:
        print_json(args.report, metric_names)
    elif args.csv:
        print_csv(args.report, metric_names)
    else:
        print_summary(args.report, metric_names, show_rules=args.rules)


if __name__ == "__main__":
    main()
