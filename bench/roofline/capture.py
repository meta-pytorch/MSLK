# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import atexit
import csv
import json
import logging
import os
import subprocess
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import click
import torch


logger = logging.getLogger(__name__)


@dataclass
class EmpiricalRoofline:
    gpu_arch: str = ""
    gpu_name: str = ""
    sclk_mhz: int = 0
    mclk_mhz: int = 0
    hbm_bw_gbps: float = 0.0
    mfma_peak_tflops: dict[str, float] = field(default_factory=dict)
    valu_peak_tflops: dict[str, float] = field(default_factory=dict)
    tool_version: str = ""
    timestamp: str = ""
    roofline_csv_path: str | None = None


_DTYPE_MAP = {
    "fp4": "FP4",
    "fp6": "FP6",
    "fp8": "FP8",
    "fp16": "FP16",
    "bf16": "BF16",
    "fp32": "FP32",
}


def _run_cmd(cmd: list[str], check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, check=check)


def _get_gpu_arch(device_id: int = 0) -> str:
    try:
        properties = torch.cuda.get_device_properties(device_id)
        return properties.gcnArchName.split(":", maxsplit=1)[0]
    except (AssertionError, AttributeError, RuntimeError) as error:
        logger.warning(
            "Unable to read GPU architecture for device %s: %s", device_id, error
        )
        return ""


def _get_gpu_name(device_id: int = 0) -> str:
    try:
        r = _run_cmd(
            ["amd-smi", "static", "--gpu", str(device_id), "--asic"],
            check=False,
        )
        for line in r.stdout.splitlines():
            if "market_name" in line.lower() or "Market" in line:
                return line.split(":")[-1].strip()
        return ""
    except (OSError, subprocess.SubprocessError) as error:
        logger.warning("Unable to read GPU name for device %s: %s", device_id, error)
        return ""


def _get_tool_version() -> str:
    try:
        r = _run_cmd(["rocprof-compute", "--version"], check=False)
        return r.stdout.strip().splitlines()[0] if r.stdout else ""
    except (OSError, subprocess.SubprocessError) as error:
        logger.warning("Unable to read rocprof-compute version: %s", error)
        return ""


def lock_clocks(
    sclk_mhz: int | None = None,
    mclk_mhz: int | None = None,
    device_id: int | None = None,
) -> tuple[int, int]:
    """Lock GPU clocks via amd-smi. Returns (actual_sclk, actual_mclk)."""
    gpu_flag = ["--gpu", str(device_id)] if device_id is not None else []

    atexit.register(unlock_clocks, device_id=device_id)
    _run_cmd(["amd-smi", "set", *gpu_flag, "--perf-level", "manual"])

    if sclk_mhz is not None:
        _run_cmd(["amd-smi", "set", *gpu_flag, "--sclk", str(sclk_mhz)])
    if mclk_mhz is not None:
        _run_cmd(["amd-smi", "set", *gpu_flag, "--mclk", str(mclk_mhz)])

    actual_sclk = sclk_mhz or 0
    actual_mclk = mclk_mhz or 0

    try:
        r = _run_cmd(["amd-smi", "metric", *gpu_flag, "--clock"], check=False)
        for line in r.stdout.splitlines():
            if "sclk" in line.lower() and "mhz" in line.lower():
                parts = [p for p in line.split() if p.replace(".", "").isdigit()]
                if parts:
                    actual_sclk = int(float(parts[0]))
            if "mclk" in line.lower() and "mhz" in line.lower():
                parts = [p for p in line.split() if p.replace(".", "").isdigit()]
                if parts:
                    actual_mclk = int(float(parts[0]))
    except (OSError, subprocess.SubprocessError, ValueError) as error:
        logger.warning("Unable to read back clocks for device %s: %s", device_id, error)

    return actual_sclk, actual_mclk


def unlock_clocks(device_id: int | None = None) -> None:
    """Restore clocks to auto mode."""
    gpu_flag = ["--gpu", str(device_id)] if device_id is not None else []
    try:
        _run_cmd(["amd-smi", "set", *gpu_flag, "--perf-level", "auto"], check=False)
    except (OSError, subprocess.SubprocessError) as error:
        logger.warning("Unable to restore clocks for device %s: %s", device_id, error)


def run_roofline_profile(
    benchmark_cmd: list[str],
    dtypes: list[str] | None = None,
    device_id: int = 0,
    output_dir: str | None = None,
    workload_name: str = "mslk_roofline",
) -> str:
    """Run rocprof-compute profile --roof-only and return path to workload directory.

    Args:
        benchmark_cmd: The command to profile (e.g. ["python", "bench.py", ...]).
            If empty, uses a built-in microbenchmark.
        dtypes: List of dtype strings (e.g. ["fp8", "bf16"]). Profiles each separately.
        device_id: GPU device index.
        output_dir: Where to store the workload. Defaults to a temp dir.
    """
    if output_dir is None:
        output_dir = tempfile.mkdtemp(prefix="mslk_roofline_")

    workload_dir = os.path.join(output_dir, workload_name)

    if not dtypes:
        dtypes = ["fp8"]

    rocprof_dtypes = []
    for d in dtypes:
        mapped = _DTYPE_MAP.get(d.lower(), d.upper())
        rocprof_dtypes.append(mapped)

    cmd = [
        "rocprof-compute",
        "profile",
        "--name",
        workload_name,
        "--path",
        output_dir,
        "--roof-only",
        "--device",
        str(device_id),
        "-R",
        *rocprof_dtypes,
        "--",
        *benchmark_cmd,
    ]

    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if result.returncode != 0:
        raise RuntimeError(
            f"rocprof-compute profile failed (rc={result.returncode}):\n"
            f"stdout: {result.stdout}\n"
            f"stderr: {result.stderr}"
        )

    return workload_dir


def parse_roofline_csv(csv_path: str, device_id: int = 0) -> dict:
    """Parse roofline.csv from rocprof-compute.

    Returns dict with keys like:
        "hbm_bw_gbps": float
        "l2_bw_gbps": float
        "mfma_peak_gflops": {"FP8": float, "BF16": float, ...}
        "valu_peak_gflops": {"FP8": float, ...}
    """
    with open(csv_path) as f:
        reader = csv.reader(f)
        rows = list(reader)

    if len(rows) < 2:
        raise ValueError(f"roofline.csv has fewer than 2 rows: {csv_path}")

    # First column is devID, skip it.
    headers = rows[0][1:]
    device_row = next(
        (row for row in rows[1:] if row and row[0].strip() == str(device_id)),
        None,
    )
    if device_row is None:
        raise ValueError(f"roofline.csv has no row for device {device_id}: {csv_path}")

    data = {}
    for header, value in zip(headers, device_row[1:]):
        try:
            data[header] = float(value)
        except ValueError:
            continue

    result: dict = {
        "hbm_bw_gbps": data.get("HBMBw", 0.0),
        "l2_bw_gbps": data.get("L2Bw", 0.0),
        "mfma_peak_gflops": {},
        "valu_peak_gflops": {},
    }

    # Column naming: MFMAF8Flops, MFMABF16Flops, MFMAF32Flops, etc.
    _MFMA_COL = {
        "FP4": "MFMAF4Flops",
        "FP6": "MFMAF6Flops",
        "FP8": "MFMAF8Flops",
        "FP16": "MFMAF16Flops",
        "BF16": "MFMABF16Flops",
        "FP32": "MFMAF32Flops",
    }
    for dtype_key, mfma_col in _MFMA_COL.items():
        valu_col = f"{dtype_key}Flops"
        if mfma_col in data:
            result["mfma_peak_gflops"][dtype_key.lower()] = data[mfma_col]
        if valu_col in data:
            result["valu_peak_gflops"][dtype_key.lower()] = data[valu_col]

    return result


def _build_microbenchmark_cmd() -> list[str]:
    """Build a simple PyTorch GEMM command for roofline profiling.

    Uses a temp script file to avoid shell quoting issues with rocprof-compute.
    """
    script_dir = tempfile.mkdtemp(prefix="mslk_roofline_script_")
    script_path = os.path.join(script_dir, "bench_kernel.py")
    with open(script_path, "w") as f:
        f.write(
            "import torch\n"
            "a = torch.randn(8192, 8192, device='cuda', dtype=torch.bfloat16)\n"
            "b = torch.randn(8192, 8192, device='cuda', dtype=torch.bfloat16)\n"
            "torch.cuda.synchronize()\n"
            "for _ in range(20):\n"
            "    torch.mm(a, b)\n"
            "torch.cuda.synchronize()\n"
        )
    return ["python3", script_path]


def capture_empirical_roofline(
    output_path: str = "roofline.json",
    dtypes: list[str] | None = None,
    lock_sclk_mhz: int | None = None,
    lock_mclk_mhz: int | None = None,
    device_id: int = 0,
    benchmark_cmd: list[str] | None = None,
    output_dir: str | None = None,
) -> EmpiricalRoofline:
    """End-to-end: lock clocks -> profile -> parse -> write JSON -> unlock clocks."""
    if dtypes is None:
        dtypes = ["fp8", "bf16"]

    actual_sclk = 0
    actual_mclk = 0
    if lock_sclk_mhz is not None or lock_mclk_mhz is not None:
        actual_sclk, actual_mclk = lock_clocks(lock_sclk_mhz, lock_mclk_mhz, device_id)

    try:
        if benchmark_cmd is None:
            benchmark_cmd = _build_microbenchmark_cmd()

        workload_dir = run_roofline_profile(
            benchmark_cmd=benchmark_cmd,
            dtypes=dtypes,
            device_id=device_id,
            output_dir=output_dir,
        )

        csv_path = os.path.join(workload_dir, "roofline.csv")
        if not os.path.exists(csv_path):
            candidates = list(Path(workload_dir).rglob("roofline.csv"))
            if candidates:
                csv_path = str(candidates[0])
            else:
                raise FileNotFoundError(f"roofline.csv not found in {workload_dir}")

        parsed = parse_roofline_csv(csv_path, device_id=device_id)

        roofline = EmpiricalRoofline(
            gpu_arch=_get_gpu_arch(device_id),
            gpu_name=_get_gpu_name(device_id),
            sclk_mhz=actual_sclk,
            mclk_mhz=actual_mclk,
            hbm_bw_gbps=parsed["hbm_bw_gbps"],
            mfma_peak_tflops={
                k: v / 1000.0 for k, v in parsed["mfma_peak_gflops"].items()
            },
            valu_peak_tflops={
                k: v / 1000.0 for k, v in parsed["valu_peak_gflops"].items()
            },
            tool_version=_get_tool_version(),
            timestamp=datetime.now(timezone.utc).isoformat(),
            roofline_csv_path=csv_path,
        )

        with open(output_path, "w") as f:
            json.dump(asdict(roofline), f, indent=2)
        print(f"Empirical roofline saved to {output_path}")

        return roofline

    finally:
        if lock_sclk_mhz is not None or lock_mclk_mhz is not None:
            unlock_clocks(device_id)


def load_empirical_roofline(json_path: str) -> EmpiricalRoofline:
    with open(json_path) as f:
        data = json.load(f)
    return EmpiricalRoofline(**data)


@click.command()
@click.option("--output", default="roofline.json", help="Output JSON path")
@click.option("--dtypes", default="fp8,bf16", help="Comma-separated dtypes")
@click.option("--lock-sclk", default=None, type=int, help="Lock SCLK in MHz")
@click.option("--lock-mclk", default=None, type=int, help="Lock MCLK in MHz")
@click.option("--device", default=0, type=int, help="GPU device ID")
@click.option(
    "--workdir",
    default=None,
    help="Directory for rocprof-compute workload output",
)
def capture_cli(
    output: str,
    dtypes: str,
    lock_sclk: Optional[int],
    lock_mclk: Optional[int],
    device: int,
    workdir: Optional[str],
) -> None:
    """Capture empirical roofline ceilings via rocprof-compute."""
    dtype_list = [d.strip() for d in dtypes.split(",")]
    roofline = capture_empirical_roofline(
        output_path=output,
        dtypes=dtype_list,
        lock_sclk_mhz=lock_sclk,
        lock_mclk_mhz=lock_mclk,
        device_id=device,
        output_dir=workdir,
    )
    print(f"\nGPU: {roofline.gpu_arch} ({roofline.gpu_name})")
    print(f"HBM BW: {roofline.hbm_bw_gbps:.1f} GB/s")
    for dtype, tflops in roofline.mfma_peak_tflops.items():
        print(f"MFMA {dtype}: {tflops:.1f} TFLOPS")


if __name__ == "__main__":
    capture_cli()
