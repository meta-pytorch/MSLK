# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

import yaml
from mslk.bench.roofline.regime import classify_regime


@dataclass
class KernelTarget:
    kernel_name: str
    M: int
    N: int
    K: int
    regime: str
    ceiling_type: str
    target_pct: float
    model_source: str = ""
    ceiling_value: float | None = None
    baseline_achieved_pct: float | None = None
    notes: str = ""


@dataclass
class TargetMatrix:
    gpu_arch: str = ""
    tool_version: str = ""
    timestamp: str = ""
    measurement_conditions: dict[str, Any] = field(default_factory=dict)
    targets: list[KernelTarget] = field(default_factory=list)


def load_target_matrix(yaml_path: str) -> TargetMatrix:
    with open(yaml_path) as f:
        data = yaml.safe_load(f)

    matrix = TargetMatrix(
        gpu_arch=data.get("gpu_arch", ""),
        tool_version=data.get("tool_version", ""),
        timestamp=data.get("timestamp", ""),
        measurement_conditions=data.get("measurement_conditions", {}),
    )

    for kernel_block in data.get("targets", []):
        kernel_name = kernel_block["kernel"]
        model_source = kernel_block.get("model_source", "")
        for shape in kernel_block.get("shapes", []):
            matrix.targets.append(
                KernelTarget(
                    kernel_name=kernel_name,
                    M=shape["M"],
                    N=shape["N"],
                    K=shape["K"],
                    regime=shape["regime"],
                    ceiling_type=shape["ceiling_type"],
                    target_pct=shape["target_pct"],
                    model_source=model_source,
                    ceiling_value=shape.get("ceiling_value"),
                    baseline_achieved_pct=shape.get("baseline_achieved_pct"),
                    notes=shape.get("notes", ""),
                )
            )

    return matrix


def save_target_matrix(matrix: TargetMatrix, yaml_path: str) -> None:
    kernel_groups: dict[tuple[str, str], list[dict]] = {}
    for t in matrix.targets:
        key = (t.kernel_name, t.model_source)
        shape_entry: dict[str, Any] = {
            "M": t.M,
            "N": t.N,
            "K": t.K,
            "regime": t.regime,
            "ceiling_type": t.ceiling_type,
            "target_pct": t.target_pct,
        }
        if t.ceiling_value is not None:
            shape_entry["ceiling_value"] = round(t.ceiling_value, 2)
        if t.baseline_achieved_pct is not None:
            shape_entry["baseline_achieved_pct"] = round(t.baseline_achieved_pct, 2)
        if t.notes:
            shape_entry["notes"] = t.notes
        kernel_groups.setdefault(key, []).append(shape_entry)

    targets_list = []
    for (kernel, source), shapes in kernel_groups.items():
        entry: dict[str, Any] = {"kernel": kernel}
        if source:
            entry["model_source"] = source
        entry["shapes"] = shapes
        targets_list.append(entry)

    out = {
        "gpu_arch": matrix.gpu_arch,
        "tool_version": matrix.tool_version,
        "timestamp": matrix.timestamp or datetime.now(timezone.utc).isoformat(),
        "measurement_conditions": matrix.measurement_conditions,
        "targets": targets_list,
    }

    with open(yaml_path, "w") as f:
        yaml.dump(out, f, default_flow_style=False, sort_keys=False, width=120)


def build_default_target_matrix(
    kernel_names: list[str],
    shapes: list[tuple[int, int, int]],
    model_source: str = "",
    is_grouped: bool = False,
    num_groups: int | None = None,
) -> TargetMatrix:
    matrix = TargetMatrix()
    for kernel in kernel_names:
        for M, N, K in shapes:
            rc = classify_regime(M, N, K, is_grouped, num_groups)
            matrix.targets.append(
                KernelTarget(
                    kernel_name=kernel,
                    M=M,
                    N=N,
                    K=K,
                    regime=rc.regime.value,
                    ceiling_type=rc.ceiling_type.value,
                    target_pct=rc.target_low,
                    model_source=model_source,
                )
            )
    return matrix


def lookup_target(
    matrix: TargetMatrix,
    kernel_name: str,
    M: int,
    N: int,
    K: int,
) -> Optional[KernelTarget]:
    for t in matrix.targets:
        if t.kernel_name == kernel_name and t.M == M and t.N == N and t.K == K:
            return t
    return None
