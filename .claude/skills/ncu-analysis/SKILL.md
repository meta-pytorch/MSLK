---
name: ncu-analysis
description: 'Analyze CUDA kernel performance using NVIDIA Nsight Compute (NCU). Use when the user wants to: (1) profile a CUDA kernel with NCU to generate .ncu-rep files, (2) analyze NCU reports programmatically using the ncu_report Python API, (3) extract specific metrics from NCU profiles (duration, throughput, occupancy, cache hit rates), (4) compare two kernel profiles side-by-side, (5) identify performance bottlenecks in GPU kernels, or (6) work with .ncu-rep files in any way.'
---

# NCU Analysis

Profile CUDA kernels and analyze results using the NCU Python Report Interface.

## Step 1: Generate an NCU Profile

### Find NCU

NCU is installed at `/opt/nvidia/nsight-compute/`. Use the newest version:

```bash
ls /opt/nvidia/nsight-compute/
NCU_PATH="/opt/nvidia/nsight-compute/<newest_version>"
export PATH="$PATH:$NCU_PATH"
```

### Select a GPU

Run `nvidia-smi` and pick a GPU with no running processes. If all GPUs are busy, warn the user and ask before proceeding.

### Run NCU

```bash
CUDA_VISIBLE_DEVICES=<GPU_ID> CUDA_INJECTION64_PATH=none \
ncu -c <NUM_CAPTURES> --set full --target-processes all \
  -o <OUTPUT_NAME> \
  --kernel-name regex:"<KERNEL_REGEX>" \
  -f --import-source yes \
  <BINARY> [BINARY_ARGS...]
```

Key flags:
- `-c N` — number of kernel captures (4 is typical)
- `--set full` — collect all metrics (`--set basic` for faster runs)
- `--kernel-name regex:"pattern"` — filter which kernels to profile
- `-o name` — output file (produces `name.ncu-rep`)
- `-f` — overwrite existing output
- `--import-source yes` — embed source code in report
- `--target-processes all` — profile child processes too

Output is a `.ncu-rep` file (~50-60 MB with `--set full`).

To download: `scp $(hostname -f):<path>/<name>.ncu-rep .`

## Step 2: Analyze with Python Report Interface

### Setup

The `ncu_report` Python module lives at `<NCU_PATH>/extras/python/`:

```python
import sys
sys.path.insert(0, "/opt/nvidia/nsight-compute/<version>/extras/python")
import ncu_report
```

Or: `export PYTHONPATH="/opt/nvidia/nsight-compute/<version>/extras/python:$PYTHONPATH"`

### Bundled Analysis Script

Use `scripts/analyze_ncu.py` for common tasks:

```bash
SKILL_DIR=~/.claude/skills/ncu-analysis

# Summary with key metrics
python3 $SKILL_DIR/scripts/analyze_ncu.py profile.ncu-rep

# Compare two profiles side-by-side (shows delta %)
python3 $SKILL_DIR/scripts/analyze_ncu.py baseline.ncu-rep --compare optimized.ncu-rep

# Specific metrics only
python3 $SKILL_DIR/scripts/analyze_ncu.py profile.ncu-rep --metrics gpu__time_duration.sum,sm__throughput.avg.pct_of_peak_sustained_elapsed

# JSON or CSV output
python3 $SKILL_DIR/scripts/analyze_ncu.py profile.ncu-rep --json
python3 $SKILL_DIR/scripts/analyze_ncu.py profile.ncu-rep --csv

# List all available metrics in the report
python3 $SKILL_DIR/scripts/analyze_ncu.py profile.ncu-rep --list-metrics

# Bottleneck analysis from NCU rules engine
python3 $SKILL_DIR/scripts/analyze_ncu.py profile.ncu-rep --rules
```

### Custom Analysis

For the full `ncu_report` Python API, see `references/python-api.md`. Common pattern:

```python
import sys
sys.path.insert(0, "/opt/nvidia/nsight-compute/2025.3.1/extras/python")
import ncu_report

report = ncu_report.load_report("profile.ncu-rep")
rng = report.range_by_idx(0)

for i in range(rng.num_actions()):
    action = rng.action_by_idx(i)
    print(f"Kernel: {action.name()}")
    duration = action["gpu__time_duration.sum"]
    print(f"  Duration: {duration.value()} {duration.unit()}")
    compute = action["sm__throughput.avg.pct_of_peak_sustained_elapsed"]
    print(f"  Compute: {compute.value():.1f}%")
    memory = action["dram__throughput.avg.pct_of_peak_sustained_elapsed"]
    print(f"  Memory: {memory.value():.1f}%")

    # Bottleneck analysis
    for rule in action.rule_results_as_dicts():
        msg = rule.get("rule_message", {})
        if msg.get("message"):
            print(f"  [{rule['section_identifier']}] {msg['title']}")
```

## Key Metrics Reference

| Metric | Measures |
|--------|----------|
| `gpu__time_duration.sum` | Kernel duration (nsec) |
| `sm__throughput.avg.pct_of_peak_sustained_elapsed` | Compute (SM) utilization % |
| `dram__throughput.avg.pct_of_peak_sustained_elapsed` | DRAM bandwidth utilization % |
| `sm__warps_active.avg.pct_of_peak_sustained_active` | Achieved occupancy % |
| `l1tex__throughput.avg.pct_of_peak_sustained_elapsed` | L1 cache utilization % |
| `lts__throughput.avg.pct_of_peak_sustained_elapsed` | L2 cache utilization % |
| `lts__t_sector_hit_rate.pct` | L2 cache hit rate % |
| `dram__bytes.sum` | Total DRAM bytes transferred |
| `launch__registers_per_thread` | Registers per thread |
| `launch__grid_size` / `launch__block_size` | Grid/block dimensions |
| `sm__inst_executed_pipe_tensor.sum` | Tensor core instructions |

Use `--list-metrics` with the analysis script to discover all metrics in a report.
