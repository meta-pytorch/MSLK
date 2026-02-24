# NCU Python Report Interface — API Reference

## Module Location

```
/opt/nvidia/nsight-compute/<version>/extras/python/ncu_report.py
/opt/nvidia/nsight-compute/<version>/extras/python/_ncu_report.so
```

Available versions (use newest): `ls /opt/nvidia/nsight-compute/`

## Setup

```python
import sys
sys.path.insert(0, "/opt/nvidia/nsight-compute/2025.3.1/extras/python")
import ncu_report
```

Or set env: `export PYTHONPATH="/opt/nvidia/nsight-compute/2025.3.1/extras/python:$PYTHONPATH"`

## Core API

### Loading Reports

```python
ctx = ncu_report.load_report("file.ncu-rep")  # returns IContext
```

### IContext (top-level report object)

```python
ctx.num_ranges()           # number of ranges (usually 1)
ctx.range_by_idx(0)        # get IRange by index
len(ctx)                   # same as num_ranges()
for rng in ctx: ...        # iterate ranges
```

### IRange (stream of kernel executions)

```python
rng.num_actions()          # number of profiled kernels
rng.action_by_idx(0)       # get IAction by index
len(rng)                   # same as num_actions()
for action in rng: ...     # iterate actions
rng.actions_by_nvtx(["RangeName"], [])  # filter by NVTX state
```

### IAction (single kernel profile result)

```python
action.name()              # kernel name (default: function signature)
action.name(ncu_report.IAction.NameBase_DEMANGLED)   # demangled name
action.name(ncu_report.IAction.NameBase_MANGLED)     # mangled name
action.workload_type()     # 0=KERNEL, see WorkloadType_* constants

# Metrics
action.metric_names()      # tuple of all metric name strings
action.metric_by_name("gpu__time_duration.sum")  # IMetric or None
action["gpu__time_duration.sum"]                 # IMetric (raises KeyError)
len(action)                # number of metrics
for name in action: ...    # iterate metric names

# Source correlation
action.source_files()      # dict: filename -> content
action.source_info(addr)   # ISourceInfo for address
action.sass_by_pc(addr)    # SASS at address
action.ptx_by_pc(addr)     # PTX at address

# Rule analysis
action.rule_results()           # tuple of IRuleResult
action.rule_results_as_dicts()  # list of dicts with full rule data

# NVTX
action.nvtx_state()        # INvtxState or None
```

### IMetric (single metric value)

```python
m.name()                   # metric name string
m.value()                  # auto-typed value (str|int|float|None)
m.as_double()              # as float (0.0 if not convertible)
m.as_uint64()              # as int (0 if not convertible)
m.as_string()              # as str (None if not convertible)
m.unit()                   # unit string (e.g. "nsecond", "%")
m.description()            # short description

# Type info
m.metric_type()            # MetricType_COUNTER|RATIO|THROUGHPUT|OTHER
m.metric_subtype()         # MetricSubtype_* or None
m.rollup_operation()       # RollupOperation_AVG|MAX|MIN|SUM or None
m.kind()                   # ValueKind_STRING|FLOAT|DOUBLE|UINT32|UINT64|...

# Instanced metrics (per-SM, per-warp, etc.)
m.num_instances()          # number of instance values
m.value(idx)               # value at instance index
m.has_correlation_ids()    # bool
m.correlation_ids()        # IMetric of correlation IDs
```

### IRuleResult (bottleneck analysis)

```python
r.name()                   # rule name
r.rule_identifier()        # rule ID
r.section_identifier()     # section ID
r.has_rule_message()       # bool
r.rule_message()           # dict: title, message, type
r.has_speedup_estimation() # bool
r.speedup_estimation()     # dict: type (SpeedupType), speedup (float)
r.focus_metrics()          # list of dicts: name, value, severity, info
r.result_tables()          # list of dicts: title, headers, data
```

## Common Metric Names

### Timing
- `gpu__time_duration.sum` — total kernel duration (nsec)
- `gpu__time_duration.avg` — average duration across invocations

### Compute Throughput
- `sm__throughput.avg.pct_of_peak_sustained_elapsed` — SM utilization %
- `sm__warps_active.avg.pct_of_peak_sustained_active` — achieved occupancy %

### Memory Throughput
- `dram__throughput.avg.pct_of_peak_sustained_elapsed` — DRAM utilization %
- `dram__bytes.sum` — total DRAM bytes transferred
- `dram__bytes_read.sum` / `dram__bytes_write.sum` — read/write split

### Cache
- `l1tex__throughput.avg.pct_of_peak_sustained_elapsed` — L1 utilization %
- `lts__throughput.avg.pct_of_peak_sustained_elapsed` — L2 utilization %
- `lts__t_sector_hit_rate.pct` — L2 hit rate %

### Launch Configuration
- `launch__grid_size` — total grid size
- `launch__block_size` — threads per block
- `launch__registers_per_thread` — registers per thread
- `launch__occupancy_limit_registers` — occupancy limited by registers
- `launch__waves_per_multiprocessor` — waves per SM

### Instructions
- `smsp__inst_executed.sum` — total instructions executed
- `sm__inst_executed_pipe_tensor.sum` — tensor core instructions
- `sm__inst_executed_pipe_fma.sum` — FMA pipe instructions
