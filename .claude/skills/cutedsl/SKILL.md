---
name: cutedsl
description: CuteDSL kernel development for CUDA. Use when working with CuteDSL, CuTe layouts, TMEM, SMEM, blockscaled GEMM, or flash attention kernels.
---

# CuteDSL Development Guide

## Version Note

The third-party CuteDSL package lives at `fbsource/third-party/pypi/nvidia-cutlass-dsl/<VERSION>/`.
The version changes over time — check `fbsource/third-party/pypi/nvidia-cutlass-dsl/VERSION.bzl` for the current default.
The library source root is: `third-party/pypi/nvidia-cutlass-dsl/<VERSION>/_whl_common_files/nvidia_cutlass_dsl/python_packages/cutlass/`

## Reference Notebooks

Read these notebooks in `fbsource/fbcode/peis/cutlass/examples/python/CuTeDSL/notebooks/` (notebooks are not shipped in the third-party package):

- `print.ipynb` - Debug printing tensors and layouts
- `tensor.ipynb` - Tensor creation and manipulation
- `cute_layout_algebra.ipynb` - Layout composition, slicing, tiling
- `tour_to_sol_gemm.ipynb` - Complete GEMM implementation

## Key Patterns

### Debug Printing
```python
with cute.arch.elect_one():
    cute.printf("value: %d", some_value)
    cute.printf("layout: {}", tensor.layout)  # {} for CuTe types
```

### Tensor Slicing
```python
# None = take all, integer = select index
sliced = tensor[(None, None, idx)]
```

### Loop Constructs
```python
cutlass.range(n, unroll_full=True)  # Required for SSA threading
cutlass.range_constexpr(n)           # Compile-time loop counter
```

### Barriers
```python
cute.arch.barrier(barrier_id=id, number_of_threads=count)  # arrive + wait
cute.arch.barrier_arrive(barrier_id=id, number_of_threads=count)  # arrive only
```

## TMA and Synchronization

TMA (Tensor Memory Accelerator, Hopper+) asynchronously copies tiles between GMEM and SMEM via descriptors. It enables single-threaded issuance, automatic OOB predication, and warp specialization. TMA loads use mbarriers for completion tracking; TMA stores require a proxy fence before issuance.

For the complete TMA workflow (descriptor creation, partitioning, pipelines, multicast, store-reduce, and code references), see [TMA Guide](references/tma-guide.md).

## Detailed Guides

- [TMA (Tensor Memory Accelerator) Guide](references/tma-guide.md) — descriptor creation, partitioning flow, synchronization, multicast, store-reduce, code references
- [Debugging SMEM Values](references/debugging-smem-values.md) — printing SMEM as hex, calculating stage strides, diagnosing zero-data issues

## Additional Resources

Search `fbsource/third-party/pypi/nvidia-cutlass-dsl/<VERSION>/` for CuteDSL syntax issues (check `VERSION.bzl` for the current default).
