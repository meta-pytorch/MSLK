---
name: mslk
description: Kernel development and testing in the MSLK repository. Use when implementing new kernels, comparing existing kernels, or trying to benchmark and understand performance or correctness.
---

# MSLK Development Guide

## High Level Description

MSLK is a repository of high performance kernels. It contains infrastructure to expose its kernels through a consistent API across hardware backends, and extensive testing and benchmarking to measure correctness and performance. Kernels may be implemented in many different languages, you'll find CUTLASS, CuteDSL, Triton, and more. While MSLK welcomes all types of kernels, we generally prefer CuteDSL when authoring new kernels as it offers a great balance of low level control and developer friendly compilation.

## Project Organization

MSLK is organized by kernel class and language. For example, python kernel are located in the `mslk/mslk` module and further organized by kernel type. GEMM kernels can be found in `mslk/mslk/gemm`, convolution kernels in `mslk/mslk/conv`, etc. For C++ implementations, the same structure is mirrored in `mslk/csrc/...` and the associated includes are in `include/mslk/...`. Each class of kernels has its own associated testing and benchmarking suite. Test suites are located in `test/...` and benchmarks are located in `bench/...`. Currently GEMM kernels are the most fully supported, so it is helpful to reference their tests and benchmarks as a gold standard, in particular `bench/gemm/gemm_ops.py`, which provides helpful wrappers around all the various GEMM kernels available in MSLK and `bench/gemm/gemm_bench.py`, which executes the benchmarking script.

## Running Tests and Benchmarks

MSLK uses Buck for its build system, which is a flavor of the bazel build system. Thus, all executable files must have an associated `BUCK` file that describes their dependencies. You will find these BUCK files in every folder of MSLK and they must be modified or extended when adding new kernel or changing dependencies. A benefit of the buck build system is that reproducing results is very straight-forward! For example, to run a GEMM benchmark sweep on a GB200 GPU, we could use a command like this:

```bash
buck2 run @//mode/{opt,inplace} -c fbcode.enable_gpu_sections=true -c fbcode.nvcc_arch=b200a -c fbcode.platform010_cuda_version=12.8 //mslk/bench/gemm:gemm_bench -- --kernels=CutlassNVFP4TorchGrouped,CutlassNVFP4GroupwiseGrouped --M=256 --N=1600 --K=1600 --groups=8 --grouped
```

It is important to target the appropriate GPU with the nvcc_arch argument in buck. For GB200 or B200 we use `b200a`, for H100 we use `h100a`, and for GB300 we use `b300a`. AMD kernels are also often supported and would use a configuration like this (assuming MI300x target): `buck2 run @//mode/{opt,amd-gpu,inplace} -c fbcode.enable_gpu_sections=true -c fbcode.triton_backend=amd -c fbcode.rocm_arch=mi300 ...`.

We similarly run tests using buck:
```bash
buck2 run @//mode/{opt,inplace} -c fbcode.enable_gpu_sections=true -c fbcode.nvcc_arch=b200a -c fbcode.platform010_cuda_version=12.8 //mslk/test/gemm:gemm_test -- mslk.test.gemm.gemm_test.BF16Tests
```

## Developing in MSLK

When writing a new kernel or improving an existsing one, we recommend starting with a strong baseline. Find the most appropriate kernel for comparison and run the benchmarking script to study its performance across shapes of interest. For example, if we are working on a new GEMM kernel, we would compare to `TorchBF16` in `gemm_ops.py`. Next, we can implement our new kernel, add tests and a benchmarking wrapper, then iterate on the benchmarking and performance debugging until we reach the desired performance outcome.

One of the most common types of optimization we do in MSLK is kernel tuning. Kernels can be dispatched with varying block sizes and warp settings and choosing these hyperparameters correctly is essential to good performance for a specific workload. We often have to enumerate shapes of interest, typically those relevant to a production model, and iterate on heuristic tuning until performance across all shapes is compelling. For an example of this heuristic tuning, you can review `get_kernel_via_heuristic` in `csrc/gemm/cutlass/f8f8bf16_rowwise.cu`. This same philosophy of tuning applies across all kernel languages and must always be done.
