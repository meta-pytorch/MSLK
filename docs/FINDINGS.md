# NSA Sparse Attention — Findings and Next Steps

## Overview

NSA (Native Sparse Attention) decomposes attention into three branches:
1. **Compressed** — mean-pool KV into blocks, attend to short sequence
2. **Selected** — top-k most relevant KV blocks via block-sparse attention
3. **Sliding window** — local causal window

All three branches now use FA4 CuteDSL kernels for both forward and backward.

## What We Built

### Backward Pass
- All three branches use `_flash_attn_bwd` (FA4 CuteDSL):
  - Compressed: `mask_mod` for block-aware causal masking
  - Selected: `block_sparse_tensors` (transposed for backward)
  - Sliding window: `window_size_left`
- Sequential per-branch processing minimizes peak memory (no gate weights case)
- Activation checkpointing: recomputes FA4 forward per branch during backward

### FA4 Block-Sparse Backward Fix
- **Root cause**: Missing `is_leader_cta` positional argument in
  `flash_bwd_sm100.py` line 1513. The `blocksparse_tensors` value was
  passed into the `is_leader_cta` parameter slot; the actual
  `blocksparse_tensors` parameter fell back to its `None` default.
- **Fix**: One line — add `is_leader_cta,` at the call site.
- **Upstream**: FlashAttention issue #2011. This fix should be submitted
  as a PR to `Dao-AILab/flash-attention`.

### Bug Fix: Sliding Window Backward Causal Masking
- Previous code: `is_causal=causal and window_size is None`
- Since `window_size` defaults to 512 (never None), `is_causal` was always
  `False`, producing incorrect non-causal, non-windowed gradients.
- Fixed by using FA4 backward with explicit `window_size_left`.

## Performance (B200, B=1, H=32, H_kv=8, D=128)

Measured on NVIDIA B200 (178 GiB), devgpu016.snb3, 2026-03-25.

### Forward: NSA vs Dense FA4

| Seq Length | Dense FA4 | NSA | Speedup |
|-----------|----------|-----|---------|
| 1K | 0.09ms | 0.91ms | 0.10x |
| 2K | 0.10ms | 1.02ms | 0.10x |
| 4K | 0.17ms | 1.28ms | 0.13x |
| 8K | 0.48ms | 1.77ms | 0.27x |
| 16K | 1.74ms | 2.84ms | 0.61x |
| **32K** | **6.93ms** | **5.27ms** | **1.32x** |
| 64K | 30.61ms | 11.22ms | 2.73x |
| 128K | 121.16ms | 27.09ms | 4.47x |
| 256K | 485.24ms | 74.81ms | 6.49x |
| 512K | 1939.75ms | 226.35ms | 8.57x |
| **1M** | **7816.42ms** | **772.59ms** | **10.12x** |
| 2M | 31604.78ms | OOM | — |
| 3M | 72051.59ms | OOM | — |

Forward crossover at ~25K tokens (without CUDA graphs), **~4K tokens with CUDA graphs**.

### CUDA Graph Speedup (forward only)

CuTe DSL kernels use `cuda.cuLaunchKernel` and are fully CUDA Graph compatible.
After warmup, the NSA forward can be captured in `torch.cuda.CUDAGraph`:

| N | No Graph | CUDA Graph | Speedup |
|------|----------|-----------|---------|
| 4K | 1.49ms | 0.29ms | 5.08x |
| 8K | 1.27ms | 0.51ms | 2.49x |
| 16K | 2.00ms | 1.01ms | 1.97x |
| 32K | 2.37ms | 2.21ms | 1.07x |
| 64K+ | — | — | <2% |

With CUDA graphs, NSA at 4K (0.29ms) is comparable to dense FA4 (0.26ms).

### Forward + Backward: NSA vs Dense FA4

| Seq Length | Dense FA4 | NSA | Speedup |
|-----------|----------|-----|---------|
| 1K | 0.41ms | 2.42ms | 0.17x |
| 2K | 0.44ms | 2.34ms | 0.19x |
| 4K | 0.68ms | 2.63ms | 0.26x |
| 8K | 1.85ms | 4.28ms | 0.43x |
| 16K | 6.31ms | 7.62ms | 0.83x |
| **32K** | **27.37ms** | **15.45ms** | **1.77x** |
| 64K | 108.85ms | 34.31ms | 3.17x |
| 128K | 418.86ms | 86.48ms | 4.84x |
| 256K | 1669.69ms | 236.09ms | 7.07x |
| 512K | 6644.73ms | 695.13ms | 9.56x |
| **1M** | **26581.04ms** | **2309.82ms** | **11.51x** |
| 2M | 106785.33ms | OOM | — |

Fwd+bwd crossover at ~20K tokens (lower than forward-only because
dense backward is O(N^2) while NSA backward stays sub-quadratic).

### Forward Component Breakdown (profiled on B200)

At N=16K (where NSA is ~1.6x slower than dense):
- Mask construction: 20% (PyTorch op launch overhead)
- Block selection: 18%
- Gating: 17%
- 3x FA4 attention: 34%
- Compression: 12%

At N=128K (where NSA is 4.5x faster than dense):
- FA4 compressed attention: 33%
- Gating: 22%
- FA4 selected attention: 18%
- Block selection: 15%
- FA4 sliding window: 7%
- Mask construction: 4%
- Compression: 2%

## 2M/3M Context Limit

On the B200 (178 GiB), CuTe DSL JIT compilation cache consumes ~165 GiB,
leaving only ~13 GiB for computation. NSA OOMs at 2M for both forward and
backward. Dense FA4 can run at 2M (31.6s fwd, 106.8s fwd+bwd) and 3M
(72.1s fwd) since it has a single kernel with lower JIT cache footprint.
The NSA backward pass itself has no architectural limit beyond the
forward's — all three branches use FA4 tiled kernels with O(1) memory
per tile. The OOM is purely a JIT cache issue.

## Next Steps

### 1. Reduce CuTe DSL JIT Cache Footprint
NSA compiles many more kernel variants than dense FA4 (3 attention
branches x fwd/bwd + compress + select + gating). Reducing the number
of compiled variants or sharing JIT cache across branches would unlock
2M+ contexts.

### 2. Submit FA4 Block-Sparse Backward Fix Upstream
The one-line fix in `flash_bwd_sm100.py` (missing `is_leader_cta` argument)
should be submitted as a PR to `Dao-AILab/flash-attention`. Include the
test from `test_fa4_block_sparse_bwd.py` as the regression test.

### 3. Lower the Crossover Point (performance)
NSA breaks even at ~32K. To lower this:

**a. Fuse mask construction into the select kernel (biggest win: 20% at small N)**
The CuteDSL select kernel (`fused_score_and_select_blocks`) already computes
block indices. It could output the `BlockSparseTensorsTorch` format directly,
eliminating the separate Python mask construction step with its 6+ PyTorch op
launches.

**b. Reduce kernel launch count**
NSA launches 7+ GPU kernels per forward (compress, select, mask, 3x FA4,
gating). At small N, kernel launch latency (~5-10us each) dominates actual
compute. Options:
- Fuse compress + select into one kernel
- Fuse sliding window into the selected branch (both attend to full K,V)
- Use CUDA graphs to amortize launch overhead

**c. Optimize gating kernel**
Gating takes 17-25% across all sizes. The current CuteDSL kernel may have
suboptimal tile sizes for small N. Profile with nsight to identify bottlenecks.

### 4. Varlen Support
Currently all sequences in a batch must have the same length. For training
with packed sequences, need to:
- Adapt compression, scoring, mask construction for variable lengths
- Pass `cu_seqlens_q/k` through to FA4 (already supported by FA4)
- Handle block indices that span sequence boundaries

### 5. Memory Optimization for Gate Weights Case
When `gate_proj_weight` is provided, the backward needs all three O_i
simultaneously for the gating backward. This prevents sequential per-branch
processing. Options:
- Recompute O_i inside the gating backward kernel
- Fuse gating backward with the attention backward

## File Map

| File | Purpose |
|------|---------|
| `mslk/attention/sparse_attn/nsa_autograd.py` | Backward pass (autograd) |
| `mslk/attention/sparse_attn/nsa_forward.py` | Forward pass + `_fa4_bwd` helper |
| `mslk/attention/sparse_attn/reference.py` | PyTorch reference implementation |
| `mslk/attention/sparse_attn/compress.py` | KV compression (fwd + bwd CuteDSL) |
| `mslk/attention/sparse_attn/gating.py` | Gating (fwd + bwd CuteDSL) |
| `mslk/attention/sparse_attn/select.py` | Block scoring + top-k (CuteDSL) |
| `mslk/attention/sparse_attn/sparsity_masks.py` | Block index → FA4 format conversion |
| `fb/mslk/attention/flash_attn/flash_bwd_sm100.py` | FA4 SM100 backward kernel (fixed) |
| `fb/mslk/attention/flash_attn/interface.py` | FA4 `_flash_attn_bwd` interface |
| `test/attention/test_sparse_attn_nsa_backward.py` | 25 backward tests |
| `test/attention/test_fa4_block_sparse_bwd.py` | FA4 block-sparse backward test |
| `test/attention/profile_nsa_forward.py` | Forward component profiler |
| `test/attention/bench_nsa_backward.py` | Backward benchmark (1K-3M) |
| `test/attention/bench_nsa_vs_dense.py` | NSA vs dense FA4 comparison |
| `docs/gen_nsa_perf_chart.py` | Performance chart generator |
| `docs/nsa_bwd_perf.svg` | Performance chart |
