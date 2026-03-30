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

## Performance (GB200, B=1, H=32, H_kv=8, D=128)

### Forward: NSA vs Dense FA4

| Seq Length | Dense FA4 | NSA | Speedup |
|-----------|----------|-----|---------|
| 4K | 0.26ms | 2.48ms | 0.11x |
| 8K | 0.53ms | 2.57ms | 0.21x |
| 16K | 1.54ms | 4.09ms | 0.38x |
| 32K | 5.47ms | 6.09ms | 0.90x |
| **64K** | **24.02ms** | **10.31ms** | **2.33x** |
| 128K | 100.17ms | 23.89ms | 4.19x |
| 256K | 400.45ms | 62.83ms | 6.37x |
| 512K | 1589.36ms | 187.37ms | 8.48x |
| **1M** | **6356.49ms** | **623.98ms** | **10.19x** |

Crossover at ~32K tokens.

### Backward (fwd+bwd)

| Seq Length | fwd+bwd |
|-----------|---------|
| 1K | 3.60ms |
| 8K | 3.69ms |
| 32K | 13.04ms |
| 64K | 28.50ms |
| 128K | 67.64ms |
| 256K | 178.69ms |
| 512K | 537.54ms |
| **1M** | **1787.53ms** |
| 2M+ | OOM (CuTe DSL compilation cache) |

### Forward Component Breakdown (profiled on GB200)

At N=16K (where NSA is ~2.7x slower than dense):
- Mask construction: 20% (PyTorch op launch overhead)
- Block selection: 18%
- Gating: 17%
- 3x FA4 attention: 34%
- Compression: 12%

At N=128K (where NSA is 4.2x faster than dense):
- FA4 compressed attention: 33%
- Gating: 22%
- FA4 selected attention: 18%
- Block selection: 15%
- FA4 sliding window: 7%
- Mask construction: 4%
- Compression: 2%

## 2M/3M Context Limit

On this GB200 (184 GiB), CuTe DSL JIT compilation cache consumes ~170 GiB,
leaving only ~14 GiB for computation. Both forward and backward OOM at 2M.
The forward was previously benchmarked at 2M/3M on a machine with more
GPU memory. The backward pass itself has no architectural limit beyond
the forward's — all three branches use FA4 tiled kernels with O(1) memory
per tile.

## Next Steps

### 1. Test on Larger GPU (immediate)
Run on a machine with more GPU memory (or fewer CuTe DSL kernel variants
compiled) to verify 2M/3M backward works. The backward should scale
identically to the forward since both use the same FA4 kernels.

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
