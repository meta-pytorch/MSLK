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

Measured on NVIDIA GB200, 2026-03-26.

### Forward: NSA vs Dense FA4

| Seq Length | Dense FA4 | NSA | Speedup |
|-----------|----------|-----|---------|
| 1K | 0.10ms | 1.46ms | 0.07x |
| 2K | 0.19ms | 2.34ms | 0.08x |
| 4K | 0.27ms | 2.27ms | 0.12x |
| 8K | 0.53ms | 2.17ms | 0.25x |
| 16K | 1.53ms | 3.87ms | 0.40x |
| **32K** | **5.16ms** | **5.72ms** | **0.90x** |
| 64K | 24.06ms | 10.35ms | 2.33x |
| 128K | 96.53ms | 24.18ms | 3.99x |
| 256K | 403.51ms | 65.11ms | 6.20x |
| **512K** | **1617.79ms** | **194.81ms** | **8.30x** |

Forward crossover at ~32K tokens.

### Forward + Backward: NSA vs Dense FA4

| Seq Length | Dense FA4 | NSA | Speedup |
|-----------|----------|-----|---------|
| 1K | 0.57ms | 3.33ms | 0.17x |
| 2K | 0.55ms | 3.83ms | 0.14x |
| 4K | 0.76ms | 3.62ms | 0.21x |
| 8K | 1.63ms | 4.51ms | 0.36x |
| 16K | 5.06ms | 7.45ms | 0.68x |
| **32K** | **21.35ms** | **13.83ms** | **1.54x** |
| 64K | 84.15ms | 29.00ms | 2.90x |
| 128K | 338.68ms | 68.78ms | 4.92x |
| 256K | 1356.51ms | 182.45ms | 7.44x |
| **512K** | **5423.31ms** | **833.78ms** | **6.50x** |

Fwd+bwd crossover at ~32K tokens.

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
The backward pass now supports native varlen (variable-length packed sequences)
without padding Q, K, V to max_seqlen:
- Selected and sliding window branches use native FA4 varlen in both fwd and bwd
- Compressed branch uses padded Q for mask_mod compatibility (FA4 limitation)
- Compress/select read from 3D varlen input directly
- Only the compressed KV output (max_seqlen/64) remains padded as an implementation detail

Remaining work:
- Add varlen + mask_mod support to FA4 SM100 backward (commits 6-7)
- This would let the compressed branch also use native varlen

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
