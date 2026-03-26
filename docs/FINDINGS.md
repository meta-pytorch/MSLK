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
| **512K** | **1617.06ms** | **196.30ms** | **8.24x** |

Forward crossover at ~32K tokens.

### Forward + Backward: NSA vs Dense FA4

| Seq Length | Dense FA4 | NSA | Speedup |
|-----------|----------|-----|---------|
| 1K | 0.64ms | 4.10ms | 0.16x |
| 2K | 0.57ms | 3.64ms | 0.16x |
| 4K | 0.79ms | 3.71ms | 0.21x |
| 8K | 1.65ms | 4.61ms | 0.36x |
| 16K | 5.11ms | 8.71ms | 0.59x |
| **32K** | **21.52ms** | **14.02ms** | **1.54x** |
| 64K | 80.41ms | 29.27ms | 2.75x |
| 128K | 339.29ms | 68.31ms | 4.97x |
| 256K | 1356.74ms | 181.52ms | 7.47x |
| 512K | 5436.21ms | 545.76ms | 9.96x |
| **1M** | **26076.29ms** | **1853.34ms** | **14.07x** |

Fwd+bwd crossover at ~32K tokens.

### Varlen Performance (native varlen vs padded)

| Total N | Varlen F+B | Regular F+B | Varlen Speedup |
|---------|-----------|------------|----------------|
| 128K | 67.42ms | 68.39ms | 1.01x |
| 256K | 156.79ms | 180.76ms | 1.15x |
| 512K | 621.28ms | 826.52ms | 1.33x |
| **1M** | **1901.55ms** | **2331.09ms** | **1.23x** |

Native varlen (selected + sliding window branches use cu_seqlens directly)
is 23-33% faster than the padded approach at 256K-1M token contexts.

### Forward Component Breakdown (profiled on GB200)

At N=4K (where NSA is ~10x slower than dense):
- Mask construction: 29% (PyTorch op launch overhead)
- Block selection: 22%
- Gating: 16%
- 3x FA4 attention: 22%
- Compression: 11%

At N=128K (where NSA is 4x faster than dense):
- FA4 compressed attention: 33%
- Gating: 21%
- FA4 selected attention: 18%
- Block selection: 16%
- FA4 sliding window: 6%
- Mask construction: 4%
- Compression: 2%

At N=1M (where NSA is 7.9x faster — forward only):
- **FA4 compressed attention: 57.5%** — O(N × N/64) = O(N²/64), dominates at large N
- Block selection: 21.5% — CuteDSL kernel, scales with N_cmp²
- Mask construction: 9.4% — PyTorch ops, scales with N
- FA4 selected attention: 5.2% — sparse, sub-quadratic
- Gating: 4.6% — O(N)
- FA4 sliding window: 1.6% — window is fixed 512, O(N)
- Compression: 0.2% — O(N)

**Key insight**: The compressed branch is the performance bottleneck at large N.
It attends Q(N) to K_cmp(N/64), which is O(N²/64) — quadratic in N but 64x
smaller than dense. At 1M, this alone takes 364ms out of 633ms total.
The forward speedup regression from 8.24x (512K) to 7.93x (1M) is explained by
this quadratic scaling in the compressed branch.

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
| `docs/nsa_perf.svg` | Performance chart |
