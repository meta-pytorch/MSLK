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

Measured on NVIDIA GB200 (184 GiB), 2026-03-27.

### Forward: NSA vs Dense FA4

| Seq Length | Dense FA4 | NSA | Speedup | NSA+CmpSparse | CmpSp Speedup |
|-----------|----------|-----|---------|---------------|---------------|
| 4K | 0.24ms | 2.99ms | 0.08x | 3.98ms | 0.06x |
| 8K | 0.60ms | 1.97ms | 0.30x | 2.88ms | 0.21x |
| 16K | 1.62ms | 3.20ms | 0.51x | 6.01ms | 0.27x |
| **32K** | **5.22ms** | **4.44ms** | **1.18x** | 8.12ms | 0.64x |
| 64K | 24.98ms | 8.40ms | 2.97x | 14.31ms | 1.75x |
| 128K | 98.59ms | 17.71ms | 5.57x | 27.27ms | 3.61x |
| 256K | 407.38ms | 43.58ms | 9.35x | 49.07ms | 8.30x |
| 512K | 1633.69ms | 138.47ms | 11.80x | 102.91ms | 15.87x |
| **1M** | **6472.56ms** | **492.89ms** | **13.13x** | **244.98ms** | **26.42x** |

Forward crossover at ~32K (NSA) or ~64K (NSA+CmpSparse with k_cmp=16).

NSA+CmpSparse uses `num_cmp_selected_blocks=16`: selects top-16 FA4 blocks
of compressed KV per Q tile (out of N_cmp/128 total blocks), making the
compressed branch sub-quadratic. Beneficial at 512K+ where the compressed
branch dominates.

### Forward + Backward: NSA vs Dense FA4

| Seq Length | Dense FA4 | NSA | Speedup |
|-----------|----------|-----|---------|
| 1K | 0.56ms | 3.70ms | 0.15x |
| 2K | 0.57ms | 3.62ms | 0.16x |
| 4K | 0.78ms | 3.51ms | 0.22x |
| 8K | 1.65ms | 5.32ms | 0.31x |
| 16K | 5.11ms | 8.02ms | 0.64x |
| **32K** | **21.02ms** | **14.17ms** | **1.48x** |
| 64K | 84.46ms | 29.46ms | 2.87x |
| 128K | 340.14ms | 68.41ms | 4.97x |
| 256K | 1339.51ms | 179.66ms | 7.46x |
| 512K | 5374.25ms | 540.77ms | 9.94x |
| **1M** | **27270.16ms** | **2432.94ms** | **11.21x** |

Fwd+bwd crossover at ~32K tokens.

### Varlen Performance (native varlen vs padded)

| Total N | Varlen F+B | Regular F+B | Varlen Speedup |
|---------|-----------|------------|----------------|
| 128K | 67.86ms | 69.22ms | 1.02x |
| 256K | 156.67ms | 181.19ms | 1.16x |
| 512K | 411.58ms | 541.68ms | 1.32x |
| **1M** | **1224.64ms** | **1797.96ms** | **1.47x** |

Native varlen (selected + sliding window branches use cu_seqlens directly)
is 32-47% faster than the padded approach at 256K-1M token contexts.

### Forward Component Breakdown (profiled on GB200, GEMM-based scoring)

At N=4K (where NSA is ~12x slower than dense):
- Block selection (GEMM): 28%
- Mask construction: 26%
- Gating: 15%
- 3x FA4 attention: 20%
- Compression: 11%

At N=128K (where NSA is 5.6x faster than dense):
- Block selection (GEMM): 43%
- FA4 compressed attention: 22%
- Gating: 14%
- FA4 selected attention: 12%
- FA4 sliding window: 4%
- Mask construction: 2%
- Compression: 2%

At N=262K:
- Block selection (GEMM): 39%
- FA4 compressed attention: 34%
- FA4 selected attention: 11%
- Gating: estimated 10%
- FA4 sliding window: 3%
- Mask construction: 1%

**Key insight**: With GEMM-based scoring, block selection is now the dominant
cost at medium N (43% at 128K). At large N, the compressed branch (still
quadratic without `num_cmp_selected_blocks`) catches up. Using
`num_cmp_selected_blocks` makes the compressed branch sub-quadratic,
leaving block selection as the universal bottleneck.

## 2M/3M Context Limit

On the B200 (178 GiB), CuTe DSL JIT compilation cache consumes ~165 GiB,
leaving only ~13 GiB for computation. NSA OOMs at 2M for both forward and
backward. Dense FA4 can run at 2M (31.6s fwd, 106.8s fwd+bwd) and 3M
(72.1s fwd) since it has a single kernel with lower JIT cache footprint.
The NSA backward pass itself has no architectural limit beyond the
forward's — all three branches use FA4 tiled kernels with O(1) memory
per tile. The OOM is purely a JIT cache issue.

## What We Optimized

### GEMM-Based Scoring (replacing CuteDSL scalar dot products)
The block selection kernel was replaced from CuteDSL scalar dot products to
cuBLAS GEMM via `torch.matmul`. Uses GQA-aware bmm to avoid expanding K_cmp
from H_kv to H heads. Eliminates CuteDSL JIT compilation for the select kernel.
The Q_mean optimization (mean(Q @ K) = mean(Q) @ K) is preserved.

### Block-Sparse Compressed Attention (sub-quadratic compressed branch)
New `num_cmp_selected_blocks` parameter selects top-k FA4 blocks of compressed
KV per Q tile, making the compressed branch O(N * k_cmp) instead of O(N²/64).
Uses FA4's `mask_mod` + `block_sparse_tensors` simultaneously: block sparsity
determines which blocks to process, mask_mod applies compressed causal masking
within each block. Beneficial at 512K+ contexts.

### Multi-Stream FA4 Branches — Investigated, Not Beneficial
Attempted overlapping the 3 FA4 branches on separate CUDA streams. Result:
stream creation/synchronization overhead (0.1-0.5ms) exceeded any overlap
benefit. Each FA4 call saturates all SMs on GB200, so there's nothing to overlap.

## Next Steps

### 1. Reduce CuTe DSL JIT Cache Footprint
NSA compiles many more kernel variants than dense FA4 (3 attention
branches x fwd/bwd + compress + gating). The select kernel no longer uses
CuteDSL (switched to GEMM). Reducing compiled variants or sharing JIT cache
across branches would unlock 2M+ contexts.

### 2. Submit FA4 Block-Sparse Backward Fix Upstream
The one-line fix in `flash_bwd_sm100.py` (missing `is_leader_cta` argument)
should be submitted as a PR to `Dao-AILab/flash-attention`. Include the
test from `test_fa4_block_sparse_bwd.py` as the regression test.

### 3. Lower the Crossover Point (performance)
NSA breaks even at ~32K. To lower this:

**a. Fuse mask construction into the select output**
The GEMM scoring + topk already produces block indices. Computing the
`BlockSparseTensorsTorch` format inline (integer division + dedup) would
eliminate 6+ separate PyTorch op launches (25% at small N).

**b. Reduce kernel launch count**
NSA launches 7+ GPU kernels per forward (compress, Q_mean, GEMM, topk,
mask, 3x FA4, gating). At small N, kernel launch latency dominates.
CUDA graphs could amortize launch overhead (existing test infrastructure
in `test_cuda_graph_nsa.py`).

**c. Optimize gating kernel**
Gating takes 14-15% across all sizes. The current CuteDSL kernel uses
4 warps/block, 1 row per warp. Increasing rows per block and using
vectorized loads could yield 20-30% improvement.

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
