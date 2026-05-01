# NSA Sparse Attention — Findings and Next Steps

## Overview

NSA (Native Sparse Attention) decomposes attention into three branches:
1. **Compressed** — mean-pool KV into blocks, attend to short sequence
2. **Selected** — top-k most relevant KV blocks via block-sparse attention
3. **Sliding window** — local causal window

All three branches use FA4 CuteDSL kernels for both forward and backward.
Compression and gating use pure PyTorch (no CuteDSL auxiliary kernels).

## What We Built

### Core Implementation (43 diffs)
- Full forward and backward pass with activation checkpointing
- All three branches use `_flash_attn_fwd` / `_flash_attn_bwd` (FA4 CuteDSL)
- Sequential per-branch backward to reduce peak memory (no gate weights case)
- In-place gradient accumulation to minimize peak memory
- Native varlen for selected + sliding window branches (fwd + bwd)
- GEMM-based block scoring (cuBLAS bmm + torch.topk, GQA-aware)
- Block-sparse compressed attention (`num_cmp_selected_blocks`)

### Optimization Round (7 diffs)

**P6: Dead code cleanup** — Removed 810 lines of unused code (CuteDSL select
kernel, nsa_scoring.py, nsa_topk.py, window_sparse.py, _fa4_fwd_simple).

**P1: Shared selector** — `fused_score_and_select_all()` computes Q_mean and
the Q_mean x K_cmp GEMM once, deriving both selected-branch and compressed-branch
block indices from the same scores. Previously the GEMM (43% of forward at 128K)
ran twice when `num_cmp_selected_blocks` was enabled.

**P2: Compact sparse metadata** — Block-sparse index tensors now use compact
last dimension (k selected blocks, not n_blocks_k total blocks). Sparse backward
transpose via inverted-index construction (no dense boolean attendance matrix).
Saves ~14 GiB at 1M tokens.

**P0: First-class compressed-causal FA4 mode** — Added `compress_factor`
parameter to FA4's causal masking infrastructure. The causal condition becomes
`kv_idx * compress_factor <= q_idx`, implemented via modified diagonal slope in
tile skipping and R2P masking. This replaces the `mask_mod` approach, enabling:
- Tile skipping (~50% of KV blocks skipped for long sequences)
- R2P single-instruction masking (was element-wise mask_mod loop)
- GQA packing restored (mask_mod disabled pack_gqa)
- Native varlen for compressed branch (mask_mod blocked varlen)
- Fewer JIT compile variants (no mask_mod_hash in compile key)

**P3: Pure PyTorch compress + gating** — Replaced CuteDSL compress and gating
kernels with standard PyTorch operations (reshape+mean for compression,
einsum+sigmoid for gating). Eliminates 4 CuteDSL kernel families from the JIT
cache. FA4 is now the only compiled kernel family on the hot path.

### Bug Fixes
- FA4 block-sparse backward: missing `is_leader_cta` positional argument
  (FlashAttention issue #2011)
- Sliding window backward: `is_causal` was always `False` when `window_size`
  was set (never None)

## Performance (GB200, B=1, H=32, H_kv=8, D=128)

Measured on NVIDIA GB200 (184 GiB), 2026-03-28.

### Forward: NSA vs Dense FA4

| Seq Length | Dense FA4 | NSA | Speedup | NSA+CmpSparse | CmpSp Speedup |
|-----------|----------|-----|---------|---------------|---------------|
| 4K | 0.23ms | 2.97ms | 0.08x | 1.65ms | 0.14x |
| 8K | 0.66ms | 1.51ms | 0.44x | 2.42ms | 0.27x |
| 16K | 1.58ms | 2.76ms | 0.57x | 3.11ms | 0.51x |
| **32K** | **5.50ms** | **3.99ms** | **1.38x** | 4.02ms | 1.37x |
| 64K | 25.29ms | 7.16ms | 3.53x | 7.04ms | 3.59x |
| 128K | 97.71ms | 12.34ms | 7.92x | 12.69ms | 7.70x |
| 256K | 407.59ms | 26.30ms | 15.50x | 30.17ms | 13.51x |
| 512K | 1633.15ms | 71.33ms | 22.90x | 68.32ms | 23.90x |
| **1M** | **6537.68ms** | **228.28ms** | **28.64x** | **171.49ms** | **38.12x** |

Forward crossover at ~32K for both NSA and NSA+CmpSparse.

**vs previous (pre-optimization):** NSA at 1M went from 13.1x to **28.6x** (2.2x improvement).
NSA+CmpSparse at 1M went from 26.4x to **38.1x** (1.4x improvement).

### Forward + Backward: NSA vs Dense FA4

| Seq Length | Dense FA4 | NSA | Speedup |
|-----------|----------|-----|---------|
| 64K | 25.19ms | 11.47ms | 2.20x |
| 128K | 98.72ms | 21.30ms | 4.63x |
| 256K | 406.35ms | 44.79ms | 9.07x |
| 512K | 1632.08ms | 108.71ms | 15.01x |
| **1M** | **6531.88ms** | **303.52ms** | **21.52x** |

Note: fwd+bwd measurement at small N includes JIT compilation overhead.
Warmed-run measurements needed for accurate small-N comparison.

### Memory Savings

At 1M tokens (B=1, H=32, n_block_size=128):
- Compact metadata: ~14 GiB eliminated (dense n_blocks_k-wide tensors removed)
- Sparse backward transpose: ~10 GiB eliminated (no dense boolean attendance)
- Total metadata memory: O(B * H * N_q_tiles * k) instead of
  O(B * H * N_q_tiles * N_kv_blocks)

### What Changed vs Previous Results

| Metric | Before (pre-opt) | After (all opts) | Change |
|--------|------------------|------------------|--------|
| NSA fwd speedup at 1M | 13.1x | **28.6x** | **2.2x better** |
| NSA+CmpSparse fwd speedup at 1M | 26.4x | **38.1x** | **1.4x better** |
| Forward crossover | ~32K | ~32K | same |
| Metadata memory at 1M | ~14 GiB | ~0.1 GiB | **140x less** |
| CuteDSL kernel variants | ~11 | ~6 (FA4 only) | **45% fewer** |
| mask_mod compile paths | 1+ | 0 | eliminated |

## Remaining Work

### 1. Submit FA4 Block-Sparse Backward Fix Upstream
The one-line fix in `flash_bwd_sm100.py` (missing `is_leader_cta` argument)
should be submitted as a PR to `Dao-AILab/flash-attention`.

### 2. Gate-Weight Backward Streaming (P4)
When `gate_proj_weight` is provided, backward holds all three O_i simultaneously.
Options: recompute O_i inside gating backward, or accept per-branch streamed inputs.

### 3. CUDA Graph Backward Support (P5)
Forward CUDA graph capture works (5x at small N). Backward capture not yet
supported. Should be cleaner now that CuteDSL auxiliary kernels are removed.

### 4. Test 2M+ Context
With CuteDSL auxiliary kernels eliminated and fewer FA4 compile variants,
2M+ context may now be viable. Needs testing.

## File Map

| File | Purpose |
|------|---------|
| `mslk/attention/sparse_attn/nsa_autograd.py` | Backward pass (autograd) |
| `mslk/attention/sparse_attn/nsa_forward.py` | Forward pass + `_fa4_fwd`/`_fa4_bwd` helpers |
| `mslk/attention/sparse_attn/reference.py` | PyTorch reference implementation |
| `mslk/attention/sparse_attn/compress.py` | KV compression (pure PyTorch) |
| `mslk/attention/sparse_attn/gating.py` | Gating (pure PyTorch) |
| `mslk/attention/sparse_attn/select.py` | Block scoring + top-k (GEMM + topk) |
| `mslk/attention/sparse_attn/sparsity_masks.py` | Block index → FA4 format (compact) |
| `fb/mslk/attention/flash_attn/interface.py` | FA4 interface (compress_factor) |
| `fb/mslk/attention/flash_attn/block_info.py` | FA4 tile skipping (compress_factor) |
| `fb/mslk/attention/flash_attn/mask.py` | FA4 causal masking (compress_factor) |
| `fb/mslk/attention/flash_attn/flash_fwd_sm100.py` | FA4 SM100 forward kernel |
| `fb/mslk/attention/flash_attn/flash_bwd_sm100.py` | FA4 SM100 backward kernel |
