# NSA Sparse Attention Optimization Review Plan

## Summary

This plan targets the NSA implementation in `mslk/attention/sparse_attn/`
with Blackwell-first prioritization.

The review is organized around three goals:

1. Make NSA materially faster than dense FA4 at large context.
2. Increase the maximum supported context within a fixed memory budget.
3. Lower the crossover point where NSA beats dense FA4 in forward and backward.

The main conclusion is that the next bottlenecks are orchestration costs around
FA4 rather than the asymptotically cheap sparse compute itself: `mask_mod`
forces separate compile paths and disables GQA packing, the two selector
pipelines duplicate the dominant GEMM scoring work, dense metadata wastes GiBs
at 1M+, and auxiliary CuteDSL kernels inflate the JIT cache.

## What We've Done (43 diffs)

### Infrastructure

- Moved sparse_attn from `fb/` to `mslk/` and scaled to 1M+ sequences.
- CuteDSL fused kernels for KV compression, block selection, and gating
  (forward and backward).
- Fixed int32 overflow in CuteDSL kernels for N >= 2M.

### Backward pass

- Full backward pass with activation checkpointing. All three branches
  (compressed, selected, sliding window) use FA4 CuteDSL backward.
- Sequential per-branch backward to reduce peak memory (no-gate-weight case).
- In-place gradient accumulation to minimize peak memory.
- Fixed FA4 block-sparse backward (missing `is_leader_cta` argument in
  `flash_bwd_sm100.py`). Upstream: FlashAttention issue #2011.
- Fixed sliding window backward causal masking bug (`is_causal` was always
  `False` when `window_size` was set).

### Varlen support

- Native varlen for selected + sliding window branches (fwd + bwd) via FA4
  varlen calls with `cu_seqlens`.
- Compressed branch uses padded 4D (FA4 doesn't support varlen + mask_mod).
- Compress/select read from 3D varlen input directly.
- 32-47% faster than padding everything at 256K-1M.

### Scoring optimization

- Replaced CuteDSL scalar dot-product scoring with GEMM-based scoring
  (`torch.bmm` + `torch.topk`). Eliminates CuteDSL JIT compilation for the
  select kernel. GQA-aware bmm avoids expanding K_cmp from H_kv to H heads.

### Block-sparse compressed attention

- New `num_cmp_selected_blocks` parameter selects top-k FA4 blocks of
  compressed KV per Q tile, making the compressed branch O(N * k_cmp) instead
  of O(N^2/64). Uses FA4 `mask_mod` + `block_sparse_tensors` simultaneously.
  Beneficial at 512K+ where the compressed branch dominates.

### Mask construction

- 14x faster mask construction at 1M tokens. Now 1-2% of forward time at
  128K+ (was ~25% at small N before optimization).

### CUDA graphs

- Forward CUDA graph capture works (5x speedup at small N).
- Backward capture not yet supported.

### Investigated and rejected

- Multi-stream FA4 branches: stream creation/sync overhead exceeded any overlap
  benefit. Each FA4 call saturates all SMs on GB200.

## Current Performance (GB200, B=1 H=32 H_kv=8 D=128)

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

### Forward + Backward: NSA vs Dense FA4

| Seq Length | Dense FA4 | NSA | Speedup |
|-----------|----------|-----|---------|
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

### Forward Component Breakdown

At N=128K (5.6x faster than dense):
- Block selection (GEMM): **43%**
- FA4 compressed attention: 22%
- Gating: 14%
- FA4 selected attention: 12%
- FA4 sliding window: 4%
- Mask construction: 2%
- Compression: 2%

At N=262K:
- Block selection (GEMM): **39%**
- FA4 compressed attention: **34%**
- FA4 selected attention: 11%
- Gating: ~10%
- FA4 sliding window: 3%
- Mask construction: 1%

### 2M Context Limit

NSA OOMs at 2M. CuTe DSL JIT cache consumes ~165 GiB of the B200's 178 GiB,
leaving only ~13 GiB for computation. Dense FA4 handles 2M and 3M because it
has a single kernel with lower JIT cache footprint.

## Remaining Work — Prioritized

### P0. First-class compressed-causal FA4 mode

The compressed branch currently uses `_make_compressed_causal_mask()` as a
`mask_mod` callback, which has cascading costs:
- Forces a separate FA4 compile path (inflating JIT cache).
- Disables `pack_gqa` (FA4 sets `pack_gqa = False` when `mask_mod` is present).
- Blocks native varlen for the compressed branch (FA4 doesn't support
  varlen + mask_mod, forcing padded 4D).
- The compressed branch is 34% of forward time at 262K (without
  `num_cmp_selected_blocks`) and still significant with it.

Solving `mask_mod` and JIT cache as one problem is higher impact than treating
them separately:

Implementation:
- Add an FA4 forward/backward mode for compressed causal attention where causal
  validity is `kv_block_start <= q_pos`, without using `mask_mod`.
- Keep the NSA public API unchanged; the internal FA4 interface gains a
  dedicated compressed-causal option instead of a generic callback.
- Use this mode for both padded and varlen compressed branches.
- This restores `pack_gqa`, removes the compressed branch's special compile
  path, reduces JIT variants, and enables native varlen for the compressed
  branch.

Expected impact:
- Lower JIT cache footprint (fewer FA4 compile variants), contributing to 2M+
  context support.
- Faster compressed branch from GQA packing + eliminated mask_mod overhead.
- Native varlen for compressed branch (currently padded).
- Lower crossover from better GQA packing and fewer special-case kernels.

Files: `nsa_forward.py` (lines 32-46, 73), `nsa_autograd.py`, FA4 kernel
interface.

### P1. Shared selector for both sparse branches

`fused_score_and_select_blocks()` and `select_compressed_blocks()` both
independently compute `Q_mean x K_cmp` scoring — the same GEMM on the same
data, with different downstream top-k parameters. Block selection (GEMM) is
**43% of forward time at 128K**. When `num_cmp_selected_blocks` is enabled,
we're doing this dominant GEMM twice.

Implementation:
- Build one shared selector pipeline that computes `Q_mean` once, scores
  `Q_mean x K_cmp` once, and emits:
  - selected-branch KV block indices (top-k over original KV blocks)
  - compressed-branch FA4 block indices (top-k over compressed KV blocks)
  - optional compact forward/backward sparse metadata directly
- Keep exact current semantics by deriving both outputs from the same score
  tensor with the same reductions and masking rules used today.
- Delete the duplicate scoring path in `select_compressed_blocks()`.

If selection remains dominant after deduplication, a two-stage approximate
selector (cheap coarse retrieval + exact scoring on shortlist) could be
evaluated separately, gated on model-quality validation.

Expected impact:
- Large-context speedup in the `num_cmp_selected_blocks` regime (eliminates
  duplicate GEMM).
- Lower medium-context crossover.
- Fewer temporary tensors and less Q/K_cmp memory traffic.

Files: `select.py` (lines 403-560, 568-725).

### P2. Compact sparse metadata end-to-end

`build_fa4_block_sparse_tensors()` in `sparsity_masks.py` still allocates
`(B, H, N_q_tiles, n_blocks_k)` dense int32 tensors. At 1M tokens, these are
~4 GiB. `_transpose_block_sparse_for_bwd()` in `nsa_autograd.py` materializes
a dense boolean attendance tensor (~1 GiB at 1M) then repacks.

After the 14x mask construction optimization, this is **no longer a latency
bottleneck** (1-2% of time at 128K+). It is purely a **memory concern** for
1M+ contexts and contributes to the 2M OOM alongside JIT cache pressure.

Implementation:
- Generate compact forward metadata (per-`(b, h, q_tile)` selected KV blocks)
  and compact backward metadata (per-`(b, h, kv_block)` contributing Q tiles)
  directly from selector output without densifying.
- Remove `_transpose_block_sparse_for_bwd()`'s dense boolean attendance path
  entirely.
- Keep sparse metadata scaling as `O(B * H * N_q_tiles * k)` for both forward
  and backward.
- Keep an adapter to FA4's `BlockSparseTensorsTorch` if needed, but make it
  consume compact metadata without allocating `n_blocks_k`-wide dense tensors.

Expected impact: ~5 GiB memory savings at 1M. Necessary for reliable 2M+
execution even after JIT improvements.

Files: `sparsity_masks.py`, `nsa_autograd.py` (lines 44-116).

### P3. Remove CuteDSL auxiliary kernels from the main path

The compress and gating CuteDSL kernels create per-shape JIT artifacts that
inflate the CuTe DSL cache. After P0 reduces FA4 variants, these become the
remaining JIT cache contributors.

Active CuteDSL kernels after GEMM scoring replacement:
- Compress fwd: 1 variant `(dtype, D, block_size)`
- Compress bwd: 1 variant `(dtype, D, block_size)`
- Gating fwd: 2 variants `(dtype, D, has_gate_weight)`
- Gating bwd: 1 variant `(dtype, D)`

Implementation:
- Replace compress and gating CuteDSL kernels with `torch.compile` or
  handwritten CUDA implementations that do not create CuTe JIT artifacts.
- Preserve current math and interfaces.
- Goal: FA4 as the only compiled kernel family on the hot path.

Expected impact: further JIT cache reduction for 2M+ context. Cleaner
CUDA graph capture behavior.

Files: `compress.py`, `gating.py`.

### P4. Stream gating backward for gate-weight case

When `gate_proj_weight` is provided, backward must hold all three branch
outputs (O_cmp, O_slc, O_sld) simultaneously for `fused_gating_backward()`.
This defeats the sequential per-branch memory optimization that already works
for the no-gate-weight case.

Implementation:
- Change gating backward to accept per-branch streamed inputs or sufficient
  summary statistics so branch outputs can be recomputed one at a time.
- Alternatively, recompute O_i inside the gating backward kernel.

Expected impact: lower training peak memory with gate weights.

Files: `nsa_autograd.py` (lines 321-355, 664-713), `gating.py`.

### P5. CUDA graph backward support

Forward CUDA graph capture works and gives 5x at small N. Backward capture
doesn't work yet, limiting the crossover improvement for training. Should be
pursued after P0/P3 structural cleanup for cleaner graph-capture behavior.

Expected impact: lower fwd+bwd crossover from 32K toward 16-20K.

Files: `nsa_autograd.py`, `test/attention/test_cuda_graph_nsa.py`.

### P6. Dead code cleanup

The following are no longer used in the production path:
- CuteDSL select kernel in `select.py` (lines 141-400, ~260 lines)
- `nsa_scoring.py` (146 lines) — building blocks for future fused kernel
- `nsa_topk.py` (273 lines) — building blocks for future fused kernel
- `window_sparse.py` (93 lines) — sliding window as block-sparse (production
  uses FA4 native `window_size_left`)
- `_fa4_fwd_simple()` in `nsa_forward.py` (line 141)

Decision: keep `nsa_scoring.py` and `nsa_topk.py` if a fused scoring-within-FA4
kernel is planned; otherwise remove.

## What NOT to Do

- **Don't re-implement CuteDSL scoring.** GEMM-based scoring eliminated JIT
  cache pressure and cuBLAS is already fast. The CuteDSL select kernel is dead.
- **Don't optimize mask construction latency further.** It's 1-2% of time at
  sizes where NSA wins. The memory footprint matters; the time doesn't.
- **Don't pursue multi-stream FA4 branches.** Already investigated — net
  negative. Each FA4 call saturates all SMs on GB200.
- **Don't add varlen from scratch.** It's already implemented. The remaining
  gap (compressed branch varlen) is subsumed by P0 (mask_mod removal enables
  native varlen).

## Test Plan

### Existing Coverage

- Forward correctness vs reference: `test_sparse_attn_nsa_forward.py` (4 tests)
- Backward correctness: `test_sparse_attn_nsa_backward.py` (25+ tests)
- Varlen fwd+bwd: `test_sparse_attn_nsa_varlen.py` (8 tests)
- Component tests: compress (9), gating (11), select (12), sparsity masks (6)
- FA4 block-sparse backward: `test_fa4_block_sparse_bwd.py`
- FA4 block-sparse varlen: `test_fa4_block_sparse_varlen.py`
- CUDA graph: `test_cuda_graph_nsa.py`
- Benchmarks: `bench_nsa_vs_dense.py`, `bench_nsa_backward.py`,
  `bench_sparse_attn.py`, `bench_sparse_attn_e2e.py`,
  `bench_sparse_compressed.py`
- Diagnostics: `diagnose_memory.py`, `probe_max_seqlen.py`,
  `profile_nsa_forward.py`

### Still Needed

- Compressed-branch parity tests comparing old `mask_mod` behavior to new FA4
  compressed-causal mode (P0).
- Memory regression checks that sparse metadata scales as
  `O(B * H * N_q_tiles * k)` rather than `O(B * H * N_q_tiles * N_kv_blocks)`.
- Peak memory tracking at `N in {128K, 256K, 512K, 1M, 2M}` after P2.
- JIT cache size measurement before/after P0/P3 changes.
- Benchmarks at 2M once JIT cache allows it.

### Acceptance Targets

- 2M forward and backward run without JIT-driven OOM.
- Lower crossover than current ~32K for both inference and training.
- Materially better 512K-1M speedup than current NSA+CmpSparse.

## Success Criteria

### Already Met

- Forward crossover at ~32K (was target: "below ~25K" — achieved at 32K).
- Forward+backward crossover at ~32K (was target: "below ~20K" — achieved at 32K).
- Varlen support with correctness coverage.

### Remaining

- Increase practical max context beyond 2M by reducing JIT cache footprint and
  metadata memory (P0, P2, P3).
- Restore GQA packing for compressed branch (P0).
- Eliminate duplicate GEMM scoring work (P1).

## Assumptions

- Hardware priority is Blackwell/B200 first.
- The current public fixed-length interfaces (`nsa_forward()`, `nsa()`) remain
  supported. Varlen uses the same functions with `cu_seqlens` argument.
- Upstream or local FA4 kernel/API work is in scope for P0.
- The main plan preserves current NSA semantics; approximate selectors are a
  separate gated track requiring model-quality validation.
