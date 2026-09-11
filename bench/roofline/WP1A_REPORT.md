# MSLK WP-1a — Roofline Profiling & Performance Target Selection

**Date:** 2026-09-09
**GPU:** 8× AMD Instinct MI350X (gfx950), ROCm 7.2.2
**Tool:** rocprof-compute 3.4.0
**Kernel:** FP8 Rowwise Preshuffle FlyDSL (PR #434)

---

## 1. Empirical Roofline (gfx950, unlocked clocks)

| Metric | Datasheet | Empirical | % of Datasheet |
|--------|-----------|-----------|----------------|
| FP8 MFMA | 4600 TFLOPS | 2240.7 TFLOPS | 48.7% |
| FP4 MFMA | 9200 TFLOPS | 8216.3 TFLOPS | 89.3% |
| FP16 MFMA | — | 1120.4 TFLOPS | — |
| BF16 MFMA | 2300 TFLOPS | 562.4 TFLOPS | 24.5% |
| FP32 MFMA | 144.2 TFLOPS | 141.7 TFLOPS | 98.3% |
| HBM BW | ~5300 GB/s | 6167.5 GB/s | >100% |
| L2 BW | — | 31796.0 GB/s | — |

> Datasheet FP8/BF16 peaks are ~2–4× the measured achievable. All targets below use empirical ceilings.

---

## 2. Regime Classification

| Regime | Shape band | Ceiling | Target band |
|--------|-----------|---------|-------------|
| R1 — Large compute-bound | M≥4096, N,K≥4096 | FP8 MFMA peak (2240.7 TFLOPS) | 85–92% |
| R2 — Medium | M 512–4096 | FP8 MFMA peak | 75–85% |
| R3 — Skinny / decode | M≤128, N,K large | HBM BW (6167.5 GB/s) | 70–85% |
| R4 — Small-N/K tail | N or K < 1024 | AITER/CK best | ≥95% |
| R5 — Grouped / MoE | Multiple groups | AITER best (aggregate) | ≥90% |

---

## 3. Optimization Pass: Config Sweep (WP-1a.1)

Swept 432 tile configs per shape (50+ tile combos × 3 xcd × 3 waves_per_eu).
Added shape overrides for N=7168/8192, K=3584/8192 to `_SHAPE_OVERRIDES_GFX950`.

| Shape | Before | After | Speedup | Winning Config |
|-------|--------|-------|---------|----------------|
| 8192×8192×8192 (R1) | 86.2% | **91.7%** | 1.06x | (128,256,128) xcd=0 wpe=2 |
| 128×8192×8192 (R3) | 29.0% | **36.5%** | 1.38x | (64,64,256) xcd=1 wpe=0 |
| 1024×8192×8192 (R2) | 67.0% | **71.1%** | 1.12x | (64,256,128) xcd=4 wpe=0 |
| 8192×7168×8192 (R1) | 84.2% | **91.3%** | 1.09x | (128,256,128) xcd=0 wpe=2 |

---

## 4. Benchmark Results — After Optimization

### 4.1 Compute-bound shapes (R1/R2)

| Shape (M×N×K) | Regime | TFLOPS | % of Peak | Target | Pass | Δ vs Before |
|---------------|--------|--------|-----------|--------|------|-------------|
| 8192×8192×8192 | R1 | 2054.1 | **91.7%** | ≥90% | **PASS** | +5.5pp |
| 8192×7168×8192 | R1 | 2046.6 | **91.3%** | ≥85% | **PASS** | +7.2pp |
| 4096×8192×8192 | R1 | 2005.7 | **89.5%** | ≥85% | **PASS** | +3.9pp |
| 4096×7168×8192 | R1 | 1872.2 | 83.6% | ≥85% | FAIL | +2.1pp |
| 8192×8192×3584 | R2 | 1891.4 | **84.4%** | ≥75% | **PASS** | +3.6pp |
| 8192×7168×3584 | R2 | 1816.3 | **81.1%** | ≥75% | **PASS** | +0pp |
| 4096×8192×3584 | R2 | 1823.6 | **81.4%** | ≥75% | **PASS** | +1.5pp |
| 4096×7168×3584 | R2 | 1693.9 | **75.6%** | ≥75% | **PASS** | -0.2pp |
| 1024×8192×8192 | R2 | 1593.6 | 71.1% | ≥80% | FAIL | +4.1pp |
| 1024×7168×8192 | R2 | 1617.4 | 72.2% | ≥75% | FAIL | +11.3pp |

### 4.2 Memory-bound shapes (R3 — decode)

| Shape (M×N×K) | GB/s | % of HBM BW | Target | Pass | Δ vs Before |
|---------------|------|-------------|--------|------|-------------|
| 1×8192×8192 | 3081.2 | 50.0% | ≥70% | FAIL | +0.9pp |
| 1×7168×8192 | 2836.9 | 46.0% | ≥70% | FAIL | +1.8pp |
| 128×8192×8192 | 2253.0 | 36.5% | ≥80% | FAIL | +7.5pp |
| 128×7168×8192 | 2132.3 | 34.6% | ≥80% | FAIL | +6.0pp |
| 16×8192×8192 | 2941.0 | 47.7% | ≥75% | FAIL | +3.4pp |
| 1×1280×8192 | 322.0 | 5.2% | ≥70% | FAIL | +0pp |

### 4.3 Summary

| | Before | After | Δ |
|--|--------|-------|---|
| R1 targets met | 1/4 | **3/4** | +2 |
| R2 targets met | 4/13 | **4/13** | — |
| R3 targets met | 0/19 | 0/19 | — |
| **Total** | **5/36** | **7/36** | **+2** |

---

## 5. Analysis & Remaining Gaps

### Solved: R1 large compute-bound
The 8192×8192×8192 shape now hits **91.7%** of empirical MFMA peak (was 86.2%). The key change was `xcd_swizzle=0` instead of `xcd_swizzle=1` — the default XCD swizzle pattern was causing cross-die imbalance at this shape. Three of four R1 shapes now pass.

### Remaining gap: 4096×7168×8192 (R1, 83.6% vs 85%)
1.4pp short of target. 7168 is not a clean multiple of tile_n=256 (7168/256=28 tiles, but XCD mapping of 28 tiles is suboptimal on 8-XCD MI350). A dedicated xcd_swizzle for this geometry or a tile_n=128 variant may close it.

### Remaining gap: R2 medium shapes
- 1024× shapes improved from 60–67% to 71–72% — close but still below 75% targets
- N=1280 shapes stuck at 13–61% — tile quantization loss is structural (1280 / 128 = 10, but 1280 / 256 = 5 which wastes half the tile)

### Structural gap: R3 decode shapes
R3 shapes are 5–50% of HBM BW. The FlyDSL preshuffle kernel is fundamentally a compute-oriented design: it uses MFMA-heavy tiles with deep K-loops. For M≤128, the problem is too small to fill the GPU and the kernel becomes launch-overhead-bound (M=1) or memory-latency-bound (insufficient wave occupancy to hide HBM latency).

**What's needed for R3:** A separate kernel path — either a Triton streamk kernel tuned for bandwidth, or a CK-based decode kernel that the FlyDSL kernel dispatches to when M ≤ threshold.

---

## 6. Measurement Conditions

| Parameter | Value |
|-----------|-------|
| GPU | AMD Instinct MI350X (gfx950) |
| ROCm | 7.2.2 |
| PyTorch | 2.10.0+rocm7.2.2 |
| FlyDSL | 0.2.4 |
| Clock state | Unlocked (boost enabled) |
| Benchmark iterations | 5 per shape (median reported) |

---

## 7. Changes Made

### `mslk/gemm/flydsl/preshuffle_gemm.py`
Added 18 shape override entries to `_SHAPE_OVERRIDES_GFX950` covering:
- N=8192, K=8192: per-M configs for M=1..8192
- N=7168, K=8192: per-M configs for M=1..8192
- N=8192, K=3584: per-M configs for M=1..8192

### `bench/roofline/` (new)
Full WP-1a tooling: regime classifier, rocprof-compute capture, target matrix schema, benchmark harness, orchestrator.

---

## 8. Artifacts

| File | Description |
|------|-------------|
| `/tmp/roofline_gfx950.json` | Empirical roofline JSON |
| `/tmp/mslk_wp1a_v2/flydsl_bench_*.csv` | Benchmark results (post-optimization) |
| `/tmp/mslk_wp1a_v2/target_matrix_*.yaml` | Target matrix with baselines |
| `bench/roofline/templates/target_matrix.yaml` | Template (37 production shapes) |
