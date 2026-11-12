# T8: Evaluate SDPA (W1–W3) — Historical Internal Evaluation Note

The measurements below are the original author's report from the combined
`t8-eval-tree`, not results from this diff's parent-relative code. That tree
included additional provider changes. In particular, the reported D=512 FlyDSL
results are not reproducible from this diff alone: its base caps FlyDSL at
D=256. The final caveat records a known forward correctness defect in that
experimental kernel, so those timings do not establish a correct implementation.

The rebased script honors `--iters` and `--warmup`, preserves shared K/V storage,
and enumerates 508 unique configurations (496 W1, 10 W2, 2 W3). The original
524 rows included 16 repeated W1 self-attention configurations. SDPA input
layout conversion and forward graph construction for backward timing are outside
the timed interval; timings are event-loop means, while the reported medians below
summarize multiple shapes. Current provider predicates determine eligibility;
the base now supports unbiased FlyDSL cross-attention for D<=256.

Fresh validation results and their limitations are linked from this diff's
Internal Test Plan and its companion explanation.

For true shared K/V inputs, SDPA and AITER autograd include accumulation into
the shared leaf in their timed backward call. Direct FlyDSL backward returns
separate dK and dV; its optional sum is timed separately.

## Original Setup

- **Hardware:** 8x AMD Instinct MI350X (gfx950), single-GPU per measurement
- **Software:** PyTorch 2.12 + ROCm 10, bf16, `op-perf-amd` container, MSLK_PYTHON_ONLY=1
- **Branch:** `t8-eval-tree` — main + PRs #583+#584+#590+#600 merged (conflicts in `flydsl.py` resolved in commit `518ac1a`)
- **Timing:** Plain-loop with CUDA events, uniform for all providers
- **Providers:** FlyDSL, triton_splitk, PyTorch SDPA, AITER flash attention (direct). CK correctly declines on gfx950 (SUPPORTED_GFX_ARCHS = {gfx942}).
- **Shapes:** 524 shapes total — actual W1/W2/W3 workload definitions (B 683–1172, H 1–2, correct Lq/Lkv/D per workload). Forward and backward.
- **Dependencies:** T4 assumed satisfied. CK on gfx942 not measured (no machine).

## Forward Latency (median us, bf16)

W1 (B 683–1172, H 1–2, Lq 16/24/32, Lkv 24–1984):
- D=80: FlyDSL 42us, AITER 51us, SDPA 63us → **FlyDSL wins**
- D=96: FlyDSL 42us, AITER 54us, SDPA 99us → **FlyDSL wins**
- D=128: FlyDSL 42us, AITER 57us, SDPA 110us, triton_splitk 105us → **FlyDSL wins**
- D=256: FlyDSL 42us, SDPA 189us, triton_splitk 201us, AITER 244us → **FlyDSL wins**

Note: FlyDSL forward with #600 declines cross-attention (Mq != Mkv). The 42us median is from the self-attention subset (Lq=Lkv). For cross-attention shapes, FlyDSL is N/A. Per Lkv breakdown below.

W1 forward by Lkv (cross-attention, FlyDSL N/A):
- D=80/96/128, Lkv 24–128: AITER wins (22–23us vs SDPA 16–68us). AITER has a flat ~22us profile at small Lkv.
- D=80, Lkv 1024: SDPA 126us, AITER 150us → SDPA wins
- D=80, Lkv 1984: AITER 286us, SDPA 424us → AITER wins (1.5x)
- D=96/128, Lkv 128–1984: AITER wins across the board (23–287us vs SDPA 39–432us)
- D=256, all Lkv: SDPA wins (40–698us vs AITER 42–975us). AITER is not competitive at D=256.

W2 (B 1024, H 1, D 256):
- Lq=32 cross (Lkv 200/400): SDPA 79–143us, AITER 82–176us → **SDPA wins**
- Lq=224 cross (Lkv 200/400): AITER 236–365us, SDPA 305–474us → **AITER wins**
- Self 2200×2200 causal: FlyDSL 6674us, AITER 7112us, SDPA 9376us → **FlyDSL wins**
- Self 2200×2200 non-causal: FlyDSL 12522us, SDPA 14021us, AITER 12946us → **FlyDSL wins**

W3 (B 1172, H 1, Lq 16, Lkv 25, D 512):
- SDPA 37us, FlyDSL 54us (T18 kernel), triton_splitk 105us, AITER N/A → **SDPA wins**

## Backward Latency (median us, bf16)

W1:
- D=80: AITER 249us, SDPA 659us, FlyDSL 983us → **AITER wins** (2.6x over SDPA)
- D=96: AITER 215us, SDPA 559us, FlyDSL 836us → **AITER wins** (2.6x)
- D=128: AITER 182us, SDPA 671us, FlyDSL 958us → **AITER wins** (3.7x)
- D=256: AITER 491us, SDPA 1184us, FlyDSL 1915us → **AITER wins** (2.4x)

W2 (D=256):
- Lq=32 cross: AITER 289–441us, SDPA 505–948us, FlyDSL 935–1369us → **AITER wins**
- Lq=224 cross: AITER 1057–1613us, SDPA 1759–3153us, FlyDSL 2188–2595us → **AITER wins**
- Self 2200×2200 causal: AITER 33776us, FlyDSL 39731us, SDPA 45187us → **AITER wins**
- Self 2200×2200 non-causal: SDPA 50024us, AITER 68583us, FlyDSL 69283us → **SDPA wins**

W3 (D=512):
- FlyDSL 149–156us, SDPA 317–320us, AITER N/A → **FlyDSL wins** (2x over SDPA)

## K-is-V dK+dV Sum Cost

K-is-V shapes show identical latency to non-K-is-V (the kernel doesn't exploit shared K=V). dK+dV elementwise sum cost is 4–28us, negligible relative to backward kernel time.

## Provider Routing Decision

W1 forward (short-query cross-attention):
- D=80/96/128, all Lkv: AITER for cross-attention (flat 22–57us), FlyDSL for self-attention
- D=256: SDPA for cross-attention, FlyDSL for self-attention

W1 backward:
- All D: AITER across the board (2.4–3.7x faster than alternatives)

W2 forward (D=256):
- Short cross (Lq=32): SDPA
- Long cross (Lq=224): AITER
- Self-attention: FlyDSL

W2 backward: AITER (except non-causal self-attention where SDPA wins)

W3 forward (D=512): SDPA (37us)
W3 backward (D=512): FlyDSL (152us, from T18 kernel)

## T14–T18 Scoping

**T14 (SDPA dispatch integration):** Proceed. SDPA is the best forward for W2 short cross-attention (Lq=32), W3 forward (D=512), and W1 D=256 cross-attention. Needed as a dispatch target.

**T15 (AITER fmha ops):** Proceed — high priority. AITER dominates backward at all D <= 256 (2.4–3.7x over SDPA/FlyDSL). Also wins W1 forward cross-attention at D=80/96/128 and W2 long cross (Lq=224). Gate by D <= 256. For forward, gate cross-attention by D <= 128 (not competitive at D=256). For backward, enable broadly at all D <= 256.

**T16 (native D=80/96 compute):** Still needed. FlyDSL forward declines cross-attention with #600, so it can't serve those shapes. But FlyDSL forward self-attention at D=80/96 (42us) is already fast without native compute — the padding overhead is hidden at these batch sizes. T16's value is primarily enabling FlyDSL forward cross-attention at D=80/96, which is currently N/A. Lower priority than T15 since AITER covers those shapes.

**T17 (short-query tiles):** Deprioritize. AITER handles short-query cross-attention well. No gap.

**T18 (D=512):** Proceed. T18 kernel already provides FlyDSL backward at D=512 (152us, 2x over SDPA 318us). Forward is 54us vs SDPA 37us — below A6 target (0.9 × 37 = 33us). Options: accept SDPA for W3 forward, or optimize the T18 forward kernel further.

## Caveats

1. CK on gfx942 not measured. No gfx942 machine available.
2. FlyDSL forward median (42us) is from self-attention shapes only — cross-attention is N/A with #600.
3. W2 self-attention 2200×2200 at B=1024 is very large (>3GB). These shapes ran but timings are in milliseconds.
4. AITER D=256 forward is not competitive (244us median vs SDPA 189us, FlyDSL 42us self-attn). AITER backward at D=256 is still the best (491us).
5. T18 FlyDSL D=512 kernel is a work in progress (forward has an MFMA register layout bug in P@V; backward compiles but untested). Numbers reported are from the current state.
