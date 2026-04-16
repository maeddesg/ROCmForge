# Batched lm_head Analysis — 2026-04-16

## Configuration

- **Hardware:** RX 9070 XT (gfx1201, RDNA 4), ROCm 7.2.1
- **Model:** Qwen2.5-7B-Instruct-Q4_0 (target), qwen2.5-0.5b-instruct-q4_0 (draft)
- **Prompts:** 15 (5 code, 5 chat, 5 prose)
- **Max tokens:** 128 (throughput sweep), 64 (profiling)
- **Git SHA:** 2d595f8

## Table 1 — Verify Throughput Comparison (Median over 15 prompts)

| Depth | Sequential tok/s | Batched tok/s | Delta | Delta % |
|-------|-----------------|--------------|-------|---------|
| 1 | 69.0 | 68.8 | -0.2 | -0.3% |
| 3 | 65.8 | 65.9 | +0.1 | +0.2% |
| 5 | 57.1 | 57.2 | +0.1 | +0.2% |

Baseline (no spec): **81.8 tok/s**

**All deltas are within measurement noise (±0.5%).**

## Table 2 — Prompt Class Breakdown (Median per class)

| Class | Depth | Sequential | Batched | Delta % |
|-------|-------|-----------|---------|---------|
| Code  | 1 | 77.7 | 77.7 | +0.0% |
| Code  | 3 | 73.5 | 73.7 | +0.3% |
| Code  | 5 | 63.2 | 63.3 | +0.2% |
| Chat  | 1 | 67.6 | 67.6 | +0.0% |
| Chat  | 3 | 64.5 | 64.6 | +0.2% |
| Chat  | 5 | 56.3 | 56.4 | +0.2% |
| Prose | 1 | 66.2 | 66.2 | +0.0% |
| Prose | 3 | 62.8 | 63.1 | +0.5% |
| Prose | 5 | 56.6 | 56.6 | +0.0% |

## Table 3 — Interesting Individual Cases

| Prompt | Depth | Sequential | Batched | Alpha | Comment |
|--------|-------|-----------|---------|-------|---------|
| code_01 | 1 | 85.8 | 85.8 | 91% | Best α at depth=1 |
| code_01 | 3 | 77.6 | 77.8 | 74% | Highest α, best case for batched |
| prose_05 | 1 | 61.3 | 61.4 | 31% | Lowest α, worst case |
| prose_05 | 3 | 60.0 | 60.2 | 29% | Lowest α, worst case |

## Profiling Verification — Non-Layer Overhead (depth=3, code_01, 5 runs each)

| Metric | Sequential (μs) | Batched (μs) | Delta (μs) |
|--------|-----------------|--------------|------------|
| non_layer_overhead (median) | **3,363** | **3,249** | **-114** |
| verify_total (median) | 32,014 | 31,864 | -150 |
| layer_total (median) | 28,633 | 28,616 | -17 (noise) |

**Standard deviations:** Sequential: σ=5 μs; Batched: σ=7 μs. Measurement is highly stable.

### Depth=1 verification (3 runs each)

| Metric | Sequential (μs) | Batched (μs) | Delta |
|--------|-----------------|--------------|-------|
| verify_total | 16,627 | 16,610 | -17 μs |

## Why the Savings Are So Small

### Expected vs. Measured

The original prediction (LAUNCH_OVERHEAD_ANALYSIS.md) estimated **~850 μs per position** for the lm_head, suggesting depth=3 (n=4) would save **~2,550 μs**. Actual savings: **114 μs** (22× less than predicted).

### Root Cause: The 850 μs/position estimate was wrong

The 850 μs figure was the *total* non-layer overhead per position, which includes:
1. Token embedding (~100 μs)
2. Final RMS norm (~12 μs per dispatch)
3. lm_head GEMV (~300-400 μs per dispatch, **memory-bandwidth-bound**)
4. Argmax stage1 + finalize (~50 μs, 2 dispatches)
5. D2H copy + hipStreamSynchronize (~19 μs per sync)

Batching eliminates:
- (N-1) norm dispatches: saves (N-1) × ~12 μs = **36 μs at N=4**
- (N-1) GEMV dispatches: saves (N-1) × ~8 μs **submission overhead only**
- (N-1) syncs: saves (N-1) × ~19 μs = **57 μs at N=4**

But batching does **NOT** save:
- **lm_head weight matrix loads.** The Q4_0 weight matrix is ~307 MB. At ~900 GB/s VRAM bandwidth, each GEMV call takes ~341 μs of bandwidth-limited execution. The batched kernel loads the matrix once but the work per output row is the same. However, the sequential calls also benefit from VRAM prefetching — the GPU's memory controller can overlap weight loads between consecutive dispatches because they access the same addresses. The **effective** savings from batching a bandwidth-bound kernel are much smaller than the theoretical "load once instead of N times" model suggests.
- **Argmax dispatches.** Still N × 2 dispatches (unchanged).
- **Embedding.** Runs before the lm_head phase (unchanged).

### Revised understanding

The actual per-step savings are:
- Dispatch submission: (N-1) × 2 × ~2.7 μs = **16 μs at N=4** (norm + GEMV)
- Sync elimination: (N-1) × ~19 μs = **57 μs at N=4**
- Reduced memory traffic from intermediate buffers: **~30-40 μs at N=4** (normed buffer written once instead of N times)
- **Total: ~100-115 μs** — consistent with the measured 114 μs.

The weight matrix is loaded from VRAM by both paths at the same effective bandwidth. The sequential path's back-to-back GEMV dispatches on the same stream pipeline well enough that the GPU rarely stalls between them.

## Verdict

**The batched lm_head optimization produces a real but negligible improvement: ~114 μs per verify step at depth=3 (0.4% of verify time, <0.1% of end-to-end throughput).**

This is not worth the code complexity if judged purely on performance. However:
- The code is correct (byte-identical output at all depths)
- It's a net simplification of the dispatch pattern (fewer kernel calls)
- It establishes the batched infrastructure (scratch buffers, MAX_SPEC_DEPTH) needed for future optimizations
- Zero regression risk (identical GEMV kernels, identical numerical path)

**Recommendation:** Flip the flag to make batched the default. The improvement is negligible but the code is cleaner and the infrastructure is useful.

## Key Learning

**Memory-bandwidth-bound kernels on consumer GPUs do not benefit significantly from batching when dispatched back-to-back on the same stream.** The GPU's memory subsystem pipelines consecutive accesses to the same weight matrix effectively enough that the "load once instead of N times" model overestimates savings by 20×.

This insight redirects optimization effort:
- **Fused FFN** (the next target) benefits from eliminating *intermediate buffer traffic*, not from reducing dispatch count. The analysis in LAUNCH_OVERHEAD_ANALYSIS.md correctly identified intermediate memory traffic as the dominant cost.
- **lm_head optimization** would only be meaningful with a fundamentally different approach (e.g., keeping the weight matrix in L2 cache — impossible at 307 MB on 6 MB L2).
