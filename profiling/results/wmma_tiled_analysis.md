# Tiled WMMA GEMM — Phase 2a Step 2 verdict

**Date:** 2026-04-17
**Commit:** on top of `854c7da`
**GPU:** RX 9070 XT (gfx1201, RDNA 4), ROCm 7.2.1
**Kernel:** `hip_kernels/wmma/wmma_gemm_tiled.hip` (64×64 output tile, 128 threads / 4 waves, K-loop over LDS)

## TL;DR

| Shape (M × N × K)            | hipBLAS median | WMMA median | Speedup | % of FP16 peak |
|------------------------------|---------------:|------------:|--------:|---------------:|
| QKV/O   (256 × 3584 × 3584)  |        644 µs  |    217.6 µs |  **2.96×** | **61.7 %** |
| Gate/Up (256 × 18944 × 3584) |      3,195 µs  |    940.4 µs |  **3.40×** | **75.4 %** |
| Down    (256 × 3584 × 18944) |      3,324 µs  |  1,039.8 µs |  **3.20×** | **68.2 %** |

First-pass tiled WMMA kernel with no bank-conflict tuning and no double-buffering already hits **62–75 % of the FP16 theoretical peak** (49 TFLOPS assumed) and is **~3× faster than hipBLAS** on the three prefill shapes that matter for Qwen2.5-7B. Phase 2b (Q4_0 inline dequant) is justified — skipping the full FP16 materialisation should recover the remaining headroom.

## Method

- Standalone binary `bench_wmma_tiled` (Cargo bin, `--features gpu`).
- 10 warmup + 100 measured iterations per shape.
- HIP events around each launch, not wall-clock.
- Inputs drawn from `[-0.25, 0.25]` with a deterministic seed (irrelevant for timing, relevant for the matching correctness test).
- hipBLAS reference uses `hipblasHgemm` in the same row-major layout (`OP_N`/`OP_N`, operands swapped) as the WMMA kernel so the two paths compute the same product.

Raw data: [`wmma_tiled_bench_854c7da_1776415672.json`](wmma_tiled_bench_854c7da_1776415672.json).

## Correctness

`tests/wmma_tiled_correctness.rs` — 5 shapes (64³, 256³, 256×3584², 256×18944×3584, 256×3584×18944). All green. Tolerances per the expected FP16 accumulation drift of the hipBLAS reference versus WMMA's FP32 accumulator:

| K      | max-tol | mean-tol | observed max     | observed mean |
|-------:|--------:|---------:|-----------------:|--------------:|
|     64 |  5.0e-3 |   7.5e-4 | 1.112e-3 ✓       | 1.391e-4 ✓    |
|    256 |  1.5e-2 |   2.3e-3 | 7.354e-3 ✓       | 5.527e-4 ✓    |
|  3,584 |  3.0e-1 |   4.5e-2 | 1.72e-1 ✓        | 7.69e-3 ✓     |
| 18,944 |  1.2e0  |   1.8e-1 | 6.68e-1 ✓        | 4.04e-2 ✓     |

Mean diff scales as ≈ K · eps(FP16) on inputs of this magnitude, consistent with the hipBLAS reference accumulating in FP16. If the kernel had a systematic bug (transpose, wrong lane mapping) the mean would be the same order of magnitude as the max, not 15–20× smaller; the mean-tolerance check guards against exactly that class of regression.

## Peak analysis

Theoretical peak (49 TFLOPS FP16):

```
peak_us = 2 · M · N · K / 49e12 · 1e6
```

- QKV/O:   2 · 256 · 3584 · 3584  / 49e12 ≈ 134 µs
- Gate/Up: 2 · 256 · 18944 · 3584 / 49e12 ≈ 709 µs
- Down:    2 · 256 · 3584 · 18944 / 49e12 ≈ 709 µs

WMMA fraction of peak:

- QKV/O:   217.6 µs actual / 134.2 µs peak = **61.7 %** → kernel overhead visible on the smaller shape.
- Gate/Up: 940.4 µs actual / 709.4 µs peak = **75.4 %** → the biggest shape, most compute-bound, closest to peak.
- Down:    1039.8 µs actual / 709.4 µs peak = **68.2 %** → more K-tile iterations (K=18944 ⇒ 1,184 K-tiles vs. 224 for Gate/Up), so more LDS round-trips per GEMM.

## What's left on the table

- **LDS bank conflicts.** The K-splitting load pattern `{0,1,2,3, 8,9,10,11}` / `{4,5,6,7, 12,13,14,15}` maps lanes 0 and 16 onto overlapping banks. Phase 2a prompt explicitly parks this in Phase 2c — noted here, not fixed. Likely responsible for a chunk of the 25–40 % peak gap.
- **No double-buffering.** Single-buffered LDS with a `__syncthreads()` per K-tile; a K-tile's global load waits for the previous K-tile's WMMA to finish. Double-buffering would overlap the two.
- **Small shape has visible wave-level overhead.** QKV/O sits at 62 % while Gate/Up reaches 75 %. Likely the load → K-loop ramp is amortised less efficiently on the shorter K. Tile-shape tuning (128×64 instead of 64×64, or smaller tiles) could help — but that's also Phase 2c.
- **M must be a multiple of 64, K a multiple of 16.** Arbitrary-shape padding is a Phase 2c task. All Qwen2.5-7B projection shapes satisfy the constraint as written.

## Verdict

**Phase 2b (inline Q4_0 dequant + WMMA) is justified.** The hard part — making WMMA produce correct, fast results on gfx1201 — is done. Inline dequant replaces the ~150 MB FP16 weight scratch round-trip with direct Q4_0 consumption in the K loop, which is pure throughput win: the FP16 materialisation currently happens per projection inside the hipBLAS wrapper and disappears in the WMMA path if we fold it in.

Extrapolated full-prefill picture (Qwen2.5-7B, 28 layers, ~5 projections per layer, pp=256):

- Current hipBLAS prefill time (dominated by the three shape classes): roughly 28 × (644 + 2·3195 + 3324) ≈ 290 ms of pure GEMM.
- WMMA without inline dequant: 28 × (218 + 2·940 + 1040) ≈ 87 ms GEMM — a 3.3× shrink on the compute side alone, plus a one-time dequant cost that currently sits around 22 ms/prefill (bandwidth-limited, unchanged by Phase 2a).
- Expected end-to-end prefill speedup vs the hipBLAS baseline: 2–2.5× (attention does not shrink). That moves the 86 tok/s ceiling into the ~200 tok/s range, still far from llama.cpp's 1,092 tok/s but a clear step.
- Phase 2b + 2c should push further: inline dequant eliminates the FP16 scratch traffic (extra win on Down-shape where K dominates), and LDS bank-conflict tuning recovers part of the 25–40 % peak gap.

## Not in scope here (deferred)

- Q4_0 inline dequant — Phase 2b.
- LDS padding / swizzle for bank-conflict avoidance — Phase 2c.
- Double-buffered K-loop — Phase 2c.
- Tile-shape autotune (64×64 vs 128×64 vs 32×128) — Phase 2c.
- Dispatch integration into the prefill path — Phase 2d.
