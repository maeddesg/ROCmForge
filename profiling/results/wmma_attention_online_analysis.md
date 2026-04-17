# Phase 3b — WMMA prefill attention with online softmax

**Date:** 2026-04-17
**Commit:** on top of `38424f0`
**GPU:** RX 9070 XT (gfx1201, RDNA 4), ROCm 7.2.1
**Kernel:** `hip_kernels/wmma/wmma_attention_prefill.hip`
  (new entry: `wmma_attention_prefill_online_kernel` / `_launch`)

## TL;DR

| seq_len | Existing (µs) | WMMA online (µs) | Speedup |
|--------:|--------------:|-----------------:|--------:|
|      64 |        21,696 |             62.4 |    348× |
|     128 |        37,994 |            114.6 |    331× |
|     256 |        77,633 |            160.4 |    484× |
|     512 |       236,006 |            771.2 |    306× |

Online softmax unlocks prefill attention beyond seq=128 without the
LDS-exhaustion limit of Phase 3a. Extrapolated attention cost for a
full 28-layer Qwen2.5-7B prefill at pp=256: **~4.5 ms** versus the
current **~2,330 ms** — a ~520× cut on the single largest chunk of
prefill wall-clock. Phase 3c (GQA + causal mask) is justified; end-to-
end prefill at pp=256 should now be dominated by the GEMM + overhead
fractions, not attention.

Raw data: [`wmma_attention_online_38424f0_1776424913.json`](wmma_attention_online_38424f0_1776424913.json).

## Method

- Same measurement harness as Step 2: HIP events around the attention
  kernel call, 10 warmup + 100 measured iterations, median µs.
- Inputs: FP32 Q/K/V shared between the two paths; WMMA path receives
  the FP16 downcast outside the timed window.
- Per iteration, both paths do one full "forward" over all 28 Q heads.
  Existing kernel issues 28 separate launches (one per head); WMMA
  does it in one dispatch.
- No causal mask on either side, no GQA.

## Correctness

`tests/wmma_attention_online_softmax_correctness.rs`:

| Test                                          | max_abs_diff | mean    | tol    |
|-----------------------------------------------|-------------:|--------:|-------:|
| seq=64, online vs Phase 3a global softmax     |     2.29e-5  | 3.72e-6 | 1.0e-4 |
| seq=128, online vs Phase 3a                   |     1.51e-5  | 2.68e-6 | 1.0e-4 |
| seq=256, online vs CPU FP32 reference         |     6.17e-6  | 1.03e-6 | 1.0e-2 |
| seq=512, online vs CPU FP32 reference         |     4.78e-6  | 7.27e-7 | 1.0e-2 |

The seq=64 test tolerance is **not** 0.0 even though only one KV tile
is processed (no rescaling). That is expected, not a bug: Phase 3a
stores normalised P in FP16 (values ≤ ~1/64); Phase 3b stores
un-normalised P in FP16 (values up to ~1) and divides by the row sum
at the end. The FP16 quantisation grid lands at different absolute
values in the two paths, so the downstream `P·V` GEMM rounds a hair
differently. A real bug (missed rescale, wrong tile index, off-by-one
in `m`/`l` bookkeeping) would show up at the 1e-3+ level, not 1e-5.

## What the kernel actually does per KV tile

```
1. WMMA: S_tile = Q_tile · K_tile^T              (64×64 FP32 in regs)
2. S_tile *= scale; store to lds_s                (16 KB FP32)
3. per-row softmax update (64 threads, one per row):
     m_new   = max(m_prev, rowmax(S_tile))
     alpha   = exp(m_prev - m_new)                ← always ≤ 1
     P_tile  = exp(S_tile - m_new)                ← un-normalised
     l_new   = alpha · l_prev + sum(P_tile)
     lds_p  ← FP16(P_tile)
     lds_α  ← alpha, lds_m ← m_new, lds_l ← l_new
4. rescale register-resident O accumulator by α    ← per-lane broadcast
5. WMMA: o_acc += P_tile · V_tile                 (64 FP32 per lane,
                                                   persists across
                                                   KV-tile loop)
```

After the loop: `O = o_acc / l`, written to global FP32.

## Register budget

Persistent O accumulator: `2 col chunks × 4 col blocks × 8 FP32 =
64 FP32 / lane`. Per-KV-tile transient WMMA accumulator for S:
`4 col blocks × 8 FP32 = 32 FP32 / lane`, discarded after the softmax
step. A/B register pairs for WMMA: 8 FP16 each, reused per K-tile.
Total working set comfortably under the gfx1201 VGPR budget (512/wave
with full occupancy).

## LDS budget

Separate allocations (no aliasing — the Phase 3a `lds_p`/`lds_s`
aliasing bug is not repeated here):

```
lds_q:    64 · 16 FP16  =  2 KB
lds_k:    16 · 64 FP16  =  2 KB
lds_s:    64 · 64 FP32  = 16 KB
lds_p:    64 · 64 FP16  =  8 KB
lds_v:    16 · 64 FP16  =  2 KB
lds_m:    64    FP32    = 256 B
lds_l:    64    FP32    = 256 B
lds_α:    64    FP32    = 256 B
────────────────────────────────
Total:                ≈ 30.7 KB
```

On gfx1201's 64 KB/block budget this allows **2 resident workgroups
per WGP**, an improvement over Phase 3a's 32 KB layout that had to
serialise across the workgroup.

## End-to-end projection

From Phase 2d (commit `38424f0`):

- pp=256 prefill time on the WMMA GEMM path: 2,770 ms.
- Attribution: GEMM ~110 ms, attention ~2,330 ms, overhead ~330 ms.

Replacing `flash_attn_prefill_strided_kernel` with the Phase 3b WMMA
kernel (28 layers × 160.4 µs per layer = 4.5 ms total attention):

```
new pp=256 prefill time ≈ 110 + 4.5 + 330 = 444.5 ms
new throughput          ≈ 256 / 0.4445 ≈ 576 tok/s
```

That is ~6× the pre-Phase-3 ceiling (92.4 tok/s at pp=256 in Phase 2d),
closing most of the gap to llama.cpp's 1,092 tok/s on the same
hardware — the residual is the 330 ms of un-fused norm/RoPE/residual
overhead.

**This projection assumes Phase 3d integration wires the new kernel
into the forward path cleanly.** Phase 3c (GQA for KV, causal mask)
is a prerequisite since Qwen2.5-7B uses 28 Q heads against 4 KV heads.

## Verdict

**Phase 3c (GQA + causal mask) justified.** The kernel demonstrates
that attention at seq=256 can run in <200 µs — far below the 2.33 s
that the current scalar kernel spends on it — so the remaining work
to make it usable against Qwen is the layout + mask story, not
further perf tuning.

### Not in scope here

- GQA KV sharing                                → Phase 3c
- causal mask                                   → Phase 3c
- arbitrary seq_len (non-64-aligned)            → Phase 3c / 3d
- forward-path dispatch integration             → Phase 3d
- LDS bank-conflict / double-buffered variants  → post-3d if warranted
