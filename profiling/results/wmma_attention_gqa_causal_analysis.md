# Phase 3c — GQA + causal mask WMMA attention

**Date:** 2026-04-17
**Commit:** on top of `4f9a446`
**GPU:** RX 9070 XT (gfx1201, RDNA 4), ROCm 7.2.1

## TL;DR

| seq_len | Existing (µs) | Phase 3b (µs) | **Phase 3c (µs)** | vs existing | vs Phase 3b |
|--------:|--------------:|--------------:|------------------:|------------:|------------:|
|      64 |        21,665 |          66.4 |           **70.7** |       306×  |     −6 %   |
|     128 |        39,116 |         121.3 |          **129.6** |       302×  |     −6 %   |
|     256 |        78,412 |         190.8 |          **220.8** |       355×  |    −13 %   |
|     512 |       235,521 |         645.0 |          **486.8** |       484×  |    +33 %   |

**Phase 3d (forward-path integration) bereit.** GQA + causal are both
correct (9/9 tests green) and the kernel delivers a 300-500× speedup
against the existing scalar kernel on the Qwen2.5-7B attention shape
(28 Q heads, 4 KV heads, head_dim=128, causal). Extrapolated full
28-layer pp=256 prefill attention: ~6.2 ms (vs ~2,330 ms baseline).
Projected end-to-end pp=256 throughput: **~574 tok/s** (vs 92.4 tok/s
after Phase 2d).

Raw data: [`wmma_attention_gqa_causal_4f9a446_1776425867.json`](wmma_attention_gqa_causal_4f9a446_1776425867.json).

## Correctness

`tests/wmma_attention_gqa_causal_correctness.rs` runs three scenarios
against a CPU FP32 reference that models the exact GQA mapping
(kv_head = q_head / gqa_ratio) and the causal mask directly.

| Test                              | max_abs_diff | mean    | tol    |
|-----------------------------------|-------------:|--------:|-------:|
| GQA 28/4, seq=64, no causal       |     1.21e-5  | 2.03e-6 | 1.0e-2 |
| GQA 28/4, seq=128, no causal      |     8.93e-6  | 1.44e-6 | 1.0e-2 |
| causal, seq=64  (28/28 heads)     |     3.31e-5  | 3.35e-6 | 1.0e-2 |
| causal, seq=128                   |     3.48e-5  | 2.43e-6 | 1.0e-2 |
| causal, seq=256                   |     3.30e-5  | 1.84e-6 | 1.0e-2 |
| GQA 28/4 + causal, seq=64         |     3.19e-5  | 3.30e-6 | 1.0e-2 |
| GQA 28/4 + causal, seq=128        |     3.55e-5  | 2.45e-6 | 1.0e-2 |
| GQA 28/4 + causal, seq=256        |     3.48e-5  | 1.82e-6 | 1.0e-2 |
| O[0] ≈ V[0] (first-token attends to itself only) | 0.0e0 | — | 1.0e-3 |

`O[0] == V[0]` bit-exact: the first query position sees only its own
key, softmax is trivially `[1, 0, …, 0]`, and the output is exactly
V[0]. The kernel passes this structural test across all 28 heads.

## What changed in the kernel

`hip_kernels/wmma/wmma_attention_prefill.hip`:

1. Separate `q_row_stride` and `kv_row_stride`. For Qwen2.5-7B the
   activation-side (Q / O) stride is `28 · 128 = 3584`; the KV-side
   stride is `4 · 128 = 512`. The kernel computes
   `kv_head = q_head / gqa_ratio` and derives `q_head_off` /
   `kv_head_off` separately.
2. `causal` flag. When set:
   - **early break**: the kernel computes
     `kv_tile_bound = (q_row_base + TILE_M + TILE_KV − 1) / TILE_KV`
     and iterates only up to that bound. Tiles strictly above the
     diagonal are never touched.
   - **diagonal masking**: for the one tile that straddles the
     diagonal, the S store masks entries where `kv_pos > q_pos` to
     `-1e30` (not `-INFINITY`, to avoid occasional NaN-propagation
     paths some compilers insert for infinities). The subsequent
     softmax handles this naturally: `expf(x - m)` for a masked
     value is `exp(large-negative) ≈ 0`.
3. Row-sum guard at the output-normalisation step: if a row is
   fully masked (`l == 0`), we multiply by zero instead of
   dividing by zero — the result for such rows is undefined but
   must not be NaN.

The Phase 3b-compatible launcher
(`wmma_attention_prefill_online_launch`) is preserved as a thin shim
that calls the unified kernel with `gqa_ratio=1, causal=0`, so the
Phase 3b correctness tests stay green.

## Performance decomposition

Causal masking saves roughly `N_skipped / N_total` tiles. For
seq=512 with 8×8 tiles the upper triangle has 28 fully-skipped tiles
and 8 diagonal tiles (which still run full-GEMM work with the mask
applied), so effective work ≈ 36 / 64 = 56 %. Measured Phase 3c /
Phase 3b = 486.8 / 645.0 = 75 %, i.e. ~25 % saving — less than the
ideal 44 % because:

- diagonal tiles pay the full WMMA cost plus a per-element mask
  check;
- the `kv_tile_bound` computation and the `is_diagonal` branch add
  constant overhead per q-tile, which is amortised across fewer
  KV tiles at short sequences;
- GQA has effectively **no compute savings** in this implementation
  — each Q head still launches its own block and loads its own K/V.
  The 7:1 sharing is implicit (L2 cache); we do not explicitly
  consolidate loads in LDS yet.

At seq ≤ 256 the Phase 3b kernel is effectively ~6–13 % faster
because it lacks these checks. Because end-to-end we need the causal
mask for correctness on a real model, Phase 3c is the required
baseline for Phase 3d.

## End-to-end projection

From Phase 2d: pp=256 prefill takes 2,770 ms (GEMM 110 + attention
2,330 + overhead 330).

Replacing the scalar attention with Phase 3c:

```
Attention (Phase 3c, seq=256): 220.8 µs × 28 layers = 6.2 ms
Prefill (pp=256)              : 110 + 6.2 + 330 = 446 ms
Throughput                    : 256 / 0.446 ≈ 574 tok/s
```

The residual is now the ~330 ms of un-fused norm/RoPE/residual
overhead. GEMM (110 ms) and attention (6 ms) are both small fractions
of the total. Further end-to-end gain needs an "overhead reduction"
pass, not more attention tuning.

Relative to llama.cpp ROCm (~1,092 tok/s at pp=19), our projected
~574 tok/s at pp=256 closes about half of the remaining gap. llama.cpp
measured pp=19, so a direct comparison at pp=256 is apples-to-oranges
— they may also slow down on longer prompts due to attention cost.
A proper head-to-head at pp=256 would be the next question for the
next benchmarking round.

## What's explicitly NOT in this phase

- Forward-pass dispatch integration (Phase 3d).
- Explicit in-LDS KV sharing across Q heads (the ~7× load reduction
  the prompt hoped for). L2 cache already covers most of this in
  practice, and the dispatch complexity is parked for after 3d if
  wall-clock needs it.
- Double-buffered K/V loads, bank-conflict tuning, mask-aware kernel
  specialisations for the upper/lower/diagonal tile families.
- Non-64-aligned seq_len — Qwen prompts that tokenise to unpadded
  shapes need either a pad layer upstream or a kernel variant with
  per-element seq_len bounds checking.
