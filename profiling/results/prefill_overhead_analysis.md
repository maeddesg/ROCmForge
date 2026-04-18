# Prefill Overhead Profiling — Phase 4 Step 1

**Date:** 2026-04-18
**Hardware:** AMD Radeon RX 9070 XT (gfx1201, RDNA 4), ROCm 7.2.2
**Model:** Qwen2.5-7B-Instruct Q4_0, 28 layers, h=3584, ff=11008, 28/4 GQA
**Instrumentation:** per-op `Instant::now()` + `hipDeviceSynchronize()`, gated behind `ROCMFORGE_PROFILE_PREFILL_OPS=1`
**Raw logs:** `profiling/results/prefill_overhead_pp{64,256}.log`

---

## TL;DR

- Measured pp256 prefill: **441.24 ms** end-to-end (vs. 411 ms unprofiled — the 30 ms delta is the synchronisation + lost pipelining cost introduced by per-op syncs; acceptable for a profiling run).
- **GEMM dominates at 94.7 % (417.7 ms).** The pre-study estimate of "110 ms GEMM + 295 ms norm/rope/residual overhead" was inverted: the non-GEMM overhead is only **~14 ms (3 %)** combined, GEMM is the gap.
- **Inside GEMM, `down_proj` alone is 303.8 ms (69 %)** — one op, out of 17, consumes more than two thirds of the prefill budget.
- gate_proj (44.4 ms) + up_proj (43.9 ms) = **88.2 ms (20 %)**. Combined with down, **FFN projections are 88.3 % of prefill**.
- Attention (WMMA) is 2.2 % — already small, not a target.
- Norm + RoPE + Residual + Activation + KV-write + Embedding + Launch ≈ 14 ms total (3 %) — fusion saves at most 14 ms.

---

## Tabelle 1 — Per-Operation Aggregation (pp256, 28 Layer summiert)

| Operation | Summe (ms) | Anteil (%) | Pro Layer median (µs) |
|---|---:|---:|---:|
| norm_pre_attn | 1.32 | 0.3 | 34 |
| q_proj | 9.70 | 2.2 | 342 |
| k_proj | 2.74 | 0.6 | 97 |
| v_proj | 2.73 | 0.6 | 98 |
| qkv_bias | 1.42 | 0.3 | 34 |
| rope_q | 1.24 | 0.3 | 34 |
| rope_k | 0.61 | 0.1 | 21 |
| kv_write | 1.07 | 0.2 | 20 |
| attention | 9.73 | 2.2 | 319 |
| o_proj | 9.56 | 2.2 | 342 |
| residual_attn | 0.73 | 0.2 | 26 |
| norm_pre_ffn | 0.90 | 0.2 | 32 |
| gate_proj | 44.36 | 10.1 | 1585 |
| up_proj | 43.86 | 10.0 | 1563 |
| silu_mul | 3.81 | 0.9 | 134 |
| **down_proj** | **303.77** | **69.3** | **10988** |
| residual_ffn | 0.84 | 0.2 | 30 |
| **Layer-Summe** | **438.40** | **100.0** | — |

The per-layer median excludes layer 0 (JIT / first-launch effects).

## Tabelle 2 — Top-Level-Breakdown (pp256)

| Komponente | Zeit (ms) | Anteil (%) |
|---|---:|---:|
| Embedding | 1.40 | 0.3 |
| 28 Layer (gesamt) | 438.83 | 99.5 |
| Final Norm | 0.03 | 0.0 |
| LM Head | 0.97 | 0.2 |
| Unaccounted | 0.00 | 0.0 |
| **Gesamt** | **441.24** | **100.0** |

Layer loop is 99.5 % of prefill. Embed, final norm, LM head together are 0.5 %.

## Tabelle 3 — Overhead-Kategorien (pp256)

| Kategorie | Summe (ms) | Anteil Prefill (%) |
|---|---:|---:|
| **GEMM (Q/K/V/O/Gate/Up/Down + LM Head)** | **417.70** | **94.7** |
| Attention (WMMA) | 9.73 | 2.2 |
| Activation (SiLU + Mul) | 3.81 | 0.9 |
| Norm (RMS × 2 pro Layer + Final) | 2.25 | 0.5 |
| RoPE (Q + K) | 1.85 | 0.4 |
| Residual (post-attn + post-FFN) | 1.57 | 0.4 |
| QKV-Bias | 1.42 | 0.3 |
| Embedding | 1.40 | 0.3 |
| KV-Cache-Write | 1.07 | 0.2 |
| Launch/Dispatch + Unaccounted | 0.44 | 0.1 |
| **Gesamt** | **441.24** | **100.0** |

**The entire "non-GEMM, non-attention" overhead — every norm, RoPE, residual, bias, SiLU, mul, KV write, embedding and dispatch call combined — is 13.8 ms (3.1 %).** Fusing all of them perfectly would save at most 14 ms out of 441 ms. The real leverage is inside GEMM.

## Tabelle 4 — pp64 vs. pp256 Skalierung

| Operation | pp64 (ms) | pp256 (ms) | Faktor | Linear wäre 4× |
|---|---:|---:|---:|---|
| Embedding | 0.50 | 1.40 | 2.80× | sub-linear |
| norm_pre_attn | 1.15 | 1.32 | 1.15× | sub-linear (launch-bound) |
| q_proj | 4.97 | 9.70 | 1.95× | sub-linear |
| k_proj | 2.62 | 2.74 | 1.04× | sub-linear (launch-bound) |
| v_proj | 2.63 | 2.73 | 1.04× | sub-linear (launch-bound) |
| qkv_bias | 1.21 | 1.42 | 1.17× | sub-linear (launch-bound) |
| rope_q | 0.92 | 1.24 | 1.35× | sub-linear |
| rope_k | 0.54 | 0.61 | 1.13× | sub-linear (launch-bound) |
| kv_write | 1.00 | 1.07 | 1.07× | sub-linear (launch-bound) |
| attention | 4.88 | 9.73 | 1.99× | sub-linear |
| o_proj | 4.83 | 9.56 | 1.98× | sub-linear |
| residual_attn | 0.58 | 0.73 | 1.27× | sub-linear (launch-bound) |
| norm_pre_ffn | 0.71 | 0.90 | 1.27× | sub-linear (launch-bound) |
| gate_proj | 18.86 | 44.36 | 2.35× | sub-linear |
| up_proj | 18.83 | 43.86 | 2.33× | sub-linear |
| silu_mul | 1.20 | 3.81 | 3.18× | ≈ linear |
| **down_proj** | **81.74** | **303.77** | **3.72×** | **≈ linear** |
| residual_ffn | 0.61 | 0.84 | 1.38× | sub-linear (launch-bound) |
| Final Norm | 0.03 | 0.03 | 1.20× | sub-linear (launch-bound) |
| LM Head | 0.92 | 0.97 | 1.06× | sub-linear (launch-bound) |
| **Gesamt** | **149.16** | **441.24** | **2.96×** | **sub-linear** |

**Reading the scaling:**
- down_proj scales almost perfectly linearly (3.72×/4×) — it's already compute/bandwidth-bound; there's no fixed-cost to amortise, optimising it means making the kernel itself faster (better dispatch, fusion, or memory-layout work).
- gate_proj / up_proj scale 2.35× — they have a noticeable fixed-cost floor (dispatch overhead, weight prefetch) that gets amortised at higher pp.
- Everything under 2× (norms, residuals, RoPE-K, KV write, LM head, final norm, K/V projections, biases) is **launch-bound**: the op runs in ~20–100 µs regardless of seq_len. These are candidates for fusion **if** you want to pay the engineering cost, but the collective win is single-digit milliseconds.
- Total prefill scales 2.96× at 4× seq_len — well sub-linear, confirming that a per-token throughput metric (tok/s) improves with longer prompts (pp64 = 437 tok/s, pp256 = 595 tok/s in the baseline).

## Tabelle 5 — Schätzung (Phase 3d) vs. Messung (Phase 4 Step 1)

| Kategorie | Schätzung (ms) | Gemessen (ms) | Delta |
|---|---:|---:|---|
| GEMM | 110 | 417.7 | **+307.7 — underestimated by 3.8×** |
| Attention | 6 | 9.73 | +3.7 |
| Norm/RoPE/Residual "overhead" | 120 | 4.1 | **−115.9 — overestimated by 29×** |
| FP32↔FP16 Shuttling | 40 | ~0 | **not observable as a distinct op** |
| Launch/Dispatch | 25 | 0.44 | **−24.6 — overestimated by 57×** |
| Rest (Embed + Bias + SiLU + KV + LM head) | — | 8.7 | new bucket |
| **Gesamt pp256** | **~301 + 110 = ~411** | **441.24** | close (sync overhead) |

**FP32↔FP16 shuttling is not a separate op anywhere in the instrumented layer.** The WMMA attention path does convert on-stream, but that cost is absorbed inside `attention_us = 9.73 ms`. No free-standing conversion kernel is called between ops, so the 40 ms "shuttling" estimate was phantom.

The real picture is: **GEMM is under-served**. The WMMA Q4_0 prefill path lowered pp256 from ~1930 ms to ~411 ms (Phase 3) but down_proj alone is still burning 304 ms of that at only ~1.9 TFLOPS effective throughput on a 48-TFLOPS GPU. There is a >10× headroom in the FFN down kernel alone.

---

## Verdict und Empfehlung

| Kategorie | Zeit (ms) | Anteil | Optimierbar? | Erwarteter Gewinn bei 2× Speed-up |
|---|---:|---:|---|---:|
| **down_proj** (FFN) | 303.8 | 69.3 % | **ja** — largest single lever | **~152 ms saved** |
| gate_proj + up_proj (FFN) | 88.2 | 20.0 % | **ja** — fuse gate+up into one kernel | ~44 ms saved |
| q_proj + o_proj | 19.3 | 4.4 % | partial — attention-side projections | ~10 ms saved |
| attention | 9.7 | 2.2 % | already WMMA-optimised | ~5 ms saved |
| Norm/RoPE/Residual (all 13 ops) | 13.8 | 3.1 % | yes but low ceiling | ~7 ms absolute max |

**Empfehlung für Schritt 2: `down_proj` (FFN down projection)**, weil sie 69 % der Prefill-Zeit verbraucht und linear mit seq_len skaliert (3.72× von pp64→pp256). Ein 2×-Speed-up hier spart ~152 ms pro Prefill bei pp256 — mehr als alle anderen Optimierungen zusammen. Mögliche Angriffspunkte: dedizierte WMMA Q4_0-Kernel-Variante für K=11008 (aktuell läuft sie über den generischen Pfad), bessere Tile-Größen für die tiefe K-Dimension, oder eine hipBLAS-getunte Variante falls die WMMA-Seite nicht weiter skaliert.

**Empfehlung für Schritt 3: Fused gate + up projection** (beide teilen dasselbe `normed`-Input und schreiben nebeneinander in `gate` / `swiglu`), weil sie zusammen 20 % der Prefill-Zeit verbrauchen und die Fusion einen einzelnen Kernel mit doppelter arithmetischer Intensität ergibt. Erwartung: ~20–30 ms ersparnis bei pp256.

**Nicht empfohlen**: Fused Norm+RoPE oder FP16-end-to-end. Der gesamte Norm+RoPE+Residual-Block ist 4.1 ms (0.9 %), FP16-Shuttling existiert nicht als messbare separate Kategorie. Beide Optimierungen würden Engineering-Aufwand verursachen, der sich nicht rechnet, solange down_proj bei 69 % sitzt.

---

## Correctness check

`diff` der generierten Antworten mit/ohne `ROCMFORGE_PROFILE_PREFILL_OPS=1`: identisch (5 Tokens, `--temperature 0.0`). Die Instrumentierung verändert keine Ergebnisse — sie fügt nur `hipDeviceSynchronize()` zwischen Ops ein.

## Plausibility check

- Prefill-Gesamtzeit (profiled): 441.24 ms — passt zur Baseline (411 ms unprofiled) mit ~30 ms Sync-Overhead (7 %).
- Layer-Summe (438.40 ms) ≈ Top-Level-Layer-Summe (438.83 ms) — Delta 0.4 ms ist die für-Schleifen + `tracing::trace!("layer launched")`-Overhead pro Layer.
- GEMM-Summe (417.7 ms) weit über der Phase-2b-Schätzung (110 ms) — weil Phase 2b nur den isolierten GEMM-Kernel gemessen hat, nicht den realen End-to-End-Dispatch inklusive Q4_0-Dequant-Overheads.
