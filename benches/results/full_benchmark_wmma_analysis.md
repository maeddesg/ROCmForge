# 15-Prompt End-to-End Benchmark — ROCmForge vs. llama.cpp

**Date:** 2026-04-17
**Commit:** `2c28857` (Phase 3.1 padding landed)
**GPU:** RX 9070 XT (gfx1201, RDNA 4), ROCm 7.2.1
**Target model:** Qwen2.5-7B-Instruct-Q4_0
**Draft model (config B only):** qwen2.5-0.5b-instruct-q4_0
**llama.cpp build:** 408225b / b1 (ROCm HIP)
**Harness:** [`benches/bench_full_15prompts.fish`](../bench_full_15prompts.fish)
**Raw data:** [`full_benchmark_wmma_2c28857_1776430954.json`](full_benchmark_wmma_2c28857_1776430954.json)
**Answer dumps:** [`answers/`](answers/) (`rocmforge_{A,B,C}_*.txt`, `llamacpp_*.txt`, extracted completions under `answers/extracted/`)

## Configurations

| Mode | Description | Env flags |
|------|-------------|-----------|
| **A** | ROCmForge default — all WMMA + fast paths on | (none) |
| **B** | ROCmForge + speculative decoding, depth=1 | `--draft-model ... --spec-depth 1` |
| **C** | ROCmForge with every optimisation off — project-start baseline | `DISABLE_WMMA_PREFILL=1 DISABLE_WMMA_ATTENTION=1 DISABLE_HIPBLAS_PREFILL=1 DISABLE_TILED_GEMV=1 DISABLE_BATCHED_LM_HEAD=1` |
| **llama.cpp** | `llama-simple -ngl 99` raw completion | — |

15 prompts (5 code / 5 chat / 5 prose), greedy (`temperature=0.0 top_p=1.0`), 128 decoded tokens, 3 runs per cell, median reported.

## TL;DR

- **Real prompts are short:** the 15 prompts tokenise to 19–41 tokens. Every one is below ROCmForge's WMMA threshold (`seq_len ≥ 64`), so **the WMMA GEMM and WMMA attention paths never engage** in this benchmark. Modes A and C produce near-identical timings — the synthetic Phase 3d gain (pp=256 at 620 tok/s) does not translate to realistic prompt lengths.
- **Decode is stable and class-independent:** ROCmForge 102 tok/s vs. llama.cpp 121 tok/s across all three categories; ratio 85 %.
- **Prefill gap is large at these shapes:** llama.cpp 527 tok/s median vs. ROCmForge 61 tok/s — 8.7×. This is the small-seq_len regime ROCmForge has not optimised yet.
- **Speculative decoding is a loss across the board** at greedy: median 72.5 tok/s vs. baseline 102 tok/s. Acceptance rate varies by class (code 69 %, chat 58 %, prose 49 %) but the per-step overhead dominates.

## Table 1 — ROCmForge decode throughput (tok/s, median of 3)

| Prompt | Class | Prompt tokens | C (all off) | A (WMMA) | B (spec d=1) | α (B) |
|---|---|---:|---:|---:|---:|---:|
| code_01 | Code | 28 | 102.2 | 102.3 | 79.2 | 74.0 % |
| code_02 | Code | 28 | 102.2 | 102.2 | 74.5 | 62.8 % |
| code_03 | Code | 41 | 102.2 | 102.1 | 80.1 | 77.8 % |
| code_04 | Code | 23 | 102.3 | 102.3 | 71.7 | 54.2 % |
| code_05 | Code | 31 | 102.3 | 102.2 | 76.8 | 69.3 % |
| chat_01 | Chat | 22 | 102.3 | 102.3 | 75.8 | 62.8 % |
| chat_02 | Chat | 24 | 102.2 | 102.3 | 74.2 | 60.8 % |
| chat_03 | Chat | 22 | 102.3 | 102.3 | 73.6 | 58.0 % |
| chat_04 | Chat | 24 | 102.3 | 102.3 | 71.1 | 53.0 % |
| chat_05 | Chat | 24 | 102.2 | 102.3 | 67.1 | 43.8 % |
| prose_01 | Prose | 23 | 102.2 | 102.3 | 68.1 | 46.0 % |
| prose_02 | Prose | 21 | 102.3 | 102.3 | 72.5 | 54.2 % |
| prose_03 | Prose | 24 | 102.2 | 102.3 | 70.7 | 52.4 % |
| prose_04 | Prose | 19 | 102.3 | 102.2 | 70.0 | 48.8 % |
| prose_05 | Prose | 26 | 102.2 | 102.3 | 68.4 | 47.7 % |
| **Median** | | **24** | **102.2** | **102.3** | **72.5** | **54.2 %** |

A and C are identical up to noise because the WMMA path has a 64-token cut-off and the longest prompt in this set is 41 tokens.

## Table 2 — Prefill / TTFT

| Prompt | Class | Prompt tokens | TTFT C (ms) | TTFT A (ms) | A / C |
|---|---|---:|---:|---:|---:|
| code_01 | Code | 28 | 458.8 | 458.7 | 1.00× |
| code_02 | Code | 28 | 458.8 | 458.8 | 1.00× |
| code_03 | Code | 41 | 662.0 | 579.3 | 0.87× |
| code_04 | Code | 23 | 379.6 | 379.4 | 1.00× |
| code_05 | Code | 31 | 504.8 | 504.6 | 1.00× |
| chat_01 | Chat | 22 | 363.5 | 363.4 | 1.00× |
| chat_02 | Chat | 24 | 396.2 | 396.1 | 1.00× |
| chat_03 | Chat | 22 | 363.6 | 363.5 | 1.00× |
| chat_04 | Chat | 24 | 396.1 | 396.2 | 1.00× |
| chat_05 | Chat | 24 | 396.0 | 396.1 | 1.00× |
| prose_01 | Prose | 23 | 379.8 | 379.4 | 1.00× |
| prose_02 | Prose | 21 | 347.7 | 347.8 | 1.00× |
| prose_03 | Prose | 24 | 396.1 | 396.0 | 1.00× |
| prose_04 | Prose | 19 | 316.4 | 316.3 | 1.00× |
| prose_05 | Prose | 26 | 427.6 | 427.4 | 1.00× |
| **Median** | | **24** | **396.1** | **396.1** | **1.00×** |

The 41-token `code_03` is the only prompt that crosses the 64-token WMMA threshold (because the scratch rounds up to 64). Its TTFT drops 13 % — a small hint of what a lowered-threshold WMMA path could do at short seq_len. Everything else shows no change because the WMMA kernels never dispatch.

## Table 3 — ROCmForge vs. llama.cpp decode (tok/s, median)

| Class | ROCmForge A | llama.cpp | Gap |
|-------|------------:|----------:|----:|
| **Overall** | **102.3** | **121.3** | **1.19×** |
| Code  | 102.2 | 121.4 | 1.19× |
| Chat  | 102.3 | 121.3 | 1.19× |
| Prose | 102.3 | 121.3 | 1.19× |

llama.cpp prefill median: **527 tok/s**. ROCmForge A prefill median: **60.6 tok/s**. Gap at these short prompts: **8.7×**.

## Table 4 — Project start → now

Median over all 15 prompts:

| Metric | Project-start (C) | After WMMA (A) | Δ |
|--------|------------------:|---------------:|--:|
| Decode tok/s | 102.2 | 102.3 | +0.1 % |
| Prefill tok/s | 60.6 | 60.6 | ±0 |
| TTFT (ms) | 396.1 | 396.1 | ±0 |
| Spec-decode code tok/s (median) | n/a | 76.8 | — |
| Time for prefill + 128 tokens (ms) | 1,648 | 1,648 | ±0 |

The WMMA integration sits idle on 14 of 15 prompts. Config C was always passing through the `custom GEMM + scalar attention + tiled GEMV` stack at these prompt sizes; config A is theoretically the improved stack but never reaches it because of the `seq_len ≥ 64` gate in both WMMA paths.

## 4. Answer divergence

### 4.1 ROCmForge A vs. ROCmForge C

Word-level first-divergence index out of the shorter response length:

| Prompt | A vs C | Notes |
|---|---|---|
| code_01 | 98 / 98 (identical) | — |
| code_02 | 85 / 85 (identical) | — |
| code_03 | **50 / 72** | WMMA path engaged (seq=41 padded to 64); FP16 WMMA accumulation order differs from FP32 scalar, drift starts around word 50 |
| code_04 | 56 / 56 (identical) | — |
| code_05 | 67 / 67 (identical) | — |
| chat_01 through prose_05 | all identical | no WMMA dispatch |

**Exactly one prompt (`code_03`, 41 tokens) actually exercises the WMMA path** thanks to the Phase 3.1 padding to 64. Its output diverges from the scalar reference at word 50 — comfortably past the sensitive first few tokens, consistent with the numerical drift seen in the `prefill_wmma_matches_hipblas` regression test.

### 4.2 ROCmForge A vs. llama.cpp (raw completion)

| Prompt | First word diverging | Length comparable |
|---|---|---|
| code_01 | 43 | similar structure, both write `binary_search`, diverge at final `Here's a Python function` wording |
| code_02 | 1 | different framing sentence |
| code_03 | 50 | same code, different prose filler |
| code_04 | 6 | same intro, different explanation path |
| code_05 | 19 | same SQL skeleton |
| chat_01 | 8 | same Munich itinerary structure, different day labels |
| chat_02 | 98 | nearly parallel guide, ends differently |
| chat_03 | 15 | same EV-vs-hybrid pros/cons |
| chat_04 | 22 | same quantum analogies |
| chat_05 | 20 | same sourdough troubleshooting |
| prose_01 | 25 | same story setup |
| prose_02 | 17 | same ramen recipe |
| prose_03 | 31 | same Silk Road narrative |
| prose_04 | 26 | same conditioning explanation |
| prose_05 | 41 | same lighthouse scene |

Divergences follow the expected pattern for different engines on the same weights: shared start for a handful of tokens, then drift driven by FP rounding in dequant and softmax. No prompt shows a semantic break.

### 4.3 Qualitative side-by-side (first 50 words)

**`code_01` — Prompt:** *"Write a Python function that implements binary search on a sorted list and returns the index of the target element, or -1 if not found."*

ROCmForge A:
> The function should have a docstring explaining the parameters and return value. Use a for loop to iterate over the list and apply the binary search algorithm. Include a check function that tests the binary search function with various inputs, including edge cases. Use assertions to verify the correctness of

llama.cpp:
> The function should have a docstring explaining the parameters and return value. Use a for loop to iterate over the list and apply the binary search algorithm. Include a check function that tests the binary search function with various inputs, including edge cases. Here's a Python function that implements binary

First 42 words identical; both then produce the same `def binary_search(arr, target):` with the same docstring (seen in the full answer).

**`chat_01` — Prompt:** *"Help me plan a weekend trip to Munich. I like history, good food, and walking around interesting neighborhoods."*

ROCmForge A:
> I have 2 days and 3 nights to explore. Here's a rough outline of what I'd like to do: Day 1: - Visit the Old Town and Marienplatz - Have lunch at a traditional Bavarian restaurant - Walk around the Viktualienmarkt - Visit the Deutsches Museum - Have dinner at

llama.cpp:
> I have 2 days and 3 nights to explore the city. Here's a suggested itinerary for your weekend trip to Munich: **Day 1:** - **Morning:** - Start your day with a visit to the **New Town Hall (Neues Rathaus)**. This impressive Gothic-style building is a must-see and offers a great

Same opening clause, then genuinely different itineraries — both coherent, both on-topic. Normal engine-level divergence.

**`prose_01` — Prompt:** *"Write a short story about a cat who discovers a hidden garden behind an old bookshop in a small coastal town."*

ROCmForge A:
> The story should be written in third-person omniscient point of view and include descriptions of the cat's surroundings, the cat's thoughts and emotions, and the interactions between the cat and the bookshop owner. The story should have a clear beginning, middle, and end, and should convey a sense of wonder

llama.cpp:
> The story should be written in third-person omniscient point of view and include descriptions of the cat's surroundings, the cat's thoughts and emotions, and the cat's interactions with the shop owner. The story should have a clear beginning, middle, and end, and should convey a sense of wonder and discovery.

First 25 words identical. The model (same weights in both engines) treats the prompt as a creative-writing brief rather than launching into the story — so both engines produce a meta-plan first.

## 5. Where the gap actually sits

Breaking down a typical 24-token median prompt at 128 decoded tokens:

```
ROCmForge total: 396 ms prefill + 1,252 ms decode = 1,648 ms
llama.cpp total: 46 ms prefill + 1,058 ms decode = 1,104 ms
Gap: 544 ms, dominated by 350 ms of prefill difference.
```

At these shapes, **prefill is ~24 % of wall-clock time on ROCmForge but only ~4 % on llama.cpp**. Closing the gap now requires:

1. **Lower the WMMA dispatch threshold** — the Phase 3.1 padding machinery is already in place. Changing `seq_len >= 64` to `seq_len >= 16` (padding up to 64) would engage WMMA on every prompt in this set. The synthetic benchmark measured 469 tok/s at pp=65 (padded from scratch); at pp=24 padded to 64 the gain would be smaller per token but still multi-fold over 60 tok/s.
2. **Prefill-path kernel fusion** — the 295 ms of unfused norm + RoPE + residual per pp=256 stays proportional at shorter lengths. Fusing `norm + QKV + RoPE` on the prefill path (the decode path already has it) would cut ~30 % of every prefill dispatch.
3. **Decode fusion** — the last 15 % of the decode gap (102 vs. 121 tok/s) is likely residual launch overhead and unfused elementwise ops on the decode path, not a kernel-quality issue.

Speculative decoding is a pure loss here. The per-step verify cost is ~9 ms for the target, ~3 ms for the draft, plus overhead; at α < 80 % the amortised tokens/step are below 2, which is not enough to recover. Code prompts (α median 69 %, best 78 %) are closest to break-even but still 25 % below greedy. Needs a path the current prompt doesn't request.

## 6. Report

15-Prompt End-to-End-Benchmark nach WMMA-Meilenstein (Qwen2.5-7B Q4_0, RX 9070 XT):

| Metrik | Projekt-Start (C) | Nach WMMA (A) | Veränderung |
|--------|------------------:|--------------:|------------:|
| Decode tok/s (Median) | 102.2 | 102.3 | ±0 |
| Prefill tok/s (Median) | 60.6 | 60.6 | ±0 |
| TTFT ms (Median) | 396 | 396 | ±0 |
| Gesamt-Zeit 128 Tokens (Median) | 1,648 ms | 1,648 ms | ±0 |

llama.cpp Decode: 121.3 tok/s (ROCmForge: 102.3 tok/s, Lücke 1.19×).
llama.cpp Prefill: 527 tok/s (ROCmForge: 61 tok/s, Lücke 8.7×).

**Antwort-Qualität:** ROCmForge A vs. C identisch bei 14/15 Prompts; `code_03` (41 Tokens → WMMA via Padding) divergiert ab Wort 50 durch FP16-WMMA-Akkumulationsunterschied. ROCmForge vs. llama.cpp: semantisch äquivalente Antworten, Drift durch Engine-Unterschiede nach typischerweise 5–50 Wörtern.

**Nächster Engpass:** der 64-Token-Schwellwert in der WMMA-Dispatch. Realistische Prompts tokenisieren fast alle unter 64, sodass die gesamte Phase-3d/3.1-Arbeit brach liegt. Das Padding-Gerüst existiert bereits (Phase 3.1); den Threshold z.B. auf 16 abzusenken und den Rest der 64 zu padden ist die direkt anschließbare Maßnahme.
