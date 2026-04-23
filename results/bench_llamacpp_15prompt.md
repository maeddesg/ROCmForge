# llama.cpp 15-Prompt Benchmark (RX 9070 XT, gfx1201)

**Date:** 2026-04-23
**Tool:** /home/maeddes/tmp/llama.cpp/build-rocm/bin/llama-completion
**Build:** 23b8cc4
**Sampling:** greedy (temp=0, seed=42), GPU-only (-ngl 99)
**Suite:** /home/maeddes/projects/ROCmForge/benches_v1/inference_test_prompts_15.json (same 15 prompts ROCmForge uses)

## Qwen3-8B-Q4_K_M.gguf

| # | Name | max_tok | Prefill tok/s | Decode tok/s | Prefill tok | Decode tok | Wall ms |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | Greeting | 64 | 315.02 | 97.64 | 9 | 63 | 1648.11904 |
| 2 | Simple Sequence | 64 | 565.59 | 98.01 | 18 | 63 | 1635.391488 |
| 3 | Prime Check (Python) | 256 | 629.86 | 98.53 | 20 | 255 | 3600.967168 |
| 4 | LRU Cache (C++) | 512 | 1018.70 | 99.51 | 36 | 511 | 6173.942272 |
| 5 | REST API (Go) | 1024 | 1296.03 | 98.52 | 51 | 1023 | 11485.023744 |
| 6 | Mutex Explanation | 128 | 563.19 | 99.07 | 18 | 127 | 2281.76768 |
| 7 | TCP vs UDP | 512 | 855.51 | 99.56 | 28 | 511 | 6168.328448 |
| 8 | GPU Architecture Blog Post | 1024 | 1298.74 | 99.18 | 47 | 1023 | 11422.430464 |
| 9 | Binary Search Complexity | 256 | 596.66 | 100.02 | 19 | 255 | 3565.052672 |
| 10 | Debug Code | 256 | 960.21 | 100.13 | 34 | 255 | 3562.95168 |
| 11 | Distributed Message Queue | 1024 | 1303.11 | 99.32 | 51 | 1023 | 11404.410368 |
| 12 | Long System Prompt + Question | 256 | 2952.49 | 100.12 | 188 | 255 | 3595.498496 |
| 13 | Long Output Story | 512 | 1264.84 | 99.85 | 49 | 511 | 6160.328448 |
| 14 | Arithmetic (Q4_K Precision) | 64 | 566.47 | 100.22 | 18 | 63 | 1631.412736 |
| 15 | Emoji/Special Characters | 128 | 718.86 | 100.07 | 23 | 127 | 2265.992704 |
| **Agg** | **Total** | — | **1127,2** | **99,3** | **609** | **6065** | **76601.617408** |

## Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf

| # | Name | max_tok | Prefill tok/s | Decode tok/s | Prefill tok | Decode tok | Wall ms |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | Greeting | 64 | 393.66 | 104.73 | 11 | 9 | 2312.71424 |
| 2 | Simple Sequence | 64 | 635.73 | 103.98 | 19 | 28 | 1381.55904 |
| 3 | Prime Check (Python) | 256 | 735.29 | 104.68 | 22 | 255 | 3584.840448 |
| 4 | LRU Cache (C++) | 512 | 1140.52 | 105.60 | 38 | 424 | 5166.36544 |
| 5 | REST API (Go) | 1024 | 1419.12 | 104.71 | 53 | 1023 | 11002.246144 |
| 6 | Mutex Explanation | 128 | 668.67 | 105.14 | 20 | 115 | 2226.765056 |
| 7 | TCP vs UDP | 512 | 972.23 | 105.22 | 30 | 511 | 6024.124928 |
| 8 | GPU Architecture Blog Post | 1024 | 1310.79 | 104.84 | 49 | 1023 | 10981.848832 |
| 9 | Binary Search Complexity | 256 | 700.82 | 105.35 | 21 | 255 | 3556.214272 |
| 10 | Debug Code | 256 | 1079.91 | 105.47 | 36 | 255 | 3585.765376 |
| 11 | Distributed Message Queue | 1024 | 1423.58 | 104.81 | 53 | 1023 | 10986.911232 |
| 12 | Long System Prompt + Question | 256 | 3145.49 | 105.44 | 190 | 255 | 3595.366144 |
| 13 | Long Output Story | 512 | 1372.33 | 105.45 | 51 | 511 | 6041.074688 |
| 14 | Arithmetic (Q4_K Precision) | 64 | 603.64 | 104.59 | 18 | 8 | 1250.784768 |
| 15 | Emoji/Special Characters | 128 | 1002.94 | 104.12 | 31 | 127 | 2365.93792 |
| **Agg** | **Total** | — | **1245,7** | **105,0** | **642** | **5822** | **74062.518528** |


## Vergleich ROCmForge v1.0 vs llama.cpp

Dieselben 15 Prompts, derselbe Modell-File (Qwen3-8B-Q4_K_M.gguf),
dieselbe Hardware (RX 9070 XT / gfx1201), derselbe Build-Typ
(ROCm/HIP, greedy sampling).

ROCmForge-Zahlen aus `results/phase2_step_2.1.5_15prompt_wmma.md`
(aggregate, Post-WMMA-Prefill).

### Qwen3-8B (15-Prompt Aggregat)

| Metrik | ROCmForge v1.0 | llama.cpp (23b8cc4) | Ratio |
|---|---:|---:|---:|
| **Prefill tok/s** | 590.3 | **1 127.2** | **1.91× slower** |
| **Decode tok/s** | 39.8 | **99.3** | **2.49× slower** |
| Prefill tok total | 835 | 609 | — |
| Decode tok total | 5 929 | 6 065 | — |
| Wallclock (ms) | 150 452 | 76 602 | **1.96× slower** |
| 15/15 ran | ja | ja | — |

*Prefill-Token-Unterschied:* ROCmForge zählt Prompt-Tokens nach
`apply_chat_template()` (fügt `<|im_start|>` / `<|im_end|>` /
`system` wrapper hinzu); llama.cpp zählt die nackten Prompt-
Tokens in `-p` (kein Chat-Template-Overhead in dieser Zählung).

*Decode-Token-Unterschied:* beide sind Greedy bis EOS oder
`max_tokens`; unterschiedliche Abbruchzeitpunkte erklären die
~2% Differenz.

### Llama-3.1-8B (15-Prompt Aggregat)

| Metrik | llama.cpp | Bemerkung |
|---|---:|---|
| Prefill tok/s | **1 245.7** | Marginally besser als Qwen3 (tiefere KV-Heads) |
| Decode tok/s | **105.0** | +6 % über Qwen3 |
| Wallclock (ms) | 74 063 | 2.5 s schneller als Qwen3 |

ROCmForge hat aktuell keinen Llama-3.1-End-to-End-Pfad in der
Test-Suite dokumentiert (Qwen3 ist Phase-1-Referenz). Ein
Llama-Vergleich wird nachgezogen sobald ROCmForge's
`--inference-test` mit dem Llama-Modell durchgelaufen ist.

### Zusammenfassung

Nach 2.1.5 (WMMA-Prefill) steht der Gap auf:
```
                ROCmForge  llama.cpp  Gap
Qwen3 Prefill    590.3     1 127.2    1.91×
Qwen3 Decode      39.8        99.3    2.49×
```

Der Prefill-Gap ist nach dem WMMA-Fix **auf ~2× geschrumpft**
(vor 2.1.5 war es ~27×). Der Decode-Gap von ~2.5× bleibt
unverändert seit Block E — das deckt sich mit dem Block-D/E-
Befund dass der `gate_up_swiglu`-Kernel und andere GEMV-Ops
nicht der Decode-Flaschenhals sind; die Lücke sitzt in der
Kombination aus Attention, LM-Head, HIP-Graph-Fehlen und
Per-Kernel-Launch-Overhead.

## Raw-log archive
Per-prompt stderr + stdout captured under `/tmp/llamacpp_bench_$fish_pid/`.

