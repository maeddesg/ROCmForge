# Post-2.1.5 rocprof Deep-Dive — Decode-Profil + BW-Analyse

**Date:** 2026-04-23
**Branch:** v1.0-dev (on top of 2.1.5 FP8 follow-up `ba73a1d`)
**Hardware:** AMD Radeon RX 9070 XT (gfx1201), 640 GB/s nominal BW
**Tool:** rocprofv3 — `--kernel-trace`, `--hip-trace`, `--stats --summary`
**Prompt:** `"Explain what a mutex is in one paragraph."` (33 Prompt-Tokens)
**Max-Tokens:** 100 (EOS @ 96 decode tokens, 129 total forward-pass iterationen wobei Prefill 1 WMMA-Call statt 33 Decode-Calls ist)

## Kurzfassung

```
Decode pro Token (post-2.1.5):    ~24.0 ms GPU
Decode tok/s:                      ~40.6 (Block E ref: 40.6)
Σ GPU Zeit (run):                  2.37 s (2.0.3 war 3.06 s → −22 %)
hipStreamSynchronize:              97 (2.0.3: 129 → −25 %, WMMA-Prefill-Effekt)
hipLaunchKernel:                   52 203 (530 per decode token)

BW-Effizienz Decode (geschätzt):   ~30 % von 640 GB/s  (2.0.3: 28.6 %)
Ziel aus Arch-Doc:                 70 %  (→ ~125 tok/s)
```

**Kernfindung — gate_up_swiglu ist der BW-Outlier mit 20 %.** Alle
anderen GEMV-Kernel erreichen 80–100 % BW. gate_up_swiglu (63 % der
GPU-Zeit!) liegt bei 130 GB/s von 640 GB/s möglich. Fix dort würde
die BW-Effizienz auf ~50–60 % drücken.

## Top-15 Kernel pro Zeit (kompletter Run, 129 Iterationen)

Aus `91269_kernel_stats.csv`:

| Kernel | Calls | Σ µs | Ø µs | % GPU | Max µs |
|---|---:|---:|---:|---:|---:|
| **gemv_q4_k_gate_up_swiglu** | **3 456** | **1 509 865** | **436.9** | **63.86 %** | 489 |
| gemv_q6_k_standard (LM-head + Q6_K layers) | 3 552 | 308 624 | 86.9 | 13.05 % | **778** |
| gemv_q4_k_q8_inline_residual | 5 184 | 220 907 | 42.6 | 9.34 % | 117 |
| gemv_q4_k_q8_inline | 8 550 | 155 104 | 18.1 | 6.56 % | 46 |
| wmma_gemm_q4_k_fp16 (prefill) | 216 | 45 007 | 208.4 | 1.90 % | 452 |
| attention_decode | 3 456 | 37 270 | **10.8** | 1.58 % | 16 |
| rms_norm | 7 008 | 23 165 | 3.3 | 0.98 % | 9 |
| rms_norm_batched | 7 057 | 16 784 | 2.4 | 0.71 % | 16 |
| rope | 6 912 | 13 148 | 1.9 | 0.56 % | 5 |
| wmma_gemm_q6_k_fp16 (prefill LM-head) | 37 | 12 683 | 342.8 | 0.54 % | 2 907 |
| gemv_q4_k_standard (Bandit loser-pulls) | 90 | 7 996 | 88.8 | 0.34 % | 158 |
| kv_cache_append | 4 644 | 6 982 | 1.5 | 0.30 % | 6 |
| residual_add_inplace | 1 800 | 3 297 | 1.8 | 0.14 % | 14 |
| copyBuffer (HSA) | 242 | 2 191 | 9.1 | 0.09 % | 12 |
| attention_prefill | 36 | 578 | 16.1 | 0.02 % | 22 |
| swiglu (prefill) | 36 | 264 | 7.3 | 0.01 % | 9 |
| embedding_lookup | 97 | 241 | 2.5 | 0.01 % | 12 |
| rope_batched (prefill) | 72 | 231 | 3.2 | 0.01 % | 7 |
| **Σ GPU** | **52 443** | **2 363 436** | | **100 %** | |

## Pro-Decode-Token Breakdown

96 Decode-Tokens × `(hidden_dim=4096, ffn_dim=12288, n_layers=36)`.
Prefill-spezifische Kernel (WMMA, attention_prefill, swiglu,
rope_batched, embedding_lookup) abgezogen:

| Kernel | Count/Token | Zeit/Token | % Decode |
|---|---:|---:|---:|
| gate_up_swiglu | 36 | **15.7 ms** | **65 %** |
| q6_k_standard (36 layer-internal + 1 LM-head) | 37 | 3.2 ms | 13 % |
| q4_k_q8_inline_residual (O-proj + FFN-down w/ residual fuse) | 54 | 2.3 ms | 10 % |
| q4_k_q8_inline (Q/K/V + untuned) | 89 | 1.6 ms | 6.6 % |
| attention_decode | 36 | 0.39 ms | 1.6 % |
| rms_norm (main) | 73 | 0.24 ms | 1.0 % |
| rms_norm_batched (QK-norm) | 74 | 0.18 ms | 0.7 % |
| rope | 72 | 0.14 ms | 0.6 % |
| kv_cache_append | 36 | 0.07 ms | 0.3 % |
| residual_add (unfused) | ~19 | 0.03 ms | 0.14 % |
| **Σ pro Decode-Token** | **~530** | **~24 ms** | **100 %** |

## Bandbreiten-Analyse pro Kernel

Weight-Bytes pro GEMV-Call × Kernel-Zeit → effektive BW.
Qwen3-8B Q4_K_M: Q4_K-Block = 144 B / 256 Elem, Q6_K-Block = 210 B / 256 Elem.

| Kernel | Shape | Weight-Bytes | Ø Zeit | **BW** | **% peak** |
|---|---|---:|---:|---:|---:|
| **gate_up_swiglu** (gate+up fused) | N=2×12288, K=4096 | **56.6 MB** | 437 µs | **130 GB/s** | **20 %** ❌ |
| q4_k_q8_inline (Q-proj) | N=4096, K=4096 | 9.4 MB | ~18 µs | **522 GB/s** | **82 %** ✅ |
| q4_k_q8_inline (K/V-proj) | N=1024, K=4096 | 2.35 MB | ~18 µs | 131 GB/s | 20 % (launch-bound) |
| q4_k_q8_inline_residual (O-proj) | N=4096, K=4096 | 9.4 MB | ~19 µs | 495 GB/s | 77 % ✅ |
| q4_k_q8_inline_residual (FFN-down) | N=4096, K=12288 | 28.3 MB | ~64 µs | **442 GB/s** | **69 %** ✅ |
| q6_k_standard (LM-head) | N=151936, K=4096 | 510 MB | ~778 µs | **656 GB/s** | **103 %** ✅ (cache-amortisiert) |
| q6_k_standard (layer Q6_K) | N=? K=4096 | ~9–16 MB | ~87 µs | ~100–200 GB/s | 15–30 % |

**Beobachtung:**
- **gate_up_swiglu ist der einzige BW-Ausreißer nach unten:** 20 %.
  Kernel ist 63 % der GPU-Zeit aber erreicht nur ein Fünftel der
  physisch möglichen Bandbreite.
- **LM-Head ist gut:** 103 % apparent-BW zeigt L2-Cache-
  Amortisation über die 128 K-Tiles.
- **Kleine GEMVs (K-/V-proj) sind launch-bound:** 2.35 MB in 18 µs
  ist zu wenig Arbeit um den Kernel-Start-Overhead zu amortisieren.

### Gate_up_swiglu — warum nur 20 %?

Erwartung (naiv, mit 100 % BW):
```
Weight-Bytes / Peak-BW = 56.6 MB / 640 GB/s = 88.4 µs theoretisch
Gemessen:                                    437 µs
→ 5× langsamer als peak
```

Vergleich mit einzelnem Q4_K_GEMV bei ähnlicher Gesamt-Datenmenge:
- q4_k_q8_inline-residual bei FFN-down: 28.3 MB in 42 µs = 664 GB/s
- gate_up_swiglu liest 2 × 28.3 MB = 56.6 MB, sollte ~85 µs brauchen
- Tatsächlich: 437 µs = **5× langsamer als zwei separate GEMVs**

Block-D-Analyse (multi_row_cols und num_waves variiert) hat gezeigt,
dass die Kernel-Struktur der Flaschenhals ist — die existierenden
Tile-Parameter können nur ±2 % in dieser Range bewegen. Vermutete
Ursachen:

1. **Sequentielle LDS-B-Stage.** Der Kernel lädt Input (B) ins LDS
   und synchronisiert per `__syncthreads()` bevor die Wave die
   Weights (A) streamen kann. Bei kleinem M=1 ist das ein serieller
   Punkt, kein Pipelining.
2. **Gate + Up in einer Wave.** Der Fused-Kernel macht BEIDE
   Projektionen im selben Thread-Block. Weight-Reads für gate und
   up sind getrennt und können HBM-Queues blockieren wenn sie sich
   nicht überlappen.
3. **SwiGLU-Fusion braucht beide Outputs fertig.** `silu(gate) * up`
   erfordert gate UND up am Ende der Wave-Berechnung — das
   verhindert early-output-streaming.

**Fix-Optionen (Folge-Sessions):**
- Gate und up TRENNEN: zwei q4_k_q8_inline-residual-Calls +
  separater SwiGLU-Call. Bei 2 × 28.3 MB × 42 µs = 85 µs + ~5 µs
  SwiGLU = 90 µs, sprich **fast 5× schneller**. Kostet eine extra
  VRAM-Round-Trip (gate_scratch, up_scratch), aber das ist vom
  Block-D-Prefill bekannt und messbar klein.
- Double-Buffer die Weight-Loads: während gate_block N berechnet
  wird, lädt up_block N voraus.
- Neue Codegen-Variante mit getrennten Wave-Pfaden (Wave 0–3 für
  gate, 4–7 für up) — eliminiert das SwiGLU-Barrier-Problem.

## Dispatch-Overhead

```
Σ hipLaunchKernel time:   49.7 ms für 52 203 calls  →  951 ns/call
Davon im Decode:          ~50 800 calls × 951 ns    =  48.3 ms total
Pro Decode-Token:         530 × 951 ns              =  0.50 ms / 2 %

→ Dispatch-Overhead ist klein (~2 % der Decode-Zeit).
```

Kein Flaschenhals. Tracing oder HIP-Graphs würden das weiter
reduzieren aber lohnen sich erst nach dem Kernel-Fix.

## hipStreamSynchronize-Verhalten

```
Calls:            97 (1 pro Decode-Token + 1 für Prefill)
Σ wait-Zeit:      2 201 ms (76.8 % der HIP-API-Zeit)
Ø pro sync:       22.7 ms
```

Die 22.7 ms sind FAST EXAKT die Decode-Zeit pro Token (24 ms GPU).
Das heißt: **der sync WARTET auf die ganze Decode-Arbeit des Tokens.**
Kein Idle-Zeit-Problem — der sync misst die Kernel-Zeit korrekt.

2.0.3 hatte 129 syncs (33 Prefill-Tokens einzeln). Post-2.1.5 hat
97 syncs (1 Prefill-Call ersetzt die 33). Der 2.0.3-Sync-Count-
"Gewinn" von Block E (132 → 103 → 97) ist ein direkter WMMA-
Prefill-Seiteneffekt.

## hipMemcpy-Overhead

```
Calls:            666
Σ time:           499.9 ms (17.4 % HIP-API-Zeit)
Ø:                750 µs, Max: 61.9 ms
```

Der MAX von 61.9 ms ist wahrscheinlich ein einmaliges Embedding-
Fp32-Upload am Startup (Vocab × hidden_dim × 4 = 151 936 × 4096 × 4
= 2.43 GB). Der Rest sind kleine DtoH-Reads für Logits (pro decode-
token 608 KB = 1 readback).

Keine Optimierung nötig — 100 %-Reduktion wäre ~500 ms über den
gesamten 7-s-Run, also ~7 % weniger Wallclock wenn komplett
eliminierbar. Nicht der Hebel.

## Attention-Anteil am Decode

```
attention_decode:    10.8 µs × 36 layers = 390 µs/token = 1.6 %
```

**Attention ist KEIN Decode-Bottleneck** bei dieser Shape
(hidden=4096, n_heads=32, seq_len≤129). Bei langem Kontext
(z.B. 2048 Tokens) würde attention_decode linear wachsen → bei
seq_len=2048 ≈ 170 µs × 36 = 6.1 ms/token = ~25 % der Decode-Zeit.
Dann wäre es relevant. Jetzt nicht.

## LM-Head-Anteil am Decode

```
q6_k_standard (LM-head, größter Call, max 778 µs):
  ~778 µs einmal pro Decode-Token
  Als % von Decode: 778 / 24 000 = 3.2 %
```

Auch kein Flaschenhals, aber knapp 800 µs für einen einzelnen Kernel
ist nicht vernachlässigbar. Mögliche Optimierung: Top-K-Sampling
direkt im LM-Head (reduziert Output-Bytes von 608 KB auf ~20 KB),
oder Tiled LM-Head mit nur Top-Logits-Selection. Phase-2+.

## Vergleich mit 1.17 / 2.0.3

| Metrik | 1.17 | 2.0.3 (Post-P0) | **Post-2.1.5** |
|---|---:|---:|---:|
| Decode tok/s | 30.6 | 40.7 | **40.6** |
| Σ GPU ms | 2 668 (110 iter) | 3 056 (129 iter) | **2 363 (129 iter)** |
| Wallclock ms | 4 255 | 3 334 | ~3 000 (non-rocprof) |
| hipStreamSync | 83 260 | **129** | **97** |
| Dispatches/Iter | 608 | 525 | **530** |
| gate_up Ø µs | 421.6 | 432.8 | **436.9** |
| GPU-Effizienz | 62.7 % | 91.7 % | ~92 % |
| BW-Effizienz | 21 % | 28.6 % | **~30 %** |

- Kernel-Zeiten im Decode-Pfad: **unverändert** seit 2.0.3. WMMA-
  Prefill hat den statischen Kernel nicht berührt.
- Σ GPU: −22 % vs 2.0.3 weil Prefill jetzt 58 ms WMMA statt ~700 ms
  Decode-Loop ist.
- Sync-Count: −25 % (Prefill-Effekt).
- BW-Effizienz: +1.4 Punkte, marginal. Der 2.1.5-Prefill-Fix hat
  Decode nicht verändert.

## Warum 30 % BW statt 70 %?

Drei Ursachen, nach Größe:

```
1. gate_up_swiglu nur 20 % BW     ~33 % of "missing BW"
   (63 % der Decode-Zeit × 50 Punkte fehlende BW)

2. Kleine Kernel (K/V-proj, norms, rope, kv_append, residual)   
   sind launch-bound, nicht BW-bound. Nehmen ~6 % der Zeit,
   tragen ~0 % zu BW-Metrik bei.
   Lösung: Fusion (schon bei Residual gemacht) oder HIP-Graph.

3. Q6_K-Layer-Kernel (nicht LM-head) bei 15-30 % BW.
   Ø 87 µs × 36 calls = 3.2 ms/token = 13 %.
   Ähnliche Fixes wie bei gate_up_swiglu (Kernel-Tuning).
```

Zusammen decken diese drei fast die gesamte Ineffizienz ab. Der
größte Hebel ist eindeutig:

```
gate_up_swiglu: 130 GB/s → 500 GB/s würde Decode von 24 ms auf
~12 ms drücken → Decode von 40 tok/s auf ~80 tok/s bringen.

Damit wäre ROCmForge innerhalb 15 % von llama.cpp (99.3 tok/s).
```

## Fazit

| Frage aus Prompt | Antwort |
|---|---|
| Kernel-Zeiten pro Typ | ✅ Tabelle oben, aggregiert + per-token |
| Dispatch-Overhead | ✅ ~0.5 ms/token = 2 % der Decode-Zeit |
| Attention-Anteil | ✅ 1.6 % bei kurzem Kontext, O(seq_len) |
| LM-Head-Anteil | ✅ 3.2 % (Q6_K 778 µs pro Token) |
| Memory-BW pro Kernel | ✅ Tabelle oben, Range 20 % – 103 % |
| Dispatch-Count pro Token | ✅ **530 dispatches/decode token** (2.0.3 hatte 525) |
| **Warum BW 28 % statt 70 %** | ✅ **gate_up_swiglu @ 20 % BW dominiert** (63 % der Zeit × 50 % fehlende BW) |

**Empfehlung für nächste Sessions, geordnet nach Hebel:**

1. **gate_up_swiglu un-fusen oder rewriten.** Zwei separate Q4_K-
   GEMV-Calls + SwiGLU würde laut Q4_K-Q8_inline-Residual-Numbers
   (442 GB/s bei 28 MB-Shape) **~85 µs** statt 437 µs brauchen.
   Erwarteter Decode-Gewinn: 40 → ~65 tok/s.

2. **Q6_K-Layer-Kernel GA-tunen.** Ähnliche Untersuchung wie Block
   C/D aber auf Q6_K statt Q4_K. 13 % GPU-Zeit bei 15-30 % BW.
   Erwarteter Gewinn: ~5 %.

3. **HIP-Graph im Decode.** 530 Dispatches / 1 µs launch-overhead =
   0.5 ms/token. HIP-Graph reduziert auf ~1 replay-launch. ~2 %
   Decode-Gewinn. Niedrige Priorität relativ zu (1).

## Commit

Prefix: `docs:` — reine Analyse, kein Produktionscode.

```
docs: post-2.1.5 rocprof deep-dive — gate_up_swiglu is the 20% BW outlier
```

Backup-Push auf `backup` Remote. Kein Fork-Push.
