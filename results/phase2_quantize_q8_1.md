# quantize_q8_1 Portierung (Schritt 1/3 llama.cpp Kernel-Port)

**Date:** 2026-04-24
**Branch:** v1.0-dev
**Hardware:** AMD Radeon RX 9070 XT (gfx1201, RDNA4), 640 GB/s nominal BW
**Reference:** llama.cpp `ggml-cuda/quantize.cu:4-48`, `ggml-common.h:248-259`
**Analysis input:** `results/phase2_llamacpp_kernel_analysis.md`

## TL;DR

Q8_1 activation pre-quantizer ist portiert, tested, und in den Decode-Pfad
verdrahtet (opt-in via `ROCMFORGE_Q8_1_PREQUANT=1` oder
`executor.set_q8_1_prequant(true)`). Der Buffer ist in dieser Session
**write-only** — noch kein Kernel konsumiert ihn. Kernel-Zeit auf 4096
Floats: **3.58 µs/Call** (Budget 10 µs). Decode-Regression auf 20 Tokens
mit aktivem Pre-Quant: **Δ −0.68 %** (innerhalb Messrausschen, greedy
Output bit-identisch).

Nächster Schritt: MMVQ-Kernel (`mul_mat_vec_q<Q4_K, Q8_1>`) portieren und
als neue Bandit-Variante registrieren. Der Q8_1-Puffer ist jetzt bereit
als Input.

## Q8_1 Block-Struct

| Feld | Offset | Größe | Bedeutung |
|---|---:|---:|---|
| `ds[0]` (half `d`) | 0 | 2 B | Scale = max(|xi|) / 127 |
| `ds[1]` (half `s`) | 2 | 2 B | ≈ d · Σ qs[i] (siehe Hinweis) |
| `qs[0..32]` (i8) | 4 | 32 B | Quantisierte Werte |
| **Total** | | **36 B** | |

Gegenüber unserer `Q8_0_block_q4k_inline` (34 B, `half d + i8[32]`) kommt
nur das `s`-Feld hinzu.

**Abweichung vom Prompt-Test „sum = Σ qs exakt (Integer)":** llama.cpp
speichert stattdessen die warp-reduzierte **Float-Summe der Original-
Inputs** `Σ xi` (`quantize.cu:33,47`). Das stimmt bis auf Rundung mit
`d · Σ qs[i]` überein — genau das dokumentiert `ggml-common.h:253`:
*"s = d * sum(qs[i])"*. Um **bit-identisch** zu llama.cpp zu bleiben
(damit der spätere MMVQ-Port Parity hat), folge ich dieser Semantik. Der
Test wurde von „exakte Integer-Gleichheit" auf „|s − d·Σqs| ≤ Rundungs-
Toleranz" umgeschrieben (siehe `test_quantize_q8_1_hidden_dim_4096`).

## Kernel-Implementierung

| | |
|---|---|
| Quelle | `~/tmp/llama.cpp/ggml/src/ggml-cuda/quantize.cu:4-48` |
| Port | `hip_kernels_v1/quantize/quantize_q8_1.hip` |
| Build-Eintrag | `hip_kernels_v1/CMakeLists.txt:52` |
| LOC | 97 (incl. License-Kopf, Doc-Block, Struct-Def) |
| Thread-Layout | Block 256 = 8 Warps × 32 Lanes; 1 Warp ↔ 1 Q8_1-Block |
| Reduktion | `__shfl_xor` über width=WARP_SIZE=32 (RDNA4 wave32) |
| OOB-Handling | Threads jenseits `n_elements` schießen mit `xi=0` mit; |
| | erst NACH der Warp-Reduce wird der Write-back übersprungen |

Rust-Seite: `src_v1/backend/gpu/quantize.rs` (`BlockQ81 repr(C)` mit
`static_assert size_of == 36`, `QK8_1 = 32`, FFI `extern "C" fn
rocmforge_launch_quantize_q8_1`).

Build-integration in `build.rs:400` (`"v1_quantize_q8_1"` in der Liste
der statisch zu linkenden Archives).

## Korrektheit (`cargo test -- --test-threads=1`)

| Test | Ergebnis |
|---|---|
| `test_block_q8_1_size` (CPU) | ok — size = 36 |
| `test_block_q8_1_field_offsets` (CPU) | ok — ds@0, qs@4 |
| `test_qk8_1_constant` (CPU) | ok — QK8_1 = 32 |
| `test_quantize_q8_1_basic` (GPU) | ok — xi=1..32 → d=32/127, qs[31]=127, qs[0]=4, monoton |
| `test_quantize_q8_1_zeros` (GPU) | ok — all-0 Input → d=0, qs=0, s=0 |
| `test_quantize_q8_1_negative` (GPU) | ok — symmetric ±15 → s≈0, sign split korrekt |
| `test_quantize_q8_1_hidden_dim_4096` (GPU) | ok — 128 Blöcke, kein NaN/Inf, |s − d·Σqs| ≤ Toleranz |
| `test_q8_1_vs_q8_0_qs_and_scale_identical` (GPU) | ok — 8 Blöcke, d + qs bit-identisch zu CPU Q8_0 Referenz |
| `test_quantize_q8_1_fast_4096` (GPU) | ok — 3.58 µs/Call |
| `test_q8_1_buffer_reusable_across_launches` (GPU) | ok — 5 Seeds × 4 Blöcke, alle Parity mit Q8_0 Referenz |
| `test_quantize_q8_1_rejects_bad_sizes` (GPU) | ok — n=0 und n=31 → hipErrorInvalidValue |
| `test_decode_with_q8_1_prequant_no_regression` (GPU, real model) | ok — Output bit-identisch, Δ −0.68 % |

**12/12 grün.**

## Performance

Gemessen mit `HipEvent::elapsed_ms` über 100 Back-to-Back-Launches auf
`hidden_dim = 4096` (Qwen3-8B, Layer-0 Embedding).

```
quantize_q8_1 on 4096 floats: 3.583 µs/call (mean over 100 runs)
```

Pro Token (einmal pro Decode-Step): **3.58 µs**.
Als % der Qwen3-8B-Token-Zeit (~17.8 ms/Token @ 56 tok/s):
**0.020 %** — vernachlässigbar.

**Budget-Headroom:** 6.4 µs. Der Kernel ist launch-bound (Grid 16×256,
256 elements per warp-iter, keine Skalar-Überraschungen); die Messung
passt zu einer Full-Queue-Latency von ~3 µs plus minimaler Compute.

## Decode-Integration

Verdrahtet via Feature-Flag (off by default):

* **Field** (executor.rs): `q8_1_buffer: Option<HipBuffer>`,
  `q8_1_prequant_enabled: bool` (initialisiert aus
  `ROCMFORGE_Q8_1_PREQUANT=1`).
* **Method** `prequantize_embedding_row_q8_1(token_id)`:
  lazy-allokiert `q8_1_buffer`, berechnet
  `input_ptr = embedding_fp32 + token_id × hidden_dim × 4` und dispatcht
  `rocmforge_launch_quantize_q8_1`.
* **Call-Site** (execute_decode, top): gated durch
  `self.q8_1_prequant_enabled`. Läuft VOR jeder HIP-Graph-Entscheidung
  und damit außerhalb jedes Capture-Blocks. Funktioniert für alle drei
  Pfade (Graph-Replay, Graph-Capture, Legacy-Dispatch).
* **Public Toggle**: `executor.set_q8_1_prequant(bool)` für Tests.

Buffer-Lifecycle:

* Allokation: einmalig lazy bei erstem Call, `(hidden_dim/32) × 36` Bytes
  — für Qwen3-8B (`hidden_dim=4096`): **128 × 36 = 4 608 Bytes**. Winzig.
* Deallokation: `Drop` läuft mit dem Executor.
* Pointer stabil über Lifetime des Executors — bereit für die nächste
  Session, in der der MMVQ-Kernel den Puffer pro GEMV liest.

**Was der Call heute macht:** quantisiert die FP32-Embedding-Row des
aktuellen Tokens in den persistenten Q8_1-Puffer. Dieser Puffer wird
von keinem Kernel konsumiert → reiner Side-Effekt mit Kernel-Overhead
~3.6 µs/Token.

**Was der Call in Schritt 2/3 machen wird:** bleibt gleich, aber der
neu portierte MMVQ-Kernel für Layer-0 QKV liest den Puffer als Q8_1-
Input. Die übrigen Layer-Aktivierungen (Output von Attention, FFN Gate/
Up, FFN Down Input) brauchen jeweils eigene Pre-Quant-Calls PRO GEMV —
das ist Schritt 2/3.

### Smoke-Test Ergebnisse

```
Prompt: "Explain what a mutex is in one paragraph."
Tokens: 20 decode steps, greedy sampling
Modell: Qwen3-8B-Q4_K_M

Run 1 (prequant OFF): 376.1 ms
Run 2 (prequant ON):  373.5 ms
                     ──────────
Wall-time Δ:          −0.68 %  (innerhalb Messrausschen)

Output text: bit-identisch zwischen beiden Runs ✓
Token count: identisch zwischen beiden Runs ✓
```

Die negative Regression (schnellerer Wert mit ON) ist messrauscheninduziert
und kein echter Speedup — die zwei Runs liegen innerhalb der üblichen
5 %-Varianz bei 20-Token-Messungen. Entscheidend ist **bit-Identität der
Ausgabe** bei aktiver Pre-Quantisierung: der Side-Effekt verändert keine
Logits, wie designed.

## Abweichung vom Prompt (transparent)

Zwei Punkte, bei denen ich vom Prompt abweiche, mit Begründung:

1. **`sum` ist float, nicht integer.** llama.cpp speichert `Σ xi`
   (Float-Warp-Reduce). Das matcht den Struct-Kommentar „d · sum(qs)"
   als Approximation und erhält Bit-Parity mit llama.cpp — das spätere
   MMVQ wird denselben Wert lesen. Der Prompt-Test
   „sum == Σ qs[i] (exakt, Integer)" war unter diesem Design nicht
   erreichbar; ich habe den Test auf die Toleranz-Version umgestellt.

2. **Call-Site ist TOKEN-weise, nicht MATMUL-weise.** Der Prompt sagte
   „VOR dem Layer-Loop einmal quantize_q8_1 aufrufen" — das habe ich
   umgesetzt (einmal pro execute_decode-Call, auf die Embedding-Row des
   aktuellen Tokens). Für die tatsächliche MMVQ-Konsumation in Schritt
   2/3 reicht das aber nicht — jede Layer-Aktivierung ist eine andere
   (Embedding-Output, Attention-Output, post-residual, SwiGLU-Output),
   also brauchen wir **pro GEMV** einen frischen Pre-Quant-Call. Das
   entspricht llama.cpp's Pattern (siehe `mmvq.cu:1091-1098`: Quantize
   direkt vor jedem `mul_mat_vec_q`). Für diesen Schritt 1/3 genügt die
   Token-weite Call-Site zur Validierung des Code-Pfads.

## Dateien in dieser Änderung

| Neu | LOC |
|---|---:|
| `hip_kernels_v1/quantize/quantize_q8_1.hip` | 97 |
| `src_v1/backend/gpu/quantize.rs` | 56 |
| `tests_v1/quantize_q8_1_test.rs` | 372 |
| `results/phase2_quantize_q8_1.md` (dieser Report) | — |

| Modifiziert | Änderung |
|---|---|
| `hip_kernels_v1/CMakeLists.txt` | +1 `add_hip_kernel` |
| `build.rs` | +1 link-lib entry `v1_quantize_q8_1` |
| `src_v1/backend/gpu/mod.rs` | +1 `pub mod quantize` |
| `src_v1/graph/executor.rs` | +2 fields, 3 methods (ensure, prequantize, set), call-site in `execute_decode` |
| `Cargo.toml` | +1 `[[test]]` entry |

## Nächster Schritt

→ **Schritt 2/3:** MMVQ-Kernel portieren (`mul_mat_vec_q<Q4_K, Q8_1>`
   spezialisiert für `ncols_dst=1`, ohne Gate-Fusion zunächst).

   * Quelle: `mmvq.cu:391-591` (Kernel-Template) +
     `vecdotq.cuh:505-527, 864-907` (Q4_K-Dot-Product).
   * Ziel-Datei: `hip_kernels_v1/gemv/mmvq_q4_k_q8_1.hip`.
   * Registrierung: `KernelId::MmvqQ4KQ8_1` in `runtime/variants.rs`
     als 3. Q4_K-Variante (vorerst optional hinter Env-Flag, damit der
     Bandit nicht wieder mit Convergence-Problemen kämpft).
   * Parity: bit-Vergleich gegen llama.cpp-Referenz-Output auf denselben
     Gewichten (die GGUF-Layout ist identisch, Kernel-Output sollte
     deckungsgleich sein).

   Aufwand geschätzt: ~500 LOC HIP + 80 LOC FFI + 200 LOC Tests ≈ 2-3
   Arbeitstage bei ruhigem Verlauf.

→ **Schritt 3/3:** Wire-in — quantize_q8_1 pro GEMV-Dispatch-Site
   (innerhalb des HIP-Graphs, direkt vor jedem Q4_K-GEMV-Knoten). Der
   Token-weise Call aus Schritt 1 wird dadurch ersetzt. Das ist der
   Punkt, an dem die +10-15 pp BW aus Hypothese H1 (siehe Analyse-Report)
   tatsächlich ankommen.
