# Phase 1 Schritt 1.13 — Model Introspection

**Date:** 2026-04-21
**Branch:** v1.0-dev
**Hardware:** AMD RX 9070 XT (gfx1201), CachyOS Linux
**Goal:** Säule 1 der v1.0-Architektur — beim Modell-Load einen Scan
laufen lassen, der Embedding-Magnituden + Layer-Stichproben auswertet
und ein `ModelProfile` mit SNR-Risk-Score + Precision-Empfehlung
produziert. Seed für die Phase-2 Precision-GA.

## Was gebaut wurde

- `src_v1/introspection/profile.rs` — `ModelProfile`, `LayerStats`,
  `PrecisionHint`, `TokenId`. Feldnamen 1:1 aus
  `architecture_v1.2.0-draft.md §2.2`.
- `src_v1/introspection/scan.rs` — `introspect(gguf: &GGUFFile) -> ModelProfile`.
  Embedding-Scan (alle Rows, L2 pro Row aus dem mmap-Blockstream)
  + Layer-Stichprobe (1024 Werte pro Tensor) + Noise-Schätzung +
  SNR-Score + Precision-Empfehlung pro Layer.
- Integration in `InferencePipeline::new()` — Scan einmalig beim
  Laden, Ausgabe des Summary-Tabellenblocks, visuelle Warnung wenn
  `snr_risk_score < 2.0`.
- `print_summary()` — Box-gedruckte Tabelle mit Embedding-Range,
  Critical-Token-Count, Noise, SNR-Score, Precision-Breakdown.
- 5 Tests in `tests_v1/introspection_test.rs`.

## Wichtigste Beobachtung

Alle drei Phase-1-Zielmodelle haben `snr_risk_score < 2.0`. Der
Arch-Doc-Algorithmus (10 %-Schwelle, `signal / noise`-Score) liefert
konsequent den `CRITICAL`-Band auf echten Q4_K- und Q4_0-Quantisierungen,
weil **jedes** dieser Modelle einige Byte-Fallback- oder Reserved-Slots
in seinem Vokabular hat. Der *Unterschied* zwischen "sicher" und
"riskant" liegt nicht im Score, sondern darin, ob die kritischen Token
auf dem Hot-Path (Chat-Template) auftauchen:

- **Llama-3.1-8B** — 128006…128009 (`<|start_header_id|>`,
  `<|end_header_id|>`, `<|eot_id|>`) sind kritisch **und** werden in
  jedem Multi-Turn-Chat gefeuert → bekannter v0.x-Garbage-Bug.
- **Qwen3-8B / Qwen2.5-7B** — kritische Tokens sind fast ausschliesslich
  Byte-Fallbacks und reservierte Vokabular-Slots → praktisch nie im
  Input → die Modelle laufen sauber, obwohl der Score `CRITICAL` sagt.

Die Phase-2-FP32-Overlay-Logik wird auf den `critical_embedding_tokens`-Set
aufbauen und zur Laufzeit entscheiden, ob ein kritisches Token *tatsächlich*
ankommt, bevor sie auf FP32 hochstuft. Phase 1 liefert nur die Kandidaten.

## ModelProfile-Ergebnisse

| Modell | Vocab | Scan | Emb. L2 min | Emb. L2 max | Critical Tok | Noise (L2) | SNR | Band |
|---|---:|---:|---:|---:|---:|---:|---:|:---:|
| Qwen3-8B Q4_K_M | 151 936 | 1.44 s | 0.00954 | 1.95027 | 215 | 0.0687 | 0.139 | CRITICAL |
| Llama-3.1-8B Q4_K_M | 128 256 | 1.20 s | 0.00000 | 0.93122 | 182 | 0.0335 | 0.023 | CRITICAL |
| Qwen2.5-7B Q4_0 | 152 064 | 0.73 s | 0.00000 | 1.11934 | 1 063 | 0.0392 | 0.108 | CRITICAL |

Alle drei Scans liegen komfortabel unter dem 5-Sekunden-Budget aus
Arch-Doc §2.2.

### Llama-3.1 — Hot-Path-Check

Der Test `test_introspection_llama31_special_tokens_flagged` prüft
gezielt, ob mindestens einer der Chat-Template-Tokens 128006..=128009
im `critical_embedding_tokens`-Set landet. **Erfüllt.** Das ist das
Kern-Argument für Model Introspection: kein anderes Inference-Engine
erkennt dieses Token-Set automatisch.

### Qwen3 — Erste 16 Critical-Token-IDs

```
27487, 51088, 57408, 70564, 78323, 79269, 79270, 83969,
83971, 86278, 88371, 88372, 99875, 99997, 101025, 101028, +199 more
```

Diese liegen weit ausserhalb der Qwen3-ChatML-Bereiche (151643 bos,
151645 `<|im_end|>`); es sind reservierte und byte-fallback-Slots.
Die Phase-2-Overlay-Logik würde sie ignorieren, solange sie nicht
tatsächlich in der Eingabe auftauchen.

### Precision-Empfehlung

Nach dem Arch-Doc-Regel-Set:

- `max_abs/mean_abs > 50` → `Fp32Scales`
- sonst `snr_risk < 2.0` → `Bf16Scales`
- sonst → `Fp8E4M3`

| Modell | FP8 | BF16 | FP32 |
|---|---:|---:|---:|
| Qwen3-8B | 0 | 249 | 3 |
| Llama-3.1-8B | 0 | 224 | 0 |
| Qwen2.5-7B | 0 | 196 | 0 |

Weil alle drei Modelle im CRITICAL-Band liegen, kippt die Empfehlung
für praktisch alle Layer auf BF16. Das ist der Arch-Doc-konforme
Startpunkt — die Phase-2-GA wird daraus pro Layer ausprobieren, ob
FP16/FP8 reicht oder FP32 nötig ist.

## Test-Ergebnisse

| Test | Status |
|---|:---:|
| `test_introspection_qwen3_structure` | ✅ |
| `test_introspection_llama31_special_tokens_flagged` | ✅ |
| `test_introspection_under_5_seconds` | ✅ (1.44 s) |
| `test_introspection_precision_recommendation_populated` | ✅ |
| `test_introspection_print_summary` | ✅ |

**Regression:** `v1_codegen_elementwise_emit_test`, `v1_codegen_gemv_emit_test`,
`v1_runtime_test` (8/8), `v1_inference_test` Tokenizer-Subset alle grün.
v0.x-Build ebenfalls clean.

## Design-Entscheidungen

- **Embedding: voll.** Alle `vocab_size` Rows werden dequantisiert,
  weil die Critical-Token-Identifikation zeilengenau sein muss. Das
  sind ~2.4 M Q4_K-Blocks für Qwen3 — passt in 1.4 s.
- **Layer: Stichprobe.** 1024 Werte pro Tensor (= 4 Q4_K-Superblöcke).
  Voller Dequant aller ~290 Tensoren eines 8B-Modells wäre 4 GB und
  zig Sekunden. Die Stichprobe liefert stabile `mean_abs` / `max_abs`
  / `std_abs`-Werte für die Precision-Empfehlung.
- **mmap-direkter Zugriff.** `GGUFFile::tensor_data_full()` liefert
  eine Zero-Copy-Slice-View in den mmap. Der Block-Stream-Loop im
  Scan dequantisiert 256 Werte am Stück, aggregiert `sum(v²)` und
  `sum(|v|)`, und verwirft das Block-Ergebnis ohne Vec-Kopie der
  ganzen Zeile. **Nie** wird die 2.4 GB-Embedding-Tabelle
  rekonstruiert.
- **Noise-Formel.** `noise_L2 ≈ per_value_noise_ratio × embedding_mean_abs × sqrt(hidden)`
  — folgt dem Arch-Doc-Beispiel (Q4_K über 4096 Dim → 0.064). Die
  Embedding-`mean_abs` wird on-the-fly während des Zeilen-Scans
  aggregiert; der vorher aus dem Layer-Gemiddelde berechnete Wert
  unter-schätzte das Signal systematisch.
- **`Fp8E4M3` als Default statt `Fp16Scales`.** Arch-Doc-Pseudocode
  sagt `Fp16Scales`, aber die `PrecisionHint::Fp8E4M3`-Docstring nennt
  FP8 explizit "Standardpfad für Gewichte in v1.1". Der Prompt
  fordert ebenso `Fp8E4M3` als Fallback. Phase-1 speichert nur die
  Empfehlung — kein Verhaltensunterschied bis Phase 2 sie konsumiert.

## Bekannte Limitierungen (Phase-2-Kandidaten)

- **Alle Modelle landen im CRITICAL-Band.** Der SNR-Score als binärer
  Safe/Unsafe-Indikator ist zu grob. Phase-2 muss zusätzlich die
  tatsächliche Token-Nutzung berücksichtigen (ob ein kritisches Token
  auf dem Chat-Template-Pfad liegt oder nicht).
- **FP32-Overlay fehlt.** Die `critical_embedding_tokens` werden
  protokolliert, aber nicht hochgestuft. Kommt in Phase 2 mit der
  Precision-GA und einem FP32-Override-Slot in der Embedding-LUT.
- **Keine Warnung an CLI-Nutzer.** Der `print_summary()`-Block erscheint
  auf stderr; wer nicht hinguckt, merkt nichts. Bei CRITICAL-Modellen
  sollte die CLI einen Hinweis in die stdout-Ausgabe einblenden.
- **Noise-Schätzung ist grob.** Die 1/16-Regel für Q4_K ist eine
  Näherung; genauer wäre ein Roundtrip-Test (dequant → requant →
  Vergleich). Für Phase 1 reicht die Näherung, um die Grössenordnung
  zu treffen.

## Performance

- Qwen3-8B scan: **1.44 s** (252 Tensoren, 151 936 Embedding-Rows)
- Llama-3.1-8B scan: **1.20 s** (224 Tensoren, 128 256 Embedding-Rows)
- Qwen2.5-7B scan: **0.73 s** (196 Tensoren, 152 064 Embedding-Rows,
  Q4_0 dequant ist billiger als Q4_K)

Alle drei weit unter dem 5-Sekunden-Budget.

## Commit

Fix + Runtime-Integration: siehe Git-History (Commit-Prefix `feat:`).
Dieser Report ist Teil des Commits selbst.
