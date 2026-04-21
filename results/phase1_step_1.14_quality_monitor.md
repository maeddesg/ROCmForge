# Phase 1 Schritt 1.14 — Quality Monitor

**Date:** 2026-04-21
**Branch:** v1.0-dev
**Hardware:** AMD RX 9070 XT (gfx1201)
**Goal:** Säule 5 der v1.0-Architektur — Laufzeit-Ergänzung zur
Model Introspection. Der Monitor kalibriert sich beim Pipeline-Start
an einem kurzen Prompt, überwacht den Hidden-State periodisch
(NaN/Inf/Mean-Abs-Drift/Max-Abs-Overflow) und erkennt
Repetition-Loops im Token-Output. **Phase 1 loggt nur** — Phase 2
konsumiert dieselben Signale für Precision-Eskalation + Layer-Rewind.

## Was gebaut wurde

- `src_v1/monitor/types.rs` — `QualityMonitor`, `ExpectedRange`,
  `PrecisionRevisionSignal`, `DriftReason`, `RevisionEvent`, `NodeId`.
  Feldnamen 1:1 aus Arch-Doc §2.6; `InfDetected`, `AttentionCollapse`
  und `RepetitionDetected` als Phase-1+-Erweiterungen additiv.
- `src_v1/monitor/calibration.rs` — `install_calibration()` +
  Konstanten (`CALIBRATION_PROMPT`, `MIN_CALIBRATION_STEPS=10`,
  `OUTPUT_HIDDEN=NodeId(0)`).
- `src_v1/monitor/check.rs` — `check_hidden_state()` (NaN/Inf →
  Max-Abs-Threshold → Mean-Abs z-Score, in genau dieser Reihenfolge).
- `src_v1/monitor/repetition.rs` — `record_token()`,
  `check_repetition()`, `should_check()`, `reset_check_counter()`.
  5× gleicher Token in Folge triggert `RepetitionDetected`.
- `ComputationGraph::hidden_state_buffer` + `GraphExecutor::read_hidden_state()`
  — synchronisiert den Stream **vor** dem `hipMemcpy`, um Race-Conditions
  mit pending-Decode-Kerneln auszuschliessen.
- Integration in `InferencePipeline::new()` + `calibrate_monitor()` +
  `generate()` (`record_token` + `check_repetition` jedes Token,
  `check_hidden_state` alle `sample_rate` Tokens).
- `cli/inference_test.rs` calibriert den Monitor vor dem 15-Prompt-
  Suiten-Lauf und druckt die Event-Summe am Ende.

## Calibration

| Modell | Prompt | Kalibrierungs-Schritte | mean_abs ± σ | max_abs | Laufzeit |
|---|---|---:|---:|---:|---:|
| Qwen3-8B Q4_K_M | "The quick brown fox jumps over the lazy dog" | 9 | 0.9522 ± 0.2216 | 74.33 | ~1 s |

Die erste Dekode-Stufe (direkt am Prefill-Ende) wird verworfen, weil
ihr Hidden-State die Prefill-Tail-Magnitude reflektiert, nicht den
Steady-State von Decode. `OUTPUT_HIDDEN` ist der einzige
Phase-1-Watchpoint (Hidden-State nach `output_norm`, direkt vor dem
LM-Head). Phase 2 verteilt zusätzliche Watchpoints pro Layer.

## Drift-Detection

Alle vier primitiven Detektoren feuern korrekt in den Einheitstests:

| Testfall | Detektor | Resultat |
|---|---|:---:|
| `hidden[100] = NaN` | `NaNDetected { count: 1 }` | ✅ |
| `hidden[100] = Inf` | `InfDetected { count: 1 }` | ✅ |
| `mean_abs = 0.01` gegen Band (0.5 ± 0.1) | `MeanAbsExceedsStdTolerance { z_score ≈ -4.9 }` | ✅ |
| `hidden[0] = 65 000` | `MaxAbsExceedsThreshold { threshold: 60000 }` | ✅ |

**Wichtige Reihenfolge:** Max-Abs-Check läuft **vor** dem z-Score,
damit ein einzelner FP16-Overflow-Outlier nicht fälschlich als
Mean-Abs-Drift rapportiert wird (der Outlier dominiert die Summe
und würde den Mean verfälschen).

## Repetition-Detection

- `record_token(42)` fünfmal in Folge → `RepetitionDetected { token_id: 42, count: 5 }` ✅
- Wechselnde Tokens (`10, 20, 30, …`) → kein Signal ✅
- Ring-Buffer-Cap: 100 Tokens — reicht für 5× + leichte n-Gram-
  Erweiterung in Phase 2.

## False-Positive-Test: 15-Prompt-Suite

| Metrik | Wert |
|---|---|
| Prompts | 15 |
| Max Tokens pro Prompt | 64 – 1024 |
| Decode-Tokens total | 6 080 |
| Monitor Hidden-State-Checks | ca. 190 (alle 32 Tokens) |
| Token-Level-Repetition-Checks | 6 080 |
| **Drift-Events** | **0** |
| **Repetition-Events** | **0** |
| 15-Prompt-Qualität | 13/15 korrekt (unverändert gegenüber 1.13) |
| Decode-Throughput | 30 tok/s (unverändert) |

Null False-Positives bei normalem Inference, Qualität identisch,
Overhead nicht messbar im Tok/s-Vergleich.

## Performance

- Calibration: ~1 Sekunde (9 Decode-Steps × ~100 ms bei 20 tok/s; der
  Executor ist beim ersten Call noch nicht optimiert).
- Monitor-Overhead pro Token:
  - `record_token` + `check_repetition`: < 100 ns (Vec push + 5-Element-Scan)
  - `read_hidden_state` + `check_hidden_state` alle 32 Tokens:
    `stream.synchronize()` + 4 KB hipMemcpy + CPU-Scan von 4096 f32.
    Bei 30 tok/s = ~1 Check/Sekunde → Overhead < 0.1 % der Decode-Zeit.

## Design-Entscheidungen

- **GPU→CPU Copy statt fused epilog.** Arch-Doc §2.6 beschreibt einen
  fused epilog im letzten Kernel jedes Layer-Blocks, der ein Dirty-Flag
  in pinned host memory schreibt. Das ist Phase-2 — Phase 1 nutzt den
  simpleren Pfad: `hipMemcpy Device → Host` nach `stream.synchronize()`,
  Check auf CPU. Bei `sample_rate=32` ist der Aufwand vernachlässigbar.
- **Stream-Sync vor Readback.** Ohne Sync würde der Monitor den
  Hidden-State vom *vorigen* Decode-Step lesen → Race-Condition,
  potentiell False-Negatives oder -Positives. Der Sync ist der
  bewusste Preis des CPU-seitigen Checks.
- **Check-Reihenfolge NaN/Inf → Max-Abs → Mean-Abs.** Härtere Fehler
  zuerst, damit ein einzelner Outlier nicht den Mean verfälscht.
- **3σ Toleranzfaktor.** Konservativ; lieber eine echte Drift
  verpassen als bei normalem Inference warnen. Arch-Doc §2.6 gibt
  explizit 3σ statt 2σ vor.
- **60 000 Max-Abs-Threshold.** FP16-max ist 65 504; der 10 %-Puffer
  gibt dem Forward-Pass Raum, bevor ein echter Overflow die Logits
  unbrauchbar macht.
- **Calibration nicht gecached.** Arch-Doc §5.2 — Treiber-Updates
  können die Hidden-States verschieben; die Kalibrierung dauert nur
  eine Sekunde, also jedes Mal neu.
- **`OUTPUT_HIDDEN=NodeId(0)` als einziger Phase-1-Watchpoint.** Der
  Hidden-State nach `output_norm` ist das einzige was der Executor
  billig exponiert. Phase 2 verteilt Watchpoints pro Layer (post-
  attention, post-FFN) via fused epilog.

## Bekannte Limitierungen (Phase-2-Kandidaten)

- **Keine Precision-Eskalation.** Phase 1 loggt nur; der
  `recommended_precision`-Wert im Signal wird nicht ausgeführt. Phase 2
  benötigt einen LRU-Cache kompilierter FP16-/BF16-/FP32-Varianten
  pro Layer.
- **Kein Checkpoint-Buffer.** Arch-Doc §5.4 beschreibt einen
  Ring-Puffer der letzten 4 Layer-Eingangs-Hidden-States für das
  Rewind-Playback. ~40 MB VRAM auf 8B-Modellen; Phase 2.
- **Kein FP8-Saturation-Count.** Phase 1 nutzt FP16-WMMA-Kernel, daher
  keine E4M3-Saturation zu messen. `Fp8SaturationExceeded` ist für
  Phase 2 schon als `DriftReason`-Variante definiert.
- **Keine Attention-Entropie-Messung.** `AttentionCollapse` ist als
  Variante reserviert; die Messung erfordert Zugriff auf die
  Softmax-Ausgabe jedes Attention-Blocks — Phase-2-Epilog.
- **Nur 1 Watchpoint.** Ohne fused epilog wäre mehr Pro-Layer-
  GPU→CPU-Copy zu teuer. Phase 2 mit epilog skaliert auf 32-48
  Watchpoints pro Token.
- **`sample_rate=32` fest.** Kein adaptives Sampling. Wenn ein Signal
  feuert, würde man im Phase-2-Regime kurzzeitig häufiger messen, bis
  die Drift stabil ist oder eskaliert wird.

## Test-Ergebnisse

| Test | Kategorie | Status |
|---|---|:---:|
| `test_install_calibration_populates_band` | Calibration | ✅ |
| `test_detect_nan` | Drift-Primitive | ✅ |
| `test_detect_inf` | Drift-Primitive | ✅ |
| `test_detect_magnitude_drift` | Drift-Primitive | ✅ |
| `test_detect_max_abs_overflow` | Drift-Primitive | ✅ |
| `test_no_signal_when_band_absent` | Korrektheit | ✅ |
| `test_detect_repetition_5x_same_token` | Repetition | ✅ |
| `test_no_repetition_on_normal_text` | Repetition | ✅ |
| `test_sample_rate` | Counter-Logik | ✅ |
| `test_calibration_populates_ranges` (GPU) | E2E | ✅ |
| `test_monitor_no_false_positives_on_normal_inference` (GPU) | E2E | ✅ |

**11/11 Tests grün.**

## Regression

| Suite | Status |
|---|:---:|
| `v1_codegen_elementwise_emit_test` / `v1_codegen_gemv_emit_test` | ✅ |
| `v1_runtime_test` (8/8) | ✅ |
| `v1_introspection_test` (5/5) | ✅ |
| `v1_inference_test` Tokenizer + 15-Prompt-Suite | ✅ (13/15 korrekt, 0 Monitor-Events) |
| v0.x Build | ✅ |

## Commit

Siehe Git-History (Commit-Prefix `feat:`). Report-Datei + Suite-Ausgabe
`results/inference_test_with_monitor.md` sind Teil des Commits.
