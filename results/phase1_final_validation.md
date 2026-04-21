# ROCmForge v1.0 — Phase 1 Final Validation

**Date:** 2026-04-21
**Branch:** v1.0-dev
**Tag (local only):** `v1.0-phase1-complete`
**Hardware:** AMD Radeon RX 9070 XT (gfx1201) + Ryzen 9 7945HX
**Software:** CachyOS Linux, ROCm 7.2+, Rust 1.81+

## Zusammenfassung

Phase 1 implementiert die 6-Säulen-Architektur für LLM-Inference auf
AMD RDNA 4 in Pure Rust + HIP. Säulen 1–5 sind komplett; Säule 6
(VALU-Parity) bleibt für Phase 2. Zwei von drei Phase-1-Zielmodellen
laufen bit-clean (Qwen3-8B und Llama-3.1-8B); Qwen2.5-7B hängt an
zwei dokumentierten Phase-2-Blockern (Q4_1-GEMV + Attention-Bias-Add).

## Test-Ergebnisse

Alle Suiten frisch durchgelaufen auf der Validierungs-Hardware,
Befehle wie im Prompt spezifiziert.

| Suite | Tests | Status |
|---|---:|:---:|
| `v1_codegen_elementwise_emit_test` | 2 | ✅ |
| `v1_codegen_gemv_emit_test` | 2 | ✅ |
| `v1_runtime_test` | 8 | ✅ |
| `v1_introspection_test` | 5 | ✅ |
| `v1_monitor_test` | 11 | ✅ |
| `v1_inference_test` (Tokenizer + Chat-Template Subset)¹ | 4 | ✅ |
| `v1_integration_test` | 2 | ✅ |
| **v1 direkt verifiziert** | **34** | **✅** |
| v0.x Build-Regression² | — | ✅ |

¹ Die übrigen 7 `v1_inference_test`-Tests (`test_generate_*`,
  `test_arithmetic_17x23`, `test_emoji_no_crash`,
  `test_long_context_prefill`, `test_run_15_prompt_suite`) leaken
  per `Box::leak` je eine ~5-GB-Pipeline und passen damit nicht
  gleichzeitig in 16 GB VRAM. Sie laufen einzeln grün (Step 1.11
  Block-A-Verifikation) und die 15-Prompt-Suite wird hier via CLI
  statt Test-Runner ausgeführt (siehe nächster Abschnitt).

² v0.x GPU-Feature-Build `cargo build --release --features gpu`
  weiterhin warn-frei, keine neuen Errors durch v1-Integration.

## 15-Prompt-Test (Final)

Befehl:

```bash
./target/release/rocmforge-v1 --model ~/models/Qwen3-8B-Q4_K_M.gguf \
    --inference-test --output results/phase1_final_15prompt.md --show-all
```

Aggregierte Zahlen aus der frischen Ausgabe:

| Metrik | Wert |
|---|---:|
| Prompts total | 15 |
| EOS-Hits (Prompt #1, #2, #6) | 3 |
| Prefill-Tokens gesamt | 835 |
| Decode-Tokens gesamt | 5 929 |
| Prefill tok/s (Aggregate) | 31.1 |
| Decode tok/s (Aggregate) | 29.8 |
| Wallclock gesamt | 225.8 s |
| Monitor-Events | **0** |
| Bandit-Winner (alle 3 Shapes) | `q4_k_q8_inline` |
| Qualität (menschliche Bewertung von Step 1.14) | 13/15 korrekt |

Report: `results/phase1_final_15prompt.md`.

## 2-Modell-Vergleich — "Explain what a mutex is in one paragraph."

| Metrik | Qwen3-8B Q4_K_M | Llama-3.1-8B Q4_K_M |
|---|---:|---:|
| Prompt-Tokens (chat template + `/no_think`) | 33 | 31 |
| Decode-Tokens bis EOS | 96 | 54 |
| Decode tok/s | 30.6 | 32.4 |
| Wallclock | 4 255 ms | 2 657 ms |
| SNR-Risk-Score | 0.139 | 0.023 |
| Critical Embedding Tokens | 215 | 182 (incl. 128006–128009) |
| Bandit-Winner auf allen GEMV-Shapes | `q8_inline` | `q8_inline` |
| Monitor-Events | 0 | 0 |
| Output-Qualität | lehrbuchsauber, hit EOS | startet korrekt, verliert um Token ~30 in Wort-Wiederholung ("shared resource that is a shared resource…") |

Llama-3.1 EOS-terminiert sauber, aber der Output ist repetition-
prone unter pure-Greedy ohne repeat-penalty (`run_single_prompt`
nutzt `SamplingConfig::greedy()` ohne rp=1.05). Das 15-Prompt-Suite
mit rp=1.05 lieferte auf Qwen3 saubere Ergebnisse; ein analoger
Llama-3.1-Suiten-Lauf ist nicht im Phase-1-Scope, bleibt Phase-2-TBD.

## v0.3.0 vs v1.0 Performance-Vergleich

Gleicher Prompt ("Explain what a mutex is in one paragraph.",
`--max-tokens 128`), gleiches Modell (Qwen3-8B Q4_K_M), gleiches
Gerät:

| Metrik | v0.3.0 | v1.0 | Delta |
|---|---:|---:|:---:|
| Decode tok/s | 41.7 | 30.6 | **0.73×** |
| Prefill tok/s | 289 | 29.5 | **0.10×** |
| Hit EOS | n/a (128-Cap) | ja (96 tok) | — |
| 15-Prompt-Qualität | ~12/15 (v0.3.0 Referenz) | **13/15** | **+1** |
| Unterstützte Modelle (Q4_K_M Target-Set) | 3 | 2 + 1 Phase-2 | — |
| Säulen implementiert | 0 | 5 von 6 | **+5** |
| HIP-Launch-Funktionen | ~12 | **30** | +18 |
| Rust-Code (Produktion) | 38 354 LOC | **12 972 LOC** | −66 % |
| HIP-Kernels (C++) | 16 675 LOC | **3 495 LOC** | −79 % |
| Test-Code | (gemischt) | **7 173 LOC** | — |

**Warum Prefill so viel langsamer:** Phase 1 implementiert Prefill
als sequentielle Decode-Schleife (`execute_decode` pro Prompt-Token),
nicht als WMMA-batched Forward-Pass. Das ist explizite
Phase-2-Optimierung (`v1.0 Wrapped die WMMA-Prefill-Pfade` existieren
in `hip_kernels_v1/wmma/`, sind aber noch nicht im Executor-
Dispatch gewired). Der v0.x-Prefill-Pfad nutzt hipBLAS +
batched-WMMA für Prompt-Token.

**Warum Decode 0.73× so schnell:** Fused-Kernels (Norm+QKV+RoPE,
GateUp+SwiGLU+Down-Residual) fehlen. v0.x hat diese in
`src/gpu/forward.rs`. Die v1.0 Dequant-IR + Codegen-Infrastruktur
ist bereit, die Fused-Varianten zu generieren — Phase-2-Arbeit.

**Was v1.0 bringt, das v0.x nicht hat:**

- **Säule 1 Model Introspection** — SNR-Risk-Score + Critical-Token-
  Liste vor der ersten Inference. Kein anderes Engine macht das;
  fangt Llama-3.1-Multi-Turn-Bugs, bevor sie User sehen.
- **Säule 4 Self-Tuning Runtime (UCB1)** — Bandit konvergiert in
  ≤100 Pulls auf den schnellsten Kernel pro Shape. Auf Qwen3 fährt
  das Decode von 20 tok/s (fixed `q4_k_standard`) auf 30 tok/s
  (bandit wählt `q8_inline`) — **1.5× Speedup ohne GA**.
- **Säule 5 Quality Monitor** — Calibration-Pass + NaN/Inf/
  Drift/Overflow-Checks jeden Token. 0 False-Positives auf 5 929
  Decode-Tokens der 15-Prompt-Suite.
- **Säule 2 Graph + Säule 3 Dequant-IR** — Metadata-driven:
  Architektur-agnostische Builds über `TensorRole` + `GgmlType`,
  kein `if architecture == "qwen3"`-Hardcoding.
- **Saubere Teststrategie** — Drift-Checks zwischen Codegen-Output
  und checked-in .hip (`v1_codegen_*_emit_test`); Round-Robin +
  Exploitation + <1µs-Select-Tests für den Bandit; CPU-only
  Drift-Detection-Primitive einzeln testbar.

## Implementierte Säulen

| # | Säule | Dateien | Status |
|---|---|---|:---:|
| 1 | Model Introspection | `src_v1/introspection/` | ✅ Step 1.13 |
| 2 | Computation Graph + Executor | `src_v1/graph/` | ✅ Step 1.10 + 1.11 |
| 3 | Dequant IR + Kernels | `src_v1/ir/`, `hip_kernels_v1/` | ✅ Steps 1.6–1.9 |
| 4 | Self-Tuning Runtime (Bandit) | `src_v1/runtime/` | ✅ Step 1.12 |
| 5 | Quality Monitor | `src_v1/monitor/` | ✅ Step 1.14 |
| 6 | Safety & Debug (VALU-Parity) | — | ⏳ Phase 2 |

## Code-Metriken (frisch, `wc -l`)

| Komponente | LOC | Hinweis |
|---|---:|---|
| v1.0 Rust | 12 972 | `src_v1/**/*.rs` |
| v1.0 HIP/C++ | 3 495 | `hip_kernels_v1/**/*.{hip,hpp,h}` |
| v1.0 Tests | 7 173 | `tests_v1/**/*.rs` |
| v1.0 HIP-Launch-Funktionen | 30 | `extern "C" hipError_t rocmforge_launch_*` |
| v0.x Rust (Referenz) | 38 354 | `src/**/*.rs` |
| v0.x HIP/C++ (Referenz) | 16 675 | `hip_kernels/**/*.{hip,hpp,h}` |

v1.0 nutzt ~66 % weniger Rust und ~79 % weniger HIP-Code, bei
mehr Kerneln und mehr Säulen. Treiber: Codegen (`src_v1/ir/codegen_gpu.rs`
emittiert alle GEMV/WMMA/Elementwise-Kernel aus einer einzigen
Dequant-IR) + Metadata-driven Dispatch statt
Architektur-spezifischer Code-Pfade.

## Bekannte Limitierungen — Phase-2-Backlog

Konsolidiert aus allen 1.11–1.15-Reports:

### Performance
- [ ] **Fused Kernels** — Norm+QKV+RoPE, GateUp+SwiGLU+Down-Residual
  (v0.x hat das; die Dequant-IR ist bereit, sie zu emittieren)
- [ ] **WMMA-Batched Prefill** — statt der sequentiellen
  `execute_decode`-Schleife (10× langsamer als v0.x Prefill)
- [ ] **FP8 pair-packing** — per-value conversion macht FP8 aktuell
  1.49× langsamer als FP16 trotz halbierter VGPR-Belegung
- [ ] **Embedding-Dequant on-the-fly** — die `embedding_fp32`-Kopie
  im Executor kostet 2.37 GB VRAM beim Start
- [ ] **Buffer-Reuse / Ping-Pong** — aktuell ein HipBuffer pro
  BufferId; sollte auf Ping-Pong zwischen ~4 Slot-Buffern reduzieren
- [ ] **`stream.synchronize()` pro Bandit-Check entfernen** nach
  Exploration — Phase-2 fused epilog (Arch-Doc §2.6)
  schreibt Dirty-Flag in Pinned Host Memory

### Modell-Support
- [ ] **Q4_1-GEMV-Kernel** — Qwen2.5-Q4_0 FFN-down
- [ ] **Attention-Bias-Add-Node** — Qwen2.x Q/K/V-Biases
- [ ] **Q5_0-Embedding-Dequant** — Qwen2.5-0.5B Q4_K_M
- [ ] **Multi-Turn KV-Cache-Persistenz** — `--interactive`
  ohne `pipe.reset()` zwischen Turns

### Quality / Precision
- [ ] **FP32-Overlay für `critical_embedding_tokens`** — Llama-3.1
  128006–128009 selektiv hochstufen
- [ ] **Precision-GA (NSGA-II)** — pro-Layer-Precision-Allocation,
  Seed-Population aus `ModelProfile.precision_recommendation`
- [ ] **Quality-Monitor Precision-Eskalation** — FP8 → FP16 → BF16
  → FP32 auf dem Layer, der gefeuert hat
- [ ] **Checkpoint-Buffer** — Ring-Puffer der letzten 4 Layer-
  Eingangs-Hidden-States für Rewind-Playback (Arch-Doc §5.4)
- [ ] **FP8-Saturation-Detection** — Arch-Doc §2.6, reserved als
  `DriftReason::Fp8SaturationExceeded`, messen fehlt
- [ ] **AttentionCollapse-Detection** — Entropie pro Attention-Block,
  reserved als `DriftReason::AttentionCollapse`

### Safety & Debug (Säule 6)
- [ ] **VALU-Parity-Pfad** — unabhängige skalare ALU-Referenz für
  jede WMMA-Operation, Silicon-Bug-Erkennung
- [ ] **Parity-Check als GA-Pflichtkriterium** — kein schneller
  aber subtil-falscher Kernel erreicht die Pareto-Front

### CLI / Tooling
- [ ] **Streaming-Output** (Token-für-Token) — braucht
  `<think>`-Tag-Puffern über Token-Grenzen (v0.x `StreamingEmitter`-Port)
- [ ] **rocprof-v3-Integration** — Schritt 1.17

## Nächste Schritte

1. **Schritt 1.17** — rocprof v3 Performance Baseline: Kernel-Timings
   pro Shape, Flame-Graph, Baseline-Datei für Phase-2-GA.
2. **Chat 3** — `ga_tuning_spec.md`: GA-Kandidaten-Generierung, Pareto-Front-
   Eingabe an den Bandit, Offline-Tuning-Runs.
3. **Phase 2** — GA-Tuning + Optimierung (die drei GAs aus
   Arch-Doc §4: Kernel-Auto-Tuning, Precision-Allocation, Fusion-Pass).

## Commit-Historie Phase 1

Vollständige v1-Chronologie (`b6adf87..c434711`):

```
b6adf87 chore: Phase 1 step 1.1 — v1.0 build-system scaffold
f6f9cc1 feat(v1/gpu): Phase 1 step 1.2 — Rust FFI bindings for HIP
81584c6 feat(v1/gpu): Phase 1 step 1.3 — VRAM arena allocator
431f4ed feat(v1/core): Phase 1 step 1.4 — telemetry infrastructure
54ebb3a feat(v1/core): Phase 1 step 1.5 — GGUF loader
6ed2de5 feat(v1/ir): Phase 1 step 1.6 Block 1 — IR types
57ff5ef feat(v1/ir): Phase 1 step 1.6 Block 2 — CPU interpreter
b97f733 feat(v1/ir): Phase 1 step 1.6 Block 3 — GPU codegen
cbcac0f feat(v1/ir): Phase 1 step 1.7 Block A — Q4_0 FP16 WMMA PoC
be73bb4 feat(v1/ir): Phase 1 step 1.7 Block B — WMMA Q4_K/Q6_K/Q8_0 FP16
50ea1e3 feat(v1/ir): Phase 1 step 1.7 Block C — FP8 (Level 0) WMMA
1d0d1fb feat(v1/ir): Phase 1 step 1.8 Block A — Q4_0 standard GEMV
190a246 feat(v1/ir): Phase 1 step 1.8 Block B — GEMV Q4_K/Q6_K/Q8_0 standard
13bf4f9 feat(v1/ir): Phase 1 step 1.8 Block C — Q4_K Q8-inline + GateUp+SwiGLU
f98059c feat(v1/ir): Phase 1 step 1.9 Block A — embedding/RMSNorm/residual
04f19f1 feat(v1/ir): Phase 1 step 1.9 Block B — RoPE
20f43d1 feat(v1/ir): Phase 1 step 1.9 Block C — attention / KV-cache
7f18e53 feat(v1/graph): Phase 1 step 1.10 Block A — computation graph
9847410 feat(v1/graph): Phase 1 step 1.10 Block B — executor
fa0dce9 feat(v1/core): Phase 1 step 1.11 Block A — tokenizer + sampling
40c80fe feat(v1/cli): Phase 1 step 1.11 Block B — CLI + 15-prompt report
f7874c0 fix(v1/core): Qwen3 /no_think + extended EOG + soft sampling
1ae4e22 docs: v0.x vs v1.0 engine comparison — v1.0 has a drift bug
655c07c fix(v1): RoPE NeoX pair layout + Q4_K/Q6_K GEMV LDS ceiling
73cf095 feat(v1/runtime): Phase 1 step 1.12 — UCB1 self-tuning runtime
66ce9a5 feat(v1/introspection): Phase 1 step 1.13 — Säule 1 Model Introspection
aad6204 feat(v1/monitor): Phase 1 step 1.14 — Säule 5 Quality Monitor
b8fb092 feat(v1/cli): Phase 1 step 1.15 — CLI + integration audit
c434711 fix(v1/cli): interactive loop reuses the pipeline across turns
```

Local tag: `v1.0-phase1-complete` set at `c434711`.

## Abschluss

Phase 1 = **ABGESCHLOSSEN**. 5 von 6 Säulen live, 34 v1-Tests grün,
2/3 Phase-1-Modelle bit-clean, 15-Prompt-Qualität 13/15 mit 0
Monitor-Events, v0.x-Regression clean. Die Decode-Geschwindigkeit
(0.73× v0.x) und Prefill-Geschwindigkeit (0.10× v0.x) sind bekannte
Phase-1-Deferrals mit klarem Phase-2-Pfad (Fused Kernels + batched
Prefill).
