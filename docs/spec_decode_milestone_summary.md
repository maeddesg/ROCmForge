# Speculative Decoding — Meilenstein-Zusammenfassung

**Stand:** 2026-04-17, commit `20733bb`
**Hardware:** RX 9070 XT (gfx1201, RDNA 4), ROCm 7.2.1
**Modelle:** Target Qwen2.5-7B-Instruct Q4_0, Draft Qwen2.5-0.5B-Instruct Q4_0

Kompakte Rekapitulation des Spec-Decode-Optimierungszyklus. Zielgruppe: ich selbst in sechs Monaten, oder ein Contributor, der den aktuellen Stand verstehen will.

## 1. Ausgangslage

- Speculative decoding implementiert und korrekt (PR #14 — Batched Verification, April 2026).
- Baseline (direktes 7B-Decode): **82 tok/s** Median über 15 Prompts.
- Spec-Decode **net-negativ** auf gemischten Prompts, profitabel nur bei hoher Acceptance-Rate.
- Gemessene Break-Even-α ≈ **41%** — unter diesem Schwellenwert ist Spec-Decode langsamer als direktes Decode.
- Code-Prompts (α ≥ 73%) profitabel, Chat/Prosa nicht.

## 2. Durchgeführte Experimente

| # | Experiment                        | Erwarteter Gewinn   | Gemessener Gewinn | Befund |
|---|-----------------------------------|---------------------|-------------------|--------|
| 1 | Spec-Step Cost Breakdown          | — (Profiling)       | —                 | Target-Verify = 88.6 % der Step-Zeit; FFN dominiert Verify (67.3 %) |
| 2 | Tiled Batched GEMV (FFN-Down)     | 2–4 ms/Step         | ~250 µs (~1.5 %)  | Memory-Controller pipelined sequentielle Zugriffe; Bandbreiten-Modell überschätzt um ~12× |
| 3 | Adaptive-Depth Threshold Sweep    | Threshold < 1.2 besser | Threshold 1.2 optimal | Superlineare Verify-Kosten mit Batch-Grösse überwiegen; niedrigere Schwellen net negativ |
| 4 | Batched lm_head                   | ~2.500 µs/Step      | ~114 µs (~0.4 %)  | Selber Pipelining-Effekt wie #2; GEMV auf gleicher Matrix pipelined auch ohne Batching |
| 5 | Buffer-Traffic-Validierung (FFN)  | 1.500–2.500 µs/Step | ~200 µs (~1.2 %)  | Hypothese falsifiziert; Fused FFN nicht wirtschaftlich; Gewichtsmatrix-Traffic dominiert Intermediate um Faktor 500 |

Details je Experiment:

- **#1** — `profiling/results/LAUNCH_OVERHEAD_ANALYSIS.md`
- **#2** — `docs/batched_verify.md` (Abschnitt "Memory-Controller Pipelining"), `benches/results/tiled_sweep_313efdf.json`
- **#3** — `benches/results/depth_threshold_sweep.md`
- **#4** — `benches/results/batched_lm_head_analysis.md`, `benches/results/batched_lm_head_sweep_2d595f8_1776356792.json`
- **#5** — `profiling/results/BUFFER_TRAFFIC_ANALYSIS.md`, `profiling/results/buffer_traffic_validation_{92d037a}_{1776402443}.json`

## 3. Was committed wurde

- **Tiled Batched GEMV** (`hip_kernels/quant/q4_0_gemv_batched_tiled.hip`, default-on). Disable: `ROCMFORGE_DISABLE_TILED_GEMV=1`.
- **Batched lm_head** (`src/gpu/forward.rs::gpu_verify_lm_head_batched`, default-on). Disable: `ROCMFORGE_DISABLE_BATCHED_LM_HEAD=1`.
- **Adaptive Speculation Depth** (EMA-basiert, Threshold 1.2/2.5, Initial = `spec_depth * 0.5`).
- **Profiling-Infrastruktur** — `ROCMFORGE_PROFILE_SPEC_STEP=1` (5-Phasen-Breakdown), `ROCMFORGE_PROFILE_VERIFY_BREAKDOWN=1` (Verify-Sub-Phasen).
- **Scratch-Buffer-Infrastruktur** — `MAX_SPEC_DEPTH=8` → `MAX_VERIFY_BATCH=9`; dedizierte `logits_batch`/`argmax_batch_device`/`argmax_batch_host`-Buffer in `GpuForwardScratch`.
- **Benchmark-Suite** — `benches/bench_spec.sh`, `benches/bench_batched_lm_head.fish` mit strukturierten JSON-Outputs.
- **Korrektheitstests** — `tests/spec_greedy_matches_baseline.rs` (greedy Spec-Decode-Output = direktes Greedy-Decode), `tests/batched_lm_head_matches_sequential.rs` (byte-identical Output depth 1/3/5), `--spec-depth`-Validierung (max 8).
- **Architecture Notes** — `docs/architecture_notes.md` mit dem Memory-Controller-Pipelining-Befund als projektübergreifende Erkenntnis.
- **Micro-Benchmark** — `profiling/buffer_traffic/bench.hip` (standalone HIP, A/B1/B2-Varianten, fma-depth-Sweep).

## 4. Zentrale architektonische Erkenntnis

Das "Load once instead of N times"-Modell überschätzt den Gewinn auf RDNA 4 **konsistent um Faktor 10–20×**. Drei unabhängige Experimente haben dieses Muster bestätigt: Dispatch-Batching, Weight-Reuse-Optimierung und Buffer-Traffic-Elimination bringen bei bandbreitenlimitierten Kernels keinen messbaren Gewinn. Der Memory-Controller pipelined sequentielle Zugriffe auf dieselbe Adressbereiche automatisch, solange keine explizite Synchronisation dazwischen liegt. Die erwarteten Einsparungen aus naiven Bandbreiten-Rechnungen waren in allen drei Fällen Modellartefakte.

Konsequenz: Optimierungshebel liegen bei **Algorithmuswechseln** (GEMV → GEMM für Prefill) und bei **Compute-Patterns mit nicht-vorhersagbaren Zugriffsmustern** (Attention-Tiling bei langem Kontext, wo der KV-Cache aus L2 fällt). Diese unterscheiden sich qualitativ von den bisherigen Experimenten, weil sie andere Memory-Access-Patterns haben, bei denen das Memory-Controller-Pipelining weniger greift.

## 5. Aktueller Performance-Stand

Median tok/s über 15 Prompts (5 code, 5 chat, 5 prose), Qwen2.5-7B-Q4_0 Target, Qwen2.5-0.5B-Q4_0 Draft, 128 generierte Tokens:

| Modus                    | Median tok/s | Code (best) | Chat (median) | Prose (median) |
|--------------------------|--------------|-------------|---------------|----------------|
| Baseline (no spec)       | **82**       | ~82         | ~82           | ~82            |
| Spec depth=1             | 69           | **86** (α=91%) | 68         | 66             |
| Spec depth=3             | 66           | 78          | 65            | 63             |
| Spec depth=5             | 57           | 63          | 56            | 57             |

**Einordnung:**

- Spec-Decode mit depth=1 ist auf **Code-Prompts (α ≥ 73%)** profitabel (+5% über Baseline im Best-Case).
- Auf **Chat/Prose net-negativ** (−15 bis −25% unter Baseline).
- **Break-Even-α ≈ 41%** — unter diesem Wert ist Baseline-Decode immer schneller.
- Adaptive Depth (EMA) konvergiert nach 5–10 Steps auf depth=1 für low-α-Prompts, hält bei high-α-Prompts die konfigurierte Maximum-Tiefe.

## 6. Empfohlene nächste Schritte

Keine Implementierungs-Pläne, nur Richtung:

1. **Prefill-GEMM** (hipBLAS oder WMMA). Grösster Performance-Rückstand zum State of the Art (Prefill pp19: 59 tok/s vs. llama.cpp 1.092 tok/s, 5% der Baseline). Algorithmuswechsel von N×GEMV zu echter GEMM ist qualitativ anders als bisherige Micro-Optimierungen — andere Memory-Access-Patterns, andere Kostenposition.
2. **Attention-Optimierung bei langem Kontext.** Bei K=4096 sinkt der Decode-Durchsatz von 115 auf 78 tok/s (−32%), Attention erreicht 34% der Decode-Zeit. KV-Cache fällt aus L2 — anderes Memory-Pattern, Tiling-Strategien (FlashAttention-artig) könnten hier echten Throughput bringen.
3. **14B-Target als Benchmark-Validation.** Kein Code-Aufwand. Validiert die Break-Even-These und zeigt, ob Spec-Decode bei grösseren Target-Modellen (grösseres Kostenverhältnis Draft:Target) profitabel wird. VRAM-limitiert auf 16 GB, aktuell knapp.
4. **Rejection Sampling.** Erst relevant, wenn Prefill und Attention gelöst sind. Würde greedy-Constraint aufheben und Acceptance-Rate auf Chat/Prose erhöhen, aber die Draft-Forward-Kosten (10% der Step-Zeit) bleiben — der grosse Hebel liegt woanders.
5. **CPU-Optimierung (Zen4+ / AVX-512).** AVX-512-VNNI-GEMV-Kernel implementiert und committed (commit `d0e4f07`). Isolierter Kernel-Speedup: **16–19 % auf 7B-Shapes, 1–7 % auf 0.5B-Shapes, 0 % End-to-End** auf 0.5B wegen Orchestrierungs-Overhead (Rayon Fork-Join + skalare Nicht-GEMV-Operationen). **Heterogenes Spec-Decode (Draft auf CPU) ist mit dem aktuellen CPU-Stack nicht machbar** — die CPU braucht 82 ms pro Draft-Token vs. 4.5 ms auf GPU. Erfordert fundamentalen CPU-Pipeline-Rewrite (Rayon-Elimination oder persistente Worker-Threads, fused Ops, SIMD für alle Pipeline-Stufen, nicht nur GEMV). Eigenständiges Grossprojekt, nicht als Nebenprodukt der Spec-Decode-Arbeit realisierbar. Details in `benches/results/cpu_avx512_analysis.md` und `docs/architecture_notes.md` (Abschnitt "Orchestration-Falle bei kleinen Modellen").

## 7. Was explizit nicht gemacht werden sollte

- **Fused FFN** — durch Experiment #5 falsifiziert. Maximaler Gewinn ~200 µs/Step (1.2%), Schwellenwert war 1.500 µs (8%). Komplexer Kernel-Rewrite mit <2% Throughput-Gain. Nicht implementieren.
- **Weitere GEMV-Micro-Optimierungen innerhalb des aktuellen Paradigmas.** Das Plateau ist erreicht — Target-Verify ist zu 88% GEMV-Execution gegen die Gewichtsmatrix, bandbreitenlimitiert. Weitere Micro-Optimierungen bringen < 2% und kollidieren typischerweise mit dem Memory-Controller-Pipelining-Muster.
- **Persistent-Thread-Kernels.** Launch-Overhead ist nicht der Engpass (gemessen ~2.7 µs/Dispatch, auf Stream gepiped quasi gratis). Architektonisch aufwändig, nur Single-digit-µs-Einsparung pro Dispatch.
- **Kleineres Draft-Modell (<0.5B).** Draft-Forward ist bereits nur 10% der Step-Zeit. Kleineres Draft spart wenig und erhöht das Acceptance-Risiko durch schlechtere Vorhersage.
- **Batched verify-Attention auf sehr langem Kontext.** KV-Cache-Lesezeit dominiert; Batch-Reduction der Dispatches hilft dort nicht, weil die Memory-Bandbreite ohnehin die Obergrenze ist.

---

Damit ist der Spec-Decode-Meilenstein sauber abgeschlossen. Ausgangspunkt für das nächste Kapitel (Prefill-GEMM) ist klar: **Algorithmuswechsel**, nicht weitere Dispatch-/Buffer-Micro-Optimierungen.
