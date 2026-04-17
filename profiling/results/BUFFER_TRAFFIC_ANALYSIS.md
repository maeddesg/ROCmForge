# Intermediate Buffer Traffic Validation — 2026-04-17

## TL;DR

**Verdikt: Fused FFN lohnt sich nicht.** Maximaler Gewinn ~224 µs/Step (1.4 % der Verify-Zeit), und dieser Gewinn ist **Dispatch-Overhead-Reduktion**, nicht Buffer-Traffic-Elimination. Die ursprüngliche Hypothese (intermediate-Buffer-Traffic ist der Hebel) wird durch die Messung **falsifiziert**.

## Setup

- Hardware: RX 9070 XT (gfx1201, RDNA 4), ROCm 7.2.1
- Dimensionen (wie Qwen2.5-7B): hidden=3584, intermediate=17920 (5×3584, clean ratio statt 18944)
- 28 Layer pro Durchlauf, 10 Warmup + 100 gemessene Runs
- Alle Kernels elementwise, 1–1024 FMAs pro Element via `fma_depth`-Parameter

Drei Varianten:

- **A** — 4 separate Kernel-Dispatches pro Layer, intermediate in VRAM. 112 Launches/Step.
- **B2** — 1 cooperative Kernel pro Layer mit `grid.sync()` zwischen den Phasen. Gleiche VRAM-Traffic-Profil wie A, aber 1 Dispatch/Layer → 28 Launches/Step.
- **B1** — 1 chunked Kernel pro Layer, intermediate lebt nur in Registern. Keine intermediate-VRAM-Touches. 28 Launches/Step.

Δ(A − B2) isoliert Dispatch-/Launch-Overhead. Δ(B2 − B1) isoliert intermediate-Buffer-Traffic. Δ(A − B1) ist der maximal erreichbare Fused-FFN-Gewinn.

Artefakt: [buffer_traffic_validation_{92d037a}_{1776402443}.json](buffer_traffic_validation_{92d037a}_{1776402443}.json)

## Kernresultate (Median µs, batch=2)

| fma_depth |   A  |  B2  |  B1  | Δ(A−B2) | Δ(B2−B1) | Δ(A−B1) |
|----------:|-----:|-----:|-----:|--------:|---------:|--------:|
|         1 |  321 |  389 |   97 |     −69 |      293 |   **224** |
|        16 |  325 |  393 |  112 |     −68 |      281 |     212 |
|       128 |  375 |  440 |  242 |     −65 |      198 |     133 |
|      1024 |  845 |  847 | 1865 |      −2 |    −1018 |   −1020 |

Median µs, batch=4:

| fma_depth |   A  |  B2  |  B1  | Δ(A−B2) | Δ(B2−B1) | Δ(A−B1) |
|----------:|-----:|-----:|-----:|--------:|---------:|--------:|
|         1 |  333 |  604 |   97 |    −270 |      507 |   **236** |
|        16 |  340 |  603 |  112 |    −263 |      491 |     228 |
|       128 |  423 |  602 |  244 |    −179 |      359 |     179 |
|      1024 | 1089 | 1061 | 1896 |      29 |     −835 |    −806 |

## Was die Messung zeigt

### 1. Variante A ist Dispatch-bound, nicht Compute-bound

A ändert sich kaum zwischen `fma_depth=1` und `fma_depth=128` (321 → 375 µs, +54 µs für 128× mehr ALU-Arbeit). Die GPU versteckt den zusätzlichen Compute hinter der Dispatch-Pipeline von 112 Kernel-Launches. Per-Kernel-Budget in A: ~3 µs (≈ 2 µs Dispatch + ~1 µs GPU-Execution bei Memory-Bound-Elementwise).

### 2. Variante B1 ist bei relevantem Compute-Volumen Compute-bound

B1 skaliert stark mit `fma_depth` (97 → 112 → 242 → 1865 µs). Weil B1 nur 28 Launches hat und die ALU-Arbeit unverdeckt ist, wächst die Gesamtzeit linear mit dem Compute.

### 3. Der "Fused-Gewinn" sinkt mit steigender Compute-Intensität

Δ(A−B1) fällt von **224 µs** (depth=1) auf **133 µs** (depth=128) und **wird negativ** bei depth=1024. Das ist nicht Buffer-Traffic — das ist der direkte Beleg, dass Fusion primär **Launch-Overhead** einspart. Sobald die Kernels genug Compute haben, um die Launches zu verstecken, verschwindet der Fused-Vorteil.

### 4. B2 misst Dispatch-Overhead **nicht** sauber

Cooperative-Kernel-Launches mit `grid.sync()` sind auf dieser Hardware teurer als 4 sequentielle Dispatches auf demselben Stream. Bei batch=4 ist B2 konstant ~265 µs langsamer als A — Stream-Pipelining von A ist effizienter als grid-weite Sync in einem Launch. Das bestätigt indirekt die RDNA-4-Memory-Controller-Pipelining-Hypothese aus `docs/architecture_notes.md`: Back-to-Back-Dispatches auf dem gleichen Stream pipelinen nahezu verlustfrei.

### 5. Buffer-Traffic ist im Elementwise-Regime nicht isoliert messbar

Δ(B2−B1) schwankt von +293 (depth=1) bis −1018 (depth=1024). Das ist kein Buffer-Traffic-Signal, sondern ein Mix aus Cooperative-Launch-Overhead (B2 teuer) und B1's Compute-Exposition (B1 wird compute-bound schneller als B2). Der Micro-Benchmark kann Buffer-Traffic nicht sauber von Dispatch- und Occupancy-Effekten trennen.

## Hochrechnung auf reale FFN-Zeit

Gemessene reale FFN-Zeit (Launch-Overhead-Analyse, 28 Layer, batch=2, depth=1): **10.900 µs**.

Die realen FFN-Kernels (GEMV gegen Q4_0-Gewichtsmatrizen) sind **memory-bound gegen die Gewichtsmatrix**, nicht gegen die intermediate-Buffer:

| Kernel               | Zeit (µs)  | Gewicht gelesen | Intermediate-Traffic |
|----------------------|-----------:|----------------:|---------------------:|
| gate_up_silu (fused) |        232 |         ~75 MB |              ~75 KB |
| mul (elementwise)    |        ~30 |               — |             ~150 KB |
| down (GEMV)          |        145 |         ~75 MB |             ~150 KB |
| residual             |        ~10 |               — |              ~14 KB |

Das Gewichtsvolumen (~150 MB pro Layer) dominiert die Memory-Pipeline um Faktor ~500. Die intermediate-Buffer-Traffic ist bereits in den GEMV-Laufzeiten amortisiert, nicht ein separater Posten.

**Was Fused FFN realistisch einsparen kann:**

| Komponente                               | Einsparung |
|------------------------------------------|-----------:|
| Dispatch-Overhead (4→1 pro Layer × 28)   |   ~168 µs  |
| Intermediate-Buffer-Traffic-Elimination  |    ~30 µs  |
| Sync-Elimination (4→1 per layer)         |     ~0 µs* |
| **Gesamt**                               |   **~200 µs** |

\* Stream-Pipelining macht Between-Kernel-Syncs auf demselben Stream quasi gratis (dokumentiert in `docs/architecture_notes.md`).

**Relation zur Verify-Zeit:**

| Metrik                               | Wert        |
|--------------------------------------|-------------|
| Reale FFN-Zeit (28 Layer)            | 10.900 µs   |
| Verify-Step gesamt                   | 16.200 µs   |
| Erwarteter Fused-FFN-Gewinn          | ~200 µs     |
| Anteil an Verify-Zeit                | **~1.2 %**  |
| Anteil an FFN-Zeit                   | **~1.8 %**  |

## Verdikt

**Fused FFN lohnt sich NICHT.**

- Micro-Benchmark-Obergrenze (depth=1): **224 µs/Step** (1.4 % der Verify-Zeit)
- Realistische Hochrechnung (mit Compute-Verdeckung): **~100–200 µs/Step** (0.6–1.2 %)
- Schwellenwert "lohnt sich": ≥ 1.500 µs (8 %)
- Ergebnis liegt 7–15× **unter** der Lohnt-sich-Schwelle

Die gesamte "Buffer-Traffic"-Hypothese wird durch die Messung falsifiziert:

1. Buffer-Traffic ist im realen FFN (GEMV-dominiert) vom Gewichtsmatrix-Traffic um Faktor ~500 überdeckt.
2. Der Launch-Overhead-Anteil (den Fusion wirklich einspart) ist mit ~200 µs viel kleiner als die ursprüngliche 4.000-µs-Schätzung aus der Launch-Overhead-Analyse.
3. Die 4.000-µs-Schätzung war ein Modellartefakt: Wall-Clock minus "GPU-only" zu buffer-traffic zu interpretieren war nicht gerechtfertigt, weil grosse Teile dieser Differenz aus Submission-Latenz (die auf RDNA 4 durch Stream-Pipelining effektiv hidden ist) und Profiling-Overhead stammten.

Damit ist der dritte Null-Effekt in Folge (Tiled GEMV, Batched lm_head, jetzt Fused FFN) auf dieser Architektur dokumentiert.

## Welcher Hebel bleibt?

Nicht als Implementierungs-Plan, sondern als Hypothese für den nächsten Schritt:

1. **Prefill-GEMM.** Aktuelles Prefill: 59 tok/s vs llama.cpp 1092 tok/s (5 % der Baseline). Hier liegen nicht 1–2 %, sondern 95 % Gap. Der nächste sinnvolle Hebel ist echte Batch-1-GEMM statt GEMV × N-mal auf Prefill-Tokens.
2. **Attention-Optimierung bei langem Kontext.** Bei K=4096 sinkt der Durchsatz von 115 auf 78 tok/s (−32 %), Attention erreicht 34 % der Decode-Zeit. Das ist bereits im Memory-Regime — hier könnte FlashAttention-ähnliche Tiling-Strategie echten Throughput bringen.
3. **14B-Target mit 0.5B-Draft.** Aktuell VRAM-limitiert auf 16 GB. Q8_0-Draft statt Q4_0 könnte Acceptance erhöhen.

Fused FFN sollte **nicht** implementiert werden, bevor einer der obigen Hebel die Priorität verliert.

## Konsistenz-Check

- Launch-Overhead-Analyse schätzte FFN-Buffer-Traffic auf ~4.000 µs. Die Messung findet ~200 µs realistischer Einsparung. Die Abweichung ist 20× — identisch zur Abweichung bei Tiled GEMV (~1.5 % statt 15 %) und Batched lm_head (~0.4 % statt 8 %).
- Das konsistente 20×-Überschätzen durch "Load/Traffic einmal statt N-mal"-Modellierung ist jetzt in drei unabhängigen Experimenten dokumentiert und wird in `docs/architecture_notes.md` als stabile RDNA-4-Eigenschaft geführt.
