# Architecture Notes

Empirische Befunde zur GPU-Mikroarchitektur, die projektübergreifend relevant sind und Optimierungsentscheidungen leiten.

## Memory-Controller-Pipelining bei sequentiellen GEMV-Dispatches (RDNA 4)

**Hardware:** RX 9070 XT (gfx1201, RDNA 4), ROCm 7.2.1

Auf RX 9070 XT (gfx1201, RDNA 4) pipelined der Memory-Controller sequentielle GEMV-Dispatches auf dieselbe Gewichtsmatrix so effizient, dass batched Dispatches keinen messbaren Bandbreitenvorteil bringen.

Bestätigt durch drei unabhängige Experimente:

1. **Tiled GEMV FFN-Down** (~1.5% Throughput-Gewinn statt erwarteter ~15%): Die FFN-Down-Projektion (18944→3584, ~135 MB Q4_0) lädt die Gewichtsmatrix bei sequentiellem Fallback N-mal. Der Tiled-Kernel lädt sie 1×. Erwartet wurde ein Gewinn proportional zur eingesparten Bandbreite — gemessen wurden nur ~9 μs/Layer statt ~150 μs/Layer. Dokumentiert in `docs/batched_verify.md` (Abschnitt "Memory-Controller Pipelining").

2. **Batched lm_head** (~0.4% Verify-Overhead-Reduktion statt erwarteter ~8%): Die lm_head-Projektion (3584→152064, ~307 MB Q4_0) wurde von N sequentiellen Dispatches auf 1 batched Dispatch umgestellt. Non-layer-overhead sank um ~114 μs/Step statt der vorhergesagten ~2.500 μs. Dokumentiert in `benches/results/batched_lm_head_analysis.md`.

3. **Buffer-Traffic-Validierung (Fused FFN)** (~1.2% Verify-Overhead-Reduktion statt erwarteter 8–14%): Micro-Benchmark simulierte die FFN-Kette in drei Varianten — separate Kernels mit VRAM-Zwischenpuffern, single Dispatch mit VRAM-Roundtrips, und tiled Fusion mit Registern/LDS. Maximaler erreichbarer Gewinn: ~224 µs/Step. Die intermediate-Buffer-Traffic (~150 KB/Layer) wird vom Gewichtsmatrix-Traffic (~150 MB/Layer) um Faktor 500 überdeckt. Dokumentiert in `profiling/results/BUFFER_TRAFFIC_ANALYSIS.md`.

### Mechanismus

Der GPU Command Processor überlappt den Tail eines GEMV-Kernels mit dem Head des nächsten, wenn beide auf dieselbe Adressbereiche zugreifen und keine explizite Synchronisation dazwischen liegt. Der Memory-Controller hält dabei nahezu volle Bandbreitenauslastung über Kernel-Grenzen hinweg aufrecht. Das "Load once instead of N times"-Modell überschätzt den Gewinn um Faktor ~20×.

### Konsistentes Überschätzungsmuster bei Bandbreiten-Modellierung

Die drei Experimente zeigen dasselbe Muster: naive Bandbreiten-Rechnungen überschätzen den Gewinn auf RDNA 4 konsistent um eine Grössenordnung.

| Experiment                    | Erwarteter Gewinn    | Gemessener Gewinn | Überschätzungsfaktor |
|-------------------------------|---------------------:|------------------:|---------------------:|
| Tiled GEMV FFN-Down           |         2–4 ms/Step  |      ~250 µs/Step |                 ~12× |
| Batched lm_head               |       ~2.500 µs/Step |      ~114 µs/Step |                 ~22× |
| Buffer-Traffic-Validierung    | ~1.500–2.500 µs/Step |      ~200 µs/Step |                 ~10× |

Das ist kein Messfehler, sondern eine stabile Eigenschaft der Memory-Pipeline auf dieser Architektur. Das naive Modell behandelt Memory-Traffic wie einen seriellen Kostenblock, der proportional zur Anzahl Dispatches skaliert. Tatsächlich pipelined der Memory-Controller die Zugriffe so, dass der zweite, dritte, … N-te Zugriff auf dieselben Adressen nahezu kostenlos wird (L2/L3-Hits, überlappendes Streaming, keine erneute Memory-Request-Latenz).

Konsequenz: **"Load N times" ist auf RDNA 4 fast gleich teuer wie "Load once", solange die Zugriffe auf dieselben Adressbereiche ohne explizite Synchronisation erfolgen.** Optimierungen, die ausschliesslich N-fache Lasten auf 1× reduzieren, bringen keinen messbaren Gewinn.

### Konsequenz für Optimierungen

Der Optimierungshebel auf dieser Architektur liegt bei **Algorithmuswechseln** und bei **Compute-Patterns mit nicht-vorhersagbaren Zugriffsmustern** — nicht bei Dispatch-Batching oder Buffer-Traffic-Elimination für bandbreitenlimitierte Kernels.

Konkret:

- **Fused FFN wurde durch Micro-Benchmark als nicht wirtschaftlich bestätigt** (~200 µs realistischer Gewinn, Schwellenwert war 1.500 µs). Nicht implementieren. Die dominante FFN-Kostenposition ist der Gewichtsmatrix-Traffic, der durch Fusion nicht eliminiert wird.
- **Spec-Decode-Verify-Optimierung hat das Plateau erreicht.** Target-Verify ist zu ~88% GEMV-Execution gegen die Gewichtsmatrix — bandbreitenlimitiert bei ~640 GB/s (RX 9070 XT Spec), nicht durch Dispatch-Overhead oder Buffer-Traffic. Weitere Micro-Optimierungen innerhalb des GEMV-Paradigmas bringen < 2%.
- **Batching bandwidth-bound Operationen** (GEMV, Attention mit langem KV-Cache) bringt nur marginale Dispatch-Overhead-Einsparung (~2.7 µs/Dispatch + Sync-Elimination). Stream-Pipelining deckt die Sync-Elimination bereits ab.
- **Kernel-Fusion** ist nur dann lohnend, wenn sie *unterschiedliche* Speicherzugriffsmuster zusammenführt (z.B. elementwise + GEMV eliminiert einen Store/Load-Zyklus für nicht-gecachte Adressen), nicht wenn sie identische Zugriffsmuster batcht oder Zwischenpuffer eliminiert, die bereits in L2 passen.
- **Optimierungshebel liegen bei Algorithmuswechseln** (GEMV → GEMM für Prefill) und bei **Compute-Patterns mit nicht-vorhersagbaren Zugriffsmustern** (Attention-Tiling bei langem Kontext, wo der KV-Cache aus L2 fällt). Diese unterscheiden sich qualitativ von den bisherigen Experimenten, weil sie andere Memory-Access-Patterns haben, bei denen das Memory-Controller-Pipelining weniger greift.

### Offene Fragen

- Ob dieser Pipelining-Effekt auch auf RDNA 3 (gfx1100, RX 7900 XT) in gleichem Maße auftritt, ist nicht gemessen. Die Memory-Controller-Architektur unterscheidet sich (Infinity Cache vs. kein Infinity Cache auf RDNA 4). Ein Vergleichsexperiment auf gfx1100 wäre aufschlussreich.
- Ob das Überschätzungsmuster auch bei **nicht-elementwise** Fusionen auftritt (z.B. Attention + FFN in einem Kernel). Hypothese: ja, solange die dominante Kostenposition bandbreitenlimitierte GEMV bleibt — sobald die Kostenposition kippt (Compute-bound, irreguläre Zugriffe), ändert sich das Bild.
- Ob **WMMA/Matrix-Instruktionen** auf RDNA 4 das Bild ändern. Diese nutzen eine andere Execution-Pipeline (Matrix-Cores) und könnten andere Pipelining-Charakteristika haben — insbesondere wenn der Matrix-Core-Scheduler anders mit dem Memory-Controller interagiert als der Vector-ALU-Scheduler.

## CPU-Zielplattform

**Primäre CPU:** AMD Ryzen 9 7945HX (Zen4, 16C/32T, AVX-512 VNNI, 64 MB L3, DDR5 Dual-Channel ~77 GB/s)

ROCmForge hat einen CPU-Fallback-Pfad (`--gpu` nicht gesetzt), der aktuell nicht SIMD-optimiert ist. Zen4 bietet AVX-512 mit VNNI-Erweiterungen, die INT8-Dot-Products in Hardware beschleunigen — direkt relevant für Q4_0/Q8_0-Inferenz.

Optimierungsansätze:

- **AVX-512 GEMV-Kernel für Q4_0:** Grösster Einzelhebel. Q4_0-Blöcke entpacken, gegen Q8_0-quantisierten Input multiplizieren, per VNNI-Instruktionen akkumulieren. 512-bit-Register verarbeiten 64 INT8-Werte pro Takt (2× AVX2, 4× SSE).
- **Multi-Threaded Inference:** Output-Dimension der GEMV über Threads partitionieren. Bei 3584 Output-Elementen und 16 Kernen: 224 Elemente pro Thread.
- **Cache-bewusstes Tiling:** L2 (1 MB/Kern) und L3 (64 MB shared) für Weight-Tiles nutzen, DRAM-Zugriffe minimieren.
- **Heterogenes Spec-Decode:** Draft-Modell (0.5B) auf CPU, Target (7B) auf GPU, parallel. Eliminiert die ~10 % Draft-GPU-Overhead aus der Spec-Step-Kostenanalyse. Voraussetzung: CPU-Pfad muss schnell genug sein, um die GPU nicht zu blockieren.

Das Memory-Controller-Pipelining-Muster (RDNA-4-Abschnitt oben) gilt nicht 1:1 für CPU-DRAM. Zen4 hat eigene Prefetcher und eine andere Memory-Hierarchie (L1/L2 pro Core, L3 shared, DDR5-Controller) — Optimierungsheuristiken müssen empirisch validiert werden, bevor die RDNA-4-Erkenntnisse übertragen werden.
