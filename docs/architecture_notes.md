# Architecture Notes

Empirische Befunde zur GPU-Mikroarchitektur, die projektübergreifend relevant sind und Optimierungsentscheidungen leiten.

## Memory-Controller-Pipelining bei sequentiellen GEMV-Dispatches (RDNA 4)

**Hardware:** RX 9070 XT (gfx1201, RDNA 4), ROCm 7.2.1

Auf RX 9070 XT (gfx1201, RDNA 4) pipelined der Memory-Controller sequentielle GEMV-Dispatches auf dieselbe Gewichtsmatrix so effizient, dass batched Dispatches keinen messbaren Bandbreitenvorteil bringen.

Bestätigt durch zwei unabhängige Experimente:

1. **Tiled GEMV FFN-Down** (~1.5% Throughput-Gewinn statt erwarteter ~15%): Die FFN-Down-Projektion (18944→3584, ~135 MB Q4_0) lädt die Gewichtsmatrix bei sequentiellem Fallback N-mal. Der Tiled-Kernel lädt sie 1×. Erwartet wurde ein Gewinn proportional zur eingesparten Bandbreite — gemessen wurden nur ~9 μs/Layer statt ~150 μs/Layer. Dokumentiert in `docs/batched_verify.md` (Abschnitt "Memory-Controller Pipelining").

2. **Batched lm_head** (~0.4% Verify-Overhead-Reduktion statt erwarteter ~8%): Die lm_head-Projektion (3584→152064, ~307 MB Q4_0) wurde von N sequentiellen Dispatches auf 1 batched Dispatch umgestellt. Non-layer-overhead sank um ~114 μs/Step statt der vorhergesagten ~2.500 μs. Dokumentiert in `benches/results/batched_lm_head_analysis.md`.

### Mechanismus

Der GPU Command Processor überlappt den Tail eines GEMV-Kernels mit dem Head des nächsten, wenn beide auf dieselbe Adressbereiche zugreifen und keine explizite Synchronisation dazwischen liegt. Der Memory-Controller hält dabei nahezu volle Bandbreitenauslastung über Kernel-Grenzen hinweg aufrecht. Das "Load once instead of N times"-Modell überschätzt den Gewinn um Faktor ~20×.

### Konsequenz für Optimierungen

Der Optimierungshebel auf dieser Architektur liegt bei der **Elimination von intermediate Buffer Traffic zwischen Kernels mit unterschiedlichen Zugriffsmustern**, nicht bei Dispatch-Batching für bandbreitenlimitierte Kernels.

Konkret:
- **Fused FFN** (gate+up+SiLU+mul+down+residual → weniger Kernels): Der Hauptgewinn kommt nicht aus weniger Dispatches, sondern aus dem Wegfall der Zwischenpuffer-Roundtrips (gate→mul→down liest/schreibt je ~75 KB intermediate data pro Layer).
- **Batching** bandwidth-bound Operationen (GEMV, Attention mit langem KV-Cache) bringt nur marginale Dispatch-Overhead-Einsparung (~2.7 μs/Dispatch + Sync-Elimination).
- **Kernel-Fusion** ist nur dann lohnend, wenn sie *unterschiedliche* Speicherzugriffsmuster zusammenführt (z.B. elementwise + GEMV eliminiert einen Store/Load-Zyklus), nicht wenn sie identische Zugriffsmuster batcht.

### Offene Frage

Ob dieser Pipelining-Effekt auch auf RDNA 3 (gfx1100, RX 7900 XT) in gleichem Maße auftritt, ist nicht gemessen. Die Memory-Controller-Architektur unterscheidet sich (Infinity Cache vs. kein Infinity Cache auf RDNA 4). Ein Vergleichsexperiment auf gfx1100 wäre aufschlussreich.
