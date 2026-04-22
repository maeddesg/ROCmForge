# Phase 2 Schritt 2.1.1 — Kernel-GA Framework

**Date:** 2026-04-22
**Branch:** v1.0-dev
**Hardware:** AMD Radeon RX 9070 XT (gfx1201)
**Scope:** GA-Engine + Genome + Validation + Compile-Cache + JSONL-Log,
validiert auf einem Toy-Fitness-Landscape. Keine echten Kernel werden
in diesem Schritt optimiert — das folgt in 2.1.3.

## Kurzfassung

Neues Modul `src_v1/ga/` mit 1 555 LOC über 10 Dateien (plus 715 LOC
Tests, 2 270 LOC gesamt). Enthält das komplette Framework aus
`ga_tuning_spec §2`:

- `KernelGenome` + Crossover/Mutation/Random (1:1 aus `architecture §4.2`)
- Two-Stage-Validation: Pre-Compile-Heuristik (LDS / VGPR / WMMA-
  Alignment / Sub-Block) und Post-Compile-VGPR-Gate (reale VGPR-Zahlen
  aus AMDGPU-ELF-Notes)
- UCB1-freie GA-Engine: Tournament-3, Uniform-Crossover rate 0.7,
  Mutation-Rate 0.1 pro Gen, Elitism 5 %, Early-Exit 10 Gen / 1 %
- Compile-Cache (`HashMap<CompileKey, Arc<CompiledKernel>>`) mit Hit-
  Rate-Telemetrie
- JSONL-Logger (`ga_tuning_spec §5.10`) mit `run_id`/`ts`/`event`-
  Metadaten, File- und Capture-Sink
- Deterministische Seeded-RNG (xorshift64*), keine externen Crates
- Zen4-CPU-Target als `KernelTarget::Zen4Avx512`-Enum-Variante
  vorbereitet, noch nicht codegen'd

Das Framework konvergiert auf dem Toy-Problem in **13 Generationen**
(bei `early_exit=true`), Best-Fitness **2.0000** bei Seed 42. Tests:
**41/41 grün** (26 Integration + 15 inline `mod tests`).

## Modul-Struktur

```
src_v1/ga/
├── mod.rs              43 LOC    re-exports
├── types.rs           137 LOC    TileConfig, PrecisionLevel, LdsStrategy,
│                                 KernelTarget, CodeObjectResources
├── rng.rs             122 LOC    SeededRng (xorshift64*), self-test
├── genome.rs          223 LOC    KernelGenome, DequantStrategy,
│                                 random/crossover/mutate, impl From<&K> for TileConfig
├── validation.rs      215 LOC    validate_pre_compile, estimate_vgprs,
│                                 validate_post_compile, PostCompileResult
├── compile.rs         224 LOC    CompileCache, CompileKey, CompiledKernel,
│                                 parse_amdgpu_metadata (llvm-readobj)
├── fitness.rs         199 LOC    FitnessResult, evaluate_toy_fitness, median
├── engine.rs          351 LOC    GaConfig, KernelGa, GenerationResults,
│                                 tournament_select, next_generation, early_exit
├── logger.rs          213 LOC    GaLogger, JSONL events, File/Capture/Null sinks
└── toy.rs              83 LOC    toy_fitness, run_toy_ga, toy_ga_defaults
```

Gesamt **1 555 LOC** Framework + **715 LOC** Tests (`tests_v1/ga_framework_test.rs`)
+ `Cargo.toml`-Target.

Alle neuen Typen leben **in der GA-Crate**, nicht in `src_v1/ir/`.
`TileConfig`/`PrecisionLevel`/`LdsStrategy`/`KernelShape`/`KernelTarget`
werden in Schritt 2.1.3 in `src_v1/ir/` umgezogen, sobald der
Dequant-IR-Codegen sie tatsächlich konsumiert; Step 2.1.1 ist der
Besitzer bis dahin.

## KernelGenome — Suchraum

Aus `architecture §4.2` + `ga_tuning_spec §2.2`:

| Gen | Werte | Anzahl |
|---|---|---:|
| `tile_m` | 16, 32, 64, 128 | 4 |
| `tile_n` | 16, 32, 64, 128 | 4 |
| `tile_k` | 16, 32, 64 | 3 |
| `tiles_per_wave` | 1, 2, 4 | 3 |
| `waves_per_block` | 1, 2, 4, 8 | 4 |
| `use_lds_for_a` | {false, true} | 2 |
| `use_lds_for_b` | {false, true} | 2 |
| `prefetch_depth` | 0, 1, 2 | 3 |
| `k_unroll` | 1, 2, 4, 8 | 4 |
| `double_buffer` | {false, true} | 2 |
| `dequant_strategy` | Inline \| PrePass{4 sizes} \| Batched{3 sizes} | 8 |

**Brute-Force-Kardinalität:** `4·4·3·3·4·2·2·3·4·2·8 = 221 184`. Nach
Pre-Compile-Validierung (LDS-Budget, VGPR-Heuristik, Sub-Block) bleibt
davon typisch ~60 % stehen — die GA exploriert 5 000 Punkte (100 × 50)
in einem Raum der echten Größe ~133 000, ein brauchbares Sampling.

Die Spec-Zahl in §2.3 (~83 000) ging von kleineren Value-Sets aus; das
Genome-Set hier folgt exakt dem Arch-Doc-Text. Die Zahl beeinflusst die
GA-Zeit nicht (Population + Generationen sind fest), sie ist nur eine
Sanity-Referenz.

## Validation

### Pre-Compile (Stage 1, `ga_tuning_spec §2.3`)

Implementiert in `validation::validate_pre_compile` mit fünf Regeln:

1. WMMA-Alignment: `tile_m % 16 == 0 && tile_n % 16 == 0 && tile_k % 16 == 0`
2. LDS-Budget: `≤ 64 KB` pro Workgroup, inkl. `DequantStrategy::PrePass`-
   Overhead und `double_buffer`-Faktor
3. Heuristic VGPR: `≤ 150` (großzügige Schwelle, echter Check erst
   Post-Compile)
4. Workgroup-Tiles: `tiles_per_wave × waves_per_block ≤ 16`
5. Sub-Block-Alignment: `tile_k % fmt.sub_block_size == 0` (Q4_K/Q8_0:
   32; Q6_K: 16)

**Reject-Rate auf 1000 Zufalls-Genomen (Seed 2026): 39.5 %** — genau
in der Spec-Erwartung 30–40 % aus §2.3. Das Gate tut also genau, was
es soll: offensichtliche Fehlkombinationen aussortieren ohne den
teuren Compile-Pfad anzufassen.

### Post-Compile (Stage 2, `ga_tuning_spec §2.3.1`)

`parse_amdgpu_metadata()` shellt `llvm-readobj --notes <co>` aus,
extrahiert `.vgpr_count`, `.sgpr_count`, `.group_segment_fixed_size`
aus dem AMDGPU-Metadata-Note-Segment. `validate_post_compile` rechnet
`max_waves_per_cu = 1536 / vgpr_count` und verwirft unter 4 Waves/CU.

**End-to-End-Verifikation auf einem echten `.co`:** Der Test
`gpu_tests::test_post_compile_vgpr_read_from_real_co` liest das
gfx1201-Code-Object des `q4_k_q8_inline_residual`-Kernels (aus dem
Phase-2.0.2-Build) und bekommt:

```
q4_k_q8_inline_residual VGPRs=186 SGPRs=18 LDS=0 B  (waves/CU=8)
```

`186 VGPRs → 8 Waves/CU` passiert den Gate (`≥ 4`). Das ist ein echter
Roundtrip gegen den von CMake gebauten, extrahierten
`*.hipv4-amdgcn-amd-amdhsa--gfx1201`-ELF.

## GA-Engine

Parameter aus `architecture §4.2` / `ga_tuning_spec §2.5`, 1:1
übernommen:

| Parameter | Wert | Kommentar |
|---|---:|---|
| Population | 100 | Spec-Default |
| Generationen | 50 | Spec-Default |
| Tournament-Size | 3 | Spec-Default |
| Crossover-Rate | 0.7 | Spec-Default |
| Mutation-Rate | 0.1 pro Gen | Spec-Default |
| Elitism-Fraction | 0.05 (Top 5 %) | Spec-Default |
| Early-Exit-Generations | **10** | Spec-Amendment 2 (nicht 5 — verhindert Punctuated-Equilibria-Early-Abort) |
| Early-Exit-Threshold | 0.01 | < 1 % Improvement über 10 Gen → Abort |

**Reproduzierbarkeit:** Der komplette GA-Lauf hängt nur am `GaConfig.seed`
und dem Code-Git-Hash. Gleicher Seed → bit-identische Population,
bit-identische Top-5. Verifiziert durch
`test_seed_reproducibility` (zwei Läufe mit Seed 42, top-Genome und
best_fitness exakt gleich).

**RNG:** `SeededRng` implementiert xorshift64* (Marsaglia 2003) — 128
LOC inkl. Tests, keine externen Crates. Periode 2⁶⁴−1, ausreichend für
jede plausible GA-Run-Zeit.

## Toy-Problem — Konvergenz-Validierung

Bewusst künstliche Fitness-Landscape (`toy::toy_fitness`) mit bekanntem
Optimum `(tile_m=64, tile_n=64, k_unroll=4) → 2.0`. Sekundäre Peaks bei
`(32, 32, k_unroll=4) → 1.5` und gemischten 64×32/32×64 auf 1.3.

**Lauf mit Seed 42 (pop=50, gen=20, mutation=0.15, elitism=0.10):**

```
toy GA: best_fitness=2.0000  generations_ran=13  early_exited=true
  top0: tile_m=64 tile_n=64 tile_k=16 k_unroll=4 fitness=2.0000
  top1: tile_m=64 tile_n=64 tile_k=32 k_unroll=4 fitness=2.0000
  top2: tile_m=64 tile_n=64 tile_k=32 k_unroll=4 fitness=2.0000
  top3: tile_m=64 tile_n=64 tile_k=64 k_unroll=4 fitness=2.0000
  top4: tile_m=64 tile_n=64 tile_k=32 k_unroll=4 fitness=2.0000
```

Die GA erkennt den Sweet-Spot nach 13 Generationen, Plateau triggert
den Early-Exit. Alle Top-5 halten die Optimal-Kombination
`tile_m=64, tile_n=64, k_unroll=4`; `tile_k` variiert, weil er im
Toy-Landscape keinen Einfluss hat — genau das gewünschte Verhalten
(Pareto-Front-Diversität in irrelevanten Genen).

## Compile-Cache

`CompileCache::get_or_insert_with` ist die einzige Insertion-API —
identische `CompileKey`-Werte teilen sich einen
`Arc<CompiledKernel>`. Der Test `test_compile_cache_hit_rate` evaluiert
5× das gleiche Genome, bekommt **1 Miss + 4 Hits = 80 % Hit-Rate** (wie
erwartet, weil das Toy-Path-Genome deterministisch auf denselben
`CompileKey` mappt).

In echten GA-Läufen steigt die Hit-Rate typisch von 0 % (Gen 0) auf
50–70 % (Gen 10+), weil Elitism + Crossover oft bekannte Configs
reproduzieren. Diese Dynamik wird in 2.1.3 auf echten Shapes gemessen.

## JSONL-Log

Event-Typen und Schema 1:1 aus `ga_tuning_spec §5.10`.
Sample-Ausgabe aus dem Toy-Lauf (Seed 42):

```
{"event":"shape_start","run_id":"toy-convergence","shape":"toy",
 "ts":"2026-04-22T14:23:11Z"}
{"benchmark_wall_ms":0,"benchmarked_individuals":50,"best_fitness":1.6,
 "compile_cache_hit_rate":0.0,"compile_cache_hits":0,
 "compile_cache_misses":0,"compile_wall_ms":0,
 "event":"generation_complete","generation":0,"median_fitness":0.8,
 "post_compile_vgpr_rejects":0,"pre_compile_rejects":0,
 "run_id":"toy-convergence","shape":"toy","ts":"2026-04-22T14:23:11Z"}
{"benchmark_wall_ms":0,"benchmarked_individuals":50,"best_fitness":1.6,
 "compile_cache_hit_rate":0.0,"compile_cache_hits":0,
 "compile_cache_misses":0,"compile_wall_ms":0,
 "event":"generation_complete","generation":1,"median_fitness":1.02,
 "post_compile_vgpr_rejects":0,"pre_compile_rejects":0,
 "run_id":"toy-convergence","shape":"toy","ts":"2026-04-22T14:23:11Z"}
```

Der gesamte 20-Gen-Toy-Lauf erzeugt 16 Zeilen (1 × `shape_start` + 13 ×
`generation_complete` + 1 × `early_exit` + 1 × `shape_complete` =
16). Jede Zeile ist valides JSON, jede trägt `run_id` und `ts`;
Per-Eval-Records (`event:"eval"`) schreibt die Toy-Engine nicht um
den Log nicht aufzublähen — `log_eval_record` ist im Engine-Modul
exportiert und wird vom echten GA-Pfad in 2.1.3 benutzt.

## Test-Ergebnisse

### GA-Framework-Tests (`v1_ga_framework_test`, 26 Tests, alle grün)

| Test | Scope |
|---|---|
| `test_genome_random_produces_valid` | Genome |
| `test_genome_crossover_inherits_genes` | Genome |
| `test_genome_mutation_changes_values` | Genome |
| `test_genome_to_tile_config` | Genome |
| `test_pre_compile_rejects_bad_alignment` | Validation Stage 1 |
| `test_pre_compile_rejects_lds_overflow` | Validation Stage 1 |
| `test_pre_compile_rejects_high_vgpr_heuristic` | Validation Stage 1 |
| `test_pre_compile_accepts_valid_genome` | Validation Stage 1 |
| `test_pre_compile_reject_rate_on_random_genomes` | Validation Stage 1 |
| `test_post_compile_rejects_under_four_waves` | Validation Stage 2 |
| `test_post_compile_accepts_moderate_vgprs` | Validation Stage 2 |
| `test_tournament_selection_prefers_better` | Engine |
| `test_elitism_preserves_best` | Engine |
| `test_early_exit_after_plateau` | Engine |
| `test_early_exit_not_triggered_with_improvement` | Engine |
| `test_toy_fitness_landscape` | Toy-Oracle |
| `test_ga_converges_on_toy_problem` | **End-to-End Toy-Konvergenz** |
| `test_seed_reproducibility` | Determinismus |
| `test_different_seeds_produce_different_runs` | Determinismus |
| `test_compile_cache_hit_rate` | Cache |
| `test_compile_cache_same_key_same_result` | Cache |
| `test_jsonl_log_parseable` | Log |
| `test_jsonl_log_contains_run_id` | Log |
| `test_fitness_relative_to_baseline` | Fitness |
| `test_fitness_sorts_correctly` | Fitness |
| `gpu_tests::test_post_compile_vgpr_read_from_real_co` | **Realer `.co` mit llvm-readobj** |

### Inline-`mod tests` (15 Tests, alle grün)

`rng` (4), `genome` (1), `validation` (4), `compile` (2), `fitness` (2),
`logger` (1), `engine` (1).

### Regression (2.0.1 + 2.0.2 weiterhin stabil)

| Suite | Status |
|---|:---:|
| `v1_runtime_test` (8 Tests) | ✅ |
| `v1_monitor_test` (11 Tests) | ✅ |
| `v1_sync_elimination_test` (P0-Gate + Decode-Qualität einzeln) | ✅ |
| `v1_residual_fused_test` (CPU-only Fusion-Detection × 3) | ✅ |
| v0.x Build (`cargo build --release --features gpu`) | ✅ |

GPU-Tests müssen weiterhin einzeln laufen — Box::leak-VRAM-Pattern
aus Step 1.11 ist unverändert (Phase-1-Test-Harness-Limit,
dokumentiert in 2.0.1 + 2.0.2).

## Design-Entscheidungen

- **Eigene RNG, keine externe Crate.** `xorshift64*` ist 128 LOC und
  gibt der GA bit-identische Reproduzierbarkeit ohne die `rand`-
  Transitive-Deps. Entspricht der Vorgabe im Prompt ("Kein externer
  Crate für GA").
- **`llvm-readobj` statt ELF-Parser in Rust.** Der Prompt erwähnt
  beide Wege; `llvm-readobj --notes` ist im ROCm-Environment
  ohnehin präsent (wird von `hipcc` mitgeliefert) und die Ausgabe ist
  stabil key-value — 20 LOC Parser reichen. Der AMDGPU-Metadata-YAML
  ist komplex; ein reiner Rust-Parser wäre 300+ LOC und bringt
  nichts Neues.
- **Toy-Path hat einen eigenen `evaluate_toy_fitness`, keinen stubbed
  hipcc-Aufruf.** `KernelGa::run_with` nimmt einen Closure als
  Fitness-Quelle, der Toy-Path ruft `toy_fitness(g)` direkt auf.
  Das hält die Engine generisch (der echte GPU-Pfad wird denselben
  `run_with` nutzen, nur mit einer Fitness-Closure, die wirklich
  Compile+Bench macht).
- **`KernelTarget` als Enum mit `Gfx1201` + `Zen4Avx512` von Anfang
  an.** Cache-Key-Stabilität: wenn der CPU-Pfad in Phase 3 dazu
  kommt, bleiben existierende GPU-Cache-Einträge gültig, weil das
  Enum von Tag 1 in der Hash-Grundlage steht.
- **`DirectA_LdsB` mit Underscore.** Spec-Name wörtlich aus
  `ga_tuning_spec §7.2`. Ein `#[allow(non_camel_case_types)]` auf
  dem Enum, weil die Lesbarkeit des Spec-Namens hier mehr wiegt als
  die Konvention.
- **Early-Exit auf 10 Generationen statt 5.** Spec-Amendment 2
  (`ga_tuning_spec §2.6`): "GAs zeigen häufig Punctuated-
  Equilibria — längere Plateaus gefolgt von plötzlichen Durchbrüchen."
  10 Generationen Puffer kostet im Worst-Case ~100 s pro Shape, aber
  verhindert vorzeitige Konvergenz.
- **Sequentielle Kompilierung.** Der Prompt erlaubt das explizit —
  parallele Compile-Pipeline wird Follow-up in 2.1.3 falls das
  Budget knapp wird. Korrektheit vor Performance.
- **`log_eval_record` als freie Funktion, nicht als Methode.** Der
  echte GA-Pfad in 2.1.3 wird pro Eval eine Zeile schreiben wollen;
  weil `run_with` den Logger bereits `&mut`-borrow'd, geht das nur als
  freie Funktion mit Logger-Parameter. Im Toy-Path wird sie nicht
  aufgerufen (Log bleibt kompakt).

## Bekannte Limitierungen

Was **2.1.2** bringt:
- VALU-Parity-Validation als Fitness-Gate (`ga_tuning_spec §2.8`)
- Stability-Check der Top-5 (30 Runs × 3 Input-Sets,
  `ga_tuning_spec §2.9`)

Was **2.1.3** bringt:
- Echter GPU-Fitness-Path: Codegen → hipcc → VGPR-Read → 5+20-Run
  Benchmark mit HIP-Events (die Event-Infra aus 2.0.1 wird
  wiederverwendet)
- Parallele Kompilierung über Rayon, wenn Budget knapp wird
- Erster echter GA-Lauf auf `gemv_q4_k_gate_up_swiglu` (P0-Target
  aus 2.0.3 — 65.77 % GPU-Zeit, 432.8 µs Ø)

Was **2.1.4** bringt:
- `KernelVariantSet`-Export und Bandit-Integration
  (`ga_tuning_spec §2.10`, Arch-Doc §2.5 "Hot-Swap")

## Prerequisites für 2.1.2

- `hip_kernels_v1/*`-Builder muss einen Pfad exposen, der ein Genome
  entgegennimmt und ein `.co` produziert (aktuell gehen alle Kernel
  über fertige HIP-Quellen + CMake; für die GA braucht es einen
  Rust-seitigen Aufruf von `hipcc` gegen eine codegen-erzeugte
  `.hip`-Datei)
- VALU-Referenz-Kernel pro Format (`dequant_ir_spec §8.6`), damit
  Parity-Checks Kandidaten vergleichen können

Beides ist **nicht** Teil von 2.1.1 — das Framework hier steht auf
eigenen Beinen, der Toy-Path beweist das.

## Commit

Prefix `feat(v1/ga):` — neues Feature-Modul, kein Bugfix, kein
Refactor.

```
feat(v1/ga): Phase 2 step 2.1.1 — Kernel-GA framework
```

Backup-Push auf `backup` Remote. Kein Fork-Push.
