# ROCmForge v1.0 — GA Tuning Specification

**Dokument:** `docs/v1.0/ga_tuning_spec.md`
**Version:** 1.0.1-final (Amendments eingearbeitet)
**Datum:** 2026-04-21
**Status:** Final (für Phase-2-Implementierung)
**Basiert auf:** `architecture_v1.2.0-draft.md` §4, `dequant_ir_spec.md` §4.4/§8, `phase1_step_1.17_rocprof_baseline.md`

**Changelog seit 1.0.0:**
- **§2.3 / §2.3.1 (neu):** Two-Stage-VGPR-Gate. Pre-Compile-Heuristik
  (großzügiger Schwellenwert 150) + Post-Compile-Check gegen echte
  VGPR-Zahlen aus dem Code-Object. Verhindert Fitness-Verzerrung durch
  LLVM-Spilling.
- **§2.4 Fitness-Funktion:** Ablauf-Reihenfolge auf 5-Phasen-Modell
  aktualisiert (Pre-Validate → Compile → Post-Validate → Warmup →
  Benchmark).
- **§2.6 / §3.8:** Early-Exit-Schwelle von 5 auf **10 Generationen**
  erhöht — GAs zeigen Punctuated-Equilibria-Verhalten, 10 Gen (20 %
  Budget) statt 5 (10 %) gibt Plateau-Durchbrüchen Raum.
- **§2.7.1 (neu):** Compile-Pipeline-Spezifikation — parallel Compile
  (CPU-Threads) + sequenzielles Benchmark (GPU) + aggressiver Compile-
  Cache. Macht das 8-min/Shape-Budget einhaltbar.
- **§2.7.2 (neu):** CPU-Thread-Management für Compile-Phase.
- **§2.9:** Top-K-Report enthält jetzt echte Post-Compile-Ressourcen.
- **§5.10:** Log-Schema um Pipeline-Telemetrie erweitert.

---

## Verhältnis zu anderen Dokumenten

Dieses Dokument ist die dritte und letzte Spec der v1.0-Architektur.
Vorgänger:

1. **`architecture_v1.2.0-draft.md`** (Blueprint, 6 Säulen) — hat
   **Vorrang** bei allen Konflikten. Struct-Namen, Genome-Definitionen,
   GA-Parameter folgen diesem Dokument exakt. Abweichungen sind als
   `[SPEC-ERWEITERUNG]` markiert.
2. **`dequant_ir_spec.md`** (Säule 3) — liefert die Kernel-Emission-
   Infrastruktur, auf der die Kernel-GA aufbaut. `PrecisionLevel`,
   `KernelVariantSet`, `DriftEvent` kommen aus dort.
3. **`phase1_step_1.17_rocprof_baseline.md`** — die **empirische
   Grundlage** für jede GA-Priorisierung. Jede Performance-Aussage in
   diesem Dokument referenziert eine Zahl aus 1.17. Keine Schätzungen
   ohne Messung.

---

## Inhaltsverzeichnis

1. Überblick: drei GAs und wie sie zusammenspielen
2. Kernel-Tuning-GA
   - 2.1 Problem und Scope
   - 2.2 Genome-Definition
   - 2.3 Suchraum + Pre-Compile-Constraints (Stage 1)
   - **2.3.1 Post-Compile VGPR-Gate (Stage 2)** *[1.0.1]*
   - 2.4 Fitness-Funktion (5-Phasen-Ablauf)
   - 2.5 GA-Parameter
   - 2.6 Priorisierte Shapes
   - 2.7 Gesamt-Zeitbudget
   - **2.7.1 Compile-Pipeline pro Generation** *[1.0.1]*
   - **2.7.2 CPU-Thread-Management** *[1.0.1]*
   - 2.8 Parity-Validation als GA-Hard-Gate
   - 2.9 Stability Validation
   - 2.10 Output: Pareto-Front + KernelVariantSet
3. Precision-Allocation-GA (NSGA-II)
4. Fusion-Pass-GA
5. rf-forge CLI
6. Profile-Cache
7. Integration mit Phase-1-Infrastruktur
8. Phase-2-Implementierungsplan
9. Anhang (rocprof-Baseline, v0.x-Vergleich, Hardware-Limits, Validation-Set, Arch-Doc-Sync)

*[1.0.1] = Amendment im v1.0.1-Changelog, siehe Header.*

---

## 1. Überblick

### 1.1 Die drei GAs und der Bandit

v1.0 trennt vier Optimierungs-Probleme sauber nach Werkzeug:

| Problem | Werkzeug | Zeitskala | Scope |
|---|---|---|---|
| Welche Tile-Config/LDS-Strategie ist auf gfx1201 für Shape X am schnellsten? | **Kernel-Tuning-GA** | Offline (~8 min pro Shape) | Pro Shape, pro Quant-Format |
| Welches Precision-Level pro Layer maximiert Speed × Qualität? | **Precision-Allocation-GA (NSGA-II)** | Offline (~15 min pro Modell) | Pro Modell-Instanz |
| Welche Fusions-Kombination ist auf dieser Architektur am besten? | **Fusion-Pass-GA** | Offline (~10 min pro Architektur) | Pro Architektur-Template |
| Welche der 3–5 GA-Winner-Varianten ist JETZT am schnellsten (angesichts GPU-Takt, VRAM-Druck)? | **UCB1 Bandit** (Säule 4) | Online (µs) | Pro Call |

Die drei GAs sind **Offline-Tuning**. Sie produzieren Top-3–5-Pareto-
Fronten, die der **Online-Bandit** zur Laufzeit orchestriert. GA ohne
Bandit wäre wertlos (keine Adaptivität an Laufzeit-Bedingungen); Bandit
ohne GA hätte keinen systematischen Kandidaten-Satz (nur die Hand-
geschriebenen v0.x-Varianten).

Diese Arbeitsteilung ist das Ergebnis aus v0.x-Erkenntnis #1
(`architecture_v1.2.0-draft.md` §10.1): "Profiling invertiert immer die
Schätzungen." Der GA hat keine Vorurteile; er misst.

### 1.2 rocprof-Baseline (1.17) als einzige empirische Referenz

Jede Performance-Aussage, jede Priorisierung und jede Fitness-Messung
in diesem Dokument referenziert die `phase1_step_1.17_rocprof_baseline.md`-
Daten. Die wichtigsten Zahlen:

**Kernel-Distribution (Tuned-Run, Bandit konvergiert):**

| Kernel | % GPU-Zeit | µs/Call | Calls (129 Iter) |
|---|---:|---:|---:|
| `gemv_q4_k_gate_up_swiglu` | **65.4 %** | 421.6 | 4 139 |
| `gemv_q4_k_q8_inline` (Q/K/V/O) | 16.0 % | 25.8 | 16 555 |
| `gemv_q6_k_standard` (LM-Head) | 13.7 % | 85.6 | 4 253 |
| `attention_decode` | 1.7 % | 10.7 | 4 139 |
| `rms_norm` + `rms_norm_batched` | 1.8 % | — | 16 670 |
| `rope` | 0.6 % | 1.9 | 8 278 |
| `residual_add_inplace` | 0.5 % | 1.7 | 8 278 |
| `kv_cache_append` | 0.3 % | 1.9 | 4 139 |

**Systemische Kennzahlen:**

| Metrik | Ist | Ziel (Arch-Doc §1.4) | Physik-Limit |
|---|---:|---:|---:|
| Decode (Qwen3-8B Q4_K_M) | 30 tok/s | 125 | 143 (BW) |
| Prefill (Qwen3-8B Q4_K_M) | 31 tok/s | 7 500 | ~15 000 |
| GPU-Effizienz (GPU-Zeit / Wallclock) | 62.7 % | >90 % | 100 % |
| BW-Effizienz (von 640 GB/s) | 21 % | ~85 % | 100 % |
| VRAM Peak | 10.05 GB | < 13 GB | 17.1 GB |
| `hipStreamSynchronize` pro 100 Token | ~65 000 | < 200 | 0 |

**Konsequenz für die GA-Spec.** Die rocprof-Daten beantworten drei
Fragen, die die Spec strukturieren:

1. **Wo ist der größte Hebel?** `gate_up_swiglu` (65 %) — das ist der
   primäre Kernel-GA-Fokus. Q8-Inline (16 %) und Q6_K LM-Head (14 %)
   sind die nächsten Prioritäten.
2. **Was limitiert uns?** NICHT Bandbreite (21 % BW-Nutzung!), NICHT
   Compute (Occupancy OK) — sondern **Dispatch-Sync** (83 260 Syncs,
   98 % HIP-API-Wall) und **fehlende Fusion** (v0.x fused Kernels
   reduzieren Dispatches pro Token drastisch).
3. **Was macht die GA wertlos ohne P0-Fix?** Solange jeder Kernel-
   Aufruf von `stream.synchronize()` eingerahmt ist, misst die GA
   keine echten Kernel-Zeiten, sondern Kernel-Zeit + Sync-Overhead.
   Die Fitness-Werte wären 37 % (= 1 - 62.7 %) zu langsam. Siehe
   §8.1 für das P0-Prerequisite-Gate.

### 1.3 Fitness-Baseline: relativ zu 1.17

Alle GA-Fitness-Funktionen messen **relativ zur rocprof-Baseline**:

```
fitness = baseline_time_us / candidate_time_us
```

Beispiele:

```
gate_up_swiglu Baseline:   421.6 µs
Kandidat mit 380 µs  → fitness = 421.6 / 380 = 1.11 (+11 %)
Kandidat mit 300 µs  → fitness = 421.6 / 300 = 1.40 (+40 %)
Kandidat mit 210 µs  → fitness = 421.6 / 210 = 2.01 (2× schneller)

q8_inline Baseline:   25.8 µs
Kandidat mit 22 µs   → fitness = 25.8 / 22 = 1.17 (+17 %)

q6_k_standard Baseline:   85.6 µs
Kandidat mit 60 µs   → fitness = 85.6 / 60 = 1.43 (+43 %)
```

**Warum relativ?** Absolute µs-Werte hängen von GPU-Takt, Wärme-State,
und anderer GPU-Last ab. Die relative Form ist robust gegen diese
Schwankungen: wenn die Baseline heute unter Hitze 440 µs statt 421 µs
läuft, ist der Kandidat mit 300 µs immer noch bei `fitness ≈ 1.47` —
der relative Vorteil bleibt stabil.

**Fitness-Aggregation.** Pro Kandidat werden **20 Messungen nach
5 Warmup-Runs** genommen; die Fitness ist `baseline_median /
candidate_median`. Median-statt-Mittelwert schützt gegen Ausreißer
(Context-Switch, GPU-Interrupt).

### 1.4 Kanonische Entscheidungen dieser Spec

Kurzform; Details in den jeweiligen Abschnitten.

1. **P0-Prerequisite-Gate.** Die GA darf nicht starten, wenn
   `hipStreamSynchronize`-Rate > 200 pro 100 Token (§8.1). Verhindert
   Fitness-Verzerrung durch Sync-Overhead.
2. **Proxy-Loss statt Perplexity.** KL-Divergenz auf 10–20 Prompts
   (~500 ms pro Evaluation) statt WikiText-Perplexity (Stunden). §3.3.
3. **Relative Fitness gegen 1.17.** Alle Zahlen referenzieren die
   rocprof-Baseline. §1.3.
4. **Mandatory Parity-Validation.** Jeder GA-Winner durchläuft VALU-
   Parity-Check (Säule 6) **bevor** er in den Cache kommt. §2.8.
5. **Monitor → GA Feedback-Loop.** DriftEvents im Produktions-Lauf
   modifizieren die nächste Precision-GA-Runde (hochgestufter Layer
   wird Constraint). §3.7.
6. **Cold-Start ohne Wartezeit.** Kein Cache → sofort Phase-1-
   Standard-Varianten, Mini-GA im Hintergrund (~2 min), volle GA
   optional/manuell. §6.5.
7. **Reproduzierbarkeit.** GA_SEED + GA_RUN_ID in jedem Log und
   Cache. §8 / §2.4.
8. **Ein GPU, sequenziell.** Kein Cluster-Parallelismus; ein Shape
   nach dem anderen. §2.7.

---

## 2. Kernel-Tuning-GA

### 2.1 Problem und Scope

Für eine gegebene `KernelShape (M, N, K)` und ein gegebenes Quant-Format
existieren tausende gültige Konfigurationen: Tile-Größen, Wave-Counts,
LDS-Nutzung, Prefetch-Tiefe, K-Unroll. Welche ist auf gfx1201 am
schnellsten? Das ist auf der Hardware messbar, nicht theoretisch
berechenbar (v0.x-Erkenntnis #1).

**Gültigkeit der Problemformulierung.** Der Bottleneck aus rocprof 1.17
ist **Kernel-Dauer + fehlende Fusion**, nicht Bandbreite. Für `gate_up_
swiglu` (65 %) ist die v0.x-Variante nur 6 % schneller (394 µs vs.
421 µs); der Kernel-GA allein bringt hier also **begrenzt** viel —
**FP8-Pfad + Fusion sind die Haupt-Hebel**. Trotzdem ist die Kernel-GA
Vorbedingung für:

- Generierung der FP8-Kandidaten (Level 0 aus `dequant_ir_spec.md` §4).
- Q8-Inline-Varianten für Q6_K (nur 1 Variante heute → Bandit hat keine
  Wahl, siehe rocprof 1.17 P1-Kandidat).
- Systematische Exploration von Tile-Configs (Direct-Global vs. LDS-
  Strategie — v0.x-Erkenntnis: Direct-Global kann 5× schneller sein).

### 2.2 Genome-Definition

Aus `architecture_v1.2.0-draft.md` §4.2, 1:1 übernommen:

```rust
pub struct KernelGenome {
    // Tile-Struktur
    pub tile_m: u8,             // 16, 32, 64, 128 (Vielfache von 16 für WMMA)
    pub tile_n: u8,             // dto.
    pub tile_k: u8,             // 16, 32, 64
    pub tiles_per_wave: u8,     // 1, 2, 4
    pub waves_per_block: u8,    // 1, 2, 4, 8

    // Speicher-Strategie
    pub use_lds_for_a: bool,
    pub use_lds_for_b: bool,
    pub prefetch_depth: u8,     // 0, 1, 2

    // K-Loop
    pub k_unroll: u8,           // 1, 2, 4, 8
    pub double_buffer: bool,

    // Quant-spezifisch
    pub dequant_strategy: DequantStrategy,
}

pub enum DequantStrategy {
    Inline,                             // Dequant pro Element im Kernel
    PrePass { lds_bytes: u16 },         // Separater Pre-Pass in LDS
    Batched { batch_size: u8 },         // Batched Dequant für mehrere Blöcke
}
```

**Kopplung an PrecisionLevel (aus `dequant_ir_spec.md` §4.4).** Der
Kernel-GA läuft pro `(Shape, QuantFormat, PrecisionLevel)`-Tupel. Für
Level 0 (FP8) sind manche Kombinationen aus dem KernelGenome profitabler
als für Level 1 (FP16), weil FP8 die LDS-Kapazität verdoppelt und den
VGPR-Druck halbiert. Die GA wird daher **pro PrecisionLevel separat**
ausgeführt (siehe §2.7 Zeitbudget).

**Kopplung an `TileConfig` aus `dequant_ir_spec.md` §5.1.** Der Kernel-
Emitter nimmt eine `TileConfig`, die ein Subset der `KernelGenome`-
Felder ist (`tile_m`, `tile_n`, `k_chunk`, `lds_strategy`, `num_waves`,
`unroll_factor`). Das `KernelGenome` ist die **Suchraum-Repräsentation**
der GA, `TileConfig` ist die **Emitter-Schnittstelle**. Konvertierung
ist trivial (Direkt-Mapping mit zusätzlichen KernelGenome-Feldern wie
`dequant_strategy`).

### 2.3 Suchraum und Constraints

**Brute-Force-Kombinatorik** (ohne Constraints):
`4 × 4 × 3 × 3 × 4 × 2 × 2 × 3 × 4 × 2 × 3 ≈ 83 000` Kandidaten pro
(Shape, Format, Level). Davon sind viele ungültig.

**Constraints (Fitness=0 bei Verletzung; verhindert ungültige Kandidaten).**

```rust
fn validate(g: &KernelGenome, fmt: &QuantFormat, level: PrecisionLevel) -> bool {
    // 1. WMMA-Alignment
    if g.tile_m % 16 != 0 || g.tile_n % 16 != 0 || g.tile_k % 16 != 0 {
        return false;
    }

    // 2. LDS-Budget (64 KB pro Workgroup auf gfx1201)
    let bytes_per_elem = match level {
        PrecisionLevel::Fp8   => 1,
        PrecisionLevel::Fp16  => 2,
        PrecisionLevel::Bf16  => 2,
        PrecisionLevel::Fp32  => 4,  // Level 3 nur Safety, kein GA
    };
    let lds = (g.tile_m as u32 * g.tile_k as u32 * bytes_per_elem
               * g.use_lds_for_a as u32)
            + (g.tile_n as u32 * g.tile_k as u32 * bytes_per_elem
               * g.use_lds_for_b as u32)
            + g.dequant_strategy.lds_overhead() as u32;
    let lds_with_double_buf = if g.double_buffer { lds * 2 } else { lds };
    if lds_with_double_buf > 64 * 1024 { return false; }

    // 3. VGPR-Budget Stage 1 (Pre-Compile, GROSSZÜGIG)
    //    Heuristische Schätzung — oft 30 % daneben (LLVM-Backend macht
    //    Spilling/Re-Materialization). Hier nur offensichtlichen Quatsch
    //    aussortieren; der echte Check kommt POST-Compile (§2.3.1).
    let vgpr_estimate = estimate_vgprs(g, fmt, level);
    if vgpr_estimate > 150 { return false; }   // großzügiger Puffer über dem 104-Ziel

    // 4. Waves × Tiles muss Workgroup-Size matchen
    if g.tiles_per_wave as u32 * g.waves_per_block as u32 > MAX_TILES_PER_BLOCK {
        return false;
    }

    // 5. Sub-Block-Alignment (Format-abhängig)
    //    Q4_K / Q8_0: sub_block_size = 32 → tile_k muss Vielfaches von 32 sein
    //    Q6_K:        sub_block_size = 16 → tile_k muss Vielfaches von 16 sein
    if g.tile_k as u32 % fmt.sub_block_size as u32 != 0 { return false; }

    true
}
```

**`MAX_TILES_PER_BLOCK`** wird aus gfx1201-Workgroup-Limits abgeleitet
(Standard 16); darüber hinaus verhindert der Compiler das Launch-Setup.

**VGPR-Schätzung** (`estimate_vgprs`): basierend auf den Dequant-IR-
Register-Budget-Tabellen (`dequant_ir_spec.md` §5.2). Für FP8:
Grundbedarf 40–56 VGPRs + `k_unroll`-Overhead. Für FP16: 48–64 + Overhead.
**Die Schätzung ist eine Heuristik und typischerweise 30 % daneben** —
LLVM entscheidet beim Register-Allocation-Pass über Spilling und
Re-Materialization auf eine Art, die analytisch nicht vorhersagbar ist.
Deshalb ist der Pre-Compile-Schwellenwert bewusst großzügig gesetzt
(150, nicht 104). Der echte VGPR-Check folgt **nach** der Kompilierung.

**Ungültige Kandidaten überleben keine Generation.** Der Pre-Compile-
Validator läuft **vor** der Kompilierung. Ungültige Individuen bekommen
Fitness=0 und werden von der Selection verworfen — kein Budget für
Compile + Messung wird verbrannt. Der Stage-1-Zweck ist,
offensichtlichen Quatsch (z. B. `tile_m=128 + k_unroll=8` → strukturell
> 200 VGPRs) ohne Compile-Aufruf herauszufiltern; ~30–40 % der
Zufalls-Kandidaten scheitern hier.

### 2.3.1 Stage 2: Post-Compile VGPR-Gate [SPEC-ERWEITERUNG]

Der Pre-Compile-Check aus §2.3 ist notwendig, aber nicht hinreichend.
Nach der Kompilierung werden die **echten** VGPR/SGPR-Zahlen aus dem
Code-Object gelesen und ein zweiter Gate-Check angewendet, **bevor**
die 20 Benchmark-Runs starten. Das spart ~100 ms pro abgelehntem
Kandidaten (20 Runs × 5 ms), was bei ~5–10 % Post-Compile-Rejects über
50 Generationen signifikant ist.

**Auslesen der echten Ressourcen-Zahlen.** Der emittierte `.co`-File
(Code Object) enthält die Ressourcen-Metadaten im ELF-Note-Segment:

```bash
# Drei Optionen, alle gleichwertig:
llvm-readobj --notes kernel.co | grep -E '\.vgpr_count|\.sgpr_count'
# oder: aus hipcc-Ausgabe bei Verbose-Flag
# oder: ELF-Notes direkt parsen (Rust-Crate `goblin`)
```

**Post-Compile-Validator:**

```rust
pub struct CodeObjectResources {
    pub vgpr_count: u16,
    pub sgpr_count: u16,
    pub lds_bytes: u32,
}

fn validate_post_compile(co: &CompiledKernel) -> bool {
    let res = read_resources(&co.binary);   // ELF-Notes-Parser

    // Echte Occupancy aus physischen VGPRs (gfx1201: 1536 VGPRs/CU, Wave32)
    let max_waves_per_cu = 1536 / res.vgpr_count.max(1) as u32;

    // Hard-Reject: weniger als 4 Waves/CU tötet Latency-Hiding
    if max_waves_per_cu < 4 {
        // entspricht > 384 VGPRs/Wave
        log_event("post_compile_vgpr_reject",
                  json!({ "actual_vgprs": res.vgpr_count,
                          "max_waves_per_cu": max_waves_per_cu }));
        return false;
    }

    // Telemetrie für alle durchgelassenen Kandidaten
    log_telemetry(co.key(), &json!({
        "actual_vgpr":            res.vgpr_count,
        "actual_sgpr":            res.sgpr_count,
        "actual_lds_bytes":       res.lds_bytes,
        "actual_waves_per_cu":    max_waves_per_cu,
        "heuristic_estimate":     co.key().heuristic_vgpr_estimate,
        "heuristic_error_pct":    estimate_error(&res, &co.key()),
    }));

    true
}
```

**Schwellenwert-Begründung (4 Waves/CU = 384 VGPRs).** Unterhalb von
4 Waves pro CU kann die GPU keine Memory-Latenz mehr durch Swap
zwischen Waves hiden — der Kernel wird Memory-Latency-bound, egal wie
gut die Kernel-Logik ist. 4 Waves ist der absolute Floor; Sweet-Spot
ist ~14 Waves/CU (`dequant_ir_spec.md` §5.2). Der Bereich 4–14 ist
akzeptabel (der Benchmark entscheidet), < 4 ist per Definition
Verschwendung.

**Graduelle Zone (4–8 Waves/CU, 192–384 VGPRs).** Diese Kandidaten
laufen durch den Benchmark, aber werden im Telemetrie-Log als
`limited_occupancy` markiert. Die GA darf sie in die Pareto-Front
aufnehmen, wenn sie trotz niedriger Occupancy schnell sind
(kontraintuitives Optimum — das ist genau der Use-Case der GA).

**Heuristik-Kalibrierung über Telemetrie.** Das Feld
`heuristic_error_pct` erlaubt, nach einem GA-Lauf statistisch zu
prüfen, ob `estimate_vgprs()` systematisch zu hoch oder zu niedrig
liegt. Bei > 40 % Durchschnittsabweichung über 1000 Kandidaten wird die
Heuristik neu kalibriert (Build-Zeit-Aufgabe, nicht GA-Laufzeit).

### 2.4 Fitness-Funktion

```rust
fn fitness(g: KernelGenome,
           shape: KernelShape,
           fmt: &QuantFormat,
           level: PrecisionLevel,
           baseline_us: f32) -> f32 {
    // Stage 1: Pre-Compile Validator (§2.3)
    if !validate_pre_compile(&g, fmt, level) { return 0.0; }

    // Kompilierung (bei Cache-Miss: ~95 ms; bei Hit: <1 ms)
    let kernel = codegen_gpu(&g.to_tile_config(), shape, fmt, level);

    // Stage 2: Post-Compile VGPR-Gate (§2.3.1)
    //   Liest echte VGPR-Zahl aus dem .co-File. Verwirft Kandidaten mit
    //   < 4 Waves/CU BEVOR die teuren 20 Benchmark-Runs laufen.
    if !validate_post_compile(&kernel) { return 0.0; }

    // Warmup (Caches, Clock ramp-up)
    for _ in 0..5 { kernel.run_dummy(); }

    // Messung: 20 Runs, Median
    let mut times = Vec::with_capacity(20);
    for _ in 0..20 {
        let t0 = gpu_timestamp_us();
        kernel.run(real_input);
        let t1 = gpu_timestamp_us();
        times.push(t1 - t0);
    }
    let median_us = median(&times);

    // Relative Fitness zur rocprof-Baseline
    baseline_us / median_us
}
```

**Ablauf pro Kandidat (Zusammenfassung):**

```
1. validate_pre_compile()     → Fail: Fitness=0, kein Compile   (~0 ms)
2. codegen_gpu() + hipcc       → Compile (~95 ms Miss / <1 ms Hit)
3. validate_post_compile()     → Fail: Fitness=0, kein Benchmark (~1 ms)
4. Warmup (5 Runs)             → (~25 ms)
5. Benchmark (20 Runs, Median) → Fitness = baseline / median      (~100 ms)
```

Das Two-Stage-Gate spart bei Post-Compile-Rejects ~125 ms pro Kandidat
(Schritte 4+5). Bei geschätzt 5–10 % Post-Compile-Rejects über 5000
Evaluationen ergibt das 30–60 s pro Shape Ersparnis — der Budget-Puffer
aus §2.7 bleibt intakt.

**Warum 20 Runs + Median, nicht mehr?** Welford-Varianz-Analyse auf
gfx1201 zeigt: nach 15+ Runs stabilisiert sich der Median; 20 gibt
ausreichend Puffer. Höhere Zahlen erhöhen nur die Tuning-Zeit.

**Fitness-Messung braucht saubere GPU-Zeiten.** Das ist der Grund für
das P0-Prerequisite-Gate (§8.1). Mit aktivem Bandit-Sync wäre
`candidate_time_us` um ~45 µs pro Sync verzerrt — im Durchschnitt
+45 µs auf jede Messung, was die relative Reihenfolge der Kandidaten
gleich lässt, aber Cross-Shape-Vergleiche (und damit Priority-
Entscheidungen des Precision-GAs) verfälscht.

### 2.5 GA-Parameter

Aus `architecture_v1.2.0-draft.md` §4.2, 1:1 übernommen:

```
Population:    100 Individuen
Generationen:  50            → 5 000 Evaluierungen pro (Shape, Level)
Selection:     Tournament-Size 3
Crossover:     Uniform, Rate 0.7
Mutation:      Gauss-Perturbation einzelner Gene, Rate 0.1 pro Gen
Elitism:       Top 5 % unverändert in nächste Generation
```

**Seed und Reproduzierbarkeit.** `GA_SEED` ist ein CLI-Parameter
(`--seed <N>`, Default: kryptographisch zufällig zum Lauf-Start,
geloggt). Gleicher Seed + gleiche Hardware + gleicher Engine-Git-Hash →
**bit-identische** Population-Evolution. Ohne Seed: Läufe sind nicht
vergleichbar (ein Fehlerfall für CI/Bug-Regression-Suche).

**`GA_RUN_ID`** ist ein UUID, der pro Lauf erzeugt wird und in jeder
Log-Zeile und im Cache-Eintrag landet. Damit sind "dieser Kernel kommt
aus Lauf X" nachvollziehbar, auch wenn zwei Läufe denselben Seed hatten
(unterschiedliche Hardware-Bedingungen).

### 2.6 Priorisierte Shapes (basierend auf rocprof 1.17!)

**Das ist der Kernpunkt der Kernel-GA-Priorisierung.** Nicht alle Shapes
sind gleich wichtig. Die rocprof-Baseline gibt die Rangfolge:

| Priorität | Shape | Kernel-Typ | Baseline-Zeit | % GPU | Fitness-Fokus |
|---|---|---|---:|---:|---|
| **P1** | `gemv_q4_k_gate_up_swiglu` 4096×12288×4096 | Fused Gate+Up+SwiGLU | 421.6 µs | **65.4 %** | FP8-Pfad aktivieren, Fusion (§4) |
| **P2** | `gemv_q4_k_q8_inline` 4096×4096×4096 | Standard GEMV mit Q8-Inline | 25.8 µs | 16.0 % | Tile-Config-Exploration |
| **P3** | `gemv_q6_k_standard` 4096×151936×4096 | LM-Head-GEMV | 85.6 µs | 13.7 % | **Q8-Inline-Variante + WMMA** |
| **P4** | alle übrigen Shapes | diverse | < 1 % | < 5 % | niedrigste Prio |

**Zeitbudget pro Shape.** ~8 Minuten (aus Arch-Doc §4.2: 5 000
Evaluierungen × 100 ms). Pro PrecisionLevel separat:

| Shape × Level | Shapes | Levels (GA-aktiv) | Gesamt |
|---|---:|---:|---:|
| Top-3 Prio × FP8 (Level 0) | 3 | 1 | 24 min |
| Top-3 Prio × FP16 (Level 1) | 3 | 1 | 24 min |
| Sekundär ~17 Shapes × FP8 | 17 | 1 | ~135 min |
| Optional BF16/FP32 | variabel | 0 (keine GA) | 0 |
| **Summe** | | | **~3 h** |

**Level 2 (BF16)** wird von der Kernel-GA **nicht** exploriert (vgl.
`dequant_ir_spec.md` §4: BF16 ist GA-Option für ausgewählte Layer via
Precision-GA, nicht Standardpfad). Level 3 (FP32 VALU) ist der
Safety-Pfad (Säule 6), kein GA-Kandidat.

**Sekundär-Shapes werden "bedarfsgetrieben" getuned.** Nur Shapes, die
im aktuellen Modell tatsächlich vorkommen, laufen durch den GA
(`dequant_ir_spec.md` §4.7, "Cache bedarfsgetrieben"). Für ein 8B-
Modell typisch 5–7 Haupt-Shapes; ein 14B-Modell bringt 2–3 weitere.

**Budget-Early-Exit.** Der GA kann pro Shape mit `--budget <TIME>`
früher gestoppt werden (Standard: 10 min Hard-Cap). Wenn sich der
Pareto-Front-Hypervolume über **10 Generationen** **nicht um > 1 %**
verbessert, bricht die GA ebenfalls ab ("Konvergenz-Early-Exit").

**Begründung für 10 statt 5 Generationen** [SPEC-ERWEITERUNG]: GAs
zeigen häufig **Punctuated-Equilibria** — längere Plateaus gefolgt von
plötzlichen Durchbrüchen durch eine Kaskade erfolgreicher Mutationen.
5 Generationen (10 % des Gesamtbudgets) waren als Schwelle zu
aggressiv: ein echter Plateau-Durchbruch wäre vorzeitig abgewürgt
worden. 10 Generationen (20 % des Budgets, Worst-Case +~100 s pro
Shape) geben der GA genug Zeit, Plateaus zu überwinden. Bei ~8 min
Gesamt-Budget pro Shape ist das eine akzeptable Ausweitung.

### 2.7 Gesamt-Zeitbudget: ein GPU, sequenziell

| Teil | Dauer | Bemerkung |
|---|---|---|
| Top-3 Shapes × FP8 + FP16 | ~48 min | hoher Fokus, volle 5 000 Eval |
| Sekundär ~17 Shapes × FP8 | ~135 min | reduzierter Budget (5 min/Shape) |
| Compile-Cache-Warmup | ~15 min | erster Kernel pro Variant: 95 ms Compile |
| Parity-Validation jedes Kandidaten (§2.8) | +~10 % | in Fitness integriert |
| **Gesamt Kernel-GA** | **~3 h** | einmalig pro Modell, gecached |

**Kein Cluster-Parallelismus.** ROCmForge läuft auf einer einzelnen
RX 9070 XT. Parallele Messungen auf derselben GPU sind **nicht** sinnvoll
(Contention verfälscht die Fitness). `--parallel-shapes <N>` ist für
Multi-GPU-Systeme vorgesehen (nicht v1.0-Scope), in v1.0 bleibt der
Default `1`.

### 2.7.1 Compile-Pipeline innerhalb einer Generation [SPEC-ERWEITERUNG]

Die naïve Reihenfolge *kompiliere → benchmark → kompiliere → benchmark*
würde das Zeitbudget aus §2.7 sprengen: `hipcc` hat einen
Compiler-Launch-Overhead von 50–100 ms allein für den Process-Startup.
Bei 30–40 % Cache-Miss-Rate in frühen Generationen summiert sich das
auf ~50 % Budget-Überschreitung.

**Lösung: innerhalb einer Generation werden die Phasen getrennt.**

```
Phase      | Parallelität | Was                               | Dauer (100 Indiv.)
───────────|──────────────|───────────────────────────────────|───────────────────
Validate 1 | parallel CPU | Pre-Compile Heuristik (§2.3)      | ~1 ms (trivial)
Compile    | parallel CPU | hipcc für ~60 gültige Kandidaten  | ~2 s (8 Threads)
Validate 2 | parallel CPU | Post-Compile VGPR-Gate (§2.3.1)   | ~60 ms
Benchmark  | SEQUENTIELL  | GPU: 5 Warmup + 20 Runs           | ~40 × 125 ms = 5 s
───────────|──────────────|───────────────────────────────────|───────────────────
Summe                                                           ~7 s / Generation
```

**Warum Compile parallel, Benchmark sequentiell?**

- **Compile ist CPU-bound** (`hipcc` → `clang` → LLVM). Bei 8
  CPU-Threads und 100 ms/Compile ergibt sich ein Batch-Durchsatz von
  ~80 Compiles/s. 60 Cache-Miss-Kandidaten sind in ~0.8 s abgearbeitet.
- **Benchmark ist GPU-bound**. Parallele Kernel-Messungen auf derselben
  GPU kollidieren (Contention um L2, Memory-Controller, Power-Budget) —
  die Fitness-Werte werden unreliabel. Der sequentielle Benchmark ist
  eine Korrektheits-Bedingung, keine Optimierungs-Option.
- **VGPR-Gate (Stage 2) zwischen Compile und Benchmark** ist der
  Scharnier-Punkt: es filtert ~5–10 % der Kandidaten nach Compile aber
  **vor** den teuren Benchmark-Runs heraus (§2.3.1).

**Compile-Cache aggressiv nutzen.**

```rust
/// Pro (TileConfig, Format, Level) nur einmal kompilieren.
pub struct CompileCache {
    entries: HashMap<CompileKey, Arc<CompiledKernel>>,
}

pub struct CompileKey {
    tile_config:     TileConfig,
    format_id:       QuantFormatId,
    precision_level: PrecisionLevel,
    engine_git_hash: [u8; 20],
}
```

- **Cache-Key:** SHA-256 über `CompileKey` (deterministisch, portabel).
- **Erwartete Cache-Hit-Rate:** nach Generation 10+ typisch 60–70 %
  (Crossover und Elitism reproduzieren oft bekannte Configs; komplett
  neue Genome werden durch Mutation seltener).
- **Cold-Start (Generationen 0–5):** Hit-Rate steigt von ~0 % auf ~50 %.
  Das erklärt den ~15-min-Warmup-Anteil in §2.7.

**Gesamtzeit-Kalkulation mit Pipeline:**

```
50 Generationen × 7 s/Generation ≈ 350 s ≈ 5.8 min pro (Shape, Level)
```

Das liegt unter dem 8-min-Budget aus §2.7. Der Puffer (~2 min) deckt
den Cache-Cold-Start in den ersten Generationen ab, wo die Hit-Rate
noch niedrig ist.

**Neue Telemetrie-Felder pro Generation** (Ergänzung §5.10):

```json
{"event":"generation_complete", "generation":12,
 "compile_cache_hits": 64, "compile_cache_misses": 36,
 "compile_cache_hit_rate": 0.64,
 "post_compile_vgpr_rejects": 8,
 "benchmarked_individuals": 52,
 "compile_wall_ms": 1823, "benchmark_wall_ms": 4920,
 "best_fitness": 1.35, "median_fitness": 0.94}
```

Diese Felder erlauben Post-hoc-Analyse der Pipeline-Gesundheit:
- Cache-Hit-Rate < 30 % in späten Generationen → Mutation-Rate zu hoch?
- VGPR-Rejects > 20 % → Pre-Compile-Schwellenwert zu großzügig?
- `compile_wall_ms > benchmark_wall_ms` → CPU-Thread-Count zu niedrig?

### 2.7.2 Compile-Parallelität auf CPU-limitierten Systemen

Die 8-Thread-Annahme in §2.7.1 entspricht dem Dev-Target (Ryzen 9 7945HX,
16C/32T — wir nutzen 8 Threads konservativ, um die restlichen Cores für
Bandit, Telemetrie und Benchmark-Dispatch frei zu lassen). Auf CI-
Systemen oder kleineren Workstations gilt:

```
CPU-Threads   | Compile-Phase (60 Kandidaten) | Gesamt pro Generation
──────────────|────────────────────────────────|───────────────────────
4             | ~1.5 s                         | ~6.5 s
8 (Default)   | ~0.8 s                         | ~7.0 s  (benchmark-bound)
16            | ~0.5 s                         | ~7.0 s  (benchmark-bound)
```

Ab ~8 Threads ist die Generation benchmark-bound, weitere Threads
bringen nichts. `rf-forge tune-kernels --compile-threads <N>` erlaubt
Override (Default `min(num_cpus/2, 8)`).

### 2.8 Parity-Validation als GA-Hard-Gate

Gemäß `dequant_ir_spec.md` §8.6 durchläuft **jede** GA-produzierte
Kernel-Variante einen VALU-Parity-Check gegen den Level-3-Referenz-
Kernel (Säule 6), **bevor** sie in die Pareto-Front aufgenommen wird.
Sampling-Rate für GA-Validation: **1/1** (jeder Kandidat wird validiert,
nicht nur 1/1000 wie in Produktion).

**Ablauf pro Kandidat:**

```
1. Kompilieren (GPU-Codegen + hipcc)
2. Run mit Test-Input (10 deterministische Blöcke)
3. Parallel: Level-3-VALU-Kernel mit demselben Input
4. Parity-Check:
   - PrecisionLevel::Fp8  → Toleranz max_abs_err < 2⁻⁷  ≈ 0.0078
   - PrecisionLevel::Fp16 → Toleranz max_abs_err < 2⁻¹⁰ ≈ 0.001
   - PrecisionLevel::Bf16 → Toleranz max_abs_err < 2⁻⁷  ≈ 0.0078
5. Bei Violation: Kandidat bekommt Fitness=0 (rejection, nicht -inf,
   damit er aus dem Log als "brittle" erkennbar bleibt)
6. Bei Bestanden: Fitness = baseline / candidate_median
```

**Warum wichtig?** Ein GA kann einen Kandidaten finden, der zwar
**schneller** ist, aber **numerisch falsch** — z. B. durch einen
Off-by-One im Dequant-Programm, der zufällig in einem Warmup-Set
plausibel aussieht, aber in Produktion Drift verursacht. Die Parity-
Validation ist die Schutz-Schicht dagegen.

### 2.9 Stability Validation der Top-Winner

Nach der GA-Konvergenz werden die **Top-5 Kandidaten** einem strengeren
Stability-Check unterzogen, bevor sie in den Produktions-Cache kommen:

```
Stability Check pro Top-Kandidat:
  1. 3 verschiedene Test-Input-Sets (short/medium/long Context)
  2. 10 Runs pro Set
  3. Median-Varianz über alle 30 Runs: muss < 2 % sein
  4. Parity-Test mit 1000 zufälligen Blöcken (nicht nur 10)
  5. Kein Regression auf bekannte Referenz-Outputs
  6. Telemetrie-Schema geloggt (§8.1.H)

Falls Stability-Check fehlschlägt:
  - Kandidat wird als "brittle" markiert
  - Fällt aus der Pareto-Front
  - Nächst-schlechterer Kandidat rückt nach
  - Log-Event mit Begründung (Varianz > 2 %, Parity-Fail, etc.)
```

**Top-5-Report-Format.** Für jeden Top-5-Kandidaten werden die **echten**
Post-Compile-Ressourcen aus §2.3.1 tabellarisch ausgegeben, nicht nur
die Genome-Parameter. Das macht Differenzen sichtbar, die der Heuristik
entgehen:

```
Shape: gemv_q4_k_gate_up_swiglu_fp8, Baseline: 421.6 µs

Rank | tile_m×n×k | lds | actual_vgpr | actual_lds | waves/CU | median µs | fitness
─────|────────────|─────|─────────────|────────────|──────────|───────────|────────
  1  | 64×128×32  | LdsB|     96      | 12 288 B   |    16    |   278.3   | 1.515
  2  | 64×128×32  | LdsAB|    88      | 20 480 B   |    17    |   291.7   | 1.445
  3  | 32×128×64  | LdsB|    112      | 16 384 B   |    13    |   298.1   | 1.414
  4  | 64×64×32   |Direc|     76      |     0 B    |    20    |   305.4   | 1.380
  5  | 64×128×32  |Direc|     92      |     0 B    |    16    |   312.0   | 1.351
```

Der Report landet in `~/.rocmforge/logs/<run-id>_topk.md` und wird vom
CI-Dashboard automatisch gerendert.

**Warum 2 %?** Die rocprof-1.17-Messungen zeigen Median-Varianz von
0.8–1.5 % zwischen Runs derselben Hardware. 2 % gibt 30–50 % Puffer;
Kandidaten mit höherer Varianz sind eher "glücklich schnell" als
"robust schnell". Brittle Kandidaten verwirren den Bandit
(UCB1-Exploration oszilliert bei hoher Varianz).

### 2.10 Output: Pareto-Front + KernelVariantSet

Der Kernel-GA produziert pro `(Shape, Format, Level)` einen
`KernelVariantSet` (aus `dequant_ir_spec.md` §8.4):

```rust
// aus dequant_ir_spec.md §8.4 (unverändert)
pub struct KernelVariantSet {
    pub base_key: KernelCacheKey,
    pub variants: Vec<KernelVariant>,    // 3–5 Top-Varianten
}

pub struct KernelVariant {
    pub binary: Arc<CompiledKernel>,
    pub tile_config: TileConfig,
    pub gemv_variant: Option<GemvVariant>,
    pub gmeasured_tps: f32,              // GA-Predicted Throughput
}
```

**Warum 3–5 Varianten, nicht nur den Sieger?** Aus Arch-Doc §4.2:
"Der GA findet im Offline-Mittel den besten; zur Runtime kann eine
andere Variante besser sein" (andere GPU-Auslastung, anderer Takt).
Der Bandit (Säule 4) erkundet die Top-Varianten und konvergiert zur
**aktuell besten**. Nach 100 Calls typ. stabil (`architecture_v1.2.0-
draft.md` §2.5).

**Pareto-Filter.** Nicht die Top-5 nach Fitness, sondern die Top-5 nach
**Pareto-Dominanz** über (Fitness, VGPR-Druck, Varianz). Das verhindert,
dass alle 5 Varianten "ähnlich teuer und schnell" sind — wir wollen
Diversität für den Bandit.

---

## 3. Precision-Allocation-GA (NSGA-II)

### 3.1 Problem

Gegeben ein Modell mit `n` Layern und vier möglichen Precision-Stufen
pro Layer (FP8-E4M3, FP8-E5M2, FP16, BF16), gibt es `4^n` Kombinationen.
Bei n=32 sind das `≈ 1.8 × 10¹⁹`. Plus drei modell-globale Dimensionen
(Embedding-Lookup, LM-Head, KV-Cache-Precision).

Jede Kombination ist ein **Trade-off**: höhere Precision → bessere
Qualität (niedrigere KL-Divergenz zu FP32), aber niedrigere Speed +
höherer Memory-Footprint. Das ist ein Multi-Objective-Problem —
klassischer NSGA-II-Case.

**Level 3 (FP32 VALU) ist nicht Teil des GA-Suchraums** (vgl.
`dequant_ir_spec.md` §4.4): er ist Safety/Debug-Pfad und wird nur durch
den Quality Monitor (Säule 5) aktiviert, nicht vom GA vorgeschlagen.

### 3.2 Suchraum und Genome-Definition

Aus `architecture_v1.2.0-draft.md` §4.3, 1:1 übernommen:

```rust
pub struct PrecisionGenome {
    /// Pro Layer: FP8-E4M3 (Default), FP8-E5M2, FP16, BF16.
    /// FP32 ist kein GA-Kandidat — nur Safety-Monitor-Fallback.
    pub per_layer: Vec<PrecisionHint>,

    /// Embedding-Lookup: meist FP8-E4M3, bei kritischen Tokens hochstufen.
    pub embedding: PrecisionHint,

    /// LM-Head (Logits): häufig FP16 oder BF16 wegen Dynamikbereich.
    pub lm_head: PrecisionHint,

    /// KV-Cache-Precision: Default FP8-E5M2, bei langem Kontext kritisch.
    pub kv_cache: PrecisionHint,

    /// Attention-Akkumulator: fast immer FP32, kein GA-Suchraum für v1.0.
    pub attn_acc_fp32: bool,
}

pub enum PrecisionHint {
    Fp8E4M3,      // Level 0, kleiner Dynamikbereich — Default für Gewichte
    Fp8E5M2,      // Level 0, größerer Dynamikbereich — KV-Cache-Standard
    Fp16,         // Level 1, 1 Instr/Elem Downcast
    Bf16,         // Level 2, 5 Instr/Elem Downcast (nur für Spezialfälle)
}
```

**Warum FP8-E4M3 Default?** Aus `dequant_ir_spec.md` §1.3: FP8-E4M3-WMMA
ist 2× schneller als FP16-WMMA auf gfx1201, halbiert den VGPR-Bedarf
für A/B-Fragmente, und halbiert die LDS-Bandbreite pro Tensor. Für die
meisten Layer (nach Introspection) ist die Numerik ausreichend.

**KV-Cache als eigene Dimension.** Arch-Doc §4.3 hebt hervor: der
KV-Cache ist bei langem Kontext der größte Memory-Block und hat direkten
Einfluss auf Max-Kontext-Länge. FP8-E5M2 (größerer Exponent-Bereich) ist
robust gegen Outlier-Werte in Keys/Values.

**Attention-Akkumulator.** In v1.0 ist `attn_acc_fp32 = true` fix (kein
GA-Freiheitsgrad). Die Softmax-Berechnung braucht FP32-Präzision für
numerische Stabilität bei langen Kontexten; der Aufwand ist vernachlässig-
bar gegenüber dem GEMV-Pfad.

**Mapping zu PrecisionLevel (aus `dequant_ir_spec.md`):**

| PrecisionHint | PrecisionLevel | WMMA-Intrinsic |
|---|---|---|
| `Fp8E4M3` / `Fp8E5M2` | Level 0 (Fp8) | `v_wmma_f32_16x16x16_fp8_fp8_w32` |
| `Fp16` | Level 1 (Fp16) | `v_wmma_f32_16x16x16_f16_w32` |
| `Bf16` | Level 2 (Bf16) | `v_wmma_f32_16x16x16_bf16_w32` |

Der Codegen pickt zur Kompilier-Zeit die passende Kernel-Variante.

### 3.3 Objective 1: Qualität (KL-Divergenz als Proxy-Loss)

**KRITISCH — warum Proxy statt Perplexity.** Ein voller Perplexity-Run
(WikiText-2, ~300 k Tokens) dauert auf 30 tok/s ungefähr **2.8 Stunden**.
Für GA-Iterationen (1800 Evaluationen pro Modell) wären das
~5000 Stunden. Unmöglich.

**Proxy-Loss: KL-Divergenz auf fixem Validierungs-Set.**

```rust
fn fitness_quality(g: &PrecisionGenome,
                   ref_logits: &CachedLogits,
                   prompts: &ValidationSet) -> f32 {
    let engine = build_engine_with_precision(g);

    let mut kl_values = Vec::with_capacity(prompts.len());
    for prompt in prompts.iter() {
        let candidate_logits = engine.forward(prompt);
        let kl = kl_divergence(&candidate_logits, &ref_logits[prompt.id]);
        kl_values.push(kl);
    }

    // Aggregate: P95 statt Mean (robust gegen einzelne Outlier-Prompts)
    let p95 = percentile(&kl_values, 95.0);
    -p95   // NSGA-II maximiert → negative Loss
}
```

**Validierungs-Set (siehe Anhang §9.4 für exakte Liste).**

| Set | Prompts | Tokens (Σ) | Dauer pro Eval |
|---|---:|---:|---:|
| `short` | 5 | ~40 | ~100 ms |
| `medium` | 10 | ~320 | ~500 ms |
| `long` | 5 | ~1 280 | ~2 000 ms |
| **Gesamt** | **20** | **~1 640** | **~2.6 s** |

Für die regulären GA-Iterationen wird **nur `medium`** verwendet (~500 ms
pro Eval, 1 800 × 500 ms = 15 min pro Modell).

Für die **finale Bewertung der Pareto-Front-Punkte** werden alle drei
Sets durchlaufen (~2.6 s × 10 Punkte = ~30 s zusätzlich).

**FP32-Referenz.** Wird **einmal beim Modell-Load** berechnet und in
`~/.rocmforge/profiles/<model-hash>/fp32_reference_logits.bin`
gespeichert. Dauer: ~1–2 min (32 Layer × Forward-Pass in FP32 VALU-Pfad
auf 1 640 Tokens).

**Reject-Threshold.** Kandidaten mit `P95 KL > 0.1` werden aus der
Pareto-Front ausgeschlossen. Das ist nicht Fitness=0 (die Genome-
Information ist immer noch wertvoll für GA-Lernen), aber als "nicht
Produktions-tauglich" markiert.

**Multilingual-Split.** Bei multilingualen Modellen (Llama-3.1, Qwen3)
wird das Validierungs-Set um 5 nicht-englische Prompts erweitert (z. B.
Chinesisch für Qwen, Deutsch/Französisch für Llama). Die P95-KL wird
separat pro Sprache berechnet; der Kandidat muss in **allen** Sprachen
unter Reject-Threshold bleiben.

**Token-wise Max Error.** Zusätzlich zu P95-KL wird `max(KL pro Token)`
geloggt. Kandidaten mit `max_kl > 1.0` werden **auch dann verworfen**,
wenn P95 ok ist — einzelne katastrophal falsche Tokens sind schlimmer
als gleichmäßig leicht schlechtere.

### 3.4 Objective 2: Speed + Memory (kombiniert)

Aus `architecture_v1.2.0-draft.md` §4.3, erweitert um relative Fitness:

```rust
fn fitness_speed_and_memory(g: &PrecisionGenome,
                            baseline_decode_tps: f32) -> f32 {
    let engine = build_engine_with_precision(g);

    // Speed-Komponente (256 Tokens Decode, median of 3 runs)
    let decode_tps = measure_decode_tps(&engine, 256, 3);
    let speed_norm = decode_tps / baseline_decode_tps;   // relativ zur 1.17-Baseline

    // Memory-Komponente: KV-Cache-Footprint bei 8k-Kontext
    let kv_mem_bytes = kv_cache_size_bytes(g.kv_cache, ctx_len=8192, n_layers=32);
    let vram_budget = 16.0 * 1e9;
    let mem_efficiency = (vram_budget - kv_mem_bytes as f32) / vram_budget;

    // Gewichtete Kombination (Hyperparameter aus Arch-Doc §4.3)
    0.7 * speed_norm + 0.3 * mem_efficiency
}
```

**Warum kombiniert?** Reine Speed-Optimierung führt bei langem Kontext
zu OOM — ein Kandidat mit FP16-KV bei 32k Kontext sprengt 16 GB VRAM,
auch wenn er schneller ist. `mem_efficiency` korrigiert das: wer mehr
VRAM braucht, wird penalisiert.

**Speed-Messung: 256 Tokens, Median of 3 Runs.** Nicht mehr Runs, weil
jeder Run ~8 s dauert (bei 30 tok/s); 3 × 3 = 9 s pro Kandidat ist
akzeptabel. Für die 1800 Evaluationen entspricht das ~4.5 h allein für
Speed — zu viel.

**Kürzeres Speed-Proxy für GA-Iterationen.** Während der GA-Läufe wird
**nur 1 Run auf 128 Tokens** gemessen (~4 s). Für die finale Pareto-Front-
Bewertung kommt der vollere 256-Token × 3-Run-Test.

**Baseline-Referenz.** `baseline_decode_tps = 30 tok/s` aus rocprof
1.17 Tuned-Run. Gleiche Logik wie Kernel-GA: relative Form ist stabil
gegen Hardware-Takt-Schwankungen.

### 3.5 Seed-Population aus ModelProfile

Die **Phase-1-Introspection** (Schritt 1.13, Säule 1) liefert pro Modell
ein `ModelProfile` mit `precision_recommendation: Vec<PrecisionHint>`.

**Nutzung als Seed:**

```rust
fn build_seed_population(profile: &ModelProfile,
                         pop_size: usize,
                         rng: &mut Rng) -> Vec<PrecisionGenome> {
    let recommendation = &profile.precision_recommendation;

    let mut pop = Vec::with_capacity(pop_size);

    // Individuum 1: exakt die Introspection-Empfehlung
    pop.push(PrecisionGenome::from_recommendation(recommendation));

    // Individuen 2..pop_size: Variationen
    for i in 1..pop_size {
        let mut genome = pop[0].clone();
        // Mutiere 1–3 Layer zufällig (schwache Perturbation)
        let n_mutations = rng.gen_range(1..=3);
        for _ in 0..n_mutations {
            let layer_idx = rng.gen_range(0..genome.per_layer.len());
            genome.per_layer[layer_idx] = random_precision(rng);
        }
        pop.push(genome);
    }
    pop
}
```

**Warum kein Zufalls-Seed?** Phase-1-Validation zeigt, dass **alle drei**
getesteten Modelle (Qwen3-8B, Llama-3.1-8B, Qwen2.5-Q4_0) vom
Introspection-Pass als **CRITICAL SNR** eingestuft wurden — die
Empfehlung startet also bei BF16 oder FP16 für kritische Layer, nicht
bei FP8. Ein Zufalls-Seed würde diese Information wegwerfen und Tausende
Evaluationen auf "offensichtlich schlechte" FP8-only-Kandidaten
verschwenden.

Die GA erkundet dann von der Empfehlung ausgehend: **senkt sie Precision
dort, wo es funktioniert**, und **hält sie hoch, wo die Qualität sonst
kollabiert**.

### 3.6 Pareto-Front und Standard-Auswahl

NSGA-II produziert am Ende der Evolution **nicht einen Gewinner**,
sondern eine Pareto-Front: die Menge der Lösungen, für die keine andere
Lösung in beiden Objectives gleichzeitig besser ist.

**Typisch 5–10 Punkte auf der Front:**

```
               Speed (relativ)
         2.0 ┤ C
             │  ▲
         1.5 ┤   ■ D
             │     ▲
         1.0 ┤      ● E (baseline)
             │         ▲
         0.5 ┤          ▼ F
             └──────────────────▶
              -0.30  -0.10   0.00   Quality (-KL, größer = besser)
```

**Standard-Auswahl: 95 % FP32-Qualität bei maximaler Speed.** Nach
`architecture_v1.2.0-draft.md` §1.4 ist das das "realistische
Performance"-Ziel:

```rust
fn select_default_pareto_point(front: &ParetoFront) -> ParetoIndex {
    let fp32_ref_kl = 0.0;  // FP32 gegen sich selbst
    let target_quality = 0.95;  // "95% der FP32-Qualität"

    // Finde den Punkt mit der höchsten Speed, deren KL-Qualität
    // mindestens 95% der FP32-Referenz entspricht.
    front
        .iter()
        .filter(|p| quality_ratio(p.kl_p95, fp32_ref_kl) >= target_quality)
        .max_by_key(|p| OrderedFloat(p.speed_norm))
        .map(|p| p.index)
        .unwrap_or_else(|| front.fastest())
}
```

**Manuelle Auswahl.** User kann per `rf-forge tune-precision
--target-point <i>` oder `--min-quality <q>` einen anderen Punkt
wählen. Die Auswahl landet in `precision_pareto.json` (siehe §6.1).

### 3.7 Runtime-Feedback: Monitor → Precision-GA

Dies ist einer der neuen Schlüssel-Mechanismen von v1.0: die Pareto-
Front **lernt aus Produktions-Drift**.

**Ablauf:**

```
1. Produktion: rocmforge chat ... läuft mit Pareto-Punkt P
2. Quality Monitor (Säule 5) sampelt Hidden-States
3. Bei einem Layer X wird Drift detektiert:
   DriftEvent {
       layer_idx: X,
       reason: Fp8SaturationExceeded { fraction: 0.15 },
       current_level: Fp8,
       recommended_level: Fp16,
   }
4. Runtime reagiert sofort:
   - Layer X wird in der aktuellen Konfiguration hochgestuft
     (FP8 → FP16, oder FP16 → Level 3 VALU bei wiederholtem Event)
   - Die neue Konfiguration ersetzt P im Cache
     (~/.rocmforge/profiles/<model>/precision_pareto.json)
5. Bei der nächsten regulären Precision-GA-Runde:
   - Layer X wird als Constraint-Gen fixiert auf ≥ Fp16
   - Die GA darf für Layer X nicht zurückstufen auf FP8
   - Die Pareto-Front wird um diesen Layer neu berechnet
6. → Langfristig konvergiert die Pareto-Front zur stabilsten
      Konfiguration für die reale Nutzung
```

**Warum das funktioniert.** Drift-Events sind selten (typisch < 1 pro
10 k Tokens), aber informativ: sie zeigen genau die Layer, bei denen
die statische Introspection das Risiko unterschätzt hat. Jedes Event
eliminiert eine Region des Suchraums, die wir nie wieder betreten wollen.

**Constraint-Repräsentation.** In der nächsten GA-Runde wird der
`PrecisionGenome` um ein Feld erweitert:

```rust
// Nur zur GA-Laufzeit, nicht im persistenten Cache
pub struct ConstrainedGenome {
    pub genome: PrecisionGenome,
    /// Pro Layer: minimales Precision-Level (vom Monitor gelernt).
    pub min_level: Vec<PrecisionHint>,
}

// Constraint-Validation: jedes Individuum muss `per_layer[i] >= min_level[i]`
// erfüllen. Bei Verletzung wird das Individuum zum min_level korrigiert
// (kein Fitness=0, sondern Repair — schneller Konvergenz).
```

**Ordering auf PrecisionHint** (für `>=`):
`Fp8E4M3 < Fp8E5M2 < Fp16 < Bf16`. E5M2 liegt über E4M3, weil größerer
Exponent-Bereich mehr Dynamik erlaubt (auch wenn gleiche Bit-Breite).

**Grenzen des Mechanismus.**
- Der Monitor muss überhaupt erst einmal feuern → ohne Produktions-Nutzung
  kein Feedback → reine Offline-GA reicht nicht ewig.
- False Positives des Monitors (z. B. durch seltene Input-Kombinationen)
  können zu unnötiger Hochstufung führen. Der Arch-Doc `tolerance_factor`
  (3σ) und `sample_rate` puffern das.

### 3.8 GA-Parameter

Aus `architecture_v1.2.0-draft.md` §4.3:

```
Population:    60 Individuen
Generationen:  30            → 1 800 Evaluierungen pro Modell
Evaluation:    ~500 ms       → Kurzer Decode (128 Tokens, 1 Run) + KL
Gesamt:        ~15 min pro Modell, einmalig, gecached

NSGA-II-spezifisch:
  Crossover:     SBX (Simulated Binary Crossover)
  Mutation:      Polynomial Mutation
  Selection:     Non-dominated Sorting + Crowding Distance
  Archive:       externe Pareto-Front (5–10 Punkte nach Konvergenz)
```

**Reproduzierbarkeit.** Seed + Run-ID wie bei Kernel-GA (§2.5).

**Early-Exit.** Wenn die Pareto-Front-Hypervolume über **10 Generationen**
sich nicht um > 1 % verbessert, stoppt die GA vorzeitig. Konsistent mit
Kernel-GA (§2.6); Begründung durch Punctuated-Equilibria-Effekt in GAs.

---

## 4. Fusion-Pass-GA

### 4.1 Problem und Scope

Der regelbasierte Fusion-Pass (Säule 2) fusioniert **sicher erkennbare**
Patterns. Aber es gibt nicht-lokale Fusionen, die je nach Register-Druck
und Cache-Verhalten **manchmal** besser, manchmal schlechter sind.
Beispiel: "Fused Residual+Norm+QKV+RoPE" in einem großen Kernel spart
Memory-Roundtrips, aber verbraucht viele VGPRs und könnte Occupancy
killen.

**Warum rocprof 1.17 die Fusion-GA besonders wichtig macht:**

Aus der Baseline:
```
gate_up_swiglu Kernel:  421 µs × 4139 Calls = 1 744 ms (65 %)
v0.x-Referenz:          394 µs × ähnlich    (nur 6 % schneller)
```

Die v0.x-Kernel sind pro Call nur marginal schneller — der **Haupt-
Hebel** bei v0.x-Speed ist die **Reduktion der Dispatches pro Token**
durch Fusion:

```
Dispatches pro Token (v1.0, Tuned):   ~608  Kernel-Launches
Dispatches pro Token (v0.x):          ~462  Kernel-Launches
Differenz:                             146  weniger Launches / Token
```

Jeder eingesparte Launch spart ~10–40 µs Dispatch-Overhead + ein
Memory-Roundtrip. Die Fusion-GA adressiert diese Differenz
systematisch.

### 4.2 Genome-Definition

Aus `architecture_v1.2.0-draft.md` §4.4, 1:1 übernommen:

```rust
pub struct FusionGenome {
    /// Pro Fusion-Kandidat: aktivieren oder nicht.
    pub active_fusions: Vec<FusionDecision>,
}

pub struct FusionDecision {
    pub pattern_id: FusionPatternId,
    pub active: bool,
    pub target_nodes: Vec<NodeId>,
}

pub enum FusionPatternId {
    NormQkv,                     // RMSNorm + QKV-GEMM
    NormQkvRope,                 // RMSNorm + QKV + RoPE
    ResidualNorm,                // Residual-Add + RMSNorm
    ResidualNormGemv,            // Residual + RMSNorm + nächster GEMV (Attention-Output-Pfad)
    GateUpSwiglu,                // Gate-GEMV + Up-GEMV + SwiGLU-Elementwise
    SwigluDown,                  // SwiGLU + Down-GEMV
    NormQkvRopeAttention,        // großer Fused-Kernel (Attention vollständig)
    KvAppendRope,                // KV-Cache-Append + RoPE
    // weitere: AttentionOutputResidualNorm, EmbeddingFromQuant, ...
}
```

### 4.3 Kandidaten-Fusionen (basierend auf rocprof 1.17)

**P0 — `ResidualNormGemv` (höchste Priorität).**

Die rocprof-Daten zeigen: `residual_add_inplace` + `rms_norm` + nächster
`gemv_q4_k_q8_inline` zusammen ~16.5 % der GPU-Zeit, aber drei separate
Kernel-Launches pro Layer (= ~108 × 36 Layer = ~3900 zusätzliche
Launches pro Token). v0.x hat `gemv_q4_k_q8_inline_residual` fertig
gebaut und spart pro Layer einen Launch + einen Memory-Roundtrip. Die
Fusion-GA bestätigt/variiert diesen Pattern für v1.0.

Erwarteter Gewinn: **+20–30 %** Decode (aus rocprof P0-Analyse):
```
30 → 36–39 tok/s nur durch diese eine Fusion.
```

**P1 — `KvAppendRope`.**

v0.x hat `kv_cache_append` + `rope` in `kv_write_rope` gefused — spart
einen Launch pro Layer. Gewinn moderat (aktueller RoPE-Anteil: 0.6 %),
aber additiv zu anderen Fusionen.

Erwarteter Gewinn: **+1–3 %** (marginal aber kostenlos).

**P2 — `GateUpSwiglu` ist bereits fused.**

Aus rocprof: `gemv_q4_k_gate_up_swiglu` existiert bereits als fused
Kernel (65 % GPU-Zeit). Der Kernel selbst ist BW-limitiert — weitere
Fusion mit Down-GEMV bringt wenig, weil Down-GEMV ein anderer Shape ist
(kein gemeinsamer Tile). Die GA darf `SwigluDown` trotzdem als Kandidat
prüfen.

**Hypothetische großen Fusionen.**

`NormQkvRopeAttention` (ein Kernel von RMSNorm bis Attention-Output)
würde drastisch viele Dispatches sparen (~5 Launches → 1), aber:

- VGPR-Druck: vermutlich > 200 VGPRs → Occupancy 4–6 Waves/CU
  (gegen Ziel 14+).
- LDS-Budget: Attention braucht KV-Cache-Zugriff + Softmax-Scratchpad
  → LDS > 32 KB, Workgroup-Limit.

Die GA prüft diesen Pattern als Kandidat; typischerweise fällt er am
Constraint durch. Aber: "typischerweise" ist genau der Ort, wo die GA
Überraschungen findet.

### 4.4 Constraints

```rust
fn validate_fusion_genome(fg: &FusionGenome, graph: &Graph) -> bool {
    // 1. Keine überlappenden Knoten
    let mut claimed: HashSet<NodeId> = HashSet::new();
    for decision in &fg.active_fusions {
        if !decision.active { continue; }
        for &node in &decision.target_nodes {
            if !claimed.insert(node) { return false; }   // schon belegt
        }
    }

    // 2. Prerequisite-Chain (z. B. NormQkvRope schließt separates NormQkv aus)
    for decision in &fg.active_fusions {
        if !decision.active { continue; }
        if let Some(conflicting) = CONFLICTING_PATTERNS.get(&decision.pattern_id) {
            for other in &fg.active_fusions {
                if other.active && conflicting.contains(&other.pattern_id) {
                    return false;
                }
            }
        }
    }

    // 3. VGPR-Budget für resultierenden Kernel
    for decision in &fg.active_fusions {
        if !decision.active { continue; }
        let estimated_vgpr = estimate_fused_vgpr(decision, graph);
        if estimated_vgpr > 104 { return false; }  // gleicher Budget wie Kernel-GA
    }

    // 4. LDS-Budget
    for decision in &fg.active_fusions {
        if !decision.active { continue; }
        let estimated_lds = estimate_fused_lds(decision, graph);
        if estimated_lds > 64 * 1024 { return false; }
    }

    true
}
```

**Konflikt-Tabelle.** Einige Fusionen sind mutual exclusive:

```rust
pub const CONFLICTING_PATTERNS: &[(FusionPatternId, &[FusionPatternId])] = &[
    (NormQkvRope, &[NormQkv]),                                // NormQkvRope umfasst NormQkv
    (NormQkvRopeAttention, &[NormQkvRope, NormQkv, ResidualNormGemv]),
    (ResidualNormGemv, &[ResidualNorm]),                      // RNG umfasst RN
    (GateUpSwiglu, &[SwigluDown]),                            // wenn Gate+Up fused, ist SwiGLU+Down separate
];
```

### 4.5 Fitness: End-to-End Decode-Speed

**Nicht einzelne Kernel-Zeiten, sondern Decode-tok/s auf einem Standard-
Prompt.** Warum? Weil Fusion-Effekte global sind: ein Fused-Kernel kann
pro-Call langsamer sein, aber durch reduzierte Dispatches das Gesamt-
Tok/s verbessern.

```rust
fn fitness_fusion(fg: &FusionGenome,
                  graph: &Graph,
                  baseline_tps: f32) -> f32 {
    if !validate_fusion_genome(fg, graph) { return 0.0; }

    let fused_graph = apply_fusions(graph, fg);
    let engine = build_engine_from_graph(&fused_graph);

    // Decode 128 Tokens, Median of 3 Runs
    let tps_samples: Vec<f32> = (0..3)
        .map(|_| measure_decode_tps(&engine, 128))
        .collect();
    let median_tps = median(&tps_samples);

    // Qualitäts-Check: wenn Fusion Numerik ändert (nur für großen Fused-
    // Kernel wie NormQkvRopeAttention relevant)
    if fg.changes_numerics() {
        let kl = measure_kl_on_validation_set(&engine);
        if kl > 0.05 { return 0.0; }  // sofortiges Reject
    }

    median_tps / baseline_tps   // relative Fitness
}
```

**Zeit pro Evaluation.** 128 Tokens × 3 Runs × ~4 s/Run = 12 s.
1 000 Evaluationen × 12 s = 200 min = **3.3 h**. Das ist zu viel.

**Kurzer Proxy: 32 Tokens × 1 Run = ~1 s.** 1 000 Evaluationen × 1 s =
17 min. Für finale Bewertung wird der lange Test auf die Top-10
Kandidaten angewendet.

### 4.6 GA-Parameter

Aus Arch-Doc §4.4:

```
Population:    50
Generationen:  20         → 1 000 Evaluierungen
Pro Evaluation: ~1 s      (32-Token Proxy)
Gesamt:        ~10 Minuten pro Modell-Architektur
Cached:        pro Architektur-Template (nicht pro Modell-Instanz)
```

**Cache-Key für Fusion-Cache.** Pro Architektur-Template (Llama-Style,
Mistral-Style, Qwen-Style), nicht pro Modell-Instanz. Zwei Llama-3.1-
Instanzen (8B und 70B) haben typisch gleiche Fusion-Decisions, nur
andere Shapes.

### 4.7 Interaktion mit Kernel-GA

**Fusion-GA braucht Kernel-GA-Ergebnisse.** Die Fitness einer Fusions-
Entscheidung hängt davon ab, wie schnell der resultierende Kernel ist —
und der wird vom Kernel-GA getuned. Sequenz:

```
1. Kernel-GA für alle Basis-Shapes (Q4_K q8_inline, Q6_K standard, ...)
   → Pareto-Fronten pro Shape im Cache
2. Fusion-GA
   → Pro Fusion-Kandidat: Kernel-GA für die fused Shape mit reduziertem
     Budget (Pop 30, Gen 15, ~3 min pro Shape — nur wenn nicht im Cache)
   → End-to-End-Messung
3. Finale Konfiguration: Fusion-Plan + Kernel-Varianten pro Fusion
```

Dieser Workflow ist in `rf-forge tune-all` fest integriert (§5).

---

## 5. rf-forge CLI

Das Offline-Tuning-Tool `rf-forge` ist aus `architecture_v1.2.0-draft.md`
§4.7 und Anhang D übernommen und hier auf die Phase-2-Realität
kalibriert.

### 5.1 Kommando-Struktur

```
rf-forge <subcommand> [options]

Subcommands:
  tune-kernels       Kernel-Auto-Tuning-GA
  tune-precision     Precision-Allocation-GA (NSGA-II)
  tune-fusion        Fusion-Pass-GA
  tune-all           Voller Cold-Start-Workflow (alle drei nacheinander)
  cache              Cache-Inspektion und Wartung
  bench              Benchmark-Suite auf gecachten Konfigurationen
  validate           VALU-Parity-Check für alle gecachten Kernel
  export             Export Config (für CI-Reproduzierbarkeit)
```

### 5.2 `tune-all` — der Standard-Cold-Start-Workflow

```fish
rf-forge tune-all --model ~/models/Qwen3-8B-Q4_K_M.gguf \
                  --target gfx1201 \
                  --log ~/tune-qwen3.log \
                  --seed 42 \
                  --budget 5h
```

**Interne Reihenfolge:**

```
1. P0-Gate-Check (§8.1)
   Messe hipStreamSynchronize-Rate. Wenn > 200/100 Tokens → ABORT mit
   klarer Fehlermeldung "Sync-Elimination noch nicht aktiv, GA würde
   falsche Zeiten messen. Fix P0 zuerst."

2. Model Introspection (Säule 1)
   Lade ModelProfile oder baue es neu (~30 s).

3. Fp32-Referenz-Logits für Precision-GA cachen
   FP32 VALU-Pfad auf Validation-Set. ~1–2 min.

4. Kernel-Tuning-GA
   - Alle benötigten Shapes × (FP8, FP16)
   - Top-3 Shapes (basierend auf rocprof-Prioritäten): volles Budget
   - Rest: reduziertes Budget (5 min/Shape)
   - Parity-Validation pro Kandidat (Säule 6)
   - Stability-Check auf Top-5-Gewinner
   - Dauer: ~3 h

5. Precision-Allocation-GA (NSGA-II)
   - Seed-Population aus ModelProfile.precision_recommendation
   - NSGA-II mit Pop 60, Gen 30
   - Monitor-Feedback-Constraints aus früheren Läufen (falls vorhanden)
   - Dauer: ~15 min

6. Fusion-Pass-GA
   - Alle aus §4.3 priorisierten Patterns
   - End-to-End Decode-Fitness
   - Dauer: ~10 min

7. Finaler Cache-Write
   - kernel_tuning.json + *.co Binaries
   - precision_pareto.json + gewählter Punkt
   - fusion_config.json
   - meta.json mit Hardware-Fingerprint + invalidation_hash
```

**Gesamtdauer für 8B-Modell:** ~3.5 h (Cold Start).
**Warm-Start** (Cache gültig): ~6 s (nur Lookup + Binary-Bind).

### 5.3 `tune-kernels` — nur Kernel-GA

```fish
rf-forge tune-kernels --model ~/models/X.gguf \
                      --shapes "gemv_q4_k_*,gemv_q6_k_*" \
                      --precision-levels fp8,fp16 \
                      --budget 30m \
                      --population 100 \
                      --generations 50 \
                      --seed 42
```

**Relevante Optionen:**

| Option | Default | Bedeutung |
|---|---|---|
| `--model <PATH>` | — | Pfad zur GGUF-Datei |
| `--target <gfx1201\|zen4>` | gfx1201 | Hardware-Target |
| `--shapes <GLOB>` | `*` | Welche Shapes tunen |
| `--precision-levels <LIST>` | `fp8,fp16` | GA läuft pro Level separat |
| `--budget <TIME>` | `10m` | Zeit-Limit pro Shape (Hard-Cap) |
| `--population <N>` | 100 | GA-Population (nur tune-kernels) |
| `--generations <N>` | 50 | GA-Generationen |
| `--seed <N>` | random (geloggt) | RNG-Seed für Reproduzierbarkeit |
| `--parity-tolerance <FLOAT>` | level-abhängig (§2.8) | Parity-Max-Abs-Err |
| `--parallel-shapes <N>` | 1 | Stabile Messung auf 1 GPU |
| `--log <PATH>` | stderr | JSONL-Log-Ausgabe |
| `--force-rebuild` | false | Cache ignorieren |
| `--dry-run` | false | Nur Plan, keine Ausführung |

### 5.4 `tune-precision` — Precision-GA separat

```fish
rf-forge tune-precision --model ~/models/Llama-3.1-8B-Q4_K_M.gguf \
                        --min-quality 0.98 \
                        --prompts ./eval-prompts.txt \
                        --target-point balanced
```

**Relevante Optionen:**

| Option | Default | Bedeutung |
|---|---|---|
| `--min-quality <FLOAT>` | 0.95 | Minimum KL-Qualitäts-Ratio auf Pareto-Front |
| `--prompts <PATH>` | Standard-Validation-Set | Custom Validation-Prompts |
| `--target-point <NAME>` | `balanced` | `fastest` / `balanced` / `quality` |
| `--speed-weight <FLOAT>` | 0.7 | Gewicht Speed vs. Memory (Objective 2) |
| `--memory-weight <FLOAT>` | 0.3 | dto. |
| `--languages <LIST>` | auto (aus Model-Metadaten) | Multilingual-Validation-Split |

**`--target-point`:**
- `fastest` — maximaler Speed, egal wie Qualität
- `balanced` — Standard: 95 %+ FP32-Qualität bei max Speed
- `quality` — 99 %+ FP32-Qualität, Speed zweitrangig

### 5.5 `tune-fusion` — Fusion-GA separat

```fish
rf-forge tune-fusion --model ~/models/Qwen3-8B.gguf \
                     --patterns "ResidualNormGemv,KvAppendRope" \
                     --min-delta 0.02
```

**Relevante Optionen:**

| Option | Default | Bedeutung |
|---|---|---|
| `--patterns <LIST>` | alle aus §4.3 | Nur bestimmte Patterns prüfen |
| `--min-delta <FLOAT>` | 0.02 (+2 %) | Pattern nur aktivieren wenn mind. +2 % |
| `--proxy-tokens <N>` | 32 | Proxy-Messung (kurz) |
| `--validation-tokens <N>` | 128 | Finale Validation (lang) |

### 5.6 `cache` — Inspektion und Wartung

```fish
# Liste aller Cache-Einträge
rf-forge cache list

# Details zu einem Shape
rf-forge cache inspect --shape "gemv_4096x12288_q4_k_m_fp8"

# Invalidierung
rf-forge cache invalidate --model ~/models/X.gguf          # ein Modell
rf-forge cache invalidate --older-than 30d                  # Zeitbasiert
rf-forge cache invalidate --all                             # komplett

# Gesamtgröße
rf-forge cache size

# Integritäts-Check
rf-forge cache verify   # prüft Signaturen aller Binaries
```

### 5.7 `validate` — VALU-Parity-Check aller Kernel

```fish
# Alle gecachten Kernel gegen VALU-Referenz prüfen
rf-forge validate --all --tolerance 1e-4

# Nur Kernel eines Modells
rf-forge validate --model ~/models/X.gguf

# Höhere Toleranz für Bf16-Kernel (separat)
rf-forge validate --all --tolerance-bf16 2e-3
```

**Exit-Code:**
- `0` — alle Kernel passen
- `1` — mindestens eine Violation (Details in Log)
- `2` — Cache beschädigt oder nicht lesbar

**Wann nutzen?** Nach ROCm-Update, nach Hardware-Änderung
(Treiber-Version), oder als Teil des CI-Pipelines.

### 5.8 `bench` — Benchmark auf gecachten Konfigurationen

```fish
rf-forge bench --model ~/models/Qwen3-8B-Q4_K_M.gguf \
               --runs 100 \
               --compare-baseline \
               --report ~/bench-qwen3.md
```

**`--compare-baseline`:** vergleicht gegen **rocprof 1.17-Werte**
(Phase-1-Baseline aus Anhang §9.1). Ergebnis als Speedup-Tabelle.

**Typische Ausgabe:**

```
                          1.17 Baseline    Current     Speedup
Decode                    30.0 tok/s      72.3 tok/s  2.41×
Prefill                   31.0 tok/s     890.2 tok/s 28.7×
gate_up_swiglu            421.6 µs        230.4 µs   1.83×
q8_inline                  25.8 µs         18.1 µs   1.42×
q6_k_standard              85.6 µs         32.0 µs   2.68×
sync_count_per_100_tok     ~65 000            120    542× weniger
```

### 5.9 `export` — Reproduzierbares Tuning für CI

```fish
# Auf Dev-Maschine
rf-forge export --model ~/models/X.gguf > tune-X.toml

# Auf CI/Build-Server
rf-forge tune-all --from-config tune-X.toml
```

**Inhalt der `.toml`:**

```toml
[run]
ga_seed = 42
ga_version = "1.0.0"
engine_git_hash = "a1b2c3d4..."

[hardware]
gpu_fingerprint = "gfx1201-ROCm7.2.2"
cpu_fingerprint = "Zen4-AMD-Ryzen-9-7945HX"

[model]
path_hint = "~/models/Qwen3-8B-Q4_K_M.gguf"
gguf_metadata_sha256 = "abc..."

[kernel_ga]
population = 100
generations = 50
parity_tolerance_fp8 = 0.0078
parity_tolerance_fp16 = 0.001

[precision_ga]
population = 60
generations = 30
target_point = "balanced"
min_quality = 0.95
speed_weight = 0.7
memory_weight = 0.3

[fusion_ga]
population = 50
generations = 20
min_delta = 0.02

[pareto_front]
# Exportierte Pareto-Front (für byte-exakte CI-Reproduktion)
points = [
  { kl_p95 = 0.002, speed_norm = 2.41, vram_gb = 10.2 },
  ...
]
```

### 5.10 Log-Output-Format (JSONL)

Jede GA-Evaluation schreibt eine Zeile:

```json
{"ts":"2026-04-21T14:12:03Z","run_id":"uuid-...","event":"eval",
 "shape":"gemv_4096x12288_q4_k_m_fp8",
 "generation":12,"individual":47,
 "genome":{"tile_m":64,"tile_n":128,"k_chunk":32,"use_lds_a":true,
           "k_unroll":4,"double_buffer":true,"dequant_strategy":"Inline"},
 "metrics":{"median_latency_us":312.5,"p95_latency_us":318.2,
            "heuristic_vgpr_estimate":78,
            "actual_vgpr_count":96, "actual_sgpr_count":24,
            "actual_lds_bytes":12288, "actual_waves_per_cu":16,
            "parity_pass":true,"max_abs_err":0.0021,
            "fitness":1.35},
 "seed":42}
```

Die Trennung zwischen `heuristic_vgpr_estimate` (vom Pre-Compile-Check)
und `actual_vgpr_count` (aus dem Code-Object, §2.3.1) erlaubt Post-hoc-
Kalibrierung der Heuristik.

Plus Event-Typen:
- `shape_start` / `shape_complete`
- `generation_complete` (siehe erweitertes Schema unten)
- `post_compile_vgpr_reject` (Kandidat nach Stage-2-VGPR-Gate verworfen)
- `parity_violation` (Details warum)
- `stability_fail` (Varianz > 2 %, Regression, ...)
- `early_exit` (Konvergenz oder Budget erreicht)

**`generation_complete`-Schema (erweitert durch §2.7.1):**

```json
{"ts":"2026-04-21T14:15:03Z","run_id":"uuid-...",
 "event":"generation_complete",
 "shape":"gemv_4096x12288_q4_k_m_fp8",
 "generation":12,
 "compile_cache_hits": 64, "compile_cache_misses": 36,
 "compile_cache_hit_rate": 0.64,
 "pre_compile_rejects": 40,
 "post_compile_vgpr_rejects": 8,
 "parity_rejects": 0,
 "benchmarked_individuals": 52,
 "compile_wall_ms": 1823, "benchmark_wall_ms": 4920,
 "best_fitness": 1.35, "median_fitness": 0.94,
 "invalid": 40}
```

Die Pipeline-Felder (`compile_cache_hit_rate`, `post_compile_vgpr_rejects`,
`compile_wall_ms`, `benchmark_wall_ms`) erlauben Post-Lauf-Analyse, ob die
Pipeline-Annahmen aus §2.7.1 in der Praxis halten.

---

## 6. Profile-Cache

### 6.1 Verzeichnis-Struktur

```
~/.rocmforge/
├── profiles/
│   └── <model-hash>/
│       ├── kernel_tuning.json       # Pareto-Fronten pro Shape
│       ├── kernels/
│       │   ├── gemv_4096x12288_q4_k_m_fp8__variant_0.co
│       │   ├── gemv_4096x12288_q4_k_m_fp8__variant_1.co
│       │   ├── ...
│       ├── precision_pareto.json    # NSGA-II Pareto-Front
│       ├── fusion_config.json       # aktive Fusionen
│       ├── fp32_reference_logits.bin # für Precision-GA Proxy-Loss
│       ├── meta.json                # Cache-Metadaten
│       └── monitor_feedback.json    # DriftEvents für nächste GA-Runde
└── logs/
    └── <run-id>.jsonl              # JSONL-Log pro rf-forge-Lauf
```

**`<model-hash>`:** SHA-256 über GGUF-Metadaten (nicht über Gewichte).
Gleiche Architektur + gleiche Shapes + gleicher Quant → gleicher Hash.
Das macht den Cache portabel zwischen Modell-Instanzen mit identischer
Form (z. B. zwei Quantisierungen desselben Basismodells).

### 6.2 `meta.json`

```json
{
  "cache_format_version": 1,
  "created_at": "2026-04-21T14:30:00Z",
  "ga_run_id": "uuid-...",
  "ga_seed": 42,
  "ga_version": "1.0.0",
  "engine_git_hash": "a1b2c3d4",

  "invalidation_hash": "sha256-of-toolchain-fingerprint",
  "toolchain": {
    "rocm_version": "7.2.2",
    "hipcc_version": "6.3.0",
    "target_arch": "gfx1201",
    "driver_version": "..."
  },

  "model": {
    "gguf_metadata_sha256": "abc...",
    "architecture": "llama",
    "n_layers": 36,
    "hidden_dim": 4096,
    "quant_format": "Q4_K_M"
  },

  "tuning_duration_seconds": 12480,
  "total_evaluations": {
    "kernel_ga": 48721,
    "precision_ga": 1823,
    "fusion_ga": 1000
  }
}
```

### 6.3 Cache-Key und Invalidierung

Entsprechend `dequant_ir_spec.md` §5.6:

```rust
pub struct KernelCacheKey {
    pub quant_format_id: QuantFormatId,
    pub shape: KernelShape,
    pub precision_level: PrecisionLevel,
    pub tile_config: TileConfig,
    pub gemv_variant: Option<GemvVariant>,
    pub target: KernelTarget,
}

pub struct CacheMetadata {
    pub invalidation_hash: [u8; 32],   // SHA-256 über Toolchain
    pub cache_format_version: u32,
    pub created_at: DateTime<Utc>,
}
```

**`invalidation_hash`** ist ein globaler Cache-Invalidator, der folgende
Komponenten hasht:

```
SHA-256(
    hipcc --version output ||
    rocm-smi --version output ||
    engine git hash ||
    ga_tuning_spec version ||
    architecture_v1.2.0-draft version ||
    dequant_ir_spec version
)
```

**Bei Änderung einer Komponente wird der komplette Cache invalidiert.**
Das ist bewusst grob: Änderungen an hipcc können **alle** Kernel
betreffen (anderer Code-Object-Format), nicht nur einzelne.

### 6.4 Portabilität

**Gleiche GPU, anderer Rechner:** Cache ist portabel, solange
`invalidation_hash` übereinstimmt (gleiche ROCm-Version, gleiche Engine-
Version). Der Nutzer kann `~/.rocmforge/` auf eine andere Maschine
kopieren.

**Andere GPU-Generation:** Cache ist **nicht** portabel. `target_arch`
im Hash → neuer invalidation_hash → Neu-Tuning nötig. Das ist korrekt:
gfx1201-Kernel laufen nicht auf gfx1100.

**Team-Workflow über `rf-forge export` / `--from-config`:**
Tuning-Ergebnisse können als `.toml` exportiert und auf einer zweiten
Maschine **reproduziert** werden (nicht nur kopiert). Vorteil: falls
Hardware leicht abweicht, wird die GA mit gleichen Seeds re-laufen, und
das Ergebnis ist nachvollziehbar.

### 6.5 Cold-Start-Verhalten (kein Cache vorhanden)

**Die wichtigste UX-Anforderung der Spec:** Wenn kein Cache existiert,
darf der User **nicht warten müssen**.

**Ablauf:**

```
User: rocmforge chat --model ~/models/Qwen3-8B-Q4_K_M.gguf
      [erster Start, kein Cache]

1. Model-Load + Introspection (~30 s)
   → ModelProfile erstellt

2. SOFORT START mit Phase-1-Standard-Varianten:
   - Kernel-Dispatch nutzt die in Phase 1 vorhandenen Standard-Kernel
     (gemv_q4_k_standard, gemv_q4_k_q8_inline, gemv_q6_k_standard, ...)
   - Bandit (Säule 4) wählt zwischen Standard und Q8-Inline
   - Decode läuft bei ~30 tok/s (rocprof 1.17-Niveau)
   - User sieht sofort Tokens, keine Wartezeit

3. IM HINTERGRUND: Mini-GA startet automatisch
   - Nur P1-Shape (gate_up_swiglu) × FP8
   - Pop 20, Gen 10 → 200 Evaluierungen
   - Dauer: ~2 min
   - Ergebnis: 1–2 verbesserte Kernel-Varianten

4. HOT-SWAP in den Bandit
   - Nach jeder Mini-GA-Generation wird die beste neue Variante
     in den Bandit eingespeist (KernelVariantSet wird live erweitert)
   - UCB1 erkundet die neue Variante automatisch
   - Wenn besser → konvergiert der Bandit dorthin

5. OPTIONAL: User startet volle GA manuell
   $ rf-forge tune-all --model ~/models/X.gguf
   → ~3.5 h im Hintergrund, Ergebnis im Cache für nächste Starts

6. BEI NÄCHSTEM START:
   - Cache-Hit → sofortiger Warm-Start mit getunten Kerneln
   - ~6 s Lade-Zeit
   - Volle Performance (~70+ tok/s nach Phase 2)
```

**Mini-GA-Parameter (festgelegt in dieser Spec):**

```
Population:     20
Generationen:   10              → 200 Evaluierungen
Pro Evaluation: ~500 ms          (ohne Parity-Check im Mini-Modus)
Gesamt:         ~2 Minuten
Shapes:         nur P1-Shape aus rocprof (gate_up_swiglu)
Levels:         nur Level 0 (FP8)
Parity:         Sampling 1/10, nicht 1/1 (Geschwindigkeit vor Gründlichkeit)
Budget-Hard-Cap: 3 Minuten (wird eher beendet als länger laufen)
```

**Warum Mini-GA-Ergebnisse akzeptieren, wenn sie weniger gründlich sind?**
Weil sie **strikt besser oder gleich** der Phase-1-Standardkernel sind —
der Bandit behält die bessere Variante. Mini-GA kann also nicht schaden;
im schlechtesten Fall ist der Gewinn 0.

**User-Eskalation.** Wenn der User volle GA explizit startet (`rf-forge
tune-all`), wird der Cache mit den **vollen Ergebnissen** überschrieben.
Mini-GA-Ergebnisse sind transient.

### 6.6 Cache-Größe

Aus `dequant_ir_spec.md` §4.7: realistisch ~10 MB pro Modell, nicht 4 GB
Vollmatrix.

| Komponente | Größe |
|---|---:|
| kernel_tuning.json (Pareto-Fronten Metadata) | ~50 KB |
| Kernel-Binaries (5 Shapes × 3 Varianten × 500 KB) | ~7.5 MB |
| precision_pareto.json | ~20 KB |
| fusion_config.json | ~5 KB |
| fp32_reference_logits.bin (1 640 Tokens × 151 936 Logits × 4 B) | ~1 GB! |
| monitor_feedback.json (für Feedback-Loop) | ~10 KB |
| meta.json | ~2 KB |

**Ausreißer: `fp32_reference_logits.bin`.** Die Logits-Referenz für den
Precision-GA-Proxy-Loss ist groß (Vocab-Dim × Tokens). Zwei Optionen:

1. **Speichern (Default).** 1 GB pro Modell. Bei ~10 Modellen im Cache:
   10 GB. Eher eine Frage der Plattenkapazität als der Performance.
2. **Nicht speichern (`--no-cache-logits`).** Die Referenz wird bei
   jedem `tune-precision`-Lauf neu berechnet (~1–2 min Overhead).
   Spart die 1 GB.

Standard: speichern. Für User mit knapper Platte: `rf-forge cache
config --no-cache-logits`.

### 6.7 Monitor-Feedback-Cache

Die DriftEvents aus Produktion landen in `monitor_feedback.json`:

```json
{
  "events": [
    {
      "ts": "2026-04-22T10:15:43Z",
      "token_index": 14823,
      "layer_idx": 7,
      "reason": "Fp8SaturationExceeded",
      "details": { "fraction": 0.15, "count": 614, "total": 4096 },
      "current_level": "Fp8E4M3",
      "recommended_level": "Fp16",
      "resolved_by_runtime": true
    }
  ],
  "summary": {
    "total_events": 23,
    "layers_escalated": [7, 14, 29],
    "constraints_for_next_ga": {
      "7": "Fp16",
      "14": "Fp16",
      "29": "Fp16"
    }
  }
}
```

Die nächste `rf-forge tune-precision`-Runde liest `constraints_for_next_ga`
und baut den `ConstrainedGenome` (§3.7) daraus.

---

## 7. Integration mit Phase-1-Infrastruktur

### 7.1 GA → Bandit (KernelVariantSet)

**Schnittstelle:** Die Kernel-GA produziert `KernelVariantSet`-Objekte
(aus `dequant_ir_spec.md` §8.4), die der UCB1-Bandit (Schritt 1.12,
Phase 1) direkt konsumiert.

```rust
// Die Kernel-GA schreibt diese Struktur in den Cache:
pub struct KernelVariantSet {
    pub base_key: KernelCacheKey,      // (Format, Shape, Level)
    pub variants: Vec<KernelVariant>,  // 3–5 Top-Pareto-Varianten
}

// Der Bandit lädt beim Modell-Start:
impl ShapeBandit {
    pub fn from_cache(cache: &Cache, shape: &ShapeKey) -> Self {
        let variant_set = cache.load_kernel_variant_set(shape);
        Self {
            shape_key: shape.clone(),
            variants: variant_set.variants,
            stats: vec![BanditArmStats::default(); variant_set.variants.len()],
            total_calls: 0,
        }
    }
}
```

**Konvergenz.** Der Bandit braucht ~100 Calls pro Shape, um zur
aktuellen Laufzeit-Optimum-Variante zu konvergieren
(`architecture_v1.2.0-draft.md` §2.5). Bei 20 unique Shapes × 100 Calls
= 2 000 Exploration-Calls, bei 50 tok/s Decode sind das 40 Sekunden
Warmup.

**Hot-Swap (Cold-Start Mini-GA).** Der Bandit akzeptiert zur Laufzeit
neue Varianten via `ShapeBandit::push_variant(new_variant)`. UCB1
erkundet sie gemäß Exploration-Term (wenig-genutzte Arme bekommen Bonus).

### 7.2 GA → Codegen (Dequant IR)

**Schnittstelle:** Die Kernel-GA ruft den Codegen aus `dequant_ir_spec.md`
§6 mit dem `TileConfig` aus dem `KernelGenome` auf.

```rust
// GA-interne Konvertierung
impl From<KernelGenome> for TileConfig {
    fn from(g: KernelGenome) -> TileConfig {
        TileConfig {
            tile_m: g.tile_m as usize,
            tile_n: g.tile_n as usize,
            k_chunk: g.tile_k as usize,
            lds_strategy: match (g.use_lds_for_a, g.use_lds_for_b) {
                (true, true) => LdsStrategy::LdsAB,
                (false, true) => LdsStrategy::DirectA_LdsB,
                (false, false) => LdsStrategy::DirectAB,
                (true, false) => LdsStrategy::LdsAB,  // Fallback (seltener Fall)
            },
            num_waves: g.waves_per_block as usize,
            unroll_factor: g.k_unroll as usize,
        }
    }
}

// Emission
let kernel_bin = codegen_gpu(
    &tile_config,
    shape,
    &quant_format,
    precision_level,
);
```

**Zusätzliche KernelGenome-Felder** (`prefetch_depth`, `double_buffer`,
`dequant_strategy`, `tiles_per_wave`) sind **nicht** Teil von
`TileConfig` aus der Dequant-IR-Spec. Sie werden über einen
**Extended-Codegen-Context** an den Emitter übergeben
([SPEC-ERWEITERUNG] gegenüber `dequant_ir_spec.md` §5.1; beide Specs
müssen in einem gemeinsamen Sync-Update abgeglichen werden).

### 7.3 GA ↔ Quality Monitor (Precision-Eskalation + Feedback-Loop)

**Bidirektional:**

**Richtung 1: Monitor → Runtime (sofort).** Beim DriftEvent stuft die
Runtime den betroffenen Layer eine Stufe hoch (§3.7). Das ist Phase-1-
Funktionalität (Schritt 1.14).

**Richtung 2: Monitor → GA (für nächste Offline-Runde).** Die
DriftEvents werden in `monitor_feedback.json` persistiert (§6.7). Die
nächste `rf-forge tune-precision`-Runde liest die Datei und baut
Constraints für die GA.

```rust
pub fn build_precision_ga_constraints(
    profile: &ModelProfile,
    feedback: &MonitorFeedback,
) -> PrecisionGaConfig {
    let mut min_level: Vec<PrecisionHint> = profile
        .precision_recommendation
        .clone();

    // Überschreibe mit Monitor-Constraints (nie absteigen)
    for (&layer_idx, &enforced_min) in &feedback.constraints_for_next_ga {
        if enforced_min > min_level[layer_idx] {
            min_level[layer_idx] = enforced_min;
        }
    }

    PrecisionGaConfig {
        seed: profile.recommended_seed(),
        min_level_per_layer: min_level,
        // ...
    }
}
```

### 7.4 GA ← Model Introspection (Seed-Population)

**Schnittstelle:** Wie §3.5 beschrieben. Das `ModelProfile` ist Phase-1-
Output (Schritt 1.13). Die Precision-GA nutzt `precision_recommendation`
als erstes Individuum; die Kernel-GA nutzt `snr_risk_layers` und
`critical_token_indices`, um Level-Constraints zu setzen.

**Zusätzlich.** Die Kernel-GA liest aus dem ModelProfile die `formats_
present`-Liste (aus `dequant_ir_spec.md` §8.1): nur tatsächlich benötigte
Formate werden getuned (bedarfsgetrieben).

### 7.5 Baseline-Referenz (1.17 als Fitness-Denominator)

**Konstante in der rf-forge-Binary:**

```rust
// src_v1/ga_tuning/baseline.rs
pub const ROCPROF_117_BASELINE: &[(ShapeName, f32)] = &[
    ("gemv_q4_k_gate_up_swiglu", 421.6),   // µs/Call
    ("gemv_q4_k_q8_inline",       25.8),
    ("gemv_q6_k_standard",        85.6),
    ("attention_decode",          10.7),
    ("rms_norm",                   3.6),
    ("rope",                       1.9),
    ("residual_add_inplace",       1.7),
    ("kv_cache_append",            1.9),
];

pub const ROCPROF_117_DECODE_TPS: f32 = 30.0;   // Tuned Run
```

Alle GA-Fitness-Funktionen konsultieren diese Konstanten. Fitness-Werte
> 1.0 bedeuten "schneller als 1.17", < 1.0 "langsamer" (und werden
verworfen, falls nicht speziell erwünscht, z. B. Precision-GA mit
Qualitäts-Priorität).

**Update bei späterer Baseline-Re-Messung.** Wenn eine neue Baseline
gemessen wird (z. B. nach ROCm-Update), wird die Konstante aktualisiert.
Das invalidiert automatisch Cache-Einträge, die gegen die alte Baseline
getuned wurden (über `invalidation_hash` aus §6.3).

---

## 8. Phase-2-Implementierungsplan

### 8.1 PREREQUISITE: Sync-Elimination (vor jeder GA-Arbeit!)

**Dies ist kein Phase-2-Feature, sondern ein Hard-Gate vor Phase 2.**

**Problem.** Die rocprof-1.17-Daten zeigen `83 260 hipStreamSynchronize`-
Aufrufe, 97.9 % der HIP-API-Wallclock. Das macht zwei Dinge kaputt:

1. **Fitness-Messungen sind verzerrt.** Jeder Kernel-Run ist von Sync
   eingerahmt (~45 µs pro Sync). Der gemessene `candidate_time_us`
   enthält Sync-Overhead, der nichts mit der Kernel-Qualität zu tun hat.
   GPU-Effizienz 62.7 % bedeutet: 37 % der Wallclock ist Leerlauf durch
   Sync.
2. **Kernel-Pipelining unmöglich.** Die GPU könnte die nächste Op bereits
   laden, während die aktuelle läuft. Mit Sync-pro-Kernel ist die
   Timeline strikt sequentiell — realistische BW-Messungen (21 % statt
   ≥ 50 %) werden unmöglich.

**Ohne Sync-Fix würde die GA kontraintuitive Ergebnisse liefern:**
"Kernel A ist schneller als B" — aber nur weil A zufällig weniger
Sync-Overhead zwischen Calls aufbaut, nicht weil A besser ist. Die
Pareto-Front wäre für die falsche Kostenfunktion optimiert.

**P0-Gate-Check (vor jedem `rf-forge`-Lauf):**

```rust
pub fn p0_gate_check() -> Result<(), P0GateError> {
    // Kurzer Test-Run (100 Tokens)
    let telemetry = run_short_decode(100);

    let sync_rate = telemetry.hip_stream_synchronize_count;
    if sync_rate > 200 {
        return Err(P0GateError {
            observed: sync_rate,
            threshold: 200,
            message: format!(
                "GA blocked: {} syncs per 100 tokens (threshold: 200).\n\
                 The Bandit is still using stream.synchronize() after each call.\n\
                 Fix P0-A (Sync-Elimination) before running the GA.\n\
                 See Arch-Doc §2.5 (Phase-2 Fused Epilog + Dirty-Flag).\n\
                 GA Fitness measurements would be corrupted by sync overhead.",
                sync_rate
            ),
        });
    }

    Ok(())
}
```

**Schwellenwert 200 pro 100 Tokens:** Begründung — mit vollständiger
Sync-Elimination sollte die Rate bei ≤ 120 liegen (v0.x-Messung), 200
gibt 60 % Puffer für Exploration-Phase.

**P0-Fixes (nicht Teil dieser Spec, sondern Phase-2-Arbeit VOR GA):**

**P0-A: Stream-Sync nach Bandit-Konvergenz eliminieren.**
```
Aktuell:  stream.synchronize() nach jedem gemv_tuned()
Phase 2:  HIP-Event-basiertes Timing während Exploration (100 Pulls)
          Nach Konvergenz: Sync nur noch periodisch (alle 32 Tokens)
          + Fused Epilog schreibt Dirty-Flag in Pinned Host Memory
          (Arch-Doc §2.6)
Aufwand:  1 Tag
Erwartung: 30 → 33 tok/s (rocprof P0-Analyse: +5–10 %)
Gate:     hipStreamSynchronize-Rate < 200/100 Tokens
```

**P0-B: Residual-fused Q4_K Q8-Inline GEMV.**
```
Aktuell:  gemv_q4_k_q8_inline + residual_add_inplace als 2 Kernel
v0.x:     gemv_q4_k_q8_inline_residual_multi_row als 1 Kernel
Phase 2:  Dequant-IR-Emitter um Residual-Fused-Variante erweitern
          (Strukturell: Dequant-Programm bekommt einen zusätzlichen
          Load-Op für Residual + finalen Add vor DowncastToFP8)
Aufwand:  2–3 Tage
Erwartung: 33 → 36–39 tok/s (rocprof P0: +20–30 %)
```

**Nach P0-Fixes liegt die Baseline bei ~40 tok/s Decode.** Ab hier
startet die produktive GA-Arbeit.

### 8.2 Kernel-GA (größter Hebel nach P0-Fixes)

**Fokus aus rocprof 1.17:** `gate_up_swiglu` (65 %) mit FP8-Pfad.

```
Phase 2.2:
  - Kernel-GA läuft auf allen relevanten Shapes × (FP8, FP16)
  - FP8-Pfad ist der strukturelle Hebel (doppelte WMMA-Rate,
    halbierter VGPR-Bedarf, halbierte LDS-BW)
  - Parity-Validation + Stability-Check (§2.8, §2.9)
  - Dauer: ~3 h pro Modell

Erwartung: 40 → 55–65 tok/s
  Begründung:
    - gate_up_swiglu FP8: v0.x fehlt dieser Pfad komplett
    - Q6_K Q8-Inline + WMMA-Variante: Q6_K heute nur 1 Variante
      im Bandit (keine Wahl); GA liefert 3–5 → Bandit konvergiert
      zum Optimum
    - Tile-Config-Exploration entdeckt LDS-vs-Direct-Global-Trade-
      offs systematisch (v0.x-Erkenntnis #4: Direct-Global kann
      5× schneller sein für bestimmte Shapes)
```

### 8.3 Precision-GA (FP8-Default → pro-Layer-Optimierung)

```
Phase 2.3:
  - NSGA-II mit Seed aus ModelProfile
  - KL-Divergenz-Proxy auf 20 Prompts
  - Pareto-Front 5–10 Punkte
  - Monitor-Feedback-Constraints aus §3.7
  - Dauer: ~15 min pro Modell

Erwartung: 55–65 → 70–80 tok/s
  Begründung:
    - Layer mit SNR-Risk bleiben auf FP16 (Numerik stabil)
    - Unkritische Layer gehen auf FP8 (2× WMMA-Rate + halbe VRAM-BW)
    - KV-Cache auf FP8-E5M2: spart Attention-Read-Bandwidth
    - Typisch 10–20 % der Layer brauchen Hochstufung (Arch-Doc §4.3)
```

### 8.4 Fusion-GA (braucht Kernel-GA-Ergebnisse als Input)

```
Phase 2.4:
  - Aktive Patterns: ResidualNormGemv (P0 aus rocprof), KvAppendRope
  - End-to-End Decode-Fitness auf 32-Token-Proxy
  - Dauer: ~10 min pro Architektur-Template

Erwartung: 70–80 → 80–90 tok/s
  Begründung:
    - ResidualNormGemv spart 1–2 Dispatches × 36 Layer × 4 GEMV-Typen
      = ~150 Dispatches pro Token
    - Bei ~40 µs pro Dispatch: ~6 ms pro Token Einsparung
    - Bei aktueller 25 ms/Token (40 tok/s): ~24 % Gewinn
```

### 8.5 ASM-Codegen (nach GA, für Top-3-Kernel)

```
Phase 2.5 (nach GA):
  - Die 3 heißesten Kernel-Varianten (aus rocprof nach Phase 2.4):
    gate_up_swiglu fused, ResidualNormGemv, Q6_K Q8-Inline
  - Handoptimierter Inline-ASM statt hipcc-Codegen
  - Dequant-Bit-Manipulation: 10–20 % über besseres Register-Scheduling
  - VNNI-Pipelining: 15–30 % bei Q8-Inline-Varianten
  - Prefetch-Platzierung: 5–15 % bei BW-nahen Kerneln

Erwartung: 80–90 → 90–110 tok/s
  Begründung:
    - v0.x-Erkenntnis #5: "Kernel sind Gold, Architektur drumherum ist
      das Problem". Auf gute Architektur (GA + Fusion) drauf, können
      Kernel dann Silicon-nah optimiert werden
    - Aufwand ~1 Woche, Ergebnis ~1–2 Jahre nutzbar
```

### 8.6 Meilensteine und erwartete tok/s

```
┌───────────────────────────────┬──────────┬──────────┬───────────┐
│ Meilenstein                   │ Decode   │ Prefill  │ Aufwand   │
├───────────────────────────────┼──────────┼──────────┼───────────┤
│ Phase 1 IST (rocprof 1.17)    │ 30 tok/s │   31     │ erledigt  │
│ P0-A: Sync-Elimination        │ 33       │   31     │ 1 Tag     │
│ P0-B: Residual-Fused GEMV     │ 40       │   31     │ 2–3 Tage  │
│ P1: WMMA-Batched Prefill      │ 40       │  300+    │ 3–4 Tage  │
│ Kernel-GA                     │ 55–65    │  500+    │ 1–2 Wochen│
│ Precision-GA                  │ 70–80    │ 1 000+   │ 1 Woche   │
│ Fusion-GA                     │ 80–90    │ 2 000+   │ 1 Woche   │
│ ASM-Codegen (Top-3)           │ 90–110   │ 3 000+   │ 1 Woche   │
│ Physik-Limit (BW)             │  143     │ ~15 000  │  —        │
│ Arch-Doc-Ziel (§1.4)          │  125     │ 7 500    │  —        │
└───────────────────────────────┴──────────┴──────────┴───────────┘
```

**Kumulative Dauer:** ~4–5 Wochen aktive Entwicklung, mit Puffern für
GA-Debug (unerwartete Constraint-Verletzungen, Parity-Failures).

**Realismus der Zahlen.** Die Decode-Spalte ist konservativ — jede
Zahl ist mit "bis zu X" zu lesen; die GA liefert die echten Werte.
Die Prefill-Spalte eskaliert stärker, weil aktuell massiv unter dem
Physik-Limit (31 tok/s vs. ~15 000 Physik): batched WMMA allein bringt
10–30× (rocprof P1-Analyse).

**Nicht-linear:** Die einzelnen Gewinne sind **nicht additiv**, weil
jeder Fix die Flaschenhälse verschiebt (v0.x-Erkenntnis #1: "Profiling
invertiert die Schätzungen"). Nach der Kernel-GA könnte sich zeigen,
dass Fusion **weniger** Gewinn bringt als hier geschätzt, weil die
Kernel-GA bereits viele Dispatches eliminiert hat. Umgekehrt könnte
die Precision-GA mehr liefern, wenn der FP8-Pfad im Kernel-GA bessere
Qualität als erwartet zeigt.

### 8.7 Risiken und Mitigations

| Risiko | Mitigation |
|---|---|
| GA-Fitness durch Residual-Sync verfälscht | P0-Gate-Check (§8.1) vor jedem Lauf |
| GA findet schnelle aber brittle Kandidaten | Stability-Check + Parity-Validation (§2.8, §2.9) |
| Precision-GA-Proxy divergiert von Perplexity | Multi-Prompt-Set (short/medium/long) + max-KL-Reject (§3.3) |
| Fusion-GA erzeugt Over-Fusion (VGPR-Overrun) | Constraint-Validator (§4.4), Compile-Time-Reject |
| Cache-Invalidierung bei ROCm-Update verwirrt User | Klare Fehlermeldung + `rf-forge cache verify` (§5.6) |
| Cold-Start-Mini-GA blockiert UI | Läuft in separatem Thread, Hot-Swap via Bandit-Push |
| Monitor-False-Positives stufen Layer hoch | Arch-Doc `tolerance_factor` 3σ + Sampling-Rate-Throttling |

---

## 9. Anhang

### 9.1 rocprof-Baseline-Daten (vollständig, aus Schritt 1.17)

**Tuned-Run (Bandit aktiv, aktueller Stand):**

| Kernel | Count | Σ µs | Ø µs | % GPU |
|---|---:|---:|---:|---:|
| `gemv_q4_k_gate_up_swiglu` | 4 139 | 1 744 852 | 421.6 | **65.4** |
| `gemv_q4_k_q8_inline` | 16 555 | 427 171 | 25.8 | 16.0 |
| `gemv_q6_k_standard` | 4 253 | 364 230 | 85.6 | 13.7 |
| `attention_decode` | 4 139 | 44 065 | 10.7 | 1.7 |
| `rms_norm` | 8 392 | 29 974 | 3.6 | 1.1 |
| `rms_norm_batched` | 8 278 | 19 094 | 2.3 | 0.7 |
| `rope` | 8 278 | 15 788 | 1.9 | 0.6 |
| `residual_add_inplace` | 8 278 | 14 426 | 1.7 | 0.5 |
| `kv_cache_append` | 4 139 | 8 016 | 1.9 | 0.3 |
| **Σ GPU** | | **2 667 616** | | 100 |

Decode: 30.6 tok/s. Prefill: 31 tok/s. VRAM Peak: 10.05 GB.

**Systemische Zahlen:**
- `hipStreamSynchronize`-Aufrufe: 83 260 (97.9 % HIP-API-Wall)
- GPU-Effizienz (GPU-Zeit / Wallclock): 62.7 %
- BW-Effizienz (von 640 GB/s): 21 %
- Σ Kernel-Launches pro Token: ~608
- Σ Gewichts-Traffic pro Token: ~4.5 GB → ~130 tok/s BW-Limit

**Occupancy (aus Kernel-Trace):**

| Kernel | VGPR | LDS/Block | Waves/CU | Hinweis |
|---|---:|---:|---:|---|
| `gemv_q4_k_gate_up_swiglu` | 192 | 0 | 8 | hoher VGPR-Druck, aber OK |
| `gemv_q4_k_q8_inline` | 192 | 0 | 8 | wie oben |
| `gemv_q4_k_standard` | 112 | 0 | 13 | Bandit verwirft diesen Pfad |
| `gemv_q6_k_standard` | 72 | 0 | 21 | gut |
| `attention_decode` | 40 | 512 B | 32 | hohe Occupancy |

### 9.2 v0.x-Vergleich — was v0.x besser macht und warum

**Schlüssel-Vergleich aus rocprof 1.17 (v0.x-Zahlen gemessen als
Referenz):**

| Kernel | v1.0 | v0.x | Delta |
|---|---:|---:|---:|
| `gate_up_swiglu` | 421 µs/Call | 394 µs/Call | v0.x 6 % schneller |
| `q8_inline_residual_multi_row` | — (nicht fused) | 41.7 µs/Call | nicht verfügbar in v1.0 |
| `q6_k_q8_inline` | — (nicht vorhanden) | ~60 µs/Call | nicht verfügbar in v1.0 |
| Dispatches / Token | ~608 | ~462 | v0.x 24 % weniger |
| `hipStreamSynchronize` | 83 260 | 114 | v0.x 730× weniger |
| **Decode** | **30 tok/s** | **41.7 tok/s** | **v0.x 39 % schneller** |

**Warum v0.x im Decode schneller ist:**

1. **Fused Kernels:** `q8_inline_residual_multi_row` spart Launches.
2. **Sync-frei nach Startup:** v0.x hat 730× weniger Syncs.
3. **Q6_K Q8-Inline existiert:** v0.x nutzt es für den LM-Head.

**Warum v1.0 trotzdem vielversprechend ist:**

1. **FP8-Pfad:** v0.x hat keinen. Strukturelle 2× WMMA-Rate nutzbar.
2. **Generischer Codegen:** neue Quant-Formate in Minuten statt Tagen
   (Dequant-IR-Spec §9). v0.x wäre für jedes neue Format monatelang
   blockiert.
3. **GA-getunte Tile-Configs:** v0.x-Kernel sind handgetunt für wenige
   Shapes; GA findet Kombinationen, die ein Mensch nicht explorieren
   würde.
4. **Pro-Layer-Precision:** v0.x ist Single-Precision pro Tensor-Typ;
   v1.0 optimiert pro Layer.

Phase 2 holt v0.x-Gewinne zurück (Fusion, Sync-Elim) **und** addiert
die v1.0-Vorteile (FP8, GA). Das ist der Weg zu 125 tok/s.

### 9.3 Hardware-Limits gfx1201 (Referenz)

```
GPU: AMD Radeon RX 9070 XT
Architecture: gfx1201 (RDNA 4, Navi 48 Vollausbau)

Compute:
  Compute Units:          64
  AI Accelerators:        128 (2× pro CU)
  WMMA Intrinsics:        FP8 (E4M3/E5M2), FP16, BF16
  Wave Size:              32

Register / SRAM:
  VGPRs pro CU:           1 536
  Ziel-VGPRs pro Wave:    104 (→ ~14 Waves/CU, Sweet-Spot)
  LDS pro Workgroup:      64 KB
  LDS pro CU:             128 KB

Memory:
  VRAM Total:             17.1 GB (16 GB GDDR7 + Reserve)
  Memory Bandwidth:       640 GB/s
  Memory Clock:           ~20 Gbps effektiv

Verhältnis zu gfx1100 (RDNA 3):
  CUs:                    64 vs 32 (2×)
  AI Accelerators:        128 vs 0 (neu!)
  FP8-WMMA:               nativ vs nicht vorhanden
```

### 9.4 Validierungs-Set für Precision-GA Proxy-Loss

**Kanonisches Validation-Set (fixiert in der Spec, aus
`inference_test_prompts_15.json` + multilinguale Ergänzungen):**

```jsonc
// Set "short" (5 Prompts, ~40 Tokens gesamt, ~100 ms pro Eval)
[
  { "id": "s01", "lang": "en", "text": "The capital of France is" },
  { "id": "s02", "lang": "en", "text": "2 + 2 equals" },
  { "id": "s03", "lang": "en", "text": "The sky is blue because" },
  { "id": "s04", "lang": "de", "text": "Die Hauptstadt Deutschlands ist" },
  { "id": "s05", "lang": "zh", "text": "中国的首都是" }
]

// Set "medium" (10 Prompts, ~320 Tokens gesamt, ~500 ms pro Eval)
// Dies ist das STANDARD-Set für die regulären GA-Iterationen.
[
  { "id": "m01", "lang": "en",
    "text": "Explain briefly what a mutex is in concurrent programming." },
  { "id": "m02", "lang": "en",
    "text": "Write a haiku about autumn leaves." },
  { "id": "m03", "lang": "en",
    "text": "List three differences between Python and Rust." },
  { "id": "m04", "lang": "en",
    "text": "What is the Chinese room argument?" },
  { "id": "m05", "lang": "en",
    "text": "Describe a function that computes the Fibonacci sequence." },
  { "id": "m06", "lang": "de",
    "text": "Erkläre kurz, warum der Himmel blau ist." },
  { "id": "m07", "lang": "de",
    "text": "Schreibe einen kurzen Dialog zwischen einem Arzt und Patienten." },
  { "id": "m08", "lang": "zh",
    "text": "什么是机器学习?请简短解释。" },
  { "id": "m09", "lang": "fr",
    "text": "Expliquez brièvement ce qu'est l'effet de serre." },
  { "id": "m10", "lang": "es",
    "text": "¿Qué es un algoritmo genético? Explica brevemente." }
]

// Set "long" (5 Prompts, ~1280 Tokens gesamt, ~2000 ms pro Eval)
// Wird für finale Pareto-Front-Bewertung verwendet, nicht für Iterationen.
[
  { "id": "l01", "lang": "en",
    "text": "Write a detailed technical explanation of how GPU memory coalescing works, including examples with warp-level access patterns and cache line alignment. The response should be around 200 tokens." },
  { "id": "l02", "lang": "en",
    "text": "Explain the difference between supervised, unsupervised, and reinforcement learning, with at least one practical example for each. Length: around 200 tokens." },
  { "id": "l03", "lang": "en",
    "text": "Describe the architecture of a transformer model in detail, covering self-attention, position embeddings, and feed-forward layers. Length: around 250 tokens." },
  { "id": "l04", "lang": "de",
    "text": "Beschreibe ausführlich, wie die TCP-Flusskontrolle funktioniert, inklusive Sliding Window und Congestion Control. Länge: etwa 250 Tokens." },
  { "id": "l05", "lang": "zh",
    "text": "请详细解释什么是量子计算,包括量子比特、叠加态和量子纠缠的概念。回答长度约250个token。" }
]
```

**Multilinguale Abdeckung.**
- Englisch: dominant (für die meisten Modelle Trainings-Mehrheit)
- Deutsch, Französisch, Spanisch: europäische Sprachen (Llama-3.1, Qwen3)
- Chinesisch: Qwen-Familie primär trainiert darauf

Wenn ein Modell **nicht** multilingual ist (z. B. pure English), werden
die nicht-englischen Prompts automatisch übersprungen. Die `ModelProfile`-
Metadaten tragen die trainierten Sprachen (aus GGUF-Metadaten
extrahiert).

### 9.5 Sync-Liste für `architecture_v1.2.0-draft.md`

Während der GA-Spec-Erstellung sind folgende Erweiterungen/Refinements
gegenüber dem Arch-Doc aufgefallen, die in einem späteren Sync-Commit
aufgenommen werden sollten:

**Sync-Update 1: §4.2 KernelGenome vs. TileConfig-Diskrepanz.**

Das `KernelGenome` in Arch-Doc §4.2 enthält Felder
(`prefetch_depth`, `double_buffer`, `dequant_strategy`, `tiles_per_wave`),
die in `dequant_ir_spec.md` §5.1 `TileConfig` nicht vorhanden sind. Die
GA-Spec §7.2 markiert das als `[SPEC-ERWEITERUNG]` und schlägt einen
**Extended-Codegen-Context** vor, der die fehlenden Felder an den
Emitter durchreicht.

**Empfohlenes Sync-Update:** `TileConfig` in
`dequant_ir_spec.md` §5.1 um die vier Felder erweitern, oder ein
separates `KernelGenomeExtras`-Struct einführen, das der Codegen
zusammen mit `TileConfig` nimmt.

**Sync-Update 2: §4.3 Precision-Allocation GA-Parameter.**

Arch-Doc §4.3 spezifiziert `Population 60, Generationen 30`. GA-Spec §3.8
übernimmt das exakt. Keine Abweichung.

**Sync-Update 3: §4.4 Fusion-GA-Parameter.**

Arch-Doc §4.4 spezifiziert `Population 50, Generationen 20`. GA-Spec §4.6
übernimmt das exakt. Keine Abweichung.

**Sync-Update 4: §2.6 Quality-Monitor Feedback-Loop.**

Arch-Doc §2.6 beschreibt den Quality-Monitor, aber nicht explizit den
**Feedback-Loop zur Precision-GA** (§3.7 dieser Spec). Der Mechanismus
ist in Arch-Doc implizit (DriftEvents persistieren in
`monitor_feedback.json`, nächste GA liest Constraints), sollte aber
explizit in §2.6 oder §4.3 gemacht werden.

**Empfohlenes Sync-Update:** neuer Absatz "Monitor → GA Feedback-Loop"
in §4.3, der auf GA-Spec §3.7 verweist.

**Sync-Update 5: P0-Gate-Mechanismus.**

Das P0-Gate (GA-Spec §8.1) ist **nicht** im Arch-Doc erwähnt. Das ist
eine Phase-2-Implementierungs-Entscheidung, basierend auf den rocprof-
1.17-Messungen, die **nach** dem Arch-Doc-Freeze entstanden sind.

**Empfohlenes Sync-Update:** neuer Abschnitt "Phase-2-Prerequisites" im
Arch-Doc §9.2 (Phase-2-Roadmap) oder §4.7 (rf-forge), der das P0-Gate
als Pflicht-Check referenziert.

**Sync-Update 6: rocprof-1.17-Baseline als Fitness-Referenz.**

Arch-Doc §4 spezifiziert Fitness abstrakt als "negative Median-Zeit".
GA-Spec §1.3 konkretisiert auf `baseline / candidate`-Form mit
rocprof-1.17-Werten als Denominator. Das ist eine Refinement-Ergänzung,
keine Abweichung.

**Empfohlenes Sync-Update:** in Arch-Doc §4.2/4.3/4.4 Hinweis
"konkrete Fitness-Konstanten siehe ga_tuning_spec.md §1.3".

**Sync-Update 7: Cold-Start-Pfad mit Mini-GA + Hot-Swap.**

Arch-Doc §4.7 beschreibt Cold-Start als "stundenlanges Warten auf
`rf-forge tune-all`". GA-Spec §6.5 spezifiziert einen alternativen
Pfad: sofort Phase-1-Varianten + Mini-GA im Hintergrund + Hot-Swap in
den Bandit. Das ist eine UX-Verbesserung, die in der Arch-Doc-Version
nicht vorhanden war.

**Empfohlenes Sync-Update:** in Arch-Doc §4.7 Absatz ergänzen, der den
Mini-GA-Pfad + Hot-Swap beschreibt, mit Verweis auf GA-Spec §6.5.

**Sync-Update 8: Two-Stage-VGPR-Gate (v1.0.1 Amendment 1).**

Arch-Doc §4.2 beschreibt Kernel-GA-Constraints mit einem einfachen
Pre-Compile-VGPR-Check (Schwellenwert 104). GA-Spec §2.3 / §2.3.1
teilt das in zwei Stages:

- Stage 1 Pre-Compile: großzügiger Schwellenwert **150** (Heuristik
  ist oft 30 % daneben wegen LLVM-Spilling).
- Stage 2 Post-Compile: echte VGPR-Zahl aus dem Code-Object (ELF-Notes)
  gelesen, Hard-Reject bei < 4 Waves/CU.

Das verhindert Fitness-Verzerrung durch Occupancy-Kollaps, den die
Heuristik nicht erkennt.

**Empfohlenes Sync-Update:** Arch-Doc §4.2 um Zwei-Stage-Formulierung
ergänzen, Verweis auf GA-Spec §2.3.1.

**Sync-Update 9: Early-Exit 10 statt 5 Generationen (v1.0.1 Amendment 2).**

Arch-Doc §4 spezifiziert keinen expliziten Konvergenz-Early-Exit-
Schwellenwert. GA-Spec §2.6 / §3.8 setzt ihn auf **10 Generationen
ohne Hypervolume-Verbesserung > 1 %** (statt 5). Begründung:
Punctuated-Equilibria-Verhalten in GAs — 5 Gen = 10 % Budget ist zu
aggressiv, 10 Gen = 20 % Budget erlaubt Plateau-Durchbrüche.

**Empfohlenes Sync-Update:** in Arch-Doc §4.2 (und analog §4.3/§4.4)
Early-Exit-Schwellen explizit dokumentieren (10 Gen).

**Sync-Update 10: Compile-Pipeline-Trennung (v1.0.1 Amendment 3).**

Arch-Doc §4.2 rechnet ~100 ms pro GA-Evaluation (5 ms Messung + 95 ms
Compile). GA-Spec §2.7.1 zeigt, dass die naïve
Compile→Benchmark→Compile-Reihenfolge das Budget um ~50 % sprengt.
Lösung: innerhalb einer Generation parallele Compiles (CPU-Threads) +
sequenzielles Benchmark (GPU) + aggressiver Compile-Cache
(60–70 % Hit-Rate erwartet).

**Empfohlenes Sync-Update:** Arch-Doc §4.2 um Pipeline-Absatz ergänzen,
Verweis auf GA-Spec §2.7.1. Zusätzlich: `--compile-threads`-Option in
Arch-Doc §4.7 `rf-forge`-CLI-Beschreibung.

---

*Ende der GA Tuning Specification v1.0.1-final.*

**Changelog:**
- **1.0.0** (2026-04-21) — Initial Release, Review abgenommen.
- **1.0.1** (2026-04-21) — Amendments: Two-Stage-VGPR-Gate,
  Early-Exit 10 Gen, Compile-Pipeline-Trennung.
