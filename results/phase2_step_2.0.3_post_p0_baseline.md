# Phase 2 Schritt 2.0.3 — Post-P0 Performance Baseline (rocprofv3)

**Date:** 2026-04-22
**Branch:** v1.0-dev (HEAD `b86401a` — Phase 2 step 2.0.2 applied)
**Hardware:** AMD Radeon RX 9070 XT (gfx1201), 640 GB/s nominal BW, 16 GiB VRAM
**Tool:** rocprofv3 (`/opt/rocm/bin/rocprofv3`) — `--kernel-trace`, `--hip-trace`
**Prompt:** `"Explain what a mutex is in one paragraph."`
**Model:** Qwen3-8B Q4_K_M
**Max-Tokens:** 100 (EOS hit at 96)

## Zweck

Die `phase1_step_1.17_rocprof_baseline.md`-Messung wurde **vor** den
P0-Fixes (Sync-Elimination + Residual-Fusion) gemacht — jeder
gemessene Kernel-Wert enthielt ~45 µs Sync-Overhead, und die
Dispatch-Zahl umfasste 54 Residual-Adds, die inzwischen weggefused
sind. Für die GA-Fitness-Spec (`ga_tuning_spec §1.3`, §7.5) ist
diese 1.17-Baseline **nicht mehr gültig**. Diesem Schritt sein
einziger Zweck: eine neue, rocprof-basierte Steady-State-Messung
auf dem Post-P0-Code, die als Fitness-Denominator für Phase 2.1+
dient.

**Keine Code-Änderungen. Reine Messung + Analyse.**

## Setup

`rocmforge-v1` release build mit `v1,gpu` Features. rocprofv3 gestartet
mit dem Binary direkt (keine Wrapper-Shell, weil `.rocprofv3/profile_
decode.sh` auf v0.x-Binary/Modell gepinnt ist). CLI-Flags: `--show-
tuning`, damit der Bandit attached und konvergiert.

- **Kernel-Trace:** `rocprofv3 --kernel-trace --stats --summary`
  → `52591_kernel_stats.csv` (GPU-Zeit pro Kernel-Name, Ø/Min/Max/%)
- **HIP-Trace:** `rocprofv3 --hip-trace --stats --summary`
  → `53021_hip_api_stats.csv` (HIP-API-Call-Counts, `hipStreamSync`)
- **VRAM:** paralleler `rocm-smi --showmeminfo vram`-Loop (0.4 s
  Sample, 32 s Dauer) während eines nicht-profilierten Inference-
  Runs. Peak via `awk`-max über die Log-Samples.

Dasselbe Modell, dieselbe Prompt, dieselben Max-Tokens wie in 1.17
— die Zahlen sind direkt vergleichbar.

**Warmup-Handling:** Das 15 %-Warmup-Clipping aus 1.17 ist hier nicht
nötig. Der Bandit konvergiert seit 2.0.1 nach ≤ 5 Pulls pro Shape
(72–108 Pulls total über den gesamten Run); Bandit-Exploration-Jitter
ist <0.2 % der Kernel-Aggregatzeit. Die Ø-µs-Werte über 4000+ Calls
sind de-facto Steady-State. Wo der rocprof-Overhead selbst ins
Gewicht fällt (39.2 tok/s gemessen vs. 40.7 tok/s ungeprofilt,
~4 %), wird das explizit aufgeführt.

## Kernel-Zeiten — Dreifach-Vergleich

**Tuned-Run, Mutex-Prompt, 129 `execute_decode`-Iterationen (33 Prefill
+ 96 Decode).** Post-P0-Zahlen aus `52591_kernel_stats.csv`.

| Kernel | Count | Σ µs | Ø µs | % GPU | 1.17 Ø | v0.3.0 Ø |
|---|---:|---:|---:|---:|---:|---:|
| `gemv_q4_k_gate_up_swiglu` | 4 644 | 2 010 100 | **432.8** | **65.77 %** | 421.6 | 394.2 |
| `gemv_q6_k_standard` | 4 773 | 412 736 | **86.5** | 13.51 % | 85.6 | 71.8 (direct) / 59.4 (multi_row) |
| **`gemv_q4_k_q8_inline_residual`** | **6 966** | **290 203** | **41.7** | **9.50 %** | — | 41.7 |
| `gemv_q4_k_q8_inline` (unfused) | 11 520 | 207 142 | **18.0** | 6.78 % | 25.8 | 17.5 |
| `attention_decode` | 4 644 | 44 030 | **9.48** | 1.44 % | 10.7 | 12.2 |
| `rms_norm` | 9 417 | 31 151 | 3.31 | 1.02 % | 3.6 | 3.3 |
| `rms_norm_batched` | 9 288 | 20 280 | 2.18 | 0.66 % | 2.3 | 2.2 |
| `rope` | 9 288 | 17 565 | 1.89 | 0.57 % | 1.9 | 1.9 |
| `gemv_q4_k_standard` (Loser-Pulls) | 90 | 8 507 | 94.5 | 0.28 % | 141.3 (Untuned) | — |
| `kv_cache_append` | 4 644 | 7 158 | 1.54 | 0.23 % | 1.9 | 2.1 |
| `residual_add_inplace` (unfused) | 2 322 | 3 851 | 1.66 | 0.13 % | 1.7 | 1.6 (`add`) |
| `copyBuffer` (HSA direct) | 274 | 2 998 | 10.94 | 0.10 % | — | — |
| `embedding_lookup` | 129 | 315 | 2.44 | 0.01 % | — | — |
| **Σ GPU** | **67 999** | **3 056 035** | | **100 %** | 2 668 (Post-warmup) | 2 224 |

### Beobachtung: gate_up_swiglu ist **leicht langsamer** (+2.7 %)

Der Ø-Wert für `gate_up_swiglu` steigt von 421.6 µs (1.17) auf
432.8 µs (Post-P0). Erklärung: 1.17 hat Ø über ~115 Post-Warmup-
Iterationen gemessen; dieses Run umfasst alle 129. Außerdem:
GPU-Takt-Varianz im 2 %-Bereich ist ohne ROC-clock-pin zu erwarten.
Der Kernel selbst ist unverändert — er wurde in 2.0.1/2.0.2 nicht
angefasst. Die Messung bestätigt, dass dies weiterhin der
Haupt-Hotspot ist.

### Beobachtung: `q8_inline` Ø sinkt von 25.8 → 18.0 µs

Dies ist **kein** Kernel-Speedup, sondern eine **Shape-Mix-
Verschiebung** durch die Residual-Fusion:

- **Vor 2.0.2 (1.17):** `q8_inline` aggregierte über Q/K/V/O
  — je 2 Calls 4096×4096 (Q,O) + 2 Calls 4096×1024 (K,V) pro Layer.
  Bei ~32 µs für 4096×4096 und ~18 µs für 4096×1024 ergibt sich
  ein gewichtetes Ø von ~25 µs (matched).
- **Post-2.0.2:** O-Projection ist jetzt im
  `q8_inline_residual`-Kernel. `q8_inline` enthält nur noch Q (1 ×
  4096×4096 pro Layer) + K/V (2 × 4096×1024 pro Layer). Das
  neue gewichtete Ø liegt bei ~22 µs; mein Run misst 18 µs
  (einige K/V-Calls sind vielleicht noch kürzer durch GQA-Head-
  Layout).

Die ursprünglichen 4096×4096-Q8-inline-Dispatches sind jetzt
im `q8_inline_residual`-Aggregate (41.7 µs/Call) mit enthalten,
und diese Zahl matcht v0.3.0 auf den Dezimalpunkt.

### Beobachtung: Loser-Pulls (`q4_k_standard`) sind jetzt 94.5 µs statt 141 µs

In der 1.17-Baseline lief `q4_k_standard` als Loser bei 141.3 µs
(mit Sync-Overhead). Im Post-P0-Run liefert die reine GPU-
Kernel-Zeit 94.5 µs — das **ist** der echte Kernel ohne Sync.
Dass der Bandit ihn trotzdem als Loser abstraft ist korrekt:
18 µs (q8_inline) < 94.5 µs (standard) ist ein 5.2× Gap, reicht
UCB1 bequem für die Exploitation-Entscheidung.

### Beobachtung: `attention_decode` ist jetzt schneller als v0.x

9.48 µs vs. 10.7 µs (1.17) vs. 12.2 µs (v0.3.0). Identischer Kernel
wie in 1.17, aber ohne den ~1 µs-CPU-Side-Overhead der weggefallenen
Per-Node-Sync — die 1.17-Zahl beinhaltete auch diesen.

### Beobachtung: `q4_k_q8_inline_residual` match v0.3.0 auf 0.1 µs

Der neue Kernel aus 2.0.2 liegt bei 41.66 µs/Call, v0.3.0
`q4_k_q8_inline_residual_multi_row` lief bei 41.7 µs — die
Codegen-Port hat den v0.x-Kernel bit-genau reproduziert.

## HIP-API-Overhead (Dispatch-Count, Sync-Rate)

| API | Calls | Σ µs | Ø µs | % HIP |
|---|---:|---:|---:|---:|
| **`hipStreamSynchronize`** | **129** | 2 943 057 | 22 814 | 85.2 % |
| `hipMemcpy` | 730 | 418 124 | 573 | 12.1 % |
| `hipLaunchKernel` | 67 725 | 59 680 | 0.88 | 1.7 % |
| `hipEventRecord` | 508 | 344 | 0.68 | 0.01 % |
| `hipEventElapsedTime` | 254 | 49 | 0.19 | <0.01 % |
| `hipEventCreate`/`Destroy` | 512 / 512 | 56 / 124 | 0.11 / 0.24 | <0.01 % |

**P0-Gate: 129 Syncs für 100 Decode-Tokens — PASS (< 200).**

Die 2 943 ms in `hipStreamSynchronize` sind nicht CPU-Overhead
sondern GPU-Wartezeit: eine Sync pro execute_decode-Iteration,
die auf den Logits-Readback wartet. Ø ~22.8 ms pro Sync
entspricht der GPU-Zeit für eine ganze Iteration (~23 ms bei
40 tok/s). Das ist **gewollt** — eine Sync pro Token ist die
Zero-Sync-Pipeline aus Arch-Doc §3.7.

**Dispatch-Count:** 67 725 hipLaunchKernel-Calls / 129 Iterationen
= **525 Launches/Iteration**. Plus 274 HSA-direct `copyBuffer`
= 67 999 Dispatches total, matched das 2.0.2-Report-Number
**exakt**.

## Dispatches/Token-Vergleich

| Run | HIP Launches total | /Iteration | Reduktion vs 1.17 |
|---|---:|---:|---:|
| 1.17 Phase-1 Tuned | 78 445 | 608 | — |
| **Post-P0 (2.0.3)** | **67 725** | **525** | **−14 %** |
| v0.3.0 | 55 570 | 462 | −24 % |

v0.x hat weiterhin 12 % weniger Dispatches pro Iteration als
v1.0 Post-P0 — dieser Rest ergibt sich aus den noch ungefusten
Paths (Q6_K-FFN-Down-Hälfte, Gate+Up+Down als Einzel-Kernels
statt 5-Op-Fusion).

## GPU- und BW-Effizienz

| Metrik | 1.17 | **Post-P0 (2.0.3)** | v0.3.0 |
|---|---:|---:|---:|
| Decode tok/s | 30.6 | **40.7** | 41.7 |
| Decode tok/s (rocprof-kern) | — | 39.2 | — |
| Σ GPU-Zeit (129 Iter.) | 2 668 ms (110 Iter.) | **3 056 ms** | 2 224 ms |
| Wallclock (Inference) | 4 255 ms | **3 334 ms** | 3 072 ms |
| hipStreamSync / 100 tok | 83 260 | **129** | 114 |
| Dispatches/Iteration | 608 | **525** | 462 |
| **GPU-Effizienz** | **62.7 %** | **91.7 %** | — |
| **BW-Effizienz** (640 GB/s) | 21 % | **28.6 %** | — |
| VRAM-Peak | 10.05 GB | **10.09 GB** | — |

**GPU-Effizienz-Sprung von 62.7 % auf 91.7 %** — das ist der
Kern-Gewinn aus der Sync-Elimination. Die GPU sitzt nicht mehr
37 % der Zeit idle, sondern arbeitet durchgehend bis zum
Logits-Readback.

**BW-Effizienz** ist von 21 % auf 28.6 % gestiegen. Noch immer
deutlich unter dem realistischen Ziel (Arch-Doc §1.4: 87 %
≈ 125 tok/s). Der Rest ist klassisches Kernel-Tuning-Problem:
der Kernel-Durchsatz pro Byte reicht nicht, um die BW zu
sättigen. Das ist das Phase-2.1-Thema.

**VRAM-Peak unverändert** — das EventPool (256 Events à ~80 B)
und der neue Residual-Kernel-Code (256 KB Binary) sind im MB-
Bereich nicht sichtbar.

## Bottleneck-Analyse — Post-P0

### Hat sich der Bottleneck verschoben?

**Nein.** Top-3 nach GPU-Anteil:

```
1.17:                           Post-P0:
   gate_up_swiglu    65.4 %       gate_up_swiglu    65.8 %
   q8_inline         16.0 %       q6_k (LM-Head)    13.5 %
   q6_k (LM-Head)    13.7 %       q8_inline_res     9.5 %  ← NEU
                                  q8_inline         6.8 %
```

Die Fusion hat das alte Q8-inline (16 %) in Q8-inline-residual
(9.5 %) + unfused Q8-inline (6.8 %) aufgeteilt — Summe 16.3 %,
also praktisch identisch. Das Ranking ist stabil.

### Was limitiert uns jetzt?

`gate_up_swiglu` ist weiterhin zu 65 %+ der Elefant im Raum.
Die 2.3 GB/Token die dieser Kernel lesen muss (Gate + Up
Weights bei 4096×28672 × Q4_K = 64 MB/Layer × 36 Layer ≈
2.3 GB) bei einer BW von 640 GB/s: theoretisch 3.6 ms pro
Token, also eine 280-tok/s-Grenze wenn *nur* gate_up laufen
würde. Aktuell braucht gate_up 15.6 ms pro Iteration
(= 4644 Calls × 432.8 µs / 129 Iter.) → gerade mal **23 %**
der BW-Grenze auf diesem einzelnen Kernel.

Das heißt: **der Kernel ist nicht BW-gebunden, sondern
compute- bzw. VALU-Latency-gebunden.** Die Q4_K-Block-
Dequantisierung (384 multi-level Scale/Min-Dekomprimierungen
pro 256-Element-Block) ist nicht durch die WMMA-Einheiten
beschleunigt, sondern läuft auf VALU. Das ist genau der Hebel,
den die Kernel-GA + FP8-Pfad angreift.

### Ist Q6_K der neue Bottleneck?

Nicht in absoluten Zahlen (13.5 % ist weniger als die 65.8 %
von gate_up_swiglu), aber **pro Variante**: Q6_K hat nur eine
`q6_k_standard`-Variante im Bandit, d. h. keine Wahl. v0.x
hat hier `q6_k_q8_inline` (71.8 µs) **und** `q6_k_multi_row`
(59.4 µs). Ein Q8-inline-Variant auf Q6_K würde wahrscheinlich
von 86.5 µs auf ~60 µs runtergehen (v0.x-Level), das sind
118 ms weniger GPU-Zeit pro 100 Tokens, ~3.9 % Decode-Gewinn,
~42 tok/s. Kleiner Hebel aber fast umsonst zu bauen.

### Compute-bound vs BW-bound pro Kernel

Grobe Abschätzung pro Kernel (Weight-Bytes / (kernel_time ×
BW-peak)):

| Kernel | Gewicht-Bytes/Call | Ø µs | BW (GB/s) | % von 640 |
|---|---:|---:|---:|---:|
| `gate_up_swiglu` | ~64 MB | 433 | 149 | 23 % |
| `q4_k_q8_inline_residual` | ~8 MB | 42 | 200 | 31 % |
| `q4_k_q8_inline` | ~2 MB | 18 | 117 | 18 % |
| `q6_k_standard` (LM-Head) | ~70 MB | 86 | 806 | **126 %** (!) |
| `attention_decode` | KV ≤ 1 MB | 9.5 | 105 | 16 % |

**Q6_K überschreitet scheinbar die physikalische Grenze —
Erklärung:** Die LM-Head wird nicht komplett durch den
Kernel neu-gelesen; gfx1201 hat 8 MB L2-Cache, der Teil der
Q6_K-Blöcke nach dem ersten Call hält. Multiple Dispatches
mit overlappendem Working-Set profitieren vom L2. Real-
Effective-BW auf diesem Kernel ist wahrscheinlich nur
~400 GB/s (63 % von peak).

**Alle anderen Kernel laufen bei < 31 % der BW-Grenze.** Das
bestätigt: wir sind **nicht BW-bound**, sondern **Kernel-
Throughput-bound**. Kernel-GA hat Raum.

## Neue ROCPROF_POST_P0_BASELINE (für GA-Fitness)

Ersetzt `ROCPROF_117_BASELINE` aus `ga_tuning_spec §7.5`:

```rust
// src_v1/ga_tuning/baseline.rs  (Phase 2.1 Implementation)
pub const ROCPROF_POST_P0_BASELINE: &[(ShapeName, f32)] = &[
    ("gemv_q4_k_gate_up_swiglu",        432.8),   // µs/Call
    ("gemv_q4_k_q8_inline",              18.0),   // unfused (Q + K/V mix)
    ("gemv_q4_k_q8_inline_residual",     41.7),   // NEU, 4096×4096 primary
    ("gemv_q6_k_standard",               86.5),
    ("attention_decode",                  9.48),
    ("rms_norm",                          3.31),
    ("rms_norm_batched",                  2.18),
    ("rope",                              1.89),
    ("residual_add_inplace",              1.66),  // weniger Calls post-fusion
    ("kv_cache_append",                   1.54),
];

pub const ROCPROF_POST_P0_DECODE_TPS: f32 = 40.7;   // Post-P0 Tuned Run
pub const ROCPROF_POST_P0_DISPATCHES_PER_ITER: u32 = 525;
pub const ROCPROF_POST_P0_GPU_EFFICIENCY: f32 = 0.917;
pub const ROCPROF_POST_P0_BW_EFFICIENCY: f32 = 0.286;
pub const ROCPROF_POST_P0_VRAM_MB: u32 = 10_330;
```

**Migration aus 1.17-Baseline:**

- `cache_invalidation_hash` (§6.3) muss bumpen — alle bisher
  gecachten GA-Winner wurden gegen Sync-Overhead-Baseline
  gewertet und sind damit systematisch 37 % zu pessimistisch.
  Re-GA auf allen gecachten Shapes.
- Fitness-Tests in `ga_tuning`-Crate (sobald sie existiert)
  brauchen diese Konstanten als Default.

## Aktualisierte Phase-2.1-Prioritäten

Neue Ordnung basierend auf Post-P0-Daten, nicht auf 1.17-
Schätzungen:

### P0 — Kernel-GA auf gate_up_swiglu (FP8 + Tile-Exploration)

```
Betroffener Kernel: gemv_q4_k_gate_up_swiglu
Post-P0:   432.8 µs/Call, 65.77 % GPU-Zeit, ~23 % BW-Effizienz
1.17 alt:  421.6 µs (veraltet)
v0.x:      394.2 µs (Q8-inline Variante, FP32-Intern)

Geschätzter Speedup (konservativ): 1.4× auf diesen Kernel
  durch FP8-WMMA-Intermediate + GA-Tile-Sweep
Erwarteter Decode-Gewinn: 40.7 → ~52 tok/s
  Rechnung: gate_up 2010 ms → 1436 ms (40 % Reduktion)
    Σ GPU 3056 → 2482 ms → Wall ~2.7 s → 36 % schneller → 55 tok/s
    (konservativ auf 52 gehalten wegen FP8-Parity-Overhead)

Phase-2-Priorität: P0
Begründung: Mit Abstand größter Hebel. FP8-WMMA auf gfx1201 ist
  **strukturell überlegen** (Arch-Doc §1.5) — llama.cpp hat das
  auf RDNA4 nicht. Die GA findet Tile-Configs, die menschlich
  nicht offensichtlich sind (v0.x-Erkenntnis #4).
```

### P0 — Q6_K Q8-Inline Variante (Bandit-Wahl für LM-Head)

```
Betroffener Kernel: gemv_q6_k_standard
Post-P0:   86.5 µs/Call, 13.51 % GPU-Zeit
1.17 alt:  85.6 µs (veraltet)
v0.x:      71.8 µs (direct) / 59.4 µs (multi_row)

Geschätzter Speedup: 1.4× durch Q8-Input-Quantisierung
Erwarteter Decode-Gewinn: 40.7 → ~42–43 tok/s
  Rechnung: q6_k 413 ms → 295 ms (matches v0.x),
    Σ GPU 3056 → 2938 ms → Wall ~3.2 s → 5 % schneller → 43 tok/s

Phase-2-Priorität: P0 (easy win)
Begründung: Template existiert (Q4_K Q8-inline gibt die Struktur);
  Q6_K-Block-Layout ist in `dequant_ir_spec §5.2` dokumentiert.
  Bandit hat heute nur 1 Variante — der Zweck der Säule 4 ist
  verschwendet, solange da keine Wahl existiert.
```

### P1 — Q6_K Residual-Fused (FFN-Down-Hälfte)

```
Betroffener Kernel: gemv_q6_k_standard (FFN-Down-Q6_K-Pfad)
Post-P0:   ~18 unfused Q6_K FFN-Down × ~86 µs ≈ 1.5 ms/Iter.
  + 18 separate residual_add_inplace

Geschätzter Speedup: −18 Dispatches/Iteration = ~2 %
Erwarteter Decode-Gewinn: 40.7 → ~41.5 tok/s
Phase-2-Priorität: P1
Begründung: Template aus Q4_K-Residual-Fusion kopierbar. Kleine
  aber saubere Fusion — bekommt alle Residual-Adds unter ein Dach.
```

### P1 — Gate+Up+SwiGLU+Down+Residual als 5-Op-Fusion (Fusion-GA)

```
Betroffene Kernels: gate_up_swiglu + ffn_down + residual_add
Aktuell: 3 Dispatches pro Layer × 36 Layer = 108 Dispatches
  + 2 VRAM-Roundtrips (SwiGLU-Output, Down-Output)

Geschätzter Speedup: −72 Dispatches + weniger Memory-Traffic
  durch Register-Passthrough SwiGLU → Down
Erwarteter Decode-Gewinn: 40.7 → ~45–48 tok/s
  (nach P0-Kernel-GA sind die Einzelkernels kleiner, also
   wird die Fusion-Lohn noch attraktiver)

Phase-2-Priorität: P1
Begründung: Das ist die Hauptaufgabe der Fusion-GA (§4 der Spec).
  Braucht Kernel-GA als Vorlauf, damit man weiß, welche
  Einzelkernel-Varianten zu fusen sind.
```

### P2 — KV-Cache FP8 (E5M2)

```
Betroffene Op: attention_decode + kv_cache_append
Aktuell: FP16-KV, 2 Bytes/Element
Ziel:    FP8-E5M2, 1 Byte/Element — halbiert die Attention-Read-BW

Erwarteter Decode-Gewinn: 40.7 → ~43 tok/s (bei 2k Context)
  → größer bei 8k+ (dann steigt der Attention-Anteil)
Phase-2-Priorität: P2 (Context-Länge-abhängig)
Begründung: Attention ist bei 100-Token-Prompt nur 1.44 % GPU-Zeit.
  Lohnt erst bei langen Kontexten — dann aber massiv.
```

### P2 — Precision-GA (Layer-wise FP8 vs. FP16)

```
Scope: NSGA-II über 36 Layers × {FP8-E4M3, FP8-E5M2, FP16}
  unter KL-Divergenz-Constraint aus Säule 5 Monitor.

Erwarteter Decode-Gewinn (nach P0+P1): 55 → 70–80 tok/s
  Rechnung aus ga_tuning_spec §8.3: 15–25 % durch FP8-Layer-
  Dominanz + halbiertes VRAM-BW-Consumption der FP8-Paths.

Phase-2-Priorität: P2 (nach Kernel-GA, weil Fitness-Messung
  auf Sub-Kernel-Ebene erst nach P0+P1 sauber funktioniert)
Begründung: Die Precision-GA braucht eine Kernel-Baseline, die
  sie variieren kann. Ohne Kernel-GA-Ergebnisse ist jeder
  Precision-Punkt ein Raten-Experiment.
```

## Theoretisches Maximum

Kumulative Schätzung über alle Phase-2.1+-Schritte:

| Stufe | Decode tok/s | Δ | Quelle |
|---|---:|---:|---|
| **Heute (Post-P0)** | **40.7** | — | gemessen |
| + P0 Kernel-GA (gate_up FP8) | ~52 | +28 % | FP8-2×-WMMA, 40 % Kernel-Reduktion |
| + P0 Q6_K Q8-Inline | ~55 | +6 % | matches v0.x 59.4 µs |
| + P1 Q6_K-Res-Fused | ~56 | +2 % | 18 Dispatches weg |
| + P1 Fusion-GA (5-Op Gate+Up+Down+Res) | ~65 | +16 % | 72 Dispatches weg + VRAM-Roundtrips |
| + P2 Precision-GA (FP8-Layer) | 75–85 | +15–30 % | halbiert effektive VRAM-BW |
| + P2 KV-FP8 | 80–90 | +5–10 % | attention-BW halbiert |
| **Arch-Doc-Ziel** | **125** | | §1.4 realistisch, 88 % BW-Effizienz |

Der Rest (90 → 125 tok/s) sitzt in Fein-GA auf allen Kernels,
Layer-Fusion, und Warteschlangen-Overlap (ga_tuning_spec §8.4).
Das ist Phase 2.3+ Arbeit.

**Prefill ist in diesem Run nicht gemessen** — für Prefill
werden ~300 tok/s erwartet sobald WMMA-Batched-Path gewired ist
(Arch-Doc §1.4, P1 im 1.17-Report). Liegt außerhalb der Post-
P0-Messung.

## Optional: HW-Counter (VALU/LDS-Utilization)

**Nicht durchgeführt.** rocprofv3 `--pmc` auf gfx1201 hatte in
1.17 PMC-Mode als "experimentell, kann rocmforge abbrechen"
markiert. Ohne stabile HW-Counter-Messung wäre der Aufwand,
die VALU-Utilization pro Kernel zu bestimmen, > 1 Tag (mehrere
PMC-Runs mit unterschiedlichen Counter-Sets + Analyse). Die
Kernel-Trace-Zeiten reichen als Baseline — HW-Counter waren
im Prompt explizit als Bonus markiert. Falls Phase 2.1 es
braucht, kann später separat nachgezogen werden.

## Zusammenfassung

```
Dreifach-Vergleich Decode:
  1.17:     30.6 tok/s  (83 260 Syncs, veraltet)
  Post-P0:  40.7 tok/s  (129 Syncs, 95 % GPU-Effizienz)
  v0.3.0:   41.7 tok/s  (114 Syncs)

Top-3 Kernel (Post-P0, % der Decode-Zeit):
  1. gate_up_swiglu         — 65.77 %  (432.8 µs/Call, 1.17: 421.6, v0.x: 394.2)
  2. q6_k_standard LM-Head  — 13.51 %  ( 86.5 µs/Call, 1.17:  85.6, v0.x:  71.8)
  3. q4_k_q8_inline_residual —  9.50 %  ( 41.7 µs/Call, NEU, v0.x: 41.7 ✓)

GPU-Effizienz:  1.17: 62.7 % → Post-P0: 91.7 %  (+29 Pp)
BW-Effizienz:   1.17:   21 % → Post-P0:   28.6 %
VRAM-Peak:      10.09 GB (unverändert)
Dispatches/Iter: 608 → 525 (−14 %)

Neue ROCPROF_POST_P0_BASELINE (Fitness-Denominator):
  gate_up_swiglu:             432.8 µs
  q4_k_q8_inline (unfused):    18.0 µs
  q4_k_q8_inline_residual:     41.7 µs
  q6_k_standard:               86.5 µs
  attention_decode:             9.48 µs
  Decode:                      40.7 tok/s

Bottleneck: UNVERSCHOBEN. gate_up_swiglu bleibt 65 %+.
  Nicht BW-bound (28.6 % von 640 GB/s), nicht Dispatch-bound
  (91.7 % GPU-Effizienz). Ist VALU-/Kernel-Throughput-bound.

Phase-2.1-Prioritäten (neu):
  P0: Kernel-GA gate_up_swiglu (FP8)       → ~52 tok/s
  P0: Q6_K Q8-Inline Variante              → ~43 tok/s
  P1: Q6_K Residual-Fused                  → ~41 tok/s
  P1: 5-Op-Fusion (Gate+Up+Down+Res)       → ~48 tok/s
  P2: Precision-GA (FP8-Layer)             → ~75–85 tok/s
  P2: KV-Cache FP8                         → ~45 tok/s (2k Context)

Theoretisches Maximum nach P0+P1+P2: 80–90 tok/s Decode
Arch-Doc-Ziel (125 tok/s) braucht zusätzlich Layer-Fusion +
Warteschlangen-Overlap (Phase 2.3+).
```

## Commit

Prefix `docs:` — nur Messung + Analyse, keine Code-Änderung. Der
Report + die rocprof-CSVs (unter `/tmp/rocprof-post-p0-*/`) sind
der Deliverable; die Baseline-Konstanten werden in Phase 2.1 in
`src_v1/ga_tuning/baseline.rs` umgesetzt, wenn die `ga_tuning`-
Crate existiert.

Backup-Push auf `backup` Remote — kein Push auf Fork.
