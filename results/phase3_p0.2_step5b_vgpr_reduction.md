# P0.2 MMQ-Port Schritt 5b — VGPR-Reduktion (negative Findung)

**Date:** 2026-04-25
**Branch:** v1.0-dev
**Status:** **VGPR-Reduktion auf 120 (von 152) ist möglich, aber netto LANGSAMER.** Strategie C (Akkumulator-Scoping) reduziert VGPR um 32, regrediert aber den Q4_K-Kernel um **+52 % Σ ms** und das E2E-Wallclock um **+18 %**. `__launch_bounds__(_,2)` als reiner Compiler-Hint pessimiert sogar (152→160). Der Kernel ist **WMMA-Pipeline-bound, nicht Occupancy-Latency-bound**: zusätzliche Waves pro CU bringen keinen Latency-Hiding-Vorteil weil die WMMA-Issue-Rate der Bottleneck ist. **Empfehlung: VGPR-Reduktion als Optimierungsweg verwerfen, beim Schritt-5-Stand (VGPR=152, 150 ms Q4_K) bleiben.**

## TL;DR

Zwei VGPR-Reduktions-Strategien getestet, beide netto-Verlust:

| Variante | VGPR | Q4_K Kernel Σ | E2E Wallclock |
|---|---:|---:|---:|
| **Schritt-5 Baseline** | **152** | **150.5 ms** | **459 ms** |
| __launch_bounds__(128, 2) | 160 (+8) | nicht gemessen | — |
| **Strategy C (acc_sb scoping)** | **120 (−32)** | **229.3 ms (+52 %)** | **540 ms (+18 %)** |

Strategy C trifft das 120-VGPR-Target sauber, aber die strukturellen Trade-Offs des Restrukturings (mehr Memory-Loads pro Sub-Block, mehr Branches im scale-fixup) kompensieren den Occupancy-Gewinn deutlich. Konsistent mit der Schritt-5-Erkenntnis dass der Kernel int-WMMA-Throughput-bound ist und Multi-Wave-Latency-Hiding nicht hilft.

## VGPR-Inventar (vor jeder Änderung)

Aus statischer Analyse von `mmq_q4_k.hip` (Schritt-5-Stand):

| Variable | Größe | Live-Range | VGPRs |
|---|---|---|---:|
| `acc_sb[8]` (`int32x8_t`) | 8 × 8 i32 | inner WMMA-loop bis scale-fixup | **64** |
| `c_acc[8]` (float) | 8 floats | outer K-loop (über alle super-blocks) | 8 |
| `d_B_arr[8]` + `sum_B_arr[8]` | 16 floats | super-block scale-fixup | 16 |
| `a_frag[2]` + `b_frag[2]` (`int32x2_t`) | 4 + 4 i32 | WMMA inner | 8 |
| Pointer (a_sb, b_lo, b_hi) | 6 × 64-bit | super-block | 12 |
| Per-slot temporaries | misc | scale-fixup-loop | ~10 |
| Loop-counters, broadcast SGPR-spilled | misc | various | ~10 |
| **Konservativ live max** | | | **~108** |
| **Compiler-gemessen (rocprof)** | | | **152** |

Compiler-Differenz von ~44 entspringt vor allem dem #pragma-unroll der 8-sb-Schleife, die acc_sb[8] live über alle Iterationen hält, plus der konservativen Spill-Avoidance-Heuristik des Backends.

**Größter Hebel laut Inventar: `acc_sb[8]` (64 VGPRs).** Falls Akkumulator-Scoping möglich ist (Strategie C), würde acc_sb[8] auf acc_one (8 VGPRs) reduziert — net −56 VGPRs.

## Strategie 1 — `__launch_bounds__(WARPS_PER_BLOCK*32, 2)`

### Was wurde gemacht

`__launch_bounds__(128, 2)` an die `rf_v1_mmq_q4_k_kernel`-Template-Definition angehängt, ohne Body-Änderungen.

### Resultat

VGPR **steigt** von 152 auf **160**.

### Erklärung

Der Compiler-Hint sagt "halte VGPRs niedrig genug für 2 Blocks/CU", aber die Constraint ist auf RDNA4 mit 128-Thread-Blocks und int32x8-Akkumulatoren strukturell nicht erreichbar ohne Scratch-Spilling. Stattdessen optimiert der Compiler das Layout um die Constraint zu respektieren, dabei werden interne Lifetimes vergrößert und mehr Lane-Register für Liveness-Buffer verbraucht. Net negativer Effekt.

`__launch_bounds__(128, 1)` wurde nicht getestet, da 1 Block/CU bei VGPR=152 ohnehin bereits erfüllt ist und der Hint daher wirkungslos wäre.

**Verwerfung:** Hint zurückgenommen.

## Strategie 2 — Strategie C: Akkumulator-Scoping (`acc_sb[8]` → `acc_one`)

### Was wurde gemacht

Die outer WMMA-Schleife (8 Sub-Blöcke × 2 WMMA-Calls je) und die scale-fixup-Schleife (8 Slots × 8 Sub-Blöcke) wurden gefused in eine sb-äußere Schleife. Pro sb-Iteration:

1. 2 WMMA-Calls in einem Single-Akkumulator `int32x8_t acc_one`.
2. Lese B-Scale (`d_B_sb`, `sum_B_sb`) inline.
3. Per-Slot scale-fixup nutzt `acc_one[slot]` direkt, akkumuliert in `c_acc[slot]`.

Pro super-block hinzugefügt: ein Vor-Cache von 8 × (d_A, dmin_A, sc_p[2], mn_p[2]) = 48 VGPRs (live über die 8 sb-Iterationen).

### Resultat

| Metrik | Schritt 5 | Strategy C | Δ |
|---|---:|---:|---:|
| **VGPR** | 152 | **120** | **−32** ✓ (am Target) |
| **Scratch** | 0 | 0 | (kein Spilling) |
| **mmq_q4_k Σ ms (216 calls)** | 150.5 | **229.3** | **+52 %** ⚠ |
| **mmq_q4_k Avg µs/call** | 696 | 1061 | +52 % |
| **E2E Wallclock (542 tok prefill)** | 459 ms | **540 ms** | **+18 %** ⚠ |
| **E2E 4W vs 1W speedup-sweep min** | 0.96× | **0.89×** | regressiert |
| **Parity 4W vs 1W (256k Elemente)** | 0 differing | **0 differing** | unverändert ✓ |

### Erklärung

VGPR-Sparen via `acc_one`-Reuse spart 56 VGPRs am Akkumulator. Die zusätzlichen Per-Slot-Caches kosten ~24 VGPRs. Compiler-net: −32 VGPRs.

Aber zwei strukturelle Kosten:

1. **Mehr Memory-Loads.** B-Scale wird jetzt pro sb gelesen (8× pro super-block, kein Vor-Cache in d_B_arr/sum_B_arr). Bei int-WMMA-Pipelined-Workload zählt jede zusätzliche Load.

2. **Mehr Branches im scale-fixup.** Das `if (n >= N) continue;` wird nun pro sb-slot-Paar evaluiert (8× mehr als zuvor pro super-block). Selbst wenn der Compiler das hoistet, vergrößert es die Code-Größe und I-Cache-Pressure.

Gemessen sind beides netto +52 % Per-Kernel-Zeit.

### Occupancy-Analyse (warum hilft 120 VGPR nicht?)

Auf RDNA4 (gfx1201) ist die Auslegung pro CU/WGP:
- 4 SIMD32 Units, je 1536 VGPR-Slots → 6144 VGPR-Slots/CU
- 1 Block (128 Threads = 4 Waves) braucht: 128 × VGPR-pro-Lane

| VGPR/Lane | Block-VGPR | Blocks/CU (theor.) | Waves/CU (theor.) |
|---:|---:|---:|---:|
| 152 | 19456 | 0.32 | 1.26 |
| 120 | 15360 | 0.40 | 1.60 |

Beide Setups landen bei **<2 Waves/CU effektiv**. Auf RDNA4 ist die WMMA-Pipeline aber pro SIMD nur 1× pro Cycle issue-fähig. Bei 1.26 vs 1.60 Waves/CU gibt es **kein nennenswertes Latency-Hiding-Potential** — beide Konfigurationen senden ständig MMA-Issues. Die theoretische Mehr-Occupancy von Strategy C wird nicht in Performance umgesetzt.

Konsistent mit der Schritt-5-Erkenntnis: Multi-Warp-Scale-Up (1W → 4W) brachte ähnlich neutrale Resultate aus dem gleichen Grund.

## Was wäre ein NETTO-Gewinn-Pfad?

Die VGPR-Diagnose hat keinen Hebel gefunden. Stattdessen die strukturellen Erkenntnisse aus 5+5b:

1. **Weight-LDS-Sharing (Schritt 5c).** Die 4 Warps eines Blocks lesen aktuell die SAME Q4_K-Bytes aus L2/L1 unabhängig. Wenn 1 Warp lädt → __syncthreads → alle 4 nutzen, spart 3/4 der Q4_K-Bandbreite. Da der Kernel ist Compute-Bound, nicht Bandbreite-Bound, ist der erwartete Gewinn klein (~5 %), aber positiv. Implementierung: ~2 KB shared LDS pro Block, einfach.

2. **Q6_K MMQ-Port (Schritt 6).** 14.7 % der Prefill-Zeit ist immer noch Q6_K (LM-Head). Selbe Integer-WMMA-Infra wie Q4_K, aber 6-bit-Quantisierung (ql + qh splits). Erwarteter Gewinn: ähnlich Q4_K-Schritt 4 (-37 % Q6_K-Kernel-Zeit → ~5 % E2E).

3. **Asynchrone DMA-Pipelining (Schritt 7).** RDNA4 hat asynchrone Memory-Engines die parallel zur Compute-Pipeline laufen können. Aktuell sind alle Loads-Compute-Stores in einer streng sequenziellen Pipeline. Mit prefetch (`__pipeline_memcpy_async`) könnten Q4_K-Bytes für sb+1 geladen werden während sb noch im WMMA ist.

4. **Q8_1-Persistent-Buffer.** Schritt 4 hat einen `mmq_activation_buffer` eingeführt, der pro Prefill-Call **lazy** allokiert wird. Falls ein langer Run mehrere Prefills macht (multi-turn-Chat), wäre eine Pre-Allocation am Konstruktor die ~9 MB Allokations-Latenz vermeiden.

## Tests

`tests_v1/mmq_multiwarp_test.rs` bleibt grün (Strategy C reverted, Schritt-5-Stand restored):

```
test mmq_4w_boundary_m_not_multiple_of_64 ... ok  (max_abs 26 < tol 1266)
test mmq_4w_vs_1w_parity_small ... ok  (0 differing elems)
test mmq_4w_vs_1w_parity_large_k ... ok  (0 differing elems)
test mmq_4w_vs_1w_parity_qwen3_qkv ... ok  (0/262144 differing)
test mmq_4w_vs_cpu_reference ... ok  (max_abs 73 < tol 6809)
test mmq_4w_faster_than_1w ... ok  (perf-sweep 0.96-1.05×, neutral)

test result: ok. 6 passed; 0 failed
```

E2E Mutex-Prompt: wort-identisch zur Schritt-5-Baseline.

## Geänderte Dateien

| Datei | Änderung | Status |
|---|---|---|
| `hip_kernels_v1/wmma/mmq_q4_k.hip` | Strategie C ausprobiert, dann revertiert | **wieder Schritt-5-Stand** |
| `tests_v1/mmq_multiwarp_test.rs` | (unverändert) | grün |

**Keine Code-Änderungen am Ende dieser Session.** Die Erkenntnis ist die Erkenntnis.

## Fazit

VGPR-Reduktion bei diesem MMQ-Q4_K-Kernel ist machbar (152 → 120) aber **bringt keinen Performance-Gewinn**, weil der Kernel WMMA-Issue-Rate-bound ist und nicht Latency-bound. Der Optimierungs-Hebel liegt nicht bei Occupancy/Register-Pressure, sondern bei:
- Weight-Sharing zwischen Warps (LDS-Sharing)
- Q6_K-Port (anderer Kernel, andere 14.7 % Prefill)
- Memory-Pipeline-Optimierung (Async-DMA)

VGPR=152 bleibt der akzeptierte Stand. **5b ist eine negative Findung mit klarem Lehrsatz: VGPR-Count alleine ist kein Performance-Indikator wenn der Kernel nicht Occupancy-bound ist.**

## Status P0.2

| Schritt | Status | Effekt |
|---|:---:|---|
| 1: Integer-WMMA + block_q8_1_mmq Infra | ✅ | Foundation |
| 2: Minimal-Kernel | ✅ | Math-Korrektheit |
| 3: Scale-Up | ✅ | Variable-Shape |
| 4: Executor-Integration | ✅ | **+28.7 % Prefill** |
| 5: Multi-Warp Scale-Up | ✅ | +1.5 % (Noise) |
| 5b: VGPR-Reduktion | **negative Findung** | reverted |
| 5c: Weight-LDS-Sharing | 🔜 | erwartet ~5 % |
| 6: Q6_K Port | 🔜 | erwartet ~5 % |
