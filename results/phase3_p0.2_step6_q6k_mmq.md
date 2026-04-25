# P0.2 MMQ-Port Schritt 6 — Q6_K Integer-WMMA

**Date:** 2026-04-25
**Branch:** v1.0-dev
**Status:** **Q6_K MMQ-Kernel landet (Korrektheit gesichert), aber als OPT-IN, nicht Default.** Per-kernel **+9.5 % langsamer** als FP16-WMMA Q6_K (72.6 + 0.75 quantize = 73.3 ms vs 67.0 ms), E2E-Wallclock **+1 % (466 → 471 ms)**. Korrektheit perfekt: 4W vs 1W bit-identisch über alle Shapes, vs CPU FP32 innerhalb FP16-Akkumulationsnoise, vs FP16-WMMA innerhalb Tol. **Empfehlung:** Q6_K-MMQ-Pfad bleibt verfügbar via `ROCMFORGE_PREFILL_MMQ_Q6K=1` für künftige LDS-Staging-Optimierung (Schritt 6b), aber Default ist FP16-WMMA Q6_K.

## TL;DR

Der Q6_K-Port von Q4_K-MMQ ist ein direkter Sibling: gleicher Integer-WMMA-Kernel-Stack, gleiche Fragment-Layouts, nur load_tiles_q6_K und ein simpler Scale-Fixup ohne Min-Term ändern sich. Die Math ist nachweislich korrekt (4 Tests, alle bit-exakt 4W vs 1W, 0 differierende Elemente). **Aber die Performance ist negativ:** der Q6_K-MMQ-Kernel ist auf RDNA4 (gfx1201) langsamer als die existierende FP16-WMMA-Q6_K-Implementierung mit LDS-Staging. Q6_K-MMQ wird daher als opt-in deployt, nicht als Default-Pfad.

## Was landet

### Neue Datei: `hip_kernels_v1/wmma/mmq_q6_k.hip` (~250 LOC)

Direkter Sibling von `mmq_q4_k.hip`:

| Aspekt | Q4_K MMQ | Q6_K MMQ |
|---|---|---|
| Sub-Block-Größe | 32 K-Elem | 16 K-Elem |
| Sub-Blöcke pro Super-Block | 8 | 16 |
| WMMA-Calls pro Sub-Block | 2 (K=16+16) | 1 (K=16) |
| **WMMA-Calls pro Super-Block** | **16** | **16** (gleich) |
| Akkumulator-Stack | `int32x8_t acc_sb[8]` (64 VGPRs) | `int32x8_t acc_sb[16]` (128 VGPRs) |
| Dequant | Nibble (4-bit) | ql + qh combine (6-bit, signed -32..31) |
| Scale-Fixup | `d*sc*d_B*raw - dmin*mn*sum_B` | `d*sc*d_B*raw` (kein min!) |
| Activation-Quantize | `block_q8_1_mmq` (DS4) | gleicher Quantizer |

Template-Pattern: `template<int WARPS_PER_BLOCK>` mit zwei extern-"C"-Entries:
- `rocmforge_launch_mmq_q6_k`     (4W, production)
- `rocmforge_launch_mmq_q6_k_1w`  (1W, parity reference)

### Executor-Integration

`src_v1/graph/executor.rs` neuer Branch:

```rust
let mmq_q6k_enabled = std::env::var("ROCMFORGE_PREFILL_MMQ_Q6K")
    .ok().as_deref() == Some("1");
let mmq_eligible = match weight.format {
    GgmlType::Q4_K => true,                 // immer wenn PREFILL_MMQ=1
    GgmlType::Q6_K => mmq_q6k_enabled,      // opt-in
    _ => false,
};
```

**Begründung Q6_K als opt-in:** die per-kernel Messungen zeigen +9.5 % gegenüber FP16-WMMA (67.0 → 73.3 ms). Default-Aktivierung wäre eine Regression.

### Tests

`tests_v1/mmq_q6_k_test.rs` (4 Tests, alle grün):

```
test mmq_q6_k_4w_vs_1w_parity ...
  Q6_K 4W vs 1W 16×16×256:  max_abs=0.0e0 differing=0/256
  Q6_K 4W vs 1W 64×64×256:  max_abs=0.0e0 differing=0/4096
  Q6_K 4W vs 1W 64×64×1024: max_abs=0.0e0 differing=0/4096
test mmq_q6_k_small_vs_cpu ...
  16×16×256: max_abs=4.77 max_mag=1537 tol=123
test mmq_q6_k_vs_cpu_larger ...
  16×16×1024: max_abs=11.2 max_mag=2620 tol=419
  32×32×256:  max_abs=5.81 max_mag=1582 tol=127
  64×64×256:  max_abs=7.90 max_mag=1907 tol=153
  64×64×1024: max_abs=11.6 max_mag=3164 tol=506
test mmq_q6_k_vs_fp16_wmma ...
  64×64×1024: max_abs=13.7 max_mag=3534 tol=565

test result: ok. 4 passed; 0 failed
```

Alle innerhalb FP16-Akkumulationstoleranz `(max_mag + 1e-3) * sqrt(K) * 5e-3`. Bit-exaktes 4W vs 1W bestätigt korrekte Warp-Skalierung.

## Performance — rocprof-Vergleich (542-Token-Prompt, Qwen3-8B, M=576 padded)

| Kernel | Calls | Σ ms | Avg µs | VGPR | Bemerkung |
|---|---:|---:|---:|---:|---|
| **A) Q4_K MMQ + Q6_K FP16-WMMA (Default)** | | | | | |
| `rf_v1_mmq_q4_k_kernel<4>`  | 216 | 150.07 | 694 | 152 | unverändert vs Schritt 5 |
| `rf_v1_wmma_gemm_q6_k_fp16` | 37 | **66.97** | 1810 | 88 | LDS-staged |
| `rf_v1_quantize_q8_1_mmq`   | 216 | 2.86 | 13 | 24 | nur Q4_K-Activations |
| **Σ Q6_K-Pfad** | | **66.97** | | | |
| **B) Q4_K MMQ + Q6_K MMQ (opt-in)** | | | | | |
| `rf_v1_mmq_q4_k_kernel<4>`  | 216 | 149.69 | 693 | 152 | unverändert |
| `rf_v1_mmq_q6_k_kernel<4>`  | 37 | **72.58** | 1962 | **184** | +9.5 % vs FP16 |
| `rf_v1_quantize_q8_1_mmq`   | 253 | 3.61 | 14 | 24 | +37 für Q6_K-Activations |
| **Σ Q6_K-Pfad** | | **73.33** (72.58 + 0.75 add'l quantize) | | | |

**Q6_K-Pfad-Differenz: +6.4 ms (+9.5 %)** zugunsten des FP16-WMMA-Pfades.

### E2E Wallclock (542-Token-Prompt, --max-tokens 1, 3 Runs Median)

| Variante | Wallclock ms | Decode tok/s |
|---|---:|---:|
| Schritt 4 (1W-Q4_K MMQ + Q6_K FP16) | ~466 | 26.9 |
| Schritt 5 (4W-Q4_K MMQ + Q6_K FP16) | 459 | 26.7 |
| **A) Default (4W-Q4_K MMQ + Q6_K FP16)** | **466–467** | 26.8 |
| **B) Opt-in (4W-Q4_K MMQ + 4W-Q6_K MMQ)** | 470–474 | 26.6–27.1 |

A vs B: **+4-7 ms** zugunsten von A (= ohne Q6_K MMQ). **Decode unverändert.**

### Mutex-Prompt Kohärenz

Beide Pfade liefern wort-identische Outputs:

> "A mutex, short for 'mutual exclusion,' is a synchronization mechanism used in multithreaded programming to ensure that only one thread can access a shared resource or critical section of code at any given time, thereby enforcing **mutual exclusion** and preventing race conditions. When a..."

## Warum ist Q6_K MMQ langsamer?

Vier Faktoren, geordnet nach Wirkungsanteil:

1. **VGPR-Druck (184 vs 88).** Der MMQ-Akkumulator-Stack `acc_sb[16]` (= 16 × 8 i32 = 128 VGPRs) ist doppelt so groß wie der Q4_K-Stack (`acc_sb[8]`, 64 VGPRs). Der Compiler landet bei 184 VGPR/Lane vs 88 für FP16-WMMA. Auf RDNA4 (1536 VGPRs/SIMD) heißt das ~2× weniger Block-Occupancy für MMQ.

2. **FP16-WMMA hat LDS-Staging.** Der existierende Q6_K-FP16-Kernel dequantisiert ql+qh erst in shared LDS (8 KB total LDS budget) und liest dann aus LDS für die WMMA-Aufrufe. Diese Dequant-Amortisation über den `K_CHUNK=32` ist das, was Q4_K-FP16-WMMA auf 244 ms brachte (dort overhead). Q6_K-FP16-WMMA hat denselben Mechanismus, profitiert aber bei Q6_K SCHIENBAR mehr — die kompliziertere ql+qh-Dequant amortisiert sich über das LDS-Sharing zwischen den 4 Warps. MMQ liest direkt aus L2/L1 ohne Sharing → mehr ALU pro K-Element.

3. **Q6_K-Dequant ist teurer per K-Element.** Pro Element: ql-byte-load + qh-byte-load + 2× shift + or + sub. Q4_K: ql-byte-load + 1 shift/and. Doppelt so viele ALU-Ops pro K-Element. In FP16-WMMA wird das einmal pro K_CHUNK gemacht (durch LDS-Staging), in MMQ pro WMMA-Call.

4. **Doppelte Subblock-Anzahl.** 16 sb × 1 WMMA-call = 16 Calls pro Super-Block (gleich wie Q4_K's 8 × 2). Aber: pro sb gibt es Setup (pair_base, qh_shift, b_sb_local-Berechnung, B-fragment-load). 16 Setups vs 8 Setups → mehr Overhead-Zyklen.

## Konsistenz mit Schritt 5b

Schritt 5b zeigte: VGPR-Reduktion bei Q4_K-MMQ bringt nichts (Kernel ist WMMA-Issue-bound). Schritt 6 zeigt das Spiegelbild: bei Q6_K ist die FP16-WMMA-Variante BESSER, weil dort LDS-Staging die Dequant-ALU amortisiert. **Q4_K's MMQ-Sieg (-37 %) liegt nicht primär an Integer-WMMA, sondern an der einfacheren Q4_K-Dequant** (1 Nibble vs 2-Bit-Combine). Bei komplexerer Dequant lohnt sich LDS-Staging mehr als Integer-WMMA-Throughput.

## Was wäre der NETTO-Gewinn-Pfad für Q6_K?

**Schritt 6b: Q6_K MMQ + Weight-LDS-Staging.**

Der existierende Q6_K-FP16-WMMA-Kernel zeigt das Pattern:
1. Pre-dequant in `__shared__ lds_b[K_CHUNK * TILE_N]`
2. WMMA aus LDS

Eine MMQ-Variante mit LDS-Staging:
1. Pre-dequant ql+qh → signed int8 in `__shared__ lds_w[K_CHUNK * TILE_N]` (1 Byte/Element statt FP16's 2 Byte → 4 KB statt 8 KB LDS)
2. Integer-WMMA aus LDS

Erwarteter Gewinn:
- Dequant 1× pro K_CHUNK über alle 4 Warps geteilt (4× weniger ALU)
- Integer-WMMA-Pipeline weiter genutzt (potential Q4_K-mäßiger Throughput)
- Gleichzeitig LDS-Pressure ist halbiert vs FP16 (kann Occupancy verbessern)

Aufwand: ~1 Session, ähnlich zu Schritt 4. Nicht trivial weil das LDS-Layout für Integer-WMMA anders ist als für FP16-WMMA.

## Geänderte Dateien

| Datei | Änderung |
|---|---|
| `hip_kernels_v1/wmma/mmq_q6_k.hip` | **neu**, ~250 LOC, template<WARPS_PER_BLOCK> |
| `hip_kernels_v1/CMakeLists.txt` | `add_hip_kernel(v1_mmq_q6_k ...)` |
| `build.rs` | Library-Linker-Eintrag |
| `src_v1/backend/gpu/wmma.rs` | FFI-Deklarationen `rocmforge_launch_mmq_q6_k(_1w)` |
| `src_v1/graph/executor.rs` | Q6_K-Branch im MMQ-Pfad, opt-in via `ROCMFORGE_PREFILL_MMQ_Q6K=1` |
| `tests_v1/mmq_q6_k_test.rs` | **neu**, 4 Korrektheitstests |
| `Cargo.toml` | Test-Registrierung |

## Status P0.2

| Schritt | Status | Effekt |
|---|:---:|---|
| 1: Integer-WMMA + block_q8_1_mmq Infra | ✅ | Foundation |
| 2: Minimal-Kernel | ✅ | Math |
| 3: Scale-Up | ✅ | Variable Shape |
| 4: Executor-Integration (Q4_K) | ✅ | **+28.7 % Prefill** |
| 5: Multi-Warp Scale-Up | ✅ | +1.5 % (Noise) |
| 5b: VGPR-Reduktion | negative | reverted |
| **6: Q6_K Port** | **negative** | **reverted to opt-in only** |
| 6b: Q6_K + LDS-Staging | 🔜 | erwartet ähnlich Q4_K |
| 7: Async-DMA Pipelining | 🔜 | Idee |

## Fazit

Q6_K-MMQ-Kernel ist **mathematisch korrekt** und produktionsreif (Tests grün, bit-exakt 4W vs 1W über alle Shapes), aber auf RDNA4 ohne LDS-Staging **netto langsamer** als die existierende FP16-WMMA-Q6_K-Implementierung. Das ist die **zweite negative Findung** in P0.2 (nach 5b VGPR-Reduktion), und beide deuten auf dasselbe strukturelle Limit: **MMQ alleine ist kein universeller Sieg auf RDNA4**, sondern profitiert nur dann wenn die Dequant einfach ist (Q4_K) und LDS-Staging keinen großen zusätzlichen Wert bringt. Bei kompliziertem Dequant (Q6_K) ist LDS-Staging mit FP16-WMMA effizienter als naive Integer-WMMA ohne Sharing.

**Q4_K MMQ bleibt der größte P0.2-Sieg (+28.7 % E2E).** Q6_K-MMQ-Code wird als opt-in im Codebase gehalten als Vorbereitung für Schritt 6b (MMQ + LDS-Staging), das voraussichtlich Q4_K-ähnliche Gewinne (+5-8 % E2E) bringen würde.
