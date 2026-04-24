# P0.2 — llama.cpp MMQ Prefill Analyse

**Date:** 2026-04-24
**Branch:** v1.0-dev
**Hardware:** AMD Radeon RX 9070 XT (gfx1201, RDNA4)
**Scope:** **Phase 1+2 (Analyse + Diff) komplett. Phase 3 (Port) nicht in dieser Session.**

---

## TL;DR

llama.cpp's Prefill auf RDNA4 nutzt **Integer WMMA** (`__builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12`), nicht FP16 WMMA wie wir. Das ist ein **fundamental anderer Compute-Pfad** mit strukturellen Vorteilen: Q4_K-Weights werden als int8 in LDS gehalten (halbierter Footprint), Dequant-Prolog macht nur Bit-Shift (statt FP-Multiply), und Scales werden einmal nach der MMA angewendet (statt pro Element). **Ein Port lohnt sich** — aber die Implementierung ist ~500-800 LOC und braucht vorgelagerte Infrastruktur (`block_q8_1_mmq`-Activation-Format, integer WMMA-Intrinsic-Bindings), also klar multi-Session-Aufwand.

**Empfehlung: Option A (1:1 Port), separater Session für die Implementierung.** Diese Session liefert die komplette Analyse + Port-Roadmap.

---

## Schlüsselfrage beantwortet: WMMA vs dp4a

Die kritische Frage aus dem Prompt war: „Nutzt llama.cpp auf RDNA4 WMMA oder dp4a für Prefill?"

**Antwort: WMMA, aber Integer-WMMA** (nicht FP16).

Evidenz aus `~/tmp/llama.cpp/ggml/src/ggml-cuda/mmq.cuh`:

```cpp
// mmq.cuh:3568-3574 — main mmq kernel dispatches based on platform
#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    constexpr vec_dot_mmq_t vec_dot = mmq_type_traits<mmq_x, mmq_y, need_check, type>::vec_dot_mma;
#else
    constexpr vec_dot_mmq_t vec_dot = mmq_type_traits<mmq_x, mmq_y, need_check, type>::vec_dot_dp4a;
#endif
```

Auf RDNA4 ist `AMD_WMMA_AVAILABLE` definiert (`common.cuh:243-244: #if defined(GGML_USE_HIP) && (defined(RDNA4) || defined(RDNA3))`), also wird der MMA-Pfad gewählt.

Evidenz für Q4_K spezifisch aus `mmq.cuh:3464-3468`:

```cpp
template <int mmq_x, int mmq_y, bool need_check>
struct mmq_type_traits<mmq_x, mmq_y, need_check, GGML_TYPE_Q4_K> {
    static constexpr vec_dot_mmq_t vec_dot_mma  = vec_dot_q8_1_q8_1_mma<mmq_x, mmq_y>;
    //                                         ^^^^^^^^^^^^^^^^^^^^^
    //                                         Q4_K → Q8_1 → integer MMA
};
```

Der eigentliche Intrinsic steht in `mma.cuh:1246-1253`:

```cpp
#if defined(RDNA4)
    acc[0] = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(
        true,       // a_signed
        a_vec[0],   // int32x2_t = 8 packed int8
        true,       // b_signed
        b_vec[0],   // int32x2_t = 8 packed int8
        acc[0],     // int32x8_t accumulator
        true        // clamp
    );
#endif
```

→ **Integer WMMA, int8 × int8 → int32.** Auf RDNA4 ist dieser Intrinsic laut AMD-Doku äquivalent in Rate zu `v_wmma_f32_16x16x16_f16` (2 WMMA/CU/Takt), aber mit halbem Speicher-Footprint für Inputs.

---

## Vollständige Diff-Tabelle

| Dimension | llama.cpp MMQ (RDNA4) | Unser WMMA-Prefill | Implikation |
|---|---|---|---|
| **Compute-Intrinsic** | `wmma_i32_16x16x16_iu8_w32_gfx12` | `wmma_f32_16x16x16_f16_w32_gfx12` | Gleiche Rate, aber int8-Inputs halbieren LDS-Footprint |
| **Weight-Dequant-Timing** | In `load_tiles_q4_K`: Nibble-Shift + Pack zu int8 in LDS | Im K-Loop-Prolog: FP-Multiply + Subtract → FP16 in LDS | llama.cpp spart Dequant-VALU-Zyklen pro Iteration |
| **Weight-Storage-Format in LDS** | `int32` buffer (4 int8 packed), 1 B/Elem | FP16 (`__half`), 2 B/Elem | LDS halbiert |
| **Aktivierung-Storage in LDS** | `int32` (4 int8 packed), 1 B/Elem | FP16, 2 B/Elem | LDS halbiert |
| **Aktivierung-Source** | `block_q8_1_mmq` (pre-quantisiert CPU→GPU) | FP32 Global, FP16 im Kernel | Pre-Quantize-Pass nötig |
| **Scale-Application-Stelle** | Nach der Integer-MMA, einmal pro 32-Elem-Block (`sum += dmA * dsB * C.x[l]`) | Pro Element im Dequant-Prolog | Riesige Reduktion der Scale-Operationen |
| **MMA-Akkumulator-Typ** | int32 (Scale-Fixup zu float am Ende) | float32 | Gleiche Accum-Genauigkeit am Ende |
| **TILE_M (mmq_y)** | 128 (fest für RDNA2+) | 64 (aktueller Default) | llama.cpp nutzt größere Tiles |
| **TILE_N (mmq_x)** | 8…128 adaptiv über `mmq_x_best` Heuristik | 64 statisch | Adaptive Grid-Größe |
| **K-Iterations-Einheit** | `MMQ_ITER_K = 256` (= ein ganzer Q4_K Super-Block) | `K_CHUNK = 32` (= 1/8 Super-Block) | llama.cpp macht größere K-Schritte |
| **Warps/Block** | 8 (`MMQ_NWARPS = 8`) | 4 | 2× Parallelität innerhalb des Blocks |
| **Double-Buffering** | Ja (zwei `by0` Loads pro K-Iter, line 3590-3622) | Nein | Memory-Latency-Hiding |
| **Grid-Heuristik** | Adaptiv per Call (`mmq_x_best` in mmq.cu:4175-4189) | Statisch | Bessere Sättigung bei variabler Shape |
| **Work-Partitioning** | "Stream-K" (https://arxiv.org/abs/2301.03598) für Tile-Load-Balance | Naive (1 Tile = 1 Block) | Bessere Tail-Cleanup bei schmalen GEMMs |

### `block_q8_1_mmq` — Aktivierungs-Pre-Quantize

llama.cpp quantisiert FP32-Aktivierungen **vor** der MMQ-Dispatch in einem eigenen Pre-Pass in `block_q8_1_mmq` Format (mmq.cuh:28-47):

```cpp
struct block_q8_1_mmq {
    union {
        float d4[4];     // 1 FP32-Scale pro 32 Elemente
        half2 ds4[4];    // 1 FP16-Scale + 1 Partial-Sum pro 32 Elemente (Q4_K-Pfad)
        half  d2s6[8];   // andere Layouts für andere Formate
    };
    int8_t qs[4*QK8_1];   // 128 int8 Werte
};
// Size: 4*32 + 4*2*2 = 144 B pro 128-Element-Block
```

→ 128 Elemente pro Block, 4 Scales (einer pro 32-Elem-Subblock). Das ist ähnlich zu Q8_1 (unser MMVQ-Format) aber mit 4-fach größerer Block-Struktur.

**Wir haben aktuell `block_q8_1`** (QK8_1 = 32 Elemente, kein mmq-Layout). Die `block_q8_1_mmq`-Variante würde ein zusätzlicher Quantize-Kernel sein oder eine Erweiterung des bestehenden `rocmforge_quantize_q8_1`.

---

## Warum unsere 8.7 % Compute-Auslastung

Mit der Analyse oben lassen sich die Hauptursachen benennen:

1. **Dequant-Prolog dominiert den Inner-Loop.** Bei jedem K_CHUNK=32 machen wir ~32 × (FP-Multiply + Subtract + Half-Convert) = ~100 VALU-Ops, plus eine einzige WMMA. Der WMMA-Teil ist ~1 Takt, der Dequant-Prolog viele mehr. **Das Compute-Verhältnis Dequant:MMA liegt bei ~100:1 statt der idealen 0:1.**

2. **Kleinerer Tile (64×64) = mehr Kernel-Launches.** Bei M=576 × N=4096 launchen wir 64*9 = 576 Blocks; llama.cpp würde 32*4 = 128 Blocks launchen (bei TM=TN=128). Weniger Kernel-Launch-Overhead.

3. **Kein Double-Buffering.** Während wir eine Kachel per WMMA verarbeiten, können wir nicht die nächste laden. llama.cpp überlappt Load und Compute.

4. **Scales pro Element statt pro Block.** Wir skalieren bei jedem der 256 Elemente pro Q4_K-Block; llama.cpp skaliert einmal pro 32-Elem-Gruppe (8× weniger Scale-Ops).

Insgesamt: Die MMA-Hardware ist nicht ausgelastet weil die VALU-Scheduler-Slots mit Dequant-Arbeit belegt sind.

---

## Portierungs-Entscheidung: Option A

| Option | Aufwand | Erwarteter Gewinn | Risiko | Entscheidung |
|---|---|---|---|---|
| **A: Kompletter 1:1 MMQ-Port** | ~500-800 LOC, 2-3 Sessions | +15-30 % Prefill (1000 → 1150-1300 tok/s) | Niedrig (bewiesener Code) | ✅ **Empfohlen** |
| B: Unseren WMMA anpassen (Int8 + Scale-Fixup umbauen) | ~300 LOC, 1-2 Sessions | +10-20 % | Mittel (Kombi ungetestet) | ❌ Zu nah an A mit mehr Risiko |
| C: STOP, Gap liegt woanders | 0 LOC | 0 % | Niedrig | ❌ Analyse zeigt klare Kernel-Unterschiede |

### Warum Option A, trotz nur ~15-30 % erwartetem Gewinn

Der Phase-3-Projektplan zielt auf ~5000 tok/s Prefill. Ein einzelner MMQ-Port wird das nicht schaffen (~1300 tok/s maximal). Aber der Port legt **die Infrastruktur frei** für:
- Weitere Optimierungen auf dem integer-Pfad (schnelleres `block_q8_1_mmq` über native RDNA4 int4→int8 Intrinsics, falls existent)
- Attention-Kernel-Port (P0.3 — llama.cpp's `fattn-wmma-f16.cu` nutzt ähnliche Infrastruktur)
- FP8-Re-Evaluation (P0.4) auf dem neuen Pfad

Option B würde genau die Infrastruktur-Abschlussarbeit vermeiden und den Weg zu weiteren Fortschritten versperren.

Option C wäre nur richtig, wenn wir die gleichen Compute-Intrinsics wie llama.cpp nutzten und der 13-%-Gap woanders läge. Wir nutzen sie **nicht** — der Gap IST im Kernel.

---

## Port-Roadmap (für die eigentliche Implementierung, separate Session)

### Schritt 1 — Infrastruktur (1 Session, ~200 LOC)

1. **Integer-WMMA-Intrinsic-Binding.** In `src_v1/backend/gpu/ffi.rs` / HIP-Wrapper:
   ```cpp
   // Neuer extern-"C" Entry in hip_kernels_v1/wmma/wmma_i32_test.hip
   // (Korrektheits-Test des int8 WMMA vs CPU-Ref)
   ```
2. **`block_q8_1_mmq` Quantize-Kernel.** Neue Datei `hip_kernels_v1/quantize/quantize_q8_1_mmq.hip` die FP32-Aktivierungen in das 128-Elem-Block-Format konvertiert (analog zu unserem bestehenden `rocmforge_launch_quantize_q8_1`, aber mit 4-Scale-pro-Block-Layout).
3. **Test: int8-WMMA-Correctness.** Ein 16×16×16-Integer-Gemm auf festen Test-Vektoren gegen skalare Referenz bit-exakt verifizieren (`tests_v1/wmma_int8_test.rs`).

### Schritt 2 — MMQ-Kernel-Port (1-2 Sessions, ~400-600 LOC)

1. **Q4_K → Q8_1 Tile-Load.** Portiere `load_tiles_q4_K` aus mmq.cuh:2151-2240 nach HIP. Input: Q4_K GMEM. Output: int8 in LDS packed als int32. Plus die per-32-Elem Scales/Mins in einem parallelen Half2-Buffer.
2. **Integer-MMA Vec-Dot.** Portiere `vec_dot_q8_1_q8_1_mma` aus mmq.cuh:1271-1323 — die innere int8-MMA-Schleife mit Scale-Fixup am Ende.
3. **Outer GEMM-Schleife.** Portiere `mul_mat_q` (mmq.cuh:3547-3630) — die stream-k Partitionierung; kann initial durch einfache statische Block-Partition ersetzt werden (Stream-K ist Perf-Optimierung, nicht Korrektheit).
4. **Adaptive `mmq_x_best` Heuristik.** Portiere die Grid-Heuristik aus mmq.cu:4169-4240.

### Schritt 3 — Integration + Tests (1 Session)

1. **Executor-Dispatch.** `dispatch_prefill_wmma_gemm` in `src_v1/graph/executor.rs` bekommt eine Q4_K-MMQ-Variante als zusätzlichen Arm.
2. **Parity-Test.** MMQ-Output vs CPU-FP32-Referenz + vs unser bestehendes FP16-WMMA (Toleranz für unterschiedliche Akkumulation).
3. **Perf-Test.** 15-Prompt-Suite + rocprof; Gate: ≥1200 tok/s Prefill auf 542-Token-Prompt.

### Geschätzter Gesamt-Aufwand

**2-3 Sessions à 4-6 h = 12-18 h Arbeit** — konsistent mit dem Projektplan-Voranschlag (5-6 Tage bei Halbtages-Arbeit).

---

## End-to-End Status (diese Session)

- **Prefill tok/s:** 1000 (unverändert — keine Code-Änderungen in dieser Session)
- **Decode tok/s:** 96.2 (unverändert)
- **Korrektheit:** 15/15 (unverändert)

Diese Session liefert Analyse-Ergebnisse, kein Code. Die Analyse ersetzt ~1-2 h der geplanten P0.2-Arbeit und gibt der nächsten Session einen scharf definierten Startpunkt.

---

## Dateien

- **Diese Analyse:** `results/phase3_p0.2_mmq_prefill_analysis.md`
- Referenz (nicht-kopierbar, da llama.cpp):
  - `~/tmp/llama.cpp/ggml/src/ggml-cuda/mmq.cuh` (Hauptkernel, 4282 LOC)
  - `~/tmp/llama.cpp/ggml/src/ggml-cuda/mma.cuh` (Integer-WMMA-Intrinsics, relevante Section 1236-1290)
  - `~/tmp/llama.cpp/ggml/src/ggml-cuda/mmq.cu` (Launch-Logic, 373 LOC)

Keine Code-Änderungen, kein Commit empfohlen — oder optional dieser Report als docs-commit.
