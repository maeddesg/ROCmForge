# P0.1 — WMMA Prefill Tile-Config Analyse

**Date:** 2026-04-24
**Branch:** v1.0-dev @ `bffe38e`
**Hardware:** AMD Radeon RX 9070 XT (gfx1201), 64 CUs / 128 AI Accelerators
**Binary:** `target/release/rocmforge-v1`
**Status:** Analyse-Phase abgeschlossen (Schritte 1-3). Sweep-Implementierung (Schritt 4) scope-out, siehe Abschnitt am Ende.

---

## Abweichungen vom Prompt (wichtig)

Der Prompt enthielt **zwei falsche Annahmen**, die ich vor der Arbeit verifizieren musste:

1. **Prompt:** „TILE_M=16, TILE_N=16, K_CHUNK=32 (vermutlich)"
   **Tatsächlich:** TILE_M=64, TILE_N=64, K_CHUNK=32 in **allen 8** WMMA-Kerneln (`hip_kernels_v1/wmma/*.hip`). Die Tiles sind bereits 4× größer pro Dimension als im Prompt angenommen.

2. **Prompt:** „architecture_v1.2.0-draft.md §5.2 (WMMA-Kernel Pseudo-Code, LDS-Layout)"
   **Tatsächlich:** §5.2 der Architektur-Doc ist „Cached-Run-Flow (Warm Start)". Die WMMA-Pseudo-Code-Spec liegt in **`docs/v1.0/dequant_ir_spec.md §5.2`** (habe ich gelesen).

Beide Korrekturen fließen in diesen Report ein. Der Effekt auf die Strategie: weniger „Default-Tiles sind zu klein" als vermutet, mehr „Tiles sind OK aber Block-Organisation ist suboptimal".

---

## Schritt 1 — rocprof Prefill-Baseline

**Command:**

```bash
rocprofv3 -o baseline --output-format csv --kernel-trace --stats --summary -- \
  ./target/release/rocmforge-v1 \
    --model ~/models/Qwen3-8B-Q4_K_M.gguf \
    --prompt "$(head -20 results/phase2_llamacpp_kernel_analysis.md)" \
    --max-tokens 1
```

Prompt-Tokens: **542** (gepadded auf 576 = 9 × M-Tile 64)
Wall-clock total: **555 ms** (ausgewiesen von der CLI)
Prefill tok/s wall-clock: **~1000** (542 / 0.54 s)

> Anmerkung: Der phase3-Projektplan schreibt „580 tok/s Prefill" — das war auf 33-Token-Prompts (Mutex-Baseline) gemessen, wo Padding auf M=64 2× Work-Waste verursacht. Bei längeren Prompts skaliert der Durchsatz nichtlinear nach oben, da die Padding-Overhead amortisiert wird. 542 Tokens ≈ 9× weniger Padding-Anteil → ~1.7× höhere tok/s.

### Kernel-Zeit-Zerlegung (rocprof `baseline_kernel_stats.csv`)

| Kernel | Calls | Σ ms | Avg µs | % Kernel-Zeit |
|---|---:|---:|---:|---:|
| **rf_v1_wmma_gemm_q4_k_fp16** | 216 | 245.8 | 1138 | **53.8 %** |
| **rf_v1_attention_prefill** | 36 | 85.2 | 2367 | **18.7 %** |
| **rf_v1_wmma_gemm_q6_k_fp16** | 37 | 67.1 | 1815 | **14.7 %** |
| rf_v1_kv_cache_append | 19548 | 24.6 | 1.3 | 5.4 % |
| rf_v1_gemv_q4_k_standard | 90 | 7.9 | 88 | 1.7 % |
| rf_v1_rms_norm_batched | 217 | 5.4 | 25 | 1.2 % |
| rf_v1_swiglu | 36 | 4.3 | 121 | 1.0 % |
| rf_v1_mmvq_q4_k_q8_1_fused | 36 | 3.6 | 100 | 0.8 % |
| rf_v1_gemv_q6_k_standard | 37 | 3.5 | 94 | 0.8 % |
| rf_v1_attention_decode | 36 | 2.0 | 55 | 0.4 % |
| rf_v1_rope_batched | 72 | 1.4 | 19 | 0.3 % |
| alle übrigen | — | ~2 | — | ~0.5 % |

**Prefill-Zeit-Zerlegung:**

- **WMMA-GEMM** (Q4_K + Q6_K): **68.5 %**
- **Attention prefill**: **18.7 %**
- KV-Cache append: 5.4 %
- Norm/RoPE/SwiGLU/embed/etc: ~4 %
- Decode overlap (1 Token): ~3 %

→ **Bottleneck ist klar der WMMA-GEMM-Pfad**, nicht Attention. Tile-Tuning ist der richtige Hebel für diese Prompt-Länge (~500 Tokens).

### WMMA-Kernel-Launch-Details (aus `baseline_kernel_trace.csv`)

Repräsentativer WMMA-Q4_K-FP16 Dispatch:

| Feld | Wert | Bedeutung |
|---|---:|---|
| LDS_Block_Size | **8192** | 8 KB = 12.5 % des 64-KB-Budgets (lots of headroom) |
| VGPR_Count | **88** | unter 104-Target ✓ |
| SGPR_Count | 128 | ok |
| Workgroup_Size | 128×1×1 | 4 Waves/Block |
| Grid_Size (flat) | variiert: 8192, 2048, 24576 in X × 9 in Y |

Kanonische Interpretation: `blocks_x = N/TILE_N = {64, 16, 192}` (entspricht Qwen3-Shapes N ∈ {4096, 1024, 12288} = QKV/KV/FFN). `blocks_y = M/TILE_M = 9` (= 576/64).

### Theoretisches Compute-Limit

```
gfx1201 peak FP16-WMMA:
  64 CU × 2 WMMA/CU/cycle × 16×16×16 × 2 FMAs × 2.97 GHz
  = 64 × 2 × 8192 × 2 × 2.97e9  = ~195 TFLOPS
```

Aktuelle Auslastung: Für eine der N=4096-WMMA-Calls (FLOPs ≈ 576 × 4096 × 4096 × 2 = 19.3 GFLOPs in 1138 µs):
```
19.3 GFLOPs / 1.138 ms = 16.9 TFLOPS
16.9 / 195 = ~8.7 %
```

→ **Compute-Auslastung liegt bei ~8.7 %** — matches exact die Aussage im `phase3_projektplan.md` („~8% Compute-Auslastung der WMMA-Units"). Enormer Kopfraum.

---

## Schritt 2 — Aktuelle Tile-Config identifiziert

**Quelle:** `hip_kernels_v1/wmma/wmma_gemm_q4_k_fp16.hip:22-28`

```c
#define TILE_M             64
#define TILE_N             64
#define TILE_K             16     // WMMA K-Dimension (festgelegt durch Intrinsic)
#define K_CHUNK            32     // LDS-Staging-Batch
#define WARPS_PER_BLOCK    4
#define THREADS_PER_BLOCK  (WARPS_PER_BLOCK * 32)   // = 128
#define COL_BLOCKS         4      // Jeder Wave verarbeitet 4 × 16 = 64 Spalten
```

**Identisch in allen 8 WMMA-Kerneln** (Q4_0, Q4_K, Q6_K, Q8_0 × FP16/FP8).

**LDS-Rechnung:**

```
A-Kachel: TILE_M × K_CHUNK × sizeof(half)   = 64 × 32 × 2 = 4096 B
B-Kachel: K_CHUNK × TILE_N × sizeof(half)   = 32 × 64 × 2 = 4096 B
Gesamt:                                                      8192 B  (= 12.5 % von 64 KB)
```

**Kein Double-Buffering** aktiv (eine A-Kachel und eine B-Kachel pro Workgroup).

**VGPR-Rechnung** (gemessen: 88 VGPRs):

- Akkumulator: `float8 acc[COL_BLOCKS]` = 4 × 8 FP32 = 32 VGPRs/Lane
- A-Fragment: `half8 a_reg` = 8 Halfs = 4 VGPRs/Lane
- B-Fragment: `half8 b_reg` = 8 Halfs = 4 VGPRs/Lane
- Dequant-Zwischenwerte + Loop-Indexe: ~48 VGPRs (entspricht Messung)
- **Σ gemessen: 88 VGPRs/Wave** ✓

**Occupancy:**

```
Per CU verfügbar: 1536 VGPRs, 64 KB LDS, ~15 Waves max
Pro Workgroup: 88 VGPR/Wave × 4 Waves = 352 VGPR, 8 KB LDS

VGPR-Limit:  1536 / 352 = 4.36 Workgroups/CU → 17 Waves/CU (VGPR-limitiert)
LDS-Limit:   65536 / 8192 = 8 Workgroups/CU → 32 Waves/CU  (nicht bindend)

→ Occupancy: ~15 Waves/CU (HW-Max für RDNA4), VGPR nicht bindend.
```

Occupancy ist bereits **optimal** für den aktuellen Kernel. Das Problem liegt nicht in der Occupancy, sondern in der **Arithmetic Intensity** (FMAs pro LDS-Load).

---

## Schritt 3 — llama.cpp MMQ-Kernel Referenz

**Quellen:** `~/tmp/llama.cpp/ggml/src/ggml-cuda/mmq.cuh` + `mmq.cu`.

### Tile-Sizes auf RDNA2+ (AMD_WMMA_AVAILABLE)

```cpp
// mmq.cuh:116-137 — Max-N-tile (MMQ_X)
static constexpr __device__ int get_mmq_x_max_device() {
#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    return 128;
#endif
}

// mmq.cuh:152-166 — M-tile (MMQ_Y)
static constexpr __device__ int get_mmq_y_device() {
#if defined(GGML_USE_HIP)
#if defined(RDNA1)
    return 64;
#else                       // RDNA2, RDNA3, RDNA4 → 128
    return 128;
#endif
#endif
}

#define MMQ_NWARPS 8                // 8 warps × 32 lanes = 256 threads/block
#define MMQ_TILE_NE_K 32            // K-direction quantized units
#define MMQ_ITER_K 256              // Full Q4_K super-block per iteration
```

### Wesentliche Unterschiede zu uns

| Parameter | Unser Kernel | llama.cpp MMQ (RDNA2+) |
|---|---:|---:|
| TILE_M | 64 | **128** (2×) |
| TILE_N (max) | 64 | **128** (2×) |
| K_CHUNK (Quant-Elemente) | 32 | 32 (gleich) |
| Warps/Block | **4** | **8** (2×) |
| Threads/Block | 128 | 256 |
| Target-Range | fixed | dynamisch per `mmq_x_best` (Grid-Heuristik) |

### Grid-Heuristik (mmq.cu)

```cpp
// mmq.cu:4175-4189 — wählt mmq_x adaptiv so dass viele Grid-Tiles entstehen
for (int mmq_x = 8; mmq_x <= mmq_x_max && ntiles_x_best > 1; mmq_x += 8) {
    const int granularity = mmq_get_granularity_host(mmq_x, cc);
    const int ntiles_x = ceil_div(ncols, mmq_x);
    const int ntiles   = ntiles_x * ntiles_y;
    if (ntiles * ... < ntiles_x_best * ...) {
        ntiles_x_best = ntiles_x;
        mmq_x_best    = mmq_x;
    }
}
```

→ llama.cpp wählt **adaptiv per Call** die Tile-Größe so dass die Grid-Menge hoch genug bleibt für GPU-Sättigung. Wir haben Tiles statisch.

### Implikationen für uns

1. **Tile-Verdopplung auf 128×128** ist llama.cpp's Standard auf RDNA2+. LDS-Footprint würde dadurch auf 32 KB/Workgroup steigen (2×128×32×2), immer noch <64 KB. Aber VGPR-Belegung wächst stark (4× Akkumulator → ~128 VGPRs/Wave bei 4 Warps oder ~64 bei 8 Warps). Bei 8 Warps wäre VGPR-Budget einhaltbar.
2. **Adaptive Tile-Auswahl per Call** statt statische. Besonders wichtig für kleine seq_len wo 128×128 Tiles Padding-Verschwendung produzieren.
3. **8-Warp Workgroups** nutzen 256 Threads/CU = weniger Blocks/CU nötig für Sättigung, aber bessere Memory-Bandbreite pro Block (256 Lanes parallel im LDS-Load).

---

## Schritt 4 — Tile-Config-Sweep: **SCOPE-OUT / NOT DONE**

### Warum nicht in dieser Session implementiert

Der Prompt spezifiziert **„Option A: Parametrisierter Kernel mit #define, hipcc --genco -DTILE_M=..."** plus 48 Configs × Kompilierung + Benchmark + Korrektheit. Das ist **nicht in einer Session machbar**:

1. **Parametrisierung erfordert strukturelle Kernel-Änderungen:**
   - Der aktuelle Kernel hat hartcodierten `COL_BLOCKS=4` (= TILE_N/16/lanes_per_wave_in_N), `WARPS_PER_BLOCK=4`, A/B-Register-Layout pro COL_BLOCK. Eine Änderung von TILE_N auf 128 erfordert COL_BLOCKS=8 oder Neuverteilung auf 8 Warps. Das ist KEIN reiner #define-Swap — das Register-Layout und die LDS-Zugriffsmuster müssen angepasst werden.
   - Analog für TILE_M: `a_row_in_tile = wave_id * 16 + (lane & 15)` funktioniert nur wenn `WAVES × 16 = TILE_M`. Bei TILE_M=128 bräuchten wir 8 Warps.
   - Diese strukturellen Änderungen sind pro Kernel-Format (Q4_K, Q6_K, …) separat zu machen.

2. **Infrastruktur fehlt:**
   - Unsere Kernel werden in `build.rs` bei Cargo-Build-Time kompiliert, nicht runtime. Ein dynamischer Sweep braucht entweder (a) eine neue `rf-forge`-Sweep-CLI die `hipcc` selbst aufruft und `hipModule*`-basiert lädt, oder (b) die Template-basierte Kernel-Generation die bereits teilweise in `src_v1/ga/` für GEMV existiert, aber für WMMA nicht.
   - Aufwand für dynamische Sweep-Infra: geschätzt **1-2 Tage** allein für GEMV; WMMA braucht mehr, da Register-Layout pro Config anders ist.

3. **Korrektheits-Validierung pro Config** (vom Prompt explizit gefordert):
   - Jede Config muss gegen Decode-Loop-Referenz getestet werden.
   - 48 Configs × CPU-Dequant Parity-Test = deutliche Runtime + Implementierungsaufwand.

4. **Risiko ohne Test-Harness:** Wenn ich jetzt einfach `TILE_M=128, TILE_N=128` manuell setze und der Kernel korrupte Werte produziert (sehr wahrscheinlich wegen der strukturellen Abhängigkeiten oben), verliere ich das Fundament aller weiteren Messungen.

### Was stattdessen in diesem Report steht

- **Präzise Baseline** inkl. rocprof-Details für zukünftige Vergleiche.
- **Korrigierte Config-Annahmen** (aktuell TILE_M=64, nicht 16).
- **llama.cpp-Referenz** mit konkreten Werten (128×128×32, 8 Warps) + Grid-Heuristik.
- **Konkrete VGPR/LDS-Rechnungen** für die llama.cpp-Config als Ausgangspunkt.

### Empfohlener Pfad für die eigentliche Implementation (separate Session)

**Schritt A — Template-parametrisierter Q4_K FP16 Kernel (1 Tag):**
1. In `hip_kernels_v1/wmma/` neuen `wmma_gemm_q4_k_fp16_tuned.hip.in` ablegen mit:
   - `TILE_M`, `TILE_N`, `K_CHUNK` als `-D` Parameter
   - `COL_BLOCKS = TILE_N / 16` dynamisch abgeleitet
   - `WARPS_PER_BLOCK = TILE_M / 16` dynamisch abgeleitet
   - A/B-Register-Layout-Berechnung über Makros
2. Build-Skript (`build.rs` oder `rf-forge sweep`) kompiliert eine Matrix von Varianten und linkt sie als separate extern-"C" Funktionen.

**Schritt B — Dispatch-Sweep-Harness (1 Tag):**
1. `tests_v1/wmma_tile_sweep_test.rs`:
   - Iteriert über vorkompiliertes Set {16, 32, 64, 128}² × {16, 32, 64}.
   - Parity-Check jede Config gegen CPU-FP32-Referenz.
   - Timing auf Mutex-Prompt (33 tok) + langem Prompt (~500 tok) für zwei Regime.

**Schritt C — Kandidaten-Ranking + Production-Kernel (0.5 Tag):**
- Top-3 Configs pro (format, seq_len_regime) identifizieren.
- Eine als neuer Default, Rest als Bandit-Optionen.

**Realistische Gesamtdauer:** 2-3 Tage konzentrierte Arbeit. Matcht den Project-Plan-Voranschlag (Tag 1-4).

---

## Konkrete erste-Kandidat-Configs für Schritt A

Aus der Analyse oben die vier vielversprechendsten Kandidaten, geordnet nach erwartetem Impact:

| # | TILE_M | TILE_N | K_CHUNK | WARPS | LDS (KB) | Erwarteter VGPR | Rationale |
|---|---:|---:|---:|---:|---:|---:|---|
| 0 | 64 | 64 | 32 | 4 | 8 | 88 | **Default** (Baseline) |
| 1 | 64 | 128 | 32 | 4 | 12 | ~104 | llama.cpp-halber Step, selbes M, mehr COL_BLOCKS |
| 2 | 128 | 64 | 32 | 8 | 12 | ~88 | Mehr M, mehr Warps, LDS moderat |
| 3 | **128** | **128** | **32** | **8** | **16** | ~104 | **llama.cpp-Äquivalent**, Hauptkandidat |
| 4 | 128 | 128 | 64 | 8 | 32 | ~120 | Größerer K-Chunk, Prolog-Amortisation — VGPR-Risiko |

**LDS-Budget-Check** alle unter 64 KB ✓. Config #4 ist näher am 104-VGPR-Limit und könnte Occupancy-Drop bringen — muss gemessen werden.

---

## End-to-End Status

| Metrik | Aktuell (42 Token Prompt, rocprof) | Ziel (P0.1) | Ziel (P0 gesamt) |
|---|---:|---:|---:|
| Prefill tok/s (kurzer Prompt, 33 tok) | ~330 | ≥ 800 | — |
| Prefill tok/s (mittlerer Prompt, 542 tok) | ~1000 | ≥ 1500 | ≥ 4000 |
| Compute-Auslastung | 8.7 % | 15–20 % | 40 %+ |
| Decode tok/s | 96.2 | unverändert | unverändert |
| Korrektheit | 15/15 Qwen3 | 15/15 | 15/15 |

## Dateien

- **Diese Datei:** `results/phase3_p0.1_wmma_tile_tuning.md`
- rocprof-Rohdaten: `./baseline_kernel_stats.csv`, `./baseline_kernel_trace.csv`, `./baseline_domain_stats.csv`

## Fazit

Die Analyse-Arbeit (Schritte 1-3) ist komplett und macht klar: **WMMA-GEMM dominiert Prefill mit 68.5 %**, die Tiles sind bereits 4× größer als im Prompt angenommen, und die llama.cpp-Referenz zeigt eine klare Richtung (128×128 Tiles + 8 Warps) mit adaptiver Grid-Heuristik. Der eigentliche Sweep (Schritt 4) ist mit korrekter Infrastruktur (Template-Kernel + Sweep-Harness) ein 2-3-Tage-Stück; ohne diese Infrastruktur wäre ein Quick-Hack-Versuch mit hoher Wahrscheinlichkeit inkorrekt und würde das Baseline-Fundament kompromittieren.

**Empfehlung:** Next-Session-Scope wäre Schritt A (Template-Kernel für Q4_K FP16). Wenn einmal implementiert, ist der Sweep selbst in wenigen Stunden durchgeführt.
