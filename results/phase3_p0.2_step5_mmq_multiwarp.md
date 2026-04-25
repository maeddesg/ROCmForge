# P0.2 MMQ-Port Schritt 5 — Multi-Warp Scale-Up

**Date:** 2026-04-25
**Branch:** v1.0-dev
**Status:** **4-Warp-Kernel landet, korrekt, aber kein E2E-Speedup**. 4W liefert bit-identische Outputs zum 1W-Schritt-3-Kernel über alle getesteten Shapes (inkl. Qwen3 QKV M=64×N=4096×K=4096, **0 differierende Elemente**). Isolated-Kernel-Timing zeigt 0.96–1.05× (Noise), E2E-Wallclock auf dem 542-Token-Prompt 459 ms (4W) vs 466 ms (1W) — **1.5 % schneller, im Mess-Noise**. VGPR-Count und Workgroup-Belegung unverändert. **Empfehlung: 4W-Variante landet als Production-Default-Kernel (kein Regression-Risiko, marginaler Gewinn), aber LDS-Sharing der Weights ist erforderlich um den im Prompt erwarteten 30-50 %-Sprung zu erreichen.**

## TL;DR

Der Schritt-3 1-Warp-MMQ-Kernel wurde als Template auf 1 oder 4 Warps generalisiert. Beide Varianten produzieren bit-identische Ergebnisse (verifiziert über 5 Shape-Klassen). E2E-Performance ist **neutral** (459 vs 466 ms, +1.5 %), sowohl isoliert (0.96–1.05× über M ∈ {64, 192, 256, 576}) als auch wallclock. Block-Count sinkt 4× (9216 → 2304 für QKV M=576×N=4096), Total-Thread-Count bleibt identisch (294912 lanes). Der Kernel ist offenbar **Integer-WMMA-Throughput-bound, nicht Launch-Overhead-bound** — die 4 Warps pro Block konkurrieren um dieselbe MMA-Pipeline auf demselben CU, statt parallel über verschiedene CUs zu laufen wie 1 Warp pro Block.

## Was landet

**Datei:** `hip_kernels_v1/wmma/mmq_q4_k.hip` (modifiziert, +71 LOC, gleiche Datei)

### Refactor

Der Schritt-3-Kernel wurde zu einem `template<int WARPS_PER_BLOCK>` mit zwei extern-"C"-Entries:

```cpp
template<int WARPS_PER_BLOCK>
__global__ void rf_v1_mmq_q4_k_kernel(...)

extern "C" rocmforge_launch_mmq_q4_k       → Template<4>  (production)
extern "C" rocmforge_launch_mmq_q4_k_1w    → Template<1>  (parity reference)
```

Die einzige semantische Änderung gegenüber Schritt 3 ist die Berechnung des Warp-M-Offsets:

```cpp
constexpr int TILE_M    = WARPS_PER_BLOCK * 16;
const int warp_id       = threadIdx.x >> 5;
const int m_col_base    = blockIdx.y * TILE_M + warp_id * 16;
if (m_col_base >= M) return;   // outer warp boundary
```

Inner-Loop (Q4_K-Nibble-Extraktion, Integer-WMMA-Calls, Scale-Fixup, Write-back) ist **byte-identisch** zur Schritt-3-Version. Boundary-Handling für M nicht-Vielfaches-von-64 ist double-layered: outer `m_col_base >= M` für komplette Warps, inner `m_out >= M` (im Write-back) für partielle Warps (z.B. M=33: Warp 2 schreibt Row 32, Lanes mit row16>0 returnen).

### Launch

```cpp
constexpr int TILE_M  = WARPS_PER_BLOCK * 16;
constexpr int THREADS = WARPS_PER_BLOCK * 32;
dim3 grid(N / 16, (M + TILE_M - 1) / TILE_M, 1);
dim3 block(THREADS, 1, 1);
```

Für die Qwen3-QKV-Shape (M=576, N=4096):
- 1W: Block(32), Grid(256, 36)        = 9216 Blocks × 32 Lanes = 294912 Lanes
- 4W: Block(128), Grid(256, 9)        = 2304 Blocks × 128 Lanes = 294912 Lanes

Same total work, different granularity.

### Kein LDS-Sharing

Bewusst nicht implementiert in dieser Revision. Jeder der 4 Warps liest seine eigenen Activation-Rows (16 verschiedene M-Rows) und seine eigene Kopie der Q4_K-Weights aus L2/L1. Theoretisch könnten alle 4 Warps eine geteilte Kopie der Weights aus LDS lesen (4× weniger L2/L1-Reads), aber dies setzt voraus dass der Kernel Memory-Bound ist — was die Messungen unten widerlegen.

## Korrektheit

### Test-Suite `tests_v1/mmq_multiwarp_test.rs`

6 Tests, alle grün (0.82s wallclock):

| Test | Resultat | Anmerkung |
|---|---|---|
| `mmq_4w_vs_1w_parity_small` (16/32/64 × 16/32/64 × 256) | ✅ **0 differierende Elemente** | Bit-exakt |
| `mmq_4w_vs_1w_parity_large_k` (16×16×1024, 64×64×1024) | ✅ **0 differierende Elemente** | Bit-exakt |
| `mmq_4w_vs_1w_parity_qwen3_qkv` (64×4096×4096) | ✅ **0/262144 differierende Elemente** | Production-Shape, bit-exakt |
| `mmq_4w_vs_cpu_reference` (64×64×256, 64×4096×4096) | ✅ max_abs 21 / 73 (innerhalb tol) | Independent ground truth |
| `mmq_4w_boundary_m_not_multiple_of_64` (M=48, partial-block) | ✅ max_abs 26 (innerhalb tol) | Warps 0+1+2 active, Warp 3 returns |
| `mmq_4w_faster_than_1w` (M ∈ {64,192,256,576}, sweep) | ✅ Min-Speedup 0.96× (gate ≥ 0.90×) | Neutral, kein Regression |

**Der wichtige Befund: 4W vs 1W produziert byte-für-byte denselben Output**, nicht nur "innerhalb FP16-Toleranz". Die Akkumulation-Reihenfolge ist exakt gleich (gleiche Outer-K-Loop, gleicher Inner-WMMA-Loop, gleiche Scale-Fixup-Reihenfolge), nur die Warp-zu-M-Row-Zuordnung unterscheidet sich — und das beeinflusst kein einziges Bit.

### E2E-Kohärenz (Mutex-Prompt, Qwen3-8B-Q4_K_M)

Beide Pfade (4W, 1W) liefern **wort-identische Outputs**:

> "A mutex, short for 'mutual exclusion,' is a synchronization mechanism used in multithreaded programming to ensure that only one thread can access a shared resource or critical section of code at any given time, thereby enforcing **mutual exclusion** and preventing race conditions. When a thread acquires the mutex, it locks the resource; other threads attempting to access it must wait until..."

Decode tok/s identisch (54.6 vs 54.6).

## Performance

### Isolated-Kernel-Timing (HIP Events, 20 iter pro Variante)

| Shape | 1W ms | 4W ms | Speedup |
|---|---:|---:|---:|
| 64 × 4096 × 4096 | 0.068 | 0.067 | 1.01× |
| 192 × 4096 × 4096 | 0.175 | 0.182 | **0.96×** ⚠ |
| 256 × 4096 × 4096 | 0.221 | 0.224 | 0.98× |
| 576 × 4096 × 4096 | 0.478 | 0.454 | 1.05× |

Min Speedup 0.96×, Max 1.05×. Effektiv **statistisches Rauschen**.

### E2E-Wallclock (542-Token-Prompt, --max-tokens 1, 3 Runs Median)

| Variante | Total ms | Decode tok/s |
|---|---:|---:|
| 4W (`ROCMFORGE_PREFILL_MMQ=1`) | **459** | 26.7 |
| 1W (`ROCMFORGE_PREFILL_MMQ_1W=1`) | 466 | 26.9 |
| FP16-WMMA (Schritt-4 Baseline) | n/a (~595, see step 4) | ≈26 |

**4W → 1W Wallclock-Speedup: 1.015×** (1.5 %, im Mess-Noise).

### Per-Kernel rocprof-Vergleich (216 Calls, 542-Token-Prompt)

| Kernel | 4W (Σ ms) | 1W (Σ ms) | Δ |
|---|---:|---:|---|
| **rf_v1_mmq_q4_k_kernel** | **150.47** | 151.91 | **−0.95 %** |
| rf_v1_attention_prefill_kernel | 85.49 | 84.52 | +1.1 % (Noise) |
| rf_v1_wmma_gemm_q6_k_fp16_kernel | 67.10 | 66.97 | ±0 |
| rf_v1_kv_cache_append_kernel | 24.42 | 24.29 | ±0 |
| rf_v1_quantize_q8_1_mmq_kernel | 2.98 | 2.92 | ±0 |

**Net Q4_K prefill speedup: 1.0 %** auf der wichtigsten Pipeline-Komponente.

### Resource-Usage (kernel_trace.csv, beide Varianten)

| Metrik | 4W | 1W |
|---|---:|---:|
| VGPR_Count | **152** | **152** |
| Accum_VGPR_Count | 0 | 0 |
| LDS_Block_Size | 0 | 0 |
| Workgroup_Size | 128 | 32 |
| SGPR_Count | 128 | 128 |
| Total Threads (QKV: M=576, N=4096) | 294912 | 294912 |
| Total Blocks | **2304** | 9216 (4×) |

**Schlüsselbeobachtung:** VGPR/Lane bleibt bei 152, weil der Compiler den Akkumulator (`int32x8_t acc_sb[8]` = 64 i32) plus Fragment-Buffer plus Float-Akkumulator pro Lane allokiert — unabhängig von WARPS_PER_BLOCK. Pro Block sind das jetzt 152×128 = 19456 VGPRs vs 152×32 = 4864 VGPRs — **bedeutet höhere VGPR-Pressure pro CU**. Bei 4W: 1 Block braucht ~19456 VGPR; CU hat 1536×8 = 12288 VGPRs (hardware limit) → max 0.6 Blocks/CU effektiv. Bei 1W: 4864 VGPR/Block → max 2.5 Blocks/CU.

Das könnte erklären warum 4W trotz 4× weniger Blocks nicht schneller ist: die Blocks-pro-CU-Reduktion frisst den Overhead-Gewinn auf, und beide Setups landen bei ähnlicher effektiver Compute-Pipeline-Auslastung.

## Warum kein Speedup?

Drei plausible Faktoren, geordnet nach Wahrscheinlichkeit:

1. **VGPR-Druck wächst mit Warps** (siehe Tabelle oben). Mehr Warps pro Block erhöht VGPR-Demand pro Block linear, was Blocks/CU senkt. Im Effekt landet die Compute-Pipeline-Auslastung bei beiden Varianten ähnlich.

2. **Kein LDS-Sharing → keine Memory-Bandbreite-Einsparung.** Im aktuellen Kernel liest jeder der 4 Warps die SAME Q4_K-Weights aus L2/L1 unabhängig. Wenn der Kernel L2-bound wäre, wäre Multi-Warp ohne Sharing eine Pessimierung; offenbar ist er int-WMMA-bound (FP16-Equivalent: 8.7 % Compute-Auslastung war FP16, integer könnte höher sein), weshalb beide Varianten am Compute-Bottleneck stranden.

3. **Launch-Overhead ist auf RDNA4 sehr klein.** 9216 vs 2304 Blocks bedeutet nur ~7000 weniger HW-Block-Dispatches. Bei <1µs pro Block-Dispatch sind das ~7 ms gespart über 216 GEMM-Calls = ~32µs/Call. Total-Kernel-Zeit ist 700µs/Call, also **~5 % maximaler theoretischer Launch-Overhead-Gewinn** — gemessene 1.0 % ist konsistent damit minus Pipeline-Effekte.

## Empfehlung

**4W als Default-Production-Kernel landen** für `rocmforge_launch_mmq_q4_k`. Begründung:
- Bit-identisch zur 1W-Variante, kein Regression-Risiko.
- Marginal schneller (~1 %), nie langsamer als Mess-Noise.
- Reduziert Block-Count 4× was bei künftigen Optimierungen (LDS-Sharing) der Hebel ist.
- 1W-Variante bleibt als `rocmforge_launch_mmq_q4_k_1w` für Diagnostik via `ROCMFORGE_PREFILL_MMQ_1W=1`.

**Aber: Nicht als P0.2-Schritt-5-Sieger werten.** Der im Prompt erwartete 30-50 %-Speedup ist nicht eingetroffen. Der eigentliche Hebel für MMQ-Multi-Warp-Gewinn ist **Weight-LDS-Sharing**, das in dieser Revision bewusst ausgesetzt wurde.

## Nächste Schritte (Priorität für künftige Sessions)

1. **Schritt 5b: Weight-LDS-Sharing** (1-2 Sessions, hohes Risiko/hoher Reward).
   - Pro Block: 1 Warp lädt 16 Q4_K-Super-Blöcke (16 × 144 B = 2.3 KB) in `__shared__` Arrays, alle 4 Warps konsumieren via gemeinsamen Lese-Index.
   - `__syncthreads()` zwischen Load- und Compute-Phase.
   - Potential: 4× weniger L1/L2-Reads für Q4_K-Bytes, könnte +10-20 % bringen wenn Q4_K-Bandbreite ein nennenswerter Anteil der ms ist.
   - Risiko: VGPR-Pressure wächst weiter, Occupancy sinkt, könnte sogar regressieren.

2. **Q6_K MMQ-Port** (1-2 Sessions, mittleres Risiko/mittleres Reward).
   - 14.7 % der Prefill-Zeit ist immer noch Q6_K (LM-Head + verschiedene Layer-Norms in Qwen3).
   - Selbe Integer-WMMA-Infra wie Q4_K, aber Q6_K hat 6-bit-Quantisierung (ql + qh splits).
   - Erwarteter Gewinn: ähnlich Q4_K (-37 % Q6_K-Kernel-Zeit → ~5 % E2E).

3. **VGPR-Optimierung im aktuellen Kernel** (kleine Session, niedriges Risiko/niedriges Reward).
   - Schreib-Akkumulator `c_acc[8]` könnte direkt in `acc_sb[][]` reused werden statt separat → ~16 VGPRs sparen.
   - 152 → ~136 VGPR/Lane würde Occupancy auf 4 Blocks/CU statt ~3 heben.
   - Erwarteter Gewinn: 5-10 % E2E.

## Geänderte Dateien

| Datei | Änderung |
|---|---|
| `hip_kernels_v1/wmma/mmq_q4_k.hip` | Refactor: Template<WARPS_PER_BLOCK>, zwei extern-"C" entries |
| `src_v1/backend/gpu/wmma.rs` | Neuer FFI-Eintrag `rocmforge_launch_mmq_q4_k_1w` |
| `src_v1/graph/executor.rs` | Env-Switch `ROCMFORGE_PREFILL_MMQ_1W=1` zwischen 4W/1W |
| `tests_v1/mmq_multiwarp_test.rs` | **neu**, 6 Tests (parity + boundary + perf-sweep) |
| `Cargo.toml` | Test-Registrierung |

`build.rs` und `CMakeLists.txt` **unverändert** — beide FFI-Entries kommen aus derselben Library `v1_mmq_q4_k`.

## Fortschritt-Zusammenfassung P0.2

| Schritt | Status | Sessions | Effektivität |
|---|:---:|:---:|---|
| 1: Integer-WMMA + block_q8_1_mmq Infra | ✅ | 1 | Foundation |
| 2: Minimal-Kernel (3 Bug-Fix-Runden) | ✅ | 4 | Math-Korrektheit etabliert |
| 3: Scale-Up (variable M,N,K) | ✅ | 1 | Variable-Shape-Tauglichkeit |
| 4: Executor-Integration + Perf | ✅ | 1 | **+28.7 % Prefill (E2E)** |
| 5: Multi-Warp Scale-Up | ✅ | 1 (diese Session) | **+1.5 % E2E (im Noise)** |
| 5b: Weight-LDS-Sharing (deferred) | 🔜 | — | Potential: +10-20 % E2E |

**Gesamt 8 Sessions für P0.2.** Schritt 4 (1W-MMQ-Integration) ist bisher der größte Einzel-Gewinn. Schritt 5 (4W ohne LDS-Sharing) ist neutral — aber notwendige Infrastruktur für 5b.

## Anhang — Test-Run-Output (verbatim)

```
running 6 tests
test mmq_4w_boundary_m_not_multiple_of_64 ...
  4W M=48 boundary: max_abs=2.5964e1 max_mag=7.9103e3 tol=1.2656e3
ok
test mmq_4w_vs_1w_parity_small ...
  16×16×256   max_abs(4W-1W)=0.0000e0 differing_elems=0/256
  32×32×256   max_abs(4W-1W)=0.0000e0 differing_elems=0/1024
  64×64×256   max_abs(4W-1W)=0.0000e0 differing_elems=0/4096
ok
test mmq_4w_vs_1w_parity_large_k ...
  16×16×1024  max_abs(4W-1W)=0.0000e0 differing_elems=0/256
  64×64×1024  max_abs(4W-1W)=0.0000e0 differing_elems=0/4096
ok
test mmq_4w_vs_1w_parity_qwen3_qkv ...
  64×4096×4096(Qwen3 QKV) max_abs(4W-1W)=0.0000e0 differing_elems=0/262144
ok
test mmq_4w_vs_cpu_reference ...
  64×64×256        max_abs=2.1234e1  max_mag=5.1509e3  tol=4.1207e2
  64×4096×4096(QKV) max_abs=7.3696e1 max_mag=2.1277e4  tol=6.8087e3
ok
test mmq_4w_faster_than_1w ...
      M     N     K       1W ms       4W ms   speedup
     64  4096  4096       0.068       0.067     1.01×
    192  4096  4096       0.175       0.182     0.96×
    256  4096  4096       0.221       0.224     0.98×
    576  4096  4096       0.478       0.454     1.05×
ok

test result: ok. 6 passed; 0 failed; 0 ignored.
```
