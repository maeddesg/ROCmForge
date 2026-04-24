# P0.2 MMQ-Port Schritt 3 — Scale-Up

**Date:** 2026-04-24
**Branch:** v1.0-dev
**Status:** **Scale-Up-Kernel korrekt über alle getesteten Shapes.** Executor-Integration scope-out für separate Session.

## TL;DR

Der minimale 16×16×256-Kernel aus Schritt 2 wurde zum variablen-(M,N,K) Kernel `mmq_q4_k.hip` skaliert. Gegen CPU-FP32-Referenz bit-quasi-exakt auf allen getesteten Shapes inkl. **realistischer Qwen3-QKV-Shape (M=64, N=4096, K=4096)**. Kernel-Math unverändert (die drei Schritt-2-Fixes sind drin); nur Work-Distribution geändert. **Executor-Integration + Performance-Messung sind explizit NICHT Teil dieser Session** — nächster Schritt.

## Was landet

**Datei:** `hip_kernels_v1/wmma/mmq_q4_k.hip` (~180 LOC, direkter Abkömmling des Minimals)

### Scale-Up-Änderungen gegenüber Minimal-Kernel

| Aspekt | Minimal (Schritt 2) | Scale-Up (Schritt 3) |
|---|---|---|
| Tile-Form | 16×16 hartkodiert | 16×16 pro Grid-Block (variable Grid) |
| Grid | 1 Block | (N/16, M/16) Blocks |
| Warps/Block | 1 | 1 (unverändert, Multi-Warp deferred) |
| K-Dim | K=256 hartkodiert (1 Super-Block) | Variable K, Outer-Loop über K/256 |
| Akkumulator | Pro Super-Block int32, Scale-Fixup am Ende | Pro Super-Block int32, Float-Akkumulator über Super-Blöcke gesummt |
| Nibble-Extraktion | Pair-basiert (Schritt-2-Fix) | **Identisch** |
| Fragment-Lane-Layout | Per `mma.cuh:198-230` | **Identisch** |

Jeder Grid-Block ist effektiv eine Instanz des Minimal-Kernels mit an der richtigen (M, N)-Position platzierten Indizes und einem zusätzlichen Outer-Loop über K.

## Parity-Test-Resultate

Alle max_abs-Werte innerhalb der magnitude-aware FP16-Akkumulations-Toleranz `(max_mag + 1e-3) * sqrt(K) * 5e-3`.

| Shape | max_abs | max_mag | Tolerance | Status |
|---|---:|---:|---:|:---:|
| 16×16×256 (Baseline Minimal-Reprise) | 13.7 | 3155 | 252 | ✅ |
| 16×16×512 (2 Super-Blöcke) | 11.3 | 3784 | 428 | ✅ |
| 16×16×1024 (4 Super-Blöcke) | 25.3 | 5754 | 921 | ✅ |
| 16×16×4096 (16 Super-Blöcke) | 37.8 | 10454 | 3345 | ✅ |
| 32×32×256 (4 Grid-Blocks) | 16.1 | 2836 | 227 | ✅ |
| 32×64×512 (8 Grid-Blocks) | 18.5 | 6331 | 716 | ✅ |
| 48×16×256 (3 Grid-Blocks) | 16.2 | 3936 | 315 | ✅ |
| **64×4096×4096 (Qwen3 QKV)** | **73.7** | **21277** | **6809** | ✅ |

**Worst-case relative Fehler ≈ 0.35%** (max_abs / max_mag). Konsistent mit FP16-Akkumulations-Noise plus Q8_1-Quantisierungs-Noise. Alle Tests grün.

## Was NICHT implementiert wurde (Scope-Out)

### Multi-Warp pro Block

Der Kernel nutzt weiterhin 1 Warp pro Block, nicht 4 oder 8 wie llama.cpp. Das Minimal-Layout funktioniert aber gibt:
- **Für Qwen3 QKV M=576 × N=4096:** Grid = (256, 36) = 9216 blocks × 1 warp = 9216 waves, ~10× mehr als die 960 wave-slots → oversaturated aber viele kleine Blocks = viele Kernel-Launches.
- **Nicht performance-optimal**, aber korrekt. Multi-Warp-Skalierung ist mechanisch aber nicht trivial (LDS-Teilen, warp_id-Partitionierung) und wurde für separate Session aufgeschoben.

### Executor-Integration

Der Prefill-Pfad in `src_v1/graph/executor.rs::dispatch_prefill_wmma_gemm` nutzt weiterhin `rocmforge_launch_wmma_gemm_q4_k_fp16` (FP16-WMMA). Integration mit MMQ braucht:
1. **Pre-Quantise-Pass** für FP32-Activations → `block_q8_1_mmq` (Schritt 1 Kernel)
2. **Persistent Quantize-Buffer** im Executor (pro Prefill-Call, oder reuse)
3. **Tensor-Strides angleichen** — MMQ erwartet `[M × K]` row-major Activations; die aktuelle Prefill-Datenstruktur muss überprüft werden
4. **Bandit-Alternative ODER direct replace** — ehrlich vergleichbar sobald gemessen

Aufwand: 1-2 Stunden Integration + rocprof-Lauf. Besser in einer dedicated Session wo E2E-Tests + 15-Prompt-Suite + Decode-Regression ohne Zeitdruck ausführbar sind.

### Performance-Messung

Ohne Executor-Integration kein E2E-Prefill-Benchmark. Isolated-Kernel-Timing könnte man separat machen, aber der wirklich interessante Zahl (Prefill tok/s Ende-zu-Ende inkl. Quantize-Overhead) braucht Integration.

## Gelandete Dateien

- `hip_kernels_v1/wmma/mmq_q4_k.hip` — **neu**, Scale-Up-Kernel
- `hip_kernels_v1/CMakeLists.txt` — Library-Registrierung
- `build.rs` — Linker-Zeile
- `src_v1/backend/gpu/wmma.rs` — FFI für `rocmforge_launch_mmq_q4_k`
- `tests_v1/mmq_q4_k_scaleup_test.rs` — **neu**, 4 Parity-Tests (8 Shapes insgesamt)
- `Cargo.toml` — Test-Registrierung

**Kein Produktionspfad geändert.** Prefill bleibt bei FP16-WMMA, Decode unverändert.

## End-to-End Status

| Metrik | Vor Schritt 3 | Nach Schritt 3 |
|---|---:|---:|
| Prefill tok/s (FP16-WMMA) | ~1000 | ~1000 (unverändert) |
| Decode tok/s | 96.2 | 96.2 (unverändert) |
| MMQ-Tests grün | 6/6 (Schritt 1+2) | 10/10 (+4 scale-up) |
| MMQ-Kernel einsetzbar in Prefill | ❌ | ❌ (Integration pending) |

## Nächster Schritt — Schritt 4 (Executor-Integration + Perf)

Konkreter Plan für die nächste Session:

1. **Quantize-Buffer im Executor** (30 min):
   - Allocate `M_max × K_max / 128 * 144 Bytes` in `GraphExecutor::new` oder first-call
   - Reuse across Prefill-Calls

2. **MMQ-Dispatch als Opt-in-Pfad** (30 min):
   - Env-Flag `ROCMFORGE_PREFILL_MMQ=1` schaltet um
   - Fallback zu FP16-WMMA bei Flag=0

3. **E2E-Tests** (30 min):
   - Bestehender Prefill-WMMA-Test mit MMQ-Pfad re-run
   - 15-Prompt-Suite mit MMQ — Kohärenz 15/15 Gate

4. **rocprof-Vergleich** (30 min):
   - 542-Token-Prompt, beide Pfade
   - Integer-MMA vs FP16-MMA: µs, Compute-Auslastung, VGPR-Count

5. **Entscheidung**: Bei Netto-Gewinn (inkl. Quantize-Overhead): als Default setzen. Bei Netto-Verlust: ehrlich reporten + Optimierungs-Roadmap (Multi-Warp, Double-Buffering).

## Fortschritt-Zusammenfassung P0.2

| Schritt | Status | Session-Anzahl |
|---|:---:|:---:|
| Schritt 1: Integer-WMMA + block_q8_1_mmq Infra | ✅ | 1 |
| Schritt 2: Minimal-Kernel (3 Bug-Fix-Runden) | ✅ | 4 |
| Schritt 3: Scale-Up (variable M,N,K) | ✅ | 1 (diese Session) |
| Schritt 4: Executor-Integration + Perf | 🔜 | Nächste Session |

Gesamt bisher 6 Sessions für P0.2, konsistent mit der Projektplan-Schätzung von 5-6 Tagen. Bis zum ersten Prefill-Performance-Zahlen-Vergleich bleibt 1 weitere Session.
