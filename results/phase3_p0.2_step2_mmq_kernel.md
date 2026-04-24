# P0.2 MMQ-Port Schritt 2 — Q4_K Integer-WMMA Prefill-Kernel

**Date:** 2026-04-24
**Branch:** v1.0-dev
**Status:** **WIP** — Minimal-Kernel (M=N=16, K=256) gelandet, kompiliert, **produziert aber falsche Ergebnisse**. Ehrlicher Scope-Out.

## TL;DR

Ein minimaler Proof-of-Concept-Kernel (`hip_kernels_v1/wmma/mmq_q4_k_minimal.hip`) für die einfachste Shape (16×16×256, ein Warp, ein Block) ist implementiert und landet in die Build-Pipeline. Parity-Test läuft durch ohne Crash, aber der Output weicht massiv von der CPU-Referenz ab (max_rel 116%, NaN in einzelnen Zellen). Der Kernel ist als WIP-Scaffold committed; Korrektheit braucht eine dedicated Debug-Session.

## Was implementiert wurde

**Datei:** `hip_kernels_v1/wmma/mmq_q4_k_minimal.hip` (~210 LOC)

1. **Helper `unpack_scales_q45_K`** — 1:1 aus `mmq.cuh:2141-2149` übernommen. Packt die 12-Byte 6-bit-Scales in int32-Arrays.
2. **Q4_K Nibble-Unpack** — `qs[byte]` → low/high Nibble als int8 (0..15), in int32 gepackt für WMMA.
3. **Integer-WMMA Calls** — Zwei `__builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12` pro Sub-Block (8 Sub-Blöcke × 2 Calls = 16 WMMA-Calls total für K=256), Fragment-Layout wie in `wmma_i32_smoke.hip`.
4. **Scale-Fixup** — `sum[n,m] += d_A * sc[sb] * d_B[sb] * raw_acc - d_A * dmin * m[sb] * sum_B[sb]` pro Sub-Block, einmal am Ende.
5. **FFI + Test-Harness** — `rocmforge_launch_mmq_q4_k_minimal`, vollständiger Parity-Test vs `dequant_block + naive GEMM` CPU-Referenz.

**Build:** `cargo build --release --features "v1 gpu"` erfolgreich. Kernel linkt gegen `v1_mmq_q4_k_minimal` static-lib.

## Was (noch) NICHT funktioniert

Testlauf-Ausgabe (max_rel 116%, max_abs 6.06×10⁹, NaN-Zellen):

```
MMQ-Q4_K minimal @ 16×16×256: max_abs=6.0575e9, max_rel=1.1648e0
First 8 GPU/CPU pairs:
  (43585084.0,    773.92)      // 56000× zu groß
  (193736060.0,   348.94)
  (-2619854800.0,-290.57)
  (NaN,          -372.94)      // NaN!
  (-3363288.8,  -1499.18)
  (NaN,         -1611.91)
```

Erwartete Größenordnung für diese Test-Daten (Weights ∈ [-7.5, 7.5], Activations ∈ [-1, 1], K=256): Output-Magnituden ~100. Beobachtet: 10⁷ bis 10⁹ plus NaN in einzelnen Zellen.

## Verdachts-Diagnose

Drei plausible Ursachen, keiner ist diese Session verifiziert:

1. **Fragment-Lane-Layout-Mismatch.** Das im `wmma_i32_smoke.hip` verifizierte Layout gilt für einen dedicated-purpose Smoke-Kernel. llama.cpp's `load_generic` in `mma.cuh` macht den Load anders, und ich habe die Semantik der Lane-Zuordnung im mmq_q4_k-Kernel nach eigener Konvention gebaut anstatt 1:1 llama.cpp's Pattern zu übernehmen. Wenn die per-Lane-Verteilung der int8-Werte innerhalb der 16×16 A-Matrix nicht matching dem WMMA-Intrinsic-Layout ist, sind die int32-Akkumulator-Ergebnisse Garbage. **Dies ist der wahrscheinlichste Fehler.**

2. **VGPR-Overflow + Spilling.** `int32x8_t acc_sb[8]` = 64 int32 per Lane nur für Akkumulatoren, plus A/B Fragmente + temporäre Arrays (`sc_all[8][8]`, `mn_all[8][8]`) — gesamt-VGPR-Bedarf wahrscheinlich >128 per Lane. Das ist über dem 104-VGPR-Budget; Compiler spillt in Scratch, und das 16×16 schreibt in Adressen, die nicht die erwarteten sind. Würde mit Pre-Load-Reduktion (Scales pro `slot` on-the-fly statt in Arrays) entlastet.

3. **Unsigned/Signed Nibble-Konvertierung.** Q4_K-Nibbles sind per Definition 0..15 (unsigned). Ich packe sie als int8 in int32, und das Intrinsic bekommt `a_signed=true`. Für Werte 0..15 ist das egal, aber bei fehlerhafter Sign-Extension (z.B. wenn ein Nibble versehentlich mit Vorzeichen-erhaltend geshiftet wird) könnten 0x80 → -128 erscheinen. Weniger wahrscheinlich als #1 und #2.

## Nächste Debug-Schritte (separate Session)

1. **Zuerst: Korrektheit ohne Scales.** Setze alle A-Scales auf `1.0`, alle Mins auf `0.0`. Dann muss `raw_acc` == `Σ nibble[n,k] * int8_B[m,k]` sein, verifizierbar gegen skalare Int-Referenz. Wenn das NICHT stimmt → Fragment-Layout-Bug (#1). Wenn stimmt → Scale-Math-Bug (#3).
2. **Falls #1: llama.cpp's `load_generic` direkt portieren.** In `mma.cuh` steht der exakte Lane→Byte-Offset-Algorithmus. Unser Smoke-Test hat ein DIFFERENT Lane-Layout verifiziert; wir müssen llama.cpp's Layout verwenden weil der Kernel's Scale-Fixup auf dem tile_C::get_i/get_j-Pattern basiert.
3. **VGPR-Check via `--save-temps`.** Compile mit `-mllvm -amdgpu-dump-hsa-metadata` und prüfe den VGPR-Count. Falls >104: Scale-Arrays aus dem Register-File raus, on-the-fly berechnen.
4. **Minimum-reduzierter Test:** Single-Sub-Block (K=32 statt 256). Eliminiert Sub-Block-Loop als Fehlerquelle.

## Gelandete Dateien

- `hip_kernels_v1/wmma/mmq_q4_k_minimal.hip` — **WIP** Kernel (kompiliert, rechnet falsch)
- `hip_kernels_v1/CMakeLists.txt` — Library-Registrierung
- `build.rs` — Linker-Zeile
- `src_v1/backend/gpu/wmma.rs` — FFI-Binding
- `tests_v1/mmq_q4_k_minimal_test.rs` — Parity-Test (failt aktuell mit clear error message)
- `Cargo.toml` — Test-Registrierung

## End-to-End Status

| Metrik | Vor Schritt 2 | Nach Schritt 2 |
|---|---:|---:|
| Prefill tok/s | ~1000 | ~1000 (unverändert — Kernel nicht in Dispatch) |
| Decode tok/s | 96.2 | 96.2 (unverändert) |
| Korrektheit (Qwen3 15/15) | ✅ | ✅ (kein Produktionspfad geändert) |
| MMQ-Kernel-Korrektheit | N/A | **❌ max_rel 116%** |

## Ehrliche Einschätzung

Diese Session hat die ursprünglich-geplante „1-2 Sessions" für Schritt 2 realistisch auf **3-4 Sessions** reskaliert:

- Session-diese: Kernel-Scaffold + erste Tests → Kernel kompiliert, aber produziert falsche Ergebnisse
- Session-next-1: Debugging mit simplified-scales-Test, Fragment-Layout-Vergleich vs llama.cpp's `load_generic`, vermutlich finden des Bugs
- Session-next-2: Full-Shape Skalierung (von 16×16×256 auf 128×128×K-variable) und Integration in Prefill-Pfad
- Session-next-3: rocprof, Perf-Messung, Scale-K-Variabilität

Das ist immer noch schneller als der Tile-Tuning-Weg (P0.1b: max 1.14× auf einigen Shapes, keine Transformation), aber deutlich aufwändiger als zunächst budgetiert. Die Analyse-Phase (P0.2 Session 1) hat den Pfad korrekt identifiziert; die eigentliche Implementierung zeigt jetzt typische Kernel-Complexity.

**Empfehlung nächste Session:** Den simplified-scales-Debug-Test zuerst ausführen, um das Root-Cause zu isolieren. Dann Fix → Korrektheit → Skalierung.
