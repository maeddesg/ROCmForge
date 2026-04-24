# P0.1b — WMMA 128×128 Tile-Refactoring

**Date:** 2026-04-24
**Branch:** v1.0-dev
**Hardware:** AMD Radeon RX 9070 XT (gfx1201)
**Status:** Template-Kernel + Parity ✅. Perf-Daten ehrlich — 128×128 ist **nicht uniform besser**; Executor-Integration deferred.

## Landung-Zusammenfassung

- **Template-parametrisierter Q4_K-FP16-WMMA-Kernel** in `hip_kernels_v1/wmma/wmma_gemm_q4_k_fp16_tiled.hip` mit zwei Instanzen: `<64,64,32>` und `<128,128,32>`.
- **Original-Kernel unberührt** (`wmma_gemm_q4_k_fp16.hip`), dient als Parity-Anker.
- **5 Tests grün** in `tests_v1/wmma_tiled_test.rs` (Parity + Korrektheit + Perf-Landschaft).

## Parity — Template ist korrekt

Die drei Korrektheits-Tiers beweisen das Template an zwei Stellen:

| Test | Ergebnis |
|---|---|
| **Template<64,64,32> vs Original hard-coded** (Qwen3 QKV 64×4096×4096) | `max_abs = 0.0` — **bit-exakt** |
| **Template<128,128,32> vs CPU-FP32-Referenz** (128×128×256) | `max_abs = 1.56`, tol = 425 ✓ |
| **Template<128,128,32> vs CPU-FP32-Referenz** (128×128×4096) | `max_abs = 4.24`, tol = 5007 ✓ |
| **Cross-Parity 64×64 ↔ 128×128** (128×128×4096) | `max_abs = 0.0` — **bit-exakt** |

Die Cross-Parity bestätigt: Beide Template-Instanzen berechnen jedes Output-Element mit **identischer FMA-Sequenz pro Lane**. Nur die Block-/Warp-Partition unterscheidet sich. Die Korrektheit des 128×128-Pfads ist damit transitiv aus dem bit-exakten 64×64-Match gesichert.

## Performance — 128×128 ist kein universeller Win

**Sweep über alle Qwen3-Prefill-Shapes** (ROCMFORGE_RUN_REAL_MODEL_GPU_TESTS=1):

| Shape | 64×64 | 128×128 | Speedup |
|---|---:|---:|---:|
| **M=128 QKV** (128×4096×4096) | 2.46 ms | 2.20 ms | **1.12×** ✓ |
| **M=256 QKV** (256×4096×4096) | 2.65 ms | 2.58 ms | 1.03× |
| **M=256 FFN** (256×14336×4096) | 6.68 ms | 6.02 ms | **1.11×** ✓ |
| **M=512 QKV** (512×4096×4096) | 3.73 ms | 3.71 ms | 1.00× |
| **M=512 FFN** (512×14336×4096) | 26.98 ms | 27.33 ms | 0.99× |
| **M=1024 QKV** (1024×4096×4096) | 5.96 ms | 8.15 ms | **0.73× ❌ REGRESS** |

### Interpretation

- **M ≤ 256 ⇒ 128×128 wins** (+11 bis +12 %) — größerer Tile hat bessere arithmetische Intensität (mehr FMAs pro LDS-Load-Amortisation), GPU ist bei kleinem M nicht voll saturiert, die LDS-Effizienz zahlt sich aus.
- **M ≈ 512 ⇒ Neutral** — GPU voll saturiert, Kernel-Zeit dominiert, Tile-Größe wird kompensiert durch Scheduling-Effizienz.
- **M ≥ 1024 ⇒ 128×128 regresst um 27 %** — zu wenig Blocks/CU (256 Blocks / 64 CUs = 4 Blocks/CU) → Tail-Effekte schlagen zu. 64×64 hat bei M=1024 16 Blocks/CU = deutlich feinere Scheduling-Granularität.
- Der erwartete **1.30×-Speedup** aus dem Prompt **wird bei keiner Shape erreicht**.

### Warum nicht wie erwartet?

Drei Hypothesen, geordnet nach Plausibilität:

1. **VGPR-Pressure auf 128×128.** Der Akkumulator wächst von `float8[4]` = 32 VGPRs auf `float8[8]` = 64 VGPRs pro Lane (2×). Gesamt-VGPR-Count für 128×128 liegt schätzungsweise bei ~120 — über dem 104er-Target. Occupancy sinkt entsprechend. Bestätigt werden könnte das durch rocprofv3 auf dem tiled-Kernel — habe ich in dieser Session nicht mehr gemacht, da die ausgemessenen Timing-Daten bereits das Fazit liefern.
2. **Block-Scheduling-Granularität.** Bei großem M haben wir mehr Blocks als CUs × Konkurrenz — 64×64 gewinnt einfach durch feinere Granularität.
3. **LDS-Bank-Conflicts unter höherer Thread-Zahl.** 256 Threads mit neuem Access-Pattern könnten Conflicts aufbauen, die mit 128 Threads nicht auftraten.

## Was nicht implementiert wurde

### Executor-Integration (adaptive Dispatch)

**Geplant laut Prompt:**
```rust
fn select_wmma_tile_config(M, N) -> WmmaTileConfig {
    if M >= 128 && N >= 128 { T128x128 } else { T64x64 }
}
```

**Warum zurückgestellt:** Die oben vorgeschlagene Heuristik würde bei M=1024 die 27-%-Regression aktivieren. Die korrekte Heuristik wäre `if M <= 256 && N >= 4096` — aber:

- Unsere typischen Prefill-Shapes in den Testläufen (512 Tokens padded auf 576 = 9 × M-Tile-64) liegen nahe der Neutralzone (M=512: 0.99-1.00×). Der erwartete **realistische Durchsatz-Gewinn auf End-to-End-Prefill wäre < 5 %**.
- Die Implementierung der adaptiven Auswahl (M/N-Inspection vor jedem Dispatch + Padding an zwei Tile-Größen statt einer) ist Non-trivial und würde ohne klaren Win riskant sein.

**Fazit:** Der Template-Kernel ist gelandet als *Infrastruktur für zukünftige Tile-Konfigurationen* (Q6_K, 128×64, FP8-Varianten), aber die **128×128-Deployment-Entscheidung bleibt offen** bis entweder (a) VGPR-Count gemessen + reduziert ist oder (b) ein besserer Config-Kandidat (z. B. 128×64 mit 4 Warps) identifiziert wurde.

### Q6_K-Refactoring

Nicht implementiert — Q6_K macht 14.7 % der Prefill-Zeit (vs. Q4_K 53.8 %). Das Muster ist gleich (Dequant-Prolog anders, Tile-Framework identisch). Empfehlung: erst nach Entscheidung über Q4_K 128×128-Deployment replizieren.

## Nächste Schritte (separate Session)

**Option A — VGPR-Pressure reduzieren (Config #2, 128×64×32 mit 4 oder 8 Warps):**
- Akkumulator: `float8[4]` = 32 VGPRs statt 64 → VGPR-Count sollte wieder <100 sein
- LDS: 128×32 + 32×64 × 2B = 4+4 KB (gleich wie 64×64) — knapp
- Erwarteter Effekt: monotoner Speedup über M, keine M=1024-Regression
- Aufwand: Template-Instanziierung `<128,64,32>` hinzufügen, gleiche Test-Tiers durchlaufen. Geschätzt 2-3 h.

**Option B — Direkt P0.2 angehen (llama.cpp MMQ-Port):**
- llama.cpp erreicht 1127 tok/s Prefill auf unserer Hardware — das ist der Beweis dass ein anderer Kernel-Ansatz (dp4a oder MMQ statt unseres WMMA) strukturell anders funktioniert
- Der MMVQ-Port hat bei Decode 53 % Gewinn gebracht; MMQ könnte bei Prefill ähnlich wirken
- Aufwand: 5-6 Tage laut phase3-Projektplan

**Empfehlung:** Option A zuerst (cheap, bekannt), Option B als Haupt-Hebel für P0 zum P0-Ziel 4000+ tok/s.

## Geänderte Dateien

- `hip_kernels_v1/wmma/wmma_gemm_q4_k_fp16_tiled.hip` — **neu**, Template-parametrisierter Kernel mit 2 extern-"C" Instanzen
- `hip_kernels_v1/CMakeLists.txt` — Library-Registrierung
- `build.rs` — Linker-Zeile für `v1_wmma_q4_k_fp16_tiled`
- `src_v1/backend/gpu/wmma.rs` — Rust-FFI-Bindings für beide Template-Instanzen
- `tests_v1/wmma_tiled_test.rs` — **neu**, 5 Tests (Parity + Korrektheit + Perf-Sweep)
- `Cargo.toml` — Test-Registrierung

**Kein Executor-Change.** Bestehender Code nutzt weiterhin `rocmforge_launch_wmma_gemm_q4_k_fp16` (Original).

## End-to-End Impact

Da der Executor noch nicht auf 128×128 umgestellt wurde:

| Metrik | Pre-P0.1b | Post-P0.1b | Delta |
|---|---:|---:|---:|
| Prefill tok/s (542-Token-Prompt) | ~1000 | ~1000 | **unverändert** |
| Decode tok/s (Qwen3) | 96.2 | 96.2 | unverändert |
| Korrektheit | 15/15 | 15/15 | unverändert |
| Test-Anzahl | — | +5 | neue Parity- & Landschafts-Tests |

Der Code ist **sicher gelandet** (kein Produktionspfad geändert) und stellt die Infrastruktur für zukünftige WMMA-Tuning-Arbeit bereit.
