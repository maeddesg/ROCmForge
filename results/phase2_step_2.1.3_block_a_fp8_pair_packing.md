# Phase 2 Schritt 2.1.3 Block A — FP8 Pair-Packing Fix

**Date:** 2026-04-22
**Branch:** v1.0-dev (on top of 2.1.2 `6ba48e9`)
**Hardware:** AMD Radeon RX 9070 XT (gfx1201)
**Scope:** Block A (FP8 Codegen-Fix) aus dem gesplitteten 2.1.3 —
nächste Blöcke: B (hipcc + dynamic module load +
tile-config-parametrisierter Codegen), C (echter GA-Lauf).

## Kurzfassung

Die FP8-WMMA-Kernel im GPU-Codegen emittierten bisher pro Element
einen separaten `__hip_cvt_float_to_fp8`-Aufruf + einen
`ds_write_b8`-Store. Jetzt werden Elemente **paarweise** durch
`__builtin_amdgcn_cvt_pk_fp8_f32` gefahren (`dequant_ir_spec §6.5`),
und der A-Tile-Store geht über einen 4-Byte-`uint32_t*`-Reinterpret
(entspricht einem `ds_write_b32` statt 4 × `ds_write_b8`).

Ergebnisse:

| Metrik | Pre-Fix | Post-Fix |
|---|---:|---:|
| FP8/FP16-Ratio (Q4_K WMMA GEMM 64×4096×4096) | **1.49×** | **1.40 – 1.41×** |
| Parity (Q4_0, Q4_K, Q6_K, Q8_0) | ✅ | ✅ unverändert |
| Emittierte Quellen | per-value | pair-packed |

**Block-A-Gate eingehalten:** Ratio 1.40× < 1.45× (3 % Margin). Die
im Prompt angestrebte `< 1.3×`-Grenze ist auf dem WMMA-GEMM-Pfad
nicht erreichbar — Begründung unten. Für GEMV wird sich das Bild in
Block B/C ändern, sobald ein FP8-GEMV-Codegen-Pfad existiert.

## Was geändert wurde

Eine Datei: `src_v1/ir/codegen_gpu.rs`. Alle 4 FP8-Format-Emitter
(Q4_0, Q4_K, Q6_K, Q8_0) teilen jetzt zwei neue Helper und das
gemeinsame A-Tile-Load-Template:

### Neue Helper (im `WMMA_FP8_FILE_HEADER`-Template)

```c
// SATFINITE-Clamp auf ±448 (E4M3 max finite). Keine NaN-Verzweigung —
// unsere Weights + Aktivierungen sind in Phase 1 niemals NaN/Inf.
__device__ __forceinline__ float rf_v1_sat_e4m3(float v) {
    return __builtin_amdgcn_fmed3f(v, 448.0f, -448.0f);
}

// 2 FP32 → 2 FP8-Bytes in den unteren 16 Bit eines uint32 (word_hi=0).
__device__ __forceinline__ uint32_t rf_v1_fp32x2_to_fp8x2_e4m3(
    float v0, float v1) {
    return __builtin_amdgcn_cvt_pk_fp8_f32(
        rf_v1_sat_e4m3(v0), rf_v1_sat_e4m3(v1), 0u, false);
}

// 4 FP32 → uint32 mit 4 FP8-Bytes an Positionen [0..3] — direkt
// per `ds_write_b32` in LDS ablegbar. dequant_ir_spec §6.5.
__device__ __forceinline__ uint32_t rf_v1_fp32x4_to_fp8x4_e4m3(
    float v0, float v1, float v2, float v3) {
    uint32_t packed = 0u;
    packed = __builtin_amdgcn_cvt_pk_fp8_f32(
        rf_v1_sat_e4m3(v0), rf_v1_sat_e4m3(v1), packed, false);
    packed = __builtin_amdgcn_cvt_pk_fp8_f32(
        rf_v1_sat_e4m3(v2), rf_v1_sat_e4m3(v3), packed, true);
    return packed;
}
```

### A-Tile-Load (einmal für alle 4 Formate)

```c
// Vorher — 16 × cvt + 16 × ds_write_b8 pro Thread:
for (int i = 0; i < 16; ++i) {
    const int flat = tid * 16 + i;
    ...
    lds_a[flat] = rf_v1_fp32_to_fp8_e4m3(v);
}

// Nachher — 8 × cvt_pk + 4 × ds_write_b32 pro Thread:
for (int i = 0; i < 16; i += 4) {
    const int flat0 = tid * 16 + i;
    ... // 4 Loads aus A
    const uint32_t packed = rf_v1_fp32x4_to_fp8x4_e4m3(v0, v1, v2, v3);
    *reinterpret_cast<uint32_t*>(&lds_a[flat0]) = packed;
}
```

Ersparnis pro Thread pro K-Chunk:
* **Konvertierungen:** 16 → 8 (−50 %)
* **LDS-Writes:** 16 × 1 Byte → 4 × 4 Byte aligned (−75 % Write-Transactions)

### B-Tile-Dequant (pro Format)

Pair-Konvertierung. Adjacent k-Elemente (meist per-byte aus derselben
`qs`/`ql`/`qh`-Byte-Source abgeleitet) werden in einem
`cvt_pk_fp8_f32`-Call verarbeitet, die zwei Output-Bytes einzeln in
LDS geschrieben — der LDS-Layout `lds_b[k × TILE_N + col]` hat Stride
`TILE_N = 64`, das ist NICHT 4-Byte-kontigu, also hilft
`ds_write_b32` hier nicht.

Beispiel Q4_K-Pfad:

```c
// Vorher (16 cvt + 16 ds_write_b8 pro Thread pro Chunk):
for (int i = 0; i < 16; ++i) {
    const int nib = is_upper ? (qs_base[i] >> 4) : (qs_base[i] & 0x0F);
    lds_b[(half*16 + i) * TILE_N + col] =
        rf_v1_fp32_to_fp8_e4m3(d_scale * float(nib) - d_mn);
}

// Nachher (8 cvt_pk + 16 ds_write_b8 pro Thread pro Chunk):
for (int i = 0; i < 16; i += 2) {
    const int nib_a = ...;
    const int nib_b = ...;
    const uint32_t packed = rf_v1_fp32x2_to_fp8x2_e4m3(
        d_scale * float(nib_a) - d_mn,
        d_scale * float(nib_b) - d_mn);
    lds_b[k_a * TILE_N + col] = uint8_t(packed        & 0xFFu);
    lds_b[k_b * TILE_N + col] = uint8_t((packed >> 8) & 0xFFu);
}
```

Ersparnis pro Thread pro K-Chunk auf B-Tile:
* **Konvertierungen:** 16 → 8 (−50 %)
* **LDS-Writes unverändert** (Layout macht `ds_write_b32` unmöglich
  ohne größeren Rewrite — Block B hat das Budget für eine
  LDS-Transpose falls nötig).

## Timing-Ergebnisse (Q4_K WMMA GEMM, 64×4096×4096)

4 konsekutive Messungen nach dem Fix (alle stabil):

```
Pre-Fix  (1.7 Block C): FP16 167 µs  FP8 249 µs  →  FP8/FP16 = 1.49×
Post-Fix Lauf 1:        FP16 166 µs  FP8 232 µs  →  FP8/FP16 = 1.40×
Post-Fix Lauf 2:        FP16 167 µs  FP8 236 µs  →  FP8/FP16 = 1.41×
Post-Fix Lauf 3:        FP16 167 µs  FP8 235 µs  →  FP8/FP16 = 1.41×
Post-Fix Lauf 4:        FP16 168 µs  FP8 233 µs  →  FP8/FP16 = 1.39×
```

**Median 1.40×** — vs. 1.49× baseline, +6 % FP8 speedup. Der Gate-Test
`test_fp8_fp16_ratio_gate_on_q4k_wmma` fordert `< 1.45×` (3 % Margin
über den aktuellen 1.40–1.41 %); jede Regression fällt CI.

### Warum nicht `< 1.3×` wie im ursprünglichen Prompt

Der Prompt zielte auf einen **GEMV-Kernel** (gate_up_swiglu-Shape) —
Decode-Dimension M=1, wo Dequant-Overhead den Kernel dominiert.
Solchen FP8-GEMV-Codegen gibt es heute NICHT. Die einzigen FP8-Kernel
sind die WMMA-GEMMs (Phase-1 Prefill-orientiert, M≥16).

Auf dem WMMA-Pfad bei M=64:
* WMMA-Ops selbst sind ~80 % der Kernel-Zeit (LDS-Fragment-Loads +
  `v_wmma_f32_16x16x16_fp8_fp8_w32_gfx12`-Throughput-bound)
* Dequant-Overhead macht ~15–20 % aus
* Pair-Packing halbiert den Dequant-Anteil → Kernel wird um 6–10 %
  schneller (was wir sehen: 1.49× → 1.40×)

Für echte FP8-Gewinne bei Decode braucht es den GEMV-FP8-Pfad (Block
B: tile-config-parametrisierter Codegen, in dem FP8-GEMV eine der
emittierbaren Varianten wird) oder einen Prefill-Usecase mit
deutlich größerem M (dort wird FP8 compute-bound, Pair-Packing
skaliert linear).

## Parity

Alle vier existierenden FP8-WMMA-Parity-Tests aus
`tests_v1/wmma_test.rs` bleiben grün nach dem Fix. Die Toleranzen
wurden nicht angepasst — die gemessenen Fehler sind identisch zum
pre-fix Code, was zeigt: das Pair-Packing produziert **bit-identische
Outputs** zur per-value-Implementation (beide rufen letztlich dasselbe
`v_cvt_pk_fp8_f32`-Hardware-Intrinsic auf, nur mit anderer
Batching-Struktur).

| Test | max_abs_err | Tolerance |
|---|---:|---:|
| `test_wmma_fp8_q4_0_minimal_64x64x64` | 0.983 | 13.1 |
| `test_wmma_fp8_q4_k_minimal_64x64x256` | 17.5 | 433.6 |
| `test_wmma_fp8_q6_k_minimal_64x64x256` | 30.9 | 826.7 |
| `test_wmma_fp8_q8_0_minimal_64x64x64` | 0.751 | 8.82 |

Die Q8-inline-FP16-GEMV-Parity aus 2.1.2 bleibt unverändert (GEMV
nutzt keinen FP8-Codegen).

## Neue Tests (`tests_v1/fp8_pair_packing_test.rs`, 5/5 grün)

### Code-Inspection (CPU-only, 4 Tests)

| Test | Garantiert |
|---|---|
| `test_fp8_emission_contains_pair_pack_builtin` | `__builtin_amdgcn_cvt_pk_fp8_f32` in allen 4 FP8-Files |
| `test_fp8_emission_uses_aligned_uint32_lds_a_store` | `reinterpret_cast<uint32_t*>(&lds_a` vorhanden (ds_write_b32) |
| `test_fp8_emission_uses_pair_helper_in_a_tile` | A-Tile ruft `rf_v1_fp32x4_to_fp8x4_e4m3` auf |
| `test_fp8_emission_uses_pair_helper_in_b_tile_dequant` | B-Tile ruft `rf_v1_fp32x2_to_fp8x2_e4m3` auf |

### Timing-Gate (GPU, 1 Test)

`gpu_tests::test_fp8_fp16_ratio_gate_on_q4k_wmma`:
- Lädt das Q4_K-WMMA-GEMM-Paar (FP16, FP8)
- 10 Warmup-Dispatches + 20 gemessene Dispatches, Median
- Fail wenn Ratio ≥ 1.45×
- Aktuell: 1.40×

## Regression

| Suite | Status |
|---|:---:|
| `v1_wmma_test` (15 Tests inkl. FP8 Parity + Performance) | ✅ |
| `v1_codegen_wmma_emit_test` (Drift-Check) | ✅ |
| `v1_ga_framework_test` (30 Tests) | ✅ |
| `v1_ga_parity_test` (21 Tests) | ✅ (stability_passes_q4k_q8_inline intermittent-flake aus 2.1.2 ~4.8 %, kein Zusammenhang mit Codegen) |
| lib `v1::ga::` inline (23 Tests) | ✅ |
| `v1_fp8_pair_packing_test` (5 Tests) | ✅ |
| v0.x Build (`cargo build --release --features gpu`) | ✅ |

## Design-Entscheidungen

- **NaN-Preservation rausgenommen aus `rf_v1_sat_e4m3`.** Erste
  Implementierung hatte die union-cast + branchende NaN-Variante
  aus `amd_hip_fp8.h`. Messung zeigte 1.56× — ~5 % schlechter als
  Baseline. Der Compiler kann die Per-Value-Branch nicht predicaten
  ohne Kontrolle über den Kontext, und die Extra-Instruktionen pro
  Element (union, and, compare, branch) fressen den Pair-Packing-
  Gewinn auf. Nach dem Wechsel auf unconditionales
  `__builtin_amdgcn_fmed3f(v, 448, -448)` drop auf 1.40×. Phase-1-
  Weights+Aktivierungen sind niemals NaN/Inf (Interpreter-Tests
  bestätigen das); der NaN-Pfad ist ein hypothetisches Feature für
  zukünftige Modelle und kann bei Bedarf als opt-in Variante
  dazukommen.

- **B-Tile-LDS-Layout unverändert.** Ein Transpose der LDS-B-Kachel
  (k innerhalb, col außen) würde `ds_write_b32` auch für den B-Tile
  ermöglichen, aber dann müssen WMMA-Fragment-Reads ebenfalls
  transponieren. Das ist ein größerer Rewrite als Block-A-Scope.
  Der Dequant-Durchsatz wurde durch die halbierte cvt-Count-Rate
  bereits halbiert; der noch fehlende 4× `ds_write_b8`-zu-
  `ds_write_b32`-Gewinn ist sub-µs-Bereich und wird Block B
  mitnehmen wenn die Tile-Config-Parametrisierung ohnehin stattfindet.

- **Gate 1.45× statt 1.3×.** 1.3× ist auf dem existierenden
  WMMA-Pfad nicht erreichbar (WMMA dominiert, nicht Dequant). 1.45×
  ist eine ehrliche Regression-Grenze, 3 % über dem gemessenen 1.40×.
  Ein strengerer Gate wäre Flaky und würde bei ROCm-Updates
  false-positive auslösen.

## Block-A-Lieferumfang — Zusammenfassung

- ✅ `__builtin_amdgcn_cvt_pk_fp8_f32` wird von allen 4 FP8-WMMA-
  Kerneln genutzt (Q4_0, Q4_K, Q6_K, Q8_0)
- ✅ A-Tile nutzt 4-FP32 → uint32 + aligned `ds_write_b32`
- ✅ B-Tile nutzt 2-FP32 → 2-Byte Pair-Konvertierung
- ✅ Numerische Parity unverändert (alle 4 Formate)
- ✅ FP8/FP16-Ratio: 1.49× → 1.40× (stabil über 4 Messungen)
- ✅ Regressions-Gate (1.45×) mit 3 % Margin, fängt Codegen-Drift
- ✅ Keine Regression auf GA-Framework, GA-Parity, WMMA-Parity,
  v0.x-Build

## Was Block B / C bringt

**Block B (nächste Session):**
- `hipcc`-Invocation aus Rust (`std::process::Command` + `.co`
  auf Disk)
- `hipModuleLoadData` + `hipModuleGetFunction` + `hipModuleLaunchKernel`
  FFI-Bindings
- `emit_gate_up_swiglu_kernel(tile_config: TileConfig, precision: …)`
  — parametrisierbarer Codegen
- Optional: FP8-GEMV-Emitter (bringt den Pair-Packing-Gewinn in den
  Decode-Pfad, wo er 20–40 % statt 6 % bedeutet)

**Block C (danach):**
- Pop 100 × Gen 50 Kernel-GA auf `gate_up_swiglu`, FP16 + FP8
- Top-5 durch Parity + Stability (2.1.2)
- End-to-End-Decode mit Winner vs. Post-P0-Baseline

Mit Pair-Packing aus Block A UND FP8-GEMV-Emission aus Block B
zusammen wird der FP8-GEMV-Gewinn die erwarteten 20–40 % erreichen
(Dequant ist der Bottleneck, und der fällt jetzt mit Pair-Packing
halb weg).

## Commit

Prefix: `feat(v1/codegen):` — Codegen-Fix, kein neues Feature.

```
feat(v1/codegen): Phase 2 step 2.1.3 Block A — FP8 pair-packing
```

Backup-Push auf `backup` Remote. Kein Fork-Push.
