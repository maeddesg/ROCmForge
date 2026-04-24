# P0.2 MMQ-Port Schritt 1 — Integer-WMMA + block_q8_1_mmq

**Date:** 2026-04-24
**Branch:** v1.0-dev
**Hardware:** AMD Radeon RX 9070 XT (gfx1201, RDNA4)
**Status:** Beide Deliverables gelandet, alle Tests grün.

## TL;DR

- **Integer-WMMA (`__builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12`) funktioniert auf gfx1201.** Bit-exakt vs skalare CPU-Referenz für signed × signed int8 → int32 GEMM.
- **`block_q8_1_mmq` Quantize-Kernel** produziert für 32 Blöcke × 4 Sub-Blöcke = 128 Sub-Blocks gegen CPU-Referenz: **0 Fehler** in Scales, Sums und qs-Bytes.
- **Infrastruktur-Voraussetzung für MMQ-Port Schritt 2 ist damit erfüllt.**

## Deliverable 1: Integer-WMMA Smoke

**Datei:** `hip_kernels_v1/wmma/wmma_i32_smoke.hip`
**Intrinsic:** `__builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(a_signed, A, b_signed, B, acc, clamp)`

### Go/No-Go: Compile-Test

```bash
$ cargo build --release --features "v1 gpu" --bin rocmforge-v1
    Finished `release` profile [optimized] target(s) in 2.08s
```

**Kompiliert ohne Fehler auf gfx1201** — der Intrinsic existiert im installierten hipcc-Compiler und ist für diese Architektur verfügbar.

### Lane-Layout (aus mma.cuh abgeleitet und validiert)

Für Wave32 auf RDNA4:
- **A-Fragment:** Lane `l` hält 8 int8 Werte = 2 int32 = `A[row_in_half, half*8 .. half*8+7]` wobei `row_in_half = l & 15` und `half = l >> 4`.
- **B-Fragment:** col-major Layout, Lane `l` hält `B[half*8 .. half*8+7, col_in_half]`.
- **C-Akkumulator:** Lane `l` hält Spalte `l & 15` und 8 aufeinanderfolgende Zeilen `(l >> 4) * 8 + 0 .. 7`.

Das Layout wurde verifiziert durch zwei Tests:
- **`int_wmma_smoke_identity_by_counting`**: A = Identity, B = signed counting-pattern → C bit-exakt mit skalarer Referenz.
- **`int_wmma_smoke_random_signed`**: A und B random int8 ∈ [-32, 31] → C bit-exakt mit skalarer Referenz.

## Deliverable 2: block_q8_1_mmq Quantize-Kernel

**Datei:** `hip_kernels_v1/quantize/quantize_q8_1_mmq.hip`

### Struct-Layout (identisch zu llama.cpp)

```c
struct block_q8_1_mmq {
    __half2 ds4[4];        // 4 × (d=scale, s=sum) pro 32-Elem Sub-Block
    int8_t  qs[128];       // 128 quantisierte Werte
};
static_assert(sizeof(block_q8_1_mmq) == 144);
```

- **144 Bytes** pro 128-Elem Block (verifiziert über `std::mem::size_of`)
- **Offset-Layout:** ds4 @ 0, qs @ 16 (Rust-seitiger Offset-Check)
- **DS4 Layout:** d und sum als `__halves2half2` gepackt — Q4_K-kompatibel

### Quantize-Algorithmus

Ein Block = 32 Threads (eine Wave), jeder Thread verarbeitet 4 floats via `float4` vectorisierte Loads. 8 Threads bilden ein 32-Element-Sub-Block.

```cpp
sub_block = tid / 8;       // 0..3
sub_lane  = tid & 7;       // 0..7
base      = mmq_block * 128 + tid * 4;

xi = float4 load from global
amax = max(|xi.x|, |xi.y|, |xi.z|, |xi.w|)
sum  = xi.x + xi.y + xi.z + xi.w

// Reduce across 8 lanes per sub-block
amax = warp_reduce_max_xor(amax)   // offsets 4, 2, 1
sum  = warp_reduce_sum_xor(sum)

d_inv = 127.0 / amax   // or 0 if amax == 0
q.x/y/z/w = round(xi.x/y/z/w * d_inv) clamped to int8

store qs[tid*4 .. tid*4+3] as char4
if sub_lane == 0:
    ds4[sub_block] = half2(1.0/d_inv, sum)
```

### Parity-Test gegen CPU-Referenz

```
quantize_q8_1_mmq: 32 blocks × 4 sub-blocks = 128 Sub-Blocks gegen CPU-Referenz
  err_scales = 0   (d und sum exakt im half-Rundungs-ULP)
  err_qs     = 0   (128 × 32 = 4096 int8-Werte, 0 Abweichungen ±1)
```

**Bit-Perfekt** — der GPU-Kernel produziert für jeden Sub-Block:
- `d` identisch zur Python-Referenz `1.0 / (127.0 / max(|x|))`
- `sum` identisch zu `Σ x[i]` (nach half-Konvertierung)
- `qs[i]` identisch zu `round(x[i] * 127 / max(|x|))` clamped to int8

### Edge-Cases

- **`quantize_q8_1_mmq_zero_input_safe`**: All-zero input (amax=0) → d=0, sum=0, qs=0 ohne NaN/Inf. ✓

## Tests

| Test | Zweck | Ergebnis |
|---|---|---|
| `block_q8_1_mmq_has_expected_size` | Struct-Layout-Guard | ✓ 144 Bytes, ds4 @ 0, qs @ 16 |
| `int_wmma_smoke_identity_by_counting` | Intrinsic läuft + Lane-Layout | ✓ bit-exakt |
| `int_wmma_smoke_random_signed` | Signed-Range + Vorzeichen | ✓ bit-exakt |
| `quantize_q8_1_mmq_matches_cpu_reference` | 128 Sub-Blocks Scales+Sums+qs | ✓ 0 Fehler |
| `quantize_q8_1_mmq_zero_input_safe` | NaN/Inf-Safety bei amax=0 | ✓ |

**5/5 grün**, Test-Laufzeit insgesamt < 0.5s.

## Geänderte Dateien

- `hip_kernels_v1/wmma/wmma_i32_smoke.hip` — **neu** (Integer-WMMA Smoke-Kernel)
- `hip_kernels_v1/quantize/quantize_q8_1_mmq.hip` — **neu** (DS4-Layout Quantize)
- `hip_kernels_v1/CMakeLists.txt` — 2 Library-Registrierungen
- `build.rs` — 2 Linker-Zeilen
- `src_v1/backend/gpu/wmma.rs` — 3 neue FFI-Bindings + `BlockQ81Mmq` Struct
- `tests_v1/mmq_infra_test.rs` — **neu** (5 Tests)
- `Cargo.toml` — Test-Registrierung

**Kein bestehender Produktionspfad geändert.** Decode bleibt bei 96.2 tok/s, Prefill bei ~1000 tok/s.

## Nächster Schritt

**MMQ-Port Schritt 2:** `load_tiles_q4_K` + `vec_dot_q8_1_q8_1_mma` + `mul_mat_q` Outer-Loop.

Mit der Schritt-1-Infrastruktur können wir jetzt:
- **Q4_K-Weights direkt in int8-Form in LDS legen** (via `load_tiles_q4_K`, Schritt-2-Aufwand ~200 LOC)
- **Integer-WMMA-Calls auf packed int8** durchführen (Schritt 1 hat den Fragment-Layout-Code für Single-Tile; Schritt 2 ist die Multi-Tile-Schleife ~300 LOC)
- **Scales aus `block_q8_1_mmq.ds4` nach der MMA-Reduktion anwenden** — Infrastruktur steht

Geschätzter Aufwand Schritt 2: 1-2 Sessions (~400-500 LOC + Korrektheits-/Perf-Tests).

## End-to-End Status

| Metrik | Pre-Schritt-1 | Post-Schritt-1 |
|---|---:|---:|
| Prefill tok/s | ~1000 | ~1000 (unverändert) |
| Decode tok/s | 96.2 | 96.2 (unverändert) |
| Korrektheit | 15/15 | 15/15 |
| Test-Anzahl | — | +5 |

Infrastruktur gelandet. Produktionspfad unberührt.
