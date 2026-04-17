# CPU AVX-512 Q4_0 GEMV — Phase 0 Plan

**Ausgangs-SHA:** `b3818d9`
**CPU:** AMD Ryzen 9 7945HX (Zen4, 16C/32T, AVX-512 VNNI bestätigt via `/proc/cpuinfo`)
**Stand:** 2026-04-17

## Baseline (vor Optimierung)

Qwen2.5-0.5B Q4_0, Prompt "Hello", `--temperature 0.0 --top-p 1.0 --no-template`, `--max-tokens 64`:

- **Decode throughput: 12.6 tok/s**
- Prefill: 6.1 tok/s (irrelevant für diese Task, aber dokumentiert)
- Startup reports "Kernel preference: AVX-512 VNNI" (Feature-Detection funktioniert) — der eigentliche GEMV-Pfad nutzt diese Erkennung aber nicht.

Ziel-Schwellenwert: **≥ 40 tok/s** → 3.2× Speedup nötig.

## Wo das Problem sitzt

**Datei:** `src/cpu/ops.rs`, Zeile 1273–1274:

```rust
let features = super::features::CpuFeatures::get();
let use_avx2 = features.has_avx2;
```

Die Funktion `gemv_q4_0_q8_0` dispatched nur auf AVX2 oder Scalar. `has_avx512` und `has_avx512_vnni` werden korrekt erkannt (`src/cpu/features.rs:163-166`) und `KernelPreference::Avx512Vnni` wird ausgewählt — aber die Q4_0-GEMV verwendet diese Information nicht.

### Bestehende Pattern, auf die aufgebaut wird

- **Rayon-Multi-Threading:** `y.par_iter_mut().enumerate().for_each(...)` (Zeile 1276) — parallel über Output-Rows. Keine neue Dependency nötig.
- **Q8_0-Input-Quantisierung:** `quantize_q8_0_single` (Zeile 1271) — macht bereits den FP32 → Q8_0 Schritt. AVX-512-Variante kann identisch darauf aufsetzen.
- **Block-Loop mit 2× Unrolling + Prefetch:** Zeile 1282–1328. Wird übernommen / neu strukturiert.
- **Scalar-Fallback:** `dot_q4_0_q8_0_block_scalar` (Zeile 2082) — bleibt unverändert als Last-Resort.
- **AVX2-Kernel mit/ohne VNNI:** `dot_q4_0_q8_0_block_avx2` (Zeile 1946) → intern Dispatch auf `mul_sum_q4_0_q8_0_block_avx2_vnni` (AVX2-VNNI, Intel-only) oder `_unscaled`. Bleibt als Fallback für Nicht-AVX-512-CPUs.

### Q4_0-Block-Layout (bestätigt, kein Umbau nötig)

Ein Q4_0-Block = 18 Bytes:
- 2 Bytes FP16 `scale`
- 16 Bytes: 32 × 4-bit Werte (je 2 pro Byte, `lo = byte & 0x0F`, `hi = byte >> 4`)
- Dequant: `value = (nibble − 8) × scale`

Das Gewichtslayout ist bereits SIMD-freundlich (zusammenhängend per Row, 16-Byte-Payload pro Block — passt auf 128-Bit-Register oder half-lane eines 256/512-Bit-Registers).

## Plan für Phase 1

1. **Neuer Kernel** `dot_q4_0_q8_0_block_avx512_vnni`:
   - Eingabe: `qs: &[u8; 16]`, `q8: &[u8; 32]`, `combined_scale: f32`
   - Nibble-Unpacking: `_mm_loadu_si128` (16 Bytes) → Split in lo/hi Nibbles → Konkatenieren zu `__m256i` mit 32 signed INT8 (−8 bias).
   - Q8 Input: `_mm256_loadu_si256` (32 Bytes).
   - Dot-Product: `_mm256_dpbssd_epi32` (AVX-512 VNNI signed/unsigned → signed ist spezifisch, auf AMD verfügbar ab Zen4). Alternativ zweistufig mit `_mm256_maddubs_epi16` wenn die signed-variante fehlt.
   - Horizontal-Sum + scale multiplikation.
2. **Zwei-Block-Variante** `dot_q4_0_q8_0_2blocks_avx512_vnni`:
   - Lädt 2 benachbarte Blöcke gleichzeitig in ein 512-bit-Register, verarbeitet beide mit einer einzigen `_mm512_dpbusd_epi32`-Instruktion. Das ist der Hauptgewinn gegenüber AVX2 — doppelte Datenbreite bei gleicher Instruktionsanzahl.
3. **Dispatch in `gemv_q4_0_q8_0`:**
   ```rust
   let use_avx512_vnni = !env_flag_disable_avx512() && features.has_avx512 && features.has_avx512_vnni;
   ```
   Opt-out via `ROCMFORGE_DISABLE_AVX512=1`.
4. **Multi-Threading:** Rayon ist bereits aktiv. Kein Änderungsbedarf. Auf 16 Kernen bei 896 Outputs (0.5B FFN-down) → 56 Rows/Thread. Passt komfortabel in L2.
5. **`has_avx512_vnni` im CpuFeatures-Struct:** Fehlt aktuell als öffentliches Feld — wird intern in `detect_x86_64` berechnet, aber nicht exponiert. Muss in das Struct aufgenommen werden, damit der Dispatch-Pfad es lesen kann.

## Multi-Threading Status

Rayon ist bereits aktiv (`par_iter_mut` in `gemv_q4_0_q8_0`). Keine neue Dependency nötig. Der existierende Thread-Pool wird automatisch mitgenutzt.

## Cache-Budget

- 0.5B, hidden=896: `896/32 × 18 = 504 B` pro Row. 56 Rows/Thread (bei 16 Kernen, 896 Outputs) = 28 KB. Passt in L1 (32 KB/Kern).
- 0.5B, FFN-up: 4864 Outputs, 896 Inputs: 304 Rows/Thread = 153 KB. Passt in L2 (1 MB/Kern).
- 7B, FFN-down: 3584 Outputs, 18944 Inputs: `18944/32 × 18 = 10.6 KB` pro Row. 224 Rows/Thread = 2.4 MB. Geht ÜBER L2 (1 MB/Kern) → Cache-Tiling würde bei 7B helfen. Für 0.5B irrelevant, daher erst in Stufe 2 nötig.

## Validierung

`tests/cpu_avx512_matches_reference.rs`:
- Lauf A: `ROCMFORGE_DISABLE_AVX512=1` (alter Pfad, AVX2).
- Lauf B: AVX-512 aktiv (default).
- Assertion: Token-IDs byte-identisch bei Greedy + festem Prompt, 50 Tokens.

## Kein Rückfrage-Bedarf

Alle strategischen Fragen aus dem Prompt sind durch den Code beantwortet:
- CPU-GEMV-Pfad: lokalisiert (`src/cpu/ops.rs:1244`).
- Q4_0-Layout: SIMD-friendly, kein Umbau.
- Rayon: bereits aktiv.
- CPU-Feature-Detection: komplett, muss nur `has_avx512_vnni` exponieren.
- Baseline: 12.6 tok/s (3.2× nötig für ≥40 tok/s).

Implementierung startet in Phase 1.
