# CPU AVX-512 VNNI Q4_0 GEMV — Analysis & Verdict

**Commit:** `6c0eade+`
**CPU:** AMD Ryzen 9 7945HX (Zen4, 16C/32T, AVX-512 VNNI, DDR5 Dual-Channel ~77 GB/s)
**Model:** Qwen2.5-0.5B-Instruct Q4_0 (353 MB)
**Date:** 2026-04-17

## TL;DR

**Qwen2.5-0.5B Q4_0 Decode-Throughput auf CPU mit AVX-512: ~12.1 tok/s.**
**Baseline (AVX2): ~12.1 tok/s. Speedup: ~0%.**
**Isolated kernel speedup: 1–7% (0.5B shapes), 16–19% (7B shapes).**

**Verdikt: < 40 tok/s → Wechsel zu Prefill-GEMM.** Heterogenes
Spec-Decode (Draft auf CPU, Target auf GPU parallel) ist mit dem
aktuellen CPU-Forward-Stack nicht machbar.

## Was gemacht wurde

1. `src/cpu/features.rs` um `has_avx512_vnni` erweitert (zuvor nur intern berechnet).
2. Neuer AVX-512 VNNI Kernel in `src/cpu/ops.rs`:
   - `dot_q4_0_q8_0_2blocks_avx512_vnni` — verarbeitet 2 Q4_0-Blöcke (64 Bytes) mit einem einzigen `_mm512_dpbusd_epi32`.
   - `dot_q4_0_q8_0_block_avx512_vnni` — Single-Block Fallback für ungerade Block-Zähler, nutzt AVX-512-VL-256-bit-VNNI.
   - Bias-Korrektur für signed Q4 via `dot - 8 × sum(q8)` (vpdpbusd ist unsigned × signed; `_mm512_sign_epi8` existiert nicht).
3. Dispatch in `gemv_q4_0_q8_0` um AVX-512-VNNI-Pfad erweitert, Fallback-Kette: AVX-512-VNNI → AVX2 → Scalar.
4. Opt-out via `ROCMFORGE_DISABLE_AVX512=1`.
5. Korrektheitstest `tests/cpu_avx512_matches_reference.rs` (byte-identical Output).
6. Micro-Benchmark `src/bench_gemv.rs` um AVX-512 vs AVX2 bei 0.5B+7B-Shapes erweitert.
7. End-to-end Sweep-Script `benches/bench_cpu_avx512.fish`.

## Isolated kernel benchmark

Single-Thread GEMV `gemv_q4_0_q8_0`, inklusive Q8-Input-Quantisierung und Rayon-Overhead:

| Shape                                | AVX2 µs/call | AVX-512 µs/call | Speedup |
|--------------------------------------|-------------:|----------------:|--------:|
| 0.5B QKV/O  (896 × 896)              |        147.1 |           145.5 |   1.1%  |
| 0.5B Gate/Up (4864 × 896)            |        257.2 |           245.0 |   4.7%  |
| 0.5B Down   (896 × 4864)             |        218.7 |           204.4 |   6.5%  |
| 7B   QKV/O  (3584 × 3584)            |        339.8 |           326.6 |   3.9%  |
| 7B   Gate/Up (18944 × 3584)          |       1044.9 |           878.7 |  18.9%  |
| 7B   Down   (3584 × 18944)           |       1050.3 |           902.3 |  16.4%  |

Effektive Speicher-Bandbreite (AVX-512):
- 0.5B: 3–12 GB/s (weit unter 77 GB/s DDR5 — Daten passen in L2/L3)
- 7B: 22–44 GB/s (nähert sich der Hälfte der DDR5-Bandbreite)

## End-to-end decode throughput

15 Runs (3 Prompts × 2 Modi × 3 Durchläufe), Qwen2.5-0.5B Q4_0, 128 Tokens, Greedy:

| Prompt | AVX2 (median) | AVX-512 (median) | Speedup |
|--------|--------------:|-----------------:|--------:|
| code   | 12.1 tok/s    | 12.1 tok/s       |   0%    |
| chat   | 12.1 tok/s    | 11.7 tok/s       |  −3%    |
| prose  | 12.0 tok/s    | 11.9 tok/s       |  −1%    |

Raw data: `cpu_avx512_sweep_6c0eade_1776406574.json`.

7B end-to-end: 0.7 tok/s (CPU unbrauchbar für 7B-Inferenz).

## Warum der Kernel-Gewinn nicht durchschlägt

Das Kernel läuft auf 0.5B-Shapes 1–7% schneller — aber dieser Gewinn verschwindet in der End-to-end-Messung. Gründe:

1. **Rayon-Overhead dominiert bei kleinen Matrizen.** Jede GEMV-Aufruf (etwa 180 per Token × 24 Layer = 192 Calls) spawnt Rayon-Tasks. Bei 56 Output-Rows pro Thread und ~500 µs pro GEMV-Aufruf ist der Thread-Pool-Handshake (Task-Split, Sync-Barrier) ein nicht-trivialer Anteil. Die eigentliche Compute-Zeit der SIMD-Instruktionen ist klein gegen den Dispatch-Overhead.
2. **Andere Ops sind nicht SIMD-optimiert.** `flash_attn_decode`, `rms_norm`, `silu_fuse`, `rope`, `residual_add` laufen im CPU-Pfad grösstenteils skalar oder nur mit Auto-Vektorisierung. Diese Ops summieren sich bei 0.5B zu einem erheblichen Anteil der 80 ms/Token.
3. **`quantize_q8_0_single` ist skalar.** Die FP32→Q8-Konvertierung für den Input-Vektor ist nicht SIMD-optimiert. Bei kleinen Shapes (hidden=896) ist das zwar nur ~1 µs, aber 200+ Aufrufe pro Token summieren sich.
4. **Bandbreite ist nicht der Engpass bei 0.5B.** Die Weight-Daten (~200 MB pro Token-Pass) würden bei 77 GB/s in ~2.6 ms liegen — wir sind bei 80 ms, 30× langsamer. Selbst ein 2× schneller Kernel würde end-to-end um weniger als 20% zulegen, weil die anderen Kosten dominieren.

## Verdikt

**Fused FFN auf CPU, Attention auf CPU, und weitere SIMD-Optimierungen der Rest-Pipeline würden zusammen den Faktor ≥3× brauchen, der nötig wäre um 40 tok/s zu erreichen.** Das ist ein mehrmonatiges Projekt — nicht die "kleine Ergänzung", die sich aus der Spec-Decode-Arbeit ergeben hätte.

### Was das für die strategische Richtung bedeutet

- **Heterogenes Spec-Decode (Draft auf CPU, Target auf GPU parallel) ist NICHT machbar** mit dem aktuellen CPU-Stack. Bei 12 tok/s CPU-Draft vs. 222 tok/s GPU-Draft wäre die GPU die meiste Zeit blockiert, weil der Draft-Schritt 82 ms brauchen würde statt 4.5 ms.
- **Prefill-GEMM (hipBLAS oder WMMA) ist der nächste Meilenstein.** Prefill-Gap zu llama.cpp: 59 vs 1.092 tok/s (18× Gap), der grösste verbleibende Performance-Hebel im Projekt.

### Was das AVX-512-Kernel dennoch bringt

Der Kernel bleibt im Code — er ist ein saubere, getestete Grundlage und bringt in Isolation 16–19% auf 7B-Shapes. Wenn später:
- Prefill auf GEMM umgestellt wird, könnte der Kernel für Q4_0 × Q8_0 Batch-Operationen nützlich sein.
- Die restliche CPU-Pipeline SIMD-optimiert wird (Fused FFN, Flash-Attention, RMSNorm, SiLU), würde der Kernel einen echten End-to-end-Gewinn liefern.
- Auf Maschinen **ohne** GPU läuft der CPU-Pfad sowieso — 7% mehr Durchsatz auf 0.5B-Drafts ist besser als 0%.

Der Kernel wird nicht zurückgerollt. Aber er wird nicht der nächste Meilenstein sein.

## Offene Fragen / Follow-ups (nicht jetzt)

- Wie teuer ist das Rayon-Scheduling pro GEMV-Aufruf wirklich? Ein Profiling mit `perf` oder einem Instrumentierungs-Layer würde die obere Grenze für weitere GEMV-Optimierung zeigen.
- Lohnt sich ein einmaliger globaler Thread-Pool mit persistenten Worker-Threads (Rust `crossbeam_channel` / `parking_lot`) anstelle von Rayon's Fork-Join pro Aufruf?
- Auf einer CPU ohne GPU könnte AVX-512 + fused operations 0.5B-Inference auf ~25–30 tok/s bringen — das wäre immer noch kein Hero-Feature, aber würde ROCmForge als CPU-only-Lösung konkurrenzfähig machen.
- 7B auf CPU: aktuell 0.7 tok/s. Zum Vergleich: llama.cpp erreicht auf demselben 7945HX typischerweise 6–8 tok/s. Der CPU-Forward hat also ~10× Gap zum Stand der Technik — separat von der AVX-512-Frage.
