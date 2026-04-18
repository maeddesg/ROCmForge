# down_proj 10× Lücke — Root Cause und Fix (Phase 4 Step 2)

**Date:** 2026-04-18
**Hardware:** AMD Radeon RX 9070 XT (gfx1201, RDNA 4), ROCm 7.2.2
**Model:** Qwen2.5-7B-Instruct Q4_0, 28 Layer, h=3584, ff=18944, 28/4 GQA
**Baseline before fix:** commit `a58e861`
**Commit with fix:** see bottom

## Root Cause — Hypothese B bestätigt

Der `ffn_down`-Gewichtstensor wird in GGUF als `[intermediate_size, hidden_size]` = `[18944, 3584]` gespeichert. In `src/cpu/transpose.rs:98-108` wird das als `needs_transpose = true` markiert, weil die GPU-Kernel das "Standard"-Layout `[out_dim, in_dim]` = `[hidden_size, intermediate_size]` erwarten.

In `src/gpu/ops.rs::gpu_dispatch_gemm` hatten vor dem Fix **beide** schnellen Pfade eine `!meta.needs_transpose`-Bedingung:

```rust
// Line 1468 — WMMA path
if seq_len >= WMMA_PREFILL_MIN_M
    && meta.wtype == GgmlType::Q4_0
    && !meta.needs_transpose    // ← down_proj fällt hier raus
    ...

// Line 1517 — hipBLAS path
if seq_len >= PREFILL_GEMM_THRESHOLD
    && meta.wtype == GgmlType::Q4_0
    && !meta.needs_transpose    // ← down_proj fällt auch hier raus
    ...
```

Konsequenz: **alle 28 down_proj-Aufrufe fielen auf den unoptimierten `gemm_q4_0_f32`-Scalar-Kernel durch** (Zeile 1603), der Q4_0 direkt in FP32 dequantisiert und im generischen GEMM-Pfad läuft. Bei einer GEMM-Shape `M=257, N=3584, K=18944` ist das ca. 10.988 µs statt 1.294 µs (8.5× Overhead zur isolierten WMMA-Baseline, 10.6× zur QKV-Referenz).

## Fix

**Änderung 1** — `src/gpu/prefill_gemm.rs::hgemm_row_major` nimmt einen `weight_transposed: bool` Parameter entgegen. Für `true` (FFN-down, tied LM-head) wird hipBLAS mit `OP_N` statt `OP_T` und `lda = out_dim` statt `lda = in_dim` aufgerufen. Die physischen Bytes werden identisch konsumiert; der Unterschied ist nur, ob hipBLAS eine weitere logische Transposition durchführt.

**Änderung 2** — `src/gpu/ops.rs::gpu_dispatch_gemm` lässt jetzt `meta.needs_transpose = true` auch in den hipBLAS-Pfad durch (WMMA bleibt ausgeschlossen — der hand-optimierte Q4_0-WMMA-Kernel geht von der Block-Reihenfolge `[out_dim, in_dim]` aus und das ist Kernel-Tuning, das nicht in diesen Schritt gehört). `dispatch_prefill_via_hipblas` reicht das Flag an `hgemm_row_major` weiter.

```rust
if seq_len >= super::prefill_gemm::PREFILL_GEMM_THRESHOLD
    && meta.wtype == GgmlType::Q4_0
    && super::safety::hipblas_prefill_enabled()
{
    ...
    dispatch_prefill_via_hipblas(
        device, weights, input, output, out_dim, in_dim, seq_len,
        meta.needs_transpose,   // ← neu
    )?;
```

## Messung

### Korrektheit

- `cargo test --release --features gpu` für 7 Korrektheits-Test-Binaries: **32/32 grün** (`wmma_q4_0_correctness` 5, `wmma_padding_correctness` 6, `prefill_wmma_attention_e2e` 1, `chat_single_turn_correctness` 3, `chat_multi_turn_correctness` 5, `wmma_attention_gqa_causal_correctness` 9, `wmma_16x16_correctness` 3).
- Greedy-Decode auf identischem Prompt, 10 Tokens:
  - Neuer hipBLAS-Pfad: `"Certainly! Below is a Python function that implements the"`
  - Alter `gemm_q4_0_f32`-Fallback (via `ROCMFORGE_DISABLE_HIPBLAS_PREFILL=1 ROCMFORGE_DISABLE_WMMA_PREFILL=1`): identischer String
  - **`diff` zeigt byte-identischen Output.** FP16-hipBLAS behält bei Greedy-Decode die Top-1-Argmax des FP32-Fallbacks.

### Profiling (pp256, aggregiert über 28 Layer, mit `ROCMFORGE_PROFILE_PREFILL_OPS=1`)

| Operation | Vorher (ms) | Nachher (ms) | Faktor |
|---|---:|---:|---|
| **down_proj** | **303.77** | **163.87** | **1.85× schneller** |
| gate_proj | 44.36 | 43.99 | — |
| up_proj | 43.86 | 44.23 | — |
| q_proj | 9.70 | 9.48 | — |
| o_proj | 9.56 | 9.33 | — |
| attention | 9.73 | 9.50 | — |
| k_proj | 2.74 | 2.66 | — |
| v_proj | 2.73 | 2.66 | — |
| silu_mul | 3.81 | 3.76 | — |
| **Prefill gesamt** | **441.24** | **300.23** | **1.47× schneller** |

Per-Layer-Median down_proj: **10.988 µs → 5.001 µs** (2.2× schneller). Immer noch nicht die isolierte WMMA-Bench-Zahl von 1.294 µs, aber der hipBLAS-Pfad kommt auf ca. ein Drittel dieser Distanz heran (Dequant + FP16-Konversion + Hgemm-Call sind nicht kostenlos).

### End-to-End-Throughput

```
pp=64  → 472.4 tok/s  (Median, 3 runs)   — unchanged vs. pre-fix 472.3
pp=128 → 679.4 tok/s                     — pre-fix 551.6 tok/s  (+23 %)
pp=256 → 904.2 tok/s                     — pre-fix 593.0 tok/s  (+52 %)
pp=512 → 1049.6 tok/s                    — pre-fix 610.2 tok/s  (+72 %)
```

Bei pp64 kein Effekt, weil seq_len=65 nahe der `PREFILL_GEMM_THRESHOLD=32`-Schwelle ist und down_proj dort sowieso die meiste Zeit dominiert (die Shape ist M-bound bei kleinem seq_len). Bei pp≥128 skaliert der Speedup stark — der hipBLAS-Pfad amortisiert die fixen Kosten (Dequant, FP16-Konversion) mit wachsender M-Dimension.

### Vergleich mit der Hypothese

| Hypothese | Zutreffend? | Evidenz |
|---|---|---|
| A — falscher Dispatch-Pfad | **zum Teil** | nur 168/196 Dispatches gehen auf `wmma_q4_0`; 28 (= down_proj) tauchten in den Logs gar nicht auf → falscher Pfad ja, aber nicht hipBLAS, sondern `gemm_q4_0_f32`-Scalar-Fallback |
| **B — Transponierte Gewichte** | **JA** | `compute_transpose_flag` setzt `needs_transpose = true` für `ffn_down.weight`, was beide schnellen Pfade ausschließt |
| C — Separate Dequant | nein | der schnelle Pfad dequantisiert sowieso on-the-fly; problem war der Pfad selbst, nicht ein extra Step |
| D — Shape-Performance im WMMA | nein | WMMA-Kernel wird für down_proj gar nicht aufgerufen; dass seine isolierte Performance bei K=18944 nur 1.294 µs ist, bleibt ungenutzt bis zum Kernel-Rewrite |
| E — Pipeline-Stall | nein | Stalls hätten auch andere Ops verlangsamt, die davon nicht betroffen sind |

### FP32↔FP16-Shuttling

Der hipBLAS-Pfad macht jetzt für jedes down_proj-Layer 3 Zusatz-Konversionen: Q4_0→FP16 (Gewicht), FP32→FP16 (Input), FP16→FP32 (Output). Bei ca. 34 MB FP16-Gewicht und ca. 3.6 MB Input ist das messbar, aber der 140-ms-Gewinn bei down_proj steht dem gegenüber. Der Netto-Effekt ist eindeutig positiv.

## Nächster Schritt

Phase 4 Step 3: **gate + up Fusion** als nächster Hebel (88 ms, 29 % des Prefills nach dem Fix — jetzt die größte Post-fix-Kategorie nach down_proj). Beide teilen sich `normed` als Input und schreiben in benachbarte Buffer — ideale Fusion-Kandidaten.

Optional für später (Phase 2c / Kernel-Tuning): WMMA-Q4_0-Kernel um eine Variante für transponierte Gewichte erweitern. Potenzielle weitere 2–3× auf down_proj (vom aktuellen 5 µs auf nähe 1.3 µs), entspricht ~90 ms weiterem Gewinn bei pp256.
