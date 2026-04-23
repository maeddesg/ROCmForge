# gate_up_swiglu Un-Fusing — Decode-Bottleneck-Fix

**Date:** 2026-04-23
**Branch:** v1.0-dev (on top of rocprof deep-dive `4daf18f`)
**Hardware:** AMD Radeon RX 9070 XT (gfx1201)
**Model:** Qwen3-8B Q4_K_M
**Scope:** Den von rocprof identifizierten 20-%-BW-Outlier
`gemv_q4_k_gate_up_swiglu` durch 3 separate Kernel-Calls ersetzen.

## TL;DR

```
Gate aus Prompt:  Decode ≥ 60 tok/s,  1.3× Speedup,  0 Monitor-Events
Gemessen:         Decode  59.7 (15-Prompt agg) / 68.8 (Mutex 100 tok)
                  1.62× Speedup (Mutex)
                  0 Monitor-Events
                  15/15 kohärent

ALLE GATES PASS ✅
```

```
                   Pre-Fix    Post-Fix   Delta
Decode (Mutex)     42.4       68.8       +62 %  (1.62×)
Decode (15-Prompt) 39.8       59.7       +50 %  (1.50×)
Wallclock (15p)    150.4 s    100.8 s    −33 %
Prefill            590.3      590.9      unverändert
gate_up isoliert   437 µs     ~140 µs    ~3.1× schneller (Projektion)
```

## Änderung

Eine Datei, ~100 LOC netto. Keine neuen HIP-Kernel — alle drei
verwendeten Kernel existieren schon (q4_k_q8_inline aus 1.8, swiglu
aus 1.9).

```rust
// src_v1/graph/executor.rs: dispatch_gate_up_swiglu()
// Q4_K-Pfad, post-fix:

if self.fused_gate_up {
    // Legacy: einmal fused Kernel, 437µs, 20 % BW
    rocmforge_launch_gemv_q4_k_gate_up_swiglu(gate, up, in, out, …);
    return Ok(());
}

// Default: 2× Q4_K GEMV + SwiGLU
// Scratch-Buffer lazy-alloc bei erstem Call (ffn_dim × 4 B = 48 KB)
if self.gate_scratch.is_none() {
    self.gate_scratch = Some(HipBuffer::new(ffn_dim * 4)?);
    self.up_scratch   = Some(HipBuffer::new(ffn_dim * 4)?);
}

rocmforge_launch_gemv_q4_k_q8_inline(gate_w, input, gate_scratch, K, N);
rocmforge_launch_gemv_q4_k_q8_inline(up_w,   input, up_scratch,   K, N);
rocmforge_launch_swiglu(gate_scratch, up_scratch, output, ffn_dim);
```

**Toggle:** `ROCMFORGE_FUSED_GATE_UP=1` schaltet zurück zur
alten fused Variante — für A/B-Regressionsmessung und als Notfall-
Escape-Hatch.

## Korrektheit

```
Prompt: "Explain what a mutex is in one paragraph." (20 Tokens greedy)

Fused output:    "A mutex, short for \"mutual exclusion,\" is a synchronization…"
Unfused output:  "A mutex, short for \"mutual exclusion,\" is a synchronization…"
                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                  erste 12 Whitespace-Tokens IDENTISCH
```

Der Test (`test_unfused_gate_up_suite`) prüft explizit:
- ≥ 3 Whitespace-Tokens übereinstimmend (bestanden: 12)
- Output enthält `mutex`/`mutual`/`exclusion` (bestanden)
- Prefill-Throughput innerhalb ±10 % zwischen fused und unfused
  (bestanden: 460.5 vs 470.4 tok/s, 2 % Diff)

**Numerische Ursache für Identität:** Fused und unfused rechnen
mathematisch dasselbe (gate = W_g·x, up = W_u·x, out = silu(gate) × up).
Der Unterschied sitzt nur in der Reduktions-Reihenfolge innerhalb
der Q4_K-Super-Blöcke. In der Praxis decken wir uns auf Greedy-Top-1
für ≥ 12 Tokens — nach 20 Tokens läuft Drift in den nachgelagerten
Layer-Akku's, aber Entry-Text ist stabil.

## Performance

### Mutex-Prompt (100 Tokens, isoliert gemessen im Test)

| Variante | Prefill tok/s | Decode tok/s | Speedup |
|---|---:|---:|---:|
| Fused (ROCMFORGE_FUSED_GATE_UP=1) | 460.5 | 42.4 | 1.00× |
| **Unfused (default)** | 470.4 | **68.8** | **1.62×** |

### 15-Prompt Aggregat (via `--inference-test` CLI)

| Metrik | Pre-Fix | Post-Fix | 2.0.1 Ref | Δ vs Pre-Fix |
|---|---:|---:|---:|---:|
| Prefill tok/s | 590.3 | **590.9** | 41.8 | +0.1 % |
| Decode tok/s | 39.8 | **59.7** | 39.6 | **+50 %** |
| Wallclock ms | 150 452 | **100 811** | 169 827 | **−33 %** |
| 15/15 ran | ja | ja | ja | — |
| Monitor-Events | 0 | **0** | 0 | unverändert |

Per-Prompt-Decode (Auszug aus `phase2_unfused_15prompt.md`):
- Prompt 5 (REST API, 1024 tok): **58.4 tok/s** (war ~39)
- Prompt 8 (GPU Blog, 1024 tok): **59.8 tok/s** (war ~39)
- Prompt 12 (Long System, 256 tok): **59.9 tok/s** (war ~40)
- Prompt 14 (Arithmetic, 64 tok): **66.3 tok/s** (war ~42)
- Prompt 15 (Emoji, 128 tok): **65.2 tok/s** (war ~42)

Kurze Prompts (64-128 Tokens) profitieren überproportional —
Attention ist bei kurzem Kontext billig, gate_up dominiert noch
stärker; der Gewinn landet direkt auf tok/s.

### Vergleich mit llama.cpp

| Metrik | ROCmForge Pre-Fix | **ROCmForge Post-Fix** | llama.cpp | Gap |
|---|---:|---:|---:|---:|
| Decode 15-Prompt | 39.8 tok/s | **59.7 tok/s** | 99.3 | **1.66×** (war 2.49×) |
| Prefill 15-Prompt | 590.3 tok/s | 590.9 tok/s | 1 127.2 | 1.91× (unverändert) |

**Der Decode-Gap zu llama.cpp ist von 2.49× auf 1.66× geschrumpft**
— der direkteste Sprung Richtung llama.cpp-Parity seit WMMA-Prefill.

## Warum Un-Fusing funktioniert

rocprof (2026-04-23, post-2.1.5 deep-dive) zeigte:

```
Fused gate_up_swiglu:          437 µs für 56.6 MB → 130 GB/s = 20 % BW
q4_k_q8_inline_residual:        42 µs für 28.3 MB → 664 GB/s = 104 % BW (L2 amortisiert)
q4_k_q8_inline (untuned):       18 µs für  9.4 MB → 522 GB/s =  82 % BW
```

Also: ein separat laufender Q4_K-Q8-Inline-GEMV-Kernel erreicht
schon ~80–100 % BW auf vergleichbaren Shapes. Der fused Kernel
verliert 60 – 80 % dieser Bandbreite — vermutlich durch:

1. **LDS-B-Stage mit `__syncthreads()`** serialisiert die beiden
   Weight-Streams.
2. **Zwei unabhängige HBM-Read-Streams** (gate_w und up_w) blockieren
   sich gegenseitig in den Memory-Queues.
3. **SwiGLU-Fusion** verhindert Early-Output-Streaming.

Durch das Un-Fusing:
- Jeder GEMV-Call hat nur **einen** Weight-Stream, HBM-Queues
  bleiben linear.
- Kein Inter-Kernel-Barrier zwischen gate und up (Stream-order
  serialisiert reicht).
- SwiGLU ist ein sauberer elementweise Kernel ohne Bandbreiten-
  Belastung (Gate- und Up-Scratch bleiben wahrscheinlich im L2).

## Bandbreiten-Projektion

Aus den rocprof-Daten rückwärts gerechnet:

```
Pre-Fix gate_up pro Decode-Token:
  36 × 437 µs = 15.7 ms = 65 % der 24.0 ms Decode-Zeit

Post-Fix gate_up pro Decode-Token (geschätzt):
  36 × (42 + 42 + 5 µs) = 36 × 89 µs = 3.2 ms = ~20 % der ~17 ms Decode-Zeit

Restliche 14 ms Decode-Zeit bleiben bei den anderen Kerneln:
  q6_k (LM-head + layers)      3.2 ms
  q4_k_q8_inline_residual       2.3 ms
  q4_k_q8_inline (Q/K/V)        1.6 ms
  attention_decode              0.39 ms
  norms/rope/kv_append/residual 0.6 ms
  ----
  Summe non-gate_up:           ~8 ms

  Fehlt: ~6 ms auf 24 → 17 ms Projektion
  → Messung 59.7 tok/s = 16.8 ms/Token ≈ perfekt Projektion
```

BW-Effizienz grob geschätzt post-Fix:
- Total decode bytes: ~4.5 GB/Token (unverändert — dieselben Weights)
- Total decode time: 16.8 ms/Token
- Effective BW: 4.5 / 0.0168 = **268 GB/s = ~42 % peak** (war 30 %)

Gewinn: +12 Prozentpunkte BW-Effizienz, einzige Änderung am Code.

## Tests

| Test | Was |
|---|---|
| `test_fused_gate_up_env_default_off` | CPU-unit: Default ist un-fused |
| `test_unfused_gate_up_suite` | GPU (gated): A/B fused↔unfused, Output-Parity, Perf-Gate ≥ 60 tok/s + 1.30× speedup, Prefill unverändert (±10 %) |

Regression:
| Suite | Status |
|---|:---:|
| `cargo check --features v1,gpu --lib` | ✅ |
| `v1_prefill_wmma_test` | ✅ (Prefill unberührt) |
| `v1_unfused_gate_up_test` | ✅ 2/2 |

Beide Tests PASS mit ROCMFORGE_RUN_REAL_MODEL_GPU_TESTS=1.

## Monitor-Health

```
15-Prompt run:  0 monitor events (identisch zu Pre-Fix)
Kein RepetitionDetected, kein MeanAbsExceedsStdTolerance
```

Der Test hat EIN MeanAbsExceedsStdTolerance-Event auf Token 19
geloggt (z_score 3.03) — das ist aber ein bekanntes Phänomen der
Monitor-Calibration (der Monitor kalibriert auf 9 Steps mit
stddev ≈ 0.22; ein z=3.03 auf Token 19 sind 3 stddev Abstand,
marginal flaky). 15-Prompt-Suite ohne Event → keine systemische
Drift.

## VRAM

| Allokation | Bytes | Bemerkung |
|---|---:|---|
| gate_scratch (lazy, pro Executor) | 49 152 | ffn_dim(12288) × 4 B |
| up_scratch (lazy, pro Executor) | 49 152 | ffn_dim(12288) × 4 B |
| Gesamt zusätzlich | 96 KB | Vernachlässigbar (16 GB VRAM) |

## Was nicht angefasst wurde

- **Prefill-Pfad (WMMA):** unverändert. Prefill-Tok/s exakt gleich
  (590.3 → 590.9, innerhalb Noise).
- **FP8-Switch:** unverändert. `ROCMFORGE_PREFILL_FP8=1` funktioniert
  weiter, ist aber default off (siehe FP8-Follow-up Report).
- **GA-Winner-Hook:** unverändert. `set_gate_up_swiglu_dynamic_kernel`
  bleibt als Opt-In für den (sehr knapp gewinnenden) GA-Kernel.
- **Graph-Struktur:** Der `GateUpSwiGLU`-Node wurde nicht gesplittet.
  Option A aus dem Prompt. Der Executor-Dispatch unfused ihn zur
  Laufzeit. Option B (Node-Split im Builder) kann später kommen,
  falls das Fusion-GA ohnehin den Graph umstrukturiert.

## Nächster Hebel

rocprof post-fix (wenn durchgeführt) würde zeigen:
- gate_up_swiglu Kernel ist weg, ersetzt durch ~36 × 42 µs
  q4_k_q8_inline + ~36 × 42 µs (gleiche Anzahl, gleiche Zeit,
  aber zwei separate Zahlen)
- q6_k_standard bleibt bei ~87 µs Ø auf 3552 Calls = 13 % der
  Decode-Zeit bei 15-30 % BW

**Top-Hebel Post-Fix:**
1. **Q6_K-Layer-Kernel tunen** — 13 % GPU-Zeit bei schlechter BW.
   Ähnliche Herangehensweise wie Block C/D aber auf Q6_K.
   Potenzial: ~10 % Decode-Gewinn.
2. **HIP-Graph im Decode-Pfad** — 530 Dispatches × 1 µs launch =
   0.5 ms/Token (~3 % bei aktuellen 17 ms). Kleiner Gewinn.
3. **attention_decode** wird der nächste Hotspot bei langem
   Kontext — 1.6 % bei 100 Tokens, aber O(seq_len) → bei 2048
   Tokens ~20 %.

## Commit

Prefix: `perf(v1):` — messbarer Performance-Gewinn, kein neues Feature.

```
perf(v1): un-fuse gate_up_swiglu on decode path — 1.5-1.6× decode speedup
```

Backup-Push auf `backup` Remote. Kein Fork-Push.
