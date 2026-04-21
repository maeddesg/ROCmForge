# Phase 1 Schritt 1.17 — Performance Baseline (rocprof v3)

**Date:** 2026-04-21
**Branch:** v1.0-dev (tag `v1.0-phase1-complete`)
**Hardware:** AMD Radeon RX 9070 XT (gfx1201), 640 GB/s BW, 16 GiB VRAM
**Tool:** rocprofv3 1.1.0 (`--kernel-trace`, `--hip-trace`)
**Prompt:** "Explain what a mutex is in one paragraph."
**Model:** Qwen3-8B Q4_K_M

## Vorgehen

rocprofv3 `--kernel-trace` + `--hip-trace` liefern auf gfx1201
verlässliche Timestamp-basierte Kernel- und API-Traces. Hardware-
Counter (`--pmc`) werden nicht benötigt, weil wir wallclock-pro-
Kernel direkt aus dem Trace bekommen. Der 1.4-Telemetry-Hook
(`ROCMFORGE_PROFILE=1`) ist gebaut aber noch nicht an den 1.10-
Executor gewired; daher ausschliesslich rocprofv3 genutzt.

Alle Traces verwerfen die ersten 15 % der Dispatches als Warmup
(Model-Load, Introspection-Embedding-Dequant, erste Bandit-
Explorations-Pulls). Ausgewertet werden nur die Steady-State-
Dispatches des Decode-Pfads.

Drei Läufe:

1. **v1.0 ohne Tuning** (Pre-Bandit-Baseline, fixed-kernel Dispatch).
2. **v1.0 mit Tuning** (`--show-tuning`, Bandit aktiv) — *der
   Hotpath den User sehen*.
3. **v0.3.0 mit GPU** — Referenz für die Bottleneck-Natur.

## Resultate

### Run 1 — v1.0 ohne Tuning (fixed kernels)

129 execute_decode-Iterationen (33 Prefill + 96 Decode).

| Kernel | Count | Σ µs | Ø µs | % GPU |
|---|---:|---:|---:|---:|
| `gemv_q4_k_standard` | 15 824 | 2 235 984 | 141.3 | 51.4 % |
| `gemv_q4_k_gate_up_swiglu` | 3 956 | 1 648 030 | 416.6 | 37.9 % |
| `gemv_q6_k_standard` | 4 062 | 346 439 | 85.3 | 8.0 % |
| `attention_decode` | 3 955 | 40 358 | 10.2 | 0.9 % |
| `rms_norm` | 8 020 | 28 251 | 3.5 | 0.6 % |
| `rms_norm_batched` | 7 910 | 17 985 | 2.3 | 0.4 % |
| `rope` | 7 910 | 14 880 | 1.9 | 0.3 % |
| `residual_add_inplace` | 7 911 | 13 575 | 1.7 | 0.3 % |
| `kv_cache_append` | 3 955 | 7 560 | 1.9 | 0.2 % |
| **Σ GPU** | | **4 353 ms** | | 100 % |

Decode: 19.0 tok/s. Prefill: sequentiell, ~19 tok/s.

### Run 2 — v1.0 mit Tuning (Bandit → q8_inline)

Selber Trace nach Bandit-Konvergenz — das ist der *ausgelieferte*
Pfad.

| Kernel | Count | Σ µs | Ø µs | % GPU |
|---|---:|---:|---:|---:|
| **`gemv_q4_k_gate_up_swiglu`** | 4 139 | **1 744 852** | **421.6** | **65.4 %** |
| `gemv_q4_k_q8_inline` | 16 555 | 427 171 | 25.8 | 16.0 % |
| `gemv_q6_k_standard` | 4 253 | 364 230 | 85.6 | 13.7 % |
| `attention_decode` | 4 139 | 44 065 | 10.7 | 1.7 % |
| `rms_norm` | 8 392 | 29 974 | 3.6 | 1.1 % |
| `rms_norm_batched` | 8 278 | 19 094 | 2.3 | 0.7 % |
| `rope` | 8 278 | 15 788 | 1.9 | 0.6 % |
| `residual_add_inplace` | 8 278 | 14 426 | 1.7 | 0.5 % |
| `kv_cache_append` | 4 139 | 8 016 | 1.9 | 0.3 % |
| **Σ GPU** | | **2 668 ms** | | 100 % |

Decode: 30.6 tok/s. **1.63× schneller** als ohne Tuning — der
Bandit wählt `q8_inline` für Q4_K Q/K/V/Output (5.5× schneller pro
Call: 141 µs → 26 µs) und bringt damit praktisch den gesamten
Phase-1-Gewinn.

### Run 3 — v0.3.0 Referenz (selber Prompt, `--gpu`)

| Kernel | Count | Σ µs | Ø µs | % GPU |
|---|---:|---:|---:|---:|
| `gemv_gate_up_swiglu_q4_k_q8_inline_multi_row_k` | 3 732 | 1 471 163 | 394.2 | 66.1 % |
| `gemv_q4_k_q8_inline_residual_multi_row` | 5 598 | 233 687 | 41.7 | 10.5 % |
| `gemv_q4_k_q8_inline_multi_row<8>` | 9 328 | 163 233 | 17.5 | 7.3 % |
| `gemv_q6_k_direct<8>` | 1 866 | 134 019 | 71.8 | 6.0 % |
| `gemv_q6_k_multi_row<8>` | 1 969 | 117 044 | 59.4 | 5.3 % |
| `flash_attn_decode_strided_multi_head_v2` | 3 732 | 45 669 | 12.2 | 2.1 % |
| `rms_norm` | 7 567 | 24 609 | 3.3 | 1.1 % |
| `rms_norm_batched` | 7 464 | 16 327 | 2.2 | 0.7 % |
| `kv_write_rope` | 3 732 | 7 654 | 2.1 | 0.3 % |
| `rope_heads` | 3 732 | 7 133 | 1.9 | 0.3 % |
| `add` | 1 866 | 2 920 | 1.6 | 0.1 % |
| **Σ GPU** | | **2 224 ms** | | 100 % |

Decode: 41.7 tok/s.

### Dispatches pro Decode-Iteration

| Run | HIP Dispatches total | pro Token (inkl. Prefill) |
|---|---:|---:|
| v1.0 ohne Tuning | 74 965 | 581 |
| v1.0 mit Tuning | 78 445 | 608 |
| v0.3.0 | 59 634 | 462 |

v0.x feuert **31 % weniger Kernel pro Token** — das ist der
direkte Ertrag der Fused Kernels (`q4_k_q8_inline_residual` fused
GEMV+Residual-Add, `kv_write_rope_kernel` fused KV-Append+RoPE).

### HIP-API-Overhead

`rocprofv3 --hip-trace`, Steady-State-Fenster:

| Run | Top-API | Count | Σ µs | Ø µs | % API |
|---|---|---:|---:|---:|---:|
| v1.0 Tuned | **`hipStreamSynchronize`** | **83 260** | **3 755 117** | 45.1 | **97.9 %** |
| v1.0 Tuned | `hipLaunchKernel` | 66 585 | 56 041 | 0.84 | 1.5 % |
| v0.3.0 | `hipLaunchKernel` | 55 570 | 52 345 | 0.94 | 2.1 % |
| v0.3.0 | `hipStreamSynchronize` | 114 | 85 | 0.74 | 0.0 % |

v1.0 hat 83 260 Stream-Syncs im selben Zeitraum; v0.x hat 114.
Quelle: `dispatch_gemv_tuned` ruft nach jedem Kernel-Launch
`self.stream.synchronize()`, um die elapsed time für den
Bandit zu messen (Step 1.12 Kommentar: "Phase-2 replaces this
with HIP-event timing batched at token end so the extra sync
goes away").

Dabei sind die 3.755 s *nicht* reine CPU-Overhead — der grösste
Teil davon ist die CPU, die darauf wartet, dass der *eine
gerade laufende* Kernel fertig wird. Der Kern-Effekt ist
trotzdem schädlich: mit Sync-nach-jedem-GEMV bleibt kein Raum
für HIP-Stream-Pipelining mehrerer Kernel hintereinander. Die
GPU könnte bei zurückgehaltenen Syncs die nächste Op schon
laden während die aktuelle noch rechnet; aktuell ist die
Timeline streng sequentiell.

### VRAM

Gemessen mit `rocm-smi --showmeminfo vram` während 200-Token-
Inference:

| | Wert |
|---|---:|
| VRAM Total | 17 096 MB |
| VRAM Used (Peak) | 10 053 MB |
| Reserve | 7 043 MB |

Komfortabel innerhalb des Budgets; grosser Puffer für KV-Cache-
Wachstum oder 14B-Modelle.

### Occupancy (aus Kernel-Trace)

Kernel | VGPR | LDS/Block | Hinweis
---|---:|---:|---
`gemv_q4_k_gate_up_swiglu` | 192 | 0 | Hoher VGPR-Druck; gfx1201 hat 1536 VGPRs/CU, 192/Wave → 8 Waves/CU — OK
`gemv_q4_k_q8_inline` | 192 | 0 | Wie oben
`gemv_q4_k_standard` | 112 | 0 | Niedriger; würde höhere Occupancy tragen, ist aber irrelevant sobald Bandit auf q8_inline wechselt
`gemv_q6_k_standard` | 72 | 0 | Gut
`attention_decode` | 40 | 512 B | LDS okay, VGPR niedrig → hohe Occupancy

Keine der Top-3-Kernel sind VGPR-limited auf RDNA 4 (1536 VGPRs/CU
erlauben 8 Waves à 192 VGPR pro CU). Die Langsamkeit kommt nicht
aus Occupancy-Problemen.

## Bottleneck-Analyse

**Frage 1 — Wo geht die Zeit hin?**

> **65 % gate_up_swiglu, 16 % q8_inline (attention-Pfad), 14 % q6_k**
> (LM head). Zusammen **95 %** der GPU-Zeit. Die anderen 5 %
> verteilen sich über RMSNorm, RoPE, Attention-Decode und
> Residual-Add.

**Frage 2 — Was limitiert uns?**

Pro Token (Ø aus Tuned-Run):
- 4139 gate_up + 16555 q8_inline + 4253 q6_k + 4139 attention +
  … = ca. 608 Kernel-Launches pro Call, ca. 4.13 M Elemente à
  4 Bytes + Q4_K-Blöcke gelesen → grob 4.5 GB/Token Gewichts-
  Traffic (entspricht ~130 tok/s BW-Limit bei 640 GB/s).
- Aktueller Decode: 30 tok/s = ~135 GB/s effektiv = **21 %**
  BW-Nutzung.

Wir sind **nicht BW-bound**. Auch nicht Occupancy-bound. Der
Engpass ist die **Kernel-Dauer an sich** (Q4_K-Dequant im
standard-Pfad vs. Q8-inline) + die **fehlende Fusion**
(Residual+GEMV, KV+RoPE).

**Frage 3 — Wo ist der grösste Hebel?**

| Kandidat | aktueller Impact | realistische Verbesserung |
|---|---:|---|
| gate_up Q8-inline Variante | 65 % der GPU-Zeit | −30 bis −50 % auf diesen Kernel (v0.x Q8-inline gate_up läuft bei 394 µs vs. v1.0 421 µs — nur ~6 % schneller; Haupt-Lever ist stattdessen Kernel-Fusion) |
| Residual-fused GEMV | 16 % attn-Pfad + 0.5 % residual | spart 1 Kernel-Launch pro Layer × 36 Layer × 4 (Q/K/V/O) — ~3 % |
| KV+RoPE fused | Σ 0.6 % | kombiniert 2 auf 1 — marginal |
| Stream-Sync-Elim. | 0 GPU, aber serialisiert | gibt HIP Freiraum für Kernel-Überlappung — schätz. 5–10 % |

**Frage 4 — Säulen-Overhead?**

| Säule | Kosten (Tuned-Run) | % der wallclock-Zeit |
|---|---|---:|
| Säule 4 Bandit — Sync pro GEMV | 83 260 Syncs à ~45 µs block-Zeit | ~80 % der HIP-API-Wall (s.o.); aber grosser Teil ist GPU-Wartezeit |
| Säule 5 Monitor — 3 hidden-reads | 3 Syncs + 3 hipMemcpy à 16 KB | < 0.1 % |
| CPU Graph-Traversal | 129 Iterationen, je ~543 Nodes, Hashmap-Dispatch | ca. 50 µs/Iteration = 6 ms total ≈ 0.1 % |

Der einzige Säulen-Overhead der messbar wird ist der Bandit-Sync.
Monitor und Graph-Traversal sind unterhalb der Messgrenze.

## Priorisierte Phase-2-Vorschläge

### P0 — Fused Kernels portieren (grösster Hebel)

```
Optimierung:      Q8-inline + Residual-Fusion auf v1.0 Codegen übertragen
Betroffene Kernel: gemv_q4_k_q8_inline, attention-output-Pfad
Aktuell:          q8_inline 26 µs/Call × 16555 Calls = 427 ms (16 %)
                  residual_add separat 1.7 µs × 8278 = 14 ms (0.5 %)
Geschätzter Speedup: 20–30 % weniger Wallclock (durch +1 gefusterten
                  Kernel und eine Runde weniger Buffer-Roundtrip)
Erwarteter Decode-Gewinn: 30 → 36 tok/s
Aufwand: 2–3 Tage (Codegen-Template + Validation)
Phase-2-Prio: P0
Begründung: v0.x hat den Kernel (gemv_q4_k_q8_inline_residual_multi_row,
            41.7 µs × 5598 Calls = 233 ms, 10.5 %). Der GPU-Gewinn ist
            moderat (~40 ms/Run), aber jede gefuserte Operation schiebt
            die Zahl der Dispatches pro Token in Richtung v0.x (608 → 462).
```

### P0 — Stream-Sync nach Bandit-Konvergenz abschalten

```
Optimierung:      Phase-2-Epilog-Dirty-Flag-Telemetrie (Arch-Doc §2.6)
Aktuell:          stream.synchronize() nach jedem gemv_tuned()
                  → 83 260 Syncs, 97.9 % der HIP-API-Wallclock
Geschätzter Speedup: 5–10 % Wallclock-Reduktion durch Kernel-Pipelining
Erwarteter Decode-Gewinn: 30 → 32–33 tok/s
Aufwand: 1–2 Tage (Bandit-Phasen-Flag, HIP-Events als Messpfad)
Phase-2-Prio: P0
Begründung: v0.x zeigt 114 Syncs statt 83 260 — die GPU kann parallel
            arbeiten wenn der CPU-Dispatch nicht synchronisiert. Der
            Bandit braucht Timing nur während Exploration (~100 Pulls
            pro Shape); danach ist die Variant-Wahl stabil.
```

### P1 — Q6_K LM-Head-Kernel optimieren

```
Optimierung:      Q6_K Q8-inline oder WMMA-Variante bauen
Aktuell:          q6_k_standard 85.6 µs × 4253 Calls = 364 ms (13.7 %)
                  inkl. LM-Head 4096×151936 pro Token
Geschätzter Speedup: Faktor 2–3× (v0.x q6_k_q8_inline läuft ~60 µs vs v1.0 85 µs)
Erwarteter Decode-Gewinn: 30 → 33 tok/s
Aufwand: 2 Tage (Codegen-Erweiterung um Q6_K Q8-inline-Variante)
Phase-2-Prio: P1
Begründung: Die Bandit hat hier keine Wahl (nur 1 Variante registriert).
            Sobald 2 Varianten existieren, konvergiert UCB1 automatisch.
```

### P1 — Batched-WMMA-Prefill

```
Optimierung:      Prefill vom sequentiellen execute_decode-Loop auf
                  WMMA-basierten batched forward-pass umstellen.
Aktuell:          Prefill 31 tok/s (33 tok in 1.07 s Wallclock-Anteil)
                  Derselbe Kernel-Pfad wie Decode, aber 33x ausgeführt.
Geschätzter Speedup: 10–30× (v0.x schafft ~290 tok/s im Warmup-Prefill,
                  genau das Pattern das die WMMA-Kernels aus Step 1.7
                  emittieren wollten aber nicht wired sind)
Erwarteter Decode-Gewinn: —  (Prefill-only)
Erwarteter Prefill-Gewinn: 31 → 300+ tok/s
Aufwand: 3–4 Tage (WMMA-Prefill-Pfad im Executor + batch-dimension
                  an Attention/RoPE/Norm weiterreichen)
Phase-2-Prio: P1
Begründung: Infrastruktur komplett (hip_kernels_v1/wmma/*.hip existiert);
            es fehlt das Executor-Wiring und ein batched Attention-Kernel.
```

### P2 — Gate+Up+SwiGLU Q8-inline

```
Optimierung:      Q8-inline-Variante von gate_up_swiglu bauen (aktuell nur standard)
Aktuell:          421 µs/Call × 4139 = 1745 ms (65 %)
Geschätzter Speedup: 5–10 % pro Call — v0.x gate_up_swiglu ist Q8-inline
                  und läuft bei 394 µs — also nur 6 % schneller. Das
                  Volumen (FFN 4096→14336 dominiert durch reine
                  Bandbreite) lässt wenig Spielraum.
Erwarteter Decode-Gewinn: 30 → 32 tok/s
Aufwand: 2–3 Tage
Phase-2-Prio: P2
Begründung: Trotz 65 % Anteil ist der absolute Spielraum begrenzt, weil
            v0.x mit demselben Kernel-Pattern auch schon bei 394 µs lief.
            Dieser Kernel ist der BW-Close-To-Limit-Pfad.
```

### P2 — Embedding-Dequant on-the-fly

```
Optimierung:      Token-Embedding-Lookup direkt auf quantisierten Daten
Aktuell:          Executor dequantisiert einmal beim Start (2.37 GB)
Geschätzter Speedup: 0 % Decode — das ist ein VRAM-Gewinn, kein Perf-Gewinn
Erwarteter VRAM-Gewinn: −2.37 GB → enables 14B Q4_K_M unter 16 GB VRAM
Aufwand: 1–2 Tage
Phase-2-Prio: P2
Begründung: Aktuell 10 GB VRAM bei 8B; 14B läge bei ~16 GB — diese
            2.37 GB sind die Differenz zwischen "passt" und "OOM".
```

### P2 — Graph-Traversal-Overhead reduzieren

```
Optimierung:      Replay-Slot-Buffer oder Statische Dispatch-Sequenz
Aktuell:          129 Iterationen × 543 Node-Dispatches × Hashmap-Lookup
                  ≈ 50 µs CPU-Overhead pro Iteration = 0.1 %
Phase-2-Prio: P2 (Rauschen-Level)
```

## Theoretisches Maximum nach P0+P1

Wenn alle P0- und P1-Vorschläge umgesetzt sind:

| Metrik | heute | nach P0+P1 | Berechnung |
|---|---:|---:|---|
| Decode tok/s | 30 | **~40** | 30 × 1.25 (P0-Fusion) × 1.07 (P0-Sync) × 1.03 (P1-Q6_K) |
| Prefill tok/s | 31 | **~300** | batched WMMA, 10× (v0.x-Baseline) |
| BW-Effizienz | 21 % | ~30 % | |

**~40 tok/s Decode ist nahezu bei v0.3.0 (41.7).** Um signifikant
*über* v0.x zu kommen, müssen wir über P0+P1 hinaus: Säule 6
VALU-Parity-Validierung + GA-getunte Pareto-Kandidaten, die
v0.x-Kernel systematisch schlagen. Das ist Phase 2 Arbeit.

**Das Arch-Doc-Ziel von 125 tok/s** (88 % der BW-Grenze) ist mit
Phase-1-Architektur nicht erreichbar. Dafür braucht es (Phase 2):
- FP8-Weight-Path (halbierte Bandbreite pro Tensor)
- Offline-GA-optimierte Kernels
- Kernel-Fusion auf Layer-Ebene (nicht nur 2-Op-Fusion)

## Zusammenfassung

```
Top-3 Kernel (% der Decode-Zeit, Tuned-Run):
  1. gate_up_swiglu Q4_K standard   — 65.4%  (421 µs/Call)
  2. Q4_K Q8-inline (Q/K/V/O)       — 16.0%  ( 26 µs/Call)
  3. Q6_K standard (LM head)        — 13.7%  ( 86 µs/Call)

Säulen-Overhead:
  Bandit Sync pro GEMV              : 97.9 % der HIP-API-Wall (aber
                                      mostly GPU-Wartezeit, nicht reine CPU)
  Monitor hipMemcpy + Check         : < 0.1 %
  CPU Graph-Traversal               : < 0.1 %

Bottleneck: NICHT BW-bound (21 % BW-Nutzung), NICHT Compute-bound
            (Occupancy OK), Dispatch-bound durch Bandit-Sync
            + Kernel-Dauer-bound durch fehlende Fusion.

GPU-Effizienz (GPU-Zeit / Wallclock): 2.668 s / 4.255 s = 62.7 %
BW-Effizienz: 21 % von 640 GB/s

VRAM-Peak: 10.05 GB / 17.1 GB (7 GB Reserve)

Phase-2-Prioritäten:
  P0: Residual-fused GEMV              → +20–30 %   → ~36 tok/s
  P0: Stream-Sync nach Konvergenz aus  → +5–10 %    → ~33 tok/s
  P1: Q6_K Q8-inline / WMMA            → +10 %      → ~33 tok/s
  P1: Batched-WMMA-Prefill             → 10–30×     → Prefill 300 tok/s
  P2: gate_up Q8-inline                → +5 %       → ~32 tok/s
  P2: Embedding on-the-fly             → VRAM −2.4 GB, enables 14B

Theoretisches Phase-2-Maximum: ~40 tok/s Decode (an v0.x-Grenze),
300+ tok/s Prefill. Um 125 tok/s zu erreichen braucht es FP8-Path +
GA-Kernels + Layer-Fusion (Phase 2 GA).
```
