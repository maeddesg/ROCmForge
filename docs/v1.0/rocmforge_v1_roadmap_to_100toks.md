# ROCmForge v1.0 — Roadmap to 100 tok/s

**Stand:** 23.04.2026 | **Hardware:** RX 9070 XT (gfx1201) | **Modell:** Qwen3-8B Q4_K_M

---

## Aktuelle Performance

```
ROCmForge v1.0 (Post-Cleanup, 23.04.2026):
  Decode:    62.7 tok/s (15-Prompt) / 68.8 tok/s (Mutex)
  Prefill:   580 tok/s
  BW-Eff:    54% von 640 GB/s
  Qualität:  15/15, 0 Monitor-Events
  HIP-Graph: AKTIV (nach Bandit-Konvergenz)

llama.cpp (Build 23b8cc4, gleiche Hardware + Modelle):
  Decode:    99.3 tok/s
  Prefill:   1127 tok/s
  BW-Eff:    ~69.5%

Gap:         Decode 1.58×, Prefill 1.94×
```

---

## Phase 2 Bilanz (abgeschlossen)

```
Treffer:
  ✅ WMMA-Prefill:        42 → 580 tok/s (+1280%)
  ✅ gate_up Unfusing:     40 → 68.8 tok/s (+62%)
  ✅ HIP-Graph + Cleanup:  56.5 → 62.7 tok/s (+11%)

Nieten (aber wertvolle Erkenntnisse):
  ❌ FP8-WMMA Prefill:     0.75× (Q4_K→FP8 Dequant-Overhead)
  ❌ Q6_K Q8-Inline:       1.5-1.9× langsamer (Sub-Block-Mismatch)
  ❌ sudot4 Kernel:         1.41× langsamer (LLVM optimiert bereits)
  ❌ Dispatch-Optimierung:  +0.6% (CPU war nur 2% der Token-Zeit)
  ❌ Epilogue-Fusion:       Nicht machbar (shared Inputs, cross-block RoPE)

Netto: Decode +109%, Prefill +1770%
```

---

## Token-Zeit-Zerlegung (definitiv)

```
16.0ms pro Token = 62.7 tok/s

  GPU Kernel-Zeit:       13.7ms  (86%)  ← BW 54%, llama.cpp 70%
  HIP-Driver Interna:     0.5ms  (3%)   ← reduziert durch HIP-Graph
  hipMemcpy (Logits):     0.6ms  (4%)   ← nötig
  hipStreamSync:          0.5ms  (3%)   ← nötig
  SetParams + Launch:     0.2ms  (1%)   ← HIP-Graph Replay
  Rust CPU + Sampling:    0.5ms  (3%)   ← optimiert

  Adressierbares Budget:  GPU Kernel (86%)
  → 54% → 70% BW = Kernel 13.7ms → 10.6ms → 86 tok/s
  → 54% → 80% BW = Kernel 13.7ms → 9.3ms → 97 tok/s
```

---

## Kernel-BW-Verteilung (Post-Cleanup rocprof)

```
Kernel                      % GPU    BW       Status
─────────────────────────────────────────────────────────
q4_k_q8_inline              48%      51%      ← HAUPTZIEL
q6_k_standard (LM-Head)     22%      95%      ← optimal
q4_k_q8_inline_residual     16%      69%      ← gut
attention_decode              8%      —        ← O(seq_len)
elementwise (norm/rope/etc)   6%      —        ← klein

→ q4_k_q8_inline (48% GPU, 51% BW) ist der einzige große Hebel
→ 51% → 70% BW auf diesem Kernel = Decode 62.7 → ~80 tok/s
```

---

## Der Pfad zu 100 tok/s — 3 Stufen

### Stufe 1: ✅ ERLEDIGT — Architektur-Optimierung (62.7 tok/s)

- WMMA-Prefill (14×)
- gate_up Unfusing (+62%)
- HIP-Graph Capture + Replay
- Bandit-Cleanup (bewiesene Verlierer deregistriert)
- Dispatch-Optimierung (HashMap→Vec)

### Stufe 2: Kernel-BW 54% → 70% (→ ~80-86 tok/s)

**Ansatz:** llama.cpp GEMV-Kernel als GANZES portieren, nicht einzelne Aspekte isolieren.

```
Was sudot4 allein uns GELEHRT hat:
  → Einzelne Optimierungen isoliert bringen NICHTS
  → Der Compiler optimiert unsere skalaren Loops BEREITS gut
  → Der Kernel ist bei 51% BW schon MEMORY-BOUND
  → Mehr Compute-Speed (sudot4) hilft nicht wenn Memory der Bottleneck ist

Was llama.cpp ANDERS macht (als KOMBINATION):
  1. Q8_1 Input-Format:  Vorberechneter sum-Term eliminiert eine 
                          Reduktion pro Sub-Block → weniger Register
                          → höhere Occupancy → mehr Memory-Requests in-flight
  2. VDR=2:              2 Blöcke pro Thread pro Iteration
                          → doppelt so viele outstanding Memory-Requests
                          → Memory-Pipeline besser gesättigt
  3. Tiling/Coalescing:  int32-aligned loads, besseres Byte-Layout
                          → weniger L2-Cache-Misses
  4. Gate-Fusion:         Gate+Up+SwiGLU im selben Kernel
                          → nur möglich wenn Compute schnell genug (Q8_1+VDR=2)

STRATEGIE: Den llama.cpp mul_mat_vec_q Kernel (mmvq.cu + vecdotq.cuh)
als REFERENZ-IMPLEMENTATION nach ROCmForge portieren:
  → vec_dot_q4_K_q8_1 (Inner-Loop mit dp4a)
  → Q8_1 Input-Quantisierung (d + sum)
  → VDR=2 Outer-Loop
  → nwarps=8 (haben wir schon!)
  → Als neue Bandit-Variante registrieren

Aufwand: ~1 Woche (Kernel + Q8_1-Quantisierung + Tests + Benchmarks)
Risiko:  NIEDRIG (bewiesener Pfad — llama.cpp macht 70% BW damit)
```

### Stufe 3: Re-Fusing + Feintuning (→ ~90-100 tok/s)

```
Voraussetzung: Stufe 2 hat Kernel-BW auf 70%+ gebracht

Mit schnellem Kernel wird Gate-Fusion WIEDER möglich:
  → llama.cpp fusiert Gate+Up+SwiGLU im GEMV (has_fusion Template)
  → Zwei Weight-Streams passen wenn der Compute schnell genug ist
  → Spart ~36 Dispatches/Token + 2 VRAM-Roundtrips/Layer

Weitere Hebel:
  → Flash-Attention für langen Kontext (>1024 Tokens)
  → FP8-KV-Cache (halbe Attention-BW)
  → Prefill WMMA-Tile-Tuning (580 → 1000+)
  → LM-Head Top-K Sampling (608 KB → 20 KB Readback)

Aufwand: ~1-2 Wochen
Risiko:  MITTEL (Gate-Fusion hat bei uns bei 20% BW versagt,
         muss mit neuem Kernel re-validiert werden)
```

---

## Physikalische Limits

```
BW-Limit für Qwen3-8B Q4_K_M (4.75 GB Weights):

  BW%     Kernel-ms   + Overhead   = Total    → tok/s
  ─────────────────────────────────────────────────────
  54%     13.7ms      + 2.3ms      = 16.0ms   →  62.7  ← JETZT
  65%     11.4ms      + 2.0ms      = 13.4ms   →  74.6
  70%     10.6ms      + 1.8ms      = 12.4ms   →  80.6  ← llama.cpp Niveau
  80%      9.3ms      + 1.5ms      = 10.8ms   →  92.6
  90%      8.2ms      + 1.3ms      =  9.5ms   → 105.3
  100%     7.4ms      + 1.2ms      =  8.6ms   → 116.3  ← HW-Limit

  llama.cpp erreicht: 99.3 tok/s ≈ ~75-80% BW
  Unser Ziel:         95-100 tok/s (Stufe 2+3)
```

---

## Nicht-Performance Roadmap

```
Phase 2.2: Precision-GA (FP8-KV-Cache, Pro-Layer Precision)
  → Kein Speed-Gewinn, aber: 50k statt 25k Max-Kontext
  → Bessere Qualität auf Llama-3.1 (SNR-Risk-Layer)

Phase 2.4: Model-Support
  → Llama-3.1 Multi-Turn Fix
  → Qwen2.5 Q4_0 Support  
  → Multi-Turn KV-Cache-Persistenz

Phase 3: Release-Vorbereitung
  → AVX-512 VNNI CPU-Pfad (Zen4)
  → rf-forge CLI-Tool
  → Streaming-Output
  → v1.0.0-rc1
```

---

## Bewiesene Prinzipien (aus Phase 2)

```
1. IMMER profilen vor optimieren (5× bestätigt)
2. GEMV×GEMV Fusion killt BW — Elementwise-Epilog ist frei
3. Bandit-Varianten KOSTEN Exploration — Verlierer DEREGISTRIEREN
4. FP8 auf Q4_K-Weights bringt NICHTS (Dequant-Overhead)
5. LLVM optimiert skalare Loops gut — naives Intrinsic schlägt Compiler nicht
6. HIP-Graph funktioniert auf gfx1201 ROCm 7.2.2
7. Einzelne Kernel-Aspekte isoliert portieren bringt NICHTS
   → llama.cpp GEMV als GANZES portieren (Q8_1+VDR=2+Tiling+Fusion)
8. Der Bandit ist das Safety-Net — Experimentieren ist billig
```

---

## Tages-Timeline (23.04.2026)

```
Phase 1 Ende:        30.0 tok/s Decode,   31 tok/s Prefill
Post-P0:             40.7 tok/s
WMMA-Prefill:        40.7 tok/s Decode,  590 tok/s Prefill (14×!)
gate_up Unfusing:    68.8 tok/s (+62%)
FP8-Prefill:         negativ (0.75×)
llama.cpp Benchmark: 99.3 tok/s (Referenz gemessen)
Q6_K Q8-Inline:      negativ (1.5-1.9×)
Dispatch-Opt:        +0.6%
sudot4:              negativ (1.41×)
Epilogue-Fusion:     nicht machbar (STOP)
HIP-Graph:           +0.7% isoliert
Cleanup + Graph:     62.7 tok/s (+11%)
nwarps=8:            schon aktiv (kein Fehler, kein Gewinn)

GESAMT: Decode +109%, Prefill +1770%, Gap 3.24× → 1.58×
```
