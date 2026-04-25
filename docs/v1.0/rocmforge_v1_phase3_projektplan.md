# ROCmForge v1.0 — Phase 3 Projektplan

**Stand:** 24.04.2026
**Basis:** Phase 2 abgeschlossen (96.2 tok/s Decode, 580 tok/s Prefill)
**Hardware:** RX 9070 XT (gfx1201), Ryzen 9 7945HX, 64 GB RAM

---

## Ausgangslage

### Was Phase 2 erreicht hat

```
Decode:    30 → 96.2 tok/s (+221%)    8/15 Prompts ÜBER llama.cpp
Prefill:   31 → 580 tok/s (+1770%)    aber 8.6× UNTER Arch-Doc Ziel!
Features:  Multi-Turn, Streaming, FP8-KV-Cache, Chat-REPL, Think-Filter
Modelle:   Qwen3-8B ✅, Llama-3.1 ⚠ (Known Limitation), Rest v1.1
```

### Wo wir vs. Arch-Doc stehen

```
                    Arch-Doc Ziel    Aktuell      Gap        Status
Decode 8B Q4_K_M:   125 tok/s        96.2        1.30×      ⚠ akzeptabel
Prefill 8B pp256:   7 500 tok/s      580         12.9×      ❌ KRITISCH
Llama-3.1 Multi-T:  funktioniert     Known Lim   —          ❌ Root Cause offen
VALU-Parity:        1/1000 Sampling  nicht impl  —          ⬜ deferred
CPU-Backend:        ≥ 8 tok/s        nicht impl  —          ⬜ deferred
rf-forge CLI:       tune-all+bench   nicht impl  —          ⬜ teilweise deferred
```

### Warum Prefill so weit unter dem Ziel liegt

```
Das Arch-Doc ging von DREI Prefill-Hebeln aus:

1. FP8-WMMA als Default (2× WMMA-Rate vs FP16)
   → Phase 2 Ergebnis: FP8-WMMA Prefill war 0.75× (LANGSAMER!)
   → Ursache: Q4_K → FP32 → FP8 Dequant-Overhead frisst den Gain
   → ABER: Das war auf Q4_K Weights. Native FP8-Weights (v1.1) 
     könnten den Vorteil realisieren
   → Für v1.0: FP16-WMMA bleibt Default

2. GA-getuntes WMMA-Tiling (TILE_M, TILE_N, K_CHUNK)
   → NICHT gemacht — wir haben Decode optimiert statt Prefill
   → Die Default-Tiles sind wahrscheinlich WEIT vom Optimum
   → llama.cpp's Tile-Configs sind community-getuned über Jahre
   → GRÖSSTER ungenutzter Hebel

3. 128 AI Accelerators (2 WMMA pro CU) voll auslasten
   → Occupancy-Analyse zeigt: WMMA-Kernel nutzt ~60% der CUs
   → Register-Pressure (192 VGPRs) limitiert Waves/CU
   → Tile-Tuning + Register-Optimierung könnte 40% heben

Aktuell: 580 tok/s bei ~8% Compute-Auslastung der WMMA-Units
Ziel:    5000+ tok/s bei ~60%+ Auslastung
```

---

## Phase 3 Struktur — 4 Blöcke

```
Block P0: Prefill-Optimierung          12-15 Tage    KRITISCH
Block P1: Decode Restgap + Attention    5-7 Tage     WICHTIG  
Block P2: Infrastruktur + Release       5-7 Tage     MUSS
Block P3: Dokumentation + v1.0.0        3-4 Tage     MUSS

Gesamt: ~25-33 Tage (~5-6 Wochen)
```

---

## Block P0: Prefill-Optimierung (580 → 5000+ tok/s)

**Das ist der P0-Block.** 580 tok/s bei einem Ziel von 7500 ist das größte
Gap im gesamten Projekt. Ohne Prefill-Fix ist v1.0 bei langen Prompts 
DEUTLICH langsamer als llama.cpp (580 vs 1127).

### P0.1: WMMA-Tile-Config Analyse + Tuning (Tag 1-4)

```
AKTUELL:
  WMMA-GEMM mit Default-Tiles:
    TILE_M = 16, TILE_N = 16, K_CHUNK = 32 (vermutlich)
    → Keine GA, keine Optimierung, nur "es funktioniert"

ANSATZ:
  1. rocprof auf Prefill-Pfad (WMMA-Kernels isoliert messen)
     → Welcher WMMA-Kernel ist der Bottleneck?
     → Wie viel % der Prefill-Zeit ist WMMA vs Attention vs Overhead?
  
  2. WMMA-Tile-Config-Raum durchsuchen:
     TILE_M ∈ {16, 32, 64, 128}
     TILE_N ∈ {16, 32, 64, 128}  
     K_CHUNK ∈ {16, 32, 64}
     Double-Buffering: on/off
     → 4 × 4 × 3 × 2 = 96 Konfigurationen
     → Brute-Force in ~30 Min (jede Config ~20s auf einem Shape)
  
  3. Top-3 Configs pro Shape identifizieren
  
  4. llama.cpp WMMA-Tiles als Referenz analysieren:
     → ~/tmp/llama.cpp/ggml/src/ggml-cuda/mmq.cu
     → Welche Tile-Sizes nutzt llama.cpp auf gfx1201?

Erwartung: 580 → 1200-1500 tok/s (+100-160%)
Gate: Prefill ≥ 1000 tok/s
Aufwand: 3-4 Tage
```

### P0.2: llama.cpp GEMM-Kernel Analyse + Port (Tag 5-10)

```
ANALOG zum MMVQ-Port für Decode:
  → llama.cpp's batched GEMM Kernel analysieren (mmq.cu)
  → Thread-Mapping, Tiling, Memory-Access-Pattern verstehen
  → Als neuen WMMA-Kernel portieren
  → Bandit wählt zwischen unserem und dem portierten

WARUM DAS FUNKTIONIEREN SOLLTE:
  → MMVQ-Port hat Decode von 62.7 → 96.2 tok/s gebracht (+53%)
  → Gleiche Methodik: llama.cpp's bewiesenen Code 1:1 portieren
  → llama.cpp macht 1127 tok/s Prefill auf unserer Hardware
  → Der Kernel IST der Beweis dass es geht

SCOPE:
  → ~/tmp/llama.cpp/ggml/src/ggml-cuda/mmq.cu (Batched GEMM)
  → Analyse: Thread-Mapping, Tile-Sizes, Memory-Pattern
  → Port: als neuen HIP-Kernel
  → Integration: Bandit-Variante oder direkter Ersatz

Erwartung: 1500 → 3000-4000 tok/s
Gate: Prefill ≥ 2500 tok/s
Aufwand: 5-6 Tage (Analyse 2 + Port 2 + Integration 1 + Test 1)
```

### P0.3: Prefill-Attention Optimierung (Tag 11-14)

```
AKTUELL:
  attention_prefill ist naiv O(N²) causal softmax
  Bei 256 Tokens: N² = 65536 Attention-Scores → LANGSAM
  Bei 1024 Tokens: N² = 1M → DOMINIERT die Prefill-Zeit

ANSATZ:
  Option A: Flash-Attention Prefill
    → Tiled Softmax, O(N × block_size) statt O(N²)
    → ~400-500 LOC neuer Kernel
    → Hilft auch für Decode bei langem Kontext (>8K)
    → 3-5 Tage

  Option B: llama.cpp Prefill-Attention portieren
    → Gleiche Methodik wie MMVQ/GEMM-Port
    → Bewiesener Code
    → 3-4 Tage

Erwartung: +30-50% auf langen Prompts (>200 Tokens)
Gate: Prefill pp256 ≥ 4000 tok/s
Aufwand: 3-5 Tage
```

### P0.4: FP8-WMMA Re-Evaluation (Tag 14-15)

```
Phase 2 Ergebnis: FP8-WMMA war 0.75× FP16 (LANGSAMER)
  → Ursache: Q4_K → FP32 → FP8 Dequant-Overhead
  → ABER: Nach WMMA-Tile-Tuning könnte sich das ändern!
  → Die Tile-Config beeinflusst wie VIEL Dequant-Overhead
    relativ zum WMMA-Compute anfällt
  → Bei größeren Tiles: Dequant amortisiert sich besser

ANSATZ:
  → FP8-WMMA NUR auf den BESTEN Tile-Configs aus P0.1/P0.2 testen
  → Falls FP8 ≥ FP16: als Option aktivieren (+2× WMMA-Rate)
  → Falls FP8 < FP16: endgültig deferred auf native FP8-Weights (v1.1)

Erwartung: Entweder +50-100% (wenn FP8 funktioniert) oder 0% (wenn nicht)
Gate: Ehrliches Ergebnis
Aufwand: 1-2 Tage
```

### P0 Gesamt-Erwartung

```
  Schritt     Prefill tok/s    Kumulativ
  ──────────────────────────────────────
  Start       580              580
  P0.1 Tiles  1200-1500        1200-1500
  P0.2 GEMM   3000-4000        3000-4000
  P0.3 Attn   +30-50%          4000-5000
  P0.4 FP8    +0-100%          4000-7500
  ──────────────────────────────────────
  ZIEL:       ≥ 5000 tok/s

  llama.cpp:  1127 tok/s (gemessen)
  Arch-Doc:   7500 tok/s (optimistisch)
  Realistisch: 4000-5000 tok/s → 3.5-4.5× ÜBER llama.cpp!
```

---

## Block P1: Decode Restgap + Attention (96.2 → 100+ tok/s)

### P1.1: Attention-Decode Tuning (Tag 16-19)

```
Prompts 11,12 laufen bei 88-92 tok/s (vs 96-104 auf kurzen Prompts)
  → Attention ist der Bottleneck bei seq_len > 200
  → Aktuell: Skalare Implementierung, O(seq_len)
  → LDS-Limit: ~12k Tokens

ANSATZ:
  → Attention-Kernel profilen (rocprof isoliert)
  → Memory-Access-Pattern optimieren (Coalescing)
  → LDS-Tiling für Score-Berechnung
  → NICHT Flash-Attention (zu komplex für Decode-Batch=1)
  → ABER: Tiling innerhalb des skalaren Kernels

Erwartung: Prompts 11,12 von 88-92 auf 94-96 tok/s
           Aggregat: 96.2 → 97-98 tok/s
Gate: Alle Prompts ≥ 92 tok/s
Aufwand: 3-4 Tage
```

### P1.2: Decode Parity-Analyse (Tag 20-21)

```
Falls nach P0+P1.1 der Decode immer noch < 100 tok/s:
  → rocprof Deep-Dive: Wo sitzen die letzten 4%?
  → Vergleich mit llama.cpp Kernel-Zeiten
  → Identifiziere den "Long Tail" an kleinen Kerneln

Falls ≥ 100 tok/s:
  → Dokumentieren und weiter zu P2

Aufwand: 1-2 Tage
```

---

## Block P2: Infrastruktur + Release-Vorbereitung

### P2.1: rf-forge bench (Tag 22-24)

```
Minimales Benchmark-Tool:
  rf-forge bench --model ~/models/X.gguf --runs 5 --report bench.md

Outputs:
  → Decode tok/s (Median, P5, P95)
  → Prefill tok/s (Median, P5, P95)
  → Per-Prompt Breakdown
  → Vergleichstabelle mit vorherigem Run

KEIN tune-all, KEIN tune-kernels, KEIN tune-precision
  → Das ist v1.1 (GA-Offline-Tool)
  → v1.0 braucht NUR das Benchmark-Subcommand

Aufwand: 2-3 Tage
```

### P2.2: Qwen2.5 Support (Tag 25-27)

```
Aus Phase-1-Backlog + Multi-Modell-Test:
  → Q4_0 GEMV-Kernel (Qwen2.5-7B nutzt Q4_0, nicht Q4_K_M)
  → Attention-Bias-Add (Qwen2.x hat Q/K/V-Biases)
  → Chat-Template: Qwen2 ChatML (ähnlich Qwen3 ohne /no_think)
  → Tokenizer: GPT-2 BPE (wie Qwen3 — sollte funktionieren!)

Tests: Qwen2.5-7B laden, Single-Turn, 5-Prompt Smoke
Erwartung: Qwen2.5 funktioniert mit ~90-100 tok/s
Aufwand: 2-3 Tage
```

### P2.3: Finale Benchmarks + Regression (Tag 28-29)

```
Benchmark-Matrix (alle Modelle × alle Metriken):

  ┌───────────────────┬────────┬─────────┬──────────┬──────────┐
  │ Modell            │ Decode │ Prefill │ Qualität │ llama.cpp│
  ├───────────────────┼────────┼─────────┼──────────┼──────────┤
  │ Qwen3-8B Q4_K_M   │ ≥96    │ ≥4000   │ 15/15    │ 99/1127  │
  │ Llama-3.1-8B      │ ≥100   │ ≥4000   │ Known    │ 105/1246 │
  │ Qwen2.5-7B Q4_0   │ ≥90    │ ≥3000   │ ≥12/15   │ —        │
  └───────────────────┴────────┴─────────┴──────────┴──────────┘

Aufwand: 1-2 Tage
```

---

## Block P3: Dokumentation + v1.0.0 Release

### P3.1: Dokumentation (Tag 30-32)

```
1. README.md (Neu):
   → Quick-Start (3 Befehle: build, download model, run)
   → Performance-Tabellen (Decode + Prefill, vs llama.cpp)
   → Architektur-Übersicht (6-Säulen-Diagramm)
   → Unterstützte Modelle + Known Limitations
   → CLI-Referenz (--prompt, --interactive, --show-think, etc.)

2. CHANGELOG.md:
   → v1.0.0 Release-Notes
   → Alle Phase 1-3 Meilensteine

3. docs/v1.0/user_guide.md:
   → Installation (Arch Linux, ROCm 7.2+)
   → Modell-Download (GGUF-Quellen)
   → Chat-Modus, Streaming, Multi-Turn
   → FP8-KV-Cache aktivieren
   → Troubleshooting (häufige Probleme)

4. Known Issues:
   → Llama-3.1 Instruction-Following (SNR <1)
   → Mistral (SPM-Tokenizer nötig → v1.1)
   → Gemma-4 (Architektur nicht unterstützt → v1.1)
   → Max-Kontext 8192 (Attention-Kernel-Limit)
   → MoE nicht unterstützt → v1.1

Aufwand: 2-3 Tage
```

### P3.2: Release-Vorbereitung (Tag 33)

```
1. Alle Tests grün (aktuell ~150+, nach Phase 3 ~200+)
2. cargo clippy clean
3. Version 1.0.0 in Cargo.toml
4. Git Tag: v1.0.0-rc1
5. Release-Notes finalisieren

KEIN Push auf GitHub ohne Bestätigung (Memory #2)
v1.0-Code bleibt lokal

Aufwand: 1 Tag
```

---

## Deferred auf v1.1

```
Diese Items sind BEWUSST aus v1.0 ausgeschlossen:

Performance:
  → Precision-GA NSGA-II (Suchraum leer bei Q4_K)
  → ASM-Codegen-Backend (Inline-ASM für Hot-Loops)
  → Flash-Attention-Decode (>12k Kontext)

Modell-Support:
  → SentencePiece Tokenizer (Mistral, Llama-2, CodeLlama)
  → MoE-Support (Qwen3-30B, Gemma4-26B)
  → Gemma-4 Architektur (PLE, Hybrid-Attention, Shared-KV)
  → Sliding-Window-Attention (Mistral, Gemma)
  → 14B Modelle (braucht Embedding-on-the-fly für VRAM)

Infrastruktur:
  → CPU AVX-512 Backend (Embedding on-the-fly, ~8 tok/s)
  → VALU-Parity 1/1000 Sampling (Safety)
  → Quality-Monitor Eskalation (FP8→FP16 automatisch)
  → rf-forge tune-all/tune-kernels/tune-precision
  → Bandit-State-Persistenz (über Runs hinweg)
  → TheRock/ROCm 7.9+ Evaluation
```

---

## Zeitplan-Übersicht

```
  Woche 1 (Tag 1-5):    P0.1 WMMA-Tile-Tuning + P0.2 GEMM-Analyse
  Woche 2 (Tag 6-10):   P0.2 GEMM-Port + Integration
  Woche 3 (Tag 11-15):  P0.3 Prefill-Attention + P0.4 FP8 Re-Eval
  Woche 4 (Tag 16-21):  P1 Decode Attention + Parity-Analyse
  Woche 5 (Tag 22-29):  P2 rf-forge + Qwen2.5 + Benchmarks
  Woche 6 (Tag 30-33):  P3 Dokumentation + Release v1.0.0-rc1

  Gesamt: ~6 Wochen (33 Arbeitstage)
  Puffer: ~20% eingerechnet (wie Phase 1+2)
```

---

## Exit-Kriterien v1.0.0

```
MUSS:
  ☐ Decode Qwen3-8B ≥ 96 tok/s (aktuell: ✅ erreicht)
  ☐ Prefill Qwen3-8B ≥ 3000 tok/s (aktuell: ❌ 580)
  ☐ Qwen3-8B 15/15 kohärent (aktuell: ✅ erreicht)
  ☐ Multi-Turn Alice-Test Qwen3 (aktuell: ✅ erreicht)
  ☐ Streaming + Think-Filter (aktuell: ✅ erreicht)
  ☐ FP8-KV-Cache funktioniert (aktuell: ✅ erreicht)
  ☐ CLI: --prompt, --interactive, --show-think
  ☐ README + User-Guide
  ☐ Alle Tests grün

SOLLTE:
  ☐ Decode ≥ 100 tok/s (Aggregat)
  ☐ Prefill ≥ 5000 tok/s
  ☐ Qwen2.5-7B funktioniert
  ☐ rf-forge bench funktioniert
  ☐ llama.cpp Decode-Parity (≤ 1.05×)

KANN (Stretch):
  ☐ Decode ≥ 125 tok/s
  ☐ Prefill ≥ 7500 tok/s
  ☐ Llama-3.1 Multi-Turn funktioniert
```

---

## Bewiesene Prinzipien (aus Phase 1+2, gelten für Phase 3)

```
1. IMMER profilen vor optimieren (5× bestätigt)
2. llama.cpp Kernel als GANZES portieren, nicht einzelne Aspekte
3. Bandit-Varianten KOSTEN Exploration — Verlierer DEREGISTRIEREN
4. Tied Bandit-Arms = Bandit-Stall = HIP-Graph blockiert
5. launch_index_spans SOFORT updaten bei neuen Kernel-Counts
6. FP8 auf Q4_K-Weights bringt NICHTS (Dequant-Overhead)
7. Elementwise-Epilog ist FREI, GEMV×GEMV-Fusion killt BW
8. WMMA-Prefill braucht CACHE-AWARE Attention bei Multi-Turn
9. Jede Performance-Projektion MESSEN, nicht glauben
10. Der Bandit ist das Safety-Net — Experimentieren ist billig
```
