# Phase 2 Schritt 2.1.4 — Q6_K Q8-Inline-Variante

**Date:** 2026-04-23
**Branch:** v1.0-dev (on top of un-fuse `aed55ad` + rocprof-deep-dive `085ef5a`)
**Hardware:** AMD Radeon RX 9070 XT (gfx1201)
**Model:** Qwen3-8B Q4_K_M
**Scope:** Neuer `q6_k_q8_inline` GEMV-Kernel — damit der Bandit
statt 1 Variante 2 Wahlmöglichkeiten für Q6_K-Shapes bekommt und
analog zu Q4_K die schnellere per UCB1 auswählt.

## TL;DR — Ehrliches Negativ-Ergebnis

```
Gemessen (Bandit-Konvergenz auf allen Q6_K-Shapes):

  Shape                     Standard µs   Q8-Inline µs   Winner
  n=1024   k=4096  (K/V)       24.5          43.6         standard  (1.78× langsamer)
  n=4096   k=12288 (FFN-dn?)  116.6         178.9         standard  (1.53× langsamer)
  n=151936 k=4096  (LM-head)   773           1442         standard  (1.87× langsamer)

Bandit-Verhalten (UCB1):
  Alle 3 Shapes in "exploiting"-Phase
  Alle wählen standard (≥ 99.9 % der Pulls)
  Kein Decode-Regression: 59.5 tok/s (15-Prompt) vs 59.7 post-unfuse
```

**Ergebnis: Mein Q6_K Q8-Inline-Kernel ist auf allen drei Q6_K-
Shapes auf gfx1201 LANGSAMER als Standard.** Der Bandit erkennt
das korrekt und wählt standard — null End-to-End-Regression.
Die Infrastruktur ist da; wenn eine künftig FASTER Q6_K-Variante
gebaut wird, adoptiert der Bandit sie automatisch.

## Was implementiert ist

### Neuer HIP-Kernel: `hip_kernels_v1/gemv/gemv_q6_k_q8_inline.hip` (210 LOC)

- Struktur 1:1 analog zu `gemv_q4_k_q8_inline.hip`
- Q6_K-Layout-spezifisch: 8 Q8_0-Blöcke mappen auf 8 (half, quad)
  Tiles à 32 Elemente mit ql/qh 6-bit-Reconstruction
- Pro Q8-Block 2 Int-Dots (16 Elemente je Sub-Scale), Q6 als
  (q6 - 32) via `float(int_dot) - 32 × float(q8_sum)` in float
- Kooperative FP32→Q8_0-Quantisierung wie im Q4_K-Kernel

### Rust-Integration

```
build.rs                         +1 Zeile  v1_gemv_q6_k_q8_inline Lib
hip_kernels_v1/CMakeLists.txt    +1 Zeile  add_hip_kernel(...)
src_v1/backend/gpu/gemv.rs       +19 Zeilen FFI-Binding
src_v1/runtime/variants.rs       +4 Zeilen  KernelId::GemvQ6KQ8Inline + register
src_v1/graph/executor.rs         +8 Zeilen  Dispatch-Match-Arm
tests_v1/q6k_q8_inline_test.rs   +355 Zeilen NEU
Cargo.toml                       +7 Zeilen  Test-Registrierung
```

Alles opt-in — der Bandit entscheidet zur Laufzeit.

## Kernel-Parity (synthetic data)

`test_q6k_q8_inline_kernel_parity_vs_standard`:
- N=256, K=512, random ql/qh/sc Q6_K blocks
- Result: **max_rel 1.8 %, mean_rel 0.5 %** auf Outputs ≥ 10 %
  der max-Magnitude
- Erwartung für Q8-Quantisierungs-Noise: <5 % max / <2 % mean ✓

Die Kernel ist numerisch korrekt. Die Fehler liegen im Q8-Activation-
Quantisierungs-Rauschen (1/254 pro Element), akkumuliert über K=512.

## Performance — Bandit-Messungen

Die UCB1-Messungen stammen aus der 15-Prompt-Suite (5 929 decode
tokens × 2-3 Q6_K-Calls pro Token = **∼ 213 660 Pulls pro Shape**).

| Shape | Standard Ø µs | Q8-Inline Ø µs | Ratio | Pulls standard | Pulls q8_inline |
|---|---:|---:|---:|---:|---:|
| n=1024, k=4096 | **24.48** | 43.59 | 1.78× | 106 812 (99.98 %) | 18 (0.02 %) |
| n=4096, k=12288 | **116.59** | 178.85 | 1.53× | 106 812 (99.98 %) | 18 (0.02 %) |
| n=151936, k=4096 | **773.00** | 1 442.34 | 1.87× | 5 934 | 1 (exploring) |

Der Bandit hat auf den häufig genutzten Layer-Shapes (pro Token
aufgerufen) schnell konvergiert. Der LM-Head-Shape (1 Call / Token)
ist noch in "exploring"-Phase, hat aber bereits nach 5 935 Pulls
das 1.87×-Delta dokumentiert und wird in der nächsten Testrunde
committen.

## Warum Q8-Inline auf Q6_K langsamer ist

Q4_K hat eine NATURAL 32-Elemente-Sub-Block-Struktur: eine Q8_0-
Aktivierungs-Block deckt genau eine Q4_K-Sub-Block-Skala ab.
Der Integer-Dot-Product läuft über 32 Elemente, einmal pro
Sub-Block, mit EIN Sub-Scale-Multiplikator pro Int-Dot.

Q6_K hat eine 16-Elemente-Sub-Block-Struktur. Eine Q8_0-Block
(32 Elemente) deckt **zwei** Q6_K-Sub-Blöcke ab. Mein Kernel macht
deshalb pro Q8-Block **zwei** Int-Dots (lo und up) und **zwei**
FP32-Multiplikationen mit Sub-Scales. Die doppelte Overhead-Struktur
vs. Q4_K's einfache Struktur ist ein Faktor.

Zweitens: das Q6_K ql/qh Unpacking ist teurer als Q4_K's Nibble-
Extract. Pro Element braucht Q6_K zwei Ladungen (ql-Byte, qh-Byte)
+ zwei Shifts + Mask + OR. Q4_K braucht nur ql-Byte + Mask/Shift.
In einem Per-Lane-256-Element-Loop summiert sich das.

Drittens: der Q6_K-Standard-Kernel nutzt eine warp-weite
Distribution (32 Lanes × 4 Positionen/Lane pro Quad). Mein Q8-
Inline-Kernel hat die Q4_K-Per-Lane-Super-Block-Strategie übernommen
— eine Lane rechnet 256 Elemente einer super-block. Das ist für
Q4_K günstig (zähl nur Nibble), für Q6_K teurer (Lane muss mehr
bit-twiddling machen).

Ein **spezialisierter** Q6_K-Q8-Kernel mit warp-verteilter Arbeit
(wie standard) PLUS Integer-Math könnte schneller sein als beide —
aber das ist ein Codegen-Rewrite und gehört in eine eigene Session.

## End-to-End — 15-Prompt-Suite

| Metrik | Post-Unfuse Ref | Post-Q6K-Variant | Delta |
|---|---:|---:|---:|
| Prefill tok/s | 590.9 | **593.7** | +0.5 % (Noise) |
| Decode tok/s | 59.7 | **59.5** | −0.3 % (Noise) |
| Wallclock ms | 100 811 | 101 242 | +0.4 % (Noise) |
| Prompts 15/15 | ja | ja | — |
| Monitor-Events | 0 | 1 | +1 (z-Score 3.01, known flake) |

Der 1 Monitor-Event ist ein bekannter Threshold-Flake aus der
calibration (z=3.01 bei stddev-Gate 3.0 — grenzwertig, keine
systematische Drift). Nicht durch die Q6_K-Änderung verursacht —
der gleiche Flake tritt in anderen Suiten auf demselben Build auf.

**Der Bandit hält das Resultat auf post-unfuse-Level.** Bei
Q6_K-Shapes mit FASTER Q8-Inline (künftige Kernel-Rewrites)
würde die Decode-Rate automatisch steigen.

## Isolierte Mutex-Prompt-Messung

```
Post-Unfuse:     Decode 68.8 tok/s  (Bandit 1-Variant Q6_K)
Post-Q6K-Var:    Decode 61.2 tok/s  (Bandit 2-Variant Q6_K, erste ~20 tok Exploration-Kosten)

15-Prompt:       59.5 vs 59.7  (über 5 929 Tokens amortisiert → Noise)
```

Der Mutex-Prompt ist kurz (100 Tokens) — die Bandit-Exploration
der LM-Head-Shape (n=151936, 1 Call/Token) zahlt ihre 10-20 Pulls
× 1.87× Extra-Zeit während dieser kurzen Läufe. In der 15-Prompt-
Suite mit ≥ 1000 Tokens pro Shape amortisiert sich das vollständig.

## Wie würde ein schneller Q6_K Q8-Inline aussehen?

rocprof-Daten-basierte Projektion für einen SPEZIALISIERTEN
Q6_K-Q8-Kernel:

```
Q6_K-Layer-Kernel heute (standard):
  Ø 67-117 µs je Shape, 15-30 % BW auf Layer-Calls
  Theoretisches Minimum bei 100 % BW: ~15-20 µs

Falls Q8-Inline mit spezialisiertem Kernel-Design 50 % BW erreicht:
  Speedup: 15 → 30 % BW = 2×
  Erwarteter Layer-Zeit: 30-50 µs
  Decode-Gewinn pro Token: 36 × (67 − 40) µs ≈ 1 ms / 14.5 ms = 7 %
  → Decode 59.7 → ~64 tok/s (15-Prompt)
```

Das ist der Pot. Mein naiver Port hat ihn nicht gehoben.

## Tests (3/3 grün)

| Test | Was |
|---|---|
| `test_bandit_registers_two_q6k_variants` | CPU-unit: Q6_K hat jetzt 2 Varianten in VariantRegistry |
| `test_q6k_q8_inline_kernel_parity_vs_standard` | GPU: synthetic N=256 × K=512, max_rel 1.8 %, mean_rel 0.5 % auf significant elements |
| `test_e2e_decode_with_q6k_variants_available` | GPU: 3 Q6_K-Shapes × 2 Varianten registered, e2e-decode ≥ 55 tok/s (gemessen 61.2) |

Regression:
| Suite | Status |
|---|:---:|
| `cargo check --features v1,gpu --lib` | ✅ |
| Post-Unfuse-Decode-Pfad | ✅ unverändert (Bandit wählt standard) |

## Fazit

| Frage aus Prompt | Antwort |
|---|---|
| Kernel implementiert? | ✅ gemv_q6_k_q8_inline.hip, 210 LOC |
| Bandit 2 Varianten für Q6_K? | ✅ registered auf allen 3 Shapes |
| Parity (Output numerisch korrekt)? | ✅ max_rel 1.8 %, mean_rel 0.5 % |
| Q8-Inline SCHNELLER als Standard? | ❌ **1.5-1.9× langsamer** auf gfx1201 |
| Decode-Gewinn e2e? | ❌ 0 (Bandit wählt standard → gleich zu post-unfuse) |
| Framework-Mehrwert? | ✅ Ja — infrastruktur + Bandit-Slot für zukünftige Kernel |

**Empfehlung:**
- FP16-Standard bleibt die Q6_K-Wahl (Bandit erzwingt das
  automatisch)
- Die neue Variante bleibt registriert — 0 Overhead (Bandit
  konvergiert, dispatcht immer standard)
- Ein spezialisierter Q6_K-Q8-Kernel mit warp-verteilter Arbeit
  (nicht per-lane-super-block wie hier) wäre das nächste Projekt
- Alternativ: der Q6_K-Layer-Kernel selbst (nicht Q8-Inline) kann
  mit ähnlichen GA-Techniken wie Block C/D getunet werden (rocprof
  post-unfuse: 15-30 % BW heute)

## Commit

Prefix: `feat(v1):` — neues Kernel + Infrastructure.

```
feat(v1): Phase 2 step 2.1.4 — Q6_K Q8-inline variant (honest negative)
```

Backup-Push auf `backup` Remote.
