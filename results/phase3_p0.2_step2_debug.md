# P0.2 MMQ-Port Schritt 2 Debug — Progress + Remaining Bug

**Date:** 2026-04-24
**Branch:** v1.0-dev @ commit before this
**Status:** **Zwei Bugs gefunden und gefixt, dritter Bug nicht gefunden.** Kernel produziert jetzt sinnvolle FP32-Werte (statt vorher 10⁹ oder NaN), aber die Parity mit CPU fehlt immer noch (max_rel ~1.99 mit plausiblen Magnituden).

## Gefundene und gefixte Bugs

### Bug 1: `block_q8_1_mmq` hat nur 4 Sub-Blöcke, nicht 8 (GEFIXT)

Gefunden **durch Code-Inspektion** (nicht Test-Run). Q4_K hat 8 Sub-Blöcke × 32 Elemente = 256 K pro Super-Block. `block_q8_1_mmq` hat **128** Elemente (4 Sub-Blöcke × 32). Mein Kernel las `B[row16].ds4[sb]` für `sb=0..7` und `B[row16].qs[32*sb..]` für sb bis 7 — aber das Struct hat nur `ds4[4]` und `qs[128]`. Reads für sb=4..7 gingen ins Nirvana (also in das folgende Struct oder Garbage).

**Fix:** Pro M-Row ZWEI `block_q8_1_mmq` Blöcke:
- Q4_K sb 0..3 → `B[2*m_row].ds4[sb]`, `B[2*m_row].qs[sb*32..]`
- Q4_K sb 4..7 → `B[2*m_row+1].ds4[sb-4]`, `B[2*m_row+1].qs[(sb-4)*32..]`

### Bug 2: K-Offset-Formel für Fragment-Loads (GEFIXT)

Per `mma.cuh:198-230` auf RDNA4 für `tile<16, 8, int, DATA_LAYOUT_I_MAJOR>`:
- `get_i(l) = threadIdx.x % 16`
- `get_j(l) = 4 * (threadIdx.x / 16) + l` (in int32-units, l = 0..3 pro Call)

Das heißt in int8-K-Positionen:
- Lane 0..15, Call 0: K = 0..7
- Lane 0..15, Call 1: K = 8..15
- Lane 16..31, Call 0: K = 16..23
- Lane 16..31, Call 1: K = 24..31

Meine alte Formel `k_in_sb = call*16 + half*8 + w` produzierte die SWAPPED Variante:
- Lane 0..15, Call 0: K = 0..7 ✓
- Lane 0..15, Call 1: K = 16..23 ✗ (sollte 8..15 sein)
- Lane 16..31, Call 0: K = 8..15 ✗ (sollte 16..23 sein)
- Lane 16..31, Call 1: K = 24..31 ✓

**Fix:** `k_in_sb = 16*half + 8*call + w`.

## Testresultat nach Fix 1+2

```
MMQ-Q4_K minimal @ 16×16×256:
  max_abs = 2.6e3   (vorher: 6.1e9)
  max_rel = 1.99    (vorher: 1.16)
  max_mag = 3.1e3   (vorher: 6.1e9)

Erste GPU/CPU Paare:
  (1020.59, 773.92)    — 1.32× zu hoch
  (-360.05, 348.94)    — SIGN FLIP
  (-528.26, -290.57)   — Sign ok, Magnitude 1.8×
  (-99.05, -372.94)    — Sign ok, aber sehr unterschiedlich
  (-774.47, -1499.18)  — Sign ok, Magnitude 0.52×
  (479.18, 72.47)      — Sign ok, Magnitude 6.6×
```

**Magnituden sind jetzt im richtigen Bereich (FP32 plausibel), keine NaN mehr.** Aber einzelne Zellen sind 1.3× bis 6.6× falsch, und eine zeigt Sign-Flip. Muster: manche Zellen nah-korrekt, manche weit daneben.

## Verbleibender Bug (nicht gelöst diese Session)

Die Pattern "manche nah, manche weit" passt zu einer von zwei Hypothesen:

### Hypothese A: C-Fragment Transposition auf RDNA4

Aus `mma.cuh:218`: *"matrix C is the transposed matrix A&B on RDNA4"*. Der Kommentar in den Tile-Definitions legt nahe, dass C's Lane→Output-Mapping transponiert ist vs A&B. Mein Write-back nutzt die Konvention aus dem Integer-WMMA-Smoke-Test (`wmma_i32_smoke.hip`):
```cpp
const int col_M = lane & 15;
const int row_N_start = (lane >> 4) * 8;
C[(row_N_start + v) * 16 + col_M] = acc[v];
```

Der Smoke-Test hat mit dieser Konvention bit-exakt vs CPU-Referenz gepasst. ABER: Der Smoke-Test berechnete `A × B` mit spezifisch gewählten Input-Layouts (A row-major in memory, B col-major in memory). Der MMQ-Kernel hat B zwar gleich logisch, aber die Interaktion zwischen "Intrinsic-C-Layout" und "Write-back-Interpretation" ist vielleicht nicht identisch.

**Debug-Test nächste Session:** Direkt vergleichen ob `out[n,m]` oder `out[m,n]` die CPU-Referenz matched. Falls Swap reicht: Write-back ist die Ursache.

### Hypothese B: B-Fragment Load-Pattern vs Smoke-Test-Konvention

Der Smoke-Test hatte `B` physisch col-major im Speicher (pro Zeile/Spalte-transponierte Kopie von B_rm). Die Load-Pattern `B[(half*8+r) + row_in_half*16]` liest aus COL-MAJOR.

Der MMQ-Kernel hat `B[m].qs[k]` in ROW-MAJOR Speicher ([M × K]). Der LOGISCHE Zugriff mapped wahrscheinlich identisch, aber die PHYSIKALISCHE Lane-Verteilung kann abweichen falls das Intrinsic nicht rein symbolisch mit den Lane-Daten arbeitet sondern mit bestimmten Memory-Patterns.

**Debug-Test nächste Session:** Kernel-Variante, die B "col-major" lädt (also via expliziter Transposition beim Load aus row-major Speicher), und vergleichen.

## Gelandete Änderungen (diese Session)

`hip_kernels_v1/wmma/mmq_q4_k_minimal.hip`:
- B-Block-Indexing: `&B[2 * row16 + (sb >> 2)]`, lokaler sb = `sb & 3`
- ds4 und qs accesses mit lokalem sb statt globalem
- K-offset formel: `16*half + 8*call + w`

## Nächste-Session Debug-Plan (konkret)

**Stufe 2 Minimal-Test (30 Min):** Kernel-Variante mit hardcoded `d_A=1, dmin=0, d_B=1, sum_B=0`. Output sollte `Σ nibble × int8` sein, CPU-seitig als integer GEMM verifizierbar. Wenn das NICHT passt → Fragment-Layout (Hypothese A oder B). Wenn passt → Scale-Math-Bug (unwahrscheinlich aber möglich).

**Stufe 3 Transpose-Test (15 Min):** Tausch `C[n*16+m]` ↔ `C[m*16+n]`. Falls plötzlich alle Zellen passen → Hypothese A bestätigt.

**Stufe 4 Fragment-Dump (1 h):** Debug-Kernel der pro Lane die A/B-Fragmente + C-Outputs in separate Buffer schreibt. Auf CPU gegen erwartete Werte pro Lane vergleichen. Zeigt exakt wo der Bug liegt.

## End-to-End Status

| Metrik | Vor Fix 1+2 | Nach Fix 1+2 | Ziel |
|---|---:|---:|---:|
| max_abs | 6.1×10⁹ | 2.6×10³ | < 150 |
| max_rel | 1.16 | 1.99 | < 0.05 |
| NaN cells | ja | nein | nein |
| FP32-Plausibel | nein | ja | ja |

**Fortschritt ist real** — die größten Bugs (Struct-Layout + K-Offset) sind gefixt und die Magnituden sind jetzt im richtigen Bereich. Der verbleibende ~2× Fehler in einzelnen Zellen ist ein kleinerer Bug (wahrscheinlich Write-back-Transposition), aber noch nicht lokalisiert.
