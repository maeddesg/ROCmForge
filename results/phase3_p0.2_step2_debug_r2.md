# P0.2 MMQ-Port Schritt 2 Debug Runde 2 — Transpose-Swap-Test

**Date:** 2026-04-24
**Status:** Stufe 3 (Transpose-Swap) **durchgeführt, widerlegt Hypothese A**. Stufe 2 + 4 nicht ausgeführt (Session-Budget).

## TL;DR

Die Vermutung aus Runde 1 war: **C-Fragment-Transposition** ist der verbleibende Bug. Stufe 3 testet das direkt durch Swap der Write-back-Interpretation (slot=N-row → slot=M-col, und umgekehrt). **Ergebnis: WORSE, nicht besser.** Mit der „tile<>-konventionellen" Interpretation kollabieren alle Output-Werte der ersten Zeile auf ähnliche Magnituden (alle ~1000), während mit der Smoke-Test-Konvention die Werte zumindest variieren wie in der CPU-Referenz. Der Bug liegt also **nicht im Write-back**, sondern weiterhin im A/B-Fragment-Loading oder in der Scale-Extraktion.

## Stufe 3 Ergebnis

### Variante A: slot = M-col Interpretation (tile<16,16,int> direkt)

```cpp
const int n_row      = lane & 15;         // lane's N
const int m_col_base = (lane >> 4) * 8;   // lane's M range
for (int slot = 0; slot < 8; ++slot) {
    const int m = m_col_base + slot;
    // read B-scales per m (= per slot)
    // C[n_row * 16 + m] = out
}
```

Output (erste Zeile, 8 GPU/CPU Paare):
```
(1020.59, 773.92)     ← passt grob
(931.50, 348.94)      ← GPU viel größer, CPU halb so groß
(932.21, -290.57)     ← Sign-Flip
(1278.46, -372.94)    ← Sign-Flip
(991.36, -1499.18)    ← Sign-Flip + GPU-Magnitude viel kleiner
(782.66, -1611.91)    ← Sign-Flip + GPU-Magnitude ~halb
(1441.91, 72.47)      ← passt grob
(1015.22, 206.76)     ← grob
```

**Alle GPU-Werte sind ähnlich (zwischen 782 und 1441)** während CPU-Werte stark variieren (−1611 bis +773). Das kann nur passieren, wenn die `acc_sb[sb][slot]` Werte für verschiedene Slots alle sehr ähnlich sind — was bedeutet, dass Slot die eigentliche Varianz NICHT mit M-col korrespondiert. max_abs = 1.52×10⁹ (schlechter als Smoke-Konvention).

### Variante: Smoke-Test-Konvention (slot = N-row)

Output erste 8 (Referenz-Baseline):
```
(1020.59, 773.92)   ← 1.32×
(-360.05, 348.94)   ← Sign-Flip
(-528.26, -290.57)  ← 1.8×
(-99.05, -372.94)   ← 0.27×
(-774.47, -1499.18) ← 0.52×
(-863.35, -1611.91) ← 0.54×
(479.18, 72.47)     ← 6.6×
(165.69, 206.76)    ← 0.80×
```

max_abs = 2.6×10³. Die Werte VARIIEREN zumindest wie in CPU-Referenz (Mix aus positiv/negativ, unterschiedliche Magnituden), nur an den FALSCHEN Positionen oder mit falschen SCALES.

### Schlussfolgerung Stufe 3

**Hypothese A (C-Transposition) falsch.** Die Smoke-Test-Konvention produziert eine strukturierte Ausgabe mit den richtigen pro-Zelle-Varianz-Mustern (auch wenn Werte um 1-7× daneben), während die Tile-Convention alle Werte ähnlich macht. Der Write-back ist damit NICHT der Bug.

→ Bug muss in A/B-Fragment-Loading oder Scale-Extraktion liegen. Kernel ist auf Smoke-Test-Write-back zurückgesetzt.

## Verbleibende Hypothesen (nicht diese Session)

### Hypothese B' (wahrscheinlichst): Sub-Block-Index <=> Accumulator-Slot Koppelung

Ich nehme an, dass `acc_sb[sb][slot]` die int-MMA-Summe für (N-row = n_row_start+slot, M=m_col) über Sub-Block sb ist. Aber wenn die int-WMMA-Lane-Verteilung die 8 K-Iterations pro Sub-Block ANDERS zusammen packt als ich denke, könnte `slot` tatsächlich eine andere semantische Bedeutung haben (z. B. „K-Offset innerhalb des Sub-Blocks" oder „alternative N-Row-Zuordnung").

**Debug-Test:** Hardcode ALLE Scales auf 1, alle Mins auf 0. Dann sollte out = Σ_sb raw_acc[sb][slot] = int-GEMM-Ergebnis sein, direkt vergleichbar mit skalarer int-Referenz. Zeigt DEFINITIV ob acc_sb[sb][slot] für die richtige (N,M)-Zelle steht.

### Hypothese C: Sub-Block-Nibble-Extraktion

Q4_K's Nibble-Layout pro Super-Block ist komplex — 8 Sub-Blöcke × 32 Elemente mit bestimmter Byte-Verteilung. Meine Extraction `byte_idx = 16*sb + (k_in_sb >> 1)` nimmt an, dass jeder Sub-Block 16 aufeinanderfolgende Bytes hat, beginnend bei Offset 16*sb. Das ist korrekt für Q4_K's flat-Layout, aber llama.cpp's `load_tiles_q4_K` macht eine spezifischere Zuordnung (Zeile 2182-2183 in mmq.cuh: `16*(txi/8) + txi%8 + 0/8`). Falls unsere Nibble-Extraktion für einige K-Positionen FALSCHE Nibbles liefert, das MMA-Ergebnis ist entsprechend verschoben.

**Debug-Test:** Fragment-Dump auf Weights-Seite (Stufe 4 aus Runde-1-Plan): Pro Lane die 8 int8-Werte zeigen, die in a_frag_call[0] und a_frag_call[1] landen. CPU berechnet was da sein SOLLTE (via dequant_block aus Q4_K-Referenz). Diff zeigt falsche Positionen.

## Gelandete Änderungen (diese Session)

- `hip_kernels_v1/wmma/mmq_q4_k_minimal.hip`: Transpose-Swap eingebaut, verifiziert als WORSE, auf Smoke-Test-Konvention zurückgesetzt. Kommentar-Block mit der Diagnose für nächste Session hinzugefügt.

## End-to-End Status (unverändert seit Runde 1)

| Metrik | Runde 1 End | Runde 2 End |
|---|---:|---:|
| max_abs | 2.6×10³ | 2.6×10³ |
| max_rel | 1.99 | 1.99 |
| NaN cells | 0 | 0 |
| FP32-Plausibel | ja | ja |

## Nächste Session (konkret)

**Stufe 2 MUSS zuerst laufen:** Scales=1, Mins=0 Test. Das isoliert Fragment-Korrektheit von Scale-Math. Ein rein-int-GEMM-Parity-Test (Q4_K-Nibbles als int8 × Q8_1-int8, aufsummiert ohne Scales) entscheidet direkt:
- **PASS (max_rel < 5%)**: Fragment-Layout ist OK, Scale-Math hat Bug → fokussiere dort
- **FAIL**: Fragment-Layout hat Bug → Fragment-Dump (Stufe 4)

Erwarteter Aufwand Stufe 2: 45 min (neuer Kernel-Variant + Test).
Erwarteter Aufwand Stufe 4 falls nötig: 1-2 h (Fragment-Dump-Kernel + CPU-Referenz-Vergleich).

Der Bug ist **lokalisierbar** mit 1-3 gezielten Tests. Der tile<16,16,int>-Write-back-Transpose-Test hat eine Hypothese eliminiert und das Suchproblem damit verkleinert.
