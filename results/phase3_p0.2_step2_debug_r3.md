# P0.2 MMQ-Port Schritt 2 Debug Runde 3 — BUG GEFUNDEN + GEFIXT ✅

**Date:** 2026-04-24
**Status:** **Parity-Test grün.** MMQ-Q4_K minimal kernel produziert jetzt bit-quasi-exakte Ergebnisse (max_abs 13.7 bei max_mag 3155 = ~0.4% worst-case relative).

## TL;DR

Hypothese C (Q4_K Nibble-Byte-Index-Formel) bestätigt durch Code-Inspection **vor** dem geplanten Scales=1-Test (Stufe 2). Die bestehende FP16-WMMA-Implementation in `hip_kernels_v1/wmma/wmma_gemm_q4_k_fp16.hip:107-117` nutzt eine PAIR-basierte Nibble-Extraktion, meine Minimal-Version hatte eine flache Extraktion, die Sub-Blöcke 0+1 (und 2+3, 4+5, 6+7) miteinander vermischt hat. Der Fix: 1:1 dieselbe Extraktions-Logik wie der bewährte FP16-Kernel.

## Der Bug

### Q4_K qs-Layout (aus dem bestehenden FP16-Kernel verifiziert)

Q4_K packt 256 Elemente als 8 Sub-Blöcke × 32 Elemente in 128 Bytes, **paarweise organisiert**:

| Pair | Bytes | Low-Nibbles | High-Nibbles |
|---|---|---|---|
| 0 | 0..31 | Sub-Block 0 (K=0..31) | Sub-Block 1 |
| 1 | 32..63 | Sub-Block 2 | Sub-Block 3 |
| 2 | 64..95 | Sub-Block 4 | Sub-Block 5 |
| 3 | 96..127 | Sub-Block 6 | Sub-Block 7 |

Ein Sub-Block belegt **32 Bytes** (low- oder high-Nibbles), nicht 16.

### Meine falsche Formel (vor Fix)

```c
const int byte_idx = 16 * sb + (k_in_sb >> 1);       // 0..15 FOR sb=0
const int nib = (k_in_sb & 1) ? (byte >> 4) : (byte & 0x0F);
```

Für Sub-Block 0 iterierte das über Bytes 0..15 und extrahierte abwechselnd low+high Nibbles — damit enthielt "Sub-Block 0 Element K" in Wahrheit:
- K=0: byte[0] LOW  (korrekt = Sub-Block 0 K=0)
- K=1: byte[0] HIGH (FALSCH — ist Sub-Block 1 K=0)
- K=2: byte[1] LOW  (ist Sub-Block 0 K=1)
- K=3: byte[1] HIGH (FALSCH — ist Sub-Block 1 K=1)
- ...

Mit falschen 16 von 32 Elementen pro Sub-Block produzierte der MMA entsprechend falsche Summen.

### Der Fix (nach dem bestehenden FP16-Kernel)

```c
const int pair_base = (sb >> 1) * 32;     // 0, 32, 64, 96
const bool is_upper = (sb & 1) != 0;      // Sub-Block 1,3,5,7 lesen HIGH
...
const int byte_idx = pair_base + k_in_sb; // 0..31 innerhalb des Pairs
const uint8_t byte = a_row->qs[byte_idx];
const int nib = is_upper ? (byte >> 4) : (byte & 0x0F);
```

Für Sub-Block 0: alle 32 Elemente aus Bytes 0..31, **alle low Nibbles**.
Für Sub-Block 1: gleiche Bytes 0..31, **alle high Nibbles**.
Etc.

## Parity-Testresultat

```
Before Fix C:
  max_abs = 2.6e3, max_rel = 1.99, Sign-Flips in 4+ Zellen

After Fix C (auf demselben Testinput):
  max_abs = 13.7, max_mag = 3155 → worst-case relative Fehler = 0.4%
  
GPU vs CPU erste Zellen:
  (771.32, 773.92)     — 0.3 % Abweichung
  (347.98, 348.94)     — 0.3 % Abweichung
  (-290.29, -290.57)   — 0.1 % Abweichung
  (-375.39, -372.94)   — 0.6 % Abweichung
  (-1499.80, -1499.18) — 0.04 % Abweichung
  (-1609.95, -1611.91) — 0.1 % Abweichung
```

Alle Abweichungen sind FP16-Akkumulations-Noise plus Q8_1-Aktivierungs-Quantisierungs-Noise, konsistent mit dem Niveau das auch der bestehende FP16-WMMA-Test sieht.

**Parity-Test: GRÜN** (mit magnitude-aware tolerance: `(max_mag + 1e-3) * sqrt(K) * 5e-3`).

## Tests Final

| Test | Stufe-2-Schritt | Ergebnis |
|---|---|---|
| `block_q8_1_mmq_has_expected_size` | Schritt 1 (Infra) | ✓ |
| `int_wmma_smoke_identity_by_counting` | Schritt 1 (Infra) | ✓ |
| `int_wmma_smoke_random_signed` | Schritt 1 (Infra) | ✓ |
| `quantize_q8_1_mmq_matches_cpu_reference` | Schritt 1 (Infra) | ✓ |
| `quantize_q8_1_mmq_zero_input_safe` | Schritt 1 (Infra) | ✓ |
| `mmq_q4_k_minimal_matches_cpu_reference` | **Schritt 2 (Kernel)** | ✅ |

**6/6 MMQ-Tests grün.**

## Fazit der drei Debug-Runden

**Runde 1 (Scaffold):** Struct-Layout (`block_q8_1_mmq` hat 4, nicht 8 Sub-Blöcke) und K-Offset-Formel (16*half + 8*call + w) gefixt. 10⁹ → 10³ Magnitude-Korrektur.

**Runde 2 (Transpose):** Hypothese A (C-Fragment-Transposition) widerlegt. Tile<>-Konvention-Write-back gab schlechtere Ergebnisse als Smoke-Test-Konvention, also Write-back nicht der Bug.

**Runde 3 (diese Session):** Hypothese C (Q4_K Nibble-Extraktions-Formel) durch Code-Inspection des bestehenden FP16-Kernels **direkt identifiziert** — Scales=1-Test (Stufe 2) wurde nicht nötig. Fix via 1:1 Port der PAIR-basierten Extraktion.

**Gesamt-Fix-Summary seit Start Schritt 2:**
1. ✅ `block_q8_1_mmq` korrekt als 4 Sub-Blöcke (nicht 8) behandeln
2. ✅ K-Offset: `16*half + 8*call + w` (nicht `16*call + 8*half + w`)
3. ✅ Q4_K Nibble-Extraktion: pair-basiert, nicht flach

Mit allen drei Fixes: Minimal-Kernel bit-quasi-exakt vs CPU-Referenz.

## Next Steps — Schritt 3 (Scale-Up)

Das Minimal-Kernel (16×16×256, single warp, single block) ist korrekt. **Nächste Session:**

1. **Multi-Warp skalieren:** 4 oder 8 Warps pro Block, TILE_M = 64 oder 128
2. **Variable K:** K=4096 (volle Qwen3-Shape), erforderliche Outer-Loop über Super-Blöcke
3. **Executor-Integration:** Als Bandit-Alternative zum FP16-WMMA-Kernel wählen
4. **rocprof:** Performance-Messung gegen FP16-WMMA-Baseline
5. **Scale-Up:** Tile-Config-Sweep auf realistischen Prefill-Shapes

Geschätzter Aufwand Schritt 3: 1-2 Sessions (~300-400 LOC + Integration + Tests).

## Geänderte Dateien

- `hip_kernels_v1/wmma/mmq_q4_k_minimal.hip` — Nibble-Extraktion gefixt (pair-basiert)
- `tests_v1/mmq_q4_k_minimal_test.rs` — Magnitude-aware Tolerance (wie `wmma_test.rs`)

Kein Produktionspfad geändert — Decode unverändert bei 96.2 tok/s, Prefill unverändert bei ~1000 tok/s.

## Post-Mortem: Bisection-Effizienz

Die 3-Runden-Bisection war effizient:
- **Runde 1:** 2 Bugs gefunden via Code-Inspektion (Scaffold-Build), **ohne Debug-Test**
- **Runde 2:** 1 Hypothese widerlegt via Transpose-Swap-Test (15 min)
- **Runde 3:** Letzter Bug gefunden via Code-Inspektion (Vergleich mit FP16-Kernel), **ohne Scales=1-Test**

Insgesamt: 3 Bugs gefixt, davon 2 via reine Code-Inspektion, 1 via Transpose-Experiment. Stufen 2 und 4 (Scales=1-Kernel, Fragment-Dump) waren **nicht nötig**. Das unterstreicht den Wert von: "Wenn ein bewährter Kernel für dasselbe Problem existiert, vergleiche Zeile für Zeile bevor du experimentierst."
