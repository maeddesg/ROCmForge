# Memory-Access-Pattern-Analyse für transponierte `ffn_down`-Gewichte

**Phase 4 Step 2.1 — rein rechnerisch.**
**Date:** 2026-04-18
**Hardware:** AMD Radeon RX 9070 XT (gfx1201, RDNA 4), ROCm 7.2.2
**Model:** Qwen2.5-7B-Instruct Q4_0, 28 Layer

rocprofv3 mit PMC-Countern hängt auf gfx1201 (bekanntes Problem, per
Hinweis übersprungen). Die Entscheidung wird auf Basis der rechnerischen
Layout-Analyse getroffen — das reicht, weil der entscheidende Datenpunkt
die byte-weise Indexformel im Kernel ist, nicht ein Hardware-Counter.

---

## TL;DR

Die WMMA-Q4_0-Kernel-Indexformel und die physische GGUF-Speicherung von
`ffn_down` sind **byte-identisch**. Der `needs_transpose = true`-Flag ist
eine semantische Etikette auf CPU-Seite (siehe `src/cpu/ops.rs` —
`gemm_q4_0` und `gemm_q4_0_transposed_gemm` adressieren die Weight-Bytes
mit identischer Formel), kein echter Layout-Unterschied.

Konsequenz:

- **Keine der drei vorgeschlagenen Optionen (A Stride-Read / B Repack /
  B.2 Permutation / C LDS-Staging) ist nötig.**
- Die korrekte Handlung ist: den `!meta.needs_transpose`-Guard aus dem
  WMMA-Pfad in `gpu_dispatch_gemm` entfernen (Einzeiler), Korrektheit auf
  einem Q4_0-`ffn_down`-Layer gegen den aktuellen hipBLAS-Pfad prüfen,
  fertig.
- Erwarteter Effekt: 25 der 28 `ffn_down`-Layer (die Q4_0-quantisierten)
  laufen von ~5.0 µs (hipBLAS-Dequant+Hgemm) auf ~1.3 µs (isolierter
  WMMA-Bench, Phase 2b-Referenz) — entspricht ~92 ms Ersparnis pro
  pp256-Prefill, Gesamt-Prefill 300 → ~208 ms, pp256-Throughput ~1240
  tok/s.
- Die 3 verbleibenden Q4_1-`ffn_down`-Layer (siehe unten) bleiben bis auf
  Weiteres auf dem `gemm_q4_1_f32`-Fallback und sind für Phase 5+
  vorgemerkt.

---

## 1. Physisches Speicherlayout — dokumentiert

### 1.1 GGUF-Dim-Konvention

`TensorDesc::dims` speichert "Dimensionen in GGUF-Reihenfolge (innermost
first)" (`src/loader/parse.rs:266`). `dims[0]` ist also die **schnelle
Achse** (niedrigster Stride), `dims[n-1]` die langsamste.

### 1.2 Tatsächliche Dims aus dem Modellfile

`./target/release/rocmforge --list-tensors` auf
`~/models/Qwen2.5-7B-Instruct-Q4_0.gguf`:

| Tensor | Typ | dims | dim[0] (fast) | dim[1] (slow) |
|---|---|---|---|---|
| `ffn_gate.weight` / `ffn_up.weight` | Q4_0 | `[3584, 18944]` | 3584 = h = K | 18944 = ff = N |
| `ffn_down.weight` | **Q4_0 × 25 Layer / Q4_1 × 3 Layer** | `[18944, 3584]` | 18944 = ff = **K** | 3584 = h = **N** |

Wichtige Entdeckung: **`ffn_down` ist in 25 der 28 Layer bereits Q4_0**.
Nur 3 Layer sind Q4_1 (llama-quantize behält für einzelne Tensoren eine
höhere Präzision). Das erklärt auch, warum der Phase-4-Step-2-hipBLAS-Fix
bereits funktioniert: 25 der 28 `ffn_down`-Layer fallen seitdem in den
hipBLAS-Pfad (25× "hipBLAS Hgemm fallback" mit `weight_transposed=true`
im Debug-Log); die 3 Q4_1-Layer bleiben auf `gemm_q4_1_f32`.

### 1.3 Physischer Speicherlayout in Bytes

**`ffn_gate` (Q4_0, `[3584, 18944]`, `needs_transpose = false`):**

- Slow-Achse 18944 = ff = N (out_dim), 18944 Slow-Gruppen
- Fast-Achse 3584 = h = K (in_dim), je 3584/32 = **112 Q4_0-Blöcke pro
  Slow-Gruppe**
- Bytes pro Slow-Gruppe: 112 × 18 = **2.016 B**
- Gesamtgröße: 18944 × 2016 = 38.18 MB
- Adresse Block(n_col, kc): `(n_col × 112 + kc) × 18`

**`ffn_down` (Q4_0, `[18944, 3584]`, `needs_transpose = true`):**

- Slow-Achse 3584 = h = N (out_dim), 3584 Slow-Gruppen
- Fast-Achse 18944 = ff = K (in_dim), je 18944/32 = **592 Q4_0-Blöcke pro
  Slow-Gruppe**
- Bytes pro Slow-Gruppe: 592 × 18 = **10.656 B**
- Gesamtgröße: 3584 × 10656 = 38.18 MB
- Adresse Block(n_col, kc): `(n_col × 592 + kc) × 18`

**Einzige Zahl, die sich unterscheidet: `blocks_per_row` (112 vs. 592) —
weil K für gate 3584 und für down 18944 ist. Die Indexformel
`(n_col * blocks_per_row + kc) * 18` ist in beiden Fällen identisch.**

### 1.4 Schnelle vs. langsame Achse — Zusammenfassung

Für **beide** Tensoren ist die **fast axis die Kontraktionsachse K**
(`in_dim`), und jeder Q4_0-Block packt 32 aufeinanderfolgende K-Werte
eines festen N-Indexes. Das ist die native Form, die der WMMA-Q4_0-Kernel
erwartet.

---

## 2. Access-Pattern-Analyse

### 2.1 Kernel-Indexformel (ground truth)

`hip_kernels/wmma/wmma_gemm_q4_0.hip`, Zeile 93–95:

```c
const size_t block_ofs =
    (size_t)(tile_n + col) * (size_t)blocks_per_row * Q4_0_BLOCK_BYTES +
    (size_t)kc * Q4_0_BLOCK_BYTES;
```

Mit `blocks_per_row = K / 32`. Der Kernel liest Block an
`(n_col * blocks_per_row + kc) * 18` — **identisch zur physischen
Adressformel der GGUF-Storage für beide** `ffn_gate` und `ffn_down`.

### 2.2 Cacheline-Effizienz (rechnerisch)

Ein K-Chunk-Load im WMMA-Kernel besteht aus 64 Q4_0-Blöcken (64 N-Spalten
im Tile). Für eine Wavefront wird der Block bei einem festen `kc`
abgerufen; die 64 Spalten-Blöcke haben die Adresse

```
ofs(n_col) = (n_col * blocks_per_row + kc) * 18
           = kc * 18 + n_col * (blocks_per_row * 18)
```

Der Stride zwischen aufeinanderfolgenden N-Spalten-Blöcken ist
**`blocks_per_row * 18` Bytes**:

| Tensor | blocks_per_row | Stride N→N+1 | Cacheline-Count für 64 Blöcke |
|---|---:|---:|---:|
| `ffn_gate` (Q4_0) | 112 | **2.016 B** | 64 × 128 / 2.016 = ~4 useful B/CL → **64 separate Cachelines** |
| `ffn_down` (Q4_0) | 592 | **10.656 B** | 64 × 128 / 10.656 = <1 useful B/CL → **64 separate Cachelines** |

Mit Cacheline-Größe 128 B auf gfx1201: jeder der 64 N-Block-Reads
triggert eine eigene Cacheline — bei **beiden** Tensoren. Der WMMA-Kernel
ist also sowieso "inkohärent" über die N-Achse; das gilt für gate genauso
wie für down. Der Stride ist bei down 5.3× größer (592/112), aber da der
Zugriff ohnehin bereits nicht coalesced ist, bedeutet das **keine
zusätzliche Strafe im selben Sinne** — die Anzahl Cachelines ist in
beiden Fällen 64.

Was sich zwischen gate und down tatsächlich unterscheidet: bei down muss
der L2 pro Layer 38 MB fetchen (gleich wie gate), **aber die L2→L1-Prefetch-
Reuse ist bei down geringer**, weil die 64 Cachelines eines K-Chunks pro
Tile über einen weiteren Adressbereich verstreut sind (64 × 10.656 B =
681 KB) als bei gate (64 × 2.016 B = 129 KB). Beide liegen deutlich über
dem L1-Cache (typisch 16–32 KB auf RDNA4 pro WGP), sodass
**Reuse ohnehin aus L2 kommt**. L2 auf der RX 9070 XT ist 4 MB —
ausreichend für alle Tile-Loads beider Kernel.

### 2.3 Nützliche Bandwidth

- Nutzdaten pro K-Chunk: 64 Blöcke × 18 B = 1.152 B
- L1-Fetch pro K-Chunk: 64 CL × 128 B = 8.192 B
- **Nutz-Effizienz: 14 %** (identisch für gate und down)

Das ist die strukturelle Grenze des Kernel-Layouts. Gate läuft damit im
isolierten Bench in ~1.2 µs für pp256 — die 86 % "verschwendete"
Cacheline-Bandbreite werden durch den niedrigen L2-Latency-Faktor und
den kleinen Working-Set (38 MB gesamt, L2 absorbiert wesentliche Teile)
ausgeglichen. Es gibt keinen Grund anzunehmen, dass das für down anders
laufen würde — der Stride ist größer, aber der Gesamtdatenumfang ist
identisch.

---

## 3. rocprofv3 — nicht verfügbar

rocprofv3 mit `--pmc TCP_REQ,TCP_REQ_MISS,FETCH_SIZE,L2CacheHit` hängt
auf gfx1201 (Sample lief >2 Minuten ohne Fortschritt, Prozess mit `kill`
beendet). Das ist ein bekanntes Problem bei gfx1201 — der Hardware-Support
für diese Counter ist in rocprofv3 noch nicht stabil.

Die rechnerische Analyse (Abschnitt 2) ersetzt die Messung: der Stride
und die Cacheline-Anzahl sind statische Properties des Kernel-Layouts und
der GGUF-Speicherung — sie lassen sich aus dem Code exakt ableiten.

---

## 4. Optionsbewertung

Ausgangsthese des Prompts war, dass `ffn_down` ein **anderes** physisches
Layout hat als `ffn_gate` und daher einen Kernel mit Stride-Read
(A) / Repack (B) / LDS-Staging (C) braucht.

**Die Analyse widerlegt die These.** Beide Layouts haben die gleiche
strukturelle Form `[N slow, K fast, blocks packing 32 K-Werte]`, nur mit
unterschiedlichen `blocks_per_row`-Werten.

| Option | Notwendig? | Begründung |
|---|---|---|
| **A** (Stride-Read im Kernel) | **nein** | Der Kernel macht bereits einen Stride-Read. Der einzige Unterschied zwischen gate und down ist der Stride-Wert (2016 B vs 10656 B), der als Parameter aus `K` abgeleitet wird. |
| **B** (Repack beim Model-Load, extra 950 MB VRAM) | **nein** | Kein Repack nötig; das Layout ist bereits kernel-kompatibel. |
| **B.2** (In-Place-Permutation) | **nein** | Wie B. |
| **C** (LDS-Transpose-Staging) | **nein** | Kein Transpose nötig. |

### Der tatsächliche Fix

```rust
// src/gpu/ops.rs, Zeile ~1468 — den needs_transpose-Guard entfernen:
if seq_len >= WMMA_PREFILL_MIN_M
    && meta.wtype == GgmlType::Q4_0
-    && !meta.needs_transpose          // ← diese Zeile streichen
    && (out_dim % 64) == 0
    && (in_dim % 32) == 0
    && super::safety::wmma_prefill_enabled()
```

Validierung: ein bestehender Correctness-Test auf `ffn_down` Layer 0
(Q4_0) gegen den hipBLAS-Pfad. Erwartung: byte-identische Output-Tensoren
(beide FP16-Akkumulation bei WMMA bzw. FP16-Hgemm; Numerik auf Greedy-
Argmax gleich wie schon in Schritt 2 verifiziert).

**Warum ist das sicher?** Weil die CPU-Referenz bereits zeigt, dass
`gemm_q4_0` (non-transposed) und `gemm_q4_0_transposed_gemm` die
gleichen Block-Bytes mit der gleichen Formel
`(o * num_blocks_per_col + b) * 18` adressieren. Diese zwei CPU-Kernel
unterscheiden sich nur darin, wie sie **semantisch** `out_dim` und
`in_dim` aus den dims ablesen, nicht darin, wie sie die Bytes anfassen
— was gleichzeitig bewist, dass der WMMA-Kernel für beide semantischen
Fälle die richtigen Bytes liest, sofern `out_dim` und `in_dim` aus dem
Call-Site-Aufruf korrekt gesetzt sind (was sie sind — `forward.rs`
übergibt für `ffn_down` `out_dim=h, in_dim=ff` unabhängig vom
`needs_transpose`-Flag).

---

## 5. Erwarteter Gewinn

| Layer-Klasse | Vor Fix Phase 2 | Nach Fix Phase 2 (jetzt) | Nach Fix Phase 2.1 (Plan) |
|---|---:|---:|---:|
| 25 × `ffn_down` Q4_0 | ~11 µs (scalar) | ~5 µs (hipBLAS) | **~1.3 µs (WMMA)** |
| 3 × `ffn_down` Q4_1 | ~11 µs (scalar) | ~11 µs (scalar) | ~11 µs (scalar, bleibt) |

Summe pro Prefill (pp256):
- Phase 2 aktuell: 25 × 5 + 3 × 11 = 158 ms (gemessen 163.9 ms — stimmt)
- Phase 2.1 Plan: 25 × 1.3 + 3 × 11 = 65.5 ms → **Einsparung ~98 ms**
- Prefill-Gesamt pp256: 300 → **~202 ms → ~1270 tok/s**

### Optional später (Phase 5+)

Für die 3 Q4_1-Layer: eine Q4_1-WMMA-Kernelvariante wäre
straightforward (gleiche Struktur, 20 B statt 18 B pro Block, ein
Scale + ein Min statt nur Scale). Erwarteter Gewinn ~30 ms pro
pp256-Prefill. Kein Blocker für Phase 4.

---

## 6. Deliverables-Check

- [x] Physisches Layout dokumentiert (Abschnitt 1.3)
- [x] Schnelle vs. langsame Achse identifiziert (Abschnitt 1.2 / 1.4)
- [x] Stride zwischen K-Blöcken berechnet (2016 / 10656 B, Abschnitt 2.2)
- [x] Cacheline-Effizienz berechnet (14 % nützlich, identisch für gate
  und down, Abschnitt 2.3)
- [x] rocprofv3: nicht verfügbar auf gfx1201, übersprungen (Abschnitt 3)
- [x] Optionsbewertung (Abschnitt 4) — Empfehlung: keine der Optionen
  A/B/B.2/C, stattdessen `!needs_transpose`-Guard im WMMA-Pfad streichen

## Kurz-Report

- **Physisches Layout:** `ffn_down [18944, 3584]` Q4_0 — schnelle Achse
  dim[0]=18944=ff=**K** (Kontraktionsachse). Identische strukturelle Form
  wie `ffn_gate [3584, 18944]`, nur mit `blocks_per_row = 592` statt 112.
- **Stride K-Block → K+1:** 18 B (contiguous innerhalb einer N-Row).
- **Stride N-Col → N+1 bei festem kc:** 10.656 B für down vs. 2.016 B
  für gate — aber beide bereits nicht-coalesced (jeweils 64 Cachelines
  pro Tile).
- **Cacheline-Effizienz:** 14 % Nutzdaten / geladene Daten — **gleich
  für gate und down**. Der WMMA-Kernel ist für beide Layouts
  strukturell gleich (in)effizient.
- **rocprofv3:** auf gfx1201 hängt PMC-Counter-Capture. Übersprungen —
  die rechnerische Analyse reicht.
- **Empfehlung: keine der drei Optionen.** Das `needs_transpose`-Flag
  ist eine CPU-seitige semantische Etikette, kein echter Layout-
  Unterschied. Der WMMA-Kernel kann `ffn_down` direkt lesen. Fix =
  `!meta.needs_transpose`-Guard in `gpu_dispatch_gemm` streichen
  (Einzeiler) + Correctness-Test. **Erwarteter Gewinn: ~98 ms pro
  pp256-Prefill (300 → 202 ms, ~1270 tok/s).**
