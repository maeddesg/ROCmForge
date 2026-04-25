# ROCmForge v1.0 — Dequant IR Spezifikation

**Dokument:** `docs/v1.0/dequant_ir_spec.md`
**Version:** 1.0.0-final
**Datum:** 2026-04-20
**Status:** Final (abgenommen)
**Säule:** 3 von 6 (siehe `architecture_v1.2.0-draft.md` §2.4)

---

## Verhältnis zu architecture_v1.2.0-draft

Dieses Dokument ist die **detaillierte Implementierungs-Spezifikation** der Dequant IR.
Das Architektur-Dokument (`architecture_v1.2.0-draft.md` §2.4) ist der **kanonische
Blueprint** und hat bei Konflikten Vorrang. Diese Spec übernimmt die dortige
Nomenklatur vollständig und ergänzt sie nur dort, wo für die Codegen-Spezifikation
zusätzliche Ops benötigt werden. Jede Ergänzung ist explizit als **[SPEC-ERWEITERUNG]**
markiert und erfordert ein späteres Synchronisations-Update des Architektur-Dokuments.

---

## Inhaltsverzeichnis

1. Motivation und Design-Prinzipien
2. DequantOp-Spezifikation
3. QuantFormat-Definitionen (Q4_0, Q4_1, Q4_K, Q6_K, Q8_0)
4. Precision-Level-System (Level 0–3)
5. Generische Kernel-Specs (WMMA + GEMV)
6. Codegen GPU (gfx1201)
7. Codegen CPU (Zen4 AVX-512) + Rounding-Konsistenz
8. Schnittstellen zu anderen Säulen
9. Walk-Through: Q5_K vom leeren Blatt
10. Fehlerbehandlung und Edge-Cases

**Anhang A:** Design Decisions Log (STOP-Resolutions)
**Anhang B:** Sync-Liste für `architecture_v1.2.0-draft.md`

---

## 1. Motivation und Design-Prinzipien

### 1.1 Das v0.x Skalierungsproblem

Die bisher größte Zeitsenke in ROCmForge v0.x war das Hinzufügen neuer
Quant-Formate. Die quantitativen Befunde aus dem v0.x-Post-Mortem
(`architecture_v1.2.0-draft.md` §10.2):

| Aufwand-Dimension | Wert |
|---|---|
| Q6_K-WMMA-Implementierung (Phase 5, v0.3.0) | 5 Tage |
| Kernel-Dateien pro Format (WMMA + GEMV + fused-Varianten) | 5–7 |
| Bekannte Numerik-Bugs während der Q6_K-Entwicklung | 3 |
| Gesamtzahl Kernel-Dateien bei Phase-8b-Abschluss | ≈ 48 |
| Faktor: Formate × Operatoren × Fused-Varianten | 4 × 3 × 4 |

Ein hypothetisches Q5_K oder Q5_K_M hätte in v0.x **12 neue Kernel-Dateien**
erzwungen, je eine pro Kombination aus GEMM/GEMV und Fused-Variante. Jede
neue Fused-Variante (z. B. GEMV-mit-integriertem-RoPE) multipliziert sich mit
allen bestehenden Formaten. Der Aufwand ist **multiplikativ**, nicht additiv.

Zusätzlich schlug jede Numerik-Eigenheit eines Formats in allen zugehörigen
Kerneln durch. Der Q4_K-`/64`-Bug (v0.x Phase 7) existierte in vier
Kernel-Dateien gleichzeitig; die Korrektur erforderte vier separate
Patch-Pässe plus Verifikation.

### 1.2 Das v1.0-Ziel: O(1) pro neuem Format

Die Dequant IR wandelt den Aufwand von **multiplikativ in additiv**:

| Aktion | v0.x | v1.0 |
|---|---|---|
| Neues Format hinzufügen | 5 Kernel-Dateien, 3 Tage, ≈ 1 000 LOC | 1 `QuantFormat`-Struct, ~30 Min, ≈ 40 LOC |
| Neue Fused-Variante | 1 Kernel pro bestehendem Format | 1 Graph-Transformations-Pattern, alle Formate gratis |
| Numerik-Bug-Fix | N-fach in allen Format-Kerneln | 1× in der Codegen-Regel |
| Precision-Variante (FP16 ↔ FP8) | Neuer Kernel pro Kombination | 1 Enum-Flag im Codegen |

Das ist die **harte Design-Anforderung**: ein Entwickler, der ein neues
GGUF-Quant-Format unterstützen will, schreibt **keine einzige neue
Kernel-Datei**. Er deklariert das Format als DequantOp-Programm, der Codegen
erzeugt automatisch GEMM-, GEMV-, FP8-, FP16- und BF16-Varianten.

### 1.3 FP8-E4M3 als Default-WMMA-Input

Die zentrale Architektur-Entscheidung der v1.1-Revision (vgl.
`architecture_v1.2.0-draft.md` §2.4): der Dequant-Pfad produziert als Default
**FP8-E4M3** in LDS bzw. in den A/B-Fragmenten, nicht FP16. Diese Entscheidung
ist durch zwei Hardware-Tests verifiziert:

**FP8 WMMA Smoke Test** (5/5 PASS, 2026-04-20):

- `__builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12` kompiliert und läuft
  auf gfx1201. Korrektheit: `max_err = 0.0` bei Single-Tile-Evaluation.
- Roundtrip `FP32 → E4M3 → FP32`: `max_err = 0.125` (erwartbar für 3-Bit-Mantisse).
- Numerik: FP8-vs-FP32-Gap = `3.2e-2`, FP16-vs-FP32-Gap = `2.4e-4`.
- **Entscheidung:** GO — FP8-E4M3 ist bestätigter Default-WMMA-Input.

**FP8-Konvertierungs-Intrinsics** (6/6 PASS):

| Intrinsic | Funktion | Kosten |
|---|---|---|
| `v_cvt_pk_fp8_f32(a, b, old, word_hi)` | 2×FP32 → 2×E4M3 packed | 1 VALU-Instruktion |
| `v_cvt_f32_fp8(packed, byte_idx)` | E4M3 → FP32 | 1 VALU-Instruktion |
| `v_cvt_pk_bf8_f32(a, b, old, word_hi)` | 2×FP32 → 2×E5M2 packed | 1 VALU-Instruktion |
| `v_cvt_f32_bf8(packed, byte_idx)` | E5M2 → FP32 | 1 VALU-Instruktion |
| `v_cvt_sr_fp8_f32(...)` | Stochastic Rounding FP32 → E4M3 | 1 VALU-Instruktion |
| `v_cvt_sr_bf8_f32(...)` | Stochastic Rounding FP32 → E5M2 | 1 VALU-Instruktion |

Alle nativ auf gfx1201 vorhanden, kein Software-Fallback nötig. `DowncastToFP8`
emittiert direkt `v_cvt_pk_fp8_f32` und kostet **0.5 Instruktionen pro
FP32-Element** (ein Intrinsic verarbeitet 2 Elemente).

**Performance-Konsequenz:** FP8 halbiert den VGPR-Bedarf für A/B-Fragmente
gegenüber FP16 (8 Byte/Lane vs. 16 Byte/Lane) und erlaubt höhere Occupancy.
RDNA 4 unterstützt FP8-WMMA mit doppelter Rate gegenüber FP16-WMMA.

**AMD-Nomenklatur:** `fp8 = E4M3`, `bf8 = E5M2`. Das ist der
OCP-Standard-Namensgebung, nicht die FNUZ-Variante.

### 1.4 Warum IR statt Templates/Generics/Macros

Die Wahl einer echten Intermediate Representation (statt C++-Templates oder
Rust-Generics) folgt aus vier Anforderungen:

1. **Zwei Backends.** Dasselbe Dequant-Programm muss GPU-Code (HIP/RDNA 4
   Assembly) und CPU-Code (AVX-512) erzeugen. Templates können beides, aber
   der Codegen-Pfad ist bei beiden nicht trivial austauschbar.

2. **Vier Precision-Varianten.** Jedes Format braucht FP8-, FP16-, BF16- und
   FP32-Pfade. Template-Explosion wäre `Formate × 4`, und die
   Codepfade divergieren nicht uniform (FP8-Pfad nutzt andere WMMA-Intrinsics,
   anderen VGPR-Bedarf, andere LDS-Layouts).

3. **GA-getriebene Optimierung.** Die Kernel-Tuning-GA (rf-forge) muss zur
   Laufzeit Kernel-Varianten mit unterschiedlichen Tile-Sizes,
   LDS-vs-Direct-Global-Entscheidungen, Loop-Unrolling-Faktoren erzeugen.
   Das ist Datenraum, kein Typraum — Templates können das nicht
   (Monomorphisierung passiert vor GA-Tuning).

4. **Graph-Level-Fusion.** Die Fusion-Passes in Säule 2 operieren auf dem
   Graph, der Dequant-Operationen referenziert. Die IR muss introspizierbar,
   nicht nur ausführbar sein — ein Template-Instantiate ist für die
   Fusion-Logik undurchsichtig.

Die IR trennt **Was passiert mit den Bytes** (DequantOp-Programm) von **Wie
landet es auf Hardware** (Codegen-Regeln). Diese Trennung ist der Schlüssel
zum additiven Aufwands-Modell aus §1.2.

### 1.5 Scope-Definition

Die Dequant IR beschreibt den Pfad **von quantisierten Gewichten zu
WMMA/GEMV-Input-Daten**:

| Gehört zur Dequant IR | Gehört NICHT zur Dequant IR |
|---|---|
| Q-Block-Bytes → FP8/FP16 für WMMA | Aktivierungs-Quantisierung (Q8-Inline) |
| Scale-Entpackung (6-Bit, int8, FP16) | Attention, RMSNorm, RoPE, Residual |
| Nibble/Bit-Extraktion | Model-Loading, GGUF-Parsing |
| Downcast (FP32 → FP8/FP16/BF16) als letzter Schritt vor WMMA | GA-Tuning-Logik (→ `ga_tuning_spec.md`) |
| Precision-Level-Auswahl (0–3) | Kernel-Dispatch zur Laufzeit (→ Säule 4) |

Der Downcast zum WMMA-Input-Format gehört zur Dequant IR, weil er der
**letzte Schritt vor dem WMMA-Aufruf** ist und direkt vom Precision-Level
abhängt, das pro Layer unterschiedlich sein kann. Das entsprechende LDS-Write
(`StoreFP8`/`StoreHalf`) ist das terminale Op eines Dequant-Programms.

### 1.6 Design-Prinzipien (Zusammenfassung)

1. **Additive Skalierung.** Neue Formate kosten O(1), nicht O(Kernel × Varianten).
2. **FP8-E4M3 als Default, FP16 als erste Eskalation, FP32/VALU als Safety.**
   BF16 nur auf expliziten GA-Wunsch (Software-Konvertierung, 5 Instr/Elem).
3. **Arithmetik immer in FP32.** Dequant-Zwischenergebnisse sind FP32;
   Downcast ist die letzte Op. Kein FP16-Arithmetik-Pfad.
4. **Introspizierbarkeit.** Das DequantOp-Programm ist ein Rust-`Vec<DequantOp>`,
   nicht ein Template-Instantiate. Die Fusion-Passes können es lesen.
5. **v0.x-Parity als Test-Gate.** Jedes generierte Programm muss bit-identische
   (oder Precision-Level-äquivalente) Ergebnisse zu den v0.x-Referenzkerneln
   produzieren. Das ist harte Akzeptanz-Bedingung, nicht Ziel.

---

## 2. DequantOp-Spezifikation

Dieser Abschnitt definiert das vollständige Op-Set, auf dem alle
QuantFormat-Programme (Abschnitt 3) aufbauen.

### 2.1 Typ-System

Die DequantOp-Programme arbeiten auf einem virtuellen Register-Pool mit
typisierten Registern. Die Typen sind:

```rust
/// Typen für Register im Dequant-IR-Programm.
/// Arithmetik ist IMMER FP32 (§1.6 Prinzip 3); andere Typen treten nur
/// bei Load/Store-Grenzen und beim finalen Downcast auf.
pub enum RegType {
    /// 8-Bit Unsigned Integer (Byte aus Block-Memory).
    U8,
    /// 16-Bit Unsigned Integer (Byte-Paar, z. B. als Packed-FP16-Bits).
    U16,
    /// 8-Bit Signed Integer (z. B. Q6_K int8-Scales).
    I8,
    /// 16-Bit Float, IEEE 754 binary16 (llama.cpp `ggml_fp16_t`).
    F16,
    /// 16-Bit Brain Float, 8-Bit Exponent, 7-Bit Mantisse.
    BF16,
    /// 8-Bit Float, 4-Bit Exponent, 3-Bit Mantisse (OCP E4M3).
    /// Wertebereich ±448, Default-WMMA-Input.
    Fp8E4M3,
    /// 8-Bit Float, 5-Bit Exponent, 2-Bit Mantisse (OCP E5M2).
    /// Wertebereich ±57 344, KV-Cache-Default.
    Fp8E5M2,
    /// 32-Bit Float — das Arithmetik-Format der IR.
    F32,
}

/// Register-Identifikator. Der Codegen mappt RegIds auf physische
/// VGPRs (GPU) oder ZMM-Register (CPU). Der Allocator wird in Abschnitt 6
/// spezifiziert.
pub type RegId = u32;

/// Format-Varianten für FP8 (identisch zu architecture_v1.2.0-draft §2.4).
pub enum Fp8Variant {
    /// E4M3: 4 Exponent-Bits, 3 Mantissen-Bits. Range ±448.
    /// Default für Gewichte (kleiner Dynamikbereich).
    E4M3,
    /// E5M2: 5 Exponent-Bits, 2 Mantissen-Bits. Range ±57 344.
    /// Default für KV-Cache (größerer Dynamikbereich).
    E5M2,
}

/// Ziel-Format für Downcast auf 16-Bit-Präzision.
/// [SYNC-UPDATE architecture_v1.2.0-draft §2.4]: reduziert von
/// {Fp16, Bf16, Fp8E4M3, Fp8E5M2} auf {Fp16, Bf16}. FP8 wird ausschließlich
/// über die separate `DowncastToFP8`-Op angesteuert; die Mischung in HalfType
/// war redundant (siehe STOP-1-Resolution Punkt 5).
pub enum HalfType {
    Fp16,
    Bf16,
}

/// Scalar-Format für Block-Header-Felder (d, dmin).
/// Übernommen aus architecture_v1.2.0-draft §2.4.
pub enum ScalarType {
    Fp16,
    Bf16,
    Fp32,
    Int8,
    Fp8E4M3,
    Fp8E5M2,
}
```

### 2.2 Register-Konventionen

Register in der IR sind **typisiert und single-assignment** (SSA):

1. Jede RegId wird genau einmal als `dst` eines Ops produziert.
2. Der Typ einer RegId steht beim Producer fest und ändert sich nicht.
3. Der Codegen (§6/§7) mappt RegIds auf physische Register mit einem
   Linear-Scan-Allocator. Die Lebensdauer einer RegId endet beim letzten
   Consumer.

**[SPEC-ERWEITERUNG]:** Das Architektur-Dokument führt `RegId` als `u32`
ein, ohne SSA-Invariante explizit festzuhalten. Die SSA-Eigenschaft ist
für den Codegen essentiell (Linear-Scan braucht klare Live-Ranges) und
wird hier normativ festgeschrieben.

### 2.3 Kategorie 1: Speicher-Ops

Speicher-Ops laden Bytes aus dem GGUF-Block in Register. Der Block-Pointer
ist implizit (wird vom Kernel-Prolog gesetzt).

#### LoadBytes

```rust
DequantOp::LoadBytes { offset: usize, count: usize, reg: RegId }
```

**Semantik.** Lädt `count` aufeinanderfolgende Bytes beginnend bei
`block_ptr + offset` in das Register `reg`. Der Ziel-Register-Typ wird durch
den Load-Umfang bestimmt:

- `count == 1` → `reg` hat Typ `U8`.
- `count == 2` → `reg` hat Typ `U16` (Little-Endian, passt für FP16-Raw-Bits).
- `count >= 4` → `reg` hat Typ **Vektor-U8** (Codegen-spezifisch: GPU ein
  VGPR-Paar, CPU ein ZMM-Vektor). Größen bis 128 Bytes sind erlaubt.

**Alignment.** `offset` muss auf die Ziel-Granularität aligned sein
(1 Byte für `U8`, 2 Byte für `U16`, 4 Byte für Vektor-Loads). Misaligned
Loads sind nicht erlaubt und werden vom Codegen als Kompilierfehler
gemeldet.

**Semantik bei FP16-Load.** Um ein FP16-Feld (z. B. Block-Scale `d`) zu
laden, wird die Sequenz `LoadBytes { count: 2 } → IntToFloat` verwendet.
Der Codegen erkennt dieses Muster und emittiert direkt
`buffer_load_u16 + v_cvt_f32_f16` (GPU) bzw. `_mm256_loadu_si256 +
_mm512_cvtph_ps` (CPU). Siehe §6 und §7.

**Rust-Enum-Match-Beispiel:**
```rust
DequantOp::LoadBytes { offset: 0, count: 2, reg: r_d_raw }
// Lädt 2 Bytes (FP16 d) bei Block-Offset 0 in r_d_raw (Typ U16)
```

#### LoadFP8

```rust
DequantOp::LoadFP8 { offset: usize, count: usize, variant: Fp8Variant, reg: RegId }
```

**Semantik.** Lädt `count` FP8-Werte beginnend bei `block_ptr + offset` in
das Register `reg`. Der Ziel-Typ ist `Fp8E4M3` oder `Fp8E5M2` je nach
`variant`, als Vektor.

Diese Op ist **nicht nötig für die v1.0-Pflicht-Formate** (Q4_0, Q4_1, Q4_K,
Q6_K, Q8_0), weil keines davon FP8 direkt speichert. Sie ist vorgesehen für
zukünftige Formate (native FP8-Modelle, Q8-FP8-Mixed). Für die Pflicht-Formate
wird sie im Test-Harness verwendet (Smoke-Test in §10).

**Codegen:** `buffer_load_u8` (raw); WMMA kann das Register direkt
konsumieren, wenn das Dequant-Programm mit `StoreFP8` endet. Kein
Format-Konvertierungsaufwand.

### 2.4 Kategorie 2: Extraktion und Bit-Manipulation

#### ExtractNibble

```rust
DequantOp::ExtractNibble { src: RegId, high: bool, dst: RegId }
```

**Semantik.** Extrahiert ein 4-Bit-Nibble aus dem 8-Bit-Wert `src`.

- `high == false`: `dst = src & 0x0F` (unteres Nibble, Bits 0–3).
- `high == true`: `dst = (src >> 4) & 0x0F` (oberes Nibble, Bits 4–7).

`src` muss Typ `U8` haben; `dst` hat Typ `U8` mit Wertebereich 0–15.

**Verwendung.** In allen 4-Bit-Quant-Formaten (Q4_0, Q4_1, Q4_K) für die
Nibble-Entpackung. In Q4_K zusätzlich als Zwischenschritt im
paarweise-interleavten Layout (siehe Abschnitt 3).

**Anmerkung.** Diese Op existiert im Architektur-Dokument §2.4 mit
identischer Signatur.

#### Combine6Bit

```rust
DequantOp::Combine6Bit { ql: RegId, qh: RegId, shift: u8, dst: RegId }
```

**Semantik.** 6-Bit-Rekonstruktion aus zwei Byte-Quellen.

```
dst = (ql & 0x0F) | (((qh >> shift) & 0x03) << 4)
```

`ql` liefert die unteren 4 Bits (Nibble), `qh` liefert 2 Bits an der
durch `shift` bestimmten Position. Resultierender Wert ist 0–63.

`ql` und `qh` müssen Typ `U8` haben; `dst` hat Typ `U8`
(Wertebereich 0–63).

**Verwendung.** Exklusiv für Q6_K: pro Element 4 Bit aus `ql[128]` plus
2 Bit aus `qh[64]`. Der `shift`-Parameter variiert pro Element-Index
innerhalb der Gruppe von 4 Elementen, die sich ein `qh`-Byte teilen
(siehe Abschnitt 3 für die Formel).

**Anmerkung.** Diese Op existiert im Architektur-Dokument §2.4. Im Dokument
heißt sie `Combine6Bit` (nicht `CombineBits` wie im ursprünglichen Prompt).

#### ExtractBits — [SPEC-ERWEITERUNG]

```rust
DequantOp::ExtractBits { src: RegId, shift: u8, mask: u32, dst: RegId }
```

**Semantik.** Extrahiert ein beliebiges Bitfeld aus `src`:

```
dst = (src >> shift) & mask
```

`src` muss Typ `U8` oder `U16` haben; `dst` ist `U8`.

**Verwendung.** Q4_K Scale-Entpackung (`get_scale_min_k4`) benötigt
Bit-Extraktionen wie `scales[j+4] & 0x0F`, `scales[j+4] >> 4`,
`(scales[j-4] >> 6) << 4`. Diese fallen nicht unter `ExtractNibble`
(variable Shifts/Masken).

**[SPEC-ERWEITERUNG]:** Das Architektur-Dokument §2.4 führt diese Op nicht
explizit auf. Sie wird hier ergänzt, um die Q4_K-Scale-Entpackung als
geschlossenes DequantOp-Programm zu formulieren (statt als separates
`unpack_program`-Feld in `SubScalesLayout::Packed6Bit`).

**Alternativ** kann die bestehende `SubScalesLayout::Packed6Bit {
unpack_program }`-Struktur beibehalten werden, die bereits einen
`Vec<DequantOp>` vorsieht. Die Entscheidung (eigene Op vs. unpack_program)
wird bei STOP 1 diskutiert.

#### CombineBits — [SPEC-ERWEITERUNG]

```rust
DequantOp::CombineBits { lo: RegId, hi: RegId, hi_shift: u8, dst: RegId }
```

**Semantik.** Kombiniert zwei Bitfelder:

```
dst = lo | (hi << hi_shift)
```

Beide Inputs Typ `U8`; Output Typ `U8`.

**Verwendung.** Q4_K Scale-Entpackung (j ≥ 4 Pfad):
`scale = (scales[j+4] & 0x0F) | ((scales[j-4] >> 6) << 4)` verwendet
`CombineBits` mit `hi_shift = 4`.

**[SPEC-ERWEITERUNG]:** Analog zu `ExtractBits`. Kann alternativ innerhalb
eines `unpack_program` als Sequenz `ExtractBits + OR` ausgedrückt werden,
aber dann fehlt eine OR-Op — siehe STOP-1-Diskussion.

### 2.5 Kategorie 3: Arithmetik (immer FP32)

Alle Arithmetik-Ops lesen und schreiben Register vom Typ `F32`. Der
Übergang von Integer-Dequant-Werten (nach Extraktion) zu FP32 erfolgt
über `IntToFloat` (§2.6).

#### MulF32

```rust
DequantOp::MulF32 { a: RegId, b: RegId, dst: RegId }
```

**Semantik.** `dst = a * b` in IEEE-754 binary32. Round-to-nearest-even,
standardmäßig.

**Codegen:** `v_mul_f32` (GPU), `_mm512_mul_ps` (CPU).

#### FmaF32

```rust
DequantOp::FmaF32 { a: RegId, b: RegId, c: RegId, dst: RegId }
```

**Semantik.** `dst = a * b + c`, fused (einzige Rundung am Ende). Entspricht
IEEE-754 `fma`.

**Codegen:** `v_fma_f32` (GPU), `_mm512_fmadd_ps` (CPU).

**Wichtig.** `FmaF32` ist **nicht** äquivalent zu `MulF32 + AddF32` wegen der
gemeinsamen Rundung. Die v0.x-Q4_K-Formel `value = (d * scale) * nibble -
(dmin * min)` wird als `FmaF32(d_scale, nibble, -dmin_min)` emittiert,
nicht als drei getrennte Ops, um Rundungsfehler zu minimieren.

#### SubF32 — [SPEC-ERWEITERUNG]

```rust
DequantOp::SubF32 { a: RegId, b: RegId, dst: RegId }
```

**Semantik.** `dst = a - b` in IEEE-754 binary32.

**Codegen:** `v_sub_f32` (GPU), `_mm512_sub_ps` (CPU).

**Verwendung.** Q4_K-Formel als explizite Subtraktion wenn `FmaF32` mit
Negation nicht passt. Primär als Konvenienz-Op; der Codegen darf
`SubF32(a, b)` auf `FmaF32(b, const(-1.0), a)` umschreiben, muss aber nicht.

**[SPEC-ERWEITERUNG]:** Nicht im Architektur-Dokument §2.4. Ergänzt für
Formel-Lesbarkeit.

#### AddF32 — [SPEC-ERWEITERUNG]

```rust
DequantOp::AddF32 { a: RegId, b: RegId, dst: RegId }
```

**Semantik.** `dst = a + b` in IEEE-754 binary32.

**Codegen:** `v_add_f32` (GPU), `_mm512_add_ps` (CPU).

**Verwendung.** Q4_1-Formel `value = d * nibble + m` als
`FmaF32(d, nibble_fp32, m)`, aber auch als separate Op für Residual-artige
Additionen falls nötig.

**[SPEC-ERWEITERUNG]:** Wie `SubF32`. In der Praxis wird `FmaF32` fast
immer bevorzugt.

#### NegF32 — [SPEC-ERWEITERUNG]

```rust
DequantOp::NegF32 { src: RegId, dst: RegId }
```

**Semantik.** `dst = -src`. Bit-Operation auf dem Vorzeichen-Bit.

**Codegen:** `v_xor_b32 dst, 0x80000000, src` (GPU; 1 Instruktion, nutzt
VALU-XOR-Pfad statt FPU), `_mm512_xor_ps(src, sign_mask)` (CPU).

**Verwendung.** Zur Erzeugung des `-dmin * min_j`-Terms in Q4_K als
`NegF32(MulF32(dmin, min_j))`, danach in `FmaF32` konsumiert.

**[SPEC-ERWEITERUNG]:** Nicht im Architektur-Dokument. Alternative:
`MulF32` mit vorab in `Const` erzeugter `-1.0`-Konstante, aber das kostet
ein Register.

### 2.6 Kategorie 4: Konvertierung Integer → Float

#### IntToFloat

```rust
DequantOp::IntToFloat { src: RegId, offset: i32, dst: RegId }
```

**Semantik.**

```
dst = (src as f32) + (offset as f32)
```

Konvertiert den Integer-Wert in `src` (Typ `U8` oder `I8`) zu FP32, optional
mit additivem Offset. `src` wird entsprechend seines Typs interpretiert:

- `U8` → unsigned, Wertebereich 0..255.
- `I8` → signed, Wertebereich -128..127.

`offset` ist ein konstanter i32, häufig verwendet für die Zentrierung:

- Q4_0: `offset = -8` (Nibbles 0..15 → -8..7)
- Q6_K: `offset = -32` (6-Bit-Werte 0..63 → -32..31)
- Q4_K, Q8_0, Q4_1: `offset = 0`

**Codegen GPU:** Bei `offset == 0` direkt `v_cvt_f32_u32` oder
`v_cvt_f32_i32`. Bei `offset != 0` der Codegen bildet
`v_cvt_f32_i32 + v_add_f32` oder (wenn `offset` klein) eine FMA-Form. Der
Architektur-Dokument-Ansatz codiert den Offset direkt in der Intrinsic
`v_cvt_f32_ubyte0` + Subtraktion.

**Anmerkung.** Diese Op existiert im Architektur-Dokument §2.4 mit
identischer Signatur.

### 2.7 Kategorie 5: Downcast (Terminal vor WMMA/LDS)

Die Downcast-Ops sind die **letzten Schritte** eines Dequant-Programms. Sie
bestimmen das WMMA-Input-Format und hängen vom aktiven `PrecisionLevel`
(siehe Abschnitt 4) ab.

#### DowncastToFP8

```rust
DequantOp::DowncastToFP8 {
    src: RegId,
    dst: RegId,
    variant: Fp8Variant,
    saturate: bool,
}
```

**Semantik.** Konvertiert `src` (Typ `F32`) zu FP8 (Typ `Fp8E4M3` oder
`Fp8E5M2` je nach `variant`).

- `saturate == true`: Werte außerhalb des Zielbereichs werden auf
  `±max_val` (E4M3: ±448; E5M2: ±57 344) geklemmt. Die `v_cvt_pk_fp8_f32`-
  Intrinsic hat Saturation integriert. Bei Klemmung wird ein Event an den
  Quality Monitor (Säule 5) gesendet (`Fp8SaturationExceeded`).
- `saturate == false`: Wrap-around (Intrinsic-Standardverhalten ohne Flag);
  im Produktions-Pfad nicht empfohlen, nur für Debug-Zwecke.

**Codegen.** Direkt `v_cvt_pk_fp8_f32` (E4M3) bzw. `v_cvt_pk_bf8_f32` (E5M2).
Ein Intrinsic verarbeitet 2×FP32 → 2×FP8 (packed in die Hälfte eines
U32-Registers). Der Codegen-Allocator muss Pairs bilden.

**Flush-to-Zero.** Werte mit Magnitude unter dem kleinsten darstellbaren
FP8-Subnormal (E4M3: 2^-9; E5M2: 2^-16) werden auf 0 geflusht. Das
entspricht Intrinsic-Verhalten und ist numerisch akzeptabel (der
Quality Monitor tracked das separat).

**Anmerkung.** Diese Op existiert im Architektur-Dokument §2.4 mit
identischer Signatur.

#### DowncastToHalf

```rust
DequantOp::DowncastToHalf { src: RegId, dst: RegId, target: HalfType }
```

**Semantik.** Konvertiert `src` (Typ `F32`) zu einem Half-Format:

- `target == HalfType::Fp16`: `dst` Typ `F16`. Codegen: `v_cvt_f16_f32`
  (GPU, 1 Instruktion, HW-nativ); `_mm512_cvtps_ph` (CPU).
- `target == HalfType::Bf16`: `dst` Typ `BF16`. Codegen: **5-Instruktionen-
  Software-Sequenz** auf gfx1201 (keine HW-Intrinsic für FP32→BF16 auf
  RDNA 4 — siehe Memory #11, vgl. `architecture_v1.2.0-draft.md` §C.3).
  Details in §6.

**Nicht für FP8.** FP8-Downcast läuft ausschließlich über die separate
`DowncastToFP8`-Op (siehe oben). Die v1.2.0-draft-Enum-Variante
`HalfType::Fp8E4M3`/`Fp8E5M2` ist in dieser Spec entfernt
([SYNC-UPDATE], siehe §2.1).

**Anmerkung.** Die Op-Signatur existiert im Architektur-Dokument §2.4. Die
BF16-Kosten von 5 Instruktionen sind der Grund, warum die Standard-
Precision-Eskalation **FP8 → FP16 → FP32/VALU** ist und nicht
FP8 → FP16 → BF16 (siehe §4).

### 2.8 Kategorie 6: Store (Terminal)

#### StoreFP8

```rust
DequantOp::StoreFP8 {
    src: RegId,
    variant: Fp8Variant,
    lds_offset_expr: Expr,
}
```

**Semantik.** Schreibt `src` (Typ `Fp8E4M3` oder `Fp8E5M2`) in LDS (GPU) bzw.
in den Output-SIMD-Register (CPU) an der durch `lds_offset_expr` berechneten
Adresse.

`lds_offset_expr` ist ein kleiner Ausdruck im Sublanguage des Codegens
(z. B. `"i * TILE_N + col"`), der vom Emitter geparst und in
entsprechende Adress-Arithmetik übersetzt wird.

**Codegen:** `ds_write_b8` (GPU, 1 Byte per FP8-Element), `_mm512_store_epi8`
(CPU, aber siehe §7: der CPU-Pfad nutzt kein FP8, diese Op wird CPU-seitig
als Error gemeldet).

**Byte-Budget.** FP8 braucht 1 Byte/Element in LDS gegenüber 2 Bytes bei
FP16 — das verdoppelt den effektiven Tile-Size-Spielraum bei gegebenem
LDS-Budget (64 KB).

**Anmerkung.** Diese Op existiert im Architektur-Dokument §2.4 mit
identischer Signatur.

#### StoreHalf

```rust
DequantOp::StoreHalf { src: RegId, lds_offset_expr: Expr }
```

**Semantik.** Schreibt `src` (Typ `F16` oder `BF16`) in LDS bzw. Output-SIMD
an `lds_offset_expr`. Der Typ muss vor dem Store durch eine
`DowncastToHalf`-Op gesetzt worden sein.

**Codegen:** `ds_write_b16` (GPU), `_mm512_store_si512` auf FP16-Vektor (CPU).

**Anmerkung.** Diese Op existiert im Architektur-Dokument §2.4 mit
identischer Signatur.

### 2.9 Kategorie 7: Spezial-Ops

#### ScaleBlockStart — [SPEC-ERWEITERUNG]

```rust
DequantOp::ScaleBlockStart { sub_block_idx: u8 }
```

**Semantik.** Non-emittierender Marker, der den Beginn eines Sub-Blocks im
Dequant-Programm annotiert. Der Codegen nutzt das, um Scale-Loads nur
einmal pro Sub-Block zu platzieren (statt pro Element).

Kein Register-Input, kein Output. Wirkt nur auf den Codegen.

**[SPEC-ERWEITERUNG]:** Nicht im Architektur-Dokument §2.4. Wird hier
ergänzt, um die Sub-Block-Strukturierung der K-Formate (Q4_K: 8 Sub-Blöcke;
Q6_K: 16 Sub-Blöcke) im IR-Programm explizit zu markieren. Alternative:
implizite Erkennung durch den Codegen anhand von Scale-Register-Lifetimes —
fehleranfällig, Marker ist sauberer.

#### Const — [SPEC-ERWEITERUNG]

```rust
DequantOp::Const { value: f32, dst: RegId }
```

**Semantik.** `dst = value`, wobei `dst` Typ `F32` hat.

**Codegen:** Als Immediate in folgenden Ops oder als Konstanten-Register
(`v_mov_b32 dst, <imm>` auf GPU; `_mm512_set1_ps(value)` auf CPU).

**Verwendung.** `Const` ist die **selbstdokumentierende** Form der
Zentrierungs-Konstanten in den Pflicht-Formaten:

- Q4_0: `Const(8.0)` vor `SubF32(nibble_fp, const_8)` — liest sich als
  "zentriere bei 8".
- Q6_K: `Const(32.0)` vor `SubF32(q6_fp, const_32)` — liest sich als
  "zentriere bei 32".

Alternativ könnte die Zentrierung auch im `IntToFloat { offset }`-Parameter
versteckt werden, aber das ist weniger lesbar: `IntToFloat { offset: -8 }`
zeigt nicht so klar, dass hier eine fachliche Zentrierungs-Konstante wirkt.
Die STOP-1-Resolution hat sich für `Const` als primäre Form entschieden
(Punkt 4), weil es die Programme selbstdokumentierend macht — der
Laufzeit-Overhead ist null (der Codegen emittiert ein Immediate, keinen
Register-Load).

**[SPEC-ERWEITERUNG, bestätigt bei STOP-1-Resolution]:** Nicht im
Architektur-Dokument §2.4. `Const` wird als First-Class-Op übernommen.

### 2.10 Übersicht: Vollständige DequantOp-Enum

```rust
/// Vollständige DequantOp-Enum (v1.0-Spec, erweitert architecture_v1.2.0-draft §2.4).
/// Ops mit [SPEC-ERWEITERUNG] sind in dieser Spec neu; alle anderen
/// entsprechen exakt dem Architektur-Dokument.
pub enum DequantOp {
    // Speicher
    LoadBytes { offset: usize, count: usize, reg: RegId },
    LoadFP8   { offset: usize, count: usize, variant: Fp8Variant, reg: RegId },

    // Extraktion
    ExtractNibble { src: RegId, high: bool, dst: RegId },
    Combine6Bit   { ql: RegId, qh: RegId, shift: u8, dst: RegId },
    ExtractBits   { src: RegId, shift: u8, mask: u32, dst: RegId },  // [SPEC-ERWEITERUNG]
    CombineBits   { lo: RegId, hi: RegId, hi_shift: u8, dst: RegId }, // [SPEC-ERWEITERUNG]

    // Konvertierung
    IntToFloat { src: RegId, offset: i32, dst: RegId },

    // Arithmetik (immer FP32)
    MulF32 { a: RegId, b: RegId, dst: RegId },
    FmaF32 { a: RegId, b: RegId, c: RegId, dst: RegId },
    SubF32 { a: RegId, b: RegId, dst: RegId },         // [SPEC-ERWEITERUNG]
    AddF32 { a: RegId, b: RegId, dst: RegId },         // [SPEC-ERWEITERUNG]
    NegF32 { src: RegId, dst: RegId },                 // [SPEC-ERWEITERUNG]

    // Downcast (Terminal-1 vor Store)
    DowncastToHalf { src: RegId, dst: RegId, target: HalfType },
    DowncastToFP8  { src: RegId, dst: RegId, variant: Fp8Variant, saturate: bool },

    // Store (Terminal)
    StoreHalf { src: RegId, lds_offset_expr: Expr },
    StoreFP8  { src: RegId, variant: Fp8Variant, lds_offset_expr: Expr },

    // Spezial
    ScaleBlockStart { sub_block_idx: u8 },             // [SPEC-ERWEITERUNG]
    Const           { value: f32, dst: RegId },        // [SPEC-ERWEITERUNG]
}
```

**Zählung:**
- 10 Ops aus `architecture_v1.2.0-draft.md` §2.4 übernommen.
- 7 Ops als `[SPEC-ERWEITERUNG]` ergänzt.
- Gesamt: 17 DequantOp-Varianten.

### 2.11 Invarianten und Constraints

Ein DequantOp-Programm ist **wohlgeformt**, wenn:

1. **SSA-Eigenschaft.** Jede RegId erscheint genau einmal als `dst` (oder
   `reg` bei Load-Ops). Mehrfach-Assignments werden vom Validator
   abgelehnt.

2. **Typ-Konsistenz.** Jede Op konsumiert Register mit erwartetem Typ:
   - `ExtractNibble`, `ExtractBits`, `CombineBits` erwarten `U8` oder `U16`.
   - `Combine6Bit` erwartet `U8` für `ql` und `qh`.
   - `IntToFloat` erwartet `U8` oder `I8`.
   - Alle `*F32`-Arithmetik-Ops erwarten `F32`.
   - `DowncastToFP8` und `DowncastToHalf` erwarten `F32`.
   - `StoreFP8` erwartet `Fp8E4M3` oder `Fp8E5M2` (passend zu `variant`).
   - `StoreHalf` erwartet `F16` oder `BF16`.

3. **Arithmetik in FP32.** Keine Op außer Downcast/Store wechselt in
   Sub-F32-Precision. `FmaF32 { a: Fp16, ... }` ist nicht erlaubt.

4. **Terminal.** Jedes Dequant-Programm muss mit genau einer
   `Store*`-Op enden.

5. **Store-Typ ↔ WMMA-Format-Kopplung.** Der Store-Typ muss zum
   angeforderten `PrecisionLevel` (siehe §4) passen. Das Programm wird
   vom Codegen mit dem Level als Parameter instanziiert; der Validator
   prüft die Kopplung.

6. **Vektorisierung.** Load-Ops mit `count > 1` erzeugen Vektor-Register.
   Arithmetik-Ops sind **element-weise** (SIMD auf GPU, AVX-512 auf CPU);
   sie operieren auf Vektor-Registern derselben Länge.

### 2.12 Nicht-Ziele der DequantOp-Ebene

Die DequantOp-Ebene ist **nicht** gedacht für:

- **Control-Flow.** Keine Ifs, keine Loops. Loops (über Elemente in einem
  Block, über K-Chunks) werden vom Codegen emittiert, nicht im Programm
  selbst. Konditionale Scale-Entpackung (Q4_K `if (j < 4)`) wird durch
  zwei separate Programm-Varianten gelöst oder durch den Codegen bei
  Kompilier-Zeit aufgelöst.

- **Atomare Operationen.** Kein Atomics, keine Barrieren. LDS-Synchronisation
  (`s_barrier`) wird vom Kernel-Template (§5) emittiert.

- **Scalar-Broadcast.** Der Broadcast eines Sub-Block-Scales auf 32 Elemente
  ist implizit durch die Kernel-Struktur (Sub-Block-Iteration); keine Op
  dafür nötig.

---


---

## 3. QuantFormat-Definitionen

Dieser Abschnitt spezifiziert die fünf v1.0-Pflicht-Formate als vollständige
DequantOp-Programme. Q5_K ist **bewusst ausgelassen** — es wird in §9 als
Walk-Through "vom leeren Blatt" erstellt, um die Erweiterbarkeit zu
demonstrieren.

### 3.1 Wichtige Klarstellung: Q4_K vs. Q4_K_M

**Q4_K** ist ein **Block-Format** (144 Bytes, 256 Elemente, 6-Bit packed
Sub-Scales). **Q4_K_M** ist eine **Mix-Strategie** des llama-quantize-Tools:
die meisten Layer eines Modells werden in Q4_K quantisiert, einige kritische
Layer (typischerweise `attn_v.weight` und `ffn_down.weight` in bestimmten
Sub-Blöcken) in Q6_K. Die Mix-Strategie erzeugt bessere Qualität bei
geringfügig größerer Dateigröße.

**Die Dequant IR spezifiziert Block-Formate.** Mix-Strategien sind eine
Angelegenheit der Model Introspection (Säule 1): sie liest die GGUF-
Metadaten und stellt für jeden Tensor fest, welches Block-Format verwendet
wurde. Die Dequant IR sieht immer einen einzelnen Tensor mit einem einzelnen
Format; die Kombination mehrerer Formate in einem Modell ist für sie
transparent.

### 3.2 Programm-Struktur und externe Inputs

Ein DequantOp-Programm operiert in zwei konzeptionellen Phasen, die der
Codegen beim Emittieren trennt:

**Phase A — Block-Prolog (einmal pro Block).** Lädt die Block-Bytes in
Vektor-Register:

- `LoadBytes`-Ops für `qs`, `qh`, und andere packed Byte-Arrays.
- `unpack_program` aus `SubScalesLayout::Packed6Bit` entpackt die Sub-Block-
  Scales/Mins in Sub-Block-adressierbare FP32-Register.

**Phase B — Element-Schleife (pro Element im Block).** Führt die
Dequant-Arithmetik durch:

- `ExtractNibble`/`ExtractBits`/`Combine6Bit` entpacken das aktuelle Element
  aus den Vektor-Registern der Phase A.
- `IntToFloat` konvertiert zu FP32.
- `Const`/`SubF32`/`MulF32`/`FmaF32` berechnen den dequantisierten Wert.
- `DowncastToFP8` oder `DowncastToHalf` produziert das WMMA-Input-Format.
- `StoreFP8` oder `StoreHalf` schreibt in LDS.

`ScaleBlockStart` markiert Sub-Block-Wechsel und dient dem Codegen als
Hinweis, welches Sub-Block-Scale-Register für die folgenden Elemente
aktiv ist.

**Externe Input-Register (vom Kernel-Prolog gesetzt).** Der Kernel-Prolog
lädt die in `QuantFormat.block_scale_offset`/`block_scale_type` bzw.
`block_min_offset`/`block_min_type` angegebenen Felder und konvertiert
sie zu FP32, bevor das DequantOp-Programm startet. Das Programm hat dann
diese Register per Konvention verfügbar:

| Register-Name | Typ | Quelle | Belegung |
|---|---|---|---|
| `r_d` | F32 | `block_scale_offset` + Konvertierung | Immer |
| `r_dmin` | F32 | `block_min_offset` + Konvertierung | Nur wenn `block_min_offset: Some` |
| `r_scale[j]` | F32 | `unpack_program` | Nur K-Formate (Q4_K, Q6_K) |
| `r_min[j]` | F32 | `unpack_program` | Nur K-Formate mit Min (Q4_K) |

Die Namen sind Konvention; in der Rust-`QuantFormat`-Definition sind es
RegIds, die vom Codegen aufgelöst werden.

### 3.3 QuantFormat-Struct (aus architecture_v1.2.0-draft §2.4)

Unverändert übernommen als kanonische Definition:

```rust
pub struct QuantFormat {
    pub id: QuantFormatId,
    pub name: &'static str,
    pub block_bytes: usize,
    pub elements_per_block: usize,
    pub sub_blocks_per_block: usize,
    pub sub_block_size: usize,

    pub dequant_program: Vec<DequantOp>,

    pub block_scale_offset: usize,
    pub block_scale_type: ScalarType,

    pub block_min_offset: Option<usize>,
    pub block_min_type: Option<ScalarType>,

    pub sub_scales_layout: SubScalesLayout,
}

pub enum SubScalesLayout {
    Int8Array { offset: usize, count: usize },

    Packed6Bit {
        offset: usize,
        count: usize,
        unpack_program: Vec<DequantOp>,
    },

    None,
}
```

### 3.4 Q4_0 — Einfachstes Format (Baseline)

**Zweck.** 4-Bit-Quantisierung mit Block-Scale, ohne Min-Offset.
Einfache symmetrische Zentrierung bei 8.

**Block-Layout** (18 Bytes, 32 Elemente):

| Offset | Bytes | Feld | Beschreibung |
|---|---|---|---|
| 0–1 | 2 | `d` | FP16 Block-Scale |
| 2–17 | 16 | `qs[16]` | 16 Bytes, je 2 Nibbles = 32 Elemente |

**Nibble-Layout.** Byte `i` (i=0..15) enthält:
- Unteres Nibble (Bits 0–3) → Element `i` (Elemente 0..15)
- Oberes Nibble (Bits 4–7) → Element `i + 16` (Elemente 16..31)

**Dequant-Formel.** `value = d * (nibble - 8)` mit `nibble ∈ {0..15}`,
ergibt Wertebereich `[-8d, 7d]`.

**QuantFormat-Definition:**

```rust
pub const Q4_0: QuantFormat = QuantFormat {
    id: 2,                          // llama.cpp ggml_type::GGML_TYPE_Q4_0
    name: "Q4_0",
    block_bytes: 18,
    elements_per_block: 32,
    sub_blocks_per_block: 1,
    sub_block_size: 32,

    block_scale_offset: 0,
    block_scale_type: ScalarType::Fp16,

    block_min_offset: None,
    block_min_type: None,

    sub_scales_layout: SubScalesLayout::None,

    dequant_program: vec![
        // Phase A — Block-Prolog: qs-Bytes laden
        DequantOp::LoadBytes { offset: 2, count: 16, reg: R_QS },

        // Phase B — Element-Loop (Codegen iteriert i = 0..31):
        //   Für i < 16:  byte = qs[i],          high = false
        //   Für i >= 16: byte = qs[i - 16],     high = true

        // Nibble extrahieren (element-weise)
        DequantOp::ExtractNibble { src: R_QS_ELEM, high: R_HIGH_FLAG, dst: R_NIBBLE },

        // Zentrierung: value = d * (nibble - 8)
        DequantOp::Const { value: 8.0, dst: R_CONST_8 },
        DequantOp::IntToFloat { src: R_NIBBLE, offset: 0, dst: R_Q_FP },
        DequantOp::SubF32 { a: R_Q_FP, b: R_CONST_8, dst: R_Q_CENTERED },
        DequantOp::MulF32 { a: R_D, b: R_Q_CENTERED, dst: R_VAL },

        // Downcast + Store (Terminal; Precision-Level bestimmt target)
        DequantOp::DowncastToFP8 {
            src: R_VAL, dst: R_OUT,
            variant: Fp8Variant::E4M3, saturate: true,
        },
        DequantOp::StoreFP8 {
            src: R_OUT,
            variant: Fp8Variant::E4M3,
            lds_offset_expr: "elem_idx * TILE_N + col".into(),
        },
    ],
};
```

**Anmerkungen.**
- `R_QS`, `R_NIBBLE`, etc. sind named RegIds (Rust-Konstanten vom Typ
  `RegId`), die vom Codegen auf VGPRs gemappt werden.
- `R_QS_ELEM` bzw. `R_HIGH_FLAG` sind vom Codegen synthetisierte Werte
  aus der Element-Schleife, nicht vom Programm gesetzt. Der Codegen
  versteht `ExtractNibble` im Element-Loop-Kontext so, dass er den
  passenden Byte und das passende Nibble-Flag für das aktuelle Element
  auswählt (siehe §6 für Details).
- `Const(8.0)` ist gemäß STOP-1-Resolution Punkt 4 die bevorzugte Form
  gegenüber `IntToFloat { offset: -8 }`: selbstdokumentierend, null
  Overhead (wird als Immediate emittiert).

**Verifikation gegen v0.x.**

Die Referenz-Formel aus `hip_kernels/wmma/wmma_gemm_q4_0.hip`:
```
value = d * (nibble_value - 8.0f);
```
ist direkt äquivalent zu der DequantOp-Sequenz
`Const(8.0) → IntToFloat(0) → SubF32 → MulF32(d, ·)`.
Golden-Vector-Verifikation: §3.9.

### 3.5 Q4_1 — Q4_0 + Additiver Min-Offset

**Zweck.** Wie Q4_0, aber mit zusätzlichem Min-Parameter `m`. Nicht-
symmetrische Quantisierung — der Wertebereich ist `[m, m + 15d]`.

**Block-Layout** (20 Bytes, 32 Elemente):

| Offset | Bytes | Feld | Beschreibung |
|---|---|---|---|
| 0–1 | 2 | `d` | FP16 Block-Scale |
| 2–3 | 2 | `m` | FP16 Min-Offset |
| 4–19 | 16 | `qs[16]` | 16 Bytes, je 2 Nibbles = 32 Elemente |

**Nibble-Layout.** Identisch zu Q4_0: Byte `i` → unteres Nibble =
Element `i`, oberes Nibble = Element `i + 16`.

**Dequant-Formel.** `value = d * nibble + m` mit `nibble ∈ {0..15}`.

**QuantFormat-Definition:**

```rust
pub const Q4_1: QuantFormat = QuantFormat {
    id: 3,                          // llama.cpp GGML_TYPE_Q4_1
    name: "Q4_1",
    block_bytes: 20,
    elements_per_block: 32,
    sub_blocks_per_block: 1,
    sub_block_size: 32,

    block_scale_offset: 0,
    block_scale_type: ScalarType::Fp16,

    block_min_offset: Some(2),      // FP16 m an Offset 2
    block_min_type: Some(ScalarType::Fp16),

    sub_scales_layout: SubScalesLayout::None,

    dequant_program: vec![
        // Phase A
        DequantOp::LoadBytes { offset: 4, count: 16, reg: R_QS },

        // Phase B (element-weise)
        DequantOp::ExtractNibble { src: R_QS_ELEM, high: R_HIGH_FLAG, dst: R_NIBBLE },
        DequantOp::IntToFloat { src: R_NIBBLE, offset: 0, dst: R_Q_FP },

        // value = d * nibble + m
        DequantOp::FmaF32 { a: R_D, b: R_Q_FP, c: R_DMIN, dst: R_VAL },

        DequantOp::DowncastToFP8 {
            src: R_VAL, dst: R_OUT,
            variant: Fp8Variant::E4M3, saturate: true,
        },
        DequantOp::StoreFP8 {
            src: R_OUT,
            variant: Fp8Variant::E4M3,
            lds_offset_expr: "elem_idx * TILE_N + col".into(),
        },
    ],
};
```

**Anmerkungen.**
- `R_DMIN` wird vom Kernel-Prolog aus `block_min_offset: Some(2)` geladen
  (FP16 → FP32). In Q4_1 steht es für `m`, nicht für ein negiertes Min —
  daher direkt in `FmaF32 c:` eingesetzt, kein `NegF32` nötig.
- Die Fma-Form `d * nibble + m` nutzt die einzige Rundung von FmaF32
  (keine separate Mul+Add), was numerisch minimal stabiler ist als die
  drei-Op-Sequenz.

**Verifikation gegen v0.x.**
Q4_1 war in v0.x unterstützt, Referenz-Formel identisch: `value = d *
nibble_value + m`. Die Fma-Form produziert numerisch sehr leicht
unterschiedliche Ergebnisse (1 ULP maximal) — das wird im Test-Gate
(§3.9) als Toleranz behandelt.

### 3.6 Q4_K — Industriestandard mit 6-Bit-packed Scales

**Zweck.** 4-Bit-Quantisierung in 256-Element-Blöcken mit 8 Sub-Blöcken
à 32 Elementen. Jeder Sub-Block hat eigene 6-Bit-Scale und 6-Bit-Min.
Verwendet in Q4_K_M und Q4_K_S Mix-Strategien (llama.cpp-Standard).

**Block-Layout** (144 Bytes, 256 Elemente):

| Offset | Bytes | Feld | Beschreibung |
|---|---|---|---|
| 0–1 | 2 | `d` | FP16 Block-Scale |
| 2–3 | 2 | `dmin` | FP16 Block-Min |
| 4–15 | 12 | `scales[12]` | 6-Bit packed: 8 Sub-Scales + 8 Sub-Mins |
| 16–143 | 128 | `qs[128]` | 128 Bytes, paarweise interleaved |

**Sub-Scales-Entpackung (`get_scale_min_k4`).** Die 12 Bytes in `scales[]`
kodieren acht 6-Bit-Scales (für Sub-Blöcke 0..7) und acht 6-Bit-Mins
(für Sub-Blöcke 0..7). Die Packung ist asymmetrisch:

```
Für j ∈ 0..4:
    scale[j] =  scales[j]     & 0x3F
    min[j]   =  scales[j + 4] & 0x3F

Für j ∈ 4..8:
    scale[j] = (scales[j + 4] & 0x0F) | ((scales[j - 4] >> 6) << 4)
    min[j]   = (scales[j + 4] >> 4)   | ((scales[j]     >> 6) << 4)
```

Die resultierenden `scale[j]` und `min[j]` sind 6-Bit-Integer im Bereich
0..63. **Sie werden NICHT durch 64 geteilt** — das ist die historische
Bug-Falle aus v0.x Phase 7 (siehe §3.10).

**Nibble-Layout (paarweise interleaved).** Das ist das komplexeste
Layout unter den Pflicht-Formaten. Die 128 qs-Bytes bedienen 256
Elemente, verteilt über 8 Sub-Blöcke à 32 Elemente.

Sub-Blöcke werden **paarweise** (j, j+1) auf 32 qs-Bytes gelegt. Für
das Paar (j = 0, j+1 = 1), die ersten 32 qs-Bytes (0..31) halten 64
Elemente:

```
qs-Byte-Index im Paar: b ∈ 0..32 (32 Bytes im Paar)
  pair_base = (sub_block_j / 2) * 32    // 0 für (0,1), 32 für (2,3), ...

  Unteres Nibble  → Sub-Block j,   Element b
  Oberes Nibble   → Sub-Block j+1, Element b

Für das Element-Mapping:
  Sub-Block j, Element e ∈ 0..32:    byte_index = pair_base + e,  high = false
  Sub-Block j+1, Element e ∈ 0..32:  byte_index = pair_base + e,  high = true
```

**Dequant-Formel.** `value = (d * scale[j]) * nibble - (dmin * min[j])`
mit `nibble ∈ {0..15}`, `scale[j], min[j] ∈ {0..63}` als rohe 6-Bit-
Integer (nicht skaliert).

**QuantFormat-Definition:**

```rust
pub const Q4_K: QuantFormat = QuantFormat {
    id: 12,                         // llama.cpp GGML_TYPE_Q4_K
    name: "Q4_K",
    block_bytes: 144,
    elements_per_block: 256,
    sub_blocks_per_block: 8,
    sub_block_size: 32,

    block_scale_offset: 0,
    block_scale_type: ScalarType::Fp16,

    block_min_offset: Some(2),
    block_min_type: Some(ScalarType::Fp16),

    sub_scales_layout: SubScalesLayout::Packed6Bit {
        offset: 4,
        count: 12,
        unpack_program: q4_k_unpack_scales(),
    },

    dequant_program: vec![
        // Phase A — qs laden
        DequantOp::LoadBytes { offset: 16, count: 128, reg: R_QS },

        // Phase B — Element-Loop (Codegen iteriert über 8 Sub-Blöcke × 32 Elem)

        // Sub-Block-Start: markiert Wechsel, Codegen lädt r_scale[j] und r_min[j]
        DequantOp::ScaleBlockStart { sub_block_idx: R_SUB_J },

        // Nibble extrahieren (element-weise, paarweises Interleaving vom Codegen aufgelöst)
        DequantOp::ExtractNibble { src: R_QS_ELEM, high: R_HIGH_FLAG, dst: R_NIBBLE },
        DequantOp::IntToFloat { src: R_NIBBLE, offset: 0, dst: R_Q_FP },

        // d_eff = d * scale[j]     (pro Sub-Block einmal, vom Codegen gehoistet)
        DequantOp::MulF32 { a: R_D, b: R_SCALE_J, dst: R_D_EFF },

        // dmin_eff = dmin * min[j] (pro Sub-Block einmal)
        DequantOp::MulF32 { a: R_DMIN, b: R_MIN_J, dst: R_DMIN_EFF },

        // neg_dmin_eff = -dmin_eff (Codegen: kann zu Operand-Modifier werden)
        DequantOp::NegF32 { src: R_DMIN_EFF, dst: R_NEG_DMIN_EFF },

        // value = d_eff * nibble_fp + (-dmin_eff)
        DequantOp::FmaF32 { a: R_D_EFF, b: R_Q_FP, c: R_NEG_DMIN_EFF, dst: R_VAL },

        DequantOp::DowncastToFP8 {
            src: R_VAL, dst: R_OUT,
            variant: Fp8Variant::E4M3, saturate: true,
        },
        DequantOp::StoreFP8 {
            src: R_OUT,
            variant: Fp8Variant::E4M3,
            lds_offset_expr: "sub_j * 32 + e * TILE_N + col".into(),
        },
    ],
};

/// Scale-Entpackungs-Programm für Q4_K (`get_scale_min_k4`).
/// Erzeugt 8 Paare (scale[j], min[j]) als F32-Register aus den 12
/// packed 6-Bit-Bytes. Nutzt ExtractBits/CombineBits als erstklassige
/// Ops gemäß STOP-1-Resolution Punkt 1.
fn q4_k_unpack_scales() -> Vec<DequantOp> {
    vec![
        // Phase A — die 12 scale-Bytes laden
        DequantOp::LoadBytes { offset: 0, count: 12, reg: R_SCALES },

        // Phase B — für j ∈ 0..4: direkte 6-Bit-Extraktion
        //   scale[j] = scales[j]     & 0x3F
        //   min[j]   = scales[j + 4] & 0x3F
        DequantOp::ExtractBits {
            src: R_SCALES_J,
            shift: 0, mask: 0x3F,
            dst: R_SCALE_J_INT,
        },
        DequantOp::ExtractBits {
            src: R_SCALES_J_PLUS_4,
            shift: 0, mask: 0x3F,
            dst: R_MIN_J_INT,
        },

        // Phase C — für j ∈ 4..8: CombineBits-Formel
        //   scale[j] = (scales[j+4] & 0x0F) | ((scales[j-4] >> 6) << 4)
        //   min[j]   = (scales[j+4] >> 4)   | ((scales[j]   >> 6) << 4)
        DequantOp::ExtractBits {
            src: R_SCALES_J_PLUS_4,
            shift: 0, mask: 0x0F,
            dst: R_LO_S,
        },
        DequantOp::ExtractBits {
            src: R_SCALES_J_MINUS_4,
            shift: 6, mask: 0x03,
            dst: R_HI_S,
        },
        DequantOp::CombineBits {
            lo: R_LO_S, hi: R_HI_S, hi_shift: 4,
            dst: R_SCALE_J_INT,
        },

        DequantOp::ExtractBits {
            src: R_SCALES_J_PLUS_4,
            shift: 4, mask: 0x0F,
            dst: R_LO_M,
        },
        DequantOp::ExtractBits {
            src: R_SCALES_J,
            shift: 6, mask: 0x03,
            dst: R_HI_M,
        },
        DequantOp::CombineBits {
            lo: R_LO_M, hi: R_HI_M, hi_shift: 4,
            dst: R_MIN_J_INT,
        },

        // Konvertierung zu FP32 (für Multiplikation mit d / dmin)
        DequantOp::IntToFloat { src: R_SCALE_J_INT, offset: 0, dst: R_SCALE_J },
        DequantOp::IntToFloat { src: R_MIN_J_INT,   offset: 0, dst: R_MIN_J   },
    ]
}
```

**Anmerkungen.**
- `q4_k_unpack_scales()` ist eine Rust-Funktion, die den `Vec<DequantOp>`
  erzeugt. Sie kann nicht `const` sein, weil `Vec` keine const-Konstruktoren
  hat — aber sie wird zur Build-Zeit einmal aufgerufen.
- Der Codegen erkennt, dass der Bereich `j ∈ 0..4` einen anderen Entpackungs-
  Pfad hat als `j ∈ 4..8`, und emittiert zwei separate Entpackungs-
  Sequenzen. Das ist keine Runtime-Verzweigung, sondern Compile-Time-
  Entscheidung (die 8 Sub-Blöcke werden vom Codegen ausgerollt).
- Die `NegF32`-Op wird vom Codegen-Peephole-Pass erkannt: das Muster
  `NegF32(R_X) → FmaF32(..., c: R_NEG)` wird zu einer einzigen
  `v_fma_f32`-Instruktion mit **negiertem Operand-Modifier** auf `c`
  gemergt — keine separate `v_xor_b32`-Instruktion wird emittiert. Die
  IR-Form (drei Ops) bleibt lesbar; der Codegen erledigt die
  Instruktions-Fusion. Details in §6.
- **KEIN `/64` auf die 6-Bit-Scales.** Die v0.x-Implementierung (Phase 7)
  hatte einen Bug, der die 6-Bit-Integer fälschlicherweise durch 64 teilte.
  Das machte die Ergebnisse 64× zu klein. Siehe §3.10 für Test-Gate.

**Verifikation gegen v0.x.**
Die Referenz-Formel aus `hip_kernels/wmma/wmma_gemm_q4_k_m.hip` nach
dem Phase-7-Fix:
```
float scale = float(scale_raw);   // scale_raw ist 0..63, KEIN /64
float min   = float(min_raw);
value = (d * scale) * nibble - (dmin * min);
```
Direkt äquivalent zum DequantOp-Programm. Golden-Vector-Verifikation:
§3.9.

### 3.7 Q6_K — Mixed-Precision-Layer

**Zweck.** 6-Bit-Quantisierung mit int8-Sub-Scales (kein Bit-Packing
der Scales). Verwendet in Q4_K_M für Präzisions-kritische Layer (typisch
`attn_v.weight`, einige `ffn_down.weight`-Blöcke).

**Block-Layout** (210 Bytes, 256 Elemente):

| Offset | Bytes | Feld | Beschreibung |
|---|---|---|---|
| 0–127 | 128 | `ql[128]` | Untere 4 Bits der 6-Bit-Quants |
| 128–191 | 64 | `qh[64]` | Obere 2 Bits der 6-Bit-Quants (4 Elem/Byte) |
| 192–207 | 16 | `scales[16]` | Int8-Scales, je ein pro Sub-Block (16 Sub-Blöcke × 16 Elem) |
| 208–209 | 2 | `d` | FP16 Block-Scale — **am Ende des Blocks, nicht am Anfang!** |

**qh-Layout (4 Elemente pro Byte).** Jedes Byte `qh[b]` enthält die
oberen 2 Bits für 4 aufeinanderfolgende Elemente:

```
Element 4b+0: Bits 0..1   → shift = 0
Element 4b+1: Bits 2..3   → shift = 2
Element 4b+2: Bits 4..5   → shift = 4
Element 4b+3: Bits 6..7   → shift = 6
```

Das Element-zu-Byte-Mapping für `qh`: Element `i` nutzt `qh[i / 4]` mit
`shift = (i % 4) * 2`.

**ql-Layout.** `ql[128]` enthält 256 Nibbles. Analog zu Q4_0:
Element `i` (i<128) = unteres Nibble von `ql[i]`; Element `i` (i≥128)
= oberes Nibble von `ql[i - 128]`.

**Rekonstruktion.** `q6_value = ql_nibble | (qh_2bits << 4)`,
Wertebereich 0..63, zentriert bei 32.

**Dequant-Formel.** `value = d * scale[j] * (q6_value - 32)` mit
`scale[j] ∈ i8`, `q6_value ∈ {0..63}`.

**QuantFormat-Definition:**

```rust
pub const Q6_K: QuantFormat = QuantFormat {
    id: 14,                         // llama.cpp GGML_TYPE_Q6_K
    name: "Q6_K",
    block_bytes: 210,
    elements_per_block: 256,
    sub_blocks_per_block: 16,
    sub_block_size: 16,

    block_scale_offset: 208,        // ACHTUNG: am Ende, nicht am Anfang!
    block_scale_type: ScalarType::Fp16,

    block_min_offset: None,
    block_min_type: None,

    sub_scales_layout: SubScalesLayout::Int8Array {
        offset: 192,
        count: 16,
    },

    dequant_program: vec![
        // Phase A — ql und qh laden
        DequantOp::LoadBytes { offset: 0,   count: 128, reg: R_QL },
        DequantOp::LoadBytes { offset: 128, count: 64,  reg: R_QH },

        // Phase B — Element-Loop (16 Sub-Blöcke × 16 Elemente)

        DequantOp::ScaleBlockStart { sub_block_idx: R_SUB_J },

        // ql-Nibble extrahieren (high-Flag abhängig vom Element-Index)
        DequantOp::ExtractNibble { src: R_QL_ELEM, high: R_HIGH_FLAG_QL, dst: R_QL_N },

        // qh-2-Bits extrahieren: shift = (elem_idx % 4) * 2
        DequantOp::ExtractBits {
            src: R_QH_ELEM,
            shift: R_QH_SHIFT,       // 0, 2, 4, oder 6 je Element
            mask: 0x03,
            dst: R_QH_2,
        },

        // 6-Bit-Kombination: q6 = ql_nibble | (qh_2bits << 4)
        DequantOp::Combine6Bit {
            ql: R_QL_N, qh: R_QH_ELEM,
            shift: R_QH_SHIFT,
            dst: R_Q6_INT,
        },

        // Zentrierung bei 32
        DequantOp::Const { value: 32.0, dst: R_CONST_32 },
        DequantOp::IntToFloat { src: R_Q6_INT, offset: 0, dst: R_Q_FP },
        DequantOp::SubF32 { a: R_Q_FP, b: R_CONST_32, dst: R_Q_CENTERED },

        // d_eff = d * scale[j]  (einmal pro Sub-Block)
        DequantOp::MulF32 { a: R_D, b: R_SCALE_J, dst: R_D_EFF },

        // value = d_eff * (q6 - 32)
        DequantOp::MulF32 { a: R_D_EFF, b: R_Q_CENTERED, dst: R_VAL },

        DequantOp::DowncastToFP8 {
            src: R_VAL, dst: R_OUT,
            variant: Fp8Variant::E4M3, saturate: true,
        },
        DequantOp::StoreFP8 {
            src: R_OUT,
            variant: Fp8Variant::E4M3,
            lds_offset_expr: "sub_j * 16 + e * TILE_N + col".into(),
        },
    ],
};
```

**Anmerkungen.**
- `block_scale_offset: 208` ist der wichtigste Fallstrick: `d` steht **am
  Ende** des Blocks, nicht am Anfang. Diese Unregelmäßigkeit ist llama.cpp-
  kompatibel und wird durch den expliziten `block_scale_offset`-Parameter
  in der `QuantFormat`-Struct problemlos behandelt — der Codegen liest
  einfach an der richtigen Stelle.
- `SubScalesLayout::Int8Array { offset: 192, count: 16 }` bedeutet: die
  16 int8-Scales liegen direkt bei Offset 192, kein Unpack-Programm nötig.
  Der Kernel-Prolog lädt sie und konvertiert via `IntToFloat` (src: I8,
  offset: 0) zu 16 F32-Register `r_scale[0..16]`.
- `Combine6Bit` wird hier effektiv redundant verwendet: wir haben schon
  `R_QL_N` (nur low-4), und extrahieren `R_QH_2` separat. `Combine6Bit`
  macht dasselbe noch einmal. Das ist bewusst: der Op ist deklarativer
  als eine `ExtractBits + CombineBits`-Sequenz, und der Codegen kann die
  redundanten Zwischenergebnisse wegoptimieren.
- `R_QH_SHIFT` wird vom Codegen zur Compile-Zeit berechnet (`(e % 4) * 2`),
  kein Laufzeit-Wert.
- `Const(32.0)` — selbstdokumentierend gemäß STOP-1-Resolution Punkt 4.

**Verifikation gegen v0.x.**
Aus `hip_kernels/wmma/wmma_gemm_q6_k.hip`:
```
int8_t q = (ql_nibble) | ((qh_2bits) << 4);
float q6_centered = float(int(q) - 32);
value = d * float(scale_int8) * q6_centered;
```
Äquivalent zum DequantOp-Programm.

### 3.8 Q8_0 — Aktivierungs-Quantisierung

**Zweck.** 8-Bit signed Integer-Quantisierung mit Block-Scale. Nicht
für Gewichte (zu groß), aber Basis für **Q8-Inline**: Aktivierungen
werden einmalig Q8_0-quantisiert, dann in GEMV mit VNNI-Integer-MAC
konsumiert. Details in §5b (GEMV-Kernel-Spec).

**Block-Layout** (34 Bytes, 32 Elemente):

| Offset | Bytes | Feld | Beschreibung |
|---|---|---|---|
| 0–1 | 2 | `d` | FP16 Block-Scale |
| 2–33 | 32 | `qs[32]` | Signed Int8, Wertebereich -128..127 |

**Dequant-Formel.** `value = d * q8` mit `q8 ∈ int8 (-128..127)`.

**QuantFormat-Definition:**

```rust
pub const Q8_0: QuantFormat = QuantFormat {
    id: 8,                          // llama.cpp GGML_TYPE_Q8_0
    name: "Q8_0",
    block_bytes: 34,
    elements_per_block: 32,
    sub_blocks_per_block: 1,
    sub_block_size: 32,

    block_scale_offset: 0,
    block_scale_type: ScalarType::Fp16,

    block_min_offset: None,
    block_min_type: None,

    sub_scales_layout: SubScalesLayout::None,

    dequant_program: vec![
        // Phase A — qs laden.
        // WICHTIG: Bytes sind signed int8 (Wertebereich -128..127).
        // Der Codegen interpretiert R_QS für Q8_0 als I8-Vektor;
        // v_cvt_f32_i32 statt v_cvt_f32_ubyte0 wird emittiert.
        DequantOp::LoadBytes { offset: 2, count: 32, reg: R_QS },

        // Phase B — Element-Loop (trivial, keine Extraktion nötig)

        // qs[i] als signed int8 → FP32 konvertieren
        DequantOp::IntToFloat { src: R_QS_ELEM, offset: 0, dst: R_Q_FP },

        // value = d * q8
        DequantOp::MulF32 { a: R_D, b: R_Q_FP, dst: R_VAL },

        DequantOp::DowncastToFP8 {
            src: R_VAL, dst: R_OUT,
            variant: Fp8Variant::E4M3, saturate: true,
        },
        DequantOp::StoreFP8 {
            src: R_OUT,
            variant: Fp8Variant::E4M3,
            lds_offset_expr: "elem_idx * TILE_N + col".into(),
        },
    ],
};
```

**Anmerkung zur Typ-Interpretation.** Q8_0 verlangt eine `signed`-Byte-
Interpretation. `LoadBytes` führt keinen expliziten Byte-Typ — die
Unterscheidung zwischen `U8` und `I8` ist **format-spezifisch im Codegen**.
Der Codegen-Emitter für Q8_0 weiß aus dem `QuantFormat::id`, dass die
Bytes als signed int8 zu interpretieren sind, und emittiert
`v_cvt_f32_i32` statt `v_cvt_f32_ubyte0`.

Dies ist die v1.0-Konvention. Falls in Zukunft weitere Formate mit
signed-Interpretation hinzukommen und die Format-spezifische Logik im
Codegen zu groß wird, kann `LoadBytes` später um ein optionales
`interpret_as`-Feld erweitert werden — das ist keine v1.0-Aufgabe.

### 3.9 Test-Gate: Golden Vectors

Jedes der fünf Pflicht-Formate muss vor dem Codegen-Aufbau den
Golden-Vector-Test bestanden haben. Das ist harte Akzeptanz-Bedingung,
kein Ziel.

**Golden-Vector-Generierung.** Für jedes Format:

1. **3 Blöcke** aus einem echten GGUF-Modell (z. B. Qwen3-8B Q4_K_M)
   werden gesampelt.
2. Die Block-Bytes werden byte-identisch dupliziert.
3. Der llama.cpp-Referenz-Dequant (`dequantize_row_q4_0`, etc.) aus
   `ggml-quants.c` wird auf die Blöcke angewendet. Output: FP32-Vektor
   mit 3 × `elements_per_block` Werten.
4. Der v1.0-Codegen produziert den Kernel im **Level 3 (VALU-FP32)**-
   Pfad (siehe §4), der kein Downcast macht — der Output ist direkt FP32.
5. Die Dequant IR wird auf dieselben Block-Bytes angewendet. Output wird
   mit dem llama.cpp-Referenz-Output verglichen.

**Akzeptanz-Kriterium.** `max_abs_err < 1e-6` gegen die llama.cpp-
Referenz. Das ist schärfer als die FP8-Toleranz (0.125) und FP16-Toleranz
(~1e-3) — wir testen hier die **Dequant-Korrektheit**, nicht die
Downcast-Qualität. Der Downcast wird in separaten Tests validiert.

**Speicherort.** Die 3 Block-Byte-Arrays + erwartete FP32-Outputs pro
Format werden als Rust-`const`-Arrays in `tests/golden_vectors/` abgelegt
(nicht in die GGUF-Laufzeit-Dependencies eingebunden).

### 3.10 Historische Bugs aus v0.x — Test-Anforderungen

Diese Bugs traten in v0.x auf. Die DequantOp-Programme vermeiden sie
strukturell, aber das Test-Gate muss sie aktiv abfangen, falls ein
zukünftiger Programm-Bearbeiter sie versehentlich wieder einführt.

**Bug 1: Q4_K `/64` auf 6-Bit-Scales (v0.x Phase 7).**

Die 6-Bit-Integer `scale[j]` und `min[j]` sind rohe Werte 0..63. Sie
werden **nicht durch 64 geteilt**. Eine frühere v0.x-Implementierung
hatte das Teilen implizit angenommen (analog zu einem anderen
Quant-Format), was die Ergebnisse 64× zu klein machte.

**Test.** Der Golden-Vector-Test für Q4_K detektiert das automatisch:
64× zu kleine Werte liegen weit außerhalb von `max_abs_err < 1e-6`.
**Zusätzlich:** Unit-Test auf das `q4_k_unpack_scales`-Programm, der
prüft dass der erste Sub-Block-Scale eines bekannten Testblocks den
erwarteten Integer-Wert in `R_SCALE_J_INT` hat (vor `IntToFloat`).

**Bug 2: Q6_K d-Offset 0 statt 208 (v0.x Phase 5).**

Der FP16 Block-Scale `d` in Q6_K steht **am Ende** des Blocks (Offset 208),
nicht am Anfang. Eine frühe v0.3.0-Implementierung hatte das Standard-
Q-Format-Muster angenommen und `d` bei Offset 0 gelesen, was Müll ergab
(dort steht `ql[0]`).

**Test.** `block_scale_offset: 208` in der `QuantFormat`-Struct ist die
einzige Quelle der Wahrheit. Unit-Test auf `QuantFormat::Q6_K.block_scale_offset
== 208`.

**Bug 3: Q4_K paarweise Nibble-Interleaving falsch (v0.x Phase 7).**

Eine frühe v0.3.0-Implementierung hat das paarweise Interleaving als
"Sub-Block j belegt Bytes 0..15, Sub-Block j+1 belegt Bytes 16..31"
implementiert (sequentiell, wie Q4_0). Korrekt ist aber das Paar-Sharing:
Bytes 0..31 halten 64 Elemente (Sub-Block j und j+1 gemeinsam), wobei
Low-Nibbles zu j und High-Nibbles zu j+1 gehen.

**Test.** Golden-Vector-Test mit einem Block, bei dem Sub-Block 0 und
Sub-Block 1 **unterschiedliche Scales** haben. Die Interleaving-Fehler
werden sofort sichtbar, weil die Werte mit dem falschen Scale skaliert
würden.

**Bug 4: FP16-Laden mit falscher Endianness.**

GGUF ist Little-Endian. `LoadBytes { count: 2 }` liefert auf gfx1201
und Zen4 korrekt Little-Endian (beide Plattformen sind nativ Little-
Endian). Auf Big-Endian-Systemen würde das fehlschlagen, aber die sind
nicht im Scope.

**Test.** Ein expliziter Unit-Test, der einen Block mit `d = 1.0f16`
(Bit-Pattern `0x3C00`, bytes `[0x00, 0x3C]`) lädt und prüft dass
`R_D = 1.0` ergibt. Abwehr gegen versehentliche Byte-Swap-Einführung.

### 3.11 Zusammenfassung: Pflicht-Formate

| Format | Block | Elem | Sub-Blocks | block_scale_offset | block_min | Sub-Scales | Besonderheit |
|---|---|---|---|---|---|---|---|
| Q4_0 | 18 | 32 | 1 | 0 | — | None | Einfachste |
| Q4_1 | 20 | 32 | 1 | 0 | 2 | None | + Additiver Min |
| Q4_K | 144 | 256 | 8 | 0 | 2 | Packed6Bit@4 | Paarweise Interleaving, 6-Bit packed |
| Q6_K | 210 | 256 | 16 | **208** | — | Int8Array@192 | d am Ende, int8-Scales |
| Q8_0 | 34 | 32 | 1 | 0 | — | None | Aktivierungs-Basis |

---


---

## 4. Precision-Level-System

Dieser Abschnitt definiert das Precision-Level-System der Dequant IR: die
vier diskreten Stufen, die das WMMA-Input-Format und den Kernel-Pfad pro
Layer steuern. Das System ist die Schnittstelle zwischen der Dequant IR
(Säule 3), der Precision-GA (Säule 5 via rf-forge), dem Quality Monitor
(Säule 5) und dem VALU-Parity-Pfad (Säule 6).

### 4.1 Die vier Precision-Levels

```
Level 0 (Default):  Dequant → FP8-E4M3 → WMMA FP8 → Akku FP32
Level 1 (Erhöht):   Dequant → FP16     → WMMA FP16 → Akku FP32
Level 2 (Hoch):     Dequant → BF16     → WMMA BF16 → Akku FP32  (SW-Downcast!)
Level 3 (Maximum):  Dequant → FP32     → VALU FP32  (kein WMMA, Safety)
```

| Level | WMMA-Input | Downcast-Op | WMMA-Intrinsic | Downcast-Kosten | Akku |
|---|---|---|---|---|---|
| 0 | FP8-E4M3 | `DowncastToFP8(E4M3)` | `v_wmma_f32_16x16x16_fp8_fp8_w32_gfx12` | **0.5 Instr/Elem** (HW-nativ) | FP32 |
| 1 | FP16 | `DowncastToHalf(Fp16)` | `v_wmma_f32_16x16x16_f16_w32_gfx12` | 1.0 Instr/Elem (HW-nativ) | FP32 |
| 2 | BF16 | `DowncastToHalf(Bf16)` | `v_wmma_f32_16x16x16_bf16_w32_gfx12` | **5.0 Instr/Elem** (SW-Emulation) | FP32 |
| 3 | — | keine | keine | 0 (kein Downcast) | FP32 (VALU-Sequenz) |

**Level 0 ist Default.** Die gesamte v1.0-Architektur ist auf FP8-E4M3 als
Default-Pfad ausgelegt. Die Kostenverdopplung von FP16 und die 10-fache
Kostensteigerung von BF16 sind kein akzeptabler Default — beide Level sind
**Eskalations-Pfade für Ausnahme-Layer**, nicht für den Normalbetrieb.

### 4.2 Standard-Eskalation: FP8 → FP16 → FP32/VALU

Die Eskalations-Reihenfolge in Produktion ist:

```
Level 0 (Default) → Level 1 (wenn SNR-Risk in Layer detektiert)
                 → Level 3 (wenn FP16 nicht reicht; VALU-Referenz)
```

**BF16 ist NICHT der natürliche Schritt nach FP16.** Das ist eine bewusste
Design-Entscheidung aufgrund der Hardware-Realität auf gfx1201 (Memory #11):

| Format-Konvertierung | Hardware-Support | Kosten |
|---|---|---|
| FP32 → FP8 (E4M3/E5M2) | **nativ**, `v_cvt_pk_fp8_f32` | 0.5 Instr/Elem |
| FP32 → FP16 | **nativ**, `v_cvt_f16_f32` | 1.0 Instr/Elem |
| FP32 → BF16 | **nicht nativ**, Compiler-SW-Sequenz | 5.0 Instr/Elem |
| BF16 → FP32 | nativ, `v_lshlrev_b32` | 1.0 Instr/Elem |

BF16 kostet in der Downcast-Richtung 5× so viel wie FP16 bei gfx1201, ohne
substantiellen Numerik-Gewinn für die meisten Layer-Typen. FP16 hat einen
kleineren Dynamikbereich (Exponent 5 Bit) als BF16 (8 Bit), aber eine
größere Mantisse (10 Bit vs. 7 Bit) — für die typischen Gewichts-Werte
(nach Dequant von 4-Bit-Quants) ist der FP16-Dynamikbereich ausreichend,
und die bessere Mantisse ist der Numerik-Vorteil.

**BF16 als explizite GA-Option.** Die Precision-GA darf Level 2 (BF16)
für Layer wählen, bei denen der Dynamikbereich tatsächlich der limitierende
Faktor ist (z. B. Exponent-Overflow-Risiko in `attn_v` bei bestimmten
Long-Context-Prompts). Das ist eine bewusste Ausnahme, keine Standard-
Eskalation.

**WMMA-BF16 funktioniert.** Das Problem ist ausschließlich der Downcast
FP32 → BF16. Das BF16-WMMA selbst (`v_wmma_f32_16x16x16_bf16_w32_gfx12`)
ist nativ und effizient. Wenn ein Layer bereits BF16-Gewichte hat (z. B.
aus einem BF16-GGUF), gibt es keinen Downcast-Overhead.

### 4.3 Level 3: VALU-Parity-Pfad

Level 3 ist der **Safety & Debug-Pfad** und die Schnittstelle zu Säule 6:

- **Kein WMMA**, stattdessen VALU-basierte FP32-MAC-Sequenz (skalare
  Multiplikation und Addition pro Element).
- **Kein Downcast** — die FP32-Dequant-Werte werden direkt akkumuliert.
- **Langsamer** als Level 0/1/2 um einen zweistelligen Faktor, aber die
  Referenz für numerische Korrektheit.

**Zwei Verwendungen:**

1. **Produktions-Sampling (1/1000).** Bei einem von tausend WMMA-Aufrufen
   wird parallel der Level-3-VALU-Kernel ausgeführt und das Ergebnis
   verglichen. Abweichungen außerhalb der per-Format-Toleranz lösen einen
   Quality-Monitor-Event aus.

2. **GA-Validierung (1/1).** Jede vom Kernel-Tuning-GA erzeugte
   Kernel-Variante wird vor der Aufnahme in den Cache gegen den
   Level-3-VALU-Referenz-Kernel validiert. Keine Kernel-Variante kommt in
   den Cache, die nicht bit-nahe zu Level 3 ist.

Details: `architecture_v1.2.0-draft.md` §2.7 (Säule 6).

### 4.4 Level-Auswahl durch die Precision-GA

Die Precision-GA (NSGA-II, siehe Memory #5 und
`architecture_v1.2.0-draft.md` §4.3) optimiert die Precision-Konfiguration
pro Layer als Pareto-Front zwischen Qualität (KL-Divergenz zu FP32) und
Speed (tok/s). Der Suchraum:

```
4^n Kombinationen bei n Layern (pro Layer: Level 0, 1, 2, oder 3)
```

Für ein 32-Layer-Modell: `4^32 ≈ 1.8 × 10^19`. NSGA-II erkundet das
praktisch in ~15 Minuten durch evolutionäre Suche (Populations-Größe,
Mutation-Rate als GA-Hyperparameter).

**Schnittstelle.** Die Dequant IR bekommt von der GA einen
`PrecisionConfig`:

```rust
/// Precision-Konfiguration pro Layer. Wird von rf-forge (Precision-GA)
/// erzeugt und vom Codegen beim Kernel-Variant-Emittieren gelesen.
pub struct PrecisionConfig {
    /// Pro Layer-Index das gewählte Level.
    pub per_layer_level: Vec<PrecisionLevel>,

    /// KV-Cache-Precision (separate GA-Dimension, siehe §2.6).
    pub kv_cache_variant: Fp8Variant,  // typisch E5M2
}

pub enum PrecisionLevel {
    Fp8,    // Level 0: DowncastToFP8(E4M3), WMMA FP8
    Fp16,   // Level 1: DowncastToHalf(Fp16), WMMA FP16
    Bf16,   // Level 2: DowncastToHalf(Bf16), WMMA BF16
    Fp32,   // Level 3: kein Downcast, VALU
}
```

Der Codegen erzeugt pro Layer einen `KernelVariant`, der das Level als
Compile-Zeit-Konstante einsetzt. Verschiedene Layer können unterschiedliche
Varianten haben — das ist keine Laufzeit-Verzweigung, sondern
Kernel-Selection per Layer-Dispatch.

### 4.5 Quality Monitor Eskalation

Der Quality Monitor (Säule 5, siehe `architecture_v1.2.0-draft.md` §2.6)
erkennt Numerik-Drift zur Laufzeit (Hidden-State-Magnitude außerhalb des
erwarteten Bereichs). Bei detektierter Drift wird der betroffene Layer auf
das nächste Level eskaliert:

```
Level 0 (FP8)  → Level 1 (FP16)  : FP8-Saturation oder SNR-Drop
Level 1 (FP16) → Level 3 (VALU)  : FP16 reicht nicht (selten)
Level 2 (BF16) → Level 3 (VALU)  : wenn BF16 explizit gesetzt war aber scheitert
```

**BF16 ist kein automatisches Eskalations-Ziel.** Die Quality-Monitor-
Eskalation überspringt BF16, weil die Laufzeit-Kosten des BF16-Downcasts
(5 Instr/Elem) höher sind als der Wechsel zu Level 3 (VALU) bei den seltenen
Drift-Fällen. BF16 kommt nur ins Spiel, wenn die Precision-GA es explizit
für einen Layer gewählt hat.

### 4.6 Codegen-Konsequenzen

Jedes `QuantFormat`-Programm wird vom Codegen mit einem
`PrecisionLevel`-Parameter instanziiert. Abhängig vom Level:

**Level 0 (FP8):** Die terminalen Ops im Programm sind `DowncastToFP8` und
`StoreFP8`. Der WMMA-Aufruf nutzt
`__builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12`.

**Level 1 (FP16):** Der Codegen rewrite `DowncastToFP8 → StoreFP8` zu
`DowncastToHalf(Fp16) → StoreHalf`. Der WMMA-Aufruf nutzt
`v_wmma_f32_16x16x16_f16_w32_gfx12`.

**Level 2 (BF16):** Analog Level 1, mit `DowncastToHalf(Bf16)` und
`v_wmma_f32_16x16x16_bf16_w32_gfx12`. Der `DowncastToHalf(Bf16)`-Emitter
produziert die 5-Instruktionen-Sequenz aus §6.

**Level 3 (VALU):** Der Codegen strippt **alle `Downcast*`- und `Store*`-Ops
am Ende** ab. Der dequantisierte FP32-Wert wird direkt in einen
FP32-Akkumulator-Register eingefädelt; der Kernel-Rahmen ist
fundamentally anders (keine LDS-Nutzung für A/B-Fragmente, keine WMMA-
Intrinsics, stattdessen eine skalare MAC-Loop). Das bedeutet: **Level 3
nutzt nicht den generischen WMMA-Kernel** (§5a), sondern einen separaten
VALU-Kernel-Emitter. Der Emitter ist einfacher — keine Tile-Size-GA,
keine LDS-Optimierung, kein Precision-Handling.

### 4.7 Kernel-Cache-Granularität

Der Kernel-Cache (von rf-forge gepflegt) ist indiziert über:

```
(QuantFormat::id, KernelShape, PrecisionLevel, TileConfig)
```

**Realistische Cache-Größe.** Ein Modell hat in der Praxis **nicht** die
volle Matrix-Explosion der Parameter-Kombinationen. Für ein 8B-Modell:

- **~5 distinkte Shapes** pro Modell: QKV-Projektion, O-Projektion,
  Gate-Proj, Up-Proj, Down-Proj. (Attention-Shapes teilen sich oft die
  Shape mit einer der Projektionen.)
- **Pro Shape** läuft die GA mit den 4 Precision-Levels × ~3
  LDS-Strategien × ~5 Tile-Configs = ~60 exploratorischen Varianten,
  behält aber nur die **Top-3** im Cache.
- **Ergebnis:** ~5 Shapes × 3 Top-Varianten = **~15 Binaries** pro
  Modell, mit ~500 KB pro Binary ≈ **7.5 MB Cache**.

Das volle Matrix-Produkt (5 Formate × 10 Shapes × 4 Levels × 20
Tile-Configs × 2 Targets = 8 000 Binaries, 4 GB) ist ein theoretisches
Konstrukt und wird nie materialisiert. Die GA baut den Cache
**bedarfsgetrieben** auf: nur Shapes, die im geladenen Modell tatsächlich
vorkommen, werden getuned.

Die meisten Layer eines Modells landen auf Level 0 (FP8). Die GA-Lauf-Zeit
wird dadurch minimiert: wenn die GA erkennt, dass ein Layer mit Level 0
ausreichende Qualität liefert, werden Level 1/2/3 nicht weiter exploriert.

---

## 5. Generische Kernel-Specs

Dieser Abschnitt spezifiziert die beiden generischen Kernel, die alle
Quant-Formate bedienen: den WMMA-Kernel (für Prefill, M ≥ 16) und den
GEMV-Kernel (für Decode, M = 1). Jeder Kernel wird vom Codegen parametrisch
für jede Kombination aus `QuantFormat`, `PrecisionLevel`, und `TileConfig`
(vom GA) instanziiert.

**Scope-Hinweis.** Diese Specs beschreiben **was der Kernel tut** und
**welche Parameter er annimmt** — die konkreten gfx1201-Instruktionen und
Codegen-Emission-Regeln folgen in §6 und §7.

### 5.1 Parameter-Raum (beide Kernel)

```rust
/// Shape-Parameter (Runtime bekannt vor der GA).
pub struct KernelShape {
    pub m: usize,           // Output-Zeilen (Batch × Seq-Len; M=1 für Decode)
    pub n: usize,           // Output-Spalten (typisch Hidden-Dim oder FFN-Dim)
    pub k: usize,           // Kontraktions-Dim
}

/// Tile-Konfiguration (GA-optimiert pro Shape).
pub struct TileConfig {
    pub tile_m: usize,          // 16, 32, 64 (Vielfache von WMMA_M=16)
    pub tile_n: usize,          // 16, 32, 64, 128
    pub k_chunk: usize,         // 16 oder 32 (abhängig von sub_block_size)
    pub lds_strategy: LdsStrategy,
    pub num_waves: usize,       // pro Workgroup
    pub unroll_factor: usize,   // K-Loop
}

pub enum LdsStrategy {
    /// A und B über LDS (Standard für GEMM).
    LdsAB,
    /// A direkt aus Global, B aus LDS (gelegentlich schneller).
    DirectA_LdsB,
    /// Beide direkt aus Global (typisch für GEMV mit kleinem N).
    DirectAB,
}
```

Die GA durchsucht diesen Parameter-Raum pro Shape mit einer
Populations-Größe von 100 und 50 Generationen (ca. 8 Minuten pro Shape,
Memory #5). Die Top-5-Varianten landen im Cache; der Bandit (Säule 4)
wählt zur Laufzeit zwischen ihnen.

### 5.2 5a — Generischer WMMA-Kernel (Prefill, M ≥ 16)

**Zweck.** Matrix-Multiplikation `C = A * B` mit:

- `A`: dequantisierte Gewichte aus `QuantFormat`-Blöcken (Shape `M × K`),
- `B`: Aktivierungen in FP16 oder FP8 (Shape `K × N`),
- `C`: Akkumulator in FP32 (Shape `M × N`).

**Inner Loop (Pseudo-Code):**

```
für k_base in 0..K schritt K_CHUNK:
  // Phase A: Block-Prolog (einmal pro Block-Zeile)
  für jeden Block in aktueller K_CHUNK-Scheibe:
    block_ptr = &weights[block_idx]
    lade Block-Header (d, dmin falls vorhanden)
    führe unpack_program aus (Scale-Entpackung)
    führe dequant_program aus für alle Elemente im Block
      → Ergebnis in LDS als FP8 / FP16 / BF16 (je nach PrecisionLevel)

  // Phase B: Aktivierungs-Load
  lade B-Fragment aus Global oder LDS

  // Phase C: WMMA-Akkumulation
  c_frag = wmma_intrinsic(a_frag, b_frag, c_frag)
    // intrinsic variiert pro PrecisionLevel:
    //   Level 0: v_wmma_f32_16x16x16_fp8_fp8_w32_gfx12
    //   Level 1: v_wmma_f32_16x16x16_f16_w32_gfx12
    //   Level 2: v_wmma_f32_16x16x16_bf16_w32_gfx12

Write-Out: c_frag → Global Memory (C-Matrix)
```

**K-Chunk-Size.** `K_CHUNK` ist `sub_block_size × n_blocks_per_chunk` und
hängt vom Format ab:

- Q4_0, Q4_1, Q8_0: `sub_block_size = 32`, also `K_CHUNK ∈ {32, 64, 96, ...}`.
- Q4_K: `sub_block_size = 32`, also `K_CHUNK ∈ {32, 64, 96, ...}` (mit 8
  Sub-Blöcken pro Block).
- Q6_K: `sub_block_size = 16`, also `K_CHUNK ∈ {16, 32, 48, ...}`.

Die GA wählt typisch `K_CHUNK = 32` oder `64`. Ein größerer Chunk
amortisiert den Block-Prolog über mehr FMAs, braucht aber mehr LDS und
VGPRs.

**LDS-Layout.** `[TILE_M × K_CHUNK]` für A-Kachel plus `[K_CHUNK × TILE_N]`
für B-Kachel. Byte-Budget pro Element:

| Level | Bytes/Elem A-Kachel | Bytes/Elem B-Kachel |
|---|---|---|
| 0 (FP8) | 1 | 1 |
| 1 (FP16) | 2 | 2 |
| 2 (BF16) | 2 | 2 |

**Bei FP8 verdoppelt sich die effektive Tile-Size** bei gegebenem
LDS-Budget (64 KB pro Workgroup). Das ist einer der strukturellen
Vorteile des FP8-Default-Pfads.

Beispiel-Rechnung für Level 0, `TILE_M = 64, TILE_N = 128, K_CHUNK = 32`:

```
A-Kachel: 64 × 32 × 1 Byte = 2048 Bytes
B-Kachel: 32 × 128 × 1 Byte = 4096 Bytes
Padding + Doppelpufferung: ~2× → 12 KB
→ passt locker in 64 KB LDS, erlaubt hohe Occupancy
```

Dieselbe Konfiguration in Level 1 (FP16): doppelt, 24 KB — immer noch ok,
aber näher am Limit bei Double-Buffering.

**Register-Budget (VGPRs).** Pro Lane (= pro Wave, da RDNA 4 Wave32):

| Komponente | Level 0 (FP8) | Level 1 (FP16) |
|---|---|---|
| A-Fragment | 4 VGPRs | 8 VGPRs |
| B-Fragment | 4 VGPRs | 8 VGPRs |
| C-Fragment (Akku) | 8 VGPRs | 8 VGPRs |
| Dequant-Zwischenwerte | 16-24 VGPRs | 16-24 VGPRs |
| Loop-Indexe, Addresser | 8-16 VGPRs | 8-16 VGPRs |
| **Gesamt (typisch)** | **40-56 VGPRs** | **48-64 VGPRs** |

Die Zielgröße ist **104 VGPRs** pro Wave (harte Grenze aus
`architecture_v1.2.0-draft.md` §3.2 — darüber sinkt die Occupancy unter
15 Waves/CU und das Latency-Hiding bricht ein). Bei FP8 gibt es
komfortablen Headroom für Ausrollung und zusätzliche GA-getriebene
Optimierungen; bei FP16 wird das Budget enger.

**Occupancy-Berechnung.** Pro CU:

- 1 536 VGPRs verfügbar.
- Bei 64 VGPRs/Wave: `1536/64 = 24` Waves/CU.
- Bei 104 VGPRs/Wave: `1536/104 ≈ 14.8 → 14` Waves/CU (abgerundet, Sweet-Spot).
- Bei 256 VGPRs/Wave: `1536/256 = 6` Waves/CU (zu niedrig für Decode).

Zusätzlich: LDS-Budget. Bei 12 KB LDS/Workgroup und 64 KB LDS/CU: 5
Workgroups/CU → mit 4 Waves/Workgroup = 20 Waves/CU. Im FP16-Fall:
24 KB LDS/Workgroup → 2 Workgroups/CU → 8 Waves/CU. FP8 gewinnt hier
deutlich.

Auf 64 CUs insgesamt: bei Level 0 `14 × 64 = 896` Waves aktiv, bei
Level 1 deutlich weniger. Mehr aktive Waves = besseres Memory-Latency-
Hiding.

**Padding-Regeln.** `M`, `N`, `K` müssen vom Codegen auf 16er-Grenzen
(WMMA-Tile-Size) gepadded werden:

```
M_padded = round_up(M, 16)
N_padded = round_up(N, 16)
K_padded = round_up(K, sub_block_size)
```

Das Padding im `K`-Dim respektiert die Block-Struktur (immer ganze
Sub-Blöcke). Für `M_padded - M > 0` werden die Padding-Zeilen mit 0
befüllt; die `WriteOut` schreibt nur die echten `M × N`-Werte in die
C-Matrix.

**Precision-Level-System im Kernel.** Der Precision-Level ist eine
**Compile-Zeit-Konstante** im Kernel-Emitter, keine Laufzeit-
Verzweigung. Der Codegen produziert vier separate Kernel-Binaries pro
`(QuantFormat, KernelShape, TileConfig)`-Kombination, einen pro Level,
und der Layer-Dispatcher wählt zur Laufzeit den richtigen.

### 5.3 5b — Generischer GEMV-Kernel (Decode, M = 1)

**Zweck.** Matrix-Vektor-Multiplikation `y = A * x`:

- `A`: dequantisierte Gewichte aus `QuantFormat`-Blöcken (Shape `N × K`),
- `x`: Aktivierungs-Vektor (FP32 oder Q8-Inline; Shape `K`),
- `y`: Output-Vektor (FP32; Shape `N`).

GEMV ist der **Decode-Hot-Path** und **bandbreiten-gebunden**. Die
Peak-Performance-Ziele aus Memory #4:

- 8B Q4_K_M: 125 tok/s Decode (BW-Limit 143, Utilization 87%)
- 14B Q4_K_M: 63 tok/s Decode (BW-Limit 72, Utilization 87%)

Das Ziel ist, das Bandbreiten-Limit zu 85%+ zu sättigen. Das bedeutet:
Compute-Ops und LDS-Zugriffe dürfen nicht auf Global-Memory warten;
Memory-Latency-Hiding durch genügend aktive Waves ist essentiell.

**Inner Loop (Pseudo-Code):**

```
// Jeder Thread bekommt eine (oder mehrere) Output-Zeilen (Output-Lanes)
row = blockIdx * WAVES_PER_BLOCK + wave_idx

acc = 0.0
für k_base in 0..K schritt K_CHUNK:
  // Block-Prolog (einmal pro Block)
  block_ptr = &weights[row * blocks_per_row + k_base / elements_per_block]
  lade Block-Header (d, dmin)
  führe unpack_program aus
  führe dequant_program aus

  // Aktivierungs-Load
  lade x[k_base..k_base + K_CHUNK]  (FP32 oder int8 bei Q8-Inline)

  // Akkumulation
  für e in 0..K_CHUNK:
    acc += dequant_weight[e] * x[k_base + e]

// Write-Out
y[row] = acc
```

**Varianten als Flags (nicht als separate Kernel).** Der GEMV-Kernel
akzeptiert vier orthogonale Flags:

```rust
pub struct GemvVariant {
    /// Input-Modus für x:
    /// - false (Standard): x ist FP32.
    /// - true (Q8-Inline): x ist int8 (mit Block-Scales), VNNI-Integer-MAC.
    pub q8_inline: bool,

    /// Residual-Addition: y = y_gemv + residual (elementweise).
    pub fuse_residual: bool,

    /// RMSNorm nach Residual: y = rms_norm(y_gemv + residual).
    pub fuse_norm: bool,

    /// Direct-Global vs. LDS-Staging für x.
    pub x_via_lds: bool,
}
```

Die `GemvVariant` ist GA-optimiert pro Shape (ähnlich `TileConfig`). Der
Codegen emittiert pro `GemvVariant` eine Kernel-Binary.

**Multi-Row-Dispatch.** Bei kleinen `K` (z. B. `attn_q` mit `K = 4096`)
unterausgelastet ein einzelner Thread die GPU. Der GEMV-Kernel nutzt
**Multi-Row-Dispatch**: ein Workgroup verarbeitet mehrere Output-Zeilen
parallel. Typisches Layout: 8 Waves × 4 Cols = 32 Output-Lanes pro
Workgroup. Die GA sucht die optimale Kombination.

**Q8-Inline-Pfad.** Für Formate mit signifikanter Quantisierungs-Bandbreite
(Q4_K, Q6_K) ist es oft günstiger, die Aktivierung **einmal** pro Token
in Q8_0 zu quantisieren und im GEMV einen **Integer-MAC** mit
VNNI-Instruktionen (CPU) bzw. `v_dot4_i32_i8` (GPU) zu verwenden, statt
jedes Element zu dequantisieren und in FP32 zu multiplizieren.

**Dual-Akkumulator-Pattern für Formate mit Min-Offset.** Bei Q4_K
(Formel `value = d * scale * q - dmin * min`) spaltet sich der Q8-Inline-
Integer-MAC in zwei Akkumulatoren:

```
int_dot = dot_product(int8_x, int8_q)    // Integer-MAC, VNNI
q8_sum  = sum(int8_x)                     // nebenher summieren

acc = d * scale * int_dot - dmin * min * q8_sum
```

Der zweite Akkumulator kompensiert den `-dmin * min`-Term, der bei der
Integer-Form sonst `N × dmin × min`-fach aufsummiert würde. Der
Dual-Akku ist ein v0.x-gelerntes Optimum; die IR-Form erzeugt das
automatisch aus der Q4_K-Dequant-Formel.

**Direct-Global vs. LDS.** In v0.x wurde beobachtet, dass **Direct-Global-
Loads für bestimmte Shapes 5× schneller** als LDS-Staging sind (Memory #8
Erkenntnis 4). Die Intuition ist: LDS-Staging lohnt sich nur, wenn
Daten mehrfach genutzt werden. Bei GEMV mit kleinem K und großem N wird
jedes Gewicht genau einmal gelesen — LDS ist unnötiger Umweg.

Die GA sucht `x_via_lds: bool` als Flag pro Shape. Der Bandit (Säule 4)
wählt zur Laufzeit zwischen den getunten Varianten.

**Residual-Fusion.** Die fused Form `y = norm(y_gemv + residual)` spart
zwei Global-Memory-Round-Trips (einmal y_gemv schreiben/residual lesen,
einmal norm schreiben). Für die Post-Attention-Projektion (`attn_o` + RMS)
ist das der Decode-Hot-Spot schlechthin.

**VGPR-Budget (GEMV).** Typisch 48-72 VGPRs pro Wave, weil GEMV weniger
Tile-Zustand als GEMM braucht. Das erlaubt 20-32 Waves/CU, was das
Memory-Latency-Hiding maximiert. FP8 hilft hier weniger als bei GEMM,
weil GEMV nicht compute-gebunden ist — aber der FP8-KV-Cache-Pfad
(Memory #5) spart beim KV-Cache-Read Bandbreite.

**Precision-Level und GEMV.** GEMV ist bandbreiten-gebunden; der
WMMA-Compute-Vorteil von FP8 greift nicht. **Aber:** der FP8-Downcast
im Dequant-Programm halbiert den LDS-Footprint des Gewichts-Stagings
(falls genutzt) und reduziert den Registerbedarf. Praktisch ergibt sich
bei GEMV ein kleiner, aber messbarer Vorteil von Level 0 gegenüber
Level 1 (~5-10% in v0.x-Messungen).

### 5.4 Fallback-Kernel: Level 3 VALU

Level 3 ist fundamentally anders strukturiert und **nutzt nicht den
generischen WMMA-Kernel** (§5.2) oder GEMV-Kernel (§5.3). Stattdessen
emittiert der Codegen einen **VALU-skalar-Kernel**:

```
für jeden Output-Element (m, n):
  acc = 0.0
  für k in 0..K:
    weight_fp32 = dequant_program_bis_fp32(block_ptr[k])
    acc += weight_fp32 * x[k]
  y[m, n] = acc
```

Das ist deutlich langsamer (kein WMMA, keine Parallelisierung pro Tile)
aber **numerisch eindeutig** und dient als Bit-Referenz. Der VALU-Kernel
wird vom Codegen **zusätzlich** zu jedem Level-0/1/2-Kernel emittiert —
er ist der Partner des VALU-Parity-Checkers (Säule 6).

**Performance-Erwartung für Level 3.** Circa 1-5 tok/s Decode — nicht
produktionstauglich, aber als Sampling-Referenz (1/1000) akzeptabel.
Der GA-Validierungs-Einsatz (1/1) ist der dominante Use-Case.

### 5.5 Kernel-Invarianten

Jeder generische Kernel erfüllt folgende Invarianten:

1. **Ein Kernel pro `(QuantFormat, KernelShape, PrecisionLevel,
   TileConfig)`-Tupel.** Keine Laufzeit-Verzweigung zwischen Formaten
   oder Levels innerhalb eines Kernels.

2. **Akkumulator ist immer FP32.** Keine Sub-FP32-Akkus. Das gilt auch
   für Q8-Inline-Integer-MAC: die Integer-Ergebnisse werden am
   Sub-Block-Ende mit dem FP32-Scale multipliziert.

3. **VRAM-Arena-Alignment.** Alle Pointer (Weights, x, y, residual,
   norm-weights) sind 256-Byte-aligned (VRAM-Arena-Konvention,
   `architecture_v1.2.0-draft.md` §3.6).

4. **Single-Stream-Dispatch.** Alle Kernel-Aufrufe laufen im Default-
   Stream, mit optionalem 2-Layer-Lookahead (Memory #7). Kein
   Multi-Stream-Overlap.

5. **Dirty-Flag-Telemetrie.** Jeder Kernel setzt bei Abschluss ein
   Dirty-Flag in der `hipHostMallocMapped`-Region, das von der
   Host-Side-Loop gepollt wird (Memory #7).

6. **`fuse_norm=true` ist Graph-Level-Guarantee.** Der Flag ist nur
   gültig, wenn der GEMV-Op der **letzte GEMV vor einem Residual-Add**
   im Computation Graph ist. Der Fusion-Pass (Säule 2) garantiert das
   vor der Kernel-Emission. Verletzung ist ein **Compile-Time-Error**,
   kein Runtime-Check — wenn der Codegen einen Kernel mit
   `fuse_norm=true` für eine Position erzeugt, an der keine Norm folgt,
   ist das ein Bug im Fusion-Pass.

7. **`q8_inline=true` nur bei PrecisionLevel 0/1/2.** Level 3 ist der
   VALU-FP32-Referenzpfad und arbeitet ausschließlich ohne
   Quantisierung. `q8_inline` bei Level 3 ist Compile-Time-Error. Die
   Codegen-Dispatch-Logik prüft das als harte Assertion.

### 5.6 Kernel-Cache-Schlüssel

Der vollständige Cache-Schlüssel für eine konkrete Kernel-Binary:

```rust
pub struct KernelCacheKey {
    pub quant_format_id: QuantFormatId,
    pub shape: KernelShape,
    pub precision_level: PrecisionLevel,
    pub tile_config: TileConfig,
    pub gemv_variant: Option<GemvVariant>,  // None für GEMM
    pub target: KernelTarget,               // Gfx1201 oder Zen4Plus
}
```

**Cache-Invalidierung statt Key-Dimension.** `rocm_version` ist **nicht**
Teil des Cache-Keys. Stattdessen wird ein globaler
`cache_invalidation_hash` in einer Cache-Metadaten-Datei abgelegt:

```rust
pub struct CacheMetadata {
    /// SHA-256 über `hipcc --version` + Toolchain-relevante Pfade.
    /// Bei Änderung wird der gesamte Cache invalidiert.
    pub invalidation_hash: [u8; 32],

    pub cache_format_version: u32,
    pub created_at: DateTime<Utc>,
}
```

Bei einem ROCm-Update (`pacman -Syu`) ändert sich
`hipcc --version`, der Hash bleibt hängen, und rf-forge invalidiert den
gesamten Cache beim nächsten Start. Das ist sauberer als
`rocm_version` pro Key, weil (a) Code-Object-Format-Änderungen meist
**alle** Kernel betreffen, nicht einzelne, und (b) der Cache-Lookup
schneller ist (ein Hash-Check statt String-Vergleich pro Key).

---


---

## 6. Codegen GPU (gfx1201)

Dieser Abschnitt spezifiziert, wie der GPU-Emitter DequantOp-Programme
in HIP-Code mit RDNA 4 Assembly-Intrinsics übersetzt. Zielplattform ist
ausschließlich gfx1201 (RX 9070 XT, Navi 48 im Vollausbau).

### 6.1 Emitter-Architektur

Der GPU-Emitter arbeitet in fünf Phasen:

```
1. Lowering:       DequantOp-Programm → interne LIR (Low-Level IR)
2. Peephole:       LIR-Transformationen (NegF32 → Operand-Modifier etc.)
3. Register-Alloc: Linear-Scan auf VGPR-Budget (max 104)
4. Kernel-Rahmen:  Prolog, LDS-Deklarationen, Main-Loop, Epilog
5. Emission:       HIP-C mit `__builtin_amdgcn_*`-Intrinsics
```

Der Output wird von `hipcc --offload-arch=gfx1201 -O3 -shared -fPIC` zu
einem `.co`-Code-Object kompiliert und zur Laufzeit via `hipModuleLoad`
in die Runtime geladen.

### 6.2 Instruktions-Mapping pro DequantOp

**Speicher-Ops:**

| DequantOp | gfx1201-Instruktion(en) | Anmerkung |
|---|---|---|
| `LoadBytes { count: 1 }` | `buffer_load_u8` + `v_cvt_f32_ubyte0` (wenn direkt zu FP32) | U8 → F32 in einer Sequenz |
| `LoadBytes { count: 2 }` (FP16-Kontext) | `buffer_load_u16` + `v_cvt_f32_f16` | FP16 Block-Scale-Laden |
| `LoadBytes { count: 1 }` (Q8_0 signed) | `buffer_load_i8` + `v_cvt_f32_i32` | I8-Interpretation (§3.8) |
| `LoadBytes { count: 4..16 }` | `buffer_load_dwordx2`/`buffer_load_dwordx4` | Vektor-Load, 1 VGPR-Paar/Quad |
| `LoadBytes { count: 32..128 }` | Mehrere `buffer_load_dwordx4` in Schleife | Vom Codegen unroll-bar |
| `LoadFP8 { ... }` | `buffer_load_u8` (raw) | WMMA liest direkt, kein cvt |

**Extraktion:**

| DequantOp | gfx1201-Instruktion | Semantik |
|---|---|---|
| `ExtractNibble { high: false }` | `v_and_b32 dst, src, 0x0F` | `dst = src & 0xF` |
| `ExtractNibble { high: true }` | `v_lshrrev_b32 dst, 4, src` | `dst = src >> 4` |
| `ExtractBits { shift, mask }` | `v_bfe_u32 dst, src, shift, width` | Bit-Field-Extract; `width = popcount(mask)` wenn zusammenhängend |
| `Combine6Bit { shift }` | `v_bfe_u32 tmp, qh, shift, 2` + `v_and_b32 q, ql, 0x0F` + `v_lshl_or_b32 dst, tmp, 4, q` | 3 Instr; `v_lshl_or_b32` ist gfx1201-nativer Combine |
| `CombineBits { hi_shift }` | `v_lshl_or_b32 dst, hi, hi_shift, lo` | 1 Instr |

**Konvertierung:**

| DequantOp | gfx1201-Instruktion | Rounding |
|---|---|---|
| `IntToFloat { offset: 0 }` auf `U8` | `v_cvt_f32_ubyte0` | Exakt (ganzzahlige Werte) |
| `IntToFloat { offset: 0 }` auf `I8` | `v_cvt_f32_i32` (nach sign-extend) | Exakt |
| `IntToFloat { offset: n }` | `v_cvt_f32_i32` + `v_add_f32 dst, imm:n, tmp` | RNE auf Add |

**Arithmetik:**

| DequantOp | gfx1201-Instruktion | Rounding |
|---|---|---|
| `MulF32` | `v_mul_f32` | RNE |
| `AddF32` | `v_add_f32` | RNE |
| `SubF32` | `v_sub_f32` | RNE |
| `FmaF32` | `v_fma_f32` | Single-Rounding RNE |
| `NegF32` | **Operand-Modifier am Konsumenten** (Peephole) | Kosten-frei |
| `Const { value }` | `v_mov_b32 dst, lit:value` oder Immediate am Konsumenten | — |

**Downcast:**

| DequantOp | gfx1201-Instruktion(en) | Anmerkung |
|---|---|---|
| `DowncastToFP8 { E4M3 }` | `v_cvt_pk_fp8_f32 dst, a, b, old, word_hi` | 1 Instr pro **Paar** (§6.5) |
| `DowncastToFP8 { E5M2 }` | `v_cvt_pk_bf8_f32 dst, a, b, old, word_hi` | 1 Instr pro Paar |
| `DowncastToHalf { Fp16 }` | `v_cvt_f16_f32` | 1 Instr, RNE |
| `DowncastToHalf { Bf16 }` | **5-Instruktionen-SW-Sequenz** | §6.6, RNE + NaN-Handling |

**Store:**

| DequantOp | gfx1201-Instruktion |
|---|---|
| `StoreFP8 { ... }` | `ds_write_b8` (1 Byte) |
| `StoreHalf { ... }` | `ds_write_b16` (2 Bytes) |

**Spezial:**

| DequantOp | Behandlung |
|---|---|
| `ScaleBlockStart { sub_block_idx }` | **Kein Instruktions-Output.** Marker wird vom Scheduler genutzt, um Scale-Register-Loads aus dem Sub-Block-Prolog an der richtigen Stelle zu platzieren (Code-Motion out of Element-Loop). |

### 6.3 Element-Loop-Synthese

Die in §3 verwendeten Register wie `R_QS_ELEM`, `R_HIGH_FLAG`, `R_SUB_J`,
`R_QH_SHIFT` werden vom Programm nicht explizit gesetzt — sie sind
**Element-Loop-gebundene Werte**, die der Codegen aus dem Loop-Index
synthetisiert. Dieser Abschnitt präzisiert die Konvention.

**Loop-Index.** Der Codegen umschließt das Dequant-Programm der Phase B
mit einer Element-Loop `for e in 0..elements_per_block`. Der Index `e`
ist eine Compile-Time-Konstante bei ausgerollten Loops oder eine
VGPR-Variable bei erhaltenen Loops.

**Synthese-Regeln (Default-Konvention):**

```
Aus e werden abgeleitet:

sub_j      = e / sub_block_size              // aktueller Sub-Block
elem_in_sb = e % sub_block_size              // Index im Sub-Block

Für Q4_0/Q4_1 (Byte-Layout: qs[i] → (elem i, elem i+16)):
  R_QS_ELEM   = R_QS[e mod 16]
  R_HIGH_FLAG = (e >= 16)

Für Q4_K (paarweises Interleaving, §6.4):
  pair_idx  = sub_j / 2
  pair_base = pair_idx * 32
  R_QS_ELEM = R_QS[pair_base + elem_in_sb]
  R_HIGH_FLAG = (sub_j % 2 == 1)

Für Q6_K (ql + qh mit 4-pro-Byte):
  R_QL_ELEM        = R_QL[e mod 128]
  R_HIGH_FLAG_QL   = (e >= 128)
  R_QH_ELEM        = R_QH[e / 4]            // 4 Elemente teilen sich ein qh-Byte
  R_QH_SHIFT       = (e mod 4) * 2          // 0, 2, 4, 6

Für Q8_0 (sequentiell):
  R_QS_ELEM = R_QS[e]
```

**Ausrollungs-Regel.** Der Codegen rollt Element-Loops vollständig aus,
wenn `elements_per_block <= 64`. Das trifft auf alle Pflicht-Formate zu
(max 256 bei Q4_K/Q6_K; ausgerollt pro Sub-Block von 16/32 Elementen).
Ausgerollte Loops erlauben Constant-Folding der Loop-Index-Arithmetik
und eliminieren Branch-Overhead.

**Konsequenz für die IR-Programme.** Die `R_*_ELEM`/`R_*_FLAG`-Register
im Programm sind **symbolisch**; der Codegen ersetzt sie beim Emittieren
durch die konkreten Register/Immediates. Das ist keine IR-Laufzeit-
Variable, sondern ein Codegen-Template-Parameter.

### 6.4 Q4_K paarweise Byte-Adressierung (Codegen-Rezept)

Das paarweise Interleaving von Q4_K ist der komplexeste Adressierungs-Fall
unter den Pflicht-Formaten. Das Rezept:

```
// Q4_K: 128 qs-Bytes für 256 Elemente, 8 Sub-Blöcke à 32 Elemente.
// Sub-Blöcke werden paarweise (j, j+1) auf 32 Bytes gelegt.

for e in 0..256 {
    let sub_j      = e / 32;                    // 0..8
    let elem_in_sb = e % 32;                    // 0..32

    let pair_idx   = sub_j / 2;                 // 0..4
    let pair_base  = pair_idx * 32;             // 0, 32, 64, 96
    let byte_idx   = pair_base + elem_in_sb;    // 0..128

    let high_flag  = sub_j % 2 == 1;            // Low-Nibble für j gerade, High für ungerade

    // ExtractNibble
    let nibble = if high_flag {
        (qs[byte_idx] >> 4) & 0x0F
    } else {
        qs[byte_idx] & 0x0F
    };

    // weiter mit IntToFloat, MulF32, FmaF32, ...
}
```

**Emission.** Der Codegen emittiert diese Arithmetik als
Compile-Time-Berechnung (wenn ausgerollt) oder als kleine VALU-Sequenz
(wenn Loop erhalten). Für die typischen ausgerollten Fälle sind die
Byte-Indizes und Nibble-Flags Konstanten, die direkt als
Immediate-Adressierungs-Offsets in `buffer_load_u8` landen.

**Alternative Sub-Block-Organisation.** Eine GA-Variante kann die
Element-Iteration **pro Sub-Block-Paar** statt pro Element organisieren:

```
for pair in 0..4 {                   // 4 Paare à (j, j+1)
    let pair_base = pair * 32;
    for e in 0..32 {                 // 32 Bytes im Paar
        let byte = qs[pair_base + e];
        let nibble_j   = byte & 0x0F;       // Sub-Block j
        let nibble_jp1 = (byte >> 4) & 0x0F; // Sub-Block j+1
        // Zwei Elemente pro Byte-Load verarbeitet → bessere LDS/VGPR-Nutzung
    }
}
```

Dieser Zwei-Elemente-pro-Byte-Pfad spart Byte-Loads (128 statt 256) und
ist in v0.x die performantere Form gewesen. Die GA darf beide Varianten
explorieren.

### 6.5 FP8 Pair-Packing

Die FP8-Konvertierungs-Intrinsic `v_cvt_pk_fp8_f32` erzeugt **zwei** FP8-
Werte aus zwei FP32-Werten in einem Aufruf:

```cpp
// Signatur: (Memory #10)
uint32_t v_cvt_pk_fp8_f32(float a, float b, uint32_t old, bool word_hi);
// Erzeugt: packed[word_hi*16 .. word_hi*16 + 8] = FP8(a)
//          packed[word_hi*16 + 8 .. word_hi*16 + 16] = FP8(b)
// `old` erhält die anderen 16 Bits (für Pack-Akkumulation)
```

**Codegen-Konsequenz.** Der Emitter muss `DowncastToFP8`-Ops **paarweise**
batchen: zwei `DowncastToFP8 { src: R_A }` und `DowncastToFP8 { src: R_B }`
werden zu einem einzigen `v_cvt_pk_fp8_f32 dst, R_A, R_B, old=0, word_hi=0`
zusammengefasst.

Das geht auf Kosten der SSA-Reinheit der Ziel-Register: beide Ziel-FP8-
Werte landen im selben physischen VGPR als gepackte Bytes, auseinander-
gezogen erst beim `ds_write_b8`.

**Batching-Strategie (Default: sequential).** Der Emitter iteriert die
Element-Loop **sequentiell**: Elemente 0-3 landen in VGPR[0], 4-7 in
VGPR[1], 8-11 in VGPR[2], 12-15 in VGPR[3]. Pro VGPR zwei
`v_cvt_pk_fp8_f32`-Instruktionen (`word_hi=0` und `word_hi=1`), dann eine
einzige `ds_write_b32` (4 Bytes) in LDS. Das ist effizienter als vier
`ds_write_b8`.

Die GA darf alternativ **interleaved** batchen (Elemente 0, 4, 8, 12 in
VGPR[0] etc.), wenn das für bestimmte Tile-Configs bessere
LDS-Bank-Conflict-Eigenschaften hat. Default ist sequential.

**Ungerade Elementzahl.** Wenn die Element-Zahl pro Workitem ungerade
ist (kommt bei den Pflicht-Formaten nicht vor, aber bei hypothetischen
Formaten möglich), wird der letzte FP8-Wert mit einem Padding-Aufruf
(zweiter Operand = 0.0) emittiert.

### 6.6 BF16 Software-Sequenz (DowncastToHalf mit target=Bf16)

gfx1201 hat **keine native HW-Instruktion** für `FP32 → BF16`
(Memory #11). Der Compiler erzeugt eine 5-Instruktionen-Sequenz mit
korrektem Round-to-Nearest-Even und NaN-Propagation:

```asm
; Input: v_src (FP32 in VGPR)
; Output: v_dst (BF16 in unteren 16 Bits eines VGPR)

; 1. Extract bits 15:0 (die "Round-Entscheidung"-Bits)
v_bfe_u32  v_round,     v_src, 0,  16     ; round = src[15:0]

; 2. Bit-16 (LSB der Mantisse nach Truncation) für RNE-Tie-Break
v_and_b32  v_lsb_mant,  v_src, 0x00010000  ; lsb_mant = src & (1 << 16)

; 3. NaN-Check: exponent == 0xFF UND mantisse != 0 → QNaN-Force
v_cmp_u_f32  vcc, v_src, v_src             ; vcc = isNaN(src)

; 4. Rounding-Hinzufügen: src + round_constant
;    round_constant = 0x7FFF + (lsb_mant >> 16) = 0x7FFF oder 0x8000
v_add3_u32  v_rounded,  v_src, v_lsb_mant, 0x7FFF

; 5. NaN-Pfad: falls NaN, setze QNaN-Pattern; sonst truncate zu BF16
v_cndmask_b32 v_dst_raw, v_rounded, 0x7FC00000, vcc  ; QNaN if NaN
; (final v_lshrrev_b32 v_dst, 16, v_dst_raw ergibt BF16 in low 16 Bits)
```

Das Resultat hat korrekte RNE-Rundung und IEEE-754-konforme NaN-
Propagation. Die 5 Instruktionen sind der Grund, warum BF16 (Level 2)
die Kostenstufe "Hoch" ist (§4).

**Truncation-Alternative** (1 Instruktion, aber 0.5 ULP Bias, nicht
IEEE-konform):
```asm
v_lshrrev_b32 v_dst, 16, v_src        ; dst = src >> 16 (truncate)
```

Der Codegen verwendet **ausschließlich die RNE-Sequenz**. Die
Truncation-Alternative ist für v1.0 nicht aktiviert; sie könnte in v1.1
als GA-Kandidat ergänzt werden ("1 Instr, aber numerisch schlechter"),
steht aber aktuell nicht im Raum.

### 6.7 Register-Allocator (Linear-Scan)

Der Emitter nutzt einen Standard-Linear-Scan-Allocator auf dem LIR:

1. Live-Range-Analyse pro RegId (von `dst`-Assignment bis letztem
   Consumer).
2. Sortierung der Live-Ranges nach Start-Punkt.
3. Pro Range: Allokierung einer freien VGPR; wenn alle 104 belegt,
   Spill (unwahrscheinlich bei korrekt dimensionierten Programmen).
4. Operand-Modifier-Peephole (siehe §6.8) vor dem Allocator, um
   `NegF32`-Ziele zu eliminieren.

**VGPR-Budget-Check.** Nach der Allokation prüft der Emitter gegen
das 104-VGPR-Ziel. Wenn überschritten: Warnung + GA-Signal "diese
Tile-Config ist zu groß" → die GA verwirft diese Variante.

### 6.8 Peephole-Passes

**NegF32-Elimination (Operand-Modifier).**

Muster:
```
NegF32  { src: R_X, dst: R_NEG }
FmaF32  { a: ..., b: ..., c: R_NEG, dst: R_Y }
```

Wenn `R_NEG` **nur** als Operand von `FmaF32`/`MulF32`/`AddF32` verwendet
wird und der Live-Range von `R_NEG` vollständig innerhalb des einen
Konsumenten liegt, wird die Sequenz gemergt:

```asm
; statt: v_xor_b32 r_neg, 0x80000000, r_x ; v_fma_f32 r_y, r_a, r_b, r_neg
; wird:                                    v_fma_f32 r_y, r_a, r_b, -r_x
```

gfx1201 VALU-Instruktionen haben Operand-Modifier `neg:` und `abs:` an
jedem Operanden. `v_fma_f32` mit negiertem `c` kostet keine zusätzliche
Instruktion.

**MulF32 + AddF32 → FmaF32-Fusion.**

Muster:
```
MulF32 { a: X, b: Y, dst: T }
AddF32 { a: T, b: Z, dst: W }
```

Wenn `T` nur von der `AddF32` konsumiert wird, fusioniert der Peephole
zu `FmaF32 { a: X, b: Y, c: Z, dst: W }`. Achtung: **Die Fusion ändert
das Rundungsverhalten** (siehe §7.4 — muss mit dem CPU-Codegen
synchronisiert sein).

**Const-Inlining.**

`Const { value, dst: C }` gefolgt von einem Konsumenten, der `C` als
immediate akzeptiert (z. B. `MulF32 { a: X, b: C }`), wird zu
`v_mul_f32 dst, lit:value, X`. Die explizite `Const`-Op im IR-Programm
ist selbstdokumentierend (§2.9, STOP-1-Resolution), der Codegen eliminiert
sie kostenneutral.

### 6.9 Kernel-Rahmen (Prolog, Main, Epilog)

**Prolog.** Workgroup-ID und Thread-ID-Berechnung, Block-Pointer-Berechnung
aus Shape-Parametern, LDS-Deklaration mit passender Byte-Größe (siehe
§5.2), Tile-Counter-Initialisierung.

**Main.** Die K-Loop, die pro Iteration:
1. Blocks für die aktuelle K-Chunk lädt (Phase A, ggf. Doppelpufferung),
2. Dequant-Programm pro Block ausführt (Phase B, in LDS schreibt),
3. LDS-Barrier (`s_waitcnt lgkmcnt(0) + s_barrier`),
4. WMMA-Intrinsic aufruft (passend zu PrecisionLevel),
5. Akku-Fragment weiterführt.

**Epilog.** C-Fragment nach Global schreibt (FP32-Output), Dirty-Flag in
`hipHostMallocMapped`-Region setzt (Invariante 5, §5.5), Kernel kehrt
zurück.

---

## 7. Codegen CPU (Zen4 AVX-512)

Dieser Abschnitt spezifiziert den CPU-Emitter für Zen4-basierte Systeme
mit AVX-512-Unterstützung. Zielplattform: Ryzen 9 7945HX (Zen4, AVX-512
Double-Pumped ohne Takt-Reduktion, VNNI).

**Rolle des CPU-Pfads.** Der CPU-Backend ist nicht der primäre
Inference-Pfad — GPU dominiert für 8B+-Modelle. Der CPU-Pfad dient:
- Als Fallback wenn die GPU nicht verfügbar ist (Development, CI).
- Für GA-Läufe in rf-forge (reproduzierbare Messungen).
- Als Teil des VALU-Parity-Checks (GPU-Level-3 ↔ CPU-FP32 Bit-Parity).

### 7.1 Emitter-Architektur

Der CPU-Emitter ist strukturell einfacher als der GPU-Emitter:

```
1. Lowering:       DequantOp-Programm → C-AST mit AVX-512-Intrinsics
2. Peephole:       Analog GPU (NegF32, FmaFusion, Const-Inline)
3. Emission:       C-Code mit <immintrin.h>
4. Compile:        System-Clang/GCC mit -march=znver4 -O3
```

Der Output ist eine AOT-kompilierte `.so`-Datei, die zur Startzeit via
`dlopen` geladen wird. JIT-Kompilierung ist für v1.0 nicht vorgesehen.

### 7.2 Instruktions-Mapping pro DequantOp

**Speicher-Ops:**

| DequantOp | AVX-512-Intrinsic(s) |
|---|---|
| `LoadBytes { count: 1..16 }` (U8) | `_mm_loadu_si128` + `_mm512_cvtepu8_epi32` |
| `LoadBytes { count: 1..16 }` (I8, Q8_0) | `_mm_loadu_si128` + `_mm512_cvtepi8_epi32` |
| `LoadBytes { count: 16..64 }` | Mehrere 128-Bit-Loads, concat via `_mm512_inserti32x4` |
| `LoadBytes { count: 2 }` (FP16) | `_mm_loadu_si128` + `_mm512_cvtph_ps` (F16C) |
| `LoadFP8 { ... }` | **Nicht unterstützt auf CPU** — Emitter wirft Compile-Error |

**Extraktion:**

| DequantOp | AVX-512-Intrinsic |
|---|---|
| `ExtractNibble { high: false }` | `_mm512_and_si512(src, _mm512_set1_epi32(0x0F))` |
| `ExtractNibble { high: true }` | `_mm512_srli_epi32(src, 4)` gefolgt von `_mm512_and_si512(..., 0x0F)` |
| `ExtractBits { shift, mask }` | `_mm512_and_si512(_mm512_srli_epi32(src, shift), _mm512_set1_epi32(mask))` |
| `Combine6Bit { shift }` | Sequenz: `shift qh` + `and 0x03` + `shift left 4` + `or ql_nibble` |
| `CombineBits { hi_shift }` | `_mm512_or_si512(lo, _mm512_slli_epi32(hi, hi_shift))` |

**Konvertierung:**

| DequantOp | AVX-512-Intrinsic |
|---|---|
| `IntToFloat { offset: 0 }` auf `U8` | `_mm512_cvtepi32_ps` (nach Zero-Extension) |
| `IntToFloat { offset: 0 }` auf `I8` | `_mm512_cvtepi32_ps` (nach Sign-Extension) |
| `IntToFloat { offset: n }` | `_mm512_cvtepi32_ps` + `_mm512_add_ps(tmp, _mm512_set1_ps(n as f32))` |

**Arithmetik:**

| DequantOp | AVX-512-Intrinsic | Rounding |
|---|---|---|
| `MulF32` | `_mm512_mul_ps` | RNE |
| `AddF32` | `_mm512_add_ps` | RNE |
| `SubF32` | `_mm512_sub_ps` | RNE |
| `FmaF32` | `_mm512_fmadd_ps` | Single-Rounding RNE |
| `NegF32` | `_mm512_xor_ps(src, _mm512_set1_ps(-0.0f))` **oder Fusion** | Peephole zu FMA-Sign-Flip |
| `Const { value }` | `_mm512_set1_ps(value)` | — |

**Downcast:**

| DequantOp | AVX-512-Intrinsic | Rounding |
|---|---|---|
| `DowncastToFP8 { ... }` | **Nicht unterstützt auf CPU** — Emitter wirft Compile-Error | — |
| `DowncastToHalf { Fp16 }` | `_mm512_cvtps_ph(src, _MM_FROUND_TO_NEAREST_INT \| _MM_FROUND_NO_EXC)` | RNE (explizit) |
| `DowncastToHalf { Bf16 }` | **Software-Emulation** (§7.4) | RNE + NaN |

**Store:**

| DequantOp | AVX-512-Intrinsic |
|---|---|
| `StoreFP8 { ... }` | **Nicht unterstützt auf CPU** |
| `StoreHalf { ... }` | `_mm256_storeu_si256` auf das FP16-Halb-Vektor |

**Spezial:**

| DequantOp | Behandlung |
|---|---|
| `ScaleBlockStart` | Scheduler-Marker, kein Output |

### 7.3 Kein FP8 auf CPU

Zen4 hat **keine FP8-Instruktionen**. Der CPU-Pfad arbeitet durchgehend in
FP32. Die folgenden DequantOps sind auf CPU Compile-Errors:

- `LoadFP8`
- `DowncastToFP8`
- `StoreFP8`

**Konsequenz für die Kernel-Specs.** Der Codegen wählt für CPU-Targets
automatisch PrecisionLevel 1 oder 3 (nicht 0). Die `PrecisionConfig` der
Precision-GA wird beim CPU-Lauf pre-processed: alle Layer mit Level 0
werden auf Level 1 hochgemappt.

**Explizites WARN-Log.** Beim ersten CPU-Load eines Modells mit
GPU-optimiertem `PrecisionConfig` loggt rf-forge pro-Modell eine einzelne
Warnung:

```
WARN: CPU-Fallback aktiv. 30 von 32 Layern wurden von Level 0 (FP8)
      auf Level 1 (FP16) hochgestuft. Perplexity-Auswirkung erwartet
      (typisch <1%). Für optimale CPU-Qualität: rf-forge tune-precision
      --target zen4 neu ausführen.
```

Kein Silent-Fallback, kein Error, kein Abbruch — der Benutzer muss den
Tradeoff bewusst wahrnehmen.

**Warum keine FP8-Emulation auf CPU?** Technisch machbar (SW-Emulation
mit Shifts/Masks), aber sinnlos: CPU-GEMV ist bandbreiten-gebunden bei
~50 GB/s, nicht compute-gebunden. Die FP8-Emulations-Kosten würden den
Bandbreiten-Gewinn aufzehren. Der FP8-Pfad ist GPU-exklusiv.

### 7.4 Rounding-Konsistenz und GPU-CPU-Parity

Der VALU-Parity-Pfad (Säule 6) prüft, dass GPU-Level-3-Kernel und
CPU-FP32-Kernel bit-identische Ergebnisse produzieren. Das ist die
kritischste Korrektheits-Invariante der gesamten Spec — eine Abweichung
macht den Parity-Check wertlos.

Diese Sektion spezifiziert pro DequantOp, welchen Rounding-Mode der
Codegen verwendet, und wo Bit-Identität garantiert ist.

**Tabelle: Rounding-Garantien pro DequantOp:**

| DequantOp | GPU-Rounding | CPU-Rounding | Bit-Identität? |
|---|---|---|---|
| `LoadBytes` + `IntToFloat` (U8/I8) | exakt | exakt | ✅ |
| `MulF32` | RNE (`v_mul_f32`) | RNE (`_mm512_mul_ps`) | ✅ |
| `AddF32` | RNE | RNE | ✅ |
| `SubF32` | RNE | RNE | ✅ |
| `FmaF32` | Single-Rounding RNE | Single-Rounding RNE | ✅ |
| `NegF32` | Bit-Flip im Vorzeichen (exakt) | Bit-Flip (exakt) | ✅ |
| `Const` | Exakte IEEE-754-Konstante | Exakte IEEE-754-Konstante | ✅ |
| `DowncastToHalf { Fp16 }` | RNE (`v_cvt_f16_f32`) | RNE (`_mm512_cvtps_ph` mit `_MM_FROUND_TO_NEAREST_INT`) | ✅ — wenn CPU-Mode explizit |
| `DowncastToHalf { Bf16 }` | 5-Instr SW (RNE + NaN) | SW-Emulation (identisches Algorithmus) | ✅ — **nur mit gemeinsamem Algorithmus** |
| `DowncastToFP8` | HW-Intrinsic | — (auf CPU verboten) | N/A (CPU nicht im Parity-Scope) |

**Regel 1: FMA vs. MUL+ADD dürfen nicht zwischen GPU und CPU divergieren.**

Das ist der häufigste Parity-Fallstrick. Beide haben Single-Rounding-FMA
(bit-identisch), aber wenn der **Peephole-Pass auf einer Seite** das
Muster `MulF32 → AddF32` zu `FmaF32` fusioniert und auf der anderen
Seite nicht, sind die Ergebnisse unterschiedlich:

```
MUL + ADD:
  tmp  = round(a * b)         ; RNE
  dst  = round(tmp + c)       ; RNE — zwei Rundungen (double-rounding)

FMA:
  dst  = round(a * b + c)     ; RNE — eine Rundung
```

Der Unterschied kann bis zu 0.5 ULP sein. Über viele FMAs in einer
GEMM-Akkumulation summiert sich das zu einer Differenz, die den
Parity-Check bricht.

**Enforcement.** Der Peephole-Pass für `MulF32+AddF32 → FmaF32`
(§6.8) muss auf GPU und CPU **identisch** implementiert sein. Das wird
in der Codegen-Architektur durch ein gemeinsames Peephole-Modul
erreicht, das beide Emitter nutzen — nicht zwei separate Implementierungen.

**Enforcement-Assertion.** Beim Build wird ein Test ausgeführt, der ein
triviales Programm (`MulF32 → AddF32` sequenz) auf beiden Targets
emittiert und prüft, dass entweder beide FMA oder beide MUL+ADD
verwenden.

**Regel 2: FP32 → FP16 braucht expliziten RNE-Mode auf CPU.**

`_mm512_cvtps_ph` akzeptiert einen Rounding-Mode-Parameter. Ohne explizite
Angabe könnte der Compiler den MXCSR-Default verwenden, der theoretisch
nicht RNE sein könnte. Der CPU-Emitter verwendet **ausschließlich**
`_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC`, um mit `v_cvt_f16_f32`
(gfx1201 RNE-hardcoded) bit-identisch zu sein.

**Enforcement-Assertion.** Unit-Test: 1000 zufällige FP32-Werte werden
auf beiden Targets zu FP16 konvertiert und byte-weise verglichen. Kein
einziger Wert darf abweichen.

**Regel 3: FP32 → BF16 Bit-Identität nur mit gemeinsamem Algorithmus.**

gfx1201 hat keine HW-Instruktion für BF16-Downcast (§6.6); der Compiler
emittiert eine 5-Instruktionen-SW-Sequenz. Auf Zen4 gibt es
`_mm512_cvtneps_pbh` (AVX-512 BF16-Extension) — **aber** der
Rounding-Mode-Default könnte abweichen, und die NaN-Propagation könnte
subtil anders sein.

**Entscheidung für v1.0.** Beide Emitter verwenden dieselbe
**Software-Sequenz** (kein `_mm512_cvtneps_pbh` auf CPU, keine
HW-Intrinsic wo verfügbar auf GPU — die gfx1201-Sequenz ist ohnehin SW).
Die Algorithmus-Quelle ist:

```c
// Gemeinsamer Algorithmus für FP32 → BF16 (RNE + NaN-Safe)
// Wird auf GPU und CPU identisch emittiert.
static inline uint16_t f32_to_bf16_rne(float f) {
    uint32_t u = bit_cast<uint32_t>(f);

    // NaN-Check
    if ((u & 0x7FFFFFFF) > 0x7F800000) {
        // NaN → Force QNaN
        return 0x7FC0 | (uint16_t)((u >> 16) & 0x007F);
    }

    // Round-to-Nearest-Even: round-bit = bit 15, sticky = any of bits 0..14
    uint32_t lsb = (u >> 16) & 1;                  // LSB after truncation
    uint32_t round_bit = (u >> 15) & 1;
    uint32_t sticky = u & 0x7FFF;
    uint32_t round_up = round_bit & (lsb | (sticky != 0 ? 1 : 0));

    return (uint16_t)((u + (round_up << 16)) >> 16);
}
```

**Bit-Identität garantiert** durch algorithmische Identität — beide Seiten
führen dieselbe Bit-Manipulation durch.

**Regel 4: Denormal-Behandlung muss zwischen GPU und CPU identisch sein.**

Denormals (Subnormals) sind ein häufiger Parity-Breaker und direkt
verbunden mit dem Llama-3.1-Special-Token-Problem (SNR<1 bei kleinen
Werten, Memory #8 Erkenntnis 3). Die Spec verlangt einheitliches
Denormal-Handling:

| Precision-Pfad | GPU (gfx1201) | CPU (Zen4) | Rationale |
|---|---|---|---|
| FP32 (Akku, Level 3) | IEEE (denormals erhalten) | IEEE (MXCSR: FTZ=0, DAZ=0) | Parity-Check muss funktionieren |
| FP16 (Level 1) | IEEE (denormals erhalten) | IEEE | Llama-3.1-Fix: Special-Token-Werte im FP16-Denormal-Bereich dürfen nicht geflusht werden |
| BF16 (Level 2) | SW-Sequenz behandelt Denormals in FP32 | SW-Sequenz (identisch) | BF16 selbst hat kaum Denormal-Problem (weiter Exponent), aber Eingabe-FP32 schon |
| FP8 (Level 0) | `v_cvt_pk_fp8_f32` mit `saturate=true` | (verboten auf CPU) | Unter E4M3-Subnormal-Grenze (2^-9) wird auf 0 geflusht; Quality Monitor sieht das |

**Kernel-Prolog setzt das MODE-Register explizit auf GPU.** Unabhängig
vom `hipcc`-Default wird am Anfang jedes Kernels das
Denormal-Handling erzwungen:

```asm
; RDNA 4 MODE-Register FP_DENORM-Feld (Bits 4..7)
; 0xF = IEEE-konform für FP32 und FP16 Input+Output (denormals erhalten)
s_setreg_imm32_b32 hwreg(HW_REG_MODE, 4, 4), 0xF
```

In HIP-C als Inline-Assembly oder per Compiler-Flag
`-fno-cuda-flush-denormals-to-zero` an der Kernel-Grenze. Der Codegen
emittiert diese Zeile als Teil des Kernel-Prologs jedes generierten
Kernels — keine Annahme an hipcc-Defaults.

**CPU-Seite setzt MXCSR explizit.** Der `rayon`-Worker-Prolog setzt:

```c
_MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_OFF);  // DAZ=0 (input)
_MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_OFF);          // FTZ=0 (output)
```

Das gilt pro Thread und wird einmalig beim Thread-Pool-Start gesetzt.

**Verifikations-Test.** Ein zusätzlicher Parity-Test injiziert einen
Block mit absichtlich kleinen Werten (Magnitude ~1e-6 bis 1e-8, dequant-
Resultate in FP16-Denormal-Bereich), führt Level-1-GPU-Kernel und
CPU-Kernel aus, und prüft bit-identische Ergebnisse. Das ist ein
direkter Regression-Test für den Llama-3.1-Bug.

**Interaktion mit Precision-Level-System.**

- **Level 0 (FP8):** Werte unter 2^-9 (E4M3-Subnormal-Grenze) werden in
  der `v_cvt_pk_fp8_f32`-Intrinsic zu 0 geflusht. Das ist
  format-bedingt, nicht MODE-Register-bedingt, und wird durch
  Eskalation auf Level 1 adressiert (nicht durch MODE-Änderung).
- **Level 1 (FP16):** Denormals erhalten bis 2^-24. Der Llama-3.1-
  Special-Token-Fix funktioniert nur, wenn FTZ=off auf beiden Seiten
  gesetzt ist — Regel 4 garantiert das.
- **Level 3 (FP32 VALU):** Denormals erhalten bis 2^-149. Der
  VALU-Parity-Check vergleicht FP32-Ergebnisse direkt — FTZ-Diskrepanz
  würde ihn bei kleinen Werten brechen.

**Regel 5: Toleranz-basierter Parity-Check als Fallback.** Wenn eine
zukünftige Op-Erweiterung Bit-Identität **nicht** garantieren kann
(z. B. wenn eine Transzendentalfunktion eingeführt würde, wo GPU und CPU
unterschiedliche Approximationen verwenden), wird der Parity-Check für
diese Op auf **Toleranz-basiert** umgestellt:

```rust
pub enum ParityCheckMode {
    /// Bit-identisch (Default für alle v1.0-Ops)
    BitExact,

    /// Toleranz-basiert: max_abs_err < epsilon (für Transzendentale, v1.1+)
    Tolerance { max_abs_err: f32 },
}
```

Für v1.0 ist der Mode **pro Op global auf `BitExact`**. Keine der
17 DequantOps des v1.0-Sets benötigt Toleranz. Wird eine neue Op
aufgenommen, die das nicht garantieren kann, muss sie explizit auf
`Tolerance` gesetzt und im Architektur-Dokument dokumentiert werden.

**Enforcement-Teststrategie.**

1. **Unit-Tests pro Op.** Für jede DequantOp ein Test mit 10 000
   zufälligen Inputs, der GPU- und CPU-Output byte-weise vergleicht.
   Fehlschlag = Blocker für Merge.

2. **End-to-End Parity.** Für jedes Pflicht-Format ein Test mit einem
   echten Block (Golden-Vector-Input), der Level-3-GPU-Kernel und
   CPU-FP32-Kernel emittiert und beide Outputs byte-weise vergleicht.

3. **Statische Code-Prüfung.** Der Peephole-Pass-Modul wird mit einem
   Differentialtest validiert: ein festes DequantOp-Programm wird
   durch beide Emitter geschickt, die erzeugten LIR-Streams werden
   verglichen. Nicht-identische Transformationen → Build-Fehler.

### 7.5 VNNI für Q8-Inline (nur GEMV)

Für den Q8-Inline-Pfad im GEMV-Kernel (§5.3) nutzt der CPU-Emitter die
VNNI-Integer-MAC-Intrinsic:

```
_mm512_dpbusd_epi32(acc, x_i8, weight_i8)
// acc += sum over 4 bytes of (x_i8[i] * weight_i8[i])
// 4-fold packed: 16 int32 accumulators × 4 bytes = 64 bytes per invocation
```

Das ersetzt 4 separate `_mm512_mullo_epi32 + _mm512_add_epi32`-Sequenzen
durch eine einzige Instruktion. VNNI ist auf Zen4 nativ; Double-Pumping
der AVX-512-Einheiten bedeutet effektiv 2× throughput vs. Zen3.

**Nicht auf GPU.** RDNA 4 hat `v_dot4_i32_i8` als Pendant, aber nur für
GEMV (nicht WMMA). Der GPU-Codegen für GEMV nutzt die analoge
Integer-Dot-Product-Instruktion.

### 7.6 CPU-Kernel-Rahmen

**Thread-Pool: rayon (kein OpenMP).** Der CPU-Pfad nutzt `rayon` als
Work-Stealing-Thread-Pool — kein OpenMP, keine C-Dependency. Gründe:

- **Deterministische Thread-Affinität** für stabile GA-Messungen. rayon
  bindet Threads per `rayon::ThreadPoolBuilder::num_threads(n)` +
  `core_affinity::set_for_current(...)` an feste Cores.
- **Keine Scheduling-Varianz** wie bei OpenMP-Guided-Scheduling, die die
  GA-Fitness-Messungen verrauschen würde.
- **Native Rust-Integration** ohne zusätzliche Build-Komplexität.

**Double-Pumping-Hinweis (Zen4 AVX-512).** Zen4 führt AVX-512-Instruktionen
als 2 × 256-Bit-Operationen aus, **ohne Takt-Reduktion** (Memory #1). Eine
`_mm512_fmadd_ps`-Instruktion hat effektiv den Durchsatz zweier
`_mm256_fmadd_ps` — keine thermische Drosselung wie bei Intel-AVX-512.

Das bedeutet: der effektive Compute-Durchsatz ist exakt das, was ein
Micro-Benchmark misst. Es gibt keine "versteckte Strafe" für AVX-512 auf
Zen4. Relevant für die Bandbreiten- und Compute-Berechnungen in §5.3
GEMV auf CPU. rf-forge-GA-Messungen reflektieren automatisch den realen
Durchsatz; **kein Spec-Constraint**, nur ein Hinweis für die
rf-forge-Dokumentation.

**Prolog.** rayon-Scope-Entry, Tensor-Pointer-Setup, AVX-512-Register-
Allocation, **MXCSR-Setup für Denormal-Konsistenz** (siehe §7.4 Regel 4).

**Main.** K-Loop mit Unrolling durch Rust-Compiler (via
`#[inline(always)]` + Optimierungs-Flags; kein eigener Unroller nötig).
Pro Iteration: Block-Load, Dequant-Programm, FMA-Akkumulation.

**Epilog.** Output-Write, Dirty-Flag-Setzen in einer lokalen
Scratchpad-Region (kein `hipHostMallocMapped` auf CPU nötig — der
Host-Inference-Loop liest direkt aus dem Akku-Puffer).

---


---

## 8. Schnittstellen zu anderen Säulen

Die Dequant IR (Säule 3) hat explizite, schmale Schnittstellen zu den
fünf anderen Säulen der v1.0-Architektur und zum rf-forge-Tool. Alle
Schnittstellen sind durch Rust-Structs definiert; es gibt keine impliziten
Seiteneffekte oder Shared-Mutable-State.

### 8.1 Dequant IR ← Model Introspection (Säule 1)

**Richtung.** Introspection liefert das Modell-Profil; die Dequant IR
verbraucht es, um die benötigten Formate zu kompilieren.

**Struct:**

```rust
/// Output von Säule 1 (Model Introspection), Input für Säule 3 (Dequant IR).
pub struct ModelProfile {
    /// Welche QuantFormats im Modell vorkommen.
    /// Für Q4_K_M typischerweise [Q4_K, Q6_K, Q8_0, Fp16].
    pub formats_present: Vec<QuantFormatId>,

    /// Pro Layer: welches QuantFormat das Gewicht nutzt.
    pub per_layer_format: Vec<QuantFormatId>,

    /// Layer mit SNR-Risk < 2.0 (Memory #8 Erkenntnis 3).
    /// Precision-GA startet hier mit Level 1 oder höher.
    pub snr_risk_layers: Vec<LayerIndex>,

    /// Special-Token-Indizes mit L2-Norm < 0.05 (Llama-3.1-Problem).
    pub critical_token_indices: Vec<TokenId>,
}

/// Vom Dequant-IR-Codegen produziert.
pub struct CompiledQuantFormat {
    pub format_id: QuantFormatId,
    pub kernels: Vec<CompiledKernel>,  // eine pro (Shape, Level, TileConfig)
}
```

**Interaktion.** Die Dequant IR registriert nur Formate, die in
`formats_present` stehen — nicht die komplette `Q4_0..Q8_0`-Matrix.
Bei einem reinen Q4_K_M-Modell wird Q4_0 gar nicht kompiliert.

### 8.2 Dequant IR ← Precision-GA (rf-forge)

**Richtung.** rf-forge liefert die per-Layer-Precision-Konfiguration;
die Dequant IR kompiliert die entsprechenden Kernel-Varianten.

**Struct** (aus §4.4, hier vollständig):

```rust
pub struct PrecisionConfig {
    pub per_layer_level: Vec<PrecisionLevel>,

    /// KV-Cache-Format (separate GA-Dimension, Memory #5).
    pub kv_cache_variant: Fp8Variant,

    /// Pareto-Index für diese Konfiguration (GA-Metadata).
    pub pareto_rank: u32,

    /// Gemessene Qualität (KL-Divergenz zu FP32-Baseline) auf
    /// WikiText-2-Validation-Set.
    pub kl_divergence: f32,

    /// Gemessener Throughput (tok/s Decode).
    pub decode_tps: f32,
}

pub enum PrecisionLevel {
    Fp8,    // Level 0
    Fp16,   // Level 1
    Bf16,   // Level 2
    Fp32,   // Level 3 (VALU, nur für Safety/Debug)
}
```

**Invarianten:**

1. `per_layer_level.len() == model.n_layers`.
2. Für CPU-Targets: kein `PrecisionLevel::Fp8` (wird bei Load
   pre-processed, WARN-Log, §7.3).
3. Level 3 darf nur für einzelne Safety-Validierungs-Runs auftauchen;
   die Precision-GA gibt Level 3 nicht als Produktions-Output zurück.

### 8.3 Dequant IR → Computation Graph (Säule 2)

**Richtung.** Die Dequant IR stellt dem Computation Graph die verfügbaren
GEMM/GEMV-Ops mit ihren Format- und Precision-Annotationen zur Verfügung.

**Struct:**

```rust
/// Wird vom Graph-Builder konsultiert, um OpNodes zu annotieren.
pub struct DequantCatalog {
    /// Alle kompilierten Formate (aus CompiledQuantFormat oben).
    pub formats: Vec<CompiledQuantFormat>,

    /// Welche (Format, Level)-Kombinationen einen Kernel im Cache haben.
    pub available: HashMap<(QuantFormatId, PrecisionLevel), Vec<KernelShape>>,
}

/// Graph-Annotation pro GEMM/GEMV-Knoten.
pub struct OpDequantAttrs {
    pub format: QuantFormatId,
    pub level: PrecisionLevel,
    pub shape: KernelShape,
}
```

**Nutzung.** Der Fusion-Pass (Säule 2) prüft bei jeder potentiellen
Fusion, dass die beteiligten Knoten kompatible `OpDequantAttrs` haben
(gleiche Precision, passende Shapes). Fusions-Gewinn wird gegen
Compile-Kosten der neuen Kernel-Variante abgewogen.

### 8.4 Dequant IR → Self-Tuning Runtime (Säule 4)

**Richtung.** Die Dequant IR übergibt die kompilierten Kernel-Varianten
an den Bandit; dieser wählt zur Laufzeit.

**Struct:**

```rust
/// Vom Codegen erzeugt, vom Bandit (Säule 4) zur Runtime konsumiert.
pub struct KernelVariantSet {
    pub base_key: KernelCacheKey,  // (Format, Shape, Level)

    /// Typischerweise 3-5 Top-Varianten aus der Kernel-Tuning-GA.
    pub variants: Vec<KernelVariant>,
}

pub struct KernelVariant {
    pub binary: Arc<CompiledKernel>,     // shared, lazy-loaded
    pub tile_config: TileConfig,
    pub gemv_variant: Option<GemvVariant>,

    /// Vom Kernel-Tuning-GA gemessener Throughput (Referenz für Bandit).
    pub gmeasured_tps: f32,
}
```

**Bandit-Logik.** Der UCB1-Bandit startet mit uniformer Prior; nach 100
Calls konvergiert er typisch zur tatsächlich schnellsten Variante. Die
Dequant IR ist an dieser Stelle passiv — sie liefert nur das Set.

### 8.5 Dequant IR ← Quality Monitor (Säule 5)

**Richtung.** Der Quality Monitor meldet Drift; die Dequant IR eskaliert
den betroffenen Layer durch Auswahl eines anderen vorkompilierten
Kernels.

**Struct:**

```rust
pub struct DriftEvent {
    pub layer_idx: LayerIndex,
    pub reason: DriftReason,
    pub current_level: PrecisionLevel,
    pub recommended_level: PrecisionLevel,
}

pub enum DriftReason {
    /// FP8-Downcast hat gesättigt → Eskalation FP8 → FP16.
    Fp8SaturationExceeded,

    /// Hidden-State-Magnitude außerhalb erwarteten Bereichs.
    HiddenStateDrift { observed: f32, expected: (f32, f32) },

    /// Token-Output-Entropie unter Schwelle (degenerate sampling).
    OutputEntropyCollapse,
}
```

**Eskalations-Regel** (§4.5): Level 0 → 1, Level 1 → 3 (VALU),
Level 2 → 3. BF16 ist kein Auto-Eskalations-Ziel.

### 8.6 Dequant IR ↔ Safety & Debug (Säule 6)

**Richtung (bidirektional).** Säule 6 fordert Level-3-Referenz-Kernel
an; die Dequant IR liefert sie. Säule 6 verifiziert Parity und meldet
Violations zurück.

**Struct:**

```rust
/// Von Säule 6 an Säule 3: Parity-Check anfordern.
pub struct ParityRequest {
    pub kernel_key: KernelCacheKey,
    pub inputs: KernelInputs,

    /// Wie streng wird geprüft (siehe §7.4 Regel 5).
    pub mode: ParityCheckMode,
}

pub enum ParityCheckMode {
    BitExact,
    Tolerance { max_abs_err: f32 },
}

/// Von Säule 3 an Säule 6: Antwort mit Referenz-Output.
pub struct ParityResponse {
    pub wmma_output: Vec<f32>,
    pub valu_reference: Vec<f32>,      // Level 3 FP32 VALU
    pub max_abs_err: f32,
    pub bit_exact: bool,
}
```

**Sampling-Raten:**
- Produktion: 1/1000 (eine aus tausend Kernel-Aufrufen wird geprüft).
- GA-Validierung: 1/1 (jede GA-produzierte Kernel-Variante wird geprüft,
  bevor sie in den Cache kommt).

### 8.7 Gesamtfluss: Cold Start bis erster Token

```
1. GGUF laden
2. Säule 1 erzeugt ModelProfile
3. Säule 3 (Dequant IR) registriert benötigte QuantFormats
4. Säule 2 (Graph) wird gebaut mit OpDequantAttrs
5. rf-forge Precision-GA läuft (~15 min) → PrecisionConfig
6. rf-forge Kernel-Tuning-GA läuft (~2 h für 5 Shapes × 4 Levels)
   → KernelVariantSet pro Shape (Top-5)
   → JEDE Variante durchläuft Parity-Check gegen Level-3-VALU (Säule 6)
7. Alle KernelVariantSets landen im Cache
8. Runtime-Inferenz:
   - Bandit (Säule 4) wählt pro Call die Variante
   - Quality Monitor (Säule 5) beobachtet Hidden States
   - Bei DriftEvent: Layer auf höheres Level eskalieren
   - Produktion: 1/1000 Calls triggern Parity-Check
```

**Warm-Start** (zweiter Modell-Load nach gecachetem Tuning): ~6 Sekunden
(nur GGUF-Load + Cache-Lookup + Kernel-Module-Bind).

---

## 9. Walk-Through — Q5_K vom leeren Blatt

Dieser Abschnitt demonstriert das v1.0-Entwickler-Erlebnis an einem
konkreten Beispiel: ein neues Format, Q5_K, wird **live** in die Dequant
IR aufgenommen. Das Format wurde bewusst aus §3 ausgelassen, um hier
die Vollständigkeit der DequantOp-Ops zu beweisen: alle Ops, die für
Q5_K nötig sind, sind bereits im v1.0-Set (§2.10).

Das Ziel: **30 Minuten, 0 neue Kernel-Dateien**.

### 9.1 Schritt 1 — llama.cpp-Referenz lesen (5 Min)

Q5_K ist in `ggml-quants.c` / `ggml-common.h` definiert:

```c
#define QK_K 256

typedef struct {
    ggml_half d;                  // super-block scale
    ggml_half dmin;               // super-block scale for quantized mins
    uint8_t scales[12];           // scales and mins, quantized with 6 bits
    uint8_t qh[QK_K/8];           // quants, high bit
    uint8_t qs[QK_K/2];           // quants, low 4 bits
} block_q5_K;                     // 176 bytes
```

Die Dequant-Formel (Pseudo-Code aus `dequantize_row_q5_K`):

```
for j in 0..8:
    scale_j, min_j = get_scale_min_k4(scales, j)   // wie Q4_K
    for i in 0..32:
        qh_bit = (qh[e/8] >> (e % 8)) & 1
        q5 = (qs_nibble) | (qh_bit << 4)           // 5-bit value 0..31
        value[e] = d * scale_j * q5 - dmin * min_j
```

**Beobachtungen:**
- Scale-Entpackung identisch zu Q4_K.
- `qh` liefert **1 Bit pro Element** (32 Bytes für 256 Bits = 256 Elemente).
- Keine Zentrierung (`q5` wird nicht `(q5 - 16)` gerechnet — wie Q4_K hat
  Q5_K den `dmin`-Term stattdessen).
- Nibble-Interleaving analog zu Q4_K (paarweise Sub-Blocks).

### 9.2 Schritt 2 — Block-Layout dokumentieren (10 Min)

| Offset | Bytes | Feld | Beschreibung |
|---|---|---|---|
| 0–1 | 2 | `d` | FP16 Block-Scale |
| 2–3 | 2 | `dmin` | FP16 Block-Min |
| 4–15 | 12 | `scales[12]` | 6-Bit packed (identisch Q4_K) |
| 16–47 | 32 | `qh[32]` | 1 Bit pro Element (5. Bit) |
| 48–175 | 128 | `qs[128]` | Lower 4 bits, paarweises Interleaving |

**Block-Bytes:** 2+2+12+32+128 = **176 Bytes** ✓

**Dequant-Formel:** `value = d * scale_j * q5 - dmin * min_j`, wobei
`q5 = qs_nibble | (qh_bit << 4)` im Bereich 0..31.

### 9.3 Schritt 3 — DequantOp-Programm schreiben (10 Min)

Das Programm ist eine Kombination aus:
- Q4_K-Scale-Entpackung (`q4_k_unpack_scales`, 1:1 wiederverwendet)
- Q4_K-Nibble-Logik (paarweises Interleaving)
- Q6_K-artige qh-Bit-Extraktion (nur 1 Bit statt 2)
- Q4_K-Formel (mit `-dmin * min_j`-Term via `NegF32` + `FmaF32`)

**QuantFormat-Definition:**

```rust
pub const Q5_K: QuantFormat = QuantFormat {
    id: 13,                         // llama.cpp GGML_TYPE_Q5_K
    name: "Q5_K",
    block_bytes: 176,
    elements_per_block: 256,
    sub_blocks_per_block: 8,
    sub_block_size: 32,

    block_scale_offset: 0,
    block_scale_type: ScalarType::Fp16,

    block_min_offset: Some(2),
    block_min_type: Some(ScalarType::Fp16),

    sub_scales_layout: SubScalesLayout::Packed6Bit {
        offset: 4,
        count: 12,
        unpack_program: q4_k_unpack_scales(),  // IDENTISCH zu Q4_K, 1:1 reused
    },

    dequant_program: vec![
        // Phase A — qh und qs laden
        DequantOp::LoadBytes { offset: 16, count: 32,  reg: R_QH },
        DequantOp::LoadBytes { offset: 48, count: 128, reg: R_QS },

        // Phase B — Element-Loop (Codegen rollt 8 Sub-Blöcke × 32 Elem aus)

        DequantOp::ScaleBlockStart { sub_block_idx: R_SUB_J },

        // ql-Nibble extrahieren (paarweises Interleaving, wie Q4_K)
        DequantOp::ExtractNibble {
            src: R_QS_ELEM, high: R_HIGH_FLAG_QS, dst: R_QS_N,
        },

        // qh-Bit extrahieren: 1 Bit pro Element
        //   byte_idx = e / 8, bit_in_byte = e % 8
        //   qh_bit = (qh[byte_idx] >> bit_in_byte) & 0x1
        DequantOp::ExtractBits {
            src: R_QH_ELEM,
            shift: R_QH_SHIFT,       // 0..7 je Element
            mask: 0x01,
            dst: R_QH_BIT,
        },

        // q5 = qs_nibble | (qh_bit << 4)
        DequantOp::CombineBits {
            lo: R_QS_N, hi: R_QH_BIT, hi_shift: 4,
            dst: R_Q5_INT,
        },

        DequantOp::IntToFloat { src: R_Q5_INT, offset: 0, dst: R_Q_FP },

        // d_eff = d * scale[j]
        DequantOp::MulF32 { a: R_D, b: R_SCALE_J, dst: R_D_EFF },

        // dmin_eff = dmin * min[j]; neg_dmin_eff = -dmin_eff
        DequantOp::MulF32 { a: R_DMIN, b: R_MIN_J, dst: R_DMIN_EFF },
        DequantOp::NegF32 { src: R_DMIN_EFF, dst: R_NEG_DMIN_EFF },

        // value = d_eff * q5 + (-dmin_eff)
        DequantOp::FmaF32 {
            a: R_D_EFF, b: R_Q_FP, c: R_NEG_DMIN_EFF, dst: R_VAL,
        },

        DequantOp::DowncastToFP8 {
            src: R_VAL, dst: R_OUT,
            variant: Fp8Variant::E4M3, saturate: true,
        },
        DequantOp::StoreFP8 {
            src: R_OUT,
            variant: Fp8Variant::E4M3,
            lds_offset_expr: "sub_j * 32 + e * TILE_N + col".into(),
        },
    ],
};
```

**Beobachtungen:**
- **17 Zeilen DequantOp-Programm** für ein neues Format.
- **0 neue DequantOp-Varianten** nötig — alle 17 v1.0-Ops reichen aus.
- `q4_k_unpack_scales` wird 1:1 wiederverwendet (identische 6-Bit-Packung).
- Der Codegen synthetisiert `R_QH_SHIFT = e % 8`, `R_HIGH_FLAG_QS` aus dem
  Q4_K-paarweise-Interleaving-Muster, `R_QS_ELEM` aus `pair_base`-Formel —
  alles identisch zu §6.3/§6.4.

### 9.4 Schritt 4 — Golden Vectors generieren (5 Min)

Python-Script `generate_q5_k_golden.py`:

```python
import ctypes
import numpy as np
import struct

# llama.cpp als Shared-Lib laden
libllama = ctypes.CDLL("./libllama.so")

# 3 Blöcke aus einem echten Q5_K-GGUF sampeln
with open("tests/golden_vectors/q5_k_blocks.bin", "rb") as f:
    blocks = f.read(3 * 176)

# Referenz-Dequant via llama.cpp
output_fp32 = np.zeros(3 * 256, dtype=np.float32)
libllama.dequantize_row_q5_K(
    blocks, output_fp32.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
    3 * 256
)

# Als Rust-const-Array speichern
with open("tests/golden_vectors/q5_k_expected.rs", "w") as out:
    out.write("pub const Q5_K_EXPECTED: [f32; 768] = [\n")
    for v in output_fp32:
        out.write(f"    {v:.10e},\n")
    out.write("];\n")
```

Resultat: `Q5_K_EXPECTED` als Rust-const-Array, `q5_k_blocks.bin` als
Binary-Test-Fixture.

### 9.5 Schritt 5 — Tests laufen lassen (<1 Min)

```bash
$ cargo test q5_k
   Compiling rocmforge v1.0.0-dev
    Finished test [optimized] target(s)
     Running unittests src/lib.rs
running 4 tests
test dequant_ir::formats::q5_k::tests::block_layout ... ok
test dequant_ir::formats::q5_k::tests::golden_vectors ... ok
test dequant_ir::codegen::tests::q5_k_gpu_emits ... ok
test dequant_ir::codegen::tests::q5_k_cpu_emits ... ok

test result: ok. 4 passed; 0 failed
```

Der Codegen erzeugt beim nächsten `cargo build` automatisch GEMM- und
GEMV-Kernel für Q5_K in allen 4 Precision-Levels. Keine neuen
`.hip`-Dateien, keine Template-Instantiierungen, keine
Kernel-Cache-Einträge — die kommen erst beim ersten Modell-Load per
`rf-forge tune-all`.

### 9.6 Schritt 6 — rf-forge (optional, Background, 8 Min pro Shape)

```bash
$ rf-forge tune-kernels --format q5_k --model ~/models/my-q5km.gguf
[00:00:03] Loading model, detected 5 distinct shapes
[00:00:15] Running Kernel-Tuning-GA for shape (4096, 4096, 4096)...
[00:08:23] Top-5 variants selected, validating against Level-3-VALU...
[00:08:45] All 5 variants passed parity check
[00:08:45] Running for shape (4096, 4096, 14336)...
...
[00:41:30] All shapes tuned. Cache: 15 binaries, 7.5 MB.
```

### 9.7 Aufwands-Vergleich v1.0 vs. v0.x

| Aufgabe | v0.x | v1.0 |
|---|---|---|
| Q5_K-Support | 5 Tage | **30 Minuten** |
| Neue Kernel-Dateien | 5 (WMMA, GEMV, 3 fused) | **0** |
| Neue DequantOp-Varianten | N/A | **0** (alle 17 reichten) |
| Code-Zeilen | ~1 000 | ~40 |
| Neue Bugs (v0.x-Durchschnitt) | 2-3 | 0 erwartet (IR validiert Parity) |
| FP8-Pfad separat? | Ja, ~200 LOC pro Format | **Automatisch** |
| GEMV-Varianten (Residual, Norm, Q8-Inline) | Manuell pro Format | **Automatisch** |

**Die Grundthese der v1.0-Architektur** (§1.2: Aufwand additiv statt
multiplikativ) ist am Q5_K-Beispiel verifiziert.

### 9.8 Was ist passiert? Technische Analyse

Der Schlüssel ist, dass Q5_K **keine konzeptionell neuen Konstrukte**
eingeführt hat — nur neue Offsets, eine neue Bit-Extraktion (1 Bit statt
2), und die Wiederverwendung bekannter Patterns:

| Q5_K-Bestandteil | Wiederverwendet aus |
|---|---|
| Scale-Entpackung | Q4_K (`q4_k_unpack_scales`, identisch) |
| Nibble-Interleaving | Q4_K (paarweise Sub-Blöcke) |
| qh-Bit-Extraktion | Q6_K-Pattern mit `mask=0x01` statt `0x03` |
| Dequant-Formel | Q4_K (`a*b+c` mit `NegF32`) |
| FP8-Downcast, Store | alle Formate (Level-0-Terminal) |
| Codegen-Synthese (R_QH_SHIFT etc.) | generische Element-Loop-Regeln (§6.3) |

Die IR abstrahiert die **Orthogonalität der Komponenten**: Scale-Layout,
Bit-Layout, Formel-Struktur, Downcast-Target sind unabhängige Dimensionen.
v0.x hat sie in handgeschriebenen Kerneln vermischt; v1.0 trennt sie
strukturell.

---

## 10. Fehlerbehandlung und Edge-Cases

Dieser Abschnitt spezifiziert das Verhalten an Grenzfällen und bei
ungültigen Inputs. Alle hier aufgezählten Fälle sind durch Unit-Tests
abgedeckt.

### 10.1 Unbekanntes QuantFormat

**Szenario.** Ein GGUF-File nutzt ein Quant-Format, das nicht in der
`all_formats()`-Registry vorhanden ist (z. B. ein zukünftiges
Community-Format wie "IQ4_XS").

**Behandlung.** Die Dequant IR gibt beim Modell-Load einen strukturierten
Fehler zurück:

```rust
pub enum DequantError {
    UnknownFormat {
        format_id: u32,
        gguf_type_name: String,
        supported: Vec<&'static str>,
    },
    // ... weitere Varianten
}

// Beispiel-Fehlermeldung:
// Error: Unknown quantization format (ggml_type=22, "IQ4_XS")
// Supported formats: Q4_0, Q4_1, Q4_K, Q5_K, Q6_K, Q8_0, FP16, FP32
// Add the format as a QuantFormat definition in src/dequant_ir/formats/
// and re-run. See docs/v1.0/dequant_ir_spec.md §9 for a 30-min walkthrough.
```

Kein Silent-Failure, kein Crash — der Nutzer bekommt einen klaren
Pfad zur Lösung.

### 10.2 K nicht durch elements_per_block teilbar

**Szenario.** Die Kontraktions-Dimension `K` ist nicht ein Vielfaches
der Format-Block-Größe. Bei den Standard-Transformer-Modellen
(`K = 4096, 5120, 8192, 14336`) passiert das nicht mit den Pflicht-
Formaten (alle teilen 256). Zukünftige Modelle könnten aber abweichen.

**Behandlung.** Padding-Strategie im Kernel:

```
K_padded = round_up(K, elements_per_block)
```

Der letzte Block wird mit **Null-Nibbles und Null-Scales** gepolstert, so
dass die dequantisierten Werte 0.0 sind und den Akku nicht beeinflussen.
Das Padding erfolgt beim Modell-Load (einmalig), nicht pro Kernel-Call.

**Erforderlicher Speicher-Overhead:** maximal `elements_per_block - 1`
zusätzliche Elemente pro Tensor-Zeile. Für `elements_per_block = 256`:
~10 KB extra VRAM für ein 32-Layer-Modell, vernachlässigbar.

### 10.3 FP8-Overflow (Wert > 448 bei E4M3)

**Szenario.** Ein dequantisierter Wert überschreitet den E4M3-Maximalwert
von ±448. Passiert typischerweise nicht bei gut trainierten Modellen,
aber möglich bei extremen Outlier-Gewichten.

**Behandlung (Level 0).** `DowncastToFP8 { saturate: true }` nutzt die
Hardware-Saturation der `v_cvt_pk_fp8_f32`-Intrinsic: Werte werden auf
`±448` geklemmt. Ein `Fp8SaturationExceeded`-Event wird an den Quality
Monitor gesendet, der den Layer gegebenenfalls auf Level 1 eskaliert.

**Frequenz.** In v0.x-Messungen: weniger als 1 Saturation pro 10⁶
Werten bei Qwen3/Llama-3.1. Unterhalb dieser Schwelle ignoriert der
Quality Monitor das Event.

### 10.4 FP8-Underflow (Wert < 2^-9 bei E4M3)

**Szenario.** Sehr kleine Werte (unter dem E4M3-Subnormal-Minimum) werden
in der FP8-Konvertierung zu 0 geflusht.

**Behandlung.** Der Flush ist format-bedingt, nicht vermeidbar durch
MODE-Register-Einstellungen. Die Eskalations-Logik:

```
FP8-Underflow-Häufigkeit > 1% der Werte eines Layers
  → Quality Monitor meldet HiddenStateDrift
  → Precision-GA eskaliert Layer auf Level 1 (FP16)
  → FP16 reicht bis 2^-24 — typischerweise ausreichend
```

**Direkter Bezug zum Llama-3.1-Bug** (Memory #8 Erkenntnis 3): Special-
Token-Embeddings haben L2=0.034, was in kombinierten Dequant-Operationen
zu Werten im 1e-5..1e-6-Bereich führt. FP8-Flush ist hier fatal; Level 1
(FP16) mit **korrektem Denormal-Handling** (§10.5) löst das Problem.

### 10.5 Denormal-Verhalten auf gfx1201

Dieser Unterabschnitt adressiert die vom Zusatz-Hinweis des Users zu
STOP 4 aufgeworfenen Fragen zum Denormal-Handling. Denormals sind ein
zentraler Korrektheits-Faktor für das Level-1-Upgrade-Pfad.

#### 10.5.1 Default-Verhalten auf gfx1201

**Das Problem mit Defaults.** Die exakten `hipcc`- und
ROCm-Kernel-Launch-Defaults für das MODE-Register-Feld `FP_DENORM`
variieren zwischen ROCm-Versionen und Compiler-Flags:

- Historisch: `hipcc` setzte auf RDNA 2/3 standardmäßig FP32 auf IEEE
  (denormals erhalten), FP16 auf FTZ (flush für Performance).
- Aktuelle ROCm-Builds: Verhalten kann per
  `-fdenormal-fp-math=ieee|preserve-sign|positive-zero` überschrieben
  werden, aber das ist nicht in allen Build-Konfigurationen gesetzt.
- WMMA-Pfad vs. VALU-Pfad: WMMA-Intrinsics nutzen einen anderen Teil des
  MODE-Registers als skalare VALU-Operationen — potentielle
  Inkonsistenz-Quelle.

**Konsequenz für die Spec.** Statt sich auf Defaults zu verlassen,
**setzt der Codegen das MODE-Register am Anfang jedes Kernels explizit**
(§7.4 Regel 4). Das macht die Spec robust gegenüber Default-Änderungen
in zukünftigen ROCm-Versionen.

#### 10.5.2 Was die Spec verlangt

**FP32:** Denormals erhalten (IEEE-konform) — sowohl VALU- als auch
WMMA-Pfad. Rechtfertigung:

- **Parity mit CPU:** CPU setzt MXCSR auf `DAZ=0, FTZ=0`; GPU muss
  dieselbe Semantik haben für den VALU-Parity-Pfad (Säule 6).
- **Level-3-Referenz:** der VALU-Kernel auf Level 3 dient als Bit-Referenz;
  Denormal-Diskrepanz würde den Parity-Check bei jedem kleinen Wert
  brechen.
- **Kosten:** Hardware-FP32-Denormal-Handling auf RDNA 4 hat keinen
  signifikanten Performance-Overhead (anders als auf älteren AMD/Intel).

**FP16:** Denormals erhalten. Rechtfertigung:

- **Llama-3.1-Fix:** Special-Token-Werte können im FP16-Denormal-Bereich
  (2^-24 bis 2^-14) landen. FTZ=on würde diese Werte zu 0 flushen und
  das Signal zerstören — der Level-0→Level-1-Eskalationspfad wäre
  wirkungslos.
- **Parity-Invariante:** CPU-Seite setzt ebenfalls
  `_MM_FLUSH_ZERO_OFF` für den FP16-Konvertierungspfad.

**BF16:** Denormals erhalten (trivial — die 5-Instr-SW-Sequenz behandelt
alle Fälle in FP32-Space, bevor der Downcast passiert).

**FP8:** format-bedingter Underflow-Flush (§10.4), keine
MODE-Register-Relevanz.

#### 10.5.3 Precision-Level-Interaktion

| Level | Format | Denormal-Range | Behandlung |
|---|---|---|---|
| 0 | FP8-E4M3 | min normal 2^-6, min subnormal 2^-9 | Werte < 2^-9 → 0 (format-bedingt, Quality Monitor sieht Event) |
| 1 | FP16 | min normal 2^-14, min denormal 2^-24 | Denormals erhalten (MODE-Register IEEE) |
| 2 | BF16 | min normal 2^-126 (wie FP32) | kein Denormal-Issue (weiter Exponent) |
| 3 | FP32 | min normal 2^-126, min denormal 2^-149 | Denormals erhalten (Parity-Anforderung) |

**Interpretation.** Level 0 hat einen engen Range — das ist der
eigentliche Grund für die Level-0→Level-1-Eskalation. Level 1 rettet
alles im FP16-Denormal-Bereich. Level 3 behandelt alles.

#### 10.5.4 Konsequenz für die Parity-Tabelle (§7.4)

Die Parity-Tabelle in §7.4 wurde um Regel 4 erweitert. Der
Verifikations-Test injiziert konkret Werte im FP16-Denormal-Bereich,
um sicherzustellen, dass **beide Seiten** dieselben Resultate produzieren
— das ist ein direkter Regression-Test für das Llama-3.1-Szenario.

### 10.6 Fehlende Instruktionen (Codegen-Fallback)

**Szenario.** Eine spezifische Intrinsic (z. B. `v_bfe_u32`) ist in einer
bestimmten ROCm-Version nicht verfügbar oder wurde umbenannt.

**Behandlung.** Der Codegen hat pro DequantOp einen **Software-Fallback-Pfad**:

- `v_bfe_u32 dst, src, shift, width` → fallback:
  `v_lshrrev_b32 tmp, shift, src; v_and_b32 dst, tmp, (1<<width)-1`

Bei Aktivierung des Fallbacks wird eine Warnung geloggt:

```
WARN: Intrinsic v_bfe_u32 not available, using 2-instruction fallback.
      Performance impact: ~5% on ExtractBits-heavy kernels.
      Check ROCm version (detected: 7.2) against docs/compatibility.md
```

Für alle v1.0-Pflicht-Intrinsics (WMMA-FP8, Pair-FP8-Konvertierung) gibt
es **keinen Fallback** — diese sind harte Anforderungen. Fehlt eine,
terminiert der Codegen mit einem strukturierten Error.

### 10.7 Null-Block-Handling

**Szenario.** Ein Block hat `d = 0.0` (alle Werte 0). Passiert gelegentlich
bei stark gepruneten Modellen.

**Behandlung.** Der Dequant-Kernel produziert korrekt 0.0 für alle 256
Elemente (die Arithmetik läuft normal). Keine Sonderbehandlung nötig.
`DowncastToFP8(0.0)` ergibt bit-exakt `+0.0`-FP8 (Alle-Bits-Null).

**Optimization-Opportunity (v1.1+).** Der Model-Introspection-Pass
(Säule 1) könnte Null-Blöcke identifizieren und einen Sparse-Skip-Pfad
aktivieren. Nicht im v1.0-Scope.

### 10.8 Inkonsistente Block-Pointer-Alignment

**Szenario.** Der Block-Pointer ist nicht auf Byte-Grenzen des
`block_bytes` aligned. Das wird vom Kernel-Prolog per Assertion geprüft:

```rust
debug_assert!(block_ptr as usize % 16 == 0,
    "Block pointer must be 16-byte aligned for vectorized loads");
```

Die VRAM-Arena (§3.6 Arch-Dokument) garantiert 256-Byte-Alignment für
alle Tensor-Pointer, das ist weit mehr als benötigt. In Unit-Tests mit
manuell allokierten Buffern muss das Alignment explizit sichergestellt
werden (`align_of_val`-Check, oder `aligned_vec`-Crate).

### 10.9 Workgroup-Size-Mismatch

**Szenario.** Der Kernel wurde mit `tile_config.num_waves = 4` kompiliert,
aber mit einer Launch-Config gestartet, die eine andere Workgroup-Größe
hat.

**Behandlung.** Dies ist **nicht zur Laufzeit zu prüfen** — die
Launch-Config wird vom Runtime-Dispatcher (Säule 4) aus dem
`KernelCacheKey` abgeleitet; ein Mismatch wäre ein interner Bug.
Unit-Tests validieren die Launch-Config-Berechnung; zusätzlich setzt
der Kernel bei ungültiger Workgroup-Size das Dirty-Flag auf einen
Error-Wert, der vom Host-Loop erkannt wird.

### 10.10 Zusammenfassung: Fehler-Klassen

| Klasse | Behandlung | Beispiel |
|---|---|---|
| Nutzer-Fehler | Strukturierter Error mit Lösungs-Hinweis | Unbekanntes Format |
| Format-bedingter Verlust | Silent + Quality-Monitor-Event | FP8-Underflow |
| Numerik-Drift | Eskalation auf nächstes Level | Hidden-State-Drift |
| Codegen-Fallback | WARN + Performance-Hinweis | Fehlende Intrinsic |
| Interner Bug | Assertion + Abort | Alignment-Violation |
| Config-Mismatch | Error-Flag im Dirty-Buffer | Workgroup-Size falsch |

Die Trennung nach Fehler-Klassen ist bewusst: Nutzer-Fehler sind
produktiv (Hinweis gibt Lösungsweg), Drift wird automatisch eskaliert
(keine Nutzer-Intervention nötig), Bugs terminieren laut.

---

---

## Anhang A — Design Decisions Log

Dieser Anhang dokumentiert die während der Spec-Entwicklung getroffenen
Entscheidungen an fünf Review-Punkten (STOPs). Jede Entscheidung ist in
ihrer Kurzform dokumentiert, die ausführliche Rationale steht im
entsprechenden Hauptabschnitt.

### A.1 STOP 1 Resolution — DequantOp-Spezifikation (§2)

| Punkt | Entscheidung | Konsequenz |
|---|---|---|
| 1. Scale-Entpackung (Ops vs. `unpack_program`) | Beide behalten — `ExtractBits`/`CombineBits` als erstklassige Ops, `unpack_program` nutzt sie intern | §3.6 Q4_K nutzt das Pattern |
| 2. `NegF32` Op vs. Operand-Modifier | Op behalten, Codegen mappt auf Operand-Modifier | §6.8 Peephole-Pass |
| 3. `ScaleBlockStart` explizit | Explizit als Marker-Op | §3.6/§3.7 markieren Sub-Block-Wechsel |
| 4. `Const` streichen | Beibehalten — selbstdokumentierend, null Overhead | §3.4 `Const(8.0)`, §3.7 `Const(32.0)` |
| 5. `HalfType`-Reduktion | Auf `{Fp16, Bf16}` reduziert; FP8 nur über `DowncastToFP8` | §2.1 aktualisiert |

### A.2 STOP 2 Resolution — QuantFormat-Definitionen (§3)

| Punkt | Entscheidung | Konsequenz |
|---|---|---|
| 1. Element-Loop-Synthese | Konvention behalten, kein `ElementLoop` in IR | §6.3 Codegen-Details |
| 2. Q4_K Byte-Adressierung | Codegen-Rezept in §6.4 | Keine IR-Änderung |
| 3. Q8_0 `I8` vs. `U8` | Kein `interpret_as` in v1.0 — format-spezifisch im Codegen | §3.8 Kommentar im Programm |
| 4. `FmaSubF32`-Op | Kein `FmaSubF32` — `NegF32` + `FmaF32` mit Peephole | §6.8 Operand-Modifier-Merge |

### A.3 STOP 3 Resolution — Precision-Level + Kernel-Specs (§4+§5)

| Punkt | Entscheidung | Konsequenz |
|---|---|---|
| 1. Cache-Größe | Nur tatsächlich genutzte Shapes (~15 Binaries, ~10 MB); globaler Invalidierungs-Hash statt Pro-Key-`rocm_version` | §4.7 + §5.6 |
| 2. `fuse_norm` Precondition | Als Invariante 6 in §5.5: Compile-Time-Guarantee durch Fusion-Pass | §5.5 |
| 3. `q8_inline` + Level 3 | Als Invariante 7 in §5.5: `q8_inline` nur bei Level 0/1/2 | §5.5 |
| 4. LDS-Strategy-Einschränkung | Keine Einschränkung — GA darf alle 3 explorieren (Memory #8 Erkenntnis 4) | Keine Änderung |

### A.4 STOP 4 Resolution — Codegen GPU + CPU (§6+§7)

| Punkt | Entscheidung | Konsequenz |
|---|---|---|
| 1. FP8-Pair-Packing | Sequential als Default, GA darf interleaved explorieren | §6.5 |
| 2. CPU Level-0→Level-1 | Explizites WARN-Log, kein silent/error/abort | §7.3 |
| 3. Double-Pumping-Doku | Nur Hinweis in §7.6, kein Spec-Constraint | §7.6 |
| 4. Thread-Pool | rayon + deterministische Affinität, kein OpenMP | §7.6 |
| 5. Denormal-Handling (Zusatz) | Regel 4 in §7.4: MODE-Register + MXCSR explizit, Llama-3.1-Fix-Regression-Test | §7.4 + §10.5 |

### A.5 STOP 5 Resolution — Finale Abnahme (§8+§9+§10)

Keine offenen Punkte. Finale Abnahme erteilt. Sync-Liste für das
Architektur-Dokument siehe Anhang B.

### A.6 Kanonische Entscheidungen (über alle STOPs)

1. **Nomenklatur folgt `architecture_v1.2.0-draft.md` §2.4** (Memory #12):
   10 Ops 1:1 übernommen, 7 als `[SPEC-ERWEITERUNG]` markiert.
2. **SSA-Eigenschaft** auf RegIds als Invariante.
3. **Arithmetik immer FP32**, Downcast ist terminal.
4. **FP8-E4M3 als Default** (Level 0), FP16 als erste Eskalation,
   FP32/VALU als Safety. BF16 nur auf expliziten GA-Wunsch.
5. **Q4_K vs. Q4_K_M** klar getrennt: IR spezifiziert Block-Formate,
   nicht Mix-Strategien.
6. **Gemeinsames Peephole-Modul** für GPU und CPU (Parity-Garantie).
7. **Gemeinsamer BF16-SW-Algorithmus** auf beiden Targets
   (`_mm512_cvtneps_pbh` auf CPU nicht verwendet).
8. **Denormal-Handling explizit via MODE-Register / MXCSR** —
   Llama-3.1-Fix.
9. **Kernel-Cache bedarfsgetrieben** (~10 MB, nicht 4 GB), mit
   globalem Invalidierungs-Hash statt Pro-Key-ROCm-Version.
10. **Parity-Check BitExact** für alle v1.0-Ops; Toleranz-Fallback
    existiert als v1.1+-Mechanismus.

---

## Anhang B — Sync-Liste für `architecture_v1.2.0-draft.md`

Die folgenden sieben Änderungen müssen im Architektur-Dokument
nachgezogen werden, um Konsistenz mit dieser Spec herzustellen. Jedes
Item enthält die genaue Quelle (STOP + Punkt) und ist als eigener Arch-
Doc-Commit ausführbar.

### Sync-Update 1: §2.4 — `HalfType`-Enum auf 2 Varianten reduzieren

**Alt:**
```rust
pub enum HalfType { Fp16, Bf16, Fp8E4M3, Fp8E5M2 }
```

**Neu:**
```rust
pub enum HalfType { Fp16, Bf16 }
```

**Rationale:** FP8-Downcast läuft ausschließlich über die separate
`DowncastToFP8`-Op; die Mischung in `HalfType` war redundant und
hat zu symmetrischer, aber funktionsloser Code-Generierung geführt.

**Quelle:** STOP-1 Resolution Punkt 5.

### Sync-Update 2: §2.4 — Sieben neue DequantOps eintragen

**Alt:** `DequantOp`-Enum enthält 10 Varianten (LoadBytes, LoadFP8,
ExtractNibble, Combine6Bit, IntToFloat, MulF32, FmaF32, DowncastToHalf,
DowncastToFP8, StoreHalf, StoreFP8).

**Neu:** Zusätzlich 7 Varianten:

```rust
// Erweiterungen aus dequant_ir_spec.md §2
ExtractBits   { src: RegId, shift: u8, mask: u32, dst: RegId },
CombineBits   { lo: RegId, hi: RegId, hi_shift: u8, dst: RegId },
SubF32        { a: RegId, b: RegId, dst: RegId },
AddF32        { a: RegId, b: RegId, dst: RegId },
NegF32        { src: RegId, dst: RegId },
ScaleBlockStart { sub_block_idx: u8 },
Const         { value: f32, dst: RegId },
```

**Rationale:** Diese Ops sind für die Q4_K-Scale-Entpackung, die
Q6_K-Zentrierung, und die Q4_K-Fma-mit-Subtraktions-Formel nötig. Sie
sind semantisch minimal (jede Op = eine Hardware-Instruktion nach
Peephole). Referenz zur Semantik: `dequant_ir_spec.md` §2.4–§2.9.

**Quelle:** STOP-1 Resolution (inkrementell über Punkte 1–4).

### Sync-Update 3: §2.4 — SSA-Invariante auf RegIds dokumentieren

**Alt:** `pub type RegId = u32;` — keine Invariante explizit.

**Neu:** Zusätzlicher Absatz:
> **SSA-Invariante.** Jede `RegId` wird genau einmal als `dst` eines
> Ops produziert; der Typ eines `RegId` steht beim Producer fest und
> ändert sich nicht. Der Codegen mappt `RegId`s per Linear-Scan auf
> physische Register (VGPRs auf GPU, ZMM auf CPU). Die Lebensdauer
> endet beim letzten Consumer.

**Rationale:** Der Linear-Scan-Allocator benötigt klare Live-Ranges.
Mehrfach-Assignments werden vom Validator abgelehnt.

**Quelle:** `dequant_ir_spec.md` §2.2, implizit angenommen in §6.7.

### Sync-Update 4: §2.4 — `KernelCacheKey` überarbeiten

**Alt:**
```rust
pub struct KernelCacheKey {
    pub quant_format_id: QuantFormatId,
    pub shape: KernelShape,
    pub precision_level: PrecisionLevel,
    pub tile_config: TileConfig,
    pub gemv_variant: Option<GemvVariant>,
    pub target: KernelTarget,
    pub rocm_version: String,  // bindet an ROCm-Build
}
```

**Neu:**
```rust
pub struct KernelCacheKey {
    pub quant_format_id: QuantFormatId,
    pub shape: KernelShape,
    pub precision_level: PrecisionLevel,
    pub tile_config: TileConfig,
    pub gemv_variant: Option<GemvVariant>,
    pub target: KernelTarget,
    // rocm_version entfernt — siehe CacheMetadata.invalidation_hash
}

pub struct CacheMetadata {
    pub invalidation_hash: [u8; 32],  // SHA-256 über `hipcc --version` + Toolchain
    pub cache_format_version: u32,
    pub created_at: DateTime<Utc>,
}
```

**Rationale:** Code-Object-Format-Änderungen betreffen meist alle
Kernel, nicht einzelne. Globaler Invalidation-Hash ist sauberer und
Lookup-schneller.

**Quelle:** STOP-3 Resolution Punkt 1.

### Sync-Update 5: §2.4 — Invarianten 6 und 7 auf `GemvVariant`

**Alt:** `GemvVariant` dokumentiert die 4 Flags (`q8_inline`,
`fuse_residual`, `fuse_norm`, `x_via_lds`) ohne Validity-Constraints.

**Neu:** Ergänze zwei normative Invarianten:

> **Invariante 6.** `fuse_norm=true` ist nur gültig, wenn der GEMV-Op der
> letzte GEMV vor einem Residual-Add im Computation Graph ist. Der
> Fusion-Pass (Säule 2) garantiert das vor der Kernel-Emission.
> Verletzung ist Compile-Time-Error.
>
> **Invariante 7.** `q8_inline=true` ist nur gültig bei PrecisionLevel
> 0, 1 oder 2. Level 3 (VALU-Referenz) arbeitet ausschließlich ohne
> Quantisierung.

**Rationale:** Beide Constraints sind semantisch, nicht performanz-
bezogen. Compile-Time-Enforcement verhindert Runtime-Fehler.

**Quelle:** STOP-3 Resolution Punkte 2+3.

### Sync-Update 6: §2.4 + §3.1/§3.3 — Rounding-Konsistenz als neue Sub-Section

**Alt:** Keine Rounding-Konsistenz-Spezifikation zwischen GPU- und
CPU-Backend. Peephole-Passes nicht als gemeinsames Modul gekennzeichnet.

**Neu:** Neue Sub-Section in §2.4 (oder als §3.5):
> **Rounding-Konsistenz GPU ↔ CPU.** Der VALU-Parity-Pfad (Säule 6)
> verlangt bit-identische Ergebnisse zwischen Level-3-GPU-Kernel und
> CPU-FP32-Kernel. Dafür gelten vier Regeln: (1) FMA-vs-MUL+ADD-
> Peephole-Pass ist gemeinsam implementiert für beide Emitter.
> (2) FP32→FP16 auf CPU explizit mit `_MM_FROUND_TO_NEAREST_INT`.
> (3) FP32→BF16 nutzt gemeinsamen SW-Algorithmus, nicht
> `_mm512_cvtneps_pbh`. (4) Denormal-Handling beidseitig IEEE-konform.
> Details: `dequant_ir_spec.md` §7.4.

**Rationale:** Verhindert Parity-Check-Bruch durch Double-Rounding oder
Rounding-Mode-Divergenz.

**Quelle:** STOP-4 Zusatz-Hinweis + §7.4 Regeln 1–3.

### Sync-Update 7: §3.1 + §3.3 — Denormal-Handling explizit

**Alt:** Kein explizites Statement zum Denormal-Handling im
GPU-Backend (§3.1) oder CPU-Backend (§3.3).

**Neu:** In §3.1 GPU-Backend ergänzen:
> **Denormal-Handling (FP32/FP16 IEEE-konform).** Jeder generierte
> Kernel setzt am Prolog das MODE-Register explizit auf IEEE-Verhalten
> (`s_setreg_imm32_b32 hwreg(HW_REG_MODE, 4, 4), 0xF`), unabhängig von
> `hipcc`-Defaults. Direkt adressiert das Llama-3.1-Special-Token-
> Problem (Memory-Erkenntnis: SNR-Drop bei kleinen Werten).

In §3.3 CPU-Backend ergänzen:
> **Denormal-Handling (MXCSR explizit).** Der rayon-Worker-Prolog setzt
> `_MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_OFF)` und
> `_MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_OFF)` einmalig pro Thread.
> Das garantiert Parity zum GPU-FP32-Pfad.

**Rationale:** Die Default-Werte dieser Register variieren zwischen
ROCm-Versionen und Compiler-Flags. Expliziter Setup macht das
Verhalten deterministisch und portabel.

**Quelle:** STOP-4 Zusatz-Hinweis + §10.5 (Denormal-Unterabschnitt).

---

*Ende der Dequant IR Spezifikation v1.0.0-final.*
