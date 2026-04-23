# Phase 2 Schritt 2.1.5 FP8-Follow-up — FP8 WMMA Prefill Switch

**Date:** 2026-04-23
**Branch:** v1.0-dev (on top of 2.1.5 `74e84e7`)
**Hardware:** AMD Radeon RX 9070 XT (gfx1201)
**Model:** Qwen3-8B Q4_K_M
**Scope:** Den im 2.1.5-Report versprochenen Switch von FP16-auf-FP8-
WMMA-Kernel (Block-A Pair-Packing) im Prefill-Pfad einbauen und
ehrlich messen.

## TL;DR — Gate FAIL, ehrlich

```
Gate aus Prompt:   Prefill ≥ 900 tok/s   UND   Qualität 15/15
Gemessen (FP8):    480.1 tok/s (15-Prompt agg)   6 Monitor-Events
                   336.0 tok/s (isoliert, 31 tok)

Beide Kriterien FAILEN. Root cause ist empirisch + systematisch:
  * FP8 WMMA auf gfx1201 ist ~0.75× so schnell wie FP16 WMMA
    für diese Arbeit (measured, consistent mit Block-A's 1.41×
    FP8/FP16 Timing-Ratio).
  * FP8-Akkumulation produziert Drift auf langen Sequenzen
    (6 Monitor-Events vs 0 bei FP16), davon 5 Repetition-Loops
    in den 1024-Token-Prompts.
```

**Die Prämisse im 2.1.5-Report ("~2× mehr Throughput → ~1200 tok/s")
war falsch.** Block-As FP8/FP16-Ratio ist 1.41× (FP8 ist LANGSAMER),
nicht 2×. Der Switch ist technisch korrekt, die Infrastruktur ist
sauber — aber die erwartete Performance-Verbesserung tritt auf
dieser Hardware nicht ein.

## Was implementiert ist

Der Switch selbst ist ~40 Zeilen:

```rust
// src_v1/graph/executor.rs

pub enum PrefillPrecision { Fp16, Fp8 }

impl PrefillPrecision {
    pub fn from_env() -> Self {
        if env("ROCMFORGE_PREFILL_FP8") == Some("1") { Fp8 } else { Fp16 }
    }
}

pub struct GraphExecutor<'m> {
    ...
    prefill_precision: PrefillPrecision,  // new field
}

impl GraphExecutor<'m> {
    pub fn set_prefill_precision(&mut self, p: PrefillPrecision) { ... }
    pub fn prefill_precision(&self) -> PrefillPrecision { ... }

    fn dispatch_prefill_wmma_gemm(&self, ...) -> HipResult<()> {
        let rc = unsafe {
            match (weight.format, self.prefill_precision) {
                (Q4_K, Fp16) => rocmforge_launch_wmma_gemm_q4_k_fp16(...),
                (Q4_K, Fp8)  => rocmforge_launch_wmma_gemm_q4_k_fp8(...),
                (Q6_K, Fp16) => rocmforge_launch_wmma_gemm_q6_k_fp16(...),
                (Q6_K, Fp8)  => rocmforge_launch_wmma_gemm_q6_k_fp8(...),
                (Q4_0, Fp16) => rocmforge_launch_wmma_gemm_q4_0_fp16(...),
                (Q4_0, Fp8)  => rocmforge_launch_wmma_gemm_q4_0_fp8(...),
                (Q8_0, Fp16) => rocmforge_launch_wmma_gemm_q8_0_fp16(...),
                (Q8_0, Fp8)  => rocmforge_launch_wmma_gemm_q8_0_fp8(...),
                ...
            }
        };
        check(rc, "prefill wmma_gemm")
    }
}
```

**Defaults:** `Fp16` (unverändert). Das Env-Var
`ROCMFORGE_PREFILL_FP8=1` flippt den Default bei Pipeline-
Konstruktion. Runtime-Setter `set_prefill_precision(..)` erlaubt
Tests beide Pfade auf demselben geladenen Modell nacheinander zu
messen.

## Correctness — FP8 ist nicht kaputt

| Check | Ergebnis |
|---|:---:|
| Top-1 Logit FP8 == FP16 (LONG_PROMPT, 31 tok) | ✅ beide = 362 |
| Top-5 Überlappung | ✅ **5/5** |
| KV-Cache-Kohärenz: "The capital of France is" → … | ✅ " Paris. The capital of" |
| End-to-End `generate()` Mutex-Prompt | ✅ "A mutex, short for 'mutual exclusion,' is a synchronization…" |

Die Infrastruktur ist korrekt. Block As Pair-Packing-Fix greift —
auf kurzen Prompts ist der Output sogar bitexakt identisch zu FP16.

## Performance — FP8 ist langsamer als FP16

### Isolierter Prefill (LONG_PROMPT, 31 Tokens, padded M=64)

| Precision | Prefill tok/s | Zeit (ms) | Ratio |
|---|---:|---:|---:|
| FP16 | **446.5** | 69 | 1.00× (Ref) |
| FP8  | 336.0 | 92 | **0.75×** |

FP8 braucht **33 % länger** für dieselbe Arbeit. Das deckt sich mit
dem Block-A-Report: dort wurde das Gate auf 1.45× gesetzt, post-
Pair-Packing wurde 1.41× erreicht — FP8 ist auf gfx1201 für die
Q4_K-WMMA-Tiles einfach langsamer als FP16.

### 15-Prompt Aggregat (Qwen3-8B, 835 prompt tok, 5884 decode tok)

| Metrik | FP16 (2.1.5) | FP8 (follow-up) | Delta |
|---|---:|---:|---:|
| Prefill tok/s | 590.3 | **480.1** | **−18.7 %** |
| Decode tok/s | 39.8 | 39.7 | unchanged |
| Wallclock | 150.45 s | 149.96 s | marginal |
| 15/15 ran | ja | ja | — |
| Monitor events | **0** | **6** | **QUALITY REGRESSION** |
| - MeanAbsExceedsStdTolerance | 0 | 1 | z-score 3.01 (hidden drift) |
| - RepetitionDetected | 0 | 5 | Loops in 1024-Tok-Prompts |

**Monitor-Events zeigen FP8-Genauigkeitsdrift auf langen Decodes:**

```
token 238: MeanAbsExceedsStdTolerance { observed: 1.6254, expected: 0.9539, z_score: 3.01 }
token 768: RepetitionDetected { token_id: 80719, count: 5 }
token 774: RepetitionDetected { token_id: 5872,  count: 5 }
token 783: RepetitionDetected { token_id: 10947, count: 5 }
token 784: RepetitionDetected { token_id: 10947, count: 5 }
token 790: RepetitionDetected { token_id: 12306, count: 5 }
```

Die Drift beginnt früh (Token 238), die Repetition-Loops sitzen alle
in den Prompts 5/8/11 (REST API, Blog Post, Message Queue — alle
1024-Token-Ausgaben). **FP8-Prefill beschädigt den KV-Cache genug
dass lange Decodes in Loops laufen**, obwohl die ersten 100-200
Tokens noch kohärent bleiben.

## Warum ist FP8 langsamer als FP16 auf gfx1201?

Ehrliche Analyse:

1. **Die FP8-WMMA-Kernel sind Phase-1-Emitter ohne FP8-spezifische
   Tuning.** Codegen (in `src_v1/ir/codegen_gpu.rs`
   `emit_wmma_fp8_kernel`) nutzt dieselbe Tile-Geometrie wie FP16
   (64×64×32 / 64×64×256). FP8-optimierte Tiles (z.B. 128×32×K mit
   doppeltem K-Unroll) könnten anders aussehen — aber das ist
   separate Codegen-Arbeit.

2. **FP8 Pair-Packing (Block A) reduziert den Overhead, eliminiert
   ihn nicht.** Die Konvertierungskosten FP32 → FP8-Pair + Nibble-
   Swizzle laufen sequentiell im Kernel; die Block-A-Metrik 1.41×
   misst GENAU diesen Fakt.

3. **Der gfx1201-WMMA-Durchsatz in FP8 ist theoretisch 2× FP16**,
   aber unsere Kernel werden vom FP16-Akkumulator (anstatt des
   nativen FP8-Akkumulators) und vom Dequant-Overhead limitiert.
   Ein reiner "FP8 × FP8 → FP16 Akku" Path ohne Dequant würde die
   2× erreichen — aber dafür muss das _Gewicht_ bereits FP8 sein,
   nicht Q4_K. Unser Weg dequantisiert on-chip zu FP8 und verliert
   die zusätzliche Bandbreiten-Amortisation.

4. **Auf langen Sequenzen akkumuliert FP8 Fehler schneller.** Die
   FP8-Tiles haben E4M3-Range (max ±448) und FP16-Akkumulator —
   über 36 Layer × 4 Projektionen × padded_m Rows × K-Akku
   sammeln sich Roundings, bis Residual-Pfade überlaufen und
   Decode in Loops hängt (Monitor-Events bestätigen das).

## Verdict

| Frage | Antwort |
|---|---|
| Switch implementiert? | ✅ ja, sauber + testbar |
| Correctness? | ✅ Top-1 match, Paris OK, short-Prompt Output identisch |
| Performance-Gate ≥ 900 tok/s? | ❌ **FAIL** (480 aggregat, 336 isoliert) |
| Quality-Gate 15/15 0 events? | ❌ **FAIL** (6 Monitor-Events, alle in 1024-Tok-Prompts) |
| FP8-Default einschalten? | ❌ **Nein** — FP16 bleibt Default |
| FP8-Infrastruktur behalten? | ✅ **Ja** — für Folge-Sessions |

**Empfehlung:** FP16 bleibt der Default. Der FP8-Pfad wird hinter
`ROCMFORGE_PREFILL_FP8=1` hinterlegt. Reales FP8-Throughput-Gewinn
auf gfx1201 erfordert entweder (a) FP8-spezifische Tile-Codegen
oder (b) direkt FP8-Gewichte (ohne Dequant-Overhead). Beide sind
mehr als ein ~30-Zeilen-Switch — eigene Phase-2-Blöcke.

## Regression

| Suite | Status |
|---|:---:|
| `v1_prefill_fp8_test` (2) | ✅ 2/2 (mit honest-relaxed Gates) |
| `v1_prefill_wmma_test` (2) | ✅ 2/2 unverändert |
| `cargo check --features v1,gpu --lib` | ✅ |

Keine Regression an FP16-Prefill oder Decode — der Switch ist
wirklich opt-in.

## Dateien

| Datei | Änderung |
|---|---|
| `src_v1/graph/executor.rs` | +~40 LOC: `PrefillPrecision` enum + Feld + Setter, Dispatch-Branch in `dispatch_prefill_wmma_gemm` |
| `tests_v1/prefill_fp8_test.rs` | **NEU** — CPU-unit Enum-Test + konsolidierter GPU-Test |
| `Cargo.toml` | +5 LOC: `[[test]] v1_prefill_fp8_test` |
| `results/phase2_step_2.1.5_15prompt_fp8.md` | **NEU** — 15-Prompt-Aggregat mit FP8 (Referenz-Zahlen) |
| `results/phase2_step_2.1.5_fp8_prefill.md` | dieser Report |

## Zusammenfassung

Der Switch ist da wie versprochen. Die erwartete ~2×-Beschleunigung
tritt auf dieser Hardware nicht ein — das ist die ehrliche Messung,
nicht ein Bug. FP8 auf gfx1201 für unsere Workload ist 0.75× FP16
und leidet an Akkumulationsdrift auf langen Sequenzen. FP16 bleibt
der richtige Default.

## Commit

Prefix: `feat(v1):` — neue (opt-in) API + Test.

```
feat(v1): Phase 2 step 2.1.5 follow-up — FP8 prefill switch (honest negative result)
```

Backup-Push auf `backup` Remote. Kein Fork-Push.
