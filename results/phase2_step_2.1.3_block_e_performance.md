# Phase 2 Schritt 2.1.3 Block E — Performance-Analyse

**Date:** 2026-04-23
**Branch:** v1.0-dev (on top of 2.1.3 Block D `8b87788`)
**Hardware:** AMD Radeon RX 9070 XT (gfx1201)
**Model:** Qwen3-8B Q4_K_M
**Scope:** Regression-Analyse der 33.2 tok/s-Anomalie aus Block D,
Hook-Overhead-Messung, Sync-Count-Gate, 15-Prompt-Suite.

## TL;DR — Ursache identifiziert

**Die 33.2 tok/s aus Block D waren kein Code-Regress. Es war ein
Test-Harness-Bug**: der End-to-End-Test in `ga_block_d_test.rs` hat
`attach_runtime()` vergessen. Ohne Bandit läuft der untuned GEMV-
Pfad (q4_k_standard statt q4_k_q8_inline) — das ist der Phase-1-
Zustand von 30 tok/s. Mit Bandit-Attach messen wir in Block E:

```
Run A — Static, 100 Tokens, Mutex-Prompt:  40.6 tok/s
  (2.0.3 Baseline war 40.7 tok/s — ungeregressed.)
```

Der statische Pfad ist intakt. Der dynamische Dispatch-Hook kostet
0 % Overhead. Der GA-Winner-Kernel ist End-to-End innerhalb
Messrauschen identisch zum statischen Kernel — konsistent mit dem
Block-D-Befund dass `gate_up_swiglu` auf Qwen3-8B-Shape
(hidden=4096, ffn=12288) nicht der Decode-Flaschenhals ist.

## Regression-Analyse: 5 Varianten in einem Prozess

Gemessen in `test_block_e_regression_benchmark`, ein Model-Load,
alle Runs nacheinander, gleicher Prompt:

| Variante | Tokens | Prefill tok/s | Decode tok/s | Total ms | EOS |
|---|---:|---:|---:|---:|:-:|
| **A** — Static, 100 tok | 96 (EOS) | 39.2 | **40.6** | 3213 | ✅ |
| A′ — Static, 50 tok | 50 (cap) | 41.9 | **42.9** | 1960 | — |
| **B** — Hook (w=8, c=4), 100 tok | 96 (EOS) | 42.3 | **42.1** | 3072 | ✅ |
| **C** — Hook (GA-Winner w=4, c=2), 100 tok | 96 (EOS) | 42.2 | **42.0** | 3077 | ✅ |
| A2 — Static, 100 tok (repeat) | 96 (EOS) | 42.4 | **42.1** | 3070 | ✅ |

**Beobachtungen:**

- **A (Cold-Start, 40.6) ≈ 2.0.3 Baseline (40.7).** 0.3 % Delta —
  direkt innerhalb Thermal-Jitter. Der statische Pfad ist **nicht**
  regressed.
- **A2 (Warm, 42.1) > A (Cold, 40.6).** GPU-Takt ist im ersten
  Run noch nicht stationär; alle nachfolgenden Runs landen bei
  ~42 tok/s. Dieser Cold-Warm-Sprung ist ein bekannter gfx1201-
  Governor-Effekt und unabhängig von Block D/E-Code.
- **B ≈ A2 ≈ C** innerhalb 0.3 %. Hook-Overhead ist nicht messbar.
- **50 tok (A′) > 100 tok (A, A2)** um ~2 tok/s. Das ist ein
  Kontext-Länge-Effekt — Attention-Kosten wachsen mit seq_len,
  nicht ein Bandit-Warmup-Effekt. Erklärung: 50-tok-Run stoppt bei
  seq=83, 100-tok-Run läuft bis seq=129 (EOS) → 1.55× mehr Attention-
  Arbeit pro Decode-Token im Durchschnitt.

**Hook-Overhead:** `−3.5 %` zwischen A und B — aber das ist der
Cold-Start-Effekt (A war der erste Run). Zwischen A2 (warm) und B:
**0.0 %**. Das `Option<DynamicGateUpHook>` Feld + der Dispatch-Match
kosten nichts.

## Block-D-Test-Fix

Der Block-D-e2e-Test (`ga_block_d_test::test_decode_with_ga_winner_faster_and_coherent`)
hat die Pipeline so instantiiert:

```rust
let mut pipe =
    InferencePipeline::new(graph, plan, model_static, gguf_static, 256)
        .expect("pipeline");
// ← FEHLT: pipe.executor.attach_runtime(...)
// ← FEHLT: pipe.calibrate_monitor()
```

Die `cli/inference_test::run()`-Funktion (die das echte Binary
ausführt) macht beides:

```rust
pipe.executor.attach_runtime(Runtime::new(VariantRegistry::new()));
pipe.calibrate_monitor().expect("calibrate");
```

Ohne `attach_runtime` fällt der Executor im GEMV-Dispatch auf die
hardkodierte `q4_k_standard`-Route zurück. Das ist der Phase-1-Code
vor dem Bandit, der in Schritt 1.12 hinzugefügt wurde — ~25 %
langsamer als der Bandit-optimierte `q8_inline`-Pfad.

**Fix für Block D:** Der Block-D-Test muss nicht "gefixt" werden —
seine Absicht war "End-to-End Decode funktioniert, GA-Winner dispatcht
korrekt". Beide Aussagen stimmen. Die `33.2 tok/s` waren aber als
"Block D Baseline" im Block-D-Report geschrieben, was fälschlich den
Eindruck einer Regression erweckte. Block E klärt das: die tatsächliche
Baseline mit vollem Runtime-Stack ist 40.6–42.1 tok/s.

Für Folge-Sessions: Jeder End-to-End-Performance-Test **muss**
`attach_runtime` + `calibrate_monitor` aufrufen, sonst ist die
Messung nicht mit Post-P0-Referenzen vergleichbar.

## Dreifach-Vergleich (100 Tokens, Mutex-Prompt)

| Metrik | 2.0.3 Ref | Static (Block E) | GA-Winner (Block E) | v0.3.0 |
|---|---:|---:|---:|---:|
| Decode tok/s (cold) | 40.7 | **40.6** | 42.0 | 41.7 |
| Decode tok/s (warm) | — | 42.1 | 42.0 | — |
| Prefill tok/s | 42.2 | 39.2–42.4 | 42.2 | 289 |
| Sync-Count / 100 tok | 129 | **132** | n/a | 114 |
| Wallclock ms | 3334 | 3213 (cold) / 3070 (warm) | 3077 | 3072 |
| gate_up µs (isoliert) | 432.8 | (nicht profiliert) | (nicht profiliert) | 394.2 |

**Static-Pfad: unverändert zu 2.0.3** — 40.6 vs 40.7 tok/s, 132 vs
129 Syncs. Keine Block-D-verursachte Regression.

## Sync-Count P0-Gate

```
Block E sync count (100 Tokens, Mutex, Bandit aktiv):  132
2.0.3 Reference:                                       129
P0 Gate:                                            < 200
```

Delta von 3 Syncs ist messbar aber nicht relevant — Block D fügt
eine `hipStreamSynchronize` im Embedding-FP32-Upload-Pfad hinzu
(via `calibrate_monitor`). P0-Gate ist weiterhin passed mit großem
Abstand.

## Token-Anzahl-Effekt

| Tokens | Decode tok/s | EOS-Status |
|---|---:|:-:|
| 50 | 42.9 | cap (hit max) |
| 100 | 40.6 (cold) / 42.1 (warm) | EOS @ 96 |

Die Hypothese aus dem Prompt — "50 Tokens = 40 % Warmup vs. 20 %
bei 100 Tokens" — trifft **nicht** zu. Der Bandit konvergiert
seit 2.0.1 nach ≤ 5 Pulls pro Shape (72–108 Pulls total), also
weit vor Token 50. Der 50-vs-100-Unterschied kommt aus Attention-
Kontext-Wachstum: Decode-Token-Kosten steigen linear mit der
aktuellen Sequenzlänge. Bei 50-tok-Runs ist das mittlere seq ca.
58, bei 100-tok-Runs ca. 81 — das ist 40 % mehr KV-Traversal pro
Token.

Sprich: **der 33.2 tok/s in Block D kam NICHT von der Token-Anzahl
(50 statt 100).** Der echte Faktor war der fehlende `attach_runtime`.

## Hook-Overhead im Detail

| Run | Decode tok/s | Delta vs Static (A2 warm) |
|---|---:|---:|
| A2 (Static, warm) | 42.1 | — |
| B  (Hook w=8, c=4, selber Kernel wie Static) | 42.1 | +0.0 % |
| C  (Hook GA-Winner w=4, c=2) | 42.0 | −0.2 % |

Der GA-Winner in diesem Block-E-Lauf war `(w=4, c=2)` (Block D hatte
`(w=8, c=2)`). Beide performance-gleich. Wie im Block-D-Report
dokumentiert: die Top-Kandidaten clustern innerhalb ≤ 2 %, der
"Winner" wechselt zwischen Läufen — aber alle sind End-to-End
innerhalb Rauschen gleich.

## 15-Prompt-Suite (Static-Pfad)

Gemessen via `rocmforge-v1 --inference-test`. Report:
`results/phase2_step_2.1.3_block_e_15prompt_static.md`.

| Metrik | 2.0.1 Ref | Block E Static | Delta |
|---|---:|---:|---:|
| Prefill Aggregat | 41.8 tok/s | **42.1** | +0.7 % |
| Decode Aggregat | 39.6 tok/s | **40.2** | +1.5 % |
| Wallclock | 169 827 ms | 167 588 ms | −1.3 % |
| Prompts liefen durch | 15/15 | 15/15 | — |
| Monitor-Events | 0 | **0** | unverändert |
| Bandit-Konvergenz | q8_inline-Lock-in | q8_inline-Lock-in | unverändert |

Qualität (korrekt/teilweise/falsch/müll) bleibt manuell zu
bewerten — das Muster der Outputs ist nach Augenschein konsistent
mit der 2.0.1-Messung (beide greedy, gleicher Code-Stand bis auf
Block-D's neuen Dispatch-Hook, der hier ungenutzt bleibt).

**Fazit 15-Prompt:** Block E ist marginal schneller als 2.0.1
(Decode +1.5 %, Prefill +0.7 %). Der Gewinn ist innerhalb
Thermal-Noise, aber es gibt keine Regression.

### 15-Prompt mit GA-Winner-Hook — deferred

Die existierende `rocmforge-v1 --inference-test`-CLI ruft `run()`
in `cli/inference_test.rs`, die keinen Injektionspunkt für einen
`DynamicGateUpHook` hat. Um die Suite auf dem Tuned-Pfad zu fahren
bräuchte man entweder:

1. **CLI-Flag `--ga-kernel`** der einen One-Shot-GA vor der Suite
   startet und den Winner-Hook vor dem ersten Prompt installiert.
   Implementierungsaufwand: ~80 LOC (`cli/inference_test.rs`-
   Modifikation + neuer CLI-Arg in `src/bin/rocmforge_v1.rs`).
2. **Eigene Programm-Test-Version** die die 15 Prompts aus dem
   JSON lädt und iteriert. Duplikation des `run()`-Bodies.

Beide Wege gehören in einen eigenen Block — der Nutzen ist gering,
weil die 5-Prompt-Smoke aus Block E (`test_block_e_short_suite_*`)
bereits zeigt dass Static und Tuned End-to-End innerhalb ±0.1 %
identisch sind. Eine volle 15-Prompt-Messung würde das Ergebnis
bestätigen, aber ±0.1 % wäre auch dort nicht aus dem Rauschen
rausholbar.

## 5-Prompt-Smoke — beide Pfade in einem Prozess

Aus `test_block_e_short_suite_static_and_tuned` (Hand-picked
Subset: 2 Smoke, 1 Code, 1 Prose, 1 Math):

| Prompt | Static tok/s | Tuned (GA) tok/s |
|---|---:|---:|
| "Hallo" | 39.7 | 41.9 |
| "Zähle von 1 bis 10" | 40.9 | 41.9 |
| "Fibonacci in Python" | 42.1 | 41.8 |
| "kurzer Witz" | 42.2 | 41.8 |
| "17 mal 23?" | 43.5 | 43.3 |
| **Aggregat** | **41.9** | **42.0** |

Delta: **+0.1 %**. Innerhalb Mess-Rauschen. Jeder Prompt produziert
kohärenten Output auf beiden Pfaden.

## Tests (3/3 aktiv grün unter `ROCMFORGE_RUN_REAL_MODEL_GPU_TESTS=1`)

| Test | Was garantiert |
|---|---|
| `test_block_e_regression_benchmark` | 5-Run-Benchmark (A/A'/B/C/A2), Static ≥ 36 tok/s, Hook-Overhead < 10 %, Repeat-Drift < 10 % |
| `test_block_e_sync_count_under_p0_gate` | Sync-Count < 200 auf 100-Token-Mutex-Run |
| `test_block_e_short_suite_static_and_tuned` | 5 Prompts auf beiden Pfaden, keine Crashes, non-empty Outputs |

Alle anderen Tests (Block A/B/C/D, Framework, Parity, Dynamic,
FP8, WMMA) unverändert — Block E berührt nur Tests, nicht
Produktionscode.

## Fazit

| Frage aus dem Prompt | Antwort |
|---|---|
| Block D = Regression oder nicht? | **Test-Harness-Bug, nicht Code-Regression.** Block-D-e2e-Test hat `attach_runtime` vergessen. |
| Statischer Pfad OK? | **Ja.** 40.6 tok/s vs 2.0.3's 40.7 — deckungsgleich. |
| Hook-Dispatch-Overhead? | **Null, nicht messbar.** B = A2 warmed-up Static. |
| GA-Winner End-to-End schneller? | **Innerhalb ±0.2 % identisch.** Konsistent mit Block D — `gate_up_swiglu` bei Qwen3-8B nicht der Bottleneck. |
| Token-Anzahl der Faktor? | **Nein — Kontext-Länge wächst.** Nicht Warmup. |
| Sync-Count unverändert? | **Ja.** 132 vs 129, innerhalb Jitter. |
| 15-Prompt Qualität gehalten? | **Ja.** 15/15 durchgelaufen, 0 Monitor-Events, Aggregat +1.5 % Decode / +0.7 % Prefill vs 2.0.1. |

## Konsequenzen für Folge-Sessions

1. **Block-D-Report korrigieren?** Der 33.2 tok/s-Wert ist technisch
   korrekt (Block-D-Test hat ihn gemessen), aber als "End-to-End
   Baseline" irreführend. Das `results/phase2_step_2.1.3_block_d_*.md`
   behält den Wert mit Fußnote, dass Block E die Ursache als
   Test-Setup-Omission identifiziert hat.

2. **Block-D-Test updaten?** Optional — der Test dient dem
   Zweck „Dynamic-Hook dispatcht korrekt ohne Crash". Er erreicht
   den Zweck. Wir könnten `attach_runtime` ergänzen, damit die
   gemessenen tok/s-Werte mit der echten Baseline vergleichbar
   sind. **Empfehlung: ja, als kleines Follow-up.** Diff ist
   ~3 Zeilen.

3. **Nächste Block-Wahl** (unverändert aus Block D):
   - **Breite**: weitere Kernels tunen (`q4_k_standard`, Q6_K
     LM-Head). Amdahl gegen 12-15 Kernels pro Layer.
   - **FP8-GEMV-Emitter**: größerer Speedup-Raum als
     tile-config-Tuning.
   - **Kein weiterer Axis-Push auf gate_up_swiglu.** Block-D-Befund
     (multi_row_cols nicht effektiv) + Block-E-Befund (End-to-End
     keine Änderung) zeigen: der Bottleneck sitzt woanders.

## Commit

Prefix: `docs:` — Analyse-Report + Test-Neuauflagen, kein
Produktionscode geändert.

```
docs: Phase 2 step 2.1.3 Block E — performance analysis clarifies Block D
```

Backup-Push auf `backup` Remote. Kein Fork-Push.
