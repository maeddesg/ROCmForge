# Phase 2 Schritt 2.0.1 — Bandit Sync-Elimination

**Date:** 2026-04-21
**Branch:** v1.0-dev
**Hardware:** AMD Radeon RX 9070 XT (gfx1201)
**Scope:** P0-Gate-Fix vor jedem GA-Lauf (ga_tuning_spec §8.1).
Arch-Doc §3.7 "Zero-Sync Pipeline" implementieren.

## Kurzfassung

Der Phase-1-`dispatch_gemv_tuned` + `dispatch_node`-Pfad rief
`hipStreamSynchronize` nach jedem Kernel-Launch. Baseline (1.17):
**83 260 Syncs / 100 Decode-Tokens** — 98 % der HIP-API-Wallclock,
GPU-Effizienz 62.7 %. Nach diesem Fix: **132 Syncs / 100 Tokens**
(≈ 1 pro Token — Logits-Readback + periodischer Monitor-Check) und
Decode-Durchsatz von **30.6 → 40.8 tok/s (+33 %)**.

## Bugs, die gefixt wurden

Zwei Sync-Quellen mussten weg, nicht eine:

1. **`dispatch_gemv_tuned` — Timing-Sync pro GEMV.** Das war der
   Phase-1-Kompromiss: `Instant::now()` um den Launch + Sync für
   den Bandit. Ersetzt durch HIP-Event-Pairs (`EventPool`), die
   auf dem Stream queue'd werden und am Token-Ende gebatched
   ausgelesen werden.
2. **`dispatch_node` — Per-Node-Sync am Ende jedes Graph-Node-
   Dispatches** (!). Das war ein versehentliches Relikt aus der
   1.10-Block-B-Debug-Phase und wurde nie entfernt. Bei ~540
   Nodes pro Decode-Iteration × 133 Iterationen = ~72 k Syncs.
   Das war der eigentliche Hauptbeitrag zu den 83 k in der 1.17-
   Baseline — nicht die Bandit-Timing-Sync wie angenommen.

HIP garantiert in-order Execution auf einem Stream, also sind
weder die per-Node-Sync noch die per-GEMV-Sync korrektheitsnötig.
Beide entfernt, ein expliziter Sync vor dem Logits-Readback
hinzugefügt (`hipMemcpy` ohne Stream-Argument serialisiert
nicht gegen user-created Streams — ohne den Sync würde
`read_buffer` den LM-Head-Kernel racen).

## Neue Dateien / Änderungen

| Datei | Was |
|---|---|
| `src_v1/runtime/events.rs` | `EventPool`, `EventPair`, `PendingMeasurement` — vorallokierte HIP-Event-Pairs, `record_start/stop/flush_into` |
| `src_v1/runtime/mod.rs` | Neue `events`-Mod + Re-Exports; `Runtime::all_exploiting()` |
| `src_v1/graph/executor.rs` | `event_pool: Option<EventPool>` auf `GraphExecutor`. Per-Node-Sync entfernt (bleibt nur unter `nan_guard`/`trace`). `dispatch_gemv_tuned` ohne Sync — Event-Record nur während Exploration. `execute_decode` syncht einmal vor Logits-Readback und flush'd das Pool danach |
| `src_v1/backend/gpu/wrappers.rs` | Prozess-weiter `SYNC_COUNTER` + `sync_count()`/`reset_sync_count()` für P0-Gate-Telemetrie; `HipStream::synchronize()` inkrementiert |
| `tests_v1/sync_elimination_test.rs` | 5 Tests (P0-Gate, Bandit-Konvergenz, Decode-Qualität + Speed, Bandit-Phase-Flag, leeres Registry) |
| `tests_v1/integration_test.rs` | Konvergenz-Check aktualisiert (`all_exploiting()` + Winner-Share ≥ 55 %); alter Pull-Count-Check war auf Pre-Elimination-Verhalten geeicht |

## HIP-Event-Timing-Genauigkeit

`hipEventElapsedTime` liefert Millisekunden, `×1000` in `EventPool::flush_into`
konvertiert auf Mikrosekunden (Bandit-Phase-1-Konvention). Reine
GPU-Zeit ohne Sync-Overhead — Ergebnis: die Bandit-Mittelwerte
sinken gegenüber Phase 1, weil der 45 µs Sync-Overhead pro Call
wegfällt:

| Shape | Kernel | Phase-1 Ø Wall | Phase-2 Ø Event |
|---|---|---:|---:|
| Q4_K 1024×4096 | `q8_inline` | 29.5 µs | **18.4 µs** |
| Q4_K 1024×4096 | `standard` | 73.4 µs | **52.6 µs** |
| Q4_K 4096×4096 | `q8_inline` | 41.1 µs | **32.4 µs** |
| Q4_K 4096×4096 | `standard` | 410.7 µs | **166.6 µs** |
| Q4_K 4096×12288 | `q8_inline` | 90.4 µs | **80.1 µs** |
| Q4_K 4096×12288 | `standard` | 506.2 µs | **493.0 µs** |

Die Kernel waren schon immer so schnell — Phase 1 hat nur
Wallclock + Sync-Overhead zusammen gemessen. Das Verhältnis
Winner/Loser bleibt konsistent (~2.5–5×), also konvergiert UCB1
weiterhin auf dasselbe q8_inline.

## Sync-Count (rocprofv3 `--hip-trace`)

| Run | hipStreamSynchronize | Decode tok/s | Dispatches/Token |
|---|---:|---:|---:|
| 1.17 Baseline (Phase 1, Tuned) | **83 260** | 30.6 | 608 |
| 2.0.1 Sync-Elim | **114 total / 132 getestet** | **40.8** | 495 |
| v0.3.0 Referenz | 114 | 41.7 | 462 |

**P0-Gate: PASS** — Messung via `sync_count()` im Test: 132 Syncs
für 100 Decode-Tokens (Ziel: < 200). rocprof-Trace zeigt 114 im
gesamten Run (inkl. Setup). Die 132 enthalten den Monitor-Check
jede 32 Tokens + den Logits-Readback-Sync pro Token + Reset.

## Decode-Performance

### Mutex-Prompt (120 Tokens max, Greedy, Tuning aktiv)

| Run | Prompt tok | Decode tok | Decode tok/s | Wallclock |
|---|---:|---:|---:|---:|
| 1.16 Phase-1-Final | 33 | 96 | 30.6 | 4 255 ms |
| **2.0.1** | **33** | **100** | **40.8** | **3 329 ms** |
| v0.3.0 | 18 | 128 | 41.7 | 3 072 ms |

Effektiv **auf Augenhöhe mit v0.3.0**. Der verbleibende 2 %-Gap
stammt vom unterschiedlichen Chat-Template (`/no_think` macht den
Prompt 15 Tokens länger → höhere Prefill-Kosten).

### 15-Prompt-Suite (Qwen3-8B Q4_K_M)

| Metrik | 1.16 Phase-1-Final | **2.0.1** | Delta |
|---|---:|---:|:---:|
| EOS-Hits | 3/15 | 3/15 | — |
| Prefill tok/s (Aggregate) | 31.1 | **41.8** | +34 % |
| Decode tok/s (Aggregate) | 29.8 | **39.6** | +33 % |
| Wallclock total | 225.8 s | **169.8 s** | **−25 %** |
| Monitor-Events | 0 | 0 | — |
| Qualität (Human-Rating) | 13/15 korrekt | unverändert | — |

Report: `results/phase2_step_2.0.1_15prompt.md`.

## Bandit-Konvergenz

Die Bandit-Arme picken weiterhin `q4_k_q8_inline` bei allen drei
multi-variant Shapes (Stichprobe aus der 15-Prompt-Suite):

| Shape | Winner (66.7 % der Pulls) | Winner Ø | Loser Ø |
|---|---|---:|---:|
| 1024 × 4096 | q8_inline | 18.4 µs | 52.6 µs |
| 4096 × 4096 | q8_inline | 32.4 µs | 166.6 µs |
| 4096 × 12288 | q8_inline | 80.1 µs | 493.0 µs |

`total_pulls` ist jetzt niedriger (54, 162, 216 statt 1800+ in
1.16) — das ist **kein Bug**: nach `all_exploiting() == true`
werden keine neuen Events mehr recorded, also stoppen die Pulls.
Die Bandit-Arme behalten ihre gelernten Mittelwerte und
wählen weiterhin den Winner. `Runtime::all_exploiting()` liefert
`true` ab ~Token 2 auf allen drei Shapes.

## Tests

### Neue Tests (`tests_v1/sync_elimination_test.rs`, 5/5 grün)

| Test | Status |
|---|:---:|
| `test_bandit_is_exploiting_threshold` (CPU) | ✅ |
| `test_runtime_all_exploiting_empty_registry` (CPU) | ✅ |
| `test_p0_gate_sync_count_under_200` (GPU) | ✅ (132 < 200) |
| `test_bandit_still_converges_with_events` (GPU) | ✅ (q8_inline wins) |
| `test_decode_quality_unchanged_and_faster` (GPU) | ✅ (40.8 tok/s > 32.0) |

Die 3 GPU-Tests müssen einzeln laufen (Box::leak-VRAM-Issue — siehe
Step 1.11).

### Regression

| Suite | Tests | Status |
|---|---:|:---:|
| `v1_codegen_elementwise_emit_test` | 2 | ✅ |
| `v1_codegen_gemv_emit_test` | 2 | ✅ |
| `v1_runtime_test` | 8 | ✅ |
| `v1_introspection_test` | 5 | ✅ |
| `v1_monitor_test` | 11 | ✅ |
| `v1_inference_test` (Tokenizer-Subset) | 3 | ✅ |
| `v1_integration_test` | 2 | ✅ (Konvergenz-Check angepasst, siehe oben) |
| v0.x Build | — | ✅ |

## Design-Entscheidungen

- **Event-Pool-Kapazität 256** — Qwen3 hat 32 Layer × 5 GEMVs + LM-Head =
  ~165 GEMVs/Token, 256 gibt ~55 % Headroom. Statisch vorallokiert;
  `hipEventCreate` ist teuer und läuft nur einmal in `attach_runtime`.
- **Flush-Strategie: 1× pro Token** — am Ende von `execute_decode`,
  direkt nach dem Logits-Sync. Das ist der einzige Sync der pro
  Token ohnehin nötig ist.
- **Exploration → Exploitation ist implizit:** `want_timing = !runtime.all_exploiting()`
  in `dispatch_gemv_tuned`. Kein separates Flag, kein Setter. Sobald
  alle Bandits konvergiert haben, hört das Recording automatisch auf.
- **Sync-Counter als Atomic statt env-gated** — ein relaxed-
  `fetch_add` ist im µs-Timing-Bereich unsichtbar, und wir brauchen
  die Zahl später im rf-forge-P0-Gate.
- **Per-Node-Sync in `dispatch_node` NUR unter `nan_guard`/`trace`**
  — der Debug-Pfad braucht den Sync, damit `check_node` den
  tatsächlichen Output sieht. Normaler Run: keine Syncs.

## Bekannte Limitierungen (weiterhin Phase-2-Backlog)

Der Sync-Fix allein bringt die Phase-2-Prio-Queue nicht weiter:

- `gate_up_swiglu` ist weiterhin 60 %+ der GPU-Zeit (BW-limitiert)
- Residual-fused GEMV fehlt (v0.x-Pattern, nächster P0)
- `q6_k` LM-Head hat nur 1 Variante, Bandit kann nichts wählen
- WMMA-Prefill ist noch nicht gewired (Prefill 41.8 tok/s vs. v0.x 289 tok/s)

Diese adressiert die reguläre Phase-2-Roadmap — der Fix hier ist
das Prerequisite, damit GA-Fitness-Messungen nicht von
Sync-Overhead verzerrt werden.

## Commit

Prefix `perf:` — gewöhnlicher Perf-Fix, kein Feature.
