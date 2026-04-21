# Phase 1 Schritt 1.15 — CLI + Integration

**Date:** 2026-04-21
**Branch:** v1.0-dev
**Hardware:** AMD RX 9070 XT (gfx1201)
**Scope:** Audit + Lückenfüllung. Keine neuen Säulen; prüft ob das
was in 1.1–1.14 gebaut wurde als eine Einheit sauber zusammenspielt.

## Bereits vorhanden (aus 1.11–1.14)

| Feature | Status | Eingeführt in |
|---|:---:|---|
| `--model <path>` | ✅ | 1.11 |
| `--prompt <text>` | ✅ | 1.11 |
| `--inference-test` + `--suite` + `--output` | ✅ | 1.11 |
| `--interactive` (KV-Cache pro Turn reset) | ✅ | 1.11 |
| `--list-tensors` | ✅ | 1.6 |
| `--max-tokens <N>` | ✅ | 1.11 |
| Startup-Flow (GGUF → Introspection → Graph → Executor) | ✅ | 1.13 |
| Monitor-Calibration per Inference-Test-Runner | ✅ | 1.14 |
| Bandit-Runtime per Inference-Test-Runner | ✅ | 1.12 |
| Bandit-Tuning-Report am Ende einer Suite | ✅ | 1.12 |

## Neu ergänzt

| Feature | Datei | Kommentar |
|---|---|---|
| `--show-introspection` | `src/bin/rocmforge_v1.rs`, `src_v1/cli/inference_test.rs` | Druckt `ModelProfile::print_summary` nach dem Laden |
| `--show-quality` | dto. | Calibriert den Monitor und druckt `revision_log`-Zähler + erste 8 Events am Ende |
| `--show-tuning` | dto. | Hängt `Runtime` an und druckt `print_tuning_report()` am Ende |
| `--show-all` | dto. | Alias für alle drei |
| Startup-Banner | `src_v1/cli/inference_test.rs::print_banner` | Eine Zeile beim Laden: Modell, GPU + GCN-Arch, Modell-Arch, Layer-Anzahl, hidden/heads/vocab |
| `quit`/`exit`/Leerzeile → Session-Ende in `--interactive` | `src/bin/rocmforge_v1.rs::run_interactive` | War vorher nur leere Zeile |
| `ShowFlags`-Struct | `src_v1/cli/inference_test.rs` | Trägt die drei Flags einheitlich durch `run_single_prompt` und `run_interactive` |

## Keine Änderung nötig

- **Startup-Flow-Reihenfolge** — `InferencePipeline::new` macht bereits
  1. GGUF-Metadaten + Config
  2. Tokenizer
  3. Model Introspection (`introspect`)
  4. GraphExecutor-Konstruktion (Graph + Executor-Buffers + KV-Cache)
  5. `QualityMonitor::new` (uncalibriert; Calibration auf Caller-Seite)
  Runtime und Monitor-Calibration sind bewusst opt-in, damit der
  CLI-Pfad pro Flag entscheidet was teuer genug ist zum Aktivieren.
- **Introspection-Auto-Print entfernt** — die Pipeline druckte den
  Summary-Block unabhängig vom CLI-Flag, was bei `--show-introspection`
  zu einer Doppel-Ausgabe führte. Die einzige bleibende automatische
  Ausgabe ist die einzeilige SNR-Warnung für kritische Modelle; der
  volle Block wird nur bei `--show-introspection` gedruckt.

## 3-Modell-Test

| Modell | Laden | Inferenz | Bemerkung |
|---|:---:|:---:|---|
| **Qwen3-8B Q4_K_M** | ✅ | ✅ (30 tok/s) | Volle Abdeckung; 15-Prompt-Suite 13/15 korrekt |
| **Meta-Llama-3.1-8B-Instruct Q4_K_M** | ✅ | ✅ (23 tok/s) | "It's nice to meet you. How can I help you…" @ EOS |
| **Qwen2.5-7B-Instruct Q4_0** | ✅ | ❌ | Crasht bei ffn_down-Dispatch: `HIP error -1 (no GEMV kernel for format Q4_1)`. Dieses GGUF mischt Q4_0-Attention mit **Q4_1**-FFN-down. Phase 1 hat keinen Q4_1-GEMV. |
| qwen2.5-0.5B Q4_K_M | ✅ | ❌ | Q5_0-Embedding-Tabelle → `embedding table format Q5_0 not supported in Phase 1` in `executor init`. |

**2 von 3 Zielmodellen laufen clean. Qwen2.5-Integration braucht zwei
Phase-2-Ergänzungen:**

1. **Q4_1-GEMV-Kernel** (für ffn_down). Der Block-Header hat ein
   zusätzliches `min`-Feld (`d + m + qs`) im Gegensatz zu Q4_0
   (`d + qs`) — ~150 LOC neuer Kernel, analog zu den existierenden
   `_standard`-Varianten.
2. **Attention-Bias-Add** als Executor-Node-Typ. Qwen2.5 hat
   `has_attention_bias=true` für Q/K/V-Projektionen; das existierende
   `Gemm { bias: Option<WeightRef> }` liest das Feld aber ignoriert
   es. Der Dequant-+ -Add-Pfad muss an `dispatch_gemv` angehängt
   werden.

Beide Punkte sind in der Step-1.10-Executor-Dokumentation explizit
als Phase-1-Deferrals vermerkt ("Biases are **ignored** for now —
Qwen2.5 attention biases would need a separate add pass that Phase 1
doesn't implement."). Dieses Audit dokumentiert die Konsequenz: das
Qwen2.5-GGUF lädt, produziert aber keine Inferenz bis Phase 2.

## Streaming-Output

Wird in Phase 1 **nicht** implementiert. `generate` akkumuliert alle
Tokens und dekodiert einmal am Ende per `tokenizer.decode` (inkl.
`strip_think_block`). Token-für-Token-Streaming bräuchte:
- einen Callback-Parameter im `generate`-Loop, oder
- eine separate `generate_streaming`-Variante die pro Step dekodiert
  und via `print!` + `stdout().flush()` ausgibt.

Beide Varianten interagieren schlecht mit dem `<think>…</think>`-
Stripping (das über den ganzen Output läuft). Phase-2-Kandidat —
dort kann das Stripping auf den `StreamingEmitter`-Pfad aus v0.x
umgestellt werden, der partielle Tags über Token-Grenzen hinweg
puffert.

## Test-Ergebnisse

Zwei neue Integration-Tests in `tests_v1/integration_test.rs`:

| Test | Status | Deckt ab |
|---|:---:|---|
| `test_startup_flow_qwen3_all_pillars` | ✅ | Säule 1+4+5 einmal im selben Prozess wired-up; Bandit-Konvergenz auf `q8_inline`; Monitor-Log clean |
| `test_full_pipeline_llama31` | ✅ | Llama-3.1 läuft regressionsfrei (1.14-Fix, NeoX-RoPE + LDS-Ceiling) |

**Auf 4 konsolidiert geplante Tests zurück reduziert:** das
`Box::leak`-Muster im Testharness hält jeden Pipeline-VRAM
permanent; mit 16 GB VRAM reichen zwei 8B-Modelle im gleichen
Prozess. Ein "sequential 3 models"-Test ist deshalb nicht
machbar — wer das testen will, muss die drei Modelle in
separaten `cargo test`-Runs laden. Stattdessen deckt ein
Test den kompletten Startup-Flow ab (alle 5 Säulen + generate)
und ein Test die Llama-3.1-Regression.

## Regression

| Suite | Status |
|---|:---:|
| `v1_codegen_elementwise_emit_test` (2/2) | ✅ |
| `v1_codegen_gemv_emit_test` (2/2) | ✅ |
| `v1_runtime_test` (8/8) | ✅ |
| `v1_introspection_test` (5/5) | ✅ |
| `v1_monitor_test` (11/11) | ✅ |
| `v1_inference_test` Tokenizer-Subset (3/3) | ✅ |
| `v1_integration_test` (2/2, neu) | ✅ |
| v0.x Build | ✅ |

## Manuell verifiziert (CLI)

```bash
$ target/release/rocmforge-v1 --model ~/models/Qwen3-8B-Q4_K_M.gguf \
    --prompt "Hallo" --max-tokens 10 --show-all
ROCmForge v1.0-dev | Qwen3-8B-Q4_K_M.gguf | AMD Radeon RX 9070 XT (gfx1201) | arch=qwen3 | 36 layers | hidden=4096 heads=32 vocab=151936
[introspect] scanned 252 tensors, 151936 embedding rows, 1.42s
⚠  SNR risk score 0.14 — precision upgrade may be needed (Phase 2 GA)
   critical embedding tokens: 215 of 151936 vocab
<ModelProfile-Block>
[monitor] calibrated: …
Hallo! Wie kann ich Ihnen heute helfen? 😊
--- 24 prompt tok, 10 decode tok, 33.3 tok/s decode, 1117 ms total ---
=== Self-Tuning Runtime Report ===
registry: 6 shapes, 3 with >1 variant (Bandit active)
  q4_k_q8_inline 100%, …
=== end report ===
Quality Monitor: 0 event(s)
```

## Bekannte Limitierungen (Phase-2-Kandidaten)

- **Streaming-Output** — aktuell akkumulierte Ausgabe am Ende; siehe oben.
- **Q4_1-GEMV-Kernel** fehlt → Qwen2.5-7B Q4_0 ffn_down.
- **Attention-Bias-Add** fehlt → Qwen2/Qwen2.5 laufen mit bias=0.
- **Q5_0-Embedding-Dequant** fehlt → Qwen2.5-0.5B Q4_K_M.
- **`--interactive` kein Multi-Turn-Kontext** — KV wird zwischen
  Turns resettet. Würde pro Turn den Cache-Offset persistieren und
  den vorherigen Turn als Prefill-Präfix einspielen; ist im Doc §5.3
  nicht für Phase 1 vorgesehen.

## Commit

Siehe Git-History (Prefix `feat:`). Der Report ist Teil des Commits.
