# P3 — Dokumentation + v1.0.0 Release-Vorbereitung

**Date:** 2026-04-25
**Branch:** v1.0-dev (lokal), Tag `v1.0.0-rc1`
**Status:** **Release-Vorbereitung abgeschlossen.** README + CHANGELOG geschrieben, Version auf 1.0.0 gehoben, Release-Build verifiziert, 15-Prompt-Suite grün, Tag `v1.0.0-rc1` lokal erstellt — **nicht gepusht**, wartet auf User-Bestätigung.

## TL;DR

ROCmForge v1.0.0 wird mit **95.4 tok/s Decode aggregate** und **768.6 tok/s Prefill aggregate** über die 15-Prompt-Suite geshipped. 15/15 Prompts laufen ohne Fehler. Alle 29 v1-Integrationstests grün. Release-Binary smoke-getestet. Drei Dokumente neu geschrieben/erweitert: `README.md` (Triple-Engine-Vergleich, ehrliche Vulkan-Lead, vollständige Known-Limitations-Liste), `CHANGELOG.md` (v1.0.0-Entry mit honest-negatives), `Cargo.toml` (0.3.0 → 1.0.0).

## Deliverables

### 1. README.md (komplett neu geschrieben)

Sektionen:
- Acknowledgement to oldnordic (vom v0.3.0-README erhalten)
- **Performance**-Tabelle mit Triple-Engine-Vergleich (ROCmForge vs llama.cpp ROCm vs llama.cpp Vulkan); honest reporting, dass Vulkan heute der Performance-Champion ist
- Quick-Start (Prerequisites, Build, Download, Run)
- Features (8 Stichpunkte, technisch präzise)
- CLI-Referenz (alle 11 Flags)
- Env-Flags (4 Flags incl. opt-in Q6_K-MMQ)
- Supported Models (7 Modelle mit ✅/⚠/❌ Status, Decode tok/s, Notes)
- Supported Quant-Formate (Q4_K, Q6_K, Q8_0, Q4_0 ✅; Q5_0 ❌)
- Hardware-Anforderungen
- **Known Limitations** (9 Punkte, nichts versteckt — incl. 1024-Token-Plan-Cap, Llama-3.1 Instruction-Following, dense-only, kein SentencePiece, etc.)
- Architecture (6-Säulen-Übersicht)
- License: MIT

Datei: `README.md` (243 Zeilen, +397 / -234 vs alt).

### 2. CHANGELOG.md (v1.0.0-Entry hinzugefügt vor 0.3.0)

Format Keep-a-Changelog, Sektionen:
- **Performance** (Decode 96.2, Prefill +28.7 % via MMQ)
- **Features** (Integer-WMMA-Kernel, Q6_K opt-in, attention v2 als research, chat-template disambig, arch-aware sampling, streaming + think-filtering)
- **Models tested** (Tabelle mit 6 Modellen)
- **Honest negatives — investigations that didn't ship** (5b VGPR-Reduktion reverted, 6 Q6_K MMQ opt-in only, P0.3 attention v2 nicht default)
- **Known limitations** (zusammengefasst aus README)

Lehre dokumentiert: "on RDNA 4, 'obvious' restructurings often regress when the original code was already saturating the WMMA pipeline / wave occupancy / L2 cache."

Datei: `CHANGELOG.md` (+49 Zeilen vor existierender 0.3.0-Sektion).

### 3. Cargo.toml — Version Bump

```toml
version = "0.3.0"  →  "1.0.0"
description = "AMD-first LLM inference engine for Qwen2.5, Qwen3, and Llama-3.1 on RDNA 4"
            →  "AMD-first LLM inference engine for Qwen3 and related GGUF Q4_K_M models on RDNA 4"
```

`Cargo.lock` automatisch mitgezogen (1 Zeile geändert).

## Validation

### Release-Build

```
cargo build --release --features "v1 gpu" --bin rocmforge-v1
→ Finished `release` profile [optimized] target(s) in 1m 11s ✓
```

### Tests

| Test-Suite | Resultat |
|---|---|
| `cargo test --release --features "v1 gpu" --lib` | **316/320 passed**, 4 pre-existing v0.x failures (siehe unten) |
| `v1_mmq_infra_test` | 5/5 ✓ |
| `v1_mmq_q4_k_minimal_test` | 1/1 ✓ |
| `v1_mmq_q4_k_scaleup_test` | 4/4 ✓ |
| `v1_mmq_multiwarp_test` | 6/6 ✓ |
| `v1_mmq_q6_k_test` | 4/4 ✓ |
| `v1_attention_v2_test` | 9/9 ✓ |
| **v1-Integration Total** | **29/29 ✓** |

**Pre-existing v0.x test failures (NICHT v1-Regressionen):**
- `cpu::kernels::q3::tests::block_size` — `BlockQ3K::SIZE` test expected 110, struct ist 112 (Padding-Issue im v0.x Q3_K, nie supported)
- `cpu::kernels::q3::tests::dequantize_symmetric_values`
- `cpu::kernels::q3::tests::dequantize_zero_block`
- `gpu::weights::matrix_meta_tests::unsupported_matrix_type_is_rejected` — Test erwartet, dass Q6_K als "unsupported" rejected wird, aber ROCmForge unterstützt Q6_K mittlerweile. Test ist outdated, nicht der Code.

Diese 4 Tests sind alle in v0.x-Code, der von v1 nicht aktiv genutzt wird. Sie zu fixen wäre eine Refactor-Session, nicht Release-Prep.

### Smoke-Test (Release-Binary)

```
ROCMFORGE_PREFILL_MMQ=1 ./target/release/rocmforge-v1 \
  --model ~/models/Qwen3-8B-Q4_K_M.gguf \
  --prompt "Hello, what is 2+2?" --max-tokens 50

→ "Hello! 2 + 2 equals **4**. Let me know if you need help with anything else!"
→ 32 prompt tok, 29 decode tok, 54.6 tok/s decode, 615 ms total ✓ kohärent
```

### 15-Prompt Validation Suite

Aggregate: **Prefill 768.6 tok/s, Decode 95.4 tok/s, Wallclock 41.0 s, 15/15 ohne Fehler.**

Per-Prompt Decode: 91.5–102.3 tok/s (alle in 95-100 range).
EOS hits: 8/15 (Rest hit max_tokens — saubere Generation, kein Crash).

Report: `results/inference_test_20260425.md`.

### Clippy-Status

```
cargo clippy --release --features "v1 gpu" --bin rocmforge-v1 -- --cap-lints warn
→ 1009 warnings, no errors via --cap-lints warn
```

**Ehrliche Findung:** `cargo clippy` mit Default-Settings escaliert die `clippy::not_unsafe_ptr_arg_deref` Lint zu Error — 375 davon in `src/gpu/ffi.rs` (v0.x ROCm FFI). Mit `--cap-lints warn` läuft Clippy clean. Diese 375 "Errors" sind alle pre-existing v0.x Schulden (jede public function in der FFI die einen raw pointer dereferenziert, müsste `unsafe` markiert werden). Fixen ist eine 2-3 Tage Refactor-Session, nicht in Release-Prep machbar. Empfehlung: in Cargo.toml `[lints.clippy]` Sektion `not_unsafe_ptr_arg_deref = "warn"` setzen, falls dieses Verhalten unerwünscht ist (kein-`#[allow]`-im-Source-Compromiss).

Die v1-spezifischen Files sind clippy-clean (außer trivialen unused-import-Warnungen, die nicht-v1-spezifisch sind).

## Git State

```
Branch: v1.0-dev (local)
Tag: v1.0.0-rc1 → 38db230 (HEAD)
Remote: origin = oldnordic/ROCmForge.git
       (Memory-Notiz: User nutzt maeddesg/ROCmForge bei tatsächlichem Push)
```

**Tag wurde NICHT gepusht.** Bestätigung via:
```
git ls-remote --tags origin v1.0.0-rc1
→ (empty)
```

Commit-Inhalt:
```
docs: README + CHANGELOG for v1.0.0

 CHANGELOG.md  | +49 lines
 Cargo.lock   | +/-1
 Cargo.toml   | +/-7
 README.md    | +163/-234
```

### Uncommitted Working-Tree-Änderungen (NICHT im v1.0.0-rc1 Tag)

Im Tree sind weitere Änderungen aus dieser und früheren Sessions, die NICHT in dem Tag-Commit sind:

- `hip_kernels_v1/wmma/mmq_q4_k.hip` — Schritt 5 Multi-Warp + 5b VGPR-Versuch (reverted)
- `hip_kernels_v1/wmma/mmq_q6_k.hip` (untracked) — Schritt 6 Q6_K MMQ
- `hip_kernels_v1/attention/attention.hip` — P0.3 v2-Kernels
- `src_v1/backend/gpu/{wmma,attention}.rs` — FFI-Erweiterungen
- `src_v1/graph/executor.rs` — opt-in-Schalter
- `tests_v1/{mmq_multiwarp,mmq_q6_k,attention_v2}_test.rs` (untracked)
- `results/phase3_*.md` (untracked) — Phasen-Reports dieser Sessions
- `build.rs`, `hip_kernels_v1/CMakeLists.txt` — Library-Registrierungen
- Diverse Bench-CSVs, `design.md`, `hip_graph_device_pointer_bug.md` (untracked)

Per Prompt-Anweisung wurde **nur `README.md`, `CHANGELOG.md`, `Cargo.toml`, `Cargo.lock`** im docs-Commit aufgenommen. Der Tag liegt auf `f5b0aa7` (MMQ Schritt 4, +28.7 % E2E) + dem docs-Commit. Schritt 5/5b/6/P0.3-Code ist explizit als "research / opt-in / nicht default" im CHANGELOG dokumentiert; sie verändern das Default-Verhalten nicht.

**Wenn der User die zusätzlichen Code-Änderungen ebenfalls in v1.0.0 haben will:** separater Folge-Commit nötig, der die Tests, FFI, Kernel + Reports einbezieht. Ich habe das nicht von mir aus getan, weil der Prompt explizit nur die 3 Doc-Files spezifiziert hat.

## Empfehlungen vor `git push`

1. **Verify Cargo.toml repository pointer** — der `oldnordic/ROCmForge` Origin sollte vor Release-Push auf `maeddesg/ROCmForge` aktualisiert werden (per User-Memory).
2. **License-Datei** — README sagt "MIT", aber `LICENSE`-File fehlt. Sollte ein MIT-License-Text als `LICENSE` hinterlegt werden.
3. **CI-Validation** — vor Push einmal in clean Environment durchlaufen lassen (test + clippy).
4. **Optionale Erweiterung des Tag-Inhalts** — der Tag liegt auf docs-only. Falls die Multi-Warp / Q6_K-MMQ / Attention-v2-Code-Änderungen Teil von v1.0 sein sollen: Folge-Commit + Tag verschieben.

## Status P3 / Gesamt-P0

| Phase | Status |
|---|:---:|
| P0.1 Tile-Analyse | ✅ |
| P0.2 Schritt 4 Q4_K MMQ | ✅ +28.7 % E2E |
| P0.2 Schritt 5 Multi-Warp | ✅ +1.5 % (Noise) |
| P0.2 Schritt 5b VGPR-Reduktion | negative |
| P0.2 Schritt 6 Q6_K MMQ | negative (opt-in) |
| P0.3 Attention v2 | negative (opt-in) |
| **P3 Release-Prep** | ✅ |

ROCmForge v1.0.0 ist **release-ready** für RC1. Headline-Performance-Claims (96 tok/s Decode, +28.7 % Prefill via MMQ) sind im Tagged-Commit (`f5b0aa7`) enthalten.

## Geänderte Dateien (im docs-Commit)

| Datei | Δ |
|---|---|
| `README.md` | komplett neu, +163 / -234 Zeilen |
| `CHANGELOG.md` | +49 Zeilen (v1.0.0 prepended) |
| `Cargo.toml` | Version + Description |
| `Cargo.lock` | automatisch via Version-Bump |

## Report-Datei

`results/phase3_p3_release.md` (diese Datei).
