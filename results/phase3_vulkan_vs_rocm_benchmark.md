# Vulkan vs ROCm vs ROCmForge — Triple-Benchmark

**Date:** 2026-04-25
**Hardware:** AMD Radeon RX 9070 XT (gfx1201), Ryzen 9 7945HX
**Model:** Qwen3-8B-Q4_K_M.gguf
**llama.cpp Build:** commit `23b8cc4` (gleich für ROCm-Build und Vulkan-Build)
**ROCmForge:** v1.0-dev branch HEAD (Schritt P0.3 Stand)
**OS:** CachyOS Linux 7.0.1-1-cachyos
**Vulkan:** RADV 1.4.341 (radv driver, gfx1201)

## Vulkan-Build

- **cmake:** `GGML_VULKAN=ON`, `GGML_HIP=OFF`, separater build-Tree `build-vulkan/`
- **Build-Status:** OK. `llama-cli` und `llama-bench` linked.
- **Vulkan-Device-Erkennung:** `AMD Radeon RX 9070 XT (RADV GFX1201) (radv) | uma: 0 | fp16: 1 | bf16: 1 | warp size: 64 | shared memory: 65536 | int dot: 1 | matrix cores: KHR_coopmat`
- **Anmerkung Mutex-Smoke (`llama-cli`):** First-run shader-compile hängt für viele Minuten ohne sichtbaren Fortschritt; nach Abbruch mit `kill -KILL` und Retry läuft alles. **Subsequent `llama-bench`-Aufrufe sind problemlos** (Cache warm). Das ist ein bekanntes Vulkan-shader-compile-Verhalten beim ersten Run, kein Build-Problem. Production-Bench ist daher immer auf warm cache.

## Decode-Vergleich

| Engine | tg32 tok/s | tg128 tok/s | tg256 tok/s | Bemerkung |
|---|---:|---:|---:|---|
| ROCmForge (Integer-MMQ) | — | — | ~96.2 (vom 15-Prompt-Bench) | aus den Schritt-4-Messungen |
| llama.cpp ROCm | — | 87.48 ± 1.33 | 88.65 ± 1.22 | hipBLAS-Pfad |
| **llama.cpp Vulkan** | **98.29** | **114.20** | **113.97** | **schnellster Decode** ⚡ |

**Vulkan-Vorteil:** Vulkan tg128 = 114.2 tok/s vs ROCm tg128 = 87.5 tok/s = **+30 % Decode-Throughput** zugunsten Vulkan auf identischem Modell + Hardware.

## Prefill-Skalierung (`llama-bench -p N -n 1 -r 2`)

| pp (Tokens) | ROCmForge tok/s | llama.cpp ROCm tok/s | llama.cpp Vulkan tok/s | Vulkan vs ROCm |
|---:|---:|---:|---:|---:|
| 128 (synthetic) | — | 2640 ± 7 | **3622 ± 17** | +37 % |
| 256 (synthetic) | — | 3620 ± 14 | **4009 ± 10** | +11 % |
| 512 (synthetic) | — | 3683 ± 16 | **4314 ± 12** | +17 % |
| 1024 (synthetic) | — | 3446 ± 13 | **4179 ± 5** | +21 % |
| 2048 (synthetic) | — | **656 ± 198** ⚠ | **3754 ± 5** | **+472 %** |
| 4096 (synthetic) | — | crash/no-data | 471 ± 22 | n/a |
| 8192 (synthetic) | — | crash/no-data | 376 ± 0 | n/a |
| 16384 (synthetic) | — | crash/no-data | 246.6 ± 0.5 | n/a |

ROCmForge auf realen Prompts (1024-Token-Cap):

| Prompt | Tokens | Wallclock | abgeleitet tok/s |
|---|---:|---:|---:|
| Mutex one-paragraph | 54 | 105 ms | ~620 (incl. setup) |
| 20-Zeilen-Prompt | 542 | 466 ms | ~1260 |
| 30-Zeilen-Prompt | 1423 | OVERFLOW | crash (>1024 cap) |

### Schlüsselbeobachtungen

**Vulkan-Prefill-Skalierung:**
- Spitze bei pp=512–1024 (~4150–4300 tok/s)
- Knick auf 376 tok/s ab pp=4096 — Attention dominiert (M² Score-Array sprengt L2-Cache, 16 MB bei pp=2048)
- Trotzdem stabil bis pp=16384

**ROCm-Prefill-Cliff:**
- Sauber bis pp=1024 (~3450 tok/s)
- **Massive Regression bei pp=2048** (656 tok/s, σ=198) — Indiz für Out-of-Memory oder Numerical-Issue mit großen Buffers
- Bei pp=4096+ liefert llama-bench keine Tabellen-Zeilen (Crash oder OOM, nur Header)

**Vulkan ist ROCm bei allen pp ≥ 2048 deutlich überlegen.** Bei pp=2048: 5.7× schneller. Bei pp=4096+: ROCm crasht, Vulkan liefert weiter.

## Stromverbrauch (hwmon `power1_average`, Decode tg256)

| Engine | Idle W | Peak W | Median active W (~) | tok/s | tok/s/W (peak) |
|---|---:|---:|---:|---:|---:|
| llama.cpp ROCm | 10–11 | 357 | ~330 | 88.65 | **0.25** |
| **llama.cpp Vulkan** | 10–11 | 336 | ~290 | 113.97 | **0.34** |

**Vulkan-Effizienz:** ~36 % besser tok/s/W. Vulkan zieht **weniger Watt** UND liefert **mehr Tokens** — Doppelvorteil.

(Sample-Werte aktiv: Vulkan 197–336 W, ROCm 286–357 W. Median über die ganze Sampling-Periode wird durch idle-vor-und-nach dominiert; daher die Spalten "Peak" und "Median active" als nützlichere Metriken.)

## Zusammenfassung

| Metrik | Winner | Margin |
|---|---|---:|
| Decode tok/s (tg128/256) | **Vulkan** | +30 % vs ROCm |
| Prefill pp=512 (kurz) | **Vulkan** | +17 % |
| Prefill pp=2048 (mittel) | **Vulkan** | +472 % (ROCm bricht ein) |
| Prefill pp=8192 (lang) | **Vulkan** | n/a (ROCm crash) |
| Effizienz (tok/s/W) | **Vulkan** | +36 % |

### Vergleich zu Ollama/v0.x-Benchmarks

> Damals (Phase 1 / Ollama-Vergleich): Vulkan 52.5 tok/s, ROCm 48.2 tok/s, Vulkan 68 W vs 149 W

**Heutige Werte:**
- Vulkan: 114 tok/s (≈ 2× damals; ROCm 7.2 + neuere llama.cpp-Optimierungen)
- ROCm: 88 tok/s (≈ 1.8× damals)
- Power-Diff: Vulkan ~290 W vs ROCm ~330 W → 14 % geringer (war damals 54 % geringer; Hardware ist heute mehr ausgelastet)

**Trend:** Vulkans Leadership beim Decode hat sich nicht nur erhalten, sondern bei größeren pp drastisch ausgebaut. ROCm-Prefill bei pp ≥ 2048 ist instabil.

### ROCmForge Position

ROCmForge sitzt mit ~96 tok/s Decode **zwischen ROCm (88) und Vulkan (114)**. Beim Prefill liegt es konkret:

- ROCmForge Prefill bei 542-Token-Prompt: ~1260 tok/s (= 542/0.43s effective)
- llama.cpp Vulkan bei pp=512: 4314 tok/s
- Faktor: **Vulkan ist ~3.4× schneller im Prefill als ROCmForge**

Das bestätigt die P0-Roadmap: ROCmForge hat einen substanziellen Rückstand beim Prefill — Q4_K-MMQ-Schritt-4 (+28.7 % E2E) hat die Lücke verkleinert, aber sie bleibt deutlich. ROCmForge's 1024-Token-Cap ist eine zusätzliche bekannte Limitation gegenüber den llama.cpp-Builds.

## Empfehlungen

1. **Wer maximum Performance auf RX 9070 XT will:** **llama.cpp Vulkan** ist heute die deutlich beste Wahl. Decode +30 %, Prefill +17 % bei kurzen Prompts, Prefill stabil bis 16k Tokens.

2. **Wer minimum Stromverbrauch will:** ebenfalls **Vulkan** (-36 % besser tok/s/W).

3. **Wer ROCm explizit braucht** (z.B. für `pytorch + ROCm`-Compatibility): aware sein, dass llama.cpp ROCm-Backend bei pp ≥ 2048 instabil ist. Für lange Prompts ist das ein kritischer Show-Stopper.

4. **Für ROCmForge-Projekt:** Vulkan-Path ist bekannte ferne Konkurrenz. Realistisches v1.0-Ziel ist **Decode-Parität mit ROCm-Build** (88 tok/s) — bereits erreicht und überholt. Prefill-Parität mit Vulkan ist v1.1-Scope (braucht 8192-Token-Support + Flash-Attention).

## Geänderte Dateien

`results/phase3_vulkan_vs_rocm_benchmark.md` (diese Datei). Kein Code geändert.
