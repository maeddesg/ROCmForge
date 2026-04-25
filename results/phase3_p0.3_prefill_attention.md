# P0.3 — Prefill-Attention Analyse (keine Implementation)

**Date:** 2026-04-25
**Branch:** v1.0-dev
**Status:** **Analyse-Phase abgeschlossen.** Der existierende Prefill-Attention-Kernel hat DREI klar identifizierte Pessimierungen: (1) naiv O(M²) mit komplettem Score-Array, (2) **Single-thread Softmax** in Phase 2 und 3 (`if tid == 0` über ~q ops), (3) **GQA-redundante KV-Reads** (4×). Empfehlung der Entscheidungs-Matrix: **GQA-LDS-Sharing + Multi-Thread-Softmax kombiniert** (~150-300 LOC, 1 Session, geschätzt +30-40 % Kernel = +5-7 % E2E). Flash-Attention bleibt v1.1-scope. Implementation NICHT in dieser Session.

## TL;DR

Der aktuelle `rf_v1_attention_prefill_kernel` ist **schwächer als der Prompt befürchtet hat**. Es ist nicht nur "naiv O(M²)" — die Phasen 2 (max-Reduktion) und 3 (exp + sum) laufen explizit `if (tid == 0)` als reiner Single-Thread-Loop. Bei query_pos=575 sind das je ~575 sequenzielle Ops auf Lane 0 während die anderen 127 Threads idlen. Die GQA-Redundanz ist die zweite Pessimierung. Beide zusammen erklären die 85 ms / 2367 µs Avg-Call-Zeit.

**Quick-Win-Potential ist DEUTLICH größer als der Prompt-ROI von "3-5 % E2E"** — eine Multi-Thread-Softmax + GQA-LDS-Sharing-Kombination dürfte +30-40 % Kernel-Speedup bringen (~+5-7 % E2E), bei moderatem Aufwand (~150-300 LOC, 1 Session).

## Prüfpunkt 1 — Tiling / Softmax / Score-Array

Quelle: `hip_kernels_v1/attention/attention.hip:150-224` (`rf_v1_attention_prefill_kernel`).

### A. Algorithmus

**Naiv O(M²)**, kein Flash-Attention. Pro `(q_head, query_pos)`-Block:

```
extern __shared__ float s_scores[];   // visible × float, in LDS

Phase 1: parallel über tid=0..127:
  for j in tid..visible step 128:
    s_scores[j] = dot(q[query_pos, q_head], k[j, kv_head]) * scale

Phase 2: SINGLE-THREAD (tid == 0):
  m = -INF
  for j in 0..visible:
    if s_scores[j] > m: m = s_scores[j]
  s_max = m

Phase 3: SINGLE-THREAD (tid == 0):
  sum = 0
  for j in 0..visible:
    s_scores[j] = exp(s_scores[j] - s_max)
    sum += s_scores[j]
  s_sum = sum

Phase 4: parallel über tid=0..127:
  for d in tid..head_dim step 128:
    out[d] = (Σ_j s_scores[j] * v[j, kv_head, d]) / s_sum
```

### B. Softmax

**Offline Softmax** (3 Phasen). Kein Online-Softmax (Milakov & Gimelshein) implementiert.
- `s_scores[]` ist das **vollständige Score-Array für visible Tokens** — keine Tile-Reduktion.
- LDS-Budget: `seq_len × 4 Bytes`. Cap auf 48 KiB → max ~12000 Tokens.

### C. Sequenz-Tiling

**Kein K/V-Tiling.** Phase 1 iteriert die volle K-Sequenz, Phase 4 iteriert die volle V-Sequenz pro `d`. Score-Array hält die ganze Zeile.

### Konsequenz

Score-Array passt bei M=576 mit 2.3 KB locker in LDS — kein Memory-Druck. **Aber:** die Phasen 2+3 sind ein **Pure-Serial-Bottleneck**. Bei query_pos=575 macht Lane 0 zwei sequenzielle Loops über 576 Elemente (1150 ops sequenziell), 127 Lanes idlen. Bei seq_len=576 mit Avg-visible=288 sind das ~576 ops × 5 Cycles = ~3000 Cycles serial pro Block, × ~288 sequenziellen Blocks/CU (18432 Blocks total, 64 CUs) ≈ 290 µs Lane-0-Bottleneck pro Call. **Das ist ~12 % des 2367 µs Avg.**

## Prüfpunkt 2 — GQA-Handling

### Befund: 4× redundante KV-Reads (Option A)

Code:
```cpp
const int kv_head = q_head / (num_heads / num_kv_heads);
// Qwen3-8B: 32 / 8 = 4 → q_head 0,1,2,3 alle map auf kv_head=0
```

Grid:
```cpp
dim3 grid(num_heads, seq_len, 1);
// Block (q_head, query_pos) = (0..31, 0..575)
```

**Konsequenz**: Die 4 Blocks `(q_head=0..3, query_pos=q)` haben identisches `kv_head=0` und lesen identische K/V-Daten aus Global Memory — vier separate, unabhängige Block-Instanzen ohne LDS-Sharing.

### Bandbreite-Rechnung

Pro Block bei query_pos=q (Avg q=288 für M=576):
- K read: visible × head_dim × 4 = ~290 × 512 = ~148 KB
- V read: gleich = ~148 KB
- Per Block: ~290 KB (Q ist klein, ~512 B)

Total Block-Count: 32 q_heads × 576 query_pos = 18432 Blocks
Total reads pro layer: 18432 × 290 KB ≈ **5.4 GB** (mit Redundanz)
Ohne Redundanz (LDS-shared): 8 kv_heads × 576 query_pos × 290 KB ≈ **1.4 GB**

Über 36 Layer: **~190 GB redundant** vs ~50 GB optimal → **140 GB redundante L2-Reads pro Prefill**.

L2-Cache hilft (4 MB auf RDNA4, hot KV cached über kurzes Zeitfenster), aber sub-optimal. **GQA-Sharing via LDS würde 75 % der KV-Reads eliminieren** (von 4 q_heads auf 1 effektiver Read).

### llama.cpp-Vergleich (`fattn.cu:454-478`)

llama.cpp hat einen expliziten `gqa_opt_applies` Branch für RDNA4 + GQA, der zu `BEST_FATTN_KERNEL_MMA_F16` oder `BEST_FATTN_KERNEL_TILE` dispatcht. **Sie haben einen GQA-bewussten Pfad.** Unser Kernel hat das nicht.

## Prüfpunkt 3 — Flash-Attention ROI bei M=576

### Memory-Footprint

| M | Score-Array (per Head) | L2 (4 MB) Pressure |
|---|---:|---|
| 576 | 1.3 MB | passt locker |
| 2048 | 16 MB | sprengt L2 |
| 8192 | 256 MB | DRAM-bound |

Bei M=576 ist das Score-Array **kein Memory-Problem**. Flash-Attentions Hauptvorteil (O(M²) → O(M × tile_size) Memory-Footprint) bringt hier wenig.

### ROI-Schätzung

| Optimierung | Aufwand | Erwarteter Kernel-Speedup | E2E-Anteil |
|---|---|---:|---:|
| **GQA-LDS-Sharing** | 100-200 LOC, 1 Session | **+20-30 %** | **+4-6 %** |
| **Multi-Thread-Softmax** | 30-50 LOC, 0.5 Session | +10-15 % | +2-3 % |
| Beide kombiniert | 150-300 LOC, 1 Session | **+30-40 %** | **+5-7 %** |
| Flash-Attention-Port | 500-800 LOC, 2-3 Sessions | +10-20 % | +2-4 % |
| Q6_K LDS-Staging (Schritt 6b) | 200-400 LOC, 1 Session | +37 % auf 67 ms | +5-6 % |

**GQA + Multi-Thread-Softmax kombiniert hat den BESTEN ROI** der Optionen.

## Entscheidungs-Matrix Anwendung

| Befund | Empfehlung |
|---|---|
| Naiv O(M²) UND GQA-redundant UND Single-thread Softmax | **GQA-Fix + Multi-Thread-Softmax kombiniert** |

Begründung:
- Quick-Win + struktureller Fix: 1 Session, ~150-300 LOC, +5-7 % E2E
- Größerer Hebel als Flash-Attention bei M=576
- Decode-Kernel wird NICHT angefasst (separater Code-Pfad)
- Korrektheit ist verifizierbar via Causal-Mask-Test + GQA-Equivalence-Test

Flash-Attention bleibt **explizit v1.1**, sobald wir M > 2048 unterstützen.

## Konkreter Implementierungs-Vorschlag (NICHT in dieser Session implementiert)

### Schritt 1 — Multi-Thread-Softmax (einfach, hoher Reward/Aufwand)

Phase 2: warp-Reduktion für max
```cpp
// Phase 2: alle 128 threads finden ihren Local-Max via reduction.
float local_max = -INFINITY;
for (int j = tid; j < visible; j += blockDim.x) {
    local_max = fmaxf(local_max, s_scores[j]);
}
// Block-weite Reduktion via __shfl_xor + LDS
__shared__ float s_block_max;
local_max = block_reduce_max(local_max);  // standard pattern
if (tid == 0) s_block_max = local_max;
__syncthreads();
const float s_max = s_block_max;
```

Phase 3 analog: jeder Thread macht seinen Anteil von exp + Local-Sum, dann block-reduction.

**Erwarteter Effekt:** Lane-0-Serial-Bottleneck weg. Phase 2+3 von ~575 ops/Lane-0 auf ~5 ops/Lane × Reduktion = ~50 ops total. Geschätzt 10-15 % Kernel-Speedup.

### Schritt 2 — GQA-LDS-Sharing (größerer Reward, mehr Code)

**Restruktur:** Grid wird `dim3(num_kv_heads, seq_len, 1)`, jeder Block bearbeitet **4 q_heads pro kv_head gemeinsam**. Block-Setup:
- 1 Block lädt K-row und V-row für `kv_head` einmal in `__shared__ s_k[head_dim]`, `s_v[head_dim]`.
- Innerhalb des Blocks: 4 Warps (128 Threads = 4 × 32) — jeder Warp bearbeitet 1 q_head.
- Alle 4 Warps lesen aus shared `s_k`/`s_v`.

LDS-Budget pro Block: `seq_len * (4 + 1 + 1) * 4 ≈ seq_len × 24 Bytes` (4 q_head scores + s_k + s_v Strecken).
Bei seq_len=576: 576 × 24 = 13.8 KB → fits in 64 KB LDS budget.

**Erwarteter Effekt:** 75 % der KV-Reads eliminiert. Geschätzt 20-30 % Kernel-Speedup.

### Kombinierte Form

Der Multi-Thread-Softmax ist eine prerequisite für die LDS-Sharing-Variante (4 Warps müssen kooperativ summieren). Beide Schritte zusammen sind ~150-300 LOC.

## Empfehlung — Was als nächstes tun

**Option 1 (empfohlen):** Schritt P0.3 als Implementation-Session aufsetzen (1 Session) und GQA + Multi-Thread-Softmax kombiniert umsetzen. Erwarteter Gewinn: +5-7 % E2E Prefill (befundbasiert höher als der Prompt-Initial-ROI von 3 %).

**Option 2:** Q6_K LDS-Staging (Schritt 6b) parallel — etwa gleicher ROI (+5-6 %), unabhängig.

**Option 3:** Beides nacheinander.

Flash-Attention v1.1 (M > 2048).

## Was NICHT empfohlen wird

- **Option C (STOP)**: nicht angemessen, weil der aktuelle Kernel ungewöhnlich pessimiert ist (Single-thread Softmax + GQA-Redundanz). 3 unabhängige Probleme gleichzeitig zu fixen lohnt sich.
- **Flash-Attention bei M=576**: Aufwand-Gewinn-Verhältnis schlecht. Score-Array passt in L2.
- **Decode-Kernel anfassen**: separates Profil (M=1, wachsendes seq_len), separater Optimierungspfad.

## Anhang — Code-Pointer

| Was | Wo |
|---|---|
| Prefill-Kernel | `hip_kernels_v1/attention/attention.hip:150-249` |
| Decode-Kernel | `hip_kernels_v1/attention/attention.hip:33-132` |
| KV-Cache-Append | `hip_kernels_v1/attention/attention.hip:256-295` |
| Executor-Dispatch | `src_v1/graph/executor.rs:2390-2419` |
| llama.cpp RDNA4-Dispatch | `~/tmp/llama.cpp/ggml/src/ggml-cuda/fattn.cu:454-478` |
| Phase-1-Baseline (85.2 ms, 36 calls) | `results/phase3_p0.1_wmma_tile_tuning.md` |

## Status P0

| Schritt | Status | Effekt |
|---|:---:|---|
| P0.1 Tile-Analyse | ✅ | Foundation |
| P0.2 Schritt 4 (Q4_K MMQ) | ✅ | **+28.7 % Prefill** |
| P0.2 Schritt 5 (Multi-Warp) | ✅ | +1.5 % (Noise) |
| P0.2 Schritt 5b (VGPR) | negative | reverted |
| P0.2 Schritt 6 (Q6_K) | negative | opt-in only |
| **P0.3 Prefill-Attention** | **Analyse done** | erwartet +5-7 % bei Implementation |
| P0.2 Schritt 6b (Q6_K LDS) | 🔜 | erwartet +5-6 % |
| P0.4 FP8 | 🔜 | unbekannt |
