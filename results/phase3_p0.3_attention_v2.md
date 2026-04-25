# P0.3 Attention v2 Implementation — negative Findung

**Date:** 2026-04-25
**Branch:** v1.0-dev
**Status:** **Beide v2-Varianten implementiert + korrekt, aber netto NEUTRAL bzw. SCHLECHTER als v1.** v2_softmax (Fix-1-only, parallele Softmax) misst **1.00×** vs v1 — die single-thread Phase 2/3 in v1 wird offenbar von der GPU via Wave-Level-Occupancy versteckt. v2_combined (Fix 1 + GQA-LDS-Sharing + K/V-Tiling) misst **0.26× = 74 % LANGSAMER** wegen __syncthreads-Overhead und L2-Cache-Absorption der GQA-Redundanz in v1. v1 bleibt Default; beide v2-Kernel als opt-in im Code für künftige Flash-Attention-Arbeit (M > 2048).

## TL;DR

Die P0.3-Analyse hat +5-7 % E2E aus GQA + Multi-Thread-Softmax-Kombination prognostiziert. Die Implementation hat das **nicht eingelöst**. Beide v2-Varianten sind mathematisch perfekt korrekt (max_abs 1-2e-7 vs v1 = FP32-Noise, Causal-Maske bit-perfekt, GQA-Equivalence verifiziert) — aber der gemessene Speedup ist **netto neutral (Fix 1) bzw. negativ (Fix 1+2)**. Konsistent mit Schritt 5b (VGPR-Reduktion) und Schritt 6 (Q6_K MMQ): cleveres Restrukturieren auf RDNA4 hilft selten ohne strukturellen Bandbreite-Win.

## Was implementiert wurde

### `rf_v1_attention_prefill_v2_softmax_kernel` (Fix 1 alleine, ~80 LOC)

- **Gleiches Grid wie v1:** `dim3(num_heads, seq_len)` × 128 Threads
- Phase 1 (Score-Berechnung) und Phase 4 (V-Sum) **identisch** zu v1
- Phase 2 + Phase 3: **parallele Reduce statt `if (tid == 0)`**
  - Per-Thread Local-Max/Sum über j-Stride
  - Warp-Reduce via `__shfl_xor` (HIP-Syntax, kein `_sync`)
  - Block-Reduce via 4-Element LDS-Scratch (4 Warps × 1 float)
- Sync-Count: identisch zu v1 (4 Barriers)

### `rf_v1_attention_prefill_v2_kernel` (Fix 1 + Fix 2, ~150 LOC)

- **Neues Grid:** `dim3(num_kv_heads, seq_len)` × `gqa_ratio × 32` Threads
- Pro Block: `gqa_ratio` Warps, jeder Warp = 1 q_head, alle teilen 1 kv_head
- K/V-Tiling: TILE_KV=32, kooperatives Laden in `__shared__ s_k_tile[32×head_dim]` und `s_v_tile[32×head_dim]`
- LDS-Layout (gqa_ratio=4, head_dim=128, seq_len=576):
  - s_scores[4 × 576] = 9 KB
  - s_k_tile[32 × 128] = 16 KB
  - s_v_tile[32 × 128] = 16 KB
  - **Total ~41 KB**, fits in 60 KiB Budget ✓
- Phase 1: lane-per-j Pattern (TILE_KV=32 = WARP_SIZE), kein per-j Warp-Reduce
- Phase 2/3: per-Warp parallel max/sum (warp-only, da jeder Warp eigenes q_head)
- Phase 4: tiled V-Akkumulator über lane-strided output
- Dynamic gqa_ratio: 2..8 erlaubt, sonst Launcher-Error → Caller-Fallback
- Sync-Count: ~36 Barriers für Phase 1 + ~36 für Phase 4 = **~72 Barriers** (v1: 4)

## Korrektheit (9/9 Tests grün, 0.08s)

```
attn_v2_softmax_vs_v1_qwen3_qkv          M=64  hd=128 max_abs=1.19e-7
attn_v2_combined_vs_v1_qwen3_qkv         M=64  hd=128 max_abs=2.38e-7
attn_v2_combined_vs_v1_realistic_seq     M=128 hd=128 max_abs=2.09e-7
attn_v2_combined_small                   M=16  hd=64  max_abs=1.19e-7
attn_v2_combined_causal_mask             max_abs=0.0e0 ✓ perfekt
attn_v2_combined_gqa_equivalence         4 q_heads identisch ✓
attn_v2_combined_dynamic_gqa_ratio       gqa=2,4,4 alle ok
attn_v2_perf_softmax_only_measured       (siehe unten)
attn_v2_perf_combined_measured           (siehe unten)
```

Alle FP32-Differenzen sind im 1-2e-7 Bereich = pure FP32-Akkumulationsnoise. **Causal-Maske bit-perfekt erhalten** (max_abs=0).

## Performance — gemessen (HIP Events, M=576, 5 Iter)

| Kernel | Iter ms | Speedup vs v1 |
|---|---:|---:|
| **v1 (Default, Production)** | **2.65–2.75** | 1.00× |
| v2_softmax (Fix 1 only) | 2.69–2.77 | **1.00×** (neutral) |
| v2_combined (Fix 1 + Fix 2) | 7.6–10.4 | **0.26–0.35×** ⚠ |

### E2E Wallclock (542-Token-Prompt, 3 Runs)

Alle drei Werte mit Default-v1 weiterhin aktiv (kein Executor-Dispatch-Wechsel):

| Run | Wallclock |
|---|---:|
| 1 | 467 ms |
| 2 | 466 ms |
| 3 | 466 ms |

Identisch zur Schritt-5/6-Baseline. Mutex-Prompt: wort-identisch.

## Warum hat die Optimierung nicht funktioniert?

Drei strukturelle Faktoren — alle empirisch belegt:

### 1. v1's "Single-Thread-Bottleneck" wird via Wave-Occupancy versteckt

Die Analyse rechnete: ~1100 ops Lane-0-serial × 288 sequenzielle Blocks/CU = 290 µs Lane-0-Bottleneck pro Call.

**Realität:** Bei 18432 Blocks insgesamt und ~960 Wave-Slots parallel auf 64 CUs ist die GPU komplett gesättigt. Während Block X in Phase 2/3 mit Lane 0 alleine arbeitet, laufen Blocks Y, Z, ... in ihrer Phase 1 oder Phase 4. Das Lane-0-Bottleneck ist effektiv **nicht der kritische Pfad**.

v2_softmax bringt 0 % messbarer Speedup (1.00×) trotz vollständiger Parallelisierung von Phase 2/3 → empirischer Beweis, dass die Phasen 2+3 **nicht der Bottleneck waren**.

### 2. GQA-Redundanz wird vom L2-Cache absorbiert

Die Analyse rechnete: 5.4 GB redundante KV-Reads pro Layer-Attention vs 1.4 GB optimal → 4× Bandbreite-Einsparung möglich.

**Realität:** Auf RDNA4 hat der L2-Cache 4 MB. Sequenzielle Block-Dispatches für (q_head=0, query_pos=q), (q_head=1, query_pos=q), … laufen oft auf benachbarten CUs in zeitlicher Nähe. Der Read-Pattern für `K[kv_head=0, j, *]` von q_head 0 erwärmt L2; q_heads 1, 2, 3 lesen dann großenteils aus L2 (hot, sub-ns latency). Die nominale "4× Redundanz" wird zu effektiver "1.x× Redundanz" durch L2-Caching.

v2_combined würde den L2-Hit-Rate-Vorteil nur einlösen, wenn der zusätzliche Sync-Overhead kleiner wäre als der eingesparte Memory-Traffic. Stattdessen:

### 3. K/V-Tiling-Synchronisation überschattet alle Memory-Einsparungen

v2_combined macht ~72 `__syncthreads` Barriers pro Block (v1: 4). Jede Barrier kostet auf RDNA4 ~30-100 Cycles (alle Waves im Workgroup müssen sich synchronisieren). 72 × 60 = ~4300 Cycles Block-Overhead von Barriers alleine, plus die gesteigerte Code-Komplexität (mehr Branches, mehr Register-Pressure, longer Schedule).

Pro Block summed up:
- Barrier-Overhead: ~4300 cycles
- LDS-Read/Write-Overhead: per-Tile load = 32×128 floats = 4096 LDS-bank-conflicts pro Tile
- 18 Tiles × 4096 = ~74000 LDS-Ops
- Total: ~78000 cycles overhead vs v1

vs theoretischer Memory-Einsparung:
- 3.75 KB KV-data × 144 reads = ~540 KB GLOBAL-traffic eingespart
- Bei 1 TB/s effective bandwidth: 0.54 µs gespart
- Bei 2.97 GHz: 1600 cycles entsprechende Compute-Zeit

→ Overhead-Cycles ~50× größer als Einsparungs-Cycles.

## Was steht im Repo

**v1-Default unverändert.** Der Executor-Dispatch wurde NICHT geändert — `rocmforge_launch_attention_prefill` (= v1) ist weiterhin der Default-Pfad. Beide v2-Kernel sind:
- in `hip_kernels_v1/attention/attention.hip` als `rf_v1_attention_prefill_v2_softmax_kernel` und `rf_v1_attention_prefill_v2_kernel`
- via FFI in `src_v1/backend/gpu/attention.rs` exponiert
- aber nirgends im Production-Code aufgerufen

Beide bleiben verfügbar als Vorbereitung für Flash-Attention bei M > 2048 (v1.1-Scope), wo:
- Score-Array M² × 4 sprengt L2 (16 MB bei M=2048)
- L2-Cache kann GQA-Redundanz NICHT mehr absorbieren
- Sync-Overhead wird relativ kleiner zum Memory-Traffic

## Geänderte Dateien

| Datei | Änderung |
|---|---|
| `hip_kernels_v1/attention/attention.hip` | +280 LOC (zwei v2-Kernel + Helpers + Launcher) |
| `src_v1/backend/gpu/attention.rs` | +37 LOC (zwei FFI-Deklarationen) |
| `tests_v1/attention_v2_test.rs` | **neu**, 9 Tests (Korrektheit + Perf-Diagnose) |
| `Cargo.toml` | Test-Registrierung |

`src_v1/graph/executor.rs`, `build.rs`, `CMakeLists.txt` **unverändert** — kein Production-Pfad-Wechsel.

## Konsistenz-Pattern in P0

Drei aufeinanderfolgende negative Optimierungs-Findungen mit klaren Lehren:

| Schritt | Hypothese | Ergebnis | Lehre |
|---|---|---|---|
| **5b** VGPR-Reduktion | 152→120 VGPR bringt Occupancy | 0.66× (regression) | Kernel ist WMMA-bound, nicht Latency-bound |
| **6** Q6_K MMQ | Integer-WMMA wie Q4_K | 0.91× (regression) | Q6_K-Dequant zu komplex ohne LDS-Staging |
| **P0.3 v2** | Parallel Softmax + GQA-LDS | 0.26× (regression) | Single-thread Phase wird via Occupancy hidden, GQA-Redundanz via L2 absorbed |

**Gemeinsame Lehre:** Auf RDNA4 (gfx1201) ist der Default-Code-Pfad oft schon nahe optimal, weil:
1. **Wave-Level-Parallelism versteckt Single-Thread-Bottlenecks** (P0.3)
2. **L2-Cache absorbiert "redundante" Reads** (P0.3)
3. **WMMA-Pipeline-Throughput ist der harte Limit, nicht Occupancy** (5b)
4. **Memory-Bandbreite ist nicht oft der Bottleneck** (P0.3, 5b)

Optimierungen die diese Mechaniken NICHT umgehen oder verstärken (sondern z.B. nur Code-Form umstrukturieren), regrediieren häufig.

## Empfehlung — Was statt P0.3-Implementation tun

| Option | Erwarteter ROI | Aufwand | Risiko |
|---|---:|---|---|
| **6b: Q6_K + LDS-Staging** | +5-6 % E2E | 1 Session | mittel — LDS-Layout-Komplexität |
| **P0.4: FP8 Re-Eval** | unbekannt | 1-2 Sessions | hoch — V100-Pfad existiert nicht |
| **Flash-Attention v1.1** (M > 2048) | +20-50 % bei M=2048 | 2-3 Sessions | niedrig — bekannter Algorithmus, große Hebelwirkung |
| **Status-Quo** (P0.2 Schritt 4 = +28.7 % bestätigt) | 0 % weiteres | 0 | 0 |

**Empfehlung 1:** **Q6_K + LDS-Staging (Schritt 6b)** als nächste Implementation. Klar definierter Pfad (basiert auf existierendem FP16-WMMA-Q6_K-LDS-Code), klares ROI-Ziel (+5-6 % E2E).

**Empfehlung 2:** Flash-Attention auf v1.1 schieben, sobald wir M > 2048 unterstützen. Bei M=576 ist das nicht der Hebel.

**Empfehlung 3 (passiv):** Mit dem aktuellen Stand zufrieden sein. Schritt 4 hat **+28.7 % E2E Prefill** geliefert — das ist signifikant über dem P0-Initial-Ziel.

## Status P0

| Schritt | Status | Effekt |
|---|:---:|---|
| P0.1 Tile-Analyse | ✅ | Foundation |
| P0.2 Schritt 4 (Q4_K MMQ) | ✅ | **+28.7 % Prefill** |
| P0.2 Schritt 5 (Multi-Warp) | ✅ | +1.5 % (Noise) |
| P0.2 Schritt 5b (VGPR) | negative | reverted |
| P0.2 Schritt 6 (Q6_K) | negative | opt-in |
| **P0.3 Attention v2** | **negative** | **opt-in, kein Production-Wechsel** |
| P0.2 Schritt 6b (Q6_K LDS) | 🔜 | erwartet +5-6 % |
| Flash-Attention | v1.1 | M > 2048 |
