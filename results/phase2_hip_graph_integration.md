# HIP-Graph Decode Integration — Option C (ehrliches Ergebnis)

**Date:** 2026-04-23
**Branch:** v1.0-dev (auf sudot4 `d3aacdd`)
**Hardware:** AMD Radeon RX 9070 XT (gfx1201)
**ROCm:** runtime 70253.21, driver 70253.21
**Scope:** Capture des Decode-Forward-Passes einmal, Replay mit
`hipGraphExecKernelNodeSetParams` für die 3 token-variablen
Parameter (`kv_pos` in Rope + KvCacheAppend, `seq_len` in
Attention). Option C aus der Scope-Analyse.

## TL;DR

```
Scope-Check:
  __builtin HIP-Graph API verfügbar auf ROCm 7.2.2 / gfx1201
  (Get/Set/Instantiate/Launch/Destroy alle getestet) ✅
  Eskalationstest L0-L6 hat bewiesen dass der Pfad stabil ist ✅

Implementation:
  Capture beim ersten Decode nach Bandit-Konvergenz
  kv_pos + seq_len als Box<i32>-Heap-Slots
  Replay: hipGraphKernelNodeGetParams → Args kopieren +
          patchen → hipGraphExecKernelNodeSetParams → Launch
  Fallback: ROCMFORGE_DISABLE_HIP_GRAPH=1 env-Toggle
           + automatischer Rollback bei Capture-Fehler

Korrektheit: ✅ Output BIT-IDENTISCH zu legacy (78/78 Tokens match)
Perf-Gate (≥ 0.95× Decode-Rate): ✅ (+0.7 % A/B)
Perf-Ziel (+5 %): ❌ (wie projiziert — siehe Ehrliche Analyse)
```

## Warum der Speedup klein ist (war projiziert)

Aus der Scope-Analyse der vorherigen Session:

```
Wall-clock pro Token: 16.8 ms
  GPU-Kernel-Zeit:       13.7 ms  (83 %)  ← BW-limited, HIP-Graph hilft null
  hipLaunchKernel:        0.5 ms  (3 %)   ← vom HIP-Graph eliminiert
  HIP-Driver-Interna:     1.0 ms  (6 %)   ← vom HIP-Graph reduziert
  hipMemcpy Logits:       0.6 ms  (4 %)   ← bleibt
  hipStreamSync:          0.5 ms  (3 %)   ← bleibt
  Rust-CPU-Pfad:          0.3 ms  (2 %)   ← größtenteils bleibt
  Sampling + Monitor:     0.2 ms  (1 %)   ← bleibt

  Max-Gewinn aus HIP-Graph: ~0.5-0.8 ms / 16.8 ms = 3-5 %
```

Nun ist der GEWONNENE Overhead durch den **SetParams-Call-Stack** vor
jedem Replay ersetzt:

```
Pro Replay patche ich 108 Nodes (36 rope × 2 + 36 kv_write + 36 attn):
  108 × hipGraphKernelNodeGetParams  (~300 ns)
  108 × hipGraphExecKernelNodeSetParams (~500 ns)
  = ~86 µs per Token

Überhang gegenüber ursprünglichen 599 hipLaunchKernel-Calls:
  599 × 879 ns = 527 µs  (legacy)
  1 × hipGraphLaunch + 108 × SetParams ≈ 90 µs
  Δ ≈ 440 µs/Token  (theoretisch)

Tatsächlich gemessen: +0.7 %, entspricht ~120 µs Gewinn.
```

Die 440 µs-Projektion ist das **obere Limit**; real dominieren
Driver-Overhead-Pfade die unter der 2 %-Schicht des vorherigen
Deep-Dives liegen aber messbar bleiben.

## Architektur

### `HipGraphDecodeCache`

```rust
pub struct HipGraphDecodeCache {
    graph: hipGraph_t,
    exec: hipGraphExec_t,
    kv_pos_slot: Box<i32>,          // heap-stable
    seq_len_slot: Box<i32>,          // heap-stable
    rope_nodes: Vec<hipGraphNode_t>,       // 72 (36 × 2)
    kv_write_nodes: Vec<hipGraphNode_t>,   // 36
    attention_nodes: Vec<hipGraphNode_t>,  // 36
}
```

`Drop` ruft `hipGraphExecDestroy` + `hipGraphDestroy`.

### Capture-Flow

1. **Token-ID-Memcpy BEFORE capture** — der `copy_from_host` muss
   außerhalb des Capture-Fensters laufen, sonst würde der Stack-
   Source-Pointer in einen MemcpyNode captured und beim Replay
   auf stale memory zeigen.
2. `hipStreamBeginCapture(hipStreamCaptureModeGlobal)`
3. Normaler `dispatch_node`-Loop über alle `graph.nodes`.
4. `hipStreamEndCapture → graph`
5. `hipGraphGetNodes` liefert Nodes in Topo-Order (= Capture-
   Order bei Single-Stream-Dispatch).
6. `LaunchSpans` aus `graph.nodes` berechnet welcher
   Graph-Node-Index welcher Kernel-Kind (Rope / KvWrite /
   Attention) entspricht — die `dispatch_node`-Logik produziert
   eine **deterministische** Launch-Sequenz.
7. Defensive Validation: `spans.total_launches == n_captured_nodes`.
   Mismatch → Fallback auf legacy.
8. `hipGraphInstantiate` → exec.
9. Ersten Token via Replay-Launch ausführen (Capture hat nicht
   gelaufen).

### Replay-Flow

1. Token-ID-Memcpy (außerhalb Graph).
2. Slots aktualisieren: `*kv_pos_slot = pos; *seq_len_slot = pos+1`.
3. Pro updatebarem Node:
   - `hipGraphKernelNodeGetParams(node, &p)` → aktuelle Params
     mit HIP-internen Arg-Pointern.
   - Lokalen `Vec<*mut c_void>` der richtigen Argument-Anzahl
     bauen, alle Pointer aus `p.kernelParams` kopieren, **einen**
     Slot durch `&*kv_pos_slot` bzw. `&*seq_len_slot` ersetzen.
   - `hipGraphExecKernelNodeSetParams(exec, node, &new_params)`.
   - HIP kopiert die Werte während des Calls — keine Pointer-
     Lifetime über den SetParams-Scope hinaus nötig.
4. `hipGraphLaunch(exec, stream)` → 1 Call statt 599.
5. `hipStreamSynchronize` → Logits lesen.

### Per-Kernel Arg-Layout

```
rope_kernel              6 args  pos at arg[1]
kv_cache_append_kernel   8 args  pos at arg[6]
attention_decode_kernel 10 args  seq_len at arg[7]
```

Konstanten in `executor.rs`:

```rust
const ROPE_POS_ARG_IDX: usize = 1;
const ROPE_N_ARGS: usize = 6;
const KV_WRITE_POS_ARG_IDX: usize = 6;
const KV_WRITE_N_ARGS: usize = 8;
const ATTENTION_SEQ_LEN_ARG_IDX: usize = 7;
const ATTENTION_N_ARGS: usize = 10;
```

### Fallback-Logik

```
execute_decode()
├── if hip_graph.is_some() → replay_decode_graph()
│     └── on error: cache.destroy(), fall through to legacy
├── elif should_capture_hip_graph() → capture_decode_graph()
│     └── on error: clean up, fall through to legacy
└── legacy path (always works)

should_capture_hip_graph():
  - !env(ROCMFORGE_DISABLE_HIP_GRAPH == "1")
  - runtime.all_exploiting()
  - hip_graph.is_none()
```

Bei Config-Änderung (`set_fused_gate_up`,
`set_gate_up_swiglu_dynamic_kernel`) wird der Graph via
`invalidate_fast_dispatch()` verworfen — neuer Capture bei
nächstem Decode.

## Korrektheit

```
Test: test_hip_graph_suite
Prompt: "Explain what a mutex is in one paragraph." (100 Tokens greedy)

Legacy (ROCMFORGE_DISABLE_HIP_GRAPH=1):
  "A mutex, short for \"mutual exclusion,\" is a synchronization..."

HIP-Graph:
  "A mutex, short for \"mutual exclusion,\" is a synchronization..."

assert_eq!(legacy.output, graph.output) → PASS
first 78 whitespace-tokens agree (= full output)
```

Bit-identisch weil:
- Gleiche Kernel-Reihenfolge
- Gleiche Committed-Bandit-Varianten
- Gleiche Weight/Buffer-Pointer
- Gleiche Arithmetik — nur Launch-Mechanismus verschieden

## Performance

### Isoliertes A/B (gleicher warmer Pipeline)

| Run | Prefill tok/s | Decode tok/s | Delta |
|---|---:|---:|---:|
| Legacy (env-forced) | 461.1 | 60.9 | 1.000× (ref) |
| **HIP-Graph** | 472.2 | **61.3** | **1.007×** (+0.7 %) |

### 15-Prompt-Suite (CLI, cold start)

| Metrik | Pre (Sudot4-Base) | Post (mit HIP-Graph) | Delta |
|---|---:|---:|---:|
| Prefill tok/s | 584.6 | 584.3 | −0.05 % |
| Decode tok/s | 56.8 | 56.5 | −0.5 % (Noise) |
| Wallclock | 106 035 ms | 106 448 ms | +0.4 % |
| 15/15 durchgelaufen | ja | ja | — |
| Monitor-Events | 1 (known flake) | 1 (same flake) | — |

**Wichtig:** Im 15-Prompt-Lauf konvergiert die Q6_K-LM-Head-Shape
(n=151936) nicht (braucht ≥ 10 000 Pulls, bekommt 5 935 in einer
Suite). Solange `runtime.all_exploiting()` == false bleibt, wird
kein Graph captured → 15-Prompt misst den Legacy-Pfad. Das ist
dasselbe Muster wie beim Dispatch-Opt-Cache aus der vorherigen
Session.

Für Szenarien in denen der Bandit schon vor der Suite warmgelaufen
ist (längere interaktive Sessions, CLI-Server-Betrieb) wird der
HIP-Graph aktiv — das A/B oben simuliert das.

## Gates

| Gate | Ziel | Gemessen | Status |
|---|---|---|:---:|
| Korrektheit bit-identisch | Pflicht | 78/78 Tokens match | ✅ |
| Decode-Rate ≥ 0.95× legacy | Pflicht | 1.007× | ✅ |
| Decode tok/s ≥ 60 floor | Pflicht | 61.3 | ✅ |
| Prefill unverändert (±10 %) | Pflicht | +2.4 % | ✅ |
| Fallback via Env-Toggle | Pflicht | `ROCMFORGE_DISABLE_HIP_GRAPH=1` greift | ✅ |
| Decode-Speedup ≥ 5 % (Stretch) | — | +0.7 % | ❌ |

Der 5 %-Stretch wurde wie in der Scope-Analyse ehrlich angekündigt
nicht erreicht.

## Was trotzdem landet

1. **Komplette HIP-Graph-FFI-Bindings** — 9 Funktionen +
   `hipKernelNodeParams`-Struct + 4 Capture-Mode-Konstanten.
2. **HipGraphDecodeCache mit sauberer Lifecycle-Verwaltung** — Drop
   ruft ExecDestroy + GraphDestroy, kein Leak.
3. **Capture/Replay-Infrastruktur** — production-ready, nicht
   experimentelles Skelett. Wenn zukünftig der Bandit früher
   konvergiert oder wenn der Code einen "long-running session"-
   Modus hat, greift der Graph-Pfad automatisch.
4. **SetParams-Update-Muster** — einmal korrekt implementiert,
   wiederverwendbar wenn weitere token-variable Parameter
   hinzukommen (z.B. wenn Batching kommt → batch_size parameter).
5. **`ROCMFORGE_DISABLE_HIP_GRAPH=1` Escape-Hatch** — garantiert
   dass wir ohne Risiko auf der gleichen Build-Config A/B messen
   können.

## Tests (2/2 grün)

| Test | Was |
|---|---|
| `test_disable_env_parsing` | CPU-unit: Env-Toggle-Semantik |
| `test_hip_graph_suite` | GPU (gated): Warmup → Legacy (env-forced) → HIP-Graph, bit-identische Outputs, ≥ 0.95× decode, Prefill unverändert |

Regression:
| Suite | Status |
|---|:---:|
| `cargo check --features v1,gpu --lib` | ✅ |
| 15-Prompt-Decode (Sudot4-Base) | ✅ 56.8 → 56.5 (Noise) |
| Post-Sudot4-Tests | ✅ unverändert |

## Vergleichs-Gesamttabelle (seit Phase 2 Start)

| Meilenstein | Decode (15p) tok/s | Gap zu llama.cpp (99.3) |
|---|---:|---:|
| 1.17 (Phase-1 End) | 30.6 | 3.24× |
| 2.0.3 Post-P0 | 40.7 | 2.44× |
| 2.1.3 Block D (GA winner, e2e) | 33.2* | — |
| 2.1.3 Block E (Bandit fix, richtig gemessen) | 39.8 | 2.49× |
| 2.1.5 WMMA-Prefill | 39.8 (decode unverändert) | 2.49× |
| Un-Fuse gate_up_swiglu | **59.7** | **1.66×** |
| 2.1.4 Q6_K Q8-Inline-Variant | 59.5 | 1.67× |
| Dispatch-Opt | 56.8 (Noise) | 1.75× |
| Q4_K sudot4-Variant | 56.8 (Noise) | 1.75× |
| **HIP-Graph Integration** | **56.5** | **1.76×** |

Der Hauptgewinn seit Phase 2 kam vom un-fuse gate_up_swiglu-Fix
(+20 tok/s auf einmal). Alles danach ist Infrastruktur + kleine
Rand-Optimierungen mit Gewinn ≤ Messrauschen.

\* Block D e2e war ein Test-Harness-Bug (Bandit nicht attached), nicht
reale Regression.

## Nächster Hebel

```
Wenn HIP-Graph-Gewinn minimal bleibt (wie gemessen):
  → Der CPU-Rust-Pfad ist nicht die Bottleneck-Quelle
  → Die verbleibende Decode-Lücke (1.76× zu llama.cpp) sitzt in:
    - GPU-Kernel-Performance (Q4_K GEMV @ 51 % BW, Raum bis ~70 %)
    - Attention bei langem Kontext (skaliert O(seq_len))
    - LM-Head Q6_K (bereits bei 95 % BW, wenig Spielraum)

Algorithmische Hebel:
  1. Flash-Attention-Tiling bei seq_len > 1024
  2. Weight-spezifische GA-Tuning per Shape (was Block C/D hätte 
     bringen können wenn die Suchraum-Achsen andere wären)
  3. Q8_1 + nwarps=8 + VDR=2 für Q4_K GEMV (Phase B+C aus sudot4-
     Scope, die alleine ohne Phase A den Gewinn brachten)
```

## Commit

Prefix: `feat(v1):` — neues Feature mit Fallback, korrekt aber mit
ehrlichem Performance-Zwischenergebnis.

```
feat(v1): HIP-Graph decode capture + SetParams replay (Option C)
```

Backup-Push auf `backup` Remote.
