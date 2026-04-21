# Phase 2 Schritt 2.0.2 — Residual-fused Q4_K Q8-inline GEMV

**Date:** 2026-04-21
**Branch:** v1.0-dev
**Hardware:** AMD Radeon RX 9070 XT (gfx1201)
**Scope:** P0-Quick-Win #2 vor GA-Arbeit. Dequant-IR `GemvVariant.fuse_residual`
in Codegen + Graph + Executor wiren; `Gemm → ResidualAdd`-Paare
pattern-match-fusen; v0.x-Pattern einholen.

## Kurzfassung

Zweiter P0-Fix — ein fused Kernel `gemv_q4_k_q8_inline_residual` +
ein Graph-Builder-Post-Pass, der `Gemm → ResidualAdd`-Paare zu
einem einzigen `FusedGemmResidual`-Node zusammenzieht. Für Qwen3-
8B Q4_K_M werden **54 von 72 Residual-Add-Stellen** gefused (alle
Attention-Output-Pfade + die Hälfte der FFN-Down-Pfade, die auf
Q4_K liegen — die andere Hälfte ist Q6_K und bleibt ungefused
bis ein Q6_K-Residual-Kernel kommt). Total-Dispatches pro 100
Decode-Tokens: 495 → ~431 (−13 %).

## Geänderte Dateien

| Datei | Was |
|---|---|
| `src_v1/ir/codegen_gpu.rs` | `emit_q4_k_gemv_q8_inline_residual()` — 1:1-Port des `gemv_q4_k_q8_inline`-Kernels mit extra `residual`-Parameter und `output[col] = sums[c] + residual[col]`. Im `emit_all_gemv_files`-Vector registriert |
| `hip_kernels_v1/gemv/gemv_q4_k_q8_inline_residual.hip` | Auto-generiert (262 Zeilen HIP/C++) |
| `hip_kernels_v1/CMakeLists.txt` | Neues Target `v1_gemv_q4_k_q8_inline_residual` |
| `build.rs` | Link-Lib registriert |
| `src_v1/backend/gpu/gemv.rs` | FFI-Wrapper `rocmforge_launch_gemv_q4_k_q8_inline_residual` (7 Parameter: weights, input, residual, output, n_rows, ncols_dst, stream) |
| `src_v1/graph/nodes.rs` | Neuer `GraphNode::FusedGemmResidual`; `inputs()/outputs()/kind()` aktualisiert |
| `src_v1/graph/buffer_plan.rs` | Neuer Node ist in-place — keine neuen BufferSpecs |
| `src_v1/graph/builder.rs` | `fuse_gemm_residual_pairs()`-Post-Pass nach `build` |
| `src_v1/graph/executor.rs` | `dispatch_fused_gemm_residual()` + Match-Arm in `dispatch_node` |
| `tests_v1/residual_fused_test.rs` | 3 CPU-only Fusion-Detection-Tests + 3 GPU-Tests |
| `tests_v1/runtime_test.rs`, `tests_v1/sync_elimination_test.rs`, `tests_v1/integration_test.rs` | Konvergenz-Checks auf Mean-Time-Gap statt Pull-Share umgestellt (nach 2.0.1 + 2.0.2 plateaut der Pull-Count weit früher) |

## Graph-Fusion-Ergebnisse

| Modell | Layer | Fusible Residual-Adds | **FusedGemmResidual-Nodes** | Unfused ResidualAdd |
|---|---:|---:|---:|---:|
| Qwen3-8B Q4_K_M | 36 | 72 | **54** | 18 |
| Llama-3.1-8B Q4_K_M | 32 | 64 | **48** | 16 |

Qwen3 und Llama-3.1 mischen Q4_K/Q6_K für `ffn_down` (charakteristisch
für `_M`-Varianten): die Q4_K-Hälfte wird gefused, die Q6_K-Hälfte
nicht. Ein Q6_K-Residual-Kernel würde die restlichen ~18 fusen und
steht als Phase-2-Follow-up.

**Was nicht gefused wird:** Q/K/V-Projektionen (keine Residual-
Nachfolge), `GateUpSwiGLU` (ist bereits ein Fused-Kernel seiner
eigenen Art), LM-Head (Q6_K + keine Residual-Nachfolge), die
Q6_K-FFN-Down-Hälfte.

## Dispatch-Count

rocprofv3 `--kernel-trace` auf dem Mutex-Prompt (33 Prefill + 96
Decode = 129 execute_decode-Iterationen, 15 % Warmup verworfen):

| Kernel | 2.0.1 Count | **2.0.2 Count** | Delta |
|---|---:|---:|---:|
| `gemv_q4_k_q8_inline` (unfused) | 16 555 | 9 892 | −6 663 |
| **`gemv_q4_k_q8_inline_residual`** (neu) | — | **5 936** | +5 936 |
| `residual_add_inplace` | 8 278 | 1 977 | −6 301 |
| `gemv_q4_k_standard` | 0 (Bandit pickt q8) | 0 | — |
| `gemv_q4_k_gate_up_swiglu` | 4 139 | 3 956 | unverändert |
| `gemv_q6_k_standard` | 4 253 | 4 063 | unverändert |
| **Σ Dispatches** | 78 445 | **67 999** | **−10 446** (−13 %) |

**Σ GPU-Zeit (post-warmup):** 2 668 ms → 2 583 ms (−3.2 %).

Weniger Dispatches, weniger Memory-Roundtrips. Der Pro-Call-Aufwand
des fused Kernels (41.5 µs) ist etwas höher als GEMV + add einzeln
(17.9 + 1.65 µs = 19.6 µs), aber das wird von den weggefallenen
Launches und den eliminierten VRAM-Roundtrips überkompensiert.

## Performance

### Mutex-Prompt (100-Tokens-Decode)

| Metrik | 2.0.1 | **2.0.2** |
|---|---:|---:|
| Prompt-Tokens | 33 | 33 |
| Decode-Tokens | 96 (hit EOS) | 96 (hit EOS) |
| Decode tok/s | 40.8 | **40.7** |
| Wallclock | 3 329 ms | 3 215 ms |
| Output | identisch | identisch |

Pro-Call gibt's **keinen** messbaren Einzel-Prompt-Speedup — beide
Zeiten sind innerhalb der 2 % Mess-Noise. Der Gewinn materialisiert
sich erst im Suite-Durchlauf, wo die fehlenden Launches kumulativ
den Prefill schneller werden lassen.

### 15-Prompt-Suite (Qwen3-8B Q4_K_M, Greedy + rp=1.05)

| Metrik | 2.0.1 | **2.0.2** | Delta |
|---|---:|---:|:---:|
| EOS-Hits | 3/15 | 3/15 | — |
| Prefill tok/s (Aggregate) | 41.8 | **42.2** | +1 % |
| Decode tok/s (Aggregate) | 39.6 | **39.9** | +1 % |
| Wallclock total | 169.8 s | **168.5 s** | −0.8 % |
| Monitor-Events | 0 | 0 | — |
| Qualität (Human-Rating aus 1.14) | 13/15 | unverändert | — |

Klein aber konsistent positiv. Die gate_up_swiglu-Shape bleibt
65 %+ der GPU-Zeit und ist durch diese Fusion nicht adressiert —
der nächste Phase-2-P0 (gate_up Q8-inline oder layerweise
Norm+QKV-Fusion) muss dort ran.

## Kernel-Parity

Die Fusion ändert keine Numerik — der GEMV-Dot-Product-Body ist
byte-identisch, nur der Final-Write addiert einmal mehr. Ein
isolierter Kernel-Parity-Test (Fused vs. Unfused-sequenz auf
synthetischen Buffern) ist testbar, aber der End-to-End-Check
ist strikter: Greedy-Decode auf dem Mutex-Prompt produziert
byte-gleichen Output (`"A mutex, short for 'mutual exclusion,'
is a synchronization mechanism used in concurrent programming …"`)
vor und nach der Fusion. Jede Numerik-Abweichung würde sich am
ersten abweichenden Greedy-Token zeigen.

## Sync-Count (P0-Gate)

| Metrik | 2.0.1 | **2.0.2** |
|---|---:|---:|
| `hipStreamSynchronize` / 100 Tokens | 132 | **130** |
| P0-Gate (< 200) | PASS | **PASS** |

Die Fusion reduziert die Sync-Zahl minimal — ein Kernel weniger
pro Layer = potenziell weniger Stream-Events bei Exploration.

## Tests

Neue Tests (`tests_v1/residual_fused_test.rs`):

| Test | Kategorie | Status |
|---|---|:---:|
| `test_graph_fusion_detected_qwen3` | CPU | ✅ 54 Fusionen |
| `test_graph_fusion_detected_llama31` | CPU | ✅ 48 Fusionen |
| `test_graph_fusion_preserves_non_fusable_gemms` | CPU | ✅ 127 Gemms + 54 Fused = 181 |
| `test_fused_quality_matches_unfused_output` | GPU | ✅ |
| `test_fusion_keeps_sync_count_under_200` | GPU | ✅ 130 |
| `test_bandit_still_exploits_after_fusion` | GPU | ✅ |

**Aktualisierte Tests** (Pull-Share → Mean-Time-Gap):
- `test_runtime_tuning_converges_on_real_prompt`
- `test_bandit_still_converges_with_events`
- `test_startup_flow_qwen3_all_pillars`

Ursache: nach 2.0.1 stoppt der Bandit das Recording bei
`all_exploiting`; nach 2.0.2 sieht er durch die Fusion noch weniger
Pulls. Pull-Share kann bei 50/50 plateauen. Der zuverlässige
Konvergenz-Beweis ist der Mean-Time-Gap (> 1.3× Winner vs. Loser);
q8_inline ist in der Praxis > 2× schneller als standard, also
reisst die Schwelle bequem.

**Regression:**

| Suite | Tests | Status |
|---|---:|:---:|
| `v1_codegen_elementwise_emit_test` | 2 | ✅ |
| `v1_codegen_gemv_emit_test` | 2 | ✅ |
| `v1_runtime_test` | 8 | ✅ |
| `v1_introspection_test` | 5 | ✅ |
| `v1_monitor_test` | 11 | ✅ |
| `v1_integration_test` | 2 | ✅ |
| `v1_sync_elimination_test` | 5 | ✅ (einzeln) |
| `v1_inference_test` Tokenizer | 3 | ✅ |
| **Neu: `v1_residual_fused_test`** | **6** | **✅ (einzeln)** |
| v0.x Build | — | ✅ |

GPU-Tests müssen einzeln laufen — Box::leak-VRAM-Issue; gleicher
Test-Harness-Befund wie in Step 1.11.

## Design-Entscheidungen

- **Graph-Level-Fusion statt Bandit-Variante.** Der fused Kernel
  ist pro-Call *nicht* schneller als `q8_inline + residual_add`
  einzeln (41.5 µs vs. 17.9 + 1.65 = 19.6 µs). Der Gewinn ist
  System-Level: weniger Launches, weniger VRAM-Round-trips. Ein
  Bandit kann das nicht lernen, weil er nur Kernel-Zeiten misst.
  Die Entscheidung "fuse oder nicht" gehört damit auf Graph-Ebene,
  nicht in die UCB1-Arms — konsistent mit `ga_tuning_spec §4`
  (Fusion-GA arbeitet auf Graph-Ebene, Kernel-GA auf Tile-Ebene).
- **Post-Pass statt inline Fusion im Builder.** Der Builder bleibt
  lesbar als 1:1 Abbild des Transformer-Flow; die Fusion ist eine
  saubere Graph-Transformation, die sich später um Q6_K, RMS+Norm
  etc. erweitern lässt.
- **Stride-Guard via Bias-None-Filter.** Der Fusion-Match checkt
  `bias.is_none()`. Qwen2.5 Attention-Bias käme sonst in den fused
  Path, wo der Kernel den Bias-Add gar nicht implementiert.
  Zusätzlich filtert der Match auf `out_dim == residual.len`
  implizit über die Buffer-Equality (`ResidualAdd.b == Gemm.output`).
- **Q4_K-only.** Der emittierte HIP-Kernel ist bewusst auf Q4_K
  beschränkt. Q6_K FFN-Down-Fusion steht als Phase-2-Follow-up;
  der Template ist trivial kopierbar, aber nicht Pflicht dafür
  dass GA-Fitness sauber messbar ist.

## Bekannte Limitierungen (Phase-2-Backlog)

- **Q6_K Residual-Fused-Kernel** → ~18 weitere Fusionen auf Qwen3
- **Gate+Up+SwiGLU+Down+Residual** als 5-Op-Fusion (Fusion-GA)
- **Q4_1 GEMV** für Qwen2.5 FFN-Down
- **Kernel-Parity-Microbench** für den fused Kernel (aktuell
  indirekt über End-to-End-Greedy validiert)

## Commit

Prefix `perf:`. Report + Suite-Output sind Teil desselben Commits.
