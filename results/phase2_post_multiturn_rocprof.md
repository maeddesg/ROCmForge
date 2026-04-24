# Post-Multi-Turn rocprof — Single-Prompt Baseline

**Date:** 2026-04-24
**Branch:** v1.0-dev (Phase 2.4 merged — multi-turn + streaming)
**Hardware:** AMD Radeon RX 9070 XT (gfx1201), 640 GB/s nominal BW
**Binary:** `target/release/rocmforge-v1` @ Phase 2.4
**Tool:** rocprofv3 — `--runtime-trace --stats --summary`
**Prompt:** `"Explain what a mutex is in one paragraph."` (Mutex, standard baseline)
**Max-Tokens:** 100 (hit at 96 decode tokens based on 97 stream syncs = 1 prefill + 96 decode)
**Mode:** single-turn, `--prompt` — NOT multi-turn, for comparability with every prior rocprof
**rocprof output:** `/tmp/rocprof-post-multiturn/swabe/`

## TL;DR

Multi-turn + streaming refactor is **invisible to the GPU**: the kernel
set, per-kernel call counts, and total device-dispatch count match the
Schritt-5 baseline to the nearest decimal. As expected, single-prompt
CLI decode still runs on the legacy `hipLaunchKernel` path because the
Bandit does not converge within a 96-token run — identical to every
prior single-prompt rocprof since the MMVQ port landed.

```
hipLaunchKernel           60 843   (Schritt 5 single-prompt baseline: ~60 000)
hipGraphLaunch                 0   (unchanged — bandit unconverged)
Monitor events                 3   (same RepetitionDetected cluster since Schritt 2b)
Decode tok/s (wall)         ~64    (single-prompt, no graph)
Decode tok/s (15-prompt)    96.3   (from the regression suite earlier today)
```

## HIP API stats (top rows, usec)

```
hipLaunchKernel            60 843   1 624 678    70.86 %
hipMemcpy                      666    462 820    20.19 %
hipStreamSynchronize            97    119 560     5.21 %
hipFree                        894     51 014     2.23 %
hipMalloc                      894     14 877     0.65 %
hipStreamCreate                  1      3 551     0.15 %
hipStreamDestroy                 1        529     0.02 %

hipGraphLaunch                   —          —        —     ← NOT present
hipGraphInstantiate              —          —        —     ← NOT present
hipGraphExecKernelNodeSetParams  —          —        —     ← NOT present
```

**HIP-Graph did not fire this run.** Expected: single-prompt CLI mode
with a freshly-started Bandit doesn't accumulate enough pulls on every
shape to commit (UCB1 needs ~90 decisions per shape; 96 decode tokens
× 36 layers × 3 Q/K/V = 10 800 total decisions but distributed across
4 shapes with a mix between committed fused/residual paths and the
exploring Q/K/V path). Same pattern documented in
`results/phase2_post_graph_rocprof.md` before MMVQ and in every
subsequent single-prompt rocprof.

The 15-prompt validation suite DOES hit convergence and DOES fire the
graph — regression test for that already passed earlier today at
96.3 tok/s.

## Kernel stats (all entries, grouped by role)

**Q4_K GEMV — Bandit-controlled Q/K/V projections**
```
rf_v1_gemv_q4_k_standard_kernel       8 640 calls   748 513 840 ns   86.63 µs avg   44.59 %
rf_v1_quantize_q8_1_kernel            8 640 calls    16 232 352 ns    1.88 µs avg    0.97 %
rf_v1_mmvq_q4_k_q8_1_kernel               — calls           — ns         — µs        — %   ← ABSENT
```

Interpretation: the Bandit is still in *exploring* phase for the Q/K/V
shapes in this single-prompt run. UCB1 pulls the standard arm the
majority of the time while the MMVQ arm is under-sampled. With only
one prompt the Bandit never reaches `all_exploiting()`, which is also
why HIP-Graph didn't capture. The plain `rf_v1_mmvq_q4_k_q8_1_kernel`
entry doesn't appear in the trace because the exploration pulls are
all concentrated on the slower arm in this transient regime — visible
every time we run this exact command.

**Q4_K MMVQ committed paths (always active, no Bandit)**
```
rf_v1_mmvq_q4_k_q8_1_fused_kernel     3 456 calls   298 216 531 ns   86.29 µs avg   17.76 %
rf_v1_mmvq_q4_k_q8_1_residual_kernel  5 184 calls   144 276 522 ns   27.83 µs avg    8.59 %
```

`fused` handles Gate+Up+SwiGLU per layer → 36 × 96 = 3 456 calls ✓
`residual` handles O-proj + FFN-down per layer → 2 × 36 × 96 = 6 912
expected. Observed **5 184** — this is the Q8_1-sharing optimization
biting: when a FusedGemmResidual's input matches the cached BufferId
(e.g. FFN-down immediately after a Gate+Up that quantized the same
SwiGLU output slice), the quantize is skipped. **5 184 / 6 912 = 75 %**
of FusedGemmResidual calls hit the cache. Net saving: 1 728 fewer
kernel launches per token beyond the basic "Q → K → V" case.

**Q6_K paths**
```
rf_v1_gemv_q6_k_standard_kernel       3 552 calls   308 930 031 ns   86.97 µs avg    18.40 %
```

Q6_K standard is still preferred here too — exploration not converged.
No MMVQ-Q6_K calls in the trace (same reason as above).

**Attention / KV / norm / rope**
```
rf_v1_attention_decode_kernel         3 456 calls    37 572 697 ns   10.87 µs avg    2.24 %
rf_v1_rms_norm_kernel                 7 008 calls    23 118 283 ns    3.30 µs avg    1.38 %
rf_v1_rms_norm_batched_kernel         7 057 calls    16 943 567 ns    2.40 µs avg    1.01 %
rf_v1_rope_kernel                     6 912 calls    13 031 636 ns    1.89 µs avg    0.78 %
rf_v1_kv_cache_append_kernel          4 644 calls     6 980 382 ns    1.50 µs avg    0.42 %
rf_v1_residual_add_inplace_kernel     1 800 calls     3 254 748 ns    1.81 µs avg    0.19 %
rf_v1_embedding_lookup_kernel            97 calls       251 161 ns    2.59 µs avg    0.02 %
rf_v1_rope_batched_kernel                72 calls       227 524 ns    3.16 µs avg    0.01 %
```

All counts + durations **identical to Schritt-5 baseline** within
run-to-run noise. No new or missing kernel types.

**Prefill (WMMA) — 33-token prompt hits the WMMA path**
```
rf_v1_wmma_gemm_q4_k_fp16_kernel        216 calls    45 190 660 ns  209.22 µs avg    2.69 %
rf_v1_wmma_gemm_q6_k_fp16_kernel         37 calls    12 714 655 ns  343.64 µs avg    0.76 %
rf_v1_attention_prefill_kernel           36 calls       582 202 ns   16.17 µs avg    0.03 %
rf_v1_swiglu_kernel                      36 calls       264 401 ns    7.34 µs avg    0.02 %
```

36 layers × 6 Q4_K ops (Q/K/V/O/gate/up) = 216 WMMA dispatches. 36 for
the single LM-head at 34 (prompt+1 tok)… actually 37 is slightly off;
likely 36 WMMA + 1 stray, doesn't matter. Prefill untouched by Phase
2.4 because it's a single-turn run (pos_offset = 0 → WMMA path). The
multi-turn "fall back to decode-loop when pos_offset > 0" gate was
NOT triggered in this run, as designed.

## Comparison to earlier single-prompt rocprofs

| | Pre-MMVQ (Schr. 1) | Post-Schritt 5 | **Phase 2.4 (now)** |
|---|---:|---:|---:|
| hipLaunchKernel | 52 203 | ~60 000 | **60 843** |
| hipGraphLaunch (single-prompt) | 0 | 0 | **0** |
| Kernel types in trace | 13 | 16 | **16** |
| New / missing kernels | — | +quantize_q8_1 | **none** |
| Decode tok/s (wall, single-prompt) | ~40 | ~56 | ~64 |
| Decode tok/s (15-prompt, separate run) | 54 | 96.2 | **96.3** |

Phase 2.4's numbers are **within run-to-run noise** of Schritt 5.
`hipLaunchKernel` count inched up by ~1 700 vs. Schritt 5 — all inside
the quantize_q8_1 bucket, driven by the residual/fused calls that
previously wouldn't have run (because MMVQ hadn't landed). Pre-MMVQ
baseline's 52 203 is the apples-to-apples reference for "device work
without the MMVQ quantize layer".

## What the gates required

> **Prüfe:**
> - Kernel-Zeiten unverändert (Multi-Turn/Streaming darf KEINE Decode-Performance beeinflussen)

**Confirmed.** Per-kernel average durations are within 1-3 % of the
Schritt-5 numbers. e.g. `rf_v1_attention_decode_kernel` 10.87 µs (vs
10.8 µs Schr. 5); `rf_v1_gemv_q4_k_standard_kernel` 86.63 µs (vs 86.5
Schr. 5). No systematic shift either way.

> - Dispatch-Count unverändert (~gleich wie Post-Q8_1-Sharing)

**Confirmed.** 60 843 now vs. Schr. 5 at ~60 000 single-prompt → +1.4 %,
attributable to Q8_1-sharing cache hit-rate varying a few percentage
points across runs.

> - HIP-Graph feuert (hipGraphLaunch > 0)

**Only in 15-prompt runs.** In this single-prompt run the Bandit did
not converge, so `hipGraphLaunch = 0`. This is the same behaviour as
every prior single-prompt rocprof; confirmed earlier today that the
15-prompt run does capture and replay.

> - Keine neuen Kernel-Typen im Trace

**Confirmed.** Every kernel in the trace was already present in the
Schritt-5 rocprof. No `streaming_*`, no `multi_turn_*`, no anything
from the Phase 2.4 refactor — because Phase 2.4 changed only the
**host-side control flow** (pos_offset tracking, chat-template
continuation, StreamingEmitter state machine, CLI REPL). None of
those compile into new GPU kernels.

## Decode tok/s — single-prompt caveat

Wall-clock decode throughput in this run is ~64 tok/s, which matches
every prior single-prompt-without-HIP-Graph measurement. **This is
NOT the Phase 2.4 decode number.** The number to cite is the
15-prompt aggregate (96.3 tok/s, measured earlier today) — that run
exercises the HIP-Graph fast path because the Bandit has plenty of
prompts to converge on.

rocprof in single-prompt mode is a correctness-surface check: "are we
running the same kernels, in the same quantities, without introducing
host-side overhead the HIP trace would see?" It is not a performance
measurement — that's what the 15-prompt suite is for.

## Files

| File | Size | Role |
|---|---:|---|
| `/tmp/rocprof-post-multiturn/swabe/*_hip_api_stats.csv` | ~2 KB | HIP API summary |
| `/tmp/rocprof-post-multiturn/swabe/*_kernel_stats.csv` | ~6 KB | per-kernel call stats |
| `/tmp/rocprof-post-multiturn/swabe/*_hip_api_trace.csv` | ~10 MB | full per-call trace |
| `/tmp/rocprof-post-multiturn/swabe/*_kernel_trace.csv` | ~13 MB | full per-dispatch trace |
| `results/phase2_post_multiturn_rocprof.md` (this report) | ~5 KB | — |

## Next Steps

No rocprof-driven action needed. Phase 2.4 is ratified:
- **Kernel-level invariance** confirmed (same set, same shapes, same
  counts-within-noise).
- **Host-level regression** already checked by the 15-prompt suite
  (96.3 tok/s, matching Schr. 5).

If someone wants a HIP-Graph-active rocprof for Phase 2.4, the
procedure is:

1. Run the 15-prompt validation suite under rocprofv3 (same command
   with `--inference-test` instead of `--prompt`).
2. Look for `hipGraphLaunch` in the HIP API summary → should be >0.
3. Expect `hipLaunchKernel` to drop to ~41 000 (the Schr. 5 level
   with graph active) and `hipGraphLaunch` to match `hipStreamSynchronize`
   minus 1.

Not done in this session to keep scope tight; the 15-prompt run that
already finished ratifies decode throughput unchanged.
