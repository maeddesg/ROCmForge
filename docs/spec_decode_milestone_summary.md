# Speculative Decoding — Milestone Summary

**Date:** 2026-04-17, commit `20733bb`
**Hardware:** RX 9070 XT (gfx1201, RDNA 4), ROCm 7.2.1
**Models:** target Qwen2.5-7B-Instruct Q4_0, draft Qwen2.5-0.5B-Instruct Q4_0

A compact recap of the speculative-decoding optimization cycle. Audience: the author six months from now, or a contributor who needs to understand the current state of things.

## 1. Starting point

- Speculative decoding implemented and correct (PR #14 — Batched Verification, April 2026).
- Baseline (direct 7B decode): **82 tok/s** median over 15 prompts.
- Spec-decode is **net negative** on mixed prompts and only profitable at high acceptance rates.
- Measured break-even α ≈ **41%** — below this threshold spec-decode is slower than direct decode.
- Code prompts (α ≥ 73%) profitable; chat / prose not.

## 2. Experiments run

| # | Experiment                         | Expected gain         | Measured gain      | Finding |
|---|------------------------------------|-----------------------|--------------------|---------|
| 1 | Spec-step cost breakdown           | — (profiling)         | —                  | target-verify = 88.6 % of step time; FFN dominates verify (67.3 %) |
| 2 | Tiled batched GEMV (FFN-down)      | 2–4 ms/step           | ~250 µs (~1.5 %)   | memory controller pipelines sequential accesses; bandwidth model overshoots by ~12× |
| 3 | Adaptive-depth threshold sweep     | threshold < 1.2 better| threshold 1.2 optimal | super-linear verify cost with batch size dominates; lower thresholds are net negative |
| 4 | Batched lm_head                    | ~2,500 µs/step        | ~114 µs (~0.4 %)   | same pipelining effect as #2; GEMV against the same matrix pipelines even without batching |
| 5 | Buffer-traffic validation (FFN)    | 1,500–2,500 µs/step   | ~200 µs (~1.2 %)   | hypothesis falsified; fused FFN not worthwhile; weight-matrix traffic dominates intermediate by factor 500 |

Per-experiment detail:

- **#1** — `profiling/results/LAUNCH_OVERHEAD_ANALYSIS.md`
- **#2** — `docs/batched_verify.md` (section "Memory-Controller Pipelining"), `benches/results/tiled_sweep_313efdf.json`
- **#3** — `benches/results/depth_threshold_sweep.md`
- **#4** — `benches/results/batched_lm_head_analysis.md`, `benches/results/batched_lm_head_sweep_2d595f8_1776356792.json`
- **#5** — `profiling/results/BUFFER_TRAFFIC_ANALYSIS.md`, `profiling/results/buffer_traffic_validation_{92d037a}_{1776402443}.json`

## 3. What landed in the repo

- **Tiled batched GEMV** (`hip_kernels/quant/q4_0_gemv_batched_tiled.hip`, default on). Disable with `ROCMFORGE_DISABLE_TILED_GEMV=1`.
- **Batched lm_head** (`src/gpu/forward.rs::gpu_verify_lm_head_batched`, default on). Disable with `ROCMFORGE_DISABLE_BATCHED_LM_HEAD=1`.
- **Adaptive speculation depth** (EMA-based, thresholds 1.2/2.5, initial = `spec_depth * 0.5`).
- **Profiling infrastructure** — `ROCMFORGE_PROFILE_SPEC_STEP=1` (5-phase breakdown), `ROCMFORGE_PROFILE_VERIFY_BREAKDOWN=1` (verify sub-phases).
- **Scratch-buffer infrastructure** — `MAX_SPEC_DEPTH=8` → `MAX_VERIFY_BATCH=9`; dedicated `logits_batch` / `argmax_batch_device` / `argmax_batch_host` buffers in `GpuForwardScratch`.
- **Benchmark suite** — `benches/bench_spec.sh`, `benches/bench_batched_lm_head.fish` with structured JSON output.
- **Correctness tests** — `tests/spec_greedy_matches_baseline.rs` (greedy spec-decode output = direct greedy decode), `tests/batched_lm_head_matches_sequential.rs` (byte-identical output at depth 1/3/5), plus `--spec-depth` validation (max 8).
- **Architecture notes** — `docs/architecture_notes.md` with the memory-controller pipelining finding as a project-wide insight.
- **Micro-benchmark** — `profiling/buffer_traffic/bench.hip` (standalone HIP, A/B1/B2 variants, fma-depth sweep).

## 4. Central architectural insight

The "load once instead of N times" model **consistently overshoots the gain on RDNA 4 by a factor of 10–20×**. Three independent experiments confirm this pattern: dispatch batching, weight-reuse optimization, and buffer-traffic elimination deliver no measurable gain on bandwidth-bound kernels. The memory controller pipelines sequential accesses to the same address ranges automatically, as long as no explicit synchronization sits between them. The expected savings derived from naive bandwidth arithmetic were modelling artifacts in all three cases.

Consequence: the real optimization levers are **algorithmic changes** (GEMV → GEMM for prefill) and **compute patterns with unpredictable access patterns** (attention tiling at long context, where the KV cache spills out of L2). They differ qualitatively from the previous experiments because their memory-access patterns are such that the memory-controller pipelining effect is weaker.

## 5. Current performance

Median tok/s over 15 prompts (5 code, 5 chat, 5 prose), target Qwen2.5-7B-Q4_0, draft Qwen2.5-0.5B-Q4_0, 128 generated tokens:

| Mode                      | Median tok/s | Code (best)       | Chat (median) | Prose (median) |
|---------------------------|-------------:|-------------------|--------------:|---------------:|
| Baseline (no spec)        | **82**       | ~82               | ~82           | ~82            |
| Spec depth=1              | 69           | **86** (α = 91 %) | 68            | 66             |
| Spec depth=3              | 66           | 78                | 65            | 63             |
| Spec depth=5              | 57           | 63                | 56            | 57             |

Interpretation:

- Spec-decode at depth=1 is profitable on **code prompts (α ≥ 73 %)**, +5 % above baseline in the best case.
- On **chat / prose it is net negative** (−15 % to −25 % below baseline).
- **Break-even α ≈ 41 %** — below that, baseline decode is always faster.
- Adaptive depth (EMA) converges to depth=1 after 5–10 steps for low-α prompts, and holds at the configured maximum for high-α prompts.

## 6. Recommended next steps

No implementation plans — direction only.

1. **Prefill GEMM** (hipBLAS or WMMA). The largest remaining gap to the state of the art (prefill pp19: 59 tok/s vs. llama.cpp 1,092 tok/s, 5 % of their baseline). Switching from N × GEMV to real GEMM is qualitatively different from the previous micro-optimizations — different memory-access patterns and different cost structure.
2. **Attention optimization at long context.** At K=4096 decode throughput drops from 115 to 78 tok/s (−32 %), attention reaches 34 % of decode time. The KV cache spills out of L2 — a different memory pattern where tiling strategies (FlashAttention-style) could deliver real throughput.
3. **14B target as benchmark validation.** No coding work needed. Validates the break-even hypothesis and shows whether spec-decode becomes profitable at larger target models (larger draft:target cost ratio). VRAM-limited to 16 GB, currently tight.
4. **Rejection sampling.** Only relevant once prefill and attention are solved. Would lift the greedy constraint and raise acceptance on chat / prose, but the draft-forward cost (10 % of step time) stays — the big lever lives elsewhere.
5. **CPU optimization (Zen4+ / AVX-512).** AVX-512 VNNI GEMV kernel implemented and committed (commit `d0e4f07`). Isolated kernel speedup: **16–19 % on 7B shapes, 1–7 % on 0.5B shapes, 0 % end-to-end** on 0.5B due to orchestration overhead (Rayon fork-join + scalar non-GEMV ops). **Heterogeneous spec-decode (draft on CPU) is not feasible with the current CPU stack** — the CPU takes 82 ms per draft token versus 4.5 ms on the GPU. Would need a fundamental CPU pipeline rewrite (Rayon elimination or persistent worker threads, fused ops, SIMD for every pipeline stage, not just GEMV). Standalone project, not realizable as a by-product of the spec-decode work. Details in `benches/results/cpu_avx512_analysis.md` and `docs/architecture_notes.md` (section "Orchestration trap on small models").

## 7. What should explicitly NOT be done

- **Fused FFN** — falsified by experiment #5. Maximum realistic gain ~200 µs/step (1.2 %), threshold was 1,500 µs (8 %). A complex kernel rewrite for < 2 % throughput. Do not implement.
- **Further GEMV micro-optimizations within the current paradigm.** The plateau has been reached — target-verify is 88 % GEMV execution against the weight matrix, bandwidth-limited. Further micro-optimizations yield < 2 % and typically collide with the memory-controller pipelining pattern.
- **Persistent-thread kernels.** Launch overhead is not the bottleneck (measured ~2.7 µs/dispatch, effectively free when piped on the stream). Architecturally heavy for single-digit-µs savings per dispatch.
- **A smaller draft model (< 0.5B).** Draft-forward is already only 10 % of step time. A smaller draft saves little and raises acceptance risk via worse prediction.
- **Batched verify attention at very long context.** KV-cache read time dominates; reducing the dispatch count does not help there — memory bandwidth is the ceiling.

---

With that the spec-decode milestone is cleanly closed. The starting point for the next chapter (prefill GEMM) is clear: **algorithmic change**, not further dispatch / buffer micro-optimizations.
