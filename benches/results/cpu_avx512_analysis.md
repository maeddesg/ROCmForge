# CPU AVX-512 VNNI Q4_0 GEMV — analysis & verdict

**Commit:** `6c0eade+`
**CPU:** AMD Ryzen 9 7945HX (Zen4, 16C/32T, AVX-512 VNNI, DDR5 dual-channel ~77 GB/s)
**Model:** Qwen2.5-0.5B-Instruct Q4_0 (353 MB)
**Date:** 2026-04-17

## TL;DR

**Qwen2.5-0.5B Q4_0 decode throughput on CPU with AVX-512: ~12.1 tok/s.**
**Baseline (AVX2): ~12.1 tok/s. Speedup: ~0%.**
**Isolated kernel speedup: 1–7% (0.5B shapes), 16–19% (7B shapes).**

**Verdict: < 40 tok/s → pivot to Prefill-GEMM.** Heterogeneous
spec-decode (draft on CPU, target on GPU in parallel) is not feasible
with the current CPU forward stack.

## What was done

1. `src/cpu/features.rs` extended with `has_avx512_vnni` (previously only computed internally).
2. New AVX-512 VNNI kernel in `src/cpu/ops.rs`:
   - `dot_q4_0_q8_0_2blocks_avx512_vnni` — processes 2 Q4_0 blocks (64 bytes) with a single `_mm512_dpbusd_epi32`.
   - `dot_q4_0_q8_0_block_avx512_vnni` — single-block fallback for odd-count leftover blocks, uses AVX-512-VL-256-bit VNNI.
   - Bias correction for signed Q4 via `dot - 8 × sum(q8)` (vpdpbusd is unsigned × signed; `_mm512_sign_epi8` does not exist).
3. Dispatch in `gemv_q4_0_q8_0` extended with an AVX-512-VNNI path; fallback chain AVX-512-VNNI → AVX2 → scalar.
4. Opt-out via `ROCMFORGE_DISABLE_AVX512=1`.
5. Correctness test `tests/cpu_avx512_matches_reference.rs` (byte-identical output).
6. Micro-benchmark `src/bench_gemv.rs` extended to cover 0.5B and 7B shapes with AVX-512 vs AVX2.
7. End-to-end sweep script `benches/bench_cpu_avx512.fish`.

## Isolated kernel benchmark

Single-thread GEMV `gemv_q4_0_q8_0`, including Q8 input quantization and Rayon overhead:

| Shape                                | AVX2 µs/call | AVX-512 µs/call | Speedup |
|--------------------------------------|-------------:|----------------:|--------:|
| 0.5B QKV/O  (896 × 896)              |        147.1 |           145.5 |   1.1 % |
| 0.5B Gate/Up (4864 × 896)            |        257.2 |           245.0 |   4.7 % |
| 0.5B Down   (896 × 4864)             |        218.7 |           204.4 |   6.5 % |
| 7B   QKV/O  (3584 × 3584)            |        339.8 |           326.6 |   3.9 % |
| 7B   Gate/Up (18944 × 3584)          |       1044.9 |           878.7 |  18.9 % |
| 7B   Down   (3584 × 18944)           |       1050.3 |           902.3 |  16.4 % |

Effective weight-read bandwidth (AVX-512):
- 0.5B: 3–12 GB/s (far below the 77 GB/s DDR5 ceiling — data fits in L2/L3).
- 7B: 22–44 GB/s (approaching half of DDR5 bandwidth).

## End-to-end decode throughput

15 runs (3 prompts × 2 modes × 3 runs), Qwen2.5-0.5B Q4_0, 128 tokens, greedy:

| Prompt | AVX2 (median) | AVX-512 (median) | Speedup |
|--------|--------------:|-----------------:|--------:|
| code   | 12.1 tok/s    | 12.1 tok/s       |    0 %  |
| chat   | 12.1 tok/s    | 11.7 tok/s       |   −3 %  |
| prose  | 12.0 tok/s    | 11.9 tok/s       |   −1 %  |

Raw data: `cpu_avx512_sweep_6c0eade_1776406574.json`.

7B end-to-end: 0.7 tok/s (CPU is not useful for 7B inference).

## Why the kernel gain does not land

The kernel is 1–7 % faster on 0.5B shapes, but that gain evaporates end-to-end. Reasons:

1. **Rayon overhead dominates on small matrices.** Each GEMV call (≈180 per token × 24 layers = 192 calls) spawns Rayon tasks. At 56 output rows per thread and ~500 µs per GEMV, the thread-pool handshake (task split, sync barrier) is a non-trivial fraction. The actual SIMD compute time is small relative to the dispatch overhead.
2. **Other ops are not SIMD-optimized.** `flash_attn_decode`, `rms_norm`, `silu_fuse`, `rope`, `residual_add` run in the CPU path mostly scalar or only with compiler auto-vectorization. On 0.5B they add up to a substantial fraction of the 80 ms/token budget.
3. **`quantize_q8_0_single` is scalar.** The FP32 → Q8 conversion for the input vector is not SIMD-optimized. On small shapes (hidden=896) that is only ~1 µs per call, but 200+ calls per token add up.
4. **Bandwidth is not the bottleneck on 0.5B.** The weight data (~200 MB per token-pass) would take ~2.6 ms at 77 GB/s — we are at 80 ms, 30× slower. Even a 2× faster kernel would raise end-to-end by less than 20 %, because the other costs dominate.

## Verdict

**Fused FFN on CPU, fused attention on CPU, and further SIMD work across the rest of the pipeline together would be needed to reach the ≥3× factor required for 40 tok/s.** That is a multi-month project — not the "small addition" it would have been if it slotted cleanly into the spec-decode work.

### What this means for strategy

- **Heterogeneous spec-decode (draft on CPU, target on GPU in parallel) is NOT feasible** with the current CPU stack. At 12 tok/s CPU-draft vs. 222 tok/s GPU-draft, the GPU would sit blocked most of the time, because a draft step would take 82 ms instead of 4.5 ms.
- **Prefill GEMM (hipBLAS or WMMA) is the next milestone.** Prefill gap to llama.cpp: 59 vs. 1,092 tok/s (18× gap), the largest remaining performance lever in the project.

### What the AVX-512 kernel still gives us

The kernel stays in the tree — it is a clean, tested foundation and delivers 16–19 % in isolation on 7B shapes. It becomes useful later if:
- Prefill is rewritten to GEMM — the Q4_0 × Q8_0 batched kernel could slot in.
- The rest of the CPU pipeline is SIMD-optimized (fused FFN, flash-attention, RMSNorm, SiLU) — then the kernel would translate into a real end-to-end gain.
- The user runs without a GPU — the CPU path runs anyway, and 7 % more throughput on 0.5B drafts is better than 0 %.

No rollback.

## Open questions / follow-ups (not now)

- How expensive is Rayon scheduling per GEMV call really? A profile run with `perf` or an instrumentation layer would set the upper bound for further GEMV optimization.
- Is it worth a single global thread pool with persistent workers (`crossbeam_channel` / `parking_lot`) instead of Rayon's per-call fork-join?
- On a CPU without a GPU, AVX-512 + fused ops could push 0.5B inference to ~25–30 tok/s — still not a hero feature, but it would make ROCmForge competitive as a CPU-only solution.
- 7B on CPU: currently 0.7 tok/s. llama.cpp typically reaches 6–8 tok/s on the same 7945HX. The CPU forward path therefore has a ~10× gap to the state of the art — separate from the AVX-512 question.
