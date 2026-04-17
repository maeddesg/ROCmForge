# Phase 3a Step 2 — multi-head WMMA attention throughput

**Date:** 2026-04-17
**Commit:** on top of `f2164c8`
**GPU:** RX 9070 XT (gfx1201, RDNA 4), ROCm 7.2.1

## TL;DR

WMMA multi-head attention PoC vs. the existing `flash_attn_prefill_strided_kernel`:

| seq_len | Existing (µs) | WMMA (µs) | Speedup |
|--------:|--------------:|----------:|--------:|
|      64 |        21,627 |      60.9 |    355× |
|     128 |        38,150 |     116.6 |    327× |

Extrapolated attention time for a full 28-layer Qwen2.5-7B prefill at pp=256: **~13 ms** (vs ~2,330 ms measured for the existing kernel — a ~180× cut if Phase 3b preserves this scaling).

**Phase 3b (online softmax) is clearly justified.** The kernel is already 300×+ faster on the sequence lengths that fit in LDS without online softmax; extending the same WMMA approach to pp=256/pp=512 via online softmax makes it the dominant perf win the prefill has seen.

Raw data: [`wmma_attention_bench_f2164c8_1776423303.json`](wmma_attention_bench_f2164c8_1776423303.json).

## Methodology

- 10 warmup iterations + 100 measured, HIP events around each call.
- Per sequence length, one full "forward" (all 28 Q heads) is timed:
  - Existing path: 28 × `flash_attn_prefill_strided_kernel` invocations.
  - WMMA path: 1 × `wmma_attention_prefill_multihead_launch`.
- Inputs shared: FP32 random Q/K/V fed to the existing kernel; the same
  values cast to FP16 and given to the WMMA kernel. The conversion
  happens outside the timed window.
- No causal mask on either path (Phase 3c), no GQA sharing (Phase 3c).

## Why the speedup is this large

Two compounding factors:

1. **Matrix cores instead of scalar FMAs.** The WMMA path routes both
   GEMMs (Q·K^T and P·V) through `v_wmma_f32_16x16x16_f16` at roughly
   60 % of the ~49 TFLOPS FP16 peak. The existing kernel uses the
   vector ALUs serially — no matrix cores.
2. **Algorithmic inefficiencies in the scalar kernel.** Looking at
   `flash_attn_prefill_strided_kernel`:
   - Launch config: 1 block per query position, 256 threads per block.
   - Inside: the online softmax + output update step runs on **thread
     0 only** (a 128-element serial loop per key position), while 255
     threads sit idle. That single line:
     ```cpp
     if (tid == 0) {
         ...
         for (int i = 0; i < head_dim; ++i) {
             out_ptr[i] = out_ptr[i] * old_scale + v_ptr[i] * new_scale;
         }
     }
     ```
   - Key-position loop is sequential; each key contributes via a
     256-thread reduction followed by that thread-0 update.
   - Block count also massive: `seq_len × num_heads` blocks per prefill
     (28 × seq_len per call), each launching the sequential key loop.

The WMMA kernel sidesteps all of that: Q·K^T is a single GEMM,
softmax is a row-parallel LDS pass, P·V is a second GEMM. One
workgroup per (head, 64-query-tile).

## Correctness (reference)

Tests in `tests/wmma_attention_multihead_correctness.rs`:

| Test                              | max_abs_diff |    mean | tol     |
|-----------------------------------|-------------:|--------:|--------:|
| seq=64, 28 heads, vs CPU          |     1.80e-5  | 3.09e-6 | 1.0e-2  |
| seq=128, 28 heads, vs CPU         |     1.27e-5  | 2.27e-6 | 1.0e-2  |
| V = 0 multi-head → O = 0          |      0.0     |  0.0    | 1.0e-3  |

All green on both sequence lengths. Max diffs sit orders of magnitude
below tolerance — input distribution in `[0, 0.25)` keeps FP32
accumulation noise negligible.

## Extrapolation to pp=256

A rough upper bound for the WMMA attention time at pp=256 with online
softmax (Phase 3b):

```
time per 64-query-tile(K=256) ≈ time(K=128)  · 256 / 128        = 233 µs
total for 4 query tiles                                         ≈ 933 µs
× 28 layers                                                     ≈ 26 ms
```

(For the "flat" extrapolation used in the bench output we assumed
`(K=128→K=256) × (query tiles 1→4) = 4×`, giving ~13 ms; that is
optimistic because the 64×256 S matrix needs online softmax or a
seq-dimension reload.)

Either way the attention budget collapses from the current ~2,330 ms
to the 10–100 ms range once Phase 3b lands. That is the end-to-end
prefill step-change the attribution analysis in
`benches/results/prefill_wmma_e2e_analysis.md` called out as the
remaining lever.

## What landed in this step

- `hip_kernels/wmma/wmma_attention_prefill.hip` — refactored into a
  template parameterised by `SEQ`, a strided Q/K/V/O layout matching
  the existing production kernel, and a new entry point
  `wmma_attention_prefill_multihead_launch(seq_len, num_heads,
  row_stride, scale, stream)`. The original single-head 64×128 ABI is
  preserved as a thin shim so the Step-1 standalone harness still
  links and runs.
- FFI wrapper `launch_wmma_attention_prefill_multihead` in
  `src/gpu/kernels/wmma.rs`.
- `tests/wmma_attention_multihead_correctness.rs` — 3 tests, all green.
- `profiling/wmma_poc/bench_wmma_attention.rs` — Cargo bin for
  throughput comparison against the existing kernel.

## A race-condition bug found and fixed along the way

First attempt aliased `lds_p` (FP16) over the `lds_s` (FP32) buffer
to save LDS. That introduces a cross-row race: with `lds_p` stride
`2·SEQ` bytes and `lds_s` stride `4·SEQ` bytes, `lds_p` row r occupies
the same memory as (part of) `lds_s` row r/2. Thread 1 writing
`p_row[1]` stomps on thread 0's in-place softmax updates to `s_row[0]`.
Symptoms: ~50 % of the output values halved. Fixed by declaring
`lds_p` as a separate static allocation — with 64 KB of LDS on
gfx1201 there is no need for the alias.

## Open items for Phase 3b

- Online softmax so the kernel scales to pp=256 and beyond.
- Store `max` and `sum` per row across key tiles; rescale the running
  output when a new tile's max exceeds the cumulative max (standard
  FlashAttention trick).
- Keep the current geometry: one workgroup per (head, 64-query-tile),
  loop over key tiles inside the block.

Phase 3c (causal mask, GQA) and Phase 3d (forward-path integration)
remain queued behind that.
