# hipBLAS Prefill — north-star benchmark & verdict

**Commit:** `b00acbf+`
**GPU:** RX 9070 XT (gfx1201, RDNA 4), 16 GB VRAM, ~640 GB/s
**Runtime:** ROCm 7.2.1, hipBLAS linked from `/opt/rocm/lib/libhipblas.so.3`
**Model:** Qwen2.5-7B-Instruct Q4_0
**Date:** 2026-04-17

## TL;DR

**Peak Qwen2.5-7B Q4_0 prefill throughput with hipBLAS: 86 tok/s at pp=256. Baseline (custom GEMM): 64 tok/s. Speedup: +25–35 %.**

**Verdict: < 200 tok/s threshold → deeper problem than the GEMV→GEMM switch.** hipBLAS Hgemm on RDNA 4 (gfx1201) does not deliver the hoped-for 10× leap. Root-cause analysis is needed before committing to Phase 2 (custom WMMA kernel with inline dequant). Nevertheless the hipBLAS path is a real, consistent win on medium-to-long prompts and is committed as the default for `seq_len ≥ 32`.

## Setup

- Phase 0 plan: `docs/prefill_gemm_plan.md`.
- hipBLAS FFI: `src/gpu/hipblas_ffi.rs` (handle, `hipblasHgemm`).
- Prefill GEMM wrapper: `src/gpu/prefill_gemm.rs` (dequant Q4_0 → FP16, FP32 ↔ FP16 conversion, row-major → column-major compute-`C^T` trick).
- Dequant kernel: `hip_kernels/quant/dequant_q4_0_to_f16.hip` (coalesced, one thread per output element, one Q4_0 scale per shared-memory slot).
- Dispatch: `src/gpu/ops.rs::gpu_dispatch_gemm` picks the hipBLAS path for `seq_len ≥ PREFILL_GEMM_THRESHOLD=32` on Q4_0 tensors without `needs_transpose`.
- Opt-out: `ROCMFORGE_DISABLE_HIPBLAS_PREFILL=1`.
- VRAM: device-level lazy scratch (`weight_f16`, `act_in_f16`, `act_out_f16`) grown on demand. Max working set ≈ 150 MB in practice.

## Results (Qwen2.5-7B Q4_0, greedy, single forward)

Raw data: `prefill_hipblas_b00acbf_1776411293.json`. Each cell is the median of 3 runs.

| prompt len | GEMV baseline (tok/s) | hipBLAS (tok/s) | Speedup |
|-----------:|----------------------:|----------------:|--------:|
|         19 |                  59.9 |            52.9 |  −12 %  |
|         64 |                  62.4 |            77.8 |  +25 %  |
|        128 |                  63.6 |            82.4 |  +30 %  |
|        256 |                  63.8 |            86.0 |  +35 %  |
|        512 |                  50.3 |            63.1 |  +25 %  |

Break-even sits around `seq_len ≈ 32`. Below that the hipBLAS setup cost (dequant + FP conversions + `hipblasHgemm` dispatch) outweighs the GEMM benefit, so the dispatch threshold is set accordingly and the GEMV paths still handle short prompts.

At `pp=512` both paths slow down together — the O(N²) cost of prefill attention kicks in and is unaffected by our GEMM change.

## Why not 500 tok/s?

The prompt's fail-fast framework split the verdict into three buckets:

- ≥ 500 tok/s → hardware can do it, plan Phase 2 (custom WMMA with inline dequant).
- 200–500 tok/s → partial, evaluate.
- < 200 tok/s → deeper problem than GEMV→GEMM, root-cause analysis first.

We sit at **86 tok/s peak**, clearly in the third bucket. The 10–20× leap that the llama.cpp pp19 number (1,092 tok/s) implies is *not* unlocked by a naive hipBLAS Hgemm + dequant drop-in on RDNA 4.

Three candidate root causes, in order of current suspicion:

1. **hipBLAS Hgemm on gfx1201 may not be using matrix cores efficiently.** RDNA 4 exposes WMMA instructions via the compiler but hipBLAS on ROCm 7.2 has only recent support for gfx1201. A plausible scenario: it falls back to an FP16 VALU path that is no faster than our hand-written `gemm_q4_0_f32` kernel. We have no direct confirmation — need to check with `rocprofv3` what kernels hipBLAS actually launches.
2. **Attention dominates at pp ≥ 256.** `flash_attn_prefill` is O(N²) per head. At pp=512 its compute is comparable to the GEMM load; any GEMM acceleration gets amortized. This matches the measured slowdown at pp=512.
3. **Non-linear ops (RMSNorm, RoPE, SiLU, residual) are not themselves accelerated.** They are a small fraction of the per-layer time but they add up at scale.

The dequant overhead itself is *not* the bottleneck. Rough envelope: 14 GB of f16 weight writes per prefill at 640 GB/s = ~22 ms. At pp=256 the full prefill takes ~2.96 s, so dequant is below 1 % of wall-clock.

Point (1) is the one worth investigating before going to Phase 2. If hipBLAS really is not tapping the matrix cores, a custom WMMA kernel *could* still deliver the step change we hoped for — but that needs to be confirmed first. If hipBLAS is tapping them and the kernel is just not running compute-bound, a custom WMMA kernel will not rescue us either.

## What we shipped in Phase 1

- hipBLAS linked into the GPU build (`build.rs`).
- FFI wrapper for `hipblasCreate/Destroy/SetStream/Hgemm` (`src/gpu/hipblas_ffi.rs`).
- Q4_0 → FP16 dequant kernel (`hip_kernels/quant/dequant_q4_0_to_f16.hip`) plus FP32↔FP16 conversion helpers.
- Device-owned lazy FP16 scratch (`GpuDevice::prefill_f16_*`). Grows on demand, single allocation per shape class.
- Dispatch hook in `gpu_dispatch_gemm` with `PREFILL_GEMM_THRESHOLD=32` and opt-out flag `ROCMFORGE_DISABLE_HIPBLAS_PREFILL=1`.
- Correctness test `tests/prefill_hipblas_matches_gemv.rs` — runs binary twice, asserts that the first decoded token matches between hipBLAS and the GEMV path (FP16 accumulation inside hipBLAS can drift later tokens, hence only first-token equality).
- Sweep script `benches/bench_prefill_hipblas.fish`.

## Recommended next step — before committing to Phase 2

**Profile the hipBLAS kernel selection.** Before investing in a custom WMMA kernel with inline dequant (a multi-week build), confirm one thing: does hipBLAS on gfx1201 actually pick a matrix-core kernel for our shapes?

```
rocprofv3 --kernel-trace -- ./target/release/rocmforge \
    --model ~/models/Qwen2.5-7B-Instruct-Q4_0.gguf \
    --prompt "$(python3 -c 'print(" ".join(["word"] * 255))')" \
    --max-tokens 1 --gpu --no-template --temperature 0.0 --top-p 1.0
```

Decision tree based on that trace:

- **Kernel name contains `wmma` / `mfma` / matrix-core mnemonic, reasonable occupancy:** hipBLAS is using matrix cores. The gap is elsewhere (attention, non-linear ops, memory). Phase 2 (custom WMMA) will not help — pivot to attention tiling or hipBLASLt with INT8 inputs.
- **Kernel name is a generic `gemm_*_NN` or `hgemm_*` without matrix-core markers, low TFLOPS utilisation:** hipBLAS is on a VALU fallback for gfx1201. A custom WMMA kernel is still the right next bet — Phase 2 can proceed.
- **Mixed (some calls matrix-core, some not):** figure out the shape threshold and dispatch accordingly.

## Consistency notes

- All runs used ROCMFORGE_DISABLE_HIPBLAS_PREFILL as the single toggle, with the same model file and the same greedy config.
- The hipBLAS path is on by default from this commit forward. If the profiling above points at something different, we can always flip the default back with the existing flag — no additional code change needed.
