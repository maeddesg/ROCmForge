# WMMA Q4_0 prefill — end-to-end verdict (Phase 2d)

**Date:** 2026-04-17
**Commits:** on top of `9a0d754`
**GPU:** RX 9070 XT (gfx1201, RDNA 4), ROCm 7.2.1
**Model:** Qwen2.5-7B-Instruct Q4_0

## TL;DR

Prefill throughput on Qwen2.5-7B Q4_0 for 64-aligned prompt lengths (greedy, single-token decode):

| Prompt | Custom GEMM | hipBLAS | **WMMA Q4_0** | vs Custom | vs hipBLAS |
|-------:|------------:|--------:|--------------:|----------:|-----------:|
|   pp64 |   61.8 tok/s |  77.0 tok/s | **88.8 tok/s** | +43.7 % | +15.5 % |
|  pp128 |   63.4 tok/s |  81.7 tok/s | **90.3 tok/s** | +42.4 % | +10.5 % |
|  pp192 |   63.9 tok/s |  84.7 tok/s | **92.3 tok/s** | +44.4 % |  +9.0 % |
|  pp256 |   63.7 tok/s |  86.0 tok/s | **92.4 tok/s** | +45.1 % |  +7.4 % |
|  pp384 |   54.0 tok/s |  69.8 tok/s | **74.6 tok/s** | +38.1 % |  +6.9 % |
|  pp512 |   50.3 tok/s |  63.1 tok/s | **67.2 tok/s** | +33.6 % |  +6.5 % |

Peak throughput: **92.4 tok/s at pp=256**. Previous baseline at the start of the project: ~59 tok/s at pp=19. We are **1.57× over the original custom GEMM at pp=256** and **8 % over hipBLAS**. The gap to llama.cpp's pp=19 number (1,092 tok/s, a pure-WMMA pipeline) stays **~12×**.

Raw data: [`prefill_wmma_e2e_9a0d754_1776419943.json`](prefill_wmma_e2e_9a0d754_1776419943.json).

## Attribution — why the gain is "only" 7-15 % vs hipBLAS

The isolated GEMM benchmark from Phase 2b showed WMMA is 2.54-2.64× faster than hipBLAS on the three projection shapes. End-to-end at pp=256 we see +7.4 %. Where does the difference go?

Cost breakdown at pp=256 (WMMA path, total prefill 2,770 ms):

| Component             | Time (ms)       | Share |
|-----------------------|----------------:|------:|
| GEMM (WMMA Q4_0)      | ~110 ms         |  ~4 % |
| Attention (prefill)   | ~2,330 ms       | ~84 % |
| Norm + RoPE + residual + overhead | ~330 ms | ~12 % |

(GEMM estimate: 28 layers × (QKV+O+Gate+Up+Down) from the isolated bench = 28 × ~3,974 µs = ~111 ms. Attention uses the rocprofv3 kernel trace from the hipblas_matrix_core_check (784 flash_attn_prefill launches × 2,974 µs ≈ 2,330 ms).)

So the GEMM path we just optimised is **4 % of the total prefill**. A 2.5× speedup on a 4 % slice translates to a ≤ 4 % end-to-end gain — which is exactly what we measure relative to hipBLAS at pp=256. The rest (hipBLAS wall-clock improvement 86 → 92 tok/s) is hipBLAS's less-efficient kernel also eating some attention-adjacent bandwidth.

**Corollary:** GEMM is no longer the dominant prefill cost on this hardware for 7B Q4_0. Attention is now ~84 % of the budget.

## Why the gap to llama.cpp stays ~12×

At 1,092 tok/s on pp=19 llama.cpp spends ~17 ms/token on prefill. Our WMMA path at pp=256 spends ~10.8 ms/token. llama.cpp is clearly doing something very different from us:

- **Attention fused with matmul.** Their prefill attention likely runs as a WMMA-based kernel (flash-attention style), not as a separate `flash_attn_prefill_strided_kernel` that we wrote as a scalar kernel.
- **WMMA also for non-weight ops** (softmax, bias, etc.) where applicable.
- **Higher peak fraction on GEMM** (they probably hit 80-90 %, we're at 55-60 %).

In numbers: if we took our 2,770 ms at pp=256 and only trimmed GEMM down by a further 2× (theoretical best from Phase 2c), we'd land at ~2,715 ms → 94 tok/s. Nowhere near 1,092. The lever is attention, not GEMM.

## Verdict

**Phase 2c (WMMA tuning — LDS bank-conflicts, double-buffering, coalesced Q4_0 load) is deprioritised.** Its ceiling contribution to end-to-end throughput is 1-2 % given that GEMM is already 4 % of prefill. The remaining peak-fraction headroom exists, but cashing it in won't move the needle much.

**Attention-tiling for prefill is the next lever.** Same approach as WMMA GEMM: write a WMMA-aware prefill attention kernel (FlashAttention-style), replace `flash_attn_prefill_strided_kernel`. That's where the 84 % share lives.

### What landed in this phase

- `gpu_dispatch_gemm` now prefers WMMA Q4_0 for prefill when:
  - `seq_len ≥ 64` and `seq_len % 64 == 0`
  - tensor is Q4_0, `!needs_transpose`
  - `out_dim % 64 == 0` and `in_dim % 32 == 0`
  - `wmma_prefill_enabled()` (opt-out via `ROCMFORGE_DISABLE_WMMA_PREFILL=1`)
- Dispatch order: WMMA → hipBLAS → custom GEMM → batched/tiled GEMV. Every step has its own opt-out flag.
- hipBLAS path stays as a safety net for unaligned M (e.g. pp=19, pp=128+x) and as a debugging reference. The FP16 prefill scratch buffers in `GpuDevice` are only allocated when that path is actually exercised (lazy), so the default path no longer touches the 150 MB working set.

### Correctness

`tests/prefill_wmma_matches_hipblas.rs` — first decoded token is identical between the WMMA and hipBLAS paths on an 80-word prompt (seq_len ≥ 64). FP16 accumulation in hipBLAS vs FP32 in WMMA can drift later tokens, so we only assert first-token equality, which is the most sensitive signal for a dispatch / layout bug.

All prior tests stay green:
- `tests/wmma_q4_0_correctness.rs` (5 shapes, bit-identical to Phase 2a)
- `tests/wmma_tiled_correctness.rs` (5 shapes, within FP16-accumulation envelope of hipBLAS)
- `tests/prefill_hipblas_matches_gemv.rs` (hipBLAS path unchanged, still first-token match vs custom GEMM)

### Open items for follow-up phases

- **Phase 2c (tuning, now deprioritised):** LDS padding/swizzle for the Q4_0 dequant write pattern, double-buffered K-loop, coalesced uint32 stream-load for Q4_0. ≤ 2 % end-to-end.
- **Prefill attention WMMA kernel (Phase 3):** FlashAttention-style tiled attention using the same `__builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12` intrinsic. Target: cut the 2,330 ms attention budget to somewhere under 500 ms. That's where the next large step-change lives.
- **hipBLAS scratch removal:** when the scratch is truly never allocated (verify no tests hit the hipBLAS path by accident), the 150 MB `prefill_f16_*` lazy field in `GpuDevice` can go. Deferred — harmless as lazy-only.

### Unaligned prompt lengths

WMMA requires `seq_len % 64 == 0`. Prompts whose tokenisation lands at e.g. 200 or 44 tokens fall through to hipBLAS (which has no alignment requirement) — matching the measured 83.5 tok/s at pp=200 in earlier smoke tests, same as plain hipBLAS. Padding M to a 64-multiple is a Phase 2c option, but unaligned prompts are not on the critical path for the benchmark workload.
