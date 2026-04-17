# Architecture Notes

Empirical findings about the GPU/CPU micro-architecture that are relevant across the whole project and drive optimization decisions.

## Memory-controller pipelining on sequential GEMV dispatches (RDNA 4)

**Hardware:** RX 9070 XT (gfx1201, RDNA 4), ROCm 7.2.1

On the RX 9070 XT (gfx1201, RDNA 4) the memory controller pipelines sequential GEMV dispatches against the same weight matrix so efficiently that batched dispatches deliver no measurable bandwidth advantage.

Confirmed by three independent experiments:

1. **Tiled GEMV FFN-Down** (~1.5% throughput gain vs. an expected ~15%): The FFN-down projection (18944→3584, ~135 MB Q4_0) reads the weight matrix N times in the sequential fallback. The tiled kernel reads it once. The expected gain was proportional to the bandwidth saved — actual measurement: ~9 μs/layer instead of the predicted ~150 μs/layer. Documented in `docs/batched_verify.md` (section "Memory-Controller Pipelining").

2. **Batched lm_head** (~0.4% verify-overhead reduction vs. expected ~8%): The lm_head projection (3584→152064, ~307 MB Q4_0) was converted from N sequential dispatches to 1 batched dispatch. Non-layer overhead dropped by ~114 μs/step instead of the predicted ~2,500 μs. Documented in `benches/results/batched_lm_head_analysis.md`.

3. **Buffer-traffic validation (Fused FFN)** (~1.2% verify-overhead reduction vs. expected 8–14%): A micro-benchmark simulated the FFN chain in three variants — separate kernels with VRAM intermediates, a single dispatch with VRAM round-trips, and a tiled fusion keeping the intermediate in registers/LDS. Maximum achievable gain: ~224 µs/step. The intermediate-buffer traffic (~150 KB/layer) is overshadowed by the weight-matrix traffic (~150 MB/layer) by a factor of ~500. Documented in `profiling/results/BUFFER_TRAFFIC_ANALYSIS.md`.

### Mechanism

The GPU command processor overlaps the tail of one GEMV kernel with the head of the next whenever both touch the same address ranges and no explicit synchronization sits between them. The memory controller holds near-full bandwidth utilization across kernel boundaries. The "load once instead of N times" model overestimates the gain by a factor of ~20×.

### Consistent overshoot pattern in bandwidth modelling

All three experiments exhibit the same pattern: naive bandwidth arithmetic overestimates the real gain on RDNA 4 by about an order of magnitude.

| Experiment                    | Expected gain        | Measured gain     | Overshoot factor |
|-------------------------------|---------------------:|------------------:|-----------------:|
| Tiled GEMV FFN-Down           |          2–4 ms/step |     ~250 µs/step  |             ~12× |
| Batched lm_head               |       ~2,500 µs/step |     ~114 µs/step  |             ~22× |
| Buffer-traffic validation     | ~1,500–2,500 µs/step |     ~200 µs/step  |             ~10× |

This is not measurement noise — it is a stable property of the memory pipeline on this architecture. The naive model treats memory traffic as a serial cost that scales linearly with the number of dispatches. In reality the memory controller pipelines the accesses such that the second, third, …, Nth access to the same addresses is nearly free (L2/L3 hits, overlapped streaming, no repeat memory-request latency).

Consequence: **"Load N times" costs almost the same as "load once" on RDNA 4, provided the accesses hit the same address ranges without explicit synchronization in between.** Optimizations whose only effect is reducing N-fold loads to a single load do not produce a measurable gain.

### Consequences for optimization

The optimization lever on this architecture sits at **algorithmic changes** and at **compute patterns with unpredictable access patterns** — not at dispatch batching or buffer-traffic elimination for bandwidth-bound kernels.

Concretely:

- **Fused FFN was confirmed as not worthwhile by the micro-benchmark** (~200 µs realistic saving, threshold was 1,500 µs). Do not implement. The dominant FFN cost is the weight-matrix traffic, which fusion cannot remove.
- **Spec-decode verify optimization has plateaued.** Target-verify is ~88% GEMV execution against the weight matrix — bandwidth-limited at ~640 GB/s (RX 9070 XT spec), not limited by dispatch overhead or buffer traffic. Further micro-optimizations within the GEMV paradigm yield < 2%.
- **Batching bandwidth-bound ops** (GEMV, attention with long KV cache) only saves marginal dispatch overhead (~2.7 µs/dispatch + sync elimination). Stream pipelining already covers the sync elimination.
- **Kernel fusion** is only worth it when it merges *different* memory-access patterns (e.g. elementwise + GEMV eliminating a store/load round-trip for non-cached addresses), not when it batches identical patterns or removes intermediates that already fit in L2.
- **The real optimization levers are algorithmic** (GEMV → GEMM for prefill) and live at **compute patterns with unpredictable access patterns** (attention tiling at long context, where the KV cache spills out of L2). These differ qualitatively from the previous experiments because they have different memory-access patterns where the memory-controller pipelining effect is weaker.

### WMMA matrix-core usage on RDNA 4 (gfx1201)

hipBLAS / Tensile in ROCm 7.2 does **not** select matrix-core kernels for gfx1201 — confirmed by a `rocprofv3 --kernel-trace` on the hipBLAS prefill path: every projection dispatches a Tensile kernel whose mangled name carries `FMA` (VALU fused-multiply-add), not `MFMA` / `WMMA`. Zero wmma / mfma / xdlops kernels appear anywhere in the full 26-kernel trace. Full write-up in `profiling/results/hipblas_matrix_core_check.md`.

ROCmForge works around this with a hand-written WMMA prefill kernel built on the `__builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12` intrinsic. The kernel consumes Q4_0 weights directly — nibble unpack and scale multiplication happen inline into LDS, so no FP16 weight scratch is ever materialised. Register-layout reference (from the AMD matrix-instruction calculator): `docs/wmma_register_layout_gfx12.md`; kernel source: `hip_kernels/wmma/wmma_gemm_q4_0.hip`.

Isolated GEMM throughput on the three Qwen2.5-7B prefill shapes (pp=256; 49 TFLOPS FP16 peak reference):

| Shape                     | hipBLAS µs | WMMA Q4_0 µs | Speedup | % of peak |
|---------------------------|-----------:|-------------:|--------:|----------:|
| QKV/O (256×3584×3584)     |        644 |          254 |  2.54×  |   52.9 %  |
| Gate/Up (256×18944×3584)  |      3,200 |        1,213 |  2.64×  |   58.5 %  |
| Down (256×3584×18944)     |      3,328 |        1,294 |  2.57×  |   54.8 %  |

End-to-end prefill on Qwen2.5-7B Q4_0 at pp=256 goes from 86 tok/s (hipBLAS) to 92.4 tok/s (WMMA). The modest 7–8 % headline gain reflects the fact that GEMM is already only ~4 % of prefill wall-clock at this model size — prefill attention dominates (~84 %) and is the next optimisation target. LDS bank-conflict tuning and double-buffering would push the WMMA kernel from 55 % toward 80 % of peak but contribute ≤ 2 % end-to-end; we parked them for that reason.

### Open questions

- Whether this pipelining effect also occurs to the same degree on RDNA 3 (gfx1100, RX 7900 XT) has not been measured. The memory-controller architecture differs (Infinity Cache vs. no Infinity Cache on RDNA 4). A comparison experiment on gfx1100 would be informative.
- Whether the overshoot pattern also applies to **non-elementwise** fusions (e.g. attention + FFN in one kernel). Hypothesis: yes, as long as the dominant cost is a bandwidth-bound GEMV — once the cost shifts (compute-bound, irregular accesses) the picture changes.
- Whether **WMMA / matrix instructions** on RDNA 4 change the picture. They use a different execution pipeline (matrix cores) and may have different pipelining characteristics — particularly if the matrix-core scheduler interacts differently with the memory controller than the vector-ALU scheduler does.

## CPU target platform

**Primary CPU:** AMD Ryzen 9 7945HX (Zen4, 16C/32T, AVX-512 VNNI, 64 MB L3, DDR5 dual-channel ~77 GB/s)

ROCmForge has a CPU fallback path (when `--gpu` is not set) that currently is not SIMD-optimized end-to-end. Zen4 provides AVX-512 with VNNI extensions — hardware-accelerated INT8 dot products — directly relevant to Q4_0 / Q8_0 inference.

Optimization angles:

- **AVX-512 GEMV kernel for Q4_0:** The biggest single lever. Unpack Q4_0 blocks, multiply against a Q8_0-quantized input, accumulate via VNNI. A 512-bit register processes 64 INT8 values per cycle (2× AVX2, 4× SSE).
- **Multi-threaded inference:** Partition the output dimension of the GEMV across threads. At 3584 output elements and 16 cores, ~224 elements per thread.
- **Cache-aware tiling:** Use L2 (1 MB/core) and L3 (64 MB shared) for weight tiles to minimize DRAM traffic.
- **Heterogeneous spec-decode:** Draft model (0.5B) on CPU, target (7B) on GPU, running in parallel. Eliminates the ~10% draft-GPU overhead from the spec-step cost analysis. Prerequisite: the CPU path must be fast enough not to block the GPU.

The RDNA-4 memory-controller pipelining pattern above does not transfer 1:1 to CPU DRAM. Zen4 has its own prefetchers and a different memory hierarchy (per-core L1/L2, shared L3, DDR5 controller) — optimization heuristics must be validated empirically before carrying the RDNA-4 findings over.

### Orchestration trap on small models (CPU)

For Qwen2.5-0.5B (hidden_size = 896) the per-call GEMV compute time is so short (~150 µs including Rayon dispatch) that orchestration overhead dominates. Measured: 80 ms/token versus a theoretical bandwidth ceiling of ~3.4 ms (260 MB weight read / 77 GB/s DDR5) — **a factor of 24× above the memory ceiling**.

Causes:

- Rayon fork-join overhead per GEMV call (~192 calls/token; each `par_iter_mut` waits on a sync barrier).
- Scalar attention (`flash_attn_decode`), RMSNorm, RoPE, SiLU — no SIMD vectorization beyond the compiler's auto-vectorization.
- Scalar Q8 input quantization (`quantize_q8_0_single` — byte-serial loop).

Empirically confirmed by the AVX-512 VNNI experiment (commit `d0e4f07`, results in `benches/results/cpu_avx512_analysis.md`): **16–19% kernel speedup on 7B shapes, 0% end-to-end on 0.5B**, because the kernel is not the bottleneck at that model size. Heterogeneous spec-decode (draft on CPU) would require a fundamental CPU pipeline rewrite, not better GEMV kernels.

### CPU performance gap versus llama.cpp

ROCmForge's CPU path on the Ryzen 9 7945HX vs. llama.cpp on the same machine:

| Model       | ROCmForge   | llama.cpp (estimated) | Factor |
|-------------|------------:|----------------------:|-------:|
| 0.5B Q4_0   |  12.1 tok/s |     ~50–80 tok/s      |    ~5× |
| 7B Q4_0     |   0.7 tok/s |       ~6–8 tok/s      |   ~10× |

The gap is **not SIMD-bound** — AVX-512 VNNI is implemented and is 16–19% faster than AVX2 in isolation on 7B shapes. It lives in the entire CPU forward pipeline: Rayon overhead per GEMV call, scalar non-GEMV operations, missing kernel fusion, and apparently duplicate memory traversals (input quantization + GEMV). A competitive CPU path would be a standalone project — not a by-product of the GPU optimization work.

## WMMA FlashAttention on RDNA 4 (Phase 3d)

The prefill attention kernel uses the same `__builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12` intrinsics as the GEMM kernel. FlashAttention-style with online softmax, 64 × 64 Q/KV tiles, and causal masking with zero-work elimination on tiles strictly above the diagonal.

The pre-existing scalar `flash_attn_prefill_strided_kernel` had a fundamental parallelisation failure: the softmax update ran on thread 0 alone in a 128-element serial loop. The WMMA kernel parallelises the whole attention path — `QKᵀ`, softmax, `P × V` — over every thread and every matrix core. At `seq_len = 256`, isolated attention dropped from 78,412 µs to 221 µs (355×); end-to-end prefill at pp = 256 went from 2,773 ms → 411 ms (6.7×).

GQA is handled with separate `q_row_stride` and `kv_row_stride`. The 7:1 KV sharing between Q heads flows through the L2 cache, not explicit LDS sharing. At `head_dim = 128` and `seq ≤ 512` the KV tiles fit comfortably in the 64 MB L2, and the measured cost of GQA overhead vs. a `gqa_ratio = 1` run is ~6 % at short sequences (where the extra stride bookkeeping is visible) and neutral at long sequences. Explicit LDS KV-dedup was evaluated and parked — attention is now ~1.5 % of pp = 256 prefill time, below the threshold where further tuning pays off.

Causal-mask zero-work elimination saves roughly `(seq_len / TILE_KV − 1) / 2` tiles per Q-tile. At pp = 512 the theoretical saving is 44 % (28 fully-skipped tiles out of 64); measured saving vs. a non-causal WMMA run is ~25 %. The gap comes from diagonal tiles that still run the full GEMM with an extra per-element mask check. Below pp = 256 the mask infrastructure costs more than the few skippable tiles save, so causal and non-causal WMMA differ by < 15 %. We keep the mask always-on because correctness on a real model requires it.
