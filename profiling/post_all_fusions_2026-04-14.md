# Post-All-Fusions Profiling — 2026-04-14

## Setup

- **GPU:** AMD Radeon RX 9070 XT (gfx1201, RDNA4)
- **ROCm:** 7.2, rocprofv3 1.1.0
- **Model:** Qwen2.5-0.5B-Instruct Q4_0
- **Workload:** 1 prompt token ("Hello"), 64 decode tokens, greedy sampling
- **Branch:** `feat/auto-detect-gpu-arch`
- **Fusions applied:**
  1. QKV+RoPE+KV-Write (Phase 1.3) — 3 kernels → 1
  2. Norm+QKV+RoPE+KV-Write (Phase 3.1) — 4 kernels → 1
  3. Norm+Gate+Up+SiLU (Phase 3.3) — 2 kernels → 1

## Three-Stage Progression: Graph Launch Overhead

| Metric | Baseline | After QKV Fusion | After All Fusions | Total Delta |
|--------|----------|------------------|--------------------|-------------|
| hipGraphLaunch total (64 tok) | 438.3 ms | 324.6 ms | **213.9 ms** | **-51.2%** |
| hipGraphLaunch avg/token | 6.85 ms | 5.07 ms | **3.34 ms** | **-51.2%** |
| hipGraphLaunch stddev | — | 56.5 µs | 48.1 µs | stable |
| Graph nodes/token | 219 | 172 | **124** | **-95 nodes (-43.4%)** |

**hipGraphLaunch dropped from 6.85 ms to 3.34 ms per token — a 2× reduction.** Each of the 95 eliminated nodes saved ~37 µs of launch overhead under the profiler.

## GPU Kernel Time Breakdown (Decode, 64 Tokens)

| Kernel | Baseline Calls | Current Calls | Baseline µs | Current µs | Avg µs/call | Share |
|--------|---------------|---------------|-------------|------------|-------------|-------|
| GEMV Residual (attn_out+ffn_down) | 3072 | 3072 | 31,025 | 31,156 | 10.1 | 32.9% |
| **Norm+Gate+Up+SiLU (fused)** | — | 1536 | — | 21,824 | 14.2 | 23.0% |
| LM-Head | 65 | 65 | 14,517 | 14,992 | 230.6 | 15.8% |
| Flash Attention | 1536 | 1536 | 10,870 | 11,020 | 7.2 | 11.6% |
| **Norm+QKV+RoPE+KV (fused)** | — | 1536 | — | 9,229 | 6.0 | 9.7% |
| RMS Norm | 3137 | **65** | 5,346 | **111** | 1.7 | 0.1% |
| Gate+Up+SiLU (non-fused) | 1536 | **0** | 19,009 | **0** | — | — |
| QKV GEMV (non-fused) | 1536 | **0** | 11,389 | **0** | — | — |
| RoPE Q | 1536 | **0** | 2,451 | **0** | — | — |
| KV Write+RoPE K | 1536 | **0** | 2,317 | **0** | — | — |

### Key Observations

1. **RMS Norm calls: 3137 → 65.** All per-layer norms are now fused into the GEMV kernels. The remaining 65 calls are prefill norms (48 batched) + output norm before LM-Head (17 calls).

2. **Norm+Gate+Up+SiLU** at 21,824 µs is ~2,800 µs more than the non-fused Gate+Up alone (19,009 µs). This is expected — the fused kernel now also does the norm computation that previously took ~2,600 µs separately. Net: eliminates ~2,600 µs norm + ~1,500 µs launch overhead vs. only ~2,800 µs added to the GEMV kernel → **~1,300 µs net savings per 64 tokens**.

3. **Norm+QKV+RoPE+KV** at 9,229 µs vs. the previous QKV-only fusion at 12,048 µs. The norm-fused version is actually *faster* per call (6.0 µs vs 7.8 µs) — likely because the data is already warm in shared memory when the GEMV phase begins.

## Per-Token Breakdown (Decode)

| Operation | Baseline | After QKV Fusion | After All Fusions | Delta vs Baseline |
|-----------|----------|------------------|--------------------|-------------------|
| GEMV Residual | 485 µs | 487 µs | 487 µs | ~0 |
| Gate+Up+SiLU | 298 µs | 302 µs | — | — |
| **Norm+Gate+Up+SiLU** | — | — | **341 µs** | — |
| LM-Head | 223 µs | 225 µs | 234 µs | ~0 |
| QKV (3 kernels) | 252 µs | — | — | — |
| QKV+RoPE+KV (fused) | — | 188 µs | — | — |
| **Norm+QKV+RoPE+KV** | — | — | **144 µs** | — |
| Attention | 170 µs | 173 µs | 172 µs | ~0 |
| RMS Norm (per-layer) | 82 µs | 85 µs | **~2 µs** | **-80 µs** |
| Argmax | 5 µs | 5 µs | 5 µs | ~0 |
| **GPU Total/token** | **~1,515 µs** | **~1,465 µs** | **~1,385 µs** | **-130 µs (-8.6%)** |

## End-to-End Time Budget (Under Profiler)

| Component | Baseline | After QKV Fusion | After All Fusions | Delta |
|-----------|----------|------------------|--------------------|-------|
| GPU kernel compute | ~1.5 ms | ~1.5 ms | ~1.4 ms | -8.6% |
| hipGraphLaunch | 6.85 ms | 5.07 ms | **3.34 ms** | **-51.2%** |
| hipStreamSynchronize | 1.1 ms | 1.1 ms | 1.0 ms | ~0 |
| **Total/token (profiler)** | ~8.0 ms | ~5.8 ms | **~4.4 ms** | **-45%** |
| **tok/s (profiler)** | 124.8 | 170.9 | **~226** | **+81%** |

## Kernel Launch Counts per Decode Token

| Operation | Baseline | Current | Delta |
|-----------|----------|---------|-------|
| GEMV Residual (attn_out+ffn_down) | 48 | 48 | 0 |
| Norm+Gate+Up+SiLU (fused) | 0 | **24** | +24 |
| Gate+Up+SiLU (separate) | 24 | **0** | -24 |
| Norm+QKV+RoPE+KV (fused) | 0 | **24** | +24 |
| QKV (separate) | 24 | **0** | -24 |
| RoPE Q (separate) | 24 | **0** | -24 |
| KV Write+RoPE K (separate) | 24 | **0** | -24 |
| Attention | 24 | 24 | 0 |
| RMS Norm (per-layer) | 48 | **0** | **-48** |
| Residual Add | 2 | 2 | 0 |
| LM-Head | 1 | 1 | 0 |
| Argmax | 2 | 2 | 0 |
| Output RMS Norm | 1 | 1 | 0 |
| **Total** | **~219** | **~124** | **-95 (-43.4%)** |

## Benchmark Results (Without Profiler Overhead)

| Metric | Baseline (RDNA4) | After QKV Fusion | After All Fusions | Total Delta |
|--------|-----------------|------------------|--------------------|----|
| Decode avg | 558.3 tok/s | 593.7 tok/s | **~630 tok/s** | **+12.9%** |
| Prefill avg | 458.8 tok/s | 499.8 tok/s | **~510 tok/s** | **+11.2%** |

## Where Time Goes Now — Bottleneck Analysis

### GPU Kernel Time Distribution (Decode)

```
GEMV Residual (attn_out+ffn_down)  ████████████████████  32.9%  — 48 launches, 10 µs avg
Norm+Gate+Up+SiLU (fused)          ██████████████        23.0%  — 24 launches, 14 µs avg
LM-Head                            ██████████            15.8%  — 1  launch,  231 µs avg
Flash Attention                    ███████               11.6%  — 24 launches, 7 µs avg
Norm+QKV+RoPE+KV (fused)          ██████                 9.7%  — 24 launches, 6 µs avg
Everything else                    ██                     7.0%
```

### hipGraphLaunch vs GPU Compute

```
hipGraphLaunch (host overhead)     ████████████████████████████  56.5%
GPU kernel compute                 ████████████████              33.4%
hipStreamSynchronize               ██████                        17.0%
```

**hipGraphLaunch remains the dominant bottleneck**, consuming 56.5% of profiled time. However, its per-token cost has dropped from 6.85 ms to 3.34 ms — the 124 remaining nodes are close to the minimum achievable with independent GEMV dispatches.

## Remaining Optimization Targets

| Priority | Target | Current Cost | Potential Gain | Notes |
|----------|--------|-------------|----------------|-------|
| 1 | **GEMV Residual** | 31.2 ms (32.9%) | Medium | 48 launches for attn_out+ffn_down — could fuse residual-add into GEMV |
| 2 | **LM-Head** | 15.0 ms (15.8%) | High | 231 µs/call, single vocab GEMV — vocabulary sharding or tiling |
| 3 | **Block-size tuning** | — | Low-Med | Currently using 256-thread blocks everywhere — profile occupancy |
| 4 | **fp16 KV cache** | — | Low | Halve KV memory bandwidth, enable larger context |

## GFX12 Instruction Analysis

The current Q4_0 dequantization in all GEMV kernels uses classical shift-and-mask code:
```cpp
const int lo = (q & 0x0F) - 8;
const int hi = (q >> 4) - 8;
return lo * x_lo + hi * x_hi;
```

GFX12 (RDNA4) supports `v_dot4_i32_iu8` and `v_dot8_i32_iu4` instructions for packed integer dot products. These are **not yet utilized** — leveraging them could improve the Q4_0×Q8_0 inner loop throughput by ~2× for the dot-product portion. This would primarily benefit the GEMV Residual kernel (32.9% of GPU time) and the Norm+Gate+Up kernel (23.0%).
