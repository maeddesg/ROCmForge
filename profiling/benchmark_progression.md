# ROCmForge Decode Performance Progression

Kernel fusion results on Qwen2.5 Q4_0 decode, 10-run averages.

## 0.5B Model — Decode Throughput (tok/s, with HIP Graph, 128 tokens)

```
tok/s  0       100      200      300      400      500      600      700
       ├────────┼────────┼────────┼────────┼────────┼────────┼────────┤

 gfx1100 Baseline (RX 7900 XT)
       ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓  527

 gfx1201 Before Fusion (RX 9070 XT)
       ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓    508

 + QKV+RoPE+KV-Write Fusion
       ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓  594  (+17%)

 + Norm+QKV+RoPE+KV, Norm+Gate+Up+SiLU Fusions
       ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓  646  (+27%)
```

## 7B Model — Decode Throughput (tok/s, no graph, 64 tokens)

```
tok/s  0        20       40       60       80      100      120
       ├────────┼────────┼────────┼────────┼────────┼────────┤

 gfx1100 Baseline (RX 7900 XT)
       ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓  107

 gfx1201 Before Fusion (RX 9070 XT)
       ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓   106

 + All Fusions (final)
       ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓  107  (~0%)
```

> **Note:** HIP graph capture fails on the 7B model (stream capture error during Q8 quantization).
> At 7B scale the GPU is fully memory-bandwidth-saturated — graph launch overhead is negligible
> relative to the ~9 ms of weight-loading per token, so this is not a regression.

## Results — 0.5B (with HIP Graph)

| Stage | GPU | Graph Nodes | Decode tok/s | vs gfx1201 Start | vs gfx1100 |
|-------|-----|-------------|-------------|-------------------|------------|
| Original baseline | gfx1100 (RX 7900 XT) | 219 | 527 | — | — |
| RDNA4 port (before fusion) | gfx1201 (RX 9070 XT) | 219 | 508 | — | -3.6% |
| + QKV+RoPE+KV-Write fusion | gfx1201 | 172 | 594 | +16.9% | +12.7% |
| + All fusions (final) | gfx1201 | 124 | 646 | +27.2% | +22.6% |

## Results — 7B (no graph)

| Stage | GPU | Decode tok/s | vs gfx1100 |
|-------|-----|-------------|------------|
| Original baseline | gfx1100 (RX 7900 XT) | 107 | — |
| RDNA4 port (before fusion) | gfx1201 (RX 9070 XT) | 106 | -0.9% |
| + All fusions (final) | gfx1201 | 107 | ~0% |

At 7B the model is purely memory-bandwidth-bound (~614 GB/s on RX 9070 XT). Kernel fusion saves launch overhead, but that overhead is dwarfed by the time spent streaming 3.5 GB of weights per token. Both GPUs land at ~107 tok/s — the theoretical bandwidth ceiling for Q4_0 at this model size.

## What Changed

Three kernel fusions that reduce HIP graph node count from 219 to 124 per decode token:

1. **QKV+RoPE+KV-Write** — merges 3 separate kernels (QKV GEMV, RoPE-Q, KV-Write+RoPE-K) into one fused kernel. Saves 48 launches/token.
2. **Norm+QKV+RoPE+KV-Write** — folds the preceding RMSNorm into the above. Saves another 24 launches/token.
3. **Norm+Gate+Up+SiLU** — folds RMSNorm into the fused Gate+Up+SiLU GEMV. Saves 24 launches/token.

Total: **95 fewer kernel launches per token (-43%)**.

## Why This Works

Profiling with rocprofv3 shows `hipGraphLaunch` consumes the majority of per-token time on consumer RDNA GPUs:

| Metric | Before Fusion | After Fusion | Delta |
|--------|--------------|-------------|-------|
| Graph nodes per token | 219 | 124 | -43.4% |
| hipGraphLaunch per token | 6.85 ms | 3.34 ms | -51.2% |
| GPU kernel compute per token | 1.52 ms | 1.39 ms | -8.6% |

The GPU kernels themselves barely got faster — the win is almost entirely from reducing host-side launch overhead.

## Dead End: udot4 (v_dot4_u32_u8)

We also implemented GFX12 `v_dot4_u32_u8` optimized GEMV kernels (~4.5x ALU reduction). Benchmark result: **0% throughput improvement**. Batch-1 GEMV at ~614 GB/s memory bandwidth is completely memory-bound — ALU units are idle waiting for weight data regardless. The udot4 code has been removed to keep the branch clean.

## Test Setup

- **GPU:** AMD Radeon RX 9070 XT (gfx1201, RDNA4, 16 GB)
- **gfx1100 reference:** RX 7900 XT
- **ROCm:** 7.2
- **Model:** Qwen2.5-0.5B-Instruct Q4_0 GGUF
- **Benchmark:** 128 decode tokens, 1 warmup + 10 measured runs, greedy sampling
- **HIP graph:** enabled (`ROCMFORGE_ENABLE_DECODE_GRAPH=1`)
