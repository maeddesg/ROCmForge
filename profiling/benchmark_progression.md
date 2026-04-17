# ROCmForge Decode Performance Progression

All measurements on Qwen2.5 Q4_0, RX 9070 XT (gfx1201), ROCm 7.2.1 unless noted.

## Current Performance vs llama.cpp ROCm (April 16, 2026)

### Decode throughput (tok/s, tg128)

```
tok/s  0        50      100      150      200      250      300      350      400
       ├────────┼────────┼────────┼────────┼────────┼────────┼────────┼────────┤

 0.5B — llama.cpp ROCm
       ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓  358

 0.5B — ROCmForge (no full-decode graph)
       ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓  222

 7B — llama.cpp ROCm
       ▓▓▓▓▓▓▓▓▓▓▓▓  117

 7B — ROCmForge
       ▓▓▓▓▓▓▓▓  82
```

### Decode comparison table

| Model | ROCmForge | llama.cpp ROCm | Ratio |
|-------|-----------|----------------|-------|
| 0.5B Q4_0 (tg64) | — | 357 tok/s | — |
| 0.5B Q4_0 (tg128) | 222 tok/s | 358 tok/s | 62% |
| 0.5B Q4_0 (tg256) | — | 367 tok/s | — |
| 7B Q4_0 (tg64) | — | 117 tok/s | — |
| 7B Q4_0 (tg128) | 82 tok/s | 118 tok/s | 70% |
| 7B Q4_0 (tg256) | — | 118 tok/s | — |

### Prefill comparison table

| Model | ROCmForge | llama.cpp ROCm |
|-------|-----------|----------------|
| 0.5B Q4_0 (pp19) | 244 tok/s | 3,421 tok/s |
| 7B Q4_0 (pp19) | 59 tok/s | 1,092 tok/s |
| 7B Q4_0 (pp512) | — | 5,141 tok/s |

Prefill gap is due to ROCmForge using custom GEMV kernels vs llama.cpp using hipBLAS GEMM.

## Speculative Decoding (0.5B draft + 7B target)

| Config | tok/s | Acceptance | Avg tokens/step | Steps |
|--------|-------|------------|-----------------|-------|
| Baseline (no spec) | 82 | — | 1.0 | — |
| depth=1 | 57 | 44.7% | 0.4 | 38 |
| depth=2 | 40 | 26.0% | 0.5 | 37 |
| depth=3 | 33 | 20.6% | 0.6 | 35 |
| depth=5 | 22 | 11.7% | 0.6 | 36 |

> **Note:** Verify currently runs token-by-token (N+1 sequential target forwards).
> Even at 100% acceptance (counting prompt), there is no speedup because
> 4 forwards for 4 tokens = same as 4 single-token decodes.
> Batched verify (1 forward for N+1 tokens) is needed for real speedup.

## Historical Progression — 0.5B (with HIP Graph)

Kernel fusion results, 10-run averages, 128 tokens.

```
tok/s  0       100      200      300      400      500      600      700
       ├────────┼────────┼────────┼────────┼────────┼────────┼────────┤

 gfx1100 Baseline (RX 7900 XT)
       ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓  527

 gfx1201 Before Fusion (RX 9070 XT)
       ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓    508

 + QKV+RoPE+KV-Write Fusion
       ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓  594  (+17%)

 + All Fusions (Norm+QKV+RoPE, Norm+Gate+Up+SiLU)
       ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓  646  (+27%)

 Current (full-decode graph disabled due to RDNA4 bug)
       ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓  222
```

| Stage | GPU | Graph Nodes | Decode tok/s | Note |
|-------|-----|-------------|-------------|------|
| Original baseline | gfx1100 (RX 7900 XT) | 219 | 527 | — |
| RDNA4 port (before fusion) | gfx1201 (RX 9070 XT) | 219 | 508 | — |
| + QKV+RoPE+KV-Write fusion | gfx1201 | 172 | 594 | +17% |
| + All fusions (final) | gfx1201 | 124 | 646 | +27% |
| Full-decode graph disabled | gfx1201 | tail-only | 222 | RDNA4 graph bug |

## Historical Progression — 7B (no graph)

| Stage | GPU | Decode tok/s |
|-------|-----|-------------|
| Original baseline | gfx1100 (RX 7900 XT) | 107 |
| RDNA4 port (before fusion) | gfx1201 (RX 9070 XT) | 106 |
| + All fusions (final) | gfx1201 | 107 |
| Current (after shared memory fix) | gfx1201 | 82 |

> 7B decode dropped from 107 to 82 tok/s. The v2 attention kernel shared memory
> bug was hiding a performance issue — the fix made output correct but exposed
> that the v2 kernel with proper s_reduce placement is slower. Investigation needed.

## Key Insights

1. **Full-decode HIP graph** was the biggest 0.5B win (508 → 646 with fusions). Losing it due to the RDNA4 device-pointer bug dropped us to 222.

2. **Kernel fusion** reduced graph nodes by 43% (219 → 124) and hipGraphLaunch time by 51%. The GPU kernel compute itself barely changed — the win was host-side launch overhead.

3. **7B is memory-bandwidth-bound** at ~614 GB/s measured effective (640 GB/s RX 9070 XT spec). Kernel fusion, ALU optimizations (udot4), and graph capture all showed ~0% improvement. The only path to faster 7B decode is algorithmic: speculative decoding with batched verify.

4. **Speculative decoding** needs batched verify to be useful. Token-by-token verify makes it 2-4x slower than baseline regardless of acceptance rate.

## Test Setup

- **GPU:** AMD Radeon RX 9070 XT (gfx1201, RDNA4, 16 GB)
- **gfx1100 reference:** RX 7900 XT (historical)
- **ROCm:** 7.2.1
- **llama.cpp:** commit 408225b, built with `-DGGML_HIP=ON -DAMDGPU_TARGETS=gfx1201`
- **Models:** Qwen2.5-0.5B-Instruct Q4_0, Qwen2.5-7B-Instruct Q4_0
- **Benchmark:** CLI with specific prompts, llama-bench with `-r 3`
