# ROCm Upgrade Diff: 7.2.2 → 7.2.2

- **Date:** 2026-04-18T17:41:27+02:00
- **Baseline:** \`benches/results/rocm_baseline/rocm_7.2.2_1776519824\`
- **New:**      \`benches/results/rocm_baseline/rocm_7.2.2_1776526523\`

## [BUILD]

```
  ROCm Version:          7.2.2 → 7.2.2   ✅ unchanged
  hipcc:                 7.2.53211-9999 → 7.2.53211-9999   ✅ unchanged
  amdclang:              22.0.0git → 22.0.0git   ✅ unchanged
  hipBLAS pkg:           7.2.2-1.1 → 7.2.2-1.1   ✅ unchanged
  HIP runtime pkg:       7.2.2-1 → 7.2.2-1   ✅ unchanged
  rocm-core pkg:         7.2.2-1.1 → 7.2.2-1.1   ✅ unchanged
  GPU Arch:              gfx1201 → gfx1201   ✅ unchanged
  Driver:                unknown → unknown   ✅ unchanged
  WMMA Intrinsic:        true → true   ✅ unchanged
  Compilation:           ✅ OK → ✅ OK   ✅ unchanged
```

## [CORRECTNESS]

```
  chat_01:  ✅ identical (80 words)
  chat_02:  ✅ identical (99 words)
  chat_03:  ✅ identical (90 words)
  chat_04:  ✅ identical (98 words)
  chat_05:  ✅ identical (98 words)
  code_01:  ✅ identical (82 words)
  code_02:  ✅ identical (86 words)
  code_03:  ✅ identical (97 words)
  code_04:  ✅ identical (70 words)
  code_05:  ✅ identical (86 words)
  prose_01:  ✅ identical (104 words)
  prose_02:  ✅ identical (77 words)
  prose_03:  ✅ identical (93 words)
  prose_04:  ✅ identical (94 words)
  prose_05:  ✅ identical (105 words)

  Verdict: 15/15 identical
```

## [PERFORMANCE]

```
  Synthetic Benchmark:
    Prefill pp64:          800.7 → 797.8 tok/s (-0.36%, ✅)
    Prefill pp128:         1131.6 → 1131.2 tok/s (-0.04%, ✅)
    Prefill pp256:         1482.6 → 1481.1 tok/s (-0.1%, ✅)
    Prefill pp512:         1693.0 → 1691.5 tok/s (-0.09%, ✅)
    Decode 128tok:         102.4 → 102.5 tok/s (+0.1%, ✅)

  TTFT Stability (trace):
    Layer-0 Warmup:        2954 → 3005 µs (+1.73%, ✅)
    Layer 1..N Median:     37 → 38 µs (+2.7%, ✅)
    Ratio L0/L1:           79.84 → 79.08 x (-0.95%, ✅)

  15-Prompt (Median):
    TTFT:                  49.1 → 49.2 ms (+0.2%, ✅)
    Decode:                101.9 → 101.8 tok/s (-0.1%, ✅)
```

## [VERDICT]

```
  Build:         ✅ Both versions compile successfully
  Correctness:   ✅ 15/15 answers identical
  Performance:   ✅ All metrics within ±5% tolerance

  ➜ ROCm 7.2.2 is safe to use. No regressions detected.
```

