# ROCm Upgrade Diff: 7.2.1 → 7.2.2

- **Date:** 2026-04-18T12:25:27+02:00
- **Baseline:** \`benches/results/rocm_baseline/rocm_7.2.1_1776506441\`
- **New:**      \`benches/results/rocm_baseline/rocm_7.2.2_1776507261\`

## [BUILD]

```
  ROCm Version:          7.2.1 → 7.2.2   ⚠️ changed
  hipcc:                 7.2.53211-9999 → 7.2.53211-9999   ✅ unchanged
  amdclang:              22.0.0git → 22.0.0git   ✅ unchanged
  hipBLAS pkg:           7.2.1-1.1 → 7.2.2-1.1   ⚠️ changed
  HIP runtime pkg:       7.2.1-2 → 7.2.2-1   ⚠️ changed
  rocm-core pkg:         7.2.1-2.1 → 7.2.2-1.1   ⚠️ changed
  GPU Arch:              gfx1201 → gfx1201   ✅ unchanged
  Driver:                unknown → unknown   ✅ unchanged
  WMMA Intrinsic:        true → true   ✅ unchanged
  Compilation:           ✅ OK → ✅ OK   ✅ unchanged
```

## [CORRECTNESS]

```
  chat_01:  ✅ identical (80 words)
  chat_02:  ✅ identical (99 words)
  chat_03:  ✅ identical (91 words)
  chat_04:  ✅ identical (98 words)
  chat_05:  ✅ identical (95 words)
  code_01:  ✅ identical (79 words)
  code_02:  ✅ identical (86 words)
  code_03:  ✅ identical (101 words)
  code_04:  ✅ identical (70 words)
  code_05:  ✅ identical (86 words)
  prose_01:  ✅ identical (102 words)
  prose_02:  ✅ identical (77 words)
  prose_03:  ✅ identical (93 words)
  prose_04:  ✅ identical (93 words)
  prose_05:  ✅ identical (108 words)

  Verdict: 15/15 identical
```

## [PERFORMANCE]

```
  Synthetic Benchmark:
    Prefill pp64:          472.3 → 475.6 tok/s (+0.7%, ✅)
    Prefill pp128:         551.6 → 556.4 tok/s (+0.87%, ✅)
    Prefill pp256:         593.0 → 594.9 tok/s (+0.32%, ✅)
    Prefill pp512:         610.2 → 611.7 tok/s (+0.25%, ✅)
    Decode 128tok:         102.3 → 102.4 tok/s (+0.1%, ✅)

  TTFT Stability (trace):
    Layer-0 Warmup:        2776 → 2635 µs (-5.08%, ⚠️)
    Layer 1..N Median:     37 → 39 µs (+5.41%, ⚠️)
    Ratio L0/L1:           75.03 → 67.56 x (-9.96%, ⚠️)

  15-Prompt (Median):
    TTFT:                  76.9 → 76.3 ms (-0.78%, ✅)
    Decode:                101.9 → 101.9 tok/s (+0%, ✅)
```

## [VERDICT]

```
  Build:         ✅ Both versions compile successfully
  Correctness:   ✅ 15/15 answers identical
  Performance:   ⚠️ 3 metrics in ±5..10% band (noise)

  ➜ ROCm 7.2.2 usable, but review drift/warnings above.
```

