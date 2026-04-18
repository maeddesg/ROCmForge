# ROCm Upgrade Diff: 7.2.1 → 7.2.2

- **Date:** 2026-04-18T12:39:00+02:00
- **Baseline:** \`benches/results/rocm_baseline/rocm_7.2.1_1776506441\`
- **New:**      \`benches/results/rocm_baseline/rocm_7.2.2_1776508380\`

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
    Prefill pp64:          472.3 → 470.0 tok/s (-0.49%, ✅)
    Prefill pp128:         551.6 → 554.4 tok/s (+0.51%, ✅)
    Prefill pp256:         593.0 → 593.4 tok/s (+0.07%, ✅)
    Prefill pp512:         610.2 → 611.0 tok/s (+0.13%, ✅)
    Decode 128tok:         102.3 → 102.4 tok/s (+0.1%, ✅)

  TTFT Stability (trace):
    Layer-0 Warmup:        2776 → 3184 µs (+14.7%, ❌)
    Layer 1..N Median:     37 → 38 µs (+2.7%, ✅)
    Ratio L0/L1:           75.03 → 83.79 x (+11.68%, ❌)

  15-Prompt (Median):
    TTFT:                  76.9 → 76.8 ms (-0.13%, ✅)
    Decode:                101.9 → 101.9 tok/s (+0%, ✅)
```

## [VERDICT]

```
  Build:         ✅ Both versions compile successfully
  Correctness:   ✅ 15/15 answers identical
  Performance:   ❌ 2 metrics outside ±10% tolerance

  ➜ Investigate before using ROCm 7.2.2 — see details above.
```

