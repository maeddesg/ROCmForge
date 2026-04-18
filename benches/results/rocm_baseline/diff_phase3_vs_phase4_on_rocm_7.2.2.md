# ROCm Upgrade Diff: 7.2.2 → 7.2.2

- **Date:** 2026-04-18T15:49:46+02:00
- **Baseline:** \`benches/results/rocm_baseline/rocm_7.2.2_1776508380\`
- **New:**      \`benches/results/rocm_baseline/rocm_7.2.2_1776519824\`

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
  chat_01:  ⚠️ diverges at word 40 (80 vs 80 words)
  chat_02:  ✅ identical (99 words)
  chat_03:  ⚠️ diverges at word 67 (91 vs 90 words)
  chat_04:  ✅ identical (98 words)
  chat_05:  ⚠️ diverges at word 57 (95 vs 98 words)
  code_01:  ⚠️ diverges at word 23 (79 vs 82 words)
  code_02:  ✅ identical (86 words)
  code_03:  ⚠️ diverges at word 46 (101 vs 97 words)
  code_04:  ✅ identical (70 words)
  code_05:  ✅ identical (86 words)
  prose_01:  ⚠️ diverges at word 12 (102 vs 104 words)
  prose_02:  ⚠️ diverges at word 21 (77 vs 77 words)
  prose_03:  ✅ identical (93 words)
  prose_04:  ⚠️ diverges at word 70 (93 vs 94 words)
  prose_05:  ⚠️ diverges at word 12 (108 vs 105 words)

  Verdict: 6/15 identical, 9 drifted
```

## [PERFORMANCE]

```
  Synthetic Benchmark:
    Prefill pp64:          470.0 → 800.7 tok/s (+70.36%, ❌)
    Prefill pp128:         554.4 → 1131.6 tok/s (+104.11%, ❌)
    Prefill pp256:         593.4 → 1482.6 tok/s (+149.85%, ❌)
    Prefill pp512:         611.0 → 1693.0 tok/s (+177.09%, ❌)
    Decode 128tok:         102.4 → 102.4 tok/s (+0%, ✅)

  TTFT Stability (trace):
    Layer-0 Warmup:        3184 → 2954 µs (-7.22%, ⚠️)
    Layer 1..N Median:     38 → 37 µs (-2.63%, ✅)
    Ratio L0/L1:           83.79 → 79.84 x (-4.71%, ✅)

  15-Prompt (Median):
    TTFT:                  76.8 → 49.1 ms (-36.07%, ❌)
    Decode:                101.9 → 101.9 tok/s (+0%, ✅)
```

## [VERDICT]

```
  Build:         ✅ Both versions compile successfully
  Correctness:   ❌ 6/15 identical, 9 drifted
  Performance:   ❌ 5 metrics outside ±10% tolerance

  ➜ Investigate before using ROCm 7.2.2 — see details above.
```

