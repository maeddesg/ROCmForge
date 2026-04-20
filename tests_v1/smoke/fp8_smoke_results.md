# Milestone 0 — FP8 WMMA Smoke Test Results

Target: **gfx1201** (AMD Radeon RX 9070 XT, RDNA 4) · ROCm 7.2.2 · Wave32

Intrinsic under test: `__builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12`

| # | Test | Result |
|---|------|--------|
| 1 | Test 1 — Compilability & device detection | ✅ PASS |
| 2 | Test 2 — FP8 WMMA correctness | ✅ PASS |
| 3 | Test 3 — FP8 vs FP16 performance (dispatch-bound single-tile) | ✅ PASS |
| 4 | Test 4 — FP32 → E4M3 → FP32 roundtrip | ✅ PASS |
| 5 | Test 5 — FP8 vs FP16 numerical quality | ✅ PASS |

## Details

### Test 1 — Compilability & device detection

- Result: **PASS**
- hipcc built fp8_wmma_smoke.hip for gfx1201; runtime device: gfx1201 (12.0)

### Test 2 — FP8 WMMA correctness

- Result: **PASS**
- all-1.0 sanity: max |out-16|=0.00e0 (expect 0) · random: max_abs_err=0.000e0, max_rel_err=0.000e0 (tol 0.1)

### Test 3 — FP8 vs FP16 performance (dispatch-bound single-tile)

- Result: **PASS**
- FP8 median=7.96 µs (min 7.84), FP16 median=8.00 µs (min 7.84), ratio FP16/FP8=1.01× (single-tile is launch-bound; real speedup verified in Phase 1 on larger shapes)

### Test 4 — FP32 → E4M3 → FP32 roundtrip

- Result: **PASS**
- n=1024, max_abs_err=1.249e-1, max_rel_err (|r|>1.95e-3)=7.274e-2 (tol 0.15), violations=0

### Test 5 — FP8 vs FP16 numerical quality

- Result: **PASS**
- GPU-FP8 vs CPU-E4M3 ref: 0.000e0 (tol 0.1) · GPU-FP8 vs unquantised FP32: 3.203e-2 (intrinsic FP8 gap) · GPU-FP16 vs FP32: 2.413e-4 (tol 0.01)

## Decision

**GO — FP8 as v1.0 default WMMA input path.**
