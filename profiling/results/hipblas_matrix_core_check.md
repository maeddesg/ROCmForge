# hipBLAS Matrix-Core Check — does hipblasHgemm use WMMA on gfx1201?

**Date:** 2026-04-17
**Commit:** `0e3d290` (hipBLAS prefill path live by default)
**GPU:** RX 9070 XT (gfx1201, RDNA 4), ROCm 7.2.1
**Model:** Qwen2.5-7B Q4_0, prompt length 255 tokens, single greedy forward

## TL;DR

**Verdict: VALU fallback confirmed.** hipBLAS on ROCm 7.2 dispatches a Tensile `FMA`-marked kernel for gfx1201 — no WMMA / MFMA / xdlops paths are used. The largest projection (Gate/Up/Down) takes **up to 3,844 µs per call** (avg 3,643 µs for the largest shape), versus the < 100 µs we would see on matrix cores. The 10× gap to llama.cpp's prefill throughput is therefore still on the table, and **Phase 2 (custom WMMA kernel with inline Q4_0 dequant) is the right next step.**

## Method

```fish
rocprofv3 --kernel-trace -d /tmp/rocprof_out -- \
    ./target/release/rocmforge \
        --model ~/models/Qwen2.5-7B-Instruct-Q4_0.gguf \
        --prompt (python3 -c 'print(" ".join(["word"] * 255))') \
        --max-tokens 1 --gpu --no-template --temperature 0.0 --top-p 1.0
```

Raw sqlite database archived at `profiling/results/hipblas_matrix_core_check/rocprofv3_kernels.db`. Top-20 kernels by total GPU time in `top_kernels.txt`.

## Kernel name inspection

The hipBLAS GEMM path produced a single kernel name repeated 168 times (= 6 projections × 28 layers of Qwen2.5-7B):

```
Cijk_Alik_Bljk_HB_MT64x64x8_SN_...FMA_...TT4_4_...WS64_WG16_16_1_WGM8
```

Tensile mangled-name legend (the parts that matter):

| Token          | Meaning |
|----------------|---------|
| `HB`           | A/B = half (FP16), Beta = half. FP16 × FP16 → FP16 datatypes. |
| `MT64x64x8`    | Macrotile: 64 × 64 output tile, 8-wide K step. |
| **`FMA`**      | **Uses vector `vfmac/vfma` instructions — the VALU FP16 fused multiply-add path. Matrix-core kernels carry `MFMA` / `WMMA` here.** |
| `TT4_4`        | Thread tile 4 × 4. |
| `WG16_16_1`    | 16 × 16 × 1 threads per workgroup = 256 threads. |
| `WS64`         | Wavesize 64 (unusual for RDNA wave32, but forced for this kernel). |

Cross-check in the full trace: the number of distinct kernels is **26**. A `SELECT DISTINCT name FROM kernels WHERE name LIKE '%wmma%' OR name LIKE '%mfma%' OR name LIKE '%xdlops%'` (case-insensitive) returns **zero rows**. No matrix-core kernel is launched anywhere in the prefill.

## Timing plausibility check

Top GPU-time kernels during prefill:

| Kernel                              |  n  | avg µs | min µs | max µs  |
|-------------------------------------|-----|-------:|-------:|--------:|
| `flash_attn_prefill_strided_kernel` | 784 |  2,974 |  2,807 |   3,443 |
| `gemm_q4_0_f32_generic_kernel`      |  25 | 10,676 | 10,541 |  10,911 |
| `Cijk_Alik_Bljk_HB_…FMA…` (hipBLAS) | 168 |  1,492 |    122 |   3,844 |

The `gemm_q4_0_f32_generic_kernel` still fires 25 times (= prefill chunking for non-projection GEMMs that remain on the custom path — LM head and anything with `needs_transpose`). Not part of the hipBLAS question.

Breaking the 168 hipBLAS calls by grid geometry (three shape classes, 56 calls each):

| grid_x | grid_y | n  | avg µs | min µs | max µs |
|-------:|-------:|---:|-------:|-------:|-------:|
|  2,048 |      4 | 56 |    126 |    122 |    149 |
| 14,336 |      4 | 56 |    708 |    672 |    838 |
| 75,776 |      4 | 56 |  3,643 |  3,466 |  3,844 |

Interpretation: the three groups correspond to small / medium / large projections in Qwen2.5-7B.

Expected times against the prompt's plausibility thresholds (largest projection = Gate/Up/Down = 3,584 × 18,944 FP16):

| Scenario                    | Expected duration | Observed max |
|-----------------------------|-------------------|-------------:|
| Matrix-core (WMMA/MFMA)     | < 100 µs          | 3,844 µs     |
| VALU FP16 fallback          | > 500 µs          | 3,844 µs     |

The max observed on the largest shape is **7.7× slower** than even the 500 µs VALU threshold — which itself is already 5–6× slower than a matrix-core kernel would be. The smallest shape (126 µs average) is close to what a matrix-core kernel of that size would take, but still slightly above the < 100 µs threshold for a well-tuned matrix-core kernel.

Both signals — kernel name (`FMA`, no `MFMA`/`WMMA`) and timing (up to 7.7× above the VALU threshold) — point at the same conclusion.

## Verdict

**hipBLAS on ROCm 7.2 does NOT use matrix-core kernels for FP16 GEMM on gfx1201.** The Tensile backend selected a hand-written VALU FP16 `FMA` kernel rather than a WMMA-based one. As a result, 86 tok/s is the throughput ceiling of this particular kernel, not a hardware ceiling.

Custom WMMA Phase 2 is justified — **the 10× potential is still on the table.** A hand-written WMMA kernel that consumes Q4_0 weights directly (inline dequant into shared memory, then WMMA accumulation in FP16 / FP32) should close most of the remaining gap to llama.cpp's ~1,092 tok/s on the same hardware, since llama.cpp ROCm does this exact pattern.

## Consequences for the next chapter

1. **Do Phase 2** — custom WMMA GEMM kernel for the prefill path with inline Q4_0 dequant (skip the FP16 weight materialisation, skip the FP32→FP16 input conversion where possible, accumulate in FP32 with FP16 operands fed to the WMMA intrinsics).
2. **Keep the hipBLAS path** as a fallback and for non-Q4_0 types. The flag `ROCMFORGE_DISABLE_HIPBLAS_PREFILL=1` is still useful during bring-up of the WMMA kernel so we can A/B against a known-good hipBLAS reference.
3. **Attention is next after GEMM.** `flash_attn_prefill_strided_kernel` takes 784 × 2,974 µs ≈ 2,330 ms for the 255-token prefill — a very large chunk of wall-clock. Once GEMM is on matrix cores, attention becomes the dominant cost and needs its own tiling pass.
4. **Open question on the three shape classes.** The `2048×4` group runs at 126 µs average, close to the matrix-core ceiling — the smaller projections may already be partially amortised by memory bandwidth. Worth re-checking after Phase 2 whether they benefit equally, or whether the gain is concentrated on the large (Gate/Up/Down) shapes.
