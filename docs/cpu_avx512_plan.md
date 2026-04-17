# CPU AVX-512 Q4_0 GEMV — Phase 0 plan

**Starting SHA:** `b3818d9`
**CPU:** AMD Ryzen 9 7945HX (Zen4, 16C/32T, AVX-512 VNNI confirmed via `/proc/cpuinfo`)
**Date:** 2026-04-17

## Baseline (before optimization)

Qwen2.5-0.5B Q4_0, prompt "Hello", `--temperature 0.0 --top-p 1.0 --no-template`, `--max-tokens 64`:

- **Decode throughput: 12.6 tok/s**
- Prefill: 6.1 tok/s (irrelevant for this task, but documented)
- Startup reports "Kernel preference: AVX-512 VNNI" (feature detection works) — but the actual GEMV path does not use this detection.

Target threshold: **≥ 40 tok/s** → 3.2× speedup required.

## Where the problem sits

**File:** `src/cpu/ops.rs`, lines 1273–1274:

```rust
let features = super::features::CpuFeatures::get();
let use_avx2 = features.has_avx2;
```

`gemv_q4_0_q8_0` dispatches on AVX2 or scalar only. `has_avx512` and `has_avx512_vnni` are detected correctly (`src/cpu/features.rs:163-166`) and `KernelPreference::Avx512Vnni` is selected — but the Q4_0 GEMV does not use that information.

### Existing patterns to build on

- **Rayon multi-threading:** `y.par_iter_mut().enumerate().for_each(...)` (line 1276) — parallel over output rows. No new dependency needed.
- **Q8_0 input quantization:** `quantize_q8_0_single` (line 1271) — already performs the FP32 → Q8_0 step. The AVX-512 variant can sit directly on top of it.
- **Block loop with 2× unroll + prefetch:** lines 1282–1328. Kept / restructured.
- **Scalar fallback:** `dot_q4_0_q8_0_block_scalar` (line 2082) — kept as a last resort.
- **AVX2 kernel with/without VNNI:** `dot_q4_0_q8_0_block_avx2` (line 1946) → internally dispatches to `mul_sum_q4_0_q8_0_block_avx2_vnni` (AVX2-VNNI, Intel-only) or `_unscaled`. Kept as a fallback for non-AVX-512 CPUs.

### Q4_0 block layout (confirmed, no restructuring needed)

A Q4_0 block is 18 bytes:
- 2 bytes FP16 `scale`
- 16 bytes: 32 × 4-bit values (two per byte, `lo = byte & 0x0F`, `hi = byte >> 4`)
- Dequant: `value = (nibble − 8) × scale`

The weight layout is already SIMD-friendly (contiguous per row, 16-byte payload per block — fits into a 128-bit register or a half-lane of a 256/512-bit register).

## Plan for phase 1

1. **New kernel** `dot_q4_0_q8_0_block_avx512_vnni`:
   - Input: `qs: &[u8; 16]`, `q8: &[u8; 32]`, `combined_scale: f32`
   - Nibble unpacking: `_mm_loadu_si128` (16 bytes) → split into lo/hi nibbles → concatenate into `__m256i` (32 signed INT8 with −8 bias).
   - Q8 input: `_mm256_loadu_si256` (32 bytes).
   - Dot product: `_mm256_dpbssd_epi32` (AVX-512 VNNI, signed/unsigned — on AMD the form available is `vpdpbusd` from Zen4 onward). If the signed-signed variant is not available, use a two-step form with `_mm256_maddubs_epi16`.
   - Horizontal sum + scale multiplication.
2. **Two-block variant** `dot_q4_0_q8_0_2blocks_avx512_vnni`:
   - Loads two adjacent blocks simultaneously into a 512-bit register, processes both with a single `_mm512_dpbusd_epi32`. This is the main gain over AVX2 — double the data width for the same instruction count.
3. **Dispatch in `gemv_q4_0_q8_0`:**
   ```rust
   let use_avx512_vnni = !env_flag_disable_avx512() && features.has_avx512 && features.has_avx512_vnni;
   ```
   Opt-out via `ROCMFORGE_DISABLE_AVX512=1`.
4. **Multi-threading:** Rayon is already in use. No change needed. On 16 cores at 896 outputs (0.5B FFN-down) → 56 rows/thread. Comfortably fits in L2.
5. **`has_avx512_vnni` on `CpuFeatures`:** currently missing as a public field — computed internally in `detect_x86_64` but not exposed. Must be added to the struct so the dispatch path can read it.

## Multi-threading status

Rayon is already active (`par_iter_mut` in `gemv_q4_0_q8_0`). No new dependency needed. The existing thread pool is reused automatically.

## Cache budget

- 0.5B, hidden=896: `896/32 × 18 = 504 B` per row. 56 rows/thread (at 16 cores, 896 outputs) = 28 KB. Fits in L1 (32 KB/core).
- 0.5B, FFN-up: 4864 outputs, 896 inputs: 304 rows/thread = 153 KB. Fits in L2 (1 MB/core).
- 7B, FFN-down: 3584 outputs, 18944 inputs: `18944/32 × 18 = 10.6 KB` per row. 224 rows/thread = 2.4 MB. Goes BEYOND L2 (1 MB/core) → cache tiling would help on 7B. Irrelevant for 0.5B, but worth keeping in mind for phase 2.

## Validation

`tests/cpu_avx512_matches_reference.rs`:
- Run A: `ROCMFORGE_DISABLE_AVX512=1` (old path, AVX2).
- Run B: AVX-512 active (default).
- Assertion: byte-identical token IDs for greedy decoding + a fixed prompt, 50 tokens.

## No open questions

All strategic questions from the prompt are answered by reading the code:
- CPU GEMV path: located (`src/cpu/ops.rs:1244`).
- Q4_0 layout: SIMD-friendly, no restructuring.
- Rayon: already in use.
- CPU feature detection: complete, only needs `has_avx512_vnni` exposed.
- Baseline: 12.6 tok/s (3.2× needed for ≥40 tok/s).

Implementation starts in phase 1.
