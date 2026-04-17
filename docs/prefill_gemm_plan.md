# Prefill GEMM — Phase 0 plan

**Starting SHA:** `a8c694e`
**GPU:** RX 9070 XT (gfx1201, RDNA 4), 16 GB VRAM (17,095,983,104 B reported), ~640 GB/s
**Target model:** Qwen2.5-7B Q4_0 (hidden=3584, intermediate=18944, 28 layers)
**Date:** 2026-04-17

## Availability check

- hipBLAS shared library: `/opt/rocm/lib/libhipblas.so` + `libhipblas.so.3` present.
- Header: `/opt/rocm/include/hipblas/hipblas.h` present.
- `libhipblaslt.so` also available (may be relevant later for INT4/INT8 GEMM).
- Current free VRAM with no model loaded: ~16 GB.

No installation needed.

## Where the current prefill lives

Entry: `src/main.rs:626` → `gpu::gpu_prefill_forward_hybrid` (`src/gpu/forward.rs:931`).

Projection dispatch: `gpu_project_rows` in `src/gpu/forward.rs:620` → `gpu_dispatch_gemm` in `src/gpu/ops.rs:1431`.

`gpu_dispatch_gemm` behaviour for `seq_len > 1`:

- `seq_len == 1` → `gpu_dispatch_gemv` (single-token path, unchanged).
- `seq_len ≤ 8` (small batch verify) → `gemv_q4_0_f32_batched_on_stream` or tiled variant.
- `seq_len > 8` → `gemm_q4_0_f32` — the current custom prefill path. This is what we are replacing. Still a stack of scalar FMAs, no matrix-core use.

Input dtype on the prefill path: **FP32** (see `*const f32 input` / `*mut f32 output` in `gpu_project_rows`). We will need FP32 → FP16 conversion on the fly for hipBLAS.

Prefill timing: `src/main.rs:361` already emits `"Prefill: {:.1}ms ({:.1} tok/s)"`. No dedicated prefill-only benchmark mode, but the existing print is enough for the sweep.

## VRAM budget

Dequantised FP16 weights for Qwen2.5-7B:

| Projection        | Shape           | FP16 bytes |
|-------------------|-----------------|-----------:|
| QKV (Q, K, V fused) | 3584 × (3584 + 2×512) | ~30 MB |
| O                 | 3584 × 3584     |     ~26 MB |
| Gate              | 3584 × 18944    |    ~136 MB |
| Up                | 3584 × 18944    |    ~136 MB |
| Down              | 18944 × 3584    |    ~136 MB |
| Per-layer total   |                 |    ~464 MB |
| All 28 layers     |                 |    ~13.0 GB |

Full-weight-tensor FP16 does not fit alongside the Q4_0 model (~3.5 GB) + KV cache in 16 GB.

**Chosen strategy: per-projection dequant (Option B).** Allocate one FP16 scratch buffer sized for the largest single projection (Gate/Up/Down = ~136 MB). Per-projection: run the dequant kernel, run hipBLAS GEMM, reuse the scratch for the next projection.

Adds one global FP16 scratch buffer (~136 MB) to `GpuPrefillScratch`. Fits comfortably; measured free VRAM with no model is ~15.9 GB.

Dispatch count during prefill: 5 projections × 28 layers = 140 dequant+GEMM pairs. Dequant is a simple elementwise kernel (one FMA per element, 64 bytes written per Q4_0 block) — we will measure its cost in Phase 2.

## hipBLAS link strategy

- `build.rs` adds `println!("cargo:rustc-link-lib=hipblas");` alongside the existing `amdhip64` / `hiprtc` links.
- The link search path (`/opt/rocm/lib`) is already set by the HIP block.
- No Cargo feature gate for now: the `gpu` feature already implies ROCm is installed, and hipBLAS is part of the standard ROCm package.

## Dequant kernel strategy

New kernel: `hip_kernels/quant/dequant_q4_0_to_f16.hip`.

- One thread per Q4_0 block (32 output values). Grid sized so that `grid × block = total_blocks`.
- Loads 18 bytes (2 B scale + 16 B quants), unpacks 32 nibbles, writes 32 × FP16.
- Output layout: column-major for hipBLAS (we will set `lda`/`ldb` accordingly rather than transposing).
- Pattern follows the existing `dequantize_q4_0_device` in `hip_kernels/quant/q4_0_dequantize.hip` but writes `half` instead of `float`.

## Dispatch logic

In `gpu_dispatch_gemm`:

```rust
const PREFILL_GEMM_THRESHOLD: usize = 8;  // tune; GEMV path currently wins up to 8

if seq_len >= PREFILL_GEMM_THRESHOLD
    && meta.wtype == GgmlType::Q4_0
    && safety::hipblas_prefill_enabled()
{
    // 1. FP32 → FP16 conversion of input: seq_len × in_dim
    // 2. Q4_0 → FP16 dequant of weight matrix into shared scratch
    // 3. hipblasHgemm: C[seq_len × out_dim] = A[seq_len × in_dim] × B[in_dim × out_dim]
    // 4. FP16 → FP32 of output: seq_len × out_dim
    return dispatch_prefill_via_hipblas(...);
}
// Fallback to the existing GEMV / custom GEMM path.
```

The FP16 conversions for input/output are small elementwise kernels. For the initial measurement we can even use `hipblasGemmEx` with mixed precision (A/B FP16, C FP32) to skip the output conversion — will decide after seeing the first numbers.

Opt-out: `ROCMFORGE_DISABLE_HIPBLAS_PREFILL=1` → fall through to the existing path. Default on when hipBLAS is linked.

## Matrix-layout note

hipBLAS is column-major (BLAS convention). Our tensors are row-major. Two ways to handle this:

- Transpose via `hipblasOperation_t`: pass `HIPBLAS_OP_T` for one side and adjust the effective shape.
- Compute `C^T = B^T × A^T` and swap the meanings of M/N.

Either works — the second approach is usually cleaner and avoids an extra transpose. Concretely for `Y = X × W` where `X` is `[seq_len × in_dim]` row-major and `W` is `[in_dim × out_dim]` row-major, we call `hipblasHgemm` with:

- transa = `N`, transb = `N`
- m = out_dim, n = seq_len, k = in_dim
- A = W (treated as out_dim × in_dim col-major = in_dim × out_dim row-major, lda = in_dim)
- B = X (treated as in_dim × seq_len col-major = seq_len × in_dim row-major, ldb = in_dim)
- C = Y (treated as out_dim × seq_len col-major = seq_len × out_dim row-major, ldc = out_dim)

This produces `C` in column-major = `Y` in row-major. No explicit transpose kernels.

Matrix-layout is the most common hipBLAS bug — the first run will include a correctness check against the GEMV path.

## Correctness validation

`tests/prefill_hipblas_matches_gemv.rs`: fixed 64-token prompt, greedy, 10 decoded tokens after prefill. Run with `ROCMFORGE_DISABLE_HIPBLAS_PREFILL=1` (baseline) vs. default. Token IDs must match.

Tolerance: FP16 accumulation can produce single-token divergence on low-confidence positions. Test will first check for exact match; if that fails at position `i`, dump the top-5 logits at that position and fail only if the top-1 delta is > 0.05 (clearly numerical vs. logic bug).

## Out of scope for Phase 1

- Custom WMMA kernel with inline dequant (that is Phase 2).
- INT4/INT8 GEMM via hipBLASLt (possibly Phase 2 or later).
- Any changes to the decode path.
- Q4_1 / Q8_0 / Q4_K prefill via hipBLAS (Phase 1 focuses on Q4_0, the format we actually benchmark against).

## No blocker

Plan is fully determined by code reading:
- hipBLAS: available.
- Prefill path: located.
- VRAM budget: fits with Option B.
- Input dtype: FP32, conversion needed — known and planned for.
- Layout: known and planned for (compute C^T trick).

Implementation starts in Phase 1.
