//! FFI bindings for the Phase-1 WMMA GEMM kernels.
//!
//! Kernels live in `hip_kernels_v1/wmma/` and are produced by
//! `rocmforge::v1::ir::codegen_gpu::emit_wmma_gemm_kernel()`. Each
//! kernel is a specialised `(QuantFormat, Precision)` binary; dispatch
//! is a straightforward match on those two parameters.
//!
//! Phase 1 Block A: only `Q4_0 × FP16` is linked. Further formats and
//! the FP8 precision level land in Blocks B and C.

use super::hip_ffi::{hipError_t, hipStream_t};

// ─── MMQ infrastructure (P0.2 Schritt 1) ────────────────────────────────────

/// llama.cpp's `block_q8_1_mmq` DS4 layout — 4 × `half2` scale+sum
/// pairs then 128 signed int8 quantised values. 144 bytes total,
/// aligned to 4 bytes. Must be binary-identical to llama.cpp's type
/// (see `~/tmp/llama.cpp/ggml/src/ggml-cuda/mmq.cuh:28-47`).
#[repr(C)]
#[derive(Copy, Clone)]
pub struct BlockQ81Mmq {
    /// 4 × `half2(d, sum)` stored as u32 so the struct is POD-safe.
    pub ds4: [u32; 4],
    pub qs:  [i8;  128],
}
const _: () = assert!(std::mem::size_of::<BlockQ81Mmq>() == 144);

#[link(name = "v1_quantize_q8_1_mmq", kind = "static")]
extern "C" {
    /// Quantise `n_elements` FP32 into `n_elements/128` `BlockQ81Mmq`
    /// records (DS4 layout). `n_elements` must be a multiple of 128.
    pub fn rocmforge_launch_quantize_q8_1_mmq(
        input: *const f32,
        output: *mut core::ffi::c_void,
        n_elements: core::ffi::c_int,
        stream: hipStream_t,
    ) -> hipError_t;
}

#[link(name = "v1_wmma_i32_smoke", kind = "static")]
extern "C" {
    /// Integer-WMMA smoke test: one wave runs
    /// `__builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12` on a fixed
    /// 16×16×16 int8 × int8 GEMM. Used to verify the intrinsic
    /// compiles and produces the expected output for a known input.
    pub fn rocmforge_launch_wmma_i32_smoke(
        a: *const i8,
        b: *const i8,
        c: *mut core::ffi::c_int,
        stream: hipStream_t,
    ) -> hipError_t;
}

#[link(name = "v1_wmma_q4_0_fp16", kind = "static")]
extern "C" {
    /// Launch the Q4_0 FP16 WMMA GEMM. `M`/`N`/`K` multiples of
    /// 64/64/32; otherwise returns `hipErrorInvalidValue`.
    pub fn rocmforge_launch_wmma_gemm_q4_0_fp16(
        input: *const f32,
        weights: *const u8,
        output: *mut f32,
        M: i32,
        N: i32,
        K: i32,
        stream: hipStream_t,
    ) -> hipError_t;
}

#[link(name = "v1_wmma_q4_k_fp16", kind = "static")]
extern "C" {
    /// Launch the Q4_K FP16 WMMA GEMM. `M`/`N`/`K` multiples of
    /// 64/64/256 (Q4_K elements-per-block = 256).
    pub fn rocmforge_launch_wmma_gemm_q4_k_fp16(
        input: *const f32,
        weights: *const u8,
        output: *mut f32,
        M: i32,
        N: i32,
        K: i32,
        stream: hipStream_t,
    ) -> hipError_t;
}

#[link(name = "v1_wmma_q4_k_fp16_tiled", kind = "static")]
extern "C" {
    /// Template<64,64,32> instantiation — parity reference. Produces
    /// bit-identical output to `rocmforge_launch_wmma_gemm_q4_k_fp16`.
    pub fn rocmforge_launch_wmma_gemm_q4_k_fp16_tiled_64x64x32(
        input: *const f32,
        weights: *const u8,
        output: *mut f32,
        M: i32,
        N: i32,
        K: i32,
        stream: hipStream_t,
    ) -> hipError_t;

    /// Template<128,128,32> instantiation — llama.cpp MMQ RDNA2+
    /// equivalent. Requires M % 128 == 0 and N % 128 == 0.
    pub fn rocmforge_launch_wmma_gemm_q4_k_fp16_tiled_128x128x32(
        input: *const f32,
        weights: *const u8,
        output: *mut f32,
        M: i32,
        N: i32,
        K: i32,
        stream: hipStream_t,
    ) -> hipError_t;
}

#[link(name = "v1_wmma_q6_k_fp16", kind = "static")]
extern "C" {
    /// Launch the Q6_K FP16 WMMA GEMM. `M`/`N`/`K` multiples of
    /// 64/64/256 (Q6_K elements-per-block = 256).
    pub fn rocmforge_launch_wmma_gemm_q6_k_fp16(
        input: *const f32,
        weights: *const u8,
        output: *mut f32,
        M: i32,
        N: i32,
        K: i32,
        stream: hipStream_t,
    ) -> hipError_t;
}

#[link(name = "v1_wmma_q8_0_fp16", kind = "static")]
extern "C" {
    /// Launch the Q8_0 FP16 WMMA GEMM. `M`/`N`/`K` multiples of
    /// 64/64/32 (Q8_0 block holds 32 elements).
    pub fn rocmforge_launch_wmma_gemm_q8_0_fp16(
        input: *const f32,
        weights: *const u8,
        output: *mut f32,
        M: i32,
        N: i32,
        K: i32,
        stream: hipStream_t,
    ) -> hipError_t;
}

#[link(name = "v1_wmma_q4_0_fp8", kind = "static")]
extern "C" {
    /// Q4_0 FP8 (Level 0) WMMA launcher. Same geometry as the FP16
    /// kernel — `M`/`N`/`K` multiples of 64/64/32.
    pub fn rocmforge_launch_wmma_gemm_q4_0_fp8(
        input: *const f32,
        weights: *const u8,
        output: *mut f32,
        M: i32,
        N: i32,
        K: i32,
        stream: hipStream_t,
    ) -> hipError_t;
}

#[link(name = "v1_wmma_q4_k_fp8", kind = "static")]
extern "C" {
    pub fn rocmforge_launch_wmma_gemm_q4_k_fp8(
        input: *const f32,
        weights: *const u8,
        output: *mut f32,
        M: i32,
        N: i32,
        K: i32,
        stream: hipStream_t,
    ) -> hipError_t;
}

#[link(name = "v1_wmma_q6_k_fp8", kind = "static")]
extern "C" {
    pub fn rocmforge_launch_wmma_gemm_q6_k_fp8(
        input: *const f32,
        weights: *const u8,
        output: *mut f32,
        M: i32,
        N: i32,
        K: i32,
        stream: hipStream_t,
    ) -> hipError_t;
}

#[link(name = "v1_wmma_q8_0_fp8", kind = "static")]
extern "C" {
    pub fn rocmforge_launch_wmma_gemm_q8_0_fp8(
        input: *const f32,
        weights: *const u8,
        output: *mut f32,
        M: i32,
        N: i32,
        K: i32,
        stream: hipStream_t,
    ) -> hipError_t;
}
