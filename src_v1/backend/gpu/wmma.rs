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
