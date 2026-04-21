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
    /// Launch the Q4_0 FP16 WMMA GEMM.
    ///
    /// Computes `D[M × N] = A[M × K] * W[N × K]` where `A` is FP32
    /// row-major, `W` is Q4_0 row-major (18 B per block along K), and
    /// `D` is FP32 row-major. `M`, `N`, `K` must be multiples of
    /// `TILE_M=64` / `TILE_N=64` / `K_CHUNK=32` respectively; otherwise
    /// the launcher returns `hipErrorInvalidValue` without dispatching.
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
