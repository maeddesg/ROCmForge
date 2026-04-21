//! FFI bindings for the Phase-1 GEMV decode kernels.
//!
//! Kernels live under `hip_kernels_v1/gemv/` and are produced by
//! `rocmforge::v1::ir::codegen_gpu::emit_gemv_kernel()`. Block A only
//! ships the Q4_0 standard launcher; Q4_K / Q6_K / Q8_0 standard and
//! all Q8-inline / fused variants land in Blocks B and C.

use super::hip_ffi::{hipError_t, hipStream_t};

#[link(name = "v1_gemv_q4_0_standard", kind = "static")]
extern "C" {
    /// Launch the Q4_0 standard GEMV.
    ///
    /// Computes `output[ncols_dst] = W[ncols_dst × n_rows] * input[n_rows]`
    /// where `W` is a row-major Q4_0 weight tensor (N rows, each with
    /// `n_rows / 32` 18-byte Q4_0 blocks along K).
    ///
    /// Constraints:
    ///   * `n_rows` must be a multiple of 32 (Q4_0 block size).
    ///   * `n_rows * 4` bytes must fit in 32 KiB shared memory — this
    ///     matches `K ≤ 8192`. Anything larger returns
    ///     `hipErrorInvalidValue`; the chunked fallback is Phase-2.
    pub fn rocmforge_launch_gemv_q4_0_standard(
        weights: *const u8,
        input: *const f32,
        output: *mut f32,
        n_rows: i32,
        ncols_dst: i32,
        stream: hipStream_t,
    ) -> hipError_t;
}
