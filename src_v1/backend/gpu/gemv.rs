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

#[link(name = "v1_gemv_q4_k_standard", kind = "static")]
extern "C" {
    /// Launch the Q4_K standard GEMV. `n_rows` must be a multiple of 256
    /// (Q4_K super-block size). Caps input staging at 32 KiB LDS (K ≤ 8192).
    pub fn rocmforge_launch_gemv_q4_k_standard(
        weights: *const u8,
        input: *const f32,
        output: *mut f32,
        n_rows: i32,
        ncols_dst: i32,
        stream: hipStream_t,
    ) -> hipError_t;
}

#[link(name = "v1_gemv_q6_k_standard", kind = "static")]
extern "C" {
    /// Launch the Q6_K standard GEMV. `n_rows` must be a multiple of 256.
    pub fn rocmforge_launch_gemv_q6_k_standard(
        weights: *const u8,
        input: *const f32,
        output: *mut f32,
        n_rows: i32,
        ncols_dst: i32,
        stream: hipStream_t,
    ) -> hipError_t;
}

#[link(name = "v1_gemv_q8_0_standard", kind = "static")]
extern "C" {
    /// Launch the Q8_0 standard GEMV. `n_rows` must be a multiple of 32.
    pub fn rocmforge_launch_gemv_q8_0_standard(
        weights: *const u8,
        input: *const f32,
        output: *mut f32,
        n_rows: i32,
        ncols_dst: i32,
        stream: hipStream_t,
    ) -> hipError_t;
}

#[link(name = "v1_gemv_q4_k_q8_inline", kind = "static")]
extern "C" {
    /// Launch the Q4_K Q8-inline GEMV. `n_rows` must be a multiple of 256.
    /// Activation is quantised into LDS cooperatively; dual-accumulator
    /// sum (int_dot + q8_sum) per sub-block handles the Q4_K dmin term.
    pub fn rocmforge_launch_gemv_q4_k_q8_inline(
        weights: *const u8,
        input: *const f32,
        output: *mut f32,
        n_rows: i32,
        ncols_dst: i32,
        stream: hipStream_t,
    ) -> hipError_t;
}

#[link(name = "v1_gemv_q4_k_q8_inline_residual", kind = "static")]
extern "C" {
    /// Same as [`rocmforge_launch_gemv_q4_k_q8_inline`] but writes
    /// `output[i] = GEMV(W,X)[i] + residual[i]` in one pass — saves
    /// one kernel launch and one round-trip through the output
    /// buffer per fusion site. `residual` must be `ncols_dst` FP32
    /// values on device. Matches v0.x's
    /// `gemv_q4_k_f32_q8_inline_residual_multi_row_kernel`.
    pub fn rocmforge_launch_gemv_q4_k_q8_inline_residual(
        weights: *const u8,
        input: *const f32,
        residual: *const f32,
        output: *mut f32,
        n_rows: i32,
        ncols_dst: i32,
        stream: hipStream_t,
    ) -> hipError_t;
}

#[link(name = "v1_gemv_q4_k_gate_up_swiglu", kind = "static")]
extern "C" {
    /// Launch the fused Q4_K Gate+Up+SwiGLU GEMV. Computes
    /// `swiglu[n] = silu(gate_gemv(n)) * up_gemv(n)` in one pass; gate
    /// and up weights share an activation-cache load. `n_rows` must be
    /// a multiple of 256; input × 4 B must fit in 32 KiB LDS.
    pub fn rocmforge_launch_gemv_q4_k_gate_up_swiglu(
        weights_gate: *const u8,
        weights_up: *const u8,
        input: *const f32,
        swiglu_out: *mut f32,
        n_rows: i32,
        ncols_dst: i32,
        stream: hipStream_t,
    ) -> hipError_t;
}
