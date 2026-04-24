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

#[link(name = "v1_gemv_q4_k_q8_inline_sudot4", kind = "static")]
extern "C" {
    /// Q4_K Q8-inline GEMV using the `v_dot4_i32_iu8`
    /// (`__builtin_amdgcn_sudot4`) intrinsic for the inner
    /// 4-lane integer dot product. Same shape constraints as
    /// `rocmforge_launch_gemv_q4_k_q8_inline`; the only differences
    /// sit in the sub-block MAC-loop which is ~8× fewer VALU
    /// instructions per element.
    pub fn rocmforge_launch_gemv_q4_k_q8_inline_sudot4(
        weights: *const u8,
        input: *const f32,
        output: *mut f32,
        n_rows: i32,
        ncols_dst: i32,
        stream: hipStream_t,
    ) -> hipError_t;
}

#[link(name = "v1_gemv_q6_k_mmvq", kind = "static")]
extern "C" {
    /// llama.cpp-style MMVQ port for Q6_K × Q8_1 GEMV (Phase 2 Schritt 4).
    /// Specialised for `ncols_dst=1` (decode), RDNA4 (nwarps=8), VDR=1.
    /// Same 16-thread-per-super-block cooperative tiling family as the
    /// Q4_K MMVQ kernel; differs in the ql/qh 6-bit reconstruction and
    /// the `-32` offset handled via `__vsubss4`.
    ///
    /// Caller must pre-quantize the activation vector via
    /// `rocmforge_launch_quantize_q8_1`. `ncols` must be a multiple of 256.
    pub fn rocmforge_launch_gemv_q6_k_mmvq(
        weights: *const std::ffi::c_void,
        q8_1_input: *const std::ffi::c_void,
        output: *mut std::ffi::c_void,
        nrows: i32,
        ncols: i32,
        stream: hipStream_t,
    ) -> hipError_t;
}

#[link(name = "v1_gemv_q6_k_q8_inline", kind = "static")]
extern "C" {
    /// Phase-2 Schritt 2.1.4 — Q6_K GEMV with Q8-inline activation.
    /// Per-16-element sub-blocks (sub-scale indexed), integer dot
    /// products with a per-super-block `d * scale_sub * (q6 - 32)`
    /// reconstruction. Activation is cooperatively quantised into
    /// LDS per call just like the Q4_K variant. `n_rows` must be a
    /// multiple of 256.
    pub fn rocmforge_launch_gemv_q6_k_q8_inline(
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

#[link(name = "v1_gemv_q4_k_mmvq", kind = "static")]
extern "C" {
    /// llama.cpp-style `mul_mat_vec_q` port for Q4_K × Q8_1 GEMV
    /// (Phase 2 Schritt 2/3). Specialised for `ncols_dst=1` (decode),
    /// `has_fusion=false`, nwarps=8, VDR=2. 16 threads cooperate on
    /// each super-block via int32-aligned loads → this is the path
    /// where the cooperative tiling (hypothesis H1 in the kernel-
    /// analysis report) is expected to lift BW from 51 % → ~70 %.
    ///
    /// Caller must pre-quantize the activation vector to Q8_1 via
    /// `rocmforge_launch_quantize_q8_1` before invoking this kernel.
    /// `ncols` must be a multiple of 256 (Q4_K super-block size).
    pub fn rocmforge_launch_gemv_q4_k_mmvq(
        weights: *const std::ffi::c_void,
        q8_1_input: *const std::ffi::c_void,
        output: *mut std::ffi::c_void,
        nrows: i32,
        ncols: i32,
        stream: hipStream_t,
    ) -> hipError_t;
}

#[link(name = "v1_gemv_q4_k_mmvq_residual", kind = "static")]
extern "C" {
    /// MMVQ with fused residual-add epilog (Phase 2 Schritt 2b). Same
    /// kernel structure as `rocmforge_launch_gemv_q4_k_mmvq` but the
    /// final store writes `dst[row] = dot + residual[row]`. Caller
    /// typically passes the same buffer as `residual` and `output`
    /// for in-place accumulation (matches the executor's decode
    /// residual-stream pattern).
    pub fn rocmforge_launch_gemv_q4_k_mmvq_residual(
        weights: *const std::ffi::c_void,
        q8_1_input: *const std::ffi::c_void,
        residual: *const std::ffi::c_void,
        output: *mut std::ffi::c_void,
        nrows: i32,
        ncols: i32,
        stream: hipStream_t,
    ) -> hipError_t;
}

#[link(name = "v1_gemv_q4_k_mmvq_fused", kind = "static")]
extern "C" {
    /// MMVQ with fused Gate + Up + SwiGLU (Phase 2 Schritt 3). Two
    /// parallel Q4_K × Q8_1 dot products over the SAME pre-quantized
    /// activation buffer, with `silu(gate_dot) * up_dot` applied at
    /// lane 0 before the single global store. Replaces the 5-kernel
    /// unfused sequence (quantize + mmvq_gate + quantize + mmvq_up +
    /// swiglu) with 2 kernels (quantize + mmvq_fused).
    ///
    /// `ncols` (K) must be a multiple of 256.
    pub fn rocmforge_launch_gemv_q4_k_mmvq_fused(
        weights_gate: *const std::ffi::c_void,
        weights_up: *const std::ffi::c_void,
        q8_1_input: *const std::ffi::c_void,
        output: *mut std::ffi::c_void,
        nrows: i32,
        ncols: i32,
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
