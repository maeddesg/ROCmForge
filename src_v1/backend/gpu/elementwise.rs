//! FFI bindings for the Phase-1 elementwise kernels.
//!
//! Block A: embedding lookup, RMSNorm (single + batched), residual add
//! (out-of-place + in-place). All FP32 on the host side; RMSNorm's
//! `eps` must come from GGUF metadata (`{arch}.attention.layer_norm_rms_epsilon`),
//! never hardcoded.

use super::hip_ffi::{hipError_t, hipStream_t};

#[link(name = "v1_elementwise_block_a", kind = "static")]
extern "C" {
    /// Gather `output[seq_pos * hidden_dim + d] = embedding_table[token_id * hidden_dim + d]`.
    /// `embedding_table` is a fully dequantised FP32 matrix — callers
    /// size it `vocab_size × hidden_dim × 4 B`. See the VRAM budget
    /// note in the Block-A report.
    pub fn rocmforge_launch_embedding_lookup(
        token_ids: *const u32,
        embedding_table: *const f32,
        output: *mut f32,
        seq_len: i32,
        hidden_dim: i32,
        stream: hipStream_t,
    ) -> hipError_t;

    /// RMSNorm for a single hidden-state row (`seq_len = 1`).
    /// `out[i] = x[i] * rsqrt(mean(x²) + eps) * weight[i]`.
    pub fn rocmforge_launch_rms_norm(
        x: *const f32,
        weight: *const f32,
        out: *mut f32,
        n: i32,
        eps: f32,
        stream: hipStream_t,
    ) -> hipError_t;

    /// Batched RMSNorm across `seq_len` rows.
    pub fn rocmforge_launch_rms_norm_batched(
        x: *const f32,
        weight: *const f32,
        out: *mut f32,
        n: i32,
        eps: f32,
        seq_len: i32,
        stream: hipStream_t,
    ) -> hipError_t;

    /// `output[i] = a[i] + b[i]` (out-of-place).
    pub fn rocmforge_launch_residual_add(
        output: *mut f32,
        a: *const f32,
        b: *const f32,
        n: i32,
        stream: hipStream_t,
    ) -> hipError_t;

    /// `a[i] += b[i]` (in-place).
    pub fn rocmforge_launch_residual_add_inplace(
        a: *mut f32,
        b: *const f32,
        n: i32,
        stream: hipStream_t,
    ) -> hipError_t;

    /// `out[i] = silu(gate[i]) * up[i]`. Used by the prefill
    /// gate+up+swiglu split path.
    pub fn rocmforge_launch_swiglu(
        gate: *const f32,
        up: *const f32,
        out: *mut f32,
        n: i32,
        stream: hipStream_t,
    ) -> hipError_t;
}

#[link(name = "v1_rope", kind = "static")]
extern "C" {
    /// Apply classic (non-NeoX) RoPE to a single-token tensor
    /// `x[num_heads × head_dim]` in-place. `freq_scale` may be NULL
    /// (standard RoPE) or a length-`head_dim/2` device pointer from the
    /// Llama-3.1 `rope_freqs.weight` tensor.
    pub fn rocmforge_launch_rope(
        x: *mut f32,
        pos: i32,
        num_heads: i32,
        head_dim: i32,
        theta_base: f32,
        freq_scale: *const f32,
        stream: hipStream_t,
    ) -> hipError_t;

    /// Batched prefill variant: each sequence row `s ∈ [0, seq_len)`
    /// gets rotation angle `(start_pos + s) * θ`. Layout
    /// `x[seq_len × num_heads × head_dim]`.
    pub fn rocmforge_launch_rope_batched(
        x: *mut f32,
        start_pos: i32,
        num_heads: i32,
        head_dim: i32,
        theta_base: f32,
        seq_len: i32,
        freq_scale: *const f32,
        stream: hipStream_t,
    ) -> hipError_t;
}
