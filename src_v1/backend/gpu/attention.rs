//! FFI for the Phase-1 attention / KV-cache kernels.
//!
//! Phase-1 philosophy: **reference-grade correctness, not peak speed.**
//! Attention is FP32 throughout with a non-tiled online-softmax; the
//! v0.x tiled FP16 flash-attention kernel (815 lines) is a Phase-2
//! optimisation. The prefill path is naive `O(seq²)` per head; all
//! three Phase-1 models' typical prompt lengths (≤ 8 192) fit within
//! the ~48 KiB per-block LDS budget for scores.
//!
//! KV cache layout:
//!   * `k_cache`, `v_cache`: row-major `[num_kv_heads × head_stride]` float32
//!   * `head_stride ≥ max_seq × head_dim`, callers align to 256 B so
//!     each head's slice lives in its own cache-line-aligned region
//!   * new token at `pos`: `[head * head_stride + pos * head_dim + d]`

use super::hip_ffi::{hipError_t, hipStream_t};

#[link(name = "v1_attention", kind = "static")]
extern "C" {
    /// Fused attention decode (`M = 1`). One block per Q-head; scores
    /// live in LDS. Tolerates `seq_len ≤ 12 288` on the 48 KiB smem cap;
    /// beyond that the launcher returns `hipErrorInvalidValue`.
    pub fn rocmforge_launch_attention_decode(
        q: *const f32,
        k_cache: *const f32,
        v_cache: *const f32,
        output: *mut f32,
        num_heads: i32,
        num_kv_heads: i32,
        head_dim: i32,
        seq_len: i32,
        head_stride: i32,
        scale: f32,
        stream: hipStream_t,
    ) -> hipError_t;

    /// Causal-masked attention prefill (`M > 1`). One block per
    /// (query_pos, q_head). O(seq²) per head — fine for Phase-1 prompt
    /// lengths; flash-attention tiling is Phase 2.
    pub fn rocmforge_launch_attention_prefill(
        q: *const f32,
        k: *const f32,
        v: *const f32,
        output: *mut f32,
        seq_len: i32,
        num_heads: i32,
        num_kv_heads: i32,
        head_dim: i32,
        scale: f32,
        stream: hipStream_t,
    ) -> hipError_t;

    /// Append one token's K and V vectors into the cache at `pos`.
    /// Both source buffers are `[num_kv_heads × head_dim]`.
    pub fn rocmforge_launch_kv_cache_append(
        k_cache: *mut f32,
        v_cache: *mut f32,
        k_new: *const f32,
        v_new: *const f32,
        num_kv_heads: i32,
        head_dim: i32,
        pos: i32,
        head_stride: i32,
        stream: hipStream_t,
    ) -> hipError_t;
}

#[link(name = "v1_kv_cache_fp8", kind = "static")]
extern "C" {
    /// Phase 2.2A — append to an FP8-E5M2 (bf8) KV cache. Inputs are
    /// still FP32 (fresh K/V from QKV projection + RoPE); the kernel
    /// converts per-element via `__builtin_amdgcn_cvt_pk_bf8_f32` and
    /// writes a single byte per element.
    ///
    /// Caller must have allocated `k_cache` / `v_cache` with 1 byte per
    /// element (`num_kv_heads × head_stride` bytes per cache) instead
    /// of 4 bytes like the FP32 variant. `head_stride` is still in
    /// **elements**, not bytes.
    pub fn rocmforge_launch_kv_cache_append_fp8(
        k_cache: *mut std::ffi::c_void,
        v_cache: *mut std::ffi::c_void,
        k_new: *const f32,
        v_new: *const f32,
        num_kv_heads: i32,
        head_dim: i32,
        pos: i32,
        head_stride: i32,
        stream: hipStream_t,
    ) -> hipError_t;

    /// Phase 2.2A — attention decode that reads bf8 KV cache.
    /// Identical algorithm to `rocmforge_launch_attention_decode`
    /// (FP32 online softmax) but with bf8 → FP32 conversion inline at
    /// each K/V load. `head_dim` must be a multiple of 4 (enforced by
    /// the launcher).
    pub fn rocmforge_launch_attention_decode_fp8(
        q: *const f32,
        k_cache: *const std::ffi::c_void,
        v_cache: *const std::ffi::c_void,
        output: *mut f32,
        num_heads: i32,
        num_kv_heads: i32,
        head_dim: i32,
        seq_len: i32,
        head_stride: i32,
        scale: f32,
        stream: hipStream_t,
    ) -> hipError_t;
}
