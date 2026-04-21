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
