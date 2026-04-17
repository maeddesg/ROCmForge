//! GPU KV cache and scratch buffers for inference.
//!
//! Safety-first design:
//! - All VRAM allocated with RAII (GpuBuffer)
//! - Bounds checked before kernel launches
//! - Never panic, always return GpuError

use super::device::GpuDevice;
use super::error::{GpuError, GpuResult};
use super::ffi::hipStream_t;
use super::graph::{CapturedDecodeGraph, DecodeGraphKey};
use super::kernels::{kv_write, kv_write_batched, kv_write_on_stream, zero_fill};
use super::weights::{GpuBuffer, GpuPinnedBuffer};
use super::forward::MAX_VERIFY_BATCH;
use crate::config::ModelConfig;

// ── KV Cache ─────────────────────────────────────────────────────────────────────

/// Key-value cache for autoregressive decoding, stored in GPU VRAM.
///
/// Layout: `k[layer][pos * kv_size + offset]` for position-based indexing.
/// All GPU memory managed via RAII (GpuBuffer).
pub struct GpuKvCache {
    /// Key cache: [num_layers][max_seq_len * kv_size]
    k: Vec<GpuBuffer>,
    /// Value cache: [num_layers][max_seq_len * kv_size]
    v: Vec<GpuBuffer>,
    /// Maximum sequence length this cache can hold
    pub max_seq_len: usize,
    /// Size of K/V per position: num_kv_heads * head_dim
    pub kv_size: usize,
    /// Number of layers
    pub num_layers: usize,
    /// Cached pointer-mix used by decode-graph key construction.
    decode_binding_tag: u64,
}

impl GpuKvCache {
    /// Allocate a new KV cache in GPU VRAM.
    ///
    /// # Arguments
    /// * `config` - Model configuration (determines num_layers, num_kv_heads, head_dim)
    /// * `max_seq_len` - Maximum sequence length to support
    ///
    /// # Returns
    /// Ok(GpuKvCache) if all allocations succeed, Err if any fail (all freed via RAII)
    pub fn new(config: &ModelConfig, max_seq_len: usize) -> GpuResult<Self> {
        let kv_size = config.num_kv_heads * config.head_dim;
        let layer_bytes = max_seq_len * kv_size * std::mem::size_of::<u16>();

        // Allocate K cache per layer
        let mut k = Vec::with_capacity(config.num_layers);
        for layer in 0..config.num_layers {
            let buf = GpuBuffer::alloc(layer_bytes).map_err(|e| {
                // On error, all previously allocated buffers are dropped (RAII cleanup)
                GpuError::CacheAllocationFailed {
                    reason: format!("K cache layer {} allocation failed: {}", layer, e),
                }
            })?;
            k.push(buf);
        }

        // Allocate V cache per layer
        let mut v = Vec::with_capacity(config.num_layers);
        for layer in 0..config.num_layers {
            let buf =
                GpuBuffer::alloc(layer_bytes).map_err(|e| GpuError::CacheAllocationFailed {
                    reason: format!("V cache layer {} allocation failed: {}", layer, e),
                })?;
            v.push(buf);
        }

        let decode_binding_tag = compute_kv_binding_tag(&k, &v);

        Ok(Self {
            k,
            v,
            max_seq_len,
            kv_size,
            num_layers: config.num_layers,
            decode_binding_tag,
        })
    }

    /// Get GPU pointer to K cache for a layer.
    ///
    /// Returns pointer suitable for kernel arguments.
    pub fn k_ptr(&self, layer: usize) -> GpuResult<*mut u16> {
        if layer >= self.num_layers {
            return Err(GpuError::HipApiError {
                code: -1,
                description: format!("Layer {} exceeds num_layers {}", layer, self.num_layers),
            });
        }
        Ok(self.k[layer].as_ptr() as *mut u16)
    }

    /// Get GPU pointer to V cache for a layer (FP16 storage).
    pub fn v_ptr(&self, layer: usize) -> GpuResult<*mut u16> {
        if layer >= self.num_layers {
            return Err(GpuError::HipApiError {
                code: -1,
                description: format!("Layer {} exceeds num_layers {}", layer, self.num_layers),
            });
        }
        Ok(self.v[layer].as_ptr() as *mut u16)
    }

    /// Write K/V vectors to cache at specific position using GPU kernel.
    ///
    /// # Arguments
    /// * `layer` - Layer index
    /// * `pos` - Position in cache (must be < max_seq_len)
    /// * `k_gpu` - GPU pointer to key vector
    /// * `v_gpu` - GPU pointer to value vector
    ///
    /// # Returns
    /// Ok(()) on successful kernel launch
    pub fn write(
        &self,
        layer: usize,
        pos: usize,
        k_gpu: *const f32,
        v_gpu: *const f32,
    ) -> GpuResult<()> {
        let k_cache = self.k_ptr(layer)?;
        let v_cache = self.v_ptr(layer)?;

        kv_write(
            k_cache,
            v_cache,
            k_gpu,
            v_gpu,
            pos,
            self.kv_size,
            self.max_seq_len,
        )
    }

    /// Write K/V vectors to cache on an explicit HIP stream.
    pub fn write_on_stream(
        &self,
        layer: usize,
        pos: usize,
        k_gpu: *const f32,
        v_gpu: *const f32,
        stream: hipStream_t,
    ) -> GpuResult<()> {
        let k_cache = self.k_ptr(layer)?;
        let v_cache = self.v_ptr(layer)?;

        kv_write_on_stream(
            k_cache,
            v_cache,
            k_gpu,
            v_gpu,
            pos,
            self.kv_size,
            self.max_seq_len,
            stream,
        )
    }

    /// Batch write K/V for prefill (multiple positions).
    ///
    /// # Arguments
    /// * `start_pos` - Starting position
    /// * `seq_len` - Number of positions to write
    /// * `k_gpu` - GPU pointer to batched key vectors [seq_len * kv_size]
    /// * `v_gpu` - GPU pointer to batched value vectors
    pub fn write_batched(
        &self,
        layer: usize,
        start_pos: usize,
        seq_len: usize,
        k_gpu: *const f32,
        v_gpu: *const f32,
    ) -> GpuResult<()> {
        let k_cache = self.k_ptr(layer)?;
        let v_cache = self.v_ptr(layer)?;

        kv_write_batched(
            k_cache,
            v_cache,
            k_gpu,
            v_gpu,
            start_pos,
            self.kv_size,
            self.max_seq_len,
            seq_len,
        )
    }

    /// Clear all cached values (zero out via kernel).
    ///
    /// Requires device reference for kernel synchronization.
    pub fn clear(&mut self, device: &GpuDevice) -> GpuResult<()> {
        // FP16 cache: each element is u16, but zero_fill works on f32.
        // Since zero bits are the same for both types, cast and use half the count.
        let f32_elements_per_layer = (self.max_seq_len * self.kv_size + 1) / 2;

        // Zero out K cache for each layer
        for layer in 0..self.num_layers {
            let k_ptr = self.k[layer].as_ptr() as *mut f32;
            zero_fill(k_ptr, f32_elements_per_layer, device)?;
        }

        // Zero out V cache for each layer
        for layer in 0..self.num_layers {
            let v_ptr = self.v[layer].as_ptr() as *mut f32;
            zero_fill(v_ptr, f32_elements_per_layer, device)?;
        }

        Ok(())
    }

    /// Get total VRAM usage in bytes.
    pub fn memory_bytes(&self) -> usize {
        let bytes_per_layer = self.max_seq_len * self.kv_size * std::mem::size_of::<u16>();
        let total_bytes = 2 * self.num_layers * bytes_per_layer; // K + V
        total_bytes
    }

    /// Cached pointer-mix used by decode-graph key construction.
    #[inline]
    pub fn binding_tag(&self) -> u64 {
        self.decode_binding_tag
    }
}

#[inline]
fn mix_binding_tag(tag: u64, ptr: usize) -> u64 {
    tag.rotate_left(13) ^ (ptr as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)
}

fn compute_kv_binding_tag(k: &[GpuBuffer], v: &[GpuBuffer]) -> u64 {
    let mut tag = 0u64;
    for buffer in k {
        tag = mix_binding_tag(tag, buffer.as_ptr() as usize);
    }
    for buffer in v {
        tag = mix_binding_tag(tag, buffer.as_ptr() as usize);
    }
    tag
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_config() -> crate::config::ModelConfig {
        crate::config::ModelConfig {
            num_layers: 2,
            num_kv_heads: 4,
            head_dim: 128,
            max_seq_len: 512,
            hidden_size: 1024,
            num_heads: 8,
            intermediate_size: 2048,
            vocab_size: 32000,
            rms_norm_eps: 1e-5,
            rope_theta: 10000.0,
            rope_neox: false,
            use_attention_bias: false,
            attention_layout: crate::config::AttentionLayout::SplitQkv,
            architecture: "test".to_string(),
            tensor_registry: crate::config::TensorNameRegistry::from_scheme(
                &crate::config::TensorNamingScheme::Gguf,
            ),
        }
    }

    #[test]
    fn new_allocates_correct_buffers() {
        let config = make_test_config();
        let cache = GpuKvCache::new(&config, 256);

        // Will fail without GPU, that's expected
        // Test that allocation is attempted correctly
        match cache {
            Ok(c) => {
                assert_eq!(c.num_layers, 2);
                assert_eq!(c.max_seq_len, 256);
                assert_eq!(c.kv_size, 4 * 128);
            }
            Err(_) => {
                // Expected when HIP unavailable
            }
        }
    }

    #[test]
    fn k_ptr_validates_layer_bounds() {
        let config = make_test_config();

        // Create a cache - will fail without GPU
        let cache = GpuKvCache::new(&config, 256);
        if let Ok(cache) = cache {
            let result = cache.k_ptr(5); // layer 5 > num_layers (2)
            assert!(result.is_err());
        }
        // If allocation failed, test passes (bounds checking exists)
    }

    #[test]
    fn binding_tag_is_stable_for_same_cache() {
        let config = make_test_config();
        let cache = GpuKvCache::new(&config, 256);
        if let Ok(cache) = cache {
            let tag_a = cache.binding_tag();
            let tag_b = cache.binding_tag();
            assert_eq!(tag_a, tag_b);
        }
    }

    #[test]
    fn binding_tag_differs_for_distinct_allocations() {
        let config = make_test_config();
        let first = GpuKvCache::new(&config, 256);
        let second = GpuKvCache::new(&config, 256);
        if let (Ok(first), Ok(second)) = (first, second) {
            assert_ne!(first.binding_tag(), second.binding_tag());
        }
    }
}

// ── Forward Scratch Buffers ───────────────────────────────────────────────────────

/// Reusable scratch buffers in GPU VRAM for a single forward pass.
///
/// Allocated once and reused across all layers to avoid repeated allocations.
/// All buffers are GPU-resident.
const GPU_ARGMAX_BLOCK_SIZE: usize = 256;
const GPU_ARGMAX_ITEMS_PER_THREAD: usize = 4;
const GPU_ARGMAX_ITEMS_PER_BLOCK: usize = GPU_ARGMAX_BLOCK_SIZE * GPU_ARGMAX_ITEMS_PER_THREAD;

pub struct GpuForwardScratch {
    /// Current hidden state [hidden_size]
    pub hidden: GpuBuffer,
    /// Normalized hidden state [hidden_size]
    pub normed: GpuBuffer,
    /// Query vector [num_heads * head_dim]
    pub q: GpuBuffer,
    /// Key vector [num_kv_heads * head_dim]
    pub k: GpuBuffer,
    /// Value vector [num_kv_heads * head_dim]
    pub v: GpuBuffer,
    /// Attention output [num_heads * head_dim]
    pub attn_out: GpuBuffer,
    /// Layer output (residual stream) [hidden_size]
    pub layer_out: GpuBuffer,
    /// FFN gate projection [intermediate_size]
    pub gate: GpuBuffer,
    /// FFN SwiGLU output [intermediate_size]
    pub swiglu: GpuBuffer,
    /// Final logits [vocab_size]
    pub logits: GpuBuffer,
    /// Partial argmax values for greedy decode [ceil(vocab_size / 1024)]
    pub argmax_partial_values: GpuBuffer,
    /// Partial argmax indices for greedy decode [ceil(vocab_size / 1024)]
    pub argmax_partial_indices: GpuBuffer,
    /// Final greedy token index [1] - Device destination
    pub argmax_result_device: GpuBuffer,
    /// Final greedy token index [1] - Pinned host buffer for async overlap
    pub argmax_result_index: GpuPinnedBuffer,
    /// Pinned host buffer for hidden state upload overlap
    pub input_hidden_pinned: GpuPinnedBuffer,
    /// Per-token decode state uploaded before full-graph replay: [pos, seq_len]
    decode_state: GpuBuffer,
    /// Pinned host staging for decode state upload to keep H2D async and tiny.
    decode_state_host: GpuPinnedBuffer,
    /// Host-tracked decode position currently resident in `decode_state[0]`.
    decode_state_next_pos: Option<usize>,
    /// Cached executable graph for repeated decode work.
    captured_decode: Option<CapturedDecodeGraph>,
    /// Atomic retire counter for inter-block sync in fused GEMV+residual+norm kernel [1 u32].
    /// Self-resetting via atomicInc wrap-around.
    pub retire_count: GpuBuffer,
    /// Batched logits for verify lm_head [MAX_VERIFY_BATCH × vocab_size f32]
    pub logits_batch: GpuBuffer,
    /// Batched argmax result indices on device [MAX_VERIFY_BATCH × i32]
    pub argmax_batch_device: GpuBuffer,
    /// Batched argmax result indices on pinned host [MAX_VERIFY_BATCH × i32]
    pub argmax_batch_host: GpuPinnedBuffer,
}

impl GpuForwardScratch {
    /// Allocate scratch buffers in GPU VRAM.
    ///
    /// # Arguments
    /// * `config` - Model configuration
    ///
    /// # Returns
    /// Ok(GpuForwardScratch) if all allocations succeed
    pub fn new(config: &ModelConfig) -> GpuResult<Self> {
        let h = config.hidden_size;
        let q = config.num_heads * config.head_dim;
        let kv = config.num_kv_heads * config.head_dim;
        let ff = config.intermediate_size;
        let v = config.vocab_size;
        let argmax_partials = v.div_ceil(GPU_ARGMAX_ITEMS_PER_BLOCK);

        // Allocate all buffers - if any fail, all are freed via RAII
        let hidden = GpuBuffer::alloc(h * std::mem::size_of::<f32>()).map_err(|e| {
            GpuError::CacheAllocationFailed {
                reason: format!("hidden buffer allocation failed: {}", e),
            }
        })?;

        let normed = GpuBuffer::alloc(h * std::mem::size_of::<f32>()).map_err(|e| {
            GpuError::CacheAllocationFailed {
                reason: format!("normed buffer allocation failed: {}", e),
            }
        })?;

        let q_buf = GpuBuffer::alloc(q * std::mem::size_of::<f32>()).map_err(|e| {
            GpuError::CacheAllocationFailed {
                reason: format!("Q buffer allocation failed: {}", e),
            }
        })?;

        let k_buf = GpuBuffer::alloc(kv * std::mem::size_of::<f32>()).map_err(|e| {
            GpuError::CacheAllocationFailed {
                reason: format!("K buffer allocation failed: {}", e),
            }
        })?;

        let v_buf = GpuBuffer::alloc(kv * std::mem::size_of::<f32>()).map_err(|e| {
            GpuError::CacheAllocationFailed {
                reason: format!("V buffer allocation failed: {}", e),
            }
        })?;

        let attn_out = GpuBuffer::alloc(q * std::mem::size_of::<f32>()).map_err(|e| {
            GpuError::CacheAllocationFailed {
                reason: format!("attn_out buffer allocation failed: {}", e),
            }
        })?;

        let layer_out = GpuBuffer::alloc(h * std::mem::size_of::<f32>()).map_err(|e| {
            GpuError::CacheAllocationFailed {
                reason: format!("layer_out buffer allocation failed: {}", e),
            }
        })?;

        let gate = GpuBuffer::alloc(ff * std::mem::size_of::<f32>()).map_err(|e| {
            GpuError::CacheAllocationFailed {
                reason: format!("gate buffer allocation failed: {}", e),
            }
        })?;

        let swiglu = GpuBuffer::alloc(ff * std::mem::size_of::<f32>()).map_err(|e| {
            GpuError::CacheAllocationFailed {
                reason: format!("swiglu buffer allocation failed: {}", e),
            }
        })?;

        let logits = GpuBuffer::alloc(v * std::mem::size_of::<f32>()).map_err(|e| {
            GpuError::CacheAllocationFailed {
                reason: format!("logits buffer allocation failed: {}", e),
            }
        })?;

        let argmax_partial_values = GpuBuffer::alloc(argmax_partials * std::mem::size_of::<f32>())
            .map_err(|e| GpuError::CacheAllocationFailed {
                reason: format!("argmax partial values allocation failed: {}", e),
            })?;

        let argmax_partial_indices = GpuBuffer::alloc(argmax_partials * std::mem::size_of::<i32>())
            .map_err(|e| GpuError::CacheAllocationFailed {
                reason: format!("argmax partial indices allocation failed: {}", e),
            })?;

        let argmax_result_device = GpuBuffer::alloc(std::mem::size_of::<i32>()).map_err(|e| {
            GpuError::CacheAllocationFailed {
                reason: format!("argmax result device allocation failed: {}", e),
            }
        })?;

        let argmax_result_index =
            GpuPinnedBuffer::alloc(std::mem::size_of::<i32>()).map_err(|e| {
                GpuError::CacheAllocationFailed {
                    reason: format!("argmax result allocation failed: {}", e),
                }
            })?;
        let input_hidden_pinned =
            GpuPinnedBuffer::alloc(h * std::mem::size_of::<f32>()).map_err(|e| {
                GpuError::CacheAllocationFailed {
                    reason: format!("input hidden pinned allocation failed: {}", e),
                }
            })?;
        let decode_state = GpuBuffer::alloc(2 * std::mem::size_of::<i32>()).map_err(|e| {
            GpuError::CacheAllocationFailed {
                reason: format!("decode state allocation failed: {}", e),
            }
        })?;
        let decode_state_host =
            GpuPinnedBuffer::alloc(2 * std::mem::size_of::<i32>()).map_err(|e| {
                GpuError::CacheAllocationFailed {
                    reason: format!("decode state host allocation failed: {}", e),
                }
            })?;

        let mut retire_count =
            GpuBuffer::alloc(std::mem::size_of::<u32>()).map_err(|e| {
                GpuError::CacheAllocationFailed {
                    reason: format!("retire_count buffer allocation failed: {}", e),
                }
            })?;
        retire_count.copy_from_host(&[0u8; 4])?;

        // Oversize to at least 64 rows so the WMMA prefill GEMM can safely
        // dispatch at padded_M = 64 when verify calls gpu_dispatch_gemm
        // with n ≤ MAX_VERIFY_BATCH rows (Phase 3.2). Only the first n rows
        // are ever read; the trailing rows are WMMA output of zero-padded
        // inputs (i.e. zero) and harmless.
        let logits_batch_rows = MAX_VERIFY_BATCH.max(64);
        let logits_batch =
            GpuBuffer::alloc(logits_batch_rows * v * std::mem::size_of::<f32>()).map_err(|e| {
                GpuError::CacheAllocationFailed {
                    reason: format!("logits_batch allocation failed: {}", e),
                }
            })?;

        let argmax_batch_device =
            GpuBuffer::alloc(MAX_VERIFY_BATCH * std::mem::size_of::<i32>()).map_err(|e| {
                GpuError::CacheAllocationFailed {
                    reason: format!("argmax_batch_device allocation failed: {}", e),
                }
            })?;

        let argmax_batch_host =
            GpuPinnedBuffer::alloc(MAX_VERIFY_BATCH * std::mem::size_of::<i32>()).map_err(|e| {
                GpuError::CacheAllocationFailed {
                    reason: format!("argmax_batch_host allocation failed: {}", e),
                }
            })?;

        Ok(Self {
            hidden,
            normed,
            q: q_buf,
            k: k_buf,
            v: v_buf,
            attn_out,
            layer_out,
            gate,
            swiglu,
            logits,
            argmax_partial_values,
            argmax_partial_indices,
            argmax_result_device,
            argmax_result_index,
            input_hidden_pinned,
            decode_state,
            decode_state_host,
            decode_state_next_pos: None,
            captured_decode: None,
            retire_count,
            logits_batch,
            argmax_batch_device,
            argmax_batch_host,
        })
    }

    /// Get GPU pointer to current hidden state.
    pub fn hidden_ptr(&self) -> *const f32 {
        self.hidden.as_ptr() as *const f32
    }

    /// Get mutable GPU pointer to current hidden state.
    pub fn hidden_mut_ptr(&mut self) -> *mut f32 {
        self.hidden.as_ptr() as *mut f32
    }

    /// Get GPU pointer to normalized hidden state
    pub fn normed_ptr(&self) -> *const f32 {
        self.normed.as_ptr() as *const f32
    }

    /// Get mutable GPU pointer to normalized hidden state.
    pub fn normed_mut_ptr(&mut self) -> *mut f32 {
        self.normed.as_ptr() as *mut f32
    }

    /// Get GPU pointer to query vector
    pub fn q_ptr(&self) -> *const f32 {
        self.q.as_ptr() as *const f32
    }

    /// Get mutable GPU pointer to query vector.
    pub fn q_mut_ptr(&mut self) -> *mut f32 {
        self.q.as_ptr() as *mut f32
    }

    /// Get GPU pointer to key vector
    pub fn k_ptr(&self) -> *const f32 {
        self.k.as_ptr() as *const f32
    }

    /// Get mutable GPU pointer to key vector.
    pub fn k_mut_ptr(&mut self) -> *mut f32 {
        self.k.as_ptr() as *mut f32
    }

    /// Get GPU pointer to value vector
    pub fn v_ptr(&self) -> *const f32 {
        self.v.as_ptr() as *const f32
    }

    /// Get mutable GPU pointer to value vector.
    pub fn v_mut_ptr(&mut self) -> *mut f32 {
        self.v.as_ptr() as *mut f32
    }

    /// Get GPU pointer to attention output.
    pub fn attn_out_ptr(&self) -> *const f32 {
        self.attn_out.as_ptr() as *const f32
    }

    /// Get mutable GPU pointer to attention output.
    pub fn attn_out_mut_ptr(&mut self) -> *mut f32 {
        self.attn_out.as_ptr() as *mut f32
    }

    /// Get GPU pointer to layer output.
    pub fn layer_out_ptr(&self) -> *const f32 {
        self.layer_out.as_ptr() as *const f32
    }

    /// Get mutable GPU pointer to layer output.
    pub fn layer_out_mut_ptr(&mut self) -> *mut f32 {
        self.layer_out.as_ptr() as *mut f32
    }

    /// Get GPU pointer to FFN gate activations.
    pub fn gate_ptr(&self) -> *const f32 {
        self.gate.as_ptr() as *const f32
    }

    /// Get mutable GPU pointer to FFN gate activations.
    pub fn gate_mut_ptr(&mut self) -> *mut f32 {
        self.gate.as_ptr() as *mut f32
    }

    /// Get GPU pointer to SwiGLU activations.
    pub fn swiglu_ptr(&self) -> *const f32 {
        self.swiglu.as_ptr() as *const f32
    }

    /// Get mutable GPU pointer to SwiGLU activations.
    pub fn swiglu_mut_ptr(&mut self) -> *mut f32 {
        self.swiglu.as_ptr() as *mut f32
    }

    /// Get GPU pointer to logits.
    pub fn logits_ptr(&self) -> *const f32 {
        self.logits.as_ptr() as *const f32
    }

    /// Get mutable GPU pointer to logits.
    pub fn logits_mut_ptr(&mut self) -> *mut f32 {
        self.logits.as_ptr() as *mut f32
    }

    /// Get GPU pointer to argmax partial values.
    pub fn argmax_partial_values_mut_ptr(&mut self) -> *mut f32 {
        self.argmax_partial_values.as_ptr() as *mut f32
    }

    /// Get GPU pointer to argmax partial indices.
    pub fn argmax_partial_indices_mut_ptr(&mut self) -> *mut i32 {
        self.argmax_partial_indices.as_ptr() as *mut i32
    }

    /// Get GPU pointer to final argmax index.
    pub fn argmax_result_index_mut_ptr(&mut self) -> *mut i32 {
        self.argmax_result_device.as_ptr() as *mut i32
    }

    pub fn decode_pos_ptr(&self) -> *const i32 {
        self.decode_state.as_ptr() as *const i32
    }

    pub fn decode_seq_len_ptr(&self) -> *const i32 {
        unsafe { (self.decode_state.as_ptr() as *const i32).add(1) }
    }

    pub fn decode_state_mut_ptr(&mut self) -> *mut i32 {
        self.decode_state.as_ptr() as *mut i32
    }

    pub fn decode_state_matches_pos(&self, pos: usize) -> bool {
        self.decode_state_next_pos == Some(pos)
    }

    pub fn mark_decode_state_next_pos(&mut self, pos: usize) {
        self.decode_state_next_pos = Some(pos);
    }

    pub fn decode_state_next_pos(&self) -> Option<usize> {
        self.decode_state_next_pos
    }

    pub fn upload_decode_state(
        &mut self,
        pos: usize,
        seq_len: usize,
        stream: hipStream_t,
    ) -> GpuResult<()> {
        let pos_i32 = i32::try_from(pos).map_err(|_| GpuError::HipApiError {
            code: -1,
            description: format!("decode pos {} exceeds i32 range", pos),
        })?;
        let seq_len_i32 = i32::try_from(seq_len).map_err(|_| GpuError::HipApiError {
            code: -1,
            description: format!("decode seq_len {} exceeds i32 range", seq_len),
        })?;
        let state = self.decode_state_host.as_slice_mut::<i32>();
        state[0] = pos_i32;
        state[1] = seq_len_i32;
        let state_bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(
                self.decode_state_host.as_ptr() as *const u8,
                2 * std::mem::size_of::<i32>(),
            )
        };
        self.decode_state
            .copy_from_host_on_stream(state_bytes, stream)?;
        self.decode_state_next_pos = Some(pos);
        Ok(())
    }

    pub fn decode_graph(&self) -> Option<&CapturedDecodeGraph> {
        self.captured_decode.as_ref()
    }

    pub fn decode_graph_mut(&mut self) -> Option<&mut CapturedDecodeGraph> {
        self.captured_decode.as_mut()
    }

    pub fn has_decode_graph_for(&self, key: DecodeGraphKey) -> bool {
        self.captured_decode
            .as_ref()
            .is_some_and(|graph| graph.matches_key(key))
    }

    pub fn replace_decode_graph(
        &mut self,
        graph: CapturedDecodeGraph,
    ) -> Option<CapturedDecodeGraph> {
        self.decode_state_next_pos = None;
        self.captured_decode.replace(graph)
    }

    pub fn try_update_decode_graph(
        &mut self,
        new_graph: &crate::gpu::graph::HipGraph,
    ) -> GpuResult<bool> {
        if let Some(graph) = &self.captured_decode {
            let updated = graph.update(new_graph)?;
            if updated {
                self.decode_state_next_pos = None;
            }
            Ok(updated)
        } else {
            Ok(false)
        }
    }

    pub fn clear_decode_graph(&mut self) {
        self.captured_decode = None;
        self.decode_state_next_pos = None;
    }
}

/// Reusable scratch buffers in GPU VRAM for batched prompt prefill.
///
/// Layout is row-major `[buffer_seq_len, dim]` for all activation buffers.
/// When `seq_len >= 64` the buffers are allocated at `buffer_seq_len`
/// (the next multiple of 64) and zero-initialised; all ops continue to
/// process only the first `seq_len` rows, and the WMMA prefill kernels
/// read/write the full `buffer_seq_len` rows with the trailing padding
/// rows staying zero throughout the forward pass. This lets the WMMA
/// paths activate for arbitrary `seq_len >= 64` without a per-call
/// zero-memset.
pub struct GpuPrefillScratch {
    pub seq_len: usize,
    /// Physical row count of every activation buffer. Equal to `seq_len`
    /// below the WMMA threshold and the next multiple of 64 at or above
    /// it.
    pub buffer_seq_len: usize,
    pub hidden: GpuBuffer,
    pub normed: GpuBuffer,
    pub q: GpuBuffer,
    pub k: GpuBuffer,
    pub v: GpuBuffer,
    pub attn_out: GpuBuffer,
    pub layer_out: GpuBuffer,
    pub gate: GpuBuffer,
    pub swiglu: GpuBuffer,
    pub token_ids: GpuBuffer,
    /// FP16 staging buffers for the WMMA prefill attention kernel. The kernel
    /// expects FP16 Q/K/V; the activation path produces FP32, so we convert
    /// per layer before dispatch. Lazily allocated on first use and
    /// zero-initialised at allocation so padding rows stay zero across
    /// dispatches.
    pub q_f16: Option<GpuBuffer>,
    pub k_f16: Option<GpuBuffer>,
    pub v_f16: Option<GpuBuffer>,
}

impl GpuPrefillScratch {
    pub fn new(config: &ModelConfig, seq_len: usize) -> GpuResult<Self> {
        if seq_len == 0 {
            return Err(GpuError::CacheAllocationFailed {
                reason: "prefill seq_len cannot be zero".to_string(),
            });
        }

        let h = config.hidden_size;
        let q = config.num_heads * config.head_dim;
        let kv = config.num_kv_heads * config.head_dim;
        let ff = config.intermediate_size;

        // Phase 3.1/3.2: oversize activation buffers to the next multiple
        // of 64 (minimum 64 rows) so the WMMA prefill path can dispatch at
        // `buffer_seq_len` while downstream ops keep running on the logical
        // `seq_len`. The WMMA kernels require M ≥ 64, so even short prompts
        // pad up to 64 — this is what brings the typical 20–40-token real
        // prompt into the WMMA path.
        let buffer_seq_len = seq_len.max(1).div_ceil(64).max(1) * 64;

        let hidden = GpuBuffer::alloc(buffer_seq_len * h * std::mem::size_of::<f32>()).map_err(
            |e| GpuError::CacheAllocationFailed {
                reason: format!("prefill hidden allocation failed: {}", e),
            },
        )?;
        let normed = GpuBuffer::alloc(buffer_seq_len * h * std::mem::size_of::<f32>()).map_err(
            |e| GpuError::CacheAllocationFailed {
                reason: format!("prefill normed allocation failed: {}", e),
            },
        )?;
        let q_buf = GpuBuffer::alloc(buffer_seq_len * q * std::mem::size_of::<f32>()).map_err(
            |e| GpuError::CacheAllocationFailed {
                reason: format!("prefill q allocation failed: {}", e),
            },
        )?;
        let k_buf = GpuBuffer::alloc(buffer_seq_len * kv * std::mem::size_of::<f32>()).map_err(
            |e| GpuError::CacheAllocationFailed {
                reason: format!("prefill k allocation failed: {}", e),
            },
        )?;
        let v_buf = GpuBuffer::alloc(buffer_seq_len * kv * std::mem::size_of::<f32>()).map_err(
            |e| GpuError::CacheAllocationFailed {
                reason: format!("prefill v allocation failed: {}", e),
            },
        )?;
        let attn_out =
            GpuBuffer::alloc(buffer_seq_len * q * std::mem::size_of::<f32>()).map_err(|e| {
                GpuError::CacheAllocationFailed {
                    reason: format!("prefill attn_out allocation failed: {}", e),
                }
            })?;
        let layer_out =
            GpuBuffer::alloc(buffer_seq_len * h * std::mem::size_of::<f32>()).map_err(|e| {
                GpuError::CacheAllocationFailed {
                    reason: format!("prefill layer_out allocation failed: {}", e),
                }
            })?;
        let gate = GpuBuffer::alloc(buffer_seq_len * ff * std::mem::size_of::<f32>()).map_err(
            |e| GpuError::CacheAllocationFailed {
                reason: format!("prefill gate allocation failed: {}", e),
            },
        )?;
        let swiglu = GpuBuffer::alloc(buffer_seq_len * ff * std::mem::size_of::<f32>()).map_err(
            |e| GpuError::CacheAllocationFailed {
                reason: format!("prefill swiglu allocation failed: {}", e),
            },
        )?;
        let token_ids = GpuBuffer::alloc(seq_len * std::mem::size_of::<i32>()).map_err(|e| {
            GpuError::CacheAllocationFailed {
                reason: format!("prefill token_ids allocation failed: {}", e),
            }
        })?;

        // Zero-initialise all activation buffers. The padding rows
        // [seq_len..buffer_seq_len] stay zero for the entire forward pass
        // because every op below the WMMA dispatch iterates only over
        // `seq_len` rows. `buffer_seq_len` is always at least 64 now, so
        // this runs unconditionally.
        {
            use super::ffi::{hip_memset_async, hip_stream_synchronize, hipStream_t};
            let stream = hipStream_t::null();
            for buf in [&hidden, &normed, &q_buf, &k_buf, &v_buf, &attn_out, &layer_out, &gate, &swiglu] {
                hip_memset_async(buf.as_ptr(), 0, buf.size(), stream).map_err(|e| {
                    GpuError::CacheAllocationFailed {
                        reason: format!("prefill scratch zero-init failed: {}", e),
                    }
                })?;
            }
            hip_stream_synchronize(stream).map_err(|e| GpuError::CacheAllocationFailed {
                reason: format!("prefill scratch zero-init sync failed: {}", e),
            })?;
        }

        Ok(Self {
            seq_len,
            buffer_seq_len,
            hidden,
            normed,
            q: q_buf,
            k: k_buf,
            v: v_buf,
            attn_out,
            layer_out,
            gate,
            swiglu,
            token_ids,
            q_f16: None,
            k_f16: None,
            v_f16: None,
        })
    }

    /// Ensure the FP16 Q/K/V staging buffers used by the WMMA prefill
    /// attention kernel are allocated. Lazy — callers that stay on the scalar
    /// attention path pay no VRAM cost. Sized to `buffer_seq_len` and
    /// zero-initialised so the padding rows [seq_len..buffer_seq_len] stay
    /// zero across dispatches (the FP32 → FP16 conversion only overwrites
    /// the first `seq_len` rows).
    pub fn ensure_attention_f16_buffers(
        &mut self,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
    ) -> GpuResult<()> {
        use super::ffi::{hip_memset_async, hip_stream_synchronize, hipStream_t};
        let q_bytes = self.buffer_seq_len * num_heads * head_dim * std::mem::size_of::<u16>();
        let kv_bytes = self.buffer_seq_len * num_kv_heads * head_dim * std::mem::size_of::<u16>();
        let mut did_alloc = false;
        if self.q_f16.is_none() {
            let buf = GpuBuffer::alloc(q_bytes).map_err(|e| {
                GpuError::CacheAllocationFailed {
                    reason: format!("prefill q_f16 allocation failed: {}", e),
                }
            })?;
            hip_memset_async(buf.as_ptr(), 0, buf.size(), hipStream_t::null()).map_err(|e| {
                GpuError::CacheAllocationFailed {
                    reason: format!("q_f16 zero-init failed: {}", e),
                }
            })?;
            self.q_f16 = Some(buf);
            did_alloc = true;
        }
        if self.k_f16.is_none() {
            let buf = GpuBuffer::alloc(kv_bytes).map_err(|e| {
                GpuError::CacheAllocationFailed {
                    reason: format!("prefill k_f16 allocation failed: {}", e),
                }
            })?;
            hip_memset_async(buf.as_ptr(), 0, buf.size(), hipStream_t::null()).map_err(|e| {
                GpuError::CacheAllocationFailed {
                    reason: format!("k_f16 zero-init failed: {}", e),
                }
            })?;
            self.k_f16 = Some(buf);
            did_alloc = true;
        }
        if self.v_f16.is_none() {
            let buf = GpuBuffer::alloc(kv_bytes).map_err(|e| {
                GpuError::CacheAllocationFailed {
                    reason: format!("prefill v_f16 allocation failed: {}", e),
                }
            })?;
            hip_memset_async(buf.as_ptr(), 0, buf.size(), hipStream_t::null()).map_err(|e| {
                GpuError::CacheAllocationFailed {
                    reason: format!("v_f16 zero-init failed: {}", e),
                }
            })?;
            self.v_f16 = Some(buf);
            did_alloc = true;
        }
        if did_alloc {
            hip_stream_synchronize(hipStream_t::null()).map_err(|e| {
                GpuError::CacheAllocationFailed {
                    reason: format!("attention FP16 zero-init sync failed: {}", e),
                }
            })?;
        }
        Ok(())
    }

    pub fn hidden_row_ptr(&self, row: usize, hidden_size: usize) -> *const f32 {
        debug_assert!(row < self.seq_len);
        unsafe { (self.hidden.as_ptr() as *const f32).add(row * hidden_size) }
    }

    pub fn normed_row_ptr(&self, row: usize, hidden_size: usize) -> *const f32 {
        debug_assert!(row < self.seq_len);
        unsafe { (self.normed.as_ptr() as *const f32).add(row * hidden_size) }
    }

    pub fn normed_row_mut_ptr(&mut self, row: usize, hidden_size: usize) -> *mut f32 {
        debug_assert!(row < self.seq_len);
        unsafe { (self.normed.as_ptr() as *mut f32).add(row * hidden_size) }
    }

    pub fn q_row_mut_ptr(&mut self, row: usize, q_size: usize) -> *mut f32 {
        debug_assert!(row < self.seq_len);
        unsafe { (self.q.as_ptr() as *mut f32).add(row * q_size) }
    }

    pub fn k_row_mut_ptr(&mut self, row: usize, kv_size: usize) -> *mut f32 {
        debug_assert!(row < self.seq_len);
        unsafe { (self.k.as_ptr() as *mut f32).add(row * kv_size) }
    }

    pub fn v_row_mut_ptr(&mut self, row: usize, kv_size: usize) -> *mut f32 {
        debug_assert!(row < self.seq_len);
        unsafe { (self.v.as_ptr() as *mut f32).add(row * kv_size) }
    }

    pub fn attn_out_row_mut_ptr(&mut self, row: usize, q_size: usize) -> *mut f32 {
        debug_assert!(row < self.seq_len);
        unsafe { (self.attn_out.as_ptr() as *mut f32).add(row * q_size) }
    }

    pub fn layer_out_row_mut_ptr(&mut self, row: usize, hidden_size: usize) -> *mut f32 {
        debug_assert!(row < self.seq_len);
        unsafe { (self.layer_out.as_ptr() as *mut f32).add(row * hidden_size) }
    }

    pub fn gate_row_mut_ptr(&mut self, row: usize, ff_size: usize) -> *mut f32 {
        debug_assert!(row < self.seq_len);
        unsafe { (self.gate.as_ptr() as *mut f32).add(row * ff_size) }
    }

    pub fn swiglu_row_mut_ptr(&mut self, row: usize, ff_size: usize) -> *mut f32 {
        debug_assert!(row < self.seq_len);
        unsafe { (self.swiglu.as_ptr() as *mut f32).add(row * ff_size) }
    }
}

#[cfg(test)]
mod scratch_tests {
    use super::*;

    fn make_test_config() -> crate::config::ModelConfig {
        crate::config::ModelConfig {
            num_layers: 2,
            num_kv_heads: 4,
            head_dim: 128,
            max_seq_len: 512,
            hidden_size: 1024,
            num_heads: 8,
            intermediate_size: 2048,
            vocab_size: 32000,
            rms_norm_eps: 1e-5,
            rope_theta: 10000.0,
            rope_neox: false,
            use_attention_bias: false,
            attention_layout: crate::config::AttentionLayout::SplitQkv,
            architecture: "test".to_string(),
            tensor_registry: crate::config::TensorNameRegistry::from_scheme(
                &crate::config::TensorNamingScheme::Gguf,
            ),
        }
    }

    #[test]
    fn new_allocates_all_buffers() {
        let config = make_test_config();
        let scratch = GpuForwardScratch::new(&config);

        // Will fail without GPU, that's expected
        match scratch {
            Ok(s) => {
                // Verify pointers are valid (or empty)
                assert!(!s.q.as_ptr().is_null() || s.q.is_empty());
                assert!(!s.hidden.as_ptr().is_null() || s.hidden.is_empty());
            }
            Err(_) => {
                // Expected when HIP unavailable
            }
        }
    }

    #[test]
    fn prefill_scratch_rejects_zero_seq_len() {
        let config = make_test_config();
        let scratch = GpuPrefillScratch::new(&config, 0);
        assert!(scratch.is_err());
    }
}
