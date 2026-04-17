//! Hybrid GPU decode forward path.
//!
//! This is the smallest end-to-end GPU runner for the current kernel set:
//! most layer math stays on HIP, while unsupported embedding and logits paths
//! may still use the existing CPU implementation.

use super::cache::{GpuForwardScratch, GpuKvCache, GpuPrefillScratch};
use super::device::GpuDevice;
use super::error::{GpuError, GpuResult};
use super::ffi;
use super::graph::{DecodeGraphKey, DecodeGraphScope, HipGraph};
use super::kernels::attention::{
    flash_attn_decode_gqa_from_state_on_stream, flash_attn_decode_gqa_on_stream,
    flash_attn_decode_strided_multi_head_from_state_on_stream,
    flash_attn_decode_strided_multi_head_on_stream, flash_attn_prefill_strided,
    kv_write_rope_from_state_on_stream, kv_write_rope_on_stream,
};
use super::kernels::elementwise::{
    add, add_batched, add_on_stream, argmax_f32, argmax_f32_on_stream, embed_q8_0_batch,
    embed_q8_0_token, silu,
};
use super::kernels::norm::{rms_norm, rms_norm_batched, rms_norm_on_stream};
use super::kernels::rope::{
    rope_heads_batched, rope_heads_from_state_on_stream, rope_heads_on_stream,
};
use super::kernels::q8_decode::q8_0_workspace_bytes;
use super::ops::{
    gpu_dispatch_fused_gate_up_on_stream, gpu_dispatch_fused_norm_gate_up_on_stream,
    gpu_dispatch_fused_norm_qkv_rope_kvwrite_on_stream, gpu_dispatch_fused_qkv_on_stream,
    gpu_dispatch_fused_qkv_rope_kvwrite_on_stream, gpu_dispatch_gemm, gpu_dispatch_gemv,
    gpu_dispatch_gemv_on_stream, gpu_dispatch_gemv_residual_on_stream, gpu_dispatch_rms_norm,
};
use super::weights::{GpuBuffer, GpuLayerWeights, GpuModelWeights, WeightMeta};
use crate::config::ModelConfig;
use crate::cpu::cache::CpuForwardScratch;
use crate::cpu::forward::cpu_embed_token;
use crate::cpu::ops::dispatch_gemv as cpu_dispatch_gemv;
use crate::cpu::weights::CpuModelWeights;
use crate::loader::GgmlType;
use std::sync::{Mutex, OnceLock};
use std::time::Instant;

/// Maximum speculative decoding depth. Determines fixed scratch buffer sizes
/// for batched verify (logits_batch = MAX_VERIFY_BATCH × vocab_size × 4B).
pub const MAX_SPEC_DEPTH: usize = 8;
/// Maximum batch size for verify lm_head (depth + 1 for the input token).
pub const MAX_VERIFY_BATCH: usize = MAX_SPEC_DEPTH + 1;

fn bytes_of_f32_slice(src: &[f32]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(src.as_ptr() as *const u8, std::mem::size_of_val(src)) }
}

fn bytes_of_i32_slice(src: &[i32]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(src.as_ptr() as *const u8, std::mem::size_of_val(src)) }
}

fn bytes_of_f32_slice_mut(dst: &mut [f32]) -> &mut [u8] {
    unsafe {
        std::slice::from_raw_parts_mut(dst.as_mut_ptr() as *mut u8, std::mem::size_of_val(dst))
    }
}

fn upload_f32(dst: &mut GpuBuffer, src: &[f32]) -> GpuResult<()> {
    dst.copy_from_host(bytes_of_f32_slice(src))
}

fn upload_f32_partial(dst: &mut GpuBuffer, src: &[f32]) -> GpuResult<()> {
    dst.copy_from_host_partial(bytes_of_f32_slice(src))
}

fn upload_i32(dst: &mut GpuBuffer, src: &[i32]) -> GpuResult<()> {
    dst.copy_from_host(bytes_of_i32_slice(src))
}

fn upload_i32_partial(dst: &mut GpuBuffer, src: &[i32]) -> GpuResult<()> {
    dst.copy_from_host_partial(bytes_of_i32_slice(src))
}

fn download_f32(src: &GpuBuffer, dst: &mut [f32]) -> GpuResult<()> {
    src.copy_to_host(bytes_of_f32_slice_mut(dst))
}

fn cpu_fallback_error(op: &str, err: impl std::fmt::Display) -> GpuError {
    GpuError::HipApiError {
        code: -1,
        description: format!("{} CPU fallback failed: {}", op, err),
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct GpuDecodeStageProfileSnapshot {
    pub layer_invocations: u64,
    pub tail_invocations: u64,
    pub attn_norm_ns: u128,
    pub qkv_ns: u128,
    pub q_rope_ns: u128,
    pub k_rope_ns: u128,
    pub kv_write_ns: u128,
    pub attention_ns: u128,
    pub attn_proj_ns: u128,
    pub attn_residual_ns: u128,
    pub ffn_norm_ns: u128,
    pub gate_up_ns: u128,
    pub ffn_down_ns: u128,
    pub ffn_residual_ns: u128,
    pub logits_norm_ns: u128,
    pub logits_proj_ns: u128,
    pub argmax_ns: u128,
}

#[derive(Clone, Copy, Debug)]
enum DecodeStage {
    AttnNorm,
    Qkv,
    QRope,
    KRope,
    KvWrite,
    Attention,
    AttnProj,
    AttnResidual,
    FfnNorm,
    GateUp,
    FfnDown,
    FfnResidual,
    LogitsNorm,
    LogitsProj,
    Argmax,
}

fn decode_stage_profile_store() -> &'static Mutex<GpuDecodeStageProfileSnapshot> {
    static STORE: OnceLock<Mutex<GpuDecodeStageProfileSnapshot>> = OnceLock::new();
    STORE.get_or_init(|| Mutex::new(GpuDecodeStageProfileSnapshot::default()))
}

fn decode_stage_profiling_enabled() -> bool {
    std::env::var_os("ROCMFORGE_PROFILE_DECODE_STAGES").is_some()
}

fn decode_graph_disabled() -> bool {
    decode_stage_profiling_enabled() || std::env::var_os("ROCMFORGE_DISABLE_DECODE_GRAPH").is_some()
}

fn record_decode_stage(stage: DecodeStage, elapsed_ns: u128) {
    let mut guard = decode_stage_profile_store()
        .lock()
        .unwrap_or_else(|poison| poison.into_inner());
    match stage {
        DecodeStage::AttnNorm => guard.attn_norm_ns += elapsed_ns,
        DecodeStage::Qkv => guard.qkv_ns += elapsed_ns,
        DecodeStage::QRope => guard.q_rope_ns += elapsed_ns,
        DecodeStage::KRope => guard.k_rope_ns += elapsed_ns,
        DecodeStage::KvWrite => guard.kv_write_ns += elapsed_ns,
        DecodeStage::Attention => guard.attention_ns += elapsed_ns,
        DecodeStage::AttnProj => guard.attn_proj_ns += elapsed_ns,
        DecodeStage::AttnResidual => guard.attn_residual_ns += elapsed_ns,
        DecodeStage::FfnNorm => guard.ffn_norm_ns += elapsed_ns,
        DecodeStage::GateUp => guard.gate_up_ns += elapsed_ns,
        DecodeStage::FfnDown => guard.ffn_down_ns += elapsed_ns,
        DecodeStage::FfnResidual => guard.ffn_residual_ns += elapsed_ns,
        DecodeStage::LogitsNorm => guard.logits_norm_ns += elapsed_ns,
        DecodeStage::LogitsProj => guard.logits_proj_ns += elapsed_ns,
        DecodeStage::Argmax => guard.argmax_ns += elapsed_ns,
    }
}

fn profile_decode_stage<T>(
    device: &GpuDevice,
    stage: DecodeStage,
    op: impl FnOnce() -> GpuResult<T>,
) -> GpuResult<T> {
    if !decode_stage_profiling_enabled() {
        return op();
    }

    let start = Instant::now();
    let result = op()?;
    device.synchronize()?;
    record_decode_stage(stage, start.elapsed().as_nanos());
    Ok(result)
}

pub fn reset_decode_stage_profile() {
    let mut guard = decode_stage_profile_store()
        .lock()
        .unwrap_or_else(|poison| poison.into_inner());
    *guard = GpuDecodeStageProfileSnapshot::default();
}

pub fn decode_stage_profile_snapshot() -> GpuDecodeStageProfileSnapshot {
    *decode_stage_profile_store()
        .lock()
        .unwrap_or_else(|poison| poison.into_inner())
}

fn validate_token_embedding_layout(meta: &WeightMeta, config: &ModelConfig) -> GpuResult<()> {
    if meta.dims.len() < 2 {
        return Err(GpuError::InvalidWeightLayout {
            tensor: "token_emb".to_string(),
            dims: meta.dims.clone(),
            reason: "embedding weights must have at least 2 dimensions".to_string(),
        });
    }

    let hidden_size = meta.dims[0] as usize;
    let vocab_size = meta.dims[1] as usize;
    if hidden_size != config.hidden_size || vocab_size != config.vocab_size {
        return Err(GpuError::InvalidWeightLayout {
            tensor: "token_emb".to_string(),
            dims: meta.dims.clone(),
            reason: format!(
                "expected [{}, {}], got [{}, {}]",
                config.hidden_size, config.vocab_size, hidden_size, vocab_size
            ),
        });
    }
    if meta.needs_transpose {
        return Err(GpuError::InvalidWeightLayout {
            tensor: "token_emb".to_string(),
            dims: meta.dims.clone(),
            reason: "token embeddings must not require transpose for GPU lookup".to_string(),
        });
    }

    Ok(())
}

fn residual_add_inplace(
    device: &GpuDevice,
    hidden: &GpuBuffer,
    residual: &GpuBuffer,
    len: usize,
) -> GpuResult<()> {
    add_on_stream(
        hidden.as_ptr() as *const f32,
        residual.as_ptr() as *const f32,
        hidden.as_ptr() as *mut f32,
        len,
        device.stream(),
    )
}

/// Controls how the hybrid GPU path handles final logits.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GpuLogitsMode {
    /// Skip final norm and output projection entirely.
    Skip,
    /// Download full logits to host memory.
    DownloadToHost,
    /// Keep logits on GPU and return the greedy token via GPU argmax.
    GreedyArgmax,
}

fn cpu_fallback_gemv(
    op: &str,
    weights: &[u8],
    meta: &crate::cpu::weights::WeightMeta,
    input_gpu: &GpuBuffer,
    input_host: &mut [f32],
    output_host: &mut [f32],
    out_dim: usize,
    in_dim: usize,
    q8_scratch: &mut Vec<u8>,
) -> GpuResult<()> {
    download_f32(input_gpu, &mut input_host[..in_dim])?;
    cpu_dispatch_gemv(
        weights,
        meta,
        &input_host[..in_dim],
        &mut output_host[..out_dim],
        out_dim,
        in_dim,
        Some(q8_scratch),
    )
    .map_err(|e| cpu_fallback_error(op, e))
}

fn gpu_launch_greedy_argmax(scratch: &mut GpuForwardScratch, vocab_size: usize) -> GpuResult<()> {
    argmax_f32(
        scratch.logits_ptr(),
        scratch.argmax_partial_values_mut_ptr(),
        scratch.argmax_partial_indices_mut_ptr(),
        scratch.argmax_result_index_mut_ptr(),
        vocab_size,
    )
}

fn gpu_read_greedy_argmax_result(
    device: &GpuDevice,
    scratch: &mut GpuForwardScratch,
    _vocab_size: usize,
) -> GpuResult<()> {
    unsafe {
        ffi::hip_memcpy_d2h_async(
            scratch.argmax_result_index.as_ptr(),
            scratch.argmax_result_device.as_ptr(),
            std::mem::size_of::<i32>(),
            device.stream(),
        )?;
    }

    Ok(())
}

fn gpu_greedy_argmax_token(
    device: &GpuDevice,
    scratch: &mut GpuForwardScratch,
    vocab_size: usize,
) -> GpuResult<u32> {
    gpu_launch_greedy_argmax(scratch, vocab_size)?;
    gpu_read_greedy_argmax_result(device, scratch, vocab_size)?;
    device.synchronize()?;
    let index = scratch.argmax_result_index.as_slice::<i32>()[0];
    Ok(index as u32)
}

fn gpu_launch_greedy_logits_tail_on_stream(
    device: &GpuDevice,
    gpu_weights: &GpuModelWeights,
    scratch: &mut GpuForwardScratch,
    config: &ModelConfig,
) -> GpuResult<()> {
    let h = config.hidden_size;
    let v = config.vocab_size;
    if decode_stage_profiling_enabled() {
        let mut guard = decode_stage_profile_store()
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        guard.tail_invocations += 1;
    }
    profile_decode_stage(device, DecodeStage::LogitsNorm, || {
        gpu_dispatch_rms_norm(
            device,
            scratch.hidden.as_ptr() as *const f32,
            gpu_weights.output_norm.as_ptr() as *const f32,
            scratch.normed.as_ptr() as *mut f32,
            h,
            config.rms_norm_eps,
            device.stream(),
        )
    })?;
    profile_decode_stage(device, DecodeStage::LogitsProj, || {
        gpu_dispatch_gemv_on_stream(
            device,
            &gpu_weights.lm_head,
            &gpu_weights.lm_head_meta,
            scratch.normed.as_ptr() as *const f32,
            scratch.logits.as_ptr() as *mut f32,
            v,
            h,
            device.stream(),
        )
    })?;
    profile_decode_stage(device, DecodeStage::Argmax, || {
        argmax_f32_on_stream(
            scratch.logits_ptr(),
            scratch.argmax_partial_values_mut_ptr(),
            scratch.argmax_partial_indices_mut_ptr(),
            scratch.argmax_result_index_mut_ptr(),
            v,
            device.stream(),
        )
    })
}

fn gpu_greedy_logits_graph_key(
    device: &GpuDevice,
    gpu_weights: &GpuModelWeights,
    config: &ModelConfig,
) -> DecodeGraphKey {
    DecodeGraphKey::from_parts_with_bindings(
        device.device_id(),
        device.warp_size(),
        config,
        GpuLogitsMode::GreedyArgmax,
        gpu_weights.output_norm.as_ptr() as usize,
        gpu_weights.lm_head.as_ptr() as usize,
        gpu_weights.lm_head_meta.wtype,
        gpu_weights.lm_head_meta.role,
    )
    .with_decode_scope(DecodeGraphScope::GreedyTail)
}

fn mix_binding_tag(tag: u64, ptr: usize) -> u64 {
    tag.rotate_left(13) ^ (ptr as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)
}

fn gpu_layer_weights_binding_tag(layer: &GpuLayerWeights) -> u64 {
    let mut tag = 0u64;
    tag = mix_binding_tag(tag, layer.attn_norm.as_ptr() as usize);
    tag = mix_binding_tag(tag, layer.attn_q.as_ptr() as usize);
    tag = mix_binding_tag(
        tag,
        layer
            .attn_q_bias
            .as_ref()
            .map_or(0usize, |buf| buf.as_ptr() as usize),
    );
    tag = mix_binding_tag(tag, layer.attn_k.as_ptr() as usize);
    tag = mix_binding_tag(
        tag,
        layer
            .attn_k_bias
            .as_ref()
            .map_or(0usize, |buf| buf.as_ptr() as usize),
    );
    tag = mix_binding_tag(tag, layer.attn_v.as_ptr() as usize);
    tag = mix_binding_tag(
        tag,
        layer
            .attn_v_bias
            .as_ref()
            .map_or(0usize, |buf| buf.as_ptr() as usize),
    );
    tag = mix_binding_tag(tag, layer.attn_o.as_ptr() as usize);
    tag = mix_binding_tag(tag, layer.ffn_norm.as_ptr() as usize);
    tag = mix_binding_tag(tag, layer.ffn_gate.as_ptr() as usize);
    tag = mix_binding_tag(tag, layer.ffn_up.as_ptr() as usize);
    mix_binding_tag(tag, layer.ffn_down.as_ptr() as usize)
}

fn gpu_model_weights_binding_tag(gpu_weights: &GpuModelWeights) -> u64 {
    let mut tag = 0u64;
    tag = mix_binding_tag(tag, gpu_weights.output_norm.as_ptr() as usize);
    tag = mix_binding_tag(tag, gpu_weights.lm_head.as_ptr() as usize);
    for layer in &gpu_weights.layers {
        tag ^= gpu_layer_weights_binding_tag(layer);
    }
    tag
}

fn gpu_kv_binding_tag(kv: &GpuKvCache) -> GpuResult<u64> {
    let mut tag = 0u64;
    for layer_idx in 0..kv.num_layers {
        tag = mix_binding_tag(tag, kv.k_ptr(layer_idx)? as usize);
        tag = mix_binding_tag(tag, kv.v_ptr(layer_idx)? as usize);
    }
    Ok(tag)
}

fn gpu_full_decode_graph_key(
    device: &GpuDevice,
    gpu_weights: &GpuModelWeights,
    kv: &GpuKvCache,
    config: &ModelConfig,
) -> GpuResult<DecodeGraphKey> {
    Ok(DecodeGraphKey::from_parts_with_bindings(
        device.device_id(),
        device.warp_size(),
        config,
        GpuLogitsMode::GreedyArgmax,
        gpu_weights.output_norm.as_ptr() as usize,
        gpu_weights.lm_head.as_ptr() as usize,
        gpu_weights.lm_head_meta.wtype,
        gpu_weights.lm_head_meta.role,
    )
    .with_decode_scope(DecodeGraphScope::FullGreedyDecode)
    .with_layer_weights_binding_tag(gpu_model_weights_binding_tag(gpu_weights))
    .with_kv_binding_tag(gpu_kv_binding_tag(kv)?))
}

fn gpu_capture_greedy_decode_graph(
    device: &GpuDevice,
    gpu_weights: &GpuModelWeights,
    scratch: &mut GpuForwardScratch,
    config: &ModelConfig,
) -> GpuResult<super::graph::CapturedDecodeGraph> {
    let key = gpu_greedy_logits_graph_key(device, gpu_weights, config);
    // Pre-allocate Q8 workspace before capture — hipMalloc is forbidden during stream capture.
    device.reserve_q8_workspace(q8_0_workspace_bytes(config.hidden_size))?;
    device.begin_capture(ffi::hipStreamCaptureMode::hipStreamCaptureModeGlobal)?;
    let capture_result =
        gpu_launch_greedy_logits_tail_on_stream(device, gpu_weights, scratch, config);
    let end_capture_result = device.end_capture();

    match capture_result {
        Ok(()) => {
            let graph = HipGraph::from_raw(end_capture_result?);
            super::graph::CapturedDecodeGraph::from_captured_graph(graph, key)
        }
        Err(err) => {
            let _ = end_capture_result;
            Err(err)
        }
    }
}

fn gpu_greedy_logits_tail_token(
    device: &GpuDevice,
    gpu_weights: &GpuModelWeights,
    scratch: &mut GpuForwardScratch,
    config: &ModelConfig,
) -> GpuResult<u32> {
    gpu_launch_greedy_logits_tail_on_stream(device, gpu_weights, scratch, config)?;
    gpu_read_greedy_argmax_result(device, scratch, config.vocab_size)?;
    device.synchronize()?;
    let index = scratch.argmax_result_index.as_slice::<i32>()[0];
    Ok(index as u32)
}

fn gpu_try_greedy_decode_graph(
    device: &GpuDevice,
    gpu_weights: &GpuModelWeights,
    scratch: &mut GpuForwardScratch,
    config: &ModelConfig,
) -> GpuResult<u32> {
    if decode_graph_disabled() {
        return gpu_greedy_logits_tail_token(device, gpu_weights, scratch, config);
    }

    let key = gpu_greedy_logits_graph_key(device, gpu_weights, config);
    if !scratch.has_decode_graph_for(key) {
        scratch.clear_decode_graph();
        let capture_status = match device.stream_capture_status() {
            Ok(status) => status,
            Err(_) => return gpu_greedy_logits_tail_token(device, gpu_weights, scratch, config),
        };
        if capture_status != ffi::hipStreamCaptureStatus::hipStreamCaptureStatusNone {
            return gpu_greedy_logits_tail_token(device, gpu_weights, scratch, config);
        }

        match gpu_capture_greedy_decode_graph(device, gpu_weights, scratch, config) {
            Ok(graph) => {
                scratch.replace_decode_graph(graph);
            }
            Err(err @ GpuError::InvalidWeightLayout { .. })
            | Err(err @ GpuError::UnsupportedWeightType { .. }) => return Err(err),
            Err(_) => return gpu_greedy_logits_tail_token(device, gpu_weights, scratch, config),
        }
    }

    // Check if we have position tracking for decode state updates
    let next_pos = scratch.decode_state_next_pos();

    if next_pos.is_some() {
        let has_graph = scratch.decode_graph().is_some();

        if has_graph {
            // CRITICAL FIX: Upload decode state before graph launch
            // The graph captures memory pointers but NOT values like position.
            // We must upload updated decode state before each replay to ensure correctness.
            let pos = next_pos.unwrap();
            scratch.upload_decode_state(pos, pos + 1, device.stream())?;

            // Get graph again after upload
            if let Some(graph) = scratch.decode_graph() {
                if graph.launch(device.stream()).is_ok() {
                    gpu_read_greedy_argmax_result(device, scratch, config.vocab_size)?;
                    device.synchronize()?;
                    let index = scratch.argmax_result_index.as_slice::<i32>()[0];
                    return Ok(index as u32);
                }
            }
            scratch.clear_decode_graph();
        }
    }

    gpu_greedy_logits_tail_token(device, gpu_weights, scratch, config)
}

fn gpu_attention_decode(
    device: &GpuDevice,
    scratch: &mut GpuForwardScratch,
    kv: &GpuKvCache,
    layer_idx: usize,
    pos: usize,
    config: &ModelConfig,
) -> GpuResult<()> {
    let seq_len = pos + 1;
    let head_dim = config.head_dim;
    let scale = 1.0f32 / (head_dim as f32).sqrt();

    let k_cache = kv.k_ptr(layer_idx)? as *const u16;
    let v_cache = kv.v_ptr(layer_idx)? as *const u16;
    let q_base = scratch.q.as_ptr() as *const f32;
    let out_base = scratch.attn_out.as_ptr() as *mut f32;

    flash_attn_decode_strided_multi_head_on_stream(
        out_base,
        q_base,
        k_cache,
        v_cache,
        seq_len,
        config.num_heads,
        config.num_kv_heads,
        head_dim,
        scale,
        device.stream(),
    )
}

fn gpu_attention_decode_from_state(
    device: &GpuDevice,
    scratch: &mut GpuForwardScratch,
    kv: &GpuKvCache,
    layer_idx: usize,
    config: &ModelConfig,
) -> GpuResult<()> {
    let head_dim = config.head_dim;
    let scale = 1.0f32 / (head_dim as f32).sqrt();

    let k_cache = kv.k_ptr(layer_idx)? as *const u16;
    let v_cache = kv.v_ptr(layer_idx)? as *const u16;
    let q_base = scratch.q.as_ptr() as *const f32;
    let out_base = scratch.attn_out.as_ptr() as *mut f32;

    flash_attn_decode_strided_multi_head_from_state_on_stream(
        out_base,
        q_base,
        k_cache,
        v_cache,
        scratch.decode_seq_len_ptr(),
        config.num_heads,
        config.num_kv_heads,
        head_dim,
        scale,
        device.stream(),
    )
}

fn gpu_project_rows(
    device: &GpuDevice,
    weights: &GpuBuffer,
    meta: &WeightMeta,
    input: *const f32,
    output: *mut f32,
    seq_len: usize,
    out_dim: usize,
    in_dim: usize,
) -> GpuResult<()> {
    gpu_dispatch_gemm(
        device, weights, meta, input, output, out_dim, in_dim, seq_len,
    )
}

fn gpu_rms_norm_rows(
    input: *const f32,
    weight: *const f32,
    output: *mut f32,
    seq_len: usize,
    hidden_size: usize,
    eps: f32,
) -> GpuResult<()> {
    rms_norm_batched(input, weight, output, hidden_size, eps, seq_len)
}

fn gpu_rope_rows(
    buffer: *mut f32,
    start_pos: usize,
    seq_len: usize,
    num_heads: usize,
    head_dim: usize,
    theta: f32,
    neox: bool,
) -> GpuResult<()> {
    rope_heads_batched(buffer, start_pos, num_heads, head_dim, theta, seq_len, neox)
}

fn gpu_attention_prefill(
    scratch: &mut GpuPrefillScratch,
    seq_len: usize,
    config: &ModelConfig,
) -> GpuResult<()> {
    let q_size = config.num_heads * config.head_dim;
    let kv_size = config.num_kv_heads * config.head_dim;
    let scale = 1.0f32 / (config.head_dim as f32).sqrt();

    // Phase 3d/3.1/3.2: WMMA GQA + causal attention. Requires head_dim == 128
    // (baked into the kernel); seq_len is zero-padded up to a multiple of 64
    // via the oversized GpuPrefillScratch buffers (min 64 rows). One
    // dispatch replaces the per-head scalar loop; 300-500× faster at
    // Qwen2.5-7B shapes.
    if seq_len >= 1
        && config.head_dim == 128
        && config.num_heads % config.num_kv_heads == 0
        && super::safety::wmma_attention_enabled()
    {
        let padded = seq_len.div_ceil(64) * 64;
        tracing::debug!(
            seq_len,
            padded_seq_len = padded,
            num_q_heads = config.num_heads,
            num_kv_heads = config.num_kv_heads,
            head_dim = config.head_dim,
            gqa_ratio = config.num_heads / config.num_kv_heads,
            causal = true,
            path = "wmma_attention",
            "Attention dispatch: WMMA FlashAttention ({}→{})",
            seq_len,
            padded
        );
        match gpu_attention_prefill_wmma(scratch, seq_len, config, scale) {
            Ok(()) => return Ok(()),
            Err(err) => {
                tracing::warn!(
                    error = %err,
                    "WMMA prefill attention failed, falling back to scalar kernel"
                );
                eprintln!(
                    "[rocmforge] WMMA prefill attention failed ({}), falling back to scalar kernel",
                    err
                );
                // fall through
            }
        }
    } else {
        let reason = if !super::safety::wmma_attention_enabled() {
            "wmma_disabled"
        } else if config.head_dim != 128 {
            "head_dim_not_128"
        } else if config.num_heads % config.num_kv_heads != 0 {
            "gqa_ratio_not_integer"
        } else {
            "seq_len_below_threshold"
        };
        tracing::debug!(
            seq_len,
            num_q_heads = config.num_heads,
            num_kv_heads = config.num_kv_heads,
            head_dim = config.head_dim,
            path = "scalar_attention",
            reason,
            "Attention dispatch: scalar flash_attn_prefill_strided"
        );
    }

    let kv_group = config.num_heads / config.num_kv_heads;
    for head in 0..config.num_heads {
        let kv_head = head / kv_group;
        let q_offset = head * config.head_dim;
        let kv_offset = kv_head * config.head_dim;

        flash_attn_prefill_strided(
            scratch.attn_out.as_ptr() as *mut f32,
            scratch.q.as_ptr() as *const f32,
            scratch.k.as_ptr() as *const f32,
            scratch.v.as_ptr() as *const f32,
            seq_len,
            config.head_dim,
            q_size,
            q_size,
            kv_size,
            q_offset,
            q_offset,
            kv_offset,
            scale,
        )?;
    }

    Ok(())
}

fn gpu_attention_prefill_wmma(
    scratch: &mut GpuPrefillScratch,
    seq_len: usize,
    config: &ModelConfig,
    scale: f32,
) -> GpuResult<()> {
    use super::kernels::wmma::launch_wmma_attention_prefill_gqa_causal;
    use super::prefill_gemm::convert_f32_to_f16_on_stream;

    // Convert only the real rows; the trailing padding rows of the FP16
    // staging buffers stay zero from allocation-time zero-init.
    let q_elems = seq_len * config.num_heads * config.head_dim;
    let kv_elems = seq_len * config.num_kv_heads * config.head_dim;

    scratch.ensure_attention_f16_buffers(
        config.num_heads,
        config.num_kv_heads,
        config.head_dim,
    )?;
    let padded_seq_len = scratch.buffer_seq_len;
    let q_f16 = scratch.q_f16.as_ref().unwrap().as_ptr() as *mut u8;
    let k_f16 = scratch.k_f16.as_ref().unwrap().as_ptr() as *mut u8;
    let v_f16 = scratch.v_f16.as_ref().unwrap().as_ptr() as *mut u8;

    let stream = ffi::hipStream_t::null();
    convert_f32_to_f16_on_stream(
        scratch.q.as_ptr() as *const f32,
        q_f16,
        q_elems,
        stream,
    )?;
    convert_f32_to_f16_on_stream(
        scratch.k.as_ptr() as *const f32,
        k_f16,
        kv_elems,
        stream,
    )?;
    convert_f32_to_f16_on_stream(
        scratch.v.as_ptr() as *const f32,
        v_f16,
        kv_elems,
        stream,
    )?;

    launch_wmma_attention_prefill_gqa_causal(
        q_f16 as *const u16,
        k_f16 as *const u16,
        v_f16 as *const u16,
        scratch.attn_out.as_ptr() as *mut f32,
        padded_seq_len,
        config.num_heads,
        config.num_kv_heads,
        true,
        scale,
        stream,
    )
}

fn gpu_embed_tokens_hybrid(
    tokens: &[u32],
    gpu_weights: &GpuModelWeights,
    cpu_weights: &CpuModelWeights,
    scratch: &mut GpuPrefillScratch,
    config: &ModelConfig,
) -> GpuResult<()> {
    let h = config.hidden_size;
    match gpu_weights.token_emb_meta.wtype {
        GgmlType::Q8_0 => {
            validate_token_embedding_layout(&gpu_weights.token_emb_meta, config)?;
            let token_ids: Vec<i32> = tokens.iter().map(|&token| token as i32).collect();
            if token_ids
                .iter()
                .any(|&token| token < 0 || token as usize >= config.vocab_size)
            {
                return Err(GpuError::HipApiError {
                    code: -1,
                    description: "prefill token id out of vocab range".to_string(),
                });
            }
            upload_i32_partial(&mut scratch.token_ids, &token_ids)?;
            embed_q8_0_batch(
                gpu_weights.token_emb.as_ptr(),
                scratch.token_ids.as_ptr() as *const i32,
                scratch.hidden.as_ptr() as *mut f32,
                h,
                config.vocab_size,
                tokens.len(),
            )
        }
        _ => {
            let mut host_hidden = vec![0.0f32; tokens.len() * h];
            for (row, &token_id) in tokens.iter().enumerate() {
                cpu_embed_token(
                    token_id,
                    cpu_weights,
                    &mut host_hidden[row * h..(row + 1) * h],
                    config,
                );
            }
            upload_f32_partial(&mut scratch.hidden, &host_hidden)
        }
    }
}

/// Hybrid single-layer prefill step over a full prompt batch.
pub fn gpu_prefill_layer_forward_hybrid(
    device: &GpuDevice,
    gpu_layer: &GpuLayerWeights,
    kv: &mut GpuKvCache,
    scratch: &mut GpuPrefillScratch,
    layer_idx: usize,
    start_pos: usize,
    config: &ModelConfig,
) -> GpuResult<()> {
    let seq_len = scratch.seq_len;
    let h = config.hidden_size;
    let q_size = config.num_heads * config.head_dim;
    let kv_size = config.num_kv_heads * config.head_dim;
    let ff_size = config.intermediate_size;
    let eps = config.rms_norm_eps;

    gpu_rms_norm_rows(
        scratch.hidden.as_ptr() as *const f32,
        gpu_layer.attn_norm.as_ptr() as *const f32,
        scratch.normed.as_ptr() as *mut f32,
        seq_len,
        h,
        eps,
    )?;

    gpu_project_rows(
        device,
        &gpu_layer.attn_q,
        &gpu_layer.attn_q_meta,
        scratch.normed.as_ptr() as *const f32,
        scratch.q.as_ptr() as *mut f32,
        seq_len,
        q_size,
        h,
    )?;
    gpu_project_rows(
        device,
        &gpu_layer.attn_k,
        &gpu_layer.attn_k_meta,
        scratch.normed.as_ptr() as *const f32,
        scratch.k.as_ptr() as *mut f32,
        seq_len,
        kv_size,
        h,
    )?;
    gpu_project_rows(
        device,
        &gpu_layer.attn_v,
        &gpu_layer.attn_v_meta,
        scratch.normed.as_ptr() as *const f32,
        scratch.v.as_ptr() as *mut f32,
        seq_len,
        kv_size,
        h,
    )?;

    if let Some(bq) = &gpu_layer.attn_q_bias {
        add_batched(
            scratch.q.as_ptr() as *const f32,
            bq.as_ptr() as *const f32,
            scratch.q.as_ptr() as *mut f32,
            q_size,
            seq_len,
        )?;
    }
    if let Some(bk) = &gpu_layer.attn_k_bias {
        add_batched(
            scratch.k.as_ptr() as *const f32,
            bk.as_ptr() as *const f32,
            scratch.k.as_ptr() as *mut f32,
            kv_size,
            seq_len,
        )?;
    }
    if let Some(bv) = &gpu_layer.attn_v_bias {
        add_batched(
            scratch.v.as_ptr() as *const f32,
            bv.as_ptr() as *const f32,
            scratch.v.as_ptr() as *mut f32,
            kv_size,
            seq_len,
        )?;
    }

    gpu_rope_rows(
        scratch.q.as_ptr() as *mut f32,
        start_pos,
        seq_len,
        config.num_heads,
        config.head_dim,
        config.rope_theta,
        config.rope_neox,
    )?;
    gpu_rope_rows(
        scratch.k.as_ptr() as *mut f32,
        start_pos,
        seq_len,
        config.num_kv_heads,
        config.head_dim,
        config.rope_theta,
        config.rope_neox,
    )?;

    kv.write_batched(
        layer_idx,
        start_pos,
        seq_len,
        scratch.k.as_ptr() as *const f32,
        scratch.v.as_ptr() as *const f32,
    )?;

    gpu_attention_prefill(scratch, seq_len, config)?;

    gpu_project_rows(
        device,
        &gpu_layer.attn_o,
        &gpu_layer.attn_o_meta,
        scratch.attn_out.as_ptr() as *const f32,
        scratch.layer_out.as_ptr() as *mut f32,
        seq_len,
        h,
        q_size,
    )?;
    add(
        scratch.hidden.as_ptr() as *const f32,
        scratch.layer_out.as_ptr() as *const f32,
        scratch.hidden.as_ptr() as *mut f32,
        seq_len * h,
    )?;

    gpu_rms_norm_rows(
        scratch.hidden.as_ptr() as *const f32,
        gpu_layer.ffn_norm.as_ptr() as *const f32,
        scratch.normed.as_ptr() as *mut f32,
        seq_len,
        h,
        eps,
    )?;
    gpu_project_rows(
        device,
        &gpu_layer.ffn_gate,
        &gpu_layer.ffn_gate_meta,
        scratch.normed.as_ptr() as *const f32,
        scratch.gate.as_ptr() as *mut f32,
        seq_len,
        ff_size,
        h,
    )?;
    gpu_project_rows(
        device,
        &gpu_layer.ffn_up,
        &gpu_layer.ffn_up_meta,
        scratch.normed.as_ptr() as *const f32,
        scratch.swiglu.as_ptr() as *mut f32,
        seq_len,
        ff_size,
        h,
    )?;

    silu(
        scratch.gate.as_ptr() as *const f32,
        scratch.gate.as_ptr() as *mut f32,
        seq_len * ff_size,
    )?;
    super::kernels::mul(
        scratch.gate.as_ptr() as *const f32,
        scratch.swiglu.as_ptr() as *const f32,
        scratch.swiglu.as_ptr() as *mut f32,
        seq_len * ff_size,
    )?;
    gpu_project_rows(
        device,
        &gpu_layer.ffn_down,
        &gpu_layer.ffn_down_meta,
        scratch.swiglu.as_ptr() as *const f32,
        scratch.layer_out.as_ptr() as *mut f32,
        seq_len,
        h,
        ff_size,
    )?;
    add(
        scratch.hidden.as_ptr() as *const f32,
        scratch.layer_out.as_ptr() as *const f32,
        scratch.hidden.as_ptr() as *mut f32,
        seq_len * h,
    )?;

    Ok(())
}

/// Full batched prompt prefill using GPU kernels plus targeted CPU fallbacks.
pub fn gpu_prefill_forward_hybrid(
    device: &GpuDevice,
    gpu_weights: &GpuModelWeights,
    cpu_weights: &CpuModelWeights,
    kv: &mut GpuKvCache,
    prefill: &mut GpuPrefillScratch,
    decode_scratch: &mut GpuForwardScratch,
    host_scratch: &mut CpuForwardScratch,
    tokens: &[u32],
    start_pos: usize,
    config: &ModelConfig,
    logits_mode: GpuLogitsMode,
) -> GpuResult<Option<u32>> {
    if tokens.is_empty() {
        return Err(GpuError::HipApiError {
            code: -1,
            description: "gpu_prefill_forward_hybrid: empty prompt".to_string(),
        });
    }
    if start_pos != 0 {
        return Err(GpuError::InvalidWeightLayout {
            tensor: "gpu_prefill_forward_hybrid".to_string(),
            dims: vec![],
            reason: "batched GPU prefill currently supports start_pos == 0 only".to_string(),
        });
    }
    if start_pos + tokens.len() > kv.max_seq_len {
        return Err(GpuError::HipApiError {
            code: -1,
            description: format!(
                "gpu_prefill_forward_hybrid: prompt length {} exceeds kv cache {}",
                start_pos + tokens.len(),
                kv.max_seq_len
            ),
        });
    }
    if prefill.seq_len != tokens.len() {
        return Err(GpuError::HipApiError {
            code: -1,
            description: format!(
                "gpu_prefill_forward_hybrid: scratch seq_len {} does not match tokens {}",
                prefill.seq_len,
                tokens.len()
            ),
        });
    }

    tracing::debug!(
        start_pos,
        seq_len = tokens.len(),
        buffer_seq_len = prefill.buffer_seq_len,
        kv_max_seq_len = kv.max_seq_len,
        num_layers = config.num_layers,
        "KV-Cache update: writing positions {}..{}",
        start_pos,
        start_pos + tokens.len()
    );

    gpu_embed_tokens_hybrid(tokens, gpu_weights, cpu_weights, prefill, config)?;

    for layer_idx in 0..config.num_layers {
        let layer_start = std::time::Instant::now();
        gpu_prefill_layer_forward_hybrid(
            device,
            gpu_weights.layer(layer_idx),
            kv,
            prefill,
            layer_idx,
            start_pos,
            config,
        )?;
        tracing::trace!(
            layer = layer_idx,
            seq_len = tokens.len(),
            launch_us = layer_start.elapsed().as_micros() as u64,
            "Prefill layer launched"
        );
    }

    if matches!(logits_mode, GpuLogitsMode::Skip) {
        return Ok(None);
    }

    let h = config.hidden_size;
    let v = config.vocab_size;
    let last_hidden =
        unsafe { (prefill.hidden.as_ptr() as *const f32).add((tokens.len() - 1) * h) };
    rms_norm(
        last_hidden,
        gpu_weights.output_norm.as_ptr() as *const f32,
        decode_scratch.normed.as_ptr() as *mut f32,
        h,
        config.rms_norm_eps,
    )?;

    match gpu_dispatch_gemv(
        device,
        &gpu_weights.lm_head,
        &gpu_weights.lm_head_meta,
        decode_scratch.normed.as_ptr() as *const f32,
        decode_scratch.logits.as_ptr() as *mut f32,
        v,
        h,
    ) {
        Ok(()) => match logits_mode {
            GpuLogitsMode::DownloadToHost => {
                download_f32(&decode_scratch.logits, &mut host_scratch.logits[..v])?;
                Ok(None)
            }
            GpuLogitsMode::GreedyArgmax => {
                gpu_greedy_argmax_token(device, decode_scratch, v).map(Some)
            }
            GpuLogitsMode::Skip => Ok(None),
        },
        Err(GpuError::InvalidWeightLayout { .. }) | Err(GpuError::UnsupportedWeightType { .. }) => {
            cpu_fallback_gemv(
                "lm_head",
                &cpu_weights.lm_head,
                &cpu_weights.lm_head_meta,
                &decode_scratch.normed,
                &mut host_scratch.normed,
                &mut host_scratch.logits,
                v,
                h,
                &mut host_scratch.q8_scratch,
            )?;
            match logits_mode {
                GpuLogitsMode::DownloadToHost => Ok(None),
                GpuLogitsMode::GreedyArgmax => Ok(Some(crate::cpu::sampler::cpu_sample_greedy(
                    &host_scratch.logits[..v],
                ))),
                GpuLogitsMode::Skip => Ok(None),
            }
        }
        Err(err) => Err(err),
    }
}

// ── Speculative Decoding: Verify Forward Pass ───────────────────────────────────

fn gpu_attention_verify(
    scratch: &mut GpuPrefillScratch,
    kv: &GpuKvCache,
    layer_idx: usize,
    start_pos: usize,
    config: &ModelConfig,
) -> GpuResult<()> {
    use super::kernels::attention::flash_attn_verify_all_heads_on_stream;

    let seq_len = scratch.seq_len;
    let scale = 1.0f32 / (config.head_dim as f32).sqrt();

    let k_cache = kv.k_ptr(layer_idx)? as *const u16;
    let v_cache = kv.v_ptr(layer_idx)? as *const u16;

    // Single dispatch: all heads × all verify positions in one kernel launch.
    // Grid: (num_kv_heads, n_verify). Previously: num_heads × n_verify launches.
    flash_attn_verify_all_heads_on_stream(
        scratch.attn_out.as_ptr() as *mut f32,
        scratch.q.as_ptr() as *const f32,
        k_cache,
        v_cache,
        seq_len,
        start_pos,
        config.num_heads,
        config.num_kv_heads,
        config.head_dim,
        scale,
        ffi::hipStream_t::null(),
    )
}

/// Verify layer forward: like prefill but reads K/V from FP16 KV cache for attention.
pub fn gpu_verify_layer_forward(
    device: &GpuDevice,
    gpu_layer: &GpuLayerWeights,
    kv: &mut GpuKvCache,
    scratch: &mut GpuPrefillScratch,
    layer_idx: usize,
    start_pos: usize,
    config: &ModelConfig,
) -> GpuResult<()> {
    use super::spec_step_profile::VerifyLayerTimer;

    let seq_len = scratch.seq_len;
    let h = config.hidden_size;
    let q_size = config.num_heads * config.head_dim;
    let kv_size = config.num_kv_heads * config.head_dim;
    let ff_size = config.intermediate_size;
    let eps = config.rms_norm_eps;

    let mut vt = VerifyLayerTimer::begin()?;

    // Attention: norm → QKV → RoPE → KV write → verify attention → O-proj → residual
    gpu_rms_norm_rows(
        scratch.hidden.as_ptr() as *const f32,
        gpu_layer.attn_norm.as_ptr() as *const f32,
        scratch.normed.as_ptr() as *mut f32,
        seq_len, h, eps,
    )?;
    if let Some(ref mut t) = vt { t.mark()?; } // → attn_qkv

    gpu_project_rows(device, &gpu_layer.attn_q, &gpu_layer.attn_q_meta,
        scratch.normed.as_ptr() as *const f32, scratch.q.as_ptr() as *mut f32,
        seq_len, q_size, h)?;
    gpu_project_rows(device, &gpu_layer.attn_k, &gpu_layer.attn_k_meta,
        scratch.normed.as_ptr() as *const f32, scratch.k.as_ptr() as *mut f32,
        seq_len, kv_size, h)?;
    gpu_project_rows(device, &gpu_layer.attn_v, &gpu_layer.attn_v_meta,
        scratch.normed.as_ptr() as *const f32, scratch.v.as_ptr() as *mut f32,
        seq_len, kv_size, h)?;

    if let Some(bq) = &gpu_layer.attn_q_bias {
        add_batched(scratch.q.as_ptr() as *const f32, bq.as_ptr() as *const f32,
            scratch.q.as_ptr() as *mut f32, q_size, seq_len)?;
    }
    if let Some(bk) = &gpu_layer.attn_k_bias {
        add_batched(scratch.k.as_ptr() as *const f32, bk.as_ptr() as *const f32,
            scratch.k.as_ptr() as *mut f32, kv_size, seq_len)?;
    }
    if let Some(bv) = &gpu_layer.attn_v_bias {
        add_batched(scratch.v.as_ptr() as *const f32, bv.as_ptr() as *const f32,
            scratch.v.as_ptr() as *mut f32, kv_size, seq_len)?;
    }
    if let Some(ref mut t) = vt { t.mark()?; } // → attn_rope

    gpu_rope_rows(scratch.q.as_ptr() as *mut f32, start_pos, seq_len,
        config.num_heads, config.head_dim, config.rope_theta, config.rope_neox)?;
    gpu_rope_rows(scratch.k.as_ptr() as *mut f32, start_pos, seq_len,
        config.num_kv_heads, config.head_dim, config.rope_theta, config.rope_neox)?;
    if let Some(ref mut t) = vt { t.mark()?; } // → attn_kv_write

    // Write new K/V to cache at start_pos..start_pos+seq_len
    kv.write_batched(layer_idx, start_pos, seq_len,
        scratch.k.as_ptr() as *const f32, scratch.v.as_ptr() as *const f32)?;
    if let Some(ref mut t) = vt { t.mark()?; } // → attn_scores

    // Verify attention: read ALL K/V from FP16 cache (positions 0..start_pos+i per query)
    gpu_attention_verify(scratch, kv, layer_idx, start_pos, config)?;
    if let Some(ref mut t) = vt { t.mark()?; } // → attn_o_residual

    // O-projection + residual
    gpu_project_rows(device, &gpu_layer.attn_o, &gpu_layer.attn_o_meta,
        scratch.attn_out.as_ptr() as *const f32, scratch.layer_out.as_ptr() as *mut f32,
        seq_len, h, q_size)?;
    add(scratch.hidden.as_ptr() as *const f32, scratch.layer_out.as_ptr() as *const f32,
        scratch.hidden.as_ptr() as *mut f32, seq_len * h)?;
    if let Some(ref mut t) = vt { t.mark()?; } // → ffn_norm

    // FFN: norm → gate → up → SiLU → mul → down → residual
    gpu_rms_norm_rows(
        scratch.hidden.as_ptr() as *const f32,
        gpu_layer.ffn_norm.as_ptr() as *const f32,
        scratch.normed.as_ptr() as *mut f32,
        seq_len, h, eps,
    )?;
    if let Some(ref mut t) = vt { t.mark()?; } // → ffn_gate_up_silu_mul

    gpu_project_rows(device, &gpu_layer.ffn_gate, &gpu_layer.ffn_gate_meta,
        scratch.normed.as_ptr() as *const f32, scratch.gate.as_ptr() as *mut f32,
        seq_len, ff_size, h)?;
    gpu_project_rows(device, &gpu_layer.ffn_up, &gpu_layer.ffn_up_meta,
        scratch.normed.as_ptr() as *const f32, scratch.swiglu.as_ptr() as *mut f32,
        seq_len, ff_size, h)?;
    silu(scratch.gate.as_ptr() as *const f32, scratch.gate.as_ptr() as *mut f32,
        seq_len * ff_size)?;
    super::kernels::mul(scratch.gate.as_ptr() as *const f32,
        scratch.swiglu.as_ptr() as *const f32, scratch.swiglu.as_ptr() as *mut f32,
        seq_len * ff_size)?;
    if let Some(ref mut t) = vt { t.mark()?; } // → ffn_down_residual

    gpu_project_rows(device, &gpu_layer.ffn_down, &gpu_layer.ffn_down_meta,
        scratch.swiglu.as_ptr() as *const f32, scratch.layer_out.as_ptr() as *mut f32,
        seq_len, h, ff_size)?;
    add(scratch.hidden.as_ptr() as *const f32, scratch.layer_out.as_ptr() as *const f32,
        scratch.hidden.as_ptr() as *mut f32, seq_len * h)?;
    if let Some(ref mut t) = vt { t.mark()?; } // end

    if let Some(t) = vt { t.finish()?; }

    Ok(())
}

/// Batched final norm + lm_head + argmax for verify.
///
/// Replaces the sequential per-position loop with:
/// 1. Single batched RMS norm (all N positions)
/// 2. Single batched GEMV (all N positions → N × vocab logits)
/// 3. N argmax reductions (reusing partial scratch)
/// 4. Single D2H copy + sync
fn gpu_verify_lm_head_batched(
    device: &GpuDevice,
    gpu_weights: &GpuModelWeights,
    prefill: &GpuPrefillScratch,
    decode_scratch: &mut GpuForwardScratch,
    config: &ModelConfig,
    n: usize,
) -> GpuResult<Vec<u32>> {
    let h = config.hidden_size;
    let v = config.vocab_size;

    // Step 1: Batched RMS norm — all N positions at once
    // Input: prefill.hidden (n × h), Output: prefill.normed (n × h)
    rms_norm_batched(
        prefill.hidden.as_ptr() as *const f32,
        gpu_weights.output_norm.as_ptr() as *const f32,
        prefill.normed.as_ptr() as *mut f32,
        h,
        config.rms_norm_eps,
        n,
    )?;

    // Step 2: Batched GEMV — all N positions at once
    // Input: prefill.normed (n × h), Output: logits_batch (n × v)
    gpu_dispatch_gemm(
        device,
        &gpu_weights.lm_head,
        &gpu_weights.lm_head_meta,
        prefill.normed.as_ptr() as *const f32,
        decode_scratch.logits_batch.as_ptr() as *mut f32,
        v,
        h,
        n,
    )?;

    // Step 3: Argmax per position (N × 2 dispatches, no sync between)
    for i in 0..n {
        let logits_row = unsafe {
            (decode_scratch.logits_batch.as_ptr() as *const f32).add(i * v)
        };
        let result_slot = unsafe {
            (decode_scratch.argmax_batch_device.as_ptr() as *mut i32).add(i)
        };
        argmax_f32(
            logits_row,
            decode_scratch.argmax_partial_values_mut_ptr(),
            decode_scratch.argmax_partial_indices_mut_ptr(),
            result_slot,
            v,
        )?;
    }

    // Step 4: Single D2H copy of all N result indices + single sync
    unsafe {
        ffi::hip_memcpy_d2h_async(
            decode_scratch.argmax_batch_host.as_ptr(),
            decode_scratch.argmax_batch_device.as_ptr(),
            n * std::mem::size_of::<i32>(),
            device.stream(),
        )?;
    }
    device.synchronize()?;

    // Step 5: Read results from pinned host buffer
    let results = decode_scratch.argmax_batch_host.as_slice::<i32>();
    Ok((0..n).map(|i| results[i] as u32).collect())
}

/// Verify forward pass for speculative decoding.
///
/// Runs `tokens` through the model starting at `start_pos`, writing to KV cache,
/// and returns the argmax token for EACH of the N positions (not just the last).
pub fn gpu_verify_forward(
    device: &GpuDevice,
    gpu_weights: &GpuModelWeights,
    cpu_weights: &CpuModelWeights,
    kv: &mut GpuKvCache,
    prefill: &mut GpuPrefillScratch,
    decode_scratch: &mut GpuForwardScratch,
    host_scratch: &mut CpuForwardScratch,
    tokens: &[u32],
    start_pos: usize,
    config: &ModelConfig,
) -> GpuResult<Vec<u32>> {
    if tokens.is_empty() {
        return Err(GpuError::HipApiError {
            code: -1,
            description: "gpu_verify_forward: empty token list".to_string(),
        });
    }
    if start_pos + tokens.len() > kv.max_seq_len {
        return Err(GpuError::HipApiError {
            code: -1,
            description: format!(
                "gpu_verify_forward: start_pos {} + tokens {} exceeds kv cache {}",
                start_pos, tokens.len(), kv.max_seq_len
            ),
        });
    }
    if tokens.len() > prefill.seq_len {
        return Err(GpuError::HipApiError {
            code: -1,
            description: format!(
                "gpu_verify_forward: tokens {} exceeds scratch capacity {}",
                tokens.len(), prefill.seq_len
            ),
        });
    }
    // Temporarily adjust seq_len for this call (buffers are large enough)
    let saved_seq_len = prefill.seq_len;
    prefill.seq_len = tokens.len();

    // Mark verify step for sub-phase breakdown profiling
    super::spec_step_profile::verify_breakdown_mark_step();

    // Embed tokens into prefill scratch
    gpu_embed_tokens_hybrid(tokens, gpu_weights, cpu_weights, prefill, config)?;

    // Layer loop
    for layer_idx in 0..config.num_layers {
        gpu_verify_layer_forward(
            device,
            gpu_weights.layer(layer_idx),
            kv,
            prefill,
            layer_idx,
            start_pos,
            config,
        )?;
    }

    // Final norm + lm_head for ALL N positions
    let h = config.hidden_size;
    let v = config.vocab_size;
    let n = tokens.len();

    // Try batched path if enabled and n fits the fixed scratch.
    if super::safety::batched_lm_head_enabled() && n <= MAX_VERIFY_BATCH {
        match gpu_verify_lm_head_batched(device, gpu_weights, prefill, decode_scratch, config, n) {
            Ok(result_tokens) => {
                prefill.seq_len = saved_seq_len;
                return Ok(result_tokens);
            }
            Err(GpuError::InvalidWeightLayout { .. })
            | Err(GpuError::UnsupportedWeightType { .. }) => {
                if std::env::var_os("ROCMFORGE_SPEC_DEBUG").is_some() {
                    eprintln!(
                        "[SPEC] batched lm_head: GPU weight type unsupported, falling back to sequential"
                    );
                }
                // Fall through to sequential path below
            }
            Err(err) => {
                prefill.seq_len = saved_seq_len;
                return Err(err);
            }
        }
    }

    // Sequential fallback (default when flag disabled, or on unsupported weight types)
    let mut result_tokens = Vec::with_capacity(n);
    for i in 0..n {
        let hidden_row = unsafe { (prefill.hidden.as_ptr() as *const f32).add(i * h) };
        rms_norm(
            hidden_row,
            gpu_weights.output_norm.as_ptr() as *const f32,
            decode_scratch.normed.as_ptr() as *mut f32,
            h,
            config.rms_norm_eps,
        )?;

        match gpu_dispatch_gemv(
            device,
            &gpu_weights.lm_head,
            &gpu_weights.lm_head_meta,
            decode_scratch.normed.as_ptr() as *const f32,
            decode_scratch.logits.as_ptr() as *mut f32,
            v,
            h,
        ) {
            Ok(()) => {
                let token = gpu_greedy_argmax_token(device, decode_scratch, v)?;
                result_tokens.push(token);
            }
            Err(GpuError::InvalidWeightLayout { .. })
            | Err(GpuError::UnsupportedWeightType { .. }) => {
                cpu_fallback_gemv(
                    "lm_head",
                    &cpu_weights.lm_head,
                    &cpu_weights.lm_head_meta,
                    &decode_scratch.normed,
                    &mut host_scratch.normed,
                    &mut host_scratch.logits,
                    v,
                    h,
                    &mut host_scratch.q8_scratch,
                )?;
                let token = crate::cpu::sampler::cpu_sample_greedy(&host_scratch.logits[..v]);
                result_tokens.push(token as u32);
            }
            Err(err) => {
                prefill.seq_len = saved_seq_len;
                return Err(err);
            }
        }
    }

    prefill.seq_len = saved_seq_len;
    Ok(result_tokens)
}

// ── Speculative Decoding: Draft→Verify→Accept ───────────────────────────────────

/// Result of one speculative decode step.
pub struct SpecDecodeResult {
    /// Accepted tokens (1..=spec_depth+1). Always at least 1 (the bonus token).
    pub accepted_tokens: Vec<u32>,
    /// How many draft tokens were accepted (0..=spec_depth).
    pub n_draft_accepted: usize,
    /// Total draft tokens generated this step.
    pub n_drafted: usize,
    /// True if generation should stop (EOS encountered).
    pub hit_eog: bool,
}

/// Run one speculative decode step: draft N tokens with draft model, verify with target.
///
/// Returns accepted tokens (greedy verification). The last token is always the target
/// model's prediction (bonus token), guaranteeing output is identical to standard greedy decode.
pub fn gpu_speculative_decode_step(
    device: &GpuDevice,
    // Draft model resources
    draft_weights: &GpuModelWeights,
    draft_cpu_weights: &CpuModelWeights,
    draft_kv: &mut GpuKvCache,
    draft_scratch: &mut GpuForwardScratch,
    draft_host_scratch: &mut CpuForwardScratch,
    draft_config: &ModelConfig,
    // Target model resources
    target_weights: &GpuModelWeights,
    target_cpu_weights: &CpuModelWeights,
    target_kv: &mut GpuKvCache,
    _verify_scratch: &mut GpuPrefillScratch,
    target_scratch: &mut GpuForwardScratch,
    target_host_scratch: &mut CpuForwardScratch,
    target_config: &ModelConfig,
    // Input
    input_token: u32,
    draft_pos: usize,
    target_pos: usize,
    spec_depth: usize,
    eog_ids: &[u32],
) -> GpuResult<SpecDecodeResult> {
    use super::spec_step_profile::SpecStepTimer;

    // ── Profiling: begin step timer (no-op if flag disabled) ────────────────
    let mut timer = SpecStepTimer::begin(device)?;

    // ── Step 1: Draft N tokens with the small model ─────────────────────────
    // Stop early if the draft model produces an EOS token.
    let mut draft_tokens = Vec::with_capacity(spec_depth);
    let mut token = input_token;
    let mut draft_hit_eog = false;

    for step in 0..spec_depth {
        let pos = draft_pos + step;
        gpu_embed_token_hybrid(
            device, token, draft_weights, draft_cpu_weights,
            draft_scratch, draft_host_scratch, draft_config,
        )?;
        let next = gpu_full_forward_hybrid(
            device, draft_weights, draft_cpu_weights,
            draft_kv, draft_scratch, draft_host_scratch,
            pos, draft_config, GpuLogitsMode::GreedyArgmax,
        )?;
        token = next.expect("draft decode should produce a token");
        draft_tokens.push(token);
        if eog_ids.contains(&token) {
            draft_hit_eog = true;
            break;
        }
    }

    // ── Profiling: mark end of draft phase ──────────────────────────────────
    if let Some(ref mut t) = timer {
        t.mark_draft_end()?;
    }

    let n_drafted = draft_tokens.len();
    let actual_depth = n_drafted;

    // ── Step 2: Batched verify through target model ────────────────────────
    // Process [input_token, draft[0], ..., draft[N-1]] in one batched forward pass.
    // Uses batched GEMV (single weight load, N dot products) for Q4_0 projections
    // and all-heads verify attention (single dispatch per layer) for attention.
    let mut verify_tokens = Vec::with_capacity(actual_depth + 1);
    verify_tokens.push(input_token);
    verify_tokens.extend_from_slice(&draft_tokens);

    let target_argmax = gpu_verify_forward(
        device, target_weights, target_cpu_weights,
        target_kv, _verify_scratch, target_scratch, target_host_scratch,
        &verify_tokens, target_pos, target_config,
    )?;

    // ── Profiling: mark end of verify phase ─────────────────────────────────
    if let Some(ref mut t) = timer {
        t.mark_verify_end()?;
        t.host_phase_begin();
    }

    // ── Step 3: Accept/reject ───────────────────────────────────────────────
    // target_argmax[i] = target's prediction after processing verify_tokens[i]
    //                   = prediction for position target_pos + i + 1
    // draft_tokens[i]  = draft's prediction for position draft_pos + i + 1
    // Compare target_argmax[i] with draft_tokens[i] for i in 0..actual_depth.
    if std::env::var_os("ROCMFORGE_SPEC_DEBUG").is_some() {
        eprintln!(
            "[SPEC] draft={:?} target={:?} draft_eog={}",
            draft_tokens, &target_argmax, draft_hit_eog
        );
    }
    let mut accepted = Vec::with_capacity(actual_depth + 1);
    let mut n_accepted = 0usize;
    let mut hit_eog = false;

    for i in 0..actual_depth {
        if target_argmax[i] == draft_tokens[i] {
            accepted.push(draft_tokens[i]);
            n_accepted += 1;
            // If the accepted token is EOS, stop — don't continue comparing
            if eog_ids.contains(&draft_tokens[i]) {
                hit_eog = true;
                break;
            }
        } else {
            // Mismatch: accept target's correction, stop.
            // Fix draft KV cache: position draft_pos+i has K/V for the rejected
            // draft token. Overwrite with the correction token's K/V so the draft
            // cache stays valid for the next round.
            let correction_pos = draft_pos + i;
            gpu_embed_token_hybrid(
                device, target_argmax[i], draft_weights, draft_cpu_weights,
                draft_scratch, draft_host_scratch, draft_config,
            )?;
            let _ = gpu_full_forward_hybrid(
                device, draft_weights, draft_cpu_weights,
                draft_kv, draft_scratch, draft_host_scratch,
                correction_pos, draft_config, GpuLogitsMode::Skip,
            )?;
            accepted.push(target_argmax[i]);
            if eog_ids.contains(&target_argmax[i]) {
                hit_eog = true;
            }
            break;
        }
    }

    // If all draft tokens matched (and we didn't hit EOS), add bonus token.
    // Also "catch up" the draft KV cache: the draft only wrote positions
    // draft_pos..draft_pos+actual_depth-1, but accepting the bonus means we advance
    // by actual_depth+1. We must fill draft position draft_pos+actual_depth by running
    // the draft on draft_tokens[actual_depth-1] so the KV cache stays contiguous.
    if n_accepted == actual_depth && !hit_eog {
        let bonus = target_argmax[actual_depth];
        accepted.push(bonus);
        if eog_ids.contains(&bonus) {
            hit_eog = true;
        }
        // Fill draft KV cache gap
        let catchup_pos = draft_pos + actual_depth;
        gpu_embed_token_hybrid(
            device, draft_tokens[actual_depth - 1], draft_weights, draft_cpu_weights,
            draft_scratch, draft_host_scratch, draft_config,
        )?;
        let _ = gpu_full_forward_hybrid(
            device, draft_weights, draft_cpu_weights,
            draft_kv, draft_scratch, draft_host_scratch,
            catchup_pos, draft_config, GpuLogitsMode::Skip,
        )?;
    }

    // ── Profiling: record accept/reject time and finalize ───────────────────
    if let Some(mut t) = timer {
        t.host_phase_end_accept_reject();
        t.finish()?;
    }

    let alpha_pct = if n_drafted > 0 {
        n_accepted as f64 / n_drafted as f64 * 100.0
    } else {
        0.0
    };
    tracing::debug!(
        drafted = n_drafted,
        accepted = n_accepted,
        alpha_pct = format!("{:.1}", alpha_pct),
        bonus = n_accepted == actual_depth && !hit_eog,
        hit_eog,
        "Spec-decode verify: {}/{} accepted ({:.1}%)",
        n_accepted,
        n_drafted,
        alpha_pct
    );

    Ok(SpecDecodeResult {
        accepted_tokens: accepted,
        n_draft_accepted: n_accepted,
        n_drafted,
        hit_eog,
    })
}

fn gpu_layer_forward_from_state_on_stream(
    device: &GpuDevice,
    gpu_layer: &GpuLayerWeights,
    kv: &mut GpuKvCache,
    scratch: &mut GpuForwardScratch,
    layer_idx: usize,
    config: &ModelConfig,
) -> GpuResult<()> {
    let h = config.hidden_size;
    let q_size = config.num_heads * config.head_dim;
    let kv_size = config.num_kv_heads * config.head_dim;
    let ff_size = config.intermediate_size;
    let eps = config.rms_norm_eps;

    // Try fused Norm + QKV + RoPE + KV-Write (1 kernel instead of 4)
    let fused_norm_qkv = gpu_dispatch_fused_norm_qkv_rope_kvwrite_on_stream(
        device,
        scratch.hidden.as_ptr() as *const f32,
        gpu_layer.attn_norm.as_ptr() as *const f32,
        eps,
        &gpu_layer.attn_q,
        &gpu_layer.attn_q_meta,
        gpu_layer.attn_q_bias.as_ref(),
        &gpu_layer.attn_k,
        &gpu_layer.attn_k_meta,
        gpu_layer.attn_k_bias.as_ref(),
        &gpu_layer.attn_v,
        &gpu_layer.attn_v_meta,
        gpu_layer.attn_v_bias.as_ref(),
        scratch.q.as_ptr() as *mut f32,
        kv.k_ptr(layer_idx)?,
        kv.v_ptr(layer_idx)?,
        q_size,
        kv_size,
        h,
        scratch.decode_pos_ptr(),
        config.head_dim,
        config.rope_theta,
        config.rope_neox,
        device.stream(),
    )?;

    if !fused_norm_qkv {
        // Fallback: separate Norm + QKV + RoPE + KV-Write (4 kernels)
        gpu_dispatch_rms_norm(
            device,
            scratch.hidden.as_ptr() as *const f32,
            gpu_layer.attn_norm.as_ptr() as *const f32,
            scratch.normed.as_ptr() as *mut f32,
            h,
            eps,
            device.stream(),
        )?;

        let fused_qkv_rope = gpu_dispatch_fused_qkv_rope_kvwrite_on_stream(
            device,
            &gpu_layer.attn_q,
            &gpu_layer.attn_q_meta,
            gpu_layer.attn_q_bias.as_ref(),
            &gpu_layer.attn_k,
            &gpu_layer.attn_k_meta,
            gpu_layer.attn_k_bias.as_ref(),
            &gpu_layer.attn_v,
            &gpu_layer.attn_v_meta,
            gpu_layer.attn_v_bias.as_ref(),
            scratch.normed.as_ptr() as *const f32,
            scratch.q.as_ptr() as *mut f32,
            kv.k_ptr(layer_idx)?,
            kv.v_ptr(layer_idx)?,
            q_size,
            kv_size,
            h,
            scratch.decode_pos_ptr(),
            config.head_dim,
            config.rope_theta,
            config.rope_neox,
            device.stream(),
        )?;

        if !fused_qkv_rope {
            gpu_dispatch_fused_qkv_on_stream(
                device,
                &gpu_layer.attn_q,
                &gpu_layer.attn_q_meta,
                gpu_layer.attn_q_bias.as_ref(),
                &gpu_layer.attn_k,
                &gpu_layer.attn_k_meta,
                gpu_layer.attn_k_bias.as_ref(),
                &gpu_layer.attn_v,
                &gpu_layer.attn_v_meta,
                gpu_layer.attn_v_bias.as_ref(),
                scratch.normed.as_ptr() as *const f32,
                scratch.q.as_ptr() as *mut f32,
                scratch.k.as_ptr() as *mut f32,
                scratch.v.as_ptr() as *mut f32,
                q_size,
                kv_size,
                h,
                device.stream(),
            )?;

            rope_heads_from_state_on_stream(
                scratch.q.as_ptr() as *mut f32,
                scratch.decode_pos_ptr(),
                config.num_heads,
                config.head_dim,
                config.rope_theta,
                config.rope_neox,
                device.stream(),
            )?;
            kv_write_rope_from_state_on_stream(
                kv.k_ptr(layer_idx)?,
                kv.v_ptr(layer_idx)?,
                scratch.k.as_ptr() as *const f32,
                scratch.v.as_ptr() as *const f32,
                scratch.decode_pos_ptr(),
                config.num_kv_heads,
                config.head_dim,
                config.rope_theta,
                config.rope_neox,
                device.stream(),
            )?;
        }
    }

    gpu_attention_decode_from_state(device, scratch, kv, layer_idx, config)?;

    let attn_residual_fused = gpu_dispatch_gemv_residual_on_stream(
        device,
        &gpu_layer.attn_o,
        &gpu_layer.attn_o_meta,
        scratch.attn_out.as_ptr() as *const f32,
        scratch.hidden.as_ptr() as *const f32,
        scratch.hidden.as_ptr() as *mut f32,
        h,
        q_size,
        device.stream(),
    )?;
    if !attn_residual_fused {
        gpu_dispatch_gemv_on_stream(
            device,
            &gpu_layer.attn_o,
            &gpu_layer.attn_o_meta,
            scratch.attn_out.as_ptr() as *const f32,
            scratch.layer_out.as_ptr() as *mut f32,
            h,
            q_size,
            device.stream(),
        )?;
        residual_add_inplace(device, &scratch.hidden, &scratch.layer_out, h)?;
    }

    // Try fused Norm + Gate+Up+SiLU (1 kernel instead of 2)
    let fused_norm_gate_up = gpu_dispatch_fused_norm_gate_up_on_stream(
        device,
        scratch.hidden.as_ptr() as *const f32,
        gpu_layer.ffn_norm.as_ptr() as *const f32,
        eps,
        &gpu_layer.ffn_gate,
        &gpu_layer.ffn_gate_meta,
        &gpu_layer.ffn_up,
        &gpu_layer.ffn_up_meta,
        scratch.swiglu.as_ptr() as *mut f32,
        ff_size,
        h,
        device.stream(),
    )?;

    if !fused_norm_gate_up {
        // Fallback: separate Norm + Gate+Up+SiLU (2 kernels)
        gpu_dispatch_rms_norm(
            device,
            scratch.hidden.as_ptr() as *const f32,
            gpu_layer.ffn_norm.as_ptr() as *const f32,
            scratch.normed.as_ptr() as *mut f32,
            h,
            eps,
            device.stream(),
        )?;
        gpu_dispatch_fused_gate_up_on_stream(
            device,
            &gpu_layer.ffn_gate,
            &gpu_layer.ffn_gate_meta,
            &gpu_layer.ffn_up,
            &gpu_layer.ffn_up_meta,
            None,
            None,
            scratch.normed.as_ptr() as *const f32,
            scratch.swiglu.as_ptr() as *mut f32,
            ff_size,
            h,
            device.stream(),
        )?;
    }

    let ffn_residual_fused = gpu_dispatch_gemv_residual_on_stream(
        device,
        &gpu_layer.ffn_down,
        &gpu_layer.ffn_down_meta,
        scratch.swiglu.as_ptr() as *const f32,
        scratch.hidden.as_ptr() as *const f32,
        scratch.hidden.as_ptr() as *mut f32,
        ff_size,
        h,
        device.stream(),
    )?;
    if !ffn_residual_fused {
        gpu_dispatch_gemv_on_stream(
            device,
            &gpu_layer.ffn_down,
            &gpu_layer.ffn_down_meta,
            scratch.swiglu.as_ptr() as *const f32,
            scratch.layer_out.as_ptr() as *mut f32,
            h,
            ff_size,
            device.stream(),
        )?;
        residual_add_inplace(device, &scratch.hidden, &scratch.layer_out, h)?;
    }

    Ok(())
}

fn gpu_launch_full_greedy_decode_on_stream(
    device: &GpuDevice,
    gpu_weights: &GpuModelWeights,
    kv: &mut GpuKvCache,
    scratch: &mut GpuForwardScratch,
    config: &ModelConfig,
) -> GpuResult<()> {
    for layer_idx in 0..config.num_layers {
        gpu_layer_forward_from_state_on_stream(
            device,
            gpu_weights.layer(layer_idx),
            kv,
            scratch,
            layer_idx,
            config,
        )?;
    }

    gpu_launch_greedy_logits_tail_on_stream(device, gpu_weights, scratch, config)
}

fn gpu_capture_full_greedy_decode_graph(
    device: &GpuDevice,
    gpu_weights: &GpuModelWeights,
    kv: &mut GpuKvCache,
    scratch: &mut GpuForwardScratch,
    config: &ModelConfig,
) -> GpuResult<super::graph::CapturedDecodeGraph> {
    let key = gpu_full_decode_graph_key(device, gpu_weights, kv, config)?;
    scratch.upload_decode_state(0, 1, device.stream())?;
    device.synchronize()?;
    // Pre-allocate Q8 workspace before capture — hipMalloc is forbidden during stream capture.
    device.reserve_q8_workspace(q8_0_workspace_bytes(config.hidden_size))?;
    device.begin_capture(ffi::hipStreamCaptureMode::hipStreamCaptureModeGlobal)?;
    let capture_result =
        gpu_launch_full_greedy_decode_on_stream(device, gpu_weights, kv, scratch, config);
    let end_capture_result = device.end_capture();

    match capture_result {
        Ok(()) => {
            let graph = HipGraph::from_raw(end_capture_result?);
            super::graph::CapturedDecodeGraph::from_captured_graph(graph, key)
        }
        Err(err) => {
            let _ = end_capture_result;
            Err(err)
        }
    }
}

fn gpu_try_full_greedy_decode_graph(
    device: &GpuDevice,
    gpu_weights: &GpuModelWeights,
    kv: &mut GpuKvCache,
    scratch: &mut GpuForwardScratch,
    pos: usize,
    config: &ModelConfig,
) -> GpuResult<Option<u32>> {
    if decode_graph_disabled() {
        return Ok(None);
    }
    // Full-decode graph (all layers + lm_head in one graph) is disabled.
    // On RDNA4 (gfx1201), graph replay of kernels that read pos/seq_len
    // from device pointers returns stale values, producing wrong output.
    // The tail-only graph (lm_head + argmax) gives equivalent performance
    // since layer kernels are memory-bound anyway.
    // TODO: re-enable when ROCm fixes device-pointer reads in graph replay.
    return Ok(None);

    let key = gpu_full_decode_graph_key(device, gpu_weights, kv, config)?;
    if !scratch.has_decode_graph_for(key) {
        // Try updating existing graph if it matches top-level key
        // (This part is simplified for prototype, true llama.cpp logic
        // would keep the executable graph alive and just patch pointers)

        let capture_status = match device.stream_capture_status() {
            Ok(status) => status,
            Err(_) => return Ok(None),
        };
        if capture_status != ffi::hipStreamCaptureStatus::hipStreamCaptureStatusNone {
            return Ok(None);
        }

        // Capture a new graph temporarily to see if we can update the executable one
        scratch.upload_decode_state(0, 1, device.stream())?;
        // Pre-allocate Q8 workspace before capture — hipMalloc is forbidden during stream capture.
        device.reserve_q8_workspace(q8_0_workspace_bytes(config.hidden_size))?;
        device.begin_capture(ffi::hipStreamCaptureMode::hipStreamCaptureModeGlobal)?;
        let capture_res =
            gpu_launch_full_greedy_decode_on_stream(device, gpu_weights, kv, scratch, config);
        let raw_graph = device.end_capture()?;
        capture_res?;

        let new_graph = HipGraph::from_raw(raw_graph);
        if scratch.try_update_decode_graph(&new_graph)? {
            // Update successful! No need to instantiate a new Executable graph.
        } else {
            // Topology changed or no existing graph, instantiate new one
            let captured = super::graph::CapturedDecodeGraph::from_captured_graph(new_graph, key)?;
            scratch.replace_decode_graph(captured);
        }
    }

    scratch.upload_decode_state(pos, pos + 1, device.stream())?;
    if let Some(graph) = scratch.decode_graph() {
        if graph.launch(device.stream()).is_ok() {
            // Wait for completion and read result safely
            gpu_read_greedy_argmax_result(device, scratch, config.vocab_size)?;
            device.synchronize()?;
            let token = scratch.argmax_result_index.as_slice::<i32>()[0];
            return Ok(Some(token as u32));
        }
        scratch.clear_decode_graph();
    }

    Ok(None)
}

/// Hybrid single-layer decode step used by the CLI path and GPU integration tests.
pub fn gpu_layer_forward_hybrid(
    device: &GpuDevice,
    gpu_layer: &GpuLayerWeights,
    kv: &mut GpuKvCache,
    scratch: &mut GpuForwardScratch,
    layer_idx: usize,
    pos: usize,
    config: &ModelConfig,
) -> GpuResult<()> {
    let h = config.hidden_size;
    let q_size = config.num_heads * config.head_dim;
    let kv_size = config.num_kv_heads * config.head_dim;
    let ff_size = config.intermediate_size;
    let eps = config.rms_norm_eps;
    if decode_stage_profiling_enabled() {
        let mut guard = decode_stage_profile_store()
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        guard.layer_invocations += 1;
    }

    profile_decode_stage(device, DecodeStage::AttnNorm, || {
        gpu_dispatch_rms_norm(
            device,
            scratch.hidden.as_ptr() as *const f32,
            gpu_layer.attn_norm.as_ptr() as *const f32,
            scratch.normed.as_ptr() as *mut f32,
            h,
            eps,
            device.stream(),
        )
    })?;

profile_decode_stage(device, DecodeStage::Qkv, || {
        gpu_dispatch_fused_qkv_on_stream(
            device,
            &gpu_layer.attn_q,
            &gpu_layer.attn_q_meta,
            gpu_layer.attn_q_bias.as_ref(),
            &gpu_layer.attn_k,
            &gpu_layer.attn_k_meta,
            gpu_layer.attn_k_bias.as_ref(),
            &gpu_layer.attn_v,
            &gpu_layer.attn_v_meta,
            gpu_layer.attn_v_bias.as_ref(),
            scratch.normed.as_ptr() as *const f32,
            scratch.q.as_ptr() as *mut f32,
            scratch.k.as_ptr() as *mut f32,
            scratch.v.as_ptr() as *mut f32,
            q_size,
            kv_size,
            h,
            device.stream(),
        )
    })?;

profile_decode_stage(device, DecodeStage::QRope, || {
        rope_heads_on_stream(
            scratch.q.as_ptr() as *mut f32,
            pos,
            config.num_heads,
            config.head_dim,
            config.rope_theta,
            config.rope_neox,
            device.stream(),
        )
    })?;

    profile_decode_stage(device, DecodeStage::KvWrite, || {
        kv_write_rope_on_stream(
            kv,
            layer_idx,
            scratch.k.as_ptr() as *mut f32,
            scratch.v.as_ptr() as *mut f32,
            pos,
            config.num_kv_heads,
            config.head_dim,
            config.rope_theta,
            config.rope_neox,
            device.stream(),
        )
    })?;

    profile_decode_stage(device, DecodeStage::Attention, || {
        gpu_attention_decode(device, scratch, kv, layer_idx, pos, config)
    })?;

    let attn_residual_fused = profile_decode_stage(device, DecodeStage::AttnProj, || {
        gpu_dispatch_gemv_residual_on_stream(
            device,
            &gpu_layer.attn_o,
            &gpu_layer.attn_o_meta,
            scratch.attn_out.as_ptr() as *const f32,
            scratch.hidden.as_ptr() as *const f32,
            scratch.hidden.as_ptr() as *mut f32,
            h,
            q_size,
            device.stream(),
        )
    })?;
    if !attn_residual_fused {
        profile_decode_stage(device, DecodeStage::AttnProj, || {
            gpu_dispatch_gemv_on_stream(
                device,
                &gpu_layer.attn_o,
                &gpu_layer.attn_o_meta,
                scratch.attn_out.as_ptr() as *const f32,
                scratch.layer_out.as_ptr() as *mut f32,
                h,
                q_size,
                device.stream(),
            )
        })?;
        profile_decode_stage(device, DecodeStage::AttnResidual, || {
            residual_add_inplace(device, &scratch.hidden, &scratch.layer_out, h)
        })?;
    }

    profile_decode_stage(device, DecodeStage::FfnNorm, || {
        gpu_dispatch_rms_norm(
            device,
            scratch.hidden.as_ptr() as *const f32,
            gpu_layer.ffn_norm.as_ptr() as *const f32,
            scratch.normed.as_ptr() as *mut f32,
            h,
            eps,
            device.stream(),
        )
    })?;
    profile_decode_stage(device, DecodeStage::GateUp, || {
        gpu_dispatch_fused_gate_up_on_stream(
            device,
            &gpu_layer.ffn_gate,
            &gpu_layer.ffn_gate_meta,
            &gpu_layer.ffn_up,
            &gpu_layer.ffn_up_meta,
            None, // w_gate_up_interleaved
            None, // w_gate_up_interleaved_tile4
            scratch.normed.as_ptr() as *const f32,
            scratch.swiglu.as_ptr() as *mut f32,
            ff_size,
            h,
            device.stream(),
        )
    })?;

    let ffn_residual_fused = profile_decode_stage(device, DecodeStage::FfnDown, || {
        gpu_dispatch_gemv_residual_on_stream(
            device,
            &gpu_layer.ffn_down,
            &gpu_layer.ffn_down_meta,
            scratch.swiglu.as_ptr() as *const f32,
            scratch.hidden.as_ptr() as *const f32,
            scratch.hidden.as_ptr() as *mut f32,
            ff_size,
            h,
            device.stream(),
        )
    })?;
    if !ffn_residual_fused {
        profile_decode_stage(device, DecodeStage::FfnDown, || {
            gpu_dispatch_gemv_on_stream(
                device,
                &gpu_layer.ffn_down,
                &gpu_layer.ffn_down_meta,
                scratch.swiglu.as_ptr() as *const f32,
                scratch.layer_out.as_ptr() as *mut f32,
                h,
                ff_size,
                device.stream(),
            )
        })?;
        profile_decode_stage(device, DecodeStage::FfnResidual, || {
            residual_add_inplace(device, &scratch.hidden, &scratch.layer_out, h)
        })?;
    }

    Ok(())
}

/// Embed one token, preferring a native GPU path and falling back to CPU upload.
pub fn gpu_embed_token_hybrid(
    device: &GpuDevice,
    token_id: u32,
    gpu_weights: &GpuModelWeights,
    cpu_weights: &CpuModelWeights,
    scratch: &mut GpuForwardScratch,
    host_scratch: &mut CpuForwardScratch,
    config: &ModelConfig,
) -> GpuResult<()> {
    let h = config.hidden_size;
    match gpu_weights.token_emb_meta.wtype {
        GgmlType::Q8_0 => {
            validate_token_embedding_layout(&gpu_weights.token_emb_meta, config)?;
            embed_q8_0_token(
                gpu_weights.token_emb.as_ptr(),
                scratch.hidden.as_ptr() as *mut f32,
                h,
                config.vocab_size,
                token_id,
            )
        }
        _ => {
            // CPU embed into pinned buffer
            cpu_embed_token(
                token_id,
                cpu_weights,
                &mut scratch.input_hidden_pinned.as_slice_mut::<f32>()[..h],
                config,
            );
            // Async upload
            unsafe {
                ffi::hip_memcpy_h2d_async(
                    scratch.hidden.as_ptr(),
                    scratch.input_hidden_pinned.as_ptr(),
                    h * std::mem::size_of::<f32>(),
                    device.stream(),
                )?;
            }
            Ok(())
        }
    }
}

/// Full decode forward pass using GPU kernels plus targeted CPU fallbacks.
pub fn gpu_full_forward_hybrid(
    device: &GpuDevice,
    gpu_weights: &GpuModelWeights,
    cpu_weights: &CpuModelWeights,
    kv: &mut GpuKvCache,
    scratch: &mut GpuForwardScratch,
    host_scratch: &mut CpuForwardScratch,
    pos: usize,
    config: &ModelConfig,
    logits_mode: GpuLogitsMode,
) -> GpuResult<Option<u32>> {
    if matches!(logits_mode, GpuLogitsMode::GreedyArgmax) {
        if let Some(token) =
            gpu_try_full_greedy_decode_graph(device, gpu_weights, kv, scratch, pos, config)?
        {
            return Ok(Some(token));
        }
    }

    for layer_idx in 0..config.num_layers {
        gpu_layer_forward_hybrid(
            device,
            gpu_weights.layer(layer_idx),
            kv,
            scratch,
            layer_idx,
            pos,
            config,
        )?;
    }

    if matches!(logits_mode, GpuLogitsMode::Skip) {
        return Ok(None);
    }

    let h = config.hidden_size;
    let v = config.vocab_size;
    let gpu_result = match logits_mode {
        GpuLogitsMode::DownloadToHost => {
            gpu_dispatch_rms_norm(
                device,
                scratch.hidden.as_ptr() as *const f32,
                gpu_weights.output_norm.as_ptr() as *const f32,
                scratch.normed.as_ptr() as *mut f32,
                h,
                config.rms_norm_eps,
                device.stream(),
            )?;
            gpu_dispatch_gemv_on_stream(
                device,
                &gpu_weights.lm_head,
                &gpu_weights.lm_head_meta,
                scratch.normed.as_ptr() as *const f32,
                scratch.logits.as_ptr() as *mut f32,
                v,
                h,
                device.stream(),
            )?;
            download_f32(&scratch.logits, &mut host_scratch.logits[..v])?;
            Ok(None)
        }
        GpuLogitsMode::GreedyArgmax => {
            gpu_try_greedy_decode_graph(device, gpu_weights, scratch, config).map(Some)
        }
        GpuLogitsMode::Skip => Ok(None),
    };

    match gpu_result {
        Ok(result) => Ok(result),
        Err(GpuError::InvalidWeightLayout { .. }) | Err(GpuError::UnsupportedWeightType { .. }) => {
            gpu_dispatch_rms_norm(
                device,
                scratch.hidden.as_ptr() as *const f32,
                gpu_weights.output_norm.as_ptr() as *const f32,
                scratch.normed.as_ptr() as *mut f32,
                h,
                config.rms_norm_eps,
                device.stream(),
            )?;
            cpu_fallback_gemv(
                "lm_head",
                &cpu_weights.lm_head,
                &cpu_weights.lm_head_meta,
                &scratch.normed,
                &mut host_scratch.normed,
                &mut host_scratch.logits,
                v,
                h,
                &mut host_scratch.q8_scratch,
            )?;
            match logits_mode {
                GpuLogitsMode::DownloadToHost => Ok(None),
                GpuLogitsMode::GreedyArgmax => Ok(Some(crate::cpu::sampler::cpu_sample_greedy(
                    &host_scratch.logits[..v],
                ))),
                GpuLogitsMode::Skip => Ok(None),
            }
        }
        Err(err) => Err(err),
    }
}
