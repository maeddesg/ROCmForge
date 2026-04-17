//! GPU math dispatch for model weights.
//!
//! This layer validates GGUF metadata before calling the raw HIP kernels.

use super::device::GpuDevice;
use super::error::{GpuError, GpuResult};
use super::ffi::hip_stream_synchronize;
use super::ffi::{hipStreamCaptureStatus, hipStream_t, hip_stream_is_capturing};
use super::kernels::{
    add_on_stream, gemm_q4_0_f32, gemm_q4_1_f32, gemm_q4_k_f32, gemm_q5_k_f32, gemm_q8_0_f32,
    gemv_q4_0_f32_batched_on_stream, gemv_q4_0_f32_batched_tiled_on_stream,
    gemv_gate_up_q4_0_f32_on_stream, gemv_gate_up_q4_0_q8_0_on_stream,
    gemv_gate_up_swiglu_q4_0_f32_on_stream,
    gemv_gate_up_swiglu_q4_0_f32_q8_inline_interleaved_on_stream,
    gemv_gate_up_swiglu_q4_0_f32_q8_inline_interleaved_tile4_on_stream,
    gemv_gate_up_swiglu_q4_0_f32_q8_inline_on_stream_variant, gemv_q4_0_f32_on_stream_unchecked,
    gemv_q4_0_f32_q8_inline_residual_norm_on_stream,
    gemv_q4_0_f32_q8_inline_residual_on_stream, gemv_q4_0_f32_q8_inline_residual_on_stream_variant,
    gemv_q4_0_f32_residual_on_stream_unchecked, gemv_q4_0_q8_0_on_stream,
    gemv_q4_0_q8_0_residual_on_stream, gemv_q4_1_f32_on_stream_unchecked,
    gemv_q4_1_f32_residual_on_stream_unchecked, gemv_q4_1_f32_residual_on_stream_variant_unchecked,
    gemv_q4_k_f32_on_stream, gemv_q5_k_f32_on_stream, gemv_q6_k_f32_on_stream,
    gemv_q8_0_f32_lm_head_on_stream,
    gemv_q8_0_f32_lm_head_on_stream_variant, gemv_q8_0_f32_on_stream, gemv_qkv_q4_0_f32_on_stream,
    gemv_qkv_q4_0_f32_on_stream_variant, mul_on_stream, q8_0_workspace_bytes,
    quantize_q8_0_on_stream, rms_norm_on_stream, rms_norm_vulkan_style, silu_on_stream,
};
use super::launch_autotune::{
    lookup_gate_up_swiglu_q8_variant, lookup_lm_head_q8_variant, lookup_q4_0_q8_residual_variant,
    select_gate_up_swiglu_q8_variant, select_lm_head_q8_variant, select_q4_0_q8_residual_variant,
    select_q4_1_residual_variant, select_qkv_variant, VariantId,
};
use super::safety::{
    disable_q8_activation_fastpath_runtime, experimental_gpu_kernels_enabled,
    experimental_q8_activation_fastpath_enabled, launch_autotune_enabled,
};
use super::weights::{GpuBuffer, TensorRole, WeightMeta};
use crate::loader::GgmlType;
fn supports_gemv_type(wtype: GgmlType) -> bool {
    matches!(
        wtype,
        GgmlType::Q4_0 | GgmlType::Q4_1 | GgmlType::Q8_0 | GgmlType::Q4_K | GgmlType::Q5_K | GgmlType::Q6_K
    )
}

fn is_lm_head_role(role: TensorRole) -> bool {
    matches!(role, TensorRole::LmHead | TensorRole::TiedLmHead)
}

fn config_num_heads(q_size: usize, h: usize) -> usize {
    // Hidden size h = n_heads * head_dim
    // For many models, head_dim is 64 or 128
    if h % 128 == 0 {
        q_size / 128
    } else {
        q_size / 64
    }
}

/// Dispatch a GPU RMS norm.
pub fn gpu_dispatch_rms_norm(
    _device: &GpuDevice,
    x: *const f32,
    weight: *const f32,
    out: *mut f32,
    n: usize,
    eps: f32,
    stream: hipStream_t,
) -> GpuResult<()> {
    if experimental_gpu_kernels_enabled() {
        if let Ok(()) = rms_norm_vulkan_style(x, weight, out, n, eps, stream) {
            return Ok(());
        }
    }

    rms_norm_on_stream(x, weight, out, n, eps, stream)
}

fn validate_gemv_layout(meta: &WeightMeta, out_dim: usize, in_dim: usize) -> GpuResult<()> {
    if meta.dims.len() < 2 {
        return Err(GpuError::InvalidWeightLayout {
            tensor: "gpu_dispatch_gemv".to_string(),
            dims: meta.dims.clone(),
            reason: "weight metadata must describe a 2D matrix".to_string(),
        });
    }

    if (meta.dims[0] as usize == in_dim && meta.dims[1] as usize == out_dim)
        || (meta.dims[0] as usize == out_dim && meta.dims[1] as usize == in_dim)
    {
        Ok(())
    } else {
        Err(GpuError::InvalidWeightLayout {
            tensor: "gpu_dispatch_gemv".to_string(),
            dims: meta.dims.clone(),
            reason: format!(
                "shape mismatch: matrix is {:?}, but vector is [{}] and output is [{}]",
                meta.dims, in_dim, out_dim
            ),
        })
    }
}

fn quantize_input_q8_workspace(
    device: &GpuDevice,
    input: *const f32,
    n_rows: usize,
    stream: hipStream_t,
) -> GpuResult<*mut u8> {
    let workspace = device.q8_workspace_ptr(q8_0_workspace_bytes(n_rows))?;
    quantize_q8_0_on_stream(input, workspace, n_rows, stream)?;
    Ok(workspace)
}

fn q8_fastpath_ok(context: &str, fastpath_result: GpuResult<()>) -> bool {
    match fastpath_result {
        Ok(()) => true,
        Err(err) => {
            disable_q8_activation_fastpath_runtime(&format!("{context}: {err}"));
            false
        }
    }
}

fn try_q4_0_q8_0_fastpath(
    device: &GpuDevice,
    weights: &GpuBuffer,
    input: *const f32,
    output: *mut f32,
    in_dim: usize,
    out_dim: usize,
    stream: hipStream_t,
) -> GpuResult<()> {
    let workspace = quantize_input_q8_workspace(device, input, in_dim, stream)?;
    gemv_q4_0_q8_0_on_stream(
        weights.as_ptr() as *const u8,
        workspace as *const u8,
        output,
        in_dim,
        out_dim,
        stream,
    )
}

fn try_q4_0_q8_0_residual_fastpath(
    weights: &GpuBuffer,
    input: *const f32,
    residual: *const f32,
    output: *mut f32,
    in_dim: usize,
    out_dim: usize,
    stream: hipStream_t,
) -> GpuResult<()> {
    gemv_q4_0_f32_q8_inline_residual_on_stream(
        weights.as_ptr() as *const u8,
        input,
        residual,
        output,
        in_dim,
        out_dim,
        stream,
    )
}

fn try_q4_0_q8_0_residual_fastpath_prequantized(
    device: &GpuDevice,
    weights: &GpuBuffer,
    input: *const f32,
    residual: *const f32,
    output: *mut f32,
    in_dim: usize,
    out_dim: usize,
    stream: hipStream_t,
) -> GpuResult<()> {
    let workspace = quantize_input_q8_workspace(device, input, in_dim, stream)?;
    gemv_q4_0_q8_0_residual_on_stream(
        weights.as_ptr() as *const u8,
        workspace as *const u8,
        residual,
        output,
        in_dim,
        out_dim,
        stream,
    )
}

fn try_q4_0_q8_0_gate_up_fastpath(
    device: &GpuDevice,
    w_gate: &GpuBuffer,
    w_up: &GpuBuffer,
    input: *const f32,
    gate_output: *mut f32,
    up_output: *mut f32,
    h: usize,
    ff_size: usize,
    stream: hipStream_t,
) -> GpuResult<()> {
    let workspace = quantize_input_q8_workspace(device, input, h, stream)?;
    gemv_gate_up_q4_0_q8_0_on_stream(
        w_gate.as_ptr() as *const u8,
        w_up.as_ptr() as *const u8,
        workspace as *const u8,
        gate_output,
        up_output,
        h,
        ff_size,
        stream,
    )
}

fn try_q4_0_q8_0_fused_gate_up_fastpath(
    w_gate: &GpuBuffer,
    w_up: &GpuBuffer,
    input: *const f32,
    output: *mut f32,
    h: usize,
    ff_size: usize,
    stream: hipStream_t,
) -> GpuResult<()> {
    try_q4_0_q8_0_fused_gate_up_fastpath_variant(w_gate, w_up, input, output, h, ff_size, 0, stream)
}

fn try_q4_0_q8_0_fused_gate_up_fastpath_prequantized(
    device: &GpuDevice,
    w_gate: &GpuBuffer,
    w_up: &GpuBuffer,
    input: *const f32,
    output: *mut f32,
    h: usize,
    ff_size: usize,
    stream: hipStream_t,
) -> GpuResult<()> {
    let workspace = quantize_input_q8_workspace(device, input, h, stream)?;
    super::kernels::gemv_gate_up_swiglu_q4_0_q8_0_on_stream(
        w_gate.as_ptr() as *const u8,
        w_up.as_ptr() as *const u8,
        workspace as *const u8,
        output,
        h,
        ff_size,
        stream,
    )
}

fn try_q4_0_q8_0_fused_gate_up_fastpath_variant(
    w_gate: &GpuBuffer,
    w_up: &GpuBuffer,
    input: *const f32,
    output: *mut f32,
    h: usize,
    ff_size: usize,
    variant: i32,
    stream: hipStream_t,
) -> GpuResult<()> {
    gemv_gate_up_swiglu_q4_0_f32_q8_inline_on_stream_variant(
        w_gate.as_ptr() as *const u8,
        w_up.as_ptr() as *const u8,
        input,
        output,
        h,
        ff_size,
        variant,
        stream,
    )
}

fn dispatch_gemv_impl(
    device: &GpuDevice,
    stream: hipStream_t,
    weights: &GpuBuffer,
    meta: &WeightMeta,
    input: *const f32,
    output: *mut f32,
    out_dim: usize,
    in_dim: usize,
) -> GpuResult<()> {
    unsafe {
        match meta.wtype {
            GgmlType::Q4_0 => {
                if experimental_q8_activation_fastpath_enabled()
                    && q8_fastpath_ok(
                        "gemv_q4_0_q8_0",
                        try_q4_0_q8_0_fastpath(
                            device, weights, input, output, in_dim, out_dim, stream,
                        ),
                    )
                {
                    return Ok(());
                }

                if experimental_gpu_kernels_enabled() {
                    let n_waves = 8;
                    if let Ok(()) = super::kernels::quant::gemv_q4_0_f32_vulkan_style(
                        device,
                        weights.as_ptr() as *const u8,
                        input,
                        output,
                        in_dim,
                        out_dim,
                        n_waves,
                        stream,
                    ) {
                        return Ok(());
                    }
                }

                gemv_q4_0_f32_on_stream_unchecked(
                    weights.as_ptr() as *const u8,
                    input,
                    output,
                    in_dim,
                    out_dim,
                    stream,
                )?
            }
            GgmlType::Q4_1 => gemv_q4_1_f32_on_stream_unchecked(
                weights.as_ptr() as *const u8,
                input,
                output,
                in_dim,
                out_dim,
                stream,
            )?,
            GgmlType::Q8_0 => {
                // Check for LM-head specialization
                if is_lm_head_role(meta.role) {
                    let capture_active = matches!(
                        hip_stream_is_capturing(stream),
                        Err(_)
                            | Ok(hipStreamCaptureStatus::hipStreamCaptureStatusActive)
                            | Ok(hipStreamCaptureStatus::hipStreamCaptureStatusInvalidated)
                    );

                    if launch_autotune_enabled() {
                        let variant = if capture_active {
                            lookup_lm_head_q8_variant(in_dim, out_dim)
                                .unwrap_or(VariantId::Baseline)
                        } else {
                            select_lm_head_q8_variant(in_dim, out_dim, |v| {
                                let result = gemv_q8_0_f32_lm_head_on_stream_variant(
                                    weights.as_ptr() as *const u8,
                                    input,
                                    output,
                                    in_dim,
                                    out_dim,
                                    v as i32,
                                    stream,
                                );
                                hip_stream_synchronize(stream)?;
                                result
                            })
                        };

                        gemv_q8_0_f32_lm_head_on_stream_variant(
                            weights.as_ptr() as *const u8,
                            input,
                            output,
                            in_dim,
                            out_dim,
                            variant as i32,
                            stream,
                        )?;
                    } else {
                        gemv_q8_0_f32_lm_head_on_stream(
                            weights.as_ptr() as *const u8,
                            input,
                            output,
                            in_dim,
                            out_dim,
                            stream,
                        )?;
                    }
                } else {
                    gemv_q8_0_f32_on_stream(
                        weights.as_ptr() as *const u8,
                        input,
                        output,
                        in_dim,
                        out_dim,
                        stream,
                    )?
                }
            }
            GgmlType::Q4_K => {
                if experimental_gpu_kernels_enabled() {
                    let n_waves = 8;
                    if let Ok(()) = super::kernels::quant::gemv_q4_k_f32_vulkan_style(
                        device,
                        weights.as_ptr() as *const u8,
                        input,
                        output,
                        in_dim,
                        out_dim,
                        n_waves,
                        stream,
                    ) {
                        return Ok(());
                    }
                }

                gemv_q4_k_f32_on_stream(
                    weights.as_ptr() as *const u8,
                    input,
                    output,
                    in_dim,
                    out_dim,
                    stream,
                )?
            }
            GgmlType::Q5_K => gemv_q5_k_f32_on_stream(
                weights.as_ptr() as *const u8,
                input,
                output,
                in_dim,
                out_dim,
                stream,
            )?,
            GgmlType::Q6_K => gemv_q6_k_f32_on_stream(
                weights.as_ptr() as *const u8,
                input,
                output,
                in_dim,
                out_dim,
                stream,
            )?,
            _ => unreachable!("unsupported types return before dispatch"),
        }
    }

    Ok(())
}

/// Dispatch a GPU GEMV for one GGUF weight tensor.
pub fn gpu_dispatch_gemv(
    device: &GpuDevice,
    weights: &GpuBuffer,
    meta: &WeightMeta,
    input: *const f32,
    output: *mut f32,
    out_dim: usize,
    in_dim: usize,
) -> GpuResult<()> {
    validate_gemv_layout(meta, out_dim, in_dim)?;

    if !supports_gemv_type(meta.wtype) {
        return Err(GpuError::UnsupportedWeightType {
            tensor: "gpu_dispatch_gemv".to_string(),
            wtype: meta.wtype,
        });
    }

    dispatch_gemv_impl(
        device,
        hipStream_t::null(),
        weights,
        meta,
        input,
        output,
        out_dim,
        in_dim,
    )
}
/// Dispatch a GPU GEMV on an explicit HIP stream.
pub fn gpu_dispatch_gemv_on_stream(
    device: &GpuDevice,
    weights: &GpuBuffer,
    meta: &WeightMeta,
    input: *const f32,
    output: *mut f32,
    out_dim: usize,
    in_dim: usize,
    stream: hipStream_t,
) -> GpuResult<()> {
    validate_gemv_layout(meta, out_dim, in_dim)?;

    if !supports_gemv_type(meta.wtype) {
        return Err(GpuError::UnsupportedWeightType {
            tensor: "gpu_dispatch_gemv_on_stream".to_string(),
            wtype: meta.wtype,
        });
    }

    dispatch_gemv_impl(
        device, stream, weights, meta, input, output, out_dim, in_dim,
    )
}

pub fn gpu_dispatch_gemv_residual_on_stream(
    device: &GpuDevice,
    weights: &GpuBuffer,
    meta: &WeightMeta,
    input: *const f32,
    residual: *const f32,
    output: *mut f32,
    in_dim: usize,
    out_dim: usize,
    stream: hipStream_t,
) -> GpuResult<bool> {
    unsafe {
        match meta.wtype {
            GgmlType::Q4_0 => {
                if experimental_q8_activation_fastpath_enabled() {
                    // Check if stream capture is active - skip autotune benchmarking during capture
                    let capture_active = matches!(
                        hip_stream_is_capturing(stream),
                        Err(_)
                            | Ok(hipStreamCaptureStatus::hipStreamCaptureStatusActive)
                            | Ok(hipStreamCaptureStatus::hipStreamCaptureStatusInvalidated)
                    );

                    if launch_autotune_enabled() {
                        let variant = if capture_active {
                            lookup_q4_0_q8_residual_variant(in_dim, out_dim)
                                .unwrap_or(VariantId::Baseline)
                        } else {
                            select_q4_0_q8_residual_variant(in_dim, out_dim, |v| {
                                let result = match v {
                                    VariantId::Baseline | VariantId::Variant1 => {
                                        gemv_q4_0_f32_q8_inline_residual_on_stream_variant(
                                            weights.as_ptr() as *const u8,
                                            input,
                                            residual,
                                            output,
                                            in_dim,
                                            out_dim,
                                            v as i32,
                                            stream,
                                        )
                                    }
                                    VariantId::Variant2 => {
                                        try_q4_0_q8_0_residual_fastpath_prequantized(
                                            device, weights, input, residual, output, in_dim,
                                            out_dim, stream,
                                        )
                                    }
                                };
                                hip_stream_synchronize(stream)?;
                                result
                            })
                        };

                        // Execute selected (or cached) fastpath variant and keep fallback behavior on failures.
                        let selected_result = match variant {
                            VariantId::Baseline | VariantId::Variant1 => {
                                gemv_q4_0_f32_q8_inline_residual_on_stream_variant(
                                    weights.as_ptr() as *const u8,
                                    input,
                                    residual,
                                    output,
                                    in_dim,
                                    out_dim,
                                    variant as i32,
                                    stream,
                                )
                            }
                            VariantId::Variant2 => try_q4_0_q8_0_residual_fastpath_prequantized(
                                device, weights, input, residual, output, in_dim, out_dim, stream,
                            ),
                        };

                        if q8_fastpath_ok("gemv_q4_0_f32_q8_inline_residual", selected_result) {
                            return Ok(true);
                        }
                    }

                    // Non-autotune fastpath: try inline residual, fall back to unchecked on failure
                    if q8_fastpath_ok(
                        "gemv_q4_0_f32_q8_inline_residual",
                        try_q4_0_q8_0_residual_fastpath(
                            weights, input, residual, output, in_dim, out_dim, stream,
                        ),
                    ) {
                        return Ok(true);
                    }
                }

                gemv_q4_0_f32_residual_on_stream_unchecked(
                    weights.as_ptr() as *const u8,
                    input,
                    residual,
                    output,
                    in_dim,
                    out_dim,
                    stream,
                )?;
                Ok(true)
            }
            GgmlType::Q4_1 => {
                // Check if stream capture is active - skip autotune benchmarking during capture
                let capture_active = matches!(
                    hip_stream_is_capturing(stream),
                    Err(_)
                        | Ok(hipStreamCaptureStatus::hipStreamCaptureStatusActive)
                        | Ok(hipStreamCaptureStatus::hipStreamCaptureStatusInvalidated)
                );

                // Autotune-aware dispatch for Q4_1 residual (skip if capturing)
                if launch_autotune_enabled() && !capture_active {
                    let variant = select_q4_1_residual_variant(in_dim, out_dim, |v| {
                        let result = unsafe {
                            gemv_q4_1_f32_residual_on_stream_variant_unchecked(
                                weights.as_ptr() as *const u8,
                                input,
                                residual,
                                output,
                                in_dim,
                                out_dim,
                                v as i32,
                                stream,
                            )
                        };
                        hip_stream_synchronize(stream)?;
                        result
                    });

                    // Execute with selected variant
                    unsafe {
                        gemv_q4_1_f32_residual_on_stream_variant_unchecked(
                            weights.as_ptr() as *const u8,
                            input,
                            residual,
                            output,
                            in_dim,
                            out_dim,
                            variant as i32,
                            stream,
                        )?;
                    }
                } else {
                    // Baseline path (backward compatible)
                    unsafe {
                        gemv_q4_1_f32_residual_on_stream_unchecked(
                            weights.as_ptr() as *const u8,
                            input,
                            residual,
                            output,
                            in_dim,
                            out_dim,
                            stream,
                        )?;
                    }
                }
                Ok(true)
            }
            _ => Ok(false),
        }
    }
}

/// Dispatch a fused GEMV + residual + RMSNorm on an explicit HIP stream.
///
/// Combines: Q4_0 GEMV + residual add + RMSNorm into a single kernel launch.
/// The last block to retire computes the full RMSNorm over the output.
/// Returns `Ok(true)` if the fused kernel was dispatched, `Ok(false)` if
/// the weight type is unsupported (caller should fall back).
/// Dispatch a fused GEMV + residual + RMSNorm on an explicit HIP stream.
///
/// NOTE: This kernel uses inter-block atomics + __threadfence() for the norm
/// reduction. Benchmarking showed ~4.5% regression vs separate kernels because
/// the fence overhead exceeds the savings. Kept for future experimentation.
#[allow(dead_code)]
pub fn gpu_dispatch_gemv_residual_norm_on_stream(
    _device: &GpuDevice,
    weights: &GpuBuffer,
    meta: &WeightMeta,
    input: *const f32,
    residual: *const f32,
    norm_weight: *const f32,
    eps: f32,
    output: *mut f32,
    output_normed: *mut f32,
    in_dim: usize,
    out_dim: usize,
    retire_count: *mut u32,
    stream: hipStream_t,
) -> GpuResult<bool> {
    match meta.wtype {
        GgmlType::Q4_0 => {
            gemv_q4_0_f32_q8_inline_residual_norm_on_stream(
                weights.as_ptr() as *const u8,
                input,
                residual,
                norm_weight,
                eps,
                output,
                output_normed,
                in_dim,
                out_dim,
                retire_count,
                stream,
            )?;
            Ok(true)
        }
        _ => Ok(false),
    }
}

/// Dispatch a fused QKV GEMV with bias on an explicit HIP stream.
pub fn gpu_dispatch_fused_qkv_on_stream(
    device: &GpuDevice,
    w_q: &GpuBuffer,
    q_meta: &WeightMeta,
    q_bias: Option<&GpuBuffer>,
    w_k: &GpuBuffer,
    k_meta: &WeightMeta,
    k_bias: Option<&GpuBuffer>,
    w_v: &GpuBuffer,
    v_meta: &WeightMeta,
    v_bias: Option<&GpuBuffer>,
    input: *const f32,
    out_q: *mut f32,
    out_k: *mut f32,
    out_v: *mut f32,
    q_size: usize,
    kv_size: usize,
    h: usize,
    stream: hipStream_t,
) -> GpuResult<()> {
    if q_meta.wtype == GgmlType::Q4_0
        && k_meta.wtype == GgmlType::Q4_0
        && v_meta.wtype == GgmlType::Q4_0
    {
        // Check if stream capture is active - skip autotune benchmarking during capture
        let capture_active = matches!(
            hip_stream_is_capturing(stream),
            Err(_)
                | Ok(hipStreamCaptureStatus::hipStreamCaptureStatusActive)
                | Ok(hipStreamCaptureStatus::hipStreamCaptureStatusInvalidated)
        );

        // Autotune-aware dispatch for QKV fused (skip if capturing)
        if launch_autotune_enabled() && !capture_active {
            let variant = select_qkv_variant(h, q_size, kv_size, |v| {
                let result = unsafe {
                    gemv_qkv_q4_0_f32_on_stream_variant(
                        w_q.as_ptr() as *const u8,
                        w_k.as_ptr() as *const u8,
                        w_v.as_ptr() as *const u8,
                        q_bias.map_or(std::ptr::null(), |b| b.as_ptr() as *const f32),
                        k_bias.map_or(std::ptr::null(), |b| b.as_ptr() as *const f32),
                        v_bias.map_or(std::ptr::null(), |b| b.as_ptr() as *const f32),
                        input,
                        out_q,
                        out_k,
                        out_v,
                        h,
                        q_size,
                        kv_size,
                        stream,
                        v as i32,
                    )
                };
                hip_stream_synchronize(stream)?;
                result
            });

            // Execute with selected variant
            unsafe {
                gemv_qkv_q4_0_f32_on_stream_variant(
                    w_q.as_ptr() as *const u8,
                    w_k.as_ptr() as *const u8,
                    w_v.as_ptr() as *const u8,
                    q_bias.map_or(std::ptr::null(), |b| b.as_ptr() as *const f32),
                    k_bias.map_or(std::ptr::null(), |b| b.as_ptr() as *const f32),
                    v_bias.map_or(std::ptr::null(), |b| b.as_ptr() as *const f32),
                    input,
                    out_q,
                    out_k,
                    out_v,
                    h,
                    q_size,
                    kv_size,
                    stream,
                    variant as i32,
                )?;
            }
        } else {
            // Baseline path (backward compatible)
            unsafe {
                gemv_qkv_q4_0_f32_on_stream(
                    w_q.as_ptr() as *const u8,
                    w_k.as_ptr() as *const u8,
                    w_v.as_ptr() as *const u8,
                    q_bias.map_or(std::ptr::null(), |b| b.as_ptr() as *const f32),
                    k_bias.map_or(std::ptr::null(), |b| b.as_ptr() as *const f32),
                    v_bias.map_or(std::ptr::null(), |b| b.as_ptr() as *const f32),
                    input,
                    out_q,
                    out_k,
                    out_v,
                    h,
                    q_size,
                    kv_size,
                    stream,
                )?;
            }
        }
        return Ok(());
    }

    // Individual dispatches are safer if we don't have a perfectly matching fused kernel
    gpu_dispatch_gemv_on_stream(device, w_q, q_meta, input, out_q, q_size, h, stream)?;
    gpu_dispatch_gemv_on_stream(device, w_k, k_meta, input, out_k, kv_size, h, stream)?;
    gpu_dispatch_gemv_on_stream(device, w_v, v_meta, input, out_v, kv_size, h, stream)?;

    // Add biases
    if let Some(bias) = q_bias {
        unsafe {
            add_on_stream(out_q, bias.as_ptr() as *const f32, out_q, q_size, stream)?;
        }
    }
    if let Some(bias) = k_bias {
        unsafe {
            add_on_stream(out_k, bias.as_ptr() as *const f32, out_k, kv_size, stream)?;
        }
    }
    if let Some(bias) = v_bias {
        unsafe {
            add_on_stream(out_v, bias.as_ptr() as *const f32, out_v, kv_size, stream)?;
        }
    }

    Ok(())
}

/// Dispatch a fused QKV + RoPE + KV-Write on an explicit HIP stream.
///
/// Replaces 3 kernel launches (QKV, RoPE-Q, RoPE-K+KV-Write) with 1.
/// Returns Ok(true) if the fused path was used, Ok(false) if the caller
/// should fall back to separate kernels (e.g. non-Q4_0 or hidden > 32KB LDS).
pub fn gpu_dispatch_fused_qkv_rope_kvwrite_on_stream(
    device: &GpuDevice,
    w_q: &GpuBuffer,
    q_meta: &WeightMeta,
    q_bias: Option<&GpuBuffer>,
    w_k: &GpuBuffer,
    k_meta: &WeightMeta,
    k_bias: Option<&GpuBuffer>,
    w_v: &GpuBuffer,
    v_meta: &WeightMeta,
    v_bias: Option<&GpuBuffer>,
    input: *const f32,
    out_q: *mut f32,
    k_cache: *mut u16,
    v_cache: *mut u16,
    q_size: usize,
    kv_size: usize,
    h: usize,
    pos_ptr: *const i32,
    head_dim: usize,
    theta_base: f32,
    neox: bool,
    stream: hipStream_t,
) -> GpuResult<bool> {
    // Only supports Q4_0 weights with dimensions compatible with v3 fast path
    if q_meta.wtype != GgmlType::Q4_0
        || k_meta.wtype != GgmlType::Q4_0
        || v_meta.wtype != GgmlType::Q4_0
    {
        return Ok(false);
    }
    // v3 kernel requires n_q % 4 == 0, n_kv % 4 == 0, and hidden fits in LDS
    if (q_size % 4) != 0 || (kv_size % 4) != 0 || h * std::mem::size_of::<f32>() > 32768 {
        return Ok(false);
    }

    use super::kernels::quant::gemv_qkv_rope_kvwrite_q4_0_f32_on_stream;
    gemv_qkv_rope_kvwrite_q4_0_f32_on_stream(
        w_q.as_ptr() as *const u8,
        w_k.as_ptr() as *const u8,
        w_v.as_ptr() as *const u8,
        q_bias.map_or(std::ptr::null(), |b| b.as_ptr() as *const f32),
        k_bias.map_or(std::ptr::null(), |b| b.as_ptr() as *const f32),
        v_bias.map_or(std::ptr::null(), |b| b.as_ptr() as *const f32),
        input,
        out_q,
        k_cache,
        v_cache,
        h,
        q_size,
        kv_size,
        pos_ptr,
        head_dim,
        theta_base,
        neox,
        stream,
    )?;
    Ok(true)
}

/// Dispatch a fused RMSNorm + QKV + RoPE + KV-Write on an explicit HIP stream.
///
/// Replaces 4 kernel launches (RMSNorm, QKV, RoPE-Q, RoPE-K+KV-Write) with 1.
/// Returns Ok(true) if the fused path was used, Ok(false) if the caller
/// should fall back to separate kernels.
pub fn gpu_dispatch_fused_norm_qkv_rope_kvwrite_on_stream(
    device: &GpuDevice,
    raw_hidden: *const f32,
    norm_weight: *const f32,
    eps: f32,
    w_q: &GpuBuffer,
    q_meta: &WeightMeta,
    q_bias: Option<&GpuBuffer>,
    w_k: &GpuBuffer,
    k_meta: &WeightMeta,
    k_bias: Option<&GpuBuffer>,
    w_v: &GpuBuffer,
    v_meta: &WeightMeta,
    v_bias: Option<&GpuBuffer>,
    out_q: *mut f32,
    k_cache: *mut u16,
    v_cache: *mut u16,
    q_size: usize,
    kv_size: usize,
    h: usize,
    pos_ptr: *const i32,
    head_dim: usize,
    theta_base: f32,
    neox: bool,
    stream: hipStream_t,
) -> GpuResult<bool> {
    if q_meta.wtype != GgmlType::Q4_0
        || k_meta.wtype != GgmlType::Q4_0
        || v_meta.wtype != GgmlType::Q4_0
    {
        return Ok(false);
    }
    // Shared memory: (h + 32) floats must fit in 32KB
    if (q_size % 4) != 0 || (kv_size % 4) != 0 || (h + 32) * std::mem::size_of::<f32>() > 32768 {
        return Ok(false);
    }

    use super::kernels::quant::gemv_norm_qkv_rope_kvwrite_q4_0_f32_on_stream;
    gemv_norm_qkv_rope_kvwrite_q4_0_f32_on_stream(
        raw_hidden,
        norm_weight,
        eps,
        w_q.as_ptr() as *const u8,
        w_k.as_ptr() as *const u8,
        w_v.as_ptr() as *const u8,
        q_bias.map_or(std::ptr::null(), |b| b.as_ptr() as *const f32),
        k_bias.map_or(std::ptr::null(), |b| b.as_ptr() as *const f32),
        v_bias.map_or(std::ptr::null(), |b| b.as_ptr() as *const f32),
        out_q,
        k_cache,
        v_cache,
        h,
        q_size,
        kv_size,
        pos_ptr,
        head_dim,
        theta_base,
        neox,
        stream,
    )?;
    Ok(true)
}

/// Dispatch a fused QKV GEMV with bias on an explicit HIP stream (decode-strict).
///
/// Strictly enforces that Q/K/V weights are all Q4_0. Returns an error for any other type.
pub fn gpu_dispatch_fused_qkv_decode_strict_on_stream(
    device: &GpuDevice,
    w_q: &GpuBuffer,
    q_meta: &WeightMeta,
    q_bias: Option<&GpuBuffer>,
    w_k: &GpuBuffer,
    k_meta: &WeightMeta,
    k_bias: Option<&GpuBuffer>,
    w_v: &GpuBuffer,
    v_meta: &WeightMeta,
    v_bias: Option<&GpuBuffer>,
    input: *const f32,
    out_q: *mut f32,
    out_k: *mut f32,
    out_v: *mut f32,
    q_size: usize,
    kv_size: usize,
    h: usize,
    stream: hipStream_t,
) -> GpuResult<()> {
    if q_meta.wtype != GgmlType::Q4_0
        || k_meta.wtype != GgmlType::Q4_0
        || v_meta.wtype != GgmlType::Q4_0
    {
        return Err(GpuError::UnsupportedWeightType {
            tensor: "decode.fused_qkv".to_string(),
            wtype: q_meta.wtype,
        });
    }

    gpu_dispatch_fused_qkv_on_stream(
        device, w_q, q_meta, q_bias, w_k, k_meta, k_bias, w_v, v_meta, v_bias, input, out_q, out_k,
        out_v, q_size, kv_size, h, stream,
    )
}

/// Dispatch a fused QKV GEMV for a single row.
pub fn gpu_dispatch_fused_qkv(
    device: &GpuDevice,
    w_q: &GpuBuffer,
    q_meta: &WeightMeta,
    q_bias: Option<&GpuBuffer>,
    w_k: &GpuBuffer,
    k_meta: &WeightMeta,
    k_bias: Option<&GpuBuffer>,
    w_v: &GpuBuffer,
    v_meta: &WeightMeta,
    v_bias: Option<&GpuBuffer>,
    input: *const f32,
    out_q: *mut f32,
    out_k: *mut f32,
    out_v: *mut f32,
    q_size: usize,
    kv_size: usize,
    h: usize,
) -> GpuResult<()> {
    gpu_dispatch_fused_qkv_on_stream(
        device,
        w_q,
        q_meta,
        q_bias,
        w_k,
        k_meta,
        k_bias,
        w_v,
        v_meta,
        v_bias,
        input,
        out_q,
        out_k,
        out_v,
        q_size,
        kv_size,
        h,
        hipStream_t::null(),
    )
}

/// Dispatch a fused Gate/Up GEMV + SwiGLU for a single row on an explicit stream.
pub(crate) fn gpu_dispatch_gate_up_raw_on_stream(
    device: &GpuDevice,
    w_gate: &GpuBuffer,
    gate_meta: &WeightMeta,
    w_up: &GpuBuffer,
    up_meta: &WeightMeta,
    input: *const f32,
    gate_output: *mut f32,
    up_output: *mut f32,
    ff_size: usize,
    h: usize,
    stream: hipStream_t,
) -> GpuResult<()> {
    validate_gemv_layout(gate_meta, ff_size, h)?;
    validate_gemv_layout(up_meta, ff_size, h)?;

    if gate_output.is_null() || up_output.is_null() {
        return Err(GpuError::HipApiError {
            code: -1,
            description: "gpu_dispatch_gate_up_raw: output pointers must be non-null".to_string(),
        });
    }
    if gate_output == up_output {
        return Err(GpuError::HipApiError {
            code: -1,
            description: "gpu_dispatch_gate_up_raw: gate and up outputs must be distinct buffers"
                .to_string(),
        });
    }

    if gate_meta.wtype == GgmlType::Q4_0 && up_meta.wtype == GgmlType::Q4_0 {
        if experimental_q8_activation_fastpath_enabled()
            && q8_fastpath_ok(
                "gemv_gate_up_q4_0_q8_0",
                try_q4_0_q8_0_gate_up_fastpath(
                    device,
                    w_gate,
                    w_up,
                    input,
                    gate_output,
                    up_output,
                    h,
                    ff_size,
                    stream,
                ),
            )
        {
            return Ok(());
        }

        gemv_gate_up_q4_0_f32_on_stream(
            w_gate.as_ptr() as *const u8,
            w_up.as_ptr() as *const u8,
            input,
            gate_output,
            up_output,
            h,
            ff_size,
            stream,
        )?;
        return Ok(());
    }

    gpu_dispatch_gemv_on_stream(
        device,
        w_gate,
        gate_meta,
        input,
        gate_output,
        ff_size,
        h,
        stream,
    )?;
    gpu_dispatch_gemv_on_stream(device, w_up, up_meta, input, up_output, ff_size, h, stream)?;
    Ok(())
}

pub(crate) fn gpu_dispatch_fused_gate_up_with_scratch_on_stream(
    device: &GpuDevice,
    w_gate: &GpuBuffer,
    gate_meta: &WeightMeta,
    w_up: &GpuBuffer,
    up_meta: &WeightMeta,
    w_gate_up_interleaved: Option<&GpuBuffer>,
    w_gate_up_interleaved_tile4: Option<&GpuBuffer>,
    input: *const f32,
    gate_scratch: *mut f32,
    output: *mut f32,
    ff_size: usize,
    h: usize,
    stream: hipStream_t,
) -> GpuResult<()> {
    validate_gemv_layout(gate_meta, ff_size, h)?;
    validate_gemv_layout(up_meta, ff_size, h)?;

    if gate_meta.wtype == GgmlType::Q4_0 && up_meta.wtype == GgmlType::Q4_0 {
        if experimental_q8_activation_fastpath_enabled() {
            // Check if stream capture is active - skip autotune benchmarking during capture
            let capture_active = matches!(
                hip_stream_is_capturing(stream),
                Err(_)
                    | Ok(hipStreamCaptureStatus::hipStreamCaptureStatusActive)
                    | Ok(hipStreamCaptureStatus::hipStreamCaptureStatusInvalidated)
            );

            if launch_autotune_enabled() {
                let variant = if capture_active {
                    lookup_gate_up_swiglu_q8_variant(h, ff_size).unwrap_or(VariantId::Baseline)
                } else {
                    select_gate_up_swiglu_q8_variant(h, ff_size, |v| {
                        let result = match v {
                            VariantId::Variant1 => {
                                if let Some(interleaved_tile4) = w_gate_up_interleaved_tile4 {
                                    gemv_gate_up_swiglu_q4_0_f32_q8_inline_interleaved_tile4_on_stream(
                                        interleaved_tile4.as_ptr() as *const u8,
                                        input,
                                        output,
                                        h,
                                        ff_size,
                                        stream,
                                    )
                                } else {
                                    try_q4_0_q8_0_fused_gate_up_fastpath_variant(
                                        w_gate, w_up, input, output, h, ff_size, 1, stream,
                                    )
                                }
                            }
                            VariantId::Variant2 => {
                                if let Some(interleaved) = w_gate_up_interleaved {
                                    gemv_gate_up_swiglu_q4_0_f32_q8_inline_interleaved_on_stream(
                                        interleaved.as_ptr() as *const u8,
                                        input,
                                        output,
                                        h,
                                        ff_size,
                                        stream,
                                    )
                                } else {
                                    try_q4_0_q8_0_fused_gate_up_fastpath_variant(
                                        w_gate, w_up, input, output, h, ff_size, 0, stream,
                                    )
                                }
                            }
                            _ => try_q4_0_q8_0_fused_gate_up_fastpath_variant(
                                w_gate, w_up, input, output, h, ff_size, v as i32, stream,
                            ),
                        };
                        hip_stream_synchronize(stream)?;
                        result
                    })
                };

                let selected_result = match variant {
                    VariantId::Variant1 => {
                        if let Some(interleaved_tile4) = w_gate_up_interleaved_tile4 {
                            gemv_gate_up_swiglu_q4_0_f32_q8_inline_interleaved_tile4_on_stream(
                                interleaved_tile4.as_ptr() as *const u8,
                                input,
                                output,
                                h,
                                ff_size,
                                stream,
                            )
                        } else {
                            try_q4_0_q8_0_fused_gate_up_fastpath_variant(
                                w_gate, w_up, input, output, h, ff_size, 1, stream,
                            )
                        }
                    }
                    VariantId::Variant2 => {
                        if let Some(interleaved) = w_gate_up_interleaved {
                            gemv_gate_up_swiglu_q4_0_f32_q8_inline_interleaved_on_stream(
                                interleaved.as_ptr() as *const u8,
                                input,
                                output,
                                h,
                                ff_size,
                                stream,
                            )
                        } else {
                            try_q4_0_q8_0_fused_gate_up_fastpath_variant(
                                w_gate, w_up, input, output, h, ff_size, 0, stream,
                            )
                        }
                    }
                    _ => try_q4_0_q8_0_fused_gate_up_fastpath_variant(
                        w_gate,
                        w_up,
                        input,
                        output,
                        h,
                        ff_size,
                        variant as i32,
                        stream,
                    ),
                };

                if q8_fastpath_ok("gemv_gate_up_swiglu_q4_0_f32_q8_inline", selected_result) {
                    return Ok(());
                }
            }

            if q8_fastpath_ok(
                "gemv_gate_up_swiglu_q4_0_f32_q8_inline",
                try_q4_0_q8_0_fused_gate_up_fastpath(
                    w_gate, w_up, input, output, h, ff_size, stream,
                ),
            ) {
                return Ok(());
            }
        }

        unsafe {
            gemv_gate_up_swiglu_q4_0_f32_on_stream(
                w_gate.as_ptr() as *const u8,
                w_up.as_ptr() as *const u8,
                input,
                output,
                h,
                ff_size,
                stream,
            )?;
        }
        return Ok(());
    }

    if gate_scratch.is_null() {
        return Err(GpuError::HipApiError {
            code: -1,
            description: "gpu_dispatch_fused_gate_up: gate scratch pointer must be non-null"
                .to_string(),
        });
    }
    if gate_scratch == output {
        return Err(GpuError::HipApiError {
            code: -1,
            description:
                "gpu_dispatch_fused_gate_up: gate scratch and output must be distinct buffers"
                    .to_string(),
        });
    }

    gpu_dispatch_gate_up_raw_on_stream(
        device,
        w_gate,
        gate_meta,
        w_up,
        up_meta,
        input,
        gate_scratch,
        output,
        ff_size,
        h,
        stream,
    )?;
    silu_on_stream(gate_scratch, gate_scratch, ff_size, stream)?;
    mul_on_stream(gate_scratch, output, output, ff_size, stream)?;
    Ok(())
}

/// Dispatch a fused RMSNorm + Gate/Up + SwiGLU on an explicit HIP stream.
///
/// Replaces 2 kernel launches (RMSNorm, Gate+Up+SiLU) with 1.
/// Returns Ok(true) if the fused path was used, Ok(false) to fall back.
pub fn gpu_dispatch_fused_norm_gate_up_on_stream(
    device: &GpuDevice,
    raw_hidden: *const f32,
    norm_weight: *const f32,
    eps: f32,
    w_gate: &GpuBuffer,
    gate_meta: &WeightMeta,
    w_up: &GpuBuffer,
    up_meta: &WeightMeta,
    output: *mut f32,
    ff_size: usize,
    h: usize,
    stream: hipStream_t,
) -> GpuResult<bool> {
    if gate_meta.wtype != GgmlType::Q4_0 || up_meta.wtype != GgmlType::Q4_0 {
        return Ok(false);
    }
    if (h % 32) != 0 || (ff_size % 4) != 0 {
        return Ok(false);
    }
    // Check shared memory: max of norm phase and gemv phase
    let n_blocks = h / 32;
    let norm_shared = (h + 32) * std::mem::size_of::<f32>();
    let gemv_shared = n_blocks * 34; // Q8_0_BLOCK_SIZE = 34
    let shared_mem = norm_shared.max(gemv_shared);
    if shared_mem > 32768 {
        return Ok(false);
    }

    use super::kernels::quant::gemv_norm_gate_up_swiglu_q4_0_f32_on_stream;
    gemv_norm_gate_up_swiglu_q4_0_f32_on_stream(
        raw_hidden,
        norm_weight,
        eps,
        w_gate.as_ptr() as *const u8,
        w_up.as_ptr() as *const u8,
        output,
        h,
        ff_size,
        stream,
    )?;
    Ok(true)
}

/// Dispatch a fused Gate/Up GEMV + SwiGLU for a single row on an explicit stream.
pub fn gpu_dispatch_fused_gate_up_on_stream(
    device: &GpuDevice,
    w_gate: &GpuBuffer,
    gate_meta: &WeightMeta,
    w_up: &GpuBuffer,
    up_meta: &WeightMeta,
    w_gate_up_interleaved: Option<&GpuBuffer>,
    w_gate_up_interleaved_tile4: Option<&GpuBuffer>,
    input: *const f32,
    output: *mut f32,
    ff_size: usize,
    h: usize,
    stream: hipStream_t,
) -> GpuResult<()> {
    if gate_meta.wtype == GgmlType::Q4_0 && up_meta.wtype == GgmlType::Q4_0 {
        return gpu_dispatch_fused_gate_up_with_scratch_on_stream(
            device,
            w_gate,
            gate_meta,
            w_up,
            up_meta,
            w_gate_up_interleaved,
            w_gate_up_interleaved_tile4,
            input,
            std::ptr::null_mut(),
            output,
            ff_size,
            h,
            stream,
        );
    }

    let gate_scratch = GpuBuffer::alloc(ff_size * std::mem::size_of::<f32>())?;
    gpu_dispatch_fused_gate_up_with_scratch_on_stream(
        device,
        w_gate,
        gate_meta,
        w_up,
        up_meta,
        w_gate_up_interleaved,
        w_gate_up_interleaved_tile4,
        input,
        gate_scratch.as_ptr() as *mut f32,
        output,
        ff_size,
        h,
        stream,
    )
}

/// Dispatch a fused Gate/Up GEMV + SwiGLU for a single row.
pub fn gpu_dispatch_fused_gate_up(
    device: &GpuDevice,
    w_gate: &GpuBuffer,
    gate_meta: &WeightMeta,
    w_up: &GpuBuffer,
    up_meta: &WeightMeta,
    w_gate_up_interleaved: Option<&GpuBuffer>,
    w_gate_up_interleaved_tile4: Option<&GpuBuffer>,
    input: *const f32,
    output: *mut f32,
    ff_size: usize,
    h: usize,
) -> GpuResult<()> {
    gpu_dispatch_fused_gate_up_on_stream(
        device,
        w_gate,
        gate_meta,
        w_up,
        up_meta,
        w_gate_up_interleaved,
        w_gate_up_interleaved_tile4,
        input,
        output,
        ff_size,
        h,
        hipStream_t::null(),
    )
}

/// Minimum `seq_len` (= WMMA M dimension) at which the WMMA Q4_0 prefill
/// kernel takes over. The kernel requires 64-aligned M; below this the
/// hipBLAS path or the batched-GEMV path handles it.
const WMMA_PREFILL_MIN_M: usize = 64;

/// Dispatch a GPU GEMM for GGUF weights.
pub fn gpu_dispatch_gemm(
    device: &GpuDevice,
    weights: &GpuBuffer,
    meta: &WeightMeta,
    input: *const f32,
    output: *mut f32,
    out_dim: usize,
    in_dim: usize,
    seq_len: usize,
) -> GpuResult<()> {
    if seq_len == 1 && supports_gemv_type(meta.wtype) {
        return gpu_dispatch_gemv(device, weights, meta, input, output, out_dim, in_dim);
    }

    // Phase 2d — preferred prefill path for Q4_0: hand-written WMMA kernel
    // with inline Q4_0 dequant. Kernel requires M (=seq_len) ≥ 64 and both
    // N (out_dim) and K (in_dim) multiples of 64 / 32 respectively.
    //
    // Ordering: WMMA → hipBLAS → custom GEMM → batched/tiled GEMV. Each
    // step has its own opt-out env flag.
    if seq_len >= WMMA_PREFILL_MIN_M
        && (seq_len % 64) == 0
        && meta.wtype == GgmlType::Q4_0
        && !meta.needs_transpose
        && (out_dim % 64) == 0
        && (in_dim % 32) == 0
        && super::safety::wmma_prefill_enabled()
    {
        match super::kernels::wmma::launch_wmma_gemm_q4_0(
            input,
            weights.as_ptr() as *const u8,
            output,
            seq_len,
            out_dim,
            in_dim,
            super::ffi::hipStream_t::null(),
        ) {
            Ok(()) => return Ok(()),
            Err(err) => {
                eprintln!(
                    "[rocmforge] WMMA Q4_0 prefill path failed ({}), falling back to hipBLAS/GEMV",
                    err
                );
                // fall through to the hipBLAS path
            }
        }
    }

    // hipBLAS prefill path: dequantise the Q4_0 weight tensor to FP16,
    // convert FP32 activations to FP16, call hipblasHgemm, convert the
    // output back to FP32. The threshold keeps short prefills on the
    // faster batched-GEMV path.
    if seq_len >= super::prefill_gemm::PREFILL_GEMM_THRESHOLD
        && meta.wtype == GgmlType::Q4_0
        && !meta.needs_transpose
        && super::safety::hipblas_prefill_enabled()
    {
        match dispatch_prefill_via_hipblas(
            device, weights, input, output, out_dim, in_dim, seq_len,
        ) {
            Ok(()) => return Ok(()),
            Err(err) => {
                eprintln!(
                    "[rocmforge] hipBLAS prefill path failed ({}), falling back to GEMV",
                    err
                );
                // fall through to the original path
            }
        }
    }

    // For small batch sizes (2-8), use batched GEMV: single weight load, N dot products.
    // This reads quantized weights once from VRAM instead of N times, saving N× bandwidth.
    // LDS limit: batch_size × (in_dim / 32) × 34 bytes must fit in 32 KB.
    const BATCHED_GEMV_LDS_LIMIT: usize = 32 * 1024;

    if seq_len <= 8 && meta.wtype == GgmlType::Q4_0 {
        let lds_bytes = seq_len * (in_dim / 32) * 34;
        if lds_bytes <= BATCHED_GEMV_LDS_LIMIT {
            return gemv_q4_0_f32_batched_on_stream(
                weights.as_ptr() as *const u8,
                input,
                output,
                in_dim,
                out_dim,
                seq_len,
                hipStream_t::null(),
            );
        }
        // Tiled batched GEMV: tiles along input dimension for large in_dim
        // (e.g., FFN down-projection in_dim=18944). Preserves single weight load
        // while fitting input quantization tiles into LDS.
        if super::safety::tiled_gemv_enabled() {
            return gemv_q4_0_f32_batched_tiled_on_stream(
                weights.as_ptr() as *const u8,
                input,
                output,
                in_dim,
                out_dim,
                seq_len,
                hipStream_t::null(),
            );
        }
    }

    // Fallback to sequential GEMV for large inputs or other quantization types
    if seq_len <= 8 && supports_gemv_type(meta.wtype) {
        for row in 0..seq_len {
            let row_input = unsafe { input.add(row * in_dim) };
            let row_output = unsafe { output.add(row * out_dim) };
            gpu_dispatch_gemv(device, weights, meta, row_input, row_output, out_dim, in_dim)?;
        }
        return Ok(());
    }

    unsafe {
        match meta.wtype {
            GgmlType::Q4_0 => gemm_q4_0_f32(
                weights.as_ptr() as *const u8,
                input,
                output,
                in_dim,
                out_dim,
                seq_len,
            )?,
            GgmlType::Q4_1 => gemm_q4_1_f32(
                weights.as_ptr() as *const u8,
                input,
                output,
                in_dim,
                out_dim,
                seq_len,
            )?,
            GgmlType::Q8_0 => gemm_q8_0_f32(
                weights.as_ptr() as *const u8,
                input,
                output,
                in_dim,
                out_dim,
                seq_len,
            )?,
            GgmlType::Q4_K => gemm_q4_k_f32(
                weights.as_ptr() as *const u8,
                input,
                output,
                in_dim,
                out_dim,
                seq_len,
            )?,
            GgmlType::Q5_K => gemm_q5_k_f32(
                weights.as_ptr() as *const u8,
                input,
                output,
                in_dim,
                out_dim,
                seq_len,
            )?,
            _ => {
                return Err(GpuError::UnsupportedWeightType {
                    tensor: "gpu_dispatch_gemm".to_string(),
                    wtype: meta.wtype,
                })
            }
        }
    }

    Ok(())
}

fn dispatch_prefill_via_hipblas(
    device: &GpuDevice,
    weights: &GpuBuffer,
    input: *const f32,
    output: *mut f32,
    out_dim: usize,
    in_dim: usize,
    seq_len: usize,
) -> GpuResult<()> {
    let stream = device.stream();
    let handle = device.hipblas()?;

    let weight_elements = out_dim * in_dim;
    let act_in_elements = seq_len * in_dim;
    let act_out_elements = seq_len * out_dim;

    let weight_f16 =
        device.prefill_f16_weight(weight_elements * std::mem::size_of::<u16>())? as *mut u8;
    let act_in_f16 =
        device.prefill_f16_act_in(act_in_elements * std::mem::size_of::<u16>())? as *mut u8;
    let act_out_f16 =
        device.prefill_f16_act_out(act_out_elements * std::mem::size_of::<u16>())? as *mut u8;

    // 1. Q4_0 → FP16 dequant of the weight matrix.
    super::prefill_gemm::dequantize_q4_0_to_f16_on_stream(
        weights.as_ptr() as *const u8,
        weight_f16,
        weight_elements,
        stream,
    )?;

    // 2. FP32 → FP16 conversion of the input activations.
    super::prefill_gemm::convert_f32_to_f16_on_stream(input, act_in_f16, act_in_elements, stream)?;

    // 3. hipblasHgemm with the "compute C^T via swapped operands" trick.
    super::prefill_gemm::hgemm_row_major(
        handle,
        weight_f16 as *const u16,
        act_in_f16 as *const u16,
        act_out_f16 as *mut u16,
        seq_len,
        in_dim,
        out_dim,
    )?;

    // 4. FP16 → FP32 conversion of the output.
    super::prefill_gemm::convert_f16_to_f32_on_stream(
        act_out_f16,
        output,
        act_out_elements,
        stream,
    )?;

    Ok(())
}
