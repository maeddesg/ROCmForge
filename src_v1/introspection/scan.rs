//! `introspect()` — the one-shot magnitude + SNR scan.
//!
//! Reads weights directly from the GGUF mmap via
//! `GGUFFile::tensor_data_full` (zero-copy) and uses the
//! Phase-1 CPU dequant interpreter (src_v1/ir/interpreter.rs,
//! Schritt 1.6) as the ground-truth FP32 source.
//!
//! Cost budget: < 5 s for an 8B model on a 7945HX.
//!   * Embedding: **every** row is dequantised because we need to
//!     identify critical tokens. At vocab=151936 × hidden=4096 with
//!     Q4_K (256 values per block) that's ~2.4 M blocks.
//!   * Layer weights: **sample** of 1024 values per tensor (4 Q4_K
//!     blocks) — plenty for mean/max/std; full dequant would blow
//!     the budget.
//!
//! Phase 1 only measures. Phase 2's precision-GA consumes the
//! output as its seed population.

use std::time::Instant;

use super::super::core::gguf::GGUFFile;
use super::super::core::tensor_info::{GgmlType, TensorInfo, TensorRole, parse_tensor_name};
use super::super::ir::{formats, interpreter::dequant_block, types::QuantFormat};
use super::profile::{LayerStats, ModelProfile, PrecisionHint, TokenId};

/// Number of values dequantised per tensor during the layer
/// Stichprobe. 1024 fits 4 Q4_K super-blocks or 32 Q4_0/Q8_0 blocks
/// — enough for a stable mean/std estimate without touching the
/// whole tensor.
const SAMPLE_VALUES_PER_TENSOR: usize = 1024;

/// Per-value dequant-noise ratio, relative to the mean magnitude
/// of the quantised tensor. Derived from the quant-spec §3 scale-
/// encoding width (4-bit → ~1/16, 6-bit → ~1/64, 8-bit → ~1/256).
/// Q4_K and Q4_0 share the 4-bit quant grain; their ratios differ
/// only in how the scales themselves are stored, which is a
/// second-order effect for magnitude noise.
fn per_value_noise_ratio(fmt: GgmlType) -> f32 {
    match fmt {
        GgmlType::Q4_0 | GgmlType::Q4_1 | GgmlType::Q4_K => 1.0 / 16.0,
        GgmlType::Q5_0 | GgmlType::Q5_1 | GgmlType::Q5_K => 1.0 / 32.0,
        GgmlType::Q6_K => 1.0 / 64.0,
        GgmlType::Q8_0 | GgmlType::Q8_K => 1.0 / 256.0,
        _ => 0.0, // F32 / F16 / BF16 — no quantisation noise
    }
}

fn quant_format_for(fmt: GgmlType) -> Option<QuantFormat> {
    match fmt {
        GgmlType::Q4_0 => Some(formats::q4_0()),
        GgmlType::Q4_1 => Some(formats::q4_1()),
        GgmlType::Q4_K => Some(formats::q4_k()),
        GgmlType::Q6_K => Some(formats::q6_k()),
        GgmlType::Q8_0 => Some(formats::q8_0()),
        _ => None,
    }
}

/// Find the embedding tensor. Phase-1 models use one of two names:
/// `token_embd.weight` (standard GGUF) or `tok_embeddings.weight`
/// (legacy Llama). We try the role-based lookup first and fall
/// back to substring matching so exotic namings still land.
fn find_embedding<'a>(gguf: &'a GGUFFile) -> Option<&'a TensorInfo> {
    for t in gguf.tensors() {
        let (role, layer) = parse_tensor_name(&t.name);
        if layer.is_none() && matches!(role, TensorRole::Embedding) {
            return Some(t);
        }
    }
    gguf.tensors()
        .iter()
        .find(|t| t.name.contains("token_embd") || t.name.contains("tok_embeddings"))
}

fn is_layer_weight_role(role: &TensorRole) -> bool {
    matches!(
        role,
        TensorRole::AttentionQ
            | TensorRole::AttentionK
            | TensorRole::AttentionV
            | TensorRole::AttentionOutput
            | TensorRole::FFNGate
            | TensorRole::FFNUp
            | TensorRole::FFNDown
    )
}

/// Compute L2 norm of a single embedding row by streaming blocks.
/// `row_bytes` is the byte slice for *one row*; `format` provides
/// the block layout and dequant program.
fn row_l2(row_bytes: &[u8], format: &QuantFormat) -> f32 {
    let mut acc = 0.0f64;
    let block_bytes = format.block_bytes;
    let n_blocks = row_bytes.len() / block_bytes;
    for b in 0..n_blocks {
        let start = b * block_bytes;
        let end = start + block_bytes;
        let values = match dequant_block(format, &row_bytes[start..end]) {
            Ok(v) => v,
            Err(_) => return f32::NAN,
        };
        for v in values {
            acc += (v as f64) * (v as f64);
        }
    }
    acc.sqrt() as f32
}

/// Sample-scan a single tensor: dequant the first ~1024 values,
/// return (mean_abs, max_abs, std_abs, element_count_sampled).
fn sample_stats(tensor_bytes: &[u8], format: &QuantFormat) -> (f32, f32, f32, usize) {
    let block_bytes = format.block_bytes;
    let elems_per_block = format.elements_per_block;
    let n_blocks_needed =
        (SAMPLE_VALUES_PER_TENSOR + elems_per_block - 1) / elems_per_block;
    let n_blocks_avail = tensor_bytes.len() / block_bytes;
    let n_blocks = n_blocks_needed.min(n_blocks_avail);

    let mut values = Vec::with_capacity(n_blocks * elems_per_block);
    for b in 0..n_blocks {
        let start = b * block_bytes;
        let end = start + block_bytes;
        if let Ok(v) = dequant_block(format, &tensor_bytes[start..end]) {
            values.extend(v);
        }
    }
    if values.is_empty() {
        return (0.0, 0.0, 0.0, 0);
    }
    let mut sum_abs = 0.0f64;
    let mut max_abs = 0.0f32;
    for v in &values {
        let a = v.abs();
        sum_abs += a as f64;
        if a > max_abs {
            max_abs = a;
        }
    }
    let mean_abs = (sum_abs / values.len() as f64) as f32;
    let mut sum_sq_dev = 0.0f64;
    for v in &values {
        let d = v.abs() - mean_abs;
        sum_sq_dev += (d as f64) * (d as f64);
    }
    let std_abs = (sum_sq_dev / values.len() as f64).sqrt() as f32;
    (mean_abs, max_abs, std_abs, values.len())
}

/// Phase-1 introspection. See the module header for the cost
/// budget and the `ModelProfile` docstrings for field semantics.
pub fn introspect(gguf: &GGUFFile) -> ModelProfile {
    let start = Instant::now();

    // ── 1. Embedding scan ────────────────────────────────────────────
    let mut embedding_min = f32::INFINITY;
    let mut embedding_max: f32 = 0.0;
    let mut row_l2s: Vec<f32> = Vec::new();
    /// Mean absolute value across the embedding's per-element samples.
    /// Used for the noise estimate — architecture_v1.2.0-draft §2.2
    /// computes Q4_K noise as `noise_ratio × mean_magnitude` and the
    /// "magnitude" there refers to the embedding element magnitude
    /// (the signal being quantised), not the layer-weight magnitude.
    let mut embedding_mean_abs: f32 = 0.0;
    let mut embedding_format: Option<GgmlType> = None;

    if let Some(emb) = find_embedding(gguf) {
        embedding_format = Some(emb.ggml_type);
        if let Some(format) = quant_format_for(emb.ggml_type) {
            let bytes = gguf
                .tensor_data_full(emb)
                .expect("embedding tensor data within mmap");
            let hidden = *emb.shape.first().unwrap_or(&0) as usize;
            let vocab = *emb.shape.get(1).unwrap_or(&0) as usize;
            let row_bytes_len = hidden / format.elements_per_block * format.block_bytes;
            row_l2s.reserve(vocab);
            // Running sum of |x| over all dequantised embedding values
            // — one pass, no extra allocations beyond the per-row
            // dequant buffer that `row_l2` already creates.
            let mut sum_abs_all = 0.0f64;
            let mut elem_count_all: u64 = 0;
            for row in 0..vocab {
                let offset = row * row_bytes_len;
                let slice = &bytes[offset..offset + row_bytes_len];
                // Reuse dequant loop: compute both L2 and sum|x|
                // in a single block-stream pass so we don't re-visit
                // the mmap twice.
                let mut acc_sq = 0.0f64;
                let mut acc_abs = 0.0f64;
                let block_bytes = format.block_bytes;
                let n_blocks = slice.len() / block_bytes;
                for b in 0..n_blocks {
                    let bstart = b * block_bytes;
                    let bend = bstart + block_bytes;
                    match dequant_block(&format, &slice[bstart..bend]) {
                        Ok(vs) => {
                            for v in vs {
                                acc_sq += (v as f64) * (v as f64);
                                acc_abs += (v as f64).abs();
                                elem_count_all += 1;
                            }
                        }
                        Err(_) => {
                            acc_sq = f64::NAN;
                            break;
                        }
                    }
                }
                sum_abs_all += acc_abs;
                let l2 = (acc_sq as f32).sqrt();
                row_l2s.push(l2);
                if l2.is_finite() {
                    if l2 < embedding_min {
                        embedding_min = l2;
                    }
                    if l2 > embedding_max {
                        embedding_max = l2;
                    }
                }
            }
            if elem_count_all > 0 {
                embedding_mean_abs = (sum_abs_all / elem_count_all as f64) as f32;
            }
        } else {
            // Non-quant embedding (F16/F32/BF16) — read as FP32 directly.
            let bytes = gguf
                .tensor_data_full(emb)
                .expect("embedding tensor data within mmap");
            let hidden = *emb.shape.first().unwrap_or(&0) as usize;
            let vocab = *emb.shape.get(1).unwrap_or(&0) as usize;
            let bytes_per_elem = match emb.ggml_type {
                GgmlType::F32 => 4,
                GgmlType::F16 | GgmlType::BF16 => 2,
                _ => 0,
            };
            if bytes_per_elem != 0 {
                let mut sum_abs_all = 0.0f64;
                let mut elem_count_all: u64 = 0;
                for row in 0..vocab {
                    let mut acc_sq = 0.0f64;
                    let mut acc_abs = 0.0f64;
                    let base = row * hidden * bytes_per_elem;
                    for i in 0..hidden {
                        let off = base + i * bytes_per_elem;
                        let v = match emb.ggml_type {
                            GgmlType::F32 => f32::from_le_bytes([
                                bytes[off],
                                bytes[off + 1],
                                bytes[off + 2],
                                bytes[off + 3],
                            ]),
                            GgmlType::F16 => half::f16::from_le_bytes([
                                bytes[off],
                                bytes[off + 1],
                            ])
                            .to_f32(),
                            GgmlType::BF16 => {
                                let raw = u16::from_le_bytes([bytes[off], bytes[off + 1]]);
                                f32::from_bits((raw as u32) << 16)
                            }
                            _ => 0.0,
                        };
                        acc_sq += (v as f64) * (v as f64);
                        acc_abs += (v as f64).abs();
                        elem_count_all += 1;
                    }
                    sum_abs_all += acc_abs;
                    let l2 = acc_sq.sqrt() as f32;
                    row_l2s.push(l2);
                    if l2 < embedding_min {
                        embedding_min = l2;
                    }
                    if l2 > embedding_max {
                        embedding_max = l2;
                    }
                }
                if elem_count_all > 0 {
                    embedding_mean_abs = (sum_abs_all / elem_count_all as f64) as f32;
                }
            }
        }
    }
    if !embedding_min.is_finite() {
        embedding_min = 0.0;
    }
    let embedding_magnitude_range = (embedding_min, embedding_max);

    // Mean L2 + critical threshold.
    let mean_l2: f32 = if row_l2s.is_empty() {
        0.0
    } else {
        let s: f64 = row_l2s.iter().map(|x| *x as f64).sum();
        (s / row_l2s.len() as f64) as f32
    };
    let critical_threshold = 0.1 * mean_l2;
    let critical_embedding_tokens: Vec<TokenId> = row_l2s
        .iter()
        .enumerate()
        .filter(|(_, l2)| l2.is_finite() && **l2 < critical_threshold && **l2 > 0.0)
        .map(|(idx, _)| idx as TokenId)
        .collect();

    // ── 2. Layer Stichprobe ──────────────────────────────────────────
    let mut layer_magnitude_stats: Vec<LayerStats> = Vec::new();
    let mut global_mean_abs_sum = 0.0f64;
    let mut global_mean_abs_count = 0usize;
    let mut dominant_format: Option<GgmlType> = None;
    let mut format_counter: std::collections::HashMap<GgmlType, usize> =
        std::collections::HashMap::new();

    for tensor in gguf.tensors() {
        let (role, layer_idx) = parse_tensor_name(&tensor.name);
        if !is_layer_weight_role(&role) {
            continue;
        }
        let Some(layer_index) = layer_idx else { continue };
        let Some(format) = quant_format_for(tensor.ggml_type) else {
            continue;
        };
        let bytes = match gguf.tensor_data_full(tensor) {
            Ok(b) => b,
            Err(_) => continue,
        };
        let (mean_abs, max_abs, std_abs, count) = sample_stats(bytes, &format);
        if count == 0 {
            continue;
        }
        global_mean_abs_sum += mean_abs as f64;
        global_mean_abs_count += 1;
        *format_counter.entry(tensor.ggml_type).or_insert(0) += 1;

        layer_magnitude_stats.push(LayerStats {
            layer_index,
            tensor_name: tensor.name.clone(),
            mean_abs,
            max_abs,
            std_abs,
            element_count: count,
        });
    }
    layer_magnitude_stats.sort_by_key(|s| (s.layer_index, s.tensor_name.clone()));

    // Pick the most common quant format for the noise estimate. Ties
    // break toward the one with the higher noise ratio (pessimistic).
    let mut best: Option<(GgmlType, usize)> = None;
    for (k, v) in &format_counter {
        match best {
            None => best = Some((*k, *v)),
            Some((bk, bv)) => {
                if *v > bv
                    || (*v == bv && per_value_noise_ratio(*k) > per_value_noise_ratio(bk))
                {
                    best = Some((*k, *v));
                }
            }
        }
    }
    dominant_format = best.map(|(k, _)| k);

    let global_mean_abs = if global_mean_abs_count > 0 {
        (global_mean_abs_sum / global_mean_abs_count as f64) as f32
    } else {
        0.0
    };

    // ── 3. Noise estimate ────────────────────────────────────────────
    //
    // Per Arch-Doc §2.2, noise L2 ≈ per_value_noise_ratio ×
    // embedding_mean_abs × sqrt(hidden). Using the embedding's own
    // mean |x| (rather than the layer mean) matches the doc's
    // reference example for Q4_K: 4096-dim row, special-token L2
    // ≈ 0.034, noise ≈ 0.064. Falls back to the layer mean and the
    // dominant layer format only when the embedding itself is F32
    // (no quantisation noise of its own).
    let emb_noise_format = embedding_format.filter(|f| per_value_noise_ratio(*f) > 0.0);
    let (noise_ratio, magnitude_for_noise) = match (emb_noise_format, embedding_mean_abs > 0.0) {
        (Some(f), true) => (per_value_noise_ratio(f), embedding_mean_abs),
        _ => (
            dominant_format.map(per_value_noise_ratio).unwrap_or(0.0),
            global_mean_abs,
        ),
    };
    let hidden_guess = if let Some(emb) = find_embedding(gguf) {
        *emb.shape.first().unwrap_or(&4096) as f32
    } else {
        4096.0
    };
    let quantization_noise_estimate = noise_ratio * magnitude_for_noise * hidden_guess.sqrt();

    // ── 4. SNR risk score ────────────────────────────────────────────
    let min_magnitude = if !critical_embedding_tokens.is_empty() {
        critical_embedding_tokens
            .iter()
            .filter_map(|&t| row_l2s.get(t as usize).copied())
            .filter(|l2| l2.is_finite() && *l2 > 0.0)
            .fold(f32::INFINITY, f32::min)
    } else if !layer_magnitude_stats.is_empty() {
        layer_magnitude_stats
            .iter()
            .map(|s| s.mean_abs)
            .fold(f32::INFINITY, f32::min)
    } else {
        0.0
    };
    let snr_risk_score = if quantization_noise_estimate > 0.0 && min_magnitude.is_finite() {
        min_magnitude / quantization_noise_estimate
    } else {
        f32::INFINITY
    };

    // ── 5. Per-layer precision recommendation ────────────────────────
    //
    // Algorithm per Arch-Doc §2.2 with the Phase-1 default raised to
    // Fp8E4M3 (matches PrecisionHint::Fp8E4M3's docstring as the v1.1
    // weight path). The GA in Phase 2 refines this seed.
    let precision_recommendation: Vec<PrecisionHint> = layer_magnitude_stats
        .iter()
        .map(|s| {
            let dyn_range = if s.mean_abs > 0.0 {
                s.max_abs / s.mean_abs
            } else {
                1.0
            };
            if dyn_range > 50.0 {
                PrecisionHint::Fp32Scales
            } else if snr_risk_score < 2.0 {
                PrecisionHint::Bf16Scales
            } else {
                PrecisionHint::Fp8E4M3
            }
        })
        .collect();

    let elapsed = start.elapsed();
    eprintln!(
        "[introspect] scanned {} tensors, {} embedding rows, {:.2}s",
        layer_magnitude_stats.len(),
        row_l2s.len(),
        elapsed.as_secs_f64()
    );

    ModelProfile {
        embedding_magnitude_range,
        critical_embedding_tokens,
        layer_magnitude_stats,
        quantization_noise_estimate,
        snr_risk_score,
        precision_recommendation,
    }
}
