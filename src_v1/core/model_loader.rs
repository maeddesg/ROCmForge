//! End-to-end GGUF → VRAM model loader.
//!
//! Flow:
//!   1. Parse GGUF (header, KV, tensor descriptors, mmap) via
//!      [`GGUFFile::open`].
//!   2. Build [`ModelConfig`] from metadata + tensor inventory.
//!   3. Plan the [`ArenaConfig`] from model shape and device VRAM.
//!   4. Allocate the [`VramArena`].
//!   5. For every **supported** tensor, `alloc_weights(size)` in Zone A
//!      and `hipMemcpy` the disk bytes into the 256-byte-aligned slice.
//!   6. Return a [`LoadedModel`] whose `tensor_map` stores only arena
//!      offsets — GGUF file offsets are not retained.
//!
//! Strict invariant: after [`LoadedModel::load`] returns, nothing
//! downstream needs to know about the GGUF file's byte layout.

#[cfg(feature = "gpu")]
use crate::hip_check;

use std::collections::HashMap;
use std::path::Path;

#[cfg(feature = "gpu")]
use crate::v1::backend::gpu::arena::{ArenaConfig, ArenaSlice, VramArena};
#[cfg(feature = "gpu")]
use crate::v1::backend::gpu::device::GpuDevice;
#[cfg(feature = "gpu")]
use crate::v1::backend::gpu::error::{HipError, HipResult};
#[cfg(feature = "gpu")]
use crate::v1::backend::gpu::hip_ffi::{hipMemcpy, hipMemcpyHostToDevice};

use super::gguf::GGUFFile;
use super::model_config::ModelConfig;
use super::tensor_info::{
    group_tensors_by_layer, parse_tensor_name, validate_quant_types, LayerTensors, TensorRole,
};

/// Result of [`LoadedModel::load`]: everything a runtime needs to
/// dispatch a forward pass.
#[cfg(feature = "gpu")]
pub struct LoadedModel {
    pub config: ModelConfig,
    pub arena: VramArena,
    /// Tensor name → VRAM offset within the arena. `file_offset` from
    /// the GGUF is **not** retained past `load()`.
    pub tensor_map: HashMap<String, ArenaSlice>,
    pub layers: Vec<LayerSlices>,
    pub global_tensors: HashMap<TensorRole, ArenaSlice>,
    pub warnings: Vec<String>,
}

/// VRAM-resident counterpart of [`LayerTensors`]: role → arena slice
/// for one transformer block.
#[cfg(feature = "gpu")]
#[derive(Debug, Clone, Default)]
pub struct LayerSlices {
    pub layer_idx: usize,
    pub slices: HashMap<TensorRole, ArenaSlice>,
}

// Free margin above the estimated weight size — covers the imprecise
// `from_model` sizing when KV/scratch share the arena.
const SAFETY_MARGIN_BYTES: usize = 512 * 1024 * 1024;

#[cfg(feature = "gpu")]
impl LoadedModel {
    pub fn load(path: impl AsRef<Path>, device: &GpuDevice) -> HipResult<Self> {
        let gguf = GGUFFile::open(&path).map_err(|e| HipError {
            code: -1,
            message: format!("GGUF parse failed: {e}"),
            context: "LoadedModel::load".to_string(),
        })?;

        let config = ModelConfig::from_metadata(gguf.metadata(), gguf.tensors()).map_err(|e| {
            HipError {
                code: -1,
                message: format!("ModelConfig failed: {e}"),
                context: "LoadedModel::load".to_string(),
            }
        })?;

        let warnings = validate_quant_types(gguf.tensors());
        for w in &warnings {
            tracing::warn!("{w}");
        }

        let arena_config = plan_arena(&config, &gguf, device);
        let mut arena = VramArena::new(arena_config)?;

        // Upload all supported tensors sequentially. Tensors whose
        // quant type we don't implement yet are skipped with a warning
        // (validate_quant_types already surfaced them).
        let mut tensor_map: HashMap<String, ArenaSlice> = HashMap::new();
        let grouped: Vec<LayerTensors> = group_tensors_by_layer(gguf.tensors());
        for tensor in gguf.tensors() {
            if !tensor.ggml_type.is_supported() {
                continue;
            }
            let slice = arena.alloc_weights(tensor.byte_size as usize).map_err(|e| {
                HipError {
                    code: -1,
                    message: format!("Zone-A alloc failed for '{}': {e}", tensor.name),
                    context: "LoadedModel::load".to_string(),
                }
            })?;

            let src = gguf.tensor_data_full(tensor).map_err(|e| HipError {
                code: -1,
                message: format!("tensor data for '{}' out of bounds: {e}", tensor.name),
                context: "LoadedModel::load".to_string(),
            })?;

            let base = arena.base_mut_ptr();
            hip_check!(
                hipMemcpy(
                    slice.as_mut_ptr(base),
                    src.as_ptr() as *const _,
                    tensor.byte_size as usize,
                    hipMemcpyHostToDevice,
                ),
                &format!("hipMemcpy H2D for tensor '{}'", tensor.name)
            )?;

            tensor_map.insert(tensor.name.clone(), slice);
        }

        // Build VRAM-side layer view from the tensor_map.
        let mut layers: Vec<LayerSlices> = Vec::with_capacity(grouped.len());
        for layer in &grouped {
            let mut slices: HashMap<TensorRole, ArenaSlice> = HashMap::new();
            for (role, info) in &layer.tensors {
                if let Some(slice) = tensor_map.get(&info.name) {
                    slices.insert(role.clone(), *slice);
                }
            }
            layers.push(LayerSlices {
                layer_idx: layer.layer_idx,
                slices,
            });
        }

        // Global tensors: look them up by name, map to their arena slice.
        let mut global_tensors: HashMap<TensorRole, ArenaSlice> = HashMap::new();
        for tensor in gguf.tensors() {
            let (role, layer) = parse_tensor_name(&tensor.name);
            if layer.is_some() {
                continue;
            }
            if matches!(role, TensorRole::Unknown(_)) {
                continue;
            }
            if let Some(slice) = tensor_map.get(&tensor.name) {
                global_tensors.insert(role, *slice);
            }
        }

        Ok(Self {
            config,
            arena,
            tensor_map,
            layers,
            global_tensors,
            warnings,
        })
    }
}

/// Plan arena zone sizes from model shape + device VRAM.
///
/// Uses the KV-element size of 2 (FP16) and a conservative
/// `max_batch_size = 1` for decode. Zone B (KV) and Zone C (scratch)
/// get what is left after Zone A; an unmet total surfaces as a
/// descriptive `validate()` error when `VramArena::new` runs.
#[cfg(feature = "gpu")]
fn plan_arena(config: &ModelConfig, gguf: &GGUFFile, device: &GpuDevice) -> ArenaConfig {
    // Sum supported-tensor byte sizes as the weight budget.
    let weights_bytes: u64 = gguf
        .tensors()
        .iter()
        .filter(|t| t.ggml_type.is_supported())
        .map(|t| t.byte_size)
        .sum();

    // VRAM budget: device-reported total minus safety margin.
    let total_vram = device
        .total_memory
        .saturating_sub(SAFETY_MARGIN_BYTES);

    // Phase-1 uses a single-token decode path: batch=1, cap context at
    // what metadata advertises (clamped so KV doesn't eat the arena).
    let max_context = std::cmp::min(config.context_length.max(2048), 8192);
    let kv_element_size = 2usize;

    ArenaConfig::from_model(
        weights_bytes as usize,
        config.n_layers,
        config.n_kv_heads,
        config.head_dim,
        max_context,
        kv_element_size,
        config.hidden_dim,
        config.ffn_dim,
        1, // max_batch_size for decode
        total_vram,
    )
}
