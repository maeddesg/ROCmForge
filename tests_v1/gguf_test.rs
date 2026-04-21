//! Phase 1 / Schritt 1.5 — GGUF loader integration tests.
//!
//! 16 tests that hit the three Phase-1 target models on disk:
//!   - Qwen2.5-7B-Instruct-Q4_0
//!   - Qwen3-8B-Q4_K_M
//!   - Meta-Llama-3.1-8B-Instruct-Q4_K_M
//!
//! CPU tests cover parser / metadata / tensor-role detection. GPU
//! tests (gated on `feature = "gpu"`) load the full Qwen3-8B into the
//! VRAM arena and verify 256-byte alignment plus a VRAM↔disk readback
//! spot-check. All GPU tests are `#[serial]` — the arena reserves
//! multi-GB slices on a single consumer GPU.

#![cfg(feature = "v1")]

use rocmforge::v1::core::gguf::GGUFFile;
use rocmforge::v1::core::model_config::ModelConfig;
use rocmforge::v1::core::tensor_info::{
    group_tensors_by_layer, validate_quant_types, GgmlType, TensorInfo, TensorRole,
};

fn model_path(name: &str) -> std::path::PathBuf {
    let home = dirs::home_dir().expect("HOME must be set");
    home.join("models").join(name)
}

const QWEN3: &str = "Qwen3-8B-Q4_K_M.gguf";
const LLAMA31: &str = "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf";
const QWEN25: &str = "Qwen2.5-7B-Instruct-Q4_0.gguf";

// ── Parser tests ─────────────────────────────────────────────────────────────

#[test]
fn test_gguf_parse_qwen3_8b() {
    let gguf = GGUFFile::open(model_path(QWEN3)).expect("open Qwen3");
    assert!(gguf.tensor_count() > 0);
    assert!(gguf.metadata_count() > 0);
    println!(
        "Qwen3-8B: {} tensors, {} metadata entries",
        gguf.tensor_count(),
        gguf.metadata_count()
    );
}

#[test]
fn test_gguf_parse_llama31_8b() {
    let gguf = GGUFFile::open(model_path(LLAMA31)).expect("open Llama-3.1");
    assert!(gguf.tensor_count() > 0);
    assert!(gguf.metadata_count() > 0);
}

#[test]
fn test_gguf_parse_qwen25_7b() {
    let gguf = GGUFFile::open(model_path(QWEN25)).expect("open Qwen2.5");
    assert!(gguf.tensor_count() > 0);
}

// ── Metadata tests ───────────────────────────────────────────────────────────

#[test]
fn test_gguf_metadata_architecture_qwen3() {
    let gguf = GGUFFile::open(model_path(QWEN3)).unwrap();
    let arch = gguf
        .metadata()
        .get("general.architecture")
        .expect("architecture key missing");
    let arch_str = arch.as_string().unwrap();
    assert!(
        arch_str == "qwen2" || arch_str == "qwen3",
        "expected qwen2 or qwen3, got {arch_str}"
    );
}

#[test]
fn test_gguf_metadata_llama_architecture() {
    let gguf = GGUFFile::open(model_path(LLAMA31)).unwrap();
    let arch = gguf.metadata().get("general.architecture").unwrap();
    assert_eq!(arch.as_string().unwrap(), "llama");
}

// ── ModelConfig tests ────────────────────────────────────────────────────────

#[test]
fn test_model_config_qwen3() {
    let gguf = GGUFFile::open(model_path(QWEN3)).unwrap();
    let config = ModelConfig::from_metadata(gguf.metadata(), gguf.tensors()).unwrap();

    assert_eq!(config.n_layers, 36);
    assert_eq!(config.hidden_dim, 4096);
    assert!(config.n_heads > 0);
    assert!(config.n_kv_heads > 0);
    assert!(config.is_gqa, "Qwen3 should be GQA");
    assert!(config.has_qk_norm, "Qwen3 should have Q/K-Norm");
    assert!(!config.has_rope_freqs, "Qwen3 should NOT have rope_freqs");

    println!(
        "Qwen3: arch={} layers={} heads={} kv_heads={} hidden={} ffn={} vocab={} head_dim={}",
        config.architecture,
        config.n_layers,
        config.n_heads,
        config.n_kv_heads,
        config.hidden_dim,
        config.ffn_dim,
        config.vocab_size,
        config.head_dim
    );
}

#[test]
fn test_model_config_llama31() {
    let gguf = GGUFFile::open(model_path(LLAMA31)).unwrap();
    let config = ModelConfig::from_metadata(gguf.metadata(), gguf.tensors()).unwrap();

    assert_eq!(config.architecture, "llama");
    assert_eq!(config.n_layers, 32);
    assert!(!config.has_qk_norm, "Llama-3.1 should NOT have Q/K-Norm");
    assert!(config.has_rope_freqs, "Llama-3.1 SHOULD have rope_freqs");
}

#[test]
fn test_model_config_qwen25() {
    let gguf = GGUFFile::open(model_path(QWEN25)).unwrap();
    let config = ModelConfig::from_metadata(gguf.metadata(), gguf.tensors()).unwrap();

    assert_eq!(config.architecture, "qwen2");
    assert!(!config.has_qk_norm, "Qwen2.5 should NOT have Q/K-Norm");
    assert!(!config.has_rope_freqs);
    // Qwen2.5 dense transformers ship attention biases (attn_q.bias, …).
    assert!(
        config.has_attention_bias,
        "Qwen2.5 should have attention biases"
    );
}

#[test]
fn test_gqa_detection() {
    let gguf = GGUFFile::open(model_path(QWEN3)).unwrap();
    let config = ModelConfig::from_metadata(gguf.metadata(), gguf.tensors()).unwrap();
    assert_ne!(config.n_heads, config.n_kv_heads);
    assert!(config.is_gqa);
    // head_dim may be explicit (`attention.key_length`) or
    // hidden/heads — both contracts are acceptable as long as it's > 0
    // and divides hidden evenly when derived.
    assert!(config.head_dim > 0);
}

// ── Tensor-inventory tests ───────────────────────────────────────────────────

#[test]
fn test_tensor_quant_types() {
    let gguf = GGUFFile::open(model_path(QWEN3)).unwrap();
    let config = ModelConfig::from_metadata(gguf.metadata(), gguf.tensors()).unwrap();

    assert!(
        config.quant_formats_used.contains(&GgmlType::Q4_K),
        "Qwen3 Q4_K_M should contain Q4_K tensors; got {:?}",
        config.quant_formats_used
    );
    assert!(
        config.quant_formats_used.contains(&GgmlType::Q6_K),
        "Qwen3 Q4_K_M should contain Q6_K tensors; got {:?}",
        config.quant_formats_used
    );
}

#[test]
fn test_tensor_layer_grouping() {
    let gguf = GGUFFile::open(model_path(QWEN3)).unwrap();
    let layers = group_tensors_by_layer(gguf.tensors());

    assert_eq!(layers.len(), 36, "Qwen3 should have 36 layers");
    for (i, layer) in layers.iter().enumerate() {
        assert_eq!(layer.layer_idx, i, "layer index must be sorted");
        assert!(
            layer.tensors.contains_key(&TensorRole::AttentionQ),
            "layer {i} missing attn_q"
        );
        assert!(
            layer.tensors.contains_key(&TensorRole::FFNGate),
            "layer {i} missing ffn_gate"
        );
        assert!(
            layer.tensors.contains_key(&TensorRole::FFNDown),
            "layer {i} missing ffn_down"
        );
    }
}

#[test]
fn test_qk_norm_detection() {
    let gguf = GGUFFile::open(model_path(QWEN3)).unwrap();
    let layers = group_tensors_by_layer(gguf.tensors());
    for layer in &layers {
        assert!(
            layer.tensors.contains_key(&TensorRole::AttentionQNorm),
            "Qwen3 layer {} should have attn_q_norm",
            layer.layer_idx
        );
        assert!(
            layer.tensors.contains_key(&TensorRole::AttentionKNorm),
            "Qwen3 layer {} should have attn_k_norm",
            layer.layer_idx
        );
    }

    let gguf_llama = GGUFFile::open(model_path(LLAMA31)).unwrap();
    let layers_llama = group_tensors_by_layer(gguf_llama.tensors());
    for layer in &layers_llama {
        assert!(
            !layer.tensors.contains_key(&TensorRole::AttentionQNorm),
            "Llama layer {} should NOT have attn_q_norm",
            layer.layer_idx
        );
    }
}

#[test]
fn test_rope_params() {
    let gguf = GGUFFile::open(model_path(QWEN3)).unwrap();
    let config = ModelConfig::from_metadata(gguf.metadata(), gguf.tensors()).unwrap();
    assert!(config.rope_freq_base > 0.0);
    println!("Qwen3 rope_freq_base: {}", config.rope_freq_base);
}

#[test]
fn test_unknown_quant_warning() {
    let fake = vec![TensorInfo {
        name: "test.weight".to_string(),
        shape: vec![4096, 4096],
        ggml_type: GgmlType::Q3_K,
        file_offset: 0,
        byte_size: 1000,
        n_elements: 4096 * 4096,
    }];
    let warnings = validate_quant_types(&fake);
    assert!(!warnings.is_empty(), "Q3_K should produce a warning");
    assert!(warnings[0].contains("Q3_K"));
}

// ── VRAM loading tests (GPU only) ────────────────────────────────────────────

#[cfg(feature = "gpu")]
mod gpu {
    use super::*;

    use rocmforge::v1::backend::gpu::device::GpuDevice;
    use rocmforge::v1::backend::gpu::hip_ffi::{hipMemcpy, hipMemcpyDeviceToHost};
    use rocmforge::v1::core::model_loader::LoadedModel;
    use serial_test::serial;

    fn load_qwen3() -> LoadedModel {
        let device = GpuDevice::detect(0).expect("GPU detect");
        LoadedModel::load(model_path(QWEN3), &device).expect("Qwen3 load")
    }

    #[test]
    #[serial]
    fn test_weights_load_to_vram() {
        let model = load_qwen3();
        assert!(!model.tensor_map.is_empty());
        assert!(model.arena.weights_used() > 0);
        println!(
            "Loaded {} tensors, {:.2} GB in VRAM",
            model.tensor_map.len(),
            model.arena.weights_used() as f64 / 1e9
        );
    }

    #[test]
    #[serial]
    fn test_weights_alignment() {
        let model = load_qwen3();
        for (name, slice) in &model.tensor_map {
            assert_eq!(
                slice.offset % 256,
                0,
                "tensor '{name}' not 256-byte aligned: offset={}",
                slice.offset
            );
        }
    }

    #[test]
    #[serial]
    fn test_weights_readback_spot_check() {
        let model = load_qwen3();
        let gguf = GGUFFile::open(model_path(QWEN3)).unwrap();

        for tensor_name in &[
            "token_embd.weight",
            "blk.17.attn_q.weight",
            "output.weight",
        ] {
            let slice = model
                .tensor_map
                .get(*tensor_name)
                .unwrap_or_else(|| panic!("tensor '{tensor_name}' missing from tensor_map"));
            let info = gguf
                .tensors()
                .iter()
                .find(|t| t.name == *tensor_name)
                .unwrap();

            // First 256 bytes are enough — the arena offset is aligned,
            // so any misconfiguration would show up immediately.
            let check_size = std::cmp::min(256usize, slice.size);
            let mut gpu_data = vec![0u8; check_size];
            let base = model.arena.base_ptr();
            let rc = unsafe {
                hipMemcpy(
                    gpu_data.as_mut_ptr() as *mut _,
                    slice.as_ptr(base),
                    check_size,
                    hipMemcpyDeviceToHost,
                )
            };
            assert_eq!(rc, 0, "hipMemcpy D2H returned {rc} for '{tensor_name}'");

            let disk_data = gguf.tensor_data(info, check_size).unwrap();
            assert_eq!(
                gpu_data, disk_data,
                "VRAM != Disk for tensor '{tensor_name}' (first {check_size} bytes)"
            );
        }
    }

    #[test]
    #[serial]
    fn test_weights_total_size() {
        let model = load_qwen3();
        let used_gb = model.arena.weights_used() as f64 / 1e9;
        assert!(
            (3.0..7.0).contains(&used_gb),
            "expected ~4.7 GB weights for Qwen3-8B Q4_K_M, got {used_gb:.2} GB"
        );
    }
}
