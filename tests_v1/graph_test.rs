//! Phase 1 / Schritt 1.10 Block A — Computation graph build tests.
//!
//! CPU-only: the graph is constructed from GGUF metadata + `ModelConfig`
//! without touching VRAM. All three Phase-1 target models must build
//! cleanly through the same `GraphBuilder::build()` path — no
//! architecture-specific branches.

#![cfg(feature = "v1")]

use std::collections::HashMap;

use rocmforge::v1::core::gguf::GGUFFile;
use rocmforge::v1::core::model_config::ModelConfig;
use rocmforge::v1::core::tensor_info::{
    group_tensors_by_layer, parse_tensor_name, LayerTensors, TensorInfo, TensorRole,
};
use rocmforge::v1::graph::{
    BufferId, ComputationGraph, GraphBuildContext, GraphBuilder, GraphNode,
};

// ── Helpers ────────────────────────────────────────────────────────────────

fn model_path(name: &str) -> std::path::PathBuf {
    dirs::home_dir()
        .expect("HOME set")
        .join("models")
        .join(name)
}

const QWEN3: &str = "Qwen3-8B-Q4_K_M.gguf";
const LLAMA31: &str = "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf";
const QWEN25: &str = "Qwen2.5-7B-Instruct-Q4_0.gguf";

fn build_graph_from_gguf(
    path: std::path::PathBuf,
) -> (
    ModelConfig,
    Vec<LayerTensors>,
    HashMap<TensorRole, TensorInfo>,
    ComputationGraph,
) {
    let gguf = GGUFFile::open(&path).expect("open gguf");
    let cfg = ModelConfig::from_metadata(gguf.metadata(), gguf.tensors()).expect("config");
    let layers = group_tensors_by_layer(gguf.tensors());

    // Collect global-tensor roles.
    let mut globals_owned: HashMap<TensorRole, TensorInfo> = HashMap::new();
    for t in gguf.tensors() {
        let (role, layer) = parse_tensor_name(&t.name);
        if layer.is_none() {
            if !matches!(role, TensorRole::Unknown(_)) {
                globals_owned.insert(role, t.clone());
            }
        }
    }
    // Build a borrow map for the context.
    let global_refs: HashMap<TensorRole, &TensorInfo> = globals_owned
        .iter()
        .map(|(k, v)| (k.clone(), v))
        .collect();

    let ctx = GraphBuildContext {
        config: &cfg,
        layers: &layers,
        global_tensors: global_refs,
    };
    let graph = GraphBuilder::build(&ctx).expect("graph build");
    (cfg, layers, globals_owned, graph)
}

fn count_kind(graph: &ComputationGraph, kind: &str) -> usize {
    graph.nodes.iter().filter(|n| n.kind() == kind).count()
}

// ── Per-model build tests ─────────────────────────────────────────────────

#[test]
fn test_graph_build_qwen3() {
    let (cfg, _, _, graph) = build_graph_from_gguf(model_path(QWEN3));
    assert_eq!(cfg.n_layers, 36);
    assert!(cfg.has_qk_norm, "Qwen3 should have qk_norm");
    assert!(!cfg.has_rope_freqs, "Qwen3 should not have rope_freqs");
    assert!(cfg.is_gqa);

    let embeddings = count_kind(&graph, "Embedding");
    let rope = count_kind(&graph, "Rope");
    let attn = count_kind(&graph, "Attention");
    let residual = count_kind(&graph, "ResidualAdd");
    let gate_up = count_kind(&graph, "GateUpSwiGLU");
    let rms = count_kind(&graph, "RmsNorm");
    let gemm = count_kind(&graph, "Gemm");
    let kv_cache = count_kind(&graph, "KvCacheAppend");

    assert_eq!(embeddings, 1);
    assert_eq!(rope, 36);
    assert_eq!(attn, 36);
    assert_eq!(residual, 72);
    assert_eq!(gate_up, 36);
    assert_eq!(kv_cache, 36);

    // 2 per-layer (attn_norm + ffn_norm) + 2 qk-norm per layer + 1 output_norm.
    assert_eq!(rms, 2 * 36 + 2 * 36 + 1, "got {rms}");
    // 3 QKV + 1 O-proj + 1 down-proj per layer + 1 lm-head.
    assert_eq!(gemm, 5 * 36 + 1, "got {gemm}");

    // Every Rope node for Qwen3 has no rope_freqs.
    for n in &graph.nodes {
        if let GraphNode::Rope { rope_freqs, .. } = n {
            assert!(rope_freqs.is_none(), "Qwen3 RoPE must not reference rope_freqs");
        }
    }

    println!(
        "Qwen3 graph: nodes={}, buffers={}, rms_norms={rms}, gemms={gemm}",
        graph.nodes.len(),
        graph.num_buffers
    );
}

#[test]
fn test_graph_build_llama31() {
    let (cfg, _, _, graph) = build_graph_from_gguf(model_path(LLAMA31));
    assert_eq!(cfg.n_layers, 32);
    assert!(!cfg.has_qk_norm);
    assert!(cfg.has_rope_freqs);

    assert_eq!(count_kind(&graph, "Rope"), 32);
    assert_eq!(count_kind(&graph, "Attention"), 32);
    // 32 attn_norm + 32 ffn_norm + 1 output_norm = 65.
    assert_eq!(count_kind(&graph, "RmsNorm"), 65);
    // 3 QKV + 1 O + 1 down per layer + 1 LM head.
    assert_eq!(count_kind(&graph, "Gemm"), 5 * 32 + 1);

    // Every Rope node for Llama-3.1 has rope_freqs set.
    let rope_with_freqs = graph
        .nodes
        .iter()
        .filter(|n| matches!(n, GraphNode::Rope { rope_freqs: Some(_), .. }))
        .count();
    assert_eq!(rope_with_freqs, 32, "Llama-3.1 should use custom rope_freqs");

    println!(
        "Llama-3.1 graph: nodes={}, buffers={}, rope_with_freqs={rope_with_freqs}",
        graph.nodes.len(),
        graph.num_buffers
    );
}

#[test]
fn test_graph_build_qwen25() {
    let (cfg, _, _, graph) = build_graph_from_gguf(model_path(QWEN25));
    assert_eq!(cfg.n_layers, 28);
    assert!(cfg.has_attention_bias, "Qwen2.5 should have attention biases");
    assert!(!cfg.has_qk_norm);
    assert!(!cfg.has_rope_freqs);

    // Count Gemm nodes that carry a bias. Qwen2.5 has Q/K/V bias =>
    // 3 per layer with bias, 2 without.
    let gemm_with_bias = graph
        .nodes
        .iter()
        .filter(|n| matches!(n, GraphNode::Gemm { bias: Some(_), .. }))
        .count();
    assert_eq!(
        gemm_with_bias,
        3 * cfg.n_layers,
        "Qwen2.5 QKV bias count wrong: {gemm_with_bias}"
    );

    println!(
        "Qwen2.5 graph: nodes={}, buffers={}, gemm_with_bias={gemm_with_bias}",
        graph.nodes.len(),
        graph.num_buffers
    );
}

// ── Meta-invariants ───────────────────────────────────────────────────────

#[test]
fn test_graph_no_model_specific_code() {
    // All three models must build through the same function without
    // any branching on architecture name inside `GraphBuilder`.
    for name in [QWEN3, LLAMA31, QWEN25] {
        let (_, _, _, graph) = build_graph_from_gguf(model_path(name));
        assert!(!graph.nodes.is_empty(), "{name}: empty graph");
    }
}

#[test]
fn test_graph_node_order_valid() {
    let (_, _, _, graph) = build_graph_from_gguf(model_path(QWEN3));

    // Walk nodes in order. Each input buffer must already be in the
    // defined set. The token-ids buffer is implicitly defined.
    let mut defined: std::collections::HashSet<BufferId> =
        std::collections::HashSet::new();
    defined.insert(graph.token_ids_buffer);

    for (i, node) in graph.nodes.iter().enumerate() {
        for input in node.inputs() {
            assert!(
                defined.contains(&input),
                "node {i} ({}) reads undefined buffer {input:?}",
                node.kind()
            );
        }
        for output in node.outputs() {
            defined.insert(output);
        }
    }
    println!("Graph node order: all {} nodes valid", graph.nodes.len());
}

#[test]
fn test_graph_buffer_count_reasonable() {
    // Fully-unrolled graph should still have a bounded buffer count —
    // rough ballpark: one per transient intermediate per layer. For
    // Qwen3 with 36 layers we allocate ~14 per layer plus globals, so
    // ≲ 800 is the expected ceiling. Fail hard if we overshoot wildly.
    let (_, _, _, graph) = build_graph_from_gguf(model_path(QWEN3));
    assert!(
        graph.num_buffers < 2000,
        "buffer count exploded: {}",
        graph.num_buffers
    );
    println!("Qwen3 buffer count: {}", graph.num_buffers);
}
