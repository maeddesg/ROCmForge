//! Buffer planner (Phase 1 — simple).
//!
//! Phase-1 philosophy: every transient `BufferId` gets its own
//! dedicated device allocation sized for the worst-case per-token
//! shape. No Ping-Pong sharing yet — that's a Phase-2 optimisation
//! once liveness analysis exists.
//!
//! The sizing heuristic covers every Phase-1 op:
//!   Embedding output, RmsNorm output, residual targets: `hidden_dim`
//!   Q projection output:   `n_heads × head_dim`
//!   K/V projection output: `n_kv_heads × head_dim`
//!   Attention output:      `n_heads × head_dim`
//!   GateUpSwiGLU output:   `ffn_dim`
//!   Down projection output (back to hidden): `hidden_dim`
//!   LM-head logits output: `vocab_size`
//!
//! Token-ids buffer carries `seq_len` u32 tokens.

use super::super::core::model_config::ModelConfig;
use super::nodes::{BufferId, GraphNode, WeightRef};
use super::ComputationGraph;

use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct BufferSpec {
    /// Number of f32 elements (for f32 buffers) or u32 elements
    /// (for the token-ids buffer).
    pub elem_count: usize,
    /// Byte size of a single element. 4 for f32 / u32.
    pub elem_size: usize,
}

impl BufferSpec {
    pub fn bytes(&self, seq_len: usize) -> usize {
        self.elem_count * self.elem_size * seq_len
    }
}

#[derive(Debug)]
pub struct BufferPlan {
    /// Per-buffer size specification (per-token; multiply by seq_len
    /// when allocating).
    pub specs: HashMap<BufferId, BufferSpec>,
}

impl BufferPlan {
    /// Produce a Phase-1 plan from a built graph. Walks every node and
    /// infers the per-token output size from the node kind + the
    /// model's shape parameters.
    pub fn plan_phase1(graph: &ComputationGraph) -> Self {
        let cfg = &graph.config;
        let mut specs: HashMap<BufferId, BufferSpec> = HashMap::new();

        // Token-ids buffer is u32, one element per token.
        specs.insert(
            graph.token_ids_buffer,
            BufferSpec {
                elem_count: 1,
                elem_size: 4,
            },
        );

        let set = |specs: &mut HashMap<BufferId, BufferSpec>, id: BufferId, elems: usize| {
            specs
                .entry(id)
                .or_insert(BufferSpec {
                    elem_count: elems,
                    elem_size: 4,
                });
        };

        for node in &graph.nodes {
            match node {
                GraphNode::Embedding { output, .. } => {
                    set(&mut specs, *output, cfg.hidden_dim);
                }
                GraphNode::RmsNorm { output, .. } => {
                    set(&mut specs, *output, cfg.hidden_dim);
                }
                GraphNode::Gemm {
                    output, out_dim, ..
                } => {
                    set(&mut specs, *output, *out_dim);
                }
                GraphNode::Rope { .. }
                | GraphNode::KvCacheAppend { .. }
                | GraphNode::ResidualAdd { .. }
                | GraphNode::FusedGemmResidual { .. } => {
                    // In-place, cache-only, or read-modify-write on
                    // an already-sized residual — nothing new to size.
                }
                GraphNode::Attention { output, .. } => {
                    set(&mut specs, *output, cfg.n_heads * cfg.head_dim);
                }
                GraphNode::GateUpSwiGLU { output, .. } => {
                    set(&mut specs, *output, cfg.ffn_dim);
                }
                GraphNode::SwiGLU { output, .. } => {
                    set(&mut specs, *output, cfg.ffn_dim);
                }
            }
        }

        // Sanity: the logits buffer must be sized for vocab_size (the
        // LM-head Gemm already covers this, but assert explicitly).
        specs.insert(
            graph.logits_buffer,
            BufferSpec {
                elem_count: cfg.vocab_size,
                elem_size: 4,
            },
        );

        Self { specs }
    }

    /// Size of the largest transient buffer in bytes for a given seq_len.
    pub fn max_bytes(&self, seq_len: usize) -> usize {
        self.specs
            .values()
            .map(|s| s.bytes(seq_len))
            .max()
            .unwrap_or(0)
    }

    /// Total bytes if every buffer is allocated separately.
    pub fn total_bytes(&self, seq_len: usize) -> usize {
        self.specs.values().map(|s| s.bytes(seq_len)).sum()
    }
}

/// Per-layer KV cache layout. Each layer owns contiguous K and V
/// regions, each `num_kv_heads × head_stride` floats. `head_stride`
/// is rounded up so a single head's slice starts on a 256-byte
/// boundary.
#[derive(Debug, Clone, Copy)]
pub struct KvCacheLayout {
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub max_seq: usize,
    /// Stride per head in floats (aligned). At least `max_seq * head_dim`.
    pub head_stride: usize,
}

impl KvCacheLayout {
    pub fn from_config(cfg: &ModelConfig, max_seq: usize) -> Self {
        let raw = max_seq * cfg.head_dim;
        // 256-byte alignment => 64 floats.
        let head_stride = (raw + 63) / 64 * 64;
        Self {
            num_kv_heads: cfg.n_kv_heads,
            head_dim: cfg.head_dim,
            max_seq,
            head_stride,
        }
    }

    pub fn bytes_per_layer(&self) -> usize {
        // K and V separately.
        2 * self.num_kv_heads * self.head_stride * 4
    }
}

// Keep a reference to WeightRef alive for the compiler (the planner
// doesn't touch weight refs but layouts use `ModelConfig`; keeping the
// import documents the relationship).
#[allow(dead_code)]
fn _weight_ref_anchor(_: &WeightRef) {}
