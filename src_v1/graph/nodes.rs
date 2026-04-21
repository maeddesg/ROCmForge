//! Graph-node types for the Phase-1 computation graph.
//!
//! Each `GraphNode` corresponds to exactly one kernel dispatch in the
//! executor (§3.6 of `architecture_v1.2.0-draft.md`). Node definitions
//! are metadata-driven — a `GraphNode::Gemm` describes what it does,
//! not what kernel runs it. The executor picks WMMA vs. GEMV at
//! runtime based on `seq_len`.
//!
//! `WeightRef` stores the GGUF tensor name and its quant type but
//! **not** the arena offset. The slice comes from `LoadedModel::tensor_map`
//! at dispatch time. This keeps graph construction independent of VRAM
//! allocation, so the Block-A tests can build graphs on the CPU side
//! without a GPU device.

use super::super::core::tensor_info::GgmlType;

/// Buffer slot for an intermediate tensor. Resolved to a physical
/// address by the buffer planner (Block B).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BufferId(pub u32);

/// Reference to a weight tensor in VRAM. The arena offset is looked up
/// at dispatch time through `name` — no direct `ArenaSlice` here so
/// the graph can be built without allocating VRAM.
#[derive(Debug, Clone)]
pub struct WeightRef {
    pub name: String,
    pub format: GgmlType,
}

/// A single graph operation. Every variant becomes a kernel dispatch.
#[derive(Debug)]
pub enum GraphNode {
    /// Token-IDs → hidden state via FP32 embedding table.
    Embedding {
        table: WeightRef,
        input: BufferId,  // [seq_len] u32 token ids
        output: BufferId, // [seq_len × hidden_dim] f32
    },

    /// RMS layer normalisation. `dim` is the inner (per-row) length —
    /// `hidden_dim` for the main norms, `head_dim` for per-head QK-norm.
    /// `num_rows` batches multiple independent RMS computations in one
    /// dispatch (1 for the main norms, `n_heads` for Q-norm,
    /// `n_kv_heads` for K-norm).
    RmsNorm {
        weight: WeightRef,
        input: BufferId,
        output: BufferId,
        eps: f32,
        dim: usize,
        num_rows: usize,
    },

    /// Matrix-multiply. Executor picks GEMV (seq_len=1) or WMMA
    /// (seq_len>1) at runtime.
    Gemm {
        weight: WeightRef,
        bias: Option<WeightRef>,
        input: BufferId,
        output: BufferId,
        out_dim: usize, // N
        in_dim: usize,  // K
    },

    /// Rotary position embedding — in-place on Q and K.
    Rope {
        q_buffer: BufferId,
        k_buffer: BufferId,
        /// `None` = standard RoPE; `Some` = Llama-3.1 custom rope_freqs.
        rope_freqs: Option<WeightRef>,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        theta_base: f32,
    },

    /// Write the freshly-computed K/V into the layer's KV cache at the
    /// current position.
    KvCacheAppend {
        k_buffer: BufferId,
        v_buffer: BufferId,
        layer_idx: usize,
    },

    /// Attention. Executor dispatches decode (M=1) or prefill (M>1).
    Attention {
        q_buffer: BufferId,
        output: BufferId,
        layer_idx: usize,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
    },

    /// `a += b` (in-place).
    ResidualAdd {
        a: BufferId,
        b: BufferId,
    },

    /// Fused gate+up GEMM + SwiGLU. For decode the executor uses the
    /// fused Q4_K GEMV kernel; for prefill it issues two WMMAs and a
    /// separate SwiGLU (§5.3).
    GateUpSwiGLU {
        gate_weight: WeightRef,
        up_weight: WeightRef,
        input: BufferId,
        output: BufferId,
        ffn_dim: usize,
        hidden_dim: usize,
    },

    /// Elementwise `silu(gate) * up`. Used only when the prefill path
    /// splits GateUpSwiGLU into (WMMA gate, WMMA up, SwiGLU).
    SwiGLU {
        gate_buffer: BufferId,
        up_buffer: BufferId,
        output: BufferId,
    },
}

impl GraphNode {
    /// Short identifier for the node kind, useful in tests and logs.
    pub fn kind(&self) -> &'static str {
        match self {
            GraphNode::Embedding { .. } => "Embedding",
            GraphNode::RmsNorm { .. } => "RmsNorm",
            GraphNode::Gemm { .. } => "Gemm",
            GraphNode::Rope { .. } => "Rope",
            GraphNode::KvCacheAppend { .. } => "KvCacheAppend",
            GraphNode::Attention { .. } => "Attention",
            GraphNode::ResidualAdd { .. } => "ResidualAdd",
            GraphNode::GateUpSwiGLU { .. } => "GateUpSwiGLU",
            GraphNode::SwiGLU { .. } => "SwiGLU",
        }
    }

    /// Buffer IDs this node reads.
    pub fn inputs(&self) -> Vec<BufferId> {
        match self {
            GraphNode::Embedding { input, .. } => vec![*input],
            GraphNode::RmsNorm { input, .. } => vec![*input],
            GraphNode::Gemm { input, .. } => vec![*input],
            GraphNode::Rope {
                q_buffer,
                k_buffer,
                ..
            } => vec![*q_buffer, *k_buffer],
            GraphNode::KvCacheAppend {
                k_buffer,
                v_buffer,
                ..
            } => vec![*k_buffer, *v_buffer],
            GraphNode::Attention { q_buffer, .. } => vec![*q_buffer],
            GraphNode::ResidualAdd { a, b } => vec![*a, *b],
            GraphNode::GateUpSwiGLU { input, .. } => vec![*input],
            GraphNode::SwiGLU {
                gate_buffer,
                up_buffer,
                ..
            } => vec![*gate_buffer, *up_buffer],
        }
    }

    /// Buffer IDs this node writes (including in-place outputs).
    pub fn outputs(&self) -> Vec<BufferId> {
        match self {
            GraphNode::Embedding { output, .. } => vec![*output],
            GraphNode::RmsNorm { output, .. } => vec![*output],
            GraphNode::Gemm { output, .. } => vec![*output],
            // RoPE rewrites Q and K in place.
            GraphNode::Rope {
                q_buffer,
                k_buffer,
                ..
            } => vec![*q_buffer, *k_buffer],
            // KV-cache append writes into the (external) KV cache, not a BufferId.
            GraphNode::KvCacheAppend { .. } => vec![],
            GraphNode::Attention { output, .. } => vec![*output],
            // Residual add is in-place on `a`.
            GraphNode::ResidualAdd { a, .. } => vec![*a],
            GraphNode::GateUpSwiGLU { output, .. } => vec![*output],
            GraphNode::SwiGLU { output, .. } => vec![*output],
        }
    }
}
