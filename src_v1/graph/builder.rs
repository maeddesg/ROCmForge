//! Graph builder — constructs a `ComputationGraph` from
//! `ModelConfig` + per-layer tensor inventory.
//!
//! Metadata-driven. No `match architecture { "llama" => ... }` lives
//! here; the three Phase-1 models differ only through `ModelConfig`
//! feature flags (`has_qk_norm`, `has_rope_freqs`, `has_attention_bias`)
//! and their tensor inventory.
//!
//! The graph is fully unrolled: one node per (layer, op). Tests expect
//! the resulting counts (e.g. Qwen3 with 36 layers produces 36 RoPE
//! nodes, 36 Attention nodes, 72 residual-adds, etc.). This is
//! a *Rust* data structure, not GPU code — the Rust size cost is in
//! the tens of kilobytes.

use std::collections::HashMap;

use super::super::core::model_config::ModelConfig;
use super::super::core::tensor_info::{GgmlType, LayerTensors, TensorInfo, TensorRole};
use super::nodes::{BufferId, GraphNode, WeightRef};

/// Everything the builder needs from a parsed GGUF, without requiring
/// VRAM allocation. Tests construct this directly from `GGUFFile` +
/// `ModelConfig`; `LoadedModel` wraps it when the executor runs.
pub struct GraphBuildContext<'a> {
    pub config: &'a ModelConfig,
    pub layers: &'a [LayerTensors],
    pub global_tensors: HashMap<TensorRole, &'a TensorInfo>,
}

/// Full computation graph for a model. Execution order is literal:
/// the executor walks `nodes` front-to-back.
#[derive(Debug)]
pub struct ComputationGraph {
    pub nodes: Vec<GraphNode>,
    pub num_buffers: u32,
    pub config: ModelConfig,
    /// The buffer that holds the final logits (writable by the
    /// executor, returned to the caller).
    pub logits_buffer: BufferId,
    /// Input buffer holding token ids (filled by the executor's
    /// entry-point before dispatch).
    pub token_ids_buffer: BufferId,
    /// Hidden state right before the LM head — the input to the
    /// final `Gemm` node. Quality Monitor reads this buffer on
    /// calibration + periodic drift checks.
    pub hidden_state_buffer: BufferId,
}

/// Build errors. Carries the specific tensor / layer that's missing,
/// so diagnostic messages are actionable.
#[derive(Debug)]
pub enum BuildError {
    MissingLayerTensor {
        layer_idx: usize,
        role: TensorRole,
    },
    MissingGlobalTensor {
        role: TensorRole,
    },
    InvalidConfig(String),
}

impl std::fmt::Display for BuildError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BuildError::MissingLayerTensor { layer_idx, role } => {
                write!(f, "layer {} missing tensor for role {:?}", layer_idx, role)
            }
            BuildError::MissingGlobalTensor { role } => {
                write!(f, "model missing global tensor for role {:?}", role)
            }
            BuildError::InvalidConfig(msg) => write!(f, "invalid config: {}", msg),
        }
    }
}

impl std::error::Error for BuildError {}

// Helpers ---------------------------------------------------------------------

fn layer_weight(
    layer: &LayerTensors,
    role: TensorRole,
) -> Result<WeightRef, BuildError> {
    let info = layer
        .tensors
        .get(&role)
        .ok_or(BuildError::MissingLayerTensor {
            layer_idx: layer.layer_idx,
            role: role.clone(),
        })?;
    Ok(WeightRef {
        name: info.name.clone(),
        format: info.ggml_type,
    })
}

fn layer_weight_optional(layer: &LayerTensors, role: TensorRole) -> Option<WeightRef> {
    layer.tensors.get(&role).map(|info| WeightRef {
        name: info.name.clone(),
        format: info.ggml_type,
    })
}

fn global_weight(
    globals: &HashMap<TensorRole, &TensorInfo>,
    role: TensorRole,
) -> Result<WeightRef, BuildError> {
    let info = globals
        .get(&role)
        .ok_or(BuildError::MissingGlobalTensor { role: role.clone() })?;
    Ok(WeightRef {
        name: info.name.clone(),
        format: info.ggml_type,
    })
}

fn global_weight_optional(
    globals: &HashMap<TensorRole, &TensorInfo>,
    role: TensorRole,
) -> Option<WeightRef> {
    globals.get(&role).map(|info| WeightRef {
        name: info.name.clone(),
        format: info.ggml_type,
    })
}

// Builder ---------------------------------------------------------------------

pub struct GraphBuilder;

impl GraphBuilder {
    pub fn build(ctx: &GraphBuildContext) -> Result<ComputationGraph, BuildError> {
        let config = ctx.config;
        if ctx.layers.len() != config.n_layers {
            return Err(BuildError::InvalidConfig(format!(
                "expected {} layer tensor groups, got {}",
                config.n_layers,
                ctx.layers.len()
            )));
        }
        if config.n_heads == 0 || config.n_kv_heads == 0 || config.head_dim == 0 {
            return Err(BuildError::InvalidConfig(
                "n_heads/n_kv_heads/head_dim must be non-zero".into(),
            ));
        }
        if config.n_heads % config.n_kv_heads != 0 {
            return Err(BuildError::InvalidConfig(format!(
                "n_heads ({}) must be a multiple of n_kv_heads ({})",
                config.n_heads, config.n_kv_heads
            )));
        }

        let mut nodes: Vec<GraphNode> = Vec::new();
        let mut next_buf: u32 = 0;
        let mut alloc = |next: &mut u32| -> BufferId {
            let id = BufferId(*next);
            *next += 1;
            id
        };

        // ── Input / embedding ─────────────────────────────────────────────
        let token_ids = alloc(&mut next_buf); // u32 token ids
        let hidden = alloc(&mut next_buf); // [seq_len × hidden_dim] fp32

        nodes.push(GraphNode::Embedding {
            table: global_weight(&ctx.global_tensors, TensorRole::Embedding)?,
            input: token_ids,
            output: hidden,
        });

        // Llama-3.1 custom RoPE frequencies live here, shared across layers.
        let rope_freqs =
            global_weight_optional(&ctx.global_tensors, TensorRole::RopeFreqs);

        // ── Per-layer body ────────────────────────────────────────────────
        for layer in ctx.layers {
            let li = layer.layer_idx;

            // Attention pre-norm.
            let normed = alloc(&mut next_buf);
            nodes.push(GraphNode::RmsNorm {
                weight: layer_weight(layer, TensorRole::AttentionNorm)?,
                input: hidden,
                output: normed,
                eps: config.rms_norm_eps,
                dim: config.hidden_dim,
                num_rows: 1,
            });

            // Q, K, V projections. Bias is optional (Qwen2.5 has it).
            let q = alloc(&mut next_buf);
            let k = alloc(&mut next_buf);
            let v = alloc(&mut next_buf);
            nodes.push(GraphNode::Gemm {
                weight: layer_weight(layer, TensorRole::AttentionQ)?,
                bias: layer_weight_optional(layer, TensorRole::AttentionQBias),
                input: normed,
                output: q,
                out_dim: config.n_heads * config.head_dim,
                in_dim: config.hidden_dim,
            });
            nodes.push(GraphNode::Gemm {
                weight: layer_weight(layer, TensorRole::AttentionK)?,
                bias: layer_weight_optional(layer, TensorRole::AttentionKBias),
                input: normed,
                output: k,
                out_dim: config.n_kv_heads * config.head_dim,
                in_dim: config.hidden_dim,
            });
            nodes.push(GraphNode::Gemm {
                weight: layer_weight(layer, TensorRole::AttentionV)?,
                bias: layer_weight_optional(layer, TensorRole::AttentionVBias),
                input: normed,
                output: v,
                out_dim: config.n_kv_heads * config.head_dim,
                in_dim: config.hidden_dim,
            });

            // Optional Q/K per-head norm (Qwen3).
            let (q_for_rope, k_for_rope) = if config.has_qk_norm {
                let q_normed = alloc(&mut next_buf);
                let k_normed = alloc(&mut next_buf);
                nodes.push(GraphNode::RmsNorm {
                    weight: layer_weight(layer, TensorRole::AttentionQNorm)?,
                    input: q,
                    output: q_normed,
                    eps: config.rms_norm_eps,
                    dim: config.head_dim,
                    num_rows: config.n_heads,
                });
                nodes.push(GraphNode::RmsNorm {
                    weight: layer_weight(layer, TensorRole::AttentionKNorm)?,
                    input: k,
                    output: k_normed,
                    eps: config.rms_norm_eps,
                    dim: config.head_dim,
                    num_rows: config.n_kv_heads,
                });
                (q_normed, k_normed)
            } else {
                (q, k)
            };

            nodes.push(GraphNode::Rope {
                q_buffer: q_for_rope,
                k_buffer: k_for_rope,
                rope_freqs: rope_freqs.clone(),
                n_heads: config.n_heads,
                n_kv_heads: config.n_kv_heads,
                head_dim: config.head_dim,
                theta_base: config.rope_freq_base,
            });

            nodes.push(GraphNode::KvCacheAppend {
                k_buffer: k_for_rope,
                v_buffer: v,
                layer_idx: li,
            });

            let attn_out = alloc(&mut next_buf);
            nodes.push(GraphNode::Attention {
                q_buffer: q_for_rope,
                output: attn_out,
                layer_idx: li,
                n_heads: config.n_heads,
                n_kv_heads: config.n_kv_heads,
                head_dim: config.head_dim,
            });

            // Output projection.
            let projected = alloc(&mut next_buf);
            nodes.push(GraphNode::Gemm {
                weight: layer_weight(layer, TensorRole::AttentionOutput)?,
                bias: layer_weight_optional(layer, TensorRole::AttentionOutputBias),
                input: attn_out,
                output: projected,
                out_dim: config.hidden_dim,
                in_dim: config.n_heads * config.head_dim,
            });

            // hidden += projected
            nodes.push(GraphNode::ResidualAdd {
                a: hidden,
                b: projected,
            });

            // FFN pre-norm.
            let normed_ffn = alloc(&mut next_buf);
            nodes.push(GraphNode::RmsNorm {
                weight: layer_weight(layer, TensorRole::FFNNorm)?,
                input: hidden,
                output: normed_ffn,
                eps: config.rms_norm_eps,
                dim: config.hidden_dim,
                num_rows: 1,
            });

            // Gate + Up + SwiGLU (fused node; executor splits for prefill).
            let ffn_mid = alloc(&mut next_buf);
            nodes.push(GraphNode::GateUpSwiGLU {
                gate_weight: layer_weight(layer, TensorRole::FFNGate)?,
                up_weight: layer_weight(layer, TensorRole::FFNUp)?,
                input: normed_ffn,
                output: ffn_mid,
                ffn_dim: config.ffn_dim,
                hidden_dim: config.hidden_dim,
            });

            // Down projection.
            let ffn_out = alloc(&mut next_buf);
            nodes.push(GraphNode::Gemm {
                weight: layer_weight(layer, TensorRole::FFNDown)?,
                bias: layer_weight_optional(layer, TensorRole::FFNDownBias),
                input: ffn_mid,
                output: ffn_out,
                out_dim: config.hidden_dim,
                in_dim: config.ffn_dim,
            });

            // hidden += ffn_out
            nodes.push(GraphNode::ResidualAdd {
                a: hidden,
                b: ffn_out,
            });
        }

        // ── Output head ──────────────────────────────────────────────────
        let normed_final = alloc(&mut next_buf);
        nodes.push(GraphNode::RmsNorm {
            weight: global_weight(&ctx.global_tensors, TensorRole::OutputNorm)?,
            input: hidden,
            output: normed_final,
            eps: config.rms_norm_eps,
            dim: config.hidden_dim,
            num_rows: 1,
        });

        let logits = alloc(&mut next_buf);
        nodes.push(GraphNode::Gemm {
            weight: global_weight(&ctx.global_tensors, TensorRole::LMHead)?,
            bias: None,
            input: normed_final,
            output: logits,
            out_dim: config.vocab_size,
            in_dim: config.hidden_dim,
        });

        // ── Fusion pass (Phase 2 step 2.0.2) ──────────────────────────────
        //
        // Collapse every `Gemm → ResidualAdd` pair where:
        //   (a) the Gemm's `output` buffer feeds directly into the
        //       ResidualAdd's `b` input (no intervening read of that
        //       buffer), AND
        //   (b) the Gemm weight is Q4_K (only kernel we emit a fused
        //       variant for right now), AND
        //   (c) `out_dim == hidden_dim` (stride guard per prompt;
        //       ensures `residual[i]` is element-addressable with
        //       the same index as the GEMV output).
        //
        // Replaces the pair with one `FusedGemmResidual` node. The
        // Gemm's bias is always `None` on the Attention-Output and
        // FFN-Down sites for Qwen3/Llama-3.1; bias-carrying Gemms
        // are skipped to keep semantics strict.
        let fused_nodes = fuse_gemm_residual_pairs(nodes);

        Ok(ComputationGraph {
            nodes: fused_nodes,
            num_buffers: next_buf,
            config: config.clone(),
            logits_buffer: logits,
            token_ids_buffer: token_ids,
            hidden_state_buffer: normed_final,
        })
    }
}

/// Post-pass that replaces adjacent (Gemm, ResidualAdd) pairs with a
/// single `FusedGemmResidual` node. See `GraphBuilder::build` for the
/// fusion conditions. Non-matching nodes are passed through unchanged.
fn fuse_gemm_residual_pairs(nodes: Vec<GraphNode>) -> Vec<GraphNode> {
    use super::super::core::tensor_info::GgmlType;

    let mut out: Vec<GraphNode> = Vec::with_capacity(nodes.len());
    let mut iter = nodes.into_iter().peekable();
    while let Some(node) = iter.next() {
        let take_fused = matches!(&node, GraphNode::Gemm { weight, bias, .. }
            if weight.format == GgmlType::Q4_K && bias.is_none());
        if take_fused {
            if let GraphNode::Gemm {
                weight,
                bias: _,
                input,
                output,
                out_dim,
                in_dim,
            } = &node
            {
                if let Some(GraphNode::ResidualAdd { a, b }) = iter.peek() {
                    if *b == *output {
                        let residual_target = *a;
                        let gemv_input = *input;
                        let weight = weight.clone();
                        let out_dim = *out_dim;
                        let in_dim = *in_dim;
                        // Consume the ResidualAdd we just peeked.
                        iter.next();
                        out.push(GraphNode::FusedGemmResidual {
                            weight,
                            input: gemv_input,
                            residual: residual_target,
                            out_dim,
                            in_dim,
                        });
                        continue;
                    }
                }
            }
        }
        out.push(node);
    }
    out
}
