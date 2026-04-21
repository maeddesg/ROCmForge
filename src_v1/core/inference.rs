//! End-to-end inference pipeline: tokenize → prefill → decode loop.
//!
//! Phase 1 wraps the `GraphExecutor` from Schritt 1.10 with a tokenizer
//! and sampler. Prefill is implemented as a sequential decode over the
//! prompt tokens (the executor's per-token API); WMMA-batched prefill
//! is a Phase-2 optimisation.

use std::time::Instant;

use super::gguf::GGUFFile;
use super::model_config::ModelConfig;
use super::sampling::{sample_token, SamplingConfig};
use super::tokenizer::Tokenizer;
use super::super::backend::gpu::error::HipResult;
use super::super::core::model_loader::LoadedModel;
use super::super::graph::{BufferPlan, ComputationGraph, GraphExecutor};

/// One generation call's metrics and output. Used by both the CLI and
/// the 15-prompt report.
#[derive(Debug, Clone)]
pub struct GenerationResult {
    pub prompt: String,
    pub output: String,
    pub prompt_tokens: usize,
    pub generated_tokens: usize,
    pub prefill_ms: f64,
    pub decode_ms: f64,
    pub total_ms: f64,
    pub prefill_tok_s: f64,
    pub decode_tok_s: f64,
    /// `true` if generation stopped because an EOS token was emitted;
    /// `false` if we hit `max_tokens`.
    pub hit_eos: bool,
}

pub struct InferencePipeline<'m> {
    pub executor: GraphExecutor<'m>,
    pub tokenizer: Tokenizer,
    pub config: ModelConfig,
}

impl<'m> InferencePipeline<'m> {
    /// Build a pipeline for the given loaded model + graph + plan +
    /// GGUF handle. `max_seq` caps KV-cache size.
    pub fn new(
        graph: ComputationGraph,
        plan: BufferPlan,
        model: &'m LoadedModel,
        gguf: &GGUFFile,
        max_seq: usize,
    ) -> HipResult<Self> {
        let config = model.config.clone();
        let tokenizer = Tokenizer::from_gguf_metadata(gguf.metadata(), &config.architecture)
            .map_err(|e| super::super::backend::gpu::error::HipError {
                code: -1,
                message: format!("tokenizer init: {e}"),
                context: "pipeline".into(),
            })?;
        let executor = GraphExecutor::new(graph, plan, model, gguf, max_seq)?;
        Ok(Self {
            executor,
            tokenizer,
            config,
        })
    }

    /// Reset KV state for a fresh prompt.
    pub fn reset(&mut self) -> HipResult<()> {
        self.executor.reset_kv_cache()
    }

    /// Tokenise the prompt, run prefill + decode, detokenise the
    /// result. `apply_chat_template = true` wraps the prompt in the
    /// architecture-appropriate instruct template.
    pub fn generate(
        &mut self,
        prompt: &str,
        max_tokens: usize,
        sampling: &SamplingConfig,
        apply_chat_template: bool,
    ) -> HipResult<GenerationResult> {
        let start = Instant::now();

        // 1) Format + tokenise.
        let formatted = if apply_chat_template {
            self.tokenizer.apply_chat_template(prompt, None)
        } else {
            prompt.to_string()
        };
        let prompt_tokens = self.tokenizer.encode(&formatted, true);
        let prompt_len = prompt_tokens.len();

        if prompt_len == 0 {
            return Err(super::super::backend::gpu::error::HipError {
                code: -1,
                message: "empty tokenised prompt".into(),
                context: "generate".into(),
            });
        }

        // 2) Prefill — sequential decode over prompt tokens.
        let prefill_start = Instant::now();
        let mut last_logits = Vec::new();
        for (i, &tok) in prompt_tokens.iter().enumerate() {
            last_logits = self.executor.execute_decode(tok, i)?;
        }
        let prefill_ms = prefill_start.elapsed().as_secs_f64() * 1000.0;

        // 3) Decode loop.
        let decode_start = Instant::now();
        let mut generated = Vec::with_capacity(max_tokens);
        let mut all_tokens = prompt_tokens.clone();
        let mut hit_eos = false;

        for step in 0..max_tokens {
            let next_tok = sample_token(&mut last_logits, sampling, &all_tokens);
            if self.tokenizer.is_eos(next_tok) {
                hit_eos = true;
                break;
            }
            generated.push(next_tok);
            all_tokens.push(next_tok);
            if step + 1 == max_tokens {
                break;
            }
            last_logits = self.executor.execute_decode(next_tok, prompt_len + step)?;
        }
        let decode_ms = decode_start.elapsed().as_secs_f64() * 1000.0;
        let total_ms = start.elapsed().as_secs_f64() * 1000.0;

        let raw_output = self.tokenizer.decode(&generated, true);
        // Strip Qwen3 <think>…</think> blocks. A no-op for non-Qwen
        // architectures (no tags present) and for Qwen3 with the
        // `/no_think` directive applied (no tags emitted).
        let output = Tokenizer::strip_think_block(&raw_output);

        let prefill_tok_s = if prefill_ms > 0.0 {
            prompt_len as f64 / (prefill_ms / 1000.0)
        } else {
            0.0
        };
        let decode_tok_s = if decode_ms > 0.0 {
            generated.len() as f64 / (decode_ms / 1000.0)
        } else {
            0.0
        };

        Ok(GenerationResult {
            prompt: prompt.to_string(),
            output,
            prompt_tokens: prompt_len,
            generated_tokens: generated.len(),
            prefill_ms,
            decode_ms,
            total_ms,
            prefill_tok_s,
            decode_tok_s,
            hit_eos,
        })
    }
}
