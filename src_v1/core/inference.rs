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
    /// Säule 1 — ModelProfile from the load-time introspection scan.
    /// Phase 1 records it and surfaces warnings; Phase 2's precision
    /// GA and the FP32-overlay path consume it downstream.
    pub profile: super::super::introspection::ModelProfile,
    /// Säule 5 — Quality Monitor. Phase 1 logs drift events;
    /// Phase 2 uses the signals to escalate precision + rewind.
    pub monitor: super::super::monitor::QualityMonitor,
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

        // Säule 1 — one-shot introspection. Cheap (< 5s for 8B) and
        // drops a summary to the log. If the SNR risk score lands in
        // the warn band we print a visible notice so the operator
        // knows this model *might* need precision hints later.
        let profile = super::super::introspection::introspect(gguf);
        profile.print_summary();
        if profile.snr_risk_score < 2.0 {
            eprintln!(
                "⚠  SNR risk score {:.2} — precision upgrade may be needed (Phase 2 GA)",
                profile.snr_risk_score
            );
            eprintln!(
                "   critical embedding tokens: {} of {} vocab",
                profile.critical_embedding_tokens.len(),
                config.vocab_size
            );
        }

        let executor = GraphExecutor::new(graph, plan, model, gguf, max_seq)?;
        // Monitor stays un-calibrated at this point — calibration
        // requires a decoding pass, which callers trigger via
        // `calibrate_monitor()` after construction.
        let monitor = super::super::monitor::QualityMonitor::new(32, 3.0);
        Ok(Self {
            executor,
            tokenizer,
            config,
            profile,
            monitor,
        })
    }

    /// One-shot calibration: decode the Arch-Doc reference prompt
    /// for a handful of steps, collect (mean_abs, max_abs) at the
    /// final hidden-state buffer, install the band for
    /// `OUTPUT_HIDDEN`. Runs on-demand so the pipeline stays cheap
    /// to construct for callers that don't need the monitor.
    pub fn calibrate_monitor(&mut self) -> HipResult<()> {
        use super::super::monitor::{CALIBRATION_PROMPT, MIN_CALIBRATION_STEPS, OUTPUT_HIDDEN};

        self.reset()?;
        let prompt_tokens = self.tokenizer.encode(CALIBRATION_PROMPT, true);
        if prompt_tokens.is_empty() {
            return Ok(());
        }

        // Prefill.
        let mut last_logits = Vec::new();
        for (i, &tok) in prompt_tokens.iter().enumerate() {
            last_logits = self.executor.execute_decode(tok, i)?;
        }

        // Greedy decode a few more steps and harvest hidden-state
        // stats after each one. 10 steps is the Phase-1 minimum
        // for a stable stddev; callers can bump this later.
        let sampling = SamplingConfig::greedy();
        let mut samples: Vec<(f32, f32)> = Vec::with_capacity(MIN_CALIBRATION_STEPS);
        let mut all_tokens = prompt_tokens.clone();
        for step in 0..MIN_CALIBRATION_STEPS {
            let next = sample_token(&mut last_logits, &sampling, &all_tokens);
            all_tokens.push(next);
            let hidden = self.executor.read_hidden_state()?;
            let n = hidden.len() as f32;
            let mean_abs = hidden.iter().map(|x| x.abs()).sum::<f32>() / n;
            let max_abs = hidden.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
            samples.push((mean_abs, max_abs));
            // Skip the very first step — its hidden state reflects
            // the end of prefill, which can be a magnitude-different
            // regime from steady-state decode.
            if step + 1 < MIN_CALIBRATION_STEPS {
                last_logits = self.executor.execute_decode(next, prompt_tokens.len() + step)?;
            }
        }
        // Drop the prefill-tail step so the band reflects decode
        // steady-state only.
        if samples.len() > 1 {
            samples.remove(0);
        }
        self.monitor.install_calibration(OUTPUT_HIDDEN, &samples);
        eprintln!(
            "[monitor] calibrated: mean_abs={:.4} ± {:.4}, max_abs={:.4} over {} steps",
            self.monitor.expected_ranges[&OUTPUT_HIDDEN].mean_abs_expected,
            self.monitor.expected_ranges[&OUTPUT_HIDDEN].mean_abs_stddev,
            self.monitor.expected_ranges[&OUTPUT_HIDDEN].max_abs_expected,
            samples.len()
        );
        self.reset()?;
        Ok(())
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

            // Säule 5 — Quality Monitor hooks.
            // Phase 1 only logs; nothing in the control flow
            // changes based on a drift signal.
            self.monitor.record_token(next_tok);
            if let Some(reason) = self.monitor.check_repetition(next_tok) {
                let event = super::super::monitor::RevisionEvent {
                    token_index: step as u64,
                    node_id: super::super::monitor::OUTPUT_HIDDEN,
                    signal: super::super::monitor::PrecisionRevisionSignal {
                        affected_node: super::super::monitor::OUTPUT_HIDDEN,
                        current_precision:
                            super::super::introspection::PrecisionHint::Fp16Scales,
                        recommended_precision:
                            super::super::introspection::PrecisionHint::Fp16Scales,
                        reason,
                    },
                    resolved: false,
                };
                eprintln!(
                    "[monitor] token {} — {:?}",
                    event.token_index, event.signal.reason
                );
                self.monitor.revision_log.push(event);
            }
            if self.monitor.should_check()
                && !self.monitor.expected_ranges.is_empty()
            {
                let hidden = self.executor.read_hidden_state()?;
                let node = super::super::monitor::OUTPUT_HIDDEN;
                if let Some(signal) = self.monitor.check_hidden_state(node, &hidden) {
                    eprintln!(
                        "[monitor] token {} — drift at {:?}: {:?}",
                        step, node, signal.reason
                    );
                    self.monitor.revision_log.push(
                        super::super::monitor::RevisionEvent {
                            token_index: step as u64,
                            node_id: node,
                            signal,
                            resolved: false,
                        },
                    );
                }
                self.monitor.reset_check_counter();
            }

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
