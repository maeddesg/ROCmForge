//! End-to-end inference pipeline: tokenize → prefill → decode loop.
//!
//! Phase 1 wraps the `GraphExecutor` from Schritt 1.10 with a tokenizer
//! and sampler. Prefill is implemented as a sequential decode over the
//! prompt tokens (the executor's per-token API); WMMA-batched prefill
//! is a Phase-2 optimisation.

use std::time::Instant;

use super::super::backend::gpu::error::HipResult;
use super::super::core::model_loader::LoadedModel;
use super::super::graph::{BufferPlan, ComputationGraph, GraphExecutor};
use super::gguf::GGUFFile;
use super::model_config::ModelConfig;
use super::sampling::{sample_token, SamplingConfig};
use super::streaming::StreamingEmitter;
use super::tokenizer::Tokenizer;

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

    /// Phase 2.4 — multi-turn state. `kv_pos` is the number of tokens
    /// currently in the KV cache (i.e. the position where the next
    /// prefill / decode should write). `turn_count` tracks which turn
    /// of the current conversation we're on (0 = first turn → full
    /// chat template with system prompt; 1+ → continuation template).
    /// Both are reset by `reset_conversation()`.
    pub kv_pos: usize,
    pub turn_count: usize,
    /// Configured max sequence length — used to detect KV-cache
    /// overflow and refuse the turn before corrupting state. Set at
    /// pipeline construction from the `max_seq` argument.
    pub max_seq: usize,
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

        // Säule 1 — one-shot introspection. Cheap (< 5s for 8B).
        // The summary is not auto-printed; the CLI gates this via
        // `--show-introspection` so programmatic callers don't get
        // 20 lines of stderr noise per pipeline. The one-line SNR
        // warn is kept — it's a single line and unambiguously
        // signals a model that needs a second look.
        let profile = super::super::introspection::introspect(gguf);
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
            kv_pos: 0,
            turn_count: 0,
            max_seq,
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
                last_logits = self
                    .executor
                    .execute_decode(next, prompt_tokens.len() + step)?;
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

    /// Reset KV state for a fresh prompt. **Single-turn path:** the
    /// existing `generate()` call wraps this, so every one-off prompt
    /// starts with an empty cache. For multi-turn conversations use
    /// `reset_conversation()` (same semantics; clearer name) and don't
    /// call it between turns.
    pub fn reset(&mut self) -> HipResult<()> {
        self.reset_conversation()
    }

    /// Phase 2.4 — explicit conversation reset. Clears the KV cache
    /// AND the multi-turn position tracker. Call this when the user
    /// asks for a fresh chat (`/reset` in the CLI) or at pipeline
    /// startup. A freshly-constructed pipeline is already reset.
    pub fn reset_conversation(&mut self) -> HipResult<()> {
        self.executor.reset_kv_cache()?;
        self.kv_pos = 0;
        self.turn_count = 0;
        Ok(())
    }

    /// Current number of tokens in the KV cache (= position where the
    /// next prefill/decode will write). 0 right after
    /// `reset_conversation()`; grows monotonically with each turn.
    pub fn kv_pos(&self) -> usize {
        self.kv_pos
    }

    /// Tokenise the prompt, run prefill + decode, detokenise the
    /// result. `apply_chat_template = true` wraps the prompt in the
    /// architecture-appropriate instruct template.
    ///
    /// **Phase-1 behaviour preserved:** this wrapper resets the
    /// conversation before generating, so each call is independent.
    /// Multi-turn callers should instead use `generate_turn()` /
    /// `generate_turn_streaming()` and call `reset_conversation()`
    /// only when a fresh chat is wanted.
    pub fn generate(
        &mut self,
        prompt: &str,
        max_tokens: usize,
        sampling: &SamplingConfig,
        apply_chat_template: bool,
    ) -> HipResult<GenerationResult> {
        self.reset_conversation()?;
        self.generate_turn(prompt, max_tokens, sampling, apply_chat_template)
    }

    /// Phase 2.4 — multi-turn generation step. Uses `self.kv_pos` as
    /// the starting prefill offset and updates it after prefill and
    /// each decoded token. Picks the initial-turn template for
    /// `turn_count == 0`, the continuation template for later turns.
    /// Does NOT reset between turns — call `reset_conversation()`
    /// explicitly when the user wants a fresh chat.
    pub fn generate_turn(
        &mut self,
        prompt: &str,
        max_tokens: usize,
        sampling: &SamplingConfig,
        apply_chat_template: bool,
    ) -> HipResult<GenerationResult> {
        let mut buffer = String::new();
        let result = self.generate_turn_streaming_internal(
            prompt,
            max_tokens,
            sampling,
            apply_chat_template,
            /*filter_think=*/ true,
            &mut |piece| buffer.push_str(piece),
        )?;
        let mut r = result;
        // Stripped output = whatever the streaming emitter forwarded
        // (already think-filtered). The legacy `strip_think_block`
        // pass is still a no-op safety net in case the emitter missed
        // anything — e.g. a closing tag that arrived right at EOS.
        r.output = Tokenizer::strip_think_block(&buffer);
        Ok(r)
    }

    /// Phase 2.4 — streaming generation with a per-token callback.
    /// The callback fires with the **filtered, user-visible** text
    /// for each piece (think-tags stripped live). Returns a
    /// `GenerationResult` whose `output` field is the cumulative
    /// emitted text — identical to `generate_turn()`'s output.
    ///
    /// `filter_think=true` is the default; pass `false` via the
    /// lower-level call if `<think>` blocks should be visible.
    pub fn generate_turn_streaming<F>(
        &mut self,
        prompt: &str,
        max_tokens: usize,
        sampling: &SamplingConfig,
        apply_chat_template: bool,
        filter_think: bool,
        mut on_token: F,
    ) -> HipResult<GenerationResult>
    where
        F: FnMut(&str),
    {
        let mut emitted = String::new();
        let mut cb = |piece: &str| {
            emitted.push_str(piece);
            on_token(piece);
        };
        let mut result = self.generate_turn_streaming_internal(
            prompt,
            max_tokens,
            sampling,
            apply_chat_template,
            filter_think,
            &mut cb,
        )?;
        result.output = emitted;
        Ok(result)
    }

    /// Shared core between `generate_turn` and `generate_turn_streaming`.
    /// Emits detokenised pieces through `emit` with `<think>`-tag
    /// filtering applied; the caller decides what to do with them
    /// (print, buffer, whatever).
    fn generate_turn_streaming_internal(
        &mut self,
        prompt: &str,
        max_tokens: usize,
        sampling: &SamplingConfig,
        apply_chat_template: bool,
        filter_think: bool,
        emit: &mut dyn FnMut(&str),
    ) -> HipResult<GenerationResult> {
        let start = Instant::now();

        // 1) Format + tokenise. Continuation template skips the system
        // prompt when we're already mid-conversation.
        let formatted = if apply_chat_template {
            if self.turn_count == 0 {
                self.tokenizer.apply_chat_template(prompt, None)
            } else {
                self.tokenizer.apply_chat_template_continuation(prompt)
            }
        } else {
            prompt.to_string()
        };
        // `add_bos = false` on continuation turns so we don't re-emit
        // <|begin_of_text|> in the middle of a Llama-3 conversation.
        let add_bos = self.turn_count == 0;
        let prompt_tokens = self.tokenizer.encode(&formatted, add_bos);
        let prompt_len = prompt_tokens.len();

        if prompt_len == 0 {
            return Err(super::super::backend::gpu::error::HipError {
                code: -1,
                message: "empty tokenised prompt".into(),
                context: "generate".into(),
            });
        }

        // Overflow-policy: stop-and-warn. No rolling cache, no auto-
        // reset — the user must call `reset_conversation()` explicitly.
        let pos_offset = self.kv_pos;
        if pos_offset + prompt_len + max_tokens > self.max_seq {
            return Err(super::super::backend::gpu::error::HipError {
                code: -1,
                message: format!(
                    "context overflow: {} prompt + {} max_tokens + {} cached > {} max_seq. \
                     Call reset_conversation() or reduce max_tokens.",
                    prompt_len, max_tokens, pos_offset, self.max_seq
                ),
                context: "generate_turn".into(),
            });
        }

        // 2) Prefill at the current cache position. `execute_prefill`
        // handles both the WMMA-batched long-prefill path and the
        // sequential decode-loop fallback, both of which respect the
        // pos_offset and write into the KV cache at that offset.
        let prefill_start = Instant::now();
        let mut last_logits = self.executor.execute_prefill(&prompt_tokens, pos_offset)?;
        let prefill_ms = prefill_start.elapsed().as_secs_f64() * 1000.0;
        // After prefill, `prompt_len` tokens sit at [pos_offset ..
        // pos_offset+prompt_len). The next decode writes at
        // pos_offset+prompt_len.
        let mut cache_pos = pos_offset + prompt_len;

        // 3) Decode loop — stream pieces through the StreamingEmitter
        // so `<think>` blocks are filtered live (Qwen3 mostly avoids
        // them under the /no_think directive, but sometimes leaks one).
        let decode_start = Instant::now();
        let mut emitter = StreamingEmitter::new(filter_think);
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

            // Live detokenisation: decode just this one new token and
            // hand the piece to the emitter. `decode(&[tok], true)`
            // uses the tokenizer's detokenise-one-token path which
            // correctly handles SentencePiece byte-fallback pieces.
            let piece = self.tokenizer.decode(std::slice::from_ref(&next_tok), true);
            if let Some(out) = emitter.process(&piece) {
                emit(&out);
            }

            // Säule 5 — Quality Monitor hooks (unchanged from Phase 1).
            self.monitor.record_token(next_tok);
            if let Some(reason) = self.monitor.check_repetition(next_tok) {
                let event = super::super::monitor::RevisionEvent {
                    token_index: step as u64,
                    node_id: super::super::monitor::OUTPUT_HIDDEN,
                    signal: super::super::monitor::PrecisionRevisionSignal {
                        affected_node: super::super::monitor::OUTPUT_HIDDEN,
                        current_precision: super::super::introspection::PrecisionHint::Fp16Scales,
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
            if self.monitor.should_check() && !self.monitor.expected_ranges.is_empty() {
                let hidden = self.executor.read_hidden_state()?;
                let node = super::super::monitor::OUTPUT_HIDDEN;
                if let Some(signal) = self.monitor.check_hidden_state(node, &hidden) {
                    eprintln!(
                        "[monitor] token {} — drift at {:?}: {:?}",
                        step, node, signal.reason
                    );
                    self.monitor
                        .revision_log
                        .push(super::super::monitor::RevisionEvent {
                            token_index: step as u64,
                            node_id: node,
                            signal,
                            resolved: false,
                        });
                }
                self.monitor.reset_check_counter();
            }

            // Always write `next_tok` to the KV cache before deciding
            // whether to stop — otherwise a max-tokens cutoff leaves
            // the last assistant token out of the cache, which breaks
            // continuation: the next turn's template would reference
            // a partial assistant message. The returned logits are
            // discarded on the final iteration.
            last_logits = self.executor.execute_decode(next_tok, cache_pos)?;
            cache_pos += 1;
            if step + 1 == max_tokens {
                break;
            }
        }
        // Final flush of any trailing buffered text (e.g. a `<` that
        // never became `<think>`).
        if let Some(tail) = emitter.flush() {
            emit(&tail);
        }
        let decode_ms = decode_start.elapsed().as_secs_f64() * 1000.0;
        let total_ms = start.elapsed().as_secs_f64() * 1000.0;

        // Update multi-turn state. The tokens written to the KV cache
        // are the prompt + the generated (non-EOS) tokens. The EOS
        // itself was sampled but NOT dispatched through execute_decode
        // (we break before that), so it's NOT in the cache — a
        // follow-up turn's continuation template injects the
        // `<|im_end|>` marker for the previous assistant turn.
        self.kv_pos = cache_pos;
        self.turn_count += 1;

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

        // `output` gets finalised by the caller (it knows whether to
        // rely on emitter output or run strip_think_block as safety).
        Ok(GenerationResult {
            prompt: prompt.to_string(),
            output: String::new(),
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
