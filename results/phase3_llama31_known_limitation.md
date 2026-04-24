# Llama-3.1-8B-Instruct-Q4_K_M — Known Limitation

**Status:** Open, root cause unidentified after a 4-hypothesis code audit + tokenizer round-trip probe vs llama.cpp.
**Symptom:** Llama-3.1 produces grammatically fluent but instruction-blind output; does not reliably follow single-turn instructions ("remember Alice") and falls into repetition loops under pure-greedy sampling.
**Workaround:** CLI auto-escalates to `greedy + repeat_penalty = 1.1` when `ModelProfile.snr_risk_score < 2.0`. This keeps the decode coherent and stops the loop, but does not restore instruction-following.
**Quality on the 15-prompt suite (2026-04-24):** 1–3/15 coherent vs Qwen3-8B at 15/15. Decode throughput is 105.4 tok/s (faster than Qwen3's 96.2).

## Ruled-out causes (code audit + GGUF probe)

| # | Hypothesis | Status | Evidence |
|---|---|---|---|
| 1 | Embedding Q4_K dequant noise destroys decode path | **Ruled out** | `src_v1/graph/executor.rs:301–416` dequantises the whole embedding table to FP32 once at load; decode reads FP32 via `rocmforge_launch_embedding_lookup`. A "FP32 overlay" would read the same Q4_K bytes with the same dequant → identical values. |
| 2 | Tied weights (`output.weight == token_embd.weight`) | **Ruled out** | GGUF probe: `output.weight` is a separate Q6_K tensor `[4096, 128256]`. Identical to Qwen3's layout. |
| 3 | RoPE `freq_base` wiring | **Ruled out** | `config.rope_freq_base` = 500000 for Llama-3.1 (1000000 for Qwen3), threaded cleanly through `GraphBuilder → GraphNode::Rope.theta_base → rocmforge_launch_rope`. |
| 4 | `rope_freqs.weight` (Llama-3.1 NTK-ramp, values 1.0→8.0 across head_dim/2) | **Ruled out (by inspection)** | Tensor is loaded into Zone A, threaded through both decode `executor.rs:1039` and prefill `:2270` paths. Kernel formula `theta /= freq_scale[i]` matches llama.cpp. A direct runtime probe of logits-with vs logits-without rope_freqs was not done in this pass. |
| 5 | GQA head count | **Ruled out** | 32/8 for both Llama-3.1 and Qwen3. |
| 6 | BOS token injection | **Ruled out** | `add_bos_token` missing → defaults to `true` for `LlamaBpe` preset (`src/tokenizer/bpe.rs:108`). Inference adds BOS on turn 0 (`inference.rs:320`). |
| 7 | Chat-template special-token splitting | **Ruled out** | C1 probe `tests_v1/llama31_token_probe.rs` encodes all 5 Llama-3 special tokens as single IDs; full 14-token Turn-1 template matches llama.cpp's `llama-tokenize` bit-exact. |

## Plausible remaining causes (not yet probed)

- **rope_freqs runtime correctness** — code inspection matches llama.cpp, but numerical end-to-end behaviour not verified. Direct test: compare our logits vs llama.cpp's `llama-cli --logits-all` on identical single-token prompts.
- **Q4_K dequant parity at specific block/row indices** — `v1_parity_test` exists and passes on random blocks, but hasn't been run against the 182 critical-SNR rows specifically.
- **WMMA prefill edge case** — both Llama-3.1 and Qwen3 go through the same WMMA kernels, but Llama-3.1's specific weight distribution or the combination with `rope_freqs` in the batched rope path may trigger an issue invisible in unit tests.

## Files added during this investigation

- `tests_v1/llama31_root_cause_probe.rs` — dumps GGUF metadata for the 4 hypotheses
- `tests_v1/llama31_token_probe.rs` — special-token encoding parity vs vocab and llama.cpp

## Recommendation

Keep Llama-3.1-Q4_K_M in the test matrix as a CLI-reachable model, but **treat it as secondary** until the root cause is isolated. Primary development + benchmarking continues on Qwen3-8B-Q4_K_M which is fully coherent.
