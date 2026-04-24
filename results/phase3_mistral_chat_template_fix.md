# Mistral Chat-Template Fix — Partial, Honest Report

**Date:** 2026-04-24
**Branch:** v1.0-dev
**Scope:** Chat-template disambiguation + GGUF-driven special-token registration.
**Outcome:** Infrastructure fixed cleanly. Mistral output still unusable; root cause is deeper than expected.

## What landed

### 1. `ChatTemplate` enum with arch-aware disambiguation

`src_v1/core/tokenizer.rs` — new enum with 7 variants (`Qwen2`, `Qwen3`, `Llama3`, `Llama2`, `MistralV3`, `Gemma`, `Generic`). Resolution at construction time via `ChatTemplate::detect(architecture, vocab_size, bos_id)`:

| arch | vocab | bos | → template |
|---|---:|---:|---|
| qwen3 | any | any | Qwen3 (ChatML + `/no_think`) |
| qwen2 | any | any | Qwen2 (ChatML plain) |
| gemma* | any | any | Gemma |
| llama | 128256 | 128000 | Llama3 ✓ Llama-3.1, DeepSeek-R1-Distill |
| llama | 32768 | 1 | **MistralV3** ✓ Mistral-7B-v0.3 |
| llama | 32000 | 1 | Llama2 (kept for future) |
| llama | other | other | Llama3 (warns, fallback) |
| _ | any | any | Generic |

Template strings added for MistralV3, Llama2, Gemma (continuation included).

### 2. GGUF `token_type`-driven special-token registration

`src_v1/core/tokenizer.rs` reads `tokenizer.ggml.token_type` (i32 array). IDs where type ∈ {3=CONTROL, 4=USER_DEFINED} are registered as BPE special tokens via a new `V0xBpe::register_special_token_ids` method on `src/tokenizer/bpe.rs`. Replaces the old `<…>`-only heuristic which missed Mistral's `[…]`-bracket family.

## Validation

### Template detection (log line at load time)

```
Mistral-7B-v0.3:  chat_template: MistralV3 (arch=llama, vocab=32768, bos=Some(1))
Llama-3.1:        chat_template: Llama3    (arch=llama, vocab=128256, bos=Some(128000))
Qwen3-8B:         chat_template: Qwen3     (arch=qwen3, vocab=151936, bos=Some(151643))
DeepSeek-R1-Dist: chat_template: Llama3    (arch=llama, vocab=128256, bos=Some(128000))
```

All correct.

### Mistral special-token encoding (post-fix)

```
Before fix:  [INST] → [29560, 29505, 29527, 29503, 29506, 29561]  (6 byte-fallback tokens)
After fix:   [INST] → [3]                                          (single token, matches vocab)
Before fix:  [/INST] → [29560, 29516, 29505, 29527, 29503, 29506, 29561]
After fix:   [/INST] → [4]
```

Confirmed via `tests_v1/mistral_token_probe.rs`.

### Prompt-token-count reduction (Mutex prompt)

| | Prompt tokens | Notes |
|---|---:|---|
| **Before any fix** | **240** | Llama-3 header literals split byte-wise by Mistral BPE |
| After template fix only | 66 | `[INST]`/`[/INST]` still split as 6 tokens each |
| **After template + special-token fix** | **55** | `[INST]`/`[/INST]` single IDs |
| llama.cpp reference | 16 | Our middle-text tokenisation is still 3× too long |

### Mistral coherence (Mutex prompt)

**Still garbage.** Representative output after all fixes:

> *It seems like you've provided a string of characters that doesn't appear to be in a valid Unicode text, but it looks like a mix of uppercase and lowercase Latin letters, numbers, and special characters…*

The model correctly recognises the text as garbled — i.e. it is receiving wrong tokens for the body of the prompt.

## Root cause of the residual garbage

**Tokenizer preset mismatch.** Mistral v0.3 uses LLaMA-1-style **SentencePiece** (SPM) with BPE merges over `▁`-prefixed word pieces. Our v0.x BPE has only two presets (`Qwen2`/GPT-2-BPE and `LlamaBpe` a.k.a. Llama-3 GPT-4-style BPE). Both emit `Ġ` for word-initial space — not `▁`. Mistral's vocab IDs for " Explain" (llama.cpp: token 14470) don't match what our encoder produces from `"ĠExplain"`.

Evidence from the probe:

```
llama.cpp:  14470 -> ' Expl',  1194 -> 'ain'
we produce: 29517 29512 29488 29482 29476 29478 29479   (byte-fallback per char)
```

The middle of every prompt is being encoded character-by-character via byte-fallback because none of our GPT-2 BPE merges exist in Mistral's SPM merge table. Fixing this requires a **full SentencePiece tokenizer preset** (read `tokenizer.ggml.scores`, implement SPM's longest-match + merge-rank algorithm). That's a separate, larger task — several days, not 2-3 h.

## Qwen3 + Llama-3.1 regression

Re-ran the Mutex prompt through both models post-fix:

- **Qwen3-8B**: identical output, 33 prompt tokens, 54.7 tok/s. **No regression.**
- **Llama-3.1-8B**: identical output (same mangled mutex text from before), 31 prompt tokens, 59.9 tok/s. **No regression.**

Both use the existing GPT-2 BPE preset via `LlamaBpe`/`Qwen2` and were unaffected by the special-token-registration change (their control tokens were already caught by the `<…>` heuristic).

## v1.1 follow-up

Upgraded the `results/phase3_multi_model_compatibility.md` backlog entry #1 from "~2-3 h" to split into two sub-tasks:

1. **Mistral chat-template + special-token handling** — **DONE** this commit.
2. **SentencePiece tokenizer preset** — **NEW**, required to make Mistral usable. Estimated 2-3 days of focused work. Will also benefit any Llama-1/2-derived model (Llama-2-7B, Vicuna, CodeLlama, OpenOrca).

## Files changed

- `src_v1/core/tokenizer.rs` — `ChatTemplate` enum + 5 new templates + `token_type`-based special registration
- `src/tokenizer/bpe.rs` — new `register_special_token_ids` method
- `tests_v1/mistral_token_probe.rs` — new, verifies `[INST]`/`[/INST]` single-token encoding
- `Cargo.toml` — registers the new probe
