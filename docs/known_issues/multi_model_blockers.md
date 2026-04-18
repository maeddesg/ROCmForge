# Phase 7 Step 5 — multi-model integration blockers

**Status:** Qwen2.5-7B Q4_0 works (v0.1.0 baseline).
Qwen3-8B Q4_K_M and Llama-3.1-8B Q4_K_M **load and dispatch but produce
gibberish end-to-end.** Two distinct upstream blockers explain this; the
WMMA / GEMV kernel path itself is correct.

## What Step 5 already fixed

Commit on `main`:

- `qwen3` traits: `use_attention_bias: false` (was `true`) — Qwen3-8B
  dense has no `.attn_q.bias` / `.attn_k.bias` / `.attn_v.bias` tensors,
  unlike Qwen2.5-7B.
- `qwen3` traits: `tensor_naming: Gguf` (was `GgufMoE`) — the dense
  8B uses standard GGUF naming. `qwen3moe` gets its own entry with
  `GgufMoE` naming for MoE variants.
- `llama` traits: `rope_style: NeoX` (was `Normal`) — HF→GGUF
  conversion preserves the `rotate_half` convention, so GGUF expects
  the split-half (NeoX) RoPE layout.

Qwen2.5-7B Q4_0 verified unchanged: "Paris. It is located in the
northern central part" at 102 tok/s, matching the v0.1.0 numbers.

All six correctness-test suites still green (37 tests):
`wmma_q4_0`, `wmma_q4_1`, `wmma_q4_k`, `gemv_q4_k`, `wmma_padding`,
`chat_single_turn`, `chat_multi_turn`.

## Remaining blockers

### Qwen3 — missing Q/K normalisation

**Symptom:** output is repetitive garbage (`ssss...`, `ff|心math...`)
even after the trait fixes above. Dispatch log shows all GEMV calls
route correctly (`path="gemv_decode"` with `wtype=Q4_K`).

**Root cause:** Qwen3 adds a per-head RMSNorm on Q and K *after* the
projection and *before* RoPE. The tensors are present in the GGUF but
the forward pass doesn't apply them:

```
$ python3 -c '...' Qwen3-8B-Q4_K_M.gguf | grep norm
  blk.N.attn_q_norm.weight      # ← applied to Q before RoPE
  blk.N.attn_k_norm.weight      # ← applied to K before RoPE
  blk.N.attn_norm.weight        # pre-attention RMSNorm (already handled)
  blk.N.ffn_norm.weight         # pre-FFN RMSNorm (already handled)
```

Neither Qwen2.5 nor Llama uses these tensors. ROCmForge's forward path
has no hook for them.

**Scope of the fix**

1. `TensorNameRegistry` — register `AttnQNorm(layer)` and `AttnKNorm(layer)`
   resolvers for the `qwen3` scheme.
2. `GpuLayerWeights` — carry `Option<GpuBuffer>` entries for `attn_q_norm`
   / `attn_k_norm`, populated only when the tensor exists.
3. `gpu_prefill_layer_forward_hybrid` and `gpu_layer_forward_hybrid`
   (both paths) — insert a head-dim RMSNorm on Q and on K between the
   projection stage and the RoPE stage. `head_dim=128` → the per-head
   vectors are 128 floats, well-suited to the existing `rms_norm_rows`
   kernel with `hidden_size=128` and `seq_len = num_heads*seq_len`
   (flatten head axis into batch).
4. A `ModelTraits::use_qk_norm: bool` flag so the new pass only fires
   for Qwen3.

Estimated effort: half a day for the forward-pass wiring + a CPU
reference test on a single layer.

### Llama-3.1 — `llama-bpe` pre-tokeniser not implemented

**Symptom:** output is garbage with a sprinkle of plausibly
relevant tokens (`assistant` appears at rank 4 with 0.4 % probability
for the prompt "The capital of France is", but top-1 is still `\n`).
Dispatch log is clean; the model's forward pass produces a sensible
distribution but is fed the wrong input-token sequence.

**Root cause:** the GGUF metadata:

```
Llama-3.1:  tokenizer.ggml.pre = "llama-bpe"
Qwen2/3:    tokenizer.ggml.pre = "qwen2"
```

ROCmForge's `src/tokenizer/bpe.rs` has a single pre-tokeniser regex
path hard-wired for the Qwen2 convention (simplified whitespace +
punctuation split). Llama-BPE uses the GPT-4-style regex:

```
(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}|
 ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+
```

With the wrong pre-split, the same UTF-8 prompt tokenises to a
completely different ID stream than what Llama-3 was trained on →
sequence looks like noise → flat output distribution.

Additionally: Llama-3 expects `<|begin_of_text|>` (id 128000) as the
BOS token. The current `add_bos` handling in
`src/tokenizer/bpe.rs` respects the GGUF metadata field, but the
Llama-3 GGUF omits `add_bos_token`, which ROCmForge currently
interprets as "no BOS." Llama's convention is BOS-always.

**Scope of the fix**

1. Pipe `tokenizer.ggml.pre` through the loader to
   `BpeTokenizer::new`.
2. Branch on the string: `"qwen2"` keeps the current regex;
   `"llama-bpe"` adopts the GPT-4 regex. Add a `TokenizerPreset` enum
   to keep the string-dispatch in one place.
3. Default `add_bos = true` when `pre == "llama-bpe"` regardless of
   the absent metadata key.
4. Port the regex — the `regex` crate handles the above pattern if
   compiled with `unicode-perl` (already a dependency).

Estimated effort: half a day plus round-trip tests against
llama.cpp's `main` tokenisation of a fixed corpus (the standard
verification for pre-tokeniser parity).

### Q6_K mixed-precision fall-through

**Symptom:** on Qwen3-8B and Llama-3.1-8B the dispatch log shows a
handful of lines like `wtype=Q6_K path="gemv_decode"` interleaved with
the Q4_K dispatches. The `gemv_q6_k_f32` pre-existing kernel handles
those correctly, but it's an older scalar-ish implementation.

**Measured impact:** decode on both models currently lands at
~20 tok/s with the tokenizer issue above. Once the tokeniser is
fixed and a fair measurement is possible, the Q6_K path *might*
still be the bottleneck on certain layers — or it might be entirely
dominated by the Q4_K path. Not worth optimising until we can
actually read the profiling data from a decode that produces
coherent output.

**Recommendation:** defer. Address after the Q/K-norm and
pre-tokeniser fixes unblock end-to-end output.

## Revised Step-5 definition of done

The user-facing goal ("coherent text on Qwen3-8B and Llama-3.1-8B")
requires the two follow-ups above, not just config trait toggles.
This landing is **partial progress** — it closes the trait-level
mismatches and documents the remaining architectural work with
concrete scoping.

Next actionable work, in priority order:

1. **Llama-BPE pre-tokeniser** (~half-day). Frees Llama-3.1-8B.
2. **Qwen3 Q/K-norm forward pass** (~half-day). Frees Qwen3-8B.
3. **Integration benchmark** on both models against llama.cpp 87 / 93
   tok/s targets from the Phase 7 Step 1 baseline.
