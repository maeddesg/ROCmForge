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

### Llama-3.1 — `llama-bpe` pre-tokeniser (✓ landed Step 5a)

**Fixed** in `src/tokenizer/bpe.rs`:

- `TokenizerPreset` enum dispatches on `tokenizer.ggml.pre`
  (`"qwen2"` → existing Qwen regex, `"llama-bpe"` → GPT-4-style regex).
- The GPT-4 regex drops `\s+(?!\S)` (the `regex` crate lacks
  lookahead); a post-processing pass in
  `redistribute_whitespace_llama_bpe` emulates the same split, so
  output is byte-identical to `llama-tokenize`.
- `add_bos` defaults to `true` when the GGUF omits the key and the
  preset is `LlamaBpe` (Llama always needs BOS id 128000).
- `ChatTemplate::LLaMA3` drops the literal `<|begin_of_text|>` so
  the tokenizer's BOS path does not double-BOS.
- New parity test `tests/tokenizer_llama_bpe_parity.rs` verifies 7
  test strings against llama.cpp's `llama-tokenize` output on
  Meta-Llama-3.1-8B.

**Remaining blocker for Llama-3.1 (new, discovered during 5a):**
Llama-3.1 ships a `rope_freqs.weight` tensor (64×f32) in the top-level
of the GGUF that scales the per-dimension RoPE frequencies as part of
Meta's "rope_scaling" formula for 4 K → 128 K context extension. Even
at position 0 the scaled frequencies differ from the standard
`base^(-i/d)` table, so ignoring the tensor produces wrong RoPE for
every token. Tokeniser parity is verified against llama.cpp, but the
model still emits garbage — top-1 for "The capital of France is" is
`' '` (10.6 %), with "Paris" absent from the top-10. See new section
below.

### Llama-3.1 — `rope_freqs.weight` (RoPE scaling) not applied

**Symptom:** after the tokeniser fix above, Llama-3.1-8B dispatches
cleanly to WMMA Q4_K for every projection (no Q6_K, no scalar
fallbacks), reaches ~21 tok/s, but the per-token logit distribution
is flat and irrelevant. Top-1 for "The capital of France is" is
`' '` at 10.6 %, not `' Paris'`. `llama-cli` on the same model
produces coherent output.

**Root cause:** the GGUF contains
`rope_freqs.weight : f32[64]` with entries `[1.0, 1.0, …, 8.0, 8.0]`
(high-frequency dims unscaled, low-frequency dims scaled by the
Llama-3.1 formula — the factor-8 matches Meta's "factor: 8.0" in the
HF `rope_scaling` config). ROCmForge's RoPE kernel computes
`theta = base^(-2i/d)` from `rope_theta = 500 000` without consulting
this tensor, so it applies the wrong RoPE to every Q and K.

Qwen2.5, Qwen3 and pre-3.1 Llama models do not ship this tensor and
are unaffected.

**Scope of the fix**

1. `GpuLayerWeights` — carry an `Option<GpuBuffer>` for
   `rope_freqs` (uploaded once per model, shared across layers).
2. `gpu_rope_apply_on_stream` and its CPU twin — add an optional
   `freq_scale: Option<&[f32]>` argument. When present, the per-pair
   frequency is `base^(-2i/d) / freq_scale[i]` instead of the pure
   power-of-base formula.
3. `ModelConfig` / `ModelTraits` — a `has_rope_freqs: bool` flag
   plumbed from tensor existence would keep the path opt-in; no
   runtime string match.
4. Parity probe: run a single-token prefill and compare the resulting
   Q/K tensors layer-0 against an `llm-verify`-style reference from
   transformers.

Estimated effort: half a day (RoPE kernel argument + loader wiring +
a CPU reference test on one layer).

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

**Recommendation:** defer. Address after the Q/K-norm,
rope_freqs, and pre-tokeniser fixes unblock end-to-end output.

## Revised Step-5 definition of done

The user-facing goal ("coherent text on Qwen3-8B and Llama-3.1-8B")
requires three architectural follow-ups, not just config trait toggles.
Steps 5 (traits) and 5a (llama-bpe tokeniser) are landed and
regression-green; the remaining architectural work is scoped below.

Next actionable work, in priority order (updated after Step 5a):

1. **Llama-3.1 `rope_freqs.weight`** (~half-day). Frees Llama-3.1-8B.
   Smallest forward-pass change of the three; kernel already has a
   RoPE path, just needs the scale tensor piped through.
2. **Qwen3 Q/K-norm forward pass** (~half-day). Frees Qwen3-8B.
   Requires a new head-dim RMSNorm pass between projection and RoPE.
3. **Integration benchmark** on both models against llama.cpp 87 / 93
   tok/s targets from the Phase 7 Step 1 baseline.
