# Llama-3.1 multi-turn — prefill divergence on prior-turn prompts

**Status:** OPEN — not blocking. Workaround: use Qwen3 for interactive
chat, or use Llama-3.1 in single-turn mode only.

**Discovered:** 2026-04-19 during v0.2.1 chat polish testing.

## Summary

On `Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf`, the chat CLI produces a
correct first-turn reply and then collapses from turn 2 onward. The
second-turn reply is typically a 2–4 token echo of the first turn's
topic rather than an answer to the new question.

Confirmed to be a **pure prefill bug**, not a chat-loop or KV-state
issue: feeding the exact same formatted prompt via
`rocmforge --gpu --no-template --prompt "…"` reproduces the collapse
identically.

## Reproduction

Build:
```
cargo build --release --features gpu
```

Multi-turn chat (fails from turn 2):
```
./target/release/rocmforge chat \
    --model ~/models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf \
    --max-tokens 30 --temperature 0.0 --top-p 1.0
> What is the capital of France?
  The capital of France is Paris.
> Hello how are you?
  Paris.
> What is 2+2?
  The capital of France
```

Equivalent single-shot prompt (also fails):
```
./target/release/rocmforge --gpu --no-template \
    --model ~/models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf \
    --max-tokens 20 --temperature 0.0 --top-p 1.0 \
    --prompt $'<|start_header_id|>system<|end_header_id|>\n\nYou are a helpful assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nWhat is the capital of France?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\nThe capital of France is Paris.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nHello how are you?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n'
```
Expected: a greeting response. Observed: `Paris.`

## Diagnosis

Ruled out on 2026-04-19:

| Hypothesis | Test | Result |
|---|---|---|
| Stale KV between turns | `kv.clear()` before every `gpu_prefill_forward_hybrid` call | no change |
| Chat-loop / history bookkeeping | reproduce as single-shot `--no-template --prompt` | same collapse |
| Special-token embedding dequant error (Q4_K, L2≈0.03) at last prefill position | FP32 overlay for all 259 `<…>` tokens (Option B) | no change |
| WMMA GQA+causal prefill attention kernel | `ROCMFORGE_DISABLE_WMMA_ATTENTION=1` | no change |
| Decode-graph replay state | `ROCMFORGE_DISABLE_DECODE_GRAPH=1` | no change |
| Tiled batched GEMV | `ROCMFORGE_DISABLE_TILED_GEMV=1` | no change |
| `rope_freqs` (Llama-3.1 per-dim RoPE scaling) | force `rope_freqs_ptr() = None` | no change |
| `ROCMFORGE_GPU_SAFE_MODE=1` | — | produces `!!!!` (pre-existing safe-mode bug, not comparable) |

Additional observations:

- The last prefill token is identical between turn 1 (28 tok) and turn
  2 (50 tok) — both end `[…, 128007, 271]` (`<|end_header_id|>\n\n`,
  token 271 is a normal token, not a special). The original
  "small-magnitude special token at final position" theory is therefore
  invalidated.
- Turn 1 → Turn 2 with `/clear` in between produces a coherent reply.
  But `/clear` only drops `ctx.conversation_history`, it does NOT clear
  the KV cache — so the difference is the prompt content, not the KV
  state.
- Short multi-turn prompts (`Hi` / `Hello`, 22→40 tok) produce
  coherent replies. A 43-token *single-turn* prompt works. The 50-token
  turn-2 prompt fails. So length alone is not the trigger.
- Qwen2.5 Q4_0 multi-turn works correctly on the same code path.
  Qwen3 Q4_K_M multi-turn also works. The bug is specific to the
  Llama-3.1 Q4_K_M combination.
- Prepending a literal `<|begin_of_text|>` (so the tokenizer
  double-BOSes → 51 tokens) produces a *different* coherent reply
  ("You're asking about …"), but trivial suffix perturbations (adding
  one or two tokens to the user question) still collapse. So the bug
  is position/content dependent, not bound to a specific seq_len.

## Next steps (not yet attempted)

1. **Layer-by-layer hidden-state diff against llama.cpp** for the exact
   50-token turn-2 sequence. The first layer where the rocmforge
   `scratch.hidden` diverges from the reference identifies the
   responsible kernel. This is the single highest-signal experiment.
2. **Numerical audit of WMMA GQA+causal prefill attention** at
   `seq_len = 50`, `num_heads = 32`, `num_kv_heads = 8`, `head_dim = 128`.
   Llama-3.1 is the only supported model with a 4:1 GQA ratio at this
   seq_len; Qwen2.5-7B uses 7 KV heads (non-integer-ratio → different
   WMMA scheduling).
3. **RoPE application on `kv_heads` at the 8:32 GQA mapping.** The
   `rope_freqs` buffer is shared between Q and K projection; a
   position-indexing bug would manifest only on Llama-3.1 because it
   is the only model with a non-trivial `rope_freqs` table among the
   supported three.

## Workaround

- Use Qwen3 for interactive chat.
- Run Llama-3.1 only in single-turn mode:
  ```
  rocmforge --gpu --model Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf \
            --prompt "…" --max-tokens 256
  ```
  Single-turn replies are correct and match llama.cpp on the prompts
  tested.

## Context links

- Investigation notes: `memory/project_llama3_multiturn_bug.md` (local
  agent memory, not checked in).
- Q4_K_M format analysis: `docs/q4_k_m_block_format.md`.
- v0.2.1 changelog entry referencing this file: see `CHANGELOG.md`.
