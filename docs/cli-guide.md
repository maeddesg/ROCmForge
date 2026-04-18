# ROCmForge CLI Guide

## Quick start

Interactive chat (recommended for most use cases):

```bash
rocmforge chat --model ~/models/Qwen2.5-7B-Instruct-Q4_0.gguf
```

One-shot inference (for scripting / benchmarking):

```bash
rocmforge --model ~/models/Qwen2.5-7B-Instruct-Q4_0.gguf \
    --prompt "Hello" --max-tokens 128 --gpu
```

## Subcommands

ROCmForge has one named subcommand (`chat`). If the first argument is
not a subcommand, the binary runs in one-shot mode.

### `rocmforge chat` — interactive chat

Multi-turn conversation with streaming output, Qwen2.5 ChatML template,
and slash commands. Token-by-token output at ~102 tok/s.

```bash
rocmforge chat --model ~/models/Qwen2.5-7B-Instruct-Q4_0.gguf
```

#### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--model <path>` | (required) | Path to GGUF model file |
| `--system <text>` | "You are a helpful assistant." | System prompt |
| `--max-tokens N` | 512 | Maximum tokens per reply |
| `--temperature F` | 0.0 | Sampling temperature (0.0 = greedy) |
| `--top-p F` | 1.0 | Top-p nucleus sampling |
| `--draft-model <path>` | (none) | Recognised but currently ignored — speculative decoding in chat is not wired up yet; use the one-shot path |
| `--spec-depth N` | 5 | Speculation depth (ignored with `--draft-model` in chat) |

#### Slash commands (inside the chat REPL)

| Command          | Description |
|------------------|-------------|
| `/help`          | Show the command list |
| `/quit`, `/exit` | Exit the chat |
| `/clear`         | Clear the conversation history |
| `/stats`         | Show session statistics (turns, avg TTFT, avg decode tok/s) |
| `/system <text>` | Change the system prompt (also clears history) |

#### Keyboard shortcuts

| Key    | During generation                    | At the `>` prompt |
|--------|--------------------------------------|-------------------|
| Ctrl+C | Interrupts generation, REPL stays open | Ignored — use `/quit` |
| Ctrl+D | —                                    | Exits the chat (EOF on stdin) |

#### Example session

```
$ rocmforge chat --model ~/models/Qwen2.5-7B-Instruct-Q4_0.gguf

  ROCmForge v0.1.0
  ─────────────────────────────────────
  GPU:       AMD Radeon RX 9070 XT (gfx1201 (RDNA 4))
  ROCm:      unknown
  Model:     Qwen2.5 7B Instruct (Q4_0, 4.1 GB)
  Layers:    28   Hidden: 3584   Vocab: 152064
  WMMA:      active (GEMM + Attention)
  VRAM:      8.9 / 15.9 GB used
  ─────────────────────────────────────
  Type /help for commands, /quit to exit.

  > Hello! How are you?
  I'm doing well, thank you! How can I assist you today?
  [TTFT: 49 ms | 102.0 tok/s | 14 tokens | stop]

  > What is the capital of France?
  The capital of France is Paris.
  [TTFT: 62 ms | 101.8 tok/s | 8 tokens | stop]

  > /stats

  Session statistics
  ─────────────────────────────────
  Turns:          2
  Avg TTFT:       56 ms
  Avg decode:     101.9 tok/s

  > /quit
  Goodbye.
```

The `ROCm:` line shows "unknown" when the ROCm version string cannot
be probed from the HIP runtime — a cosmetic gap, not a failure.

#### Multi-turn cost

Every turn re-prefills the entire conversation history: there is no
KV-cache reuse across turns yet. TTFT grows roughly linearly with the
total number of tokens exchanged so far. For long conversations,
`/clear` after a logical break is the current workaround.

### One-shot inference

Without a subcommand, ROCmForge runs a single inference and exits.
Useful for scripting, benchmarks, and `bash | grep` pipelines.

```bash
rocmforge --model <path> --prompt "Your prompt" --max-tokens 128 --gpu
```

#### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--model <path>` / `-m` | (required) | Path to GGUF model file |
| `--prompt <text>` / `-p` | (required) | Input prompt |
| `--max-tokens N` | 256 | Maximum tokens to generate |
| `--temperature F` | 1.0 | Sampling temperature |
| `--top-p F` | 0.9 | Top-p nucleus sampling |
| `--gpu` | off | Use GPU backend (requires ROCm/HIP; CPU path runs without this) |
| `--no-template` | off | Raw completion — do not wrap the prompt in the Qwen2.5 ChatML template |
| `--list-tensors` | off | List all tensors in the model file and exit (debug) |
| `--debug` | off | Print top logits and sampler diagnostics |
| `--draft-model <path>` | (none) | GPU-only: enable speculative decoding with a draft model |
| `--spec-depth N` | 5 | Speculation depth (only used with `--draft-model`) |

For deterministic greedy output (useful for benchmarks):

```bash
rocmforge --model <path> --prompt "..." --max-tokens 128 --gpu \
    --temperature 0.0 --top-p 1.0 --no-template
```

#### Speculative decoding

```bash
rocmforge \
    --model ~/models/Qwen2.5-7B-Instruct-Q4_0.gguf \
    --draft-model ~/models/qwen2.5-0.5b-instruct-q4_0.gguf \
    --spec-depth 1 \
    --prompt "Hello" --max-tokens 128 --gpu
```

Honest note: on the 15-prompt benchmark this is a loss at every prompt
vs. plain greedy (median 72 tok/s vs. 102 tok/s baseline). The per-step
verify cost outweighs the amortisation at observed acceptance rates
(~54 % median, break-even ~80 %). It may still help on very repetitive
or templated outputs.

## Environment variables

ROCmForge reads optional environment flags at startup (cached on first
access via `CachedEnvFlag`, so toggling during a single run has no
effect).

### Feature toggles — performance tuning

All default OFF unless noted. Setting any of these to `1` disables the
named optimisation path.

| Variable                             | Effect when set to `1` |
|--------------------------------------|------------------------|
| `ROCMFORGE_DISABLE_WMMA_PREFILL`     | Do not take the WMMA-Q4_0 prefill path; fall through to hipBLAS / scalar |
| `ROCMFORGE_DISABLE_WMMA_ATTENTION`   | Do not take the WMMA FlashAttention path; use the scalar per-head kernel |
| `ROCMFORGE_DISABLE_HIPBLAS_PREFILL`  | Do not take the hipBLAS-Hgemm prefill path; fall through to custom-GEMV |
| `ROCMFORGE_DISABLE_TILED_GEMV`       | Disable tiled batched GEMV for large FFN projections |
| `ROCMFORGE_DISABLE_BATCHED_LM_HEAD`  | Disable batched verify lm_head; fall back to sequential per-position dispatch |
| `ROCMFORGE_GPU_SAFE_MODE`            | Disable every experimental fast path at once (includes the three above plus decode-graph etc.) |

### Debugging and profiling

| Variable                          | Values                 | Effect |
|-----------------------------------|------------------------|--------|
| `RUST_LOG`                        | `warn` (default), `info`, `debug`, `trace`, or per-module selectors | Logging verbosity via the `tracing` crate |
| `ROCMFORGE_PROFILE_PREFILL_OPS`   | `1`                    | Emit one `tracing::info!` event per prefill layer with per-operation microsecond timings. Adds ~30 ms of `hipDeviceSynchronize` overhead — do NOT leave on in normal runs |
| `ROCMFORGE_PROFILE_SPEC_STEP`     | `1`                    | HIP-event timing for speculative-decode step breakdown |
| `ROCMFORGE_PROFILE_VERIFY_BREAKDOWN` | `1`                 | Sub-phase timing within verify layers (requires `ROCMFORGE_PROFILE_SPEC_STEP`) |
| `ROCMFORGE_SPEC_DEBUG`            | `1`                    | Print draft/target token comparison per spec-decode step |

#### Logging examples

See which GEMM dispatch path each prefill call is taking:

```bash
RUST_LOG=rocmforge::gpu::ops=debug rocmforge chat --model <path>
```

Typical output (one line per GEMM dispatch):

```
DEBUG rocmforge::gpu::ops: GEMM dispatch: WMMA Q4_0 (257→320) path="wmma_q4_0" ...
DEBUG rocmforge::gpu::ops: GEMM dispatch: fused Gate+Up WMMA Q4_0 path="wmma_q4_0_fused_gate_up" ...
```

Per-layer timing during prefill (verbose):

```bash
RUST_LOG=trace rocmforge chat --model <path>
```

Separate log file for long sessions:

```bash
RUST_LOG=debug rocmforge chat --model <path> 2>/tmp/rocmforge.log
# In another terminal:
tail -f /tmp/rocmforge.log
```

## ROCm upgrade validation

ROCmForge ships a two-script validation harness for safe ROCm version
upgrades. The baseline captures build fingerprint, synthetic prefill,
per-layer timing, and a 15-prompt real-world benchmark with answer
dumps. The diff tool compares two baselines and emits a
BUILD / CORRECTNESS / PERFORMANCE / VERDICT report.

### Capture a baseline (before upgrading ROCm)

```bash
fish benches/rocm_validate.fish
```

Output lands in `benches/results/rocm_baseline/rocm_<ver>_<ts>/`. The
directory contains `fingerprint.json`, `synthetic_bench.json`,
`layer_timing.json`, `prompt_bench.json`, `summary.json`, raw logs,
and the 15 greedy completions.

### Capture a new baseline (after upgrading ROCm)

```bash
cargo clean && cargo build --release --features gpu   # force clean rebuild
fish benches/rocm_validate.fish
```

### Diff the two baselines

```bash
fish benches/rocm_diff.fish \
    benches/results/rocm_baseline/rocm_<old>_<ts> \
    benches/results/rocm_baseline/rocm_<new>_<ts>
```

Emits stdout + a Markdown file under
`benches/results/rocm_baseline/diff_<old>_vs_<new>.md`. The
Performance section uses `±5 % ✅ / ±10 % ⚠️ / >10 % ❌` tolerance
bands — expect ❌ on every row when the diff spans a code change, not
just a ROCm upgrade (that's a feature, not a false alarm).

Reference baselines in this repo:

- `rocm_7.2.1_1776506441/` — ROCm 7.2.1, Phase 3 code
- `rocm_7.2.2_1776508380/` — ROCm 7.2.2, Phase 3 code (same binary)
- `rocm_7.2.2_1776519824/` — ROCm 7.2.2, Phase 4 code
  (**Post-Phase-4 reference** — use this for future ROCm upgrade diffs)

The last one is marked in its `summary.json::baseline_note` field.

## Exit codes

| Code | Meaning |
|-----:|---------|
|    0 | Normal exit (generation finished, `/quit`, EOF, or ctrl+D) |
|    1 | Argument parse error, model load failure, or fatal runtime error |

Ctrl+C during chat generation returns control to the prompt and does
not exit the process. Ctrl+C at the `>` prompt is ignored — use
`/quit` or Ctrl+D to exit.
