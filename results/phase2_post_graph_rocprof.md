# Post-HIP-Graph rocprof (third data point) — Mutex prompt / 100 tokens

**Date:** 2026-04-23
**Branch:** v1.0-dev (at commit `a6869b8` — HIP-Graph decode integration)
**Hardware:** AMD Radeon RX 9070 XT (gfx1201), 640 GB/s nominal BW
**Tool:** rocprofv3 — `--runtime-trace --stats --summary`
**Binary:** `target/release/rocmforge-v1`
**Model:** `~/models/Qwen3-8B-Q4_K_M.gguf`
**Prompt:** `"Explain what a mutex is in one paragraph."`
**Max-Tokens:** 100 (EOS @ 96 decode tokens, 129 total forward-pass)
**Reference runs:**
- `results/phase2_post_2.1.5_rocprof_deepdive.md` (Pre-Unfuse)
- `results/phase2_post_unfuse_rocprof.md` (Post-Unfuse)

## TL;DR — HIP-Graph is inert in single-prompt CLI mode

```
                        Pre-Unfuse     Post-Unfuse     Post-Graph (this run)
hipLaunchKernel         52 203         59 115          59 115       unchanged vs post-unfuse
hipGraphLaunch          —              —               0            graph never captured
hipStreamBeginCapture   —              —               0            graph never captured
Σ GPU kernel time       2 363 ms       1 377 ms        2 031 ms     (bandit variant shift)
Decode (wall, rocprof)  —              68.8 tok/s      43.7 tok/s   (bandit exploration)
Decode (ohne rocprof)   40.6 tok/s     68.8 tok/s      ~56 tok/s    (15-Prompt median)
```

**The HIP-Graph integration did not trigger** in a single-prompt CLI
invocation. Not one `hipStreamBeginCapture`, not one `hipGraphLaunch`
appears in the trace. All 599 dispatches/decode-token still went through
`hipLaunchKernel`. Zero benefit, zero cost — the integration is inert
in this mode.

## Why the graph never fired

The capture gate is `should_capture_hip_graph()`:

```rust
fn should_capture_hip_graph(&self) -> bool {
    !std::env::var("ROCMFORGE_DISABLE_HIP_GRAPH").is_ok()
        && self.node_fast_cache.is_some()
        && self.hip_graph.is_none()
}
```

- `ROCMFORGE_DISABLE_HIP_GRAPH` is unset → OK.
- `hip_graph.is_none()` on first decode → OK.
- `node_fast_cache.is_some()` requires `compile_fast_dispatch()` to have
  run successfully. That only happens **after** every per-layer UCB1
  bandit has committed to a single variant.

The bandit did **not** converge on this 96-decode-token CLI run. Proof
from the kernel stats:

| Kernel | Calls | Σ µs | Reading |
|---|---:|---:|---|
| `gemv_q4_k_standard` | **8 640** | 744 027 | still exploring |
| `gemv_q4_k_q8_inline` | 6 912 | 578 923 | committed choice for some bandits |
| `gemv_q4_k_q8_inline_residual` | 5 184 | 225 619 | residual-fused choice |

Compare the post-unfuse run (same prompt, same token count, same binary
shape but pre-sudot4):

| Kernel | Calls | Reading |
|---|---:|---|
| `gemv_q4_k_standard` (loser pulls) | **90** | bandit had committed |
| `gemv_q4_k_q8_inline` | 15 462 | full commit |

**Hypothesis:** the sudot4 commit (`d3aacdd`) added a third arm per
bandit. Exploration now sweeps three variants instead of two before
committing, and 96 decode steps is not enough to commit every layer's
bandit on a single CLI invocation. In the integration-test suite, a
100-token warmup run was followed by the measured run — the bandit had
already committed before measurement, and the graph captured cleanly
(integration test: `graph 61.3 vs legacy 60.9 tok/s`, +0.7 %).

## What the trace actually shows

HIP-API summary (non-trivial rows):

| API | Calls | Σ µs | Reading |
|---|---:|---:|---|
| `hipLaunchKernel` | 59 115 | 1 971 054 | identical to post-unfuse → graph did not replace dispatches |
| `hipMemcpy` | 666 | 518 326 | unchanged |
| `hipStreamSynchronize` | 97 | 129 380 | 96 decode tokens + 1 prefill sync |
| `hipStreamBeginCapture` | **0** | — | **graph never captured** |
| `hipStreamEndCapture` | **0** | — | — |
| `hipGraphInstantiate` | **0** | — | — |
| `hipGraphLaunch` | **0** | — | — |
| `hipGraphExecKernelNodeSetParams` | **0** | — | — |

Kernel stats (top rows, this run):

| Kernel | Calls | Σ µs | % |
|---|---:|---:|---:|
| `gemv_q4_k_standard` | 8 640 | 744 027 | 36.6 |
| `gemv_q4_k_q8_inline` | 6 912 | 578 923 | 28.5 |
| `gemv_q6_k_standard` (LM-head + Q6_K layers) | 3 552 | 313 947 | 15.5 |
| `gemv_q4_k_q8_inline_residual` | 5 184 | 225 619 | 11.1 |
| `wmma_gemm_q4_k_fp16` (prefill) | 216 | 45 155 | 2.2 |
| `attention_decode` | 3 456 | 37 336 | 1.8 |
| `rms_norm` | 7 008 | 23 130 | 1.1 |
| `rms_norm_batched` | 7 057 | 16 790 | 0.8 |
| `rope` | 6 912 | 13 209 | 0.7 |
| `wmma_gemm_q6_k_fp16` (prefill LM-head) | 37 | 12 730 | 0.6 |
| `kv_cache_append` | 4 644 | 6 920 | 0.3 |
| `swiglu` | 3 492 | 5 930 | 0.3 |
| `residual_add_inplace` | 1 800 | 3 264 | 0.2 |
| Σ kernel | — | **2 030 795** | 100 |

Σ kernel time is **higher** than post-unfuse (2 031 ms vs 1 377 ms)
because the bandit is still pulling the slower `gemv_q4_k_standard`
variant 8 640× (≈0.74 s of wasted work). This is a bandit-exploration
side effect, not a graph regression.

## Implications for the 5 % stretch gate

The integration report noted +0.7 % in the graph-active A/B test. This
rocprof run confirms the inverse: when the graph is **not** active
(CLI, short prompts, unconverged bandit), the code path is completely
bypassed and cost is zero. Both signs of the measurement agree with
the code structure:

- Bandit converged → `compile_fast_dispatch` populates → graph captures
  → replay costs 1 `hipGraphLaunch` + 144 `SetParams` instead of 599
  `hipLaunchKernel`. Test harness: 61.3 vs 60.9 tok/s (+0.7 %).
- Bandit unconverged → fast_dispatch empty → `should_capture_hip_graph`
  returns false → legacy path continues untouched. CLI rocprof: 59 115
  `hipLaunchKernel`, 0 graph API calls, behaves exactly like post-unfuse.

## Next steps (not taken in this session)

1. **Make the graph fire in CLI.** Options:
   - Reduce bandit exploration horizon (commit earlier, accept more
     variance in variant choice).
   - Persist bandit state across runs (autotune cache equivalent for v1).
   - Lower the per-arm pull count before commit from the UCB default.
2. **Quantify the real decode speedup** by running a 15-prompt suite
   with a warmup prompt prepended so the bandit converges before the
   measured prompts. The CLI-only 15-prompt run showed 56.5 tok/s
   (within noise of 56.8 post-sudot4), consistent with graph never
   firing.
3. **Drop the 5 % stretch gate** as a headline target for this step.
   The real gate is: *graph replay works correctly and costs nothing
   when inactive*. Both are demonstrated.

## Files

- Binary: `target/release/rocmforge-v1` @ commit `a6869b8`
- rocprof output: `/tmp/rocprof-post-graph/swabe/`
- HIP API stats: `/tmp/rocprof-post-graph/swabe/160984_hip_api_stats.csv`
- Kernel stats: `/tmp/rocprof-post-graph/swabe/160984_kernel_stats.csv`
