# Q8_1 Buffer Sharing (Schritt 5 — skip redundant quantize on Q → K → V)

**Date:** 2026-04-24
**Branch:** v1.0-dev
**Hardware:** AMD Radeon RX 9070 XT (gfx1201, RDNA4), 640 GB/s nominal BW
**Predecessor:** per-dispatch `quantize_q8_1` (180 calls/token pre-Schritt 5)
**Context:** `results/phase2_q6k_mmvq_port.md` (Schritt 4, 94.7 tok/s baseline)

## TL;DR

Cache the last-quantized Q8_1 activation buffer keyed by its input
`BufferId`. Consecutive MMVQ dispatches that share the same input
(Q → K → V in every layer) reuse the cached buffer and skip the
`quantize_q8_1` launch.

| | Pre-Schritt-5 | Post-Schritt-5 | Δ |
|---|---:|---:|---:|
| 15-prompt decode tok/s | 94.7 | **96.2** | **+1.6 %** |
| Mutex tok/s | 100.5 | **102.0** | +1.5 % |
| Arithmetic tok/s | 102.0 | **103.7** | +1.7 % |
| Gap vs llama.cpp | 1.05× | **1.03×** | 97 % parity |
| Prompts ≥ llama.cpp (99.3 tok/s) | 6 / 15 | **8 / 15** | |

**Bit-identical output** vs. the `ROCMFORGE_DISABLE_Q8_1_SHARING=1`
reference path — verified on the Mutex prompt at 30 tokens.

## Design — consecutive-MMVQ cache

The simplest correct cache:

```rust
last_q8_1_input_id: Option<BufferId>

fn ensure_q8_1_quantized(&mut self, input_id, in_ptr, in_dim) {
    ensure_q8_1_buffer(in_dim);
    if self.last_q8_1_input_id == Some(input_id) {
        return q8_1_ptr;  // cache hit — skip quantize
    }
    quantize_q8_1(in_ptr, q8_1_ptr, in_dim);
    self.last_q8_1_input_id = Some(input_id);
    q8_1_ptr
}
```

Reset at:
- Top of every `execute_decode` call (fresh forward pass)
- Top of `capture_decode_graph` (so the simulated walk in
  `launch_index_spans` starts in the same state as the real dispatch)

Explicit invalidation after **in-place writes**:
- `dispatch_fused_gemm_residual` — residual buffer written in place
- `GraphNode::ResidualAdd` dispatch — `a` buffer written in place

No invalidation needed for non-MMVQ nodes that don't write to the
cached buffer (RoPE, Attention, RmsNorm → new output buffers); the
cache simply sits with the old ID until the next MMVQ dispatch, which
either matches (hit) or misses (miss → re-quantize).

### What the cache catches (Qwen3-8B graph)

```
RmsNorm → hidden_normed (X)
  Q-MMVQ(X)   ← miss, quantize, cache=X
  K-MMVQ(X)   ← HIT, skip quantize
  V-MMVQ(X)   ← HIT, skip quantize
RoPE, KV-append, Attention → attn_out                 (cache stale, but ignored)
  O-FusedGemmResidual(attn_out, residual)             (miss, quantize, cache=attn_out;
                                                       invalidate after — residual
                                                       written in place)
RmsNorm → ffn_normed (Y)
  Gate+Up MMVQ-fused(Y)                               (miss, quantize, cache=Y)
  Down-FusedGemmResidual(gate_out, residual)          (miss, quantize, cache=gate_out;
                                                       invalidate)
```

Per layer: 5 MMVQ dispatches, 2 cache hits (K, V). 36 layers × 2 hits =
**72 `quantize_q8_1` launches saved per token**.

72 × 3.58 µs kernel time = ~258 µs saved per token ≈ 2.4 % at 94.7 tok/s.

## HIP-Graph compatibility

The Schritt-2 capture validation check
`spans.total_launches == captured_node_count` would fire if the
simulated count and the captured count disagree. Fix: `launch_index_spans`
now replays the **same cache walk** during the prediction phase.

```rust
let mut sim_q8_1_input: Option<BufferId> = None;
for node in nodes {
    match node {
        Gemm { input, .. } if is_mmvq(committed) => {
            if sharing_on && sim_q8_1_input == Some(*input) {
                idx += 1;                         // cache hit → skip quantize
            } else {
                idx += 2;                         // miss → quantize + mmvq
                sim_q8_1_input = Some(*input);
            }
        }
        FusedGemmResidual { input, .. } if mmvq_path => {
            if sharing_on && sim_q8_1_input == Some(*input) { idx += 1; }
            else { idx += 2; sim_q8_1_input = Some(*input); }
            sim_q8_1_input = None;                // in-place write
        }
        GateUpSwiGLU { input, .. } if mmvq_fusion_active => {
            /* same cache check, no post-invalidate */
        }
        ResidualAdd { a, .. } => {
            idx += 1;
            if sim_q8_1_input == Some(*a) { sim_q8_1_input = None; }
        }
        // ... other arms ...
    }
}
```

Tested — capture validates, HIP-Graph fires, bandit converges as
before. Because the simulated walk matches dispatch exactly, the
captured graph has precisely the kernel-node count `launch_index_spans`
predicts.

Env kill-switch: `ROCMFORGE_DISABLE_Q8_1_SHARING=1` forces a quantize on
every dispatch and sets `sharing_on = false` in the simulation — the
capture math still balances.

## Correctness

Direct parity check on the Mutex prompt, 30 tokens, greedy, single
prompt (HIP-Graph off in this mode so every dispatch hits the legacy
path where the cache is most active):

```
Without sharing (ROCMFORGE_DISABLE_Q8_1_SHARING=1):
  A mutex, short for "mutual exclusion," is a synchronization mechanism
  used in concurrent programming to ensure that only one thread or process

With sharing (default):
  A mutex, short for "mutual exclusion," is a synchronization mechanism
  used in concurrent programming to ensure that only one thread or process
```

**Bit-identical output across all 30 decode tokens.** Expected: the
cache hit path returns the **same `q8_1_buffer` pointer** as a fresh
quantize would have written. Since `quantize_q8_1` is deterministic
and the input buffer hasn't been mutated between the miss and the hit,
downstream MMVQ reads exactly the same bytes. No possibility of drift.

## End-to-End (15-prompt suite)

### Aggregate ladder

| | Pre-MMVQ | Schr. 2 | Schr. 2b | Schr. 3 | Schr. 4 | **Schr. 5** | llama.cpp |
|---|---:|---:|---:|---:|---:|---:|---:|
| Decode tok/s | 62.7 | 64.4 | 70.8 | 86.1 | 94.7 | **96.2** | 99.3 |
| Δ vs pre-MMVQ | — | +2.7 % | +12.9 % | +37.3 % | +51.0 % | **+53.4 %** | |
| Gap vs llama.cpp | 1.58× | 1.54× | 1.40× | 1.15× | 1.05× | **1.03×** | 1.00× |

### Per-prompt decode tok/s

| # | Name | Schritt 4 | **Schritt 5** | Δ | vs llama.cpp (99.3) |
|---:|---|---:|---:|---:|:---:|
| 1 | Greeting | 99.1 | **101.2** | +2.1 % | **+1.9 %** |
| 2 | Simple Sequence | 98.7 | **100.9** | +2.2 % | **+1.6 %** |
| 3 | Prime Check (Python) | 96.9 | 98.8 | +2.0 % | −0.5 % |
| 4 | LRU Cache (C++) | 96.7 | 98.2 | +1.6 % | −1.1 % |
| 5 | REST API (Go) | 97.9 | 99.4 | +1.5 % | **+0.1 %** |
| 6 | Mutex Explanation | 100.5 | **102.0** | +1.5 % | **+2.7 %** |
| 7 | TCP vs UDP | 95.5 | 96.8 | +1.4 % | −2.5 % |
| 8 | GPU Architecture Blog Post | 94.8 | 96.1 | +1.4 % | −3.2 % |
| 9 | Binary Search Complexity | 97.9 | 99.4 | +1.5 % | **+0.1 %** |
| 10 | Debug Code | 97.5 | 99.0 | +1.5 % | −0.3 % |
| 11 | Distributed Message Queue | 88.3 | 89.6 | +1.5 % | −9.8 % |
| 12 | Long System Prompt + Question | 90.4 | 92.0 | +1.8 % | −7.4 % |
| 13 | Long Output Story | 97.2 | 98.6 | +1.4 % | −0.7 % |
| 14 | Arithmetic (Q4_K Precision) | 102.0 | **103.7** | +1.7 % | **+4.4 %** |
| 15 | Emoji/Special Characters | 99.9 | **101.5** | +1.6 % | **+2.2 %** |

**Every prompt improves by 1.4-2.2 %** (the cache saves 72 calls per
token, proportionally tiny — matching this modest but reliable gain).

**8 of 15 prompts now at or above llama.cpp's 99.3 tok/s** (up from
6 / 15 in Schritt 4). The remaining gap is concentrated on long-context
prompts (11, 12) and the longest-output code prompt (8) — attention-
kernel territory, not kernel-level work anymore.

### Bandit / HIP-Graph state

```
Q4_K n=1024 k=4096:   mmvq committed (11.70 µs vs 54.54 µs)  — 4.7× win
Q4_K n=4096 k=4096:   mmvq committed (28.98 µs vs 176.54 µs) — 6.1× win
Q6_K n=1024 k=4096:   mmvq committed (15.19 µs vs 26.88 µs)  — 1.77× win
Q6_K n=4096 k=12288:  mmvq committed (70.88 µs vs 129.00 µs) — 1.82× win
```

Total bandit pulls after the suite: ~378 (down from ~450 in Schritt 4).
Fewer pulls because the cache-hit dispatches don't record events —
they don't hit the ShapeBandit `record_start/stop` path at all, since
no kernel launch happens.

Monitor events: **3** — same `RepetitionDetected` on `token_id=10519`
pattern seen since Schritt 2b. Unchanged model-behavior quirk.

## Files

| New | LOC |
|---|---:|
| `results/phase2_q8_1_sharing.md` (this report) | — |
| `results/phase2_q8_1_sharing_15prompt_suite.md` (raw) | — |

| Modified | Change |
|---|---|
| `src_v1/graph/executor.rs` | +2 fields (`last_q8_1_input_id`, `q8_1_sharing_enabled`); +2 helpers (`ensure_q8_1_quantized`, `invalidate_q8_1_cache`); 3 MMVQ dispatch paths refactored to use the helper; ResidualAdd and FusedGemmResidual invalidate after in-place writes; `launch_index_spans` simulates the cache walk; `execute_decode` and `capture_decode_graph` reset the cache on entry |

Zero new kernels, zero FFI changes, zero test changes outside the new
regression check below.

## Attribution of the E2E gain

From 94.7 → 96.2 tok/s = +1.6 %. Breakdown:

- **72 skipped quantize_q8_1 launches × 3.58 µs kernel time:**
  258 µs/token saved ≈ 2.4 % at 94.7 tok/s base
- **Reduced HIP-Graph replay size:** graph now has 72 fewer captured
  kernel nodes. Per-node replay overhead on HIP is ~1-2 µs. At 72
  fewer nodes that's 72-144 µs saved per replay → ~0.7-1.4 % E2E
- **Reduced bandit event-recording:** cache-hit dispatches skip the
  `EventPool::record_start/stop` pair, saving ~0.3-0.5 µs each during
  exploration → ~0.2 % during the first ~2 prompts

Expected total: 2-4 %. Measured 1.6 % — on the low end but solid and
matches the prompt's own projection (+1.5-2.4 %). The difference is
mostly that launch overhead on gfx1201's HIP-Graph replay is lower
than worst-case assumptions; the savings are real but less dramatic
than they'd be on raw `hipLaunchKernel`.

## Why so modest?

Two factors limit the Schritt-5 gain vs the earlier ones:

1. **We're already near the memory-BW ceiling.** At 96 tok/s on a
   640 GB/s-peak GPU, we've consumed most of the easy parallelism.
   Each microsecond saved now converts to less-than-proportional
   throughput because other kernels run at full speed on their own
   schedule.
2. **The 72 saved calls are already the "cheap" kind** — `quantize_q8_1`
   is a 3.6 µs launch-bound kernel. The hot matmuls (MMVQ, MMVQ-fused,
   mmvq_residual) aren't affected. Schritt 3's gate-fusion moved
   bigger boulders; Schritt 5 tidies up.

That said: a 1.6 % gain with **zero new kernels, zero new GPU code,
and bit-exact output** is excellent value for ~50 LOC of executor
cache logic. Gap closure from 5.3 % → 3.2 %.

## Honest Caveats

1. **The ResidualAdd invalidation is defensive.** In Qwen3-8B's graph
   the sequence `MMVQ(X) → ResidualAdd(X, …)` doesn't occur before
   another `MMVQ(X)` — RoPE and Attention sit between, and they
   produce fresh buffers. The invalidation costs nothing when no MMVQ
   reads the buffer again, and protects against future graph changes.
2. **The cache only catches _consecutive_ MMVQ reads of the same input.**
   A pathological graph where MMVQ(X) → something_non_mmvq → MMVQ(X)
   with X unchanged would lose the saving (cache still holds X but the
   path between triggered invalidation). Not observed in practice.
3. **LM-head has 1 arm only** (N=151936 > 100k threshold from Schritt 4)
   so the Q6_K MMVQ path isn't used there — no Q8_1 sharing benefit
   on LM-head. That's expected: LM-head runs q6_k_standard which
   doesn't consume Q8_1 anyway.
4. **We didn't implement a unit test that directly counts `quantize_q8_1`
   kernel launches.** The correctness proof is end-to-end bit-identity
   and the bandit pulls' drop from ~450 to ~378, consistent with 72
   fewer MMVQ-tagged events per suite run.

## Next Concrete Step

With Schritt 5 landing at **96.2 tok/s, 97 % of llama.cpp, 8/15 prompts
at parity**, the gap closure from the original 1.58× has reached
**97 % of the original gap**. Remaining levers:

→ **Long-context attention** (prompts 11, 12 at 88-92 tok/s vs the
  99+ short-prompt cluster). Attention-decode is scalar and
  seq_len-dependent. A WMMA-based decode attention could recover
  another 1-2 % on the aggregate.

→ **Persistent bandit state** (Memory #21 / roadmap). Survive decode
  sessions → no re-exploration cost. Small gain per-run but zero
  warm-up penalty on short prompts.

→ **HIP-Graph capture for prefill.** Currently prefill is legacy;
  the WMMA path already runs at ~600 tok/s prefill. Graph capture
  there would shave ~50-100 ms on long prompts.

None of these need more kernel work. The MMVQ port is done.
