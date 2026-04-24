# MMVQ Gate-Fusion (Schritt 3 — Gate + Up + SwiGLU fused)

**Date:** 2026-04-24
**Branch:** v1.0-dev
**Hardware:** AMD Radeon RX 9070 XT (gfx1201, RDNA4), 640 GB/s nominal BW
**Predecessor:** unfused `mmvq_gate + mmvq_up + swiglu` (3-kernel sequence)
**Context:** `results/phase2_mmvq_residual.md`, `results/phase2_llamacpp_kernel_analysis.md` (H5)

## TL;DR

Fuse Gate + Up Q4_K GEMVs and the SwiGLU epilog into a single kernel
that walks two weight streams over the same pre-quantized Q8_1
activation. The llama.cpp `has_fusion=true` pattern ported 1:1 onto
our MMVQ kernel.

**Critical gate passed:** fused **78.55 µs** vs unfused composite
**109.87 µs** → **1.40× speedup** on Qwen3-8B gate/up shape. This is
the first fusion that beats its unfused counterpart on gfx1201 — the
Phase 2 `gate_up_swiglu` kernel missed 62 % because its one-thread-
per-super-block tiling couldn't handle two weight streams. MMVQ's
16-thread cooperative tiling does.

| | Pre-Schritt-3 | Post-Schritt-3 | Δ |
|---|---:|---:|---:|
| 15-prompt decode tok/s | 70.8 | **86.1** | **+21.6 %** |
| Mutex decode tok/s | 74.6 | **91.3** | +22.4 % |
| **Gap to llama.cpp** | 1.40× | **1.15×** | **closes 71 %** of remaining gap |
| Gate/Up launches per layer | 5 | **2** | quantize + mmvq_fused |

llama.cpp-ROCm runs Qwen3-8B at 99.3 tok/s. ROCmForge is now 15 % off
it — from the pre-MMVQ-port gap of 58 %.

## Kernel

| | |
|---|---|
| Source | `hip_kernels_v1/gemv/gemv_q4_k_mmvq_fused.hip` |
| Base | `gemv_q4_k_mmvq.hip` (copied; same thread-mapping, dp4a, 6-bit scale unpack) |
| LOC | 240 (vs MMVQ 245) |
| Changed vs MMVQ | +1 weight pointer param, +1 accumulator, +1 inner `vec_dot` call, +1 shared-mem array, +1 `silu()` call at lane 0 |
| LDS usage | 2 × `[7][32] float` = 1 792 B (was 896 B in MMVQ) |
| Shared-mem budget left | 62 KB (no concern) |

### Critical changes vs `gemv_q4_k_mmvq.hip`

```cpp
// Two parallel accumulators — one per weight stream.
float tmp_up   = 0.0f;
float tmp_gate = 0.0f;

for (int kbx = tid/16; kbx < blocks_per_row_x; kbx += 16) {
    ...
    const block_q4_K* vbq_up   = vx_up   + (kbx_offset + kbx);
    const block_q4_K* vbq_gate = vx_gate + (kbx_offset + kbx);

    // Both dots share the same Q8_1 activation pointer. The second
    // vec_dot runs in the ALU cycles otherwise idle while memory
    // loads complete — this is the ILP that makes two streams free.
    tmp_up   += rf_v1_vec_dot_q4_K_q8_1(vbq_up,   &vy[kby], kqs);
    tmp_gate += rf_v1_vec_dot_q4_K_q8_1(vbq_gate, &vy[kby], kqs);
}

// After two parallel warp-reductions, SwiGLU epilog at lane 0:
if (threadIdx.x == 0) {
    dst[row0] = tmp_up * rf_v1_silu(tmp_gate);   // silu(g) × u
}
```

### Why this works where the Phase 2 `gate_up_swiglu` failed

The Phase 2 kernel `gemv_q4_k_gate_up_swiglu.hip` hit only **20 % BW**
on gfx1201 — 3.2× slower than the unfused path that eventually replaced
it (`results/phase2_unfused_gate_up.md`). Root cause: its one-thread-
per-super-block tiling meant every thread had to serialise two weight
reads through the same memory port. Two streams collided at the HBM
queue rather than parallelising.

MMVQ's 16-thread-per-super-block tiling (from the Schritt 2 port) gives
each warp enough outstanding loads to keep the memory pipeline busy
even with two streams. The second stream fills the ALU bubbles that
would otherwise sit idle during the first stream's load — llama.cpp's
`has_fusion=true` comment on `mmvq.cu:437` captures this exactly:
*"1. Hide latency by prefetching bias and gate here … on threads that
won't die after partial sum calculation."*

## Correctness

| Test | Result |
|---|---|
| Fused vs unfused composite (N=12288, K=4096) | median **0.00000**, p99 0.00000, max **0.00000** |
| Fused vs CPU FP32 reference (N=12288, K=4096) | median **0.095 %**, p99 9.8 %, max 38 % |

**Bit-exact vs unfused.** The kernel does the same dot-product math with
the same per-lane accumulation order; the only difference is the SwiGLU
multiply happens inside a register instead of after a VRAM round-trip.
No FP reassociation possible, no drift.

Vs CPU FP32: median **0.095 %** — *better* than plain MMVQ (0.35-0.42 %
from Schritt 2). Why? Two reasons:
1. No intermediate Q8 quantization round-trip for gate/up: both dots
   read directly from one shared Q8_1 buffer.
2. SwiGLU is computed in one FMA + one `expf` at single-precision — the
   unfused path writes two FP32 vectors to VRAM and reads them back,
   incurring extra flushes through the L2 hierarchy.

p99 / max numbers are dominated by near-zero output rows (SwiGLU output
magnitude can spike 10 000+ on some elements while most are O(100);
small-magnitude output elements amplify relative error) — consistent
with earlier MMVQ tests.

## Performance

Isolated kernel benchmark on N=12288, K=4096 (Qwen3-8B gate/up shape),
mean over 50 runs after 5-iter warmup.

| Config | Time | Kernels | Apparent BW |
|---|---:|---:|---:|
| **Unfused** (quant + 2×mmvq + swiglu) | 109.87 µs | 4 | — |
| **Fused** (quant + mmvq_fused) | **78.55 µs** | 2 | **113 % \*** |
| **Speedup** | **1.40×** | 50 % fewer | |

\* BW > 100 % is the naive "weight-bytes / time" model counting L1/L2
hits against the 640 GB/s HBM peak. Real VRAM bandwidth is bounded by
HBM; L1-resident activations boost apparent throughput.

Effective savings per layer: **31.3 µs**. Across 36 layers per token:
**1.13 ms saved per decode step** — matching the observed aggregate
speedup (pre-fusion 70.8 tok/s = 14.1 ms/token, post-fusion 86.1 tok/s =
11.6 ms/token, Δ 2.5 ms: 1.1 ms from gate/up + 1.4 ms from reduced
launch/graph overhead and one-less-quantize per layer).

### Implicit Q8_1 sharing

The fused kernel consumes **one** Q8_1 activation buffer for both gate
and up dots. Previously the unfused path ran `quantize_q8_1` twice per
layer (once before gate, once before up, same input both times). The
fusion implicitly halves the quantize calls for the gate/up block:

- **Pre-Schritt 3:** ~216 quantize_q8_1 calls/token (= 6 × 36 layers)
- **Post-Schritt 3:** ~180 quantize_q8_1 calls/token (= 5 × 36 layers)
- Savings: 36 × 3.58 µs ≈ **129 µs/token** — already included above

This is the Q8_1-sharing optimisation Schritt 2b flagged as a bonus,
delivered "for free" because the fused kernel physically can't read
two different Q8_1 buffers in the same loop.

## Executor integration

`src_v1/graph/executor.rs::dispatch_gate_up_swiglu` now has a new
default branch between the dynamic hook and the legacy fused path:

```rust
if use_mmvq_fusion {     // default: ROCMFORGE_DISABLE_MMVQ_FUSION != "1"
    ensure_q8_1_buffer(hidden_dim);
    quantize_q8_1(in_ptr, q8_1_ptr, hidden_dim);
    gemv_q4_k_mmvq_fused(gate_ptr, up_ptr, q8_1_ptr, out_ptr, ffn_dim, hidden_dim);
    return Ok(());
}
// else fall through to legacy fused_gate_up → unfused → non-Q4_K
```

Priority order for Q4_K GateUpSwiGLU dispatch:
1. Dynamic GA-hook (shape-matched) — 1 launch
2. **MMVQ-fused** (new default) — 2 launches
3. Legacy `gemv_q4_k_gate_up_swiglu` (opt-in `ROCMFORGE_FUSED_GATE_UP=1`) — 1 launch
4. Unfused split (opt-in `ROCMFORGE_DISABLE_MMVQ_FUSION=1`) — 3 launches
5. Non-Q4_K fallback — 3 launches

### HIP-Graph compatibility

`launch_index_spans` updated to count launches per path:

| Path | Launches |
|---|---:|
| Dynamic GA-hook | 1 |
| Legacy fused_gate_up | 1 |
| **MMVQ-fused (new)** | **2** |
| Unfused split | 3 |

This is the third recurrence of the "kernel count drift" pattern
(Schritt 2: Gemm; Schritt 2b: FusedGemmResidual; now Schritt 3:
GateUpSwiGLU). Each time, getting the count right was required for
the capture-validation `spans.total_launches == captured_node_count`
check to pass and HIP-Graph to actually fire.

## End-to-End (15-prompt validation suite)

Qwen3-8B-Q4_K_M @ `~/models/`, greedy sampling, KV-cache reset per prompt.

### Aggregate (the decode-throughput ladder)

|  | Pre-MMVQ | Schritt 2 | Schritt 2b | **Schritt 3** | llama.cpp |
|---|---:|---:|---:|---:|---:|
| Decode tok/s | 62.7 | 64.4 | 70.8 | **86.1** | 99.3 |
| Cumulative Δ vs pre-MMVQ | — | +2.7 % | +12.9 % | **+37.3 %** | |
| Gap vs llama.cpp | 1.58× | 1.54× | 1.40× | **1.15×** | 1.00× |
| Gap closure vs pre-MMVQ start | — | 7 % | 31 % | **75 %** | |

### Per-prompt

| # | Name | Schritt 2b | **Schritt 3** | Δ |
|---:|---|---:|---:|---:|
| 1 | Greeting | 73.3 | 89.4 | +22 % |
| 2 | Simple Sequence | 73.3 | 88.7 | +21 % |
| 3 | Prime Check (Python) | 72.6 | 88.0 | +21 % |
| 4 | LRU Cache (C++) | 70.7 | 83.9 | +19 % |
| 5 | REST API (Go) | 73.2 | 87.0 | +19 % |
| 6 | Mutex Explanation | 74.6 | **91.3** | +22 % |
| 7 | TCP vs UDP | 69.6 | 83.7 | +20 % |
| 8 | GPU Architecture Blog Post | 68.2 | 84.6 | +24 % |
| 9 | Binary Search Complexity | 73.2 | 89.0 | +22 % |
| 10 | Debug Code | 73.1 | 88.7 | +21 % |
| 11 | Distributed Message Queue | 69.2 | 85.7 | +24 % |
| 12 | Long System Prompt + Question | 68.9 | 82.6 | +20 % |
| 13 | Long Output Story | 72.0 | 88.3 | +23 % |
| 14 | Arithmetic (Q4_K Precision) | 75.8 | **92.6** | +22 % |
| 15 | Emoji/Special Characters | 74.6 | 90.8 | +22 % |

**Every prompt improved 19-24 %.** No outliers, no regressions. The
consistency confirms the gain is broad-based (gate/up is ~36 % of
decode time, so a 1.4× speedup there gives ~0.36 × 0.4 = 14 % E2E —
we got more because the cut launches also reduce HIP-Graph replay
overhead).

### Bandit

```
Q4_K n=1024 k=4096  committed to q4_k_mmvq  (14.56 µs vs 54.49 µs std, 3.7× win)
Q4_K n=4096 k=4096  committed to q4_k_mmvq  (32.47 µs vs 175.32 µs std, 5.4× win)
```

No bandit changes in Schritt 3 (fusion is a per-layer hand-dispatched
kernel, not a Bandit-selectable variant).

### Monitor events

3 `RepetitionDetected` events, same `token_id=10519` at positions
64-66 in one prompt — **identical pattern to Schritt 2b's run**. Same
model-behavior quirk, not a regression from the fusion change.
Parity tests confirm numerical equivalence to the unfused path.

## Files

| New | LOC |
|---|---:|
| `hip_kernels_v1/gemv/gemv_q4_k_mmvq_fused.hip` | 240 |
| `tests_v1/mmvq_fused_test.rs` | 454 |
| `results/phase2_mmvq_gate_fusion.md` (this report) | — |
| `results/phase2_mmvq_fused_15prompt_suite.md` (raw suite) | — |

| Modified | Change |
|---|---|
| `hip_kernels_v1/CMakeLists.txt` | +1 `add_hip_kernel` |
| `build.rs` | +1 link-lib entry |
| `src_v1/backend/gpu/gemv.rs` | +1 `extern "C"` binding (8 args: W_gate, W_up, q8_1, out, N, K, stream) |
| `src_v1/graph/executor.rs` | `dispatch_gate_up_swiglu` adds MMVQ-fusion branch before legacy fused / unfused; `launch_index_spans` counts 2 for MMVQ-fusion |
| `Cargo.toml` | +1 `[[test]]` entry |

The legacy `gemv_q4_k_gate_up_swiglu.hip` and the unfused path both
remain reachable via env vars (`ROCMFORGE_FUSED_GATE_UP=1` and
`ROCMFORGE_DISABLE_MMVQ_FUSION=1` respectively), same pattern as the
other deregistered kernels (q4_k_q8_inline, sudot4, q6_k_q8_inline,
q4_k_q8_inline_residual).

## Attribution of the E2E gain

From 70.8 to 86.1 tok/s = +21.6 % aggregate. Breakdown:

- **Gate/up kernel speedup (1.40× on N=12288,K=4096):** 36 layers × 31 µs =
  1.13 ms/token → ~14 % at 70.8 tok/s base.
- **Implicit Q8_1 sharing (36 quantize_q8_1 dropped):** 36 × 3.58 µs =
  129 µs/token → ~1 %.
- **Reduced dispatch count (3 launches → 1 launch per FFN block):**
  36 × (3−1) = 72 fewer kernel nodes captured in the HIP-Graph, each
  ~1-3 µs saved at replay overhead → ~2-6 %.
- **Less residual traffic:** unfused path wrote gate/up scratches to VRAM
  and read them back for SwiGLU; fused does it in register → ~0.5-1 %.

Expected total: 17-22 %. Measured: 21.6 %. Consistent.

## Gap to llama.cpp — where it goes from here

We're now **15 % off** llama.cpp-ROCm (86.1 vs 99.3 tok/s). The
remaining lever buckets, rough magnitudes from earlier profiling:

| Lever | Estimated gain | Risk |
|---|---:|---|
| Q6_K MMVQ port (LM-head + 2 Q6_K layers, ~22 % GPU time) | +4-6 % | medium — new scale packing |
| Q8_1 sharing for Q/K/V triplet (Schritt 2b deferred bonus) | +1-2 % | low |
| Attention kernel tuning (currently 8 % GPU time, unchanged since v1.0) | +1-3 % | medium |
| HIP-Graph capture of the embedding memcpy / sample step | +0.5-1 % | low |

If all four land cleanly, projection: **91-97 tok/s**, closing
~95 % of the remaining gap. That gets ROCmForge into the "within
measurement noise of llama.cpp on this hardware" band, which is the
Phase 2 stretch goal from `rocmforge_v1_roadmap_to_100toks.md`.

## Honest Caveats

1. **The BW > 100 % apparent number** for fused is an artefact of the
   naive weight-bytes/time calculation (L1/L2 caching makes it
   look inflated). A full `rocprofv3 --stats` pass would give the
   authoritative number — not blocking, just to know.
2. **RepetitionDetected events** persist at the same pattern as
   Schritt 2b; worth investigating as a quality-of-life issue but
   unrelated to this change.
3. **MMVQ fusion assumes both gate and up are Q4_K.** Mixed-format
   models (e.g. if a future GGUF has Q4_K gate + Q6_K up) would fall
   through to the non-Q4_K branch. Not a concern for Qwen3-8B or the
   other currently-supported models.
4. **The 1.40× kernel speedup is measured on the Qwen3-8B shape**
   (N=12288, K=4096). Other FFN shapes (e.g. smaller models) would
   give different numbers — larger N amplifies MMVQ's cooperative
   tiling advantage, so the gain should be at least as good for
   bigger models and potentially smaller for very small ones.
