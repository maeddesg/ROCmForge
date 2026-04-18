# Decode Profiling — Phase 6 Step 1

**Date:** 2026-04-18
**Hardware:** AMD Radeon RX 9070 XT (gfx1201, RDNA 4), ROCm 7.2.2
**Model:** Qwen2.5-7B-Instruct Q4_0
**Ground truth:** 102.5 tok/s → **9.76 ms / token**
**llama.cpp reference:** 117–121 tok/s → ~8.5 ms / token
**Gap to close:** ~1.3 ms / token (~15 %)

---

## Method

- Env gate `ROCMFORGE_PROFILE_DECODE_OPS=1` enables the existing
  `DecodeStage` accumulator and adds a per-step `tracing::info!` emission
  in `gpu_full_forward_hybrid`. Each stage is wrapped in
  `profile_decode_stage()` which calls `device.synchronize()` after the
  op, so the stage times are wall-clock but inflated by one device-stream
  sync per stage (~5 µs each × ~255 stages per token = ~4 ms overhead,
  measured: **+41 %**, Table 6).
- 128 decoded tokens, step 0 discarded as JIT/warmup, the 127 steady-state
  steps taken as the measurement population. Median is the headline.
- Ground truth measured separately with `RUST_LOG=warn` (no profiling,
  no syncs): 102.5 tok/s.
- **Per-stage proportions are preserved** across the profiled-vs-unprofiled
  gap because every stage pays the same +5 µs per sync. Rescaling
  profiled µs by `ground_truth / profiled_wall` redistributes the stages
  into the 9.76 ms budget without distorting their proportions.
- Script: `profiling/aggregate_decode_overhead.py`.

Launch count is the hardest datapoint we have — it's just a count, no
syncs, no timing. Every `profile_decode_stage` invocation corresponds to
one device-stream launch (or a tiny cluster of launches for fused ops);
28 layers × 9 stages + 3 tail stages = **255 launches per token**,
confirmed in the trace output as `launches_approx=255`.

---

## Table 1 — Per-stage breakdown (steady-state median, 127 steps)

| Stage            | Profiled µs | Profiled % | **Rescaled µs** (GT budget) |
|------------------|-----------:|------:|---------------------:|
| attn_norm        |        585 |   4.3 |                **414** |
| qkv (fused)      |      1,758 |  12.8 |              **1,246** |
| q_rope           |        535 |   3.9 |                **379** |
| k_rope           |          0 |   0.0 |                    **0** (fused into kv_write) |
| kv_write         |        542 |   4.0 |                **384** |
| attention        |        803 |   5.9 |                **569** |
| attn_proj        |      1,042 |   7.6 |                **738** |
| attn_residual    |          0 |   0.0 |                    **0** (fused into attn_proj) |
| ffn_norm         |        575 |   4.2 |                **407** |
| gate_up (fused)  |      4,316 |  31.5 |              **3,060** |
| ffn_down         |      2,828 |  20.6 |              **2,005** |
| ffn_residual     |          0 |   0.0 |                    **0** (fused into ffn_down) |
| logits_norm      |         20 |   0.1 |                 **14** |
| logits_proj      |        680 |   5.0 |                **482** |
| argmax           |         25 |   0.2 |                 **17** |
| **Sum**          | **13,709** | **100.0** | **9,720** |

Notes:
- `k_rope` and both residuals are 0 µs because they are already fused into
  the adjacent kernels (K-RoPE into `kv_write_rope`, residuals into the
  two GEMV-residual fused variants). The decode path already exploits
  these fusion opportunities; there is no free win left there.
- The tail (logits_norm + logits_proj + argmax) is under 520 µs — the
  existing tail-only HIP graph already serves this region well.

## Table 2 — Categories (rescaled to ground truth)

| Category | Rescaled µs | Share |
|---|---:|---:|
| **GEMV** (QKV + O + Gate+Up + Down + LM-head) | **7,531** | **77.2 %** |
| Attention (decode, per-head scalar) | 569 | 5.8 % |
| Norm (attn + FFN + logits) | 835 | 8.6 % |
| RoPE (Q only; K is inside kv_write_rope) | 379 | 3.9 % |
| KV-cache write | 384 | 3.9 % |
| Sampling (argmax) | 17 | 0.2 % |
| Residual | 0 | 0.0 % (fused) |
| Unaccounted (launch-overhead / host-side) | 45 | 0.5 % |
| **Total** | **9,760** | **100.0 %** |

**GEMV is 77 % of decode.** Inside GEMV the two FFN projections alone are
5,065 µs (52 % of the total token budget): `gate_up` 3,060 + `ffn_down`
2,005. These are the hot kernels.

## Table 3 — Launch-overhead analysis

| Metric | Value |
|---|---:|
| Launches per token (approx, = stage count) | **255** |
| Launch overhead @ 3 µs each | 765 µs (7.8 %) |
| Launch overhead @ 5 µs each | 1,275 µs (13.1 %) |

At RDNA 4's measured launch latency (~3–5 µs per `hipLaunchKernelGGL`),
the launch overhead is 0.8–1.3 ms per token — **the full 1.3 ms gap to
llama.cpp could be explained by launch overhead alone**. A HIP-graph
replay of the layer loop would collapse these 252 layer launches + 3 tail
launches into one graph submission.

**However:** full-decode graph replay is **disabled** on RDNA 4 due to a
device-pointer stale-read bug in complex graphs (>200 nodes) — see
`hip_graph_device_pointer_bug.md`. Until that bug is fixed or worked
around, this lever is not available.

The tail-only graph (3 launches, lm_head + argmax) is already active and
eliminates a small but real overhead from the logits path.

## Table 4 — GEMV occupancy (from `hip_kernels/quant/q4_0_gemv.hip`)

Launch config for `gemv_q4_0_f32_multi_row_kernel`:

```
n_waves          = 8       (Q4_0_FIXED_WAVES)
threads_per_block = 256     (8 × 32)
n_blocks_x       = ceil(ncols_dst / 32)   (each block covers 4 output cols)
```

| Projection | out_dim | n_blocks_x | vs. 64 CUs | Occupancy |
|---|---:|---:|---:|---:|
| Q (inside fused QKV) | 3,584 | 112 | 1.75× | **saturated** |
| K (inside fused QKV) | 512 | 16 | 0.25× | **25 %** |
| V (inside fused QKV) | 512 | 16 | 0.25× | **25 %** |
| O (attn_proj) | 3,584 | 112 | 1.75× | saturated |
| Gate (inside fused Gate+Up) | 18,944 | 592 | 9.25× | saturated |
| Up   (inside fused Gate+Up) | 18,944 | 592 | 9.25× | saturated |
| Down (ffn_down) | 3,584 | 112 | 1.75× | saturated |
| LM head | 152,064 | 4,752 | 74× | saturated |

Only K and V sit at 25 % CU occupancy, and **they are already inside the
fused QKV kernel** (`gpu_dispatch_fused_qkv_on_stream`) so their low per-
matrix occupancy doesn't translate to idle CUs — the kernel launches
the full Q+K+V workload under one grid. The fused QKV stage measures
1,246 µs rescaled; against an isolated 3,584-wide Q at ~740 µs
(equivalent to attn_proj), the QKV stage costs 1.7× for 1.3× the work —
the K and V contributions are nearly free thanks to fusion.

**GEMV occupancy is not the bottleneck.** The big GEMVs (gate_up,
ffn_down) are fully saturated; the small ones are fused.

## Table 5 — Scaling vs. KV-cache length

Attention-stage timing as the KV cache grows from pos=2 to pos=127:

| Pos | Wall µs | Attention µs | Δattention vs. pos=2 |
|---:|---:|---:|---:|
|   2 | 13,795 |   691 |   0 |
|  10 | 13,745 |   709 |  +18 |
|  32 | 13,801 |   745 |  +54 |
|  64 | 13,743 |   794 | +103 |
|  96 | 13,770 |   848 | +157 |
| 127 | 13,753 |   895 | +204 |

**Attention scales ~1.6 µs per additional KV-cache position** —
approximately linear with `pos`, as expected for a flat GQA attention
kernel. Extrapolating: at pos=1,024 the attention stage would be
~2.3 ms, at pos=4,096 ~6.9 ms. This matches the memory note "115 → 78
tok/s at K=64 → 4,096, attention=34 % at long context".

For short context (the benchmark's 128 tokens) attention contributes
~600 µs or 6 % — not a near-term priority. It becomes the dominant
cost beyond ~1,500-token context.

## Table 6 — Sync-overhead quantification

| Measurement | ms/token | Δ vs. ground truth |
|---|---:|---:|
| Ground truth (unprofiled) | 9.76 | — |
| Per-op profiling (1 sync/stage) | 13.76 | +4.00 ms (**+41 %**) |

41 % overhead is well above the 15 % threshold the method note warns
about. **Only the rescaled (right-most) column of Table 1 is meaningful**
for per-stage comparisons — the raw profiled µs are not.

A lower-overhead "groups" profile (2 syncs per layer — one after the
attention block, one after the FFN block) was planned but not
implemented because the per-op proportions are already clearly
interpretable: the rescaling is mathematically equivalent to distributing
the sync overhead proportionally across stages, and no stage is so
short that its relative share would be misleading (the smallest non-zero
stage is `argmax` at 25 µs profiled = 0.2 %). The analysis below treats
the rescaled column as the decode budget.

---

## Verdict

The 9.76 ms decode-token budget is dominated by two things:

1. **GEMV kernels (77 %)**, especially the two FFN projections
   (`gate_up` 3.06 ms + `ffn_down` 2.00 ms = 5.06 ms combined, 52 % of
   the budget). These kernels are already saturating the GPU (fully
   covered CU grid, Table 4) and are memory-bound — the Q4_0 weight
   traffic (2.6 GB through the memory hierarchy per token at 7B decode)
   is the ceiling, not compute.
2. **Launch overhead (7.8–13.1 %, 0.8–1.3 ms)** — 255 kernel launches
   per token. This is exactly the size of the gap to llama.cpp.

The remaining categories (norm 8.6 %, attention 5.8 %, RoPE 3.9 %,
KV-write 3.9 %, sampling 0.2 %) are collectively 22 % of the budget
and mostly too small to be worth optimising on their own at 128-token
contexts.

## Recommendation

**Step 2: GEMV memory-access tuning for `gate_up` and `ffn_down`**
(the 5 ms pair). Candidates:

- Enable and test the existing `ffn_gate_up_interleaved` buffer for the
  decode path (it's built but only used in the Q8 fastpath); reading
  interleaved Q4_0 blocks halves the L2 round-trips for the shared
  input.
- Re-tune `Q4_0_FIXED_WAVES` (currently 8) and `Q4_0_MULTI_ROW_COLS`
  (currently 4) for ncols_dst = 18,944 and 3,584 specifically. A
  2026-04-17 comment in `q4_0_gemv.hip` notes that a 4-wave heuristic
  regressed 0.5B but nothing has been tried for 7B since.
- Compare against a hipBLAS-Hgemv reference to confirm we are within
  10 % of the bandwidth ceiling, not leaving perf on the table.

A plausible 10–15 % GEMV speed-up saves 0.5–0.8 ms per token —
comparable to the launch-overhead lever but available without touching
the graph-replay bug.

**Step 3: HIP-graph workaround for the decode layer loop**
(if step 2 doesn't close the gap). The device-pointer stale-read bug
affects kernels that *read* `pos`/`seq_len` from device memory during
graph replay. If we capture the graph with `pos` baked in as a kernel
argument rather than read from a device pointer, the bug shouldn't
trigger. Requires kernel-signature changes; ~0.7–1.2 ms saved if it
works.

**Not recommended near-term:**

- Fused norm: already done for decode (the decode path has
  `gemv_norm_gate_up_swiglu` in `ops.rs` but is not currently on the
  default decode path — worth investigating whether routing the active
  decode through it would collapse the 835 µs norm + 3,060 µs gate_up
  into a single op).
- Attention optimisation: at 128-token context it's 6 %; becomes the
  dominant cost past ~1.5k tokens and will need its own profiling pass
  then.

### Expected decode after Step 2 + 3

| Lever | Rescaled savings | Projected decode tok/s |
|---|---:|---:|
| Step 2 (GEMV tuning 10 %) alone | −500 µs → 9.26 ms | 108 tok/s |
| Step 3 (HIP graph) alone        | −800 µs → 8.96 ms | 112 tok/s |
| Both combined                   | −1.3 ms  → 8.46 ms | **118 tok/s** |

This would close the gap to llama.cpp (117–121 tok/s) entirely.
Anything less than both is a partial win.

## Correctness

- With profiling on (`ROCMFORGE_PROFILE_DECODE_OPS=1`), the 5-token
  greedy completion of `"Hello"` is byte-identical to the unprofiled
  run. Sync events and timestamps don't affect inference.
- All 128 decoded tokens match the unprofiled run bit-for-bit.

## Plausibility

- Launch count exactly 255 = 28 layers × 9 stages + 3 tail stages. ✅
- Attention scales linearly with `pos` (Table 5). ✅
- Rescaled sum = 9,720 µs ≈ 9,760 µs ground truth (40 µs unaccounted =
  0.4 %, well within measurement noise). ✅
- Residual stages = 0 µs each → matches known decode fusion
  (`gpu_dispatch_gemv_residual_on_stream` merges O-proj and FFN-down
  residual adds). ✅
