# Spec-Step Cost Breakdown Analysis

**Date:** 2026-04-16  
**Commit:** f360a32 (feat/gqa-tiled-attention-experimental)  
**Hardware:** RX 9070 XT (gfx1201), ROCm 7.2  
**Config:** depth=1, 128 max_tokens, Qwen2.5-7B Q4_0 target + 0.5B Q4_0 draft

## Raw Measurements (avg μs per spec step)

| Phase | code_01 | chat_01 | prose_03 | Avg | % of Step |
|-------|---------|---------|----------|-----|-----------|
| draft_forward | 1,992 | 1,978 | 1,983 | 1,984 | 10.2% |
| target_verify | 17,237 | 17,230 | 17,228 | 17,232 | 88.6% |
| accept_reject | 234 | 236 | 233 | 234 | 1.2% |
| host_overhead | ~0 | ~0 | ~0 | ~0 | ~0% |
| **TOTAL** | **19,463** | **19,444** | **19,444** | **19,450** | **100%** |

**Baseline 7B decode:** ~82 tok/s → ~12,195 μs/token (single forward pass)

## Hypothesis Verdicts

### H_A: Draft Compute is the Bottleneck — REJECTED

**Claim:** The 0.5B draft model's forward pass dominates the spec-step cost.

**Verdict:** Draft forward accounts for only **10.2%** (1,984 μs) of the total step time.
At depth=1, the draft model runs a single decode step (embed + 24 layers + logits).
The 0.5B model's GEMV operations are memory-bandwidth-cheap on the 9070 XT (256 GB/s).
Even at depth=5, draft would be ~10,000 μs — still well below target verify.

**Conclusion:** Draft compute is **not** the bottleneck. Increasing draft depth up to ~3-5 
adds negligible relative cost.

### H_B: Target Compute is the Bottleneck — CONFIRMED

**Claim:** The 7B target model's batched verify forward dominates spec-step cost.

**Verdict:** Target verify accounts for **88.6%** (17,232 μs) of the total step time.
This is **1.41x** the baseline single-token decode (12,195 μs), despite processing 
only N+1=2 tokens in the batched verify.

The 41% overhead over baseline comes from:
1. **Batched GEMV dispatch**: For each weight matrix, the batched kernel loads weights
   once but computes N+1 dot products. At depth=1 this doubles ALU work per weight load.
   However, batched GEMV falls back to sequential loops for large projections 
   (FFN intermediate = 18,944 for 7B) when LDS exceeds 32 KB.
2. **All-heads verify attention**: The verify attention kernel processes all heads × 
   all query positions in one dispatch, but the KV cache reads scale with 
   `seq_len × n_verify × num_kv_heads`.
3. **Batched RoPE, RMSNorm, elementwise ops**: Each requires N+1x the work of decode.

The key insight: **target verify at depth=1 costs 17.2ms but produces up to 2 tokens**, 
while baseline costs 12.2ms for exactly 1 token. For spec decode to be profitable:

```
verify_cost / (1 + acceptance_rate) < baseline_cost
17,232 / (1 + α) < 12,195
α > 0.413 → need >41.3% acceptance rate to break even
```

This matches benchmarks: code prompts (α=73-90%) are profitable, creative prose 
(α=28-35%) is not.

### H_C: Orchestration & Host-Device Sync is the Bottleneck — REJECTED

**Claim:** hipDeviceSynchronize, D2H memcpy, and host-side orchestration between 
draft and verify phases add significant overhead.

**Verdict:** Accept/reject logic (including KV correction forward on mismatch) 
accounts for only **1.2%** (234 μs). Host overhead is effectively **0 μs** — 
the Vec allocations, token comparisons, and env flag checks are invisible at 
microsecond granularity.

The accept_reject phase includes a draft model forward pass on mismatch 
(KV cache correction), which explains why it's ~234 μs rather than pure host work.
On full acceptance (no mismatch), this would drop to near-zero.

There is **no measurable host-device synchronization gap** between the draft and 
verify phases. The HIP Event timestamps show the GPU transitions from draft to 
verify work without idle time.

## Implications for Optimization

### Why depth=1 outperforms depth=3 and depth=5

At depth=1:
- Draft cost: 1,984 μs (1 step)
- Verify cost: 17,232 μs (2 tokens batched)
- Total: 19,450 μs for up to 2 tokens

At depth=3 (extrapolated):
- Draft cost: ~5,950 μs (3 sequential steps)
- Verify cost: ~20,000+ μs (4 tokens batched, larger FFN sequential fallback)
- Total: ~26,000 μs for up to 4 tokens

The sequential draft steps scale linearly, but the verify cost scales sub-linearly 
because weight loading is amortized. However, the acceptance rate drops compoundingly:
if per-token α=0.73, then all-3-accepted probability = 0.73³ = 0.389.

### Priority optimization targets

1. **Target verify GEMV (88.6%)**: The batched GEMV fallback to sequential loops for 
   large projections (FFN intermediate=18,944) negates much of the batching benefit.
   Fixing the LDS overflow for large projections would reduce verify cost significantly.

2. **Batched GEMV LDS tiling**: The 32 KB LDS limit forces sequential fallback when 
   `batch_size * output_dim * sizeof(f32) > 32,768`. For depth=1 with FFN, that's 
   `2 * 18,944 * 4 = 151,552` bytes — 4.6x the limit. A tiled accumulation strategy 
   could keep the single-weight-load advantage.

3. **Draft model is already cheap enough**: At 10.2% of step cost, optimizing the 
   draft model has diminishing returns. The draft model's purpose is accuracy 
   (acceptance rate), not speed.

4. **Host overhead is non-existent**: No synchronization optimization needed.

## Break-Even Analysis

| Acceptance Rate | Effective tok/s (depth=1) | vs Baseline (82 tok/s) |
|----------------|---------------------------|----------------------|
| 90% | 97.8 | +19.3% |
| 73% | 88.9 | +8.4% |
| 50% | 77.1 | -6.0% |
| 41.3% | 72.7 | -11.3% (break-even verify cost) |
| 30% | 66.8 | -18.5% |

Formula: `effective_tok_s = (1 + α) / (step_time_s)` where `step_time_s = 19,450 / 1e6`
