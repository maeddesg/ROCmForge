# Batched Verification for Speculative Decoding

This document describes the batched verify implementation as of PR #14.

## Speculative Decode Loop

Entry point: `src/main.rs`, speculative decode branch (after `if let Some((...)) = draft_state`).

Each iteration:
1. Emit `next_token` (first iteration: prefill result)
2. `gpu_speculative_decode_step()` in `src/gpu/forward.rs:1307`
   - Draft phase: run 0.5B model for `spec_depth` tokens sequentially
   - Verify phase: run 7B model on `[input_token, draft[0], ..., draft[N-1]]` in one batched forward
   - Accept/reject: compare `target_argmax[i]` with `draft_tokens[i]`
   - KV cache correction: on mismatch, overwrite draft KV at the rejection position
3. Output `accepted_tokens`, update positions

## Batched Target Forward (`gpu_verify_forward`)

Location: `src/gpu/forward.rs`, `gpu_verify_forward()`.

**Input:** `verify_tokens: &[u32]` of length `N+1` (input_token + N draft tokens), `start_pos`.

**Processing:** Reuses the prefill infrastructure (`GpuPrefillScratch`) with `seq_len` temporarily set to `tokens.len()`. Each layer runs:
- RMSNorm → QKV projection → RoPE → KV write (positions `start_pos..start_pos+N`)
- All-heads verify attention (single dispatch)
- O-projection → residual → FFN (RMSNorm → gate/up → SiLU → down → residual)

GEMV dispatch (`src/gpu/ops.rs`): for `seq_len <= 8` and Q4_0 weights, uses the batched GEMV kernel when LDS fits (32 KB). Falls back to a sequential GEMV loop for large projections (FFN intermediate = 18944 for 7B).

**Output:** `Vec<u32>` of length `N+1` — argmax token IDs for each position.

## All-Heads Verify Attention

Location: `hip_kernels/attention.hip`, `flash_attn_verify_all_heads_kernel<TILE_SIZE>`.

**Why:** The naive approach launches one kernel per head per query position — 784 dispatches per layer for 7B (28 heads × 28 layers). The all-heads kernel collapses this to 1 dispatch per layer.

**Grid:** `dim3(num_kv_heads, n_verify)`. Each block handles one KV-head group for one query position. Within a block, `gqa_ratio` warps each handle one Q-head (GQA-aware).

**Structure:** 5 phases matching `flash_attn_decode_gqa_kernel`:
1. Load Q vectors from strided `[n_verify × num_heads × head_dim]` layout
2. Tiled K loading (TILE=32) from FP16 KV cache
3. Q·K dot products with causal masking
4. Online softmax (running max + sum for numerical stability)
5. Tiled V loading + weighted accumulation → write output

## Causal Mask

```
causal_len = start_pos + query_idx + 1
```

Query at index `i` in the verify batch attends to KV positions `0..start_pos+i` (inclusive). This is correct because verify token `i` sits at absolute position `start_pos + i`, and causal attention allows seeing all positions up to and including itself.

## KV Cache Rollback

The KV cache uses a linear layout: `[num_layers][max_seq_len × kv_size]` in FP16. Positions are written by absolute index, not appended sequentially.

On rejection at position `j` (where `j < spec_depth`), the target KV cache already has the correct entries at positions `start_pos..start_pos+j` from the batched verify forward. Entries at positions `start_pos+j+1..start_pos+N` contain stale data from rejected draft tokens, but these are never read — the next verify batch starts at `start_pos + len(accepted_tokens)`, and attention uses `causal_len` to ignore positions beyond the valid range.

No explicit rollback, zeroing, or recompute needed. The pointer-reset is implicit in the position arithmetic.

## Adaptive Speculation Depth

Location: `src/main.rs`, inside the speculative decode loop.

Tracks an exponential moving average (EMA) of accepted draft tokens per step:

```
ema_acceptance = 0.8 * ema_acceptance + 0.2 * accepted_this_step
```

Decision thresholds:
- `ema < 1.2` → decrease depth by 1 (min 1)
- `ema > 2.5` → increase depth by 1 (max `args.spec_depth`)

Initial value: `spec_depth * 0.5` (mildly optimistic). The depth converges within 5-10 steps. For low-acceptance prompts (creative text), it quickly reaches depth=1, which minimizes draft overhead. For high-acceptance content (code, repetitive text), it stays at the configured maximum.

Final depth shown in stats output when it differs from the initial setting.

## GEMV Dispatch for Batched Verify

Location: `src/gpu/ops.rs`, batched GEMV dispatch in `gpu_dispatch_gemm`.

**Standard batched GEMV** (`q4_0_gemv_batched.hip`): caches all Q8-quantized inputs in LDS. LDS budget: `batch × (in_dim/32) × 34` bytes, max 32 KB. Works for all projections where in_dim ≤ ~3584 (QKV, O-proj, gate, up, LM head).

**Tiled batched GEMV** (`q4_0_gemv_batched_tiled.hip`): tiles along the input dimension in chunks of 1024 elements when LDS limit is exceeded. Preserves single weight load. Gated behind `ROCMFORGE_EXPERIMENTAL_TILED_GEMV=1`.

**Sequential fallback**: when neither batched kernel fits, loops over batch elements with single-row GEMV. This is the default for large in_dim.

### Which projections overflow at each depth?

For Qwen2.5-7B (hidden=3584, FFN intermediate=18944):

| Depth | Batch | QKV/O (3584) | Gate/Up (3584→18944) | Down (18944→3584) |
|-------|-------|-------------|---------------------|-------------------|
| 1 | 2 | 7.6 KB ✓ | 7.6 KB ✓ | 40.3 KB ✗ |
| 3 | 4 | 15.2 KB ✓ | 15.2 KB ✓ | 80.5 KB ✗ |
| 5 | 6 | 22.8 KB ✓ | 22.8 KB ✓ | 120.8 KB ✗ |

Only the **FFN down-projection** exceeds the 32 KB LDS limit. It is called 28 times per verify pass (once per layer).

### Memory-Controller Pipelining (RDNA 4 Architecture Insight)

Profiling on RX 9070 XT (gfx1201) showed that the sequential GEMV fallback for FFN-down is **not as expensive as predicted**. The theoretical cost model assumes N separate weight reads from VRAM (one per batch element), but the actual overhead is much smaller:

- **Predicted**: 2× weight bandwidth at depth=1 → ~150 μs wasted per layer → ~4.2 ms over 28 layers
- **Measured**: only ~250 μs total savings from tiled kernel at depth=1 (~9 μs/layer)

The RDNA 4 memory controller pipelines back-to-back GEMV launches effectively. When sequential GEMV calls are dispatched without explicit synchronization between them, the GPU command processor overlaps the tail of one kernel with the head of the next. The weight data for kernel N+1 begins streaming while kernel N is still reducing its last warp — the memory controller maintains near-full bandwidth utilization across kernel boundaries.

**Implication**: eliminating kernel launches (via fusion or batching) primarily saves **launch submission overhead** and **command processor scheduling latency**, not memory bandwidth. The sequential fallback "wastes" ~2× bandwidth in theory but achieves ~1.02× in practice because the pipeline keeps the memory bus saturated.

This finding is critical for evaluating future optimizations:
- **Fused FFN** (gate+up+SiLU+down in fewer launches): primary benefit is reduced launch overhead, not bandwidth savings
- **WMMA GEMM**: benefit comes from compute efficiency (matrix cores), not from avoiding redundant weight reads
- **Launch overhead profiling** should measure kernel submission + sync cost per verify step to quantify the actual launch overhead budget
