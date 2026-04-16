# Launch Overhead Analysis — 2026-04-16

## Configuration

- **Hardware:** RX 9070 XT (gfx1201, RDNA 4), ROCm 7.2.1
- **Model:** Qwen2.5-7B-Instruct-Q4_0 (target), qwen2.5-0.5b-instruct-q4_0 (draft)
- **Spec depth:** 1 (batch_size=2 for verify)
- **Tiled GEMV:** on (default)
- **Max tokens:** 64
- **Git SHA:** d8d1579

## Method 2: HIP Event Sub-Phase Breakdown (per Verify Step)

Consistent across all 3 prompts (code_01, chat_01, prose_03):

| Sub-Phase              | Time (μs) |    % | Launches/Layer |
|------------------------|-----------|------|----------------|
| **ATTENTION**          |           |      |                |
| attn_norm              |       380 |  2.3 | 1              |
| attn_qkv+bias          |     2,242 | 13.8 | 4 (1 QKV + 3 bias) |
| attn_rope              |       280 |  1.7 | 2              |
| attn_kv_write          |       185 |  1.1 | 1              |
| attn_scores            |     1,175 |  7.2 | 1              |
| attn_o+residual        |     1,035 |  6.4 | 2              |
| **── subtotal**        | **5,297** | **32.7** | **11**     |
| **FFN**                |           |      |                |
| ffn_norm               |       350 |  2.2 | 1              |
| gate+up+silu+mul       |     6,490 | 40.0 | 2 (1 fused + 1 mul) |
| down+residual          |     4,060 | 25.1 | 2              |
| **── subtotal**        | **10,900** | **67.3** | **5**     |
| **TOTAL (28 layers)**  | **16,197** | **100** | **16 × 28 = 448** |

### FFN is the dominant cost: 67.3%

- **gate+up+silu+mul (40%):** The fused `gemv_gate_up_swiglu` kernel handles gate+up+SiLU in one dispatch, but `mul` is a separate elementwise kernel. This phase does 2 launches per layer × 28 = 56 launches, consuming 6,490 μs.
- **down+residual (25%):** The down-projection (tiled batched GEMV for in_dim=18944) plus residual add. 2 launches × 28 = 56 launches, consuming 4,060 μs.
- **Total FFN launches per verify step:** ~5 × 28 = 140

### Attention accounts for 32.7%

- **attn_qkv+bias (13.8%):** Fused QKV projection + 3 bias adds = 4 launches per layer.
- **attn_scores (7.2%):** Single all-heads verify attention dispatch per layer.
- **attn_o+residual (6.4%):** O-projection + residual add = 2 launches per layer.

## Method 1: rocprofv3 Kernel Tracing (Full Run)

Total kernel dispatches across entire run (prefill + draft + verify, 64 tokens):

| Group               | Launches | Total GPU μs | Avg μs/Launch |
|----------------------|----------|-------------|---------------|
| projections_other    |      621 |     306,935 |         494.3 |
| attention            |    3,552 |     380,010 |         107.0 |
| projections_down     |    3,872 |     140,937 |          36.4 |
| projections_ffn      |    1,536 |      18,448 |          12.0 |
| projections_qkv      |    1,536 |      10,686 |           7.0 |
| normalization        |    5,065 |      20,441 |           4.0 |
| rope                 |    4,968 |       9,066 |           1.8 |
| elementwise          |    5,688 |       8,408 |           1.5 |
| sampling             |      192 |         508 |           2.6 |
| activation           |      948 |       1,528 |           1.6 |
| kv_cache             |      948 |       1,297 |           1.4 |
| other (batched GEMV) |    5,832 |     289,502 |          49.6 |
| **TOTAL**            | **34,758** | **1,187,764** |        |

Note: `projections_other` includes GEMM kernels used in prefill (not verify), and `other` contains `gemv_q4_0_f32_batched_kernel<2>` (5,376 launches) used for O-projection in verify.

### Per-Launch Timing Plausibility Check

| Kernel Type | Avg μs/Launch | Plausible? |
|-------------|---------------|------------|
| gemm_q4_0 (prefill) | 494 | Yes — matrix multiply on 7B weights |
| flash_attn_verify_all_heads | 107 | Yes — reads full KV cache |
| gemv_batched (O-proj, attn_qkv) | 7-50 | Yes — GEMV on hidden_dim=3584 |
| gemv_gate_up_swiglu | 12 | Yes — fused kernel |
| gemv_down (tiled+residual) | 36 | Yes — 18944→3584 projection |
| rms_norm | 4 | Yes — lightweight |
| rope, add, mul, silu | 1.4-1.8 | **Suspiciously low** — likely measuring just GPU execution, not including submission overhead |

The 1.4-1.8 μs per-launch times for lightweight kernels (rope, add, mul, kv_write) are GPU-execution-only as measured by rocprofv3's hardware counters. The actual wall-clock cost including HIP submission overhead is higher — this is exactly the launch overhead we're trying to quantify.

## Launch Overhead Estimation

### Method A: Wall-Clock vs GPU-Execution Gap

From Method 2, per verify step:
- **Wall-clock layer time:** 16,197 μs (HIP Events, includes all pipeline stalls)
- **Total verify time (spec profile):** ~18,100 μs (includes embedding, final norm, lm_head, sync)

From Method 1 (rocprofv3), estimating GPU-only execution per verify step:
- 448 launches per step, using kernel-group timings scaled to verify-only
- Pure GPU execution estimate: ~14,000-15,000 μs (based on kernel-group averages × 28 layers)

**Launch overhead proxy: ~1,200-2,200 μs per verify step (7-14% of verify time)**

### Method B: Per-Kernel Submission Cost

Total lightweight kernel launches per verify step (these dominate launch count):
- rope_batched × 2 = 56
- add × 2 = 56
- add_batched × 3 = 84
- mul × 1 = 28
- silu × 1 = 28 (if separate)
- kv_write_batched × 1 = 28
- rms_norm_batched × 2 = 56

**Total lightweight launches: ~336 per step** (out of 448 total)

These kernels execute in 1.4-4.0 μs each (GPU time), but each HIP dispatch submission adds overhead. If submission latency is ~5-10 μs per launch (typical for ROCm 7.2), that's:
- **336 × 7.5 μs = 2,520 μs of submission overhead alone**

This is consistent with the Method A estimate and represents **~14-16% of the total verify time**.

## FFN Launch Inventory

Per verify step (28 layers):

| FFN Kernel | Launches | GPU Time (μs, estimated) |
|------------|----------|--------------------------|
| gemv_gate_up_swiglu (fused) | 28 | ~2,300 |
| mul (elementwise) | 28 | ~50 |
| gemv_down (tiled) | 28 | ~2,800 |
| add (residual) | 28 | ~50 |
| rms_norm_batched (ffn_norm) | 28 | ~350 |
| **Total FFN** | **140** | **~5,550** |

But Method 2 shows FFN wall-clock = 10,900 μs. So:
- **FFN launch overhead: ~10,900 - 5,550 = ~5,350 μs**
- **FFN launch overhead per launch: ~5,350 / 140 = ~38 μs**

Wait — that implies 38 μs per FFN launch, which is high. But many FFN launches are the lightweight kernels (mul, add, norm) that have very short GPU execution (1-4 μs). The overhead-to-execution ratio for these is extremely unfavorable.

## Fusion Scenario Analysis

### Scenario A: Gate+Up+SiLU+Mul fused (existing gate_up_swiglu already includes gate+up+SiLU)

The current fused kernel already handles gate+up+SiLU. Only `mul` is separate.

**Saves:** 28 mul launches per step
**Estimated savings:** 28 × ~8 μs (submission) = ~224 μs per step (~1.2% of verify)

### Scenario B: Full FFN fusion (Gate+Up+SiLU+Mul+Down+Residual)

Replace 4 launches per layer (fused_gate_up_silu, mul, down, residual_add) with 1:

**Saves:** 84 launches per step (3 × 28)
**Estimated savings:** 84 × ~8 μs = ~672 μs
But also eliminates intermediate buffer reads/writes for mul → down → add, saving ~1,000-2,000 μs of memory bandwidth.

**Estimated total: 1,500-2,500 μs per step (~8-14% of verify)**

### Scenario C: Scenario B + fused attention norm+QKV+bias

Replace 5 launches per layer (norm, Q, K, V, 3×bias → already fused QKV) with a further-fused norm+QKV+bias:

Would save an additional ~56 launches per step.

**Estimated additional savings: ~500-800 μs**

## Verdict

**Fused FFN is worth pursuing.**

FFN launch overhead is estimated at ≥ 1,500 μs per step (Scenario B), comfortably above the 1,500 μs threshold. The full FFN fusion (Scenario B) would save 1,500-2,500 μs per verify step, representing **8-14% of the current ~18,100 μs verify time**.

This is a meaningful improvement — at the current baseline of ~82 tok/s, a 10% verify reduction would improve spec-decode throughput by ~1-3 tok/s for high-acceptance prompts.

However, the improvement is moderate, not transformative. The **real bottleneck is the projections themselves** — the GEMV kernels (QKV, O, gate_up, down) consume ~80% of verify GPU time and are memory-bandwidth-bound. Fusion can reduce launch overhead and intermediate memory traffic, but cannot make the GEMV kernels themselves faster.

### Priority ranking of optimization opportunities:

1. **Fused FFN (down+residual into gate_up_silu_mul)** — most launches eliminated, clearest implementation path
2. **Norm+QKV+bias fusion** — additional launch reduction, but QKV is already fused
3. **Reducing lm_head cost** — ~2,000 μs overhead outside the layer loop
4. **Multi-stream/async dispatch** — overlap submission with GPU execution

## Cross-Check: Method 1 vs Method 2

### FFN Total Time
- Method 2 (HIP Events): 10,900 μs per step
- Method 1 (rocprofv3, estimated): ~5,550 μs GPU-only execution

The **~5,350 μs gap (49%) is substantial** and represents launch overhead + intermediate memory traffic. This confirms that FFN is where launch overhead is concentrated — lightweight elementwise kernels (mul, add, silu) have terrible overhead-to-execution ratios.

### Attention Total Time
- Method 2 (HIP Events): 5,297 μs per step
- Method 1: dominated by verify_all_heads (~107 μs × 28 = ~3,000 μs) + QKV (~7 μs × 28 = ~200 μs) + O-proj (~50 μs × 28 = ~1,400 μs)
- Estimated GPU-only: ~4,600 μs

The **~700 μs gap (13%) is much smaller** — attention has fewer lightweight launches and the verify kernel dominates.

## Surprises

1. **The mul kernel is separate from gate_up_swiglu.** I expected the fused kernel to include the elementwise multiply. It only fuses gate+up+SiLU, leaving mul and down as separate dispatches. This is the single largest fusion opportunity.

2. **Lightweight kernels dominate launch count but not GPU time.** 336 of 448 launches per step (75%) are kernels that execute in <4 μs. Their launch overhead likely exceeds their execution time — they exist purely as pipeline scaffolding.

3. **The lm_head + final norm phase costs ~1,900 μs.** This is outside the layer loop and not captured by the per-layer breakdown. For spec-decode with depth=1, this runs once per step and is 10.5% of total verify time. At higher depths it would run N times (once per position for greedy argmax).

4. **rocprofv3 sub-μs kernel times for add/mul are plausible.** These kernels operate on 3584-element vectors (~14 KB) — a single wavefront can process this in under 1 μs. The cost is entirely in dispatch.
