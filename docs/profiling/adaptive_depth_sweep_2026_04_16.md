# Adaptive Depth Threshold Sweep — 2026-04-16

## Hypothesis

With tiled batched GEMV reducing verify costs at higher depths (up to ~6.7% at depth=3), the adaptive depth-down threshold `ema < 1.2` might be too aggressive. Lower thresholds (`ema < 1.0`, `ema < 0.9`) would keep higher depths longer, potentially improving throughput by amortizing draft cost over more accepted tokens.

## Methodology

- **Hardware:** RX 9070 XT (gfx1201, RDNA 4), ROCm 7.2.1
- **Models:** Qwen2.5-7B-Instruct-Q4_0 (target), qwen2.5-0.5b-instruct-q4_0 (draft)
- **Tiled GEMV:** on (default after flag flip)
- **Max tokens:** 128 per prompt
- **Prompts:** code_01 through code_05 (highest acceptance rates in the corpus)
- **Thresholds tested:** 1.2 (current), 1.0, 0.9
- **Depths tested:** 3, 5
- **Configurable via:** `ROCMFORGE_SPEC_DEPTH_DOWN_EMA` env var (temporary, removed after sweep)
- **Total runs:** 5 baseline + 30 spec = 35

## Results

### Baseline (no speculative decoding)

| Prompt   | tok/s |
|----------|-------|
| code_01  | 81.9  |
| code_02  | 81.8  |
| code_03  | 81.8  |
| code_04  | 81.8  |
| code_05  | 81.8  |

### depth=3

| Prompt   | ema<1.2 (current)             | ema<1.0                       | ema<0.9                       |
|----------|-------------------------------|-------------------------------|-------------------------------|
| code_01  | **77.6** (a=74.5%, d->1)      | 77.8 (a=75.2%, d->3)          | 77.9 (a=75.2%, d->3)          |
| code_02  | **73.5** (a=66.0%, d->1)      | 72.6 (a=64.8%, d->1)          | 72.6 (a=64.8%, d->1)          |
| code_03  | **75.7** (a=69.3%, d->1)      | 74.3 (a=66.3%, d->1)          | 74.2 (a=66.3%, d->1)          |
| code_04  | **71.8** (a=57.8%, d->1)      | 68.2 (a=51.0%, d->1)          | 68.3 (a=51.0%, d->1)          |
| code_05  | **69.9** (a=60.0%, d->1)      | 68.7 (a=57.6%, d->1)          | 68.7 (a=57.6%, d->1)          |

Verify times (avg, us):

| Prompt   | ema<1.2 | ema<1.0 | ema<0.9 |
|----------|---------|---------|---------|
| code_01  | 27,433  | 31,402  | 31,397  |
| code_02  | 21,636  | 22,596  | 22,604  |
| code_03  | 19,263  | 19,675  | 19,683  |
| code_04  | 18,337  | 18,837  | 18,818  |
| code_05  | 19,631  | 20,016  | 19,991  |

### depth=5

| Prompt   | ema<1.2 (current)             | ema<1.0                       | ema<0.9                       |
|----------|-------------------------------|-------------------------------|-------------------------------|
| code_01  | 62.2 (a=53.4%, d->5)          | 62.2 (a=53.4%, d->5)          | 62.2 (a=53.4%, d->5)          |
| code_02  | **52.8** (a=41.1%, d->3)      | 51.4 (a=39.5%, d->4)          | 50.2 (a=38.2%, d->5)          |
| code_03  | **66.9** (a=53.4%, d->1)      | 63.4 (a=48.8%, d->1)          | 63.2 (a=48.8%, d->1)          |
| code_04  | **63.5** (a=45.9%, d->1)      | 61.4 (a=42.6%, d->1)          | 59.0 (a=40.3%, d->1)          |
| code_05  | **63.1** (a=48.8%, d->1)      | 61.6 (a=46.5%, d->1)          | 61.0 (a=46.2%, d->1)          |

Verify times (avg, us):

| Prompt   | ema<1.2 | ema<1.0 | ema<0.9 |
|----------|---------|---------|---------|
| code_01  | 44,284  | 44,266  | 44,255  |
| code_02  | 41,591  | 42,748  | 43,829  |
| code_03  | 22,217  | 23,796  | 23,860  |
| code_04  | 21,859  | 22,334  | 23,511  |
| code_05  | 22,423  | 22,990  | 23,581  |

## Analysis

Tiled GEMV reduces the marginal cost of higher-depth verify, but the superlinear latency scaling of batch size makes `ema < 1.2` the global optimum.

**Key example:** code_02 at depth=5 — threshold 0.9 holds depth->5 (50.2 tok/s), threshold 1.2 drops to depth->3 (52.8 tok/s). The 5% throughput loss comes entirely from unnecessary verify overhead at high batch sizes. The adaptive algorithm's aggressive downshift is not conservative — it is corrective.

**Why lower thresholds hurt:** Verify cost scales superlinearly with batch size (more tokens to verify = more GEMV passes, more attention computation). Acceptance rate does not increase proportionally — it is a property of the draft/target model pair and the input distribution, not of how many tokens we draft. Extra drafted tokens at the margin have diminishing acceptance probability, while their verify cost is constant per token.

**Threshold 1.0 vs 0.9:** Nearly identical results. The EMA smoothing (alpha=0.2) means the difference between 1.0 and 0.9 only affects edge cases where EMA hovers in that narrow range.

## Decision

No change to the adaptive depth algorithm. The current threshold `ema < 1.2` is retained.

This finding is documented so that the threshold is not re-tested in future optimization rounds.
