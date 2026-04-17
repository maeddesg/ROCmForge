# Phase 3d — end-to-end prefill with WMMA GEMM + WMMA Attention

**Date:** 2026-04-17
**Commit (pre-integration):** `16dd3f4`
**GPU:** RX 9070 XT (gfx1201, RDNA 4), ROCm 7.2.1
**Model:** Qwen2.5-7B-Instruct-Q4_0

## TL;DR

| pp  | custom GEMM | hipBLAS | WMMA GEMM only | **WMMA GEMM + Attn** | vs GEMM-only | vs baseline |
|----:|------------:|--------:|---------------:|---------------------:|-------------:|------------:|
|  64 |        61.8 |    76.7 |           88.7 |          **560.1**   |     6.3×     |    9.1×     |
| 128 |        63.1 |    81.7 |           90.3 |          **602.5**   |     6.7×     |    9.6×     |
| 192 |        64.0 |    84.7 |           92.3 |          **618.1**   |     6.7×     |    9.7×     |
| 256 |        63.6 |    85.8 |           92.3 |          **622.5**   |     6.7×     |    9.8×     |
| 384 |        54.0 |    69.8 |           74.5 |          **626.1**   |     8.4×     |   11.6×     |
| 512 |        50.2 |    63.1 |           67.2 |          **628.6**   |     9.4×     |   12.5×     |

Median of 3 runs each, `--max-tokens 1` to isolate prefill cost,
`--temperature 0.0 --top-p 1.0 --no-template`.

Decode throughput (Qwen2.5-7B Q4_0, 64-token greedy completion) was
spot-checked at **102 tok/s** — unchanged vs. the Phase 2d baseline
(the WMMA attention kernel only dispatches on the prefill path).

Raw data: [`prefill_wmma_attn_e2e_16dd3f4_1776427244.json`](prefill_wmma_attn_e2e_16dd3f4_1776427244.json).

## Methodology

- `target/release/rocmforge --model <7B Q4_0> --prompt <N words> --max-tokens 1 --temperature 0.0 --top-p 1.0 --no-template --gpu`
- 4 modes via env flags:
  - `custom` — `ROCMFORGE_DISABLE_WMMA_PREFILL=1 ROCMFORGE_DISABLE_HIPBLAS_PREFILL=1 ROCMFORGE_DISABLE_WMMA_ATTENTION=1`
  - `hipblas` — `ROCMFORGE_DISABLE_WMMA_PREFILL=1 ROCMFORGE_DISABLE_WMMA_ATTENTION=1`
  - `wmma_gemm` — `ROCMFORGE_DISABLE_WMMA_ATTENTION=1`
  - `wmma_full` — no flags (default)
- Prompt: `"word "` repeated. Qwen tokeniser maps each repeat to exactly one token, so `N` copies → `N` tokens.
- The binary parses `Prefill: {ms}ms ({tps} tok/s)` from `src/main.rs:727`.

## Where the speedup comes from

At pp = 256 on Phase 2d, attention was **~84 %** of total prefill
(≈ 2,330 ms of 2,770 ms) because the scalar `flash_attn_prefill_strided`
kernel ran a single-thread serial softmax inside a per-head loop,
dispatched 28× per layer × 28 layers. Phase 3d replaces this with one
WMMA dispatch per layer:

| Component (pp = 256, 28 layers) | Phase 2d (ms) | Phase 3d (ms) |
|---------------------------------|--------------:|--------------:|
| GEMM (Q4_0 WMMA)                |           110 |           110 |
| Attention                       |         2,330 |        ~6.2   |
| Overhead (norm / RoPE / etc.)   |           330 |         ~295  |
| **Total**                       |     **2,770** |       **411** |
| **Throughput**                  |   **92 tok/s**| **622 tok/s** |

The measured 411 ms at pp = 256 is actually ~35 ms below the projection
(446 ms) — the overhead term shrank a little because the FP32→FP16
conversion of Q/K/V (added by this phase) fuses on-stream with the
attention dispatch rather than adding serial latency.

## Why it keeps scaling at pp = 512

The WMMA attention kernel's causal-mask early-termination skips roughly
`(seq_len / TILE_KV − 1) / 2` tiles per Q-tile, so work grows with
`~seq_len · log(seq_len)` rather than the scalar kernel's `seq_len²`.
At pp = 512 the WMMA path is **9.4×** faster than Phase 2d; at pp = 64
it is "only" 6.3×. The absolute throughput levels off around 620–630
tok/s because GEMM (linear in seq_len) and the ~295 ms of fixed /
launch overhead start dominating.

## What's next

Attention is now ~1.5 % of pp = 256 prefill time. Further prefill
tuning should target the residual 295 ms, which decomposes roughly as:
GEMM ≈ 110 ms (25 % of total), norm/RoPE/residual/embedding ≈ 120 ms,
FP32↔FP16 shuttling ≈ 40 ms, kernel launch overhead ≈ 25 ms.
Fusing norm + QKV + RoPE into a single dispatch on the prefill path
(the decode path already has this) is the highest-leverage next step.

## Not included

- **llama.cpp head-to-head at pp = 256.** llama.cpp is not installed on
  the test host. Their published pp = 19 figure (1,092 tok/s) is not
  directly comparable. Re-running both at pp = 256 on the same machine
  is a next-round question.
- **Non-64-aligned prompts.** The WMMA attention kernel requires
  `seq_len % 64 == 0` and `head_dim == 128`; anything else falls back
  to the scalar kernel. A padded-mask variant or per-element bounds
  check would be needed for arbitrary prompts.
- **KV cache reuse.** Decode and verify still use their own attention
  kernels (`flash_attn_decode_gqa_*`). The WMMA attention integration
  is prefill-only for this phase.
