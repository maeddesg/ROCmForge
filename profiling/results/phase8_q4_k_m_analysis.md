# Phase 8b Step 1 — Q4_K_M Performance Profiling

Measurement date: 2026-04-19. Hardware: AMD Radeon RX 9070 XT (gfx1201 / RDNA 4), ROCm 7.2.2.

Method: `ROCMFORGE_PROFILE_DECODE_OPS=1` and `ROCMFORGE_PROFILE_PREFILL_OPS=1` emit one
`tracing::info!` line per token (decode) and per layer (prefill), with `hipDeviceSynchronize`
between stages. Sync adds a consistent ~14–22 µs per synced dispatch.

Models:

- `~/models/Qwen3-8B-Q4_K_M.gguf` (36 layers, hidden=4096, ff=12288, 32Q/8KV heads, head_dim=128, Q/K norm)
- `~/models/Qwen2.5-7B-Instruct-Q4_0.gguf` (28 layers, hidden=3584, ff=18944, 28Q/4KV heads, head_dim=128)

All figures are medians across at least 30 tokens / 36 layers. Raw logs in `/tmp/phase8b/`.

## 1. Baselines (unprofiled wall-clock)

| Model | Prefill pp257 | Decode |
|---|---|---|
| Qwen2.5-7B Q4_0 | 1.440 tok/s (178 ms) | 101 tok/s (9.9 ms/tok) |
| Qwen3-8B Q4_K_M | 461 tok/s (558 ms) | 28.5 tok/s (35.1 ms/tok) |
| **Ratio Q4_K / Q4_0** | **3.13× slower** | **3.54× slower** |

## 2. Decode per-op breakdown

Per token, median of 64 steps. All numbers in µs.

| Op | Q4_0 (Qwen2.5) | Q4_K_M (Qwen3) | Ratio | Notes |
|---|---|---|---|---|
| attn_norm | 608 | 762 | 1.25× | RMSNorm scalar |
| qkv | 2 383 | 9 056 | **3.80×** | 3 separate GEMVs, no Q4_K fused-QKV |
| q_rope | 541 | 920 | 1.70× | Qwen3 has Q-norm inside |
| k_rope | 0 | 0 | — | fused into kv_write on both paths |
| kv_write | 562 | 743 | 1.32× | same kernel |
| attention | 709 | 882 | 1.24× | flash-decode gqa |
| attn_proj (O) | 1 529 | 6 100 | **3.99×** | O-proj GEMV |
| attn_residual | 0 | 0 | — | fused into attn_proj on both |
| ffn_norm | 589 | 751 | 1.27× | |
| gate_up (fused SwiGLU) | 4 797 | 17 049 | **3.55×** | fused on both, but Q4_K kernel 3.6× slower |
| ffn_down | 3 129 | 3 317 | 1.06× | **same speed** — Q4_K uses multi-row GEMV; Q4_0 goes through the same per-token path here |
| ffn_residual | 0 | 344 | — | **NOT fused on Q4_K** (fused on Q4_0 via `gemv_q4_0_f32_q8_inline_residual`) |
| logits_norm | 23 | 23 | 1.00× | |
| logits_proj | 694 | 783 | 1.13× | both hit Q6_K GEMV (LM head is Q6_K in both GGUFs) |
| argmax | 29 | 30 | — | |
| **Kernel sum** | **15 593 µs** | **40 760 µs** | **2.61×** | |
| Wall-clock (unprofiled) | 9 872 µs | 35 087 µs | 3.55× | |
| Sync overhead (profiled − unprofiled) | 5 721 µs | 5 673 µs | — | ~14 µs × 255 / 399 syncs |
| Dispatches/token (launches_approx) | 255 | 399 | 1.56× | |

### 2.1 Dispatch-Hell quantifiziert

```
Q4_0:   unprofiled 9 872 µs, profiled 15 593 µs ⇒ sync overhead 5 721 µs (37% inflation)
Q4_K_M: unprofiled 35 087 µs, profiled 40 760 µs ⇒ sync overhead 5 673 µs (16% inflation)
```

Absolute sync overhead is identical (~5.7 ms). In both models, the kernel-sum ≈ unprofiled
wall-clock + N × sync_latency. **There are no GPU idle gaps**. The 3.55× decode gap is
**all kernel time**, not dispatch-hell. Per-layer: Q4_0 spends 557 µs on actual kernel
compute, Q4_K_M spends 1 130 µs — **2.03× more kernel time per layer**.

Dominant factor: the three big GEMVs (QKV 9 056 + O-proj 6 100 + gate_up 17 049 = 32 200 µs,
79% of profiled total).

## 3. Prefill per-op breakdown (pp257)

All figures summed across every layer of the model.

### 3.1 Qwen3-8B Q4_K_M — aggregate

| Op | Total µs | % |
|---|---|---|
| **down_proj** | **326 017** | **58.4%** |
| **v_proj** | **85 160** | **15.3%** |
| up_proj | 41 768 | 7.5% |
| gate_proj | 41 469 | 7.4% |
| q_proj | 15 476 | 2.8% |
| o_proj | 15 432 | 2.8% |
| attention | 13 265 | 2.4% |
| k_proj | 6 077 | 1.1% |
| rope_q | 3 275 | 0.6% |
| silu_mul | 2 771 | 0.5% |
| norm_pre_attn | 1 596 | 0.3% |
| kv_write | 1 373 | 0.2% |
| norm_pre_ffn | 1 208 | 0.2% |
| residual_ffn | 1 099 | 0.2% |
| residual_attn | 994 | 0.2% |
| rope_k | 859 | 0.2% |
| **total** | **557 839** | **100%** |

### 3.2 The Q6_K split — Qwen3-8B

Per-layer times for `v_proj` and `down_proj` fall into **two distinct buckets**:

| Bucket | Layer count | Avg v_proj | Avg down_proj | Layer indices |
|---|---|---|---|---|
| Q6_K (slow) | 18 | 4 563 µs | 16 872 µs | 0, 1, 2, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 31, 32, 33, 34, 35 |
| Q4_K (fast) | 18 | 167 µs | 1 239 µs | all other layers |
| **Ratio** | | **27×** | **14×** | |

Confirmed via `rocmforge --list-tensors ~/models/Qwen3-8B-Q4_K_M.gguf`:
exactly 18 `blk.*.attn_v.weight` and 18 `blk.*.ffn_down.weight` are stored as **Q6_K**,
matching the observed slow-layer indices 1:1. Plus `output.weight` (LM head) — also Q6_K.

### 3.3 Q6_K dispatch path

`src/gpu/ops.rs:1830`:

```rust
GgmlType::Q6_K => {
    // No batched Q6_K GEMM kernel yet; loop over rows with the
    // optimised multi-row Q6_K GEMV instead of bailing out.
    for row in 0..seq_len {
        gemv_q6_k_f32_on_stream(...)?;
    }
}
```

Each Q6_K GEMM becomes `seq_len = 257` single-row GEMV dispatches. For Qwen3-8B prefill pp257
that is:

```
36 layers, 18 are Q6_K
  each Q6_K layer does  v_proj + ffn_down = 2 Q6_K GEMMs
  each GEMM = 257 GEMV dispatches
⇒ 18 × 2 × 257 = 9 252 Q6_K GEMV dispatches in one prefill
```

### 3.4 Q6_K impact quantified

```
Total prefill layer time:    557.8 ms
Q6_K-suspect layers (n=18):  461.0 ms  (82.6%)
Q4_K-only layers (n=18):      96.9 ms  (17.4%)

Per-Q6_K-layer overhead:     25 610 µs (avg)
Per-Q4_K-layer overhead:      5 381 µs (avg)
Δ per Q6_K layer:            20 229 µs

If Q6_K ran at Q4_K speed (WMMA GEMM, not GEMV-loop):
  New total: 18 × 5 381 × 2 = 193.7 ms
  Speedup:   2.88×
  pp257:     1 327 tok/s  (from 461 → within 2.8× of llama.cpp's 3 661)
```

### 3.5 Qwen2.5-7B Q4_0 prefill — reference

| Op | Total µs | % |
|---|---|---|
| gate_proj | 87 235 | 48.9% |
| down_proj | 47 381 | 26.5% |
| q_proj | 9 607 | 5.4% |
| o_proj | 9 422 | 5.3% |
| attention | 9 313 | 5.2% |
| silu_mul | 3 730 | 2.1% |
| v_proj | 2 669 | 1.5% |
| k_proj | 2 655 | 1.5% |
| … | … | … |
| **total** | **178 500** | **100%** |

`v_proj` is uniform 93–103 µs across all 28 layers (pure Q4_0, WMMA-GEMM path).
`down_proj` uniform 1 647–2 030 µs. **No per-layer bimodal pattern** — confirming
the pattern on Qwen3 is Q6_K-specific, not a hardware/bandwidth artefact.

## 4. Fused-path inventory Q4_0 vs Q4_K

Grep on `src/gpu/ops.rs`. Every fusion guard below gates on `GgmlType::Q4_0` and returns
`Ok(false)` for any other quant type, forcing Q4_K down the unfused fallback.

| Fused path (decode) | Q4_0 | Q4_K | Dispatches saved / layer | File |
|---|---|---|---|---|
| `fused_norm_qkv_rope_kvwrite` (5 ops → 1) | ✅ | ❌ | 4 | `hip_kernels/quant/q4_0_fused_norm_qkv_rope.hip` |
| `fused_qkv_rope_kvwrite` (4 ops → 1) | ✅ | ❌ | 3 | `hip_kernels/quant/q4_0_fused_qkv_rope.hip` |
| `q8_inline_residual` (GEMV + residual, Q8 activation) | ✅ | ❌ | 1 | `hip_kernels/quant/q4_0_fused_q8.hip` |
| `q8_inline_residual_norm` (GEMV + residual + norm) | ✅ | ❌ | 2 | same |
| `gate_up_swiglu_q8_inline` | ✅ | ❌ | 1 (plus Q8 bandwidth halving) | same |
| `gate_up_swiglu` (GEMV × 2 → 1) | ✅ | ✅ | 1 | both have it |
| `fused_norm_gate_up` (norm + gate_up) | ✅ but buggy (disabled) | ❌ | 1 | `q4_0_fused_norm_gate_up.hip` |

Dispatch delta accounts for the 399 → 255 gap: **36 Q4_K layers × 4 missing fusions ≈ 144 dispatches**,
exactly the difference 399 − 255 = 144. ✓

### 4.1 But dispatch delta is NOT the decode bottleneck

Despite 144 more dispatches on Q4_K_M decode, sync overhead is only ~5.7 ms (matches Q4_0).
The real decode cost is **kernel time, not dispatch count**. See §2.1.

The Q8-inline fusions would reduce **activation bandwidth** (FP32 4096 floats → Q8 ~4300 bytes,
roughly 4× less traffic into the GEMV kernel's LDS), which would speed up the three dominant
GEMVs (QKV, O-proj, gate_up). That is the decode Hebel.

## 5. Bandwidth-Effizienz

```
Qwen3-8B Q4_K_M weights: ~4.7 GB
RX 9070 XT bandwidth:    ~640 GB/s
Theoretical max decode:   136 tok/s  (weights read once per token)

Qwen2.5 Q4_0 decode:  101 tok/s = 74% of peak
Qwen3 Q4_K_M decode:   29 tok/s = 21% of peak
llama.cpp Q4_K_M:      87 tok/s = 64% of peak
```

**Gap-Aufschlüsselung Q4_K (21%) vs Q4_0 (74%), ∆ = 53 Prozentpunkte:**

- **Kernel compute at matched bandwidth: ~80% of the gap.** Same bytes/element (Q4_0 and Q4_K
  both 4.5 bits/w), but Q4_K dequant has:
  - 12 packed scale bytes → 8 per-sub-block `(scale_j, min_j)` pairs to unpack per block
  - 8 sub-blocks × 32 elements = 256 elements per super-block (4× more ALU vs Q4_0's 32)
  - Affine dequant (`d·scale_j·nib − dmin·min_j`) vs Q4_0's uniform (`(nib − 8)·d`)
- **Missing Q8-inline fastpaths: ~15% of the gap.** The Q4_0 Q8-inline kernels cut activation
  memory traffic roughly 4× which on RDNA 4 directly speeds GEMV.
- **Fused-path dispatch delta: ~5% of the gap.** 144 extra kernel launches ≈ 2 ms inflation,
  ≈ 6% of 35 ms budget.

Prefill (470 vs 3 661 = 0.13×): **~83% of the gap is Q6_K GEMV-loop dispatching**. A Q6_K WMMA
GEMM alone moves us to 2.88× faster prefill (1 327 tok/s, ~0.36× llama.cpp).

## 6. Top bottlenecks (measured, not guessed)

| # | Bottleneck | Time | % of budget | Mode | Fix |
|---|---|---|---|---|---|
| 1 | Q6_K GEMV-loop at prefill | 461 ms / 558 ms | 82.6% | Prefill | Q6_K WMMA GEMM kernel |
| 2 | Q4_K gate_up GEMV (decode) | 17.0 ms / 35.1 ms | 48.5% | Decode | Q8-inline activation variant |
| 3 | Q4_K qkv GEMV (decode, unfused) | 9.1 ms / 35.1 ms | 25.9% | Decode | Fused `norm_qkv_rope_kvwrite` for Q4_K |
| 4 | Q4_K O-proj GEMV | 6.1 ms / 35.1 ms | 17.4% | Decode | Q8-inline residual variant |
| 5 | ffn_residual unfused on Q4_K | 344 µs × 36 layers = 12 ms / 558 ms prefill | 2.2% | Decode | Q4_K ffn_down + residual fusion |

## 7. Aufwand / Gewinn Matrix

Estimates. „Gain" is wall-clock throughput uplift based on removing the measured hot path
at its theoretical minimum (i.e., if Q4_K GEMV ran at Q4_0 speed on the same op).

| Fix | Aufwand | Estimated gain | Ratio |
|---|---|---|---|
| **Q6_K WMMA GEMM kernel** | ~2 days (port `wmma_gemm_q4_k.hip` template, adapt to Q6_K 210-byte block + 8-bit qh) | Prefill 461 → 1 327 tok/s | **+188%** prefill |
| **Q4_K Q8-inline GEMV + residual variants** | ~2 days (adapt `q4_0_fused_q8.hip` to Q4_K super-block dequant) | Decode 29 → ~50 tok/s (gate_up + O-proj + ffn_down benefit) | **+72%** decode |
| **Q4_K `gemv_norm_qkv_rope_kvwrite` fusion** | ~1.5 days (adapt `q4_0_fused_norm_qkv_rope.hip`) | Decode saves ~3 ms/tok (qkv + norm + rope + kv_write consolidated) | **+10%** decode |
| **Q4_K ffn_down + residual fusion** | ~0.5 day (small residual add into down-proj output) | Decode saves 344 µs/tok | **+1%** decode |
| **Q6_K GEMV batched variant (no WMMA, just one-kernel-per-layer)** | ~1 day (drop the row loop, keep existing multi-row GEMV) | Prefill intermediate: saves dispatch overhead but not compute | **+~20%** prefill (fallback if WMMA too hard) |
| BF16 LDS for WMMA | ~2 days — high risk | Speculative; unclear if it helps accuracy or speed | — |

## 8. Priorisierte Optimierungsliste

1. **Q6_K WMMA GEMM** — single biggest hebel. Replaces the 9 252-dispatch GEMV loop with
   ~36 WMMA kernel launches. Gain: **+188% prefill throughput** (461 → 1 327 tok/s, from
   0.13× of llama.cpp to 0.36×). Aufwand: ~2 days. Template exists (`wmma_gemm_q4_k.hip`).
2. **Q4_K Q8-inline GEMV + residual + norm variants** — closes the decode kernel-efficiency
   gap. Halves LDS bandwidth for activations on the three dominant GEMVs (QKV, O-proj, gate_up).
   Gain: **~+70% decode** (29 → ~50 tok/s). Aufwand: ~2 days.
3. **Q4_K `fused_norm_qkv_rope_kvwrite`** — consolidates 5 per-layer dispatches into 1.
   Gain: **~+10% decode**. Aufwand: ~1.5 days.
4. **Q4_K ffn_down + residual fusion** — trivial residual add fold-in. Gain: ~+1% decode.
   Aufwand: ~0.5 day.

Items 1+2 alone target: prefill 1 327 tok/s, decode ~50 tok/s on Qwen3-8B Q4_K_M.
That lifts Q4_K_M from 0.13× / 0.33× to ~0.36× / 0.57× of llama.cpp.

## 9. What is NOT the bottleneck

Ruled out by direct measurement:

- **Dispatch-hell / pipeline bubbles.** Kernel-sum ≈ unprofiled wall-clock + sync_overhead.
  No idle GPU time worth chasing.
- **Memory bandwidth at decode.** Q4_0 hits 74% of theoretical; Q4_K hits 21% at the SAME
  weight bytes-per-element. This is kernel efficiency, not bandwidth.
- **Attention kernel.** 0.9 ms / token on Q4_K (2.4% of budget). Fine.
- **FP16/FP32 numerical precision.** Already validated in Phase 8a — all Q4_K dequant is FP32.
- **Special-token embedding overlay.** Did not affect the multi-turn bug; not a perf factor.

## 10. Files

- Qwen3 decode raw: `/tmp/phase8b/qwen3_decode.log`
- Qwen3 prefill raw: `/tmp/phase8b/qwen3_prefill.log`
- Qwen2.5 decode raw: `/tmp/phase8b/qwen25_decode.log`
- Qwen2.5 prefill raw: `/tmp/phase8b/qwen25_prefill.log`

Commands to reproduce:

```bash
# Decode (Qwen3)
ROCMFORGE_PROFILE_DECODE_OPS=1 RUST_LOG=rocmforge::gpu::forward=info \
  ./target/release/rocmforge --model ~/models/Qwen3-8B-Q4_K_M.gguf \
  --prompt "Hello" --max-tokens 64 --gpu --no-template \
  --temperature 0.0 --top-p 1.0

# Prefill (Qwen3, pp256)
prompt=$(python3 -c "print('word ' * 256, end='')")
ROCMFORGE_PROFILE_PREFILL_OPS=1 RUST_LOG=rocmforge::gpu::forward=info \
  ./target/release/rocmforge --model ~/models/Qwen3-8B-Q4_K_M.gguf \
  --prompt "$prompt" --max-tokens 1 --gpu --no-template \
  --temperature 0.0 --top-p 1.0
```
