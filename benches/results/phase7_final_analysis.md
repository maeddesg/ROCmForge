# Phase 7 — final analysis

**Date:** 2026-04-19
**Branch:** `main`
**Hardware:** AMD Radeon RX 9070 XT (gfx1201, RDNA 4, 16 GB VRAM) /
AMD Ryzen 9 7945HX (Zen 4) / 64 GB DDR5.
**ROCm:** 7.2.2.

Phase 7 took ROCmForge from a Qwen2.5-only Q4_0 engine to a
multi-model Q4_0 + Q4_K_M engine. This doc summarises the scope, the
ship list, the performance delta, and what's left for v0.2.1.

## 1. What shipped

Three models now decode coherent greedy text from the same binary:

| Model                         | Quant  | Decode on "capital of France" prompt                                |
|-------------------------------|--------|----------------------------------------------------------------------|
| Qwen2.5-7B-Instruct           | Q4_0   | "Paris. It is located in the northern central part"                  |
| Qwen3-8B                      | Q4_K_M | "Paris. The capital of the United States is Washington, D.C. The"    |
| Meta-Llama-3.1-8B-Instruct    | Q4_K_M | "a city of love, a city of love, fashion, art, fashion, …Paris…"     |

Two architectures, two quantisation formats, 28 / 32 / 36 layers,
three different tokenisers, three different chat templates — all
auto-detected from GGUF metadata, no model-name conditionals in
dispatch code.

### New kernels

| Kernel                                       | Purpose                                                          |
|----------------------------------------------|------------------------------------------------------------------|
| `wmma_gemm_q4_k.hip`                         | WMMA prefill GEMM for Q4_K super-blocks (256 elements, 144 B)    |
| `q4_k_gemv.hip` multi-row                    | 8-waves × 4-cols decode GEMV for Q4_K                            |
| `gemv_gate_up_swiglu_q4_k_f32`               | Fused gate + up + SwiGLU for Q4_K FFN decode                     |
| `q6_k_gemv.hip` multi-row                    | 8-waves × 4-cols decode GEMV for Q6_K (replaces 1-warp baseline) |
| `embed_q4_k_token_kernel` / `_batch`         | GPU-native Q4_K token embedding (eliminates CPU fallback)        |
| Five `_scaled` variants of RoPE kernels      | Optional `freq_scale` pointer for Llama-3.1 `rope_freqs`          |
| `rms_norm_batched_on_stream`                 | Stream-aware batched RMSNorm for Qwen3 per-head Q/K norm         |

### New configuration surface

- `ModelConfig::rope_freqs: Option<Vec<f32>>` — populated from
  `rope_freqs.weight` when present (Llama-3.1).
- `ModelConfig::use_qk_norm: bool` — true for Qwen3 dense / MoE.
- `ModelTraits::use_qk_norm` — architecture-level flag.
- `GpuLayerWeights::attn_q_norm` / `attn_k_norm` — optional
  per-head RMSNorm gains (Qwen3 only).
- `GpuModelWeights::rope_freqs` — optional global freq-scale buffer.
- `TensorName::AttnQNorm` / `AttnKNorm` registered for the GGUF
  naming scheme.

### Latent bugs caught and fixed

Three of these were not in the Step-5 scope — they surfaced as root
causes once the obvious blockers were fixed:

1. **`embed_q4_k` had three compounding errors** (signed vs unsigned
   6-bit scale, `+` instead of `-` for the min offset, wrong nibble
   layout). The GPU hybrid embed falls back to this CPU function for
   every non-Q8_0 embedding type, so every Qwen3 / Llama-3.1 decode
   token started from a wrong hidden state. Commit `e11c01f`.
2. **`gpu_dispatch_gemm` did not support Q6_K**. The first Q6_K
   tensor in a Q4_K_M GGUF (V-projection or `ffn_down`) raised
   `UnsupportedWeightType`, which `main.rs` caught and re-ran the
   entire prefill as 256 sequential decode steps. Commit `e122e51`.
3. **Prefill never used the fused gate + up + SwiGLU path for Q4_K**
   because that fastpath was Q4_0-only. Commit `f5c52d3`.

## 2. Performance

All numbers measured on the same machine, ROCm 7.2.2, greedy decode,
`--no-template`, median of 3 runs.

### Synthetic benchmark

| Model                    | pp64 | pp128 | pp256    | pp512 | Decode 128 |
|--------------------------|-----:|------:|---------:|------:|-----------:|
| Qwen2.5-7B Q4_0          |  788 | 1,131 | **1,482**| 1,693 |      102   |
| Qwen3-8B Q4_K_M          |  388 |   447 |   **470**|   472 |       29.9 |
| Llama-3.1-8B Q4_K_M      |  400 |   457 |   **475**|   471 |       30.5 |
| *(llama.cpp ROCm)*       |      |       |          |       |            |
| Qwen2.5-7B Q4_0          | 2,912 | 3,966 | 4,951   | 5,158 |      121   |
| Qwen3-8B Q4_K_M          | 2,000 | 2,657 | 3,661   | 3,756 |       87   |
| Llama-3.1-8B Q4_K_M      | 2,198 | 2,975 | 3,925   | 3,922 |       93   |

### ROCmForge / llama.cpp ratio

| Model                    | pp256 prefill | Decode |
|--------------------------|--------------:|-------:|
| Qwen2.5-7B Q4_0          |        0.30×  |  0.84× |
| Qwen3-8B Q4_K_M          |        0.13×  |  0.34× |
| Llama-3.1-8B Q4_K_M      |        0.12×  |  0.33× |

### Where ROCmForge is competitive

- **Qwen2.5-7B Q4_0 decode: 0.84× llama.cpp.** Unchanged since
  Phase 4. The 1.3 ms gap is fully explained by the 255 kernel
  launches per token × ~5 µs launch overhead (Phase 6 analysis).
- **Qwen2.5-7B Q4_0 real-prompt prefill: 0.94× llama.cpp.** Realistic
  prompt lengths (19–41 tokens) keep the WMMA path saturated.

### Where ROCmForge is not yet competitive

- **Q4_K_M prefill: 0.12–0.13×.** Q4_K goes through the WMMA kernel
  at near-Q4_0 speed, but the Q6_K V-projection and `ffn_down`
  layers in every mixed-precision Q4_K_M GGUF fall back to a GEMV
  loop (one GEMV per input row). A batched Q6_K GEMM kernel would
  close most of this gap.
- **Q4_K_M decode: 0.33–0.34×.** Profile after Step 5d shows
  `gate_up` at 17 ms (down from 33 ms pre-fusion), `qkv` at 9 ms
  (three separate GEMVs — Q, K, V; V is Q6_K), `attn_proj` at 6 ms,
  `ffn_down` at 3 ms, attention / norm / RoPE / KV-write at ~3 ms
  total, for 40.7 ms wall per token. Closing this to the llama.cpp
  11 ms needs a fused norm + QKV + RoPE pass for Q4_K (analogous to
  the Q4_0 fastpath) and a Q8-inline activation variant.

## 3. Project timeline

| Milestone                        | Prefill pp256 (Q4_0) | Decode (Q4_0) | Models supported |
|----------------------------------|---------------------:|--------------:|-------------------|
| Project start                    |              64 tok/s |       82 tok/s | Qwen2.5 Q4_0 |
| + WMMA GEMM (Phase 2)            |                    92 |            102 | Qwen2.5 Q4_0 |
| + WMMA Attention (Phase 3)       |                   623 |            102 | Qwen2.5 Q4_0 |
| + Dispatch fixes (Phase 4)       |              **1,484** |            102 | Qwen2.5 Q4_0 |
| + Decode profiling (Phase 6)     |                 1,484 |            102 | Qwen2.5 Q4_0 |
| **+ Q4_K_M + multi-model (Phase 7)** |         **1,482** |        **102** | **+ Qwen3 + Llama-3.1 (Q4_K_M)** |

Qwen2.5-7B Q4_0 numbers are **within run-to-run variance** of the
Phase 4 / Phase 6 baseline. Phase 7 did not regress the v0.1.0
reference model.

## 4. Roadmap for v0.2.1

Ordered by expected impact on Q4_K_M parity:

1. **Batched Q6_K GEMM kernel.** The Q6_K GEMV-loop fallback in
   `gpu_dispatch_gemm` is the dominant cost in Q4_K_M prefill. A
   proper M-row-batched kernel (either WMMA-based or in the vulkan
   multi-row style) would likely lift pp256 from 470 to ~2,000 tok/s.
2. **Fused norm + QKV + RoPE + KV-write for Q4_K** (decode). Q4_0
   already has `gpu_dispatch_fused_norm_qkv_rope_kvwrite_on_stream`;
   Q4_K doesn't. This would shave ~3 launches × 32–36 layers = ~100
   dispatches per decode token.
3. **Q8-inline activation for Q4_K** (decode). Q4_0 has four autotuned
   variants of `gemv_gate_up_swiglu_q4_0_f32_q8_inline_*`; Q4_K has
   none. Q8-inline quantises the activation vector once per FFN and
   reuses it across both gate and up dot products, halving the LDS
   traffic.
4. **WMMA Q6_K prefill kernel.** Closing the last bit of the prefill
   gap after item 1 lands.

## 5. Honest assessment

ROCmForge v0.2.0 is the first open-source engine to run Q4_K_M on
RDNA 4 matrix cores without hipBLAS — the WMMA Q4_K kernel is
hand-written, the dispatch is clean (zero scalar fallbacks for Q4_K
tensors, zero `UnsupportedWeightType` aborts), and the output is
coherent. But the end-to-end throughput on Q4_K_M is 0.12× of
llama.cpp's prefill and 0.33× of its decode.

That gap has a name: fused decode kernels for Q4_K that mirror the
Q4_0 fastpath, and a batched Q6_K GEMM that doesn't exist yet. Both
are tractable — v0.2.1 territory, not research problems.
