# ROCmForge v1.0

GPU-accelerated LLM inference engine for AMD RDNA 4 (RX 9070 XT, gfx1201). Written in Rust with hand-tuned HIP/C++ matrix-core kernels for GGUF Q4_K_M models, optimised for single-batch decode throughput on consumer GPUs.

## Acknowledgement

This project builds on the foundational work of [oldnordic](https://github.com/oldnordic/ROCmForge). Without his original ROCmForge implementation — model loader, CPU inference path, GGUF parser, and overall architecture — none of the v1.0 work (matrix-core optimisations, multi-model chat, integer-WMMA prefill) would have been possible. Thank you for making this project a reality.

---

## Performance (v1.0.0)

Qwen3-8B Q4_K_M on AMD Radeon RX 9070 XT (gfx1201, RDNA 4), ROCm 7.2:

| Engine | Decode tok/s | Prefill tok/s (real prompt 542 tok) | Long-prompt stability (pp ≥ 2048) |
|---|---:|---:|:---|
| **ROCmForge v1.0** | **96.2** | ~1260 | limited by 1024-token max_seq plan |
| llama.cpp ROCm (build `23b8cc4`) | 87.5 | ~3700 | regression at pp=2048, no data pp=4096+ |
| llama.cpp Vulkan (build `23b8cc4`) | 114.2 | ~4300 | stable through pp=16384 |

ROCmForge **leads ROCm-backend** for decode (+10 %) on this hardware. **Vulkan is the fastest backend overall** today and remains the recommended choice if absolute throughput is the only criterion. ROCmForge's purpose is research and learning around RDNA 4 LLM inference, with measured competitive decode against the ROCm reference path.

Detailed triple-benchmark: [`results/phase3_vulkan_vs_rocm_benchmark.md`](results/phase3_vulkan_vs_rocm_benchmark.md).

## Quick-Start

### Prerequisites

- Rust 1.75+ (`rustup` recommended)
- ROCm 7.2+ with `hipcc` and `rocm-smi`
- AMD GPU: RDNA 4 (RX 9070 XT validated, gfx1201). Other RDNA generations are not actively tested in this release.
- Linux, ≥16 GB RAM, ≥16 GB VRAM for 8B models
- See [`INSTALL.md`](INSTALL.md) for distribution-specific notes (Arch / Ubuntu / Fedora).

### Build

```bash
git clone https://github.com/maeddesg/ROCmForge.git
cd ROCmForge
cargo build --release --features "v1 gpu" --bin rocmforge-v1
```

### Download a model

Recommended: Qwen3-8B Q4_K_M (~4.7 GB, fully validated):

```bash
mkdir -p ~/models
hf download Qwen/Qwen3-8B-GGUF Qwen3-8B-Q4_K_M.gguf --local-dir ~/models
```

### Run

```bash
# Single prompt with greedy decode + Integer-WMMA prefill:
ROCMFORGE_PREFILL_MMQ=1 ./target/release/rocmforge-v1 \
  --model ~/models/Qwen3-8B-Q4_K_M.gguf \
  --prompt "Explain what a mutex is."

# Interactive chat REPL (KV-cache resets between turns in v1):
ROCMFORGE_PREFILL_MMQ=1 ./target/release/rocmforge-v1 \
  --model ~/models/Qwen3-8B-Q4_K_M.gguf \
  --interactive

# 15-prompt benchmark suite + Markdown report:
ROCMFORGE_PREFILL_MMQ=1 ./target/release/rocmforge-v1 \
  --model ~/models/Qwen3-8B-Q4_K_M.gguf \
  --inference-test
```

## Features

- **Token-by-token streaming** in single-prompt and interactive modes.
- **`<think>`-tag filtering** for Qwen3 reasoning models (on by default; `--show-think` displays them inline).
- **FP8 KV-cache** (`ROCMFORGE_KV_FP8=1`) — 4× context length at the same VRAM, speed-neutral.
- **Integer-WMMA Q4_K prefill** on gfx1201 — `wmma_i32_16x16x16_iu8_w32_gfx12` intrinsic instead of FP16-WMMA. Ports llama.cpp's MMQ path; **+28.7 % E2E prefill** on the 15-prompt suite (583 → 751 tok/s aggregate). Opt-in via `ROCMFORGE_PREFILL_MMQ=1`.
- **HIP-Graph capture+replay** for the decode loop, dispatched automatically by the self-tuning runtime.
- **Self-tuning runtime** — UCB1 multi-armed bandit chooses between MMVQ and standard GEMV per layer position; converges within ~15 prompts.
- **Quality monitor** — z-score drift detection on decode-logit statistics, calibrated against a reference run.
- **Model introspection** — SNR-risk score over embedding-quantisation noise, critical-token enumeration, automatic sampling override (`repeat_penalty=1.1`) when SNR < 2.0 to avoid repetition loops on low-SNR models.
- **Chat-template disambiguation** for 7 variants (Qwen2/Qwen3/Llama-2/Llama-3/MistralV3/Gemma/Generic) resolved at load time via `(vocab_size, bos_id)`. The GGUF `general.architecture = "llama"` field alone is ambiguous (shared by Llama-2, Llama-3, Mistral, DeepSeek-distill).

## CLI reference

```
Usage: rocmforge-v1 --model <path.gguf> [command]

Commands:
  --list-tensors                     GGUF tensor inventory (CPU only)
  --prompt <text>                    Single-prompt generation
  --inference-test                   Run 15-prompt validation suite
  --interactive                      REPL chat loop

Options:
  --max-tokens <N>                   Generation cap (default 256)
  --suite <path>                     Suite JSON for --inference-test
  --output <path>                    Report file for --inference-test
  --show-introspection               Print ModelProfile summary
  --show-quality                     Calibrate + print Quality-Monitor report
  --show-tuning                      Attach self-tuning runtime + print Bandit
  --show-think                       Show <think> tags instead of filtering
  --show-all                         Equivalent to --show-introspection
                                     --show-quality --show-tuning
```

Full CLI guide: [`docs/cli-guide.md`](docs/cli-guide.md).

## Environment flags

| Flag | Effect |
|---|---|
| `ROCMFORGE_PREFILL_MMQ=1` | Integer-WMMA Q4_K prefill (**recommended**, +28.7 % E2E) |
| `ROCMFORGE_PREFILL_MMQ_Q6K=1` | Use integer-MMQ for Q6_K too. Currently ~9 % slower than the FP16-WMMA Q6_K path on RDNA 4; opt-in for future LDS-staging optimisation work |
| `ROCMFORGE_PREFILL_MMQ_1W=1` | Diagnostic: 1-warp variant of the MMQ kernel instead of the 4-warp default (for A/B comparison) |
| `ROCMFORGE_KV_FP8=1` | FP8 KV-cache (4× context length, speed-neutral) |

## Supported models

| Model | Status | Decode tok/s | Notes |
|---|---|---:|---|
| Qwen3-8B Q4_K_M | ✅ Full | 96.2 | 15/15 coherent, recommended |
| Qwen3-8B Q4_K_M (FP8 KV) | ✅ Full | ~96 | with `ROCMFORGE_KV_FP8=1` |
| Llama-3.1-8B-Instruct Q4_K_M | ⚠ Partial | ~60 | Loads + decodes, instruction-following degraded (see Limitations §2) |
| DeepSeek-R1-Distill-Llama-8B Q4_K_M | ⚠ Partial | ~60 | Same SNR class as Llama-3.1 |
| Mistral-7B-Instruct-v0.3 Q4_K_M | ⚠ Tokenizer | ~60 | Loads with chat-template fix; semantically wrong without SentencePiece tokenizer |
| Gemma-3 / "Gemma-4-E4B" Q4_K_M | ❌ Architecture | — | Zone-A arena overflow + unsupported tensor roles (PLE, hybrid attention) |
| Qwen2.5 (any size) | ❌ Quant format | — | Embedding table in Q5_0 (not supported); Attention-bias tensors not yet wired into the graph builder |

Compatibility matrix details: [`results/phase3_multi_model_compatibility.md`](results/phase3_multi_model_compatibility.md).

## Supported quantisation formats

| Format | Decode GEMV | Prefill GEMM | Status |
|---|:---:|:---:|---|
| Q4_K | ✅ MMVQ + standard | ✅ FP16-WMMA + Integer-MMQ | recommended |
| Q6_K | ✅ MMVQ | ✅ FP16-WMMA (MMQ opt-in, ~9 % slower) | good |
| Q8_0 | ✅ standard | ✅ FP16-WMMA | supported |
| Q4_0 | ✅ standard | ✅ FP16-WMMA | supported |
| Q5_0 | ❌ | ❌ | **not supported** |

## Hardware requirements

- **GPU:** AMD RDNA 4 (RX 9070 XT validated, gfx1201). RDNA 3 (gfx1100) referenced in code but not actively tested in v1.0.
- **VRAM:** ≥16 GB for 8B models (4.7 GB model + KV cache + workspace).
- **CPU:** x86_64 with AVX2.
- **RAM:** ≥16 GB.
- **OS:** Linux (Arch Linux, CachyOS validated).
- **ROCm:** 7.2+.

## Known limitations

1. **Max context: 1024 tokens** in the default pipeline plan. The attention score-LDS budget supports up to ~12 000 tokens (48 KiB cap, `seq_len × 4 B`), but the Phase-1 buffer planner is conservatively capped at `max_seq = 1024`. Longer prompts must extend that plan; >12 k tokens would require Flash-Attention (v1.1 scope).
2. **Llama-3.1-8B Q4_K_M instruction-following is degraded.** The model decodes fluent English but does not follow single-turn instructions reliably. Seven root-cause hypotheses were ruled out (embedding SNR, RoPE freq, GQA heads, tied weights, BOS injection, chat-template token splitting, RoPE-NTK ramp). True cause is open. Qwen3-8B Q4_K_M on the same infrastructure works fully — the bug is Llama-3.1-specific. Details: [`results/phase3_llama31_validation.md`](results/phase3_llama31_validation.md).
3. **Dense models only** — no MoE (Mixtral, DeepSeek-V3, Qwen3-MoE). Routing logic is not implemented.
4. **No SentencePiece tokenizer.** Mistral and Llama-2 families use the `▁` whitespace marker; ROCmForge implements only GPT-2 / Llama-3 BPE (`Ġ` marker). Mistral loads and decodes but is semantically wrong on real content.
5. **No sliding-window attention.** Mistral and Gemma-3 local layers need it; not implemented.
6. **First ~15 prompts have higher latency** while the self-tuning UCB1 bandit explores kernel variants. Subsequent prompts use the converged selection.
7. **Multi-turn REPL resets the KV-cache between turns** — each turn is treated as a new session. This is a Phase-1 design choice; v1.1 may persist the cache.
8. **Q5_0 not supported** (neither dequant nor GEMM/GEMV). Affects most small models including the Qwen2.5 family.
9. **Full-decode HIP graph disabled on RDNA 4** due to a device-pointer stale-read bug in complex graphs ([`hip_graph_device_pointer_bug.md`](hip_graph_device_pointer_bug.md)). Tail-only graph (lm_head + argmax) remains active.

## Architecture

ROCmForge is organised around six pillars (see [`docs/v1.0/architecture_v1.2.0-draft.md`](docs/v1.0/architecture_v1.2.0-draft.md) for details):

1. **Model Introspection** — at load time: SNR-risk over embedding quantisation noise, critical-token identification, architecture detection.
2. **Computation Graph + Fusion** — IR-based forward pass with Gate+Up+SwiGLU fusion and residual folding.
3. **Dequant IR** — codegen path for Q4_K/Q6_K kernels specific to gfx1201.
4. **Self-Tuning Runtime** — UCB1 bandit between MMVQ and standard GEMV per layer position; HIP-graph capture for stabilised decode loop.
5. **Quality Monitor** — z-score drift detection on logit statistics, calibrated against a reference run.
6. **Safety & Debug** — VALU-only parity path without WMMA for numerical correctness validation.

## Documentation

- [`INSTALL.md`](INSTALL.md) — installation notes per distribution
- [`docs/cli-guide.md`](docs/cli-guide.md) — full CLI reference
- [`docs/v1.0/architecture_v1.2.0-draft.md`](docs/v1.0/architecture_v1.2.0-draft.md) — architecture overview
- [`CHANGELOG.md`](CHANGELOG.md) — release notes
- [`results/`](results/) — per-phase performance reports

## License

MIT.
