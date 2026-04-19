# ROCmForge

GPU-accelerated LLM inference engine for AMD GPUs, written in Rust
with hand-tuned HIP/C++ WMMA (Wave Matrix Multiply-Accumulate) kernels
for RDNA 4.

## Acknowledgement

This project builds on the foundational work of
[oldnordic](https://github.com/oldnordic/ROCmForge). Without his
original ROCmForge implementation — the model loader, the CPU
inference path, the GGUF parser, and the overall architecture — none
of the WMMA matrix-core optimisations, the multi-model support, or
the interactive chat CLI would have been possible. Thank you for
making this project a reality.

---

First open-source project to bypass hipBLAS/Tensile on gfx1201 with
direct matrix-core utilisation. Supports Qwen2.5 (Q4_0), Qwen3, and
Llama-3.1 (Q4_K_M) GGUFs with interactive chat, streaming output, and
speculative decoding (opt-in).

- [Install](INSTALL.md)
- [CLI guide](docs/cli-guide.md)
- [Changelog](CHANGELOG.md)

## Features

- **WMMA matrix-core acceleration** — hand-written RDNA 4 WMMA kernels
  for prefill GEMM (Q4_0, Q4_1, Q4_K) and FlashAttention (online softmax,
  GQA, causal masking). Bypasses hipBLAS/Tensile which lacks gfx1201
  matrix-core support in ROCm 7.2.
- **Q4_K_M quantisation support** — industry-standard K-quant format
  with hierarchical scale/min unpacking, super-block-aware nibble
  layout, and a fused Q4_K gate+up+SwiGLU decode kernel.
- **Multi-model, multi-architecture** — Qwen2.5, Qwen3, and Llama-3.1
  load from the same binary. GQA ratio, RoPE base, rope_freqs scaling,
  per-head Q/K RMSNorm, norm epsilon, and chat templates are all
  auto-configured from GGUF metadata.
- **Llama-3 tokeniser** — GPT-4-style BPE pre-tokeniser with 7/7 token
  parity against llama.cpp on Meta-Llama-3.1-8B-Instruct.
- **Interactive chat CLI** — multi-turn conversations with streaming
  output, per-architecture chat templates, session statistics, slash
  commands, Ctrl+C to interrupt generation.
- **Speculative decoding** (opt-in) — 0.5B draft + 7B target with
  configurable depth. Currently slower than greedy on most prompt
  types (acceptance rate ~54 % median, break-even needs ~80 %).
- **Arbitrary prompt lengths** — WMMA kernels handle any `seq_len ≥ 1`
  via automatic padding to 64-token boundaries.
- **Structured logging** — `RUST_LOG=debug` shows kernel dispatch
  decisions, `RUST_LOG=trace` shows per-layer timing.
- **ROCm upgrade validation** — built-in benchmark harness with an
  automatic diff tool for safe ROCm version upgrades.
- **CPU fallback path** — pure-Rust CPU backend with AVX-512 VNNI for
  debugging and zero-GPU environments. Not intended for production.

## Performance

Measured on AMD Radeon RX 9070 XT (gfx1201, RDNA 4), ROCm 7.2.2. All
numbers are median of 3 runs.

### Qwen2.5-7B Q4_0 (v0.1.0 reference model)

| Prompt length (pp) | ROCmForge | llama.cpp ROCm | Ratio |
|-------------------:|----------:|---------------:|------:|
| pp64               |       788 |          2,912 | 0.27× |
| pp128              |     1,131 |          3,966 | 0.29× |
| pp256              | **1,482** |          4,951 | 0.30× |
| pp512              |     1,693 |          5,158 | 0.33× |
| Decode 128 (tok/s) |       102 |            121 | 0.84× |

### Qwen3-8B Q4_K_M (new in v0.2.0)

| Prompt length (pp) | ROCmForge | llama.cpp ROCm | Ratio |
|-------------------:|----------:|---------------:|------:|
| pp64               |       388 |          2,000 | 0.19× |
| pp128              |       447 |          2,657 | 0.17× |
| pp256              |   **470** |          3,661 | 0.13× |
| pp512              |       472 |          3,756 | 0.13× |
| Decode 128 (tok/s) |      29.9 |             87 | 0.34× |

### Llama-3.1-8B-Instruct Q4_K_M (new in v0.2.0)

| Prompt length (pp) | ROCmForge | llama.cpp ROCm | Ratio |
|-------------------:|----------:|---------------:|------:|
| pp64               |       400 |          2,198 | 0.18× |
| pp128              |       457 |          2,975 | 0.15× |
| pp256              |   **475** |          3,925 | 0.12× |
| pp512              |       471 |          3,922 | 0.12× |
| Decode 128 (tok/s) |      30.5 |             93 | 0.33× |

**Reading the three tables.** On the v0.1.0 reference Q4_0 model
ROCmForge is within 6 % of llama.cpp on realistic prompt lengths
(15-prompt benchmark, see below). Q4_K_M support is functional across
both Qwen3 and Llama-3.1 — all three models emit coherent greedy
decode — but the Q4_K GEMV kernel and Q6_K GEMV-loop fallback in the
prefill path are not yet competitive with llama.cpp's batched Q4_K_M
GEMM. See Known Issues below for the roadmap.

### Real-world prompts — Qwen2.5-7B Q4_0 (15 prompts, 19–41 tokens, 128 generated)

| Metric                | ROCmForge | llama.cpp | Ratio |
|-----------------------|----------:|----------:|------:|
| Prefill (tok/s)       |     493.5 |     525.4 | 0.94× |
| Time-to-first-token   |     49 ms |     46 ms | 1.07× |
| Decode (tok/s)        |       102 |       121 | 0.84× |
| Total wall-clock (ms) |     1,303 |     1,103 | 1.18× |

### Optimisation history

| Milestone                       | Prefill pp256 (Q4_0) | Decode (Q4_0) | Models |
|---------------------------------|---------------------:|--------------:|:-------|
| Project start                   |             64 tok/s |      82 tok/s | Qwen2.5 Q4_0 |
| + WMMA GEMM (Phase 2)           |                   92 |           102 | Qwen2.5 Q4_0 |
| + WMMA Attention (Phase 3)      |                  623 |           102 | Qwen2.5 Q4_0 |
| + Dispatch fixes (Phase 4)      |          **1,484**   |           102 | Qwen2.5 Q4_0 |
| + Decode profiling (Phase 6)    |                1,484 |           102 | Qwen2.5 Q4_0 |
| **+ Q4_K_M + multi-model (Phase 7)** |           **1,482** |       **102** | **+ Qwen3 + Llama-3.1 (Q4_K_M)** |

Full analyses:
[`benches/results/phase4_final_analysis.md`](benches/results/phase4_final_analysis.md),
[`benches/results/phase7_final_analysis.md`](benches/results/phase7_final_analysis.md).

### Speculative decoding

Implemented but opt-in. Measured honestly on the 15-prompt benchmark it
is a loss at every prompt in greedy mode: median 72 tok/s at depth=1
vs. 102 tok/s baseline. Break-even needs α ≈ 80 % on the current verify
path; best observed is 78 % (`code_03`). Useful only for workloads with
very high acceptance rates (tightly constrained code, templated output).

**Note:** speculative decoding is currently wired into the one-shot
`rocmforge` entry point via `--draft-model`, but `rocmforge chat`
accepts the flag and ignores it (banner shows standard path). See
[`docs/cli-guide.md`](docs/cli-guide.md) for details.

### CPU backend

Pure-Rust with AVX-512 VNNI Q4_0 GEMV on Zen 4. Functional but not
performance-competitive: Qwen2.5-0.5B ~12 tok/s, Qwen2.5-7B ~0.7 tok/s.
For production inference, use `--gpu`.

## Compatibility

### Tested ✅

| Component     | Version                            | Notes |
|---------------|------------------------------------|-------|
| GPU           | AMD Radeon RX 9070 XT              | gfx1201, RDNA 4, 16 GB VRAM |
| CPU           | AMD Ryzen 9 7945HX                 | Zen 4, AVX-512 (CPU fallback) |
| OS            | CachyOS (Arch Linux-based)         | Kernel 7.x |
| ROCm          | 7.2.1 and 7.2.2                    | Both validated via upgrade diff tool |
| Model         | Qwen2.5-7B-Instruct Q4_0           | GGUF, mixed 25×Q4_0 + 3×Q4_1 for `ffn_down`; 102 tok/s decode |
| Model         | **Qwen3-8B Q4_K_M** (v0.2.0)       | **Q4_K + Q6_K mixed-precision, per-head Q/K RMSNorm, 30 tok/s decode** |
| Model         | **Meta-Llama-3.1-8B-Instruct Q4_K_M** (v0.2.0) | **Q4_K + Q6_K, rope_freqs 128 k-context scaling, 31 tok/s decode** |
| Draft model   | Qwen2.5-0.5B-Instruct Q4_0         | For speculative decoding (opt-in) |
| RAM           | 64 GB DDR5                         | |

### Expected to work (untested) ⚠️

| Component             | Notes |
|-----------------------|-------|
| Other RDNA 4 GPUs     | RX 9070, RX 9060 XT — same gfx1201 arch, should work |
| Arch Linux (vanilla)  | Same ROCm packages as CachyOS |
| Ubuntu 24.04          | ROCm 7.2 officially supported; see [INSTALL.md](INSTALL.md) for notes |
| Fedora 41+ / RHEL 10  | ROCm 7.2 officially supported; see [INSTALL.md](INSTALL.md) for notes |
| Other Qwen / LLaMA sizes | Loader is metadata-driven. 14B hits VRAM on 16 GB. 0.5B tested as draft. |
| Mistral / GLM / Phi / Gemma | Trait table covers these architectures; only Qwen and Llama have been end-to-end validated. |

### Not expected to work ❌

| Component                          | Reason |
|------------------------------------|--------|
| RDNA 3 (gfx1100, RX 7900 XTX, etc.)| Kernels use `__builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12`. RDNA 3 uses a different intrinsic format, a different A/B register layout (8 VGPRs instead of 4), and needs lane duplication (lanes 0–15 → 16–31) that gfx12 does not. Porting is feasible but not attempted. |
| RDNA 2 (gfx1030) and older         | No matrix cores. |
| CDNA (MI100/MI200/MI300)           | Uses MFMA, not WMMA. Different instruction set. |
| Intel / NVIDIA GPUs                | HIP/ROCm backend only. |
| Windows                            | ROCm is Linux-only for consumer GPUs. |
| Q5_K / Q8_0 prefill                | WMMA kernels implemented for Q4_0, Q4_1, and Q4_K only. Q5_K / Q8_0 fall back to scalar / GEMV kernels. |

## Known issues

- **Q4_K_M decode gap:** Qwen3-8B decode at ~30 tok/s and Llama-3.1-8B
  at ~31 tok/s vs. llama.cpp's 87 / 93 tok/s (~0.33×). The dispatch
  log is clean — every GEMV goes through the multi-row vulkan-style
  Q4_K path and the FFN through a fused gate+up+SwiGLU kernel — but
  the per-token kernel work is ~32 ms vs. ~11 ms on llama.cpp. Closing
  this needs fused norm + QKV + RoPE for Q4_K (analogous to the Q4_0
  fastpath) and a Q8-inline activation variant for Q4_K.
- **Q4_K_M prefill gap:** pp256 at ~470 tok/s vs. llama.cpp 3,661–
  3,925. Q4_K goes through the WMMA prefill kernel, but the Q6_K
  layers (V-projection, `ffn_down`) use a GEMV-loop fallback because
  there is no batched Q6_K GEMM kernel yet. Writing one would close
  most of this gap.
- **Q4_0 decode gap:** 102 tok/s vs. llama.cpp 117–121 tok/s (~0.84×),
  unchanged since project start. **Fully profiled in Phase 6.** The
  9.76 ms per-token budget splits: GEMV 77 % (memory-bandwidth-bound
  and CU-saturated), launch overhead 8–13 % (255 kernel launches per
  token — exactly the size of the gap to llama.cpp), attention 6 %,
  everything else 9 %. A fused RMSNorm + Gate + Up + SwiGLU kernel
  exists but has a latent state-corruption bug that appears from
  token 2+ when routed into the active decode path — see
  [`docs/known_issues/fused_norm_gate_up_bug.md`](docs/known_issues/fused_norm_gate_up_bug.md).
  Full analysis: [`profiling/results/decode_profiling_analysis.md`](profiling/results/decode_profiling_analysis.md).
- **Synthetic prefill gap at long sequences:** at pp256+ ROCmForge is
  ~3× slower than llama.cpp. The GEMM path is fully WMMA after Phase 4;
  the gap is unfused norm/RoPE/embedding orchestration and FP32↔FP16
  activation shuttling that llama.cpp packs into fewer launches.
- **Speculative decoding is slower than greedy** on most prompt types.
  Acceptance rate ~54 % median, below the ~80 % break-even threshold.
  Works via `--draft-model` in the one-shot CLI; accepted-but-ignored
  in `rocmforge chat`.
- **Chat CLI does not persist KV cache between turns:** each turn
  re-prefills the entire conversation history. Multi-turn TTFT grows
  with history length (roughly linear in total tokens so far).
- **Llama-3.1 multi-turn chat produces degraded output.** Single-turn
  works correctly. Root cause is a position/content-dependent prefill
  divergence for longer prompts (~50+ tokens). Qwen3 multi-turn works
  perfectly. Workaround: use Qwen3 for interactive chat, or use
  Llama-3.1 in single-turn mode with `--no-template`.
  Tracked in [`docs/known_issues/llama3_multiturn_prefill_bug.md`](docs/known_issues/llama3_multiturn_prefill_bug.md).
- **Full-decode HIP graph disabled on RDNA 4:** graph replay of
  device-pointer reads in large graphs (~200+ nodes) returns stale
  values — see [`hip_graph_device_pointer_bug.md`](hip_graph_device_pointer_bug.md).
  Tail-only graph (lm_head + argmax) is still active.
- **rocprofv3 PMC counters hang on gfx1201:** hardware perf-counter
  capture via `rocprofv3 --pmc` is unstable on gfx1201. Use
  `RUST_LOG=trace` + `ROCMFORGE_PROFILE_PREFILL_OPS=1` for profiling.
- **WMMA attention `head_dim == 128` only:** Qwen2.5, LLaMA-2,
  Mistral are covered. Other head dimensions fall back to the scalar
  attention kernel.

## Documentation

- [`INSTALL.md`](INSTALL.md) — installation (Arch tested, Ubuntu / Fedora notes)
- [`docs/cli-guide.md`](docs/cli-guide.md) — CLI reference (chat + one-shot, slash commands, env flags)
- [`CHANGELOG.md`](CHANGELOG.md) — release notes and optimisation history
- [`benches/results/phase4_final_analysis.md`](benches/results/phase4_final_analysis.md) — Phase 4 final benchmark analysis
- [`docs/architecture_notes.md`](docs/architecture_notes.md) — RDNA 4 memory pipelining and WMMA details
- [`docs/spec_decode_milestone_summary.md`](docs/spec_decode_milestone_summary.md) — speculative decoding investigation

## License

MIT. See [`LICENSE`](LICENSE).
