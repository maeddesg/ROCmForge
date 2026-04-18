rocmforge - LLM inference on AMD GPUs (HIP) with a CPU fallback path.

## Current status

- End-to-end GPU inference for Qwen2.5, LLaMA, Mistral, and GLM GGUF models
- Pure HIP/ROCm GPU backend — no CUDA, no cross-vendor abstraction
- **Developed and tested exclusively on RDNA 4** (RX 9070 XT, gfx1201) with ROCm 7.2.2 on Arch Linux. See [Supported hardware](#supported-hardware).
- Custom WMMA GEMM (Q4_0) and WMMA FlashAttention kernels using gfx12 matrix cores
- Padding support for arbitrary `seq_len ≥ 1` (Phase 3.2) — all prompt lengths engage WMMA
- Speculative decoding (draft + target model, batched verify, tiled GEMV, batched lm_head, adaptive depth)
- FP16 KV cache
- Fused kernels: RMSNorm+QKV+RoPE+KV-Write, RMSNorm+Gate+Up+SiLU (decode path)

## Supported hardware

| Architecture | Status |
|--------------|--------|
| RDNA 4 (gfx1201, e.g. RX 9070 XT) | **Developed and tested.** |
| RDNA 3 (gfx1100, e.g. RX 7900 XT/XTX) | **Untested and not expected to work without kernel changes.** The WMMA kernels use `__builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12`, which is gfx12-specific. RDNA 3 has a different intrinsic (no `_gfx12` suffix), a different A/B matrix register layout (8 VGPRs instead of 4), and requires lane duplication (lanes 0–15 → 16–31) that gfx12 does not. Porting is possible but not attempted here. |
| RDNA 2 (gfx1030, e.g. RX 6900 XT) | Untested. No matrix cores. |
| CDNA (gfx908/90a/940, MI100/MI200/MI300) | Untested. Different matrix-core ISA. |

## Performance (RX 9070 XT / gfx1201 / ROCm 7.2.2 / Qwen2.5-7B Q4_0)

### Head-to-head vs. llama.cpp ROCm (same host, same model)

15-prompt real-world benchmark (5 code / 5 chat / 5 prose, 19–41 prompt tokens, 128 decoded tokens, greedy, 3 runs median — see [`benches/results/phase4_final_analysis.md`](benches/results/phase4_final_analysis.md)):

| Metric | ROCmForge | llama.cpp ROCm | Ratio |
|--------|----------:|---------------:|------:|
| Decode (tok/s) | 102 | 121 | 0.84× |
| Prefill real prompts (tok/s, median) | 493 | 525 | 0.94× |
| TTFT real prompts (ms, median) | 49 | 46 | 1.07× |
| Total wall-clock, 24 prompt + 128 gen (ms, median) | 1,303 | 1,103 | 1.18× |

Synthetic `llama-bench` at aligned pp sizes (`build 408225b`):

| Prompt length | ROCmForge (tok/s) | llama.cpp ROCm (tok/s) | Ratio |
|--------------:|------------------:|-----------------------:|------:|
| pp64  |   801 | 2,912 | 0.28× |
| pp128 | 1,131 | 3,966 | 0.29× |
| pp256 | 1,484 | 4,951 | 0.30× |
| pp512 | 1,693 | 5,158 | 0.33× |

**Reading the two tables.** After the Phase 4 dispatch fixes (routing all `ffn_down` layers through WMMA, adding a Q4_1 WMMA kernel for mixed-quant files, fusing gate+up into one launch) the real 15-prompt gap to llama.cpp narrowed to **0.94× on prefill and 1.18× on wall-clock**. On synthetic pp-sweeps ROCmForge now scales with `seq_len` (prior to Phase 4 it plateaued around 620 tok/s regardless of pp); the remaining 3× gap is unfused norm/RoPE/embedding overhead that llama.cpp fuses into fewer kernel launches.

### ROCmForge prefill throughput (tok/s)

Synthetic (`"word "`-repeated prompts, exact pp size):

| Prompt length | Phase 4.4 (WMMA Q4_0+Q4_1+fused G/U) |
|--------------:|-------------------------------------:|
| pp64  |   801 |
| pp128 | 1,131 |
| pp256 | **1,484** |
| pp512 | 1,693 |

Real prompts (15-prompt benchmark, median, 19–41 tokens):

| Start (all opt off) | Phase 3.2 (WMMA ≥1) | **Phase 4.4 (all WMMA)** | vs. Start |
|--------------------:|--------------------:|-------------------------:|----------:|
|           60.6 tok/s |           356 tok/s |              **493 tok/s** |     8.1× |

### ROCmForge decode throughput (tok/s)

| Model | ROCmForge | llama.cpp ROCm |
|-------|----------:|---------------:|
| Qwen2.5-0.5B Q4_0 | 222 | 358 |
| Qwen2.5-7B Q4_0  | 102 | 117–121 |

### Prefill kernel detail

Prefill uses custom WMMA GEMM and WMMA FlashAttention kernels on gfx1201 matrix cores (`__builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12`). hipBLAS/Tensile on ROCm 7.2 does not use matrix cores on gfx1201 — ROCmForge bypasses this with hand-written WMMA kernels.

After Phase 4, every Q4_0 and Q4_1 prefill GEMM dispatches to WMMA:

- `wmma_gemm_q4_0` — Q/K/V/O, `ffn_down` for 25/28 layers
- `wmma_gemm_q4_0_fused_gate_up` — one launch for both gate and up (Phase 4 Step 3)
- `wmma_gemm_q4_1` — mixed-precision `ffn_down` for the remaining 3 layers (Phase 4 Step 4)

Zero scalar fallbacks on a standard Qwen2.5-7B Q4_0 model. The attention kernel implements FlashAttention-style online softmax, GQA (28 Q heads / 4 KV heads), and causal masking with zero-work elimination above the diagonal. `GpuPrefillScratch` pads all activation buffers up to a multiple of 64 rows and zero-initialises them once, so every `seq_len ≥ 1` engages the WMMA path. Full optimisation arc and per-op breakdown: [`benches/results/phase4_final_analysis.md`](benches/results/phase4_final_analysis.md) and [`profiling/results/prefill_overhead_analysis.md`](profiling/results/prefill_overhead_analysis.md).

Flags (all opt-out, defaults enabled):

- `ROCMFORGE_DISABLE_WMMA_ATTENTION=1` — scalar per-head attention kernel.
- `ROCMFORGE_DISABLE_WMMA_PREFILL=1` — fall through to hipBLAS GEMM.
- `ROCMFORGE_DISABLE_HIPBLAS_PREFILL=1` — original custom-GEMV dispatch.
- `ROCMFORGE_PROFILE_PREFILL_OPS=1` — per-op timing with `hipDeviceSynchronize`, emits one `tracing::info!` event per layer. Adds ~30 ms sync overhead, off by default.

### Speculative decoding (0.5B draft + 7B target)

Implemented: depth 1–8, batched verify, greedy token comparison, adaptive depth (EMA-based), tiled GEMV, batched lm_head. **Measured honestly on the 15-prompt real benchmark, speculative decoding is a loss at every prompt** — median 72 tok/s at depth=1 vs. 102 tok/s baseline. The per-step verify cost outweighs the amortised tokens per step at observed acceptance rates:

| Class | Median α (depth=1) | Median spec tok/s | Baseline |
|-------|-------------------:|------------------:|---------:|
| Code  | 69 %               | 77                | 102 |
| Chat  | 58 %               | 74                | 102 |
| Prose | 49 %               | 70                | 102 |

Break-even needs α ≈ 80 % or more on the current verify path. The best observed single prompt (`code_03`, α = 78 %) still only reached 80 tok/s. Speculative decode remains exposed via `--draft-model` for workloads with very high acceptance (tightly constrained code, repeated templates) but is not a default-on win. Rejection-sampling and multi-stream verify are the identified next steps; see [`docs/spec_decode_milestone_summary.md`](docs/spec_decode_milestone_summary.md).

### CPU decode throughput (Ryzen 9 7945HX, AVX-512 VNNI)

| Model             | tok/s |
|-------------------|------:|
| Qwen2.5-0.5B Q4_0 |  12.1 |
| Qwen2.5-7B Q4_0   |   0.7 |

The CPU path is functional and has an AVX-512 VNNI Q4_0 GEMV kernel on Zen 4+, but Rayon fork-join overhead per GEMV call and scalar attention/norm/SiLU dominate the forward pass. For production inference, use `--gpu`. See [`docs/architecture_notes.md`](docs/architecture_notes.md) for the empirical breakdown.

## Limitations

- **GPU support is RDNA 4 only.** Kernels use gfx12-specific WMMA intrinsics. RDNA 3 needs different intrinsics, register layout, and lane duplication; untested. Other AMD architectures (RDNA 2, CDNA) have no matrix cores or a different matrix-core ISA; untested.
- **Quantisation:** WMMA acceleration is implemented for Q4_0 and Q4_1 weights. Other formats (Q4_K, Q5_K, Q6_K, Q8_0) fall back to scalar / GEMV kernels.
- **Synthetic prefill gap remains ~3×:** at pp=256 ROCmForge is 1,484 tok/s vs. llama.cpp 4,951 tok/s. After Phase 4 the GEMM path is fully on WMMA; the remaining gap is unfused norm/RoPE/embedding orchestration (~14 ms/pp256 combined) that llama.cpp packs into fewer launches. Real short-prompt performance is on par (0.94× on the 15-prompt benchmark).
- **Decode gap:** 102 tok/s vs. llama.cpp 117–121 tok/s (≈ 0.84×), unchanged from project start. Not profiled yet — the biggest remaining user-visible lever.
- **Full-decode HIP graph disabled on RDNA 4:** graph replay of kernels reading device pointers returns stale values in complex graphs (~200+ nodes). See [`hip_graph_device_pointer_bug.md`](hip_graph_device_pointer_bug.md). Tail-only graph (lm_head + argmax) is still active.
- **WMMA attention `head_dim == 128` only:** Qwen2.5 / LLaMA-2 / Mistral are covered; other head dimensions fall back to the scalar attention kernel. Arbitrary `seq_len ≥ 1` is supported via zero-padding in `GpuPrefillScratch`.

## Requirements

- Rust 1.81+
- ROCm/HIP toolkit (tested on ROCm 7.2.1 and 7.2.2)
- ROCm runtime libraries (`libamdhip64.so.7`)
- A GGUF model file

## Build

```bash
cargo build --release                  # CPU-only
cargo build --release --features gpu   # With GPU support
```

## Run

GPU inference:

```bash
./target/release/rocmforge \
  --model /path/to/model.gguf \
  --prompt "Hello" \
  --gpu
```

Speculative decoding (two models):

```bash
./target/release/rocmforge \
  --model /path/to/target-7b.gguf \
  --draft-model /path/to/draft-0.5b.gguf \
  --spec-depth 3 \
  --prompt "Hello" \
  --gpu
```

CPU fallback:

```bash
./target/release/rocmforge \
  --model /path/to/model.gguf \
  --prompt "Hello"
```

### CLI options

| Option | Description |
|---|---|
| `--model <path>` | GGUF model path |
| `--prompt <text>` | Prompt text |
| `--gpu` | Use GPU backend |
| `--max-tokens N` | Max generated tokens (default: 256) |
| `--temperature F` | Sampling temperature (default: 1.0) |
| `--top-p F` | Nucleus sampling threshold (default: 0.9) |
| `--no-template` | Disable chat template |
| `--draft-model <path>` | Draft model for speculative decoding (GPU only) |
| `--spec-depth N` | Speculation depth (default: 5) |
| `--list-tensors` | Print tensors and exit |
| `--debug` | Print debug logits info |

## Runtime environment flags

| Flag | Effect |
|------|--------|
| `ROCMFORGE_ENABLE_DECODE_GRAPH=1` | Enable HIP graph capture + replay for decode |
| `ROCMFORGE_DISABLE_DECODE_GRAPH=1` | Force-disable graph (safety override) |
| `ROCMFORGE_GPU_SAFE_MODE=1` | Disable all experimental fast paths |
| `ROCMFORGE_ENABLE_EXPERIMENTAL_Q8_ACTIVATION_FASTPATH=1` | Specialised Q8 decode activation path |
| `ROCMFORGE_ENABLE_EXPERIMENTAL_FFN_FASTPATH=1` | Fused FFN kernels |
| `ROCMFORGE_SPEC_DEBUG=1` | Print speculative decode draft/target token comparison |
| `ROCMFORGE_PROFILE_SPEC_STEP=1` | HIP-event timing for spec-step cost breakdown |
| `ROCMFORGE_PROFILE_VERIFY_BREAKDOWN=1` | Sub-phase timing within verify layers (requires `PROFILE_SPEC_STEP`) |
| `ROCMFORGE_DISABLE_TILED_GEMV=1` | Disable tiled batched GEMV for large FFN projections |
| `ROCMFORGE_DISABLE_BATCHED_LM_HEAD=1` | Disable batched verify lm_head, fall back to sequential per-position dispatch |
| `ROCMFORGE_DISABLE_AVX512=1` | Force the Q4_0 × Q8_0 CPU GEMV back to the AVX2 path (auto-enabled on Zen 4+) |
| `ROCMFORGE_DISABLE_WMMA_PREFILL=1` | Disable the WMMA Q4_0 prefill GEMM, fall through to hipBLAS |
| `ROCMFORGE_DISABLE_WMMA_ATTENTION=1` | Disable the WMMA GQA + causal prefill attention, fall back to scalar kernel |

## Documentation

- Manual: [MANUAL.md](MANUAL.md)
- Speculative decoding milestone summary: [docs/spec_decode_milestone_summary.md](docs/spec_decode_milestone_summary.md)
- Architecture notes (RDNA 4 memory-pipelining findings, WMMA details): [docs/architecture_notes.md](docs/architecture_notes.md)
- Benchmark progression: [profiling/benchmark_progression.md](profiling/benchmark_progression.md)
- HIP graph bug report: [hip_graph_device_pointer_bug.md](hip_graph_device_pointer_bug.md)
- Full real-prompt benchmark analysis: [benches/results/full_benchmark_wmma_analysis.md](benches/results/full_benchmark_wmma_analysis.md)

## License

GPL-3.0. See [LICENSE](LICENSE).
