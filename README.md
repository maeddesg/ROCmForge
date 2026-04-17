rocmforge - LLM inference on AMD GPUs (HIP) with a CPU fallback path.

## Current status

- End-to-end GPU inference for Qwen2.5, LLaMA, Mistral, and GLM GGUF models
- Pure HIP/ROCm GPU backend — no CUDA, no cross-vendor abstraction
- **Developed and tested exclusively on RDNA 4** (RX 9070 XT, gfx1201) with ROCm 7.2.1 on Arch Linux. See [Supported hardware](#supported-hardware).
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

## Performance (RX 9070 XT / gfx1201 / ROCm 7.2.1 / Qwen2.5-7B Q4_0)

### Head-to-head vs. llama.cpp ROCm (same host, same model)

15-prompt real-world benchmark (5 code / 5 chat / 5 prose, 19–41 prompt tokens, 128 decoded tokens, greedy, 3 runs median — see [`benches/results/full_benchmark_wmma_analysis.md`](benches/results/full_benchmark_wmma_analysis.md)):

| Metric | ROCmForge | llama.cpp ROCm | Ratio |
|--------|----------:|---------------:|------:|
| Decode (tok/s) | 102 | 121 | 0.84× |
| Prefill real prompts (tok/s, median) | 356 | 526 | 0.68× |
| TTFT real prompts (ms, median) | 67 | ~46 | 1.46× |
| Total wall-clock, 24 prompt + 128 gen (ms, median) | 1,319 | 1,104 | 1.20× |

Synthetic `llama-bench` at aligned pp sizes (`build 408225b`):

| Prompt length | ROCmForge (tok/s) | llama.cpp ROCm (tok/s) | Ratio |
|--------------:|------------------:|-----------------------:|------:|
| pp64  | 560 | 2,912 | 0.19× |
| pp128 | 602 | 3,966 | 0.15× |
| pp256 | 620 | 4,951 | 0.13× |
| pp512 | 629 | 5,158 | 0.12× |

**Reading the two tables.** At aligned benchmark shapes (pp256+) ROCmForge plateaus near 620–630 tok/s while llama.cpp scales to 5,000+; the synthetic ratio is ~8×. On realistic prompts (19–41 tokens) the picture narrows to ~1.5× on prefill and ~0.84× on decode, so end-to-end wall clock is 1.2× — much closer. The gap at long prompts is **not** a kernel-performance issue (the isolated WMMA kernels sit at 62–75 % of FP16 peak); it is the ~295 ms of unfused norm/RoPE/residual/embedding overhead per pp=256 prefill, which llama.cpp does fuse.

### ROCmForge prefill throughput (tok/s)

Synthetic (`"word "`-repeated prompts, exact pp size):

| Prompt length | WMMA GEMM + Attn |
|--------------:|-----------------:|
| pp64  | 560 |
| pp128 | 602 |
| pp192 | 618 |
| pp256 | **620** |
| pp384 | 626 |
| pp512 | 629 |

Real prompts (15-prompt benchmark, median, 19–41 tokens):

| Before Phase 3.2 (≥64 gate) | After Phase 3.2 (≥1 gate) | Speedup |
|----------------------------:|--------------------------:|--------:|
| 60.6 tok/s | **356 tok/s** | 5.9× |

### ROCmForge decode throughput (tok/s)

| Model | ROCmForge | llama.cpp ROCm |
|-------|----------:|---------------:|
| Qwen2.5-0.5B Q4_0 | 222 | 358 |
| Qwen2.5-7B Q4_0  | 102 | 117–121 |

### Prefill kernel detail

Prefill uses custom WMMA GEMM and WMMA FlashAttention kernels on gfx1201 matrix cores (`__builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12`). hipBLAS/Tensile on ROCm 7.2 does not use matrix cores on gfx1201 — ROCmForge bypasses this with hand-written WMMA kernels. The attention kernel implements FlashAttention-style online softmax, GQA (28 Q heads / 4 KV heads on Qwen2.5-7B), and causal masking with zero-work elimination on tiles strictly above the diagonal. `GpuPrefillScratch` pads all activation buffers up to a multiple of 64 rows and zero-initialises them once, so every `seq_len ≥ 1` engages the WMMA path without per-call memset overhead. Analysis: [`benches/results/prefill_wmma_attn_e2e_analysis.md`](benches/results/prefill_wmma_attn_e2e_analysis.md) and [`benches/results/full_benchmark_wmma_analysis.md`](benches/results/full_benchmark_wmma_analysis.md).

Flags (all opt-out, defaults enabled):

- `ROCMFORGE_DISABLE_WMMA_ATTENTION=1` — scalar per-head attention kernel.
- `ROCMFORGE_DISABLE_WMMA_PREFILL=1` — fall through to hipBLAS GEMM.
- `ROCMFORGE_DISABLE_HIPBLAS_PREFILL=1` — original custom-GEMV dispatch.

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
- **Quantisation:** WMMA acceleration is implemented for Q4_0 weights only. Other formats (Q4_K, Q5_K, Q6_K, Q8_0) fall back to scalar / GEMV kernels.
- **Prefill overhead:** ~295 ms of unfused norm/RoPE/residual operations per pp=256 prefill caps throughput. llama.cpp is ~8× faster at pp=256 synthetic — GEMM and attention kernels are competitive in isolation; the gap is structural, on the orchestration side.
- **Decode gap:** 102 tok/s vs. llama.cpp 117–121 tok/s (≈ 0.84×), unchanged from project start. Root cause not yet profiled.
- **Full-decode HIP graph disabled on RDNA 4:** graph replay of kernels reading device pointers returns stale values in complex graphs (~200+ nodes). See [`hip_graph_device_pointer_bug.md`](hip_graph_device_pointer_bug.md). Tail-only graph (lm_head + argmax) is still active.
- **WMMA attention `head_dim == 128` only:** Qwen2.5 / LLaMA-2 / Mistral are covered; other head dimensions fall back to the scalar attention kernel. Arbitrary `seq_len ≥ 1` is supported via zero-padding in `GpuPrefillScratch`.

## Requirements

- Rust 1.81+
- ROCm/HIP toolkit (tested on ROCm 7.2.1)
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
