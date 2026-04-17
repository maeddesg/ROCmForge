rocmforge - LLM inference on AMD GPUs (HIP) with a CPU fallback path.

## Current status

- End-to-end GPU inference for Qwen2.5, LLaMA, Mistral, and GLM GGUF models
- Pure HIP/ROCm GPU backend — no CUDA, no cross-vendor abstraction
- Tested on RDNA4 (RX 9070 XT, gfx1201) and RDNA3 (RX 7900 XT, gfx1100)
- Speculative decoding support (draft + target model)
- FP16 KV cache for reduced VRAM usage
- Fused kernels: RMSNorm+QKV+RoPE+KV-Write, RMSNorm+Gate+Up+SiLU

## Performance (RX 9070 XT / gfx1201 / ROCm 7.2.1)

### Decode throughput (tok/s)

| Model | ROCmForge | llama.cpp ROCm | Gap |
|-------|-----------|----------------|-----|
| Qwen2.5-0.5B Q4_0 | 222 | 358 | 62% |
| Qwen2.5-7B Q4_0 | 82 | 117 | 70% |

### Prefill throughput (tok/s)

| Model | ROCmForge WMMA (pp256) | ROCmForge hipBLAS (pp256) | ROCmForge baseline (pp19) | llama.cpp ROCm (pp19) |
|-------|-----------------------:|--------------------------:|--------------------------:|----------------------:|
| Qwen2.5-7B Q4_0 | 92.4 | 86.0 | 59 | 1,092 |

Prefill now has a dedicated WMMA Q4_0 GEMM path using gfx1201 matrix cores (`__builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12`), with inline Q4_0 dequant so there is no FP16 weight scratch round-trip. Isolated GEMM is 2.5× faster than hipBLAS (whose Tensile backend falls back to a VALU kernel on this hardware — confirmed via rocprofv3). End-to-end prefill gain at pp=256 is modest (+7-8 % over hipBLAS) because GEMM is only ~4 % of total prefill time; prefill attention dominates (~84 %) and is the next optimisation target. Full analysis in [`benches/results/prefill_wmma_e2e_analysis.md`](benches/results/prefill_wmma_e2e_analysis.md).

Disable with `ROCMFORGE_DISABLE_WMMA_PREFILL=1` to fall through to the hipBLAS path, or also set `ROCMFORGE_DISABLE_HIPBLAS_PREFILL=1` to exercise the original custom-GEMM kernel.

### Speculative decoding (0.5B draft + 7B target)

Speculative decoding is profitable for high-acceptance workloads (code generation, α ≥ 73%). For mixed prompts, baseline decode is faster. Batched verify (PR #14), tiled GEMV, batched lm_head and adaptive depth are implemented; further verify-path micro-optimizations have reached a plateau (see `docs/spec_decode_milestone_summary.md`).

| Depth | tok/s (median) | Acceptance rate |
|-------|---------------:|-----------------|
| Baseline (no spec) | 82 | — |
| depth=1 (median) | 69 | ~50% |
| depth=1 (code, best) | **86** | 91% |
| depth=3 | 66 | ~46% |
| depth=5 | 57 | ~35% |

Break-even α ≈ 41%. Below that, baseline decode is faster. Adaptive depth (EMA-based) automatically converges to the profitable tier per prompt.

### CPU decode throughput (Ryzen 9 7945HX, AVX-512 VNNI)

| Model             | tok/s |
|-------------------|------:|
| Qwen2.5-0.5B Q4_0 |  12.1 |
| Qwen2.5-7B Q4_0   |   0.7 |

CPU path is functional and has an AVX-512 VNNI Q4_0 GEMV kernel on Zen4+, but is not otherwise optimized — Rayon fork-join overhead per GEMV call and scalar attention/norm/SiLU dominate the 0.5B forward pass. For production inference, use `--gpu`. See [docs/architecture_notes.md](docs/architecture_notes.md) ("Orchestration-Falle bei kleinen Modellen") for the empirical breakdown.

### Known issues

- **Full-decode HIP graph disabled on RDNA4**: Graph replay of kernels reading device pointers returns stale values in complex graphs (~200+ nodes). See `hip_graph_device_pointer_bug.md`. Tail-only graph (lm_head + argmax) still active.
- **Prefill throughput is the largest gap vs. llama.cpp** (5% of their pp19 baseline). Custom GEMV kernels instead of GEMM (hipBLAS/WMMA). Next optimization target.

## Requirements

- Rust 1.81+
- ROCm/HIP toolkit (tested on ROCm 7.2)
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
| `ROCMFORGE_ENABLE_EXPERIMENTAL_Q8_ACTIVATION_FASTPATH=1` | Specialized Q8 decode activation path |
| `ROCMFORGE_ENABLE_EXPERIMENTAL_FFN_FASTPATH=1` | Fused FFN kernels |
| `ROCMFORGE_SPEC_DEBUG=1` | Print speculative decode draft/target token comparison |
| `ROCMFORGE_PROFILE_SPEC_STEP=1` | HIP Event timing for spec-step cost breakdown (5 phases) |
| `ROCMFORGE_PROFILE_VERIFY_BREAKDOWN=1` | Sub-phase timing within verify layers (requires PROFILE_SPEC_STEP) |
| `ROCMFORGE_DISABLE_TILED_GEMV=1` | Disable tiled batched GEMV for large FFN projections (default on) |
| `ROCMFORGE_DISABLE_BATCHED_LM_HEAD=1` | Disable batched verify lm_head, fall back to sequential per-position dispatch |
| `ROCMFORGE_DISABLE_AVX512=1` | Force the Q4_0 × Q8_0 CPU GEMV back to the AVX2 path (auto-enabled on Zen4+) |
| `ROCMFORGE_DISABLE_WMMA_PREFILL=1` | Skip the WMMA Q4_0 prefill kernel and fall through to hipBLAS (auto-enabled on gfx12+) |

## Documentation

- Manual: [MANUAL.md](MANUAL.md)
- Speculative decoding milestone summary: [docs/spec_decode_milestone_summary.md](docs/spec_decode_milestone_summary.md)
- Architecture notes (RDNA 4 memory-pipelining findings): [docs/architecture_notes.md](docs/architecture_notes.md)
- Benchmark progression: [profiling/benchmark_progression.md](profiling/benchmark_progression.md)
- HIP graph bug report: [hip_graph_device_pointer_bug.md](hip_graph_device_pointer_bug.md)

## License

GPL-3.0. See [LICENSE](LICENSE).
