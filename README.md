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

| Model (pp256) | ROCmForge WMMA GEMM+Attn | ROCmForge WMMA GEMM only | ROCmForge hipBLAS | ROCmForge baseline |
|---------------|-------------------------:|-------------------------:|------------------:|-------------------:|
| Qwen2.5-7B Q4_0 | **620** | 92.3 | 85.7 | 63.6 |

| Prompt length | WMMA GEMM+Attn tok/s |
|--------------:|--------------------:|
| pp64  | 560 |
| pp128 | 602 |
| pp192 | 618 |
| pp256 | **620** |
| pp384 | 626 |
| pp512 | 628 |

Prefill now uses custom WMMA GEMM and WMMA FlashAttention kernels on gfx1201 matrix cores (`__builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12`). hipBLAS/Tensile on ROCm 7.2 does not use matrix cores for gfx1201 — ROCmForge bypasses this with hand-written WMMA kernels. The attention kernel uses online softmax, GQA (28Q / 4KV), causal masking with zero-work elimination on upper-triangle tiles, and replaces a scalar per-head loop that was ~84 % of pp=256 prefill time. At pp=256 total prefill is now 446 ms → 6.7× faster than the WMMA-GEMM-only path and 10× over the original baseline. Analysis: [`benches/results/prefill_wmma_attn_e2e_analysis.md`](benches/results/prefill_wmma_attn_e2e_analysis.md).

Flags:
- `ROCMFORGE_DISABLE_WMMA_ATTENTION=1` — fall back to the scalar per-head attention kernel.
- `ROCMFORGE_DISABLE_WMMA_PREFILL=1` — fall through to the hipBLAS GEMM path.
- Both plus `ROCMFORGE_DISABLE_HIPBLAS_PREFILL=1` — exercise the original custom-GEMV dispatch.

A head-to-head pp=256 measurement against llama.cpp ROCm is still outstanding; their published pp=19 figure (1,092 tok/s) is apples-to-oranges at this prompt length.

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
- **Prefill seq_len alignment**: The WMMA prefill GEMM requires seq_len divisible by 64; the WMMA attention kernel additionally requires `head_dim == 128`. Unaligned prompts fall back to hipBLAS/scalar kernels and lose most of the gain. Kernel variants for arbitrary seq_len are planned.

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
| `ROCMFORGE_DISABLE_WMMA_ATTENTION=1` | Skip the WMMA GQA + causal prefill attention kernel and fall back to the scalar per-head kernel (auto-enabled when `seq_len % 64 == 0`) |

## Documentation

- Manual: [MANUAL.md](MANUAL.md)
- Speculative decoding milestone summary: [docs/spec_decode_milestone_summary.md](docs/spec_decode_milestone_summary.md)
- Architecture notes (RDNA 4 memory-pipelining findings): [docs/architecture_notes.md](docs/architecture_notes.md)
- Benchmark progression: [profiling/benchmark_progression.md](profiling/benchmark_progression.md)
- HIP graph bug report: [hip_graph_device_pointer_bug.md](hip_graph_device_pointer_bug.md)

## License

GPL-3.0. See [LICENSE](LICENSE).
