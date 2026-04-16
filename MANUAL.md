# ROCmForge Manual

## 1. Scope

This manual describes the current command-line workflow for `rocmforge`.

## 2. Prerequisites

- Rust 1.81+
- ROCm/HIP toolkit (validated on ROCm 7.2.1)
- AMD GPU for HIP path (validated on RX 9070 XT / gfx1201 and RX 7900 XT / gfx1100)
- ROCm runtime libraries available at execution time (`libamdhip64.so.7`)
- GGUF model file (Qwen2.5, LLaMA, Mistral, or GLM)

## 3. Build

```bash
cargo build --release                  # CPU-only
cargo build --release --features gpu   # With GPU support (requires ROCm)
```

## 4. Run Inference

### Standard decode (single model)

```bash
./target/release/rocmforge \
  --model /path/to/model.gguf \
  --prompt "Hello" \
  --gpu
```

### Speculative decoding (draft + target)

Uses a small draft model to generate candidate tokens, verified by the larger target model. Requires both models to be Qwen2.5 (or same architecture with compatible vocab).

```bash
./target/release/rocmforge \
  --model /path/to/target-7b.gguf \
  --draft-model /path/to/draft-0.5b.gguf \
  --spec-depth 3 \
  --prompt "Hello" \
  --gpu
```

Output includes acceptance statistics:
```
54 tokens in 1659.1ms = 32.5 tok/s (speculative: 21/102 accepted = 20.6%, avg 0.6/step over 35 steps)
```

### CPU fallback

```bash
./target/release/rocmforge \
  --model /path/to/model.gguf \
  --prompt "Hello"
```

### CLI options

| Option | Description |
|---|---|
| `--model <path>` | GGUF model path (required) |
| `--prompt <text>` | Prompt text (required) |
| `--gpu` | Use GPU backend |
| `--max-tokens N` | Max generated tokens (default: 256) |
| `--temperature F` | Sampling temperature (default: 1.0) |
| `--top-p F` | Nucleus sampling threshold (default: 0.9) |
| `--no-template` | Disable chat template wrapping |
| `--draft-model <path>` | Draft model for speculative decoding (GPU only) |
| `--spec-depth N` | Speculation depth — tokens drafted per step (default: 5) |
| `--list-tensors` | Print model tensors and exit |
| `--debug` | Print debug logits info |

## 5. Environment Flags

### Safety and performance

| Flag | Effect |
|------|--------|
| `ROCMFORGE_GPU_SAFE_MODE=1` | Disable all experimental fast paths |
| `ROCMFORGE_ENABLE_DECODE_GRAPH=1` | Enable HIP graph capture + replay for decode |
| `ROCMFORGE_DISABLE_DECODE_GRAPH=1` | Force-disable graph (safety override) |
| `ROCMFORGE_ENABLE_EXPERIMENTAL_Q8_ACTIVATION_FASTPATH=1` | Specialized Q8 decode activation path |
| `ROCMFORGE_ENABLE_EXPERIMENTAL_FFN_FASTPATH=1` | Fused FFN kernels |

### Diagnostics

| Flag | Effect |
|------|--------|
| `ROCMFORGE_SPEC_DEBUG=1` | Print draft/target token comparison per spec decode step |

### Benchmarking

| Flag | Effect |
|------|--------|
| `ROCMFORGE_RUN_REAL_MODEL_GPU_TESTS=1` | Enable real-model benchmark tests |
| `ROCMFORGE_RUN_GPU_BENCHES=1` | Enable Criterion GPU benchmarks |
| `ROCMFORGE_BENCH_RUNS=N` | Number of benchmark runs |
| `ROCMFORGE_BENCH_WARMUP=N` | Number of warmup runs |
| `ROCMFORGE_BENCH_TOKENS=N` | Tokens per benchmark run |
| `ROCMFORGE_BENCH_CONTEXT=N` | Context length for benchmarks |

## 6. Benchmarks

### 6.1 Quick CLI benchmark

```bash
./target/release/rocmforge \
  --model /path/to/model.gguf \
  --prompt "Explain what a transformer is." \
  --gpu --max-tokens 128
```

### 6.2 Real-model decode harness

```bash
ROCMFORGE_RUN_REAL_MODEL_GPU_TESTS=1 \
ROCMFORGE_BENCH_RUNS=10 ROCMFORGE_BENCH_WARMUP=1 ROCMFORGE_BENCH_TOKENS=128 \
cargo test --release --features gpu --test gpu_decode_real \
  test_gpu_greedy_decode_benchmark_real_model_multi_run \
  -- --ignored --nocapture --test-threads=1
```

### 6.3 Criterion benchmark

```bash
ROCMFORGE_RUN_GPU_BENCHES=1 cargo bench --bench gpu_decode --features gpu -- --noplot
```

### 6.4 rocprofv3 profiling

```bash
./.rocprofv3/profile_decode.sh runtime
./.rocprofv3/profile_decode.sh runtime-graph
```

## 7. Measured Results (April 16, 2026)

All measurements on RX 9070 XT (gfx1201), ROCm 7.2.1, Qwen2.5 Q4_0.

### 7.1 Decode throughput comparison vs llama.cpp ROCm

| Model | ROCmForge | llama.cpp ROCm | Ratio |
|-------|-----------|----------------|-------|
| 0.5B Q4_0 (tg128) | 222 tok/s | 358 tok/s | 62% |
| 7B Q4_0 (tg128) | 82 tok/s | 117 tok/s | 70% |

### 7.2 Prefill throughput comparison

| Model | ROCmForge | llama.cpp ROCm |
|-------|-----------|----------------|
| 7B Q4_0 (pp19) | 59 tok/s | 1,092 tok/s |
| 7B Q4_0 (pp512) | — | 5,141 tok/s |

### 7.3 Speculative decoding (0.5B draft + 7B target)

| Config | tok/s | Acceptance | Avg tokens/step |
|--------|-------|------------|-----------------|
| Baseline (no spec) | 82 | — | 1.0 |
| depth=1 | 57 | 44.7% | 0.4 |
| depth=2 | 40 | 26.0% | 0.5 |
| depth=3 | 33 | 20.6% | 0.6 |
| depth=5 | 22 | 11.7% | 0.6 |

Note: Verify runs token-by-token (N+1 sequential forward passes). Batched verify needed for speedup.

### 7.4 Historical progression (0.5B with HIP graph)

| Stage | Decode tok/s |
|-------|-------------|
| gfx1100 baseline | 527 |
| gfx1201 port | 508 |
| + QKV+RoPE+KV-Write fusion | 594 |
| + All fusions (Norm+QKV+RoPE, Norm+Gate+Up) | 646 |
| Current (full-decode graph disabled*) | 222 |

*Full-decode graph disabled due to RDNA4 HIP graph device-pointer bug. See `hip_graph_device_pointer_bug.md`.

## 8. What Works

- End-to-end GPU inference for Qwen2.5, LLaMA, Mistral, GLM
- FP16 KV cache (halved VRAM vs FP32)
- Fused kernels (RMSNorm+QKV+RoPE+KV-Write, RMSNorm+Gate+Up+SiLU)
- Speculative decoding with draft + target model
- Q4_0, Q4_1, Q4_K, Q5_K, Q6_K, Q8_0 quantization formats
- Tail-only HIP graph (lm_head + argmax)
- CPU fallback with AVX2/AVX-512/NEON SIMD

## 9. What Needs Work

- Batched verify for speculative decoding (current token-by-token verify negates the speedup)
- Prefill throughput (no GEMM, only GEMV — 18x slower than llama.cpp)
- Full-decode HIP graph on RDNA4 (blocked by ROCm bug, see `hip_graph_device_pointer_bug.md`)
- Broader model validation beyond Qwen2.5

## 10. Troubleshooting

If performance is unexpectedly low:

1. Confirm `--release` build
2. Confirm `--gpu` is used
3. Check whether `ROCMFORGE_GPU_SAFE_MODE` is set
4. Confirm ROCm runtime environment resolves `libamdhip64.so.7`
5. Run the benchmark harness (section 6.2) and compare against section 7
