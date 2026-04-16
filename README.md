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

| Model | ROCmForge | llama.cpp ROCm |
|-------|-----------|----------------|
| Qwen2.5-7B Q4_0 (pp19) | 59 | 1,092 |

Prefill is significantly slower because ROCmForge uses custom GEMV kernels while llama.cpp uses hipBLAS GEMM.

### Speculative decoding (0.5B draft + 7B target)

Currently slower than baseline due to token-by-token verification (N+1 sequential target forwards instead of one batched forward). Batched verify is the next optimization target.

| Depth | tok/s | Acceptance rate |
|-------|-------|-----------------|
| Baseline (no spec) | 82 | — |
| depth=1 | 57 | 44.7% |
| depth=3 | 33 | 20.6% |

### Known issues

- **Full-decode HIP graph disabled on RDNA4**: Graph replay of kernels reading device pointers returns stale values in complex graphs (~200+ nodes). See `hip_graph_device_pointer_bug.md`. Tail-only graph (lm_head + argmax) still active.
- Shared memory bug in v2 attention kernel fixed (was causing 7B decode to produce NaN)

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

## Documentation

- Manual: [MANUAL.md](MANUAL.md)
- Benchmark progression: [profiling/benchmark_progression.md](profiling/benchmark_progression.md)
- HIP graph bug report: [hip_graph_device_pointer_bug.md](hip_graph_device_pointer_bug.md)

## License

GPL-3.0. See [LICENSE](LICENSE).
