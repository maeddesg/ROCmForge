# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ROCmForge is an AMD-first LLM inference engine for GGUF-format models (Qwen2.5, LLaMA, Mistral, GLM) written in pure Rust with HIP/ROCm GPU acceleration. It targets batch-1 decode throughput on consumer AMD GPUs. Primary development hardware is RDNA4 (RX 9070 XT / gfx1201), also tested on RDNA3 (RX 7900 XT / gfx1100). Supports speculative decoding with a draft model.

## Build Commands

```bash
# CPU-only (no GPU dependencies)
cargo build --release

# With GPU support (requires ROCm 7.2+ and AMD GPU)
cargo build --release --features gpu

# Type check / lint / format
cargo check --features gpu --all-targets
cargo clippy --features gpu
cargo fmt --check
```

## Testing

GPU tests **must run serially** — parallel GPU access causes driver corruption on consumer hardware.

```bash
# CPU unit tests (no GPU required)
cargo test --release --lib

# All GPU tests (serial required)
cargo test --release --features gpu -- --test-threads=1

# Single GPU integration test suite
cargo test --release --features gpu --test integration_gpu -- --test-threads=1

# Quantization tests
cargo test --release --test quant_unit

# Real-model benchmark harness (requires actual GGUF file + env flag)
ROCMFORGE_RUN_REAL_MODEL_GPU_TESTS=1 \
ROCMFORGE_BENCH_RUNS=10 ROCMFORGE_BENCH_WARMUP=1 ROCMFORGE_BENCH_TOKENS=128 \
cargo test --release --features gpu --test gpu_decode_real \
  test_gpu_greedy_decode_benchmark_real_model_multi_run \
  -- --ignored --nocapture --test-threads=1

# Criterion benchmarks
ROCMFORGE_RUN_GPU_BENCHES=1 cargo bench --bench gpu_decode --features gpu -- --noplot
```

All GPU test functions must use `#[serial]` from the `serial_test` crate.

## Runtime Environment Flags (GPU only)

All flags are opt-in; defaults are conservative.

| Flag | Effect |
|------|--------|
| `ROCMFORGE_ENABLE_DECODE_GRAPH=1` | Enable HIP graph capture + replay for decode |
| `ROCMFORGE_DISABLE_DECODE_GRAPH=1` | Force-disable graph (safety override) |
| `ROCMFORGE_GPU_SAFE_MODE=1` | Disable all experimental fast paths |
| `ROCMFORGE_ENABLE_EXPERIMENTAL_Q8_ACTIVATION_FASTPATH=1` | Specialized Q8 decode activation path |
| `ROCMFORGE_ENABLE_EXPERIMENTAL_FFN_FASTPATH=1` | Fused FFN kernels |
| `ROCMFORGE_ENABLE_LAUNCH_AUTOTUNE=1` | Cache kernel variant selections |
| `ROCMFORGE_SPEC_DEBUG=1` | Print draft/target token comparison per spec decode step |
| `ROCMFORGE_PROFILE_SPEC_STEP=1` | HIP Event timing for spec-step cost breakdown (5 phases) |
| `ROCMFORGE_PROFILE_VERIFY_BREAKDOWN=1` | Sub-phase timing within verify layers (requires PROFILE_SPEC_STEP) |
| `ROCMFORGE_DISABLE_TILED_GEMV=1` | Disable tiled batched GEMV for large FFN projections (default on) |
| `ROCMFORGE_DISABLE_BATCHED_LM_HEAD=1` | Disable batched verify lm_head, fall back to sequential per-position dispatch |
| `ROCMFORGE_DISABLE_WMMA_PREFILL=1` | Disable WMMA Q4_0 prefill GEMM, fall through to hipBLAS |
| `ROCMFORGE_DISABLE_WMMA_ATTENTION=1` | Disable WMMA GQA+causal prefill attention, fall back to scalar kernel |
| `ROCMFORGE_RUN_REAL_MODEL_GPU_TESTS=1` | Enable real-model benchmark tests |
| `ROCMFORGE_RUN_GPU_BENCHES=1` | Enable Criterion GPU benchmarks |

## Architecture

### Data Flow

```
GGUF file (mmap) → loader/ (zero-copy TensorView)
  → config/ (ModelTraits + TensorNameRegistry resolves tensor names)
  → cpu/weights.rs or gpu/weights.rs (buffer init)
  → tokenizer/ (BPE)
  → cpu/prefill.rs or gpu/forward.rs (prefill: all tokens at once)
  → cpu/forward.rs or gpu/forward.rs (decode: one token per step)
      Attention (RoPE → QKV → softmax → V) + MLP (SwiGLU + RMSNorm)
  → cpu/sampler.rs (greedy / top-p)
  → repeat decode until EOS or max_tokens

Speculative decode path (GPU only, --draft-model):
  Draft model (0.5B) drafts N tokens → Target model (7B) verifies N+1 tokens
  → greedy accept/reject → KV cache sync → emit accepted tokens
  → gpu_speculative_decode_step() in gpu/forward.rs
```

### Key Modules

| Module | Purpose |
|--------|---------|
| `src/config.rs` | `ModelConfig`, `ModelTraits`, `TensorNameRegistry` — architecture-agnostic tensor resolution (1095 LOC) |
| `src/loader/` | Zero-copy GGUF parsing; `TensorView` exposes raw byte slices via memmap |
| `src/cpu/` | Pure Rust backend: ops, prefill, decode, SIMD dispatch, quantization |
| `src/cpu/kernels/` | SIMD kernels (AVX2, AVX-512, NEON, SVE) per quantization format |
| `src/gpu/` | HIP backend (conditional on `gpu` feature): forward pass, ops, weights, KV cache |
| `src/gpu/ffi.rs` | Safe wrappers for HIP API (30 unsafe blocks, all other GPU/CPU code is safe Rust) |
| `src/gpu/graph.rs` | HIP graph capture + replay for decode loop (tail-only on RDNA4) |
| `src/gpu/safety.rs` | Atomic caching of runtime env flags for experimental fast paths |
| `src/gpu/kernels/` | HIP kernel FFI wrappers: dequant, attention, RoPE, RMSNorm, elementwise, Q6_K |
| `src/tokenizer/` | GPT-2 style BPE compatible with Qwen2.5 |
| `src/hardware/` | CPU feature detection (SIMD caps, L3-based prefill batch sizing) |
| `src/bench/` | Benchmark discovery and result reporting utilities |

### Critical Design Rules

**Metadata-driven, never model-name-specific.** Always use `config.tensor_registry.resolve(TensorName::*, layer)` to look up tensor names. Never write `if architecture == "qwen2" { ... }`. New architectures are added by registering traits in `ModelTraits::detect()`.

**No panics on the GPU path.** All HIP calls return `GpuResult<T>`. A GPU panic can wedge consumer hardware requiring a driver reset.

**Serial GPU tests.** Any test that touches GPU state must be annotated `#[serial]`. Running GPU tests in parallel will corrupt driver state.

**CPU SIMD dispatch is cached once.** `CpuFeatures::get()` uses `OnceLock`; the result is a function-pointer-based `KernelPreference`. No runtime CPUID branches in hot paths.

**Unsafe code is concentrated in FFI.** `gpu/ffi.rs` (HIP API) and `cpu/ops.rs` (slice casts for quantization) hold nearly all unsafe blocks. Forward pass code is safe Rust.

**Quantization format isolation.** Each format (Q4_0, Q4_1, Q4_K, Q5_K, Q8_0, …) has its own block constants, dequantization kernel, and round-trip tests. No mixed-format kernel calls.

**32 KB LDS budget (GPU).** Consumer gfx1100 has ~32 KB practical shared memory. Large tensors (7B+) use automatic chunked LDS loading — no special-casing by model name.

**CPU modules target <1000 LOC.** `ops.rs` (2323 LOC) is an accepted exception as the core dispatch file. New CPU features should be split into separate files.

## Performance Baseline (April 17, 2026 — RX 9070 XT / gfx1201)

| Model | ROCmForge | llama.cpp ROCm | Ratio |
|-------|-----------|----------------|-------|
| Qwen2.5-0.5B Q4_0 decode | 222 tok/s | 358 tok/s | 62% |
| Qwen2.5-7B Q4_0 decode | 82 tok/s | 117 tok/s | 70% |
| Qwen2.5-7B Q4_0 prefill (pp19) | 59 tok/s | 1,092 tok/s | 5% |
| Qwen2.5-7B Q4_0 prefill (pp256) | 620 tok/s | not measured | — |

Full-decode HIP graph is disabled on RDNA4 due to a device-pointer stale-read bug in complex graphs. See `hip_graph_device_pointer_bug.md`. With full-decode graph enabled (gfx1100, before bug), 0.5B decode reached ~646 tok/s after kernel fusions.

Speculative decoding (0.5B draft + 7B target) is implemented but currently slower than baseline because verify runs token-by-token. Batched verify is the next optimization target.

Gap vs. llama.cpp ROCm: primarily launch overhead and missing GEMM for prefill. See `improvements.md` and `OPTIMIZATION_PLAN.md`.

### Profiling

```bash
# Kernel timeline trace
./.rocprofv3/profile_decode.sh runtime

# With HIP graph
./.rocprofv3/profile_decode.sh runtime-graph

# VRAM usage
rocm-smi --showmeminfo vram
```

## Dependencies

**GPU build requires**: ROCm 7.2+, `libamdhip64.so.7`, `libhiprtc.so`, `hipcc`, `cmake`.

**CPU build requires**: Rust 1.81+, no native dependencies.
