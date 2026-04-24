# Phase 2.2A — FP8-E5M2 KV-Cache

**Date:** 2026-04-24
**Branch:** v1.0-dev
**Hardware:** AMD Radeon RX 9070 XT (gfx1201, RDNA4)
**Toggle:** `ROCMFORGE_KV_FP8=1` (default: FP32)

## TL;DR

Opt-in FP8-E5M2 (OCP "bf8") KV-cache via `ROCMFORGE_KV_FP8=1`. Each
cache element drops from 4 bytes (FP32, current baseline) to 1 byte —
**4× more context at the same VRAM budget**. Conversion is per-element
on read/write via the native gfx1201 intrinsics
`__builtin_amdgcn_cvt_pk_bf8_f32` and `__builtin_amdgcn_cvt_f32_bf8`.

| | FP32-KV (baseline) | **FP8-KV** |
|---|---:|---:|
| Bytes per element | 4 | **1** (×0.25) |
| Max context @ 10 GB free VRAM (Qwen3-8B) | ~68 k tokens | **~272 k tokens** |
| 15-prompt decode tok/s | 96.2 | **95.9** (−0.3 %, speed-neutral) |
| Monitor events | 3 RepetitionDetected | **0** (improved!) |
| 15 / 15 coherent | yes | **yes** |
| Attention cosine vs FP32 (seq=32, h=128) | 1.0000 | **0.99856** |

The prompt projected FP16→FP8 (2× savings); the actual baseline was
FP32 so we get **4× savings** for free. With 8192-token context and 16
GB VRAM, the FP32 KV-cache consumes ~1.2 GB; FP8 drops that to **~300
MB**, freeing 900 MB for longer contexts or larger models.

## Kernel

Single new source file `hip_kernels_v1/kv_cache/kv_cache_fp8.hip`
(211 LOC). Two kernels:

| Entry | Role | Notes |
|---|---|---|
| `rocmforge_launch_kv_cache_append_fp8` | FP32 input → bf8 cache bytes | Per-element `__builtin_amdgcn_cvt_pk_bf8_f32(x, 0, 0, false)` takes the low byte. Same grid shape as the FP32 append. |
| `rocmforge_launch_attention_decode_fp8` | Reads bf8 cache, FP32 online softmax | Template-style copy of the FP32 `attention_decode` kernel with bf8→FP32 conversion inline at each K/V load. |

### Phase 1 (K-read): 4-byte int load → 4 FP32 conversions

```cpp
for (int d = 0; d < head_dim; d += 4) {
    const int packed = *(const int*)&k_vec[d];
    acc += q_vec[d + 0] * __builtin_amdgcn_cvt_f32_bf8(packed, 0);
    acc += q_vec[d + 1] * __builtin_amdgcn_cvt_f32_bf8(packed, 1);
    acc += q_vec[d + 2] * __builtin_amdgcn_cvt_f32_bf8(packed, 2);
    acc += q_vec[d + 3] * __builtin_amdgcn_cvt_f32_bf8(packed, 3);
}
```

The `byte_sel` argument to the intrinsic is a **compile-time constant**
(hipcc rejects runtime values). For the K-read this is free — `d` is a
loop induction and the unrolled 0/1/2/3 indices are literal.

### Phase 4 (V-read): shift-and-extract trick

The V-phase loop has each thread responsible for one `d` across all
`j`. The byte position `d & 3` is runtime-varying. Instead of 4 copies
of the loop body (one per constant byte_sel), we shift the packed
`int` so that byte 0 always holds what we want and pass the literal
`0` to the intrinsic:

```cpp
const int shift = (d & 3) * 8;           // runtime shift is free
const int d_aligned = d & ~3;
for (int j = 0; j < seq_len; ++j) {
    const int packed = *(const int*)&v_head[j * head_dim + d_aligned];
    acc += s_scores[j] * __builtin_amdgcn_cvt_f32_bf8(packed >> shift, 0);
}
```

This approach compiled cleanly (the literal `0` satisfies hipcc's
compile-time-constant requirement) and generates the same code as
4-way specialisation would.

## Plumbing

| Component | Change |
|---|---|
| `src_v1/graph/buffer_plan.rs` | New `KvPrecision {Fp32, Fp8E5M2}` enum with `from_env()` + `bytes_per_element()`. `KvCacheLayout` gains a `precision` field; new constructor `from_config_with_precision`; `bytes_per_side()` helper for the per-layer allocation. |
| `src_v1/graph/mod.rs` | Re-exports `KvPrecision`. |
| `src_v1/backend/gpu/attention.rs` | 2 new `extern "C"` bindings (`rocmforge_launch_kv_cache_append_fp8`, `rocmforge_launch_attention_decode_fp8`). |
| `src_v1/graph/executor.rs` | Selects precision at construction via `KvPrecision::from_env()`. All 3 dispatch sites (decode append, decode attention, prefill per-token append) branch on `self.kv_layout.precision`. `reset_kv_cache` uses `bytes_per_side()` — zero-fill is bit-pattern-compatible between FP32 0x00000000 and bf8 0x00 (both represent +0). |
| `build.rs` + `hip_kernels_v1/CMakeLists.txt` | +1 library entry. |
| `Cargo.toml` | +1 `[[test]]` entry. |

`launch_index_spans` is **unchanged**: the FP8 and FP32 paths each emit
exactly 1 kernel launch per `KvCacheAppend` / `Attention` node. The
HIP-Graph capture validation keeps matching.

## Correctness

### bf8 roundtrip on known values

```
  [ 0]     1.0000 →     1.0000  (  0.00%)    [ 8]     4.0000 →     4.0000  (  0.00%)
  [ 1]    -1.0000 →    -1.0000  (  0.00%)    [ 9]    -4.0000 →    -4.0000  (  0.00%)
  [ 2]     0.5000 →     0.5000  (  0.00%)    [10]    16.0000 →    16.0000  (  0.00%)
  [ 3]    -0.5000 →    -0.5000  (  0.00%)    [11]   256.0000 →   256.0000  (  0.00%)
  [ 4]    10.0000 →    10.0000  (  0.00%)    [12]  1024.0000 →  1024.0000  (  0.00%)
  [ 5]   -10.0000 →   -10.0000  (  0.00%)    [13]     0.2500 →     0.2500  (  0.00%)
  [ 6]   100.0000 →    96.0000  (  0.04%)    [14]     0.1250 →     0.1250  (  0.00%)
  [ 7]     0.0010 →     0.0010  (  0.02%)    [15]     0.0625 →     0.0625  (  0.00%)
```

Powers of 2 and small integers round trip bit-exact. The one case that
loses precision is **100.0 → 96.0** (4 % error) because 100 isn't
representable in E5M2; it rounds to the nearest representable value,
which is 96 (= 64 × 1.5). This is the expected E5M2 quantum — 2
mantissa bits → ~25 % worst-case relative error but typically much
better for values that happen to land on representable fractions.

Attention cosine similarity (FP32 vs FP8-KV on h=128, seq=32):
**0.99856** — well above the 0.99 gate.

### End-to-end sanity

Mutex prompt at 40 decode tokens, greedy, both precisions:

```
FP32-KV:  "A mutex, short for 'mutual exclusion,' is a synchronization
           mechanism used in concurrent programming to ensure that only
           one thread or process can access a shared resource or
           critical section of code"

FP8-KV:   "A mutex, short for 'mutual exclusion,' is a synchronization
           mechanism used in concurrent programming to ensure that only
           one thread or process can access a shared resource or
           critical section of code"
```

**Byte-identical output** at short decode horizon despite the
bf8-quantized cache.

## End-to-End (15-prompt validation suite)

| Prompt | FP32-KV (Schr. 5) | **FP8-KV** | Δ tok/s |
|---|---:|---:|---:|
| 1 | Greeting | 101.2 | **101.9** | +0.7 % |
| 2 | Simple Sequence | 100.9 | **101.5** | +0.6 % |
| 3 | Prime Check (Python) | 98.8 | 98.7 | −0.1 % |
| 4 | LRU Cache (C++) | 98.2 | 94.1 | −4.2 % |
| 5 | REST API (Go) | 99.4 | 96.2 | −3.2 % |
| 6 | Mutex Explanation | 102.0 | **102.5** | +0.5 % |
| 7 | TCP vs UDP | 96.8 | 92.8 | −4.1 % |
| 8 | GPU Architecture Blog Post | 96.1 | 92.7 | −3.5 % |
| 9 | Binary Search Complexity | 99.4 | **100.0** | +0.6 % |
| 10 | Debug Code | 99.0 | **99.5** | +0.5 % |
| 11 | Distributed Message Queue | 89.6 | **94.8** | **+5.8 %** |
| 12 | Long System Prompt + Question | 92.0 | 91.8 | −0.2 % |
| 13 | Long Output Story | 98.6 | 98.5 | −0.1 % |
| 14 | Arithmetic (Q4_K Precision) | 103.7 | **104.4** | +0.7 % |
| 15 | Emoji/Special Characters | 101.5 | **102.2** | +0.7 % |

**Aggregate: 96.2 → 95.9 tok/s (−0.3 %, speed-neutral ±5 % gate).**

**Quality: 0 monitor events** (FP32-KV had 3 `RepetitionDetected` on
token_id 10519). The bf8 quantum appears to have broken the specific
logit tie that triggered those events — surprising but legitimate.

**8/15 prompts faster under FP8-KV**, 7/15 slower. The pattern is not
length-dependent (prompt 11 with 376 decode tokens is +5.8 %, prompt
12 with 256 is −0.2 %). The per-prompt noise cancels out.

Decode-token counts differ slightly between runs (the bf8 quantization
occasionally nudges the greedy-argmax to a neighbouring token at some
step, which then produces a slightly different continuation). **All
outputs remain coherent, terminated at either EOS or the max-token
cap.** No nonsense, no NaN, no infinite repetitions.

### Bandit commit

Identical to the FP32-KV baseline:
```
Q4_K n=1024 k=4096   → q4_k_mmvq (4.7× win over standard)
Q4_K n=4096 k=4096   → q4_k_mmvq (6.0× win)
Q6_K n=1024 k=4096   → q6_k_mmvq (1.78× win)
Q6_K n=4096 k=12288  → q6_k_mmvq (1.89× win)
```

## Max-context projection

For Qwen3-8B (n_layers=36, n_kv_heads=4, head_dim=128):

| Precision | Bytes / token | Max context @ 10 GB free |
|---|---:|---:|
| FP32 | 147 456 | ~67 800 tokens |
| FP16 *(not implemented)* | 73 728 | ~135 700 |
| **FP8-E5M2** | **36 864** | **~271 500 tokens** |

ROCmForge currently caps context at 8 192 (`model_loader.rs:191`).
With FP8-KV the cap could be raised without VRAM concerns — follow-up
work.

## Files

| New | LOC |
|---|---:|
| `hip_kernels_v1/kv_cache/kv_cache_fp8.hip` | 211 |
| `tests_v1/fp8_kv_cache_test.rs` | 360 |
| `results/phase2_fp8_kv_cache.md` (this report) | — |
| `results/phase2_fp8_kv_cache_15prompt_suite.md` (raw) | — |

| Modified | Change |
|---|---|
| `hip_kernels_v1/CMakeLists.txt` | +1 `add_hip_kernel` |
| `build.rs` | +1 link-lib entry |
| `src_v1/graph/buffer_plan.rs` | +`KvPrecision` enum; `KvCacheLayout` gains `precision` field, `from_config_with_precision` constructor, `bytes_per_side` helper |
| `src_v1/graph/mod.rs` | re-export `KvPrecision` |
| `src_v1/backend/gpu/attention.rs` | +2 `extern "C"` bindings |
| `src_v1/graph/executor.rs` | precision-aware buffer sizing; match on `self.kv_layout.precision` at 3 dispatch sites (decode append, decode attention, prefill append); `reset_kv_cache` uses `bytes_per_side` |
| `Cargo.toml` | +1 `[[test]]` entry |

Total new kernel code + plumbing: **~320 LOC** (excluding tests) — on
target with the scope estimate.

## Honest Caveats

1. **Speed gain is ~0 %, not positive.** The FP8 cache uses 1/4 the
   bytes, which could in theory speed up long-context attention (less
   KV-read bandwidth). At decode-time 8192-cap context the KV-read is
   not the bottleneck on gfx1201; the MMVQ weight reads dominate. FP8
   would become a speed lever at longer contexts or on
   attention-bound workloads — neither of which this suite exercises.
2. **E5M2 is lossy.** The 100.0 → 96.0 quantization hit shows it.
   Attention scores cover a narrow range where this error averages
   out, but downstream code that reads values directly from the cache
   (e.g. a future `kv_cache_dequant` kernel for offline inspection)
   would need to know this.
3. **No intermediate FP16 path.** The prompt suggested FP16 as the
   baseline; our actual baseline was FP32. Adding FP16 would be 3-way
   precision selection and more Bandit work — skipped for now as FP8
   covers the "save memory" use case and FP32 covers the "preserve
   precision" use case.
4. **bf8 is OCP E5M2, not MXFP8 / Deepseek-FP8.** Some other projects
   use different FP8 formats (E4M3 for weights, E5M2 for activations
   is NVIDIA's convention). Our choice matches llama.cpp, AMD MI300X
   tuning, and the mainstream industry pattern for KV-caches.

## Next Concrete Step

FP8-KV is now opt-in and stable. Natural follow-ups:

→ **Raise the 8 192-token context cap** when FP8-KV is on. Current cap
  is conservative for FP32; with FP8 the cap can go to 32 k or 50 k
  without risking the VRAM budget. Needs:
  - env-aware cap logic in `model_loader.rs::plan_arena`
  - tests at seq_len = 32 k for attention correctness / numerical drift
  - rocprof on a real long-context prompt

→ **Long-context attention rewrite** — the current `attention_decode`
  kernel has an O(seq²/blockDim) softmax-reduce that's fine at 8 k but
  caps out around 12 k (LDS budget). Both FP32 and FP8 variants share
  this. A tiled flash-attention decode would lift the cap to 32 k+.

→ **Periodic cache refresh** in FP8-KV for super-long contexts. Every
  ~1 k tokens, the attention of early tokens might drift as bf8 noise
  accumulates. A lazy "dequant-requant" pass could recover precision
  at a fixed interval. Premature optimisation — wait for data from
  actual long-context runs.
