# Batched lm_head Plan — Phase 0 Orientation

## Problem

The verify path in `gpu_verify_forward` (forward.rs:1259-1303) runs the final norm + lm_head + argmax **sequentially** for each position:

```
for i in 0..n:
    rms_norm(hidden[i]) → normed        // 1 dispatch
    gemv(normed) → logits               // 1 dispatch
    argmax(logits) → token              // 2 dispatches (stage1 + finalize)
    sync                                // full roundtrip
```

This is **4 dispatches + 1 sync per position**. At depth=1 (n=2): 8 dispatches + 2 syncs. At depth=3 (n=4): 16 dispatches + 4 syncs.

Measured cost: ~850-900 μs/position = **1,917 μs at depth=1, 3,366 μs at depth=3** (10.5% of verify).

## Goal

Replace with batched dispatch:

```
rms_norm_batched(hidden[0..n]) → normed_batch     // 1 dispatch
gpu_dispatch_gemm(normed_batch, n) → logits_batch  // 1 dispatch (batched GEMV)
for i in 0..n:
    argmax(logits_batch[i]) → token                // 2 dispatches each
sync once                                          // 1 roundtrip
```

Saves: (n-1) norm dispatches + (n-1) GEMV dispatches + (n-1) syncs.
Expected: ~800 μs at depth=1, ~2,500 μs at depth=3.

## Building Blocks Found

### 1. Sequential loop to replace
- **File:** `src/gpu/forward.rs:1259-1303`
- **Called from:** `gpu_verify_forward` (line 1194)
- **n** = `tokens.len()`, typically 2 (depth=1) to 9 (depth=8)

### 2. rms_norm_batched — READY
- **File:** `src/gpu/kernels/norm.rs:108`
- **Signature:** `rms_norm_batched(x, weight, out, n, eps, seq_len)` where seq_len = batch
- **Semantics:** Applies RMS norm to `seq_len` rows of `n` elements each, reading from `x[i*n..(i+1)*n]`, writing to `out[i*n..(i+1)*n]`
- **Stream:** Default (null) — matches verify path

### 3. gpu_dispatch_gemm — READY for batched GEMV
- **File:** `src/gpu/ops.rs:1431`
- **For Q4_0, seq_len ≤ 8:** Dispatches `gemv_q4_0_f32_batched_kernel` or `gemv_q4_0_f32_batched_tiled_kernel`
- **LDS check for lm_head (in_dim=3584):**
  - `batch × (3584/32) × 34 = batch × 3808`
  - batch=2: 7,616 B ✓
  - batch=4: 15,232 B ✓
  - batch=8: 30,464 B ✓
  - batch=9: 34,272 B > 32 KB → falls back to tiled kernel ✓
- **All batch sizes fit** in standard or tiled GEMV. No code changes needed.

### 4. argmax_f32 — NOT batched
- **File:** `src/gpu/kernels/elementwise.rs:206`
- **Signature:** `argmax_f32(input, partial_values, partial_indices, output_index, n)`
- **Semantics:** Single reduction over `n` elements → 1 index
- **Plan:** Call N times with offset pointers into the batched logits buffer. Each call operates on `logits_batch[i * vocab_size .. (i+1) * vocab_size]`.
- **Scratch needed:** Can reuse single `argmax_partial_values` / `argmax_partial_indices` since calls are sequential on the same stream.
- **Output:** Need N output slots. Options:
  - (a) Allocate `argmax_result_device` as `MAX_SPEC_DEPTH+1` i32s (36 bytes)
  - (b) Loop argmax N times writing to same slot, D2H copy each → same as current but fewer syncs
  - **Decision:** Option (a) — allocate result array of MAX_SPEC_DEPTH+1 slots, batch the D2H copy, sync once.

### 5. gpu_greedy_argmax_token — wraps argmax + D2H + sync
- **File:** `src/gpu/forward.rs:298-308`
- **Current:** `argmax(logits) → D2H copy → sync → read pinned host buffer`
- **Batched version:** Loop argmax N times (2 dispatches each), then single D2H copy of N results, then single sync.

### 6. Scratch buffers — NEED EXTENSION

**GpuForwardScratch** (`src/gpu/cache.rs:336`):
- `normed`: `hidden_size` f32s — need `MAX_BATCH × hidden_size`
- `logits`: `vocab_size` f32s — need `MAX_BATCH × vocab_size`
- `argmax_result_device`: 1 × i32 — need `MAX_BATCH` × i32
- `argmax_result_index` (pinned host): 1 × i32 — need `MAX_BATCH` × i32

**New allocations for MAX_SPEC_DEPTH=8 (MAX_BATCH=9):**
- `normed_batch`: 9 × 3584 × 4 = 129,024 B (~126 KB)
- `logits_batch`: 9 × 152064 × 4 = 5,474,304 B (~5.22 MB)
- `argmax_result_device_batch`: 9 × 4 = 36 B
- `argmax_result_index_batch` (pinned): 9 × 4 = 36 B
- **Total: ~5.35 MB** (user spec says 4.86 MB — difference is normed_batch)

**Alternative:** Reuse `prefill.normed` which is already sized for `seq_len × hidden_size` (allocated as `GpuPrefillScratch::new(config, spec_depth+1)`). The prefill scratch is available in `gpu_verify_forward`. This avoids a new `normed_batch` allocation.

**Decision:** Add `logits_batch` + `argmax_batch_device` + `argmax_batch_host` to `GpuForwardScratch`. Reuse `prefill.normed` for batched norm output.

### 7. CPU fallback path
- **Lines 1282-1296:** Falls back to CPU GEMV if GPU returns `InvalidWeightLayout` or `UnsupportedWeightType`
- **Must preserve:** If the first position fails with these errors, fall back to fully sequential CPU path for all positions
- **Approach:** Try batched GPU path first. On unsupported weight type, fall back to sequential loop (current code).

### 8. DISABLE flag
- **Pattern:** `ROCMFORGE_DISABLE_BATCHED_LM_HEAD=1` (opt-out, default enabled)
- **File:** `src/gpu/safety.rs` — add `CachedEnvFlag` following `DISABLE_TILED_GEMV` pattern
- **Initially:** Flag disabled by default (batched path ON) — per user spec "Phase 1 … Flag auf OFF (= batched aktiv)"

### 9. CLI validation
- **File:** `src/main.rs:112-118`
- **Add:** `if spec_depth > 8 { eprintln!("--spec-depth max is 8"); usage(); }`
- **Constant:** `pub const MAX_SPEC_DEPTH: usize = 8;` in `src/gpu/forward.rs` or `src/gpu/mod.rs`

## Scratch Buffer Layout (Batched)

```
logits_batch:  [pos0: vocab_size f32s][pos1: vocab_size f32s]...[pos8: vocab_size f32s]
               Total: 9 × vocab_size × 4 bytes

argmax_device: [idx0: i32][idx1: i32]...[idx8: i32]
               Total: 9 × 4 bytes

argmax_host:   [idx0: i32][idx1: i32]...[idx8: i32]  (pinned)
               Total: 9 × 4 bytes
```

## Batched lm_head Pseudocode

```rust
const MAX_SPEC_DEPTH: usize = 8;
const MAX_VERIFY_BATCH: usize = MAX_SPEC_DEPTH + 1; // 9

fn gpu_verify_lm_head_batched(
    device: &GpuDevice,
    gpu_weights: &GpuModelWeights,
    prefill: &GpuPrefillScratch,
    decode_scratch: &mut GpuForwardScratch,
    config: &ModelConfig,
    n: usize,  // number of positions
) -> GpuResult<Vec<u32>> {
    let h = config.hidden_size;
    let v = config.vocab_size;

    // Step 1: Batched RMS norm — all N positions at once
    // Input: prefill.hidden (n × h), Output: prefill.normed (n × h)
    rms_norm_batched(
        prefill.hidden.as_ptr() as *const f32,
        gpu_weights.output_norm.as_ptr() as *const f32,
        prefill.normed.as_ptr() as *mut f32,
        h,
        config.rms_norm_eps,
        n,
    )?;

    // Step 2: Batched GEMV — all N positions at once
    // Input: prefill.normed (n × h), Output: logits_batch (n × v)
    gpu_dispatch_gemm(
        device,
        &gpu_weights.lm_head,
        &gpu_weights.lm_head_meta,
        prefill.normed.as_ptr() as *const f32,
        decode_scratch.logits_batch.as_ptr() as *mut f32,
        v,
        h,
        n,  // seq_len = batch size
    )?;

    // Step 3: Argmax per position (still N dispatches, but no sync between)
    for i in 0..n {
        let logits_row = unsafe {
            (decode_scratch.logits_batch.as_ptr() as *const f32).add(i * v)
        };
        let result_slot = unsafe {
            (decode_scratch.argmax_batch_device.as_ptr() as *mut i32).add(i)
        };
        argmax_f32(
            logits_row,
            decode_scratch.argmax_partial_values_mut_ptr(),
            decode_scratch.argmax_partial_indices_mut_ptr(),
            result_slot,
            v,
        )?;
    }

    // Step 4: Single D2H copy of all N results + single sync
    unsafe {
        ffi::hip_memcpy_d2h_async(
            decode_scratch.argmax_batch_host.as_ptr(),
            decode_scratch.argmax_batch_device.as_ptr(),
            n * std::mem::size_of::<i32>(),
            device.stream(),
        )?;
    }
    device.synchronize()?;

    // Step 5: Read results from pinned host buffer
    let results = decode_scratch.argmax_batch_host.as_slice::<i32>();
    Ok((0..n).map(|i| results[i] as u32).collect())
}
```

## Dispatch Count Comparison

| Component | Sequential (per pos) | Batched (total) | Savings at n=2 | Savings at n=4 |
|-----------|---------------------|-----------------|----------------|----------------|
| rms_norm  | 1 × n               | 1               | -1 dispatch    | -3 dispatches  |
| GEMV      | 1 × n               | 1               | -1 dispatch    | -3 dispatches  |
| argmax    | 2 × n               | 2 × n           | 0              | 0              |
| sync      | 1 × n               | 1               | -1 sync        | -3 syncs       |
| **Total** | **4n + n syncs**     | **2 + 2n + 1 sync** | **-2 disp, -1 sync** | **-6 disp, -3 sync** |

## Files to Modify

| File | Change |
|------|--------|
| `src/gpu/cache.rs` | Add `logits_batch`, `argmax_batch_device`, `argmax_batch_host` to `GpuForwardScratch::new()` |
| `src/gpu/forward.rs` | Add `MAX_SPEC_DEPTH` const, `gpu_verify_lm_head_batched` fn, replace sequential loop in `gpu_verify_forward` |
| `src/gpu/safety.rs` | Add `DISABLE_BATCHED_LM_HEAD` flag |
| `src/gpu/mod.rs` | Export new flag/const |
| `src/main.rs` | Add `--spec-depth` max validation |
| `CLAUDE.md` | Add `ROCMFORGE_DISABLE_BATCHED_LM_HEAD=1` flag |

## Abort Criteria

- If `gpu_dispatch_gemm` returns `InvalidWeightLayout` or `UnsupportedWeightType` for the batched call → fall back to sequential loop (preserves correctness for non-Q4_0 models)
- If LDS calculation fails for batch=9 at lm_head dims → tiled GEMV handles it automatically
- If any argmax call fails → propagate error (same as current behavior)

## Validation Plan (Phase 2)

1. **Token-identical test:** Run verify with batched vs sequential (via DISABLE flag), assert identical token sequences for multiple prompts
2. **Performance sweep:** depth=1,3,5 with SPEC_PROFILE timing, compare non-layer overhead
3. **Verify breakdown:** Confirm reduced dispatch count in lm_head phase
