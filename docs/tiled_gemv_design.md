# Tiled Batched GEMV Design

**Date:** 2026-04-16
**Branch:** feat/gqa-tiled-attention-experimental
**Target:** Eliminate sequential GEMV fallback for FFN-intermediate projections in speculative decode verify

## Problem

The existing `q4_0_gemv_batched` kernel caches **all** Q8-quantized input vectors in LDS:

```
LDS = batch_size × (in_dim / 32) × 34 bytes
```

For Qwen2.5-7B FFN (in_dim=3584, out_dim=18944) at depth=1 (batch=2):
- `2 × (3584/32) × 34 = 7,616 bytes` — fits in 32 KB, **batched path used**

For FFN down-projection (in_dim=18944, out_dim=3584) at depth=1 (batch=2):
- `2 × (18944/32) × 34 = 40,256 bytes` — **exceeds 32 KB, falls back to sequential**

The sequential fallback loads the full weight matrix N times (once per batch element), negating the core advantage of batched GEMV. Profiling shows target_verify takes 17.2 ms per spec step — 41% above single-token baseline (12.2 ms).

**Which projections overflow?** Per 7B verify layer (28 layers, depth=1, batch=2):
| Projection | in_dim | out_dim | LDS bytes | Fits? |
|------------|--------|---------|-----------|-------|
| Q proj | 3,584 | 3,584 | 7,616 | yes |
| K proj | 3,584 | 512 | 7,616 | yes |
| V proj | 3,584 | 512 | 7,616 | yes |
| O proj | 3,584 | 3,584 | 7,616 | yes |
| Gate | 3,584 | 18,944 | 7,616 | yes |
| Up | 3,584 | 18,944 | 7,616 | yes |
| Down | 18,944 | 3,584 | 40,256 | **NO** |
| LM head | 3,584 | 151,936 | 7,616 | yes |

Only the **FFN down-projection** overflows — but it's called 28 times per verify pass. At depth=3 (batch=4), gate/up also overflow: `4 × 112 × 34 = 15,232` — still fits. At depth=5 (batch=6): `6 × 112 × 34 = 22,848` — fits. Down at batch=4: `4 × 592 × 34 = 80,512` — far exceeds 32 KB.

## Design: Tiled Input Caching

### Core Idea

Instead of caching the entire Q8-quantized input in LDS, tile along the **input dimension** (reduction axis). Each tile loads a slice of the input into LDS, computes partial dot products against weight blocks in that slice, and accumulates into per-thread FP32 registers.

```
for tile in 0..num_tiles:
    quantize input[tile_start..tile_end] → Q8 in LDS     // batch × tile_blocks × 34 bytes
    for each output column (wave-parallel):
        for block in tile_start..tile_end:
            acc[batch][col] += weight[col][block] · input_q8[batch][block]
```

Weight blocks are still streamed from VRAM — one load per block, N dot products. The key insight: **weight bandwidth is unchanged**, we just break the input quantization into phases.

### 1. Tile Dimensioning

**Target: >= 2 resident workgroups per WGP (RDNA 4) → ≤ 16 KB LDS per workgroup.**

LDS per tile = `batch_size × tile_blocks × sizeof(Q8_0_block_batched)`
where `sizeof(Q8_0_block_batched) = 34` (2-byte half scale + 32 int8 quants).

With a tile of 256 elements (8 Q8 blocks of 32 each, `tile_blocks=8`):

| batch_size | LDS per tile | Fits 16 KB? |
|------------|-------------|-------------|
| 2 (depth=1) | 2 × 8 × 34 = 544 B | yes |
| 4 (depth=3) | 4 × 8 × 34 = 1,088 B | yes |
| 6 (depth=5) | 6 × 8 × 34 = 2,040 B | yes |
| 8 (max) | 8 × 8 × 34 = 2,176 B | yes |

This is extremely conservative. We can increase the tile to 1024 elements (32 blocks):

| batch_size | LDS per tile (1024 elems) | Fits 16 KB? |
|------------|---------------------------|-------------|
| 2 | 2 × 32 × 34 = 2,176 B | yes |
| 6 | 6 × 32 × 34 = 6,528 B | yes |
| 8 | 8 × 32 × 34 = 8,704 B | yes |

**Chosen tile size: `TILE_K = 1024` elements (32 Q8 blocks).**

At batch=8 this uses 8,704 bytes — well under 16 KB, leaving room for registers and achieving >= 2 workgroups/WGP.

For FFN down (in_dim=18,944): `num_tiles = ceil(18944 / 1024) = 19 tiles`.

### 2. Double-Buffering Strategy

**Decision: No double-buffering for Phase 1.**

Double-buffering overlaps tile N+1 load with tile N compute. But in this kernel, the "load" is actually a **quantization** step (f32 → Q8 in LDS), not a simple memcpy. The quantization requires:
1. Loading f32 values from global memory
2. Warp-reduction to find absmax
3. Scale computation + int8 quantization
4. LDS store

This is ALU-heavy, not a pure memory stall — double-buffering provides limited overlap. The barrier between tiles is also unavoidable because all threads must see the new Q8 data before computing.

The compute phase (Q4×Q8 dot products) is also memory-bound (streaming weights from VRAM). Overlapping two memory-bound phases doesn't help — the memory controller is saturated either way.

**If profiling shows the tile-quantize → compute serialization is a bottleneck, double-buffering can be added later with 2× LDS reservation (still within budget at ~17 KB for batch=8).**

### 3. Q4_0 Dequant Handling

**Decision: Keep Q4_0 weights in packed format, dequant during dot product (same as existing kernel).**

Trade-off analysis:

| Strategy | LDS footprint | Dequant cost | Bank conflicts |
|----------|---------------|-------------|----------------|
| **Q4_0 in registers, Q8 in LDS** (current) | Q8 only | Once per weight block per batch element | None (Q8 sequential) |
| FP16 dequant to LDS | 2× weight tiles in LDS | Once per weight block total | Possible (strided access) |
| FP32 dequant to LDS | 4× weight tiles in LDS | Once per weight block total | Possible |

The existing kernel streams Q4_0 weight blocks from VRAM directly into registers, dequantizes inline via `q4_0_q8_0_dot()`, and computes against Q8 data in LDS. This is the right approach because:

1. **Weight data is NOT re-read** — each weight block is loaded once from VRAM and used for all N batch elements. The dequant cost (unpacking nibbles, subtracting 8) is pure ALU and happens in-register.
2. **LDS is used exclusively for input data** — the tiled approach only needs to fit `batch × tile_blocks × 34` bytes, which is tiny.
3. **No bank conflicts** — Q8 blocks are accessed sequentially by block index, and all lanes in a warp access the same block (broadcast).

Putting dequantized weights in LDS would double LDS pressure for no bandwidth savings, since the weights are already streaming at full memory bandwidth from VRAM.

### 4. Dispatch Thresholds

The dispatch logic in `src/gpu/ops.rs` currently checks:

```rust
let lds_bytes = seq_len * (in_dim / 32) * 34;
if lds_bytes <= 32 * 1024 { /* batched kernel */ }
else { /* sequential fallback */ }
```

New dispatch (under `ROCMFORGE_EXPERIMENTAL_TILED_GEMV=1`):

```rust
const BATCHED_GEMV_LDS_LIMIT: usize = 32 * 1024;
const TILED_GEMV_MAX_BATCH: usize = 8;

let lds_bytes = seq_len * (in_dim / 32) * 34;

if lds_bytes <= BATCHED_GEMV_LDS_LIMIT {
    // Original batched kernel — proven, no tiling overhead
    gemv_q4_0_f32_batched_on_stream(...)
} else if seq_len <= TILED_GEMV_MAX_BATCH && tiled_gemv_enabled() {
    // Tiled batched kernel — for large in_dim (FFN down)
    gemv_q4_0_f32_batched_tiled_on_stream(...)
} else {
    // Sequential fallback
    for row in 0..seq_len { gemv(...) }
}
```

**Threshold constant:** `BATCHED_GEMV_LDS_LIMIT = 32 * 1024` (existing). The tiled kernel activates for any Q4_0 batched GEMV that exceeds this limit, gated by the feature flag.

### 5. Kernel Launch Configuration

Matching the existing kernel's structure:
- **Block size:** 512 threads (8 warps × 64 threads/warp on RDNA 4)
- **Columns per wave:** 4 (BATCHED_COLS)
- **Grid:** `ceil(out_dim / (8 warps × 4 cols))` blocks
- **LDS:** Dynamic, `batch_size × tile_blocks × 34` bytes per tile

The only structural change vs. the existing kernel is the outer tile loop around Phase 1 (quantize) and Phase 2 (compute). Phase 3 (warp reduction + output) remains identical.

### 6. Expected Performance

**FFN down-projection (18944 → 3584), batch=2:**

Current (sequential): 2 × separate GEMV launches. Each streams 18944 × 3584 / 32 × 18 = ~38.5 MB of weights. Total weight reads: **77 MB**.

Tiled batched: 1 launch, streams weights once: **38.5 MB**. 19 tiles of input quantization, each ~2 KB of work. Expected **~1.8-1.9×** speedup (less than 2× due to tile quantization overhead and barrier costs).

Over 28 layers: saves ~28 × (one FFN down sequential overhead) ≈ **2-4 ms** from target verify, bringing it from 17.2 ms toward 13-15 ms.

## Implementation Decisions (Post-Review)

- **Tile size**: 1024 (32 Q8 blocks). Fewer barriers, LDS budget comfortable.
- **Launch bounds**: `__launch_bounds__(512, 2)` — two resident workgroups per CU.

## Measured Results (2026-04-16, RX 9070 XT)

### depth=1 (batch=2)

| Prompt | Verify Baseline (μs) | Verify Tiled (μs) | Delta |
|--------|---------------------|-------------------|-------|
| code_01 | 17,265 | 17,007 | -258 μs (-1.5%) |
| chat_01 | 17,193 | 16,916 | -277 μs (-1.6%) |
| prose_03 | 17,192 | 17,027 | -165 μs (-1.0%) |

### depth=3 (batch=2-4, adaptive)

| Prompt | Verify Baseline (μs) | Verify Tiled (μs) | Delta |
|--------|---------------------|-------------------|-------|
| prose_03 | 21,839 | 20,386 | **-1,453 μs (-6.7%)** |
| chat_01 | 17,907 | 17,699 | -208 μs (-1.2%) |

At depth=3, steps with effective batch=3-4 see a larger improvement because the
sequential fallback does 3-4× weight loads vs. the tiled kernel's single load.

### Key finding: memory-controller pipelining

The sequential fallback is not as expensive as predicted. The RDNA 4 memory controller
pipelines back-to-back GEMV launches, maintaining near-full bandwidth utilization across
kernel boundaries. See `docs/batched_verify.md` for the full analysis.

The tiled kernel provides a consistent ~1-2% verify improvement at depth=1 and up to
~7% at higher depths where the sequential fallback multiplier is larger.
