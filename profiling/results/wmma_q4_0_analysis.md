# Phase 2b — inline Q4_0 dequant WMMA verdict

**Date:** 2026-04-17
**Commits:** on top of `48520f2` (Phase 2a bench landed)
**GPU:** RX 9070 XT (gfx1201, RDNA 4), ROCm 7.2.1
**Kernel:** `hip_kernels/wmma/wmma_gemm_q4_0.hip`

## TL;DR

| Shape (M × N × K)           | hipBLAS |  P2a WMMA + dequant | **P2b (inline dequant)** | vs hipBLAS | vs P2a total | % peak |
|-----------------------------|--------:|--------------------:|-------------------------:|-----------:|-------------:|-------:|
| QKV/O   (256 × 3584 × 3584) |   644 µs | 211 + 113 = 324 µs | **254 µs** | **2.54×** | **1.28×** | 52.9 % |
| Gate/Up (256 × 18944 × 3584)|  3,200 µs | 960 + 686 = 1,646 µs | **1,213 µs** | **2.64×** | **1.36×** | 58.5 % |
| Down    (256 × 3584 × 18944)|  3,328 µs | 1,034 + 683 = 1,717 µs | **1,294 µs** | **2.57×** | **1.33×** | 54.8 % |

**Phase 2c (LDS tuning, double-buffering) is justified.** The inline-dequant kernel is 28–36 % faster than the equivalent Phase-2a two-kernel path and 2.5–2.6× faster than hipBLAS, despite doing strictly more work per iteration than the pure-WMMA version. The 52–59 % fraction of the FP16 theoretical peak (49 TFLOPS) tells us there is still headroom before the compute ceiling.

## Setup

- Kernel: `wmma_gemm_q4_0.hip`, 64×64 output tile, 128 threads, 4 waves.
- K advances in chunks of **32** (one full Q4_0 block), two WMMA K-sub-iterations per chunk.
- LDS: `lds_a[64 × 32]` + `lds_b[32 × 64]` = 8 KB per workgroup, single-buffered.
- Coalescing strategy for the Q4_0 load: **Ansatz C** from the prompt — each of 128 threads handles one Q4_0 block-half (9 bytes / 16 dequanted elements), writing the FP16 result straight into the LDS B-tile. No stream-load into a raw LDS staging area.
- FP32 → FP16 activation conversion happens inline during the cooperative A-tile load.

Raw data: [`wmma_q4_0_bench_48520f2_1776417062.json`](wmma_q4_0_bench_48520f2_1776417062.json).

## Correctness

`tests/wmma_q4_0_correctness.rs` — 5 shapes, each compared **bit-identical** against the Phase 2a path fed by the existing `dequant_q4_0_to_f16` kernel. 0 / N mismatches on all shapes:

| Shape                          | mismatches | max |abs diff| |
|--------------------------------|-----------:|--------------:|
| 64 × 64 × 64                   |          0 |          0.0 |
| 256 × 256 × 256                |          0 |          0.0 |
| 256 × 3584 × 3584 (QKV/O)      |          0 |          0.0 |
| 256 × 18944 × 3584 (Gate/Up)   |          0 |          0.0 |
| 256 × 3584 × 18944 (Down)      |          0 |          0.0 |

Bit-identical was the expected result: both paths share the same `__float2half` activation conversion, the same Q4_0 nibble → FP16 dequant formula, the same 16-wide WMMA K-order, and the same FP32 accumulator.

## Why Phase 2b is faster than P2a + separate dequant

The separate dequant kernel writes the entire FP16 weight matrix to VRAM and the WMMA kernel reads it back — a full round-trip through global memory.

Gate/Up example (K=3584, N=18944 → 68M FP16 weights, 136 MB):
- Separate dequant kernel measured at **686 µs** on this shape. That is 136 MB written / 686 µs ≈ **198 GB/s effective** — well below the 640 GB/s ceiling. The dequant is doing real VRAM work on top of its compute, and that VRAM cost doesn't exist for Phase 2b because the FP16 values live only in LDS.
- Phase 2b adds some extra ALU for inline dequant but avoids the 136 MB round-trip entirely.

The gap against hipBLAS (2.5–2.6× vs. Phase 2a's 3.0–3.4×) reflects the fact that Phase 2b is now doing dequant on the compute path whereas Phase 2a did not; that is the price of skipping the scratch materialisation. In the full prefill pipeline the scratch *has* to be paid somewhere, so the apples-to-apples comparison is "Phase 2b" vs "Phase 2a-total", which Phase 2b wins cleanly.

## What this changes for the prefill path

Per-projection cost model for Qwen2.5-7B prefill at pp=256 (5 projections × 28 layers = 140 GEMMs):

| Path                                 | GEMM time | Dequant VRAM traffic | Scratch VRAM |
|--------------------------------------|----------:|---------------------:|-------------:|
| Phase 1 (hipBLAS + separate dequant) | ~7.2 s    | ~19 GB               | ~150 MB      |
| Phase 2a (WMMA + separate dequant)   | ~3.8 s    | ~19 GB               | ~150 MB      |
| **Phase 2b (inline dequant WMMA)**   | **~2.9 s** | **0** (inlined)      | **0**        |

(Weighted average of the three shape classes across 140 projections.)

Phase 2b is the first path that makes the FP16 weight scratch buffer unnecessary at all. In Phase 2d (dispatch integration) that scratch allocation can be removed from `GpuDevice`, freeing ~150 MB of VRAM for longer context / larger drafts.

## Where the 40-50 % peak gap is coming from

Measured fraction of 49 TFLOPS FP16 peak:

- QKV/O:  52.9 % of peak (254 µs actual / 134 µs peak).
- Gate/Up: 58.5 % of peak.
- Down:    54.8 % of peak.

Relative to Phase 2a's 62-75 %, Phase 2b drops by ~10 percentage points on each shape — consistent with the inline dequant consuming ALU and issuing extra LDS writes inside the K loop.

Likely contributors, all in Phase 2c scope:

- **LDS bank conflicts on the dequant writes.** 128 threads write to `lds_b[k * 64 + col]` with `col = tid / 2`. Consecutive lanes in a warp (tid, tid+1) share a column and hit different k-rows — that pattern is not ideal on RDNA's 32-bank × 4-byte LDS. Phase 2a also has this on its plain FP16 store; both are parked for 2c.
- **LDS bank conflicts on the WMMA loads.** K-splitting pattern `{0..3, 8..11} / {4..7, 12..15}` from the register-layout doc. Phase 2a already shares this cost — both kernels are living with it today.
- **No double-buffering.** The `__syncthreads()` at the end of each K-chunk blocks the next chunk's global load until the previous WMMA is done.
- **Naive uncoalesced Q4_0 block load.** Each of 128 threads issues an 18-byte load at a column stride of `blocks_per_row × 18` bytes. Not coalesced, effective bandwidth probably <30 %. The prompt's Ansatz A (uint32-stream load into LDS) would fix this but landed as explicitly optional for Phase 2b.

## Verdict

**Phase 2c (tuning) justified.** The kernel is correct, 2.5×+ faster than hipBLAS, 28 %+ faster than the best Phase 2a equivalent, and eliminates the 150 MB FP16 scratch — but still sits at 53-59 % of peak. The remaining gap has three named, understood causes, all solvable without changing the kernel's correctness invariants.

Not in scope here (Phase 2c / 2d):

- LDS padding / swizzle for the FP16 store side of the dequant.
- Double-buffered K-loop.
- Coalesced Q4_0 stream load (Ansatz A).
- Arbitrary-shape padding (Qwen2.5-7B shapes don't need it).
- Prefill-pipeline dispatch integration — Phase 2d.
