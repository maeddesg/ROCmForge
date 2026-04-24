# llama.cpp vs ROCmForge Q4_K GEMV Kernel Analysis

Date: 2026-04-24
Hardware: RX 9070 XT (gfx1201, RDNA4), 640 GB/s nominal BW
ROCmForge baseline: 62.7 tok/s @ 51% BW on `gemv_q4_k_q8_inline`
llama.cpp baseline: 99.3 tok/s @ ~70% BW on `mul_mat_vec_q<GGML_TYPE_Q4_K>`

---

## Summary

The two kernels are **structurally different in three compounding ways**, not one. ROCmForge assigns **one thread to one full 256-element Q4_K super-block** and does byte-wise scalar int MACs; llama.cpp assigns **one warp-group-of-16-threads to one super-block**, each thread consuming a 2-block slice (VDR=2) with `dp4a` over int32-aligned loads. The compute savings of `dp4a`/`sudot4` we already measured in isolation were a no-op because the bottleneck is the **memory-request rate** per lane-iteration, not ALU throughput. llama.cpp issues ~16× more outstanding 4-byte loads per warp-cycle than we do because 16 lanes cooperate on the same super-block at int32 granularity. That — plus the fact that the Q8_1 side already carries a pre-computed `d` delivered per bq8_1 and the scale/min compute is amortized over 4 dp4a pairs — is what lifts BW from 51 % to 70 %.

**Recommendation: Option A** — port the llama.cpp MMVQ kernel 1:1 as a new Bandit variant, with a dedicated Q8_1 activation quantizer. Incremental alignment of our kernel (Option B) does not land the cooperative 16-lane-per-super-block pattern without rewriting the outer loop anyway; at that point Option A is simpler.

**LOC/time estimate:** ~500 LOC HIP + ~150 LOC Rust FFI + ~80 LOC Q8_1 quantizer + ~200 LOC tests ≈ 5–8 working days. Risk LOW — llama.cpp demonstrably runs it on the same hardware and reaches 70 % BW today.

---

## Full Comparison Table

| Dimension | llama.cpp (file:line) | ROCmForge (file:line) | BW-relevance |
|---|---|---|---|
| Threads per block | `warp_size * nwarps = 32 * 8 = 256` (mmvq.cu:664, calc_nwarps=8 for RDNA4 Q4_K ncols_dst=1 at mmvq.cu:328-344) | `Q4_K_Q8_FIXED_WAVES * WARP_SIZE = 8 * 32 = 256` (gemv_q4_k_q8_inline.hip:22) | Identical nwarps — this knob is not the difference |
| Warps per block | 8 (mmvq.cu:341) | 8 (gemv_q4_k_q8_inline.hip:21) | Identical |
| Threads per row (one output column) | `nwarps*warp_size / (qi/vdr) = 256/16 = 16` cooperating threads (mmvq.cu:414, 483) | `warp_size = 32` threads (one warp per 4-column group); within a warp a single lane owns the whole super-block (gemv_q4_k_q8_inline.hip:175 `for (int sb = lane_id; ...)`) | **KEY DIFFERENCE**. llama.cpp gives 16 threads to *one* super-block and loads it in 16 × int32 parallel; we give 32 lanes to 4 columns with each lane touching a full super-block serially |
| Rows per block (output rows) | `rows_per_cuda_block = 1` for ncols_dst=1 on RDNA4 (mmvq.cu:373-386, calc_rows_per_block with ncols_dst=1 returns 1; small_k branch omitted by `GGML_CUDA_CC_IS_RDNA` at mmvq.cu:778) | 4 output columns per warp (`Q4_K_Q8_COLS=4`, gemv_q4_k_q8_inline.hip:20, col_base at :149) | We amortize activation reads over 4 weight columns; llama.cpp runs ncols_dst=1 at nwarps=8 (row parallelism via grid.x) |
| VDR | 2 (vecdotq.cuh:501) | effectively 1 (one super-block per lane per iteration at gemv_q4_k_q8_inline.hip:175, 1-byte granularity at :60-66) | VDR=2 doubles outstanding loads per lane → fewer iteration overhead, better MLP |
| Q8 input format | Q8_1: `{d, s, qs[32]}` with s = sum(qs) (ggml-common.h:248-258; quantize.cu:36,47) | Q8_0: `{d, qs[32]}` (gemv_q4_k_q8_inline.hip:26-29) | Q4_K still computes q8_sum per sub-block in the MMVQ path — so for Q4_K specifically Q8_1.s is **unused** (vecdotq.cuh:518). But see dp4a note. |
| Q8 quantization location | Pre-kernel, separate CUDA kernel `quantize_q8_1` writing to a pool buffer (mmvq.cu:1091-1098; quantize.cu:4-48) | Cooperative, in-kernel, to LDS (gemv_q4_k_q8_inline.hip:104-134, called at :158) | Ours re-quantizes per-call; llama.cpp quantizes once per forward pass and reuses across all Q4_K matmuls. Saves `n_layers × n_matmuls_per_layer` redundant quantizations per token. |
| Inner-loop structure | `for kbx = tid/16; kbx < blocks_per_row; kbx += blocks_per_iter=16` — each lane-iteration loads 2 int32 (=8 nibble-pairs) from one 144-byte super-block (mmvq.cu:483-503, impl_vmmq at vecdotq.cuh:505-527) | `for sb = lane_id; sb < super_blocks; sb += 32` — each lane-iteration processes one full 144-byte super-block via 8 sub-blocks × 32 byte-wise MACs (gemv_q4_k_q8_inline.hip:175-184, sub-block at :59-66) | llama.cpp: 16 threads × 8 dp4a per iter = 128 byte-dots in parallel. Ours: 1 thread × 256 byte-dots serial. Same byte-count, but llama.cpp exposes ~16× more outstanding memory transactions. |
| Dot-product instruction | `ggml_cuda_dp4a` → `__builtin_amdgcn_sudot4` on RDNA4 (common.cuh:672-677) | scalar int8 MACs in a #pragma unroll 32 loop (gemv_q4_k_q8_inline.hip:60-66) | Relevant only indirectly: dp4a lets the compiler fit more logical 8-bit dots per lane without dedicating registers to 32 expanded int8 values |
| Nibble extraction | int32 aligned load then `(v[i] >> (4*i)) & 0x0F0F0F0F` — 4 nibbles per instr (vecdotq.cuh:514-515) | byte-wise `is_upper ? (byte>>4) : (byte & 0x0F)` per element (gemv_q4_k_q8_inline.hip:62) | int32 path is 1 global load per 4 weights; byte-wise relies on compiler widening. Measured: compiler *does* widen, but pattern still loads 4 separate bytes per unrolled iteration. |
| Scale/min handling | 6-bit packed via `uint16_t aux[2]` (so 16-byte loads of scales), one `dm` half2 load per super-block, `dm4f = __half22float2(dm4)` (vecdotq.cuh:885-896, 524) | `unpack_scale_min(j, scales, ...)` with four byte-loads per sub-block; two `__half` loads for d/dmin (gemv_q4_k_q8_inline.hip:31-41, 77-80) | llama.cpp hoists the scale/min unpack above the 4 dp4a pairs (QR4_K=2 iterations × VDR=2); ours re-derives scale/min inside the sub-block loop 8× per super-block |
| Reduction | `__shared__ float tmp_shared[nwarps-1][ncols_dst][rows_per_cuda_block][warp_size]`, lane-0 warp finishes (mmvq.cu:505-547) | `__shared__` not used for reduction; `__shfl_down` warp-reduce per column, lane-0 writes (gemv_q4_k_q8_inline.hip:186-200) | Different but both efficient. Ours is marginally lighter (no shared-mem buffer) but llama.cpp's 8-warp partial-sum tree is well-pipelined. |
| Output write | `dst[j*stride_col_dst + threadIdx.x] = result` (mmvq.cu:584) | `output[col_base + c] = sums[c]` (gemv_q4_k_q8_inline.hip:197) | Equivalent |
| Gate-fusion support | Template param `has_fusion`; loads gate in parallel via separate vec_dot invocation on `vgate` (mmvq.cu:437-502, 495-500) — activates SwiGLU/GeGLU in the write-back (mmvq.cu:567-581) | Not implemented; we have unfused q4_k_q8_inline × 2 (gate, up) + separate swiglu | See Q16/17 |
| Shared-memory layout | `tmp_shared[nwarps-1=7][ncols_dst=1][rows_per_cuda_block=1][32]` = 7×32×4 = 896 B reduction only (mmvq.cu:505) | `s_input_q8[n_blocks_total]` = Q8_0 activation blocks (up to ~32 KB at K=12288) (gemv_q4_k_q8_inline.hip:153, launch :214) | llama.cpp keeps LDS tiny and uses ALL bandwidth for weights; we burn ~9 KB LDS at K=4096 and ~25 KB at K=12288 on activations, pressuring occupancy at the larger shapes |
| Weight-loading pattern | int32 aligned via cast `(const int *)(bq4_K->qs + 16*bq8_offset + 4*((iqs/2)%4))` loading `q4[0]` and `q4[4]` — two 4-byte reads, naturally coalesced across 4 adjacent threads (16-byte transactions) (vecdotq.cuh:881-883) | byte pointer `qs_pair + i` loaded one byte at a time through the unroll (gemv_q4_k_q8_inline.hip:61) | llama.cpp: 16 threads × 2 int32 = **128 B coalesced transaction** per lane-group per iter. Ours: even if widened, the 4-column replication (`w_cols[c]`) means 4 independent base pointers and a single lane walking sequentially — transactions are serial per lane, not coalesced across lanes. |

---

## The 17 Analysis Questions

### Q1. block_q8_1 struct layout

```c
typedef struct {
    union {
        struct { ggml_half d; ggml_half s; } GGML_COMMON_AGGR_S;
        ggml_half2 ds;
    } GGML_COMMON_AGGR_U;
    int8_t qs[QK8_1];   // QK8_1 = 32
} block_q8_1;
// static_assert(sizeof(block_q8_1) == 2*sizeof(ggml_half) + QK8_1)
```
Source: `ggml/src/ggml-common.h:248-259`. Total **36 bytes** (4 bytes header `ds` + 32 bytes `qs`), 4-byte aligned. QK8_1=32, so one Q8_1 block covers 32 input activations.

Our equivalent `Q8_0_block_q4k_inline` (gemv_q4_k_q8_inline.hip:26-29) is **34 bytes** (2-byte `d` + 32 bytes `qs`) — we have **no `s` field**, so no pre-computed sum.

### Q2. Where Q8_1.s is computed

In `quantize.cu:32-47` (llama.cpp CUDA Q8_1 quantizer):
```c
float amax = fabsf(xi);
float sum = xi;
amax = warp_reduce_max<QK8_1>(amax);
sum  = warp_reduce_sum<QK8_1>(sum);
...
y[ib].ds = make_half2(d, sum);
```
It is computed **at input-quantization time**, one warp-reduce across the 32 block elements, stored once before any mat-vec kernel reads the block. Every downstream MMVQ/MMQ call reuses the same pre-computed sum.

### Q3. How `vec_dot_q4_K_q8_1` uses the sum — instructions saved

**For Q4_K specifically: the Q8_1 `s` field is NOT used.** Confirmed by `vec_dot_q4_K_q8_1_impl_vmmq` at vecdotq.cuh:505-527:
- Line 517: `dot1 = dp4a(v1i, u[2i+1], dp4a(v0i, u[2i+0], 0))` — weight-dot
- Line 518: `dot2 = dp4a(0x01010101, u[2i+1], dp4a(0x01010101, u[2i+0], 0))` — q8_sum via dp4a against 1s
- Line 858-900: only `d8[i] = __low2float(bq8i->ds)` is read, i.e. `ds.x` (the `d`), never `ds.y`.

Reason: Q4_K has **8 different per-sub-block 6-bit mins**, so the "subtract sum*min" correction has to run per sub-block, not per Q8_1 block. The pre-computed `s` doesn't align.

BUT: the `dot2 = dp4a(0x01010101, ...)` trick still saves enormously vs our kernel. Per sub-block, llama.cpp does:
- 2 dp4a for int_dot
- 2 dp4a for q8_sum (with 0x01010101)
= **4 dp4a instructions**, 4 cycles if pipelined.

Our kernel does a 32-iter unrolled loop with `int_dot += nib * x; q8_sum += x` (gemv_q4_k_q8_inline.hip:64-65) = **64 scalar ops + 32 nibble extracts** per sub-block. LLVM does widen and auto-vectorize (see `phase2_sudot4_kernel_upgrade.md:149-153`), but it cannot compress to 4 dp4a-equivalents because our *control flow and data layout* force byte-wise access.

**Instructions saved per sub-block (llama.cpp vs ours): ~16×** on ALU count — but that is NOT the BW lever. The BW lever is that the 4 dp4a's consume `u[0..3]` which is a **single int32 load per Q8 block chunk**, whereas our 32-iter loop implicitly walks `x_qs[0..31]` one byte at a time.

### Q4. Register impact: Q8_1 vs Q8_0

We cannot measure VGPR counts without compiling both kernels and inspecting `.s`. Rough estimate from the Rust source:

- ROCmForge: per sub-block the compiler allocates registers for `qs_pair[i]`, nib computation, `x_qs[i]`, `int_dot`, `q8_sum`. With `#pragma unroll 32` on gemv_q4_k_q8_inline.hip:59, this is aggressive — likely 48-64 VGPRs/lane. Combined with `Q4_K_Q8_COLS=4` columns each carrying accumulators, this sits at the high end of RDNA4's per-wave budget, triggering the observed `__launch_bounds__(256, 1)` (only 1 block per CU).
- llama.cpp: 16 threads share the super-block work. Each lane's live state is `v[2]`, `u[4]`, `d8[2]`, `sc[2]`, `m[2]`, `tmp[1][1] += partial` — very compact (~24-32 VGPRs/lane). The `__launch_bounds__(nwarps*warp_size, 1)` at mmvq.cu:392 with nwarps=8 also says 1 block per CU, but each *wave* is much smaller.

**Occupancy impact**: RDNA4 gfx1201 has 1536 VGPRs per SIMD. At 32 VGPRs/lane a wave uses 32*32=1024, allowing 1 wave/SIMD. At 64 VGPRs/lane a wave uses 2048 — overflow; 1 wave per 2 SIMDs (effectively half occupancy). Our per-thread 4-column replication is a register-pressure hazard; llama.cpp's 16-thread cooperation + ncols_dst=1 is not. This is a **BW-relevant** register-pressure difference: higher occupancy → more in-flight memory requests → better BW utilization.

Honest caveat: the above is an educated guess from source, not a measurement. Recommended concrete first step is to compile and extract `.amdgcn_target` metadata for both to confirm VGPR counts.

### Q5. VDR_Q4_K_Q8_1_MMVQ meaning

Defined at vecdotq.cuh:501: `#define VDR_Q4_K_Q8_1_MMVQ 2`.

VDR = "Value Dimension Ratio" = **how many Q8_1 blocks one thread consumes per inner-loop iteration**, expressed as a multiplier of the base lane unit (QI8_1/2 = 4 int32 elements).

In the outer loop at mmvq.cu:414:
```
blocks_per_iter = vdr * nwarps*warp_size / qi
                = 2  * 8     *32        / 32 = 16
```
So one iteration advances `kbx` by 16 Q4_K super-blocks (=16 × 256 = 4096 input elements). Since `nwarps*warp_size = 256` threads and each processes `qi/vdr = 16` bytes of kbx-stride, the threads 0..15 share kbx=0, threads 16..31 share kbx=1, …, threads 240..255 share kbx=15. **ILP: each lane issues 2 independent int32 weight loads (`v[0]=q4[0], v[1]=q4[4]`) and 4 independent int32 activation loads (`u[0..3]` via `q8[0]`, `q8[4]` per i in QR4_K=2)**, giving a per-lane outstanding-load count of 6 — vs our 1 (we load the super-block in a single serial pass per lane).

### Q6. Can VDR=2 be added to our kernel?

Yes, but only by first changing the thread-to-super-block mapping. The critical coupling:

- VDR=2 in our kernel would require one lane to process 2 super-blocks per iteration. That alone doesn't help because we already process all super-blocks serially per lane.
- To get the BW win, VDR=2 must be **combined** with cooperative lane-grouping per super-block — otherwise the extra loads per lane-iteration just pressure registers and don't parallelize.

Thread-mapping changes required:
1. Drop the 4-column replication (`Q4_K_Q8_COLS=4`); set ncols_dst=1 at compile time (match llama.cpp).
2. Partition 16 lanes to one super-block (qi/vdr = 16). Each lane owns one of 16 byte-offsets into the super-block's 128 nibble bytes.
3. Outer loop iterates super-blocks at `blocks_per_iter = 16` stride.

Register pressure impact: with 1 column (not 4) each lane drops 3 accumulators, gaining ~12-18 VGPRs. With 2 int32 weight + 4 int32 activation per iter, lane state is ~24 VGPRs. Net: **lower pressure, higher occupancy**.

### Q7. dp4a usage with Q8_1 — is `dp4a(0x01010101, q8, 0)` needed when Q8_1 has s?

For Q4_0 / Q4_1 / Q5_1 / Q8_1-target: `ds.y` is used directly (e.g. vecdotq.cuh:133, 162-166) and no `dp4a(0x01010101, ...)` is needed.

**For Q4_K: `dp4a(0x01010101, u, 0)` IS used** (vecdotq.cuh:518) despite Q8_1 carrying `s`, because Q4_K has 8 per-sub-block mins, one per 32-element sub-block. Q8_1's `s` covers a 32-element block too, BUT we need the sum *only over the 16-element half* that aligns with the nibble half (recall `v[0] >> 0` and `v[0] >> 4` pick different nibble halves). `dp4a(0x01010101, u_half, 0)` gives exactly the 4-byte sum of `u_half`, which is the 8-element sum — and actually the formula iterates QR4_K=2 so ends up with the full block sum, but split so the Q4_K per-sub-block min applies correctly.

So the answer is: **the Q8_1 `s` field saves a dp4a-sum for Q4_0/Q4_1/Q8_0, but is orthogonal to Q4_K's saving**. The Q4_K saving comes purely from **dp4a replacing 32 scalar byte-MACs with 2 dp4as**.

### Q8. Is dp4a the only reason for int32-aligned weight loads?

No. The int32 alignment at vecdotq.cuh:881-883 (`const int * q4 = (const int *)(bq4_K->qs + 16*bq8_offset + 4*((iqs/2)%4)); v[0] = q4[0]; v[1] = q4[4]`) comes first from the **cooperative tiling pattern**: 16 threads reading 16 adjacent int32s = 64-byte naturally-coalesced transaction. The dp4a consumption is downstream of that.

If you removed dp4a and kept int32-aligned loads with nibble unpacking (`(v[i] >> (4*j)) & 0x0F0F0F0F`), the compiler would have to re-split to 4 bytes anyway and probably re-emit the dp4a on RDNA4 (LLVM does this). The point is the **int32-aligned load lets 16 threads collaborate in one L2 transaction**, regardless of whether dp4a or 4 scalar byte-MACs follow.

### Q9. Porting Q8_1 + VDR=2 without dp4a — does coalescing survive?

Yes, **provided you keep the 16-thread-per-super-block mapping**. The coalescing is a property of the thread→memory mapping, not of the ALU op. Without dp4a, the compute latency hides behind the memory latency better (more ALU), but the BW achieved is the same.

Caveat: if the compiler can't fuse the 4 byte-MACs cleanly, there's more register pressure in the ALU phase, which *can* regress occupancy and *indirectly* reduce BW utilization. So dp4a is not mandatory for the BW win but it is a cheap latent win.

### Q10. llama.cpp thread→output mapping

For ncols_dst=1 on RDNA4 (the decode case):
- Grid: `(nrows/rows_per_block=nrows/1, nchannels, nsamples)` = `(n_rows, 1, 1)` in standard decode. One output row per block.
- Block: `(warp_size=32, nwarps=8, 1)` = 256 threads/block.
- Within a block, 256 threads split into 16 "super-block groups" of 16 threads each (since `qi/vdr=16`). Each group processes one super-block at a time; across the K dimension, the 16 groups interleave every 16 super-blocks.

Source: mmvq.cu:411-414, 483-487:
```c
const int tid = warp_size*threadIdx.y + threadIdx.x;  // 0..255
const int row0 = rows_per_cuda_block*blockIdx.x;      // one row per block
blocks_per_iter = vdr * nwarps*warp_size / qi = 16
for (int kbx = tid/(qi/vdr); kbx < blocks_per_row_x; kbx += blocks_per_iter) {
    const int kqs = vdr * (tid % (qi/vdr));  // 0..30 in steps of 2
```

ncols_y per block: 1 (ncols_dst=1 → each block computes 1 output column).
Columns split across warps: all 8 warps cooperate on the same output row, each warp holds 2 of the 16 super-block groups.
K-dim striping: super-block k is owned by group `k mod 16` (group = tid/16).

### Q11. Our thread→output mapping (gemv_q4_k_q8_inline.hip)

- Grid: `n_blocks_x = ceil(ncols_dst / (8 waves * 4 cols)) = ncols_dst / 32`. One block per 32 output columns.
- Block: 256 threads = 8 waves × 32 lanes.
- Each wave owns 4 output columns (`col_base = (blockIdx.x * 8 + wave_id) * 4`, line 149).
- Within a wave, 32 lanes stripe the K dimension: `for (int sb = lane_id; sb < super_blocks_per_row; sb += 32)` (line 175).
- Each lane processes 4 columns × all its assigned super-blocks serially.

So: llama.cpp gives 256 threads to **1** output column; we give 32 threads to **4** output columns. Roughly: llama.cpp is **32× more threads per output column**.

### Q12. Memory-coalescing delta

**llama.cpp**: 16 adjacent threads each load `v[0] = q4[0], v[1] = q4[4]` where their base pointers differ by 4 bytes each (`4 * ((iqs/2)%4)` increments by 4 with tid). So threads 0..3 inside a group load 4 adjacent int32 = one 16-byte coalesced transaction; across 16 threads there are 4 such coalesced transactions per super-block per iter = ~64 B total, then 4 more for v[1]. Equally, the Q8_1 activations are loaded `u[0..3]` as 4 int32 per thread from `(const int*)(bq8i->qs) + ((iqs/2)%4)`, also nicely coalesced.

**ROCmForge**: one lane reads a super-block sequentially: `qs_pair[i]` in an unroll. Across 32 lanes in the warp, each lane is reading a *different* super-block (lane_id stride), so **no coalescing across lanes within an iteration**. Each lane's 32 byte reads may compact into a 32-byte transaction per sub-block if the compiler re-widens, but the 32 different super-blocks across the warp produce 32 different cache lines — 32 separate L2 lookups per warp-iteration. This is the **primary BW-efficiency penalty**.

Order-of-magnitude: llama.cpp issues ~4 coalesced L2 transactions per super-block per iter; we issue ~32 scattered L1/L2 lookups. Even if the scatter hits in L2 cache, the tag-lookup throughput is higher.

### Q13. llama.cpp Q4_K scale/min handling

Source vecdotq.cuh:885-896, 524:
```c
const uint16_t * scales = (const uint16_t *)bq4_K->scales;
uint16_t aux[2];
const int j = bq8_offset/2;
if (j < 2) {
    aux[0] = scales[j+0] & 0x3f3f;
    aux[1] = scales[j+2] & 0x3f3f;
} else {
    aux[0] = ((scales[j+2] >> 0) & 0x0f0f) | ((scales[j-2] & 0xc0c0) >> 2);
    aux[1] = ((scales[j+2] >> 4) & 0x0f0f) | ((scales[j-0] & 0xc0c0) >> 2);
}
const uint8_t * sc = (const uint8_t *)aux;
const uint8_t * m  = sc + 2;
...
const float2 dm4f = __half22float2(dm4);
return dm4f.x*sumf_d - dm4f.y*sumf_m;
```

- `dm4 = bq4_K->dm` is a `half2 = {d, dmin}` — ONE 4-byte load per super-block.
- Scales read as `uint16_t` (2-byte aligned), at most 4 uint16 loads to build `aux[2]`.
- 6-bit unpack done via SIMD masks on int32 (0x3f3f, 0x0f0f, 0xc0c0) — no per-byte branches.
- Two float multiplies (`dm4f.x`, `dm4f.y`) applied ONCE at the very end on two scalar sums (`sumf_d`, `sumf_m`).

### Q14. Our scale/min handling

Source gemv_q4_k_q8_inline.hip:31-41, 77-99:
```c
void unpack_scale_min(int j, const uint8_t* scales, int* scale_out, int* min_out) {
    if (j < 4) { *scale_out = scales[j] & 0x3F; *min_out = scales[j+4] & 0x3F; }
    else       { *scale_out = (scales[j+4] & 0x0F) | ((scales[j-4] >> 6) << 4);
                 *min_out   = (scales[j+4] >> 4) | ((scales[j] >> 6) << 4); }
}
```
Called inside the 8-iteration sub-block loop (line 88). Each call does up to 4 `uint8_t` byte loads. `d_f` and `dmin_f` are loaded once at line 77-80 from two separate `__half*` pointer reads.

Then line 94-97:
```c
acc += x_scale * (d_f * (float)scale_j * (float)int_dot
                 - dmin_f * (float)min_j * (float)q8_sum);
```
The float scale multiplication happens **8 times per super-block** (once per sub-block), and each sub-block iteration does 4 scalar FMAs + 2 int→float casts.

### Q15. Memory-access impact of scale/min loads

- llama.cpp: 1 half2 load (4 B) + up to 2 uint16 loads (4 B) = **8 B per super-block for all scales/mins**. Combined with the int32-aligned weight loads, total loaded per super-block per kbx-iteration = 4 + 2*8 (weights v[0], v[4]) + 8 (scales) ≈ 28 B, shared across 16 threads = 1.75 B/thread.
- ROCmForge: 2 half loads (4 B) + 12 B scales read multiple times across the 8 sub-block iterations (compiler may hoist but the source structure does not). Per-super-block per lane: 4 (d/dmin) + 12 (scales) = 16 B, **all owned by one lane** (no sharing). 

So per-lane scale/min traffic: **9× higher in ours** than in llama.cpp (counting the lack of 16-way sharing). This contributes to our lower effective weight BW since L2 bandwidth is shared between weight and metadata traffic.

### Q16. Gate-fusion in llama.cpp

mmvq.cu:432-502, 495-502:
```c
if constexpr (has_fusion) {
    use_gate = fusion.gate != nullptr;
    vgate = fusion.gate;
}
...
for (int kbx ...) {
    for (int j = 0; j < ncols_dst; ++j)
        for (int i = 0; i < rows_per_cuda_block; ++i) {
            tmp[j][i] += vec_dot_q_cuda(vx, &y[...], ...);
            if constexpr (has_fusion) {
                if (use_gate) {
                    tmp_gate[j][i] += vec_dot_q_cuda(vgate, &y[...], ...);
                }
            }
        }
}
```

Key mechanism: **the same Q8_1 activation slice feeds two independent weight-dots** — one for up (`vx`), one for gate (`vgate`). Both `vec_dot_q_cuda` calls hit the same `y[...]` (activation) but different base pointers for weights. Because 16 threads already cooperate on each super-block of vx, the same 16 threads also cooperate on the paired super-block of vgate, re-using the same `u[..]` activation registers.

Why this doesn't kill BW (whereas ours did at 20 %): the cooperative tiling means each lane issues 2 coalesced 16-byte weight reads per iter (one to up, one to gate) — the activation LDS / registers are already live from the up dot, so the marginal cost is ~1 extra int32 weight-load + ~4 dp4as per lane per iter. Since the activation is the "hot" part (reused), and the weight reads are coalesced across lanes, both weight streams get near-peak BW.

Our failed gate-fusion was a different beast: we fused Q4_K GEMV × Q4_K GEMV × Swiglu across 4 columns per warp, with byte-wise dequant and no cooperative tiling, so the second weight stream produced 32 *scattered* L2 misses rather than 16 *coalesced* L2 hits.

Dependencies: gate-fusion depends on (a) int32-aligned weight loads, (b) shared activation registers (implies Q8_1 quantizer producing them once in the outer loop), (c) the 16-thread-per-super-block mapping. Drop any one, and the second weight stream becomes an extra scatter read.

### Q17. Could gate-fusion work for ROCmForge once Q8_1 + VDR=2 lands?

**Yes, but only after the cooperative tiling lands.** The llama.cpp design is internally self-consistent: has_fusion is safe because the base kernel is coalesced. If we port just Q8_1 + VDR=2 without changing thread mapping, gate-fusion will fail again the same way it failed the first time.

Recommended sequencing:
1. Port mmvq-style kernel (Option A) with `has_fusion = false` first — validate 65-70 % BW per kernel.
2. Then enable gate-fusion. Two weight streams should drop decode time by another ~30 % per the gate/up kernel elimination already seen in un-fusing (phase2_post_unfuse_rocprof.md:39,62-80).

---

## Root-Cause Hypotheses (ranked by estimated BW impact)

### H1. Cooperative 16-threads-per-super-block tiling (COUPLED to VDR and int32 loads)

- llama.cpp code: `mmvq.cu:483-503`, `vecdotq.cuh:881-883`
- Our code: `gemv_q4_k_q8_inline.hip:175-184` (one lane, full super-block)
- Mechanism: 16 lanes cooperate on one super-block → 16-byte coalesced global loads instead of 32 independent 32-byte scatter loads per warp-iter. Cuts unique L2 cache lines touched per warp-iter by ~16×.
- Estimated impact: **+10 to +15 percentage points BW** (51 % → 61-66 %). This is the single biggest structural lever.
- Dependencies: must land with VDR=2 and Q8_1 int32-aligned loads simultaneously. Isolated VDR=2 on our current thread map would make things worse.

### H2. int32-aligned Q8 activation representation (COUPLED to H1)

- llama.cpp code: `vecdotq.cuh:902-904` `u[2i+0] = q8[0]; u[2i+1] = q8[4]` reading `(const int*)bq8i->qs`
- Our code: `gemv_q4_k_q8_inline.hip:63` reads `int8_t x_qs[i]` byte-wise inside unroll
- Mechanism: same coalescing story on the activation side — without this, the activation LDS access is byte-strided across lanes and the LDS-bank-conflict profile worsens under a 16-thread cooperative tiling.
- Estimated impact: subsumed in H1 — required for H1 to work.
- Dependencies: **must** come with Q8_1 (padding the 34-byte struct to a 36-byte 4-aligned one with `ds`).

### H3. Per-forward-pass Q8_1 quantization reuse (independent of H1/H2)

- llama.cpp code: `mmvq.cu:1091-1098` — quantize once in a pool buffer, reuse across all matmuls of a forward step
- Our code: `gemv_q4_k_q8_inline.hip:104-134, 158` — re-quantize activations into LDS inside **every** matmul kernel
- Mechanism: each Q4_K matmul pays a cooperative-LDS-quantize tax proportional to K. For Qwen3-8B at K=4096: each of the 36 layers × ~7 matmuls/layer × 96 tokens × 8 waves writing 4096/32=128 blocks of 34 bytes ≈ 11 MB of LDS write traffic per token just for redundant Q8 generation. Moving this to one pre-pass saves LDS bandwidth and also frees LDS for weight caching in H1's cooperative tiling.
- Estimated impact: **+2 to +5 pp BW** standalone; larger when combined with H1 because LDS pressure drops.
- Dependencies: none structurally. Could land standalone as a "prequantize activations" Pass 0 in the graph, with downstream kernels reading the pre-quantized buffer. Risks: one more buffer allocation per step, but tiny.

### H4. Scale/min hoisting out of the 8-sub-block inner loop (minor)

- llama.cpp code: `vecdotq.cuh:885-897` — scales computed ONCE per super-block (specifically per bq8_offset), 2 outer iterations (QR4_K=2) not 8.
- Our code: `gemv_q4_k_q8_inline.hip:86-89` — `rf_v1_q4k_q8_unpack_scale_min` called 8× per super-block per lane.
- Mechanism: fewer redundant byte loads of `scales[...]` + fewer branch-per-j computations. Small BW effect (metadata is cached), larger register/ALU effect that helps occupancy.
- Estimated impact: **+1 to +2 pp BW** via occupancy.
- Dependencies: independent; could be refactored standalone in our kernel for a safe small win.

### H5. Gate-fusion (depends on all above)

- llama.cpp code: `mmvq.cu:437-502, 567-581`
- Our code: not implemented (unfused currently; prior attempt failed at 20 % BW)
- Mechanism: reuse activation registers across two weight streams, saving one LDS-load round-trip per super-block.
- Estimated impact: on top of a 65-70 % BW kernel, gate-fusion saves ~30 % of gate+up wall-time (matches the pre-unfuse fused-kernel gap). **+10 to +15 pp tok/s** after H1-H3 land, **0 to negative** before.
- Dependencies: HARD requires H1 (cooperative tiling) and H2 (int32 loads). Retrying without those will regress again.

Cumulative (H1+H2+H3+H4): 51 % → ~67-73 % BW, projected decode 62.7 → ~78-84 tok/s.
Plus H5 once kernel is ready: → ~92-100 tok/s. Matches the roadmap's Stufe 2+3 projection (docs/v1.0/rocmforge_v1_roadmap_to_100toks.md:84-147).

---

## Porting Plan

### Option A: Port llama.cpp kernel 1:1

Scope:
- New file `hip_kernels_v1/gemv/mmvq_q4_k_q8_1.hip` implementing the `mul_mat_vec_q<GGML_TYPE_Q4_K, 1, false>` specialization (ncols_dst=1, no fusion first).
- New file `hip_kernels_v1/quant/quantize_q8_1.hip` — port of `quantize.cu:4-48` (CUDA to HIP is nearly a rename).
- Helpers: port `get_int_b2`, `get_int_b4` (vecdotq.cuh:18-29), `ggml_cuda_dp4a` (common.cuh:672-697 — already have `__builtin_amdgcn_sudot4`).
- Port `vec_dot_q4_K_q8_1` + `vec_dot_q4_K_q8_1_impl_vmmq` (vecdotq.cuh:864-907, 505-527).
- Ensure our `block_q4_K` memory layout matches ggml's 144-byte layout bit-for-bit (it does — same 2×half + 12×scales + 128×nibbles).

Integration:
- Add `KernelId::MmvqQ4KQ8_1` variant in `runtime/variants.rs`.
- Register as a 4th variant for Q4_K shapes in the Bandit. Bandit will compare vs `q4_k_q8_inline` over 30+ pulls and commit the winner.
- Need a "prequantize activations" pipeline step before each Q4_K block — write result to `scratch_q8_1` buffer, feed as separate input.

LOC estimate: ~500 HIP + 150 Rust FFI + 80 quantizer + 200 tests ≈ **~930 LOC**.
Time: **5-8 working days** (1-2 days kernel, 1-2 days quantizer + pipeline integration, 1-2 days tests + Bandit + Parity, 1 day 15-prompt benchmark and docs).
Risk: **LOW**. The code is copy-paste-modify with a well-known upstream behavior. Parity is bit-verifiable against a llama.cpp reference run on the same weights.

### Option B: Incremental alignment of our kernel

Top 3 changes to land TOGETHER:
1. Drop `Q4_K_Q8_COLS=4` column replication; set ncols_dst=1 compile-time.
2. Reshape thread mapping to 16 threads per super-block, 8 waves cooperating on K.
3. Rewrite `rf_v1_q4k_q8_subblock_int_dot` to use int32-aligned loads + `__builtin_amdgcn_sudot4` on a 2-iteration QR4_K loop.
Plus move Q8 activation quantization out to a separate pre-kernel buffer (H3).

LOC estimate: ~350 HIP rewrite + 100 pipeline + 100 tests = **~550 LOC**.
Time: **4-6 working days**.
Risk: **MEDIUM**. The rewrite of the thread mapping is essentially a full rewrite of the kernel. At that point, you might as well follow the upstream reference (Option A) exactly because you're importing all three dependencies anyway.

### Recommendation: **Option A**.

Reasoning: Option B requires rewriting the same three interlocking pieces (thread map, int32 loads, VDR=2 scheduling) that Option A has already solved upstream. By adopting the mmvq.cu structure we inherit llama.cpp's measured 70 % BW, keep the option to enable `has_fusion` later with zero additional kernel work, and hand ourselves a directly-comparable reference for debugging. Option B's only advantage (keeping our existing code path) disappears when you count the redesign required.

---

## Next Concrete Step

Port `quantize_q8_1` from `quantize.cu:4-48` to HIP and wire it into the GPU forward pass as a pre-matmul step. Before adding the new MMVQ kernel, landing this standalone:
- Gives a measurable isolated result (should shrink in-kernel LDS traffic on current `q4_k_q8_inline` by ~5 %, i.e. +1-3 pp BW on that kernel).
- Delivers the Q8_1 buffer needed by the new MMVQ kernel without the risk of conflating two changes.
- Lets the Bandit safely A/B the current kernel pre- vs post-Q8_1 quantize-source as a 0-risk sanity check.

After that lands cleanly, proceed to mmvq kernel port (the rest of Option A).

---

## Honest Caveats

1. **We have not compiled either kernel and inspected assembly.** All VGPR/occupancy claims in Q4 are inferences from the C++ source. A concrete `llvm-objdump -d` of our committed `gemv_q4_k_q8_inline.co` and a hypothetical ported mmvq `.co` would validate the register-pressure hypothesis.

2. **The 51 % BW figure is a model-level average, not a kernel-isolated microbenchmark.** It comes from `4.75 GB / 13.7 ms = 347 GB/s = 54 %` across all kernels per token (phase2_post_unfuse_rocprof.md:239-245). The per-kernel breakdown shows the N=4096,K=4096 Q4_K gets ~82 % BW while the N=1024,K=4096 gets only 26 % due to launch-bound behavior (phase2_post_unfuse_rocprof.md:104-112). The biggest absolute win is on the **N=12288 gate/up shapes at 65 % BW** — this is where llama.cpp's design should help most.

3. **The Q8_1.s field turns out not to help Q4_K directly** (Q3, Q7). This was surprising given the prompt's framing. The real Q8_1 wins for Q4_K come from (a) it being a reusable pre-quantized buffer (H3) and (b) int32 alignment of its qs[] array allowing dp4a on `u[..]` reads (H1/H2). So "port Q8_1" is still the right move, but not for the pre-computed-sum reason one might first guess.

4. **Gate-fusion is a third-order win** (H5). It only pays off after H1+H2 are in place. Do not attempt it standalone — that was already tried and regressed (gate_up_swiglu at 20 % BW, phase2_post_unfuse_rocprof.md:113, 115-123).

5. **This analysis does not address the decode gap that remains at long context.** At seq_len > 1024, attention_decode becomes a bigger share (MEMORY.md:project_context_degradation.md). The Q4_K GEMV fix is for the 54 % BW baseline, not for context scaling.

6. **If Option A is implemented and BW does not rise above ~60 %, the residual gap is likely not in kernel design but in HIP runtime / driver overhead** — suggesting the VulkanForge pivot hinted at in the task prompt becomes the right move. llama.cpp getting 70 % on the same hardware with the same SDK establishes 70 % as reachable; anything below that is a ROCmForge-specific gap.

---

## Appendix: Key File-Line Citations

| Topic | File:Line |
|---|---|
| mmvq kernel template | /home/maeddes/tmp/llama.cpp/ggml/src/ggml-cuda/mmvq.cu:391-591 |
| nwarps=8 whitelist (Q4_K on RDNA4) | /home/maeddes/tmp/llama.cpp/ggml/src/ggml-cuda/mmvq.cu:324-346 |
| blocks_per_iter formula | /home/maeddes/tmp/llama.cpp/ggml/src/ggml-cuda/mmvq.cu:414 |
| 16-thread cooperative kbx loop | /home/maeddes/tmp/llama.cpp/ggml/src/ggml-cuda/mmvq.cu:483-503 |
| gate-fusion template branch | /home/maeddes/tmp/llama.cpp/ggml/src/ggml-cuda/mmvq.cu:429-502, 567-581 |
| VDR_Q4_K_Q8_1_MMVQ = 2 | /home/maeddes/tmp/llama.cpp/ggml/src/ggml-cuda/vecdotq.cuh:501 |
| vec_dot_q4_K_q8_1_impl_vmmq | /home/maeddes/tmp/llama.cpp/ggml/src/ggml-cuda/vecdotq.cuh:505-527 |
| vec_dot_q4_K_q8_1 entry (int32 weight load) | /home/maeddes/tmp/llama.cpp/ggml/src/ggml-cuda/vecdotq.cuh:864-908 |
| get_int_b4 (int32 aligned load helper) | /home/maeddes/tmp/llama.cpp/ggml/src/ggml-cuda/vecdotq.cuh:27-29 |
| ggml_cuda_dp4a on RDNA4 | /home/maeddes/tmp/llama.cpp/ggml/src/ggml-cuda/common.cuh:672-677 (uses __builtin_amdgcn_sudot4) |
| block_q8_1 struct | /home/maeddes/tmp/llama.cpp/ggml/src/ggml-common.h:248-259 |
| block_q4_K struct | /home/maeddes/tmp/llama.cpp/ggml/src/ggml-common.h:317-328 |
| quantize_q8_1 CUDA kernel | /home/maeddes/tmp/llama.cpp/ggml/src/ggml-cuda/quantize.cu:4-48 |
| Q4_K type traits (qk=256, qi=32, qr=2) | /home/maeddes/tmp/llama.cpp/ggml/src/ggml-cuda/common.cuh:997-1002 |
| ROCmForge Q4_K GEMV — thread map | /home/maeddes/projects/ROCmForge/hip_kernels_v1/gemv/gemv_q4_k_q8_inline.hip:146-184 |
| ROCmForge sub-block dot (byte-wise) | /home/maeddes/projects/ROCmForge/hip_kernels_v1/gemv/gemv_q4_k_q8_inline.hip:44-69 |
| ROCmForge scale/min unpack | /home/maeddes/projects/ROCmForge/hip_kernels_v1/gemv/gemv_q4_k_q8_inline.hip:31-41 |
| ROCmForge cooperative Q8 quantize | /home/maeddes/projects/ROCmForge/hip_kernels_v1/gemv/gemv_q4_k_q8_inline.hip:104-134 |
| Prior sudot4 negative result | /home/maeddes/projects/ROCmForge/results/phase2_sudot4_kernel_upgrade.md |
| Post-unfuse BW baseline (51 %) | /home/maeddes/projects/ROCmForge/results/phase2_post_unfuse_rocprof.md:104-112, 239-252 |
| Roadmap Stufe 2 (this port) | /home/maeddes/projects/ROCmForge/docs/v1.0/rocmforge_v1_roadmap_to_100toks.md:93-127 |
