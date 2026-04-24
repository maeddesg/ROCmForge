// Q6_K × Q8_1 GEMV — 1:1 port of llama.cpp's `mul_mat_vec_q` specialised
// for Q6_K, ncols_dst=1, RDNA4 (nwarps=8), VDR=1.
//
// Sources (ROCm 7.2.2, gfx1201):
//   mmvq.cu:391-591               — kernel template (same as Q4_K)
//   vecdotq.cuh:620-644           — vec_dot_q6_K_q8_1_impl_mmvq
//   vecdotq.cuh:956-980           — vec_dot_q6_K_q8_1 wrapper
//   ggml-common.h:352-358         — block_q6_K (210 bytes)
//   vendors/hip.h:256-274         — __vsubss4 HIP fallback
//
// Key differences vs gemv_q4_k_mmvq.hip:
//   * VDR=1 (Q6_K) vs VDR=2 (Q4_K) → `qi/vdr = 32` → 32 threads cooperate
//     per super-block (whole warp instead of half-warp).
//   * `blocks_per_iter = 1 * 8 * 32 / 32 = 8` (vs Q4_K's 16).
//   * Single accumulator `sumf` (no per-block min — Q6_K has no `dmin`).
//   * ql/qh reconstruction: 4-bit from ql OR'd with 2-bit from qh shifted
//     to bits 4-5, then `- 32` per byte (signed-saturated) before dp4a.
//   * `scales[16]` is `int8_t` direct — no 6-bit packing.
//
// Thread mapping (from llama.cpp for iqs ∈ [0,32), QR6_K=2, QI6_K=32):
//   bq8_offset  = 2*QR6_K*(iqs/16) + ((iqs%16)/8)             ∈ {0, 1, 4, 5}
//   scale_offset = (QI6_K/4)*(iqs/16) + ((iqs%16)/4)          ∈ [0, 11]
//   vh_shift    = 2 * ((iqs%16)/8)                             ∈ {0, 2}

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>
#include <limits>

// ─── Constants ──────────────────────────────────────────────────────────────

#define RF_QK_K          256
#define RF_QR6_K         2
#define RF_QI6_K         32     // (QK_K / (4 * QR6_K))
#define RF_QK8_1         32
#define RF_QI8_1         8
#define RF_VDR_Q6_K      1

#define RF_WARP_SIZE     32
#define RF_NWARPS        8
#define RF_ROWS_PER_BLK  1
#define RF_THREADS_PER_BLOCK (RF_WARP_SIZE * RF_NWARPS)

// ─── Block structs — bit-identical to ggml-common.h + GGUF layout ──────────

struct block_q6_K {
    uint8_t ql[128];       // lower 4 bits of each 6-bit quant (QK_K/2)
    uint8_t qh[64];        // upper 2 bits of each 6-bit quant (QK_K/4)
    int8_t  scales[16];    // 8-bit per-sub-block scales (QK_K/16)
    __half  d;             // super-block scale
};
static_assert(sizeof(block_q6_K) == 210, "block_q6_K must be 210 bytes");

struct block_q8_1 {
    __half2 ds;
    int8_t  qs[32];
};
static_assert(sizeof(block_q8_1) == 36, "block_q8_1 must be 36 bytes");

// ─── Primitives ────────────────────────────────────────────────────────────

__device__ __forceinline__ int rf_v1_dp4a(const int a, const int b, int c) {
    return __builtin_amdgcn_sudot4(true, a, true, b, c, false);
}

__device__ __forceinline__ float rf_v1_warp_reduce_sum(float x) {
    #pragma unroll
    for (int off = RF_WARP_SIZE / 2; off > 0; off >>= 1) {
        x += __shfl_xor(x, off);
    }
    return x;
}

// Per-byte signed-saturated subtract. Q6_K subtracts 32 from each 6-bit
// quant to map [0, 63] → [-32, 31] before dp4a. The result fits in int8
// by construction (no actual saturation needed for Q6_K) but we use the
// saturated op for parity with llama.cpp. Port of vendors/hip.h:256-274.
typedef int8_t rf_int8x4_t __attribute__((ext_vector_type(4)));

__device__ __forceinline__ int rf_v1_vsubss4(const int a, const int b) {
    const rf_int8x4_t va = reinterpret_cast<const rf_int8x4_t&>(a);
    const rf_int8x4_t vb = reinterpret_cast<const rf_int8x4_t&>(b);
#if __has_builtin(__builtin_elementwise_sub_sat)
    const rf_int8x4_t c = __builtin_elementwise_sub_sat(va, vb);
    return reinterpret_cast<const int&>(c);
#else
    rf_int8x4_t c;
    int16_t tmp;
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        tmp = va[i] - vb[i];
        if (tmp > 127)  tmp = 127;
        if (tmp < -128) tmp = -128;
        c[i] = (int8_t)tmp;
    }
    return reinterpret_cast<const int&>(c);
#endif
}

// 2-byte-aligned int32 read — qh is not always 4-byte-aligned at
// arbitrary offsets. Port of vecdotq.cuh:18-25.
__device__ __forceinline__ int rf_v1_get_int_b2(const void* x, int i32) {
    const uint16_t* x16 = (const uint16_t*)x;
    int x32  = (int)x16[2*i32 + 0];
    x32     |= (int)x16[2*i32 + 1] << 16;
    return x32;
}

__device__ __forceinline__ int rf_v1_get_int_b4(const void* x, int i32) {
    return ((const int*)x)[i32];
}

// ─── vec_dot_q6_K_q8_1 — ported from vecdotq.cuh:956-980 + 624-644 ─────────

__device__ __forceinline__ float rf_v1_vec_dot_q6_K_q8_1(
    const block_q6_K* __restrict__ bq6_K,
    const block_q8_1* __restrict__ bq8_1,
    int iqs)
{
    const int bq8_offset  = 2 * RF_QR6_K * (iqs / (RF_QI6_K / 2))
                          + (iqs % (RF_QI6_K / 2)) / (RF_QI6_K / 4);
    const int scale_offset = (RF_QI6_K / 4) * (iqs / (RF_QI6_K / 2))
                           + (iqs % (RF_QI6_K / 2)) / (RF_QI6_K / 8);
    const int vh_shift    = 2 * ((iqs % (RF_QI6_K / 2)) / (RF_QI6_K / 4));

    const int vl = rf_v1_get_int_b2(bq6_K->ql, iqs);
    const int vh = rf_v1_get_int_b2(
                        bq6_K->qh,
                        (RF_QI6_K / 4) * (iqs / (RF_QI6_K / 2))
                            + iqs % (RF_QI6_K / 4))
                    >> vh_shift;

    const int8_t* scales = bq6_K->scales + scale_offset;

    int   u[RF_QR6_K];
    float d8[RF_QR6_K];

    #pragma unroll
    for (int i = 0; i < RF_QR6_K; ++i) {
        u[i]  = rf_v1_get_int_b4(bq8_1[bq8_offset + 2*i].qs, iqs % RF_QI8_1);
        d8[i] = __half2float(__low2half(bq8_1[bq8_offset + 2*i].ds));
    }

    // impl_mmvq — vecdotq.cuh:624-644.
    float sumf = 0.0f;
    #pragma unroll
    for (int i = 0; i < RF_QR6_K; ++i) {
        const int sc  = (int)scales[4 * i];
        const int vil = (vl >> (4 * i)) & 0x0F0F0F0F;
        const int vih = ((vh >> (4 * i)) << 4) & 0x30303030;
        const int vi  = rf_v1_vsubss4((vil | vih), 0x20202020);  // -32 per byte

        sumf += d8[i] * ((float)rf_v1_dp4a(vi, u[i], 0) * (float)sc);
    }

    const float d = __half2float(bq6_K->d);
    return d * sumf;
}

// ─── MMVQ kernel — one output row per grid block, 8 warps cooperating ─────

__launch_bounds__(RF_THREADS_PER_BLOCK, 1)
__global__ void rf_v1_mmvq_q6_k_q8_1_kernel(
    const block_q6_K* __restrict__ vx,
    const block_q8_1* __restrict__ vy,
    float*            __restrict__ dst,
    uint32_t ncols_x,
    uint32_t stride_row_x,
    uint32_t nrows)
{
    constexpr int qi              = RF_QI6_K;
    constexpr int qk              = RF_QK_K;
    constexpr int vdr             = RF_VDR_Q6_K;
    // qi/vdr = 32 → one whole warp cooperates on each super-block.
    // blocks_per_iter = vdr * nwarps * warp_size / qi = 1*8*32/32 = 8.
    constexpr int blocks_per_iter = vdr * RF_NWARPS * RF_WARP_SIZE / qi;

    const int tid  = RF_WARP_SIZE * threadIdx.y + threadIdx.x;
    const int row0 = RF_ROWS_PER_BLK * (int)blockIdx.x;

    if ((uint32_t)row0 >= nrows) return;

    const int blocks_per_row_x = (int)(ncols_x / (uint32_t)qk);
    const int kbx_offset       = row0 * (int)stride_row_x;

    float tmp_val = 0.0f;

    for (int kbx = tid / (qi / vdr); kbx < blocks_per_row_x; kbx += blocks_per_iter) {
        const int kby = kbx * (qk / RF_QK8_1);        // 8 Q8_1 blocks per super-block
        const int kqs = vdr * (tid % (qi / vdr));     // iqs ∈ [0, 32)

        const block_q6_K* vbq = vx + (kbx_offset + kbx);
        tmp_val += rf_v1_vec_dot_q6_K_q8_1(vbq, &vy[kby], kqs);
    }

    // Same reduction scheme as Q4_K MMVQ.
    __shared__ float tmp_shared[RF_NWARPS - 1][RF_WARP_SIZE];

    if (threadIdx.y > 0) {
        tmp_shared[threadIdx.y - 1][threadIdx.x] = tmp_val;
    }
    __syncthreads();

    if (threadIdx.y > 0) {
        return;
    }

    #pragma unroll
    for (int l = 0; l < RF_NWARPS - 1; ++l) {
        tmp_val += tmp_shared[l][threadIdx.x];
    }

    tmp_val = rf_v1_warp_reduce_sum(tmp_val);

    if (threadIdx.x == 0) {
        dst[row0] = tmp_val;
    }
}

// ─── Launcher ───────────────────────────────────────────────────────────────

extern "C" hipError_t rocmforge_launch_gemv_q6_k_mmvq(
    const void*  weights,
    const void*  q8_1_input,
    void*        output,
    int          nrows,
    int          ncols,
    hipStream_t  stream)
{
    if (nrows <= 0 || ncols <= 0)            return hipErrorInvalidValue;
    if ((ncols % RF_QK_K) != 0)              return hipErrorInvalidValue;

    const uint32_t blocks_per_row = (uint32_t)(ncols / RF_QK_K);

    const dim3 block_nums((uint32_t)nrows, 1, 1);
    const dim3 block_dims(RF_WARP_SIZE, RF_NWARPS, 1);

    rf_v1_mmvq_q6_k_q8_1_kernel<<<block_nums, block_dims, 0, stream>>>(
        reinterpret_cast<const block_q6_K*>(weights),
        reinterpret_cast<const block_q8_1*>(q8_1_input),
        reinterpret_cast<float*>(output),
        (uint32_t)ncols,
        blocks_per_row,
        (uint32_t)nrows);

    return hipGetLastError();
}
