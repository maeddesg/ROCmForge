// Q4_K × Q8_1 GEMV — 1:1 port of llama.cpp's `mul_mat_vec_q` kernel,
// specialised for Q4_K, ncols_dst=1, has_fusion=false, small_k=false.
//
// Sources (ROCm 7.2.2, gfx1201 / RDNA4):
//   mmvq.cu:391-591       — kernel template
//   mmvq.cu:324-346       — nwarps=8 whitelist for Q4_K on RDNA4
//   vecdotq.cuh:501-527   — vec_dot_q4_K_q8_1_impl_vmmq
//   vecdotq.cuh:864-907   — vec_dot_q4_K_q8_1 wrapper
//   common.cuh:672-697    — ggml_cuda_dp4a (→ __builtin_amdgcn_sudot4)
//
// CUDA → HIP changes:
//   * `__shfl_xor_sync(mask, val, off, width)` → `__shfl_xor(val, off)`
//     (RDNA wave32 ignores the mask; width defaults to WARP_SIZE)
//   * `ggml_cuda_dp4a` → `__builtin_amdgcn_sudot4(true, a, true, b, c, false)`
//   * `__half22float2` → `__half22float2` (same name in hip_fp16.h)
//
// Spezialisierte Konstanten (fixe für Q4_K + ncols_dst=1 + RDNA4):
//   QK_K=256, QI4_K=32, QR4_K=2, VDR=2, QK8_1=32, QI8_1=8
//   nwarps=8, rows_per_cuda_block=1, warp_size=32
//   blocks_per_iter = vdr * nwarps * warp_size / qi = 2 * 8 * 32 / 32 = 16
//   16 Threads cooperate on each super-block (→ int32-coalesced loads).

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>

// ─── Constants ──────────────────────────────────────────────────────────────

#define RF_QK_K          256
#define RF_QR4_K         2
#define RF_QI4_K         32   // (QK_K / (4 * QR4_K))
#define RF_QK8_1         32
#define RF_QI8_1         8    // (QK8_1 / (4 * QR8_1)), QR8_1=1
#define RF_VDR_Q4_K      2

#define RF_WARP_SIZE     32
#define RF_NWARPS        8
#define RF_ROWS_PER_BLK  1
#define RF_THREADS_PER_BLOCK (RF_WARP_SIZE * RF_NWARPS)

// ─── Block structs — bit-identical to llama.cpp + GGUF on-disk layout ───────

struct block_q4_K {
    __half2 dm;            // dm.x = d (super-block scale), dm.y = dmin
    uint8_t scales[12];    // 6-bit packed scales + mins (K_SCALE_SIZE=12)
    uint8_t qs[128];       // 4-bit nibbles (QK_K/2 = 128)
};
static_assert(sizeof(block_q4_K) == 144, "block_q4_K must be 144 bytes");

struct block_q8_1 {
    __half2 ds;            // ds.x = d, ds.y ≈ d * Σ qs[i]
    int8_t  qs[32];        // QK8_1 = 32
};
static_assert(sizeof(block_q8_1) == 36, "block_q8_1 must be 36 bytes");

// ─── dp4a primitive ─────────────────────────────────────────────────────────

// signed-a × unsigned-b → int32 accumulator, native on gfx1201 (v_dot4_i32_iu8).
// Mirrors llama.cpp common.cuh:672-697 RDNA3/4 branch.
__device__ __forceinline__ int rf_v1_dp4a(const int a, const int b, int c) {
    return __builtin_amdgcn_sudot4(true, a, true, b, c, false);
}

// Warp-reduce sum over full wave32 (mask is ignored on AMD; width = WARP_SIZE).
__device__ __forceinline__ float rf_v1_warp_reduce_sum(float x) {
    #pragma unroll
    for (int off = RF_WARP_SIZE / 2; off > 0; off >>= 1) {
        x += __shfl_xor(x, off);
    }
    return x;
}

// ─── vec_dot_q4_K_q8_1_impl_vmmq (inline) ───────────────────────────────────

// Port of vecdotq.cuh:505-527. Computes one (Q4_K super-block × 2 Q8_1 blocks)
// partial dot product from pre-loaded v[2]/u[4]/sc[2]/m[2]/dm/d8[2] state.
__device__ __forceinline__ float rf_v1_q4_K_impl_vmmq(
    const int v[2], const int u[4], const uint8_t sc[2], const uint8_t m[2],
    const __half2 dm4, const float d8[2])
{
    float sumf_d = 0.0f;
    float sumf_m = 0.0f;

    #pragma unroll
    for (int i = 0; i < RF_QR4_K; ++i) {
        const int v0i = (v[0] >> (4*i)) & 0x0F0F0F0F;
        const int v1i = (v[1] >> (4*i)) & 0x0F0F0F0F;

        // 2 dp4a → 8 byte-dots in 2 cycles (weights × activations)
        const int dot1 = rf_v1_dp4a(v1i, u[2*i+1],
                                     rf_v1_dp4a(v0i, u[2*i+0], 0));
        // sum-of-u term via dp4a against 1s — cheap, no dedicated instruction
        const int dot2 = rf_v1_dp4a(0x01010101, u[2*i+1],
                                     rf_v1_dp4a(0x01010101, u[2*i+0], 0));

        sumf_d += d8[i] * (dot1 * (int)sc[i]);
        sumf_m += d8[i] * (dot2 * (int)m[i]);
    }

    const float2 dm4f = __half22float2(dm4);
    return dm4f.x * sumf_d - dm4f.y * sumf_m;
}

// ─── vec_dot_q4_K_q8_1 wrapper (loads + calls impl) ─────────────────────────

// Port of vecdotq.cuh:864-907. `kbx` already includes the row-stride offset
// (i.e. it is an index into the global block_q4_K array, not row-local).
// `iqs` is the per-thread offset into the super-block, values 0,2,4,...,30.
__device__ __forceinline__ float rf_v1_vec_dot_q4_K_q8_1(
    const block_q4_K* __restrict__ vbq,
    const block_q8_1* __restrict__ bq8_1,
    int iqs)
{
    int   v[2];
    int   u[2 * RF_QR4_K];  // u[4]
    float d8[RF_QR4_K];     // d8[2]

    // iqs = 0,2,...,30. bq8_offset ∈ {0,2,4,6} selects a pair of Q8_1 blocks.
    const int bq8_offset = RF_QR4_K * ((iqs / 2) / (RF_QI8_1 / 2));

    const int* q4 = (const int*)(vbq->qs + 16*bq8_offset + 4*((iqs/2) % 4));
    v[0] = q4[0];
    v[1] = q4[4];

    // 6-bit packed Q4_K scales + mins. j ∈ {0,1,2,3}.
    const uint16_t* scales16 = (const uint16_t*)vbq->scales;
    uint16_t aux[2];
    const int j = bq8_offset / 2;
    if (j < 2) {
        aux[0] = scales16[j+0] & 0x3f3f;
        aux[1] = scales16[j+2] & 0x3f3f;
    } else {
        aux[0] = ((scales16[j+2] >> 0) & 0x0f0f) | ((scales16[j-2] & 0xc0c0) >> 2);
        aux[1] = ((scales16[j+2] >> 4) & 0x0f0f) | ((scales16[j-0] & 0xc0c0) >> 2);
    }
    const uint8_t* sc = (const uint8_t*)aux;
    const uint8_t* m  = sc + 2;

    #pragma unroll
    for (int i = 0; i < RF_QR4_K; ++i) {
        const block_q8_1* bq8i = bq8_1 + bq8_offset + i;
        // d8 is the Q8_1 scale (`d` = lower half of ds).
        d8[i] = __half2float(__low2half(bq8i->ds));

        const int* q8 = (const int*)bq8i->qs + ((iqs/2) % 4);
        u[2*i+0] = q8[0];
        u[2*i+1] = q8[4];
    }

    return rf_v1_q4_K_impl_vmmq(v, u, sc, m, vbq->dm, d8);
}

// ─── MMVQ kernel — one output row per grid block, nwarps=8 cooperating ─────

__launch_bounds__(RF_THREADS_PER_BLOCK, 1)
__global__ void rf_v1_mmvq_q4_k_q8_1_kernel(
    const block_q4_K* __restrict__ vx,
    const block_q8_1* __restrict__ vy,
    float*            __restrict__ dst,
    uint32_t ncols_x,        // K (input dimension in elements)
    uint32_t stride_row_x,   // blocks per row of x = ncols_x / QK_K
    uint32_t nrows)          // N (output dimension), used for bounds check
{
    constexpr int qi              = RF_QI4_K;
    constexpr int qk              = RF_QK_K;
    constexpr int vdr             = RF_VDR_Q4_K;
    constexpr int blocks_per_iter = vdr * RF_NWARPS * RF_WARP_SIZE / qi;  // 16

    const int tid  = RF_WARP_SIZE * threadIdx.y + threadIdx.x;  // 0..255
    const int row0 = RF_ROWS_PER_BLK * (int)blockIdx.x;         // one row per block

    if ((uint32_t)row0 >= nrows) return;

    const int blocks_per_row_x = (int)(ncols_x / (uint32_t)qk);
    const int kbx_offset       = row0 * (int)stride_row_x;

    float tmp_val = 0.0f;

    // Cooperative K-loop: 16 threads share each kbx, each handling a
    // 2-byte slice of the Q8_1 block (VDR=2) via int32-aligned loads.
    for (int kbx = tid / (qi / vdr); kbx < blocks_per_row_x; kbx += blocks_per_iter) {
        const int kby = kbx * (qk / RF_QK8_1);              // 8 Q8_1 blocks per super-block
        const int kqs = vdr * (tid % (qi / vdr));           // 0, 2, 4, ..., 30

        const block_q4_K* vbq = vx + (kbx_offset + kbx);
        tmp_val += rf_v1_vec_dot_q4_K_q8_1(vbq, &vy[kby], kqs);
    }

    // Reduction: nwarps partial sums → 1 final sum.
    // Layout is flattened [NWARPS-1][WARP_SIZE] — ncols_dst=rows_per_cuda_block=1
    // collapses the two inner dims away. Size: 7×32×4 = 896 B (negligible).
    __shared__ float tmp_shared[RF_NWARPS - 1][RF_WARP_SIZE];

    if (threadIdx.y > 0) {
        tmp_shared[threadIdx.y - 1][threadIdx.x] = tmp_val;
    }
    __syncthreads();

    if (threadIdx.y > 0) {
        return;  // Only wave 0 finishes.
    }

    // Wave 0: add the 7 stored partials.
    #pragma unroll
    for (int l = 0; l < RF_NWARPS - 1; ++l) {
        tmp_val += tmp_shared[l][threadIdx.x];
    }

    // Warp-reduce across 32 lanes of wave 0.
    tmp_val = rf_v1_warp_reduce_sum(tmp_val);

    // Lane 0 writes (rows_per_cuda_block=1 → only threadIdx.x==0).
    if (threadIdx.x == 0) {
        dst[row0] = tmp_val;
    }
}

// ─── Launcher ───────────────────────────────────────────────────────────────

extern "C" hipError_t rocmforge_launch_gemv_q4_k_mmvq(
    const void*  weights,          // block_q4_K*
    const void*  q8_1_input,       // block_q8_1*, pre-quantized via quantize_q8_1
    void*        output,           // float*
    int          nrows,            // N (output dimension)
    int          ncols,            // K (input dimension), must be multiple of 256
    hipStream_t  stream)
{
    if (nrows <= 0 || ncols <= 0)            return hipErrorInvalidValue;
    if ((ncols % RF_QK_K) != 0)              return hipErrorInvalidValue;

    const uint32_t blocks_per_row = (uint32_t)(ncols / RF_QK_K);

    const dim3 block_nums((uint32_t)nrows, 1, 1);
    const dim3 block_dims(RF_WARP_SIZE, RF_NWARPS, 1);

    rf_v1_mmvq_q4_k_q8_1_kernel<<<block_nums, block_dims, 0, stream>>>(
        reinterpret_cast<const block_q4_K*>(weights),
        reinterpret_cast<const block_q8_1*>(q8_1_input),
        reinterpret_cast<float*>(output),
        (uint32_t)ncols,
        blocks_per_row,
        (uint32_t)nrows);

    return hipGetLastError();
}
