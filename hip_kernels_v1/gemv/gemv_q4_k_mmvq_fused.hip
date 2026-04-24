// Q4_K × Q8_1 GEMV WITH FUSED GATE+UP+SwiGLU — Phase 2 Schritt 3.
//
// Computes `dst[row] = silu(gate_dot[row]) * up_dot[row]` in one kernel,
// where both dot products are Q4_K × Q8_1 GEMV over the same activation.
// This is the llama.cpp `has_fusion=true` path from `mmvq.cu:437-502,
// 567-581` ported 1:1 onto our cooperative-tiling MMVQ kernel.
//
// Why the old `gate_up_swiglu` failed (20 % BW, 3.2× slower than unfused)
// but this one shouldn't:
//   * Old kernel: one-thread-per-super-block tiling. Two weight streams
//     produced two scattered 32-byte reads per warp-iter, serially
//     contending for the L2/HBM queue.
//   * This kernel: 16-threads-per-super-block tiling. Two coalesced
//     64-byte reads per warp-iter — one outstanding request per stream,
//     processed in parallel by the memory controller. The second
//     vec_dot runs in the ALU cycles that would otherwise idle waiting
//     on the first.
//
// Shared memory: 2 × [7][32] = 1 792 bytes (one reduction scratch per
// stream). Still negligible vs the 64 KB budget.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>

// ─── Constants ──────────────────────────────────────────────────────────────

#define RF_QK_K          256
#define RF_QR4_K         2
#define RF_QI4_K         32
#define RF_QK8_1         32
#define RF_QI8_1         8
#define RF_VDR_Q4_K      2

#define RF_WARP_SIZE     32
#define RF_NWARPS        8
#define RF_ROWS_PER_BLK  1
#define RF_THREADS_PER_BLOCK (RF_WARP_SIZE * RF_NWARPS)

struct block_q4_K {
    __half2 dm;
    uint8_t scales[12];
    uint8_t qs[128];
};
static_assert(sizeof(block_q4_K) == 144, "block_q4_K must be 144 bytes");

struct block_q8_1 {
    __half2 ds;
    int8_t  qs[32];
};
static_assert(sizeof(block_q8_1) == 36, "block_q8_1 must be 36 bytes");

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

// silu(x) = x / (1 + exp(-x)). Matches llama.cpp's
// `ggml_cuda_op_silu_single` (unary.cuh:94) and our existing
// `rf_v1_swiglu_kernel` in elementwise_block_a.hip.
__device__ __forceinline__ float rf_v1_silu(float x) {
    return x / (1.0f + expf(-x));
}

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

        const int dot1 = rf_v1_dp4a(v1i, u[2*i+1],
                                     rf_v1_dp4a(v0i, u[2*i+0], 0));
        const int dot2 = rf_v1_dp4a(0x01010101, u[2*i+1],
                                     rf_v1_dp4a(0x01010101, u[2*i+0], 0));

        sumf_d += d8[i] * (dot1 * (int)sc[i]);
        sumf_m += d8[i] * (dot2 * (int)m[i]);
    }

    const float2 dm4f = __half22float2(dm4);
    return dm4f.x * sumf_d - dm4f.y * sumf_m;
}

__device__ __forceinline__ float rf_v1_vec_dot_q4_K_q8_1(
    const block_q4_K* __restrict__ vbq,
    const block_q8_1* __restrict__ bq8_1,
    int iqs)
{
    int   v[2];
    int   u[2 * RF_QR4_K];
    float d8[RF_QR4_K];

    const int bq8_offset = RF_QR4_K * ((iqs / 2) / (RF_QI8_1 / 2));

    const int* q4 = (const int*)(vbq->qs + 16*bq8_offset + 4*((iqs/2) % 4));
    v[0] = q4[0];
    v[1] = q4[4];

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
        d8[i] = __half2float(__low2half(bq8i->ds));

        const int* q8 = (const int*)bq8i->qs + ((iqs/2) % 4);
        u[2*i+0] = q8[0];
        u[2*i+1] = q8[4];
    }

    return rf_v1_q4_K_impl_vmmq(v, u, sc, m, vbq->dm, d8);
}

// ─── Fused kernel: two weight streams + SwiGLU epilog ──────────────────────

__launch_bounds__(RF_THREADS_PER_BLOCK, 1)
__global__ void rf_v1_mmvq_q4_k_q8_1_fused_kernel(
    const block_q4_K* __restrict__ vx_gate,   // Q4_K gate weights
    const block_q4_K* __restrict__ vx_up,     // Q4_K up weights
    const block_q8_1* __restrict__ vy,        // shared Q8_1 activations
    float*            __restrict__ dst,       // SwiGLU(gate) * up output
    uint32_t ncols_x,
    uint32_t stride_row_x,
    uint32_t nrows)
{
    constexpr int qi              = RF_QI4_K;
    constexpr int qk              = RF_QK_K;
    constexpr int vdr             = RF_VDR_Q4_K;
    constexpr int blocks_per_iter = vdr * RF_NWARPS * RF_WARP_SIZE / qi;  // 16

    const int tid  = RF_WARP_SIZE * threadIdx.y + threadIdx.x;
    const int row0 = RF_ROWS_PER_BLK * (int)blockIdx.x;

    if ((uint32_t)row0 >= nrows) return;

    const int blocks_per_row_x = (int)(ncols_x / (uint32_t)qk);
    const int kbx_offset       = row0 * (int)stride_row_x;

    // Two parallel accumulators — one per weight stream.
    float tmp_up   = 0.0f;
    float tmp_gate = 0.0f;

    // Cooperative K-loop: 16 threads share each kbx. Both weight streams
    // are walked in lock-step with the same Q8_1 activation slice — the
    // second vec_dot runs in the ALU cycles otherwise idle waiting on
    // the first stream's memory latency.
    for (int kbx = tid / (qi / vdr); kbx < blocks_per_row_x; kbx += blocks_per_iter) {
        const int kby = kbx * (qk / RF_QK8_1);
        const int kqs = vdr * (tid % (qi / vdr));

        const block_q4_K* vbq_up   = vx_up   + (kbx_offset + kbx);
        const block_q4_K* vbq_gate = vx_gate + (kbx_offset + kbx);

        tmp_up   += rf_v1_vec_dot_q4_K_q8_1(vbq_up,   &vy[kby], kqs);
        tmp_gate += rf_v1_vec_dot_q4_K_q8_1(vbq_gate, &vy[kby], kqs);
    }

    // Two reduction scratch arrays — matches llama.cpp mmvq.cu:505-506.
    __shared__ float tmp_shared_up  [RF_NWARPS - 1][RF_WARP_SIZE];
    __shared__ float tmp_shared_gate[RF_NWARPS - 1][RF_WARP_SIZE];

    if (threadIdx.y > 0) {
        tmp_shared_up  [threadIdx.y - 1][threadIdx.x] = tmp_up;
        tmp_shared_gate[threadIdx.y - 1][threadIdx.x] = tmp_gate;
    }
    __syncthreads();

    if (threadIdx.y > 0) {
        return;
    }

    #pragma unroll
    for (int l = 0; l < RF_NWARPS - 1; ++l) {
        tmp_up   += tmp_shared_up  [l][threadIdx.x];
        tmp_gate += tmp_shared_gate[l][threadIdx.x];
    }

    tmp_up   = rf_v1_warp_reduce_sum(tmp_up);
    tmp_gate = rf_v1_warp_reduce_sum(tmp_gate);

    if (threadIdx.x == 0) {
        // SwiGLU epilog at lane 0. silu(gate) × up — free in register.
        dst[row0] = tmp_up * rf_v1_silu(tmp_gate);
    }
}

// ─── Launcher ───────────────────────────────────────────────────────────────

extern "C" hipError_t rocmforge_launch_gemv_q4_k_mmvq_fused(
    const void*  weights_gate,     // block_q4_K*
    const void*  weights_up,       // block_q4_K*
    const void*  q8_1_input,       // block_q8_1*, pre-quantized shared
    void*        output,           // float* [nrows]
    int          nrows,            // N = ffn_dim (output dim)
    int          ncols,            // K = hidden_dim (input dim)
    hipStream_t  stream)
{
    if (nrows <= 0 || ncols <= 0)            return hipErrorInvalidValue;
    if ((ncols % RF_QK_K) != 0)              return hipErrorInvalidValue;

    const uint32_t blocks_per_row = (uint32_t)(ncols / RF_QK_K);

    const dim3 block_nums((uint32_t)nrows, 1, 1);
    const dim3 block_dims(RF_WARP_SIZE, RF_NWARPS, 1);

    rf_v1_mmvq_q4_k_q8_1_fused_kernel<<<block_nums, block_dims, 0, stream>>>(
        reinterpret_cast<const block_q4_K*>(weights_gate),
        reinterpret_cast<const block_q4_K*>(weights_up),
        reinterpret_cast<const block_q8_1*>(q8_1_input),
        reinterpret_cast<float*>(output),
        (uint32_t)ncols,
        blocks_per_row,
        (uint32_t)nrows);

    return hipGetLastError();
}
