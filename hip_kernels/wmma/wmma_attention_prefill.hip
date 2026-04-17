// Phase 3a — minimal WMMA attention kernel (single head, seq=64, head_dim=128).
//
// Computes O = softmax(Q · K^T / sqrt(head_dim)) · V using the gfx12 Wave32
// WMMA intrinsic (`__builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12`).
//
//   Q: [64 × 128] FP16 row-major
//   K: [64 × 128] FP16 row-major
//   V: [64 × 128] FP16 row-major
//   O: [64 × 128] FP32 row-major
//
// Deliberately a PoC:
//   * no causal mask           (Phase 3c)
//   * no online softmax        (Phase 3b — everything fits in a 64×64 S tile)
//   * no GQA sharing           (Phase 3c)
//   * single head per kernel   (Phase 3d)
//
// Layout of S = Q·K^T as 16×16 WMMA blocks:
//   Each wave owns a 16×64 row-strip of S (4 column blocks).
//   The whole 64×64 S fits in registers after GEMM #1.
//
// Softmax strategy: Option A from the prompt — write S to LDS (FP32),
// let a thread per row compute max/sum/normalise, write FP16 P back to
// LDS for GEMM #2. Simple and easy to validate; register-resident
// reductions are a Phase 3b / 3c optimisation.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

using half8  = _Float16 __attribute__((ext_vector_type(8)));
using float8 = float    __attribute__((ext_vector_type(8)));

#define SEQ          64
#define HEAD_DIM     128
#define TILE_K       16
#define WARPS        4
#define THREADS      (WARPS * 32)
#define COL_BLOCKS   4   // 64 cols / 16 per WMMA block

__device__ __forceinline__ void load_a_reg_from_lds(
    const __half* __restrict__ lds_base,
    int row_in_tile,
    int k_set,
    int stride_k,    // = TILE_K
    half8& a_reg)
{
    const int base = row_in_tile * stride_k;
    a_reg[0] = lds_base[base + k_set + 0];
    a_reg[1] = lds_base[base + k_set + 1];
    a_reg[2] = lds_base[base + k_set + 2];
    a_reg[3] = lds_base[base + k_set + 3];
    a_reg[4] = lds_base[base + k_set + 8];
    a_reg[5] = lds_base[base + k_set + 9];
    a_reg[6] = lds_base[base + k_set + 10];
    a_reg[7] = lds_base[base + k_set + 11];
}

__device__ __forceinline__ void load_b_reg_from_lds(
    const __half* __restrict__ lds_base,
    int col_in_tile,
    int k_set,
    int stride_n,
    half8& b_reg)
{
    b_reg[0] = lds_base[(k_set + 0)  * stride_n + col_in_tile];
    b_reg[1] = lds_base[(k_set + 1)  * stride_n + col_in_tile];
    b_reg[2] = lds_base[(k_set + 2)  * stride_n + col_in_tile];
    b_reg[3] = lds_base[(k_set + 3)  * stride_n + col_in_tile];
    b_reg[4] = lds_base[(k_set + 8)  * stride_n + col_in_tile];
    b_reg[5] = lds_base[(k_set + 9)  * stride_n + col_in_tile];
    b_reg[6] = lds_base[(k_set + 10) * stride_n + col_in_tile];
    b_reg[7] = lds_base[(k_set + 11) * stride_n + col_in_tile];
}

__global__ void wmma_attention_prefill_64_kernel(
    const __half* __restrict__ Q,
    const __half* __restrict__ K,
    const __half* __restrict__ V,
    float*        __restrict__ O,
    float scale)
{
    const int tid     = threadIdx.x;   // 0..127
    const int wave_id = tid >> 5;      // 0..3
    const int lane    = tid & 31;      // 0..31
    const int lane_col_in_block = lane & 15;
    const int lane_k_set        = (lane >> 4) * 4;
    const int lane_row_start    = (lane >> 4) * 8;
    const int wave_row_base     = wave_id * 16;

    // LDS layout — static because the PoC target (1 resident WG) is fine.
    __shared__ __half lds_q[SEQ * TILE_K];              // 2 KB
    __shared__ __half lds_k[TILE_K * SEQ];              // 2 KB (K^T tile)
    __shared__ float  lds_s[SEQ * SEQ];                 // 16 KB (S in FP32)
    __shared__ __half lds_p[SEQ * SEQ];                 // 8 KB (P in FP16)
    __shared__ __half lds_v[TILE_K * HEAD_DIM];         // 4 KB (V tile per K step)

    // =========================================================
    // PHASE 1 — S = Q · K^T   (M=64, N=64, K=128)
    // =========================================================
    float8 acc_s[COL_BLOCKS];
    #pragma unroll
    for (int cb = 0; cb < COL_BLOCKS; ++cb) {
        acc_s[cb] = {0, 0, 0, 0, 0, 0, 0, 0};
    }

    const int num_k_tiles_1 = HEAD_DIM / TILE_K;  // 8

    for (int kt = 0; kt < num_k_tiles_1; ++kt) {
        const int k_start = kt * TILE_K;

        // Load Q tile [64 × 16] into lds_q.
        // 1024 halfs / 128 threads = 8 per thread. Two threads cover
        // each row of 16 halfs → contiguous within rows.
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int flat = tid * 8 + i;
            const int row = flat / TILE_K;
            const int col = flat & (TILE_K - 1);
            lds_q[flat] = Q[row * HEAD_DIM + k_start + col];
        }

        // Load K^T tile [16 × 64] into lds_k. Because we want to compute
        // Q · K^T and K is stored row-major [64 × 128], we access K with
        // swapped indices: lds_k[k_in_tile][j] = K[j][k_start + k_in_tile].
        // lds_k is therefore the "transposed-at-load" view.
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int flat = tid * 8 + i;
            const int k_in_tile = flat / SEQ;   // 0..15
            const int j         = flat & (SEQ - 1);   // 0..63
            lds_k[k_in_tile * SEQ + j] = K[j * HEAD_DIM + k_start + k_in_tile];
        }

        __syncthreads();

        const int a_row_in_tile = wave_row_base + (lane & 15);
        half8 a_reg;
        load_a_reg_from_lds(lds_q, a_row_in_tile, lane_k_set, TILE_K, a_reg);

        #pragma unroll
        for (int cb = 0; cb < COL_BLOCKS; ++cb) {
            const int col_in_lds = cb * 16 + lane_col_in_block;
            half8 b_reg;
            load_b_reg_from_lds(lds_k, col_in_lds, lane_k_set, SEQ, b_reg);
            acc_s[cb] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(
                a_reg, b_reg, acc_s[cb]);
        }

        __syncthreads();
    }

    // Scale and store S to LDS for softmax.
    #pragma unroll
    for (int cb = 0; cb < COL_BLOCKS; ++cb) {
        const int col = cb * 16 + lane_col_in_block;
        #pragma unroll
        for (int v = 0; v < 8; ++v) {
            const int row = wave_row_base + lane_row_start + v;
            lds_s[row * SEQ + col] = acc_s[cb][v] * scale;
        }
    }
    __syncthreads();

    // =========================================================
    // PHASE 2 — P = softmax(S)  (row-wise, one thread per row)
    // =========================================================
    if (tid < SEQ) {
        const int row = tid;
        float* s_row = &lds_s[row * SEQ];
        __half* p_row = &lds_p[row * SEQ];

        // Row max (stable softmax).
        float m = s_row[0];
        #pragma unroll
        for (int j = 1; j < SEQ; ++j) {
            m = fmaxf(m, s_row[j]);
        }

        // Exp + sum.
        float sum = 0.0f;
        float exps[SEQ];
        #pragma unroll
        for (int j = 0; j < SEQ; ++j) {
            const float e = expf(s_row[j] - m);
            exps[j] = e;
            sum += e;
        }

        const float inv_sum = 1.0f / sum;

        // Normalise + downcast to FP16.
        #pragma unroll
        for (int j = 0; j < SEQ; ++j) {
            p_row[j] = __float2half(exps[j] * inv_sum);
        }
    }
    __syncthreads();

    // =========================================================
    // PHASE 3 — O = P · V   (M=64, N=128, K=64)
    //
    // Run two 64-wide column chunks back-to-back. Each chunk covers 64
    // output columns via 4 WMMA col blocks, reusing the same P loaded
    // once in LDS. V reload per chunk.
    // =========================================================
    const int num_k_tiles_3 = SEQ / TILE_K;  // 4

    for (int col_chunk = 0; col_chunk < HEAD_DIM / SEQ; ++col_chunk) {
        const int n_col_base = col_chunk * SEQ;

        float8 acc_o[COL_BLOCKS];
        #pragma unroll
        for (int cb = 0; cb < COL_BLOCKS; ++cb) {
            acc_o[cb] = {0, 0, 0, 0, 0, 0, 0, 0};
        }

        for (int kt = 0; kt < num_k_tiles_3; ++kt) {
            const int k_start = kt * TILE_K;

            // Reuse lds_v's 4 KB buffer to hold a [16 × 64] V tile for
            // this chunk. Even though lds_v was declared as [16 × 128]
            // (4 KB) we only use the first [16 × 64] = 2 KB here.
            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                const int flat = tid * 8 + i;
                const int k_in_tile = flat / SEQ;
                const int n_in_chunk = flat & (SEQ - 1);
                lds_v[k_in_tile * SEQ + n_in_chunk] =
                    V[(k_start + k_in_tile) * HEAD_DIM + n_col_base + n_in_chunk];
            }

            __syncthreads();

            // A-side load from lds_p: row stride is SEQ (not TILE_K), so
            // we inline the pattern from load_a_reg_from_lds with a
            // custom base offset that includes the K-tile column shift.
            const int a_row_in_tile = wave_row_base + (lane & 15);
            half8 a_reg;
            {
                const int base = a_row_in_tile * SEQ + k_start;
                a_reg[0] = lds_p[base + lane_k_set + 0];
                a_reg[1] = lds_p[base + lane_k_set + 1];
                a_reg[2] = lds_p[base + lane_k_set + 2];
                a_reg[3] = lds_p[base + lane_k_set + 3];
                a_reg[4] = lds_p[base + lane_k_set + 8];
                a_reg[5] = lds_p[base + lane_k_set + 9];
                a_reg[6] = lds_p[base + lane_k_set + 10];
                a_reg[7] = lds_p[base + lane_k_set + 11];
            }

            #pragma unroll
            for (int cb = 0; cb < COL_BLOCKS; ++cb) {
                const int col_in_lds = cb * 16 + lane_col_in_block;
                half8 b_reg;
                load_b_reg_from_lds(lds_v, col_in_lds, lane_k_set, SEQ, b_reg);
                acc_o[cb] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(
                    a_reg, b_reg, acc_o[cb]);
            }

            __syncthreads();
        }

        // Store this 64×64 chunk of O to global FP32.
        #pragma unroll
        for (int cb = 0; cb < COL_BLOCKS; ++cb) {
            const int out_col_in_chunk = cb * 16 + lane_col_in_block;
            const int out_col = n_col_base + out_col_in_chunk;
            #pragma unroll
            for (int v = 0; v < 8; ++v) {
                const int out_row = wave_row_base + lane_row_start + v;
                O[out_row * HEAD_DIM + out_col] = acc_o[cb][v];
            }
        }
    }
}

extern "C" hipError_t wmma_attention_prefill_64_launch(
    const void* Q,
    const void* K,
    const void* V,
    float*      O,
    float       scale,
    hipStream_t stream)
{
    hipLaunchKernelGGL(
        wmma_attention_prefill_64_kernel,
        dim3(1, 1, 1), dim3(THREADS, 1, 1),
        0, stream,
        reinterpret_cast<const __half*>(Q),
        reinterpret_cast<const __half*>(K),
        reinterpret_cast<const __half*>(V),
        O,
        scale
    );
    return hipGetLastError();
}
