// Phase 4 Step 4 — WMMA tiled GEMM with inline Q4_1 weight dequant on gfx12.
//
// Structural copy of `wmma_gemm_q4_0.hip`. Only three things change:
//   * block size:   18 → 20 bytes  (Q4_1 carries a per-block `min`)
//   * nibble offset: +2 → +4 bytes within a block  (because of the min)
//   * dequant:      (nib - 8) * scale   →   nib * scale + min
//
// Q4_1 block (20 bytes):
//   [0..1]  FP16 scale (d)
//   [2..3]  FP16 min   (m)
//   [4..19] 16 bytes packed as
//     qs[i] & 0x0F -> element  i   (i ∈ 0..15)
//     qs[i] >> 4   -> element i+16 (i ∈ 0..15)
//
// The K-loop, WMMA issue, LDS layout, output store — everything else is
// byte-identical to the Q4_0 kernel.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

using half8  = _Float16 __attribute__((ext_vector_type(8)));
using float8 = float    __attribute__((ext_vector_type(8)));

#define TILE_M          64
#define TILE_N          64
#define TILE_K          16
#define K_CHUNK         32           // one Q4_1 block wide
#define WARPS_PER_BLOCK 4
#define THREADS_PER_BLOCK (WARPS_PER_BLOCK * 32)
#define COL_BLOCKS      4            // 64 / 16
#define Q4_1_BLOCK_BYTES 20

__global__ void wmma_gemm_q4_1_kernel(
    const float*   __restrict__ A,       // [M × K] FP32 row-major
    const uint8_t* __restrict__ W,       // Q4_1 weights [N × K/32] blocks of 20 B
    float*         __restrict__ D,       // [M × N] FP32 row-major
    int M, int N, int K)
{
    const int tile_m = blockIdx.y * TILE_M;
    const int tile_n = blockIdx.x * TILE_N;

    const int tid     = threadIdx.x;       // 0..127
    const int wave_id = tid >> 5;          // 0..3
    const int lane    = tid & 31;          // 0..31

    __shared__ __half lds_a[TILE_M * K_CHUNK];   // 64 × 32 = 4 KB
    __shared__ __half lds_b[K_CHUNK * TILE_N];   // 32 × 64 = 4 KB

    float8 acc[COL_BLOCKS];
    #pragma unroll
    for (int b = 0; b < COL_BLOCKS; ++b) {
        acc[b] = {0, 0, 0, 0, 0, 0, 0, 0};
    }

    const int a_row_in_tile = wave_id * 16 + (lane & 15);
    const int a_k_set       = (lane >> 4) * 4;
    const int col_in_lane   = lane & 15;
    const int b_k_set       = (lane >> 4) * 4;

    const int blocks_per_row = K / 32;
    const int num_k_chunks   = K / K_CHUNK;

    for (int kc = 0; kc < num_k_chunks; ++kc) {
        const int k_start_abs = kc * K_CHUNK;

        // ---- Cooperative global → LDS load of A tile [64 × 32] --------
        #pragma unroll
        for (int i = 0; i < 16; ++i) {
            const int flat        = tid * 16 + i;
            const int row_in_tile = flat / K_CHUNK;
            const int k_in_tile   = flat & (K_CHUNK - 1);
            const float v = A[(tile_m + row_in_tile) * K + k_start_abs + k_in_tile];
            lds_a[flat] = __float2half(v);
        }

        // ---- Cooperative Q4_1 → FP16 dequant into LDS -----------------
        // 64 columns (one Q4_1 block per column per K_CHUNK) × 32 K-values.
        // 128 threads, 2 per column (one per block half).
        {
            const int col  = tid >> 1;         // 0..63  → output column in tile
            const int half = tid & 1;          // 0 = lower 16 elems, 1 = upper 16

            const size_t block_ofs =
                (size_t)(tile_n + col) * (size_t)blocks_per_row * Q4_1_BLOCK_BYTES +
                (size_t)kc * Q4_1_BLOCK_BYTES;
            const uint8_t* block_ptr = W + block_ofs;

            // Scale (FP16) at byte 0, Min (FP16) at byte 2.
            __half scale_h;
            __half min_h;
            memcpy(&scale_h, block_ptr,     sizeof(__half));
            memcpy(&min_h,   block_ptr + 2, sizeof(__half));
            const float scale_f = __half2float(scale_h);
            const float min_f   = __half2float(min_h);

            // Nibbles start at byte 4 (not 2 — that's the Q4_0 offset).
            const int qs_ofs = 4 + half * 8;
            uint8_t my_qs[8];
            #pragma unroll
            for (int i = 0; i < 8; ++i) my_qs[i] = block_ptr[qs_ofs + i];

            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                const int k_lo = half * 8 + i;
                const int k_hi = k_lo + 16;
                // Q4_1 dequant: q * scale + min (unsigned nibble, no -8).
                const int nib_lo = int(my_qs[i] & 0x0F);
                const int nib_hi = int(my_qs[i] >> 4);
                lds_b[k_lo * TILE_N + col] = __float2half(float(nib_lo) * scale_f + min_f);
                lds_b[k_hi * TILE_N + col] = __float2half(float(nib_hi) * scale_f + min_f);
            }
        }

        __syncthreads();

        // ---- Two WMMA iterations per K_CHUNK (k_sub = 0 and 16) -------
        #pragma unroll
        for (int k_sub = 0; k_sub < K_CHUNK; k_sub += TILE_K) {

            half8 a_reg;
            {
                const int base = a_row_in_tile * K_CHUNK + k_sub;
                a_reg[0] = lds_a[base + a_k_set + 0];
                a_reg[1] = lds_a[base + a_k_set + 1];
                a_reg[2] = lds_a[base + a_k_set + 2];
                a_reg[3] = lds_a[base + a_k_set + 3];
                a_reg[4] = lds_a[base + a_k_set + 8];
                a_reg[5] = lds_a[base + a_k_set + 9];
                a_reg[6] = lds_a[base + a_k_set + 10];
                a_reg[7] = lds_a[base + a_k_set + 11];
            }

            #pragma unroll
            for (int cb = 0; cb < COL_BLOCKS; ++cb) {
                const int b_col_in_tile = cb * 16 + col_in_lane;

                half8 b_reg;
                b_reg[0] = lds_b[(k_sub + b_k_set + 0)  * TILE_N + b_col_in_tile];
                b_reg[1] = lds_b[(k_sub + b_k_set + 1)  * TILE_N + b_col_in_tile];
                b_reg[2] = lds_b[(k_sub + b_k_set + 2)  * TILE_N + b_col_in_tile];
                b_reg[3] = lds_b[(k_sub + b_k_set + 3)  * TILE_N + b_col_in_tile];
                b_reg[4] = lds_b[(k_sub + b_k_set + 8)  * TILE_N + b_col_in_tile];
                b_reg[5] = lds_b[(k_sub + b_k_set + 9)  * TILE_N + b_col_in_tile];
                b_reg[6] = lds_b[(k_sub + b_k_set + 10) * TILE_N + b_col_in_tile];
                b_reg[7] = lds_b[(k_sub + b_k_set + 11) * TILE_N + b_col_in_tile];

                acc[cb] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(
                    a_reg, b_reg, acc[cb]);
            }
        }

        __syncthreads();
    }

    // ---- Store ---------------------------------------------------------
    const int out_col_in_lane  = lane & 15;
    const int out_row_start    = (lane >> 4) * 8;
    const int wave_row_in_tile = wave_id * 16;

    #pragma unroll
    for (int cb = 0; cb < COL_BLOCKS; ++cb) {
        const int out_col_in_tile = cb * 16 + out_col_in_lane;
        const int global_col      = tile_n + out_col_in_tile;

        #pragma unroll
        for (int v = 0; v < 8; ++v) {
            const int out_row_in_tile = wave_row_in_tile + out_row_start + v;
            const int global_row      = tile_m + out_row_in_tile;
            D[global_row * N + global_col] = acc[cb][v];
        }
    }
}

extern "C" hipError_t wmma_gemm_q4_1_launch(
    const float*   input,
    const void*    weights_q4_1,
    float*         output,
    int M, int N, int K,
    hipStream_t stream)
{
    if (M <= 0 || N <= 0 || K <= 0) return hipErrorInvalidValue;
    if ((M % TILE_M) != 0)  return hipErrorInvalidValue;
    if ((N % TILE_N) != 0)  return hipErrorInvalidValue;
    if ((K % K_CHUNK) != 0) return hipErrorInvalidValue;

    dim3 grid(N / TILE_N, M / TILE_M, 1);
    dim3 block(THREADS_PER_BLOCK, 1, 1);

    hipLaunchKernelGGL(
        wmma_gemm_q4_1_kernel,
        grid, block,
        0, stream,
        input,
        reinterpret_cast<const uint8_t*>(weights_q4_1),
        output,
        M, N, K
    );
    return hipGetLastError();
}
