// Phase 7 Step 3 — WMMA tiled GEMM with inline Q4_K_M weight dequant on
// gfx12. Structural copy of `wmma_gemm_q4_0.hip`. K-loop cadence, WMMA
// issue, LDS layout, output store — all identical. Only the weight-load
// + dequant block differs.
//
// Q4_K super-block (144 bytes, 256 elements):
//   [  0..1 ]  FP16 d     — super-block scale
//   [  2..3 ]  FP16 dmin  — super-block min
//   [  4..15]  scales[12] — 8×6-bit scales + 8×6-bit mins, packed 4:2
//   [ 16..143] qs[128]    — 4-bit quantised values, paired-sub-block
//                           interleaving (see docs/q4_k_m_block_format.md §4)
//
// Unpacked per sub-block j ∈ [0, 8):
//   if j < 4:  scale_j = scales[j]     & 0x3F          (low 6 bits)
//              min_j   = scales[j + 4] & 0x3F
//   else:      scale_j = (scales[j+4] & 0x0F) | ((scales[j-4] >> 6) << 4)
//              min_j   = (scales[j+4] >>  4) | ((scales[j]   >> 6) << 4)
//
// Nibble pairing: sub-blocks (0,1), (2,3), (4,5), (6,7) share a 32-byte
// qs region. Even sub-block takes low nibbles, odd takes high nibbles.
//
// Dequant: value = (d * scale_j) * nibble - (dmin * min_j)
// There is no /64 anywhere. The 6-bit values are raw integers 0..63.
//
// Register layout reference: docs/wmma_register_layout_gfx12.md.
// Ground-truth format: docs/q4_k_m_block_format.md.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <string.h>

using half8  = _Float16 __attribute__((ext_vector_type(8)));
using float8 = float    __attribute__((ext_vector_type(8)));

#define TILE_M           64
#define TILE_N           64
#define TILE_K           16
#define K_CHUNK          32            // one Q4_K sub-block wide
#define WARPS_PER_BLOCK  4
#define THREADS_PER_BLOCK (WARPS_PER_BLOCK * 32)
#define COL_BLOCKS       4             // 64 / 16
#define Q4_K_BLOCK_BYTES 144
#define Q4_K_ELEMS_PER_BLOCK 256
#define Q4_K_SUB_BLOCKS  8

__global__ void wmma_gemm_q4_k_kernel(
    const float*   __restrict__ A,       // [M × K] FP32 row-major
    const uint8_t* __restrict__ W,       // Q4_K weights [N × K/256] super-blocks
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

    const int super_blocks_per_row = K / Q4_K_ELEMS_PER_BLOCK;
    const int num_k_chunks         = K / K_CHUNK;

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

        // ---- Cooperative Q4_K → FP16 dequant into LDS -----------------
        // 64 output columns × 32 K-values per sub-block.
        // 128 threads, 2 per column — each thread handles 16 K-values.
        {
            const int col  = tid >> 1;         // 0..63  → N column in tile
            const int half = tid & 1;          // 0 → K[0..15], 1 → K[16..31]

            // Locate this K-chunk's super-block and sub-index.
            const int super_block_idx = kc >> 3;   // kc / 8
            const int sub_idx         = kc & 7;    // kc % 8

            // Super-block byte offset in the weight buffer.
            const size_t sb_ofs =
                (size_t)(tile_n + col) * (size_t)super_blocks_per_row * Q4_K_BLOCK_BYTES
                + (size_t)super_block_idx * Q4_K_BLOCK_BYTES;
            const uint8_t* sb_ptr = W + sb_ofs;

            // Super-block header (FP16 d, FP16 dmin).
            __half d_h, dmin_h;
            memcpy(&d_h,    sb_ptr + 0, sizeof(__half));
            memcpy(&dmin_h, sb_ptr + 2, sizeof(__half));
            const float d_f    = __half2float(d_h);
            const float dmin_f = __half2float(dmin_h);

            // 6-bit scale/min unpack for sub-block `sub_idx`.
            const uint8_t* scales = sb_ptr + 4;
            int scale_j, min_j;
            if (sub_idx < 4) {
                scale_j = scales[sub_idx]     & 0x3F;
                min_j   = scales[sub_idx + 4] & 0x3F;
            } else {
                scale_j = (scales[sub_idx + 4] & 0x0F) | ((scales[sub_idx - 4] >> 6) << 4);
                min_j   = (scales[sub_idx + 4] >> 4)   | ((scales[sub_idx]     >> 6) << 4);
            }
            const float d_scale = d_f    * (float)scale_j;   // per-element MUL factor
            const float d_mn    = dmin_f * (float)min_j;     // per-element SUB offset

            // Nibble region for the sub-block pair (sub_idx paired with sub_idx^1).
            const int pair_base = (sub_idx >> 1) * 32;
            const bool is_upper = (sub_idx & 1) != 0;
            const uint8_t* qs_base = sb_ptr + 16 + pair_base + half * 16;

            // Each thread writes 16 LDS slots for K in {half*16 .. half*16+15}.
            #pragma unroll
            for (int i = 0; i < 16; ++i) {
                const uint8_t byte = qs_base[i];
                const int nib = is_upper ? (byte >> 4) : (byte & 0x0F);
                const int k   = half * 16 + i;
                lds_b[k * TILE_N + col] = __float2half(d_scale * (float)nib - d_mn);
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

    // ---- Store -----------------------------------------------------------
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

extern "C" hipError_t wmma_gemm_q4_k_launch(
    const float*   input,
    const void*    weights_q4_k,
    float*         output,
    int M, int N, int K,
    hipStream_t stream)
{
    if (M <= 0 || N <= 0 || K <= 0) return hipErrorInvalidValue;
    if ((M % TILE_M) != 0)  return hipErrorInvalidValue;
    if ((N % TILE_N) != 0)  return hipErrorInvalidValue;
    if ((K % Q4_K_ELEMS_PER_BLOCK) != 0) return hipErrorInvalidValue;

    dim3 grid(N / TILE_N, M / TILE_M, 1);
    dim3 block(THREADS_PER_BLOCK, 1, 1);

    hipLaunchKernelGGL(
        wmma_gemm_q4_k_kernel,
        grid, block,
        0, stream,
        input,
        reinterpret_cast<const uint8_t*>(weights_q4_k),
        output,
        M, N, K
    );
    return hipGetLastError();
}
