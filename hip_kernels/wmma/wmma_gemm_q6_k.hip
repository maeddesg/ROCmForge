// Phase 8b Step 2 — WMMA tiled GEMM with inline Q6_K weight dequant on
// gfx12. Structural copy of `wmma_gemm_q4_k.hip`. K-loop cadence, WMMA
// issue, LDS layout, output store — all identical. Only the weight-load
// + dequant block differs.
//
// Q6_K super-block (210 bytes, 256 elements):
//   [  0..127] ql[128]   — low 4 bits per element
//   [128..191] qh[64]    — high 2 bits per element (stride-32 packed)
//   [192..207] scales[16]— int8 per-sub-block scale (16 sub-blocks × 16 el)
//   [208..209] d (FP16)  — super-block scale
//
// Dequant (per element e ∈ [0, 256), ground truth ggml-quants.c:1888):
//   sub_idx  = e / 16
//   n_half   = e / 128                       half = 0..1
//   e_half   = e % 128
//   which_q  = e_half / 32                   0..3
//   l        = e_half % 32                   0..31
//   ql_byte  = ql[n_half*64 + (which_q&1)*32 + l]
//   qh_byte  = qh[n_half*32 + l]
//   ql_nib   = (which_q >= 2) ? (ql_byte >> 4) : (ql_byte & 0xF)
//   qh_bits  = (qh_byte >> (2*which_q)) & 0x3
//   q6       = ql_nib | (qh_bits << 4)                      // 0..63
//   value    = d * scales[sub_idx] * (q6 - 32)
//
// Ground-truth format: docs/q6_k_block_format.md.
// Register layout reference: docs/wmma_register_layout_gfx12.md.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <string.h>

using half8  = _Float16 __attribute__((ext_vector_type(8)));
using float8 = float    __attribute__((ext_vector_type(8)));

#define TILE_M           64
#define TILE_N           64
#define TILE_K           16
#define K_CHUNK          32            // 32 elements = one half of a Q6_K strip
#define WARPS_PER_BLOCK  4
#define THREADS_PER_BLOCK (WARPS_PER_BLOCK * 32)
#define COL_BLOCKS       4             // 64 / 16
#define Q6_K_BLOCK_BYTES 210
#define Q6_K_ELEMS_PER_BLOCK 256

__global__ void wmma_gemm_q6_k_kernel(
    const float*   __restrict__ A,       // [M × K] FP32 row-major
    const uint8_t* __restrict__ W,       // Q6_K weights [N × K/256] super-blocks
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

    const int super_blocks_per_row = K / Q6_K_ELEMS_PER_BLOCK;
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

        // ---- Cooperative Q6_K → FP16 dequant into LDS -----------------
        // 64 output columns × 32 K-values per K_CHUNK.
        // 128 threads, 2 per column — each thread handles 16 K-values
        // from one 16-element sub-block (thus one `sub_idx`, one scale).
        {
            const int col  = tid >> 1;         // 0..63  → N column in tile
            const int half = tid & 1;          // 0 → K[0..15], 1 → K[16..31]

            // Locate this K-chunk's super-block and sub-index.
            const int super_block_idx = kc >> 3;   // kc / 8  (8 K_CHUNKs per SB)
            const int chunk_in_sb     = kc & 7;    // kc % 8

            // Super-block byte offset in the weight buffer.
            const size_t sb_ofs =
                (size_t)(tile_n + col) * (size_t)super_blocks_per_row * Q6_K_BLOCK_BYTES
                + (size_t)super_block_idx * Q6_K_BLOCK_BYTES;
            const uint8_t* sb_ptr = W + sb_ofs;

            // Super-block scale (FP16 at end of block).
            __half d_h;
            memcpy(&d_h, sb_ptr + 208, sizeof(__half));
            const float d_f = __half2float(d_h);

            // Per-element closed form from docs/q6_k_block_format.md §3.3.
            // This thread owns elements [chunk_base, chunk_base + 16) where
            //   chunk_base = chunk_in_sb * 32 + half * 16
            // Since chunk_base is a multiple of 16, sub_idx / n_half /
            // which_q / l_base are constant across the thread's 16 elements.
            const int chunk_base = chunk_in_sb * 32 + half * 16;
            const int sub_idx    = chunk_base >> 4;          // 0..15
            const int n_half     = chunk_base >> 7;          // 0 or 1
            const int e_half     = chunk_base & 127;         // 0..112, multiple of 16
            const int which_q    = e_half >> 5;              // 0..3
            const int l_base     = e_half & 31;              // 0 or 16
            const bool is_high_nib = (which_q >= 2);
            const int qh_shift   = 2 * which_q;              // 0, 2, 4, 6

            const int8_t* scales = reinterpret_cast<const int8_t*>(sb_ptr + 192);
            const float d_scale = d_f * (float)scales[sub_idx];

            // Base pointers into ql and qh for this thread's 16 elements.
            const uint8_t* ql_base = sb_ptr
                + (size_t)n_half * 64
                + (size_t)((which_q & 1) * 32)
                + (size_t)l_base;
            const uint8_t* qh_base = sb_ptr
                + 128
                + (size_t)n_half * 32
                + (size_t)l_base;

            // Each thread writes 16 LDS slots for K in {half*16 .. half*16+15}.
            #pragma unroll
            for (int i = 0; i < 16; ++i) {
                const uint8_t ql_byte = ql_base[i];
                const uint8_t qh_byte = qh_base[i];
                const int ql_nib = is_high_nib ? (ql_byte >> 4) : (ql_byte & 0x0F);
                const int qh_bits = (qh_byte >> qh_shift) & 0x3;
                const int q6 = ql_nib | (qh_bits << 4);      // 0..63
                const int signed_q = q6 - 32;                // -32..31
                const int k = half * 16 + i;
                lds_b[k * TILE_N + col] = __float2half(d_scale * (float)signed_q);
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

extern "C" hipError_t wmma_gemm_q6_k_launch(
    const float*   input,
    const void*    weights_q6_k,
    float*         output,
    int M, int N, int K,
    hipStream_t stream)
{
    if (M <= 0 || N <= 0 || K <= 0) return hipErrorInvalidValue;
    if ((M % TILE_M) != 0)  return hipErrorInvalidValue;
    if ((N % TILE_N) != 0)  return hipErrorInvalidValue;
    if ((K % Q6_K_ELEMS_PER_BLOCK) != 0) return hipErrorInvalidValue;

    dim3 grid(N / TILE_N, M / TILE_M, 1);
    dim3 block(THREADS_PER_BLOCK, 1, 1);

    hipLaunchKernelGGL(
        wmma_gemm_q6_k_kernel,
        grid, block,
        0, stream,
        input,
        reinterpret_cast<const uint8_t*>(weights_q6_k),
        output,
        M, N, K
    );
    return hipGetLastError();
}
