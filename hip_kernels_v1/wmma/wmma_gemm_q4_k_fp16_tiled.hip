// Template-parametrised WMMA GEMM for Q4_K × FP16 accumulating to FP32.
// Instantiated for two tile shapes:
//   Tiled<64, 64, 32>    — parity-reference against the hard-coded
//                          wmma_gemm_q4_k_fp16.hip (must be bit-identical).
//   Tiled<128, 128, 32>  — the llama.cpp MMQ RDNA2+ equivalent: 2× tile
//                          in both M and N, 8 warps/block, ~16 KB LDS.
//
// Derived compile-time constants:
//   WARPS_PER_BLOCK = TM / 16
//   COL_BLOCKS      = TN / 16
//   K_ITERS         = KC / TILE_K  (TILE_K is 16 = WMMA K-dim)
// Thread layout: WARPS_PER_BLOCK × 32 = TM * 2 threads per block.
// All per-wave row/col/k math is TM/TN agnostic — only the COUNT of
// waves changes. Tested: the 64,64,32 instantiation reproduces the
// hard-coded kernel bit-for-bit (see tests_v1/wmma_tiled_test.rs).

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>
#include <cstring>

using half8  = _Float16 __attribute__((ext_vector_type(8)));
using float8 = float    __attribute__((ext_vector_type(8)));

__device__ __forceinline__ void rf_v1_wmma_tiled_set_ieee_denormal_mode() {
    asm volatile("s_setreg_imm32_b32 hwreg(HW_REG_MODE, 4, 4), 0xF");
}

#define Q4_K_BLOCK_BYTES       144
#define Q4_K_ELEMS_PER_BLOCK   256
#define TILE_K_WMMA            16

template<int TM, int TN, int KC>
__global__ void rf_v1_wmma_gemm_q4_k_fp16_tiled_kernel(
    const float*   __restrict__ A,
    const uint8_t* __restrict__ W,
    float*         __restrict__ D,
    int M, int N, int K)
{
    static_assert(TM % 16 == 0, "TM must be a multiple of 16 (WMMA row dim)");
    static_assert(TN % 16 == 0, "TN must be a multiple of 16 (WMMA col dim)");
    static_assert(KC % TILE_K_WMMA == 0, "KC must be a multiple of 16 (WMMA K dim)");
    static_assert((TN % 2) == 0, "TN must be even (2 threads per column in B-load)");

    constexpr int WARPS_PER_BLOCK = TM / 16;
    constexpr int THREADS         = WARPS_PER_BLOCK * 32;
    constexpr int COL_BLOCKS      = TN / 16;
    constexpr int K_ITERS         = KC / TILE_K_WMMA;

    rf_v1_wmma_tiled_set_ieee_denormal_mode();

    const int tile_m = blockIdx.y * TM;
    const int tile_n = blockIdx.x * TN;

    const int tid     = threadIdx.x;
    const int wave_id = tid >> 5;
    const int lane    = tid & 31;

    __shared__ __half lds_a[TM * KC];
    __shared__ __half lds_b[KC * TN];

    float8 acc[COL_BLOCKS];
    #pragma unroll
    for (int b = 0; b < COL_BLOCKS; ++b) {
        acc[b] = {0, 0, 0, 0, 0, 0, 0, 0};
    }

    // Per-wave row/col/k addressing — identical layout as the
    // hard-coded 64×64 kernel. Each wave still owns 16 rows of A and
    // 16 half-rows of C per COL_BLOCK (lane 0-15 vs 16-31 differ by a
    // k_set offset only, matching the WMMA FP16 fragment layout).
    const int a_row_in_tile = wave_id * 16 + (lane & 15);
    const int a_k_set       = (lane >> 4) * 4;
    const int col_in_lane   = lane & 15;
    const int b_k_set       = (lane >> 4) * 4;

    const int super_blocks_per_row = K / Q4_K_ELEMS_PER_BLOCK;
    const int num_k_chunks         = K / KC;

    // ── Outer K loop: one K_CHUNK slice at a time ──────────────────
    for (int kc = 0; kc < num_k_chunks; ++kc) {
        const int k_start_abs = kc * KC;

        // A-tile load. Each thread loads 16 halfs contiguously.
        // For TM×KC halfs total / THREADS threads = 16 halfs/thread
        // when THREADS × 16 == TM × KC. Holds for (64,64,32):4W:128T
        // and (128,128,32):8W:256T both.
        static_assert((TM * KC) == (THREADS * 16),
            "A-tile partitioning requires TM*KC == THREADS*16");
        #pragma unroll
        for (int i = 0; i < 16; ++i) {
            const int flat        = tid * 16 + i;
            const int row_in_tile = flat / KC;
            const int k_in_tile   = flat & (KC - 1);
            const float v = A[(tile_m + row_in_tile) * K + k_start_abs + k_in_tile];
            lds_a[flat] = __float2half(v);
        }

        // B-tile dequant from Q4_K. 2 threads per output column.
        // For TN columns and 2 threads/col: need THREADS == 2*TN.
        // (64,64,32):128T==2*64 ✓   (128,128,32):256T==2*128 ✓
        static_assert(THREADS == 2 * TN,
            "B-tile dequant pattern requires THREADS == 2*TN");
        {
            const int col  = tid >> 1;
            const int half = tid & 1;

            const int super_block_idx = kc >> 3;
            const int sub_idx         = kc & 7;

            const size_t sb_ofs =
                (size_t)(tile_n + col) * (size_t)super_blocks_per_row * Q4_K_BLOCK_BYTES
                + (size_t)super_block_idx * Q4_K_BLOCK_BYTES;
            const uint8_t* sb_ptr = W + sb_ofs;

            __half d_h, dmin_h;
            std::memcpy(&d_h,    sb_ptr + 0, sizeof(__half));
            std::memcpy(&dmin_h, sb_ptr + 2, sizeof(__half));
            const float d_f    = __half2float(d_h);
            const float dmin_f = __half2float(dmin_h);

            const uint8_t* scales = sb_ptr + 4;
            int scale_j, min_j;
            if (sub_idx < 4) {
                scale_j = scales[sub_idx]     & 0x3F;
                min_j   = scales[sub_idx + 4] & 0x3F;
            } else {
                scale_j = (scales[sub_idx + 4] & 0x0F) | ((scales[sub_idx - 4] >> 6) << 4);
                min_j   = (scales[sub_idx + 4] >> 4)   | ((scales[sub_idx]     >> 6) << 4);
            }
            const float d_scale = d_f    * (float)scale_j;
            const float d_mn    = dmin_f * (float)min_j;

            const int pair_base = (sub_idx >> 1) * 32;
            const bool is_upper = (sub_idx & 1) != 0;
            const uint8_t* qs_base = sb_ptr + 16 + pair_base + half * 16;

            #pragma unroll
            for (int i = 0; i < 16; ++i) {
                const uint8_t byte = qs_base[i];
                const int nib = is_upper ? (byte >> 4) : (byte & 0x0F);
                const int k   = half * 16 + i;
                lds_b[k * TN + col] = __float2half(d_scale * (float)nib - d_mn);
            }
        }

        __syncthreads();

        // ── WMMA accumulation loop ──────────────────────────────────
        #pragma unroll
        for (int k_sub = 0; k_sub < KC; k_sub += TILE_K_WMMA) {
            half8 a_reg;
            {
                const int base = a_row_in_tile * KC + k_sub;
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
                b_reg[0] = lds_b[(k_sub + b_k_set + 0)  * TN + b_col_in_tile];
                b_reg[1] = lds_b[(k_sub + b_k_set + 1)  * TN + b_col_in_tile];
                b_reg[2] = lds_b[(k_sub + b_k_set + 2)  * TN + b_col_in_tile];
                b_reg[3] = lds_b[(k_sub + b_k_set + 3)  * TN + b_col_in_tile];
                b_reg[4] = lds_b[(k_sub + b_k_set + 8)  * TN + b_col_in_tile];
                b_reg[5] = lds_b[(k_sub + b_k_set + 9)  * TN + b_col_in_tile];
                b_reg[6] = lds_b[(k_sub + b_k_set + 10) * TN + b_col_in_tile];
                b_reg[7] = lds_b[(k_sub + b_k_set + 11) * TN + b_col_in_tile];

                acc[cb] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(
                    a_reg, b_reg, acc[cb]);
            }
        }

        __syncthreads();
    }

    // ── Write-out ───────────────────────────────────────────────────
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

// ── Launch helpers ─────────────────────────────────────────────────────────

template<int TM, int TN, int KC>
static inline hipError_t rf_v1_launch_tiled(
    const float*   A,
    const uint8_t* W,
    float*         D,
    int M, int N, int K,
    hipStream_t stream)
{
    if (M <= 0 || N <= 0 || K <= 0)                  return hipErrorInvalidValue;
    if ((M % TM) != 0)                               return hipErrorInvalidValue;
    if ((N % TN) != 0)                               return hipErrorInvalidValue;
    if ((K % Q4_K_ELEMS_PER_BLOCK) != 0)             return hipErrorInvalidValue;
    if ((K % KC) != 0)                               return hipErrorInvalidValue;

    constexpr int THREADS = (TM / 16) * 32;

    dim3 grid(N / TN, M / TM, 1);
    dim3 block(THREADS, 1, 1);

    rf_v1_wmma_gemm_q4_k_fp16_tiled_kernel<TM, TN, KC><<<grid, block, 0, stream>>>(
        A, W, D, M, N, K);
    return hipGetLastError();
}

// 64×64×32 — parity reference (matches hard-coded kernel bit-for-bit).
extern "C" hipError_t rocmforge_launch_wmma_gemm_q4_k_fp16_tiled_64x64x32(
    const float*   input,
    const uint8_t* weights,
    float*         output,
    int M, int N, int K,
    hipStream_t stream)
{
    return rf_v1_launch_tiled<64, 64, 32>(input, weights, output, M, N, K, stream);
}

// 128×128×32 — llama.cpp MMQ RDNA2+ equivalent, main perf target.
extern "C" hipError_t rocmforge_launch_wmma_gemm_q4_k_fp16_tiled_128x128x32(
    const float*   input,
    const uint8_t* weights,
    float*         output,
    int M, int N, int K,
    hipStream_t stream)
{
    return rf_v1_launch_tiled<128, 128, 32>(input, weights, output, M, N, K, stream);
}
