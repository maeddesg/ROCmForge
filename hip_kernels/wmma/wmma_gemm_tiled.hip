// Phase 2a Step 2 — Tiled 64×64 WMMA GEMM on gfx12 / RDNA 4.
//
// Computes D[M×N] = A[M×K] × B[K×N] in FP16 with FP32 accumulation.
// All three operands are row-major. Dimensions must currently be
// multiples of 64 (M, N) and 16 (K).
//
// Geometry
// --------
//   Output tile: 64×64 (= 4×4 WMMA blocks of 16×16)
//   K tile:      16 (one WMMA K-step)
//   Workgroup:   128 threads = 4 wavefronts × 32 lanes
//
//   Wave w ∈ {0,1,2,3} owns row strip [w*16 .. w*16+15] of the output
//   tile. Within its strip every wave computes four 16×16 D blocks that
//   cover all 64 columns (col_block 0..3). Each wave therefore holds
//   four float8 accumulators — one per column block.
//
// Register layout
// ---------------
//   A per-lane load pattern (from docs/wmma_register_layout_gfx12.md):
//       row   = wave*16 + (lane % 16)
//       k_set = (lane / 16) * 4                // 0 or 4
//       indices taken from row at k = {k_set + 0,1,2,3,   k_set + 8,9,10,11}
//
//   B per-lane load pattern (per column block b):
//       col   = b*16 + (lane % 16)
//       k_set = (lane / 16) * 4
//       indices taken at k = same {k_set + 0,1,2,3,   k_set + 8,9,10,11}
//
//   D store pattern:
//       col       = lane % 16
//       row_start = (lane / 16) * 8
//       for v in 0..8: D[row_start + v][col] = acc_d[v]
//
// LDS
// ---
//   lds_a[64 × 16] + lds_b[16 × 64] = 4 KB total, single-buffered. Under
//   the 16 KB LDS budget, so ≥ 2 resident workgroups per WGP. Bank-
//   conflict tuning is intentionally deferred to Phase 2c — see the
//   Phase 2a prompt notes.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

using half8  = _Float16 __attribute__((ext_vector_type(8)));
using float8 = float    __attribute__((ext_vector_type(8)));

#define TILE_M 64
#define TILE_N 64
#define TILE_K 16
#define WARPS_PER_BLOCK 4
#define THREADS_PER_BLOCK (WARPS_PER_BLOCK * 32)
#define COL_BLOCKS 4   // 64 / 16
// 1024 halfs per tile / 128 threads per block
#define HALFS_PER_THREAD 8

__global__ void wmma_gemm_tiled_kernel(
    const __half* __restrict__ A,
    const __half* __restrict__ B,
    float*        __restrict__ D,
    int M, int N, int K)
{
    const int tile_m = blockIdx.y * TILE_M;
    const int tile_n = blockIdx.x * TILE_N;

    const int tid     = threadIdx.x;       // 0..127
    const int wave_id = tid >> 5;          // 0..3
    const int lane    = tid & 31;          // 0..31

    __shared__ __half lds_a[TILE_M * TILE_K];   // 2 KB
    __shared__ __half lds_b[TILE_K * TILE_N];   // 2 KB

    float8 acc[COL_BLOCKS];
    #pragma unroll
    for (int b = 0; b < COL_BLOCKS; ++b) {
        acc[b] = {0, 0, 0, 0, 0, 0, 0, 0};
    }

    const int num_k_tiles = K / TILE_K;

    const int a_row_in_tile = wave_id * 16 + (lane & 15);
    const int a_k_set       = (lane >> 4) * 4;
    const int col_in_lane   = lane & 15;
    const int b_k_set       = (lane >> 4) * 4;

    for (int kt = 0; kt < num_k_tiles; ++kt) {
        const int k_start = kt * TILE_K;

        // ---- Cooperative global → LDS load of A tile [64 × 16] ----
        // 1024 halfs / 128 threads = 8 halfs per thread.
        // Thread `tid` writes lds_a[tid*8 .. tid*8+7]; two threads
        // cover each tile row of 16 halfs → naturally contiguous in
        // global memory within each row.
        #pragma unroll
        for (int i = 0; i < HALFS_PER_THREAD; ++i) {
            const int flat = tid * HALFS_PER_THREAD + i;
            const int row_in_tile = flat / TILE_K;      // 0..63
            const int k_in_tile   = flat & (TILE_K - 1);
            lds_a[flat] = A[(tile_m + row_in_tile) * K + k_start + k_in_tile];
        }

        // ---- Cooperative global → LDS load of B tile [16 × 64] ----
        #pragma unroll
        for (int i = 0; i < HALFS_PER_THREAD; ++i) {
            const int flat = tid * HALFS_PER_THREAD + i;
            const int k_in_tile   = flat / TILE_N;      // 0..15
            const int col_in_tile = flat & (TILE_N - 1);
            lds_b[flat] = B[(k_start + k_in_tile) * N + tile_n + col_in_tile];
        }

        __syncthreads();

        // ---- LDS → registers: one A block for this wave's row strip ----
        half8 a_reg;
        {
            const int base = a_row_in_tile * TILE_K;
            a_reg[0] = lds_a[base + a_k_set + 0];
            a_reg[1] = lds_a[base + a_k_set + 1];
            a_reg[2] = lds_a[base + a_k_set + 2];
            a_reg[3] = lds_a[base + a_k_set + 3];
            a_reg[4] = lds_a[base + a_k_set + 8];
            a_reg[5] = lds_a[base + a_k_set + 9];
            a_reg[6] = lds_a[base + a_k_set + 10];
            a_reg[7] = lds_a[base + a_k_set + 11];
        }

        // ---- LDS → registers + WMMA for each of 4 column blocks ----
        #pragma unroll
        for (int cb = 0; cb < COL_BLOCKS; ++cb) {
            const int b_col_in_tile = cb * 16 + col_in_lane;

            half8 b_reg;
            b_reg[0] = lds_b[(b_k_set + 0)  * TILE_N + b_col_in_tile];
            b_reg[1] = lds_b[(b_k_set + 1)  * TILE_N + b_col_in_tile];
            b_reg[2] = lds_b[(b_k_set + 2)  * TILE_N + b_col_in_tile];
            b_reg[3] = lds_b[(b_k_set + 3)  * TILE_N + b_col_in_tile];
            b_reg[4] = lds_b[(b_k_set + 8)  * TILE_N + b_col_in_tile];
            b_reg[5] = lds_b[(b_k_set + 9)  * TILE_N + b_col_in_tile];
            b_reg[6] = lds_b[(b_k_set + 10) * TILE_N + b_col_in_tile];
            b_reg[7] = lds_b[(b_k_set + 11) * TILE_N + b_col_in_tile];

            acc[cb] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(
                a_reg, b_reg, acc[cb]);
        }

        __syncthreads();
    }

    // ---- Store accumulators per the D layout ----
    // Lane L holds column L%16 at rows (L/16)*8 .. (L/16)*8 + 7.
    // This wave owns output rows [tile_m + wave_id*16 ..  + 15].
    const int out_col_in_lane   = lane & 15;
    const int out_row_start     = (lane >> 4) * 8;
    const int wave_row_in_tile  = wave_id * 16;

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

extern "C" hipError_t wmma_gemm_tiled_launch(
    const void* A,
    const void* B,
    float*      D,
    int M, int N, int K,
    hipStream_t stream)
{
    if (M <= 0 || N <= 0 || K <= 0) return hipErrorInvalidValue;
    if ((M % TILE_M) != 0) return hipErrorInvalidValue;
    if ((N % TILE_N) != 0) return hipErrorInvalidValue;
    if ((K % TILE_K) != 0) return hipErrorInvalidValue;

    dim3 grid(N / TILE_N, M / TILE_M, 1);
    dim3 block(THREADS_PER_BLOCK, 1, 1);

    hipLaunchKernelGGL(
        wmma_gemm_tiled_kernel,
        grid, block,
        0, stream,
        reinterpret_cast<const __half*>(A),
        reinterpret_cast<const __half*>(B),
        D,
        M, N, K
    );
    return hipGetLastError();
}
