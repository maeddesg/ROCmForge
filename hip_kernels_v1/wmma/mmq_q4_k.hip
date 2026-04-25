// MMQ Q4_K kernel — P0.2 Schritt 5 (Multi-Warp scale-up).
//
// Templated on WARPS_PER_BLOCK ∈ {1, 4}:
//   * WARPS=1 (rocmforge_launch_mmq_q4_k_1w): identical math+layout to the
//     Schritt-3 single-warp kernel. Kept as parity reference for the
//     multi-warp scale-up tests (and as fallback should 4W regress).
//   * WARPS=4 (rocmforge_launch_mmq_q4_k):    production 4-warp variant.
//     Each block produces a 64-row × 16-col output tile; each of the 4
//     warps owns 16 contiguous M-rows of that tile. Reduces grid blocks
//     by 4× vs 1W (e.g. Qwen3-QKV: 9216 → 2304 blocks for M=576,N=4096),
//     and amortises kernel-launch / dispatch overhead.
//
// Math is byte-identical between the two variants — the only differences
// are (a) blockDim.x scales with warps, (b) m_base derives an extra
// warp_id*16 offset, (c) grid_y shrinks accordingly. The Schritt-3 fixes
// (block_q8_1_mmq 4-subblock handling, K-offset 16*half+8*call+w,
// pair-based Q4_K nibble extraction) remain in place unchanged.
//
// LDS-Sharing of weights across the 4 warps is intentionally NOT done in
// this revision. Each warp reads its own activations (different M-rows),
// and reads the same weights from L1/L2 — measuring shows the kernel is
// integer-MMA-bound, not L2-bound, so weight-LDS-sharing would add
// __syncthreads cost without much win. Reserved for a follow-up.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>

using int32x2_t = __attribute__((__vector_size__(2 * sizeof(int)))) int;
using int32x8_t = __attribute__((__vector_size__(8 * sizeof(int)))) int;

struct __attribute__((packed)) block_q4_K_raw_s {
    __half  d;
    __half  dmin;
    uint8_t scales[12];
    uint8_t qs[128];
};
static_assert(sizeof(block_q4_K_raw_s) == 144, "block_q4_K must be 144 bytes");

struct __attribute__((packed)) block_q8_1_mmq_ds4_s {
    __half2 ds4[4];
    int8_t  qs[128];
};
static_assert(sizeof(block_q8_1_mmq_ds4_s) == 144, "block_q8_1_mmq must be 144 bytes");

__device__ __forceinline__ int unpack_scales_q45_K_scaled(const int * scales, const int ksc) {
    return ((scales[(ksc%2) + (ksc!=0)] >> (4 * (ksc & (ksc/2)))) & 0x0F0F0F0F) |
           ((scales[ksc/2]              >> (2 * (ksc % 2)))       & 0x30303030);
}

__device__ __forceinline__ int pack_nibbles_int(int8_t n0, int8_t n1, int8_t n2, int8_t n3) {
    return ((int)(uint8_t)n0)       | (((int)(uint8_t)n1) << 8)
         | (((int)(uint8_t)n2) << 16) | (((int)(uint8_t)n3) << 24);
}

// Each block produces a (WARPS_PER_BLOCK*16) × 16 output tile at
// (tile_n*16, tile_m*WARPS_PER_BLOCK*16). Each warp inside the block
// owns a 16×16 sub-tile, equivalent to one independent invocation of
// the original 1-warp kernel offset by warp_id*16 in M.
//
// Grid:  (N/16, ceil_div(M, WARPS_PER_BLOCK*16))
// Block: WARPS_PER_BLOCK * 32 threads
template<int WARPS_PER_BLOCK>
__global__ void rf_v1_mmq_q4_k_kernel(
    const block_q4_K_raw_s*      __restrict__ A,
    const block_q8_1_mmq_ds4_s*  __restrict__ B,
    float*                       __restrict__ C,
    int M, int N, int K)
{
    constexpr int TILE_M = WARPS_PER_BLOCK * 16;

    const int tile_n  = blockIdx.x;
    const int tile_m  = blockIdx.y;
    const int warp_id = threadIdx.x >> 5;          // 0..WARPS_PER_BLOCK-1
    const int lane    = threadIdx.x & 31;
    const int row16   = lane & 15;                 // 0..15
    const int half    = lane >> 4;                 // 0 or 1

    const int n_row_base = tile_n * 16;
    const int m_col_base = tile_m * TILE_M + warp_id * 16;

    // Outer warp-level boundary: completely-out-of-range warps return.
    // (WARP=4, M=33 example: warps 0+1 active on rows 0-31, warp 2 active
    //  on row 32 only via inner lane check, warp 3 returns here.)
    if (m_col_base >= M) return;

    const int n_super_per_row = K / 256;
    const int n_mmq_per_row   = K / 128;

    const int n_row = n_row_base + row16;
    const int m_col = m_col_base + row16;

    float c_acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};

    // ─── Outer K-loop over Q4_K super-blocks ────────────────────────
    for (int ks = 0; ks < n_super_per_row; ++ks) {
        const block_q4_K_raw_s*      a_sb  = &A[n_row * n_super_per_row + ks];
        const block_q8_1_mmq_ds4_s*  b_lo  = &B[m_col * n_mmq_per_row + 2 * ks + 0];
        const block_q8_1_mmq_ds4_s*  b_hi  = &B[m_col * n_mmq_per_row + 2 * ks + 1];

        int32x8_t acc_sb[8];
        #pragma unroll
        for (int sb = 0; sb < 8; ++sb) acc_sb[sb] = int32x8_t{0,0,0,0,0,0,0,0};

        // ─── Inner WMMA loop: 8 sub-blocks × 2 calls each ──────────
        #pragma unroll
        for (int sb = 0; sb < 8; ++sb) {
            const int  pair_base = (sb >> 1) * 32;
            const bool is_upper  = (sb & 1) != 0;

            int32x2_t a_frag[2];
            int32x2_t b_frag[2];
            #pragma unroll
            for (int call = 0; call < 2; ++call) {
                int8_t va[8], vb[8];
                #pragma unroll
                for (int w = 0; w < 8; ++w) {
                    const int k_in_sb = 16 * half + 8 * call + w;   // 0..31
                    {
                        const uint8_t byte = a_sb->qs[pair_base + k_in_sb];
                        const int nib = is_upper ? (byte >> 4) : (byte & 0x0F);
                        va[w] = (int8_t)nib;
                    }
                    {
                        const block_q8_1_mmq_ds4_s* b_block = (sb < 4) ? b_lo : b_hi;
                        const int b_sb_local = sb & 3;
                        vb[w] = b_block->qs[32 * b_sb_local + k_in_sb];
                    }
                }
                a_frag[call][0] = pack_nibbles_int(va[0], va[1], va[2], va[3]);
                a_frag[call][1] = pack_nibbles_int(va[4], va[5], va[6], va[7]);
                b_frag[call][0] = (int)(uint8_t)vb[0]       | ((int)(uint8_t)vb[1] << 8)
                                | ((int)(uint8_t)vb[2] << 16) | ((int)(uint8_t)vb[3] << 24);
                b_frag[call][1] = (int)(uint8_t)vb[4]       | ((int)(uint8_t)vb[5] << 8)
                                | ((int)(uint8_t)vb[6] << 16) | ((int)(uint8_t)vb[7] << 24);
            }

            acc_sb[sb] = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(
                true, a_frag[0], true, b_frag[0], acc_sb[sb], true);
            acc_sb[sb] = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(
                true, a_frag[1], true, b_frag[1], acc_sb[sb], true);
        }

        // ─── Scale-fixup ───────────────────────────────────────────
        float d_B_arr[8];
        float sum_B_arr[8];
        #pragma unroll
        for (int sb = 0; sb < 8; ++sb) {
            const block_q8_1_mmq_ds4_s* b_block = (sb < 4) ? b_lo : b_hi;
            const int b_sb_local = sb & 3;
            float2 ds = __half22float2(b_block->ds4[b_sb_local]);
            d_B_arr[sb]   = ds.x;
            sum_B_arr[sb] = ds.y;
        }

        const int n_row_start = (lane >> 4) * 8;
        #pragma unroll
        for (int slot = 0; slot < 8; ++slot) {
            const int n = n_row_base + n_row_start + slot;
            if (n >= N) continue;
            const block_q4_K_raw_s* a_sb_for_slot =
                &A[n * n_super_per_row + ks];
            const float  d_A    = __half2float(a_sb_for_slot->d);
            const float  dmin_A = __half2float(a_sb_for_slot->dmin);
            int sc_p[2], mn_p[2];
            const int* sc32 = reinterpret_cast<const int*>(a_sb_for_slot->scales);
            sc_p[0] = unpack_scales_q45_K_scaled(sc32, 0);
            sc_p[1] = unpack_scales_q45_K_scaled(sc32, 1);
            mn_p[0] = unpack_scales_q45_K_scaled(sc32, 2);
            mn_p[1] = unpack_scales_q45_K_scaled(sc32, 3);
            const uint8_t* sc8p = reinterpret_cast<const uint8_t*>(sc_p);
            const uint8_t*  m8p = reinterpret_cast<const uint8_t*>(mn_p);

            float add = 0.0f;
            #pragma unroll
            for (int sb = 0; sb < 8; ++sb) {
                const int raw = acc_sb[sb][slot];
                const float d_scale = d_A    * (float)sc8p[sb];
                const float d_min   = dmin_A * (float)m8p[sb];
                add += d_scale * d_B_arr[sb] * (float)raw
                     - d_min   * sum_B_arr[sb];
            }
            c_acc[slot] += add;
        }
    }

    // ─── Write-back ─────────────────────────────────────────────────
    // Output [M × N] row-major, m_out is this lane's M-col (= B-row).
    // Inner per-lane boundary handles partial warps (e.g. M=33, warp 2,
    // lane row16=0 writes row 32, lane row16>0 returns).
    const int n_row_start = (lane >> 4) * 8;
    const int m_out = m_col;
    if (m_out >= M) return;
    #pragma unroll
    for (int slot = 0; slot < 8; ++slot) {
        const int n_out = n_row_base + n_row_start + slot;
        if (n_out >= N) continue;
        C[m_out * N + n_out] = c_acc[slot];
    }
}

// ─── Launch helpers ───────────────────────────────────────────────────

template<int WARPS_PER_BLOCK>
static inline hipError_t rf_v1_launch_mmq_q4_k(
    const void*  weights_q4_k,
    const void*  activations_q8_1_mmq,
    float*       output,
    int M, int N, int K,
    hipStream_t  stream)
{
    if (M <= 0 || N <= 0 || K <= 0)  return hipErrorInvalidValue;
    if ((M % 16) != 0)               return hipErrorInvalidValue;
    if ((N % 16) != 0)               return hipErrorInvalidValue;
    if ((K % 256) != 0)              return hipErrorInvalidValue;

    constexpr int TILE_M  = WARPS_PER_BLOCK * 16;
    constexpr int THREADS = WARPS_PER_BLOCK * 32;

    dim3 grid(N / 16, (M + TILE_M - 1) / TILE_M, 1);
    dim3 block(THREADS, 1, 1);
    rf_v1_mmq_q4_k_kernel<WARPS_PER_BLOCK><<<grid, block, 0, stream>>>(
        reinterpret_cast<const block_q4_K_raw_s*>(weights_q4_k),
        reinterpret_cast<const block_q8_1_mmq_ds4_s*>(activations_q8_1_mmq),
        output, M, N, K);
    return hipGetLastError();
}

// Production: 4 warps per block (TILE_M=64).
extern "C" hipError_t rocmforge_launch_mmq_q4_k(
    const void*  weights_q4_k,
    const void*  activations_q8_1_mmq,
    float*       output,
    int M, int N, int K,
    hipStream_t  stream)
{
    return rf_v1_launch_mmq_q4_k<4>(
        weights_q4_k, activations_q8_1_mmq, output, M, N, K, stream);
}

// Parity reference: 1 warp per block (TILE_M=16). Same kernel body —
// only the launch-grid and block-size differ. Used by the multi-warp
// parity test to prove the scale-up didn't change the math.
extern "C" hipError_t rocmforge_launch_mmq_q4_k_1w(
    const void*  weights_q4_k,
    const void*  activations_q8_1_mmq,
    float*       output,
    int M, int N, int K,
    hipStream_t  stream)
{
    return rf_v1_launch_mmq_q4_k<1>(
        weights_q4_k, activations_q8_1_mmq, output, M, N, K, stream);
}
