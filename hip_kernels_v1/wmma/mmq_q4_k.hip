// MMQ Q4_K scaled kernel — P0.2 Schritt 3.
//
// Scale-up of mmq_q4_k_minimal.hip to variable (M, N, K). The per-tile
// math is IDENTICAL to the minimal kernel — all three bugs from Schritt 2
// are baked in (block_q8_1_mmq 4-subblock handling, K-offset
// 16*half+8*call+w, pair-based Q4_K nibble extraction). This version adds:
//
//   1. Grid parallelism: one block per 16×16 output tile.
//      grid = (N/16, M/16), block = 32 threads (one warp).
//   2. Variable K via outer loop over Q4_K super-blocks (256 K each).
//      Per-super-block scale-fixup accumulates into a per-slot float.
//   3. Requires M % 16 == 0 and N % 16 == 0 and K % 256 == 0. The
//      dispatcher pads M up to a multiple of 16 when needed.
//
// Activation buffer layout (same as quantize_q8_1_mmq output):
//   N_mmq_blocks_per_row = K / 128 (=32 for K=4096). Block at row m, offset
//   k_block_idx is at B[m * n_mmq_per_row + k_block_idx].

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

// Each block handles one 16×16 output tile at (tile_n, tile_m).
// Grid: (N/16, M/16). Block: 32 threads (one warp).
__global__ void rf_v1_mmq_q4_k_kernel(
    const block_q4_K_raw_s*      __restrict__ A,     // [N_total rows × K/256 super-blocks]
    const block_q8_1_mmq_ds4_s*  __restrict__ B,     // [M_total rows × K/128 mmq blocks]
    float*                       __restrict__ C,     // [M × N] row-major (M is "row", N is "col")
    int M, int N, int K)
{
    const int tile_n = blockIdx.x;   // 0..N/16-1
    const int tile_m = blockIdx.y;   // 0..M/16-1
    const int lane   = threadIdx.x & 31;
    const int row16  = lane & 15;    // 0..15
    const int half   = lane >> 4;    // 0 or 1

    const int n_row_base = tile_n * 16;   // THIS block's 16 A-rows
    const int m_col_base = tile_m * 16;   // THIS block's 16 B-rows

    const int n_super_per_row = K / 256;           // Q4_K super-blocks per weight row
    const int n_mmq_per_row   = K / 128;           // block_q8_1_mmq blocks per activation row

    // Base indices for the A-row and M-col this lane owns.
    const int n_row = n_row_base + row16;          // weight row (N-dim)
    const int m_col = m_col_base + row16;          // activation row (M-dim)

    // Accumulator: per-slot float (slot = 8 N-rows owned by this lane within
    // the 16-N-tile, per the smoke-test / minimal-kernel convention).
    float c_acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};

    // ─── Outer K-loop over super-blocks ─────────────────────────────
    for (int ks = 0; ks < n_super_per_row; ++ks) {
        const block_q4_K_raw_s*      a_sb  = &A[n_row * n_super_per_row + ks];
        const block_q8_1_mmq_ds4_s*  b_lo  = &B[m_col * n_mmq_per_row + 2 * ks + 0];
        const block_q8_1_mmq_ds4_s*  b_hi  = &B[m_col * n_mmq_per_row + 2 * ks + 1];

        // Per-super-block int32 accumulators, one per sub-block.
        int32x8_t acc_sb[8];
        #pragma unroll
        for (int sb = 0; sb < 8; ++sb) acc_sb[sb] = int32x8_t{0,0,0,0,0,0,0,0};

        // ─── Inner WMMA loop: 8 sub-blocks × 2 calls each ──────────
        #pragma unroll
        for (int sb = 0; sb < 8; ++sb) {
            // Q4_K pair layout (same as wmma_gemm_q4_k_fp16.hip:107-117):
            //   pair 0 = sub-blocks (0,1) = bytes 0..31
            //   pair 1 = sub-blocks (2,3) = bytes 32..63
            //   pair 2 = sub-blocks (4,5) = bytes 64..95
            //   pair 3 = sub-blocks (6,7) = bytes 96..127
            // Even sb → low nibbles, odd sb → high nibbles.
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
                    // Weights: one pair of bytes → two nibbles (even/odd sb).
                    {
                        const uint8_t byte = a_sb->qs[pair_base + k_in_sb];
                        const int nib = is_upper ? (byte >> 4) : (byte & 0x0F);
                        va[w] = (int8_t)nib;
                    }
                    // Activations: sb 0..3 → b_lo, sb 4..7 → b_hi. Within the
                    // chosen mmq block the sub-block index is sb%4, K offset
                    // within the 128-elem block is 32*(sb%4) + k_in_sb.
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

        // ─── Scale-fixup for THIS super-block, across 8 slots ─────
        // Each lane owns 8 N-rows; scales are per (N-row, sub-block).
        // A-scales come from per-N-row Q4_K super-block header.
        // B-scales come from the two mmq blocks' ds4[] pairs.
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

        // Each slot corresponds to a different N-row owned by this lane.
        const int n_row_start = (lane >> 4) * 8;    // 0 or 8
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
    // Output layout: [M rows × N cols] row-major (matches the existing
    // FP16 WMMA kernel — `C[m * N + n]`, M-index outer). This is the
    // convention the prefill executor expects, so the MMQ kernel can
    // drop-in-replace `rocmforge_launch_wmma_gemm_q4_k_fp16`.
    const int n_row_start = (lane >> 4) * 8;
    const int m_out = m_col;   // this lane's M-col = B-row
    if (m_out >= M) return;
    #pragma unroll
    for (int slot = 0; slot < 8; ++slot) {
        const int n_out = n_row_base + n_row_start + slot;
        if (n_out >= N) continue;
        C[m_out * N + n_out] = c_acc[slot];
    }
}

extern "C" hipError_t rocmforge_launch_mmq_q4_k(
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

    dim3 grid(N / 16, M / 16, 1);
    dim3 block(32, 1, 1);
    rf_v1_mmq_q4_k_kernel<<<grid, block, 0, stream>>>(
        reinterpret_cast<const block_q4_K_raw_s*>(weights_q4_k),
        reinterpret_cast<const block_q8_1_mmq_ds4_s*>(activations_q8_1_mmq),
        output, M, N, K);
    return hipGetLastError();
}
