// MMQ Q6_K kernel — P0.2 Schritt 6.
//
// Direct sibling of mmq_q4_k.hip. Same Integer-WMMA infra, same
// fragment layout, same outer K-loop, same write-back. Only the
// per-sub-block dequant (load_tiles) and the scale-fixup change:
//
//   * Q6_K has 16 sub-blocks of 16 K-elements each (not 8 sub-blocks
//     of 32 K-elements like Q4_K). One 16×16×16 WMMA call covers a
//     full sub-block, so 16 WMMA calls per Q4_K-aligned super-block —
//     same total work as Q4_K's 8 × 2 calls.
//   * Q6_K weights store low-nibble ql[128] + high-2bit qh[64] +
//     int8 scales[16] + half d (210 bytes total). The 6-bit value
//     (0..63) is reconstructed as `(ql_nib | (qh_bits << 4)) - 32`,
//     yielding a SIGNED int8 in -32..31. The integer WMMA's signed
//     `iu8` mode (both flags `true`) consumes this directly.
//   * Q6_K has NO dmin/min term — the scale-fixup is just
//     `d_A * scales[sb] * d_B * raw`. No `sum_B[sb]` array is
//     needed. This shrinks the per-super-block VGPR footprint vs
//     Q4_K (which carried sum_B_arr[8] = 8 VGPRs in addition to
//     d_B_arr[8]).
//
// Layout-source: hip_kernels_v1/wmma/wmma_gemm_q6_k_fp16.hip:76-125
// (same byte-mapping; the FP16 kernel was bit-validated against
// llama.cpp's CUDA path) and hip_kernels_v1/gemv/gemv_q6_k_mmvq.hip
// (same dequant for the GEMV decode path).

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>

using int32x2_t = __attribute__((__vector_size__(2 * sizeof(int)))) int;
using int32x8_t = __attribute__((__vector_size__(8 * sizeof(int)))) int;

struct __attribute__((packed)) block_q6_K_raw_s {
    uint8_t ql[128];
    uint8_t qh[64];
    int8_t  scales[16];
    __half  d;
};
static_assert(sizeof(block_q6_K_raw_s) == 210, "block_q6_K must be 210 bytes");

// Activation block (DS4 layout — same as block_q8_1_mmq from Q4_K).
struct __attribute__((packed)) block_q8_1_mmq_ds4_q6_s {
    __half2 ds4[4];
    int8_t  qs[128];
};
static_assert(sizeof(block_q8_1_mmq_ds4_q6_s) == 144, "block_q8_1_mmq must be 144 bytes");

__device__ __forceinline__ int rf_q6_pack4(int8_t n0, int8_t n1, int8_t n2, int8_t n3) {
    return ((int)(uint8_t)n0)       | (((int)(uint8_t)n1) << 8)
         | (((int)(uint8_t)n2) << 16) | (((int)(uint8_t)n3) << 24);
}

template<int WARPS_PER_BLOCK>
__global__ void rf_v1_mmq_q6_k_kernel(
    const block_q6_K_raw_s*         __restrict__ A,
    const block_q8_1_mmq_ds4_q6_s*  __restrict__ B,
    float*                          __restrict__ C,
    int M, int N, int K)
{
    constexpr int TILE_M = WARPS_PER_BLOCK * 16;

    const int tile_n  = blockIdx.x;
    const int tile_m  = blockIdx.y;
    const int warp_id = threadIdx.x >> 5;
    const int lane    = threadIdx.x & 31;
    const int row16   = lane & 15;
    const int half    = lane >> 4;

    const int n_row_base = tile_n * 16;
    const int m_col_base = tile_m * TILE_M + warp_id * 16;
    if (m_col_base >= M) return;

    const int n_super_per_row = K / 256;
    const int n_mmq_per_row   = K / 128;

    const int n_row = n_row_base + row16;
    const int m_col = m_col_base + row16;

    float c_acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};

    // ─── Outer K-loop over Q6_K super-blocks ────────────────────────
    for (int ks = 0; ks < n_super_per_row; ++ks) {
        const block_q6_K_raw_s*           a_sb  = &A[n_row * n_super_per_row + ks];
        const block_q8_1_mmq_ds4_q6_s*    b_lo  = &B[m_col * n_mmq_per_row + 2 * ks + 0];
        const block_q8_1_mmq_ds4_q6_s*    b_hi  = &B[m_col * n_mmq_per_row + 2 * ks + 1];

        // 16 sub-blocks × 1 WMMA-call each. acc_sb[16] stays live until
        // the scale-fixup loop below — same pattern as Q4_K but twice
        // the count of accumulators (16 vs 8). Per-block VGPR cost is
        // higher; trade-off accepted because the alternative
        // (sub-block-sequential, "Strategy C") was a measured net loss
        // on Q4_K (Schritt 5b).
        int32x8_t acc_sb[16];
        #pragma unroll
        for (int sb = 0; sb < 16; ++sb) acc_sb[sb] = int32x8_t{0,0,0,0,0,0,0,0};

        // ─── Inner WMMA loop ───────────────────────────────────────
        #pragma unroll
        for (int sb = 0; sb < 16; ++sb) {
            // Q6_K layout — derived from wmma_gemm_q6_k_fp16.hip:93-112.
            //   sub-block sb covers 16 K-elements at element-indices
            //   sb*16 .. sb*16+15. Within the GGUF block:
            //     n_half       = sb >> 3       (low half 0..127 vs high 128..255)
            //     which_q      = (sb >> 1) & 3 (which 32-elem quartet, 0..3)
            //     is_high_nib  = which_q >= 2  (low or high nibble of ql)
            //     qh_shift     = 2 * which_q   (bit-position in qh byte)
            //     l_base_start = (sb & 1) * 16 (offset in 32-byte chunk)
            const int  n_half       = sb >> 3;
            const int  which_q      = (sb >> 1) & 3;
            const bool is_high_nib  = which_q >= 2;
            const int  qh_shift     = 2 * which_q;
            const int  l_base_start = (sb & 1) * 16;

            const uint8_t* ql_block = a_sb->ql + n_half * 64 + (which_q & 1) * 32;
            const uint8_t* qh_block = a_sb->qh + n_half * 32;

            // Activation mapping: Q6_K sub-block sb (16 elements) maps
            // to half of an activation sub-block (32 elements):
            //   sb's mmq block        = (sb < 8) ? b_lo : b_hi
            //   sb's act-sub-block    = (sb >> 1) & 3      (0..3)
            //   K-offset within act-sub-block = (sb & 1) * 16   (0 or 16)
            const block_q8_1_mmq_ds4_q6_s* b_block =
                (sb < 8) ? b_lo : b_hi;
            const int b_sb_local           = (sb >> 1) & 3;
            const int k_offset_in_act_sb   = (sb & 1) * 16;

            // 1 WMMA call (K=16). Lane l holds K-positions 8*half..8*half+7
            // within the 16-element sub-block.
            int32x2_t a_frag, b_frag;
            int8_t va[8], vb[8];
            #pragma unroll
            for (int w = 0; w < 8; ++w) {
                const int i_in_sb = 8 * half + w;   // 0..15 within Q6_K sb
                {
                    const uint8_t ql_byte = ql_block[l_base_start + i_in_sb];
                    const uint8_t qh_byte = qh_block[l_base_start + i_in_sb];
                    const int ql_nib  = is_high_nib ? (ql_byte >> 4) : (ql_byte & 0x0F);
                    const int qh_bits = (qh_byte >> qh_shift) & 0x3;
                    const int q6      = ql_nib | (qh_bits << 4);
                    va[w] = (int8_t)(q6 - 32);   // signed -32..31
                }
                {
                    const int k_in_act_sb = k_offset_in_act_sb + 8 * half + w;
                    vb[w] = b_block->qs[32 * b_sb_local + k_in_act_sb];
                }
            }
            a_frag[0] = rf_q6_pack4(va[0], va[1], va[2], va[3]);
            a_frag[1] = rf_q6_pack4(va[4], va[5], va[6], va[7]);
            b_frag[0] = (int)(uint8_t)vb[0]       | ((int)(uint8_t)vb[1] << 8)
                      | ((int)(uint8_t)vb[2] << 16) | ((int)(uint8_t)vb[3] << 24);
            b_frag[1] = (int)(uint8_t)vb[4]       | ((int)(uint8_t)vb[5] << 8)
                      | ((int)(uint8_t)vb[6] << 16) | ((int)(uint8_t)vb[7] << 24);

            acc_sb[sb] = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(
                true, a_frag, true, b_frag, acc_sb[sb], true);
        }

        // ─── Scale-fixup (no min term!) ────────────────────────────
        // d_B[sb] is shared between Q6_K sub-blocks 0+1, 2+3, … (each
        // activation sub-block covers 32 K-elements = two Q6_K sb's).
        // Read once per Q4_K-style act-sb (4 per mmq block × 2 mmq
        // blocks = 8 total). Sum-B is unused for Q6_K so .y is dropped.
        float d_B_act[8];   // 8 act-sub-blocks per Q6_K super-block
        #pragma unroll
        for (int act_sb = 0; act_sb < 8; ++act_sb) {
            const block_q8_1_mmq_ds4_q6_s* b_block =
                (act_sb < 4) ? b_lo : b_hi;
            const int local = act_sb & 3;
            d_B_act[act_sb] = __half2float(b_block->ds4[local].x);
        }

        const int n_row_start = (lane >> 4) * 8;
        #pragma unroll
        for (int slot = 0; slot < 8; ++slot) {
            const int n = n_row_base + n_row_start + slot;
            if (n >= N) continue;
            const block_q6_K_raw_s* a_sb_slot =
                &A[n * n_super_per_row + ks];
            const float d_A = __half2float(a_sb_slot->d);

            float add = 0.0f;
            #pragma unroll
            for (int sb = 0; sb < 16; ++sb) {
                const int raw = acc_sb[sb][slot];
                const int act_sb = sb >> 1;          // 0..7 (each act-sb covers 2 q6 sbs)
                const int8_t sc  = a_sb_slot->scales[sb];   // signed
                const float d_scale = d_A * (float)sc;
                add += d_scale * d_B_act[act_sb] * (float)raw;
            }
            c_acc[slot] += add;
        }
    }

    // ─── Write-back ─────────────────────────────────────────────────
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
static inline hipError_t rf_v1_launch_mmq_q6_k(
    const void*  weights_q6_k,
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
    rf_v1_mmq_q6_k_kernel<WARPS_PER_BLOCK><<<grid, block, 0, stream>>>(
        reinterpret_cast<const block_q6_K_raw_s*>(weights_q6_k),
        reinterpret_cast<const block_q8_1_mmq_ds4_q6_s*>(activations_q8_1_mmq),
        output, M, N, K);
    return hipGetLastError();
}

extern "C" hipError_t rocmforge_launch_mmq_q6_k(
    const void*  weights_q6_k,
    const void*  activations_q8_1_mmq,
    float*       output,
    int M, int N, int K,
    hipStream_t  stream)
{
    return rf_v1_launch_mmq_q6_k<4>(
        weights_q6_k, activations_q8_1_mmq, output, M, N, K, stream);
}

extern "C" hipError_t rocmforge_launch_mmq_q6_k_1w(
    const void*  weights_q6_k,
    const void*  activations_q8_1_mmq,
    float*       output,
    int M, int N, int K,
    hipStream_t  stream)
{
    return rf_v1_launch_mmq_q6_k<1>(
        weights_q6_k, activations_q8_1_mmq, output, M, N, K, stream);
}
