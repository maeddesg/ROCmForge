// MMQ Q4_K minimal single-tile kernel — P0.2 Schritt 2 proof-of-concept.
//
// Hard-coded simplification: M = N = 16, K = 256 (one Q4_K super-block per
// row of A), one warp per grid block, one grid block per call. This is the
// narrowest slice of llama.cpp's mul_mat_q<Q4_K> that exercises every
// architectural piece: Q4_K nibble unpack, integer WMMA, scale/min fixup.
// If it passes bit-exact parity vs CPU, scaling to full MMQ tiling is
// mechanical.
//
// Math (per Q4_K sub-block of 32 K-elements, 8 sub-blocks per row):
//     raw_q_A[k]  = (qs_byte >> (k%2 ? 4 : 0)) & 0x0F         // 0..15
//     sub_dot[n,m,sb] = Σ_{k in sb}  raw_q_A[n,k] * int8_B[m,k]   (int32 MMA)
//     contrib[n,m,sb] = d_A * sc_sub[n,sb] * d_B[m,sb] * sub_dot[n,m,sb]
//                    - d_A * dmin_A * m_sub[n,sb] * sum_B[m,sb]
//     C[n,m] = Σ_sb contrib[n,m,sb]
//
// Ported from ~/tmp/llama.cpp/ggml/src/ggml-cuda/mmq.cuh:
//   - load_tiles_q4_K (2151-2240) — nibble split + scale unpack
//   - unpack_scales_q45_K (2141-2149)
//   - vec_dot_q8_1_q8_1_mma (1271-1323) — integer WMMA + fixup math
//   - mma_i32_16x16x16_iu8 (mma.cuh:1236-1263) — RDNA4 intrinsic
//
// Fragment layout (verified bit-exact in wmma_i32_smoke.hip):
//   A row-major, lane l holds rows (l&15) cols (l>>4)*8..+7
//   B col-major, lane l holds cols (l&15) rows (l>>4)*8..+7
//   C lane l holds col (l&15), rows (l>>4)*8..+7

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>

using int32x2_t = __attribute__((__vector_size__(2 * sizeof(int)))) int;
using int32x8_t = __attribute__((__vector_size__(8 * sizeof(int)))) int;

// Q4_K block layout per ggml-common.h: d(half), dmin(half), scales[12],
// qs[128]. Total 144 bytes for 256 elements.
struct __attribute__((packed)) block_q4_K_raw {
    __half  d;
    __half  dmin;
    uint8_t scales[12];
    uint8_t qs[128];
};
static_assert(sizeof(block_q4_K_raw) == 144, "block_q4_K must be 144 bytes");

// Activation block (Q4_K DS4 layout — same as quantize_q8_1_mmq output).
struct __attribute__((packed)) block_q8_1_mmq_ds4 {
    __half2 ds4[4];       // 4 × (d, sum) per 32-elem sub-block
    int8_t  qs[128];
};
static_assert(sizeof(block_q8_1_mmq_ds4) == 144, "block_q8_1_mmq must be 144 bytes");

// llama.cpp's 6-bit scale unpacker (mmq.cuh:2141-2149). Returns 4 packed
// 6-bit values (either scales sc0..sc7 or mins m0..m7) as int32 of 4 bytes.
__device__ __forceinline__ int unpack_scales_q45_K(const int * scales, const int ksc) {
    return ((scales[(ksc%2) + (ksc!=0)] >> (4 * (ksc & (ksc/2)))) & 0x0F0F0F0F) |
           ((scales[ksc/2]              >> (2 * (ksc % 2)))       & 0x30303030);
}

// Pack 4 nibbles 0..15 into one int32 (little-endian, one nibble per byte).
__device__ __forceinline__ int pack_nibbles_to_int(int8_t n0, int8_t n1, int8_t n2, int8_t n3) {
    return ((int)(uint8_t)n0)       | (((int)(uint8_t)n1) << 8)
         | (((int)(uint8_t)n2) << 16) | (((int)(uint8_t)n3) << 24);
}

// One Q4_K super-block covers 256 K-elements (8 sub-blocks × 32 each).
// One block_q8_1_mmq covers 128 K-elements (4 sub-blocks × 32 each).
// → Each M-row of activations needs TWO block_q8_1_mmq blocks to match
//   one K=256 super-block:
//     Q4_K sb 0..3 → B[2*row].ds4[sb],   B[2*row].qs[sb*32..]
//     Q4_K sb 4..7 → B[2*row+1].ds4[sb-4], B[2*row+1].qs[(sb-4)*32..]

__global__ void rf_v1_mmq_q4_k_minimal_kernel(
    const block_q4_K_raw*      __restrict__ A,   // 16 weight rows, 1 super-block (K=256) each
    const block_q8_1_mmq_ds4*  __restrict__ B,   // 32 activation blocks = 2 per M-row × 16 rows
    float*                     __restrict__ C)   // 16 N-rows × 16 M-cols output, row-major
{
    const int lane      = threadIdx.x & 31;
    const int row16     = lane & 15;       // 0..15
    const int half      = lane >> 4;       // 0 or 1
    const int col_in_lane = lane & 15;     // for C write-out

    // Integer-MMA accumulators, one per sub-block. Each sub-block covers
    // 32 K of a row; two WMMA calls per sub-block (K=16+16). 8 sub-blocks
    // total in a 256-K super-block.
    int32x8_t acc_sb[8];
    #pragma unroll
    for (int sb = 0; sb < 8; ++sb) {
        acc_sb[sb] = int32x8_t{0,0,0,0,0,0,0,0};
    }

    // For each sub-block (covers 32 K), do 2 int-WMMA calls (K=16 each).
    #pragma unroll
    for (int sb = 0; sb < 8; ++sb) {
        // Each sub-block owns bytes [sb*16 .. sb*16+15] of the qs array
        // (16 bytes × 2 nibbles = 32 values). Each byte yields low/high
        // nibbles; K-position of the low nibble = 2*byte_idx, high = 2*byte_idx+1.
        //
        // For the WMMA, lane l holds 8 int8 values (= 2 int32). The fragment
        // layout (verified in wmma_i32_smoke.hip): lane l holds row=(l&15)
        // K-positions (l>>4)*8 + 0..7. For two back-to-back K=16 calls:
        //   Call 0: K-positions 0..15  (lane 0-15 → K 0-7, lane 16-31 → K 8-15)
        //   Call 1: K-positions 16..31 (lane 0-15 → K 16-23, lane 16-31 → K 24-31)
        //
        // For A (Q4_K): row r is A[r], the Q4_K super-block. Bytes 16*sb..16*sb+15
        // of qs hold this sub-block. The K-element at position k within the
        // sub-block is the (k>>1)-th byte, low/high nibble for k even/odd.
        const block_q4_K_raw* a_row = &A[row16];
        int32x2_t a_frag_call[2];
        #pragma unroll
        for (int call = 0; call < 2; ++call) {
            int8_t v[8];
            #pragma unroll
            for (int w = 0; w < 8; ++w) {
                // Per mma.cuh tile<16,8,int>::get_j on RDNA4:
                //   j = 4*(tx/16) + l (in int32 units; l = 0..3 per call)
                // The two WMMA calls split x[] into halves: call 0 uses
                // x[0], x[1] (l=0,1 → int32 cols 4*half, 4*half+1), call
                // 1 uses x[2], x[3] (l=2,3 → int32 cols 4*half+2, 4*half+3).
                // So in int8 K-positions: base = 16*half + 8*call.
                const int k_in_sb = 16 * half + 8 * call + w;      // 0..31
                const int byte_idx = 16 * sb + (k_in_sb >> 1);
                const uint8_t byte = a_row->qs[byte_idx];
                const int nib = (k_in_sb & 1) ? (byte >> 4) : (byte & 0x0F);
                v[w] = (int8_t)nib;
            }
            a_frag_call[call][0] = pack_nibbles_to_int(v[0], v[1], v[2], v[3]);
            a_frag_call[call][1] = pack_nibbles_to_int(v[4], v[5], v[6], v[7]);
        }

        // For B (Q8_1_mmq): B is activations; our intrinsic wants B
        // col-major — lane l holds col (l&15) rows (l>>4)*8..+7. Here
        // "col" is M and "row" is K. So lane l loads B[M=row16,
        // K=call*16 + half*8 + 0..7].
        //
        // CRUCIAL: one block_q8_1_mmq has only 128 elements (4 sub-blocks).
        // For Q4_K sub-blocks sb=0..3 use the first of row row16's two mmq
        // blocks; for sb=4..7 use the second. Within the chosen mmq block
        // the sub-block index is sb%4 and the qs offset is 32*(sb%4).
        const block_q8_1_mmq_ds4* b_row = &B[2 * row16 + (sb >> 2)];
        const int b_sb_local = sb & 3;
        int32x2_t b_frag_call[2];
        #pragma unroll
        for (int call = 0; call < 2; ++call) {
            int8_t v[8];
            #pragma unroll
            for (int w = 0; w < 8; ++w) {
                // Per mma.cuh tile<16,8,int>::get_j on RDNA4:
                //   j = 4*(tx/16) + l (in int32 units; l = 0..3 per call)
                // The two WMMA calls split x[] into halves: call 0 uses
                // x[0], x[1] (l=0,1 → int32 cols 4*half, 4*half+1), call
                // 1 uses x[2], x[3] (l=2,3 → int32 cols 4*half+2, 4*half+3).
                // So in int8 K-positions: base = 16*half + 8*call.
                const int k_in_sb = 16 * half + 8 * call + w;      // 0..31
                const int byte_idx = 32 * b_sb_local + k_in_sb;   // within the mmq block
                v[w] = b_row->qs[byte_idx];
            }
            b_frag_call[call][0] = (int)(uint8_t)v[0]       | ((int)(uint8_t)v[1] << 8)
                                 | ((int)(uint8_t)v[2] << 16) | ((int)(uint8_t)v[3] << 24);
            b_frag_call[call][1] = (int)(uint8_t)v[4]       | ((int)(uint8_t)v[5] << 8)
                                 | ((int)(uint8_t)v[6] << 16) | ((int)(uint8_t)v[7] << 24);
        }

        // Two int-WMMA calls, accumulating into acc_sb[sb].
        acc_sb[sb] = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(
            true, a_frag_call[0], true, b_frag_call[0], acc_sb[sb], true);
        acc_sb[sb] = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(
            true, a_frag_call[1], true, b_frag_call[1], acc_sb[sb], true);
    }

    // C layout per fragment: lane l holds col=(l&15), rows=(l>>4)*8..+7.
    // "Col" maps to B-row (= M dimension), "row" to A-row (= N dimension).
    const int m_col       = col_in_lane;          // 0..15, this lane's M
    const int n_row_start = (lane >> 4) * 8;      // 0 or 8, lane's N range

    // Pull per-row scales/mins once (each lane owns a distinct N=n_row_start..+7).
    // scales array (12 bytes) → int[3]; unpack_scales_q45_K builds the 8
    // per-sub-block scales and 8 mins.
    int     sc_packed[2]; // sc_packed[0] = sc0..sc3, [1] = sc4..sc7
    int     mn_packed[2]; // mn_packed[0] = m0..m3,   [1] = m4..m7
    float   d_A_arr[8];
    float   dmin_A_arr[8];
    uint8_t sc_all[8][8]; // [n-slot][sub-block]
    uint8_t mn_all[8][8];
    #pragma unroll
    for (int slot = 0; slot < 8; ++slot) {
        const int n = n_row_start + slot;
        d_A_arr[slot]    = __half2float(A[n].d);
        dmin_A_arr[slot] = __half2float(A[n].dmin);
        const int* sc32  = reinterpret_cast<const int*>(A[n].scales);
        sc_packed[0] = unpack_scales_q45_K(sc32, 0);
        sc_packed[1] = unpack_scales_q45_K(sc32, 1);
        mn_packed[0] = unpack_scales_q45_K(sc32, 2);
        mn_packed[1] = unpack_scales_q45_K(sc32, 3);
        const uint8_t* sc8p = reinterpret_cast<const uint8_t*>(sc_packed);
        const uint8_t*  m8p = reinterpret_cast<const uint8_t*>(mn_packed);
        #pragma unroll
        for (int sb = 0; sb < 8; ++sb) {
            sc_all[slot][sb] = sc8p[sb];
            mn_all[slot][sb] =  m8p[sb];
        }
    }

    // B's ds4 = (d_B, sum_B) per sub-block, for this lane's m_col row.
    // Same split as above: sb=0..3 → B[2*m_col].ds4[sb],
    //                      sb=4..7 → B[2*m_col+1].ds4[sb-4].
    float d_B_arr[8];
    float sum_B_arr[8];
    #pragma unroll
    for (int sb = 0; sb < 8; ++sb) {
        const int b_idx       = 2 * m_col + (sb >> 2);
        const int b_sb_local  = sb & 3;
        float2 ds = __half22float2(B[b_idx].ds4[b_sb_local]);
        d_B_arr[sb]   = ds.x;
        sum_B_arr[sb] = ds.y;
    }

    // Combine the 8 sub-block int32 accumulators with their scales.
    // Each lane writes 8 output cells (C[n_row_start..+7, m_col]).
    #pragma unroll
    for (int slot = 0; slot < 8; ++slot) {
        float out = 0.0f;
        #pragma unroll
        for (int sb = 0; sb < 8; ++sb) {
            const int raw = acc_sb[sb][slot];
            const float d_scale = d_A_arr[slot]    * (float)sc_all[slot][sb];
            const float d_min   = dmin_A_arr[slot] * (float)mn_all[slot][sb];
            out += d_scale * d_B_arr[sb] * (float)raw
                 - d_min   * sum_B_arr[sb];
        }
        const int n = n_row_start + slot;
        C[n * 16 + m_col] = out;
    }
}

extern "C" hipError_t rocmforge_launch_mmq_q4_k_minimal(
    const void*  weights_q4_k,         // block_q4_K_raw* (16 blocks)
    const void*  activations_q8_1_mmq, // block_q8_1_mmq_ds4* (16 blocks)
    float*       output,
    hipStream_t  stream)
{
    dim3 grid(1, 1, 1);
    dim3 block(32, 1, 1);
    rf_v1_mmq_q4_k_minimal_kernel<<<grid, block, 0, stream>>>(
        reinterpret_cast<const block_q4_K_raw*>(weights_q4_k),
        reinterpret_cast<const block_q8_1_mmq_ds4*>(activations_q8_1_mmq),
        output);
    return hipGetLastError();
}
