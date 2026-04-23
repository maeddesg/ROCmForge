// Phase 2 follow-up: Q4_K GEMV with Q8-inline activations, inner
// dot-product replaced by the gfx1201 `v_dot4_i32_iu8` intrinsic
// (`__builtin_amdgcn_sudot4`). Identical shape constraints and
// thread layout to `gemv_q4_k_q8_inline.hip`; the only changes are
// in the sub-block dot.
//
// Per-sub-block work before (32 int8 × int8 MACs):
//   32 × byte-load + 32 × nibble-extract + 32 × mul + 32 × add
//   + 32 × add for q8_sum
//   ≈ 160 VALU-ops per sub-block
//
// Per-sub-block work after:
//   8 × int32 weight-load + 8 × int32 q8-load
//   + 8 × sudot4 for the weighted dot (4 MACs each)
//   + 8 × sudot4(0x01010101, q8) for q8_sum (4 adds each)
//   ≈ 16 VALU-ops per sub-block
//
// Correctness: Q4_K nibbles are 0..15, safe to interpret as
// signed int8. sudot4(signed,signed) matches the signed mul that
// the reference kernel performs. q8_sum reduction via
// `sudot4(0x01010101, q8)` is the llama.cpp trick — the identity
// multiplier 1 times a byte just sums.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>

// MODE.FP_DENORM = 0xF — IEEE denormals preserved (§7.4 Rule 4 / §10.5).
__device__ __forceinline__ void rf_v1_gemv_set_ieee_denormal_mode() {
    asm volatile("s_setreg_imm32_b32 hwreg(HW_REG_MODE, 4, 4), 0xF");
}

#define WARP_SIZE                 32
#define Q4_K_BLOCK_BYTES          144
#define Q4_K_ELEMS_PER_BLOCK      256
#define Q4_K_SUB_BLOCKS           8
#define Q4_K_SUB_BLOCK_ELEMS      32
#define Q4_K_Q8_COLS              4
#define Q4_K_Q8_FIXED_WAVES       8
#define Q4_K_Q8_THREADS_PER_BLOCK (Q4_K_Q8_FIXED_WAVES * WARP_SIZE)
#define Q4_K_Q8_SHARED_MEM_LIMIT  (48 * 1024)
#define Q8_0_BLOCK_SIZE_BYTES     34

struct Q8_0_block_q4k_inline_sudot4 {
    __half d;
    int8_t qs[32];
};

__device__ __forceinline__
void rf_v1_q4k_q8_sudot4_unpack_scale_min(int j, const uint8_t* scales,
                                           int* scale_out, int* min_out) {
    if (j < 4) {
        *scale_out = scales[j]     & 0x3F;
        *min_out   = scales[j + 4] & 0x3F;
    } else {
        *scale_out = (scales[j + 4] & 0x0F) | ((scales[j - 4] >> 6) << 4);
        *min_out   = (scales[j + 4] >> 4)   | ((scales[j]     >> 6) << 4);
    }
}

// One Q4_K sub-block (32 elements) × one Q8_0 activation block →
// (int_dot, q8_sum). int_dot accumulates sum of (nibble × q8[i]),
// q8_sum accumulates sum of q8[i] (for the -dmin*min*sum branch).
//
// qs layout: two sub-blocks share the same 32 bytes of `qs_pair`.
// j even → use LOW nibbles;  j odd → use HIGH nibbles. This mirrors
// the reference kernel so the caller's sub-block indexing is
// unchanged.
__device__ __forceinline__
void rf_v1_q4k_q8_sudot4_subblock_int_dot(
    const uint8_t* __restrict__ qs,
    const Q8_0_block_q4k_inline_sudot4* __restrict__ x_block,
    int j,
    int* int_dot_out,
    int* q8_sum_out)
{
    const int pair_base = (j >> 1) * 32;
    const bool is_upper = (j & 1) != 0;

    // `qs + pair_base` is always 4-byte aligned: Q4_K blocks are
    // 144 B = 36 ints long and `pair_base ∈ {0, 32, 64, 96}`, both
    // multiples of 4. The Q4_K ql payload starts at block+16, also
    // 4-byte aligned. Reinterpret as const int*.
    const int* qs32 = reinterpret_cast<const int*>(qs + pair_base);
    const int* x32  = reinterpret_cast<const int*>(x_block->qs);

    int int_dot = 0;
    int q8_sum  = 0;

    // 32 bytes of qs = 8 × int32. Same for 32 bytes of q8.
    // Inner loop: 8 sudot4 for weighted dot + 8 sudot4 for q8_sum.
    #pragma unroll
    for (int k = 0; k < 8; ++k) {
        const int qs_word = qs32[k];
        const int q8_word = x32[k];

        // Extract LOW nibbles (4 × 4 bits packed as 4 × int8) or
        // HIGH nibbles, depending on sub-block parity.
        const int nibbles = is_upper
            ? ((qs_word >> 4) & 0x0F0F0F0F)
            :  (qs_word        & 0x0F0F0F0F);

        // sudot4(signed, signed): Q4_K nibbles fit [0..15] so the
        // signed interpretation is exact; Q8 activations are signed
        // int8. Compiler lowers to v_dot4_i32_iu8 on gfx1201.
        int_dot = __builtin_amdgcn_sudot4(true, nibbles, true, q8_word, int_dot, false);

        // q8_sum via sudot4(0x01010101, q8) — the identity multiplier
        // collapses the 4 products to a plain sum.
        q8_sum  = __builtin_amdgcn_sudot4(true, 0x01010101, true, q8_word, q8_sum,  false);
    }

    *int_dot_out = int_dot;
    *q8_sum_out  = q8_sum;
}

__device__ __forceinline__
float rf_v1_q4k_q8_sudot4_super_block_dot(
    const uint8_t* __restrict__ sb_ptr,
    const Q8_0_block_q4k_inline_sudot4* __restrict__ x_blocks_base)
{
    const __half d_h    = *reinterpret_cast<const __half*>(sb_ptr + 0);
    const __half dmin_h = *reinterpret_cast<const __half*>(sb_ptr + 2);
    const float  d_f    = __half2float(d_h);
    const float  dmin_f = __half2float(dmin_h);
    const uint8_t* scales = sb_ptr + 4;
    const uint8_t* qs     = sb_ptr + 16;

    float acc = 0.0f;
    #pragma unroll
    for (int j = 0; j < Q4_K_SUB_BLOCKS; ++j) {
        int scale_j, min_j;
        rf_v1_q4k_q8_sudot4_unpack_scale_min(j, scales, &scale_j, &min_j);

        int int_dot, q8_sum;
        rf_v1_q4k_q8_sudot4_subblock_int_dot(qs, &x_blocks_base[j], j, &int_dot, &q8_sum);

        const float x_scale = __half2float(x_blocks_base[j].d);
        acc += x_scale * (
              d_f    * (float)scale_j * (float)int_dot
            - dmin_f * (float)min_j   * (float)q8_sum);
    }
    return acc;
}

// Cooperative Q8_0 quantisation of the activation vector into LDS —
// identical to the reference kernel.
__device__ __forceinline__
void rf_v1_q4k_q8_sudot4_quantize_activations_to_lds(
    const float* __restrict__ input,
    Q8_0_block_q4k_inline_sudot4* __restrict__ s_input_q8,
    int n_blocks_total,
    int wave_id,
    int lane_id,
    int n_waves)
{
    for (int block_idx = wave_id; block_idx < n_blocks_total; block_idx += n_waves) {
        const int row_offset = block_idx * 32;
        const float val = input[row_offset + lane_id];
        float max_val = fabsf(val);

        #pragma unroll
        for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
            max_val = fmaxf(max_val, __shfl_down(max_val, offset));
        }

        const float scale = fmaxf(__shfl(max_val, 0, WARP_SIZE) / 127.0f, 1e-30f);
        const __half scale_h = __float2half(scale);
        const float scale_f = __half2float(scale_h);
        if (lane_id == 0) {
            s_input_q8[block_idx].d = scale_h;
        }

        const float inv_scale = 1.0f / scale_f;
        const float qf = fminf(fmaxf(val * inv_scale, -127.0f), 127.0f);
        s_input_q8[block_idx].qs[lane_id] = static_cast<int8_t>(qf);
    }
}

__launch_bounds__(Q4_K_Q8_THREADS_PER_BLOCK, 1)
__global__ void rf_v1_gemv_q4_k_q8_inline_sudot4_kernel(
    const uint8_t* __restrict__ weights,
    const float*   __restrict__ input,
    float*         __restrict__ output,
    int n_rows,
    int ncols_dst)
{
    rf_v1_gemv_set_ieee_denormal_mode();

    const int tid = threadIdx.x;
    const int wave_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;
    const int col_base = (blockIdx.x * Q4_K_Q8_FIXED_WAVES + wave_id) * Q4_K_Q8_COLS;
    const int super_blocks_per_row = n_rows / Q4_K_ELEMS_PER_BLOCK;
    const int q8_blocks_total      = n_rows / 32;

    extern __shared__ Q8_0_block_q4k_inline_sudot4 s_input_q8_sudot4[];

    rf_v1_q4k_q8_sudot4_quantize_activations_to_lds(
        input, s_input_q8_sudot4, q8_blocks_total, wave_id, lane_id, Q4_K_Q8_FIXED_WAVES);
    __syncthreads();

    if (col_base >= ncols_dst) return;

    const uint8_t* w_cols[Q4_K_Q8_COLS];
    #pragma unroll
    for (int c = 0; c < Q4_K_Q8_COLS; ++c) {
        const int col = col_base + c;
        w_cols[c] = (col < ncols_dst)
            ? weights + (size_t)col * (size_t)super_blocks_per_row * Q4_K_BLOCK_BYTES
            : nullptr;
    }

    float sums[Q4_K_Q8_COLS] = {0.0f, 0.0f, 0.0f, 0.0f};

    for (int sb = lane_id; sb < super_blocks_per_row; sb += WARP_SIZE) {
        const Q8_0_block_q4k_inline_sudot4* x_blocks = &s_input_q8_sudot4[sb * Q4_K_SUB_BLOCKS];
        #pragma unroll
        for (int c = 0; c < Q4_K_Q8_COLS; ++c) {
            if (w_cols[c]) {
                const uint8_t* sb_ptr = w_cols[c] + (size_t)sb * Q4_K_BLOCK_BYTES;
                sums[c] += rf_v1_q4k_q8_sudot4_super_block_dot(sb_ptr, x_blocks);
            }
        }
    }

    #pragma unroll
    for (int c = 0; c < Q4_K_Q8_COLS; ++c) {
        for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
            sums[c] += __shfl_down(sums[c], offset);
        }
    }

    if (lane_id == 0) {
        #pragma unroll
        for (int c = 0; c < Q4_K_Q8_COLS; ++c) {
            if (col_base + c < ncols_dst) {
                output[col_base + c] = sums[c];
            }
        }
    }
}

extern "C" hipError_t rocmforge_launch_gemv_q4_k_q8_inline_sudot4(
    const uint8_t* weights,
    const float*   input,
    float*         output,
    int n_rows,
    int ncols_dst,
    hipStream_t stream)
{
    if (n_rows <= 0 || ncols_dst <= 0)               return hipErrorInvalidValue;
    if ((n_rows % Q4_K_ELEMS_PER_BLOCK) != 0)        return hipErrorInvalidValue;
    const size_t shared_mem =
        (size_t)(n_rows / 32) * Q8_0_BLOCK_SIZE_BYTES;
    if (shared_mem > Q4_K_Q8_SHARED_MEM_LIMIT)        return hipErrorInvalidValue;

    const int n_blocks_x =
        (ncols_dst + (Q4_K_Q8_FIXED_WAVES * Q4_K_Q8_COLS) - 1)
        / (Q4_K_Q8_FIXED_WAVES * Q4_K_Q8_COLS);

    rf_v1_gemv_q4_k_q8_inline_sudot4_kernel<<<n_blocks_x, Q4_K_Q8_THREADS_PER_BLOCK, shared_mem, stream>>>(
        weights, input, output, n_rows, ncols_dst);
    return hipGetLastError();
}
