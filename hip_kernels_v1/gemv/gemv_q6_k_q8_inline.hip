// Hand-written for Phase 2 Schritt 2.1.4. Structure mirrors
// hip_kernels_v1/gemv/gemv_q4_k_q8_inline.hip; the only substantive
// differences sit in the weight-unpack path (Q6_K 6-bit ql|qh split
// with per-16-element sub-scales) and the absence of a per-super-
// block `dmin` term (Q6_K uses `d * scale_sub * (q6 - 32)` only).

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>

// MODE.FP_DENORM = 0xF — IEEE denormals preserved (§7.4 Rule 4 / §10.5).
__device__ __forceinline__ void rf_v1_gemv_set_ieee_denormal_mode() {
    asm volatile("s_setreg_imm32_b32 hwreg(HW_REG_MODE, 4, 4), 0xF");
}

// Q6_K GEMV with Q8-inline activation.
//
// Layout reminder (see docs/v1.0/dequant_ir_spec.md §3.7):
//   Block = 210 bytes / 256 elements
//     ql[0 .. 128)  — 4-bit nibbles, low / high packed per byte
//     qh[128 .. 192) — 2-bit pairs, 4 per byte
//     sc[192 .. 208) — 16 signed int8 sub-scales
//     d  [208 .. 210) — fp16 super-block scale
//
// The standard Q6_K kernel organises the super-block into "two halves
// of 128 elements", each half in turn into four 32-element quads
// (q1 / q2 / q3 / q4 at offsets 0, 32, 64, 96). We keep the same
// half/quad decomposition here but reshape the inner loop so that
// **one lane processes one full super-block**, matching the Q4_K
// q8-inline kernel's per-lane granularity. 8 Q8_0 activation blocks
// map 1:1 onto the 8 (half, quad) tiles — block b8 covers positions
// `[b8*32, b8*32+32)` within the super-block.

#define WARP_SIZE                 32
#define Q6_K_BLOCK_BYTES          210
#define Q6_K_ELEMS_PER_BLOCK      256
#define Q6_K_Q8_COLS              4
#define Q6_K_Q8_FIXED_WAVES       8
#define Q6_K_Q8_THREADS_PER_BLOCK (Q6_K_Q8_FIXED_WAVES * WARP_SIZE)
#define Q6_K_Q8_SHARED_MEM_LIMIT  (48 * 1024)
#define Q8_0_BLOCK_SIZE_BYTES     34

struct Q8_0_block_q6k_inline {
    __half d;
    int8_t qs[32];
};

// Unpack one 32-element "Q8 block" (two adjacent 16-element Q6 sub-blocks)
// against a Q8_0 activation block. Returns two (int_dot, q8_sum) pairs —
// lower 16 (sc_lo) and upper 16 (sc_up). The caller applies the sub-
// scales.
//
// `b8` selects which of the 8 (half, quad) tiles this call covers.
//   b8 = 0 : half 0, quad 0 (positions 0   .. 31)
//   b8 = 1 : half 0, quad 1 (positions 32  .. 63)
//   b8 = 2 : half 0, quad 2 (positions 64  .. 95)
//   b8 = 3 : half 0, quad 3 (positions 96  .. 127)
//   b8 = 4 : half 1, quad 0 (positions 128 .. 159)
//   b8 = 5 : half 1, quad 1 (positions 160 .. 191)
//   b8 = 6 : half 1, quad 2 (positions 192 .. 223)
//   b8 = 7 : half 1, quad 3 (positions 224 .. 255)
__device__ __forceinline__
void rf_v1_q6k_q8_block32_int_dot(
    const uint8_t* __restrict__ ql,
    const uint8_t* __restrict__ qh,
    int b8,
    const int8_t* __restrict__ x_qs,
    int* int_dot_lo_out,
    int* q8_sum_lo_out,
    int* int_dot_up_out,
    int* q8_sum_up_out)
{
    const int n = b8 >> 2;            // half index 0 or 1
    const int q_idx = b8 & 3;         // quad index 0..3 within half

    // Half-local pointers.
    const uint8_t* ql_half = ql + (size_t)n * 64;
    const uint8_t* qh_half = qh + (size_t)n * 32;

    // Which ql-offset and shift to use for this quad:
    //   quad 0: ql[i]      LOW  + qh[i] bits 0-1
    //   quad 1: ql[i + 32] LOW  + qh[i] bits 2-3
    //   quad 2: ql[i]      HIGH + qh[i] bits 4-5
    //   quad 3: ql[i + 32] HIGH + qh[i] bits 6-7
    const int ql_off  = (q_idx & 1) ? 32 : 0;
    const bool ql_hi  = (q_idx & 2) != 0;
    const int qh_shift = q_idx * 2;

    int int_dot_lo = 0, q8_sum_lo = 0;
    int int_dot_up = 0, q8_sum_up = 0;

    #pragma unroll
    for (int i = 0; i < 16; ++i) {
        // lower 16 elements (sub-scale A)
        {
            const uint8_t qlb = ql_half[i + ql_off];
            const int qln = ql_hi ? (int)((qlb >> 4) & 0xF) : (int)(qlb & 0xF);
            const int qhn = (int)((qh_half[i] >> qh_shift) & 0x3);
            const int q6  = qln | (qhn << 4);   // un-offset 6-bit value [0..63]
            const int x   = (int)x_qs[i];
            int_dot_lo += q6 * x;
            q8_sum_lo  += x;
        }
        // upper 16 elements (sub-scale B)
        {
            const int j = i + 16;
            const uint8_t qlb = ql_half[j + ql_off];
            const int qln = ql_hi ? (int)((qlb >> 4) & 0xF) : (int)(qlb & 0xF);
            const int qhn = (int)((qh_half[j] >> qh_shift) & 0x3);
            const int q6  = qln | (qhn << 4);
            const int x   = (int)x_qs[j];
            int_dot_up += q6 * x;
            q8_sum_up  += x;
        }
    }

    *int_dot_lo_out = int_dot_lo;
    *q8_sum_lo_out  = q8_sum_lo;
    *int_dot_up_out = int_dot_up;
    *q8_sum_up_out  = q8_sum_up;
}

// Full super-block dot: 256 elements against 8 matching Q8_0 blocks.
__device__ __forceinline__
float rf_v1_q6k_q8_super_block_dot(
    const uint8_t* __restrict__ sb_ptr,
    const Q8_0_block_q6k_inline* __restrict__ x_blocks_base)
{
    __half d_half;
    memcpy(&d_half, sb_ptr + 208, sizeof(__half));
    const float d = __half2float(d_half);

    const uint8_t* ql = sb_ptr;
    const uint8_t* qh = sb_ptr + 128;
    const int8_t*  sc = reinterpret_cast<const int8_t*>(sb_ptr + 192);

    float acc = 0.0f;
    #pragma unroll
    for (int b8 = 0; b8 < 8; ++b8) {
        int int_dot_lo, q8_sum_lo, int_dot_up, q8_sum_up;
        rf_v1_q6k_q8_block32_int_dot(
            ql, qh, b8,
            x_blocks_base[b8].qs,
            &int_dot_lo, &q8_sum_lo,
            &int_dot_up, &q8_sum_up);

        // Sub-scale layout (matches the standard kernel's
        //   `s1..4 = d * sc[is + 0, 2, 4, 6]` with is = lane/16):
        //     block b8: sc[b8*2] for lower 16, sc[b8*2 + 1] for upper 16
        const int sc_lo = (int)sc[b8 * 2];
        const int sc_up = (int)sc[b8 * 2 + 1];

        const float x_scale = __half2float(x_blocks_base[b8].d);

        // (q6 - 32) × q8 = q6 × q8 - 32 × q8 — keep the integer sum
        // in i32, then convert once at the end.
        const float partial_lo = x_scale * d * (float)sc_lo
            * ((float)int_dot_lo - 32.0f * (float)q8_sum_lo);
        const float partial_up = x_scale * d * (float)sc_up
            * ((float)int_dot_up - 32.0f * (float)q8_sum_up);
        acc += partial_lo + partial_up;
    }
    return acc;
}

// Cooperative Q8_0 quantisation of the activation vector into LDS.
// Identical in structure to the Q4_K variant — 8 waves × 32 lanes,
// each wave quantises one 32-element block at a time.
__device__ __forceinline__
void rf_v1_q6k_q8_quantize_activations_to_lds(
    const float* __restrict__ input,
    Q8_0_block_q6k_inline* __restrict__ s_input_q8,
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

__launch_bounds__(Q6_K_Q8_THREADS_PER_BLOCK, 1)
__global__ void rf_v1_gemv_q6_k_q8_inline_kernel(
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
    const int col_base = (blockIdx.x * Q6_K_Q8_FIXED_WAVES + wave_id) * Q6_K_Q8_COLS;
    const int super_blocks_per_row = n_rows / Q6_K_ELEMS_PER_BLOCK;
    const int q8_blocks_total      = n_rows / 32;

    extern __shared__ Q8_0_block_q6k_inline s_input_q8_q6[];

    // Cooperative Q8 quantisation — ALL threads must pass the barrier,
    // even those whose output columns lie past ncols_dst.
    rf_v1_q6k_q8_quantize_activations_to_lds(
        input, s_input_q8_q6, q8_blocks_total, wave_id, lane_id, Q6_K_Q8_FIXED_WAVES);
    __syncthreads();

    if (col_base >= ncols_dst) return;

    const uint8_t* w_cols[Q6_K_Q8_COLS];
    #pragma unroll
    for (int c = 0; c < Q6_K_Q8_COLS; ++c) {
        const int col = col_base + c;
        w_cols[c] = (col < ncols_dst)
            ? weights + (size_t)col * (size_t)super_blocks_per_row * Q6_K_BLOCK_BYTES
            : nullptr;
    }

    float sums[Q6_K_Q8_COLS] = {0.0f, 0.0f, 0.0f, 0.0f};

    // Each lane walks a disjoint stripe of super-blocks — Q4_K-style
    // per-lane super-block parallelism. 8 Q8 sub-blocks live at
    // [sb * 8 .. sb * 8 + 8) in s_input_q8_q6.
    for (int sb = lane_id; sb < super_blocks_per_row; sb += WARP_SIZE) {
        const Q8_0_block_q6k_inline* x_blocks = &s_input_q8_q6[sb * 8];
        #pragma unroll
        for (int c = 0; c < Q6_K_Q8_COLS; ++c) {
            if (w_cols[c]) {
                const uint8_t* sb_ptr = w_cols[c] + (size_t)sb * Q6_K_BLOCK_BYTES;
                sums[c] += rf_v1_q6k_q8_super_block_dot(sb_ptr, x_blocks);
            }
        }
    }

    #pragma unroll
    for (int c = 0; c < Q6_K_Q8_COLS; ++c) {
        for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
            sums[c] += __shfl_down(sums[c], offset);
        }
    }

    if (lane_id == 0) {
        #pragma unroll
        for (int c = 0; c < Q6_K_Q8_COLS; ++c) {
            if (col_base + c < ncols_dst) {
                output[col_base + c] = sums[c];
            }
        }
    }
}

extern "C" hipError_t rocmforge_launch_gemv_q6_k_q8_inline(
    const uint8_t* weights,
    const float*   input,
    float*         output,
    int n_rows,
    int ncols_dst,
    hipStream_t stream)
{
    if (n_rows <= 0 || ncols_dst <= 0)            return hipErrorInvalidValue;
    if ((n_rows % Q6_K_ELEMS_PER_BLOCK) != 0)     return hipErrorInvalidValue;

    const size_t shared_mem = (size_t)(n_rows / 32) * Q8_0_BLOCK_SIZE_BYTES;
    if (shared_mem > Q6_K_Q8_SHARED_MEM_LIMIT)    return hipErrorInvalidValue;

    const int n_blocks_x =
        (ncols_dst + (Q6_K_Q8_FIXED_WAVES * Q6_K_Q8_COLS) - 1)
        / (Q6_K_Q8_FIXED_WAVES * Q6_K_Q8_COLS);

    rf_v1_gemv_q6_k_q8_inline_kernel<<<n_blocks_x, Q6_K_Q8_THREADS_PER_BLOCK, shared_mem, stream>>>(
        weights, input, output, n_rows, ncols_dst);
    return hipGetLastError();
}
