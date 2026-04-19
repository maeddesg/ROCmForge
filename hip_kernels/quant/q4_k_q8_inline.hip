// Phase 8b Step 3 — Q4_K GEMV variants with inline Q8_0 activation quantization.
//
// Mirrors the Q4_0 "q8_inline" pattern from q4_0_gemv.hip and
// q4_0_fused_q8.hip. The activation vector is quantized to Q8_0
// (shared per 32-element block) cooperatively at block start and kept
// in LDS; every sub-block dot then runs as an integer dot-product
// against the raw Q4_K nibbles plus one FP32 scale multiplication.
//
// Key difference from Q4_0's q8_inline: the Q4_K dequant formula is
// affine, not just `nib·scale`. Per sub-block j ∈ [0, 8):
//
//     value[i] = d · scale_j · nib[i] − dmin · min_j
//
// Against Q8 activations (common scale `x_scale` per Q8_0 block):
//
//     Σ_i value[i] · (x_scale · q8[i])
//         = x_scale · ( d · scale_j · Σ_i nib[i]·q8[i]
//                       − dmin · min_j · Σ_i q8[i] )
//         = x_scale · ( d·scale_j · int_dot − dmin·min_j · q8_sum )
//
// So each sub-block needs TWO integer accumulators (int_dot, q8_sum),
// not just one as in Q4_0. That's the defining implementation detail.
//
// LDS holds `n_rows / 32` Q8_0 blocks (34 bytes each). For Qwen3 K=4096
// → 128 blocks × 34 = 4.3 KB. For Qwen3/Llama ffn_down K=12288/14336 →
// 384/448 blocks × 34 = 13/15 KB. All fit in the 32 KB LDS budget used
// by the rest of the Q4_K GEMV path.

#include "common.hip"
#include <hip/hip_fp16.h>

#define Q4_K_BLOCK_BYTES        144
#define Q4_K_ELEMS_PER_BLOCK    256
#define Q4_K_SUB_BLOCKS         8
#define Q4_K_SUB_BLOCK_ELEMS    32
#define Q4_K_Q8_COLS            4
#define Q4_K_Q8_FIXED_WAVES     8
#define Q4_K_Q8_THREADS_PER_BLOCK (Q4_K_Q8_FIXED_WAVES * WARP_SIZE)
#define Q4_K_Q8_SHARED_MEM_LIMIT 32768

// Q8_0 block shared with the Q4_0 q8_inline path.
struct Q8_0_block_q4k {
    half d;
    int8_t qs[32];
};
#define Q8_0_BLOCK_SIZE_BYTES 34

// ── Inline 6-bit (scale, min) unpack ──────────────────────────────────
__device__ __forceinline__
void q4k_q8_unpack_scale_min(int j, const uint8_t* scales,
                             int* scale_out, int* min_out) {
    if (j < 4) {
        *scale_out = scales[j]     & 0x3F;
        *min_out   = scales[j + 4] & 0x3F;
    } else {
        *scale_out = (scales[j + 4] & 0x0F) | ((scales[j - 4] >> 6) << 4);
        *min_out   = (scales[j + 4] >>  4)  | ((scales[j]     >> 6) << 4);
    }
}

// ── Integer-dot + sum over one Q4_K sub-block against Q8 activations ──
//
// Returns (int_dot, q8_sum) via out-params. `sb_ptr` points to the Q4_K
// super-block header (d, dmin, scales, qs...), `x_block` is the Q8_0
// sub-block for the matching 32-element input slice, and `j` is the
// sub-block index within the super-block (0..7).
__device__ __forceinline__
void q4k_q8_subblock_int_dot(
    const uint8_t* __restrict__ qs,
    const Q8_0_block_q4k* __restrict__ x_block,
    int j,
    int* int_dot_out,
    int* q8_sum_out)
{
    // Q4_K pair-interleaved nibble layout: sub-blocks (2k, 2k+1) share
    // the 32 qs bytes at offset `k*32`; even j uses low nibble, odd uses
    // high nibble.
    const int pair_base = (j >> 1) * 32;
    const bool is_upper = (j & 1) != 0;
    const uint8_t* qs_pair = qs + pair_base;
    const int8_t* x_qs = x_block->qs;

    int int_dot = 0;
    int q8_sum = 0;
    #pragma unroll
    for (int i = 0; i < Q4_K_SUB_BLOCK_ELEMS; ++i) {
        const uint8_t byte = qs_pair[i];
        const int nib = is_upper ? (byte >> 4) : (byte & 0x0F);
        const int x = (int)x_qs[i];
        int_dot += nib * x;
        q8_sum  += x;
    }
    *int_dot_out = int_dot;
    *q8_sum_out  = q8_sum;
}

// ── One Q4_K super-block × matching 8 Q8_0 blocks dot ─────────────────
__device__ __forceinline__
float q4k_q8_super_block_dot(
    const uint8_t* __restrict__ sb_ptr,
    const Q8_0_block_q4k* __restrict__ x_blocks_base)
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
        q4k_q8_unpack_scale_min(j, scales, &scale_j, &min_j);

        int int_dot, q8_sum;
        q4k_q8_subblock_int_dot(qs, &x_blocks_base[j], j, &int_dot, &q8_sum);

        const float x_scale = __half2float(x_blocks_base[j].d);
        acc += x_scale * (
              d_f    * (float)scale_j * (float)int_dot
            - dmin_f * (float)min_j   * (float)q8_sum);
    }
    return acc;
}

// ── Cooperative Q8_0 quantization of the activation vector into LDS ───
//
// Caller layout: 8 waves × 32 threads = 256 threads. Each wave
// processes one Q8_0 block at a time, striped across all waves; 32
// lanes cooperate via shuffle to find max|x| → scale → quantize.
__device__ __forceinline__
void q4k_q8_quantize_activations_to_lds(
    const float* __restrict__ input,
    Q8_0_block_q4k* __restrict__ s_input_q8,
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
        const half scale_h = __float2half(scale);
        const float scale_f = __half2float(scale_h);
        if (lane_id == 0) {
            s_input_q8[block_idx].d = scale_h;
        }

        const float inv_scale = 1.0f / scale_f;
        const float qf = fminf(fmaxf(val * inv_scale, -127.0f), 127.0f);
        s_input_q8[block_idx].qs[lane_id] = static_cast<int8_t>(qf);
    }
}

// ── Plain GEMV: q4_k × q8-inline activation ───────────────────────────
template<int N_WAVES>
__launch_bounds__(Q4_K_Q8_THREADS_PER_BLOCK, 1)
__global__ void gemv_q4_k_f32_q8_inline_multi_row_kernel(
    const void* __restrict__ weights_q4_k,
    const float* __restrict__ input,
    float* __restrict__ output,
    int n_rows,
    int ncols_dst)
{
    const int tid = threadIdx.x;
    const int wave_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;
    const int col_base = (blockIdx.x * N_WAVES + wave_id) * Q4_K_Q8_COLS;
    const int super_blocks_per_row = n_rows / Q4_K_ELEMS_PER_BLOCK;
    const int q8_blocks_total      = n_rows / 32;
    if (col_base >= ncols_dst) return;

    extern __shared__ Q8_0_block_q4k s_input_q8[];
    q4k_q8_quantize_activations_to_lds(
        input, s_input_q8, q8_blocks_total, wave_id, lane_id, N_WAVES);
    __syncthreads();

    const uint8_t* w_cols[Q4_K_Q8_COLS];
    #pragma unroll
    for (int c = 0; c < Q4_K_Q8_COLS; ++c) {
        const int col = col_base + c;
        w_cols[c] = (col < ncols_dst)
            ? static_cast<const uint8_t*>(weights_q4_k)
              + (size_t)col * (size_t)super_blocks_per_row * Q4_K_BLOCK_BYTES
            : nullptr;
    }

    float sums[Q4_K_Q8_COLS] = {0.0f, 0.0f, 0.0f, 0.0f};

    for (int sb = lane_id; sb < super_blocks_per_row; sb += WARP_SIZE) {
        const Q8_0_block_q4k* x_blocks = &s_input_q8[sb * Q4_K_SUB_BLOCKS];
        #pragma unroll
        for (int c = 0; c < Q4_K_Q8_COLS; ++c) {
            if (w_cols[c]) {
                const uint8_t* sb_ptr = w_cols[c] + (size_t)sb * Q4_K_BLOCK_BYTES;
                sums[c] += q4k_q8_super_block_dot(sb_ptr, x_blocks);
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

// ── GEMV + residual: out = (w · x) + residual ─────────────────────────
template<int N_WAVES>
__launch_bounds__(Q4_K_Q8_THREADS_PER_BLOCK, 1)
__global__ void gemv_q4_k_f32_q8_inline_residual_multi_row_kernel(
    const void* __restrict__ weights_q4_k,
    const float* __restrict__ input,
    const float* __restrict__ residual,
    float* __restrict__ output,
    int n_rows,
    int ncols_dst)
{
    const int tid = threadIdx.x;
    const int wave_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;
    const int col_base = (blockIdx.x * N_WAVES + wave_id) * Q4_K_Q8_COLS;
    const int super_blocks_per_row = n_rows / Q4_K_ELEMS_PER_BLOCK;
    const int q8_blocks_total      = n_rows / 32;
    if (col_base >= ncols_dst) return;

    extern __shared__ Q8_0_block_q4k s_input_q8[];
    q4k_q8_quantize_activations_to_lds(
        input, s_input_q8, q8_blocks_total, wave_id, lane_id, N_WAVES);
    __syncthreads();

    const uint8_t* w_cols[Q4_K_Q8_COLS];
    #pragma unroll
    for (int c = 0; c < Q4_K_Q8_COLS; ++c) {
        const int col = col_base + c;
        w_cols[c] = (col < ncols_dst)
            ? static_cast<const uint8_t*>(weights_q4_k)
              + (size_t)col * (size_t)super_blocks_per_row * Q4_K_BLOCK_BYTES
            : nullptr;
    }

    float sums[Q4_K_Q8_COLS] = {0.0f, 0.0f, 0.0f, 0.0f};

    for (int sb = lane_id; sb < super_blocks_per_row; sb += WARP_SIZE) {
        const Q8_0_block_q4k* x_blocks = &s_input_q8[sb * Q4_K_SUB_BLOCKS];
        #pragma unroll
        for (int c = 0; c < Q4_K_Q8_COLS; ++c) {
            if (w_cols[c]) {
                const uint8_t* sb_ptr = w_cols[c] + (size_t)sb * Q4_K_BLOCK_BYTES;
                sums[c] += q4k_q8_super_block_dot(sb_ptr, x_blocks);
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
            const int col = col_base + c;
            if (col < ncols_dst) {
                output[col] = sums[c] + residual[col];
            }
        }
    }
}

// ── Fused Gate+Up+SwiGLU with q8-inline activation ────────────────────
//
// Replaces the 4-dispatch sequence (gate GEMV, up GEMV, SiLU, mul) with
// one kernel that reads the Q8-quantized activation ONCE from LDS and
// feeds both weight matrices.
template<int N_WAVES>
__launch_bounds__(Q4_K_Q8_THREADS_PER_BLOCK, 1)
__global__ void gemv_gate_up_swiglu_q4_k_f32_q8_inline_multi_row_kernel(
    const void* __restrict__ weights_gate_q4_k,
    const void* __restrict__ weights_up_q4_k,
    const float* __restrict__ input,
    float* __restrict__ swiglu_out,
    int n_rows,
    int ncols_dst)
{
    const int tid = threadIdx.x;
    const int wave_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;
    const int col_base = (blockIdx.x * N_WAVES + wave_id) * Q4_K_Q8_COLS;
    const int super_blocks_per_row = n_rows / Q4_K_ELEMS_PER_BLOCK;
    const int q8_blocks_total      = n_rows / 32;
    if (col_base >= ncols_dst) return;

    extern __shared__ Q8_0_block_q4k s_input_q8[];
    q4k_q8_quantize_activations_to_lds(
        input, s_input_q8, q8_blocks_total, wave_id, lane_id, N_WAVES);
    __syncthreads();

    const uint8_t* w_gate_cols[Q4_K_Q8_COLS];
    const uint8_t* w_up_cols[Q4_K_Q8_COLS];
    #pragma unroll
    for (int c = 0; c < Q4_K_Q8_COLS; ++c) {
        const int col = col_base + c;
        if (col < ncols_dst) {
            const size_t off = (size_t)col * (size_t)super_blocks_per_row * Q4_K_BLOCK_BYTES;
            w_gate_cols[c] = static_cast<const uint8_t*>(weights_gate_q4_k) + off;
            w_up_cols[c]   = static_cast<const uint8_t*>(weights_up_q4_k) + off;
        } else {
            w_gate_cols[c] = nullptr;
            w_up_cols[c]   = nullptr;
        }
    }

    float gate_sums[Q4_K_Q8_COLS] = {0.0f, 0.0f, 0.0f, 0.0f};
    float up_sums[Q4_K_Q8_COLS]   = {0.0f, 0.0f, 0.0f, 0.0f};

    for (int sb = lane_id; sb < super_blocks_per_row; sb += WARP_SIZE) {
        const Q8_0_block_q4k* x_blocks = &s_input_q8[sb * Q4_K_SUB_BLOCKS];
        #pragma unroll
        for (int c = 0; c < Q4_K_Q8_COLS; ++c) {
            if (w_gate_cols[c]) {
                gate_sums[c] += q4k_q8_super_block_dot(
                    w_gate_cols[c] + (size_t)sb * Q4_K_BLOCK_BYTES, x_blocks);
                up_sums[c] += q4k_q8_super_block_dot(
                    w_up_cols[c] + (size_t)sb * Q4_K_BLOCK_BYTES, x_blocks);
            }
        }
    }

    #pragma unroll
    for (int c = 0; c < Q4_K_Q8_COLS; ++c) {
        for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
            gate_sums[c] += __shfl_down(gate_sums[c], offset);
            up_sums[c]   += __shfl_down(up_sums[c], offset);
        }
    }

    if (lane_id == 0) {
        #pragma unroll
        for (int c = 0; c < Q4_K_Q8_COLS; ++c) {
            const int col = col_base + c;
            if (col < ncols_dst) {
                const float gate = gate_sums[c];
                const float silu = gate / (1.0f + expf(-gate));
                swiglu_out[col] = silu * up_sums[c];
            }
        }
    }
}

// ── Launchers ─────────────────────────────────────────────────────────

static inline bool q4k_q8_fits_lds(int n_rows) {
    // LDS stores n_rows/32 Q8_0 blocks of 34 bytes each.
    const size_t bytes = (size_t)(n_rows / 32) * Q8_0_BLOCK_SIZE_BYTES;
    return bytes <= Q4_K_Q8_SHARED_MEM_LIMIT;
}

extern "C" hipError_t gemv_q4_k_f32_q8_inline_launch(
    const void* weights_q4_k,
    const float* input,
    float* output,
    int n_rows,
    int ncols_dst,
    hipStream_t stream)
{
    if (n_rows <= 0 || ncols_dst <= 0) return hipErrorInvalidValue;
    if ((n_rows % Q4_K_ELEMS_PER_BLOCK) != 0) return hipErrorInvalidValue;
    if (!q4k_q8_fits_lds(n_rows)) return hipErrorInvalidValue;

    const int threads_per_block = Q4_K_Q8_THREADS_PER_BLOCK;
    const int n_blocks_x =
        (ncols_dst + (Q4_K_Q8_FIXED_WAVES * Q4_K_Q8_COLS) - 1)
        / (Q4_K_Q8_FIXED_WAVES * Q4_K_Q8_COLS);
    const size_t shared_mem = (size_t)(n_rows / 32) * Q8_0_BLOCK_SIZE_BYTES;

    gemv_q4_k_f32_q8_inline_multi_row_kernel<Q4_K_Q8_FIXED_WAVES>
        <<<n_blocks_x, threads_per_block, shared_mem, stream>>>(
            weights_q4_k, input, output, n_rows, ncols_dst);
    return hipGetLastError();
}

extern "C" hipError_t gemv_q4_k_f32_q8_inline_residual_launch(
    const void* weights_q4_k,
    const float* input,
    const float* residual,
    float* output,
    int n_rows,
    int ncols_dst,
    hipStream_t stream)
{
    if (n_rows <= 0 || ncols_dst <= 0 || residual == nullptr) return hipErrorInvalidValue;
    if ((n_rows % Q4_K_ELEMS_PER_BLOCK) != 0) return hipErrorInvalidValue;
    if (!q4k_q8_fits_lds(n_rows)) return hipErrorInvalidValue;

    const int threads_per_block = Q4_K_Q8_THREADS_PER_BLOCK;
    const int n_blocks_x =
        (ncols_dst + (Q4_K_Q8_FIXED_WAVES * Q4_K_Q8_COLS) - 1)
        / (Q4_K_Q8_FIXED_WAVES * Q4_K_Q8_COLS);
    const size_t shared_mem = (size_t)(n_rows / 32) * Q8_0_BLOCK_SIZE_BYTES;

    gemv_q4_k_f32_q8_inline_residual_multi_row_kernel<Q4_K_Q8_FIXED_WAVES>
        <<<n_blocks_x, threads_per_block, shared_mem, stream>>>(
            weights_q4_k, input, residual, output, n_rows, ncols_dst);
    return hipGetLastError();
}

extern "C" hipError_t gemv_gate_up_swiglu_q4_k_f32_q8_inline_launch(
    const void* weights_gate_q4_k,
    const void* weights_up_q4_k,
    const float* input,
    float* swiglu_out,
    int n_rows,
    int ncols_dst,
    hipStream_t stream)
{
    if (n_rows <= 0 || ncols_dst <= 0) return hipErrorInvalidValue;
    if ((n_rows % Q4_K_ELEMS_PER_BLOCK) != 0) return hipErrorInvalidValue;
    if (weights_gate_q4_k == nullptr || weights_up_q4_k == nullptr) {
        return hipErrorInvalidValue;
    }
    if (!q4k_q8_fits_lds(n_rows)) return hipErrorInvalidValue;

    const int threads_per_block = Q4_K_Q8_THREADS_PER_BLOCK;
    const int n_blocks_x =
        (ncols_dst + (Q4_K_Q8_FIXED_WAVES * Q4_K_Q8_COLS) - 1)
        / (Q4_K_Q8_FIXED_WAVES * Q4_K_Q8_COLS);
    const size_t shared_mem = (size_t)(n_rows / 32) * Q8_0_BLOCK_SIZE_BYTES;

    gemv_gate_up_swiglu_q4_k_f32_q8_inline_multi_row_kernel<Q4_K_Q8_FIXED_WAVES>
        <<<n_blocks_x, threads_per_block, shared_mem, stream>>>(
            weights_gate_q4_k, weights_up_q4_k, input, swiglu_out, n_rows, ncols_dst);
    return hipGetLastError();
}
