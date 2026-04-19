#include "common.hip"
#include <hip/hip_fp16.h>

// Q6_K constants
#define QK_K 256
#define Q6_K_BLOCK_SIZE 210

// Q6_K block layout (210 bytes per 256 elements):
//   uint8_t ql[128]     — low 4 bits of quantized values
//   uint8_t qh[64]      — high 2 bits of quantized values
//   int8_t  scales[16]  — signed 8-bit per-group scales
//   half    d           — super-block scale at offset 208

__device__ inline float vec_dot_q6_k(const void* block_ptr, const float* vec, int offset) {
    const int tid = threadIdx.x;  // 0..31
    const uint8_t* block_bytes = static_cast<const uint8_t*>(block_ptr);

    // d is at the END of the block (offset 208), not the beginning
    half d_half;
    memcpy(&d_half, block_bytes + 208, sizeof(half));
    const float d = __half2float(d_half);

    const uint8_t* ql = block_bytes;        // 128 bytes
    const uint8_t* qh = block_bytes + 128;  // 64 bytes
    const int8_t* sc = reinterpret_cast<const int8_t*>(block_bytes + 192); // 16 bytes

    float sum = 0.0f;

    // Two passes: first 128 elements (n=0), then next 128 (n=128)
    for (int n = 0; n < 2; ++n) {
        const int is = tid / 16;  // 0 or 1

        // Reconstruct 6-bit quantized values from ql (4 low bits) + qh (2 high bits)
        const int q1 = static_cast<int>((ql[tid +  0] & 0xF) | (((qh[tid] >> 0) & 3) << 4)) - 32;
        const int q2 = static_cast<int>((ql[tid + 32] & 0xF) | (((qh[tid] >> 2) & 3) << 4)) - 32;
        const int q3 = static_cast<int>(((ql[tid +  0] >> 4) & 0xF) | (((qh[tid] >> 4) & 3) << 4)) - 32;
        const int q4 = static_cast<int>(((ql[tid + 32] >> 4) & 0xF) | (((qh[tid] >> 6) & 3) << 4)) - 32;

        const float scale1 = d * static_cast<float>(sc[is + 0]);
        const float scale2 = d * static_cast<float>(sc[is + 2]);
        const float scale3 = d * static_cast<float>(sc[is + 4]);
        const float scale4 = d * static_cast<float>(sc[is + 6]);

        sum += scale1 * static_cast<float>(q1) * vec[offset + tid +  0];
        sum += scale2 * static_cast<float>(q2) * vec[offset + tid + 32];
        sum += scale3 * static_cast<float>(q3) * vec[offset + tid + 64];
        sum += scale4 * static_cast<float>(q4) * vec[offset + tid + 96];

        // Advance pointers for next 128 elements
        ql += 64;
        qh += 32;
        sc += 8;
        offset += 128;
    }

    return sum;
}

__global__ void gemv_q6_k_f32_kernel(
    const void* __restrict__ weights_q6_k,
    const float* __restrict__ input,
    float* __restrict__ output,
    int n_rows,
    int ncols_dst
) {
    const int col = blockIdx.x;
    const int tid = threadIdx.x;
    const int n_blocks = n_rows / QK_K;

    if (col >= ncols_dst) return;

    const uint8_t* col_base = static_cast<const uint8_t*>(weights_q6_k) + col * n_blocks * Q6_K_BLOCK_SIZE;

    float sum = 0.0f;
    for (int block_idx = 0; block_idx < n_blocks; ++block_idx) {
        sum += vec_dot_q6_k(col_base + block_idx * Q6_K_BLOCK_SIZE, input, block_idx * QK_K);
    }

    // Warp reduction (warp size = 32)
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down(sum, offset);
    }

    if (tid == 0) {
        output[col] = sum;
    }
}

// ── Optimized multi-row GEMV (Phase 7 Step 5d) ─────────────────────────
// The scalar kernel above launches one wave per output column. For decode
// shapes with large ncols_dst (4096 V-proj rows, 4096 FFN-down rows,
// ~128K LM-head rows) that's 32× under-utilised. This multi-row variant
// mirrors the Q4_K layout: 8 waves × 32 threads per block, each wave owns
// 4 output columns; a shared-memory cache of the input vector is used
// whenever K fits in 32 KB LDS, otherwise fall through to direct global
// loads (Qwen3 / Llama-3 FFN-down with K = 12,288 or 14,336).
//
// Dequant formula matches the scalar kernel: value = d · scale_j · q,
// where scale_j is the signed int8 at scales[j], q = ((ql & 0xF) |
// (((qh >> shift) & 3) << 4)) - 32.
//
// scales[16] covers 16 sub-groups of 16 elements each; `is = (n << 3) +
// (tid / 16) + {0,2,4,6}` selects the scale for each of the 4 quads.

#define Q6_K_BLOCK_BYTES        Q6_K_BLOCK_SIZE
#define Q6_K_MULTI_ROW_COLS     4
#define Q6_K_FIXED_WAVES        8
#define Q6_K_THREADS_PER_BLOCK  (Q6_K_FIXED_WAVES * 32)
#define Q6_K_SHARED_MEM_LIMIT   32768

__device__ __forceinline__
float q6_k_super_block_dot(const uint8_t* __restrict__ sb_ptr,
                           const float* __restrict__ input_base,
                           int lane_id) {
    half d_half;
    memcpy(&d_half, sb_ptr + 208, sizeof(half));
    const float d = __half2float(d_half);

    const uint8_t* ql = sb_ptr;
    const uint8_t* qh = sb_ptr + 128;
    const int8_t*  sc = reinterpret_cast<const int8_t*>(sb_ptr + 192);

    float acc = 0.0f;

    // Two halves of the 256-element block. Each half uses 64 ql bytes,
    // 32 qh bytes, and 8 of the 16 int8 scales. The 32 lanes split the
    // half into 4 quads of 32 elements.
    for (int n = 0; n < 2; ++n) {
        const int is = lane_id / 16;  // 0 or 1

        const int q1 = static_cast<int>((ql[lane_id +  0] & 0xF) | (((qh[lane_id] >> 0) & 3) << 4)) - 32;
        const int q2 = static_cast<int>((ql[lane_id + 32] & 0xF) | (((qh[lane_id] >> 2) & 3) << 4)) - 32;
        const int q3 = static_cast<int>(((ql[lane_id +  0] >> 4) & 0xF) | (((qh[lane_id] >> 4) & 3) << 4)) - 32;
        const int q4 = static_cast<int>(((ql[lane_id + 32] >> 4) & 0xF) | (((qh[lane_id] >> 6) & 3) << 4)) - 32;

        const float s1 = d * static_cast<float>(sc[is + 0]);
        const float s2 = d * static_cast<float>(sc[is + 2]);
        const float s3 = d * static_cast<float>(sc[is + 4]);
        const float s4 = d * static_cast<float>(sc[is + 6]);

        const int base = n * 128;
        acc += s1 * static_cast<float>(q1) * input_base[base + lane_id +  0];
        acc += s2 * static_cast<float>(q2) * input_base[base + lane_id + 32];
        acc += s3 * static_cast<float>(q3) * input_base[base + lane_id + 64];
        acc += s4 * static_cast<float>(q4) * input_base[base + lane_id + 96];

        ql += 64;
        qh += 32;
        sc += 8;
    }
    return acc;
}

template<int N_WAVES>
__launch_bounds__(Q6_K_THREADS_PER_BLOCK, 1)
__global__ void gemv_q6_k_f32_multi_row_kernel(
    const void* __restrict__ weights_q6_k,
    const float* __restrict__ input,
    float* __restrict__ output,
    int n_rows,
    int ncols_dst
) {
    const int tid = threadIdx.x;
    const int wave_id = tid / 32;
    const int lane_id = tid % 32;

    const int col_base = (blockIdx.x * N_WAVES + wave_id) * Q6_K_MULTI_ROW_COLS;
    const int n_blocks = n_rows / QK_K;
    if (col_base >= ncols_dst) return;

    extern __shared__ float s_input[];
    for (int i = tid; i < n_rows; i += blockDim.x) {
        s_input[i] = input[i];
    }
    __syncthreads();

    const uint8_t* w_cols[Q6_K_MULTI_ROW_COLS];
    #pragma unroll
    for (int c = 0; c < Q6_K_MULTI_ROW_COLS; ++c) {
        const int col = col_base + c;
        w_cols[c] = (col < ncols_dst)
            ? static_cast<const uint8_t*>(weights_q6_k)
              + static_cast<size_t>(col) * n_blocks * Q6_K_BLOCK_BYTES
            : nullptr;
    }

    float sums[Q6_K_MULTI_ROW_COLS] = {0.0f, 0.0f, 0.0f, 0.0f};

    // Each lane strides over super-blocks; each wave covers the whole K
    // dimension for its 4 columns.
    for (int sb = 0; sb < n_blocks; ++sb) {
        const float* in_base = s_input + sb * QK_K;
        #pragma unroll
        for (int c = 0; c < Q6_K_MULTI_ROW_COLS; ++c) {
            if (w_cols[c]) {
                const uint8_t* sb_ptr = w_cols[c] + static_cast<size_t>(sb) * Q6_K_BLOCK_BYTES;
                sums[c] += q6_k_super_block_dot(sb_ptr, in_base, lane_id);
            }
        }
    }

    #pragma unroll
    for (int c = 0; c < Q6_K_MULTI_ROW_COLS; ++c) {
        for (int offset = 16; offset > 0; offset >>= 1) {
            sums[c] += __shfl_down(sums[c], offset);
        }
    }

    if (lane_id == 0) {
        #pragma unroll
        for (int c = 0; c < Q6_K_MULTI_ROW_COLS; ++c) {
            if (col_base + c < ncols_dst) {
                output[col_base + c] = sums[c];
            }
        }
    }
}

template<int N_WAVES>
__launch_bounds__(Q6_K_THREADS_PER_BLOCK, 1)
__global__ void gemv_q6_k_f32_direct_kernel(
    const void* __restrict__ weights_q6_k,
    const float* __restrict__ input,
    float* __restrict__ output,
    int n_rows,
    int ncols_dst
) {
    const int tid = threadIdx.x;
    const int wave_id = tid / 32;
    const int lane_id = tid % 32;

    const int col_base = (blockIdx.x * N_WAVES + wave_id) * Q6_K_MULTI_ROW_COLS;
    const int n_blocks = n_rows / QK_K;
    if (col_base >= ncols_dst) return;

    const uint8_t* w_cols[Q6_K_MULTI_ROW_COLS];
    #pragma unroll
    for (int c = 0; c < Q6_K_MULTI_ROW_COLS; ++c) {
        const int col = col_base + c;
        w_cols[c] = (col < ncols_dst)
            ? static_cast<const uint8_t*>(weights_q6_k)
              + static_cast<size_t>(col) * n_blocks * Q6_K_BLOCK_BYTES
            : nullptr;
    }

    float sums[Q6_K_MULTI_ROW_COLS] = {0.0f, 0.0f, 0.0f, 0.0f};

    for (int sb = 0; sb < n_blocks; ++sb) {
        const float* in_base = input + sb * QK_K;
        #pragma unroll
        for (int c = 0; c < Q6_K_MULTI_ROW_COLS; ++c) {
            if (w_cols[c]) {
                const uint8_t* sb_ptr = w_cols[c] + static_cast<size_t>(sb) * Q6_K_BLOCK_BYTES;
                sums[c] += q6_k_super_block_dot(sb_ptr, in_base, lane_id);
            }
        }
    }

    #pragma unroll
    for (int c = 0; c < Q6_K_MULTI_ROW_COLS; ++c) {
        for (int offset = 16; offset > 0; offset >>= 1) {
            sums[c] += __shfl_down(sums[c], offset);
        }
    }

    if (lane_id == 0) {
        #pragma unroll
        for (int c = 0; c < Q6_K_MULTI_ROW_COLS; ++c) {
            if (col_base + c < ncols_dst) {
                output[col_base + c] = sums[c];
            }
        }
    }
}

extern "C" hipError_t gemv_q6_k_f32_launch(
    const void* weights_q6_k,
    const float* input,
    float* output,
    int n_rows,
    int ncols_dst,
    hipStream_t stream
) {
    if (n_rows <= 0 || ncols_dst <= 0) return hipErrorInvalidValue;
    if (n_rows % QK_K != 0) return hipErrorInvalidValue;

    const int threads_per_block = Q6_K_THREADS_PER_BLOCK;
    const int n_blocks_x =
        (ncols_dst + (Q6_K_FIXED_WAVES * Q6_K_MULTI_ROW_COLS) - 1)
        / (Q6_K_FIXED_WAVES * Q6_K_MULTI_ROW_COLS);

    if (static_cast<size_t>(n_rows) * sizeof(float) <= Q6_K_SHARED_MEM_LIMIT) {
        const size_t shared_mem = static_cast<size_t>(n_rows) * sizeof(float);
        gemv_q6_k_f32_multi_row_kernel<Q6_K_FIXED_WAVES>
            <<<n_blocks_x, threads_per_block, shared_mem, stream>>>(
                weights_q6_k, input, output, n_rows, ncols_dst
            );
    } else {
        gemv_q6_k_f32_direct_kernel<Q6_K_FIXED_WAVES>
            <<<n_blocks_x, threads_per_block, 0, stream>>>(
                weights_q6_k, input, output, n_rows, ncols_dst
            );
    }
    return hipGetLastError();
}
