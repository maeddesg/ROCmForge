#include "common.hip"

// ── GFX12 (RDNA4) Optimized Q4_0 GEMV ──────────────────────────────────────────
//
// Uses v_dot4_u32_u8 (unsigned 4×8-bit dot product) to replace scalar
// multiply-accumulate in the Q4_0 × Q8_0 inner loop.
//
// On gfx1201, all dot-product instructions operate as unsigned×unsigned.
// We handle signs via bias correction:
//
//   Q4_0 nibbles: unsigned 0-15 (signed = nibble - 8)
//   Q8_0 values:  stored as unsigned (signed + 128)
//
//   wanted = Σ (q4_u - 8) × (q8_u - 128)
//          = Σ q4_u × q8_u - 128 × Σ q4_u - 8 × Σ q8_s
//          = unsigned_dot  - 128 × q4_sum - 8 × q8_signed_sum
//
// The Q8_0 signed block sum is precomputed during quantization in shared memory.
// The Q4_0 nibble sum is computed on-the-fly using udot4 with 0x01010101.

#define QK4_0 32
#define Q4_0_BLOCK_SIZE 18
constexpr int GFX12_MULTI_ROW_COLS = 4;
constexpr int GFX12_SMALL_WAVES = 4;
constexpr int GFX12_FIXED_WAVES = 8;
constexpr int GFX12_SHARED_MEM_LIMIT = 32 * 1024;

struct Q4_0_block_gfx12 {
    half d;
    int8_t qs[16];
};

// Modified Q8_0 block: unsigned values + precomputed signed sum
// Layout: [half d][int16 sum_s][uint8 qs_u[32]] = 36 bytes
struct Q8_0_block_gfx12 {
    half d;            // 2 bytes: quantization scale
    int16_t sum_s;     // 2 bytes: sum of original signed values (range -4064..+4064)
    uint8_t qs_u[QK4_0]; // 32 bytes: unsigned values (signed + 128)
};

constexpr int Q8_0_GFX12_BLOCK_SIZE = sizeof(Q8_0_block_gfx12); // 36 bytes

// ── Inline assembly for v_dot4_u32_u8 ───────────────────────────────────────────

__device__ __forceinline__ int udot4_u8(unsigned int a, unsigned int b, int c) {
    int result;
    asm volatile("v_dot4_u32_u8 %0, %1, %2, %3"
                 : "=v"(result) : "v"(a), "v"(b), "v"(c));
    return result;
}

// ── GFX12-optimized Q4_0 × Q8_0 block dot product ──────────────────────────────
//
// Replaces q4_0_q8_0_block_dot with udot4-based computation.
// Returns the CORRECTED signed dot product (same semantics as the original).

__device__ __forceinline__ int q4_0_q8_0_block_dot_gfx12(
    const Q4_0_block_gfx12* __restrict__ w_block,
    const Q8_0_block_gfx12* __restrict__ x_block
) {
    const unsigned int* w_words = reinterpret_cast<const unsigned int*>(w_block->qs);
    const unsigned int* x_lo_words = reinterpret_cast<const unsigned int*>(x_block->qs_u);
    const unsigned int* x_hi_words = reinterpret_cast<const unsigned int*>(x_block->qs_u + 16);

    int unsigned_dot = 0;
    int q4_nibble_sum = 0;
    const unsigned int ONES = 0x01010101u;

    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        const unsigned int w = w_words[i];
        const unsigned int lo = w & 0x0F0F0F0Fu;
        const unsigned int hi = (w >> 4) & 0x0F0F0F0Fu;

        // Unsigned dot products: q4_unsigned × q8_unsigned
        unsigned_dot = udot4_u8(lo, x_lo_words[i], unsigned_dot);
        unsigned_dot = udot4_u8(hi, x_hi_words[i], unsigned_dot);

        // Accumulate nibble sum for bias correction
        // lo+hi per byte is safe: max 15+15=30 < 256, no inter-byte carry
        const unsigned int combined = lo + hi;
        q4_nibble_sum = udot4_u8(combined, ONES, q4_nibble_sum);
    }

    // Bias correction: wanted = unsigned_dot - 128*q4_sum - 8*q8_signed_sum
    return unsigned_dot - 128 * q4_nibble_sum - 8 * (int)x_block->sum_s;
}

// ── Residual GEMV Kernel (GFX12 fast path) ──────────────────────────────────────
//
// Drop-in replacement for gemv_q4_0_f32_q8_inline_residual_multi_row_kernel.
// Same interface, same dispatch geometry, but uses udot4 in the inner loop.

template<int N_WAVES>
__global__ void gemv_q4_0_f32_q8_inline_residual_gfx12_kernel(
    const void* __restrict__ weights_q4_0,
    const float* __restrict__ input,
    const float* __restrict__ residual,
    float* __restrict__ output,
    int n_rows,
    int ncols_dst
) {
    const int tid = threadIdx.x;
    const int wave_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;

    const int col_base = (blockIdx.x * N_WAVES + wave_id) * GFX12_MULTI_ROW_COLS;
    const int n_blocks_total = n_rows / QK4_0;

    if (col_base >= ncols_dst) return;

    // ── Phase 1: Quantize input to biased Q8_0 in shared memory ─────────

    extern __shared__ Q8_0_block_gfx12 s_input_q8[];

    for (int block_idx = wave_id; block_idx < n_blocks_total; block_idx += N_WAVES) {
        const int row_offset = block_idx * QK4_0;
        const float val = input[row_offset + lane_id];

        // Compute block-wise absmax
        float max_val = fabsf(val);
        #pragma unroll
        for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
            max_val = fmaxf(max_val, __shfl_down(max_val, offset));
        }

        const float scale = fmaxf(__shfl(max_val, 0, WARP_SIZE) / 127.0f, 1e-30f);
        const half scale_h = __float2half(scale);
        const float scale_f = __half2float(scale_h);

        const float inv_scale = 1.0f / scale_f;
        const float qf = fminf(fmaxf(val * inv_scale, -127.0f), 127.0f);
        const int8_t qs = static_cast<int8_t>(qf);

        // Store unsigned biased value (signed + 128)
        s_input_q8[block_idx].qs_u[lane_id] = static_cast<uint8_t>(static_cast<int>(qs) + 128);

        // Compute signed block sum via warp reduction
        int lane_sum = static_cast<int>(qs);
        #pragma unroll
        for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
            lane_sum += __shfl_down(lane_sum, offset);
        }

        if (lane_id == 0) {
            s_input_q8[block_idx].d = scale_h;
            s_input_q8[block_idx].sum_s = static_cast<int16_t>(lane_sum);
        }
    }
    __syncthreads();

    // ── Phase 2: GEMV with udot4 fast path ──────────────────────────────

    const Q4_0_block_gfx12* w_cols[GFX12_MULTI_ROW_COLS];
    #pragma unroll
    for (int c = 0; c < GFX12_MULTI_ROW_COLS; ++c) {
        int col = col_base + c;
        w_cols[c] = (col < ncols_dst) ? reinterpret_cast<const Q4_0_block_gfx12*>(
            static_cast<const uint8_t*>(weights_q4_0) + (col * n_blocks_total) * Q4_0_BLOCK_SIZE
        ) : nullptr;
    }

    float sums[GFX12_MULTI_ROW_COLS] = {0.0f, 0.0f, 0.0f, 0.0f};

    for (int block_idx = lane_id; block_idx < n_blocks_total; block_idx += WARP_SIZE) {
        const Q8_0_block_gfx12* x_block = &s_input_q8[block_idx];
        const float x_scale = __half2float(x_block->d);

        #pragma unroll
        for (int c = 0; c < GFX12_MULTI_ROW_COLS; ++c) {
            if (!w_cols[c]) continue;

            const Q4_0_block_gfx12* w_block = &w_cols[c][block_idx];
            const float combined_scale = __half2float(w_block->d) * x_scale;
            const int block_sum = q4_0_q8_0_block_dot_gfx12(w_block, x_block);

            sums[c] += combined_scale * static_cast<float>(block_sum);
        }
    }

    // Warp reduction
    #pragma unroll
    for (int c = 0; c < GFX12_MULTI_ROW_COLS; ++c) {
        for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
            sums[c] += __shfl_down(sums[c], offset);
        }
    }

    if (lane_id == 0) {
        #pragma unroll
        for (int c = 0; c < GFX12_MULTI_ROW_COLS; ++c) {
            if (col_base + c < ncols_dst) {
                output[col_base + c] = sums[c] + residual[col_base + c];
            }
        }
    }
}

// ── Non-residual variant ────────────────────────────────────────────────────────

template<int N_WAVES>
__global__ void gemv_q4_0_f32_q8_inline_gfx12_kernel(
    const void* __restrict__ weights_q4_0,
    const float* __restrict__ input,
    float* __restrict__ output,
    int n_rows,
    int ncols_dst
) {
    const int tid = threadIdx.x;
    const int wave_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;

    const int col_base = (blockIdx.x * N_WAVES + wave_id) * GFX12_MULTI_ROW_COLS;
    const int n_blocks_total = n_rows / QK4_0;

    if (col_base >= ncols_dst) return;

    extern __shared__ Q8_0_block_gfx12 s_input_q8[];

    for (int block_idx = wave_id; block_idx < n_blocks_total; block_idx += N_WAVES) {
        const int row_offset = block_idx * QK4_0;
        const float val = input[row_offset + lane_id];

        float max_val = fabsf(val);
        #pragma unroll
        for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
            max_val = fmaxf(max_val, __shfl_down(max_val, offset));
        }

        const float scale = fmaxf(__shfl(max_val, 0, WARP_SIZE) / 127.0f, 1e-30f);
        const half scale_h = __float2half(scale);
        const float scale_f = __half2float(scale_h);

        const float inv_scale = 1.0f / scale_f;
        const float qf = fminf(fmaxf(val * inv_scale, -127.0f), 127.0f);
        const int8_t qs = static_cast<int8_t>(qf);

        s_input_q8[block_idx].qs_u[lane_id] = static_cast<uint8_t>(static_cast<int>(qs) + 128);

        int lane_sum = static_cast<int>(qs);
        #pragma unroll
        for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
            lane_sum += __shfl_down(lane_sum, offset);
        }

        if (lane_id == 0) {
            s_input_q8[block_idx].d = scale_h;
            s_input_q8[block_idx].sum_s = static_cast<int16_t>(lane_sum);
        }
    }
    __syncthreads();

    const Q4_0_block_gfx12* w_cols[GFX12_MULTI_ROW_COLS];
    #pragma unroll
    for (int c = 0; c < GFX12_MULTI_ROW_COLS; ++c) {
        int col = col_base + c;
        w_cols[c] = (col < ncols_dst) ? reinterpret_cast<const Q4_0_block_gfx12*>(
            static_cast<const uint8_t*>(weights_q4_0) + (col * n_blocks_total) * Q4_0_BLOCK_SIZE
        ) : nullptr;
    }

    float sums[GFX12_MULTI_ROW_COLS] = {0.0f, 0.0f, 0.0f, 0.0f};

    for (int block_idx = lane_id; block_idx < n_blocks_total; block_idx += WARP_SIZE) {
        const Q8_0_block_gfx12* x_block = &s_input_q8[block_idx];
        const float x_scale = __half2float(x_block->d);

        #pragma unroll
        for (int c = 0; c < GFX12_MULTI_ROW_COLS; ++c) {
            if (!w_cols[c]) continue;

            const Q4_0_block_gfx12* w_block = &w_cols[c][block_idx];
            const float combined_scale = __half2float(w_block->d) * x_scale;
            const int block_sum = q4_0_q8_0_block_dot_gfx12(w_block, x_block);

            sums[c] += combined_scale * static_cast<float>(block_sum);
        }
    }

    #pragma unroll
    for (int c = 0; c < GFX12_MULTI_ROW_COLS; ++c) {
        for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
            sums[c] += __shfl_down(sums[c], offset);
        }
    }

    if (lane_id == 0) {
        #pragma unroll
        for (int c = 0; c < GFX12_MULTI_ROW_COLS; ++c) {
            if (col_base + c < ncols_dst) {
                output[col_base + c] = sums[c];
            }
        }
    }
}

// ── C Dispatch Functions ────────────────────────────────────────────────────────

static inline int select_gfx12_waves(int n_blocks_total) {
    if (n_blocks_total <= 24) {
        return GFX12_SMALL_WAVES;
    }
    return GFX12_FIXED_WAVES;
}

extern "C" hipError_t gemv_q4_0_f32_q8_inline_residual_gfx12_launch(
    const void* weights_q4_0,
    const float* input,
    const float* residual,
    float* output,
    int n_rows,
    int ncols_dst,
    hipStream_t stream
) {
    if (n_rows <= 0 || ncols_dst <= 0 || input == nullptr ||
        residual == nullptr || output == nullptr) {
        return hipErrorInvalidValue;
    }
    if (n_rows % QK4_0 != 0) return hipErrorInvalidValue;

    const int n_blocks_total = n_rows / QK4_0;
    const size_t shared_mem = n_blocks_total * Q8_0_GFX12_BLOCK_SIZE;
    if (shared_mem > GFX12_SHARED_MEM_LIMIT) return hipErrorInvalidValue;

    const int n_waves = select_gfx12_waves(n_blocks_total);
    const int threads_per_block = n_waves * WARP_SIZE;
    const int n_blocks_x =
        (ncols_dst + (n_waves * GFX12_MULTI_ROW_COLS) - 1) / (n_waves * GFX12_MULTI_ROW_COLS);

    if (n_waves == GFX12_SMALL_WAVES) {
        gemv_q4_0_f32_q8_inline_residual_gfx12_kernel<GFX12_SMALL_WAVES>
            <<<n_blocks_x, threads_per_block, shared_mem, stream>>>(
                weights_q4_0, input, residual, output, n_rows, ncols_dst);
    } else {
        gemv_q4_0_f32_q8_inline_residual_gfx12_kernel<GFX12_FIXED_WAVES>
            <<<n_blocks_x, threads_per_block, shared_mem, stream>>>(
                weights_q4_0, input, residual, output, n_rows, ncols_dst);
    }
    return hipGetLastError();
}

extern "C" hipError_t gemv_q4_0_f32_q8_inline_gfx12_launch(
    const void* weights_q4_0,
    const float* input,
    float* output,
    int n_rows,
    int ncols_dst,
    hipStream_t stream
) {
    if (n_rows <= 0 || ncols_dst <= 0 || input == nullptr || output == nullptr) {
        return hipErrorInvalidValue;
    }
    if (n_rows % QK4_0 != 0) return hipErrorInvalidValue;

    const int n_blocks_total = n_rows / QK4_0;
    const size_t shared_mem = n_blocks_total * Q8_0_GFX12_BLOCK_SIZE;
    if (shared_mem > GFX12_SHARED_MEM_LIMIT) return hipErrorInvalidValue;

    const int n_waves = select_gfx12_waves(n_blocks_total);
    const int threads_per_block = n_waves * WARP_SIZE;
    const int n_blocks_x =
        (ncols_dst + (n_waves * GFX12_MULTI_ROW_COLS) - 1) / (n_waves * GFX12_MULTI_ROW_COLS);

    if (n_waves == GFX12_SMALL_WAVES) {
        gemv_q4_0_f32_q8_inline_gfx12_kernel<GFX12_SMALL_WAVES>
            <<<n_blocks_x, threads_per_block, shared_mem, stream>>>(
                weights_q4_0, input, output, n_rows, ncols_dst);
    } else {
        gemv_q4_0_f32_q8_inline_gfx12_kernel<GFX12_FIXED_WAVES>
            <<<n_blocks_x, threads_per_block, shared_mem, stream>>>(
                weights_q4_0, input, output, n_rows, ncols_dst);
    }
    return hipGetLastError();
}
