#include "common.hip"

// Batched Q4_0 GEMV: Single Weight Load, Multiple Dot-Products
//
// For speculative decoding verify: N input vectors (2-8) share the same
// weight matrix. Instead of N separate GEMV launches that each stream
// the full weight matrix from VRAM, this kernel reads each weight block
// once and computes N dot products against N Q8-quantized input vectors
// cached in LDS.
//
// Memory savings: N× less weight bandwidth (the dominant cost at batch-1).
// LDS cost: N × (n_rows/32) × 34 bytes. For 7B (n_rows=3584, N=5): ~19 KB.

#define QK4_0 32
#define Q4_0_BLOCK_SIZE 18

struct Q4_0_block {
    half d;
    int8_t qs[16];
};

struct Q8_0_block_batched {
    half d;
    int8_t qs[QK4_0];
};

constexpr int BATCHED_WAVES = 8;
constexpr int BATCHED_THREADS = BATCHED_WAVES * WARP_SIZE;
constexpr int BATCHED_COLS = 4;
constexpr int BATCHED_LDS_LIMIT = 32 * 1024;

__device__ __forceinline__ int q4_0_q8_0_dot(
    const Q4_0_block* __restrict__ w_block,
    const int8_t* __restrict__ x_qs
) {
    int sum = 0;
    const uint32_t* packed = reinterpret_cast<const uint32_t*>(w_block->qs);

    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        const uint32_t pw = packed[i];
        const int base = i * 4;

        const uint8_t b0 = static_cast<uint8_t>(pw & 0xFF);
        const uint8_t b1 = static_cast<uint8_t>((pw >> 8) & 0xFF);
        const uint8_t b2 = static_cast<uint8_t>((pw >> 16) & 0xFF);
        const uint8_t b3 = static_cast<uint8_t>((pw >> 24) & 0xFF);

        sum += (static_cast<int>(b0 & 0x0F) - 8) * static_cast<int>(x_qs[base + 0]);
        sum += (static_cast<int>(b1 & 0x0F) - 8) * static_cast<int>(x_qs[base + 1]);
        sum += (static_cast<int>(b2 & 0x0F) - 8) * static_cast<int>(x_qs[base + 2]);
        sum += (static_cast<int>(b3 & 0x0F) - 8) * static_cast<int>(x_qs[base + 3]);

        sum += (static_cast<int>(b0 >> 4) - 8) * static_cast<int>(x_qs[base + 16 + 0]);
        sum += (static_cast<int>(b1 >> 4) - 8) * static_cast<int>(x_qs[base + 16 + 1]);
        sum += (static_cast<int>(b2 >> 4) - 8) * static_cast<int>(x_qs[base + 16 + 2]);
        sum += (static_cast<int>(b3 >> 4) - 8) * static_cast<int>(x_qs[base + 16 + 3]);
    }
    return sum;
}

// ── Batched GEMV Kernel ──────────────────────────────────────────────────────────
//
// Template N_BATCH: number of input vectors (compile-time for register allocation).
// Each wave processes BATCHED_COLS output columns for ALL N_BATCH input vectors.
//
// LDS layout: N_BATCH consecutive Q8 input arrays, each n_blocks_total Q8_0 blocks.
//   s_inputs[batch * n_blocks_total + block_idx]
//
// Phase 1: Cooperatively quantize all N_BATCH f32 inputs to Q8 in LDS.
// Phase 2: For each weight block, compute N_BATCH × BATCHED_COLS dot products.
// Phase 3: Warp reduction + write N_BATCH × ncols_dst outputs.

template<int N_BATCH>
__launch_bounds__(BATCHED_THREADS, 1)
__global__ void gemv_q4_0_f32_batched_kernel(
    const void* __restrict__ weights_q4_0,
    const float* __restrict__ input,       // [N_BATCH × n_rows], row-major
    float* __restrict__ output,             // [N_BATCH × ncols_dst], row-major
    int n_rows,
    int ncols_dst,
    int batch_size                          // runtime batch size (≤ N_BATCH)
) {
    const int tid = threadIdx.x;
    const int wave_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;
    const int col_base = (blockIdx.x * BATCHED_WAVES + wave_id) * BATCHED_COLS;
    const int n_blocks_total = n_rows / QK4_0;

    if (col_base >= ncols_dst) return;

    // ── Phase 1: Quantize N_BATCH f32 inputs → Q8 in LDS ─────────────────
    extern __shared__ Q8_0_block_batched s_inputs[];

    for (int b = 0; b < batch_size; ++b) {
        Q8_0_block_batched* s_batch = &s_inputs[b * n_blocks_total];
        const float* batch_input = &input[b * n_rows];

        for (int block_idx = wave_id; block_idx < n_blocks_total; block_idx += BATCHED_WAVES) {
            const int row_offset = block_idx * QK4_0;
            const float val = batch_input[row_offset + lane_id];
            float max_val = fabsf(val);

            #pragma unroll
            for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
                max_val = fmaxf(max_val, __shfl_down(max_val, offset));
            }

            const float scale = fmaxf(__shfl(max_val, 0, WARP_SIZE) / 127.0f, 1e-30f);
            const half scale_h = __float2half(scale);
            const float scale_f = __half2float(scale_h);
            if (lane_id == 0) {
                s_batch[block_idx].d = scale_h;
            }

            const float inv_scale = 1.0f / scale_f;
            const float qf = fminf(fmaxf(val * inv_scale, -127.0f), 127.0f);
            s_batch[block_idx].qs[lane_id] = static_cast<int8_t>(qf);
        }
    }
    __syncthreads();

    // ── Phase 2: Batched Q4×Q8 GEMV ──────────────────────────────────────
    const Q4_0_block* w_cols[BATCHED_COLS];
    #pragma unroll
    for (int c = 0; c < BATCHED_COLS; ++c) {
        int col = col_base + c;
        w_cols[c] = (col < ncols_dst) ? reinterpret_cast<const Q4_0_block*>(
            static_cast<const uint8_t*>(weights_q4_0) + (col * n_blocks_total) * Q4_0_BLOCK_SIZE
        ) : nullptr;
    }

    float sums[N_BATCH][BATCHED_COLS];
    #pragma unroll
    for (int b = 0; b < N_BATCH; ++b) {
        #pragma unroll
        for (int c = 0; c < BATCHED_COLS; ++c) {
            sums[b][c] = 0.0f;
        }
    }

    for (int block_idx = lane_id; block_idx < n_blocks_total; block_idx += WARP_SIZE) {
        // Load weight data once for all batch items
        #pragma unroll
        for (int c = 0; c < BATCHED_COLS; ++c) {
            if (!w_cols[c]) continue;

            const Q4_0_block* w_block = &w_cols[c][block_idx];
            const float w_scale = __half2float(w_block->d);

            // Compute dot product against each batch's Q8 input
            #pragma unroll
            for (int b = 0; b < N_BATCH; ++b) {
                if (b >= batch_size) break;

                const Q8_0_block_batched* x_block = &s_inputs[b * n_blocks_total + block_idx];
                const float combined_scale = w_scale * __half2float(x_block->d);
                const int dot = q4_0_q8_0_dot(w_block, x_block->qs);
                sums[b][c] += combined_scale * static_cast<float>(dot);
            }
        }
    }

    // ── Phase 3: Warp reduction + output ─────────────────────────────────
    #pragma unroll
    for (int b = 0; b < N_BATCH; ++b) {
        #pragma unroll
        for (int c = 0; c < BATCHED_COLS; ++c) {
            for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
                sums[b][c] += __shfl_down(sums[b][c], offset);
            }
        }
    }

    if (lane_id == 0) {
        #pragma unroll
        for (int b = 0; b < N_BATCH; ++b) {
            if (b >= batch_size) break;
            #pragma unroll
            for (int c = 0; c < BATCHED_COLS; ++c) {
                if (col_base + c < ncols_dst) {
                    output[b * ncols_dst + col_base + c] = sums[b][c];
                }
            }
        }
    }
}

// ── Launch Functions ─────────────────────────────────────────────────────────────

template<int N_BATCH>
static hipError_t launch_batched(
    const void* weights_q4_0,
    const float* input,
    float* output,
    int n_rows,
    int ncols_dst,
    int batch_size,
    hipStream_t stream
) {
    const int n_blocks_total = n_rows / QK4_0;
    const size_t shared_mem = batch_size * n_blocks_total * sizeof(Q8_0_block_batched);

    if (shared_mem > BATCHED_LDS_LIMIT) {
        return hipErrorInvalidValue;
    }

    const int n_blocks_x =
        (ncols_dst + (BATCHED_WAVES * BATCHED_COLS) - 1) / (BATCHED_WAVES * BATCHED_COLS);

    gemv_q4_0_f32_batched_kernel<N_BATCH><<<n_blocks_x, BATCHED_THREADS, shared_mem, stream>>>(
        weights_q4_0, input, output, n_rows, ncols_dst, batch_size
    );

    return hipGetLastError();
}

extern "C" hipError_t gemv_q4_0_f32_batched_launch(
    const void* weights_q4_0,
    const float* input,
    float* output,
    int n_rows,
    int ncols_dst,
    int batch_size,
    hipStream_t stream
) {
    if (n_rows <= 0 || ncols_dst <= 0 || batch_size <= 0 || batch_size > 8) {
        return hipErrorInvalidValue;
    }
    if (n_rows % QK4_0 != 0) {
        return hipErrorInvalidValue;
    }

    // Select template specialization matching batch size
    switch (batch_size) {
        case 1: return launch_batched<1>(weights_q4_0, input, output, n_rows, ncols_dst, batch_size, stream);
        case 2: return launch_batched<2>(weights_q4_0, input, output, n_rows, ncols_dst, batch_size, stream);
        case 3: return launch_batched<3>(weights_q4_0, input, output, n_rows, ncols_dst, batch_size, stream);
        case 4: return launch_batched<4>(weights_q4_0, input, output, n_rows, ncols_dst, batch_size, stream);
        case 5: return launch_batched<5>(weights_q4_0, input, output, n_rows, ncols_dst, batch_size, stream);
        case 6: return launch_batched<6>(weights_q4_0, input, output, n_rows, ncols_dst, batch_size, stream);
        case 7: return launch_batched<7>(weights_q4_0, input, output, n_rows, ncols_dst, batch_size, stream);
        case 8: return launch_batched<8>(weights_q4_0, input, output, n_rows, ncols_dst, batch_size, stream);
        default: return hipErrorInvalidValue;
    }
}
