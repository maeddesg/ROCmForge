#include "common.hip"

// Tiled Batched Q4_0 GEMV: Single Weight Load, Tiled Input Quantization
//
// For speculative decoding verify when the full Q8 input doesn't fit in LDS
// (e.g., FFN down-projection: in_dim=18944 at batch=2 needs 40 KB > 32 KB limit).
//
// Strategy: Tile along the input/reduction dimension (TILE_K elements per tile).
// Each tile:
//   1. Cooperatively quantize batch_size × TILE_K f32 inputs → Q8 in LDS
//   2. Stream weight blocks for this tile from VRAM, compute partial dot products
//   3. Accumulate in FP32 registers across tiles
//
// Weight bandwidth is identical to the non-tiled kernel (one load per block).
// The only overhead is multiple quantization passes and tile barriers.
//
// See docs/tiled_gemv_design.md for full design rationale.

#define QK4_0 32
#define Q4_0_BLOCK_SIZE 18

// Tile size: 1024 input elements = 32 Q8 blocks of 32 each.
// LDS per tile at batch=8: 8 × 32 × 34 = 8,704 bytes — well under 16 KB.
constexpr int TILE_K = 1024;
constexpr int TILE_BLOCKS = TILE_K / QK4_0;  // 32

constexpr int TILED_WAVES = 8;
constexpr int TILED_THREADS = TILED_WAVES * WARP_SIZE;  // 256
constexpr int TILED_COLS = 4;  // output columns per wave

struct Q4_0_block_t {
    half d;
    int8_t qs[16];
};

struct Q8_0_block_tiled {
    half d;
    int8_t qs[QK4_0];
};

__device__ __forceinline__ int q4_0_q8_0_dot_tiled(
    const Q4_0_block_t* __restrict__ w_block,
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

// ── Tiled Batched GEMV Kernel ────────────────────────────────────────────────────
//
// Template N_BATCH: compile-time max batch size for register allocation.
// Each wave handles TILED_COLS output columns for ALL batch items.
// Outer loop tiles along in_dim in chunks of TILE_K.
//
// LDS layout per tile: N_BATCH × TILE_BLOCKS consecutive Q8 blocks.
//   s_inputs[batch * TILE_BLOCKS + local_block_idx]
//
// Phase 1 (per tile): Cooperatively quantize batch_size f32 inputs → Q8 in LDS.
// Phase 2 (per tile): For each weight block in tile, compute N_BATCH × TILED_COLS dots.
// Phase 3 (once): Warp reduction + write outputs.

template<int N_BATCH>
__launch_bounds__(TILED_THREADS, 2)  // Target 2 resident workgroups per CU
__global__ void gemv_q4_0_f32_batched_tiled_kernel(
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
    const int col_base = (blockIdx.x * TILED_WAVES + wave_id) * TILED_COLS;
    const int n_blocks_total = n_rows / QK4_0;
    const int num_tiles = (n_blocks_total + TILE_BLOCKS - 1) / TILE_BLOCKS;

    if (col_base >= ncols_dst) return;

    // Weight column pointers (constant across tiles)
    const Q4_0_block_t* w_cols[TILED_COLS];
    #pragma unroll
    for (int c = 0; c < TILED_COLS; ++c) {
        int col = col_base + c;
        w_cols[c] = (col < ncols_dst) ? reinterpret_cast<const Q4_0_block_t*>(
            static_cast<const uint8_t*>(weights_q4_0) + (col * n_blocks_total) * Q4_0_BLOCK_SIZE
        ) : nullptr;
    }

    // FP32 accumulators in registers — persist across all tiles
    float sums[N_BATCH][TILED_COLS];
    #pragma unroll
    for (int b = 0; b < N_BATCH; ++b) {
        #pragma unroll
        for (int c = 0; c < TILED_COLS; ++c) {
            sums[b][c] = 0.0f;
        }
    }

    // LDS for one tile of Q8-quantized input
    extern __shared__ Q8_0_block_tiled s_inputs[];

    for (int tile = 0; tile < num_tiles; ++tile) {
        const int tile_block_start = tile * TILE_BLOCKS;
        const int tile_blocks_this = min(TILE_BLOCKS, n_blocks_total - tile_block_start);

        // ── Phase 1: Quantize batch_size f32 inputs → Q8 in LDS for this tile ──
        for (int b = 0; b < batch_size; ++b) {
            Q8_0_block_tiled* s_batch = &s_inputs[b * TILE_BLOCKS];
            const float* batch_input = &input[b * n_rows];

            for (int local_blk = wave_id; local_blk < tile_blocks_this; local_blk += TILED_WAVES) {
                const int global_blk = tile_block_start + local_blk;
                const int row_offset = global_blk * QK4_0;

                // Each lane loads one f32 value (warp covers 32 elements = 1 Q8 block)
                const float val = (row_offset + lane_id < n_rows) ?
                    batch_input[row_offset + lane_id] : 0.0f;

                // Warp-wide absmax reduction
                float max_val = fabsf(val);
                #pragma unroll
                for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
                    max_val = fmaxf(max_val, __shfl_down(max_val, offset));
                }

                const float scale = fmaxf(__shfl(max_val, 0, WARP_SIZE) / 127.0f, 1e-30f);
                const half scale_h = __float2half(scale);
                const float scale_f = __half2float(scale_h);
                if (lane_id == 0) {
                    s_batch[local_blk].d = scale_h;
                }

                const float inv_scale = 1.0f / scale_f;
                const float qf = fminf(fmaxf(val * inv_scale, -127.0f), 127.0f);
                s_batch[local_blk].qs[lane_id] = static_cast<int8_t>(qf);
            }
        }
        __syncthreads();

        // ── Phase 2: Compute Q4×Q8 dot products for this tile ───────────────
        for (int local_blk = lane_id; local_blk < tile_blocks_this; local_blk += WARP_SIZE) {
            const int global_blk = tile_block_start + local_blk;

            #pragma unroll
            for (int c = 0; c < TILED_COLS; ++c) {
                if (!w_cols[c]) continue;

                const Q4_0_block_t* w_block = &w_cols[c][global_blk];
                const float w_scale = __half2float(w_block->d);

                #pragma unroll
                for (int b = 0; b < N_BATCH; ++b) {
                    if (b >= batch_size) break;

                    const Q8_0_block_tiled* x_block = &s_inputs[b * TILE_BLOCKS + local_blk];
                    const float combined_scale = w_scale * __half2float(x_block->d);
                    const int dot = q4_0_q8_0_dot_tiled(w_block, x_block->qs);
                    sums[b][c] += combined_scale * static_cast<float>(dot);
                }
            }
        }

        // Barrier before next tile overwrites LDS
        __syncthreads();
    }

    // ── Phase 3: Warp reduction + output ─────────────────────────────────────
    #pragma unroll
    for (int b = 0; b < N_BATCH; ++b) {
        #pragma unroll
        for (int c = 0; c < TILED_COLS; ++c) {
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
            for (int c = 0; c < TILED_COLS; ++c) {
                if (col_base + c < ncols_dst) {
                    output[b * ncols_dst + col_base + c] = sums[b][c];
                }
            }
        }
    }
}

// ── Launch Functions ─────────────────────────────────────────────────────────────

template<int N_BATCH>
static hipError_t launch_tiled(
    const void* weights_q4_0,
    const float* input,
    float* output,
    int n_rows,
    int ncols_dst,
    int batch_size,
    hipStream_t stream
) {
    // LDS for one tile: batch_size × TILE_BLOCKS × sizeof(Q8_0_block_tiled)
    const size_t shared_mem = batch_size * TILE_BLOCKS * sizeof(Q8_0_block_tiled);

    const int n_blocks_x =
        (ncols_dst + (TILED_WAVES * TILED_COLS) - 1) / (TILED_WAVES * TILED_COLS);

    gemv_q4_0_f32_batched_tiled_kernel<N_BATCH><<<n_blocks_x, TILED_THREADS, shared_mem, stream>>>(
        weights_q4_0, input, output, n_rows, ncols_dst, batch_size
    );

    return hipGetLastError();
}

extern "C" hipError_t gemv_q4_0_f32_batched_tiled_launch(
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

    switch (batch_size) {
        case 1: return launch_tiled<1>(weights_q4_0, input, output, n_rows, ncols_dst, batch_size, stream);
        case 2: return launch_tiled<2>(weights_q4_0, input, output, n_rows, ncols_dst, batch_size, stream);
        case 3: return launch_tiled<3>(weights_q4_0, input, output, n_rows, ncols_dst, batch_size, stream);
        case 4: return launch_tiled<4>(weights_q4_0, input, output, n_rows, ncols_dst, batch_size, stream);
        case 5: return launch_tiled<5>(weights_q4_0, input, output, n_rows, ncols_dst, batch_size, stream);
        case 6: return launch_tiled<6>(weights_q4_0, input, output, n_rows, ncols_dst, batch_size, stream);
        case 7: return launch_tiled<7>(weights_q4_0, input, output, n_rows, ncols_dst, batch_size, stream);
        case 8: return launch_tiled<8>(weights_q4_0, input, output, n_rows, ncols_dst, batch_size, stream);
        default: return hipErrorInvalidValue;
    }
}
