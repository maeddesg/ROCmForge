#include "common.hip"

// ── Fused RMSNorm + Gate+Up+SiLU with udot4 (RDNA2+) ───────────────────────────
//
// Same fusion as q4_0_fused_norm_gate_up.hip but uses v_dot4_u32_u8
// for the Q4_0×Q8_0 inner loop. See q4_0_gemv_gfx12.hip for details
// on the unsigned dot-product with bias correction approach.

#define QK4_0 32
#define Q4_0_BLOCK_SIZE 18

struct Q4_0_block_ngud {
    half d;
    int8_t qs[16];
};

// Modified Q8_0 block: unsigned values + precomputed signed sum
struct Q8_0_block_ngud {
    half d;            // 2 bytes
    int16_t sum_s;     // 2 bytes: sum of original signed values
    uint8_t qs_u[QK4_0]; // 32 bytes: unsigned values (signed + 128)
};

constexpr int Q8_0_NGUD_BLOCK_SIZE = sizeof(Q8_0_block_ngud); // 36 bytes

__device__ inline float silu_ngud(float x) {
    return x / (1.0f + expf(-x));
}

__device__ __forceinline__ int udot4_u8_ngud(unsigned int a, unsigned int b, int c) {
    int result;
    asm volatile("v_dot4_u32_u8 %0, %1, %2, %3"
                 : "=v"(result) : "v"(a), "v"(b), "v"(c));
    return result;
}

// udot4-based Q4_0 × Q8_0 block dot product with bias correction
__device__ __forceinline__ int q4_0_q8_0_block_dot_udot4(
    const Q4_0_block_ngud* __restrict__ w_block,
    const Q8_0_block_ngud* __restrict__ x_block
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

        unsigned_dot = udot4_u8_ngud(lo, x_lo_words[i], unsigned_dot);
        unsigned_dot = udot4_u8_ngud(hi, x_hi_words[i], unsigned_dot);

        // Nibble sum: lo+hi per byte safe (max 30 < 256)
        q4_nibble_sum = udot4_u8_ngud(lo + hi, ONES, q4_nibble_sum);
    }

    return unsigned_dot - 128 * q4_nibble_sum - 8 * (int)x_block->sum_s;
}

// ── Kernel ──────────────────────────────────────────────────────────────────────

template<int N_WAVES>
__global__ void gemv_norm_gate_up_swiglu_q4_0_f32_q8_udot4_kernel(
    const float* __restrict__ raw_hidden,
    const float* __restrict__ norm_weight,
    float eps,
    const void* __restrict__ w_gate,
    const void* __restrict__ w_up,
    float* __restrict__ out_swiglu,
    int n_rows,
    int n_ff
) {
    const int tid = threadIdx.x;
    const int wave_id = tid / 32;
    const int lane_id = tid % 32;
    const int ff_idx_base = (blockIdx.x * N_WAVES + wave_id) * 4;
    const int n_blocks_total = n_rows / QK4_0;

    // ── Phase 1: RMS Norm in shared memory ─────────────────────────────

    extern __shared__ char s_raw[];
    float* s_f32 = reinterpret_cast<float*>(s_raw);
    float* s_reduction = &s_f32[n_rows];

    float local_sum_sq = 0.0f;
    for (int i = tid; i < n_rows; i += blockDim.x) {
        float val = raw_hidden[i];
        s_f32[i] = val;
        local_sum_sq += val * val;
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        local_sum_sq += __shfl_down(local_sum_sq, offset);
    }

    if (lane_id == 0) {
        s_reduction[wave_id] = local_sum_sq;
    }
    __syncthreads();

    float final_sum_sq = (tid < (blockDim.x / 32)) ? s_reduction[tid] : 0.0f;
    if (wave_id == 0) {
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            final_sum_sq += __shfl_down(final_sum_sq, offset);
        }
        if (tid == 0) {
            s_reduction[0] = final_sum_sq;
        }
    }
    __syncthreads();

    float inv_rms = rsqrtf(s_reduction[0] / (float)n_rows + eps);

    for (int i = tid; i < n_rows; i += blockDim.x) {
        s_f32[i] = s_f32[i] * inv_rms * norm_weight[i];
    }
    __syncthreads();

    // ── Phase 2: Quantize to biased Q8_0 (unsigned + signed sum) ──────

    Q8_0_block_ngud* s_input_q8 = reinterpret_cast<Q8_0_block_ngud*>(s_raw);

    for (int block_idx = wave_id; block_idx < n_blocks_total; block_idx += N_WAVES) {
        const int row_offset = block_idx * QK4_0;
        const float val = s_f32[row_offset + lane_id];

        float max_val = fabsf(val);
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            max_val = fmaxf(max_val, __shfl_down(max_val, offset, 32));
        }

        const float scale = fmaxf(__shfl(max_val, 0, 32) / 127.0f, 1e-30f);
        const half scale_h = __float2half(scale);
        const float scale_f = __half2float(scale_h);

        const float inv_scale = 1.0f / scale_f;
        const float qf = fminf(fmaxf(val * inv_scale, -127.0f), 127.0f);
        const int8_t qs = static_cast<int8_t>(qf);

        // Store unsigned biased value
        s_input_q8[block_idx].qs_u[lane_id] = static_cast<uint8_t>(static_cast<int>(qs) + 128);

        // Compute signed block sum via warp reduction
        int lane_sum = static_cast<int>(qs);
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            lane_sum += __shfl_down(lane_sum, offset);
        }

        if (lane_id == 0) {
            s_input_q8[block_idx].d = scale_h;
            s_input_q8[block_idx].sum_s = static_cast<int16_t>(lane_sum);
        }
    }
    __syncthreads();

    // ── Phase 3: Gate+Up GEMV with udot4 + SiLU ──────────────────────

    if (ff_idx_base >= n_ff) return;

    const Q4_0_block_ngud* w_gate_cols[4];
    const Q4_0_block_ngud* w_up_cols[4];
    #pragma unroll
    for (int c = 0; c < 4; ++c) {
        const int ff_idx = ff_idx_base + c;
        w_gate_cols[c] = reinterpret_cast<const Q4_0_block_ngud*>(
            static_cast<const uint8_t*>(w_gate) + (ff_idx * n_blocks_total) * Q4_0_BLOCK_SIZE
        );
        w_up_cols[c] = reinterpret_cast<const Q4_0_block_ngud*>(
            static_cast<const uint8_t*>(w_up) + (ff_idx * n_blocks_total) * Q4_0_BLOCK_SIZE
        );
    }

    float sum_gate[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float sum_up[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    for (int block_idx = lane_id; block_idx < n_blocks_total; block_idx += 32) {
        const Q8_0_block_ngud* x_block = &s_input_q8[block_idx];
        const float x_scale = __half2float(x_block->d);

        #pragma unroll
        for (int c = 0; c < 4; ++c) {
            const Q4_0_block_ngud* gate_block = &w_gate_cols[c][block_idx];
            const Q4_0_block_ngud* up_block = &w_up_cols[c][block_idx];

            const int gate_dot = q4_0_q8_0_block_dot_udot4(gate_block, x_block);
            const int up_dot = q4_0_q8_0_block_dot_udot4(up_block, x_block);

            sum_gate[c] += __half2float(gate_block->d) * x_scale * static_cast<float>(gate_dot);
            sum_up[c] += __half2float(up_block->d) * x_scale * static_cast<float>(up_dot);
        }
    }

    // Warp reduction
    #pragma unroll
    for (int c = 0; c < 4; ++c) {
        for (int offset = 16; offset > 0; offset >>= 1) {
            sum_gate[c] += __shfl_down(sum_gate[c], offset, 32);
            sum_up[c] += __shfl_down(sum_up[c], offset, 32);
        }
    }

    if (lane_id == 0) {
        #pragma unroll
        for (int c = 0; c < 4; ++c) {
            out_swiglu[ff_idx_base + c] = silu_ngud(sum_gate[c]) * sum_up[c];
        }
    }
}

// ── C Dispatch Function ─────────────────────────────────────────────────────────

extern "C" hipError_t gemv_norm_gate_up_swiglu_q4_0_f32_q8_udot4_launch(
    const float* raw_hidden,
    const float* norm_weight,
    float eps,
    const void* w_gate,
    const void* w_up,
    float* out_swiglu,
    int n_rows,
    int n_ff,
    hipStream_t stream
) {
    if (raw_hidden == nullptr || norm_weight == nullptr) return hipErrorInvalidValue;
    if (w_gate == nullptr || w_up == nullptr || out_swiglu == nullptr) return hipErrorInvalidValue;
    if (n_rows <= 0 || n_ff <= 0) return hipErrorInvalidValue;
    if (n_rows % QK4_0 != 0) return hipErrorInvalidValue;
    if ((n_ff % 4) != 0) return hipErrorInvalidValue;

    const int n_blocks_total = n_rows / QK4_0;
    const size_t norm_shared = (n_rows + 32) * sizeof(float);
    const size_t gemv_shared = n_blocks_total * Q8_0_NGUD_BLOCK_SIZE;
    const size_t shared_mem = (norm_shared > gemv_shared) ? norm_shared : gemv_shared;

    if (shared_mem > 32768) {
        return hipErrorInvalidValue;
    }

    if (n_rows <= 4096) {
        if (n_ff >= 4096) {
            const int N_WAVES = 8;
            const int n_blocks_x = (n_ff + (N_WAVES * 4) - 1) / (N_WAVES * 4);
            gemv_norm_gate_up_swiglu_q4_0_f32_q8_udot4_kernel<N_WAVES><<<n_blocks_x, 256, shared_mem, stream>>>(
                raw_hidden, norm_weight, eps,
                w_gate, w_up, out_swiglu, n_rows, n_ff
            );
        } else {
            const int N_WAVES = 4;
            const int n_blocks_x = (n_ff + (N_WAVES * 4) - 1) / (N_WAVES * 4);
            gemv_norm_gate_up_swiglu_q4_0_f32_q8_udot4_kernel<N_WAVES><<<n_blocks_x, 128, shared_mem, stream>>>(
                raw_hidden, norm_weight, eps,
                w_gate, w_up, out_swiglu, n_rows, n_ff
            );
        }
    } else {
        return hipErrorInvalidValue;
    }

    return hipGetLastError();
}
