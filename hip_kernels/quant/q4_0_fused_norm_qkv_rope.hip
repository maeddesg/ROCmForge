#include "common.hip"

// Q4_0 constants
#define QK4_0 32
#define Q4_0_BLOCK_SIZE 18

struct Q4_0_block_nqr {
    half d;
    int8_t qs[16];
};

// ── Fused RMSNorm + QKV + RoPE(Q) + RoPE(K)+KV-Write Kernel ──────────────────
//
// Replaces 4 separate kernel launches per layer:
//   1. rms_norm (attn_norm)
//   2. gemv_qkv (Q4_0 fused QKV projection)
//   3. rope_heads_state (RoPE on Q)
//   4. kv_write_rope_state (RoPE on K + write K,V to cache)
//
// The kernel:
//   1. Loads raw hidden state into shared memory
//   2. Computes RMS norm in shared memory (sum_sq → inv_rms → apply weight)
//   3. Runs GEMV on the normalized input
//   4. Applies RoPE in-register and writes Q/K/V outputs
//
// Shared memory layout:
//   [0 .. n_rows-1]           : f32 input (raw, then normalized)
//   [n_rows .. n_rows+31]     : f32 reduction scratch (32 floats)
//
// Requirements:
//   - n_q % 4 == 0 && n_kv % 4 == 0
//   - (n_rows + 32) * sizeof(float) <= 32768 (shared memory budget)
//   - head_dim must be even

template<int N_WAVES>
__global__ void gemv_norm_qkv_rope_kvwrite_q4_0_f32_kernel(
    // RMSNorm parameters
    const float* __restrict__ raw_hidden,
    const float* __restrict__ norm_weight,
    float eps,
    // QKV weights
    const void* __restrict__ w_q,
    const void* __restrict__ w_k,
    const void* __restrict__ w_v,
    const float* __restrict__ bias_q,
    const float* __restrict__ bias_k,
    const float* __restrict__ bias_v,
    // Q output (with RoPE applied)
    float* __restrict__ out_q,
    // KV cache pointers
    half* __restrict__ k_cache,
    half* __restrict__ v_cache,
    // Dimensions
    int n_rows,         // hidden_size (input dimension)
    int n_q,            // num_heads * head_dim
    int n_kv,           // num_kv_heads * head_dim
    // RoPE parameters
    const int* __restrict__ pos_ptr,    // device pointer to position
    int head_dim,
    float theta_base,
    int neox            // 0=consecutive pairs, 1=split-half (NeoX)
) {
    const int tid = threadIdx.x;
    const int wave_id = tid / 32;
    const int lane_id = tid % 32;
    const int col_base = (blockIdx.x * N_WAVES + wave_id) * 4;
    const int n_blocks_total = n_rows / QK4_0;
    const int total_cols = n_q + 2 * n_kv;
    const int kv_size = n_kv;

    // Shared memory: [n_rows floats for input] + [32 floats for reduction]
    extern __shared__ float s_data[];
    float* s_input = s_data;
    float* s_reduction = &s_data[n_rows];

    // ── Phase 1: Load raw hidden state + compute RMS norm ──────────────

    // Load raw hidden into shared memory and accumulate sum_sq
    float local_sum_sq = 0.0f;
    for (int i = tid; i < n_rows; i += blockDim.x) {
        float val = raw_hidden[i];
        s_input[i] = val;
        local_sum_sq += val * val;
    }

    // Warp-level reduction of sum_sq
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        local_sum_sq += __shfl_down(local_sum_sq, offset);
    }

    // Write per-warp results to shared reduction buffer
    if (lane_id == 0) {
        s_reduction[wave_id] = local_sum_sq;
    }
    __syncthreads();

    // First warp reduces across all warps
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

    // Compute inv_rms and apply normalization in-place in shared memory
    float inv_rms = rsqrtf(s_reduction[0] / (float)n_rows + eps);
    for (int i = tid; i < n_rows; i += blockDim.x) {
        s_input[i] = s_input[i] * inv_rms * norm_weight[i];
    }
    __syncthreads();

    // ── Phase 2: GEMV on normalized input (identical to non-norm version) ──

    if (col_base >= total_cols) return;

    // Determine which weight matrix this wave's columns belong to
    const void* weights_base;
    int out_col_base;
    int output_kind; // 0=Q, 1=K, 2=V
    if (col_base < n_q) {
        weights_base = w_q;
        out_col_base = col_base;
        output_kind = 0;
    } else if (col_base < n_q + n_kv) {
        weights_base = w_k;
        out_col_base = col_base - n_q;
        output_kind = 1;
    } else {
        weights_base = w_v;
        out_col_base = col_base - n_q - n_kv;
        output_kind = 2;
    }

    // Setup weight column pointers for 4 columns
    const Q4_0_block_nqr* w_cols[4];
    #pragma unroll
    for (int c = 0; c < 4; ++c) {
        w_cols[c] = reinterpret_cast<const Q4_0_block_nqr*>(
            static_cast<const uint8_t*>(weights_base) + ((out_col_base + c) * n_blocks_total) * Q4_0_BLOCK_SIZE
        );
    }

    // GEMV dot product
    float sums[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    for (int block_idx = lane_id; block_idx < n_blocks_total; block_idx += 32) {
        const int row_offset = block_idx * QK4_0;

        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            float4 in_l = reinterpret_cast<float4*>(&s_input[row_offset + 4 * i])[0];
            float4 in_h = reinterpret_cast<float4*>(&s_input[row_offset + 4 * i + 16])[0];

            #pragma unroll
            for (int c = 0; c < 4; ++c) {
                const Q4_0_block_nqr* b = &w_cols[c][block_idx];
                const float d = __half2float(b->d);
                const uint32_t q = reinterpret_cast<const uint32_t*>(b->qs)[i];

                sums[c] += d * (static_cast<float>( q        & 0x0F) - 8.0f) * in_l.x;
                sums[c] += d * (static_cast<float>((q >>  8) & 0x0F) - 8.0f) * in_l.y;
                sums[c] += d * (static_cast<float>((q >> 16) & 0x0F) - 8.0f) * in_l.z;
                sums[c] += d * (static_cast<float>((q >> 24) & 0x0F) - 8.0f) * in_l.w;
                sums[c] += d * (static_cast<float>((q >>  4) & 0x0F) - 8.0f) * in_h.x;
                sums[c] += d * (static_cast<float>((q >> 12) & 0x0F) - 8.0f) * in_h.y;
                sums[c] += d * (static_cast<float>((q >> 20) & 0x0F) - 8.0f) * in_h.z;
                sums[c] += d * (static_cast<float>((q >> 28) & 0x0F) - 8.0f) * in_h.w;
            }
        }
    }

    // Warp reduction
    #pragma unroll
    for (int c = 0; c < 4; ++c) {
        for (int offset = 16; offset > 0; offset >>= 1) {
            sums[c] += __shfl_down(sums[c], offset);
        }
    }

    // ── Phase 3: Post-GEMV — bias + RoPE + output/KV-write ────────────

    if (lane_id == 0) {
        const int pos = *pos_ptr;

        if (output_kind == 0) {
            // Q columns: add bias, apply RoPE, write to out_q
            #pragma unroll
            for (int c = 0; c < 4; ++c) {
                sums[c] += (bias_q ? bias_q[out_col_base + c] : 0.0f);
            }

            const int col_in_q = out_col_base;
            const int head = col_in_q / head_dim;
            const int idx_in_head = col_in_q - head * head_dim;
            const int half = head_dim / 2;

            #pragma unroll
            for (int p = 0; p < 2; ++p) {
                int pair_idx;
                if (neox) {
                    const int local_idx = idx_in_head + 2 * p;
                    pair_idx = (local_idx < half) ? local_idx : (local_idx - half);
                } else {
                    pair_idx = (idx_in_head + 2 * p) / 2;
                }

                const float exponent = (float)(2 * pair_idx) / (float)head_dim;
                const float theta = 1.0f / powf(theta_base, exponent);
                const float angle = (float)pos * theta;
                const float cos_val = cosf(angle);
                const float sin_val = sinf(angle);

                const float x0 = sums[2 * p];
                const float x1 = sums[2 * p + 1];

                if (neox) {
                    const int local_idx = idx_in_head + 2 * p;
                    if (local_idx < half) {
                        sums[2 * p]     = x0 * cos_val - x1 * sin_val;
                        sums[2 * p + 1] = x0 * sin_val + x1 * cos_val;
                    } else {
                        sums[2 * p]     = x0 * sin_val + x1 * cos_val;
                        sums[2 * p + 1] = x0 * cos_val - x1 * sin_val;
                    }
                } else {
                    sums[2 * p]     = x0 * cos_val - x1 * sin_val;
                    sums[2 * p + 1] = x0 * sin_val + x1 * cos_val;
                }
            }

            #pragma unroll
            for (int c = 0; c < 4; ++c) {
                out_q[out_col_base + c] = sums[c];
            }

        } else if (output_kind == 1) {
            // K columns: add bias, apply RoPE, write to k_cache
            #pragma unroll
            for (int c = 0; c < 4; ++c) {
                sums[c] += (bias_k ? bias_k[out_col_base + c] : 0.0f);
            }

            const int col_in_k = out_col_base;
            const int head = col_in_k / head_dim;
            const int idx_in_head = col_in_k - head * head_dim;
            const int half = head_dim / 2;

            #pragma unroll
            for (int p = 0; p < 2; ++p) {
                int pair_idx;
                if (neox) {
                    const int local_idx = idx_in_head + 2 * p;
                    pair_idx = (local_idx < half) ? local_idx : (local_idx - half);
                } else {
                    pair_idx = (idx_in_head + 2 * p) / 2;
                }

                const float exponent = (float)(2 * pair_idx) / (float)head_dim;
                const float theta = 1.0f / powf(theta_base, exponent);
                const float angle = (float)pos * theta;
                const float cos_val = cosf(angle);
                const float sin_val = sinf(angle);

                const float x0 = sums[2 * p];
                const float x1 = sums[2 * p + 1];

                if (neox) {
                    const int local_idx = idx_in_head + 2 * p;
                    if (local_idx < half) {
                        sums[2 * p]     = x0 * cos_val - x1 * sin_val;
                        sums[2 * p + 1] = x0 * sin_val + x1 * cos_val;
                    } else {
                        sums[2 * p]     = x0 * sin_val + x1 * cos_val;
                        sums[2 * p + 1] = x0 * cos_val - x1 * sin_val;
                    }
                } else {
                    sums[2 * p]     = x0 * cos_val - x1 * sin_val;
                    sums[2 * p + 1] = x0 * sin_val + x1 * cos_val;
                }
            }

            const size_t cache_base = (size_t)pos * kv_size;
            #pragma unroll
            for (int c = 0; c < 4; ++c) {
                k_cache[cache_base + out_col_base + c] = __float2half(sums[c]);
            }

        } else {
            // V columns: add bias, write directly to v_cache (no RoPE, f32 → f16)
            const size_t cache_base = (size_t)pos * kv_size;
            #pragma unroll
            for (int c = 0; c < 4; ++c) {
                const float val = sums[c] + (bias_v ? bias_v[out_col_base + c] : 0.0f);
                v_cache[cache_base + out_col_base + c] = __float2half(val);
            }
        }
    }
}

// ── C Dispatch Function ─────────────────────────────────────────────────────────────

extern "C" hipError_t gemv_norm_qkv_rope_kvwrite_q4_0_f32_launch(
    const float* raw_hidden,
    const float* norm_weight,
    float eps,
    const void* w_q, const void* w_k, const void* w_v,
    const float* bias_q, const float* bias_k, const float* bias_v,
    float* out_q,
    half* k_cache, half* v_cache,
    int n_rows, int n_q, int n_kv,
    const int* pos_ptr,
    int head_dim,
    float theta_base,
    int neox,
    hipStream_t stream
) {
    if (pos_ptr == nullptr) return hipErrorInvalidValue;
    if (raw_hidden == nullptr || norm_weight == nullptr) return hipErrorInvalidValue;
    if (n_rows <= 0 || n_q <= 0 || n_kv <= 0) return hipErrorInvalidValue;
    if ((n_q % 4) != 0 || (n_kv % 4) != 0) return hipErrorInvalidValue;
    if (head_dim <= 0 || (head_dim % 2) != 0) return hipErrorInvalidValue;

    const int N_WAVES = 8;
    const int total_cols = n_q + 2 * n_kv;
    // Shared memory: n_rows floats for input + 32 floats for reduction scratch
    const size_t shared_mem = (n_rows + 32) * sizeof(float);

    if (shared_mem <= 32768) {
        const int n_blocks_x = (total_cols + (N_WAVES * 4) - 1) / (N_WAVES * 4);
        gemv_norm_qkv_rope_kvwrite_q4_0_f32_kernel<N_WAVES><<<n_blocks_x, 256, shared_mem, stream>>>(
            raw_hidden, norm_weight, eps,
            w_q, w_k, w_v, bias_q, bias_k, bias_v,
            out_q, k_cache, v_cache,
            n_rows, n_q, n_kv,
            pos_ptr, head_dim, theta_base, neox
        );
    } else {
        return hipErrorInvalidValue;
    }

    return hipGetLastError();
}
