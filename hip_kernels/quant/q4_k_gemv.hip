// Phase 7 Step 4 — Q4_K super-block GEMV for decode (M = 1).
//
// Replaces an earlier stub that had a garbled dequant formula
// (`q / d + dmin`), wrong scale unpacking, and no super-block-aware
// nibble layout. The new kernel mirrors the multi-row pattern in
// `q4_0_gemv.hip`: 8 waves × 32 threads per block, each wave owns 4
// output columns, each lane iterates over super-blocks stepping by
// WARP_SIZE, wave-shuffle sum-reduction at the end. A direct-global
// variant handles the cases where the input vector exceeds the
// 32 KB LDS budget (Qwen3/Llama-3 FFN-down, K = 12,288 or 14,336).
//
// Ground truth for the format: `docs/q4_k_m_block_format.md`.
// Dequant: value = (d · scale_j) · nibble − (dmin · min_j)      (no /64)
// Scale unpack: get_scale_min_k4 from ggml-quants.c              (4:2 split)
// Nibble pairing: sub-blocks (2k, 2k+1) share qs[k*32 .. k*32+31]
//
// No WMMA. GEMV at M=1 is bandwidth-bound; matrix cores don't help.

#include "common.hip"
#include <hip/hip_fp16.h>

#define Q4_K_BLOCK_BYTES     144
#define Q4_K_ELEMS_PER_BLOCK 256
#define Q4_K_SUB_BLOCKS      8
#define Q4_K_SUB_BLOCK_ELEMS 32
#define Q4_K_MULTI_ROW_COLS  4
#define Q4_K_MULTI_ROW_SHARED_MEM_LIMIT 32768
#define Q4_K_FIXED_WAVES     8
#define Q4_K_THREADS_PER_BLOCK (Q4_K_FIXED_WAVES * WARP_SIZE)

// ── Inline helper: unpack 6-bit scale/min for sub-block j ──────────────
__device__ __forceinline__
void q4_k_unpack_scale_min(int j, const uint8_t* scales,
                           int* scale_out, int* min_out) {
    if (j < 4) {
        *scale_out = scales[j]     & 0x3F;
        *min_out   = scales[j + 4] & 0x3F;
    } else {
        *scale_out = (scales[j + 4] & 0x0F) | ((scales[j - 4] >> 6) << 4);
        *min_out   = (scales[j + 4] >>  4)  | ((scales[j]     >> 6) << 4);
    }
}

// ── Dequant + 256-element dot for one super-block against `input_base`──
__device__ __forceinline__
float q4_k_super_block_dot(const uint8_t* __restrict__ sb_ptr,
                           const float*   __restrict__ input_base) {
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
        q4_k_unpack_scale_min(j, scales, &scale_j, &min_j);
        const float d_scale = d_f    * (float)scale_j;
        const float d_min   = dmin_f * (float)min_j;

        const int pair_base = (j >> 1) * 32;
        const bool is_upper = (j & 1) != 0;
        const float* in_ptr = input_base + j * Q4_K_SUB_BLOCK_ELEMS;

        #pragma unroll
        for (int i = 0; i < Q4_K_SUB_BLOCK_ELEMS; ++i) {
            const uint8_t byte = qs[pair_base + i];
            const int nib = is_upper ? (byte >> 4) : (byte & 0x0F);
            acc += (d_scale * (float)nib - d_min) * in_ptr[i];
        }
    }
    return acc;
}

// ── Multi-row GEMV with LDS input cache (fast path, K × 4 ≤ 32 KB) ─────

template<int N_WAVES>
__launch_bounds__(Q4_K_THREADS_PER_BLOCK, 1)
__global__ void gemv_q4_k_f32_multi_row_kernel(
    const void* __restrict__ weights_q4_k,
    const float* __restrict__ input,
    float* __restrict__ output,
    int n_rows,
    int ncols_dst
) {
    const int tid = threadIdx.x;
    const int wave_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;

    const int col_base = (blockIdx.x * N_WAVES + wave_id) * Q4_K_MULTI_ROW_COLS;
    const int super_blocks_per_row = n_rows / Q4_K_ELEMS_PER_BLOCK;
    if (col_base >= ncols_dst) return;

    extern __shared__ float s_input[];
    for (int i = tid; i < n_rows; i += blockDim.x) {
        s_input[i] = input[i];
    }
    __syncthreads();

    const uint8_t* w_cols[Q4_K_MULTI_ROW_COLS];
    #pragma unroll
    for (int c = 0; c < Q4_K_MULTI_ROW_COLS; ++c) {
        const int col = col_base + c;
        w_cols[c] = (col < ncols_dst)
            ? static_cast<const uint8_t*>(weights_q4_k)
              + (size_t)col * (size_t)super_blocks_per_row * Q4_K_BLOCK_BYTES
            : nullptr;
    }

    float sums[Q4_K_MULTI_ROW_COLS] = {0.0f, 0.0f, 0.0f, 0.0f};

    for (int sb = lane_id; sb < super_blocks_per_row; sb += WARP_SIZE) {
        const float* in_base = s_input + sb * Q4_K_ELEMS_PER_BLOCK;
        #pragma unroll
        for (int c = 0; c < Q4_K_MULTI_ROW_COLS; ++c) {
            if (w_cols[c]) {
                const uint8_t* sb_ptr = w_cols[c] + (size_t)sb * Q4_K_BLOCK_BYTES;
                sums[c] += q4_k_super_block_dot(sb_ptr, in_base);
            }
        }
    }

    #pragma unroll
    for (int c = 0; c < Q4_K_MULTI_ROW_COLS; ++c) {
        for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
            sums[c] += __shfl_down(sums[c], offset);
        }
    }

    if (lane_id == 0) {
        #pragma unroll
        for (int c = 0; c < Q4_K_MULTI_ROW_COLS; ++c) {
            if (col_base + c < ncols_dst) {
                output[col_base + c] = sums[c];
            }
        }
    }
}

template<int N_WAVES>
__launch_bounds__(Q4_K_THREADS_PER_BLOCK, 1)
__global__ void gemv_q4_k_f32_residual_multi_row_kernel(
    const void* __restrict__ weights_q4_k,
    const float* __restrict__ input,
    const float* __restrict__ residual,
    float* __restrict__ output,
    int n_rows,
    int ncols_dst
) {
    const int tid = threadIdx.x;
    const int wave_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;

    const int col_base = (blockIdx.x * N_WAVES + wave_id) * Q4_K_MULTI_ROW_COLS;
    const int super_blocks_per_row = n_rows / Q4_K_ELEMS_PER_BLOCK;
    if (col_base >= ncols_dst) return;

    extern __shared__ float s_input[];
    for (int i = tid; i < n_rows; i += blockDim.x) {
        s_input[i] = input[i];
    }
    __syncthreads();

    const uint8_t* w_cols[Q4_K_MULTI_ROW_COLS];
    #pragma unroll
    for (int c = 0; c < Q4_K_MULTI_ROW_COLS; ++c) {
        const int col = col_base + c;
        w_cols[c] = (col < ncols_dst)
            ? static_cast<const uint8_t*>(weights_q4_k)
              + (size_t)col * (size_t)super_blocks_per_row * Q4_K_BLOCK_BYTES
            : nullptr;
    }

    float sums[Q4_K_MULTI_ROW_COLS] = {0.0f, 0.0f, 0.0f, 0.0f};

    for (int sb = lane_id; sb < super_blocks_per_row; sb += WARP_SIZE) {
        const float* in_base = s_input + sb * Q4_K_ELEMS_PER_BLOCK;
        #pragma unroll
        for (int c = 0; c < Q4_K_MULTI_ROW_COLS; ++c) {
            if (w_cols[c]) {
                const uint8_t* sb_ptr = w_cols[c] + (size_t)sb * Q4_K_BLOCK_BYTES;
                sums[c] += q4_k_super_block_dot(sb_ptr, in_base);
            }
        }
    }

    #pragma unroll
    for (int c = 0; c < Q4_K_MULTI_ROW_COLS; ++c) {
        for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
            sums[c] += __shfl_down(sums[c], offset);
        }
    }

    if (lane_id == 0) {
        #pragma unroll
        for (int c = 0; c < Q4_K_MULTI_ROW_COLS; ++c) {
            const int col = col_base + c;
            if (col < ncols_dst) {
                output[col] = sums[c] + residual[col];
            }
        }
    }
}

// ── Direct-global GEMV (fallback for K > LDS budget) ───────────────────
// Qwen3-8B / Llama-3.1-8B FFN-down (K = 12,288 / 14,336) take this path.

template<int N_WAVES>
__launch_bounds__(Q4_K_THREADS_PER_BLOCK, 1)
__global__ void gemv_q4_k_f32_direct_kernel(
    const void* __restrict__ weights_q4_k,
    const float* __restrict__ input,
    float* __restrict__ output,
    int n_rows,
    int ncols_dst
) {
    const int tid = threadIdx.x;
    const int wave_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;

    const int col_base = (blockIdx.x * N_WAVES + wave_id) * Q4_K_MULTI_ROW_COLS;
    const int super_blocks_per_row = n_rows / Q4_K_ELEMS_PER_BLOCK;
    if (col_base >= ncols_dst) return;

    const uint8_t* w_cols[Q4_K_MULTI_ROW_COLS];
    #pragma unroll
    for (int c = 0; c < Q4_K_MULTI_ROW_COLS; ++c) {
        const int col = col_base + c;
        w_cols[c] = (col < ncols_dst)
            ? static_cast<const uint8_t*>(weights_q4_k)
              + (size_t)col * (size_t)super_blocks_per_row * Q4_K_BLOCK_BYTES
            : nullptr;
    }

    float sums[Q4_K_MULTI_ROW_COLS] = {0.0f, 0.0f, 0.0f, 0.0f};

    for (int sb = lane_id; sb < super_blocks_per_row; sb += WARP_SIZE) {
        const float* in_base = input + sb * Q4_K_ELEMS_PER_BLOCK;
        #pragma unroll
        for (int c = 0; c < Q4_K_MULTI_ROW_COLS; ++c) {
            if (w_cols[c]) {
                const uint8_t* sb_ptr = w_cols[c] + (size_t)sb * Q4_K_BLOCK_BYTES;
                sums[c] += q4_k_super_block_dot(sb_ptr, in_base);
            }
        }
    }

    #pragma unroll
    for (int c = 0; c < Q4_K_MULTI_ROW_COLS; ++c) {
        for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
            sums[c] += __shfl_down(sums[c], offset);
        }
    }

    if (lane_id == 0) {
        #pragma unroll
        for (int c = 0; c < Q4_K_MULTI_ROW_COLS; ++c) {
            if (col_base + c < ncols_dst) {
                output[col_base + c] = sums[c];
            }
        }
    }
}

template<int N_WAVES>
__launch_bounds__(Q4_K_THREADS_PER_BLOCK, 1)
__global__ void gemv_q4_k_f32_residual_direct_kernel(
    const void* __restrict__ weights_q4_k,
    const float* __restrict__ input,
    const float* __restrict__ residual,
    float* __restrict__ output,
    int n_rows,
    int ncols_dst
) {
    const int tid = threadIdx.x;
    const int wave_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;

    const int col_base = (blockIdx.x * N_WAVES + wave_id) * Q4_K_MULTI_ROW_COLS;
    const int super_blocks_per_row = n_rows / Q4_K_ELEMS_PER_BLOCK;
    if (col_base >= ncols_dst) return;

    const uint8_t* w_cols[Q4_K_MULTI_ROW_COLS];
    #pragma unroll
    for (int c = 0; c < Q4_K_MULTI_ROW_COLS; ++c) {
        const int col = col_base + c;
        w_cols[c] = (col < ncols_dst)
            ? static_cast<const uint8_t*>(weights_q4_k)
              + (size_t)col * (size_t)super_blocks_per_row * Q4_K_BLOCK_BYTES
            : nullptr;
    }

    float sums[Q4_K_MULTI_ROW_COLS] = {0.0f, 0.0f, 0.0f, 0.0f};

    for (int sb = lane_id; sb < super_blocks_per_row; sb += WARP_SIZE) {
        const float* in_base = input + sb * Q4_K_ELEMS_PER_BLOCK;
        #pragma unroll
        for (int c = 0; c < Q4_K_MULTI_ROW_COLS; ++c) {
            if (w_cols[c]) {
                const uint8_t* sb_ptr = w_cols[c] + (size_t)sb * Q4_K_BLOCK_BYTES;
                sums[c] += q4_k_super_block_dot(sb_ptr, in_base);
            }
        }
    }

    #pragma unroll
    for (int c = 0; c < Q4_K_MULTI_ROW_COLS; ++c) {
        for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
            sums[c] += __shfl_down(sums[c], offset);
        }
    }

    if (lane_id == 0) {
        #pragma unroll
        for (int c = 0; c < Q4_K_MULTI_ROW_COLS; ++c) {
            const int col = col_base + c;
            if (col < ncols_dst) {
                output[col] = sums[c] + residual[col];
            }
        }
    }
}

// ── Fused Gate/Up + SwiGLU (Phase 7 Step 5d) ───────────────────────────
// Replaces four dispatches (gate-GEMV, up-GEMV, SiLU, mul) with one.
// For each block of 32 output columns (8 waves × 4 cols/wave) the kernel
// computes both gate[c] and up[c] from the shared input, then writes
// swiglu_out[c] = silu(gate[c]) * up[c]. The shared-memory input cache
// is loaded once and reused across both weight matrices.

__device__ __forceinline__ float silu_f32(float x) {
    // SiLU (swish) = x * sigmoid(x) = x / (1 + exp(-x))
    return x / (1.0f + expf(-x));
}

template<int N_WAVES>
__launch_bounds__(Q4_K_THREADS_PER_BLOCK, 1)
__global__ void gemv_gate_up_swiglu_q4_k_f32_multi_row_kernel(
    const void* __restrict__ weights_gate_q4_k,
    const void* __restrict__ weights_up_q4_k,
    const float* __restrict__ input,
    float* __restrict__ swiglu_out,
    int n_rows,
    int ncols_dst
) {
    const int tid = threadIdx.x;
    const int wave_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;

    const int col_base = (blockIdx.x * N_WAVES + wave_id) * Q4_K_MULTI_ROW_COLS;
    const int super_blocks_per_row = n_rows / Q4_K_ELEMS_PER_BLOCK;
    if (col_base >= ncols_dst) return;

    extern __shared__ float s_input[];
    for (int i = tid; i < n_rows; i += blockDim.x) {
        s_input[i] = input[i];
    }
    __syncthreads();

    const uint8_t* w_gate_cols[Q4_K_MULTI_ROW_COLS];
    const uint8_t* w_up_cols[Q4_K_MULTI_ROW_COLS];
    #pragma unroll
    for (int c = 0; c < Q4_K_MULTI_ROW_COLS; ++c) {
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

    float gate_sums[Q4_K_MULTI_ROW_COLS] = {0.0f, 0.0f, 0.0f, 0.0f};
    float up_sums[Q4_K_MULTI_ROW_COLS]   = {0.0f, 0.0f, 0.0f, 0.0f};

    for (int sb = lane_id; sb < super_blocks_per_row; sb += WARP_SIZE) {
        const float* in_base = s_input + sb * Q4_K_ELEMS_PER_BLOCK;
        #pragma unroll
        for (int c = 0; c < Q4_K_MULTI_ROW_COLS; ++c) {
            if (w_gate_cols[c]) {
                gate_sums[c] +=
                    q4_k_super_block_dot(w_gate_cols[c] + (size_t)sb * Q4_K_BLOCK_BYTES, in_base);
                up_sums[c] +=
                    q4_k_super_block_dot(w_up_cols[c] + (size_t)sb * Q4_K_BLOCK_BYTES, in_base);
            }
        }
    }

    #pragma unroll
    for (int c = 0; c < Q4_K_MULTI_ROW_COLS; ++c) {
        for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
            gate_sums[c] += __shfl_down(gate_sums[c], offset);
            up_sums[c]   += __shfl_down(up_sums[c], offset);
        }
    }

    if (lane_id == 0) {
        #pragma unroll
        for (int c = 0; c < Q4_K_MULTI_ROW_COLS; ++c) {
            const int col = col_base + c;
            if (col < ncols_dst) {
                swiglu_out[col] = silu_f32(gate_sums[c]) * up_sums[c];
            }
        }
    }
}

template<int N_WAVES>
__launch_bounds__(Q4_K_THREADS_PER_BLOCK, 1)
__global__ void gemv_gate_up_swiglu_q4_k_f32_direct_kernel(
    const void* __restrict__ weights_gate_q4_k,
    const void* __restrict__ weights_up_q4_k,
    const float* __restrict__ input,
    float* __restrict__ swiglu_out,
    int n_rows,
    int ncols_dst
) {
    const int tid = threadIdx.x;
    const int wave_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;

    const int col_base = (blockIdx.x * N_WAVES + wave_id) * Q4_K_MULTI_ROW_COLS;
    const int super_blocks_per_row = n_rows / Q4_K_ELEMS_PER_BLOCK;
    if (col_base >= ncols_dst) return;

    const uint8_t* w_gate_cols[Q4_K_MULTI_ROW_COLS];
    const uint8_t* w_up_cols[Q4_K_MULTI_ROW_COLS];
    #pragma unroll
    for (int c = 0; c < Q4_K_MULTI_ROW_COLS; ++c) {
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

    float gate_sums[Q4_K_MULTI_ROW_COLS] = {0.0f, 0.0f, 0.0f, 0.0f};
    float up_sums[Q4_K_MULTI_ROW_COLS]   = {0.0f, 0.0f, 0.0f, 0.0f};

    for (int sb = lane_id; sb < super_blocks_per_row; sb += WARP_SIZE) {
        const float* in_base = input + sb * Q4_K_ELEMS_PER_BLOCK;
        #pragma unroll
        for (int c = 0; c < Q4_K_MULTI_ROW_COLS; ++c) {
            if (w_gate_cols[c]) {
                gate_sums[c] +=
                    q4_k_super_block_dot(w_gate_cols[c] + (size_t)sb * Q4_K_BLOCK_BYTES, in_base);
                up_sums[c] +=
                    q4_k_super_block_dot(w_up_cols[c] + (size_t)sb * Q4_K_BLOCK_BYTES, in_base);
            }
        }
    }

    #pragma unroll
    for (int c = 0; c < Q4_K_MULTI_ROW_COLS; ++c) {
        for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
            gate_sums[c] += __shfl_down(gate_sums[c], offset);
            up_sums[c]   += __shfl_down(up_sums[c], offset);
        }
    }

    if (lane_id == 0) {
        #pragma unroll
        for (int c = 0; c < Q4_K_MULTI_ROW_COLS; ++c) {
            const int col = col_base + c;
            if (col < ncols_dst) {
                swiglu_out[col] = silu_f32(gate_sums[c]) * up_sums[c];
            }
        }
    }
}

extern "C" hipError_t gemv_gate_up_swiglu_q4_k_f32_launch(
    const void* weights_gate_q4_k,
    const void* weights_up_q4_k,
    const float* input,
    float* swiglu_out,
    int n_rows,
    int ncols_dst,
    hipStream_t stream
) {
    if (n_rows <= 0 || ncols_dst <= 0) return hipErrorInvalidValue;
    if ((n_rows % Q4_K_ELEMS_PER_BLOCK) != 0) return hipErrorInvalidValue;
    if (weights_gate_q4_k == nullptr || weights_up_q4_k == nullptr) {
        return hipErrorInvalidValue;
    }

    const int threads_per_block = Q4_K_THREADS_PER_BLOCK;
    const int n_blocks_x =
        (ncols_dst + (Q4_K_FIXED_WAVES * Q4_K_MULTI_ROW_COLS) - 1)
        / (Q4_K_FIXED_WAVES * Q4_K_MULTI_ROW_COLS);

    if ((size_t)n_rows * sizeof(float) <= Q4_K_MULTI_ROW_SHARED_MEM_LIMIT) {
        const size_t shared_mem = (size_t)n_rows * sizeof(float);
        gemv_gate_up_swiglu_q4_k_f32_multi_row_kernel<Q4_K_FIXED_WAVES>
            <<<n_blocks_x, threads_per_block, shared_mem, stream>>>(
                weights_gate_q4_k, weights_up_q4_k, input, swiglu_out, n_rows, ncols_dst
            );
    } else {
        gemv_gate_up_swiglu_q4_k_f32_direct_kernel<Q4_K_FIXED_WAVES>
            <<<n_blocks_x, threads_per_block, 0, stream>>>(
                weights_gate_q4_k, weights_up_q4_k, input, swiglu_out, n_rows, ncols_dst
            );
    }
    return hipGetLastError();
}

// ── Launchers — signatures match the prior stub for the base GEMV ──────

extern "C" hipError_t gemv_q4_k_f32_launch(
    const void* weights_q4_k,
    const float* input,
    float* output,
    int n_rows,
    int ncols_dst,
    hipStream_t stream
) {
    if (n_rows <= 0 || ncols_dst <= 0) return hipErrorInvalidValue;
    if ((n_rows % Q4_K_ELEMS_PER_BLOCK) != 0) return hipErrorInvalidValue;

    const int threads_per_block = Q4_K_THREADS_PER_BLOCK;
    const int n_blocks_x =
        (ncols_dst + (Q4_K_FIXED_WAVES * Q4_K_MULTI_ROW_COLS) - 1)
        / (Q4_K_FIXED_WAVES * Q4_K_MULTI_ROW_COLS);

    if ((size_t)n_rows * sizeof(float) <= Q4_K_MULTI_ROW_SHARED_MEM_LIMIT) {
        const size_t shared_mem = (size_t)n_rows * sizeof(float);
        gemv_q4_k_f32_multi_row_kernel<Q4_K_FIXED_WAVES>
            <<<n_blocks_x, threads_per_block, shared_mem, stream>>>(
                weights_q4_k, input, output, n_rows, ncols_dst
            );
    } else {
        gemv_q4_k_f32_direct_kernel<Q4_K_FIXED_WAVES>
            <<<n_blocks_x, threads_per_block, 0, stream>>>(
                weights_q4_k, input, output, n_rows, ncols_dst
            );
    }
    return hipGetLastError();
}

extern "C" hipError_t gemv_q4_k_f32_residual_launch(
    const void* weights_q4_k,
    const float* input,
    const float* residual,
    float* output,
    int n_rows,
    int ncols_dst,
    hipStream_t stream
) {
    if (n_rows <= 0 || ncols_dst <= 0 || residual == nullptr) return hipErrorInvalidValue;
    if ((n_rows % Q4_K_ELEMS_PER_BLOCK) != 0) return hipErrorInvalidValue;

    const int threads_per_block = Q4_K_THREADS_PER_BLOCK;
    const int n_blocks_x =
        (ncols_dst + (Q4_K_FIXED_WAVES * Q4_K_MULTI_ROW_COLS) - 1)
        / (Q4_K_FIXED_WAVES * Q4_K_MULTI_ROW_COLS);

    if ((size_t)n_rows * sizeof(float) <= Q4_K_MULTI_ROW_SHARED_MEM_LIMIT) {
        const size_t shared_mem = (size_t)n_rows * sizeof(float);
        gemv_q4_k_f32_residual_multi_row_kernel<Q4_K_FIXED_WAVES>
            <<<n_blocks_x, threads_per_block, shared_mem, stream>>>(
                weights_q4_k, input, residual, output, n_rows, ncols_dst
            );
    } else {
        gemv_q4_k_f32_residual_direct_kernel<Q4_K_FIXED_WAVES>
            <<<n_blocks_x, threads_per_block, 0, stream>>>(
                weights_q4_k, input, residual, output, n_rows, ncols_dst
            );
    }
    return hipGetLastError();
}
