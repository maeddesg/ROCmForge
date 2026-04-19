#include "common.hip"

// Q4_K constants (must match Rust types.rs)
#define QK_K 256
#define Q4_K_BLOCK_SIZE 144

// ── Type Definitions ───────────────────────────────────────────────────────────────

/// Q4_K block layout (matches Rust Q4KBlock)
struct Q4_K_block {
    half d;              // scale (2 bytes)
    half dmin;           // min scale (2 bytes)
    uint8_t scales[12];  // non-uniform scales (12 bytes)
    uint8_t qs[128];     // packed 4-bit values (128 bytes)
};

// ── Device Functions ───────────────────────────────────────────────────────────────

/// Inline 6-bit (scale, min) unpack for Q4_K sub-block `j ∈ [0, 8)`.
/// Mirrors llama.cpp `get_scale_min_k4` and the helper in
/// `q4_k_gemv.hip` / `elementwise.hip`.
__device__ __forceinline__
void q4_k_get_scale_min(int j, const uint8_t* scales,
                        int* scale_out, int* min_out) {
    if (j < 4) {
        *scale_out = scales[j]     & 0x3F;
        *min_out   = scales[j + 4] & 0x3F;
    } else {
        *scale_out = (scales[j + 4] & 0x0F) | ((scales[j - 4] >> 6) << 4);
        *min_out   = (scales[j + 4] >>  4)  | ((scales[j]     >> 6) << 4);
    }
}

/// Dot product of one Q4_K super-block (256 elements, 144 bytes) against a
/// 256-element slice of the input vector, cooperatively computed by a
/// 32-thread warp. Thread `t` owns 8 consecutive elements within a single
/// sub-block (sub-block `j = t / 4`, positions `(t % 4) * 8 .. + 7`).
///
/// Dequant formula (super-block-aware, matches `q4_k_super_block_dot` in
/// `q4_k_gemv.hip` and the WMMA prefill kernel):
///     value = (d · scale_j) · nibble − (dmin · min_j)
__device__ inline float vec_dot_q4_k(const void* block_ptr, const float* vec, int offset) {
    const int tid = threadIdx.x;

    const uint8_t* block_bytes = static_cast<const uint8_t*>(block_ptr);

    half d_half, dmin_half;
    memcpy(&d_half,    block_bytes,     sizeof(half));
    memcpy(&dmin_half, block_bytes + 2, sizeof(half));
    const float d    = __half2float(d_half);
    const float dmin = __half2float(dmin_half);

    const uint8_t* scales = block_bytes + 4;
    const uint8_t* qs     = block_bytes + 16;

    // Sub-block + intra-sub-block position for this thread's 8 elements.
    // Sub-blocks (2k, 2k+1) share a 32-byte qs region; even j takes the low
    // nibbles, odd j takes the high nibbles of the same bytes.
    const int j         = tid / 4;
    const int pos_base  = (tid % 4) * 8;
    const int pair_k    = j / 2;
    const bool is_upper = (j & 1) != 0;
    const uint8_t* qs_pair = qs + pair_k * 32;

    int scale_j, min_j;
    q4_k_get_scale_min(j, scales, &scale_j, &min_j);
    const float d_scale = d    * (float)scale_j;
    const float d_min   = dmin * (float)min_j;

    float sum = 0.0f;
    #pragma unroll
    for (int l = 0; l < 8; ++l) {
        const int pos = pos_base + l;
        const uint8_t byte = qs_pair[pos];
        const int nib = is_upper ? (byte >> 4) : (byte & 0x0F);
        const float val = d_scale * (float)nib - d_min;
        const int elem_idx = j * 32 + pos;
        sum += val * vec[offset + elem_idx];
    }

    return sum;
}

// ── Template Kernel with ncols_dst ──────────────────────────────────────────────────

template<int ncols_dst>
__launch_bounds__(32, 1)
__global__ void gemm_q4_k_f32_kernel(
    const void* __restrict__ weights_q4_k,
    const float* __restrict__ input,
    float* __restrict__ output,
    int n_rows,
    int batch_size
) {
    const int col = blockIdx.x;
    const int batch_idx = blockIdx.y;
    const int tid = threadIdx.x;

    if (col >= ncols_dst || batch_idx >= batch_size) return;

    const uint8_t* col_base = static_cast<const uint8_t*>(weights_q4_k) + col * (n_rows / QK_K) * Q4_K_BLOCK_SIZE;
    const float* input_batch = input + batch_idx * n_rows;

    const int n_blocks = n_rows / QK_K;
    float sum = 0.0f;

    for (int block_idx = 0; block_idx < n_blocks; ++block_idx) {
        const void* block_ptr = col_base + block_idx * Q4_K_BLOCK_SIZE;
        sum += vec_dot_q4_k(block_ptr, input_batch, block_idx * QK_K);
    }

    // Warp reduction (RDNA: 32 threads = 1 wavefront)
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down(sum, offset);
    }

    if (tid == 0) {
        output[batch_idx * ncols_dst + col] = sum;
    }
}

// ── Generic Kernel (non-template) ───────────────────────────────────────────────────

__global__ void gemm_q4_k_f32_generic_kernel(
    const void* __restrict__ weights_q4_k,
    const float* __restrict__ input,
    float* __restrict__ output,
    int n_rows,
    int ncols_dst,
    int batch_size
) {
    const int col = blockIdx.x;
    const int batch_idx = blockIdx.y;
    const int tid = threadIdx.x;

    if (col >= ncols_dst || batch_idx >= batch_size) return;

    const uint8_t* col_base = static_cast<const uint8_t*>(weights_q4_k) + col * (n_rows / QK_K) * Q4_K_BLOCK_SIZE;
    const float* input_batch = input + batch_idx * n_rows;

    const int n_blocks = n_rows / QK_K;
    float sum = 0.0f;

    for (int block_idx = 0; block_idx < n_blocks; ++block_idx) {
        const void* block_ptr = col_base + block_idx * Q4_K_BLOCK_SIZE;
        sum += vec_dot_q4_k(block_ptr, input_batch, block_idx * QK_K);
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down(sum, offset);
    }

    if (tid == 0) {
        output[batch_idx * ncols_dst + col] = sum;
    }
}

// ── Dispatch Function (Called from Rust) ───────────────────────────────────────────

extern "C" hipError_t gemm_q4_k_f32_launch(
    const void* weights_q4_k,
    const float* input,
    float* output,
    int n_rows,
    int ncols_dst,
    int batch_size,
    hipStream_t stream
) {
    if (n_rows <= 0 || ncols_dst <= 0 || batch_size <= 0) {
        return hipErrorInvalidValue;
    }
    if (n_rows % QK_K != 0) {
        return hipErrorInvalidValue;
    }

    dim3 grid(ncols_dst, batch_size);
    dim3 block(32);

    switch (ncols_dst) {
        case 1:
            gemm_q4_k_f32_kernel<1><<<grid, block, 0, stream>>>(
                weights_q4_k, input, output, n_rows, batch_size);
            break;
        case 2:
            gemm_q4_k_f32_kernel<2><<<grid, block, 0, stream>>>(
                weights_q4_k, input, output, n_rows, batch_size);
            break;
        case 3:
            gemm_q4_k_f32_kernel<3><<<grid, block, 0, stream>>>(
                weights_q4_k, input, output, n_rows, batch_size);
            break;
        case 4:
            gemm_q4_k_f32_kernel<4><<<grid, block, 0, stream>>>(
                weights_q4_k, input, output, n_rows, batch_size);
            break;
        case 5:
            gemm_q4_k_f32_kernel<5><<<grid, block, 0, stream>>>(
                weights_q4_k, input, output, n_rows, batch_size);
            break;
        case 6:
            gemm_q4_k_f32_kernel<6><<<grid, block, 0, stream>>>(
                weights_q4_k, input, output, n_rows, batch_size);
            break;
        case 7:
            gemm_q4_k_f32_kernel<7><<<grid, block, 0, stream>>>(
                weights_q4_k, input, output, n_rows, batch_size);
            break;
        case 8:
            gemm_q4_k_f32_kernel<8><<<grid, block, 0, stream>>>(
                weights_q4_k, input, output, n_rows, batch_size);
            break;
        default:
            gemm_q4_k_f32_generic_kernel<<<grid, block, 0, stream>>>(
                weights_q4_k, input, output, n_rows, ncols_dst, batch_size);
            break;
    }

    return hipGetLastError();
}
