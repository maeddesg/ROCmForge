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

    gemv_q6_k_f32_kernel<<<ncols_dst, 32, 0, stream>>>(
        weights_q6_k, input, output, n_rows, ncols_dst
    );

    return hipGetLastError();
}
