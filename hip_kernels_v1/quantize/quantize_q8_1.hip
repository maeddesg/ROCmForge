// Q8_1 activation pre-quantizer — Phase 2 Schritt 1/3 of the
// llama.cpp MMVQ kernel port. Ported from
// ~/tmp/llama.cpp/ggml/src/ggml-cuda/quantize.cu:4-48.
//
// Block-layout bit-identical to llama.cpp's `block_q8_1`:
//     half2 ds;         // ds.x = d (scale), ds.y ≈ d * Σ qs[i]
//     int8_t qs[32];    // quantised values
// total: 36 bytes per 32-element block. The `ds.y` field stores the
// warp-reduced float sum of the original `xi`, which equals
// d * Σ qs[i] up to rounding — this is exactly what ggml-common.h:253
// documents ("s = d * sum(qs[i])") and what downstream Q4_0 / Q4_1
// MMVQ kernels expect.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>

#define QK8_1                    32
#define CUDA_QUANTIZE_BLOCK_SIZE 256
#define WARP_SIZE                32

struct block_q8_1 {
    __half2 ds;
    int8_t  qs[QK8_1];
};
static_assert(sizeof(block_q8_1) == 36, "block_q8_1 must be exactly 36 bytes");

__device__ __forceinline__ float rf_v1_q8_1_warp_reduce_max(float x) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        x = fmaxf(x, __shfl_xor(x, offset));
    }
    return x;
}

__device__ __forceinline__ float rf_v1_q8_1_warp_reduce_sum(float x) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        x += __shfl_xor(x, offset);
    }
    return x;
}

__launch_bounds__(CUDA_QUANTIZE_BLOCK_SIZE, 1)
__global__ void rf_v1_quantize_q8_1_kernel(
    const float*  __restrict__ x,
    block_q8_1*   __restrict__ y,
    int                         n_elements)
{
    const int i0 = blockDim.x * blockIdx.x + threadIdx.x;

    // OOB threads must still participate in the warp reduction so the
    // active lanes get correct amax/sum. Use sentinel zeros.
    const bool in_range = (i0 < n_elements);
    const float xi = in_range ? x[i0] : 0.0f;

    float amax = fabsf(xi);
    float sum  = xi;

    amax = rf_v1_q8_1_warp_reduce_max(amax);
    sum  = rf_v1_q8_1_warp_reduce_sum(sum);

    if (!in_range) return;

    const int ib  = i0 / QK8_1;
    const int iqs = i0 % QK8_1;

    const float  d = amax / 127.0f;
    const int8_t q = (amax == 0.0f) ? (int8_t)0 : (int8_t)roundf(xi / d);

    y[ib].qs[iqs] = q;

    if (iqs == 0) {
        __half d_h = __float2half(d);
        __half s_h = __float2half(sum);
        y[ib].ds   = __halves2half2(d_h, s_h);
    }
}

extern "C" hipError_t rocmforge_launch_quantize_q8_1(
    const float*  input,
    void*         q8_1_output,
    int           n_elements,
    hipStream_t   stream)
{
    if (n_elements <= 0)            return hipErrorInvalidValue;
    if ((n_elements % QK8_1) != 0)  return hipErrorInvalidValue;

    const int n_blocks =
        (n_elements + CUDA_QUANTIZE_BLOCK_SIZE - 1) / CUDA_QUANTIZE_BLOCK_SIZE;

    rf_v1_quantize_q8_1_kernel<<<n_blocks, CUDA_QUANTIZE_BLOCK_SIZE, 0, stream>>>(
        input, reinterpret_cast<block_q8_1*>(q8_1_output), n_elements);

    return hipGetLastError();
}
