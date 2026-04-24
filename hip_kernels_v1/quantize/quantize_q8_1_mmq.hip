// MMQ Q8_1 activation pre-quantizer — MMQ-Port Schritt 1 of P0.2.
// Ported from ~/tmp/llama.cpp/ggml/src/ggml-cuda/quantize.cu:176-271
// (DS4 layout only — used by Q4_K / Q5_K / Q4_1 / Q5_1 / Q8_1 MMQ paths).
//
// Block-layout bit-identical to llama.cpp's `block_q8_1_mmq`:
//     half2 ds4[4];       // 4 × (d=scale, s=sum) per 32-element sub-block
//     int8_t qs[128];     // 128 quantized values (4 sub-blocks × 32)
// total: 16 + 128 = 144 bytes per 128-element block.
//
// Difference vs our existing `block_q8_1` (hip_kernels_v1/quantize/quantize_q8_1.hip):
//   block_q8_1    : 36 B / 32 elements  (1 scale + 32 int8)
//   block_q8_1_mmq: 144 B / 128 elements (4 scales + 128 int8)
// Same bytes-per-element ratio (1.125 B/elem), but the larger block is
// what MMQ's load_tiles_* functions expect in shared memory.
//
// Within a mmq-block the 4 sub-blocks of 32 elements are each quantized
// independently (separate amax, separate d). That means the first 32
// int8 values of any mmq-block are numerically identical to a bare
// block_q8_1 built from the same 32 floats — the parity test in
// tests_v1/wmma_int8_mmq_test.rs checks this.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>

#define QK8_1                    32     // 32 elements per sub-block (scale unit)
#define MMQ_BLOCK_SIZE           128    // 128 elements per mmq-block
#define SUB_BLOCKS_PER_MMQ       4
#define THREADS_PER_MMQ_BLOCK    32     // one wave per mmq-block

struct block_q8_1_mmq {
    __half2 ds4[SUB_BLOCKS_PER_MMQ];    // 4 × (d, sum) per 32-element sub-block
    int8_t  qs[MMQ_BLOCK_SIZE];         // 128 quantised values
};
static_assert(sizeof(block_q8_1_mmq) == 144,
    "block_q8_1_mmq must be exactly 144 bytes (16 B scales + 128 B qs)");

// Reduce amax across 8 lanes (one sub-block of 32 elems / 4 per thread = 8 threads).
__device__ __forceinline__ float rf_v1_mmq_sub_reduce_max(float x) {
    #pragma unroll
    for (int offset = 4; offset > 0; offset >>= 1) {
        x = fmaxf(x, __shfl_xor(x, offset));
    }
    return x;
}

__device__ __forceinline__ float rf_v1_mmq_sub_reduce_sum(float x) {
    #pragma unroll
    for (int offset = 4; offset > 0; offset >>= 1) {
        x += __shfl_xor(x, offset);
    }
    return x;
}

// One kernel-block = 32 threads (one wave) = 128 elements = 1 mmq-block.
// Grid.x = n_mmq_blocks.
__launch_bounds__(THREADS_PER_MMQ_BLOCK, 1)
__global__ void rf_v1_quantize_q8_1_mmq_kernel(
    const float*        __restrict__ x,
    block_q8_1_mmq*     __restrict__ y,
    int                              n_elements)
{
    const int mmq_block_idx = blockIdx.x;
    const int tid           = threadIdx.x;       // 0..31
    const int sub_block     = tid / 8;           // 0..3 (which 32-elem sub-block)
    const int sub_lane      = tid & 7;           // 0..7 (which float4 within sub-block)

    const int base = mmq_block_idx * MMQ_BLOCK_SIZE + tid * 4;

    // Load 4 floats per thread as a vector (matches llama.cpp's float4 path).
    float4 xi = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    if (base + 3 < n_elements) {
        xi = reinterpret_cast<const float4*>(x)[base / 4];
    } else {
        // Tail path — fill per-element to avoid OOB.
        if (base + 0 < n_elements) xi.x = x[base + 0];
        if (base + 1 < n_elements) xi.y = x[base + 1];
        if (base + 2 < n_elements) xi.z = x[base + 2];
        if (base + 3 < n_elements) xi.w = x[base + 3];
    }

    // Per-thread amax + sum.
    float amax = fmaxf(fmaxf(fabsf(xi.x), fabsf(xi.y)),
                       fmaxf(fabsf(xi.z), fabsf(xi.w)));
    float sum  = xi.x + xi.y + xi.z + xi.w;

    // Reduce across the 8 lanes that form one 32-element sub-block.
    amax = rf_v1_mmq_sub_reduce_max(amax);
    sum  = rf_v1_mmq_sub_reduce_sum(sum);

    // Quantize. `d_inv` = 127/amax; safe when amax == 0 → all-zero output,
    // scale stored as 0.
    const float d_inv = (amax > 0.0f) ? (127.0f / amax) : 0.0f;

    char4 q;
    q.x = (int8_t)roundf(xi.x * d_inv);
    q.y = (int8_t)roundf(xi.y * d_inv);
    q.z = (int8_t)roundf(xi.z * d_inv);
    q.w = (int8_t)roundf(xi.w * d_inv);

    // Write 4 int8 in one 32-bit store.
    char4* qs4 = reinterpret_cast<char4*>(y[mmq_block_idx].qs);
    qs4[tid] = q;

    // One lane per sub-block (the first lane) writes the ds4 slot.
    if (sub_lane == 0) {
        const float d = (amax > 0.0f) ? (1.0f / d_inv) : 0.0f;
        y[mmq_block_idx].ds4[sub_block] =
            __halves2half2(__float2half(d), __float2half(sum));
    }
}

extern "C" hipError_t rocmforge_launch_quantize_q8_1_mmq(
    const float*  input,
    void*         mmq_output,
    int           n_elements,
    hipStream_t   stream)
{
    if (n_elements <= 0)                     return hipErrorInvalidValue;
    if ((n_elements % MMQ_BLOCK_SIZE) != 0)  return hipErrorInvalidValue;

    const int n_mmq_blocks = n_elements / MMQ_BLOCK_SIZE;
    dim3 grid(n_mmq_blocks, 1, 1);
    dim3 block(THREADS_PER_MMQ_BLOCK, 1, 1);

    rf_v1_quantize_q8_1_mmq_kernel<<<grid, block, 0, stream>>>(
        input, reinterpret_cast<block_q8_1_mmq*>(mmq_output), n_elements);

    return hipGetLastError();
}
