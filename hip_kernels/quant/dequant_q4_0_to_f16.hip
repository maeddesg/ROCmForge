// Q4_0 → FP16 dequantisation for the hipBLAS prefill path.
//
// Layout: input weight tensor is row-major [rows × cols] where cols must be a
// multiple of 32 (Q4_0 block size). Output is a row-major FP16 tensor of the
// same logical shape. The tensor is then re-interpreted as column-major by
// hipBLAS (with transa=T on the A operand).
//
// Thread mapping: one thread per Q4_0 block (32 elements). Each block is
// 18 bytes in, 64 bytes out.

#include "../common.hip"
#include <hip/hip_fp16.h>

#define QK4_0 32
#define Q4_0_BLOCK_BYTES 18

// Coalesced variant: each workgroup processes WG_BLOCKS Q4_0 blocks
// (= WG_BLOCKS * 32 output values) with 32-wide threads writing consecutive
// output indices per warp. WG_BLOCKS = 8 → 256 threads per WG matches
// wavefront geometry on RDNA.
__global__ void dequant_q4_0_to_f16_kernel_impl(
    const uint8_t* __restrict__ input,
    __half* __restrict__ output,
    int total_blocks
) {
    constexpr int WG_BLOCKS = 8;   // 8 Q4_0 blocks per workgroup
    constexpr int WG_OUTPUTS = WG_BLOCKS * QK4_0;  // 256
    const int wg_block_base = blockIdx.x * WG_BLOCKS;
    const int tid = threadIdx.x;

    // Shared scale table — one per Q4_0 block in this workgroup.
    __shared__ float s_scale[WG_BLOCKS];
    if (tid < WG_BLOCKS) {
        const int blk = wg_block_base + tid;
        if (blk < total_blocks) {
            __half d_h;
            memcpy(&d_h, input + blk * Q4_0_BLOCK_BYTES, sizeof(__half));
            s_scale[tid] = __half2float(d_h);
        } else {
            s_scale[tid] = 0.0f;
        }
    }
    __syncthreads();

    // Each thread writes ONE output value. Thread `tid` owns output
    // element `wg_block_base * 32 + tid` → block index = tid / 32,
    // element-within-block = tid % 32.
    const int local_blk = tid >> 5;              // tid / 32, range 0..7
    const int elem_idx = tid & 31;               // tid % 32
    const int global_blk = wg_block_base + local_blk;
    if (global_blk >= total_blocks) return;

    const uint8_t* qs = input + global_blk * Q4_0_BLOCK_BYTES + 2;
    // Q4_0 layout: qs[i] low nibble → element i; qs[i] high nibble → element i+16
    uint8_t byte;
    int q;
    if (elem_idx < 16) {
        byte = qs[elem_idx];
        q = int(byte & 0x0F) - 8;
    } else {
        byte = qs[elem_idx - 16];
        q = int(byte >> 4) - 8;
    }
    const float v = float(q) * s_scale[local_blk];
    output[global_blk * QK4_0 + elem_idx] = __float2half(v);
}

extern "C" hipError_t dequant_q4_0_to_f16(
    const void* input,
    void* output,
    int n_elements,
    hipStream_t stream
) {
    if (n_elements <= 0 || (n_elements % QK4_0) != 0) {
        return hipErrorInvalidValue;
    }
    constexpr int WG_BLOCKS = 8;
    const int total_blocks = n_elements / QK4_0;
    const int grid = (total_blocks + WG_BLOCKS - 1) / WG_BLOCKS;

    hipLaunchKernelGGL(
        dequant_q4_0_to_f16_kernel_impl,
        dim3(grid), dim3(WG_BLOCKS * QK4_0),
        0, stream,
        reinterpret_cast<const uint8_t*>(input),
        reinterpret_cast<__half*>(output),
        total_blocks
    );
    return hipGetLastError();
}

// FP32 → FP16 elementwise conversion (for input activations).
__global__ void convert_f32_to_f16_kernel_impl(
    const float* __restrict__ input,
    __half* __restrict__ output,
    int n
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    output[idx] = __float2half(input[idx]);
}

extern "C" hipError_t convert_f32_to_f16(
    const float* input,
    void* output,
    int n,
    hipStream_t stream
) {
    if (n <= 0) return hipErrorInvalidValue;
    const int block = 256;
    const int grid = (n + block - 1) / block;
    hipLaunchKernelGGL(
        convert_f32_to_f16_kernel_impl,
        dim3(grid), dim3(block),
        0, stream,
        input, reinterpret_cast<__half*>(output), n
    );
    return hipGetLastError();
}

// FP16 → FP32 elementwise conversion (for output activations).
__global__ void convert_f16_to_f32_kernel_impl(
    const __half* __restrict__ input,
    float* __restrict__ output,
    int n
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    output[idx] = __half2float(input[idx]);
}

extern "C" hipError_t convert_f16_to_f32(
    const void* input,
    float* output,
    int n,
    hipStream_t stream
) {
    if (n <= 0) return hipErrorInvalidValue;
    const int block = 256;
    const int grid = (n + block - 1) / block;
    hipLaunchKernelGGL(
        convert_f16_to_f32_kernel_impl,
        dim3(grid), dim3(block),
        0, stream,
        reinterpret_cast<const __half*>(input), output, n
    );
    return hipGetLastError();
}
