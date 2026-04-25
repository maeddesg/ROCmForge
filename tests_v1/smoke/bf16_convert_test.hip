// Milestone 0 follow-up — BF16 conversion intrinsics gate (gfx1201).
//
// Goal: determine which FP32 ↔ BF16 conversion pathway emits a HW
// instruction on gfx1201 (RDNA 4), vs. a software sequence. The answer
// drives how the Dequant-IR Level-2 (BF16-WMMA) emitter generates code.

#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>

// --- Test A: HIP portable API (__float2bfloat16 / __float22bfloat162_rn) ----
// Must compile because the header exists. Question is WHAT it lowers to.
__global__ void k_hip_float2bfloat16(const float* in, __hip_bfloat16* out) {
    out[threadIdx.x] = __float2bfloat16(in[threadIdx.x]);
}

__global__ void k_hip_bfloat162float(const __hip_bfloat16* in, float* out) {
    out[threadIdx.x] = __bfloat162float(in[threadIdx.x]);
}

__global__ void k_hip_float22bfloat162_rn(const float* in, __hip_bfloat162* out) {
    float2 f2 = {in[2*threadIdx.x + 0], in[2*threadIdx.x + 1]};
    out[threadIdx.x] = __float22bfloat162_rn(f2);
}

// --- Test B: Clang native __bf16 casts (lower-level) ------------------------
using bf16x2 = __bf16 __attribute__((ext_vector_type(2)));

__global__ void k_native_bf16_cast(const float* in, unsigned int* out) {
    __bf16 a = static_cast<__bf16>(in[2*threadIdx.x + 0]);
    __bf16 b = static_cast<__bf16>(in[2*threadIdx.x + 1]);
    bf16x2 v = {a, b};
    unsigned int packed;
    __builtin_memcpy(&packed, &v, 4);
    out[threadIdx.x] = packed;
}

// --- Test C: WMMA BF16 (expected to exist per Milestone 0 smoke) ------------
using bf16x8 = __bf16 __attribute__((ext_vector_type(8)));
using fp32x8 = float  __attribute__((ext_vector_type(8)));

__global__ void k_wmma_bf16(
    const __bf16* A,
    const __bf16* B,
    float* D
) {
    bf16x8 a_reg, b_reg;
    // Layout dummy (we only care about compilation).
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        a_reg[i] = A[threadIdx.x * 8 + i];
        b_reg[i] = B[threadIdx.x * 8 + i];
    }
    fp32x8 c_reg = {0,0,0,0,0,0,0,0};
    fp32x8 d_reg = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(
        a_reg, b_reg, c_reg);
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        D[threadIdx.x * 8 + i] = d_reg[i];
    }
}
