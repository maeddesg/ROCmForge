// Milestone 0 follow-up — FP8 conversion intrinsics gate (gfx1201).
//
// Verifies that each of the native FP8 / BF8 conversion builtins compiles
// and emits a HW instruction (v_cvt_pk_fp8_f32 etc.) rather than a SW
// emulation sequence.
//
// Each test kernel is kept small and uses a unique output address so the
// compiler cannot coalesce them.

#include <hip/hip_runtime.h>

// Test 1: FP32 × 2 → packed FP8x2 (E4M3). Writes the packed result back into
// the lower half (word=false) of an incoming int "old". This is the builtin
// HIP's __hip_cvt_float2_to_fp8x2 ultimately lowers to.
__global__ void k_cvt_pk_fp8_f32(const float* in, int* out) {
    int old = out[0];
    int r = __builtin_amdgcn_cvt_pk_fp8_f32(in[0], in[1], old, false);
    out[threadIdx.x] = r;
}

// Test 2: Unpack one FP8 (E4M3) byte from a packed int32 → FP32. Index 0..3
// picks which lane in the dword.
__global__ void k_cvt_f32_fp8(const int* packed, float* out) {
    float r = __builtin_amdgcn_cvt_f32_fp8(packed[0], 0);
    out[threadIdx.x] = r;
}

// Test 3: FP32 × 2 → packed BF8x2 (E5M2).
__global__ void k_cvt_pk_bf8_f32(const float* in, int* out) {
    int old = out[0];
    int r = __builtin_amdgcn_cvt_pk_bf8_f32(in[0], in[1], old, false);
    out[threadIdx.x] = r;
}

// Test 4: Unpack one BF8 (E5M2) byte → FP32.
__global__ void k_cvt_f32_bf8(const int* packed, float* out) {
    float r = __builtin_amdgcn_cvt_f32_bf8(packed[0], 0);
    out[threadIdx.x] = r;
}

// Test 5a: Stochastic-rounding FP32 → FP8 (E4M3).
__global__ void k_cvt_sr_fp8_f32(const float* in, int* out) {
    unsigned int seed = 12345u;
    int r = __builtin_amdgcn_cvt_sr_fp8_f32(in[0], seed, 0, 0);
    out[threadIdx.x] = r;
}

// Test 5b: Stochastic-rounding FP32 → BF8 (E5M2).
__global__ void k_cvt_sr_bf8_f32(const float* in, int* out) {
    unsigned int seed = 12345u;
    int r = __builtin_amdgcn_cvt_sr_bf8_f32(in[0], seed, 0, 0);
    out[threadIdx.x] = r;
}
