// Probe for direct BF16 conversion builtins on gfx1201.
#include <hip/hip_runtime.h>

#if defined(TEST_PK_BF16)
__global__ void k(const float* in, unsigned int* out) {
    unsigned int r = __builtin_amdgcn_cvt_pk_bf16_f32(in[0], in[1]);
    out[threadIdx.x] = r;
}
#elif defined(TEST_PKRTZ_BF16)
__global__ void k(const float* in, unsigned int* out) {
    unsigned int r = __builtin_amdgcn_cvt_pkrtz_bf16_f32(in[0], in[1]);
    out[threadIdx.x] = r;
}
#elif defined(TEST_PKRTZ_F16)
// Known to exist on RDNA — baseline for comparison.
__global__ void k(const float* in, unsigned int* out) {
    unsigned int r = __builtin_amdgcn_cvt_pkrtz_f16_f32(in[0], in[1]);
    out[threadIdx.x] = r;
}
#elif defined(TEST_CVT_F32_BF16)
__global__ void k(const unsigned int* in, float* out) {
    float r = __builtin_amdgcn_cvt_f32_bf16(in[0], 0);
    out[threadIdx.x] = r;
}
#endif
