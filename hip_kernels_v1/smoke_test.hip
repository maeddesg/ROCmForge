// v1.0 build-system smoke kernel.
//
// Writes idx → out[idx] for a single launch. Used only to verify the
// CMake-via-build.rs pipeline produces a linkable static library and
// that the Rust harness can find the `extern "C"` launch symbol on
// gfx1201. No numerical or correctness claims beyond that.

#include <hip/hip_runtime.h>

__global__ void v1_smoke_test(float* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = (float)idx;
    }
}

extern "C" int launch_v1_smoke_test(float* out, int n, hipStream_t stream) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    hipLaunchKernelGGL(v1_smoke_test,
                       dim3(blocks), dim3(threads), 0, stream,
                       out, n);
    return (int)hipGetLastError();
}
