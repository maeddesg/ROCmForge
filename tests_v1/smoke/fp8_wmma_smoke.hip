// Meilenstein 0 — FP8 WMMA Smoke Test Kernel (gfx1201 / RDNA 4)
//
// Goal: verify that `__builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12`
// compiles, runs, and produces correct results on the RX 9070 XT. If any
// of the C entry points fails, the v1.0 architecture falls back to FP16.
//
// Five C entry points, invoked by the Rust harness:
//   smoke_get_device_info      — report detected GPU arch
//   smoke_run_fp8_wmma         — 16×16×16 matmul with FP8 E4M3 inputs
//   smoke_run_fp16_wmma        — same shape with FP16 inputs, for timing
//   smoke_benchmark_wmma       — GEMM bench (N iterations, median µs)
//   smoke_fp32_to_fp8_roundtrip — FP32 → E4M3 → FP32 on host-supplied floats

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_fp8.h>

#include <cstdint>
#include <cstring>

using fp16x8  = _Float16      __attribute__((ext_vector_type(8)));
using fp32x8  = float         __attribute__((ext_vector_type(8)));
using int32x2 = int           __attribute__((ext_vector_type(2)));

// --- kernels ----------------------------------------------------------------

// FP8 (E4M3) × FP8 (E4M3) → FP32. One workgroup, one wave, one WMMA call.
// A, B: 16×16 byte-packed row-major. D: 16×16 FP32 row-major.
// Lane L: row = L%16, k_half = L/16 (∈ {0,1}). Lane holds A[row][k_half*8..+7]
// and B[k_half*8..+7][col=row]. This matches the CK-tile gfx12 layout
// (kABKLane=2, kABK1PerLane=8) — contiguous K, not interleaved.
__global__ void wmma_gemm_fp8_e4m3_16x16(
    const uint8_t* __restrict__ A,   // 256 bytes
    const uint8_t* __restrict__ B,   // 256 bytes
    float*         __restrict__ D    // 256 floats
) {
    const int lane   = threadIdx.x;
    const int row    = lane % 16;
    const int col    = lane % 16;
    const int k_base = (lane / 16) * 8;

    uint8_t a_bytes[8];
    uint8_t b_bytes[8];
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        a_bytes[i] = A[row * 16 + k_base + i];
        b_bytes[i] = B[(k_base + i) * 16 + col];
    }

    int32x2 a_reg, b_reg;
    std::memcpy(&a_reg, a_bytes, 8);
    std::memcpy(&b_reg, b_bytes, 8);

    fp32x8 c_reg = {0, 0, 0, 0, 0, 0, 0, 0};
    fp32x8 d_reg = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(
        a_reg, b_reg, c_reg);

    // D layout: lane L holds col=L%16, rows (L/16)*8 + v for v∈[0,8).
    const int row_start = (lane / 16) * 8;
    #pragma unroll
    for (int v = 0; v < 8; ++v) {
        D[(row_start + v) * 16 + col] = d_reg[v];
    }
}

// FP16 × FP16 → FP32. Same 16×16×16 shape, for FP8-vs-FP16 timing.
// A-fragment layout follows the gfx12 matrix-calculator output used in
// hip_kernels/wmma_gemm_16x16.hip: offsets {0..3, 8..11} with k_set = (L/16)*4.
__global__ void wmma_gemm_fp16_16x16(
    const __half* __restrict__ A,
    const __half* __restrict__ B,
    float*        __restrict__ D
) {
    const int lane  = threadIdx.x;
    const int row   = lane % 16;
    const int col   = lane % 16;
    const int k_set = (lane / 16) * 4;

    fp16x8 a_reg, b_reg;
    a_reg[0] = A[row * 16 + k_set + 0];
    a_reg[1] = A[row * 16 + k_set + 1];
    a_reg[2] = A[row * 16 + k_set + 2];
    a_reg[3] = A[row * 16 + k_set + 3];
    a_reg[4] = A[row * 16 + k_set + 8];
    a_reg[5] = A[row * 16 + k_set + 9];
    a_reg[6] = A[row * 16 + k_set + 10];
    a_reg[7] = A[row * 16 + k_set + 11];

    b_reg[0] = B[(k_set + 0)  * 16 + col];
    b_reg[1] = B[(k_set + 1)  * 16 + col];
    b_reg[2] = B[(k_set + 2)  * 16 + col];
    b_reg[3] = B[(k_set + 3)  * 16 + col];
    b_reg[4] = B[(k_set + 8)  * 16 + col];
    b_reg[5] = B[(k_set + 9)  * 16 + col];
    b_reg[6] = B[(k_set + 10) * 16 + col];
    b_reg[7] = B[(k_set + 11) * 16 + col];

    fp32x8 c_reg = {0, 0, 0, 0, 0, 0, 0, 0};
    fp32x8 d_reg = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(
        a_reg, b_reg, c_reg);

    const int row_start = (lane / 16) * 8;
    #pragma unroll
    for (int v = 0; v < 8; ++v) {
        D[(row_start + v) * 16 + col] = d_reg[v];
    }
}

// --- FP32 ↔ FP8 conversion kernel (Test 4) ----------------------------------

// Downcast FP32 → FP8-E4M3 (OCP), then upcast back to FP32. Per thread, one
// element. Uses hip_fp8 conversion helpers for portability.
__global__ void fp32_fp8_e4m3_roundtrip(
    const float* __restrict__ src,
    float*       __restrict__ dst,
    int          n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    __hip_fp8_storage_t q = __hip_cvt_float_to_fp8(
        src[idx], __HIP_SATFINITE, __HIP_E4M3);
    __half h = __half(__hip_cvt_fp8_to_halfraw(q, __HIP_E4M3));
    dst[idx] = __half2float(h);
}

// --- C entry points for Rust harness ----------------------------------------

extern "C" {

// Test 1 sanity: report the GPU arch actually detected at runtime.
int smoke_get_device_info(char* out_name, int out_cap, int* out_major, int* out_minor) {
    hipDeviceProp_t prop;
    hipError_t err = hipGetDeviceProperties(&prop, 0);
    if (err != hipSuccess) return (int)err;
    if (out_cap > 0 && out_name) {
        int n = (int)strlen(prop.gcnArchName);
        if (n >= out_cap) n = out_cap - 1;
        memcpy(out_name, prop.gcnArchName, n);
        out_name[n] = '\0';
    }
    if (out_major) *out_major = prop.major;
    if (out_minor) *out_minor = prop.minor;
    return 0;
}

// Test 2 — Correctness. Host provides 256 FP8 A bytes, 256 FP8 B bytes,
// receives 256 FP32 D values. Allocates/copies/launches/copies-back.
int smoke_run_fp8_wmma(const uint8_t* h_a, const uint8_t* h_b, float* h_d) {
    uint8_t* d_a = nullptr;
    uint8_t* d_b = nullptr;
    float*   d_d = nullptr;
    hipError_t err;
    if ((err = hipMalloc(&d_a, 256)) != hipSuccess) goto err;
    if ((err = hipMalloc(&d_b, 256)) != hipSuccess) goto err;
    if ((err = hipMalloc(&d_d, 256 * sizeof(float))) != hipSuccess) goto err;
    if ((err = hipMemcpy(d_a, h_a, 256, hipMemcpyHostToDevice)) != hipSuccess) goto err;
    if ((err = hipMemcpy(d_b, h_b, 256, hipMemcpyHostToDevice)) != hipSuccess) goto err;

    hipLaunchKernelGGL(wmma_gemm_fp8_e4m3_16x16,
        dim3(1), dim3(32), 0, 0,
        d_a, d_b, d_d);
    err = hipGetLastError();
    if (err != hipSuccess) goto err;
    err = hipDeviceSynchronize();
    if (err != hipSuccess) goto err;

    err = hipMemcpy(h_d, d_d, 256 * sizeof(float), hipMemcpyDeviceToHost);
err:
    if (d_a) hipFree(d_a);
    if (d_b) hipFree(d_b);
    if (d_d) hipFree(d_d);
    return (int)err;
}

// FP16 counterpart — same shape, used by Test 3 for correctness cross-check.
// A, B are host-side FP16 (passed as uint16_t raw bits).
int smoke_run_fp16_wmma(const uint16_t* h_a_bits, const uint16_t* h_b_bits, float* h_d) {
    __half*  d_a = nullptr;
    __half*  d_b = nullptr;
    float*   d_d = nullptr;
    hipError_t err;
    if ((err = hipMalloc(&d_a, 256 * sizeof(__half))) != hipSuccess) goto err;
    if ((err = hipMalloc(&d_b, 256 * sizeof(__half))) != hipSuccess) goto err;
    if ((err = hipMalloc(&d_d, 256 * sizeof(float))) != hipSuccess) goto err;
    if ((err = hipMemcpy(d_a, h_a_bits, 256 * sizeof(uint16_t), hipMemcpyHostToDevice))
        != hipSuccess) goto err;
    if ((err = hipMemcpy(d_b, h_b_bits, 256 * sizeof(uint16_t), hipMemcpyHostToDevice))
        != hipSuccess) goto err;

    hipLaunchKernelGGL(wmma_gemm_fp16_16x16,
        dim3(1), dim3(32), 0, 0,
        d_a, d_b, d_d);
    err = hipGetLastError();
    if (err != hipSuccess) goto err;
    err = hipDeviceSynchronize();
    if (err != hipSuccess) goto err;

    err = hipMemcpy(h_d, d_d, 256 * sizeof(float), hipMemcpyDeviceToHost);
err:
    if (d_a) hipFree(d_a);
    if (d_b) hipFree(d_b);
    if (d_d) hipFree(d_d);
    return (int)err;
}

// Test 3 — Benchmark. Runs the kernel `iters` times with `warmup` warmups.
// Returns the median elapsed time (µs) through `out_median_us`, and the
// minimum through `out_min_us`. `kind`: 0 = FP8, 1 = FP16.
int smoke_benchmark_wmma(
    int kind,
    int warmup,
    int iters,
    float* out_median_us,
    float* out_min_us
) {
    if (iters <= 0 || iters > 1024) return -1;
    uint8_t* d_a8 = nullptr;
    uint8_t* d_b8 = nullptr;
    __half*  d_ah = nullptr;
    __half*  d_bh = nullptr;
    float*   d_d  = nullptr;
    hipEvent_t e_start = nullptr, e_stop = nullptr;
    hipError_t err;

    if ((err = hipMalloc(&d_d, 256 * sizeof(float))) != hipSuccess) goto err;
    if (kind == 0) {
        if ((err = hipMalloc(&d_a8, 256)) != hipSuccess) goto err;
        if ((err = hipMalloc(&d_b8, 256)) != hipSuccess) goto err;
        if ((err = hipMemset(d_a8, 0x3C, 256)) != hipSuccess) goto err;  // ~1.0 E4M3 bias
        if ((err = hipMemset(d_b8, 0x3C, 256)) != hipSuccess) goto err;
    } else {
        if ((err = hipMalloc(&d_ah, 256 * sizeof(__half))) != hipSuccess) goto err;
        if ((err = hipMalloc(&d_bh, 256 * sizeof(__half))) != hipSuccess) goto err;
        if ((err = hipMemset(d_ah, 0x3C, 256 * sizeof(__half))) != hipSuccess) goto err;
        if ((err = hipMemset(d_bh, 0x3C, 256 * sizeof(__half))) != hipSuccess) goto err;
    }
    if ((err = hipEventCreate(&e_start)) != hipSuccess) goto err;
    if ((err = hipEventCreate(&e_stop))  != hipSuccess) goto err;

    for (int i = 0; i < warmup; ++i) {
        if (kind == 0) {
            hipLaunchKernelGGL(wmma_gemm_fp8_e4m3_16x16,
                dim3(1), dim3(32), 0, 0, d_a8, d_b8, d_d);
        } else {
            hipLaunchKernelGGL(wmma_gemm_fp16_16x16,
                dim3(1), dim3(32), 0, 0, d_ah, d_bh, d_d);
        }
    }
    if ((err = hipDeviceSynchronize()) != hipSuccess) goto err;

    {
        float times[1024];
        for (int i = 0; i < iters; ++i) {
            hipEventRecord(e_start, 0);
            if (kind == 0) {
                hipLaunchKernelGGL(wmma_gemm_fp8_e4m3_16x16,
                    dim3(1), dim3(32), 0, 0, d_a8, d_b8, d_d);
            } else {
                hipLaunchKernelGGL(wmma_gemm_fp16_16x16,
                    dim3(1), dim3(32), 0, 0, d_ah, d_bh, d_d);
            }
            hipEventRecord(e_stop, 0);
            hipEventSynchronize(e_stop);
            float ms = 0.f;
            hipEventElapsedTime(&ms, e_start, e_stop);
            times[i] = ms * 1000.0f;
        }
        // sort
        for (int i = 1; i < iters; ++i) {
            float t = times[i];
            int j = i - 1;
            while (j >= 0 && times[j] > t) { times[j+1] = times[j]; --j; }
            times[j+1] = t;
        }
        if (out_median_us) *out_median_us = times[iters / 2];
        if (out_min_us)    *out_min_us    = times[0];
        err = hipSuccess;
    }

err:
    if (d_a8) hipFree(d_a8);
    if (d_b8) hipFree(d_b8);
    if (d_ah) hipFree(d_ah);
    if (d_bh) hipFree(d_bh);
    if (d_d)  hipFree(d_d);
    if (e_start) hipEventDestroy(e_start);
    if (e_stop)  hipEventDestroy(e_stop);
    return (int)err;
}

// Test 4 — FP32 → FP8 E4M3 → FP32 roundtrip on host-supplied floats.
int smoke_fp32_to_fp8_roundtrip(const float* h_src, float* h_dst, int n) {
    if (n <= 0) return -1;
    float* d_src = nullptr;
    float* d_dst = nullptr;
    hipError_t err;
    if ((err = hipMalloc(&d_src, n * sizeof(float))) != hipSuccess) goto err;
    if ((err = hipMalloc(&d_dst, n * sizeof(float))) != hipSuccess) goto err;
    if ((err = hipMemcpy(d_src, h_src, n * sizeof(float), hipMemcpyHostToDevice))
        != hipSuccess) goto err;

    {
        int block = 256;
        int grid  = (n + block - 1) / block;
        hipLaunchKernelGGL(fp32_fp8_e4m3_roundtrip,
            dim3(grid), dim3(block), 0, 0, d_src, d_dst, n);
        err = hipGetLastError();
        if (err != hipSuccess) goto err;
        err = hipDeviceSynchronize();
        if (err != hipSuccess) goto err;
    }
    err = hipMemcpy(h_dst, d_dst, n * sizeof(float), hipMemcpyDeviceToHost);
err:
    if (d_src) hipFree(d_src);
    if (d_dst) hipFree(d_dst);
    return (int)err;
}

} // extern "C"
