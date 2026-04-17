// Host harness for the Phase 2a WMMA proof of concept.
//
// Runs three tests in sequence:
//   1. bit-exact test against a CPU reference with FP32 accumulation.
//   2. identity diagnostic: B = I_16 → expect D = A. If this passes but
//      test 1 fails, the store pattern is transposed.
//   3. transposed-identity diagnostic: A = I_16 → expect D = B. Confirms
//      which operand the mismatch belongs to.
//
// Standalone — builds with `hipcc`, no ROCmForge dependencies.

#include "../../hip_kernels/wmma/wmma_gemm_16x16.hip"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <cmath>

#define HIP_CHECK(x) do {                                                    \
    hipError_t _e = (x);                                                     \
    if (_e != hipSuccess) {                                                  \
        fprintf(stderr, "HIP error %s at %s:%d\n",                           \
                hipGetErrorString(_e), __FILE__, __LINE__);                  \
        std::exit(1);                                                        \
    }                                                                        \
} while (0)

static void matmul_16x16_cpu_reference(
    const __half* a,
    const __half* b,
    float*        d)
{
    for (int i = 0; i < 16; ++i) {
        for (int j = 0; j < 16; ++j) {
            float acc = 0.0f;
            for (int k = 0; k < 16; ++k) {
                acc += __half2float(a[i * 16 + k]) *
                       __half2float(b[k * 16 + j]);
            }
            d[i * 16 + j] = acc;
        }
    }
}

static void fill_deterministic(__half* m) {
    for (int i = 0; i < 256; ++i) {
        // Small values, reproducible, no FP16 overflow risk. At most
        // 16 multiplies of |v| ≤ 63/128 sum to |acc| ≤ 16 · 0.5 · 0.5 = 4,
        // well inside FP16/FP32 range.
        const int v = (i % 64);
        m[i] = __float2half(static_cast<float>(v) / 128.0f);
    }
}

static void fill_identity(__half* m) {
    for (int i = 0; i < 256; ++i) m[i] = __float2half(0.0f);
    for (int i = 0; i < 16; ++i)  m[i * 16 + i] = __float2half(1.0f);
}

struct RunResult {
    std::vector<float> gpu;
    std::vector<float> cpu;
    bool               bit_exact;
    float              max_abs_diff;
    int                mismatch_count;
};

static RunResult run_once(const __half* h_A, const __half* h_B) {
    RunResult r;
    r.gpu.resize(256);
    r.cpu.resize(256);

    __half* d_A = nullptr;
    __half* d_B = nullptr;
    float*  d_D = nullptr;
    HIP_CHECK(hipMalloc(&d_A, 256 * sizeof(__half)));
    HIP_CHECK(hipMalloc(&d_B, 256 * sizeof(__half)));
    HIP_CHECK(hipMalloc(&d_D, 256 * sizeof(float)));

    HIP_CHECK(hipMemcpy(d_A, h_A, 256 * sizeof(__half), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_B, h_B, 256 * sizeof(__half), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemset(d_D, 0, 256 * sizeof(float)));

    HIP_CHECK(wmma_gemm_16x16_launch(d_A, d_B, d_D, 0));
    HIP_CHECK(hipDeviceSynchronize());

    HIP_CHECK(hipMemcpy(r.gpu.data(), d_D, 256 * sizeof(float), hipMemcpyDeviceToHost));

    matmul_16x16_cpu_reference(h_A, h_B, r.cpu.data());

    r.bit_exact = true;
    r.max_abs_diff = 0.0f;
    r.mismatch_count = 0;
    for (int i = 0; i < 256; ++i) {
        const float diff = std::fabs(r.gpu[i] - r.cpu[i]);
        if (r.gpu[i] != r.cpu[i]) {
            r.bit_exact = false;
            r.mismatch_count++;
        }
        if (diff > r.max_abs_diff) r.max_abs_diff = diff;
    }

    hipFree(d_A); hipFree(d_B); hipFree(d_D);
    return r;
}

static bool looks_transposed(const RunResult& r) {
    for (int i = 0; i < 16; ++i) {
        for (int j = 0; j < 16; ++j) {
            const float gpu = r.gpu[i * 16 + j];
            const float cpu_t = r.cpu[j * 16 + i];
            if (gpu != cpu_t) return false;
        }
    }
    return true;
}

static void dump_corner(const char* label, const std::vector<float>& m) {
    fprintf(stderr, "%s top-left 4x4:\n", label);
    for (int i = 0; i < 4; ++i) {
        fprintf(stderr, "  ");
        for (int j = 0; j < 4; ++j) {
            fprintf(stderr, "%9.4f ", m[i * 16 + j]);
        }
        fprintf(stderr, "\n");
    }
}

int main() {
    int dev = 0;
    HIP_CHECK(hipSetDevice(dev));
    hipDeviceProp_t prop;
    HIP_CHECK(hipGetDeviceProperties(&prop, dev));
    fprintf(stderr, "Device: %s (arch=%s)\n", prop.name, prop.gcnArchName);

    __half A[256], B[256];
    int failures = 0;

    // Test 1: deterministic random-ish inputs, bit-exact compare.
    fprintf(stderr, "\n[test 1] deterministic inputs, expect bit-exact match\n");
    fill_deterministic(A);
    fill_deterministic(B);
    RunResult r1 = run_once(A, B);
    fprintf(stderr, "  max_abs_diff = %.6f, mismatches = %d / 256\n",
            r1.max_abs_diff, r1.mismatch_count);
    if (r1.bit_exact) {
        fprintf(stderr, "  PASS (bit-exact)\n");
    } else {
        fprintf(stderr, "  FAIL — output differs from CPU reference\n");
        dump_corner("CPU", r1.cpu);
        dump_corner("GPU", r1.gpu);
        if (looks_transposed(r1)) {
            fprintf(stderr, "  ! GPU output matches CPU^T: store pattern is transposed.\n");
        }
        failures++;
    }

    // Test 2: B = I, expect D = A.
    fprintf(stderr, "\n[test 2] B = I_16, expect D = A\n");
    fill_deterministic(A);
    fill_identity(B);
    RunResult r2 = run_once(A, B);
    fprintf(stderr, "  max_abs_diff = %.6f, mismatches = %d / 256\n",
            r2.max_abs_diff, r2.mismatch_count);
    if (r2.bit_exact) {
        fprintf(stderr, "  PASS\n");
    } else {
        fprintf(stderr, "  FAIL — output differs from A\n");
        dump_corner("A",   r2.cpu);
        dump_corner("GPU", r2.gpu);
        if (looks_transposed(r2)) {
            fprintf(stderr, "  ! GPU output matches A^T: store pattern is transposed.\n");
        }
        failures++;
    }

    // Test 3: A = I, expect D = B.
    fprintf(stderr, "\n[test 3] A = I_16, expect D = B\n");
    fill_identity(A);
    fill_deterministic(B);
    RunResult r3 = run_once(A, B);
    fprintf(stderr, "  max_abs_diff = %.6f, mismatches = %d / 256\n",
            r3.max_abs_diff, r3.mismatch_count);
    if (r3.bit_exact) {
        fprintf(stderr, "  PASS\n");
    } else {
        fprintf(stderr, "  FAIL — output differs from B\n");
        dump_corner("B",   r3.cpu);
        dump_corner("GPU", r3.gpu);
        if (looks_transposed(r3)) {
            fprintf(stderr, "  ! GPU output matches B^T: store pattern is transposed.\n");
        }
        failures++;
    }

    fprintf(stderr, "\nsummary: %d/3 tests failed\n", failures);
    return failures == 0 ? 0 : 1;
}
