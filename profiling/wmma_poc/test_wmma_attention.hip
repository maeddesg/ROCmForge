// Standalone correctness harness for the Phase 3a WMMA attention PoC.
//
// Runs three tests:
//   1. deterministic Q/K/V vs CPU reference (absolute tol ≤ 1e-2).
//   2. identity case: K = Q → S is symmetric, P distribution stable.
//   3. zero-V: V = 0 → O = 0 regardless of Q/K. Sanity check.
//
// Builds with hipcc, no ROCmForge dependencies. Uses the same kernel
// source via direct include.

#include "../../hip_kernels/wmma/wmma_attention_prefill.hip"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>

#define SEQ      64
#define HEAD_DIM 128

#define HIP_CHECK(x) do {                                                    \
    hipError_t _e = (x);                                                     \
    if (_e != hipSuccess) {                                                  \
        fprintf(stderr, "HIP error %s at %s:%d\n",                           \
                hipGetErrorString(_e), __FILE__, __LINE__);                  \
        std::exit(1);                                                        \
    }                                                                        \
} while (0)

static void cpu_attention_reference(
    const __half* q,
    const __half* k,
    const __half* v,
    float* o,
    float scale)
{
    std::vector<float> s(SEQ * SEQ, 0.0f);
    // S = Q · K^T / sqrt(head_dim)
    for (int i = 0; i < SEQ; ++i) {
        for (int j = 0; j < SEQ; ++j) {
            float acc = 0.0f;
            for (int d = 0; d < HEAD_DIM; ++d) {
                acc += __half2float(q[i * HEAD_DIM + d]) *
                       __half2float(k[j * HEAD_DIM + d]);
            }
            s[i * SEQ + j] = acc * scale;
        }
    }
    // Row-wise softmax.
    for (int i = 0; i < SEQ; ++i) {
        float* row = &s[i * SEQ];
        float m = row[0];
        for (int j = 1; j < SEQ; ++j) if (row[j] > m) m = row[j];
        float sum = 0.0f;
        for (int j = 0; j < SEQ; ++j) {
            row[j] = std::exp(row[j] - m);
            sum += row[j];
        }
        for (int j = 0; j < SEQ; ++j) row[j] /= sum;
    }
    // O = P · V
    for (int i = 0; i < SEQ; ++i) {
        for (int d = 0; d < HEAD_DIM; ++d) {
            float acc = 0.0f;
            for (int j = 0; j < SEQ; ++j) {
                acc += s[i * SEQ + j] * __half2float(v[j * HEAD_DIM + d]);
            }
            o[i * HEAD_DIM + d] = acc;
        }
    }
}

struct RunResult {
    std::vector<float> gpu;
    std::vector<float> cpu;
    float              max_abs_diff;
    float              mean_abs_diff;
    int                worst_idx;
};

static RunResult run_once(
    const __half* h_Q,
    const __half* h_K,
    const __half* h_V,
    float scale)
{
    RunResult r;
    r.gpu.assign(SEQ * HEAD_DIM, 0.0f);
    r.cpu.assign(SEQ * HEAD_DIM, 0.0f);

    __half* d_Q = nullptr;
    __half* d_K = nullptr;
    __half* d_V = nullptr;
    float*  d_O = nullptr;
    HIP_CHECK(hipMalloc(&d_Q, SEQ * HEAD_DIM * sizeof(__half)));
    HIP_CHECK(hipMalloc(&d_K, SEQ * HEAD_DIM * sizeof(__half)));
    HIP_CHECK(hipMalloc(&d_V, SEQ * HEAD_DIM * sizeof(__half)));
    HIP_CHECK(hipMalloc(&d_O, SEQ * HEAD_DIM * sizeof(float)));

    HIP_CHECK(hipMemcpy(d_Q, h_Q, SEQ * HEAD_DIM * sizeof(__half), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_K, h_K, SEQ * HEAD_DIM * sizeof(__half), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_V, h_V, SEQ * HEAD_DIM * sizeof(__half), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemset(d_O, 0, SEQ * HEAD_DIM * sizeof(float)));

    HIP_CHECK(wmma_attention_prefill_64_launch(d_Q, d_K, d_V, d_O, scale, 0));
    HIP_CHECK(hipDeviceSynchronize());

    HIP_CHECK(hipMemcpy(r.gpu.data(), d_O, SEQ * HEAD_DIM * sizeof(float), hipMemcpyDeviceToHost));

    cpu_attention_reference(h_Q, h_K, h_V, r.cpu.data(), scale);

    r.max_abs_diff = 0.0f;
    double sum = 0.0;
    r.worst_idx = 0;
    for (int i = 0; i < SEQ * HEAD_DIM; ++i) {
        const float d = std::fabs(r.gpu[i] - r.cpu[i]);
        sum += d;
        if (d > r.max_abs_diff) { r.max_abs_diff = d; r.worst_idx = i; }
    }
    r.mean_abs_diff = float(sum / (SEQ * HEAD_DIM));

    hipFree(d_Q); hipFree(d_K); hipFree(d_V); hipFree(d_O);
    return r;
}

static void fill_deterministic(__half* m, int n, int seed) {
    for (int i = 0; i < n; ++i) {
        const int v = ((i + seed) * 37) & 0x3F;     // 0..63
        m[i] = __float2half(float(v) / 256.0f);     // [0, 0.25)
    }
}

static void fill_zero(__half* m, int n) {
    for (int i = 0; i < n; ++i) m[i] = __float2half(0.0f);
}

int main() {
    int dev = 0;
    HIP_CHECK(hipSetDevice(dev));
    hipDeviceProp_t prop;
    HIP_CHECK(hipGetDeviceProperties(&prop, dev));
    fprintf(stderr, "Device: %s (arch=%s)\n", prop.name, prop.gcnArchName);

    const float scale = 1.0f / std::sqrt(float(HEAD_DIM));
    int failures = 0;

    std::vector<__half> Q(SEQ * HEAD_DIM);
    std::vector<__half> K(SEQ * HEAD_DIM);
    std::vector<__half> V(SEQ * HEAD_DIM);

    // ----- test 1: deterministic -----
    fprintf(stderr, "\n[test 1] deterministic Q/K/V vs CPU reference\n");
    fill_deterministic(Q.data(), Q.size(), 1);
    fill_deterministic(K.data(), K.size(), 7);
    fill_deterministic(V.data(), V.size(), 13);
    {
        RunResult r = run_once(Q.data(), K.data(), V.data(), scale);
        const float tol = 1.0e-2f;
        fprintf(stderr, "  max_abs_diff = %.4e  mean_abs_diff = %.4e  (tolerance %.1e)\n",
                r.max_abs_diff, r.mean_abs_diff, tol);
        if (r.max_abs_diff > tol) {
            fprintf(stderr, "  FAIL (worst idx=%d: gpu=%g cpu=%g)\n",
                    r.worst_idx, r.gpu[r.worst_idx], r.cpu[r.worst_idx]);
            failures++;
        } else {
            fprintf(stderr, "  PASS\n");
        }
    }

    // ----- test 2: K = Q identity -----
    fprintf(stderr, "\n[test 2] K = Q  (symmetric S)\n");
    fill_deterministic(Q.data(), Q.size(), 2);
    memcpy(K.data(), Q.data(), Q.size() * sizeof(__half));
    fill_deterministic(V.data(), V.size(), 17);
    {
        RunResult r = run_once(Q.data(), K.data(), V.data(), scale);
        const float tol = 1.0e-2f;
        fprintf(stderr, "  max_abs_diff = %.4e  mean_abs_diff = %.4e  (tolerance %.1e)\n",
                r.max_abs_diff, r.mean_abs_diff, tol);
        if (r.max_abs_diff > tol) {
            fprintf(stderr, "  FAIL (worst idx=%d: gpu=%g cpu=%g)\n",
                    r.worst_idx, r.gpu[r.worst_idx], r.cpu[r.worst_idx]);
            failures++;
        } else {
            fprintf(stderr, "  PASS\n");
        }
    }

    // ----- test 3: V = 0 → O = 0 -----
    fprintf(stderr, "\n[test 3] V = 0  ⇒  O = 0\n");
    fill_deterministic(Q.data(), Q.size(), 3);
    fill_deterministic(K.data(), K.size(), 5);
    fill_zero(V.data(), V.size());
    {
        RunResult r = run_once(Q.data(), K.data(), V.data(), scale);
        const float tol = 1.0e-3f;
        fprintf(stderr, "  max_abs_diff = %.4e  mean_abs_diff = %.4e  (tolerance %.1e)\n",
                r.max_abs_diff, r.mean_abs_diff, tol);
        if (r.max_abs_diff > tol) {
            fprintf(stderr, "  FAIL\n");
            failures++;
        } else {
            fprintf(stderr, "  PASS\n");
        }
    }

    fprintf(stderr, "\nsummary: %d/3 tests failed\n", failures);
    return failures == 0 ? 0 : 1;
}
