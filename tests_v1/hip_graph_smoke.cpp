// HIP-Graph crash-threshold smoke test for gfx1201 (RX 9070 XT).
//
// Finds the exact number of kernel nodes at which hipGraphInstantiate
// / hipGraphLaunch breaks on this GPU. Standalone — compile with:
//
//   hipcc tests_v1/hip_graph_smoke.cpp -o hip_graph_smoke \
//         --offload-arch=gfx1201 -O2
//
// Run:
//   ./hip_graph_smoke 2>&1 | tee hip_graph_smoke_results.txt
//
// The algorithm:
//   Phase 1 — exponential search 1, 2, 4, ..., 1024.
//   Phase 2 — binary search between the last OK N and the first
//             FAIL N.
//
// Ping-pong between two buffers A and B so each kernel in the graph
// writes to a distinct (alternating) destination — prevents the
// HIP graph capture from collapsing the dependency chain.

#include <hip/hip_runtime.h>

#include <cmath>
#include <csetjmp>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>

// ── Scale kernel: out[i] = in[i] * 2.0f ─────────────────────────────

__global__ void scale_kernel(float* __restrict__ dst,
                             const float* __restrict__ src,
                             int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        dst[i] = src[i] * 2.0f;
    }
}

// Secondary kernel (used only in the optional multi-kernel bonus pass
// — same shape, different math so the graph has a heterogeneous node
// set).
__global__ void add_one_kernel(float* __restrict__ dst,
                               const float* __restrict__ src,
                               int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        dst[i] = src[i] + 1.0f;
    }
}

// ── Signal handler — HIP-Graph crashes can SIGSEGV instead of
//    returning a hipError ────────────────────────────────────────────

static volatile sig_atomic_t g_current_n = 0;
static volatile sig_atomic_t g_last_ok_n = 0;
static volatile const char*  g_current_phase = "init";

extern "C" void crash_handler(int sig) {
    // Only async-signal-safe calls here. write(2, ...) to a small
    // static buffer + _exit.
    const char* signame =
        sig == SIGSEGV ? "SIGSEGV" :
        sig == SIGABRT ? "SIGABRT" :
        sig == SIGBUS  ? "SIGBUS"  :
        sig == SIGILL  ? "SIGILL"  : "unknown";
    char buf[256];
    const int n  = (int)g_current_n;
    const int ok = (int)g_last_ok_n;
    int len = snprintf(buf, sizeof(buf),
        "\n*** HARD CRASH: %s during phase=%s N=%d (last_ok=%d) ***\n"
        "=== RESULT ===\nHard crash at N=%d\nLast OK: N=%d\n",
        signame, (const char*)g_current_phase, n, ok, n, ok);
    (void)!write(2, buf, (size_t)len);
    _exit(1);
}

// ── Helpers ─────────────────────────────────────────────────────────

#define HIP_CHECK_SOFT(call, label_fail) do { \
    hipError_t _e = (call); \
    if (_e != hipSuccess) { \
        printf("  FAIL %s (code %d) at %s\n", \
               hipGetErrorString(_e), (int)_e, #call); \
        goto label_fail; \
    } \
} while (0)

#define HIP_CHECK_FATAL(call) do { \
    hipError_t _e = (call); \
    if (_e != hipSuccess) { \
        fprintf(stderr, "FATAL: %s (code %d) at %s\n", \
                hipGetErrorString(_e), (int)_e, #call); \
        exit(2); \
    } \
} while (0)

static constexpr int N_ELEMS      = 4096;
static constexpr int THREADS_PER_BLOCK = 256;
static constexpr int BLOCKS       = (N_ELEMS + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

// Reset A[i] = 1.0f, B[i] = 0.0f
static void reset_buffers(float* d_A, float* d_B) {
    float ones[N_ELEMS];
    for (int i = 0; i < N_ELEMS; ++i) ones[i] = 1.0f;
    HIP_CHECK_FATAL(hipMemcpy(d_A, ones, N_ELEMS * sizeof(float),
                              hipMemcpyHostToDevice));
    HIP_CHECK_FATAL(hipMemset(d_B, 0, N_ELEMS * sizeof(float)));
}

enum TrialResult {
    OK = 0,
    FAIL_CAPTURE,
    FAIL_END,
    FAIL_INSTANTIATE,
    FAIL_LAUNCH,
    FAIL_SYNC,
    FAIL_VALUE,
};

// Run one trial with exactly N graph nodes. Returns OK or a FAIL
// reason. Sets *final_value to the host-read result[0] (for N ≤ 100,
// also checked against 2^N).
static TrialResult trial(int N, hipStream_t stream,
                         float* d_A, float* d_B,
                         float* final_value_out) {
    g_current_n = N;

    reset_buffers(d_A, d_B);

    // Begin capture.
    if (hipStreamBeginCapture(stream, hipStreamCaptureModeGlobal) != hipSuccess) {
        return FAIL_CAPTURE;
    }

    // Ping-pong N times: even iters A → B, odd iters B → A.
    for (int i = 0; i < N; ++i) {
        float* dst = (i & 1) ? d_A : d_B;
        const float* src = (i & 1) ? d_B : d_A;
        scale_kernel<<<BLOCKS, THREADS_PER_BLOCK, 0, stream>>>(dst, src, N_ELEMS);
    }

    hipGraph_t graph = nullptr;
    if (hipStreamEndCapture(stream, &graph) != hipSuccess) {
        return FAIL_END;
    }

    hipGraphExec_t exec = nullptr;
    hipError_t inst_err = hipGraphInstantiate(&exec, graph, nullptr, nullptr, 0);
    if (inst_err != hipSuccess) {
        printf("  hipGraphInstantiate failed: %s (%d)\n",
               hipGetErrorString(inst_err), (int)inst_err);
        hipGraphDestroy(graph);
        return FAIL_INSTANTIATE;
    }

    hipError_t launch_err = hipGraphLaunch(exec, stream);
    if (launch_err != hipSuccess) {
        printf("  hipGraphLaunch failed: %s (%d)\n",
               hipGetErrorString(launch_err), (int)launch_err);
        hipGraphExecDestroy(exec);
        hipGraphDestroy(graph);
        return FAIL_LAUNCH;
    }

    hipError_t sync_err = hipStreamSynchronize(stream);
    if (sync_err != hipSuccess) {
        printf("  hipStreamSynchronize failed: %s (%d)\n",
               hipGetErrorString(sync_err), (int)sync_err);
        hipGraphExecDestroy(exec);
        hipGraphDestroy(graph);
        return FAIL_SYNC;
    }

    // Read back result.
    float* result_ptr = (N & 1) ? d_B : d_A;
    float host_val = 0.0f;
    HIP_CHECK_FATAL(hipMemcpy(&host_val, result_ptr, sizeof(float),
                              hipMemcpyDeviceToHost));
    *final_value_out = host_val;

    hipGraphExecDestroy(exec);
    hipGraphDestroy(graph);

    return OK;
}

static void report_trial(int N, TrialResult r, float val) {
    if (r == OK) {
        g_last_ok_n = N;
        if (N <= 100) {
            float expected = powf(2.0f, (float)N);
            bool correct = fabsf(val - expected) < expected * 1e-5f;
            printf("N=%4d:  %s  result=%.6e  (expected %.6e)\n",
                   N, correct ? "OK   " : "WRONG", val, expected);
        } else {
            // 2^N overflows float32 for N > 127. `inf` is the
            // mathematically correct outcome for a kernel that
            // faithfully doubles the value 128+ times; NaN or 0
            // would indicate a real kernel bug. We mark overflow-
            // to-inf as OK for the crash-detection purposes of
            // this test.
            bool kernel_ran = !std::isnan(val) && val != 0.0f;
            const char* status = kernel_ran ? "OK   " : "WRONG";
            const char* note   =
                std::isinf(val) ? "(expected inf — 2^N overflows float32)"
                                : "(finite value, past overflow boundary)";
            printf("N=%4d:  %s  result=%.6e  %s\n", N, status, val, note);
        }
    } else {
        const char* why =
            r == FAIL_CAPTURE     ? "BeginCapture" :
            r == FAIL_END         ? "EndCapture" :
            r == FAIL_INSTANTIATE ? "Instantiate" :
            r == FAIL_LAUNCH      ? "Launch" :
            r == FAIL_SYNC        ? "Synchronize" :
            r == FAIL_VALUE       ? "Value-mismatch" : "?";
        printf("N=%4d:  FAIL (%s)\n", N, why);
    }
}

int main() {
    signal(SIGSEGV, crash_handler);
    signal(SIGABRT, crash_handler);
    signal(SIGBUS,  crash_handler);
    signal(SIGILL,  crash_handler);

    // Device info.
    int dev = 0;
    hipDeviceProp_t props;
    HIP_CHECK_FATAL(hipGetDeviceProperties(&props, dev));
    int runtime_v = 0;
    HIP_CHECK_FATAL(hipRuntimeGetVersion(&runtime_v));
    int driver_v = 0;
    HIP_CHECK_FATAL(hipDriverGetVersion(&driver_v));

    printf("=== HIP Graph Crash-Threshold Smoke Test ===\n");
    printf("GPU:     %s (arch=%s)\n", props.name, props.gcnArchName);
    printf("ROCm:    runtime=%d.%d driver=%d.%d\n",
           runtime_v / 1000, (runtime_v % 1000) / 10,
           driver_v / 1000, (driver_v % 1000) / 10);
    printf("Compile: --offload-arch=gfx1201\n\n");

    // Allocate working buffers.
    float* d_A = nullptr;
    float* d_B = nullptr;
    HIP_CHECK_FATAL(hipMalloc(&d_A, N_ELEMS * sizeof(float)));
    HIP_CHECK_FATAL(hipMalloc(&d_B, N_ELEMS * sizeof(float)));
    hipStream_t stream = nullptr;
    HIP_CHECK_FATAL(hipStreamCreate(&stream));

    // ── Phase 1: exponential search ────────────────────────────────
    printf("--- Phase 1: Exponential Search ---\n");
    g_current_phase = "Phase 1";

    const int phase1_ns[] = {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024};
    const int phase1_count = sizeof(phase1_ns) / sizeof(phase1_ns[0]);

    int last_ok = 0;
    int first_fail = -1;

    for (int i = 0; i < phase1_count; ++i) {
        int N = phase1_ns[i];
        float val = 0.0f;
        TrialResult r = trial(N, stream, d_A, d_B, &val);
        report_trial(N, r, val);
        if (r == OK) {
            last_ok = N;
        } else {
            first_fail = N;
            break;
        }
    }

    // ── Phase 2: binary search ─────────────────────────────────────
    if (first_fail < 0) {
        printf("\n--- Phase 1 completed without a failure up to N=%d ---\n",
               phase1_ns[phase1_count - 1]);
        printf("\n=== RESULT ===\n");
        printf("No crash threshold found in range [1, %d].\n",
               phase1_ns[phase1_count - 1]);
        printf("Last OK: N=%d\n", last_ok);
        printf("(HIP-Graph may have been fixed on this build, or the\n");
        printf(" crash threshold is higher than 1024.)\n");
    } else {
        printf("\n--- Phase 2: Binary Search [%d, %d] ---\n", last_ok, first_fail);
        g_current_phase = "Phase 2";

        int lo = last_ok;     // known OK
        int hi = first_fail;  // known FAIL
        int last_ok_binary = last_ok;

        while (hi - lo > 1) {
            int mid = lo + (hi - lo) / 2;
            float val = 0.0f;
            TrialResult r = trial(mid, stream, d_A, d_B, &val);
            report_trial(mid, r, val);
            if (r == OK) {
                lo = mid;
                last_ok_binary = mid;
            } else {
                hi = mid;
            }
        }

        printf("\n=== RESULT ===\n");
        printf("Crash Threshold: %d <= N_OK  and  N=%d is first FAIL\n",
               last_ok_binary, hi);
        printf("Last OK:     N=%d kernels\n", last_ok_binary);
        printf("First FAIL:  N=%d kernels\n", hi);

        printf("\nImplications for ROCmForge (Qwen3-8B decode):\n");
        printf("  Decode dispatches/token : 599\n");
        printf("  Per-layer dispatches    : ~16\n");
        printf("  Full-decode graph       : %s (need N<=%d, have 599)\n",
               (last_ok_binary >= 599) ? "FEASIBLE" : "NOT FEASIBLE",
               last_ok_binary);
        printf("  Per-layer sub-graph     : %s (need N<=%d, have ~16)\n",
               (last_ok_binary >= 16) ? "FEASIBLE" : "NOT FEASIBLE",
               last_ok_binary);
    }

    // ── Optional bonus: heterogeneous kernel graph ─────────────────
    printf("\n--- Bonus: Heterogeneous Kernel Graph (scale + add_one) ---\n");
    g_current_phase = "Bonus";

    // Use the smallest N that previously FAILED (or a small sample
    // from Phase 1 if no failure). Test whether alternating two
    // different kernels has a different threshold.
    int bonus_n = (first_fail > 0) ? first_fail : 64;
    printf("Testing heterogeneous graph at N=%d ...\n", bonus_n);

    reset_buffers(d_A, d_B);
    if (hipStreamBeginCapture(stream, hipStreamCaptureModeGlobal) == hipSuccess) {
        for (int i = 0; i < bonus_n; ++i) {
            float* dst = (i & 1) ? d_A : d_B;
            const float* src = (i & 1) ? d_B : d_A;
            if (i & 1) {
                add_one_kernel<<<BLOCKS, THREADS_PER_BLOCK, 0, stream>>>(dst, src, N_ELEMS);
            } else {
                scale_kernel<<<BLOCKS, THREADS_PER_BLOCK, 0, stream>>>(dst, src, N_ELEMS);
            }
        }
        hipGraph_t bonus_graph = nullptr;
        if (hipStreamEndCapture(stream, &bonus_graph) == hipSuccess) {
            hipGraphExec_t bonus_exec = nullptr;
            hipError_t e = hipGraphInstantiate(&bonus_exec, bonus_graph, nullptr, nullptr, 0);
            if (e == hipSuccess) {
                e = hipGraphLaunch(bonus_exec, stream);
                if (e == hipSuccess) {
                    e = hipStreamSynchronize(stream);
                }
                hipGraphExecDestroy(bonus_exec);
            }
            hipGraphDestroy(bonus_graph);
            if (e == hipSuccess) {
                printf("Heterogeneous N=%d: OK  (same threshold behaviour as homogeneous)\n",
                       bonus_n);
            } else {
                printf("Heterogeneous N=%d: FAIL %s (%d)\n",
                       bonus_n, hipGetErrorString(e), (int)e);
            }
        } else {
            printf("Heterogeneous N=%d: EndCapture FAIL\n", bonus_n);
        }
    } else {
        printf("Heterogeneous N=%d: BeginCapture FAIL\n", bonus_n);
    }

    // Cleanup.
    hipStreamDestroy(stream);
    hipFree(d_A);
    hipFree(d_B);

    printf("\n(Done.)\n");
    return 0;
}
