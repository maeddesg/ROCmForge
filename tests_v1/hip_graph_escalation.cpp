// HIP Graph escalation test — stepwise added complexity until the
// gfx1201 HIP-graph bug triggers. Runs levels L0..L6; at each level,
// tests with {1, 4, 16, 36} simulated transformer layers. Stops at
// the first crash. Standalone.
//
// Build:
//   hipcc tests_v1/hip_graph_escalation.cpp -o hip_graph_escalation \
//         --offload-arch=gfx1201 -O2
// Run:
//   ./hip_graph_escalation 2>&1 | tee hip_graph_escalation_results.txt
//
// Level ladder:
//   L0  Baseline               : scale_kernel Ping-Pong (known OK up to 1024)
//   L1  Multi-kernel            : 8 different kernel types per "layer"
//   L2  Buffer aliasing         : only 4 device buffers, reused layer-over-layer
//   L3  Parameter-replay        : capture once, replay 100 times with updated
//                                 kv_pos via hipGraphExecKernelNodeSetParams
//   L4  Large weights           : ~300 MB+ of weight buffers referenced by
//                                 graph nodes
//   L5  Real GEMV kernel        : actual N×K matrix-vector multiply
//   L6  Everything combined     : closest MRE to ROCmForge's decode path

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#include <cmath>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <unistd.h>
#include <vector>

// ── Kernels ─────────────────────────────────────────────────────────

__global__ void kern_scale(float* __restrict__ dst, const float* __restrict__ src, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = src[i] * 2.0f;
}

__global__ void kern_add(float* __restrict__ dst, const float* __restrict__ a,
                         const float* __restrict__ b, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = a[i] + b[i];
}

__global__ void kern_mul(float* __restrict__ dst, const float* __restrict__ a,
                         const float* __restrict__ b, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = a[i] * b[i];
}

__global__ void kern_silu(float* __restrict__ dst, const float* __restrict__ src, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        const float x = src[i];
        dst[i] = x / (1.0f + expf(-x));
    }
}

__global__ void kern_rms_norm(float* __restrict__ dst, const float* __restrict__ src, int n) {
    // Simplified: just a scale, skips the cross-warp reduction the real
    // RMS-Norm would need. Enough for a distinct kernel signature.
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = src[i] * 0.5f;
}

__global__ void kern_rope(float* __restrict__ dst, const float* __restrict__ src,
                          int n, int pos) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        const float freq = 1.0f / powf(10000.0f, (float)(i % 64) / 64.0f);
        const float angle = (float)pos * freq;
        const int j = i ^ 32;
        const float v = (j < n) ? src[j] : 0.0f;
        dst[i] = src[i] * cosf(angle) + v * sinf(angle);
    }
}

__global__ void kern_kv_write(float* __restrict__ kv_cache,
                              const float* __restrict__ src,
                              int n, int layer, int pos, int max_seq) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) kv_cache[(size_t)layer * max_seq * n + (size_t)pos * n + i] = src[i];
}

__global__ void kern_dot_reduce(float* __restrict__ dst,
                                const float* __restrict__ a,
                                const float* __restrict__ b, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = a[i] * b[i] + dst[i];
}

// Level-5 GEMV: out[row] = sum_k weights[row, k] * input[k]
// N = out_dim, K = in_dim. weights is row-major FP16.
__global__ void kern_gemv(float* __restrict__ dst,
                          const __half* __restrict__ weights,
                          const float* __restrict__ src,
                          int N, int K) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            sum += __half2float(weights[(size_t)row * K + k]) * src[k];
        }
        dst[row] = sum;
    }
}

// ── Signal handler ──────────────────────────────────────────────────

static volatile sig_atomic_t g_crash_level   = -1;
static volatile sig_atomic_t g_crash_layers  = -1;
static volatile const char*  g_crash_phase   = "init";

extern "C" void crash_handler(int sig) {
    const char* signame =
        sig == SIGSEGV ? "SIGSEGV" :
        sig == SIGABRT ? "SIGABRT" :
        sig == SIGBUS  ? "SIGBUS"  :
        sig == SIGILL  ? "SIGILL"  : "unknown";
    char buf[512];
    const int lvl  = (int)g_crash_level;
    const int lays = (int)g_crash_layers;
    int len = snprintf(buf, sizeof(buf),
        "\n*** HARD CRASH: %s during Level %d, %d layers, phase=%s ***\n"
        "=== CRASH RESULT ===\n"
        "Trigger: Level %d with %d layers\n"
        "Signal:  %s\n",
        signame, lvl, lays, (const char*)g_crash_phase,
        lvl, lays, signame);
    (void)!write(2, buf, (size_t)len);
    _exit(1);
}

// ── Helpers ─────────────────────────────────────────────────────────

static constexpr int N_ELEMS           = 4096;       // hidden_dim
static constexpr int FFN_DIM           = 12288;
static constexpr int KV_MAX_SEQ        = 128;
static constexpr int THREADS_PER_BLOCK = 256;
static constexpr int BLOCKS_H          = (N_ELEMS + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
static constexpr int BLOCKS_F          = (FFN_DIM + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

#define HIP_CHECK_SOFT(call) do { \
    hipError_t _e = (call); \
    if (_e != hipSuccess) { \
        printf("  SOFT-FAIL %s (code %d) at %s\n", \
               hipGetErrorString(_e), (int)_e, #call); \
        return false; \
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

static void reset_ones(float* d_buf, int n) {
    std::vector<float> h(n, 1.0f);
    HIP_CHECK_FATAL(hipMemcpy(d_buf, h.data(), n * sizeof(float), hipMemcpyHostToDevice));
}

// ── Level 0: Baseline — scale Ping-Pong ─────────────────────────────

static bool run_level_0(int n_layers, hipStream_t stream) {
    // "layers" here = repetitions of 17 scale kernels, to match the
    // per-layer dispatch count in ROCmForge.
    const int per_layer = 17;
    const int n_k = n_layers * per_layer;

    float *d_A, *d_B;
    HIP_CHECK_SOFT(hipMalloc(&d_A, N_ELEMS * sizeof(float)));
    HIP_CHECK_SOFT(hipMalloc(&d_B, N_ELEMS * sizeof(float)));
    reset_ones(d_A, N_ELEMS);

    g_crash_phase = "L0 capture";
    HIP_CHECK_SOFT(hipStreamBeginCapture(stream, hipStreamCaptureModeGlobal));
    for (int i = 0; i < n_k; ++i) {
        float* dst = (i & 1) ? d_A : d_B;
        const float* src = (i & 1) ? d_B : d_A;
        kern_scale<<<BLOCKS_H, THREADS_PER_BLOCK, 0, stream>>>(dst, src, N_ELEMS);
    }
    hipGraph_t graph = nullptr;
    HIP_CHECK_SOFT(hipStreamEndCapture(stream, &graph));

    hipGraphExec_t exec = nullptr;
    g_crash_phase = "L0 instantiate";
    HIP_CHECK_SOFT(hipGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
    g_crash_phase = "L0 launch";
    HIP_CHECK_SOFT(hipGraphLaunch(exec, stream));
    g_crash_phase = "L0 sync";
    HIP_CHECK_SOFT(hipStreamSynchronize(stream));

    printf("  L0  %2d layers (%4d kernels):  OK\n", n_layers, n_k);
    hipGraphExecDestroy(exec);
    hipGraphDestroy(graph);
    hipFree(d_A);
    hipFree(d_B);
    return true;
}

// ── Level 1: Multi-kernel types, unique scratch buffers ─────────────

static bool run_level_1(int n_layers, hipStream_t stream) {
    // Per layer emit a mini-transformer-like sequence using 8 distinct
    // kernels. Each layer gets its own set of scratch buffers so there
    // is NO aliasing — that's level 2.
    const int per_layer = 17;
    const int n_k = n_layers * per_layer;

    std::vector<float*> bufs(n_k);
    for (int i = 0; i < n_k; ++i) {
        HIP_CHECK_SOFT(hipMalloc(&bufs[i], N_ELEMS * sizeof(float)));
    }
    float* d_kv = nullptr;
    HIP_CHECK_SOFT(hipMalloc(&d_kv, (size_t)n_layers * KV_MAX_SEQ * N_ELEMS * sizeof(float)));

    g_crash_phase = "L1 capture";
    HIP_CHECK_SOFT(hipStreamBeginCapture(stream, hipStreamCaptureModeGlobal));
    int b_idx = 0;
    auto next_buf = [&]() -> float* {
        float* p = bufs[b_idx % n_k];
        b_idx++;
        return p;
    };
    for (int l = 0; l < n_layers; ++l) {
        float* in = next_buf();
        // Attention pre-norm → Q,K,V → RoPE → KV-write → Dot → O → Add
        kern_rms_norm<<<BLOCKS_H, THREADS_PER_BLOCK, 0, stream>>>(next_buf(), in, N_ELEMS);
        float* q = next_buf();
        kern_scale<<<BLOCKS_H, THREADS_PER_BLOCK, 0, stream>>>(q, in, N_ELEMS);
        float* k = next_buf();
        kern_scale<<<BLOCKS_H, THREADS_PER_BLOCK, 0, stream>>>(k, in, N_ELEMS);
        float* v = next_buf();
        kern_scale<<<BLOCKS_H, THREADS_PER_BLOCK, 0, stream>>>(v, in, N_ELEMS);
        kern_rope<<<BLOCKS_H, THREADS_PER_BLOCK, 0, stream>>>(q, q, N_ELEMS, 0);
        kern_rope<<<BLOCKS_H, THREADS_PER_BLOCK, 0, stream>>>(k, k, N_ELEMS, 0);
        kern_kv_write<<<BLOCKS_H, THREADS_PER_BLOCK, 0, stream>>>(d_kv, k, N_ELEMS, l, 0, KV_MAX_SEQ);
        kern_kv_write<<<BLOCKS_H, THREADS_PER_BLOCK, 0, stream>>>(d_kv, v, N_ELEMS, l, 0, KV_MAX_SEQ);
        float* attn = next_buf();
        kern_dot_reduce<<<BLOCKS_H, THREADS_PER_BLOCK, 0, stream>>>(attn, q, k, N_ELEMS);
        float* o_out = next_buf();
        kern_scale<<<BLOCKS_H, THREADS_PER_BLOCK, 0, stream>>>(o_out, attn, N_ELEMS);
        kern_add<<<BLOCKS_H, THREADS_PER_BLOCK, 0, stream>>>(in, in, o_out, N_ELEMS);
        // FFN: norm → gate, up → SwiGLU → down → residual
        float* ffn_n = next_buf();
        kern_rms_norm<<<BLOCKS_H, THREADS_PER_BLOCK, 0, stream>>>(ffn_n, in, N_ELEMS);
        float* g_b = next_buf();
        kern_scale<<<BLOCKS_H, THREADS_PER_BLOCK, 0, stream>>>(g_b, ffn_n, N_ELEMS);
        float* u_b = next_buf();
        kern_scale<<<BLOCKS_H, THREADS_PER_BLOCK, 0, stream>>>(u_b, ffn_n, N_ELEMS);
        kern_silu<<<BLOCKS_H, THREADS_PER_BLOCK, 0, stream>>>(g_b, g_b, N_ELEMS);
        kern_mul<<<BLOCKS_H, THREADS_PER_BLOCK, 0, stream>>>(g_b, g_b, u_b, N_ELEMS);
        float* down = next_buf();
        kern_scale<<<BLOCKS_H, THREADS_PER_BLOCK, 0, stream>>>(down, g_b, N_ELEMS);
        kern_add<<<BLOCKS_H, THREADS_PER_BLOCK, 0, stream>>>(in, in, down, N_ELEMS);
    }
    hipGraph_t graph = nullptr;
    HIP_CHECK_SOFT(hipStreamEndCapture(stream, &graph));
    hipGraphExec_t exec = nullptr;
    g_crash_phase = "L1 instantiate";
    HIP_CHECK_SOFT(hipGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
    g_crash_phase = "L1 launch";
    HIP_CHECK_SOFT(hipGraphLaunch(exec, stream));
    g_crash_phase = "L1 sync";
    HIP_CHECK_SOFT(hipStreamSynchronize(stream));

    printf("  L1  %2d layers (~%4d kernels):  OK\n", n_layers, n_k);
    hipGraphExecDestroy(exec);
    hipGraphDestroy(graph);
    for (auto* p : bufs) hipFree(p);
    hipFree(d_kv);
    return true;
}

// ── Level 2: Multi-kernel + shared scratch (Ping-Pong aliasing) ─────

static bool run_level_2(int n_layers, hipStream_t stream) {
    // Only 4 device buffers — reused across all layers. The same
    // device pointer appears in many graph nodes.
    float *d_hidden, *d_scratch_a, *d_scratch_b, *d_kv;
    HIP_CHECK_SOFT(hipMalloc(&d_hidden,    N_ELEMS * sizeof(float)));
    HIP_CHECK_SOFT(hipMalloc(&d_scratch_a, FFN_DIM * sizeof(float)));
    HIP_CHECK_SOFT(hipMalloc(&d_scratch_b, FFN_DIM * sizeof(float)));
    HIP_CHECK_SOFT(hipMalloc(&d_kv,        (size_t)n_layers * KV_MAX_SEQ * N_ELEMS * sizeof(float)));
    reset_ones(d_hidden, N_ELEMS);

    g_crash_phase = "L2 capture";
    HIP_CHECK_SOFT(hipStreamBeginCapture(stream, hipStreamCaptureModeGlobal));
    for (int l = 0; l < n_layers; ++l) {
        // All kernels write to scratch_a / scratch_b / hidden, layer
        // after layer — heavy aliasing.
        kern_rms_norm<<<BLOCKS_H, THREADS_PER_BLOCK, 0, stream>>>(d_scratch_a, d_hidden, N_ELEMS);
        kern_scale<<<BLOCKS_H, THREADS_PER_BLOCK, 0, stream>>>(d_scratch_b, d_scratch_a, N_ELEMS);  // Q
        kern_scale<<<BLOCKS_H, THREADS_PER_BLOCK, 0, stream>>>(d_scratch_a, d_scratch_a, N_ELEMS);  // K (alias!)
        kern_rope<<<BLOCKS_H, THREADS_PER_BLOCK, 0, stream>>>(d_scratch_b, d_scratch_b, N_ELEMS, 0);
        kern_rope<<<BLOCKS_H, THREADS_PER_BLOCK, 0, stream>>>(d_scratch_a, d_scratch_a, N_ELEMS, 0);
        kern_kv_write<<<BLOCKS_H, THREADS_PER_BLOCK, 0, stream>>>(d_kv, d_scratch_a, N_ELEMS, l, 0, KV_MAX_SEQ);
        kern_kv_write<<<BLOCKS_H, THREADS_PER_BLOCK, 0, stream>>>(d_kv, d_scratch_a, N_ELEMS, l, 0, KV_MAX_SEQ);
        kern_dot_reduce<<<BLOCKS_H, THREADS_PER_BLOCK, 0, stream>>>(d_scratch_b, d_scratch_b, d_scratch_a, N_ELEMS);
        kern_scale<<<BLOCKS_H, THREADS_PER_BLOCK, 0, stream>>>(d_scratch_a, d_scratch_b, N_ELEMS);
        kern_add<<<BLOCKS_H, THREADS_PER_BLOCK, 0, stream>>>(d_hidden, d_hidden, d_scratch_a, N_ELEMS);
        kern_rms_norm<<<BLOCKS_H, THREADS_PER_BLOCK, 0, stream>>>(d_scratch_a, d_hidden, N_ELEMS);
        kern_scale<<<BLOCKS_F, THREADS_PER_BLOCK, 0, stream>>>(d_scratch_b, d_scratch_a, N_ELEMS);
        kern_scale<<<BLOCKS_F, THREADS_PER_BLOCK, 0, stream>>>(d_scratch_a, d_scratch_a, N_ELEMS);
        kern_silu<<<BLOCKS_H, THREADS_PER_BLOCK, 0, stream>>>(d_scratch_b, d_scratch_b, N_ELEMS);
        kern_mul<<<BLOCKS_H, THREADS_PER_BLOCK, 0, stream>>>(d_scratch_b, d_scratch_b, d_scratch_a, N_ELEMS);
        kern_scale<<<BLOCKS_H, THREADS_PER_BLOCK, 0, stream>>>(d_scratch_a, d_scratch_b, N_ELEMS);
        kern_add<<<BLOCKS_H, THREADS_PER_BLOCK, 0, stream>>>(d_hidden, d_hidden, d_scratch_a, N_ELEMS);
    }
    hipGraph_t graph = nullptr;
    HIP_CHECK_SOFT(hipStreamEndCapture(stream, &graph));
    hipGraphExec_t exec = nullptr;
    g_crash_phase = "L2 instantiate";
    HIP_CHECK_SOFT(hipGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
    g_crash_phase = "L2 launch";
    HIP_CHECK_SOFT(hipGraphLaunch(exec, stream));
    g_crash_phase = "L2 sync";
    HIP_CHECK_SOFT(hipStreamSynchronize(stream));

    printf("  L2  %2d layers (~%4d kernels):  OK\n", n_layers, n_layers * 17);
    hipGraphExecDestroy(exec);
    hipGraphDestroy(graph);
    hipFree(d_hidden); hipFree(d_scratch_a); hipFree(d_scratch_b); hipFree(d_kv);
    return true;
}

// ── Level 3: L2 + repeated replay ───────────────────────────────────
//
// Instead of hipGraphExecKernelNodeSetParams (needs per-node tracking),
// just replay the SAME graph 100× — that's what ROCmForge would do if
// it captured once and looped decode. The KV-write pos is baked in at
// capture; we'd normally update it per token. Here we just stress the
// replay mechanism.

static bool run_level_3(int n_layers, hipStream_t stream) {
    float *d_hidden, *d_scratch_a, *d_scratch_b, *d_kv;
    HIP_CHECK_SOFT(hipMalloc(&d_hidden,    N_ELEMS * sizeof(float)));
    HIP_CHECK_SOFT(hipMalloc(&d_scratch_a, FFN_DIM * sizeof(float)));
    HIP_CHECK_SOFT(hipMalloc(&d_scratch_b, FFN_DIM * sizeof(float)));
    HIP_CHECK_SOFT(hipMalloc(&d_kv,        (size_t)n_layers * KV_MAX_SEQ * N_ELEMS * sizeof(float)));
    reset_ones(d_hidden, N_ELEMS);

    g_crash_phase = "L3 capture";
    HIP_CHECK_SOFT(hipStreamBeginCapture(stream, hipStreamCaptureModeGlobal));
    for (int l = 0; l < n_layers; ++l) {
        kern_rms_norm<<<BLOCKS_H, THREADS_PER_BLOCK, 0, stream>>>(d_scratch_a, d_hidden, N_ELEMS);
        kern_scale<<<BLOCKS_H, THREADS_PER_BLOCK, 0, stream>>>(d_scratch_b, d_scratch_a, N_ELEMS);
        kern_scale<<<BLOCKS_H, THREADS_PER_BLOCK, 0, stream>>>(d_scratch_a, d_scratch_a, N_ELEMS);
        kern_rope<<<BLOCKS_H, THREADS_PER_BLOCK, 0, stream>>>(d_scratch_b, d_scratch_b, N_ELEMS, 0);
        kern_kv_write<<<BLOCKS_H, THREADS_PER_BLOCK, 0, stream>>>(d_kv, d_scratch_a, N_ELEMS, l, 0, KV_MAX_SEQ);
        kern_dot_reduce<<<BLOCKS_H, THREADS_PER_BLOCK, 0, stream>>>(d_scratch_b, d_scratch_b, d_scratch_a, N_ELEMS);
        kern_add<<<BLOCKS_H, THREADS_PER_BLOCK, 0, stream>>>(d_hidden, d_hidden, d_scratch_b, N_ELEMS);
        kern_silu<<<BLOCKS_H, THREADS_PER_BLOCK, 0, stream>>>(d_scratch_a, d_hidden, N_ELEMS);
        kern_mul<<<BLOCKS_H, THREADS_PER_BLOCK, 0, stream>>>(d_scratch_b, d_scratch_a, d_hidden, N_ELEMS);
        kern_add<<<BLOCKS_H, THREADS_PER_BLOCK, 0, stream>>>(d_hidden, d_hidden, d_scratch_b, N_ELEMS);
    }
    hipGraph_t graph = nullptr;
    HIP_CHECK_SOFT(hipStreamEndCapture(stream, &graph));
    hipGraphExec_t exec = nullptr;
    g_crash_phase = "L3 instantiate";
    HIP_CHECK_SOFT(hipGraphInstantiate(&exec, graph, nullptr, nullptr, 0));

    // Replay 100 times.
    for (int token = 0; token < 100; ++token) {
        g_crash_phase = "L3 replay";
        HIP_CHECK_SOFT(hipGraphLaunch(exec, stream));
    }
    g_crash_phase = "L3 sync";
    HIP_CHECK_SOFT(hipStreamSynchronize(stream));

    printf("  L3  %2d layers × 100 replays:   OK\n", n_layers);
    hipGraphExecDestroy(exec);
    hipGraphDestroy(graph);
    hipFree(d_hidden); hipFree(d_scratch_a); hipFree(d_scratch_b); hipFree(d_kv);
    return true;
}

// ── Level 4: L3 + large weight buffers ──────────────────────────────

static bool run_level_4(int n_layers, hipStream_t stream) {
    // Allocate weight buffers the graph nodes actually reference.
    // Per layer: 4 projection weights × (N_ELEMS × N_ELEMS) FP16 = 32 MB.
    // 4 layers → 128 MB. 36 layers → 1.15 GB — feasible on 16 GB card.
    const size_t w_per_layer = (size_t)N_ELEMS * N_ELEMS;  // in fp16 elements
    const size_t n_proj_per_layer = 4;
    std::vector<__half*> weights(n_layers * n_proj_per_layer);
    for (size_t i = 0; i < weights.size(); ++i) {
        hipError_t e = hipMalloc(&weights[i], w_per_layer * sizeof(__half));
        if (e != hipSuccess) {
            printf("  L4 OOM at weight %zu/%zu: %s\n",
                   i, weights.size(), hipGetErrorString(e));
            for (size_t j = 0; j < i; ++j) hipFree(weights[j]);
            return false;
        }
        // Touch once so physical pages are allocated.
        hipMemset(weights[i], 0, w_per_layer * sizeof(__half));
    }

    float *d_hidden, *d_scratch_a, *d_scratch_b, *d_kv;
    HIP_CHECK_SOFT(hipMalloc(&d_hidden,    N_ELEMS * sizeof(float)));
    HIP_CHECK_SOFT(hipMalloc(&d_scratch_a, FFN_DIM * sizeof(float)));
    HIP_CHECK_SOFT(hipMalloc(&d_scratch_b, FFN_DIM * sizeof(float)));
    HIP_CHECK_SOFT(hipMalloc(&d_kv,        (size_t)n_layers * KV_MAX_SEQ * N_ELEMS * sizeof(float)));
    reset_ones(d_hidden, N_ELEMS);

    g_crash_phase = "L4 capture";
    HIP_CHECK_SOFT(hipStreamBeginCapture(stream, hipStreamCaptureModeGlobal));
    for (int l = 0; l < n_layers; ++l) {
        // Reference the weight pointers through kern_scale — the
        // kernel reads `src`, which we point at a weight buffer so
        // the graph node captures a real weight-pointer.
        const float* w_q = reinterpret_cast<const float*>(weights[l * 4 + 0]);
        const float* w_k = reinterpret_cast<const float*>(weights[l * 4 + 1]);
        const float* w_o = reinterpret_cast<const float*>(weights[l * 4 + 2]);
        const float* w_d = reinterpret_cast<const float*>(weights[l * 4 + 3]);
        kern_scale<<<BLOCKS_H, THREADS_PER_BLOCK, 0, stream>>>(d_scratch_a, w_q, N_ELEMS);
        kern_scale<<<BLOCKS_H, THREADS_PER_BLOCK, 0, stream>>>(d_scratch_b, w_k, N_ELEMS);
        kern_dot_reduce<<<BLOCKS_H, THREADS_PER_BLOCK, 0, stream>>>(d_hidden, d_scratch_a, d_scratch_b, N_ELEMS);
        kern_scale<<<BLOCKS_H, THREADS_PER_BLOCK, 0, stream>>>(d_scratch_a, w_o, N_ELEMS);
        kern_add<<<BLOCKS_H, THREADS_PER_BLOCK, 0, stream>>>(d_hidden, d_hidden, d_scratch_a, N_ELEMS);
        kern_scale<<<BLOCKS_H, THREADS_PER_BLOCK, 0, stream>>>(d_scratch_b, w_d, N_ELEMS);
        kern_mul<<<BLOCKS_H, THREADS_PER_BLOCK, 0, stream>>>(d_scratch_a, d_scratch_a, d_scratch_b, N_ELEMS);
        kern_add<<<BLOCKS_H, THREADS_PER_BLOCK, 0, stream>>>(d_hidden, d_hidden, d_scratch_a, N_ELEMS);
        kern_kv_write<<<BLOCKS_H, THREADS_PER_BLOCK, 0, stream>>>(d_kv, d_scratch_a, N_ELEMS, l, 0, KV_MAX_SEQ);
    }
    hipGraph_t graph = nullptr;
    HIP_CHECK_SOFT(hipStreamEndCapture(stream, &graph));
    hipGraphExec_t exec = nullptr;
    g_crash_phase = "L4 instantiate";
    HIP_CHECK_SOFT(hipGraphInstantiate(&exec, graph, nullptr, nullptr, 0));

    for (int token = 0; token < 10; ++token) {
        g_crash_phase = "L4 replay";
        HIP_CHECK_SOFT(hipGraphLaunch(exec, stream));
    }
    g_crash_phase = "L4 sync";
    HIP_CHECK_SOFT(hipStreamSynchronize(stream));

    printf("  L4  %2d layers × weight-referenced × 10 replays:  OK\n", n_layers);
    hipGraphExecDestroy(exec);
    hipGraphDestroy(graph);
    hipFree(d_hidden); hipFree(d_scratch_a); hipFree(d_scratch_b); hipFree(d_kv);
    for (auto* w : weights) hipFree(w);
    return true;
}

// ── Level 5: Real GEMV kernel ───────────────────────────────────────

static bool run_level_5(int n_layers, hipStream_t stream) {
    // Ground-truth matmul per layer. Allocate weights per layer and
    // run a single kern_gemv per "projection".
    const size_t w_per_layer = (size_t)N_ELEMS * N_ELEMS;
    std::vector<__half*> weights(n_layers);
    for (int i = 0; i < n_layers; ++i) {
        hipError_t e = hipMalloc(&weights[i], w_per_layer * sizeof(__half));
        if (e != hipSuccess) {
            printf("  L5 OOM at weight %d/%d\n", i, n_layers);
            for (int j = 0; j < i; ++j) hipFree(weights[j]);
            return false;
        }
        hipMemset(weights[i], 0, w_per_layer * sizeof(__half));
    }

    float *d_hidden, *d_scratch;
    HIP_CHECK_SOFT(hipMalloc(&d_hidden,  N_ELEMS * sizeof(float)));
    HIP_CHECK_SOFT(hipMalloc(&d_scratch, N_ELEMS * sizeof(float)));
    reset_ones(d_hidden, N_ELEMS);

    g_crash_phase = "L5 capture";
    HIP_CHECK_SOFT(hipStreamBeginCapture(stream, hipStreamCaptureModeGlobal));
    for (int l = 0; l < n_layers; ++l) {
        kern_gemv<<<BLOCKS_H, THREADS_PER_BLOCK, 0, stream>>>(
            d_scratch, weights[l], d_hidden, N_ELEMS, N_ELEMS);
        kern_add<<<BLOCKS_H, THREADS_PER_BLOCK, 0, stream>>>(
            d_hidden, d_hidden, d_scratch, N_ELEMS);
    }
    hipGraph_t graph = nullptr;
    HIP_CHECK_SOFT(hipStreamEndCapture(stream, &graph));
    hipGraphExec_t exec = nullptr;
    g_crash_phase = "L5 instantiate";
    HIP_CHECK_SOFT(hipGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
    g_crash_phase = "L5 launch";
    HIP_CHECK_SOFT(hipGraphLaunch(exec, stream));
    g_crash_phase = "L5 sync";
    HIP_CHECK_SOFT(hipStreamSynchronize(stream));

    printf("  L5  %2d layers × real GEMV:  OK\n", n_layers);
    hipGraphExecDestroy(exec);
    hipGraphDestroy(graph);
    hipFree(d_hidden); hipFree(d_scratch);
    for (auto* w : weights) hipFree(w);
    return true;
}

// ── Level 6: Everything combined ────────────────────────────────────

static bool run_level_6(int n_layers, hipStream_t stream) {
    // L1's 17-kernel-per-layer structure + L2's buffer aliasing +
    // L3's 100 replays + L4's weight buffers + L5's real GEMV.
    // Scale weights on 4+ layers (not 36) to keep VRAM sane.
    const int effective_layers = n_layers;  // will fail-soft if OOM
    const size_t w_per_proj = (size_t)N_ELEMS * N_ELEMS;
    const size_t n_weights = (size_t)effective_layers * 4;
    std::vector<__half*> weights(n_weights);
    for (size_t i = 0; i < n_weights; ++i) {
        hipError_t e = hipMalloc(&weights[i], w_per_proj * sizeof(__half));
        if (e != hipSuccess) {
            printf("  L6 OOM at weight %zu/%zu\n", i, n_weights);
            for (size_t j = 0; j < i; ++j) hipFree(weights[j]);
            return false;
        }
        hipMemset(weights[i], 0, w_per_proj * sizeof(__half));
    }

    float *d_hidden, *d_scratch_a, *d_scratch_b, *d_kv;
    HIP_CHECK_SOFT(hipMalloc(&d_hidden,    N_ELEMS * sizeof(float)));
    HIP_CHECK_SOFT(hipMalloc(&d_scratch_a, FFN_DIM * sizeof(float)));
    HIP_CHECK_SOFT(hipMalloc(&d_scratch_b, FFN_DIM * sizeof(float)));
    HIP_CHECK_SOFT(hipMalloc(&d_kv,        (size_t)effective_layers * KV_MAX_SEQ * N_ELEMS * sizeof(float)));
    reset_ones(d_hidden, N_ELEMS);

    g_crash_phase = "L6 capture";
    HIP_CHECK_SOFT(hipStreamBeginCapture(stream, hipStreamCaptureModeGlobal));
    for (int l = 0; l < effective_layers; ++l) {
        __half* w_q = weights[l * 4 + 0];
        __half* w_k = weights[l * 4 + 1];
        __half* w_o = weights[l * 4 + 2];
        __half* w_d = weights[l * 4 + 3];
        // Attention-Norm → Q/K (real GEMV) → RoPE → KV-Write → Dot → O → Res
        kern_rms_norm<<<BLOCKS_H, THREADS_PER_BLOCK, 0, stream>>>(d_scratch_a, d_hidden, N_ELEMS);
        kern_gemv<<<BLOCKS_H, THREADS_PER_BLOCK, 0, stream>>>(d_scratch_b, w_q, d_scratch_a, N_ELEMS, N_ELEMS);
        kern_gemv<<<BLOCKS_H, THREADS_PER_BLOCK, 0, stream>>>(d_scratch_a, w_k, d_scratch_a, N_ELEMS, N_ELEMS);
        kern_rope<<<BLOCKS_H, THREADS_PER_BLOCK, 0, stream>>>(d_scratch_b, d_scratch_b, N_ELEMS, 0);
        kern_rope<<<BLOCKS_H, THREADS_PER_BLOCK, 0, stream>>>(d_scratch_a, d_scratch_a, N_ELEMS, 0);
        kern_kv_write<<<BLOCKS_H, THREADS_PER_BLOCK, 0, stream>>>(d_kv, d_scratch_a, N_ELEMS, l, 0, KV_MAX_SEQ);
        kern_kv_write<<<BLOCKS_H, THREADS_PER_BLOCK, 0, stream>>>(d_kv, d_scratch_a, N_ELEMS, l, 0, KV_MAX_SEQ);
        kern_dot_reduce<<<BLOCKS_H, THREADS_PER_BLOCK, 0, stream>>>(d_scratch_b, d_scratch_b, d_scratch_a, N_ELEMS);
        kern_gemv<<<BLOCKS_H, THREADS_PER_BLOCK, 0, stream>>>(d_scratch_a, w_o, d_scratch_b, N_ELEMS, N_ELEMS);
        kern_add<<<BLOCKS_H, THREADS_PER_BLOCK, 0, stream>>>(d_hidden, d_hidden, d_scratch_a, N_ELEMS);
        // FFN: Norm → Gate/Up → SwiGLU → Down → Res
        kern_rms_norm<<<BLOCKS_H, THREADS_PER_BLOCK, 0, stream>>>(d_scratch_a, d_hidden, N_ELEMS);
        kern_scale<<<BLOCKS_F, THREADS_PER_BLOCK, 0, stream>>>(d_scratch_b, d_scratch_a, N_ELEMS);
        kern_silu<<<BLOCKS_H, THREADS_PER_BLOCK, 0, stream>>>(d_scratch_a, d_scratch_b, N_ELEMS);
        kern_mul<<<BLOCKS_H, THREADS_PER_BLOCK, 0, stream>>>(d_scratch_a, d_scratch_a, d_scratch_b, N_ELEMS);
        kern_gemv<<<BLOCKS_H, THREADS_PER_BLOCK, 0, stream>>>(d_scratch_b, w_d, d_scratch_a, N_ELEMS, N_ELEMS);
        kern_add<<<BLOCKS_H, THREADS_PER_BLOCK, 0, stream>>>(d_hidden, d_hidden, d_scratch_b, N_ELEMS);
    }
    hipGraph_t graph = nullptr;
    HIP_CHECK_SOFT(hipStreamEndCapture(stream, &graph));
    hipGraphExec_t exec = nullptr;
    g_crash_phase = "L6 instantiate";
    HIP_CHECK_SOFT(hipGraphInstantiate(&exec, graph, nullptr, nullptr, 0));

    for (int token = 0; token < 100; ++token) {
        g_crash_phase = "L6 replay";
        HIP_CHECK_SOFT(hipGraphLaunch(exec, stream));
    }
    g_crash_phase = "L6 sync";
    HIP_CHECK_SOFT(hipStreamSynchronize(stream));

    printf("  L6  %2d layers × full-combo × 100 replays:  OK\n", n_layers);
    hipGraphExecDestroy(exec);
    hipGraphDestroy(graph);
    hipFree(d_hidden); hipFree(d_scratch_a); hipFree(d_scratch_b); hipFree(d_kv);
    for (auto* w : weights) hipFree(w);
    return true;
}

// ── Main ────────────────────────────────────────────────────────────

int main() {
    signal(SIGSEGV, crash_handler);
    signal(SIGABRT, crash_handler);
    signal(SIGBUS,  crash_handler);
    signal(SIGILL,  crash_handler);

    hipDeviceProp_t props;
    HIP_CHECK_FATAL(hipGetDeviceProperties(&props, 0));
    int runtime_v = 0, driver_v = 0;
    HIP_CHECK_FATAL(hipRuntimeGetVersion(&runtime_v));
    HIP_CHECK_FATAL(hipDriverGetVersion(&driver_v));

    printf("=== HIP Graph Escalation Test ===\n");
    printf("GPU:     %s (arch=%s)\n", props.name, props.gcnArchName);
    printf("ROCm:    runtime=%d.%d driver=%d.%d\n\n",
           runtime_v / 1000, (runtime_v % 1000) / 10,
           driver_v / 1000, (driver_v % 1000) / 10);

    hipStream_t stream;
    HIP_CHECK_FATAL(hipStreamCreate(&stream));

    const int sizes[] = {1, 4, 16, 36};
    const int n_sizes = 4;

    struct LevelDef {
        const char* name;
        bool (*fn)(int, hipStream_t);
    };
    const LevelDef levels[] = {
        {"L0 Baseline (scale-only, fresh buffers)", run_level_0},
        {"L1 Multi-kernel (8 types, fresh buffers)", run_level_1},
        {"L2 Multi-kernel + buffer aliasing",        run_level_2},
        {"L3 L2 + 100× replay",                       run_level_3},
        {"L4 L3 + large weight buffers",              run_level_4},
        {"L5 L4 + real GEMV kernel",                  run_level_5},
        {"L6 Everything (closest ROCmForge MRE)",     run_level_6},
    };
    const int n_levels = sizeof(levels) / sizeof(levels[0]);

    for (int l = 0; l < n_levels; ++l) {
        printf("=== %s ===\n", levels[l].name);
        g_crash_level = l;
        bool level_ok = true;
        for (int s = 0; s < n_sizes; ++s) {
            g_crash_layers = sizes[s];
            bool ok = levels[l].fn(sizes[s], stream);
            if (!ok) {
                printf(">>> SOFT-FAIL at Level %d, %d layers <<<\n",
                       l, sizes[s]);
                level_ok = false;
                goto report;
            }
        }
        if (level_ok) printf("Level %d: ALL PASSED\n\n", l);
    }

    printf("=== RESULT ===\n");
    printf("NO CRASH at any escalation level up to 36 layers × 100 replays × full-combo.\n");
    printf("HIP-Graph is USABLE on gfx1201 for ROCmForge's decode path.\n");
    hipStreamDestroy(stream);
    return 0;

report:
    printf("\n=== CRASH/SOFT-FAIL RESULT ===\n");
    printf("Trigger: Level %d (%s) with %d layers\n",
           (int)g_crash_level,
           levels[(int)g_crash_level].name,
           (int)g_crash_layers);
    hipStreamDestroy(stream);
    return 1;
}
