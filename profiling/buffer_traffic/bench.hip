// Intermediate-Buffer-Traffic Micro-Benchmark
//
// Purpose: quantify how much time is spent on VRAM roundtrips for FFN
// intermediate buffers (~18944-wide) vs. keeping the intermediate in LDS/
// registers. We approximate the Qwen2.5-7B FFN dimensions with a clean
// 5:1 ratio (INTERMEDIATE = 5 * HIDDEN = 17920) so that all three variants
// use identical reduction patterns and only the memory-placement of the
// intermediate differs.
//
// Variants per "layer" (28 layers ≈ one verify-step):
//   A  — 4 separate kernels, intermediate materialized in VRAM.
//        Full grid parallelism each dispatch. Matches the current FFN path.
//   B2 — 1 cooperative kernel, same VRAM traffic as A, same grid parallelism.
//        grid.sync() between phases. Isolates dispatch/sync overhead.
//   B1 — 1 kernel, intermediate never materialized.  dim3(batch*NUM_CHUNKS)
//        blocks, each block owns CHUNK contiguous output elements and
//        computes them directly from input (registers only). Isolates the
//        VRAM-roundtrip cost of the intermediate buffer.
//
// Deltas:
//   Δ(A-B2)  = dispatch + sync overhead (same work, one vs. four launches)
//   Δ(B2-B1) = intermediate VRAM-roundtrip cost (same launch count)
//   Δ(A-B1)  = maximum total gain from a perfect fused FFN
//
// All kernels are pure elementwise with 1-2 FMAs per element. Absolute
// timings do NOT match real FFN wall-clock — only the deltas are meaningful.

#include <hip/hip_runtime.h>
#include <hip/hip_cooperative_groups.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <cmath>
#include <algorithm>
#include <string>

namespace cg = cooperative_groups;

#define HIP_CHECK(x) do {                                            \
    hipError_t _e = (x);                                             \
    if (_e != hipSuccess) {                                          \
        fprintf(stderr, "HIP error %s at %s:%d\n",                   \
                hipGetErrorString(_e), __FILE__, __LINE__);          \
        std::exit(1);                                                \
    }                                                                \
} while (0)

constexpr int HIDDEN = 3584;
constexpr int RATIO = 5;                         // ≈ 18944/3584 rounded down
constexpr int INTERMEDIATE = HIDDEN * RATIO;     // = 17920
constexpr int N_LAYERS = 28;
constexpr int BLOCK = 256;
constexpr int B1_BLOCK = 128;                    // block size for B1 (matches CHUNK)
constexpr int CHUNK = 128;                       // outputs per block in B1
static_assert(HIDDEN % CHUNK == 0, "HIDDEN must be divisible by CHUNK");

// ---------- Variant A: four separate kernels (status quo) ----------

__device__ inline float fma_chain(float x, int depth) {
    #pragma unroll 4
    for (int k = 0; k < depth; k++) {
        x = x * 0.99f + 0.01f;
    }
    return x;
}

__global__ void A_gate_up(const float* __restrict__ input,
                          float* __restrict__ intermediate,
                          int batch, int fma_depth) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * INTERMEDIATE;
    if (idx >= total) return;
    int b = idx / INTERMEDIATE;
    int i = idx % INTERMEDIATE;
    float x = input[b * HIDDEN + (i % HIDDEN)];
    intermediate[idx] = fma_chain(x, fma_depth);
}

__global__ void A_mul(float* __restrict__ intermediate, int batch, int fma_depth) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * INTERMEDIATE;
    if (idx >= total) return;
    float x = intermediate[idx];
    intermediate[idx] = fma_chain(x, fma_depth);
}

__global__ void A_down(const float* __restrict__ intermediate,
                       float* __restrict__ output,
                       int batch, int fma_depth) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * HIDDEN;
    if (idx >= total) return;
    int b = idx / HIDDEN;
    int i = idx % HIDDEN;
    float acc = 0.0f;
    #pragma unroll
    for (int k = 0; k < RATIO; k++) {
        acc += intermediate[b * INTERMEDIATE + i * RATIO + k];
    }
    output[idx] = fma_chain(acc, fma_depth);
}

__global__ void A_residual(float* __restrict__ output,
                           const float* __restrict__ residual,
                           int batch, int fma_depth) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * HIDDEN;
    if (idx >= total) return;
    float x = output[idx];
    output[idx] = fma_chain(x, fma_depth) + residual[idx];
}

// ---------- Variant B2: cooperative, same VRAM traffic ----------

__global__ void B2_fused_vram_coop(const float* __restrict__ input,
                                   float* __restrict__ intermediate,
                                   float* __restrict__ output,
                                   const float* __restrict__ residual,
                                   int batch, int fma_depth) {
    cg::grid_group grid = cg::this_grid();
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int gstride = gridDim.x * blockDim.x;

    int total_int = batch * INTERMEDIATE;
    int total_hid = batch * HIDDEN;

    // Phase 1: gate_up
    for (int idx = gid; idx < total_int; idx += gstride) {
        int b = idx / INTERMEDIATE;
        int i = idx % INTERMEDIATE;
        float x = input[b * HIDDEN + (i % HIDDEN)];
        intermediate[idx] = fma_chain(x, fma_depth);
    }
    grid.sync();

    // Phase 2: mul
    for (int idx = gid; idx < total_int; idx += gstride) {
        float x = intermediate[idx];
        intermediate[idx] = fma_chain(x, fma_depth);
    }
    grid.sync();

    // Phase 3: down
    for (int idx = gid; idx < total_hid; idx += gstride) {
        int b = idx / HIDDEN;
        int i = idx % HIDDEN;
        float acc = 0.0f;
        #pragma unroll
        for (int k = 0; k < RATIO; k++) {
            acc += intermediate[b * INTERMEDIATE + i * RATIO + k];
        }
        output[idx] = fma_chain(acc, fma_depth);
    }
    grid.sync();

    // Phase 4: residual
    for (int idx = gid; idx < total_hid; idx += gstride) {
        float x = output[idx];
        output[idx] = fma_chain(x, fma_depth) + residual[idx];
    }
}

// ---------- Variant B1: chunked fused (LDS/register-only intermediate) ----
//
// dim3(batch * HIDDEN/CHUNK) blocks. Each block handles CHUNK contiguous
// output elements for one batch. Intermediate values are computed on the
// fly in registers, never written to VRAM.

__global__ void B1_fused_chunked(const float* __restrict__ input,
                                 float* __restrict__ output,
                                 const float* __restrict__ residual,
                                 int batch, int fma_depth) {
    int blocks_per_batch = HIDDEN / CHUNK;
    int b = blockIdx.x / blocks_per_batch;
    int chunk = blockIdx.x % blocks_per_batch;
    int chunk_start = chunk * CHUNK;
    int tid = threadIdx.x;

    for (int c = tid; c < CHUNK; c += blockDim.x) {
        int i = chunk_start + c;
        float acc = 0.0f;
        #pragma unroll
        for (int k = 0; k < RATIO; k++) {
            int j = i * RATIO + k;
            float x = input[b * HIDDEN + (j % HIDDEN)];
            float mid = fma_chain(x, fma_depth);    // gate_up
            mid = fma_chain(mid, fma_depth);        // mul
            acc += mid;
        }
        float down = fma_chain(acc, fma_depth);     // down
        output[b * HIDDEN + i] = fma_chain(down, fma_depth) + residual[b * HIDDEN + i];
    }
}

// ---------- Runner ----------

struct Stats {
    double mean;
    double median;
    double p95;
    double stdev;
};

static Stats compute_stats(std::vector<double> samples) {
    std::sort(samples.begin(), samples.end());
    size_t n = samples.size();
    double sum = 0.0;
    for (double s : samples) sum += s;
    double mean = sum / n;
    double median = samples[n / 2];
    double p95 = samples[(size_t)((n - 1) * 0.95)];
    double var = 0.0;
    for (double s : samples) var += (s - mean) * (s - mean);
    double stdev = std::sqrt(var / n);
    return {mean, median, p95, stdev};
}

enum class Variant { A, B1, B2 };

static int g_b2_coop_grid = 0;  // set per-batch

static double run_once(Variant v, int batch, int fma_depth,
                       float* d_input,
                       float* d_intermediate,
                       float* d_output,
                       float* d_residual,
                       hipEvent_t start, hipEvent_t stop) {
    HIP_CHECK(hipEventRecord(start, 0));

    for (int layer = 0; layer < N_LAYERS; ++layer) {
        if (v == Variant::A) {
            int total_int = batch * INTERMEDIATE;
            int total_hid = batch * HIDDEN;
            int g_int = (total_int + BLOCK - 1) / BLOCK;
            int g_hid = (total_hid + BLOCK - 1) / BLOCK;
            hipLaunchKernelGGL(A_gate_up, dim3(g_int), dim3(BLOCK), 0, 0,
                               d_input, d_intermediate, batch, fma_depth);
            hipLaunchKernelGGL(A_mul, dim3(g_int), dim3(BLOCK), 0, 0,
                               d_intermediate, batch, fma_depth);
            hipLaunchKernelGGL(A_down, dim3(g_hid), dim3(BLOCK), 0, 0,
                               d_intermediate, d_output, batch, fma_depth);
            hipLaunchKernelGGL(A_residual, dim3(g_hid), dim3(BLOCK), 0, 0,
                               d_output, d_residual, batch, fma_depth);
        } else if (v == Variant::B2) {
            void* args[] = {&d_input, &d_intermediate, &d_output, &d_residual,
                            (void*)&batch, (void*)&fma_depth};
            HIP_CHECK(hipLaunchCooperativeKernel(
                (void*)B2_fused_vram_coop,
                dim3(g_b2_coop_grid), dim3(BLOCK), args, 0, 0));
        } else {  // B1
            int grid = batch * (HIDDEN / CHUNK);
            hipLaunchKernelGGL(B1_fused_chunked, dim3(grid), dim3(B1_BLOCK), 0, 0,
                               d_input, d_output, d_residual, batch, fma_depth);
        }
    }

    HIP_CHECK(hipEventRecord(stop, 0));
    HIP_CHECK(hipEventSynchronize(stop));
    float ms = 0.0f;
    HIP_CHECK(hipEventElapsedTime(&ms, start, stop));
    return ms * 1000.0;
}

static Stats measure(Variant v, int batch, int fma_depth, int warmup, int iters,
                     float* d_input, float* d_intermediate,
                     float* d_output, float* d_residual) {
    hipEvent_t start, stop;
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));

    for (int i = 0; i < warmup; ++i) {
        run_once(v, batch, fma_depth, d_input, d_intermediate, d_output, d_residual,
                 start, stop);
    }

    std::vector<double> samples;
    samples.reserve(iters);
    for (int i = 0; i < iters; ++i) {
        samples.push_back(run_once(v, batch, fma_depth, d_input, d_intermediate,
                                   d_output, d_residual, start, stop));
    }

    HIP_CHECK(hipEventDestroy(start));
    HIP_CHECK(hipEventDestroy(stop));
    return compute_stats(samples);
}

static void print_stats(const char* name, const Stats& s) {
    fprintf(stderr, "  %-28s mean=%8.2f  median=%8.2f  p95=%8.2f  stdev=%7.2f us\n",
            name, s.mean, s.median, s.p95, s.stdev);
}

static void emit_stats_json(FILE* out, const Stats& s) {
    fprintf(out, "{\"mean\":%.2f,\"median\":%.2f,\"p95\":%.2f,\"stdev\":%.2f}",
            s.mean, s.median, s.p95, s.stdev);
}

struct BatchResult {
    int batch;
    int fma_depth;
    Stats a;
    Stats b2;
    Stats b1;
    int b2_grid;
};

static int choose_coop_grid(int batch) {
    // Pick a grid size big enough to cover batch * INTERMEDIATE with good
    // occupancy, but small enough to satisfy cooperative-launch constraints.
    int total_int = batch * INTERMEDIATE;
    int desired = (total_int + BLOCK - 1) / BLOCK;

    int dev = 0;
    int max_blocks_per_sm = 0;
    int num_sms = 0;
    HIP_CHECK(hipDeviceGetAttribute(&num_sms, hipDeviceAttributeMultiprocessorCount, dev));
    hipOccupancyMaxActiveBlocksPerMultiprocessor(
        &max_blocks_per_sm, (void*)B2_fused_vram_coop, BLOCK, 0);
    int coop_cap = num_sms * std::max(max_blocks_per_sm, 1);
    int grid = std::min(desired, coop_cap);
    if (grid < 1) grid = 1;
    return grid;
}

static BatchResult run_batch(int batch, int fma_depth, int warmup, int iters) {
    size_t input_sz     = (size_t)batch * HIDDEN * sizeof(float);
    size_t intermed_sz  = (size_t)batch * INTERMEDIATE * sizeof(float);
    size_t output_sz    = (size_t)batch * HIDDEN * sizeof(float);
    size_t residual_sz  = (size_t)batch * HIDDEN * sizeof(float);

    float *d_input, *d_intermediate, *d_output, *d_residual;
    HIP_CHECK(hipMalloc(&d_input, input_sz));
    HIP_CHECK(hipMalloc(&d_intermediate, intermed_sz));
    HIP_CHECK(hipMalloc(&d_output, output_sz));
    HIP_CHECK(hipMalloc(&d_residual, residual_sz));

    std::vector<float> hinp(batch * HIDDEN), hres(batch * HIDDEN);
    for (int i = 0; i < batch * HIDDEN; i++) {
        hinp[i] = ((i * 1103515245 + 12345) & 0xffff) / 65535.0f - 0.5f;
        hres[i] = ((i * 2654435769u + 7) & 0xffff) / 65535.0f - 0.5f;
    }
    HIP_CHECK(hipMemcpy(d_input, hinp.data(), input_sz, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_residual, hres.data(), residual_sz, hipMemcpyHostToDevice));

    g_b2_coop_grid = choose_coop_grid(batch);
    fprintf(stderr, "\n[batch=%d fma_depth=%d] warmup=%d iters=%d  (B2 coop grid=%d)\n",
            batch, fma_depth, warmup, iters, g_b2_coop_grid);

    Stats a  = measure(Variant::A,  batch, fma_depth, warmup, iters, d_input, d_intermediate, d_output, d_residual);
    print_stats("A  (separate kernels)",     a);
    Stats b2 = measure(Variant::B2, batch, fma_depth, warmup, iters, d_input, d_intermediate, d_output, d_residual);
    print_stats("B2 (coop kernel, VRAM)",    b2);
    Stats b1 = measure(Variant::B1, batch, fma_depth, warmup, iters, d_input, d_intermediate, d_output, d_residual);
    print_stats("B1 (chunked fused, LDS)",   b1);

    fprintf(stderr, "  Δ(A-B2) dispatch overhead : %.2f us\n", a.median - b2.median);
    fprintf(stderr, "  Δ(B2-B1) buffer traffic   : %.2f us\n", b2.median - b1.median);
    fprintf(stderr, "  Δ(A-B1) max fused gain    : %.2f us\n", a.median - b1.median);

    HIP_CHECK(hipFree(d_input));
    HIP_CHECK(hipFree(d_intermediate));
    HIP_CHECK(hipFree(d_output));
    HIP_CHECK(hipFree(d_residual));

    return {batch, fma_depth, a, b2, b1, g_b2_coop_grid};
}

static void emit_batch_json(FILE* out, const char* key, const BatchResult& br) {
    fprintf(out, "  \"%s\": {\n", key);
    fprintf(out, "    \"batch\": %d,\n", br.batch);
    fprintf(out, "    \"fma_depth\": %d,\n", br.fma_depth);
    fprintf(out, "    \"b2_coop_grid\": %d,\n", br.b2_grid);
    fprintf(out, "    \"variant_a_separate_us\": ");
    emit_stats_json(out, br.a); fprintf(out, ",\n");
    fprintf(out, "    \"variant_b2_single_dispatch_us\": ");
    emit_stats_json(out, br.b2); fprintf(out, ",\n");
    fprintf(out, "    \"variant_b1_tiled_fused_us\": ");
    emit_stats_json(out, br.b1); fprintf(out, ",\n");
    double d_ab2 = br.a.median - br.b2.median;
    double d_b2b1 = br.b2.median - br.b1.median;
    double d_ab1 = br.a.median - br.b1.median;
    double pct = br.a.median > 0 ? (d_ab1 / br.a.median) * 100.0 : 0.0;
    fprintf(out, "    \"delta_a_minus_b2_us\": %.2f,\n", d_ab2);
    fprintf(out, "    \"delta_b2_minus_b1_us\": %.2f,\n", d_b2b1);
    fprintf(out, "    \"delta_a_minus_b1_us\": %.2f,\n", d_ab1);
    fprintf(out, "    \"buffer_traffic_percent_of_a\": %.2f\n", pct);
    fprintf(out, "  }");
}

int main(int argc, char** argv) {
    int warmup = 10;
    int iters = 100;
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--warmup" && i + 1 < argc) {
            warmup = std::atoi(argv[++i]);
        } else if (std::string(argv[i]) == "--iters" && i + 1 < argc) {
            iters = std::atoi(argv[++i]);
        }
    }

    int dev = 0;
    HIP_CHECK(hipSetDevice(dev));
    hipDeviceProp_t prop;
    HIP_CHECK(hipGetDeviceProperties(&prop, dev));
    fprintf(stderr, "Device: %s (arch=%s, cooperativeLaunch=%d)\n",
            prop.name, prop.gcnArchName, prop.cooperativeLaunch);

    // Sweep over (batch, fma_depth) to separate dispatch overhead
    // (visible at low fma_depth) from buffer-traffic cost (visible at
    // high fma_depth, where kernels are compute-bound).
    struct Cfg { int batch; int depth; const char* key; };
    Cfg configs[] = {
        {2,    1, "batch2_depth1"},
        {2,   16, "batch2_depth16"},
        {2,  128, "batch2_depth128"},
        {2, 1024, "batch2_depth1024"},
        {4,    1, "batch4_depth1"},
        {4,   16, "batch4_depth16"},
        {4,  128, "batch4_depth128"},
        {4, 1024, "batch4_depth1024"},
    };
    const int n_configs = sizeof(configs) / sizeof(configs[0]);

    std::vector<BatchResult> results;
    results.reserve(n_configs);
    for (const auto& c : configs) {
        results.push_back(run_batch(c.batch, c.depth, warmup, iters));
    }

    fprintf(stdout, "{\n");
    fprintf(stdout, "  \"device\": \"%s\",\n", prop.name);
    fprintf(stdout, "  \"arch\": \"%s\",\n", prop.gcnArchName);
    fprintf(stdout, "  \"cooperative_launch\": %d,\n", prop.cooperativeLaunch);
    fprintf(stdout, "  \"config\": {\n");
    fprintf(stdout, "    \"hidden_size\": %d,\n", HIDDEN);
    fprintf(stdout, "    \"intermediate_size\": %d,\n", INTERMEDIATE);
    fprintf(stdout, "    \"ratio\": %d,\n", RATIO);
    fprintf(stdout, "    \"n_layers\": %d,\n", N_LAYERS);
    fprintf(stdout, "    \"chunk\": %d,\n", CHUNK);
    fprintf(stdout, "    \"block\": %d,\n", BLOCK);
    fprintf(stdout, "    \"b1_block\": %d,\n", B1_BLOCK);
    fprintf(stdout, "    \"warmup\": %d,\n", warmup);
    fprintf(stdout, "    \"iters\": %d,\n", iters);
    fprintf(stdout, "    \"element_type\": \"f32\"\n");
    fprintf(stdout, "  },\n");
    for (int i = 0; i < (int)results.size(); ++i) {
        emit_batch_json(stdout, configs[i].key, results[i]);
        fprintf(stdout, i + 1 < (int)results.size() ? ",\n" : "\n");
    }
    fprintf(stdout, "}\n");

    return 0;
}
