// Phase 2.2A — FP8-E5M2 (OCP bf8) KV-cache kernels.
//
// The default FP32 KV-cache (see `hip_kernels_v1/attention/attention.hip`)
// burns 4 bytes per element. Switching to FP8-E5M2 cuts that to 1 byte
// per element → 4× more context at the same VRAM budget.
//
// Why E5M2 (not E4M3):
//   * E5M2 range ±57344 — comfortably above any attention score at
//     decode-time context lengths we care about (seq_len up to 50k
//     produces scores in ±10..100 range).
//   * E4M3 range ±448 is too tight for attention scores on long runs.
//   * Industry standard (GPT-4 KV-cache, AMD MI300X, NVIDIA H100).
//
// AMD / LLVM names this format `bf8` (the OCP name):
//   __builtin_amdgcn_cvt_pk_bf8_f32(a, b, old, word_sel)
//     packs two FP32 into a 32-bit int holding two bf8 bytes
//     (plus two bytes of `old`). For a single conversion we pass
//     `b=0.0f, old=0, word_sel=false` and take the low byte.
//   __builtin_amdgcn_cvt_f32_bf8(packed, byte_sel)
//     extracts byte `byte_sel` ∈ [0,3] and converts to FP32.
//
// Validated on gfx1201 — see `results/fp8_smoke_results.md`.

#include <hip/hip_runtime.h>
#include <stdint.h>

__device__ __forceinline__ void rf_v1_bf8_set_ieee_denormal_mode() {
    asm volatile("s_setreg_imm32_b32 hwreg(HW_REG_MODE, 4, 4), 0xF");
}

// Single-value FP32 → bf8 byte. The intrinsic produces a 32-bit packed
// value; we take the low byte of the low bf8. Faster alternatives that
// write 4 elements per thread are not worth the indexing cost for the
// kv_cache_append pattern (1 element per thread, head-major layout).
__device__ __forceinline__ uint8_t rf_v1_f32_to_bf8_byte(float x) {
    const int packed = __builtin_amdgcn_cvt_pk_bf8_f32(x, 0.0f, 0, false);
    return (uint8_t)(packed & 0xFF);
}

// ─── kv_cache_append (FP32 in → BF8 cache byte) ─────────────────────────────

__global__ void rf_v1_kv_cache_append_fp8_kernel(
    uint8_t*     __restrict__ k_cache,      // bf8 cache, byte-strided
    uint8_t*     __restrict__ v_cache,
    const float* __restrict__ k_new,
    const float* __restrict__ v_new,
    int num_kv_heads,
    int head_dim,
    int pos,
    int head_stride)
{
    const int head = blockIdx.y;
    const int d    = blockIdx.x * blockDim.x + threadIdx.x;
    if (head >= num_kv_heads || d >= head_dim) return;

    const size_t dst = (size_t)head * head_stride + (size_t)pos * head_dim + d;
    const size_t src = (size_t)head * head_dim + d;

    k_cache[dst] = rf_v1_f32_to_bf8_byte(k_new[src]);
    v_cache[dst] = rf_v1_f32_to_bf8_byte(v_new[src]);
}

extern "C" hipError_t rocmforge_launch_kv_cache_append_fp8(
    void*        k_cache,
    void*        v_cache,
    const float* k_new,
    const float* v_new,
    int num_kv_heads,
    int head_dim,
    int pos,
    int head_stride,
    hipStream_t stream)
{
    if (num_kv_heads <= 0 || head_dim <= 0 || pos < 0) return hipErrorInvalidValue;
    const int threads = 128;
    dim3 grid((head_dim + threads - 1) / threads, num_kv_heads, 1);
    rf_v1_kv_cache_append_fp8_kernel<<<grid, threads, 0, stream>>>(
        reinterpret_cast<uint8_t*>(k_cache),
        reinterpret_cast<uint8_t*>(v_cache),
        k_new, v_new, num_kv_heads, head_dim, pos, head_stride);
    return hipGetLastError();
}

// ─── attention_decode variant that reads FP8 KV-cache ──────────────────────
//
// Same online-softmax algorithm as the FP32 attention_decode kernel
// (`hip_kernels_v1/attention/attention.hip`). Only difference: K and V
// are read from byte-strided bf8 storage and converted to FP32 inline
// via `__builtin_amdgcn_cvt_f32_bf8`. The scores tensor (in LDS) is
// still FP32 — the conversion happens at the dot-product level where
// it's essentially free (compiler folds it into the VALU).
//
// Alignment: our hipMalloc'd cache buffers are 256-byte aligned and
// `head_dim` is always divisible by 4 on Qwen3-8B (128 = 32×4). So
// the byte offset `j*head_dim + d` has the same low 2 bits as `d`,
// and a 4-byte-aligned load at `&cache[base + (d & ~3)]` gives us the
// int that contains our byte.

__global__ void rf_v1_attention_decode_fp8_kernel(
    const float*   __restrict__ q,
    const uint8_t* __restrict__ k_cache,      // bf8 K cache
    const uint8_t* __restrict__ v_cache,      // bf8 V cache
    float*         __restrict__ output,
    int num_heads,
    int num_kv_heads,
    int head_dim,
    int seq_len,
    int head_stride,
    float scale)
{
    rf_v1_bf8_set_ieee_denormal_mode();

    const int q_head = blockIdx.x;
    if (q_head >= num_heads) return;
    const int kv_head = q_head / (num_heads / num_kv_heads);
    const int tid = threadIdx.x;

    extern __shared__ float s_scores[];

    // Phase 1 — scores[j] = dot(q[q_head], k_cache[kv_head, j]) * scale
    // Each thread processes one or more `j`; loops over `d` read 4 bf8
    // bytes at a time via an int32 load + 4 conversions (head_dim is
    // assumed div-by-4, which holds on all Phase-1 models).
    const float*   q_vec  = q + q_head * head_dim;
    const uint8_t* k_head = k_cache + (size_t)kv_head * head_stride;

    for (int j = tid; j < seq_len; j += blockDim.x) {
        const uint8_t* k_vec = k_head + (size_t)j * head_dim;
        float acc = 0.0f;
        #pragma unroll 4
        for (int d = 0; d < head_dim; d += 4) {
            const int packed = *reinterpret_cast<const int*>(&k_vec[d]);
            acc += q_vec[d + 0] * __builtin_amdgcn_cvt_f32_bf8(packed, 0);
            acc += q_vec[d + 1] * __builtin_amdgcn_cvt_f32_bf8(packed, 1);
            acc += q_vec[d + 2] * __builtin_amdgcn_cvt_f32_bf8(packed, 2);
            acc += q_vec[d + 3] * __builtin_amdgcn_cvt_f32_bf8(packed, 3);
        }
        s_scores[j] = acc * scale;
    }
    __syncthreads();

    // Phase 2 — reduce max(scores) across the block. Identical to FP32.
    __shared__ float s_max;
    if (tid == 0) {
        float m = -INFINITY;
        for (int j = 0; j < seq_len; ++j) {
            if (s_scores[j] > m) m = s_scores[j];
        }
        s_max = m;
    }
    __syncthreads();

    // Phase 3 — exp(score - max), sum. Identical to FP32.
    __shared__ float s_sum;
    if (tid == 0) {
        float sum = 0.0f;
        for (int j = 0; j < seq_len; ++j) {
            const float e = expf(s_scores[j] - s_max);
            s_scores[j] = e;
            sum += e;
        }
        s_sum = sum;
    }
    __syncthreads();

    // Phase 4 — output[d] = Σ_j (scores[j] / sum) * v_cache[kv_head, j, d]
    // Each thread handles one or more `d`. The bf8 intrinsic requires a
    // compile-time constant for `byte_sel`, so we shift the packed int
    // at runtime so that byte 0 always holds the byte we want, and pass
    // the literal `0` to the intrinsic. (Runtime shifts are free on
    // gfx1201's VALU; the alternative would be a 4-way switch-case
    // that bloats the code with identical loop bodies.)
    const float inv_sum = 1.0f / s_sum;
    const uint8_t* v_head = v_cache + (size_t)kv_head * head_stride;
    float* out_vec = output + q_head * head_dim;

    for (int d = tid; d < head_dim; d += blockDim.x) {
        const int shift = (d & 3) * 8;
        const int d_aligned = d & ~3;
        float acc = 0.0f;
        for (int j = 0; j < seq_len; ++j) {
            const int packed = *reinterpret_cast<const int*>(
                &v_head[(size_t)j * head_dim + d_aligned]);
            const int shifted = packed >> shift;
            const float vv = __builtin_amdgcn_cvt_f32_bf8(shifted, 0);
            acc += s_scores[j] * vv;
        }
        out_vec[d] = acc * inv_sum;
    }
}

extern "C" hipError_t rocmforge_launch_attention_decode_fp8(
    const float* q,
    const void*  k_cache,
    const void*  v_cache,
    float*       output,
    int num_heads,
    int num_kv_heads,
    int head_dim,
    int seq_len,
    int head_stride,
    float scale,
    hipStream_t stream)
{
    if (num_heads <= 0 || num_kv_heads <= 0 || head_dim <= 0 || seq_len <= 0)
        return hipErrorInvalidValue;
    if ((num_heads % num_kv_heads) != 0) return hipErrorInvalidValue;
    if ((head_dim % 4) != 0)              return hipErrorInvalidValue;
    const size_t smem = (size_t)seq_len * sizeof(float);
    if (smem > 48 * 1024) return hipErrorInvalidValue;

    const int threads = 128;
    rf_v1_attention_decode_fp8_kernel<<<num_heads, threads, smem, stream>>>(
        q,
        reinterpret_cast<const uint8_t*>(k_cache),
        reinterpret_cast<const uint8_t*>(v_cache),
        output,
        num_heads, num_kv_heads, head_dim,
        seq_len, head_stride, scale);
    return hipGetLastError();
}
