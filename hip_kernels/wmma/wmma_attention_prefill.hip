// Phase 3a — WMMA prefill attention.
//
//   Step 1 (commit 3fd24c5 / f2164c8): single-head, seq=64, head_dim=128.
//   Step 2 (this file): multi-head dispatch, seq_len ∈ {64, 128},
//                       strided Q/K/V/O layout matching the existing
//                       `flash_attn_prefill_strided_kernel`.
//
// Still deliberately out of scope:
//   * causal mask           (Phase 3c)
//   * online softmax        (Phase 3b — we keep SEQ ≤ 128 so the full
//                            S tile fits in LDS)
//   * GQA KV sharing        (Phase 3c)
//   * integration into the forward path (Phase 3d)
//
// Memory layout — same as the existing strided attention kernel:
//   Q, K, V, O all row-major [total_seq × row_stride] where
//   row_stride = num_heads * head_dim for the no-GQA case this kernel
//   supports. A block owns one (head, 64-query-tile) pair and attends
//   to the full SEQ keys/values for that head. Per-head slices start
//   at column `head_idx * head_dim` in each row.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

using half8  = _Float16 __attribute__((ext_vector_type(8)));
using float8 = float    __attribute__((ext_vector_type(8)));

#define HEAD_DIM     128
#define TILE_M       64
#define TILE_K       16
#define WARPS        4
#define THREADS      (WARPS * 32)
#define COL_BLOCKS   4     // one S-column tile = 4 × 16 = 64 cols

// Helper: coalesced cooperative load from row-major global into LDS,
// handling the strided layout used by the production Q/K/V buffers.
//
//   dst[i*inner_cols + j] = src[(row_base + i) * row_stride + col_base + j]
//
// The kernel uses this to pull a `rows × inner_cols` tile of half data
// out of a wider strided tensor. `n_threads` is the workgroup size.
__device__ __forceinline__ void cooperative_load_rows_half(
    __half*        __restrict__ dst,
    const __half*  __restrict__ src,
    int row_base,
    int col_base,
    int rows,
    int inner_cols,
    int row_stride,
    int tid,
    int n_threads)
{
    const int total = rows * inner_cols;
    for (int flat = tid; flat < total; flat += n_threads) {
        const int r = flat / inner_cols;
        const int c = flat - r * inner_cols;
        dst[flat] = src[(row_base + r) * row_stride + col_base + c];
    }
}

// --------------------------------------------------------------------
//   Templated kernel — SEQ = 64 or 128
// --------------------------------------------------------------------
template<int SEQ>
__global__ void wmma_attention_prefill_kernel(
    const __half* __restrict__ Q,
    const __half* __restrict__ K,
    const __half* __restrict__ V,
    float*        __restrict__ O,
    int row_stride,
    float scale)
{
    const int head_idx  = blockIdx.x;
    const int q_tile    = blockIdx.y;     // which 64-query tile of this block
    const int q_row_base = q_tile * 64;   // first query row in global Q
    const int head_offset = head_idx * HEAD_DIM;

    const int tid     = threadIdx.x;
    const int wave_id = tid >> 5;
    const int lane    = tid & 31;
    const int lane_col_in_block = lane & 15;
    const int lane_k_set        = (lane >> 4) * 4;
    const int lane_row_start    = (lane >> 4) * 8;
    const int wave_row_base     = wave_id * 16;

    // LDS layout. Kept static so the compiler sees the sizes up-front.
    //
    // A previous attempt aliased `lds_p` over the `lds_s` buffer — that
    // has a latent race: with `lds_p` stride of `SEQ` FP16s (= 2·SEQ
    // bytes) and `lds_s` stride of `SEQ` FP32s (= 4·SEQ bytes), `lds_p`
    // row r falls inside `lds_s` row r/2's memory region, so thread 1
    // writing p_row[1] collides with thread 0's in-place softmax writes
    // to s_row[0]. Separate allocations are cheap (we have 64 KB of
    // LDS on gfx1201) and avoid the race entirely.
    __shared__ __half lds_q[TILE_M * TILE_K];          // 2 KB
    __shared__ __half lds_k[TILE_K * 64];              // 2 KB (one S-col tile)
    __shared__ float  lds_s[TILE_M * SEQ];             // 16 KB @ SEQ=64, 32 KB @ SEQ=128
    __shared__ __half lds_p[TILE_M * SEQ];             // 8 KB @ SEQ=64,  16 KB @ SEQ=128
    __shared__ __half lds_v[TILE_K * 64];              // 2 KB (one O-col chunk)

    constexpr int N_S_COL_TILES = SEQ / 64;     // 1 (seq=64) or 2 (seq=128)
    constexpr int N_K_TILES_1   = HEAD_DIM / TILE_K;  // 8 (head_dim=128)
    constexpr int N_K_TILES_2   = SEQ / TILE_K;       // 4 or 8
    constexpr int N_O_COL_CHUNKS = HEAD_DIM / 64;     // 2 at head_dim=128

    // ========================================================
    // PHASE 1 — S = Q · K^T / sqrt(head_dim).  Loop over S-col tiles.
    // ========================================================
    for (int s_tile = 0; s_tile < N_S_COL_TILES; ++s_tile) {
        const int s_col_base = s_tile * 64;

        float8 acc_s[COL_BLOCKS];
        #pragma unroll
        for (int cb = 0; cb < COL_BLOCKS; ++cb) {
            acc_s[cb] = {0, 0, 0, 0, 0, 0, 0, 0};
        }

        for (int kt = 0; kt < N_K_TILES_1; ++kt) {
            const int k_start = kt * TILE_K;

            // Q tile:  [64 rows × 16 cols] at (q_row_base, head_offset + k_start)
            cooperative_load_rows_half(
                lds_q,
                Q,
                q_row_base, head_offset + k_start,
                TILE_M, TILE_K,
                row_stride,
                tid, THREADS);

            // K^T tile: we want lds_k[k_in_tile][j] = K[s_col_base + j][head_offset + k_start + k_in_tile]
            // That is a [16 × 64] tile sourced from K's rows [s_col_base..s_col_base+64]
            // at columns [head_offset + k_start..head_offset + k_start + 16], transposed.
            {
                const int total = TILE_K * 64;  // 1024 halfs
                for (int flat = tid; flat < total; flat += THREADS) {
                    const int k_in_tile = flat >> 6;          // / 64
                    const int j         = flat & 63;          // % 64
                    lds_k[k_in_tile * 64 + j] =
                        K[(s_col_base + j) * row_stride +
                          head_offset + k_start + k_in_tile];
                }
            }

            __syncthreads();

            const int a_row_in_tile = wave_row_base + (lane & 15);
            half8 a_reg;
            {
                const int base = a_row_in_tile * TILE_K;
                a_reg[0] = lds_q[base + lane_k_set + 0];
                a_reg[1] = lds_q[base + lane_k_set + 1];
                a_reg[2] = lds_q[base + lane_k_set + 2];
                a_reg[3] = lds_q[base + lane_k_set + 3];
                a_reg[4] = lds_q[base + lane_k_set + 8];
                a_reg[5] = lds_q[base + lane_k_set + 9];
                a_reg[6] = lds_q[base + lane_k_set + 10];
                a_reg[7] = lds_q[base + lane_k_set + 11];
            }

            #pragma unroll
            for (int cb = 0; cb < COL_BLOCKS; ++cb) {
                const int col_in_lds = cb * 16 + lane_col_in_block;
                half8 b_reg;
                b_reg[0] = lds_k[(lane_k_set + 0)  * 64 + col_in_lds];
                b_reg[1] = lds_k[(lane_k_set + 1)  * 64 + col_in_lds];
                b_reg[2] = lds_k[(lane_k_set + 2)  * 64 + col_in_lds];
                b_reg[3] = lds_k[(lane_k_set + 3)  * 64 + col_in_lds];
                b_reg[4] = lds_k[(lane_k_set + 8)  * 64 + col_in_lds];
                b_reg[5] = lds_k[(lane_k_set + 9)  * 64 + col_in_lds];
                b_reg[6] = lds_k[(lane_k_set + 10) * 64 + col_in_lds];
                b_reg[7] = lds_k[(lane_k_set + 11) * 64 + col_in_lds];
                acc_s[cb] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(
                    a_reg, b_reg, acc_s[cb]);
            }
            __syncthreads();
        }

        // Write scaled S columns into lds_s for this S-col tile.
        #pragma unroll
        for (int cb = 0; cb < COL_BLOCKS; ++cb) {
            const int col = s_col_base + cb * 16 + lane_col_in_block;
            #pragma unroll
            for (int v = 0; v < 8; ++v) {
                const int row = wave_row_base + lane_row_start + v;
                lds_s[row * SEQ + col] = acc_s[cb][v] * scale;
            }
        }
    }
    __syncthreads();

    // ========================================================
    // PHASE 2 — row-wise softmax. One thread per row.
    // ========================================================
    if (tid < TILE_M) {
        const int row = tid;
        float* s_row = &lds_s[row * SEQ];

        // stable max
        float m = s_row[0];
        #pragma unroll 1
        for (int j = 1; j < SEQ; ++j) m = fmaxf(m, s_row[j]);

        // exp + sum (in-place in the FP32 buffer)
        float sum = 0.0f;
        #pragma unroll 1
        for (int j = 0; j < SEQ; ++j) {
            const float e = expf(s_row[j] - m);
            s_row[j] = e;
            sum += e;
        }

        const float inv_sum = 1.0f / sum;

        // Normalise + downcast to FP16 into the first half of lds_s,
        // reinterpreted as `lds_p`. The FP32 S we wrote above is no longer
        // read after this point, so overwriting it is safe.
        // Row `row` lives at lds_p[row * SEQ .. row * SEQ + SEQ].
        __half* p_row = &lds_p[row * SEQ];
        #pragma unroll 1
        for (int j = 0; j < SEQ; ++j) {
            p_row[j] = __float2half(s_row[j] * inv_sum);
        }
    }
    __syncthreads();

    // ========================================================
    // PHASE 3 — O = P · V.  Two 64-wide column chunks along head_dim.
    // ========================================================
    #pragma unroll 1
    for (int o_chunk = 0; o_chunk < N_O_COL_CHUNKS; ++o_chunk) {
        const int v_col_base = o_chunk * 64;

        float8 acc_o[COL_BLOCKS];
        #pragma unroll
        for (int cb = 0; cb < COL_BLOCKS; ++cb) {
            acc_o[cb] = {0, 0, 0, 0, 0, 0, 0, 0};
        }

        for (int kt = 0; kt < N_K_TILES_2; ++kt) {
            const int k_start = kt * TILE_K;

            // V tile: [16 × 64] from V[k_start..k_start+16][head_offset+v_col_base..+64]
            {
                const int total = TILE_K * 64;
                for (int flat = tid; flat < total; flat += THREADS) {
                    const int k_in_tile = flat >> 6;
                    const int j         = flat & 63;
                    lds_v[k_in_tile * 64 + j] =
                        V[(k_start + k_in_tile) * row_stride +
                          head_offset + v_col_base + j];
                }
            }
            __syncthreads();

            const int a_row_in_tile = wave_row_base + (lane & 15);
            half8 a_reg;
            {
                const int base = a_row_in_tile * SEQ + k_start;
                a_reg[0] = lds_p[base + lane_k_set + 0];
                a_reg[1] = lds_p[base + lane_k_set + 1];
                a_reg[2] = lds_p[base + lane_k_set + 2];
                a_reg[3] = lds_p[base + lane_k_set + 3];
                a_reg[4] = lds_p[base + lane_k_set + 8];
                a_reg[5] = lds_p[base + lane_k_set + 9];
                a_reg[6] = lds_p[base + lane_k_set + 10];
                a_reg[7] = lds_p[base + lane_k_set + 11];
            }

            #pragma unroll
            for (int cb = 0; cb < COL_BLOCKS; ++cb) {
                const int col_in_lds = cb * 16 + lane_col_in_block;
                half8 b_reg;
                b_reg[0] = lds_v[(lane_k_set + 0)  * 64 + col_in_lds];
                b_reg[1] = lds_v[(lane_k_set + 1)  * 64 + col_in_lds];
                b_reg[2] = lds_v[(lane_k_set + 2)  * 64 + col_in_lds];
                b_reg[3] = lds_v[(lane_k_set + 3)  * 64 + col_in_lds];
                b_reg[4] = lds_v[(lane_k_set + 8)  * 64 + col_in_lds];
                b_reg[5] = lds_v[(lane_k_set + 9)  * 64 + col_in_lds];
                b_reg[6] = lds_v[(lane_k_set + 10) * 64 + col_in_lds];
                b_reg[7] = lds_v[(lane_k_set + 11) * 64 + col_in_lds];
                acc_o[cb] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(
                    a_reg, b_reg, acc_o[cb]);
            }
            __syncthreads();
        }

        // Store the O chunk.
        #pragma unroll
        for (int cb = 0; cb < COL_BLOCKS; ++cb) {
            const int out_col = head_offset + v_col_base + cb * 16 + lane_col_in_block;
            #pragma unroll
            for (int v = 0; v < 8; ++v) {
                const int out_row_q_local = wave_row_base + lane_row_start + v;
                const int out_row_global  = q_row_base + out_row_q_local;
                O[out_row_global * row_stride + out_col] = acc_o[cb][v];
            }
        }
    }
}

// --------------------------------------------------------------------
//   Back-compat: keep the Step-1 signature as a thin shim.
// --------------------------------------------------------------------
// The Step-1 kernel operated on a single head laid out as dense
// [64 × 128] FP16 buffers (row_stride == HEAD_DIM, num_heads == 1).
// We keep that ABI so existing tests / the standalone HIP harness
// still link and pass.
extern "C" hipError_t wmma_attention_prefill_64_launch(
    const void* Q,
    const void* K,
    const void* V,
    float*      O,
    float       scale,
    hipStream_t stream)
{
    hipLaunchKernelGGL(
        (wmma_attention_prefill_kernel<64>),
        dim3(1, 1, 1), dim3(THREADS, 1, 1),
        0, stream,
        reinterpret_cast<const __half*>(Q),
        reinterpret_cast<const __half*>(K),
        reinterpret_cast<const __half*>(V),
        O,
        /* row_stride = */ HEAD_DIM,
        scale
    );
    return hipGetLastError();
}

// --------------------------------------------------------------------
//   Step-2 multi-head launcher.
//
//   Q/K/V: row-major [seq × row_stride] FP16, where
//          row_stride = num_heads * head_dim (no GQA in this PoC).
//   O    : row-major [seq × row_stride] FP32, same shape as Q.
//   seq  : number of attended key/value tokens (must equal num_query
//          tokens for this PoC; grid Y = seq / 64 query tiles).
//
//   Supported: seq ∈ {64, 128}, head_dim = 128, any num_heads ≥ 1.
// --------------------------------------------------------------------
extern "C" hipError_t wmma_attention_prefill_multihead_launch(
    const void* Q,
    const void* K,
    const void* V,
    float*      O,
    int seq_len,
    int num_heads,
    int row_stride,
    float scale,
    hipStream_t stream)
{
    if (seq_len != 64 && seq_len != 128) return hipErrorInvalidValue;
    if (num_heads <= 0)                  return hipErrorInvalidValue;
    if (row_stride <= 0)                 return hipErrorInvalidValue;
    if ((seq_len % TILE_M) != 0)         return hipErrorInvalidValue;

    dim3 grid(num_heads, seq_len / TILE_M, 1);
    dim3 block(THREADS, 1, 1);

    if (seq_len == 64) {
        hipLaunchKernelGGL(
            (wmma_attention_prefill_kernel<64>),
            grid, block, 0, stream,
            reinterpret_cast<const __half*>(Q),
            reinterpret_cast<const __half*>(K),
            reinterpret_cast<const __half*>(V),
            O,
            row_stride,
            scale);
    } else {
        hipLaunchKernelGGL(
            (wmma_attention_prefill_kernel<128>),
            grid, block, 0, stream,
            reinterpret_cast<const __half*>(Q),
            reinterpret_cast<const __half*>(K),
            reinterpret_cast<const __half*>(V),
            O,
            row_stride,
            scale);
    }
    return hipGetLastError();
}
