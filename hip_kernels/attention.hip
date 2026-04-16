#include "common.hip"

namespace {

constexpr float kNegInf = -3.402823466e+38f;
constexpr int kDecodeScoreCapacity = 16000;  // max scores in LDS (~64KB with reduction scratch)
constexpr int kDecodeReductionScratch = MAX_BLOCK_SIZE / 32;

__device__ inline float warp_reduce_sum(float value) {
    #pragma unroll
    for (int delta = warpSize / 2; delta > 0; delta >>= 1) {
        value += __shfl_down(value, delta);
    }
    return value;
}

__device__ inline float warp_reduce_max(float value) {
    #pragma unroll
    for (int delta = warpSize / 2; delta > 0; delta >>= 1) {
        value = fmaxf(value, __shfl_down(value, delta));
    }
    return value;
}

__device__ inline float block_reduce_sum(float value, float* scratch) {
    const int lane = threadIdx.x % warpSize;
    const int warp_id = threadIdx.x / warpSize;
    const int warp_count = (blockDim.x + warpSize - 1) / warpSize;

    value = warp_reduce_sum(value);
    if (lane == 0) {
        scratch[warp_id] = value;
    }
    __syncthreads();

    if (warp_id == 0) {
        value = (lane < warp_count) ? scratch[lane] : 0.0f;
        value = warp_reduce_sum(value);
        if (lane == 0) {
            scratch[0] = value;
        }
    }
    __syncthreads();
    return scratch[0];
}

__device__ inline float block_reduce_max(float value, float* scratch) {
    const int lane = threadIdx.x % warpSize;
    const int warp_id = threadIdx.x / warpSize;
    const int warp_count = (blockDim.x + warpSize - 1) / warpSize;

    value = warp_reduce_max(value);
    if (lane == 0) {
        scratch[warp_id] = value;
    }
    __syncthreads();

    if (warp_id == 0) {
        value = (lane < warp_count) ? scratch[lane] : kNegInf;
        value = warp_reduce_max(value);
        if (lane == 0) {
            scratch[0] = value;
        }
    }
    __syncthreads();
    return scratch[0];
}

} // namespace

// ── Multi-Head Flash Attention (Single Token Decode) ──────────────────────────────

// One block handles ONE head.
// The grid size is (num_heads).
// This uses all CUs effectively.

__global__ void flash_attn_decode_strided_multi_head_kernel(
    float* __restrict__ out,            // [num_heads * head_dim]
    const float* __restrict__ q,        // [num_heads * head_dim]
    const half* __restrict__ k_cache,   // [max_seq * kv_size] (FP16)
    const half* __restrict__ v_cache,   // [max_seq * kv_size] (FP16)
    const int seq_len,
    const int num_heads,
    const int num_kv_heads,
    const int head_dim,
    const float scale
) {
    extern __shared__ float smem[]; // Size: blockDim.x * sizeof(float)

    const int head_idx = blockIdx.x;
    const int tid = threadIdx.x;
    const int kv_head_idx = head_idx / (num_heads / num_kv_heads);
    const int kv_size = num_kv_heads * head_dim;
    const int head_offset = kv_head_idx * head_dim;

    if (head_idx >= num_heads) return;

    // Output pointer for this head
    float* head_out = out + head_idx * head_dim;
    const float* head_q = q + head_idx * head_dim;

    // Local registers for online softmax
    float m = kNegInf;
    float l = 0.0f;

    // Buffer for results in shared memory (one per thread)
    // We reuse smem for reduction scores
    
    // Process all positions
    for (int pos = 0; pos < seq_len; ++pos) {
        const size_t cache_base = (size_t)pos * kv_size + head_offset;
        
        // Dot product Q * K[pos]
        float score = 0.0f;
        for (int i = tid; i < head_dim; i += blockDim.x) {
            score += head_q[i] * __half2float(k_cache[cache_base + i]);
        }

        // Parallel reduction within block
        smem[tid] = score;
        __syncthreads();
        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s) smem[tid] += smem[tid + s];
            __syncthreads();
        }
        score = smem[0] * scale;

        // Softmax update
        const float m_prev = m;
        const float l_prev = l;
        m = max(m_prev, score);
        const float exp_score = expf(score - m);
        const float exp_diff = expf(m_prev - m);
        l = l_prev * exp_diff + exp_score;

        // Update output (incremental)
        // This part is memory-intensive. We can optimize it by doing it at the end
        // but online softmax usually does it per-step.
        // For better performance, we'll store the partial results.
    }

    // Wait! A better way for small models:
    // Compute all scores first, store in LDS if possible, then do ONE pass over V.
}

// ── Optimized Multi-Head Attention (One Pass over V) ──────────────────────────────

__global__ void flash_attn_decode_strided_multi_head_v2_kernel(
    float* __restrict__ out,
    const float* __restrict__ q,
    const half* __restrict__ k_cache,
    const half* __restrict__ v_cache,
    const int seq_len,
    const int num_heads,
    const int num_kv_heads,
    const int head_dim,
    const float scale
) {
    const int head_idx = blockIdx.x;
    const int tid = threadIdx.x;
    const int kv_head_idx = head_idx / (num_heads / num_kv_heads);
    const int kv_size = num_kv_heads * head_dim;
    const int head_offset = kv_head_idx * head_dim;

    if (head_idx >= num_heads) return;

    const float* head_q = q + head_idx * head_dim;

    // 1. Compute scores for all positions
    // We use shared memory to store scores if seq_len is small.
    extern __shared__ float s_scores[];
    float* s_reduce = s_scores + seq_len;  // reduction scratch follows scores

    // Each thread handles some positions
    for (int pos = tid; pos < seq_len; pos += blockDim.x) {
        const size_t cache_base = (size_t)pos * kv_size + head_offset;
        float score = 0.0f;
        for (int i = 0; i < head_dim; ++i) {
            score += head_q[i] * __half2float(k_cache[cache_base + i]);
        }
        score *= scale;
        s_scores[pos] = score;
    }
    __syncthreads();

    float local_max = kNegInf;
    for (int pos = tid; pos < seq_len; pos += blockDim.x) {
        local_max = fmaxf(local_max, s_scores[pos]);
    }
    const float m = block_reduce_max(local_max, s_reduce);

    // 3. Compute exp and sum
    float local_sum = 0.0f;
    for (int pos = tid; pos < seq_len; pos += blockDim.x) {
        const float weight = expf(s_scores[pos] - m);
        s_scores[pos] = weight;
        local_sum += weight;
    }
    const float s_sum = block_reduce_sum(local_sum, s_reduce);

    // 4. Compute final output values
    const float inv_sum = 1.0f / s_sum;
    float* head_out = out + head_idx * head_dim;

    for (int i = tid; i < head_dim; i += blockDim.x) {
        float val = 0.0f;
        for (int pos = 0; pos < seq_len; ++pos) {
            const size_t cache_base = (size_t)pos * kv_size + head_offset;
            val += s_scores[pos] * __half2float(v_cache[cache_base + i]);
        }
        head_out[i] = val * inv_sum;
    }
}

__global__ void flash_attn_decode_strided_multi_head_state_kernel(
    float* __restrict__ out,
    const float* __restrict__ q,
    const half* __restrict__ k_cache,
    const half* __restrict__ v_cache,
    const int* __restrict__ seq_len_ptr,
    const int num_heads,
    const int num_kv_heads,
    const int head_dim,
    const float scale
) {
    const int seq_len = *seq_len_ptr;
    const int head_idx = blockIdx.x;
    const int tid = threadIdx.x;
    const int kv_head_idx = head_idx / (num_heads / num_kv_heads);
    const int kv_size = num_kv_heads * head_dim;
    const int head_offset = kv_head_idx * head_dim;

    if (head_idx >= num_heads) return;

    const float* head_q = q + head_idx * head_dim;
    extern __shared__ float s_scores[];

    float* s_reduce = s_scores + kDecodeScoreCapacity;

    for (int pos = tid; pos < seq_len; pos += blockDim.x) {
        const size_t cache_base = (size_t)pos * kv_size + head_offset;
        float score = 0.0f;
        for (int i = 0; i < head_dim; ++i) {
            score += head_q[i] * __half2float(k_cache[cache_base + i]);
        }
        score *= scale;
        s_scores[pos] = score;
    }
    __syncthreads();

    float local_max = kNegInf;
    for (int pos = tid; pos < seq_len; pos += blockDim.x) {
        local_max = fmaxf(local_max, s_scores[pos]);
    }
    const float m = block_reduce_max(local_max, s_reduce);

    float local_sum = 0.0f;
    for (int pos = tid; pos < seq_len; pos += blockDim.x) {
        const float weight = expf(s_scores[pos] - m);
        s_scores[pos] = weight;
        local_sum += weight;
    }
    const float s_sum = block_reduce_sum(local_sum, s_reduce);

    const float inv_sum = 1.0f / s_sum;
    float* head_out = out + head_idx * head_dim;

    for (int i = tid; i < head_dim; i += blockDim.x) {
        float val = 0.0f;
        for (int pos = 0; pos < seq_len; ++pos) {
            const size_t cache_base = (size_t)pos * kv_size + head_offset;
            val += s_scores[pos] * __half2float(v_cache[cache_base + i]);
        }
        head_out[i] = val * inv_sum;
    }
}

// ── Tiled Flash Attention Decode (Online Softmax) ────────────────────────────────
//
// Key differences from v2:
//   1. K and V are loaded tile-by-tile into LDS (not read N× from VRAM)
//   2. Online softmax eliminates the s_scores[seq_len] buffer
//   3. Scales to arbitrary seq_len (no 2048 cap)
//   4. Each byte of KV cache is read exactly once from VRAM
//
// LDS layout (reused across phases within each tile):
//   [0 .. head_dim-1]                          : Q vector (f32)
//   [head_dim .. head_dim + TILE*head_dim - 1] : K or V tile (f32, converted from f16)
//   [overlap region]                           : tile scores (f32, TILE elements)
//   [end]                                      : warp reduction scratch
//
// Grid:  (num_heads) blocks
// Block: 256 threads

template<int TILE_SIZE>
__global__ void flash_attn_decode_tiled_kernel(
    float* __restrict__ out,
    const float* __restrict__ q,
    const half* __restrict__ k_cache,
    const half* __restrict__ v_cache,
    const int seq_len,
    const int num_heads,
    const int num_kv_heads,
    const int head_dim,
    const float scale
) {
    const int head_idx = blockIdx.x;
    const int tid = threadIdx.x;

    if (head_idx >= num_heads) return;

    const int kv_head_idx = head_idx / (num_heads / num_kv_heads);
    const int kv_size = num_kv_heads * head_dim;
    const int head_offset = kv_head_idx * head_dim;

    // ── LDS allocation ──────────────────────────────────────────────
    // s_q:     [head_dim] floats — Q vector
    // s_tile:  [TILE_SIZE * head_dim] floats — K or V tile (converted from f16)
    // s_score: [TILE_SIZE] floats — dot-product scores for current tile
    // s_reduce:[warp_count] floats — reduction scratch
    //
    // s_tile and s_score can overlap since we use s_score AFTER computing
    // all dot products and BEFORE loading V. But for simplicity, keep
    // s_score separate (only TILE_SIZE × 4 = 256 bytes).
    extern __shared__ char s_raw[];
    float* s_q = reinterpret_cast<float*>(s_raw);
    float* s_tile = s_q + head_dim;
    float* s_score = s_tile + TILE_SIZE * head_dim;
    float* s_reduce = s_score + TILE_SIZE;

    // ── Load Q into LDS ─────────────────────────────────────────────
    const float* head_q = q + head_idx * head_dim;
    for (int i = tid; i < head_dim; i += blockDim.x) {
        s_q[i] = head_q[i];
    }
    __syncthreads();

    // ── Online softmax state (per-thread partial accumulators) ──────
    // Each thread accumulates over a subset of head_dim dimensions.
    // o[d] = running weighted sum for output dimension d.
    // m = running max score, l = running sum of exp(score - m).
    float m = kNegInf;
    float l = 0.0f;

    // Output accumulator in registers — each thread handles a slice of head_dim
    float* head_out = out + head_idx * head_dim;

    // We need per-thread partial output. Since head_dim can be up to 128
    // and we have 256 threads, each thread handles at most 1 element
    // (or 0 if head_dim < blockDim.x). We'll accumulate in global memory
    // at the end. For now, use register-based approach with thread-strided dims.
    //
    // Actually, for the V accumulation we need all threads to cooperate
    // on the matrix-vector multiply. Let's use a simpler approach:
    // accumulate the full output in LDS after each V tile.

    // We'll reuse s_tile for the output accumulator (head_dim floats).
    // But s_tile is also needed for K/V loading. So we need a separate
    // output buffer. Let's put it after s_reduce.
    float* s_out = s_reduce + (blockDim.x / 32);

    // Initialize output accumulator to zero
    for (int i = tid; i < head_dim; i += blockDim.x) {
        s_out[i] = 0.0f;
    }
    __syncthreads();

    // ── Tile loop ───────────────────────────────────────────────────
    for (int tile_start = 0; tile_start < seq_len; tile_start += TILE_SIZE) {
        const int tile_end = min(tile_start + TILE_SIZE, seq_len);
        const int tile_len = tile_end - tile_start;

        // ── Phase 1: Cooperatively load K[tile] from VRAM into LDS ──
        // Layout in LDS: s_tile[pos_in_tile * head_dim + d]
        // Each thread loads multiple elements.
        const int k_elems = tile_len * head_dim;
        for (int idx = tid; idx < k_elems; idx += blockDim.x) {
            const int pos_in_tile = idx / head_dim;
            const int d = idx % head_dim;
            const size_t cache_addr = (size_t)(tile_start + pos_in_tile) * kv_size + head_offset + d;
            s_tile[pos_in_tile * head_dim + d] = __half2float(k_cache[cache_addr]);
        }
        __syncthreads();

        // ── Phase 2: Compute Q·K scores — warp-cooperative ──────────
        // Each warp (32 lanes) cooperates on one position's dot product.
        // Lanes split head_dim, then reduce via __shfl_down.
        {
            const int warp_id = tid >> 5;       // tid / 32
            const int lane_id = tid & 31;       // tid % 32
            const int n_warps = blockDim.x >> 5;

            for (int p = warp_id; p < tile_len; p += n_warps) {
                const float* k_row = &s_tile[p * head_dim];
                float partial = 0.0f;
                for (int d = lane_id; d < head_dim; d += 32) {
                    partial += s_q[d] * k_row[d];
                }
                // Warp reduction
                for (int offset = 16; offset > 0; offset >>= 1) {
                    partial += __shfl_down(partial, offset);
                }
                if (lane_id == 0) {
                    s_score[p] = partial * scale;
                }
            }
        }
        __syncthreads();

        // ── Phase 3: Online softmax update ──────────────────────────
        // Find max score in this tile (block-wide reduction)
        float tile_max = kNegInf;
        for (int p = tid; p < tile_len; p += blockDim.x) {
            tile_max = fmaxf(tile_max, s_score[p]);
        }
        tile_max = block_reduce_max(tile_max, s_reduce);
        // tile_max is now broadcast to all threads via s_reduce[0]

        // Compute new global max and rescale factor
        const float m_new = fmaxf(m, tile_max);
        const float exp_m_old = expf(m - m_new);  // rescale factor for old state

        // Compute exp(score - m_new) for each position and sum
        float tile_sum = 0.0f;
        for (int p = tid; p < tile_len; p += blockDim.x) {
            const float w = expf(s_score[p] - m_new);
            s_score[p] = w;  // overwrite score with weight
            tile_sum += w;
        }
        tile_sum = block_reduce_sum(tile_sum, s_reduce);

        // Update running l: l_new = l_old * exp(m_old - m_new) + tile_sum
        const float l_new = l * exp_m_old + tile_sum;

        // Rescale existing output accumulator: O *= (l_old * exp_m_old) / l_new
        const float rescale = (l > 0.0f) ? (l * exp_m_old / l_new) : 0.0f;
        const float new_weight = (l_new > 0.0f) ? (1.0f / l_new) : 0.0f;

        for (int i = tid; i < head_dim; i += blockDim.x) {
            s_out[i] *= rescale;
        }
        __syncthreads();

        // Update state
        m = m_new;
        l = l_new;

        // ── Phase 4: Cooperatively load V[tile] into LDS ────────────
        // Reuse s_tile (K data is no longer needed, scores are in s_score)
        const int v_elems = tile_len * head_dim;
        for (int idx = tid; idx < v_elems; idx += blockDim.x) {
            const int pos_in_tile = idx / head_dim;
            const int d = idx % head_dim;
            const size_t cache_addr = (size_t)(tile_start + pos_in_tile) * kv_size + head_offset + d;
            s_tile[pos_in_tile * head_dim + d] = __half2float(v_cache[cache_addr]);
        }
        __syncthreads();

        // ── Phase 5: Accumulate weighted V ──────────────────────────
        // O += (1/l_new) * sum_p( weight[p] * V[p] )
        // Each thread handles a subset of head_dim dimensions.
        for (int i = tid; i < head_dim; i += blockDim.x) {
            float val = 0.0f;
            for (int p = 0; p < tile_len; ++p) {
                val += s_score[p] * s_tile[p * head_dim + i];
            }
            s_out[i] += val * new_weight;
        }
        __syncthreads();
    }

    // ── Write output ────────────────────────────────────────────────
    for (int i = tid; i < head_dim; i += blockDim.x) {
        head_out[i] = s_out[i];
    }
}

// State variant: reads seq_len from device memory (for HIP graph capture)
template<int TILE_SIZE>
__global__ void flash_attn_decode_tiled_state_kernel(
    float* __restrict__ out,
    const float* __restrict__ q,
    const half* __restrict__ k_cache,
    const half* __restrict__ v_cache,
    const int* __restrict__ seq_len_ptr,
    const int num_heads,
    const int num_kv_heads,
    const int head_dim,
    const float scale
) {
    const int seq_len = *seq_len_ptr;
    const int head_idx = blockIdx.x;
    const int tid = threadIdx.x;

    if (head_idx >= num_heads) return;

    const int kv_head_idx = head_idx / (num_heads / num_kv_heads);
    const int kv_size = num_kv_heads * head_dim;
    const int head_offset = kv_head_idx * head_dim;

    extern __shared__ char s_raw[];
    float* s_q = reinterpret_cast<float*>(s_raw);
    float* s_tile = s_q + head_dim;
    float* s_score = s_tile + TILE_SIZE * head_dim;
    float* s_reduce = s_score + TILE_SIZE;
    float* s_out = s_reduce + (blockDim.x / 32);

    const float* head_q = q + head_idx * head_dim;
    for (int i = tid; i < head_dim; i += blockDim.x) {
        s_q[i] = head_q[i];
    }
    for (int i = tid; i < head_dim; i += blockDim.x) {
        s_out[i] = 0.0f;
    }
    __syncthreads();

    float m = kNegInf;
    float l = 0.0f;
    float* head_out = out + head_idx * head_dim;

    for (int tile_start = 0; tile_start < seq_len; tile_start += TILE_SIZE) {
        const int tile_end = min(tile_start + TILE_SIZE, seq_len);
        const int tile_len = tile_end - tile_start;

        // Load K tile
        const int k_elems = tile_len * head_dim;
        for (int idx = tid; idx < k_elems; idx += blockDim.x) {
            const int pos_in_tile = idx / head_dim;
            const int d = idx % head_dim;
            const size_t cache_addr = (size_t)(tile_start + pos_in_tile) * kv_size + head_offset + d;
            s_tile[pos_in_tile * head_dim + d] = __half2float(k_cache[cache_addr]);
        }
        __syncthreads();

        // Q·K scores — warp-cooperative dot product
        {
            const int warp_id = tid >> 5;
            const int lane_id = tid & 31;
            const int n_warps = blockDim.x >> 5;

            for (int p = warp_id; p < tile_len; p += n_warps) {
                const float* k_row = &s_tile[p * head_dim];
                float partial = 0.0f;
                for (int d = lane_id; d < head_dim; d += 32) {
                    partial += s_q[d] * k_row[d];
                }
                for (int offset = 16; offset > 0; offset >>= 1) {
                    partial += __shfl_down(partial, offset);
                }
                if (lane_id == 0) {
                    s_score[p] = partial * scale;
                }
            }
        }
        __syncthreads();

        // Online softmax
        float tile_max = kNegInf;
        for (int p = tid; p < tile_len; p += blockDim.x) {
            tile_max = fmaxf(tile_max, s_score[p]);
        }
        tile_max = block_reduce_max(tile_max, s_reduce);

        const float m_new = fmaxf(m, tile_max);
        const float exp_m_old = expf(m - m_new);

        float tile_sum = 0.0f;
        for (int p = tid; p < tile_len; p += blockDim.x) {
            const float w = expf(s_score[p] - m_new);
            s_score[p] = w;
            tile_sum += w;
        }
        tile_sum = block_reduce_sum(tile_sum, s_reduce);

        const float l_new = l * exp_m_old + tile_sum;
        const float rescale = (l > 0.0f) ? (l * exp_m_old / l_new) : 0.0f;
        const float new_weight = (l_new > 0.0f) ? (1.0f / l_new) : 0.0f;

        for (int i = tid; i < head_dim; i += blockDim.x) {
            s_out[i] *= rescale;
        }
        __syncthreads();

        m = m_new;
        l = l_new;

        // Load V tile
        const int v_elems = tile_len * head_dim;
        for (int idx = tid; idx < v_elems; idx += blockDim.x) {
            const int pos_in_tile = idx / head_dim;
            const int d = idx % head_dim;
            const size_t cache_addr = (size_t)(tile_start + pos_in_tile) * kv_size + head_offset + d;
            s_tile[pos_in_tile * head_dim + d] = __half2float(v_cache[cache_addr]);
        }
        __syncthreads();

        // Accumulate weighted V
        for (int i = tid; i < head_dim; i += blockDim.x) {
            float val = 0.0f;
            for (int p = 0; p < tile_len; ++p) {
                val += s_score[p] * s_tile[p * head_dim + i];
            }
            s_out[i] += val * new_weight;
        }
        __syncthreads();
    }

    for (int i = tid; i < head_dim; i += blockDim.x) {
        head_out[i] = s_out[i];
    }
}

// ── GQA-Aware Tiled Attention (Option B) ─────────────────────────────────────────
//
// 1 block = 1 KV-head. All Q-heads sharing that KV-head (gqa_ratio heads)
// are processed in parallel by separate warps within the same block.
// The KV tile is loaded once into LDS and shared across all Q-heads,
// reducing VRAM bandwidth by gqa_ratio× compared to v2.
//
// Thread mapping: 256 threads = 8 warps. Warp w handles Q-head w (if w < gqa_ratio).
// Supports gqa_ratio up to 8 (covers Qwen2.5 0.5B/7B=7, 14B=5, Llama 3=8).
//
// LDS layout:
//   s_tile:   [TILE × head_dim]        — K or V tile (f32, shared)
//   s_q:      [gqa_ratio × head_dim]   — Q vectors for all local heads
//   s_scores: [gqa_ratio × TILE]       — per-head attention scores
//   s_out:    [gqa_ratio × head_dim]   — per-head output accumulators
//
// Grid:  num_kv_heads blocks
// Block: 256 threads (8 warps)

template<int TILE_SIZE>
__global__ void flash_attn_decode_gqa_kernel(
    float* __restrict__ out,
    const float* __restrict__ q,
    const half* __restrict__ k_cache,
    const half* __restrict__ v_cache,
    const int seq_len,
    const int num_heads,
    const int num_kv_heads,
    const int head_dim,
    const float scale
) {
    const int kv_head_idx = blockIdx.x;
    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane_id = tid & 31;
    const int gqa_ratio = num_heads / num_kv_heads;

    if (kv_head_idx >= num_kv_heads) return;

    const int kv_size = num_kv_heads * head_dim;
    const int head_offset = kv_head_idx * head_dim;

    // Which Q-head does this warp handle?
    const int local_head = warp_id;
    const int global_head = kv_head_idx * gqa_ratio + local_head;
    const bool active = (local_head < gqa_ratio) && (global_head < num_heads);

    // ── LDS allocation ──────────────────────────────────────────────
    extern __shared__ char s_raw[];
    float* s_tile   = reinterpret_cast<float*>(s_raw);
    float* s_q      = s_tile + TILE_SIZE * head_dim;
    float* s_scores = s_q + gqa_ratio * head_dim;
    float* s_out    = s_scores + gqa_ratio * TILE_SIZE;

    // ── Load Q vectors for all local heads into LDS ─────────────────
    const int q_elems = gqa_ratio * head_dim;
    for (int idx = tid; idx < q_elems; idx += blockDim.x) {
        const int h = idx / head_dim;
        const int d = idx % head_dim;
        const int gh = kv_head_idx * gqa_ratio + h;
        s_q[idx] = (gh < num_heads) ? q[gh * head_dim + d] : 0.0f;
    }

    // ── Initialize output accumulators ──────────────────────────────
    const int out_elems = gqa_ratio * head_dim;
    for (int idx = tid; idx < out_elems; idx += blockDim.x) {
        s_out[idx] = 0.0f;
    }
    __syncthreads();

    // ── Per-warp online softmax state (registers) ───────────────────
    float m = kNegInf;
    float l = 0.0f;

    // ── Tile loop ───────────────────────────────────────────────────
    for (int tile_start = 0; tile_start < seq_len; tile_start += TILE_SIZE) {
        const int tile_end = min(tile_start + TILE_SIZE, seq_len);
        const int tile_len = tile_end - tile_start;

        // Phase 1: ALL threads cooperatively load K tile into LDS
        const int k_elems = tile_len * head_dim;
        for (int idx = tid; idx < k_elems; idx += blockDim.x) {
            const int p = idx / head_dim;
            const int d = idx % head_dim;
            const size_t addr = (size_t)(tile_start + p) * kv_size + head_offset + d;
            s_tile[p * head_dim + d] = __half2float(k_cache[addr]);
        }
        __syncthreads();

        // Phase 2: Each warp computes Q·K for its Q-head
        // 32 lanes split head_dim, warp-reduce via __shfl_down
        if (active) {
            const float* my_q = &s_q[local_head * head_dim];
            for (int p = 0; p < tile_len; ++p) {
                const float* k_row = &s_tile[p * head_dim];
                float partial = 0.0f;
                for (int d = lane_id; d < head_dim; d += 32) {
                    partial += my_q[d] * k_row[d];
                }
                for (int off = 16; off > 0; off >>= 1) {
                    partial += __shfl_down(partial, off);
                }
                if (lane_id == 0) {
                    s_scores[local_head * TILE_SIZE + p] = partial * scale;
                }
            }
        }
        __syncthreads();

        // Phase 3: Online softmax (warp-local, per Q-head)
        float inv_l_new = 0.0f;
        if (active) {
            // Tile max via warp reduction
            float tile_max = kNegInf;
            for (int p = lane_id; p < tile_len; p += 32) {
                tile_max = fmaxf(tile_max, s_scores[local_head * TILE_SIZE + p]);
            }
            for (int off = 16; off > 0; off >>= 1) {
                tile_max = fmaxf(tile_max, __shfl_xor(tile_max, off));
            }
            // tile_max is now broadcast to all lanes

            const float m_new = fmaxf(m, tile_max);
            const float exp_m_old = expf(m - m_new);

            // Compute weights and tile sum
            float tile_sum = 0.0f;
            for (int p = lane_id; p < tile_len; p += 32) {
                const float w = expf(s_scores[local_head * TILE_SIZE + p] - m_new);
                s_scores[local_head * TILE_SIZE + p] = w;
                tile_sum += w;
            }
            for (int off = 16; off > 0; off >>= 1) {
                tile_sum += __shfl_xor(tile_sum, off);
            }
            // tile_sum broadcast to all lanes

            const float l_new = l * exp_m_old + tile_sum;
            const float rescale = (l > 0.0f) ? (l * exp_m_old / l_new) : 0.0f;
            inv_l_new = (l_new > 0.0f) ? (1.0f / l_new) : 0.0f;

            // Rescale existing output accumulator
            float* my_out = &s_out[local_head * head_dim];
            for (int d = lane_id; d < head_dim; d += 32) {
                my_out[d] *= rescale;
            }

            m = m_new;
            l = l_new;
        }
        __syncthreads();

        // Phase 4: ALL threads cooperatively load V tile into LDS
        // (s_tile is reused; s_scores survive since they're in separate LDS region)
        const int v_elems = tile_len * head_dim;
        for (int idx = tid; idx < v_elems; idx += blockDim.x) {
            const int p = idx / head_dim;
            const int d = idx % head_dim;
            const size_t addr = (size_t)(tile_start + p) * kv_size + head_offset + d;
            s_tile[p * head_dim + d] = __half2float(v_cache[addr]);
        }
        __syncthreads();

        // Phase 5: Each warp accumulates weighted V
        if (active) {
            float* my_out = &s_out[local_head * head_dim];
            for (int d = lane_id; d < head_dim; d += 32) {
                float val = 0.0f;
                for (int p = 0; p < tile_len; ++p) {
                    val += s_scores[local_head * TILE_SIZE + p] * s_tile[p * head_dim + d];
                }
                my_out[d] += val * inv_l_new;
            }
        }
        __syncthreads();
    }

    // ── Write output ────────────────────────────────────────────────
    if (active) {
        float* head_out = out + global_head * head_dim;
        const float* my_out = &s_out[local_head * head_dim];
        for (int d = lane_id; d < head_dim; d += 32) {
            head_out[d] = my_out[d];
        }
    }
}

// State variant (reads seq_len from device memory for HIP graph capture)
template<int TILE_SIZE>
__global__ void flash_attn_decode_gqa_state_kernel(
    float* __restrict__ out,
    const float* __restrict__ q,
    const half* __restrict__ k_cache,
    const half* __restrict__ v_cache,
    const int* __restrict__ seq_len_ptr,
    const int num_heads,
    const int num_kv_heads,
    const int head_dim,
    const float scale
) {
    const int seq_len = *seq_len_ptr;
    const int kv_head_idx = blockIdx.x;
    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane_id = tid & 31;
    const int gqa_ratio = num_heads / num_kv_heads;

    if (kv_head_idx >= num_kv_heads) return;

    const int kv_size = num_kv_heads * head_dim;
    const int head_offset = kv_head_idx * head_dim;

    const int local_head = warp_id;
    const int global_head = kv_head_idx * gqa_ratio + local_head;
    const bool active = (local_head < gqa_ratio) && (global_head < num_heads);

    extern __shared__ char s_raw[];
    float* s_tile   = reinterpret_cast<float*>(s_raw);
    float* s_q      = s_tile + TILE_SIZE * head_dim;
    float* s_scores = s_q + gqa_ratio * head_dim;
    float* s_out    = s_scores + gqa_ratio * TILE_SIZE;

    const int q_elems = gqa_ratio * head_dim;
    for (int idx = tid; idx < q_elems; idx += blockDim.x) {
        const int h = idx / head_dim;
        const int d = idx % head_dim;
        const int gh = kv_head_idx * gqa_ratio + h;
        s_q[idx] = (gh < num_heads) ? q[gh * head_dim + d] : 0.0f;
    }
    const int out_elems = gqa_ratio * head_dim;
    for (int idx = tid; idx < out_elems; idx += blockDim.x) {
        s_out[idx] = 0.0f;
    }
    __syncthreads();

    float m = kNegInf;
    float l = 0.0f;

    for (int tile_start = 0; tile_start < seq_len; tile_start += TILE_SIZE) {
        const int tile_end = min(tile_start + TILE_SIZE, seq_len);
        const int tile_len = tile_end - tile_start;

        const int k_elems = tile_len * head_dim;
        for (int idx = tid; idx < k_elems; idx += blockDim.x) {
            const int p = idx / head_dim;
            const int d = idx % head_dim;
            const size_t addr = (size_t)(tile_start + p) * kv_size + head_offset + d;
            s_tile[p * head_dim + d] = __half2float(k_cache[addr]);
        }
        __syncthreads();

        if (active) {
            const float* my_q = &s_q[local_head * head_dim];
            for (int p = 0; p < tile_len; ++p) {
                const float* k_row = &s_tile[p * head_dim];
                float partial = 0.0f;
                for (int d = lane_id; d < head_dim; d += 32) {
                    partial += my_q[d] * k_row[d];
                }
                for (int off = 16; off > 0; off >>= 1) {
                    partial += __shfl_down(partial, off);
                }
                if (lane_id == 0) {
                    s_scores[local_head * TILE_SIZE + p] = partial * scale;
                }
            }
        }
        __syncthreads();

        float inv_l_new = 0.0f;
        if (active) {
            float tile_max = kNegInf;
            for (int p = lane_id; p < tile_len; p += 32) {
                tile_max = fmaxf(tile_max, s_scores[local_head * TILE_SIZE + p]);
            }
            for (int off = 16; off > 0; off >>= 1) {
                tile_max = fmaxf(tile_max, __shfl_xor(tile_max, off));
            }

            const float m_new = fmaxf(m, tile_max);
            const float exp_m_old = expf(m - m_new);

            float tile_sum = 0.0f;
            for (int p = lane_id; p < tile_len; p += 32) {
                const float w = expf(s_scores[local_head * TILE_SIZE + p] - m_new);
                s_scores[local_head * TILE_SIZE + p] = w;
                tile_sum += w;
            }
            for (int off = 16; off > 0; off >>= 1) {
                tile_sum += __shfl_xor(tile_sum, off);
            }

            const float l_new = l * exp_m_old + tile_sum;
            const float rescale = (l > 0.0f) ? (l * exp_m_old / l_new) : 0.0f;
            inv_l_new = (l_new > 0.0f) ? (1.0f / l_new) : 0.0f;

            float* my_out = &s_out[local_head * head_dim];
            for (int d = lane_id; d < head_dim; d += 32) {
                my_out[d] *= rescale;
            }

            m = m_new;
            l = l_new;
        }
        __syncthreads();

        const int v_elems = tile_len * head_dim;
        for (int idx = tid; idx < v_elems; idx += blockDim.x) {
            const int p = idx / head_dim;
            const int d = idx % head_dim;
            const size_t addr = (size_t)(tile_start + p) * kv_size + head_offset + d;
            s_tile[p * head_dim + d] = __half2float(v_cache[addr]);
        }
        __syncthreads();

        if (active) {
            float* my_out = &s_out[local_head * head_dim];
            for (int d = lane_id; d < head_dim; d += 32) {
                float val = 0.0f;
                for (int p = 0; p < tile_len; ++p) {
                    val += s_scores[local_head * TILE_SIZE + p] * s_tile[p * head_dim + d];
                }
                my_out[d] += val * inv_l_new;
            }
        }
        __syncthreads();
    }

    if (active) {
        float* head_out = out + global_head * head_dim;
        const float* my_out = &s_out[local_head * head_dim];
        for (int d = lane_id; d < head_dim; d += 32) {
            head_out[d] = my_out[d];
        }
    }
}

extern "C" hipError_t gpu_flash_attn_decode_strided_multi_head(
    float* d_out,
    const float* d_q,
    const half* d_k_cache,
    const half* d_v_cache,
    const int seq_len,
    const int num_heads,
    const int num_kv_heads,
    const int head_dim,
    const float scale,
    hipStream_t stream
) {
    const int threads_per_block = 256;
    const size_t smem_size = (seq_len + kDecodeReductionScratch) * sizeof(float);

    flash_attn_decode_strided_multi_head_v2_kernel<<<num_heads, threads_per_block, smem_size, stream>>>(
        d_out, d_q, d_k_cache, d_v_cache, seq_len, num_heads, num_kv_heads, head_dim, scale
    );

    return hipGetLastError();
}

extern "C" hipError_t gpu_flash_attn_decode_strided_multi_head_state(
    float* d_out,
    const float* d_q,
    const half* d_k_cache,
    const half* d_v_cache,
    const int* d_seq_len,
    const int num_heads,
    const int num_kv_heads,
    const int head_dim,
    const float scale,
    hipStream_t stream
) {
    if (d_seq_len == nullptr) {
        return hipErrorInvalidValue;
    }

    const int threads_per_block = 256;
    const size_t smem_size = (kDecodeScoreCapacity + kDecodeReductionScratch) * sizeof(float);

    flash_attn_decode_strided_multi_head_state_kernel<<<num_heads, threads_per_block, smem_size, stream>>>(
        d_out, d_q, d_k_cache, d_v_cache, d_seq_len, num_heads, num_kv_heads, head_dim, scale
    );

    return hipGetLastError();
}

// ── Verify Attention (Speculative Decoding) ─────────────────────────────────────
//
// Like prefill attention, but reads K/V from the FP16 KV cache instead of
// FP32 scratch. Used for the verify step in speculative decoding where N
// draft tokens are verified in one batch pass against the target model.
//
// Grid:  n_verify blocks (one per query position in the verify batch)
// Block: 256 threads
// Called once per head from Rust side (same pattern as prefill).
//
// Causal mask: query at batch index i attends to KV positions 0..start_pos+i.

__global__ void flash_attn_verify_strided_kernel(
    float* __restrict__ out,
    const float* __restrict__ q,
    const half* __restrict__ k_cache,
    const half* __restrict__ v_cache,
    const int n_verify,
    const int start_pos,
    const int head_dim,
    const int out_stride,
    const int q_stride,
    const int kv_size,
    const int out_head_offset,
    const int q_head_offset,
    const int kv_head_offset,
    const float scale
) {
    extern __shared__ float smem[];
    const int query_idx = blockIdx.x;
    const int tid = threadIdx.x;
    if (query_idx >= n_verify) return;

    const int total_seq_len = start_pos + query_idx + 1;

    const float* q_ptr = q + (size_t)query_idx * q_stride + q_head_offset;
    float* out_ptr = out + (size_t)query_idx * out_stride + out_head_offset;

    if (tid < head_dim) out_ptr[tid] = 0.0f;
    __syncthreads();

    float m = kNegInf;
    float l = 0.0f;

    for (int key_pos = 0; key_pos < total_seq_len; ++key_pos) {
        const size_t cache_base = (size_t)key_pos * kv_size + kv_head_offset;

        // Q·K dot product (Q is FP32, K is FP16 in cache)
        float score = 0.0f;
        for (int i = tid; i < head_dim; i += blockDim.x) {
            score += q_ptr[i] * __half2float(k_cache[cache_base + i]);
        }

        smem[tid] = score;
        __syncthreads();
        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s) smem[tid] += smem[tid + s];
            __syncthreads();
        }
        score = smem[0] * scale;

        // Online softmax + V accumulation (single-thread, same as prefill)
        if (tid == 0) {
            const float prev_m = m;
            const float prev_l = l;
            m = max(prev_m, score);
            const float exp_diff = expf(prev_m - m);
            const float exp_score = expf(score - m);
            l = prev_l * exp_diff + exp_score;
            const float old_scale = (l > 0.0f) ? (prev_l * exp_diff / l) : 0.0f;
            const float new_scale = (l > 0.0f) ? (exp_score / l) : 0.0f;

            for (int i = 0; i < head_dim; ++i) {
                out_ptr[i] = out_ptr[i] * old_scale
                    + __half2float(v_cache[cache_base + i]) * new_scale;
            }
        }
        __syncthreads();
    }
}

extern "C" hipError_t gpu_flash_attn_verify_strided(
    float* d_out,
    const float* d_q,
    const half* d_k_cache,
    const half* d_v_cache,
    const int n_verify,
    const int start_pos,
    const int head_dim,
    const int out_stride,
    const int q_stride,
    const int kv_size,
    const int out_head_offset,
    const int q_head_offset,
    const int kv_head_offset,
    const float scale,
    hipStream_t stream
) {
    if (n_verify <= 0) return hipErrorInvalidValue;
    hipLaunchKernelGGL(flash_attn_verify_strided_kernel,
        n_verify, 256, 256 * sizeof(float), stream,
        d_out, d_q, d_k_cache, d_v_cache, n_verify, start_pos,
        head_dim, out_stride, q_stride, kv_size,
        out_head_offset, q_head_offset, kv_head_offset, scale);
    return hipGetLastError();
}

// ── All-Heads Verify Attention (single dispatch per layer) ───────────────────
//
// For speculative decode verify: processes N query tokens × all heads in ONE
// kernel launch per layer, instead of N × num_heads separate launches.
//
// Grid: (num_kv_heads, n_verify, 1)
//   blockIdx.x = KV head group
//   blockIdx.y = query position (0..n_verify-1)
//
// Each block handles one KV-head group for one query position.
// Causal mask: query i attends to positions 0..start_pos+i (inclusive).
//
// Layout: Q/Out are [n_verify × num_heads × head_dim], strided.
// K/V cache: [max_seq × num_kv_heads × head_dim], FP16.

template<int TILE_SIZE>
__global__ void flash_attn_verify_all_heads_kernel(
    float* __restrict__ out,         // [n_verify × num_heads × head_dim]
    const float* __restrict__ q,     // [n_verify × num_heads × head_dim]
    const half* __restrict__ k_cache,
    const half* __restrict__ v_cache,
    const int n_verify,
    const int start_pos,
    const int num_heads,
    const int num_kv_heads,
    const int head_dim,
    const float scale
) {
    const int kv_head_idx = blockIdx.x;
    const int query_idx = blockIdx.y;
    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane_id = tid & 31;
    const int gqa_ratio = num_heads / num_kv_heads;

    if (kv_head_idx >= num_kv_heads || query_idx >= n_verify) return;

    // Causal mask: this query position can see start_pos + query_idx + 1 KV entries
    const int causal_len = start_pos + query_idx + 1;

    const int kv_size = num_kv_heads * head_dim;
    const int q_size = num_heads * head_dim;
    const int head_offset = kv_head_idx * head_dim;

    // Which Q-head does this warp handle?
    const int local_head = warp_id;
    const int global_head = kv_head_idx * gqa_ratio + local_head;
    const bool active = (local_head < gqa_ratio) && (global_head < num_heads);

    // ── LDS allocation ──────────────────────────────────────────────
    extern __shared__ char s_raw[];
    float* s_tile   = reinterpret_cast<float*>(s_raw);
    float* s_q      = s_tile + TILE_SIZE * head_dim;
    float* s_scores = s_q + gqa_ratio * head_dim;
    float* s_out    = s_scores + gqa_ratio * TILE_SIZE;

    // ── Load Q vectors for all local heads (from this query row) ────
    const float* q_row = q + (size_t)query_idx * q_size;
    const int q_elems = gqa_ratio * head_dim;
    for (int idx = tid; idx < q_elems; idx += blockDim.x) {
        const int h = idx / head_dim;
        const int d = idx % head_dim;
        const int gh = kv_head_idx * gqa_ratio + h;
        s_q[idx] = (gh < num_heads) ? q_row[gh * head_dim + d] : 0.0f;
    }

    // ── Initialize output accumulators ──────────────────────────────
    const int out_elems = gqa_ratio * head_dim;
    for (int idx = tid; idx < out_elems; idx += blockDim.x) {
        s_out[idx] = 0.0f;
    }
    __syncthreads();

    // ── Per-warp online softmax state ───────────────────────────────
    float m = kNegInf;
    float l = 0.0f;

    // ── Tile loop (respecting causal mask) ──────────────────────────
    for (int tile_start = 0; tile_start < causal_len; tile_start += TILE_SIZE) {
        const int tile_end = min(tile_start + TILE_SIZE, causal_len);
        const int tile_len = tile_end - tile_start;

        // Phase 1: ALL threads cooperatively load K tile into LDS
        const int k_elems = tile_len * head_dim;
        for (int idx = tid; idx < k_elems; idx += blockDim.x) {
            const int p = idx / head_dim;
            const int d = idx % head_dim;
            const size_t addr = (size_t)(tile_start + p) * kv_size + head_offset + d;
            s_tile[p * head_dim + d] = __half2float(k_cache[addr]);
        }
        __syncthreads();

        // Phase 2: Each warp computes Q·K for its Q-head
        if (active) {
            const float* my_q = &s_q[local_head * head_dim];
            for (int p = 0; p < tile_len; ++p) {
                const float* k_row = &s_tile[p * head_dim];
                float partial = 0.0f;
                for (int d = lane_id; d < head_dim; d += 32) {
                    partial += my_q[d] * k_row[d];
                }
                for (int off = 16; off > 0; off >>= 1) {
                    partial += __shfl_down(partial, off);
                }
                if (lane_id == 0) {
                    s_scores[local_head * TILE_SIZE + p] = partial * scale;
                }
            }
        }
        __syncthreads();

        // Phase 3: Online softmax
        float inv_l_new = 0.0f;
        if (active) {
            float tile_max = kNegInf;
            for (int p = lane_id; p < tile_len; p += 32) {
                tile_max = fmaxf(tile_max, s_scores[local_head * TILE_SIZE + p]);
            }
            for (int off = 16; off > 0; off >>= 1) {
                tile_max = fmaxf(tile_max, __shfl_xor(tile_max, off));
            }

            const float m_new = fmaxf(m, tile_max);
            const float exp_m_old = expf(m - m_new);

            float tile_sum = 0.0f;
            for (int p = lane_id; p < tile_len; p += 32) {
                const float w = expf(s_scores[local_head * TILE_SIZE + p] - m_new);
                s_scores[local_head * TILE_SIZE + p] = w;
                tile_sum += w;
            }
            for (int off = 16; off > 0; off >>= 1) {
                tile_sum += __shfl_xor(tile_sum, off);
            }

            const float l_new = l * exp_m_old + tile_sum;
            const float rescale = (l > 0.0f) ? (l * exp_m_old / l_new) : 0.0f;
            inv_l_new = (l_new > 0.0f) ? (1.0f / l_new) : 0.0f;

            float* my_out = &s_out[local_head * head_dim];
            for (int d = lane_id; d < head_dim; d += 32) {
                my_out[d] *= rescale;
            }

            m = m_new;
            l = l_new;
        }
        __syncthreads();

        // Phase 4: Load V tile into LDS (reuse s_tile)
        const int v_elems = tile_len * head_dim;
        for (int idx = tid; idx < v_elems; idx += blockDim.x) {
            const int p = idx / head_dim;
            const int d = idx % head_dim;
            const size_t addr = (size_t)(tile_start + p) * kv_size + head_offset + d;
            s_tile[p * head_dim + d] = __half2float(v_cache[addr]);
        }
        __syncthreads();

        // Phase 5: Accumulate weighted V
        if (active) {
            float* my_out = &s_out[local_head * head_dim];
            for (int d = lane_id; d < head_dim; d += 32) {
                float val = 0.0f;
                for (int p = 0; p < tile_len; ++p) {
                    val += s_scores[local_head * TILE_SIZE + p] * s_tile[p * head_dim + d];
                }
                my_out[d] += val * inv_l_new;
            }
        }
        __syncthreads();
    }

    // ── Write output (to this query row's output) ───────────────────
    float* out_row = out + (size_t)query_idx * q_size;
    if (active) {
        float* head_out = out_row + global_head * head_dim;
        const float* my_out = &s_out[local_head * head_dim];
        for (int d = lane_id; d < head_dim; d += 32) {
            head_out[d] = my_out[d];
        }
    }
}

extern "C" hipError_t gpu_flash_attn_verify_all_heads(
    float* d_out,
    const float* d_q,
    const half* d_k_cache,
    const half* d_v_cache,
    const int n_verify,
    const int start_pos,
    const int num_heads,
    const int num_kv_heads,
    const int head_dim,
    const float scale,
    hipStream_t stream
) {
    if (n_verify <= 0 || num_heads <= 0 || num_kv_heads <= 0 || head_dim <= 0) {
        return hipErrorInvalidValue;
    }

    constexpr int TILE = 32;
    const int gqa_ratio = num_heads / num_kv_heads;
    const int threads_per_block = ((gqa_ratio + 7) / 8) * 256;
    const size_t smem_size = (TILE * head_dim + gqa_ratio * head_dim
                              + gqa_ratio * TILE + gqa_ratio * head_dim) * sizeof(float);

    dim3 grid(num_kv_heads, n_verify);
    flash_attn_verify_all_heads_kernel<TILE><<<grid, threads_per_block, smem_size, stream>>>(
        d_out, d_q, d_k_cache, d_v_cache, n_verify, start_pos,
        num_heads, num_kv_heads, head_dim, scale
    );

    return hipGetLastError();
}

extern "C" hipError_t gpu_flash_attn_decode_gqa(
    float* d_out,
    const float* d_q,
    const half* d_k_cache,
    const half* d_v_cache,
    const int seq_len,
    const int num_heads,
    const int num_kv_heads,
    const int head_dim,
    const float scale,
    hipStream_t stream
) {
    constexpr int TILE = 32;
    const int gqa_ratio = num_heads / num_kv_heads;
    // Need at least gqa_ratio warps; round up to multiple of 32
    const int threads_per_block = ((gqa_ratio + 7) / 8) * 256;  // 256 for gqa<=8
    // LDS: tile(TILE*head_dim) + q(gqa*head_dim) + scores(gqa*TILE) + out(gqa*head_dim)
    const size_t smem_size = (TILE * head_dim + gqa_ratio * head_dim
                              + gqa_ratio * TILE + gqa_ratio * head_dim) * sizeof(float);

    flash_attn_decode_gqa_kernel<TILE><<<num_kv_heads, threads_per_block, smem_size, stream>>>(
        d_out, d_q, d_k_cache, d_v_cache, seq_len, num_heads, num_kv_heads, head_dim, scale
    );

    return hipGetLastError();
}

extern "C" hipError_t gpu_flash_attn_decode_gqa_state(
    float* d_out,
    const float* d_q,
    const half* d_k_cache,
    const half* d_v_cache,
    const int* d_seq_len,
    const int num_heads,
    const int num_kv_heads,
    const int head_dim,
    const float scale,
    hipStream_t stream
) {
    if (d_seq_len == nullptr) {
        return hipErrorInvalidValue;
    }

    constexpr int TILE = 32;
    const int gqa_ratio = num_heads / num_kv_heads;
    const int threads_per_block = ((gqa_ratio + 7) / 8) * 256;
    const size_t smem_size = (TILE * head_dim + gqa_ratio * head_dim
                              + gqa_ratio * TILE + gqa_ratio * head_dim) * sizeof(float);

    flash_attn_decode_gqa_state_kernel<TILE><<<num_kv_heads, threads_per_block, smem_size, stream>>>(
        d_out, d_q, d_k_cache, d_v_cache, d_seq_len, num_heads, num_kv_heads, head_dim, scale
    );

    return hipGetLastError();
}

// ── KV Cache Write ───────────────────────────────────────────────────────────────

__global__ void kv_write_kernel(
    half* __restrict__ k_cache,
    half* __restrict__ v_cache,
    const float* __restrict__ k,
    const float* __restrict__ v,
    const int pos,
    const int kv_size
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < kv_size) {
        const size_t cache_idx = (size_t)pos * kv_size + i;
        k_cache[cache_idx] = __float2half(k[i]);
        v_cache[cache_idx] = __float2half(v[i]);
    }
}

__global__ void kv_write_state_kernel(
    half* __restrict__ k_cache,
    half* __restrict__ v_cache,
    const float* __restrict__ k,
    const float* __restrict__ v,
    const int* __restrict__ pos_ptr,
    const int kv_size
) {
    const int pos = *pos_ptr;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < kv_size) {
        const size_t cache_idx = (size_t)pos * kv_size + i;
        k_cache[cache_idx] = __float2half(k[i]);
        v_cache[cache_idx] = __float2half(v[i]);
    }
}

__global__ void kv_write_rope_state_kernel(
    half* __restrict__ k_cache,
    half* __restrict__ v_cache,
    const float* __restrict__ k,
    const float* __restrict__ v,
    const int* __restrict__ pos_ptr,
    const int num_kv_heads,
    const int head_dim,
    const float theta_base,
    const int neox
) {
    const int pos = *pos_ptr;
    const int kv_size = num_kv_heads * head_dim;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= kv_size) {
        return;
    }

    const size_t cache_idx = (size_t)pos * kv_size + i;
    v_cache[cache_idx] = __float2half(v[i]);

    const int head = i / head_dim;
    const int idx_in_head = i - head * head_dim;
    const int head_base = head * head_dim;
    const int half = head_dim / 2;

    int pair_idx;
    int idx0;
    int idx1;

    if (neox) {
        if (idx_in_head < half) {
            pair_idx = idx_in_head;
            idx0 = head_base + idx_in_head;
            idx1 = idx0 + half;
        } else {
            pair_idx = idx_in_head - half;
            idx0 = head_base + pair_idx;
            idx1 = idx0 + half;
        }
    } else {
        pair_idx = idx_in_head / 2;
        idx0 = head_base + 2 * pair_idx;
        idx1 = idx0 + 1;
    }

    const float exponent = (float)(2 * pair_idx) / (float)head_dim;
    const float theta = 1.0f / pow(theta_base, exponent);
    const float angle = (float)pos * theta;
    const float cos_val = cos(angle);
    const float sin_val = sin(angle);
    const float x0 = k[idx0];
    const float x1 = k[idx1];

    const bool first = neox ? (idx_in_head < half) : ((idx_in_head & 1) == 0);
    k_cache[cache_idx] = __float2half(first
        ? (x0 * cos_val - x1 * sin_val)
        : (x0 * sin_val + x1 * cos_val));
}

__global__ void kv_write_batched_kernel(
    half* __restrict__ k_cache,
    half* __restrict__ v_cache,
    const float* __restrict__ k,
    const float* __restrict__ v,
    const int start_pos,
    const int kv_size,
    const int seq_len
) {
    const int s = blockIdx.y;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (s < seq_len && i < kv_size) {
        const int pos = start_pos + s;
        const size_t cache_idx = (size_t)pos * kv_size + i;
        const size_t input_idx = (size_t)s * kv_size + i;
        k_cache[cache_idx] = __float2half(k[input_idx]);
        v_cache[cache_idx] = __float2half(v[input_idx]);
    }
}

// ── Flash Attention (Batch Prefill) ───────────────────────────────────────────────

__global__ void flash_attn_prefill_strided_kernel(
    float* __restrict__ out,
    const float* __restrict__ q,
    const float* __restrict__ k,
    const float* __restrict__ v,
    const int seq_len,
    const int head_dim,
    const int out_stride,
    const int q_stride,
    const int kv_stride,
    const int out_head_offset,
    const int q_head_offset,
    const int kv_head_offset,
    const float scale
) {
    extern __shared__ float smem[];
    const int query_pos = blockIdx.x;
    const int tid = threadIdx.x;
    if (query_pos >= seq_len) return;

    const float* q_ptr = q + (size_t)query_pos * q_stride + q_head_offset;
    float* out_ptr = out + (size_t)query_pos * out_stride + out_head_offset;

    if (tid < head_dim) out_ptr[tid] = 0.0f;
    __syncthreads();

    float m = -3.402823466e+38f;
    float l = 0.0f;

    for (int key_pos = 0; key_pos <= query_pos; ++key_pos) {
        const float* k_ptr = k + (size_t)key_pos * kv_stride + kv_head_offset;
        const float* v_ptr = v + (size_t)key_pos * kv_stride + kv_head_offset;

        float score = 0.0f;
        for (int i = tid; i < head_dim; i += blockDim.x) score += q_ptr[i] * k_ptr[i];

        smem[tid] = score;
        __syncthreads();
        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s) smem[tid] += smem[tid + s];
            __syncthreads();
        }
        score = smem[0] * scale;

        if (tid == 0) {
            const float prev_m = m;
            const float prev_l = l;
            m = max(prev_m, score);
            const float exp_diff = expf(prev_m - m);
            const float exp_score = expf(score - m);
            l = prev_l * exp_diff + exp_score;
            const float old_scale = (l > 0.0f) ? (prev_l * exp_diff / l) : 0.0f;
            const float new_scale = (l > 0.0f) ? (exp_score / l) : 0.0f;

            for (int i = 0; i < head_dim; ++i) {
                out_ptr[i] = out_ptr[i] * old_scale + v_ptr[i] * new_scale;
            }
        }
        __syncthreads();
    }
}

__global__ void kv_write_rope_kernel(
    half* __restrict__ k_cache,
    half* __restrict__ v_cache,
    const float* __restrict__ k,
    const float* __restrict__ v,
    const int pos,
    const int num_kv_heads,
    const int head_dim,
    const float theta_base,
    const int neox
) {
    const int kv_size = num_kv_heads * head_dim;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= kv_size) return;

    const size_t cache_idx = (size_t)pos * kv_size + i;
    v_cache[cache_idx] = __float2half(v[i]);

    const int head = i / head_dim;
    const int idx_in_head = i % head_dim;
    const int head_base = head * head_dim;
    const int half = head_dim / 2;

    int pair_idx;
    int idx0;
    int idx1;

    if (neox) {
        if (idx_in_head < half) {
            pair_idx = idx_in_head;
            idx0 = head_base + idx_in_head;
            idx1 = idx0 + half;
        } else {
            pair_idx = idx_in_head - half;
            idx0 = head_base + pair_idx;
            idx1 = idx0 + half;
        }
    } else {
        pair_idx = idx_in_head / 2;
        idx0 = head_base + 2 * pair_idx;
        idx1 = idx0 + 1;
    }

    const float exponent = (float)(2 * pair_idx) / (float)head_dim;
    const float theta = 1.0f / powf(theta_base, exponent);
    const float angle = (float)pos * theta;
    const float cos_val = cosf(angle);
    const float sin_val = sinf(angle);
    const float x0 = k[idx0];
    const float x1 = k[idx1];

    const bool first = neox ? (idx_in_head < half) : ((idx_in_head & 1) == 0);
    k_cache[cache_idx] = __float2half(first
        ? (x0 * cos_val - x1 * sin_val)
        : (x0 * sin_val + x1 * cos_val));
}

extern "C" {

hipError_t gpu_kv_write_rope(half* d_k_cache, half* d_v_cache, const float* d_k, const float* d_v, const int pos, const int num_kv_heads, const int head_dim, const float theta_base, const int neox, hipStream_t stream) {
    const int kv_size = num_kv_heads * head_dim;
    hipLaunchKernelGGL(kv_write_rope_kernel, (kv_size + 255) / 256, 256, 0, stream, d_k_cache, d_v_cache, d_k, d_v, pos, num_kv_heads, head_dim, theta_base, neox);
    return hipGetLastError();
}

hipError_t gpu_kv_write(half* d_k_cache, half* d_v_cache, const float* d_k, const float* d_v, const int pos, const int kv_size, const int max_seq, hipStream_t stream) {
    hipLaunchKernelGGL(kv_write_kernel, (kv_size + 255) / 256, 256, 0, stream, d_k_cache, d_v_cache, d_k, d_v, pos, kv_size);
    return hipGetLastError();
}

hipError_t gpu_kv_write_state(half* d_k_cache, half* d_v_cache, const float* d_k, const float* d_v, const int* d_pos, const int kv_size, const int max_seq, hipStream_t stream) {
    (void)max_seq;
    if (d_pos == nullptr) {
        return hipErrorInvalidValue;
    }
    hipLaunchKernelGGL(kv_write_state_kernel, (kv_size + 255) / 256, 256, 0, stream, d_k_cache, d_v_cache, d_k, d_v, d_pos, kv_size);
    return hipGetLastError();
}

hipError_t gpu_kv_write_rope_state(half* d_k_cache, half* d_v_cache, const float* d_k, const float* d_v, const int* d_pos, const int num_kv_heads, const int head_dim, const float theta_base, const int neox, hipStream_t stream) {
    if (d_pos == nullptr) {
        return hipErrorInvalidValue;
    }
    const int kv_size = num_kv_heads * head_dim;
    hipLaunchKernelGGL(kv_write_rope_state_kernel, (kv_size + 255) / 256, 256, 0, stream, d_k_cache, d_v_cache, d_k, d_v, d_pos, num_kv_heads, head_dim, theta_base, neox);
    return hipGetLastError();
}

hipError_t gpu_kv_write_batched(half* d_k_cache, half* d_v_cache, const float* d_k, const float* d_v, const int start_pos, const int kv_size, const int max_seq, const int seq_len) {
    hipLaunchKernelGGL(kv_write_batched_kernel, dim3((kv_size + 255) / 256, seq_len), 256, 0, 0, d_k_cache, d_v_cache, d_k, d_v, start_pos, kv_size, seq_len);
    return hipGetLastError();
}

hipError_t gpu_flash_attn_prefill_strided(float* d_out, const float* d_q, const float* d_k, const float* d_v, const int seq_len, const int head_dim, const int out_stride, const int q_stride, const int kv_stride, const int out_head_offset, const int q_head_offset, const int kv_head_offset, const float scale) {
    hipLaunchKernelGGL(flash_attn_prefill_strided_kernel, seq_len, 256, 256 * sizeof(float), 0, d_out, d_q, d_k, d_v, seq_len, head_dim, out_stride, q_stride, kv_stride, out_head_offset, q_head_offset, kv_head_offset, scale);
    return hipGetLastError();
}

} // extern "C"
