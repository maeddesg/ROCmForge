// Integer-WMMA smoke test — MMQ-Port Schritt 1 / Deliverable 1.
//
// Single purpose: confirm that
//   __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12
// compiles and runs on gfx1201, and produces bit-exact output for a
// known 16×16×16 int8 × int8 → int32 GEMM.
//
// Lane layout per llama.cpp/ggml/src/ggml-cuda/mma.cuh:1236-1263
// (RDNA4 path). One wave of 32 lanes holds:
//   A-fragment: 2 × int32 per lane (= 8 int8 per lane, = 16×16 total)
//   B-fragment: 2 × int32 per lane (= 8 int8 per lane, = 16×16 total)
//   C accumulator: 8 × int32 per lane (= 16×16 total)
//
// WMMA layout semantics on RDNA4 wave32 (cross-checked against the AMD
// RDNA4 ISA doc):
//   A is row-major 16×16 int8. Lane `l` holds rows (l%16) × cols
//   [(l/16)*8 .. (l/16)*8+7].
//   B is col-major 16×16 int8. Lane `l` holds cols (l%16) × rows
//   [(l/16)*8 .. (l/16)*8+7].
//   C is 16×16 int32. Lane `l` holds col (l&15) and rows
//   [(l>>4)*8 + 0 .. (l>>4)*8 + 7] as 8 consecutive accumulator slots.
//
// This kernel loads A/B from global memory in canonical row-/col-major
// form and stores C back row-major, using the above lane mapping. A
// simple scalar CPU reference computes the same GEMM; the test asserts
// bit-exact equality.

#include <hip/hip_runtime.h>
#include <stdint.h>

using int32x2_t = __attribute__((__vector_size__(2 * sizeof(int)))) int;
using int32x8_t = __attribute__((__vector_size__(8 * sizeof(int)))) int;

// Compute one 16×16×16 int8 GEMM per workgroup.
// Grid: 1 block, 32 threads (one wave).
__global__ void rf_v1_wmma_i32_smoke_kernel(
    const int8_t*  __restrict__ A,   // 16×16 row-major int8
    const int8_t*  __restrict__ B,   // 16×16 col-major int8 (= B^T row-major)
    int*           __restrict__ C)   // 16×16 row-major int32 output
{
    const int lane = threadIdx.x & 31;
    const int row_in_half = lane & 15;       // 0..15, identical in both halves
    const int half        = lane >> 4;       // 0 or 1 (which k-half this lane owns)

    // Load A fragment: lane holds 8 int8 = A[row_in_half, half*8 .. half*8+7].
    // Pack to int32x2_t (2 int32 = 8 int8).
    int32x2_t a_vec;
    {
        int8_t buf[8];
        #pragma unroll
        for (int c = 0; c < 8; ++c) {
            buf[c] = A[row_in_half * 16 + half * 8 + c];
        }
        a_vec[0] = (uint8_t)buf[0]       | ((uint8_t)buf[1] << 8)
                 | ((uint8_t)buf[2] << 16)| ((uint8_t)buf[3] << 24);
        a_vec[1] = (uint8_t)buf[4]       | ((uint8_t)buf[5] << 8)
                 | ((uint8_t)buf[6] << 16)| ((uint8_t)buf[7] << 24);
    }

    // Load B fragment: B is provided col-major, so B[i + col*16] gives
    // element (i,col). Lane holds 8 int8 = B[half*8 .. half*8+7, col_in_half].
    int32x2_t b_vec;
    {
        int8_t buf[8];
        #pragma unroll
        for (int r = 0; r < 8; ++r) {
            buf[r] = B[(half * 8 + r) + row_in_half * 16];
        }
        b_vec[0] = (uint8_t)buf[0]       | ((uint8_t)buf[1] << 8)
                 | ((uint8_t)buf[2] << 16)| ((uint8_t)buf[3] << 24);
        b_vec[1] = (uint8_t)buf[4]       | ((uint8_t)buf[5] << 8)
                 | ((uint8_t)buf[6] << 16)| ((uint8_t)buf[7] << 24);
    }

    // Accumulator starts at zero.
    int32x8_t acc = {0, 0, 0, 0, 0, 0, 0, 0};

    // Single 16×16×16 integer WMMA. Signature per mma.cuh:1246-1253:
    //   (a_signed, A_fragment, b_signed, B_fragment, C_fragment, clamp)
    acc = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(
        true, a_vec, true, b_vec, acc, true);

    // Per RDNA4 WMMA accumulator layout: lane `l` holds column (l & 15)
    // and the 8 accumulator values are rows `(l>>4)*8 + 0 .. (l>>4)*8 + 7`.
    const int col = lane & 15;
    const int row_base = (lane >> 4) * 8;
    #pragma unroll
    for (int v = 0; v < 8; ++v) {
        C[(row_base + v) * 16 + col] = acc[v];
    }
}

extern "C" hipError_t rocmforge_launch_wmma_i32_smoke(
    const int8_t*  A,
    const int8_t*  B,
    int*           C,
    hipStream_t    stream)
{
    dim3 grid(1, 1, 1);
    dim3 block(32, 1, 1);
    rf_v1_wmma_i32_smoke_kernel<<<grid, block, 0, stream>>>(A, B, C);
    return hipGetLastError();
}
