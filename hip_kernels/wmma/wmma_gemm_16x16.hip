// Minimal 16×16 × 16×16 FP16 GEMM via a single `v_wmma_f32_16x16x16_f16` on
// gfx12 / RDNA 4, Wave32. One workgroup, one wave, one WMMA call. Intended
// as a Phase 2a proof of concept — correctness first, tuning later.
//
// Layout reference: docs/wmma_register_layout_gfx12.md (matrix calculator
// output). The load/store patterns below follow those tables literally.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

using half8  = _Float16 __attribute__((ext_vector_type(8)));
using float8 = float    __attribute__((ext_vector_type(8)));

// The gfx12 Wave32 WMMA builtin is a compiler intrinsic — no prototype is
// needed or accepted. Clang recognises `__builtin_amdgcn_wmma_f32_16x16x16_
// f16_w32_gfx12(V8h, V8h, V8f) -> V8f` when compiling for `gfx12-insts,
// wavefrontsize32` (which `--offload-arch=gfx1201` implies).

__global__ void wmma_gemm_16x16_single(
    const __half* __restrict__ A,   // [16 × 16] row-major
    const __half* __restrict__ B,   // [16 × 16] row-major
    float*        __restrict__ D    // [16 × 16] row-major, output
) {
    const int lane = threadIdx.x;               // 0..31, one wave
    const int row  = lane % 16;                 // used by A
    const int col  = lane % 16;                 // used by B and D
    const int k_set = (lane / 16) * 4;          // 0 or 4

    // --- Load A per the A layout table -------------------------------
    // A_regs[idx] = A[row][k_set + offsets[idx]] where offsets are
    // {0,1,2,3, 8,9,10,11}. See docs/wmma_register_layout_gfx12.md.
    half8 a_reg;
    a_reg[0] = A[row * 16 + k_set + 0];
    a_reg[1] = A[row * 16 + k_set + 1];
    a_reg[2] = A[row * 16 + k_set + 2];
    a_reg[3] = A[row * 16 + k_set + 3];
    a_reg[4] = A[row * 16 + k_set + 8];
    a_reg[5] = A[row * 16 + k_set + 9];
    a_reg[6] = A[row * 16 + k_set + 10];
    a_reg[7] = A[row * 16 + k_set + 11];

    // --- Load B per the B layout table -------------------------------
    // B_regs[idx] = B[k_set + offsets[idx]][col] with the same offset
    // set as A. The transposed access into row-major B is intentional.
    half8 b_reg;
    b_reg[0] = B[(k_set + 0)  * 16 + col];
    b_reg[1] = B[(k_set + 1)  * 16 + col];
    b_reg[2] = B[(k_set + 2)  * 16 + col];
    b_reg[3] = B[(k_set + 3)  * 16 + col];
    b_reg[4] = B[(k_set + 8)  * 16 + col];
    b_reg[5] = B[(k_set + 9)  * 16 + col];
    b_reg[6] = B[(k_set + 10) * 16 + col];
    b_reg[7] = B[(k_set + 11) * 16 + col];

    // --- Accumulator init to zero ------------------------------------
    float8 c_reg = {0, 0, 0, 0, 0, 0, 0, 0};

    // --- The matrix-core call ----------------------------------------
    float8 d_reg = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(
        a_reg, b_reg, c_reg);

    // --- Store D per the D layout table ------------------------------
    // Lane L holds column L%16 at rows (L/16)*8 .. (L/16)*8+7.
    // d_reg[v] goes to D[(L/16)*8 + v][L%16].
    const int row_start = (lane / 16) * 8;
    #pragma unroll
    for (int v = 0; v < 8; ++v) {
        D[(row_start + v) * 16 + col] = d_reg[v];
    }
}

extern "C" hipError_t wmma_gemm_16x16_launch(
    const void* A,
    const void* B,
    float*      D,
    hipStream_t stream)
{
    hipLaunchKernelGGL(
        wmma_gemm_16x16_single,
        dim3(1), dim3(32),       // one workgroup, one wave (32 threads)
        0, stream,
        reinterpret_cast<const __half*>(A),
        reinterpret_cast<const __half*>(B),
        D
    );
    return hipGetLastError();
}
