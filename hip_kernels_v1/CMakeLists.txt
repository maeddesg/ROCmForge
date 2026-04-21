cmake_minimum_required(VERSION 3.21)
project(rocmforge_v1_kernels LANGUAGES HIP)

# Exactly one GPU target. No multi-arch fat binaries — per
# architecture_v1.2.0-draft §3.5 the engine has no hardware-abstraction
# layer and every kernel knows it runs on gfx1201.
set(GPU_TARGETS "gfx1201" CACHE STRING "GPU architectures")

# C++17 for WMMA intrinsic attribute syntax.
set(CMAKE_HIP_STANDARD 17)
set(CMAKE_HIP_STANDARD_REQUIRED ON)

# FTZ / denormals IEEE-conformant. `hipcc -O3` without `-ffast-math`
# honours denormals; nothing else is needed at build-system level.
# Kernel prolog sets MODE register explicitly at runtime per
# architecture_v1.2.0-draft §3.1.

function(add_hip_kernel NAME SOURCE)
    add_library(${NAME} STATIC ${SOURCE})
    set_target_properties(${NAME} PROPERTIES
        HIP_ARCHITECTURES "${GPU_TARGETS}"
        POSITION_INDEPENDENT_CODE ON
    )
    target_compile_options(${NAME} PRIVATE
        --offload-arch=gfx1201
        -O3
        -Wall
        -std=c++17
    )
endfunction()

add_hip_kernel(v1_smoke smoke_test.hip)
add_hip_kernel(v1_device_info device_info.hip)
add_hip_kernel(v1_dequant_parity dequant_parity.hip)
add_hip_kernel(v1_wmma_q4_0_fp16 wmma/wmma_gemm_q4_0_fp16.hip)
add_hip_kernel(v1_wmma_q4_k_fp16 wmma/wmma_gemm_q4_k_fp16.hip)
add_hip_kernel(v1_wmma_q6_k_fp16 wmma/wmma_gemm_q6_k_fp16.hip)
add_hip_kernel(v1_wmma_q8_0_fp16 wmma/wmma_gemm_q8_0_fp16.hip)
add_hip_kernel(v1_wmma_q4_0_fp8 wmma/wmma_gemm_q4_0_fp8.hip)
add_hip_kernel(v1_wmma_q4_k_fp8 wmma/wmma_gemm_q4_k_fp8.hip)
add_hip_kernel(v1_wmma_q6_k_fp8 wmma/wmma_gemm_q6_k_fp8.hip)
add_hip_kernel(v1_wmma_q8_0_fp8 wmma/wmma_gemm_q8_0_fp8.hip)
add_hip_kernel(v1_gemv_q4_0_standard gemv/gemv_q4_0_standard.hip)
add_hip_kernel(v1_gemv_q4_k_standard gemv/gemv_q4_k_standard.hip)
add_hip_kernel(v1_gemv_q6_k_standard gemv/gemv_q6_k_standard.hip)
add_hip_kernel(v1_gemv_q8_0_standard gemv/gemv_q8_0_standard.hip)
add_hip_kernel(v1_gemv_q4_k_q8_inline gemv/gemv_q4_k_q8_inline.hip)
add_hip_kernel(v1_gemv_q4_k_gate_up_swiglu gemv/gemv_q4_k_gate_up_swiglu.hip)
add_hip_kernel(v1_elementwise_block_a elementwise/elementwise_block_a.hip)
