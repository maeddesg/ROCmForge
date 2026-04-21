//! FFI bindings for the Phase-1 Dequant-IR parity kernels.
//!
//! The kernels themselves are in `hip_kernels_v1/dequant_parity.hip`,
//! produced by `src_v1::v1::ir::codegen_gpu::emit_all_parity_kernels()`.
//! Each kernel consumes one block per block-id and writes
//! `elements_per_block` FP32 values; launch geometry is `<<<n_blocks,
//! elements_per_block>>>`.

use std::ffi::c_void;

use super::hip_ffi::{hipError_t, hipStream_t};

#[link(name = "v1_dequant_parity", kind = "static")]
extern "C" {
    pub fn rocmforge_launch_dequant_q4_0_parity(
        d_blocks: *const u8,
        d_output: *mut f32,
        n_blocks: i32,
        stream: hipStream_t,
    ) -> hipError_t;

    pub fn rocmforge_launch_dequant_q4_1_parity(
        d_blocks: *const u8,
        d_output: *mut f32,
        n_blocks: i32,
        stream: hipStream_t,
    ) -> hipError_t;

    pub fn rocmforge_launch_dequant_q8_0_parity(
        d_blocks: *const u8,
        d_output: *mut f32,
        n_blocks: i32,
        stream: hipStream_t,
    ) -> hipError_t;

    pub fn rocmforge_launch_dequant_q4_k_parity(
        d_blocks: *const u8,
        d_output: *mut f32,
        n_blocks: i32,
        stream: hipStream_t,
    ) -> hipError_t;

    pub fn rocmforge_launch_dequant_q6_k_parity(
        d_blocks: *const u8,
        d_output: *mut f32,
        n_blocks: i32,
        stream: hipStream_t,
    ) -> hipError_t;
}

// Suppress the "unused c_void" warning if we ever drop a parameter.
#[allow(dead_code)]
fn _keep_c_void_in_scope(_p: *mut c_void) {}
