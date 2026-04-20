//! gfx1201 (RDNA 4) backend — HIP kernels, WMMA dispatch, VRAM arena.
//!
//! Wave32, 64 CUs, 128 AI accelerators, 104 VGPR occupancy target, 64 KB
//! LDS. Native FP8-WMMA via
//! `__builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12` as default
//! input path. See architecture_v1.2.0-draft §3.1, §3.2, §3.6–§3.8.

pub mod device;
pub mod error;
pub mod hip_ffi;
pub mod wrappers;
