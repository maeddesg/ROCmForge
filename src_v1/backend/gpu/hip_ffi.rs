//! Raw FFI bindings for the HIP runtime API.
//!
//! The scope is the ~20 functions ROCmForge v1.0 actually needs. All
//! signatures were verified against `/opt/rocm/include/hip/hip_runtime_api.h`
//! on ROCm 7.2.2.
//!
//! Notable ABI decision: `hipGetDeviceProperties` is rewritten to
//! `hipGetDevicePropertiesR0600` by the HIP header (line 103), so calling
//! it directly from Rust would resolve to the old `hip_4.2` symbol and
//! read a struct with the old layout. To avoid mirroring the full
//! `hipDeviceProp_t` (~800 bytes) in Rust we route every property query
//! through the `rf_v1_device_*` C++ helpers defined in
//! `hip_kernels_v1/device_info.hip`, which include the header and
//! therefore see the macro rewrite.

#![allow(non_camel_case_types)]
#![allow(non_upper_case_globals)]

use std::ffi::{c_char, c_int, c_void};

// --- Opaque handles ---------------------------------------------------------

pub type hipError_t = c_int;
pub type hipStream_t = *mut c_void; // struct ihipStream_t*
pub type hipEvent_t = *mut c_void; // struct ihipEvent_t*

// --- Enum / constant mirrors ------------------------------------------------
//
// Only the discriminants we actually pass into HIP. Defined as `c_int`
// constants rather than a `#[repr(C)] enum` so future additions to the
// upstream enum cannot miscompile us.

/// `hipSuccess` — see `driver_types.h`.
pub const HIP_SUCCESS: hipError_t = 0;

/// `hipMemcpyKind` discriminants — driver_types.h:373.
pub const hipMemcpyHostToDevice: c_int = 1;
pub const hipMemcpyDeviceToHost: c_int = 2;
pub const hipMemcpyDeviceToDevice: c_int = 3;

/// `hipHostMalloc` flags — hip_runtime_api.h:817-848.
pub const hipHostMallocDefault: u32 = 0x0;
pub const hipHostMallocPortable: u32 = 0x1;
pub const hipHostMallocMapped: u32 = 0x2;
pub const hipHostMallocWriteCombined: u32 = 0x4;

// --- Direct HIP runtime functions ------------------------------------------

#[link(name = "amdhip64")]
extern "C" {
    // Device management (ABI-stable subset).
    pub fn hipSetDevice(device_id: c_int) -> hipError_t;
    pub fn hipGetDeviceCount(count: *mut c_int) -> hipError_t;
    pub fn hipDeviceSynchronize() -> hipError_t;
    pub fn hipMemGetInfo(free: *mut usize, total: *mut usize) -> hipError_t;

    // Memory.
    pub fn hipMalloc(ptr: *mut *mut c_void, size: usize) -> hipError_t;
    pub fn hipFree(ptr: *mut c_void) -> hipError_t;
    pub fn hipMemcpy(
        dst: *mut c_void,
        src: *const c_void,
        size: usize,
        kind: c_int,
    ) -> hipError_t;
    pub fn hipMemcpyAsync(
        dst: *mut c_void,
        src: *const c_void,
        size: usize,
        kind: c_int,
        stream: hipStream_t,
    ) -> hipError_t;

    // Pinned / mapped host memory — used by the Quality-Monitor dirty-flag
    // telemetry path (architecture_v1.2.0-draft §3.7).
    pub fn hipHostMalloc(ptr: *mut *mut c_void, size: usize, flags: u32) -> hipError_t;
    pub fn hipHostFree(ptr: *mut c_void) -> hipError_t;

    // Stream management.
    pub fn hipStreamCreate(stream: *mut hipStream_t) -> hipError_t;
    pub fn hipStreamSynchronize(stream: hipStream_t) -> hipError_t;
    pub fn hipStreamDestroy(stream: hipStream_t) -> hipError_t;

    // Event management — timing + Bandit measurements.
    pub fn hipEventCreate(event: *mut hipEvent_t) -> hipError_t;
    pub fn hipEventRecord(event: hipEvent_t, stream: hipStream_t) -> hipError_t;
    pub fn hipEventSynchronize(event: hipEvent_t) -> hipError_t;
    pub fn hipEventElapsedTime(
        ms: *mut f32,
        start: hipEvent_t,
        stop: hipEvent_t,
    ) -> hipError_t;
    pub fn hipEventDestroy(event: hipEvent_t) -> hipError_t;

    // Error strings.
    pub fn hipGetLastError() -> hipError_t;
    pub fn hipGetErrorString(error: hipError_t) -> *const c_char;
}

// --- Device-info helpers (linked from hip_kernels_v1/libv1_device_info.a) --
//
// These live in C++ so the `hipGetDeviceProperties` header macro resolves
// to the R0600 symbol with the matching struct layout. Every function
// returns a `hipError_t` (0 = success); out-parameters are plain scalars
// or a NUL-terminated char buffer.

extern "C" {
    pub fn rf_v1_device_get_name(
        device_id: c_int,
        out_name: *mut c_char,
        cap: c_int,
    ) -> c_int;

    pub fn rf_v1_device_get_gcn_arch_name(
        device_id: c_int,
        out_name: *mut c_char,
        cap: c_int,
    ) -> c_int;

    pub fn rf_v1_device_get_scalars(
        device_id: c_int,
        out_compute_units: *mut c_int,
        out_warp_size: *mut c_int,
        out_max_threads_per_block: *mut c_int,
        out_shared_mem_per_block: *mut c_int,
        out_total_global_mem_bytes: *mut u64,
    ) -> c_int;
}
