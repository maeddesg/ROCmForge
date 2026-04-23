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
pub type hipModule_t = *mut c_void; // struct ihipModule_t*
pub type hipFunction_t = *mut c_void; // struct ihipModuleSymbol_t*
pub type hipGraph_t = *mut c_void; // struct ihipGraph*
pub type hipGraphExec_t = *mut c_void; // struct hipGraphExec*
pub type hipGraphNode_t = *mut c_void; // struct hipGraphNode*

/// Minimal `dim3` mirror — HIP wire format is three u32 fields.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct hip_dim3 {
    pub x: u32,
    pub y: u32,
    pub z: u32,
}

/// Mirror of `hipKernelNodeParams` (hip_runtime_api.h).
/// Matches the field order the HIP headers use — verified against
/// ROCm 7.2.2 for gfx1201. Layout must not change without re-verifying
/// against the header.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct hipKernelNodeParams {
    pub blockDim: hip_dim3,
    pub extra: *mut *mut c_void,
    pub func: *mut c_void,
    pub gridDim: hip_dim3,
    pub kernelParams: *mut *mut c_void,
    pub sharedMemBytes: u32,
}

impl Default for hipKernelNodeParams {
    fn default() -> Self {
        Self {
            blockDim: hip_dim3 { x: 0, y: 0, z: 0 },
            extra: std::ptr::null_mut(),
            func: std::ptr::null_mut(),
            gridDim: hip_dim3 { x: 0, y: 0, z: 0 },
            kernelParams: std::ptr::null_mut(),
            sharedMemBytes: 0,
        }
    }
}

/// `hipStreamCaptureMode` — hip_runtime_api.h.
pub type hipStreamCaptureMode = c_int;
pub const hipStreamCaptureModeGlobal: hipStreamCaptureMode = 0;
pub const hipStreamCaptureModeThreadLocal: hipStreamCaptureMode = 1;
pub const hipStreamCaptureModeRelaxed: hipStreamCaptureMode = 2;

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
    pub fn hipMemcpy(dst: *mut c_void, src: *const c_void, size: usize, kind: c_int) -> hipError_t;
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
    pub fn hipEventElapsedTime(ms: *mut f32, start: hipEvent_t, stop: hipEvent_t) -> hipError_t;
    pub fn hipEventDestroy(event: hipEvent_t) -> hipError_t;

    // Error strings.
    pub fn hipGetLastError() -> hipError_t;
    pub fn hipGetErrorString(error: hipError_t) -> *const c_char;

    // Dynamic module loading (Phase-2 step 2.1.3 Block B) — takes a
    // fatbinary / code-object buffer produced by an out-of-process
    // `hipcc` invocation and binds it into the current context.
    pub fn hipModuleLoadData(module: *mut hipModule_t, image: *const c_void) -> hipError_t;

    pub fn hipModuleGetFunction(
        function: *mut hipFunction_t,
        module: hipModule_t,
        name: *const c_char,
    ) -> hipError_t;

    #[allow(clippy::too_many_arguments)]
    pub fn hipModuleLaunchKernel(
        f: hipFunction_t,
        gridDimX: u32,
        gridDimY: u32,
        gridDimZ: u32,
        blockDimX: u32,
        blockDimY: u32,
        blockDimZ: u32,
        sharedMemBytes: u32,
        stream: hipStream_t,
        kernelParams: *mut *mut c_void,
        extra: *mut *mut c_void,
    ) -> hipError_t;

    pub fn hipModuleUnload(module: hipModule_t) -> hipError_t;

    // --- HIP-Graph (capture + execute) — added for the dispatch-
    // overhead elimination follow-up (2026-04-23). API verified on
    // ROCm 7.2.2 with the standalone `hip_graph_api_probe.cpp`.

    pub fn hipStreamBeginCapture(stream: hipStream_t, mode: hipStreamCaptureMode) -> hipError_t;

    pub fn hipStreamEndCapture(stream: hipStream_t, pGraph: *mut hipGraph_t) -> hipError_t;

    pub fn hipGraphInstantiate(
        pGraphExec: *mut hipGraphExec_t,
        graph: hipGraph_t,
        pErrorNode: *mut hipGraphNode_t,
        pLogBuffer: *mut c_char,
        bufferSize: usize,
    ) -> hipError_t;

    pub fn hipGraphLaunch(exec: hipGraphExec_t, stream: hipStream_t) -> hipError_t;

    pub fn hipGraphDestroy(graph: hipGraph_t) -> hipError_t;
    pub fn hipGraphExecDestroy(exec: hipGraphExec_t) -> hipError_t;

    /// Returns topologically ordered nodes. `*numNodes` is set to the
    /// actual count; if `nodes == nullptr` only the count is returned.
    pub fn hipGraphGetNodes(
        graph: hipGraph_t,
        nodes: *mut hipGraphNode_t,
        numNodes: *mut usize,
    ) -> hipError_t;

    /// Retrieves the kernel-node parameters as the graph currently
    /// holds them. The `kernelParams` pointer inside the returned
    /// struct references HIP-owned memory that stays valid for the
    /// lifetime of the graph.
    pub fn hipGraphKernelNodeGetParams(
        node: hipGraphNode_t,
        pNodeParams: *mut hipKernelNodeParams,
    ) -> hipError_t;

    /// Updates the parameters of an instantiated exec's kernel node
    /// in place. HIP copies the `kernelParams` values during the call
    /// — the caller only needs the passed memory valid for the
    /// duration of this one call.
    pub fn hipGraphExecKernelNodeSetParams(
        hGraphExec: hipGraphExec_t,
        node: hipGraphNode_t,
        pNodeParams: *const hipKernelNodeParams,
    ) -> hipError_t;
}

// --- Device-info helpers (linked from hip_kernels_v1/libv1_device_info.a) --
//
// These live in C++ so the `hipGetDeviceProperties` header macro resolves
// to the R0600 symbol with the matching struct layout. Every function
// returns a `hipError_t` (0 = success); out-parameters are plain scalars
// or a NUL-terminated char buffer.

extern "C" {
    pub fn rf_v1_device_get_name(device_id: c_int, out_name: *mut c_char, cap: c_int) -> c_int;

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
