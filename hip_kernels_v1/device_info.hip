// Device-info helpers exposed as `extern "C"` so the Rust FFI layer can
// query `hipDeviceProp_t` fields without having to mirror the ~800-byte
// struct layout in Rust. The HIP header rewrites
// `hipGetDeviceProperties` to `hipGetDevicePropertiesR0600` via #define
// (hip_runtime_api.h:103) — doing the query in C++ means the versioned
// symbol is selected correctly at link time.
//
// One function per piece of information we need. All return an
// `int` hipError_t (0 = success) plus out-parameters; no complex types
// cross the FFI boundary.

#include <hip/hip_runtime.h>
#include <cstring>

extern "C" {

/// Name as reported by the driver (marketing name, e.g. "AMD Radeon RX 9070 XT").
/// Writes up to `cap-1` bytes + NUL terminator into `out_name`.
int rf_v1_device_get_name(int device_id, char* out_name, int cap) {
    if (!out_name || cap <= 0) return -1;
    out_name[0] = '\0';
    hipDeviceProp_t prop;
    hipError_t err = hipGetDeviceProperties(&prop, device_id);
    if (err != hipSuccess) return (int)err;
    int n = (int)std::strlen(prop.name);
    if (n >= cap) n = cap - 1;
    std::memcpy(out_name, prop.name, n);
    out_name[n] = '\0';
    return 0;
}

/// GCN arch name (e.g. "gfx1201", "gfx1201:xnack-"). Same truncation rule.
int rf_v1_device_get_gcn_arch_name(int device_id, char* out_name, int cap) {
    if (!out_name || cap <= 0) return -1;
    out_name[0] = '\0';
    hipDeviceProp_t prop;
    hipError_t err = hipGetDeviceProperties(&prop, device_id);
    if (err != hipSuccess) return (int)err;
    int n = (int)std::strlen(prop.gcnArchName);
    if (n >= cap) n = cap - 1;
    std::memcpy(out_name, prop.gcnArchName, n);
    out_name[n] = '\0';
    return 0;
}

/// Scalar fields rolled into one call so we only pay one
/// hipGetDeviceProperties round-trip. `nullptr` outputs are skipped.
int rf_v1_device_get_scalars(
    int device_id,
    int* out_compute_units,
    int* out_warp_size,
    int* out_max_threads_per_block,
    int* out_shared_mem_per_block,
    unsigned long long* out_total_global_mem_bytes
) {
    hipDeviceProp_t prop;
    hipError_t err = hipGetDeviceProperties(&prop, device_id);
    if (err != hipSuccess) return (int)err;
    if (out_compute_units)      *out_compute_units = prop.multiProcessorCount;
    if (out_warp_size)          *out_warp_size = prop.warpSize;
    if (out_max_threads_per_block) *out_max_threads_per_block = prop.maxThreadsPerBlock;
    if (out_shared_mem_per_block) *out_shared_mem_per_block = (int)prop.sharedMemPerBlock;
    if (out_total_global_mem_bytes) *out_total_global_mem_bytes =
        (unsigned long long)prop.totalGlobalMem;
    return 0;
}

} // extern "C"
