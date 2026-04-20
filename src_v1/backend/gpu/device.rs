//! GPU device detection. All property queries go through the C++
//! helpers in `hip_kernels_v1/device_info.hip` so the Rust side never
//! has to mirror the R0600-versioned `hipDeviceProp_t` layout.

use std::ffi::{c_char, c_int, CStr};

use crate::hip_check;

use super::error::{check, HipError, HipResult};
use super::hip_ffi::{self, hipSetDevice};

pub struct GpuDevice {
    pub device_id: i32,
    pub name: String,
    pub gcn_arch_name: String,
    pub compute_units: i32,
    pub warp_size: i32,
    pub max_threads_per_block: i32,
    pub shared_mem_per_block: i32,
    pub total_memory: usize,
}

impl GpuDevice {
    pub fn detect(device_id: i32) -> HipResult<Self> {
        hip_check!(hipSetDevice(device_id), "GpuDevice::detect set_device")?;

        let name = read_c_string(device_id, hip_ffi::rf_v1_device_get_name, "device name")?;
        let gcn_arch_name = read_c_string(
            device_id,
            hip_ffi::rf_v1_device_get_gcn_arch_name,
            "gcn arch name",
        )?;

        let mut compute_units: c_int = 0;
        let mut warp_size: c_int = 0;
        let mut max_threads: c_int = 0;
        let mut shared_mem: c_int = 0;
        let mut total_mem: u64 = 0;
        let rc = unsafe {
            hip_ffi::rf_v1_device_get_scalars(
                device_id,
                &mut compute_units,
                &mut warp_size,
                &mut max_threads,
                &mut shared_mem,
                &mut total_mem,
            )
        };
        check(rc, "GpuDevice::detect scalar query")?;

        Ok(Self {
            device_id,
            name,
            gcn_arch_name,
            compute_units,
            warp_size,
            max_threads_per_block: max_threads,
            shared_mem_per_block: shared_mem,
            total_memory: total_mem as usize,
        })
    }

    /// Verify that the detected device is `gfx1201`. The arch string
    /// typically looks like "gfx1201" or "gfx1201:xnack-" — we match on
    /// the prefix to keep feature-suffix variants passing.
    pub fn verify_gfx1201(&self) -> HipResult<()> {
        if self.gcn_arch_name.starts_with("gfx1201") {
            return Ok(());
        }
        Err(HipError {
            code: -1,
            message: format!(
                "expected gfx1201, got gcnArchName='{}' name='{}'",
                self.gcn_arch_name, self.name
            ),
            context: "GpuDevice::verify_gfx1201".to_string(),
        })
    }
}

/// Call one of the string-returning device-info helpers and convert its
/// NUL-terminated output into a Rust `String`.
fn read_c_string(
    device_id: i32,
    f: unsafe extern "C" fn(c_int, *mut c_char, c_int) -> c_int,
    what: &str,
) -> HipResult<String> {
    let mut buf = [0u8; 256];
    let rc = unsafe { f(device_id, buf.as_mut_ptr() as *mut c_char, buf.len() as c_int) };
    check(rc, &format!("GpuDevice::detect {what}"))?;
    let cstr = unsafe { CStr::from_ptr(buf.as_ptr() as *const c_char) };
    Ok(cstr.to_string_lossy().into_owned())
}
