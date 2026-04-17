//! GPU device handle with RAII safety.
//!
//! Ensures proper cleanup on drop.
//! Safe to use even if GPU unavailable (returns error).

use super::error::GpuResult;
use super::ffi;
use super::hipblas_ffi::HipBlasHandle;
use super::weights::GpuBuffer;
use std::sync::{Mutex, OnceLock};

const DEFAULT_Q8_WORKSPACE_BYTES: usize = 64 * 1024;

struct Q8Workspace {
    buffer: Option<GpuBuffer>,
    size: usize,
}

/// Device-owned FP16 scratch used by the hipBLAS prefill path.
///
/// Three buffers live here:
/// - `weight_f16`: dequantised weight matrix for the current projection.
/// - `act_in_f16`:  FP32 → FP16 conversion of the input activations.
/// - `act_out_f16`: hipBLAS GEMM output before FP16 → FP32.
///
/// They grow on demand. Kept on `GpuDevice` (not on `GpuPrefillScratch`)
/// because many callers reach `gpu_dispatch_gemm` without a scratch
/// reference — the device handle is the shared resource the dispatcher
/// can always access.
pub(crate) struct PrefillF16Workspace {
    pub weight: Option<GpuBuffer>,
    pub weight_bytes: usize,
    pub act_in: Option<GpuBuffer>,
    pub act_in_bytes: usize,
    pub act_out: Option<GpuBuffer>,
    pub act_out_bytes: usize,
}

impl PrefillF16Workspace {
    fn new() -> Self {
        Self {
            weight: None,
            weight_bytes: 0,
            act_in: None,
            act_in_bytes: 0,
            act_out: None,
            act_out_bytes: 0,
        }
    }
}

/// GPU device handle.
///
/// Represents an initialized GPU device with proper resource management.
/// Uses RAII pattern to ensure cleanup on drop.
pub struct GpuDevice {
    device_id: i32,
    stream: ffi::hipStream_t,
    warp_size: usize,
    max_shared_mem_per_block: usize,
    q8_workspace: Mutex<Q8Workspace>,
    hipblas_handle: OnceLock<HipBlasHandle>,
    prefill_f16: Mutex<PrefillF16Workspace>,
}

impl GpuDevice {
    /// Initialize GPU device with safety checks.
    ///
    /// Returns error if device ID invalid or HIP not available.
    pub fn init(device_id: i32) -> GpuResult<Self> {
        // Verify device exists
        let info = ffi::hip_get_device_info(device_id)?;

        // Create HIP stream
        let stream = ffi::hip_stream_create()?;

        Ok(Self {
            device_id,
            stream,
            warp_size: info.warp_size,
            max_shared_mem_per_block: info.max_shared_mem_per_block,
            q8_workspace: Mutex::new(Q8Workspace {
                buffer: None,
                size: 0,
            }),
            hipblas_handle: OnceLock::new(),
            prefill_f16: Mutex::new(PrefillF16Workspace::new()),
        })
    }

    /// Ensure the FP16 weight scratch is at least `bytes` big and return a
    /// raw pointer to it. Grows the buffer on demand; synchronises the
    /// stream before replacing any live buffer.
    pub fn prefill_f16_weight(&self, bytes: usize) -> GpuResult<*mut u8> {
        self.prefill_grow(bytes, |w| (&mut w.weight, &mut w.weight_bytes))
    }

    pub fn prefill_f16_act_in(&self, bytes: usize) -> GpuResult<*mut u8> {
        self.prefill_grow(bytes, |w| (&mut w.act_in, &mut w.act_in_bytes))
    }

    pub fn prefill_f16_act_out(&self, bytes: usize) -> GpuResult<*mut u8> {
        self.prefill_grow(bytes, |w| (&mut w.act_out, &mut w.act_out_bytes))
    }

    fn prefill_grow<F>(&self, bytes: usize, pick: F) -> GpuResult<*mut u8>
    where
        F: FnOnce(&mut PrefillF16Workspace) -> (&mut Option<GpuBuffer>, &mut usize),
    {
        let mut ws = self
            .prefill_f16
            .lock()
            .unwrap_or_else(|p| p.into_inner());
        let (buf_slot, size_slot) = pick(&mut ws);
        if *size_slot < bytes {
            if *size_slot != 0 {
                // Any outstanding kernel must finish before we free.
                ffi::hip_stream_synchronize(self.stream)?;
            }
            *buf_slot = Some(GpuBuffer::alloc(bytes)?);
            *size_slot = bytes;
        }
        Ok(buf_slot
            .as_ref()
            .map(|b| b.as_ptr())
            .unwrap_or(std::ptr::null_mut()))
    }

    /// Lazily create and return the hipBLAS handle bound to this device's
    /// stream. Handle lifetime is tied to the `GpuDevice` lifetime.
    pub fn hipblas(&self) -> GpuResult<&HipBlasHandle> {
        if let Some(handle) = self.hipblas_handle.get() {
            return Ok(handle);
        }
        let handle = HipBlasHandle::create()?;
        handle.set_stream(self.stream)?;
        // `OnceLock::set` returns an error if another thread raced us —
        // in that case we discard our handle and return the winner.
        let _ = self.hipblas_handle.set(handle);
        Ok(self.hipblas_handle.get().unwrap())
    }

    /// Get device ID.
    pub fn device_id(&self) -> i32 {
        self.device_id
    }

    /// Get device properties.
    pub fn get_properties(&self) -> GpuResult<super::detect::GpuCapabilities> {
        let info = ffi::hip_get_device_info(self.device_id)?;
        let (free_vram, _) = ffi::hip_get_mem_info(self.device_id)?;

        Ok(super::detect::GpuCapabilities {
            device_name: info.name,
            total_vram_bytes: info.total_vram_bytes,
            free_vram_bytes: free_vram,
            compute_units: info.compute_units,
            max_clock_mhz: info.max_clock_mhz,
            hip_driver_version: 0,
            device_id: self.device_id,
            architecture: super::arch::GpuArchitecture::from_name(&info.arch_name)
                .unwrap_or(super::arch::GpuArchitecture::Unknown(0)),
            max_shared_mem_per_block: info.max_shared_mem_per_block,
        })
    }

    /// Get the HIP stream for this device.
    ///
    /// Used for async kernel execution.
    pub fn stream(&self) -> ffi::hipStream_t {
        self.stream
    }

    /// Begin capture on the device stream.
    pub fn begin_capture(&self, mode: ffi::hipStreamCaptureMode) -> GpuResult<()> {
        ffi::hip_stream_begin_capture(self.stream, mode)
    }

    /// End capture on the device stream and return the graph template.
    pub fn end_capture(&self) -> GpuResult<ffi::hipGraph_t> {
        ffi::hip_stream_end_capture(self.stream)
    }

    /// Return the current stream capture status.
    pub fn stream_capture_status(&self) -> GpuResult<ffi::hipStreamCaptureStatus> {
        ffi::hip_stream_is_capturing(self.stream)
    }

    /// Report the device wavefront size exposed by HIP.
    pub fn warp_size(&self) -> usize {
        self.warp_size
    }

    /// Get maximum shared memory available per block in bytes.
    pub fn max_shared_mem_per_block(&self) -> usize {
        self.max_shared_mem_per_block
    }

    /// Synchronize all queued operations on this device's stream.
    ///
    /// Blocks until all previously queued operations on the stream complete.
    pub fn synchronize(&self) -> GpuResult<()> {
        ffi::hip_stream_synchronize(self.stream)
    }

    /// Get a reusable device-local Q8 workspace of at least `min_bytes`.
    ///
    /// The buffer grows on demand and is reused across decode launches to avoid
    /// per-token allocations in hot paths. Growth is synchronized so in-flight
    /// kernels never observe a freed workspace.
    pub fn q8_workspace_ptr(&self, min_bytes: usize) -> GpuResult<*mut u8> {
        if min_bytes == 0 {
            return Ok(std::ptr::null_mut());
        }

        let mut workspace = self
            .q8_workspace
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());

        if workspace.size < min_bytes {
            if workspace.size != 0 {
                self.synchronize()?;
            }

            let target_bytes = min_bytes.max(DEFAULT_Q8_WORKSPACE_BYTES);
            workspace.buffer = Some(GpuBuffer::alloc(target_bytes)?);
            workspace.size = target_bytes;
        }

        Ok(workspace
            .buffer
            .as_ref()
            .map(|buf| buf.as_ptr())
            .unwrap_or(std::ptr::null_mut()))
    }

    /// Reserve a reusable Q8 workspace before entering a capture-sensitive path.
    ///
    /// This keeps lazy `hipMalloc` growth outside HIP graph capture so later
    /// decode launches can safely reuse the same buffer.
    pub fn reserve_q8_workspace(&self, min_bytes: usize) -> GpuResult<()> {
        let _ = self.q8_workspace_ptr(min_bytes)?;
        Ok(())
    }
}

impl Drop for GpuDevice {
    fn drop(&mut self) {
        // Ignore errors during drop (can't handle them anyway)
        // SAFETY: stream was created by hip_stream_create and not yet destroyed
        let _ = unsafe { ffi::hip_stream_destroy(self.stream) };
    }
}

impl std::fmt::Debug for GpuDevice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GpuDevice")
            .field("device_id", &self.device_id)
            .field("stream", &self.stream)
            .field("warp_size", &self.warp_size)
            .field("max_shared_mem_per_block", &self.max_shared_mem_per_block)
            .field(
                "q8_workspace_bytes",
                &self
                    .q8_workspace
                    .lock()
                    .unwrap_or_else(|poison| poison.into_inner())
                    .size,
            )
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn init_works_if_gpu_available() {
        let device = GpuDevice::init(0);
        match device {
            Ok(_) => println!("GPU device initialized"),
            Err(e) => println!("GPU init failed: {} (expected if no GPU)", e),
        }
    }

    #[test]
    fn invalid_device_returns_error() {
        let device = GpuDevice::init(999);
        assert!(device.is_err());
    }

    #[test]
    fn stream_lifecycle_works() {
        let device = GpuDevice::init(0);
        match device {
            Ok(d) => {
                // Verify stream accessor works
                let _stream = d.stream();
                // Verify sync works (no-op if no operations queued)
                let _ = d.synchronize();
                // Drop will clean up stream
            }
            Err(e) => println!("GPU init failed: {} (expected if no GPU)", e),
        }
    }
}
