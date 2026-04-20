//! RAII wrappers around raw HIP handles. Each wrapper owns its handle
//! and releases it in `Drop`; release errors are logged via `tracing`
//! rather than panicking — Drop running during unwind must never abort.
//!
//! These types are deliberately `!Send + !Sync` by default: GPU
//! resources are tied to the calling thread's HIP context and moving
//! them across threads without explicit synchronisation is a common
//! source of driver corruption.

use std::ffi::c_void;
use std::marker::PhantomData;

use crate::hip_check;

use super::error::HipResult;
use super::hip_ffi::{
    hipEventCreate, hipEventDestroy, hipEventElapsedTime, hipEventRecord, hipEventSynchronize,
    hipEvent_t, hipFree, hipHostFree, hipHostMalloc, hipHostMallocMapped, hipMalloc, hipMemcpy,
    hipMemcpyAsync, hipMemcpyDeviceToHost, hipMemcpyHostToDevice, hipStreamCreate, hipStreamDestroy,
    hipStreamSynchronize, hipStream_t,
};

// --- HipBuffer: device memory -----------------------------------------------

pub struct HipBuffer {
    ptr: *mut c_void,
    size: usize,
    _not_send: PhantomData<*const ()>,
}

impl HipBuffer {
    pub fn new(size: usize) -> HipResult<Self> {
        let mut ptr: *mut c_void = std::ptr::null_mut();
        hip_check!(hipMalloc(&mut ptr, size), "HipBuffer::new")?;
        Ok(Self {
            ptr,
            size,
            _not_send: PhantomData,
        })
    }

    pub fn as_ptr(&self) -> *const c_void {
        self.ptr as *const _
    }

    pub fn as_mut_ptr(&mut self) -> *mut c_void {
        self.ptr
    }

    pub fn size(&self) -> usize {
        self.size
    }

    pub fn copy_from_host(&mut self, data: &[u8]) -> HipResult<()> {
        assert!(data.len() <= self.size, "data too large for buffer");
        hip_check!(
            hipMemcpy(
                self.ptr,
                data.as_ptr() as *const _,
                data.len(),
                hipMemcpyHostToDevice,
            ),
            "HipBuffer::copy_from_host"
        )
    }

    pub fn copy_to_host(&self, data: &mut [u8]) -> HipResult<()> {
        assert!(data.len() <= self.size, "buffer too small for data");
        hip_check!(
            hipMemcpy(
                data.as_mut_ptr() as *mut _,
                self.ptr,
                data.len(),
                hipMemcpyDeviceToHost,
            ),
            "HipBuffer::copy_to_host"
        )
    }

    pub fn copy_from_host_async(&mut self, data: &[u8], stream: &HipStream) -> HipResult<()> {
        assert!(data.len() <= self.size, "data too large for buffer");
        hip_check!(
            hipMemcpyAsync(
                self.ptr,
                data.as_ptr() as *const _,
                data.len(),
                hipMemcpyHostToDevice,
                stream.raw(),
            ),
            "HipBuffer::copy_from_host_async"
        )
    }
}

impl Drop for HipBuffer {
    fn drop(&mut self) {
        if self.ptr.is_null() {
            return;
        }
        let rc = unsafe { hipFree(self.ptr) };
        if rc != 0 {
            tracing::warn!(code = rc, "hipFree failed in HipBuffer::drop");
        }
    }
}

// --- HipStream --------------------------------------------------------------

pub struct HipStream {
    stream: hipStream_t,
    _not_send: PhantomData<*const ()>,
}

impl HipStream {
    pub fn new() -> HipResult<Self> {
        let mut stream: hipStream_t = std::ptr::null_mut();
        hip_check!(hipStreamCreate(&mut stream), "HipStream::new")?;
        Ok(Self {
            stream,
            _not_send: PhantomData,
        })
    }

    pub fn synchronize(&self) -> HipResult<()> {
        hip_check!(
            hipStreamSynchronize(self.stream),
            "HipStream::synchronize"
        )
    }

    pub fn raw(&self) -> hipStream_t {
        self.stream
    }
}

impl Drop for HipStream {
    fn drop(&mut self) {
        if self.stream.is_null() {
            return;
        }
        let rc = unsafe { hipStreamDestroy(self.stream) };
        if rc != 0 {
            tracing::warn!(code = rc, "hipStreamDestroy failed in HipStream::drop");
        }
    }
}

// --- HipEvent: timing -------------------------------------------------------

pub struct HipEvent {
    event: hipEvent_t,
    _not_send: PhantomData<*const ()>,
}

impl HipEvent {
    pub fn new() -> HipResult<Self> {
        let mut event: hipEvent_t = std::ptr::null_mut();
        hip_check!(hipEventCreate(&mut event), "HipEvent::new")?;
        Ok(Self {
            event,
            _not_send: PhantomData,
        })
    }

    pub fn record(&self, stream: &HipStream) -> HipResult<()> {
        hip_check!(
            hipEventRecord(self.event, stream.raw()),
            "HipEvent::record"
        )
    }

    pub fn synchronize(&self) -> HipResult<()> {
        hip_check!(hipEventSynchronize(self.event), "HipEvent::synchronize")
    }

    /// Milliseconds elapsed between two recorded events. Both events must
    /// already be on a synchronised stream.
    pub fn elapsed_ms(start: &HipEvent, stop: &HipEvent) -> HipResult<f32> {
        let mut ms: f32 = 0.0;
        hip_check!(
            hipEventElapsedTime(&mut ms, start.event, stop.event),
            "HipEvent::elapsed_ms"
        )?;
        Ok(ms)
    }

    pub fn raw(&self) -> hipEvent_t {
        self.event
    }
}

impl Drop for HipEvent {
    fn drop(&mut self) {
        if self.event.is_null() {
            return;
        }
        let rc = unsafe { hipEventDestroy(self.event) };
        if rc != 0 {
            tracing::warn!(code = rc, "hipEventDestroy failed in HipEvent::drop");
        }
    }
}

// --- HipHostBuffer: pinned / mapped host memory -----------------------------

pub struct HipHostBuffer {
    host_ptr: *mut c_void,
    size: usize,
    _not_send: PhantomData<*const ()>,
}

impl HipHostBuffer {
    /// Allocate mapped host memory: accessible from both CPU and GPU at
    /// the same address. Used for the Quality-Monitor dirty-flag
    /// telemetry path (architecture_v1.2.0-draft §3.7).
    pub fn new_mapped(size: usize) -> HipResult<Self> {
        let mut ptr: *mut c_void = std::ptr::null_mut();
        hip_check!(
            hipHostMalloc(&mut ptr, size, hipHostMallocMapped),
            "HipHostBuffer::new_mapped"
        )?;
        Ok(Self {
            host_ptr: ptr,
            size,
            _not_send: PhantomData,
        })
    }

    pub fn as_ptr(&self) -> *const c_void {
        self.host_ptr as *const _
    }

    pub fn as_mut_ptr(&mut self) -> *mut c_void {
        self.host_ptr
    }

    pub fn size(&self) -> usize {
        self.size
    }

    /// Typed slice view for CPU-side access. Panics (debug) if the
    /// buffer size is not an integer multiple of `size_of::<T>()`.
    pub fn as_slice<T>(&self) -> &[T] {
        let stride = std::mem::size_of::<T>();
        assert!(stride > 0 && self.size % stride == 0);
        let count = self.size / stride;
        unsafe { std::slice::from_raw_parts(self.host_ptr as *const T, count) }
    }

    pub fn as_mut_slice<T>(&mut self) -> &mut [T] {
        let stride = std::mem::size_of::<T>();
        assert!(stride > 0 && self.size % stride == 0);
        let count = self.size / stride;
        unsafe { std::slice::from_raw_parts_mut(self.host_ptr as *mut T, count) }
    }
}

impl Drop for HipHostBuffer {
    fn drop(&mut self) {
        if self.host_ptr.is_null() {
            return;
        }
        let rc = unsafe { hipHostFree(self.host_ptr) };
        if rc != 0 {
            tracing::warn!(code = rc, "hipHostFree failed in HipHostBuffer::drop");
        }
    }
}
