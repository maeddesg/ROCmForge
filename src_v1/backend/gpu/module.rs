//! Dynamic GPU module loading — `hipModuleLoadData` + friends.
//!
//! New in Phase 2 / step 2.1.3 Block B. Lets Rust take a
//! `Vec<u8>` code object (produced by the out-of-process `hipcc`
//! invocation in `ga::compile::compile_hip_source`) and bind it into
//! the current HIP context. The resulting [`HipModule`] owns the
//! loaded module; on `Drop` it calls `hipModuleUnload`. [`HipFunction`]
//! is a borrow-like handle into that module — its lifetime is
//! checked at the type system level so a function can't outlive its
//! parent module.
//!
//! This module is small by design: four FFI functions, two RAII
//! wrappers, one `launch` helper. The caller is responsible for
//! laying out the `**void` kernel-parameter array correctly; the
//! shape of that array is kernel-specific and better handled at the
//! codegen caller than generically here.

use std::ffi::{c_void, CString};
use std::marker::PhantomData;

use super::error::{check, HipResult};
use super::hip_ffi::{
    hipFunction_t, hipModuleGetFunction, hipModuleLaunchKernel, hipModuleLoadData, hipModuleUnload,
    hipModule_t,
};
use super::wrappers::HipStream;

/// RAII wrapper around `hipModule_t`. Drop-unload.
pub struct HipModule {
    module: hipModule_t,
    _not_send: PhantomData<*const ()>,
}

impl HipModule {
    /// Load a gfx1201 code object (e.g. the bytes of a `.co` produced
    /// by `hipcc --offload-arch=gfx1201 -shared -fPIC -O3`) into the
    /// current HIP context.
    pub fn load(code_object: &[u8]) -> HipResult<Self> {
        let mut module: hipModule_t = std::ptr::null_mut();
        let rc = unsafe { hipModuleLoadData(&mut module, code_object.as_ptr() as *const c_void) };
        check(rc, "hipModuleLoadData")?;
        Ok(Self {
            module,
            _not_send: PhantomData,
        })
    }

    /// Look up a kernel by its extern-"C" symbol name.
    pub fn get_function<'a>(&'a self, name: &str) -> HipResult<HipFunction<'a>> {
        let c_name = CString::new(name).expect("kernel name with interior NUL");
        let mut function: hipFunction_t = std::ptr::null_mut();
        let rc = unsafe { hipModuleGetFunction(&mut function, self.module, c_name.as_ptr()) };
        check(rc, "hipModuleGetFunction")?;
        Ok(HipFunction {
            function,
            _module: PhantomData,
        })
    }

    pub fn raw(&self) -> hipModule_t {
        self.module
    }
}

impl Drop for HipModule {
    fn drop(&mut self) {
        if self.module.is_null() {
            return;
        }
        let rc = unsafe { hipModuleUnload(self.module) };
        if rc != 0 {
            tracing::warn!(code = rc, "hipModuleUnload failed in HipModule::drop");
        }
    }
}

/// Kernel handle — lifetime bound to the parent [`HipModule`]. No
/// `Drop` because HIP doesn't expose a per-function destructor;
/// module unload invalidates every function obtained from it.
pub struct HipFunction<'m> {
    function: hipFunction_t,
    _module: PhantomData<&'m HipModule>,
}

impl<'m> HipFunction<'m> {
    /// Launch the kernel.
    ///
    /// `grid` and `block` are the CUDA-style 3-D grid and block
    /// dimensions. `shared_mem_bytes` is the dynamic shared-memory
    /// allocation (for `extern __shared__ ...`). `args` is an array
    /// of `*mut c_void` that each point at a kernel argument (one
    /// pointer per argument).
    pub fn launch(
        &self,
        grid: (u32, u32, u32),
        block: (u32, u32, u32),
        shared_mem_bytes: u32,
        stream: &HipStream,
        args: &mut [*mut c_void],
    ) -> HipResult<()> {
        unsafe { self.launch_raw(grid, block, shared_mem_bytes, stream.raw(), args) }
    }

    /// Raw-stream variant. Used when the caller already holds a
    /// `hipStream_t` and not a `&HipStream` wrapper — e.g. the graph
    /// executor, which keeps a `HipStream` as a field but hands raw
    /// stream handles to its FFI call sites.
    ///
    /// # Safety
    /// `stream` must be a live `hipStream_t` for the duration of the
    /// launch (until the kernel completes or the stream is synced).
    pub unsafe fn launch_raw(
        &self,
        grid: (u32, u32, u32),
        block: (u32, u32, u32),
        shared_mem_bytes: u32,
        stream: super::hip_ffi::hipStream_t,
        args: &mut [*mut c_void],
    ) -> HipResult<()> {
        let rc = hipModuleLaunchKernel(
            self.function,
            grid.0,
            grid.1,
            grid.2,
            block.0,
            block.1,
            block.2,
            shared_mem_bytes,
            stream,
            args.as_mut_ptr(),
            std::ptr::null_mut(),
        );
        check(rc, "hipModuleLaunchKernel")
    }

    pub fn raw(&self) -> hipFunction_t {
        self.function
    }
}
