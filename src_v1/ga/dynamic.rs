//! `DynamicKernel` — owns a dynamically-loaded HIP module + looked-up
//! function pointer + the launch parameters derived from the
//! parametric codegen. Introduced in Phase 2 / step 2.1.3 Block B.
//!
//! The struct is intentionally kernel-specific (gate_up_swiglu today)
//! because each codegen variant has its own argument layout and grid
//! sizing. A second kernel (Q6_K LM-head, residual-fused GEMV, …)
//! gets its own `launch_*` helper here; we don't try to hide the
//! kernel-specific wiring behind a trait.

#![cfg(feature = "gpu")]

use std::ffi::c_void;

use crate::v1::backend::gpu::error::HipResult;
use crate::v1::backend::gpu::module::{HipFunction, HipModule};
use crate::v1::backend::gpu::wrappers::{HipBuffer, HipStream};

/// Geometry knobs for a compiled gate_up_swiglu kernel — everything
/// the host-side launcher needs beyond the module+function handles.
/// Fields mirror the `#define`s the codegen substituted into the
/// emitted HIP source.
#[derive(Debug, Clone, Copy)]
pub struct GateUpSwigluGeometry {
    pub num_waves: u32,
    pub multi_row_cols: u32, // 4 for the Block-B kernel
    pub warp_size: u32,      // 32 on gfx1201
}

impl GateUpSwigluGeometry {
    /// Values baked into the Block-B parametric kernel. `num_waves`
    /// varies per GA candidate; everything else is fixed for now
    /// (these dimensions move into the parametric axis in later
    /// blocks).
    pub fn for_num_waves(num_waves: u32) -> Self {
        Self {
            num_waves,
            multi_row_cols: 4,
            warp_size: 32,
        }
    }

    pub fn threads_per_block(&self) -> u32 {
        self.num_waves * self.warp_size
    }

    /// Columns the block spans in total.
    pub fn cols_per_block(&self) -> u32 {
        self.num_waves * self.multi_row_cols
    }

    pub fn grid_x(&self, ncols_dst: u32) -> u32 {
        let cpb = self.cols_per_block();
        (ncols_dst + cpb - 1) / cpb
    }

    /// Dynamic shared-memory footprint: n_rows × sizeof(float) for the
    /// FP32 input-staging buffer the kernel declares as
    /// `extern __shared__ float s_input[]`.
    pub fn shared_mem_bytes(&self, n_rows: u32) -> u32 {
        n_rows * std::mem::size_of::<f32>() as u32
    }
}

/// Owns the loaded module (via `HipModule` RAII) and carries the
/// kernel-specific geometry. Dropping the `DynamicKernel` unloads the
/// module.
pub struct DynamicKernel {
    module: HipModule,
    symbol: String,
    geometry: GateUpSwigluGeometry,
}

impl DynamicKernel {
    /// Build a new `DynamicKernel` from a `.co` byte buffer (produced
    /// by `ga::compile::compile_hip_source`) and the kernel's
    /// symbol + geometry.
    pub fn from_code_object(
        code_object: &[u8],
        symbol: String,
        geometry: GateUpSwigluGeometry,
    ) -> HipResult<Self> {
        let module = HipModule::load(code_object)?;
        // Touch the function up front to validate the symbol exists
        // — if it doesn't, fail now instead of on every launch.
        let _ = module.get_function(&symbol)?;
        Ok(Self {
            module,
            symbol,
            geometry,
        })
    }

    pub fn geometry(&self) -> GateUpSwigluGeometry {
        self.geometry
    }

    pub fn symbol(&self) -> &str {
        &self.symbol
    }

    /// Re-resolve the function handle. Cheap — hipModuleGetFunction is
    /// a pointer lookup. We don't cache it on the struct because
    /// `HipFunction` borrows the `HipModule`, and carrying that
    /// lifetime through the struct would force a self-referential
    /// layout.
    pub fn function(&self) -> HipResult<HipFunction<'_>> {
        self.module.get_function(&self.symbol)
    }

    /// Launch gate_up_swiglu on pre-allocated GPU buffers. `n_rows`
    /// is the K-dimension (matches the static Phase-1 launcher's
    /// terminology); `ncols_dst` is N.
    pub fn launch_gate_up_swiglu(
        &self,
        weights_gate: &HipBuffer,
        weights_up: &HipBuffer,
        input: &HipBuffer,
        swiglu_out: &mut HipBuffer,
        n_rows: i32,
        ncols_dst: i32,
        stream: &HipStream,
    ) -> HipResult<()> {
        let func = self.function()?;

        // Argument addresses — `hipModuleLaunchKernel` expects an
        // array of pointers to argument values. For pointer args the
        // "value" is the raw device pointer (a `*mut c_void`); for
        // ints it's the int itself.
        //
        // The kernel sees the args by copy (pass-by-value), but the
        // `kernelParams` array needs `&arg` for each arg.
        let gate_ptr = weights_gate.as_ptr();
        let up_ptr = weights_up.as_ptr();
        let in_ptr = input.as_ptr();
        let out_ptr = swiglu_out.as_mut_ptr();

        let mut args: [*mut c_void; 6] = [
            &gate_ptr as *const _ as *mut c_void,
            &up_ptr as *const _ as *mut c_void,
            &in_ptr as *const _ as *mut c_void,
            &out_ptr as *const _ as *mut c_void,
            &n_rows as *const _ as *mut c_void,
            &ncols_dst as *const _ as *mut c_void,
        ];

        let grid = (self.geometry.grid_x(ncols_dst as u32), 1u32, 1u32);
        let block = (self.geometry.threads_per_block(), 1u32, 1u32);
        let shared = self.geometry.shared_mem_bytes(n_rows as u32);

        func.launch(grid, block, shared, stream, &mut args)
    }
}
