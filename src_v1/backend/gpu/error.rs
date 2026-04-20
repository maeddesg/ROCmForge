//! HIP error handling: `HipError` with call-site context, `HipResult<T>`
//! alias, `check` helper, and a `hip_check!` macro that wraps the
//! `unsafe` FFI call.

use std::ffi::CStr;
use std::fmt;

use super::hip_ffi;

#[derive(Debug, Clone)]
pub struct HipError {
    pub code: i32,
    pub message: String,
    pub context: String,
}

impl fmt::Display for HipError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "HIP error {} ({}) in {}",
            self.code, self.message, self.context
        )
    }
}

impl std::error::Error for HipError {}

pub type HipResult<T> = Result<T, HipError>;

/// Translate a raw `hipError_t` into `Ok(())` or a `HipError` with a
/// human-readable description and a caller-supplied context tag.
pub fn check(error: hip_ffi::hipError_t, context: &str) -> HipResult<()> {
    if error == hip_ffi::HIP_SUCCESS {
        return Ok(());
    }
    let message = unsafe {
        let ptr = hip_ffi::hipGetErrorString(error);
        if ptr.is_null() {
            "unknown error".to_string()
        } else {
            CStr::from_ptr(ptr).to_string_lossy().into_owned()
        }
    };
    Err(HipError {
        code: error,
        message,
        context: context.to_string(),
    })
}

/// `hip_check!(hipMalloc(&mut p, n), "HipBuffer::new")?;`
///
/// Wraps the FFI call in `unsafe` and converts the return code to a
/// `HipResult<()>`. Import via `use super::error::hip_check;`.
#[macro_export]
macro_rules! hip_check {
    ($call:expr, $ctx:expr) => {
        $crate::v1::backend::gpu::error::check(unsafe { $call }, $ctx)
    };
}

pub use hip_check;
