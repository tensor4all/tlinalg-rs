//! Writing library-computed results into caller-owned buffers.
//!
//! Two kinds of checks run at different times. [`validate_capacity`] is element-type independent and
//! runs before the provider does any work, so a too-short or null buffer is rejected before it
//! could be written. [`check_output`] runs on every computed output before any of them is copied,
//! so a call that cannot finish writing writes nothing.

use std::ffi::c_void;
use std::mem::{align_of, size_of};

use crate::status::{invalid, shape, FfiError};

/// Reject a caller buffer that cannot hold `required` elements, or a null buffer that must hold at
/// least one.
pub(crate) fn validate_capacity(
    name: &str,
    ptr: *mut c_void,
    capacity: usize,
    required: usize,
) -> Result<(), FfiError> {
    if capacity < required {
        return Err(shape(format!(
            "{name} buffer holds {capacity} elements, {required} required"
        )));
    }
    if required > 0 && ptr.is_null() {
        return Err(invalid(format!("{name} buffer is null")));
    }
    Ok(())
}

/// Check a computed output against its buffer: element count, null and alignment.
pub(crate) fn check_output<T>(
    src: &[T],
    ptr: *mut c_void,
    capacity: usize,
    name: &str,
) -> Result<(), FfiError> {
    if src.len() > capacity {
        return Err(shape(format!(
            "{name} has {} elements, buffer holds {capacity}",
            src.len()
        )));
    }
    if !src.is_empty() {
        if ptr.is_null() {
            return Err(invalid(format!("{name} buffer is null")));
        }
        if !(ptr as usize).is_multiple_of(align_of::<T>()) {
            return Err(invalid(format!(
                "{name} buffer is not aligned for a {}-byte scalar",
                size_of::<T>()
            )));
        }
    }
    Ok(())
}

/// Copy a computed output into its buffer. The matching [`check_output`] must have passed.
pub(crate) fn copy_output<T: Copy>(src: &[T], ptr: *mut c_void) {
    if src.is_empty() {
        return;
    }
    // SAFETY: `check_output` proved non-null, aligned storage of at least `src.len()` elements;
    // `src` is a fresh library-owned vector, so the ranges do not overlap.
    unsafe { std::ptr::copy_nonoverlapping(src.as_ptr(), ptr.cast::<T>(), src.len()) };
}
