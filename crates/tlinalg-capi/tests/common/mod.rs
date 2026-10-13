//! Shared helpers for the `tlinalg-capi` test suites.

#![allow(dead_code)]

use tlinalg_capi::tlinalg_strided;
use tlinalg_testkit::{Complex32, Complex64};

/// The last error message on this thread, for assertion diagnostics.
pub fn last_error() -> String {
    // SAFETY: the pointer is never null and stays valid until the next call.
    unsafe { std::ffi::CStr::from_ptr(tlinalg_capi::tlinalg_last_error()) }
        .to_string_lossy()
        .into_owned()
}

/// The backing arrays of a `tlinalg_strided`; keep it alive across the call it describes.
pub struct Layout {
    /// Element extents.
    pub extents: Vec<i64>,
    /// Element strides.
    pub strides: Vec<i64>,
    /// Element offset.
    pub offset: i64,
}

impl Layout {
    /// A layout with explicit extents and strides.
    pub fn new(extents: &[i64], strides: &[i64], offset: i64) -> Self {
        Self {
            extents: extents.to_vec(),
            strides: strides.to_vec(),
            offset,
        }
    }

    /// Compact column-major strides for `[d_0, d_1, ...]`, first axis fastest.
    pub fn compact(dims: &[usize]) -> Self {
        let extents = dims.iter().map(|&d| d as i64).collect::<Vec<_>>();
        let mut strides = vec![1i64; dims.len()];
        for i in 1..dims.len() {
            strides[i] = strides[i - 1] * dims[i - 1] as i64;
        }
        Self::new(&extents, &strides, 0)
    }

    /// The C descriptor borrowing this layout.
    pub fn as_c(&self) -> tlinalg_strided {
        tlinalg_strided {
            rank: self.extents.len() as i32,
            extents: self.extents.as_ptr(),
            strides: self.strides.as_ptr(),
            offset: self.offset,
        }
    }

    /// The extents as `usize`, for the provider's Rust entry points.
    pub fn dims_usize(&self) -> Vec<usize> {
        self.extents.iter().map(|&e| e as usize).collect()
    }

    /// The strides as `isize`, for the provider's Rust entry points.
    pub fn strides_isize(&self) -> Vec<isize> {
        self.strides.iter().map(|&s| s as isize).collect()
    }
}

/// Element counts of a compact `[rows, cols, b...]` input.
pub fn compact_dims(rows: usize, cols: usize, batch: usize) -> Vec<usize> {
    let mut dims = vec![rows, cols];
    if batch > 1 {
        dims.push(batch);
    }
    dims
}

/// Assert two `f64` slices agree within `tol`.
pub fn assert_close_f64(actual: &[f64], expected: &[f64], tol: f64) {
    assert_eq!(actual.len(), expected.len(), "length mismatch");
    for (i, (a, e)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (a - e).abs() <= tol,
            "index {i}: {a} vs expected {e} (tol {tol})"
        );
    }
}

/// Assert two `f32` slices agree within `tol`.
pub fn assert_close_f32(actual: &[f32], expected: &[f32], tol: f32) {
    assert_eq!(actual.len(), expected.len(), "length mismatch");
    for (i, (a, e)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (a - e).abs() <= tol,
            "index {i}: {a} vs expected {e} (tol {tol})"
        );
    }
}

/// Assert two `Complex64` slices agree within `tol`.
pub fn assert_close_c64(actual: &[Complex64], expected: &[Complex64], tol: f64) {
    assert_eq!(actual.len(), expected.len(), "length mismatch");
    for (i, (a, e)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (a - e).norm() <= tol,
            "index {i}: {a:?} vs expected {e:?} (tol {tol})"
        );
    }
}

/// Assert two `Complex32` slices agree within `tol`.
pub fn assert_close_c32(actual: &[Complex32], expected: &[Complex32], tol: f32) {
    assert_eq!(actual.len(), expected.len(), "length mismatch");
    for (i, (a, e)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (a - e).norm() <= tol,
            "index {i}: {a:?} vs expected {e:?} (tol {tol})"
        );
    }
}
