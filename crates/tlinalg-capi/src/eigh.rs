//! `tlinalg_eigh_sizes` / `tlinalg_eigh_into` and the values-only pair.
//!
//! `eigh` reads the **lower triangle** of each Hermitian input and returns non-decreasing **real**
//! eigenvalues. `eigh` writes the eigenvalues in the scalar type (a zero imaginary part for the
//! complex dtypes) and the column-major eigenvectors; `eigh_values` writes only the real
//! eigenvalues, matching the provider's two entry points.

use std::ffi::c_void;

use num_complex::{Complex32, Complex64};
use tlinalg::{eigh, FaerScalar, Op, Parallel};

use crate::descriptor::{parse_layout, tlinalg_strided, with_view, TlinalgDtype};
use crate::output::{check_output, copy_output, validate_capacity};
use crate::sizes::{batch_of, product, require_square, tlinalg_eigh_out, tlinalg_eigh_values_out};
use crate::status::{ffi, invalid, map_error, tlinalg_status, FfiError};

/// The `eigh` output sizes for a validated descriptor.
fn eigh_sizes(desc: *const tlinalg_strided) -> Result<tlinalg_eigh_out, FfiError> {
    let (dims, _, _) = parse_layout(desc)?;
    let n = require_square(&dims)?;
    Ok(tlinalg_eigh_out {
        batch: batch_of(&dims)?,
        values: n,
        vectors: product(&[n, n])?,
        workspace: 0,
    })
}

/// The `eigh_values` output sizes for a validated descriptor.
fn eigh_values_sizes(desc: *const tlinalg_strided) -> Result<tlinalg_eigh_values_out, FfiError> {
    let (dims, _, _) = parse_layout(desc)?;
    let n = require_square(&dims)?;
    Ok(tlinalg_eigh_values_out {
        batch: batch_of(&dims)?,
        values: n,
        workspace: 0,
    })
}

#[allow(clippy::too_many_arguments)]
fn eigh_into_typed<T: FaerScalar>(
    desc: *const tlinalg_strided,
    data: *const c_void,
    data_len: usize,
    values: *mut c_void,
    vectors: *mut c_void,
    sizes: &tlinalg_eigh_out,
) -> Result<(), FfiError> {
    with_view::<T, _>(desc, data, data_len, |input| {
        let (mut vv, mut ww) = (Vec::new(), Vec::new());
        eigh::eigh(Op::Eigh, input, &mut vv, &mut ww, Parallel::Sequential).map_err(map_error)?;
        let (values_total, vectors_total) = sizes.totals()?;
        check_output(&vv, values, values_total, "values")?;
        check_output(&ww, vectors, vectors_total, "vectors")?;
        copy_output(&vv, values);
        copy_output(&ww, vectors);
        Ok(())
    })
}

fn eigh_values_into_typed<T: FaerScalar>(
    desc: *const tlinalg_strided,
    data: *const c_void,
    data_len: usize,
    values: *mut c_void,
    sizes: &tlinalg_eigh_values_out,
) -> Result<(), FfiError> {
    with_view::<T, _>(desc, data, data_len, |input| {
        let mut vv = Vec::new();
        eigh::eigh_values(Op::EighValues, input, &mut vv, Parallel::Sequential)
            .map_err(map_error)?;
        check_output(&vv, values, sizes.total()?, "values")?;
        copy_output(&vv, values);
        Ok(())
    })
}

/// `tlinalg_eigh_sizes`: fill the per-item output element counts for a Hermitian eigendecomposition.
///
/// # Safety
///
/// `input` is a valid descriptor or null; `out` is writable.
#[no_mangle]
pub unsafe extern "C" fn tlinalg_eigh_sizes(
    input: *const tlinalg_strided,
    out: *mut tlinalg_eigh_out,
) -> tlinalg_status {
    ffi(|| {
        if out.is_null() {
            return Err(invalid("sizes output pointer is null"));
        }
        let sizes = eigh_sizes(input)?;
        // SAFETY: `out` was proved non-null above and is the caller's storage.
        unsafe { *out = sizes };
        Ok(())
    })
}

/// `tlinalg_eigh_into`: write a Hermitian eigendecomposition into caller-owned buffers.
///
/// # Safety
///
/// Every pointer is valid per the header's contract; `_workspace` is unused in v1.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn tlinalg_eigh_into(
    input: *const tlinalg_strided,
    dtype: i32,
    data: *const c_void,
    data_len: usize,
    values: *mut c_void,
    values_len: usize,
    vectors: *mut c_void,
    vectors_len: usize,
    _workspace: *mut c_void,
    _workspace_len: usize,
) -> tlinalg_status {
    ffi(|| {
        let dtype = TlinalgDtype::from_tag(dtype)?;
        let sizes = eigh_sizes(input)?;
        let (values_total, vectors_total) = sizes.totals()?;
        validate_capacity("values", values, values_len, values_total)?;
        validate_capacity("vectors", vectors, vectors_len, vectors_total)?;
        match dtype {
            TlinalgDtype::F32 => {
                eigh_into_typed::<f32>(input, data, data_len, values, vectors, &sizes)
            }
            TlinalgDtype::F64 => {
                eigh_into_typed::<f64>(input, data, data_len, values, vectors, &sizes)
            }
            TlinalgDtype::C32 => {
                eigh_into_typed::<Complex32>(input, data, data_len, values, vectors, &sizes)
            }
            TlinalgDtype::C64 => {
                eigh_into_typed::<Complex64>(input, data, data_len, values, vectors, &sizes)
            }
        }
    })
}

/// `tlinalg_eigh_values_sizes`: fill the per-item element counts for an eigenvalues-only query.
///
/// # Safety
///
/// `input` is a valid descriptor or null; `out` is writable.
#[no_mangle]
pub unsafe extern "C" fn tlinalg_eigh_values_sizes(
    input: *const tlinalg_strided,
    out: *mut tlinalg_eigh_values_out,
) -> tlinalg_status {
    ffi(|| {
        if out.is_null() {
            return Err(invalid("sizes output pointer is null"));
        }
        let sizes = eigh_values_sizes(input)?;
        // SAFETY: `out` was proved non-null above and is the caller's storage.
        unsafe { *out = sizes };
        Ok(())
    })
}

/// `tlinalg_eigh_values_into`: write Hermitian eigenvalues into a caller-owned buffer.
///
/// The `values` output is in the **real** scalar of `dtype`, non-decreasing.
///
/// # Safety
///
/// Every pointer is valid per the header's contract; `_workspace` is unused in v1.
#[no_mangle]
pub unsafe extern "C" fn tlinalg_eigh_values_into(
    input: *const tlinalg_strided,
    dtype: i32,
    data: *const c_void,
    data_len: usize,
    values: *mut c_void,
    values_len: usize,
    _workspace: *mut c_void,
    _workspace_len: usize,
) -> tlinalg_status {
    ffi(|| {
        let dtype = TlinalgDtype::from_tag(dtype)?;
        let sizes = eigh_values_sizes(input)?;
        validate_capacity("values", values, values_len, sizes.total()?)?;
        match dtype {
            TlinalgDtype::F32 => {
                eigh_values_into_typed::<f32>(input, data, data_len, values, &sizes)
            }
            TlinalgDtype::F64 => {
                eigh_values_into_typed::<f64>(input, data, data_len, values, &sizes)
            }
            TlinalgDtype::C32 => {
                eigh_values_into_typed::<Complex32>(input, data, data_len, values, &sizes)
            }
            TlinalgDtype::C64 => {
                eigh_values_into_typed::<Complex64>(input, data, data_len, values, &sizes)
            }
        }
    })
}
