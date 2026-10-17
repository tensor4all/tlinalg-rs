//! `tlinalg_svd_sizes` / `tlinalg_svd_into` and the values-only pair.
//!
//! `svd` returns `U`, `min(rows, cols)` singular values and `Vᴴ` (not `V`), matching
//! [`tlinalg::svd::svd`]. `full` selects the square unitary factors (`u` is `rows x rows`, `vt` is
//! `cols x cols`); otherwise both thin factors use `min(rows, cols)`. The `s` output of `svd` is in
//! the scalar type (a complex scalar with a zero imaginary part for the complex dtypes), while
//! `svd_values` returns the real scalar, exactly as the provider's two entry points differ.

use std::ffi::c_void;

use num_complex::{Complex32, Complex64};
use tcontract_exec::Exec;
use tlinalg::{svd, FaerScalar, Op};

use crate::descriptor::{parse_layout, tlinalg_strided, with_view, TlinalgDtype};
use crate::executor::with_executor;
use crate::output::{check_output, copy_output, validate_capacity};
use crate::sizes::{batch_of, product, tlinalg_svd_out, tlinalg_svd_values_out};
use crate::status::{ffi, invalid, map_error, tlinalg_status, FfiError};

/// Validate the `full` flag.
fn validate_full(full: i32) -> Result<bool, FfiError> {
    match full {
        0 => Ok(false),
        1 => Ok(true),
        _ => Err(invalid(format!("full must be 0 or 1, got {full}"))),
    }
}

/// The `svd` output sizes for a validated descriptor.
fn svd_sizes(desc: *const tlinalg_strided, full: bool) -> Result<tlinalg_svd_out, FfiError> {
    let (dims, _, _) = parse_layout(desc)?;
    let (m, n) = (dims[0], dims[1]);
    let k = m.min(n);
    let (u_cols, v_cols) = if full { (m, n) } else { (k, k) };
    Ok(tlinalg_svd_out {
        batch: batch_of(&dims)?,
        u: product(&[m, u_cols])?,
        s: k,
        vt: product(&[v_cols, n])?,
        workspace: 0,
    })
}

/// The `svd_values` output sizes for a validated descriptor.
fn svd_values_sizes(desc: *const tlinalg_strided) -> Result<tlinalg_svd_values_out, FfiError> {
    let (dims, _, _) = parse_layout(desc)?;
    Ok(tlinalg_svd_values_out {
        batch: batch_of(&dims)?,
        s: dims[0].min(dims[1]),
        workspace: 0,
    })
}

#[allow(clippy::too_many_arguments)]
fn svd_into_typed<T: FaerScalar>(
    desc: *const tlinalg_strided,
    data: *const c_void,
    data_len: usize,
    full: bool,
    u: *mut c_void,
    s: *mut c_void,
    vt: *mut c_void,
    sizes: &tlinalg_svd_out,
    parallel: Exec<'_>,
) -> Result<(), FfiError> {
    let op = if full { Op::SvdFull } else { Op::Svd };
    with_view::<T, _>(desc, data, data_len, |input| {
        let (mut uu, mut ss, mut vv) = (Vec::new(), Vec::new(), Vec::new());
        svd::svd(op, input, full, &mut uu, &mut ss, &mut vv, parallel).map_err(map_error)?;
        let (u_total, s_total, vt_total) = sizes.totals()?;
        check_output(&uu, u, u_total, "u")?;
        check_output(&ss, s, s_total, "s")?;
        check_output(&vv, vt, vt_total, "vt")?;
        copy_output(&uu, u);
        copy_output(&ss, s);
        copy_output(&vv, vt);
        Ok(())
    })
}

fn svd_values_into_typed<T: FaerScalar>(
    desc: *const tlinalg_strided,
    data: *const c_void,
    data_len: usize,
    s: *mut c_void,
    sizes: &tlinalg_svd_values_out,
    parallel: Exec<'_>,
) -> Result<(), FfiError> {
    with_view::<T, _>(desc, data, data_len, |input| {
        let mut ss = Vec::new();
        svd::svd_values(Op::SvdValues, input, &mut ss, parallel).map_err(map_error)?;
        check_output(&ss, s, sizes.total()?, "s")?;
        copy_output(&ss, s);
        Ok(())
    })
}

/// `tlinalg_svd_sizes`: fill the per-item output element counts for an SVD.
///
/// # Safety
///
/// `input` is a valid descriptor or null; `out` is writable.
#[no_mangle]
pub unsafe extern "C" fn tlinalg_svd_sizes(
    input: *const tlinalg_strided,
    full: i32,
    out: *mut tlinalg_svd_out,
) -> tlinalg_status {
    ffi(|| {
        if out.is_null() {
            return Err(invalid("sizes output pointer is null"));
        }
        let sizes = svd_sizes(input, validate_full(full)?)?;
        // SAFETY: `out` was proved non-null above and is the caller's storage.
        unsafe { *out = sizes };
        Ok(())
    })
}

/// `tlinalg_svd_into`: write an SVD into caller-owned buffers.
///
/// The `s` output is `min(rows, cols)` elements per item in the scalar type of `dtype`. A buffer
/// shorter than the query reports is rejected before anything is written.
///
/// # Safety
///
/// Every pointer is valid per the header's contract; `_workspace` is unused in v1.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn tlinalg_svd_into(
    input: *const tlinalg_strided,
    dtype: i32,
    exec: isize,
    data: *const c_void,
    data_len: usize,
    full: i32,
    u: *mut c_void,
    u_len: usize,
    s: *mut c_void,
    s_len: usize,
    vt: *mut c_void,
    vt_len: usize,
    _workspace: *mut c_void,
    _workspace_len: usize,
) -> tlinalg_status {
    ffi(|| {
        let dtype = TlinalgDtype::from_tag(dtype)?;
        let full = validate_full(full)?;
        let sizes = svd_sizes(input, full)?;
        let (u_total, s_total, vt_total) = sizes.totals()?;
        validate_capacity("u", u, u_len, u_total)?;
        validate_capacity("s", s, s_len, s_total)?;
        validate_capacity("vt", vt, vt_len, vt_total)?;
        with_executor(exec, |parallel| match dtype {
            TlinalgDtype::F32 => {
                svd_into_typed::<f32>(input, data, data_len, full, u, s, vt, &sizes, parallel)
            }
            TlinalgDtype::F64 => {
                svd_into_typed::<f64>(input, data, data_len, full, u, s, vt, &sizes, parallel)
            }
            TlinalgDtype::C32 => {
                svd_into_typed::<Complex32>(input, data, data_len, full, u, s, vt, &sizes, parallel)
            }
            TlinalgDtype::C64 => {
                svd_into_typed::<Complex64>(input, data, data_len, full, u, s, vt, &sizes, parallel)
            }
        })
    })
}

/// `tlinalg_svd_values_sizes`: fill the per-item element counts for a singular-values-only query.
///
/// # Safety
///
/// `input` is a valid descriptor or null; `out` is writable.
#[no_mangle]
pub unsafe extern "C" fn tlinalg_svd_values_sizes(
    input: *const tlinalg_strided,
    out: *mut tlinalg_svd_values_out,
) -> tlinalg_status {
    ffi(|| {
        if out.is_null() {
            return Err(invalid("sizes output pointer is null"));
        }
        let sizes = svd_values_sizes(input)?;
        // SAFETY: `out` was proved non-null above and is the caller's storage.
        unsafe { *out = sizes };
        Ok(())
    })
}

/// `tlinalg_svd_values_into`: write singular values into a caller-owned buffer.
///
/// The `s` output is in the **real** scalar of `dtype`.
///
/// # Safety
///
/// Every pointer is valid per the header's contract; `_workspace` is unused in v1.
#[no_mangle]
pub unsafe extern "C" fn tlinalg_svd_values_into(
    input: *const tlinalg_strided,
    dtype: i32,
    exec: isize,
    data: *const c_void,
    data_len: usize,
    s: *mut c_void,
    s_len: usize,
    _workspace: *mut c_void,
    _workspace_len: usize,
) -> tlinalg_status {
    ffi(|| {
        let dtype = TlinalgDtype::from_tag(dtype)?;
        let sizes = svd_values_sizes(input)?;
        validate_capacity("s", s, s_len, sizes.total()?)?;
        with_executor(exec, |parallel| match dtype {
            TlinalgDtype::F32 => {
                svd_values_into_typed::<f32>(input, data, data_len, s, &sizes, parallel)
            }
            TlinalgDtype::F64 => {
                svd_values_into_typed::<f64>(input, data, data_len, s, &sizes, parallel)
            }
            TlinalgDtype::C32 => {
                svd_values_into_typed::<Complex32>(input, data, data_len, s, &sizes, parallel)
            }
            TlinalgDtype::C64 => {
                svd_values_into_typed::<Complex64>(input, data, data_len, s, &sizes, parallel)
            }
        })
    })
}
