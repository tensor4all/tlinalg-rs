//! `tlinalg_cholesky_sizes` / `tlinalg_cholesky_into`.
//!
//! `cholesky` reads the **lower triangle** of each Hermitian positive-definite input and writes the
//! lower-triangular factor `L` with `A = L Lᴴ`, column-major with zeros above the diagonal (the
//! provider's [`tlinalg::cholesky::cholesky`]). A non-positive-definite item is reported as
//! [`crate::TLINALG_ERR_NONCONVERGENCE`], which is the classification the provider uses.

use std::ffi::c_void;

use num_complex::{Complex32, Complex64};
use tlinalg::{cholesky, FaerScalar, Op, Parallel};

use crate::descriptor::{parse_layout, tlinalg_strided, with_view, TlinalgDtype};
use crate::executor::with_executor;
use crate::output::{check_output, copy_output, validate_capacity};
use crate::sizes::{batch_of, product, require_square, tlinalg_cholesky_out};
use crate::status::{ffi, invalid, map_error, tlinalg_status, FfiError};

/// The `cholesky` output sizes for a validated descriptor.
fn cholesky_sizes(desc: *const tlinalg_strided) -> Result<tlinalg_cholesky_out, FfiError> {
    let (dims, _, _) = parse_layout(desc)?;
    let n = require_square(&dims)?;
    Ok(tlinalg_cholesky_out {
        batch: batch_of(&dims)?,
        l: product(&[n, n])?,
        workspace: 0,
    })
}

fn cholesky_into_typed<T: FaerScalar>(
    desc: *const tlinalg_strided,
    data: *const c_void,
    data_len: usize,
    l: *mut c_void,
    sizes: &tlinalg_cholesky_out,
    parallel: Parallel<'_>,
) -> Result<(), FfiError> {
    with_view::<T, _>(desc, data, data_len, |input| {
        let mut ll = Vec::new();
        cholesky::cholesky(Op::Cholesky, input, &mut ll, parallel).map_err(map_error)?;
        check_output(&ll, l, sizes.total()?, "l")?;
        copy_output(&ll, l);
        Ok(())
    })
}

/// `tlinalg_cholesky_sizes`: fill the per-item output element counts for a Cholesky factorization.
///
/// # Safety
///
/// `input` is a valid descriptor or null; `out` is writable.
#[no_mangle]
pub unsafe extern "C" fn tlinalg_cholesky_sizes(
    input: *const tlinalg_strided,
    out: *mut tlinalg_cholesky_out,
) -> tlinalg_status {
    ffi(|| {
        if out.is_null() {
            return Err(invalid("sizes output pointer is null"));
        }
        let sizes = cholesky_sizes(input)?;
        // SAFETY: `out` was proved non-null above and is the caller's storage.
        unsafe { *out = sizes };
        Ok(())
    })
}

/// `tlinalg_cholesky_into`: write Cholesky factors into a caller-owned buffer.
///
/// # Safety
///
/// Every pointer is valid per the header's contract; `_workspace` is unused in v1.
#[no_mangle]
pub unsafe extern "C" fn tlinalg_cholesky_into(
    input: *const tlinalg_strided,
    dtype: i32,
    exec: isize,
    data: *const c_void,
    data_len: usize,
    l: *mut c_void,
    l_len: usize,
    _workspace: *mut c_void,
    _workspace_len: usize,
) -> tlinalg_status {
    ffi(|| {
        let dtype = TlinalgDtype::from_tag(dtype)?;
        let sizes = cholesky_sizes(input)?;
        validate_capacity("l", l, l_len, sizes.total()?)?;
        with_executor(exec, |parallel| match dtype {
            TlinalgDtype::F32 => {
                cholesky_into_typed::<f32>(input, data, data_len, l, &sizes, parallel)
            }
            TlinalgDtype::F64 => {
                cholesky_into_typed::<f64>(input, data, data_len, l, &sizes, parallel)
            }
            TlinalgDtype::C32 => {
                cholesky_into_typed::<Complex32>(input, data, data_len, l, &sizes, parallel)
            }
            TlinalgDtype::C64 => {
                cholesky_into_typed::<Complex64>(input, data, data_len, l, &sizes, parallel)
            }
        })
    })
}
