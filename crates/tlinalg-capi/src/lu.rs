//! `tlinalg_lu_sizes` / `tlinalg_lu_into`.
//!
//! `lu` is partial-pivot LU of every `rows x cols` matrix, `P A = L U`. Per item the outputs are
//! the `rows x rows` permutation `P`, the unit-lower-trapezoidal `rows x min(rows, cols)` factor
//! `L`, the upper-trapezoidal `min(rows, cols) x cols` factor `U`, and the permutation parity
//! `+1`/`-1` in the scalar type. An exactly singular input is **not** an error, matching
//! [`tlinalg::lu::lu`].

use std::ffi::c_void;

use num_complex::{Complex32, Complex64};
use tlinalg::{lu, FaerScalar, Op, Parallel};

use crate::descriptor::{parse_layout, tlinalg_strided, with_view, TlinalgDtype};
use crate::output::{check_output, copy_output, validate_capacity};
use crate::sizes::{batch_of, product, tlinalg_lu_out};
use crate::status::{ffi, invalid, map_error, tlinalg_status, FfiError};

/// The `lu` output sizes for a validated descriptor.
fn lu_sizes(desc: *const tlinalg_strided) -> Result<tlinalg_lu_out, FfiError> {
    let (dims, _, _) = parse_layout(desc)?;
    let (m, n) = (dims[0], dims[1]);
    let k = m.min(n);
    Ok(tlinalg_lu_out {
        batch: batch_of(&dims)?,
        p: product(&[m, m])?,
        l: product(&[m, k])?,
        u: product(&[k, n])?,
        parity: 1,
        workspace: 0,
    })
}

#[allow(clippy::too_many_arguments)]
fn lu_into_typed<T: FaerScalar>(
    desc: *const tlinalg_strided,
    data: *const c_void,
    data_len: usize,
    p: *mut c_void,
    l: *mut c_void,
    u: *mut c_void,
    parity: *mut c_void,
    sizes: &tlinalg_lu_out,
) -> Result<(), FfiError> {
    with_view::<T, _>(desc, data, data_len, |input| {
        let (mut pp, mut ll, mut uu, mut par) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        lu::lu(
            Op::Lu,
            input,
            lu::LuFactors {
                p: &mut pp,
                l: &mut ll,
                u: &mut uu,
                parity: &mut par,
            },
            Parallel::Sequential,
        )
        .map_err(map_error)?;
        let (p_total, l_total, u_total, par_total) = sizes.totals()?;
        check_output(&pp, p, p_total, "p")?;
        check_output(&ll, l, l_total, "l")?;
        check_output(&uu, u, u_total, "u")?;
        check_output(&par, parity, par_total, "parity")?;
        copy_output(&pp, p);
        copy_output(&ll, l);
        copy_output(&uu, u);
        copy_output(&par, parity);
        Ok(())
    })
}

/// `tlinalg_lu_sizes`: fill the per-item output element counts for a partial-pivot LU factorization.
///
/// # Safety
///
/// `input` is a valid descriptor or null; `out` is writable.
#[no_mangle]
pub unsafe extern "C" fn tlinalg_lu_sizes(
    input: *const tlinalg_strided,
    out: *mut tlinalg_lu_out,
) -> tlinalg_status {
    ffi(|| {
        if out.is_null() {
            return Err(invalid("sizes output pointer is null"));
        }
        let sizes = lu_sizes(input)?;
        // SAFETY: `out` was proved non-null above and is the caller's storage.
        unsafe { *out = sizes };
        Ok(())
    })
}

/// `tlinalg_lu_into`: write partial-pivot LU factors into caller-owned buffers.
///
/// # Safety
///
/// Every pointer is valid per the header's contract; `_workspace` is unused in v1.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn tlinalg_lu_into(
    input: *const tlinalg_strided,
    dtype: i32,
    data: *const c_void,
    data_len: usize,
    p: *mut c_void,
    p_len: usize,
    l: *mut c_void,
    l_len: usize,
    u: *mut c_void,
    u_len: usize,
    parity: *mut c_void,
    parity_len: usize,
    _workspace: *mut c_void,
    _workspace_len: usize,
) -> tlinalg_status {
    ffi(|| {
        let dtype = TlinalgDtype::from_tag(dtype)?;
        let sizes = lu_sizes(input)?;
        let (p_total, l_total, u_total, par_total) = sizes.totals()?;
        validate_capacity("p", p, p_len, p_total)?;
        validate_capacity("l", l, l_len, l_total)?;
        validate_capacity("u", u, u_len, u_total)?;
        validate_capacity("parity", parity, parity_len, par_total)?;
        match dtype {
            TlinalgDtype::F32 => {
                lu_into_typed::<f32>(input, data, data_len, p, l, u, parity, &sizes)
            }
            TlinalgDtype::F64 => {
                lu_into_typed::<f64>(input, data, data_len, p, l, u, parity, &sizes)
            }
            TlinalgDtype::C32 => {
                lu_into_typed::<Complex32>(input, data, data_len, p, l, u, parity, &sizes)
            }
            TlinalgDtype::C64 => {
                lu_into_typed::<Complex64>(input, data, data_len, p, l, u, parity, &sizes)
            }
        }
    })
}
