//! `tlinalg_full_piv_lu_sizes` / `tlinalg_full_piv_lu_into`.
//!
//! `full_piv_lu` is full-pivot LU of every square matrix, `P A Qᵀ = L U`. Per item the outputs are
//! the `n x n` row permutation `P`, the unit-lower-triangular `L`, the upper-triangular `U`, the
//! `n x n` column permutation `Q`, and the parity `+1`/`-1` in the scalar type. The factorization
//! never fails on an exactly singular input, matching [`tlinalg::full_piv_lu::full_piv_lu`].

use std::ffi::c_void;

use num_complex::{Complex32, Complex64};
use tcontract_exec::Exec;
use tlinalg::{full_piv_lu, FaerScalar, Op};

use crate::descriptor::{parse_layout, tlinalg_strided, with_view, TlinalgDtype};
use crate::executor::with_executor;
use crate::output::{check_output, copy_output, validate_capacity};
use crate::sizes::{batch_of, product, require_square, tlinalg_full_piv_lu_out};
use crate::status::{ffi, invalid, map_error, tlinalg_status, FfiError};

/// The `full_piv_lu` output sizes for a validated descriptor.
fn full_piv_lu_sizes(desc: *const tlinalg_strided) -> Result<tlinalg_full_piv_lu_out, FfiError> {
    let (dims, _, _) = parse_layout(desc)?;
    let n = require_square(&dims)?;
    let nn = product(&[n, n])?;
    Ok(tlinalg_full_piv_lu_out {
        batch: batch_of(&dims)?,
        p: nn,
        l: nn,
        u: nn,
        q: nn,
        parity: 1,
        workspace: 0,
    })
}

#[allow(clippy::too_many_arguments)]
fn full_piv_lu_into_typed<T: FaerScalar>(
    desc: *const tlinalg_strided,
    data: *const c_void,
    data_len: usize,
    p: *mut c_void,
    l: *mut c_void,
    u: *mut c_void,
    q: *mut c_void,
    parity: *mut c_void,
    sizes: &tlinalg_full_piv_lu_out,
    parallel: &mut Exec<'_>,
) -> Result<(), FfiError> {
    with_view::<T, _>(desc, data, data_len, |input| {
        let (mut pp, mut ll, mut uu, mut qq, mut par) =
            (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new());
        full_piv_lu::full_piv_lu(
            Op::FullPivLu,
            input,
            full_piv_lu::FullPivLuFactors {
                p: &mut pp,
                l: &mut ll,
                u: &mut uu,
                q: &mut qq,
                parity: &mut par,
            },
            parallel,
        )
        .map_err(map_error)?;
        let (p_total, l_total, u_total, q_total, par_total) = sizes.totals()?;
        check_output(&pp, p, p_total, "p")?;
        check_output(&ll, l, l_total, "l")?;
        check_output(&uu, u, u_total, "u")?;
        check_output(&qq, q, q_total, "q")?;
        check_output(&par, parity, par_total, "parity")?;
        copy_output(&pp, p);
        copy_output(&ll, l);
        copy_output(&uu, u);
        copy_output(&qq, q);
        copy_output(&par, parity);
        Ok(())
    })
}

/// `tlinalg_full_piv_lu_sizes`: fill the per-item element counts for a full-pivot LU factorization.
///
/// # Safety
///
/// `input` is a valid descriptor or null; `out` is writable.
#[no_mangle]
pub unsafe extern "C" fn tlinalg_full_piv_lu_sizes(
    input: *const tlinalg_strided,
    out: *mut tlinalg_full_piv_lu_out,
) -> tlinalg_status {
    ffi(|| {
        if out.is_null() {
            return Err(invalid("sizes output pointer is null"));
        }
        let sizes = full_piv_lu_sizes(input)?;
        // SAFETY: `out` was proved non-null above and is the caller's storage.
        unsafe { *out = sizes };
        Ok(())
    })
}

/// `tlinalg_full_piv_lu_into`: write full-pivot LU factors into caller-owned buffers.
///
/// # Safety
///
/// Every pointer is valid per the header's contract; `_workspace` is unused in v1.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn tlinalg_full_piv_lu_into(
    input: *const tlinalg_strided,
    dtype: i32,
    exec: isize,
    data: *const c_void,
    data_len: usize,
    p: *mut c_void,
    p_len: usize,
    l: *mut c_void,
    l_len: usize,
    u: *mut c_void,
    u_len: usize,
    q: *mut c_void,
    q_len: usize,
    parity: *mut c_void,
    parity_len: usize,
    _workspace: *mut c_void,
    _workspace_len: usize,
) -> tlinalg_status {
    ffi(|| {
        let dtype = TlinalgDtype::from_tag(dtype)?;
        let sizes = full_piv_lu_sizes(input)?;
        let (p_total, l_total, u_total, q_total, par_total) = sizes.totals()?;
        validate_capacity("p", p, p_len, p_total)?;
        validate_capacity("l", l, l_len, l_total)?;
        validate_capacity("u", u, u_len, u_total)?;
        validate_capacity("q", q, q_len, q_total)?;
        validate_capacity("parity", parity, parity_len, par_total)?;
        with_executor(exec, |parallel| match dtype {
            TlinalgDtype::F32 => full_piv_lu_into_typed::<f32>(
                input, data, data_len, p, l, u, q, parity, &sizes, parallel,
            ),
            TlinalgDtype::F64 => full_piv_lu_into_typed::<f64>(
                input, data, data_len, p, l, u, q, parity, &sizes, parallel,
            ),
            TlinalgDtype::C32 => full_piv_lu_into_typed::<Complex32>(
                input, data, data_len, p, l, u, q, parity, &sizes, parallel,
            ),
            TlinalgDtype::C64 => full_piv_lu_into_typed::<Complex64>(
                input, data, data_len, p, l, u, q, parity, &sizes, parallel,
            ),
        })
    })
}
