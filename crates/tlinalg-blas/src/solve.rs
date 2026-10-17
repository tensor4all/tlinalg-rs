//! Batched linear solve `op(A) X = B` through partial-pivot LU on LAPACK (`?getrf` + `?getrs`).
//!
//! Moved from tenferro-linalg's LAPACK backend (same project, MIT OR Apache-2.0). Two routes, as
//! the host had them:
//!
//! * [`solve`] fills an owned, compact output;
//! * [`solve_into`] writes into a caller-owned strided output. Each item's right-hand side is
//!   copied into the output only **after** that item factored successfully, so an exactly singular
//!   `A` leaves that item's output untouched.
//!
//! Exactly singular `A` is [`Error::Singular`] on both routes. One LU copy and one pivot buffer are
//! acquired from the workspace per call, reused across the batch, and **released on success**, so a
//! host that recycles buffers sees steady-state solve calls allocate nothing. An error path (a
//! singular factor, an illegal argument, a positive `?getrs` `info`) abandons them instead: the
//! workspace's abandonment contract leaves unwind replenishment to the host owner.

use strided_view::{RawStridedMut, RawStridedRef};

use crate::batch::{batch_len, clear_on_error, Input, Output};
use crate::common::{check_info, check_scratch, checked_product, dim_i32};
use crate::{Error, IndexWorkspace, LapackScalar, Op, Result, Workspace};

/// `?getrf` on one compact `n x n` matrix, rejecting exact singularity.
fn getrf_nonsingular<T: LapackScalar>(
    op: Op,
    n_i32: i32,
    lu: &mut [T],
    ipiv: &mut [i32],
) -> Result<()> {
    let mut info = 0;
    // SAFETY: callers pass a compact `n x n` matrix and `n` pivots.
    unsafe {
        T::getrf(n_i32, n_i32, lu, n_i32.max(1), ipiv, &mut info);
    }
    check_info(op, "getrf", info.min(0))?;
    if info > 0 {
        return Err(Error::Singular { op });
    }
    Ok(())
}

/// `?getrs` on one factored system.
#[allow(clippy::too_many_arguments)]
fn getrs<T: LapackScalar>(
    op: Op,
    transpose_a: bool,
    n_i32: i32,
    nrhs_i32: i32,
    lu: &[T],
    ipiv: &[i32],
    rhs: &mut [T],
    ldb_i32: i32,
) -> Result<()> {
    let mut info = 0;
    // SAFETY: `lu`/`ipiv` are this system's `?getrf` factors (pivots in range by LAPACK's
    // contract) and `rhs` covers the `ldb`-strided `n x nrhs` block.
    unsafe {
        T::getrs(
            if transpose_a { b'T' } else { b'N' },
            n_i32,
            nrhs_i32,
            lu,
            n_i32.max(1),
            ipiv,
            rhs,
            ldb_i32,
            &mut info,
        );
    }
    check_info(op, "getrs", info)
}

/// The shared shape checks of both routes: `(n, nrhs)`.
fn system_dims<T: Copy>(op: Op, a: &Input<'_, T>, b: &Input<'_, T>) -> Result<(usize, usize)> {
    let n = a.square(op)?;
    if b.layout.rows != n {
        return Err(Error::InvalidArgument {
            op,
            role: "shape",
            detail: format!(
                "the right-hand side has {} rows, expected {n}",
                b.layout.rows
            ),
        });
    }
    a.layout.same_batch(op, &b.layout)?;
    Ok((n, b.layout.cols))
}

/// Solve `op(A) X = B` for every system of a batch into an owned output.
///
/// `a` has dims `[n, n, batch...]` and `b` dims `[n, nrhs, batch...]` with the same batch dims;
/// neither is modified. `op(A)` is `A`, or `Aᵀ` with `transpose_a`. `out` is cleared and receives
/// `X` (`n * nrhs` elements per system, compact column-major in batch order).
///
/// # Errors
///
/// [`Error::Singular`] for an exactly singular matrix, [`Error::InvalidArgument`] for malformed or
/// mismatched operands, a dimension outside the LAPACK `i32` range or an illegal LAPACK argument,
/// and [`Error::NonConvergence`] for a positive `?getrs` `info`; always for the first failing
/// system in batch order. On error `out` is left empty.
pub fn solve<T, W>(
    op: Op,
    transpose_a: bool,
    a: RawStridedRef<'_, T>,
    b: RawStridedRef<'_, T>,
    out: &mut Vec<T>,
    workspace: &mut W,
) -> Result<()>
where
    T: LapackScalar,
    W: Workspace<T> + IndexWorkspace,
{
    out.clear();
    let result = solve_run(op, transpose_a, a, b, out, workspace);
    clear_on_error(result, || out.clear())
}

fn solve_run<T, W>(
    op: Op,
    transpose_a: bool,
    a: RawStridedRef<'_, T>,
    b: RawStridedRef<'_, T>,
    out: &mut Vec<T>,
    workspace: &mut W,
) -> Result<()>
where
    T: LapackScalar,
    W: Workspace<T> + IndexWorkspace,
{
    let a = Input::new(op, "a", a)?;
    let b = Input::new(op, "b", b)?;
    let (n, nrhs) = system_dims(op, &a, &b)?;
    let matrix_len = checked_product(op, "matrix", &[n, n])?;
    let rhs_len = checked_product(op, "rhs", &[n, nrhs])?;
    let n_i32 = dim_i32(op, n)?;
    let nrhs_i32 = dim_i32(op, nrhs)?;
    let count = b.layout.count;
    if matrix_len == 0 || rhs_len == 0 || count == 0 {
        return Ok(());
    }
    out.reserve(batch_len(op, rhs_len, count)?);
    let mut lu = workspace.acquire_capacity(matrix_len);
    let mut ipiv = workspace.acquire_zeroed_index(n);
    check_scratch(op, ipiv.len(), n)?;
    // INVARIANT: both operands describe `count` items in the same batch order. LAPACK overwrites
    // only the private LU scratch and this item's block of `out`. The serial loop is intentional:
    // LAPACK owns threading, and the scratch is reused across the batch.
    for (a_offset, b_offset) in a.layout.offsets().zip(b.layout.offsets()) {
        lu.clear();
        a.gather(a_offset, &mut lu);
        getrf_nonsingular(op, n_i32, &mut lu, &mut ipiv[..n])?;
        let start = out.len();
        b.gather(b_offset, out);
        getrs(
            op,
            transpose_a,
            n_i32,
            nrhs_i32,
            &lu,
            &ipiv[..n],
            &mut out[start..],
            n_i32,
        )?;
    }
    workspace.release(lu);
    workspace.release_index(ipiv);
    Ok(())
}

/// Solve `op(A) X = B` for every system of a batch directly into a caller-owned strided output.
///
/// `a` and `b` are as for [`solve`]; `out` has dims `[n, nrhs, batch...]` with the same batch dims,
/// **unit row stride** and a column stride of at least `n` (any column stride for one column). Systems are solved in batch order; an item's output is written only
/// once its `A` has factored, so the first failing item and every later item keep their previous
/// contents, while earlier items hold their solutions.
///
/// # Errors
///
/// [`Error::InvalidArgument`] with role `"out"` for an output without unit row stride or with a
/// column stride below `n`, plus everything [`solve`] reports.
pub fn solve_into<T, W>(
    op: Op,
    transpose_a: bool,
    a: RawStridedRef<'_, T>,
    b: RawStridedRef<'_, T>,
    mut out: RawStridedMut<'_, T>,
    workspace: &mut W,
) -> Result<()>
where
    T: LapackScalar,
    W: Workspace<T> + IndexWorkspace,
{
    let a = Input::new(op, "a", a)?;
    let b = Input::new(op, "b", b)?;
    let (n, nrhs) = system_dims(op, &a, &b)?;
    let mut out = Output::new(op, "out", &mut out)?;
    if (out.layout.rows, out.layout.cols) != (n, nrhs) {
        return Err(Error::InvalidArgument {
            op,
            role: "out",
            detail: format!(
                "output is {}x{}, expected {n}x{nrhs}",
                out.layout.rows, out.layout.cols
            ),
        });
    }
    b.layout.same_batch(op, &out.layout)?;
    if n > 1 && out.layout.row_stride != 1 {
        return Err(Error::InvalidArgument {
            op,
            role: "out",
            detail: "direct LAPACK solve requires unit row stride".to_owned(),
        });
    }
    let ldb = out
        .layout
        .lapack_lda()
        .ok_or_else(|| Error::InvalidArgument {
            op,
            role: "out",
            detail: "output leading dimension is smaller than the row count".to_owned(),
        })?;
    let matrix_len = checked_product(op, "matrix", &[n, n])?;
    let n_i32 = dim_i32(op, n)?;
    let nrhs_i32 = dim_i32(op, nrhs)?;
    let ldb_i32 = dim_i32(op, ldb)?;
    let count = b.layout.count;
    if matrix_len == 0 || nrhs == 0 || count == 0 {
        return Ok(());
    }
    let mut lu = workspace.acquire_capacity(matrix_len);
    let mut ipiv = workspace.acquire_zeroed_index(n);
    check_scratch(op, ipiv.len(), n)?;
    let offsets = a
        .layout
        .offsets()
        .zip(b.layout.offsets())
        .zip(out.layout.offsets());
    for ((a_offset, b_offset), out_offset) in offsets {
        lu.clear();
        a.gather(a_offset, &mut lu);
        getrf_nonsingular(op, n_i32, &mut lu, &mut ipiv[..n])?;
        let rhs = out.item_mut(out_offset, ldb);
        for col in 0..nrhs {
            let column = &mut rhs[col * ldb..col * ldb + n];
            b.copy_column(b_offset, col, column);
        }
        getrs(
            op,
            transpose_a,
            n_i32,
            nrhs_i32,
            &lu,
            &ipiv[..n],
            rhs,
            ldb_i32,
        )?;
    }
    workspace.release(lu);
    workspace.release_index(ipiv);
    Ok(())
}
