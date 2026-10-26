//! faer-backed batched Cholesky factorization.
//!
//! Moved from `tenferro-linalg`'s faer backend (same project, MIT OR Apache-2.0).
//!
//! The factorization reads the **lower** triangle of each `A` and returns the lower-triangular `L`
//! with `A = L Lᴴ`, column-major, zero above the diagonal. The output chunk is the destructive
//! work matrix and faer's stack scratch is per-lane scratch carved from the caller's borrowed
//! workspace, reused for every item of the lane.
//!
//! # Borrowed workspace
//!
//! [`cholesky_workspace`] reports the bytes a call needs and [`cholesky_into`] runs it from
//! caller-owned storage; [`cholesky`] is the wrapper.

use core::mem::MaybeUninit;

use faer::dyn_stack::{MemStack, StackReq};
use faer::MatMut;
use strided_view::RawStridedRef;

use crate::batch::{self, fixed_out, BatchedRef, Sink, VecOut};
use crate::scalar::ScalarEntity;
use crate::util::{checked_product, weighted_muladds};
use crate::workspace::{bytes, faer_req, lane_requirement, total_of};
use crate::{Error, FaerScalar, Op, Result};
use tcontract_exec::Exec;

fn cholesky_item<T: FaerScalar>(
    op: Op,
    mat: faer::MatRef<'_, T::Entity>,
    l: &mut Sink<'_, T>,
    mem: &mut [MaybeUninit<u8>],
    par: faer::Par<'_>,
) -> Result<()> {
    let n = mat.nrows();
    // The output chunk is the destructible copy. Fill it with `mat` and factor in place;
    // `cholesky_in_place` reads the lower triangle only, so the upper part is cleared afterwards.
    // The bulk `copy_from` is the optimized copy path the staging matrix used to get.
    let work = l.fill(n * n, |_| T::default());
    MatMut::from_column_major_slice_mut(T::entity_slice_mut(work), n, n).copy_from(mat);
    faer::linalg::cholesky::llt::factor::cholesky_in_place(
        MatMut::from_column_major_slice_mut(T::entity_slice_mut(work), n, n),
        Default::default(),
        par,
        MemStack::new(mem),
        Default::default(),
    )
    .map_err(|_| Error::NonConvergence { op })?;
    // faer may leave scratch values in the strictly-upper triangle; clear it so the output is the
    // documented lower-triangular `L` with zeros above, as `push_masked` used to write.
    for col in 0..n {
        work[col * n..col * n + col].fill(T::default());
    }
    Ok(())
}

/// faer's own factor scratch for `n x n`.
fn req<E: faer::traits::ComplexField>(n: usize, par: faer::ParShape) -> StackReq {
    faer::linalg::cholesky::llt::factor::cholesky_in_place_scratch::<E>(n, par, Default::default())
}

/// Reject a shape whose square the lane requirement multiplies (faer's request does).
fn check_shape(op: Op, n: usize) -> Result<()> {
    checked_product(op, "matrix", &[n, n])?;
    Ok(())
}

/// The work estimate a Cholesky factorization reports to the batch policy.
fn work<T: ScalarEntity>(n: usize) -> f64 {
    weighted_muladds::<T>((n as f64).powi(3) / 6.0)
}

/// Bytes one batched Cholesky call needs in its borrowed workspace.
///
/// # Errors
///
/// [`Error::InvalidArgument`] when a size overflows (canonical `Op`
/// is `Op::Cholesky`).
pub fn cholesky_workspace<T: FaerScalar>(n: usize, batch: usize, exec: &Exec<'_>) -> Result<usize> {
    cholesky_bytes::<T>(Op::Cholesky, n, batch, exec)
}

/// [`cholesky_workspace`] with the caller's operation, so a wrapper reports its own `Op`.
fn cholesky_bytes<T: FaerScalar>(op: Op, n: usize, batch: usize, exec: &Exec<'_>) -> Result<usize> {
    let requirement = lane_requirement(
        op,
        exec,
        batch,
        work::<T>(n),
        || check_shape(op, n),
        |par| faer_req(req::<T::Entity>(n, par)),
    )?;
    match requirement {
        None => Ok(0),
        Some((lanes, per_lane)) => total_of(op, lanes, per_lane),
    }
}

/// Cholesky factors of every `n x n` Hermitian positive-definite matrix of a batch into
/// caller-owned storage.
///
/// `l` holds `n * n` uninitialised elements per batch item. Workspace contract as
/// [`crate::qr::qr_into`].
///
/// # Errors
///
/// [`Error::InvalidArgument`] when `input` is not a batch of square matrices or an output length
/// does not match, [`crate::Error::InsufficientWorkspace`] for a short workspace, and
/// [`Error::NonConvergence`] for the lowest-indexed item that is not numerically positive definite.
pub fn cholesky_into<T: FaerScalar>(
    op: Op,
    input: RawStridedRef<'_, T>,
    l: &mut [MaybeUninit<T>],
    workspace: &mut [MaybeUninit<u8>],
    exec: &mut Exec<'_>,
) -> Result<()> {
    let input = BatchedRef::square(op, "input", input)?;
    let n = input.rows();
    let l_len = checked_product(op, "L", &[n, n])?;
    let item_muladds = work::<T>(n);
    let requirement = lane_requirement(
        op,
        exec,
        input.batch(),
        item_muladds,
        || check_shape(op, n),
        |par| faer_req(req::<T::Entity>(n, par)),
    )?;
    let (lanes, per_lane) = requirement.unwrap_or((1, 0));
    batch::check_workspace(op, workspace, per_lane, lanes)?;
    batch::run(
        op,
        input.batch(),
        exec,
        item_muladds,
        &mut (fixed_out(l, l_len, "L"),),
        batch::ScratchArena::new(workspace, per_lane),
        |par, stack| {
            let stack = MemStack::new(stack);
            let (mem, _) = bytes(stack, req::<T::Entity>(n, par).unaligned_bytes_required());
            mem
        },
        |index, (l,), mem, par| cholesky_item::<T>(op, input.item(index), l, mem, par),
    )
}

/// Cholesky factors of every `n x n` Hermitian positive-definite matrix of a batch.
///
/// `input` is `[n, n, b_1, ..., b_B]`; `l` receives `n * n` elements per item.
///
/// This is the `Vec` wrapper of [`cholesky_into`].
///
/// # Errors
///
/// [`Error::InvalidArgument`] when `input` is not a batch of square matrices, and
/// [`Error::NonConvergence`] for the lowest-indexed item that is not numerically positive definite
/// (the classification the pre-extraction code reported). `l` is empty on error.
///
/// # Examples
///
/// ```
/// use strided_view::RawStridedRef;
/// use tcontract_exec::Exec;
/// use tlinalg::{cholesky::cholesky, Op};
///
/// let a = [4.0_f64, 2.0, 2.0, 3.0];
/// let mut l = Vec::new();
/// cholesky(
///     Op::Cholesky, RawStridedRef::new(&a, &[2, 2], &[1, 2], 0).unwrap(), &mut l,
///     &mut Exec::serial(),
/// ).unwrap();
/// assert!((l[0] - 2.0).abs() < 1e-12 && l[2] == 0.0);
/// ```
pub fn cholesky<T: FaerScalar>(
    op: Op,
    input: RawStridedRef<'_, T>,
    l: &mut Vec<T>,
    exec: &mut Exec<'_>,
) -> Result<()> {
    l.clear();
    let input = BatchedRef::square(op, "input", input)?;
    let n = input.rows();
    let l_len = checked_product(op, "L", &[n, n])?;
    // The published order: operand validation before any workspace query or allocation.
    check_shape(op, n)?;
    let batch = input.batch();
    let bytes = cholesky_bytes::<T>(op, n, batch, exec)?;
    let mut workspace = vec![MaybeUninit::<u8>::uninit(); bytes];
    let mut l_out = VecOut::new(l, checked_product(op, "L", &[l_len, batch])?);
    cholesky_into(op, input.raw(), l_out.spare(), &mut workspace, exec)?;
    l_out.publish();
    Ok(())
}
