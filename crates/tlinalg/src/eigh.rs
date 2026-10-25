//! faer-backed batched Hermitian (self-adjoint) eigendecomposition.
//!
//! Moved from `tenferro-linalg`'s faer backend (same project, MIT OR Apache-2.0).
//!
//! The decomposition reads the **lower** triangle and returns non-decreasing eigenvalues. As with
//! [`crate::svd`], [`eigh`] carries the (real) eigenvalues **in the scalar type itself** — complex
//! values with a zero imaginary part for the complex scalars — because that is what the
//! pre-extraction code produced; [`eigh_values`] returns them in the real type, matching the
//! values-only path.
//!
//! # Borrowed workspace
//!
//! [`eigh_workspace`] reports the bytes a call needs and [`eigh_into`]/[`eigh_values_into`] run it
//! from caller-owned storage; [`eigh`]/[`eigh_values`] are the `Vec` wrappers.

use core::mem::MaybeUninit;

use faer::diag::DiagMut;
use faer::dyn_stack::{MemStack, StackReq};
use faer::linalg::evd::ComputeEigenvectors;
use faer::{MatMut, MatRef};
use strided_view::RawStridedRef;

use crate::batch::{self, fixed_out, BatchedRef, VecOut};
use crate::scalar::ScalarEntity;
use crate::util::{checked_product, weighted_muladds};
use crate::workspace::{bytes, faer_req, lane_requirement, total_of, zeros};
use crate::{Error, FaerScalar, Op, Result};
use tcontract_exec::Exec;

/// Lane scratch for `n x n` Hermitian eigendecompositions. The eigenvectors are written straight
/// into the caller's output, so only the eigenvalues and faer's workspace live here.
struct EighScratch<'w, E: faer::traits::ComplexField> {
    values: DiagMut<'w, E>,
    mem: &'w mut [MaybeUninit<u8>],
    /// Whether an earlier item has written `values`, so the next one has to be reset first.
    reused: bool,
}

/// faer's eigensolver scratch (empty for `n == 0`, which is never decomposed).
fn evd_req<E: faer::traits::ComplexField>(
    n: usize,
    vectors: ComputeEigenvectors,
    par: faer::Par,
) -> StackReq {
    if n == 0 {
        StackReq::EMPTY
    } else {
        faer::linalg::evd::self_adjoint_evd_scratch::<E>(n, vectors, par, Default::default())
    }
}

/// One lane's complete requirement: the eigenvalue diagonal and faer's scratch.
fn eigh_req<E: faer::traits::ComplexField>(
    n: usize,
    vectors: ComputeEigenvectors,
    par: faer::Par,
) -> StackReq {
    StackReq::new::<E>(n).and(faer_req(evd_req::<E>(n, vectors, par)))
}

impl<'w, E: faer::traits::ComplexField> EighScratch<'w, E> {
    fn new(
        n: usize,
        vectors: ComputeEigenvectors,
        par: faer::Par,
        stack: &'w mut MemStack,
    ) -> Self {
        let zero = <E as faer::traits::ComplexField>::zero_impl();
        let (values, stack) = zeros::<E>(stack, n, &zero);
        let (mem, _) = bytes(
            stack,
            evd_req::<E>(n, vectors, par).unaligned_bytes_required(),
        );
        Self {
            values: DiagMut::from_slice_mut(values),
            mem,
            reused: false,
        }
    }
}

/// Decompose one matrix into the lane scratch.
fn decompose<E: faer::traits::ComplexField>(
    op: Op,
    mat: MatRef<'_, E>,
    vectors: Option<MatMut<'_, E>>,
    scratch: &mut EighScratch<'_, E>,
    par: faer::Par,
) -> Result<()> {
    // The carved diagonal is already zeroed; only a scratch that an earlier item wrote needs
    // resetting.
    if scratch.reused {
        scratch.values.fill(E::zero_impl());
    }
    scratch.reused = true;
    faer::linalg::evd::self_adjoint_evd(
        mat,
        scratch.values.as_mut(),
        vectors,
        par,
        MemStack::new(&mut *scratch.mem),
        Default::default(),
    )
    .map_err(|_| Error::NonConvergence { op })
}

/// Bytes one batched Hermitian eigendecomposition call needs in its borrowed workspace.
///
/// `vectors` selects the decomposition: `false` sizes the values-only route. The canonical `Op` of
/// the refusal is `Op::Eigh` (`Op::EighValues` for the values-only query).
///
/// # Errors
///
/// [`Error::InvalidArgument`] when a size overflows.
pub fn eigh_workspace<T: FaerScalar>(
    n: usize,
    vectors: bool,
    batch: usize,
    exec: &Exec<'_>,
) -> Result<usize> {
    let compute = if vectors {
        ComputeEigenvectors::Yes
    } else {
        ComputeEigenvectors::No
    };
    eigh_bytes::<T>(Op::Eigh, n, compute, batch, exec)
}

/// [`eigh_workspace`] for the values-only route (the canonical `Op` is `Op::EighValues`).
///
/// # Errors
///
/// As [`eigh_workspace`].
pub fn eigh_values_workspace<T: FaerScalar>(
    n: usize,
    batch: usize,
    exec: &Exec<'_>,
) -> Result<usize> {
    eigh_bytes::<T>(Op::EighValues, n, ComputeEigenvectors::No, batch, exec)
}

/// Reject a shape whose square the lane requirement multiplies.
fn check_shape(op: Op, n: usize) -> Result<()> {
    checked_product(op, "matrix", &[n, n])?;
    Ok(())
}

/// The work estimate a Hermitian eigendecomposition reports to the batch policy.
fn work<T: ScalarEntity>(n: usize) -> f64 {
    weighted_muladds::<T>((n as f64).powi(3))
}

/// The query with the caller's operation, so a wrapper reports its own `Op`.
fn eigh_bytes<T: FaerScalar>(
    op: Op,
    n: usize,
    vectors: ComputeEigenvectors,
    batch: usize,
    exec: &Exec<'_>,
) -> Result<usize> {
    let requirement = lane_requirement(
        op,
        exec,
        batch,
        work::<T>(n),
        || check_shape(op, n),
        |par| eigh_req::<T::Entity>(n, vectors, par),
    )?;
    match requirement {
        None => Ok(0),
        Some((lanes, per_lane)) => total_of(op, lanes, per_lane),
    }
}

/// Eigenvalues of every `n x n` Hermitian matrix of a batch, without the vectors, into
/// caller-owned storage.
///
/// `values` holds exactly `n` uninitialised real elements per batch item. Workspace contract as
/// [`crate::qr::qr_into`].
///
/// # Errors
///
/// [`Error::InvalidArgument`] when `input` is not a batch of square matrices or an output length
/// does not match, [`crate::Error::InsufficientWorkspace`] for a short workspace, and
/// [`Error::NonConvergence`] for the lowest-indexed item faer fails to converge on.
pub fn eigh_values_into<T: FaerScalar>(
    op: Op,
    input: RawStridedRef<'_, T>,
    values: &mut [MaybeUninit<<T as ScalarEntity>::Real>],
    workspace: &mut [MaybeUninit<u8>],
    exec: &mut Exec<'_>,
) -> Result<()> {
    let input = BatchedRef::square(op, "input", input)?;
    let n = input.rows();
    let item_muladds = work::<T>(n);
    let requirement = lane_requirement(
        op,
        exec,
        input.batch(),
        item_muladds,
        || check_shape(op, n),
        |par| eigh_req::<T::Entity>(n, ComputeEigenvectors::No, par),
    )?;
    let (lanes, per_lane) = requirement.unwrap_or((1, 0));
    batch::check_workspace(op, workspace, per_lane, lanes)?;
    batch::run(
        op,
        input.batch(),
        exec,
        item_muladds,
        &mut (fixed_out(values, n, "eigenvalues"),),
        batch::ScratchArena::new(workspace, per_lane),
        |par, stack| {
            EighScratch::<T::Entity>::new(n, ComputeEigenvectors::No, par, MemStack::new(stack))
        },
        |index, (values,), scratch, par| {
            if n == 0 {
                return Ok(());
            }
            decompose(op, input.item(index), None, scratch, par)?;
            let diagonal = scratch.values.as_ref();
            for i in 0..n {
                values.push(T::real_from_entity(diagonal[i]));
            }
            Ok(())
        },
    )
}

/// Eigenvalues of every `n x n` Hermitian matrix of a batch, without the vectors.
///
/// `input` is `[n, n, b...]`; `values` receives `n` non-decreasing real eigenvalues per item.
///
/// This is the `Vec` wrapper of [`eigh_values_into`].
///
/// # Errors
///
/// [`Error::InvalidArgument`] when `input` is not a batch of square matrices, and
/// [`Error::NonConvergence`] for the lowest-indexed item faer fails to converge on. `values` is
/// empty on error.
///
/// # Examples
///
/// ```
/// use strided_view::RawStridedRef;
/// use tcontract_exec::Exec;
/// use tlinalg::{eigh::eigh_values, Op};
///
/// let a = [2.0_f64, 0.0, 0.0, 1.0];
/// let mut w = Vec::new();
/// eigh_values(
///     Op::EighValues, RawStridedRef::new(&a, &[2, 2], &[1, 2], 0).unwrap(), &mut w,
///     &mut Exec::serial(),
/// ).unwrap();
/// assert_eq!(w, [1.0, 2.0]);
/// ```
pub fn eigh_values<T: FaerScalar>(
    op: Op,
    input: RawStridedRef<'_, T>,
    values: &mut Vec<<T as ScalarEntity>::Real>,
    exec: &mut Exec<'_>,
) -> Result<()> {
    values.clear();
    let input = BatchedRef::square(op, "input", input)?;
    let n = input.rows();
    // The published order: operand validation before any workspace query or allocation.
    check_shape(op, n)?;
    let batch = input.batch();
    let bytes = eigh_bytes::<T>(op, n, ComputeEigenvectors::No, batch, exec)?;
    let mut workspace = vec![MaybeUninit::<u8>::uninit(); bytes];
    let mut values_out = VecOut::new(values, checked_product(op, "eigenvalues", &[n, batch])?);
    eigh_values_into(op, input.raw(), values_out.spare(), &mut workspace, exec)?;
    values_out.publish();
    Ok(())
}

/// Eigendecomposition of every `n x n` Hermitian matrix of a batch into caller-owned storage,
/// `A = V diag(w) Vᴴ`.
///
/// `values` holds `n` and `vectors` `n * n` uninitialised elements per batch item. Workspace
/// contract as [`crate::qr::qr_into`].
///
/// # Errors
///
/// As [`eigh_values_into`], plus [`Error::InvalidArgument`] when the vector count overflows.
pub fn eigh_into<T: FaerScalar>(
    op: Op,
    input: RawStridedRef<'_, T>,
    values: &mut [MaybeUninit<T>],
    vectors: &mut [MaybeUninit<T>],
    workspace: &mut [MaybeUninit<u8>],
    exec: &mut Exec<'_>,
) -> Result<()> {
    let input = BatchedRef::square(op, "input", input)?;
    let n = input.rows();
    let v_len = checked_product(op, "eigenvector matrix", &[n, n])?;
    let item_muladds = work::<T>(n);
    let requirement = lane_requirement(
        op,
        exec,
        input.batch(),
        item_muladds,
        || check_shape(op, n),
        |par| eigh_req::<T::Entity>(n, ComputeEigenvectors::Yes, par),
    )?;
    let (lanes, per_lane) = requirement.unwrap_or((1, 0));
    batch::check_workspace(op, workspace, per_lane, lanes)?;
    batch::run(
        op,
        input.batch(),
        exec,
        item_muladds,
        &mut (
            fixed_out(values, n, "eigenvalues"),
            fixed_out(vectors, v_len, "eigenvectors"),
        ),
        batch::ScratchArena::new(workspace, per_lane),
        |par, stack| {
            EighScratch::<T::Entity>::new(n, ComputeEigenvectors::Yes, par, MemStack::new(stack))
        },
        |index, (values, vectors), scratch, par| {
            if n == 0 {
                return Ok(());
            }
            // The output region is zeroed before faer sees it, as the lane scratch was.
            let region = vectors.fill(v_len, |_| T::default());
            let region = T::entity_slice_mut(region);
            let target = MatMut::from_column_major_slice_mut(region, n, n);
            decompose(op, input.item(index), Some(target), scratch, par)?;
            let diagonal = scratch.values.as_ref();
            for i in 0..n {
                values.push(T::from_real(T::real_from_entity(diagonal[i])));
            }
            Ok(())
        },
    )
}

/// Eigendecomposition of every `n x n` Hermitian matrix of a batch, `A = V diag(w) Vᴴ`.
///
/// `input` is `[n, n, b...]`. Per item, `values` receives the `n` non-decreasing eigenvalues in the
/// scalar type (zero imaginary part for the complex scalars) and `vectors` the column-major `n x n`
/// eigenvectors.
///
/// This is the `Vec` wrapper of [`eigh_into`].
///
/// # Errors
///
/// [`Error::InvalidArgument`] when `input` is not a batch of square matrices or the vector count
/// overflows, and [`Error::NonConvergence`] for the lowest-indexed item faer fails to converge on.
/// The outputs are empty on error.
///
/// # Examples
///
/// ```
/// use strided_view::RawStridedRef;
/// use tcontract_exec::Exec;
/// use tlinalg::{eigh::eigh, Op};
///
/// let a = [2.0_f64, 0.0, 0.0, 1.0];
/// let (mut w, mut v) = (Vec::new(), Vec::new());
/// eigh(
///     Op::Eigh, RawStridedRef::new(&a, &[2, 2], &[1, 2], 0).unwrap(), &mut w, &mut v,
///     &mut Exec::serial(),
/// ).unwrap();
/// assert_eq!(w, [1.0, 2.0]);
/// assert_eq!(v.len(), 4);
/// ```
pub fn eigh<T: FaerScalar>(
    op: Op,
    input: RawStridedRef<'_, T>,
    values: &mut Vec<T>,
    vectors: &mut Vec<T>,
    exec: &mut Exec<'_>,
) -> Result<()> {
    values.clear();
    vectors.clear();
    let input = BatchedRef::square(op, "input", input)?;
    let n = input.rows();
    let v_len = checked_product(op, "eigenvector matrix", &[n, n])?;
    // The published order: operand validation before any workspace query or allocation.
    check_shape(op, n)?;
    let batch = input.batch();
    let bytes = eigh_bytes::<T>(op, n, ComputeEigenvectors::Yes, batch, exec)?;
    let mut workspace = vec![MaybeUninit::<u8>::uninit(); bytes];
    let mut values_out = VecOut::new(values, checked_product(op, "eigenvalues", &[n, batch])?);
    let mut vectors_out = VecOut::new(
        vectors,
        checked_product(op, "eigenvectors", &[v_len, batch])?,
    );
    eigh_into(
        op,
        input.raw(),
        values_out.spare(),
        vectors_out.spare(),
        &mut workspace,
        exec,
    )?;
    values_out.publish();
    vectors_out.publish();
    Ok(())
}
