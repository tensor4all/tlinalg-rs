//! faer-backed batched partial-pivot LU factorization and the LU-based linear solve.
//!
//! Moved from `tenferro-linalg`'s faer backend (same project, MIT OR Apache-2.0). The packed
//! (LAPACK-format) LU family used by prepared solves lives in [`crate::packed_lu`]; this module is
//! the explicit-factor `P, L, U` decomposition and the one-shot solve.
//!
//! Faer's work matrix, permutation vectors and scratch are per-lane scratch carved from the
//! caller's borrowed workspace, reused for every item of the lane.
//!
//! # Borrowed workspace
//!
//! [`lu_workspace`]/[`lu_solve_workspace`] report the bytes a call needs and
//! [`lu_into`]/[`solve_into`] run it from caller-owned storage; [`lu`]/[`solve`] are the wrappers.

use core::mem::MaybeUninit;

use faer::dyn_stack::{MemStack, StackReq};
use faer::{MatMut, MatRef};
use strided_view::{RawStridedMut, RawStridedRef};

use crate::batch::{self, fixed_out, same_batch, BatchedMut, BatchedRef, VecOut};
use crate::scalar::ScalarEntity;
use crate::util::{checked_product, invalid, push_masked, push_permutation, weighted_muladds};
use crate::workspace::{bytes, faer_req, lane_requirement, total_of, zeros};
use crate::{Error, FaerScalar, Op, Result};
use tcontract_exec::Exec;

/// The explicit factors of a batch of partial-pivot LU decompositions, as caller-provided buffers.
///
/// Each vector is cleared and then filled with compact column-major items in batch order.
#[derive(Debug)]
pub struct LuFactors<'a, T> {
    /// The `m x m` row permutation `P` per item, with `P A = L U` (equivalently `A = Pᵀ L U`).
    pub p: &'a mut Vec<T>,
    /// The unit-lower-trapezoidal `m x min(m, n)` factor `L` per item.
    pub l: &'a mut Vec<T>,
    /// The upper-trapezoidal `min(m, n) x n` factor `U` per item.
    pub u: &'a mut Vec<T>,
    /// The permutation parity (`+1` or `-1` in the scalar type), one per item.
    pub parity: &'a mut Vec<T>,
}

/// Lane scratch for `m x n` partial-pivot factorizations.
pub(crate) struct LuScratch<'w, E: faer::traits::ComplexField> {
    lu: MatMut<'w, E>,
    perm: &'w mut [usize],
    perm_inv: &'w mut [usize],
    mem: &'w mut [MaybeUninit<u8>],
}

/// faer's own factor scratch for `m x n`.
fn factor_req<E: faer::traits::ComplexField>(m: usize, n: usize, par: faer::Par) -> StackReq {
    faer::linalg::lu::partial_pivoting::factor::lu_in_place_scratch::<usize, E>(
        m,
        n,
        par,
        Default::default(),
    )
}

/// One lane's complete requirement for a factorization: the work matrix, both permutations and
/// faer's scratch.
fn lu_req<E: faer::traits::ComplexField>(m: usize, n: usize, par: faer::Par) -> StackReq {
    StackReq::new::<E>(m * n)
        .and(StackReq::new::<usize>(m))
        .and(StackReq::new::<usize>(m))
        .and(faer_req(factor_req::<E>(m, n, par)))
}

impl<'w, E: faer::traits::ComplexField> LuScratch<'w, E> {
    fn new(
        m: usize,
        n: usize,
        par: faer::Par,
        stack: &'w mut MemStack,
    ) -> (LuScratch<'w, E>, &'w mut MemStack) {
        let zero = <E as faer::traits::ComplexField>::zero_impl();
        let (lu, stack) = zeros::<E>(stack, m * n, &zero);
        let (perm, stack) = zeros::<usize>(stack, m, &0);
        let (perm_inv, stack) = zeros::<usize>(stack, m, &0);
        let (mem, stack) = bytes(stack, factor_req::<E>(m, n, par).unaligned_bytes_required());
        (
            LuScratch {
                lu: MatMut::from_column_major_slice_mut(lu, m, n),
                perm,
                perm_inv,
                mem,
            },
            stack,
        )
    }

    /// Factor `mat` into the scratch; returns the transposition count.
    fn factor(&mut self, mat: MatRef<'_, E>, par: faer::Par) -> usize {
        self.lu.copy_from(mat);
        faer::linalg::lu::partial_pivoting::factor::lu_in_place(
            self.lu.as_mut(),
            &mut *self.perm,
            &mut *self.perm_inv,
            par,
            MemStack::new(&mut *self.mem),
            Default::default(),
        )
        .0
        .transposition_count
    }
}

type LuSinks<'s, 'a, T> = (
    &'s mut batch::Sink<'a, T>,
    &'s mut batch::Sink<'a, T>,
    &'s mut batch::Sink<'a, T>,
    &'s mut batch::Sink<'a, T>,
);

fn lu_item<T: FaerScalar>(
    mat: MatRef<'_, T::Entity>,
    (p, l, u, parity): LuSinks<'_, '_, T>,
    scratch: &mut LuScratch<'_, T::Entity>,
    par: faer::Par,
) {
    let (m, n) = (mat.nrows(), mat.ncols());
    let k = m.min(n);
    let transpositions = scratch.factor(mat, par);
    push_permutation(p, scratch.perm_inv);
    let lu = scratch.lu.as_ref();
    for col in 0..k {
        for row in 0..m {
            l.push(if row == col {
                T::parity(false)
            } else if row > col {
                T::from_entity(lu[(row, col)])
            } else {
                T::default()
            });
        }
    }
    push_masked(u, lu, k, n, |row, col| row <= col);
    parity.push(T::parity(transpositions % 2 != 0));
}

/// The work estimate a partial-pivot factorization reports to the batch policy.
fn work<T: ScalarEntity>(m: usize, n: usize) -> f64 {
    let k = m.min(n);
    let (mf, nf, kf) = (m as f64, n as f64, k as f64);
    weighted_muladds::<T>(mf * nf * kf - (mf + nf) * kf * kf / 2.0 + kf.powi(3) / 3.0)
}

/// Reject shapes whose products the lane requirement multiplies.
fn check_shape(op: Op, m: usize, n: usize) -> Result<()> {
    checked_product(op, "matrix", &[m, n])?;
    checked_product(op, "workspace", &[m, m])?;
    Ok(())
}

/// The output lengths of an `m x n` batch, with every product checked.
fn lu_shape<T: FaerScalar>(op: Op, m: usize, n: usize) -> Result<(usize, usize, usize, f64)> {
    check_shape(op, m, n)?;
    let k = m.min(n);
    let p_len = checked_product(op, "permutation matrix", &[m, m])?;
    let l_len = checked_product(op, "L", &[m, k])?;
    let u_len = checked_product(op, "U", &[k, n])?;
    Ok((p_len, l_len, u_len, work::<T>(m, n)))
}

fn lu_bytes<T: FaerScalar>(
    op: Op,
    m: usize,
    n: usize,
    batch: usize,
    exec: &Exec<'_>,
) -> Result<usize> {
    let requirement = lane_requirement(
        op,
        exec,
        batch,
        work::<T>(m, n),
        || check_shape(op, m, n),
        |par| lu_req::<T::Entity>(m, n, par),
    )?;
    match requirement {
        None => Ok(0),
        Some((lanes, per_lane)) => total_of(op, lanes, per_lane),
    }
}

/// Bytes one batched partial-pivot LU call needs in its borrowed workspace.
///
/// # Errors
///
/// [`Error::InvalidArgument`] when a size overflows (canonical `Op`
/// is `Op::Lu`).
pub fn lu_workspace<T: FaerScalar>(
    m: usize,
    n: usize,
    batch: usize,
    exec: &Exec<'_>,
) -> Result<usize> {
    lu_bytes::<T>(Op::Lu, m, n, batch, exec)
}

/// Partial-pivot LU of every `m x n` matrix of a batch into caller-owned storage, `P A = L U`.
///
/// The four buffers hold `m * m`, `m * min(m, n)`, `min(m, n) * n` and `1` uninitialised elements
/// per batch item. Workspace contract as [`crate::qr::qr_into`]. An exactly singular input is not
/// an error.
///
/// # Errors
///
/// [`Error::InvalidArgument`] for a malformed descriptor, an output length that does not match, or
/// a size overflow, and [`crate::Error::InsufficientWorkspace`] for a short workspace.
// INVARIANT: descriptor, four output buffers, workspace, token are distinct operands of the
// batched factorization.
#[allow(clippy::too_many_arguments)]
pub fn lu_into<T: FaerScalar>(
    op: Op,
    input: RawStridedRef<'_, T>,
    p: &mut [MaybeUninit<T>],
    l: &mut [MaybeUninit<T>],
    u: &mut [MaybeUninit<T>],
    parity: &mut [MaybeUninit<T>],
    workspace: &mut [MaybeUninit<u8>],
    exec: &mut Exec<'_>,
) -> Result<()> {
    let input = BatchedRef::new(op, "input", input)?;
    let (m, n) = (input.rows(), input.cols());
    let (p_len, l_len, u_len, item_muladds) = lu_shape::<T>(op, m, n)?;
    let requirement = lane_requirement(
        op,
        exec,
        input.batch(),
        item_muladds,
        || check_shape(op, m, n),
        |par| lu_req::<T::Entity>(m, n, par),
    )?;
    let (lanes, per_lane) = requirement.unwrap_or((1, 0));
    batch::check_workspace(op, workspace, per_lane, lanes)?;
    batch::run(
        op,
        input.batch(),
        exec,
        item_muladds,
        &mut (
            fixed_out(p, p_len, "P"),
            fixed_out(l, l_len, "L"),
            fixed_out(u, u_len, "U"),
            fixed_out(parity, 1, "parity"),
        ),
        batch::ScratchArena::new(workspace, per_lane),
        |par, stack| LuScratch::<T::Entity>::new(m, n, par, MemStack::new(stack)).0,
        |index, (p, l, u, parity), scratch, par| {
            lu_item::<T>(input.item(index), (p, l, u, parity), scratch, par);
            Ok(())
        },
    )
}

/// Partial-pivot LU of every `m x n` matrix of a batch, `P A = L U`.
///
/// `input` is `[m, n, b...]`. An exactly singular input is not an error.
///
/// This is the `Vec` wrapper of [`lu_into`].
///
/// # Errors
///
/// [`Error::InvalidArgument`] when `input` has rank below 2 or an output size overflows. The
/// outputs are empty on error.
///
/// # Examples
///
/// ```
/// use strided_view::RawStridedRef;
/// use tcontract_exec::Exec;
/// use tlinalg::lu::{lu, LuFactors};
/// use tlinalg::Op;
///
/// let a = [1.0_f64, 3.0, 2.0, 4.0];
/// let (mut p, mut l, mut u, mut parity) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
/// lu(
///     Op::Lu, RawStridedRef::new(&a, &[2, 2], &[1, 2], 0).unwrap(),
///     LuFactors { p: &mut p, l: &mut l, u: &mut u, parity: &mut parity },
///     &mut Exec::serial(),
/// ).unwrap();
/// assert_eq!(parity, [-1.0]);
/// assert_eq!(u[0], 3.0);
/// ```
pub fn lu<T: FaerScalar>(
    op: Op,
    input: RawStridedRef<'_, T>,
    factors: LuFactors<'_, T>,
    exec: &mut Exec<'_>,
) -> Result<()> {
    let LuFactors { p, l, u, parity } = factors;
    p.clear();
    l.clear();
    u.clear();
    parity.clear();
    // The published order: operand validation before any workspace query or allocation.
    let input = BatchedRef::new(op, "input", input)?;
    let (m, n) = (input.rows(), input.cols());
    let (p_len, l_len, u_len, _) = lu_shape::<T>(op, m, n)?;
    let batch = input.batch();
    let bytes = lu_bytes::<T>(op, m, n, batch, exec)?;
    let mut workspace = vec![MaybeUninit::<u8>::uninit(); bytes];
    let mut p_out = VecOut::new(p, checked_product(op, "P", &[p_len, batch])?);
    let mut l_out = VecOut::new(l, checked_product(op, "L", &[l_len, batch])?);
    let mut u_out = VecOut::new(u, checked_product(op, "U", &[u_len, batch])?);
    let mut parity_out = VecOut::new(parity, batch);
    lu_into(
        op,
        input.raw(),
        p_out.spare(),
        l_out.spare(),
        u_out.spare(),
        parity_out.spare(),
        &mut workspace,
        exec,
    )?;
    p_out.publish();
    l_out.publish();
    u_out.publish();
    parity_out.publish();
    Ok(())
}

/// Lane scratch for `n x nrhs` solves: the factorization storage plus faer's solve scratch.
struct LuSolveScratch<'w, E: faer::traits::ComplexField> {
    factor: LuScratch<'w, E>,
    solve_mem: &'w mut [MaybeUninit<u8>],
}

/// faer's own in-place solve scratch for `n x nrhs`.
fn solve_req<E: faer::traits::ComplexField>(
    n: usize,
    nrhs: usize,
    transpose_a: bool,
    par: faer::Par,
) -> StackReq {
    if transpose_a {
        faer::linalg::lu::partial_pivoting::solve::solve_transpose_in_place_scratch::<usize, E>(
            n, nrhs, par,
        )
    } else {
        faer::linalg::lu::partial_pivoting::solve::solve_in_place_scratch::<usize, E>(n, nrhs, par)
    }
}

/// One lane's complete requirement for a solve: the factorization plus faer's solve scratch.
fn lu_solve_req<E: faer::traits::ComplexField>(
    n: usize,
    nrhs: usize,
    transpose_a: bool,
    par: faer::Par,
) -> StackReq {
    lu_req::<E>(n, n, par).and(faer_req(solve_req::<E>(n, nrhs, transpose_a, par)))
}

impl<'w, E: faer::traits::ComplexField> LuSolveScratch<'w, E> {
    fn new(
        n: usize,
        nrhs: usize,
        transpose_a: bool,
        par: faer::Par,
        stack: &'w mut MemStack,
    ) -> Self {
        let (factor, stack) = LuScratch::<E>::new(n, n, par, stack);
        let (solve_mem, _) = bytes(
            stack,
            solve_req::<E>(n, nrhs, transpose_a, par).unaligned_bytes_required(),
        );
        Self { factor, solve_mem }
    }
}

/// Solve one system into `destination`; written only after the singularity check.
fn solve_item<T: FaerScalar>(
    op: Op,
    a: MatRef<'_, T::Entity>,
    source: Option<MatRef<'_, T::Entity>>,
    mut destination: MatMut<'_, T::Entity>,
    transpose_a: bool,
    scratch: &mut LuSolveScratch<'_, T::Entity>,
    par: faer::Par,
) -> Result<()> {
    let n = a.nrows();
    let lu = &mut scratch.factor;
    lu.factor(a, par);
    for i in 0..n {
        if lu.lu[(i, i)] == <T::Entity as Default>::default() {
            return Err(Error::Singular { op });
        }
    }
    if let Some(source) = source {
        destination.copy_from(source);
    }
    // SAFETY: `perm`/`perm_inv` were filled by the factorization above as mutually inverse
    // permutations of `0..n`, which is exactly the invariant `PermRef::new_unchecked` requires.
    let perm = unsafe { faer::perm::PermRef::new_unchecked(&*lu.perm, &*lu.perm_inv, n) };
    let stack = MemStack::new(&mut *scratch.solve_mem);
    if transpose_a {
        faer::linalg::lu::partial_pivoting::solve::solve_transpose_in_place(
            lu.lu.as_ref(),
            lu.lu.as_ref(),
            perm,
            destination,
            par,
            stack,
        );
    } else {
        faer::linalg::lu::partial_pivoting::solve::solve_in_place(
            lu.lu.as_ref(),
            lu.lu.as_ref(),
            perm,
            destination,
            par,
            stack,
        );
    }
    Ok(())
}

/// Bytes one batched LU-based solve needs in its borrowed workspace.
///
/// # Errors
///
/// [`Error::InvalidArgument`] when a size overflows (canonical `Op`
/// is `Op::Solve`).
pub fn lu_solve_workspace<T: FaerScalar>(
    n: usize,
    nrhs: usize,
    transpose_a: bool,
    batch: usize,
    exec: &Exec<'_>,
) -> Result<usize> {
    lu_solve_bytes::<T>(Op::Solve, n, nrhs, transpose_a, batch, exec)
}

/// The work estimate a factor-then-solve reports to the batch policy.
fn solve_work<T: ScalarEntity>(n: usize, nrhs: usize) -> f64 {
    let (nf, krhs) = (n as f64, nrhs as f64);
    weighted_muladds::<T>(nf.powi(3) / 3.0 + nf * nf * krhs)
}

/// Reject shapes whose products the solve requirement multiplies.
fn check_solve_shape(op: Op, n: usize, nrhs: usize) -> Result<()> {
    checked_product(op, "matrix", &[n, n])?;
    checked_product(op, "rhs", &[n, nrhs])?;
    Ok(())
}

fn lu_solve_bytes<T: FaerScalar>(
    op: Op,
    n: usize,
    nrhs: usize,
    transpose_a: bool,
    batch: usize,
    exec: &Exec<'_>,
) -> Result<usize> {
    let requirement = lane_requirement(
        op,
        exec,
        batch,
        solve_work::<T>(n, nrhs),
        || check_solve_shape(op, n, nrhs),
        |par| lu_solve_req::<T::Entity>(n, nrhs, transpose_a, par),
    )?;
    match requirement {
        None => Ok(0),
        Some((lanes, per_lane)) => total_of(op, lanes, per_lane),
    }
}

/// Validate the coefficient, the destination and the optional right-hand side of a solve.
///
/// This is the published validation order, and `_into` and the `Vec`/destination wrapper share it
/// so a wrapper can run it before it sizes a workspace.
fn solve_shape<'a, T: FaerScalar>(
    op: Op,
    a: &BatchedRef<'_, T>,
    destination: &BatchedMut<'_, T>,
    rhs: Option<RawStridedRef<'a, T>>,
) -> Result<(usize, usize, f64, Option<BatchedRef<'a, T>>)> {
    let n = a.rows();
    let nrhs = destination.cols();
    if destination.rows() != n {
        return Err(invalid(
            op,
            "configuration",
            format!("out has {} rows, expected {n}", destination.rows()),
        ));
    }
    same_batch(op, "out", a.batch_dims(), destination.batch_dims())?;
    let source = match rhs {
        Some(b) => {
            let b = BatchedRef::new(op, "B", b)?;
            if (b.rows(), b.cols()) != (n, nrhs) {
                return Err(invalid(
                    op,
                    "configuration",
                    format!("B is {}x{}, expected {n}x{nrhs}", b.rows(), b.cols()),
                ));
            }
            same_batch(op, "B", a.batch_dims(), b.batch_dims())?;
            Some(b)
        }
        None => None,
    };
    Ok((n, nrhs, solve_work::<T>(n, nrhs), source))
}

/// Solve every linear system `A X = B` (or `Aᵀ X = B`) of a batch into caller-owned storage.
///
/// `a` is `[n, n, b...]` and `out` is `[n, nrhs, b...]` with the same batch shape. When `rhs` is
/// `Some(b)`, `b` (`[n, nrhs, b...]`) is copied into `out` per item first; when it is `None`, `out`
/// already holds the right-hand sides. `out` must not alias itself. Workspace contract as
/// [`crate::qr::qr_into`] (`lu_solve_workspace`).
///
/// # Failure behaviour
///
/// Each item's destination is written only after that item's factorization and singularity check
/// succeed, so a failing item's destination is left exactly as it was. Other items may already have
/// been solved (a multi-lane run solves items of other lanes concurrently).
///
/// # Errors
///
/// [`Error::InvalidArgument`] when a descriptor has the wrong shape, rank or batch shape, or `out`
/// aliases itself, [`crate::Error::InsufficientWorkspace`] for a short workspace, and
/// [`Error::Singular`] for the lowest-indexed item whose LU factor has an exactly zero pivot.
// INVARIANT: coefficient, optional source, destination, transpose flag, workspace, token are
// distinct operands of the batched solve.
#[allow(clippy::too_many_arguments)]
pub fn solve_into<T: FaerScalar>(
    op: Op,
    a: RawStridedRef<'_, T>,
    rhs: Option<RawStridedRef<'_, T>>,
    mut out: RawStridedMut<'_, T>,
    transpose_a: bool,
    workspace: &mut [MaybeUninit<u8>],
    exec: &mut Exec<'_>,
) -> Result<()> {
    let a = BatchedRef::square(op, "A", a)?;
    let destination = BatchedMut::new(op, "out", &mut out)?;
    let (n, nrhs, item_muladds, source) = solve_shape::<T>(op, &a, &destination, rhs)?;
    let requirement = lane_requirement(
        op,
        exec,
        a.batch(),
        item_muladds,
        || check_solve_shape(op, n, nrhs),
        |par| lu_solve_req::<T::Entity>(n, nrhs, transpose_a, par),
    )?;
    let (lanes, per_lane) = requirement.unwrap_or((1, 0));
    batch::check_workspace(op, workspace, per_lane, lanes)?;
    batch::run(
        op,
        a.batch(),
        exec,
        item_muladds,
        &mut (),
        batch::ScratchArena::new(workspace, per_lane),
        |par, stack| {
            LuSolveScratch::<T::Entity>::new(n, nrhs, transpose_a, par, MemStack::new(stack))
        },
        |index, (), scratch, par| {
            // SAFETY: `batch::run` hands each index below the batch count to exactly one lane, and
            // this closure is the only user of `destination`, so item `index` is not covered by any
            // other live reference while it is solved.
            let item = unsafe { destination.item(index) };
            solve_item::<T>(
                op,
                a.item(index),
                source.as_ref().map(|b| b.item(index)),
                item,
                transpose_a,
                scratch,
                par,
            )
        },
    )
}

/// Solve every linear system `A X = B` (or `Aᵀ X = B`) of a batch through a partial-pivot LU of
/// `A`, writing the solutions into a caller-owned destination.
///
/// `a` is `[n, n, b...]` and `out` is `[n, nrhs, b...]` with the same batch shape. When `rhs` is
/// `Some(b)`, `b` (`[n, nrhs, b...]`) is copied into `out` per item first; when it is `None`, `out`
/// already holds the right-hand sides. `out` must not alias itself.
///
/// This is the wrapper of [`solve_into`]: it allocates one scratch block of
/// [`lu_solve_workspace`] bytes per call and otherwise passes everything through.
///
/// # Failure behaviour
///
/// Each item's destination is written only after that item's factorization and singularity check
/// succeed, so a failing item's destination is left exactly as it was. Other items may already have
/// been solved (a multi-lane run solves items of other lanes concurrently).
///
/// # Errors
///
/// [`Error::InvalidArgument`] when a descriptor has the wrong shape, rank or batch shape, or `out`
/// aliases itself, and [`Error::Singular`] for the lowest-indexed item whose LU factor has an
/// exactly zero pivot.
///
/// # Examples
///
/// ```
/// use strided_view::{RawStridedMut, RawStridedRef};
/// use tcontract_exec::Exec;
/// use tlinalg::{lu::solve, Op};
///
/// let a = [2.0_f64, 0.0, 0.0, 4.0];
/// let b = [2.0_f64, 8.0];
/// let mut x = [0.0_f64; 2];
/// solve(
///     Op::Solve,
///     RawStridedRef::new(&a, &[2, 2], &[1, 2], 0).unwrap(),
///     Some(RawStridedRef::new(&b, &[2, 1], &[1, 2], 0).unwrap()),
///     RawStridedMut::new(&mut x, &[2, 1], &[1, 2], 0).unwrap(),
///     false, &mut Exec::serial(),
/// ).unwrap();
/// assert_eq!(x, [1.0, 2.0]);
/// ```
// INVARIANT: coefficient, optional source, destination, transpose flag, token are
// distinct operands of the batched solve.
#[allow(clippy::too_many_arguments)]
pub fn solve<T: FaerScalar>(
    op: Op,
    a: RawStridedRef<'_, T>,
    rhs: Option<RawStridedRef<'_, T>>,
    mut out: RawStridedMut<'_, T>,
    transpose_a: bool,
    exec: &mut Exec<'_>,
) -> Result<()> {
    // The published order: operand validation (including the destination's rank) before any
    // workspace query or allocation.
    let (n, nrhs, batch) = {
        let dims = out.dims();
        if dims.len() < 2 {
            return Err(invalid(
                op,
                "configuration",
                format!("out has rank {}, expected at least 2", dims.len()),
            ));
        }
        let shape = BatchedRef::square(op, "A", a)?;
        let destination = BatchedMut::new(op, "out", &mut out)?;
        let (n, nrhs, _, _) = solve_shape::<T>(op, &shape, &destination, rhs)?;
        (n, nrhs, shape.batch())
    };
    let bytes = lu_solve_bytes::<T>(op, n, nrhs, transpose_a, batch, exec)?;
    let mut workspace = vec![MaybeUninit::<u8>::uninit(); bytes];
    solve_into(op, a, rhs, out, transpose_a, &mut workspace, exec)
}
