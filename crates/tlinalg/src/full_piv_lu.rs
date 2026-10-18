//! faer-backed batched full-pivot LU factorization and solve.
//!
//! Moved from `tenferro-linalg`'s faer backend (same project, MIT OR Apache-2.0).
//!
//! The factorization never fails on an exactly singular input; the solve rejects an effectively
//! singular factor (a pivot at or below `ε · max|pivot|`), exactly as before. Faer's work matrix,
//! permutation vectors and scratch are per-lane scratch carved from the caller's borrowed
//! workspace, reused for every item of the lane.
//!
//! # Borrowed workspace
//!
//! [`full_piv_lu_workspace`]/[`full_piv_lu_solve_workspace`] report the bytes a call needs and
//! [`full_piv_lu_into`]/[`full_piv_lu_solve_into`] run it from caller-owned storage; the `Vec`
//! entry points are the wrappers.

use core::mem::MaybeUninit;

use faer::dyn_stack::{MemStack, StackReq};
use faer::{MatMut, MatRef};
use strided_view::RawStridedRef;

use crate::batch::{self, fixed_out, same_batch, BatchedRef, Push, Sink, VecOut};
use crate::scalar::ScalarEntity;
use crate::util::{checked_product, invalid, push_masked, push_permutation, weighted_muladds};
use crate::workspace::{bytes, faer_req, lane_requirement, total_of, zeros};
use crate::{Error, FaerScalar, Op, Result};
use tcontract_exec::Exec;

/// The explicit factors of a batch of full-pivot LU decompositions, as caller-provided buffers.
///
/// Each vector is cleared and then filled with compact column-major items in batch order.
#[derive(Debug)]
pub struct FullPivLuFactors<'a, T> {
    /// The `n x n` row permutation `P` per item, with `P A Qᵀ = L U` (equivalently `A = Pᵀ L U Q`).
    pub p: &'a mut Vec<T>,
    /// The unit-lower-triangular factor `L` per item.
    pub l: &'a mut Vec<T>,
    /// The upper-triangular factor `U` per item.
    pub u: &'a mut Vec<T>,
    /// The `n x n` column permutation `Q` per item.
    pub q: &'a mut Vec<T>,
    /// The parity of the transposition count (`+1` or `-1` in the scalar type), one per item.
    pub parity: &'a mut Vec<T>,
}

/// Lane scratch for `n x n` full-pivot factorizations.
struct FullPivScratch<'w, E: faer::traits::ComplexField> {
    lu: MatMut<'w, E>,
    row_perm: &'w mut [usize],
    row_perm_inv: &'w mut [usize],
    col_perm: &'w mut [usize],
    col_perm_inv: &'w mut [usize],
    mem: &'w mut [MaybeUninit<u8>],
}

/// faer's own factor scratch for `n x n`.
fn factor_req<E: faer::traits::ComplexField>(n: usize, par: faer::Par) -> StackReq {
    faer::linalg::lu::full_pivoting::factor::lu_in_place_scratch::<usize, E>(
        n,
        n,
        par,
        Default::default(),
    )
}

/// One lane's complete requirement for a factorization: the work matrix, four permutations and
/// faer's scratch.
fn full_piv_req<E: faer::traits::ComplexField>(n: usize, par: faer::Par) -> StackReq {
    StackReq::new::<E>(n * n)
        .and(StackReq::new::<usize>(n))
        .and(StackReq::new::<usize>(n))
        .and(StackReq::new::<usize>(n))
        .and(StackReq::new::<usize>(n))
        .and(faer_req(factor_req::<E>(n, par)))
}

impl<'w, E: faer::traits::ComplexField> FullPivScratch<'w, E> {
    fn new(n: usize, par: faer::Par, stack: &'w mut MemStack) -> (Self, &'w mut MemStack) {
        let zero = <E as faer::traits::ComplexField>::zero_impl();
        let (lu, stack) = zeros::<E>(stack, n * n, &zero);
        let (row_perm, stack) = zeros::<usize>(stack, n, &0);
        let (row_perm_inv, stack) = zeros::<usize>(stack, n, &0);
        let (col_perm, stack) = zeros::<usize>(stack, n, &0);
        let (col_perm_inv, stack) = zeros::<usize>(stack, n, &0);
        let (mem, stack) = bytes(stack, factor_req::<E>(n, par).unaligned_bytes_required());
        (
            Self {
                lu: MatMut::from_column_major_slice_mut(lu, n, n),
                row_perm,
                row_perm_inv,
                col_perm,
                col_perm_inv,
                mem,
            },
            stack,
        )
    }

    /// Factor `mat` into the scratch; returns the transposition count.
    fn factor(&mut self, mat: MatRef<'_, E>, par: faer::Par) -> usize {
        self.lu.copy_from(mat);
        faer::linalg::lu::full_pivoting::factor::lu_in_place(
            self.lu.as_mut(),
            &mut *self.row_perm,
            &mut *self.row_perm_inv,
            &mut *self.col_perm,
            &mut *self.col_perm_inv,
            par,
            MemStack::new(&mut *self.mem),
            Default::default(),
        )
        .0
        .transposition_count
    }
}

fn full_piv_lu_item<T: FaerScalar>(
    mat: MatRef<'_, T::Entity>,
    (p, l, u, q, parity): (
        &mut impl Push<T>,
        &mut impl Push<T>,
        &mut impl Push<T>,
        &mut impl Push<T>,
        &mut impl Push<T>,
    ),
    scratch: &mut FullPivScratch<'_, T::Entity>,
    par: faer::Par,
) {
    let n = mat.nrows();
    let transpositions = scratch.factor(mat, par);
    push_permutation(p, scratch.row_perm_inv);
    let lu = scratch.lu.as_ref();
    for col in 0..n {
        for row in 0..n {
            l.push(if row == col {
                T::parity(false)
            } else if row > col {
                T::from_entity(lu[(row, col)])
            } else {
                T::default()
            });
        }
    }
    push_masked(u, lu, n, n, |row, col| row <= col);
    push_permutation(q, scratch.col_perm_inv);
    parity.push(T::parity(transpositions % 2 != 0));
}

/// The work estimate a full-pivot factorization reports to the batch policy.
fn work<T: ScalarEntity>(n: usize) -> f64 {
    weighted_muladds::<T>((n as f64).powi(3) / 3.0)
}

/// The work estimate of a full-pivot factor-then-solve.
fn solve_work<T: ScalarEntity>(n: usize, nrhs: usize) -> f64 {
    let (nf, krhs) = (n as f64, nrhs as f64);
    weighted_muladds::<T>(nf.powi(3) / 3.0 + nf * nf * krhs)
}

/// Reject shapes whose products the lane requirement multiplies.
fn check_shape(op: Op, n: usize) -> Result<()> {
    checked_product(op, "matrix", &[n, n])?;
    Ok(())
}

/// Reject shapes whose products the solve requirement multiplies.
fn check_solve_shape(op: Op, n: usize, nrhs: usize) -> Result<()> {
    checked_product(op, "matrix", &[n, n])?;
    checked_product(op, "rhs", &[n, nrhs])?;
    Ok(())
}

fn full_piv_bytes<T: FaerScalar>(op: Op, n: usize, batch: usize, exec: Exec<'_>) -> Result<usize> {
    let requirement = lane_requirement(
        op,
        exec,
        batch,
        Some(n),
        work::<T>(n),
        || check_shape(op, n),
        |par| full_piv_req::<T::Entity>(n, par),
    )?;
    match requirement {
        None => Ok(0),
        Some((lanes, per_lane)) => total_of(op, lanes, per_lane),
    }
}

fn full_piv_solve_bytes<T: FaerScalar>(
    op: Op,
    n: usize,
    nrhs: usize,
    batch: usize,
    exec: Exec<'_>,
) -> Result<usize> {
    let requirement = lane_requirement(
        op,
        exec,
        batch,
        Some(batch::auto_extent(n, nrhs)),
        solve_work::<T>(n, nrhs),
        || check_solve_shape(op, n, nrhs),
        |par| full_piv_solve_req::<T::Entity>(n, nrhs, par),
    )?;
    match requirement {
        None => Ok(0),
        Some((lanes, per_lane)) => total_of(op, lanes, per_lane),
    }
}

/// Bytes one batched full-pivot LU call needs in its borrowed workspace.
///
/// # Errors
///
/// [`Error::InvalidArgument`] when a size overflows (canonical `Op`
/// is `Op::FullPivLu`), and the executor's refusal for a forced strategy this route cannot honor.
pub fn full_piv_lu_workspace<T: FaerScalar>(
    n: usize,
    batch: usize,
    exec: Exec<'_>,
) -> Result<usize> {
    full_piv_bytes::<T>(Op::FullPivLu, n, batch, exec)
}

/// Full-pivot LU of every `n x n` matrix of a batch into caller-owned storage, `P A Qᵀ = L U`.
///
/// The five buffers hold `n * n`, `n * n`, `n * n`, `n * n` and `1` uninitialised elements per batch
/// item. Workspace contract as [`crate::qr::qr_into`].
///
/// # Errors
///
/// [`Error::InvalidArgument`] when `input` is not a batch of square matrices, an output length does
/// not match, or a size overflows, and [`crate::Error::InsufficientWorkspace`] for a short
/// workspace.
// INVARIANT: descriptor, five output buffers, workspace, token are distinct operands.
#[allow(clippy::too_many_arguments)]
pub fn full_piv_lu_into<T: FaerScalar>(
    op: Op,
    input: RawStridedRef<'_, T>,
    p: &mut [MaybeUninit<T>],
    l: &mut [MaybeUninit<T>],
    u: &mut [MaybeUninit<T>],
    q: &mut [MaybeUninit<T>],
    parity: &mut [MaybeUninit<T>],
    workspace: &mut [MaybeUninit<u8>],
    exec: Exec<'_>,
) -> Result<()> {
    let input = BatchedRef::square(op, "input", input)?;
    let n = input.rows();
    let len = checked_product(op, "matrix", &[n, n])?;
    let item_muladds = work::<T>(n);
    let requirement = lane_requirement(
        op,
        exec,
        input.batch(),
        Some(n),
        item_muladds,
        || check_shape(op, n),
        |par| full_piv_req::<T::Entity>(n, par),
    )?;
    let (lanes, per_lane) = requirement.unwrap_or((1, 0));
    batch::check_workspace(op, workspace, per_lane, lanes)?;
    batch::run(
        op,
        input.batch(),
        exec,
        Some(n),
        item_muladds,
        &mut (
            fixed_out(p, len, "P"),
            fixed_out(l, len, "L"),
            fixed_out(u, len, "U"),
            fixed_out(q, len, "Q"),
            fixed_out(parity, 1, "parity"),
        ),
        batch::ScratchArena::new(workspace, per_lane),
        |par, stack| FullPivScratch::<T::Entity>::new(n, par, MemStack::new(stack)).0,
        |index, (p, l, u, q, parity), scratch, par| {
            full_piv_lu_item::<T>(input.item(index), (p, l, u, q, parity), scratch, par);
            Ok(())
        },
    )
}

/// Full-pivot LU of every `n x n` matrix of a batch, `P A Qᵀ = L U`.
///
/// `input` is `[n, n, b...]`.
///
/// This is the `Vec` wrapper of [`full_piv_lu_into`].
///
/// # Errors
///
/// [`Error::InvalidArgument`] when `input` is not a batch of square matrices or an output size
/// overflows. The outputs are empty on error.
///
/// # Examples
///
/// ```
/// use strided_view::RawStridedRef;
/// use tcontract_exec::Exec;
/// use tlinalg::full_piv_lu::{full_piv_lu, FullPivLuFactors};
/// use tlinalg::Op;
///
/// let a = [1.0_f64, 0.0, 0.0, 2.0];
/// let (mut p, mut l, mut u, mut q, mut parity) =
///     (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new());
/// full_piv_lu(
///     Op::FullPivLu, RawStridedRef::new(&a, &[2, 2], &[1, 2], 0).unwrap(),
///     FullPivLuFactors { p: &mut p, l: &mut l, u: &mut u, q: &mut q, parity: &mut parity },
///     Exec::serial(),
/// ).unwrap();
/// assert_eq!(u[0], 2.0);
/// ```
pub fn full_piv_lu<T: FaerScalar>(
    op: Op,
    input: RawStridedRef<'_, T>,
    factors: FullPivLuFactors<'_, T>,
    exec: Exec<'_>,
) -> Result<()> {
    let FullPivLuFactors { p, l, u, q, parity } = factors;
    p.clear();
    l.clear();
    u.clear();
    q.clear();
    parity.clear();
    let input = BatchedRef::square(op, "input", input)?;
    let n = input.rows();
    let len = checked_product(op, "matrix", &[n, n])?;
    // The published order: operand validation before any workspace query or allocation.
    check_shape(op, n)?;
    let batch = input.batch();
    let bytes = full_piv_bytes::<T>(op, n, batch, exec)?;
    let mut workspace = vec![MaybeUninit::<u8>::uninit(); bytes];
    let mut p_out = VecOut::new(p, checked_product(op, "P", &[len, batch])?);
    let mut l_out = VecOut::new(l, checked_product(op, "L", &[len, batch])?);
    let mut u_out = VecOut::new(u, checked_product(op, "U", &[len, batch])?);
    let mut q_out = VecOut::new(q, checked_product(op, "Q", &[len, batch])?);
    let mut parity_out = VecOut::new(parity, batch);
    full_piv_lu_into(
        op,
        input.raw(),
        p_out.spare(),
        l_out.spare(),
        u_out.spare(),
        q_out.spare(),
        parity_out.spare(),
        &mut workspace,
        exec,
    )?;
    p_out.publish();
    l_out.publish();
    u_out.publish();
    q_out.publish();
    parity_out.publish();
    Ok(())
}

/// Lane scratch for `n x nrhs` solves: the factorization storage plus faer's solve scratch.
struct FullPivSolveScratch<'w, E: faer::traits::ComplexField> {
    factor: FullPivScratch<'w, E>,
    solve_mem: &'w mut [MaybeUninit<u8>],
}

/// faer's own in-place solve scratch for `n x nrhs`.
fn solve_req<E: faer::traits::ComplexField>(n: usize, nrhs: usize, par: faer::Par) -> StackReq {
    faer::linalg::lu::full_pivoting::solve::solve_in_place_scratch::<usize, E>(n, nrhs, par)
}

/// One lane's complete requirement for a solve: the factorization plus faer's solve scratch.
fn full_piv_solve_req<E: faer::traits::ComplexField>(
    n: usize,
    nrhs: usize,
    par: faer::Par,
) -> StackReq {
    full_piv_req::<E>(n, par).and(faer_req(solve_req::<E>(n, nrhs, par)))
}

impl<'w, E: faer::traits::ComplexField> FullPivSolveScratch<'w, E> {
    fn new(n: usize, nrhs: usize, par: faer::Par, stack: &'w mut MemStack) -> Self {
        let (factor, stack) = FullPivScratch::<E>::new(n, par, stack);
        let (solve_mem, _) = bytes(
            stack,
            solve_req::<E>(n, nrhs, par).unaligned_bytes_required(),
        );
        Self { factor, solve_mem }
    }
}

/// Solve one system in the item's output region.
///
/// `b` is written into the region (initialising it) only after the singularity check, and the solve
/// runs there in place: one copy of `b`, as the pre-extraction route made into its pooled
/// right-hand side.
fn full_piv_lu_solve_item<T: FaerScalar>(
    op: Op,
    a: MatRef<'_, T::Entity>,
    b: MatRef<'_, T::Entity>,
    transpose_a: bool,
    x: &mut Sink<'_, T>,
    scratch: &mut FullPivSolveScratch<'_, T::Entity>,
    par: faer::Par,
) -> Result<()> {
    let n = a.nrows();
    let factor = &mut scratch.factor;
    factor.factor(a, par);
    let lu = factor.lu.as_ref();
    let max_diagonal = (0..n)
        .map(|i| T::entity_magnitude(lu[(i, i)]))
        .fold(0.0, f64::max);
    for i in 0..n {
        if T::entity_magnitude(lu[(i, i)]) <= <T as ScalarEntity>::EPSILON * max_diagonal {
            return Err(Error::Singular { op });
        }
    }
    let nrhs = b.ncols();
    let region = x.fill(n * nrhs, |index| T::from_entity(b[(index % n, index / n)]));
    let mut work = MatMut::from_column_major_slice_mut(T::entity_slice_mut(region), n, nrhs);
    // SAFETY: the permutation vectors were filled by the factorization above as mutually inverse
    // permutations of `0..n`, which is exactly the invariant `PermRef::new_unchecked` requires.
    let (row_perm, col_perm) = unsafe {
        (
            faer::perm::PermRef::new_unchecked(&*factor.row_perm, &*factor.row_perm_inv, n),
            faer::perm::PermRef::new_unchecked(&*factor.col_perm, &*factor.col_perm_inv, n),
        )
    };
    let stack = MemStack::new(&mut *scratch.solve_mem);
    if transpose_a {
        faer::linalg::lu::full_pivoting::solve::solve_transpose_in_place(
            lu,
            lu,
            row_perm,
            col_perm,
            work.as_mut(),
            par,
            stack,
        );
    } else {
        faer::linalg::lu::full_pivoting::solve::solve_in_place(
            lu,
            lu,
            row_perm,
            col_perm,
            work.as_mut(),
            par,
            stack,
        );
    }
    Ok(())
}

/// Bytes one batched full-pivot LU solve needs in its borrowed workspace.
///
/// # Errors
///
/// [`Error::InvalidArgument`] when a size overflows (canonical `Op`
/// is `Op::FullPivLuSolve`), and the executor's refusal for a forced strategy this route cannot
/// honor.
pub fn full_piv_lu_solve_workspace<T: FaerScalar>(
    n: usize,
    nrhs: usize,
    batch: usize,
    exec: Exec<'_>,
) -> Result<usize> {
    full_piv_solve_bytes::<T>(Op::FullPivLuSolve, n, nrhs, batch, exec)
}

/// Validate the coefficient and right-hand side of a full-pivot solve.
///
/// This is the published validation order, shared by `_into` and the `Vec` wrapper so the wrapper
/// can validate before it sizes a workspace.
fn solve_shape<T: FaerScalar>(
    op: Op,
    a: &BatchedRef<'_, T>,
    b: &BatchedRef<'_, T>,
) -> Result<(usize, usize, usize, f64)> {
    let n = a.rows();
    if b.rows() != n {
        return Err(invalid(
            op,
            "configuration",
            format!("B has {} rows, expected {n}", b.rows()),
        ));
    }
    same_batch(op, "B", a.batch_dims(), b.batch_dims())?;
    let nrhs = b.cols();
    let x_len = checked_product(op, "X", &[n, nrhs])?;
    Ok((n, nrhs, x_len, solve_work::<T>(n, nrhs)))
}

/// Solve every linear system of a batch through a full-pivot LU of `A`, into caller-owned storage.
///
/// `a` is `[n, n, b...]` and `b` is `[n, nrhs, b...]` with the same batch shape. `x` holds
/// `n * nrhs` uninitialised elements per batch item and receives the column-major solution of
/// `A X = B` (or `Aᵀ X = B` when `transpose_a`). Each item's right-hand side is read only after that
/// item's factor passed the singularity check. Workspace contract as [`crate::qr::qr_into`].
///
/// # Errors
///
/// [`Error::InvalidArgument`] when a descriptor has the wrong shape, rank or batch shape, or an
/// output length does not match, [`crate::Error::InsufficientWorkspace`] for a short workspace, and
/// [`Error::Singular`] for the lowest-indexed item with a pivot at or below `ε · max|pivot|` in
/// magnitude.
// INVARIANT: coefficient, right-hand side, transpose flag, output, workspace, token are distinct
// operands of the batched solve.
#[allow(clippy::too_many_arguments)]
pub fn full_piv_lu_solve_into<T: FaerScalar>(
    op: Op,
    a: RawStridedRef<'_, T>,
    b: RawStridedRef<'_, T>,
    transpose_a: bool,
    x: &mut [MaybeUninit<T>],
    workspace: &mut [MaybeUninit<u8>],
    exec: Exec<'_>,
) -> Result<()> {
    let a = BatchedRef::square(op, "A", a)?;
    let b = BatchedRef::new(op, "B", b)?;
    let (n, nrhs, x_len, item_muladds) = solve_shape::<T>(op, &a, &b)?;
    let requirement = lane_requirement(
        op,
        exec,
        a.batch(),
        Some(batch::auto_extent(n, nrhs)),
        item_muladds,
        || check_solve_shape(op, n, nrhs),
        |par| full_piv_solve_req::<T::Entity>(n, nrhs, par),
    )?;
    let (lanes, per_lane) = requirement.unwrap_or((1, 0));
    batch::check_workspace(op, workspace, per_lane, lanes)?;
    batch::run(
        op,
        a.batch(),
        exec,
        Some(batch::auto_extent(n, nrhs)),
        item_muladds,
        &mut (fixed_out(x, x_len, "X"),),
        batch::ScratchArena::new(workspace, per_lane),
        |par, stack| FullPivSolveScratch::<T::Entity>::new(n, nrhs, par, MemStack::new(stack)),
        |index, (x,), scratch, par| {
            full_piv_lu_solve_item::<T>(
                op,
                a.item(index),
                b.item(index),
                transpose_a,
                x,
                scratch,
                par,
            )
        },
    )
}

/// Solve every linear system of a batch through a full-pivot LU of `A`.
///
/// `a` is `[n, n, b...]` and `b` is `[n, nrhs, b...]` with the same batch shape. `x` receives the
/// column-major `n x nrhs` solution of `A X = B` (or `Aᵀ X = B` when `transpose_a`) per item.
///
/// Each item's right-hand side is read only after that item's factor passed the singularity
/// check, as in the pre-extraction code.
///
/// This is the `Vec` wrapper of [`full_piv_lu_solve_into`].
///
/// # Errors
///
/// [`Error::InvalidArgument`] when a descriptor has the wrong shape, rank or batch shape, and
/// [`Error::Singular`] for the lowest-indexed item with a pivot at or below `ε · max|pivot|` in
/// magnitude. `x` is empty on error.
///
/// # Examples
///
/// ```
/// use strided_view::RawStridedRef;
/// use tcontract_exec::Exec;
/// use tlinalg::{full_piv_lu::full_piv_lu_solve, Op};
///
/// let a = [2.0_f64, 0.0, 0.0, 4.0];
/// let b = [2.0_f64, 8.0];
/// let mut x = Vec::new();
/// full_piv_lu_solve(
///     Op::FullPivLuSolve,
///     RawStridedRef::new(&a, &[2, 2], &[1, 2], 0).unwrap(),
///     RawStridedRef::new(&b, &[2, 1], &[1, 2], 0).unwrap(),
///     false, &mut x, Exec::serial(),
/// ).unwrap();
/// assert_eq!(x, [1.0, 2.0]);
/// ```
// INVARIANT: coefficient, right-hand side, transpose flag, output, token are distinct
// operands of the batched solve.
#[allow(clippy::too_many_arguments)]
pub fn full_piv_lu_solve<T: FaerScalar>(
    op: Op,
    a: RawStridedRef<'_, T>,
    b: RawStridedRef<'_, T>,
    transpose_a: bool,
    x: &mut Vec<T>,
    exec: Exec<'_>,
) -> Result<()> {
    x.clear();
    // The published order: operand validation (including B's rank) before any workspace query or
    // allocation, and no unchecked `n * nrhs` in the output length.
    let (n, nrhs, x_len, batch) = {
        let dims = b.dims();
        if dims.len() < 2 {
            return Err(invalid(
                op,
                "configuration",
                format!("B has rank {}, expected at least 2", dims.len()),
            ));
        }
        let a_view = BatchedRef::square(op, "A", a)?;
        let b_view = BatchedRef::new(op, "B", b)?;
        let (n, nrhs, x_len, _) = solve_shape::<T>(op, &a_view, &b_view)?;
        (n, nrhs, x_len, a_view.batch())
    };
    let bytes = full_piv_solve_bytes::<T>(op, n, nrhs, batch, exec)?;
    let mut workspace = vec![MaybeUninit::<u8>::uninit(); bytes];
    let mut x_out = VecOut::new(x, checked_product(op, "X", &[x_len, batch])?);
    full_piv_lu_solve_into(op, a, b, transpose_a, x_out.spare(), &mut workspace, exec)?;
    x_out.publish();
    Ok(())
}
