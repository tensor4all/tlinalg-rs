//! faer-backed batched full-pivot LU factorization and solve.
//!
//! Moved from `tenferro-linalg`'s faer backend (same project, MIT OR Apache-2.0).
//!
//! The factorization takes `m x n` matrices: with `k = min(m, n)` it returns the `m x m` row
//! permutation `P`, the unit-lower `m x k` factor `L`, the upper `k x n` factor `U`, the `n x n`
//! column permutation `Q` and the parity, with `P A Qᵀ = L U`. The solve stays square-only.
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
    /// The `m x m` row permutation `P` per item, with `P A Qᵀ = L U` (equivalently `A = Pᵀ L U Q`).
    pub p: &'a mut Vec<T>,
    /// The unit-lower-triangular `m x min(m, n)` factor `L` per item.
    pub l: &'a mut Vec<T>,
    /// The upper-triangular `min(m, n) x n` factor `U` per item.
    pub u: &'a mut Vec<T>,
    /// The `n x n` column permutation `Q` per item.
    pub q: &'a mut Vec<T>,
    /// The parity of the transposition count (`+1` or `-1` in the scalar type), one per item.
    pub parity: &'a mut Vec<T>,
}

/// Lane scratch for `m x n` full-pivot factorizations.
struct FullPivScratch<'w, E: faer::traits::ComplexField> {
    lu: MatMut<'w, E>,
    row_perm: &'w mut [usize],
    row_perm_inv: &'w mut [usize],
    col_perm: &'w mut [usize],
    col_perm_inv: &'w mut [usize],
    mem: &'w mut [MaybeUninit<u8>],
}

/// faer's own factor scratch for `m x n`.
fn factor_req<E: faer::traits::ComplexField>(m: usize, n: usize, par: faer::ParShape) -> StackReq {
    faer::linalg::lu::full_pivoting::factor::lu_in_place_scratch::<usize, E>(
        m,
        n,
        par,
        Default::default(),
    )
}

/// One lane's complete requirement for a factorization: the work matrix, four permutations and
/// faer's scratch. The row permutations are `m` long and the column permutations `n`.
fn full_piv_req<E: faer::traits::ComplexField>(
    m: usize,
    n: usize,
    par: faer::ParShape,
) -> StackReq {
    StackReq::new::<E>(m * n)
        .and(StackReq::new::<usize>(m))
        .and(StackReq::new::<usize>(m))
        .and(StackReq::new::<usize>(n))
        .and(StackReq::new::<usize>(n))
        .and(faer_req(factor_req::<E>(m, n, par)))
}

impl<'w, E: faer::traits::ComplexField> FullPivScratch<'w, E> {
    fn new(
        m: usize,
        n: usize,
        par: faer::ParShape,
        stack: &'w mut MemStack,
    ) -> (Self, &'w mut MemStack) {
        let zero = <E as faer::traits::ComplexField>::zero_impl();
        let (lu, stack) = zeros::<E>(stack, m * n, &zero);
        let (row_perm, stack) = zeros::<usize>(stack, m, &0);
        let (row_perm_inv, stack) = zeros::<usize>(stack, m, &0);
        let (col_perm, stack) = zeros::<usize>(stack, n, &0);
        let (col_perm_inv, stack) = zeros::<usize>(stack, n, &0);
        let (mem, stack) = bytes(stack, factor_req::<E>(m, n, par).unaligned_bytes_required());
        (
            Self {
                lu: MatMut::from_column_major_slice_mut(lu, m, n),
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
    fn factor(&mut self, mat: MatRef<'_, E>, par: faer::Par<'_>) -> usize {
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
    par: faer::Par<'_>,
) {
    let (m, n) = (mat.nrows(), mat.ncols());
    let k = m.min(n);
    let transpositions = scratch.factor(mat, par);
    push_permutation(p, scratch.row_perm_inv);
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
    push_permutation(q, scratch.col_perm_inv);
    parity.push(T::parity(transpositions % 2 != 0));
}

/// The work estimate a full-pivot factorization reports to the batch policy.
fn work<T: ScalarEntity>(m: usize, n: usize) -> f64 {
    let k = m.min(n);
    let (mf, nf, kf) = (m as f64, n as f64, k as f64);
    weighted_muladds::<T>(mf * nf * kf - (mf + nf) * kf * kf / 2.0 + kf.powi(3) / 3.0)
}

/// The work estimate of a full-pivot factor-then-solve.
fn solve_work<T: ScalarEntity>(n: usize, nrhs: usize) -> f64 {
    let (nf, krhs) = (n as f64, nrhs as f64);
    weighted_muladds::<T>(nf.powi(3) / 3.0 + nf * nf * krhs)
}

/// Cost of one full pivot LU factorization item of shape `m x n`, in weighted multiply adds.
///
/// One unit is one real scalar multiply add; a complex scalar counts four.
/// This returns the planner's existing estimate. Existing formulas are evaluated
/// in `f64`; integers beyond `2^53` cannot all be represented exactly in `f64`.
/// The query is O(1), allocates nothing, and is independent of the machine,
/// batch count, thread width and execution strategy. It selects no strategy.
/// Convert to seconds as `work / rate` with the rate in weighted multiply adds per
/// second. [`crate::estimate::Rates`] measures its rates per nanosecond
/// instead, so `work / rate` with one of those is nanoseconds.
///
/// # Examples
///
/// ```
/// assert_eq!(tlinalg::full_piv_lu::estimate_work::<f64>(6, 6), 72.0);
/// ```
pub fn estimate_work<T: FaerScalar>(m: usize, n: usize) -> f64 {
    work::<T>(m, n)
}

/// The work a team inside one full pivot LU factorization item can pay back.
///
/// This is [`estimate_work`] times the family's share of one, with the same
/// units, independence and allocation guarantees.
///
/// # Examples
///
/// ```
/// use tlinalg::full_piv_lu::{estimate_payback_work, estimate_work};
/// assert_eq!(estimate_payback_work::<f64>(6, 6), estimate_work::<f64>(6, 6));
/// ```
pub fn estimate_payback_work<T: FaerScalar>(m: usize, n: usize) -> f64 {
    estimate_work::<T>(m, n)
}

/// Cost of one full pivot LU factor then solve item with `nrhs` right hand sides.
///
/// Includes the factorization. Units, `f64` rounding, rate ownership and query
/// guarantees are those of [`estimate_work`]. This calls the existing solve model.
///
/// # Examples
///
/// ```
/// assert_eq!(tlinalg::full_piv_lu::estimate_solve_work::<f64>(6, 2), 144.0);
/// ```
pub fn estimate_solve_work<T: FaerScalar>(n: usize, nrhs: usize) -> f64 {
    solve_work::<T>(n, nrhs)
}

/// The work a team inside one full pivot LU factor then solve item can pay back.
///
/// This is [`estimate_solve_work`] times the family's share of one.
///
/// # Examples
///
/// ```
/// use tlinalg::full_piv_lu::{estimate_solve_payback_work, estimate_solve_work};
/// assert_eq!(estimate_solve_payback_work::<f64>(6, 2), estimate_solve_work::<f64>(6, 2));
/// ```
pub fn estimate_solve_payback_work<T: FaerScalar>(n: usize, nrhs: usize) -> f64 {
    estimate_solve_work::<T>(n, nrhs)
}

/// Reject shapes whose products the lane requirement multiplies.
fn check_shape(op: Op, m: usize, n: usize) -> Result<()> {
    checked_product(op, "matrix", &[m, n])?;
    checked_product(op, "workspace", &[m, m])?;
    Ok(())
}

/// The checked per-item output lengths of an `m x n` batch: `P (m x m)`, `L (m x k)`,
/// `U (k x n)`, `Q (n x n)`, `k = min(m, n)`.
fn full_piv_lu_shapes<T: FaerScalar>(
    op: Op,
    m: usize,
    n: usize,
) -> Result<(usize, usize, usize, usize, f64)> {
    check_shape(op, m, n)?;
    let k = m.min(n);
    let p_len = checked_product(op, "permutation matrix", &[m, m])?;
    let l_len = checked_product(op, "L", &[m, k])?;
    let u_len = checked_product(op, "U", &[k, n])?;
    let q_len = checked_product(op, "permutation matrix", &[n, n])?;
    Ok((p_len, l_len, u_len, q_len, work::<T>(m, n)))
}

/// Reject shapes whose products the solve requirement multiplies.
fn check_solve_shape(op: Op, n: usize, nrhs: usize) -> Result<()> {
    checked_product(op, "matrix", &[n, n])?;
    checked_product(op, "rhs", &[n, nrhs])?;
    Ok(())
}

fn full_piv_bytes<T: FaerScalar>(
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
        |par| full_piv_req::<T::Entity>(m, n, par),
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
    exec: &Exec<'_>,
) -> Result<usize> {
    let requirement = lane_requirement(
        op,
        exec,
        batch,
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
/// is `Op::FullPivLu`).
pub fn full_piv_lu_workspace<T: FaerScalar>(
    m: usize,
    n: usize,
    batch: usize,
    exec: &Exec<'_>,
) -> Result<usize> {
    full_piv_bytes::<T>(Op::FullPivLu, m, n, batch, exec)
}

/// Full-pivot LU of every `m x n` matrix of a batch into caller-owned storage, `P A Qᵀ = L U`.
///
/// With `k = min(m, n)` the five buffers hold `m * m`, `m * k`, `k * n`, `n * n` and `1`
/// uninitialised elements per batch item: the `m x m` row permutation `P`, the unit-lower
/// `m x k` factor `L`, the upper `k x n` factor `U`, the `n x n` column permutation `Q` and the
/// parity. Workspace contract as [`crate::qr::qr_into`].
///
/// # Errors
///
/// [`Error::InvalidArgument`] when `input` is not a batch of matrices, an output length does not
/// match, or a size overflows, and [`crate::Error::InsufficientWorkspace`] for a short workspace.
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
    exec: &mut Exec<'_>,
) -> Result<()> {
    let input = BatchedRef::new(op, "input", input)?;
    let (m, n) = (input.rows(), input.cols());
    let (p_len, l_len, u_len, q_len, item_muladds) = full_piv_lu_shapes::<T>(op, m, n)?;
    let requirement = lane_requirement(
        op,
        exec,
        input.batch(),
        item_muladds,
        || check_shape(op, m, n),
        |par| full_piv_req::<T::Entity>(m, n, par),
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
            fixed_out(q, q_len, "Q"),
            fixed_out(parity, 1, "parity"),
        ),
        batch::ScratchArena::new(workspace, per_lane),
        |par, stack| FullPivScratch::<T::Entity>::new(m, n, par, MemStack::new(stack)).0,
        |index, (p, l, u, q, parity), scratch, par| {
            full_piv_lu_item::<T>(input.item(index), (p, l, u, q, parity), scratch, par);
            Ok(())
        },
    )
}

/// Full-pivot LU of every `m x n` matrix of a batch, `P A Qᵀ = L U`.
///
/// `input` is `[m, n, b...]`. With `k = min(m, n)` the outputs are the `m x m` row permutation
/// `P`, the unit-lower `m x k` factor `L`, the upper `k x n` factor `U`, the `n x n` column
/// permutation `Q` and the parity, all `Vec` outputs being compact column-major in batch order.
///
/// This is the `Vec` wrapper of [`full_piv_lu_into`].
///
/// # Errors
///
/// [`Error::InvalidArgument`] when `input` has rank below 2 or an output size
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
///     &mut Exec::serial(),
/// ).unwrap();
/// assert_eq!(u[0], 2.0);
/// ```
pub fn full_piv_lu<T: FaerScalar>(
    op: Op,
    input: RawStridedRef<'_, T>,
    factors: FullPivLuFactors<'_, T>,
    exec: &mut Exec<'_>,
) -> Result<()> {
    let FullPivLuFactors { p, l, u, q, parity } = factors;
    p.clear();
    l.clear();
    u.clear();
    q.clear();
    parity.clear();
    // The published order: operand validation before any workspace query or allocation.
    let input = BatchedRef::new(op, "input", input)?;
    let (m, n) = (input.rows(), input.cols());
    let (p_len, l_len, u_len, q_len, _) = full_piv_lu_shapes::<T>(op, m, n)?;
    let batch = input.batch();
    // Every output total is checked before the first reservation: a broadcast batch view can carry
    // a batch count whose product with one factor's item length overflows while an earlier
    // factor's does not, and a reservation would panic where the contract promises an error.
    let p_total = checked_product(op, "P", &[p_len, batch])?;
    let l_total = checked_product(op, "L", &[l_len, batch])?;
    let u_total = checked_product(op, "U", &[u_len, batch])?;
    let q_total = checked_product(op, "Q", &[q_len, batch])?;
    let bytes = full_piv_bytes::<T>(op, m, n, batch, exec)?;
    let mut workspace = vec![MaybeUninit::<u8>::uninit(); bytes];
    let mut p_out = VecOut::new(p, p_total);
    let mut l_out = VecOut::new(l, l_total);
    let mut u_out = VecOut::new(u, u_total);
    let mut q_out = VecOut::new(q, q_total);
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
fn solve_req<E: faer::traits::ComplexField>(
    n: usize,
    nrhs: usize,
    par: faer::ParShape,
) -> StackReq {
    faer::linalg::lu::full_pivoting::solve::solve_in_place_scratch::<usize, E>(n, nrhs, par)
}

/// One lane's complete requirement for a solve: the factorization plus faer's solve scratch.
fn full_piv_solve_req<E: faer::traits::ComplexField>(
    n: usize,
    nrhs: usize,
    par: faer::ParShape,
) -> StackReq {
    full_piv_req::<E>(n, n, par).and(faer_req(solve_req::<E>(n, nrhs, par)))
}

impl<'w, E: faer::traits::ComplexField> FullPivSolveScratch<'w, E> {
    fn new(n: usize, nrhs: usize, par: faer::ParShape, stack: &'w mut MemStack) -> Self {
        let (factor, stack) = FullPivScratch::<E>::new(n, n, par, stack);
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
    mut par: faer::Par<'_>,
) -> Result<()> {
    let n = a.nrows();
    let factor = &mut scratch.factor;
    factor.factor(a, par.reborrow());
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
            par.reborrow(),
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
/// is `Op::FullPivLuSolve`).
pub fn full_piv_lu_solve_workspace<T: FaerScalar>(
    n: usize,
    nrhs: usize,
    batch: usize,
    exec: &Exec<'_>,
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
    exec: &mut Exec<'_>,
) -> Result<()> {
    let a = BatchedRef::square(op, "A", a)?;
    let b = BatchedRef::new(op, "B", b)?;
    let (n, nrhs, x_len, item_muladds) = solve_shape::<T>(op, &a, &b)?;
    let requirement = lane_requirement(
        op,
        exec,
        a.batch(),
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
///     false, &mut x, &mut Exec::serial(),
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
    exec: &mut Exec<'_>,
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
