//! faer-backed batched full-pivot LU factorization and solve.
//!
//! Moved from `tenferro-linalg`'s faer backend (same project, MIT OR Apache-2.0).
//!
//! The factorization takes `m x n` matrices: with `k = min(m, n)` it returns the `m x m` row
//! permutation `P`, the unit-lower `m x k` factor `L`, the upper `k x n` factor `U`, the `n x n`
//! column permutation `Q` and the parity, with `P A Qᵀ = L U`. The solve stays square-only.
//!
//! [`full_piv_lu_truncated`]/[`full_piv_lu_truncated_into`] stop early under
//! [`FullPivLuTruncation`] (`max_rank`, `abs_tol`, `rel_tol`) and report the number of accepted
//! pivots per item; the permutations stay full. The truncating kernel is faer's
//! (`faer::linalg::lu::full_pivoting::factor::lu_in_place` with the truncation fields of
//! `FullPivLuParams`, `colmat` 0.34.0).
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
use faer::linalg::lu::full_pivoting::factor::FullPivLuParams;
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

/// The rank-truncation options of [`full_piv_lu_truncated`] and [`full_piv_lu_truncated_into`].
///
/// All three are `None` by default, and then the factorization is the untruncated one of
/// [`full_piv_lu`] (bitwise the same factors) with `rank = min(m, n)` reported for every item.
/// With `k = min(m, n)`:
///
/// - `max_rank`: at most this many pivots are accepted. It also sets the output **capacity**
///   `c = min(k, max_rank)`: `L` is `m x c` and `U` is `c x n` per item.
/// - `abs_tol`, `rel_tol`: the pivot of step `j` is accepted while
///   `|pivot_j| > max(abs_tol, rel_tol * |pivot_0|)`, `|.|` being the modulus and an absent
///   tolerance counting as `0`. `rel_tol` is relative to the **first** pivot, the largest entry of
///   the matrix under full pivoting, not to the previous pivot. The first rejected pivot ends the
///   factorization. A tolerance must be non-negative and finite in the scalar's real type (so at
///   most `f32::MAX` for `f32` and `Complex32`). If `rel_tol * |pivot_0|` itself overflows, which
///   needs `rel_tol > 1`, faer ignores that tolerance.
/// - With any option set, an exactly zero (or subnormal) remaining block also ends the
///   factorization and counts as the stop.
/// - An item that holds a NaN or an infinity ignores the tolerances: its rank is `c`.
///
/// The stopping rule is faer's (`colmat` 0.34.0,
/// `faer::linalg::lu::full_pivoting::factor::FullPivLuParams`); this type forwards it unchanged.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct FullPivLuTruncation {
    /// Accept at most this many pivots; also the output capacity `min(m, n, max_rank)`.
    pub max_rank: Option<usize>,
    /// Absolute pivot tolerance: a pivot of modulus at or below it ends the factorization.
    pub abs_tol: Option<f64>,
    /// Pivot tolerance relative to the modulus of the first pivot.
    pub rel_tol: Option<f64>,
}

impl FullPivLuTruncation {
    /// The per-item output capacity `c = min(m, n, max_rank)`: `L` is `m x c` and `U` is `c x n`.
    ///
    /// # Examples
    ///
    /// ```
    /// use tlinalg::full_piv_lu::FullPivLuTruncation;
    /// assert_eq!(FullPivLuTruncation::default().capacity(5, 3), 3);
    /// let two = FullPivLuTruncation { max_rank: Some(2), ..Default::default() };
    /// assert_eq!(two.capacity(5, 3), 2);
    /// ```
    pub fn capacity(&self, m: usize, n: usize) -> usize {
        let k = m.min(n);
        self.max_rank.map_or(k, |r| r.min(k))
    }

    /// Reject a negative or NaN tolerance, and one that is not finite in the real type `R` of the
    /// scalar. faer converts each tolerance to `R` and ignores a non-finite threshold, so an
    /// `abs_tol` of `1e300` for `f32` would otherwise silently disable truncation.
    fn validate<R: faer::traits::RealField>(&self, op: Op) -> Result<()> {
        for (name, tol) in [("abs_tol", self.abs_tol), ("rel_tol", self.rel_tol)] {
            if let Some(tol) = tol {
                let in_real = <R as faer::traits::ComplexField>::from_f64_impl(tol);
                if !(tol >= 0.0 && <R as faer::traits::ComplexField>::is_finite_impl(&in_real)) {
                    return Err(invalid(
                        op,
                        "truncation",
                        format!(
                            "{name} is {tol}, expected a non-negative tolerance that is finite in \
                             the scalar's real type"
                        ),
                    ));
                }
            }
        }
        Ok(())
    }
}

/// The factors and ranks of a batch of rank-truncated full-pivot LU decompositions, as
/// caller-provided buffers.
///
/// Each vector is cleared and then filled with compact column-major items in batch order. With the
/// capacity `c = min(m, n, max_rank)` (see [`FullPivLuTruncation::capacity`]) and an item's rank
/// `r <= c`, `Pᵀ L U Q` is that item's rank-`r` approximation of `A`.
#[derive(Debug)]
pub struct FullPivLuTruncatedFactors<'a, T> {
    /// The full `m x m` row permutation `P` per item, also on an early stop.
    pub p: &'a mut Vec<T>,
    /// The `m x c` factor `L` per item: unit-lower in its first `r` columns, zero in the rest.
    pub l: &'a mut Vec<T>,
    /// The `c x n` factor `U` per item: upper-triangular in its first `r` rows, zero in the rest.
    pub u: &'a mut Vec<T>,
    /// The full `n x n` column permutation `Q` per item, also on an early stop.
    pub q: &'a mut Vec<T>,
    /// The parity of `P` and `Q` together (`+1` or `-1` in the scalar type), one per item.
    pub parity: &'a mut Vec<T>,
    /// The number of accepted pivots `r` per item.
    pub rank: &'a mut Vec<usize>,
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
        self.factor_with(mat, par, Default::default())
            .transposition_count
    }

    /// Factor `mat` into the scratch with faer's parameters `params`; returns faer's info (the
    /// transposition count and the number of accepted pivots).
    fn factor_with(
        &mut self,
        mat: MatRef<'_, E>,
        par: faer::Par<'_>,
        params: faer::Spec<FullPivLuParams, E>,
    ) -> faer::linalg::lu::full_pivoting::factor::FullPivLuInfo {
        self.lu.copy_from(mat);
        faer::linalg::lu::full_pivoting::factor::lu_in_place(
            self.lu.as_mut(),
            &mut *self.row_perm,
            &mut *self.row_perm_inv,
            &mut *self.col_perm,
            &mut *self.col_perm_inv,
            par,
            MemStack::new(&mut *self.mem),
            params,
        )
        .0
    }
}

/// Faer's parameters for `truncation`: the defaults (what the untruncated entry points pass) with
/// the three truncation fields set.
fn truncated_params<E: faer::traits::ComplexField>(
    truncation: &FullPivLuTruncation,
) -> faer::Spec<FullPivLuParams, E> {
    let mut params: faer::Spec<FullPivLuParams, E> = Default::default();
    params.config.max_rank = truncation.max_rank;
    params.config.abs_tol = truncation.abs_tol;
    params.config.rel_tol = truncation.rel_tol;
    params
}

/// Write one item's factors from the factored scratch: `P (m x m)`, the unit-lower `L (m x cap)`,
/// the upper `U (cap x n)`, `Q (n x n)` and the parity. Columns `rank..cap` of `L` and rows
/// `rank..cap` of `U` are written as zeros, so the trailing block faer leaves behind on an early
/// stop (the Schur complement) never reaches an output; `rank == cap == min(m, n)` is the
/// untruncated factorization.
#[allow(clippy::too_many_arguments)]
fn push_factors<T: FaerScalar>(
    (p, l, u, q, parity): (
        &mut impl Push<T>,
        &mut impl Push<T>,
        &mut impl Push<T>,
        &mut impl Push<T>,
        &mut impl Push<T>,
    ),
    scratch: &FullPivScratch<'_, T::Entity>,
    m: usize,
    n: usize,
    cap: usize,
    rank: usize,
    transpositions: usize,
) {
    debug_assert!(rank <= cap && cap <= m.min(n), "rank={rank} cap={cap}");
    push_permutation(p, scratch.row_perm_inv);
    let lu = scratch.lu.as_ref();
    for col in 0..cap {
        for row in 0..m {
            l.push(if col >= rank {
                T::default()
            } else if row == col {
                T::parity(false)
            } else if row > col {
                T::from_entity(lu[(row, col)])
            } else {
                T::default()
            });
        }
    }
    push_masked(u, lu, cap, n, |row, col| row < rank && row <= col);
    push_permutation(q, scratch.col_perm_inv);
    parity.push(T::parity(!transpositions.is_multiple_of(2)));
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
    push_factors::<T>((p, l, u, q, parity), scratch, m, n, k, k, transpositions);
}

/// One item of [`full_piv_lu_truncated_into`]: factor with the truncation, write the factors at
/// capacity `cap` and the accepted rank.
#[allow(clippy::type_complexity)]
fn full_piv_lu_truncated_item<T: FaerScalar>(
    mat: MatRef<'_, T::Entity>,
    (p, l, u, q, parity, rank_out): (
        &mut impl Push<T>,
        &mut impl Push<T>,
        &mut impl Push<T>,
        &mut impl Push<T>,
        &mut impl Push<T>,
        &mut impl Push<usize>,
    ),
    scratch: &mut FullPivScratch<'_, T::Entity>,
    par: faer::Par<'_>,
    truncation: &FullPivLuTruncation,
) {
    let (m, n) = (mat.nrows(), mat.ncols());
    let cap = truncation.capacity(m, n);
    let info = scratch.factor_with(mat, par, truncated_params::<T::Entity>(truncation));
    // faer caps its steps at `max_rank` (clamped to `min(m, n)`), so `info.rank <= cap`; the `min`
    // only keeps the zero-fill above in bounds should that ever not hold.
    let rank = info.rank.min(cap);
    push_factors::<T>(
        (p, l, u, q, parity),
        scratch,
        m,
        n,
        cap,
        rank,
        info.transposition_count,
    );
    rank_out.push(rank);
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

/// The checked per-item output lengths of a truncated `m x n` batch at capacity `cap`:
/// `P (m x m)`, `L (m x cap)`, `U (cap x n)`, `Q (n x n)`.
fn truncated_shapes<T: FaerScalar>(
    op: Op,
    m: usize,
    n: usize,
    cap: usize,
) -> Result<(usize, usize, usize, usize, f64)> {
    check_shape(op, m, n)?;
    let p_len = checked_product(op, "permutation matrix", &[m, m])?;
    let l_len = checked_product(op, "L", &[m, cap])?;
    let u_len = checked_product(op, "U", &[cap, n])?;
    let q_len = checked_product(op, "permutation matrix", &[n, n])?;
    Ok((p_len, l_len, u_len, q_len, work::<T>(m, n)))
}

/// Rank-truncated full-pivot LU of every `m x n` matrix of a batch into caller-owned storage.
///
/// The factorization stops per item as [`FullPivLuTruncation`] describes. With the capacity
/// `c = min(m, n, max_rank)` the six buffers hold `m * m`, `m * c`, `c * n`, `n * n`, `1` and `1`
/// uninitialised elements per batch item: the full `m x m` row permutation `P`, the `m x c` factor
/// `L`, the `c x n` factor `U`, the full `n x n` column permutation `Q`, the parity and the rank
/// `r`. An item that stops at `r < c` has unit-lower `L[:, :r]` and upper `U[:r, :]` with zeros in
/// `L[:, r..]` and `U[r.., :]`, so `Pᵀ L U Q` is its rank-`r` approximation; the permutations stay
/// full. Items of one batch may stop at different ranks, which is why the shapes are fixed by `c`
/// and the rank is a per-item output.
///
/// The workspace is the one [`full_piv_lu_workspace`] reports for the same `m`, `n`, batch and
/// token; truncation does not change it. With [`FullPivLuTruncation::default`] the factors are
/// bitwise those of [`full_piv_lu_into`] and every rank is `min(m, n)`.
///
/// Truncated factors are not a valid input to [`full_piv_lu_solve`], which factors on its own.
///
/// # Errors
///
/// [`Error::InvalidArgument`] when `input` is not a batch of matrices, a tolerance is negative or
/// not finite in the scalar's real type, an output length does not match, or a size overflows, and
/// [`crate::Error::InsufficientWorkspace`] for a short workspace.
// INVARIANT: descriptor, six output buffers, workspace, token are distinct operands.
#[allow(clippy::too_many_arguments)]
pub fn full_piv_lu_truncated_into<T: FaerScalar>(
    op: Op,
    input: RawStridedRef<'_, T>,
    truncation: FullPivLuTruncation,
    p: &mut [MaybeUninit<T>],
    l: &mut [MaybeUninit<T>],
    u: &mut [MaybeUninit<T>],
    q: &mut [MaybeUninit<T>],
    parity: &mut [MaybeUninit<T>],
    rank: &mut [MaybeUninit<usize>],
    workspace: &mut [MaybeUninit<u8>],
    exec: &mut Exec<'_>,
) -> Result<()> {
    let input = BatchedRef::new(op, "input", input)?;
    truncation.validate::<<T as ScalarEntity>::Real>(op)?;
    let (m, n) = (input.rows(), input.cols());
    let cap = truncation.capacity(m, n);
    let (p_len, l_len, u_len, q_len, item_muladds) = truncated_shapes::<T>(op, m, n, cap)?;
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
            fixed_out(rank, 1, "rank"),
        ),
        batch::ScratchArena::new(workspace, per_lane),
        |par, stack| FullPivScratch::<T::Entity>::new(m, n, par, MemStack::new(stack)).0,
        |index, (p, l, u, q, parity, rank), scratch, par| {
            full_piv_lu_truncated_item::<T>(
                input.item(index),
                (p, l, u, q, parity, rank),
                scratch,
                par,
                &truncation,
            );
            Ok(())
        },
    )
}

/// Rank-truncated full-pivot LU of every `m x n` matrix of a batch, `P A Qᵀ ≈ L U` at rank `r`.
///
/// `input` is `[m, n, b...]`. This is the `Vec` wrapper of [`full_piv_lu_truncated_into`], which
/// describes the stopping rule, the output shapes and the per-item ranks.
///
/// # Errors
///
/// [`Error::InvalidArgument`] when `input` has rank below 2, a tolerance is negative or not
/// finite in the scalar's real type, or an output size overflows. The outputs are empty on error.
///
/// # Examples
///
/// A rank-one `3 x 2` matrix stops after one pivot; `L` and `U` keep the capacity `min(m, n) = 2`
/// with a zero second column and row, and the permutations stay `3 x 3` and `2 x 2`.
///
/// ```
/// use strided_view::RawStridedRef;
/// use tcontract_exec::Exec;
/// use tlinalg::full_piv_lu::{
///     full_piv_lu_truncated, FullPivLuTruncatedFactors, FullPivLuTruncation,
/// };
/// use tlinalg::Op;
///
/// let a = [1.0_f64, 2.0, 4.0, 2.0, 4.0, 8.0];
/// let (mut p, mut l, mut u, mut q, mut parity, mut rank) =
///     (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new());
/// full_piv_lu_truncated(
///     Op::FullPivLu,
///     RawStridedRef::new(&a, &[3, 2], &[1, 3], 0).unwrap(),
///     FullPivLuTruncation { rel_tol: Some(1e-12), ..Default::default() },
///     FullPivLuTruncatedFactors {
///         p: &mut p, l: &mut l, u: &mut u, q: &mut q, parity: &mut parity, rank: &mut rank,
///     },
///     &mut Exec::serial(),
/// ).unwrap();
/// assert_eq!(rank, [1]);
/// assert_eq!((p.len(), l.len(), u.len(), q.len()), (9, 6, 4, 4));
/// assert_eq!(u[0], 8.0);
/// assert_eq!((l[5], u[3]), (0.0, 0.0));
/// ```
pub fn full_piv_lu_truncated<T: FaerScalar>(
    op: Op,
    input: RawStridedRef<'_, T>,
    truncation: FullPivLuTruncation,
    factors: FullPivLuTruncatedFactors<'_, T>,
    exec: &mut Exec<'_>,
) -> Result<()> {
    let FullPivLuTruncatedFactors {
        p,
        l,
        u,
        q,
        parity,
        rank,
    } = factors;
    p.clear();
    l.clear();
    u.clear();
    q.clear();
    parity.clear();
    rank.clear();
    // The published order: operand validation before any workspace query or allocation.
    let input = BatchedRef::new(op, "input", input)?;
    truncation.validate::<<T as ScalarEntity>::Real>(op)?;
    let (m, n) = (input.rows(), input.cols());
    let cap = truncation.capacity(m, n);
    let (p_len, l_len, u_len, q_len, _) = truncated_shapes::<T>(op, m, n, cap)?;
    let batch = input.batch();
    // Every output total is checked before the first reservation (see `full_piv_lu`).
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
    let mut rank_out = VecOut::new(rank, batch);
    full_piv_lu_truncated_into(
        op,
        input.raw(),
        truncation,
        p_out.spare(),
        l_out.spare(),
        u_out.spare(),
        q_out.spare(),
        parity_out.spare(),
        rank_out.spare(),
        &mut workspace,
        exec,
    )?;
    p_out.publish();
    l_out.publish();
    u_out.publish();
    q_out.publish();
    parity_out.publish();
    rank_out.publish();
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
