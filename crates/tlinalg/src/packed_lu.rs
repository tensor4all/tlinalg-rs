//! Batched packed partial-pivot LU: factor, prepared solve, and fused factor+solve.
//!
//! Moved from `tenferro-linalg`'s faer backend (same project, MIT OR Apache-2.0). The kernels keep
//! the LAPACK packed format — unit-lower `L` below the diagonal, `U` on and above it, one-based
//! row-swap pivots — so the factors stay interchangeable with a LAPACK implementation.
//!
//! # Batches
//!
//! Every entry point operates on a whole batch, with library-owned Auto lanes (see
//! `docs/design/batched-api.md`): the factors and right-hand sides are the host's compact,
//! batch-contiguous buffers, updated in place. The prepared solve reads its factors and pivots
//! through strided descriptors, so a host can broadcast one factorization (stride 0 on a batch axis)
//! against many right-hand sides.
//!
//! Scratch is native and per lane: four permutation vectors and one faer `MemBuffer` region, carved
//! from the caller's borrowed workspace for the `*_into` routes and reused across every matrix of
//! the lane.
//!
//! In-place buffers are not rolled back on error: items before the failing one (and items of other
//! lanes) may already hold their results.
//!
//! # Borrowed workspace
//!
//! [`factor_workspace`] reports the bytes a factor call needs; [`factor_into`] runs it from
//! caller-owned storage and [`factor`] is its wrapper. [`solve_prepared_into`] and
//! [`factor_solve_into`] take the same borrowed workspace; the prepared solve needs none (the
//! triangular solves take no scratch), and it accepts an empty slice.

use core::mem::MaybeUninit;

use faer::dyn_stack::{MemStack, StackReq};
use faer::prelude::ReborrowMut;
use faer::{Conj, MatMut, MatRef};

use strided_view::RawStridedRef;

use crate::batch::{self, in_place, same_batch, BatchAxes, BatchedRef};
use crate::scalar::ScalarEntity;
use crate::util::{checked_product, invalid, weighted_muladds};
use crate::workspace::{bytes, faer_req, lane_requirement, total_of, zeros};
use crate::{Error, FaerScalar, Op, Result};
use tcontract_exec::Exec;

/// Reusable per-lane state for factoring a run of `m x n` matrices.
///
/// Typed by the scalar it was sized for, so mixing a scratch with another scalar is a compile
/// error. The batch driver builds one per lane for the call's shape.
pub(crate) struct FactorScratch<'w, T: FaerScalar> {
    scalar: core::marker::PhantomData<T>,
    m: usize,
    k: usize,
    perm: &'w mut [usize],
    perm_inv: &'w mut [usize],
    current: &'w mut [usize],
    position: &'w mut [usize],
    mem: &'w mut [MaybeUninit<u8>],
}

/// faer's own factor scratch for `m x n`.
fn factor_req<T: FaerScalar>(m: usize, n: usize, par: faer::ParShape) -> StackReq {
    faer::linalg::lu::partial_pivoting::factor::lu_in_place_scratch::<usize, T::Entity>(
        m,
        n,
        par,
        Default::default(),
    )
}

/// One lane's complete requirement for a factorization: four permutations and faer's scratch. The
/// matrix is the caller's in-place buffer, not scratch.
fn factor_lane_req<T: FaerScalar>(m: usize, n: usize, par: faer::ParShape) -> StackReq {
    StackReq::new::<usize>(m)
        .and(StackReq::new::<usize>(m))
        .and(StackReq::new::<usize>(m))
        .and(StackReq::new::<usize>(m))
        .and(faer_req(factor_req::<T>(m, n, par)))
}

impl<'w, T: FaerScalar> FactorScratch<'w, T> {
    /// Size the scratch for `m x n` matrices from one lane's borrowed workspace chunk.
    ///
    /// The parallelism participates because faer sizes its scratch per thread.
    fn new(m: usize, n: usize, par: faer::ParShape, stack: &'w mut MemStack) -> Self {
        let (perm, stack) = zeros::<usize>(stack, m, &0);
        let (perm_inv, stack) = zeros::<usize>(stack, m, &0);
        let (current, stack) = zeros::<usize>(stack, m, &0);
        let (position, stack) = zeros::<usize>(stack, m, &0);
        let (mem, _) = bytes(stack, factor_req::<T>(m, n, par).unaligned_bytes_required());
        Self {
            scalar: core::marker::PhantomData,
            m,
            k: m.min(n),
            perm,
            perm_inv,
            current,
            position,
            mem,
        }
    }

    /// Factor one compact column-major matrix in place and write its one-based swap sequence into
    /// `ipiv`. Returns whether the permutation is odd.
    fn factor(
        &mut self,
        par: faer::Par<'_>,
        matrix: MatMut<'_, T::Entity>,
        ipiv: &mut [i32],
        op: Op,
    ) -> Result<bool> {
        let stack = MemStack::new(&mut *self.mem);
        let info = faer::linalg::lu::partial_pivoting::factor::lu_in_place(
            matrix,
            &mut *self.perm,
            &mut *self.perm_inv,
            par,
            stack,
            Default::default(),
        )
        .0;

        // faer returns `perm` with `(P A)[i, :] = A[perm[i], :]`. Replay it as the LAPACK swap
        // sequence: at step `i`, swap row `i` with the current position of row `perm[i]`.
        // `current`/`position` track the running permutation and its inverse, so each step is O(1).
        for (idx, (slot, pos)) in self
            .current
            .iter_mut()
            .zip(self.position.iter_mut())
            .enumerate()
        {
            *slot = idx;
            *pos = idx;
        }
        // INVARIANT: the driver builds this scratch per lane for the call's `(m, n)`, so `self.k ==
        // m.min(n)`. `perm` is a permutation of `0..m` (faer contract) and the caller passed
        // `ipiv.len() == k <= m`, so every index below is in bounds.
        for (step, slot) in ipiv.iter_mut().enumerate().take(self.k) {
            let wanted = self.perm[step];
            let pivot = self.position[wanted];
            if pivot >= self.m {
                return Err(invalid(op, "configuration", "invalid row permutation"));
            }
            let displaced = self.current[step];
            self.current.swap(step, pivot);
            self.position[wanted] = step;
            self.position[displaced] = pivot;
            *slot = i32::try_from(pivot + 1)
                .map_err(|_| invalid(op, "configuration", "pivot index exceeds i32 range"))?;
        }
        Ok(info.transposition_count % 2 == 1)
    }
}

/// Check that every `(per_matrix_len, buffer_len)` pair describes the same number of matrices.
fn check_batches(op: Op, buffers: [(usize, usize); 3], batch: usize) -> Result<()> {
    for ((per_matrix, len), what) in buffers
        .into_iter()
        .zip(["packed LU", "pivots", "rhs batch"])
    {
        if len != checked_product(op, what, &[per_matrix, batch])? {
            return Err(Error::Inconsistent {
                op,
                detail: "packed LU, pivot, and batch buffers describe different batches",
            });
        }
    }
    Ok(())
}

/// The work estimate a packed partial-pivot factorization reports to the batch policy.
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

/// Reject shapes whose products the fused requirement multiplies.
fn check_solve_shape(op: Op, n: usize, nrhs: usize) -> Result<()> {
    checked_product(op, "matrix", &[n, n])?;
    checked_product(op, "rhs", &[n, nrhs])?;
    Ok(())
}

/// The validated shape of a packed-LU factor call: `(matrix_len, k, batch, work estimate)`.
fn factor_shape<T: FaerScalar>(
    op: Op,
    m: usize,
    n: usize,
    lu_len: usize,
    pivots_len: usize,
    parity_len: usize,
) -> Result<(usize, usize, usize, f64)> {
    check_shape(op, m, n)?;
    let k = m.min(n);
    let matrix_len = checked_product(op, "matrix shape", &[m, n])?;
    let batch = parity_len;
    check_batches(
        op,
        [(matrix_len, lu_len), (k, pivots_len), (1, parity_len)],
        batch,
    )?;
    Ok((matrix_len, k, batch, work::<T>(m, n)))
}

/// The validated shape of a fused packed-LU factor+solve.
fn factor_solve_shape<T: FaerScalar>(
    op: Op,
    n: usize,
    nrhs: usize,
    packed_lu_len: usize,
    pivots_len: usize,
    output_len: usize,
) -> Result<(usize, usize, usize, f64)> {
    check_solve_shape(op, n, nrhs)?;
    let matrix_len = checked_product(op, "matrix", &[n, n])?;
    let rhs_len = checked_product(op, "rhs", &[n, nrhs])?;
    let batch = packed_lu_len / matrix_len.max(1);
    check_batches(
        op,
        [
            (matrix_len, packed_lu_len),
            (n, pivots_len),
            (rhs_len, output_len),
        ],
        batch,
    )?;
    let nf = n as f64;
    let muladds = weighted_muladds::<T>(nf.powi(3) / 3.0 + nf * nf * nrhs as f64);
    Ok((matrix_len, rhs_len, batch, muladds))
}

fn factor_bytes<T: FaerScalar>(
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
        |par| factor_lane_req::<T>(m, n, par),
    )?;
    match requirement {
        None => Ok(0),
        Some((lanes, per_lane)) => total_of(op, lanes, per_lane),
    }
}

fn factor_solve_bytes<T: FaerScalar>(
    op: Op,
    n: usize,
    nrhs: usize,
    batch: usize,
    exec: &Exec<'_>,
) -> Result<usize> {
    let nf = n as f64;
    let muladds = weighted_muladds::<T>(nf.powi(3) / 3.0 + nf * nf * nrhs as f64);
    let requirement = lane_requirement(
        op,
        exec,
        batch,
        muladds,
        || check_solve_shape(op, n, nrhs),
        |par| factor_lane_req::<T>(n, n, par),
    )?;
    match requirement {
        None => Ok(0),
        Some((lanes, per_lane)) => total_of(op, lanes, per_lane),
    }
}

/// Bytes one batched packed-LU factor call needs in its borrowed workspace.
///
/// # Errors
///
/// [`Error::InvalidArgument`] when a size overflows (canonical `Op`
/// is `Op::LuFactor`).
pub fn factor_workspace<T: FaerScalar>(
    m: usize,
    n: usize,
    batch: usize,
    exec: &Exec<'_>,
) -> Result<usize> {
    factor_bytes::<T>(Op::LuFactor, m, n, batch, exec)
}

/// Factor every compact column-major `m x n` matrix of `lu` in place, into caller-owned workspace
/// storage.
///
/// `lu` holds `batch = parity.len()` matrices; `pivots` receives `min(m, n)` one-based pivots per
/// matrix and `parity` one permutation parity per matrix. Exactly singular matrices are **not** an
/// error, matching LAPACK `?getrf` with positive `info`. Workspace contract as
/// [`crate::qr::qr_into`] (`factor_workspace`).
///
/// # Errors
///
/// Returns [`Error::Inconsistent`] when the buffers describe different batches,
/// [`Error::InvalidArgument`] when faer returns an invalid row permutation (lowest-indexed item),
/// and [`crate::Error::InsufficientWorkspace`] for a short workspace (before any write).
// INVARIANT: shape, the three in-place batch buffers, workspace, token are distinct operands.
#[allow(clippy::too_many_arguments)]
pub fn factor_into<T: FaerScalar>(
    op: Op,
    m: usize,
    n: usize,
    lu: &mut [T],
    pivots: &mut [i32],
    parity: &mut [T],
    workspace: &mut [MaybeUninit<u8>],
    exec: &mut Exec<'_>,
) -> Result<()> {
    let (matrix_len, k, batch, item_muladds) =
        factor_shape::<T>(op, m, n, lu.len(), pivots.len(), parity.len())?;
    // The route is resolved even when there is nothing to factor, so the workspace contract is
    // the same for an empty batch (tlinalg-rs#57/#58).
    let requirement = lane_requirement(
        op,
        exec,
        batch,
        item_muladds,
        || check_shape(op, m, n),
        |par| factor_lane_req::<T>(m, n, par),
    )?;
    let (lanes, per_lane) = requirement.unwrap_or((1, 0));
    batch::check_workspace(op, workspace, per_lane, lanes)?;
    if matrix_len == 0 || batch == 0 {
        return Ok(());
    }
    batch::run(
        op,
        batch,
        exec,
        item_muladds,
        &mut (
            in_place(lu, matrix_len, "packed LU"),
            in_place(pivots, k, "pivots"),
            in_place(parity, 1, "parity"),
        ),
        batch::ScratchArena::new(workspace, per_lane),
        |par, stack| FactorScratch::<T>::new(m, n, par, MemStack::new(stack)),
        |_index, (lu, pivots, parity), scratch, par| {
            let mat = MatMut::from_column_major_slice_mut(T::entity_slice_mut(lu), m, n);
            let odd = scratch.factor(par, mat, pivots, op)?;
            parity[0] = T::parity(odd);
            Ok(())
        },
    )
}

/// Factor every compact column-major `m x n` matrix of `lu` in place.
///
/// `lu` holds `batch = parity.len()` matrices; `pivots` receives `min(m, n)` one-based pivots per
/// matrix and `parity` one permutation parity per matrix. Exactly singular matrices are **not** an
/// error, matching LAPACK `?getrf` with positive `info`.
///
/// This is the wrapper of [`factor_into`]: it allocates one scratch block of [`factor_workspace`]
/// bytes per call.
///
/// # Errors
///
/// Returns [`Error::Inconsistent`] when the buffers describe different batches, and
/// [`Error::InvalidArgument`] when faer returns an invalid row permutation (lowest-indexed item).
///
/// # Examples
///
/// ```
/// use tcontract_exec::Exec;
/// use tlinalg::{packed_lu::factor, Op};
///
/// let mut lu = [1.0_f64, 3.0, 2.0, 4.0];
/// let (mut pivots, mut parity) = ([0_i32; 2], [0.0_f64; 1]);
/// factor(Op::LuFactor, 2, 2, &mut lu, &mut pivots, &mut parity, &mut Exec::serial()).unwrap();
/// assert_eq!((pivots, parity), ([2, 2], [-1.0]));
/// ```
// INVARIANT: shape, the three in-place batch buffers, token are distinct operands.
#[allow(clippy::too_many_arguments)]
pub fn factor<T: FaerScalar>(
    op: Op,
    m: usize,
    n: usize,
    lu: &mut [T],
    pivots: &mut [i32],
    parity: &mut [T],
    exec: &mut Exec<'_>,
) -> Result<()> {
    // The published order: shape and buffer validation before any workspace query or allocation.
    let (_, _, batch, _) = factor_shape::<T>(op, m, n, lu.len(), pivots.len(), parity.len())?;
    let bytes = factor_bytes::<T>(op, m, n, batch, exec)?;
    let mut workspace = vec![MaybeUninit::<u8>::uninit(); bytes];
    factor_into(op, m, n, lu, pivots, parity, &mut workspace, exec)
}

/// Apply a one-based LAPACK swap sequence to the rows of a compact column-major `n x nrhs` block,
/// forward (`P b`) or in reverse (`P^T b`).
fn apply_row_swaps<T: Copy>(rhs: &mut [T], n: usize, nrhs: usize, ipiv: &[i32], reverse: bool) {
    let swap = |rhs: &mut [T], step: usize, pivot_one_based: i32| {
        // INVARIANT: callers validated every pivot in `1..=n` first.
        let pivot = pivot_one_based as usize - 1;
        if pivot != step {
            for col in 0..nrhs {
                rhs.swap(step + col * n, pivot + col * n);
            }
        }
    };
    if reverse {
        for (step, &pivot) in ipiv.iter().enumerate().rev() {
            swap(rhs, step, pivot);
        }
    } else {
        for (step, &pivot) in ipiv.iter().enumerate() {
            swap(rhs, step, pivot);
        }
    }
}

/// Reject a pivot outside `1..=n`.
///
/// A host can validate the whole batch before calling a mutating routine, so that an invalid
/// pivot is reported before any chunk mutates its output. Every solve entry point also validates the
/// pivots it receives, so an implementation is safe on its own.
///
/// # Errors
///
/// Returns [`Error::InvalidArgument`] with role `"pivot"`.
pub fn validate_pivots(op: Op, n: usize, ipiv: &[i32]) -> Result<()> {
    for &pivot_one_based in ipiv {
        let in_range = usize::try_from(pivot_one_based)
            .map(|pivot| (1..=n).contains(&pivot))
            .unwrap_or(false);
        if !in_range {
            return Err(invalid(
                op,
                "pivot",
                format!("LU pivot index {pivot_one_based} is outside 1..={n}"),
            ));
        }
    }
    Ok(())
}

/// Solve `op(A) x = b` for one matrix from packed factors, in place.
///
/// With `P A = L U`: `A x = b` is `x = U^-1 L^-1 P b`, and `A^T x = b` is `x = P^T L^-T U^-T b`.
/// Conjugation conjugates `L` and `U` implicitly.
// INVARIANT: the flags mirror the host's solve attributes one-to-one.
fn solve_one<T: FaerScalar>(
    mut par: faer::Par<'_>,
    (n, nrhs): (usize, usize),
    lu: MatRef<'_, T::Entity>,
    ipiv: &[i32],
    rhs: &mut [T],
    (transpose_a, conjugate_a): (bool, bool),
) {
    let conj = if conjugate_a { Conj::Yes } else { Conj::No };
    if transpose_a {
        {
            let mut x = MatMut::from_column_major_slice_mut(T::entity_slice_mut(rhs), n, nrhs);
            let lu_t = lu.transpose();
            faer::linalg::triangular_solve::solve_lower_triangular_in_place_with_conj(
                lu_t,
                conj,
                x.rb_mut(),
                par.reborrow(),
            );
            faer::linalg::triangular_solve::solve_unit_upper_triangular_in_place_with_conj(
                lu_t,
                conj,
                x,
                par.reborrow(),
            );
        }
        apply_row_swaps(rhs, n, nrhs, ipiv, true);
    } else {
        apply_row_swaps(rhs, n, nrhs, ipiv, false);
        let mut x = MatMut::from_column_major_slice_mut(T::entity_slice_mut(rhs), n, nrhs);
        faer::linalg::triangular_solve::solve_unit_lower_triangular_in_place_with_conj(
            lu,
            conj,
            x.rb_mut(),
            par.reborrow(),
        );
        faer::linalg::triangular_solve::solve_upper_triangular_in_place_with_conj(lu, conj, x, par);
    }
}

/// The pivot rows of one item of a `[n, b...]` pivot descriptor.
struct BatchedPivots<'a> {
    pivots: RawStridedRef<'a, i32>,
    axes: BatchAxes,
    n: usize,
}

impl<'a> BatchedPivots<'a> {
    fn new(
        op: Op,
        n: usize,
        lu: &BatchedRef<'_, impl FaerScalar>,
        pivots: RawStridedRef<'a, i32>,
    ) -> Result<Self> {
        let dims = pivots.dims();
        if dims.first() != Some(&n) {
            return Err(invalid(
                op,
                "configuration",
                format!("pivots describe {dims:?}, expected [{n}, batch...]"),
            ));
        }
        same_batch(op, "pivots", lu.batch_dims(), &dims[1..])?;
        if n > 1 && pivots.strides()[0] != 1 {
            return Err(invalid(
                op,
                "configuration",
                "pivots must be contiguous within an item",
            ));
        }
        Ok(Self {
            axes: BatchAxes::new(&dims[1..], &pivots.strides()[1..]),
            pivots,
            n,
        })
    }

    /// The `n` pivots of item `index` (below the batch count, `n > 0`).
    fn item(&self, index: usize) -> &'a [i32] {
        // SAFETY: `RawStridedRef::new` validated every reachable offset against the borrowed data.
        // Item `index` starts at its batch offset and covers `n` consecutive elements (the core
        // stride is 1, checked in `new`), all of which are reachable offsets, so the slice lies in
        // the borrowed data for `'a`.
        unsafe {
            core::slice::from_raw_parts(
                self.pivots.ptr().wrapping_offset(self.axes.offset(index)),
                self.n,
            )
        }
    }
}

/// Solve `op(A) X = B` for every system of a batch from packed partial-pivot factors.
///
/// `packed_lu` is `[n, n, b...]` and `pivots` is `[n, b...]` (contiguous within an item), both
/// borrowed strided descriptors with the same batch shape; a stride of 0 on a batch axis broadcasts
/// one factorization over many right-hand sides. `output` is the compact, batch-contiguous
/// `n x nrhs` right-hand-side batch; it enters holding `B` and leaves holding `X`. The factors must
/// be nonsingular.
///
/// The triangular solves take no workspace, so `workspace` may be empty.
///
/// # Errors
///
/// Returns [`Error::InvalidArgument`] for a pivot outside `1..=n` or a malformed descriptor, and
/// [`Error::Inconsistent`] when `output` does not hold one right-hand side per item. Pivots are
/// validated for the whole batch **before** any output is written.
///
/// # Examples
///
/// ```
/// use strided_view::RawStridedRef;
/// use tcontract_exec::Exec;
/// use tlinalg::{packed_lu::solve_prepared, Op};
///
/// // A = diag(2, 4) is its own packed factor with identity pivots.
/// let lu = [2.0_f64, 0.0, 0.0, 4.0];
/// let pivots = [1_i32, 2];
/// let mut x = [2.0_f64, 8.0];
/// solve_prepared(
///     Op::LuSolvePrepared,
///     RawStridedRef::new(&lu, &[2, 2], &[1, 2], 0).unwrap(),
///     RawStridedRef::new(&pivots, &[2], &[1], 0).unwrap(),
///     1, &mut x, false, false, &mut Exec::serial(),
/// ).unwrap();
/// assert_eq!(x, [1.0, 2.0]);
/// ```
// INVARIANT: factors, pivots, right-hand-side shape and buffer, the two solve flags, token are distinct operands of the prepared-solve contract.
#[allow(clippy::too_many_arguments)]
pub fn solve_prepared<T: FaerScalar>(
    op: Op,
    packed_lu: RawStridedRef<'_, T>,
    pivots: RawStridedRef<'_, i32>,
    nrhs: usize,
    output: &mut [T],
    transpose_a: bool,
    conjugate_a: bool,
    exec: &mut Exec<'_>,
) -> Result<()> {
    solve_prepared_into(
        op,
        packed_lu,
        pivots,
        nrhs,
        output,
        transpose_a,
        conjugate_a,
        &mut [],
        exec,
    )
}

/// [`solve_prepared`] with the borrowed workspace made explicit.
///
/// The prepared solve needs no library scratch, so any workspace length is accepted (an empty slice
/// is the natural argument).
///
/// # Errors
///
/// As [`solve_prepared`].
#[allow(clippy::too_many_arguments)]
pub fn solve_prepared_into<T: FaerScalar>(
    op: Op,
    packed_lu: RawStridedRef<'_, T>,
    pivots: RawStridedRef<'_, i32>,
    nrhs: usize,
    output: &mut [T],
    transpose_a: bool,
    conjugate_a: bool,
    workspace: &mut [MaybeUninit<u8>],
    exec: &mut Exec<'_>,
) -> Result<()> {
    let lu = BatchedRef::square(op, "packed LU", packed_lu)?;
    let n = lu.rows();
    let rhs_len = checked_product(op, "rhs", &[n, nrhs])?;
    // A prepared solve is the triangular solves alone; the factorization happened in `factor`.
    let item_muladds = weighted_muladds::<T>(n as f64 * n as f64 * nrhs as f64);
    let batch = lu.batch();
    let lanes = batch::decide(exec, batch, item_muladds).lanes;
    batch::check_workspace(op, workspace, 0, lanes)?;
    if rhs_len == 0 {
        return Ok(());
    }
    let pivots = BatchedPivots::new(op, n, &lu, pivots)?;
    if output.len() != checked_product(op, "rhs batch", &[rhs_len, batch])? {
        return Err(Error::Inconsistent {
            op,
            detail: "packed LU, pivot, and batch buffers describe different batches",
        });
    }
    // A fully broadcast pivot batch points every item at the same `n` entries, so validate that
    // vector once instead of re-validating it per item. Any other layout validates every item; the
    // one remembered pointer is not a deduplication cache. Validation still covers the whole
    // logical batch before the driver mutates any output.
    let mut validated: Option<*const i32> = None;
    for index in 0..batch {
        let item = pivots.item(index);
        if validated == Some(item.as_ptr()) {
            continue;
        }
        validate_pivots(op, n, item)?;
        validated = Some(item.as_ptr());
    }
    batch::run(
        op,
        batch,
        exec,
        item_muladds,
        &mut (in_place(output, rhs_len, "rhs batch"),),
        batch::ScratchArena::new(workspace, 0),
        |_par, _stack| (),
        |index, (output,), (), par| {
            // INVARIANT: every pivot was validated above, so `apply_row_swaps` stays in range.
            solve_one::<T>(
                par,
                (n, nrhs),
                lu.item(index),
                pivots.item(index),
                output,
                (transpose_a, conjugate_a),
            );
            Ok(())
        },
    )
}

/// Bytes one batched fused factor+solve call needs in its borrowed workspace.
///
/// # Errors
///
/// [`Error::InvalidArgument`] when a size overflows (canonical `Op`
/// is `Op::LuFactorSolve`).
pub fn factor_solve_workspace<T: FaerScalar>(
    n: usize,
    nrhs: usize,
    batch: usize,
    exec: &Exec<'_>,
) -> Result<usize> {
    factor_solve_bytes::<T>(Op::LuFactorSolve, n, nrhs, batch, exec)
}

/// Factor and solve `A X = B` for every system of a batch, keeping the packed factors, into
/// caller-owned workspace storage.
///
/// `packed_lu` enters holding the compact `n x n` `A` batch and leaves holding the packed factors;
/// `pivots` receives one-based pivots; `output` enters holding the compact `n x nrhs` RHS batch and
/// leaves holding `X`. Workspace contract as [`crate::qr::qr_into`] (`factor_solve_workspace`).
///
/// # Errors
///
/// Returns [`Error::Singular`] for the lowest-indexed item whose factor has an exactly zero `U`
/// diagonal when there is a nonempty RHS to solve, [`Error::Inconsistent`] when the buffers
/// describe different batches, and [`crate::Error::InsufficientWorkspace`] for a short workspace.
/// A zero-column RHS only factors, matching [`factor`] on singular input.
// INVARIANT: shape, the three in-place batch buffers, workspace, token are distinct operands.
#[allow(clippy::too_many_arguments)]
pub fn factor_solve_into<T: FaerScalar>(
    op: Op,
    n: usize,
    nrhs: usize,
    packed_lu: &mut [T],
    pivots: &mut [i32],
    output: &mut [T],
    workspace: &mut [MaybeUninit<u8>],
    exec: &mut Exec<'_>,
) -> Result<()> {
    let (matrix_len, rhs_len, batch, item_muladds) =
        factor_solve_shape::<T>(op, n, nrhs, packed_lu.len(), pivots.len(), output.len())?;
    let requirement = lane_requirement(
        op,
        exec,
        batch,
        item_muladds,
        || check_solve_shape(op, n, nrhs),
        |par| factor_lane_req::<T>(n, n, par),
    )?;
    let (lanes, per_lane) = requirement.unwrap_or((1, 0));
    batch::check_workspace(op, workspace, per_lane, lanes)?;
    if matrix_len == 0 {
        return Ok(());
    }
    let zero = T::default();
    batch::run(
        op,
        batch,
        exec,
        item_muladds,
        &mut (
            in_place(packed_lu, matrix_len, "packed LU"),
            in_place(pivots, n, "pivots"),
            in_place(output, rhs_len, "rhs batch"),
        ),
        batch::ScratchArena::new(workspace, per_lane),
        |par, stack| FactorScratch::<T>::new(n, n, par, MemStack::new(stack)),
        |_index, (lu, ipiv, output), scratch, mut par| {
            {
                let mat = MatMut::from_column_major_slice_mut(T::entity_slice_mut(lu), n, n);
                scratch.factor(par.reborrow(), mat, ipiv, op)?;
            }
            if rhs_len > 0 {
                if (0..n).any(|i| lu[i + i * n] == zero) {
                    return Err(Error::Singular { op });
                }
                let lu = MatRef::from_column_major_slice(T::entity_slice(lu), n, n);
                solve_one::<T>(par, (n, nrhs), lu, ipiv, output, (false, false));
            }
            Ok(())
        },
    )
}

/// Factor and solve `A X = B` for every system of a batch, keeping the packed factors.
///
/// `packed_lu` enters holding the compact `n x n` `A` batch and leaves holding the packed factors;
/// `pivots` receives one-based pivots; `output` enters holding the compact `n x nrhs` RHS batch and
/// leaves holding `X`.
///
/// This is the wrapper of [`factor_solve_into`].
///
/// # Errors
///
/// Returns [`Error::Singular`] when a matrix is exactly singular and there is a nonempty RHS to
/// solve, [`Error::Inconsistent`] for inconsistent buffers, and [`Error::InvalidArgument`] for a
/// malformed shape.
///
/// # Examples
///
/// ```
/// use tcontract_exec::Exec;
/// use tlinalg::{packed_lu::factor_solve, Op};
///
/// let mut lu = [2.0_f64, 0.0, 0.0, 4.0];
/// let mut pivots = [0_i32; 2];
/// let mut x = [2.0_f64, 8.0];
/// factor_solve(Op::LuFactorSolve, 2, 1, &mut lu, &mut pivots, &mut x, &mut Exec::serial()).unwrap();
/// assert_eq!(x, [1.0, 2.0]);
/// ```
// INVARIANT: shape, the three in-place batch buffers, token are distinct operands.
#[allow(clippy::too_many_arguments)]
pub fn factor_solve<T: FaerScalar>(
    op: Op,
    n: usize,
    nrhs: usize,
    packed_lu: &mut [T],
    pivots: &mut [i32],
    output: &mut [T],
    exec: &mut Exec<'_>,
) -> Result<()> {
    // The published order: shape and buffer validation before any workspace query or allocation.
    let (_, _, batch, _) =
        factor_solve_shape::<T>(op, n, nrhs, packed_lu.len(), pivots.len(), output.len())?;
    let bytes = factor_solve_bytes::<T>(op, n, nrhs, batch, exec)?;
    let mut workspace = vec![MaybeUninit::<u8>::uninit(); bytes];
    factor_solve_into(op, n, nrhs, packed_lu, pivots, output, &mut workspace, exec)
}
