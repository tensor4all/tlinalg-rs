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
//! batch-contiguous buffers, updated in place, and the batch is split into
//! `batch.div_ceil(lanes)`-item chunks that run as tasks on the caller's pool. This replaces the
//! host-side chunk fan-out the pre-extraction code used. The prepared solve reads its factors and
//! pivots through strided descriptors, so a host can broadcast one factorization (stride 0 on a batch
//! axis) against many right-hand sides.
//!
//! Scratch is native and per lane: four permutation vectors and one faer `MemBuffer`, reused across
//! every matrix of the lane.
//!
//! In-place buffers are not rolled back on error: items before the failing one (and items of other
//! lanes) may already hold their results.

use core::marker::PhantomData;

use faer::dyn_stack::{MemBuffer, MemStack};
use faer::prelude::ReborrowMut;
use faer::{Conj, MatMut, MatRef};

use strided_view::RawStridedRef;

use crate::batch::{self, in_place, same_batch, BatchAxes, BatchedRef};
use crate::util::{checked_product, invalid};
use crate::{Error, FaerScalar, Op, Parallel, Result};

/// Reusable per-lane state for factoring a run of `m x n` matrices.
///
/// Typed by the scalar it was sized for, so mixing a scratch with another scalar is a compile
/// error. The batch driver builds one per lane for the call's shape.
pub(crate) struct FactorScratch<T: FaerScalar> {
    m: usize,
    k: usize,
    perm: Vec<usize>,
    perm_inv: Vec<usize>,
    current: Vec<usize>,
    position: Vec<usize>,
    mem: MemBuffer,
    scalar: PhantomData<T>,
}

impl<T: FaerScalar> FactorScratch<T> {
    /// Size the scratch for `m x n` matrices and the given parallelism.
    ///
    /// The parallelism participates because faer sizes its scratch per thread.
    fn new(m: usize, n: usize, par: faer::Par) -> Self {
        Self {
            m,
            k: m.min(n),
            perm: vec![0; m],
            perm_inv: vec![0; m],
            current: vec![0; m],
            position: vec![0; m],
            mem: MemBuffer::new(
                faer::linalg::lu::partial_pivoting::factor::lu_in_place_scratch::<usize, T::Entity>(
                    m,
                    n,
                    par,
                    Default::default(),
                ),
            ),
            scalar: PhantomData,
        }
    }

    /// Factor one compact column-major matrix in place and write its one-based swap sequence into
    /// `ipiv`. Returns whether the permutation is odd.
    fn factor(
        &mut self,
        par: faer::Par,
        matrix: MatMut<'_, T::Entity>,
        ipiv: &mut [i32],
        op: Op,
    ) -> Result<bool> {
        let stack = MemStack::new(&mut self.mem);
        let info = faer::linalg::lu::partial_pivoting::factor::lu_in_place(
            matrix,
            &mut self.perm,
            &mut self.perm_inv,
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

/// Factor every compact column-major `m x n` matrix of `lu` in place.
///
/// `lu` holds `batch = parity.len()` matrices; `pivots` receives `min(m, n)` one-based pivots per
/// matrix and `parity` one permutation parity per matrix. Exactly singular matrices are **not** an
/// error, matching LAPACK `?getrf` with positive `info`.
///
/// # Errors
///
/// Returns [`Error::Inconsistent`] when the buffers describe different batches, and
/// [`Error::InvalidArgument`] when faer returns an invalid row permutation (lowest-indexed item).
///
/// # Examples
///
/// ```
/// use tlinalg::{packed_lu::factor, Op, Parallel};
///
/// let mut lu = [1.0_f64, 3.0, 2.0, 4.0];
/// let (mut pivots, mut parity) = ([0_i32; 2], [0.0_f64; 1]);
/// factor(Op::LuFactor, 2, 2, &mut lu, &mut pivots, &mut parity, Parallel::Sequential).unwrap();
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
    par: Parallel<'_>,
) -> Result<()> {
    let k = m.min(n);
    let matrix_len = checked_product(op, "matrix shape", &[m, n])?;
    let batch = parity.len();
    check_batches(
        op,
        [(matrix_len, lu.len()), (k, pivots.len()), (1, parity.len())],
        batch,
    )?;
    if matrix_len == 0 || batch == 0 {
        return Ok(());
    }
    batch::run(
        op,
        batch,
        par,
        None,
        &mut (
            in_place(lu, matrix_len, "packed LU"),
            in_place(pivots, k, "pivots"),
            in_place(parity, 1, "parity"),
        ),
        |par| FactorScratch::<T>::new(m, n, par),
        |index, (lu, pivots, parity), scratch, par| {
            let mat =
                MatMut::from_column_major_slice_mut(T::entity_slice_mut(lu.item(index)), m, n);
            let odd = scratch.factor(par, mat, pivots.item(index), op)?;
            parity.item(index)[0] = T::parity(odd);
            Ok(())
        },
    )
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
    par: faer::Par,
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
                par,
            );
            faer::linalg::triangular_solve::solve_unit_upper_triangular_in_place_with_conj(
                lu_t, conj, x, par,
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
            par,
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
/// use tlinalg::{packed_lu::solve_prepared, Op, Parallel};
///
/// // A = diag(2, 4) is its own packed factor with identity pivots.
/// let lu = [2.0_f64, 0.0, 0.0, 4.0];
/// let pivots = [1_i32, 2];
/// let mut x = [2.0_f64, 8.0];
/// solve_prepared(
///     Op::LuSolvePrepared,
///     RawStridedRef::new(&lu, &[2, 2], &[1, 2], 0).unwrap(),
///     RawStridedRef::new(&pivots, &[2], &[1], 0).unwrap(),
///     1, &mut x, false, false, Parallel::Sequential,
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
    par: Parallel<'_>,
) -> Result<()> {
    let lu = BatchedRef::square(op, "packed LU", packed_lu)?;
    let n = lu.rows();
    let rhs_len = checked_product(op, "rhs", &[n, nrhs])?;
    if rhs_len == 0 {
        return Ok(());
    }
    let pivots = BatchedPivots::new(op, n, &lu, pivots)?;
    let batch = lu.batch();
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
        par,
        None,
        &mut (in_place(output, rhs_len, "rhs batch"),),
        |_| (),
        |index, (output,), (), par| {
            // INVARIANT: every pivot was validated above, so `apply_row_swaps` stays in range.
            solve_one::<T>(
                par,
                (n, nrhs),
                lu.item(index),
                pivots.item(index),
                output.item(index),
                (transpose_a, conjugate_a),
            );
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
/// # Errors
///
/// Returns [`Error::Singular`] for the lowest-indexed item whose factor has an exactly zero `U`
/// diagonal when there is a nonempty RHS to solve, and [`Error::Inconsistent`] when the buffers
/// describe different batches. A zero-column RHS only factors, matching [`factor`] on singular
/// input.
///
/// # Examples
///
/// ```
/// use tlinalg::{packed_lu::factor_solve, Op, Parallel};
///
/// let mut lu = [2.0_f64, 0.0, 0.0, 4.0];
/// let mut pivots = [0_i32; 2];
/// let mut x = [2.0_f64, 8.0];
/// factor_solve(Op::LuFactorSolve, 2, 1, &mut lu, &mut pivots, &mut x, Parallel::Sequential).unwrap();
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
    par: Parallel<'_>,
) -> Result<()> {
    let matrix_len = checked_product(op, "matrix", &[n, n])?;
    let rhs_len = checked_product(op, "rhs", &[n, nrhs])?;
    if matrix_len == 0 {
        return Ok(());
    }
    let batch = packed_lu.len() / matrix_len;
    check_batches(
        op,
        [
            (matrix_len, packed_lu.len()),
            (n, pivots.len()),
            (rhs_len, output.len()),
        ],
        batch,
    )?;
    let zero = T::default();
    batch::run(
        op,
        batch,
        par,
        None,
        &mut (
            in_place(packed_lu, matrix_len, "packed LU"),
            in_place(pivots, n, "pivots"),
            in_place(output, rhs_len, "rhs batch"),
        ),
        |par| FactorScratch::<T>::new(n, n, par),
        |index, (lu, pivots, output), scratch, par| {
            let matrix = lu.item(index);
            let ipiv = pivots.item(index);
            {
                let mat = MatMut::from_column_major_slice_mut(T::entity_slice_mut(matrix), n, n);
                scratch.factor(par, mat, ipiv, op)?;
            }
            if rhs_len > 0 {
                if (0..n).any(|i| matrix[i + i * n] == zero) {
                    return Err(Error::Singular { op });
                }
                let lu = MatRef::from_column_major_slice(T::entity_slice(matrix), n, n);
                solve_one::<T>(par, (n, nrhs), lu, ipiv, output.item(index), (false, false));
            }
            Ok(())
        },
    )
}
