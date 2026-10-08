//! faer-backed batched triangular solve.
//!
//! Moved from `tenferro-linalg`'s faer backend (same project, MIT OR Apache-2.0).
//!
//! # Boundary
//!
//! The coefficient batch `a` is `[n, n, b...]` and the right-hand-side batch `b` is
//! `[b_rows, b_cols, b...]`, both borrowed strided descriptors. The solution `x` is cleared and
//! filled with compact column-major items in batch order:
//!
//! * a left-side solve `A X = B` writes `B` straight into the item's output chunk (one copy) and
//!   solves it there in place;
//! * a right-side solve `X A = B` writes `B` into the output chunk and solves the transposed
//!   system `Aᵀ Xᵀ = Bᵀ` on that chunk's transposed view, so no lane-local work matrix is needed.

use faer::{MatMut, MatRef};
use strided_view::RawStridedRef;

use crate::batch::{self, out, same_batch, BatchedRef, Sink};
use crate::util::invalid;
use crate::{FaerScalar, Op, Parallel, Result};

/// Which triangle and which variant of the coefficient matrix a triangular solve uses.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TriangularSolveFlags {
    /// Solve `A X = B` when `true`, `X A = B` when `false`.
    pub left_side: bool,
    /// `A` is stored in its lower triangle when `true`, its upper triangle when `false`.
    pub lower: bool,
    /// Use `Aᵀ` instead of `A`.
    pub transpose_a: bool,
    /// Treat the diagonal of `A` as ones without reading it.
    pub unit_diagonal: bool,
}

/// Dispatch faer's triangular solve from the triangle/transpose/unit flags.
///
/// Transposing `A` swaps which triangle is stored, so the four faer routines cover all eight flag
/// combinations once that flip is applied.
fn solve_in_place<E: faer::traits::ComplexField>(
    a: MatRef<'_, E>,
    rhs: MatMut<'_, E>,
    lower: bool,
    transpose_a: bool,
    unit_diagonal: bool,
    par: faer::Par,
) {
    let effective_lower = lower != transpose_a;
    let a = if transpose_a { a.transpose() } else { a };
    match (effective_lower, unit_diagonal) {
        (true, false) => {
            faer::linalg::triangular_solve::solve_lower_triangular_in_place(a, rhs, par);
        }
        (true, true) => {
            faer::linalg::triangular_solve::solve_unit_lower_triangular_in_place(a, rhs, par);
        }
        (false, false) => {
            faer::linalg::triangular_solve::solve_upper_triangular_in_place(a, rhs, par);
        }
        (false, true) => {
            faer::linalg::triangular_solve::solve_unit_upper_triangular_in_place(a, rhs, par);
        }
    }
}

/// Solve one system into the lane's output chunk.
///
/// Left side: `B` is written straight into the item's output region (that write initialises it)
/// and solved there in place. Right side: `B` is written there too and the transposed system is
/// solved on the region's transposed view.
fn triangular_solve_item<T: FaerScalar>(
    a: MatRef<'_, T::Entity>,
    b: MatRef<'_, T::Entity>,
    flags: TriangularSolveFlags,
    x: &mut Sink<'_, T>,
    par: faer::Par,
) {
    let TriangularSolveFlags {
        left_side,
        lower,
        transpose_a,
        unit_diagonal,
    } = flags;
    let (b_rows, b_cols) = (b.nrows(), b.ncols());
    if left_side {
        let region = x.fill(b_rows * b_cols, |index| {
            T::from_entity(b[(index % b_rows, index / b_rows)])
        });
        let rhs = MatMut::from_column_major_slice_mut(T::entity_slice_mut(region), b_rows, b_cols);
        solve_in_place(a, rhs, lower, transpose_a, unit_diagonal, par);
    } else {
        // Right-side solve `X A = B` is the left-side solve of the transposed system. Write `B`
        // into the output region and solve `Aᵀ Xᵀ = Bᵀ` on its transposed view in place; the
        // storage then holds `X` in its original `b_rows x n` layout.
        let region = x.fill(b_rows * b_cols, |index| {
            T::from_entity(b[(index % b_rows, index / b_rows)])
        });
        let rhs = MatMut::from_column_major_slice_mut(T::entity_slice_mut(region), b_rows, b_cols);
        solve_in_place(
            a,
            rhs.transpose_mut(),
            lower,
            !transpose_a,
            unit_diagonal,
            par,
        );
    }
}

/// Solve every triangular system of a batch.
///
/// `a` is `[n, n, b...]` and `b` is `[b_rows, b_cols, b...]` with the same batch shape. `x`
/// receives the column-major solution per item: `n x b_cols` for a left-side solve, `b_rows x n`
/// for a right-side one.
///
/// # Errors
///
/// [`Error::InvalidArgument`](crate::Error::InvalidArgument) when `a` is not a batch of square
/// matrices, the batch shapes differ, or the right-hand side's solved dimension is not `n`
/// (`b_rows` for a left-side solve, `b_cols` for a right-side one). A host that reports a shape
/// mismatch for that case checks it before calling. `x` is empty on error.
///
/// # Examples
///
/// ```
/// use strided_view::RawStridedRef;
/// use tlinalg::triangular_solve::{triangular_solve, TriangularSolveFlags};
/// use tlinalg::{Op, Parallel};
///
/// // Lower-triangular A = [[2, 0], [1, 1]], B = [2, 3].
/// let a = [2.0_f64, 1.0, 0.0, 1.0];
/// let b = [2.0_f64, 3.0];
/// let flags = TriangularSolveFlags { left_side: true, lower: true, transpose_a: false, unit_diagonal: false };
/// let mut x = Vec::new();
/// triangular_solve(
///     Op::TriangularSolve,
///     RawStridedRef::new(&a, &[2, 2], &[1, 2], 0).unwrap(),
///     RawStridedRef::new(&b, &[2, 1], &[1, 2], 0).unwrap(),
///     flags, &mut x, Parallel::Sequential,
/// ).unwrap();
/// assert_eq!(x, [1.0, 2.0]);
/// ```
pub fn triangular_solve<T: FaerScalar>(
    op: Op,
    a: RawStridedRef<'_, T>,
    b: RawStridedRef<'_, T>,
    flags: TriangularSolveFlags,
    x: &mut Vec<T>,
    par: Parallel<'_>,
) -> Result<()> {
    x.clear();
    let a = BatchedRef::square(op, "A", a)?;
    let b = BatchedRef::new(op, "B", b)?;
    same_batch(op, "B", a.batch_dims(), b.batch_dims())?;
    let n = a.rows();
    let (b_rows, b_cols) = (b.rows(), b.cols());
    if flags.left_side {
        if b_rows != n {
            return Err(invalid(
                op,
                "configuration",
                format!("right-hand side has {b_rows} rows, expected {n}"),
            ));
        }
    } else if b_cols != n {
        return Err(invalid(
            op,
            "configuration",
            format!("right-hand side has {b_cols} columns, expected {n}"),
        ));
    }
    let item_len = crate::util::checked_product(op, "X", &[b_rows, b_cols])?;
    batch::run(
        op,
        a.batch(),
        par,
        Some(n.max(b_rows).max(b_cols)),
        &mut (out(x, item_len),),
        |_| (),
        |index, (x,), (), par| {
            triangular_solve_item::<T>(a.item(index), b.item(index), flags, x, par);
            Ok(())
        },
    )
}
