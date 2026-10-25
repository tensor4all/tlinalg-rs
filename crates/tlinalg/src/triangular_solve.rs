//! faer-backed batched triangular solve.
//!
//! Moved from `tenferro-linalg`'s faer backend (same project, MIT OR Apache-2.0).
//!
//! # Boundary
//!
//! The coefficient batch `a` is `[n, n, b...]` and the right-hand-side batch `b` is
//! `[b_rows, b_cols, b...]`, both borrowed strided descriptors. The solution `x` is filled with
//! compact column-major items in batch order:
//!
//! * a left-side solve `A X = B` writes `B` straight into the item's output chunk (one copy) and
//!   solves it there in place;
//! * a right-side solve `X A = B` writes `B` into the output chunk and solves the transposed
//!   system `Aᵀ Xᵀ = Bᵀ` on that chunk's transposed view, so no lane-local work matrix is needed.
//!
//! This family needs no library scratch: the output chunk is the work matrix, and faer's
//! triangular solve takes no workspace. Its `*_workspace` query therefore reports zero bytes, and the caller may pass an empty slice to `*_into`.

use core::mem::MaybeUninit;

use faer::{MatMut, MatRef};
use strided_view::RawStridedRef;

use crate::batch::{self, fixed_out, same_batch, BatchedRef, Sink, VecOut};
use crate::util::{invalid, weighted_muladds};
use crate::{FaerScalar, Op, Result};
use tcontract_exec::Exec;

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
    mut par: faer::Par<'_>,
) {
    let effective_lower = lower != transpose_a;
    let a = if transpose_a { a.transpose() } else { a };
    match (effective_lower, unit_diagonal) {
        (true, false) => {
            faer::linalg::triangular_solve::solve_lower_triangular_in_place(a, rhs, par.reborrow());
        }
        (true, true) => {
            faer::linalg::triangular_solve::solve_unit_lower_triangular_in_place(
                a,
                rhs,
                par.reborrow(),
            );
        }
        (false, false) => {
            faer::linalg::triangular_solve::solve_upper_triangular_in_place(a, rhs, par.reborrow());
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
    mut par: faer::Par<'_>,
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
        solve_in_place(a, rhs, lower, transpose_a, unit_diagonal, par.reborrow());
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

fn item_muladds<T: FaerScalar>(
    n: usize,
    b_rows: usize,
    b_cols: usize,
    flags: TriangularSolveFlags,
) -> f64 {
    // One triangular solve touches `n x n` against the solved right-hand side count.
    let solved = if flags.left_side { b_cols } else { b_rows };
    weighted_muladds::<T>(n as f64 * n as f64 * solved as f64 / 2.0)
}

/// Bytes one batched triangular solve needs in its borrowed workspace: none.
///
/// The output chunk is the work matrix and faer's triangular solve takes no workspace, so this is
/// `0` for every shape and route. The matching
/// [`triangular_solve_into`] accepts any workspace slice, including an empty one.
///
/// # Errors
///
/// None today: the signature keeps the `Result` every `*_workspace` query shares.
pub fn triangular_solve_workspace<T: FaerScalar>(
    n: usize,
    b_rows: usize,
    b_cols: usize,
    flags: TriangularSolveFlags,
    batch: usize,
    exec: &Exec<'_>,
) -> Result<usize> {
    // Nothing to size: the answer does not depend on the shape or the route.
    let _ = (n, b_rows, b_cols, flags, batch, exec);
    Ok(0)
}

/// Solve every triangular system of a batch into caller-owned storage.
///
/// `x` holds `b_rows * b_cols` uninitialised elements per batch item and receives the column-major
/// solution per item: `n x b_cols` for a left-side solve, `b_rows x n` for a right-side one. The
/// workspace may be empty (see [`triangular_solve_workspace`]).
///
/// # Errors
///
/// [`Error::InvalidArgument`](crate::Error::InvalidArgument) when `a` is not a batch of square
/// matrices, the batch shapes differ, the right-hand side's solved dimension is not `n` (`b_rows`
/// for a left-side solve, `b_cols` for a right-side one), or the output length does not match. A
/// host that reports a shape mismatch for that case checks it before calling.
pub fn triangular_solve_into<T: FaerScalar>(
    op: Op,
    a: RawStridedRef<'_, T>,
    b: RawStridedRef<'_, T>,
    flags: TriangularSolveFlags,
    x: &mut [MaybeUninit<T>],
    workspace: &mut [MaybeUninit<u8>],
    exec: &mut Exec<'_>,
) -> Result<()> {
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
    let muladds = item_muladds::<T>(n, b_rows, b_cols, flags);
    let lanes = batch::decide(exec, a.batch(), muladds).lanes;
    batch::check_workspace(op, workspace, 0, lanes)?;
    batch::run(
        op,
        a.batch(),
        exec,
        muladds,
        &mut (fixed_out(x, item_len, "X"),),
        batch::ScratchArena::new(workspace, 0),
        |_par, _stack| (),
        |index, (x,), (), par| {
            triangular_solve_item::<T>(a.item(index), b.item(index), flags, x, par);
            Ok(())
        },
    )
}

/// Solve every triangular system of a batch.
///
/// `a` is `[n, n, b...]` and `b` is `[b_rows, b_cols, b...]` with the same batch shape. `x`
/// receives the column-major solution per item: `n x b_cols` for a left-side solve, `b_rows x n`
/// for a right-side one.
///
/// This is the `Vec` wrapper of [`triangular_solve_into`] (no scratch to allocate).
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
/// use tcontract_exec::Exec;
/// use tlinalg::triangular_solve::{triangular_solve, TriangularSolveFlags};
/// use tlinalg::Op;
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
///     flags, &mut x, &mut Exec::serial(),
/// ).unwrap();
/// assert_eq!(x, [1.0, 2.0]);
/// ```
pub fn triangular_solve<T: FaerScalar>(
    op: Op,
    a: RawStridedRef<'_, T>,
    b: RawStridedRef<'_, T>,
    flags: TriangularSolveFlags,
    x: &mut Vec<T>,
    exec: &mut Exec<'_>,
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
    // The published order: operand validation first. No scratch is needed, so the workspace is empty.
    let batch = a.batch();
    let mut workspace: [MaybeUninit<u8>; 0] = [];
    let mut x_out = VecOut::new(
        x,
        crate::util::checked_product(op, "X", &[item_len, batch])?,
    );
    triangular_solve_into(
        op,
        a.raw(),
        b.raw(),
        flags,
        x_out.spare(),
        &mut workspace,
        exec,
    )?;
    x_out.publish();
    Ok(())
}
