//! faer-backed batched full-pivot LU factorization and solve.
//!
//! Moved from `tenferro-linalg`'s faer backend (same project, MIT OR Apache-2.0).
//!
//! The factorization never fails on an exactly singular input; the solve rejects an effectively
//! singular factor (a pivot at or below `ε · max|pivot|`), exactly as before. Faer's work matrix,
//! permutation vectors and scratch are lane scratch, reused for every item of the lane.

use faer::dyn_stack::{MemBuffer, MemStack};
use faer::{Mat, MatMut, MatRef};
use strided_view::RawStridedRef;

use crate::batch::{self, out, same_batch, BatchedRef, Push, Sink};
use crate::scalar::ScalarEntity;
use crate::util::{checked_product, invalid, push_masked, push_permutation, weighted_muladds};
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
struct FullPivScratch<E: faer::traits::ComplexField> {
    lu: Mat<E>,
    row_perm: Vec<usize>,
    row_perm_inv: Vec<usize>,
    col_perm: Vec<usize>,
    col_perm_inv: Vec<usize>,
    mem: MemBuffer,
}

impl<E: faer::traits::ComplexField> FullPivScratch<E> {
    fn new(n: usize, par: faer::Par) -> Self {
        Self {
            lu: Mat::zeros(n, n),
            row_perm: vec![0; n],
            row_perm_inv: vec![0; n],
            col_perm: vec![0; n],
            col_perm_inv: vec![0; n],
            mem: MemBuffer::new(
                faer::linalg::lu::full_pivoting::factor::lu_in_place_scratch::<usize, E>(
                    n,
                    n,
                    par,
                    Default::default(),
                ),
            ),
        }
    }

    /// Factor `mat` into the scratch; returns the transposition count.
    fn factor(&mut self, mat: MatRef<'_, E>, par: faer::Par) -> usize {
        self.lu.copy_from(mat);
        faer::linalg::lu::full_pivoting::factor::lu_in_place(
            self.lu.as_mut(),
            &mut self.row_perm,
            &mut self.row_perm_inv,
            &mut self.col_perm,
            &mut self.col_perm_inv,
            par,
            MemStack::new(&mut self.mem),
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
    scratch: &mut FullPivScratch<T::Entity>,
    par: faer::Par,
) {
    let n = mat.nrows();
    let transpositions = scratch.factor(mat, par);
    push_permutation(p, &scratch.row_perm_inv);
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
    push_permutation(q, &scratch.col_perm_inv);
    parity.push(T::parity(transpositions % 2 != 0));
}

/// Full-pivot LU of every `n x n` matrix of a batch, `P A Qᵀ = L U`.
///
/// `input` is `[n, n, b...]`.
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
    let item_muladds = weighted_muladds::<T>((n as f64).powi(3) / 3.0);
    batch::run(
        op,
        input.batch(),
        exec,
        item_muladds,
        &mut (
            out(p, len),
            out(l, len),
            out(u, len),
            out(q, len),
            out(parity, 1),
        ),
        |par| FullPivScratch::<T::Entity>::new(n, par),
        |index, (p, l, u, q, parity), scratch, par| {
            full_piv_lu_item::<T>(input.item(index), (p, l, u, q, parity), scratch, par);
            Ok(())
        },
    )
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
    (scratch, solve_mem): &mut (FullPivScratch<T::Entity>, MemBuffer),
    par: faer::Par,
) -> Result<()> {
    let n = a.nrows();
    scratch.factor(a, par);
    let lu = scratch.lu.as_ref();
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
            faer::perm::PermRef::new_unchecked(&scratch.row_perm, &scratch.row_perm_inv, n),
            faer::perm::PermRef::new_unchecked(&scratch.col_perm, &scratch.col_perm_inv, n),
        )
    };
    let stack = MemStack::new(solve_mem);
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

/// Solve every linear system of a batch through a full-pivot LU of `A`.
///
/// `a` is `[n, n, b...]` and `b` is `[n, nrhs, b...]` with the same batch shape. `x` receives the
/// column-major `n x nrhs` solution of `A X = B` (or `Aᵀ X = B` when `transpose_a`) per item.
///
/// Each item's right-hand side is read only after that item's factor passed the singularity
/// check, as in the pre-extraction code.
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
    let a = BatchedRef::square(op, "A", a)?;
    let b = BatchedRef::new(op, "B", b)?;
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
    let (nf, krhs) = (n as f64, nrhs as f64);
    let item_muladds = weighted_muladds::<T>(nf.powi(3) / 3.0 + nf * nf * krhs);
    batch::run(
        op,
        a.batch(),
        exec,
        item_muladds,
        &mut (out(x, x_len),),
        |par| {
            (
                FullPivScratch::<T::Entity>::new(n, par),
                MemBuffer::new(
                    faer::linalg::lu::full_pivoting::solve::solve_in_place_scratch::<
                        usize,
                        T::Entity,
                    >(n, nrhs, par),
                ),
            )
        },
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
