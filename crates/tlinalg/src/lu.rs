//! faer-backed batched partial-pivot LU factorization and the LU-based linear solve.
//!
//! Moved from `tenferro-linalg`'s faer backend (same project, MIT OR Apache-2.0). The packed
//! (LAPACK-format) LU family used by prepared solves lives in [`crate::packed_lu`]; this module is
//! the explicit-factor `P, L, U` decomposition and the one-shot solve.
//!
//! Faer's work matrices, permutation vectors and scratch are lane scratch, reused for every item of
//! the lane.

use faer::dyn_stack::{MemBuffer, MemStack};
use faer::{Mat, MatMut, MatRef};
use strided_view::{RawStridedMut, RawStridedRef};

use crate::batch::{self, out, same_batch, BatchedMut, BatchedRef};
use crate::util::{checked_product, invalid, push_masked, push_permutation, weighted_muladds};
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
pub(crate) struct LuScratch<E: faer::traits::ComplexField> {
    lu: Mat<E>,
    perm: Vec<usize>,
    perm_inv: Vec<usize>,
    mem: MemBuffer,
}

impl<E: faer::traits::ComplexField> LuScratch<E> {
    fn new(m: usize, n: usize, par: faer::Par) -> Self {
        Self {
            lu: Mat::zeros(m, n),
            perm: vec![0; m],
            perm_inv: vec![0; m],
            mem: MemBuffer::new(
                faer::linalg::lu::partial_pivoting::factor::lu_in_place_scratch::<usize, E>(
                    m,
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
        faer::linalg::lu::partial_pivoting::factor::lu_in_place(
            self.lu.as_mut(),
            &mut self.perm,
            &mut self.perm_inv,
            par,
            MemStack::new(&mut self.mem),
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
    scratch: &mut LuScratch<T::Entity>,
    par: faer::Par,
) {
    let (m, n) = (mat.nrows(), mat.ncols());
    let k = m.min(n);
    let transpositions = scratch.factor(mat, par);
    push_permutation(p, &scratch.perm_inv);
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

/// Partial-pivot LU of every `m x n` matrix of a batch, `P A = L U`.
///
/// `input` is `[m, n, b...]`. An exactly singular input is not an error.
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
///     Exec::serial(),
/// ).unwrap();
/// assert_eq!(parity, [-1.0]);
/// assert_eq!(u[0], 3.0);
/// ```
pub fn lu<T: FaerScalar>(
    op: Op,
    input: RawStridedRef<'_, T>,
    factors: LuFactors<'_, T>,
    exec: Exec<'_>,
) -> Result<()> {
    let LuFactors { p, l, u, parity } = factors;
    p.clear();
    l.clear();
    u.clear();
    parity.clear();
    let input = BatchedRef::new(op, "input", input)?;
    let (m, n) = (input.rows(), input.cols());
    let k = m.min(n);
    let p_len = checked_product(op, "permutation matrix", &[m, m])?;
    let l_len = checked_product(op, "L", &[m, k])?;
    let u_len = checked_product(op, "U", &[k, n])?;
    let (mf, nf, kf) = (m as f64, n as f64, k as f64);
    let item_muladds =
        weighted_muladds::<T>(mf * nf * kf - (mf + nf) * kf * kf / 2.0 + kf.powi(3) / 3.0);
    batch::run(
        op,
        input.batch(),
        exec,
        Some(m.max(n)),
        item_muladds,
        &mut (out(p, p_len), out(l, l_len), out(u, u_len), out(parity, 1)),
        |par| LuScratch::<T::Entity>::new(m, n, par),
        |index, (p, l, u, parity), scratch, par| {
            lu_item::<T>(input.item(index), (p, l, u, parity), scratch, par);
            Ok(())
        },
    )
}

/// Solve one system into `destination`; written only after the singularity check.
fn solve_item<T: FaerScalar>(
    op: Op,
    a: MatRef<'_, T::Entity>,
    source: Option<MatRef<'_, T::Entity>>,
    mut destination: MatMut<'_, T::Entity>,
    transpose_a: bool,
    scratch: &mut (LuScratch<T::Entity>, MemBuffer),
    par: faer::Par,
) -> Result<()> {
    let n = a.nrows();
    let (lu, solve_mem) = scratch;
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
    let perm = unsafe { faer::perm::PermRef::new_unchecked(&lu.perm, &lu.perm_inv, n) };
    let stack = MemStack::new(solve_mem);
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

/// Solve every linear system `A X = B` (or `Aᵀ X = B`) of a batch through a partial-pivot LU of
/// `A`, writing the solutions into a caller-owned destination.
///
/// `a` is `[n, n, b...]` and `out` is `[n, nrhs, b...]` with the same batch shape. When `rhs` is
/// `Some(b)`, `b` (`[n, nrhs, b...]`) is copied into `out` per item first; when it is `None`, `out`
/// already holds the right-hand sides. `out` must not alias itself.
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
///     false, Exec::serial(),
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
    exec: Exec<'_>,
) -> Result<()> {
    let a = BatchedRef::square(op, "A", a)?;
    let destination = BatchedMut::new(op, "out", &mut out)?;
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
    let (nf, krhs) = (n as f64, nrhs as f64);
    // `solve_item` factors `A` (an `n x n` LU) and then solves, so the estimate is
    // factor-and-solve, not a prepared solve.
    let item_muladds = weighted_muladds::<T>(nf.powi(3) / 3.0 + nf * nf * krhs);
    batch::run(
        op,
        a.batch(),
        exec,
        Some(batch::auto_extent(n, nrhs)),
        item_muladds,
        &mut (),
        |par| {
            let solve_req = if transpose_a {
                faer::linalg::lu::partial_pivoting::solve::solve_transpose_in_place_scratch::<
                    usize,
                    T::Entity,
                >(n, nrhs, par)
            } else {
                faer::linalg::lu::partial_pivoting::solve::solve_in_place_scratch::<usize, T::Entity>(
                    n, nrhs, par,
                )
            };
            (
                LuScratch::<T::Entity>::new(n, n, par),
                MemBuffer::new(solve_req),
            )
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
