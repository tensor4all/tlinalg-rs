//! faer-backed batched Cholesky factorization.
//!
//! Moved from `tenferro-linalg`'s faer backend (same project, MIT OR Apache-2.0).
//!
//! The factorization reads the **lower** triangle of each `A` and returns the lower-triangular `L`
//! with `A = L Lᴴ`, column-major, zero above the diagonal. The output chunk is the destructive
//! work matrix and faer's stack scratch is lane scratch, reused for every item of the lane.

use faer::dyn_stack::{MemBuffer, MemStack};
use faer::{MatMut, MatRef};
use strided_view::RawStridedRef;

use crate::batch::{self, out, BatchedRef, Sink};
use crate::util::checked_product;
use crate::{Error, FaerScalar, Op, Parallel, Result};

fn cholesky_item<T: FaerScalar>(
    op: Op,
    mat: MatRef<'_, T::Entity>,
    l: &mut Sink<'_, T>,
    mem: &mut MemBuffer,
    par: faer::Par,
) -> Result<()> {
    let n = mat.nrows();
    // The output chunk is the destructible copy. Fill it with `mat` and factor in place;
    // `cholesky_in_place` reads the lower triangle only, so the upper part is cleared afterwards.
    // The bulk `copy_from` is the optimized copy path the staging matrix used to get.
    let work = l.fill(n * n, |_| T::default());
    MatMut::from_column_major_slice_mut(T::entity_slice_mut(work), n, n).copy_from(mat);
    faer::linalg::cholesky::llt::factor::cholesky_in_place(
        MatMut::from_column_major_slice_mut(T::entity_slice_mut(work), n, n),
        Default::default(),
        par,
        MemStack::new(mem),
        Default::default(),
    )
    .map_err(|_| Error::NonConvergence { op })?;
    // faer may leave scratch values in the strictly-upper triangle; clear it so the output is the
    // documented lower-triangular `L` with zeros above, as `push_masked` used to write.
    for col in 0..n {
        work[col * n..col * n + col].fill(T::default());
    }
    Ok(())
}

/// Cholesky factors of every `n x n` Hermitian positive-definite matrix of a batch.
///
/// `input` is `[n, n, b_1, ..., b_B]`; `l` receives `n * n` elements per item.
///
/// # Errors
///
/// [`Error::InvalidArgument`] when `input` is not a batch of square matrices, and
/// [`Error::NonConvergence`] for the lowest-indexed item that is not numerically positive definite
/// (the classification the pre-extraction code reported). `l` is empty on error.
///
/// # Examples
///
/// ```
/// use strided_view::RawStridedRef;
/// use tlinalg::{cholesky::cholesky, Op, Parallel};
///
/// let a = [4.0_f64, 2.0, 2.0, 3.0];
/// let mut l = Vec::new();
/// cholesky(
///     Op::Cholesky, RawStridedRef::new(&a, &[2, 2], &[1, 2], 0).unwrap(), &mut l,
///     Parallel::Sequential,
/// ).unwrap();
/// assert!((l[0] - 2.0).abs() < 1e-12 && l[2] == 0.0);
/// ```
pub fn cholesky<T: FaerScalar>(
    op: Op,
    input: RawStridedRef<'_, T>,
    l: &mut Vec<T>,
    par: Parallel<'_>,
) -> Result<()> {
    l.clear();
    let input = BatchedRef::square(op, "input", input)?;
    let n = input.rows();
    let l_len = checked_product(op, "L", &[n, n])?;
    batch::run(
        op,
        input.batch(),
        par,
        Some(n),
        &mut (out(l, l_len),),
        |par| {
            MemBuffer::new(
                faer::linalg::cholesky::llt::factor::cholesky_in_place_scratch::<T::Entity>(
                    n,
                    par,
                    Default::default(),
                ),
            )
        },
        |index, (l,), mem, par| cholesky_item::<T>(op, input.item(index), l, mem, par),
    )
}
