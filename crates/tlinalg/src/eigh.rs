//! faer-backed batched Hermitian (self-adjoint) eigendecomposition.
//!
//! Moved from `tenferro-linalg`'s faer backend (same project, MIT OR Apache-2.0).
//!
//! The decomposition reads the **lower** triangle and returns non-decreasing eigenvalues. As with
//! [`crate::svd`], [`eigh`] carries the (real) eigenvalues **in the scalar type itself** — complex
//! values with a zero imaginary part for the complex scalars — because that is what the
//! pre-extraction code produced; [`eigh_values`] returns them in the real type, matching the
//! values-only path.

use faer::diag::Diag;
use faer::dyn_stack::{MemBuffer, MemStack, StackReq};
use faer::linalg::evd::ComputeEigenvectors;
use faer::{MatMut, MatRef};
use strided_view::RawStridedRef;

use crate::batch::{self, out, BatchedRef};
use crate::scalar::ScalarEntity;
use crate::util::checked_product;
use crate::{Error, FaerScalar, Op, Parallel, Result};

/// Lane scratch for `n x n` Hermitian eigendecompositions.
struct EighScratch<E: faer::traits::ComplexField> {
    values: Diag<E>,
    /// Whether an item was already processed. `Diag::zeros` zeroes a fresh scratch once, so only a
    /// reused one needs the values reset before the next decomposition.
    reused: bool,
    mem: MemBuffer,
}

impl<E: faer::traits::ComplexField> EighScratch<E> {
    fn new(n: usize, vectors: ComputeEigenvectors, par: faer::Par) -> Self {
        // faer's eigensolvers do not accept an empty matrix; an empty item is never decomposed.
        let req = if n == 0 {
            StackReq::EMPTY
        } else {
            faer::linalg::evd::self_adjoint_evd_scratch::<E>(n, vectors, par, Default::default())
        };
        Self {
            values: Diag::zeros(n),
            reused: false,
            mem: MemBuffer::new(req),
        }
    }
}

/// Decompose one matrix into the lane scratch.
fn decompose<E: faer::traits::ComplexField>(
    op: Op,
    mat: MatRef<'_, E>,
    vectors: Option<MatMut<'_, E>>,
    scratch: &mut EighScratch<E>,
    par: faer::Par,
) -> Result<()> {
    // A fresh scratch is already zeroed by `Diag::zeros`; only a reused one needs the reset.
    if scratch.reused {
        scratch.values.as_mut().fill(E::zero_impl());
    }
    scratch.reused = true;
    faer::linalg::evd::self_adjoint_evd(
        mat,
        scratch.values.as_mut(),
        vectors,
        par,
        MemStack::new(&mut scratch.mem),
        Default::default(),
    )
    .map_err(|_| Error::NonConvergence { op })
}

/// Eigenvalues of every `n x n` Hermitian matrix of a batch, without the vectors.
///
/// `input` is `[n, n, b...]`; `values` receives `n` non-decreasing real eigenvalues per item.
///
/// # Errors
///
/// [`Error::InvalidArgument`] when `input` is not a batch of square matrices, and
/// [`Error::NonConvergence`] for the lowest-indexed item faer fails to converge on. `values` is
/// empty on error.
///
/// # Examples
///
/// ```
/// use strided_view::RawStridedRef;
/// use tlinalg::{eigh::eigh_values, Op, Parallel};
///
/// let a = [2.0_f64, 0.0, 0.0, 1.0];
/// let mut w = Vec::new();
/// eigh_values(
///     Op::EighValues, RawStridedRef::new(&a, &[2, 2], &[1, 2], 0).unwrap(), &mut w,
///     Parallel::Sequential,
/// ).unwrap();
/// assert_eq!(w, [1.0, 2.0]);
/// ```
pub fn eigh_values<T: FaerScalar>(
    op: Op,
    input: RawStridedRef<'_, T>,
    values: &mut Vec<<T as ScalarEntity>::Real>,
    par: Parallel<'_>,
) -> Result<()> {
    values.clear();
    let input = BatchedRef::square(op, "input", input)?;
    let n = input.rows();
    batch::run(
        op,
        input.batch(),
        par,
        Some(n),
        &mut (out(values, n),),
        |par| EighScratch::<T::Entity>::new(n, ComputeEigenvectors::No, par),
        |index, (values,), scratch, par| {
            if n == 0 {
                return Ok(());
            }
            decompose(op, input.item(index), None, scratch, par)?;
            for i in 0..n {
                values.push(T::real_from_entity(scratch.values[i]));
            }
            Ok(())
        },
    )
}

/// Eigendecomposition of every `n x n` Hermitian matrix of a batch, `A = V diag(w) Vᴴ`.
///
/// `input` is `[n, n, b...]`. Per item, `values` receives the `n` non-decreasing eigenvalues in the
/// scalar type (zero imaginary part for the complex scalars) and `vectors` the column-major `n x n`
/// eigenvectors.
///
/// # Errors
///
/// [`Error::InvalidArgument`] when `input` is not a batch of square matrices or the vector count
/// overflows, and [`Error::NonConvergence`] for the lowest-indexed item faer fails to converge on.
/// The outputs are empty on error.
///
/// # Examples
///
/// ```
/// use strided_view::RawStridedRef;
/// use tlinalg::{eigh::eigh, Op, Parallel};
///
/// let a = [2.0_f64, 0.0, 0.0, 1.0];
/// let (mut w, mut v) = (Vec::new(), Vec::new());
/// eigh(
///     Op::Eigh, RawStridedRef::new(&a, &[2, 2], &[1, 2], 0).unwrap(), &mut w, &mut v,
///     Parallel::Sequential,
/// ).unwrap();
/// assert_eq!(w, [1.0, 2.0]);
/// assert_eq!(v.len(), 4);
/// ```
pub fn eigh<T: FaerScalar>(
    op: Op,
    input: RawStridedRef<'_, T>,
    values: &mut Vec<T>,
    vectors: &mut Vec<T>,
    par: Parallel<'_>,
) -> Result<()> {
    values.clear();
    vectors.clear();
    let input = BatchedRef::square(op, "input", input)?;
    let n = input.rows();
    let v_len = checked_product(op, "eigenvector matrix", &[n, n])?;
    batch::run(
        op,
        input.batch(),
        par,
        Some(n),
        &mut (out(values, n), out(vectors, v_len)),
        |par| EighScratch::<T::Entity>::new(n, ComputeEigenvectors::Yes, par),
        |index, (values, vectors), scratch, par| {
            if n == 0 {
                return Ok(());
            }
            // Initialize the caller's eigenvector output and hand it to faer directly, instead of
            // decomposing into lane scratch and copying the whole matrix out afterwards.
            let region = vectors.fill(v_len, |_| T::default());
            let vmat = MatMut::from_column_major_slice_mut(T::entity_slice_mut(region), n, n);
            decompose(op, input.item(index), Some(vmat), scratch, par)?;
            for i in 0..n {
                values.push(T::from_real(T::real_from_entity(scratch.values[i])));
            }
            Ok(())
        },
    )
}
