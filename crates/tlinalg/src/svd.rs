//! faer-backed batched singular value decomposition.
//!
//! Moved from `tenferro-linalg`'s faer backend (same project, MIT OR Apache-2.0).
//!
//! # Boundary
//!
//! The input is a borrowed rank-`2 + B` [`RawStridedRef`] `[m, n, b_1, ..., b_B]`; the outputs are
//! caller-provided vectors, **cleared and then filled** with compact column-major items in batch
//! order (see `docs/design/batched-api.md`). Faer's `Mat`/`Diag`/`MemBuffer` storage is lane
//! scratch, sized once per lane and reused for every item of the lane.
//!
//! # Conventions
//!
//! Per item, `u` is `m x u_cols` column-major, `vt` is `v_cols x n` column-major and holds `Vᴴ`
//! (not `V`), and `s` holds `min(m, n)` singular values in non-increasing order **in the scalar type
//! itself**. For the complex scalars that means complex values with a zero imaginary part, which is
//! what the pre-extraction code produced and what its callers' plumbing expects; [`svd_values`]
//! returns the real values instead, matching the values-only path. `full` selects the square
//! unitary factors `u_cols = m`, `v_cols = n`; otherwise `u_cols = v_cols = min(m, n)`. A full SVD
//! of an empty matrix returns identity `U` and `Vᴴ`.
//!
//! # Accuracy
//!
//! faer's default parameters use a divide-and-conquer bidiagonal SVD once the smaller dimension
//! reaches 128, and in faer 0.24.4 that path can return inaccurate factors without an error. A
//! decomposition with vectors of that size is therefore checked against its input, by the
//! Frobenius norm of `A - U diag(S) Vᴴ`, and repeated with the QR algorithm when the check fails
//! or faer reports an error; [`svd_values`] always uses the QR algorithm. `docs/worklogs/svd-recursion-threshold.md` records the cost.

use faer::diag::Diag;
use faer::dyn_stack::{MemBuffer, MemStack};
use faer::linalg::svd::{ComputeSvdVectors, SvdParams};
use faer::{Mat, MatMut, MatRef};

use strided_view::RawStridedRef;

use crate::batch::{self, out, BatchedRef, Push, Sink};
use crate::scalar::ScalarEntity;
use crate::util::{checked_product, push_identity};
use crate::{Error, FaerScalar, Op, Parallel, Result};

/// faer's parameters with the QR algorithm at every size.
///
/// faer's defaults switch from the QR algorithm to a divide-and-conquer bidiagonal SVD once the
/// smaller dimension reaches `recursion_threshold` (128). In faer 0.24.4 that path returns a
/// spurious singular value and inaccurate factors for some rank-deficient matrices with clustered
/// singular values, without reporting an error (see the test
/// `clustered_singular_values_of_a_rank_deficient_matrix`). These parameters never take it.
fn qr_params<E: faer::traits::ComplexField>() -> faer::Spec<SvdParams, E> {
    faer::Spec::new(SvdParams {
        recursion_threshold: usize::MAX,
        ..<SvdParams as faer::Auto<E>>::auto()
    })
}

/// Whether faer's default parameters use divide and conquer for a smaller dimension of `k`.
fn divides<E: faer::traits::ComplexField>(k: usize) -> bool {
    k >= <SvdParams as faer::Auto<E>>::auto().recursion_threshold
}

/// Constant of [`residual_bound`]. Accurate decompositions of sizes up to 1024 in all four scalar
/// types have relative residuals of 10 to 60 machine epsilons, at most a quarter of the bound at
/// size 128; the inaccurate ones observed exceed it by a factor of 1e8.
const RESIDUAL_FACTOR: f64 = 24.0;

/// Largest accepted relative Frobenius residual `‖A - U diag(S) Vᴴ‖ / ‖A‖` of an `m x n`
/// decomposition: `RESIDUAL_FACTOR * sqrt(max(m, n)) * epsilon`.
///
/// A singular value missing from an `n x n` identity leaves a relative residual of `1 / sqrt(n)`,
/// which this bound rejects for `n < 1 / (RESIDUAL_FACTOR * epsilon)`: about 350 000 in single
/// precision.
fn residual_bound(m: usize, n: usize, epsilon: f64) -> f64 {
    RESIDUAL_FACTOR * (m.max(n) as f64).sqrt() * epsilon
}

/// Storage for checking a decomposition with vectors against its input.
struct Check<E: faer::traits::ComplexField> {
    /// `U` with its columns scaled by the singular values, `m x k`.
    scaled: Mat<E>,
    /// The reconstruction and then its difference from the input, `m x n`.
    residual: Mat<E>,
}

impl<E: faer::traits::ComplexField> Check<E> {
    fn new(m: usize, n: usize) -> Self {
        Self {
            scaled: Mat::zeros(m, m.min(n)),
            residual: Mat::zeros(m, n),
        }
    }
}

/// One lane's faer storage for `m x n` decompositions.
struct SvdScratch<E: faer::traits::ComplexField> {
    v: Mat<E>,
    s: Diag<E>,
    mem: MemBuffer,
    /// Whether an item was already processed. `Diag::zeros` zeroes a fresh scratch once, so only a
    /// reused one needs the singular values reset before the next decomposition.
    reused: bool,
    /// Present when vectors are computed and the default parameters use divide and conquer.
    check: Option<Check<E>>,
}

impl<E: faer::traits::ComplexField> SvdScratch<E> {
    fn new(m: usize, n: usize, vectors: ComputeSvdVectors, par: faer::Par) -> Self {
        let k = m.min(n);
        let v_cols = match vectors {
            ComputeSvdVectors::Full => n,
            ComputeSvdVectors::Thin => k,
            ComputeSvdVectors::No => 0,
        };
        let checked = !matches!(vectors, ComputeSvdVectors::No) && divides::<E>(k);
        // An empty matrix is never decomposed, so it needs no faer scratch. A checked
        // decomposition may run a second time with the QR parameters in the same buffer.
        let req = if k == 0 {
            faer::dyn_stack::StackReq::EMPTY
        } else {
            let qr = faer::linalg::svd::svd_scratch::<E>(m, n, vectors, vectors, par, qr_params());
            if checked {
                qr.or(faer::linalg::svd::svd_scratch::<E>(
                    m,
                    n,
                    vectors,
                    vectors,
                    par,
                    Default::default(),
                ))
            } else {
                qr
            }
        };
        Self {
            v: Mat::zeros(n, v_cols),
            s: Diag::zeros(k),
            mem: MemBuffer::new(req),
            reused: false,
            check: checked.then(|| Check::new(m, n)),
        }
    }
}

/// Whether the factors reproduce `mat`: the Frobenius norm of `mat - U diag(S) Vᴴ` is at most
/// [`residual_bound`] times that of `mat`. `u` and `v` hold at least `min(m, n)` columns. A
/// residual that is not finite is rejected.
///
/// The test is on the whole residual, so it holds for every input: factors that pass reproduce
/// the matrix to the bound. It costs one matrix product.
fn reproduces<E: faer::traits::ComplexField>(
    mat: MatRef<'_, E>,
    u: MatRef<'_, E>,
    s: faer::diag::DiagRef<'_, E>,
    v: MatRef<'_, E>,
    check: &mut Check<E>,
    epsilon: f64,
    par: faer::Par,
) -> bool {
    use faer::traits::math_utils::{from_f64, mul, neg, one};

    let (m, n) = (mat.nrows(), mat.ncols());
    let k = m.min(n);
    for col in 0..k {
        let value = &s[col];
        for row in 0..m {
            check.scaled[(row, col)] = mul(&u[(row, col)], value);
        }
    }
    check.residual.as_mut().copy_from(mat);
    faer::linalg::matmul::matmul(
        check.residual.as_mut(),
        faer::Accum::Add,
        check.scaled.as_ref(),
        v.get(.., ..k).adjoint(),
        neg(&one::<E>()),
        par,
    );
    let bound = from_f64::<E::Real>(residual_bound(m, n, epsilon));
    // A NaN residual fails this comparison and is rejected with the large ones.
    check.residual.norm_l2() <= mul(&bound, &mat.norm_l2())
}

/// Singular values of one matrix, pushed in the real type.
fn svd_values_item<T: FaerScalar>(
    op: Op,
    mat: MatRef<'_, T::Entity>,
    s: &mut impl Push<<T as ScalarEntity>::Real>,
    scratch: &mut SvdScratch<T::Entity>,
    par: faer::Par,
) -> Result<()> {
    let k = mat.nrows().min(mat.ncols());
    if k == 0 {
        return Ok(());
    }
    if scratch.reused {
        scratch
            .s
            .as_mut()
            .fill(<T::Entity as faer::traits::ComplexField>::zero_impl());
    }
    scratch.reused = true;
    faer::linalg::svd::svd(
        mat,
        scratch.s.as_mut(),
        None,
        None,
        par,
        MemStack::new(&mut scratch.mem),
        // Without vectors there is no reconstruction to check, so the values never take the
        // divide-and-conquer path.
        qr_params(),
    )
    .map_err(|_| Error::NonConvergence { op })?;
    for index in 0..k {
        s.push(<T as ScalarEntity>::real_from_entity(scratch.s[index]));
    }
    Ok(())
}

/// SVD of one matrix, pushed as `U`, `S` and `Vᴴ`.
fn svd_item<T: FaerScalar>(
    op: Op,
    mat: MatRef<'_, T::Entity>,
    full: bool,
    (u, s, vt): (&mut Sink<'_, T>, &mut impl Push<T>, &mut impl Push<T>),
    scratch: &mut SvdScratch<T::Entity>,
    par: faer::Par,
) -> Result<()> {
    let (m, n) = (mat.nrows(), mat.ncols());
    let k = m.min(n);
    if k == 0 {
        if full {
            push_identity(u, m);
            push_identity(vt, n);
        }
        return Ok(());
    }
    let (u_cols, v_cols) = if full { (m, n) } else { (k, k) };
    let zero = <T::Entity as faer::traits::ComplexField>::zero_impl();
    // A fresh scratch is already zeroed; only a reused one needs the reset, so reuse cannot leak a
    // previous item into this one.
    if scratch.reused {
        scratch.v.as_mut().fill(zero);
        scratch.s.as_mut().fill(zero);
    }
    scratch.reused = true;
    // Initialize the caller's `U` output and hand it to faer directly, instead of decomposing into
    // lane scratch and copying the whole factor out afterwards.
    let u_region = u.fill(m * u_cols, |_| T::default());
    let mut u_mat = MatMut::from_column_major_slice_mut(T::entity_slice_mut(u_region), m, u_cols);
    // The divide-and-conquer path of faer's default parameters can return inaccurate factors
    // without an error. A decomposition that can take it runs with the defaults first and is
    // repeated with the QR algorithm when faer reports an error or the factors do not reproduce
    // the input.
    let accurate = match scratch.check.as_mut() {
        None => false,
        Some(check) => {
            faer::linalg::svd::svd(
                mat,
                scratch.s.as_mut(),
                Some(u_mat.as_mut()),
                Some(scratch.v.as_mut()),
                par,
                MemStack::new(&mut scratch.mem),
                Default::default(),
            )
            .is_ok()
                && reproduces(
                    mat,
                    u_mat.as_ref(),
                    scratch.s.as_ref(),
                    scratch.v.as_ref(),
                    check,
                    <T as ScalarEntity>::EPSILON,
                    par,
                )
        }
    };
    if !accurate {
        if scratch.check.is_some() {
            #[cfg(test)]
            tests::record_repeat();
            // faer's tall path does not clear the top-right block of `U`, so the repeat must seed
            // it with the same zeros the first attempt saw.
            u_mat.as_mut().fill(zero);
            scratch.v.as_mut().fill(zero);
            scratch.s.as_mut().fill(zero);
        }
        faer::linalg::svd::svd(
            mat,
            scratch.s.as_mut(),
            Some(u_mat.as_mut()),
            Some(scratch.v.as_mut()),
            par,
            MemStack::new(&mut scratch.mem),
            qr_params(),
        )
        .map_err(|_| Error::NonConvergence { op })?;
    }

    // `U` was written straight into the caller's output. Push the `min(m, n)` real singular values
    // and the column-major `Vᴴ` in the order the previous implementation produced them.
    for index in 0..k {
        // A real singular value carried in the scalar type: zero imaginary part for the complex
        // scalars, which is the shape the pre-extraction callers consumed.
        s.push(T::from_entity(scratch.s[index]));
    }
    for col in 0..n {
        for row in 0..v_cols {
            // `V` is transposed into `Vᴴ`, so the complex scalars conjugate here. The real ones are
            // their own conjugate, which is why the two implementations share this call.
            vt.push(T::from_entity_conj(scratch.v[(col, row)]));
        }
    }
    Ok(())
}

/// Singular values of every matrix of a batch, without the vectors.
///
/// `input` is `[m, n, b_1, ..., b_B]`; `s` receives `min(m, n)` real values per item.
///
/// # Errors
///
/// [`Error::InvalidArgument`] when `input` has rank below 2, and [`Error::NonConvergence`] for the
/// lowest-indexed item faer fails to converge on. `s` is empty on error.
///
/// # Examples
///
/// ```
/// use strided_view::RawStridedRef;
/// use tlinalg::{svd::svd_values, Op, Parallel};
///
/// // Two 2x2 diagonal matrices, batch-contiguous.
/// let a = [3.0_f64, 0.0, 0.0, 1.0, 2.0, 0.0, 0.0, 5.0];
/// let mut s = Vec::new();
/// svd_values(
///     Op::SvdValues, RawStridedRef::new(&a, &[2, 2, 2], &[1, 2, 4], 0).unwrap(), &mut s,
///     Parallel::Sequential,
/// ).unwrap();
/// assert_eq!(s, [3.0, 1.0, 5.0, 2.0]);
/// ```
pub fn svd_values<T: FaerScalar>(
    op: Op,
    input: RawStridedRef<'_, T>,
    s: &mut Vec<<T as ScalarEntity>::Real>,
    par: Parallel<'_>,
) -> Result<()> {
    s.clear();
    let input = BatchedRef::new(op, "input", input)?;
    let (m, n) = (input.rows(), input.cols());
    batch::run(
        op,
        input.batch(),
        par,
        Some(m.max(n)),
        &mut (out(s, m.min(n)),),
        |par| SvdScratch::<T::Entity>::new(m, n, ComputeSvdVectors::No, par),
        |index, (s,), scratch, par| svd_values_item::<T>(op, input.item(index), s, scratch, par),
    )
}

/// Singular value decomposition of every matrix of a batch.
///
/// `input` is `[m, n, b_1, ..., b_B]`. Per item, `u` receives `m x u_cols`, `s` `min(m, n)` and
/// `vt` `v_cols x n` elements (see the module conventions).
///
/// # Errors
///
/// [`Error::InvalidArgument`] when `input` has rank below 2 or an output size overflows, and
/// [`Error::NonConvergence`] for the lowest-indexed item faer fails to converge on. The outputs are
/// empty on error.
///
/// # Examples
///
/// ```
/// use strided_view::RawStridedRef;
/// use tlinalg::{svd::svd, Op, Parallel};
///
/// let a = [3.0_f64, 0.0, 0.0, 1.0];
/// let (mut u, mut s, mut vt) = (Vec::new(), Vec::new(), Vec::new());
/// svd(
///     Op::Svd, RawStridedRef::new(&a, &[2, 2], &[1, 2], 0).unwrap(), false,
///     &mut u, &mut s, &mut vt, Parallel::Sequential,
/// ).unwrap();
/// assert_eq!(s, [3.0, 1.0]);
/// ```
// INVARIANT: descriptor, mode, three output buffers, token are distinct operands of one
// batched decomposition; grouping them would add a wrapper without removing an argument.
#[allow(clippy::too_many_arguments)]
pub fn svd<T: FaerScalar>(
    op: Op,
    input: RawStridedRef<'_, T>,
    full: bool,
    u: &mut Vec<T>,
    s: &mut Vec<T>,
    vt: &mut Vec<T>,
    par: Parallel<'_>,
) -> Result<()> {
    u.clear();
    s.clear();
    vt.clear();
    let input = BatchedRef::new(op, "input", input)?;
    let (m, n) = (input.rows(), input.cols());
    let k = m.min(n);
    let (u_cols, v_cols, vectors) = if full {
        (m, n, ComputeSvdVectors::Full)
    } else {
        (k, k, ComputeSvdVectors::Thin)
    };
    let u_len = checked_product(op, "U", &[m, u_cols])?;
    let vt_len = checked_product(op, "VH", &[v_cols, n])?;
    batch::run(
        op,
        input.batch(),
        par,
        Some(m.max(n)),
        &mut (out(u, u_len), out(s, k), out(vt, vt_len)),
        |par| SvdScratch::<T::Entity>::new(m, n, vectors, par),
        |index, (u, s, vt), scratch, par| {
            svd_item::<T>(op, input.item(index), full, (u, s, vt), scratch, par)
        },
    )
}

#[cfg(test)]
mod tests;
