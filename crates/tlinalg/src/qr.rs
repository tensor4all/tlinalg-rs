//! faer-backed batched thin QR and column-pivoted (rank-revealing) QR.
//!
//! Moved from `tenferro-linalg`'s faer backend (same project, MIT OR Apache-2.0).
//!
//! Both return the **raw** faer factors: no sign/phase gauge is applied, matching the
//! pre-extraction `qr` and `rank_revealing_qr` paths (the positive-diagonal gauge of the compact
//! Householder family is applied by the host when it extracts `Q` and `R`). For the rank-revealing
//! variant the rank decision (tolerances) and the non-finite screen stay in the host; [`magnitude`]
//! is the diagonal measure it was computed with. An all-zero item is not factored: it gets the
//! leading identity columns as `Q`, a zero `R` and the identity permutation, exactly as
//! `tlinalg-blas` and the pre-extraction host did.

use faer::dyn_stack::{MemBuffer, MemStack};
use faer::{Conj, Mat, MatMut, MatRef};
use strided_view::RawStridedRef;

use crate::batch::{self, out, BatchedRef, Push, Sink};
use crate::util::{checked_product, invalid, push_masked, weighted_muladds};
use crate::{FaerScalar, Op, Result};
use tcontract_exec::Exec;

/// Lane scratch for `m x n` QR factorizations, plain or column-pivoted.
struct QrScratch<E: faer::traits::ComplexField> {
    work: Mat<E>,
    coeff: Mat<E>,
    permutation: Vec<usize>,
    inverse_permutation: Vec<usize>,
    factor_mem: MemBuffer,
    apply_mem: MemBuffer,
}

impl<E: faer::traits::ComplexField> QrScratch<E> {
    fn new(m: usize, n: usize, pivoting: bool, par: faer::Par) -> Self {
        let k = m.min(n);
        let block_size = faer::linalg::qr::no_pivoting::factor::recommended_block_size::<E>(m, n);
        let factor_req = if pivoting {
            faer::linalg::qr::col_pivoting::factor::qr_in_place_scratch::<usize, E>(
                m,
                n,
                block_size,
                par,
                Default::default(),
            )
        } else {
            faer::linalg::qr::no_pivoting::factor::qr_in_place_scratch::<E>(
                m,
                n,
                block_size,
                par,
                Default::default(),
            )
        };
        Self {
            work: Mat::zeros(m, n),
            coeff: Mat::zeros(block_size, k),
            // On 64-bit targets the permutation is written straight into the `i64` output, so only
            // the inverse needs lane storage.
            permutation: vec![0; if pivoting && !PERMUTATION_IN_OUTPUT { n } else { 0 }],
            inverse_permutation: vec![0; if pivoting { n } else { 0 }],
            factor_mem: MemBuffer::new(factor_req),
            apply_mem: MemBuffer::new(
                faer::linalg::householder::apply_block_householder_sequence_on_the_left_in_place_scratch::<E>(
                    m, block_size, k,
                ),
            ),
        }
    }

    /// Write the thin `Q` from the reflectors in `work`/`coeff` into the caller's output.
    fn form_q<T: FaerScalar<Entity = E>>(&mut self, q: &mut Sink<'_, T>, par: faer::Par) {
        let (m, k) = (self.work.nrows(), self.work.nrows().min(self.work.ncols()));
        // `Q = H_1 ... H_k I`: the output region starts as the thin identity.
        let region = T::entity_slice_mut(q.fill(m * k, |index| {
            if index % m == index / m {
                T::parity(false)
            } else {
                T::default()
            }
        }));
        faer::linalg::householder::apply_block_householder_sequence_on_the_left_in_place_with_conj(
            self.work.as_ref().subcols(0, k),
            self.coeff.as_ref(),
            Conj::No,
            MatMut::from_column_major_slice_mut(region, m, k),
            par,
            MemStack::new(&mut self.apply_mem),
        );
    }
}

fn push_factors<T: FaerScalar>(scratch: &QrScratch<T::Entity>, r: &mut impl Push<T>) {
    let k = scratch.work.nrows().min(scratch.work.ncols());
    push_masked(
        r,
        scratch.work.as_ref(),
        k,
        scratch.work.ncols(),
        |row, col| row <= col,
    );
}

fn qr_item<T: FaerScalar>(
    mat: MatRef<'_, T::Entity>,
    q: &mut Sink<'_, T>,
    r: &mut impl Push<T>,
    scratch: &mut QrScratch<T::Entity>,
    par: faer::Par,
) {
    scratch.work.copy_from(mat);
    scratch
        .coeff
        .fill(<T::Entity as faer::traits::ComplexField>::zero_impl());
    faer::linalg::qr::no_pivoting::factor::qr_in_place(
        scratch.work.as_mut(),
        scratch.coeff.as_mut(),
        par,
        MemStack::new(&mut scratch.factor_mem),
        Default::default(),
    );
    scratch.form_q(q, par);
    push_factors::<T>(scratch, r);
}

/// Whether faer's `usize` permutation can be written straight into the `i64` output: true where
/// the two types share size and alignment (64-bit targets). Elsewhere a lane buffer is converted.
const PERMUTATION_IN_OUTPUT: bool = core::mem::size_of::<usize>() == core::mem::size_of::<i64>()
    && core::mem::align_of::<usize>() == core::mem::align_of::<i64>();

fn rank_revealing_qr_item<T: FaerScalar>(
    op: Op,
    mat: MatRef<'_, T::Entity>,
    (q, r, permutation): (&mut Sink<'_, T>, &mut impl Push<T>, &mut Sink<'_, i64>),
    scratch: &mut QrScratch<T::Entity>,
    par: faer::Par,
) -> Result<()> {
    let (m, n) = (mat.nrows(), mat.ncols());
    let k = m.min(n);
    let all_zero = (0..n).all(|col| (0..m).all(|row| T::entity_magnitude(mat[(row, col)]) == 0.0));
    if all_zero {
        // Rank zero by the host's rule: the leading identity columns, a zero `R`, no pivoting.
        for col in 0..k {
            for row in 0..m {
                q.push(if row == col {
                    T::parity(false)
                } else {
                    T::default()
                });
            }
        }
        for _ in 0..k * n {
            r.push(T::default());
        }
        permutation.fill(n, |column| column as i64);
        return Ok(());
    }
    scratch.work.copy_from(mat);
    scratch
        .coeff
        .fill(<T::Entity as faer::traits::ComplexField>::zero_impl());
    let region = permutation.fill(n, |_| 0);
    let forward: &mut [usize] = if PERMUTATION_IN_OUTPUT {
        // SAFETY: `PERMUTATION_IN_OUTPUT` holds only when `usize` and `i64` have the same size and
        // alignment, and every bit pattern is valid for both, so the initialised `i64` region is a
        // valid `[usize]` of the same length for the borrow. faer writes column indices below
        // `n <= isize::MAX`, which read back as the same non-negative `i64` values.
        unsafe {
            core::slice::from_raw_parts_mut(region.as_mut_ptr().cast::<usize>(), region.len())
        }
    } else {
        &mut scratch.permutation
    };
    // Following faer 0.24's public column-pivoted QR factor API; the provider's forward
    // permutation is the public gather convention.
    faer::linalg::qr::col_pivoting::factor::qr_in_place(
        scratch.work.as_mut(),
        scratch.coeff.as_mut(),
        forward,
        &mut scratch.inverse_permutation,
        par,
        MemStack::new(&mut scratch.factor_mem),
        Default::default(),
    );
    if !PERMUTATION_IN_OUTPUT {
        for (slot, &column) in region.iter_mut().zip(&scratch.permutation) {
            *slot = i64::try_from(column).map_err(|_| {
                invalid(op, "configuration", "column permutation exceeds i64 range")
            })?;
        }
    }
    scratch.form_q(q, par);
    push_factors::<T>(scratch, r);
    Ok(())
}

/// Thin QR of every `m x n` matrix of a batch, `A = Q R`.
///
/// `input` is `[m, n, b...]`. Per item, `q` receives the column-major `m x min(m, n)` factor with
/// orthonormal columns and `r` the column-major upper-trapezoidal `min(m, n) x n` factor.
///
/// # Errors
///
/// [`Error::InvalidArgument`](crate::Error::InvalidArgument) when `input` has rank below 2 or an
/// output size overflows. The outputs are empty on error.
///
/// # Examples
///
/// ```
/// use strided_view::RawStridedRef;
/// use tcontract_exec::Exec;
/// use tlinalg::{qr::qr, Op};
///
/// let a = [3.0_f64, 4.0];
/// let (mut q, mut r) = (Vec::new(), Vec::new());
/// qr(
///     Op::Qr, RawStridedRef::new(&a, &[2, 1], &[1, 2], 0).unwrap(), &mut q, &mut r,
///     Exec::serial(),
/// ).unwrap();
/// assert!((r[0].abs() - 5.0).abs() < 1e-12);
/// ```
pub fn qr<T: FaerScalar>(
    op: Op,
    input: RawStridedRef<'_, T>,
    q: &mut Vec<T>,
    r: &mut Vec<T>,
    exec: Exec<'_>,
) -> Result<()> {
    q.clear();
    r.clear();
    let input = BatchedRef::new(op, "input", input)?;
    let (m, n) = (input.rows(), input.cols());
    let k = m.min(n);
    let q_len = checked_product(op, "Q", &[m, k])?;
    let r_len = checked_product(op, "R", &[k, n])?;
    let item_muladds =
        weighted_muladds::<T>(m as f64 * n as f64 * k as f64 - (k as f64).powi(3) / 3.0);
    batch::run(
        op,
        input.batch(),
        exec,
        Some(m.max(n)),
        item_muladds,
        &mut (out(q, q_len), out(r, r_len)),
        |par| QrScratch::<T::Entity>::new(m, n, false, par),
        |index, (q, r), scratch, par| {
            qr_item::<T>(input.item(index), q, r, scratch, par);
            Ok(())
        },
    )
}

/// The magnitude a rank decision compares against its tolerance: `|x|` for the real scalars and
/// `hypot(re, im)` evaluated in `f64` for the complex ones.
///
/// A host reading the diagonal of the `R` returned by [`rank_revealing_qr`] gets exactly the values
/// the pre-extraction rank decision used.
///
/// # Examples
///
/// ```
/// use num_complex::Complex64;
/// assert_eq!(tlinalg::qr::magnitude(Complex64::new(3.0, 4.0)), 5.0);
/// assert_eq!(tlinalg::qr::magnitude(-2.0_f32), 2.0);
/// ```
#[must_use]
pub fn magnitude<T: FaerScalar>(value: T) -> f64 {
    T::entity_magnitude(T::entity_slice(core::slice::from_ref(&value))[0])
}

/// Column-pivoted QR of every `m x n` matrix of a batch, `A P = Q R`.
///
/// `input` is `[m, n, b...]`. Per item, `q` (`m x min(m, n)`) and `r` (`min(m, n) x n`, upper
/// trapezoidal) receive column-major factors, and `permutation` receives `n` column indices in the
/// public gather convention: column `j` of `A P` is column `permutation[j]` of `A`. An all-zero
/// item is not factored: it gets the leading `min(m, n)` identity columns as `Q`, a zero `R` and
/// the identity permutation (its rank is zero by the host's rule). Non-finite input is the host's
/// to screen.
///
/// # Errors
///
/// [`Error::InvalidArgument`](crate::Error::InvalidArgument) when `input` has rank below 2, an
/// output size overflows, or a column index does not fit `i64`. The outputs are empty on error.
///
/// # Examples
///
/// ```
/// use strided_view::RawStridedRef;
/// use tcontract_exec::Exec;
/// use tlinalg::{qr::rank_revealing_qr, Op};
///
/// let a = [1.0_f64, 0.0, 0.0, 5.0];
/// let (mut q, mut r, mut perm) = (Vec::new(), Vec::new(), Vec::new());
/// rank_revealing_qr(
///     Op::RankRevealingQr, RawStridedRef::new(&a, &[2, 2], &[1, 2], 0).unwrap(),
///     &mut q, &mut r, &mut perm, Exec::serial(),
/// ).unwrap();
/// assert_eq!(perm, [1, 0]);
/// ```
pub fn rank_revealing_qr<T: FaerScalar>(
    op: Op,
    input: RawStridedRef<'_, T>,
    q: &mut Vec<T>,
    r: &mut Vec<T>,
    permutation: &mut Vec<i64>,
    exec: Exec<'_>,
) -> Result<()> {
    q.clear();
    r.clear();
    permutation.clear();
    let input = BatchedRef::new(op, "input", input)?;
    let (m, n) = (input.rows(), input.cols());
    let k = m.min(n);
    let q_len = checked_product(op, "Q", &[m, k])?;
    let r_len = checked_product(op, "R", &[k, n])?;
    let item_muladds =
        weighted_muladds::<T>(m as f64 * n as f64 * k as f64 - (k as f64).powi(3) / 3.0);
    batch::run(
        op,
        input.batch(),
        exec,
        Some(m.max(n)),
        item_muladds,
        &mut (out(q, q_len), out(r, r_len), out(permutation, n)),
        |par| QrScratch::<T::Entity>::new(m, n, true, par),
        |index, (q, r, permutation), scratch, par| {
            rank_revealing_qr_item::<T>(op, input.item(index), (q, r, permutation), scratch, par)
        },
    )
}
