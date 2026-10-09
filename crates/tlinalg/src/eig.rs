//! faer-backed general (non-Hermitian) eigendecomposition.
//!
//! Moved from `tenferro-linalg`'s faer backend (same project, MIT OR Apache-2.0). Faer's work
//! storage is lane scratch, reused for every item of the lane.
//!
//! # Conventions
//!
//! The outputs are always complex: `f32`/`Complex32` input yields `Complex32`, `f64`/`Complex64`
//! input yields `Complex64`. For real input, faer returns real Schur-form eigenpairs; they are
//! converted exactly as before:
//!
//! * [`eig`] treats an eigenvalue whose imaginary part is at most `ε · max(|re|, 1)` as real (zero
//!   imaginary part, real eigenvector); otherwise it emits the conjugate pair `re ± i·im` with the
//!   eigenvectors `u_j ± i·u_{j+1}`.
//! * [`eig_values`] copies the real and imaginary parts as faer returned them, without that test.

use faer::diag::Diag;
use faer::dyn_stack::{MemBuffer, MemStack, StackReq};
use faer::linalg::evd::ComputeEigenvectors;
use faer::{Mat, MatRef};
use num_complex::{Complex32, Complex64};
use strided_view::RawStridedRef;

use crate::batch::{self, out, BatchedRef, Push};
use crate::scalar::{EigScalar, ScalarEntity};
use crate::util::checked_product;
use crate::{Error, FaerScalar, Op, Parallel, Result};

/// Eigenvalues and eigenvectors of every `n x n` matrix of a batch, `A V = V diag(w)`.
///
/// `input` is `[n, n, b...]`. Per item, `values` receives `n` values and `vectors` the column-major
/// `n x n` eigenvectors, in the complex type of the input scalar.
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
/// use num_complex::Complex64;
/// use strided_view::RawStridedRef;
/// use tlinalg::{eig::eig, Op, Parallel};
///
/// // Rotation by 90 degrees: eigenvalues ±i.
/// let a = [0.0_f64, 1.0, -1.0, 0.0];
/// let (mut w, mut v) = (Vec::<Complex64>::new(), Vec::new());
/// eig(
///     Op::Eig, RawStridedRef::new(&a, &[2, 2], &[1, 2], 0).unwrap(), &mut w, &mut v,
///     Parallel::Sequential,
/// ).unwrap();
/// assert!((w[0].im.abs() - 1.0).abs() < 1e-12 && (w[0].conj() - w[1]).norm() < 1e-12);
/// ```
pub fn eig<T: FaerScalar>(
    op: Op,
    input: RawStridedRef<'_, T>,
    values: &mut Vec<<T as ScalarEntity>::Complex>,
    vectors: &mut Vec<<T as ScalarEntity>::Complex>,
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
        |par| T::eig_scratch(n, true, par),
        |index, (values, vectors), scratch, par| {
            if n == 0 {
                return Ok(());
            }
            T::eig_into(op, input.item(index), values, Some(vectors), scratch, par)
        },
    )
}

/// Eigenvalues of every `n x n` matrix of a batch, without the eigenvectors.
///
/// `input` is `[n, n, b...]`; `values` receives `n` values per item in the complex type of the
/// input scalar.
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
/// use num_complex::Complex64;
/// use strided_view::RawStridedRef;
/// use tlinalg::{eig::eig_values, Op, Parallel};
///
/// let a = [2.0_f64, 0.0, 0.0, 3.0];
/// let mut w = Vec::<Complex64>::new();
/// eig_values(
///     Op::EigValues, RawStridedRef::new(&a, &[2, 2], &[1, 2], 0).unwrap(), &mut w,
///     Parallel::Sequential,
/// ).unwrap();
/// assert_eq!(w.len(), 2);
/// ```
pub fn eig_values<T: FaerScalar>(
    op: Op,
    input: RawStridedRef<'_, T>,
    values: &mut Vec<<T as ScalarEntity>::Complex>,
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
        |par| T::eig_scratch(n, false, par),
        |index, (values,), scratch, par| {
            if n == 0 {
                return Ok(());
            }
            T::eig_into(op, input.item(index), values, None, scratch, par)
        },
    )
}

/// `|im| <= ε · max(|re|, 1)`, the test for treating a real-input eigenvalue as real.
fn imag_is_effectively_zero(real: f64, imag: f64, eps: f64) -> bool {
    imag.abs() <= eps * real.abs().max(1.0)
}

/// faer's eigensolver scratch for `n x n` (empty for `n == 0`, which is never decomposed).
fn evd_mem<E: faer::traits::ComplexField>(n: usize, vectors: bool, par: faer::Par) -> MemBuffer {
    let req = if n == 0 {
        StackReq::EMPTY
    } else {
        let right = if vectors {
            ComputeEigenvectors::Yes
        } else {
            ComputeEigenvectors::No
        };
        faer::linalg::evd::evd_scratch::<E>(
            n,
            ComputeEigenvectors::No,
            right,
            par,
            Default::default(),
        )
    };
    MemBuffer::new(req)
}

/// Lane storage types, public only so the crate-internal `EigScalar` can name them.
mod lane_storage {
    use faer::diag::Diag;
    use faer::dyn_stack::MemBuffer;
    use faer::Mat;

    /// Lane storage of the real eigensolver.
    pub struct RealEigScratch<R: faer::traits::RealField> {
        pub(super) u: Mat<R>,
        pub(super) s_re: Diag<R>,
        pub(super) s_im: Diag<R>,
        pub(super) mem: MemBuffer,
    }

    /// Lane storage of the complex eigensolver.
    pub struct ComplexEigScratch<E: faer::traits::ComplexField> {
        pub(super) u: Mat<E>,
        pub(super) s: Diag<E>,
        pub(super) mem: MemBuffer,
        /// Whether an earlier item has written `u` and `s`, so the next one has to be reset first.
        pub(super) reused: bool,
    }
}

use lane_storage::{ComplexEigScratch, RealEigScratch};

macro_rules! impl_eig_real {
    ($real:ty, $complex:ty) => {
        impl EigScalar for $real {
            type EigScratch = RealEigScratch<$real>;

            fn eig_scratch(n: usize, vectors: bool, par: faer::Par) -> Self::EigScratch {
                RealEigScratch {
                    u: Mat::zeros(n, if vectors { n } else { 0 }),
                    s_re: Diag::zeros(n),
                    s_im: Diag::zeros(n),
                    mem: evd_mem::<$real>(n, vectors, par),
                }
            }

            fn eig_into(
                op: Op,
                mat: MatRef<'_, $real>,
                values: &mut dyn Push<$complex>,
                vectors: Option<&mut dyn Push<$complex>>,
                scratch: &mut Self::EigScratch,
                par: faer::Par,
            ) -> Result<()> {
                let n = mat.nrows();
                let RealEigScratch { u, s_re, s_im, mem } = scratch;
                s_re.as_mut().fill(0.0);
                s_im.as_mut().fill(0.0);
                let u_out = if vectors.is_some() {
                    u.as_mut().fill(0.0);
                    Some(u.as_mut())
                } else {
                    None
                };
                faer::linalg::evd::evd_real(
                    mat,
                    s_re.as_mut(),
                    s_im.as_mut(),
                    None,
                    u_out,
                    par,
                    MemStack::new(mem),
                    Default::default(),
                )
                .map_err(|_| Error::NonConvergence { op })?;
                let Some(vectors) = vectors else {
                    // Values only: the real and imaginary parts as faer returned them.
                    for j in 0..n {
                        values.push(<$complex>::new(s_re[j], s_im[j]));
                    }
                    return Ok(());
                };
                let mut j = 0;
                while j < n {
                    if j + 1 >= n
                        || imag_is_effectively_zero(
                            s_re[j] as f64,
                            s_im[j] as f64,
                            <$real>::EPSILON as f64,
                        )
                    {
                        values.push(<$complex>::new(s_re[j], 0.0));
                        for i in 0..n {
                            vectors.push(<$complex>::new(u[(i, j)], 0.0));
                        }
                        j += 1;
                    } else {
                        values.push(<$complex>::new(s_re[j], s_im[j]));
                        values.push(<$complex>::new(s_re[j], -s_im[j]));
                        for i in 0..n {
                            vectors.push(<$complex>::new(u[(i, j)], u[(i, j + 1)]));
                        }
                        for i in 0..n {
                            vectors.push(<$complex>::new(u[(i, j)], -u[(i, j + 1)]));
                        }
                        j += 2;
                    }
                }
                Ok(())
            }
        }
    };
}

macro_rules! impl_eig_complex {
    ($complex:ty, $entity:ty) => {
        impl EigScalar for $complex {
            type EigScratch = ComplexEigScratch<$entity>;

            fn eig_scratch(n: usize, vectors: bool, par: faer::Par) -> Self::EigScratch {
                ComplexEigScratch {
                    u: Mat::zeros(n, if vectors { n } else { 0 }),
                    s: Diag::zeros(n),
                    mem: evd_mem::<$entity>(n, vectors, par),
                    reused: false,
                }
            }

            fn eig_into(
                op: Op,
                mat: MatRef<'_, $entity>,
                values: &mut dyn Push<$complex>,
                vectors: Option<&mut dyn Push<$complex>>,
                scratch: &mut Self::EigScratch,
                par: faer::Par,
            ) -> Result<()> {
                let n = mat.nrows();
                let zero = <$entity>::new(0.0, 0.0);
                let ComplexEigScratch { u, s, mem, reused } = scratch;
                // `Diag::zeros`/`Mat::zeros` already zeroed these; only a scratch that an earlier
                // item wrote needs resetting.
                if *reused {
                    s.as_mut().fill(zero);
                    if vectors.is_some() {
                        u.as_mut().fill(zero);
                    }
                }
                *reused = true;
                let u_out = if vectors.is_some() {
                    Some(u.as_mut())
                } else {
                    None
                };
                faer::linalg::evd::evd_cplx(
                    mat,
                    s.as_mut(),
                    None,
                    u_out,
                    par,
                    MemStack::new(mem),
                    Default::default(),
                )
                .map_err(|_| Error::NonConvergence { op })?;
                for j in 0..n {
                    values.push(<$complex>::new(s[j].re, s[j].im));
                }
                if let Some(vectors) = vectors {
                    for col in 0..n {
                        for row in 0..n {
                            let value = u[(row, col)];
                            vectors.push(<$complex>::new(value.re, value.im));
                        }
                    }
                }
                Ok(())
            }
        }
    };
}

impl_eig_real!(f32, Complex32);
impl_eig_real!(f64, Complex64);
impl_eig_complex!(Complex32, faer::c32);
impl_eig_complex!(Complex64, faer::c64);
