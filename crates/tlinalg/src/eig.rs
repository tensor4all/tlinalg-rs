//! faer-backed general (non-Hermitian) eigendecomposition.
//!
//! Moved from `tenferro-linalg`'s faer backend (same project, MIT OR Apache-2.0). Faer's work
//! storage is per-lane scratch carved from the caller's borrowed workspace, reused for every item of
//! the lane.
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

use core::mem::MaybeUninit;

use faer::diag::DiagMut;
use faer::dyn_stack::{MemStack, StackReq};
use faer::linalg::evd::ComputeEigenvectors;
use faer::{MatMut, MatRef};
use num_complex::{Complex32, Complex64};
use strided_view::RawStridedRef;

use crate::batch::{self, fixed_out, BatchedRef, Push, VecOut};
use crate::scalar::{EigScalar, ScalarEntity};
use crate::util::{checked_product, weighted_muladds};
use crate::workspace::{bytes, faer_req, per_lane_bytes, total_bytes, zeros};
use crate::{Error, FaerScalar, Op, Result};
use tcontract_exec::Exec;

/// Bytes one batched eigendecomposition call needs in its borrowed workspace.
///
/// `vectors` selects the route: `false` sizes the values-only decomposition. The canonical `Op` of
/// a refusal is `Op::Eig` (`Op::EigValues` for the values-only query).
///
/// # Errors
///
/// [`Error::InvalidArgument`] when a size overflows, and the
/// executor's refusal for a forced strategy this route cannot honor.
pub fn eig_workspace<T: FaerScalar>(
    n: usize,
    vectors: bool,
    batch: usize,
    exec: Exec<'_>,
) -> Result<usize> {
    workspace_bytes::<T>(n, vectors, batch, exec)
}

/// [`eig_workspace`] for the values-only route (the canonical `Op` is `Op::EigValues`).
///
/// # Errors
///
/// As [`eig_workspace`].
pub fn eig_values_workspace<T: FaerScalar>(
    n: usize,
    batch: usize,
    exec: Exec<'_>,
) -> Result<usize> {
    workspace_bytes::<T>(n, false, batch, exec)
}

fn workspace_bytes<T: FaerScalar>(
    n: usize,
    vectors: bool,
    batch: usize,
    exec: Exec<'_>,
) -> Result<usize> {
    let op = if vectors { Op::Eig } else { Op::EigValues };
    let item_muladds = weighted_muladds::<T>((n as f64).powi(3));
    let (lanes, item_threads) = batch::decide(op, exec, batch, Some(n), item_muladds)?;
    let par = batch::route_par(lanes, item_threads);
    total_bytes(op, T::eig_req(n, vectors, par), lanes, batch)
}

/// Eigenvalues and eigenvectors of every `n x n` matrix of a batch, into caller-owned storage.
///
/// `values` holds `n` and `vectors` `n * n` uninitialised elements per batch item, in the complex
/// type of the input scalar. Workspace contract as [`crate::qr::qr_into`].
///
/// # Errors
///
/// [`Error::InvalidArgument`] when `input` is not a batch of square matrices, an output length does
/// not match, or the vector count overflows, [`crate::Error::InsufficientWorkspace`] for a short
/// workspace, and [`Error::NonConvergence`] for the lowest-indexed item faer fails to converge on.
pub fn eig_into<T: FaerScalar>(
    op: Op,
    input: RawStridedRef<'_, T>,
    values: &mut [MaybeUninit<<T as ScalarEntity>::Complex>],
    vectors: &mut [MaybeUninit<<T as ScalarEntity>::Complex>],
    workspace: &mut [MaybeUninit<u8>],
    exec: Exec<'_>,
) -> Result<()> {
    let input = BatchedRef::square(op, "input", input)?;
    let n = input.rows();
    let v_len = checked_product(op, "eigenvector matrix", &[n, n])?;
    let item_muladds = weighted_muladds::<T>((n as f64).powi(3));
    let (lanes, item_threads) = batch::decide(op, exec, input.batch(), Some(n), item_muladds)?;
    let par = batch::route_par(lanes, item_threads);
    let per_lane = per_lane_bytes(T::eig_req(n, true, par), input.batch());
    batch::check_workspace(op, workspace, per_lane, lanes)?;
    batch::run(
        op,
        input.batch(),
        exec,
        Some(n),
        item_muladds,
        &mut (
            fixed_out(values, n, "eigenvalues"),
            fixed_out(vectors, v_len, "eigenvectors"),
        ),
        batch::ScratchArena::new(workspace, per_lane),
        |par, stack| T::eig_scratch(n, true, par, MemStack::new(stack)),
        |index, (values, vectors), scratch, par| {
            if n == 0 {
                return Ok(());
            }
            T::eig_into(op, input.item(index), values, Some(vectors), scratch, par)
        },
    )
}

/// Eigenvalues of every `n x n` matrix of a batch, into caller-owned storage.
///
/// `values` holds `n` uninitialised complex elements per batch item. Workspace contract as
/// [`crate::qr::qr_into`].
///
/// # Errors
///
/// As [`eig_into`], without the eigenvector count.
pub fn eig_values_into<T: FaerScalar>(
    op: Op,
    input: RawStridedRef<'_, T>,
    values: &mut [MaybeUninit<<T as ScalarEntity>::Complex>],
    workspace: &mut [MaybeUninit<u8>],
    exec: Exec<'_>,
) -> Result<()> {
    let input = BatchedRef::square(op, "input", input)?;
    let n = input.rows();
    let item_muladds = weighted_muladds::<T>((n as f64).powi(3));
    let (lanes, item_threads) = batch::decide(op, exec, input.batch(), Some(n), item_muladds)?;
    let par = batch::route_par(lanes, item_threads);
    let per_lane = per_lane_bytes(T::eig_req(n, false, par), input.batch());
    batch::check_workspace(op, workspace, per_lane, lanes)?;
    batch::run(
        op,
        input.batch(),
        exec,
        Some(n),
        item_muladds,
        &mut (fixed_out(values, n, "eigenvalues"),),
        batch::ScratchArena::new(workspace, per_lane),
        |par, stack| T::eig_scratch(n, false, par, MemStack::new(stack)),
        |index, (values,), scratch, par| {
            if n == 0 {
                return Ok(());
            }
            T::eig_into(op, input.item(index), values, None, scratch, par)
        },
    )
}

/// Eigenvalues and eigenvectors of every `n x n` matrix of a batch, `A V = V diag(w)`.
///
/// `input` is `[n, n, b...]`. Per item, `values` receives `n` values and `vectors` the column-major
/// `n x n` eigenvectors, in the complex type of the input scalar.
///
/// This is the `Vec` wrapper of [`eig_into`].
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
/// use tcontract_exec::Exec;
/// use tlinalg::{eig::eig, Op};
///
/// // Rotation by 90 degrees: eigenvalues ±i.
/// let a = [0.0_f64, 1.0, -1.0, 0.0];
/// let (mut w, mut v) = (Vec::<Complex64>::new(), Vec::new());
/// eig(
///     Op::Eig, RawStridedRef::new(&a, &[2, 2], &[1, 2], 0).unwrap(), &mut w, &mut v,
///     Exec::serial(),
/// ).unwrap();
/// assert!((w[0].im.abs() - 1.0).abs() < 1e-12 && (w[0].conj() - w[1]).norm() < 1e-12);
/// ```
pub fn eig<T: FaerScalar>(
    op: Op,
    input: RawStridedRef<'_, T>,
    values: &mut Vec<<T as ScalarEntity>::Complex>,
    vectors: &mut Vec<<T as ScalarEntity>::Complex>,
    exec: Exec<'_>,
) -> Result<()> {
    values.clear();
    vectors.clear();
    let input = BatchedRef::square(op, "input", input)?;
    let n = input.rows();
    let v_len = checked_product(op, "eigenvector matrix", &[n, n])?;
    let batch = input.batch();
    let mut workspace =
        vec![MaybeUninit::<u8>::uninit(); eig_workspace::<T>(n, true, batch, exec)?];
    let mut values_out = VecOut::new(values, checked_product(op, "eigenvalues", &[n, batch])?);
    let mut vectors_out = VecOut::new(
        vectors,
        checked_product(op, "eigenvectors", &[v_len, batch])?,
    );
    eig_into(
        op,
        input.raw(),
        values_out.spare(),
        vectors_out.spare(),
        &mut workspace,
        exec,
    )?;
    values_out.publish();
    vectors_out.publish();
    Ok(())
}

/// Eigenvalues of every `n x n` matrix of a batch, without the eigenvectors.
///
/// `input` is `[n, n, b...]`; `values` receives `n` values per item in the complex type of the
/// input scalar.
///
/// This is the `Vec` wrapper of [`eig_values_into`].
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
/// use tcontract_exec::Exec;
/// use tlinalg::{eig::eig_values, Op};
///
/// let a = [2.0_f64, 0.0, 0.0, 3.0];
/// let mut w = Vec::<Complex64>::new();
/// eig_values(
///     Op::EigValues, RawStridedRef::new(&a, &[2, 2], &[1, 2], 0).unwrap(), &mut w,
///     Exec::serial(),
/// ).unwrap();
/// assert_eq!(w.len(), 2);
/// ```
pub fn eig_values<T: FaerScalar>(
    op: Op,
    input: RawStridedRef<'_, T>,
    values: &mut Vec<<T as ScalarEntity>::Complex>,
    exec: Exec<'_>,
) -> Result<()> {
    values.clear();
    let input = BatchedRef::square(op, "input", input)?;
    let n = input.rows();
    let batch = input.batch();
    let mut workspace =
        vec![MaybeUninit::<u8>::uninit(); eig_values_workspace::<T>(n, batch, exec)?];
    let mut values_out = VecOut::new(values, checked_product(op, "eigenvalues", &[n, batch])?);
    eig_values_into(op, input.raw(), values_out.spare(), &mut workspace, exec)?;
    values_out.publish();
    Ok(())
}

/// `|im| <= ε · max(|re|, 1)`, the test for treating a real-input eigenvalue as real.
fn imag_is_effectively_zero(real: f64, imag: f64, eps: f64) -> bool {
    imag.abs() <= eps * real.abs().max(1.0)
}

/// faer's eigensolver scratch for `n x n` (empty for `n == 0`, which is never decomposed).
fn evd_req<E: faer::traits::ComplexField>(n: usize, vectors: bool, par: faer::Par) -> StackReq {
    if n == 0 {
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
    }
}

/// Lane storage types, public only so the crate-internal `EigScalar` can name them.
mod lane_storage {
    use core::mem::MaybeUninit;

    use faer::diag::DiagMut;
    use faer::MatMut;

    /// Lane storage of the real eigensolver.
    pub struct RealEigScratch<'w, R: faer::traits::ComplexField> {
        pub(super) u: MatMut<'w, R>,
        pub(super) s_re: DiagMut<'w, R>,
        pub(super) s_im: DiagMut<'w, R>,
        pub(super) mem: &'w mut [MaybeUninit<u8>],
    }

    /// Lane storage of the complex eigensolver.
    pub struct ComplexEigScratch<'w, E: faer::traits::ComplexField> {
        pub(super) u: MatMut<'w, E>,
        pub(super) s: DiagMut<'w, E>,
        pub(super) mem: &'w mut [MaybeUninit<u8>],
        /// Whether an earlier item has written `u` and `s`, so the next one has to be reset first.
        pub(super) reused: bool,
    }
}

use lane_storage::{ComplexEigScratch, RealEigScratch};

macro_rules! impl_eig_real {
    ($real:ty, $complex:ty) => {
        impl EigScalar for $real {
            type EigScratch<'w> = RealEigScratch<'w, $real>;

            fn eig_req(n: usize, vectors: bool, par: faer::Par) -> StackReq {
                StackReq::new::<$real>(n * if vectors { n } else { 0 })
                    .and(StackReq::new::<$real>(n))
                    .and(StackReq::new::<$real>(n))
                    .and(faer_req(evd_req::<$real>(n, vectors, par)))
            }

            fn eig_scratch<'w>(
                n: usize,
                vectors: bool,
                par: faer::Par,
                stack: &'w mut MemStack,
            ) -> Self::EigScratch<'w> {
                let (u, stack) = if vectors {
                    let (u, stack) = zeros::<$real>(stack, n * n, &0.0);
                    (MatMut::from_column_major_slice_mut(u, n, n), stack)
                } else {
                    let (u, stack) = zeros::<$real>(stack, 0, &0.0);
                    (MatMut::from_column_major_slice_mut(u, n, 0), stack)
                };
                let (s_re, stack) = zeros::<$real>(stack, n, &0.0);
                let (s_im, stack) = zeros::<$real>(stack, n, &0.0);
                let (mem, _) = bytes(
                    stack,
                    evd_req::<$real>(n, vectors, par).unaligned_bytes_required(),
                );
                RealEigScratch {
                    u,
                    s_re: DiagMut::from_slice_mut(s_re),
                    s_im: DiagMut::from_slice_mut(s_im),
                    mem,
                }
            }

            fn eig_into(
                op: Op,
                mat: MatRef<'_, $real>,
                values: &mut dyn Push<$complex>,
                vectors: Option<&mut dyn Push<$complex>>,
                scratch: &mut Self::EigScratch<'_>,
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
                    MemStack::new(&mut **mem),
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
            type EigScratch<'w> = ComplexEigScratch<'w, $entity>;

            fn eig_req(n: usize, vectors: bool, par: faer::Par) -> StackReq {
                StackReq::new::<$entity>(n * if vectors { n } else { 0 })
                    .and(StackReq::new::<$entity>(n))
                    .and(faer_req(evd_req::<$entity>(n, vectors, par)))
            }

            fn eig_scratch<'w>(
                n: usize,
                vectors: bool,
                par: faer::Par,
                stack: &'w mut MemStack,
            ) -> Self::EigScratch<'w> {
                let zero = <$entity>::new(0.0, 0.0);
                let (u, stack) = if vectors {
                    zeros::<$entity>(stack, n * n, &zero)
                } else {
                    zeros::<$entity>(stack, 0, &zero)
                };
                let (s, stack) = zeros::<$entity>(stack, n, &zero);
                let (mem, _) = bytes(
                    stack,
                    evd_req::<$entity>(n, vectors, par).unaligned_bytes_required(),
                );
                ComplexEigScratch {
                    u: MatMut::from_column_major_slice_mut(u, n, if vectors { n } else { 0 }),
                    s: DiagMut::from_slice_mut(s),
                    mem,
                    reused: false,
                }
            }

            fn eig_into(
                op: Op,
                mat: MatRef<'_, $entity>,
                values: &mut dyn Push<$complex>,
                vectors: Option<&mut dyn Push<$complex>>,
                scratch: &mut Self::EigScratch<'_>,
                par: faer::Par,
            ) -> Result<()> {
                let n = mat.nrows();
                let zero = <$entity>::new(0.0, 0.0);
                let ComplexEigScratch { u, s, mem, reused } = scratch;
                // The carved buffers are already zeroed; only a scratch that an earlier item wrote
                // needs resetting.
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
                    MemStack::new(&mut **mem),
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
