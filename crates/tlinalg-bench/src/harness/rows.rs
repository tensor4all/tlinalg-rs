//! Provider call rows and CSV records.

use crate::harness::cases::{applicable, Case};
use crate::harness::timing;
#[cfg(any(
    feature = "link-openblas",
    feature = "link-openblas-static",
    feature = "link-mkl"
))]
use crate::RecyclingWorkspace;
use crate::{Batch, BenchScalar, Env};
use num_complex::Complex64;
use std::cell::RefCell;
use tlinalg::{Op, Parallel};

/// One output record.
#[derive(Clone, Debug)]
pub struct Record {
    /// Campaign regime.
    pub regime: String,
    /// Family name.
    pub family: String,
    /// Dtype label.
    pub dtype: String,
    /// Rows.
    pub m: usize,
    /// Columns.
    pub n: usize,
    /// Batch count.
    pub batch: usize,
    /// Provider row label.
    pub row: String,
    /// Declared thread budget.
    pub threads: usize,
    /// Best timing.
    pub total_ms: f64,
    /// Timing divided by batch.
    pub per_item_us: f64,
    /// `ok`, `skipped`, or `failed`.
    pub status: String,
    /// Diagnostic note.
    pub note: String,
}

impl Record {
    pub fn csv(&self) -> String {
        format!(
            "{},{},{},{},{},{},{},{},{:.6},{:.6},{},{}",
            self.regime,
            self.family,
            self.dtype,
            self.m,
            self.n,
            self.batch,
            self.row,
            self.threads,
            self.total_ms,
            self.per_item_us,
            self.status,
            self.note.replace(',', ";")
        )
    }
}

/// Execute one provider family. This is intentionally the same call vocabulary as `benches/kernels.rs`.
pub fn call<T: BenchScalar>(
    family: &str,
    c: Case,
    par: Parallel<'_>,
    lapack: bool,
) -> Result<(), String> {
    if lapack {
        return call_lapack::<T>(family, c);
    }
    let a = Batch::<T>::general(c.m, c.n, c.batch);
    let b = Batch::<T>::general(c.m, 4, c.batch);
    let op = match family {
        "cholesky" | "eigh" | "eigvalsh" => Batch::<T>::hpd(c.m, c.batch),
        _ => Batch::<T>::general(c.m, c.n, c.batch),
    };
    let mut out = Vec::<T>::new();
    match family {
        "lu_factor" => {
            let mut x = a.data.clone();
            let mut p = vec![0; c.m * c.batch];
            let mut parity = vec![T::default(); c.batch];
            tlinalg::packed_lu::factor(Op::LuFactor, c.m, c.n, &mut x, &mut p, &mut parity, par)
                .map_err(|e| e.to_string())?;
        }
        "lu_solve_prepared" => {
            let mut x = a.data.clone();
            let mut p = vec![0; c.m * c.batch];
            let mut parity = vec![T::default(); c.batch];
            tlinalg::packed_lu::factor(
                Op::LuFactor,
                c.m,
                c.n,
                &mut x,
                &mut p,
                &mut parity,
                Parallel::Sequential,
            )
            .map_err(|e| e.to_string())?;
            let dims = [c.m, c.m, c.batch];
            let ps = [1, c.m as isize, (c.m * c.m) as isize];
            let pd = [c.m, c.batch];
            let pss = [1, c.m as isize];
            let mut rhs = b.data.clone();
            tlinalg::packed_lu::solve_prepared(
                Op::LuSolvePrepared,
                strided_view::RawStridedRef::new(&x, &dims, &ps, 0).unwrap(),
                strided_view::RawStridedRef::new(&p, &pd, &pss, 0).unwrap(),
                4,
                &mut rhs,
                false,
                false,
                par,
            )
            .map_err(|e| e.to_string())?;
        }
        "lu_factor_solve" => {
            let mut x = a.data.clone();
            let mut p = vec![0; c.m * c.batch];
            let mut rhs = b.data.clone();
            tlinalg::packed_lu::factor_solve(
                Op::LuFactorSolve,
                c.m,
                4,
                &mut x,
                &mut p,
                &mut rhs,
                par,
            )
            .map_err(|e| e.to_string())?;
        }
        "solve" => {
            let dims = [c.m, 4, c.batch];
            let strides = [1, c.m as isize, (c.m * 4) as isize];
            let mut x = vec![T::default(); c.m * 4 * c.batch];
            let dst = strided_view::RawStridedMut::new(&mut x, &dims, &strides, 0).unwrap();
            tlinalg::lu::solve(Op::Solve, a.view(), Some(b.view()), dst, false, par)
                .map_err(|e| e.to_string())?;
        }
        "lu" => {
            let (mut p, mut l, mut u, mut parity) =
                (Vec::new(), Vec::new(), Vec::new(), Vec::new());
            tlinalg::lu::lu(
                Op::Lu,
                a.view(),
                tlinalg::lu::LuFactors {
                    p: &mut p,
                    l: &mut l,
                    u: &mut u,
                    parity: &mut parity,
                },
                par,
            )
            .map_err(|e| e.to_string())?;
        }
        "cholesky" => {
            tlinalg::cholesky::cholesky(Op::Cholesky, op.view(), &mut out, par)
                .map_err(|e| e.to_string())?;
        }
        "triangular_solve" => {
            let flags = tlinalg::triangular_solve::TriangularSolveFlags {
                left_side: true,
                lower: true,
                transpose_a: false,
                unit_diagonal: false,
            };
            tlinalg::triangular_solve::triangular_solve(
                Op::TriangularSolve,
                a.view(),
                b.view(),
                flags,
                &mut out,
                par,
            )
            .map_err(|e| e.to_string())?;
        }
        "qr" => {
            let (mut q, mut r) = (Vec::new(), Vec::new());
            tlinalg::qr::qr(Op::Qr, a.view(), &mut q, &mut r, par).map_err(|e| e.to_string())?;
        }
        "rank_revealing_qr" => {
            let (mut q, mut r, mut p) = (Vec::new(), Vec::new(), Vec::new());
            tlinalg::qr::rank_revealing_qr(
                Op::RankRevealingQr,
                a.view(),
                &mut q,
                &mut r,
                &mut p,
                par,
            )
            .map_err(|e| e.to_string())?;
        }
        "svd_thin" | "svd_full" => {
            let (mut u, mut s, mut vt) = (Vec::new(), Vec::new(), Vec::new());
            tlinalg::svd::svd(
                Op::Svd,
                a.view(),
                family == "svd_full",
                &mut u,
                &mut s,
                &mut vt,
                par,
            )
            .map_err(|e| e.to_string())?;
        }
        "svd_values" => {
            let mut s = Vec::new();
            tlinalg::svd::svd_values(Op::SvdValues, a.view(), &mut s, par)
                .map_err(|e| e.to_string())?;
        }
        "eigh" => {
            let (mut w, mut v) = (Vec::new(), Vec::new());
            tlinalg::eigh::eigh(Op::Eigh, op.view(), &mut w, &mut v, par)
                .map_err(|e| e.to_string())?;
        }
        "eigvalsh" => {
            let mut w = Vec::new();
            tlinalg::eigh::eigh_values(Op::EighValues, op.view(), &mut w, par)
                .map_err(|e| e.to_string())?;
        }
        "eig" => {
            let (mut w, mut v) = (Vec::new(), Vec::new());
            tlinalg::eig::eig(Op::Eig, a.view(), &mut w, &mut v, par).map_err(|e| e.to_string())?;
        }
        "eigvals" => {
            let mut w = Vec::new();
            tlinalg::eig::eig_values(Op::EigValues, a.view(), &mut w, par)
                .map_err(|e| e.to_string())?;
        }
        "full_piv_lu" => {
            let (mut p, mut l, mut u, mut q, mut parity) =
                (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new());
            tlinalg::full_piv_lu::full_piv_lu(
                Op::FullPivLu,
                a.view(),
                tlinalg::full_piv_lu::FullPivLuFactors {
                    p: &mut p,
                    l: &mut l,
                    u: &mut u,
                    q: &mut q,
                    parity: &mut parity,
                },
                par,
            )
            .map_err(|e| e.to_string())?;
        }
        "full_piv_lu_solve" => {
            let mut x = Vec::new();
            tlinalg::full_piv_lu::full_piv_lu_solve(
                Op::FullPivLuSolve,
                a.view(),
                b.view(),
                false,
                &mut x,
                par,
            )
            .map_err(|e| e.to_string())?;
        }
        "householder_factor" => {
            let mut x = a.data.clone();
            let mut tau = Vec::new();
            tlinalg::householder::compact_factor(
                Op::HouseholderQr,
                c.m,
                c.n,
                c.batch,
                &mut x,
                &mut tau,
                par,
            )
            .map_err(|e| e.to_string())?;
        }
        "householder_apply" => {
            let mut x = a.data.clone();
            let mut tau = Vec::new();
            tlinalg::householder::compact_factor(
                Op::HouseholderQr,
                c.m,
                c.n,
                c.batch,
                &mut x,
                &mut tau,
                Parallel::Sequential,
            )
            .map_err(|e| e.to_string())?;
            let mut target = Batch::<T>::general(c.m, c.m.min(c.n), c.batch).data;
            let shape = tlinalg::householder::ReflectorShape {
                rows: c.m,
                a_cols: c.n,
                cols: c.m.min(c.n),
                k: c.m.min(c.n),
            };
            tlinalg::householder::apply_reflectors(
                Op::HouseholderQrQColumns,
                shape,
                c.batch,
                &x,
                &tau,
                &mut target,
                false,
                par,
            )
            .map_err(|e| e.to_string())?;
        }
        _ => return Err(format!("unknown family {family}")),
    }
    Ok(())
}

#[cfg(any(
    feature = "link-openblas",
    feature = "link-openblas-static",
    feature = "link-mkl"
))]
fn call_lapack<T: BenchScalar>(family: &str, c: Case) -> Result<(), String> {
    let a = Batch::<T>::general(c.m, c.n, c.batch);
    let b = Batch::<T>::general(c.m, 4, c.batch);
    let op = match family {
        "cholesky" | "eigh" | "eigvalsh" => Batch::<T>::hpd(c.m, c.batch),
        _ => Batch::<T>::general(c.m, c.n, c.batch),
    };
    let mut ws = RecyclingWorkspace::default();
    match family {
        "lu_factor" => {
            let mut x = a.data.clone();
            let mut p = vec![0; c.m * c.batch];
            let mut q = vec![T::default(); c.batch];
            tlinalg_blas::lu::lu_factor(
                tlinalg_blas::Op::LuFactor,
                c.m,
                c.n,
                &mut x,
                &mut p,
                &mut q,
            )
            .map_err(|e| e.to_string())?;
        }
        "lu_solve_prepared" => {
            let mut x = a.data.clone();
            let mut p = vec![0; c.m * c.batch];
            let mut q = vec![T::default(); c.batch];
            tlinalg_blas::lu::lu_factor(
                tlinalg_blas::Op::LuFactor,
                c.m,
                c.n,
                &mut x,
                &mut p,
                &mut q,
            )
            .map_err(|e| e.to_string())?;
            let mut rhs = b.data.clone();
            tlinalg_blas::lu::lu_solve_prepared(
                tlinalg_blas::Op::LuSolvePrepared,
                c.m,
                4,
                &x,
                &p,
                &mut rhs,
                false,
                false,
            )
            .map_err(|e| e.to_string())?;
        }
        "lu_factor_solve" => {
            let mut x = a.data.clone();
            let mut p = vec![0; c.m * c.batch];
            let mut rhs = b.data.clone();
            tlinalg_blas::lu::lu_factor_solve(
                tlinalg_blas::Op::LuFactorSolve,
                c.m,
                4,
                &mut x,
                &mut p,
                &mut rhs,
            )
            .map_err(|e| e.to_string())?;
        }
        "solve" => {
            let mut out = Vec::new();
            tlinalg_blas::solve::solve(
                tlinalg_blas::Op::Solve,
                false,
                a.view(),
                b.view(),
                &mut out,
                &mut ws,
            )
            .map_err(|e| e.to_string())?;
        }
        "lu" => {
            let (mut p, mut l, mut u, mut q) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
            tlinalg_blas::lu::lu(
                tlinalg_blas::Op::Lu,
                a.view(),
                tlinalg_blas::lu::LuOutputs {
                    p: &mut p,
                    l: &mut l,
                    u: &mut u,
                    parity: &mut q,
                },
                &mut ws,
            )
            .map_err(|e| e.to_string())?;
        }
        "cholesky" => {
            let mut out = Vec::new();
            tlinalg_blas::cholesky::cholesky(
                tlinalg_blas::Op::Cholesky,
                op.view(),
                &mut out,
                &mut ws,
            )
            .map_err(|e| e.to_string())?;
        }
        "triangular_solve" => {
            let mut out = Vec::new();
            let flags = tlinalg_blas::triangular_solve::TriangularSolveOptions {
                left_side: true,
                lower: true,
                transpose_a: false,
                unit_diagonal: false,
            };
            tlinalg_blas::triangular_solve::triangular_solve(
                tlinalg_blas::Op::TriangularSolve,
                flags,
                a.view(),
                b.view(),
                &mut out,
                &mut ws,
            )
            .map_err(|e| e.to_string())?;
        }
        "qr" => {
            let (mut q, mut r) = (Vec::new(), Vec::new());
            tlinalg_blas::qr::qr(tlinalg_blas::Op::Qr, a.view(), &mut q, &mut r, &mut ws)
                .map_err(|e| e.to_string())?;
        }
        "rank_revealing_qr" => {
            let (mut q, mut r, mut p) = (Vec::new(), Vec::new(), Vec::new());
            tlinalg_blas::qr::rank_revealing_qr(
                tlinalg_blas::Op::RankRevealingQr,
                a.view(),
                tlinalg_blas::qr::RankRevealingQrOutputs {
                    q: &mut q,
                    r: &mut r,
                    permutation: &mut p,
                },
                &mut ws,
            )
            .map_err(|e| e.to_string())?;
        }
        "svd_thin" | "svd_full" => {
            let (mut s, mut u, mut vt) = (Vec::new(), Vec::new(), Vec::new());
            tlinalg_blas::svd::svd(
                tlinalg_blas::Op::Svd,
                if family == "svd_full" {
                    tlinalg_blas::svd::SvdMode::Full
                } else {
                    tlinalg_blas::svd::SvdMode::Thin
                },
                a.view(),
                tlinalg_blas::svd::SvdOutputs {
                    s: &mut s,
                    u: &mut u,
                    vt: &mut vt,
                },
                &mut ws,
            )
            .map_err(|e| e.to_string())?;
        }
        "svd_values" => {
            let (mut s, mut u, mut vt) = (Vec::new(), Vec::new(), Vec::new());
            tlinalg_blas::svd::svd(
                tlinalg_blas::Op::SvdValues,
                tlinalg_blas::svd::SvdMode::Values,
                a.view(),
                tlinalg_blas::svd::SvdOutputs {
                    s: &mut s,
                    u: &mut u,
                    vt: &mut vt,
                },
                &mut ws,
            )
            .map_err(|e| e.to_string())?;
        }
        "eigh" => {
            let (mut w, mut v) = (Vec::new(), Vec::new());
            tlinalg_blas::eigh::eigh(
                tlinalg_blas::Op::Eigh,
                op.view(),
                &mut w,
                Some(&mut v),
                &mut ws,
            )
            .map_err(|e| e.to_string())?;
        }
        "eigvalsh" => {
            let mut w = Vec::new();
            tlinalg_blas::eigh::eigh(tlinalg_blas::Op::Eigh, op.view(), &mut w, None, &mut ws)
                .map_err(|e| e.to_string())?;
        }
        "eig" => {
            let (mut w, mut v) = (Vec::new(), Vec::new());
            tlinalg_blas::eig::eig(
                tlinalg_blas::Op::Eig,
                a.view(),
                &mut w,
                Some(&mut v),
                &mut ws,
            )
            .map_err(|e| e.to_string())?;
        }
        "eigvals" => {
            let mut w = Vec::new();
            tlinalg_blas::eig::eig(tlinalg_blas::Op::Eig, a.view(), &mut w, None, &mut ws)
                .map_err(|e| e.to_string())?;
        }
        "full_piv_lu" => {
            let (mut p, mut l, mut u, mut q, mut parity) =
                (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new());
            tlinalg_blas::full_piv_lu::full_piv_lu(
                tlinalg_blas::Op::FullPivLu,
                a.view(),
                tlinalg_blas::full_piv_lu::FullPivLuOutputs {
                    p: &mut p,
                    l: &mut l,
                    u: &mut u,
                    q: &mut q,
                    parity: &mut parity,
                },
                &mut ws,
            )
            .map_err(|e| e.to_string())?;
        }
        "full_piv_lu_solve" => {
            let mut x = Vec::new();
            tlinalg_blas::full_piv_lu::full_piv_lu_solve(
                tlinalg_blas::Op::FullPivLuSolve,
                false,
                a.view(),
                b.view(),
                &mut x,
                &mut ws,
            )
            .map_err(|e| e.to_string())?;
        }
        "householder_factor" => {
            let mut x = a.data.clone();
            let mut tau = Vec::new();
            tlinalg_blas::householder::factor(
                tlinalg_blas::Op::HouseholderQr,
                c.m,
                c.n,
                &mut x,
                &mut tau,
                &mut ws,
            )
            .map_err(|e| e.to_string())?;
        }
        "householder_apply" => {
            let mut x = a.data.clone();
            let mut tau = Vec::new();
            tlinalg_blas::householder::factor(
                tlinalg_blas::Op::HouseholderQr,
                c.m,
                c.n,
                &mut x,
                &mut tau,
                &mut ws,
            )
            .map_err(|e| e.to_string())?;
            let mut target = Batch::<T>::general(c.m, c.m.min(c.n), c.batch).data;
            tlinalg_blas::householder::apply_reflectors(
                tlinalg_blas::Op::HouseholderQrQColumns,
                c.m,
                c.n,
                c.m.min(c.n),
                c.m.min(c.n),
                false,
                &x,
                &tau,
                &mut target,
                &mut ws,
            )
            .map_err(|e| e.to_string())?;
        }
        _ => return Err(format!("unknown family {family}")),
    }
    Ok(())
}
#[cfg(not(any(
    feature = "link-openblas",
    feature = "link-openblas-static",
    feature = "link-mkl"
)))]
fn call_lapack<T: BenchScalar>(_family: &str, _c: Case) -> Result<(), String> {
    // The signature has to match the linked variant's, which the caller uses with a turbofish.
    let _ = core::marker::PhantomData::<T>;
    Err("lapack vendor not linked".into())
}

/// Outputs returned by the correctness-only path.
#[derive(Debug)]
pub enum Output<T> {
    /// A single factor or solution buffer.
    One(Vec<T>),
    /// Two factors (for example Q and R, or eigenvalues and vectors).
    Two(Vec<T>, Vec<T>),
    /// Real values and scalar vectors (eigenvalue output).
    TwoReal(Vec<f64>, Vec<T>),
    /// Three factors (for example U, S and Vᴴ).
    Three(Vec<T>, Vec<T>, Vec<T>),
    /// SVD factors with real singular values from LAPACK.
    ThreeReal(Vec<T>, Vec<f64>, Vec<T>),
    /// Packed LU factors, pivots, and parity.
    PackedLu {
        lu: Vec<T>,
        piv: Vec<i32>,
        parity: Vec<T>,
    },
    /// Explicit partial-pivot LU factors and parity.
    Lu {
        p: Vec<T>,
        l: Vec<T>,
        u: Vec<T>,
        parity: Vec<T>,
    },
    /// Complete-pivot LU factors and parity.
    FullLu {
        p: Vec<T>,
        l: Vec<T>,
        u: Vec<T>,
        q: Vec<T>,
        parity: Vec<T>,
    },
    /// QR factors and the returned column permutation.
    TwoPerm(Vec<T>, Vec<T>, Vec<i64>),
    /// General eigenvalues and eigenvectors widened to the reference type.
    Eig(Vec<Complex64>, Vec<Complex64>),
    /// Compact Householder state and coefficients, with an optional applied target.
    Householder {
        a: Vec<T>,
        tau: Vec<T>,
        target: Option<Vec<T>>,
    },
}

/// The two supported scalar implementations have the same `Complex64` eigensolver output, but the
/// provider keeps that associated type private; this tiny adapter keeps the conversion at the edge.
pub(crate) trait EigBuffers: BenchScalar {
    fn faer_eig(
        a: &Batch<Self>,
        values: &mut Vec<Complex64>,
        vectors: Option<&mut Vec<Complex64>>,
    ) -> Result<(), String>;
    fn faer_eig_values(a: &Batch<Self>, values: &mut Vec<Complex64>) -> Result<(), String>;
    #[cfg(any(
        feature = "link-openblas",
        feature = "link-openblas-static",
        feature = "link-mkl"
    ))]
    fn lapack_eig(
        a: &Batch<Self>,
        values: &mut Vec<Complex64>,
        vectors: Option<&mut Vec<Complex64>>,
    ) -> Result<(), String>;
    #[cfg(any(
        feature = "link-openblas",
        feature = "link-openblas-static",
        feature = "link-mkl"
    ))]
    fn lapack_eig_values(a: &Batch<Self>, values: &mut Vec<Complex64>) -> Result<(), String>;
}

macro_rules! impl_eig_buffers {
    ($scalar:ty) => {
        impl EigBuffers for $scalar {
            fn faer_eig(
                a: &Batch<Self>,
                values: &mut Vec<Complex64>,
                vectors: Option<&mut Vec<Complex64>>,
            ) -> Result<(), String> {
                let mut w = Vec::new();
                let mut v = Vec::new();
                tlinalg::eig::eig(Op::Eig, a.view(), &mut w, &mut v, Parallel::Sequential)
                    .map_err(|e| e.to_string())?;
                *values = w;
                if let Some(dst) = vectors {
                    *dst = v;
                }
                Ok(())
            }
            fn faer_eig_values(a: &Batch<Self>, values: &mut Vec<Complex64>) -> Result<(), String> {
                tlinalg::eig::eig_values(Op::EigValues, a.view(), values, Parallel::Sequential)
                    .map_err(|e| e.to_string())
            }
            #[cfg(any(
                feature = "link-openblas",
                feature = "link-openblas-static",
                feature = "link-mkl"
            ))]
            fn lapack_eig(
                a: &Batch<Self>,
                values: &mut Vec<Complex64>,
                vectors: Option<&mut Vec<Complex64>>,
            ) -> Result<(), String> {
                let mut ws = RecyclingWorkspace::default();
                let mut w = Vec::new();
                let mut v = Vec::new();
                tlinalg_blas::eig::eig(
                    tlinalg_blas::Op::Eig,
                    a.view(),
                    &mut w,
                    Some(&mut v),
                    &mut ws,
                )
                .map_err(|e| e.to_string())?;
                *values = w;
                if let Some(dst) = vectors {
                    *dst = v;
                }
                Ok(())
            }
            #[cfg(any(
                feature = "link-openblas",
                feature = "link-openblas-static",
                feature = "link-mkl"
            ))]
            fn lapack_eig_values(
                a: &Batch<Self>,
                values: &mut Vec<Complex64>,
            ) -> Result<(), String> {
                let mut ws = RecyclingWorkspace::default();
                tlinalg_blas::eig::eig(
                    tlinalg_blas::Op::EigValues,
                    a.view(),
                    values,
                    None,
                    &mut ws,
                )
                .map_err(|e| e.to_string())?;
                Ok(())
            }
        }
    };
}
impl_eig_buffers!(f64);
impl_eig_buffers!(Complex64);

pub(crate) trait LapackEigh: BenchScalar {
    /// Only the vendor verify path calls this, and that path is compiled out when no vendor
    /// library is linked; the bound stays unconditional so the generic code keeps one shape.
    #[allow(dead_code)]
    fn lapack_eigh(
        a: &Batch<Self>,
        values: &mut Vec<f64>,
        vectors: Option<&mut Vec<Self>>,
    ) -> Result<(), String>;
}
macro_rules! impl_lapack_eigh {
    ($scalar:ty) => {
        impl LapackEigh for $scalar {
            fn lapack_eigh(
                a: &Batch<Self>,
                values: &mut Vec<f64>,
                vectors: Option<&mut Vec<Self>>,
            ) -> Result<(), String> {
                #[cfg(any(
                    feature = "link-openblas",
                    feature = "link-openblas-static",
                    feature = "link-mkl"
                ))]
                {
                    let mut ws = RecyclingWorkspace::default();
                    return tlinalg_blas::eigh::eigh(
                        tlinalg_blas::Op::Eigh,
                        a.view(),
                        values,
                        vectors,
                        &mut ws,
                    )
                    .map_err(|e| e.to_string());
                }
                #[cfg(not(any(
                    feature = "link-openblas",
                    feature = "link-openblas-static",
                    feature = "link-mkl"
                )))]
                {
                    let _ = (a, values, vectors);
                    Err("lapack vendor not linked".into())
                }
            }
        }
    };
}
impl_lapack_eigh!(f64);
impl_lapack_eigh!(Complex64);
pub(crate) trait RealValues: BenchScalar {
    fn faer_values(a: &Batch<Self>) -> Result<Vec<f64>, String>;
    #[cfg(any(
        feature = "link-openblas",
        feature = "link-openblas-static",
        feature = "link-mkl"
    ))]
    fn lapack_values(a: &Batch<Self>) -> Result<Vec<f64>, String>;
}
macro_rules! impl_real_values {
    ($scalar:ty) => {
        impl RealValues for $scalar {
            fn faer_values(a: &Batch<Self>) -> Result<Vec<f64>, String> {
                let mut w = Vec::new();
                tlinalg::eigh::eigh_values(Op::EighValues, a.view(), &mut w, Parallel::Sequential)
                    .map_err(|e| e.to_string())?;
                Ok(w)
            }
            #[cfg(any(
                feature = "link-openblas",
                feature = "link-openblas-static",
                feature = "link-mkl"
            ))]
            fn lapack_values(a: &Batch<Self>) -> Result<Vec<f64>, String> {
                let mut ws = RecyclingWorkspace::default();
                let mut w = Vec::new();
                tlinalg_blas::eigh::eigh(tlinalg_blas::Op::Eigh, a.view(), &mut w, None, &mut ws)
                    .map_err(|e| e.to_string())?;
                Ok(w)
            }
        }
    };
}
impl_real_values!(f64);
impl_real_values!(Complex64);

/// Execute a provider once and return its produced buffers for verification.
pub fn output_call<T: BenchScalar>(
    family: &str,
    c: Case,
    lapack: bool,
) -> Result<Output<T>, String> {
    let a = Batch::<T>::general(c.m, c.n, c.batch);
    let b = Batch::<T>::general(c.m, 4, c.batch);
    let op = if matches!(family, "cholesky" | "eigh" | "eigvalsh") {
        Batch::<T>::hpd(c.m, c.batch)
    } else {
        Batch::<T>::general(c.m, c.n, c.batch)
    };
    if lapack {
        #[cfg(any(
            feature = "link-openblas",
            feature = "link-openblas-static",
            feature = "link-mkl"
        ))]
        {
            let mut ws = RecyclingWorkspace::default();
            return match family {
                "cholesky" => {
                    let mut x = Vec::new();
                    tlinalg_blas::cholesky::cholesky(
                        tlinalg_blas::Op::Cholesky,
                        op.view(),
                        &mut x,
                        &mut ws,
                    )
                    .map_err(|e| e.to_string())?;
                    Ok(Output::One(x))
                }
                "qr" => {
                    let (mut q, mut r) = (Vec::new(), Vec::new());
                    tlinalg_blas::qr::qr(tlinalg_blas::Op::Qr, a.view(), &mut q, &mut r, &mut ws)
                        .map_err(|e| e.to_string())?;
                    Ok(Output::Two(q, r))
                }
                "svd_thin" | "svd_full" => {
                    let (mut s, mut u, mut v) = (Vec::new(), Vec::new(), Vec::new());
                    tlinalg_blas::svd::svd(
                        tlinalg_blas::Op::Svd,
                        if family == "svd_full" {
                            tlinalg_blas::svd::SvdMode::Full
                        } else {
                            tlinalg_blas::svd::SvdMode::Thin
                        },
                        a.view(),
                        tlinalg_blas::svd::SvdOutputs {
                            s: &mut s,
                            u: &mut u,
                            vt: &mut v,
                        },
                        &mut ws,
                    )
                    .map_err(|e| e.to_string())?;
                    let s = s.into_iter().map(|x| T::real_to_f64(x)).collect();
                    Ok(Output::ThreeReal(u, s, v))
                }
                "svd_values" => {
                    let (mut s, mut u, mut v) = (Vec::new(), Vec::new(), Vec::new());
                    tlinalg_blas::svd::svd(
                        tlinalg_blas::Op::Svd,
                        tlinalg_blas::svd::SvdMode::Values,
                        a.view(),
                        tlinalg_blas::svd::SvdOutputs {
                            s: &mut s,
                            u: &mut u,
                            vt: &mut v,
                        },
                        &mut ws,
                    )
                    .map_err(|e| e.to_string())?;
                    Ok(Output::One(
                        s.into_iter()
                            .map(|x| {
                                T::from_c64(num_complex::Complex64::new(T::real_to_f64(x), 0.))
                            })
                            .collect(),
                    ))
                }
                "eigh" => {
                    let (mut w, mut v) = (Vec::new(), Vec::new());
                    tlinalg_blas::eigh::eigh(
                        tlinalg_blas::Op::Eigh,
                        op.view(),
                        &mut w,
                        Some(&mut v),
                        &mut ws,
                    )
                    .map_err(|e| e.to_string())?;
                    let w = w.into_iter().map(|value| T::real_to_f64(value)).collect();
                    Ok(Output::TwoReal(w, v))
                }
                "solve" => {
                    let mut x = Vec::new();
                    tlinalg_blas::solve::solve(
                        tlinalg_blas::Op::Solve,
                        false,
                        a.view(),
                        b.view(),
                        &mut x,
                        &mut ws,
                    )
                    .map_err(|e| e.to_string())?;
                    Ok(Output::One(x))
                }
                _ => {
                    call_lapack::<T>(family, c)?;
                    Err("output path unavailable for this family".into())
                }
            };
        }
        #[cfg(not(any(
            feature = "link-openblas",
            feature = "link-openblas-static",
            feature = "link-mkl"
        )))]
        {
            return Err("lapack vendor not linked".into());
        }
    }
    match family {
        "cholesky" => {
            let mut x = Vec::new();
            tlinalg::cholesky::cholesky(Op::Cholesky, op.view(), &mut x, Parallel::Sequential)
                .map_err(|e| e.to_string())?;
            Ok(Output::One(x))
        }
        "qr" => {
            let (mut q, mut r) = (Vec::new(), Vec::new());
            tlinalg::qr::qr(Op::Qr, a.view(), &mut q, &mut r, Parallel::Sequential)
                .map_err(|e| e.to_string())?;
            Ok(Output::Two(q, r))
        }
        "svd_thin" | "svd_full" => {
            let (mut u, mut s, mut v) = (Vec::new(), Vec::new(), Vec::new());
            tlinalg::svd::svd(
                Op::Svd,
                a.view(),
                family == "svd_full",
                &mut u,
                &mut s,
                &mut v,
                Parallel::Sequential,
            )
            .map_err(|e| e.to_string())?;
            Ok(Output::Three(u, s, v))
        }
        "svd_values" => {
            let (mut u, mut s, mut v) = (Vec::new(), Vec::new(), Vec::new());
            tlinalg::svd::svd(
                Op::Svd,
                a.view(),
                false,
                &mut u,
                &mut s,
                &mut v,
                Parallel::Sequential,
            )
            .map_err(|e| e.to_string())?;
            Ok(Output::One(s))
        }
        "eigh" => {
            let (mut w, mut v) = (Vec::new(), Vec::new());
            tlinalg::eigh::eigh(Op::Eigh, op.view(), &mut w, &mut v, Parallel::Sequential)
                .map_err(|e| e.to_string())?;
            Ok(Output::Two(w, v))
        }
        "solve" => {
            let dims = [c.m, 4, c.batch];
            let st = [1, c.m as isize, (c.m * 4) as isize];
            let mut x = vec![T::default(); c.m * 4 * c.batch];
            let dst = strided_view::RawStridedMut::new(&mut x, &dims, &st, 0).unwrap();
            tlinalg::lu::solve(
                Op::Solve,
                a.view(),
                Some(b.view()),
                dst,
                false,
                Parallel::Sequential,
            )
            .map_err(|e| e.to_string())?;
            Ok(Output::One(x))
        }
        _ => {
            call::<T>(family, c, Parallel::Sequential, false)?;
            Err("output path unavailable for this family".into())
        }
    }
}

/// Execute one provider and retain every buffer needed by the correctness gate.
///
/// This path is deliberately separate from the timed path: verification may allocate and inspect
/// outputs, while the measured rows only call the provider on prepared buffers.
pub(crate) fn output_call_checked<T: BenchScalar + EigBuffers + RealValues>(
    family: &str,
    c: Case,
    lapack: bool,
) -> Result<Output<T>, String> {
    let a = if matches!(family, "eig" | "eigvals") {
        Batch::new(c.m, c.n, c.batch, |item| {
            (0..c.m * c.n)
                .map(|i| {
                    let row = i % c.m;
                    let col = i / c.m;
                    if row == col {
                        T::from_c64(Complex64::new(2.0 + row as f64 + item as f64 * 0.01, 0.0))
                    } else {
                        T::default()
                    }
                })
                .collect()
        })
    } else {
        Batch::<T>::general(c.m, c.n, c.batch)
    };
    let b = Batch::<T>::general(c.m, 4, c.batch);
    let op = if matches!(family, "cholesky" | "eigh" | "eigvalsh") {
        Batch::<T>::hpd(c.m, c.batch)
    } else {
        Batch::<T>::general(c.m, c.n, c.batch)
    };
    if lapack {
        #[cfg(any(
            feature = "link-openblas",
            feature = "link-openblas-static",
            feature = "link-mkl"
        ))]
        {
            let mut ws = RecyclingWorkspace::default();
            macro_rules! packed {
                ($call:expr) => {{
                    let mut lu = a.data.clone();
                    let mut piv = vec![0; c.m * c.batch];
                    let mut parity = vec![T::default(); c.batch];
                    $call(&mut lu, &mut piv, &mut parity).map_err(|e| e.to_string())?;
                    Ok(Output::PackedLu { lu, piv, parity })
                }};
            }
            return match family {
                "lu_factor" => packed!(|lu, piv, parity| tlinalg_blas::lu::lu_factor(
                    tlinalg_blas::Op::LuFactor,
                    c.m,
                    c.n,
                    lu,
                    piv,
                    parity
                )),
                "lu_solve_prepared" => {
                    let mut lu = a.data.clone();
                    let mut piv = vec![0; c.m * c.batch];
                    let mut parity = vec![T::default(); c.batch];
                    tlinalg_blas::lu::lu_factor(
                        tlinalg_blas::Op::LuFactor,
                        c.m,
                        c.n,
                        &mut lu,
                        &mut piv,
                        &mut parity,
                    )
                    .map_err(|e| e.to_string())?;
                    let mut x = b.data.clone();
                    tlinalg_blas::lu::lu_solve_prepared(
                        tlinalg_blas::Op::LuSolvePrepared,
                        c.m,
                        4,
                        &lu,
                        &piv,
                        &mut x,
                        false,
                        false,
                    )
                    .map_err(|e| e.to_string())?;
                    Ok(Output::One(x))
                }
                "lu_factor_solve" => {
                    let mut lu = a.data.clone();
                    let mut piv = vec![0; c.m * c.batch];
                    let mut x = b.data.clone();
                    tlinalg_blas::lu::lu_factor_solve(
                        tlinalg_blas::Op::LuFactorSolve,
                        c.m,
                        4,
                        &mut lu,
                        &mut piv,
                        &mut x,
                    )
                    .map_err(|e| e.to_string())?;
                    Ok(Output::One(x))
                }
                "solve" => {
                    let mut x = Vec::new();
                    tlinalg_blas::solve::solve(
                        tlinalg_blas::Op::Solve,
                        false,
                        a.view(),
                        b.view(),
                        &mut x,
                        &mut ws,
                    )
                    .map_err(|e| e.to_string())?;
                    Ok(Output::One(x))
                }
                "triangular_solve" => {
                    let mut x = Vec::new();
                    tlinalg_blas::triangular_solve::triangular_solve(
                        tlinalg_blas::Op::TriangularSolve,
                        tlinalg_blas::triangular_solve::TriangularSolveOptions {
                            left_side: true,
                            lower: true,
                            transpose_a: false,
                            unit_diagonal: false,
                        },
                        a.view(),
                        b.view(),
                        &mut x,
                        &mut ws,
                    )
                    .map_err(|e| e.to_string())?;
                    Ok(Output::One(x))
                }
                "lu" => {
                    let (mut p, mut l, mut u, mut parity) =
                        (Vec::new(), Vec::new(), Vec::new(), Vec::new());
                    tlinalg_blas::lu::lu(
                        tlinalg_blas::Op::Lu,
                        a.view(),
                        tlinalg_blas::lu::LuOutputs {
                            p: &mut p,
                            l: &mut l,
                            u: &mut u,
                            parity: &mut parity,
                        },
                        &mut ws,
                    )
                    .map_err(|e| e.to_string())?;
                    Ok(Output::Lu { p, l, u, parity })
                }
                "cholesky" => {
                    let mut l = Vec::new();
                    tlinalg_blas::cholesky::cholesky(
                        tlinalg_blas::Op::Cholesky,
                        op.view(),
                        &mut l,
                        &mut ws,
                    )
                    .map_err(|e| e.to_string())?;
                    Ok(Output::One(l))
                }
                "qr" => {
                    let (mut q, mut r) = (Vec::new(), Vec::new());
                    tlinalg_blas::qr::qr(tlinalg_blas::Op::Qr, a.view(), &mut q, &mut r, &mut ws)
                        .map_err(|e| e.to_string())?;
                    Ok(Output::Two(q, r))
                }
                "rank_revealing_qr" => {
                    let (mut q, mut r, mut p) = (Vec::new(), Vec::new(), Vec::new());
                    tlinalg_blas::qr::rank_revealing_qr(
                        tlinalg_blas::Op::RankRevealingQr,
                        a.view(),
                        tlinalg_blas::qr::RankRevealingQrOutputs {
                            q: &mut q,
                            r: &mut r,
                            permutation: &mut p,
                        },
                        &mut ws,
                    )
                    .map_err(|e| e.to_string())?;
                    Ok(Output::TwoPerm(q, r, p))
                }
                "svd_thin" | "svd_full" => {
                    let (mut s, mut u, mut vt) = (Vec::new(), Vec::new(), Vec::new());
                    tlinalg_blas::svd::svd(
                        tlinalg_blas::Op::Svd,
                        if family == "svd_full" {
                            tlinalg_blas::svd::SvdMode::Full
                        } else {
                            tlinalg_blas::svd::SvdMode::Thin
                        },
                        a.view(),
                        tlinalg_blas::svd::SvdOutputs {
                            s: &mut s,
                            u: &mut u,
                            vt: &mut vt,
                        },
                        &mut ws,
                    )
                    .map_err(|e| e.to_string())?;
                    Ok(Output::ThreeReal(
                        u,
                        s.into_iter().map(T::real_to_f64).collect(),
                        vt,
                    ))
                }
                "svd_values" => {
                    let (mut s, mut u, mut vt) = (Vec::new(), Vec::new(), Vec::new());
                    tlinalg_blas::svd::svd(
                        tlinalg_blas::Op::SvdValues,
                        tlinalg_blas::svd::SvdMode::Values,
                        a.view(),
                        tlinalg_blas::svd::SvdOutputs {
                            s: &mut s,
                            u: &mut u,
                            vt: &mut vt,
                        },
                        &mut ws,
                    )
                    .map_err(|e| e.to_string())?;
                    Ok(Output::One(
                        s.into_iter()
                            .map(|x| T::from_c64(Complex64::new(T::real_to_f64(x), 0.)))
                            .collect(),
                    ))
                }
                "eigh" => {
                    let (mut w, mut v) = (Vec::new(), Vec::new());
                    tlinalg_blas::eigh::eigh(
                        tlinalg_blas::Op::Eigh,
                        op.view(),
                        &mut w,
                        Some(&mut v),
                        &mut ws,
                    )
                    .map_err(|e| e.to_string())?;
                    Ok(Output::TwoReal(
                        w.into_iter().map(T::real_to_f64).collect(),
                        v,
                    ))
                }
                "eigvalsh" => Ok(Output::TwoReal(
                    <T as RealValues>::lapack_values(&op)?,
                    Vec::new(),
                )),
                "eig" | "eigvals" => {
                    let mut w = Vec::new();
                    let mut v = Vec::new();
                    if family == "eig" {
                        <T as EigBuffers>::lapack_eig(&a, &mut w, Some(&mut v))?;
                    } else {
                        <T as EigBuffers>::lapack_eig_values(&a, &mut w)?;
                    }
                    Ok(Output::Eig(w, v))
                }
                "full_piv_lu" => {
                    let (mut p, mut l, mut u, mut q, mut parity) =
                        (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new());
                    tlinalg_blas::full_piv_lu::full_piv_lu(
                        tlinalg_blas::Op::FullPivLu,
                        a.view(),
                        tlinalg_blas::full_piv_lu::FullPivLuOutputs {
                            p: &mut p,
                            l: &mut l,
                            u: &mut u,
                            q: &mut q,
                            parity: &mut parity,
                        },
                        &mut ws,
                    )
                    .map_err(|e| e.to_string())?;
                    Ok(Output::FullLu { p, l, u, q, parity })
                }
                "full_piv_lu_solve" => {
                    let mut x = Vec::new();
                    tlinalg_blas::full_piv_lu::full_piv_lu_solve(
                        tlinalg_blas::Op::FullPivLuSolve,
                        false,
                        a.view(),
                        b.view(),
                        &mut x,
                        &mut ws,
                    )
                    .map_err(|e| e.to_string())?;
                    Ok(Output::One(x))
                }
                "householder_factor" | "householder_apply" => {
                    let mut state = a.data.clone();
                    let mut tau = Vec::new();
                    tlinalg_blas::householder::factor(
                        tlinalg_blas::Op::HouseholderQr,
                        c.m,
                        c.n,
                        &mut state,
                        &mut tau,
                        &mut ws,
                    )
                    .map_err(|e| e.to_string())?;
                    let target = {
                        let k = c.m.min(c.n);
                        let mut target = vec![T::default(); c.m * k * c.batch];
                        for batch in 0..c.batch {
                            for j in 0..k {
                                target[batch * c.m * k + j * c.m + j] = T::parity(false);
                            }
                        }
                        tlinalg_blas::householder::apply_reflectors(
                            tlinalg_blas::Op::HouseholderQrQColumns,
                            c.m,
                            c.n,
                            k,
                            k,
                            false,
                            &state,
                            &tau,
                            &mut target,
                            &mut ws,
                        )
                        .map_err(|e| e.to_string())?;
                        Some(target)
                    };
                    Ok(Output::Householder {
                        a: state,
                        tau,
                        target,
                    })
                }
                _ => Err(format!("unknown family {family}")),
            };
        }
        #[cfg(not(any(
            feature = "link-openblas",
            feature = "link-openblas-static",
            feature = "link-mkl"
        )))]
        {
            return Err("lapack vendor not linked".into());
        }
    }
    macro_rules! packed_faer {
        ($call:expr) => {{
            let mut lu = a.data.clone();
            let mut piv = vec![0; c.m * c.batch];
            let mut parity = vec![T::default(); c.batch];
            $call(&mut lu, &mut piv, &mut parity).map_err(|e| e.to_string())?;
            Ok(Output::PackedLu { lu, piv, parity })
        }};
    }
    match family {
        "lu_factor" => packed_faer!(|lu, piv, parity| tlinalg::packed_lu::factor(
            Op::LuFactor,
            c.m,
            c.n,
            lu,
            piv,
            parity,
            Parallel::Sequential
        )),
        "lu_solve_prepared" => {
            let mut lu = a.data.clone();
            let mut piv = vec![0; c.m * c.batch];
            let mut parity = vec![T::default(); c.batch];
            tlinalg::packed_lu::factor(
                Op::LuFactor,
                c.m,
                c.n,
                &mut lu,
                &mut piv,
                &mut parity,
                Parallel::Sequential,
            )
            .map_err(|e| e.to_string())?;
            let dims = [c.m, c.m, c.batch];
            let ps = [1, c.m as isize, (c.m * c.m) as isize];
            let pd = [c.m, c.batch];
            let pss = [1, c.m as isize];
            let mut x = b.data.clone();
            tlinalg::packed_lu::solve_prepared(
                Op::LuSolvePrepared,
                strided_view::RawStridedRef::new(&lu, &dims, &ps, 0).unwrap(),
                strided_view::RawStridedRef::new(&piv, &pd, &pss, 0).unwrap(),
                4,
                &mut x,
                false,
                false,
                Parallel::Sequential,
            )
            .map_err(|e| e.to_string())?;
            Ok(Output::One(x))
        }
        "lu_factor_solve" => {
            let mut lu = a.data.clone();
            let mut piv = vec![0; c.m * c.batch];
            let mut x = b.data.clone();
            tlinalg::packed_lu::factor_solve(
                Op::LuFactorSolve,
                c.m,
                4,
                &mut lu,
                &mut piv,
                &mut x,
                Parallel::Sequential,
            )
            .map_err(|e| e.to_string())?;
            Ok(Output::One(x))
        }
        "solve" => {
            let dims = [c.m, 4, c.batch];
            let st = [1, c.m as isize, (c.m * 4) as isize];
            let mut x = vec![T::default(); c.m * 4 * c.batch];
            let dst = strided_view::RawStridedMut::new(&mut x, &dims, &st, 0).unwrap();
            tlinalg::lu::solve(
                Op::Solve,
                a.view(),
                Some(b.view()),
                dst,
                false,
                Parallel::Sequential,
            )
            .map_err(|e| e.to_string())?;
            Ok(Output::One(x))
        }
        "triangular_solve" => {
            let mut x = Vec::new();
            tlinalg::triangular_solve::triangular_solve(
                Op::TriangularSolve,
                a.view(),
                b.view(),
                tlinalg::triangular_solve::TriangularSolveFlags {
                    left_side: true,
                    lower: true,
                    transpose_a: false,
                    unit_diagonal: false,
                },
                &mut x,
                Parallel::Sequential,
            )
            .map_err(|e| e.to_string())?;
            Ok(Output::One(x))
        }
        "lu" => {
            let (mut p, mut l, mut u, mut parity) =
                (Vec::new(), Vec::new(), Vec::new(), Vec::new());
            tlinalg::lu::lu(
                Op::Lu,
                a.view(),
                tlinalg::lu::LuFactors {
                    p: &mut p,
                    l: &mut l,
                    u: &mut u,
                    parity: &mut parity,
                },
                Parallel::Sequential,
            )
            .map_err(|e| e.to_string())?;
            Ok(Output::Lu { p, l, u, parity })
        }
        "cholesky" => {
            let mut l = Vec::new();
            tlinalg::cholesky::cholesky(Op::Cholesky, op.view(), &mut l, Parallel::Sequential)
                .map_err(|e| e.to_string())?;
            Ok(Output::One(l))
        }
        "qr" => {
            let (mut q, mut r) = (Vec::new(), Vec::new());
            tlinalg::qr::qr(Op::Qr, a.view(), &mut q, &mut r, Parallel::Sequential)
                .map_err(|e| e.to_string())?;
            Ok(Output::Two(q, r))
        }
        "rank_revealing_qr" => {
            let (mut q, mut r, mut p) = (Vec::new(), Vec::new(), Vec::new());
            tlinalg::qr::rank_revealing_qr(
                Op::RankRevealingQr,
                a.view(),
                &mut q,
                &mut r,
                &mut p,
                Parallel::Sequential,
            )
            .map_err(|e| e.to_string())?;
            Ok(Output::TwoPerm(q, r, p))
        }
        "svd_thin" | "svd_full" => {
            let (mut u, mut s, mut vt) = (Vec::new(), Vec::new(), Vec::new());
            tlinalg::svd::svd(
                Op::Svd,
                a.view(),
                family == "svd_full",
                &mut u,
                &mut s,
                &mut vt,
                Parallel::Sequential,
            )
            .map_err(|e| e.to_string())?;
            Ok(Output::Three(u, s, vt))
        }
        "svd_values" => {
            let (mut u, mut s, mut vt) = (Vec::new(), Vec::new(), Vec::new());
            tlinalg::svd::svd(
                Op::Svd,
                a.view(),
                false,
                &mut u,
                &mut s,
                &mut vt,
                Parallel::Sequential,
            )
            .map_err(|e| e.to_string())?;
            Ok(Output::One(s))
        }
        "eigh" => {
            let (mut w, mut v) = (Vec::new(), Vec::new());
            tlinalg::eigh::eigh(Op::Eigh, op.view(), &mut w, &mut v, Parallel::Sequential)
                .map_err(|e| e.to_string())?;
            Ok(Output::Two(w, v))
        }
        "eigvalsh" => Ok(Output::TwoReal(
            <T as RealValues>::faer_values(&op)?,
            Vec::new(),
        )),
        "eig" | "eigvals" => {
            let mut w = Vec::new();
            let mut v = Vec::new();
            if family == "eig" {
                <T as EigBuffers>::faer_eig(&a, &mut w, Some(&mut v))?;
            } else {
                <T as EigBuffers>::faer_eig_values(&a, &mut w)?;
            }
            Ok(Output::Eig(w, v))
        }
        "full_piv_lu" => {
            let (mut p, mut l, mut u, mut q, mut parity) =
                (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new());
            tlinalg::full_piv_lu::full_piv_lu(
                Op::FullPivLu,
                a.view(),
                tlinalg::full_piv_lu::FullPivLuFactors {
                    p: &mut p,
                    l: &mut l,
                    u: &mut u,
                    q: &mut q,
                    parity: &mut parity,
                },
                Parallel::Sequential,
            )
            .map_err(|e| e.to_string())?;
            Ok(Output::FullLu { p, l, u, q, parity })
        }
        "full_piv_lu_solve" => {
            let mut x = Vec::new();
            tlinalg::full_piv_lu::full_piv_lu_solve(
                Op::FullPivLuSolve,
                a.view(),
                b.view(),
                false,
                &mut x,
                Parallel::Sequential,
            )
            .map_err(|e| e.to_string())?;
            Ok(Output::One(x))
        }
        "householder_factor" | "householder_apply" => {
            let mut state = a.data.clone();
            let mut tau = Vec::new();
            tlinalg::householder::compact_factor(
                Op::HouseholderQr,
                c.m,
                c.n,
                c.batch,
                &mut state,
                &mut tau,
                Parallel::Sequential,
            )
            .map_err(|e| e.to_string())?;
            let target = {
                let k = c.m.min(c.n);
                let mut z = vec![T::default(); c.m * k * c.batch];
                for batch in 0..c.batch {
                    for j in 0..k {
                        z[batch * c.m * k + j * c.m + j] = T::parity(false);
                    }
                }
                let shape = tlinalg::householder::ReflectorShape {
                    rows: c.m,
                    a_cols: c.n,
                    cols: k,
                    k,
                };
                tlinalg::householder::apply_reflectors(
                    Op::HouseholderQrQColumns,
                    shape,
                    c.batch,
                    &state,
                    &tau,
                    &mut z,
                    false,
                    Parallel::Sequential,
                )
                .map_err(|e| e.to_string())?;
                Some(z)
            };
            Ok(Output::Householder {
                a: state,
                tau,
                target,
            })
        }
        _ => Err(format!("unknown family {family}")),
    }
}

/// Reusable fixture and output storage for one provider row.
struct Prepared<T: BenchScalar> {
    a: Batch<T>,
    b: Batch<T>,
    op: Batch<T>,
    x: Vec<T>,
    rhs: Vec<T>,
    out: Vec<T>,
    out2: Vec<T>,
    out3: Vec<T>,
    out4: Vec<T>,
    real_out: Vec<f64>,
    piv: Vec<i32>,
    parity: Vec<T>,
    tau: Vec<T>,
    perm: Vec<i64>,
    target: Vec<T>,
    target_initial: Vec<T>,
    #[cfg(any(
        feature = "link-openblas",
        feature = "link-openblas-static",
        feature = "link-mkl"
    ))]
    ws: RecyclingWorkspace,
}

impl<T: BenchScalar + LapackEigh> Prepared<T> {
    fn new(c: Case, family: &str) -> Self {
        let a = Batch::<T>::general(c.m, c.n, c.batch);
        let b = Batch::<T>::general(c.m, 4, c.batch);
        let op = match family {
            "cholesky" | "eigh" | "eigvalsh" => Batch::<T>::hpd(c.m, c.batch),
            _ => Batch::<T>::general(c.m, c.n, c.batch),
        };
        let target_initial = Batch::<T>::general(c.m, c.m.min(c.n), c.batch).data;
        let mut out = Vec::new();
        if family == "solve" {
            out.resize(c.m * 4 * c.batch, T::default());
        }
        Self {
            x: a.data.clone(),
            a,
            b,
            op,
            rhs: vec![T::default(); c.m * 4 * c.batch],
            out,
            out2: Vec::new(),
            out3: Vec::new(),
            out4: Vec::new(),
            real_out: Vec::new(),
            piv: vec![0; c.m * c.batch],
            parity: vec![T::default(); c.batch],
            tau: Vec::new(),
            perm: vec![0; c.n * c.batch],
            target: target_initial.clone(),
            target_initial,
            #[cfg(any(
                feature = "link-openblas",
                feature = "link-openblas-static",
                feature = "link-mkl"
            ))]
            ws: RecyclingWorkspace::default(),
        }
    }

    fn reset(&mut self, family: &str) {
        if matches!(
            family,
            "lu_factor" | "lu_factor_solve" | "householder_factor"
        ) {
            self.x.copy_from_slice(&self.a.data);
        }
        if matches!(
            family,
            "lu_solve_prepared"
                | "lu_factor_solve"
                | "solve"
                | "full_piv_lu_solve"
                | "triangular_solve"
        ) {
            self.rhs.copy_from_slice(&self.b.data);
        }
        if family == "householder_apply" {
            self.target.copy_from_slice(&self.target_initial);
        }
        if family != "solve" {
            self.out.clear();
        }
        for v in [&mut self.out2, &mut self.out3, &mut self.out4] {
            v.clear();
        }
        self.real_out.clear();
        if family != "householder_apply" {
            self.tau.clear();
        }
    }

    fn setup(&mut self, family: &str, c: Case) -> Result<(), String> {
        match family {
            "lu_solve_prepared" => tlinalg::packed_lu::factor(
                Op::LuFactor,
                c.m,
                c.n,
                &mut self.x,
                &mut self.piv,
                &mut self.parity,
                Parallel::Sequential,
            )
            .map_err(|e| e.to_string()),
            "householder_apply" => tlinalg::householder::compact_factor(
                Op::HouseholderQr,
                c.m,
                c.n,
                c.batch,
                &mut self.x,
                &mut self.tau,
                Parallel::Sequential,
            )
            .map_err(|e| e.to_string()),
            _ => Ok(()),
        }
    }

    #[cfg(any(
        feature = "link-openblas",
        feature = "link-openblas-static",
        feature = "link-mkl"
    ))]
    fn lapack(&mut self, family: &str, c: Case) -> Result<(), String> {
        if family == "eigh" {
            return <T as LapackEigh>::lapack_eigh(
                &self.op,
                &mut self.real_out,
                Some(&mut self.out),
            );
        }
        if family == "eigvalsh" {
            return <T as LapackEigh>::lapack_eigh(&self.op, &mut self.real_out, None);
        }
        let result = match family {
            "lu_factor" => tlinalg_blas::lu::lu_factor(
                tlinalg_blas::Op::LuFactor,
                c.m,
                c.n,
                &mut self.x,
                &mut self.piv,
                &mut self.parity,
            ),
            "lu_solve_prepared" => tlinalg_blas::lu::lu_solve_prepared(
                tlinalg_blas::Op::LuSolvePrepared,
                c.m,
                4,
                &self.x,
                &self.piv,
                &mut self.rhs,
                false,
                false,
            ),
            "lu_factor_solve" => tlinalg_blas::lu::lu_factor_solve(
                tlinalg_blas::Op::LuFactorSolve,
                c.m,
                4,
                &mut self.x,
                &mut self.piv,
                &mut self.rhs,
            ),
            "solve" => tlinalg_blas::solve::solve(
                tlinalg_blas::Op::Solve,
                false,
                self.a.view(),
                self.b.view(),
                &mut self.out,
                &mut self.ws,
            ),
            "triangular_solve" => tlinalg_blas::triangular_solve::triangular_solve(
                tlinalg_blas::Op::TriangularSolve,
                tlinalg_blas::triangular_solve::TriangularSolveOptions {
                    left_side: true,
                    lower: true,
                    transpose_a: false,
                    unit_diagonal: false,
                },
                self.a.view(),
                self.b.view(),
                &mut self.out,
                &mut self.ws,
            ),
            "lu" => tlinalg_blas::lu::lu(
                tlinalg_blas::Op::Lu,
                self.a.view(),
                tlinalg_blas::lu::LuOutputs {
                    p: &mut self.out,
                    l: &mut self.out2,
                    u: &mut self.out3,
                    parity: &mut self.parity,
                },
                &mut self.ws,
            ),
            "cholesky" => tlinalg_blas::cholesky::cholesky(
                tlinalg_blas::Op::Cholesky,
                self.op.view(),
                &mut self.out,
                &mut self.ws,
            ),
            "qr" => tlinalg_blas::qr::qr(
                tlinalg_blas::Op::Qr,
                self.a.view(),
                &mut self.out,
                &mut self.out2,
                &mut self.ws,
            ),
            "rank_revealing_qr" => tlinalg_blas::qr::rank_revealing_qr(
                tlinalg_blas::Op::RankRevealingQr,
                self.a.view(),
                tlinalg_blas::qr::RankRevealingQrOutputs {
                    q: &mut self.out,
                    r: &mut self.out2,
                    permutation: &mut self.perm,
                },
                &mut self.ws,
            ),
            "svd_thin" | "svd_full" | "svd_values" => {
                return call::<T>(family, c, Parallel::Sequential, true)
            }
            "eigh" | "eigvalsh" => unreachable!(),
            "eig" | "eigvals" => return call::<T>(family, c, Parallel::Sequential, true),
            "full_piv_lu" => tlinalg_blas::full_piv_lu::full_piv_lu(
                tlinalg_blas::Op::FullPivLu,
                self.a.view(),
                tlinalg_blas::full_piv_lu::FullPivLuOutputs {
                    p: &mut self.out,
                    l: &mut self.out2,
                    u: &mut self.out3,
                    q: &mut self.out4,
                    parity: &mut self.parity,
                },
                &mut self.ws,
            ),
            "full_piv_lu_solve" => tlinalg_blas::full_piv_lu::full_piv_lu_solve(
                tlinalg_blas::Op::FullPivLuSolve,
                false,
                self.a.view(),
                self.b.view(),
                &mut self.out,
                &mut self.ws,
            ),
            "householder_factor" => tlinalg_blas::householder::factor(
                tlinalg_blas::Op::HouseholderQr,
                c.m,
                c.n,
                &mut self.x,
                &mut self.tau,
                &mut self.ws,
            ),
            "householder_apply" => tlinalg_blas::householder::apply_reflectors(
                tlinalg_blas::Op::HouseholderQrQColumns,
                c.m,
                c.n,
                c.m.min(c.n),
                c.m.min(c.n),
                false,
                &self.x,
                &self.tau,
                &mut self.target,
                &mut self.ws,
            ),
            _ => return Err(format!("unknown family {family}")),
        };
        result.map_err(|e| e.to_string())
    }

    fn faer(&mut self, family: &str, c: Case, par: Parallel<'_>) -> Result<(), String> {
        match family {
            "lu_factor" => tlinalg::packed_lu::factor(
                Op::LuFactor,
                c.m,
                c.n,
                &mut self.x,
                &mut self.piv,
                &mut self.parity,
                par,
            ),
            "lu_solve_prepared" => {
                let dims = [c.m, c.m, c.batch];
                let ps = [1, c.m as isize, (c.m * c.m) as isize];
                let pd = [c.m, c.batch];
                let pss = [1, c.m as isize];
                tlinalg::packed_lu::solve_prepared(
                    Op::LuSolvePrepared,
                    strided_view::RawStridedRef::new(&self.x, &dims, &ps, 0).unwrap(),
                    strided_view::RawStridedRef::new(&self.piv, &pd, &pss, 0).unwrap(),
                    4,
                    &mut self.rhs,
                    false,
                    false,
                    par,
                )
            }
            "lu_factor_solve" => tlinalg::packed_lu::factor_solve(
                Op::LuFactorSolve,
                c.m,
                4,
                &mut self.x,
                &mut self.piv,
                &mut self.rhs,
                par,
            ),
            "solve" => {
                let dims = [c.m, 4, c.batch];
                let st = [1, c.m as isize, (c.m * 4) as isize];
                let dst = strided_view::RawStridedMut::new(&mut self.out, &dims, &st, 0).unwrap();
                tlinalg::lu::solve(
                    Op::Solve,
                    self.a.view(),
                    Some(self.b.view()),
                    dst,
                    false,
                    par,
                )
            }
            "lu" => tlinalg::lu::lu(
                Op::Lu,
                self.a.view(),
                tlinalg::lu::LuFactors {
                    p: &mut self.out,
                    l: &mut self.out2,
                    u: &mut self.out3,
                    parity: &mut self.parity,
                },
                par,
            ),
            "cholesky" => {
                tlinalg::cholesky::cholesky(Op::Cholesky, self.op.view(), &mut self.out, par)
            }
            "triangular_solve" => tlinalg::triangular_solve::triangular_solve(
                Op::TriangularSolve,
                self.a.view(),
                self.b.view(),
                tlinalg::triangular_solve::TriangularSolveFlags {
                    left_side: true,
                    lower: true,
                    transpose_a: false,
                    unit_diagonal: false,
                },
                &mut self.out,
                par,
            ),
            "qr" => tlinalg::qr::qr(Op::Qr, self.a.view(), &mut self.out, &mut self.out2, par),
            "rank_revealing_qr" => tlinalg::qr::rank_revealing_qr(
                Op::RankRevealingQr,
                self.a.view(),
                &mut self.out,
                &mut self.out2,
                &mut self.perm,
                par,
            ),
            "svd_thin" | "svd_full" => tlinalg::svd::svd(
                Op::Svd,
                self.a.view(),
                family == "svd_full",
                &mut self.out,
                &mut self.out2,
                &mut self.out3,
                par,
            ),
            "svd_values" => return call::<T>("svd_values", c, par, false),
            "eigh" => {
                tlinalg::eigh::eigh(Op::Eigh, self.op.view(), &mut self.out2, &mut self.out, par)
            }
            "eigvalsh" => return call::<T>("eigvalsh", c, par, false),
            "eig" => return call::<T>("eig", c, par, false),
            "eigvals" => return call::<T>("eigvals", c, par, false),
            "full_piv_lu" => tlinalg::full_piv_lu::full_piv_lu(
                Op::FullPivLu,
                self.a.view(),
                tlinalg::full_piv_lu::FullPivLuFactors {
                    p: &mut self.out,
                    l: &mut self.out2,
                    u: &mut self.out3,
                    q: &mut self.out4,
                    parity: &mut self.parity,
                },
                par,
            ),
            "full_piv_lu_solve" => tlinalg::full_piv_lu::full_piv_lu_solve(
                Op::FullPivLuSolve,
                self.a.view(),
                self.b.view(),
                false,
                &mut self.out,
                par,
            ),
            "householder_factor" => tlinalg::householder::compact_factor(
                Op::HouseholderQr,
                c.m,
                c.n,
                c.batch,
                &mut self.x,
                &mut self.tau,
                par,
            ),
            "householder_apply" => {
                let shape = tlinalg::householder::ReflectorShape {
                    rows: c.m,
                    a_cols: c.n,
                    cols: c.m.min(c.n),
                    k: c.m.min(c.n),
                };
                tlinalg::householder::apply_reflectors(
                    Op::HouseholderQrQColumns,
                    shape,
                    c.batch,
                    &self.x,
                    &self.tau,
                    &mut self.target,
                    false,
                    par,
                )
            }
            _ => return Err(format!("unknown family {family}")),
        }
        .map_err(|e| e.to_string())
    }
}

/// Measure all requested provider rows for a case.
pub(crate) fn measure_case<T: BenchScalar + LapackEigh>(
    family: &str,
    c: Case,
    env: &Env,
    reps: usize,
    prime_ms: u64,
    regime: &str,
) -> Vec<Record> {
    let mut rows = Vec::new();
    if !applicable(family, c) {
        for row in ["faer-1lane", "faer-pool", crate::vendor::LAPACK_ROW] {
            if row == "faer-pool" && env.threads() == 1
                || row == crate::vendor::LAPACK_ROW && !crate::vendor::LINKED
            {
                continue;
            }
            rows.push(Record {
                regime: regime.into(),
                family: family.into(),
                dtype: T::LABEL.into(),
                m: c.m,
                n: c.n,
                batch: c.batch,
                row: row.into(),
                threads: env.threads(),
                total_ms: 0.,
                per_item_us: 0.,
                status: "skipped".into(),
                note: "shape not applicable".into(),
            });
        }
        return rows;
    }
    let record = |row: &str, result: Result<timing::Timing, String>| match result {
        Ok(t) => Record {
            regime: regime.into(),
            family: family.into(),
            dtype: T::LABEL.into(),
            m: c.m,
            n: c.n,
            batch: c.batch,
            row: row.into(),
            threads: env.threads(),
            total_ms: t.best_ms,
            per_item_us: t.best_ms * 1000. / c.batch as f64,
            status: "ok".into(),
            note: String::new(),
        },
        Err(e) => Record {
            regime: regime.into(),
            family: family.into(),
            dtype: T::LABEL.into(),
            m: c.m,
            n: c.n,
            batch: c.batch,
            row: row.into(),
            threads: env.threads(),
            total_ms: 0.,
            per_item_us: 0.,
            status: "failed".into(),
            note: e,
        },
    };
    let prepared = RefCell::new(Prepared::<T>::new(c, family));
    if let Err(error) = prepared.borrow_mut().setup(family, c) {
        return vec![record("faer-1lane", Err(error))];
    }
    let result = timing::measure(
        prime_ms,
        reps,
        || {
            prepared.borrow_mut().reset(family);
            crate::vendor::require_threads(env.threads())
        },
        || prepared.borrow_mut().faer(family, c, Parallel::Sequential),
    );
    rows.push(record("faer-1lane", result));
    if env.threads() > 1 {
        let result = timing::measure(
            prime_ms,
            reps,
            || {
                prepared.borrow_mut().reset(family);
                crate::vendor::require_threads(env.threads())
            },
            || prepared.borrow_mut().faer(family, c, env.par()),
        );
        rows.push(record("faer-pool", result));
    }
    if crate::vendor::LINKED {
        let result = timing::measure(
            prime_ms,
            reps,
            || {
                prepared.borrow_mut().reset(family);
                crate::vendor::require_threads(env.threads())
            },
            || {
                #[cfg(any(
                    feature = "link-openblas",
                    feature = "link-openblas-static",
                    feature = "link-mkl"
                ))]
                {
                    prepared.borrow_mut().lapack(family, c)
                }
                #[cfg(not(any(
                    feature = "link-openblas",
                    feature = "link-openblas-static",
                    feature = "link-mkl"
                )))]
                {
                    call::<T>(family, c, Parallel::Sequential, true)
                }
            },
        );
        rows.push(record(crate::vendor::LAPACK_ROW, result));
    }
    rows
}

#[cfg(any())]
/// Legacy compatibility entry point.
fn measure_case_legacy<T: BenchScalar>(
    family: &str,
    c: Case,
    env: &Env,
    reps: usize,
    prime_ms: u64,
    regime: &str,
) -> Vec<Record> {
    let mut rows = Vec::new();
    if !applicable(family, c) {
        let mut names = vec!["faer-1lane", "lapack-openblas"];
        if env.threads() > 1 {
            names.insert(1, "faer-pooled");
        }
        for name in names {
            if name == "lapack-openblas" && !crate::vendor::LINKED {
                continue;
            }
            rows.push(Record {
                regime: regime.into(),
                family: family.into(),
                dtype: T::LABEL.into(),
                m: c.m,
                n: c.n,
                batch: c.batch,
                row: if name == "faer-pooled" {
                    format!("faer-{}t", env.threads())
                } else {
                    name.into()
                },
                threads: env.threads(),
                total_ms: 0.,
                per_item_us: 0.,
                status: "skipped".into(),
                note: "shape not applicable".into(),
            });
        }
        return rows;
    }
    for name in ["faer-1lane", "faer-pooled"] {
        if name == "faer-pooled" && env.threads() == 1 {
            continue;
        }
        let par = if name == "faer-1lane" {
            Parallel::Sequential
        } else {
            env.par()
        };
        let result = timing::measure(
            prime_ms,
            reps,
            || crate::vendor::require_threads(env.threads()),
            || call::<T>(family, c, par, false),
        );
        match result {
            Ok(t) => rows.push(Record {
                regime: regime.into(),
                family: family.into(),
                dtype: T::LABEL.into(),
                m: c.m,
                n: c.n,
                batch: c.batch,
                row: if name == "faer-pooled" {
                    format!("faer-{}t", env.threads())
                } else {
                    name.into()
                },
                threads: env.threads(),
                total_ms: t.best_ms,
                per_item_us: t.best_ms * 1000. / c.batch as f64,
                status: "ok".into(),
                note: String::new(),
            }),
            Err(e) => rows.push(Record {
                regime: regime.into(),
                family: family.into(),
                dtype: T::LABEL.into(),
                m: c.m,
                n: c.n,
                batch: c.batch,
                row: name.into(),
                threads: env.threads(),
                total_ms: 0.,
                per_item_us: 0.,
                status: "failed".into(),
                note: e,
            }),
        }
    }
    if crate::vendor::LINKED {
        let result = timing::measure(
            prime_ms,
            reps,
            || crate::vendor::require_threads(env.threads()),
            || call::<T>(family, c, Parallel::Sequential, true),
        );
        match result {
            Ok(t) => rows.push(Record {
                regime: regime.into(),
                family: family.into(),
                dtype: T::LABEL.into(),
                m: c.m,
                n: c.n,
                batch: c.batch,
                row: "lapack-openblas".into(),
                threads: env.threads(),
                total_ms: t.best_ms,
                per_item_us: t.best_ms * 1000. / c.batch as f64,
                status: "ok".into(),
                note: String::new(),
            }),
            Err(e) => rows.push(Record {
                regime: regime.into(),
                family: family.into(),
                dtype: T::LABEL.into(),
                m: c.m,
                n: c.n,
                batch: c.batch,
                row: "lapack-openblas".into(),
                threads: env.threads(),
                total_ms: 0.,
                per_item_us: 0.,
                status: "failed".into(),
                note: e,
            }),
        }
    }
    rows
}
