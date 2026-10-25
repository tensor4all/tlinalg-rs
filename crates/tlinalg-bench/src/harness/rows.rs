//! Provider call rows and CSV records.

use crate::harness::cases::{applicable, Case};
use crate::harness::timing;
use crate::RecyclingWorkspace;
use crate::{Batch, BenchScalar, Env};
use num_complex::Complex64;
use std::cell::RefCell;
use tcontract_exec::Exec;
use tlinalg::Op;

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
    /// Every measured call, in the order it happened.
    pub samples_ms: Vec<f64>,
    /// Best or median of `samples_ms`, per the requested statistic.
    pub total_ms: f64,
    /// Timing divided by batch.
    pub per_item_us: f64,
    /// `ok`, `skipped`, or `failed`.
    pub status: String,
    /// Diagnostic note.
    pub note: String,
    /// Threads the harness's own pool ran with, observed in this process.
    pub pool_threads: usize,
    /// Threads the vendor library reported for this row.
    pub vendor_threads: usize,
}

impl Record {
    pub fn csv(&self) -> String {
        // The chronological calls are a JSON list, quoted because it contains commas; the campaign's
        // evidence check parses them back. A row that was not measured carries an empty field rather
        // than an empty list, because a skipped row must not look as if it had been timed.
        let samples = if self.samples_ms.is_empty() {
            String::new()
        } else {
            format!(
                "[{}]",
                self.samples_ms
                    .iter()
                    .map(|v| format!("{v:.6}"))
                    .collect::<Vec<_>>()
                    .join(", ")
            )
        };
        format!(
            "{},{},{},{},{},{},{},{},{:.6},{:.6},{},{},\"{}\",{},{}",
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
            self.note.replace(',', ";"),
            samples,
            self.pool_threads,
            self.vendor_threads
        )
    }
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
                tlinalg::eig::eig(Op::Eig, a.view(), &mut w, &mut v, &mut Exec::serial())
                    .map_err(|e| e.to_string())?;
                *values = w;
                if let Some(dst) = vectors {
                    *dst = v;
                }
                Ok(())
            }
            fn faer_eig_values(a: &Batch<Self>, values: &mut Vec<Complex64>) -> Result<(), String> {
                tlinalg::eig::eig_values(Op::EigValues, a.view(), values, &mut Exec::serial())
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
    /// Only the vendor arm calls this, and that path is compiled out when no vendor library is
    /// linked; the bound stays unconditional so the generic code keeps one shape. The workspace
    /// is the row's prepared one: a fresh `RecyclingWorkspace` per timed call would time its
    /// allocation, like the fixture rebuilds this harness was fixed for.
    #[allow(dead_code)]
    fn lapack_eigh(
        a: &Batch<Self>,
        values: &mut Vec<f64>,
        vectors: Option<&mut Vec<Self>>,
        ws: &mut RecyclingWorkspace,
    ) -> Result<(), String>;
}
macro_rules! impl_lapack_eigh {
    ($scalar:ty) => {
        impl LapackEigh for $scalar {
            fn lapack_eigh(
                a: &Batch<Self>,
                values: &mut Vec<f64>,
                vectors: Option<&mut Vec<Self>>,
                ws: &mut RecyclingWorkspace,
            ) -> Result<(), String> {
                #[cfg(any(
                    feature = "link-openblas",
                    feature = "link-openblas-static",
                    feature = "link-mkl"
                ))]
                {
                    tlinalg_blas::eigh::eigh(tlinalg_blas::Op::Eigh, a.view(), values, vectors, ws)
                        .map_err(|e| e.to_string())
                }
                #[cfg(not(any(
                    feature = "link-openblas",
                    feature = "link-openblas-static",
                    feature = "link-mkl"
                )))]
                {
                    let _ = (a, values, vectors, ws);
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
                tlinalg::eigh::eigh_values(Op::EighValues, a.view(), &mut w, &mut Exec::serial())
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
            &mut Exec::serial()
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
                &mut Exec::serial(),
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
                &mut Exec::serial(),
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
                &mut Exec::serial(),
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
                &mut Exec::serial(),
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
                &mut Exec::serial(),
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
                &mut Exec::serial(),
            )
            .map_err(|e| e.to_string())?;
            Ok(Output::Lu { p, l, u, parity })
        }
        "cholesky" => {
            let mut l = Vec::new();
            tlinalg::cholesky::cholesky(Op::Cholesky, op.view(), &mut l, &mut Exec::serial())
                .map_err(|e| e.to_string())?;
            Ok(Output::One(l))
        }
        "qr" => {
            let (mut q, mut r) = (Vec::new(), Vec::new());
            tlinalg::qr::qr(Op::Qr, a.view(), &mut q, &mut r, &mut Exec::serial())
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
                &mut Exec::serial(),
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
                &mut Exec::serial(),
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
                &mut Exec::serial(),
            )
            .map_err(|e| e.to_string())?;
            Ok(Output::One(s))
        }
        "eigh" => {
            let (mut w, mut v) = (Vec::new(), Vec::new());
            tlinalg::eigh::eigh(Op::Eigh, op.view(), &mut w, &mut v, &mut Exec::serial())
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
                &mut Exec::serial(),
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
                &mut Exec::serial(),
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
                &mut Exec::serial(),
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
                    &mut Exec::serial(),
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
    /// Reusable vendor singular-value output, through the vendor's own public projection.
    #[cfg(any(
        feature = "link-openblas",
        feature = "link-openblas-static",
        feature = "link-mkl"
    ))]
    v_real: Vec<<T as tlinalg_blas::symbols::Symbols>::Real>,
    #[cfg(any(
        feature = "link-openblas",
        feature = "link-openblas-static",
        feature = "link-mkl"
    ))]
    v_complex: Vec<<T as tlinalg_blas::symbols::Symbols>::Complex>,
    /// Second complex output for `eig`'s eigenvectors.
    #[cfg(any(
        feature = "link-openblas",
        feature = "link-openblas-static",
        feature = "link-mkl"
    ))]
    v_complex2: Vec<<T as tlinalg_blas::symbols::Symbols>::Complex>,
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
            #[cfg(any(
                feature = "link-openblas",
                feature = "link-openblas-static",
                feature = "link-mkl"
            ))]
            v_real: Vec::new(),
            #[cfg(any(
                feature = "link-openblas",
                feature = "link-openblas-static",
                feature = "link-mkl"
            ))]
            v_complex: Vec::new(),
            #[cfg(any(
                feature = "link-openblas",
                feature = "link-openblas-static",
                feature = "link-mkl"
            ))]
            v_complex2: Vec::new(),
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
        #[cfg(any(
            feature = "link-openblas",
            feature = "link-openblas-static",
            feature = "link-mkl"
        ))]
        {
            self.v_real.clear();
            self.v_complex.clear();
            self.v_complex2.clear();
        }
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
                &mut Exec::serial(),
            )
            .map_err(|e| e.to_string()),
            "householder_apply" => tlinalg::householder::compact_factor(
                Op::HouseholderQr,
                c.m,
                c.n,
                c.batch,
                &mut self.x,
                &mut self.tau,
                &mut Exec::serial(),
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
                &mut self.ws,
            );
        }
        if family == "eigvalsh" {
            return <T as LapackEigh>::lapack_eigh(
                &self.op,
                &mut self.real_out,
                None,
                &mut self.ws,
            );
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
            // The prepared input and workspace, like every other row here. `u`/`vt` are `Vec<T>`
            // and use `out`/`out2`; the singular values and `eig`'s outputs are `T::Real`/
            // `T::Complex`, prepared through the vendor's own public projection (the faer side
            // cannot name those types and keeps them local).
            "svd_thin" | "svd_full" => tlinalg_blas::svd::svd(
                tlinalg_blas::Op::Svd,
                if family == "svd_full" {
                    tlinalg_blas::svd::SvdMode::Full
                } else {
                    tlinalg_blas::svd::SvdMode::Thin
                },
                self.op.view(),
                tlinalg_blas::svd::SvdOutputs {
                    s: &mut self.v_real,
                    u: &mut self.out,
                    vt: &mut self.out2,
                },
                &mut self.ws,
            ),
            "svd_values" => tlinalg_blas::svd::svd(
                tlinalg_blas::Op::SvdValues,
                tlinalg_blas::svd::SvdMode::Values,
                self.op.view(),
                tlinalg_blas::svd::SvdOutputs {
                    s: &mut self.v_real,
                    u: &mut self.out,
                    vt: &mut self.out2,
                },
                &mut self.ws,
            ),
            "eigh" | "eigvalsh" => unreachable!(),
            "eig" => tlinalg_blas::eig::eig(
                tlinalg_blas::Op::Eig,
                self.op.view(),
                &mut self.v_complex,
                Some(&mut self.v_complex2),
                &mut self.ws,
            ),
            "eigvals" => tlinalg_blas::eig::eig(
                tlinalg_blas::Op::Eig,
                self.op.view(),
                &mut self.v_complex,
                None,
                &mut self.ws,
            ),
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

    fn faer(&mut self, family: &str, c: Case, par: &mut Exec<'_>) -> Result<(), String> {
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
            // These rows take the *prepared* input, like every other family here. Their values
            // are returned in the real or complex type rather than in `T`, so the output vectors
            // stay local to the call; the input is the expensive fixture (`hpd` builds it with an
            // n^3 product), and rebuilding it inside the timed region measured the fixture rather
            // than the solver.
            "svd_values" => {
                let mut s = Vec::new();
                tlinalg::svd::svd_values(Op::SvdValues, self.op.view(), &mut s, par)
            }
            "eigh" => {
                tlinalg::eigh::eigh(Op::Eigh, self.op.view(), &mut self.out2, &mut self.out, par)
            }
            "eigvalsh" => {
                let mut w = Vec::new();
                tlinalg::eigh::eigh_values(Op::EighValues, self.op.view(), &mut w, par)
            }
            "eig" => {
                let (mut w, mut v) = (Vec::new(), Vec::new());
                tlinalg::eig::eig(Op::Eig, self.op.view(), &mut w, &mut v, par)
            }
            "eigvals" => {
                let mut w = Vec::new();
                tlinalg::eig::eig_values(Op::EigValues, self.op.view(), &mut w, par)
            }
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

/// The provider rows a case measures, with whether the shape applies at all.
///
/// The campaign's verification census must name exactly the cells its timing run will emit, so both
/// paths ask this one function. Every declared row is listed: `faer-1lane` always, `faer-pool`
/// whenever the campaign declares it (a 1-thread run reports it as a declared skip rather than
/// dropping the cell), and the vendor row whenever a vendor library is linked.
pub(crate) fn plan(family: &str, c: Case, env: &Env) -> Vec<(String, bool, String)> {
    let mut rows = Vec::new();
    for row in ["faer-1lane", "faer-pool", crate::vendor::LAPACK_ROW] {
        if row == crate::vendor::LAPACK_ROW && !crate::vendor::LINKED {
            continue;
        }
        let (applies, note) = if row == "faer-pool" && env.threads() == 1 {
            (
                false,
                "a 1-thread run does not measure the pool row".to_owned(),
            )
        } else if applicable(family, c) {
            (true, String::new())
        } else {
            (false, "shape not applicable".to_owned())
        };
        rows.push((row.to_owned(), applies, note));
    }
    rows
}

/// Measure all requested provider rows for a case.
pub(crate) fn measure_case<T: BenchScalar + LapackEigh>(
    family: &str,
    c: Case,
    env: &Env,
    reps: usize,
    prime_ms: u64,
    regime: &str,
    statistic: timing::Statistic,
) -> Vec<Record> {
    let mut rows = Vec::new();
    let planned = plan(family, c, env);
    if !applicable(family, c) {
        for (row, _, note) in planned.clone() {
            rows.push(Record {
                regime: regime.into(),
                family: family.into(),
                dtype: T::LABEL.into(),
                m: c.m,
                n: c.n,
                batch: c.batch,
                row,
                threads: env.threads(),
                samples_ms: Vec::new(),
                total_ms: 0.,
                per_item_us: 0.,
                status: "skipped".into(),
                note,
                pool_threads: env.threads(),
                vendor_threads: env.threads(),
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
            per_item_us: t.total_ms * 1000. / c.batch as f64,
            total_ms: t.total_ms,
            samples_ms: t.samples_ms,
            status: "ok".into(),
            note: String::new(),
            pool_threads: env.threads(),
            vendor_threads: env.threads(),
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
            samples_ms: Vec::new(),
            total_ms: 0.,
            per_item_us: 0.,
            status: "failed".into(),
            note: e,
            pool_threads: env.threads(),
            vendor_threads: env.threads(),
        },
    };
    let prepared = RefCell::new(Prepared::<T>::new(c, family));
    if let Err(error) = prepared.borrow_mut().setup(family, c) {
        return vec![record("faer-1lane", Err(error))];
    }
    let result = timing::measure(
        prime_ms,
        reps,
        statistic,
        || {
            prepared.borrow_mut().reset(family);
            crate::vendor::require_threads(env.threads())
        },
        || prepared.borrow_mut().faer(family, c, &mut Exec::serial()),
    );
    rows.push(record("faer-1lane", result));
    if env.threads() > 1 {
        let result = timing::measure(
            prime_ms,
            reps,
            statistic,
            || {
                prepared.borrow_mut().reset(family);
                crate::vendor::require_threads(env.threads())
            },
            || prepared.borrow_mut().faer(family, c, &mut env.par()),
        );
        rows.push(record("faer-pool", result));
    }
    if crate::vendor::LINKED {
        let result = timing::measure(
            prime_ms,
            reps,
            statistic,
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
                    // This arm only compiles in a build with no vendor library, where `LINKED` is
                    // false and the enclosing block never executes.
                    unreachable!()
                }
            },
        );
        rows.push(record(crate::vendor::LAPACK_ROW, result));
    }
    // Declared rows this process start does not measure are recorded as skips, so the campaign's
    // census and the page name the cell instead of silently losing it.
    for (row, applies, note) in planned {
        if !applies {
            rows.push(Record {
                regime: regime.into(),
                family: family.into(),
                dtype: T::LABEL.into(),
                m: c.m,
                n: c.n,
                batch: c.batch,
                row,
                threads: env.threads(),
                samples_ms: Vec::new(),
                total_ms: 0.,
                per_item_us: 0.,
                status: "skipped".into(),
                note,
                pool_threads: env.threads(),
                vendor_threads: env.threads(),
            });
        }
    }
    rows
}
