//! Behavioural tests for the batched families moved from tenferro-linalg's LAPACK backend.
//!
//! Requires a linked LAPACK: run with `--features link-openblas` (and with `provider-inject` too).
//! Every per-family check runs for `f32`, `f64`, `Complex32` and `Complex64` on a compact batch of
//! two; the layout checks (B = 0/1, non-mergeable and broadcast batch axes, mid-batch failure,
//! aliased outputs) run on `f64`.

#![cfg(feature = "link-openblas")]
#![allow(clippy::needless_range_loop)]

mod common;

use common::*;
use num_complex::{Complex32, Complex64};
use tlinalg_blas::cholesky::cholesky;
use tlinalg_blas::eig::eig;
use tlinalg_blas::eigh::eigh;
use tlinalg_blas::full_piv_lu::{full_piv_lu, full_piv_lu_solve, FullPivLuOutputs};
use tlinalg_blas::lu::{lu, LuOutputs};
use tlinalg_blas::qr::{qr, rank_revealing_qr, RankRevealingQrOutputs};
use tlinalg_blas::solve::{solve, solve_into};
use tlinalg_blas::triangular_solve::{triangular_solve, TriangularSolveOptions};
use tlinalg_blas::{householder, Error, NonFiniteRole, Op};

macro_rules! for_each_scalar {
    ($($name:ident => $body:ident),* $(,)?) => {
        $(
            mod $name {
                use super::*;
                #[test]
                fn f32() { $body::<f32>(); }
                #[test]
                fn f64() { $body::<f64>(); }
                #[test]
                fn c32() { $body::<Complex32>(); }
                #[test]
                fn c64() { $body::<Complex64>(); }
            }
        )*
    };
}

for_each_scalar!(
    cholesky_reconstructs => check_cholesky,
    triangular_solve_solves => check_triangular_solve,
    lu_reconstructs => check_lu,
    full_piv_lu_reconstructs => check_full_piv_lu,
    full_piv_lu_solve_solves => check_full_piv_lu_solve,
    solve_solves => check_solve,
    solve_into_writes_a_strided_output => check_solve_into,
    householder_factor_and_apply => check_householder,
    qr_reconstructs => check_qr,
    rank_revealing_qr_reconstructs => check_rrqr,
    eigh_diagonalizes => check_eigh,
    eig_diagonalizes => check_eig,
);

const BATCH: usize = 2;

/// A Hermitian positive definite `n x n` matrix: `M Mᴴ + n I`.
fn hpd<T: TestScalar>(n: usize, seed: usize) -> Vec<T> {
    let m = widen(&matrix::<T>(n, n, seed));
    let mut a = matmul(&m, &adjoint(&m, n, n), n, n, n);
    for i in 0..n {
        a[i + i * n] += Complex64::new(n as f64, 0.0);
    }
    a.into_iter().map(T::from_c64).collect()
}

fn hpd_batch<T: TestScalar>(n: usize, count: usize) -> Vec<T> {
    (0..count).flat_map(|seed| hpd::<T>(n, seed)).collect()
}

fn item<T: Copy>(data: &[T], len: usize, index: usize) -> &[T] {
    &data[index * len..(index + 1) * len]
}

fn check_cholesky<T: TestScalar>() {
    let n = 5;
    let a = Strided::compact(hpd_batch::<T>(n, BATCH), n, n, Some(BATCH));
    let mut l = Vec::new();
    let mut ws = TestWorkspace::default();
    cholesky(Op::Cholesky, a.r(), &mut l, &mut ws).unwrap();
    assert_eq!(ws.outstanding(), 0, "the factor copy is released");
    // The caller's output is the destructive work matrix, so no workspace copy is acquired.
    assert_eq!(ws.acquired, 0, "the output is the factor storage");
    for index in 0..BATCH {
        let li = widen(item(&l, n * n, index));
        for col in 0..n {
            for row in 0..col {
                assert_eq!(
                    li[row + col * n],
                    Complex64::new(0.0, 0.0),
                    "upper triangle is zero"
                );
            }
        }
        assert_close(
            &matmul(&li, &adjoint(&li, n, n), n, n, n),
            &widen(item(&a.data, n * n, index)),
            T::LOOSE_TOL,
            "L Lᴴ",
        );
    }

    // Not positive definite in the middle of the batch: the error, and nothing left behind.
    let mut data = hpd_batch::<T>(n, 3);
    for i in 0..n {
        data[n * n + i + i * n] = T::from_c64(Complex64::new(-1.0, 0.0));
    }
    let bad = Strided::compact(data, n, n, Some(3));
    let mut out = vec![T::default(); 3];
    assert_eq!(
        cholesky(Op::Cholesky, bad.r(), &mut out, &mut ws),
        Err(Error::NonConvergence { op: Op::Cholesky })
    );
    assert!(out.is_empty());
}

/// Keep only the requested triangle, with a unit diagonal when asked.
fn triangle(a: &[Complex64], n: usize, lower: bool, unit: bool) -> Vec<Complex64> {
    let mut out = vec![Complex64::new(0.0, 0.0); n * n];
    for col in 0..n {
        for row in 0..n {
            let keep = if lower { row >= col } else { row <= col };
            if keep {
                out[row + col * n] = a[row + col * n];
            }
        }
        if unit {
            out[col + col * n] = Complex64::new(1.0, 0.0);
        }
    }
    out
}

fn check_triangular_solve<T: TestScalar>() {
    let n = 4;
    let other = 3;
    for left_side in [true, false] {
        for lower in [true, false] {
            for transpose_a in [true, false] {
                for unit_diagonal in [true, false] {
                    let options = TriangularSolveOptions {
                        left_side,
                        lower,
                        transpose_a,
                        unit_diagonal,
                    };
                    let (rows, cols) = if left_side { (n, other) } else { (other, n) };
                    let a = Strided::compact(batch_of::<T>(n, n, BATCH, 0), n, n, Some(BATCH));
                    let b = Strided::compact(
                        batch_of::<T>(rows, cols, BATCH, 3),
                        rows,
                        cols,
                        Some(BATCH),
                    );
                    let mut x = Vec::new();
                    let mut ws = TestWorkspace::default();
                    triangular_solve(Op::TriangularSolve, options, a.r(), b.r(), &mut x, &mut ws)
                        .unwrap();
                    assert_eq!(ws.acquired, 0, "a compact triangle is read in place");
                    for index in 0..BATCH {
                        let t =
                            triangle(&widen(item(&a.data, n * n, index)), n, lower, unit_diagonal);
                        let op_a = if transpose_a { transpose(&t, n, n) } else { t };
                        let xi = widen(item(&x, rows * cols, index));
                        let product = if left_side {
                            matmul(&op_a, &xi, n, n, cols)
                        } else {
                            matmul(&xi, &op_a, rows, n, n)
                        };
                        assert_close(
                            &product,
                            &widen(item(&b.data, rows * cols, index)),
                            T::LOOSE_TOL,
                            &format!("{options:?} item {index}"),
                        );
                    }
                }
            }
        }
    }

    // A zero diagonal on the right-side, non-unit route is reported before BLAS divides by it.
    let mut data = batch_of::<T>(n, n, BATCH, 0);
    data[n * n + 1 + n] = T::default();
    let a = Strided::compact(data, n, n, Some(BATCH));
    let b = Strided::compact(batch_of::<T>(other, n, BATCH, 1), other, n, Some(BATCH));
    let options = TriangularSolveOptions {
        left_side: false,
        lower: true,
        ..Default::default()
    };
    let mut x = Vec::new();
    assert_eq!(
        triangular_solve(
            Op::TriangularSolve,
            options,
            a.r(),
            b.r(),
            &mut x,
            &mut TestWorkspace::default()
        ),
        Err(Error::Singular {
            op: Op::TriangularSolve
        })
    );
    assert!(x.is_empty());
}

fn check_lu<T: TestScalar>() {
    for (m, n) in [(4usize, 4usize), (5, 3), (3, 5), (1, 1)] {
        let k = m.min(n);
        let a = Strided::compact(batch_of::<T>(m, n, BATCH, 2), m, n, Some(BATCH));
        let (mut p, mut l, mut u, mut parity) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        let mut ws = TestWorkspace::default();
        lu(
            Op::Lu,
            a.r(),
            LuOutputs {
                p: &mut p,
                l: &mut l,
                u: &mut u,
                parity: &mut parity,
            },
            &mut ws,
        )
        .unwrap();
        assert_eq!(ws.outstanding(), 0);
        assert_eq!(
            (p.len(), l.len(), u.len(), parity.len()),
            (m * m * BATCH, m * k * BATCH, k * n * BATCH, BATCH)
        );
        for index in 0..BATCH {
            let (pi, li, ui) = (
                widen(item(&p, m * m, index)),
                widen(item(&l, m * k, index)),
                widen(item(&u, k * n, index)),
            );
            assert_close(
                &matmul(&pi, &widen(item(&a.data, m * n, index)), m, m, n),
                &matmul(&li, &ui, m, k, n),
                T::LOOSE_TOL,
                "P A = L U",
            );
            let sign = parity[index].to_c64();
            assert!(sign == Complex64::new(1.0, 0.0) || sign == Complex64::new(-1.0, 0.0));
        }
    }
    // Exactly singular input is not an error.
    let zero = Strided::compact(vec![T::default(); 9], 3, 3, None);
    let (mut p, mut l, mut u, mut parity) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    lu(
        Op::Lu,
        zero.r(),
        LuOutputs {
            p: &mut p,
            l: &mut l,
            u: &mut u,
            parity: &mut parity,
        },
        &mut TestWorkspace::default(),
    )
    .unwrap();
}

fn permutation_sign(p: &[Complex64], n: usize) -> f64 {
    let perm: Vec<usize> = (0..n)
        .map(|row| (0..n).find(|&col| p[row + col * n].re == 1.0).unwrap())
        .collect();
    let mut seen = vec![false; n];
    let mut sign = 1.0;
    for start in 0..n {
        let mut len = 0;
        let mut i = start;
        while !seen[i] {
            seen[i] = true;
            i = perm[i];
            len += 1;
        }
        if len > 0 && len % 2 == 0 {
            sign = -sign;
        }
    }
    sign
}

fn check_full_piv_lu<T: TestScalar>() {
    let n = 4;
    let a = Strided::compact(batch_of::<T>(n, n, BATCH, 3), n, n, Some(BATCH));
    let (mut p, mut l, mut u, mut q, mut parity) =
        (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new());
    let mut ws = TestWorkspace::default();
    full_piv_lu(
        Op::FullPivLu,
        a.r(),
        FullPivLuOutputs {
            p: &mut p,
            l: &mut l,
            u: &mut u,
            q: &mut q,
            parity: &mut parity,
        },
        &mut ws,
    )
    .unwrap();
    assert_eq!(ws.outstanding(), 0);
    let len = n * n;
    for index in 0..BATCH {
        let (pi, li, ui, qi) = (
            widen(item(&p, len, index)),
            widen(item(&l, len, index)),
            widen(item(&u, len, index)),
            widen(item(&q, len, index)),
        );
        let paq = matmul(
            &matmul(&pi, &widen(item(&a.data, len, index)), n, n, n),
            &transpose(&qi, n, n),
            n,
            n,
            n,
        );
        assert_close(
            &paq,
            &matmul(&li, &ui, n, n, n),
            T::LOOSE_TOL,
            "P A Qᵀ = L U",
        );
        let expected = permutation_sign(&pi, n) * permutation_sign(&qi, n);
        assert_eq!(parity[index].to_c64(), Complex64::new(expected, 0.0));
    }
    let zero = Strided::compact(vec![T::default(); n * n], n, n, None);
    let (mut p, mut l, mut u, mut q, mut parity) =
        (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new());
    assert_eq!(
        full_piv_lu(
            Op::FullPivLu,
            zero.r(),
            FullPivLuOutputs {
                p: &mut p,
                l: &mut l,
                u: &mut u,
                q: &mut q,
                parity: &mut parity,
            },
            &mut ws,
        ),
        Err(Error::Singular { op: Op::FullPivLu })
    );
    assert!(p.is_empty() && parity.is_empty());
}

fn check_full_piv_lu_solve<T: TestScalar>() {
    let (n, nrhs) = (4, 2);
    for transpose_a in [false, true] {
        let a = Strided::compact(batch_of::<T>(n, n, BATCH, 0), n, n, Some(BATCH));
        let b = Strided::compact(batch_of::<T>(n, nrhs, BATCH, 5), n, nrhs, Some(BATCH));
        let mut x = Vec::new();
        let mut ws = TestWorkspace::default();
        full_piv_lu_solve(
            Op::FullPivLuSolve,
            transpose_a,
            a.r(),
            b.r(),
            &mut x,
            &mut ws,
        )
        .unwrap();
        assert_eq!(ws.outstanding(), 0, "the batch scratch is released");
        for index in 0..BATCH {
            let ai = widen(item(&a.data, n * n, index));
            let op_a = if transpose_a {
                transpose(&ai, n, n)
            } else {
                ai
            };
            assert_close(
                &matmul(&op_a, &widen(item(&x, n * nrhs, index)), n, n, nrhs),
                &widen(item(&b.data, n * nrhs, index)),
                T::LOOSE_TOL,
                "full-pivot residual",
            );
        }
    }
}

fn check_solve<T: TestScalar>() {
    let (n, nrhs) = (5, 3);
    for transpose_a in [false, true] {
        let a = Strided::compact(batch_of::<T>(n, n, BATCH, 0), n, n, Some(BATCH));
        let b = Strided::compact(batch_of::<T>(n, nrhs, BATCH, 7), n, nrhs, Some(BATCH));
        let mut x = Vec::new();
        let mut ws = TestWorkspace::default();
        solve(Op::Solve, transpose_a, a.r(), b.r(), &mut x, &mut ws).unwrap();
        assert_eq!(ws.acquired, 2, "one LU copy and one pivot buffer per call");
        for index in 0..BATCH {
            let ai = widen(item(&a.data, n * n, index));
            let op_a = if transpose_a {
                transpose(&ai, n, n)
            } else {
                ai
            };
            assert_close(
                &matmul(&op_a, &widen(item(&x, n * nrhs, index)), n, n, nrhs),
                &widen(item(&b.data, n * nrhs, index)),
                T::LOOSE_TOL,
                "solve residual",
            );
        }
    }
}

fn check_solve_into<T: TestScalar>() {
    let (n, nrhs, ldb) = (4, 2, 6);
    let a = Strided::compact(batch_of::<T>(n, n, BATCH, 4), n, n, Some(BATCH));
    let b = Strided::compact(batch_of::<T>(n, nrhs, BATCH, 9), n, nrhs, Some(BATCH));
    // A strided output: column `j` of item `i` starts at `j * ldb + i * ldb * nrhs`; the padding
    // between columns must survive.
    let sentinel = T::from_c64(Complex64::new(42.0, 0.0));
    let mut out = Strided {
        data: vec![sentinel; ldb * nrhs * BATCH],
        dims: vec![n, nrhs, BATCH],
        strides: vec![1, ldb as isize, (ldb * nrhs) as isize],
    };
    solve_into(
        Op::Solve,
        false,
        a.r(),
        b.r(),
        out.m(),
        &mut TestWorkspace::default(),
    )
    .unwrap();
    for index in 0..BATCH {
        let mut x = Vec::new();
        for col in 0..nrhs {
            let start = index * ldb * nrhs + col * ldb;
            x.extend_from_slice(&out.data[start..start + n]);
            assert!(out.data[start + n..start + ldb]
                .iter()
                .all(|&v| v == sentinel));
        }
        assert_close(
            &matmul(&widen(item(&a.data, n * n, index)), &widen(&x), n, n, nrhs),
            &widen(item(&b.data, n * nrhs, index)),
            T::LOOSE_TOL,
            "strided solve",
        );
    }
}

fn check_householder<T: TestScalar>() {
    for (m, n) in [(5usize, 3usize), (3, 5), (4, 4)] {
        let k = m.min(n);
        let a = batch_of::<T>(m, n, BATCH, 6);
        let mut packed = a.clone();
        let mut tau = Vec::new();
        let mut ws = TestWorkspace::default();
        householder::factor(Op::HouseholderFactor, m, n, &mut packed, &mut tau, &mut ws).unwrap();
        assert_eq!(tau.len(), k * BATCH);
        assert_eq!(
            (ws.acquired, ws.released),
            (2, 2),
            "one query slot and one work buffer"
        );

        // Qᴴ A = R: apply the adjoint reflectors to a copy of A.
        let mut qha = a.clone();
        householder::apply_reflectors(
            Op::HouseholderApply,
            m,
            n,
            n,
            k,
            true,
            &packed,
            &tau,
            &mut qha,
            &mut ws,
        )
        .unwrap();
        for index in 0..BATCH {
            let qha = widen(item(&qha, m * n, index));
            let packed64 = widen(item(&packed, m * n, index));
            for col in 0..n {
                for row in 0..m {
                    let expected = if row <= col {
                        packed64[row + col * m]
                    } else {
                        Complex64::new(0.0, 0.0)
                    };
                    assert!(
                        (qha[row + col * m] - expected).norm() <= T::LOOSE_TOL * 10.0,
                        "{m}x{n} QᴴA({row},{col})"
                    );
                }
            }
        }
        // Q (Qᴴ C) = C.
        let c = batch_of::<T>(m, 2, BATCH, 8);
        let mut round = c.clone();
        householder::apply_reflectors(
            Op::HouseholderApply,
            m,
            n,
            2,
            k,
            true,
            &packed,
            &tau,
            &mut round,
            &mut ws,
        )
        .unwrap();
        householder::apply_reflectors(
            Op::HouseholderApply,
            m,
            n,
            2,
            k,
            false,
            &packed,
            &tau,
            &mut round,
            &mut ws,
        )
        .unwrap();
        assert_close(&widen(&round), &widen(&c), T::LOOSE_TOL, "Q Qᴴ C");
    }
    // Empty input: no tau, no workspace traffic.
    let mut ws = TestWorkspace::default();
    let mut tau = vec![T::default()];
    householder::factor::<T, _>(Op::HouseholderFactor, 0, 3, &mut [], &mut tau, &mut ws).unwrap();
    assert!(tau.is_empty());
    assert_eq!(ws.acquired, 0);
    let a = matrix::<T>(3, 2, 0);
    let mut c = matrix::<T>(3, 1, 0);
    assert!(matches!(
        householder::apply_reflectors(
            Op::HouseholderApply,
            3,
            2,
            1,
            3,
            false,
            &a,
            &[T::default(); 3],
            &mut c,
            &mut ws
        ),
        Err(Error::InvalidArgument {
            role: "dimensions",
            ..
        })
    ));
}

fn check_qr<T: TestScalar>() {
    for (m, n) in [(5usize, 3usize), (3, 5), (4, 4)] {
        let k = m.min(n);
        let a = Strided::compact(batch_of::<T>(m, n, BATCH, 0), m, n, Some(BATCH));
        let (mut q, mut r) = (Vec::new(), Vec::new());
        let mut ws = TestWorkspace::default();
        qr(Op::Qr, a.r(), &mut q, &mut r, &mut ws).unwrap();
        assert_eq!(ws.outstanding(), 0);
        assert_eq!((q.len(), r.len()), (m * k * BATCH, k * n * BATCH));
        for index in 0..BATCH {
            let qi = widen(item(&q, m * k, index));
            let ri = widen(item(&r, k * n, index));
            for col in 0..n {
                for row in (col + 1)..k {
                    assert_eq!(ri[row + col * k], Complex64::new(0.0, 0.0));
                }
            }
            assert_close(
                &matmul(&qi, &ri, m, k, n),
                &widen(item(&a.data, m * n, index)),
                T::LOOSE_TOL,
                "QR",
            );
            assert_close(
                &matmul(&adjoint(&qi, m, k), &qi, k, m, k),
                &identity(k),
                T::LOOSE_TOL,
                "QᴴQ",
            );
        }
    }
}

fn check_rrqr<T: TestScalar>() {
    for (m, n) in [(5usize, 3usize), (3, 5), (4, 4)] {
        let k = m.min(n);
        // The middle item is all zero: it gets the identity result without a factorization.
        let mut data = batch_of::<T>(m, n, 3, 1);
        data[m * n..2 * m * n].fill(T::default());
        let a = Strided::compact(data, m, n, Some(3));
        let (mut q, mut r, mut perm) = (Vec::new(), Vec::new(), Vec::new());
        let mut ws = TestWorkspace::default();
        rank_revealing_qr(
            Op::RankRevealingQr,
            a.r(),
            RankRevealingQrOutputs {
                q: &mut q,
                r: &mut r,
                permutation: &mut perm,
            },
            &mut ws,
        )
        .unwrap();
        assert_eq!(ws.outstanding(), 0);
        for index in 0..3 {
            let a64 = widen(item(&a.data, m * n, index));
            let p = item(&perm, n, index);
            let mut ap = Vec::with_capacity(m * n);
            for &col in p {
                ap.extend_from_slice(&a64[col as usize * m..(col as usize + 1) * m]);
            }
            let (qi, ri) = (widen(item(&q, m * k, index)), widen(item(&r, k * n, index)));
            assert_close(&matmul(&qi, &ri, m, k, n), &ap, T::LOOSE_TOL, "A P = Q R");
            for i in 1..k {
                assert!(
                    ri[i + i * k].norm() <= ri[(i - 1) + (i - 1) * k].norm() * (1.0 + T::LOOSE_TOL)
                );
            }
            if index == 1 {
                assert!(ri.iter().all(|v| v.norm() == 0.0));
                assert_eq!(p, (0..n as i64).collect::<Vec<_>>().as_slice());
                assert_close(
                    &matmul(&adjoint(&qi, m, k), &qi, k, m, k),
                    &identity(k),
                    0.0,
                    "identity Q",
                );
            }
        }
    }
    let mut nan = matrix::<T>(2, 3, 0);
    nan[2] = T::from_c64(Complex64::new(f64::NAN, 0.0));
    let nan = Strided::compact(nan, 2, 3, None);
    let (mut q, mut r, mut perm) = (Vec::new(), Vec::new(), Vec::new());
    assert_eq!(
        rank_revealing_qr(
            Op::RankRevealingQr,
            nan.r(),
            RankRevealingQrOutputs {
                q: &mut q,
                r: &mut r,
                permutation: &mut perm,
            },
            &mut TestWorkspace::default(),
        ),
        Err(Error::NonFinite {
            op: Op::RankRevealingQr,
            role: NonFiniteRole::Input
        })
    );
}

fn check_eigh<T: TestScalar>()
where
    T::Real: TestScalar,
{
    let n = 4;
    let a = Strided::compact(hpd_batch::<T>(n, BATCH), n, n, Some(BATCH));
    let (mut values, mut vectors) = (Vec::new(), Vec::new());
    let mut ws = TestWorkspace::default();
    eigh(Op::Eigh, a.r(), &mut values, Some(&mut vectors), &mut ws).unwrap();
    assert_eq!(ws.outstanding(), 0);
    let mut values_only = Vec::new();
    eigh(Op::EighValues, a.r(), &mut values_only, None, &mut ws).unwrap();
    assert_eq!(ws.outstanding(), 0);
    assert_close(
        &widen(&values_only),
        &widen(&values),
        T::LOOSE_TOL * 100.0,
        "values-only agrees",
    );
    for index in 0..BATCH {
        let ai = widen(item(&a.data, n * n, index));
        let vi = widen(item(&vectors, n * n, index));
        let wi = widen(item(&values, n, index));
        for pair in wi.windows(2) {
            assert!(pair[0].re <= pair[1].re);
        }
        let mut vl = vi.clone();
        for col in 0..n {
            for row in 0..n {
                vl[row + col * n] *= wi[col];
            }
        }
        assert_close(
            &matmul(&ai, &vi, n, n, n),
            &vl,
            T::LOOSE_TOL * 10.0,
            "A V = V Λ",
        );
    }
}

fn check_eig<T: TestScalar>()
where
    T::Complex: TestScalar,
{
    // A rotation block gives a real input a complex-conjugate pair.
    let n = 4;
    let mut data = batch_of::<T>(n, n, BATCH, 2);
    for index in 0..BATCH {
        data[index * n * n + 1] = T::from_c64(Complex64::new(-3.0, 0.0));
        data[index * n * n + n] = T::from_c64(Complex64::new(3.0, 0.0));
    }
    let a = Strided::compact(data, n, n, Some(BATCH));
    let (mut values, mut vectors) = (Vec::new(), Vec::new());
    let mut ws = TestWorkspace::default();
    eig(Op::Eig, a.r(), &mut values, Some(&mut vectors), &mut ws).unwrap();
    assert_eq!(ws.outstanding(), 0);
    for index in 0..BATCH {
        let wi = widen(item(&values, n, index));
        let vi = widen(item(&vectors, n * n, index));
        let mut vl = vi.clone();
        for col in 0..n {
            for row in 0..n {
                vl[row + col * n] *= wi[col];
            }
        }
        assert_close(
            &matmul(&widen(item(&a.data, n * n, index)), &vi, n, n, n),
            &vl,
            T::LOOSE_TOL * 10.0,
            "A V = V Λ",
        );
        if !<T as SharedTestScalar>::COMPLEX {
            assert!(
                wi.iter().any(|v| v.im.abs() > 0.1),
                "the rotation block yields a complex pair"
            );
        }
    }
    let mut only = Vec::new();
    eig(Op::EigValues, a.r(), &mut only, None, &mut ws).unwrap();
    assert_close(
        &widen(&only),
        &widen(&values),
        T::LOOSE_TOL * 100.0,
        "values-only agrees",
    );
}

// ---- layout checks (f64) ----

/// `B = 0` (a single rank-2 matrix) and `B = 1` give the same result as one item of a batch.
#[test]
fn rank_two_and_singleton_batches_match_a_batch_item() {
    let n = 4;
    let data = hpd::<f64>(n, 0);
    let mut reference = Vec::new();
    let mut ws = TestWorkspace::default();
    cholesky(
        Op::Cholesky,
        Strided::compact(data.clone(), n, n, None).r(),
        &mut reference,
        &mut ws,
    )
    .unwrap();
    let mut single = Vec::new();
    cholesky(
        Op::Cholesky,
        Strided::compact(data.clone(), n, n, Some(1)).r(),
        &mut single,
        &mut ws,
    )
    .unwrap();
    assert_eq!(reference, single);
    let mut empty = vec![1.0];
    cholesky(
        Op::Cholesky,
        Strided::compact(Vec::new(), n, n, Some(0)).r(),
        &mut empty,
        &mut ws,
    )
    .unwrap();
    assert!(empty.is_empty());
}

/// A non-mergeable batch view (two batch axes with padding between items, and a transposed
/// matrix layout) walks the items in batch order and gathers each one correctly.
#[test]
fn non_mergeable_strided_batches_are_walked_in_order() {
    let n = 3;
    let items: Vec<Vec<f64>> = (0..4).map(|seed| hpd::<f64>(n, seed)).collect();
    // Storage: item (i, j) at offset 20 * i + 50 * j, stored row-major (row stride n, col stride 1);
    // symmetric, so the transposed storage holds the same matrix.
    let mut data = vec![f64::NAN; 200];
    for j in 0..2 {
        for i in 0..2 {
            let base = 20 * i + 50 * j;
            let m = &items[i + 2 * j];
            for col in 0..n {
                for row in 0..n {
                    data[base + row * n + col] = m[row + col * n];
                }
            }
        }
    }
    let view = Strided {
        data,
        dims: vec![n, n, 2, 2],
        strides: vec![n as isize, 1, 20, 50],
    };
    let mut out = Vec::new();
    cholesky(
        Op::Cholesky,
        view.r(),
        &mut out,
        &mut TestWorkspace::default(),
    )
    .unwrap();
    for (index, m) in items.iter().enumerate() {
        let mut expected = Vec::new();
        cholesky(
            Op::Cholesky,
            Strided::compact(m.clone(), n, n, None).r(),
            &mut expected,
            &mut TestWorkspace::default(),
        )
        .unwrap();
        assert_eq!(
            item(&out, n * n, index),
            expected.as_slice(),
            "item {index}"
        );
    }
}

/// One `A` broadcast (stride 0 on the batch axis) against several right-hand sides.
#[test]
fn a_stride_zero_batch_axis_broadcasts_an_input() {
    let (n, nrhs, count) = (4, 2, 3);
    let a = Strided {
        data: matrix::<f64>(n, n, 1),
        dims: vec![n, n, count],
        strides: vec![1, n as isize, 0],
    };
    let b = Strided::compact(batch_of::<f64>(n, nrhs, count, 3), n, nrhs, Some(count));
    let mut x = Vec::new();
    solve(
        Op::Solve,
        false,
        a.r(),
        b.r(),
        &mut x,
        &mut TestWorkspace::default(),
    )
    .unwrap();
    for index in 0..count {
        assert_close(
            &matmul(
                &widen(&a.data),
                &widen(item(&x, n * nrhs, index)),
                n,
                n,
                nrhs,
            ),
            &widen(item(&b.data, n * nrhs, index)),
            1e-9,
            "broadcast solve",
        );
    }
}

/// A failure in the middle of a batch reports that item's error and leaves the outputs empty; the
/// direct-output route leaves the failing and later items untouched.
#[test]
fn a_mid_batch_failure_reports_the_first_failing_item() {
    let (n, count) = (3, 4);
    let mut data = batch_of::<f64>(n, n, count, 0);
    // Items 1 and 3 are singular; item 1 is reported.
    data[n * n..2 * n * n].fill(0.0);
    data[3 * n * n..4 * n * n].fill(0.0);
    let a = Strided::compact(data, n, n, Some(count));
    let b = Strided::compact(batch_of::<f64>(n, 1, count, 5), n, 1, Some(count));
    let mut x = vec![7.0];
    assert_eq!(
        solve(
            Op::Solve,
            false,
            a.r(),
            b.r(),
            &mut x,
            &mut TestWorkspace::default()
        ),
        Err(Error::Singular { op: Op::Solve })
    );
    assert!(x.is_empty());

    let mut out = Strided::compact(vec![-1.0; n * count], n, 1, Some(count));
    assert_eq!(
        solve_into(
            Op::Solve,
            false,
            a.r(),
            b.r(),
            out.m(),
            &mut TestWorkspace::default()
        ),
        Err(Error::Singular { op: Op::Solve })
    );
    assert!(
        out.data[..n].iter().all(|&v| v != -1.0),
        "item 0 was solved"
    );
    assert!(
        out.data[n..].iter().all(|&v| v == -1.0),
        "items 1.. keep their contents"
    );
}

/// A writable descriptor whose batch items alias each other is rejected before any write.
#[test]
fn aliased_outputs_are_rejected() {
    let (n, count) = (3, 2);
    let a = Strided::compact(batch_of::<f64>(n, n, count, 0), n, n, Some(count));
    let b = Strided::compact(batch_of::<f64>(n, 1, count, 5), n, 1, Some(count));
    for strides in [vec![1, n as isize, 0], vec![1, n as isize, 1]] {
        let mut out = Strided {
            data: vec![-1.0; 2 * n],
            dims: vec![n, 1, count],
            strides,
        };
        assert!(matches!(
            solve_into(
                Op::Solve,
                false,
                a.r(),
                b.r(),
                out.m(),
                &mut TestWorkspace::default()
            ),
            Err(Error::InvalidArgument { role: "out", .. })
        ));
        assert!(out.data.iter().all(|&v| v == -1.0));
    }
    // A row stride other than one cannot be handed to `?getrs`.
    let mut out = Strided {
        data: vec![-1.0; 2 * n * count],
        dims: vec![n, 1, count],
        strides: vec![2, (2 * n) as isize, (2 * n) as isize],
    };
    assert!(matches!(
        solve_into(
            Op::Solve,
            false,
            a.r(),
            b.r(),
            out.m(),
            &mut TestWorkspace::default()
        ),
        Err(Error::InvalidArgument { role: "out", .. })
    ));
}

/// Mismatched batch dims between operands are rejected.
#[test]
fn mismatched_batch_dims_are_rejected() {
    let n = 3;
    let a = Strided::compact(batch_of::<f64>(n, n, 2, 0), n, n, Some(2));
    let b = Strided::compact(batch_of::<f64>(n, 1, 3, 5), n, 1, Some(3));
    let mut x = Vec::new();
    assert!(matches!(
        solve(
            Op::Solve,
            false,
            a.r(),
            b.r(),
            &mut x,
            &mut TestWorkspace::default()
        ),
        Err(Error::InvalidArgument { role: "batch", .. })
    ));
}

#[test]
fn householder_ops_report_the_host_kernel_names() {
    assert_eq!(Op::HouseholderFactor.as_str(), "compact_factor_2d");
    assert_eq!(Op::HouseholderApply.as_str(), "apply_reflectors_2d");
}
