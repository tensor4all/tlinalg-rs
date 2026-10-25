//! Cross-provider parity: the same batched cases through `tlinalg` (faer) and `tlinalg-blas`
//! (LAPACK), compared gauge-aware.
//!
//! What is compared directly is what the mathematics makes unique: singular values, Hermitian and
//! general eigenvalues (as a multiset), Cholesky factors, triangular and general solves, and QR
//! factors once both are brought to a positive real `R` diagonal. Factorizations whose factors depend
//! on pivot or phase choices are checked by reconstruction against each provider's documented
//! convention, and agree across providers on the gauge-free part (`|R_jj|`, parity, solutions).
//!
//! Every case runs for `f32`, `f64`, `Complex32` and `Complex64` over batch shapes `[]`, `[3]`
//! (compact) and `[2, 3]` laid out both gapped (non-mergeable) and with transposed batch strides;
//! binary families also run with a stride-0 broadcast coefficient matrix.
//!
//! Requires a linked LAPACK: run with `--features tlinalg-parity/link-openblas`.

#![cfg(feature = "link-openblas")]
#![allow(clippy::needless_range_loop)]

use core::mem::MaybeUninit;

use tlinalg_blas::{IndexWorkspace, Workspace};
use tlinalg_testkit::{
    adjoint, assert_close, batch_buf, broadcast_buf, hermitian_seeded, hpd_seeded, identity,
    is_symmetric, matmul, matrix, pivoting, transpose, widen, BatchBuf, Complex64, Layout,
    TestScalar,
};

/// A plain host-shaped workspace for the LAPACK provider.
#[derive(Default)]
struct Ws;

impl<S: tlinalg_blas::Scalar + Default> Workspace<S> for Ws {
    fn acquire_zeroed(&mut self, len: usize) -> Vec<S> {
        vec![S::default(); len]
    }
    fn acquire_capacity(&mut self, cap: usize) -> Vec<S> {
        Vec::with_capacity(cap)
    }
    fn acquire_uninit(&mut self, len: usize) -> Vec<MaybeUninit<S>> {
        (0..len).map(|_| MaybeUninit::uninit()).collect()
    }
    fn release(&mut self, _buf: Vec<S>) {}
}

impl IndexWorkspace for Ws {
    fn acquire_zeroed_index(&mut self, len: usize) -> Vec<i32> {
        vec![0; len]
    }
    fn release_index(&mut self, _buf: Vec<i32>) {}
}

/// The batch shapes and layouts every unary family runs over.
fn configs() -> Vec<(Vec<usize>, Layout)> {
    vec![
        (vec![], Layout::Compact),
        (vec![3], Layout::Compact),
        (vec![2, 3], Layout::Gapped),
        (vec![2, 3], Layout::Transposed),
    ]
}

/// Item `index` of a compact batch-contiguous output.
fn item<U: Copy>(data: &[U], len: usize, index: usize) -> &[U] {
    &data[index * len..(index + 1) * len]
}

/// Rows `0..k` of a column-major `rows x cols` matrix.
fn top_rows(a: &[Complex64], rows: usize, cols: usize, k: usize) -> Vec<Complex64> {
    let mut out = Vec::with_capacity(k * cols);
    for col in 0..cols {
        for row in 0..k {
            out.push(a[row + col * rows]);
        }
    }
    out
}

/// `diag(d) * a` for a column-major `k x n` matrix.
fn scale_rows(a: &[Complex64], d: &[Complex64], k: usize, n: usize) -> Vec<Complex64> {
    let mut out = a.to_vec();
    for col in 0..n {
        for row in 0..k {
            out[row + col * k] *= d[row];
        }
    }
    out
}

/// `a * diag(d)` for a column-major `m x k` matrix.
fn scale_cols(a: &[Complex64], d: &[Complex64], m: usize, k: usize) -> Vec<Complex64> {
    let mut out = a.to_vec();
    for col in 0..k {
        for row in 0..m {
            out[row + col * m] *= d[col];
        }
    }
    out
}

/// The unit phase of each diagonal entry of a column-major `k x n` matrix (`1` for zero).
fn diag_phases(r: &[Complex64], k: usize) -> Vec<Complex64> {
    (0..k)
        .map(|j| {
            let value = r[j + j * k];
            if value.norm() == 0.0 {
                Complex64::new(1.0, 0.0)
            } else {
                value / value.norm()
            }
        })
        .collect()
}

/// Columns of a column-major `m x n` matrix gathered by `perm`.
fn gather_cols(a: &[Complex64], m: usize, perm: &[i64]) -> Vec<Complex64> {
    let mut out = Vec::with_capacity(m * perm.len());
    for &col in perm {
        out.extend_from_slice(&a[col as usize * m..(col as usize + 1) * m]);
    }
    out
}

/// Sort a multiset of eigenvalues by real then imaginary part, rounding so roundoff-level
/// differences in near-ties do not reorder conjugate pairs.
fn sorted(values: &[Complex64]) -> Vec<Complex64> {
    let mut out = values.to_vec();
    let key = |z: &Complex64| ((z.re * 1e5).round() as i64, (z.im * 1e5).round() as i64);
    out.sort_by_key(key);
    out
}

/// The triangle of `a` that a triangular solve reads, with a unit diagonal when requested.
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

macro_rules! parity_suite {
    ($module:ident, $scalar:ty) => {
        mod $module {
            use super::*;

            type T = $scalar;
            const TOL: f64 = <T as TestScalar>::LOOSE_TOL;
            const SEQ: tcontract_exec::Exec<'static> = &mut tcontract_exec::Exec::serial();

            /// `count` items of `gen(index)` laid out over `dims`.
            fn input(
                rows: usize,
                cols: usize,
                dims: &[usize],
                layout: Layout,
                gen: impl Fn(usize) -> Vec<T>,
            ) -> BatchBuf<T> {
                let count: usize = dims.iter().product();
                let items: Vec<Vec<T>> = (0..count).map(gen).collect();
                batch_buf(&items, rows, cols, dims, layout)
            }

            /// A compact zero output buffer `[rows, cols, dims...]` for in-place destinations.
            fn zeros(rows: usize, cols: usize, dims: &[usize]) -> BatchBuf<T> {
                let count: usize = dims.iter().product();
                let items = vec![vec![T::default(); rows * cols]; count];
                batch_buf(&items, rows, cols, dims, Layout::Compact)
            }

            #[test]
            fn svd() {
                for (m, n) in [(4, 4), (5, 3), (3, 5)] {
                    let k = m.min(n);
                    for (dims, layout) in configs() {
                        let a = input(m, n, &dims, layout, |i| matrix::<T>(m, n, i));
                        let count = a.count();
                        for full in [false, true] {
                            let (uc, vr) = if full { (m, n) } else { (k, k) };
                            let (mut fu, mut fs, mut fvt) = (Vec::new(), Vec::new(), Vec::new());
                            tlinalg::svd::svd(
                                tlinalg::Op::Svd, a.view(), full, &mut fu, &mut fs, &mut fvt, SEQ,
                            )
                            .unwrap();
                            let (mut bs, mut bu, mut bvt) = (Vec::new(), Vec::new(), Vec::new());
                            let mode = if full {
                                tlinalg_blas::svd::SvdMode::Full
                            } else {
                                tlinalg_blas::svd::SvdMode::Thin
                            };
                            tlinalg_blas::svd::svd(
                                tlinalg_blas::Op::Svd,
                                mode,
                                a.view(),
                                tlinalg_blas::svd::SvdOutputs { s: &mut bs, u: &mut bu, vt: &mut bvt },
                                &mut Ws,
                            )
                            .unwrap();
                            assert_eq!(fs.len(), count * k);
                            let what = format!("svd {m}x{n} full={full} {dims:?} {layout:?}");
                            assert_close(&widen(&fs), &widen(&bs), TOL, &format!("{what}: S"));
                            for index in 0..count {
                                let want = widen(&a.item(index));
                                for (u, s, vt, who) in [
                                    (widen(item(&fu, m * uc, index)), widen(item(&fs, k, index)), widen(item(&fvt, vr * n, index)), "faer"),
                                    (widen(item(&bu, m * uc, index)), widen(item(&bs, k, index)), widen(item(&bvt, vr * n, index)), "lapack"),
                                ] {
                                    let us = scale_cols(&u[..m * k], &s, m, k);
                                    let got = matmul(&us, &top_rows(&vt, vr, n, k), m, k, n);
                                    assert_close(&got, &want, TOL, &format!("{what} {who} item {index}: U S Vᴴ"));
                                    let uhu = matmul(&adjoint(&u, m, uc), &u, uc, m, uc);
                                    assert_close(&uhu, &identity(uc), TOL, &format!("{what} {who}: UᴴU"));
                                }
                            }
                        }
                        let mut fs = Vec::new();
                        tlinalg::svd::svd_values(tlinalg::Op::SvdValues, a.view(), &mut fs, SEQ)
                            .unwrap();
                        let (mut bs, mut bu, mut bvt) = (Vec::new(), Vec::new(), Vec::new());
                        tlinalg_blas::svd::svd(
                            tlinalg_blas::Op::SvdValues,
                            tlinalg_blas::svd::SvdMode::Values,
                            a.view(),
                            tlinalg_blas::svd::SvdOutputs { s: &mut bs, u: &mut bu, vt: &mut bvt },
                            &mut Ws,
                        )
                        .unwrap();
                        assert_close(&widen(&fs), &widen(&bs), TOL, &format!("svdvals {m}x{n} {dims:?}"));
                    }
                }
            }

            #[test]
            fn cholesky() {
                let n = 5;
                for (dims, layout) in configs() {
                    let a = input(n, n, &dims, layout, |i| hpd_seeded::<T>(n, i + 1));
                    let mut fl = Vec::new();
                    tlinalg::cholesky::cholesky(tlinalg::Op::Cholesky, a.view(), &mut fl, SEQ)
                        .unwrap();
                    let mut bl = Vec::new();
                    tlinalg_blas::cholesky::cholesky(tlinalg_blas::Op::Cholesky, a.view(), &mut bl, &mut Ws)
                        .unwrap();
                    let what = format!("cholesky {dims:?} {layout:?}");
                    assert_close(&widen(&fl), &widen(&bl), TOL, &what);
                    for index in 0..a.count() {
                        let l = widen(item(&fl, n * n, index));
                        let got = matmul(&l, &adjoint(&l, n, n), n, n, n);
                        assert_close(&got, &widen(&a.item(index)), TOL, &format!("{what}: L Lᴴ"));
                    }
                }
            }

            fn check_triangular(a: &BatchBuf<T>, b: &BatchBuf<T>, n: usize, what: &str) {
                let (rows, cols) = (b.dims[0], b.dims[1]);
                for bits in 0..16u32 {
                    let (left_side, lower, transpose_a, unit_diagonal) =
                        (bits & 1 != 0, bits & 2 != 0, bits & 4 != 0, bits & 8 != 0);
                    if (left_side && rows != n) || (!left_side && cols != n) {
                        continue;
                    }
                    let mut fx = Vec::new();
                    tlinalg::triangular_solve::triangular_solve(
                        tlinalg::Op::TriangularSolve,
                        a.view(),
                        b.view(),
                        tlinalg::triangular_solve::TriangularSolveFlags { left_side, lower, transpose_a, unit_diagonal },
                        &mut fx,
                        SEQ,
                    )
                    .unwrap();
                    let mut bx = Vec::new();
                    tlinalg_blas::triangular_solve::triangular_solve(
                        tlinalg_blas::Op::TriangularSolve,
                        tlinalg_blas::triangular_solve::TriangularSolveOptions { left_side, lower, transpose_a, unit_diagonal },
                        a.view(),
                        b.view(),
                        &mut bx,
                        &mut Ws,
                    )
                    .unwrap();
                    let flags = format!("{what} left={left_side} lower={lower} trans={transpose_a} unit={unit_diagonal}");
                    assert_close(&widen(&fx), &widen(&bx), TOL, &flags);
                    let count: usize = b.dims[2..].iter().product();
                    for index in 0..count {
                        let tri = triangle(&widen(&a.item(index)), n, lower, unit_diagonal);
                        let op_a = if transpose_a { transpose(&tri, n, n) } else { tri };
                        let x = widen(item(&fx, rows * cols, index));
                        let got = if left_side {
                            matmul(&op_a, &x, n, n, cols)
                        } else {
                            matmul(&x, &op_a, rows, n, n)
                        };
                        assert_close(&got, &widen(&b.item(index)), TOL, &format!("{flags}: residual"));
                    }
                }
            }

            #[test]
            fn triangular_solve() {
                let n = 4;
                for (dims, layout) in configs() {
                    let a = input(n, n, &dims, layout, |i| matrix::<T>(n, n, i));
                    let left = input(n, 3, &dims, layout, |i| matrix::<T>(n, 3, i + 7));
                    let right = input(3, n, &dims, layout, |i| matrix::<T>(3, n, i + 9));
                    check_triangular(&a, &left, n, &format!("trsm {dims:?} {layout:?}"));
                    check_triangular(&a, &right, n, &format!("trsm {dims:?} {layout:?}"));
                }
                let a = broadcast_buf(&matrix::<T>(n, n, 3), n, n, &[3]);
                let b = input(n, 2, &[3], Layout::Compact, |i| matrix::<T>(n, 2, i + 4));
                check_triangular(&a, &b, n, "trsm broadcast A");
            }

            #[test]
            fn lu() {
                for (m, n) in [(4, 4), (5, 3), (3, 5)] {
                    let k = m.min(n);
                    for (dims, layout) in configs() {
                        let a = input(m, n, &dims, layout, |i| pivoting::<T>(m, n, i));
                        let (mut fp, mut fl, mut fu, mut fpar) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
                        tlinalg::lu::lu(
                            tlinalg::Op::Lu,
                            a.view(),
                            tlinalg::lu::LuFactors { p: &mut fp, l: &mut fl, u: &mut fu, parity: &mut fpar },
                            SEQ,
                        )
                        .unwrap();
                        let (mut bp, mut bl, mut bu, mut bpar) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
                        tlinalg_blas::lu::lu(
                            tlinalg_blas::Op::Lu,
                            a.view(),
                            tlinalg_blas::lu::LuOutputs { p: &mut bp, l: &mut bl, u: &mut bu, parity: &mut bpar },
                            &mut Ws,
                        )
                        .unwrap();
                        let what = format!("lu {m}x{n} {dims:?} {layout:?}");
                        assert_close(&widen(&fpar), &widen(&bpar), TOL, &format!("{what}: parity"));
                        // Both providers document `P A = L U`, and with well-separated pivots they
                        // choose the same rows, so the factors agree exactly as computed.
                        assert_close(&widen(&fp), &widen(&bp), TOL, &format!("{what}: P"));
                        assert_close(&widen(&fl), &widen(&bl), TOL, &format!("{what}: L"));
                        assert_close(&widen(&fu), &widen(&bu), TOL, &format!("{what}: U"));
                        for index in 0..a.count() {
                            let want = widen(&a.item(index));
                            for (p, l, u, who) in [(&fp, &fl, &fu, "faer"), (&bp, &bl, &bu, "lapack")] {
                                let (p, l, u) = (widen(item(p, m * m, index)), widen(item(l, m * k, index)), widen(item(u, k * n, index)));
                                let got = matmul(&p, &want, m, m, n);
                                assert_close(&got, &matmul(&l, &u, m, k, n), TOL, &format!("{what} {who}: P A = L U"));
                                if m > 2 {
                                    // The generator exists to make the convention observable.
                                    assert!(!is_symmetric(&p, m), "{what}: P is symmetric, so the test cannot tell P from Pᵀ");
                                }
                            }
                        }
                    }
                }
            }

            fn check_solve(a: &BatchBuf<T>, b: &BatchBuf<T>, what: &str) {
                let (n, nrhs) = (b.dims[0], b.dims[1]);
                let batch = &b.dims[2..];
                for transpose_a in [false, true] {
                    let mut out = zeros(n, nrhs, batch);
                    tlinalg::lu::solve(tlinalg::Op::Solve, a.view(), Some(b.view()), out.view_mut(), transpose_a, SEQ)
                        .unwrap();
                    let mut bx = Vec::new();
                    tlinalg_blas::solve::solve(tlinalg_blas::Op::Solve, transpose_a, a.view(), b.view(), &mut bx, &mut Ws)
                        .unwrap();
                    let what = format!("{what} trans={transpose_a}");
                    let count: usize = batch.iter().product();
                    let fsolved: Vec<T> = (0..count).flat_map(|index| out.item(index)).collect();
                    assert_close(&widen(&fsolved), &widen(&bx), TOL, &format!("{what}: lu solve"));
                    let mut fx = Vec::new();
                    tlinalg::full_piv_lu::full_piv_lu_solve(tlinalg::Op::FullPivLuSolve, a.view(), b.view(), transpose_a, &mut fx, SEQ)
                        .unwrap();
                    let mut bfx = Vec::new();
                    tlinalg_blas::full_piv_lu::full_piv_lu_solve(tlinalg_blas::Op::FullPivLuSolve, transpose_a, a.view(), b.view(), &mut bfx, &mut Ws)
                        .unwrap();
                    assert_close(&widen(&fx), &widen(&bfx), TOL, &format!("{what}: full-pivot solve"));
                    assert_close(&widen(&fx), &widen(&bx), TOL, &format!("{what}: full-pivot vs partial-pivot"));
                    for index in 0..count {
                        let a_i = widen(&a.item(index));
                        let op_a = if transpose_a { transpose(&a_i, n, n) } else { a_i };
                        let got = matmul(&op_a, &widen(item(&bx, n * nrhs, index)), n, n, nrhs);
                        assert_close(&got, &widen(&b.item(index)), TOL, &format!("{what}: residual"));
                    }
                }
            }

            #[test]
            fn solve() {
                let n = 4;
                for (dims, layout) in configs() {
                    let a = input(n, n, &dims, layout, |i| pivoting::<T>(n, n, i));
                    let b = input(n, 2, &dims, layout, |i| matrix::<T>(n, 2, i + 5));
                    check_solve(&a, &b, &format!("solve {dims:?} {layout:?}"));
                }
                let a = broadcast_buf(&pivoting::<T>(n, n, 3), n, n, &[3]);
                let b = input(n, 2, &[3], Layout::Compact, |i| matrix::<T>(n, 2, i + 4));
                check_solve(&a, &b, "solve broadcast A");
            }

            #[test]
            fn full_piv_lu() {
                let n = 4;
                for (dims, layout) in configs() {
                    let a = input(n, n, &dims, layout, |i| pivoting::<T>(n, n, i));
                    let (mut fp, mut fl, mut fu, mut fq, mut fpar) = (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new());
                    tlinalg::full_piv_lu::full_piv_lu(
                        tlinalg::Op::FullPivLu,
                        a.view(),
                        tlinalg::full_piv_lu::FullPivLuFactors { p: &mut fp, l: &mut fl, u: &mut fu, q: &mut fq, parity: &mut fpar },
                        SEQ,
                    )
                    .unwrap();
                    let (mut bp, mut bl, mut bu, mut bq, mut bpar) = (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new());
                    tlinalg_blas::full_piv_lu::full_piv_lu(
                        tlinalg_blas::Op::FullPivLu,
                        a.view(),
                        tlinalg_blas::full_piv_lu::FullPivLuOutputs { p: &mut bp, l: &mut bl, u: &mut bu, q: &mut bq, parity: &mut bpar },
                        &mut Ws,
                    )
                    .unwrap();
                    let what = format!("full_piv_lu {dims:?} {layout:?}");
                    assert_close(&widen(&fpar), &widen(&bpar), TOL, &format!("{what}: parity"));
                    for (f, b, name) in [(&fp, &bp, "P"), (&fl, &bl, "L"), (&fu, &bu, "U"), (&fq, &bq, "Q")] {
                        assert_close(&widen(f), &widen(b), TOL, &format!("{what}: {name}"));
                    }
                    let nn = n * n;
                    for index in 0..a.count() {
                        let want = widen(&a.item(index));
                        for (p, l, u, q, who) in [(&fp, &fl, &fu, &fq, "faer"), (&bp, &bl, &bu, &bq, "lapack")] {
                            // Both document `P A Qᵀ = L U`.
                            let (p, l, u, q) = (widen(item(p, nn, index)), widen(item(l, nn, index)), widen(item(u, nn, index)), widen(item(q, nn, index)));
                            let got = matmul(&matmul(&p, &want, n, n, n), &transpose(&q, n, n), n, n, n);
                            assert_close(&got, &matmul(&l, &u, n, n, n), TOL, &format!("{what} {who}: P A Qᵀ = L U"));
                            assert!(!is_symmetric(&p, n) && !is_symmetric(&q, n), "{what}: symmetric permutations cannot tell the convention apart");
                        }
                    }
                }
            }

            #[test]
            fn qr() {
                for (m, n) in [(4, 4), (5, 3), (3, 5)] {
                    let k = m.min(n);
                    for (dims, layout) in configs() {
                        let a = input(m, n, &dims, layout, |i| matrix::<T>(m, n, i));
                        let (mut fq, mut fr) = (Vec::new(), Vec::new());
                        tlinalg::qr::qr(tlinalg::Op::Qr, a.view(), &mut fq, &mut fr, SEQ).unwrap();
                        let (mut bq, mut br) = (Vec::new(), Vec::new());
                        tlinalg_blas::qr::qr(tlinalg_blas::Op::Qr, a.view(), &mut bq, &mut br, &mut Ws).unwrap();
                        let what = format!("qr {m}x{n} {dims:?} {layout:?}");
                        for index in 0..a.count() {
                            let want = widen(&a.item(index));
                            let mut gauged = Vec::new();
                            for (q, r, who) in [
                                (widen(item(&fq, m * k, index)), widen(item(&fr, k * n, index)), "faer"),
                                (widen(item(&bq, m * k, index)), widen(item(&br, k * n, index)), "lapack"),
                            ] {
                                assert_close(&matmul(&q, &r, m, k, n), &want, TOL, &format!("{what} {who}: Q R"));
                                // Bring R's diagonal to the positive reals; the result is unique.
                                let d = diag_phases(&r, k);
                                let dconj: Vec<Complex64> = d.iter().map(|z| z.conj()).collect();
                                gauged.push((scale_cols(&q, &d, m, k), scale_rows(&r, &dconj, k, n)));
                            }
                            assert_close(&gauged[0].0, &gauged[1].0, TOL, &format!("{what} item {index}: gauged Q"));
                            assert_close(&gauged[0].1, &gauged[1].1, TOL, &format!("{what} item {index}: gauged R"));
                        }
                    }
                }
            }

            #[test]
            fn rank_revealing_qr() {
                for (m, n) in [(4, 4), (5, 3), (3, 5)] {
                    let k = m.min(n);
                    for (dims, layout) in configs() {
                        let a = input(m, n, &dims, layout, |i| matrix::<T>(m, n, i));
                        let (mut fq, mut fr, mut fperm) = (Vec::new(), Vec::new(), Vec::new());
                        tlinalg::qr::rank_revealing_qr(tlinalg::Op::RankRevealingQr, a.view(), &mut fq, &mut fr, &mut fperm, SEQ)
                            .unwrap();
                        let (mut bq, mut br, mut bperm) = (Vec::new(), Vec::new(), Vec::new());
                        tlinalg_blas::qr::rank_revealing_qr(
                            tlinalg_blas::Op::RankRevealingQr,
                            a.view(),
                            tlinalg_blas::qr::RankRevealingQrOutputs { q: &mut bq, r: &mut br, permutation: &mut bperm },
                            &mut Ws,
                        )
                        .unwrap();
                        let what = format!("rrqr {m}x{n} {dims:?} {layout:?}");
                        assert_eq!(fperm, bperm, "{what}: column permutation");
                        for index in 0..a.count() {
                            let want = widen(&a.item(index));
                            let mut diag = Vec::new();
                            for (q, r, perm, who) in [
                                (widen(item(&fq, m * k, index)), widen(item(&fr, k * n, index)), item(&fperm, n, index), "faer"),
                                (widen(item(&bq, m * k, index)), widen(item(&br, k * n, index)), item(&bperm, n, index), "lapack"),
                            ] {
                                let ap = gather_cols(&want, m, perm);
                                assert_close(&matmul(&q, &r, m, k, n), &ap, TOL, &format!("{what} {who}: A P = Q R"));
                                let d: Vec<Complex64> = (0..k).map(|j| Complex64::new(r[j + j * k].norm(), 0.0)).collect();
                                for pair in d.windows(2) {
                                    assert!(pair[0].re + TOL >= pair[1].re, "{what} {who}: |R_jj| not non-increasing");
                                }
                                diag.push(d);
                            }
                            assert_close(&diag[0], &diag[1], TOL, &format!("{what} item {index}: |diag R|"));
                        }
                    }
                }
            }

            #[test]
            fn rank_revealing_qr_all_zero_items() {
                // Items 1 and 3 of five are all zero; both providers skip the factorization for them
                // and agree exactly: identity Q columns, zero R, identity permutation.
                for (m, n) in [(4, 4), (5, 3), (3, 5)] {
                    let k = m.min(n);
                    let a = input(m, n, &[5], Layout::Gapped, |i| {
                        if i == 1 || i == 3 {
                            vec![T::default(); m * n]
                        } else {
                            matrix::<T>(m, n, i)
                        }
                    });
                    let (mut fq, mut fr, mut fperm) = (Vec::new(), Vec::new(), Vec::new());
                    tlinalg::qr::rank_revealing_qr(tlinalg::Op::RankRevealingQr, a.view(), &mut fq, &mut fr, &mut fperm, SEQ)
                        .unwrap();
                    let (mut bq, mut br, mut bperm) = (Vec::new(), Vec::new(), Vec::new());
                    tlinalg_blas::qr::rank_revealing_qr(
                        tlinalg_blas::Op::RankRevealingQr,
                        a.view(),
                        tlinalg_blas::qr::RankRevealingQrOutputs { q: &mut bq, r: &mut br, permutation: &mut bperm },
                        &mut Ws,
                    )
                    .unwrap();
                    let what = format!("rrqr all-zero {m}x{n}");
                    assert_eq!(fperm, bperm, "{what}: column permutation");
                    for index in [1, 3] {
                        assert_eq!(item(&fq, m * k, index), item(&bq, m * k, index), "{what}: Q");
                        assert_eq!(item(&fr, k * n, index), item(&br, k * n, index), "{what}: R");
                        let identity: Vec<i64> = (0..n as i64).collect();
                        assert_eq!(item(&fperm, n, index), &identity[..], "{what}: permutation");
                        let q = widen(item(&fq, m * k, index));
                        for col in 0..k {
                            for row in 0..m {
                                let want = if row == col { 1.0 } else { 0.0 };
                                assert_eq!(q[row + col * m], Complex64::new(want, 0.0), "{what}: Q entry");
                            }
                        }
                        assert!(widen(item(&fr, k * n, index)).iter().all(|z| *z == Complex64::new(0.0, 0.0)), "{what}: R");
                    }
                }
            }

            #[test]
            fn eigh() {
                let n = 5;
                for (dims, layout) in configs() {
                    let a = input(n, n, &dims, layout, |i| hermitian_seeded::<T>(n, i + 2));
                    let (mut fw, mut fv) = (Vec::new(), Vec::new());
                    tlinalg::eigh::eigh(tlinalg::Op::Eigh, a.view(), &mut fw, &mut fv, SEQ).unwrap();
                    let (mut bw, mut bv) = (Vec::new(), Vec::new());
                    tlinalg_blas::eigh::eigh(tlinalg_blas::Op::Eigh, a.view(), &mut bw, Some(&mut bv), &mut Ws)
                        .unwrap();
                    let what = format!("eigh {dims:?} {layout:?}");
                    assert_close(&widen(&fw), &widen(&bw), TOL, &format!("{what}: values"));
                    for index in 0..a.count() {
                        let want = widen(&a.item(index));
                        let w = widen(item(&bw, n, index));
                        let mut vectors = Vec::new();
                        for (v, who) in [(widen(item(&fv, n * n, index)), "faer"), (widen(item(&bv, n * n, index)), "lapack")] {
                            let av = matmul(&want, &v, n, n, n);
                            assert_close(&av, &scale_cols(&v, &w, n, n), TOL, &format!("{what} {who}: A V = V W"));
                            vectors.push(v);
                        }
                        // Distinct eigenvalues: the eigenvectors agree up to a unit phase per column.
                        let overlap = matmul(&adjoint(&vectors[0], n, n), &vectors[1], n, n, n);
                        for j in 0..n {
                            assert!((overlap[j + j * n].norm() - 1.0).abs() <= TOL * 10.0, "{what}: eigenvector {j} overlap {}", overlap[j + j * n]);
                        }
                    }
                    let mut fw = Vec::new();
                    tlinalg::eigh::eigh_values(tlinalg::Op::EighValues, a.view(), &mut fw, SEQ).unwrap();
                    let mut bw = Vec::new();
                    tlinalg_blas::eigh::eigh(tlinalg_blas::Op::EighValues, a.view(), &mut bw, None, &mut Ws).unwrap();
                    assert_close(&widen(&fw), &widen(&bw), TOL, &format!("eigvalsh {dims:?} {layout:?}"));
                }
            }

            #[test]
            fn eig() {
                let n = 4;
                for (dims, layout) in configs() {
                    let a = input(n, n, &dims, layout, |i| matrix::<T>(n, n, i));
                    let (mut fw, mut fv) = (Vec::new(), Vec::new());
                    tlinalg::eig::eig(tlinalg::Op::Eig, a.view(), &mut fw, &mut fv, SEQ).unwrap();
                    let (mut bw, mut bv) = (Vec::new(), Vec::new());
                    tlinalg_blas::eig::eig(tlinalg_blas::Op::Eig, a.view(), &mut bw, Some(&mut bv), &mut Ws).unwrap();
                    let (mut fwo, mut bwo) = (Vec::new(), Vec::new());
                    tlinalg::eig::eig_values(tlinalg::Op::EigValues, a.view(), &mut fwo, SEQ).unwrap();
                    tlinalg_blas::eig::eig(tlinalg_blas::Op::EigValues, a.view(), &mut bwo, None, &mut Ws).unwrap();
                    let what = format!("eig {dims:?} {layout:?}");
                    for index in 0..a.count() {
                        let want = widen(&a.item(index));
                        let reference = sorted(&widen(item(&bw, n, index)));
                        for (values, who) in [
                            (widen(item(&fw, n, index)), "faer eig"),
                            (widen(item(&fwo, n, index)), "faer eig_values"),
                            (widen(item(&bwo, n, index)), "lapack values-only"),
                        ] {
                            assert_close(&sorted(&values), &reference, TOL, &format!("{what} item {index}: {who} values"));
                        }
                        for (w, v, who) in [
                            (widen(item(&fw, n, index)), widen(item(&fv, n * n, index)), "faer"),
                            (widen(item(&bw, n, index)), widen(item(&bv, n * n, index)), "lapack"),
                        ] {
                            let av = matmul(&want, &v, n, n, n);
                            assert_close(&av, &scale_cols(&v, &w, n, n), TOL * 10.0, &format!("{what} {who}: A V = V W"));
                        }
                    }
                }
            }

            #[test]
            fn packed_lu() {
                let (n, nrhs) = (4, 2);
                for count in [1usize, 3, 6] {
                    let a: Vec<T> = (0..count).flat_map(|i| pivoting::<T>(n, n, i)).collect();
                    let b: Vec<T> = (0..count).flat_map(|i| matrix::<T>(n, nrhs, i + 5)).collect();
                    let (mut flu, mut fpiv, mut fpar) = (a.clone(), vec![0i32; n * count], vec![T::default(); count]);
                    tlinalg::packed_lu::factor(tlinalg::Op::LuFactor, n, n, &mut flu, &mut fpiv, &mut fpar, SEQ).unwrap();
                    let (mut blu, mut bpiv, mut bpar) = (a.clone(), vec![0i32; n * count], vec![T::default(); count]);
                    tlinalg_blas::lu::lu_factor(tlinalg_blas::Op::LuFactor, n, n, &mut blu, &mut bpiv, &mut bpar).unwrap();
                    let what = format!("packed lu batch {count}");
                    assert_close(&widen(&fpar), &widen(&bpar), TOL, &format!("{what}: parity"));
                    // Same LAPACK `ipiv` convention and, with well-separated pivots, the same choices.
                    assert_eq!(fpiv, bpiv, "{what}: pivots");
                    assert!(fpiv.iter().enumerate().any(|(i, &p)| p as usize != i % n + 1), "{what}: pivoting is trivial");
                    assert_close(&widen(&flu), &widen(&blu), TOL, &format!("{what}: packed factors"));
                    let lu_dims = [n, n, count];
                    let lu_strides = [1, n as isize, (n * n) as isize];
                    let piv_dims = [n, count];
                    let piv_strides = [1, n as isize];
                    for (transpose_a, conjugate_a) in [(false, false), (true, false), (true, true)] {
                        let mut results = Vec::new();
                        // Each provider solves with its own factors and with the other's: the packed
                        // LAPACK format is the same contract on both sides.
                        for (lu, piv, who) in [(&flu, &fpiv, "faer factors"), (&blu, &bpiv, "lapack factors")] {
                            let mut fx = b.clone();
                            tlinalg::packed_lu::solve_prepared(
                                tlinalg::Op::LuSolvePrepared,
                                strided_view::RawStridedRef::new(lu, &lu_dims, &lu_strides, 0).unwrap(),
                                strided_view::RawStridedRef::new(piv, &piv_dims, &piv_strides, 0).unwrap(),
                                nrhs,
                                &mut fx,
                                transpose_a,
                                conjugate_a,
                                SEQ,
                            )
                            .unwrap();
                            let mut bx = b.clone();
                            tlinalg_blas::lu::lu_solve_prepared(tlinalg_blas::Op::LuSolvePrepared, n, nrhs, lu, piv, &mut bx, transpose_a, conjugate_a)
                                .unwrap();
                            results.push((widen(&fx), format!("faer solve, {who}")));
                            results.push((widen(&bx), format!("lapack solve, {who}")));
                        }
                        for (x, who) in &results[1..] {
                            assert_close(x, &results[0].0, TOL, &format!("{what} trans={transpose_a} conj={conjugate_a}: {who}"));
                        }
                        for index in 0..count {
                            let a_i = widen(item(&a, n * n, index));
                            let op_a = match (transpose_a, conjugate_a) {
                                (false, _) => a_i,
                                (true, false) => transpose(&a_i, n, n),
                                (true, true) => adjoint(&a_i, n, n),
                            };
                            let got = matmul(&op_a, item(&results[0].0, n * nrhs, index), n, n, nrhs);
                            assert_close(&got, &widen(item(&b, n * nrhs, index)), TOL, &format!("{what}: residual"));
                        }
                    }
                    let (mut flu2, mut fpiv2, mut fx) = (a.clone(), vec![0i32; n * count], b.clone());
                    tlinalg::packed_lu::factor_solve(tlinalg::Op::LuFactorSolve, n, nrhs, &mut flu2, &mut fpiv2, &mut fx, SEQ).unwrap();
                    let (mut blu2, mut bpiv2, mut bx) = (a.clone(), vec![0i32; n * count], b.clone());
                    tlinalg_blas::lu::lu_factor_solve(tlinalg_blas::Op::LuFactorSolve, n, nrhs, &mut blu2, &mut bpiv2, &mut bx).unwrap();
                    assert_close(&widen(&fx), &widen(&bx), TOL, &format!("{what}: fused factor+solve"));
                }
            }

            #[test]
            fn householder() {
                for (rows, cols) in [(5, 3), (4, 4)] {
                    let k = rows.min(cols);
                    let count = 3;
                    let a: Vec<T> = (0..count).flat_map(|i| matrix::<T>(rows, cols, i)).collect();
                    let mut fdata = a.clone();
                    let mut fcoeff = Vec::new();
                    tlinalg::householder::compact_factor(tlinalg::Op::HouseholderQr, rows, cols, count, &mut fdata, &mut fcoeff, SEQ)
                        .unwrap();
                    let mut bdata = a.clone();
                    let mut btau = Vec::new();
                    tlinalg_blas::householder::factor(tlinalg_blas::Op::HouseholderFactor, rows, cols, &mut bdata, &mut btau, &mut Ws)
                        .unwrap();
                    let what = format!("householder {rows}x{cols}");
                    // Q applied to the identity's leading columns, per provider, reconstructs A from
                    // that provider's own R.
                    let eye: Vec<T> = (0..count)
                        .flat_map(|_| {
                            let id = identity(rows);
                            id[..rows * k].iter().map(|&z| <T as TestScalar>::from_c64(z)).collect::<Vec<T>>()
                        })
                        .collect();
                    let mut fq = eye.clone();
                    tlinalg::householder::apply_reflectors(
                        tlinalg::Op::HouseholderQrQColumns,
                        tlinalg::householder::ReflectorShape { rows, a_cols: cols, cols: k, k },
                        count,
                        &fdata,
                        &fcoeff,
                        &mut fq,
                        false,
                        SEQ,
                    )
                    .unwrap();
                    let mut bq = eye.clone();
                    tlinalg_blas::householder::apply_reflectors(tlinalg_blas::Op::HouseholderApply, rows, cols, k, k, false, &bdata, &btau, &mut bq, &mut Ws)
                        .unwrap();
                    for index in 0..count {
                        let want = widen(item(&a, rows * cols, index));
                        let mut gauged = Vec::new();
                        for (data, q, who) in [(&fdata, &fq, "faer"), (&bdata, &bq, "lapack")] {
                            let packed = widen(item(data, rows * cols, index));
                            let mut r = vec![Complex64::new(0.0, 0.0); k * cols];
                            for col in 0..cols {
                                for row in 0..k.min(col + 1) {
                                    r[row + col * k] = packed[row + col * rows];
                                }
                            }
                            let q = widen(item(q, rows * k, index));
                            assert_close(&matmul(&q, &r, rows, k, cols), &want, TOL, &format!("{what} {who} item {index}: Q R"));
                            let d = diag_phases(&r, k);
                            let dconj: Vec<Complex64> = d.iter().map(|z| z.conj()).collect();
                            gauged.push(scale_rows(&r, &dconj, k, cols));
                        }
                        assert_close(&gauged[0], &gauged[1], TOL, &format!("{what} item {index}: gauged R"));
                    }
                }
            }
        }
    };
}

parity_suite!(f32_parity, f32);
parity_suite!(f64_parity, f64);
parity_suite!(c32_parity, tlinalg_testkit::Complex32);
parity_suite!(c64_parity, tlinalg_testkit::Complex64);
