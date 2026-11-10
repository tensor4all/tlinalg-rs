//! Behavioural tests for the ported partial-pivot LU, the LU solve, and the full-pivot LU family.

mod common;

use common::single::{full_piv_lu, full_piv_lu_solve, lu, solve, FullPivLuFactors, LuFactors};
use common::*;
use num_complex::Complex64;
use strided_view::{RawStridedMut, RawStridedRef};
use tcontract_exec::Exec;
use tlinalg::FaerScalar;
use tlinalg::{Error, Op};

/// Determinant sign of a permutation matrix, by counting inversions of its row map.
fn permutation_sign(p: &[Complex64], n: usize) -> f64 {
    let map: Vec<usize> = (0..n)
        .map(|row| (0..n).find(|&col| p[row + col * n].re == 1.0).unwrap())
        .collect();
    let mut inversions = 0;
    for i in 0..n {
        for j in i + 1..n {
            if map[i] > map[j] {
                inversions += 1;
            }
        }
    }
    if inversions % 2 == 0 {
        1.0
    } else {
        -1.0
    }
}

fn lu_reconstructs<T: TestScalar + FaerScalar>() {
    for (m, n) in [(4, 4), (5, 3), (3, 5), (1, 1)] {
        let a = pivoting::<T>(m, n, 11);
        let (storage, strides, offset) = padded(&a, m, n);
        let dims = [m, n];
        let (mut p, mut l, mut u) = (Vec::new(), Vec::new(), Vec::new());
        let parity = lu(
            Op::Lu,
            m,
            n,
            RawStridedRef::new(&storage, &dims, &strides, offset).unwrap(),
            LuFactors {
                p: &mut p,
                l: &mut l,
                u: &mut u,
            },
            &mut Exec::serial(),
        )
        .unwrap();
        let k = m.min(n);
        assert_eq!((p.len(), l.len(), u.len()), (m * m, m * k, k * n));
        let (p, l, u) = (widen(&p), widen(&l), widen(&u));
        for i in 0..k {
            assert_eq!(l[i + i * m], Complex64::new(1.0, 0.0), "unit diagonal");
        }
        // `P A = L U`.
        let rebuilt = matmul(&p, &widen(&a), m, m, n);
        assert_close(
            &rebuilt,
            &matmul(&l, &u, m, k, n),
            T::TOL,
            &format!("P A = L U {m}x{n}"),
        );
        assert_eq!(parity.to_c64().re, permutation_sign(&p, m), "parity");
    }
}
for_each_scalar!(lu_reconstruction, lu_reconstructs);

#[test]
fn lu_of_a_singular_matrix_is_not_an_error() {
    let a = [1.0_f64, 2.0, 2.0, 4.0];
    let (mut p, mut l, mut u) = (Vec::new(), Vec::new(), Vec::new());
    lu(
        Op::Lu,
        2,
        2,
        RawStridedRef::new(&a, &[2, 2], &[1, 2], 0).unwrap(),
        LuFactors {
            p: &mut p,
            l: &mut l,
            u: &mut u,
        },
        &mut Exec::serial(),
    )
    .unwrap();
    assert_eq!(u[3], 0.0);
}

fn solve_residual<T: TestScalar + FaerScalar>() {
    let n = 4;
    let nrhs = 2;
    let a = pivoting::<T>(n, n, 13);
    let b = matrix::<T>(n, nrhs, 17);
    for transpose_a in [false, true] {
        // Strided destination: leading dimension n + 2, offset 3.
        let (mut storage, strides, offset) = padded(&vec![T::default(); n * nrhs], n, nrhs);
        let dims = [n, nrhs];
        solve(
            Op::Solve,
            n,
            nrhs,
            RawStridedRef::new(&a, &[n, n], &[1, n as isize], 0).unwrap(),
            Some(RawStridedRef::new(&b, &dims, &[1, n as isize], 0).unwrap()),
            RawStridedMut::new(&mut storage, &dims, &strides, offset).unwrap(),
            transpose_a,
            &mut Exec::serial(),
        )
        .unwrap();
        let lda = n + 2;
        let x: Vec<Complex64> = (0..n * nrhs)
            .map(|i| storage[3 + i % n + (i / n) * lda].to_c64())
            .collect();
        let op_a = if transpose_a {
            transpose(&widen(&a), n, n)
        } else {
            widen(&a)
        };
        assert_close(
            &matmul(&op_a, &x, n, n, nrhs),
            &widen(&b),
            T::TOL,
            &format!("solve transpose={transpose_a}"),
        );
        // The padding outside the destination is untouched.
        assert!(storage[0].to_c64().re > 1.0e29);
    }

    // `None`: the destination already holds the right-hand side.
    let mut x = b.clone();
    solve(
        Op::Solve,
        n,
        nrhs,
        RawStridedRef::new(&a, &[n, n], &[1, n as isize], 0).unwrap(),
        None,
        RawStridedMut::new(&mut x, &[n, nrhs], &[1, n as isize], 0).unwrap(),
        false,
        &mut Exec::serial(),
    )
    .unwrap();
    assert_close(
        &matmul(&widen(&a), &widen(&x), n, n, nrhs),
        &widen(&b),
        T::TOL,
        "solve in place",
    );
}
for_each_scalar!(solve_residuals, solve_residual);

fn solve_singular_leaves_destination<T: TestScalar + FaerScalar>() {
    let a = narrow::<T>(&[
        Complex64::new(1.0, 0.0),
        Complex64::new(2.0, 0.0),
        Complex64::new(2.0, 0.0),
        Complex64::new(4.0, 0.0),
    ]);
    let b = matrix::<T>(2, 1, 1);
    let sentinel = T::from_c64(Complex64::new(7.0, 0.0));
    let mut out = vec![sentinel; 2];
    let err = solve(
        Op::Solve,
        2,
        1,
        RawStridedRef::new(&a, &[2, 2], &[1, 2], 0).unwrap(),
        Some(RawStridedRef::new(&b, &[2, 1], &[1, 2], 0).unwrap()),
        RawStridedMut::new(&mut out, &[2, 1], &[1, 2], 0).unwrap(),
        false,
        &mut Exec::serial(),
    )
    .unwrap_err();
    assert_eq!(err, Error::Singular { op: Op::Solve });
    assert_eq!(out, vec![sentinel; 2], "destination untouched on failure");
}
for_each_scalar!(solve_singular, solve_singular_leaves_destination);

fn full_piv_reconstructs<T: TestScalar + FaerScalar>() {
    for (m, n) in [(4, 4), (5, 3), (3, 5), (1, 1)] {
        let a = pivoting::<T>(m, n, 19);
        let (storage, strides, offset) = padded(&a, m, n);
        let dims = [m, n];
        let (mut p, mut l, mut u, mut q) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        let parity = full_piv_lu(
            Op::FullPivLu,
            m,
            RawStridedRef::new(&storage, &dims, &strides, offset).unwrap(),
            FullPivLuFactors {
                p: &mut p,
                l: &mut l,
                u: &mut u,
                q: &mut q,
            },
            &mut Exec::serial(),
        )
        .unwrap();
        let k = m.min(n);
        assert_eq!(
            (p.len(), l.len(), u.len(), q.len()),
            (m * m, m * k, k * n, n * n)
        );
        let (p, l, u, q) = (widen(&p), widen(&l), widen(&u), widen(&q));
        for i in 0..k {
            assert_eq!(l[i + i * m], Complex64::new(1.0, 0.0), "unit diagonal");
        }
        for col in 0..k {
            for row in 0..m {
                if row < col {
                    assert_eq!(l[row + col * m], Complex64::new(0.0, 0.0), "L upper");
                }
            }
        }
        for col in 0..n {
            for row in 0..k {
                if row > col {
                    assert_eq!(u[row + col * k], Complex64::new(0.0, 0.0), "U lower");
                }
            }
        }
        // `P A Qᵀ = L U`.
        let rebuilt = matmul(
            &matmul(&p, &widen(&a), m, m, n),
            &transpose(&q, n, n),
            m,
            n,
            n,
        );
        assert_close(&rebuilt, &matmul(&l, &u, m, k, n), T::TOL, "P A Qᵀ = L U");
        let sign = permutation_sign(&p, m) * permutation_sign(&q, n);
        assert_eq!(parity.to_c64().re, sign, "parity");
    }
}
for_each_scalar!(full_piv_reconstruction, full_piv_reconstructs);

fn full_piv_reveals_rectangular_rank<T: TestScalar + FaerScalar>() {
    // Rank one `3 x 2`: the second column is twice the first. The factorization is not truncated,
    // so it runs its `k = 2` steps and the second pivot is numerically zero; a singular input is
    // not an error here.
    let a = narrow::<T>(&[
        Complex64::new(1.0, 0.0),
        Complex64::new(2.0, 0.0),
        Complex64::new(4.0, 0.0),
        Complex64::new(2.0, 0.0),
        Complex64::new(4.0, 0.0),
        Complex64::new(8.0, 0.0),
    ]);
    let (m, n) = (3, 2);
    let k = m.min(n);
    let (mut p, mut l, mut u, mut q) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    full_piv_lu(
        Op::FullPivLu,
        m,
        RawStridedRef::new(&a, &[m, n], &[1, m as isize], 0).unwrap(),
        FullPivLuFactors {
            p: &mut p,
            l: &mut l,
            u: &mut u,
            q: &mut q,
        },
        &mut Exec::serial(),
    )
    .unwrap();
    assert_eq!(
        (p.len(), l.len(), u.len(), q.len()),
        (m * m, m * k, k * n, n * n)
    );
    let u = widen(&u);
    let diagonal: Vec<f64> = (0..k).map(|i| u[i + i * k].norm()).collect();
    let largest = diagonal.iter().copied().fold(0.0_f64, f64::max);
    let rank = diagonal.iter().filter(|&&d| d > 1e-10 * largest).count();
    assert_eq!(rank, 1, "diagonal {diagonal:?}");
    // The reconstruction still holds, with the zero pivot on `U`'s diagonal.
    let (p, l, q) = (widen(&p), widen(&l), widen(&q));
    let rebuilt = matmul(
        &matmul(&p, &widen(&a), m, m, n),
        &transpose(&q, n, n),
        m,
        n,
        n,
    );
    assert_close(&rebuilt, &matmul(&l, &u, m, k, n), T::TOL, "P A Qᵀ = L U");
}
for_each_scalar!(full_piv_rectangular_rank, full_piv_reveals_rectangular_rank);

fn full_piv_accepts_empty_core_dimensions<T: TestScalar + FaerScalar>() {
    // A zero core extent keeps the permutation factors that still have a side: `P` is `m x m`,
    // `Q` is `n x n`, and the factors with a zero side are empty.
    for (m, n) in [(0, 3), (3, 0)] {
        let k = m.min(n);
        let (mut p, mut l, mut u, mut q, mut parity) =
            (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new());
        tlinalg::full_piv_lu::full_piv_lu(
            Op::FullPivLu,
            RawStridedRef::new(&[] as &[T], &[m, n], &[1, m as isize], 0).unwrap(),
            tlinalg::full_piv_lu::FullPivLuFactors {
                p: &mut p,
                l: &mut l,
                u: &mut u,
                q: &mut q,
                parity: &mut parity,
            },
            &mut Exec::serial(),
        )
        .unwrap();
        assert_eq!(
            (p.len(), l.len(), u.len(), q.len()),
            (m * m, m * k, k * n, n * n)
        );
        let (p, q, parity) = (widen(&p), widen(&q), widen(&parity));
        assert_eq!(parity.len(), 1);
        assert_eq!(parity[0], Complex64::new(1.0, 0.0));
        for i in 0..m {
            assert_eq!(p[i + i * m], Complex64::new(1.0, 0.0), "identity P");
        }
        for j in 0..n {
            assert_eq!(q[j + j * n], Complex64::new(1.0, 0.0), "identity Q");
        }
    }
}
for_each_scalar!(
    full_piv_empty_dimensions,
    full_piv_accepts_empty_core_dimensions
);

fn full_piv_solves<T: TestScalar + FaerScalar>() {
    let n = 4;
    let nrhs = 3;
    let a = pivoting::<T>(n, n, 23);
    let b = matrix::<T>(n, nrhs, 29);
    for transpose_a in [false, true] {
        let mut x = b.clone();
        full_piv_lu_solve(
            Op::FullPivLuSolve,
            n,
            nrhs,
            RawStridedRef::new(&a, &[n, n], &[1, n as isize], 0).unwrap(),
            &mut x,
            transpose_a,
            &mut Exec::serial(),
        )
        .unwrap();
        let op_a = if transpose_a {
            transpose(&widen(&a), n, n)
        } else {
            widen(&a)
        };
        assert_close(
            &matmul(&op_a, &widen(&x), n, n, nrhs),
            &widen(&b),
            T::TOL,
            "full-pivot solve",
        );
    }

    // Rank one: the second pivot is effectively zero.
    let singular = narrow::<T>(&[
        Complex64::new(1.0, 0.0),
        Complex64::new(2.0, 0.0),
        Complex64::new(2.0, 0.0),
        Complex64::new(4.0, 0.0),
    ]);
    let mut x = matrix::<T>(2, 1, 0);
    let before = x.clone();
    let err = full_piv_lu_solve(
        Op::FullPivLuSolve,
        2,
        1,
        RawStridedRef::new(&singular, &[2, 2], &[1, 2], 0).unwrap(),
        &mut x,
        false,
        &mut Exec::serial(),
    )
    .unwrap_err();
    assert_eq!(
        err,
        Error::Singular {
            op: Op::FullPivLuSolve
        }
    );
    assert_eq!(x, before, "right-hand side untouched on failure");

    let err = full_piv_lu_solve(
        Op::FullPivLuSolve,
        2,
        2,
        RawStridedRef::new(&singular, &[2, 2], &[1, 2], 0).unwrap(),
        &mut x,
        false,
        &mut Exec::serial(),
    )
    .unwrap_err();
    assert!(matches!(err, Error::InvalidArgument { .. }));
}
for_each_scalar!(full_piv_solve, full_piv_solves);

#[test]
fn full_piv_lu_refuses_an_overflowing_output_total_without_panicking() {
    // A broadcast batch axis: two stored elements, `2^62` items. `P`'s item is `1 x 1` so its
    // total fits while `Q`'s (`2 x 2` per item) does not; the refusal has to come from the checked
    // total, not from a reservation that panics first.
    let a = [1.0_f64, 2.0];
    let (mut p, mut l, mut u, mut q, mut parity) =
        (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new());
    let err = tlinalg::full_piv_lu::full_piv_lu(
        Op::FullPivLu,
        RawStridedRef::new(&a, &[1, 2, 1usize << 62], &[1, 1, 0], 0).unwrap(),
        tlinalg::full_piv_lu::FullPivLuFactors {
            p: &mut p,
            l: &mut l,
            u: &mut u,
            q: &mut q,
            parity: &mut parity,
        },
        &mut Exec::serial(),
    )
    .unwrap_err();
    assert!(matches!(err, Error::InvalidArgument { .. }), "{err:?}");
    assert!(p.is_empty() && l.is_empty() && u.is_empty() && q.is_empty());
}

#[test]
fn full_piv_lu_solve_rejects_a_rectangular_coefficient() {
    // The rectangular widening covers the factorization only; the solve stays square-only and
    // says so instead of factoring a non-square `A`.
    let mut x = vec![0.0_f64; 6];
    let err = full_piv_lu_solve(
        Op::FullPivLuSolve,
        3,
        2,
        RawStridedRef::new(&[1.0_f64, 0.0, 0.0, 2.0, 0.0, 0.0], &[3, 2], &[1, 3], 0).unwrap(),
        &mut x,
        false,
        &mut Exec::serial(),
    )
    .unwrap_err();
    match err {
        Error::InvalidArgument { op, detail, .. } => {
            assert_eq!(op, Op::FullPivLuSolve);
            assert!(detail.contains("square"), "{detail}");
        }
        other => panic!("expected InvalidArgument, got {other:?}"),
    }
}

#[test]
fn solve_on_a_supplied_pool_matches_sequential() {
    let pool = tcontract_exec::Pool::owned(threadteam::Pool::builder().threads(2).build().unwrap());
    let n = 40;
    let a = matrix::<f64>(n, n, 3);
    let b = matrix::<f64>(n, 2, 4);
    let run = |par: &mut Exec<'_>| {
        let mut x = b.clone();
        solve(
            Op::Solve,
            n,
            2,
            RawStridedRef::new(&a, &[n, n], &[1, n as isize], 0).unwrap(),
            None,
            RawStridedMut::new(&mut x, &[n, 2], &[1, n as isize], 0).unwrap(),
            false,
            par,
        )
        .unwrap();
        x
    };
    let sequential = run(&mut Exec::serial());
    let pooled = run(&mut tcontract_exec::Exec::new(&pool, 2).unwrap());
    assert_close(
        &widen(&pooled),
        &widen(&sequential),
        1e-12,
        "pool vs sequential",
    );
}
