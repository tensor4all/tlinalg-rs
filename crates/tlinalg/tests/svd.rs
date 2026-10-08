//! Behavioural tests for the ported SVD kernel.
//!
//! The check is reconstruction: `U diag(S) Vᴴ ≈ A` for thin and full, square, tall, wide and
//! complex, plus that values-only agrees with the full decomposition's values.

use num_complex::{Complex32, Complex64};
use strided_view::RawStridedRef;
use tlinalg::svd::{svd, svd_values};
use tlinalg::{Op, Parallel};

/// Column-major `m x n` matrix with well-separated singular values.
fn matrix(m: usize, n: usize) -> Vec<f64> {
    (0..m * n)
        .map(|index| {
            let row = index % m;
            let col = index / m;
            if row == col {
                3.0 + (row as f64)
            } else {
                0.5 + ((row * 3 + col * 7) % 5) as f64 * 0.25
            }
        })
        .collect()
}

/// `U diag(S) Vᴴ` in column-major, with `u` `m x u_cols`, `s` `k`, `vt` `v_cols x n`.
fn reconstruct(u: &[f64], s: &[f64], vt: &[f64], m: usize, n: usize, v_cols: usize) -> Vec<f64> {
    let k = s.len();
    let mut out = vec![0.0; m * n];
    for row in 0..m {
        for col in 0..n {
            let mut acc = 0.0;
            for j in 0..k {
                acc += u[row + j * m] * s[j] * vt[j + col * v_cols];
            }
            out[row + col * m] = acc;
        }
    }
    out
}

fn check_svd(m: usize, n: usize, full: bool) {
    let a = matrix(m, n);
    let k = m.min(n);
    let (u_cols, v_cols) = if full { (m, n) } else { (k, k) };
    // Empty vectors with the right capacity: the kernel pushes the factors.
    let mut u = Vec::with_capacity(m * u_cols);
    let mut s = Vec::with_capacity(k);
    let mut vt = Vec::with_capacity(v_cols * n);
    svd(
        Op::Svd,
        RawStridedRef::new(&a, &[m, n], &[1, m as isize], 0).unwrap(),
        full,
        &mut u,
        &mut s,
        &mut vt,
        Parallel::Sequential,
    )
    .unwrap();

    assert_eq!(u.len(), m * u_cols);
    assert_eq!(s.len(), k);
    assert_eq!(vt.len(), v_cols * n);

    // Values are non-increasing and positive.
    for pair in s.windows(2) {
        assert!(
            pair[0] >= pair[1],
            "{m}x{n} full={full}: {:?} not sorted",
            s
        );
    }
    assert!(s[k - 1] > 1e-8, "{m}x{n}: unexpectedly singular");

    let rebuilt = reconstruct(&u, &s, &vt, m, n, v_cols);
    for (index, (got, want)) in rebuilt.iter().zip(a.iter()).enumerate() {
        assert!(
            (got - want).abs() < 1e-10,
            "{m}x{n} full={full} entry {index}: reconstructed {got} != {want}"
        );
    }
}

#[test]
fn svd_reconstructs_square_tall_and_wide_matrices() {
    for (m, n) in [(4usize, 4usize), (6, 3), (3, 6), (1, 5), (5, 1)] {
        check_svd(m, n, false);
        check_svd(m, n, true);
    }
}

#[test]
fn values_only_agrees_with_the_full_decomposition() {
    let (m, n) = (5usize, 4usize);
    let a = matrix(m, n);
    assert!(m.min(n) > 0);
    let mut full = Vec::new();
    svd(
        Op::Svd,
        RawStridedRef::new(&a, &[m, n], &[1, m as isize], 0).unwrap(),
        false,
        &mut Vec::new(),
        &mut full,
        &mut Vec::new(),
        Parallel::Sequential,
    )
    .unwrap();
    let mut only = Vec::new();
    svd_values(
        Op::SvdValues,
        RawStridedRef::new(&a, &[m, n], &[1, m as isize], 0).unwrap(),
        &mut only,
        Parallel::Sequential,
    )
    .unwrap();
    for (got, want) in only.iter().zip(full.iter()) {
        assert!((got - want).abs() < 1e-12, "{got} != {want}");
    }
}

#[test]
fn complex_and_f32_reconstruct() {
    let (m, n) = (4usize, 3usize);
    let k = m.min(n);

    // Complex64.
    let a: Vec<Complex64> = matrix(m, n)
        .into_iter()
        .enumerate()
        .map(|(index, real)| Complex64::new(real, if index % 3 == 0 { 0.5 } else { -0.25 }))
        .collect();
    let mut u = Vec::new();
    let mut s = Vec::new();
    let mut vt = Vec::new();
    svd(
        Op::Svd,
        RawStridedRef::new(&a, &[m, n], &[1, m as isize], 0).unwrap(),
        false,
        &mut u,
        &mut s,
        &mut vt,
        Parallel::Sequential,
    )
    .unwrap();
    // `s` is carried in the complex scalar type with a zero imaginary part.
    for value in &s {
        assert_eq!(value.im, 0.0);
        assert!(value.re > 0.0);
    }
    let mut rebuilt = vec![Complex64::new(0.0, 0.0); m * n];
    for row in 0..m {
        for col in 0..n {
            // `vt` already holds `Vᴴ`, so the product needs no further conjugation.
            let mut acc = Complex64::new(0.0, 0.0);
            for j in 0..k {
                acc += u[row + j * m] * s[j] * vt[j + col * k];
            }
            rebuilt[row + col * m] = acc;
        }
    }
    for (index, (got, want)) in rebuilt.iter().zip(a.iter()).enumerate() {
        assert!(
            (got - want).norm() < 1e-10,
            "complex entry {index}: {got} != {want}"
        );
    }

    // Complex32 and f32 compile and run through the same paths.
    let a32: Vec<Complex32> = a
        .iter()
        .map(|v| Complex32::new(v.re as f32, v.im as f32))
        .collect();
    let mut u32 = Vec::new();
    let mut s32 = Vec::new();
    let mut vt32 = Vec::new();
    svd(
        Op::Svd,
        RawStridedRef::new(&a32, &[m, n], &[1, m as isize], 0).unwrap(),
        false,
        &mut u32,
        &mut s32,
        &mut vt32,
        Parallel::Sequential,
    )
    .unwrap();

    let ar: Vec<f32> = matrix(m, n).into_iter().map(|v| v as f32).collect();
    let mut ur = Vec::new();
    let mut sr = Vec::new();
    let mut vtr = Vec::new();
    svd(
        Op::Svd,
        RawStridedRef::new(&ar, &[m, n], &[1, m as isize], 0).unwrap(),
        false,
        &mut ur,
        &mut sr,
        &mut vtr,
        Parallel::Sequential,
    )
    .unwrap();
    assert!(sr[0] > 0.0);
}

#[test]
fn outputs_are_cleared_before_being_filled() {
    // The caller may reuse a pooled vector for the next call, so a stale length must not survive.
    let (m, n) = (3usize, 2usize);
    let a = matrix(m, n);
    let k = m.min(n);
    let mut u = vec![7.0f64; 99];
    let mut s = vec![7.0f64; 99];
    let mut vt = vec![7.0f64; 99];
    svd(
        Op::Svd,
        RawStridedRef::new(&a, &[m, n], &[1, m as isize], 0).unwrap(),
        false,
        &mut u,
        &mut s,
        &mut vt,
        Parallel::Sequential,
    )
    .unwrap();
    assert_eq!(u.len(), m * k);
    assert_eq!(s.len(), k);
    assert_eq!(vt.len(), k * n);
}

/// A rank-deficient matrix with clustered singular values, for which faer's divide-and-conquer
/// SVD is inaccurate (<https://github.com/tensor4all/tlinalg-rs/issues/13>). Both entry points
/// have to return its singular values, and the factors have to reproduce it.
#[test]
fn clustered_singular_values_of_a_rank_deficient_matrix() {
    let n = 160usize;
    let spectrum = tlinalg_testkit::clustered_spectrum(n);
    for full in [false, true] {
        let a: Vec<Complex64> = tlinalg_testkit::with_singular_values(&spectrum, 1);
        let (mut u, mut s, mut vt) = (Vec::new(), Vec::new(), Vec::new());
        svd(
            Op::Svd,
            RawStridedRef::new(&a, &[n, n], &[1, n as isize], 0).unwrap(),
            full,
            &mut u,
            &mut s,
            &mut vt,
            Parallel::Sequential,
        )
        .unwrap();
        let mut values = Vec::new();
        svd_values(
            Op::SvdValues,
            RawStridedRef::new(&a, &[n, n], &[1, n as isize], 0).unwrap(),
            &mut values,
            Parallel::Sequential,
        )
        .unwrap();

        for (index, want) in spectrum.iter().enumerate() {
            assert!(
                (s[index].re - want).abs() < 1e-12,
                "full={full}: singular value {index}: {} != {want}",
                s[index].re
            );
            assert!(
                (values[index] - want).abs() < 1e-12,
                "singular value {index} of the values-only path: {} != {want}",
                values[index]
            );
        }
        let mut error = 0.0f64;
        for col in 0..n {
            for row in 0..n {
                let mut rebuilt = Complex64::new(0.0, 0.0);
                for j in 0..n {
                    rebuilt += u[row + j * n] * s[j] * vt[j + col * n];
                }
                error += (rebuilt - a[row + col * n]).norm_sqr();
            }
        }
        // The Frobenius norm of the matrix is about sqrt(n / 2).
        let relative = (error / (n / 2) as f64).sqrt();
        assert!(
            relative < 1e-12,
            "full={full}: relative reconstruction error {relative:e}"
        );
    }
}

/// The full factor of a tall SVD is unitary over its whole square, not just its leading columns.
///
/// Reconstruction tests only read the leading `min(m, n)` columns, so this exercises the whole
/// `m x m` direct-written `U`. faer's tall path is accurate for this input, so it does not force
/// the QR repeat; it covers the direct full-factor write.
#[test]
fn full_tall_svd_has_a_unitary_u() {
    let n = 160usize;
    let m = 300usize;
    let spectrum = tlinalg_testkit::clustered_spectrum(n);
    let b: Vec<Complex64> = tlinalg_testkit::with_singular_values(&spectrum, 1);
    // The clustered square matrix padded with zero rows: `m / n > 11 / 6` selects faer's tall path,
    // whose divide-and-conquer attempt fails the reconstruction check and is repeated.
    let mut a = vec![Complex64::new(0.0, 0.0); m * n];
    for col in 0..n {
        for row in 0..n {
            a[row + col * m] = b[row + col * n];
        }
    }
    let (mut u, mut s, mut vt) = (Vec::new(), Vec::new(), Vec::new());
    svd(
        Op::Svd,
        RawStridedRef::new(&a, &[m, n], &[1, m as isize], 0).unwrap(),
        true,
        &mut u,
        &mut s,
        &mut vt,
        Parallel::Sequential,
    )
    .unwrap();
    assert_eq!(u.len(), m * m);
    assert_eq!(vt.len(), n * n);
    let mut worst = 0.0f64;
    for j in 0..m {
        for k in 0..m {
            let mut acc = Complex64::new(0.0, 0.0);
            for i in 0..m {
                acc += u[i + j * m].conj() * u[i + k * m];
            }
            let want = if j == k {
                Complex64::new(1.0, 0.0)
            } else {
                Complex64::new(0.0, 0.0)
            };
            worst = worst.max((acc - want).norm());
        }
    }
    assert!(worst < 1e-10, "full U is not unitary: {worst:e}");
}
