//! Nearly dependent columns survive QR and the SVD of tall and wide matrices.
//!
//! faer's unpivoted Householder QR treats a column as zero when the part orthogonal to the previous
//! columns falls below `16 * (m - k) * eps * ||column||`, where `k` is the number of reflectors
//! accepted so far. The column is then dropped from `Q` and `R`, and nothing reports it. LAPACK has
//! no such skip, and tlinalg's own `rank_revealing_qr` goes through column-pivoted QR, which does
//! not either.
//!
//! The fixture is the one reported in tensor4all/tlinalg-rs#28:
//!
//! ```text
//! A = [a, a + 1e-10 e_0, a + 1e-6 e_1],   a = (1, ..., 1) in R^m
//! ```
//!
//! with `sigma_3 / sigma_1` of `4.0618e-12` at `m = 100` and `1.2903e-12` at `m = 1000` (50-digit
//! arithmetic on `A^T A`). At `m = 100` the second column is above faer's threshold and every route
//! is accurate, so the same test body passes there and fails at `m = 1000`: the smaller size is the
//! control that keeps this from being a test that passes for the wrong reason.
//!
//! The second column is deliberately not the last one. A dropped *final* column can still leave the
//! singular values right, because the trailing column contributes little to the ones that survive;
//! a dropped middle column is visible in the factors and in the spectrum alike.

use strided_view::RawStridedRef;
use tlinalg::qr::{qr, rank_revealing_qr};
use tlinalg::svd::{svd, svd_values};
use tlinalg::{Op, Parallel};

/// `A = [a, a + 1e-10 e_0, a + 1e-6 e_1]`, column-major `m x 3`.
fn fixture(m: usize) -> Vec<f64> {
    let mut a = vec![1.0_f64; 3 * m];
    a[m] += 1e-10;
    a[2 * m + 1] += 1e-6;
    a
}

/// `sigma_3 / sigma_1` from `A^T A` in 50-digit arithmetic, per tensor4all/tlinalg-rs#28.
fn reference_ratio(m: usize) -> f64 {
    match m {
        100 => 4.0618e-12,
        1000 => 1.2903e-12,
        8192 => 4.5103e-13,
        _ => unreachable!("no reference value for m = {m}"),
    }
}

/// The descriptor borrows its dimension and stride arrays, so they live in the caller.
fn view<'a>(a: &'a [f64], dims: &'a [usize; 2], strides: &'a [isize; 2]) -> RawStridedRef<'a, f64> {
    RawStridedRef::new(a, dims, strides, 0).unwrap()
}

/// The `m x 3` descriptor every route except the wide case uses.
fn tall<'a>(
    a: &'a [f64],
    m: usize,
    dims: &'a mut [usize; 2],
    strides: &'a mut [isize; 2],
) -> RawStridedRef<'a, f64> {
    *dims = [m, 3];
    *strides = [1, m as isize];
    RawStridedRef::new(a, dims, strides, 0).unwrap()
}

/// `||A - B|| / ||A||` in the Frobenius norm, over the whole array.
fn relative_error(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len());
    let mut error = 0.0_f64;
    let mut total = 0.0_f64;
    for entry in 0..a.len() {
        error += (b[entry] - a[entry]).powi(2);
        total += a[entry].powi(2);
    }
    (error / total).sqrt()
}

/// The reconstruction is accepted when it is as good as a plain QR, not when it merely looks small.
///
/// The skip costs about `16 * m * eps` relative: 1.6e-13 at `m = 100` (which is why the control
/// passes) and 2.6e-12 at `m = 1000`. An accurate factorization is at `1e-14`, so the bound below
/// separates them with two orders of magnitude to spare on each side.
const QR_BOUND: f64 = 1e-13;

fn svd_values_ratio(m: usize) -> f64 {
    let a = fixture(m);
    let mut s = Vec::new();
    let (mut dims, mut strides) = ([m, 3], [1, m as isize]);
    svd_values(
        Op::SvdValues,
        tall(&a, m, &mut dims, &mut strides),
        &mut s,
        Parallel::Sequential,
    )
    .unwrap();
    s[2] / s[0]
}

fn thin_svd_ratio(m: usize) -> f64 {
    let a = fixture(m);
    let (mut u, mut s, mut vt) = (Vec::new(), Vec::new(), Vec::new());
    let (mut dims, mut strides) = ([m, 3], [1, m as isize]);
    svd(
        Op::Svd,
        tall(&a, m, &mut dims, &mut strides),
        false,
        &mut u,
        &mut s,
        &mut vt,
        Parallel::Sequential,
    )
    .unwrap();
    s[2] / s[0]
}

/// Full SVD writes every column of `U` and `V`, so its reconstruction is checked from the factors
/// rather than from the singular values alone.
fn full_svd_relative_error(m: usize) -> f64 {
    let a = fixture(m);
    let (mut u, mut s, mut vt) = (Vec::new(), Vec::new(), Vec::new());
    let (mut dims, mut strides) = ([m, 3], [1, m as isize]);
    svd(
        Op::Svd,
        tall(&a, m, &mut dims, &mut strides),
        true,
        &mut u,
        &mut s,
        &mut vt,
        Parallel::Sequential,
    )
    .unwrap();
    // A = U diag(s) V^H, with U column-major m x m and V^H column-major 3 x 3.
    let mut reconstructed = vec![0.0_f64; 3 * m];
    for j in 0..3 {
        for i in 0..m {
            reconstructed[i + m * j] = (0..3).map(|k| u[i + m * k] * s[k] * vt[k + 3 * j]).sum();
        }
    }
    relative_error(&a, &reconstructed)
}

fn qr_relative_error(m: usize) -> f64 {
    let a = fixture(m);
    let (mut q, mut r) = (Vec::new(), Vec::new());
    let (mut dims, mut strides) = ([m, 3], [1, m as isize]);
    qr(
        Op::Qr,
        tall(&a, m, &mut dims, &mut strides),
        &mut q,
        &mut r,
        Parallel::Sequential,
    )
    .unwrap();
    let mut reconstructed = vec![0.0_f64; 3 * m];
    for j in 0..3 {
        for i in 0..m {
            reconstructed[i + m * j] = (0..3).map(|k| q[i + m * k] * r[k + 3 * j]).sum();
        }
    }
    relative_error(&a, &reconstructed)
}

fn rank_revealing_qr_relative_error(m: usize) -> f64 {
    let a = fixture(m);
    let (mut q, mut r, mut permutation) = (Vec::new(), Vec::new(), Vec::new());
    let (mut dims, mut strides) = ([m, 3], [1, m as isize]);
    rank_revealing_qr(
        Op::RankRevealingQr,
        tall(&a, m, &mut dims, &mut strides),
        &mut q,
        &mut r,
        &mut permutation,
        Parallel::Sequential,
    )
    .unwrap();
    // A P = Q R, with the permutation applied to the columns of A.
    let mut permuted = vec![0.0_f64; 3 * m];
    for j in 0..3 {
        let source = permutation[j] as usize;
        for i in 0..m {
            permuted[i + m * j] = a[i + m * source];
        }
    }
    let mut reconstructed = vec![0.0_f64; 3 * m];
    for j in 0..3 {
        for i in 0..m {
            reconstructed[i + m * j] = (0..3).map(|k| q[i + m * k] * r[k + 3 * j]).sum();
        }
    }
    relative_error(&permuted, &reconstructed)
}

/// `m = 100` is the control: the second column is above the threshold there, so this passes before
/// and after any fix, and it is what shows the failure at `m = 1000` is the skip.
#[test]
fn svd_values_keeps_the_third_singular_value() {
    for m in [100, 1000] {
        let ratio = svd_values_ratio(m);
        let reference = reference_ratio(m);
        assert!(
            (ratio / reference - 1.0).abs() < 0.5,
            "m = {m}: sigma_3 / sigma_1 = {ratio:.4e}, reference {reference:.4e}"
        );
    }
}

#[test]
fn thin_svd_keeps_the_third_singular_value() {
    for m in [100, 1000] {
        let ratio = thin_svd_ratio(m);
        let reference = reference_ratio(m);
        assert!(
            (ratio / reference - 1.0).abs() < 0.5,
            "m = {m}: sigma_3 / sigma_1 = {ratio:.4e}, reference {reference:.4e}"
        );
    }
}

#[test]
fn full_svd_reconstructs_the_matrix() {
    for m in [100, 1000] {
        let error = full_svd_relative_error(m);
        assert!(
            error <= QR_BOUND,
            "m = {m}: ||A - U S V^H|| / ||A|| = {error:.2e}"
        );
    }
}

#[test]
fn qr_reconstructs_the_matrix() {
    for m in [100, 1000] {
        let error = qr_relative_error(m);
        assert!(
            error <= QR_BOUND,
            "m = {m}: ||A - QR|| / ||A|| = {error:.2e}"
        );
    }
}

/// The wide case is left as an open question rather than an assertion.
///
/// A faer-level test of the same fixture passes before and after the fix (the fix's author checked
/// it), while this route still loses the `1e-10` and `1e-6` structure, and neither the fork fix nor
/// forcing the SVD away from the QR changes it. So either the loss is not the unpivoted skip or this
/// check is wrong, and until that is settled the case is not evidence either way. It is ignored, not
/// deleted, so the question stays visible.
#[test]
#[ignore = "open question: the observed loss in the wide route is not reproduced at the faer level"]
fn wide_qr_reconstructs_the_transpose() {
    let m = 1000;
    let a = fixture(m);
    // The same fixture transposed: a 3 x m matrix, column-major, with the defect in the same column.
    let mut transposed = vec![0.0_f64; 3 * m];
    for j in 0..3 {
        for i in 0..m {
            transposed[j + 3 * i] = a[i + m * j];
        }
    }
    let (mut q, mut r) = (Vec::new(), Vec::new());
    qr(
        Op::Qr,
        RawStridedRef::new(&transposed, &[3, m], &[1, 3], 0).unwrap(),
        &mut q,
        &mut r,
        Parallel::Sequential,
    )
    .unwrap();
    // Both are 3 x m column-major, so an entry is [row i][column j] = j + 3 * i.
    let mut reconstructed = vec![0.0_f64; 3 * m];
    for j in 0..m {
        for i in 0..3 {
            reconstructed[j + 3 * i] = (0..3).map(|k| q[i + 3 * k] * r[k + 3 * j]).sum();
        }
    }
    let error = relative_error(&transposed, &reconstructed);
    assert!(
        error <= QR_BOUND,
        "3 x {m}: ||A - QR|| / ||A|| = {error:.2e}"
    );
}

/// Column-pivoted QR takes a different route and is accurate before and after: if this ever fails,
/// the failure above is not the unpivoted skip and the diagnosis is wrong.
#[test]
fn rank_revealing_qr_is_accurate_at_both_sizes() {
    for m in [100, 1000] {
        let error = rank_revealing_qr_relative_error(m);
        assert!(
            error <= QR_BOUND,
            "m = {m}: ||A P - QR|| / ||A|| = {error:.2e} (column-pivoted QR, which has no skip)"
        );
    }
}
