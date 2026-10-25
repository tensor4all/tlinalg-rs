//! Behavioural tests for the ported thin QR, column-pivoted QR, and compact Householder kernels.

mod common;

use common::single::{apply_reflectors, compact_factor, qr, rank_revealing_qr};
use common::*;
use num_complex::Complex64;
use strided_view::RawStridedRef;
use tcontract_exec::Exec;
use tlinalg::qr::magnitude;
use tlinalg::FaerScalar;
use tlinalg::{Error, Op};

fn assert_orthonormal_columns(q: &[Complex64], m: usize, k: usize, tol: f64) {
    let gram = matmul(&adjoint(q, m, k), q, k, m, k);
    assert_close(&gram, &identity(k), tol, "Qᴴ Q");
}

fn assert_upper(r: &[Complex64], rows: usize, cols: usize) {
    for col in 0..cols {
        for row in col + 1..rows {
            assert_eq!(
                r[row + col * rows],
                Complex64::new(0.0, 0.0),
                "below diagonal"
            );
        }
    }
}

fn qr_reconstructs<T: TestScalar + FaerScalar>() {
    for (m, n) in [(4, 4), (6, 3), (3, 6)] {
        let a = matrix::<T>(m, n, 31);
        let (storage, strides, offset) = padded(&a, m, n);
        let dims = [m, n];
        let (mut q, mut r) = (Vec::new(), Vec::new());
        qr(
            Op::Qr,
            m,
            n,
            RawStridedRef::new(&storage, &dims, &strides, offset).unwrap(),
            &mut q,
            &mut r,
            &mut Exec::serial(),
        )
        .unwrap();
        let k = m.min(n);
        let (q, r) = (widen(&q), widen(&r));
        assert_eq!((q.len(), r.len()), (m * k, k * n));
        assert_orthonormal_columns(&q, m, k, T::TOL);
        assert_upper(&r, k, n);
        assert_close(&matmul(&q, &r, m, k, n), &widen(&a), T::TOL, "Q R");
    }
}
for_each_scalar!(qr_reconstruction, qr_reconstructs);

fn rrqr_reconstructs<T: TestScalar + FaerScalar>() {
    for (m, n) in [(4, 4), (6, 3), (3, 6)] {
        let a = matrix::<T>(m, n, 37);
        let (mut q, mut r) = (Vec::new(), Vec::new());
        let perm = rank_revealing_qr(
            Op::RankRevealingQr,
            m,
            n,
            RawStridedRef::new(&a, &[m, n], &[1, m as isize], 0).unwrap(),
            &mut q,
            &mut r,
            &mut Exec::serial(),
        )
        .unwrap();
        let k = m.min(n);
        let mut sorted = perm.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, (0..n as i64).collect::<Vec<_>>(), "a permutation");
        // |r_ii| is non-increasing: the pivoting is rank revealing.
        let diag: Vec<f64> = (0..k).map(|i| magnitude(r[i + i * k])).collect();
        for pair in diag.windows(2) {
            assert!(pair[0] + T::TOL >= pair[1], "{diag:?}");
        }
        let (q, r) = (widen(&q), widen(&r));
        assert_orthonormal_columns(&q, m, k, T::TOL);
        assert_upper(&r, k, n);
        let a = widen(&a);
        let mut ap = vec![Complex64::new(0.0, 0.0); m * n];
        for (col, &source) in perm.iter().enumerate() {
            for row in 0..m {
                ap[row + col * m] = a[row + source as usize * m];
            }
        }
        assert_close(&matmul(&q, &r, m, k, n), &ap, T::TOL, "Q R = A P");
    }
}
for_each_scalar!(rrqr_reconstruction, rrqr_reconstructs);

fn householder_roundtrip<T: TestScalar + FaerScalar>() {
    for (rows, cols) in [(5, 3), (3, 5), (4, 4)] {
        let a = matrix::<T>(rows, cols, 41);
        let mut packed = a.clone();
        let mut coeff = Vec::new();
        compact_factor(
            Op::HouseholderQr,
            &mut packed,
            rows,
            cols,
            &mut coeff,
            &mut Exec::serial(),
        )
        .unwrap();
        let k = rows.min(cols);
        assert_eq!(coeff.len(), k);
        for &c in &coeff {
            assert_eq!(c.to_c64().im, 0.0, "coefficients are real");
        }

        // Q applied to the identity is Q itself (full width).
        let mut q = narrow::<T>(&identity(rows));
        apply_reflectors(
            Op::HouseholderQrQColumns,
            &packed,
            cols,
            &coeff,
            &mut q,
            rows,
            rows,
            k,
            false,
            &mut Exec::serial(),
        )
        .unwrap();
        let q = widen(&q);
        assert_orthonormal_columns(&q, rows, rows, T::TOL);
        let packed_w = widen(&packed);
        let mut r = vec![Complex64::new(0.0, 0.0); rows * cols];
        for col in 0..cols {
            for row in 0..k.min(col + 1) {
                r[row + col * rows] = packed_w[row + col * rows];
            }
        }
        assert_close(
            &matmul(&q, &r, rows, rows, cols),
            &widen(&a),
            T::TOL,
            "Q R from compact state",
        );

        // Qᴴ A recovers R.
        let mut qha = a.clone();
        apply_reflectors(
            Op::HouseholderQrAppend,
            &packed,
            cols,
            &coeff,
            &mut qha,
            rows,
            cols,
            k,
            true,
            &mut Exec::serial(),
        )
        .unwrap();
        assert_close(&widen(&qha), &r, T::TOL, "Qᴴ A");
    }
}
for_each_scalar!(householder, householder_roundtrip);

#[test]
fn householder_rejects_bad_dimensions() {
    let mut data = [1.0_f64; 3];
    assert!(matches!(
        compact_factor(
            Op::HouseholderQr,
            &mut data,
            2,
            2,
            &mut Vec::new(),
            &mut Exec::serial()
        ),
        Err(Error::InvalidArgument { .. })
    ));
    let mut c = [1.0_f64; 4];
    assert!(matches!(
        apply_reflectors(
            Op::HouseholderQrQColumns,
            &[1.0; 4],
            2,
            &[1.0],
            &mut c,
            2,
            2,
            3,
            false,
            &mut Exec::serial()
        ),
        Err(Error::InvalidArgument { .. })
    ));
    // Empty inputs are no-ops.
    let mut coeff = vec![1.0_f64];
    compact_factor(
        Op::HouseholderQr,
        &mut [],
        0,
        3,
        &mut coeff,
        &mut Exec::serial(),
    )
    .unwrap();
    assert!(coeff.is_empty());
}

#[test]
fn rank_revealing_qr_returns_the_identity_for_an_all_zero_item() {
    use tcontract_exec::Exec;
    // Item 0 is zero, item 1 is not: the zero item is not factored, the other is.
    let (m, n) = (3, 4);
    let mut a = vec![0.0_f64; m * n];
    a.extend(matrix::<f64>(m, n, 1));
    let (mut q, mut r, mut perm) = (Vec::new(), Vec::new(), Vec::new());
    tlinalg::qr::rank_revealing_qr(
        Op::RankRevealingQr,
        RawStridedRef::new(&a, &[m, n, 2], &[1, m as isize, (m * n) as isize], 0).unwrap(),
        &mut q,
        &mut r,
        &mut perm,
        &mut Exec::serial(),
    )
    .unwrap();
    let k = m.min(n);
    assert_eq!(&q[..m * k], &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]);
    assert!(r[..k * n].iter().all(|&v| v == 0.0));
    assert_eq!(&perm[..n], &[0, 1, 2, 3]);
    assert!(r[k * n..].iter().any(|&v| v != 0.0));
}
