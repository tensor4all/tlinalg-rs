//! Behavioural tests for the ported Cholesky kernel: `L Lᴴ = A`, zero upper triangle, strided
//! input, empty input, and the non-definite failure.

mod common;

use common::single::cholesky;
use common::*;
use num_complex::Complex64;
use strided_view::RawStridedRef;
use tcontract_exec::Exec;
use tlinalg::FaerScalar;
use tlinalg::{Error, Op};

fn reconstructs<T: TestScalar + FaerScalar>() {
    for n in [1, 3, 6] {
        let a = hpd::<T>(n);
        // Read through a padded descriptor: leading dimension and offset must be honoured.
        let (storage, strides, offset) = padded(&a, n, n);
        let dims = [n, n];
        let mut l = Vec::new();
        cholesky(
            Op::Cholesky,
            n,
            RawStridedRef::new(&storage, &dims, &strides, offset).unwrap(),
            &mut l,
            Exec::serial(),
        )
        .unwrap();
        assert_eq!(l.len(), n * n);
        let l = widen(&l);
        for col in 0..n {
            for row in 0..col {
                assert_eq!(l[row + col * n], Complex64::new(0.0, 0.0), "upper entry");
            }
        }
        let rebuilt = matmul(&l, &adjoint(&l, n, n), n, n, n);
        assert_close(&rebuilt, &widen(&a), T::TOL, "L Lᴴ");
    }
}
for_each_scalar!(reconstruction, reconstructs);

fn empty_and_failure<T: TestScalar + FaerScalar>() {
    let mut l = vec![T::default(); 3];
    cholesky::<T>(
        Op::Cholesky,
        0,
        RawStridedRef::new(&[], &[0, 0], &[1, 0], 0).unwrap(),
        &mut l,
        Exec::serial(),
    )
    .unwrap();
    assert!(l.is_empty(), "outputs are cleared");

    // -I is not positive definite.
    let a = narrow::<T>(&[
        Complex64::new(-1.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(-1.0, 0.0),
    ]);
    let err = cholesky(
        Op::Cholesky,
        2,
        RawStridedRef::new(&a, &[2, 2], &[1, 2], 0).unwrap(),
        &mut l,
        Exec::serial(),
    )
    .unwrap_err();
    assert_eq!(err, Error::NonConvergence { op: Op::Cholesky });

    let err = cholesky(
        Op::Cholesky,
        2,
        RawStridedRef::new(&a, &[2, 1], &[1, 2], 0).unwrap(),
        &mut l,
        Exec::serial(),
    )
    .unwrap_err();
    assert!(matches!(err, Error::InvalidArgument { .. }));
}
for_each_scalar!(edge_cases, empty_and_failure);
