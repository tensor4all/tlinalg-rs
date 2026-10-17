//! Behavioural tests for the ported triangular solve: every flag combination on both sides
//! checked by residual, transposed coefficient layout, and shape errors.

mod common;

use common::single::triangular_solve;
use common::*;
use num_complex::Complex64;
use strided_view::RawStridedRef;
use tcontract_exec::Exec;
use tlinalg::triangular_solve::TriangularSolveFlags;
use tlinalg::FaerScalar;
use tlinalg::{Error, Op};

/// The triangle of `a` the flags select, as a dense matrix, with the unit diagonal applied.
fn effective<T: TestScalar + FaerScalar>(
    a: &[T],
    n: usize,
    flags: TriangularSolveFlags,
) -> Vec<Complex64> {
    let mut out = vec![Complex64::new(0.0, 0.0); n * n];
    for col in 0..n {
        for row in 0..n {
            let keep = if flags.lower { row >= col } else { row <= col };
            if keep {
                out[row + col * n] = if row == col && flags.unit_diagonal {
                    Complex64::new(1.0, 0.0)
                } else {
                    a[row + col * n].to_c64()
                };
            }
        }
    }
    if flags.transpose_a {
        transpose(&out, n, n)
    } else {
        out
    }
}

fn all_flags<T: TestScalar + FaerScalar>() {
    let n = 4;
    let nrhs = 3;
    let a = matrix::<T>(n, n, 3);
    for bits in 0..16u8 {
        let flags = TriangularSolveFlags {
            left_side: bits & 1 != 0,
            lower: bits & 2 != 0,
            transpose_a: bits & 4 != 0,
            unit_diagonal: bits & 8 != 0,
        };
        let (b_rows, b_cols) = if flags.left_side {
            (n, nrhs)
        } else {
            (nrhs, n)
        };
        let b = matrix::<T>(b_rows, b_cols, 5);
        let mut workspace = CountingWorkspace;
        let x = triangular_solve(
            Op::TriangularSolve,
            n,
            RawStridedRef::new(&a, &[n, n], &[1, n as isize], 0).unwrap(),
            b.clone(),
            b_rows,
            b_cols,
            flags,
            &mut workspace,
            Exec::serial(),
        )
        .unwrap();
        let op_a = effective(&a, n, flags);
        let x = widen(&x);
        let rebuilt = if flags.left_side {
            matmul(&op_a, &x, n, n, nrhs)
        } else {
            matmul(&x, &op_a, nrhs, n, n)
        };
        assert_close(&rebuilt, &widen(&b), T::TOL, &format!("{flags:?}"));
    }
}
for_each_scalar!(flag_combinations, all_flags);

fn transposed_layout<T: TestScalar + FaerScalar>() {
    // A row-major descriptor of `a` is the column-major matrix `aᵀ`.
    let n = 3;
    let a = matrix::<T>(n, n, 7);
    let at = narrow::<T>(&transpose(&widen(&a), n, n));
    let b = matrix::<T>(n, 2, 1);
    let flags = TriangularSolveFlags {
        left_side: true,
        lower: false,
        transpose_a: false,
        unit_diagonal: false,
    };
    let solve = |data: &[T], strides: [isize; 2]| {
        triangular_solve(
            Op::TriangularSolve,
            n,
            RawStridedRef::new(data, &[n, n], &strides, 0).unwrap(),
            b.clone(),
            n,
            2,
            flags,
            &mut CountingWorkspace,
            Exec::serial(),
        )
        .unwrap()
    };
    let direct = solve(&a, [1, n as isize]);
    let via_rows = solve(&at, [n as isize, 1]);
    assert_close(&widen(&via_rows), &widen(&direct), T::TOL, "row-major A");
}
for_each_scalar!(strided_coefficient, transposed_layout);

#[test]
fn shape_errors() {
    let a = [1.0_f64; 4];
    let flags = TriangularSolveFlags {
        left_side: true,
        lower: true,
        transpose_a: false,
        unit_diagonal: false,
    };
    let solve = |rhs: Vec<f64>, rows, cols, flags| {
        triangular_solve(
            Op::TriangularSolve,
            2,
            RawStridedRef::new(&a, &[2, 2], &[1, 2], 0).unwrap(),
            rhs,
            rows,
            cols,
            flags,
            &mut CountingWorkspace,
            Exec::serial(),
        )
    };
    assert!(matches!(
        solve(vec![1.0; 3], 3, 1, flags),
        Err(Error::InvalidArgument { .. })
    ));
    assert!(matches!(
        solve(vec![1.0; 2], 2, 2, flags),
        Err(Error::InvalidArgument { .. })
    ));
    let right = TriangularSolveFlags {
        left_side: false,
        ..flags
    };
    assert!(matches!(
        solve(vec![1.0; 3], 1, 3, right),
        Err(Error::InvalidArgument { .. })
    ));
    // An empty right-hand side is a no-op.
    assert!(solve(Vec::new(), 2, 0, flags).unwrap().is_empty());
}
