//! Behavioural tests for the ported Hermitian and general eigendecompositions.

mod common;

use common::single::{eig, eig_values, eigh, eigh_values};
use common::*;
use num_complex::{Complex32, Complex64};
use strided_view::RawStridedRef;
use tcontract_exec::Exec;
use tlinalg::FaerScalar;
use tlinalg::Op;

fn eigh_decomposes<T: TestScalar + FaerScalar>() {
    let n = 5;
    let a = hermitian::<T>(n);
    let (storage, strides, offset) = padded(&a, n, n);
    let dims = [n, n];
    let input = RawStridedRef::new(&storage, &dims, &strides, offset).unwrap();
    let (mut w, mut v) = (Vec::new(), Vec::new());
    eigh(Op::Eigh, n, input, &mut w, &mut v, Exec::serial()).unwrap();
    let (w, v) = (widen(&w), widen(&v));
    for value in &w {
        assert_eq!(value.im, 0.0, "real eigenvalues in the scalar type");
    }
    for pair in w.windows(2) {
        assert!(pair[0].re <= pair[1].re, "non-decreasing");
    }
    let mut vw = v.clone();
    for col in 0..n {
        for row in 0..n {
            vw[row + col * n] *= w[col];
        }
    }
    assert_close(&matmul(&widen(&a), &v, n, n, n), &vw, T::TOL, "A V = V w");

    let mut values = Vec::new();
    eigh_values(Op::EighValues, n, input, &mut values, Exec::serial()).unwrap();
    assert_eq!(values.len(), n);
}
for_each_scalar!(eigh_decomposition, eigh_decomposes);

#[test]
fn eigh_values_match_the_decomposition() {
    let n = 4;
    let a = hermitian::<Complex64>(n);
    let (dims, strides) = ([n, n], [1, n as isize]);
    let input = RawStridedRef::new(&a, &dims, &strides, 0).unwrap();
    let (mut w, mut v, mut values) = (Vec::new(), Vec::new(), Vec::new());
    eigh(Op::Eigh, n, input, &mut w, &mut v, Exec::serial()).unwrap();
    eigh_values(Op::EighValues, n, input, &mut values, Exec::serial()).unwrap();
    for (full, only) in w.iter().zip(&values) {
        assert!((full.re - only).abs() < 1e-10);
    }
}

/// Check `A V = V diag(w)` in `Complex64`.
fn check_eig(a: &[Complex64], w: &[Complex64], v: &[Complex64], n: usize, tol: f64) {
    let mut vw = v.to_vec();
    for col in 0..n {
        for row in 0..n {
            vw[row + col * n] *= w[col];
        }
    }
    assert_close(&matmul(a, v, n, n, n), &vw, tol, "A V = V w");
}

fn sort_key(values: &[Complex64]) -> Vec<(i64, i64)> {
    let mut keys: Vec<(i64, i64)> = values
        .iter()
        .map(|v| ((v.re * 1e6).round() as i64, (v.im * 1e6).round() as i64))
        .collect();
    keys.sort_unstable();
    keys
}

#[test]
fn eig_real_input_yields_conjugate_pairs() {
    // Block diagonal: a rotation (eigenvalues 1 ± 2i) and a real eigenvalue 3.
    let n = 3;
    let a = [1.0_f64, 2.0, 0.0, -2.0, 1.0, 0.0, 0.0, 0.0, 3.0];
    let (dims, strides) = ([n, n], [1, n as isize]);
    let input = RawStridedRef::new(&a, &dims, &strides, 0).unwrap();
    let (mut w, mut v) = (Vec::<Complex64>::new(), Vec::new());
    eig(Op::Eig, n, input, &mut w, &mut v, Exec::serial()).unwrap();
    let a_c: Vec<Complex64> = a.iter().map(|&x| Complex64::new(x, 0.0)).collect();
    check_eig(&a_c, &w, &v, n, 1e-10);
    assert_eq!(
        sort_key(&w),
        sort_key(&[
            Complex64::new(1.0, -2.0),
            Complex64::new(1.0, 2.0),
            Complex64::new(3.0, 0.0)
        ])
    );
    // The real eigenvalue is exactly real; the pair is exactly conjugate.
    let real = w.iter().find(|v| (v.re - 3.0).abs() < 1e-8).unwrap();
    assert_eq!(real.im, 0.0);

    let mut values = Vec::new();
    eig_values(Op::EigValues, n, input, &mut values, Exec::serial()).unwrap();
    assert_eq!(sort_key(&values), sort_key(&w));
}

#[test]
fn eig_real32_and_complex() {
    let n = 4;
    let a = matrix::<f32>(n, n, 3);
    let (mut w, mut v) = (Vec::<Complex32>::new(), Vec::new());
    eig(
        Op::Eig,
        n,
        RawStridedRef::new(&a, &[n, n], &[1, n as isize], 0).unwrap(),
        &mut w,
        &mut v,
        Exec::serial(),
    )
    .unwrap();
    check_eig(&widen(&a), &widen(&w), &widen(&v), n, 1e-4);

    for seed in [1, 2] {
        let a = matrix::<Complex64>(n, n, seed);
        let (storage, strides, offset) = padded(&a, n, n);
        let dims = [n, n];
        let input = RawStridedRef::new(&storage, &dims, &strides, offset).unwrap();
        let (mut w, mut v) = (Vec::<Complex64>::new(), Vec::new());
        eig(Op::Eig, n, input, &mut w, &mut v, Exec::serial()).unwrap();
        check_eig(&a, &w, &v, n, 1e-10);
        let mut values = Vec::new();
        eig_values(Op::EigValues, n, input, &mut values, Exec::serial()).unwrap();
        assert_eq!(sort_key(&values), sort_key(&w));
    }

    let a = matrix::<Complex32>(n, n, 5);
    let (mut w, mut v) = (Vec::<Complex32>::new(), Vec::new());
    eig(
        Op::Eig,
        n,
        RawStridedRef::new(&a, &[n, n], &[1, n as isize], 0).unwrap(),
        &mut w,
        &mut v,
        Exec::serial(),
    )
    .unwrap();
    check_eig(&widen(&a), &widen(&w), &widen(&v), n, 1e-4);
}

#[test]
fn empty_inputs_clear_the_outputs() {
    let mut w = vec![Complex64::new(1.0, 0.0)];
    let mut v = vec![Complex64::new(1.0, 0.0)];
    let empty = RawStridedRef::<f64>::new(&[], &[0, 0], &[1, 0], 0).unwrap();
    eig(Op::Eig, 0, empty, &mut w, &mut v, Exec::serial()).unwrap();
    assert!(w.is_empty() && v.is_empty());
    let mut real = vec![1.0_f64];
    let mut values = vec![1.0_f64];
    eigh(Op::Eigh, 0, empty, &mut real, &mut values, Exec::serial()).unwrap();
    assert!(real.is_empty() && values.is_empty());
}
