//! Behavioural tests for the batched LAPACK SVD.
//!
//! Requires a linked LAPACK: run with `--features link-openblas` (with and without
//! `provider-inject`, which selects `?gesvd` instead of `?gesdd`).

#![cfg(feature = "link-openblas")]

mod common;

use common::*;
use num_complex::{Complex32, Complex64};
use strided_view::RawStridedRef;
use tlinalg_blas::svd::{svd, SvdMode, SvdOutputs};
use tlinalg_blas::{Error, Op};

fn check<T: TestScalar>(m: usize, n: usize, mode: SvdMode)
where
    T::Real: TestScalar,
{
    let k = m.min(n);
    let (u_cols, vt_rows) = match mode {
        SvdMode::Thin => (k, k),
        SvdMode::Full => (m, n),
        SvdMode::Values => (0, 0),
    };
    // Two matrices in the batch, so the loop and the once-per-call workspace are both exercised.
    let batch = 2;
    let data: Vec<T> = (0..batch)
        .flat_map(|seed| matrix::<T>(m, n, seed))
        .collect();
    let dims = [m, n, batch];
    let strides = [1, m as isize, (m * n) as isize];
    let a = RawStridedRef::new(&data, &dims, &strides, 0).unwrap();
    let (mut s, mut u, mut vt) = (Vec::new(), Vec::new(), Vec::new());
    let mut ws = TestWorkspace::default();
    svd(
        Op::Svd,
        mode,
        a,
        SvdOutputs {
            s: &mut s,
            u: &mut u,
            vt: &mut vt,
        },
        &mut ws,
    )
    .unwrap();
    assert_eq!(ws.outstanding(), 0, "every buffer must come back");
    assert_eq!(s.len(), k * batch);
    assert_eq!(u.len(), m * u_cols * batch);
    assert_eq!(vt.len(), vt_rows * n * batch);

    for index in 0..batch {
        let s_i = widen(&s[index * k..(index + 1) * k]);
        for pair in s_i.windows(2) {
            assert!(pair[0].re >= pair[1].re, "{m}x{n} {mode:?}: not sorted");
        }
        if mode == SvdMode::Values {
            continue;
        }
        let u_i = widen(&u[index * m * u_cols..(index + 1) * m * u_cols]);
        let vt_i = widen(&vt[index * vt_rows * n..(index + 1) * vt_rows * n]);
        let mut us = vec![Complex64::new(0.0, 0.0); m * k];
        for col in 0..k {
            for row in 0..m {
                us[row + col * m] = u_i[row + col * m] * s_i[col];
            }
        }
        // `Vᴴ` is stored, so `A = U diag(S) Vᴴ` multiplies the leading `k` rows directly.
        let mut vt_k = vec![Complex64::new(0.0, 0.0); k * n];
        for col in 0..n {
            for row in 0..k {
                vt_k[row + col * k] = vt_i[row + col * vt_rows];
            }
        }
        assert_close(
            &matmul(&us, &vt_k, m, k, n),
            &widen(&data[index * m * n..(index + 1) * m * n]),
            T::LOOSE_TOL,
            "U S Vᴴ",
        );
    }
}

fn every_mode_and_shape<T: TestScalar>()
where
    T::Real: TestScalar,
{
    for (m, n) in [(4usize, 4usize), (6, 3), (3, 6), (1, 4)] {
        for mode in [SvdMode::Thin, SvdMode::Full, SvdMode::Values] {
            check::<T>(m, n, mode);
        }
    }
}

#[test]
fn svd_reconstructs_f32() {
    every_mode_and_shape::<f32>();
}

#[test]
fn svd_reconstructs_f64() {
    every_mode_and_shape::<f64>();
}

#[test]
fn svd_reconstructs_c32() {
    every_mode_and_shape::<Complex32>();
}

#[test]
fn svd_reconstructs_c64() {
    every_mode_and_shape::<Complex64>();
}

#[test]
fn an_empty_batch_leaves_every_output_empty() {
    let data: Vec<f64> = Vec::new();
    let dims = [3, 3, 0];
    let strides = [1, 3, 9];
    let a = RawStridedRef::new(&data, &dims, &strides, 0).unwrap();
    let (mut s, mut u, mut vt) = (vec![1.0], vec![1.0], vec![1.0]);
    let mut ws = TestWorkspace::default();
    svd(
        Op::Svd,
        SvdMode::Thin,
        a,
        SvdOutputs {
            s: &mut s,
            u: &mut u,
            vt: &mut vt,
        },
        &mut ws,
    )
    .unwrap();
    assert!(s.is_empty() && u.is_empty() && vt.is_empty());
    assert_eq!(ws.acquired, 0);
}

/// A zero-stride descriptor whose logical element count overflows `usize` reports the `shape` role.
///
/// `RawStridedRef::new` bounds-checks the *reachable* offsets, not the logical element count, so a
/// zero-stride layout passes it and the kernel's own product check is what rejects the input. `svd`
/// used to carry its own `checked_product` that said `configuration` instead (tlinalg-rs#79). The
/// check runs before any LAPACK call, so no workspace is acquired either.
#[test]
fn an_overflowing_svd_matrix_reports_the_shape_role() {
    let data = [0.0_f64];
    let dims = [usize::MAX / 2 + 1, 2];
    let strides = [0_isize, 0];
    let a = RawStridedRef::new(&data, &dims, &strides, 0).unwrap();
    let (mut s, mut u, mut vt) = (Vec::new(), Vec::new(), Vec::new());
    let mut ws = TestWorkspace::default();
    let error = svd(
        Op::Svd,
        SvdMode::Thin,
        a,
        SvdOutputs {
            s: &mut s,
            u: &mut u,
            vt: &mut vt,
        },
        &mut ws,
    )
    .expect_err("a logical element count that overflows usize must be rejected");
    match error {
        Error::InvalidArgument { role, detail, .. } => {
            assert_eq!(role, "shape", "role");
            assert_eq!(detail, "matrix element count overflow", "detail");
        }
        other => panic!("expected InvalidArgument, got {other:?}"),
    }
    assert_eq!(
        ws.acquired, 0,
        "the argument check runs before any workspace"
    );
}
