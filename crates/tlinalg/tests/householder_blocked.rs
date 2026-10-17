//! The compact Householder kernels against the convention they publish, at shapes that block.
//!
//! `compact_factor` and `apply_reflectors` now run their reflectors a panel at a time through faer's
//! compact-WY form. The rest of the suite only uses fixtures small enough not to be blocked (the
//! kernel's threshold is `rows * cols <= 48 * 48`), so it never reaches the block path; this file
//! keeps shapes that do, including a partial last block, and checks the
//! stored state against a reference that applies reflectors one at a time straight from the
//! documented `H = I - v vᴴ / τ` and `coeff = 1/τ`.
//!
//! The reference mirrors the branch structure and `τ` formula of faer's
//! `linalg::householder::make_householder_imp`, which is what the kernel under test calls; the
//! reflector *application* it uses is the documented rank-one map, not faer's block form, so the
//! two sides of the comparison do not share an implementation.

mod common;

use common::*;
use num_complex::{Complex64, ComplexFloat};
use tcontract_exec::Exec;
use tlinalg::householder::{apply_reflectors, compact_factor, ReflectorShape};
use tlinalg::Op;

/// One `rows x cols` matrix factored the way the compact contract defines it: reflector `j` is
/// built from column `j` and applied to columns `j + 1..` before the next one is built.
///
/// Returns the packed state and the coefficients, column-major.
fn reference_factor(
    input: &[Complex64],
    rows: usize,
    cols: usize,
) -> (Vec<Complex64>, Vec<Complex64>) {
    let k = rows.min(cols);
    let mut a = input.to_vec();
    let mut coeff = Vec::with_capacity(k);
    for j in 0..k {
        let tail_norm = (j + 1..rows)
            .map(|row| a[row + j * rows].norm_sqr())
            .sum::<f64>()
            .sqrt();
        if tail_norm < f64::MIN_POSITIVE {
            // A zero tail: the reflector is the identity, `τ` is infinite and `1/τ` is zero. The
            // column is left exactly as it was found.
            coeff.push(Complex64::new(0.0, 0.0));
            continue;
        }
        let mut head = a[j + j * rows];
        let mut head_abs = head.abs();
        if head_abs < f64::MIN_POSITIVE {
            head = Complex64::new(0.0, 0.0);
            head_abs = 0.0;
        }
        let norm = head_abs.hypot(tail_norm);
        let sign = if head_abs != 0.0 {
            head / head_abs
        } else {
            Complex64::new(1.0, 0.0)
        };
        let signed = sign * norm;
        let inv = 1.0 / (head + signed);
        for row in j + 1..rows {
            a[row + j * rows] *= inv;
        }
        let tau = 0.5 * (1.0 + (tail_norm * inv.norm()).powi(2));
        // The diagonal carries `β` only after the reflector has been applied, so it is `1` while
        // the basis is in use.
        a[j + j * rows] = Complex64::new(1.0, 0.0);
        let basis: Vec<Complex64> = (0..rows - j).map(|i| a[j + i + j * rows]).collect();
        for col in j + 1..cols {
            let dot: Complex64 = (0..rows - j)
                .map(|i| basis[i].conj() * a[j + i + col * rows])
                .sum();
            let scale = -dot / tau;
            for (i, &v) in basis.iter().enumerate() {
                a[j + i + col * rows] += scale * v;
            }
        }
        a[j + j * rows] = -signed;
        coeff.push(Complex64::new(1.0 / tau, 0.0));
    }
    (a, coeff)
}

/// Apply the first `k` reflectors of one compact state to one target, one reflector at a time.
///
/// `Q C` walks the reflectors backwards, `Qᴴ C` forwards; `τ` is real, so a reflector and its
/// adjoint are the same map.
fn reference_apply(
    a: &[Complex64],
    coeff: &[Complex64],
    c: &[Complex64],
    rows: usize,
    cols: usize,
    k: usize,
    transpose: bool,
) -> Vec<Complex64> {
    let mut c = c.to_vec();
    let order: Vec<usize> = if transpose {
        (0..k).collect()
    } else {
        (0..k).rev().collect()
    };
    for j in order {
        let tau = 1.0 / coeff[j].re;
        let basis: Vec<Complex64> = (0..rows - j)
            .map(|i| {
                if i == 0 {
                    Complex64::new(1.0, 0.0)
                } else {
                    a[j + i + j * rows]
                }
            })
            .collect();
        let n = rows - j;
        for col in 0..cols {
            let dot: Complex64 = (0..n)
                .map(|i| basis[i].conj() * c[j + i + col * rows])
                .sum();
            let scale = -dot / tau;
            for (i, &v) in basis.iter().enumerate() {
                c[j + i + col * rows] += scale * v;
            }
        }
    }
    c
}

/// Shapes whose faer-recommended block size is above one, including a partial last block, plus the
/// unblocked case, so both the block path and its absence are covered.
const SHAPES: [(usize, usize); 5] = [(64, 48), (48, 64), (71, 43), (130, 130), (5, 3)];

fn factor_matches_the_reference<T: TestScalar + tlinalg::FaerScalar>() {
    let (batch, par) = (3, Exec::serial());
    for (rows, cols) in SHAPES {
        let k = rows.min(cols);
        let a: Vec<T> = (0..batch)
            .flat_map(|seed| matrix::<T>(rows, cols, seed))
            .collect();
        let (mut data, mut coeff) = (a.clone(), Vec::new());
        compact_factor(
            Op::HouseholderQr,
            rows,
            cols,
            batch,
            &mut data,
            &mut coeff,
            par,
        )
        .unwrap();
        assert_eq!(coeff.len(), batch * k);
        for item in 0..batch {
            let (want, want_coeff) = reference_factor(
                &widen(&a[item * rows * cols..(item + 1) * rows * cols]),
                rows,
                cols,
            );
            let got = widen(&data[item * rows * cols..(item + 1) * rows * cols]);
            assert_close(&got, &want, T::TOL, "packed state");
            assert_close(
                &widen(&coeff[item * k..(item + 1) * k]),
                &want_coeff,
                T::TOL,
                "coefficients",
            );
        }
    }
}
for_each_scalar!(householder_blocked_factor, factor_matches_the_reference);

fn apply_matches_the_reference<T: TestScalar + tlinalg::FaerScalar>() {
    // A batch, so the per-lane factor scratch is reused across items, and a prefix shorter than the
    // full reflector count, so `shape.k < a_cols` is exercised too.
    let (batch, par) = (3, Exec::serial());
    for (rows, cols) in SHAPES {
        let k = rows.min(cols);
        for k_used in [k, k / 2 + 1] {
            let a: Vec<T> = (0..batch)
                .flat_map(|seed| matrix::<T>(rows, cols, seed + 3))
                .collect();
            let (mut data, mut coeff) = (a.clone(), Vec::new());
            compact_factor(
                Op::HouseholderQr,
                rows,
                cols,
                batch,
                &mut data,
                &mut coeff,
                par,
            )
            .unwrap();
            let c: Vec<T> = (0..batch)
                .flat_map(|seed| matrix::<T>(rows, k_used, seed + 11))
                .collect();
            // `apply_reflectors` takes exactly `shape.k` coefficients per item, so a prefix is
            // re-packed the way a host holding a shorter sequence would hold it.
            let prefix: Vec<T> = (0..batch)
                .flat_map(|item| coeff[item * k..item * k + k_used].iter().copied())
                .collect();
            for transpose in [false, true] {
                let mut got = c.clone();
                let shape = ReflectorShape {
                    rows,
                    a_cols: cols,
                    cols: k_used,
                    k: k_used,
                };
                apply_reflectors(
                    Op::HouseholderQrQColumns,
                    shape,
                    batch,
                    &data,
                    &prefix,
                    &mut got,
                    transpose,
                    par,
                )
                .unwrap();
                for item in 0..batch {
                    let want = reference_apply(
                        &widen(&data[item * rows * cols..(item + 1) * rows * cols]),
                        &widen(&prefix[item * k_used..(item + 1) * k_used]),
                        &widen(&c[item * rows * k_used..(item + 1) * rows * k_used]),
                        rows,
                        k_used,
                        k_used,
                        transpose,
                    );
                    assert_close(
                        &widen(&got[item * rows * k_used..(item + 1) * rows * k_used]),
                        &want,
                        T::TOL,
                        "reflector apply",
                    );
                }
            }
        }
    }
}
for_each_scalar!(householder_blocked_apply, apply_matches_the_reference);

/// A column whose tail is zero has an infinite `τ`, so its compact-WY factor's diagonal is
/// infinite and `1/τ` is zero. Both the all-zero matrix (every reflector an identity) and a single
/// zero column must still give the reference's state, at shapes that block.
fn zero_columns_match_the_reference<T: TestScalar + tlinalg::FaerScalar>() {
    let zero = T::from_c64(Complex64::new(0.0, 0.0));
    for (rows, cols, zeroed) in [(64, 48, vec![3]), (64, 48, vec![0, 47]), (64, 48, vec![])] {
        let mut a = matrix::<T>(rows, cols, 5);
        if zeroed.is_empty() {
            // The all-zero matrix: every reflector is the identity.
            a.iter_mut().for_each(|v| *v = zero);
        }
        for &col in &zeroed {
            for row in 0..rows {
                a[row + col * rows] = zero;
            }
        }
        let (mut data, mut coeff) = (a.clone(), Vec::new());
        compact_factor(
            Op::HouseholderQr,
            rows,
            cols,
            1,
            &mut data,
            &mut coeff,
            Exec::serial(),
        )
        .unwrap();
        let (want, want_coeff) = reference_factor(&widen(&a), rows, cols);
        assert_close(&widen(&data), &want, T::TOL, "packed state");
        assert_close(&widen(&coeff), &want_coeff, T::TOL, "coefficients");
    }
}
for_each_scalar!(
    householder_blocked_zero_columns,
    zero_columns_match_the_reference
);
