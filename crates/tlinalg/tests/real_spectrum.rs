//! The spectrum of `svd_into` and `eigh_into` is written in the real type, and for a complex input
//! it agrees with the values-only route's real spectrum.
//!
//! The agreement is to rounding, not bit for bit: faer computes the values with and without
//! vectors along different paths (the bidiagonal or tridiagonal QR iteration with and without
//! rotation accumulation, and for SVDs of 128 or more divide and conquer), so the two routes
//! differ by a few `epsilon * max |value|` on most matrices. That difference predates the real
//! output type; the bit-exact checks are that `*_into`, `*_into_init` and the `Vec` wrappers agree
//! (`into_init.rs`, `alloc_counts.rs`).

use core::mem::MaybeUninit;

use num_complex::{Complex32, Complex64};
use strided_view::RawStridedRef;
use tcontract_exec::Exec;
use tlinalg::eigh::{eigh_into, eigh_values_into, eigh_values_workspace, eigh_workspace};
use tlinalg::svd::{svd_into, svd_values_into, svd_values_workspace, svd_workspace};
use tlinalg::Op;

fn value(i: usize, salt: usize) -> f64 {
    ((i * 7919 + salt) % 101) as f64 / 17.0 - 2.5
}

fn uninit<T>(len: usize) -> Vec<MaybeUninit<T>> {
    (0..len).map(|_| MaybeUninit::uninit()).collect()
}

/// Bound on the difference between two backward-stable spectra of one matrix, in
/// `epsilon * max |value|` per value of the item.
const SPECTRUM_FACTOR: f64 = 4.0;

/// # Safety
/// Every element was written.
unsafe fn assume<T>(v: Vec<MaybeUninit<T>>) -> Vec<T> {
    v.into_iter().map(|x| unsafe { x.assume_init() }).collect()
}

/// Per item of `len` values, the largest difference in units of `epsilon * max |value|`, checked
/// against `SPECTRUM_FACTOR * len`.
macro_rules! assert_spectra_agree {
    ($r:ty, $got:expr, $want:expr, $len:expr, $what:expr) => {{
        let (got, want, len): (&[$r], &[$r], usize) = ($got, $want, $len);
        assert_eq!(got.len(), want.len());
        for (g, w) in got.chunks(len.max(1)).zip(want.chunks(len.max(1))) {
            let scale = w
                .iter()
                .fold(0.0 as $r, |m, x| m.max(x.abs()))
                .max(<$r>::MIN_POSITIVE);
            let worst = g
                .iter()
                .zip(w)
                .map(|(a, b)| ((a - b).abs() / (scale * <$r>::EPSILON)) as f64)
                .fold(0.0, f64::max);
            assert!(
                worst <= SPECTRUM_FACTOR * len as f64,
                "{}: spectra differ by {worst} eps * max|value|",
                $what
            );
        }
    }};
}

macro_rules! complex_svd_spectrum_agrees {
    ($name:ident, $c:ty, $r:ty) => {
        #[test]
        fn $name() {
            // Small shapes take the QR algorithm; 130 x 130 takes faer's divide-and-conquer path
            // when vectors are computed, which the values-only route never does.
            for (m, n, batch) in [(4, 3, 2), (3, 5, 3), (7, 7, 1), (130, 130, 1)] {
                let a: Vec<$c> = (0..m * n * batch)
                    .map(|i| <$c>::new(value(i, 3) as $r, value(i, 41) as $r))
                    .collect();
                let (dims, strides) = ([m, n, batch], [1, m as isize, (m * n) as isize]);
                let input = RawStridedRef::new(&a, &dims, &strides, 0).unwrap();
                let k = m.min(n);
                for full in [false, true] {
                    let (u_cols, v_cols) = if full { (m, n) } else { (k, k) };
                    let mut exec = Exec::serial();
                    let mut ws =
                        uninit::<u8>(svd_workspace::<$c>(m, n, full, batch, &exec).unwrap());
                    let mut u = uninit::<$c>(m * u_cols * batch);
                    let mut s = uninit::<$r>(k * batch);
                    let mut vt = uninit::<$c>(v_cols * n * batch);
                    svd_into(
                        Op::Svd,
                        input,
                        full,
                        &mut u,
                        &mut s,
                        &mut vt,
                        &mut ws,
                        &mut exec,
                    )
                    .unwrap();
                    // SAFETY: `svd_into` returned `Ok`.
                    let s = unsafe { assume(s) };

                    let mut ws =
                        uninit::<u8>(svd_values_workspace::<$c>(m, n, batch, &exec).unwrap());
                    let mut values = uninit::<$r>(k * batch);
                    svd_values_into(Op::SvdValues, input, &mut values, &mut ws, &mut exec).unwrap();
                    // SAFETY: `svd_values_into` returned `Ok`.
                    let values = unsafe { assume(values) };
                    assert_spectra_agree!(
                        $r,
                        &s,
                        &values,
                        k,
                        format!("{m}x{n} batch {batch} full={full}")
                    );
                }
            }
        }
    };
}

complex_svd_spectrum_agrees!(
    c64_svd_spectrum_agrees_with_the_values_route,
    Complex64,
    f64
);
complex_svd_spectrum_agrees!(
    c32_svd_spectrum_agrees_with_the_values_route,
    Complex32,
    f32
);

macro_rules! hermitian_eigh_spectrum_agrees {
    ($name:ident, $c:ty, $r:ty) => {
        #[test]
        fn $name() {
            for (n, batch) in [(1, 1), (4, 3), (9, 2), (40, 1), (100, 1), (200, 1)] {
                let mut a = vec![<$c>::new(0.0, 0.0); n * n * batch];
                for b in 0..batch {
                    for j in 0..n {
                        for i in 0..n {
                            let (lo, hi) = (i.min(j), i.max(j));
                            let seed = b * n * n + hi * n + lo;
                            let im = match i.cmp(&j) {
                                core::cmp::Ordering::Less => value(seed, 41),
                                core::cmp::Ordering::Equal => 0.0,
                                core::cmp::Ordering::Greater => -value(seed, 41),
                            };
                            a[b * n * n + j * n + i] = <$c>::new(value(seed, 3) as $r, im as $r);
                        }
                    }
                }
                let (dims, strides) = ([n, n, batch], [1, n as isize, (n * n) as isize]);
                let input = RawStridedRef::new(&a, &dims, &strides, 0).unwrap();
                let mut exec = Exec::serial();

                let mut ws = uninit::<u8>(eigh_workspace::<$c>(n, true, batch, &exec).unwrap());
                let mut w = uninit::<$r>(n * batch);
                let mut v = uninit::<$c>(n * n * batch);
                eigh_into(Op::Eigh, input, &mut w, &mut v, &mut ws, &mut exec).unwrap();
                // SAFETY: `eigh_into` returned `Ok`.
                let (w, v) = unsafe { (assume(w), assume(v)) };

                let mut ws = uninit::<u8>(eigh_values_workspace::<$c>(n, batch, &exec).unwrap());
                let mut reference = uninit::<$r>(n * batch);
                eigh_values_into(Op::EighValues, input, &mut reference, &mut ws, &mut exec)
                    .unwrap();
                // SAFETY: `eigh_values_into` returned `Ok`.
                let reference = unsafe { assume(reference) };

                // The real eigenvalues reproduce `A V = V diag(w)`.
                let tol = 100.0 * <$r>::EPSILON as f64 * n.max(1) as f64 * 8.0;
                for b in 0..batch {
                    let (a, v, w) = (
                        &a[b * n * n..(b + 1) * n * n],
                        &v[b * n * n..(b + 1) * n * n],
                        &w[b * n..(b + 1) * n],
                    );
                    for col in 0..n {
                        for row in 0..n {
                            let mut av = Complex64::new(0.0, 0.0);
                            for p in 0..n {
                                let x = a[row + p * n];
                                let y = v[p + col * n];
                                av += Complex64::new(x.re as f64, x.im as f64)
                                    * Complex64::new(y.re as f64, y.im as f64);
                            }
                            let y = v[row + col * n];
                            let vw = Complex64::new(y.re as f64, y.im as f64) * w[col] as f64;
                            assert!((av - vw).norm() < tol, "A V = V w, n={n} batch item {b}");
                        }
                    }
                }
                assert_spectra_agree!($r, &w, &reference, n, format!("n={n} batch {batch}"));
            }
        }
    };
}

hermitian_eigh_spectrum_agrees!(
    c64_eigh_spectrum_agrees_with_the_values_route,
    Complex64,
    f64
);
hermitian_eigh_spectrum_agrees!(
    c32_eigh_spectrum_agrees_with_the_values_route,
    Complex32,
    f32
);
