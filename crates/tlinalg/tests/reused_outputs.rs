//! Outputs written in place: a call into vectors that already hold capacity and stale contents
//! gives the same result as a call into fresh vectors, for one matrix and for a batch.

use num_complex::Complex64;
use strided_view::RawStridedRef;
use tcontract_exec::Exec;
use tlinalg::eigh::eigh;
use tlinalg::qr::{qr, rank_revealing_qr};
use tlinalg::svd::svd;
use tlinalg::{FaerScalar, Op};

fn input<T: FaerScalar>(m: usize, n: usize, batch: usize, entry: fn(f64) -> T) -> Vec<T> {
    (0..m * n * batch)
        .map(|i| {
            entry(((i * 7919) % 101) as f64 / 17.0 - 2.5 + if i % (m + 1) == 0 { 4.0 } else { 0.0 })
        })
        .collect()
}

fn run_twice<T: FaerScalar + PartialEq + core::fmt::Debug>(
    dims: [usize; 3],
    data: &[T],
    stale: T,
    call: impl Fn(RawStridedRef<'_, T>, &mut [Vec<T>]),
    outputs: usize,
) {
    let [m, n, batch] = dims;
    let (dims, strides) = ([m, n, batch], [1, m as isize, (m * n) as isize]);
    let view = RawStridedRef::new(data, &dims, &strides, 0).unwrap();
    let mut fresh = vec![Vec::new(); outputs];
    call(view, &mut fresh);
    let mut reused: Vec<Vec<T>> = fresh.iter().map(|v| vec![stale; 2 * v.len() + 3]).collect();
    call(view, &mut reused);
    assert_eq!(fresh, reused);
}

/// [`run_twice`] for a decomposition whose outputs are `T` factors plus one real spectrum.
fn run_twice_with_spectrum<T: FaerScalar + PartialEq + core::fmt::Debug>(
    dims: [usize; 3],
    data: &[T],
    stale: T,
    call: impl Fn(RawStridedRef<'_, T>, &mut [Vec<T>], &mut Vec<T::Real>),
    outputs: usize,
) where
    T::Real: core::fmt::Debug,
{
    let [m, n, batch] = dims;
    let (dims, strides) = ([m, n, batch], [1, m as isize, (m * n) as isize]);
    let view = RawStridedRef::new(data, &dims, &strides, 0).unwrap();
    let mut fresh = vec![Vec::new(); outputs];
    let mut fresh_spectrum = Vec::new();
    call(view, &mut fresh, &mut fresh_spectrum);
    let mut reused: Vec<Vec<T>> = fresh.iter().map(|v| vec![stale; 2 * v.len() + 3]).collect();
    let stale_real = stale.real_part();
    let mut reused_spectrum = vec![stale_real; 2 * fresh_spectrum.len() + 3];
    call(view, &mut reused, &mut reused_spectrum);
    assert_eq!(fresh, reused);
    assert_eq!(fresh_spectrum, reused_spectrum);
}

fn check<T: FaerScalar + PartialEq + core::fmt::Debug>(entry: fn(f64) -> T)
where
    T::Real: core::fmt::Debug,
{
    let stale = entry(f64::NAN);
    for (m, n) in [(5, 5), (7, 4), (4, 7)] {
        for batch in [1, 3] {
            let a = input(m, n, batch, entry);
            run_twice(
                [m, n, batch],
                &a,
                stale,
                |view, o| {
                    let [q, r] = o else { unreachable!() };
                    qr(Op::Qr, view, q, r, &mut Exec::serial()).unwrap();
                },
                2,
            );
            run_twice_with_spectrum(
                [m, n, batch],
                &a,
                stale,
                |view, o, s| {
                    let [u, vt] = o else { unreachable!() };
                    svd(Op::Svd, view, false, u, s, vt, &mut Exec::serial()).unwrap();
                },
                2,
            );
            run_twice_with_spectrum(
                [m, n, batch],
                &a,
                stale,
                |view, o, s| {
                    let [u, vt] = o else { unreachable!() };
                    svd(Op::Svd, view, true, u, s, vt, &mut Exec::serial()).unwrap();
                },
                2,
            );
        }
    }
    for n in [1, 5] {
        for batch in [1, 3] {
            let a = input(n, n, batch, entry);
            run_twice_with_spectrum(
                [n, n, batch],
                &a,
                stale,
                |view, o, w| {
                    let [v] = o else { unreachable!() };
                    eigh(Op::Eigh, view, w, v, &mut Exec::serial()).unwrap();
                },
                1,
            );
        }
    }
}

#[test]
fn f64_outputs_do_not_depend_on_prior_contents() {
    check::<f64>(|x| x);
}

#[test]
fn c64_outputs_do_not_depend_on_prior_contents() {
    check::<Complex64>(|x| Complex64::new(x, 0.5 * x));
}

#[test]
fn rank_revealing_outputs_do_not_depend_on_prior_contents() {
    let (m, n, batch) = (6, 4, 2);
    let a = input::<f64>(m, n, batch, |x| x);
    let (dims, strides) = ([m, n, batch], [1, m as isize, (m * n) as isize]);
    let view = RawStridedRef::new(&a, &dims, &strides, 0).unwrap();
    let (mut q, mut r, mut p) = (Vec::new(), Vec::new(), Vec::new());
    rank_revealing_qr(
        Op::RankRevealingQr,
        view,
        &mut q,
        &mut r,
        &mut p,
        &mut Exec::serial(),
    )
    .unwrap();
    let (mut q2, mut r2, mut p2) = (vec![f64::NAN; 99], vec![f64::NAN; 99], vec![-1i64; 99]);
    rank_revealing_qr(
        Op::RankRevealingQr,
        view,
        &mut q2,
        &mut r2,
        &mut p2,
        &mut Exec::serial(),
    )
    .unwrap();
    assert_eq!((q, r, p), (q2, r2, p2));
}
