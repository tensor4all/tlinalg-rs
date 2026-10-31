//! The `*_into_init` entries: initialised destinations are overwritten with exactly what the
//! `*_into` route writes, and bad calls fail with the same error, in the same order, before any
//! destination element is touched.

use core::mem::MaybeUninit;

use num_complex::{Complex32, Complex64};
use strided_view::RawStridedRef;
use tcontract_exec::Exec;
use tlinalg::eigh::{eigh, eigh_into, eigh_into_init, eigh_workspace};
use tlinalg::qr::{qr, qr_into, qr_into_init, qr_workspace};
use tlinalg::svd::{svd, svd_into, svd_into_init, svd_workspace};
use tlinalg::{Error, FaerScalar, Op};

/// Bit patterns, so `-0.0` and `0.0` (and NaN payloads) are told apart.
trait Bits: FaerScalar {
    fn bits(self) -> (u64, u64);
    fn make(re: f64, im: f64) -> Self;
}

impl Bits for f32 {
    fn bits(self) -> (u64, u64) {
        (self.to_bits().into(), 0)
    }
    fn make(re: f64, _: f64) -> Self {
        re as f32
    }
}

impl Bits for f64 {
    fn bits(self) -> (u64, u64) {
        (self.to_bits(), 0)
    }
    fn make(re: f64, _: f64) -> Self {
        re
    }
}

impl Bits for Complex32 {
    fn bits(self) -> (u64, u64) {
        (self.re.to_bits().into(), self.im.to_bits().into())
    }
    fn make(re: f64, im: f64) -> Self {
        Complex32::new(re as f32, im as f32)
    }
}

impl Bits for Complex64 {
    fn bits(self) -> (u64, u64) {
        (self.re.to_bits(), self.im.to_bits())
    }
    fn make(re: f64, im: f64) -> Self {
        Complex64::new(re, im)
    }
}

fn assert_bitwise<T: Bits>(got: &[T], want: &[T]) {
    let got: Vec<_> = got.iter().map(|&x| x.bits()).collect();
    let want: Vec<_> = want.iter().map(|&x| x.bits()).collect();
    assert_eq!(got, want);
}

fn value(i: usize, salt: usize) -> f64 {
    ((i * 7919 + salt) % 101) as f64 / 17.0 - 2.5
}

/// A compact `[m, n, batch]` batch.
fn general<T: Bits>(m: usize, n: usize, batch: usize) -> Vec<T> {
    (0..m * n * batch)
        .map(|i| T::make(value(i, 3), value(i, 41)))
        .collect()
}

/// A compact `[n, n, batch]` batch of Hermitian matrices.
fn hermitian<T: Bits>(n: usize, batch: usize) -> Vec<T> {
    let mut a = vec![T::make(0.0, 0.0); n * n * batch];
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
                a[b * n * n + j * n + i] = T::make(value(seed, 3), im);
            }
        }
    }
    a
}

/// Dims and compact column-major strides of an `[m, n, batch]` batch.
fn layout(m: usize, n: usize, batch: usize) -> ([usize; 3], [isize; 3]) {
    ([m, n, batch], [1, m as isize, (m * n) as isize])
}

fn workspace(bytes: usize) -> Vec<MaybeUninit<u8>> {
    vec![MaybeUninit::uninit(); bytes]
}

fn uninit<T>(len: usize) -> Vec<MaybeUninit<T>> {
    (0..len).map(|_| MaybeUninit::uninit()).collect()
}

const SHAPES: [(usize, usize); 6] = [(1, 1), (4, 4), (6, 3), (3, 6), (0, 3), (3, 0)];
const BATCHES: [usize; 3] = [0, 1, 3];

/// Purpose: every sentinel element is overwritten, bit for bit, with the `*_into` route's output.
fn overwrites_like_into<T: Bits>() {
    let sentinel = T::make(f64::NAN, f64::NAN);
    for (m, n) in SHAPES {
        for batch in BATCHES {
            let a = general::<T>(m, n, batch);
            let (dims, strides) = layout(m, n, batch);
            let input = RawStridedRef::new(&a, &dims, &strides, 0).unwrap();
            let (q, r) = (&mut Vec::new(), &mut Vec::new());
            qr(Op::Qr, input, q, r, &mut Exec::serial()).unwrap();
            let (mut q2, mut r2) = (vec![sentinel; q.len()], vec![sentinel; r.len()]);
            let mut ws = workspace(qr_workspace::<T>(m, n, batch, &Exec::serial()).unwrap());
            qr_into_init(
                Op::Qr,
                input,
                &mut q2,
                &mut r2,
                &mut ws,
                &mut Exec::serial(),
            )
            .unwrap();
            assert_bitwise(&q2, q);
            assert_bitwise(&r2, r);

            for full in [false, true] {
                let (u, s, vt) = (&mut Vec::new(), &mut Vec::new(), &mut Vec::new());
                svd(Op::Svd, input, full, u, s, vt, &mut Exec::serial()).unwrap();
                let mut u2 = vec![sentinel; u.len()];
                let mut s2 = vec![sentinel; s.len()];
                let mut vt2 = vec![sentinel; vt.len()];
                let bytes = svd_workspace::<T>(m, n, full, batch, &Exec::serial()).unwrap();
                svd_into_init(
                    Op::Svd,
                    input,
                    full,
                    &mut u2,
                    &mut s2,
                    &mut vt2,
                    &mut workspace(bytes),
                    &mut Exec::serial(),
                )
                .unwrap();
                assert_bitwise(&u2, u);
                assert_bitwise(&s2, s);
                assert_bitwise(&vt2, vt);
            }
        }
    }
    for n in [0, 1, 5] {
        for batch in BATCHES {
            let a = hermitian::<T>(n, batch);
            let (dims, strides) = layout(n, n, batch);
            let input = RawStridedRef::new(&a, &dims, &strides, 0).unwrap();
            let (w, v) = (&mut Vec::new(), &mut Vec::new());
            eigh(Op::Eigh, input, w, v, &mut Exec::serial()).unwrap();
            let (mut w2, mut v2) = (vec![sentinel; w.len()], vec![sentinel; v.len()]);
            let mut ws = workspace(eigh_workspace::<T>(n, true, batch, &Exec::serial()).unwrap());
            eigh_into_init(
                Op::Eigh,
                input,
                &mut w2,
                &mut v2,
                &mut ws,
                &mut Exec::serial(),
            )
            .unwrap();
            assert_bitwise(&w2, w);
            assert_bitwise(&v2, v);
        }
    }
}

#[test]
fn f32_destinations_are_fully_overwritten() {
    overwrites_like_into::<f32>();
}

#[test]
fn f64_destinations_are_fully_overwritten() {
    overwrites_like_into::<f64>();
}

#[test]
fn c32_destinations_are_fully_overwritten() {
    overwrites_like_into::<Complex32>();
}

#[test]
fn c64_destinations_are_fully_overwritten() {
    overwrites_like_into::<Complex64>();
}

/// One bad call through both entries: `(into, into_init)` results, and the init outputs after it.
struct Pair {
    into: Result<(), Error>,
    init: Result<(), Error>,
    untouched: bool,
}

/// A QR call with output lengths `lens`, `ws` workspace bytes and input `dims`.
fn qr_pair(dims: &[usize], lens: [usize; 2], ws: usize) -> Pair {
    let a = vec![1.0_f64; dims.iter().product()];
    let strides: Vec<isize> = (0..dims.len())
        .map(|k| dims[..k].iter().product::<usize>() as isize)
        .collect();
    let input = RawStridedRef::new(&a, dims, &strides, 0).unwrap();
    let (mut q, mut r) = (uninit::<f64>(lens[0]), uninit::<f64>(lens[1]));
    let into = qr_into(
        Op::Qr,
        input,
        &mut q,
        &mut r,
        &mut workspace(ws),
        &mut Exec::serial(),
    );
    let (mut q, mut r) = (vec![f64::NAN; lens[0]], vec![f64::NAN; lens[1]]);
    let init = qr_into_init(
        Op::Qr,
        input,
        &mut q,
        &mut r,
        &mut workspace(ws),
        &mut Exec::serial(),
    );
    let untouched = q.iter().chain(&r).all(|x| x.is_nan());
    Pair {
        into,
        init,
        untouched,
    }
}

/// An SVD call with output lengths `lens`, `ws` workspace bytes and the matrix `a`.
fn svd_pair(a: &[f64], dims: &[usize], lens: [usize; 3], ws: usize) -> Pair {
    let strides: Vec<isize> = (0..dims.len())
        .map(|k| dims[..k].iter().product::<usize>() as isize)
        .collect();
    let input = RawStridedRef::new(a, dims, &strides, 0).unwrap();
    let (mut u, mut s, mut vt) = (uninit::<f64>(lens[0]), uninit(lens[1]), uninit(lens[2]));
    let into = svd_into(
        Op::Svd,
        input,
        false,
        &mut u,
        &mut s,
        &mut vt,
        &mut workspace(ws),
        &mut Exec::serial(),
    );
    let (mut u, mut s, mut vt) = (
        vec![-7.0; lens[0]],
        vec![-7.0; lens[1]],
        vec![-7.0; lens[2]],
    );
    let init = svd_into_init(
        Op::Svd,
        input,
        false,
        &mut u,
        &mut s,
        &mut vt,
        &mut workspace(ws),
        &mut Exec::serial(),
    );
    let untouched = u.iter().chain(&s).chain(&vt).all(|&x| x == -7.0);
    Pair {
        into,
        init,
        untouched,
    }
}

/// An eigh call with output lengths `lens`, `ws` workspace bytes and the matrix `a`.
fn eigh_pair(a: &[f64], dims: &[usize], lens: [usize; 2], ws: usize) -> Pair {
    let strides: Vec<isize> = (0..dims.len())
        .map(|k| dims[..k].iter().product::<usize>() as isize)
        .collect();
    let input = RawStridedRef::new(a, dims, &strides, 0).unwrap();
    let (mut w, mut v) = (uninit::<f64>(lens[0]), uninit(lens[1]));
    let into = eigh_into(
        Op::Eigh,
        input,
        &mut w,
        &mut v,
        &mut workspace(ws),
        &mut Exec::serial(),
    );
    let (mut w, mut v) = (vec![-7.0; lens[0]], vec![-7.0; lens[1]]);
    let init = eigh_into_init(
        Op::Eigh,
        input,
        &mut w,
        &mut v,
        &mut workspace(ws),
        &mut Exec::serial(),
    );
    let untouched = w.iter().chain(&v).all(|&x| x == -7.0);
    Pair {
        into,
        init,
        untouched,
    }
}

fn assert_same_refusal(pair: Pair, expect: impl Fn(&Error) -> bool) {
    let err = pair.into.expect_err("the `*_into` call should fail");
    assert!(expect(&err), "unexpected error {err:?}");
    assert_eq!(pair.init, Err(err));
    assert!(pair.untouched, "a refused call wrote into its destination");
}

fn invalid(err: &Error) -> bool {
    matches!(err, Error::InvalidArgument { .. })
}

fn short_workspace(err: &Error) -> bool {
    matches!(err, Error::InsufficientWorkspace { .. })
}

/// Purpose: QR refusals (rank, output length, workspace, and workspace before output length) are
/// the `*_into` refusals and leave the destination untouched.
#[test]
fn qr_refusals_match_into_and_write_nothing() {
    let need = qr_workspace::<f64>(4, 3, 2, &Exec::serial()).unwrap();
    assert!(need > 0);
    assert_same_refusal(qr_pair(&[4], [4, 1], need), invalid);
    assert_same_refusal(qr_pair(&[4, 3, 2], [24, 17], need), invalid);
    assert_same_refusal(qr_pair(&[4, 3, 2], [23, 18], need), invalid);
    assert_same_refusal(qr_pair(&[4, 3, 2], [24, 18], need - 1), short_workspace);
    assert_same_refusal(qr_pair(&[4, 3, 2], [1, 1], need - 1), short_workspace);
}

/// Purpose: SVD refusals, including the short destination and the order of the workspace and
/// output-length checks, are the `*_into` refusals and leave the destination untouched.
#[test]
fn svd_refusals_match_into_and_write_nothing() {
    let a = vec![1.0_f64; 24];
    let need = svd_workspace::<f64>(4, 3, false, 2, &Exec::serial()).unwrap();
    assert!(need > 0);
    let ok = [24, 6, 18];
    assert_same_refusal(svd_pair(&a, &[24], ok, need), invalid);
    assert_same_refusal(svd_pair(&a, &[4, 3, 2], [23, 6, 18], need), invalid);
    assert_same_refusal(svd_pair(&a, &[4, 3, 2], [24, 5, 18], need), invalid);
    assert_same_refusal(svd_pair(&a, &[4, 3, 2], [24, 6, 19], need), invalid);
    assert_same_refusal(svd_pair(&a, &[4, 3, 2], ok, need - 1), short_workspace);
    assert_same_refusal(
        svd_pair(&a, &[4, 3, 2], [0, 0, 0], need - 1),
        short_workspace,
    );
}

/// Purpose: eigh refusals (rank, non-square input, output length, workspace, and their order) are
/// the `*_into` refusals and leave the destination untouched.
#[test]
fn eigh_refusals_match_into_and_write_nothing() {
    let a = vec![1.0_f64; 18];
    let need = eigh_workspace::<f64>(3, true, 2, &Exec::serial()).unwrap();
    assert!(need > 0);
    assert_same_refusal(eigh_pair(&a, &[18], [6, 18], need), invalid);
    assert_same_refusal(eigh_pair(&a, &[3, 2, 3], [6, 18], need), invalid);
    assert_same_refusal(eigh_pair(&a, &[3, 3, 2], [5, 18], need), invalid);
    assert_same_refusal(eigh_pair(&a, &[3, 3, 2], [6, 17], need), invalid);
    assert_same_refusal(
        eigh_pair(&a, &[3, 3, 2], [6, 18], need - 1),
        short_workspace,
    );
    assert_same_refusal(eigh_pair(&a, &[3, 3, 2], [0, 0], need - 1), short_workspace);
}

/// Purpose: a non-finite item fails both entries with the same `NonConvergence` (the destination's
/// contents are unspecified after it, so they are not checked).
#[test]
fn non_convergence_matches_into() {
    let mut a = vec![1.0_f64; 18];
    a[9..].fill(f64::NAN);
    let need = svd_workspace::<f64>(3, 3, false, 2, &Exec::serial()).unwrap();
    let pair = svd_pair(&a, &[3, 3, 2], [18, 6, 18], need);
    assert_eq!(pair.into, Err(Error::NonConvergence { op: Op::Svd }));
    assert_eq!(pair.init, pair.into);
    let need = eigh_workspace::<f64>(3, true, 2, &Exec::serial()).unwrap();
    let pair = eigh_pair(&a, &[3, 3, 2], [6, 18], need);
    assert_eq!(pair.into, Err(Error::NonConvergence { op: Op::Eigh }));
    assert_eq!(pair.init, pair.into);
}
