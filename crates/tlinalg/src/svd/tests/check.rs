//! The check of a decomposition against its input, and the repeat with the QR algorithm.

use std::cell::Cell;

use core::mem::MaybeUninit;

use faer::dyn_stack::MemStack;
use faer::linalg::svd::ComputeSvdVectors;
use faer::{c64, Mat, MatMut};
use num_complex::Complex64;
use strided_view::RawStridedRef;
use tlinalg_testkit::{clustered_spectrum, with_singular_values, TestScalar};

use crate::svd::{divides, reproduces, residual_bound, svd, svd_req, Check, SvdScratch};
use crate::{FaerScalar, Op};
use tcontract_exec::Exec;

thread_local! {
    // The kernel records a repeat on the thread that runs it; with `Exec::serial()` that is the
    // test's own thread, so tests on other threads do not interfere.
    static REPEATS: Cell<usize> = const { Cell::new(0) };
}

pub(in crate::svd) fn record_repeat() {
    REPEATS.set(REPEATS.get() + 1);
}

/// Decompose one `m x n` matrix on this thread and return how often the kernel repeated the
/// decomposition, with the singular values.
fn repeats<T: FaerScalar + TestScalar>(a: &[T], m: usize, n: usize, full: bool) -> (usize, Vec<T>) {
    REPEATS.set(0);
    let (mut u, mut s, mut vt) = (Vec::new(), Vec::new(), Vec::new());
    svd(
        Op::Svd,
        RawStridedRef::new(a, &[m, n], &[1, m as isize], 0).unwrap(),
        full,
        &mut u,
        &mut s,
        &mut vt,
        Exec::serial(),
    )
    .unwrap();
    (REPEATS.get(), s)
}

#[test]
fn clustered_decompositions_do_not_need_a_qr_retry() {
    // t4a-faer 0.24.8 fixes the secular-root stagnation that previously required a retry here.
    let n = 160usize;
    let spectrum = clustered_spectrum(n);
    for full in [false, true] {
        let (count, s) = repeats::<Complex64>(&with_singular_values(&spectrum, 1), n, n, full);
        assert_eq!(count, 0, "Complex64, full={full}");
        for (index, want) in spectrum.iter().enumerate() {
            assert!(
                (s[index].re - want).abs() < 1e-12,
                "Complex64 value {index}"
            );
        }
        let (count, s) = repeats::<f64>(&with_singular_values(&spectrum, 1), n, n, full);
        assert_eq!(count, 0, "f64, full={full}");
        for (index, want) in spectrum.iter().enumerate() {
            assert!((s[index] - want).abs() < 1e-12, "f64 value {index}");
        }
    }
}

#[test]
fn accurate_decompositions_are_not_repeated() {
    // Sizes at and past faer's switch to divide and conquer, square, tall and wide.
    for (m, n) in [(128usize, 128usize), (160, 160), (300, 140), (140, 300)] {
        for full in [false, true] {
            let (count, _) = repeats(&tlinalg_testkit::matrix::<f64>(m, n, 3), m, n, full);
            assert_eq!(count, 0, "f64 {m}x{n} full={full}");
            let (count, _) = repeats(&tlinalg_testkit::matrix::<f32>(m, n, 3), m, n, full);
            assert_eq!(count, 0, "f32 {m}x{n} full={full}");
            let (count, _) = repeats(&tlinalg_testkit::matrix::<Complex64>(m, n, 3), m, n, full);
            assert_eq!(count, 0, "Complex64 {m}x{n} full={full}");
            let (count, _) = repeats(
                &tlinalg_testkit::matrix::<num_complex::Complex32>(m, n, 3),
                m,
                n,
                full,
            );
            assert_eq!(count, 0, "Complex32 {m}x{n} full={full}");
        }
    }
    // A spectrum with the same clusters at sizes where faer's default path is accurate.
    for n in [128usize, 256] {
        let a: Vec<Complex64> = with_singular_values(&clustered_spectrum(n), 1);
        assert_eq!(repeats(&a, n, n, false).0, 0, "clustered, n={n}");
    }
}

#[test]
fn only_decompositions_that_can_divide_are_checked() {
    let threshold = 128usize;
    assert!(!divides::<c64>(threshold - 1));
    assert!(divides::<c64>(threshold));
    let checked = |m: usize, n: usize, vectors| {
        let mut bytes = vec![
            MaybeUninit::<u8>::uninit();
            svd_req::<c64>(m, n, vectors, faer::Par::Seq)
                .unaligned_bytes_required()
        ];
        SvdScratch::<c64>::new(m, n, vectors, faer::Par::Seq, MemStack::new(&mut bytes))
            .check
            .is_some()
    };
    assert!(!checked(127, 400, ComputeSvdVectors::Thin));
    assert!(checked(128, 128, ComputeSvdVectors::Thin));
    assert!(checked(400, 128, ComputeSvdVectors::Full));
    // Without vectors there is nothing to check.
    assert!(!checked(400, 400, ComputeSvdVectors::No));
}

/// Whether the check accepts `u diag(values) vᴴ` as a decomposition of `a`.
fn accepts<E: faer::traits::ComplexField>(
    a: &Mat<E>,
    u: &Mat<E>,
    values: &[f64],
    v: &Mat<E>,
    epsilon: f64,
) -> bool {
    let s = faer::Col::<E>::from_fn(values.len(), |index| {
        faer::traits::math_utils::from_f64::<E>(values[index])
    });
    let (m, n) = (a.nrows(), a.ncols());
    let k = m.min(n);
    let mut scaled = vec![E::zero_impl(); m * k];
    let mut residual = vec![E::zero_impl(); m * n];
    let mut check = Check {
        scaled: MatMut::from_column_major_slice_mut(&mut scaled, m, k),
        residual: MatMut::from_column_major_slice_mut(&mut residual, m, n),
    };
    reproduces(
        a.as_ref(),
        u.as_ref(),
        s.as_diagonal(),
        v.as_ref(),
        &mut check,
        epsilon,
        faer::Par::Seq,
    )
}

#[test]
fn the_check_accepts_exact_factors_and_rejects_perturbed_ones() {
    // A 3 x 2 matrix with singular values 3 and 2: U = [e1 e2], V = identity.
    let (m, n) = (3usize, 2usize);
    let a = Mat::<f64>::from_fn(m, n, |row, col| match (row, col) {
        (0, 0) => 3.0,
        (1, 1) => 2.0,
        _ => 0.0,
    });
    let u = Mat::<f64>::from_fn(m, n, |row, col| if row == col { 1.0 } else { 0.0 });
    let v = Mat::<f64>::identity(n, n);
    let accepts = |values: [f64; 2]| accepts(&a, &u, &values, &v, f64::EPSILON);
    assert!(accepts([3.0, 2.0]));
    // The bound is 24 * sqrt(3) * epsilon = 9.2e-15 relative to the norm sqrt(13) of the matrix.
    assert!(accepts([3.0, 2.0 + 1e-15]));
    assert!(!accepts([3.0, 2.0 + 1e-12]));
    assert!(!accepts([3.0, f64::NAN]));
    assert!(!accepts([f64::INFINITY, 2.0]));
}

/// A singular value that is missing from, or wrong by one percent in, the decomposition of an
/// identity matrix is rejected in every scalar type. In single precision a bound that grows in
/// proportion to the size would accept the missing value from a size of about 2000.
#[test]
fn a_missing_or_wrong_singular_value_is_rejected_in_every_precision() {
    fn check<E: faer::traits::ComplexField>(epsilon: f64, what: &str) {
        let n = 256usize;
        let identity = Mat::<E>::identity(n, n);
        let mut values = vec![1.0; n];
        assert!(
            accepts(&identity, &identity, &values, &identity, epsilon),
            "{what}: exact"
        );
        values[n - 1] = 0.0;
        assert!(
            !accepts(&identity, &identity, &values, &identity, epsilon),
            "{what}: missing"
        );
        values[n - 1] = 1.01;
        assert!(
            !accepts(&identity, &identity, &values, &identity, epsilon),
            "{what}: wrong"
        );
    }
    check::<f64>(f64::EPSILON, "f64");
    check::<f32>(f32::EPSILON as f64, "f32");
    check::<c64>(f64::EPSILON, "Complex64");
    check::<faer::c32>(f32::EPSILON as f64, "Complex32");

    // The same holds for the bound itself at sizes too large to decompose in a test: the relative
    // residual of a missing unit singular value of an n x n identity is 1 / sqrt(n).
    for n in [2048usize, 65536] {
        let missing = 1.0 / (n as f64).sqrt();
        assert!(
            residual_bound(n, n, f32::EPSILON as f64) < missing,
            "f32, n={n}"
        );
        assert!(residual_bound(n, n, f64::EPSILON) < missing, "f64, n={n}");
    }
}

/// The residual is taken over the whole matrix, so no nonzero matrix passes with zero factors,
/// whichever of its rows carries the entries.
#[test]
fn zero_factors_are_rejected_for_every_rank_one_matrix() {
    let n = 128usize;
    let identity = Mat::<f64>::identity(n, n);
    let zeros = vec![0.0; n];
    for row in [0usize, 77, n - 1] {
        // One nonzero row with entries of both signs.
        let a = Mat::<f64>::from_fn(n, n, |i, j| {
            if i == row {
                (j as f64 * 0.37).sin()
            } else {
                0.0
            }
        });
        assert!(
            !accepts(&a, &identity, &zeros, &identity, f64::EPSILON),
            "row {row}"
        );
    }
}
