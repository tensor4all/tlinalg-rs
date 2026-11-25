//! Behavioural tests for the rank-truncated full-pivot LU (`full_piv_lu_truncated[_into]`,
//! tenferro-rs#2488 step 2).
//!
//! The regression net for the untruncated contract is `lu.rs`; here the default options are checked
//! bitwise against it, and the truncated outputs against their own contract: fixed capacity
//! `c = min(m, n, max_rank)`, per-item rank, zeros past the rank, full permutations.

mod common;

use core::mem::MaybeUninit;

use common::*;
use num_complex::Complex64;
use strided_view::RawStridedRef;
use tcontract_exec::Exec;
use tlinalg::full_piv_lu::{
    full_piv_lu, full_piv_lu_truncated, full_piv_lu_truncated_into, full_piv_lu_workspace,
    FullPivLuFactors, FullPivLuTruncatedFactors, FullPivLuTruncation,
};
use tlinalg::FaerScalar;
use tlinalg::{Error, Op};

/// The six outputs of one truncated call.
struct Truncated<T> {
    p: Vec<T>,
    l: Vec<T>,
    u: Vec<T>,
    q: Vec<T>,
    parity: Vec<T>,
    rank: Vec<usize>,
}

fn truncated<T: TestScalar + FaerScalar>(
    input: RawStridedRef<'_, T>,
    truncation: FullPivLuTruncation,
    exec: &mut Exec<'_>,
) -> Result<Truncated<T>, Error> {
    let mut out = Truncated {
        p: Vec::new(),
        l: Vec::new(),
        u: Vec::new(),
        q: Vec::new(),
        parity: Vec::new(),
        rank: Vec::new(),
    };
    full_piv_lu_truncated(
        Op::FullPivLu,
        input,
        truncation,
        FullPivLuTruncatedFactors {
            p: &mut out.p,
            l: &mut out.l,
            u: &mut out.u,
            q: &mut out.q,
            parity: &mut out.parity,
            rank: &mut out.rank,
        },
        exec,
    )?;
    Ok(out)
}

/// Whether the column-major `n x n` matrix is a permutation matrix: entries in {0, 1}, exactly one
/// `1` per row and per column.
fn is_permutation(p: &[Complex64], n: usize) -> bool {
    let zero = Complex64::new(0.0, 0.0);
    let one = Complex64::new(1.0, 0.0);
    p.iter().all(|&v| v == zero || v == one)
        && (0..n).all(|i| (0..n).filter(|&j| p[i + j * n] == one).count() == 1)
        && (0..n).all(|j| (0..n).filter(|&i| p[i + j * n] == one).count() == 1)
}

/// Determinant sign of a permutation matrix, by counting inversions of its row map.
fn permutation_sign(p: &[Complex64], n: usize) -> f64 {
    let map: Vec<usize> = (0..n)
        .map(|row| (0..n).find(|&col| p[row + col * n].re == 1.0).unwrap())
        .collect();
    let inversions = (0..n)
        .flat_map(|i| (i + 1..n).map(move |j| (i, j)))
        .filter(|&(i, j)| map[i] > map[j])
        .count();
    if inversions % 2 == 0 {
        1.0
    } else {
        -1.0
    }
}

/// The column-major `m x n` product of an `m x r` and an `r x n` matrix from [`matrix`]: rank `r`
/// (generically), narrowed to `T` after the product is formed in `f64`.
fn rank_r<T: TestScalar>(m: usize, n: usize, r: usize, seed: usize) -> Vec<T> {
    let left = widen(&matrix::<T>(m, r, seed));
    let right = widen(&matrix::<T>(r, n, seed + 3));
    narrow(&matmul(&left, &right, m, r, n))
}

/// Check one item's truncated factors against the contract and return `P A Qᵀ - L U`.
#[allow(clippy::too_many_arguments)]
fn check_item(
    a: &[Complex64],
    (p, l, u, q, parity): (&[Complex64], &[Complex64], &[Complex64], &[Complex64], f64),
    m: usize,
    n: usize,
    cap: usize,
    rank: usize,
    what: &str,
) -> Vec<Complex64> {
    let zero = Complex64::new(0.0, 0.0);
    assert!(rank <= cap, "{what}: rank {rank} above capacity {cap}");
    assert!(is_permutation(p, m), "{what}: P is a full permutation");
    assert!(is_permutation(q, n), "{what}: Q is a full permutation");
    assert_eq!(
        parity,
        permutation_sign(p, m) * permutation_sign(q, n),
        "{what}: parity"
    );
    for col in 0..cap {
        for row in 0..m {
            let v = l[row + col * m];
            if col >= rank || row < col {
                assert_eq!(v, zero, "{what}: L[{row}, {col}] is zero");
            } else if row == col {
                assert_eq!(v, Complex64::new(1.0, 0.0), "{what}: L unit diagonal");
            }
        }
    }
    for col in 0..n {
        for row in 0..cap {
            if row >= rank || row > col {
                assert_eq!(u[row + col * cap], zero, "{what}: U[{row}, {col}] is zero");
            }
        }
    }
    let pa_qt = matmul(&matmul(p, a, m, m, n), &transpose(q, n, n), m, n, n);
    let lu = matmul(l, u, m, cap, n);
    pa_qt.iter().zip(&lu).map(|(x, y)| x - y).collect()
}

fn default_options_match_the_untruncated_factorization<T: TestScalar + FaerScalar>() {
    // Square (the regression net of step 1), tall, wide, a single element, padded strides and a
    // batch: with no option set the factors are bitwise `full_piv_lu`'s and every rank is `k`.
    for (m, n, batch) in [
        (4, 4, 1),
        (5, 3, 1),
        (3, 5, 1),
        (1, 1, 1),
        (6, 6, 3),
        (7, 4, 2),
    ] {
        let k = m.min(n);
        let a: Vec<T> = (0..batch)
            .flat_map(|item| pivoting::<T>(m, n, 19 + item))
            .collect();
        let dims = [m, n, batch];
        let strides = [1, m as isize, (m * n) as isize];
        let view = RawStridedRef::new(&a, &dims, &strides, 0).unwrap();
        let (mut p, mut l, mut u, mut q, mut parity) =
            (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new());
        full_piv_lu(
            Op::FullPivLu,
            view,
            FullPivLuFactors {
                p: &mut p,
                l: &mut l,
                u: &mut u,
                q: &mut q,
                parity: &mut parity,
            },
            &mut Exec::serial(),
        )
        .unwrap();
        let t = truncated(view, FullPivLuTruncation::default(), &mut Exec::serial()).unwrap();
        let what = format!("{m}x{n} batch {batch}");
        assert_eq!(widen(&t.p), widen(&p), "{what}: P");
        assert_eq!(widen(&t.l), widen(&l), "{what}: L");
        assert_eq!(widen(&t.u), widen(&u), "{what}: U");
        assert_eq!(widen(&t.q), widen(&q), "{what}: Q");
        assert_eq!(widen(&t.parity), widen(&parity), "{what}: parity");
        assert_eq!(t.rank, vec![k; batch], "{what}: rank");
    }
    // A padded single matrix, through the same path.
    let (m, n) = (5, 4);
    let a = pivoting::<T>(m, n, 7);
    let (storage, strides, offset) = padded(&a, m, n);
    let dims = [m, n];
    let view = RawStridedRef::new(&storage, &dims, &strides, offset).unwrap();
    let t = truncated(view, FullPivLuTruncation::default(), &mut Exec::serial()).unwrap();
    let (mut p, mut l, mut u, mut q, mut parity) =
        (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new());
    full_piv_lu(
        Op::FullPivLu,
        view,
        FullPivLuFactors {
            p: &mut p,
            l: &mut l,
            u: &mut u,
            q: &mut q,
            parity: &mut parity,
        },
        &mut Exec::serial(),
    )
    .unwrap();
    assert_eq!(
        (widen(&t.p), widen(&t.l), widen(&t.u), widen(&t.q)),
        (widen(&p), widen(&l), widen(&u), widen(&q)),
        "padded"
    );
    assert_eq!(t.rank, [4]);
}
for_each_scalar!(
    default_options,
    default_options_match_the_untruncated_factorization
);

fn tolerance_reports_a_rank_per_batch_item<T: TestScalar + FaerScalar>() {
    // Three `6 x 5` items of ranks 1, 3 and 5 in one batch: the shapes are fixed by the capacity
    // `min(6, 5) = 5`, each item reports its own rank and is zero past it, and each reconstructs.
    let (m, n) = (6, 5);
    let ranks = [1, 3, 5];
    let items: Vec<Vec<T>> = ranks
        .iter()
        .enumerate()
        .map(|(i, &r)| rank_r::<T>(m, n, r, 2 + i))
        .collect();
    let a: Vec<T> = items.iter().flatten().copied().collect();
    let batch = ranks.len();
    let (dims, strides) = ([m, n, batch], [1, m as isize, (m * n) as isize]);
    let view = RawStridedRef::new(&a, &dims, &strides, 0).unwrap();
    let truncation = FullPivLuTruncation {
        rel_tol: Some(T::TOL),
        ..Default::default()
    };
    let t = truncated(view, truncation, &mut Exec::serial()).unwrap();
    let cap = 5;
    assert_eq!(t.rank, ranks, "per-item ranks");
    assert_eq!(
        (t.p.len(), t.l.len(), t.u.len(), t.q.len(), t.parity.len()),
        (
            m * m * batch,
            m * cap * batch,
            cap * n * batch,
            n * n * batch,
            batch
        )
    );
    for (i, item) in items.iter().enumerate() {
        let a = widen(item);
        let at = |v: &[T], len: usize| widen(&v[i * len..(i + 1) * len]);
        let residual = check_item(
            &a,
            (
                &at(&t.p, m * m),
                &at(&t.l, m * cap),
                &at(&t.u, cap * n),
                &at(&t.q, n * n),
                t.parity[i].to_c64().re,
            ),
            m,
            n,
            cap,
            t.rank[i],
            &format!("item {i}"),
        );
        let scale = a.iter().map(|v| v.norm()).fold(1.0, f64::max);
        let worst = residual.iter().map(|v| v.norm()).fold(0.0, f64::max);
        assert!(
            worst <= 10.0 * T::TOL * scale,
            "item {i}: rank-{} reconstruction residual {worst} (scale {scale})",
            t.rank[i]
        );
    }
}
for_each_scalar!(batch_ranks, tolerance_reports_a_rank_per_batch_item);

fn max_rank_sets_the_capacity_and_leaves_the_schur_complement<T: TestScalar + FaerScalar>() {
    // A full-rank `7 x 5` matrix stopped after two pivots: `L` is `7 x 2`, `U` is `2 x 5`, the
    // permutations stay `7 x 7` and `5 x 5`, `P A Qᵀ - L U` vanishes outside its trailing
    // `5 x 3` block (the Schur complement), and that block is bounded by the rejected third pivot,
    // which is the third diagonal entry of the untruncated `U`.
    let (m, n, r) = (7, 5, 2);
    let a = pivoting::<T>(m, n, 23);
    let (dims, strides) = ([m, n], [1, m as isize]);
    let view = RawStridedRef::new(&a, &dims, &strides, 0).unwrap();
    let truncation = FullPivLuTruncation {
        max_rank: Some(r),
        ..Default::default()
    };
    let t = truncated(view, truncation, &mut Exec::serial()).unwrap();
    assert_eq!(t.rank, [r]);
    assert_eq!(
        (t.p.len(), t.l.len(), t.u.len(), t.q.len()),
        (m * m, m * r, r * n, n * n)
    );
    let wa = widen(&a);
    let residual = check_item(
        &wa,
        (
            &widen(&t.p),
            &widen(&t.l),
            &widen(&t.u),
            &widen(&t.q),
            t.parity[0].to_c64().re,
        ),
        m,
        n,
        r,
        r,
        "max_rank 2",
    );
    let scale = wa.iter().map(|v| v.norm()).fold(1.0, f64::max);
    for col in 0..n {
        for row in 0..m {
            if row < r || col < r {
                assert!(
                    residual[row + col * m].norm() <= T::TOL * scale,
                    "residual outside the trailing block at ({row}, {col})"
                );
            }
        }
    }
    let (mut p, mut l, mut u, mut q, mut parity) =
        (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new());
    full_piv_lu(
        Op::FullPivLu,
        view,
        FullPivLuFactors {
            p: &mut p,
            l: &mut l,
            u: &mut u,
            q: &mut q,
            parity: &mut parity,
        },
        &mut Exec::serial(),
    )
    .unwrap();
    let k = m.min(n);
    let rejected = widen(&u)[r + r * k].norm();
    let worst = residual.iter().map(|v| v.norm()).fold(0.0, f64::max);
    assert!(
        worst <= 2f64.sqrt() * rejected * (1.0 + T::TOL) + T::TOL * scale,
        "Schur complement {worst} above the rejected pivot {rejected}"
    );
    // The accepted pivots are the untruncated ones.
    let full_u = widen(&u);
    let tu = widen(&t.u);
    for i in 0..r {
        assert_eq!(tu[i + i * r], full_u[i + i * k], "pivot {i}");
    }
}
for_each_scalar!(
    max_rank_capacity,
    max_rank_sets_the_capacity_and_leaves_the_schur_complement
);

fn max_rank_zero_keeps_full_permutations<T: TestScalar + FaerScalar>() {
    let (m, n) = (4, 3);
    let a = pivoting::<T>(m, n, 5);
    let (dims, strides) = ([m, n], [1, m as isize]);
    let view = RawStridedRef::new(&a, &dims, &strides, 0).unwrap();
    let truncation = FullPivLuTruncation {
        max_rank: Some(0),
        ..Default::default()
    };
    let t = truncated(view, truncation, &mut Exec::serial()).unwrap();
    assert_eq!(t.rank, [0]);
    assert!(t.l.is_empty() && t.u.is_empty(), "L is m x 0, U is 0 x n");
    assert_eq!(widen(&t.p), identity(m), "P is the identity");
    assert_eq!(widen(&t.q), identity(n), "Q is the identity");
    assert_eq!(t.parity[0].to_c64().re, 1.0);
}
for_each_scalar!(max_rank_zero, max_rank_zero_keeps_full_permutations);

fn a_zero_matrix_has_rank_zero_once_an_option_is_set<T: TestScalar + FaerScalar>() {
    // The exactly zero remaining block has always ended the factorization; with an option set it
    // is reported as the stop (rank 0), with none the rank stays `min(m, n)` as in `full_piv_lu`.
    let (m, n) = (3, 4);
    let a = vec![T::default(); m * n];
    let (dims, strides) = ([m, n], [1, m as isize]);
    let view = RawStridedRef::new(&a, &dims, &strides, 0).unwrap();
    let revealed = FullPivLuTruncation {
        abs_tol: Some(0.0),
        ..Default::default()
    };
    let t = truncated(view, revealed, &mut Exec::serial()).unwrap();
    assert_eq!(t.rank, [0]);
    assert!(widen(&t.l).iter().all(|v| v.norm() == 0.0), "L is zero");
    assert!(is_permutation(&widen(&t.p), m) && is_permutation(&widen(&t.q), n));
    let t = truncated(view, FullPivLuTruncation::default(), &mut Exec::serial()).unwrap();
    assert_eq!(t.rank, [3]);
}
for_each_scalar!(
    zero_matrix,
    a_zero_matrix_has_rank_zero_once_an_option_is_set
);

#[test]
fn a_non_finite_item_ignores_the_tolerances() {
    // faer's rule: a NaN or infinity disables the tolerances, so the rank is the capacity and a
    // non-finite entry never shows up as a small truncated rank. The finite item of the same batch
    // still truncates.
    let (m, n) = (3, 3);
    let mut a = rank_r::<f64>(m, n, 1, 4);
    a.extend(rank_r::<f64>(m, n, 1, 4));
    a[m * n + 4] = f64::NAN;
    let (dims, strides) = ([m, n, 2], [1, 3, 9]);
    let view = RawStridedRef::new(&a, &dims, &strides, 0).unwrap();
    let truncation = FullPivLuTruncation {
        max_rank: Some(2),
        abs_tol: Some(1e-8),
        rel_tol: Some(1e-8),
    };
    let t = truncated(view, truncation, &mut Exec::serial()).unwrap();
    assert_eq!(t.rank, [1, 2]);
}

#[test]
fn invalid_tolerances_are_rejected_and_leave_the_outputs_empty() {
    let a = [1.0_f64, 0.0, 0.0, 2.0];
    let (dims, strides) = ([2, 2], [1, 2]);
    let view = RawStridedRef::new(&a, &dims, &strides, 0).unwrap();
    for (abs_tol, rel_tol) in [
        (Some(-1.0), None),
        (None, Some(f64::NAN)),
        (Some(f64::INFINITY), None),
        (None, Some(-0.5)),
    ] {
        let truncation = FullPivLuTruncation {
            max_rank: None,
            abs_tol,
            rel_tol,
        };
        let mut rank = vec![7];
        let (mut p, mut l, mut u, mut q, mut parity) =
            (vec![1.0], vec![1.0], vec![1.0], vec![1.0], vec![1.0]);
        let err = full_piv_lu_truncated(
            Op::FullPivLu,
            view,
            truncation,
            FullPivLuTruncatedFactors {
                p: &mut p,
                l: &mut l,
                u: &mut u,
                q: &mut q,
                parity: &mut parity,
                rank: &mut rank,
            },
            &mut Exec::serial(),
        )
        .unwrap_err();
        assert!(
            matches!(
                err,
                Error::InvalidArgument {
                    op: Op::FullPivLu,
                    role: "truncation",
                    ..
                }
            ),
            "{abs_tol:?} {rel_tol:?}: {err:?}"
        );
        assert!(p.is_empty() && l.is_empty() && u.is_empty() && q.is_empty());
        assert!(parity.is_empty() && rank.is_empty());
    }
}

#[test]
fn into_checks_the_capacity_sized_buffers() {
    // `L` and `U` are sized by the capacity, not by `min(m, n)`: a buffer of the untruncated size
    // is a mismatch, and so is a rank buffer of the wrong batch length.
    let (m, n) = (4, 4);
    let a = pivoting::<f64>(m, n, 3);
    let (dims, strides) = ([m, n], [1, m as isize]);
    let view = RawStridedRef::new(&a, &dims, &strides, 0).unwrap();
    let truncation = FullPivLuTruncation {
        max_rank: Some(2),
        ..Default::default()
    };
    let seq = &mut Exec::serial();
    let ws = full_piv_lu_workspace::<f64>(m, n, 1, seq).unwrap();
    let mut workspace = vec![MaybeUninit::<u8>::uninit(); ws];
    let run = |l_len: usize, rank_len: usize, workspace: &mut [MaybeUninit<u8>]| {
        let mut p = vec![MaybeUninit::<f64>::uninit(); m * m];
        let mut l = vec![MaybeUninit::<f64>::uninit(); l_len];
        let mut u = vec![MaybeUninit::<f64>::uninit(); 2 * n];
        let mut q = vec![MaybeUninit::<f64>::uninit(); n * n];
        let mut parity = vec![MaybeUninit::<f64>::uninit(); 1];
        let mut rank = vec![MaybeUninit::<usize>::uninit(); rank_len];
        let result = full_piv_lu_truncated_into(
            Op::FullPivLu,
            view,
            truncation,
            &mut p,
            &mut l,
            &mut u,
            &mut q,
            &mut parity,
            &mut rank,
            workspace,
            &mut Exec::serial(),
        );
        // SAFETY: read only on success, when the call wrote the one rank element.
        result.map(|()| unsafe { rank[0].assume_init() })
    };
    assert_eq!(run(m * 2, 1, &mut workspace), Ok(2));
    for (l_len, rank_len) in [(m * m, 1), (m * 2, 2)] {
        let err = run(l_len, rank_len, &mut workspace).unwrap_err();
        assert!(
            matches!(err, Error::InvalidArgument { role: "output", .. }),
            "{err:?}"
        );
    }
}

#[test]
fn a_pooled_batch_matches_the_serial_one() {
    let (m, n, batch) = (6, 5, 24);
    let a: Vec<f64> = (0..batch)
        .flat_map(|i| rank_r::<f64>(m, n, 1 + i % 5, i))
        .collect();
    let (dims, strides) = ([m, n, batch], [1, m as isize, (m * n) as isize]);
    let view = RawStridedRef::new(&a, &dims, &strides, 0).unwrap();
    let truncation = FullPivLuTruncation {
        max_rank: Some(4),
        rel_tol: Some(1e-10),
        ..Default::default()
    };
    let serial = truncated(view, truncation, &mut Exec::serial()).unwrap();
    let pool = lanes_pool();
    let pooled = truncated(view, truncation, &mut pool_token(&pool)).unwrap();
    assert_eq!(
        serial.rank,
        (0..batch).map(|i| (1 + i % 5).min(4)).collect::<Vec<_>>()
    );
    assert_eq!(pooled.rank, serial.rank);
    assert_eq!(
        (pooled.p, pooled.l, pooled.u, pooled.q, pooled.parity),
        (serial.p, serial.l, serial.u, serial.q, serial.parity)
    );
}
