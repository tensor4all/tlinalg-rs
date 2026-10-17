//! Kernel-level comparison of the faer provider (`tlinalg`) and the LAPACK provider
//! (`tlinalg-blas`).
//!
//! Every family is a Criterion group `"{family}/{dtype}"`; within it each case (`n4xb1024` is
//! 1024 matrices of 4x4, `t64x24` a 64x24 matrix) has up to three rows:
//!
//! * `faer-1lane` — one sequential lane: the per-item cost and the batch loop, no threading;
//! * `faer-{N}t` — a host-owned `N`-worker pool (`TLINALG_BENCH_THREADS`, else the available
//!   parallelism), with the provider choosing its batch lanes;
//! * `lapack` — the LAPACK provider (only with `--features link-openblas`), whose batch loop is
//!   serial and whose threading belongs to the vendor library.
//!
//! Run everything with
//! `cargo bench -p tlinalg-bench --features link-openblas`, or one family with a filter, e.g.
//! `cargo bench -p tlinalg-bench --features link-openblas -- 'eigh/f64'`.

use std::time::Duration;

use criterion::measurement::WallTime;
use criterion::{criterion_group, criterion_main, BenchmarkGroup, BenchmarkId, Criterion};
use strided_view::RawStridedRef;
use tcontract_exec::Exec;
use tlinalg::Op;
#[cfg(feature = "link-openblas")]
use tlinalg_bench::RecyclingWorkspace;
use tlinalg_bench::{Batch, BenchScalar, Env};

/// Small batched cases: `n x n` matrices, `batch` of them.
const SMALL_N: [usize; 3] = [2, 4, 8];
const SMALL_BATCH: [usize; 5] = [1, 3, 4, 8, 1024];
/// Larger single matrices.
const SQUARE_N: [usize; 3] = [32, 128, 512];
/// Right-hand sides for the solve families.
const NRHS: usize = 4;

/// A LAPACK row, compiled only when the vendor library is linked.
type LapackRow<'a> = Option<Box<dyn FnMut() + 'a>>;

macro_rules! lapack_row {
    ($body:expr) => {{
        #[cfg(feature = "link-openblas")]
        let row: LapackRow<'_> = Some(Box::new($body));
        #[cfg(not(feature = "link-openblas"))]
        let row: LapackRow<'_> = None;
        row
    }};
}

fn group<'c>(c: &'c mut Criterion, family: &str, dtype: &str) -> BenchmarkGroup<'c, WallTime> {
    let mut group = c.benchmark_group(format!("{family}/{dtype}"));
    group
        .sample_size(10)
        .warm_up_time(Duration::from_millis(200))
        .measurement_time(Duration::from_millis(800));
    group
}

/// Bench the faer rows (one lane and a host pool) and the LAPACK row of one case.
fn rows(
    group: &mut BenchmarkGroup<'_, WallTime>,
    env: &Env,
    case: &str,
    mut faer: impl FnMut(Exec<'_>),
    lapack: LapackRow<'_>,
) {
    group.bench_function(BenchmarkId::new("faer-1lane", case), |b| {
        b.iter(|| faer(Exec::serial()))
    });
    if env.threads() > 1 {
        group.bench_function(
            BenchmarkId::new(format!("faer-{}t", env.threads()), case),
            |b| b.iter(|| faer(env.par())),
        );
    }
    if let Some(mut lapack) = lapack {
        group.bench_function(BenchmarkId::new("lapack", case), |b| b.iter(&mut lapack));
    }
}

/// `(case label, n, batch)` for the small grid and the larger single matrices.
fn square_cases(small: bool, large: bool) -> Vec<(String, usize, usize)> {
    let mut cases = Vec::new();
    if small {
        for n in SMALL_N {
            for batch in SMALL_BATCH {
                cases.push((format!("n{n}xb{batch}"), n, batch));
            }
        }
    }
    if large {
        for n in SQUARE_N {
            cases.push((format!("n{n}xb1"), n, 1));
        }
    }
    cases
}

fn packed_lu<T: BenchScalar>(c: &mut Criterion, env: &Env, cases: &[(String, usize, usize)]) {
    let mut factor = group(c, "lu_factor", T::LABEL);
    for (case, n, batch) in cases {
        let (n, batch) = (*n, *batch);
        let a = Batch::<T>::general(n, n, batch);
        let (mut lu, mut piv, mut parity) = (
            a.data.clone(),
            vec![0i32; n * batch],
            vec![T::default(); batch],
        );
        let lapack = lapack_row!({
            let (mut lu, mut piv, mut parity) = (
                a.data.clone(),
                vec![0i32; n * batch],
                vec![T::default(); batch],
            );
            let a = &a;
            move || {
                lu.copy_from_slice(&a.data);
                tlinalg_blas::lu::lu_factor(
                    tlinalg_blas::Op::LuFactor,
                    n,
                    n,
                    &mut lu,
                    &mut piv,
                    &mut parity,
                )
                .unwrap();
            }
        });
        rows(
            &mut factor,
            env,
            case,
            |par| {
                lu.copy_from_slice(&a.data);
                tlinalg::packed_lu::factor(Op::LuFactor, n, n, &mut lu, &mut piv, &mut parity, par)
                    .unwrap();
            },
            lapack,
        );
    }
    factor.finish();

    let mut prepared = group(c, "lu_solve_prepared", T::LABEL);
    for (case, n, batch) in cases {
        let (n, batch) = (*n, *batch);
        let a = Batch::<T>::general(n, n, batch);
        let b = Batch::<T>::general(n, NRHS, batch);
        let (mut lu, mut piv, mut parity) = (
            a.data.clone(),
            vec![0i32; n * batch],
            vec![T::default(); batch],
        );
        tlinalg::packed_lu::factor(
            Op::LuFactor,
            n,
            n,
            &mut lu,
            &mut piv,
            &mut parity,
            Exec::serial(),
        )
        .unwrap();
        let lu_dims = [n, n, batch];
        let lu_strides = [1, n as isize, (n * n) as isize];
        let piv_dims = [n, batch];
        let piv_strides = [1, n as isize];
        let mut x = b.data.clone();
        let lapack = lapack_row!({
            let (lu, piv, b) = (&lu, &piv, &b);
            let mut x = b.data.clone();
            move || {
                x.copy_from_slice(&b.data);
                tlinalg_blas::lu::lu_solve_prepared(
                    tlinalg_blas::Op::LuSolvePrepared,
                    n,
                    NRHS,
                    lu,
                    piv,
                    &mut x,
                    false,
                    false,
                )
                .unwrap();
            }
        });
        rows(
            &mut prepared,
            env,
            case,
            |par| {
                x.copy_from_slice(&b.data);
                tlinalg::packed_lu::solve_prepared(
                    Op::LuSolvePrepared,
                    RawStridedRef::new(&lu, &lu_dims, &lu_strides, 0).unwrap(),
                    RawStridedRef::new(&piv, &piv_dims, &piv_strides, 0).unwrap(),
                    NRHS,
                    &mut x,
                    false,
                    false,
                    par,
                )
                .unwrap();
            },
            lapack,
        );
    }
    prepared.finish();

    let mut fused = group(c, "lu_factor_solve", T::LABEL);
    for (case, n, batch) in cases {
        let (n, batch) = (*n, *batch);
        let a = Batch::<T>::general(n, n, batch);
        let b = Batch::<T>::general(n, NRHS, batch);
        let (mut lu, mut piv, mut x) = (a.data.clone(), vec![0i32; n * batch], b.data.clone());
        let lapack = lapack_row!({
            let (a, b) = (&a, &b);
            let (mut lu, mut piv, mut x) = (a.data.clone(), vec![0i32; n * batch], b.data.clone());
            move || {
                lu.copy_from_slice(&a.data);
                x.copy_from_slice(&b.data);
                tlinalg_blas::lu::lu_factor_solve(
                    tlinalg_blas::Op::LuFactorSolve,
                    n,
                    NRHS,
                    &mut lu,
                    &mut piv,
                    &mut x,
                )
                .unwrap();
            }
        });
        rows(
            &mut fused,
            env,
            case,
            |par| {
                lu.copy_from_slice(&a.data);
                x.copy_from_slice(&b.data);
                tlinalg::packed_lu::factor_solve(
                    Op::LuFactorSolve,
                    n,
                    NRHS,
                    &mut lu,
                    &mut piv,
                    &mut x,
                    par,
                )
                .unwrap();
            },
            lapack,
        );
    }
    fused.finish();
}

fn solve<T: BenchScalar>(c: &mut Criterion, env: &Env, cases: &[(String, usize, usize)]) {
    let mut g = group(c, "solve", T::LABEL);
    for (case, n, batch) in cases {
        let (n, batch) = (*n, *batch);
        let a = Batch::<T>::general(n, n, batch);
        let b = Batch::<T>::general(n, NRHS, batch);
        let mut x = Batch::<T>::general(n, NRHS, batch);
        let x_dims = [n, NRHS, batch];
        let x_strides = [1, n as isize, (n * NRHS) as isize];
        let lapack = lapack_row!({
            let (a, b) = (&a, &b);
            let (mut ws, mut out) = (RecyclingWorkspace::default(), Vec::new());
            move || {
                tlinalg_blas::solve::solve(
                    tlinalg_blas::Op::Solve,
                    false,
                    a.view(),
                    b.view(),
                    &mut out,
                    &mut ws,
                )
                .unwrap()
            }
        });
        rows(
            &mut g,
            env,
            case,
            |par| {
                let out =
                    strided_view::RawStridedMut::new(&mut x.data, &x_dims, &x_strides, 0).unwrap();
                tlinalg::lu::solve(Op::Solve, a.view(), Some(b.view()), out, false, par).unwrap();
            },
            lapack,
        );
    }
    g.finish();
}

fn cholesky<T: BenchScalar>(c: &mut Criterion, env: &Env, cases: &[(String, usize, usize)]) {
    let mut g = group(c, "cholesky", T::LABEL);
    for (case, n, batch) in cases {
        let (n, batch) = (*n, *batch);
        let a = Batch::<T>::hpd(n, batch);
        let mut l = Vec::new();
        let lapack = lapack_row!({
            let a = &a;
            let (mut ws, mut out) = (RecyclingWorkspace::default(), Vec::new());
            move || {
                tlinalg_blas::cholesky::cholesky(
                    tlinalg_blas::Op::Cholesky,
                    a.view(),
                    &mut out,
                    &mut ws,
                )
                .unwrap()
            }
        });
        rows(
            &mut g,
            env,
            case,
            |par| {
                tlinalg::cholesky::cholesky(Op::Cholesky, a.view(), &mut l, par).unwrap();
            },
            lapack,
        );
    }
    g.finish();
}

fn triangular_solve<T: BenchScalar>(
    c: &mut Criterion,
    env: &Env,
    cases: &[(String, usize, usize)],
) {
    let mut g = group(c, "triangular_solve", T::LABEL);
    for (case, n, batch) in cases {
        let (n, batch) = (*n, *batch);
        let a = Batch::<T>::general(n, n, batch);
        let b = Batch::<T>::general(n, NRHS, batch);
        let mut x = Vec::new();
        let flags = tlinalg::triangular_solve::TriangularSolveFlags {
            left_side: true,
            lower: true,
            transpose_a: false,
            unit_diagonal: false,
        };
        let lapack = lapack_row!({
            let (a, b) = (&a, &b);
            let (mut ws, mut out) = (RecyclingWorkspace::default(), Vec::new());
            let options = tlinalg_blas::triangular_solve::TriangularSolveOptions {
                left_side: true,
                lower: true,
                transpose_a: false,
                unit_diagonal: false,
            };
            move || {
                tlinalg_blas::triangular_solve::triangular_solve(
                    tlinalg_blas::Op::TriangularSolve,
                    options,
                    a.view(),
                    b.view(),
                    &mut out,
                    &mut ws,
                )
                .unwrap()
            }
        });
        rows(
            &mut g,
            env,
            case,
            |par| {
                tlinalg::triangular_solve::triangular_solve(
                    Op::TriangularSolve,
                    a.view(),
                    b.view(),
                    flags,
                    &mut x,
                    par,
                )
                .unwrap();
            },
            lapack,
        );
    }
    g.finish();
}

/// `(case label, m, n, batch)` for the rectangular families.
fn rect_cases(square: &[(String, usize, usize)], tall: bool) -> Vec<(String, usize, usize, usize)> {
    let mut cases: Vec<_> = square
        .iter()
        .map(|(case, n, batch)| (case.clone(), *n, *n, *batch))
        .collect();
    if tall {
        cases.push(("t64x24".to_owned(), 64, 24, 1));
    }
    cases
}

fn qr<T: BenchScalar>(c: &mut Criterion, env: &Env, cases: &[(String, usize, usize, usize)]) {
    let mut g = group(c, "qr", T::LABEL);
    for (case, m, n, batch) in cases {
        let (m, n, batch) = (*m, *n, *batch);
        let a = Batch::<T>::general(m, n, batch);
        let (mut q, mut r) = (Vec::new(), Vec::new());
        let lapack = lapack_row!({
            let a = &a;
            let (mut ws, mut q, mut r) = (RecyclingWorkspace::default(), Vec::new(), Vec::new());
            move || {
                tlinalg_blas::qr::qr(tlinalg_blas::Op::Qr, a.view(), &mut q, &mut r, &mut ws)
                    .unwrap()
            }
        });
        rows(
            &mut g,
            env,
            case,
            |par| {
                tlinalg::qr::qr(Op::Qr, a.view(), &mut q, &mut r, par).unwrap();
            },
            lapack,
        );
    }
    g.finish();
}

fn rank_revealing_qr<T: BenchScalar>(
    c: &mut Criterion,
    env: &Env,
    cases: &[(String, usize, usize, usize)],
) {
    let mut g = group(c, "rank_revealing_qr", T::LABEL);
    for (case, m, n, batch) in cases {
        let (m, n, batch) = (*m, *n, *batch);
        let a = Batch::<T>::general(m, n, batch);
        let (mut q, mut r, mut perm) = (Vec::new(), Vec::new(), Vec::new());
        let lapack = lapack_row!({
            let a = &a;
            let (mut ws, mut q, mut r, mut perm) = (
                RecyclingWorkspace::default(),
                Vec::new(),
                Vec::new(),
                Vec::new(),
            );
            move || {
                tlinalg_blas::qr::rank_revealing_qr(
                    tlinalg_blas::Op::RankRevealingQr,
                    a.view(),
                    tlinalg_blas::qr::RankRevealingQrOutputs {
                        q: &mut q,
                        r: &mut r,
                        permutation: &mut perm,
                    },
                    &mut ws,
                )
                .unwrap()
            }
        });
        rows(
            &mut g,
            env,
            case,
            |par| {
                tlinalg::qr::rank_revealing_qr(
                    Op::RankRevealingQr,
                    a.view(),
                    &mut q,
                    &mut r,
                    &mut perm,
                    par,
                )
                .unwrap();
            },
            lapack,
        );
    }
    g.finish();
}

/// Single matrices past the size at which faer's default parameters would leave the QR algorithm
/// for divide and conquer (128): square, tall and wide.
fn large_svd_cases() -> Vec<(String, usize, usize, usize)> {
    [
        ("n256xb1", 256, 256),
        ("n1024xb1", 1024, 1024),
        ("t1024x256", 1024, 256),
        ("w256x1024", 256, 1024),
    ]
    .into_iter()
    .map(|(case, m, n)| (case.to_owned(), m, n, 1))
    .collect()
}

/// The matrices a singular value decomposition is timed on.
#[derive(Clone, Copy, PartialEq)]
enum SvdInput {
    /// [`Batch::general`]: a growing diagonal plus periodic off-diagonal entries.
    General,
    /// [`Batch::clustered`]: rank-deficient with clustered singular values. At size 160 the `f64`
    /// and `Complex64` decompositions with vectors are repeated with the QR algorithm.
    Clustered,
}

/// Square matrices with clustered singular values: the size at which the decomposition with
/// vectors is repeated, and one at which it is not.
fn clustered_svd_cases() -> Vec<(String, usize, usize, usize)> {
    [160usize, 256]
        .into_iter()
        .map(|n| (format!("n{n}xb1"), n, n, 1))
        .collect()
}

/// Thin factors in the group `svd`, full factors in `svd_full`; `svd_clustered` for
/// [`SvdInput::Clustered`].
fn svd<T: BenchScalar>(
    c: &mut Criterion,
    env: &Env,
    cases: &[(String, usize, usize, usize)],
    full: bool,
    input: SvdInput,
) {
    let family = match (full, input) {
        (false, SvdInput::General) => "svd",
        (true, SvdInput::General) => "svd_full",
        (false, SvdInput::Clustered) => "svd_clustered",
        (true, SvdInput::Clustered) => "svd_full_clustered",
    };
    let mut g = group(c, family, T::LABEL);
    for (case, m, n, batch) in cases {
        let (m, n, batch) = (*m, *n, *batch);
        let a = match input {
            SvdInput::General => Batch::<T>::general(m, n, batch),
            SvdInput::Clustered => Batch::<T>::clustered(m, batch),
        };
        let (mut u, mut s, mut vt) = (Vec::new(), Vec::new(), Vec::new());
        let lapack = lapack_row!({
            let a = &a;
            let (mut ws, mut s, mut u, mut vt) = (
                RecyclingWorkspace::default(),
                Vec::new(),
                Vec::new(),
                Vec::new(),
            );
            move || {
                tlinalg_blas::svd::svd(
                    tlinalg_blas::Op::Svd,
                    if full {
                        tlinalg_blas::svd::SvdMode::Full
                    } else {
                        tlinalg_blas::svd::SvdMode::Thin
                    },
                    a.view(),
                    tlinalg_blas::svd::SvdOutputs {
                        s: &mut s,
                        u: &mut u,
                        vt: &mut vt,
                    },
                    &mut ws,
                )
                .unwrap()
            }
        });
        rows(
            &mut g,
            env,
            case,
            |par| {
                tlinalg::svd::svd(Op::Svd, a.view(), full, &mut u, &mut s, &mut vt, par).unwrap();
            },
            lapack,
        );
    }
    g.finish();
}

fn svdvals<T: BenchScalar>(c: &mut Criterion, env: &Env, cases: &[(String, usize, usize, usize)]) {
    let mut g = group(c, "svdvals", T::LABEL);
    for (case, m, n, batch) in cases {
        let (m, n, batch) = (*m, *n, *batch);
        let a = Batch::<T>::general(m, n, batch);
        let mut s = Vec::new();
        let lapack = lapack_row!({
            let a = &a;
            let (mut ws, mut s, mut u, mut vt) = (
                RecyclingWorkspace::default(),
                Vec::new(),
                Vec::new(),
                Vec::new(),
            );
            move || {
                tlinalg_blas::svd::svd(
                    tlinalg_blas::Op::SvdValues,
                    tlinalg_blas::svd::SvdMode::Values,
                    a.view(),
                    tlinalg_blas::svd::SvdOutputs {
                        s: &mut s,
                        u: &mut u,
                        vt: &mut vt,
                    },
                    &mut ws,
                )
                .unwrap()
            }
        });
        rows(
            &mut g,
            env,
            case,
            |par| {
                tlinalg::svd::svd_values(Op::SvdValues, a.view(), &mut s, par).unwrap();
            },
            lapack,
        );
    }
    g.finish();
}

fn eigh<T: BenchScalar>(
    c: &mut Criterion,
    env: &Env,
    cases: &[(String, usize, usize)],
    values_only: bool,
) {
    let mut g = group(c, if values_only { "eigvalsh" } else { "eigh" }, T::LABEL);
    for (case, n, batch) in cases {
        let (n, batch) = (*n, *batch);
        let a = Batch::<T>::hpd(n, batch);
        let (mut w, mut v, mut wr) = (Vec::new(), Vec::new(), Vec::new());
        let lapack = lapack_row!({
            let a = &a;
            let (mut ws, mut w, mut v) = (RecyclingWorkspace::default(), Vec::new(), Vec::new());
            move || {
                let vectors = if values_only { None } else { Some(&mut v) };
                tlinalg_blas::eigh::eigh(tlinalg_blas::Op::Eigh, a.view(), &mut w, vectors, &mut ws)
                    .unwrap()
            }
        });
        rows(
            &mut g,
            env,
            case,
            |par| {
                if values_only {
                    tlinalg::eigh::eigh_values(Op::EighValues, a.view(), &mut wr, par).unwrap();
                } else {
                    tlinalg::eigh::eigh(Op::Eigh, a.view(), &mut w, &mut v, par).unwrap();
                }
            },
            lapack,
        );
    }
    g.finish();
}

fn eig<T: BenchScalar>(c: &mut Criterion, env: &Env, cases: &[(String, usize, usize)]) {
    let mut g = group(c, "eig", T::LABEL);
    for (case, n, batch) in cases {
        let (n, batch) = (*n, *batch);
        let a = Batch::<T>::general(n, n, batch);
        let (mut w, mut v) = (Vec::new(), Vec::new());
        let lapack = lapack_row!({
            let a = &a;
            let (mut ws, mut w, mut v) = (RecyclingWorkspace::default(), Vec::new(), Vec::new());
            move || {
                tlinalg_blas::eig::eig(
                    tlinalg_blas::Op::Eig,
                    a.view(),
                    &mut w,
                    Some(&mut v),
                    &mut ws,
                )
                .unwrap()
            }
        });
        rows(
            &mut g,
            env,
            case,
            |par| {
                tlinalg::eig::eig(Op::Eig, a.view(), &mut w, &mut v, par).unwrap();
            },
            lapack,
        );
    }
    g.finish();
}

fn full_piv_lu<T: BenchScalar>(c: &mut Criterion, env: &Env, cases: &[(String, usize, usize)]) {
    let mut g = group(c, "full_piv_lu", T::LABEL);
    for (case, n, batch) in cases {
        let (n, batch) = (*n, *batch);
        let a = Batch::<T>::general(n, n, batch);
        let (mut p, mut l, mut u, mut q, mut parity) =
            (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new());
        let lapack = lapack_row!({
            let a = &a;
            let mut ws = RecyclingWorkspace::default();
            let (mut p, mut l, mut u, mut q, mut parity) =
                (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new());
            move || {
                tlinalg_blas::full_piv_lu::full_piv_lu(
                    tlinalg_blas::Op::FullPivLu,
                    a.view(),
                    tlinalg_blas::full_piv_lu::FullPivLuOutputs {
                        p: &mut p,
                        l: &mut l,
                        u: &mut u,
                        q: &mut q,
                        parity: &mut parity,
                    },
                    &mut ws,
                )
                .unwrap()
            }
        });
        rows(
            &mut g,
            env,
            case,
            |par| {
                tlinalg::full_piv_lu::full_piv_lu(
                    Op::FullPivLu,
                    a.view(),
                    tlinalg::full_piv_lu::FullPivLuFactors {
                        p: &mut p,
                        l: &mut l,
                        u: &mut u,
                        q: &mut q,
                        parity: &mut parity,
                    },
                    par,
                )
                .unwrap();
            },
            lapack,
        );
    }
    g.finish();
}

fn real_f64(c: &mut Criterion) {
    let env = Env::new();
    let all = square_cases(true, true);
    let large = square_cases(false, true);
    let rect = rect_cases(&all, true);
    packed_lu::<f64>(c, &env, &all);
    solve::<f64>(c, &env, &all);
    cholesky::<f64>(c, &env, &all);
    triangular_solve::<f64>(c, &env, &all);
    qr::<f64>(c, &env, &rect);
    rank_revealing_qr::<f64>(c, &env, &rect_cases(&large, true));
    let large_svd = large_svd_cases();
    svd::<f64>(
        c,
        &env,
        &[rect.as_slice(), large_svd.as_slice()].concat(),
        false,
        SvdInput::General,
    );
    svd::<f64>(c, &env, &large_svd, true, SvdInput::General);
    svd::<f64>(c, &env, &clustered_svd_cases(), false, SvdInput::Clustered);
    svdvals::<f64>(c, &env, &[rect.as_slice(), large_svd.as_slice()].concat());
    eigh::<f64>(c, &env, &all, false);
    eigh::<f64>(c, &env, &all, true);
    eig::<f64>(c, &env, &large);
    full_piv_lu::<f64>(c, &env, &large);
}

/// A representative complex subset: the small-batch and the mid-size single-matrix regimes.
fn complex_c64(c: &mut Criterion) {
    let env = Env::new();
    let cases = vec![
        ("n4xb1024".to_owned(), 4, 1024),
        ("n128xb1".to_owned(), 128, 1),
    ];
    let rect = rect_cases(&cases, false);
    packed_lu::<num_complex::Complex64>(c, &env, &cases);
    solve::<num_complex::Complex64>(c, &env, &cases);
    qr::<num_complex::Complex64>(c, &env, &rect);
    let large_svd = large_svd_cases();
    let svd_cases = [rect.as_slice(), large_svd.as_slice()].concat();
    svd::<num_complex::Complex64>(c, &env, &svd_cases, false, SvdInput::General);
    svd::<num_complex::Complex64>(c, &env, &large_svd, true, SvdInput::General);
    svd::<num_complex::Complex64>(c, &env, &clustered_svd_cases(), false, SvdInput::Clustered);
    svdvals::<num_complex::Complex64>(c, &env, &svd_cases);
    eigh::<num_complex::Complex64>(c, &env, &cases, false);
}

criterion_group!(benches, real_f64, complex_c64);
criterion_main!(benches);
