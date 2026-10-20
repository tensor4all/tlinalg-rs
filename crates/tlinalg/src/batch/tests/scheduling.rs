//! Batch-lane scheduling: the temporary step-1 Auto rule, the exact partition, and forced
//! strategies.
//!
//! tlinalg-rs#58 "Defaults" and tcontract-rs#123 step 1 keep tlinalg's pre-0.3 `Auto` fan-out in
//! 0.3: outer lanes only while the largest item extent is at most 64 (packed LU excepted), RHS
//! columns counted by default behind the `TLINALG_AUTO_RHS` switch. A one-lane decision now takes
//! its item width from the shared work model, so a small item stays on the caller. These tests pin
//! the decisions on a table of `(items, n, k, budget)`, including RHS-heavy solves and reflector
//! targets, and pin the exact nonempty partition, the forced routes and the typed vendor refusal.

use crate::batch::{auto_extent_with, decide, fixed_out, run, ScratchArena, VecOut};
use crate::Op;
use core::sync::atomic::{AtomicUsize, Ordering::SeqCst};
use std::cell::Cell;
use strided_view::{RawStridedMut, RawStridedRef};
use tcontract_exec::batch::{BatchPolicy, BatchStrategy, WorkModel};
use tcontract_exec::{Exec, Pool};

thread_local! {
    // The driver records on its caller before dispatch: real API shape decisions are observable
    // without changing release code or interfering with tests on other threads.
    static LAST_LANES: Cell<usize> = const { Cell::new(0) };
}

pub(in crate::batch) fn record_lanes(lanes: usize) {
    LAST_LANES.set(lanes);
}

/// An owned pool, kept alive by `Pool::owned`, so tests can return a token borrowing it.
fn owned_pool(threads: usize) -> Pool<'static> {
    let tp = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .unwrap();
    Pool::owned(tp)
}

fn token<'a>(pool: &'a Pool<'a>, budget: usize) -> Exec<'a> {
    Exec::rayon(pool).with_budget(budget).unwrap()
}

/// The step-1 Auto decision, written out independently of the driver. `item_threads` is the width
/// the shared work model already resolved for the one-lane case.
fn legacy_auto(
    batch: usize,
    budget: usize,
    max_item_dim: Option<usize>,
    item_threads: usize,
) -> (usize, usize) {
    let budget = budget.max(1);
    let fans_out = budget > 1 && batch >= budget && max_item_dim.is_none_or(|dim| dim <= 64);
    if fans_out {
        (budget.min(batch), 1)
    } else {
        (1, item_threads.clamp(1, budget))
    }
}

#[test]
fn driver_bounds_lanes_and_hints_and_visits_each_item_once() {
    let pool = owned_pool(8);
    // (batch, budget, max_item_dim, lanes, item_threads). `dim` is the legacy extent the family
    // reports (`None` is packed LU's exception); `(5, 4)` is the exact-partition fix, four nonempty
    // ranges where the old `div_ceil` split produced three.
    for (batch, budget, dim, lanes, hint) in [
        (0usize, 4usize, Some(1usize), 0usize, 0usize),
        (1, 4, Some(1), 1, 1),
        (3, 4, Some(1), 1, 1),
        (12, 1, Some(1), 1, 1),
        (12, 2, Some(1), 2, 1),
        (12, 3, Some(1), 3, 1),
        (12, 4, Some(1), 4, 1),
        (16, 99, Some(1), 8, 1),
        (5, 4, Some(1), 4, 1),
        (8, 4, Some(64), 4, 1),
        (8, 4, Some(65), 1, 1),
        (8, 4, None, 4, 1),
    ] {
        let (want_lanes, _) = if batch == 0 {
            (0, 0)
        } else {
            legacy_auto(batch, budget.min(8), dim, hint)
        };
        assert_eq!(lanes, want_lanes, "table row {batch}/{budget}/{dim:?}");
        let mut values = vec![99];
        let mut values_out = VecOut::new(&mut values, batch);
        let scratch = AtomicUsize::new(0);
        let active = AtomicUsize::new(0);
        let peak = AtomicUsize::new(0);
        let seen: Vec<_> = (0..batch).map(|_| AtomicUsize::new(0)).collect();
        let caller = std::thread::current().id();
        LAST_LANES.set(0);
        run(
            Op::Svd,
            batch,
            token(&pool, budget),
            dim,
            1.0,
            &mut (fixed_out(values_out.spare(), 1, "values"),),
            ScratchArena::new(&mut [], 0),
            |par, _| {
                scratch.fetch_add(1, SeqCst);
                assert_eq!(par.degree(), hint);
            },
            |index, (values,), (), par| {
                assert_eq!(par.degree(), hint);
                if lanes == 1 && hint == 1 {
                    assert_eq!(std::thread::current().id(), caller);
                } else {
                    assert!(
                        pool.rayon_pool().current_thread_index().is_some(),
                        "wrong numerical pool"
                    );
                }
                let n = active.fetch_add(1, SeqCst) + 1;
                peak.fetch_max(n, SeqCst);
                if lanes > 1 {
                    std::thread::sleep(std::time::Duration::from_millis(1));
                }
                seen[index].fetch_add(1, SeqCst);
                values.push(index);
                active.fetch_sub(1, SeqCst);
                Ok(())
            },
        )
        .unwrap();
        assert_eq!(
            LAST_LANES.get(),
            lanes,
            "batch={batch}, budget={budget}, dim={dim:?}"
        );
        assert_eq!(scratch.load(SeqCst), lanes);
        values_out.publish();
        assert_eq!(values, (0..batch).collect::<Vec<_>>());
        assert!(seen.iter().all(|n| n.load(SeqCst) == 1));
        assert_eq!(active.load(SeqCst), 0);
        // This probes driver callback concurrency, not faer's internal numerical concurrency.
        assert!(peak.load(SeqCst) <= budget.min(8));
    }
}

/// A single small item must not hand the whole pool to faer or enter the pool at all. The shared
/// work model sizes one item below its serial threshold, so the driver resolves `(1, 1)` and
/// `Exec::install(1, ..)` runs the item inline on the caller.
#[test]
fn a_single_small_item_stays_on_the_caller_at_width_one() {
    let pool = owned_pool(4);
    let exec = token(&pool, 4);
    // A 4x4 solve is a few dozen multiply-adds, far below the model's serial threshold.
    let item_muladds = 4.0 * 4.0 * 4.0;
    assert_eq!(
        decide(Op::Solve, exec, 1, Some(4), item_muladds).unwrap(),
        (1, 1)
    );

    let caller = std::thread::current().id();
    let mut values = Vec::new();
    let mut values_out = VecOut::new(&mut values, 1);
    let scratch = AtomicUsize::new(0);
    LAST_LANES.set(0);
    run(
        Op::Solve,
        1,
        exec,
        Some(4),
        item_muladds,
        &mut (fixed_out(values_out.spare(), 1, "values"),),
        ScratchArena::new(&mut [], 0),
        |par, _| {
            assert_eq!(par.degree(), 1);
            scratch.fetch_add(1, SeqCst);
        },
        |index, (values,), (), par| {
            assert_eq!(par.degree(), 1);
            assert_eq!(std::thread::current().id(), caller);
            assert!(pool.rayon_pool().current_thread_index().is_none());
            values.push(index);
            Ok(())
        },
    )
    .unwrap();
    assert_eq!(LAST_LANES.get(), 1);
    assert_eq!(scratch.load(SeqCst), 1);
    values_out.publish();
    assert_eq!(values, [0]);
}

/// One large item still receives the whole budget: the work model's width is the full pool, and
/// the driver installs it once for that item.
#[test]
fn a_single_large_item_still_gets_the_whole_budget() {
    let pool = owned_pool(4);
    let exec = token(&pool, 4);
    let item_muladds = 1e9;
    assert_eq!(
        decide(Op::Svd, exec, 1, Some(1024), item_muladds).unwrap(),
        (1, 4)
    );

    let mut values = Vec::new();
    let mut values_out = VecOut::new(&mut values, 1);
    LAST_LANES.set(0);
    run(
        Op::Svd,
        1,
        exec,
        Some(1024),
        item_muladds,
        &mut (fixed_out(values_out.spare(), 1, "values"),),
        ScratchArena::new(&mut [], 0),
        |par, _| assert_eq!(par.degree(), 4),
        |index, (values,), (), par| {
            assert_eq!(par.degree(), 4);
            assert!(pool.rayon_pool().current_thread_index().is_some());
            values.push(index);
            Ok(())
        },
    )
    .unwrap();
    assert_eq!(LAST_LANES.get(), 1);
    values_out.publish();
    assert_eq!(values, [0]);
}

#[test]
fn auto_extent_switch_counts_rhs_by_default() {
    // Default (include): the RHS/reflector extent wins when it is larger.
    assert_eq!(auto_extent_with(2, 65, true), 65);
    assert_eq!(auto_extent_with(16, 1, true), 16);
    // Exclude: the matrix order alone decides.
    assert_eq!(auto_extent_with(2, 65, false), 2);
    assert_eq!(auto_extent_with(16, 1, false), 16);
}

#[test]
fn effective_one_stays_on_caller_and_nested_work_uses_only_selected_pool() {
    let selected = owned_pool(4);
    let foreign = owned_pool(2);
    let one = owned_pool(1);
    let check = |exec: Exec<'_>, expected_hint: usize, stays: bool| {
        let caller = std::thread::current().id();
        exec.install(expected_hint, |par| {
            assert_eq!(crate::faer_par(par).degree(), expected_hint);
            if stays {
                assert_eq!(std::thread::current().id(), caller);
            } else {
                assert!(selected.rayon_pool().current_thread_index().is_some());
                assert!(foreign.rayon_pool().current_thread_index().is_none());
            }
        });
    };
    check(token(&selected, 1), 1, true);
    check(token(&one, 4), 1, true);
    foreign
        .rayon_pool()
        .install(|| check(token(&selected, 1), 1, true));
    foreign
        .rayon_pool()
        .install(|| check(token(&selected, 3), 3, false));
    selected
        .rayon_pool()
        .install(|| check(token(&selected, 3), 3, false));
    foreign.rayon_pool().install(|| {
        let mut values = Vec::new();
        let mut values_out = VecOut::new(&mut values, 8);
        run(
            Op::Qr,
            8,
            token(&selected, 2),
            Some(1),
            1.0,
            &mut (fixed_out(values_out.spare(), 1, "values"),),
            ScratchArena::new(&mut [], 0),
            |_, _| (),
            |index, (values,), (), par| {
                assert_eq!(par.degree(), 1);
                assert!(selected.rayon_pool().current_thread_index().is_some());
                assert!(foreign.rayon_pool().current_thread_index().is_none());
                values.push(index);
                Ok(())
            },
        )
        .unwrap();
        values_out.publish();
        assert_eq!(values, (0..8).collect::<Vec<_>>());
    });
    assert_eq!(
        token(&one, 99).budget(),
        1,
        "budget clamps to the pool size"
    );
}

/// Forced strategies are clamped, never silently downgraded, and `item_threads` reaches both the
/// scratch and the numerical execution inside each lane. Every grant is asserted directly, so an
/// oversized grant cannot hide behind a minimum.
#[test]
fn forced_strategies_are_clamped_and_item_threads_reach_the_items() {
    let pool = owned_pool(8);
    for (strategy, lanes, item_threads) in [
        (BatchStrategy::Sequential, 1usize, 1usize),
        (BatchStrategy::Outer, 4, 1),
        (BatchStrategy::Inner, 1, 4),
        (
            BatchStrategy::Split {
                lanes: core::num::NonZeroUsize::new(2).unwrap(),
            },
            2,
            2,
        ),
    ] {
        let exec = token(&pool, 4).with_batch_policy(BatchPolicy::new(strategy));
        let mut values = Vec::new();
        let mut values_out = VecOut::new(&mut values, 8);
        let scratch_builds = AtomicUsize::new(0);
        let items = AtomicUsize::new(0);
        LAST_LANES.set(0);
        run(
            Op::Svd,
            8,
            exec,
            Some(1),
            1.0,
            &mut (fixed_out(values_out.spare(), 1, "values"),),
            ScratchArena::new(&mut [], 0),
            |par, _| {
                assert_eq!(par.degree(), item_threads, "scratch {strategy:?}");
                scratch_builds.fetch_add(1, SeqCst);
            },
            |index, (values,), (), par| {
                assert_eq!(par.degree(), item_threads, "item {strategy:?}");
                items.fetch_add(1, SeqCst);
                values.push(index);
                Ok(())
            },
        )
        .unwrap();
        assert_eq!(LAST_LANES.get(), lanes, "{strategy:?}");
        // Exactly one scratch per lane and every item visited once.
        assert_eq!(scratch_builds.load(SeqCst), lanes, "{strategy:?}");
        assert_eq!(items.load(SeqCst), 8, "{strategy:?}");
        values_out.publish();
        assert_eq!(values, (0..8).collect::<Vec<_>>());
    }
}

/// Forced policies run the **real** kernels: QR under `Outer`/`Inner`/`Split` reproduces every
/// input matrix in the residual, so nested-lane scratch sizing and results are exercised, not just
/// index-pushing closures. Item parallelism can change the arithmetic, so the check is a residual
/// rather than bitwise equality.
#[test]
fn forced_kernels_match_sequential_in_the_residual() {
    let n = 6;
    let batch = 4;
    let a: Vec<f64> = (0..n * n * batch)
        .map(|i| {
            let (row, col) = (i % n, (i / n) % n);
            if row == col {
                4.0 + row as f64
            } else {
                0.1 * (row + col) as f64
            }
        })
        .collect();
    let dims = [n, n, batch];
    let strides = [1, n as isize, (n * n) as isize];
    let view = RawStridedRef::new(&a, &dims, &strides, 0).unwrap();
    let residual = |q: &[f64], r: &[f64]| -> f64 {
        let mut worst = 0.0f64;
        for item in 0..batch {
            let start = item * n * n;
            let (q, r, a) = (
                &q[start..start + n * n],
                &r[start..start + n * n],
                &a[start..start + n * n],
            );
            for row in 0..n {
                for col in 0..n {
                    let mut acc = 0.0;
                    for j in 0..n {
                        acc += q[row + j * n] * r[j + col * n];
                    }
                    worst = worst.max((acc - a[row + col * n]).abs());
                }
            }
        }
        worst
    };
    let mut seq_q = Vec::new();
    let mut seq_r = Vec::new();
    crate::qr::qr(Op::Qr, view, &mut seq_q, &mut seq_r, Exec::serial()).unwrap();
    assert!(residual(&seq_q, &seq_r) < 1e-12, "sequential baseline");
    let pool = owned_pool(4);
    for strategy in [
        BatchStrategy::Outer,
        BatchStrategy::Inner,
        BatchStrategy::Split {
            lanes: core::num::NonZeroUsize::new(2).unwrap(),
        },
    ] {
        let exec = token(&pool, 2).with_batch_policy(BatchPolicy::new(strategy));
        let (mut q, mut r) = (Vec::new(), Vec::new());
        crate::qr::qr(Op::Qr, view, &mut q, &mut r, exec).unwrap();
        assert!(
            residual(&q, &r) < 1e-10,
            "{strategy:?}: QR does not reproduce A"
        );
        assert_eq!(q.len(), seq_q.len(), "{strategy:?}");
    }
}

/// The legacy Auto measure counts the RHS/reflector target extent by default, exactly as 0.2 did:
/// a 65-column solve or reflector application stays on one lane where the 64-column case fans out.
#[test]
fn auto_counts_rhs_and_reflector_extents() {
    use crate::householder::{apply_reflectors, ReflectorShape};
    use crate::triangular_solve::{triangular_solve, TriangularSolveFlags};
    let pool = owned_pool(4);
    let exec = token(&pool, 2);

    let n = 2;
    let batch = 4;
    let a = [1.0_f64, 0.0, 0.0, 1.0];
    let adims = [n, n, batch];
    let astrides = [1, n as isize, 0];
    for (width, lanes) in [(64usize, 2usize), (65, 1)] {
        let dims = [n, width, batch];
        let strides = [1, n as isize, (n * width) as isize];
        let mut x = vec![1.0; n * width * batch];
        crate::lu::solve(
            Op::Solve,
            RawStridedRef::new(&a, &adims, &astrides, 0).unwrap(),
            None,
            RawStridedMut::new(&mut x, &dims, &strides, 0).unwrap(),
            false,
            exec,
        )
        .unwrap();
        assert_eq!(LAST_LANES.get(), lanes, "solve width={width}");
    }
    // The triangular solve reports the larger of the two RHS extents.
    let flags = TriangularSolveFlags {
        left_side: true,
        lower: true,
        transpose_a: false,
        unit_diagonal: false,
    };
    for (width, lanes) in [(64usize, 2usize), (65, 1)] {
        let dims = [n, width, batch];
        let strides = [1, n as isize, (n * width) as isize];
        let b = vec![3.0; n * width * batch];
        let mut x = Vec::new();
        triangular_solve(
            Op::TriangularSolve,
            RawStridedRef::new(&a, &adims, &astrides, 0).unwrap(),
            RawStridedRef::new(&b, &dims, &strides, 0).unwrap(),
            flags,
            &mut x,
            exec,
        )
        .unwrap();
        assert_eq!(LAST_LANES.get(), lanes, "triangular width={width}");
    }
    // A reflector application counts both the compact state extent and the target columns.
    let rows = 2;
    for (a_cols, cols, lanes) in [
        (64usize, 1usize, 2usize),
        (65, 1, 1),
        (1, 64, 2),
        (1, 65, 1),
    ] {
        let mut state = vec![0.0; rows * a_cols * batch];
        for item in state.chunks_exact_mut(rows * a_cols) {
            item[0] = 1.0;
        }
        let mut c = vec![3.0; rows * cols * batch];
        apply_reflectors(
            Op::HouseholderQrQColumns,
            ReflectorShape {
                rows,
                a_cols,
                cols,
                k: 1,
            },
            batch,
            &state,
            &[2.0; 4],
            &mut c,
            false,
            exec,
        )
        .unwrap();
        assert_eq!(
            LAST_LANES.get(),
            lanes,
            "reflector a_cols={a_cols} cols={cols}"
        );
    }
}

/// The packed routes keep their no-cutoff exception: `max_item_dim = None`, so a large item still
/// fans out.
#[test]
fn packed_routes_keep_the_no_cutoff_exception() {
    use crate::packed_lu::{factor, factor_solve, solve_prepared};
    let pool = owned_pool(4);
    let exec = token(&pool, 2);
    let batch = 4;
    let n = 65;
    let mut lu = vec![0.0; n * n * batch];
    for item in lu.chunks_exact_mut(n * n) {
        for i in 0..n {
            item[i + i * n] = 1.0;
        }
    }
    let mut pivots = vec![0; n * batch];
    let mut parity = vec![0.0; batch];
    factor(Op::LuFactor, n, n, &mut lu, &mut pivots, &mut parity, exec).unwrap();
    assert_eq!(LAST_LANES.get(), 2);
    let dims = [n, n, batch];
    let strides = [1, n as isize, (n * n) as isize];
    let pdims = [n, batch];
    let pstrides = [1, n as isize];
    let mut rhs = vec![3.0; n * 2 * batch];
    solve_prepared(
        Op::LuSolvePrepared,
        RawStridedRef::new(&lu, &dims, &strides, 0).unwrap(),
        RawStridedRef::new(&pivots, &pdims, &pstrides, 0).unwrap(),
        2,
        &mut rhs,
        false,
        false,
        exec,
    )
    .unwrap();
    assert_eq!(LAST_LANES.get(), 2);
    factor_solve(
        Op::LuFactorSolve,
        n,
        2,
        &mut lu,
        &mut pivots,
        &mut rhs,
        exec,
    )
    .unwrap();
    assert_eq!(LAST_LANES.get(), 2);
}

/// Work estimates must cast before multiplying: `2^22` cubed overflows `usize` on a 64-bit debug
/// build, but an empty batch must return before any numerical work. (Reproducer for the
/// post-review finding in `svd.rs`.)
#[test]
fn work_estimates_do_not_overflow_before_the_empty_exit() {
    let big = 1usize << 22;
    let dims = [big, big, 0usize];
    let strides = [1isize, 1, 1];
    let empty: [f64; 0] = [];
    let view = RawStridedRef::new(&empty, &dims, &strides, 0).unwrap();
    let mut s = Vec::new();
    crate::svd::svd_values(Op::SvdValues, view, &mut s, Exec::serial()).unwrap();
    assert!(s.is_empty());
    let mut state: [f64; 0] = [];
    let mut coeff = Vec::new();
    crate::householder::compact_factor(
        Op::HouseholderQr,
        big,
        big,
        0,
        &mut state,
        &mut coeff,
        Exec::serial(),
    )
    .unwrap();
    assert!(coeff.is_empty());
}

/// The forced routes are evaluated against the original budget even though `WorkModel` exists.
#[test]
fn forced_strategies_ignore_the_auto_cutoff() {
    let pool = owned_pool(4);
    let policy = BatchPolicy::new(BatchStrategy::Outer)
        .with_model(WorkModel::default().with_min_work_per_lane_ns(10.0));
    let exec = token(&pool, 2).with_batch_policy(policy);
    // An ordinary family at extent 65: Auto keeps one lane, `Outer` still clamps to the budget.
    let n = 65;
    let batch = 4;
    let a: Vec<f64> = (0..n * n * batch)
        .map(|i| if i % (n + 1) == 0 { 4.0 } else { 0.1 })
        .collect();
    let dims = [n, n, batch];
    let strides = [1, n as isize, (n * n) as isize];
    let view = RawStridedRef::new(&a, &dims, &strides, 0).unwrap();
    let (mut q, mut r) = (Vec::new(), Vec::new());
    crate::qr::qr(Op::Qr, view, &mut q, &mut r, exec).unwrap();
    assert_eq!(LAST_LANES.get(), 2);
}
