//! Batch-lane scheduling: the lane route over the token, the exact partition, forced strategies
//! and the ledger bound.
//!
//! The route is the shared `Exec::batch_decision` (see `crate::batch::decide`): `Auto` fans out
//! whenever the decision gives two or more lanes and then forks the whole budget over them, so a
//! batch with at least `budget` items runs `Par::Seq` items and a smaller batch gives every item
//! its share as faer width. These tests pin the decisions on a table of `(items, budget)`, the
//! widths the lanes receive (they add up to the budget), the exact nonempty partition, the forced
//! routes, which pool the items run on, and, through the `strided-exec` ledger, that no more
//! tasks than the budget are ever live.

use crate::batch::{decide, fixed_out, run, Route, ScratchArena, VecOut};
use crate::Op;
use core::sync::atomic::{AtomicUsize, Ordering::SeqCst};
use std::cell::Cell;
use std::sync::Mutex;
use strided_exec::ledger;
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

/// An owned pool whose workers are named `{name}-{index}`, so a test can tell which pool ran an
/// item.
fn named_pool(name: &'static str, threads: usize) -> Pool<'static> {
    let pool = strided_exec::Pool::builder()
        .threads(threads)
        .thread_name(move |index| format!("{name}-{index}"))
        .build()
        .unwrap();
    Pool::owned(pool)
}

fn owned_pool(threads: usize) -> Pool<'static> {
    named_pool("test", threads)
}

fn token<'a>(pool: &'a Pool<'a>, budget: usize) -> Exec<'a> {
    Exec::new(pool, budget).unwrap()
}

/// A token whose default-strategy model prices every multiply-add at a millisecond, so that even
/// the tiny items of these scheduling tests carry enough estimated work to fan out. The tests of
/// the work gate itself use [`token`] with the default model.
pub(super) fn eager_token<'a>(pool: &'a Pool<'a>, budget: usize) -> Exec<'a> {
    let model = WorkModel::default().with_muladds_per_ns(1e-6);
    Exec::new(pool, budget)
        .unwrap()
        .with_batch_policy(BatchPolicy::new(BatchStrategy::Auto).with_model(model))
}

/// The ledger root of a token made by `Exec::new`.
fn root(exec: &mut Exec<'_>) -> u64 {
    exec.token()
        .ledger_root()
        .expect("a pooled token is a root")
}

/// Whether the current thread is a worker of the pool named `name`.
fn on_pool(name: &str) -> bool {
    std::thread::current()
        .name()
        .is_some_and(|thread| thread.starts_with(name))
}

/// The decision table, written out independently of the driver for `Auto` and the default model
/// (no work gate): lanes are `min(budget, batch)` when that is at least two.
fn expected_lanes(batch: usize, budget: usize) -> usize {
    if batch == 0 {
        0
    } else {
        batch.min(budget).max(1)
    }
}

#[test]
fn driver_bounds_lanes_and_widths_and_visits_each_item_once() {
    let pool = owned_pool(8);
    // (batch, budget). `(5, 4)` is the exact-partition case: four nonempty ranges. `(3, 4)` is a
    // batch smaller than the budget: three lanes whose shares are 2, 1 and 1.
    for (batch, budget) in [
        (0usize, 4usize),
        (1, 4),
        (3, 4),
        (12, 1),
        (12, 2),
        (12, 3),
        (12, 4),
        (16, 99),
        (5, 4),
        (8, 8),
    ] {
        let mut exec = eager_token(&pool, budget);
        let granted = exec.budget();
        let lanes = expected_lanes(batch, granted);
        let mut values = vec![99];
        let mut values_out = VecOut::new(&mut values, batch);
        let widths = Mutex::new(Vec::new());
        let active = AtomicUsize::new(0);
        let peak = AtomicUsize::new(0);
        let seen: Vec<_> = (0..batch).map(|_| AtomicUsize::new(0)).collect();
        let max_width = decide(&exec, batch, 1.0).item_width();
        let caller = std::thread::current().id();
        let root = root(&mut exec);
        LAST_LANES.set(0);
        run(
            Op::Svd,
            batch,
            &mut exec,
            1.0,
            &mut (fixed_out(values_out.spare(), 1, "values"),),
            ScratchArena::new(&mut [], 0),
            |par, _| {
                widths.lock().unwrap().push(par.degree());
            },
            |index, (values,), (), par| {
                assert!(par.degree() >= 1 && par.degree() <= max_width);
                if lanes <= 1 && max_width == 1 {
                    assert_eq!(std::thread::current().id(), caller, "a serial route");
                } else {
                    assert!(on_pool("test"), "wrong numerical pool");
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
        assert_eq!(LAST_LANES.get(), if batch == 0 { 0 } else { lanes });
        let widths = widths.into_inner().unwrap();
        assert_eq!(
            widths.len(),
            lanes,
            "one scratch per lane, {batch}/{budget}"
        );
        if lanes > 1 {
            // The shares add up to the budget, and a batch with at least `budget` items runs every
            // item sequentially (`Par::Seq`).
            assert_eq!(widths.iter().sum::<usize>(), granted, "{batch}/{budget}");
            if batch >= granted {
                assert!(widths.iter().all(|&w| w == 1), "{batch}/{budget}");
            }
        }
        values_out.publish();
        assert_eq!(values, (0..batch).collect::<Vec<_>>());
        assert!(seen.iter().all(|n| n.load(SeqCst) == 1));
        assert_eq!(active.load(SeqCst), 0);
        // The token's budget is the only bound on live tasks, at every nesting depth.
        assert!(
            ledger::max_live(root) <= granted,
            "{batch}/{budget}: max live {} > budget {granted}",
            ledger::max_live(root)
        );
        // This probes driver callback concurrency, not faer's internal numerical concurrency.
        assert!(peak.load(SeqCst) <= granted);
    }
}

/// A single small item must not hand the pool to faer or enter it at all. The shared work model
/// sizes one item below its serial threshold, so the route is one lane of one slot and the item
/// runs inline on the caller: the ledger records no pool entry.
#[test]
fn a_single_small_item_stays_on_the_caller_at_width_one() {
    let pool = owned_pool(4);
    let mut exec = token(&pool, 4);
    // A 4x4 solve is a few dozen multiply-adds, far below the model's serial threshold.
    let item_muladds = 4.0 * 4.0 * 4.0;
    assert_eq!(decide(&exec, 1, item_muladds), Route { lanes: 1, slots: 1 });

    let caller = std::thread::current().id();
    let root = root(&mut exec);
    let mut values = Vec::new();
    let mut values_out = VecOut::new(&mut values, 1);
    let scratch = AtomicUsize::new(0);
    LAST_LANES.set(0);
    run(
        Op::Solve,
        1,
        &mut exec,
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
            assert!(!on_pool("test"));
            values.push(index);
            Ok(())
        },
    )
    .unwrap();
    assert_eq!(LAST_LANES.get(), 1);
    assert_eq!(scratch.load(SeqCst), 1);
    assert_eq!(ledger::installs(root), 0, "no pool entry for a tiny item");
    assert!(ledger::max_live(root) <= 1);
    values_out.publish();
    assert_eq!(values, [0]);
}

/// A batch of tiny items under the default policy stays serial and never enters the pool: the
/// shared work model narrows the tiny total work to width one, so no lane fan-out pays the pool
/// entry (this replaces the deleted per-dimension cutoff).
#[test]
fn a_tiny_batch_does_not_enter_the_pool() {
    let pool = owned_pool(4);
    let mut exec = token(&pool, 4);
    let batch = 8;
    let item_muladds = 4.0 * 4.0 * 4.0;
    assert_eq!(
        decide(&exec, batch, item_muladds),
        Route { lanes: 1, slots: 1 }
    );

    let root = root(&mut exec);
    let mut values = Vec::new();
    let mut values_out = VecOut::new(&mut values, batch);
    LAST_LANES.set(0);
    run(
        Op::Solve,
        batch,
        &mut exec,
        item_muladds,
        &mut (fixed_out(values_out.spare(), 1, "values"),),
        ScratchArena::new(&mut [], 0),
        |par, _| assert_eq!(par.degree(), 1),
        |index, (values,), (), par| {
            assert_eq!(par.degree(), 1);
            assert!(!on_pool("test"));
            values.push(index);
            Ok(())
        },
    )
    .unwrap();
    assert_eq!(LAST_LANES.get(), 1);
    assert_eq!(ledger::installs(root), 0, "no pool entry for a tiny batch");
    values_out.publish();
    assert_eq!(values, (0..batch).collect::<Vec<_>>());
}

/// One large item still receives the whole budget: the work model's width is the full pool, and
/// the driver enters the pool once for that item.
#[test]
fn a_single_large_item_still_gets_the_whole_budget() {
    let pool = owned_pool(4);
    let mut exec = token(&pool, 4);
    let item_muladds = 1e9;
    assert_eq!(decide(&exec, 1, item_muladds), Route { lanes: 1, slots: 4 });

    let root = root(&mut exec);
    let mut values = Vec::new();
    let mut values_out = VecOut::new(&mut values, 1);
    LAST_LANES.set(0);
    run(
        Op::Svd,
        1,
        &mut exec,
        item_muladds,
        &mut (fixed_out(values_out.spare(), 1, "values"),),
        ScratchArena::new(&mut [], 0),
        |par, _| assert_eq!(par.degree(), 4),
        |index, (values,), (), par| {
            assert_eq!(par.degree(), 4);
            assert!(on_pool("test"));
            values.push(index);
            Ok(())
        },
    )
    .unwrap();
    assert_eq!(LAST_LANES.get(), 1);
    assert_eq!(ledger::installs(root), 1, "the pool is entered once");
    assert!(ledger::max_live(root) <= 4);
    values_out.publish();
    assert_eq!(values, [0]);
}

/// A serial token resolves to one lane of one slot whatever the batch, and every item runs on the
/// calling thread at `Par::Seq`.
#[test]
fn a_serial_token_runs_every_item_on_the_caller() {
    let caller = std::thread::current().id();
    let mut exec = Exec::serial();
    assert_eq!(decide(&exec, 8, 1e9), Route { lanes: 1, slots: 1 });
    let mut values = Vec::new();
    let mut values_out = VecOut::new(&mut values, 8);
    LAST_LANES.set(0);
    run(
        Op::Svd,
        8,
        &mut exec,
        1e9,
        &mut (fixed_out(values_out.spare(), 1, "values"),),
        ScratchArena::new(&mut [], 0),
        |par, _| assert_eq!(par.degree(), 1),
        |index, (values,), (), par| {
            assert_eq!(par.degree(), 1);
            assert_eq!(std::thread::current().id(), caller);
            values.push(index);
            Ok(())
        },
    )
    .unwrap();
    assert_eq!(LAST_LANES.get(), 1);
    values_out.publish();
    assert_eq!(values, (0..8).collect::<Vec<_>>());
}

/// The batch runs on the pool the caller's token names, from any calling thread, and a budget
/// above the pool width is clamped once, by `Exec::new`.
#[test]
fn work_runs_on_the_selected_pool_whoever_calls() {
    let selected = named_pool("selected", 4);
    let foreign = named_pool("foreign", 2);
    let one = named_pool("one", 1);
    assert_eq!(
        token(&one, 99).budget(),
        1,
        "budget clamps to the pool size"
    );

    let run_on_selected = |what: &str| {
        let mut exec = eager_token(&selected, 2);
        let root = root(&mut exec);
        let mut values = Vec::new();
        let mut values_out = VecOut::new(&mut values, 8);
        run(
            Op::Qr,
            8,
            &mut exec,
            1.0,
            &mut (fixed_out(values_out.spare(), 1, "values"),),
            ScratchArena::new(&mut [], 0),
            |_, _| (),
            |index, (values,), (), par| {
                assert_eq!(par.degree(), 1);
                assert!(on_pool("selected"), "{what}: wrong pool");
                values.push(index);
                Ok(())
            },
        )
        .unwrap();
        assert!(ledger::max_live(root) <= 2, "{what}");
        values_out.publish();
        assert_eq!(values, (0..8).collect::<Vec<_>>());
    };
    run_on_selected("plain thread");
    // Called from a worker of another pool: the foreign worker waits, the selected pool works.
    let mut foreign_exec = token(&foreign, 2);
    foreign_exec
        .token()
        .external(|_| run_on_selected("foreign worker"));
    // And from a worker of the selected pool itself: the batch runs in place.
    token(&selected, 2)
        .token()
        .external(|_| run_on_selected("selected worker"));
}

/// Forced strategies are clamped, never silently downgraded, and the lane width reaches both the
/// scratch and the numerical execution inside each lane. Every grant is asserted directly, so an
/// oversized grant cannot hide behind a minimum, and the ledger bounds the live tasks by the
/// budget.
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
        let mut exec = token(&pool, 4).with_batch_policy(BatchPolicy::new(strategy));
        let root = root(&mut exec);
        let mut values = Vec::new();
        let mut values_out = VecOut::new(&mut values, 8);
        let scratch_builds = AtomicUsize::new(0);
        let items = AtomicUsize::new(0);
        LAST_LANES.set(0);
        run(
            Op::Svd,
            8,
            &mut exec,
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
        assert!(ledger::max_live(root) <= 4, "{strategy:?}");
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
    crate::qr::qr(Op::Qr, view, &mut seq_q, &mut seq_r, &mut Exec::serial()).unwrap();
    assert!(residual(&seq_q, &seq_r) < 1e-12, "sequential baseline");
    let pool = owned_pool(4);
    for strategy in [
        BatchStrategy::Outer,
        BatchStrategy::Inner,
        BatchStrategy::Split {
            lanes: core::num::NonZeroUsize::new(2).unwrap(),
        },
    ] {
        let mut exec = token(&pool, 2).with_batch_policy(BatchPolicy::new(strategy));
        let (mut q, mut r) = (Vec::new(), Vec::new());
        crate::qr::qr(Op::Qr, view, &mut q, &mut r, &mut exec).unwrap();
        assert!(
            residual(&q, &r) < 1e-10,
            "{strategy:?}: QR does not reproduce A"
        );
        assert_eq!(q.len(), seq_q.len(), "{strategy:?}");
    }
}

/// The `Auto` lane count does not depend on the item extent: a 65-column solve or reflector
/// application fans out exactly like the 64-column case, for every family.
#[test]
fn auto_lanes_do_not_depend_on_the_item_extent() {
    use crate::householder::{apply_reflectors, ReflectorShape};
    use crate::triangular_solve::{triangular_solve, TriangularSolveFlags};
    let pool = owned_pool(4);
    let mut exec = eager_token(&pool, 2);

    let n = 2;
    let batch = 4;
    let a = [1.0_f64, 0.0, 0.0, 1.0];
    let adims = [n, n, batch];
    let astrides = [1, n as isize, 0];
    for (width, lanes) in [(64usize, 2usize), (65, 2)] {
        let dims = [n, width, batch];
        let strides = [1, n as isize, (n * width) as isize];
        let mut x = vec![1.0; n * width * batch];
        crate::lu::solve(
            Op::Solve,
            RawStridedRef::new(&a, &adims, &astrides, 0).unwrap(),
            None,
            RawStridedMut::new(&mut x, &dims, &strides, 0).unwrap(),
            false,
            &mut exec,
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
    for (width, lanes) in [(64usize, 2usize), (65, 2)] {
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
            &mut exec,
        )
        .unwrap();
        assert_eq!(LAST_LANES.get(), lanes, "triangular width={width}");
    }
    // A reflector application counts both the compact state extent and the target columns.
    let rows = 2;
    for (a_cols, cols, lanes) in [
        (64usize, 1usize, 2usize),
        (65, 1, 2),
        (1, 64, 2),
        (1, 65, 2),
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
            &mut exec,
        )
        .unwrap();
        assert_eq!(
            LAST_LANES.get(),
            lanes,
            "reflector a_cols={a_cols} cols={cols}"
        );
    }
}

/// The packed routes fan out over the budget like every other family, also for a large item.
#[test]
fn packed_routes_fan_out_for_a_large_item() {
    use crate::packed_lu::{factor, factor_solve, solve_prepared};
    let pool = owned_pool(4);
    let mut exec = eager_token(&pool, 2);
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
    factor(
        Op::LuFactor,
        n,
        n,
        &mut lu,
        &mut pivots,
        &mut parity,
        &mut exec,
    )
    .unwrap();
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
        &mut exec,
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
        &mut exec,
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
    crate::svd::svd_values(Op::SvdValues, view, &mut s, &mut Exec::serial()).unwrap();
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
        &mut Exec::serial(),
    )
    .unwrap();
    assert!(coeff.is_empty());
}

/// The forced routes are evaluated against the original budget; the work model does not gate them.
#[test]
fn forced_strategies_ignore_the_work_model() {
    let pool = owned_pool(4);
    let policy = BatchPolicy::new(BatchStrategy::Outer)
        .with_model(WorkModel::default().with_min_work_per_lane_ns(10.0));
    let mut exec = token(&pool, 2).with_batch_policy(policy);
    // A work gate no item reaches: `Outer` still clamps to the budget.
    let n = 65;
    let batch = 4;
    let a: Vec<f64> = (0..n * n * batch)
        .map(|i| if i % (n + 1) == 0 { 4.0 } else { 0.1 })
        .collect();
    let dims = [n, n, batch];
    let strides = [1, n as isize, (n * n) as isize];
    let view = RawStridedRef::new(&a, &dims, &strides, 0).unwrap();
    let (mut q, mut r) = (Vec::new(), Vec::new());
    crate::qr::qr(Op::Qr, view, &mut q, &mut r, &mut exec).unwrap();
    assert_eq!(LAST_LANES.get(), 2);
}
