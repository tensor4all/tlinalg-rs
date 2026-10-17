//! Batch-lane scheduling: the shared work model, the exact partition, and forced strategies.
//!
//! The pre-0.3 driver chose lanes from `AUTO_FAN_OUT_MAX_ITEM_DIM` (a 64 item-extent cutoff) and
//! split with `div_ceil`, which could return fewer lanes than requested. 0.3 asks
//! `tcontract_exec::Exec::batch_decision` and partitions into exactly the decided number of
//! nonempty ranges (tlinalg-rs#58). These tests pin the new `(lanes, item_threads)` cases,
//! including the solve/reflector extents and the packed routes, so the scheduling change is
//! explicit rather than implicit (tlinalg-rs#57/#58).

use crate::batch::{check_route, out, run};
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

#[test]
fn driver_bounds_lanes_and_hints_and_visits_each_item_once() {
    let pool = owned_pool(8);
    // (batch, budget, item_muladds, lanes, item_threads). Auto now follows the shared work model:
    // a small item fans out to `min(budget, items)` lanes with item threads one, and a single
    // large item keeps one lane with the whole budget. `(5, 4)` is the partition fix: it is exactly
    // four nonempty ranges, where the old `div_ceil` split produced three.
    for (batch, budget, item_muladds, lanes, hint) in [
        (0usize, 4usize, 1.0, 0usize, 1usize),
        (1, 4, 1.0, 1, 1),
        (12, 1, 1.0, 1, 1),
        (12, 2, 1.0, 2, 1),
        (12, 3, 1.0, 3, 1),
        (12, 4, 1.0, 4, 1),
        (16, 99, 1.0, 8, 1),
        (3, 4, 1.0, 3, 1),
        (5, 4, 1.0, 4, 1),
        (8, 4, 1.0, 4, 1),
        (1, 4, 1e9, 1, 4),
        (8, 4, 1e9, 4, 1),
    ] {
        let mut values = vec![99];
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
            item_muladds,
            &mut (out(&mut values, 1),),
            |par| {
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
            "batch={batch}, budget={budget}, item_muladds={item_muladds}"
        );
        assert_eq!(scratch.load(SeqCst), lanes);
        assert_eq!(values, (0..batch).collect::<Vec<_>>());
        assert!(seen.iter().all(|n| n.load(SeqCst) == 1));
        assert_eq!(active.load(SeqCst), 0);
        // This probes driver callback concurrency, not faer's internal numerical concurrency.
        assert!(peak.load(SeqCst) <= budget.min(8));
    }
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
        run(
            Op::Qr,
            8,
            token(&selected, 2),
            1.0,
            &mut (out(&mut values, 1),),
            |_| (),
            |index, (values,), (), par| {
                assert_eq!(par.degree(), 1);
                assert!(selected.rayon_pool().current_thread_index().is_some());
                assert!(foreign.rayon_pool().current_thread_index().is_none());
                values.push(index);
                Ok(())
            },
        )
        .unwrap();
        assert_eq!(values, (0..8).collect::<Vec<_>>());
    });
    assert_eq!(
        token(&one, 99).budget(),
        1,
        "budget clamps to the pool size"
    );
}

/// Forced strategies are clamped, never silently downgraded, and `item_threads` reaches both the
/// scratch and the numerical execution inside each lane.
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
        let observed = AtomicUsize::new(usize::MAX);
        LAST_LANES.set(0);
        run(
            Op::Svd,
            8,
            exec,
            1.0,
            &mut (out(&mut values, 1),),
            |par| {
                observed.fetch_min(par.degree(), SeqCst);
            },
            |index, (values,), (), par| {
                observed.fetch_min(par.degree(), SeqCst);
                values.push(index);
                Ok(())
            },
        )
        .unwrap();
        assert_eq!(LAST_LANES.get(), lanes, "{strategy:?}");
        // Every item ran at the decision's item width (faer's degree is the requested count).
        assert_eq!(observed.load(SeqCst), item_threads, "{strategy:?}");
        assert_eq!(values, (0..8).collect::<Vec<_>>());
    }
}

/// `WholeBatchVendor` is refused on this route even when the batch (and the early packed exits)
/// have no work; the refusal is typed and precedes any write.
#[test]
fn whole_batch_vendor_is_unavailable_even_on_an_empty_batch() {
    let policy = BatchPolicy::new(BatchStrategy::WholeBatchVendor);
    let exec = Exec::serial().with_batch_policy(policy);
    assert!(matches!(
        check_route(Op::Svd, exec, 0, 1.0),
        Err(crate::Error::StrategyUnavailable {
            op: Op::Svd,
            strategy: BatchStrategy::WholeBatchVendor,
        })
    ));
    let pool = owned_pool(2);
    let exec = token(&pool, 2).with_batch_policy(policy);
    assert!(matches!(
        run(
            Op::Qr,
            0,
            exec,
            1.0,
            &mut (out(&mut Vec::<f64>::new(), 1),),
            |_| (),
            |_, _, _, _| Ok(()),
        ),
        Err(crate::Error::StrategyUnavailable { .. })
    ));
}

/// The work estimate a family reports is the input to `Auto`, so the RHS extent of a solve and the
/// target extent of a reflector application move the lane count once the model has a work gate.
/// This is what "work, not a dimension cutoff" means for the routes the old cutoff mishandled.
#[test]
fn solve_and_reflector_extents_enter_the_work_estimate() {
    use crate::householder::{apply_reflectors, ReflectorShape};
    use crate::triangular_solve::{triangular_solve, TriangularSolveFlags};
    let pool = owned_pool(4);
    let model = WorkModel::default().with_min_work_per_lane_ns(10.0);
    let policy = BatchPolicy::new(BatchStrategy::Auto).with_model(model);
    let exec = |budget: usize| token(&pool, budget).with_batch_policy(policy);

    let n = 2;
    let batch = 4;
    let a = [1.0_f64, 0.0, 0.0, 1.0];
    let adims = [n, n, batch];
    let astrides = [1, n as isize, 0];
    // A solve with more RHS columns is more work, so the work gate grants it more lanes.
    for (width, lanes) in [(1usize, 1usize), (1000, 2)] {
        let dims = [n, width, batch];
        let strides = [1, n as isize, (n * width) as isize];
        let mut x = vec![1.0; n * width * batch];
        crate::lu::solve(
            Op::Solve,
            RawStridedRef::new(&a, &adims, &astrides, 0).unwrap(),
            None,
            RawStridedMut::new(&mut x, &dims, &strides, 0).unwrap(),
            false,
            exec(2),
        )
        .unwrap();
        assert_eq!(LAST_LANES.get(), lanes, "solve width={width}");
    }
    // A reflector application with a wider target is more work too.
    for (a_cols, cols, lanes) in [(1usize, 1usize, 1usize), (1, 1000, 2)] {
        let rows = 2;
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
            exec(2),
        )
        .unwrap();
        assert_eq!(LAST_LANES.get(), lanes, "reflector cols={cols}");
    }
    // The triangular solve reports half the solved work and follows the same gate.
    let flags = TriangularSolveFlags {
        left_side: true,
        lower: true,
        transpose_a: false,
        unit_diagonal: false,
    };
    for (width, lanes) in [(1usize, 1usize), (1000, 2)] {
        let dims = [n, width, batch];
        let strides = [1, n as isize, (n * width) as isize];
        let mut x = Vec::new();
        triangular_solve(
            Op::TriangularSolve,
            RawStridedRef::new(&a, &adims, &astrides, 0).unwrap(),
            RawStridedRef::new(&vec![3.0; n * width * batch], &dims, &strides, 0).unwrap(),
            flags,
            &mut x,
            exec(2),
        )
        .unwrap();
        assert_eq!(LAST_LANES.get(), lanes, "triangular width={width}");
    }
}

/// The packed routes keep their own work estimate (`n^3/3` factor, `n^2 k` prepared solve,
/// factor + solve) instead of the old dimension exception.
#[test]
fn packed_routes_use_their_own_work_estimate() {
    use crate::packed_lu::{factor, factor_solve, solve_prepared};
    let pool = owned_pool(4);
    let policy = BatchPolicy::new(BatchStrategy::Auto)
        .with_model(WorkModel::default().with_min_work_per_lane_ns(10.0));
    let exec = token(&pool, 2).with_batch_policy(policy);
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
