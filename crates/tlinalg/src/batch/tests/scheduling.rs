//! Batch-lane scheduling: the temporary step-1 Auto rule, the exact partition, and forced
//! strategies.
//!
//! tlinalg-rs#58 "Defaults" and tcontract-rs#123 step 1 keep tlinalg's pre-0.3 `Auto` in 0.3.0: fan
//! out to outer lanes only while the largest item extent is at most 64 (packed LU excepted), RHS
//! columns counted by default behind the `TLINALG_AUTO_RHS` switch. The switch to the shared work
//! model lands in #123 step 3 with the sweep as evidence. These tests pin the step-1 decisions on a
//! table of `(items, n, k, budget)`, including RHS-heavy solves and reflector targets, and pin the
//! exact nonempty partition, the forced routes and the typed vendor refusal.

use crate::batch::{auto_extent_with, check_route, out, run};
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

/// The step-1 Auto decision, written out independently of the driver.
fn legacy_auto(batch: usize, budget: usize, max_item_dim: Option<usize>) -> (usize, usize) {
    let budget = budget.max(1);
    let fans_out = budget > 1 && batch >= budget && max_item_dim.is_none_or(|dim| dim <= 64);
    if fans_out {
        (budget.min(batch), 1)
    } else {
        (1, budget)
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
        (1, 4, Some(1), 1, 4),
        (3, 4, Some(1), 1, 4),
        (12, 1, Some(1), 1, 1),
        (12, 2, Some(1), 2, 1),
        (12, 3, Some(1), 3, 1),
        (12, 4, Some(1), 4, 1),
        (16, 99, Some(1), 8, 1),
        (5, 4, Some(1), 4, 1),
        (8, 4, Some(64), 4, 1),
        (8, 4, Some(65), 1, 4),
        (8, 4, None, 4, 1),
    ] {
        let (want_lanes, _) = if batch == 0 {
            (0, 0)
        } else {
            legacy_auto(batch, budget.min(8), dim)
        };
        assert_eq!(lanes, want_lanes, "table row {batch}/{budget}/{dim:?}");
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
            dim,
            1.0,
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
            "batch={batch}, budget={budget}, dim={dim:?}"
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
        run(
            Op::Qr,
            8,
            token(&selected, 2),
            Some(1),
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
            Some(1),
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
            Some(1),
            1.0,
            &mut (out(&mut Vec::<f64>::new(), 1),),
            |_| (),
            |_, _, _, _| Ok(()),
        ),
        Err(crate::Error::StrategyUnavailable { .. })
    ));
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

/// The forced routes are evaluated against the original budget even though `WorkModel` exists.
#[test]
fn forced_strategies_ignore_the_auto_cutoff() {
    let pool = owned_pool(4);
    let policy = BatchPolicy::new(BatchStrategy::Outer)
        .with_model(WorkModel::default().with_min_work_per_lane_ns(10.0));
    let exec = token(&pool, 2).with_batch_policy(policy);
    // n = 65 would take one lane under Auto; `Outer` still clamps to the budget.
    let n = 65;
    let batch = 4;
    let mut lu = vec![0.0; n * n * batch];
    for item in lu.chunks_exact_mut(n * n) {
        for i in 0..n {
            item[i + i * n] = 1.0;
        }
    }
    let mut pivots = vec![0; n * batch];
    let mut parity = vec![0.0; batch];
    crate::packed_lu::factor(Op::LuFactor, n, n, &mut lu, &mut pivots, &mut parity, exec).unwrap();
    assert_eq!(LAST_LANES.get(), 2);
}
