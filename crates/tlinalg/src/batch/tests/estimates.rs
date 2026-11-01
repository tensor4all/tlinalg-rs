//! Compare public estimates with every actual input to the planner, not a copy of its formula.

use core::cell::Cell;
use core::mem::MaybeUninit;

use num_complex::Complex64;
use strided_view::RawStridedRef;
use tcontract_exec::batch::{BatchPolicy, BatchStrategy};
use tcontract_exec::Exec;

use crate::batch::Work;
use crate::householder::ReflectorShape;
use crate::triangular_solve::TriangularSolveFlags;
use crate::{
    cholesky, eig, eigh, full_piv_lu, householder, lu, packed_lu, qr, svd, triangular_solve,
};
use crate::{FaerScalar, Op};

// A calling thread owns its window; parallel tests cannot change another test's expectation.
thread_local! {
    static EXPECTED: Cell<Option<(f64, f64, usize)>> = const { Cell::new(None) };
}

pub(crate) fn record_work(work: Work) {
    EXPECTED.with(|cell| {
        if let Some((total, payback, seen)) = cell.get() {
            assert_eq!(work.total.to_bits(), total.to_bits(), "planner total");
            assert_eq!(
                (work.total * work.payback_share).to_bits(),
                payback.to_bits(),
                "planner payback"
            );
            cell.set(Some((total, payback, seen + 1)));
        }
    });
}

fn check_planner(total: f64, payback: f64, query: impl FnOnce()) {
    EXPECTED.with(|cell| cell.set(Some((total, payback, 0))));
    query();
    let (_, _, seen) = EXPECTED.with(|cell| cell.take()).unwrap();
    assert!(seen > 0, "the entry point must actually reach the planner");
}

fn workspace_models<T: FaerScalar>(exec: &Exec<'_>) -> Vec<(u64, u64)> {
    let (m, n, nrhs, batch) = (13, 7, 3, 2);
    let shape = ReflectorShape {
        rows: m,
        a_cols: n,
        cols: nrhs,
        k: n,
    };
    let mut estimates = Vec::new();
    macro_rules! check {
        ($total:expr, $payback:expr, $query:expr) => {{
            let (total, payback) = ($total, $payback);
            check_planner(total, payback, || {
                $query.unwrap();
            });
            estimates.push((total.to_bits(), payback.to_bits()));
        }};
    }
    check!(
        cholesky::estimate_work::<T>(n),
        cholesky::estimate_payback_work::<T>(n),
        cholesky::cholesky_workspace::<T>(n, batch, exec)
    );
    check!(
        eig::estimate_work::<T>(n),
        eig::estimate_payback_work::<T>(n),
        eig::eig_workspace::<T>(n, true, batch, exec)
    );
    check!(
        eig::estimate_work::<T>(n),
        eig::estimate_payback_work::<T>(n),
        eig::eig_values_workspace::<T>(n, batch, exec)
    );
    check!(
        eigh::estimate_work::<T>(n),
        eigh::estimate_payback_work::<T>(n),
        eigh::eigh_workspace::<T>(n, true, batch, exec)
    );
    check!(
        eigh::estimate_work::<T>(n),
        eigh::estimate_payback_work::<T>(n),
        eigh::eigh_values_workspace::<T>(n, batch, exec)
    );
    check!(
        full_piv_lu::estimate_work::<T>(n),
        full_piv_lu::estimate_payback_work::<T>(n),
        full_piv_lu::full_piv_lu_workspace::<T>(n, batch, exec)
    );
    check!(
        full_piv_lu::estimate_solve_work::<T>(n, nrhs),
        full_piv_lu::estimate_solve_payback_work::<T>(n, nrhs),
        full_piv_lu::full_piv_lu_solve_workspace::<T>(n, nrhs, batch, exec)
    );
    check!(
        householder::estimate_work::<T>(m, n),
        householder::estimate_payback_work::<T>(m, n),
        householder::compact_factor_workspace::<T>(m, n, batch, exec)
    );
    check!(
        householder::estimate_apply_reflectors_work::<T>(shape),
        householder::estimate_apply_reflectors_payback_work::<T>(shape),
        householder::apply_reflectors_workspace::<T>(shape, batch, false, exec)
    );
    check!(
        lu::estimate_work::<T>(m, n),
        lu::estimate_payback_work::<T>(m, n),
        lu::lu_workspace::<T>(m, n, batch, exec)
    );
    check!(
        lu::estimate_solve_work::<T>(n, nrhs),
        lu::estimate_solve_payback_work::<T>(n, nrhs),
        lu::lu_solve_workspace::<T>(n, nrhs, false, batch, exec)
    );
    check!(
        packed_lu::estimate_work::<T>(m, n),
        packed_lu::estimate_payback_work::<T>(m, n),
        packed_lu::factor_workspace::<T>(m, n, batch, exec)
    );
    check!(
        packed_lu::estimate_factor_solve_work::<T>(n, nrhs),
        packed_lu::estimate_factor_solve_payback_work::<T>(n, nrhs),
        packed_lu::factor_solve_workspace::<T>(n, nrhs, batch, exec)
    );
    check!(
        qr::estimate_work::<T>(m, n),
        qr::estimate_payback_work::<T>(m, n),
        qr::qr_workspace::<T>(m, n, batch, exec)
    );
    check!(
        qr::estimate_work::<T>(m, n),
        qr::estimate_payback_work::<T>(m, n),
        qr::rank_revealing_qr_workspace::<T>(m, n, batch, exec)
    );
    for full in [false, true] {
        check!(
            svd::estimate_work::<T>(m, n),
            svd::estimate_payback_work::<T>(m, n),
            svd::svd_workspace::<T>(m, n, full, batch, exec)
        );
    }
    check!(
        svd::estimate_work::<T>(m, n),
        svd::estimate_payback_work::<T>(m, n),
        svd::svd_values_workspace::<T>(m, n, batch, exec)
    );
    estimates
}

// These two routes have constant zero workspace queries, so observe the actual execution instead.
fn scratchless_models(exec: &mut Exec<'_>) -> Vec<(u64, u64)> {
    let matrix = [1.0, 0.0, 0.0, 1.0];
    let dims = [2, 2];
    let strides = [1, 2];
    let a = RawStridedRef::new(&matrix, &dims, &strides, 0).unwrap();
    let rhs = [1.0; 6];
    let mut estimates = Vec::new();
    for left_side in [true, false] {
        let flags = TriangularSolveFlags {
            left_side,
            lower: true,
            transpose_a: false,
            unit_diagonal: false,
        };
        let rhs_dims = if left_side { [2, 3] } else { [3, 2] };
        let rhs_strides = [1, rhs_dims[0] as isize];
        let b = RawStridedRef::new(&rhs, &rhs_dims, &rhs_strides, 0).unwrap();
        let total = triangular_solve::estimate_work::<f64>(2, rhs_dims[0], rhs_dims[1], flags);
        let payback =
            triangular_solve::estimate_payback_work::<f64>(2, rhs_dims[0], rhs_dims[1], flags);
        check_planner(total, payback, || {
            triangular_solve::triangular_solve_into(
                Op::TriangularSolve,
                a,
                b,
                flags,
                &mut [MaybeUninit::uninit(); 6],
                &mut [],
                exec,
            )
            .unwrap();
        });
        estimates.push((total.to_bits(), payback.to_bits()));
    }
    let pivots = [1, 2];
    let pivot_dims = [2];
    let pivot_strides = [1];
    let pivots = RawStridedRef::new(&pivots, &pivot_dims, &pivot_strides, 0).unwrap();
    let total = packed_lu::estimate_solve_prepared_work::<f64>(2, 3);
    let payback = packed_lu::estimate_solve_prepared_payback_work::<f64>(2, 3);
    check_planner(total, payback, || {
        packed_lu::solve_prepared_into(
            Op::LuSolvePrepared,
            a,
            pivots,
            3,
            &mut [1.0; 6],
            false,
            false,
            &mut [],
            exec,
        )
        .unwrap();
    });
    estimates.push((total.to_bits(), payback.to_bits()));
    estimates
}

#[test]
fn estimates_match_planner_totals_at_width_one_and_eight() {
    // A GitHub runner can be smaller than eight threads, and the pool refuses to reserve
    // members it cannot get. That is not a property of the estimates, so skip instead of
    // failing: `payback.rs` takes the same route for the same reason.
    let Ok(team) = threadteam::Pool::builder()
        .threads(8)
        .acquire_timeout(core::time::Duration::from_secs(10))
        .build()
        .unwrap()
        .into_team(8)
    else {
        return;
    };
    let mut baseline = None;
    for strategy in [
        BatchStrategy::Inner,
        BatchStrategy::Sequential,
        BatchStrategy::Auto,
    ] {
        for width in [1, 8] {
            let mut exec = Exec::from_token(team.exec(width), None, BatchPolicy::new(strategy));
            assert_eq!(exec.budget(), width);
            if matches!(strategy, BatchStrategy::Inner) {
                assert_eq!(
                    crate::batch::decide(&exec, 1, Work::all(1.0e8)).item_width(),
                    width
                );
            }
            let mut estimates = workspace_models::<f64>(&exec);
            estimates.extend(workspace_models::<Complex64>(&exec));
            estimates.extend(scratchless_models(&mut exec));
            if let Some(expected) = &baseline {
                assert_eq!(&estimates, expected, "width {width}, strategy {strategy:?}");
            } else {
                baseline = Some(estimates);
            }
        }
    }
}
