//! Placement through the token, measured with the `strided-exec` ledger: whatever the batch size
//! relative to the budget, no more tasks than the budget are ever live, the result matches the
//! serial one, and the two width regimes (a batch with at least `budget` items runs `Par::Seq`
//! items; a smaller batch gives each item its share) both stay inside the bound.

mod common;

use common::matrix;
use strided_exec::ledger;
use strided_view::RawStridedRef;
use tcontract_exec::{Exec, Pool};
use tlinalg::Op;
use tlinalg_testkit::hpd_seeded;

const BUDGET: usize = 4;

fn pool() -> Pool<'static> {
    Pool::owned(
        strided_exec::Pool::builder()
            .threads(BUDGET)
            .build()
            .unwrap(),
    )
}

fn stack(items: &[Vec<f64>]) -> Vec<f64> {
    items.iter().flatten().copied().collect()
}

fn close(got: &[f64], want: &[f64], what: &str) {
    assert_eq!(got.len(), want.len(), "{what}");
    for (g, w) in got.iter().zip(want) {
        assert!(
            (g - w).abs() <= 1e-10 * (1.0 + w.abs()),
            "{what}: {g} vs {w}"
        );
    }
}

/// Run every family on `batch` items with a token of `BUDGET` slots and compare with the serial
/// result; return the largest number of live tasks the ledger saw.
fn run(batch: usize, n: usize) -> usize {
    let pool = pool();
    let mut exec = Exec::new(&pool, BUDGET).unwrap();
    let root = exec.token().ledger_root().unwrap();
    let dims = [n, n, batch];
    let strides = [1isize, n as isize, (n * n) as isize];

    let general = stack(
        &(0..batch)
            .map(|i| matrix::<f64>(n, n, i))
            .collect::<Vec<_>>(),
    );
    let spd = stack(
        &(0..batch)
            .map(|i| hpd_seeded::<f64>(n, i))
            .collect::<Vec<_>>(),
    );
    let general = RawStridedRef::new(&general, &dims, &strides, 0).unwrap();
    let spd = RawStridedRef::new(&spd, &dims, &strides, 0).unwrap();

    // QR
    let (mut q, mut r) = (Vec::new(), Vec::new());
    let (mut sq, mut sr) = (Vec::new(), Vec::new());
    tlinalg::qr::qr(Op::Qr, general, &mut q, &mut r, &mut exec).unwrap();
    tlinalg::qr::qr(Op::Qr, general, &mut sq, &mut sr, &mut Exec::serial()).unwrap();
    close(&q, &sq, "qr Q");
    close(&r, &sr, "qr R");

    // singular values
    let (mut s, mut ss) = (Vec::new(), Vec::new());
    tlinalg::svd::svd_values(Op::SvdValues, general, &mut s, &mut exec).unwrap();
    tlinalg::svd::svd_values(Op::SvdValues, general, &mut ss, &mut Exec::serial()).unwrap();
    close(&s, &ss, "svd values");

    // Hermitian eigenvalues
    let (mut w, mut sw) = (Vec::new(), Vec::new());
    tlinalg::eigh::eigh_values(Op::EighValues, spd, &mut w, &mut exec).unwrap();
    tlinalg::eigh::eigh_values(Op::EighValues, spd, &mut sw, &mut Exec::serial()).unwrap();
    close(&w, &sw, "eigh values");

    // Cholesky
    let (mut l, mut sl) = (Vec::new(), Vec::new());
    tlinalg::cholesky::cholesky(Op::Cholesky, spd, &mut l, &mut exec).unwrap();
    tlinalg::cholesky::cholesky(Op::Cholesky, spd, &mut sl, &mut Exec::serial()).unwrap();
    close(&l, &sl, "cholesky");

    ledger::max_live(root)
}

/// At least `budget` items: every slot runs a sequential item stream.
#[test]
fn a_batch_at_least_the_budget_stays_within_it() {
    for batch in [BUDGET, BUDGET + 1, 3 * BUDGET] {
        let live = run(batch, 8);
        assert!(
            live <= BUDGET,
            "batch {batch}: {live} live > budget {BUDGET}"
        );
    }
}

/// Fewer items than the budget: the items share the slots, each as faer width.
#[test]
fn a_batch_below_the_budget_stays_within_it() {
    for batch in [1, 2, BUDGET - 1] {
        let live = run(batch, 96);
        assert!(
            live <= BUDGET,
            "batch {batch}: {live} live > budget {BUDGET}"
        );
    }
}
