//! No operation of this crate may start the process-global Rayon pool: serial calls stay on the
//! calling thread, and pooled calls run on the pool the token names.
//!
//! This is its own test binary with a single test, so nothing else in the process can have started
//! the global pool before the final check (`global_pool_unstarted` answers by trying to start it).

mod common;

use common::matrix;
use strided_view::RawStridedRef;
use tcontract_exec::{Exec, Pool};
use threadteam::ledger;
use tlinalg::Op;

#[test]
fn no_call_starts_the_global_pool() {
    let n = 96;
    for batch in [1usize, 3, 8] {
        let data: Vec<f64> = (0..batch).flat_map(|i| matrix::<f64>(n, n, i)).collect();
        let dims = [n, n, batch];
        let strides = [1isize, n as isize, (n * n) as isize];
        let view = RawStridedRef::new(&data, &dims, &strides, 0).unwrap();

        // The serial token never enters a pool.
        let (mut q, mut r) = (Vec::new(), Vec::new());
        tlinalg::qr::qr(Op::Qr, view, &mut q, &mut r, &mut Exec::serial()).unwrap();

        // A pooled token, in both width regimes, runs on its own pool only.
        let pool = Pool::owned(threadteam::Pool::builder().threads(4).build().unwrap());
        let mut exec = Exec::new(&pool, 4).unwrap();
        let root = exec.token().ledger_root().unwrap();
        let (mut pq, mut pr) = (Vec::new(), Vec::new());
        tlinalg::qr::qr(Op::Qr, view, &mut pq, &mut pr, &mut exec).unwrap();
        let mut s = Vec::new();
        tlinalg::svd::svd_values(Op::SvdValues, view, &mut s, &mut exec).unwrap();
        assert!(ledger::max_live(root) <= 4);
    }
    assert!(
        ledger::global_pool_unstarted(),
        "a call started the process-global Rayon pool"
    );
}
