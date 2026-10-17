//! Overhead smoke: a seconds-not-minutes timing layer next to `kernels.rs`.
//!
//! The allocation gates catch allocation regressions; some regressions allocate nothing (a
//! per-item dispatch or scratch-reset bug). This compares two paths that should cost about the same
//! on the same inputs and gates on the ratio measured in the same run, never on absolute time.
//! Thresholds start loose (tlinalg-rs#57); the per-case ns/op and allocs/op are reported either way.
//!
//! `OVERHEAD_GATE=1 cargo bench --bench overhead_smoke` gates the L2 case; without the variable it
//! reports only, so a noisy runner cannot fail a PR. Single thread only
//! (`RAYON_NUM_THREADS=1`, BLAS threads one).

use std::hint::black_box;
use std::time::Instant;

use strided_view::RawStridedRef;
use tcontract_exec::Exec;
use tlinalg::Op;
use tlinalg_testkit::alloc::{allocations_all_threads, Counting};

#[global_allocator]
static GLOBAL: Counting = Counting;

/// Five rounds of `iters`; the reported cost is the fastest round (minimum), in nanoseconds per
/// matrix.
fn per_matrix<F: FnMut()>(matrices: usize, iters: usize, mut call: F) -> f64 {
    let mut best = f64::INFINITY;
    for _ in 0..5 {
        let start = Instant::now();
        for _ in 0..iters {
            call();
        }
        let elapsed = start.elapsed().as_nanos() as f64;
        best = best.min(elapsed / (iters * matrices) as f64);
    }
    best
}

/// The L2 case: a batch of 64 identical n=6 QR factorizations against 64 separate batch-1 calls of
/// the *same* matrix. Both run on `Exec::serial()`. A batch call builds its lane scratch once; the
/// loop rebuilds it per call, so the loop must not become cheaper than the batch path.
fn l2(gate: bool) {
    let n = 6;
    let batch = 64;
    let one: Vec<f64> = (0..n * n)
        .map(|i| if i % (n + 1) == 0 { 4.0 } else { 0.1 })
        .collect();
    let mut a = Vec::with_capacity(n * n * batch);
    for _ in 0..batch {
        a.extend_from_slice(&one);
    }
    let dims = [n, n, batch];
    let strides = [1, n as isize, (n * n) as isize];
    let view = RawStridedRef::new(&a, &dims, &strides, 0).unwrap();
    let single_dims = [n, n];
    let single_strides = [1, n as isize];
    let single = RawStridedRef::new(&one, &single_dims, &single_strides, 0).unwrap();
    let (mut q, mut r) = (Vec::new(), Vec::new());
    let seq = Exec::serial();

    let batch_ns = per_matrix(batch, 64, || {
        tlinalg::qr::qr(Op::Qr, view, &mut q, &mut r, seq).unwrap();
        black_box((q.as_slice(), r.as_slice()));
    });
    let loop_ns = per_matrix(batch, 64, || {
        for _ in 0..batch {
            tlinalg::qr::qr(Op::Qr, single, &mut q, &mut r, seq).unwrap();
            black_box((q.as_slice(), r.as_slice()));
        }
    });
    let allocs = allocations_all_threads(|| {
        tlinalg::qr::qr(Op::Qr, view, &mut q, &mut r, seq).unwrap();
    });
    let ratio = batch_ns / loop_ns;
    println!(
        "L2 qr batch64 vs 64x batch1 n={n}: {batch_ns:.1} / {loop_ns:.1} ns/matrix = {ratio:.3} ({allocs} allocs)"
    );
    // Guards tlinalg-rs#57 item 7: per-call scratch in the faer lane.
    if gate {
        assert!(
            ratio <= 1.2,
            "L2 qr batch/loop per-matrix ratio {ratio:.3} exceeds 1.2"
        );
    }
}

/// L4, report only: a 2-worker pool against the serial path, batch 64 n=6. The rayon constant is
/// not a gate here.
fn l4() {
    let n = 6;
    let batch = 64;
    let a: Vec<f64> = (0..n * n * batch)
        .map(|i| if i % (n + 1) == 0 { 4.0 } else { 0.1 })
        .collect();
    let dims = [n, n, batch];
    let strides = [1, n as isize, (n * n) as isize];
    let view = RawStridedRef::new(&a, &dims, &strides, 0).unwrap();
    let (mut q, mut r) = (Vec::new(), Vec::new());
    let serial_ns = per_matrix(batch, 64, || {
        tlinalg::qr::qr(Op::Qr, view, &mut q, &mut r, Exec::serial()).unwrap();
    });
    let tp = rayon::ThreadPoolBuilder::new()
        .num_threads(2)
        .build()
        .unwrap();
    let pool = tcontract_exec::Pool::owned(tp);
    let exec = Exec::rayon(&pool).with_budget(2).unwrap();
    let pool_ns = per_matrix(batch, 64, || {
        tlinalg::qr::qr(Op::Qr, view, &mut q, &mut r, exec).unwrap();
    });
    println!("L4 qr serial vs 2T pool, batch64 n={n}: {serial_ns:.1} / {pool_ns:.1} ns/matrix");
    black_box((serial_ns, pool_ns));
}

fn main() {
    let gate = std::env::var("OVERHEAD_GATE").as_deref() == Ok("1");
    // L1 (workspace `_into` vs the `Vec` wrapper) is not run: the borrowed-workspace `_into` API is
    // not part of this change, so there is no second path to compare.
    l2(gate);
    l4();
    println!("overhead smoke done (gate={gate})");
}
