//! Overhead smoke: a seconds-not-minutes timing layer next to `kernels.rs`.
//!
//! The allocation gates catch allocation regressions; some regressions allocate nothing (a
//! per-item dispatch or scratch-reset bug). This compares two paths that should cost about the same
//! on the same inputs and gates on the ratio measured in the same run, never on absolute time.
//! Thresholds start loose (tlinalg-rs#57); the per-case ns/op and allocs/op are reported either way.
//!
//! `OVERHEAD_GATE=1 cargo bench --bench overhead_smoke` gates the L1 and L2 cases; without the
//! variable it reports only, so a noisy runner cannot fail a PR. Single thread only
//! (`RAYON_NUM_THREADS=1`, BLAS threads one).

use core::mem::MaybeUninit;
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

/// A batch of `batch` identical, well-conditioned `n x n` matrices.
fn well_conditioned(n: usize, batch: usize) -> Vec<f64> {
    let mut data = Vec::with_capacity(n * n * batch);
    for _ in 0..batch {
        for col in 0..n {
            for row in 0..n {
                data.push(if row == col {
                    4.0 + row as f64
                } else {
                    0.05 * (((row + 1) * (col + 1)) % 3) as f64
                });
            }
        }
    }
    data
}

/// L1: the borrowed-workspace `*_into` call against the `Vec` wrapper with warm (reserved)
/// outputs. The wrapper's only remaining cost is one scratch allocation of `*_workspace` bytes, so
/// the two paths should be within a small constant of each other. Guards tlinalg-rs#57's wrapper
/// contract: the wrapper must not fall back to per-lane scratch.
fn l1(gate: bool) {
    let batch = 64;
    let seq = &mut Exec::serial();
    for n in [2usize, 8] {
        let a = well_conditioned(n, batch);
        let dims = [n, n, batch];
        let strides = [1isize, n as isize, (n * n) as isize];
        let view = RawStridedRef::new(&a, &dims, &strides, 0).unwrap();

        // qr: `_into` with a workspace from the query, then the wrapper with warm outputs.
        let ws = tlinalg::qr::qr_workspace::<f64>(n, n, batch, seq).unwrap();
        let mut workspace = vec![MaybeUninit::<u8>::uninit(); ws];
        let (mut q, mut r) = (
            vec![MaybeUninit::<f64>::uninit(); n * n * batch],
            vec![MaybeUninit::<f64>::uninit(); n * n * batch],
        );
        let into_ns = per_matrix(batch, 64, || {
            tlinalg::qr::qr_into(Op::Qr, view, &mut q, &mut r, &mut workspace, seq).unwrap();
            black_box((&q, &r));
        });
        let (mut wq, mut wr) = (Vec::new(), Vec::new());
        // Warm the wrapper's output capacity (`reserve` is the only cold allocation left).
        tlinalg::qr::qr(Op::Qr, view, &mut wq, &mut wr, seq).unwrap();
        let wrapper_ns = per_matrix(batch, 64, || {
            tlinalg::qr::qr(Op::Qr, view, &mut wq, &mut wr, seq).unwrap();
            black_box((&wq, &wr));
        });
        let ratio = wrapper_ns / into_ns;
        println!("L1 qr n={n}: into {into_ns:.1} / wrapper {wrapper_ns:.1} ns/matrix = {ratio:.3}");
        // Guards tlinalg-rs#57: wrapper overhead once the `_into` API exists.
        if gate {
            assert!(
                ratio <= 1.3,
                "L1 qr n={n}: wrapper/into ratio {ratio:.3} exceeds 1.3"
            );
        }

        // eigh: the values-and-vectors route.
        let ws = tlinalg::eigh::eigh_workspace::<f64>(n, true, batch, seq).unwrap();
        let mut workspace = vec![MaybeUninit::<u8>::uninit(); ws];
        let (mut values, mut vectors) = (
            vec![MaybeUninit::<f64>::uninit(); n * batch],
            vec![MaybeUninit::<f64>::uninit(); n * n * batch],
        );
        let into_ns = per_matrix(batch, 64, || {
            tlinalg::eigh::eigh_into(
                Op::Eigh,
                view,
                &mut values,
                &mut vectors,
                &mut workspace,
                seq,
            )
            .unwrap();
            black_box((&values, &vectors));
        });
        let (mut wvalues, mut wvectors) = (Vec::new(), Vec::new());
        tlinalg::eigh::eigh(Op::Eigh, view, &mut wvalues, &mut wvectors, seq).unwrap();
        let wrapper_ns = per_matrix(batch, 64, || {
            tlinalg::eigh::eigh(Op::Eigh, view, &mut wvalues, &mut wvectors, seq).unwrap();
            black_box((&wvalues, &wvectors));
        });
        let ratio = wrapper_ns / into_ns;
        println!(
            "L1 eigh n={n}: into {into_ns:.1} / wrapper {wrapper_ns:.1} ns/matrix = {ratio:.3}"
        );
        if gate {
            assert!(
                ratio <= 1.3,
                "L1 eigh n={n}: wrapper/into ratio {ratio:.3} exceeds 1.3"
            );
        }
    }
}

/// The L2 case: a batch of 64 identical n=6 QR factorizations against 64 separate batch-1 calls of
/// the *same* matrix. Both run on `Exec::serial()`. A batch call builds its lane scratch once; the
/// loop rebuilds it per call, so the loop must not become cheaper than the batch path.
fn l2(gate: bool) {
    let n = 6;
    let batch = 64;
    let a = well_conditioned(n, batch);
    let single = well_conditioned(n, 1);
    let dims = [n, n, batch];
    let strides = [1, n as isize, (n * n) as isize];
    let view = RawStridedRef::new(&a, &dims, &strides, 0).unwrap();
    let single_dims = [n, n];
    let single_strides = [1, n as isize];
    let single = RawStridedRef::new(&single, &single_dims, &single_strides, 0).unwrap();
    let (mut q, mut r) = (Vec::new(), Vec::new());
    let seq = &mut Exec::serial();

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

/// L3, report only: the faer provider against the pooled LAPACK provider, QR and eigh, n = 8 and
/// 64. The LAPACK rows need the vendor library linked (`--features link-openblas`).
#[cfg(feature = "link-openblas")]
fn l3() {
    use tlinalg_bench::RecyclingWorkspace;

    let batch = 64;
    for n in [8usize, 64] {
        let a = well_conditioned(n, batch);
        let dims = [n, n, batch];
        let strides = [1isize, n as isize, (n * n) as isize];
        let view = RawStridedRef::new(&a, &dims, &strides, 0).unwrap();
        let (mut q, mut r) = (Vec::new(), Vec::new());
        let faer_ns = per_matrix(batch, 32, || {
            tlinalg::qr::qr(Op::Qr, view, &mut q, &mut r, &mut Exec::serial()).unwrap();
        });
        let mut workspace = RecyclingWorkspace::default();
        let (mut q, mut r) = (Vec::new(), Vec::new());
        let lapack_ns = per_matrix(batch, 32, || {
            tlinalg_blas::qr::qr(tlinalg_blas::Op::Qr, view, &mut q, &mut r, &mut workspace)
                .unwrap();
        });
        println!("L3 qr n={n}: faer {faer_ns:.1} / lapack {lapack_ns:.1} ns/matrix");

        let (mut values, mut vectors) = (Vec::new(), Vec::new());
        let faer_ns = per_matrix(batch, 32, || {
            tlinalg::eigh::eigh(
                Op::Eigh,
                view,
                &mut values,
                &mut vectors,
                &mut Exec::serial(),
            )
            .unwrap();
        });
        let mut workspace = RecyclingWorkspace::default();
        let (mut values, mut vectors) = (Vec::new(), Vec::new());
        let lapack_ns = per_matrix(batch, 32, || {
            tlinalg_blas::eigh::eigh(
                tlinalg_blas::Op::Eigh,
                view,
                &mut values,
                Some(&mut vectors),
                &mut workspace,
            )
            .unwrap();
        });
        println!("L3 eigh n={n}: faer {faer_ns:.1} / lapack {lapack_ns:.1} ns/matrix");
    }
}

/// L3 is report-only; without the vendor library there is no second provider to report.
#[cfg(not(feature = "link-openblas"))]
fn l3() {
    println!("L3 skipped: build with --features link-openblas for the LAPACK comparison");
}

/// L4, report only: a 2-worker pool against the serial path, batch 64 n=6. The pool-entry constant is
/// not a gate here.
fn l4() {
    let n = 6;
    let batch = 64;
    let a = well_conditioned(n, batch);
    let dims = [n, n, batch];
    let strides = [1, n as isize, (n * n) as isize];
    let view = RawStridedRef::new(&a, &dims, &strides, 0).unwrap();
    let (mut q, mut r) = (Vec::new(), Vec::new());
    let serial_ns = per_matrix(batch, 64, || {
        tlinalg::qr::qr(Op::Qr, view, &mut q, &mut r, &mut Exec::serial()).unwrap();
    });
    let team = threadteam::Pool::builder()
        .threads(2)
        .build()
        .unwrap()
        .into_team(2)
        .expect("a 2T team");
    let mut exec = Exec::from_token(team.exec(2), None, tcontract_exec::BatchPolicy::default());
    let pool_ns = per_matrix(batch, 64, || {
        tlinalg::qr::qr(Op::Qr, view, &mut q, &mut r, &mut exec).unwrap();
    });
    println!("L4 qr serial vs 2T pool, batch64 n={n}: {serial_ns:.1} / {pool_ns:.1} ns/matrix");
    black_box((serial_ns, pool_ns));
}

fn main() {
    let gate = std::env::var("OVERHEAD_GATE").as_deref() == Ok("1");
    l1(gate);
    l2(gate);
    l3();
    l4();
    println!("overhead smoke done (gate={gate})");
}
