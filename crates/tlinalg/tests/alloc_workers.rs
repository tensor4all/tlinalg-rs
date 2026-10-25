//! Proves the cross-thread allocation window observes work handed to pool workers.
//!
//! This is its own test binary (one `#[global_allocator]`, one test) so the process-wide counter has
//! no other test allocating concurrently: the count it reports is exactly the worker's allocation.
//! `allocations` (thread-local) cannot see it; `allocations_all_threads` must.

use strided_exec::{Exec, Pool};
use tlinalg_testkit::alloc::{allocations, allocations_all_threads, Counting};

#[global_allocator]
static GLOBAL: Counting = Counting;

#[test]
fn the_cross_thread_counter_sees_a_worker_allocation() {
    let pool = Pool::builder().threads(2).build().unwrap();
    let mut token = Exec::new(&pool, 2).unwrap();
    // Warm the pool so entering it once does not contribute later.
    token.reborrow().external(|_| {});

    let this_thread = allocations(|| {
        token.reborrow().external(|_| {
            let worker: Vec<u8> = Vec::with_capacity(1024);
            std::hint::black_box(&worker);
        });
    });
    let every_thread = allocations_all_threads(|| {
        token.reborrow().external(|_| {
            let worker: Vec<u8> = Vec::with_capacity(1024);
            std::hint::black_box(&worker);
        });
    });

    assert!(
        every_thread > 0,
        "the cross-thread counter missed a deliberate worker allocation"
    );
    // The worker allocation is not on the calling thread, which is why the thread-local counter is
    // not enough for a pooled gate.
    assert_eq!(
        this_thread, 0,
        "the thread-local counter somehow saw the worker allocation"
    );
}
