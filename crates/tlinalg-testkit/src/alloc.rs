//! A counting global allocator for steady-state allocation tests.
//!
//! A test binary installs it with
//!
//! ```ignore
//! #[global_allocator]
//! static GLOBAL: tlinalg_testkit::alloc::Counting = tlinalg_testkit::alloc::Counting;
//! ```
//!
//! and measures with [`allocations`] or [`steady`]. Counts are per thread, so concurrently running
//! tests in the same binary do not disturb each other; work a kernel hands to other threads (rayon
//! lanes) is not counted by those two, so measure one sequential lane, or use
//! [`allocations_all_threads`] when the workers themselves are what the gate must see.

use core::cell::Cell;
use core::sync::atomic::{AtomicUsize, Ordering};
use std::alloc::{GlobalAlloc, Layout, System};

/// Forwards to the system allocator and counts `alloc`, `alloc_zeroed` and `realloc` calls made on
/// the current thread, and in a process-wide total.
pub struct Counting;

thread_local! {
    static COUNT: Cell<usize> = const { Cell::new(0) };
}

/// Process-wide allocation count, used by [`allocations_all_threads`].
static ALL: AtomicUsize = AtomicUsize::new(0);

fn bump() {
    // `try_with` because the allocator can run during thread-local teardown.
    let _ = COUNT.try_with(|count| count.set(count.get() + 1));
    ALL.fetch_add(1, Ordering::Relaxed);
}

// SAFETY: every method forwards to the system allocator with the caller's arguments unchanged; the
// counter only observes.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        bump();
        // SAFETY: forwarded with the caller's layout.
        unsafe { System.alloc(layout) }
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        bump();
        // SAFETY: forwarded with the caller's layout.
        unsafe { System.alloc_zeroed(layout) }
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        bump();
        // SAFETY: forwarded unchanged.
        unsafe { System.realloc(ptr, layout, new_size) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: forwarded with the caller's pointer and layout.
        unsafe { System.dealloc(ptr, layout) }
    }
}

/// Allocations `f` makes on the current thread (meaningful only with [`Counting`] installed).
pub fn allocations(f: impl FnOnce()) -> usize {
    let before = COUNT.with(Cell::get);
    f();
    COUNT.with(Cell::get) - before
}

/// Allocations made by the second of two identical calls: the first warms up any reused buffers.
pub fn steady(mut call: impl FnMut()) -> usize {
    call();
    allocations(call)
}

/// Allocations on **every** thread during `f`, from a process-wide counter.
///
/// [`allocations`] observes only the calling thread, so a gate that runs a kernel on a pool can pass
/// while its workers allocate once per item. This window also sees worker allocations. The counter
/// is process-wide, so a test that asserts an exact number with it must run alone (no other test in
/// the binary allocating concurrently); a test that only asserts "workers were observed" is robust to
/// such noise. Counts are monotonic, so the difference cannot underflow unless the process wraps.
pub fn allocations_all_threads(f: impl FnOnce()) -> usize {
    let before = ALL.load(Ordering::Relaxed);
    f();
    ALL.load(Ordering::Relaxed) - before
}
