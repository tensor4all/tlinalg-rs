//! Turning a TAPP executor handle into this crate's parallelism.
//!
//! The handle is tcontract-capi's `TAPP_executor`, the same one a host passes to
//! the contraction side of this stack, so one host pool serves both. Its budget
//! is the executor's, clamped to its pool width there, and the call is held in
//! flight while the closure runs, so the pool cannot be released under it.

use core::num::NonZeroUsize;

use tcontract_capi::executor_from_handle;
use tlinalg::Parallel;

/// Run `body` with the parallelism `exec` denotes.
///
/// Zero (or an executor without a pool) means [`Parallel::Sequential`]: this
/// library never creates a pool and never infers a width from the environment.
pub(crate) fn with_executor<R>(exec: isize, body: impl FnOnce(Parallel<'_>) -> R) -> R {
    // SAFETY: the ABI contract says the handle is zero or a live TAPP_executor.
    match unsafe { executor_from_handle(exec) } {
        Some(executor) => executor.with_pool(|policy| match policy {
            Some((pool, budget)) => body(Parallel::Pool {
                pool,
                budget: NonZeroUsize::new(budget).unwrap_or(NonZeroUsize::MIN),
            }),
            None => body(Parallel::Sequential),
        }),
        None => body(Parallel::Sequential),
    }
}
