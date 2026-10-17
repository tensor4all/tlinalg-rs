//! Turning a TAPP executor handle into this crate's parallelism.
//!
//! The handle is tcontract-capi's `TAPP_executor`, the same one a host passes to
//! the contraction side of this stack, so one host pool serves both. Its 0.3.0
//! `Executor::with` hands out a *guarded* [`Exec`]: the call is held in flight
//! so `tcontract_pool_destroy` cannot release the pool under it, and the pool's
//! retained workspace and SPMD gate stay the executor's. This layer forwards
//! that `Exec`; it does not build a second `Pool` wrapper per call.
//!
//! The ABI bodies return their own [`FfiError`], so this keeps the `FnOnce -> R`
//! shape the entry points already use and runs the body inside `Executor::with`'s
//! guarded closure. The only error `with` can synthesize is a zero budget, which
//! the executor's own clamping makes unreachable; anything else would be a bug,
//! and it is reported as an internal failure.

use tcontract_exec::Exec;

/// Run `body` with the guarded `Exec` the handle denotes.
///
/// Zero (or an executor without a pool) means [`Exec::serial()`]: this library
/// never creates a pool and never infers a width from the environment.
pub(crate) fn with_executor<R>(exec: isize, body: impl FnOnce(Exec<'_>) -> R) -> R {
    // SAFETY: the ABI contract says the handle is zero or a live TAPP_executor.
    match unsafe { tcontract_capi::executor_from_handle(exec) } {
        Some(executor) => {
            let mut slot: Option<R> = None;
            let outcome = executor.with(|exec| {
                slot = Some(body(*exec));
                Ok(())
            });
            match outcome {
                Ok(()) => slot.expect("Executor::with ran its closure"),
                Err(error) => panic!("guarded executor rejected the call: {}", error.message),
            }
        }
        None => body(Exec::serial()),
    }
}
