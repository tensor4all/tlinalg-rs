//! faer-backed, tensor-free linear-algebra kernels.
//!
//! This crate owns the numerical kernels, the batch loop over them, their scratch, and the small
//! vocabulary its callers use to drive them: borrowed strided I/O, [`tcontract_exec::Exec`] and
//! typed [`Error`]. It does not own tensors, allocation policy, dtype dispatch, placement, or the
//! execution context; a host supplies operands, output vectors and one `Exec` token. Kernel scratch
//! is per lane.
//!
//! The interface a host requires of its linear-algebra providers belongs to that host (tenferro
//! defines it); this crate is one provider and defines only the types its own entry points take.
//! The LAPACK provider (`tlinalg-blas`) is independent of this crate.
//!
//! # Batches
//!
//! Every entry point is batched (`docs/design/batched-api.md` in the repository): an input is a
//! rank-`2 + B` [`strided_view::RawStridedRef`] `[rows, cols, b_1, ..., b_B]`, outputs are compact
//! column-major items in batch order (first batch axis fastest). `Auto` keeps the pre-0.3 local
//! decision in 0.3.0 (an item-extent cutoff); every forced strategy is decided by the context
//! ([`tcontract_exec::Exec::batch_decision`], configured through
//! [`BatchPolicy`](tcontract_exec::BatchPolicy)) on the original budget. A forced
//! `WholeBatchVendor` is refused with [`Error::StrategyUnavailable`] before any write — this
//! provider has no vendor batched call. A failing call returns the error of its lowest-indexed
//! failing item and leaves every library-created output vector empty.
//!
//! # Borrowed workspace
//!
//! Every family has a `*_workspace(<shape args>, batch, exec)` query that returns the number of
//! caller-owned bytes one call needs (worst-case alignment padding included, at the item width the
//! batch decision grants the same call) and a `*_into(..., workspace: &mut [MaybeUninit<u8>], exec)`
//! entry that carves all of its per-lane scratch from that region. The `Vec` entry points are
//! wrappers: they reserve their outputs and allocate one block of the queried size. An undersized
//! workspace is refused with [`Error::InsufficientWorkspace`] before any output is written. The
//! full contract, including the family inventory, is `docs/design/borrowed-workspace.md`.
//!
//! # Conventions
//!
//! Callers and their tests depend on these:
//!
//! | Operation | Convention |
//! |---|---|
//! | `svd` | returns `U`, non-increasing singular values, and `Vᴴ` (not `V`) |
//! | `eigh` | reads the lower triangle, returns non-decreasing **real** values |
//! | `cholesky` | reads the lower triangle |
//! | factorizations | do **not** fail on exactly singular input unless the table below says so |
//! | solves | report [`Error::Singular`] for a singular factor |
//!
//! Failure behaviour of this provider (it is deliberately not normalized against other providers):
//!
//! | Route | Exactly singular input |
//! |---|---|
//! | partial-pivot LU in `solve` | [`Error::Singular`] |
//! | full-pivot LU | not an error |
//!
//! Dtype dispatch and the deliberate "unsupported for this dtype" cases stay in the host: this
//! crate never names a host `DType`. Terminal storage—tensor construction, placement tagging, and
//! the error wrapper that carries an error into the host's error type—also stays in the host.
//!
//! # Parallelism
//!
//! Numerical work is faer's: a call installs the context's pool and passes faer the item thread
//! count the batch decision granted. faer's count is a hint, not a strict active-thread bound (faer
//! can expose more intra-item tasks than requested); the context bounds the outer lanes and the
//! count requested from faer, not every native numerical thread. Admission across simultaneous
//! calls belongs to the resource owner. `Exec::Serial` runs on the calling thread with no
//! implementation-owned fan-out.

#![warn(missing_docs)]

pub mod cholesky;
pub mod eig;
pub mod eigh;
pub mod error;
pub mod full_piv_lu;
pub mod householder;
pub mod lu;
pub mod packed_lu;
pub mod qr;
pub mod scratch;
pub mod svd;
pub mod triangular_solve;

mod batch;
mod scalar;
mod util;
mod workspace;

pub use error::{Error, NonFiniteRole, Op, Result};
pub use scalar::FaerScalar;
pub use scratch::Scalar;

mod sealed {
    /// Seals [`crate::Scalar`].
    pub trait Sealed {}
}

/// Map a [`tcontract_exec::Par`] grant to faer's parallelism token.
///
/// The context has already entered its pool (or kept the caller on width one), so faer runs on the
/// registry the context installed; only the degree differs.
pub(crate) fn faer_par(par: tcontract_exec::Par) -> faer::Par {
    match par {
        tcontract_exec::Par::Seq => faer::Par::Seq,
        tcontract_exec::Par::Threads(n) => faer::Par::rayon(n.get()),
    }
}

#[cfg(test)]
mod tests {
    use super::faer_par;
    use tcontract_exec::{Exec, Pool};

    /// The pool the caller lends is the pool the work runs on: not the ambient one, and not a
    /// pool discovered from the thread the call happens to start on.
    #[test]
    fn the_supplied_pool_is_the_one_installed() {
        let two = rayon::ThreadPoolBuilder::new()
            .num_threads(2)
            .build()
            .expect("a 2-thread pool");
        let three = rayon::ThreadPoolBuilder::new()
            .num_threads(3)
            .build()
            .expect("a 3-thread pool");
        let two_pool = Pool::borrow(&two);
        let three_pool = Pool::borrow(&three);
        let observed_two = Exec::rayon(&two_pool).install(2, |par| {
            assert_eq!(faer_par(par).degree(), 2);
            rayon::current_num_threads()
        });
        let observed_three = Exec::rayon(&three_pool).install(3, |par| {
            assert_eq!(faer_par(par).degree(), 3);
            rayon::current_num_threads()
        });
        assert_eq!(observed_two, 2);
        assert_eq!(observed_three, 3);
    }
}
