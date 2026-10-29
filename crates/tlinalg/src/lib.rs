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
//! column-major items in batch order (first batch axis fastest). The lane decision is the shared
//! [`tcontract_exec::Exec::batch_decision`], configured through
//! [`BatchPolicy`](tcontract_exec::BatchPolicy); the lanes are forked over the call's token (see
//! "Parallelism"). A failing call returns the error of its lowest-indexed failing item and leaves
//! every library-created output vector empty.
//!
//! # Borrowed workspace
//!
//! Every family has a `*_workspace(<shape args>, batch, exec)` query that returns the number of
//! caller-owned bytes one call needs (worst-case alignment padding included, at the widest lane share
//! the same call grants) and a `*_into(..., workspace: &mut [MaybeUninit<u8>], exec)`
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
//! Every entry point takes `&mut` [`tcontract_exec::Exec`], which bundles one
//! [`threadteam::Exec`] token (a pool reference and a budget of slots), the workspace and the
//! batch policy. The token is the only budget and the only way work starts: the batch is split
//! into lanes with `fork`, which divides the budget among the lanes, and each lane hands its share
//! to faer as `Par::Team` when the token carries a team (`threadteam::Exec::carries_team`) and the
//! share is more than one slot, and runs faer as `Par::Seq` otherwise
//! (`tcontract_exec::with_faer_par`): only a team gives faer internal parallelism, while batch
//! lanes fork on any token. Whether a batch fans out is the token's rule
//! (`Exec::width_for`: serial time against the dispatch cost of the token's kind). A batch with at least `budget` items
//! therefore runs one sequential item stream per slot (`Par::Seq` items); a smaller batch gives
//! each item its share as a team (on a team token). A serial token runs on the calling thread and enters no
//! pool; nothing falls back to an ambient pool. The token bounds the lanes it starts, not faer's
//! internal tasks: how faer schedules a width is faer's. Admission across simultaneous calls
//! belongs to the resource owner.

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

/// The shape of faer's parallelism for a team of `width` participants, for the workspace queries
/// (which depend on the degree, not on how the members are reached). Width one, and every token
/// without a team, is `Seq`: only a team gives faer internal parallelism.
pub(crate) fn faer_shape(width: usize) -> faer::ParShape {
    if width <= 1 {
        faer::ParShape::Seq
    } else {
        faer::ParShape::team(width)
    }
}
