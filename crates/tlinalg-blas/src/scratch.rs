//! Scratch acquisition.
//!
//! Copied from the former shared `tlinalg-traits` vocabulary (same project, MIT OR Apache-2.0) and
//! now owned here.
//!
//! Implementations obtain every reusable buffer through [`Workspace`]. The host implements it over
//! its own pool, so the pool's identity, retention policy and steady-state allocation behaviour are
//! preserved and no second allocator is introduced. A few short-lived **index** buffers are still
//! allocated per call rather than pooled, because they are `usize` permutations the `i32`
//! [`IndexWorkspace`] cannot hand out; they are enumerated here so the inventory is complete and
//! `tests/alloc_counts.rs` pins their count:
//!
//! * rank-revealing QR: `tau` (`T`), `jpvt` (`i32`) and the per-item permutation-check `seen`
//!   (`bool`) — three allocations;
//! * partial-pivot LU factor: the `usize` `permutation` — one allocation;
//! * full-pivot LU factor: the `usize` `row_perm` and `col_perm` — two allocations.
//!
//! General-eigendecomposition real-eigenvector assembly no longer adds a per-item staging column:
//! it pushes each output element straight into the caller's reserved vector.
//!
//! # Ownership
//!
//! Acquisition **transfers ownership** of a buffer to the caller, and [`Workspace::release`] returns
//! it. This mirrors the host pool: several buffers may be live at once (a thin SVD holds four), a
//! buffer that becomes a result is simply not released, and a miss allocates. A borrowing "lease"
//! contract would break both the simultaneous holds and the result hand-off.
//!
//! # Initialization
//!
//! Three shapes, and an implementation must pick the one each call site uses today:
//!
//! * [`Workspace::acquire_zeroed`] guarantees every element is zero.
//! * [`Workspace::acquire_uninit`] is the full-overwrite path: the contents are unspecified and
//!   reading an element before writing it is undefined behaviour.
//! * [`Workspace::acquire_capacity`] returns an empty buffer with room for `cap` elements, for call
//!   sites that populate with `push`/`extend` rather than by index.
//!
//! They are not interchangeable: swapping one for another changes either correctness or the
//! zero-fill and allocation counts the host measures.
//!
//! # Index scratch
//!
//! Use [`IndexWorkspace`] for integer scratch such as a LAPACK `iwork`: an index buffer is not a
//! numerical scalar, so it stays out of [`Workspace`].
//!
//! # Concurrency
//!
//! A workspace is **not** used across lanes. Hosts acquire batch and output buffers before fanning
//! out, and per-lane scratch is allocated per lane. `Workspace` therefore has no `Sync` requirement
//! and no concurrent acquisition path.
//!
//! # Abandonment
//!
//! A provider may stop using a buffer without releasing it: on an error return, an early `?`, or a
//! panic. Abandoning is permitted and is not the provider's responsibility to tidy up — the host
//! owns token cancellation and unwind replenishment, exactly as it does for its own buffers today.
//! A provider must **not** release a buffer it abandoned, and must not release on a path where the
//! current implementation drops it instead; doing so would change the host's in-flight accounting.
//! Releasing is required only where a buffer is handed back on a successful path.
//!
//! # Example
//!
//! ```
//! use core::mem::MaybeUninit;
//! use tlinalg_blas::{Scalar, Workspace};
//!
//! struct Owned;
//! impl Workspace<f64> for Owned {
//!     fn acquire_zeroed(&mut self, len: usize) -> Vec<f64> { vec![0.0; len] }
//!     fn acquire_capacity(&mut self, cap: usize) -> Vec<f64> { Vec::with_capacity(cap) }
//!     fn acquire_uninit(&mut self, len: usize) -> Vec<MaybeUninit<f64>> {
//!         (0..len).map(|_| MaybeUninit::uninit()).collect()
//!     }
//!     fn release(&mut self, _buf: Vec<f64>) {}
//! }
//!
//! let mut ws = Owned;
//! assert_eq!(ws.acquire_zeroed(2), [0.0, 0.0]);
//! ```

use core::mem::MaybeUninit;

use num_complex::{Complex32, Complex64};

/// Element types an implementation supports.
///
/// Sealed: only `f32`, `f64`, `Complex32` and `Complex64` implement it. Complex values are
/// `num_complex::Complex`.
pub trait Scalar: crate::sealed::Sealed + Copy + Send + Sync + 'static {}

impl crate::sealed::Sealed for f32 {}
impl crate::sealed::Sealed for f64 {}
impl crate::sealed::Sealed for Complex32 {}
impl crate::sealed::Sealed for Complex64 {}

impl Scalar for f32 {}
impl Scalar for f64 {}
impl Scalar for Complex32 {}
impl Scalar for Complex64 {}

/// Scratch and buffer acquisition for one scalar type.
///
/// Implemented by the host (per scalar) over its buffer pool. See the module documentation for the
/// ownership, initialization and concurrency contracts.
pub trait Workspace<T: Scalar> {
    /// Acquire an owned buffer of `len` elements, every element zeroed.
    ///
    /// # Contract
    ///
    /// Reading any element before writing it yields zero.
    fn acquire_zeroed(&mut self, len: usize) -> Vec<T>;

    /// Acquire an owned, **empty** buffer with room for at least `cap` elements.
    ///
    /// # Contract
    ///
    /// The returned vector has length zero, so population is by `push`/`extend` and needs no unsafe
    /// code. The host may hand back a recycled allocation, so the capacity is a lower bound, not an
    /// exact size. This matches the host's capacity-acquisition operation, and a buffer obtained
    /// this way is released with [`Workspace::release`] like any other.
    fn acquire_capacity(&mut self, cap: usize) -> Vec<T>;

    /// Acquire an owned buffer of `len` elements whose contents are unspecified.
    ///
    /// # Contract
    ///
    /// The caller must write every element before reading any, and must [`Workspace::release`] it or
    /// transfer it into a result. Reading an unwritten element is undefined behaviour.
    fn acquire_uninit(&mut self, len: usize) -> Vec<MaybeUninit<T>>;

    /// Return a buffer to the host's pool.
    ///
    /// # Contract
    ///
    /// `buf` was obtained from this workspace. A buffer that becomes a result is **not** released.
    fn release(&mut self, buf: Vec<T>);
}

/// Scratch for integer work buffers a provider interface requires.
///
/// An index buffer is not a numerical scalar, so it is not a [`Scalar`] and does not belong in
/// [`Workspace`]. Implemented by the host over the same pool, so integer scratch keeps the host's
/// retention and accounting.
///
/// # Example
///
/// ```
/// use tlinalg_blas::IndexWorkspace;
///
/// struct Owned;
/// impl IndexWorkspace for Owned {
///     fn acquire_zeroed_index(&mut self, len: usize) -> Vec<i32> { vec![0; len] }
///     fn release_index(&mut self, _buf: Vec<i32>) {}
/// }
///
/// let mut ws = Owned;
/// assert_eq!(ws.acquire_zeroed_index(2), [0, 0]);
/// ```
pub trait IndexWorkspace {
    /// Acquire an owned index buffer of `len` elements, every element zero.
    ///
    /// Zeroed because the provider interfaces this serves acquire integer scratch zeroed today, and
    /// this returns initialized `Vec<i32>` storage without an unsafe step.
    fn acquire_zeroed_index(&mut self, len: usize) -> Vec<i32>;

    /// Return an index buffer to the host's pool.
    ///
    /// # Contract
    ///
    /// `buf` was obtained from this workspace.
    fn release_index(&mut self, buf: Vec<i32>);
}
