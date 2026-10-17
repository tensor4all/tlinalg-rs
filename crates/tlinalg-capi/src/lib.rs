//! `libtlinalg_capi`: the caller-owned-output C ABI of the tlinalg kernels, built as one shared
//! library (`cdylib`), one static library and this `rlib`.
//!
//! The provider ([`tlinalg`]) allocates its outputs: its Rust entry points take `&mut Vec<T>` and
//! fill them. That cannot cross an FFI boundary safely — the C side must not free Rust-allocated
//! memory, and a host that already owns buffers (a pool, DLPack, a Julia array) wants the kernels
//! to write into them. This crate therefore exposes a **two-phase** ABI, the LAPACK shape:
//!
//! 1. `tlinalg_<op>_sizes` fills, per batch item, the element count of every output and the
//!    workspace byte count;
//! 2. `tlinalg_<op>_into` writes compact column-major items into caller-owned buffers, which the
//!    caller passes with their element capacities.
//!
//! A buffer shorter than the query reports is rejected before anything is written. Every entry
//! point catches panics through `status::ffi`; no unwind crosses the C boundary.
//!
//! # Inputs
//!
//! An input is a borrowed, dtype-erased [`tlinalg_strided`] descriptor (rank `2 + B`, extents and
//! strides in elements, `[rows, cols, b_1, ..., b_B]`, first batch axis fastest, plus an element
//! offset). The caller passes the element pointer, the buffer's element count and a [`TlinalgDtype`]
//! tag separately. The descriptor must not reach outside the buffer: a stride or offset that would
//! is rejected with [`TLINALG_ERR_SHAPE`].
//!
//! # Conventions the header carries
//!
//! * SVD returns `U`, non-increasing singular values and `Vᴴ` (not `V`); the `full` flag selects
//!   square unitary factors.
//! * `eigh` and `cholesky` read the **lower** triangle; `eigh` returns non-decreasing **real**
//!   eigenvalues.
//! * `svd` and `eigh` write their values in the scalar type (a complex scalar with a zero imaginary
//!   part for the complex dtypes); `svd_values` and `eigh_values` write the real scalar.
//! * `lu` and `full_piv_lu` do not fail on an exactly singular input; `cholesky` reports a
//!   non-positive-definite item as [`TLINALG_ERR_NONCONVERGENCE`].
//!
//! # Threads
//!
//! `exec` is a TAPP executor handle from tcontract-capi (`<tapp.h>`), the same
//! vocabulary the contraction side of this stack uses: a host that owns one pool
//! creates it with `tcontract_pool_create_rayon`, lends it with
//! `tcontract_tapp_executor_borrow_pool`, and every call here runs on it with the
//! executor's budget. Zero means a sequential call. This crate never creates a
//! pool and never infers a width from the environment; only this C-API glue
//! depends on tcontract-rs, never `tlinalg` itself.
//!
//! # Example
//!
//! ```
//! use std::ffi::c_void;
//! use tlinalg_capi::*;
//!
//! // A = [[3, 0], [0, 1]], column-major.
//! let a = [3.0_f64, 0.0, 0.0, 1.0];
//! let extents = [2i64, 2];
//! let strides = [1i64, 2];
//! let desc = tlinalg_strided {
//!     rank: 2,
//!     extents: extents.as_ptr(),
//!     strides: strides.as_ptr(),
//!     offset: 0,
//! };
//! let mut sizes = tlinalg_svd_out {
//!     batch: 0,
//!     u: 0,
//!     s: 0,
//!     vt: 0,
//!     workspace: 0,
//! };
//! unsafe {
//!     assert_eq!(tlinalg_svd_sizes(&desc, 0, &mut sizes), TLINALG_OK);
//!     assert_eq!((sizes.batch, sizes.u, sizes.s, sizes.vt), (1, 4, 2, 4));
//!     let (mut u, mut s, mut vt) = ([0.0f64; 4], [0.0f64; 2], [0.0f64; 4]);
//!     let status = tlinalg_svd_into(
//!         &desc,
//!         TLINALG_F64,
//!         0, // a sequential call
//!         a.as_ptr().cast::<c_void>(),
//!         a.len(),
//!         0,
//!         u.as_mut_ptr().cast::<c_void>(),
//!         u.len(),
//!         s.as_mut_ptr().cast::<c_void>(),
//!         s.len(),
//!         vt.as_mut_ptr().cast::<c_void>(),
//!         vt.len(),
//!         std::ptr::null_mut(),
//!         0,
//!     );
//!     assert_eq!(status, TLINALG_OK);
//!     assert!((s[0] - 3.0).abs() < 1e-12 && (s[1] - 1.0).abs() < 1e-12);
//! }
//! ```

#![warn(missing_docs)]

pub mod cholesky;
pub mod descriptor;
pub mod eigh;
mod executor;
pub mod full_piv_lu;
pub mod lu;
pub mod sizes;
pub mod status;
pub mod svd;

mod output;

pub use cholesky::{tlinalg_cholesky_into, tlinalg_cholesky_sizes};
pub use descriptor::{
    tlinalg_dtype, tlinalg_strided, TlinalgDtype, TLINALG_BF16, TLINALG_C32, TLINALG_C64,
    TLINALG_F16, TLINALG_F32, TLINALG_F64,
};
pub use eigh::{
    tlinalg_eigh_into, tlinalg_eigh_sizes, tlinalg_eigh_values_into, tlinalg_eigh_values_sizes,
};
pub use full_piv_lu::{tlinalg_full_piv_lu_into, tlinalg_full_piv_lu_sizes};
pub use lu::{tlinalg_lu_into, tlinalg_lu_sizes};
pub use sizes::{
    tlinalg_cholesky_out, tlinalg_eigh_out, tlinalg_eigh_values_out, tlinalg_full_piv_lu_out,
    tlinalg_lu_out, tlinalg_svd_out, tlinalg_svd_values_out,
};
pub use status::{
    describe as tlinalg_describe_status, tlinalg_explain_error, tlinalg_last_error, tlinalg_status,
    FfiError, TLINALG_ERR_INTERNAL, TLINALG_ERR_INVALID_ARGUMENT, TLINALG_ERR_NONCONVERGENCE,
    TLINALG_ERR_NONFINITE, TLINALG_ERR_SHAPE, TLINALG_ERR_SINGULAR, TLINALG_ERR_UNSUPPORTED,
    TLINALG_OK,
};
pub use svd::{
    tlinalg_svd_into, tlinalg_svd_sizes, tlinalg_svd_values_into, tlinalg_svd_values_sizes,
};

/// ABI version: major * 10000 + minor * 100 + patch. `300` is 0.3.0.
pub const ABI_VERSION: u32 = 300;

/// The ABI version of the loaded library (`TLINALG_ABI_VERSION` in the header). Compare it with
/// the header's macro to detect a header and a library from different versions.
#[no_mangle]
pub extern "C" fn tlinalg_abi_version() -> u32 {
    ABI_VERSION
}
