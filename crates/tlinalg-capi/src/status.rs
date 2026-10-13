//! Status codes, the thread-local last-error message, and the panic-catching
//! wrapper every `extern "C"` entry point uses.
//!
//! The status codes mirror the provider's typed errors ([`tlinalg::Error`]) so a
//! host can map one back without parsing text. The mapping is the one table
//! `map_error`.

use std::cell::RefCell;
use std::ffi::{c_char, CString};

/// `tlinalg_status`: the C result of every entry point.
#[allow(non_camel_case_types)]
pub type tlinalg_status = i32;

/// Success. The only status a caller may assume means "outputs written".
pub const TLINALG_OK: tlinalg_status = 0;
/// A null pointer, an unknown enumerator, or an argument the provider rejected as invalid.
pub const TLINALG_ERR_INVALID_ARGUMENT: tlinalg_status = 1;
/// Extents, strides, offset, batch count or a caller buffer size are inconsistent or overflow.
pub const TLINALG_ERR_SHAPE: tlinalg_status = 2;
/// A factor, or an input the operation requires to be non-singular, was exactly singular.
pub const TLINALG_ERR_SINGULAR: tlinalg_status = 3;
/// A well-formed request this implementation declines (for example a reserved dtype tag).
pub const TLINALG_ERR_UNSUPPORTED: tlinalg_status = 4;
/// A bug or an allocation failure inside the library.
pub const TLINALG_ERR_INTERNAL: tlinalg_status = 5;
/// The algorithm did not converge for this input.
pub const TLINALG_ERR_NONCONVERGENCE: tlinalg_status = 6;
/// An input or a required computed quantity was not finite.
pub const TLINALG_ERR_NONFINITE: tlinalg_status = 7;

/// A one-line description of `status` (the text behind [`tlinalg_explain_error`]).
#[must_use]
pub fn describe(status: tlinalg_status) -> &'static str {
    match status {
        TLINALG_OK => "success",
        TLINALG_ERR_INVALID_ARGUMENT => "null pointer or invalid argument",
        TLINALG_ERR_SHAPE => "inconsistent or overflowing shape, stride or buffer size",
        TLINALG_ERR_SINGULAR => "singular factor",
        TLINALG_ERR_UNSUPPORTED => "operation not supported by this implementation",
        TLINALG_ERR_INTERNAL => "internal error",
        TLINALG_ERR_NONCONVERGENCE => "the algorithm did not converge",
        TLINALG_ERR_NONFINITE => "non-finite input or computed value",
        _ => "unknown error",
    }
}

/// An error crossing the ABI: a status and a message.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FfiError {
    /// Status code.
    pub status: tlinalg_status,
    /// Message for [`tlinalg_last_error`].
    pub message: String,
}

impl FfiError {
    /// A new error.
    pub fn new(status: tlinalg_status, message: impl Into<String>) -> Self {
        Self {
            status,
            message: message.into(),
        }
    }
}

/// An invalid argument (null pointer, unknown enumerator, rejected flag).
pub(crate) fn invalid(message: impl Into<String>) -> FfiError {
    FfiError::new(TLINALG_ERR_INVALID_ARGUMENT, message)
}

/// An inconsistent or overflowing shape, stride, offset or buffer size.
pub(crate) fn shape(message: impl Into<String>) -> FfiError {
    FfiError::new(TLINALG_ERR_SHAPE, message)
}

/// A well-formed request this implementation declines. Currently the reserved dtype tags.
pub(crate) fn unsupported(message: impl Into<String>) -> FfiError {
    FfiError::new(TLINALG_ERR_UNSUPPORTED, message)
}

/// The provider's typed error as a C status and message: the one mapping table of the ABI.
///
/// The provider distinguishes a factorization that did not converge from an exactly singular
/// factor, so the ABI does too. `Error::InvalidArgument` is the provider's own argument/shape
/// rejection (for example a non-square input to a square-only operation) and stays
/// [`TLINALG_ERR_INVALID_ARGUMENT`]; this crate's descriptor checks are what report
/// [`TLINALG_ERR_SHAPE`].
pub(crate) fn map_error(e: tlinalg::Error) -> FfiError {
    let status = match &e {
        tlinalg::Error::Singular { .. } => TLINALG_ERR_SINGULAR,
        tlinalg::Error::NonConvergence { .. } => TLINALG_ERR_NONCONVERGENCE,
        tlinalg::Error::NonFinite { .. } => TLINALG_ERR_NONFINITE,
        tlinalg::Error::InvalidArgument { .. } => TLINALG_ERR_INVALID_ARGUMENT,
        tlinalg::Error::Inconsistent { .. } => TLINALG_ERR_SHAPE,
        tlinalg::Error::InvalidWorkspace { .. } => TLINALG_ERR_INTERNAL,
        // `tlinalg::Error` is `#[non_exhaustive]`; a new variant is internal until mapped.
        _ => TLINALG_ERR_INTERNAL,
    };
    FfiError::new(status, e.to_string())
}

thread_local! {
    static LAST: RefCell<CString> = RefCell::new(CString::default());
}

/// Record `message` for [`tlinalg_last_error`] on this thread.
pub(crate) fn set_last_error(message: &str) {
    let c = CString::new(message.replace('\0', " ")).unwrap_or_default();
    // try_with: during thread teardown the slot may be gone; drop the message.
    let _ = LAST.try_with(|l| *l.borrow_mut() = c);
}

/// Clear the message without allocating when it is already empty (the success path).
fn clear_last_error() {
    let _ = LAST.try_with(|l| {
        if !l.borrow().as_bytes().is_empty() {
            *l.borrow_mut() = CString::default();
        }
    });
}

/// Run an ABI body: catch panics, record the error message, return a status.
///
/// No unwind crosses the C boundary: a panic becomes [`TLINALG_ERR_INTERNAL`] with its payload in
/// the thread-local message.
pub(crate) fn ffi(body: impl FnOnce() -> Result<(), FfiError>) -> tlinalg_status {
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(body)) {
        Ok(Ok(())) => {
            clear_last_error();
            TLINALG_OK
        }
        Ok(Err(e)) => {
            set_last_error(&e.message);
            e.status
        }
        Err(p) => {
            let msg = p
                .downcast_ref::<&str>()
                .map(|s| (*s).to_owned())
                .or_else(|| p.downcast_ref::<String>().cloned())
                .unwrap_or_else(|| "panic".into());
            set_last_error(&format!("panic: {msg}"));
            TLINALG_ERR_INTERNAL
        }
    }
}

/// `tlinalg_explain_error`: copy a description of `status` into `message`
/// (NUL-terminated, at most `maxlen - 1` characters). Returns the untruncated
/// length without the NUL, as `snprintf` does; `maxlen == 0` writes nothing.
///
/// # Safety
///
/// `message` is valid for `maxlen` bytes, or `maxlen` is zero.
#[no_mangle]
pub unsafe extern "C" fn tlinalg_explain_error(
    status: tlinalg_status,
    maxlen: usize,
    message: *mut c_char,
) -> usize {
    let s = describe(status).as_bytes();
    if !message.is_null() && maxlen > 0 {
        let n = s.len().min(maxlen - 1);
        // SAFETY: `n < maxlen` bytes are writable per the contract.
        unsafe {
            std::ptr::copy_nonoverlapping(s.as_ptr(), message.cast::<u8>(), n);
            *message.add(n) = 0;
        }
    }
    s.len()
}

/// The message of the last failed call on this thread (empty after a success).
/// Valid until the next tlinalg call on this thread; never freed.
#[no_mangle]
pub extern "C" fn tlinalg_last_error() -> *const c_char {
    LAST.try_with(|l| l.borrow().as_ptr())
        .unwrap_or(c"".as_ptr())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tlinalg::{Error, NonFiniteRole, Op};

    /// The mapping table is the contract: every provider variant has exactly one C status.
    #[test]
    fn the_error_table_maps_every_typed_variant() {
        let cases: [(Error, tlinalg_status); 6] = [
            (Error::Singular { op: Op::Cholesky }, TLINALG_ERR_SINGULAR),
            (
                Error::NonConvergence { op: Op::Svd },
                TLINALG_ERR_NONCONVERGENCE,
            ),
            (
                Error::NonFinite {
                    op: Op::Cholesky,
                    role: NonFiniteRole::Input,
                },
                TLINALG_ERR_NONFINITE,
            ),
            (
                Error::InvalidArgument {
                    op: Op::Eigh,
                    role: "configuration",
                    detail: "detail".into(),
                },
                TLINALG_ERR_INVALID_ARGUMENT,
            ),
            (
                Error::Inconsistent {
                    op: Op::Lu,
                    detail: "detail",
                },
                TLINALG_ERR_SHAPE,
            ),
            (
                Error::InvalidWorkspace {
                    op: Op::Svd,
                    library: "LAPACK",
                    routine: "routine",
                    detail: "detail".into(),
                },
                TLINALG_ERR_INTERNAL,
            ),
        ];
        for (error, want) in cases {
            let mapped = map_error(error);
            assert_eq!(mapped.status, want);
            assert!(!mapped.message.is_empty());
        }
    }

    /// Every published status has a description.
    #[test]
    fn describe_covers_every_status() {
        for status in [
            TLINALG_OK,
            TLINALG_ERR_INVALID_ARGUMENT,
            TLINALG_ERR_SHAPE,
            TLINALG_ERR_SINGULAR,
            TLINALG_ERR_UNSUPPORTED,
            TLINALG_ERR_INTERNAL,
            TLINALG_ERR_NONCONVERGENCE,
            TLINALG_ERR_NONFINITE,
        ] {
            assert_ne!(describe(status), "unknown error");
        }
        assert_eq!(describe(999), "unknown error");
    }
}
