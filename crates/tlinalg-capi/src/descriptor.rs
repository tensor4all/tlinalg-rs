//! The borrowed, dtype-erased input descriptor and its validation.
//!
//! A descriptor carries no pointer, no dtype and no ownership: it is rank, extents, strides and an
//! element offset. The caller passes the element pointer and the buffer's element count separately
//! (`with_view`), which is what lets this crate reject a descriptor that reaches outside the buffer
//! before the provider ever sees it.

use std::ffi::c_void;
use std::mem::{align_of, size_of};

use strided_view::RawStridedRef;

use crate::status::{invalid, shape, unsupported, FfiError};

/// `tlinalg_dtype`: the C dtype tag (`int32_t`).
#[allow(non_camel_case_types)]
pub type tlinalg_dtype = i32;

/// `tlinalg_dtype`: single-precision real, i.e. C `float`.
pub const TLINALG_F32: i32 = 0;
/// `tlinalg_dtype`: double-precision real, i.e. C `double`.
pub const TLINALG_F64: i32 = 1;
/// `tlinalg_dtype`: single-precision complex, interleaved, layout-compatible with C99
/// `float _Complex`.
pub const TLINALG_C32: i32 = 2;
/// `tlinalg_dtype`: double-precision complex, interleaved.
pub const TLINALG_C64: i32 = 3;
/// `tlinalg_dtype`: half precision. Reserved; rejected with [`crate::TLINALG_ERR_UNSUPPORTED`].
pub const TLINALG_F16: i32 = 4;
/// `tlinalg_dtype`: bfloat16. Reserved; rejected with [`crate::TLINALG_ERR_UNSUPPORTED`].
pub const TLINALG_BF16: i32 = 5;

/// The scalar storage types this ABI dispatches to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TlinalgDtype {
    /// `TLINALG_F32`.
    F32,
    /// `TLINALG_F64`.
    F64,
    /// `TLINALG_C32`.
    C32,
    /// `TLINALG_C64`.
    C64,
}

impl TlinalgDtype {
    /// The four supported tags, or the status a rejected tag reports.
    pub fn from_tag(tag: i32) -> Result<Self, FfiError> {
        match tag {
            TLINALG_F32 => Ok(Self::F32),
            TLINALG_F64 => Ok(Self::F64),
            TLINALG_C32 => Ok(Self::C32),
            TLINALG_C64 => Ok(Self::C64),
            TLINALG_F16 | TLINALG_BF16 => Err(unsupported(format!(
                "dtype tag {tag} is reserved and not implemented"
            ))),
            _ => Err(invalid(format!("unknown dtype tag {tag}"))),
        }
    }
}

/// `tlinalg_strided`: a borrowed rank-`2 + B` input layout, dtype-erased.
///
/// `extents` and `strides` are `rank` element counts and element strides; the first batch axis is
/// `extents[2]` and is the fastest-varying. `offset` is the element offset of the first matrix
/// element from the buffer `data`. The buffer itself, its element count and its dtype are separate
/// arguments of each entry point.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct tlinalg_strided {
    /// Rank `2 + B`, at least 2.
    pub rank: i32,
    /// `rank` extents in elements: `[rows, cols, b_1, ..., b_B]`.
    pub extents: *const i64,
    /// `rank` strides in elements, non-negative for the compact case.
    pub strides: *const i64,
    /// Element offset from `data` of the first matrix element.
    pub offset: i64,
}

/// Read a descriptor into owned extents and strides.
///
/// `desc` may be null, which is reported as an invalid argument. The descriptor's `rank`,
/// `extents` and `strides` are the caller's contract; this reads exactly `rank` entries from each
/// array.
pub(crate) fn parse_layout(
    desc: *const tlinalg_strided,
) -> Result<(Vec<usize>, Vec<isize>, isize), FfiError> {
    if desc.is_null() {
        return Err(invalid("input descriptor is null"));
    }
    // SAFETY: the caller hands a valid descriptor for the duration of the call.
    let desc = unsafe { *desc };
    if desc.rank < 2 {
        return Err(shape(format!("input rank {} is below 2", desc.rank)));
    }
    if desc.extents.is_null() || desc.strides.is_null() {
        return Err(invalid("input extents or strides pointer is null"));
    }
    let rank = desc.rank as usize;
    // SAFETY: the caller guarantees `rank` readable entries in each array.
    let extents = unsafe { std::slice::from_raw_parts(desc.extents, rank) };
    // SAFETY: as above.
    let strides = unsafe { std::slice::from_raw_parts(desc.strides, rank) };

    let dims = extents
        .iter()
        .map(|&e| {
            usize::try_from(e)
                .map_err(|_| shape(format!("input extent {e} is negative or overflows usize")))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let strides = strides
        .iter()
        .map(|&s| {
            isize::try_from(s).map_err(|_| shape(format!("input stride {s} overflows isize")))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let offset = isize::try_from(desc.offset)
        .map_err(|_| shape(format!("input offset {} overflows isize", desc.offset)))?;
    Ok((dims, strides, offset))
}

/// Build a typed borrowed view of the caller's buffer and run `body` on it.
///
/// `data_len` is the number of `T` elements in the buffer; the descriptor must not reach outside
/// `[0, data_len)`. A pointer that is null, misaligned or reached outside the buffer is rejected
/// here, before the provider is called.
pub(crate) fn with_view<T, R>(
    desc: *const tlinalg_strided,
    data: *const c_void,
    data_len: usize,
    body: impl FnOnce(RawStridedRef<'_, T>) -> Result<R, FfiError>,
) -> Result<R, FfiError> {
    let (dims, strides, offset) = parse_layout(desc)?;
    let slice: &[T] = unsafe { typed_slice::<T>(data, data_len)? };
    let view = RawStridedRef::new(slice, &dims, &strides, offset)
        .map_err(|e| shape(format!("input descriptor is invalid: {e}")))?;
    body(view)
}

/// Borrow `len` `T` elements from `data`, checking null and alignment.
///
/// # Safety
///
/// The caller must guarantee that `data` points to `len` initialized, readable `T` values with
/// valid provenance for the duration of the returned slice, or that `len` is zero.
unsafe fn typed_slice<'a, T>(data: *const c_void, len: usize) -> Result<&'a [T], FfiError> {
    if len == 0 {
        return Ok(&[]);
    }
    if data.is_null() {
        return Err(invalid("input data pointer is null"));
    }
    if !(data as usize).is_multiple_of(align_of::<T>()) {
        return Err(invalid(format!(
            "input data pointer is not aligned for a {}-byte scalar",
            size_of::<T>()
        )));
    }
    // SAFETY: non-null and aligned per the checks above, and the caller owns the storage.
    Ok(unsafe { std::slice::from_raw_parts(data.cast::<T>(), len) })
}
