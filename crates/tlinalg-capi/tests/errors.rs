//! Error paths of the C ABI: rejection before any write, provider failures, and the status
//! vocabulary.

mod common;

use std::ffi::{c_char, c_void, CStr};

use common::{last_error, Layout};
use tlinalg_capi::*;

/// A 2x2 identity, compact column-major.
fn identity_2x2() -> [f64; 4] {
    [1.0, 0.0, 0.0, 1.0]
}

#[test]
fn too_short_buffer_is_rejected_before_any_write() {
    let a = identity_2x2();
    let layout = Layout::compact(&[2, 2]);
    let desc = layout.as_c();

    // u is one element short; every buffer is prefilled with a sentinel.
    let mut u = [123.0f64; 4];
    let mut s = [123.0f64; 2];
    let mut vt = [123.0f64; 4];
    let status = unsafe {
        tlinalg_svd_into(
            &desc,
            TLINALG_F64,
            0,
            a.as_ptr().cast::<c_void>(),
            a.len(),
            0,
            u.as_mut_ptr().cast::<c_void>(),
            3, // one short of the required 4
            s.as_mut_ptr().cast::<c_void>(),
            s.len(),
            vt.as_mut_ptr().cast::<c_void>(),
            vt.len(),
            std::ptr::null_mut(),
            0,
        )
    };
    assert_eq!(status, TLINALG_ERR_SHAPE, "{}", last_error());
    assert_eq!(u, [123.0; 4], "u was written");
    assert_eq!(s, [123.0; 2], "s was written");
    assert_eq!(vt, [123.0; 4], "vt was written");

    // A short middle buffer still leaves every buffer untouched: all capacities are
    // checked before the provider runs.
    let status = unsafe {
        tlinalg_svd_into(
            &desc,
            TLINALG_F64,
            0,
            a.as_ptr().cast::<c_void>(),
            a.len(),
            0,
            u.as_mut_ptr().cast::<c_void>(),
            u.len(),
            s.as_mut_ptr().cast::<c_void>(),
            1, // one short of 2
            vt.as_mut_ptr().cast::<c_void>(),
            vt.len(),
            std::ptr::null_mut(),
            0,
        )
    };
    assert_eq!(status, TLINALG_ERR_SHAPE, "{}", last_error());
    assert_eq!(u, [123.0; 4], "u was written");
    assert_eq!(s, [123.0; 2], "s was written");
}

#[test]
fn singular_cholesky_reports_the_provider_classification() {
    // [[1, 1], [1, 1]] is singular positive semi-definite, not positive definite.
    let a = [1.0f64, 1.0, 1.0, 1.0];
    let layout = Layout::compact(&[2, 2]);
    let desc = layout.as_c();
    let mut l = [123.0f64; 4];
    let status = unsafe {
        tlinalg_cholesky_into(
            &desc,
            TLINALG_F64,
            0,
            a.as_ptr().cast::<c_void>(),
            a.len(),
            l.as_mut_ptr().cast::<c_void>(),
            l.len(),
            std::ptr::null_mut(),
            0,
        )
    };
    // The provider classifies a non-positive-definite item as `NonConvergence`, so this is the
    // status the ABI reports; `Error::Singular` (and TLINALG_ERR_SINGULAR) belongs to the solve
    // routes, which are not part of v1.
    assert_eq!(status, TLINALG_ERR_NONCONVERGENCE, "{}", last_error());
    assert_eq!(l, [123.0; 4], "a failed call wrote the caller buffer");
}

#[test]
fn non_square_input_is_a_shape_error() {
    let a = [0.0f64; 6];
    let layout = Layout::compact(&[3, 2]);
    let desc = layout.as_c();
    let mut out = tlinalg_eigh_out {
        batch: 0,
        values: 0,
        vectors: 0,
        workspace: 0,
    };
    assert_eq!(
        unsafe { tlinalg_eigh_sizes(&desc, &mut out) },
        TLINALG_ERR_SHAPE,
        "{}",
        last_error()
    );
    let mut chol = tlinalg_cholesky_out {
        batch: 0,
        l: 0,
        workspace: 0,
    };
    assert_eq!(
        unsafe { tlinalg_cholesky_sizes(&desc, &mut chol) },
        TLINALG_ERR_SHAPE,
        "{}",
        last_error()
    );
    assert_eq!(
        unsafe {
            tlinalg_eigh_into(
                &desc,
                TLINALG_F64,
                0,
                a.as_ptr().cast::<c_void>(),
                a.len(),
                std::ptr::null_mut(),
                0,
                std::ptr::null_mut(),
                0,
                std::ptr::null_mut(),
                0,
            )
        },
        TLINALG_ERR_SHAPE,
        "{}",
        last_error()
    );
}

#[test]
fn a_descriptor_reaching_outside_the_buffer_is_a_shape_error() {
    // The second column's stride reaches far past the 4-element buffer.
    let a = identity_2x2();
    let layout = Layout::new(&[2, 2], &[1, 100], 0);
    let desc = layout.as_c();
    let mut u = [123.0f64; 4];
    let mut s = [123.0f64; 2];
    let mut vt = [123.0f64; 4];
    let status = unsafe {
        tlinalg_svd_into(
            &desc,
            TLINALG_F64,
            0,
            a.as_ptr().cast::<c_void>(),
            a.len(),
            0,
            u.as_mut_ptr().cast::<c_void>(),
            u.len(),
            s.as_mut_ptr().cast::<c_void>(),
            s.len(),
            vt.as_mut_ptr().cast::<c_void>(),
            vt.len(),
            std::ptr::null_mut(),
            0,
        )
    };
    assert_eq!(status, TLINALG_ERR_SHAPE, "{}", last_error());
    assert_eq!(
        u, [123.0; 4],
        "a rejected descriptor wrote the caller buffer"
    );
}

#[test]
fn null_pointers_are_invalid_arguments() {
    let a = identity_2x2();
    let layout = Layout::compact(&[2, 2]);
    let desc = layout.as_c();
    let mut out = tlinalg_svd_out {
        batch: 0,
        u: 0,
        s: 0,
        vt: 0,
        workspace: 0,
    };

    // Null descriptor and null sizes output.
    assert_eq!(
        unsafe { tlinalg_svd_sizes(std::ptr::null(), 0, &mut out) },
        TLINALG_ERR_INVALID_ARGUMENT
    );
    assert_eq!(
        unsafe { tlinalg_svd_sizes(&desc, 0, std::ptr::null_mut()) },
        TLINALG_ERR_INVALID_ARGUMENT
    );

    // Null input data.
    let (mut u, mut s, mut vt) = ([0.0f64; 4], [0.0f64; 2], [0.0f64; 4]);
    assert_eq!(
        unsafe {
            tlinalg_svd_into(
                &desc,
                TLINALG_F64,
                0,
                std::ptr::null(),
                a.len(),
                0,
                u.as_mut_ptr().cast::<c_void>(),
                u.len(),
                s.as_mut_ptr().cast::<c_void>(),
                s.len(),
                vt.as_mut_ptr().cast::<c_void>(),
                vt.len(),
                std::ptr::null_mut(),
                0,
            )
        },
        TLINALG_ERR_INVALID_ARGUMENT
    );

    // Null output buffer with a non-zero requirement.
    assert_eq!(
        unsafe {
            tlinalg_svd_into(
                &desc,
                TLINALG_F64,
                0,
                a.as_ptr().cast::<c_void>(),
                a.len(),
                0,
                std::ptr::null_mut(),
                4,
                s.as_mut_ptr().cast::<c_void>(),
                s.len(),
                vt.as_mut_ptr().cast::<c_void>(),
                vt.len(),
                std::ptr::null_mut(),
                0,
            )
        },
        TLINALG_ERR_INVALID_ARGUMENT
    );
}

#[test]
fn reserved_and_unknown_dtype_tags_are_rejected() {
    let a = identity_2x2();
    let layout = Layout::compact(&[2, 2]);
    let desc = layout.as_c();
    let (mut u, mut s, mut vt) = ([0.0f64; 4], [0.0f64; 2], [0.0f64; 4]);
    let mut call = |dtype: tlinalg_dtype| unsafe {
        tlinalg_svd_into(
            &desc,
            dtype,
            0,
            a.as_ptr().cast::<c_void>(),
            a.len(),
            0,
            u.as_mut_ptr().cast::<c_void>(),
            u.len(),
            s.as_mut_ptr().cast::<c_void>(),
            s.len(),
            vt.as_mut_ptr().cast::<c_void>(),
            vt.len(),
            std::ptr::null_mut(),
            0,
        )
    };
    assert_eq!(
        call(TLINALG_F16),
        TLINALG_ERR_UNSUPPORTED,
        "{}",
        last_error()
    );
    assert_eq!(
        call(TLINALG_BF16),
        TLINALG_ERR_UNSUPPORTED,
        "{}",
        last_error()
    );
    assert_eq!(call(99), TLINALG_ERR_INVALID_ARGUMENT, "{}", last_error());
}

#[test]
fn an_invalid_full_flag_is_an_invalid_argument() {
    let layout = Layout::compact(&[2, 2]);
    let desc = layout.as_c();
    let mut out = tlinalg_svd_out {
        batch: 0,
        u: 0,
        s: 0,
        vt: 0,
        workspace: 0,
    };
    assert_eq!(
        unsafe { tlinalg_svd_sizes(&desc, 2, &mut out) },
        TLINALG_ERR_INVALID_ARGUMENT
    );
    assert_eq!(
        unsafe { tlinalg_svd_sizes(&desc, -1, &mut out) },
        TLINALG_ERR_INVALID_ARGUMENT
    );
}

#[test]
fn a_batch_larger_than_one_queries_and_writes_the_same_totals() {
    let mut a = Vec::new();
    for item in 0..3u32 {
        let k = f64::from(item) + 1.0;
        a.extend_from_slice(&[k, 0.0, 0.0, k]); // k * I
    }
    let layout = Layout::compact(&[2, 2, 3]);
    let desc = layout.as_c();

    let mut sizes = tlinalg_svd_out {
        batch: 0,
        u: 0,
        s: 0,
        vt: 0,
        workspace: 0,
    };
    assert_eq!(
        unsafe { tlinalg_svd_sizes(&desc, 0, &mut sizes) },
        TLINALG_OK
    );
    assert_eq!(sizes.batch, 3);
    assert_eq!((sizes.u, sizes.s, sizes.vt), (4, 2, 4));

    let mut u = vec![0.0f64; sizes.u * sizes.batch];
    let mut s = vec![0.0f64; sizes.s * sizes.batch];
    let mut vt = vec![0.0f64; sizes.vt * sizes.batch];
    assert_eq!(
        unsafe {
            tlinalg_svd_into(
                &desc,
                TLINALG_F64,
                0,
                a.as_ptr().cast::<c_void>(),
                a.len(),
                0,
                u.as_mut_ptr().cast::<c_void>(),
                u.len(),
                s.as_mut_ptr().cast::<c_void>(),
                s.len(),
                vt.as_mut_ptr().cast::<c_void>(),
                vt.len(),
                std::ptr::null_mut(),
                0,
            )
        },
        TLINALG_OK,
        "{}",
        last_error()
    );
    // Every item's singular values are its scale factor, in order.
    for item in 0..3 {
        let k = item as f64 + 1.0;
        assert_eq!(&s[2 * item..2 * item + 2], &[k, k]);
    }
}

#[test]
fn explain_error_truncates_and_last_error_tracks_the_call() {
    let mut buf = [0 as c_char; 8];
    let full = unsafe { tlinalg_explain_error(TLINALG_ERR_SHAPE, buf.len(), buf.as_mut_ptr()) };
    let text = unsafe { CStr::from_ptr(buf.as_ptr()) }.to_str().unwrap();
    assert!(full >= buf.len());
    assert_eq!(text.len(), buf.len() - 1);

    let mut whole = [0 as c_char; 128];
    let full = unsafe { tlinalg_explain_error(TLINALG_OK, whole.len(), whole.as_mut_ptr()) };
    let text = unsafe { CStr::from_ptr(whole.as_ptr()) }.to_str().unwrap();
    assert_eq!(text, "success");
    assert_eq!(full, "success".len());

    // maxlen zero writes nothing and still reports the length.
    let full = unsafe { tlinalg_explain_error(TLINALG_OK, 0, std::ptr::null_mut()) };
    assert_eq!(full, "success".len());

    // The thread-local message is set on failure and cleared on success.
    let layout = Layout::compact(&[2, 2]);
    let desc = layout.as_c();
    let mut out = tlinalg_svd_out {
        batch: 0,
        u: 0,
        s: 0,
        vt: 0,
        workspace: 0,
    };
    assert_eq!(
        unsafe { tlinalg_svd_sizes(std::ptr::null(), 0, &mut out) },
        TLINALG_ERR_INVALID_ARGUMENT
    );
    assert!(!last_error().is_empty());
    assert_eq!(unsafe { tlinalg_svd_sizes(&desc, 0, &mut out) }, TLINALG_OK);
    assert!(last_error().is_empty());
}
