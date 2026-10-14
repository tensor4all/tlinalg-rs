//! The header re-declared, and a full call through the re-declaration.
//!
//! The other suites call the crate's `extern "C"` functions by Rust path, which checks their
//! behaviour but not their *linkage*: a `#[no_mangle]` that was renamed, dropped, or given a
//! signature the header does not describe would sail through. This file transcribes
//! `include/tlinalg/tlinalg.h` into an `extern "C"` block and drives one call per entry point
//! through it, so every symbol has to exist, be `extern "C"`, and take the argument list the
//! header publishes. The C struct layouts are also checked against the crate's.

use std::ffi::c_void;
use std::mem::{align_of, offset_of, size_of};
use std::os::raw::{c_char, c_int};

/// `include/tlinalg/tlinalg.h`, transcribed.
#[allow(non_camel_case_types)]
mod c_abi {
    use std::ffi::c_void;
    use std::os::raw::{c_char, c_int};

    #[repr(C)]
    pub struct tlinalg_strided {
        pub rank: i32,
        pub extents: *const i64,
        pub strides: *const i64,
        pub offset: i64,
    }

    #[repr(C)]
    pub struct tlinalg_svd_out {
        pub batch: usize,
        pub u: usize,
        pub s: usize,
        pub vt: usize,
        pub workspace: usize,
    }

    #[repr(C)]
    pub struct tlinalg_svd_values_out {
        pub batch: usize,
        pub s: usize,
        pub workspace: usize,
    }

    #[repr(C)]
    pub struct tlinalg_eigh_out {
        pub batch: usize,
        pub values: usize,
        pub vectors: usize,
        pub workspace: usize,
    }

    #[repr(C)]
    pub struct tlinalg_eigh_values_out {
        pub batch: usize,
        pub values: usize,
        pub workspace: usize,
    }

    #[repr(C)]
    pub struct tlinalg_cholesky_out {
        pub batch: usize,
        pub l: usize,
        pub workspace: usize,
    }

    #[repr(C)]
    pub struct tlinalg_lu_out {
        pub batch: usize,
        pub p: usize,
        pub l: usize,
        pub u: usize,
        pub parity: usize,
        pub workspace: usize,
    }

    #[repr(C)]
    pub struct tlinalg_full_piv_lu_out {
        pub batch: usize,
        pub p: usize,
        pub l: usize,
        pub u: usize,
        pub q: usize,
        pub parity: usize,
        pub workspace: usize,
    }

    extern "C" {
        pub fn tlinalg_abi_version() -> u32;
        pub fn tlinalg_last_error() -> *const c_char;
        pub fn tlinalg_explain_error(status: c_int, maxlen: usize, message: *mut c_char) -> usize;

        pub fn tlinalg_svd_sizes(
            input: *const tlinalg_strided,
            full: i32,
            out: *mut tlinalg_svd_out,
        ) -> c_int;
        #[allow(clippy::too_many_arguments)]
        pub fn tlinalg_svd_into(
            input: *const tlinalg_strided,
            dtype: i32,
            exec: isize,
            data: *const c_void,
            data_len: usize,
            full: i32,
            u: *mut c_void,
            u_len: usize,
            s: *mut c_void,
            s_len: usize,
            vt: *mut c_void,
            vt_len: usize,
            workspace: *mut c_void,
            workspace_len: usize,
        ) -> c_int;

        pub fn tlinalg_svd_values_sizes(
            input: *const tlinalg_strided,
            out: *mut tlinalg_svd_values_out,
        ) -> c_int;
        pub fn tlinalg_svd_values_into(
            input: *const tlinalg_strided,
            dtype: i32,
            exec: isize,
            data: *const c_void,
            data_len: usize,
            s: *mut c_void,
            s_len: usize,
            workspace: *mut c_void,
            workspace_len: usize,
        ) -> c_int;

        pub fn tlinalg_eigh_sizes(
            input: *const tlinalg_strided,
            out: *mut tlinalg_eigh_out,
        ) -> c_int;
        #[allow(clippy::too_many_arguments)]
        pub fn tlinalg_eigh_into(
            input: *const tlinalg_strided,
            dtype: i32,
            exec: isize,
            data: *const c_void,
            data_len: usize,
            values: *mut c_void,
            values_len: usize,
            vectors: *mut c_void,
            vectors_len: usize,
            workspace: *mut c_void,
            workspace_len: usize,
        ) -> c_int;

        pub fn tlinalg_eigh_values_sizes(
            input: *const tlinalg_strided,
            out: *mut tlinalg_eigh_values_out,
        ) -> c_int;
        pub fn tlinalg_eigh_values_into(
            input: *const tlinalg_strided,
            dtype: i32,
            exec: isize,
            data: *const c_void,
            data_len: usize,
            values: *mut c_void,
            values_len: usize,
            workspace: *mut c_void,
            workspace_len: usize,
        ) -> c_int;

        pub fn tlinalg_cholesky_sizes(
            input: *const tlinalg_strided,
            out: *mut tlinalg_cholesky_out,
        ) -> c_int;
        pub fn tlinalg_cholesky_into(
            input: *const tlinalg_strided,
            dtype: i32,
            exec: isize,
            data: *const c_void,
            data_len: usize,
            l: *mut c_void,
            l_len: usize,
            workspace: *mut c_void,
            workspace_len: usize,
        ) -> c_int;

        pub fn tlinalg_lu_sizes(input: *const tlinalg_strided, out: *mut tlinalg_lu_out) -> c_int;
        #[allow(clippy::too_many_arguments)]
        pub fn tlinalg_lu_into(
            input: *const tlinalg_strided,
            dtype: i32,
            exec: isize,
            data: *const c_void,
            data_len: usize,
            p: *mut c_void,
            p_len: usize,
            l: *mut c_void,
            l_len: usize,
            u: *mut c_void,
            u_len: usize,
            parity: *mut c_void,
            parity_len: usize,
            workspace: *mut c_void,
            workspace_len: usize,
        ) -> c_int;

        pub fn tlinalg_full_piv_lu_sizes(
            input: *const tlinalg_strided,
            out: *mut tlinalg_full_piv_lu_out,
        ) -> c_int;
        #[allow(clippy::too_many_arguments)]
        pub fn tlinalg_full_piv_lu_into(
            input: *const tlinalg_strided,
            dtype: i32,
            exec: isize,
            data: *const c_void,
            data_len: usize,
            p: *mut c_void,
            p_len: usize,
            l: *mut c_void,
            l_len: usize,
            u: *mut c_void,
            u_len: usize,
            q: *mut c_void,
            q_len: usize,
            parity: *mut c_void,
            parity_len: usize,
            workspace: *mut c_void,
            workspace_len: usize,
        ) -> c_int;
    }
}

use c_abi::*;

const OK: c_int = 0;
const F64: i32 = tlinalg_capi::TLINALG_F64;

/// The C structs must have the crate's layout, or a re-declared call would corrupt them.
#[test]
fn the_redeclared_structs_have_the_crate_layout() {
    use tlinalg_capi as capi;
    assert_eq!(
        size_of::<tlinalg_strided>(),
        size_of::<capi::tlinalg_strided>()
    );
    assert_eq!(
        offset_of!(tlinalg_strided, offset),
        offset_of!(capi::tlinalg_strided, offset)
    );
    assert_eq!(
        align_of::<tlinalg_strided>(),
        align_of::<capi::tlinalg_strided>()
    );

    assert_eq!(
        size_of::<tlinalg_svd_out>(),
        size_of::<capi::tlinalg_svd_out>()
    );
    assert_eq!(
        size_of::<tlinalg_svd_values_out>(),
        size_of::<capi::tlinalg_svd_values_out>()
    );
    assert_eq!(
        size_of::<tlinalg_eigh_out>(),
        size_of::<capi::tlinalg_eigh_out>()
    );
    assert_eq!(
        size_of::<tlinalg_eigh_values_out>(),
        size_of::<capi::tlinalg_eigh_values_out>()
    );
    assert_eq!(
        size_of::<tlinalg_cholesky_out>(),
        size_of::<capi::tlinalg_cholesky_out>()
    );
    assert_eq!(
        size_of::<tlinalg_lu_out>(),
        size_of::<capi::tlinalg_lu_out>()
    );
    assert_eq!(
        size_of::<tlinalg_full_piv_lu_out>(),
        size_of::<capi::tlinalg_full_piv_lu_out>()
    );
}

fn desc(extents: &[i64], strides: &[i64]) -> tlinalg_strided {
    tlinalg_strided {
        rank: extents.len() as i32,
        extents: extents.as_ptr(),
        strides: strides.as_ptr(),
        offset: 0,
    }
}

/// One call per entry point, through the re-declared prototypes.
#[test]
fn every_entry_point_links_and_runs() {
    let two = [2i64, 2];
    let col = [1i64, 2];
    let d = desc(&two, &col);

    unsafe {
        assert_eq!(tlinalg_abi_version(), tlinalg_capi::ABI_VERSION);
    }
    assert!(!tlinalg_last_error_is_null());

    // SVD, economy.
    let a = [3.0f64, 0.0, 0.0, 1.0];
    let mut svd = tlinalg_svd_out {
        batch: 0,
        u: 0,
        s: 0,
        vt: 0,
        workspace: 0,
    };
    assert_eq!(unsafe { tlinalg_svd_sizes(&d, 0, &mut svd) }, OK);
    assert_eq!((svd.batch, svd.u, svd.s, svd.vt), (1, 4, 2, 4));
    let (mut u, mut s, mut vt) = ([0.0f64; 4], [0.0f64; 2], [0.0f64; 4]);
    assert_eq!(
        unsafe {
            tlinalg_svd_into(
                &d,
                F64,
                0,
                a.as_ptr().cast::<c_void>(),
                4,
                0,
                u.as_mut_ptr().cast::<c_void>(),
                4,
                s.as_mut_ptr().cast::<c_void>(),
                2,
                vt.as_mut_ptr().cast::<c_void>(),
                4,
                std::ptr::null_mut(),
                0,
            )
        },
        OK
    );
    assert!((s[0] - 3.0).abs() < 1e-12 && (s[1] - 1.0).abs() < 1e-12);

    // SVD values.
    let mut sv = tlinalg_svd_values_out {
        batch: 0,
        s: 0,
        workspace: 0,
    };
    assert_eq!(unsafe { tlinalg_svd_values_sizes(&d, &mut sv) }, OK);
    let mut ss = [0.0f64; 2];
    assert_eq!(
        unsafe {
            tlinalg_svd_values_into(
                &d,
                F64,
                0,
                a.as_ptr().cast::<c_void>(),
                4,
                ss.as_mut_ptr().cast::<c_void>(),
                2,
                std::ptr::null_mut(),
                0,
            )
        },
        OK
    );

    // eigh and eigh_values.
    let sym = [2.0f64, 0.0, 0.0, 3.0];
    let mut eh = tlinalg_eigh_out {
        batch: 0,
        values: 0,
        vectors: 0,
        workspace: 0,
    };
    assert_eq!(unsafe { tlinalg_eigh_sizes(&d, &mut eh) }, OK);
    let (mut ev, mut ew) = ([0.0f64; 2], [0.0f64; 4]);
    assert_eq!(
        unsafe {
            tlinalg_eigh_into(
                &d,
                F64,
                0,
                sym.as_ptr().cast::<c_void>(),
                4,
                ev.as_mut_ptr().cast::<c_void>(),
                2,
                ew.as_mut_ptr().cast::<c_void>(),
                4,
                std::ptr::null_mut(),
                0,
            )
        },
        OK
    );
    assert!((ev[0] - 2.0).abs() < 1e-12 && (ev[1] - 3.0).abs() < 1e-12);

    let mut ehv = tlinalg_eigh_values_out {
        batch: 0,
        values: 0,
        workspace: 0,
    };
    assert_eq!(unsafe { tlinalg_eigh_values_sizes(&d, &mut ehv) }, OK);
    let mut evv = [0.0f64; 2];
    assert_eq!(
        unsafe {
            tlinalg_eigh_values_into(
                &d,
                F64,
                0,
                sym.as_ptr().cast::<c_void>(),
                4,
                evv.as_mut_ptr().cast::<c_void>(),
                2,
                std::ptr::null_mut(),
                0,
            )
        },
        OK
    );

    // Cholesky.
    let hpd = [4.0f64, 2.0, 2.0, 3.0];
    let mut ch = tlinalg_cholesky_out {
        batch: 0,
        l: 0,
        workspace: 0,
    };
    assert_eq!(unsafe { tlinalg_cholesky_sizes(&d, &mut ch) }, OK);
    let mut l = [0.0f64; 4];
    assert_eq!(
        unsafe {
            tlinalg_cholesky_into(
                &d,
                F64,
                0,
                hpd.as_ptr().cast::<c_void>(),
                4,
                l.as_mut_ptr().cast::<c_void>(),
                4,
                std::ptr::null_mut(),
                0,
            )
        },
        OK
    );
    assert!((l[0] - 2.0).abs() < 1e-12);

    // Partial-pivot LU.
    let mut lo = tlinalg_lu_out {
        batch: 0,
        p: 0,
        l: 0,
        u: 0,
        parity: 0,
        workspace: 0,
    };
    assert_eq!(unsafe { tlinalg_lu_sizes(&d, &mut lo) }, OK);
    let (mut p, mut ll, mut uu, mut par) = ([0.0f64; 4], [0.0f64; 4], [0.0f64; 4], [0.0f64; 1]);
    assert_eq!(
        unsafe {
            tlinalg_lu_into(
                &d,
                F64,
                0,
                hpd.as_ptr().cast::<c_void>(),
                4,
                p.as_mut_ptr().cast::<c_void>(),
                4,
                ll.as_mut_ptr().cast::<c_void>(),
                4,
                uu.as_mut_ptr().cast::<c_void>(),
                4,
                par.as_mut_ptr().cast::<c_void>(),
                1,
                std::ptr::null_mut(),
                0,
            )
        },
        OK
    );

    // Full-pivot LU.
    let mut fo = tlinalg_full_piv_lu_out {
        batch: 0,
        p: 0,
        l: 0,
        u: 0,
        q: 0,
        parity: 0,
        workspace: 0,
    };
    assert_eq!(unsafe { tlinalg_full_piv_lu_sizes(&d, &mut fo) }, OK);
    let (mut fp, mut fl, mut fu, mut fq, mut fpar) = (
        [0.0f64; 4],
        [0.0f64; 4],
        [0.0f64; 4],
        [0.0f64; 4],
        [0.0f64; 1],
    );
    assert_eq!(
        unsafe {
            tlinalg_full_piv_lu_into(
                &d,
                F64,
                0,
                hpd.as_ptr().cast::<c_void>(),
                4,
                fp.as_mut_ptr().cast::<c_void>(),
                4,
                fl.as_mut_ptr().cast::<c_void>(),
                4,
                fu.as_mut_ptr().cast::<c_void>(),
                4,
                fq.as_mut_ptr().cast::<c_void>(),
                4,
                fpar.as_mut_ptr().cast::<c_void>(),
                1,
                std::ptr::null_mut(),
                0,
            )
        },
        OK
    );

    // The explanation helper.
    let mut message = [0 as c_char; 32];
    let n = unsafe { tlinalg_explain_error(OK, message.len(), message.as_mut_ptr()) };
    assert_eq!(n, "success".len());
}

fn tlinalg_last_error_is_null() -> bool {
    // SAFETY: the function never returns null.
    unsafe { tlinalg_last_error().is_null() }
}
