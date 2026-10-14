//! The `exec` handle: a host's pool serves these calls, and the results match
//! the sequential path.

mod common;

use std::ffi::c_void;

use common::{assert_close_f64, last_error, Layout};
use tcontract_capi::{
    tcontract_pool, tcontract_pool_create_rayon, tcontract_pool_destroy,
    tcontract_tapp_executor_borrow_pool, tcontract_tapp_executor_create_rayon,
    tcontract_tapp_executor_get_threads, TAPP_check_success, TAPP_create_executor,
    TAPP_destroy_executor,
};
use tlinalg_capi::{
    tlinalg_cholesky_into, tlinalg_cholesky_sizes, tlinalg_svd_into, tlinalg_svd_sizes,
    TLINALG_F64, TLINALG_OK,
};

/// The SVD of `A = [[3, 0], [0, 1]]` through `exec`, and the sizes query.
fn svd_with(exec: isize) -> ([f64; 4], [f64; 2], [f64; 4]) {
    let a = [3.0f64, 0.0, 0.0, 1.0];
    let layout = Layout::compact(&[2, 2]);
    let desc = layout.as_c();
    let mut sizes = std::mem::MaybeUninit::uninit();
    // SAFETY: a valid descriptor and a writable out-parameter.
    assert_eq!(
        unsafe { tlinalg_svd_sizes(&desc, 0, sizes.as_mut_ptr()) },
        TLINALG_OK,
        "{}",
        last_error()
    );
    // SAFETY: the query above initialized it.
    let sizes = unsafe { sizes.assume_init() };
    assert_eq!((sizes.batch, sizes.u, sizes.s, sizes.vt), (1, 4, 2, 4));

    let (mut u, mut s, mut vt) = ([0.0f64; 4], [0.0f64; 2], [0.0f64; 4]);
    // SAFETY: every pointer is valid for the lengths passed, and `exec` is zero
    // or a live executor handle.
    let status = unsafe {
        tlinalg_svd_into(
            &desc,
            TLINALG_F64,
            exec,
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
    assert_eq!(status, TLINALG_OK, "{}", last_error());
    (u, s, vt)
}

fn cholesky_with(exec: isize) -> [f64; 4] {
    let a = [4.0f64, 0.0, 0.0, 9.0];
    let layout = Layout::compact(&[2, 2]);
    let desc = layout.as_c();
    let mut sizes = std::mem::MaybeUninit::uninit();
    // SAFETY: a valid descriptor and a writable out-parameter.
    assert_eq!(
        unsafe { tlinalg_cholesky_sizes(&desc, sizes.as_mut_ptr()) },
        TLINALG_OK,
        "{}",
        last_error()
    );
    let mut l = [0.0f64; 4];
    // SAFETY: as above.
    let status = unsafe {
        tlinalg_cholesky_into(
            &desc,
            TLINALG_F64,
            exec,
            a.as_ptr().cast::<c_void>(),
            a.len(),
            l.as_mut_ptr().cast::<c_void>(),
            l.len(),
            std::ptr::null_mut(),
            0,
        )
    };
    assert_eq!(status, TLINALG_OK, "{}", last_error());
    l
}

#[test]
fn an_owned_executor_runs_the_svd_on_its_pool() {
    let (u, s, vt) = svd_with(0);

    let mut exec: isize = 0;
    // SAFETY: a writable out-parameter and a null options pointer.
    let status = unsafe { tcontract_tapp_executor_create_rayon(&mut exec, 2, std::ptr::null()) };
    assert!(TAPP_check_success(status), "{}", last_error());

    let (pu, ps, pvt) = svd_with(exec);
    assert_close_f64(&pu, &u, 1e-12);
    assert_close_f64(&ps, &s, 1e-12);
    assert_close_f64(&pvt, &vt, 1e-12);
    assert!((ps[0] - 3.0).abs() < 1e-12 && (ps[1] - 1.0).abs() < 1e-12);

    let (mut width, mut budget) = (0usize, 0usize);
    // SAFETY: a live handle and two writable out-parameters.
    let status = unsafe { tcontract_tapp_executor_get_threads(exec, &mut width, &mut budget) };
    assert!(TAPP_check_success(status));
    assert_eq!((width, budget), (2, 2));

    // SAFETY: a live handle no call uses afterwards.
    assert!(TAPP_check_success(unsafe { TAPP_destroy_executor(exec) }));
}

#[test]
fn a_host_owned_pool_can_be_lent_to_these_calls() {
    let (u, s, vt) = svd_with(0);

    let mut pool: tcontract_pool = 0;
    // SAFETY: a writable out-parameter and a null options pointer.
    let status = unsafe { tcontract_pool_create_rayon(&mut pool, 2, std::ptr::null()) };
    assert!(TAPP_check_success(status), "{}", last_error());

    let mut exec: isize = 0;
    // SAFETY: a live pool handle and a writable out-parameter.
    let status = unsafe { tcontract_tapp_executor_borrow_pool(&mut exec, pool, 2) };
    assert!(TAPP_check_success(status), "{}", last_error());

    let (pu, ps, pvt) = svd_with(exec);
    assert_close_f64(&pu, &u, 1e-12);
    assert_close_f64(&ps, &s, 1e-12);
    assert_close_f64(&pvt, &vt, 1e-12);
    assert_close_f64(&cholesky_with(exec), &cholesky_with(0), 1e-12);

    // SAFETY: a live handle no call uses afterwards.
    assert!(TAPP_check_success(unsafe { TAPP_destroy_executor(exec) }));
    // SAFETY: the pool is no longer borrowed.
    assert!(TAPP_check_success(unsafe { tcontract_pool_destroy(pool) }));
}

#[test]
fn a_serial_executor_matches_the_sequential_path() {
    let (u, s, vt) = svd_with(0);

    let mut exec: isize = 0;
    // SAFETY: a writable out-parameter.
    assert!(TAPP_check_success(unsafe {
        TAPP_create_executor(&mut exec)
    }));

    let (su, ss, svt) = svd_with(exec);
    assert_close_f64(&su, &u, 1e-12);
    assert_close_f64(&ss, &s, 1e-12);
    assert_close_f64(&svt, &vt, 1e-12);

    // SAFETY: a live handle no call uses afterwards.
    assert!(TAPP_check_success(unsafe { TAPP_destroy_executor(exec) }));
}
