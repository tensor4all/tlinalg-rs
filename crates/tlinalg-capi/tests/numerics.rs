//! Numerical agreement between the C ABI and `tlinalg` itself, per operation.
//!
//! The ABI calls the provider with `Parallel::Sequential`, so the same input through the C entry
//! point and through the Rust entry point must produce the same numbers. Each test also checks that
//! the size query agrees with what the writing call fills, per item and in total.

mod common;

use std::ffi::c_void;

use common::{
    assert_close_c64, assert_close_f32, assert_close_f64, compact_dims, last_error, Layout,
};
use strided_view::RawStridedRef;
use tlinalg::{cholesky, eigh, full_piv_lu, lu, svd, Op, Parallel};
use tlinalg_capi::*;
use tlinalg_testkit::{Complex32, Complex64};

const TOL: f64 = 1e-10;
const TOL_F32: f32 = 1e-4;
const TOL_C32: f32 = 1e-4;

// ---------------------------------------------------------------- f64

#[test]
fn svd_f64_full_and_economy_batched() {
    let a = [
        1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0, // item 0, 3x2 column-major
        0.0, 1.0, 0.0, 0.0, 1.0, 1.0, // item 1
    ];
    let dims = compact_dims(3, 2, 2);
    let layout = Layout::compact(&dims);
    let desc = layout.as_c();
    let dims_u = layout.dims_usize();
    let strides_i = layout.strides_isize();

    for full in [false, true] {
        let mut sizes = tlinalg_svd_out {
            batch: 0,
            u: 0,
            s: 0,
            vt: 0,
            workspace: 0,
        };
        assert_eq!(
            unsafe { tlinalg_svd_sizes(&desc, i32::from(full), &mut sizes) },
            TLINALG_OK
        );
        assert_eq!(sizes.batch, 2);
        assert_eq!(sizes.workspace, 0);

        let (mut eu, mut es, mut evt) = (Vec::new(), Vec::new(), Vec::new());
        let op = if full { Op::SvdFull } else { Op::Svd };
        svd::svd(
            op,
            RawStridedRef::new(&a, &dims_u, &strides_i, 0).unwrap(),
            full,
            &mut eu,
            &mut es,
            &mut evt,
            Parallel::Sequential,
        )
        .unwrap();

        // The query agrees with what the provider wrote, per item and in total.
        assert_eq!(sizes.u * sizes.batch, eu.len());
        assert_eq!(sizes.s * sizes.batch, es.len());
        assert_eq!(sizes.vt * sizes.batch, evt.len());
        let (u_cols, v_rows) = if full { (3, 2) } else { (2, 2) };
        assert_eq!((sizes.u, sizes.s, sizes.vt), (3 * u_cols, 2, v_rows * 2));

        let (mut u, mut s, mut vt) = (
            vec![0.0f64; eu.len()],
            vec![0.0f64; es.len()],
            vec![0.0f64; evt.len()],
        );
        let status = unsafe {
            tlinalg_svd_into(
                &desc,
                TLINALG_F64,
                0,
                a.as_ptr().cast::<c_void>(),
                a.len(),
                i32::from(full),
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
        assert_close_f64(&u, &eu, TOL);
        assert_close_f64(&s, &es, TOL);
        assert_close_f64(&vt, &evt, TOL);
    }
}

#[test]
fn svd_values_f64_matches_tlinalg() {
    let a = [
        1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0,
    ];
    let dims = compact_dims(3, 2, 2);
    let layout = Layout::compact(&dims);
    let desc = layout.as_c();
    let dims_u = layout.dims_usize();
    let strides_i = layout.strides_isize();

    let mut sizes = tlinalg_svd_values_out {
        batch: 0,
        s: 0,
        workspace: 0,
    };
    assert_eq!(
        unsafe { tlinalg_svd_values_sizes(&desc, &mut sizes) },
        TLINALG_OK
    );
    assert_eq!(sizes.batch, 2);

    let mut es = Vec::new();
    svd::svd_values(
        Op::SvdValues,
        RawStridedRef::new(&a, &dims_u, &strides_i, 0).unwrap(),
        &mut es,
        Parallel::Sequential,
    )
    .unwrap();
    assert_eq!(sizes.s * sizes.batch, es.len());

    let mut s = vec![0.0f64; es.len()];
    assert_eq!(
        unsafe {
            tlinalg_svd_values_into(
                &desc,
                TLINALG_F64,
                0,
                a.as_ptr().cast::<c_void>(),
                a.len(),
                s.as_mut_ptr().cast::<c_void>(),
                s.len(),
                std::ptr::null_mut(),
                0,
            )
        },
        TLINALG_OK,
        "{}",
        last_error()
    );
    assert_close_f64(&s, &es, TOL);
}

#[test]
fn eigh_f64_matches_tlinalg() {
    let a = [2.0f64, 0.0, 0.0, 3.0, 4.0, 1.0, 1.0, 4.0];
    let dims = compact_dims(2, 2, 2);
    let layout = Layout::compact(&dims);
    let desc = layout.as_c();
    let dims_u = layout.dims_usize();
    let strides_i = layout.strides_isize();

    let mut sizes = tlinalg_eigh_out {
        batch: 0,
        values: 0,
        vectors: 0,
        workspace: 0,
    };
    assert_eq!(unsafe { tlinalg_eigh_sizes(&desc, &mut sizes) }, TLINALG_OK);
    assert_eq!((sizes.batch, sizes.values, sizes.vectors), (2, 2, 4));

    let (mut ev, mut ew) = (Vec::new(), Vec::new());
    eigh::eigh(
        Op::Eigh,
        RawStridedRef::new(&a, &dims_u, &strides_i, 0).unwrap(),
        &mut ev,
        &mut ew,
        Parallel::Sequential,
    )
    .unwrap();
    assert_eq!(sizes.values * sizes.batch, ev.len());
    assert_eq!(sizes.vectors * sizes.batch, ew.len());

    let (mut v, mut w) = (vec![0.0f64; ev.len()], vec![0.0f64; ew.len()]);
    assert_eq!(
        unsafe {
            tlinalg_eigh_into(
                &desc,
                TLINALG_F64,
                0,
                a.as_ptr().cast::<c_void>(),
                a.len(),
                v.as_mut_ptr().cast::<c_void>(),
                v.len(),
                w.as_mut_ptr().cast::<c_void>(),
                w.len(),
                std::ptr::null_mut(),
                0,
            )
        },
        TLINALG_OK,
        "{}",
        last_error()
    );
    assert_close_f64(&v, &ev, TOL);
    assert_close_f64(&w, &ew, TOL);
}

#[test]
fn eigh_values_f64_matches_tlinalg() {
    let a = [2.0f64, 0.0, 0.0, 3.0, 4.0, 1.0, 1.0, 4.0];
    let dims = compact_dims(2, 2, 2);
    let layout = Layout::compact(&dims);
    let desc = layout.as_c();
    let dims_u = layout.dims_usize();
    let strides_i = layout.strides_isize();

    let mut sizes = tlinalg_eigh_values_out {
        batch: 0,
        values: 0,
        workspace: 0,
    };
    assert_eq!(
        unsafe { tlinalg_eigh_values_sizes(&desc, &mut sizes) },
        TLINALG_OK
    );

    let mut ev = Vec::new();
    eigh::eigh_values(
        Op::EighValues,
        RawStridedRef::new(&a, &dims_u, &strides_i, 0).unwrap(),
        &mut ev,
        Parallel::Sequential,
    )
    .unwrap();
    assert_eq!(sizes.values * sizes.batch, ev.len());

    let mut v = vec![0.0f64; ev.len()];
    assert_eq!(
        unsafe {
            tlinalg_eigh_values_into(
                &desc,
                TLINALG_F64,
                0,
                a.as_ptr().cast::<c_void>(),
                a.len(),
                v.as_mut_ptr().cast::<c_void>(),
                v.len(),
                std::ptr::null_mut(),
                0,
            )
        },
        TLINALG_OK,
        "{}",
        last_error()
    );
    assert_close_f64(&v, &ev, TOL);
}

#[test]
fn cholesky_f64_matches_tlinalg() {
    let a = [4.0f64, 2.0, 2.0, 3.0, 9.0, 0.0, 0.0, 4.0];
    let dims = compact_dims(2, 2, 2);
    let layout = Layout::compact(&dims);
    let desc = layout.as_c();
    let dims_u = layout.dims_usize();
    let strides_i = layout.strides_isize();

    let mut sizes = tlinalg_cholesky_out {
        batch: 0,
        l: 0,
        workspace: 0,
    };
    assert_eq!(
        unsafe { tlinalg_cholesky_sizes(&desc, &mut sizes) },
        TLINALG_OK
    );
    assert_eq!((sizes.batch, sizes.l), (2, 4));

    let mut el = Vec::new();
    cholesky::cholesky(
        Op::Cholesky,
        RawStridedRef::new(&a, &dims_u, &strides_i, 0).unwrap(),
        &mut el,
        Parallel::Sequential,
    )
    .unwrap();
    assert_eq!(sizes.l * sizes.batch, el.len());

    let mut l = vec![0.0f64; el.len()];
    assert_eq!(
        unsafe {
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
        },
        TLINALG_OK,
        "{}",
        last_error()
    );
    assert_close_f64(&l, &el, TOL);
}

#[test]
fn lu_f64_rectangular_batched() {
    let a = [
        1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0, // item 0, 3x2
        2.0, 0.0, 0.0, 1.0, 1.0, 1.0, // item 1
    ];
    let dims = compact_dims(3, 2, 2);
    let layout = Layout::compact(&dims);
    let desc = layout.as_c();
    let dims_u = layout.dims_usize();
    let strides_i = layout.strides_isize();

    let mut sizes = tlinalg_lu_out {
        batch: 0,
        p: 0,
        l: 0,
        u: 0,
        parity: 0,
        workspace: 0,
    };
    assert_eq!(unsafe { tlinalg_lu_sizes(&desc, &mut sizes) }, TLINALG_OK);
    assert_eq!(
        (sizes.batch, sizes.p, sizes.l, sizes.u, sizes.parity),
        (2, 9, 6, 4, 1)
    );

    let (mut ep, mut el, mut eu, mut epar) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    lu::lu(
        Op::Lu,
        RawStridedRef::new(&a, &dims_u, &strides_i, 0).unwrap(),
        lu::LuFactors {
            p: &mut ep,
            l: &mut el,
            u: &mut eu,
            parity: &mut epar,
        },
        Parallel::Sequential,
    )
    .unwrap();
    assert_eq!(sizes.p * sizes.batch, ep.len());
    assert_eq!(sizes.l * sizes.batch, el.len());
    assert_eq!(sizes.u * sizes.batch, eu.len());
    assert_eq!(sizes.parity * sizes.batch, epar.len());

    let (mut p, mut l, mut u, mut par) = (
        vec![0.0f64; ep.len()],
        vec![0.0f64; el.len()],
        vec![0.0f64; eu.len()],
        vec![0.0f64; epar.len()],
    );
    assert_eq!(
        unsafe {
            tlinalg_lu_into(
                &desc,
                TLINALG_F64,
                0,
                a.as_ptr().cast::<c_void>(),
                a.len(),
                p.as_mut_ptr().cast::<c_void>(),
                p.len(),
                l.as_mut_ptr().cast::<c_void>(),
                l.len(),
                u.as_mut_ptr().cast::<c_void>(),
                u.len(),
                par.as_mut_ptr().cast::<c_void>(),
                par.len(),
                std::ptr::null_mut(),
                0,
            )
        },
        TLINALG_OK,
        "{}",
        last_error()
    );
    assert_close_f64(&p, &ep, TOL);
    assert_close_f64(&l, &el, TOL);
    assert_close_f64(&u, &eu, TOL);
    assert_close_f64(&par, &epar, TOL);
}

#[test]
fn full_piv_lu_f64_matches_tlinalg() {
    let a = [1.0f64, 2.0, 3.0, 4.0, 0.0, 1.0, 1.0, 0.0];
    let dims = compact_dims(2, 2, 2);
    let layout = Layout::compact(&dims);
    let desc = layout.as_c();
    let dims_u = layout.dims_usize();
    let strides_i = layout.strides_isize();

    let mut sizes = tlinalg_full_piv_lu_out {
        batch: 0,
        p: 0,
        l: 0,
        u: 0,
        q: 0,
        parity: 0,
        workspace: 0,
    };
    assert_eq!(
        unsafe { tlinalg_full_piv_lu_sizes(&desc, &mut sizes) },
        TLINALG_OK
    );
    assert_eq!((sizes.batch, sizes.p, sizes.parity), (2, 4, 1));

    let (mut ep, mut el, mut eu, mut eq, mut epar) =
        (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new());
    full_piv_lu::full_piv_lu(
        Op::FullPivLu,
        RawStridedRef::new(&a, &dims_u, &strides_i, 0).unwrap(),
        full_piv_lu::FullPivLuFactors {
            p: &mut ep,
            l: &mut el,
            u: &mut eu,
            q: &mut eq,
            parity: &mut epar,
        },
        Parallel::Sequential,
    )
    .unwrap();

    let (mut p, mut l, mut u, mut q, mut par) = (
        vec![0.0f64; ep.len()],
        vec![0.0f64; el.len()],
        vec![0.0f64; eu.len()],
        vec![0.0f64; eq.len()],
        vec![0.0f64; epar.len()],
    );
    assert_eq!(
        unsafe {
            tlinalg_full_piv_lu_into(
                &desc,
                TLINALG_F64,
                0,
                a.as_ptr().cast::<c_void>(),
                a.len(),
                p.as_mut_ptr().cast::<c_void>(),
                p.len(),
                l.as_mut_ptr().cast::<c_void>(),
                l.len(),
                u.as_mut_ptr().cast::<c_void>(),
                u.len(),
                q.as_mut_ptr().cast::<c_void>(),
                q.len(),
                par.as_mut_ptr().cast::<c_void>(),
                par.len(),
                std::ptr::null_mut(),
                0,
            )
        },
        TLINALG_OK,
        "{}",
        last_error()
    );
    assert_close_f64(&p, &ep, TOL);
    assert_close_f64(&l, &el, TOL);
    assert_close_f64(&u, &eu, TOL);
    assert_close_f64(&q, &eq, TOL);
    assert_close_f64(&par, &epar, TOL);
}

// ---------------------------------------------------------------- c64

fn c64_matrix() -> Vec<Complex64> {
    // [[1+2i, 3], [0, 1-i]] column-major, twice.
    let item = [
        Complex64::new(1.0, 2.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(3.0, 0.0),
        Complex64::new(1.0, -1.0),
    ];
    item.into_iter().chain(item).collect()
}

#[test]
fn svd_c64_full_and_economy_batched() {
    let a = c64_matrix();
    let dims = compact_dims(2, 2, 2);
    let layout = Layout::compact(&dims);
    let desc = layout.as_c();
    let dims_u = layout.dims_usize();
    let strides_i = layout.strides_isize();

    for full in [false, true] {
        let mut sizes = tlinalg_svd_out {
            batch: 0,
            u: 0,
            s: 0,
            vt: 0,
            workspace: 0,
        };
        assert_eq!(
            unsafe { tlinalg_svd_sizes(&desc, i32::from(full), &mut sizes) },
            TLINALG_OK
        );

        let (mut eu, mut es, mut evt) = (Vec::new(), Vec::new(), Vec::new());
        let op = if full { Op::SvdFull } else { Op::Svd };
        svd::svd(
            op,
            RawStridedRef::new(&a, &dims_u, &strides_i, 0).unwrap(),
            full,
            &mut eu,
            &mut es,
            &mut evt,
            Parallel::Sequential,
        )
        .unwrap();
        // `svd` writes the singular values in the scalar type, zero imaginary part.
        assert!(es.iter().all(|z| z.im == 0.0));

        let (mut u, mut s, mut vt) = (
            vec![Complex64::new(0.0, 0.0); eu.len()],
            vec![Complex64::new(0.0, 0.0); es.len()],
            vec![Complex64::new(0.0, 0.0); evt.len()],
        );
        let status = unsafe {
            tlinalg_svd_into(
                &desc,
                TLINALG_C64,
                0,
                a.as_ptr().cast::<c_void>(),
                a.len(),
                i32::from(full),
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
        assert_close_c64(&u, &eu, TOL);
        assert_close_c64(&s, &es, TOL);
        assert_close_c64(&vt, &evt, TOL);
    }
}

#[test]
fn svd_values_c64_writes_real_scalars() {
    let a = c64_matrix();
    let dims = compact_dims(2, 2, 2);
    let layout = Layout::compact(&dims);
    let desc = layout.as_c();
    let dims_u = layout.dims_usize();
    let strides_i = layout.strides_isize();

    let mut sizes = tlinalg_svd_values_out {
        batch: 0,
        s: 0,
        workspace: 0,
    };
    assert_eq!(
        unsafe { tlinalg_svd_values_sizes(&desc, &mut sizes) },
        TLINALG_OK
    );

    let mut es = Vec::new();
    svd::svd_values(
        Op::SvdValues,
        RawStridedRef::new(&a, &dims_u, &strides_i, 0).unwrap(),
        &mut es,
        Parallel::Sequential,
    )
    .unwrap();
    assert_eq!(sizes.s * sizes.batch, es.len());

    // Real scalars: the buffer is f64, not Complex64.
    let mut s = vec![0.0f64; es.len()];
    assert_eq!(
        unsafe {
            tlinalg_svd_values_into(
                &desc,
                TLINALG_C64,
                0,
                a.as_ptr().cast::<c_void>(),
                a.len(),
                s.as_mut_ptr().cast::<c_void>(),
                s.len(),
                std::ptr::null_mut(),
                0,
            )
        },
        TLINALG_OK,
        "{}",
        last_error()
    );
    assert_close_f64(&s, &es, TOL);
}

fn c64_hermitian() -> Vec<Complex64> {
    // [[2, 0], [-i, 3]] column-major, twice; eigh reads the lower triangle.
    let item = [
        Complex64::new(2.0, 0.0),
        Complex64::new(0.0, -1.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(3.0, 0.0),
    ];
    item.into_iter().chain(item).collect()
}

#[test]
fn eigh_c64_matches_tlinalg() {
    let a = c64_hermitian();
    let dims = compact_dims(2, 2, 2);
    let layout = Layout::compact(&dims);
    let desc = layout.as_c();
    let dims_u = layout.dims_usize();
    let strides_i = layout.strides_isize();

    let (mut ev, mut ew) = (Vec::new(), Vec::new());
    eigh::eigh(
        Op::Eigh,
        RawStridedRef::new(&a, &dims_u, &strides_i, 0).unwrap(),
        &mut ev,
        &mut ew,
        Parallel::Sequential,
    )
    .unwrap();
    assert!(ev.iter().all(|z| z.im == 0.0));

    let (mut v, mut w) = (
        vec![Complex64::new(0.0, 0.0); ev.len()],
        vec![Complex64::new(0.0, 0.0); ew.len()],
    );
    assert_eq!(
        unsafe {
            tlinalg_eigh_into(
                &desc,
                TLINALG_C64,
                0,
                a.as_ptr().cast::<c_void>(),
                a.len(),
                v.as_mut_ptr().cast::<c_void>(),
                v.len(),
                w.as_mut_ptr().cast::<c_void>(),
                w.len(),
                std::ptr::null_mut(),
                0,
            )
        },
        TLINALG_OK,
        "{}",
        last_error()
    );
    assert_close_c64(&v, &ev, TOL);
    assert_close_c64(&w, &ew, TOL);
}

#[test]
fn eigh_values_c64_writes_real_scalars() {
    let a = c64_hermitian();
    let dims = compact_dims(2, 2, 2);
    let layout = Layout::compact(&dims);
    let desc = layout.as_c();
    let dims_u = layout.dims_usize();
    let strides_i = layout.strides_isize();

    let mut ev = Vec::new();
    eigh::eigh_values(
        Op::EighValues,
        RawStridedRef::new(&a, &dims_u, &strides_i, 0).unwrap(),
        &mut ev,
        Parallel::Sequential,
    )
    .unwrap();

    let mut v = vec![0.0f64; ev.len()];
    assert_eq!(
        unsafe {
            tlinalg_eigh_values_into(
                &desc,
                TLINALG_C64,
                0,
                a.as_ptr().cast::<c_void>(),
                a.len(),
                v.as_mut_ptr().cast::<c_void>(),
                v.len(),
                std::ptr::null_mut(),
                0,
            )
        },
        TLINALG_OK,
        "{}",
        last_error()
    );
    assert_close_f64(&v, &ev, TOL);
}

fn c64_hpd() -> Vec<Complex64> {
    // [[4, 0], [-i, 5]] lower triangle, twice.
    let item = [
        Complex64::new(4.0, 0.0),
        Complex64::new(0.0, -1.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(5.0, 0.0),
    ];
    item.into_iter().chain(item).collect()
}

#[test]
fn cholesky_c64_matches_tlinalg() {
    let a = c64_hpd();
    let dims = compact_dims(2, 2, 2);
    let layout = Layout::compact(&dims);
    let desc = layout.as_c();
    let dims_u = layout.dims_usize();
    let strides_i = layout.strides_isize();

    let mut el = Vec::new();
    cholesky::cholesky(
        Op::Cholesky,
        RawStridedRef::new(&a, &dims_u, &strides_i, 0).unwrap(),
        &mut el,
        Parallel::Sequential,
    )
    .unwrap();

    let mut l = vec![Complex64::new(0.0, 0.0); el.len()];
    assert_eq!(
        unsafe {
            tlinalg_cholesky_into(
                &desc,
                TLINALG_C64,
                0,
                a.as_ptr().cast::<c_void>(),
                a.len(),
                l.as_mut_ptr().cast::<c_void>(),
                l.len(),
                std::ptr::null_mut(),
                0,
            )
        },
        TLINALG_OK,
        "{}",
        last_error()
    );
    assert_close_c64(&l, &el, TOL);
}

fn c64_general() -> Vec<Complex64> {
    // [[1+i, 2], [0, 1]] column-major, twice.
    let item = [
        Complex64::new(1.0, 1.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(2.0, 0.0),
        Complex64::new(1.0, 0.0),
    ];
    item.into_iter().chain(item).collect()
}

#[test]
fn lu_c64_matches_tlinalg() {
    let a = c64_general();
    let dims = compact_dims(2, 2, 2);
    let layout = Layout::compact(&dims);
    let desc = layout.as_c();
    let dims_u = layout.dims_usize();
    let strides_i = layout.strides_isize();

    let (mut ep, mut el, mut eu, mut epar) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    lu::lu(
        Op::Lu,
        RawStridedRef::new(&a, &dims_u, &strides_i, 0).unwrap(),
        lu::LuFactors {
            p: &mut ep,
            l: &mut el,
            u: &mut eu,
            parity: &mut epar,
        },
        Parallel::Sequential,
    )
    .unwrap();

    let (mut p, mut l, mut u, mut par) = (
        vec![Complex64::new(0.0, 0.0); ep.len()],
        vec![Complex64::new(0.0, 0.0); el.len()],
        vec![Complex64::new(0.0, 0.0); eu.len()],
        vec![Complex64::new(0.0, 0.0); epar.len()],
    );
    assert_eq!(
        unsafe {
            tlinalg_lu_into(
                &desc,
                TLINALG_C64,
                0,
                a.as_ptr().cast::<c_void>(),
                a.len(),
                p.as_mut_ptr().cast::<c_void>(),
                p.len(),
                l.as_mut_ptr().cast::<c_void>(),
                l.len(),
                u.as_mut_ptr().cast::<c_void>(),
                u.len(),
                par.as_mut_ptr().cast::<c_void>(),
                par.len(),
                std::ptr::null_mut(),
                0,
            )
        },
        TLINALG_OK,
        "{}",
        last_error()
    );
    assert_close_c64(&p, &ep, TOL);
    assert_close_c64(&l, &el, TOL);
    assert_close_c64(&u, &eu, TOL);
    assert_close_c64(&par, &epar, TOL);
}

#[test]
fn full_piv_lu_c64_matches_tlinalg() {
    let a = c64_general();
    let dims = compact_dims(2, 2, 2);
    let layout = Layout::compact(&dims);
    let desc = layout.as_c();
    let dims_u = layout.dims_usize();
    let strides_i = layout.strides_isize();

    let (mut ep, mut el, mut eu, mut eq, mut epar) =
        (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new());
    full_piv_lu::full_piv_lu(
        Op::FullPivLu,
        RawStridedRef::new(&a, &dims_u, &strides_i, 0).unwrap(),
        full_piv_lu::FullPivLuFactors {
            p: &mut ep,
            l: &mut el,
            u: &mut eu,
            q: &mut eq,
            parity: &mut epar,
        },
        Parallel::Sequential,
    )
    .unwrap();

    let (mut p, mut l, mut u, mut q, mut par) = (
        vec![Complex64::new(0.0, 0.0); ep.len()],
        vec![Complex64::new(0.0, 0.0); el.len()],
        vec![Complex64::new(0.0, 0.0); eu.len()],
        vec![Complex64::new(0.0, 0.0); eq.len()],
        vec![Complex64::new(0.0, 0.0); epar.len()],
    );
    assert_eq!(
        unsafe {
            tlinalg_full_piv_lu_into(
                &desc,
                TLINALG_C64,
                0,
                a.as_ptr().cast::<c_void>(),
                a.len(),
                p.as_mut_ptr().cast::<c_void>(),
                p.len(),
                l.as_mut_ptr().cast::<c_void>(),
                l.len(),
                u.as_mut_ptr().cast::<c_void>(),
                u.len(),
                q.as_mut_ptr().cast::<c_void>(),
                q.len(),
                par.as_mut_ptr().cast::<c_void>(),
                par.len(),
                std::ptr::null_mut(),
                0,
            )
        },
        TLINALG_OK,
        "{}",
        last_error()
    );
    assert_close_c64(&p, &ep, TOL);
    assert_close_c64(&l, &el, TOL);
    assert_close_c64(&u, &eu, TOL);
    assert_close_c64(&q, &eq, TOL);
    assert_close_c64(&par, &epar, TOL);
}

// ------------------------------------------------------- f32 / c32 dispatch

#[test]
fn svd_f32_dispatch_matches_tlinalg() {
    let a = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let dims = compact_dims(3, 2, 1);
    let layout = Layout::compact(&dims);
    let desc = layout.as_c();
    let dims_u = layout.dims_usize();
    let strides_i = layout.strides_isize();

    let (mut eu, mut es, mut evt) = (Vec::new(), Vec::new(), Vec::new());
    svd::svd(
        Op::Svd,
        RawStridedRef::new(&a, &dims_u, &strides_i, 0).unwrap(),
        false,
        &mut eu,
        &mut es,
        &mut evt,
        Parallel::Sequential,
    )
    .unwrap();

    let (mut u, mut s, mut vt) = (
        vec![0.0f32; eu.len()],
        vec![0.0f32; es.len()],
        vec![0.0f32; evt.len()],
    );
    assert_eq!(
        unsafe {
            tlinalg_svd_into(
                &desc,
                TLINALG_F32,
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
    assert_close_f32(&u, &eu, TOL_F32);
    assert_close_f32(&s, &es, TOL_F32);
    assert_close_f32(&vt, &evt, TOL_F32);
}

#[test]
fn eigh_values_c32_dispatch_matches_tlinalg() {
    let a = [
        Complex32::new(2.0, 0.0),
        Complex32::new(0.0, -1.0),
        Complex32::new(0.0, 0.0),
        Complex32::new(3.0, 0.0),
    ];
    let dims = compact_dims(2, 2, 1);
    let layout = Layout::compact(&dims);
    let desc = layout.as_c();
    let dims_u = layout.dims_usize();
    let strides_i = layout.strides_isize();

    let mut ev = Vec::new();
    eigh::eigh_values(
        Op::EighValues,
        RawStridedRef::new(&a, &dims_u, &strides_i, 0).unwrap(),
        &mut ev,
        Parallel::Sequential,
    )
    .unwrap();

    let mut v = vec![0.0f32; ev.len()];
    assert_eq!(
        unsafe {
            tlinalg_eigh_values_into(
                &desc,
                TLINALG_C32,
                0,
                a.as_ptr().cast::<c_void>(),
                a.len(),
                v.as_mut_ptr().cast::<c_void>(),
                v.len(),
                std::ptr::null_mut(),
                0,
            )
        },
        TLINALG_OK,
        "{}",
        last_error()
    );
    assert_close_f32(&v, &ev, TOL_C32);
}
