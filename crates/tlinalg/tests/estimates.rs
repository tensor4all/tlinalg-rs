//! Public work queries reuse the scalar weights and allocate no storage.

use num_complex::{Complex32, Complex64};
use tlinalg::householder::ReflectorShape;
use tlinalg::triangular_solve::TriangularSolveFlags;
use tlinalg::FaerScalar;
use tlinalg::{
    cholesky, eig, eigh, full_piv_lu, householder, lu, packed_lu, qr, svd, triangular_solve,
};
use tlinalg_testkit::alloc::{allocations, Counting};

#[global_allocator]
static GLOBAL: Counting = Counting;

fn estimates<T: FaerScalar>(m: usize, n: usize, nrhs: usize) -> [(&'static str, f64, f64); 15] {
    let shape = ReflectorShape {
        rows: m,
        a_cols: n,
        cols: nrhs,
        k: m.min(n),
    };
    let flags = TriangularSolveFlags {
        left_side: true,
        lower: true,
        transpose_a: false,
        unit_diagonal: false,
    };
    [
        (
            "cholesky",
            cholesky::estimate_work::<T>(n),
            cholesky::estimate_payback_work::<T>(n),
        ),
        (
            "eig",
            eig::estimate_work::<T>(n),
            eig::estimate_payback_work::<T>(n),
        ),
        (
            "eigh",
            eigh::estimate_work::<T>(n),
            eigh::estimate_payback_work::<T>(n),
        ),
        (
            "full pivot LU",
            full_piv_lu::estimate_work::<T>(n),
            full_piv_lu::estimate_payback_work::<T>(n),
        ),
        (
            "full pivot solve",
            full_piv_lu::estimate_solve_work::<T>(n, nrhs),
            full_piv_lu::estimate_solve_payback_work::<T>(n, nrhs),
        ),
        (
            "compact factor",
            householder::estimate_work::<T>(m, n),
            householder::estimate_payback_work::<T>(m, n),
        ),
        (
            "apply reflectors",
            householder::estimate_apply_reflectors_work::<T>(shape),
            householder::estimate_apply_reflectors_payback_work::<T>(shape),
        ),
        (
            "LU",
            lu::estimate_work::<T>(m, n),
            lu::estimate_payback_work::<T>(m, n),
        ),
        (
            "solve",
            lu::estimate_solve_work::<T>(n, nrhs),
            lu::estimate_solve_payback_work::<T>(n, nrhs),
        ),
        (
            "packed factor",
            packed_lu::estimate_work::<T>(m, n),
            packed_lu::estimate_payback_work::<T>(m, n),
        ),
        (
            "packed factor solve",
            packed_lu::estimate_factor_solve_work::<T>(n, nrhs),
            packed_lu::estimate_factor_solve_payback_work::<T>(n, nrhs),
        ),
        (
            "packed prepared solve",
            packed_lu::estimate_solve_prepared_work::<T>(n, nrhs),
            packed_lu::estimate_solve_prepared_payback_work::<T>(n, nrhs),
        ),
        (
            "QR",
            qr::estimate_work::<T>(m, n),
            qr::estimate_payback_work::<T>(m, n),
        ),
        (
            "SVD",
            svd::estimate_work::<T>(m, n),
            svd::estimate_payback_work::<T>(m, n),
        ),
        (
            "triangular solve",
            triangular_solve::estimate_work::<T>(n, n, nrhs, flags),
            triangular_solve::estimate_payback_work::<T>(n, n, nrhs, flags),
        ),
    ]
}

#[test]
fn complex_work_is_four_times_real_work() {
    for (m, n, nrhs) in [(13, 7, 3), (7, 13, 5), (0, 0, 0), (1, 1, 1)] {
        for ((name, real, _), (_, complex, _)) in estimates::<f64>(m, n, nrhs)
            .into_iter()
            .zip(estimates::<Complex64>(m, n, nrhs))
        {
            assert_eq!(complex.to_bits(), (4.0 * real).to_bits(), "{name}, {m}x{n}");
        }
    }
}

#[test]
fn payback_work_is_total_times_the_family_share() {
    fn check<T: FaerScalar>() {
        for (name, total, payback) in estimates::<T>(13, 7, 3) {
            let share = match name {
                "QR" => 1.0e-2,
                "eigh" | "SVD" => 6.0e-4,
                _ => 1.0,
            };
            assert_eq!(payback.to_bits(), (total * share).to_bits(), "{name}");
        }
    }
    check::<f64>();
    check::<Complex64>();
}

#[test]
fn empty_shapes_have_zero_work() {
    for (name, total, payback) in estimates::<f64>(0, 0, 0) {
        assert_eq!(total, 0.0, "{name}");
        assert_eq!(payback, 0.0, "{name}");
    }
}

#[test]
fn prepared_solve_does_not_charge_for_factorization() {
    let prepared = packed_lu::estimate_solve_prepared_work::<f64>(6, 3);
    let fused = packed_lu::estimate_factor_solve_work::<f64>(6, 3);
    assert_eq!(fused, prepared + packed_lu::estimate_work::<f64>(6, 6));
    assert_eq!(packed_lu::estimate_solve_prepared_work::<f64>(6, 0), 0.0);
}

#[test]
fn triangular_work_uses_the_solved_rhs_count() {
    let left = TriangularSolveFlags {
        left_side: true,
        lower: true,
        transpose_a: false,
        unit_diagonal: false,
    };
    let right = TriangularSolveFlags {
        left_side: false,
        ..left
    };
    let total = triangular_solve::estimate_work::<f64>(7, 7, 3, left);
    assert_eq!(
        total,
        triangular_solve::estimate_work::<f64>(7, 3, 7, right)
    );
    let other_triangle = TriangularSolveFlags {
        lower: false,
        transpose_a: true,
        unit_diagonal: true,
        ..left
    };
    assert_eq!(
        total,
        triangular_solve::estimate_work::<f64>(7, 7, 3, other_triangle)
    );
    assert_eq!(
        total,
        triangular_solve::estimate_payback_work::<f64>(7, 3, 7, right)
    );
}

#[test]
fn every_work_query_is_allocation_free() {
    let count = allocations(|| {
        let (m, n, nrhs) = std::hint::black_box((13, 7, 3));
        std::hint::black_box(estimates::<f32>(m, n, nrhs));
        std::hint::black_box(estimates::<f64>(m, n, nrhs));
        std::hint::black_box(estimates::<Complex32>(m, n, nrhs));
        std::hint::black_box(estimates::<Complex64>(m, n, nrhs));
    });
    assert_eq!(count, 0);
}
