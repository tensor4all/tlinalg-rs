//! Steady-state heap allocations per batched call, counted with a global allocator.
//!
//! The output vectors are reused with enough capacity, as a host recycling pooled buffers would.
//! The batch driver allocates nothing on one lane; what remains is each family's faer lane
//! scratch, independent of the batch size. This test pins those counts for one lane so a
//! regression shows up as a number.

use strided_view::{RawStridedMut, RawStridedRef};
use tlinalg::triangular_solve::{triangular_solve, TriangularSolveFlags};
use tlinalg::{Op, Parallel};
use tlinalg_testkit::alloc::{steady, Counting};

#[global_allocator]
static GLOBAL: Counting = Counting;

/// `batch` compact, well-conditioned `n x n` matrices.
fn batch_of(n: usize, batch: usize) -> Vec<f64> {
    (0..n * n * batch)
        .map(|i| {
            let (row, col) = (i % n, (i / n) % n);
            if row == col {
                4.0 + row as f64
            } else {
                0.1
            }
        })
        .collect()
}

/// Steady-state allocations per call of each family, one sequential lane, `n = 6`.
fn family_counts(batch: usize) -> [(&'static str, usize); 11] {
    let n = 6;
    let a = batch_of(n, batch);
    let (dims, strides) = ([n, n, batch], [1, n as isize, (n * n) as isize]);
    let view = RawStridedRef::new(&a, &dims, &strides, 0).unwrap();
    let seq = Parallel::Sequential;
    let cap = n * n * batch;
    let (mut v1, mut v2, mut v3) = (
        Vec::with_capacity(cap),
        Vec::with_capacity(cap),
        Vec::with_capacity(cap),
    );
    let cholesky = steady(|| {
        tlinalg::cholesky::cholesky(Op::Cholesky, view, &mut v1, seq).unwrap();
    });
    let qr = steady(|| tlinalg::qr::qr(Op::Qr, view, &mut v1, &mut v2, seq).unwrap());
    let eigh = steady(|| {
        tlinalg::eigh::eigh(Op::Eigh, view, &mut v1, &mut v2, seq).unwrap();
    });
    let svd = steady(|| {
        tlinalg::svd::svd(Op::Svd, view, false, &mut v1, &mut v2, &mut v3, seq).unwrap();
    });
    let (mut lu, mut piv, mut parity) = (a.clone(), vec![0; n * batch], vec![0.0; batch]);
    let packed_lu = steady(|| {
        lu.copy_from_slice(&a);
        tlinalg::packed_lu::factor(Op::LuFactor, n, n, &mut lu, &mut piv, &mut parity, seq)
            .unwrap();
    });
    let mut perm = Vec::with_capacity(n * batch);
    let rrqr = steady(|| {
        tlinalg::qr::rank_revealing_qr(Op::RankRevealingQr, view, &mut v1, &mut v2, &mut perm, seq)
            .unwrap();
    });
    let b = batch_of(n, batch);
    let b_view = RawStridedRef::new(&b, &dims, &strides, 0).unwrap();
    let mut x = vec![0.0; cap];
    let solve = steady(|| {
        tlinalg::lu::solve(
            Op::Solve,
            view,
            Some(b_view),
            RawStridedMut::new(&mut x, &dims, &strides, 0).unwrap(),
            false,
            seq,
        )
        .unwrap();
    });
    let full_piv_lu_solve = steady(|| {
        tlinalg::full_piv_lu::full_piv_lu_solve(
            Op::FullPivLuSolve,
            view,
            b_view,
            false,
            &mut v1,
            seq,
        )
        .unwrap();
    });
    let (mut state, mut coeff) = (a.clone(), Vec::with_capacity(n * batch));
    let compact_factor = steady(|| {
        state.copy_from_slice(&a);
        tlinalg::householder::compact_factor(
            Op::HouseholderQr,
            n,
            n,
            batch,
            &mut state,
            &mut coeff,
            seq,
        )
        .unwrap();
    });
    [
        ("compact_factor", compact_factor),
        ("cholesky", cholesky),
        ("rank_revealing_qr", rrqr),
        ("solve", solve),
        ("full_piv_lu_solve", full_piv_lu_solve),
        ("qr", qr),
        ("eigh", eigh),
        ("svd", svd),
        ("packed_lu factor", packed_lu),
        ("triangular_solve left", triangular(true, batch)),
        ("triangular_solve right", triangular(false, batch)),
    ]
}

/// Allocations made by one triangular solve over `batch` systems on one sequential lane.
fn triangular(left_side: bool, batch: usize) -> usize {
    let n = 6;
    let nrhs = 3;
    let a: Vec<f64> = (0..n * n)
        .map(|i| if i % (n + 1) == 0 { 4.0 } else { 0.1 })
        .collect();
    let (rows, cols) = if left_side { (n, nrhs) } else { (nrhs, n) };
    let b = vec![1.0_f64; rows * cols * batch];
    let flags = TriangularSolveFlags {
        left_side,
        lower: true,
        transpose_a: false,
        unit_diagonal: false,
    };
    let (a_dims, a_strides) = ([n, n, batch], [1, n as isize, 0]);
    let (b_dims, b_strides) = (
        [rows, cols, batch],
        [1, rows as isize, (rows * cols) as isize],
    );
    let a_view = RawStridedRef::new(&a, &a_dims, &a_strides, 0).unwrap();
    let b_view = RawStridedRef::new(&b, &b_dims, &b_strides, 0).unwrap();
    let mut x = Vec::with_capacity(rows * cols * batch);
    steady(|| {
        triangular_solve(
            Op::TriangularSolve,
            a_view,
            b_view,
            flags,
            &mut x,
            Parallel::Sequential,
        )
        .unwrap();
    })
}

/// The driver itself allocates nothing on one lane; what remains is each family's faer lane
/// scratch, built once per call and independent of the batch size.
#[test]
fn per_call_allocations_do_not_grow_with_the_batch() {
    let one = family_counts(1);
    let many = family_counts(64);
    for ((name, at_one), (_, at_many)) in one.iter().zip(&many) {
        println!("{name}: {at_one} (batch 1), {at_many} (batch 64)");
        assert_eq!(at_one, at_many, "{name}: allocations grow with the batch");
    }
    // Each count is at most the pre-extraction route's native allocations for the same call
    // (rank_revealing_qr was 7, solve 5, full_piv_lu_solve 7). The direct `svd` `U` and `eigh`
    // eigenvector outputs removed the two lane buffers that used to be copied out, so `eigh` and
    // `svd` each dropped by one (`svd` also lost its values-only `U` buffer).
    let expected = [
        ("compact_factor", 2),
        ("cholesky", 1),
        ("rank_revealing_qr", 6),
        ("solve", 5),
        ("full_piv_lu_solve", 7),
        ("qr", 5),
        ("eigh", 2),
        ("svd", 3),
        ("packed_lu factor", 5),
        ("triangular_solve left", 0),
        ("triangular_solve right", 0),
    ];
    assert_eq!(one, expected);
}
