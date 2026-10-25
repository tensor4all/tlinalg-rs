//! Behavioural tests for the ported packed-LU kernels.
//!
//! These do not compare against the implementation the code came from (that A/B belongs to the host,
//! which has both), but they pin the format and the semantics the host depends on: the packed
//! layout, the one-based swap sequence, the parity, every solve flag combination, the fused path,
//! the singular rule, and the chunk contract.

use num_complex::{Complex32, Complex64};
use strided_view::RawStridedRef;
use tcontract_exec::Exec;
use tlinalg::packed_lu::{
    factor as factor_chunk, factor_solve as factor_solve_chunk, solve_prepared,
};
use tlinalg::{Error, FaerScalar, Op};

// The prepared-solve shim builds descriptors from the tests' compact buffers.

#[allow(clippy::too_many_arguments)]
fn solve_prepared_chunk<T: FaerScalar>(
    op: Op,
    n: usize,
    nrhs: usize,
    packed_lu: &[T],
    pivots: &[i32],
    output: &mut [T],
    transpose_a: bool,
    conjugate_a: bool,
    par: &mut Exec<'_>,
) -> tlinalg::Result<()> {
    let batch = if n == 0 { 0 } else { packed_lu.len() / (n * n) };
    let lu_dims = [n, n, batch];
    let lu_strides = [1, n as isize, (n * n) as isize];
    let piv_dims = [n, batch];
    let piv_strides = [1, n as isize];
    solve_prepared(
        op,
        RawStridedRef::new(packed_lu, &lu_dims, &lu_strides, 0).unwrap(),
        RawStridedRef::new(pivots, &piv_dims, &piv_strides, 0).unwrap(),
        nrhs,
        output,
        transpose_a,
        conjugate_a,
        par,
    )
}

/// Compact column-major `n x n` matrix chosen so partial pivoting actually swaps: the leading entry
/// is zero, but the rest of the first column is not, so the matrix stays nonsingular.
fn matrix(n: usize) -> Vec<f64> {
    (0..n * n)
        .map(|index| {
            let row = index % n;
            let col = index / n;
            if row == 0 && col == 0 && n > 1 {
                0.0
            } else if row == col {
                5.0
            } else {
                (((row * 3 + col * 7) % 5) as f64) * 0.25 + 0.5
            }
        })
        .collect()
}

/// A second, different matrix of the same shape.
fn matrix_b(n: usize) -> Vec<f64> {
    (0..n * n)
        .map(|index| {
            let row = index % n;
            let col = index / n;
            if row == col {
                4.0 + col as f64
            } else {
                (((row * 5 + col * 2) % 7) as f64) * 0.2 + 0.3
            }
        })
        .collect()
}

/// `A x` for a compact column-major `n x n` matrix.
fn matvec<T: Copy + core::ops::Mul<Output = T> + core::ops::Add<Output = T> + Default>(
    a: &[T],
    n: usize,
    x: &[T],
) -> Vec<T> {
    let mut out = vec![T::default(); n];
    for row in 0..n {
        for col in 0..n {
            out[row] = out[row] + a[row + col * n] * x[col];
        }
    }
    out
}

/// `A^T x`, optionally conjugating `A` first.
fn matvec_transposed(
    a: &[Complex64],
    n: usize,
    x: &[Complex64],
    conjugate: bool,
) -> Vec<Complex64> {
    let mut out = vec![Complex64::new(0.0, 0.0); n];
    for row in 0..n {
        for col in 0..n {
            let entry = a[col + row * n];
            let entry = if conjugate { entry.conj() } else { entry };
            out[row] += entry * x[col];
        }
    }
    out
}

/// Undo a one-based swap sequence on the rows of a compact column-major block.
fn undo_swaps(a: &mut [f64], n: usize, ipiv: &[i32]) {
    for (step, &pivot) in ipiv.iter().enumerate().rev() {
        let pivot = pivot as usize - 1;
        if pivot != step {
            for col in 0..n {
                a.swap(step + col * n, pivot + col * n);
            }
        }
    }
}

/// `L * U` from the packed factor, where `L` is unit lower and `U` is upper.
fn reconstruct_lu(packed: &[f64], n: usize) -> Vec<f64> {
    let mut lu = vec![0.0; n * n];
    for row in 0..n {
        for col in 0..n {
            let k_max = row.min(col);
            let mut acc = 0.0;
            for k in 0..=k_max {
                let l = if row == k {
                    1.0
                } else if row > k {
                    packed[row + k * n]
                } else {
                    0.0
                };
                let u = if k <= col { packed[k + col * n] } else { 0.0 };
                acc += l * u;
            }
            lu[row + col * n] = acc;
        }
    }
    lu
}

/// The `U` diagonal of a packed factor, for checking that the test matrix is nonsingular.
fn u_diagonal(packed: &[f64], n: usize) -> Vec<f64> {
    (0..n).map(|i| packed[i + i * n]).collect()
}

fn assert_nonsingular(packed: &[f64], n: usize) {
    for (index, value) in u_diagonal(packed, n).iter().enumerate() {
        assert!(
            value.abs() > 1e-8,
            "test matrix is singular at U[{index}][{index}] = {value}"
        );
    }
}

/// Factor one `n x n` matrix and return `(packed, pivots, parity)`.
fn factor_one(a: &[f64], n: usize) -> (Vec<f64>, Vec<i32>, f64) {
    let mut packed = a.to_vec();
    let mut pivots = vec![0i32; n];
    let mut parity = vec![0.0f64; 1];
    factor_chunk(
        Op::LuFactor,
        n,
        n,
        &mut packed,
        &mut pivots,
        &mut parity,
        &mut Exec::serial(),
    )
    .unwrap();
    (packed, pivots, parity[0])
}

#[test]
fn factor_chunk_reproduces_the_input_and_the_parity() {
    for n in [1usize, 3, 5] {
        let a = matrix(n);
        let (packed, pivots, parity) = factor_one(&a, n);

        let mut rebuilt = reconstruct_lu(&packed, n);
        undo_swaps(&mut rebuilt, n, &pivots);
        for (index, (got, want)) in rebuilt.iter().zip(a.iter()).enumerate() {
            assert!(
                (got - want).abs() < 1e-10,
                "n={n} entry {index}: reconstructed {got} != input {want}"
            );
        }

        let swapped = pivots
            .iter()
            .enumerate()
            .filter(|(i, p)| **p as usize - 1 != *i)
            .count();
        let expected = if swapped % 2 == 1 { -1.0 } else { 1.0 };
        assert_eq!(parity, expected, "n={n} parity");
    }
}

#[test]
fn factor_chunk_rejects_buffers_that_describe_different_batches() {
    let mut lu = vec![0.0f64; 4];
    let mut pivots = vec![0i32; 1];
    let mut parity = vec![0.0f64; 2];
    let error = factor_chunk(
        Op::LuFactor,
        2,
        2,
        &mut lu,
        &mut pivots,
        &mut parity,
        &mut Exec::serial(),
    )
    .unwrap_err();
    assert!(matches!(
        error,
        Error::Inconsistent {
            op: Op::LuFactor,
            ..
        }
    ));
}

#[test]
fn empty_batches_and_empty_rhs_are_accepted() {
    let n = 2;
    // No matrices at all.
    let mut no_matrices: Vec<f64> = Vec::new();
    let mut pivots: Vec<i32> = Vec::new();
    let mut parity: Vec<f64> = Vec::new();
    factor_chunk(
        Op::LuFactor,
        n,
        n,
        &mut no_matrices,
        &mut pivots,
        &mut parity,
        &mut Exec::serial(),
    )
    .unwrap();

    // A zero-column RHS only factors.
    let a = matrix(n);
    let mut packed = a.clone();
    let mut pivots = vec![0i32; n];
    let mut no_rhs: Vec<f64> = Vec::new();
    factor_solve_chunk(
        Op::LuFactorSolve,
        n,
        0,
        &mut packed,
        &mut pivots,
        &mut no_rhs,
        &mut Exec::serial(),
    )
    .unwrap();
}

#[test]
fn prepared_solve_recovers_a_known_solution_for_every_flag_combination() {
    let n = 4;
    let a = matrix(n);
    let (packed, pivots, _) = factor_one(&a, n);
    assert_nonsingular(&packed, n);

    let x_true = [1.0, -2.0, 0.5, 3.0];

    // Plain solve: A x = b.
    let b = matvec(&a, n, &x_true);
    let mut out = b.clone();
    solve_prepared_chunk(
        Op::LuSolvePrepared,
        n,
        1,
        &packed,
        &pivots,
        &mut out,
        false,
        false,
        &mut Exec::serial(),
    )
    .unwrap();
    for (got, want) in out.iter().zip(x_true.iter()) {
        assert!((got - want).abs() < 1e-10, "plain: {got} != {want}");
    }

    // Transposed solve: A^T x = b.
    let mut a_t = vec![0.0f64; n * n];
    for row in 0..n {
        for col in 0..n {
            a_t[row + col * n] = a[col + row * n];
        }
    }
    let b_t = matvec(&a_t, n, &x_true);
    let mut out = b_t.clone();
    solve_prepared_chunk(
        Op::LuSolvePrepared,
        n,
        1,
        &packed,
        &pivots,
        &mut out,
        true,
        false,
        &mut Exec::serial(),
    )
    .unwrap();
    for (got, want) in out.iter().zip(x_true.iter()) {
        assert!((got - want).abs() < 1e-10, "transposed: {got} != {want}");
    }

    // Two right-hand sides at once: A X = B.
    let x2 = [-0.5, 1.5, 2.0, -1.0];
    let mut rhs = b.clone();
    rhs.extend(matvec(&a, n, &x2));
    solve_prepared_chunk(
        Op::LuSolvePrepared,
        n,
        2,
        &packed,
        &pivots,
        &mut rhs,
        false,
        false,
        &mut Exec::serial(),
    )
    .unwrap();
    for (got, want) in rhs[..n].iter().zip(x_true.iter()) {
        assert!((got - want).abs() < 1e-10, "rhs 0: {got} != {want}");
    }
    for (got, want) in rhs[n..].iter().zip(x2.iter()) {
        assert!((got - want).abs() < 1e-10, "rhs 1: {got} != {want}");
    }
}

#[test]
fn factor_solve_matches_factor_then_solve() {
    let n = 3;
    let a = matrix(n);
    let (packed, _, _) = factor_one(&a, n);
    let x_true = [1.0, -2.0, 0.5];
    let b = matvec(&a, n, &x_true);
    let mut expected = b.clone();
    let (reference_packed, reference_pivots, _) = factor_one(&a, n);
    solve_prepared_chunk(
        Op::LuSolvePrepared,
        n,
        1,
        &reference_packed,
        &reference_pivots,
        &mut expected,
        false,
        false,
        &mut Exec::serial(),
    )
    .unwrap();

    let mut fused = a.clone();
    let mut pivots = vec![0i32; n];
    let mut solved = b.clone();
    factor_solve_chunk(
        Op::LuFactorSolve,
        n,
        1,
        &mut fused,
        &mut pivots,
        &mut solved,
        &mut Exec::serial(),
    )
    .unwrap();

    assert_eq!(fused, packed, "fused factors differ from factor only");
    for (got, want) in solved.iter().zip(expected.iter()) {
        assert!((got - want).abs() < 1e-12, "fused {got} != {want}");
    }
    for (got, want) in solved.iter().zip(x_true.iter()) {
        assert!((got - want).abs() < 1e-10, "fused {got} != {want}");
    }
}

#[test]
fn factor_solve_reports_singular_only_when_a_rhs_is_present() {
    let n = 2;
    // Second column equals the first, so the factor is exactly singular.
    let a = [1.0, 2.0, 1.0, 2.0];
    let mut packed = a;
    let mut pivots = vec![0i32; n];
    let mut no_rhs: Vec<f64> = Vec::new();
    factor_solve_chunk(
        Op::LuFactorSolve,
        n,
        0,
        &mut packed,
        &mut pivots,
        &mut no_rhs,
        &mut Exec::serial(),
    )
    .unwrap();

    let mut packed = a;
    let mut pivots = vec![0i32; n];
    let mut rhs = vec![1.0f64; n];
    let error = factor_solve_chunk(
        Op::LuFactorSolve,
        n,
        1,
        &mut packed,
        &mut pivots,
        &mut rhs,
        &mut Exec::serial(),
    )
    .unwrap_err();
    assert!(matches!(
        error,
        Error::Singular {
            op: Op::LuFactorSolve
        }
    ));
}

#[test]
fn an_uneven_chunk_partition_matches_the_same_matrices_taken_alone() {
    // A host splits a batch of five into `div_ceil` chunks of two, two and one, and calls the chunk
    // entries once per chunk. Distinct matrices catch a chunk that leaks state.
    let n = 3;
    let matrices = [matrix(n), matrix_b(n), matrix(n), matrix_b(n), matrix(n)];
    let mut batched = matrices.concat();
    let mut pivots = vec![0i32; 5 * n];
    let mut parity = [0.0f64; 5];
    let chunk_len = 2;
    for (index, chunk) in batched.chunks_mut(chunk_len * n * n).enumerate() {
        let offset = index * chunk_len;
        let count = chunk.len() / (n * n);
        assert!(count > 0 && count <= chunk_len);
        factor_chunk(
            Op::LuFactor,
            n,
            n,
            chunk,
            &mut pivots[offset * n..(offset + count) * n],
            &mut parity[offset..offset + count],
            &mut Exec::serial(),
        )
        .unwrap();
    }

    for (index, a) in matrices.iter().enumerate() {
        let (single, single_pivots, single_parity) = factor_one(a, n);
        assert_eq!(
            &batched[index * n * n..(index + 1) * n * n],
            &single[..],
            "matrix {index} differs"
        );
        assert_eq!(&pivots[index * n..(index + 1) * n], &single_pivots[..]);
        assert_eq!(parity[index], single_parity);
    }
}

#[test]
fn invalid_pivots_are_rejected_before_any_write() {
    let n = 2;
    let (packed, mut pivots, _) = factor_one(&matrix(n), n);
    pivots[0] = 7;
    let mut output = vec![1.0f64; n];
    let before = output.clone();
    let error = solve_prepared_chunk(
        Op::LuSolvePrepared,
        n,
        1,
        &packed,
        &pivots,
        &mut output,
        false,
        false,
        &mut Exec::serial(),
    )
    .unwrap_err();
    assert!(matches!(
        error,
        Error::InvalidArgument { role: "pivot", .. }
    ));
    assert_eq!(output, before);
}

#[test]
fn real_and_complex_scalars_factor_and_solve() {
    let n = 3;

    // f32.
    let a32: Vec<f32> = matrix(n).into_iter().map(|v| v as f32).collect();
    let mut packed = a32.clone();
    let mut pivots = vec![0i32; n];
    let mut parity = vec![0.0f32; 1];
    factor_chunk(
        Op::LuFactor,
        n,
        n,
        &mut packed,
        &mut pivots,
        &mut parity,
        &mut Exec::serial(),
    )
    .unwrap();

    // Complex64, all four flag combinations.
    let a: Vec<Complex64> = matrix(n)
        .into_iter()
        .enumerate()
        .map(|(index, real)| {
            let imag = if index % 2 == 0 { 0.5 } else { -0.25 };
            Complex64::new(real, imag)
        })
        .collect();
    let mut packed_c = a.clone();
    let mut pivots_c = vec![0i32; n];
    let mut parity_c = vec![Complex64::new(0.0, 0.0); 1];
    factor_chunk(
        Op::LuFactor,
        n,
        n,
        &mut packed_c,
        &mut pivots_c,
        &mut parity_c,
        &mut Exec::serial(),
    )
    .unwrap();

    let x_true = [
        Complex64::new(1.0, 0.5),
        Complex64::new(-2.0, 1.0),
        Complex64::new(0.0, -1.0),
    ];
    for (transpose, conjugate) in [(false, false), (true, false), (true, true)] {
        let expected = if transpose {
            matvec_transposed(&a, n, &x_true, conjugate)
        } else {
            matvec(&a, n, &x_true)
        };
        let mut out = expected.clone();
        solve_prepared_chunk(
            Op::LuSolvePrepared,
            n,
            1,
            &packed_c,
            &pivots_c,
            &mut out,
            transpose,
            conjugate,
            &mut Exec::serial(),
        )
        .unwrap();
        for (got, want) in out.iter().zip(x_true.iter()) {
            assert!(
                (got - want).norm() < 1e-10,
                "transpose={transpose} conjugate={conjugate}: {got} != {want}"
            );
        }
    }

    // Complex32 factors.
    let a32c: Vec<Complex32> = matrix(n)
        .into_iter()
        .map(|real| Complex32::new(real as f32, 0.25))
        .collect();
    let mut packed = a32c.clone();
    let mut pivots = vec![0i32; n];
    let mut parity = vec![Complex32::new(0.0, 0.0); 1];
    factor_chunk(
        Op::LuFactor,
        n,
        n,
        &mut packed,
        &mut pivots,
        &mut parity,
        &mut Exec::serial(),
    )
    .unwrap();
}
