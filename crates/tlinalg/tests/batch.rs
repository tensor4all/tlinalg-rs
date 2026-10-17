//! The batch contract shared by every family (`docs/design/batched-api.md`): batched results equal
//! the per-item results for `B = 0, 1, 2` batch axes and compact, gapped (non-mergeable) and
//! transposed batch layouts; several lanes on a pool give exactly the single-lane result; an empty
//! batch yields empty outputs; a failing item leaves the vector outputs empty; stride-0 batch axes
//! broadcast an input; an aliased destination is rejected.

mod common;

use common::*;
use num_complex::Complex64;
use strided_view::{RawStridedMut, RawStridedRef};
use tcontract_exec::Exec;
use tlinalg::triangular_solve::{triangular_solve, TriangularSolveFlags};
use tlinalg::{Error, FaerScalar, Op, Result};

/// Batch shapes every family is run over.
const SHAPES: &[&[usize]] = &[&[], &[1], &[3], &[0], &[2, 3], &[2, 1, 2]];
const LAYOUTS: [Layout; 3] = [Layout::Compact, Layout::Gapped, Layout::Transposed];

/// A family under test: inputs in, every output widened to `Complex64`, one `Vec` per output.
type Call<'f, T> = &'f dyn Fn(RawStridedRef<'_, T>, Exec<'_>) -> Result<Vec<Vec<Complex64>>>;

/// Run `call` over every batch shape and layout and check the batch contract against per-item
/// calls on compact single matrices.
fn check_family<T: TestScalar + FaerScalar>(
    name: &str,
    (m, n): (usize, usize),
    make: impl Fn(usize) -> Vec<T>,
    call: Call<'_, T>,
) {
    let pool = lanes_pool();
    for &shape in SHAPES {
        let count: usize = shape.iter().product();
        let items: Vec<Vec<T>> = (0..count).map(&make).collect();
        // The reference: each item alone, as a rank-2 compact matrix.
        let mut expected: Option<Vec<Vec<Complex64>>> = None;
        for item in &items {
            let single = call(
                RawStridedRef::new(item, &[m, n], &[1, m as isize], 0).unwrap(),
                Exec::serial(),
            )
            .unwrap();
            match &mut expected {
                None => expected = Some(single),
                Some(acc) => {
                    for (all, one) in acc.iter_mut().zip(single) {
                        all.extend(one);
                    }
                }
            }
        }
        for layout in LAYOUTS {
            let buf = batch_buf(&items, m, n, shape, layout);
            let sequential = call(buf.view(), Exec::serial()).unwrap();
            let pooled = call(buf.view(), pool_token(&pool)).unwrap();
            let what = format!("{name} {shape:?} {layout:?}");
            assert_eq!(
                pooled, sequential,
                "{what}: lanes must not change the result"
            );
            match &expected {
                Some(expected) => {
                    for (got, want) in sequential.iter().zip(expected) {
                        assert_close(got, want, T::TOL, &what);
                    }
                }
                None => {
                    assert!(sequential.iter().all(Vec::is_empty), "{what}: empty batch");
                }
            }
        }
    }
}

fn families<T: TestScalar + FaerScalar>()
where
    T::Real: TestScalar,
    T::Complex: TestScalar,
{
    let general = |seed: usize| move |i: usize| matrix::<T>(4, 3, seed + i);
    let square = |seed: usize| move |i: usize| matrix::<T>(4, 4, seed + i);
    let hpd_item = |i: usize| {
        let mut a = hpd::<T>(4);
        a[0] = T::from_c64(a[0].to_c64() + Complex64::new(i as f64, 0.0));
        a
    };
    let herm_item = |i: usize| {
        let mut a = hermitian::<T>(4);
        a[0] = T::from_c64(a[0].to_c64() + Complex64::new(i as f64, 0.0));
        a
    };

    for full in [false, true] {
        check_family::<T>("svd", (4, 3), general(1), &|a, par| {
            let (mut u, mut s, mut vt) = (Vec::new(), Vec::new(), Vec::new());
            tlinalg::svd::svd(Op::Svd, a, full, &mut u, &mut s, &mut vt, par)?;
            // Singular vectors are unique up to phase; compare the gauge-free `U diag(S) Vᴴ` and S.
            Ok(vec![widen(&s), reconstruct_svd(&u, &s, &vt, 4, 3, full)])
        });
    }
    check_family::<T>("svd_values", (3, 4), |i| matrix::<T>(3, 4, i), &|a, par| {
        let mut s = Vec::new();
        tlinalg::svd::svd_values(Op::SvdValues, a, &mut s, par)?;
        Ok(vec![widen(&s)])
    });
    check_family::<T>("cholesky", (4, 4), hpd_item, &|a, par| {
        let mut l = Vec::new();
        tlinalg::cholesky::cholesky(Op::Cholesky, a, &mut l, par)?;
        Ok(vec![widen(&l)])
    });
    check_family::<T>("lu", (4, 3), general(2), &|a, par| {
        let (mut p, mut l, mut u, mut parity) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        let factors = tlinalg::lu::LuFactors {
            p: &mut p,
            l: &mut l,
            u: &mut u,
            parity: &mut parity,
        };
        tlinalg::lu::lu(Op::Lu, a, factors, par)?;
        Ok(vec![widen(&p), widen(&l), widen(&u), widen(&parity)])
    });
    check_family::<T>("full_piv_lu", (4, 4), square(3), &|a, par| {
        let (mut p, mut l, mut u, mut q, mut parity) =
            (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new());
        let factors = tlinalg::full_piv_lu::FullPivLuFactors {
            p: &mut p,
            l: &mut l,
            u: &mut u,
            q: &mut q,
            parity: &mut parity,
        };
        tlinalg::full_piv_lu::full_piv_lu(Op::FullPivLu, a, factors, par)?;
        Ok(vec![
            widen(&p),
            widen(&l),
            widen(&u),
            widen(&q),
            widen(&parity),
        ])
    });
    check_family::<T>("qr", (4, 3), general(4), &|a, par| {
        let (mut q, mut r) = (Vec::new(), Vec::new());
        tlinalg::qr::qr(Op::Qr, a, &mut q, &mut r, par)?;
        Ok(vec![widen(&q), widen(&r)])
    });
    check_family::<T>(
        "rank_revealing_qr",
        (3, 4),
        |i| matrix::<T>(3, 4, 5 + i),
        &|a, par| {
            let (mut q, mut r, mut perm) = (Vec::new(), Vec::new(), Vec::new());
            tlinalg::qr::rank_revealing_qr(Op::RankRevealingQr, a, &mut q, &mut r, &mut perm, par)?;
            let perm = perm
                .iter()
                .map(|&p| Complex64::new(p as f64, 0.0))
                .collect();
            Ok(vec![widen(&q), widen(&r), perm])
        },
    );
    check_family::<T>("eigh", (4, 4), herm_item, &|a, par| {
        let (mut w, mut v) = (Vec::new(), Vec::new());
        tlinalg::eigh::eigh(Op::Eigh, a, &mut w, &mut v, par)?;
        // Eigenvectors are unique up to phase; `V diag(w) Vᴴ` is not.
        Ok(vec![widen(&w), reconstruct_eigh(&w, &v, 4)])
    });
    check_family::<T>("eigh_values", (4, 4), herm_item, &|a, par| {
        let mut w = Vec::new();
        tlinalg::eigh::eigh_values(Op::EighValues, a, &mut w, par)?;
        Ok(vec![widen(&w)])
    });
    check_family::<T>("eig_values", (4, 4), square(6), &|a, par| {
        let mut w = Vec::new();
        tlinalg::eig::eig_values(Op::EigValues, a, &mut w, par)?;
        Ok(vec![sorted(widen_complex::<T>(&w))])
    });
    check_family::<T>("eig", (4, 4), square(7), &|a, par| {
        let (mut w, mut v) = (Vec::new(), Vec::new());
        tlinalg::eig::eig(Op::Eig, a, &mut w, &mut v, par)?;
        Ok(vec![widen_complex::<T>(&w), widen_complex::<T>(&v)])
    });
}
for_each_scalar!(every_family, families);

/// Widen the complex output type of an eigensolver.
fn widen_complex<T: FaerScalar>(values: &[T::Complex]) -> Vec<Complex64>
where
    T::Complex: TestScalar,
{
    values.iter().map(|&v| v.to_c64()).collect()
}

/// Sort eigenvalues per item of four so order differences between layouts cannot matter.
fn sorted(mut values: Vec<Complex64>) -> Vec<Complex64> {
    for item in values.chunks_mut(4) {
        item.sort_by(|a, b| (a.re, a.im).partial_cmp(&(b.re, b.im)).unwrap());
    }
    values
}

/// `U diag(S) Vᴴ` per item (compact `m x n` each).
fn reconstruct_svd<T: TestScalar>(
    u: &[T],
    s: &[T],
    vt: &[T],
    m: usize,
    n: usize,
    full: bool,
) -> Vec<Complex64> {
    let k = m.min(n);
    let (u_cols, v_cols) = if full { (m, n) } else { (k, k) };
    let items = s.len() / k;
    let (u, s, vt) = (widen(u), widen(s), widen(vt));
    let mut out = Vec::new();
    for item in 0..items {
        let u = &u[item * m * u_cols..];
        let vt = &vt[item * v_cols * n..];
        for col in 0..n {
            for row in 0..m {
                let mut acc = Complex64::new(0.0, 0.0);
                for j in 0..k {
                    acc += u[row + j * m] * s[item * k + j] * vt[j + col * v_cols];
                }
                out.push(acc);
            }
        }
    }
    out
}

/// `V diag(w) Vᴴ` per item.
fn reconstruct_eigh<T: TestScalar>(w: &[T], v: &[T], n: usize) -> Vec<Complex64> {
    let (w, v) = (widen(w), widen(v));
    let mut out = Vec::new();
    for item in 0..w.len() / n {
        let v = &v[item * n * n..(item + 1) * n * n];
        let mut vw = v.to_vec();
        for col in 0..n {
            for row in 0..n {
                vw[row + col * n] *= w[item * n + col];
            }
        }
        out.extend(matmul(&vw, &adjoint(v, n, n), n, n, n));
    }
    out
}

#[test]
fn a_failing_item_empties_the_vector_outputs() {
    let pool = lanes_pool();
    let n = 3;
    // Five items, items 1 and 3 not positive definite; three lanes put them in different lanes.
    let items: Vec<Vec<f64>> = (0..5)
        .map(|i| {
            let mut a = hpd::<f64>(n);
            if i == 1 || i == 3 {
                a[0] = -100.0;
            }
            a
        })
        .collect();
    let buf = batch_buf(&items, n, n, &[5], Layout::Gapped);
    for par in [Exec::serial(), pool_token(&pool)] {
        let mut l = vec![1.0; 7];
        let err = tlinalg::cholesky::cholesky(Op::Cholesky, buf.view(), &mut l, par).unwrap_err();
        assert_eq!(err, Error::NonConvergence { op: Op::Cholesky });
        assert!(l.is_empty(), "no partially written output");
    }
}

#[test]
fn a_failing_solve_item_stops_only_its_own_lane() {
    let pool = lanes_pool();
    let n = 2;
    // Six items over three lanes of two: item 1 (lane 0) and item 4 (lane 2) are singular.
    let a_items: Vec<Vec<f64>> = (0..6)
        .map(|i| {
            if i == 1 || i == 4 {
                vec![1.0, 2.0, 2.0, 4.0]
            } else {
                vec![2.0 + i as f64, 0.0, 0.0, 1.0]
            }
        })
        .collect();
    let a = batch_buf(&a_items, n, n, &[6], Layout::Compact);
    let sentinel = 7.0;
    let mut out = batch_buf(&vec![vec![sentinel; n]; 6], n, 1, &[6], Layout::Gapped);
    let b = vec![2.0, 3.0];
    let err = tlinalg::lu::solve(
        Op::Solve,
        a.view(),
        Some(RawStridedRef::new(&b, &[n, 1, 6], &[1, n as isize, 0], 0).unwrap()),
        out.view_mut(),
        false,
        pool_token(&pool),
    )
    .unwrap_err();
    assert_eq!(err, Error::Singular { op: Op::Solve });
    // Lane 0: item 0 solved, item 1 failed and is untouched. Lane 1 (items 2, 3) ran to the end
    // although lane 0 failed. Lane 2: item 4 failed and stopped the lane, so item 5 is untouched.
    let x = |i: usize| out.item(i);
    let close = |got: Vec<f64>, want: [f64; 2]| {
        assert_close(&widen(&got), &widen(&want), 1e-12, "solved item");
    };
    close(x(0), [1.0, 3.0]);
    assert_eq!(x(1), vec![sentinel; 2]);
    close(x(2), [0.5, 3.0]);
    close(x(3), [0.4, 3.0]);
    assert_eq!(x(4), vec![sentinel; 2]);
    assert_eq!(x(5), vec![sentinel; 2]);
}

#[test]
fn solve_broadcasts_a_and_rejects_an_aliased_destination() {
    let pool = lanes_pool();
    let n = 3;
    let a = matrix::<f64>(n, n, 9);
    let b_items: Vec<Vec<f64>> = (0..4).map(|i| matrix::<f64>(n, 2, i)).collect();
    let b = batch_buf(&b_items, n, 2, &[2, 2], Layout::Gapped);
    // One `A`, stride 0 on both batch axes.
    let (dims1, strides1) = ([n, n, 2, 2], [1, n as isize, 0, 0]);

    let a_view = RawStridedRef::new(&a, &dims1, &strides1, 0).unwrap();
    for par in [Exec::serial(), pool_token(&pool)] {
        let mut out = batch_buf(
            &vec![vec![0.0; n * 2]; 4],
            n,
            2,
            &[2, 2],
            Layout::Transposed,
        );
        tlinalg::lu::solve(
            Op::Solve,
            a_view,
            Some(b.view()),
            out.view_mut(),
            false,
            par,
        )
        .unwrap();
        for (i, b_item) in b_items.iter().enumerate() {
            let x = widen(&out.item(i));
            assert_close(
                &matmul(&widen(&a), &x, n, n, 2),
                &widen(b_item),
                1e-10,
                "broadcast solve",
            );
        }
    }

    // A destination whose batch axis has stride 0 writes every item to the same place.
    let mut storage = vec![0.0; n];
    let err = tlinalg::lu::solve(
        Op::Solve,
        RawStridedRef::new(&a, &[n, n, 2], &[1, n as isize, 0], 0).unwrap(),
        None,
        RawStridedMut::new(&mut storage, &[n, 1, 2], &[1, n as isize, 0], 0).unwrap(),
        false,
        Exec::serial(),
    )
    .unwrap_err();
    assert!(matches!(err, Error::InvalidArgument { .. }), "{err:?}");
    // Overlapping item ranges: the batch stride is smaller than an item.
    let mut storage = vec![0.0; n + 1];
    let err = tlinalg::lu::solve(
        Op::Solve,
        RawStridedRef::new(&a, &[n, n, 2], &[1, n as isize, 0], 0).unwrap(),
        None,
        RawStridedMut::new(&mut storage, &[n, 1, 2], &[1, n as isize, 1], 0).unwrap(),
        false,
        Exec::serial(),
    )
    .unwrap_err();
    assert!(matches!(err, Error::InvalidArgument { .. }), "{err:?}");
}

#[test]
fn triangular_solve_broadcasts_a_and_matches_lanes() {
    let pool = lanes_pool();
    let n = 3;
    let a = matrix::<Complex64>(n, n, 3);
    for left_side in [true, false] {
        let flags = TriangularSolveFlags {
            left_side,
            lower: true,
            transpose_a: false,
            unit_diagonal: false,
        };
        let (rows, cols) = if left_side { (n, 2) } else { (2, n) };
        let b_items: Vec<Vec<Complex64>> =
            (0..5).map(|i| matrix::<Complex64>(rows, cols, i)).collect();
        let b = batch_buf(&b_items, rows, cols, &[5], Layout::Gapped);
        let (dims2, strides2) = ([n, n, 5], [1, n as isize, 0]);

        let a_view = RawStridedRef::new(&a, &dims2, &strides2, 0).unwrap();
        let mut seq = Vec::new();
        triangular_solve(
            Op::TriangularSolve,
            a_view,
            b.view(),
            flags,
            &mut seq,
            Exec::serial(),
        )
        .unwrap();
        let mut lanes = Vec::new();
        triangular_solve(
            Op::TriangularSolve,
            a_view,
            b.view(),
            flags,
            &mut lanes,
            pool_token(&pool),
        )
        .unwrap();
        assert_eq!(seq, lanes);
        for (i, b_item) in b_items.iter().enumerate() {
            let mut single = Vec::new();
            triangular_solve(
                Op::TriangularSolve,
                RawStridedRef::new(&a, &[n, n], &[1, n as isize], 0).unwrap(),
                RawStridedRef::new(b_item, &[rows, cols], &[1, rows as isize], 0).unwrap(),
                flags,
                &mut single,
                Exec::serial(),
            )
            .unwrap();
            assert_eq!(&seq[i * rows * cols..(i + 1) * rows * cols], &single[..]);
        }
    }
}

#[test]
fn full_piv_lu_solve_batches_and_broadcasts() {
    let pool = lanes_pool();
    let n = 3;
    let a = matrix::<f64>(n, n, 2);
    let b_items: Vec<Vec<f64>> = (0..4).map(|i| matrix::<f64>(n, 1, i)).collect();
    let b = batch_buf(&b_items, n, 1, &[4], Layout::Gapped);
    let (dims3, strides3) = ([n, n, 4], [1, n as isize, 0]);

    let a_view = RawStridedRef::new(&a, &dims3, &strides3, 0).unwrap();
    let mut seq = Vec::new();
    tlinalg::full_piv_lu::full_piv_lu_solve(
        Op::FullPivLuSolve,
        a_view,
        b.view(),
        false,
        &mut seq,
        Exec::serial(),
    )
    .unwrap();
    let mut lanes = Vec::new();
    tlinalg::full_piv_lu::full_piv_lu_solve(
        Op::FullPivLuSolve,
        a_view,
        b.view(),
        false,
        &mut lanes,
        pool_token(&pool),
    )
    .unwrap();
    assert_eq!(seq, lanes);
    for (i, b_item) in b_items.iter().enumerate() {
        let x = widen(&seq[i * n..(i + 1) * n]);
        assert_close(
            &matmul(&widen(&a), &x, n, n, 1),
            &widen(b_item),
            1e-10,
            "full-pivot solve",
        );
    }
}

#[test]
fn packed_lu_lanes_match_and_prepared_solve_broadcasts() {
    use tlinalg::packed_lu::{factor, factor_solve, solve_prepared};
    let pool = lanes_pool();
    let n = 3;
    let batch = 5;
    let a: Vec<f64> = (0..batch).flat_map(|i| matrix::<f64>(n, n, i)).collect();
    let run_factor = |par| {
        let (mut lu, mut piv, mut parity) = (a.clone(), vec![0; n * batch], vec![0.0; batch]);
        factor(Op::LuFactor, n, n, &mut lu, &mut piv, &mut parity, par).unwrap();
        (lu, piv, parity)
    };
    let sequential = run_factor(Exec::serial());
    assert_eq!(run_factor(pool_token(&pool)), sequential);

    let b: Vec<f64> = (0..batch)
        .flat_map(|i| matrix::<f64>(n, 2, 10 + i))
        .collect();
    let run_fused = |par| {
        let (mut lu, mut piv, mut x) = (a.clone(), vec![0; n * batch], b.clone());
        factor_solve(Op::LuFactorSolve, n, 2, &mut lu, &mut piv, &mut x, par).unwrap();
        x
    };
    assert_eq!(run_fused(pool_token(&pool)), run_fused(Exec::serial()));

    // Broadcast item 0's factors (stride 0) over every right-hand side.
    let (lu, piv, _) = &sequential;
    let (dims4, strides4) = ([n, n, batch], [1, n as isize, 0]);

    let lu_view = RawStridedRef::new(&lu[..n * n], &dims4, &strides4, 0).unwrap();
    let (dims5, strides5) = ([n, batch], [1, 0]);

    let piv_view = RawStridedRef::new(&piv[..n], &dims5, &strides5, 0).unwrap();
    let solve = |par| {
        let mut x = b.clone();
        solve_prepared(
            Op::LuSolvePrepared,
            lu_view,
            piv_view,
            2,
            &mut x,
            false,
            false,
            par,
        )
        .unwrap();
        x
    };
    let x = solve(Exec::serial());
    assert_eq!(solve(pool_token(&pool)), x);
    let a0 = widen(&a[..n * n]);
    for i in 0..batch {
        let xi = widen(&x[i * n * 2..(i + 1) * n * 2]);
        assert_close(
            &matmul(&a0, &xi, n, n, 2),
            &widen(&b[i * n * 2..(i + 1) * n * 2]),
            1e-10,
            "broadcast prepared solve",
        );
    }
}

#[test]
fn householder_lanes_match() {
    use tlinalg::householder::{apply_reflectors, compact_factor, ReflectorShape};
    let pool = lanes_pool();
    let (rows, cols, batch) = (5, 3, 4);
    let a: Vec<Complex64> = (0..batch)
        .flat_map(|i| matrix::<Complex64>(rows, cols, i))
        .collect();
    let run = |par| {
        let (mut data, mut coeff) = (a.clone(), Vec::new());
        compact_factor(
            Op::HouseholderQr,
            rows,
            cols,
            batch,
            &mut data,
            &mut coeff,
            par,
        )
        .unwrap();
        let mut c: Vec<Complex64> = (0..batch)
            .flat_map(|_| narrow::<Complex64>(&identity(rows)))
            .collect();
        let shape = ReflectorShape {
            rows,
            a_cols: cols,
            cols: rows,
            k: cols,
        };
        apply_reflectors(
            Op::HouseholderQrQColumns,
            shape,
            batch,
            &data,
            &coeff,
            &mut c,
            false,
            par,
        )
        .unwrap();
        (data, coeff, c)
    };
    let sequential = run(Exec::serial());
    assert_eq!(run(pool_token(&pool)), sequential);
    assert_eq!(sequential.1.len(), batch * cols);
}

/// An argument the entry point rejects before the driver runs must still leave every library-created
/// output empty, so a host recycling a pooled buffer never reads a previous call's values.
#[test]
fn a_rejected_argument_still_empties_the_outputs() {
    let par = Exec::serial();

    // `cholesky` requires a square input.
    let a = [1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0];
    let view = || RawStridedRef::new(&a, &[2, 3], &[1, 2], 0).unwrap();
    let mut l = vec![7.0; 4];
    assert!(tlinalg::cholesky::cholesky(Op::Cholesky, view(), &mut l, par).is_err());
    assert!(l.is_empty(), "cholesky");

    // The same rejection must clear every output of a multi-output family at once.
    let (mut values, mut vectors) = (vec![7.0; 4], vec![7.0; 4]);
    assert!(tlinalg::eigh::eigh(Op::Eigh, view(), &mut values, &mut vectors, par).is_err());
    assert!(values.is_empty() && vectors.is_empty(), "eigh");

    // A batch-shape mismatch is checked before the driver too.
    let a2 = [4.0_f64, 0.0, 0.0, 4.0];
    let b = [1.0_f64, 1.0, 1.0, 1.0];
    let mut x = vec![7.0; 4];
    assert!(
        triangular_solve(
            Op::TriangularSolve,
            RawStridedRef::new(&a2, &[2, 2, 1], &[1, 2, 0], 0).unwrap(),
            RawStridedRef::new(&b, &[2, 1, 2], &[1, 2, 0], 0).unwrap(),
            TriangularSolveFlags {
                left_side: true,
                lower: true,
                transpose_a: false,
                unit_diagonal: false,
            },
            &mut x,
            par,
        )
        .is_err(),
        "triangular_solve"
    );
    assert!(x.is_empty(), "triangular_solve");

    // A caller-owned buffer of the wrong length fails while the outputs are being prepared.
    let mut state: Vec<f64> = Vec::new();
    let mut coeff = vec![7.0; 2];
    assert!(
        tlinalg::householder::compact_factor(
            Op::HouseholderQr,
            2,
            1,
            1,
            &mut state,
            &mut coeff,
            par,
        )
        .is_err(),
        "compact_factor"
    );
    assert!(coeff.is_empty(), "compact_factor");
}

#[test]
fn rank_below_two_is_rejected() {
    let a = [1.0_f64; 3];
    let mut l = Vec::new();
    let err = tlinalg::cholesky::cholesky(
        Op::Cholesky,
        RawStridedRef::new(&a, &[3], &[1], 0).unwrap(),
        &mut l,
        Exec::serial(),
    )
    .unwrap_err();
    assert!(matches!(err, Error::InvalidArgument { .. }));
}
