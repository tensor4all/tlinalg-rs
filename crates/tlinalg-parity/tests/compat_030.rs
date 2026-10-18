//! Bitwise compatibility of the tlinalg `Vec` wrappers with the published 0.3.0 release.
//!
//! Lives in this unpublished crate because it needs the published `tlinalg 0.3.0` (and its own
//! `tcontract-exec`) as renamed dev-dependencies, which `cargo publish` rejects in `tlinalg` itself.

use num_complex::{Complex32, Complex64};
use strided_view::{RawStridedMut, RawStridedRef};
use tcontract_exec::Exec;
use tlinalg_testkit::TestScalar;

/// Serialises the tests in this binary.
fn serialised() -> std::sync::MutexGuard<'static, ()> {
    static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
    LOCK.lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

/// The raw bytes of an output buffer: the bitwise comparison (ordinary `==` on floats is not
/// bitwise, and `NaN` payloads must compare).
fn bytes_of<T: Copy>(values: &[T]) -> Vec<u8> {
    // SAFETY: every `T` here is a plain `Copy` scalar (f32/f64/`Complex`/i32/i64), so reading the
    // initialised slice as bytes is sound and exposes the exact bit patterns.
    unsafe {
        core::slice::from_raw_parts(values.as_ptr().cast::<u8>(), core::mem::size_of_val(values))
    }
    .to_vec()
}

// ---------------------------------------------------------------------------------------------
// Bitwise compatibility with the published 0.3.0 wrappers.
// ---------------------------------------------------------------------------------------------

/// The same fingerprint battery, driven through one crate's **`Vec` wrappers** only.
///
/// Expanded once for this checkout and once for the published `tlinalg 0.3.0` dev-dependency, so the
/// comparison is against the release the patch claims not to change, not against a second call of
/// the new code. `$exec` is the matching `Exec` type (the two crates consume different
/// `tcontract-exec` packages).
macro_rules! wrapper_fingerprints {
    ($krate:ident, $exec:expr, $t:ty, $real:ty, $cplx:ty) => {{
        let seq = $exec;
        let value = move |x: f64| <$t as TestScalar>::from_c64(Complex64::new(x, 0.0));
        let spd = move |n: usize, batch: usize| -> Vec<$t> {
            (0..n * n * batch)
                .map(|k| {
                    let (row, col) = (k % n, (k / n) % n);
                    if row == col {
                        value(n as f64 + 2.0)
                    } else {
                        value(0.05 * ((row + 1) * (col + 1) % 3) as f64)
                    }
                })
                .collect()
        };
        let transposed = move |data: &[$t], n: usize| -> Vec<$t> {
            (0..data.len())
                .map(|k| {
                    let (item, rest) = (k / (n * n), k % (n * n));
                    let (row, col) = (rest % n, rest / n);
                    data[item * n * n + col * n + row]
                })
                .collect()
        };
        move |n: usize, batch: usize, transposed_layout: bool| -> Vec<(&'static str, Vec<u8>)> {
            let data = spd(n, batch);
            let (owned, dims, strides): (Vec<$t>, [usize; 3], [isize; 3]) = if transposed_layout {
                (
                    transposed(&data, n),
                    [n, n, batch],
                    [n as isize, 1, (n * n) as isize],
                )
            } else {
                (
                    data.clone(),
                    [n, n, batch],
                    [1, n as isize, (n * n) as isize],
                )
            };
            let view = RawStridedRef::new(&owned, &dims, &strides, 0).unwrap();
            let square = n * n;
            let mut out: Vec<(&'static str, Vec<u8>)> = Vec::new();
            // qr
            {
                let (mut q, mut r) = (Vec::new(), Vec::new());
                $krate::qr::qr($krate::Op::Qr, view, &mut q, &mut r, seq).unwrap();
                out.push(("qr.q", bytes_of(&q)));
                out.push(("qr.r", bytes_of(&r)));
            }
            // rank_revealing_qr
            {
                let (mut q, mut r, mut perm) = (Vec::new(), Vec::new(), Vec::new());
                $krate::qr::rank_revealing_qr(
                    $krate::Op::RankRevealingQr,
                    view,
                    &mut q,
                    &mut r,
                    &mut perm,
                    seq,
                )
                .unwrap();
                out.push(("rrqr.q", bytes_of(&q)));
                out.push(("rrqr.r", bytes_of(&r)));
                out.push(("rrqr.perm", bytes_of(&perm)));
            }
            // svd
            {
                let (mut u, mut s, mut vt) = (Vec::new(), Vec::new(), Vec::new());
                $krate::svd::svd($krate::Op::Svd, view, false, &mut u, &mut s, &mut vt, seq)
                    .unwrap();
                out.push(("svd.u", bytes_of(&u)));
                out.push(("svd.s", bytes_of(&s)));
                out.push(("svd.vt", bytes_of(&vt)));
            }
            // svd_values
            {
                let mut s: Vec<$real> = Vec::new();
                $krate::svd::svd_values($krate::Op::SvdValues, view, &mut s, seq).unwrap();
                out.push(("svd_values.s", bytes_of(&s)));
            }
            // eigh
            {
                let (mut values, mut vectors) = (Vec::new(), Vec::new());
                $krate::eigh::eigh($krate::Op::Eigh, view, &mut values, &mut vectors, seq).unwrap();
                out.push(("eigh.values", bytes_of(&values)));
                out.push(("eigh.vectors", bytes_of(&vectors)));
            }
            // eigh_values
            {
                let mut values: Vec<$real> = Vec::new();
                $krate::eigh::eigh_values($krate::Op::EighValues, view, &mut values, seq).unwrap();
                out.push(("eigh_values.values", bytes_of(&values)));
            }
            // eig
            {
                let (mut values, mut vectors): (Vec<$cplx>, Vec<$cplx>) = (Vec::new(), Vec::new());
                $krate::eig::eig($krate::Op::Eig, view, &mut values, &mut vectors, seq).unwrap();
                out.push(("eig.values", bytes_of(&values)));
                out.push(("eig.vectors", bytes_of(&vectors)));
            }
            // eig_values
            {
                let mut values: Vec<$cplx> = Vec::new();
                $krate::eig::eig_values($krate::Op::EigValues, view, &mut values, seq).unwrap();
                out.push(("eig_values.values", bytes_of(&values)));
            }
            // lu
            {
                let (mut p, mut l, mut u, mut parity) =
                    (Vec::new(), Vec::new(), Vec::new(), Vec::new());
                $krate::lu::lu(
                    $krate::Op::Lu,
                    view,
                    $krate::lu::LuFactors {
                        p: &mut p,
                        l: &mut l,
                        u: &mut u,
                        parity: &mut parity,
                    },
                    seq,
                )
                .unwrap();
                out.push(("lu.p", bytes_of(&p)));
                out.push(("lu.l", bytes_of(&l)));
                out.push(("lu.u", bytes_of(&u)));
                out.push(("lu.parity", bytes_of(&parity)));
            }
            // lu::solve
            {
                let mut x = vec![value(1.0); square * batch];
                let destination = RawStridedMut::new(&mut x, &dims, &strides, 0).unwrap();
                $krate::lu::solve($krate::Op::Solve, view, Some(view), destination, false, seq)
                    .unwrap();
                out.push(("solve.x", bytes_of(&x)));
            }
            // full_piv_lu
            {
                let (mut p, mut l, mut u, mut q, mut parity) =
                    (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new());
                $krate::full_piv_lu::full_piv_lu(
                    $krate::Op::FullPivLu,
                    view,
                    $krate::full_piv_lu::FullPivLuFactors {
                        p: &mut p,
                        l: &mut l,
                        u: &mut u,
                        q: &mut q,
                        parity: &mut parity,
                    },
                    seq,
                )
                .unwrap();
                out.push(("full_piv_lu.p", bytes_of(&p)));
                out.push(("full_piv_lu.l", bytes_of(&l)));
                out.push(("full_piv_lu.u", bytes_of(&u)));
                out.push(("full_piv_lu.q", bytes_of(&q)));
                out.push(("full_piv_lu.parity", bytes_of(&parity)));
            }
            // full_piv_lu_solve
            {
                let mut x = Vec::new();
                $krate::full_piv_lu::full_piv_lu_solve(
                    $krate::Op::FullPivLuSolve,
                    view,
                    view,
                    false,
                    &mut x,
                    seq,
                )
                .unwrap();
                out.push(("full_piv_lu_solve.x", bytes_of(&x)));
            }
            // cholesky
            {
                let mut l = Vec::new();
                $krate::cholesky::cholesky($krate::Op::Cholesky, view, &mut l, seq).unwrap();
                out.push(("cholesky.l", bytes_of(&l)));
            }
            // triangular_solve
            {
                let flags = $krate::triangular_solve::TriangularSolveFlags {
                    left_side: true,
                    lower: true,
                    transpose_a: false,
                    unit_diagonal: false,
                };
                let mut x = Vec::new();
                $krate::triangular_solve::triangular_solve(
                    $krate::Op::TriangularSolve,
                    view,
                    view,
                    flags,
                    &mut x,
                    seq,
                )
                .unwrap();
                out.push(("triangular_solve.x", bytes_of(&x)));
            }
            // packed_lu::factor, then solve_prepared on that factor
            {
                let mut lu = owned.clone();
                let mut pivots = vec![0_i32; n * batch];
                let mut parity = vec![value(0.0); batch];
                $krate::packed_lu::factor(
                    $krate::Op::LuFactor,
                    n,
                    n,
                    &mut lu,
                    &mut pivots,
                    &mut parity,
                    seq,
                )
                .unwrap();
                out.push(("packed_lu.lu", bytes_of(&lu)));
                out.push(("packed_lu.pivots", bytes_of(&pivots)));
                out.push(("packed_lu.parity", bytes_of(&parity)));
                let factor_dims = [n, n, batch];
                let factor_strides = [1isize, n as isize, (n * n) as isize];
                let factor_view =
                    RawStridedRef::new(&lu, &factor_dims, &factor_strides, 0).unwrap();
                let pivot_dims = [n, batch];
                let pivot_strides = [1isize, n as isize];
                let pivot_view =
                    RawStridedRef::new(&pivots, &pivot_dims, &pivot_strides, 0).unwrap();
                let mut x = owned.clone();
                $krate::packed_lu::solve_prepared(
                    $krate::Op::LuSolvePrepared,
                    factor_view,
                    pivot_view,
                    n,
                    &mut x,
                    false,
                    false,
                    seq,
                )
                .unwrap();
                out.push(("packed_lu.solve.x", bytes_of(&x)));
            }
            // householder::compact_factor, then apply_reflectors on that state
            {
                let mut state = owned.clone();
                let mut coeff = Vec::new();
                $krate::householder::compact_factor(
                    $krate::Op::HouseholderQr,
                    n,
                    n,
                    batch,
                    &mut state,
                    &mut coeff,
                    seq,
                )
                .unwrap();
                out.push(("compact_factor.state", bytes_of(&state)));
                out.push(("compact_factor.coeff", bytes_of(&coeff)));
                let shape = $krate::householder::ReflectorShape {
                    rows: n,
                    a_cols: n,
                    cols: n,
                    k: n,
                };
                let mut c = state.clone();
                $krate::householder::apply_reflectors(
                    $krate::Op::HouseholderQrQColumns,
                    shape,
                    batch,
                    &state,
                    &coeff,
                    &mut c,
                    false,
                    seq,
                )
                .unwrap();
                out.push(("apply_reflectors.c", bytes_of(&c)));
            }
            out
        }
    }};
}

/// The wrapper outputs against the published 0.3.0 release, family by family.
///
/// The 0.3.0 lane scratch was faer's own `Mat`/`Diag` storage (padded, over-aligned); the patch
/// carves compact typed slices out of borrowed bytes. That is a storage change inside the same
/// algorithms, so this test reports where — if anywhere — a result is not bitwise identical instead
/// of assuming it. See the family table in the test's assertions for the decision per family.
#[test]
fn the_wrappers_match_the_published_030_release() {
    let _guard = serialised();
    let mut differences: Vec<String> = Vec::new();
    macro_rules! compare {
        ($t:ty, $real:ty, $cplx:ty) => {{
            let now = wrapper_fingerprints!(tlinalg, Exec::serial(), $t, $real, $cplx);
            let old = wrapper_fingerprints!(
                tlinalg_030,
                tcontract_exec_030::Exec::serial(),
                $t,
                $real,
                $cplx
            );
            for n in [1usize, 2, 8] {
                for batch in [1usize, 64] {
                    for transposed in [false, true] {
                        let now = now(n, batch, transposed);
                        let old = old(n, batch, transposed);
                        assert_eq!(now.len(), old.len(), "fingerprint lists");
                        for ((name, new), (old_name, old)) in now.iter().zip(&old) {
                            assert_eq!(name, old_name);
                            if new != old {
                                let first = new
                                    .iter()
                                    .zip(old)
                                    .position(|(a, b)| a != b)
                                    .unwrap_or_default();
                                let total = new.iter().zip(old).filter(|(a, b)| a != b).count();
                                differences.push(format!(
                                    "{:?} {name} n={n} batch={batch} transposed={transposed}: \
                                     first differing byte {first}, {total} of {} differ",
                                    stringify!($t),
                                    new.len()
                                ));
                            }
                        }
                    }
                }
            }
        }};
    }
    compare!(f32, f32, Complex32);
    compare!(f64, f64, Complex64);
    compare!(Complex32, f32, Complex32);
    compare!(Complex64, f64, Complex64);
    for difference in &differences {
        println!("030-diff {difference}");
    }
    println!("030 comparison: {} differing outputs", differences.len());
    assert!(
        differences.is_empty(),
        "{} outputs differ from the published 0.3.0 wrappers: the storage change (padded faer \
         `Mat` to compact borrowed slices) moved a result. Report it rather than loosening this \
         gate. First few: {:#?}",
        differences.len(),
        &differences[..differences.len().min(5)]
    );
    // The new `_into` routes are bitwise equal to the new wrappers (asserted by
    // `into_is_allocation_free_for_every_family_and_layout` on the same shapes), so wrapper
    // equality here covers `_into` as well.
}
