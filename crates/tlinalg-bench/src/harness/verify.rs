//! Numerical correctness gate for every campaign family.
//!
//! Every check below reduces to a product of the returned factors against the matrix they were
//! computed from, so at a case's batch extent those products are what a verify pass costs. They are
//! computed by `gemm` — the harness's own complex, column-major GEMM — and by `dot_conj` for the
//! Gram matrix `QᴴQ`. Neither is faer (the `faer-*` rows) nor the vendor BLAS the `lapack-*` row
//! links, so a residual never compares a provider with itself; and both arms are checked by the
//! same arithmetic, which is what keeps `provider_residual` comparable across arms. `tlbench
//! verify` names that backend on stderr once per process.
//!
//! The reference matrix is generated once per case (`rows::reference_a`) and handed to both arms and
//! to the checks, so a case pays for it once rather than once per call.

use super::cases::{applicable, Case};
use super::rows::{output_call_with, reference_a, Output};
use crate::{Batch, BenchScalar, Env};
use num_complex::Complex64;

const ZERO: Complex64 = Complex64::new(0., 0.);
const ONE: Complex64 = Complex64::new(1., 0.);

fn tol<T: BenchScalar>(c: Case) -> f64 {
    64.0 * (c.m.max(c.n) as f64).sqrt() * T::TOL
}

/// The reference matrix and right-hand side one case is called on, widened once for the checks.
struct Reference<T> {
    a: Batch<T>,
    b: Batch<T>,
    /// `a`, widened: every check computes on `Complex64`.
    wide_a: Vec<Complex64>,
    /// `b`, widened: the right-hand side the solve residuals are measured against.
    wide_b: Vec<Complex64>,
}

impl<T: BenchScalar> Reference<T> {
    fn new(family: &str, c: Case) -> Self {
        let a = reference_a::<T>(family, c);
        let b = Batch::<T>::general(c.m, 4, c.batch);
        let (wide_a, wide_b) = (
            tlinalg_testkit::widen(&a.data),
            tlinalg_testkit::widen(&b.data),
        );
        Self {
            a,
            b,
            wide_a,
            wide_b,
        }
    }
}

/// `out = a · op(b)` in column-major, allocation-free.
///
/// `a` is `ar x ac` with `ar = a.len() / ac`, `op(b)` is `ac x bc` with `bc = out.len() / ar`, and
/// `b`'s leading dimension is `ldb`. `conj_b` reads `b` as `bᴴ`, i.e. `b` is then `bc x ac` and each
/// entry is conjugated as it is loaded — the transpose a `Qᴴ` or `Lᴴ` product needs, without
/// materialising it.
///
/// The loop order (`j`, `k`, `i`) and the real/imag split mirror the naive triple loop this
/// replaced, so a rewritten product rounds exactly as the one it replaced did.
fn gemm(
    ac: usize,
    a: &[Complex64],
    b: &[Complex64],
    ldb: usize,
    conj_b: bool,
    out: &mut [Complex64],
) {
    let ar = a.len() / ac;
    let bc = out.len() / ar;
    debug_assert_eq!(a.len(), ar * ac, "gemm: a is not `ar x {ac}`");
    debug_assert_eq!(out.len(), ar * bc, "gemm: out is not `{ar} x {bc}`");
    out.fill(ZERO);
    for j in 0..bc {
        for k in 0..ac {
            let bv = if conj_b {
                b[j + k * ldb].conj()
            } else {
                b[k + j * ldb]
            };
            let (br, bi) = (bv.re, bv.im);
            let column = &a[k * ar..k * ar + ar];
            let target = &mut out[j * ar..j * ar + ar];
            for (o, av) in target.iter_mut().zip(column) {
                let (xr, xi) = (av.re, av.im);
                o.re += xr * br - xi * bi;
                o.im += xr * bi + xi * br;
            }
        }
    }
}

/// `Σ conj(x[i]) · y[i]`: the inner product a Gram entry `(QᴴQ)[j][k]` is made of.
fn dot_conj(x: &[Complex64], y: &[Complex64]) -> Complex64 {
    let (mut re, mut im) = (0.0, 0.0);
    for (a, b) in x.iter().zip(y) {
        re += a.re * b.re + a.im * b.im;
        im += a.re * b.im - a.im * b.re;
    }
    Complex64::new(re, im)
}

fn absmax(x: &[Complex64]) -> f64 {
    x.iter().map(|z| z.norm()).fold(0.0, f64::max)
}
fn gate(residual: f64, scale: f64, bound: f64) -> Result<f64, String> {
    let limit = bound * scale.max(1.0);
    if residual > limit {
        Err(format!("measured residual={residual:.3e} > {limit:.3e}"))
    } else {
        Ok(residual)
    }
}
fn get(a: &[Complex64], rows: usize, row: usize, col: usize) -> Complex64 {
    a[row + col * rows]
}
fn mat_residual(a: &[Complex64], b: &[Complex64]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (*x - *y).norm())
        .fold(0.0, f64::max)
}
/// `‖QᴴQ − I‖max` for the `rows x cols` factor `Q` of a thin decomposition.
fn orth(q: &[Complex64], rows: usize, cols: usize) -> f64 {
    let mut r: f64 = 0.0;
    for j in 0..cols {
        for k in 0..cols {
            let x = dot_conj(&q[j * rows..j * rows + rows], &q[k * rows..k * rows + rows]);
            r = r.max((x - if j == k { ONE } else { ZERO }).norm());
        }
    }
    r
}
/// `out = a` with its rows permuted by the 1-based `piv`, into a buffer the caller reuses.
fn permute_rows_into(a: &[Complex64], n: usize, piv: &[i32], out: &mut [Complex64]) {
    out.copy_from_slice(a);
    for (i, &p) in piv.iter().enumerate() {
        let p = p as usize - 1;
        if p != i {
            for col in 0..n {
                out.swap(i + col * n, p + col * n);
            }
        }
    }
}
fn parity_from_pivots(piv: &[i32]) -> f64 {
    if piv
        .iter()
        .enumerate()
        .filter(|(i, p)| **p != *i as i32 + 1)
        .count()
        .is_multiple_of(2)
    {
        1.
    } else {
        -1.
    }
}
fn parity_of_perm(p: &[Complex64], n: usize) -> f64 {
    let mut perm = vec![0usize; n];
    for (row, slot) in perm.iter_mut().enumerate() {
        *slot = (0..n)
            .find(|&col| get(p, n, row, col).norm() > 0.5)
            .unwrap_or(row);
    }
    let inv = (0..n)
        .flat_map(|i| (i + 1..n).map(move |j| (i, j)))
        .filter(|&(i, j)| perm[i] > perm[j])
        .count();
    if inv.is_multiple_of(2) {
        1.
    } else {
        -1.
    }
}
fn check_solve(
    a: &[Complex64],
    b: &[Complex64],
    x: &[Complex64],
    c: Case,
    bound: f64,
    triangular: bool,
) -> Result<f64, String> {
    let n = c.m;
    let mut worst: f64 = 0.;
    for batch in 0..c.batch {
        let ao = batch * n * n;
        let xo = batch * n * 4;
        let bo = batch * n * 4;
        let an = if triangular {
            a[ao..ao + n * n]
                .iter()
                .enumerate()
                .filter(|(i, _)| i % n <= i / n)
                .map(|(_, z)| z.norm())
                .fold(0., f64::max)
        } else {
            absmax(&a[ao..ao + n * n])
        };
        let xn = absmax(&x[xo..xo + n * 4]);
        let bn = absmax(&b[bo..bo + n * 4]);
        for col in 0..4 {
            for row in 0..n {
                let mut lhs = Complex64::new(0., 0.);
                for k in 0..n {
                    let av = if triangular && k > row {
                        Complex64::new(0., 0.)
                    } else {
                        a[ao + row + k * n]
                    };
                    lhs += av * x[xo + k + col * n];
                }
                worst = worst.max((lhs - b[bo + row + col * n]).norm());
            }
        }
        let scale = an * xn + bn;
        if worst > bound * scale.max(1.) {
            return Err(format!(
                "measured residual={worst:.3e} > {:.3e}",
                bound * scale.max(1.)
            ));
        }
    }
    Ok(worst)
}
fn check_packed<T: BenchScalar>(
    a: &[Complex64],
    lu: &[T],
    piv: &[i32],
    parity: &[T],
    c: Case,
    bound: f64,
) -> Result<f64, String> {
    if piv.len() < c.batch * c.m || piv.iter().any(|&p| p < 1 || p as usize > c.m) {
        return Err("pivot vector is invalid".into());
    }
    let n = c.m;
    let packed = tlinalg_testkit::widen(lu);
    let mut worst: f64 = 0.;
    // The packed factor holds unit-lower `L` and upper `U` in one array; split them once per
    // matrix so the product is a plain GEMM instead of a triple loop with two triangular tests.
    let mut pa = vec![ZERO; n * n];
    let mut lo = vec![ZERO; n * n];
    let mut up = vec![ZERO; n * n];
    let mut rec = vec![ZERO; n * n];
    for batch in 0..c.batch {
        let off = batch * n * n;
        let pivot_off = batch * n;
        let expected_parity = parity_from_pivots(&piv[pivot_off..pivot_off + n]);
        permute_rows_into(
            &a[off..off + n * n],
            n,
            &piv[pivot_off..pivot_off + n],
            &mut pa,
        );
        for col in 0..n {
            for row in 0..n {
                let v = packed[off + row + col * n];
                up[row + col * n] = if row <= col { v } else { ZERO };
                lo[row + col * n] = if row == col {
                    ONE
                } else if row > col {
                    v
                } else {
                    ZERO
                };
            }
        }
        gemm(n, &lo, &up, n, false, &mut rec);
        worst = worst.max(mat_residual(&pa, &rec));
        if (tlinalg_testkit::widen(&parity[batch..batch + 1])[0].re - expected_parity).abs() > bound
        {
            return Err("permutation parity mismatch".into());
        }
    }
    gate(worst, absmax(a), bound)
}
fn check_explicit(
    a: &[Complex64],
    p: &[Complex64],
    l: &[Complex64],
    u: &[Complex64],
    parity: &[Complex64],
    c: Case,
    bound: f64,
) -> Result<f64, String> {
    let n = c.m;
    let mut worst: f64 = 0.;
    let mut pa = vec![ZERO; n * n];
    let mut rec = vec![ZERO; n * n];
    for (batch, parity_value) in parity.iter().enumerate().take(c.batch) {
        let off = batch * n * n;
        gemm(
            n,
            &p[off..off + n * n],
            &a[off..off + n * n],
            n,
            false,
            &mut pa,
        );
        gemm(
            n,
            &l[off..off + n * n],
            &u[off..off + n * n],
            n,
            false,
            &mut rec,
        );
        worst = worst.max(mat_residual(&pa, &rec));
        let expected = parity_of_perm(&p[off..off + n * n], n);
        if (parity_value.re - expected).abs() > bound {
            return Err("permutation parity mismatch".into());
        }
    }
    gate(worst, absmax(a), bound)
}
#[allow(clippy::too_many_arguments)]
fn check_full(
    a: &[Complex64],
    p: &[Complex64],
    l: &[Complex64],
    u: &[Complex64],
    q: &[Complex64],
    parity: &[Complex64],
    c: Case,
    bound: f64,
) -> Result<f64, String> {
    let n = c.m;
    let mut worst: f64 = 0.;
    let mut pa = vec![ZERO; n * n];
    let mut left = vec![ZERO; n * n];
    let mut rec = vec![ZERO; n * n];
    for (batch, parity_value) in parity.iter().enumerate().take(c.batch) {
        let off = batch * n * n;
        gemm(
            n,
            &p[off..off + n * n],
            &a[off..off + n * n],
            n,
            false,
            &mut pa,
        );
        // `pa · Qᴴ`, with the conjugate folded into the load instead of materialising `Qᴴ`.
        gemm(n, &pa, &q[off..off + n * n], n, true, &mut left);
        gemm(
            n,
            &l[off..off + n * n],
            &u[off..off + n * n],
            n,
            false,
            &mut rec,
        );
        worst = worst.max(mat_residual(&left, &rec));
        let expected =
            parity_of_perm(&p[off..off + n * n], n) * parity_of_perm(&q[off..off + n * n], n);
        if (parity_value.re - expected).abs() > bound {
            return Err("permutation parity mismatch".into());
        }
    }
    gate(worst, absmax(a), bound)
}
fn check_eigen(
    a: &[Complex64],
    w: &[Complex64],
    v: &[Complex64],
    c: Case,
    bound: f64,
) -> Result<f64, String> {
    let n = c.m;
    let mut worst: f64 = 0.;
    let mut av = vec![ZERO; n * n];
    for batch in 0..c.batch {
        let ao = batch * n * n;
        let vo = ao;
        let wo = batch * n;
        let q = &v[vo..vo + n * n];
        worst = worst.max(orth(q, n, n));
        gemm(n, &a[ao..ao + n * n], q, n, false, &mut av);
        for col in 0..n {
            let lambda = w[wo + col];
            for row in 0..n {
                worst = worst.max((av[row + col * n] - q[row + col * n] * lambda).norm());
            }
        }
    }
    gate(worst, absmax(a), bound)
}
fn check_hermitian_values(w: &[Complex64], c: Case, bound: f64) -> Result<(), String> {
    for b in 0..c.batch {
        let values = &w[b * c.m..(b + 1) * c.m];
        if values.iter().any(|x| x.im.abs() > bound)
            || values.windows(2).any(|x| x[1].re < x[0].re - bound)
        {
            return Err("eigenvalues are not real and ascending".into());
        }
    }
    Ok(())
}
fn check_conjugate_pairs(w: &[Complex64], c: Case, bound: f64) -> Result<(), String> {
    for b in 0..c.batch {
        let values = &w[b * c.m..(b + 1) * c.m];
        for (i, value) in values.iter().enumerate() {
            if !values.iter().enumerate().any(|(j, other)| {
                j == i && value.im.abs() <= bound || (*other - value.conj()).norm() <= bound
            }) {
                return Err("real-input eigenvalues lack conjugate pairs".into());
            }
        }
    }
    Ok(())
}
fn house_q<T: BenchScalar>(state: &[T], tau: &[T], c: Case) -> Vec<Complex64> {
    let a = tlinalg_testkit::widen(state);
    let t = tlinalg_testkit::widen(tau);
    let n = c.m;
    let k = c.m.min(c.n);
    let mut q = vec![Complex64::new(0., 0.); c.batch * n * k];
    for b in 0..c.batch {
        for j in 0..k {
            q[b * n * k + j * n + j] = Complex64::new(1., 0.);
        }
        for h in (0..k).rev() {
            let coeff = t[b * k + h];
            if coeff.norm() < f64::EPSILON {
                continue;
            }
            let denom = coeff;
            for col in 0..k {
                let mut inner = Complex64::new(0., 0.);
                for row in h..n {
                    let v = if row == h {
                        Complex64::new(1., 0.)
                    } else {
                        a[b * n * c.n + row + h * n]
                    };
                    inner += v.conj() * q[b * n * k + row + col * n];
                }
                for row in h..n {
                    let v = if row == h {
                        Complex64::new(1., 0.)
                    } else {
                        a[b * n * c.n + row + h * n]
                    };
                    q[b * n * k + row + col * n] -= v * inner * denom;
                }
            }
        }
    }
    q
}
fn check_house<T: BenchScalar>(
    state: &[T],
    tau: &[T],
    target: Option<&[T]>,
    c: Case,
    bound: f64,
) -> Result<f64, String> {
    let q = house_q(state, tau, c);
    let mut worst: f64 = 0.;
    for b in 0..c.batch {
        worst = worst.max(orth(
            &q[b * c.m * c.m.min(c.n)..(b + 1) * c.m * c.m.min(c.n)],
            c.m,
            c.m.min(c.n),
        ));
    }
    if let Some(target) = target {
        let got = tlinalg_testkit::widen(target);
        worst = worst.max(mat_residual(&got, &q));
    }
    gate(worst, 1., bound)
}

fn check_one<T: BenchScalar + super::rows::EigBuffers + super::rows::RealValues>(
    family: &str,
    c: Case,
    out: Output<T>,
    r: &Reference<T>,
) -> Result<f64, String> {
    let bound = tol::<T>(c);
    let (a, b) = (r.wide_a.as_slice(), r.wide_b.as_slice());
    let scale = absmax(a);
    match out {
        Output::PackedLu { lu, piv, parity } => check_packed(a, &lu, &piv, &parity, c, bound),
        Output::Lu { p, l, u, parity } => {
            let p = tlinalg_testkit::widen(&p);
            let l = tlinalg_testkit::widen(&l);
            let u = tlinalg_testkit::widen(&u);
            let parity = tlinalg_testkit::widen(&parity);
            check_explicit(a, &p, &l, &u, &parity, c, bound)
        }
        Output::FullLu { p, l, u, q, parity } => {
            let p = tlinalg_testkit::widen(&p);
            let l = tlinalg_testkit::widen(&l);
            let u = tlinalg_testkit::widen(&u);
            let q = tlinalg_testkit::widen(&q);
            let parity = tlinalg_testkit::widen(&parity);
            check_full(a, &p, &l, &u, &q, &parity, c, bound)
        }
        Output::One(x)
            if matches!(
                family,
                "solve" | "lu_solve_prepared" | "lu_factor_solve" | "full_piv_lu_solve"
            ) =>
        {
            let x = tlinalg_testkit::widen(&x);
            check_solve(a, b, &x, c, bound, false)
        }
        Output::One(x) if family == "triangular_solve" => {
            let x = tlinalg_testkit::widen(&x);
            check_solve(a, b, &x, c, bound, true)
        }
        Output::One(l) if family == "cholesky" => {
            let l = tlinalg_testkit::widen(&l);
            let n = c.m;
            let mut worst: f64 = 0.;
            // `L·Lᴴ` needs `L`'s upper triangle to be zero, and only the lower one is written by
            // the factorization, so the factor is masked into a reused buffer first.
            let mut lo = vec![ZERO; n * n];
            let mut rec = vec![ZERO; n * n];
            for batch in 0..c.batch {
                let off = batch * n * n;
                for col in 0..n {
                    for row in 0..n {
                        lo[row + col * n] = if row >= col {
                            l[off + row + col * n]
                        } else {
                            ZERO
                        };
                    }
                }
                gemm(n, &lo, &lo, n, true, &mut rec);
                worst = worst.max(mat_residual(&a[off..off + n * n], &rec));
            }
            gate(worst, scale, bound)
        }
        Output::Two(q, r) if family == "qr" => {
            let q = tlinalg_testkit::widen(&q);
            let r = tlinalg_testkit::widen(&r);
            let k = c.m.min(c.n);
            let mut worst: f64 = 0.;
            let mut rec = vec![ZERO; c.m * c.n];
            for bch in 0..c.batch {
                let qo = bch * c.m * k;
                let ro = bch * k * c.n;
                gemm(
                    k,
                    &q[qo..qo + c.m * k],
                    &r[ro..ro + k * c.n],
                    k,
                    false,
                    &mut rec,
                );
                worst = worst.max(mat_residual(
                    &a[bch * c.m * c.n..(bch + 1) * c.m * c.n],
                    &rec,
                ));
                worst = worst.max(orth(&q[qo..qo + c.m * k], c.m, k));
            }
            gate(worst, scale, bound)
        }
        Output::TwoPerm(q, r, p) => {
            if p.len() != c.batch * c.n || p.iter().any(|&x| x < 0 || x as usize >= c.n) {
                return Err("QR permutation vector is invalid".into());
            }
            for chunk in p.chunks_exact(c.n) {
                let mut seen = vec![false; c.n];
                for &x in chunk {
                    if std::mem::replace(&mut seen[x as usize], true) {
                        return Err("QR permutation vector repeats a column".into());
                    }
                }
            }
            let q = tlinalg_testkit::widen(&q);
            let r = tlinalg_testkit::widen(&r);
            let k = c.m.min(c.n);
            let mut worst: f64 = 0.;
            let mut ap = vec![ZERO; c.m * c.n];
            let mut rec = vec![ZERO; c.m * c.n];
            for bch in 0..c.batch {
                let ao = bch * c.m * c.n;
                let qo = bch * c.m * k;
                let ro = bch * k * c.n;
                for col in 0..c.n {
                    let src = p[bch * c.n + col] as usize;
                    for row in 0..c.m {
                        ap[row + col * c.m] = a[ao + row + src * c.m];
                    }
                }
                gemm(
                    k,
                    &q[qo..qo + c.m * k],
                    &r[ro..ro + k * c.n],
                    k,
                    false,
                    &mut rec,
                );
                worst = worst
                    .max(mat_residual(&ap, &rec))
                    .max(orth(&q[qo..qo + c.m * k], c.m, k));
            }
            gate(worst, scale, bound)
        }
        Output::Three(u, s, vt) => {
            let u = tlinalg_testkit::widen(&u);
            let s = tlinalg_testkit::widen(&s);
            let vt = tlinalg_testkit::widen(&vt);
            check_svd(a, &u, &s, &vt, c, bound, false)
        }
        Output::ThreeReal(u, s, vt) => {
            let u = tlinalg_testkit::widen(&u);
            let vt = tlinalg_testkit::widen(&vt);
            check_svd_real(a, &u, &s, &vt, c, bound, family == "svd_full")
        }
        Output::Two(w, v) => {
            let w = tlinalg_testkit::widen(&w);
            let v = tlinalg_testkit::widen(&v);
            check_hermitian_values(&w, c, bound)?;
            check_eigen(a, &w, &v, c, bound)
        }
        Output::TwoReal(w, v) if family == "eigh" => {
            let v = tlinalg_testkit::widen(&v);
            let w = w
                .into_iter()
                .map(|x| Complex64::new(x, 0.))
                .collect::<Vec<_>>();
            check_hermitian_values(&w, c, bound)?;
            check_eigen(a, &w, &v, c, bound)
        }
        Output::TwoReal(w, _) if family == "eigvalsh" => {
            let w = w
                .into_iter()
                .map(|x| Complex64::new(x, 0.))
                .collect::<Vec<_>>();
            check_hermitian_values(&w, c, bound)?;
            Ok(0.)
        }
        Output::One(s) if family == "svd_values" => {
            let thin = output_call_with::<T>("svd_thin", c, false, &r.a, &r.b)
                .map_err(|e| format!("reference SVD: {e}"))?;
            let vals = match thin {
                Output::ThreeReal(_, s, _) => s
                    .into_iter()
                    .map(|x| Complex64::new(x, 0.))
                    .collect::<Vec<_>>(),
                _ => return Err("SVD output unavailable".into()),
            };
            let got = tlinalg_testkit::widen(&s);
            let r = got
                .iter()
                .zip(vals.iter())
                .map(|(x, y)| (*x - *y).norm())
                .fold(0., f64::max);
            gate(r, scale, bound)
        }
        Output::Eig(w, v) if family == "eig" => {
            if !<T as tlinalg_testkit::TestScalar>::COMPLEX {
                check_conjugate_pairs(&w, c, bound)?;
            }
            check_eigen(a, &w, &v, c, bound)
        }
        Output::Eig(w, _) if family == "eigvals" => {
            let eig = output_call_with::<T>("eig", c, false, &r.a, &r.b)
                .map_err(|e| format!("reference eig: {e}"))?;
            let (ew, ev) = match eig {
                Output::Eig(w, v) => (w, v),
                _ => return Err("eigen output unavailable".into()),
            };
            if !<T as tlinalg_testkit::TestScalar>::COMPLEX {
                check_conjugate_pairs(&ew, c, bound)?;
            }
            let n = c.m;
            let mut worst: f64 = 0.;
            // Per batch: how far each returned value is from the nearest reference value. The
            // eigenvector residual is one check for the case, not one per batch.
            for bch in 0..c.batch {
                for j in 0..n {
                    let mut best = f64::INFINITY;
                    for k in 0..n {
                        let d = (w[bch * n + j] - ew[bch * n + k]).norm();
                        if d < best {
                            best = d;
                        }
                    }
                    worst = worst.max(best);
                }
            }
            worst = worst.max(check_eigen(a, &ew, &ev, c, bound)?);
            gate(worst, scale, bound)
        }
        Output::Householder { a, tau, target } => {
            check_house(&a, &tau, target.as_deref(), c, bound)
        }
        _ => Err(format!("unexpected output for {family}")),
    }
}
fn check_svd(
    a: &[Complex64],
    u: &[Complex64],
    s: &[Complex64],
    vt: &[Complex64],
    c: Case,
    bound: f64,
    _full: bool,
) -> Result<f64, String> {
    let k = c.m.min(c.n);
    let mut worst: f64 = 0.;
    let mut us = vec![ZERO; c.m * k];
    let mut rec = vec![ZERO; c.m * c.n];
    for bch in 0..c.batch {
        let uo = bch * c.m * k;
        let vo = bch * k * c.n;
        worst = worst.max(orth(&u[uo..uo + c.m * k], c.m, k));
        for i in 1..k {
            if s[bch * k + i].re > s[bch * k + i - 1].re + bound || s[bch * k + i].re < -bound {
                return Err("singular values are not descending and non-negative".into());
            }
        }
        // `U·diag(s)·V`: the singular values are folded into `U`'s columns, one GEMM per matrix.
        for col in 0..k {
            let sigma = s[bch * k + col];
            for row in 0..c.m {
                us[row + col * c.m] = u[uo + row + col * c.m] * sigma;
            }
        }
        gemm(k, &us, &vt[vo..vo + k * c.n], k, false, &mut rec);
        worst = worst.max(mat_residual(
            &a[bch * c.m * c.n..(bch + 1) * c.m * c.n],
            &rec,
        ));
    }
    gate(worst, absmax(a), bound)
}
fn check_svd_real(
    a: &[Complex64],
    u: &[Complex64],
    s: &[f64],
    vt: &[Complex64],
    c: Case,
    bound: f64,
    full: bool,
) -> Result<f64, String> {
    let k = c.m.min(c.n);
    let uc = if full { c.m } else { k };
    let vr = if full { c.n } else { k };
    let mut worst: f64 = 0.;
    let mut us = vec![ZERO; c.m * k];
    let mut rec = vec![ZERO; c.m * c.n];
    for bch in 0..c.batch {
        let uo = bch * c.m * uc;
        let vo = bch * vr * c.n;
        worst = worst.max(orth(&u[uo..uo + c.m * uc], c.m, uc));
        for i in 1..k {
            if s[bch * k + i] > s[bch * k + i - 1] + bound || s[bch * k + i] < -bound {
                return Err("singular values are not descending and non-negative".into());
            }
        }
        for z in 0..k {
            let sigma = Complex64::new(s[bch * k + z], 0.);
            for row in 0..c.m {
                us[row + z * c.m] = u[uo + row + z * c.m] * sigma;
            }
        }
        gemm(k, &us, &vt[vo..vo + vr * c.n], vr, false, &mut rec);
        worst = worst.max(mat_residual(
            &a[bch * c.m * c.n..(bch + 1) * c.m * c.n],
            &rec,
        ));
    }
    gate(worst, absmax(a), bound)
}

/// Run the gate for every requested family and case.
pub(crate) fn verify<T: BenchScalar + super::rows::EigBuffers + super::rows::RealValues>(
    families: &[String],
    cases: &[Case],
    _env: &Env,
) -> bool {
    let mut clean = true;
    for family in families {
        for &case in cases {
            if !applicable(family, case) {
                println!(
                    "SKIPPED {family} {} {}: shape not applicable",
                    T::LABEL,
                    case
                );
                continue;
            }
            let reference = Reference::<T>::new(family, case);
            let f = output_call_with::<T>(family, case, false, &reference.a, &reference.b)
                .and_then(|o| check_one::<T>(family, case, o, &reference));
            let l = if crate::vendor::LINKED {
                output_call_with::<T>(family, case, true, &reference.a, &reference.b)
                    .and_then(|o| check_one::<T>(family, case, o, &reference))
            } else {
                Ok(0.)
            };
            match (f, l) {
                (Err(e), _) | (_, Err(e)) if matches!(family.as_str(), "full_piv_lu" | "full_piv_lu_solve")
                    && e.to_ascii_lowercase().contains("singular") => println!(
                    "OK {family} {} {} residual=0.000e0 provider_residual=0.000e0 note=documented singular failure: {e}", T::LABEL, case),
                (Ok(x), Ok(y)) => println!(
                    "OK {family} {} {} residual={x:.3e} provider_residual={y:.3e}",
                    T::LABEL,
                    case
                ),
                (Err(e), _) => {
                    println!("MISMATCH {family} {} {}: faer: {e}", T::LABEL, case);
                    clean = false
                }
                (_, Err(e)) => {
                    println!("MISMATCH {family} {} {}: lapack: {e}", T::LABEL, case);
                    clean = false
                }
            }
        }
    }
    clean
}

#[cfg(test)]
mod tests;
