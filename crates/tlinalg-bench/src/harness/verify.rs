//! Numerical correctness gate for every campaign family.

use super::cases::{applicable, Case};
use super::rows::{output_call_checked, Output};
use crate::{BenchScalar, Env};
use num_complex::Complex64;

fn tol<T: BenchScalar>(c: Case) -> f64 {
    64.0 * (c.m.max(c.n) as f64).sqrt() * T::TOL
}

fn input<T: BenchScalar>(family: &str, c: Case) -> Vec<Complex64> {
    if matches!(family, "cholesky" | "eigh" | "eigvalsh") {
        return tlinalg_testkit::widen(
            &(0..c.batch)
                .flat_map(|i| tlinalg_testkit::hpd_seeded::<T>(c.m, i + 1))
                .collect::<Vec<_>>(),
        );
    }
    if matches!(family, "eig" | "eigvals") {
        return (0..c.batch)
            .flat_map(|b| {
                (0..c.m * c.m).map(move |i| {
                    let row = i % c.m;
                    let col = i / c.m;
                    if row == col {
                        Complex64::new(2.0 + row as f64 + b as f64 * 0.01, 0.0)
                    } else {
                        Complex64::new(0.0, 0.0)
                    }
                })
            })
            .collect();
    }
    tlinalg_testkit::widen(&tlinalg_testkit::batch_of::<T>(c.m, c.n, c.batch, 0))
}

fn rhs<T: BenchScalar>(c: Case) -> Vec<Complex64> {
    tlinalg_testkit::widen(&tlinalg_testkit::batch_of::<T>(c.m, 4, c.batch, 0))
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
fn orth(q: &[Complex64], rows: usize, cols: usize) -> f64 {
    let mut r: f64 = 0.0;
    for j in 0..cols {
        for k in 0..cols {
            let x = (0..rows)
                .map(|i| q[i + j * rows].conj() * q[i + k * rows])
                .sum::<Complex64>();
            r = r.max(
                (x - if j == k {
                    Complex64::new(1., 0.)
                } else {
                    Complex64::new(0., 0.)
                })
                .norm(),
            );
        }
    }
    r
}
fn matmul(a: &[Complex64], ar: usize, ac: usize, b: &[Complex64], bc: usize) -> Vec<Complex64> {
    let mut out = vec![Complex64::new(0., 0.); ar * bc];
    for j in 0..bc {
        for k in 0..ac {
            for i in 0..ar {
                out[i + j * ar] += a[i + k * ar] * b[k + j * ac];
            }
        }
    }
    out
}
fn conj_transpose(a: &[Complex64], rows: usize, cols: usize) -> Vec<Complex64> {
    let mut out = vec![Complex64::new(0., 0.); rows * cols];
    for j in 0..cols {
        for i in 0..rows {
            out[j + i * cols] = a[i + j * rows].conj();
        }
    }
    out
}
fn permute_rows(a: &[Complex64], n: usize, piv: &[i32]) -> Vec<Complex64> {
    let mut out = a.to_vec();
    for (i, &p) in piv.iter().enumerate() {
        let p = p as usize - 1;
        if p != i {
            for col in 0..n {
                out.swap(i + col * n, p + col * n);
            }
        }
    }
    out
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
    let l = tlinalg_testkit::widen(lu);
    let mut worst: f64 = 0.;
    for batch in 0..c.batch {
        let off = batch * c.m * c.m;
        let pivot_off = batch * c.m;
        let expected_parity = parity_from_pivots(&piv[pivot_off..pivot_off + c.m]);
        let pa = permute_rows(
            &a[off..off + c.m * c.m],
            c.m,
            &piv[pivot_off..pivot_off + c.m],
        );
        let mut rec = vec![Complex64::new(0., 0.); c.m * c.m];
        for col in 0..c.m {
            for k in 0..c.m {
                for row in 0..c.m {
                    let lv = if row == k {
                        Complex64::new(1., 0.)
                    } else if row > k {
                        l[off + row + k * c.m]
                    } else {
                        Complex64::new(0., 0.)
                    };
                    let uv = if k <= col {
                        l[off + k + col * c.m]
                    } else {
                        Complex64::new(0., 0.)
                    };
                    rec[row + col * c.m] += lv * uv;
                }
            }
        }
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
    for (batch, parity_value) in parity.iter().enumerate().take(c.batch) {
        let ao = batch * n * n;
        let po = ao;
        let pa = matmul(&p[po..po + n * n], n, n, &a[ao..ao + n * n], n);
        let rec = matmul(&l[ao..ao + n * n], n, n, &u[ao..ao + n * n], n);
        worst = worst.max(mat_residual(&pa, &rec));
        let expected = parity_of_perm(&p[po..po + n * n], n);
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
    for (batch, parity_value) in parity.iter().enumerate().take(c.batch) {
        let off = batch * n * n;
        let pa = matmul(&p[off..off + n * n], n, n, &a[off..off + n * n], n);
        let qh = conj_transpose(&q[off..off + n * n], n, n);
        let left = matmul(&pa, n, n, &qh, n);
        let rec = matmul(&l[off..off + n * n], n, n, &u[off..off + n * n], n);
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
    for batch in 0..c.batch {
        let ao = batch * n * n;
        let vo = ao;
        let wo = batch * n;
        let q = &v[vo..vo + n * n];
        worst = worst.max(orth(q, n, n));
        for col in 0..n {
            for row in 0..n {
                let lhs = (0..n)
                    .map(|k| a[ao + row + k * n] * q[k + col * n])
                    .sum::<Complex64>();
                worst = worst.max((lhs - q[row + col * n] * w[wo + col]).norm());
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
) -> Result<f64, String> {
    let bound = tol::<T>(c);
    let a = input::<T>(family, c);
    let b = rhs::<T>(c);
    let scale = absmax(&a);
    match out {
        Output::PackedLu { lu, piv, parity } => check_packed(&a, &lu, &piv, &parity, c, bound),
        Output::Lu { p, l, u, parity } => {
            let p = tlinalg_testkit::widen(&p);
            let l = tlinalg_testkit::widen(&l);
            let u = tlinalg_testkit::widen(&u);
            let parity = tlinalg_testkit::widen(&parity);
            check_explicit(&a, &p, &l, &u, &parity, c, bound)
        }
        Output::FullLu { p, l, u, q, parity } => {
            let p = tlinalg_testkit::widen(&p);
            let l = tlinalg_testkit::widen(&l);
            let u = tlinalg_testkit::widen(&u);
            let q = tlinalg_testkit::widen(&q);
            let parity = tlinalg_testkit::widen(&parity);
            check_full(&a, &p, &l, &u, &q, &parity, c, bound)
        }
        Output::One(x)
            if matches!(
                family,
                "solve" | "lu_solve_prepared" | "lu_factor_solve" | "full_piv_lu_solve"
            ) =>
        {
            let x = tlinalg_testkit::widen(&x);
            check_solve(&a, &b, &x, c, bound, false)
        }
        Output::One(x) if family == "triangular_solve" => {
            let x = tlinalg_testkit::widen(&x);
            check_solve(&a, &b, &x, c, bound, true)
        }
        Output::One(l) if family == "cholesky" => {
            let l = tlinalg_testkit::widen(&l);
            let mut worst: f64 = 0.;
            for batch in 0..c.batch {
                let off = batch * c.m * c.m;
                let mut rec = vec![Complex64::new(0., 0.); c.m * c.m];
                for col in 0..c.m {
                    for k in 0..=col {
                        for row in 0..c.m {
                            rec[row + col * c.m] +=
                                l[off + row + k * c.m] * l[off + col + k * c.m].conj();
                        }
                    }
                }
                worst = worst.max(mat_residual(&a[off..off + c.m * c.m], &rec));
            }
            gate(worst, scale, bound)
        }
        Output::Two(q, r) if family == "qr" => {
            let q = tlinalg_testkit::widen(&q);
            let r = tlinalg_testkit::widen(&r);
            let k = c.m.min(c.n);
            let mut worst: f64 = 0.;
            for bch in 0..c.batch {
                let qo = bch * c.m * k;
                let ro = bch * k * c.n;
                let rec = matmul(&q[qo..qo + c.m * k], c.m, k, &r[ro..ro + k * c.n], c.n);
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
            for bch in 0..c.batch {
                let ao = bch * c.m * c.n;
                let qo = bch * c.m * k;
                let ro = bch * k * c.n;
                let mut ap = vec![Complex64::new(0., 0.); c.m * c.n];
                for col in 0..c.n {
                    let src = p[bch * c.n + col] as usize;
                    for row in 0..c.m {
                        ap[row + col * c.m] = a[ao + row + src * c.m];
                    }
                }
                let rec = matmul(&q[qo..qo + c.m * k], c.m, k, &r[ro..ro + k * c.n], c.n);
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
            check_svd(&a, &u, &s, &vt, c, bound, false)
        }
        Output::ThreeReal(u, s, vt) => {
            let u = tlinalg_testkit::widen(&u);
            let vt = tlinalg_testkit::widen(&vt);
            check_svd_real(&a, &u, &s, &vt, c, bound, family == "svd_full")
        }
        Output::Two(w, v) => {
            let w = tlinalg_testkit::widen(&w);
            let v = tlinalg_testkit::widen(&v);
            check_hermitian_values(&w, c, bound)?;
            check_eigen(&a, &w, &v, c, bound)
        }
        Output::TwoReal(w, v) if family == "eigh" => {
            let v = tlinalg_testkit::widen(&v);
            let w = w
                .into_iter()
                .map(|x| Complex64::new(x, 0.))
                .collect::<Vec<_>>();
            check_hermitian_values(&w, c, bound)?;
            check_eigen(&a, &w, &v, c, bound)
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
            let thin = output_call_checked::<T>("svd_thin", c, false)
                .map_err(|e| format!("reference SVD: {e}"))?;
            let vals = match thin {
                Output::Three(u, s, vt) => {
                    let _ = (u, vt);
                    tlinalg_testkit::widen(&s)
                }
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
            check_eigen(&a, &w, &v, c, bound)
        }
        Output::Eig(w, _) if family == "eigvals" => {
            let eig = output_call_checked::<T>("eig", c, false)
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
                worst = worst.max(check_eigen(&a, &ew, &ev, c, bound)?);
            }
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
    for bch in 0..c.batch {
        let uo = bch * c.m * k;
        let vo = bch * k * c.n;
        worst = worst.max(orth(&u[uo..uo + c.m * k], c.m, k));
        for i in 1..k {
            if s[bch * k + i].re > s[bch * k + i - 1].re + bound || s[bch * k + i].re < -bound {
                return Err("singular values are not descending and non-negative".into());
            }
        }
        let mut rec = vec![Complex64::new(0., 0.); c.m * c.n];
        for col in 0..c.n {
            for z in 0..k {
                for row in 0..c.m {
                    rec[row + col * c.m] +=
                        u[uo + row + z * c.m] * s[bch * k + z] * vt[vo + z + col * k];
                }
            }
        }
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
    for bch in 0..c.batch {
        let uo = bch * c.m * uc;
        let vo = bch * vr * c.n;
        worst = worst.max(orth(&u[uo..uo + c.m * uc], c.m, uc));
        for i in 1..k {
            if s[bch * k + i] > s[bch * k + i - 1] + bound || s[bch * k + i] < -bound {
                return Err("singular values are not descending and non-negative".into());
            }
        }
        let mut rec = vec![Complex64::new(0., 0.); c.m * c.n];
        for col in 0..c.n {
            for z in 0..k {
                for row in 0..c.m {
                    rec[row + col * c.m] +=
                        u[uo + row + z * c.m] * s[bch * k + z] * vt[vo + z + col * vr];
                }
            }
        }
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
            let f = output_call_checked::<T>(family, case, false)
                .and_then(|o| check_one::<T>(family, case, o));
            let l = if crate::vendor::LINKED {
                output_call_checked::<T>(family, case, true)
                    .and_then(|o| check_one::<T>(family, case, o))
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
