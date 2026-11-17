use super::*;

/// The naive triple loop `gemm` replaced, kept as the oracle: `out = a · b` (`conj_b` reads `bᴴ`),
/// column-major, `a` being `ar x ac` and `b`'s leading dimension `ldb`.
fn gemm_oracle(
    ar: usize,
    ac: usize,
    a: &[Complex64],
    b: &[Complex64],
    ldb: usize,
    conj_b: bool,
    bc: usize,
) -> Vec<Complex64> {
    let mut out = vec![ZERO; ar * bc];
    for j in 0..bc {
        for k in 0..ac {
            let bv = if conj_b {
                b[j + k * ldb].conj()
            } else {
                b[k + j * ldb]
            };
            for i in 0..ar {
                out[i + j * ar] += a[i + k * ar] * bv;
            }
        }
    }
    out
}

/// `(n*m + m + k)` deterministic complex entries, so shapes need not be square or equal.
fn fill(len: usize, seed: f64) -> Vec<Complex64> {
    (0..len)
        .map(|i| {
            Complex64::new(
                (i as f64 * 0.37 + seed).sin(),
                (i as f64 * 0.11 + seed).cos(),
            )
        })
        .collect()
}

#[test]
fn gemm_matches_the_naive_product() {
    for (ar, ac, bc) in [(3, 5, 7), (8, 8, 8), (1, 6, 4), (6, 1, 3), (13, 9, 17)] {
        let a = fill(ar * ac, 0.5);
        // The adjoint form is `bc x ac`; the plain form is `ac x bc`.
        let b_plain = fill(ac * bc, 1.5);
        let b_adj = fill(bc * ac, 2.5);
        let mut out = vec![ZERO; ar * bc];
        gemm(ac, &a, &b_plain, ac, false, &mut out);
        let want = gemm_oracle(ar, ac, &a, &b_plain, ac, false, bc);
        assert_eq!(out, want, "plain {ar}x{ac} · {ac}x{bc}");
        gemm(ac, &a, &b_adj, bc, true, &mut out);
        let want = gemm_oracle(ar, ac, &a, &b_adj, bc, true, bc);
        assert_eq!(out, want, "adjoint {ar}x{ac} · {bc}x{ac}ᴴ");
    }
}

#[test]
fn orth_is_the_gram_residual() {
    let (rows, cols) = (6, 4);
    // A thin, nearly orthonormal factor: the residual must be small but not zero.
    let mut q = vec![ZERO; rows * cols];
    for j in 0..cols {
        for i in 0..rows {
            q[i + j * rows] = Complex64::new((i as f64 - j as f64).cos(), (i * j) as f64 * 0.1);
        }
    }
    let got = orth(&q, rows, cols);
    let mut want: f64 = 0.0;
    for j in 0..cols {
        for k in 0..cols {
            let mut x = Complex64::new(0., 0.);
            for i in 0..rows {
                x += q[i + j * rows].conj() * q[i + k * rows];
            }
            want = want.max(
                (x - if j == k {
                    Complex64::new(1., 0.)
                } else {
                    Complex64::new(0., 0.)
                })
                .norm(),
            );
        }
    }
    assert_eq!(got, want);
}

#[test]
fn negative_controls_reject_and_exact_pass() {
    let c = Case {
        m: 2,
        n: 2,
        batch: 1,
    };
    let a = vec![
        Complex64::new(2., 0.),
        Complex64::new(0., 0.),
        Complex64::new(0., 0.),
        Complex64::new(3., 0.),
    ];
    let l = vec![
        Complex64::new(1., 0.),
        Complex64::new(0., 0.),
        Complex64::new(0., 0.),
        Complex64::new(1., 0.),
    ];
    let u = a.clone();
    let p = vec![
        Complex64::new(1., 0.),
        Complex64::new(0., 0.),
        Complex64::new(0., 0.),
        Complex64::new(1., 0.),
    ];
    let parity = vec![Complex64::new(1., 0.)];
    assert!(check_explicit(&a, &p, &l, &u, &parity, c, 1e-10).is_ok());
    let mut bad = u.clone();
    bad[0] *= 1.001;
    assert!(check_explicit(&a, &p, &l, &bad, &parity, c, 1e-10).is_err());
    let mut q = p.clone();
    q[0] = Complex64::new(0., 0.);
    assert!(check_explicit(&a, &q, &l, &u, &parity, c, 1e-10).is_err());
}

#[test]
fn solve_and_eigen_negative_controls_reject() {
    let c = Case {
        m: 2,
        n: 2,
        batch: 1,
    };
    let a = [
        Complex64::new(2., 0.),
        Complex64::new(0., 0.),
        Complex64::new(0., 0.),
        Complex64::new(3., 0.),
    ];
    let b = [
        Complex64::new(2., 0.),
        Complex64::new(3., 0.),
        Complex64::new(2., 0.),
        Complex64::new(3., 0.),
        Complex64::new(2., 0.),
        Complex64::new(3., 0.),
        Complex64::new(2., 0.),
        Complex64::new(3., 0.),
    ];
    let x = [Complex64::new(1., 0.); 8];
    assert!(check_solve(&a, &b, &x, c, 1e-10, false).is_ok());
    let mut bad = x;
    bad[0] *= 1.001;
    assert!(check_solve(&a, &b, &bad, c, 1e-10, false).is_err());
    let w = [Complex64::new(2., 0.), Complex64::new(3., 0.)];
    let v = [
        Complex64::new(1., 0.),
        Complex64::new(0., 0.),
        Complex64::new(0., 0.),
        Complex64::new(1., 0.),
    ];
    assert!(check_eigen(&a, &w, &v, c, 1e-10).is_ok());
    let mut zero = v;
    zero[0] = Complex64::new(0., 0.);
    assert!(check_eigen(&a, &w, &zero, c, 1e-10).is_err());
}
