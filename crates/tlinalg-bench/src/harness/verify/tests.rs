use super::*;
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
