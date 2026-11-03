use super::{push_real_pairs, push_real_values};
use num_complex::Complex64;

/// A real matrix whose eigenvalues have a tiny imaginary part must report what LAPACK returned:
/// the pair folding `push_real_pairs` needs for its vectors would report these as real.
#[test]
fn values_only_keeps_a_tiny_imaginary_part() {
    let (w, wi) = ([1.0_f64, 1.0], [1.0e-18_f64, -1.0e-18]);
    let mut raw = Vec::new();
    push_real_values::<f64>(&w, &wi, &mut raw);
    assert_eq!(
        raw,
        vec![Complex64::new(1.0, 1.0e-18), Complex64::new(1.0, -1.0e-18)]
    );

    let mut folded = Vec::new();
    push_real_pairs::<f64>(2, &w, &wi, None, &mut folded, None);
    assert_eq!(folded, vec![Complex64::new(1.0, 0.0); 2]);
}
