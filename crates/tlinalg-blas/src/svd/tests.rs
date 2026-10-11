//! Unit tests for the complex real workspace the SVD drivers take.

use super::complex_rwork_len;
use crate::Op;

/// A values-only complex `?gesdd` must take `7 * min(m, n)` reals.
///
/// `SRC/zgesdd.f` documents `5*mn` for `JOBZ = 'N'` and adds "(LAPACK <= 3.6 needs 7*mn)".
/// Accelerate ships that earlier interface, and its `?gesdd` wrote past a `5*mn` buffer on shapes
/// from just wider than square up to roughly `1.6*mn` (`DBDSQR`), corrupting the heap of a
/// downstream crate. Wide complex values-only shapes are the regression.
#[cfg(not(feature = "provider-inject"))]
#[test]
fn values_only_complex_takes_the_pre_3_7_length() {
    for (m, n) in [
        (1usize, 1usize),
        (3, 4),
        (4, 5),
        (5, 6),
        (7, 10),
        (8, 9),
        (12, 8),
        (24, 24),
    ] {
        let need = 7 * m.min(n);
        let len = complex_rwork_len(Op::Svd, b'N', m, n).expect("sizing cannot overflow");
        assert!(
            len >= need,
            "{m}x{n}: the real workspace is {len}, below the {need} the driver may write"
        );
    }
}

/// The injected symbol set drives `?gesvd`, whose real workspace is a fixed `5 * max(k, 1)` for
/// every `jobu`/`jobvt` combination: it does not grow with the shape, and asking for the factors
/// does not enlarge it.
#[cfg(feature = "provider-inject")]
#[test]
fn injected_gesvd_takes_a_fixed_real_workspace() {
    for (m, n) in [(0usize, 5usize), (1, 1), (3, 4), (7, 10), (24, 24)] {
        let len = complex_rwork_len(Op::Svd, b'S', m, n).expect("sizing cannot overflow");
        assert_eq!(len, 5 * m.min(n).max(1), "{m}x{n}");
    }
}
