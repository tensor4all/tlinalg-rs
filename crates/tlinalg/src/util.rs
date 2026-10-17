//! Shared argument checks and descriptor-to-faer conversions for the per-matrix kernels.

use faer::MatRef;

use crate::batch::Push;

use crate::scalar::ScalarEntity;
use crate::{Error, Op, Result};

/// A caller-supplied argument was invalid.
pub(crate) fn invalid(op: Op, role: &'static str, detail: impl Into<String>) -> Error {
    Error::InvalidArgument {
        op,
        role,
        detail: detail.into(),
    }
}

/// A shape product overflowed `usize`.
pub(crate) fn checked_product(op: Op, role: &'static str, shape: &[usize]) -> Result<usize> {
    shape
        .iter()
        .try_fold(1usize, |acc, &dim| acc.checked_mul(dim))
        .ok_or_else(|| {
            invalid(
                op,
                "configuration",
                format!("{role} element count overflows usize"),
            )
        })
}

/// The work estimate a family reports to the batch policy, scaled for the scalar: four real
/// multiply-adds per complex one (see [`ScalarEntity::COMPLEX_WEIGHT`]).
pub(crate) fn weighted_muladds<T: ScalarEntity>(real_muladds: f64) -> f64 {
    real_muladds * T::COMPLEX_WEIGHT
}

/// Push the column-major `rows x cols` leading block of `mat`, keeping only the entries the
/// predicate selects and writing zero elsewhere.
///
/// One pass, by `push`: the pre-extraction code zero-initialized and then scattered, so this writes
/// each element once instead of twice and allocates the same single buffer.
pub(crate) fn push_masked<T: ScalarEntity>(
    out: &mut impl Push<T>,
    mat: MatRef<'_, T::Entity>,
    rows: usize,
    cols: usize,
    keep: impl Fn(usize, usize) -> bool,
) {
    for col in 0..cols {
        for row in 0..rows {
            out.push(if keep(row, col) {
                T::from_entity(mat[(row, col)])
            } else {
                T::default()
            });
        }
    }
}

/// Push the `n x n` permutation matrix with a one at `(row, perm[row])`, column-major.
///
/// `perm_inv` is the inverse permutation, so column `col` has its one in row `perm_inv[col]`.
pub(crate) fn push_permutation<T: ScalarEntity>(out: &mut impl Push<T>, perm_inv: &[usize]) {
    let n = perm_inv.len();
    let one = T::parity(false);
    for &one_row in perm_inv {
        for row in 0..n {
            out.push(if row == one_row { one } else { T::default() });
        }
    }
}

/// Push the column-major `n x n` identity.
pub(crate) fn push_identity<T: ScalarEntity>(out: &mut impl Push<T>, n: usize) {
    let one = T::parity(false);
    for col in 0..n {
        for row in 0..n {
            out.push(if row == col { one } else { T::default() });
        }
    }
}
