//! faer-backed compact Householder QR primitives.
//!
//! Moved from `tenferro-linalg`'s faer backend (same project, MIT OR Apache-2.0). These are the two
//! batched
//! numerical kernels of the compact (incremental) Householder QR state; the state operations built
//! on them — factor, append, from-factors, `R` and `Q` extraction with the positive-diagonal gauge —
//! stay in the host.
//!
//! # Compact state
//!
//! A compact factor stores, column-major `rows x cols`, `R` on and above the diagonal and the
//! Householder vectors below it (each with an implicit unit head). The coefficients are
//! `coeff[j] = 1 / τ_j` for faer's reflector `H_j = I - v_j v_jᴴ / τ_j`, a real value carried in the
//! scalar type. Faer's denominator form is rebuilt transiently where a reflector is applied.

//!
//! # Batches
//!
//! Compact state is the host's own buffer, so both kernels take compact, batch-contiguous slices:
//! item `i` of a `rows x cols` batch occupies `[i * rows * cols, (i + 1) * rows * cols)`, and the
//! coefficients `k` per item likewise.

use faer::dyn_stack::{MemBuffer, MemStack};
use faer::prelude::{IntoConst, ReborrowMut};
use faer::{Conj, Mat, MatMut, MatRef};

use crate::batch::{self, in_place, out, Push};
use crate::util::{checked_product, invalid};
use crate::{Error, FaerScalar, Op, Parallel, Result};

/// The faer entity one, the implicit head of every stored reflector.
fn head_one<T: FaerScalar>() -> T::Entity {
    T::entity_from_real(T::parity(false).real_part())
}

/// One lane's storage for [`compact_factor`]: the reflector factor and the faer scratch, sized
/// once for the widest reflector the lane's matrices have and reused by every one.
struct CompactFactorScratch<E: faer::traits::ComplexField> {
    factor: Mat<E>,
    mem: MemBuffer,
}

impl<E: faer::traits::ComplexField> CompactFactorScratch<E> {
    /// Size the storage for every reflector of a `rows x cols` matrix.
    ///
    /// Reflector `j` applies a `(rows - j) x 1` block to `cols - j - 1` columns, and faer's
    /// workspace request is `1 * (cols - j - 1)`, so the first reflector's shapes bound every later
    /// one.
    fn new(rows: usize, cols: usize, _par: faer::Par) -> Self {
        let (basis_rows, targets) = if rows.min(cols) == 0 {
            (0, 0)
        } else {
            (rows, cols - 1)
        };
        Self {
            factor: Mat::zeros(1, 1),
            mem: MemBuffer::new(
                faer::linalg::householder::apply_block_householder_on_the_left_in_place_scratch::<E>(
                    basis_rows, 1, targets,
                ),
            ),
        }
    }
}

/// Factor one compact `rows x cols` matrix in place, pushing its coefficients.
fn compact_factor_item<T: FaerScalar>(
    data: &mut [T],
    rows: usize,
    cols: usize,
    coeff: &mut impl Push<T>,
    scratch: &mut CompactFactorScratch<T::Entity>,
    par: faer::Par,
) {
    let CompactFactorScratch { factor, mem } = scratch;
    let k = rows.min(cols);
    let mut qr = MatMut::from_column_major_slice_mut(T::entity_slice_mut(data), rows, cols);
    for j in 0..k {
        let mut column = qr.rb_mut().col_mut(j).subrows_mut(j, rows - j);
        let (mut head, tail) = column.rb_mut().split_at_row_mut(1);
        let info = faer::linalg::householder::make_householder_in_place(&mut head[0], tail);
        let beta = head[0];
        head[0] = head_one::<T>();
        // Faer stores H = I - vv^H/tau; compact state stores 1/tau.
        coeff.push(T::from_real(T::recip_real(info.tau)));
        if j + 1 < cols {
            let basis_rows = rows - j;
            // Borrow the reflector column as the basis against the disjoint target columns: split
            // the state at column `j + 1`, keep the head+column part immutable and apply to the
            // rest. No copy and no unsafe overlap.
            let (column_part, target) = qr.rb_mut().split_at_col_mut(j + 1);
            let basis = column_part
                .into_const()
                .subcols(j, 1)
                .subrows(j, basis_rows);
            factor[(0, 0)] = T::entity_from_real(info.tau);
            faer::linalg::householder::apply_block_householder_on_the_left_in_place_with_conj(
                basis,
                factor.as_ref(),
                Conj::No,
                target.subrows_mut(j, basis_rows),
                par,
                MemStack::new(mem),
            );
        }
        qr[(j, j)] = beta;
    }
}

/// Factor every compact column-major `rows x cols` matrix of a batch in place into the compact
/// Householder state.
///
/// `data` holds `batch` matrices; on return each holds `R` on and above the diagonal and the
/// reflector vectors below it, and `coeff` (cleared, then filled) holds the `min(rows, cols)`
/// coefficients per item.
///
/// # Errors
///
/// [`Error::Inconsistent`] when `data` does not hold `batch * rows * cols` elements, and
/// [`Error::InvalidArgument`] when a size overflows. `coeff` is empty on error.
///
/// # Examples
///
/// ```
/// use tlinalg::{householder::compact_factor, Op, Parallel};
///
/// let mut a = [3.0_f64, 4.0];
/// let mut coeff = Vec::new();
/// compact_factor(
///     Op::HouseholderQr, 2, 1, 1, &mut a, &mut coeff, Parallel::Sequential,
/// ).unwrap();
/// assert!((a[0].abs() - 5.0).abs() < 1e-12);
/// assert_eq!(coeff.len(), 1);
/// ```
// INVARIANT: shape, batch, state buffer, coefficients, token are distinct operands.
#[allow(clippy::too_many_arguments)]
pub fn compact_factor<T: FaerScalar>(
    op: Op,
    rows: usize,
    cols: usize,
    batch: usize,
    data: &mut [T],
    coeff: &mut Vec<T>,
    par: Parallel<'_>,
) -> Result<()> {
    coeff.clear();
    let matrix_len = checked_product(op, "matrix", &[rows, cols])?;
    batch::run(
        op,
        batch,
        par,
        Some(rows.max(cols)),
        &mut (
            in_place(data, matrix_len, "matrix"),
            out(coeff, rows.min(cols)),
        ),
        |par| CompactFactorScratch::<T::Entity>::new(rows, cols, par),
        |index, (data, coeff), scratch, par| {
            compact_factor_item::<T>(data.item(index), rows, cols, coeff, scratch, par);
            Ok(())
        },
    )
}

/// The dimensions of a batched reflector application.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ReflectorShape {
    /// Rows of the state and of the target.
    pub rows: usize,
    /// Columns of the compact state.
    pub a_cols: usize,
    /// Columns of the target.
    pub cols: usize,
    /// Number of reflectors applied (at most `min(rows, a_cols)`).
    pub k: usize,
}

/// Apply the first `k` reflectors of each compact state to the matching target in place:
/// `C ← Q C`, or `C ← Qᴴ C` when `transpose`.
///
/// Per item, `a` holds the compact `rows x a_cols` state, `coeff` its `k` coefficients, and `c` the
/// compact `rows x cols` target. All three are batch-contiguous with `batch` items.
///
/// # Errors
///
/// [`Error::InvalidArgument`] when `k` exceeds `rows` or `a_cols`, and [`Error::Inconsistent`] when
/// a buffer length does not match `batch` items of its shape.
///
/// # Examples
///
/// ```
/// use tlinalg::householder::{apply_reflectors, compact_factor, ReflectorShape};
/// use tlinalg::{Op, Parallel};
///
/// let mut a = [3.0_f64, 4.0];
/// let mut coeff = Vec::new();
/// compact_factor(Op::HouseholderQr, 2, 1, 1, &mut a, &mut coeff, Parallel::Sequential).unwrap();
/// // Q applied to e1 is the first column of Q, which is ±(3, 4)/5.
/// let mut c = [1.0_f64, 0.0];
/// let shape = ReflectorShape { rows: 2, a_cols: 1, cols: 1, k: 1 };
/// apply_reflectors(
///     Op::HouseholderQrQColumns, shape, 1, &a, &coeff, &mut c, false, Parallel::Sequential,
/// ).unwrap();
/// assert!((c[0].abs() - 0.6).abs() < 1e-12);
/// ```
// INVARIANT: these buffers and dimensions mirror the host reflector ABI the call replaces.
#[allow(clippy::too_many_arguments)]
pub fn apply_reflectors<T: FaerScalar>(
    op: Op,
    shape: ReflectorShape,
    batch: usize,
    a: &[T],
    coeff: &[T],
    c: &mut [T],
    transpose: bool,
    par: Parallel<'_>,
) -> Result<()> {
    let ReflectorShape {
        rows,
        a_cols,
        cols,
        k,
    } = shape;
    if k > rows || k > a_cols {
        return Err(invalid(
            op,
            "configuration",
            "dimensions: reflector count exceeds matrix dimensions",
        ));
    }
    let a_len = checked_product(op, "A", &[rows, a_cols])?;
    let c_len = checked_product(op, "C", &[rows, cols])?;
    if a.len() != checked_product(op, "A", &[a_len, batch])?
        || coeff.len() != checked_product(op, "coefficients", &[k, batch])?
    {
        return Err(Error::Inconsistent {
            op,
            detail: "batch buffers describe different batches",
        });
    }
    let apply = rows != 0 && cols != 0 && k != 0;
    batch::run(
        op,
        batch,
        par,
        Some(rows.max(a_cols).max(cols)),
        &mut (in_place(c, c_len, "C"),),
        |_| {
            let req = if !apply {
                faer::dyn_stack::StackReq::EMPTY
            } else if transpose {
                faer::linalg::householder::apply_block_householder_sequence_transpose_on_the_left_in_place_scratch::<T::Entity>(rows, 1, cols)
            } else {
                faer::linalg::householder::apply_block_householder_sequence_on_the_left_in_place_scratch::<T::Entity>(rows, 1, cols)
            };
            (MemBuffer::new(req), Mat::<T::Entity>::zeros(1, k))
        },
        |index, (c,), (mem, factors), par| {
            if !apply {
                return Ok(());
            }
            let a = &a[index * a_len..(index + 1) * a_len];
            let coeff = &coeff[index * k..(index + 1) * k];
            let basis =
                MatRef::from_column_major_slice(T::entity_slice(a), rows, a_cols).subcols(0, k);
            // Rebuild faer's denominator form transiently from the state coefficients.
            for (col, value) in coeff.iter().enumerate() {
                factors[(0, col)] = T::entity_from_real(T::recip_real(value.real_part()));
            }
            let matrix =
                MatMut::from_column_major_slice_mut(T::entity_slice_mut(c.item(index)), rows, cols);
            let stack = MemStack::new(mem);
            if transpose {
                // `Qᴴ`: the conjugate of the transposed sequence. The real scalars are their own
                // conjugate, so this is the plain transpose for them, as before.
                faer::linalg::householder::apply_block_householder_sequence_transpose_on_the_left_in_place_with_conj(
                    basis, factors.as_ref(), Conj::Yes, matrix, par, stack,
                );
            } else {
                faer::linalg::householder::apply_block_householder_sequence_on_the_left_in_place_with_conj(
                    basis, factors.as_ref(), Conj::No, matrix, par, stack,
                );
            }
            Ok(())
        },
    )
}
