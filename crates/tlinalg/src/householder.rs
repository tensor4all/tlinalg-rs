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
use faer::prelude::{IntoConst, Reborrow, ReborrowMut};
use faer::{Conj, Mat, MatMut, MatRef};

use crate::batch::{self, in_place, out, Push};
use crate::util::{checked_product, invalid, weighted_muladds};
use crate::{Error, FaerScalar, Op, Result};
use tcontract_exec::Exec;

/// The faer entity one, the implicit head of every stored reflector.
fn head_one<T: FaerScalar>() -> T::Entity {
    T::entity_from_real(T::parity(false).real_part())
}

/// At or below this size a matrix is not blocked at all: faer's own `QrParams::blocking_threshold`,
/// the same shape rule faer applies before it sub-blocks a panel. Below it the compact-WY factor
/// costs more than the level-3 trailing update it buys, and the campaign's `medium` 32x32 cells
/// (where faer recommends a block of 4) read 1.4x slower at 1T and lose the OpenBLAS comparison they
/// used to win.
const COMPACT_BLOCKING_THRESHOLD: usize = 48 * 48;

/// faer's own recommendation, capped for this kernel: the recommendation assumes faer's recursively
/// blocked panel, and the panel here is factored one reflector at a time, so its width is what a
/// pooled row scales against. Measured at `householder_factor` 4096x256 b1 c64 (512x512 b1),
/// 1T / 8T pool: faer's own 48 gives 57.9 / 31.4 (17.3 / 5.95), 16 gives 53.6 / 19.6 (16.5 /
/// 4.41), and recursing the panel was slower still (the factor each level builds is another
/// `O(rows * block^2)`). `ponytail:` the cap is the price of the unblocked panel.
const COMPACT_BLOCK_SIZE_CAP: usize = 16;

/// The compact-WY block size for a `rows x cols` reflector panel.
///
/// It is `1` at the shapes where blocking does not pay, where the loop below degrades to the
/// unblocked one.
fn compact_block_size<E: faer::traits::ComplexField>(rows: usize, cols: usize) -> usize {
    if rows.saturating_mul(cols) <= COMPACT_BLOCKING_THRESHOLD {
        return 1;
    }
    faer::linalg::qr::no_pivoting::factor::recommended_block_size::<E>(rows, cols)
        .clamp(1, COMPACT_BLOCK_SIZE_CAP)
}

/// One lane's storage for [`compact_factor`]: the block compact-WY factor, the reflector panel's
/// own scratch, and the faer scratch for the trailing update, sized once for the widest shape the
/// lane's matrices have and reused by every one.
struct CompactFactorScratch<E: faer::traits::ComplexField> {
    factor: Mat<E>,
    mem: MemBuffer,
}

impl<E: faer::traits::ComplexField> CompactFactorScratch<E> {
    /// Size the storage for a `rows x cols` matrix blocked by [`compact_block_size`].
    ///
    /// A `bs`-wide panel applies a `(rows - j) x bs` block to `cols - j - bs` columns, and faer's
    /// workspace request is a `bs x (cols - j - bs)` temporary, so `bs x cols` bounds every block.
    fn new(rows: usize, cols: usize, _par: faer::Par) -> Self {
        let block = compact_block_size::<E>(rows, cols);
        let (basis_rows, targets) = if rows.min(cols) == 0 {
            (0, 0)
        } else {
            (rows, cols)
        };
        Self {
            factor: Mat::zeros(block, block),
            mem: MemBuffer::new(
                faer::linalg::householder::apply_block_householder_transpose_on_the_left_in_place_scratch::<E>(
                    basis_rows, block, targets,
                ),
            ),
        }
    }
}

/// Factor a `bs`-wide panel of `qr` starting at column `start`, storing the reflectors in place.
///
/// Reflector `j` is applied to the rest of the panel only: the trailing columns take the whole
/// block at once, through the compact-WY factor [`upgrade_householder_factor`] builds from the
/// coefficients the panel left in `factor`'s diagonal.
fn compact_factor_panel<T: FaerScalar>(
    qr: &mut MatMut<'_, T::Entity>,
    factor: &mut Mat<T::Entity>,
    mem: &mut MemBuffer,
    start: usize,
    bs: usize,
    coeff: &mut impl Push<T>,
    par: faer::Par,
) {
    let rows = qr.nrows();
    for j in start..start + bs {
        let mut column = qr.rb_mut().col_mut(j).subrows_mut(j, rows - j);
        let (mut head, tail) = column.rb_mut().split_at_row_mut(1);
        let info = faer::linalg::householder::make_householder_in_place(&mut head[0], tail);
        let beta = head[0];
        head[0] = head_one::<T>();
        let inside = j - start;
        factor[(inside, inside)] = T::entity_from_real(info.tau);
        // Faer stores H = I - vv^H/tau; compact state stores 1/tau.
        coeff.push(T::from_real(T::recip_real(info.tau)));
        if j + 1 < start + bs {
            let basis_rows = rows - j;
            // Borrow the reflector column as the basis against the disjoint target columns: split
            // the state at column `j + 1`, keep the head+column part immutable and apply to the
            // rest of the panel. No copy and no unsafe overlap.
            let (column_part, target) = qr.rb_mut().split_at_col_mut(j + 1);
            let basis = column_part
                .into_const()
                .subcols(j, 1)
                .subrows(j, basis_rows);
            let target = target
                .subrows_mut(j, basis_rows)
                .subcols_mut(0, start + bs - j - 1);
            let tile = factor.rb().subrows(inside, 1).subcols(inside, 1);
            faer::linalg::householder::apply_block_householder_on_the_left_in_place_with_conj(
                basis,
                tile,
                Conj::No,
                target,
                par,
                MemStack::new(mem),
            );
        }
        qr[(j, j)] = beta;
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
    let block = factor.nrows();
    let k = rows.min(cols);
    let mut qr = MatMut::from_column_major_slice_mut(T::entity_slice_mut(data), rows, cols);
    let mut start = 0;
    while start < k {
        let bs = block.min(k - start);
        compact_factor_panel::<T>(&mut qr, factor, mem, start, bs, coeff, par);
        if start + bs < cols {
            // Apply the panel's whole reflector sequence to the trailing columns at once, as the
            // compact-WY factor of the block. QR applies `H_0` first and `H_(bs-1)` last, so the
            // trailing columns take `H_(bs-1) ... H_0` -- the adjoint of `I - V T^-1 V^H`, which
            // is the transpose form faer's own blocked QR uses.
            let (panel, mut trailing) = qr.rb_mut().split_at_col_mut(start + bs);
            let basis = panel
                .into_const()
                .subrows(start, rows - start)
                .subcols(start, bs);
            faer::linalg::householder::upgrade_householder_factor(
                factor.rb_mut().subrows_mut(0, bs).subcols_mut(0, bs),
                basis,
                bs,
                1,
                par,
            );
            let factor_tile = factor.rb().subrows(0, bs).subcols(0, bs);
            let target = trailing.rb_mut().subrows_mut(start, rows - start);
            faer::linalg::householder::apply_block_householder_transpose_on_the_left_in_place_with_conj(
                basis,
                factor_tile,
                Conj::Yes,
                target,
                par,
                MemStack::new(mem),
            );
        }
        start += bs;
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
/// use tcontract_exec::Exec;
/// use tlinalg::{householder::compact_factor, Op};
///
/// let mut a = [3.0_f64, 4.0];
/// let mut coeff = Vec::new();
/// compact_factor(
///     Op::HouseholderQr, 2, 1, 1, &mut a, &mut coeff, Exec::serial(),
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
    exec: Exec<'_>,
) -> Result<()> {
    coeff.clear();
    let matrix_len = checked_product(op, "matrix", &[rows, cols])?;
    let item_muladds = weighted_muladds::<T>((rows * cols * rows.min(cols)) as f64);
    batch::run(
        op,
        batch,
        exec,
        Some(rows.max(cols)),
        item_muladds,
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
/// use tcontract_exec::Exec;
/// use tlinalg::householder::{apply_reflectors, compact_factor, ReflectorShape};
/// use tlinalg::Op;
///
/// let mut a = [3.0_f64, 4.0];
/// let mut coeff = Vec::new();
/// compact_factor(Op::HouseholderQr, 2, 1, 1, &mut a, &mut coeff, Exec::serial()).unwrap();
/// // Q applied to e1 is the first column of Q, which is ±(3, 4)/5.
/// let mut c = [1.0_f64, 0.0];
/// let shape = ReflectorShape { rows: 2, a_cols: 1, cols: 1, k: 1 };
/// apply_reflectors(
///     Op::HouseholderQrQColumns, shape, 1, &a, &coeff, &mut c, false, Exec::serial(),
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
    exec: Exec<'_>,
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
    let block = compact_block_size::<T::Entity>(rows, a_cols).min(k).max(1);
    let item_muladds = weighted_muladds::<T>((rows * cols * k) as f64);
    batch::run(
        op,
        batch,
        exec,
        Some(batch::auto_extent(rows.max(a_cols), cols)),
        item_muladds,
        &mut (in_place(c, c_len, "C"),),
        |_| {
            let req = if !apply {
                faer::dyn_stack::StackReq::EMPTY
            } else if transpose {
                faer::linalg::householder::apply_block_householder_sequence_transpose_on_the_left_in_place_scratch::<T::Entity>(rows, block, cols)
            } else {
                faer::linalg::householder::apply_block_householder_sequence_on_the_left_in_place_scratch::<T::Entity>(rows, block, cols)
            };
            (MemBuffer::new(req), Mat::<T::Entity>::zeros(block, k))
        },
        |index, (c,), (mem, factors), par| {
            if !apply {
                return Ok(());
            }
            let a = &a[index * a_len..(index + 1) * a_len];
            let coeff = &coeff[index * k..(index + 1) * k];
            let basis =
                MatRef::from_column_major_slice(T::entity_slice(a), rows, a_cols).subcols(0, k);
            // Rebuild faer's denominator form transiently from the state coefficients, and upgrade
            // each `block`-wide run of them into the compact-WY factor the sequence apply expects:
            // the per-reflector taus become the block's `T`, so the apply is one blocked update
            // instead of one rank-one update per reflector.
            let mut start = 0;
            while start < k {
                let bs = block.min(k - start);
                let mut tile = factors.rb_mut().subrows_mut(0, bs).subcols_mut(start, bs);
                for inside in 0..bs {
                    let tau = T::recip_real(coeff[start + inside].real_part());
                    tile[(inside, inside)] = T::entity_from_real(tau);
                }
                faer::linalg::householder::upgrade_householder_factor(
                    tile,
                    basis.subrows(start, rows - start).subcols(start, bs),
                    bs,
                    1,
                    par,
                );
                start += bs;
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
