//! faer-backed batched thin QR and column-pivoted (rank-revealing) QR.
//!
//! Moved from `tenferro-linalg`'s faer backend (same project, MIT OR Apache-2.0).
//!
//! Both return the **raw** faer factors: no sign/phase gauge is applied, matching the
//! pre-extraction `qr` and `rank_revealing_qr` paths (the positive-diagonal gauge of the compact
//! Householder family is applied by the host when it extracts `Q` and `R`). For the rank-revealing
//! variant the rank decision (tolerances) and the non-finite screen stay in the host; [`magnitude`]
//! is the diagonal measure it was computed with. An all-zero item is not factored: it gets the
//! leading identity columns as `Q`, a zero `R` and the identity permutation, exactly as
//! `tlinalg-blas` and the pre-extraction host did.
//!
//! # Borrowed workspace
//!
//! [`qr_workspace`] reports the bytes a call needs and [`qr_into`] runs it from caller-owned
//! storage; [`qr`] is the `Vec`-in/`Vec`-out wrapper (one scratch allocation per call, none once
//! the caller reserves its outputs).

use core::mem::MaybeUninit;

use faer::dyn_stack::{MemStack, StackReq};
use faer::{Conj, MatMut, MatRef};
use strided_view::RawStridedRef;

use crate::batch::{self, fixed_out, BatchedRef, Push, Sink, VecOut};
use crate::scalar::ScalarEntity;
use crate::util::{checked_product, invalid, push_masked, weighted_muladds};
use crate::workspace::{bytes, faer_req, lane_requirement, total_of, zeros};
use crate::{FaerScalar, Op, Result};
use tcontract_exec::Exec;

/// Lane scratch for `m x n` QR factorizations, plain or column-pivoted.
struct QrScratch<'w, E: faer::traits::ComplexField> {
    work: MatMut<'w, E>,
    coeff: MatMut<'w, E>,
    permutation: &'w mut [usize],
    inverse_permutation: &'w mut [usize],
    factor_mem: &'w mut [MaybeUninit<u8>],
    apply_mem: &'w mut [MaybeUninit<u8>],
}

/// The block size faer recommends for an `m x n` QR factorization.
fn block_size<E: faer::traits::ComplexField>(m: usize, n: usize) -> usize {
    faer::linalg::qr::no_pivoting::factor::recommended_block_size::<E>(m, n)
}

/// The apply scratch of the thin-`Q` formation (faer's sequence apply on the left).
fn apply_req<E: faer::traits::ComplexField>(m: usize, n: usize) -> StackReq {
    let k = m.min(n);
    let block = block_size::<E>(m, n);
    faer::linalg::householder::apply_block_householder_sequence_on_the_left_in_place_scratch::<E>(
        m, block, k,
    )
}

/// One lane's complete requirement: the work matrix, the block coefficients, the permutations and
/// faer's factor/apply scratch.
///
/// Call [`check_shape`] first: it checks every product this function multiplies, including the one
/// inside faer's block-size recommendation.
fn qr_req<E: faer::traits::ComplexField>(
    m: usize,
    n: usize,
    pivoting: bool,
    par: faer::Par,
) -> StackReq {
    let k = m.min(n);
    let block = block_size::<E>(m, n);
    // On 64-bit targets the permutation is written straight into the `i64` output, so only the
    // inverse needs lane storage.
    let permutation = if pivoting && !PERMUTATION_IN_OUTPUT {
        n
    } else {
        0
    };
    let inverse_permutation = if pivoting { n } else { 0 };
    StackReq::new::<E>(m * n)
        .and(StackReq::new::<E>(block * k))
        .and(StackReq::new::<usize>(permutation))
        .and(StackReq::new::<usize>(inverse_permutation))
        .and(faer_req(qr_req_factor::<E>(m, n, pivoting, par)))
        .and(faer_req(apply_req::<E>(m, n)))
}

/// Reject shapes whose products the lane requirement multiplies, before any faer helper sees them.
fn check_shape<E: faer::traits::ComplexField>(op: Op, m: usize, n: usize) -> Result<()> {
    checked_product(op, "matrix", &[m, n])?;
    // faer's recommendation multiplies the dimensions itself; the product above covers it.
    let block = block_size::<E>(m, n);
    checked_product(op, "workspace", &[block, m.min(n)])?;
    Ok(())
}

/// The output lengths and the work estimate of an `m x n` batch, with every product the lane
/// requirement multiplies checked (this is the validation the `Vec` wrappers run before they size a
/// workspace, and `_into` runs before it builds one).
fn qr_shape<T: FaerScalar>(op: Op, m: usize, n: usize) -> Result<(usize, usize, f64)> {
    check_shape::<T::Entity>(op, m, n)?;
    let k = m.min(n);
    let q_len = checked_product(op, "Q", &[m, k])?;
    let r_len = checked_product(op, "R", &[k, n])?;
    Ok((q_len, r_len, qr_work::<T>(m, n)))
}

/// The work estimate a QR factorization reports to the batch policy.
fn qr_work<T: ScalarEntity>(m: usize, n: usize) -> f64 {
    let k = m.min(n);
    weighted_muladds::<T>(m as f64 * n as f64 * k as f64 - (k as f64).powi(3) / 3.0)
}

impl<'w, E: faer::traits::ComplexField> QrScratch<'w, E> {
    /// Carve the lane's buffers from its borrowed workspace chunk.
    fn new(
        m: usize,
        n: usize,
        pivoting: bool,
        par: faer::Par,
        stack: &'w mut MemStack,
    ) -> QrScratch<'w, E> {
        let k = m.min(n);
        let block = block_size::<E>(m, n);
        let zero = <E as faer::traits::ComplexField>::zero_impl();
        let (work, stack) = zeros::<E>(stack, m * n, &zero);
        let (coeff, stack) = zeros::<E>(stack, block * k, &zero);
        let permutation = if pivoting && !PERMUTATION_IN_OUTPUT {
            n
        } else {
            0
        };
        let (permutation, stack) = zeros::<usize>(stack, permutation, &0);
        let (inverse_permutation, stack) = zeros::<usize>(stack, if pivoting { n } else { 0 }, &0);
        let factor_req = qr_req_factor::<E>(m, n, pivoting, par);
        let (factor_mem, stack) = bytes(stack, factor_req.unaligned_bytes_required());
        let (apply_mem, _) = bytes(stack, apply_req::<E>(m, n).unaligned_bytes_required());
        QrScratch {
            work: MatMut::from_column_major_slice_mut(work, m, n),
            coeff: MatMut::from_column_major_slice_mut(coeff, block, k),
            permutation,
            inverse_permutation,
            factor_mem,
            apply_mem,
        }
    }

    /// Write the thin `Q` from the reflectors in `work`/`coeff` into the caller's output.
    fn form_q<T: FaerScalar<Entity = E>>(&mut self, q: &mut Sink<'_, T>, par: faer::Par) {
        let (m, k) = (self.work.nrows(), self.work.nrows().min(self.work.ncols()));
        // `Q = H_1 ... H_k I`: the output region starts as the thin identity.
        let region = T::entity_slice_mut(q.fill(m * k, |index| {
            if index % m == index / m {
                T::parity(false)
            } else {
                T::default()
            }
        }));
        faer::linalg::householder::apply_block_householder_sequence_on_the_left_in_place_with_conj(
            self.work.as_ref().subcols(0, k),
            self.coeff.as_ref(),
            Conj::No,
            MatMut::from_column_major_slice_mut(region, m, k),
            par,
            MemStack::new(&mut *self.apply_mem),
        );
    }
}

/// faer's own factor scratch for the chosen variant (shared by the query and the carve).
fn qr_req_factor<E: faer::traits::ComplexField>(
    m: usize,
    n: usize,
    pivoting: bool,
    par: faer::Par,
) -> StackReq {
    let block = block_size::<E>(m, n);
    if pivoting {
        faer::linalg::qr::col_pivoting::factor::qr_in_place_scratch::<usize, E>(
            m,
            n,
            block,
            par,
            Default::default(),
        )
    } else {
        faer::linalg::qr::no_pivoting::factor::qr_in_place_scratch::<E>(
            m,
            n,
            block,
            par,
            Default::default(),
        )
    }
}

fn push_factors<T: FaerScalar>(scratch: &QrScratch<'_, T::Entity>, r: &mut impl Push<T>) {
    let k = scratch.work.nrows().min(scratch.work.ncols());
    push_masked(
        r,
        scratch.work.as_ref(),
        k,
        scratch.work.ncols(),
        |row, col| row <= col,
    );
}

fn qr_item<T: FaerScalar>(
    mat: MatRef<'_, T::Entity>,
    q: &mut Sink<'_, T>,
    r: &mut impl Push<T>,
    scratch: &mut QrScratch<'_, T::Entity>,
    par: faer::Par,
) {
    scratch.work.copy_from(mat);
    scratch
        .coeff
        .fill(<T::Entity as faer::traits::ComplexField>::zero_impl());
    faer::linalg::qr::no_pivoting::factor::qr_in_place(
        scratch.work.as_mut(),
        scratch.coeff.as_mut(),
        par,
        MemStack::new(&mut *scratch.factor_mem),
        Default::default(),
    );
    scratch.form_q(q, par);
    push_factors::<T>(scratch, r);
}

/// Whether faer's `usize` permutation can be written straight into the `i64` output: true where
/// the two types share size and alignment (64-bit targets). Elsewhere a lane buffer is converted.
const PERMUTATION_IN_OUTPUT: bool = core::mem::size_of::<usize>() == core::mem::size_of::<i64>()
    && core::mem::align_of::<usize>() == core::mem::align_of::<i64>();

fn rank_revealing_qr_item<T: FaerScalar>(
    op: Op,
    mat: MatRef<'_, T::Entity>,
    (q, r, permutation): (&mut Sink<'_, T>, &mut impl Push<T>, &mut Sink<'_, i64>),
    scratch: &mut QrScratch<'_, T::Entity>,
    par: faer::Par,
) -> Result<()> {
    let (m, n) = (mat.nrows(), mat.ncols());
    let k = m.min(n);
    let all_zero = (0..n).all(|col| (0..m).all(|row| T::entity_magnitude(mat[(row, col)]) == 0.0));
    if all_zero {
        // Rank zero by the host's rule: the leading identity columns, a zero `R`, no pivoting.
        for col in 0..k {
            for row in 0..m {
                q.push(if row == col {
                    T::parity(false)
                } else {
                    T::default()
                });
            }
        }
        for _ in 0..k * n {
            r.push(T::default());
        }
        permutation.fill(n, |column| column as i64);
        return Ok(());
    }
    scratch.work.copy_from(mat);
    scratch
        .coeff
        .fill(<T::Entity as faer::traits::ComplexField>::zero_impl());
    let region = permutation.fill(n, |_| 0);
    let forward: &mut [usize] = if PERMUTATION_IN_OUTPUT {
        // SAFETY: `PERMUTATION_IN_OUTPUT` holds only when `usize` and `i64` have the same size and
        // alignment, and every bit pattern is valid for both, so the initialised `i64` region is a
        // valid `[usize]` of the same length for the borrow. faer writes column indices below
        // `n <= isize::MAX`, which read back as the same non-negative `i64` values.
        unsafe {
            core::slice::from_raw_parts_mut(region.as_mut_ptr().cast::<usize>(), region.len())
        }
    } else {
        &mut *scratch.permutation
    };
    // Following faer 0.24's public column-pivoted QR factor API; the provider's forward
    // permutation is the public gather convention.
    faer::linalg::qr::col_pivoting::factor::qr_in_place(
        scratch.work.as_mut(),
        scratch.coeff.as_mut(),
        forward,
        &mut *scratch.inverse_permutation,
        par,
        MemStack::new(&mut *scratch.factor_mem),
        Default::default(),
    );
    if !PERMUTATION_IN_OUTPUT {
        for (slot, &column) in region.iter_mut().zip(&*scratch.permutation) {
            *slot = i64::try_from(column).map_err(|_| {
                invalid(op, "configuration", "column permutation exceeds i64 range")
            })?;
        }
    }
    scratch.form_q(q, par);
    push_factors::<T>(scratch, r);
    Ok(())
}

/// Bytes one batched thin QR call needs in its borrowed workspace.
///
/// `m x n` matrices, `batch` of them: the resolved route's lane count times one lane's complete
/// inventory, worst-case alignment padding included. Pass the result to [`qr_into`]; the count does
/// not depend on the caller's buffer alignment, only on the shape and the `Exec` route.
///
/// # Errors
///
/// [`Error::InvalidArgument`](crate::Error::InvalidArgument) when a size overflows, and the
/// executor's refusal for a forced strategy this route cannot honor (the canonical `Op` is
/// `Op::Qr`).
pub fn qr_workspace<T: FaerScalar>(
    m: usize,
    n: usize,
    batch: usize,
    exec: Exec<'_>,
) -> Result<usize> {
    qr_bytes::<T>(Op::Qr, m, n, false, batch, exec)
}

/// [`qr_workspace`] with the caller's operation, so a wrapper reports its own `Op`.
///
/// The order matches the contract: resolve the route (refusing an unavailable strategy), return
/// zero for an empty batch, and only then check the shape products and build the requirement.
fn qr_bytes<T: FaerScalar>(
    op: Op,
    m: usize,
    n: usize,
    pivoting: bool,
    batch: usize,
    exec: Exec<'_>,
) -> Result<usize> {
    let item_muladds = qr_work::<T>(m, n);
    let requirement = lane_requirement(
        op,
        exec,
        batch,
        Some(m.max(n)),
        item_muladds,
        || check_shape::<T::Entity>(op, m, n),
        |par| qr_req::<T::Entity>(m, n, pivoting, par),
    )?;
    match requirement {
        None => Ok(0),
        Some((lanes, per_lane)) => total_of(op, lanes, per_lane),
    }
}

/// Thin QR of every `m x n` matrix of a batch, `A = Q R`, into caller-owned storage.
///
/// `q` and `r` hold exactly `m * min(m, n)` and `min(m, n) * n` **uninitialised** elements per batch
/// item, compact column-major in batch order; `workspace` comes from [`qr_workspace`] and is
/// borrowed for the call only. Every element of `q` and `r` is written. A workspace shorter than the
/// query's answer is refused with [`crate::Error::InsufficientWorkspace`] before anything is
/// written; after an error the output buffers' contents are unspecified (items already run stay
/// written).
///
/// # Errors
///
/// [`Error::InvalidArgument`](crate::Error::InvalidArgument) for a malformed descriptor, an output
/// length that does not match the shape, or a size overflow;
/// [`crate::Error::InsufficientWorkspace`] for a short workspace.
pub fn qr_into<T: FaerScalar>(
    op: Op,
    input: RawStridedRef<'_, T>,
    q: &mut [MaybeUninit<T>],
    r: &mut [MaybeUninit<T>],
    workspace: &mut [MaybeUninit<u8>],
    exec: Exec<'_>,
) -> Result<()> {
    let input = BatchedRef::new(op, "input", input)?;
    let (m, n) = (input.rows(), input.cols());
    let (q_len, r_len, item_muladds) = qr_shape::<T>(op, m, n)?;
    let requirement = lane_requirement(
        op,
        exec,
        input.batch(),
        Some(m.max(n)),
        item_muladds,
        || check_shape::<T::Entity>(op, m, n),
        |par| qr_req::<T::Entity>(m, n, false, par),
    )?;
    let (lanes, per_lane) = requirement.unwrap_or((1, 0));
    batch::check_workspace(op, workspace, per_lane, lanes)?;
    batch::run(
        op,
        input.batch(),
        exec,
        Some(m.max(n)),
        item_muladds,
        &mut (fixed_out(q, q_len, "Q"), fixed_out(r, r_len, "R")),
        batch::ScratchArena::new(workspace, per_lane),
        |par, stack| QrScratch::<T::Entity>::new(m, n, false, par, MemStack::new(stack)),
        |index, (q, r), scratch, par| {
            qr_item::<T>(input.item(index), q, r, scratch, par);
            Ok(())
        },
    )
}

/// Thin QR of every `m x n` matrix of a batch, `A = Q R`.
///
/// `input` is `[m, n, b...]`. Per item, `q` receives the column-major `m x min(m, n)` factor with
/// orthonormal columns and `r` the column-major upper-trapezoidal `min(m, n) x n` factor.
///
/// This is the `Vec` wrapper of [`qr_into`]: it reserves both outputs and allocates one scratch
/// block of [`qr_workspace`] bytes per call.
///
/// # Errors
///
/// [`Error::InvalidArgument`](crate::Error::InvalidArgument) when `input` has rank below 2 or an
/// output size overflows. The outputs are empty on error.
///
/// # Examples
///
/// ```
/// use strided_view::RawStridedRef;
/// use tcontract_exec::Exec;
/// use tlinalg::{qr::qr, Op};
///
/// let a = [3.0_f64, 4.0];
/// let (mut q, mut r) = (Vec::new(), Vec::new());
/// qr(
///     Op::Qr, RawStridedRef::new(&a, &[2, 1], &[1, 2], 0).unwrap(), &mut q, &mut r,
///     Exec::serial(),
/// ).unwrap();
/// assert!((r[0].abs() - 5.0).abs() < 1e-12);
/// ```
pub fn qr<T: FaerScalar>(
    op: Op,
    input: RawStridedRef<'_, T>,
    q: &mut Vec<T>,
    r: &mut Vec<T>,
    exec: Exec<'_>,
) -> Result<()> {
    q.clear();
    r.clear();
    // The published order: operand validation before any workspace query or allocation.
    let input = BatchedRef::new(op, "input", input)?;
    let (m, n) = (input.rows(), input.cols());
    let (q_len, r_len, _) = qr_shape::<T>(op, m, n)?;
    let batch = input.batch();
    let bytes = qr_bytes::<T>(op, m, n, false, batch, exec)?;
    let mut workspace = vec![MaybeUninit::<u8>::uninit(); bytes];
    let mut q_out = VecOut::new(q, checked_product(op, "Q", &[q_len, batch])?);
    let mut r_out = VecOut::new(r, checked_product(op, "R", &[r_len, batch])?);
    qr_into(
        op,
        input.raw(),
        q_out.spare(),
        r_out.spare(),
        &mut workspace,
        exec,
    )?;
    q_out.publish();
    r_out.publish();
    Ok(())
}

/// The magnitude a rank decision compares against its tolerance: `|x|` for the real scalars and
/// `hypot(re, im)` evaluated in `f64` for the complex ones.
///
/// A host reading the diagonal of the `R` returned by [`rank_revealing_qr`] gets exactly the values
/// the pre-extraction rank decision used.
///
/// # Examples
///
/// ```
/// use num_complex::Complex64;
/// assert_eq!(tlinalg::qr::magnitude(Complex64::new(3.0, 4.0)), 5.0);
/// assert_eq!(tlinalg::qr::magnitude(-2.0_f32), 2.0);
/// ```
#[must_use]
pub fn magnitude<T: FaerScalar>(value: T) -> f64 {
    T::entity_magnitude(T::entity_slice(core::slice::from_ref(&value))[0])
}

/// Bytes one batched rank-revealing QR call needs in its borrowed workspace.
///
/// Same composition as [`qr_workspace`] with the column-pivoted factor and its permutation
/// storage; the canonical `Op` of the refusal is `Op::RankRevealingQr`.
///
/// # Errors
///
/// [`Error::InvalidArgument`](crate::Error::InvalidArgument) when a size overflows, and the
/// executor's refusal for a forced strategy this route cannot honor.
pub fn rank_revealing_qr_workspace<T: FaerScalar>(
    m: usize,
    n: usize,
    batch: usize,
    exec: Exec<'_>,
) -> Result<usize> {
    qr_bytes::<T>(Op::RankRevealingQr, m, n, true, batch, exec)
}

/// Column-pivoted QR of every `m x n` matrix of a batch, `A P = Q R`, into caller-owned storage.
///
/// The output lengths are those of [`qr_into`], plus `n` `i64` column indices per item. See
/// [`rank_revealing_qr`] for the conventions and [`qr_into`] for the workspace contract.
///
/// # Errors
///
/// As [`qr_into`], plus [`Error::InvalidArgument`](crate::Error::InvalidArgument) when a column
/// index does not fit `i64`.
pub fn rank_revealing_qr_into<T: FaerScalar>(
    op: Op,
    input: RawStridedRef<'_, T>,
    q: &mut [MaybeUninit<T>],
    r: &mut [MaybeUninit<T>],
    permutation: &mut [MaybeUninit<i64>],
    workspace: &mut [MaybeUninit<u8>],
    exec: Exec<'_>,
) -> Result<()> {
    let input = BatchedRef::new(op, "input", input)?;
    let (m, n) = (input.rows(), input.cols());
    let (q_len, r_len, item_muladds) = qr_shape::<T>(op, m, n)?;
    let requirement = lane_requirement(
        op,
        exec,
        input.batch(),
        Some(m.max(n)),
        item_muladds,
        || check_shape::<T::Entity>(op, m, n),
        |par| qr_req::<T::Entity>(m, n, true, par),
    )?;
    let (lanes, per_lane) = requirement.unwrap_or((1, 0));
    batch::check_workspace(op, workspace, per_lane, lanes)?;
    batch::run(
        op,
        input.batch(),
        exec,
        Some(m.max(n)),
        item_muladds,
        &mut (
            fixed_out(q, q_len, "Q"),
            fixed_out(r, r_len, "R"),
            fixed_out(permutation, n, "permutation"),
        ),
        batch::ScratchArena::new(workspace, per_lane),
        |par, stack| QrScratch::<T::Entity>::new(m, n, true, par, MemStack::new(stack)),
        |index, (q, r, permutation), scratch, par| {
            rank_revealing_qr_item::<T>(op, input.item(index), (q, r, permutation), scratch, par)
        },
    )
}

/// Column-pivoted QR of every `m x n` matrix of a batch, `A P = Q R`.
///
/// `input` is `[m, n, b...]`. Per item, `q` (`m x min(m, n)`) and `r` (`min(m, n) x n`, upper
/// trapezoidal) receive column-major factors, and `permutation` receives `n` column indices in the
/// public gather convention: column `j` of `A P` is column `permutation[j]` of `A`. An all-zero
/// item is not factored: it gets the leading `min(m, n)` identity columns as `Q`, a zero `R` and
/// the identity permutation (its rank is zero by the host's rule). Non-finite input is the host's
/// to screen.
///
/// This is the `Vec` wrapper of [`rank_revealing_qr_into`].
///
/// # Errors
///
/// [`Error::InvalidArgument`](crate::Error::InvalidArgument) when `input` has rank below 2, an
/// output size overflows, or a column index does not fit `i64`. The outputs are empty on error.
///
/// # Examples
///
/// ```
/// use strided_view::RawStridedRef;
/// use tcontract_exec::Exec;
/// use tlinalg::{qr::rank_revealing_qr, Op};
///
/// let a = [1.0_f64, 0.0, 0.0, 5.0];
/// let (mut q, mut r, mut perm) = (Vec::new(), Vec::new(), Vec::new());
/// rank_revealing_qr(
///     Op::RankRevealingQr, RawStridedRef::new(&a, &[2, 2], &[1, 2], 0).unwrap(),
///     &mut q, &mut r, &mut perm, Exec::serial(),
/// ).unwrap();
/// assert_eq!(perm, [1, 0]);
/// ```
pub fn rank_revealing_qr<T: FaerScalar>(
    op: Op,
    input: RawStridedRef<'_, T>,
    q: &mut Vec<T>,
    r: &mut Vec<T>,
    permutation: &mut Vec<i64>,
    exec: Exec<'_>,
) -> Result<()> {
    q.clear();
    r.clear();
    permutation.clear();
    // The published order: operand validation before any workspace query or allocation.
    let input = BatchedRef::new(op, "input", input)?;
    let (m, n) = (input.rows(), input.cols());
    let (q_len, r_len, _) = qr_shape::<T>(op, m, n)?;
    let batch = input.batch();
    let bytes = qr_bytes::<T>(op, m, n, true, batch, exec)?;
    let mut workspace = vec![MaybeUninit::<u8>::uninit(); bytes];
    let mut q_out = VecOut::new(q, checked_product(op, "Q", &[q_len, batch])?);
    let mut r_out = VecOut::new(r, checked_product(op, "R", &[r_len, batch])?);
    let mut perm_out = VecOut::new(
        permutation,
        checked_product(op, "permutation", &[n, batch])?,
    );
    rank_revealing_qr_into(
        op,
        input.raw(),
        q_out.spare(),
        r_out.spare(),
        perm_out.spare(),
        &mut workspace,
        exec,
    )?;
    q_out.publish();
    r_out.publish();
    perm_out.publish();
    Ok(())
}
