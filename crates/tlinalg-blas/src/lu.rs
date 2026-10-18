//! Partial-pivot LU on LAPACK: the batched packed family (factor, prepared solve, fused
//! factor+solve) and the batched explicit [`lu`].
//!
//! Moved from `tenferro-linalg`'s LAPACK provider (same project, MIT OR Apache-2.0). The kernels
//! keep the LAPACK packed format — unit-lower `L` below the diagonal, `U` on and above it, one-based
//! row-swap pivots — so the factors stay interchangeable with the faer-backed implementation.
//!
//! # Host contract
//!
//! The packed family is an in-place route: every entry point takes the **whole batch** as compact,
//! batch-contiguous buffers (`m x n` matrices, `min(m, n)` pivots and one parity per matrix) and
//! loops over it serially. LAPACK owns its threading, so no entry point takes a parallelism token.
//!
//! The packed family takes no pooled buffers; the explicit [`lu`] takes a [`Workspace`].

use crate::{Error, IndexWorkspace, Op, Result, Workspace};

use crate::LapackScalar;

/// A shape product overflowed `usize`.
fn checked_product(op: Op, role: &'static str, shape: &[usize]) -> Result<usize> {
    shape
        .iter()
        .try_fold(1usize, |acc, &dim| acc.checked_mul(dim))
        .ok_or_else(|| Error::InvalidArgument {
            op,
            role: "configuration",
            detail: format!("{role} element count overflows usize"),
        })
}

/// A dimension LAPACK cannot express in its `i32` interface.
fn dim_i32(op: Op, value: usize) -> Result<i32> {
    i32::try_from(value).map_err(|_| Error::InvalidArgument {
        op,
        role: "dimension",
        detail: format!("dimension {value} exceeds the LAPACK i32 range"),
    })
}

/// Translate a LAPACK `info` into this crate's error vocabulary.
///
/// `info < 0` is an illegal argument; `info > 0` is a numerical failure. Callers pass
/// `info.min(0)` where a positive value is meaningful rather than a failure — an exactly singular
/// `?getrf` is not an error, and the packed factors stay valid for a caller that checks the `U`
/// diagonal.
fn check_info(op: Op, routine: &'static str, info: i32) -> Result<()> {
    if info < 0 {
        return Err(Error::InvalidArgument {
            op,
            role: "lapack_argument",
            detail: format!("LAPACK {routine} argument {} had an illegal value", -info),
        });
    }
    if info > 0 {
        return Err(Error::NonConvergence { op });
    }
    Ok(())
}

/// Reject a pivot outside `1..=n`.
///
/// `?getrs` applies `ipiv` through `?laswp` without bounds checks, so every stored pivot must be a
/// one-based row index in `1..=n` before the call. A host that splits a batch into chunks can call
/// this on the whole batch first.
///
/// # Errors
///
/// Returns [`Error::InvalidArgument`] with role `"pivot"`.
pub fn validate_pivots(op: Op, n: usize, ipiv: &[i32]) -> Result<()> {
    for &pivot_one_based in ipiv {
        let in_range = usize::try_from(pivot_one_based)
            .map(|pivot| (1..=n).contains(&pivot))
            .unwrap_or(false);
        if !in_range {
            return Err(Error::InvalidArgument {
                op,
                role: "pivot",
                detail: format!("LU pivot index {pivot_one_based} is outside 1..={n}"),
            });
        }
    }
    Ok(())
}

/// Check that every `(per_matrix_len, buffer_len)` pair describes the same number of matrices.
fn check_batches(op: Op, buffers: [(usize, usize); 3], batch: usize) -> Result<()> {
    for ((per_matrix, len), what) in buffers
        .into_iter()
        .zip(["packed LU", "pivots", "rhs batch"])
    {
        if len != checked_product(op, what, &[per_matrix, batch])? {
            return Err(Error::Inconsistent {
                op,
                detail: "packed LU, pivot, and rhs buffers describe different batches",
            });
        }
    }
    Ok(())
}

/// Factor one matrix in place and write its parity.
fn factor_one<T: LapackScalar>(
    op: Op,
    m_i32: i32,
    n_i32: i32,
    matrix: &mut [T],
    ipiv: &mut [i32],
    parity: Option<&mut T>,
    reject_singular: bool,
) -> Result<()> {
    let mut info = 0;
    // SAFETY: `matrix` is a compact column-major `m x n` block of this call's `lu`, `ipiv` holds
    // `min(m, n)` pivots, and `m_i32`/`n_i32` were derived from those lengths.
    unsafe {
        T::getrf(m_i32, n_i32, matrix, m_i32, ipiv, &mut info);
    }
    check_info(op, "getrf", info.min(0))?;
    if reject_singular && info > 0 {
        return Err(Error::Singular { op });
    }
    if let Some(parity) = parity {
        let swaps = ipiv
            .iter()
            .enumerate()
            .filter(|(index, pivot)| **pivot != (*index as i32 + 1))
            .count();
        *parity = if swaps.is_multiple_of(2) {
            T::one()
        } else {
            T::minus_one()
        };
    }
    Ok(())
}

/// Factor every compact column-major `m x n` matrix of the batch `lu` in place.
///
/// `pivots` receives `min(m, n)` one-based pivots per matrix and `parity` one permutation parity per
/// matrix. Exactly singular matrices are **not** an error: `?getrf` reports them through a positive
/// `info`, and the packed factors stay usable.
///
/// # Errors
///
/// Returns [`Error::Inconsistent`] when the buffers describe different batches,
/// [`Error::InvalidArgument`] for a dimension outside the LAPACK `i32` range or an illegal LAPACK
/// argument, and [`Error::NonConvergence`] for a positive `info`.
pub fn lu_factor<T: LapackScalar>(
    op: Op,
    m: usize,
    n: usize,
    lu: &mut [T],
    pivots: &mut [i32],
    parity: &mut [T],
) -> Result<()> {
    let k = m.min(n);
    let matrix_len = checked_product(op, "matrix shape", &[m, n])?;
    let batch = parity.len();
    check_batches(
        op,
        [(matrix_len, lu.len()), (k, pivots.len()), (1, parity.len())],
        batch,
    )?;
    if batch == 0 || matrix_len == 0 {
        return Ok(());
    }
    let m_i32 = dim_i32(op, m)?;
    let n_i32 = dim_i32(op, n)?;
    // INVARIANT: the buffers were checked above to hold exactly `batch` matrices, pivot vectors and
    // parities, and the early return gives `matrix_len > 0`, hence `k > 0`, so the chunk iterators
    // stay in lockstep. The serial loop is intentional: LAPACK owns threading inside `?getrf`, and
    // one call per matrix writes straight into the caller's buffers without scratch.
    for ((matrix, ipiv), parity) in lu
        .chunks_exact_mut(matrix_len)
        .zip(pivots.chunks_exact_mut(k))
        .zip(parity.iter_mut())
    {
        factor_one::<T>(op, m_i32, n_i32, matrix, ipiv, Some(parity), false)?;
    }
    Ok(())
}

/// Solve `op(A) X = B` for every matrix of the batch from packed `?getrf` factors.
///
/// `output` enters holding the compact column-major RHS batch and leaves holding the solution.
/// `op(A)` is `A`, `A^T`, `A^H`, or `conj(A)` from the flags: the first three map onto `?getrs` with
/// `trans = N/T/C`, and `conj(A) x = b` is solved as `A conj(x) = conj(b)`.
///
/// # Errors
///
/// Returns [`Error::InvalidArgument`] for a pivot outside `1..=n`, a dimension outside the LAPACK
/// `i32` range, inconsistent buffer lengths, or an illegal LAPACK argument, and
/// [`Error::NonConvergence`] for a positive `info`. Pivots are validated for the whole batch
/// **before** any output is written.
// INVARIANT: the argument list mirrors the host call it replaces; each value is a distinct operand
// or flag, so grouping them would add a wrapper without removing an argument.
#[allow(clippy::too_many_arguments)]
pub fn lu_solve_prepared<T: LapackScalar>(
    op: Op,
    n: usize,
    nrhs: usize,
    packed_lu: &[T],
    pivots: &[i32],
    output: &mut [T],
    transpose_a: bool,
    conjugate_a: bool,
) -> Result<()> {
    let matrix_len = checked_product(op, "matrix", &[n, n])?;
    let rhs_len = checked_product(op, "rhs", &[n, nrhs])?;
    if matrix_len == 0 || rhs_len == 0 {
        return Ok(());
    }
    let batch = packed_lu.len() / matrix_len;
    check_batches(
        op,
        [
            (matrix_len, packed_lu.len()),
            (n, pivots.len()),
            (rhs_len, output.len()),
        ],
        batch,
    )?;
    validate_pivots(op, n, pivots)?;
    let n_i32 = dim_i32(op, n)?;
    let nrhs_i32 = dim_i32(op, nrhs)?;
    let (trans, conjugate_rhs) = match (transpose_a, conjugate_a) {
        (false, false) => (b'N', false),
        (true, false) => (b'T', false),
        (true, true) => (b'C', false),
        (false, true) => (b'N', true),
    };
    if conjugate_rhs {
        T::conj_in_place(output);
    }
    // INVARIANT: the buffers were checked above to hold exactly `batch` nonempty matrices, pivot
    // vectors and RHS blocks, and every pivot is in `1..=n`, so each `?getrs` reads only its own
    // factors and writes only its own RHS block.
    for ((matrix, ipiv), rhs) in packed_lu
        .chunks_exact(matrix_len)
        .zip(pivots.chunks_exact(n))
        .zip(output.chunks_exact_mut(rhs_len))
    {
        let mut info = 0;
        // SAFETY: `matrix` and `ipiv` are this item's matching `?getrf` factor, `rhs` is its
        // `n x nrhs` right-hand side, and the dimensions were derived from those lengths.
        unsafe {
            T::getrs(
                trans, n_i32, nrhs_i32, matrix, n_i32, ipiv, rhs, n_i32, &mut info,
            );
        }
        check_info(op, "getrs", info)?;
    }
    if conjugate_rhs {
        T::conj_in_place(output);
    }
    Ok(())
}

/// Factor and solve `A X = B` for every matrix of the batch, keeping the packed factors.
///
/// `packed_lu` enters holding the compact `A` batch and leaves holding the packed factors;
/// `pivots` receives the one-based pivots; `output` enters holding the RHS batch and leaves holding
/// `X`. One `?getrf` and one `?getrs` per matrix, with no scratch because the factors are an output.
///
/// # Errors
///
/// Returns [`Error::Singular`] when a matrix is exactly singular and there is a nonempty RHS to
/// solve, [`Error::Inconsistent`] for inconsistent buffers, [`Error::InvalidArgument`] for a
/// dimension outside the LAPACK `i32` range or an illegal LAPACK argument, and
/// [`Error::NonConvergence`] for a positive `info` during a solve.
// INVARIANT: the argument list mirrors the host call it replaces (see `lu_solve_prepared`).
#[allow(clippy::too_many_arguments)]
pub fn lu_factor_solve<T: LapackScalar>(
    op: Op,
    n: usize,
    nrhs: usize,
    packed_lu: &mut [T],
    pivots: &mut [i32],
    output: &mut [T],
) -> Result<()> {
    let matrix_len = checked_product(op, "matrix", &[n, n])?;
    let rhs_len = checked_product(op, "rhs", &[n, nrhs])?;
    if matrix_len == 0 {
        return Ok(());
    }
    let batch = packed_lu.len() / matrix_len;
    check_batches(
        op,
        [
            (matrix_len, packed_lu.len()),
            (n, pivots.len()),
            (rhs_len, output.len()),
        ],
        batch,
    )?;
    let n_i32 = dim_i32(op, n)?;
    let nrhs_i32 = dim_i32(op, nrhs)?;
    if rhs_len == 0 {
        // Nothing to solve: only factor, matching `lu_factor` on singular input.
        for (matrix, ipiv) in packed_lu
            .chunks_exact_mut(matrix_len)
            .zip(pivots.chunks_exact_mut(n))
        {
            factor_one::<T>(op, n_i32, n_i32, matrix, ipiv, None, false)?;
        }
        return Ok(());
    }
    for ((matrix, ipiv), rhs) in packed_lu
        .chunks_exact_mut(matrix_len)
        .zip(pivots.chunks_exact_mut(n))
        .zip(output.chunks_exact_mut(rhs_len))
    {
        factor_one::<T>(op, n_i32, n_i32, matrix, ipiv, None, true)?;
        let mut info = 0;
        // SAFETY: this item's factor, pivots and right-hand side, as the prepared solve validates.
        unsafe {
            T::getrs(
                b'N', n_i32, nrhs_i32, matrix, n_i32, ipiv, rhs, n_i32, &mut info,
            );
        }
        check_info(op, "getrs", info)?;
    }
    Ok(())
}

/// The batched outputs of [`lu`], each cleared and then filled in batch order.
#[derive(Debug)]
pub struct LuOutputs<'o, T> {
    /// `m x m` permutation matrices `P` with `P A = L U`, column-major.
    pub p: &'o mut Vec<T>,
    /// Unit-lower `m x k` factors, column-major.
    pub l: &'o mut Vec<T>,
    /// Upper `k x n` factors, column-major.
    pub u: &'o mut Vec<T>,
    /// One permutation parity (`1` or `-1`) per matrix.
    pub parity: &'o mut Vec<T>,
}

impl<T> LuOutputs<'_, T> {
    fn clear(&mut self) {
        self.p.clear();
        self.l.clear();
        self.u.clear();
        self.parity.clear();
    }
}

/// Explicit partial-pivot LU, `P A = L U`, of every matrix of a batch.
///
/// Moved from tenferro-linalg's LAPACK backend (same project, MIT OR Apache-2.0). `a` has dims
/// `[m, n, batch...]` and is not modified; `k = min(m, n)`. One destructible `m x n` copy and one
/// pivot buffer are acquired from `workspace` for the whole call, reused per matrix and released on
/// success. Exactly singular input is **not** an error.
///
/// # Errors
///
/// [`Error::InvalidArgument`] for a malformed operand, a dimension outside the LAPACK `i32` range
/// or an illegal LAPACK argument, and [`Error::Internal`] when LAPACK returns a pivot outside the
/// matrix; always for the first failing matrix in batch order. On error every output is empty.
pub fn lu<T, W>(
    op: Op,
    a: strided_view::RawStridedRef<'_, T>,
    mut outputs: LuOutputs<'_, T>,
    workspace: &mut W,
) -> Result<()>
where
    T: LapackScalar,
    W: Workspace<T> + IndexWorkspace,
{
    outputs.clear();
    let result = lu_run(op, a, &mut outputs, workspace);
    if result.is_err() {
        outputs.clear();
    }
    result
}

fn lu_run<T, W>(
    op: Op,
    a: strided_view::RawStridedRef<'_, T>,
    out: &mut LuOutputs<'_, T>,
    workspace: &mut W,
) -> Result<()>
where
    T: LapackScalar,
    W: Workspace<T> + IndexWorkspace,
{
    use crate::batch::{batch_len, Input};
    use crate::common;

    let input = Input::new(op, "input", a)?;
    let (m, n) = (input.layout.rows, input.layout.cols);
    let k = m.min(n);
    let count = input.layout.count;
    let m_i32 = common::dim_i32(op, m)?;
    let n_i32 = common::dim_i32(op, n)?;
    let matrix_len = common::checked_product(op, "matrix", &[m, n])?;
    let p_len = common::checked_product(op, "permutation matrix", &[m, m])?;
    let l_len = common::checked_product(op, "lower factor", &[m, k])?;
    let u_len = common::checked_product(op, "upper factor", &[k, n])?;
    if count == 0 {
        return Ok(());
    }
    if matrix_len == 0 {
        // No factorization to run: the identity permutation of the (possibly nonempty) row space,
        // empty factors, and parity one.
        out.p.reserve(batch_len(op, p_len, count)?);
        for _ in 0..count {
            let start = out.p.len();
            out.p.resize(start + p_len, T::default());
            for row in 0..m {
                out.p[start + row + row * m] = T::one();
            }
        }
        out.parity.resize(count, T::one());
        return Ok(());
    }
    out.p.reserve(batch_len(op, p_len, count)?);
    out.l.reserve(batch_len(op, l_len, count)?);
    out.u.reserve(batch_len(op, u_len, count)?);
    out.parity.reserve(count);
    let mut lu = workspace.acquire_capacity(matrix_len);
    let mut ipiv = workspace.acquire_zeroed_index(k);
    common::check_scratch(op, ipiv.len(), k)?;
    for offset in input.layout.offsets() {
        lu.clear();
        input.gather(offset, &mut lu);
        let mut info = 0;
        // SAFETY: `lu` is a mutable compact column-major `m x n` matrix and `ipiv` holds
        // `min(m, n)` pivots.
        unsafe {
            T::getrf(m_i32, n_i32, &mut lu, m_i32, &mut ipiv, &mut info);
        }
        common::check_info(op, "getrf", info.min(0))?;

        // `P` is the identity with each `?getrf` swap applied to its rows, so it is built in place
        // from the pivots: no permutation vector (and no per-call allocation) is needed.
        let start = out.p.len();
        out.p.resize(start + p_len, T::default());
        for row in 0..m {
            out.p[start + row + row * m] = T::one();
        }
        let mut swap_count = 0usize;
        for (idx, &pivot_one_based) in ipiv.iter().take(k).enumerate() {
            let pivot = match usize::try_from(pivot_one_based - 1) {
                Ok(pivot) => pivot,
                Err(_) => {
                    return Err(Error::Internal {
                        op,
                        detail: "LAPACK getrf returned an invalid pivot index".to_owned(),
                    });
                }
            };
            if pivot >= m {
                return Err(Error::Internal {
                    op,
                    detail: "LAPACK getrf returned an out-of-bounds pivot index".to_owned(),
                });
            }
            if pivot != idx {
                for col in 0..m {
                    out.p.swap(start + idx + col * m, start + pivot + col * m);
                }
                swap_count += 1;
            }
        }
        out.parity.push(if swap_count.is_multiple_of(2) {
            T::one()
        } else {
            T::minus_one()
        });
        for col in 0..k {
            out.l.extend(core::iter::repeat_n(T::default(), col));
            out.l.push(T::one());
            out.l
                .extend_from_slice(&lu[col + 1 + col * m..(col + 1) * m]);
        }
        for col in 0..n {
            let diag = k.min(col + 1);
            out.u.extend_from_slice(&lu[col * m..col * m + diag]);
            out.u.extend(core::iter::repeat_n(T::default(), k - diag));
        }
    }
    workspace.release(lu);
    workspace.release_index(ipiv);
    Ok(())
}
