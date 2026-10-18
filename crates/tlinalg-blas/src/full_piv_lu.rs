//! Batched complete-pivot LU on LAPACK (`?getc2`/`?gesc2`).
//!
//! Moved from tenferro-linalg's LAPACK backend (same project, MIT OR Apache-2.0). Exactly singular
//! input is [`Error::Singular`] on both the factorization and the solve.

use strided_view::RawStridedRef;

use crate::batch::{batch_len, clear_on_error, Input};
use crate::common::{check_info, check_scratch, checked_product, dim_i32};
use crate::symbols::Symbols;
use crate::{Error, IndexWorkspace, LapackScalar, Op, Result, Workspace};

/// Run `?getc2` on one compact `n x n` matrix in place.
///
/// Reference LAPACK sets `IPIV(N) = JPIV(N) = N` on return, but some providers (Apple Accelerate)
/// leave that last entry untouched. The last step of a complete-pivot elimination never swaps, and
/// `?gesc2` only applies the first `N - 1` interchanges, so writing `N` here is exact.
fn getc2_in_place<T: LapackScalar>(
    op: Op,
    data: &mut [T],
    n_i32: i32,
    ipiv: &mut [i32],
    jpiv: &mut [i32],
) -> Result<()> {
    let mut info = 0;
    // SAFETY: callers pass a compact `n x n` matrix and at least `n` entries per pivot array.
    unsafe {
        T::getc2(n_i32, data, n_i32.max(1), ipiv, jpiv, &mut info);
    }
    check_info(op, "getc2", info.min(0))?;
    if info > 0 {
        return Err(Error::Singular { op });
    }
    if let (Some(last_row), Some(last_col)) = (ipiv.last_mut(), jpiv.last_mut()) {
        *last_row = n_i32;
        *last_col = n_i32;
    }
    Ok(())
}

/// Append an `n x n` permutation matrix built from `pivots` in place: the identity with each swap
/// applied to its rows.
///
/// Both `P` and `Q` use this convention (`M[row][perm[row]] = 1`), exactly as the previous
/// `permutation_from_pivots` + `push_permutation_matrix` pair did, so `Q` is the same matrix the
/// faer provider returns for the same pivots. The swap sequence is applied to the identity block
/// directly, so no permutation vector (and no per-call allocation) is needed.
fn push_permutation_matrix<T: LapackScalar>(
    op: Op,
    pivots: &[i32],
    out: &mut Vec<T>,
) -> Result<()> {
    let n = pivots.len();
    let start = out.len();
    out.resize(start + n * n, T::default());
    for index in 0..n {
        out[start + index + index * n] = T::one();
    }
    for (idx, &pivot_one_based) in pivots.iter().enumerate() {
        let pivot = match usize::try_from(pivot_one_based - 1) {
            Ok(pivot) if pivot < n => pivot,
            _ => {
                return Err(Error::Internal {
                    op,
                    detail: format!(
                        "{}: LAPACK getc2 returned an invalid pivot index",
                        op.as_str()
                    ),
                });
            }
        };
        if pivot == idx {
            continue;
        }
        for col in 0..n {
            out.swap(start + idx + col * n, start + pivot + col * n);
        }
    }
    Ok(())
}

fn swaps(pivots: &[i32]) -> usize {
    pivots
        .iter()
        .enumerate()
        .filter(|(idx, pivot)| **pivot != (*idx as i32 + 1))
        .count()
}

/// The batched outputs of [`full_piv_lu`], each cleared and then filled in batch order.
#[derive(Debug)]
pub struct FullPivLuOutputs<'o, T> {
    /// `n x n` row permutation matrices `P`, column-major.
    pub p: &'o mut Vec<T>,
    /// Unit-lower `n x n` factors, column-major.
    pub l: &'o mut Vec<T>,
    /// Upper `n x n` factors, column-major.
    pub u: &'o mut Vec<T>,
    /// `n x n` column permutation matrices `Q` with `P A Qᵀ = L U`, column-major.
    pub q: &'o mut Vec<T>,
    /// One combined parity (`1` or `-1`) per matrix.
    pub parity: &'o mut Vec<T>,
}

impl<T> FullPivLuOutputs<'_, T> {
    fn clear(&mut self) {
        self.p.clear();
        self.l.clear();
        self.u.clear();
        self.q.clear();
        self.parity.clear();
    }
}

/// Explicit complete-pivot LU, `P A Qᵀ = L U`, of every square matrix of a batch.
///
/// `a` has dims `[n, n, batch...]` and is not modified. One destructible copy and both pivot
/// arrays are acquired from `workspace` for the whole call, reused per matrix and released on
/// success.
///
/// # Errors
///
/// [`Error::Singular`] for an exactly singular matrix, [`Error::InvalidArgument`] for a malformed
/// operand, a dimension outside the LAPACK `i32` range or an illegal LAPACK argument, and
/// [`Error::Internal`] when LAPACK returns a pivot outside the matrix; always for the first failing
/// matrix in batch order. On error every output is empty.
pub fn full_piv_lu<T, W>(
    op: Op,
    a: RawStridedRef<'_, T>,
    mut outputs: FullPivLuOutputs<'_, T>,
    workspace: &mut W,
) -> Result<()>
where
    T: LapackScalar,
    W: Workspace<T> + IndexWorkspace,
{
    outputs.clear();
    let result = factor_run(op, a, &mut outputs, workspace);
    if result.is_err() {
        outputs.clear();
    }
    result
}

fn factor_run<T, W>(
    op: Op,
    a: RawStridedRef<'_, T>,
    out: &mut FullPivLuOutputs<'_, T>,
    workspace: &mut W,
) -> Result<()>
where
    T: LapackScalar,
    W: Workspace<T> + IndexWorkspace,
{
    let input = Input::new(op, "input", a)?;
    let n = input.square(op)?;
    let n_i32 = dim_i32(op, n)?;
    let matrix_len = checked_product(op, "matrix", &[n, n])?;
    let count = input.layout.count;
    if count == 0 {
        return Ok(());
    }
    if matrix_len == 0 {
        out.parity.resize(count, T::one());
        return Ok(());
    }
    let total = batch_len(op, matrix_len, count)?;
    for buffer in [&mut *out.p, &mut *out.l, &mut *out.u, &mut *out.q] {
        buffer.reserve(total);
    }
    out.parity.reserve(count);
    let mut lu = workspace.acquire_capacity(matrix_len);
    let mut ipiv = workspace.acquire_zeroed_index(n);
    let mut jpiv = workspace.acquire_zeroed_index(n);
    check_scratch(op, ipiv.len().min(jpiv.len()), n)?;
    for offset in input.layout.offsets() {
        lu.clear();
        input.gather(offset, &mut lu);
        getc2_in_place(op, &mut lu, n_i32, &mut ipiv[..n], &mut jpiv[..n])?;
        push_permutation_matrix(op, &ipiv[..n], out.p)?;
        push_permutation_matrix(op, &jpiv[..n], out.q)?;
        for col in 0..n {
            out.l.extend(core::iter::repeat_n(T::default(), col));
            out.l.push(T::one());
            out.l
                .extend_from_slice(&lu[col + 1 + col * n..(col + 1) * n]);
            out.u.extend_from_slice(&lu[col * n..col * n + col + 1]);
            out.u
                .extend(core::iter::repeat_n(T::default(), n - col - 1));
        }
        out.parity.push(
            if (swaps(&ipiv[..n]) + swaps(&jpiv[..n])).is_multiple_of(2) {
                T::one()
            } else {
                T::minus_one()
            },
        );
    }
    workspace.release(lu);
    workspace.release_index(ipiv);
    workspace.release_index(jpiv);
    Ok(())
}

/// Solve `op(A) X = B` through complete-pivot LU for every system of a batch.
///
/// `a` has dims `[n, n, batch...]` and `b` dims `[n, nrhs, batch...]` with the same batch dims;
/// neither is modified. `op(A)` is `A`, or `Aᵀ` with `transpose_a`. `out` is cleared and receives
/// `X` (`n * nrhs` elements per system, compact column-major in batch order). Each column is solved
/// by `?gesc2` and then divided by the scale it reports. One LU copy and both pivot arrays are
/// acquired from `workspace` for the whole call and released on success.
///
/// # Errors
///
/// [`Error::Singular`] for an exactly singular matrix and [`Error::InvalidArgument`] for malformed
/// or mismatched operands, a dimension outside the LAPACK `i32` range or an illegal LAPACK
/// argument; always for the first failing system in batch order. On error `out` is left empty.
pub fn full_piv_lu_solve<T, W>(
    op: Op,
    transpose_a: bool,
    a: RawStridedRef<'_, T>,
    b: RawStridedRef<'_, T>,
    out: &mut Vec<T>,
    workspace: &mut W,
) -> Result<()>
where
    T: LapackScalar,
    W: Workspace<T> + IndexWorkspace,
{
    out.clear();
    let result = solve_run(op, transpose_a, a, b, out, workspace);
    clear_on_error(result, || out.clear())
}

fn solve_run<T, W>(
    op: Op,
    transpose_a: bool,
    a: RawStridedRef<'_, T>,
    b: RawStridedRef<'_, T>,
    out: &mut Vec<T>,
    workspace: &mut W,
) -> Result<()>
where
    T: LapackScalar,
    W: Workspace<T> + IndexWorkspace,
{
    let a = Input::new(op, "a", a)?;
    let b = Input::new(op, "b", b)?;
    let n = a.square(op)?;
    let nrhs = b.layout.cols;
    if b.layout.rows != n {
        return Err(Error::InvalidArgument {
            op,
            role: "shape",
            detail: format!(
                "the right-hand side has {} rows, expected {n}",
                b.layout.rows
            ),
        });
    }
    a.layout.same_batch(op, &b.layout)?;
    let matrix_len = checked_product(op, "matrix", &[n, n])?;
    let rhs_len = checked_product(op, "rhs", &[n, nrhs])?;
    let n_i32 = dim_i32(op, n)?;
    let count = b.layout.count;
    if matrix_len == 0 || rhs_len == 0 || count == 0 {
        return Ok(());
    }
    out.reserve(batch_len(op, rhs_len, count)?);
    let mut lu = workspace.acquire_capacity(matrix_len);
    let mut ipiv = workspace.acquire_zeroed_index(n);
    let mut jpiv = workspace.acquire_zeroed_index(n);
    check_scratch(op, ipiv.len().min(jpiv.len()), n)?;
    // INVARIANT: both operands describe `count` items in the same batch order, and every RHS block
    // splits into `nrhs` exact columns of length `n`. The serial loop is intentional: the LAPACK
    // provider owns threading.
    for (a_offset, b_offset) in a.layout.offsets().zip(b.layout.offsets()) {
        lu.clear();
        if transpose_a {
            a.gather_transposed(a_offset, &mut lu);
        } else {
            a.gather(a_offset, &mut lu);
        }
        getc2_in_place(op, &mut lu, n_i32, &mut ipiv[..n], &mut jpiv[..n])?;
        let start = out.len();
        b.gather(b_offset, out);
        for column in out[start..].chunks_exact_mut(n) {
            let mut scale = <T as Symbols>::real_one();
            // SAFETY: `lu`, `ipiv` and `jpiv` are this matrix's `?getc2` factors and `column`
            // holds `n` entries.
            unsafe {
                T::gesc2(n_i32, &lu, n_i32, column, &ipiv, &jpiv, &mut scale);
            }
            T::apply_inverse_scale(column, scale);
        }
    }
    workspace.release(lu);
    workspace.release_index(ipiv);
    workspace.release_index(jpiv);
    Ok(())
}
