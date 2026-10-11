//! Batched singular value decomposition on LAPACK.
//!
//! Moved from `tenferro-linalg`'s LAPACK provider (same project, MIT OR Apache-2.0). Two policies
//! from there are preserved exactly, because they are what the mechanism counts measure:
//!
//! * the workspace is queried **once per call** and acquired once, then reused across the **serial**
//!   batch loop — LAPACK owns threading inside each call;
//! * `iwork` (`?gesdd`) and the complex `rwork` have lengths the caller computes, not queries, so
//!   they are acquired once per batch too.
//!
//! The factors use the same conventions as the faer-backed implementation: `s` holds `min(m, n)`
//! real singular values, `u` is `m x u_cols` and `vt` is `vt_rows x n`, both column-major, with
//! `vt` holding `Vᴴ`. The input is a rank `2 + B` strided operand `[m, n, batch...]`; outputs are
//! compact and batch-contiguous.

use crate::{Error, IndexWorkspace, Op, Result, Workspace};

use crate::LapackScalar;

/// Which singular factors to compute.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SvdMode {
    /// Thin factors: `U` is `m x k` and `Vᴴ` is `k x n`.
    Thin,
    /// Full factors: `U` is `m x m` and `Vᴴ` is `n x n`.
    Full,
    /// Singular values only.
    Values,
}

impl SvdMode {
    /// The LAPACK `jobz` letter.
    fn job(self) -> u8 {
        match self {
            Self::Thin => b'S',
            Self::Full => b'A',
            Self::Values => b'N',
        }
    }

    /// `(U columns, Vᴴ rows)`; zero in values-only mode.
    fn factor_dims(self, m: usize, n: usize) -> (usize, usize) {
        let k = m.min(n);
        match self {
            Self::Thin => (k, k),
            Self::Full => (m, n),
            Self::Values => (0, 0),
        }
    }
}

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

fn dim_i32(op: Op, value: usize) -> Result<i32> {
    i32::try_from(value).map_err(|_| Error::InvalidArgument {
        op,
        role: "dimension",
        detail: format!("dimension {value} exceeds the LAPACK i32 range"),
    })
}

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

/// The `lwork` LAPACK reports, as an `i32`.
fn work_len(op: Op, routine: &'static str, query: f64) -> Result<i32> {
    if !(query.is_finite() && query >= 1.0) {
        return Err(Error::InvalidWorkspace {
            op,
            library: "LAPACK",
            routine,
            detail: format!("returned invalid workspace size {query}"),
        });
    }
    dim_i32(op, query.ceil() as usize)
}

/// The integer workspace length the selected driver needs.
///
/// `?gesdd` needs `8 * max(k, 1)`; `?gesvd` has no integer workspace at all.
#[cfg(not(feature = "provider-inject"))]
fn iwork_len(op: Op, k: usize) -> Result<usize> {
    checked_product(op, "integer workspace", &[8, k.max(1)])
}

#[cfg(feature = "provider-inject")]
fn iwork_len(_op: Op, _k: usize) -> Result<usize> {
    Ok(0)
}

/// The real workspace length the selected driver needs for complex input.
///
/// `?gesvd` takes a fixed `5 * max(k, 1)`; `?gesdd` needs the length LAPACK documents, which grows
/// with the shape unless the larger dimension exceeds a crossover.
#[cfg(feature = "provider-inject")]
fn complex_rwork_len(op: Op, _jobz: u8, m: usize, n: usize) -> Result<usize> {
    checked_product(op, "real workspace", &[5, m.min(n).max(1)])
}

#[cfg(not(feature = "provider-inject"))]
fn complex_rwork_len(op: Op, jobz: u8, m: usize, n: usize) -> Result<usize> {
    let mn = m.min(n);
    let mx = m.max(n);
    if jobz == b'N' {
        // `SRC/zgesdd.f` documents `5*mn` for `JOBZ = 'N'` and adds "(LAPACK <= 3.6 needs 7*mn)".
        // Accelerate ships the 3.2.1 interface, whose `?gesdd` writes past `5*mn` from just wider
        // than square up to about `1.6*mn` (`DBDSQR`), corrupting the heap. The two extra reals per
        // singular value are cheaper than a second sizing rule, so ask for the pre-3.7 length
        // everywhere.
        return checked_product(op, "real workspace", &[7, mn.max(1)]);
    }
    let threshold = checked_product(op, "workspace crossover", &[10, mn])?;
    let square_term = checked_product(op, "real workspace square term", &[5, mn, mn])?;
    let linear_term = checked_product(op, "real workspace linear term", &[5, mn])?;
    let small_shape_len =
        square_term
            .checked_add(linear_term)
            .ok_or_else(|| Error::InvalidArgument {
                op,
                role: "configuration",
                detail: "real workspace length overflows usize".to_owned(),
            })?;
    if mx > threshold {
        return Ok(small_shape_len);
    }
    let rectangular_term = checked_product(op, "real workspace rectangular term", &[2, mx, mn])?;
    let second_square_term =
        checked_product(op, "real workspace secondary square term", &[2, mn, mn])?;
    let large_shape_len = rectangular_term
        .checked_add(second_square_term)
        .and_then(|len| len.checked_add(mn))
        .ok_or_else(|| Error::InvalidArgument {
            op,
            role: "configuration",
            detail: "real workspace length overflows usize".to_owned(),
        })?;
    Ok(small_shape_len.max(large_shape_len))
}

/// The batched outputs of [`svd`], each cleared and then filled in batch order.
#[derive(Debug)]
pub struct SvdOutputs<'o, T, R> {
    /// `min(m, n)` non-increasing real singular values per matrix.
    pub s: &'o mut Vec<R>,
    /// `m x u_cols` left factors, column-major; empty in values-only mode.
    pub u: &'o mut Vec<T>,
    /// `vt_rows x n` right factors `Vᴴ`, column-major; empty in values-only mode.
    pub vt: &'o mut Vec<T>,
}

impl<T, R> SvdOutputs<'_, T, R> {
    fn clear(&mut self) {
        self.s.clear();
        self.u.clear();
        self.vt.clear();
    }
}

/// Singular value decomposition of every matrix of a batch.
///
/// `a` has dims `[m, n, batch...]` and is not modified: each matrix is copied into one destructible
/// `m x n` scratch acquired once per call. The workspace is queried once and acquired once, then
/// reused for every matrix; `iwork` (`?gesdd`) and the complex `rwork` are acquired once too. All
/// scratch is released on success. The outputs LAPACK writes through slices are extended with
/// zeros per matrix before the call, because an FFI out-slice must be initialised.
///
/// # Errors
///
/// [`Error::InvalidArgument`] for a malformed operand, a dimension outside the LAPACK `i32` range
/// or an illegal LAPACK argument, [`Error::InvalidWorkspace`] for an unusable workspace size, and
/// [`Error::NonConvergence`] when LAPACK fails to converge; always for the first failing matrix in
/// batch order. On error every output is empty.
pub fn svd<T, W>(
    op: Op,
    mode: SvdMode,
    a: strided_view::RawStridedRef<'_, T>,
    mut outputs: SvdOutputs<'_, T, T::Real>,
    workspace: &mut W,
) -> Result<()>
where
    T: LapackScalar,
    W: Workspace<T> + Workspace<T::Real> + IndexWorkspace,
{
    outputs.clear();
    let result = run(op, mode, a, &mut outputs, workspace);
    if result.is_err() {
        outputs.clear();
    }
    result
}

fn run<T, W>(
    op: Op,
    mode: SvdMode,
    a: strided_view::RawStridedRef<'_, T>,
    out: &mut SvdOutputs<'_, T, T::Real>,
    workspace: &mut W,
) -> Result<()>
where
    T: LapackScalar,
    W: Workspace<T> + Workspace<T::Real> + IndexWorkspace,
{
    use crate::batch::{batch_len, Input};

    let input = Input::new(op, "input", a)?;
    let (m, n) = (input.layout.rows, input.layout.cols);
    let count = input.layout.count;
    let k = m.min(n);
    let (u_cols, vt_rows) = mode.factor_dims(m, n);
    let a_len = checked_product(op, "matrix", &[m, n])?;
    let u_len = checked_product(op, "left singular vectors", &[m, u_cols])?;
    let vt_len = checked_product(op, "right singular vectors", &[vt_rows, n])?;
    if count == 0 || a_len == 0 {
        // Nothing to decompose. A full SVD of an empty matrix still has square unitary factors;
        // the host builds those (identity blocks), as before the move.
        return Ok(());
    }
    let job = mode.job();
    let m_i32 = dim_i32(op, m)?;
    let n_i32 = dim_i32(op, n)?;
    let (ldu, ldvt) = if mode == SvdMode::Values {
        (1, 1)
    } else {
        (m_i32, dim_i32(op, vt_rows)?)
    };
    out.s.resize(batch_len(op, k, count)?, <T::Real>::default());
    out.u.resize(batch_len(op, u_len, count)?, T::default());
    out.vt.resize(batch_len(op, vt_len, count)?, T::default());
    let mut matrix: Vec<T> = workspace.acquire_capacity(a_len);
    let iwork_len = iwork_len(op, k)?;
    // `?gesvd` has no integer workspace at all, so it must not touch the pool for one: taking an
    // integer buffer and returning it would churn the retained capacity of a pool it never used.
    let mut iwork = if iwork_len > 0 {
        workspace.acquire_zeroed_index(iwork_len)
    } else {
        Vec::new()
    };
    // The real routines have no real workspace at all; the complex ones need one of a computed
    // length, held for the whole batch alongside `work` and `iwork`.
    let rwork_len = if T::COMPLEX {
        complex_rwork_len(op, job, m, n)?
    } else {
        0
    };
    let mut rwork: Vec<T::Real> = if rwork_len > 0 {
        workspace.acquire_zeroed(rwork_len)
    } else {
        Vec::new()
    };
    if iwork.len() < iwork_len || rwork.len() < rwork_len {
        return Err(Error::Inconsistent {
            op,
            detail: "the workspace returned fewer elements than the routine requires",
        });
    }
    let mut work: Option<(i32, Vec<T>)> = None;
    let mut info = 0;
    for (index, offset) in input.layout.offsets().enumerate() {
        matrix.clear();
        input.gather(offset, &mut matrix);
        let s_i = &mut out.s[index * k..(index + 1) * k];
        let u_i = &mut out.u[index * u_len..(index + 1) * u_len];
        let vt_i = &mut out.vt[index * vt_len..(index + 1) * vt_len];
        if work.is_none() {
            // A stack slot, not a heap buffer: the workspace query writes one value and the
            // caller-visible allocation count must not grow because of it.
            let mut query = [T::default(); 1];
            // SAFETY: every buffer is sized for this item as documented above.
            unsafe {
                T::svd_driver(
                    job,
                    job,
                    m_i32,
                    n_i32,
                    &mut matrix,
                    m_i32,
                    s_i,
                    u_i,
                    ldu,
                    vt_i,
                    ldvt,
                    &mut query,
                    -1,
                    &mut rwork,
                    &mut iwork,
                    &mut info,
                );
            }
            check_info(op, "svd(work query)", info)?;
            let lwork = work_len(op, T::routine_name(), T::work_query_len(query[0]))?;
            let buffer: Vec<T> = workspace.acquire_zeroed(lwork as usize);
            // A `Workspace` implementation is safe code and may return anything, but LAPACK
            // writes into these buffers up to the lengths it was told, so the host's promise is
            // checked here rather than trusted at a raw FFI boundary.
            if buffer.len() < lwork as usize {
                return Err(Error::Inconsistent {
                    op,
                    detail: "the workspace returned fewer elements than the routine requires",
                });
            }
            work = Some((lwork, buffer));
        }
        let Some((lwork, buffer)) = work.as_mut() else {
            unreachable!("the workspace was just queried");
        };
        // SAFETY: every buffer is sized for this item, and `buffer` has the queried length for
        // this shape.
        unsafe {
            T::svd_driver(
                job,
                job,
                m_i32,
                n_i32,
                &mut matrix,
                m_i32,
                s_i,
                u_i,
                ldu,
                vt_i,
                ldvt,
                buffer,
                *lwork,
                &mut rwork,
                &mut iwork,
                &mut info,
            );
        }
        check_info(op, T::routine_name(), info)?;
    }
    workspace.release(matrix);
    if let Some((_, buffer)) = work {
        workspace.release(buffer);
    }
    if iwork_len > 0 {
        workspace.release_index(iwork);
    }
    if rwork_len > 0 {
        workspace.release(rwork);
    }
    Ok(())
}

#[cfg(test)]
mod tests;
