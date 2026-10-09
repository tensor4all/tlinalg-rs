//! Batched general (non-Hermitian) eigendecomposition on LAPACK (`?geev`).
//!
//! Moved from tenferro-linalg's LAPACK backend (same project, MIT OR Apache-2.0). Outputs are always
//! in the complex counterpart of the input scalar: a real input's conjugate pairs are assembled from
//! `?geev`'s real/imaginary parts exactly as before the move, treating an imaginary part within
//! `ε · max(|re|, 1)` of zero as a real eigenvalue.

use strided_view::RawStridedRef;

use crate::batch::{batch_len, Input};
use crate::common::{
    check_info, check_query_info, check_scratch, checked_product, dim_i32, work_len,
};
use crate::{LapackScalar, Op, Result, Workspace};

fn eig_imag_is_effectively_zero(real: f64, imag: f64, eps: f64) -> bool {
    imag.abs() <= eps * real.abs().max(1.0)
}

/// Eigenvalues, and with `vectors` the right eigenvectors, of every square matrix of a batch.
///
/// `a` has dims `[n, n, batch...]` and is not modified. `values` is cleared and receives `n`
/// eigenvalues per matrix; when `vectors` is `Some`, it is cleared and receives the `n x n`
/// eigenvector matrices, compact column-major in batch order. Both are in the complex counterpart
/// of `T`. The destructible copy, the `?geev` value/vector buffers, the query slot, the real
/// `rwork` (complex scalars) and the work buffer are acquired from `workspace` once per call,
/// reused for every matrix and released on success.
///
/// # Errors
///
/// [`crate::Error::InvalidArgument`] for a malformed operand, a dimension outside the LAPACK `i32`
/// range or an illegal LAPACK argument, [`crate::Error::InvalidWorkspace`] for an unusable
/// workspace size, and [`crate::Error::NonConvergence`] when `?geev` fails to converge; always for
/// the first failing matrix in batch order. On error `values` and `vectors` are left empty.
pub fn eig<T, W>(
    op: Op,
    a: RawStridedRef<'_, T>,
    values: &mut Vec<T::Complex>,
    mut vectors: Option<&mut Vec<T::Complex>>,
    workspace: &mut W,
) -> Result<()>
where
    T: LapackScalar,
    W: Workspace<T> + Workspace<T::Real>,
{
    values.clear();
    if let Some(vectors) = vectors.as_deref_mut() {
        vectors.clear();
    }
    let result = run(op, a, values, vectors.as_deref_mut(), workspace);
    if result.is_err() {
        values.clear();
        if let Some(vectors) = vectors {
            vectors.clear();
        }
    }
    result
}

fn run<T, W>(
    op: Op,
    a: RawStridedRef<'_, T>,
    values: &mut Vec<T::Complex>,
    mut vectors: Option<&mut Vec<T::Complex>>,
    workspace: &mut W,
) -> Result<()>
where
    T: LapackScalar,
    W: Workspace<T> + Workspace<T::Real>,
{
    let input = Input::new(op, "input", a)?;
    let n = input.square(op)?;
    let matrix_len = checked_product(op, "matrix", &[n, n])?;
    let count = input.layout.count;
    if matrix_len == 0 || count == 0 {
        return Ok(());
    }
    let n_i32 = dim_i32(op, n)?;
    values.reserve(batch_len(op, n, count)?);
    if let Some(vectors) = vectors.as_deref_mut() {
        vectors.reserve(batch_len(op, matrix_len, count)?);
    }
    let (vector_len, ldvr, jobvr) = if vectors.is_some() {
        (matrix_len, n_i32, b'V')
    } else {
        (1, 1, b'N')
    };
    let mut a_copy: Vec<T> = workspace.acquire_capacity(matrix_len);
    // The real routine reports values as separate real and imaginary parts; in this crate's uniform
    // binding the real parts land in `w` (as `T`) and the imaginary parts in `wi`. The complex
    // routine reports `w` directly and ignores `wi`.
    let mut w: Vec<T> = workspace.acquire_zeroed(n);
    let mut wi: Vec<T::Real> = if T::COMPLEX {
        Vec::new()
    } else {
        workspace.acquire_zeroed(n)
    };
    let mut vl: Vec<T> = workspace.acquire_zeroed(1);
    let mut vr: Vec<T> = workspace.acquire_zeroed(vector_len);
    let mut query: Vec<T> = workspace.acquire_zeroed(1);
    let rwork_len = if T::COMPLEX {
        checked_product(op, "real workspace", &[2, n.max(1)])?
    } else {
        0
    };
    let mut rwork: Vec<T::Real> = if T::COMPLEX {
        workspace.acquire_zeroed(rwork_len)
    } else {
        Vec::new()
    };
    check_scratch(op, w.len(), n)?;
    check_scratch(op, wi.len(), if T::COMPLEX { 0 } else { n })?;
    check_scratch(op, vl.len(), 1)?;
    check_scratch(op, query.len(), 1)?;
    check_scratch(op, vr.len(), vector_len)?;
    check_scratch(op, rwork.len(), rwork_len)?;

    let mut work: Option<(i32, Vec<T>)> = None;
    for offset in input.layout.offsets() {
        a_copy.clear();
        input.gather(offset, &mut a_copy);
        if work.is_none() {
            let mut info = 0;
            // SAFETY: every buffer matches the validated `n x n` problem and `lwork = -1` makes
            // `query` the only workspace output.
            unsafe {
                T::geev(
                    jobvr,
                    n_i32,
                    &mut a_copy,
                    n_i32,
                    &mut w,
                    &mut wi,
                    &mut vl,
                    &mut vr,
                    ldvr,
                    &mut query,
                    -1,
                    &mut rwork,
                    &mut info,
                );
            }
            check_query_info(op, T::GEEV, info)?;
            let lwork = work_len(op, T::GEEV, T::work_query_len(query[0]))?;
            let buffer: Vec<T> = workspace.acquire_zeroed(lwork as usize);
            check_scratch(op, buffer.len(), lwork as usize)?;
            work = Some((lwork, buffer));
        }
        let Some((lwork, buffer)) = work.as_mut() else {
            unreachable!("the workspace was just queried");
        };
        let mut info = 0;
        // SAFETY: the buffers and leading dimensions match the validated problem and `buffer` has
        // the length the query on this shape returned.
        unsafe {
            T::geev(
                jobvr,
                n_i32,
                &mut a_copy,
                n_i32,
                &mut w,
                &mut wi,
                &mut vl,
                &mut vr,
                ldvr,
                buffer,
                *lwork,
                &mut rwork,
                &mut info,
            );
        }
        check_info(op, T::GEEV, info)?;
        if T::COMPLEX {
            values.extend(w.iter().map(|&value| value.to_complex()));
            if let Some(vectors) = vectors.as_deref_mut() {
                vectors.extend(vr.iter().map(|&value| value.to_complex()));
            }
        } else if vectors.is_some() {
            push_real_pairs::<T>(n, &w, &wi, Some(&vr[..]), values, vectors.as_deref_mut());
        } else {
            push_real_values::<T>(&w, &wi, values);
        }
    }
    workspace.release(a_copy);
    workspace.release(w);
    if !T::COMPLEX {
        workspace.release(wi);
    }
    workspace.release(vl);
    workspace.release(vr);
    workspace.release(query);
    if T::COMPLEX {
        workspace.release(rwork);
    }
    if let Some((_, buffer)) = work {
        workspace.release(buffer);
    }
    Ok(())
}

/// Assemble one real input's eigenvalues (and vectors) in the complex counterpart.
/// Push the real and imaginary parts LAPACK returned, unfiltered.
///
/// [`push_real_pairs`] folds an eigenvalue whose imaginary part is within `epsilon *
/// max(|re|, 1)` of zero, because its vector conversion has to decide whether a column is a real
/// eigenvector or the first of a conjugate pair. A values-only call wants no such decision, and the
/// faer provider reports the raw parts too, so this is what keeps the two providers — and each
/// provider's pre-extraction output — identical.
fn push_real_values<T: LapackScalar>(w: &[T], wi: &[T::Real], values: &mut Vec<T::Complex>) {
    values.extend(
        w.iter()
            .zip(wi)
            .map(|(value, imag)| T::complex_from_parts(value.real_part(), *imag)),
    );
}

fn push_real_pairs<T: LapackScalar>(
    n: usize,
    w: &[T],
    wi: &[T::Real],
    vr: Option<&[T]>,
    values: &mut Vec<T::Complex>,
    mut vectors: Option<&mut Vec<T::Complex>>,
) {
    let zero = <T::Real>::default();
    // Each column is assembled in this buffer and appended, in column order, so no entry is ever
    // written twice: the previous shape grew the output with `n * n` zeros and then overwrote
    // every one of them.
    let mut column: Vec<T::Complex> = Vec::with_capacity(n);
    let mut col = 0;
    while col < n {
        let re = w[col].real_part();
        if col + 1 >= n
            || eig_imag_is_effectively_zero(
                T::real_to_f64(re),
                T::real_to_f64(wi[col]),
                T::real_epsilon(),
            )
        {
            values.push(T::complex_from_parts(re, zero));
            if let (Some(vectors), Some(vr)) = (vectors.as_deref_mut(), vr) {
                for row in 0..n {
                    column.push(T::complex_from_parts(vr[row + col * n].real_part(), zero));
                }
                vectors.append(&mut column);
            }
            col += 1;
        } else {
            let im = wi[col];
            values.push(T::complex_from_parts(re, im));
            values.push(T::complex_from_parts(re, -im));
            if let (Some(vectors), Some(vr)) = (vectors.as_deref_mut(), vr) {
                for row in 0..n {
                    let first = vr[row + col * n].real_part();
                    let second = vr[row + (col + 1) * n].real_part();
                    column.push(T::complex_from_parts(first, second));
                }
                vectors.append(&mut column);
                for row in 0..n {
                    let first = vr[row + col * n].real_part();
                    let second = vr[row + (col + 1) * n].real_part();
                    column.push(T::complex_from_parts(first, -second));
                }
                vectors.append(&mut column);
            }
            col += 2;
        }
    }
}

#[cfg(test)]
mod tests {
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
}
