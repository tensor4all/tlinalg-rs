//! faer-backed batched singular value decomposition.
//!
//! Moved from `tenferro-linalg`'s faer backend (same project, MIT OR Apache-2.0).
//!
//! # Boundary
//!
//! The input is a borrowed rank-`2 + B` [`RawStridedRef`] `[m, n, b_1, ..., b_B]`; the outputs are
//! caller-provided buffers (the `*_into` routes) or vectors (the `Vec` wrappers), filled with
//! compact column-major items in batch order (see `docs/design/batched-api.md`). Faer's
//! `Mat`/`Diag` scratch is carved from the caller's borrowed workspace, sized once per lane and
//! reused for every item of the lane.
//!
//! # Conventions
//!
//! Per item, `u` is `m x u_cols` column-major, `vt` is `v_cols x n` column-major and holds `Vᴴ`
//! (not `V`), and `s` holds `min(m, n)` singular values in non-increasing order **in the scalar type
//! itself**. For the complex scalars that means complex values with a zero imaginary part, which is
//! what the pre-extraction code produced and what its callers' plumbing expects; [`svd_values`]
//! returns the real values instead, matching the values-only path. `full` selects the square
//! unitary factors `u_cols = m`, `v_cols = n`; otherwise `u_cols = v_cols = min(m, n)`. A full SVD
//! of an empty matrix returns identity `U` and `Vᴴ`.
//!
//! # Accuracy
//!
//! faer's default parameters use a divide-and-conquer bidiagonal SVD once the smaller dimension
//! reaches 128, and in faer 0.24.4 that path can return inaccurate factors without an error. A
//! decomposition with vectors of that size is therefore checked against its input, by the
//! Frobenius norm of `A - U diag(S) Vᴴ`, and repeated with the QR algorithm when the check fails
//! or faer reports an error; [`svd_values`] always uses the QR algorithm. `docs/worklogs/svd-recursion-threshold.md` records the cost.

use core::mem::MaybeUninit;

use faer::diag::{DiagMut, DiagRef};
use faer::dyn_stack::{MemStack, StackReq};
use faer::linalg::svd::{ComputeSvdVectors, SvdParams};
use faer::{MatMut, MatRef};

use strided_view::RawStridedRef;

use crate::batch::{self, fixed_out, BatchedRef, Push, Sink, VecOut};
use crate::scalar::ScalarEntity;
use crate::util::{checked_product, push_identity, weighted_muladds};
use crate::workspace::{bytes, faer_req, lane_requirement, total_of, zeros};
use crate::{Error, FaerScalar, Op, Result};
use tcontract_exec::Exec;

/// faer's parameters with the QR algorithm at every size.
///
/// faer's defaults switch from the QR algorithm to a divide-and-conquer bidiagonal SVD once the
/// smaller dimension reaches `recursion_threshold` (128). In faer 0.24.4 that path returns a
/// spurious singular value and inaccurate factors for some rank-deficient matrices with clustered
/// singular values, without reporting an error (see the test
/// `clustered_singular_values_of_a_rank_deficient_matrix`). These parameters never take it.
fn qr_params<E: faer::traits::ComplexField>() -> faer::Spec<SvdParams, E> {
    faer::Spec::new(SvdParams {
        recursion_threshold: usize::MAX,
        ..<SvdParams as faer::Auto<E>>::auto()
    })
}

/// Whether faer's default parameters use divide and conquer for a smaller dimension of `k`.
fn divides<E: faer::traits::ComplexField>(k: usize) -> bool {
    k >= <SvdParams as faer::Auto<E>>::auto().recursion_threshold
}

/// Constant of [`residual_bound`]. Accurate decompositions of sizes up to 1024 in all four scalar
/// types have relative residuals of 10 to 60 machine epsilons, at most a quarter of the bound at
/// size 128; the inaccurate ones observed exceed it by a factor of 1e8.
const RESIDUAL_FACTOR: f64 = 24.0;

/// Largest accepted relative Frobenius residual `‖A - U diag(S) Vᴴ‖ / ‖A‖` of an `m x n`
/// decomposition: `RESIDUAL_FACTOR * sqrt(max(m, n)) * epsilon`.
///
/// A singular value missing from an `n x n` identity leaves a relative residual of `1 / sqrt(n)`,
/// which this bound rejects for `n < 1 / (RESIDUAL_FACTOR * epsilon)`: about 350 000 in single
/// precision.
fn residual_bound(m: usize, n: usize, epsilon: f64) -> f64 {
    RESIDUAL_FACTOR * (m.max(n) as f64).sqrt() * epsilon
}

/// Storage for checking a decomposition with vectors against its input.
struct Check<'w, E: faer::traits::ComplexField> {
    /// `U` with its columns scaled by the singular values, `m x k`.
    scaled: MatMut<'w, E>,
    /// The reconstruction and then its difference from the input, `m x n`.
    residual: MatMut<'w, E>,
}

/// One lane's faer storage for `m x n` decompositions.
struct SvdScratch<'w, E: faer::traits::ComplexField> {
    v: MatMut<'w, E>,
    s: DiagMut<'w, E>,
    mem: &'w mut [MaybeUninit<u8>],
    /// Present when vectors are computed and the default parameters use divide and conquer.
    check: Option<Check<'w, E>>,
    /// Whether an earlier item has written these buffers, so the next one has to be reset first.
    reused: bool,
}

/// The column count of the `V` lane buffer for a vector mode.
fn v_cols(m: usize, n: usize, vectors: ComputeSvdVectors) -> usize {
    match vectors {
        ComputeSvdVectors::Full => n,
        ComputeSvdVectors::Thin => m.min(n),
        ComputeSvdVectors::No => 0,
    }
}

/// Whether a decomposition with this vector mode can take the divide-and-conquer path and is
/// therefore checked.
fn checked<E: faer::traits::ComplexField>(m: usize, n: usize, vectors: ComputeSvdVectors) -> bool {
    !matches!(vectors, ComputeSvdVectors::No) && divides::<E>(m.min(n))
}

/// faer's own SVD scratch: the QR parameters, or the larger of those and the default parameters
/// when the decomposition may run twice in the same buffer.
fn svd_mem_req<E: faer::traits::ComplexField>(
    m: usize,
    n: usize,
    vectors: ComputeSvdVectors,
    par: faer::ParShape,
) -> StackReq {
    if m.min(n) == 0 {
        // An empty matrix is never decomposed, so it needs no faer scratch.
        return StackReq::EMPTY;
    }
    let qr = faer::linalg::svd::svd_scratch::<E>(m, n, vectors, vectors, par, qr_params());
    if checked::<E>(m, n, vectors) {
        qr.or(faer::linalg::svd::svd_scratch::<E>(
            m,
            n,
            vectors,
            vectors,
            par,
            Default::default(),
        ))
    } else {
        qr
    }
}

/// One lane's complete requirement: `V`, the singular values, faer's scratch and (when checked) the
/// two reconstruction matrices.
fn svd_req<E: faer::traits::ComplexField>(
    m: usize,
    n: usize,
    vectors: ComputeSvdVectors,
    par: faer::ParShape,
) -> StackReq {
    let k = m.min(n);
    let req = StackReq::new::<E>(n * v_cols(m, n, vectors))
        .and(StackReq::new::<E>(k))
        .and(faer_req(svd_mem_req::<E>(m, n, vectors, par)));
    if checked::<E>(m, n, vectors) {
        req.and(StackReq::new::<E>(m * k))
            .and(StackReq::new::<E>(m * n))
    } else {
        req
    }
}

impl<'w, E: faer::traits::ComplexField> SvdScratch<'w, E> {
    fn new(
        m: usize,
        n: usize,
        vectors: ComputeSvdVectors,
        par: faer::ParShape,
        stack: &'w mut MemStack,
    ) -> Self {
        let k = m.min(n);
        let zero = <E as faer::traits::ComplexField>::zero_impl();
        let (v, stack) = zeros::<E>(stack, n * v_cols(m, n, vectors), &zero);
        let (s, stack) = zeros::<E>(stack, k, &zero);
        let (mem, stack) = bytes(
            stack,
            svd_mem_req::<E>(m, n, vectors, par).unaligned_bytes_required(),
        );
        let check = checked::<E>(m, n, vectors).then(|| {
            let (scaled, stack) = zeros::<E>(stack, m * k, &zero);
            let (residual, _) = zeros::<E>(stack, m * n, &zero);
            Check {
                scaled: MatMut::from_column_major_slice_mut(scaled, m, k),
                residual: MatMut::from_column_major_slice_mut(residual, m, n),
            }
        });
        Self {
            v: MatMut::from_column_major_slice_mut(v, n, v_cols(m, n, vectors)),
            s: DiagMut::from_slice_mut(s),
            mem,
            check,
            reused: false,
        }
    }
}

/// Whether the factors reproduce `mat`: the Frobenius norm of `mat - U diag(S) Vᴴ` is at most
/// [`residual_bound`] times that of `mat`. `u` and `v` hold at least `min(m, n)` columns. A
/// residual that is not finite is rejected.
///
/// The test is on the whole residual, so it holds for every input: factors that pass reproduce
/// the matrix to the bound. It costs one matrix product.
fn reproduces<E: faer::traits::ComplexField>(
    mat: MatRef<'_, E>,
    u: MatRef<'_, E>,
    s: DiagRef<'_, E>,
    v: MatRef<'_, E>,
    check: &mut Check<'_, E>,
    epsilon: f64,
    par: faer::Par<'_>,
) -> bool {
    use faer::traits::math_utils::{from_f64, mul, neg, one};

    let (m, n) = (mat.nrows(), mat.ncols());
    let k = m.min(n);
    for col in 0..k {
        let value = &s[col];
        for row in 0..m {
            check.scaled[(row, col)] = mul(&u[(row, col)], value);
        }
    }
    check.residual.as_mut().copy_from(mat);
    faer::linalg::matmul::matmul(
        check.residual.as_mut(),
        faer::Accum::Add,
        check.scaled.as_ref(),
        v.get(.., ..k).adjoint(),
        neg(&one::<E>()),
        par,
    );
    let bound = from_f64::<E::Real>(residual_bound(m, n, epsilon));
    // A NaN residual fails this comparison and is rejected with the large ones.
    check.residual.norm_l2() <= mul(&bound, &mat.norm_l2())
}

/// Singular values of one matrix, pushed in the real type.
fn svd_values_item<T: FaerScalar>(
    op: Op,
    mat: MatRef<'_, T::Entity>,
    s: &mut impl Push<<T as ScalarEntity>::Real>,
    scratch: &mut SvdScratch<'_, T::Entity>,
    par: faer::Par<'_>,
) -> Result<()> {
    let k = mat.nrows().min(mat.ncols());
    if k == 0 {
        return Ok(());
    }
    scratch
        .s
        .as_mut()
        .fill(<T::Entity as faer::traits::ComplexField>::zero_impl());
    faer::linalg::svd::svd(
        mat,
        scratch.s.as_mut(),
        None,
        None,
        par,
        MemStack::new(&mut *scratch.mem),
        // Without vectors there is no reconstruction to check, so the values never take the
        // divide-and-conquer path.
        qr_params(),
    )
    .map_err(|_| Error::NonConvergence { op })?;
    for index in 0..k {
        s.push(<T as ScalarEntity>::real_from_entity(scratch.s[index]));
    }
    Ok(())
}

/// SVD of one matrix, pushed as `U`, `S` and `Vᴴ`.
fn svd_item<T: FaerScalar>(
    op: Op,
    mat: MatRef<'_, T::Entity>,
    full: bool,
    (u, s, vt): (&mut Sink<'_, T>, &mut impl Push<T>, &mut impl Push<T>),
    scratch: &mut SvdScratch<'_, T::Entity>,
    mut par: faer::Par<'_>,
) -> Result<()> {
    let (m, n) = (mat.nrows(), mat.ncols());
    let k = m.min(n);
    if k == 0 {
        if full {
            push_identity(u, m);
            push_identity(vt, n);
        }
        return Ok(());
    }
    let (u_cols, v_cols) = if full { (m, n) } else { (k, k) };
    // `U` is computed in place in the caller's output. faer receives zeroed storage there and in
    // the lane buffers, so reuse cannot leak a previous item into this one.
    let zero = <T::Entity as faer::traits::ComplexField>::zero_impl();
    let u_region = T::entity_slice_mut(u.fill(m * u_cols, |_| T::default()));
    // The lane buffers are zeroed when the scratch is built, so the first item of that scratch is
    // only repeating the constructor's work; later items have to be reset.
    if scratch.reused {
        scratch.v.as_mut().fill(zero);
        scratch.s.as_mut().fill(zero);
    }
    scratch.reused = true;
    // The divide-and-conquer path of faer's default parameters can return inaccurate factors
    // without an error. A decomposition that can take it runs with the defaults first and is
    // repeated with the QR algorithm when faer reports an error or the factors do not reproduce
    // the input.
    let accurate = match scratch.check.as_mut() {
        None => false,
        Some(check) => {
            faer::linalg::svd::svd(
                mat,
                scratch.s.as_mut(),
                Some(MatMut::from_column_major_slice_mut(
                    &mut *u_region,
                    m,
                    u_cols,
                )),
                Some(scratch.v.as_mut()),
                par.reborrow(),
                MemStack::new(&mut *scratch.mem),
                Default::default(),
            )
            .is_ok()
                && reproduces(
                    mat,
                    MatRef::from_column_major_slice(&*u_region, m, u_cols),
                    scratch.s.as_ref(),
                    scratch.v.as_ref(),
                    check,
                    <T as ScalarEntity>::EPSILON,
                    par.reborrow(),
                )
        }
    };
    if !accurate {
        if scratch.check.is_some() {
            #[cfg(test)]
            tests::record_repeat();
            u_region.fill(zero);
            scratch.v.as_mut().fill(zero);
            scratch.s.as_mut().fill(zero);
        }
        faer::linalg::svd::svd(
            mat,
            scratch.s.as_mut(),
            Some(MatMut::from_column_major_slice_mut(
                &mut *u_region,
                m,
                u_cols,
            )),
            Some(scratch.v.as_mut()),
            par,
            MemStack::new(&mut *scratch.mem),
            qr_params(),
        )
        .map_err(|_| Error::NonConvergence { op })?;
    }

    for index in 0..k {
        // A real singular value carried in the scalar type: zero imaginary part for the complex
        // scalars, which is the shape the pre-extraction callers consumed.
        s.push(T::from_entity(scratch.s[index]));
    }
    for col in 0..n {
        for row in 0..v_cols {
            // `V` is transposed into `Vᴴ`, so the complex scalars conjugate here. The real ones are
            // their own conjugate, which is why the two implementations share this call.
            vt.push(T::from_entity_conj(scratch.v[(col, row)]));
        }
    }
    Ok(())
}

/// The vector mode and the output widths for `full`.
fn mode(m: usize, n: usize, full: bool) -> (usize, usize, ComputeSvdVectors) {
    if full {
        (m, n, ComputeSvdVectors::Full)
    } else {
        let k = m.min(n);
        (k, k, ComputeSvdVectors::Thin)
    }
}

fn work<T: ScalarEntity>(m: usize, n: usize) -> f64 {
    weighted_muladds::<T>(m as f64 * n as f64 * m.min(n) as f64)
}

/// Reject shapes whose products the lane requirement multiplies (faer's `svd_scratch` multiplies
/// the dimensions itself).
fn check_shape(op: Op, m: usize, n: usize, vectors: ComputeSvdVectors) -> Result<()> {
    checked_product(op, "matrix", &[m, n])?;
    let k = m.min(n);
    let cols = v_cols(m, n, vectors);
    checked_product(op, "workspace", &[n, cols])?;
    checked_product(op, "workspace", &[m, k])?;
    Ok(())
}

/// The output lengths of an `m x n` batch in `full` mode, with every product checked.
fn svd_shape<T: FaerScalar>(
    op: Op,
    m: usize,
    n: usize,
    full: bool,
) -> Result<(usize, usize, usize, f64)> {
    let (u_cols, v_cols, vectors) = mode(m, n, full);
    check_shape(op, m, n, vectors)?;
    let u_len = checked_product(op, "U", &[m, u_cols])?;
    let vt_len = checked_product(op, "VH", &[v_cols, n])?;
    Ok((u_len, vt_len, m.min(n), work::<T>(m, n)))
}

fn svd_values_bytes<T: FaerScalar>(
    op: Op,
    m: usize,
    n: usize,
    batch: usize,
    exec: &Exec<'_>,
) -> Result<usize> {
    let requirement = lane_requirement(
        op,
        exec,
        batch,
        work::<T>(m, n),
        || check_shape(op, m, n, ComputeSvdVectors::No),
        |par| svd_req::<T::Entity>(m, n, ComputeSvdVectors::No, par),
    )?;
    match requirement {
        None => Ok(0),
        Some((lanes, per_lane)) => total_of(op, lanes, per_lane),
    }
}

fn svd_bytes<T: FaerScalar>(
    op: Op,
    m: usize,
    n: usize,
    full: bool,
    batch: usize,
    exec: &Exec<'_>,
) -> Result<usize> {
    let vectors = mode(m, n, full).2;
    let requirement = lane_requirement(
        op,
        exec,
        batch,
        work::<T>(m, n),
        || check_shape(op, m, n, vectors),
        |par| svd_req::<T::Entity>(m, n, vectors, par),
    )?;
    match requirement {
        None => Ok(0),
        Some((lanes, per_lane)) => total_of(op, lanes, per_lane),
    }
}

/// Bytes one batched singular-values call needs in its borrowed workspace.
///
/// The canonical `Op` of a refusal is `Op::SvdValues`.
///
/// # Errors
///
/// [`Error::InvalidArgument`] when a size overflows.
pub fn svd_values_workspace<T: FaerScalar>(
    m: usize,
    n: usize,
    batch: usize,
    exec: &Exec<'_>,
) -> Result<usize> {
    svd_values_bytes::<T>(Op::SvdValues, m, n, batch, exec)
}

/// Bytes one batched SVD call needs in its borrowed workspace.
///
/// `full` selects the square unitary factors, exactly as [`svd`] does. The canonical `Op` of a
/// refusal is `Op::Svd`.
///
/// # Errors
///
/// [`Error::InvalidArgument`] when a size overflows.
pub fn svd_workspace<T: FaerScalar>(
    m: usize,
    n: usize,
    full: bool,
    batch: usize,
    exec: &Exec<'_>,
) -> Result<usize> {
    svd_bytes::<T>(Op::Svd, m, n, full, batch, exec)
}

/// Singular values of every matrix of a batch, without the vectors, into caller-owned storage.
///
/// `s` holds `min(m, n)` uninitialised real elements per batch item. Workspace contract as
/// [`crate::qr::qr_into`].
///
/// # Errors
///
/// [`Error::InvalidArgument`] when `input` has rank below 2 or an output length does not match,
/// [`crate::Error::InsufficientWorkspace`] for a short workspace, and [`Error::NonConvergence`] for
/// the lowest-indexed item faer fails to converge on.
pub fn svd_values_into<T: FaerScalar>(
    op: Op,
    input: RawStridedRef<'_, T>,
    s: &mut [MaybeUninit<<T as ScalarEntity>::Real>],
    workspace: &mut [MaybeUninit<u8>],
    exec: &mut Exec<'_>,
) -> Result<()> {
    let input = BatchedRef::new(op, "input", input)?;
    let (m, n) = (input.rows(), input.cols());
    check_shape(op, m, n, ComputeSvdVectors::No)?;
    let item_muladds = work::<T>(m, n);
    let requirement = lane_requirement(
        op,
        exec,
        input.batch(),
        item_muladds,
        || check_shape(op, m, n, ComputeSvdVectors::No),
        |par| svd_req::<T::Entity>(m, n, ComputeSvdVectors::No, par),
    )?;
    let (lanes, per_lane) = requirement.unwrap_or((1, 0));
    batch::check_workspace(op, workspace, per_lane, lanes)?;
    batch::run(
        op,
        input.batch(),
        exec,
        item_muladds,
        &mut (fixed_out(s, m.min(n), "singular values"),),
        batch::ScratchArena::new(workspace, per_lane),
        |par, stack| {
            SvdScratch::<T::Entity>::new(m, n, ComputeSvdVectors::No, par, MemStack::new(stack))
        },
        |index, (s,), scratch, par| svd_values_item::<T>(op, input.item(index), s, scratch, par),
    )
}

/// Singular values of every matrix of a batch, without the vectors.
///
/// `input` is `[m, n, b_1, ..., b_B]`; `s` receives `min(m, n)` real values per item.
///
/// This is the `Vec` wrapper of [`svd_values_into`].
///
/// # Errors
///
/// [`Error::InvalidArgument`] when `input` has rank below 2, and [`Error::NonConvergence`] for the
/// lowest-indexed item faer fails to converge on. `s` is empty on error.
///
/// # Examples
///
/// ```
/// use strided_view::RawStridedRef;
/// use tcontract_exec::Exec;
/// use tlinalg::{svd::svd_values, Op};
///
/// // Two 2x2 diagonal matrices, batch-contiguous.
/// let a = [3.0_f64, 0.0, 0.0, 1.0, 2.0, 0.0, 0.0, 5.0];
/// let mut s = Vec::new();
/// svd_values(
///     Op::SvdValues, RawStridedRef::new(&a, &[2, 2, 2], &[1, 2, 4], 0).unwrap(), &mut s,
///     &mut Exec::serial(),
/// ).unwrap();
/// assert_eq!(s, [3.0, 1.0, 5.0, 2.0]);
/// ```
pub fn svd_values<T: FaerScalar>(
    op: Op,
    input: RawStridedRef<'_, T>,
    s: &mut Vec<<T as ScalarEntity>::Real>,
    exec: &mut Exec<'_>,
) -> Result<()> {
    s.clear();
    let input = BatchedRef::new(op, "input", input)?;
    // The published order: operand validation before any workspace query or allocation.
    let (m, n) = (input.rows(), input.cols());
    check_shape(op, m, n, ComputeSvdVectors::No)?;
    let batch = input.batch();
    let bytes = svd_values_bytes::<T>(op, m, n, batch, exec)?;
    let mut workspace = vec![MaybeUninit::<u8>::uninit(); bytes];
    let mut s_out = VecOut::new(
        s,
        checked_product(op, "singular values", &[m.min(n), batch])?,
    );
    svd_values_into(op, input.raw(), s_out.spare(), &mut workspace, exec)?;
    s_out.publish();
    Ok(())
}

/// SVD of every matrix of a batch into caller-owned storage.
///
/// The output lengths follow the module conventions (`u`: `m * u_cols`, `s`: `min(m, n)`, `vt`:
/// `v_cols * n` per item), all uninitialised. Workspace contract as [`crate::qr::qr_into`].
///
/// # Errors
///
/// [`Error::InvalidArgument`] when `input` has rank below 2, an output length does not match, or a
/// size overflows, [`crate::Error::InsufficientWorkspace`] for a short workspace, and
/// [`Error::NonConvergence`] for the lowest-indexed item faer fails to converge on.
// INVARIANT: descriptor, mode, three output buffers, workspace, token are distinct operands of one
// batched decomposition; grouping them would add a wrapper without removing an argument.
#[allow(clippy::too_many_arguments)]
pub fn svd_into<T: FaerScalar>(
    op: Op,
    input: RawStridedRef<'_, T>,
    full: bool,
    u: &mut [MaybeUninit<T>],
    s: &mut [MaybeUninit<T>],
    vt: &mut [MaybeUninit<T>],
    workspace: &mut [MaybeUninit<u8>],
    exec: &mut Exec<'_>,
) -> Result<()> {
    let input = BatchedRef::new(op, "input", input)?;
    let (m, n) = (input.rows(), input.cols());
    let (u_len, vt_len, k, item_muladds) = svd_shape::<T>(op, m, n, full)?;
    let vectors = mode(m, n, full).2;
    let requirement = lane_requirement(
        op,
        exec,
        input.batch(),
        item_muladds,
        || check_shape(op, m, n, vectors),
        |par| svd_req::<T::Entity>(m, n, vectors, par),
    )?;
    let (lanes, per_lane) = requirement.unwrap_or((1, 0));
    batch::check_workspace(op, workspace, per_lane, lanes)?;
    batch::run(
        op,
        input.batch(),
        exec,
        item_muladds,
        &mut (
            fixed_out(u, u_len, "U"),
            fixed_out(s, k, "singular values"),
            fixed_out(vt, vt_len, "VH"),
        ),
        batch::ScratchArena::new(workspace, per_lane),
        |par, stack| SvdScratch::<T::Entity>::new(m, n, vectors, par, MemStack::new(stack)),
        |index, (u, s, vt), scratch, par| {
            svd_item::<T>(op, input.item(index), full, (u, s, vt), scratch, par)
        },
    )
}

/// Singular value decomposition of every matrix of a batch.
///
/// `input` is `[m, n, b_1, ..., b_B]`. Per item, `u` receives `m x u_cols`, `s` `min(m, n)` and
/// `vt` `v_cols x n` elements (see the module conventions).
///
/// This is the `Vec` wrapper of [`svd_into`].
///
/// # Errors
///
/// [`Error::InvalidArgument`] when `input` has rank below 2 or an output size overflows, and
/// [`Error::NonConvergence`] for the lowest-indexed item faer fails to converge on. The outputs are
/// empty on error.
///
/// # Examples
///
/// ```
/// use strided_view::RawStridedRef;
/// use tcontract_exec::Exec;
/// use tlinalg::{svd::svd, Op};
///
/// let a = [3.0_f64, 0.0, 0.0, 1.0];
/// let (mut u, mut s, mut vt) = (Vec::new(), Vec::new(), Vec::new());
/// svd(
///     Op::Svd, RawStridedRef::new(&a, &[2, 2], &[1, 2], 0).unwrap(), false,
///     &mut u, &mut s, &mut vt, &mut Exec::serial(),
/// ).unwrap();
/// assert_eq!(s, [3.0, 1.0]);
/// ```
// INVARIANT: descriptor, mode, three output buffers, token are distinct operands of one
// batched decomposition; grouping them would add a wrapper without removing an argument.
#[allow(clippy::too_many_arguments)]
pub fn svd<T: FaerScalar>(
    op: Op,
    input: RawStridedRef<'_, T>,
    full: bool,
    u: &mut Vec<T>,
    s: &mut Vec<T>,
    vt: &mut Vec<T>,
    exec: &mut Exec<'_>,
) -> Result<()> {
    u.clear();
    s.clear();
    vt.clear();
    // The published order: operand validation before any workspace query or allocation.
    let input = BatchedRef::new(op, "input", input)?;
    let (m, n) = (input.rows(), input.cols());
    let (u_len, vt_len, k, _) = svd_shape::<T>(op, m, n, full)?;
    let batch = input.batch();
    let bytes = svd_bytes::<T>(op, m, n, full, batch, exec)?;
    let mut workspace = vec![MaybeUninit::<u8>::uninit(); bytes];
    let mut u_out = VecOut::new(u, checked_product(op, "U", &[u_len, batch])?);
    let mut s_out = VecOut::new(s, checked_product(op, "S", &[k, batch])?);
    let mut vt_out = VecOut::new(vt, checked_product(op, "VH", &[vt_len, batch])?);
    svd_into(
        op,
        input.raw(),
        full,
        u_out.spare(),
        s_out.spare(),
        vt_out.spare(),
        &mut workspace,
        exec,
    )?;
    u_out.publish();
    s_out.publish();
    vt_out.publish();
    Ok(())
}

#[cfg(test)]
mod tests;
