//! Single-matrix (`B = 0`) wrappers over the batched entry points, with the per-matrix argument
//! order the family tests were written against. Each builds the rank-2 descriptors and calls the
//! batched API on one sequential lane, so the family tests exercise exactly the public surface.

#![allow(dead_code, clippy::too_many_arguments)]

use strided_view::{RawStridedMut, RawStridedRef};
use tcontract_exec::Exec;
use tlinalg::householder::ReflectorShape;
use tlinalg::triangular_solve::TriangularSolveFlags;
use tlinalg::{Error, FaerScalar, Op, Result};

fn length_error(op: Op) -> Error {
    Error::InvalidArgument {
        op,
        role: "configuration",
        detail: "buffer length does not match its shape".into(),
    }
}

pub fn cholesky<T: FaerScalar>(
    op: Op,
    _n: usize,
    input: RawStridedRef<'_, T>,
    l: &mut Vec<T>,
    par: &mut Exec<'_>,
) -> Result<()> {
    tlinalg::cholesky::cholesky(op, input, l, par)
}

pub fn triangular_solve<T: FaerScalar, W: ?Sized>(
    op: Op,
    _n: usize,
    a: RawStridedRef<'_, T>,
    rhs: Vec<T>,
    b_rows: usize,
    b_cols: usize,
    flags: TriangularSolveFlags,
    _workspace: &mut W,
    par: &mut Exec<'_>,
) -> Result<Vec<T>> {
    if b_rows.checked_mul(b_cols) != Some(rhs.len()) {
        return Err(length_error(op));
    }
    let dims = [b_rows, b_cols];
    let strides = [1, b_rows as isize];
    let b = RawStridedRef::new(&rhs, &dims, &strides, 0).unwrap();
    let mut x = Vec::new();
    tlinalg::triangular_solve::triangular_solve(op, a, b, flags, &mut x, par)?;
    Ok(x)
}

pub struct LuFactors<'a, T> {
    pub p: &'a mut Vec<T>,
    pub l: &'a mut Vec<T>,
    pub u: &'a mut Vec<T>,
}

pub fn lu<T: FaerScalar>(
    op: Op,
    _m: usize,
    _n: usize,
    input: RawStridedRef<'_, T>,
    factors: LuFactors<'_, T>,
    par: &mut Exec<'_>,
) -> Result<T> {
    let mut parity = Vec::new();
    tlinalg::lu::lu(
        op,
        input,
        tlinalg::lu::LuFactors {
            p: factors.p,
            l: factors.l,
            u: factors.u,
            parity: &mut parity,
        },
        par,
    )?;
    Ok(parity[0])
}

pub fn solve<T: FaerScalar>(
    op: Op,
    _n: usize,
    _nrhs: usize,
    a: RawStridedRef<'_, T>,
    rhs: Option<RawStridedRef<'_, T>>,
    out: RawStridedMut<'_, T>,
    transpose_a: bool,
    par: &mut Exec<'_>,
) -> Result<()> {
    tlinalg::lu::solve(op, a, rhs, out, transpose_a, par)
}

pub struct FullPivLuFactors<'a, T> {
    pub p: &'a mut Vec<T>,
    pub l: &'a mut Vec<T>,
    pub u: &'a mut Vec<T>,
    pub q: &'a mut Vec<T>,
}

pub fn full_piv_lu<T: FaerScalar>(
    op: Op,
    _n: usize,
    input: RawStridedRef<'_, T>,
    factors: FullPivLuFactors<'_, T>,
    par: &mut Exec<'_>,
) -> Result<T> {
    let mut parity = Vec::new();
    tlinalg::full_piv_lu::full_piv_lu(
        op,
        input,
        tlinalg::full_piv_lu::FullPivLuFactors {
            p: factors.p,
            l: factors.l,
            u: factors.u,
            q: factors.q,
            parity: &mut parity,
        },
        par,
    )?;
    Ok(parity[0])
}

/// In place on `rhs`, as the per-matrix API was; `rhs` is untouched on error.
pub fn full_piv_lu_solve<T: FaerScalar>(
    op: Op,
    n: usize,
    nrhs: usize,
    a: RawStridedRef<'_, T>,
    rhs: &mut [T],
    transpose_a: bool,
    par: &mut Exec<'_>,
) -> Result<()> {
    if n.checked_mul(nrhs) != Some(rhs.len()) {
        return Err(length_error(op));
    }
    let dims = [n, nrhs];
    let strides = [1, n as isize];
    let mut x = Vec::new();
    tlinalg::full_piv_lu::full_piv_lu_solve(
        op,
        a,
        RawStridedRef::new(rhs, &dims, &strides, 0).unwrap(),
        transpose_a,
        &mut x,
        par,
    )?;
    rhs.copy_from_slice(&x);
    Ok(())
}

pub fn qr<T: FaerScalar>(
    op: Op,
    _m: usize,
    _n: usize,
    input: RawStridedRef<'_, T>,
    q: &mut Vec<T>,
    r: &mut Vec<T>,
    par: &mut Exec<'_>,
) -> Result<()> {
    tlinalg::qr::qr(op, input, q, r, par)
}

pub fn rank_revealing_qr<T: FaerScalar>(
    op: Op,
    _m: usize,
    _n: usize,
    input: RawStridedRef<'_, T>,
    q: &mut Vec<T>,
    r: &mut Vec<T>,
    par: &mut Exec<'_>,
) -> Result<Vec<i64>> {
    let mut permutation = Vec::new();
    tlinalg::qr::rank_revealing_qr(op, input, q, r, &mut permutation, par)?;
    Ok(permutation)
}

pub fn eigh<T: FaerScalar>(
    op: Op,
    _n: usize,
    input: RawStridedRef<'_, T>,
    values: &mut Vec<T>,
    vectors: &mut Vec<T>,
    par: &mut Exec<'_>,
) -> Result<()> {
    tlinalg::eigh::eigh(op, input, values, vectors, par)
}

pub fn eigh_values<T: FaerScalar>(
    op: Op,
    _n: usize,
    input: RawStridedRef<'_, T>,
    values: &mut Vec<T::Real>,
    par: &mut Exec<'_>,
) -> Result<()> {
    tlinalg::eigh::eigh_values(op, input, values, par)
}

pub fn eig<T: FaerScalar>(
    op: Op,
    _n: usize,
    input: RawStridedRef<'_, T>,
    values: &mut Vec<T::Complex>,
    vectors: &mut Vec<T::Complex>,
    par: &mut Exec<'_>,
) -> Result<()> {
    tlinalg::eig::eig(op, input, values, vectors, par)
}

pub fn eig_values<T: FaerScalar>(
    op: Op,
    _n: usize,
    input: RawStridedRef<'_, T>,
    values: &mut Vec<T::Complex>,
    par: &mut Exec<'_>,
) -> Result<()> {
    tlinalg::eig::eig_values(op, input, values, par)
}

pub fn compact_factor<T: FaerScalar>(
    op: Op,
    data: &mut [T],
    rows: usize,
    cols: usize,
    coeff: &mut Vec<T>,
    par: &mut Exec<'_>,
) -> Result<()> {
    if rows.checked_mul(cols) != Some(data.len()) {
        return Err(length_error(op));
    }
    tlinalg::householder::compact_factor(op, rows, cols, 1, data, coeff, par)
}

pub fn apply_reflectors<T: FaerScalar>(
    op: Op,
    a: &[T],
    a_cols: usize,
    coeff: &[T],
    c: &mut [T],
    rows: usize,
    cols: usize,
    k: usize,
    transpose: bool,
    par: &mut Exec<'_>,
) -> Result<()> {
    if coeff.len() != k {
        return Err(length_error(op));
    }
    let shape = ReflectorShape {
        rows,
        a_cols,
        cols,
        k,
    };
    tlinalg::householder::apply_reflectors(op, shape, 1, a, coeff, c, transpose, par)
}
