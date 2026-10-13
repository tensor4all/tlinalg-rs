//! The size-query results: per batch item, the element count of every output, the batch count and
//! the workspace byte count.
//!
//! Every output is compact column-major and every item in a batch has the same shape (the batch
//! extents are shared by every operand), so one item count per output plus the batch count is the
//! whole answer. Total buffer elements are `item * batch`.

use crate::status::{shape, FfiError};

/// `tlinalg_svd_out`: the `svd` query result.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct tlinalg_svd_out {
    /// Number of batch items.
    pub batch: usize,
    /// Elements per item of `u` (`rows * u_cols`).
    pub u: usize,
    /// Elements per item of `s` (`min(rows, cols)`).
    pub s: usize,
    /// Elements per item of `vt` (`v_cols * cols`).
    pub vt: usize,
    /// Workspace bytes. Zero in v1.
    pub workspace: usize,
}

/// `tlinalg_svd_values_out`: the `svd_values` query result.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct tlinalg_svd_values_out {
    /// Number of batch items.
    pub batch: usize,
    /// Elements per item of `s` (`min(rows, cols)`).
    pub s: usize,
    /// Workspace bytes. Zero in v1.
    pub workspace: usize,
}

/// `tlinalg_eigh_out`: the `eigh` query result.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct tlinalg_eigh_out {
    /// Number of batch items.
    pub batch: usize,
    /// Elements per item of `values` (`n`).
    pub values: usize,
    /// Elements per item of `vectors` (`n * n`).
    pub vectors: usize,
    /// Workspace bytes. Zero in v1.
    pub workspace: usize,
}

/// `tlinalg_eigh_values_out`: the `eigh_values` query result.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct tlinalg_eigh_values_out {
    /// Number of batch items.
    pub batch: usize,
    /// Elements per item of `values` (`n`).
    pub values: usize,
    /// Workspace bytes. Zero in v1.
    pub workspace: usize,
}

/// `tlinalg_cholesky_out`: the `cholesky` query result.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct tlinalg_cholesky_out {
    /// Number of batch items.
    pub batch: usize,
    /// Elements per item of `l` (`n * n`).
    pub l: usize,
    /// Workspace bytes. Zero in v1.
    pub workspace: usize,
}

/// `tlinalg_lu_out`: the `lu` query result.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct tlinalg_lu_out {
    /// Number of batch items.
    pub batch: usize,
    /// Elements per item of `p` (`rows * rows`).
    pub p: usize,
    /// Elements per item of `l` (`rows * min(rows, cols)`).
    pub l: usize,
    /// Elements per item of `u` (`min(rows, cols) * cols`).
    pub u: usize,
    /// Elements per item of `parity` (always 1).
    pub parity: usize,
    /// Workspace bytes. Zero in v1.
    pub workspace: usize,
}

/// `tlinalg_full_piv_lu_out`: the `full_piv_lu` query result.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct tlinalg_full_piv_lu_out {
    /// Number of batch items.
    pub batch: usize,
    /// Elements per item of `p` (`n * n`).
    pub p: usize,
    /// Elements per item of `l` (`n * n`).
    pub l: usize,
    /// Elements per item of `u` (`n * n`).
    pub u: usize,
    /// Elements per item of `q` (`n * n`).
    pub q: usize,
    /// Elements per item of `parity` (always 1).
    pub parity: usize,
    /// Workspace bytes. Zero in v1.
    pub workspace: usize,
}

/// The batch count of a rank-`2 + B` descriptor: the product of the trailing extents.
pub(crate) fn batch_of(dims: &[usize]) -> Result<usize, FfiError> {
    dims.get(2..)
        .unwrap_or(&[])
        .iter()
        .try_fold(1usize, |acc, &dim| {
            acc.checked_mul(dim)
                .ok_or_else(|| shape("batch count overflows usize"))
        })
}

/// The checked product of `parts`, or a shape error on overflow.
pub(crate) fn product(parts: &[usize]) -> Result<usize, FfiError> {
    parts.iter().try_fold(1usize, |acc, &part| {
        acc.checked_mul(part)
            .ok_or_else(|| shape("output element count overflows usize"))
    })
}

impl tlinalg_svd_out {
    /// The total element counts of `u`, `s` and `vt` (`item * batch`).
    pub(crate) fn totals(&self) -> Result<(usize, usize, usize), FfiError> {
        Ok((
            product(&[self.u, self.batch])?,
            product(&[self.s, self.batch])?,
            product(&[self.vt, self.batch])?,
        ))
    }
}

impl tlinalg_svd_values_out {
    /// The total element count of `s` (`item * batch`).
    pub(crate) fn total(&self) -> Result<usize, FfiError> {
        product(&[self.s, self.batch])
    }
}

impl tlinalg_eigh_out {
    /// The total element counts of `values` and `vectors` (`item * batch`).
    pub(crate) fn totals(&self) -> Result<(usize, usize), FfiError> {
        Ok((
            product(&[self.values, self.batch])?,
            product(&[self.vectors, self.batch])?,
        ))
    }
}

impl tlinalg_eigh_values_out {
    /// The total element count of `values` (`item * batch`).
    pub(crate) fn total(&self) -> Result<usize, FfiError> {
        product(&[self.values, self.batch])
    }
}

impl tlinalg_cholesky_out {
    /// The total element count of `l` (`item * batch`).
    pub(crate) fn total(&self) -> Result<usize, FfiError> {
        product(&[self.l, self.batch])
    }
}

impl tlinalg_lu_out {
    /// The total element counts of `p`, `l`, `u` and `parity` (`item * batch`).
    pub(crate) fn totals(&self) -> Result<(usize, usize, usize, usize), FfiError> {
        Ok((
            product(&[self.p, self.batch])?,
            product(&[self.l, self.batch])?,
            product(&[self.u, self.batch])?,
            product(&[self.parity, self.batch])?,
        ))
    }
}

impl tlinalg_full_piv_lu_out {
    /// The total element counts of `p`, `l`, `u`, `q` and `parity` (`item * batch`).
    pub(crate) fn totals(&self) -> Result<(usize, usize, usize, usize, usize), FfiError> {
        Ok((
            product(&[self.p, self.batch])?,
            product(&[self.l, self.batch])?,
            product(&[self.u, self.batch])?,
            product(&[self.q, self.batch])?,
            product(&[self.parity, self.batch])?,
        ))
    }
}

/// The side of a square descriptor, or a shape error. The square-only operations share this so the
/// size query and the writing call reject a non-square input the same way.
pub(crate) fn require_square(dims: &[usize]) -> Result<usize, FfiError> {
    if dims[0] != dims[1] {
        return Err(shape(format!(
            "input is {}x{}, expected a square matrix",
            dims[0], dims[1]
        )));
    }
    Ok(dims[0])
}
