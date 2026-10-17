//! Batch iteration over a rank `2 + B` strided operand.
//!
//! Every public entry point takes its matrices as one [`RawStridedRef`] (or [`RawStridedMut`]) with
//! dims `[rows, cols, b_1, ..., b_B]`: matrix dims first, batch dims after, the first batch dim
//! varying fastest (tenferro's column-major convention). `B = 0` is a single matrix. Strides are
//! arbitrary but non-negative; the descriptor itself was bounds-checked when it was constructed.
//!
//! Outputs are compact and batch-contiguous: item `i` occupies `[i * item_len, (i + 1) * item_len)`.

use strided_view::{RawStridedMut, RawStridedRef};

use crate::{Error, Op, Result};

/// Normalised batch axes kept inline; only a view with more non-mergeable axes spills to the heap.
const INLINE_AXES: usize = 8;

/// A list of up to `N` elements stored inline, spilling to the heap only beyond `N`.
///
/// The driver keeps normalised batch metadata inline up to `N` axes, so a descriptor with up to
/// `N` non-mergeable batch axes allocates nothing; a view with more spills its axis list to the
/// heap once per layout. That bound is the allocation claim: arbitrary-rank layout independence is
/// allocation-free *within the inline axis capacity*, and a wider rank allocates metadata
/// independently of the numerical workspace. `N = 8` covers every compact or mildly gapped batch;
/// a rank that spills is still supported, just not allocation-free.
#[derive(Clone, Debug)]
pub(crate) struct SmallList<T: Copy + Default, const N: usize> {
    inline: [T; N],
    len: usize,
    /// Holds every element once `len` exceeds `N`; empty (no allocation) until then.
    spill: Vec<T>,
}

impl<T: Copy + Default, const N: usize> SmallList<T, N> {
    pub(crate) fn new() -> Self {
        Self {
            inline: [T::default(); N],
            len: 0,
            spill: Vec::new(),
        }
    }

    pub(crate) fn push(&mut self, value: T) {
        if self.len < N {
            self.inline[self.len] = value;
        } else {
            if self.len == N {
                self.spill.extend_from_slice(&self.inline);
            }
            self.spill.push(value);
        }
        self.len += 1;
    }

    pub(crate) fn len(&self) -> usize {
        self.len
    }

    pub(crate) fn as_slice(&self) -> &[T] {
        if self.len <= N {
            &self.inline[..self.len]
        } else {
            &self.spill
        }
    }

    pub(crate) fn as_mut_slice(&mut self) -> &mut [T] {
        if self.len <= N {
            &mut self.inline[..self.len]
        } else {
            &mut self.spill
        }
    }

    pub(crate) fn iter(&self) -> core::slice::Iter<'_, T> {
        self.as_slice().iter()
    }

    pub(crate) fn last_mut(&mut self) -> Option<&mut T> {
        self.as_mut_slice().last_mut()
    }
}

/// The layout of one batched operand, without its data.
///
/// The batch axes are kept as given for shape comparisons and **normalised** for iteration: size-1
/// axes are dropped and adjacent axes with `stride[i + 1] == stride[i] * dim[i]` are coalesced, so a
/// compact batch walks one axis with a plain stride step and only a genuinely non-mergeable view pays
/// for the multi-index odometer. A stride of 0 on a batch axis is a broadcast and is allowed on
/// inputs; outputs reject it (see [`Output::new`]).
#[derive(Clone, Debug)]
pub(crate) struct Layout<'a> {
    pub(crate) rows: usize,
    pub(crate) cols: usize,
    pub(crate) row_stride: usize,
    pub(crate) col_stride: usize,
    batch_dims: &'a [usize],
    /// Normalised batch axes `(dim, stride)` for iteration.
    axes: SmallList<(usize, usize), INLINE_AXES>,
    offset: usize,
    /// The number of matrices.
    pub(crate) count: usize,
}

impl<'a> Layout<'a> {
    fn new(
        op: Op,
        role: &'static str,
        dims: &'a [usize],
        strides: &'a [isize],
        offset: isize,
    ) -> Result<Self> {
        if dims.len() < 2 || strides.len() != dims.len() {
            return Err(Error::InvalidArgument {
                op,
                role,
                detail: format!(
                    "expected a rank 2 + B operand [rows, cols, batch...], got dims {dims:?}"
                ),
            });
        }
        if strides.iter().any(|&stride| stride < 0) || offset < 0 {
            return Err(Error::InvalidArgument {
                op,
                role,
                detail: "a negative stride or offset is not a supported layout".to_owned(),
            });
        }
        let count = dims[2..]
            .iter()
            .try_fold(1usize, |acc, &dim| acc.checked_mul(dim))
            .ok_or_else(|| Error::InvalidArgument {
                op,
                role: "shape",
                detail: "batch element count overflow".to_owned(),
            })?;
        let mut axes = SmallList::<(usize, usize), INLINE_AXES>::new();
        for (&dim, &stride) in dims[2..].iter().zip(&strides[2..]) {
            if dim == 1 {
                continue;
            }
            let stride = stride as usize;
            match axes.last_mut() {
                Some((last_dim, last_stride))
                    if last_stride.checked_mul(*last_dim) == Some(stride) =>
                {
                    *last_dim *= dim;
                }
                _ => axes.push((dim, stride)),
            }
        }
        Ok(Self {
            rows: dims[0],
            cols: dims[1],
            row_stride: strides[0] as usize,
            col_stride: strides[1] as usize,
            batch_dims: &dims[2..],
            axes,
            offset: offset as usize,
            count,
        })
    }

    /// Whether the matrix has no element (so no item touches memory).
    pub(crate) fn is_empty_matrix(&self) -> bool {
        self.rows == 0 || self.cols == 0
    }

    /// The element offset of every item, in batch order.
    pub(crate) fn offsets(&self) -> Offsets {
        let mut index = SmallList::new();
        for _ in 0..self.axes.len() {
            index.push(0);
        }
        Offsets {
            index,
            axes: self.axes.clone(),
            current: self.offset,
            remaining: self.count,
        }
    }

    /// The leading dimension LAPACK can read this layout with in place, when it has one: unit row
    /// stride (or a single row) and a column stride of at least `max(rows, 1)`.
    pub(crate) fn lapack_lda(&self) -> Option<usize> {
        let lda = if self.cols <= 1 {
            self.rows.max(1)
        } else {
            self.col_stride
        };
        let unit_rows = self.rows <= 1 || self.row_stride == 1;
        (unit_rows && lda >= self.rows.max(1)).then_some(lda)
    }

    /// The number of elements an in-place LAPACK read with leading dimension `lda` spans.
    pub(crate) fn span(&self, lda: usize) -> usize {
        if self.is_empty_matrix() {
            0
        } else {
            (self.cols - 1) * lda + self.rows
        }
    }

    /// Reject a writable layout whose items (or whose elements within one item) could alias.
    ///
    /// Writing through an aliased item would make the result depend on the item order (and, for a
    /// parallel host, race), so a mutable descriptor must place every element of every item at a
    /// distinct address. Checked conservatively: sort the non-trivial axes (the two matrix axes and
    /// the normalised batch axes) by stride and require each stride to exceed the extent of every
    /// smaller one.
    pub(crate) fn reject_self_overlap(&self, op: Op, role: &'static str) -> Result<()> {
        if self.count == 0 || self.is_empty_matrix() {
            return Ok(());
        }
        let mut axes = SmallList::<(usize, usize), { INLINE_AXES + 2 }>::new();
        for axis in [(self.rows, self.row_stride), (self.cols, self.col_stride)]
            .into_iter()
            .chain(self.axes.iter().copied())
            .filter(|&(dim, _)| dim > 1)
        {
            axes.push(axis);
        }
        // Unstable: a stable sort may allocate a merge buffer.
        axes.as_mut_slice()
            .sort_unstable_by_key(|&(_, stride)| stride);
        let mut extent = 0usize;
        for &(dim, stride) in axes.iter() {
            if stride == 0 || stride <= extent {
                return Err(Error::InvalidArgument {
                    op,
                    role,
                    detail: "a writable operand must not alias its own elements or batch items"
                        .to_owned(),
                });
            }
            extent += stride * (dim - 1);
        }
        Ok(())
    }

    /// Require the same batch dims as `other`.
    pub(crate) fn same_batch(&self, op: Op, other: &Layout<'_>) -> Result<()> {
        if self.batch_dims != other.batch_dims {
            return Err(Error::InvalidArgument {
                op,
                role: "batch",
                detail: format!(
                    "batch dims {:?} and {:?} differ",
                    self.batch_dims, other.batch_dims
                ),
            });
        }
        Ok(())
    }
}

/// Item offsets in batch order (first batch dim fastest).
pub(crate) struct Offsets {
    axes: SmallList<(usize, usize), INLINE_AXES>,
    index: SmallList<usize, INLINE_AXES>,
    current: usize,
    remaining: usize,
}

impl Iterator for Offsets {
    type Item = usize;

    fn next(&mut self) -> Option<usize> {
        if self.remaining == 0 {
            return None;
        }
        self.remaining -= 1;
        let out = self.current;
        if self.remaining > 0 {
            let index = self.index.as_mut_slice();
            for (axis, &(dim, stride)) in self.axes.iter().enumerate() {
                index[axis] += 1;
                if index[axis] < dim {
                    self.current += stride;
                    break;
                }
                self.current -= stride * (index[axis] - 1);
                index[axis] = 0;
            }
        }
        Some(out)
    }
}

/// A batched read-only operand.
#[derive(Clone)]
pub(crate) struct Input<'a, T> {
    data: &'a [T],
    pub(crate) layout: Layout<'a>,
}

impl<'a, T: Copy> Input<'a, T> {
    pub(crate) fn new(op: Op, role: &'static str, a: RawStridedRef<'a, T>) -> Result<Self> {
        Ok(Self {
            data: a.data(),
            layout: Layout::new(op, role, a.dims(), a.strides(), a.offset())?,
        })
    }

    /// Require a square matrix and return its order.
    pub(crate) fn square(&self, op: Op) -> Result<usize> {
        if self.layout.rows != self.layout.cols {
            return Err(Error::InvalidArgument {
                op,
                role: "shape",
                detail: format!(
                    "expected a square matrix, got {}x{}",
                    self.layout.rows, self.layout.cols
                ),
            });
        }
        Ok(self.layout.rows)
    }

    /// Append the item at `offset` to `dst` as a compact column-major matrix.
    pub(crate) fn gather(&self, offset: usize, dst: &mut Vec<T>) {
        let Layout {
            rows,
            cols,
            row_stride,
            col_stride,
            ..
        } = self.layout;
        // INVARIANT: the descriptor was bounds-checked at construction, so every reachable offset
        // `offset + i * row_stride + j * col_stride` is inside `data`.
        if row_stride == 1 {
            for col in 0..cols {
                let start = offset + col * col_stride;
                dst.extend_from_slice(&self.data[start..start + rows]);
            }
        } else {
            for col in 0..cols {
                let start = offset + col * col_stride;
                dst.extend((0..rows).map(|row| self.data[start + row * row_stride]));
            }
        }
    }

    /// Copy column `col` of the item at `offset` into `dst` (`rows` elements).
    pub(crate) fn copy_column(&self, offset: usize, col: usize, dst: &mut [T]) {
        let start = offset + col * self.layout.col_stride;
        let row_stride = self.layout.row_stride;
        // INVARIANT: as in `gather`.
        for (row, slot) in dst.iter_mut().enumerate().take(self.layout.rows) {
            *slot = self.data[start + row * row_stride];
        }
    }

    /// Append the transpose of the item at `offset` to `dst` as a compact column-major matrix.
    pub(crate) fn gather_transposed(&self, offset: usize, dst: &mut Vec<T>) {
        let Layout {
            rows,
            cols,
            row_stride,
            col_stride,
            ..
        } = self.layout;
        // INVARIANT: as in `gather`; column `j` of the transpose is row `j` of the item.
        for row in 0..rows {
            let start = offset + row * row_stride;
            dst.extend((0..cols).map(|col| self.data[start + col * col_stride]));
        }
    }

    /// The item at `offset` as a LAPACK-readable slice and leading dimension, when the layout
    /// allows reading it in place.
    pub(crate) fn in_place(&self, offset: usize) -> Option<(&'a [T], usize)> {
        let lda = self.layout.lapack_lda()?;
        let span = self.layout.span(lda);
        if span == 0 {
            return Some((&[], lda));
        }
        Some((&self.data[offset..offset + span], lda))
    }
}

/// A batched writable operand.
pub(crate) struct Output<'a, 'b, T> {
    data: &'b mut [T],
    pub(crate) layout: Layout<'a>,
}

impl<'a, 'b, T> Output<'a, 'b, T> {
    pub(crate) fn new(
        op: Op,
        role: &'static str,
        out: &'b mut RawStridedMut<'a, T>,
    ) -> Result<Self> {
        let (dims, strides, offset) = (out.dims(), out.strides(), out.offset());
        let layout = Layout::new(op, role, dims, strides, offset)?;
        layout.reject_self_overlap(op, role)?;
        Ok(Self {
            layout,
            data: out.data_mut(),
        })
    }

    /// The item at `offset` as a mutable slice spanning `lda`-strided columns.
    pub(crate) fn item_mut(&mut self, offset: usize, lda: usize) -> &mut [T] {
        let span = self.layout.span(lda);
        if span == 0 {
            return &mut [];
        }
        &mut self.data[offset..offset + span]
    }
}

/// Clear every output on failure, so an error never leaves a partially filled batch behind.
pub(crate) fn clear_on_error<R>(result: Result<R>, clear: impl FnOnce()) -> Result<R> {
    if result.is_err() {
        clear();
    }
    result
}

/// `rows * cols * count`, checked.
pub(crate) fn batch_len(op: Op, item_len: usize, count: usize) -> Result<usize> {
    item_len
        .checked_mul(count)
        .ok_or_else(|| Error::InvalidArgument {
            op,
            role: "shape",
            detail: "batched output element count overflow".to_owned(),
        })
}
