//! The shared batch driver: strided batch iteration, lane fan-out, and output assembly.
//!
//! Every public entry point of this crate is batched (see `docs/design/batched-api.md`). This module
//! is the one place that
//!
//! * walks the batch dimensions of a rank-`2 + B` strided descriptor ([`BatchedRef`],
//!   [`BatchedMut`]),
//! * selects Auto lanes within the requested width and runs them on the caller's pool ([`run`]),
//! * hands every batch item its own disjoint output view: spare capacity of a library-owned vector
//!   ([`VecOut`]), a caller-owned uninitialised slice ([`FixedOut`]), or a caller-owned compact
//!   in-place buffer ([`InPlace`]),
//! * carves each lane's scratch out of the caller's borrowed workspace ([`ScratchArena`]).
//!
//! # Output soundness
//!
//! Every output is addressed **per item** (`item_len` elements per batch item), and each lane writes
//! the items the driver gives it strictly sequentially through a [`Sink`] — or, for an in-place
//! buffer, straight into that item's slice. So "the sink reached the end of the item's chunk" is
//! exactly "every element of the item was written", whether that chunk is a caller-owned buffer or
//! spare capacity. The driver checks it per item; for a library-owned vector it calls `set_len` only
//! when **every** item was full, so a caller never observes a partially initialised vector.

use core::mem::MaybeUninit;
use std::sync::Mutex;

use faer::{MatMut, MatRef};
use strided_view::{RawStridedMut, RawStridedRef};
use tcontract_exec::batch::{BatchQuery, BatchStrategy};
use tcontract_exec::{Exec, ExecError};

use crate::scalar::ScalarEntity;
use crate::util::{checked_product, invalid};
use crate::{Error, Op, Result};

/// The batch count of a descriptor's trailing dimensions.
fn batch_count(op: Op, batch_dims: &[usize]) -> Result<usize> {
    checked_product(op, "batch", batch_dims)
}

/// Batch axes after normalisation: extent-one axes dropped and mergeable neighbours coalesced.
///
/// Adjacent axes `i`, `i + 1` (first batch axis fastest) merge when `stride[i + 1] == stride[i] *
/// dim[i]`, so a compact batch — the common case — collapses to one axis and an item's offset is a
/// single multiply. Only a genuinely non-mergeable view keeps several axes. A stride-0 axis (host
/// broadcasting) is kept like any other.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct BatchAxes {
    axes: AxisList,
}

/// Normalised axes kept inline up to [`INLINE_AXES`], on the heap only beyond that, so building a
/// descriptor allocates nothing for any realistic batch rank.
const INLINE_AXES: usize = 8;

#[derive(Clone, Debug, PartialEq, Eq)]
enum AxisList {
    Inline {
        len: usize,
        axes: [(usize, isize); INLINE_AXES],
    },
    Heap(Vec<(usize, isize)>),
}

impl AxisList {
    fn new() -> Self {
        Self::Inline {
            len: 0,
            axes: [(0, 0); INLINE_AXES],
        }
    }

    fn as_slice(&self) -> &[(usize, isize)] {
        match self {
            Self::Inline { len, axes } => &axes[..*len],
            Self::Heap(axes) => axes,
        }
    }

    fn as_mut_slice(&mut self) -> &mut [(usize, isize)] {
        match self {
            Self::Inline { len, axes } => &mut axes[..*len],
            Self::Heap(axes) => axes,
        }
    }

    fn last_mut(&mut self) -> Option<&mut (usize, isize)> {
        match self {
            Self::Inline { len, axes } => axes[..*len].last_mut(),
            Self::Heap(axes) => axes.last_mut(),
        }
    }

    fn push(&mut self, axis: (usize, isize)) {
        match self {
            Self::Inline { len, axes } if *len < INLINE_AXES => {
                axes[*len] = axis;
                *len += 1;
            }
            Self::Inline { axes, .. } => {
                let mut heap = axes.to_vec();
                heap.push(axis);
                *self = Self::Heap(heap);
            }
            Self::Heap(axes) => axes.push(axis),
        }
    }

    #[cfg(test)]
    fn len(&self) -> usize {
        self.as_slice().len()
    }
}

impl BatchAxes {
    pub(crate) fn new(dims: &[usize], strides: &[isize]) -> Self {
        let mut axes = AxisList::new();
        for (&dim, &stride) in dims.iter().zip(strides) {
            if dim == 1 {
                continue;
            }
            match axes.last_mut() {
                Some((last_dim, last_stride))
                    if last_stride.checked_mul(*last_dim as isize) == Some(stride) =>
                {
                    *last_dim *= dim;
                }
                _ => axes.push((dim, stride)),
            }
        }
        Self { axes }
    }

    /// Element offset of batch item `index`.
    ///
    /// One axis (the compact case) is a single multiply. Several axes — a non-mergeable view —
    /// decompose `index` once per item; that is `O(B)` integer work against an `O(n³)` kernel, and
    /// it keeps each item independent of the order lanes visit them.
    pub(crate) fn offset(&self, index: usize) -> isize {
        match self.axes.as_slice() {
            [] => 0,
            [(_, stride)] => index as isize * stride,
            axes => {
                let mut rest = index;
                let mut offset = 0isize;
                for &(dim, stride) in axes {
                    offset += (rest % dim) as isize * stride;
                    rest /= dim;
                }
                offset
            }
        }
    }

    #[cfg(test)]
    fn len(&self) -> usize {
        self.axes.len()
    }

    #[cfg(test)]
    fn is_inline(&self) -> bool {
        matches!(self.axes, AxisList::Inline { .. })
    }
}

/// A borrowed rank-`2 + B` input: `[rows, cols, b_1, ..., b_B]`.
#[derive(Clone)]
pub(crate) struct BatchedRef<'a, T> {
    input: RawStridedRef<'a, T>,
    axes: BatchAxes,
    rows: usize,
    cols: usize,
    batch: usize,
}

impl<'a, T: ScalarEntity> BatchedRef<'a, T> {
    /// Validate the rank of `input` and read its matrix and batch shape.
    pub(crate) fn new(op: Op, role: &'static str, input: RawStridedRef<'a, T>) -> Result<Self> {
        let dims = input.dims();
        if dims.len() < 2 {
            return Err(invalid(
                op,
                "configuration",
                format!("{role} has rank {}, expected at least 2", dims.len()),
            ));
        }
        Ok(Self {
            rows: dims[0],
            cols: dims[1],
            batch: batch_count(op, &dims[2..])?,
            axes: BatchAxes::new(&dims[2..], &input.strides()[2..]),
            input,
        })
    }

    /// Validate as [`BatchedRef::new`] and require a square matrix shape.
    pub(crate) fn square(op: Op, role: &'static str, input: RawStridedRef<'a, T>) -> Result<Self> {
        let this = Self::new(op, role, input)?;
        if this.rows != this.cols {
            return Err(invalid(
                op,
                "configuration",
                format!(
                    "{role} is {}x{}, expected a square matrix",
                    this.rows, this.cols
                ),
            ));
        }
        Ok(this)
    }

    pub(crate) fn rows(&self) -> usize {
        self.rows
    }

    pub(crate) fn cols(&self) -> usize {
        self.cols
    }

    pub(crate) fn batch(&self) -> usize {
        self.batch
    }

    pub(crate) fn batch_dims(&self) -> &'a [usize] {
        &self.input.dims()[2..]
    }

    /// The descriptor this view was built from.
    pub(crate) fn raw(&self) -> RawStridedRef<'a, T> {
        self.input
    }

    /// Batch item `index` as a faer matrix over the scalar's faer entity.
    ///
    /// `index` must be below [`BatchedRef::batch`].
    pub(crate) fn item(&self, index: usize) -> MatRef<'a, T::Entity> {
        debug_assert!(index < self.batch);
        let strides = self.input.strides();
        let offset = self.axes.offset(index);
        // SAFETY: `RawStridedRef::new` validated that every offset reachable from `ptr()` through
        // `dims`/`strides` lies inside the borrowed data, for the borrow `'a`. Batch item `index`
        // (below the batch count) is the sub-rectangle at the batch offset computed above, so every
        // element faer reads is one of those validated offsets. `wrapping_offset` keeps the
        // pointer's provenance; see `item_ptr` for the empty case. The cast is the layout-preserving
        // one asserted in `crate::scalar`.
        unsafe {
            MatRef::from_raw_parts(
                item_ptr(self.input.ptr(), offset, self.rows, self.cols).cast::<T::Entity>(),
                self.rows,
                self.cols,
                strides[0],
                strides[1],
            )
        }
    }
}

/// Reject a second operand whose batch shape differs from the first's.
pub(crate) fn same_batch(
    op: Op,
    role: &'static str,
    expected: &[usize],
    got: &[usize],
) -> Result<()> {
    if expected != got {
        return Err(invalid(
            op,
            "configuration",
            format!("{role} has batch shape {got:?}, expected {expected:?}"),
        ));
    }
    Ok(())
}

/// The pointer faer receives for one batch item.
///
/// An item with a zero dimension has no elements, so its batch offset carries no meaning and is not
/// applied. `strided-view` accepts an empty layout without validating offsets and then hands back
/// its `dangling()` pointer, which is aligned but names no allocation: adding a large or negative
/// batch stride to it can wrap it to null, and faer stores the pointer it is given in a `NonNull`
/// built with `new_unchecked` (`faer::MatRef::from_raw_parts`). The descriptor's own pointer is
/// non-null and aligned, so an empty item keeps it and nothing is dereferenced either way.
#[inline]
fn item_ptr<T>(base: *const T, offset: isize, rows: usize, cols: usize) -> *const T {
    if rows == 0 || cols == 0 {
        base
    } else {
        base.wrapping_offset(offset)
    }
}

/// Whether a strided layout maps distinct indices to distinct offsets.
///
/// Sufficient test: after dropping extent-one dimensions and sorting by absolute stride, each stride
/// exceeds the span of every smaller dimension. A layout that fails it may still be injective, but
/// the hosts this crate serves never build one, so rejecting it costs nothing.
fn is_injective(dims: &[usize], strides: &[isize]) -> bool {
    if dims.contains(&0) {
        return true;
    }
    // Inline (no allocation for up to eight non-trivial axes) and sorted in place: unstable, since
    // a stable sort may allocate a merge buffer.
    let mut axes = AxisList::new();
    for (&dim, &stride) in dims.iter().zip(strides) {
        if dim > 1 {
            axes.push((dim, stride));
        }
    }
    let axes = axes.as_mut_slice();
    axes.sort_unstable_by_key(|&(dim, stride)| (stride.unsigned_abs(), dim));
    let mut span = 0usize;
    for &(dim, stride) in axes.iter() {
        let stride = stride.unsigned_abs();
        if stride <= span {
            return false;
        }
        span = match (dim - 1)
            .checked_mul(stride)
            .and_then(|s| s.checked_add(span))
        {
            Some(span) => span,
            None => return false,
        };
    }
    true
}

/// A raw pointer that lanes may share because they touch disjoint elements through it.
pub(crate) struct SharedPtr<T>(*mut T);

impl<T> Clone for SharedPtr<T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T> Copy for SharedPtr<T> {}

// SAFETY: the pointer is only dereferenced through `BatchedMut::item`, whose contract gives each
// batch item to one lane at a time, and an injective layout makes distinct items disjoint.
unsafe impl<T: Send> Send for SharedPtr<T> {}
// SAFETY: as above.
unsafe impl<T: Send> Sync for SharedPtr<T> {}

/// A borrowed, mutable rank-`2 + B` destination with an injective layout.
pub(crate) struct BatchedMut<'a, T> {
    base: SharedPtr<T>,
    dims: &'a [usize],
    strides: &'a [isize],
    axes: BatchAxes,
    batch: usize,
}

impl<'a, T: ScalarEntity> BatchedMut<'a, T> {
    /// Validate the rank and injectivity of `output` and take exclusive access to it.
    pub(crate) fn new(
        op: Op,
        role: &'static str,
        output: &'a mut RawStridedMut<'_, T>,
    ) -> Result<Self> {
        let dims = output.dims();
        let strides = output.strides();
        if dims.len() < 2 {
            return Err(invalid(
                op,
                "configuration",
                format!("{role} has rank {}, expected at least 2", dims.len()),
            ));
        }
        if !is_injective(dims, strides) {
            return Err(invalid(
                op,
                "configuration",
                format!("{role} layout aliases two elements"),
            ));
        }
        Ok(Self {
            batch: batch_count(op, &dims[2..])?,
            base: SharedPtr(output.as_mut_ptr()),
            axes: BatchAxes::new(&dims[2..], &strides[2..]),
            dims,
            strides,
        })
    }

    pub(crate) fn rows(&self) -> usize {
        self.dims[0]
    }

    pub(crate) fn cols(&self) -> usize {
        self.dims[1]
    }

    pub(crate) fn batch_dims(&self) -> &'a [usize] {
        &self.dims[2..]
    }

    /// Batch item `index` as a mutable faer matrix.
    ///
    /// # Safety
    ///
    /// `index` is below the batch count, and no other live reference (from this or any other call)
    /// covers item `index` for as long as the returned matrix is used.
    pub(crate) unsafe fn item(&self, index: usize) -> MatMut<'_, T::Entity> {
        debug_assert!(index < self.batch);
        let offset = self.axes.offset(index);
        // SAFETY: `RawStridedMut::new` validated every reachable offset against the exclusively
        // borrowed data; item `index` is a sub-rectangle of those offsets, and the injective layout
        // (checked in `new`) makes it disjoint from every other item. The caller guarantees no other
        // reference covers it. See `item_ptr` for the empty case.
        unsafe {
            MatMut::from_raw_parts_mut(
                item_ptr(self.base.0, offset, self.dims[0], self.dims[1])
                    .cast_mut()
                    .cast::<T::Entity>(),
                self.dims[0],
                self.dims[1],
                self.strides[0],
                self.strides[1],
            )
        }
    }
}

/// Sequential writer over one item's uninitialised output chunk.
pub(crate) struct Sink<'a, U> {
    buf: &'a mut [MaybeUninit<U>],
    pos: usize,
}

impl<U> Sink<'_, U> {
    /// Write the next element. Panics (safely) if the kernel writes past its chunk.
    pub(crate) fn push(&mut self, value: U) {
        self.buf[self.pos].write(value);
        self.pos += 1;
    }

    /// Write the next `len` elements as `value(0), ..., value(len - 1)` and return them, now
    /// initialised, for the kernel to update in place. Panics (safely) past the chunk end.
    pub(crate) fn fill(&mut self, len: usize, mut value: impl FnMut(usize) -> U) -> &mut [U] {
        let start = self.pos;
        let region = &mut self.buf[start..start + len];
        for (index, slot) in region.iter_mut().enumerate() {
            slot.write(value(index));
        }
        self.pos += len;
        // SAFETY: every element of `region` was written just above, so the `MaybeUninit<U>`
        // slice is fully initialised; `MaybeUninit<U>` has the layout of `U`, and the returned
        // borrow keeps `self` (and so the chunk) exclusively borrowed while it lives.
        unsafe { &mut *(region as *mut [MaybeUninit<U>] as *mut [U]) }
    }

    fn is_full(&self) -> bool {
        self.pos == self.buf.len()
    }
}

/// Something a per-item kernel can append output elements to, the item it writes into.
///
/// A trait rather than the concrete [`Sink`] so `EigScalar::eig_into` can take a `&mut dyn Push` and
/// erase the lane's lifetime. `pub` only so the crate-internal `EigScalar` can name it; this module
/// is private.
pub trait Push<U> {
    /// Append one element.
    fn push(&mut self, value: U);
}

impl<U> Push<U> for Sink<'_, U> {
    fn push(&mut self, value: U) {
        Sink::push(self, value);
    }
}

/// Lane `lane`'s half-open item range when `batch` items are split into `lanes` nonempty
/// contiguous ranges. `lanes <= batch`, so every range is nonempty. Computed from the quotient and
/// remainder so `lane * batch` cannot overflow for a huge valid batch.
fn lane_bounds(batch: usize, lanes: usize, lane: usize) -> (usize, usize) {
    let base = batch / lanes;
    let extra = batch % lanes;
    let first = lane * base + lane.min(extra);
    let last = first + base + usize::from(lane < extra);
    (first, last)
}

/// The outputs of one batched call, handed out per batch item.
///
/// An item's view is derived from its index on demand ([`Outputs::item`]), so the driver builds no
/// per-lane list: a single-lane run allocates nothing, and a multi-lane run creates each view inside
/// its own task. Items are addressed by index, so a lane touches only the indices the driver gives
/// it.
pub(crate) trait Outputs {
    /// Raw handles to every output, shared by the lane tasks.
    type Parts: Sync;
    /// One batch item's view of every output.
    type Item: Send;

    /// Clear and reserve the vector outputs, check the caller-owned ones.
    fn prepare(&mut self, op: Op, batch: usize) -> Result<()>;

    /// The raw handles the items are carved from.
    ///
    /// # Safety
    ///
    /// [`Outputs::prepare`] succeeded for this `batch`; `self` is not used again until every item
    /// derived from the returned handles has been dropped.
    unsafe fn parts(&mut self, batch: usize) -> Self::Parts;

    /// Item `index`'s view.
    ///
    /// # Safety
    ///
    /// `parts` came from [`Outputs::parts`] for this `batch`, `index` is below it, and no two live
    /// views are given the same `index` (the driver partitions `0..batch` into disjoint ranges).
    unsafe fn item(parts: &Self::Parts, index: usize) -> Self::Item;

    /// Whether a kernel wrote every element of the item it was given.
    fn item_complete(item: &Self::Item) -> bool;

    /// Publish the written vector outputs.
    ///
    /// # Safety
    ///
    /// Every index `0..batch` was created once, completed (see [`Outputs::item_complete`]) and has
    /// been dropped.
    unsafe fn commit(&mut self, batch: usize);
}

/// A caller-owned compact buffer of `item_len` elements per batch item, updated in place.
pub(crate) struct InPlace<'a, U> {
    slice: &'a mut [U],
    item_len: usize,
    role: &'static str,
}

/// A compact in-place buffer of `item_len` elements per batch item.
pub(crate) fn in_place<'a, U>(
    slice: &'a mut [U],
    item_len: usize,
    role: &'static str,
) -> InPlace<'a, U> {
    InPlace {
        slice,
        item_len,
        role,
    }
}

impl<'a, U: Send> Outputs for InPlace<'a, U> {
    type Parts = (SharedPtr<U>, usize);
    type Item = &'a mut [U];

    fn prepare(&mut self, op: Op, batch: usize) -> Result<()> {
        if self.slice.len() != checked_product(op, self.role, &[self.item_len, batch])? {
            return Err(Error::Inconsistent {
                op,
                detail: "batch buffers describe different batches",
            });
        }
        Ok(())
    }

    unsafe fn parts(&mut self, _: usize) -> Self::Parts {
        (SharedPtr(self.slice.as_mut_ptr()), self.item_len)
    }

    unsafe fn item(parts: &Self::Parts, index: usize) -> &'a mut [U] {
        let (ptr, item_len) = *parts;
        // SAFETY: the slice is exclusively borrowed for `'a` and holds `batch * item_len` elements,
        // `index * item_len..` lies inside it, it is not touched until the items are gone, and
        // distinct indices are disjoint.
        unsafe { core::slice::from_raw_parts_mut(ptr.0.add(index * item_len), item_len) }
    }

    fn item_complete(_: &&'a mut [U]) -> bool {
        true
    }

    unsafe fn commit(&mut self, _: usize) {}
}

/// A caller-owned buffer of `item_len` elements per batch item, written from uninitialised storage.
///
/// This is the output side of the `*_into` entry points: the caller owns the storage, the kernel
/// initialises every element, and a kernel that writes fewer elements than the item holds is a
/// detected bug rather than uninitialised memory handed back to the caller.
pub(crate) struct FixedOut<'a, U> {
    slice: &'a mut [MaybeUninit<U>],
    item_len: usize,
    role: &'static str,
}

/// A caller-owned output buffer of `item_len` elements per batch item.
pub(crate) fn fixed_out<'a, U>(
    slice: &'a mut [MaybeUninit<U>],
    item_len: usize,
    role: &'static str,
) -> FixedOut<'a, U> {
    FixedOut {
        slice,
        item_len,
        role,
    }
}

impl<'a, U: Send> Outputs for FixedOut<'a, U> {
    type Parts = (SharedPtr<MaybeUninit<U>>, usize);
    type Item = Sink<'a, U>;

    fn prepare(&mut self, op: Op, batch: usize) -> Result<()> {
        let expected = checked_product(op, self.role, &[self.item_len, batch])?;
        if self.slice.len() != expected {
            return Err(invalid(
                op,
                "output",
                format!(
                    "{role} has {} elements, expected {expected} ({item_len} x {batch})",
                    self.slice.len(),
                    role = self.role,
                    item_len = self.item_len,
                ),
            ));
        }
        Ok(())
    }

    unsafe fn parts(&mut self, _: usize) -> Self::Parts {
        (SharedPtr(self.slice.as_mut_ptr()), self.item_len)
    }

    unsafe fn item(parts: &Self::Parts, index: usize) -> Sink<'a, U> {
        let (ptr, item_len) = *parts;
        // SAFETY: as for `Out`: the caller's slice holds `batch * item_len` elements and distinct
        // indices cover disjoint ranges.
        let buf = unsafe { core::slice::from_raw_parts_mut(ptr.0.add(index * item_len), item_len) };
        Sink { buf, pos: 0 }
    }

    fn item_complete(item: &Sink<'a, U>) -> bool {
        item.is_full()
    }

    unsafe fn commit(&mut self, _: usize) {}
}

/// A caller-owned `Vec` output whose spare capacity a `*_into` call writes into.
///
/// The vector is cleared and reserved here; the call writes exactly `len` elements into the spare
/// capacity and [`VecOut::publish`] sets the length. An error leaves the vector at length zero, so a
/// wrapper keeps the "empty on error" contract without a second pass over the data.
pub(crate) struct VecOut<'v, U> {
    vec: &'v mut Vec<U>,
    len: usize,
}

impl<'v, U> VecOut<'v, U> {
    /// Clear `vec` and reserve room for `len` elements.
    pub(crate) fn new(vec: &'v mut Vec<U>, len: usize) -> Self {
        vec.clear();
        vec.reserve(len);
        Self { vec, len }
    }

    /// The `len` spare elements the call writes.
    pub(crate) fn spare(&mut self) -> &mut [MaybeUninit<U>] {
        &mut self.vec.spare_capacity_mut()[..self.len]
    }

    /// Publish the written elements.
    pub(crate) fn publish(self) {
        // SAFETY: the matching `*_into` call wrote every one of the `len` elements (the driver
        // rejects an item that did not fill its output), and `reserve` gave the capacity.
        unsafe { self.vec.set_len(self.len) }
    }
}

/// No outputs beyond what the item closure captures itself (a [`BatchedMut`] destination).
impl Outputs for () {
    type Parts = ();
    type Item = ();

    fn prepare(&mut self, _: Op, _: usize) -> Result<()> {
        Ok(())
    }

    unsafe fn parts(&mut self, _: usize) {}

    unsafe fn item(_: &(), _: usize) {}

    fn item_complete(_: &()) -> bool {
        true
    }

    unsafe fn commit(&mut self, _: usize) {}
}

macro_rules! impl_outputs_tuple {
    ($($name:ident $index:tt),+) => {
        impl<$($name: Outputs),+> Outputs for ($($name,)+) {
            type Parts = ($($name::Parts,)+);
            type Item = ($($name::Item,)+);

            fn prepare(&mut self, op: Op, batch: usize) -> Result<()> {
                $(self.$index.prepare(op, batch)?;)+
                Ok(())
            }

            unsafe fn parts(&mut self, batch: usize) -> Self::Parts {
                // SAFETY: forwarded; the caller's guarantee covers every component.
                ($(unsafe { self.$index.parts(batch) },)+)
            }

            unsafe fn item(parts: &Self::Parts, index: usize) -> Self::Item {
                // SAFETY: forwarded; the caller's guarantee covers every component.
                ($(unsafe { $name::item(&parts.$index, index) },)+)
            }

            fn item_complete(item: &Self::Item) -> bool {
                true $(&& $name::item_complete(&item.$index))+
            }

            unsafe fn commit(&mut self, batch: usize) {
                // SAFETY: forwarded; the caller's guarantee covers every component.
                $(unsafe { self.$index.commit(batch) };)+
            }
        }
    };
}

impl_outputs_tuple!(A 0);
impl_outputs_tuple!(A 0, B 1);
impl_outputs_tuple!(A 0, B 1, C 2);
impl_outputs_tuple!(A 0, B 1, C 2, D 3);
impl_outputs_tuple!(A 0, B 1, C 2, D 3, E 4);

/// The per-lane chunks of a caller-owned workspace, handed to the lane tasks.
///
/// Lane `l` owns exactly `[l * per_lane, (l + 1) * per_lane)`. `per_lane` includes the worst-case
/// alignment padding of one lane's requirement, so a sub-buffer carved out of any lane chunk is
/// correctly aligned whatever the caller buffer's own alignment is.
pub(crate) struct ScratchArena<'w> {
    base: *mut MaybeUninit<u8>,
    per_lane: usize,
    _borrow: core::marker::PhantomData<&'w mut [MaybeUninit<u8>]>,
}

// SAFETY: lanes touch disjoint byte ranges, and [`ScratchArena::lane`] hands each index out once.
unsafe impl Sync for ScratchArena<'_> {}

impl<'w> ScratchArena<'w> {
    /// Partition `workspace` into equal lane chunks of `per_lane` bytes.
    pub(crate) fn new(workspace: &'w mut [MaybeUninit<u8>], per_lane: usize) -> Self {
        Self {
            base: workspace.as_mut_ptr(),
            per_lane,
            _borrow: core::marker::PhantomData,
        }
    }

    /// Lane `lane`'s chunk.
    ///
    /// # Safety
    ///
    /// No two live chunks are requested for the same `lane`, and the chunk does not outlive the
    /// workspace it was carved from.
    pub(crate) unsafe fn lane(&self, lane: usize) -> &'w mut [MaybeUninit<u8>] {
        // SAFETY: the caller guarantees a distinct lane index per live chunk and `new` sized the
        // arena from a workspace of at least `lanes * per_lane` bytes, so this range is inside the
        // exclusively borrowed workspace.
        unsafe {
            core::slice::from_raw_parts_mut(self.base.add(lane * self.per_lane), self.per_lane)
        }
    }
}

/// Check that `workspace` covers the requirement of a `lanes`-lane route, or report the typed
/// refusal the borrowed-workspace contract specifies.
pub(crate) fn check_workspace(
    op: Op,
    workspace: &[MaybeUninit<u8>],
    per_lane: usize,
    lanes: usize,
) -> Result<()> {
    let needed = per_lane
        .checked_mul(lanes)
        .ok_or_else(|| invalid(op, "workspace", "the workspace requirement overflows usize"))?;
    if workspace.len() < needed {
        return Err(Error::InsufficientWorkspace {
            op,
            needed,
            have: workspace.len(),
        });
    }
    Ok(())
}

/// The pre-0.3 Auto cutoff: ordinary families fan out on a pool only while the largest item
/// extent is at most this. Packed LU keeps its no-cutoff exception (`None`).
///
/// tlinalg-rs#58 / tcontract-rs#123 step 1 keep this local Auto in 0.3.0; the switch to the shared
/// work model (with the sweep as evidence) lands in step 3.
const AUTO_FAN_OUT_MAX_ITEM_DIM: usize = 64;

/// The legacy Auto item measure: `order`, or `order.max(rhs)` when the RHS/reflector target extent
/// is counted.
///
/// This is the temporary step-1 switch of tlinalg-rs#58 "Defaults". The default includes the RHS
/// extent, matching 0.2; set `TLINALG_AUTO_RHS=exclude` in the environment to measure the
/// RHS-excluded variant. The knob and the cutoff are removed when #123 step 3 adopts the shared
/// work model.
pub(crate) fn auto_extent_with(order: usize, rhs: usize, include_rhs: bool) -> usize {
    if include_rhs {
        order.max(rhs)
    } else {
        order
    }
}

/// [`auto_extent_with`] with the process's `TLINALG_AUTO_RHS` switch (default include).
pub(crate) fn auto_extent(order: usize, rhs: usize) -> usize {
    static INCLUDES_RHS: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    let include = *INCLUDES_RHS.get_or_init(|| {
        !matches!(
            std::env::var("TLINALG_AUTO_RHS").ok().as_deref(),
            Some("exclude")
        )
    });
    auto_extent_with(order, rhs, include)
}

/// The step-1 Auto decision: the 0.2 `LanePlan` rule on the context's budget.
///
/// On a pool with more than one worker, fan out to `min(budget, batch)` outer lanes only when the
/// batch fills the budget and `max_item_dim <= AUTO_FAN_OUT_MAX_ITEM_DIM`. `None` is the packed
/// exception (never cut off). Otherwise one lane runs the batch at the shared work model's width
/// for one item; a small item therefore stays on the caller instead of entering the pool with a
/// thread count it cannot use. The partition is exact (see [`run`]); 0.2's `div_ceil` split could
/// drop a lane, which this keeps fixed.
fn legacy_auto(
    batch: usize,
    budget: usize,
    max_item_dim: Option<usize>,
    item_threads: usize,
) -> (usize, usize) {
    let budget = budget.max(1);
    let fans_out = budget > 1
        && batch >= budget
        && max_item_dim.is_none_or(|dim| dim <= AUTO_FAN_OUT_MAX_ITEM_DIM);
    if fans_out {
        (budget.min(batch), 1)
    } else {
        (1, item_threads.clamp(1, budget))
    }
}

/// The one-lane item width the shared work model grants an item of `item_muladds` work.
///
/// An `Auto` decision for one item is the generic formula's one-lane arm: the lanes stay at one and
/// `item_threads` is `Exec::width_for` sized by the model's item cost. Feeding the model one item
/// keeps the conversion from multiply-adds to nanoseconds in `tcontract_exec` instead of copying it
/// here. `Auto` never refuses, but map any executor error through the same typed path as the forced
/// routes.
fn auto_item_threads(op: Op, exec: Exec<'_>, item_muladds: f64) -> Result<usize> {
    let query = BatchQuery::new(1, item_muladds);
    match exec.batch_decision(&query) {
        Ok(decision) => Ok(decision.item_threads),
        Err(error) => Err(strategy_error(op, error)),
    }
}

/// The batch route for this call: `(lanes, item_threads)`.
///
/// `Auto` keeps [`legacy_auto`]'s fan-out rule (the pre-0.3 local cutoff, tlinalg-rs#58 "Defaults",
/// step 1) and sizes its one lane from `item_muladds` through [`auto_item_threads`]. Every forced
/// strategy goes through `tcontract_exec::Exec::batch_decision` on the original budget.
/// `max_item_dim` is the legacy Auto fan-out measure.
pub(crate) fn decide(
    op: Op,
    exec: Exec<'_>,
    batch: usize,
    max_item_dim: Option<usize>,
    item_muladds: f64,
) -> Result<(usize, usize)> {
    if matches!(exec.batch_policy().strategy, BatchStrategy::Auto) {
        let item_threads = auto_item_threads(op, exec, item_muladds)?;
        return Ok(legacy_auto(
            batch,
            exec.budget(),
            max_item_dim,
            item_threads,
        ));
    }
    let query = BatchQuery::new(batch, item_muladds);
    match exec.batch_decision(&query) {
        Ok(decision) => Ok((decision.lanes, decision.item_threads)),
        Err(error) => Err(strategy_error(op, error)),
    }
}

/// The faer parallelism token a resolved route runs its items with.
///
/// This is exactly what [`run`] passes to the item closure and to `make_scratch`: one lane stays on
/// the caller at width one (`faer::Par::Seq`), otherwise the item width the decision granted is
/// installed on the pool. The workspace queries size their faer scratch with it, so a query and the
/// matching call resolve the same width.
pub(crate) fn route_par(lanes: usize, item_threads: usize) -> faer::Par {
    if lanes > 1 || item_threads > 1 {
        faer::Par::rayon(item_threads.max(1))
    } else {
        faer::Par::Seq
    }
}

/// The one mapping from the executor's forced-route refusal to this crate's typed error.
fn strategy_error(op: Op, error: ExecError) -> Error {
    match error {
        ExecError::StrategyUnavailable { strategy } => Error::StrategyUnavailable { op, strategy },
        other => Error::InvalidArgument {
            op,
            role: "executor",
            detail: other.to_string(),
        },
    }
}

/// Run the batch on one caller-owned resource.
///
/// The route comes from [`decide`]: `lanes` nonempty contiguous ranges of the batch, each item run
/// with `item_threads` (one lane may run items on the caller with a faer count hint; several lanes
/// run concurrently on the context's pool). Forced strategies are resolved before any write; a
/// `WholeBatchVendor` request on this route is refused with [`Error::StrategyUnavailable`].
///
/// `max_item_dim` is the legacy Auto fan-out measure (see [`decide`]); `item_muladds` is the
/// family's work estimate, used by the forced strategies and to size Auto's one-lane item width.
///
/// `make_scratch` builds one lane's scratch from that lane's borrowed workspace chunk, reused for
/// every item of that lane. It is called with the item width the decision granted, so scratch sized
/// per thread matches the width the items actually run at. `ScratchArena` hands each lane its own
/// chunk of a caller-owned byte region; a `Vec` wrapper passes a zero-length arena for a family
/// whose route needs no scratch.
///
/// Each lane stops at its own first failure and is not stopped by another lane's. The returned
/// error is the lowest-indexed failing lane's, which is the lowest-indexed failing item, so the
/// result does not depend on scheduling. On error every [`Out`] output is left empty.
///
/// # faer's hint
///
/// `item_threads` is the faer count requested on the pool, not a strict active-thread bound: faer
/// can expose more intra-item tasks than requested (see [`tcontract_exec::Exec`]'s contract).
/// Admission across simultaneous calls is the resource owner's.
#[allow(clippy::too_many_arguments)]
pub(crate) fn run<'w, O, S, MK, F>(
    op: Op,
    batch: usize,
    exec: Exec<'_>,
    max_item_dim: Option<usize>,
    item_muladds: f64,
    outputs: &mut O,
    workspace: ScratchArena<'w>,
    make_scratch: MK,
    item: F,
) -> Result<()>
where
    O: Outputs,
    MK: Fn(faer::Par, &'w mut [MaybeUninit<u8>]) -> S + Sync,
    F: Fn(usize, &mut O::Item, &mut S, faer::Par) -> Result<()> + Sync,
{
    // Resolve availability **before** touching the outputs: a forced `WholeBatchVendor` is
    // refused even when the batch is empty, and a huge-but-valid batch must not reach
    // `outputs.prepare`'s reservation first. The entry points clear their library-created outputs
    // before calling the driver, so a refusal still leaves them empty.
    let (lanes, item_threads) = decide(op, exec, batch, max_item_dim, item_muladds)?;
    outputs.prepare(op, batch)?;
    if batch == 0 {
        return Ok(());
    }
    // INVARIANT: `batch_decision` clamps every route to the item count, so `lanes <= batch` and
    // `lane_bounds` yields exactly `lanes` nonempty ranges.
    debug_assert!(lanes >= 1 && lanes <= batch, "lanes={lanes} batch={batch}");
    let item_threads = item_threads.max(1);
    #[cfg(test)]
    tests::record_lanes(lanes);
    let result = {
        // SAFETY: `prepare` succeeded above; every item is created once from its own index and
        // consumed inside this block, before `outputs` is touched again by `commit`.
        let parts = unsafe { outputs.parts(batch) };
        let parts = &parts;
        let run_lane = |lane: usize, faer_par: faer::Par| -> Result<()> {
            let (first, last) = lane_bounds(batch, lanes, lane);
            // SAFETY: each call site passes a distinct lane index, whose chunk is disjoint from the
            // others'.
            let region = unsafe { workspace.lane(lane) };
            let mut scratch = make_scratch(faer_par, region);
            for index in first..last {
                // SAFETY: `index` is below the batch count and the lane ranges are disjoint.
                let mut item_out = unsafe { O::item(parts, index) };
                item(index, &mut item_out, &mut scratch, faer_par)?;
                if !O::item_complete(&item_out) {
                    return Err(Error::Inconsistent {
                        op,
                        detail: "a kernel did not fill its output",
                    });
                }
            }
            Ok(())
        };
        if lanes == 1 {
            // One lane, item-parallel: `install` enters the pool for `item_threads > 1` and stays
            // on the caller otherwise.
            exec.install(item_threads, |par| run_lane(0, crate::faer_par(par)))
        } else {
            // Each lane stops independently; report the lowest failing lane deterministically.
            // The items inside a lane run at the decision's item width, so a `Split`/`Inner`-style
            // forced width is not silently dropped. `install` enters the pool once, at `lanes`.
            let failure: Mutex<Option<(usize, Error)>> = Mutex::new(None);
            let (run_lane, failure_ref) = (&run_lane, &failure);
            exec.install(lanes, |_| {
                rayon::scope(|scope| {
                    for lane in 0..lanes {
                        scope.spawn(move |_| {
                            let par = faer::Par::rayon(item_threads);
                            if let Err(error) = run_lane(lane, par) {
                                let mut slot = failure_ref
                                    .lock()
                                    .unwrap_or_else(std::sync::PoisonError::into_inner);
                                if slot.as_ref().is_none_or(|(first, _)| lane < *first) {
                                    *slot = Some((lane, error));
                                }
                            }
                        });
                    }
                });
            });
            match failure
                .into_inner()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
            {
                Some((_, error)) => Err(error),
                None => Ok(()),
            }
        }
    };
    if result.is_ok() {
        // SAFETY: every item `0..batch` was created once and returned `Ok`, which it does only
        // after `O::item_complete` confirmed it filled its output; each item view was dropped at
        // the end of its run.
        unsafe { outputs.commit(batch) };
    }
    result
}

#[cfg(test)]
mod tests;
