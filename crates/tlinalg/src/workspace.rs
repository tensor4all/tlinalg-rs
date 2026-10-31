//! Carving one lane's scratch out of caller-owned bytes.
//!
//! The `*_workspace` queries of this crate return a **byte count**, worst-case alignment padding
//! included (the whole composition is `dyn_stack::StackReq`'s `unaligned_bytes_required`, times the
//! number of lanes the resolved route may run). The matching `*_into` entry point takes a
//! `&mut [MaybeUninit<u8>]` of at least that many bytes and carves every per-lane buffer from it
//! with faer's [`MemStack`], through the helpers below.
//!
//! Why bytes and not a layout type: the requirement is one lane's complete inventory at the item
//! width the batch decision granted (faer sizes part of its scratch per thread), not a
//! provider-independent shape. The host only has to keep the bytes alive across the call.
//!
//! The sub-buffers are carved in a fixed order, so a re-run of the same shape maps the same
//! offsets. Every zero-initialised buffer is reset here exactly as the `Vec` wrapper's counterpart
//! used to be built, and the per-item reset policy of the kernels is unchanged. On an error the
//! storage is abandoned, not released: it belongs to the caller.

use core::mem::MaybeUninit;

use faer::dyn_stack::{MemStack, StackReq};

use crate::util::invalid;
use crate::{Error, Op, Result};
use tcontract_exec::Exec;

/// The resolved route and one lane's requirement, or `None` when the batch has no items.
///
/// The order is the borrowed-workspace contract, in one place for every family:
///
/// 1. resolve the route from the shared batch decision (arithmetic; it never fails);
/// 2. return `None` for `batch == 0`, before any requirement is built, so a shape-only huge extent
///    never turns into a huge (or overflowing) byte count;
/// 3. run the family's checked shape products — the values every requirement builder multiplies,
///    including the ones inside faer's recommendation helpers;
/// 4. build the requirement at the widest lane share the route grants and return `(lanes, per_lane_bytes)`.
pub(crate) fn lane_requirement(
    op: Op,
    exec: &Exec<'_>,
    batch: usize,
    work: impl Into<crate::batch::Work>,
    check: impl FnOnce() -> Result<()>,
    req: impl FnOnce(faer::ParShape) -> StackReq,
) -> Result<Option<(usize, usize)>> {
    let route = crate::batch::decide(exec, batch, work);
    if batch == 0 {
        return Ok(None);
    }
    check()?;
    let bytes = req(route.par()).unaligned_bytes_required();
    if bytes == usize::MAX {
        return Err(overflow(op));
    }
    Ok(Some((route.lanes, bytes)))
}

/// One lane's requirement times the resolved lane count.
pub(crate) fn total_of(op: Op, lanes: usize, per_lane: usize) -> Result<usize> {
    per_lane.checked_mul(lanes).ok_or_else(|| overflow(op))
}

/// The typed refusal for a requirement that does not fit in `usize`.
fn overflow(op: Op) -> Error {
    invalid(op, "workspace", "the workspace requirement overflows usize")
}

/// A faer scratch requirement as one lane-layout component.
///
/// `StackReq::unaligned_bytes_required` already accounts for the worst-case alignment offset of the
/// region, so a byte region of that length always satisfies `req` whatever its own base alignment
/// is: `MemStack::can_hold` is then guaranteed for the region handed to faer.
pub(crate) fn faer_req(req: StackReq) -> StackReq {
    StackReq::new::<u8>(req.unaligned_bytes_required())
}

/// A `len`-element buffer with every element set to `zero`.
pub(crate) fn zeros<'w, T: Clone>(
    stack: &'w mut MemStack,
    len: usize,
    zero: &T,
) -> (&'w mut [T], &'w mut MemStack) {
    if len == 0 {
        // `MemStack::make_uninit` may reject a zero-length allocation on a misaligned empty buffer,
        // so an empty buffer never touches the stack.
        return (&mut [], stack);
    }
    let (slots, rest) = stack.make_uninit::<T>(len);
    for slot in slots.iter_mut() {
        slot.write(zero.clone());
    }
    let count = slots.len();
    let ptr = slots.as_mut_ptr().cast::<T>();
    // SAFETY: every element was just written with `zero`, and the `MaybeUninit<T>` array has the
    // same layout as `[T]`.
    (unsafe { core::slice::from_raw_parts_mut(ptr, count) }, rest)
}

/// A `len`-element region faer fills with its own scratch.
pub(crate) fn bytes(stack: &mut MemStack, len: usize) -> (&mut [MaybeUninit<u8>], &mut MemStack) {
    if len == 0 {
        return (&mut [], stack);
    }
    stack.make_uninit::<u8>(len)
}
