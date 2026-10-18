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
use crate::{Op, Result};

/// One lane's live scratch requirement for a `lanes`-lane route, in bytes.
///
/// Computed as `lanes * unaligned_bytes_required(per_lane)`: every lane chunk then carries its own
/// worst-case alignment padding, so a lane never depends on where the caller's buffer happens to
/// start.
pub(crate) fn lanes_bytes(op: Op, per_lane: StackReq, lanes: usize) -> Result<usize> {
    let bytes = per_lane.unaligned_bytes_required();
    if bytes == usize::MAX {
        return Err(invalid(
            op,
            "workspace",
            "the workspace requirement overflows usize",
        ));
    }
    bytes
        .checked_mul(lanes)
        .ok_or_else(|| invalid(op, "workspace", "the workspace requirement overflows usize"))
}

/// One lane's live requirement, or zero when the call has no items at all (nothing runs, so no
/// workspace is needed and a shape-only huge extent must not turn into a huge byte count).
pub(crate) fn per_lane_bytes(req: StackReq, batch: usize) -> usize {
    if batch == 0 {
        0
    } else {
        req.unaligned_bytes_required()
    }
}

/// The complete requirement of a `lanes`-lane, `batch`-item call.
pub(crate) fn total_bytes(op: Op, req: StackReq, lanes: usize, batch: usize) -> Result<usize> {
    lanes_bytes(op, req, if batch == 0 { 0 } else { lanes })
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
