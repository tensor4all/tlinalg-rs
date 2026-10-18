mod scheduling;
pub(super) use scheduling::record_lanes;

use super::{
    fixed_out, is_injective, run, BatchAxes, BatchedMut, BatchedRef, ScratchArena, VecOut,
};
use crate::{Error, Op};
use core::sync::atomic::{AtomicUsize, Ordering};
use strided_view::{RawStridedMut, RawStridedRef};
use tcontract_exec::{Exec, Pool};

/// An owned pool (kept alive by `Pool::owned`) and an `Exec` borrowing it.
fn owned_pool(threads: usize) -> Pool<'static> {
    let tp = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .unwrap();
    Pool::owned(tp)
}

/// A failing outer lane must not stop the other lanes.
#[test]
fn a_failing_lane_does_not_stop_the_later_ones() {
    let pool = owned_pool(3);
    let ran = AtomicUsize::new(0);
    // The production output path: a reserved vector plus the `FixedOut` view of its spare capacity.
    let mut values: Vec<usize> = Vec::new();
    let mut values_out = VecOut::new(&mut values, 6);
    let err = run(
        Op::Svd,
        6,
        Exec::rayon(&pool).with_budget(3).unwrap(),
        Some(1),
        1.0,
        &mut (fixed_out(values_out.spare(), 1, "values"),),
        ScratchArena::new(&mut [], 0),
        |_, _| (),
        |index, (values,), (), _| {
            if index == 1 {
                return Err(Error::Inconsistent {
                    op: Op::Svd,
                    detail: "one",
                });
            }
            values.push(index);
            ran.fetch_add(1, Ordering::Relaxed);
            Ok(())
        },
    )
    .unwrap_err();
    assert_eq!(
        err,
        Error::Inconsistent {
            op: Op::Svd,
            detail: "one"
        }
    );
    // Items 0 and 2..5; the failing item 1 is not counted.
    assert_eq!(ran.load(Ordering::Relaxed), 5);
}

/// Items 2 and 5 fail with distinguishable errors; whatever the scheduling, the error of the
/// lowest-indexed failing item is returned and the output stays empty.
#[test]
fn the_lowest_failing_item_wins_and_outputs_stay_empty() {
    let pool = owned_pool(8);
    for budget in [1, 2, 3, 4, 8] {
        let exec = Exec::rayon(&pool).with_budget(budget).unwrap();
        for _ in 0..20 {
            let mut values = vec![9usize; 3];
            let mut values_out = VecOut::new(&mut values, 8);
            let err = run(
                Op::Svd,
                8,
                exec,
                Some(1),
                1.0,
                &mut (fixed_out(values_out.spare(), 1, "values"),),
                ScratchArena::new(&mut [], 0),
                |_, _| (),
                |index, (values,), (), _| {
                    if index == 2 || index == 5 {
                        return Err(Error::Inconsistent {
                            op: Op::Svd,
                            detail: if index == 2 { "two" } else { "five" },
                        });
                    }
                    values.push(index);
                    Ok(())
                },
            )
            .unwrap_err();
            assert_eq!(
                err,
                Error::Inconsistent {
                    op: Op::Svd,
                    detail: "two"
                },
                "budget={budget}"
            );
            assert!(values.is_empty());
        }
    }
}

#[test]
fn a_kernel_that_writes_too_little_is_an_error() {
    let mut values: Vec<u8> = Vec::new();
    // Two items of two elements each; the kernel writes one, so each sink is incomplete.
    let mut values_out = VecOut::new(&mut values, 4);
    let err = run(
        Op::Svd,
        2,
        Exec::serial(),
        Some(1),
        1.0,
        &mut (fixed_out(values_out.spare(), 2, "values"),),
        ScratchArena::new(&mut [], 0),
        |_, _| (),
        |_, (values,), (), _| {
            values.push(1);
            Ok(())
        },
    );
    assert!(matches!(err, Err(Error::Inconsistent { .. })));
    assert!(values.is_empty());
}

/// `strided-view` skips offset validation for an empty layout and then returns its `dangling()`
/// pointer, so an item that applied its batch offset could wrap that pointer to null. faer turns
/// the pointer it is handed into a `NonNull`, so a null is not an accepted input.
#[test]
fn an_empty_item_keeps_a_non_null_pointer() {
    let empty: [f64; 0] = [];
    let dims = [0usize, 0, 2];
    // `wrapping_offset` counts elements, so a batch stride of `-1` is accepted for the empty
    // layout and wraps the dangling pointer of address `align_of::<f64>() == 8` to null.
    let strides = [1isize, 0, -1];
    let view = RawStridedRef::new(&empty, &dims, &strides, 0).unwrap();
    let batched = BatchedRef::new(Op::Svd, "input", view).unwrap();
    assert_eq!(batched.batch(), 2);
    for index in 0..batched.batch() {
        let matrix = batched.item(index);
        assert!(!matrix.as_ptr().is_null(), "item {index}");
        assert_eq!(matrix.as_ptr() as usize % core::mem::align_of::<f64>(), 0);
    }
}

#[test]
fn an_empty_mutable_item_keeps_a_non_null_pointer() {
    let mut empty: [f64; 0] = [];
    let dims = [0usize, 0, 2];
    let strides = [1isize, 0, -1];
    let mut view = RawStridedMut::new(&mut empty, &dims, &strides, 0).unwrap();
    let batched = BatchedMut::new(Op::Svd, "output", &mut view).unwrap();
    assert_eq!(batched.batch, 2);
    for index in 0..batched.batch {
        // SAFETY: every item is visited once and its matrix is dropped before the next.
        let matrix = unsafe { batched.item(index) };
        assert!(!matrix.as_ptr().is_null(), "item {index}");
    }
}

#[test]
fn injectivity() {
    assert!(is_injective(&[2, 3], &[1, 2]));
    assert!(is_injective(&[2, 3, 4], &[1, 4, 12]));
    assert!(!is_injective(&[2, 3], &[1, 1]));
    assert!(!is_injective(&[2, 2, 2], &[1, 2, 0]));
    assert!(is_injective(&[2, 1, 2], &[1, 0, 2]));
    assert!(is_injective(&[0, 2], &[0, 0]));
}

#[test]
fn offsets_walk_the_first_batch_dim_fastest() {
    let axes = BatchAxes::new(&[2, 3], &[10, 100]);
    assert_eq!(axes.len(), 2, "not mergeable");
    assert_eq!(axes.offset(0), 0);
    assert_eq!(axes.offset(1), 10);
    assert_eq!(axes.offset(2), 100);
    assert_eq!(axes.offset(5), 210);
}

#[test]
fn compact_axes_coalesce_and_unit_axes_drop() {
    let axes = BatchAxes::new(&[2, 1, 3, 4], &[9, 77, 18, 54]);
    assert_eq!(axes.len(), 1);
    assert_eq!(axes.offset(23), 23 * 9);
    assert_eq!(BatchAxes::new(&[1, 1], &[5, 6]).len(), 0);
    // Broadcast axis: stride 0 is kept and never merges with a non-zero neighbour.
    let broadcast = BatchAxes::new(&[3, 2], &[0, 4]);
    assert_eq!(broadcast.len(), 2);
    assert_eq!(broadcast.offset(4), 4);
    assert_eq!(BatchAxes::new(&[3, 2], &[0, 0]).offset(5), 0);
}

#[test]
fn axes_stay_inline_up_to_eight_and_spill_beyond() {
    let dims = [2usize; 10];
    // Non-mergeable: every stride leaves a gap.
    let strides: Vec<isize> = (0..10).map(|axis| 3isize.pow(axis)).collect();
    let eight = BatchAxes::new(&dims[..8], &strides[..8]);
    assert!(eight.is_inline() && eight.len() == 8);
    let ten = BatchAxes::new(&dims, &strides);
    assert!(!ten.is_inline() && ten.len() == 10);
    assert_eq!(ten.offset(1023), strides.iter().sum::<isize>());
}
