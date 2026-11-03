use super::{mark_visited, normalize_into};
use crate::Op;

/// The visited marker must represent every pivot LAPACK may return, including the largest
/// `i32` value: negation is safe there, while the former `value + n` marker was not.
#[test]
fn the_visited_marker_does_not_overflow() {
    let mut entries = [i32::MAX];
    assert_eq!(mark_visited(&mut entries, 0), Some(i32::MAX as usize - 1));
    assert_eq!(entries, [-i32::MAX]);
    // A marked entry is a re-entry, not a second visit.
    assert_eq!(mark_visited(&mut entries, 0), None);
    assert_eq!(entries, [-i32::MAX]);
}

/// The cycle walk accepts a permutation, restores it, and rejects a duplicate target.
#[test]
fn the_permutation_check_restores_and_rejects_duplicates() {
    let mut valid = [2_i32, 3, 1];
    let mut out = Vec::new();
    normalize_into(Op::RankRevealingQr, &mut valid, &mut out).unwrap();
    assert_eq!(valid, [2, 3, 1], "the marker is restored");
    assert_eq!(out, [1, 2, 0]);

    // Two positions map to entry 1: not a permutation.
    let mut duplicate = [1_i32, 1, 3];
    let mut out = Vec::new();
    assert!(normalize_into(Op::RankRevealingQr, &mut duplicate, &mut out).is_err());

    // Out of range.
    let mut out_of_range = [1_i32, 4, 3];
    let mut out = Vec::new();
    assert!(normalize_into(Op::RankRevealingQr, &mut out_of_range, &mut out).is_err());
}
