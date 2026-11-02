//! The one conversion the estimate performs: a work count over a rate, as a `Duration`.

use core::time::Duration;

use super::ns_duration;

#[test]
fn a_nanosecond_count_is_a_duration() {
    assert_eq!(ns_duration(1.0), Duration::from_nanos(1));
    assert_eq!(ns_duration(1e9), Duration::from_secs(1));
    assert_eq!(ns_duration(2.5e6), Duration::from_micros(2500));
}

#[test]
fn a_count_that_cannot_be_a_duration_is_zero() {
    assert_eq!(ns_duration(0.0), Duration::ZERO);
    assert_eq!(ns_duration(-1.0), Duration::ZERO);
    assert_eq!(ns_duration(f64::NAN), Duration::ZERO);
    assert_eq!(ns_duration(f64::NEG_INFINITY), Duration::ZERO);
}

#[test]
fn an_unrepresentable_count_saturates_instead_of_panicking() {
    assert_eq!(ns_duration(f64::INFINITY), Duration::MAX);
    assert_eq!(ns_duration(f64::MAX), Duration::MAX);
    // A work count no `usize` shape can produce, priced at a very fast rate, is still a `Duration`.
    assert_eq!(ns_duration(1e29), Duration::MAX);
    assert!(ns_duration(1e28) < Duration::MAX);
}
