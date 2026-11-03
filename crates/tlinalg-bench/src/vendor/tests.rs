use super::*;

#[test]
fn a_version_is_read_out_of_the_config_string() {
    if cfg!(feature = "link-mkl") {
        assert_eq!(
            version_of(
                "Intel(R) oneAPI Math Kernel Library Version 2026.1-Product Build 20260612 \
                 for Intel(R) 64 architecture applications"
            )
            .as_deref(),
            Some("2026.1")
        );
        assert_eq!(version_of("something else"), None);
        return;
    }
    assert_eq!(
        version_of("OpenBLAS 0.3.32 NO_AFFINITY COOPERLAKE").as_deref(),
        Some("0.3.32")
    );
    assert_eq!(version_of("something else"), None);
}

/// The point of the module: a budget that was set is the budget in force, and it can be read
/// back. Without a vendor feature this only asserts the feature-independent behaviour.
#[test]
fn the_vendor_budget_is_set_and_read_back() {
    if !LINKED {
        assert!(set_threads(4).is_err(), "no vendor library to set");
        assert!(require_threads(4).is_ok(), "nothing to contradict");
        return;
    }
    prepare().expect("the vendor accepts the layers this build needs");
    for threads in [1, 3, 8] {
        set_threads(threads).expect("the vendor accepts a thread budget");
        assert_eq!(sys::threads(), Some(threads));
    }
    set_threads(1).unwrap();
    assert_eq!(
        VENDOR_NAME,
        if cfg!(feature = "link-mkl") {
            "mkl"
        } else {
            "openblas"
        }
    );
    assert_eq!(identity().name, VENDOR_NAME);
    let identity = identity();
    assert!(
        identity.config.is_some(),
        "the vendor must be able to name itself"
    );
    println!(
        "{}",
        identity
            .lines()
            .iter()
            .map(|(k, v)| format!("{k}={v}"))
            .collect::<Vec<_>>()
            .join("\n")
    );
}
