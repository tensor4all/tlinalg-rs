use super::*;
use tlinalg::{Error, NonFiniteRole, Op};

/// The mapping table is the contract: every provider variant has exactly one C status.
#[test]
fn the_error_table_maps_every_typed_variant() {
    let cases: [(Error, tlinalg_status); 6] = [
        (Error::Singular { op: Op::Cholesky }, TLINALG_ERR_SINGULAR),
        (
            Error::NonConvergence { op: Op::Svd },
            TLINALG_ERR_NONCONVERGENCE,
        ),
        (
            Error::NonFinite {
                op: Op::Cholesky,
                role: NonFiniteRole::Input,
            },
            TLINALG_ERR_NONFINITE,
        ),
        (
            Error::InvalidArgument {
                op: Op::Eigh,
                role: "configuration",
                detail: "detail".into(),
            },
            TLINALG_ERR_INVALID_ARGUMENT,
        ),
        (
            Error::Inconsistent {
                op: Op::Lu,
                detail: "detail",
            },
            TLINALG_ERR_SHAPE,
        ),
        (
            Error::InvalidWorkspace {
                op: Op::Svd,
                library: "LAPACK",
                routine: "routine",
                detail: "detail".into(),
            },
            TLINALG_ERR_INTERNAL,
        ),
    ];
    for (error, want) in cases {
        let mapped = map_error(error);
        assert_eq!(mapped.status, want);
        assert!(!mapped.message.is_empty());
    }
}

/// Every published status has a description.
#[test]
fn describe_covers_every_status() {
    for status in [
        TLINALG_OK,
        TLINALG_ERR_INVALID_ARGUMENT,
        TLINALG_ERR_SHAPE,
        TLINALG_ERR_SINGULAR,
        TLINALG_ERR_UNSUPPORTED,
        TLINALG_ERR_INTERNAL,
        TLINALG_ERR_NONCONVERGENCE,
        TLINALG_ERR_NONFINITE,
    ] {
        assert_ne!(describe(status), "unknown error");
    }
    assert_eq!(describe(999), "unknown error");
}
