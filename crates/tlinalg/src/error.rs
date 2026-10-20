//! Typed errors a numerical implementation reports to its host.
//!
//! The variants are shaped so that a host can rebuild its own error **kind, role and typed
//! payload** exactly. Two consequences follow from that requirement:
//!
//! * [`Error::Singular`] and [`Error::NonConvergence`] are distinct, because which one a provider
//!   reports for a failed factorization is routine-dependent (see the crate table).
//! * [`Error::NonFinite`] carries a [`NonFiniteRole`], because callers distinguish a non-finite
//!   input from a non-finite computed diagonal.
//!
//! [`Error::InvalidArgument`] and [`Error::Inconsistent`] are not "numerical" failures: they exist
//! because a host may have to reproduce its own argument/internal error shapes for these cases.
//! An implementation must not invent its own strings for them.

/// Linear-algebra operation that produced the failure.
///
/// The host maps these to its own operation names; implementations never format these values into
/// user-visible text.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
#[non_exhaustive]
pub enum Op {
    /// Cholesky factorization.
    Cholesky,
    /// Triangular solve.
    TriangularSolve,
    /// Partial-pivot LU factorization.
    Lu,
    /// Full-pivot LU factorization.
    FullPivLu,
    /// Full-pivot LU solve.
    FullPivLuSolve,
    /// Linear solve.
    Solve,
    /// Thin SVD.
    Svd,
    /// Full SVD.
    SvdFull,
    /// Singular values only.
    SvdValues,
    /// Thin QR.
    Qr,
    /// Rank-revealing (column-pivoted) QR.
    RankRevealingQr,
    /// Compact Householder QR factorization.
    HouseholderQr,
    /// Householder QR from stored factors.
    HouseholderQrFromFactors,
    /// Incremental Householder QR append.
    HouseholderQrAppend,
    /// Householder QR `R` extraction.
    HouseholderQrR,
    /// Householder QR `Q` column extraction.
    HouseholderQrQColumns,
    /// Hermitian eigendecomposition.
    Eigh,
    /// Hermitian eigenvalues only.
    EighValues,
    /// General eigendecomposition.
    Eig,
    /// General eigenvalues only.
    EigValues,
    /// Packed LU factor for later solves.
    LuFactor,
    /// Solve from a prepared packed LU factor.
    LuSolvePrepared,
    /// Fused packed LU factor and solve.
    LuFactorSolve,
}

impl Op {
    /// The host-facing operation name.
    ///
    /// This is the exact string a host uses in its own error payloads and diagnostics.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Cholesky => "cholesky",
            Self::TriangularSolve => "triangular_solve",
            Self::Lu => "lu",
            Self::FullPivLu => "full_piv_lu",
            Self::FullPivLuSolve => "full_piv_lu_solve",
            Self::Solve => "solve",
            Self::Svd => "svd",
            Self::SvdFull => "svd_full",
            Self::SvdValues => "svd_values",
            Self::Qr => "qr",
            Self::RankRevealingQr => "rank_revealing_qr",
            Self::HouseholderQr => "householder_qr",
            Self::HouseholderQrFromFactors => "householder_qr_from_factors",
            Self::HouseholderQrAppend => "householder_qr_append",
            Self::HouseholderQrR => "householder_qr_r",
            Self::HouseholderQrQColumns => "householder_qr_q_columns",
            Self::Eigh => "eigh",
            Self::EighValues => "eigh_values",
            Self::Eig => "eig",
            Self::EigValues => "eig_values",
            Self::LuFactor => "lu_factor",
            Self::LuSolvePrepared => "lu_solve_prepared",
            Self::LuFactorSolve => "lu_factor_solve",
        }
    }
}

/// Which quantity was non-finite.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
#[non_exhaustive]
pub enum NonFiniteRole {
    /// An input operand was non-finite.
    Input,
    /// The computed diagonal of `R` was non-finite.
    RDiagonal,
}

impl NonFiniteRole {
    /// The host-facing role name.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Input => "input",
            Self::RDiagonal => "R diagonal",
        }
    }
}

/// A failure reported by a numerical implementation.
#[derive(Clone, Debug, Eq, PartialEq, thiserror::Error)]
#[non_exhaustive]
pub enum Error {
    /// The algorithm could not converge for this input.
    #[error("{op} did not converge", op = .op.as_str())]
    NonConvergence {
        /// Operation that failed to converge.
        op: Op,
    },

    /// An input or a required computed quantity was non-finite.
    #[error("{op} encountered non-finite {role}", op = .op.as_str(), role = .role.as_str())]
    NonFinite {
        /// Operation that encountered the value.
        op: Op,
        /// Which quantity was non-finite.
        role: NonFiniteRole,
    },

    /// A factor, or an input the operation requires to be non-singular, was exactly singular.
    #[error("{op} is singular", op = .op.as_str())]
    Singular {
        /// Operation that required a non-singular input.
        op: Op,
    },

    /// A caller-supplied argument was invalid.
    ///
    /// `role` is the host's name for the offending argument — `"pivot"`, `"config"`,
    /// `"lapack_argument"` — and `detail` carries the provider's own text. Both are part of the
    /// host-visible payload, so an implementation must not invent its own strings.
    #[error("{op}: invalid {role} ({detail})", op = .op.as_str())]
    InvalidArgument {
        /// Operation being executed.
        op: Op,
        /// Which argument was invalid, in the host's vocabulary.
        role: &'static str,
        /// Provider-provided detail.
        detail: String,
    },

    /// A provider returned an unusable workspace size.
    #[error("{library} routine {routine} returned an invalid workspace: {detail}")]
    InvalidWorkspace {
        /// Operation being executed.
        op: Op,
        /// Provider library name, for example `"LAPACK"`.
        library: &'static str,
        /// Provider routine name.
        routine: &'static str,
        /// Provider-provided detail.
        detail: String,
    },

    /// Caller-supplied batch buffers disagree about the problem count or shape.
    #[error("{op}: inconsistent batch buffers ({detail})", op = .op.as_str())]
    Inconsistent {
        /// Operation being executed.
        op: Op,
        /// What disagreed.
        detail: &'static str,
    },

    /// The caller's borrowed workspace is too small for the route this call resolves to.
    ///
    /// Reported by the `*_into` entry points **before** any output is written; the query
    /// (`*_workspace`) returns the byte count a matching call needs, worst-case alignment padding
    /// included.
    #[error(
        "{op}: workspace of {have} bytes is too small, {needed} bytes are required",
        op = .op.as_str()
    )]
    InsufficientWorkspace {
        /// Operation being executed.
        op: Op,
        /// Bytes the resolved route requires (all lanes).
        needed: usize,
        /// Bytes the caller supplied.
        have: usize,
    },
}

/// Result type for this crate's operations.
pub type Result<T> = core::result::Result<T, Error>;
