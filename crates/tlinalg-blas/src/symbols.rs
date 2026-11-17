//! The vendor symbols a scalar needs, sealed away from the public bound.
//!
//! Every LAPACK/BLAS routine this crate calls is reached through [`Symbols`], so the set of symbols a
//! host has to supply (linked or injected) is exactly the set named here.

use cblas_sys::{CBLAS_DIAG, CBLAS_LAYOUT, CBLAS_SIDE, CBLAS_TRANSPOSE, CBLAS_UPLO};
use num_complex::{Complex32, Complex64};

// SAFETY: the full-pivot LU routines are not bound by the `lapack` crate; these declarations keep
// the provider's LAPACK LP64 Fortran ABI, and every caller validates the buffers it passes.
unsafe extern "C" {
    #[link_name = "sgetc2_"]
    fn sgetc2_ffi(
        n: *const i32,
        a: *mut f32,
        lda: *const i32,
        ipiv: *mut i32,
        jpiv: *mut i32,
        info: *mut i32,
    );
    #[link_name = "sgesc2_"]
    fn sgesc2_ffi(
        n: *const i32,
        a: *const f32,
        lda: *const i32,
        rhs: *mut f32,
        ipiv: *const i32,
        jpiv: *const i32,
        scale: *mut f32,
    );
    #[link_name = "dgetc2_"]
    fn dgetc2_ffi(
        n: *const i32,
        a: *mut f64,
        lda: *const i32,
        ipiv: *mut i32,
        jpiv: *mut i32,
        info: *mut i32,
    );
    #[link_name = "dgesc2_"]
    fn dgesc2_ffi(
        n: *const i32,
        a: *const f64,
        lda: *const i32,
        rhs: *mut f64,
        ipiv: *const i32,
        jpiv: *const i32,
        scale: *mut f64,
    );
    #[link_name = "cgetc2_"]
    fn cgetc2_ffi(
        n: *const i32,
        a: *mut Complex32,
        lda: *const i32,
        ipiv: *mut i32,
        jpiv: *mut i32,
        info: *mut i32,
    );
    #[link_name = "cgesc2_"]
    fn cgesc2_ffi(
        n: *const i32,
        a: *const Complex32,
        lda: *const i32,
        rhs: *mut Complex32,
        ipiv: *const i32,
        jpiv: *const i32,
        scale: *mut f32,
    );
    #[link_name = "zgetc2_"]
    fn zgetc2_ffi(
        n: *const i32,
        a: *mut Complex64,
        lda: *const i32,
        ipiv: *mut i32,
        jpiv: *mut i32,
        info: *mut i32,
    );
    #[link_name = "zgesc2_"]
    fn zgesc2_ffi(
        n: *const i32,
        a: *const Complex64,
        lda: *const i32,
        rhs: *mut Complex64,
        ipiv: *const i32,
        jpiv: *const i32,
        scale: *mut f64,
    );
}

/// The LAPACK/BLAS routines and scalar literals this crate needs.
///
/// `pub` only so [`crate::LapackScalar`] can name its associated types; sealed by
/// [`crate::Scalar`], which no new type can implement, and by the orphan rule for the existing ones.
///
/// Where the real and complex routines differ in shape, the method takes the union of their
/// arguments and documents which ones each kind ignores; the callers branch on
/// [`Symbols::COMPLEX`] to size them.
pub trait Symbols: crate::Scalar + Default + PartialEq {
    /// The real scalar this type's singular values and eigenvalues live in.
    type Real: crate::Scalar
        + Default
        + PartialEq
        + PartialOrd
        + core::ops::Neg<Output = Self::Real>;
    /// The complex scalar a general eigendecomposition of this type returns.
    type Complex: crate::Scalar + Default + PartialEq;

    /// Whether this is a complex scalar.
    const COMPLEX: bool;
    /// The letter LAPACK takes for the adjoint of a reflector sequence: `T` for real, `C` for
    /// complex.
    const ADJOINT: u8;
    /// `?geqrf`, for diagnostics.
    const GEQRF: &'static str;
    /// `?orgqr`/`?ungqr`, for diagnostics.
    const ORGQR: &'static str;
    /// `?ormqr`/`?unmqr`, for diagnostics.
    const ORMQR: &'static str;
    /// `?geqp3`, for diagnostics.
    const GEQP3: &'static str;
    /// The Hermitian eigensolver this scalar uses (`?syevd` for real, `?heev` for complex).
    const EIGH: &'static str;
    /// `?geev`, for diagnostics.
    const GEEV: &'static str;

    /// `1` in this scalar.
    fn one() -> Self;
    /// `-1` in this scalar.
    fn minus_one() -> Self;
    /// `1` in the real scalar.
    fn real_one() -> Self::Real;
    /// The `lwork` a workspace query returned, as a real scalar.
    fn work_query_len(query: Self) -> f64;
    /// Whether every component is finite.
    fn is_finite(self) -> bool;
    /// `|self|` widened to `f64`.
    fn magnitude(self) -> f64;
    /// The real part.
    fn real_part(self) -> Self::Real;
    /// A real value widened to `f64`.
    fn real_to_f64(value: Self::Real) -> f64;
    /// The machine epsilon of the real scalar, widened to `f64`.
    fn real_epsilon() -> f64;
    /// `re + i im` in the complex counterpart.
    fn complex_from_parts(re: Self::Real, im: Self::Real) -> Self::Complex;
    /// The real eigenvalue/vector entry `value` as a real scalar; the identity for real scalars.
    fn real_as_self(value: Self::Real) -> Self;
    /// This value in the complex counterpart (the identity for complex scalars).
    fn to_complex(self) -> Self::Complex;
    /// Divide every element by `scale` unless it is exactly one.
    fn apply_inverse_scale(rhs: &mut [Self], scale: Self::Real);

    /// The LAPACK routine the selected SVD driver calls, for diagnostics.
    fn routine_name() -> &'static str;
    /// `?getrf`: factor one column-major `m x n` matrix in place.
    ///
    /// # Safety
    ///
    /// `data` holds a mutable column-major `lda x n` matrix, `ipiv` holds at least
    /// `min(m, n)` pivots, and `m`, `n`, `lda` describe them.
    unsafe fn getrf(m: i32, n: i32, data: &mut [Self], lda: i32, ipiv: &mut [i32], info: &mut i32);
    /// `?getrs`: solve `op(A) X = B` from `?getrf` factors.
    ///
    /// # Safety
    ///
    /// `a` and `ipiv` are a prior matching `?getrf` factorization, `b` holds a mutable
    /// `ldb x nrhs` right-hand side, and the dimensions describe them.
    #[allow(clippy::too_many_arguments)]
    unsafe fn getrs(
        trans: u8,
        n: i32,
        nrhs: i32,
        a: &[Self],
        lda: i32,
        ipiv: &[i32],
        b: &mut [Self],
        ldb: i32,
        info: &mut i32,
    );
    /// Conjugate every element in place; a no-op for real scalars.
    fn conj_in_place(_data: &mut [Self]) {}

    /// The SVD driver for one column-major `m x n` matrix, in place.
    ///
    /// `jobu` and `jobvt` are the two job letters `?gesvd` takes; `?gesdd` shares one letter, so
    /// it is given `jobu`. Which routine is called is a compiled-in choice: `?gesvd` with
    /// `provider-inject` (the injected LAPACK symbol set exports `?gesvd` only) or `accelerate`
    /// (Accelerate's `zgesdd` returns wrong factors, #110), and `?gesdd` otherwise.
    ///
    /// `rwork` is empty for the real routines and required by the complex ones; `iwork` is
    /// required by `?gesdd` and ignored by `?gesvd`.
    ///
    /// # Safety
    ///
    /// Every buffer holds what the selected LAPACK routine documents for these dimensions:
    /// `a` a mutable `m x n` matrix, `work` at least the queried `lwork`, `rwork` the
    /// documented real workspace, `iwork` the integer workspace, and `u`/`vt` the factors.
    #[allow(clippy::too_many_arguments)]
    unsafe fn svd_driver(
        jobu: u8,
        jobvt: u8,
        m: i32,
        n: i32,
        a: &mut [Self],
        lda: i32,
        s: &mut [Self::Real],
        u: &mut [Self],
        ldu: i32,
        vt: &mut [Self],
        ldvt: i32,
        work: &mut [Self],
        lwork: i32,
        rwork: &mut [Self::Real],
        iwork: &mut [i32],
        info: &mut i32,
    );

    /// `?potrf`.
    ///
    /// # Safety
    ///
    /// `a` holds a mutable column-major `lda x n` matrix.
    unsafe fn potrf(uplo: u8, n: i32, a: &mut [Self], lda: i32, info: &mut i32);

    /// `?trtrs`.
    ///
    /// # Safety
    ///
    /// `a` holds an `lda x n` triangular matrix and `b` a mutable `ldb x nrhs` right-hand side.
    #[allow(clippy::too_many_arguments)]
    unsafe fn trtrs(
        uplo: u8,
        trans: u8,
        diag: u8,
        n: i32,
        nrhs: i32,
        a: &[Self],
        lda: i32,
        b: &mut [Self],
        ldb: i32,
        info: &mut i32,
    );

    /// `cblas_?trsm` with `alpha = 1`, column-major, triangular factor on the **right**.
    ///
    /// # Safety
    ///
    /// `a` holds an `lda x n` triangular matrix and `b` a mutable `ldb x n` matrix with `m` rows.
    #[allow(clippy::too_many_arguments)]
    unsafe fn trsm_right(
        lower: bool,
        transpose: bool,
        unit_diagonal: bool,
        m: i32,
        n: i32,
        a: &[Self],
        lda: i32,
        b: &mut [Self],
        ldb: i32,
    );

    /// `?getc2`: complete-pivot LU of one `n x n` matrix in place.
    ///
    /// # Safety
    ///
    /// `a` holds a mutable `lda x n` matrix and both pivot arrays hold at least `n` entries.
    unsafe fn getc2(
        n: i32,
        a: &mut [Self],
        lda: i32,
        ipiv: &mut [i32],
        jpiv: &mut [i32],
        info: &mut i32,
    );

    /// `?gesc2`: solve one right-hand side from `?getc2` factors.
    ///
    /// # Safety
    ///
    /// `a`, `ipiv` and `jpiv` are a prior matching `?getc2` factorization and `rhs` holds `n`
    /// entries.
    #[allow(clippy::too_many_arguments)]
    unsafe fn gesc2(
        n: i32,
        a: &[Self],
        lda: i32,
        rhs: &mut [Self],
        ipiv: &[i32],
        jpiv: &[i32],
        scale: &mut Self::Real,
    );

    /// `?geqrf`.
    ///
    /// # Safety
    ///
    /// `a` holds a mutable `lda x n` matrix, `tau` `min(m, n)` entries, and `work` `lwork` entries
    /// (one for a query).
    #[allow(clippy::too_many_arguments)]
    unsafe fn geqrf(
        m: i32,
        n: i32,
        a: &mut [Self],
        lda: i32,
        tau: &mut [Self],
        work: &mut [Self],
        lwork: i32,
        info: &mut i32,
    );

    /// `?orgqr` (real) / `?ungqr` (complex).
    ///
    /// # Safety
    ///
    /// `a` holds `k` reflectors in a mutable `lda x n` matrix, `tau` `k` entries, `work` `lwork`.
    #[allow(clippy::too_many_arguments)]
    unsafe fn orgqr(
        m: i32,
        n: i32,
        k: i32,
        a: &mut [Self],
        lda: i32,
        tau: &[Self],
        work: &mut [Self],
        lwork: i32,
        info: &mut i32,
    );

    /// `?ormqr` (real) / `?unmqr` (complex).
    ///
    /// # Safety
    ///
    /// `a` holds `k` reflectors with leading dimension `lda`, `tau` `k` entries, `c` a mutable
    /// `ldc x n` matrix, `work` `lwork` entries.
    #[allow(clippy::too_many_arguments)]
    unsafe fn ormqr(
        side: u8,
        trans: u8,
        m: i32,
        n: i32,
        k: i32,
        a: &[Self],
        lda: i32,
        tau: &[Self],
        c: &mut [Self],
        ldc: i32,
        work: &mut [Self],
        lwork: i32,
        info: &mut i32,
    );

    /// `?geqp3`. `rwork` (`2n` reals) is required by the complex routine and ignored by the real
    /// one.
    ///
    /// # Safety
    ///
    /// `a` holds a mutable `lda x n` matrix, `jpvt` `n` entries, `tau` `min(m, n)`, `work`
    /// `lwork`, and `rwork` what the complex routine documents.
    #[allow(clippy::too_many_arguments)]
    unsafe fn geqp3(
        m: i32,
        n: i32,
        a: &mut [Self],
        lda: i32,
        jpvt: &mut [i32],
        tau: &mut [Self],
        work: &mut [Self],
        lwork: i32,
        rwork: &mut [Self::Real],
        info: &mut i32,
    );

    /// The Hermitian eigensolver, lower triangle: `?syevd` for real scalars, `?heev` for complex.
    ///
    /// `rwork` is used only by `?heev`; `iwork`/`liwork` only by `?syevd`.
    ///
    /// # Safety
    ///
    /// `a` holds a mutable `lda x n` matrix, `w` `n` values, and the workspaces what the selected
    /// routine documents (one slot each for a query).
    #[allow(clippy::too_many_arguments)]
    unsafe fn eigh_driver(
        jobz: u8,
        n: i32,
        a: &mut [Self],
        lda: i32,
        w: &mut [Self::Real],
        work: &mut [Self],
        lwork: i32,
        rwork: &mut [Self::Real],
        iwork: &mut [i32],
        liwork: i32,
        info: &mut i32,
    );

    /// `?geev` with no left eigenvectors.
    ///
    /// For real scalars `w` receives the real parts and `wi` the imaginary parts; for complex
    /// scalars `w` receives the eigenvalues, `wi` is ignored, and `rwork` (`2n` reals) is required.
    ///
    /// # Safety
    ///
    /// `a` holds a mutable `lda x n` matrix, `w`/`wi` `n` entries, `vl` one entry, `vr` an
    /// `ldvr x n` matrix (or one entry when `jobvr = N`), and `work` `lwork` entries.
    #[allow(clippy::too_many_arguments)]
    unsafe fn geev(
        jobvr: u8,
        n: i32,
        a: &mut [Self],
        lda: i32,
        w: &mut [Self],
        wi: &mut [Self::Real],
        vl: &mut [Self],
        vr: &mut [Self],
        ldvr: i32,
        work: &mut [Self],
        lwork: i32,
        rwork: &mut [Self::Real],
        info: &mut i32,
    );
}

fn cblas_uplo(lower: bool) -> CBLAS_UPLO {
    if lower {
        CBLAS_UPLO::CblasLower
    } else {
        CBLAS_UPLO::CblasUpper
    }
}

fn cblas_transpose(transpose: bool) -> CBLAS_TRANSPOSE {
    if transpose {
        CBLAS_TRANSPOSE::CblasTrans
    } else {
        CBLAS_TRANSPOSE::CblasNoTrans
    }
}

fn cblas_diag(unit_diagonal: bool) -> CBLAS_DIAG {
    if unit_diagonal {
        CBLAS_DIAG::CblasUnit
    } else {
        CBLAS_DIAG::CblasNonUnit
    }
}

/// The routines whose real and complex bindings take the same arguments.
macro_rules! common_symbols {
    (
        $getrf:path, $getrs:path, $potrf:path, $trtrs:path, $getc2:path, $gesc2:path,
        $geqrf:path, $orgqr:path, $ormqr:path
    ) => {
        unsafe fn getrf(
            m: i32,
            n: i32,
            data: &mut [Self],
            lda: i32,
            ipiv: &mut [i32],
            info: &mut i32,
        ) {
            // SAFETY: callers validate `m`, `n` and `lda`, provide a mutable column-major
            // `lda x n` matrix, `min(m, n)` pivots, and live `info`.
            unsafe { $getrf(m, n, data, lda, ipiv, info) }
        }

        unsafe fn getrs(
            trans: u8,
            n: i32,
            nrhs: i32,
            a: &[Self],
            lda: i32,
            ipiv: &[i32],
            b: &mut [Self],
            ldb: i32,
            info: &mut i32,
        ) {
            // SAFETY: `a` holds a prior `getrf` factorization, `ipiv` matches it, `b` is a mutable
            // `ldb x nrhs` right-hand side, and dimensions are validated.
            unsafe { $getrs(trans, n, nrhs, a, lda, ipiv, b, ldb, info) }
        }

        unsafe fn potrf(uplo: u8, n: i32, a: &mut [Self], lda: i32, info: &mut i32) {
            // SAFETY: callers pass a mutable column-major `lda x n` factor buffer, validated i32
            // dimensions, and a live `info` output.
            unsafe { $potrf(uplo, n, a, lda, info) }
        }

        unsafe fn trtrs(
            uplo: u8,
            trans: u8,
            diag: u8,
            n: i32,
            nrhs: i32,
            a: &[Self],
            lda: i32,
            b: &mut [Self],
            ldb: i32,
            info: &mut i32,
        ) {
            // SAFETY: callers validate the triangular matrix and RHS shapes and provide
            // column-major `a`/`b` buffers matching `lda`/`ldb`.
            unsafe { $trtrs(uplo, trans, diag, n, nrhs, a, lda, b, ldb, info) }
        }

        unsafe fn getc2(
            n: i32,
            a: &mut [Self],
            lda: i32,
            ipiv: &mut [i32],
            jpiv: &mut [i32],
            info: &mut i32,
        ) {
            // SAFETY: `a` stores an `lda x n` column-major matrix, the pivot arrays hold at least
            // `n` entries, and every pointer is valid for the duration of the call.
            unsafe {
                $getc2(
                    &n,
                    a.as_mut_ptr(),
                    &lda,
                    ipiv.as_mut_ptr(),
                    jpiv.as_mut_ptr(),
                    info,
                );
            }
        }

        unsafe fn gesc2(
            n: i32,
            a: &[Self],
            lda: i32,
            rhs: &mut [Self],
            ipiv: &[i32],
            jpiv: &[i32],
            scale: &mut Self::Real,
        ) {
            // SAFETY: `a` stores the factored `lda x n` matrix, `rhs` and the pivot arrays hold at
            // least `n` entries, and LAPACK writes only through `rhs` and `scale`.
            unsafe {
                $gesc2(
                    &n,
                    a.as_ptr(),
                    &lda,
                    rhs.as_mut_ptr(),
                    ipiv.as_ptr(),
                    jpiv.as_ptr(),
                    scale,
                );
            }
        }

        unsafe fn geqrf(
            m: i32,
            n: i32,
            a: &mut [Self],
            lda: i32,
            tau: &mut [Self],
            work: &mut [Self],
            lwork: i32,
            info: &mut i32,
        ) {
            // SAFETY: callers validate the storage, tau and workspace lengths.
            unsafe { $geqrf(m, n, a, lda, tau, work, lwork, info) }
        }

        unsafe fn orgqr(
            m: i32,
            n: i32,
            k: i32,
            a: &mut [Self],
            lda: i32,
            tau: &[Self],
            work: &mut [Self],
            lwork: i32,
            info: &mut i32,
        ) {
            // SAFETY: callers supply `m x n` reflector storage, `k` tau entries, `k <= n <= m`,
            // and either one query slot or the queried workspace length.
            unsafe { $orgqr(m, n, k, a, lda, tau, work, lwork, info) }
        }

        unsafe fn ormqr(
            side: u8,
            trans: u8,
            m: i32,
            n: i32,
            k: i32,
            a: &[Self],
            lda: i32,
            tau: &[Self],
            c: &mut [Self],
            ldc: i32,
            work: &mut [Self],
            lwork: i32,
            info: &mut i32,
        ) {
            // SAFETY: callers validate the reflector, tau and `C` slices and the workspace.
            unsafe { $ormqr(side, trans, m, n, k, a, lda, tau, c, ldc, work, lwork, info) }
        }
    };
}

macro_rules! impl_real_symbols {
    (
        $scalar:ty, $complex:ty,
        $getrf:path, $getrs:path, $gesdd:path, $gesvd:path, $gesdd_routine:literal, $gesvd_routine:literal,
        $potrf:path, $trtrs:path, $trsm:path, $getc2:path, $gesc2:path,
        $geqrf:path, $orgqr:path, $ormqr:path, $geqp3:path, $syevd:path, $geev:path,
        $geqrf_name:literal, $orgqr_name:literal, $ormqr_name:literal, $geqp3_name:literal,
        $eigh_name:literal, $geev_name:literal
    ) => {
        impl Symbols for $scalar {
            type Real = $scalar;
            type Complex = $complex;

            const COMPLEX: bool = false;
            const ADJOINT: u8 = b'T';
            const GEQRF: &'static str = $geqrf_name;
            const ORGQR: &'static str = $orgqr_name;
            const ORMQR: &'static str = $ormqr_name;
            const GEQP3: &'static str = $geqp3_name;
            const EIGH: &'static str = $eigh_name;
            const GEEV: &'static str = $geev_name;

            fn one() -> Self {
                1.0
            }

            fn minus_one() -> Self {
                -1.0
            }

            fn real_one() -> Self {
                1.0
            }

            fn work_query_len(query: Self) -> f64 {
                query as f64
            }

            fn is_finite(self) -> bool {
                <$scalar>::is_finite(self)
            }

            fn magnitude(self) -> f64 {
                self.abs() as f64
            }

            fn real_part(self) -> Self {
                self
            }

            fn real_to_f64(value: Self) -> f64 {
                value as f64
            }

            fn real_epsilon() -> f64 {
                <$scalar>::EPSILON as f64
            }

            fn complex_from_parts(re: Self, im: Self) -> $complex {
                <$complex>::new(re, im)
            }

            fn real_as_self(value: Self) -> Self {
                value
            }

            fn to_complex(self) -> $complex {
                <$complex>::new(self, 0.0)
            }

            fn apply_inverse_scale(rhs: &mut [Self], scale: Self) {
                if scale != 1.0 {
                    for value in rhs {
                        *value /= scale;
                    }
                }
            }

            #[cfg(not(svd_gesvd))]
            fn routine_name() -> &'static str {
                $gesdd_routine
            }

            #[cfg(svd_gesvd)]
            fn routine_name() -> &'static str {
                $gesvd_routine
            }

            common_symbols!($getrf, $getrs, $potrf, $trtrs, $getc2, $gesc2, $geqrf, $orgqr, $ormqr);

            unsafe fn svd_driver(
                jobu: u8,
                jobvt: u8,
                m: i32,
                n: i32,
                a: &mut [Self],
                lda: i32,
                s: &mut [Self],
                u: &mut [Self],
                ldu: i32,
                vt: &mut [Self],
                ldvt: i32,
                work: &mut [Self],
                lwork: i32,
                _rwork: &mut [Self],
                iwork: &mut [i32],
                info: &mut i32,
            ) {
                // The driver is a compiled-in choice; see `Symbols::svd_driver`.
                #[cfg(svd_gesvd)]
                let _ = iwork;
                #[cfg(not(svd_gesvd))]
                let _ = jobvt;
                // SAFETY: callers validate the dimensions and layouts and supply buffers of the
                // lengths LAPACK documents; `lwork = -1` with a one-element `work` is the
                // workspace query.
                #[cfg(not(svd_gesvd))]
                unsafe {
                    $gesdd(
                        jobu, m, n, a, lda, s, u, ldu, vt, ldvt, work, lwork, iwork, info,
                    );
                }
                #[cfg(svd_gesvd)]
                unsafe {
                    $gesvd(
                        jobu, jobvt, m, n, a, lda, s, u, ldu, vt, ldvt, work, lwork, info,
                    );
                }
            }

            unsafe fn trsm_right(
                lower: bool,
                transpose: bool,
                unit_diagonal: bool,
                m: i32,
                n: i32,
                a: &[Self],
                lda: i32,
                b: &mut [Self],
                ldb: i32,
            ) {
                // SAFETY: callers validate dimensions and provide compact column-major `a` and
                // writable `b` buffers with matching leading dimensions.
                unsafe {
                    $trsm(
                        CBLAS_LAYOUT::CblasColMajor,
                        CBLAS_SIDE::CblasRight,
                        cblas_uplo(lower),
                        cblas_transpose(transpose),
                        cblas_diag(unit_diagonal),
                        m,
                        n,
                        1.0,
                        a.as_ptr(),
                        lda,
                        b.as_mut_ptr(),
                        ldb,
                    );
                }
            }

            unsafe fn geqp3(
                m: i32,
                n: i32,
                a: &mut [Self],
                lda: i32,
                jpvt: &mut [i32],
                tau: &mut [Self],
                work: &mut [Self],
                lwork: i32,
                _rwork: &mut [Self],
                info: &mut i32,
            ) {
                // SAFETY: callers validate the matrix, pivot, tau and workspace lengths.
                unsafe { $geqp3(m, n, a, lda, jpvt, tau, work, lwork, info) }
            }

            unsafe fn eigh_driver(
                jobz: u8,
                n: i32,
                a: &mut [Self],
                lda: i32,
                w: &mut [Self],
                work: &mut [Self],
                lwork: i32,
                _rwork: &mut [Self],
                iwork: &mut [i32],
                liwork: i32,
                info: &mut i32,
            ) {
                // SAFETY: callers validate the matrix and value lengths and pass either the query
                // slots or the queried workspace lengths.
                unsafe { $syevd(jobz, b'L', n, a, lda, w, work, lwork, iwork, liwork, info) }
            }

            unsafe fn geev(
                jobvr: u8,
                n: i32,
                a: &mut [Self],
                lda: i32,
                w: &mut [Self],
                wi: &mut [Self],
                vl: &mut [Self],
                vr: &mut [Self],
                ldvr: i32,
                work: &mut [Self],
                lwork: i32,
                _rwork: &mut [Self],
                info: &mut i32,
            ) {
                // SAFETY: callers validate the matrix, value and vector buffers; `lwork = -1`
                // writes only the query slot.
                unsafe {
                    $geev(
                        b'N', jobvr, n, a, lda, w, wi, vl, 1, vr, ldvr, work, lwork, info,
                    )
                }
            }
        }
    };
}

macro_rules! impl_complex_symbols {
    (
        $scalar:ty, $real:ty,
        $getrf:path, $getrs:path, $gesdd:path, $gesvd:path, $gesdd_routine:literal, $gesvd_routine:literal,
        $potrf:path, $trtrs:path, $trsm:path, $getc2:path, $gesc2:path,
        $geqrf:path, $orgqr:path, $ormqr:path, $geqp3:path, $heev:path, $geev:path,
        $geqrf_name:literal, $orgqr_name:literal, $ormqr_name:literal, $geqp3_name:literal,
        $eigh_name:literal, $geev_name:literal
    ) => {
        impl Symbols for $scalar {
            type Real = $real;
            type Complex = $scalar;

            const COMPLEX: bool = true;
            const ADJOINT: u8 = b'C';
            const GEQRF: &'static str = $geqrf_name;
            const ORGQR: &'static str = $orgqr_name;
            const ORMQR: &'static str = $ormqr_name;
            const GEQP3: &'static str = $geqp3_name;
            const EIGH: &'static str = $eigh_name;
            const GEEV: &'static str = $geev_name;

            fn one() -> Self {
                Self::new(1.0, 0.0)
            }

            fn minus_one() -> Self {
                Self::new(-1.0, 0.0)
            }

            fn real_one() -> $real {
                1.0
            }

            fn work_query_len(query: Self) -> f64 {
                query.re as f64
            }

            fn is_finite(self) -> bool {
                self.re.is_finite() && self.im.is_finite()
            }

            fn magnitude(self) -> f64 {
                self.norm() as f64
            }

            fn real_part(self) -> $real {
                self.re
            }

            fn real_to_f64(value: $real) -> f64 {
                value as f64
            }

            fn real_epsilon() -> f64 {
                <$real>::EPSILON as f64
            }

            fn complex_from_parts(re: $real, im: $real) -> Self {
                Self::new(re, im)
            }

            fn real_as_self(value: $real) -> Self {
                Self::new(value, 0.0)
            }

            fn to_complex(self) -> Self {
                self
            }

            fn apply_inverse_scale(rhs: &mut [Self], scale: $real) {
                if scale != 1.0 {
                    for value in rhs {
                        *value /= scale;
                    }
                }
            }

            fn conj_in_place(data: &mut [Self]) {
                for value in data {
                    *value = value.conj();
                }
            }

            #[cfg(not(svd_gesvd))]
            fn routine_name() -> &'static str {
                $gesdd_routine
            }

            #[cfg(svd_gesvd)]
            fn routine_name() -> &'static str {
                $gesvd_routine
            }

            common_symbols!($getrf, $getrs, $potrf, $trtrs, $getc2, $gesc2, $geqrf, $orgqr, $ormqr);

            unsafe fn svd_driver(
                jobu: u8,
                jobvt: u8,
                m: i32,
                n: i32,
                a: &mut [Self],
                lda: i32,
                s: &mut [$real],
                u: &mut [Self],
                ldu: i32,
                vt: &mut [Self],
                ldvt: i32,
                work: &mut [Self],
                lwork: i32,
                rwork: &mut [$real],
                iwork: &mut [i32],
                info: &mut i32,
            ) {
                // The driver is a compiled-in choice, as in the real case.
                #[cfg(svd_gesvd)]
                let _ = iwork;
                #[cfg(not(svd_gesvd))]
                let _ = jobvt;
                // SAFETY: as in the real case; the complex routines additionally require `rwork`
                // of the length LAPACK documents.
                #[cfg(not(svd_gesvd))]
                unsafe {
                    $gesdd(
                        jobu, m, n, a, lda, s, u, ldu, vt, ldvt, work, lwork, rwork, iwork, info,
                    );
                }
                #[cfg(svd_gesvd)]
                unsafe {
                    $gesvd(
                        jobu, jobvt, m, n, a, lda, s, u, ldu, vt, ldvt, work, lwork, rwork, info,
                    );
                }
            }

            unsafe fn trsm_right(
                lower: bool,
                transpose: bool,
                unit_diagonal: bool,
                m: i32,
                n: i32,
                a: &[Self],
                lda: i32,
                b: &mut [Self],
                ldb: i32,
            ) {
                let alpha = Self::new(1.0, 0.0);
                // SAFETY: callers validate dimensions and provide compact column-major `a` and
                // writable `b` buffers with matching leading dimensions; `num_complex` scalars
                // have the C complex layout the CBLAS binding expects.
                unsafe {
                    $trsm(
                        CBLAS_LAYOUT::CblasColMajor,
                        CBLAS_SIDE::CblasRight,
                        cblas_uplo(lower),
                        cblas_transpose(transpose),
                        cblas_diag(unit_diagonal),
                        m,
                        n,
                        (&alpha as *const Self).cast(),
                        a.as_ptr().cast(),
                        lda,
                        b.as_mut_ptr().cast(),
                        ldb,
                    );
                }
            }

            unsafe fn geqp3(
                m: i32,
                n: i32,
                a: &mut [Self],
                lda: i32,
                jpvt: &mut [i32],
                tau: &mut [Self],
                work: &mut [Self],
                lwork: i32,
                rwork: &mut [$real],
                info: &mut i32,
            ) {
                // SAFETY: callers validate the matrix, pivot, tau, real-work and workspace lengths.
                unsafe { $geqp3(m, n, a, lda, jpvt, tau, work, lwork, rwork, info) }
            }

            unsafe fn eigh_driver(
                jobz: u8,
                n: i32,
                a: &mut [Self],
                lda: i32,
                w: &mut [$real],
                work: &mut [Self],
                lwork: i32,
                rwork: &mut [$real],
                _iwork: &mut [i32],
                _liwork: i32,
                info: &mut i32,
            ) {
                // SAFETY: callers validate the matrix, value and real-work lengths and pass either
                // the query slot or the queried workspace length.
                unsafe { $heev(jobz, b'L', n, a, lda, w, work, lwork, rwork, info) }
            }

            unsafe fn geev(
                jobvr: u8,
                n: i32,
                a: &mut [Self],
                lda: i32,
                w: &mut [Self],
                _wi: &mut [$real],
                vl: &mut [Self],
                vr: &mut [Self],
                ldvr: i32,
                work: &mut [Self],
                lwork: i32,
                rwork: &mut [$real],
                info: &mut i32,
            ) {
                // SAFETY: callers validate the matrix, value, vector and real-work buffers;
                // `lwork = -1` writes only the query slot.
                unsafe {
                    $geev(
                        b'N', jobvr, n, a, lda, w, vl, 1, vr, ldvr, work, lwork, rwork, info,
                    )
                }
            }
        }
    };
}

impl_real_symbols!(
    f32,
    Complex32,
    lapack::sgetrf,
    lapack::sgetrs,
    lapack::sgesdd,
    lapack::sgesvd,
    "sgesdd",
    "sgesvd",
    lapack::spotrf,
    lapack::strtrs,
    cblas_sys::cblas_strsm,
    sgetc2_ffi,
    sgesc2_ffi,
    lapack::sgeqrf,
    lapack::sorgqr,
    lapack::sormqr,
    lapack::sgeqp3,
    lapack::ssyevd,
    lapack::sgeev,
    "sgeqrf",
    "sorgqr",
    "sormqr",
    "sgeqp3",
    "ssyevd",
    "sgeev"
);
impl_real_symbols!(
    f64,
    Complex64,
    lapack::dgetrf,
    lapack::dgetrs,
    lapack::dgesdd,
    lapack::dgesvd,
    "dgesdd",
    "dgesvd",
    lapack::dpotrf,
    lapack::dtrtrs,
    cblas_sys::cblas_dtrsm,
    dgetc2_ffi,
    dgesc2_ffi,
    lapack::dgeqrf,
    lapack::dorgqr,
    lapack::dormqr,
    lapack::dgeqp3,
    lapack::dsyevd,
    lapack::dgeev,
    "dgeqrf",
    "dorgqr",
    "dormqr",
    "dgeqp3",
    "dsyevd",
    "dgeev"
);
impl_complex_symbols!(
    Complex32,
    f32,
    lapack::cgetrf,
    lapack::cgetrs,
    lapack::cgesdd,
    lapack::cgesvd,
    "cgesdd",
    "cgesvd",
    lapack::cpotrf,
    lapack::ctrtrs,
    cblas_sys::cblas_ctrsm,
    cgetc2_ffi,
    cgesc2_ffi,
    lapack::cgeqrf,
    lapack::cungqr,
    lapack::cunmqr,
    lapack::cgeqp3,
    lapack::cheev,
    lapack::cgeev,
    "cgeqrf",
    "cungqr",
    "cunmqr",
    "cgeqp3",
    "cheev",
    "cgeev"
);
impl_complex_symbols!(
    Complex64,
    f64,
    lapack::zgetrf,
    lapack::zgetrs,
    lapack::zgesdd,
    lapack::zgesvd,
    "zgesdd",
    "zgesvd",
    lapack::zpotrf,
    lapack::ztrtrs,
    cblas_sys::cblas_ztrsm,
    zgetc2_ffi,
    zgesc2_ffi,
    lapack::zgeqrf,
    lapack::zungqr,
    lapack::zunmqr,
    lapack::zgeqp3,
    lapack::zheev,
    lapack::zgeev,
    "zgeqrf",
    "zungqr",
    "zunmqr",
    "zgeqp3",
    "zheev",
    "zgeev"
);
