//! The vendor BLAS/LAPACK library this harness was built against.
//!
//! `tlinalg-blas` deliberately does not know which vendor it links: the host selects it. The
//! harness does have to know, for two reasons.
//!
//! * A row labelled `1T` is only a single-thread measurement if the vendor library also ran on one
//!   thread. LAPACK and BLAS own their threading, so the only way to make that true is to set the
//!   vendor's budget and read it back.
//! * A number is only evidence if the library that produced it can be named. `openblas_get_config`
//!   and friends report what was actually linked and which kernel the runtime dispatched to, which
//!   is not the same as the target the build was configured for.
//!
//! Two vendors are supported and they are mutually exclusive, because they are not
//! interchangeable at link time: one process can resolve `dgesvd_` to one library, and both
//! vendors define it. A binary links one vendor, and the engine row it emits is named after that
//! vendor rather than after a string chosen here.
//!
//! MKL's single dynamic library is one more step removed: the interface and threading layers are
//! selectable at run time, by the environment or by this process, and the interface layer is an
//! ABI decision rather than a tuning knob. [`prepare`] pins both before the first vendor call.
//!
//! Without a vendor feature the library is absent and every accessor says so; nothing here is
//! conditional in the public API, so the harness does not grow two shapes.

#[cfg(all(
    feature = "link-mkl",
    any(feature = "link-openblas", feature = "link-openblas-static")
))]
compile_error!(
    "link-mkl and link-openblas are mutually exclusive: one binary links one vendor library. \
     Build the harness once per vendor arm and record one run per arm."
);

/// Whether a vendor library is linked into this binary.
pub const LINKED: bool = cfg!(any(
    feature = "link-openblas",
    feature = "link-openblas-static",
    feature = "link-mkl"
));

/// The vendor this binary links, or `none`.
#[cfg(any(feature = "link-openblas", feature = "link-openblas-static"))]
const VENDOR_NAME: &str = "openblas";
#[cfg(feature = "link-mkl")]
const VENDOR_NAME: &str = "mkl";
#[cfg(not(any(
    feature = "link-openblas",
    feature = "link-openblas-static",
    feature = "link-mkl"
)))]
const VENDOR_NAME: &str = "none";

/// The engine row this binary measures the LAPACK provider under.
///
/// Derived from the linked vendor, never from a literal: a row named `lapack-openblas` in a binary
/// that linked MKL would be a provenance claim the binary cannot support.
#[cfg(any(feature = "link-openblas", feature = "link-openblas-static"))]
pub const LAPACK_ROW: &str = "lapack-openblas";
#[cfg(feature = "link-mkl")]
pub const LAPACK_ROW: &str = "lapack-mkl";
#[cfg(not(any(
    feature = "link-openblas",
    feature = "link-openblas-static",
    feature = "link-mkl"
)))]
pub const LAPACK_ROW: &str = "lapack-none";

/// How the vendor library was linked: `static`, `shared`, or `none`.
pub fn linkage() -> &'static str {
    if cfg!(feature = "link-openblas-static") {
        "static"
    } else if cfg!(any(feature = "link-openblas", feature = "link-mkl")) {
        "shared"
    } else {
        "none"
    }
}

/// What the linked vendor library says about itself.
#[derive(Debug, Clone)]
pub struct Identity {
    /// The vendor library, or `"none"`.
    pub name: &'static str,
    /// [`linkage`] of this binary.
    pub linkage: &'static str,
    /// Version, when the vendor reports one.
    pub version: Option<String>,
    /// The build configuration string, e.g. `OpenBLAS 0.3.32 NO_AFFINITY COOPERLAKE`.
    pub config: Option<String>,
    /// The kernel the runtime dispatched to, e.g. `COOPERLAKE`. A Zen host can select an
    /// Intel-named kernel; that is a fact to record, not an error to hide.
    pub corename: Option<String>,
    /// The threading implementation: `sequential`, `pthread` or `openmp`.
    pub parallel: Option<&'static str>,
    /// Processors the vendor library reports.
    pub procs: Option<usize>,
}

impl Identity {
    /// The identity as `key=value` lines, for a manifest to record verbatim.
    pub fn lines(&self) -> Vec<(&'static str, String)> {
        vec![
            ("vendor.name", self.name.to_owned()),
            ("vendor.linkage", self.linkage.to_owned()),
            ("vendor.version", self.version.clone().unwrap_or_default()),
            ("vendor.config", self.config.clone().unwrap_or_default()),
            ("vendor.corename", self.corename.clone().unwrap_or_default()),
            (
                "vendor.parallel",
                self.parallel.unwrap_or_default().to_owned(),
            ),
            (
                "vendor.procs",
                self.procs.map_or(String::new(), |p| p.to_string()),
            ),
        ]
    }
}

/// The linked vendor library's identity.
pub fn identity() -> Identity {
    Identity {
        name: VENDOR_NAME,
        linkage: linkage(),
        version: sys::config().as_deref().and_then(version_of),
        config: sys::config(),
        corename: sys::corename(),
        parallel: sys::parallel(),
        procs: sys::procs(),
    }
}

/// The vendor's own version string reduced to its version.
fn version_of(config: &str) -> Option<String> {
    if cfg!(feature = "link-mkl") {
        return mkl_version_of(config);
    }
    // `OpenBLAS 0.3.32 NO_AFFINITY COOPERLAKE` -> `0.3.32`.
    let mut fields = config.split_whitespace();
    match (fields.next(), fields.next()) {
        (Some("OpenBLAS"), Some(version)) => Some(version.to_owned()),
        _ => None,
    }
}

/// `Intel(R) oneAPI Math Kernel Library Version 2026.1-Product Build 20260612 ...` -> `2026.1`.
fn mkl_version_of(config: &str) -> Option<String> {
    let version = config.split("Version ").nth(1)?.split_whitespace().next()?;
    Some(version.split('-').next()?.to_owned())
}

/// Bring the vendor library into the state this build requires, before it is used at all.
///
/// A vendor library whose interface layer or threading layer is chosen at run time can be handed
/// a different one than the build was written against, and both are process-level inputs the
/// environment can set. What cannot be established must be refused: this is called before the
/// first vendor call, and it is not a warning.
pub fn prepare() -> Result<(), String> {
    sys::prepare()
}

/// Set the vendor library's thread budget.
///
/// This is a process-global setting in the vendor library, so the harness sets it once per
/// process from the row's declared budget and re-checks it before each measurement.
pub fn set_threads(threads: usize) -> Result<(), String> {
    if !LINKED {
        return if threads == 1 {
            Ok(())
        } else {
            Err("no vendor library is linked, so a vendor thread budget cannot be set".to_owned())
        };
    }
    sys::set_threads(threads);
    require_threads(threads)
}

/// Check that the vendor library is running on `threads` threads.
///
/// A budget that was requested but not in force would make the row's label a lie, so a mismatch is
/// an error and not a warning.
pub fn require_threads(threads: usize) -> Result<(), String> {
    if !LINKED {
        return Ok(());
    }
    match sys::threads() {
        Some(actual) if actual == threads => Ok(()),
        Some(actual) => Err(format!(
            "{} reports {actual} threads, but this row declares {threads}",
            VENDOR_NAME
        )),
        None => Err(format!("{} does not report its thread count", VENDOR_NAME)),
    }
}

/// Whether the vendor library supports introspection at all.
pub fn introspectable() -> bool {
    LINKED
}

/// MKL's single dynamic library, driven through `mkl_rt`.
///
/// MKL reports no dispatched-kernel name (`mkl_service.h` has no equivalent of
/// `openblas_get_corename`), and no processor count, so `corename` and `procs` are empty rather
/// than inferred: `MKL_Get_Max_Threads` is the thread *budget*, which is what [`require_threads`]
/// reads back, not a property of the machine.
#[cfg(feature = "link-mkl")]
mod sys {
    use core::ffi::{c_char, c_int};

    /// `MKL_INTERFACE_LP64` and `MKL_THREADING_INTEL`, from `mkl_service.h`.
    const INTERFACE_LP64: c_int = 0;
    const THREADING_INTEL: c_int = 0;

    unsafe extern "C" {
        fn MKL_Set_Interface_Layer(code: c_int) -> c_int;
        fn MKL_Set_Threading_Layer(code: c_int) -> c_int;
        fn MKL_Get_Version_String(buffer: *mut c_char, len: c_int);
        fn MKL_Set_Num_Threads(nth: c_int);
        fn MKL_Get_Max_Threads() -> c_int;
    }

    /// Pin the two layers this build's bindings and link require.
    ///
    /// The interface layer is an ABI, not a preference: these bindings are LP64 -- `i32`
    /// dimensions, pivots and `info` -- so an ILP64 layer would hand 32-bit descriptors to
    /// routines expecting 64-bit ones. `MKL_INTERFACE_LAYER` and `MKL_THREADING_LAYER` can select
    /// either from the environment, so the selection is made here, before any other MKL call, and
    /// a refusal is an error rather than a warning.
    pub fn prepare() -> Result<(), String> {
        // SAFETY: both are plain setters on the vendor's layer selection, valid until the first
        // numerical call; called from `prepare`, which runs before anything else touches MKL.
        let (interface, threading) = unsafe {
            (
                MKL_Set_Interface_Layer(INTERFACE_LP64),
                MKL_Set_Threading_Layer(THREADING_INTEL),
            )
        };
        if interface < 0 || threading < 0 {
            return Err(format!(
                "MKL refused the LP64/Intel OpenMP layers this build needs \
                 (interface={interface}, threading={threading})"
            ));
        }
        Ok(())
    }

    pub fn set_threads(threads: usize) {
        // SAFETY: a plain setter on the vendor's global thread budget. The uppercase spelling is
        // the C entry point: `mkl_set_num_threads` is a macro for it in `mkl_service.h`, and the
        // lowercase symbol `mkl_rt` also exports is the Fortran interface, which takes its
        // argument by reference.
        unsafe { MKL_Set_Num_Threads(threads as c_int) }
    }

    pub fn threads() -> Option<usize> {
        // SAFETY: no arguments; reads the vendor's own thread budget.
        let threads = unsafe { MKL_Get_Max_Threads() };
        (threads > 0).then_some(threads as usize)
    }

    pub fn procs() -> Option<usize> {
        None
    }

    /// MKL's own report, space-padded into the buffer it is handed.
    pub fn config() -> Option<String> {
        const LEN: usize = 256;
        let mut buffer = [0 as c_char; LEN];
        // SAFETY: writes at most `LEN` bytes into a buffer this process owns.
        unsafe { MKL_Get_Version_String(buffer.as_mut_ptr(), LEN as c_int) };
        let text: String = buffer
            .iter()
            .take_while(|c| **c != 0)
            .map(|&c| c as u8 as char)
            .collect();
        let text = text.trim();
        (!text.is_empty()).then(|| text.to_owned())
    }

    pub fn corename() -> Option<String> {
        None
    }

    /// The threading layer this build pins to the Intel OpenMP runtime it links with the vendor
    /// library, by [`prepare`], rather than whichever layer the environment might otherwise
    /// select.
    pub fn parallel() -> Option<&'static str> {
        Some("openmp")
    }
}

#[cfg(any(feature = "link-openblas", feature = "link-openblas-static"))]
mod sys {
    use core::ffi::{c_char, c_int, CStr};

    unsafe extern "C" {
        fn openblas_set_num_threads(num_threads: c_int);
        fn openblas_get_num_threads() -> c_int;
        fn openblas_get_num_procs() -> c_int;
        fn openblas_get_config() -> *const c_char;
        fn openblas_get_corename() -> *const c_char;
        fn openblas_get_parallel() -> c_int;
    }

    /// A `char *` the vendor library owns and keeps alive.
    fn string_of(pointer: *const c_char) -> Option<String> {
        if pointer.is_null() {
            return None;
        }
        // SAFETY: the vendor library returns a pointer to a NUL-terminated string that stays valid
        // for the lifetime of the process.
        let text = unsafe { CStr::from_ptr(pointer) }
            .to_string_lossy()
            .into_owned();
        Some(text)
    }

    pub fn set_threads(threads: usize) {
        // SAFETY: a plain setter on the vendor's global thread count; any positive value is
        // accepted and clamping is the vendor's business.
        unsafe { openblas_set_num_threads(threads as c_int) }
    }

    pub fn prepare() -> Result<(), String> {
        Ok(())
    }

    pub fn threads() -> Option<usize> {
        // SAFETY: no arguments, no state touched beyond reading the vendor's thread count.
        let threads = unsafe { openblas_get_num_threads() };
        (threads > 0).then_some(threads as usize)
    }

    pub fn procs() -> Option<usize> {
        // SAFETY: as `threads`.
        let procs = unsafe { openblas_get_num_procs() };
        (procs > 0).then_some(procs as usize)
    }

    pub fn config() -> Option<String> {
        // SAFETY: returns a pointer to a static string owned by the vendor library.
        string_of(unsafe { openblas_get_config() })
    }

    pub fn corename() -> Option<String> {
        // SAFETY: as `config`.
        string_of(unsafe { openblas_get_corename() })
    }

    pub fn parallel() -> Option<&'static str> {
        // SAFETY: no arguments, returns an enumerator: 0 sequential, 1 pthread, 2 openmp.
        match unsafe { openblas_get_parallel() } {
            0 => Some("sequential"),
            1 => Some("pthread"),
            2 => Some("openmp"),
            _ => None,
        }
    }
}

#[cfg(not(any(
    feature = "link-openblas",
    feature = "link-openblas-static",
    feature = "link-mkl"
)))]
mod sys {
    pub fn set_threads(_threads: usize) {}

    pub fn prepare() -> Result<(), String> {
        Ok(())
    }
    pub fn threads() -> Option<usize> {
        None
    }
    pub fn procs() -> Option<usize> {
        None
    }
    pub fn config() -> Option<String> {
        None
    }
    pub fn corename() -> Option<String> {
        None
    }
    pub fn parallel() -> Option<&'static str> {
        None
    }
}

#[cfg(test)]
mod tests {
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
}
