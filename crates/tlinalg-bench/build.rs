//! Final-link configuration for the MKL vendor arm.
//!
//! A library cannot configure another package's executable: `cargo:rustc-link-arg` is restricted to
//! linkable targets of the package whose build script emits it, so the two things MKL additionally
//! needs at the executable link line are asked for here, by the package `tlbench` belongs to.
//!
//! MKL's default threading layer is Intel OpenMP, and `libmkl_intel_thread.so` is loaded by MKL at
//! run time with an undefined `omp_get_num_procs`. Unless `libiomp5` is already in the process's
//! global symbol scope, the first threaded call dies with a symbol-lookup error. Linking it
//! normally is not enough: nothing references an `iomp5` symbol, so `--as-needed` drops it. Hence
//! `--no-as-needed` around an exact-filename `-l:` reference.
//!
//! The run paths are likewise not optional. `rustc` does not turn a native search path into an
//! `-rpath`, so without them the binary needs `LD_LIBRARY_PATH` arranged for it -- and a harness
//! that only runs under a loader search path cannot be executed the way the campaign executes it.
//!
//! Both inputs are declared, never inherited: `MKLROOT` names the vendor, and `MKL_THREADING_LIB`
//! names the threading runtime to link eagerly.
use std::path::PathBuf;

fn main() {
    println!("cargo:rerun-if-env-changed=MKLROOT");
    println!("cargo:rerun-if-env-changed=MKL_THREADING_LIB");
    if std::env::var_os("CARGO_FEATURE_LINK_MKL").is_none() {
        return;
    }
    // The two variables name files, and `rerun-if-env-changed` does not notice a library replaced
    // behind an unchanged path -- an MKL upgraded in place would otherwise be linked stale.
    let lib = require_mkl_lib();
    let iomp = PathBuf::from(std::env::var("MKL_THREADING_LIB").unwrap_or_else(|_| {
        panic!(
            "link-mkl needs MKL_THREADING_LIB, the full path to the threading runtime to link \
             with the vendor library, for example \
             /opt/intel/oneapi/compiler/2026.1/lib/libiomp5.so"
        )
    }));
    let iomp_dir = iomp
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or_else(|| {
            panic!(
                "MKL_THREADING_LIB={} is not a path with a directory",
                iomp.display()
            )
        });
    let iomp_name = iomp
        .file_name()
        .unwrap_or_else(|| panic!("MKL_THREADING_LIB={} names no file", iomp.display()));
    assert!(
        iomp.exists(),
        "MKL_THREADING_LIB={} does not exist",
        iomp.display()
    );

    println!("cargo:rustc-link-search=native={}", iomp_dir.display());
    println!("cargo:rustc-link-arg=-Wl,--push-state,--no-as-needed");
    println!("cargo:rustc-link-arg=-l:{}", iomp_name.to_string_lossy());
    println!("cargo:rustc-link-arg=-Wl,--pop-state");
    println!("cargo:rustc-link-arg=-Wl,-rpath,{}", lib.display());
    println!("cargo:rustc-link-arg=-Wl,-rpath,{}", iomp_dir.display());
}

fn require_mkl_lib() -> PathBuf {
    let root = std::env::var("MKLROOT").unwrap_or_else(|_| {
        panic!(
            "link-mkl needs MKLROOT to name the installed MKL. Source the oneAPI environment \
             (`source /opt/intel/oneapi/setvars.sh`) or set MKLROOT to the MKL installation."
        )
    });
    let root = PathBuf::from(root);
    ["lib/intel64", "lib"]
        .iter()
        .map(|sub| root.join(sub))
        .find(|dir| dir.join("libmkl_rt.so").exists())
        .unwrap_or_else(|| panic!("MKLROOT={} holds no lib/libmkl_rt.so", root.display()))
}
