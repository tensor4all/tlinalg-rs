//! Select the SVD driver, and link Accelerate (`link-accelerate`) or the *installed* Intel oneAPI
//! MKL (`link-mkl`) for this crate's own checks.
//!
//! `link-mkl` deliberately does not go through `blas-src`/`lapack-src`. Their `intel-mkl-*`
//! features route through `intel-mkl-src`, which downloads its own MKL tree, and this crate
//! already declares both source dependencies with `features = ["openblas"]`; Cargo features are
//! additive, so asking for MKL that way would activate OpenBLAS as well and the two
//! implementations would collide on the same symbols. A measurement also has to name the vendor
//! that produced the number, and which MKL that is has to be the one the host has installed.
//!
//! This emits the library side only: the search path and `-lmkl_rt`, both of which propagate to
//! dependent crates. The rest of the recipe belongs to the executable that finally links the
//! harness -- see `tlinalg-bench/build.rs`.
//!
//! `MKLROOT` is required and never guessed. `/opt/intel/oneapi/mkl/latest` is a moving symlink, and
//! a build that silently followed it could record a version it never read back.
use std::path::PathBuf;

fn feature(name: &str) -> bool {
    std::env::var_os(format!("CARGO_FEATURE_{name}")).is_some()
}

fn main() {
    println!("cargo:rerun-if-env-changed=MKLROOT");
    println!("cargo:rerun-if-env-changed=MKL_THREADING_LIB");

    // `svd_gesvd` selects the `?gesvd` driver. The injected symbol set exports no `?gesdd`, and
    // Accelerate's `zgesdd` returns wrong factors (#110); every other LAPACK uses `?gesdd`.
    println!("cargo:rustc-check-cfg=cfg(svd_gesvd)");
    if feature("PROVIDER_INJECT") || feature("ACCELERATE") {
        println!("cargo:rustc-cfg=svd_gesvd");
    }

    let mkl = feature("LINK_MKL");
    let openblas = feature("LINK_OPENBLAS") || feature("LINK_OPENBLAS_STATIC");
    let accelerate = feature("LINK_ACCELERATE");
    // Cargo features are additive, so two vendors can be asked for at once through different
    // packages -- the harness's own feature is not the only way in. The refusal belongs here,
    // where the features are declared.
    assert!(
        [mkl, openblas, accelerate].iter().filter(|&&on| on).count() <= 1,
        "link-mkl, link-openblas and link-accelerate are mutually exclusive: one binary links one \
         vendor library. Build the harness once per vendor arm and record one run per arm."
    );
    if accelerate {
        assert!(
            std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("macos"),
            "link-accelerate links Apple's Accelerate framework and needs a macOS target"
        );
        println!("cargo:rustc-link-lib=framework=Accelerate");
        return;
    }
    if !mkl {
        return;
    }
    let root = std::env::var("MKLROOT").unwrap_or_else(|_| {
        panic!(
            "link-mkl needs MKLROOT to name the installed MKL. Source the oneAPI environment \
             (`source /opt/intel/oneapi/setvars.sh`) or set MKLROOT to the MKL installation."
        )
    });
    let root = PathBuf::from(root);
    let lib = ["lib/intel64", "lib"]
        .iter()
        .map(|sub| root.join(sub))
        .find(|dir| dir.join("libmkl_rt.so").exists())
        .unwrap_or_else(|| panic!("MKLROOT={} holds no lib/libmkl_rt.so", root.display()));
    println!(
        "cargo:rerun-if-changed={}",
        lib.join("libmkl_rt.so").display()
    );
    println!("cargo:rustc-link-search=native={}", lib.display());
    println!("cargo:rustc-link-lib=dylib=mkl_rt");
}
