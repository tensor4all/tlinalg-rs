//! Compiles `tests/c/abi_test.c` against the shipped header, links it to the built
//! `libtlinalg_capi` shared library and runs it. Also checks the header is valid C and C++.
//!
//! Skipped, with a message, when no C compiler is installed.

use std::path::{Path, PathBuf};
use std::process::Command;

/// The directory that holds this crate's shared library.
///
/// Cargo 1.99 puts the test executable in `target/<profile>/deps/` and the
/// shared library directly in `target/<profile>/`. Cargo 1.100 puts the test
/// executable in `target/<profile>/build/<pkg>/<hash>/out/` and the library in
/// a *different* `<hash>/out/` of the same package. Search upward for either
/// layout instead of encoding one Cargo version's directory choice.
fn library_dir() -> Option<PathBuf> {
    let exe = std::env::current_exe().expect("exe");
    let mut dir = exe.parent()?.to_path_buf();
    loop {
        if has_shared_library(&dir) {
            return Some(dir);
        }
        if let Ok(entries) = dir.join("build").join("tlinalg-capi").read_dir() {
            for entry in entries.flatten() {
                let out = entry.path().join("out");
                if has_shared_library(&out) {
                    return Some(out);
                }
            }
        }
        dir = dir.parent()?.to_path_buf();
    }
}

fn compiler() -> Option<String> {
    let cc = std::env::var("CC").unwrap_or_else(|_| "cc".into());
    Command::new(&cc).arg("--version").output().ok()?;
    Some(cc)
}

/// Whether a shared library of this crate is present in `target`.
fn has_shared_library(target: &Path) -> bool {
    [
        "libtlinalg_capi.so",
        "libtlinalg_capi.dylib",
        "tlinalg_capi.dll",
    ]
    .iter()
    .any(|name| target.join(name).exists())
}

#[test]
fn c_program_links_and_runs() {
    let Some(cc) = compiler() else {
        eprintln!("skipping: no C compiler");
        return;
    };
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let include = root.join("include");
    let target = library_dir().unwrap_or_else(|| {
        panic!(
            "libtlinalg_capi is not built under {}: run `cargo build -p tlinalg-capi` first",
            std::env::current_exe().expect("exe").display()
        )
    });

    let out = std::env::temp_dir().join(format!("tlinalg_abi_test_{}", std::process::id()));
    let status = Command::new(&cc)
        .args(["-std=c11", "-Wall", "-Werror", "-o"])
        .arg(&out)
        .arg(root.join("tests/c/abi_test.c"))
        .arg(format!("-I{}", include.display()))
        .arg(format!("-L{}", target.display()))
        .arg(format!("-Wl,-rpath,{}", target.display()))
        .args(["-ltlinalg_capi", "-lm"])
        .status()
        .expect("cc");
    assert!(status.success(), "C compile failed");
    let run = Command::new(&out).output().expect("run");
    let _ = std::fs::remove_file(&out);
    assert!(
        run.status.success(),
        "abi_test failed: {}{}",
        String::from_utf8_lossy(&run.stdout),
        String::from_utf8_lossy(&run.stderr)
    );
}

#[test]
fn the_header_compiles_as_c_and_cxx() {
    let Some(cc) = compiler() else {
        eprintln!("skipping: no C compiler");
        return;
    };
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let include = root.join("include");
    let src = root.join("tests/c/headers_include.c");
    for (lang, flag) in [("c", "-std=c11"), ("c++", "-std=c++17")] {
        let status = Command::new(&cc)
            .args([flag, "-x", lang, "-Wall", "-Werror", "-fsyntax-only"])
            .arg(format!("-I{}", include.display()))
            .arg(&src)
            .status()
            .expect("cc");
        assert!(status.success(), "header failed to compile as {lang}");
    }
}
