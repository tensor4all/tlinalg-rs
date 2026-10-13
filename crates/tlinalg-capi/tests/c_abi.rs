//! Compiles `tests/c/abi_test.c` against the shipped header, links it to the built
//! `libtlinalg_capi` shared library and runs it. Also checks the header is valid C and C++.
//!
//! Skipped, with a message, when no C compiler is installed.

use std::path::{Path, PathBuf};
use std::process::Command;

fn target_dir() -> PathBuf {
    // .../target/<profile>/deps/c_abi-<hash>
    let exe = std::env::current_exe().expect("exe");
    exe.parent()
        .and_then(|p| p.parent())
        .expect("target dir")
        .to_path_buf()
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
    let target = target_dir();
    assert!(
        has_shared_library(&target),
        "libtlinalg_capi is not built in {}: run `cargo build -p tlinalg-capi` first",
        target.display()
    );

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
