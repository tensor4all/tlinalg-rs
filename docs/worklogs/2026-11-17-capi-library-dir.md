# `c_abi`: find the shared library under Cargo 1.100's build-directory layout

## Problem

`crates/tlinalg-capi/tests/c_abi.rs` resolved the target directory by walking two levels up from
its own executable: `target/<profile>/deps/<test>` -> `target/<profile>`, where
`cargo build -p tlinalg-capi` writes `libtlinalg_capi.<ext>`. Cargo 1.100 puts the test executable
in `target/<profile>/build/<pkg>/<hash>/out/` and the shared library in a *different*
`<hash>/out/` of the same package. The walk then resolved to
`target/<profile>/build/tlinalg-capi/<hash>` and every run failed with "libtlinalg_capi is not
built ...". Observed on stable `1.100.0` in run 43998854833 on PR #116; reproduced locally on
`227dba0` with `cargo 1.100.0` (`build/.../out` executables) and not with `cargo 1.99.0`
(`deps/`).

## Fix

`library_dir()` walks up from the executable and, at each level, also checks
`<level>/build/tlinalg-capi/*/out/`. The first directory holding the platform's shared library is
used for `-L` and the rpath. Both Cargo layouts pass without special-casing a version.

## Verification

- `cargo 1.100.0`: `cargo build --profile ci -p tlinalg-capi` then
  `cargo test --profile ci -p tlinalg-capi --test c_abi` -> 2 passed.
- `cargo 1.99.0`: the same two commands -> 2 passed (old layout).
- `cargo fmt --all -- --check` and `cargo clippy -p tlinalg-capi --tests -- -D warnings` pass.

## Remaining risk

The search stops at the first matching directory. A target tree that holds a stale
`target/<profile>/libtlinalg_capi.*` beside a fresh build-directory tree would pick the stale
library; CI builds one profile per job and does not have both.
