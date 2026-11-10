<!-- unified contribution policy: ../../docs/contributing/rust-repositories.md in tensor4all/tensor4all-meta -->
# Contributing to tlinalg-rs

The rules that apply to every Rust repository in `tensor4all` are in one place:
**[tensor4all-meta · docs/contributing/rust-repositories.md](https://github.com/tensor4all/tensor4all-meta/blob/main/docs/contributing/rust-repositories.md)**.
Read that first — it covers the issue-first workflow, what a pull request body has to contain, red-first tests, the
gates every repository runs, the review requirement, and the version policy.

This file only records what is specific to `tlinalg-rs`:

## Local facts

- **MSRV**: `rust-version = "1.89"` in the workspace `Cargo.toml`, inherited by every crate
  (`rust-version.workspace = true`). CI pins `stable` only and has no `msrv` job, so the claim is not checked by CI.
- **Feature combinations CI builds**: default features (the faer provider, nothing linked) in the unlinked run on
  macOS, then two linked lanes on Linux: `tlinalg-blas/link-openblas,tlinalg-parity/link-openblas`, and the same plus
  `tlinalg-blas/provider-inject` (whose symbol set exports `?gesvd` but not `?gesdd`). `clippy` lints default features
  and the `provider-inject` set. `link-openblas` builds OpenBLAS from source through `openblas-src`, so no system
  package is needed; `--all-features` is not a lane here.
- **Platforms claimed**: `ubuntu-latest` for the linked tests (fmt, `deny`, clippy and `doc` also run there) and
  `macos-latest` for the unlinked run, where the LAPACK tests are skipped.
- **Local gate**:

  ```sh
  cargo build --profile ci -p tlinalg-capi && cargo test --workspace --profile ci
  ```

  The `cargo build` is the step people forget: `crates/tlinalg-capi/tests/c_abi.rs` links the built shared library, and
  `cargo test` alone builds only the rlib, so the C ABI test fails with `libtlinalg_capi is not built`. Add the
  `tlinalg-blas/link-openblas,tlinalg-parity/link-openblas` run from CI when the change touches the LAPACK or provider
  path, and a C++ compiler because the header test compiles as C and C++.

## Anything here that departs from the shared rules

One does: `rust-version = "1.89"` is declared but no `msrv` job builds on 1.89, so the claim can drift with a
dependency bump. That is a gap to close, not an exemption from the rest of the gates.

This repository adds a `deny` lane (a direct `rayon` dependency anywhere but `threadteam` is rejected) and the
ratio-gated `overhead_smoke` benchmark.
