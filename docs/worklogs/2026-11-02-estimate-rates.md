# Calibrated rates and the advisory duration

Date: 2026-11-02

## Decision

`tensor4all/tlinalg-rs#88` and `tensor4all/tcontract-rs#163` settled the cross-crate half that the
previous step (`docs/worklogs/2026-11-01-estimate-work.md`) left open: `tlinalg` now owns the rate,
each crate keeps its own multiply-add model, and both numbers are published together. This entry
records how that landed here and reverses one sentence of the previous worklog — "rate ownership
belongs in the caller or a crate above the providers" — deliberately; the older entry is left as the
record of what was true then.

The rate is a plain value, `estimate::Rates`, produced by `Rates::calibrate()` and held by the
caller, and `svd::estimate`, `qr::estimate` and `eigh::estimate` return
`Estimate { work, duration }`: the same count `estimate_work` already published, next to
`work / rate`. `duration` is `None` without a calibration, so no caller is handed a made-up time.

Two parts of the decision could not be taken literally, and both are documented where they bite:

* **"cached in `Exec`".** `Exec` is `tcontract_exec::Exec`. A rate field on it is a tcontract-exec
  change and a publish chain, and the repository rule is that this crate does not widen the host's
  interface for its own bookkeeping. The cache is therefore the caller's `Rates` value, which the
  API doc says to keep next to the token it describes.
* **`svd_estimate(..)` as a free name.** Every existing estimate is module-qualified
  (`svd::estimate_work`), so the new entry point is `svd::estimate`; a second naming path for the
  same thing is what the repository rules forbid. `op` and `full` are not parameters because neither
  enters the work model.

Calibration is opt-in because the probe is short but not free: a test suite creates many short-lived
`Exec` values and must not pay for a number it never reads. Nothing in the crate behaves differently
without it.

## The rates

| rate | probe | priced at |
|---|---|---|
| `gemm` | one 1024-cube `f64` product | `n^3` muladds |
| `bidiagonal` | this crate's thin `svd` of a 128-cube `f64` matrix | `svd::estimate_work` |
| `panel` | this crate's `qr` of a 128-cube `f64` matrix | `qr::estimate_work` |

Each is an effective whole-call rate: the probe's workspace allocation, `Q` formation,
reconstruction check and possible QR retry are inside the measurement, because a caller's call pays
them. The best of three timed calls per probe is taken, so one slow call does not set the rate. An
`eigh` borrows the `svd` probe's rate as an explicit proxy.

## Measured cost (AMD EPYC 7702P, one core, `-j 8`, nothing else running)

| build | one `Rates::calibrate()` | `gemm` | `bidiagonal` | `panel` |
|---|---|---|---|---|
| `--release` | 270 ms | 13.25 muladds/ns | 0.462 | 1.48 |
| `test`/`ci` profile (opt-level 0) | 2030 ms | 13.09 | 0.0039 | 0.0436 |

The 1024-cube product is about 80% of the release cost. The unoptimized figure is dominated by the
`svd` probe, and it is the one number to watch if the test gate ever feels slow: the calibration runs
once in `crates/tlinalg/tests/estimates.rs`.

## Verification

`cargo fmt --check`, `cargo clippy --workspace --all-targets -D warnings`, `RUSTDOCFLAGS=-D warnings
cargo doc --no-deps`, `cargo test -p tlinalg` (229 passed, 1 pre-existing ignored), and
`cargo test --workspace --profile ci` after `cargo build --profile ci -p tlinalg-capi`. The one
failing test on this host is `tlinalg-capi`'s `the_header_compiles_as_c_and_cxx`, which needs a C++
frontend (`cc1plus`) this machine does not have; its C half passes.

The public tests assert the model without asserting a wall-clock value: the three formulas, the
monotonic shape dependence, the complex weight of four, that `Estimate::work` is bit-identical with
and without a calibration (so no rate can reach a decision), `None` without a calibration,
`Some(work / rate)` to the nanosecond `Duration` can hold, `Some(Duration::ZERO)` for an empty item,
and `Duration::MAX` saturation for a count too large to represent. `batch/tests/estimates.rs`
continues to observe every actual planner input bit for bit, which is what holds the advisory
boundary.

## Scope

Only the native `tlinalg` API, its tests and its documentation change: no dependency, no C binding,
no version bump, and no entry point signature or result changed. The `tcontract` side is a separate
change in its own repository; only the `gemm` rate is aimed at it, and it now exists.
