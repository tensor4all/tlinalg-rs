# Work, and the rate that prices it

Two questions, two answers, deliberately not merged:

* **How much work is one item?** A count of weighted multiply-adds, from dtype and shape only
  (`estimate_work::<T>(shape args)` in every family). It is the number the lane and width decisions
  of this crate consume, and it does not move with the machine, the batch, the width or a
  calibration.
* **How long does that take here?** The count divided by a rate this host measured. That rate is a
  guess about a machine, so it is published *next to* the count rather than in place of it
  (`svd::estimate`, `qr::estimate`, `eigh::estimate` → `Estimate { work, duration }`).

This is the shared model agreed in `tensor4all/tlinalg-rs#88` and `tensor4all/tcontract-rs#163`:
`tlinalg` owns the rate, each crate owns its own multiply-add model (`tlinalg` for
`svd`/`qr`/`eigh`, `tcontract` from `JobMeta.work`), and neither re-derives the other's formula nor
re-selects a strategy inside an estimate.

## Why a rate is measured, not written down

A compile-time constant is wrong on exactly the machines the estimate is for: the same curve priced
at a Zen 2 rate and reused on Zen 5 moved a decision by about 2%, and a GEMM-shaped multiply-add
count underestimates a small SVD by more than 30× because its wall clock is a serial phase, not a
product. So `estimate::Rates::calibrate()` measures three rates on the calling thread (width one, by
definition) and returns them:

| rate | probe | priced at |
|---|---|---|
| `gemm` | one 1024-cube `f64` product | `n^3` multiply-adds |
| `bidiagonal` | this crate's `svd` of a 128-cube `f64` matrix | `svd::estimate_work` |
| `panel` | this crate's `qr` of a 128-cube `f64` matrix | `qr::estimate_work` |

Each is the family's own work model for the probe shape over the best of three timed calls. Two
consequences are stated in the API docs and are part of the contract:

* It is a **whole-call** rate: the probe's allocate-workspace, form-`Q`, reconstruction-check and
  retry behaviour is inside the measurement, because that is what a caller's call pays.
* It is calibrated where this crate's own width decisions live (the 128-cube neighbourhood), and it
  is extrapolation elsewhere. An `eigh` borrows the `svd` probe's rate as a proxy.

## Why calibration is opt-in

The probe is short but not free (about 0.27 s in a release build on the development machine, of
which the 1024-cube product is roughly 80%); a test suite creates many short-lived `Exec` values and
must pay nothing for a number it never reads. `calibrate()` is therefore the only entry point that
measures anything, and everything else in the crate behaves identically without it. An uncalibrated
estimate reports `duration: None` — never a made-up time.

The value is the caller's to hold and reuse. It is not stored in the `Exec`: that type belongs to
`tcontract-exec`, and this crate does not widen it for its own bookkeeping. Keep one `Rates` next to
the token it describes, for as long as that host and CPU domain run the work.

## Advisory, and enforced as such

A duration is the model's output, not a measurement of the item the caller asked about, and it says
which rate it came from. No lane, width or strategy decision in this crate may be made from it: the
decisions consume `batch::Work`, and `crates/tlinalg/src/batch/tests/estimates.rs` asserts
bit-for-bit that the planner receives exactly the public work count. Nothing on that path can reach a
`Rates`, so a calibration cannot narrow a route, grant a team or move a lane.

A caller that does not want this host's guess can take `work` and divide by its own measurement;
that is the whole reason both numbers are published.

## Evidence

`crates/tlinalg/tests/estimates.rs` fixes the model without asserting a single wall-clock value: the
work formulas, monotonicity in the shape, the scalar weight (a complex item is four real ones), that
`Estimate::work` is bit-identical with and without a calibration, that an uncalibrated estimate has
no duration, that a calibrated one is exactly its own work over its own rate to the nanosecond a
`Duration` can hold, that an empty item is `Some(Duration::ZERO)`, and that an unrepresentable time
saturates at `Duration::MAX` instead of panicking.
