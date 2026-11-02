//! What a work model's numbers become on this host: a measured rate, and the estimate it prices.
//!
//! A family's work query (`svd::estimate_work` and friends) is the crate's own multiply-add model
//! for one item: a count, independent of machine, width and strategy. Turning it into a time needs
//! one more number — how fast this host turns that kind of work into seconds — and that number is a
//! measurement, not a constant: a rate measured on one CPU generation is wrong on the next.
//!
//! [`Rates::calibrate`] measures it once, on the calling thread (width one, by definition), with a
//! short probe per operation class, and returns the result as a plain, reusable value. Nothing
//! measures unless a caller asks: the probes sit behind this one opt-in entry point, so a
//! short-lived `Exec` in a test suite pays nothing, and every entry point of the crate runs exactly
//! as before without it.
//!
//! The value is the caller's to hold and reuse. It is not stored in the `Exec`: that type belongs to
//! `tcontract-exec`, and this crate does not widen it for its own bookkeeping — so calibrate once
//! and keep the result next to the token it describes, for as long as that host and CPU domain are
//! the ones running the work.
//!
//! # Advisory
//!
//! [`Estimate::duration`] is `work` divided by a rate: the model's output, not a measurement of the
//! item the caller asked about. It is advisory only. No width, lane or strategy decision in this
//! crate is made from it — the width and lane decisions consume the work count itself, which no
//! calibration can change — and a caller that wants seconds from its own measurement should use the
//! work count and its own rate.

use core::time::Duration;

use strided_view::RawStridedRef;
use tcontract_exec::Exec;

use crate::{qr, svd, Op};

/// Weighted multiply-adds per nanosecond at width one, measured on this host.
///
/// One unit is one real scalar multiply-add; a complex one is four, which is why a work count
/// already carries the scalar's weight and a rate is scalar-free. Each rate is an **effective
/// whole-call** rate: the family's whole work model for the probe shape over the whole probe call,
/// allocator and everything (see [`Rates::calibrate`]). It is not the throughput of one isolated
/// phase, and it is not a property of a CPU generation — it is what this host measured just now.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Rates {
    gemm: f64,
    bidiagonal: f64,
    panel: f64,
}

impl Rates {
    /// Measure the rates on this host with one short probe per class.
    ///
    /// * [`Rates::gemm`]: one 1024-cube `f64` product, priced at `n^3` multiply-adds.
    /// * [`Rates::bidiagonal`]: this crate's own thin `svd` of a 128-cube `f64` matrix, priced at
    ///   `svd::estimate_work`.
    /// * [`Rates::panel`]: this crate's own `qr` of a 128-cube `f64` matrix, priced at
    ///   `qr::estimate_work`.
    ///
    /// Each rate is that work divided by the best of three timed calls, so a slow first call does
    /// not set the number. The probes run on the calling thread, so the caller's budget and team
    /// have no part in the number — it is a width-one rate by definition.
    ///
    /// Two things follow from measuring whole calls, and both bound what a duration can mean. The
    /// `svd` probe includes the reconstruction check and, on a failure, the QR retry, and the `qr`
    /// probe forms `Q`; and the factorizations are small on purpose, because one rate cannot follow
    /// a model across every shape (a large SVD converts more of its work per multiply-add than a
    /// small one, since blocking and cache behaviour change). The estimate is therefore closest
    /// around the `128`-cube neighbourhood the crate's own width decisions live in, and is
    /// extrapolation away from it.
    ///
    /// The probes allocate their workspace per call, exactly as the `Vec` entry points do, so the
    /// rate describes the call a caller actually makes.
    ///
    /// # Examples
    ///
    /// Calibrate once and reuse the result; the estimate takes it as a plain reference.
    ///
    /// ```no_run
    /// use tlinalg::estimate::Rates;
    /// use tlinalg::svd;
    ///
    /// let rates = Rates::calibrate();
    /// assert!(rates.gemm() > 0.0);
    /// let estimate = svd::estimate::<f64>(128, 128, Some(&rates));
    /// assert!(estimate.duration.is_some());
    /// ```
    pub fn calibrate() -> Self {
        Self {
            gemm: gemm_rate(),
            bidiagonal: svd_rate(),
            panel: qr_rate(),
        }
    }

    /// Weighted multiply-adds per nanosecond of a dense product: the rate a GEMM-shaped plan of a
    /// crate above this one is priced at.
    pub fn gemm(&self) -> f64 {
        self.gemm
    }

    /// Weighted multiply-adds per nanosecond of an `svd`: the effective whole-call rate of this
    /// crate's own `svd` probe, so it includes the workspace allocation, the reconstruction check
    /// and a possible QR retry. An `eigh` borrows it as a proxy — its tridiagonal reduction is the
    /// same kind of serial phase, which is not the same as measuring it.
    pub fn bidiagonal(&self) -> f64 {
        self.bidiagonal
    }

    /// Weighted multiply-adds per nanosecond of a `qr`: the effective whole-call rate of this
    /// crate's own `qr` probe, so it includes forming `Q` as well as the factor panels.
    pub fn panel(&self) -> f64 {
        self.panel
    }
}

/// One work model's output: the work it counted, and this host's time for it if calibrated.
///
/// `work` is the number [`crate::svd::estimate_work`] and its siblings publish, in weighted
/// multiply-adds. `duration` is `work` divided by a calibrated [`Rates`] entry; it is `None` when
/// the caller supplied no calibration, so an uncalibrated caller never receives a made-up time. Use
/// `work` with a rate of your own when this host's calibration does not apply.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Estimate {
    /// Weighted multiply-adds of one item: one real multiply-add is one unit, a complex one four.
    pub work: f64,
    /// `work` over a calibrated rate, or `None` when nothing was calibrated.
    ///
    /// Advisory: a model's output, not a measurement. Do not schedule from it. A work count too
    /// large for a `Duration` saturates at [`Duration::MAX`].
    pub duration: Option<Duration>,
}

impl Estimate {
    /// `work` priced at `rate` work units per nanosecond, or left unpriced when `rate` is `None`.
    pub(crate) fn of(work: f64, rate: Option<f64>) -> Self {
        Self {
            work,
            duration: rate.map(|rate| ns_duration(work / rate)),
        }
    }
}

/// A nanosecond count as a `Duration`. A count that cannot be a duration is zero, and one beyond
/// what `Duration` holds saturates instead of panicking.
fn ns_duration(ns: f64) -> Duration {
    let seconds = ns * 1e-9;
    if seconds.is_nan() || seconds <= 0.0 {
        Duration::ZERO
    } else if seconds >= Duration::MAX.as_secs_f64() {
        Duration::MAX
    } else {
        Duration::from_secs_f64(seconds)
    }
}

/// The dense product the GEMM probe runs: a cube, so the probe is one `n^3`-multiply-add call.
const GEMM_PROBE: usize = 1024;
/// The cube the factorization probes factorize (see [`Rates::calibrate`] for why it is small).
const FACTOR_PROBE: usize = 128;
/// Timed calls per probe. The best of them is taken, so one slow call does not set a rate.
const PROBE_RUNS: usize = 3;

fn gemm_rate() -> f64 {
    let n = GEMM_PROBE;
    // Values of order one and no structure: the product neither overflows nor short-circuits into
    // a special path (a zero or identity operand would).
    let pattern = |row: usize, col: usize| 1.0 + ((row * 3 + col * 5) % 7) as f64 * 0.25;
    let lhs = faer::Mat::<f64>::from_fn(n, n, &pattern);
    let rhs = faer::Mat::<f64>::from_fn(n, n, |row, col| pattern(col, row));
    let mut dst = faer::Mat::<f64>::zeros(n, n);
    rate((n as f64).powi(3), || {
        faer::linalg::matmul::matmul(
            dst.as_mut(),
            faer::Accum::Replace,
            lhs.as_ref(),
            rhs.as_ref(),
            1.0,
            faer::Par::Seq,
        )
    })
}

/// The rate this crate's own thin `svd` measures: its work model for a 128-cube probe over the best
/// of [`PROBE_RUNS`] calls.
fn svd_rate() -> f64 {
    let n = FACTOR_PROBE;
    let a = probe_matrix(n);
    let (dims, strides) = ([n, n], [1isize, n as isize]);
    let (mut u, mut s, mut vt) = (Vec::new(), Vec::new(), Vec::new());
    rate(svd::estimate_work::<f64>(n, n), || {
        svd::svd(
            Op::Svd,
            RawStridedRef::new(&a, &dims, &strides, 0).unwrap(),
            false,
            &mut u,
            &mut s,
            &mut vt,
            &mut Exec::serial(),
        )
        .expect("the probe matrix is positive definite")
    })
}

/// The rate this crate's own `qr` measures, on the same probe and statistic as [`svd_rate`].
fn qr_rate() -> f64 {
    let n = FACTOR_PROBE;
    let a = probe_matrix(n);
    let (dims, strides) = ([n, n], [1isize, n as isize]);
    let (mut q, mut r) = (Vec::new(), Vec::new());
    rate(qr::estimate_work::<f64>(n, n), || {
        qr::qr(
            Op::Qr,
            RawStridedRef::new(&a, &dims, &strides, 0).unwrap(),
            &mut q,
            &mut r,
            &mut Exec::serial(),
        )
        .expect("the probe matrix is positive definite")
    })
}

/// A deterministic, positive-definite, column-major `n x n` matrix — `diag(1.5 + row)` shifted by
/// the rank-one `0.5·11ᵀ` — so a probe measures its factorization and not how badly conditioned a
/// random input happened to be.
fn probe_matrix(n: usize) -> Vec<f64> {
    (0..n * n)
        .map(|index| {
            let (row, col) = (index % n, index / n);
            if row == col {
                2.0 + row as f64
            } else {
                0.5
            }
        })
        .collect()
}

/// `work` over the best of [`PROBE_RUNS`] timed calls of `probe`, in work units per nanosecond.
fn rate(work: f64, mut probe: impl FnMut()) -> f64 {
    let mut best = f64::INFINITY;
    for _ in 0..PROBE_RUNS {
        let start = std::time::Instant::now();
        probe();
        // A slice under a nanosecond is not resolvable, and is priced as one, so a rate is always
        // positive and finite.
        best = best.min((start.elapsed().as_nanos() as f64).max(1.0));
    }
    work / best
}

#[cfg(test)]
mod tests;
