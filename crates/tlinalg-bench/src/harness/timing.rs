//! Wall-clock priming, repetition policy and the chronological calls behind a row.

use std::time::{Duration, Instant};

/// How one row's chronological calls are reduced to the number the page prints.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Statistic {
    /// The best call, which is what the v1 pages reported.
    Best,
    /// The median call, which the timing-v2 contract asks for.
    Median,
}

impl Statistic {
    pub fn label(self) -> &'static str {
        match self {
            Self::Best => "best",
            Self::Median => "median",
        }
    }
}

/// Result of one measured row.
#[derive(Clone, Debug)]
pub struct Timing {
    /// Every measured call, in the order it happened. Retained so a reader can see the spread the
    /// single number hides; the campaign's evidence check rejects a run without them.
    pub samples_ms: Vec<f64>,
    /// The value the row reports: best or median of `samples_ms`, per the requested statistic.
    pub total_ms: f64,
}

impl Timing {
    fn from_samples(mut samples_ms: Vec<f64>, statistic: Statistic) -> Self {
        let total_ms = match statistic {
            Statistic::Best => samples_ms.iter().copied().fold(f64::INFINITY, f64::min),
            Statistic::Median => median(&mut samples_ms),
        };
        Self {
            samples_ms,
            total_ms,
        }
    }
}

fn median(samples: &mut [f64]) -> f64 {
    if samples.is_empty() {
        return 0.0;
    }
    samples.sort_by(|a, b| a.partial_cmp(b).expect("finite samples"));
    let mid = samples.len() / 2;
    if samples.len().is_multiple_of(2) {
        (samples[mid - 1] + samples[mid]) / 2.0
    } else {
        samples[mid]
    }
}

/// Prime for a wall-clock duration, then measure `reps` operations and keep every call.
///
/// `prepare` is deliberately called outside each clocked region: fixture restoration is host work,
/// not provider work (unlike the old Criterion closures, which restored inside their boundary).
pub fn measure(
    prime_ms: u64,
    reps: usize,
    statistic: Statistic,
    mut prepare: impl FnMut() -> Result<(), String>,
    mut operation: impl FnMut() -> Result<(), String>,
) -> Result<Timing, String> {
    let until = Instant::now() + Duration::from_millis(prime_ms);
    while Instant::now() < until {
        prepare()?;
        operation()?;
    }
    let mut samples = Vec::with_capacity(reps.max(1));
    for _ in 0..reps.max(1) {
        prepare()?;
        let start = Instant::now();
        operation()?;
        samples.push(start.elapsed().as_secs_f64() * 1_000.0);
    }
    Ok(Timing::from_samples(samples, statistic))
}
