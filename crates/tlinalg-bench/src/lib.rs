//! Fixtures for the kernel-level benchmarks in `benches/kernels.rs`.
//!
//! The benchmarks call each provider's batched entry points directly, the way a host does after it
//! has selected its execution pool: inputs are compact `[n, n, batch]` descriptors, output vectors are
//! reused across iterations (a host hands back pooled buffers), and the LAPACK provider gets a
//! recycling [`Workspace`], so steady-state allocation is not what is measured.
//!
//! These measure the numerical kernels only. Tensor construction, dtype dispatch, session entry
//! and pool checkout are tenferro's; route-level performance lives in `tenferro-benchmark`.

use core::any::{Any, TypeId};
use core::mem::{ManuallyDrop, MaybeUninit};
use core::num::NonZeroUsize;
use std::collections::HashMap;

use strided_view::RawStridedRef;
use tlinalg_blas::{IndexWorkspace, Workspace};

pub mod harness;
pub mod vendor;

/// A scalar both providers and the test generators support.
pub trait BenchScalar:
    tlinalg::FaerScalar + tlinalg_blas::LapackScalar + tlinalg_testkit::TestScalar
{
    /// Short dtype label for benchmark ids.
    const LABEL: &'static str;
}

impl BenchScalar for f64 {
    const LABEL: &'static str = "f64";
}

impl BenchScalar for num_complex::Complex64 {
    const LABEL: &'static str = "c64";
}

/// The thread budget the parallel rows use: `TLINALG_BENCH_THREADS`, else the machine's available
/// parallelism.
pub fn bench_threads() -> NonZeroUsize {
    std::env::var("TLINALG_BENCH_THREADS")
        .ok()
        .and_then(|value| value.parse().ok())
        .and_then(NonZeroUsize::new)
        .or_else(|| std::thread::available_parallelism().ok())
        .unwrap_or(NonZeroUsize::MIN)
}

/// The batch strategy from `TLINALG_BENCH_STRATEGY` (`auto|sequential|outer|inner|split:N`).
///
/// This is the benchmark sweep's escape hatch onto the shared `BatchPolicy`; an unset or unknown
/// value is `Auto`.
pub fn bench_strategy() -> tcontract_exec::BatchStrategy {
    match std::env::var("TLINALG_BENCH_STRATEGY").ok().as_deref() {
        Some("sequential") => tcontract_exec::BatchStrategy::Sequential,
        Some("outer") => tcontract_exec::BatchStrategy::Outer,
        Some("inner") => tcontract_exec::BatchStrategy::Inner,
        Some(value) if value.starts_with("split:") => tcontract_exec::BatchStrategy::Split {
            lanes: value[6..]
                .parse()
                .ok()
                .and_then(NonZeroUsize::new)
                .unwrap_or(NonZeroUsize::MIN),
        },
        _ => tcontract_exec::BatchStrategy::Auto,
    }
}

/// The pool and budget the parallel faer rows run on.
pub struct Env {
    /// A team, not a plain pool: faer runs parallel only on a team token.
    team: threadteam::Team,
    threads: NonZeroUsize,
    /// Read once at construction, so `par()` does not touch the environment inside a timed call.
    policy: tcontract_exec::BatchPolicy,
}

impl Env {
    /// Build a team of [`bench_threads`] workers.
    pub fn new() -> Self {
        Self::with_threads(bench_threads())
    }

    /// Build a team with the caller's explicit thread budget.
    pub fn with_threads(threads: NonZeroUsize) -> Self {
        let pool = threadteam::Pool::builder()
            .threads(threads.get())
            .build()
            .expect("a benchmark pool");
        let team = pool
            .into_team(threads.get())
            .unwrap_or_else(|e| panic!("a benchmark team: {e}"));
        Self {
            team,
            threads,
            policy: tcontract_exec::BatchPolicy::new(bench_strategy()),
        }
    }

    /// The worker count.
    pub fn threads(&self) -> usize {
        self.threads.get()
    }

    /// The token for the pool, with the cached `TLINALG_BENCH_STRATEGY` policy applied.
    pub fn par(&self) -> tcontract_exec::Exec<'_> {
        tcontract_exec::Exec::from_token(self.team.exec(self.threads.get()), None, self.policy)
    }
}

impl Default for Env {
    fn default() -> Self {
        Self::new()
    }
}

/// A compact column-major batch `[rows, cols, batch]`.
pub struct Batch<T> {
    /// Backing storage, items back to back.
    pub data: Vec<T>,
    dims: [usize; 3],
    strides: [isize; 3],
}

impl<T: tlinalg_testkit::TestScalar> Batch<T> {
    /// `batch` items `generate(item)` of shape `rows x cols`.
    pub fn new(rows: usize, cols: usize, batch: usize, generate: impl Fn(usize) -> Vec<T>) -> Self {
        let data: Vec<T> = (0..batch).flat_map(generate).collect();
        assert_eq!(data.len(), rows * cols * batch);
        Self {
            data,
            dims: [rows, cols, batch],
            strides: [1, rows as isize, (rows * cols) as isize],
        }
    }

    /// Well-conditioned general matrices.
    pub fn general(rows: usize, cols: usize, batch: usize) -> Self {
        Self::new(rows, cols, batch, |item| {
            tlinalg_testkit::matrix::<T>(rows, cols, item)
        })
    }

    /// Square rank-deficient matrices with clustered singular values
    /// ([`tlinalg_testkit::clustered_spectrum`]), the kind of input of
    /// <https://github.com/tensor4all/tlinalg-rs/issues/13>.
    pub fn clustered(n: usize, batch: usize) -> Self {
        let spectrum = tlinalg_testkit::clustered_spectrum(n);
        Self::new(n, n, batch, |item| {
            tlinalg_testkit::with_singular_values::<T>(&spectrum, item as u64 + 1)
        })
    }

    /// Hermitian positive-definite matrices.
    pub fn hpd(n: usize, batch: usize) -> Self {
        Self::new(n, n, batch, |item| {
            tlinalg_testkit::hpd_seeded::<T>(n, item + 1)
        })
    }

    /// The item count.
    pub fn count(&self) -> usize {
        self.dims[2]
    }

    /// Borrow as an input descriptor.
    pub fn view(&self) -> RawStridedRef<'_, T> {
        RawStridedRef::new(&self.data, &self.dims, &self.strides, 0)
            .expect("a compact batch describes its own storage")
    }
}

/// A host-shaped LAPACK workspace that recycles: a released buffer is handed out again, so a
/// warmed-up call allocates nothing through it, as with tenferro's session pool.
#[derive(Default)]
pub struct RecyclingWorkspace {
    buffers: HashMap<TypeId, Vec<Box<dyn Any + Send>>>,
    index: Vec<Vec<i32>>,
}

impl RecyclingWorkspace {
    fn take<S: tlinalg_blas::Scalar>(&mut self, cap: usize) -> Vec<S> {
        let pool = self.buffers.entry(TypeId::of::<S>()).or_default();
        let found = pool.iter().position(|buffer| {
            buffer
                .downcast_ref::<Vec<S>>()
                .is_some_and(|vec| vec.capacity() >= cap)
        });
        match found {
            Some(at) => {
                let mut vec = *pool
                    .swap_remove(at)
                    .downcast::<Vec<S>>()
                    .expect("the pool is keyed by element type");
                vec.clear();
                vec
            }
            None => Vec::with_capacity(cap),
        }
    }
}

impl<S: tlinalg_blas::Scalar + Default> Workspace<S> for RecyclingWorkspace {
    fn acquire_zeroed(&mut self, len: usize) -> Vec<S> {
        let mut vec = self.take::<S>(len);
        vec.resize(len, S::default());
        vec
    }

    fn acquire_capacity(&mut self, cap: usize) -> Vec<S> {
        self.take::<S>(cap)
    }

    fn acquire_uninit(&mut self, len: usize) -> Vec<MaybeUninit<S>> {
        let vec = ManuallyDrop::new(self.take::<S>(len));
        let (ptr, cap) = (vec.as_ptr().cast_mut(), vec.capacity());
        // SAFETY: `MaybeUninit<S>` has the layout of `S`, so the allocation (pointer, capacity,
        // layout) is reinterpreted unchanged; the length is set to `len <= cap` of uninitialised
        // elements, which `MaybeUninit` permits.
        unsafe { Vec::from_raw_parts(ptr.cast::<MaybeUninit<S>>(), len, cap) }
    }

    fn release(&mut self, buf: Vec<S>) {
        self.buffers
            .entry(TypeId::of::<S>())
            .or_default()
            .push(Box::new(buf));
    }
}

impl IndexWorkspace for RecyclingWorkspace {
    fn acquire_zeroed_index(&mut self, len: usize) -> Vec<i32> {
        let mut vec = match self.index.iter().position(|vec| vec.capacity() >= len) {
            Some(at) => self.index.swap_remove(at),
            None => Vec::with_capacity(len),
        };
        vec.clear();
        vec.resize(len, 0);
        vec
    }

    fn release_index(&mut self, buf: Vec<i32>) {
        self.index.push(buf);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recycled_buffers_come_back() {
        let mut ws = RecyclingWorkspace::default();
        let first: Vec<f64> = Workspace::<f64>::acquire_zeroed(&mut ws, 8);
        let ptr = first.as_ptr();
        Workspace::<f64>::release(&mut ws, first);
        let again: Vec<MaybeUninit<f64>> = Workspace::<f64>::acquire_uninit(&mut ws, 4);
        assert_eq!(again.as_ptr().cast::<f64>(), ptr);
        assert_eq!(again.len(), 4);
    }
}
