//! Steady-state heap allocation counts per call, at batch 1 and batch 64.
//!
//! A counting global allocator records every `alloc`/`realloc` on the calling thread. Each case is
//! warmed up, its output `Vec`s carry enough capacity, and the host workspace recycles every
//! buffer it is handed back, so what remains is what the kernel itself allocates per call. The
//! count must not grow with the batch size; the batch driver (layout, axis normalisation and item
//! iteration) contributes nothing, which the families with no per-call bookkeeping show as zero.
//!
//! Run with `--features link-openblas -- --nocapture` to see the table. Vendor-internal
//! allocations (OpenBLAS uses the C allocator) are outside the Rust allocator and not counted.

#![cfg(feature = "link-openblas")]

use core::mem::MaybeUninit;

use strided_view::RawStridedRef;
use tlinalg_blas::cholesky::cholesky;
use tlinalg_blas::eigh::eigh;
use tlinalg_blas::full_piv_lu::{full_piv_lu, FullPivLuOutputs};
use tlinalg_blas::lu::{lu, lu_factor, LuOutputs};
use tlinalg_blas::qr::{qr, rank_revealing_qr, RankRevealingQrOutputs};
use tlinalg_blas::solve::solve;
use tlinalg_blas::svd::{svd, SvdMode, SvdOutputs};
use tlinalg_blas::triangular_solve::{triangular_solve, TriangularSolveOptions};
use tlinalg_blas::{IndexWorkspace, Op, Workspace};
use tlinalg_testkit::alloc::{allocations, Counting};

#[global_allocator]
static GLOBAL: Counting = Counting;

/// A host-shaped workspace that recycles: a released buffer is handed out again (best fit), so a
/// warmed-up call allocates nothing through it.
struct Recycling {
    real: Vec<Vec<f64>>,
    index: Vec<Vec<i32>>,
}

impl Recycling {
    fn new() -> Self {
        Self {
            real: Vec::with_capacity(64),
            index: Vec::with_capacity(64),
        }
    }
}

fn take<T>(pool: &mut Vec<Vec<T>>, cap: usize) -> Vec<T> {
    let best = pool
        .iter()
        .enumerate()
        .filter(|(_, buffer)| buffer.capacity() >= cap)
        .min_by_key(|(_, buffer)| buffer.capacity())
        .map(|(index, _)| index);
    match best {
        Some(index) => {
            let mut buffer = pool.swap_remove(index);
            buffer.clear();
            buffer
        }
        None => Vec::with_capacity(cap),
    }
}

impl Workspace<f64> for Recycling {
    fn acquire_zeroed(&mut self, len: usize) -> Vec<f64> {
        let mut buffer = take(&mut self.real, len);
        buffer.resize(len, 0.0);
        buffer
    }
    fn acquire_capacity(&mut self, cap: usize) -> Vec<f64> {
        take(&mut self.real, cap)
    }
    fn acquire_uninit(&mut self, len: usize) -> Vec<MaybeUninit<f64>> {
        (0..len).map(|_| MaybeUninit::uninit()).collect()
    }
    fn release(&mut self, buf: Vec<f64>) {
        self.real.push(buf);
    }
}

impl IndexWorkspace for Recycling {
    fn acquire_zeroed_index(&mut self, len: usize) -> Vec<i32> {
        let mut buffer = take(&mut self.index, len);
        buffer.resize(len, 0);
        buffer
    }
    fn release_index(&mut self, buf: Vec<i32>) {
        self.index.push(buf);
    }
}

const N: usize = 6;

/// `count` compact symmetric positive definite `N x N` matrices.
fn spd_batch(count: usize) -> Vec<f64> {
    let mut data = Vec::with_capacity(N * N * count);
    for item in 0..count {
        for col in 0..N {
            for row in 0..N {
                let off = 0.1 / (1.0 + (row + col + item % 3) as f64);
                data.push(if row == col { N as f64 + 1.0 } else { off });
            }
        }
    }
    data
}

/// Warm a case up, then count one call.
fn measure(mut call: impl FnMut()) -> usize {
    for _ in 0..3 {
        call();
    }
    allocations(call)
}

fn row(name: &str, count: usize) -> usize {
    let a_data = spd_batch(count);
    let dims = [N, N, count];
    let strides = [1, N as isize, (N * N) as isize];
    let a = RawStridedRef::new(&a_data, &dims, &strides, 0).unwrap();
    let b_dims = [N, 2, count];
    let b_strides = [1, N as isize, (N * 2) as isize];
    let b_data = vec![1.0; N * 2 * count];
    let b = RawStridedRef::new(&b_data, &b_dims, &b_strides, 0).unwrap();
    let big = N * N * count;
    let mut ws = Recycling::new();
    let (mut o1, mut o2, mut o3) = (
        Vec::with_capacity(big),
        Vec::with_capacity(big),
        Vec::with_capacity(big),
    );
    match name {
        "cholesky" => measure(|| cholesky(Op::Cholesky, a, &mut o1, &mut ws).unwrap()),
        "triangular_solve" => measure(|| {
            let options = TriangularSolveOptions {
                left_side: true,
                lower: true,
                ..Default::default()
            };
            triangular_solve(Op::TriangularSolve, options, a, b, &mut o1, &mut ws).unwrap()
        }),
        "solve" => measure(|| solve(Op::Solve, false, a, b, &mut o1, &mut ws).unwrap()),
        "qr" => measure(|| qr(Op::Qr, a, &mut o1, &mut o2, &mut ws).unwrap()),
        "rank_revealing_qr" => {
            // `jpvt` is `i32` and `IndexWorkspace` is not in this entry point's published bound, so
            // it stays a per-call allocation; `tau` is now pooled and the duplicate check marks
            // `jpvt` in place. One is an **upper bound** for this site (tlinalg-rs#57).
            let mut permutation = Vec::with_capacity(N * count);
            measure(|| {
                let outputs = RankRevealingQrOutputs {
                    q: &mut o1,
                    r: &mut o2,
                    permutation: &mut permutation,
                };
                rank_revealing_qr(Op::RankRevealingQr, a, outputs, &mut ws).unwrap()
            })
        }
        "eigh" => measure(|| eigh(Op::Eigh, a, &mut o1, Some(&mut o2), &mut ws).unwrap()),
        "svd" => measure(|| {
            let outputs = SvdOutputs {
                s: &mut o1,
                u: &mut o2,
                vt: &mut o3,
            };
            svd(Op::Svd, SvdMode::Thin, a, outputs, &mut ws).unwrap()
        }),
        "lu_factor" => {
            let mut lu = a_data.clone();
            let mut pivots = vec![0_i32; N * count];
            let mut parity = vec![0.0; count];
            measure(|| {
                lu.copy_from_slice(&a_data);
                lu_factor(Op::LuFactor, N, N, &mut lu, &mut pivots, &mut parity).unwrap()
            })
        }
        "lu" => {
            let mut parity = Vec::with_capacity(count);
            measure(|| {
                let outputs = LuOutputs {
                    p: &mut o1,
                    l: &mut o2,
                    u: &mut o3,
                    parity: &mut parity,
                };
                lu(Op::Lu, a, outputs, &mut ws).unwrap()
            })
        }
        "full_piv_lu" => {
            let mut o4 = Vec::with_capacity(big);
            let mut parity = Vec::with_capacity(count);
            measure(|| {
                let outputs = FullPivLuOutputs {
                    p: &mut o1,
                    l: &mut o2,
                    u: &mut o3,
                    q: &mut o4,
                    parity: &mut parity,
                };
                full_piv_lu(Op::FullPivLu, a, outputs, &mut ws).unwrap()
            })
        }
        _ => unreachable!(),
    }
}

#[test]
fn steady_state_allocations_do_not_grow_with_the_batch() {
    // `solve` now returns its LU copy and pivot buffer on success, so a recycling host sees a
    // steady-state call allocate nothing. Everything else, including the batch driver, allocates
    // nothing once warmed up. Error paths abandon the two buffers (see the module docs).
    let families = [
        ("cholesky", 0),
        ("triangular_solve", 0),
        ("solve", 0),
        ("qr", 0),
        // Upper bound: one `jpvt` buffer (see the row above); one is what remains.
        ("rank_revealing_qr", 1),
        ("eigh", 0),
        ("svd", 0),
        ("lu_factor", 0),
        // The permutation matrices are written straight from the LAPACK pivots (identity plus
        // in-place row/column swaps), so both families are steady-state allocation free.
        ("lu", 0),
        ("full_piv_lu", 0),
    ];
    println!("| family | batch 1 | batch 64 |");
    println!("|---|---|---|");
    for (family, expected) in families {
        let one = row(family, 1);
        let many = row(family, 64);
        println!("| {family} | {one} | {many} |");
        assert_eq!(one, many, "{family}: allocations grow with the batch");
        assert_eq!(
            one, expected,
            "{family}: steady-state allocation count changed"
        );
    }
}
