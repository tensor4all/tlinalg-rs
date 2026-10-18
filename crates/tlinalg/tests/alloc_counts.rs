//! Heap allocations per batched call, counted with a global allocator.
//!
//! Three layers, all in this one binary (one counting allocator):
//!
//! * `per_call_allocations_do_not_grow_with_the_batch` pins the **`Vec` wrappers**' steady-state
//!   count: they reserve their outputs and allocate exactly one scratch block of `*_workspace`
//!   bytes per call, independent of the batch size (zero when the family needs no scratch).
//! * `into_is_allocation_free_for_every_family` is the gate the borrowed-workspace API exists for
//!   (`tlinalg-rs#57` test 1): every family's `*_into` call, in all four scalar types, at sizes
//!   {1, 2, 8}, batches {1, 64}, with contiguous **and** transposed input, on `Exec::serial()`,
//!   allocates nothing at all.
//! * `the_pool_route_adds_a_constant` is test 7: a pooled `Exec` adds the pool's own fixed cost,
//!   which does not grow with the batch.
//!
//! Tests 2 (batch independence) and 3 (layout independence) live here too: the battery compares a
//! transposed input and a deliberately misaligned workspace against the aligned contiguous run,
//! bitwise. The wrapper/`_into` bitwise equality in that battery checks the wrapper's publication
//! path; it is **not** the 0.3.0 comparison, which is its own test against the published release.

use strided_view::{RawStridedMut, RawStridedRef};
use tcontract_exec::Exec;
use tlinalg::triangular_solve::{triangular_solve, TriangularSolveFlags};
use tlinalg::Op;
use tlinalg_testkit::alloc::{steady, Counting};

#[global_allocator]
static GLOBAL: Counting = Counting;

/// `batch` compact, well-conditioned `n x n` matrices.
fn batch_of(n: usize, batch: usize) -> Vec<f64> {
    (0..n * n * batch)
        .map(|i| {
            let (row, col) = (i % n, (i / n) % n);
            if row == col {
                4.0 + row as f64
            } else {
                0.1
            }
        })
        .collect()
}

/// Steady-state allocations per call of each family, one sequential lane, `n = 6`.
fn family_counts(batch: usize) -> [(&'static str, usize); 11] {
    let n = 6;
    let a = batch_of(n, batch);
    let (dims, strides) = ([n, n, batch], [1, n as isize, (n * n) as isize]);
    let view = RawStridedRef::new(&a, &dims, &strides, 0).unwrap();
    let seq = Exec::serial();
    let cap = n * n * batch;
    let (mut v1, mut v2, mut v3) = (
        Vec::with_capacity(cap),
        Vec::with_capacity(cap),
        Vec::with_capacity(cap),
    );
    let cholesky = steady(|| {
        tlinalg::cholesky::cholesky(Op::Cholesky, view, &mut v1, seq).unwrap();
    });
    let qr = steady(|| tlinalg::qr::qr(Op::Qr, view, &mut v1, &mut v2, seq).unwrap());
    let eigh = steady(|| {
        tlinalg::eigh::eigh(Op::Eigh, view, &mut v1, &mut v2, seq).unwrap();
    });
    let svd = steady(|| {
        tlinalg::svd::svd(Op::Svd, view, false, &mut v1, &mut v2, &mut v3, seq).unwrap();
    });
    let (mut lu, mut piv, mut parity) = (a.clone(), vec![0; n * batch], vec![0.0; batch]);
    let packed_lu = steady(|| {
        lu.copy_from_slice(&a);
        tlinalg::packed_lu::factor(Op::LuFactor, n, n, &mut lu, &mut piv, &mut parity, seq)
            .unwrap();
    });
    let mut perm = Vec::with_capacity(n * batch);
    let rrqr = steady(|| {
        tlinalg::qr::rank_revealing_qr(Op::RankRevealingQr, view, &mut v1, &mut v2, &mut perm, seq)
            .unwrap();
    });
    let b = batch_of(n, batch);
    let b_view = RawStridedRef::new(&b, &dims, &strides, 0).unwrap();
    let mut x = vec![0.0; cap];
    let solve = steady(|| {
        tlinalg::lu::solve(
            Op::Solve,
            view,
            Some(b_view),
            RawStridedMut::new(&mut x, &dims, &strides, 0).unwrap(),
            false,
            seq,
        )
        .unwrap();
    });
    let full_piv_lu_solve = steady(|| {
        tlinalg::full_piv_lu::full_piv_lu_solve(
            Op::FullPivLuSolve,
            view,
            b_view,
            false,
            &mut v1,
            seq,
        )
        .unwrap();
    });
    let (mut state, mut coeff) = (a.clone(), Vec::with_capacity(n * batch));
    let compact_factor = steady(|| {
        state.copy_from_slice(&a);
        tlinalg::householder::compact_factor(
            Op::HouseholderQr,
            n,
            n,
            batch,
            &mut state,
            &mut coeff,
            seq,
        )
        .unwrap();
    });
    [
        ("compact_factor", compact_factor),
        ("cholesky", cholesky),
        ("rank_revealing_qr", rrqr),
        ("solve", solve),
        ("full_piv_lu_solve", full_piv_lu_solve),
        ("qr", qr),
        ("eigh", eigh),
        ("svd", svd),
        ("packed_lu factor", packed_lu),
        ("triangular_solve left", triangular(true, batch)),
        ("triangular_solve right", triangular(false, batch)),
    ]
}

/// Allocations made by one triangular solve over `batch` systems on one sequential lane.
fn triangular(left_side: bool, batch: usize) -> usize {
    let n = 6;
    let nrhs = 3;
    let a: Vec<f64> = (0..n * n)
        .map(|i| if i % (n + 1) == 0 { 4.0 } else { 0.1 })
        .collect();
    let (rows, cols) = if left_side { (n, nrhs) } else { (nrhs, n) };
    let b = vec![1.0_f64; rows * cols * batch];
    let flags = TriangularSolveFlags {
        left_side,
        lower: true,
        transpose_a: false,
        unit_diagonal: false,
    };
    let (a_dims, a_strides) = ([n, n, batch], [1, n as isize, 0]);
    let (b_dims, b_strides) = (
        [rows, cols, batch],
        [1, rows as isize, (rows * cols) as isize],
    );
    let a_view = RawStridedRef::new(&a, &a_dims, &a_strides, 0).unwrap();
    let b_view = RawStridedRef::new(&b, &b_dims, &b_strides, 0).unwrap();
    let mut x = Vec::with_capacity(rows * cols * batch);
    steady(|| {
        triangular_solve(
            Op::TriangularSolve,
            a_view,
            b_view,
            flags,
            &mut x,
            Exec::serial(),
        )
        .unwrap();
    })
}

// The blocked compact Householder path is deliberately not pinned here. Its trailing update goes
// through faer's GEMM, and how many buffers that allocates is the backend's business: on
// `x86_64-linux` a blocking shape costs the same two allocations per call from batch 1 to 64
// (measured for 32x32 through 512x512), while on `aarch64-darwin` the same call allocates about
// twelve buffers *per item* (14 for batch 1 against 770 for batch 64 at 64x48) inside `nano-gemm`'s
// packing. Those are not this crate's allocations, and a count that differs by target is not a
// regression this file could report. The unblocked shapes above stay pinned because the driver and
// the lane scratch are what they measure.

/// The driver itself allocates nothing on one lane; what remains is each family's faer lane
/// scratch, built once per call and independent of the batch size.
#[test]
fn per_call_allocations_do_not_grow_with_the_batch() {
    let _guard = serialised();
    let one = family_counts(1);
    let many = family_counts(64);
    for ((name, at_one), (_, at_many)) in one.iter().zip(&many) {
        println!("{name}: {at_one} (batch 1), {at_many} (batch 64)");
        assert_eq!(at_one, at_many, "{name}: allocations grow with the batch");
    }
    // Each count is at most the pre-extraction route's native allocations for the same call
    // (rank_revealing_qr was 7, solve 5, full_piv_lu_solve 7).
    let expected = [
        // One scratch block of `*_workspace` bytes per call (the caller-owned lane storage the
        // wrapper allocates for its `*_into` route); no per-family, per-lane allocations remain.
        // `compact_factor`, `rank_revealing_qr`, `solve`, `full_piv_lu_solve`, `qr`, `eigh`, `svd`
        // and `packed_lu factor` used to allocate 2/5/5/7/4/2/3/5 times: their work matrices,
        // permutation vectors and faer buffers are now carved from that one block.
        ("compact_factor", 1),
        ("cholesky", 1),
        ("rank_revealing_qr", 1),
        ("solve", 1),
        ("full_piv_lu_solve", 1),
        ("qr", 1),
        ("eigh", 1),
        ("svd", 1),
        ("packed_lu factor", 1),
        // No library scratch at all: the output chunk is the work matrix.
        ("triangular_solve left", 0),
        ("triangular_solve right", 0),
    ];
    assert_eq!(one, expected);
}

// ---------------------------------------------------------------------------------------------
// The borrowed-workspace gates (tlinalg-rs#57 tests 1-5, 7).
// ---------------------------------------------------------------------------------------------

use core::mem::MaybeUninit;

use num_complex::{Complex32, Complex64};
use tcontract_exec::{BatchPolicy, BatchStrategy, Pool};
use tlinalg::full_piv_lu::{
    full_piv_lu, full_piv_lu_into, full_piv_lu_solve, full_piv_lu_solve_into,
    full_piv_lu_solve_workspace, full_piv_lu_workspace, FullPivLuFactors,
};
use tlinalg::householder::{
    apply_reflectors, apply_reflectors_into, apply_reflectors_workspace, compact_factor,
    compact_factor_into, compact_factor_workspace, ReflectorShape,
};
use tlinalg::lu::{lu, lu_into, lu_solve_workspace, lu_workspace, solve, solve_into, LuFactors};
use tlinalg::{cholesky, eig, eigh, packed_lu, qr, svd, triangular_solve, Error};
use tlinalg_testkit::alloc::{allocations, allocations_all_threads};
use tlinalg_testkit::TestScalar;

/// Serialises the tests in this binary: the process-wide counter behind
/// [`allocations_all_threads`] sees every thread, so the pool gate must not run next to another
/// test's allocations.
fn serialised() -> std::sync::MutexGuard<'static, ()> {
    static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
    LOCK.lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

/// The raw bytes of an output buffer: the bitwise comparison the workspace gates need (ordinary
/// `==` on floats is not bitwise, and `NaN` payloads must compare).
fn bytes_of<T: Copy>(values: &[T]) -> Vec<u8> {
    // SAFETY: every `T` here is a plain `Copy` scalar (f32/f64/`Complex`/i32/i64), so reading the
    // initialised slice as bytes is sound and exposes the exact bit patterns.
    unsafe {
        core::slice::from_raw_parts(values.as_ptr().cast::<u8>(), core::mem::size_of_val(values))
    }
    .to_vec()
}

/// The raw bytes of a `MaybeUninit` output buffer the `*_into` call just filled.
///
/// # Safety
///
/// Every element must be initialised; the callers only use it after `Ok`.
unsafe fn bytes_uninit<T: Copy>(buf: &[MaybeUninit<T>]) -> Vec<u8> {
    // SAFETY: the caller guarantees full initialisation and `T` is a plain `Copy` scalar.
    bytes_of(unsafe { core::slice::from_raw_parts(buf.as_ptr().cast::<T>(), buf.len()) })
}

/// Read a fully written `MaybeUninit` buffer.
///
/// # Safety
///
/// Every element was written by the `*_into` call that just returned `Ok`.
unsafe fn assume<T>(buf: Vec<MaybeUninit<T>>) -> Vec<T> {
    let len = buf.len();
    let mut buf = core::mem::ManuallyDrop::new(buf);
    // SAFETY: the caller guarantees every element is initialised; `MaybeUninit<T>` has the layout
    // of `T` and the `ManuallyDrop` keeps the `MaybeUninit` allocation from being dropped twice.
    unsafe { Vec::from_raw_parts(buf.as_mut_ptr().cast::<T>(), len, len) }
}

/// One dtype's full battery: the zero-allocation gate, batch independence, layout independence and
/// the bitwise wrapper/`_into` equality.
///
/// The macro is expanded once per scalar type so every output buffer has its concrete type
/// (`eigh_values` returns the real type, `eig` the complex one) without a public type mapping.
macro_rules! dtype_battery {
    ($t:ty, $real:ty, $cplx:ty) => {{
        let seq = Exec::serial();
        let value = |x: f64| <$t as TestScalar>::from_c64(Complex64::new(x, 0.0));
        // Diagonally dominant and symmetric: SPD, nonsingular, and well conditioned enough for
        // every decomposition to converge at these sizes.
        let spd = |n: usize, batch: usize| -> Vec<$t> {
            (0..n * n * batch)
                .map(|k| {
                    let (row, col) = (k % n, (k / n) % n);
                    if row == col {
                        value(n as f64 + 2.0)
                    } else {
                        value(0.05 * ((row + 1) * (col + 1) % 3) as f64)
                    }
                })
                .collect()
        };
        // The same logical matrices in row-major storage, read through strides `[n, 1]`.
        let transposed = |data: &[$t], n: usize| -> Vec<$t> {
            (0..data.len())
                .map(|k| {
                    let (item, rest) = (k / (n * n), k % (n * n));
                    let (row, col) = (rest % n, rest / n);
                    data[item * n * n + col * n + row]
                })
                .collect()
        };

        // One full pass over every family for one shape, one batch size and one input layout.
        // Returns the bit patterns of every output plus the zero-allocation assertions.
        let run = |n: usize,
                   batch: usize,
                   transposed_layout: bool,
                   offset: usize,
                   compare_wrapper: bool| {
            let data = spd(n, batch);
            let (owned, dims, strides): (Vec<$t>, [usize; 3], [isize; 3]) = if transposed_layout {
                (
                    transposed(&data, n),
                    [n, n, batch],
                    [n as isize, 1, (n * n) as isize],
                )
            } else {
                (
                    data.clone(),
                    [n, n, batch],
                    [1, n as isize, (n * n) as isize],
                )
            };
            let view = RawStridedRef::new(&owned, &dims, &strides, 0).unwrap();
            let square = n * n;
            // A deliberately misaligned workspace base: the query's byte count must cover the
            // worst-case padding, so every carve below works from any alignment.
            let scratch = |bytes: usize| -> Vec<MaybeUninit<u8>> {
                vec![MaybeUninit::<u8>::uninit(); bytes + offset]
            };
            let mut out: Vec<(&'static str, Vec<u8>)> = Vec::new();
            macro_rules! zero_gate {
                ($name:literal, $call:block) => {{
                    let allocs = allocations(|| $call);
                    assert_eq!(
                        allocs, 0,
                        "{}: `_into` allocated ({n}x{n}, batch {batch})",
                        $name
                    );
                }};
            }

            // qr
            {
                let ws = qr::qr_workspace::<$t>(n, n, batch, seq).unwrap();
                let mut workspace = scratch(ws);
                let mut q = vec![MaybeUninit::<$t>::uninit(); square * batch];
                let mut r = vec![MaybeUninit::<$t>::uninit(); square * batch];
                zero_gate!("qr", {
                    qr::qr_into(
                        Op::Qr,
                        view,
                        &mut q,
                        &mut r,
                        &mut workspace[offset..offset + ws],
                        seq,
                    )
                    .unwrap()
                });
                out.push(("qr.q", unsafe { bytes_uninit(&q) }));
                out.push(("qr.r", unsafe { bytes_uninit(&r) }));
                if compare_wrapper {
                    let (mut wq, mut wr) = (Vec::new(), Vec::new());
                    qr::qr(Op::Qr, view, &mut wq, &mut wr, seq).unwrap();
                    assert_eq!(unsafe { bytes_uninit(&q) }, bytes_of(&wq), "qr wrapper Q");
                    assert_eq!(unsafe { bytes_uninit(&r) }, bytes_of(&wr), "qr wrapper R");
                }
            }
            // rank_revealing_qr
            {
                let ws = qr::rank_revealing_qr_workspace::<$t>(n, n, batch, seq).unwrap();
                let mut workspace = scratch(ws);
                let mut q = vec![MaybeUninit::<$t>::uninit(); square * batch];
                let mut r = vec![MaybeUninit::<$t>::uninit(); square * batch];
                let mut perm = vec![MaybeUninit::<i64>::uninit(); n * batch];
                zero_gate!("rank_revealing_qr", {
                    qr::rank_revealing_qr_into(
                        Op::RankRevealingQr,
                        view,
                        &mut q,
                        &mut r,
                        &mut perm,
                        &mut workspace[offset..offset + ws],
                        seq,
                    )
                    .unwrap()
                });
                out.push(("rrqr.q", unsafe { bytes_uninit(&q) }));
                out.push(("rrqr.r", unsafe { bytes_uninit(&r) }));
                // SAFETY: `rank_revealing_qr_into` returned `Ok`.
                let perm_values = unsafe { assume(perm) };
                out.push(("rrqr.perm", bytes_of(&perm_values)));
                if compare_wrapper {
                    let (mut wq, mut wr, mut wp) = (Vec::new(), Vec::new(), Vec::new());
                    qr::rank_revealing_qr(
                        Op::RankRevealingQr,
                        view,
                        &mut wq,
                        &mut wr,
                        &mut wp,
                        seq,
                    )
                    .unwrap();
                    assert_eq!(unsafe { bytes_uninit(&q) }, bytes_of(&wq), "rrqr wrapper Q");
                    assert_eq!(unsafe { bytes_uninit(&r) }, bytes_of(&wr), "rrqr wrapper R");
                    assert_eq!(bytes_of(&perm_values), bytes_of(&wp), "rrqr wrapper perm");
                }
            }
            // svd
            {
                let ws = svd::svd_workspace::<$t>(n, n, false, batch, seq).unwrap();
                let mut workspace = scratch(ws);
                let mut u = vec![MaybeUninit::<$t>::uninit(); square * batch];
                let mut s = vec![MaybeUninit::<$t>::uninit(); n * batch];
                let mut vt = vec![MaybeUninit::<$t>::uninit(); square * batch];
                zero_gate!("svd", {
                    svd::svd_into(
                        Op::Svd,
                        view,
                        false,
                        &mut u,
                        &mut s,
                        &mut vt,
                        &mut workspace[offset..offset + ws],
                        seq,
                    )
                    .unwrap()
                });
                out.push(("svd.u", unsafe { bytes_uninit(&u) }));
                out.push(("svd.s", unsafe { bytes_uninit(&s) }));
                out.push(("svd.vt", unsafe { bytes_uninit(&vt) }));
                if compare_wrapper {
                    let (mut wu, mut ws_, mut wv) = (Vec::new(), Vec::new(), Vec::new());
                    svd::svd(Op::Svd, view, false, &mut wu, &mut ws_, &mut wv, seq).unwrap();
                    assert_eq!(unsafe { bytes_uninit(&u) }, bytes_of(&wu), "svd wrapper U");
                    assert_eq!(unsafe { bytes_uninit(&s) }, bytes_of(&ws_), "svd wrapper S");
                    assert_eq!(
                        unsafe { bytes_uninit(&vt) },
                        bytes_of(&wv),
                        "svd wrapper VH"
                    );
                }
            }
            // svd_values
            {
                let ws = svd::svd_values_workspace::<$t>(n, n, batch, seq).unwrap();
                let mut workspace = scratch(ws);
                let mut s = vec![MaybeUninit::<$real>::uninit(); n * batch];
                zero_gate!("svd_values", {
                    svd::svd_values_into(
                        Op::SvdValues,
                        view,
                        &mut s,
                        &mut workspace[offset..offset + ws],
                        seq,
                    )
                    .unwrap()
                });
                // SAFETY: `svd_values_into` returned `Ok`.
                let s_values = unsafe { assume(s) };
                out.push(("svd_values.s", bytes_of(&s_values)));
                if compare_wrapper {
                    let mut ws_ = Vec::new();
                    svd::svd_values(Op::SvdValues, view, &mut ws_, seq).unwrap();
                    assert_eq!(bytes_of(&s_values), bytes_of(&ws_), "svd_values wrapper");
                }
            }
            // eigh
            {
                let ws = eigh::eigh_workspace::<$t>(n, true, batch, seq).unwrap();
                let mut workspace = scratch(ws);
                let mut values = vec![MaybeUninit::<$t>::uninit(); n * batch];
                let mut vectors = vec![MaybeUninit::<$t>::uninit(); square * batch];
                zero_gate!("eigh", {
                    eigh::eigh_into(
                        Op::Eigh,
                        view,
                        &mut values,
                        &mut vectors,
                        &mut workspace[offset..offset + ws],
                        seq,
                    )
                    .unwrap()
                });
                out.push(("eigh.values", unsafe { bytes_uninit(&values) }));
                out.push(("eigh.vectors", unsafe { bytes_uninit(&vectors) }));
                if compare_wrapper {
                    let (mut wv, mut wm) = (Vec::new(), Vec::new());
                    eigh::eigh(Op::Eigh, view, &mut wv, &mut wm, seq).unwrap();
                    assert_eq!(
                        unsafe { bytes_uninit(&values) },
                        bytes_of(&wv),
                        "eigh wrapper values"
                    );
                    assert_eq!(
                        unsafe { bytes_uninit(&vectors) },
                        bytes_of(&wm),
                        "eigh wrapper vectors"
                    );
                }
            }
            // eigh_values
            {
                let ws = eigh::eigh_values_workspace::<$t>(n, batch, seq).unwrap();
                let mut workspace = scratch(ws);
                let mut values = vec![MaybeUninit::<$real>::uninit(); n * batch];
                zero_gate!("eigh_values", {
                    eigh::eigh_values_into(
                        Op::EighValues,
                        view,
                        &mut values,
                        &mut workspace[offset..offset + ws],
                        seq,
                    )
                    .unwrap()
                });
                // SAFETY: `eigh_values_into` returned `Ok`.
                let values_buf = unsafe { assume(values) };
                out.push(("eigh_values.values", bytes_of(&values_buf)));
                if compare_wrapper {
                    let mut wv = Vec::new();
                    eigh::eigh_values(Op::EighValues, view, &mut wv, seq).unwrap();
                    assert_eq!(bytes_of(&values_buf), bytes_of(&wv), "eigh_values wrapper");
                }
            }
            // eig
            {
                let ws = eig::eig_workspace::<$t>(n, true, batch, seq).unwrap();
                let mut workspace = scratch(ws);
                let mut values = vec![MaybeUninit::<$cplx>::uninit(); n * batch];
                let mut vectors = vec![MaybeUninit::<$cplx>::uninit(); square * batch];
                zero_gate!("eig", {
                    eig::eig_into(
                        Op::Eig,
                        view,
                        &mut values,
                        &mut vectors,
                        &mut workspace[offset..offset + ws],
                        seq,
                    )
                    .unwrap()
                });
                out.push(("eig.values", unsafe { bytes_uninit(&values) }));
                out.push(("eig.vectors", unsafe { bytes_uninit(&vectors) }));
                if compare_wrapper {
                    let (mut wv, mut wm) = (Vec::new(), Vec::new());
                    eig::eig(Op::Eig, view, &mut wv, &mut wm, seq).unwrap();
                    assert_eq!(
                        unsafe { bytes_uninit(&values) },
                        bytes_of(&wv),
                        "eig wrapper values"
                    );
                    assert_eq!(
                        unsafe { bytes_uninit(&vectors) },
                        bytes_of(&wm),
                        "eig wrapper vectors"
                    );
                }
            }
            // eig_values
            {
                let ws = eig::eig_values_workspace::<$t>(n, batch, seq).unwrap();
                let mut workspace = scratch(ws);
                let mut values = vec![MaybeUninit::<$cplx>::uninit(); n * batch];
                zero_gate!("eig_values", {
                    eig::eig_values_into(
                        Op::EigValues,
                        view,
                        &mut values,
                        &mut workspace[offset..offset + ws],
                        seq,
                    )
                    .unwrap()
                });
                out.push(("eig_values.values", unsafe { bytes_uninit(&values) }));
                if compare_wrapper {
                    let mut wv = Vec::new();
                    eig::eig_values(Op::EigValues, view, &mut wv, seq).unwrap();
                    assert_eq!(
                        unsafe { bytes_uninit(&values) },
                        bytes_of(&wv),
                        "eig_values wrapper"
                    );
                }
            }
            // lu
            {
                let ws = lu_workspace::<$t>(n, n, batch, seq).unwrap();
                let mut workspace = scratch(ws);
                let mut p = vec![MaybeUninit::<$t>::uninit(); square * batch];
                let mut l = vec![MaybeUninit::<$t>::uninit(); square * batch];
                let mut u = vec![MaybeUninit::<$t>::uninit(); square * batch];
                let mut parity = vec![MaybeUninit::<$t>::uninit(); batch];
                zero_gate!("lu", {
                    lu_into(
                        Op::Lu,
                        view,
                        &mut p,
                        &mut l,
                        &mut u,
                        &mut parity,
                        &mut workspace[offset..offset + ws],
                        seq,
                    )
                    .unwrap()
                });
                out.push(("lu.p", unsafe { bytes_uninit(&p) }));
                out.push(("lu.l", unsafe { bytes_uninit(&l) }));
                out.push(("lu.u", unsafe { bytes_uninit(&u) }));
                out.push(("lu.parity", unsafe { bytes_uninit(&parity) }));
                if compare_wrapper {
                    let (mut wp, mut wl, mut wu, mut wpar) =
                        (Vec::new(), Vec::new(), Vec::new(), Vec::new());
                    lu(
                        Op::Lu,
                        view,
                        LuFactors {
                            p: &mut wp,
                            l: &mut wl,
                            u: &mut wu,
                            parity: &mut wpar,
                        },
                        seq,
                    )
                    .unwrap();
                    assert_eq!(unsafe { bytes_uninit(&p) }, bytes_of(&wp), "lu wrapper P");
                    assert_eq!(unsafe { bytes_uninit(&l) }, bytes_of(&wl), "lu wrapper L");
                    assert_eq!(unsafe { bytes_uninit(&u) }, bytes_of(&wu), "lu wrapper U");
                    assert_eq!(
                        unsafe { bytes_uninit(&parity) },
                        bytes_of(&wpar),
                        "lu wrapper parity"
                    );
                }
            }
            // lu::solve
            {
                let ws = lu_solve_workspace::<$t>(n, n, false, batch, seq).unwrap();
                let mut workspace = scratch(ws);
                let mut x = vec![value(1.0); square * batch];
                let out_dims = [n, n, batch];
                let out_strides = [1isize, n as isize, (n * n) as isize];
                zero_gate!("solve", {
                    let destination =
                        RawStridedMut::new(&mut x, &out_dims, &out_strides, 0).unwrap();
                    solve_into(
                        Op::Solve,
                        view,
                        Some(view),
                        destination,
                        false,
                        &mut workspace[offset..offset + ws],
                        seq,
                    )
                    .unwrap()
                });
                out.push(("solve.x", bytes_of(&x)));
                if compare_wrapper {
                    let mut wx = vec![value(1.0); square * batch];
                    let destination =
                        RawStridedMut::new(&mut wx, &out_dims, &out_strides, 0).unwrap();
                    solve(Op::Solve, view, Some(view), destination, false, seq).unwrap();
                    assert_eq!(bytes_of(&x), bytes_of(&wx), "solve wrapper");
                }
            }
            // full_piv_lu
            {
                let ws = full_piv_lu_workspace::<$t>(n, batch, seq).unwrap();
                let mut workspace = scratch(ws);
                let mut p = vec![MaybeUninit::<$t>::uninit(); square * batch];
                let mut l = vec![MaybeUninit::<$t>::uninit(); square * batch];
                let mut u = vec![MaybeUninit::<$t>::uninit(); square * batch];
                let mut q = vec![MaybeUninit::<$t>::uninit(); square * batch];
                let mut parity = vec![MaybeUninit::<$t>::uninit(); batch];
                zero_gate!("full_piv_lu", {
                    full_piv_lu_into(
                        Op::FullPivLu,
                        view,
                        &mut p,
                        &mut l,
                        &mut u,
                        &mut q,
                        &mut parity,
                        &mut workspace[offset..offset + ws],
                        seq,
                    )
                    .unwrap()
                });
                out.push(("full_piv_lu.p", unsafe { bytes_uninit(&p) }));
                out.push(("full_piv_lu.l", unsafe { bytes_uninit(&l) }));
                out.push(("full_piv_lu.u", unsafe { bytes_uninit(&u) }));
                out.push(("full_piv_lu.q", unsafe { bytes_uninit(&q) }));
                out.push(("full_piv_lu.parity", unsafe { bytes_uninit(&parity) }));
                if compare_wrapper {
                    let (mut wp, mut wl, mut wu, mut wq, mut wpar) =
                        (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new());
                    full_piv_lu(
                        Op::FullPivLu,
                        view,
                        FullPivLuFactors {
                            p: &mut wp,
                            l: &mut wl,
                            u: &mut wu,
                            q: &mut wq,
                            parity: &mut wpar,
                        },
                        seq,
                    )
                    .unwrap();
                    assert_eq!(
                        unsafe { bytes_uninit(&p) },
                        bytes_of(&wp),
                        "full_piv_lu wrapper P"
                    );
                    assert_eq!(
                        unsafe { bytes_uninit(&l) },
                        bytes_of(&wl),
                        "full_piv_lu wrapper L"
                    );
                    assert_eq!(
                        unsafe { bytes_uninit(&u) },
                        bytes_of(&wu),
                        "full_piv_lu wrapper U"
                    );
                    assert_eq!(
                        unsafe { bytes_uninit(&q) },
                        bytes_of(&wq),
                        "full_piv_lu wrapper Q"
                    );
                    assert_eq!(
                        unsafe { bytes_uninit(&parity) },
                        bytes_of(&wpar),
                        "full_piv_lu wrapper parity"
                    );
                }
            }
            // full_piv_lu_solve
            {
                let ws = full_piv_lu_solve_workspace::<$t>(n, n, batch, seq).unwrap();
                let mut workspace = scratch(ws);
                let mut x = vec![MaybeUninit::<$t>::uninit(); square * batch];
                zero_gate!("full_piv_lu_solve", {
                    full_piv_lu_solve_into(
                        Op::FullPivLuSolve,
                        view,
                        view,
                        false,
                        &mut x,
                        &mut workspace[offset..offset + ws],
                        seq,
                    )
                    .unwrap()
                });
                out.push(("full_piv_lu_solve.x", unsafe { bytes_uninit(&x) }));
                if compare_wrapper {
                    let mut wx = Vec::new();
                    full_piv_lu_solve(Op::FullPivLuSolve, view, view, false, &mut wx, seq).unwrap();
                    assert_eq!(
                        unsafe { bytes_uninit(&x) },
                        bytes_of(&wx),
                        "full_piv_lu_solve wrapper"
                    );
                }
            }
            // cholesky
            {
                let ws = cholesky::cholesky_workspace::<$t>(n, batch, seq).unwrap();
                let mut workspace = scratch(ws);
                let mut l = vec![MaybeUninit::<$t>::uninit(); square * batch];
                zero_gate!("cholesky", {
                    cholesky::cholesky_into(
                        Op::Cholesky,
                        view,
                        &mut l,
                        &mut workspace[offset..offset + ws],
                        seq,
                    )
                    .unwrap()
                });
                out.push(("cholesky.l", unsafe { bytes_uninit(&l) }));
                if compare_wrapper {
                    let mut wl = Vec::new();
                    cholesky::cholesky(Op::Cholesky, view, &mut wl, seq).unwrap();
                    assert_eq!(
                        unsafe { bytes_uninit(&l) },
                        bytes_of(&wl),
                        "cholesky wrapper"
                    );
                }
            }
            // triangular_solve
            {
                let flags = TriangularSolveFlags {
                    left_side: true,
                    lower: true,
                    transpose_a: false,
                    unit_diagonal: false,
                };
                let ws =
                    triangular_solve::triangular_solve_workspace::<$t>(n, n, n, flags, batch, seq)
                        .unwrap();
                assert_eq!(ws, 0, "triangular_solve needs no workspace");
                let mut workspace = scratch(ws);
                let mut x = vec![MaybeUninit::<$t>::uninit(); square * batch];
                zero_gate!("triangular_solve", {
                    triangular_solve::triangular_solve_into(
                        Op::TriangularSolve,
                        view,
                        view,
                        flags,
                        &mut x,
                        &mut workspace[offset..offset + ws],
                        seq,
                    )
                    .unwrap()
                });
                out.push(("triangular_solve.x", unsafe { bytes_uninit(&x) }));
                if compare_wrapper {
                    let mut wx = Vec::new();
                    triangular_solve(Op::TriangularSolve, view, view, flags, &mut wx, seq).unwrap();
                    assert_eq!(
                        unsafe { bytes_uninit(&x) },
                        bytes_of(&wx),
                        "triangular_solve wrapper"
                    );
                }
            }
            // packed_lu::factor
            {
                let ws = packed_lu::factor_workspace::<$t>(n, n, batch, seq).unwrap();
                let mut workspace = scratch(ws);
                let mut lu = owned.clone();
                let mut pivots = vec![0_i32; n * batch];
                let mut parity = vec![value(0.0); batch];
                zero_gate!("packed_lu::factor", {
                    packed_lu::factor_into(
                        Op::LuFactor,
                        n,
                        n,
                        &mut lu,
                        &mut pivots,
                        &mut parity,
                        &mut workspace[offset..offset + ws],
                        seq,
                    )
                    .unwrap()
                });
                out.push(("packed_lu.lu", bytes_of(&lu)));
                out.push(("packed_lu.pivots", bytes_of(&pivots)));
                out.push(("packed_lu.parity", bytes_of(&parity)));
                if compare_wrapper {
                    let (mut wlu, mut wpiv, mut wpar) = (
                        owned.clone(),
                        vec![0_i32; n * batch],
                        vec![value(0.0); batch],
                    );
                    packed_lu::factor(Op::LuFactor, n, n, &mut wlu, &mut wpiv, &mut wpar, seq)
                        .unwrap();
                    assert_eq!(bytes_of(&lu), bytes_of(&wlu), "packed_lu wrapper lu");
                    assert_eq!(
                        bytes_of(&pivots),
                        bytes_of(&wpiv),
                        "packed_lu wrapper pivots"
                    );
                    assert_eq!(
                        bytes_of(&parity),
                        bytes_of(&wpar),
                        "packed_lu wrapper parity"
                    );
                }
            }
            // packed_lu::solve_prepared
            {
                let mut factored = owned.clone();
                let mut pivots = vec![0_i32; n * batch];
                let mut parity = vec![value(0.0); batch];
                {
                    let mut ws = vec![
                        MaybeUninit::<u8>::uninit();
                        packed_lu::factor_workspace::<$t>(n, n, batch, seq).unwrap()
                    ];
                    packed_lu::factor_into(
                        Op::LuFactor,
                        n,
                        n,
                        &mut factored,
                        &mut pivots,
                        &mut parity,
                        &mut ws,
                        seq,
                    )
                    .unwrap();
                }
                let factor_dims = [n, n, batch];
                let factor_strides = [1isize, n as isize, (n * n) as isize];
                let factor_view =
                    RawStridedRef::new(&factored, &factor_dims, &factor_strides, 0).unwrap();
                let pivot_dims = [n, batch];
                let pivot_strides = [1isize, n as isize];
                let pivot_view =
                    RawStridedRef::new(&pivots, &pivot_dims, &pivot_strides, 0).unwrap();
                let mut x = owned.clone();
                let mut empty: [MaybeUninit<u8>; 0] = [];
                zero_gate!("packed_lu::solve_prepared", {
                    packed_lu::solve_prepared_into(
                        Op::LuSolvePrepared,
                        factor_view,
                        pivot_view,
                        n,
                        &mut x,
                        false,
                        false,
                        &mut empty,
                        seq,
                    )
                    .unwrap()
                });
                out.push(("packed_lu.solve.x", bytes_of(&x)));
                if compare_wrapper {
                    let mut wx = owned.clone();
                    packed_lu::solve_prepared(
                        Op::LuSolvePrepared,
                        factor_view,
                        pivot_view,
                        n,
                        &mut wx,
                        false,
                        false,
                        seq,
                    )
                    .unwrap();
                    assert_eq!(bytes_of(&x), bytes_of(&wx), "packed_lu solve wrapper");
                }
            }
            // householder::compact_factor and apply_reflectors
            {
                let ws = compact_factor_workspace::<$t>(n, n, batch, seq).unwrap();
                let mut workspace = scratch(ws);
                let mut state = owned.clone();
                let mut coeff = vec![MaybeUninit::<$t>::uninit(); n * batch];
                zero_gate!("compact_factor", {
                    compact_factor_into(
                        Op::HouseholderQr,
                        n,
                        n,
                        batch,
                        &mut state,
                        &mut coeff,
                        &mut workspace[offset..offset + ws],
                        seq,
                    )
                    .unwrap()
                });
                out.push(("compact_factor.state", bytes_of(&state)));
                out.push(("compact_factor.coeff", unsafe { bytes_uninit(&coeff) }));
                if compare_wrapper {
                    let (mut wstate, mut wcoeff) = (owned.clone(), Vec::new());
                    compact_factor(
                        Op::HouseholderQr,
                        n,
                        n,
                        batch,
                        &mut wstate,
                        &mut wcoeff,
                        seq,
                    )
                    .unwrap();
                    assert_eq!(
                        bytes_of(&state),
                        bytes_of(&wstate),
                        "compact_factor wrapper state"
                    );
                    assert_eq!(
                        unsafe { bytes_uninit(&coeff) },
                        bytes_of(&wcoeff),
                        "compact_factor wrapper coeff"
                    );
                }
                let shape = ReflectorShape {
                    rows: n,
                    a_cols: n,
                    cols: n,
                    k: n,
                };
                let coeff_values = unsafe { bytes_uninit(&coeff) };
                let _ = coeff_values;
                // SAFETY: `compact_factor_into` returned `Ok`.
                let coeff_values = unsafe { assume(coeff) };
                let ws = apply_reflectors_workspace::<$t>(shape, batch, false, seq).unwrap();
                let mut workspace = scratch(ws);
                let mut c = state.clone();
                zero_gate!("apply_reflectors", {
                    apply_reflectors_into(
                        Op::HouseholderQrQColumns,
                        shape,
                        batch,
                        &state,
                        &coeff_values,
                        &mut c,
                        false,
                        &mut workspace[offset..offset + ws],
                        seq,
                    )
                    .unwrap()
                });
                out.push(("apply_reflectors.c", bytes_of(&c)));
                if compare_wrapper {
                    let mut wc = state.clone();
                    apply_reflectors(
                        Op::HouseholderQrQColumns,
                        shape,
                        batch,
                        &state,
                        &coeff_values,
                        &mut wc,
                        false,
                        seq,
                    )
                    .unwrap();
                    assert_eq!(bytes_of(&c), bytes_of(&wc), "apply_reflectors wrapper");
                }
            }

            out
        };

        // Tests 1, 2 and 3: zero allocations for every family at every size, batch and layout;
        // the batch size does not change the count (both are zero) and the input layout does not
        // change a single bit of the result.
        for n in [1usize, 2, 8] {
            for batch in [1usize, 64] {
                let contiguous = run(n, batch, false, 0, true);
                let shifted = run(n, batch, false, 7, false);
                let transposed = run(n, batch, true, 0, false);
                assert_eq!(
                    contiguous.len(),
                    transposed.len(),
                    "layout comparison is per output"
                );
                for (((name, left), (_, offset)), (_, right)) in
                    contiguous.iter().zip(&shifted).zip(&transposed)
                {
                    assert_eq!(
                        left, offset,
                        "{name}: a 7-byte workspace offset changed the bits (n={n})"
                    );
                    assert_eq!(
                        left, right,
                        "{name}: transposed input changed the bits (n={n})"
                    );
                }
            }
        }
    }};
}

/// Test 1/2/3: `_into` allocates nothing; the batch size and the input layout change nothing;
/// `_into` and the `Vec` wrapper agree bitwise.
#[test]
fn into_is_allocation_free_for_every_family_and_layout() {
    let _guard = serialised();
    dtype_battery!(f32, f32, Complex32);
    dtype_battery!(f64, f64, Complex64);
    dtype_battery!(Complex32, f32, Complex32);
    dtype_battery!(Complex64, f64, Complex64);
}

/// Test 7: a pooled `Exec` adds the pool's own fixed cost (one scratch chunk per lane, task
/// creation), not a per-item cost, so the count is the same for batch 2 and batch 64.
#[test]
fn the_pool_route_adds_a_constant() {
    let _guard = serialised();
    let n = 6;
    let batch = |count: usize| -> Vec<f64> {
        (0..n * n * count)
            .map(|k| {
                let (row, col) = (k % n, (k / n) % n);
                if row == col {
                    4.0 + row as f64
                } else {
                    0.05
                }
            })
            .collect()
    };
    let tp = rayon::ThreadPoolBuilder::new()
        .num_threads(2)
        .build()
        .unwrap();
    let pool = Pool::borrow(&tp);
    let exec = Exec::rayon(&pool).with_budget(2).unwrap();
    let measure = |count: usize| {
        let data = batch(count);
        let dims = [n, n, count];
        let strides = [1isize, n as isize, (n * n) as isize];
        let view = RawStridedRef::new(&data, &dims, &strides, 0).unwrap();
        let ws = qr::qr_workspace::<f64>(n, n, count, exec).unwrap();
        let mut workspace = vec![MaybeUninit::<u8>::uninit(); ws];
        let mut q = vec![MaybeUninit::<f64>::uninit(); n * n * count];
        let mut r = vec![MaybeUninit::<f64>::uninit(); n * n * count];
        for _ in 0..3 {
            qr::qr_into(Op::Qr, view, &mut q, &mut r, &mut workspace, exec).unwrap();
        }
        // Minimum of three rounds: a worker the warm-up never scheduled pays a one-time
        // first touch (thread-local registration allocates on macOS) the first time it
        // picks up a lane; a per-call allocation shows up in every round.
        (0..3)
            .map(|_| {
                allocations_all_threads(|| {
                    qr::qr_into(Op::Qr, view, &mut q, &mut r, &mut workspace, exec).unwrap();
                })
            })
            .min()
            .unwrap_or(0)
    };
    let two = measure(2);
    let many = measure(64);
    println!("pool qr allocations: batch 2 = {two}, batch 64 = {many}");
    assert_eq!(
        two,
        many,
        "the pooled route added {} allocations for 62 more items",
        many.saturating_sub(two)
    );
}

/// The refusal is a typed error and precedes every output write, on each output flavour: fresh
/// (`MaybeUninit`) buffers, a strided caller-owned destination, and an in-place caller buffer.
#[test]
fn an_undersized_workspace_is_refused_before_any_write() {
    let _guard = serialised();
    let n = 4;
    let batch = 3;
    let seq = Exec::serial();
    let data: Vec<f64> = (0..n * n * batch)
        .map(|k| {
            let (row, col) = (k % n, (k / n) % n);
            if row == col {
                4.0 + row as f64
            } else {
                0.05
            }
        })
        .collect();
    let dims = [n, n, batch];
    let strides = [1isize, n as isize, (n * n) as isize];
    let view = RawStridedRef::new(&data, &dims, &strides, 0).unwrap();
    let sentinel = 7.0_f64;

    // Fresh `MaybeUninit` outputs: fill them with a sentinel and read it back.
    let ws = qr::qr_workspace::<f64>(n, n, batch, seq).unwrap();
    assert!(ws > 0);
    let mut workspace = vec![MaybeUninit::<u8>::uninit(); ws - 1];
    let mut q = vec![MaybeUninit::new(sentinel); n * n * batch];
    let mut r = vec![MaybeUninit::new(sentinel); n * n * batch];
    let error = qr::qr_into(Op::Qr, view, &mut q, &mut r, &mut workspace, seq).unwrap_err();
    assert!(
        matches!(
            error,
            Error::InsufficientWorkspace {
                op: Op::Qr,
                needed,
                have,
            } if needed == ws && have == ws - 1
        ),
        "unexpected error: {error:?}"
    );
    // SAFETY: every element still holds the sentinel the test wrote.
    let q = unsafe { assume(q) };
    // SAFETY: as above.
    let r = unsafe { assume(r) };
    assert!(q.iter().all(|&x| x == sentinel), "qr wrote Q on refusal");
    assert!(r.iter().all(|&x| x == sentinel), "qr wrote R on refusal");

    // A strided caller-owned destination (lu::solve).
    let ws = lu_solve_workspace::<f64>(n, n, false, batch, seq).unwrap();
    assert!(ws > 0);
    let mut workspace = vec![MaybeUninit::<u8>::uninit(); ws - 1];
    let mut x = vec![sentinel; n * n * batch];
    let destination = RawStridedMut::new(&mut x, &dims, &strides, 0).unwrap();
    let error = solve_into(
        Op::Solve,
        view,
        Some(view),
        destination,
        false,
        &mut workspace,
        seq,
    )
    .unwrap_err();
    assert!(
        matches!(error, Error::InsufficientWorkspace { op: Op::Solve, .. }),
        "unexpected error: {error:?}"
    );
    assert!(
        x.iter().all(|&value| value == sentinel),
        "solve wrote on refusal"
    );

    // An in-place caller buffer (packed LU).
    let ws = packed_lu::factor_workspace::<f64>(n, n, batch, seq).unwrap();
    assert!(ws > 0);
    let mut workspace = vec![MaybeUninit::<u8>::uninit(); ws - 1];
    let mut lu = data.clone();
    let mut pivots = vec![0_i32; n * batch];
    let mut parity = vec![0.0_f64; batch];
    let error = packed_lu::factor_into(
        Op::LuFactor,
        n,
        n,
        &mut lu,
        &mut pivots,
        &mut parity,
        &mut workspace,
        seq,
    )
    .unwrap_err();
    assert!(
        matches!(
            error,
            Error::InsufficientWorkspace {
                op: Op::LuFactor,
                ..
            }
        ),
        "unexpected error: {error:?}"
    );
    assert_eq!(lu, data, "packed LU factor wrote on refusal");
    assert!(pivots.iter().all(|&p| p == 0), "packed LU wrote pivots");
}

// ---------------------------------------------------------------------------------------------
// Test 6: the GEMM path (tlinalg-rs#57 item 4 / tcontract-rs#121).
// ---------------------------------------------------------------------------------------------

/// The faer matmul backend this host dispatches to for an `m x n x k` product, mirroring
/// `t4a-faer-0.24.6/src/linalg/matmul/mod.rs`: `nano-gemm` at or below the 16³ threshold, then
/// `private-gemm-x86` on x86_64 when AVX2+FMA or AVX-512 is available, else the crates.io `gemm`
/// fallback (the one whose `gemm-common` packing is tcontract-rs#121's subject).
#[cfg(target_arch = "x86_64")]
fn matmul_backend(mnk: usize) -> &'static str {
    if mnk <= 16 * 16 * 16 {
        "nano-gemm"
    } else if std::arch::is_x86_feature_detected!("avx512f")
        || (std::arch::is_x86_feature_detected!("avx2")
            && std::arch::is_x86_feature_detected!("fma"))
    {
        "private-gemm-x86"
    } else {
        "gemm (crates.io 0.19, gemm-common packing)"
    }
}

/// Allocations of one `compact_factor_into` call at `n x n`, `batch` items, on `Exec::serial()`.
#[cfg(target_arch = "x86_64")]
fn compact_factor_allocs(n: usize, batch: usize) -> usize {
    let seq = Exec::serial();
    let data: Vec<f64> = (0..n * n * batch)
        .map(|k| {
            let (row, col) = (k % n, (k / n) % n);
            if row == col {
                4.0 + row as f64
            } else {
                0.05
            }
        })
        .collect();
    let ws = compact_factor_workspace::<f64>(n, n, batch, seq).unwrap();
    let mut workspace = vec![MaybeUninit::<u8>::uninit(); ws];
    let mut state = data.clone();
    let mut coeff = vec![MaybeUninit::<f64>::uninit(); n * batch];
    // Warm every lazily built internal state first; the gate is the steady-state count.
    for _ in 0..2 {
        state.copy_from_slice(&data);
        compact_factor_into(
            Op::HouseholderQr,
            n,
            n,
            batch,
            &mut state,
            &mut coeff,
            &mut workspace,
            seq,
        )
        .unwrap();
    }
    allocations(|| {
        state.copy_from_slice(&data);
        compact_factor_into(
            Op::HouseholderQr,
            n,
            n,
            batch,
            &mut state,
            &mut coeff,
            &mut workspace,
            seq,
        )
        .unwrap();
    })
}

/// The **active upper bound** on today's GEMM-path allocations at sizes 64 and 256.
///
/// The blocked compact-Householder trailing update runs through faer's matmul, whose dispatch
/// depends on the host:
///
/// * `nano-gemm` (at or below the 16³ threshold) and `private-gemm-x86` (x86_64 with AVX2+FMA or
///   AVX-512) need no heap buffer, so the bound there is **zero** and the count must not grow with
///   the batch: a regression that adds a packing buffer per call fails here;
/// * the crates.io `gemm 0.19` fallback (x86_64 without those features) packs into a per-call
///   buffer vector (`gemm-common-0.19.0/src/gemm.rs:490`), so its cost is **per GEMM invocation** —
///   one trailing update per compact-Householder block, per item. That is exactly what
///   tcontract-rs#121 has to remove, so this test pins that behaviour as an upper bound and does
///   **not** demand batch independence from it.
///
/// The measured backend is printed.
#[test]
#[cfg(target_arch = "x86_64")]
fn gemm_path_allocations_are_a_pinned_upper_bound() {
    for n in [64usize, 256] {
        let one = compact_factor_allocs(n, 1);
        let many = compact_factor_allocs(n, 64);
        // The compact-Householder block is capped at 16, so the trailing update is roughly
        // `n x (n - 16) x 16`: above nano-gemm's 16³ threshold at both sizes.
        let backend = matmul_backend(n * n.saturating_sub(16).max(1) * 16);
        println!(
            "compact_factor n={n}: batch 1 = {one}, batch 64 = {many} (matmul backend: {backend})"
        );
        if backend.starts_with("gemm (crates.io") {
            // One `gemm-common` packing vector per GEMM invocation: at most one trailing update per
            // 16-column block, so `n / 16` invocations per item. Pinned per item and per shape.
            let per_item = n.div_ceil(16) + 1;
            assert!(
                one <= per_item,
                "n={n}: {one} allocations on {backend} exceed the per-item bound {per_item}"
            );
            assert!(
                many <= per_item * 64,
                "n={n}: {many} allocations exceed the per-item bound {per_item} x 64 items \
                 (tcontract-rs#121 is where this stops being per item)"
            );
        } else {
            assert_eq!(
                one, many,
                "n={n}: the allocation-free GEMM backend allocates per item"
            );
            assert_eq!(
                one, 0,
                "n={n}: {one} allocations on {backend}, which needs no heap buffer"
            );
        }
    }
}

/// The zero-allocation assertion this site is working toward, kept separately from the active bound.
///
/// On this host's `private-gemm-x86` path it already holds. It is `#[ignore]`d because the
/// crates.io `gemm 0.19` fallback (and `nano-gemm` per item on other architectures) packs into
/// per-call buffers, so the assertion is a gate only once tcontract-rs#121 lands — and the pointer
/// is what keeps this test from being deleted as unreachable.
#[test]
#[ignore = "waits for tcontract-rs#121 (crates.io gemm 0.19 packs into per-call buffers)"]
#[cfg(target_arch = "x86_64")]
fn gemm_path_is_allocation_free() {
    let _guard = serialised();
    for n in [64usize, 256] {
        assert_eq!(
            compact_factor_allocs(n, 64),
            0,
            "n={n}: the GEMM path still allocates (tcontract-rs#121)"
        );
    }
}

// ---------------------------------------------------------------------------------------------
// Overflow refusals (the query contract) and the real forced-route widths.
// ---------------------------------------------------------------------------------------------

/// A shape whose products the requirement builders multiply, on a 64-bit host.
const HUGE: usize = 1usize << 32;

/// Every `*_workspace` refuses a shape-product overflow with `InvalidArgument` instead of panicking
/// in the builder, and returns `0` for an empty batch before it builds anything.
#[test]
fn huge_shapes_are_refused_and_an_empty_batch_needs_nothing() {
    let _guard = serialised();
    let seq = Exec::serial();
    let shape = ReflectorShape {
        rows: HUGE,
        a_cols: HUGE,
        cols: HUGE,
        k: 1,
    };
    // A batch of one: the products overflow.
    let one = 1usize;
    let calls: Vec<(&str, tlinalg::Result<usize>)> = vec![
        ("qr", qr::qr_workspace::<f64>(HUGE, HUGE, one, seq)),
        (
            "rank_revealing_qr",
            qr::rank_revealing_qr_workspace::<f64>(HUGE, HUGE, one, seq),
        ),
        (
            "svd",
            svd::svd_workspace::<f64>(HUGE, HUGE, false, one, seq),
        ),
        (
            "svd_values",
            svd::svd_values_workspace::<Complex64>(HUGE, HUGE, one, seq),
        ),
        ("eigh", eigh::eigh_workspace::<f64>(HUGE, true, one, seq)),
        (
            "eigh_values",
            eigh::eigh_values_workspace::<Complex64>(HUGE, one, seq),
        ),
        ("eig", eig::eig_workspace::<Complex64>(HUGE, true, one, seq)),
        (
            "eig_values",
            eig::eig_values_workspace::<f64>(HUGE, one, seq),
        ),
        ("lu", lu_workspace::<f64>(HUGE, HUGE, one, seq)),
        (
            "lu save",
            lu_solve_workspace::<Complex64>(HUGE, HUGE, false, one, seq),
        ),
        ("full_piv_lu", full_piv_lu_workspace::<f64>(HUGE, one, seq)),
        (
            "full_piv_lu_solve",
            full_piv_lu_solve_workspace::<f64>(HUGE, HUGE, one, seq),
        ),
        (
            "cholesky",
            cholesky::cholesky_workspace::<f64>(HUGE, one, seq),
        ),
        (
            "packed_lu factor",
            packed_lu::factor_workspace::<f64>(HUGE, HUGE, one, seq),
        ),
        (
            "packed_lu factor_solve",
            packed_lu::factor_solve_workspace::<f64>(HUGE, HUGE, one, seq),
        ),
        (
            "compact_factor",
            compact_factor_workspace::<f64>(HUGE, HUGE, one, seq),
        ),
        (
            "apply_reflectors",
            apply_reflectors_workspace::<f64>(shape, one, false, seq),
        ),
    ];
    for (name, result) in calls {
        match result {
            Err(Error::InvalidArgument { .. }) => {}
            other => panic!("{name}: expected InvalidArgument, got {other:?}"),
        }
    }
    // An empty batch resolves the route and returns zero *before* any requirement is built, so a
    // shape-only huge extent cannot overflow a query that will not run anything.
    assert_eq!(qr::qr_workspace::<f64>(HUGE, HUGE, 0, seq).unwrap(), 0);
    assert_eq!(
        svd::svd_workspace::<f64>(HUGE, HUGE, true, 0, seq).unwrap(),
        0
    );
    assert_eq!(eigh::eigh_workspace::<f64>(HUGE, true, 0, seq).unwrap(), 0);
    assert_eq!(eig::eig_workspace::<f64>(HUGE, true, 0, seq).unwrap(), 0);
    assert_eq!(lu_workspace::<f64>(HUGE, HUGE, 0, seq).unwrap(), 0);
    assert_eq!(full_piv_lu_workspace::<f64>(HUGE, 0, seq).unwrap(), 0);
    assert_eq!(
        cholesky::cholesky_workspace::<f64>(HUGE, 0, seq).unwrap(),
        0
    );
    assert_eq!(
        packed_lu::factor_workspace::<f64>(HUGE, HUGE, 0, seq).unwrap(),
        0
    );
    assert_eq!(
        compact_factor_workspace::<f64>(HUGE, HUGE, 0, seq).unwrap(),
        0
    );
    assert_eq!(
        apply_reflectors_workspace::<f64>(shape, 0, false, seq).unwrap(),
        0
    );
    // `_into` and the wrappers never see such a shape in the first place: `strided-view` rejects a
    // descriptor whose reachable offsets overflow `isize`, which bounds every accepted dimension
    // product well below `usize::MAX`. The queries above are the entry points that take shape
    // arguments alone, and they are what this test covers.
    let empty: [f64; 0] = [];
    let huge_dims = [HUGE, HUGE];
    let huge_strides = [1isize, HUGE as isize];
    assert!(
        RawStridedRef::new(&empty, &huge_dims, &huge_strides, 0).is_err(),
        "the descriptor layer rejects an unreachable layout first"
    );
}

/// A genuinely wider forced route: `Inner` and `Split { lanes: 2 }` at budget 4 on a 4-worker
/// pool grant item widths of 4 and 2, and the query's workspace must be the one that call needs.
///
/// Two-sided evidence. `eigh`'s faer request carries a per-thread term, so its query is strictly
/// larger under both routes than under `Exec::serial()` — that is the width reaching the sizing.
/// For every family the wider query is at least the serial one (never an underestimate), and
/// `_into` with exactly the queried workspace reproduces the serial result, so the carve was big
/// enough and the items really ran at the granted width.
#[test]
fn wider_routes_use_the_workspace_they_query() {
    let _guard = serialised();
    let tp = rayon::ThreadPoolBuilder::new()
        .num_threads(4)
        .build()
        .unwrap();
    let pool = Pool::borrow(&tp);
    // Large enough that faer's per-thread scratch term is visible.
    let n = 128;
    let batch = 4;
    let seq = Exec::serial();
    let a: Vec<f64> = (0..n * n * batch)
        .map(|k| {
            let (row, col) = (k % n, (k / n) % n);
            if row == col {
                4.0 + row as f64
            } else {
                0.05 * (((row + 1) * (col + 1)) % 3) as f64
            }
        })
        .collect();
    let dims = [n, n, batch];
    let strides = [1isize, n as isize, (n * n) as isize];
    let view = RawStridedRef::new(&a, &dims, &strides, 0).unwrap();

    for (label, strategy) in [
        ("Inner", BatchStrategy::Inner),
        (
            "Split2",
            BatchStrategy::Split {
                lanes: std::num::NonZeroUsize::new(2).unwrap(),
            },
        ),
    ] {
        let exec = Exec::rayon(&pool)
            .with_budget(4)
            .unwrap()
            .with_batch_policy(BatchPolicy::new(strategy));

        // qr
        let serial_ws = qr::qr_workspace::<f64>(n, n, batch, seq).unwrap();
        let wide_ws = qr::qr_workspace::<f64>(n, n, batch, exec).unwrap();
        println!("{label} qr: serial {serial_ws} bytes, route {wide_ws} bytes");
        assert!(
            wide_ws >= serial_ws,
            "{label} qr: the query underestimates the wider route"
        );
        let mut q = vec![MaybeUninit::<f64>::uninit(); n * n * batch];
        let mut r = vec![MaybeUninit::<f64>::uninit(); n * n * batch];
        let mut workspace = vec![MaybeUninit::<u8>::uninit(); wide_ws];
        qr::qr_into(Op::Qr, view, &mut q, &mut r, &mut workspace, exec).unwrap();
        // SAFETY: `qr_into` returned `Ok`, so every element is initialised.
        let r_wide = unsafe { assume(r) };
        let (mut wq, mut wr) = (Vec::new(), Vec::new());
        qr::qr(Op::Qr, view, &mut wq, &mut wr, seq).unwrap();
        assert_close("qr R", &r_wide, &wr, 1e-8);

        // svd
        let serial_ws = svd::svd_workspace::<f64>(n, n, false, batch, seq).unwrap();
        let wide_ws = svd::svd_workspace::<f64>(n, n, false, batch, exec).unwrap();
        println!("{label} svd: serial {serial_ws} bytes, route {wide_ws} bytes");
        assert!(
            wide_ws >= serial_ws,
            "{label} svd: the query underestimates the wider route"
        );
        let mut u = vec![MaybeUninit::<f64>::uninit(); n * n * batch];
        let mut s = vec![MaybeUninit::<f64>::uninit(); n * batch];
        let mut vt = vec![MaybeUninit::<f64>::uninit(); n * n * batch];
        let mut workspace = vec![MaybeUninit::<u8>::uninit(); wide_ws];
        svd::svd_into(
            Op::Svd,
            view,
            false,
            &mut u,
            &mut s,
            &mut vt,
            &mut workspace,
            exec,
        )
        .unwrap();
        // SAFETY: `svd_into` returned `Ok`.
        let s_wide = unsafe { assume(s) };
        let mut s_serial = Vec::new();
        svd::svd_values(Op::SvdValues, view, &mut s_serial, seq).unwrap();
        assert_close("svd S", &s_wide, &s_serial, 1e-8);

        // eigh: the family whose requirement visibly carries the item width.
        let serial_ws = eigh::eigh_workspace::<f64>(n, true, batch, seq).unwrap();
        let wide_ws = eigh::eigh_workspace::<f64>(n, true, batch, exec).unwrap();
        println!("{label} eigh: serial {serial_ws} bytes, route {wide_ws} bytes");
        assert!(
            wide_ws > serial_ws,
            "{label} eigh: the query did not size at the granted item width"
        );
        let mut values = vec![MaybeUninit::<f64>::uninit(); n * batch];
        let mut vectors = vec![MaybeUninit::<f64>::uninit(); n * n * batch];
        let mut workspace = vec![MaybeUninit::<u8>::uninit(); wide_ws];
        eigh::eigh_into(
            Op::Eigh,
            view,
            &mut values,
            &mut vectors,
            &mut workspace,
            exec,
        )
        .unwrap();
        // SAFETY: `eigh_into` returned `Ok`.
        let values_wide = unsafe { assume(values) };
        let mut values_serial = Vec::new();
        eigh::eigh_values(Op::EighValues, view, &mut values_serial, seq).unwrap();
        assert_close("eigh values", &values_wide, &values_serial, 1e-8);

        // lu
        let serial_ws = lu_workspace::<f64>(n, n, batch, seq).unwrap();
        let wide_ws = lu_workspace::<f64>(n, n, batch, exec).unwrap();
        println!("{label} lu: serial {serial_ws} bytes, route {wide_ws} bytes");
        assert!(
            wide_ws >= serial_ws,
            "{label} lu: the query underestimates the wider route"
        );
        let mut u = vec![MaybeUninit::<f64>::uninit(); n * n * batch];
        let mut l = vec![MaybeUninit::<f64>::uninit(); n * n * batch];
        let mut p = vec![MaybeUninit::<f64>::uninit(); n * n * batch];
        let mut parity = vec![MaybeUninit::<f64>::uninit(); batch];
        let mut workspace = vec![MaybeUninit::<u8>::uninit(); wide_ws];
        lu_into(
            Op::Lu,
            view,
            &mut p,
            &mut l,
            &mut u,
            &mut parity,
            &mut workspace,
            exec,
        )
        .unwrap();
        // SAFETY: `lu_into` returned `Ok`.
        let u_wide = unsafe { assume(u) };
        let (mut wp, mut wl, mut wu, mut wparity) =
            (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        lu(
            Op::Lu,
            view,
            LuFactors {
                p: &mut wp,
                l: &mut wl,
                u: &mut wu,
                parity: &mut wparity,
            },
            seq,
        )
        .unwrap();
        assert_close("lu U", &u_wide, &wu, 1e-8);
    }
}

/// Elementwise close within a relative tolerance (parallelism may reorder arithmetic).
fn assert_close(name: &str, left: &[f64], right: &[f64], tol: f64) {
    assert_eq!(left.len(), right.len(), "{name}: length");
    let scale = right.iter().fold(0.0_f64, |acc, v| acc.max(v.abs()));
    for (index, (a, b)) in left.iter().zip(right).enumerate() {
        assert!(
            (a - b).abs() <= tol * scale.max(1.0),
            "{name}: element {index} is {a}, expected {b}"
        );
    }
}

/// The wrappers keep the caller's operation in every error, and validate their operands in the
/// published order — before the workspace query resolves the route.
#[test]
fn wrappers_keep_the_callers_op_and_validate_before_routing() {
    let _guard = serialised();
    let seq = Exec::serial();
    let vendor =
        Exec::serial().with_batch_policy(BatchPolicy::new(BatchStrategy::WholeBatchVendor));
    let n = 3;
    let a: Vec<f64> = (0..n * n)
        .map(|k| {
            let (row, col) = (k % n, (k / n) % n);
            if row == col {
                4.0 + row as f64
            } else {
                0.05
            }
        })
        .collect();
    let dims = [n, n];
    let strides = [1isize, n as isize];
    let view = RawStridedRef::new(&a, &dims, &strides, 0).unwrap();

    // A caller that names the full-SVD route sees its own `Op` in the refusal, exactly as 0.3.0
    // reported it (`crates/tlinalg-capi` calls `Op::SvdFull` on this entry point).
    let (mut u, mut s, mut vt) = (Vec::new(), Vec::new(), Vec::new());
    let error = svd::svd(Op::SvdFull, view, true, &mut u, &mut s, &mut vt, vendor).unwrap_err();
    assert!(
        matches!(
            error,
            Error::StrategyUnavailable {
                op: Op::SvdFull,
                ..
            }
        ),
        "unexpected error: {error:?}"
    );
    assert!(u.is_empty() && s.is_empty() && vt.is_empty());

    // A malformed right-hand side is reported before the forced-route refusal, as 0.3.0 did.
    let wrong_rows = [1.0_f64; 2];
    let b_dims = [2usize, 1];
    let b_strides = [1isize, 2];
    let b = RawStridedRef::new(&wrong_rows, &b_dims, &b_strides, 0).unwrap();
    let mut x = Vec::new();
    let error = full_piv_lu_solve(Op::FullPivLuSolve, view, b, false, &mut x, vendor).unwrap_err();
    assert!(
        matches!(error, Error::InvalidArgument { .. }),
        "unexpected error: {error:?}"
    );

    // A valid solve under a forced route this provider cannot honor reports the caller's op.
    let mut out = vec![0.0_f64; n];
    let out_dims = [n, 1];
    let out_strides = [1isize, n as isize];
    let error = solve(
        Op::Solve,
        view,
        None,
        RawStridedMut::new(&mut out, &out_dims, &out_strides, 0).unwrap(),
        false,
        seq.with_batch_policy(BatchPolicy::new(BatchStrategy::WholeBatchVendor)),
    )
    .unwrap_err();
    assert!(
        matches!(error, Error::StrategyUnavailable { op: Op::Solve, .. }),
        "unexpected error: {error:?}"
    );
}
