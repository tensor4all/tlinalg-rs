//! The `payback_share` of a decomposition family: the boundary at which a team inside one item
//! starts to pay.
//!
//! The crate's width rule grants a team when the serial estimate reaches `PAYS` times the dispatch
//! cost of that team (`threadteam`), and it reads the dispatch costs of the host class it runs on.
//! For a GEMM-shaped family the whole estimate is convertible, so `Work::all` passes the decision's
//! width through unchanged. For `svd`/`svd_values`, `qr` and `eigh` only a small share of the
//! estimate is convertible: those families spend most of their time in a serial
//! (bi/tri)diagonalization phase, and a team entered for the trailing update alone makes the call
//! slower — measured, not assumed (`tlinalg-rs#88`, `docs/design/team-payback.md`).
//!
//! These tests are the acceptance record for the shares. Each row is one cell of the measured width
//! curve (96-core `x86_64-avx512` node, one exclusive, `tlinalg` 0.8.0, batch of one, team tokens,
//! width 8): the crate's own estimate of that cell, and the measured `s(8) = t(1)/t(8)`. The model
//! must refuse the team exactly where the team loses time and grant it where the team wins. Cells
//! measured within 10% of break-even are excluded: they say nothing about the boundary either way.
//!
//! The dispatch costs are the measured `x86_64-avx512` team row of `threadteam`, so the table does
//! not depend on the host that runs the tests.

use core::num::NonZeroUsize;
use core::time::Duration;

use num_complex::Complex64;
use threadteam::{width_for_by, DispatchCosts, DispatchKind, HostClass};

use crate::batch::Work;

/// `s(8)` within this of break-even says nothing about the grant boundary.
const NEUTRAL: f64 = 0.10;

/// The rate the measured table priced the estimates at: the `x86_64-avx512` row of the model.
const X86_RATE: f64 = 15.6;

/// One measured cell: `(family and dtype, rows, cols, measured s(8))`.
type Cell = (&'static str, usize, usize, f64);

/// The f64 SVD curve (`tlinalg-rs#88`, first table).
const SVD_F64: &[Cell] = &[
    ("svd f64", 16, 16, 1.08),
    ("svd f64", 32, 32, 0.98),
    ("svd f64", 48, 48, 0.85),
    ("svd f64", 64, 64, 0.76),
    ("svd f64", 96, 96, 0.64),
    ("svd f64", 128, 128, 0.60),
    ("svd f64", 192, 192, 0.73),
    ("svd f64", 256, 256, 0.81),
    ("svd f64", 512, 512, 1.36),
    ("svd f64", 256, 64, 0.79),
    ("svd f64", 512, 128, 0.72),
    ("svd f64", 64, 256, 0.77),
];

/// The f64 QR curve.
const QR_F64: &[Cell] = &[
    ("qr f64", 16, 16, 0.95),
    ("qr f64", 32, 32, 1.00),
    ("qr f64", 48, 48, 0.59),
    ("qr f64", 64, 64, 0.58),
    ("qr f64", 96, 96, 0.65),
    ("qr f64", 128, 128, 0.74),
    ("qr f64", 192, 192, 1.03),
    ("qr f64", 256, 256, 1.23),
    ("qr f64", 512, 512, 2.16),
    ("qr f64", 256, 64, 0.72),
    ("qr f64", 512, 128, 0.97),
    ("qr f64", 64, 256, 0.73),
];

/// The f64 `eigh` curve.
const EIGH_F64: &[Cell] = &[
    ("eigh f64", 16, 16, 0.98),
    ("eigh f64", 32, 32, 0.98),
    ("eigh f64", 48, 48, 0.88),
    ("eigh f64", 64, 64, 0.90),
    ("eigh f64", 96, 96, 0.73),
    ("eigh f64", 128, 128, 0.75),
    ("eigh f64", 192, 192, 0.87),
    ("eigh f64", 256, 256, 0.92),
    ("eigh f64", 512, 512, 1.17),
];

/// The c64 curves (SVD, QR, `eigh`), the same node.
const C64: &[Cell] = &[
    ("svd c64", 48, 48, 0.80),
    ("svd c64", 128, 128, 0.74),
    ("svd c64", 256, 64, 0.78),
    ("qr c64", 48, 48, 0.67),
    ("qr c64", 128, 128, 1.10),
    ("qr c64", 256, 64, 0.98),
    ("eigh c64", 48, 48, 0.85),
    ("eigh c64", 128, 128, 0.78),
];

/// The crate's estimate for one measured cell, in the cell's own scalar.
fn cell_work((family, m, n, _): Cell) -> Work {
    let mut parts = family.split(' ');
    let family = parts.next().expect("a cell names its family");
    let dtype = parts.next().expect("a cell names its dtype");
    match (family, dtype) {
        ("svd", "f64") => crate::svd::work_for_test::<f64>(m, n),
        ("svd", "c64") => crate::svd::work_for_test::<Complex64>(m, n),
        ("qr", "f64") => crate::qr::work_for_test::<f64>(m, n),
        ("qr", "c64") => crate::qr::work_for_test::<Complex64>(m, n),
        ("eigh", "f64") => crate::eigh::work_for_test::<f64>(n),
        ("eigh", "c64") => crate::eigh::work_for_test::<Complex64>(n),
        other => panic!("unknown family/dtype {other:?}"),
    }
}

/// The nanosecond estimate of a `Work`, at the rate the measured table used.
fn estimate_ns(work: Work) -> f64 {
    work.total / X86_RATE
}

/// The nanoseconds of a `Work`'s estimate that a team inside one item can pay back.
fn convertible_ns(work: Work) -> f64 {
    work.total * work.payback_share / X86_RATE
}

/// The width the rule would grant an item with `ns` nanoseconds of convertible work, on a team
/// token of the measured `x86_64-avx512` host class.
fn grant_width(ns: f64, budget: usize) -> usize {
    let costs = DispatchCosts::for_class(HostClass::X86_64Avx512);
    let ns = if ns.is_finite() && ns > 0.0 { ns } else { 0.0 };
    let estimate = Duration::from_nanos(ns as u64);
    let budget = NonZeroUsize::new(budget).expect("a nonzero budget");
    width_for_by(estimate, 1, budget, |width| {
        costs.cost(DispatchKind::Team, width)
    })
    .get()
}

#[test]
fn the_shares_put_the_grant_boundary_where_the_measurements_put_it() {
    let mut checked = 0;
    for table in [SVD_F64, QR_F64, EIGH_F64, C64] {
        for cell in table {
            let (family, _, _, s8) = *cell;
            if (s8 - 1.0).abs() < NEUTRAL {
                continue;
            }
            let work = cell_work(*cell);
            let width = grant_width(convertible_ns(work), 8);
            let expected = if s8 > 1.0 { "grant" } else { "refuse" };
            let got = if width > 1 { "grant" } else { "refuse" };
            assert_eq!(
                got,
                expected,
                "{family} {m}x{n}: measured s(8) = {s8}, the model grants width {width} \
                 (estimate {:.0} ns, convertible {:.0} ns)",
                estimate_ns(work),
                convertible_ns(work),
                m = cell.1,
                n = cell.2,
            );
            checked += 1;
        }
    }
    assert!(checked >= 28, "only {checked} cells were checked");
}

/// The cap is inert when the whole estimate is convertible: a GEMM-shaped family keeps the
/// decision's width, so no GEMM or solve route changed.
#[test]
fn a_fully_convertible_work_is_not_capped() {
    let work = Work::all(1.0e8);
    assert_eq!(work.payback_share, 1.0);
    assert_eq!(convertible_ns(work), estimate_ns(work));
    // A wide team is what the rule grants a large GEMM-shaped item, and the cap leaves it alone.
    assert_eq!(grant_width(convertible_ns(work), 8), 8);
    // A decomposition's convertible part of the same estimate is far below what a team needs.
    let svd = crate::svd::work_for_test::<f64>(48, 48);
    assert!(grant_width(convertible_ns(svd), 8) == 1);
}

/// On `Auto`, an item whose whole estimate is large but whose convertible part is not receives no
/// team; the same family at a size where the convertible part pays does. The comparison is against
/// `Work::all`, which is what the decision saw before the cap: the cap narrows, never widens.
#[test]
fn auto_refuses_an_item_a_team_cannot_pay_back() {
    use tcontract_exec::batch::{BatchPolicy, BatchStrategy};
    use tcontract_exec::Exec;

    let Ok(team) = threadteam::Pool::builder()
        .threads(8)
        .acquire_timeout(Duration::from_secs(10))
        .build()
        .unwrap()
        .into_team(8)
    else {
        return; // The host cannot reserve eight members; nothing to decide about.
    };
    let make = || Exec::from_token(team.exec(8), None, BatchPolicy::new(BatchStrategy::Auto));
    let width_of =
        |work: Work, exec: &mut Exec<'_>| crate::batch::decide(exec, 1, work).item_width();

    // A 256 x 256 SVD is 1.1e6 multiply-adds of estimate, but only 6e-4 of that is convertible:
    // the decision alone would grant a team (measured `s(8)` = 0.81), the cap refuses it.
    let svd = crate::svd::work_for_test::<f64>(256, 256);
    let mut exec = make();
    assert_eq!(width_of(svd, &mut exec), 1);
    assert!(
        width_of(Work::all(svd.total), &mut exec) > 1,
        "the decision alone grants a team"
    );

    // A QR of the same shape converts ten times more, and a large enough item still pays.
    let qr = crate::qr::work_for_test::<f64>(128, 128);
    assert_eq!(width_of(qr, &mut exec), 1);
    let qr_big = crate::qr::work_for_test::<f64>(512, 512);
    assert!(width_of(qr_big, &mut exec) > 1);

    // A GEMM-shaped family is untouched: its whole estimate is convertible.
    let gemm = Work::all(1.0e6);
    assert_eq!(
        width_of(gemm, &mut exec),
        width_of(Work::all(1.0e6), &mut exec)
    );
    assert!(width_of(gemm, &mut exec) > 1);
}
