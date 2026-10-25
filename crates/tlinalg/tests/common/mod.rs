//! Shared helpers for the per-family kernel tests. The provider-neutral part (scalar conversion,
//! deterministic matrices, reference arithmetic, batched layouts) lives in `tlinalg-testkit`; this
//! module re-exports it and adds what is specific to `tlinalg` (its scalar bound, pools and lane
//! plans).

#![allow(dead_code, unused_imports)]

pub mod single;

/// In scope for its methods (`to_c64`, `from_c64`) on every scalar, including output types such as
/// `ScalarEntity::Complex` that only carry the shared bound.
pub use tlinalg_testkit::TestScalar as SharedTestScalar;
pub use tlinalg_testkit::{
    adjoint, assert_close, batch_buf, broadcast_buf, for_each_scalar, hermitian, hpd, identity,
    is_symmetric, matmul, matrix, narrow, padded, pivoting, transpose, widen, BatchBuf, Layout,
};

/// A scalar the faer tests instantiate a kernel for: the shared test scalar, as a `tlinalg` scalar.
pub trait TestScalar: tlinalg_testkit::TestScalar + tlinalg::Scalar {}

impl<T: tlinalg_testkit::TestScalar + tlinalg::Scalar> TestScalar for T {}

/// A placeholder for the scratch argument the pre-batching triangular solve took.
#[derive(Default)]
pub struct CountingWorkspace;

/// A three-worker pool, owned by the returned [`tcontract_exec::Pool`] so a token can borrow it.
pub fn lanes_pool() -> tcontract_exec::Pool<'static> {
    let pool = threadteam::Pool::builder().threads(3).build().unwrap();
    tcontract_exec::Pool::owned(pool)
}

/// A three-thread batch token on `pool`. Its default-strategy model prices a multiply-add at a
/// millisecond, so the tiny items of these lane-semantics tests still fan out; the default work
/// gate (a tiny batch stays on the caller) is tested in `batch::tests::scheduling`.
pub fn pool_token<'a>(pool: &'a tcontract_exec::Pool<'a>) -> tcontract_exec::Exec<'a> {
    use tcontract_exec::batch::{BatchPolicy, BatchStrategy, WorkModel};
    let model = WorkModel::default().with_muladds_per_ns(1e-6);
    tcontract_exec::Exec::new(pool, 3)
        .unwrap()
        .with_batch_policy(BatchPolicy::new(BatchStrategy::Auto).with_model(model))
}
