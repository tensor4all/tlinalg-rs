# tlinalg-rs

Tensor-free linear algebra for the Tensor4all stack. The name is `t` for *tensor* plus `linalg` for
*linear algebra* — read "tee-lin-alg" — and the `t` is the same one that prefixes `tprims`.

`tlinalg` is the numerical layer that tenferro's CPU linear algebra is being extracted into. It owns
the kernels and the batch/scheduling behaviour; the host owns tensors, allocation, dtype dispatch,
placement, execution context and error wrapping.

## Crates

| Crate | Role |
|---|---|
| `tlinalg` | The faer-backed provider, with the batch-direction lane fan-out on the caller's rayon pool. |
| `tlinalg-blas` | The LAPACK/BLAS provider: vendor calls and their argument marshalling, a serial batch loop, vendor-owned threading. |
| `tlinalg-testkit` | Dev-only, unpublished: provider-neutral test helpers (scalar test trait, generators, reference arithmetic, batched layouts, counting allocator). |
| `tlinalg-parity` | Dev-only, unpublished: the same batched cases through both providers, compared gauge-aware. |
| `tlinalg-bench` | Unpublished: Criterion benchmarks of each provider's kernels, faer vs LAPACK. |

The two providers are siblings: neither depends on the other, and each owns its own vocabulary.
The dev-only crates depend on the providers, never the other way round (`tlinalg-testkit` is only
ever a `[dev-dependencies]` entry of a provider).

`tlinalg` is the faer-backed provider. It owns batched kernels (SVD, packed LU, Cholesky,
triangular solve, LU and solve, full-pivot LU, QR and column-pivoted QR, Hermitian and general
eigendecompositions, compact Householder QR), the batch loop and lane fan-out over them, and the
vocabulary its entry points take: borrowed strided I/O, one resource token (`Parallel`) and typed
errors (`Error`). Auto lane policy belongs to the library; no tensor types.

`tlinalg-blas` has the same batched shape without a parallelism token: LAPACK and BLAS own their
threading, the batch loop is serial, and scratch comes from a host `Workspace`, queried and acquired
once per call and reused for every item. It was developed as `tensor4all/tlinalg-blas-rs` and merged
here with its history.

The interface a host requires of its linear-algebra providers is defined by the host: tenferro owns
it and adapts each provider to it.

## Contracts

Every entry point is batched, torch-style: one call per batch over a rank-`2 + B` strided
descriptor, with the library owning the batch loop and the lane fan-out. The contract is
[`docs/design/batched-api.md`](docs/design/batched-api.md).

The crate documentation is the specification for the rest:

* numerical conventions and failure behaviour — crate root of `tlinalg`;
* the parallelism and budget contract — `tlinalg::Parallel`;
* library-owned Auto lanes and the accepted faer count-hint limitation —
  [`docs/design/library-owned-lanes.md`](docs/design/library-owned-lanes.md);
* output assembly and lane scratch — `docs/design/batched-api.md`;
* the error vocabulary — `tlinalg::Error`.

## Status

Extraction in progress. Nothing is published; `publish = false` is set deliberately until the
interface and the package names settle.

## Build

```sh
cargo fmt --all -- --check
cargo clippy -j 16 --workspace --all-targets --features tlinalg-blas/link-openblas,tlinalg-parity/link-openblas,tlinalg-bench/link-openblas -- -D warnings
cargo test -j 16 --workspace --features tlinalg-blas/link-openblas,tlinalg-parity/link-openblas
cargo test -j 16 --workspace --features tlinalg-blas/link-openblas,tlinalg-blas/provider-inject,tlinalg-parity/link-openblas
```

`link-openblas` exists only so `tlinalg-blas` has an executable check of its own; it builds OpenBLAS
from source through `openblas-src`. Tenferro selects the vendor and the injected-symbol path
itself, so the feature is not part of the implementation contract. Without it the workspace still
builds and lints; the LAPACK tests are skipped. `tlinalg-parity/link-openblas` turns on the
cross-provider parity suite, which needs both providers to run.

## Parity

`tlinalg-parity` runs every family (SVD thin/full/values, Cholesky, triangular solve in all flag
combinations, LU, solve, full-pivot LU and its solve, QR, column-pivoted QR, `eigh`/`eigvalsh`,
`eig`/`eigvals`, packed LU factor/prepared/fused solve, and the compact Householder pair) through
both providers for `f32`, `f64`, `Complex32` and `Complex64`, over batch shapes `[]`, `[3]` and
`[2, 3]` (gapped, so no axis merges, and with transposed batch strides), plus a stride-0 broadcast
coefficient for the binary families. What is unique is compared directly (singular values,
eigenvalues as a multiset, Cholesky factors, solutions, permutation-convention LU factors, QR factors
after fixing the `R` diagonal phase); the rest is checked by reconstruction against each
provider's documented convention.

## Benchmarks

`tlinalg-bench` measures the ordinary single-token batched entry points directly: compact inputs,
output vectors reused across iterations, and a recycling
LAPACK workspace, so steady-state allocation is not what is timed.

```sh
# Everything (faer rows only without the feature):
OPENBLAS_NUM_THREADS=1 cargo bench -p tlinalg-bench --features link-openblas
# One family or case (Criterion filter on "family/dtype/row/case"):
OPENBLAS_NUM_THREADS=1 cargo bench -p tlinalg-bench --features link-openblas -- '^eigh/f64/.*/n4xb1024$'
```

Groups are `family/dtype` (`f64` everywhere, `c64` for a representative subset); cases are
`n{n}xb{batch}` (`n x n` matrices, `batch` of them: `n = 2, 4, 8` with `batch = 1, 3, 4, 8, 1024`,
and `n = 32, 128, 512` single matrices) and `t64x24` (a tall matrix, SVD and QR). Each case has up
to three rows:

* `faer-1lane` — one sequential lane: per-item cost plus the batch loop, no threading.
* `faer-{N}t` — an explicit `N`-worker pool and requested budget (`TLINALG_BENCH_THREADS`,
  default the available parallelism); tlinalg's ordinary Auto policy chooses outer lanes or
  faer's intra-item count hint. Use `TLINALG_BENCH_THREADS=1` for overhead baselines.
  **The faer hint is not a hard active-thread limit**; see the `Parallel` budget contract.
* `lapack` — `tlinalg-blas` (with `--features link-openblas`): a serial batch loop, vendor-owned
  threading.

Read `faer-1lane` against `lapack` with `OPENBLAS_NUM_THREADS=1` for a like-for-like kernel
comparison; with OpenBLAS's default thread count, small LAPACK calls pay its threading overhead,
which says more about the vendor configuration than the kernel. `faer-{N}t` against `faer-1lane`
shows what lane fan-out buys for a batch and what intra-item parallelism costs or buys for one
matrix. These are kernel numbers only: tensor construction, dtype dispatch, session entry and pool
checkout belong to tenferro, and integrated route-level performance lives in [`tenferro-benchmark`](https://github.com/tensor4all/tenferro-benchmark).

## License

MIT OR Apache-2.0.
