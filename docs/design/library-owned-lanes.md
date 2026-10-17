# Library-owned batch lanes on `tcontract_exec::Exec`

Owning-layer prerequisite for tensor4all/tenferro-rs#2004, U6, updated for the 0.3 chain
(tensor4all/tcontract-rs#123 step 1, tlinalg-rs#58). This supersedes the host-policy parts of
`batched-api.md`; tensor formats, numerical leaves, output/error guarantees and the independent
vendor provider stay unchanged.

## Contract

Every native batched entry point takes one `tcontract_exec::Exec` token and nothing else to drive
parallelism. The token carries the caller's pool and budget, the `WorkspaceProvider` the future
allocation-free API needs, and the shared `BatchPolicy`. `tlinalg::Parallel`, `LanePlan` and
`AUTO_FAN_OUT_MAX_ITEM_DIM` are gone. Forced strategies are reachable from outside, and
`WholeBatchVendor` — which this provider has no route for — is refused with the typed
`Error::StrategyUnavailable` before any output is written (including the empty packed routes).

## Scheduling change, explicitly

0.2 chose lanes from a **dimension cutoff**: fan out when `max(rows, cols) <= 64` (with RHS columns
counted) and the batch fills every worker; otherwise one lane with the whole budget passed to faer.
0.3 asks `Exec::batch_decision` with each family's `item_muladds` and the shared `WorkModel`
(tenferro's restored `CpuBatchThresholds`), so the decision follows estimated work, not a
per-item extent. With the step-1 default model (`min_work_per_lane_ns = 0`) this means:

* a small item fans out to `min(budget, items)` lanes where the 64 cutoff would fan out too;
* an item **above** the old 64 cutoff now also fans out, because its extent no longer disables
  lanes (the case tlinalg-rs#58 was filed for: a 16x16 solve with many RHS columns);
* a single large item keeps one lane with the whole budget, as before.

This is the change the independent pre-review asked to make explicit and pin. The pinned cases
live in `crates/tlinalg/src/batch/tests/scheduling.rs` (`driver_bounds_...`,
`solve_and_reflector_extents_enter_the_work_estimate`, `packed_routes_use_their_own_work_estimate`).
The shared defaults are still provisional: tenferro#2000's work-model sweep is the umbrella's later
step, and `WorkModel` is configurable on the `Exec` meanwhile.

## Per-family work estimates

`batch::run` is fed one item's multiply-adds, so the RHS/reflector extents enter the decision as
work instead of as a size. The estimates distinguish the actual route:

| route | `item_muladds` |
|---|---|
| cholesky | `n^3/6` |
| lu (factor) | `m n k - (m+n) k^2/2 + k^3/3`, `k = min(m,n)` |
| full_piv_lu (factor) | `n^3/3` |
| lu::solve | factor + `n^2 nrhs` (it factors then solves) |
| full_piv_lu_solve | factor + `n^2 nrhs` |
| triangular_solve | `n^2 k/2`, `k` the solved RHS count |
| qr / rank_revealing_qr | `m n k - k^3/3` |
| compact_factor / apply_reflectors | `rows * cols * reflectors` |
| eigh / eig | `n^3` |
| svd | `m n k` |
| packed_lu factor | `m n k - (m+n) k^2/2 + k^3/3` |
| packed_lu solve_prepared | `n^2 nrhs` (prepared) |
| packed_lu factor_solve | factor + `n^2 nrhs` |

Every estimate is scaled by the scalar's weight (4 for a complex multiply-add). A prepared packed
solve is deliberately **not** charged for a factorization it does not run.

## Partition and execution

* `lanes` is the decision's lane count, at most the item count; the driver partitions `0..batch`
  into **exactly `lanes` nonempty contiguous ranges** (`[lane*batch/lanes, (lane+1)*batch/lanes)`).
  0.2's `div_ceil` chunking could return fewer lanes than requested (five items and four lanes gave
  three); 0.3 gives four.
* One lane runs on the caller for `item_threads == 1`, or inside the pool for `item_threads > 1`.
  Several lanes run concurrently on the context's pool; the decision's `item_threads` is passed to
  **both** the per-lane scratch sizing and the numerical item call, so a `Split`/`Inner`-style
  forced width is not silently dropped inside the lanes.
* Errors: each lane stops at its own first failure and does **not** stop because another lane
  failed; the returned error is the one from the lowest-indexed failing lane, i.e. the
  lowest-indexed failing item — deterministic regardless of scheduling.

## faer's hint limitation

`item_threads` is the faer count requested on the pool, not a strict active-thread bound. With faer
0.24.4, triangular-solve RHS recursion ignores split tokens and `join_raw` gives both children a
rounded-up half, so even a pool-clamped faer count can expose more active numerical tasks than
requested. The maintainer accepted proceeding with this limitation recorded: pass the effective
count and install only the supplied pool, no silent serial downgrade or replacement pool. The
pool's physical worker count remains its bound, not the hint. Admission across concurrent calls
belongs to the resource owner. Vendor BLAS/LAPACK threading stays vendor-owned.

## Scope and integration

The borrowed-workspace `*_workspace`/`*_into` API (`tlinalg-rs#57`) is contracted in
[borrowed-workspace.md](borrowed-workspace.md) but not implemented in this change; the `Vec`
wrappers remain. When it lands, the `Workspace` is a component-wise checked per-lane region and an
insufficient workspace is a typed refusal before any write.

Every native entry point, executable doc, native test, parity call and benchmark moves to `Exec`;
`tlinalg-bench` reads `TLINALG_BENCH_STRATEGY` for the forced-route sweep. The C API migrates its
tcontract-capi pin to the 0.3.0 revision and forwards that executor's guarded `Exec` instead of
building a fresh `Pool` wrapper per call. Library-created outputs keep empty-on-error and the
deterministic lowest failing item; direct/in-place guarantees stay per item. Vendor signatures, the
serial batch loop, the queried-once workspace and vendor threading are unchanged.

Benchmarks use the ordinary API only, with an explicit `Exec` budget. Overhead baselines use 1T
settings and report the effective pool width; 4T/8T are scheduling coverage, not default overhead
baselines. No speedup or completed integrated-tenferro performance claim is made here.

## Acceptance

* Native calls compile with only `Exec`; no public `Parallel`, `LanePlan` or cutoff remains.
* Driver tests pin the new `(lanes, item_threads)` cases (including RHS/reflector extents and the
  packed routes), the exact nonempty partition, forced strategies, and the `WholeBatchVendor`
  refusal on empty batches. Do not mistake observing a faer count for proving its active-thread
  bound.
* Production numerical tests cover four scalar types, batch strides/broadcast, both
  triangular-solve sides, and all packed routes.
* Allocation gates measure provider-controlled scratch separately from dispatch; the inline
  metadata bound (8 non-mergeable axes) is documented, and a cross-thread window
  (`tlinalg_testkit::alloc::allocations_all_threads`) observes worker allocations that the
  thread-local counter misses.
* Repository fmt, clippy with warnings denied, debug/CI tests, and the release native tests.

## Provenance

The work-model vocabulary and defaults are adapted from same-project `tenferro-rs` commit
`527f58cd`, `crates/tenferro-linalg/src/cpu/tlinalg.rs:42-115` and
`crates/tenferro-cpu/src/batch_policy.rs:78-97,312-316` (MIT OR Apache-2.0); the shared types live in
`tcontract_exec::batch`. Preserve existing numerical sources/notices; no new third-party numerical
implementation or copied tests.
