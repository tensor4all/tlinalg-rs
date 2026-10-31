# Library-owned batch lanes on `tcontract_exec::Exec`

tlinalg owns the batch route of its entry points; the host owns the resources. Tensor formats,
numerical leaves, output/error guarantees and the independent vendor provider are described in
`batched-api.md`. The placement model is the cross-repo one: "Where placement is decided" in
`tensor4all-meta/docs/design/core-boundary-and-execution-ownership.md`.

## Contract

Every native batched entry point takes one `&mut tcontract_exec::Exec` and nothing else to drive
parallelism. The bundle carries one `threadteam::Exec` token (a pool reference and a budget of
slots), the `WorkspaceProvider`, and the shared `BatchPolicy`. The token cannot be copied, so two
operations cannot both claim the whole budget, and the budget it holds is the only budget: this
crate never clamps against a pool and never starts a task by any route other than the token's
`fork`. A serial token runs on the calling thread and enters no pool; nothing falls
back to an ambient pool.

## Route

`batch::decide(exec, batch, work)` resolves a `Route { lanes, slots }` from the shared
`Exec::batch_decision`, which is the one policy source (`BatchPolicy` and `WorkModel` stay with
`tcontract_exec`, rule 4 of the cross-repo document):

* `Auto`: if the decision gives two or more lanes, the batch fans out into `lanes` lanes and the
  call spends the whole budget (`slots = budget`). A batch with at least `budget` items therefore
  runs one sequential item stream per slot, each item at `Par::Seq`; a smaller batch runs one lane
  per item and each lane's share of the budget is its item's faer width. If the decision keeps one
  lane, that lane runs the items one after another at the width the work model gives one item, so
  a small item stays on the caller and a large one takes the budget.
* Forced strategies (`Sequential`, `Outer`, `Inner`, `Split`): the decision's `lanes` lanes, each at
  its `item_threads`, so `slots = lanes * item_threads`.

The decision depends on the budget, the policy and the batch alone, so the workspace query and the
matching `_into` call resolve the same route. The item extent does not enter it; the work estimate
does, through the shared model.

The estimate a family feeds the decision is a `batch::Work`: the whole work of one item, and the
share of it a team *inside one item* can pay back (`payback_share`). The lanes split the whole
estimate, so the fan-out sees all of it; an item's own width comes from the convertible share, and
`Auto` never hands an item more slots than that. A GEMM-shaped family reports a share of one and its
routes are exactly what the decision gives; the decomposition families whose serial phase dominates
report a small share, and those shares are calibrated against measured width curves — see
[team-payback.md](team-payback.md). A forced strategy is the caller's own instruction and is not
narrowed by the cap.

## Per-family work estimates

`batch::run` is fed one item's estimate, the input of the shared decision, so RHS/reflector extents
are expressed as work rather than as a size. The estimates distinguish the actual route:

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

`qr`, `eigh` and `svd` carry a `payback_share` below one; every other row reports `Work::all` (a
share of one).

Every estimate is scaled by the scalar's weight (4 for a complex multiply-add). A prepared packed
solve is deliberately **not** charged for a factorization it does not run.

## Partition and execution

* `lanes` is at most the item count; the driver partitions `0..batch` into **exactly `lanes` nonempty
  contiguous ranges** (`[lane*batch/lanes, (lane+1)*batch/lanes)`).
* `token.limit(slots)` narrows the call to its slots. One lane runs on the token directly;
  several lanes run `token.fork(lanes, ..)`, which divides the slots among the lanes in shares that
  differ by at most one and add up to `slots`. Inside a lane, a token (or child) of more than one
  slot that carries a team hands faer its team as `Par::Team` (`tcontract_exec::with_faer_par`);
  a token of one slot runs faer as `Par::Seq`, and a token without a team does too and never
  enters a pool for it, so the route grants such a token no item width and
  the lanes are its only parallel axis. There is no `Par::rayon` route. The workspace queries use
  `ParShape::team(width)` for a team token of width above one and `ParShape::Seq` otherwise, the
  shape that will run.
* A lane builds its scratch once, from its own chunk of the caller's workspace, and reuses it for
  every item of the lane. Shares differ by at most one, so the workspace query sizes a chunk at the
  widest share (`Route::item_width`), which covers every lane.
* Errors: each lane stops at its own first failure and does **not** stop because another lane
  failed; the returned error is the one from the lowest-indexed failing lane, i.e. the
  lowest-indexed failing item, deterministic regardless of scheduling.

## What the token bounds

The `threadteam` ledger counts the tasks a token starts. `tests/placement.rs` and the driver tests
assert `max_live(root) <= budget` for both regimes (a batch of at least `budget` items, and a smaller
one), `ledger::installs` pins that a tiny item never enters the pool and a large one enters it
once, and `tests/global_pool.rs` (a binary of its own) asserts `global_pool_unstarted()` after
serial and pooled calls.

The count handed to faer is a width, not a strict active-thread bound. With faer 0.24.4,
triangular-solve RHS recursion ignores split tokens and `join_raw` gives both children a rounded-up
half, so even a pool-clamped width can expose more active numerical tasks than requested; how
t4a-faer schedules a width (and whether it opens a spindle region for it) is decided there
(tensor4all/faer#59), not here. The token bounds the lanes it starts, not faer's internal tasks.
Admission across concurrent calls belongs to the resource owner. Vendor BLAS/LAPACK threading stays
vendor-owned.

## Scope and integration

The borrowed-workspace `*_workspace`/`*_into` API is contracted in
[borrowed-workspace.md](borrowed-workspace.md); the `Vec` entry points are wrappers over it.
`*_workspace` queries take `&Exec` (they read the budget and the policy), every other entry point
takes `&mut Exec`. Library-created outputs keep empty-on-error and the deterministic lowest failing
item; direct/in-place guarantees stay per item. The C API forwards the executor's guarded `&mut Exec`
unchanged.

`tlinalg-bench` reads `TLINALG_BENCH_STRATEGY` for the forced-route sweep and builds its pool with
`threadteam::Pool::builder()`. Overhead baselines use 1T settings and report the effective pool
width; 4T/8T are scheduling coverage, not default overhead baselines.

## Acceptance

* Native calls compile with only `&mut Exec`; no public `Parallel`, `LanePlan` or rayon type
  appears, and `cargo deny check bans` rejects a direct `rayon` dependency.
* Driver tests pin the `(lanes, widths)` cases, including a small item staying on the caller, a
  large item taking the whole budget, shares adding up to the budget, the exact nonempty partition,
  forced strategies, which pool the items run on, and the ledger bound.
* Production numerical tests cover four scalar types, batch strides/broadcast, both triangular-solve
  sides, and all packed routes.
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
