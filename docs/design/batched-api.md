# tlinalg batched API (torch-like) — decision 2026-10-05

The maintainer's intent: the whole tlinalg (and tlinalg-blas) public API is **torch-like batched**.
A host calls each family **once per batch**; the library owns the batch loop (and, for tlinalg, the
batch-direction fan-out). Per-matrix entry points become crate-private helpers.

## Shape of every entry point

* Input: one `RawStridedRef<'_, T>` of rank `2 + B`, dims `[rows, cols, b_1, ..., b_B]`, arbitrary
  (validated, non-negative) strides. Matrix dims first, batch dims after (tenferro's column-major
  convention). `B = 0` is a single matrix. Batch iteration over arbitrary-rank strided batch dims is
  the library's job (shared helper), so the host never packs a batch just to call us.
* Outputs: compact column-major per item, items contiguous in batch order (item `i` occupies
  `[i*item_len, (i+1)*item_len)`). Output `Vec`s are **cleared and then filled**; total length is
  `item_len * batch`. No zero-fill pass: sequential paths push; parallel paths write into
  `spare_capacity_mut()` per lane (disjoint `MaybeUninit` chunks) and `set_len` once after every lane
  succeeded. On error the `Vec`s are left empty (cleared) — never partially initialised.
* In-place / direct-output routes (solve into a host view, compact Householder data, packed LU) take
  a batched `RawStridedMut` / compact `&mut [T]` covering the whole batch, and keep their existing
  "unchanged on failure" guarantees per item where they had them.
* Per-item scalar results (LU parity, RRQR permutation, …) are batch-length outputs too.

## Parallelism (tlinalg only)

* One argument: `exec: &mut tcontract_exec::Exec<'_>` (a `threadteam` token, a workspace and a
  `BatchPolicy`; `*_workspace` queries take `&Exec`). **tlinalg owns the batch route**, not the
  host. Forced strategies are reachable from outside; there is no public `LanePlan`. See
  [library-owned-lanes.md](library-owned-lanes.md).
* The route is the shared `Exec::batch_decision`. `Auto` fans out when the decision gives two or
  more lanes and then forks the whole budget over them: a batch with at least `budget` items runs
  one sequential (`Par::Seq`) item stream per slot, a smaller batch gives each item its share of
  the budget as faer width. When `Auto` keeps one lane, the lane runs at the work model's width for
  one item, so a small item stays on the caller instead of entering the pool.
* Forced strategies go through the same decision on the token's budget. The driver partitions into
  **exactly** the decided number of nonempty contiguous ranges, never above `batch` or the budget.
  Lanes are `token.fork`ed, and each lane hands its share to faer as `Par::Team` when the token carries a team and runs faer as
  `Par::Seq` otherwise (no item width is granted then); the scratch is sized for the shape that runs
  (`ParShape::team(width)` or `ParShape::Seq`).
* **faer 0.24.4 does not guarantee a strict active-thread bound from that width.** Its wide-RHS
  recursion and rounded-up splitting can exceed the requested width. The token bounds the lanes it
  starts (checked with the `threadteam` ledger), not faer's internal tasks; no hard bound on all
  native work is claimed.
* Scratch is per lane and reused across the items of that lane. Each family also has a
  `*_workspace` byte query and a `*_into` route that carves that scratch from caller-owned storage
  (see [borrowed-workspace.md](borrowed-workspace.md)); the `Vec` entry points are wrappers that
  allocate one block of the queried size and reserve their outputs.
* Errors: each lane stops at its own first failure and does **not** stop because another lane
  failed; the returned error is the one from the lowest-indexed failing lane, i.e. the
  lowest-indexed failing item — deterministic regardless of scheduling.

## tlinalg-blas

Same batched signatures, minus `par`/`plan` (LAPACK threading is vendor-owned): a sequential loop
over the batch, with **one** workspace query + acquisition per batch call outside the loop, reused
for every item (as `svd_batch` already does). Injected-symbol (`provider-inject`) support unchanged.

## Scope

Applies to every family, including the already-extracted SVD and packed LU (packed LU's chunk API
becomes the batched API; its fan-out moves from tenferro into tlinalg).

## Host (tenferro) keeps

Resource ownership/admission, dtype/placement dispatch, tensor construction, QR gauges, RRQR rank
decision, Householder append/from-factors/gemm, negative-stride rejection, error mapping.

## Behaviour change to record

Families other than packed LU previously looped sequentially over the batch in tenferro; the batch
route is now tlinalg's. Host forced strategies and custom thresholds are expressed only through the
`BatchPolicy` the host puts in its `Exec`.

## Addendum: multiple batch axes (2026-10-05)

* Keep `B >= 0` batch axes. One shared helper normalises them first: drop size-1 axes and
  **coalesce mergeable adjacent axes** (`stride[i+1] == stride[i] * dim[i]`), so the common compact
  case collapses to a single axis with a plain stride step — no per-item multi-index arithmetic.
  Only genuinely non-mergeable views pay for the multi-index walk (an odometer, not div/mod per item).
* Broadcasting: every operand of a call shares the **same batch dims**. The host expresses
  torch-style batch broadcasting by giving an input a **stride of 0** on a batch axis
  (e.g. one `A` against many `B` in solve/triangular_solve/lu_solve). Inputs may have stride-0
  batch axes; the kernel needs no broadcasting logic of its own.
* Mutable descriptors (in-place / direct output) must **not** self-overlap across batch items:
  reject stride 0 on a batch axis of size > 1 and any overlapping item ranges with
  `Error::InvalidArgument`. `RawStridedMut::new` only checks bounds, and parallel lanes writing an
  aliased item would be a data race, so this check is a soundness requirement, not a nicety.
  Overlap between a mutable output and an input is likewise rejected unless the family documents
  in-place semantics.
* Outputs created by the library (`Vec`s) are always compact with one flattened batch extent; the
  host reshapes them to the batch shape it owns.

## Implementation notes (tlinalg)

Recorded with the implementation; they refine, not replace, the contract above.

* **Shared driver.** `crates/tlinalg/src/batch.rs` is the only batch loop: descriptor validation
  and batch-axis normalisation (`BatchedRef`, `BatchedMut`, `BatchAxes`), lane split and fan-out
  (`run`), and output assembly (`Out` for library-created vectors, `InPlace` for host-owned compact
  buffers).
* **Lane split and fan-out.** Every strategy, `Auto` included, uses
  `tcontract_exec::Exec::batch_decision`. The driver partitions `0..batch` into exactly the decided
  number of nonempty ranges. Several lanes are forked over the token, each item at its lane's share
  of the budget; one lane runs on the caller when its width is one, or in the pool otherwise. A
  serial token never resolves to multiple lanes.
* **Item offsets.** After normalisation a compact batch has one axis and an item offset is one
  multiply. A non-mergeable view decomposes the item index per item (`O(B)` integer work next to an
  `O(n³)` kernel) instead of carrying an odometer: items are addressed by index from independent
  lanes, so a per-item computation keeps them order-independent.
* **Destination aliasing.** A mutable descriptor is rejected unless its whole layout (matrix and
  batch axes) is injective under a sufficient test: after dropping extent-one axes and sorting by
  absolute stride, every stride exceeds the span of the smaller axes. Input/output overlap cannot
  be expressed: inputs borrow `&[T]` and destinations `&mut [T]`.
* **No driver allocation, within the inline bound.** Normalised batch axes are stored inline (up to
  8 axes; heap only beyond, independently of the numerical workspace). Item views are derived from
  the item index on demand, never collected into a list, and a multi-lane run records the lowest
  failing lane in a mutex instead of a per-lane result vector. On one lane the driver allocates
  nothing. `tests/alloc_counts.rs` pins two levels: every `*_into` call allocates **nothing** in
  all four scalar types, at sizes {1, 2, 8}, batches {1, 64} and both input layouts, on a serial
  `Exec`; and each `Vec` wrapper allocates exactly one block of `*_workspace` bytes (zero for
  triangular solve), independent of the batch. A pooled `Exec` adds the pool's own constant, also
  independent of the batch. The GEMM path of the blocked compact Householder update is pinned as an
  upper bound (test 6) because crates.io `gemm 0.19` packs into per-call buffers (tcontract-rs#121).
  Worker allocations are observed by `tlinalg_testkit::alloc::allocations_all_threads`, which the
  thread-local counter cannot see.
* **Output assembly.** A `_into` route writes each item into its own caller-owned slice (compact,
  `item_len` elements per item) through the same sequential `Sink`; an item that does not write its
  whole slice is a detected `Error::Inconsistent`, never uninitialised memory handed back. A `Vec`
  wrapper clears and reserves its vector and hands the spare capacity to the same route, so
  `set_len` runs only after every item was full; otherwise the vector stays empty.
* **Scratch.** faer work matrices and `MemStack`s are carved once per lane from the caller's
  borrowed bytes and reused; buffers whose prior contents faer could observe (decomposition
  outputs, QR block coefficients, the thin `Q` seed) are reset per item to the state the
  pre-batching code allocated them in.
* **Empty `full` SVD.** For a matrix with a zero dimension in `full` mode the two providers differ,
  as they did before the extraction: faer emits the identity `U` and `Vᴴ` blocks while `tlinalg-blas`
  returns empty factors and leaves those blocks to the host.
