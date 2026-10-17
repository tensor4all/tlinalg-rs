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

* One argument: `exec: tcontract_exec::Exec<'_>` (serial, or a caller-owned pool + budget +
  `BatchPolicy`). **tlinalg owns the batch route**, not the host: `batch::run` asks
  `Exec::batch_decision` from each family's work estimate. Forced strategies are reachable from
  outside; there is no private cutoff. See [library-owned-lanes.md](library-owned-lanes.md).
* `Auto` follows the shared `WorkModel` (estimated work), not a per-item dimension. The step-1
  default model fans small items out over the batch where the old 64 cutoff would, and (the case
  tlinalg-rs#58 was filed for) also fans out an item above the old cutoff; a single large item keeps
  one lane with the whole budget. RHS/reflector target extents enter the estimate as work.
* The driver partitions into **exactly** the decided number of nonempty contiguous ranges, never
  above `batch` or the budget. Every outer child runs its items at the decision's `item_threads`,
  passed to both scratch sizing and faer; a forced `WholeBatchVendor` is refused with
  [`crate::Error::StrategyUnavailable`] before any write.
* **faer 0.24.4 does not guarantee a strict active-thread bound from that count.** Its wide-RHS
  recursion and rounded-up splitting can exceed the requested count. This accepted limitation is
  not hidden by serial fallback or replacement pools; no hard bound on all native work is claimed.
* Scratch is per lane and reused across the items of that lane (faer `MemBuffer` sized once for the
  item shape; host `Workspace` where the family already uses it).
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

Families other than packed LU previously looped sequentially over the batch in tenferro. Batched
extraction enabled outer fan-out; the single-token revision moves the existing default Auto policy
into tlinalg. Host forced strategies/custom thresholds are removed from this provider's API.
Integrated tenferro migration and its performance measurements remain a separate follow-up.

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
* **Lane split and fan-out.** `tcontract_exec::Exec::batch_decision` selects the route; the driver
  partitions `0..batch` into exactly the decided number of nonempty ranges. Several lanes run on
  the context's pool, each item at the decided `item_threads`; one lane runs on the caller when
  `item_threads` is one, or in the pool otherwise. A serial context never resolves to multiple
  lanes.
* **Item offsets.** After normalisation a compact batch has one axis and an item offset is one
  multiply. A non-mergeable view decomposes the item index per item (`O(B)` integer work next to an
  `O(n³)` kernel) instead of carrying an odometer: items are addressed by index from independent
  lanes, so a per-item computation keeps them order-independent.
* **Destination aliasing.** A mutable descriptor is rejected unless its whole layout (matrix and
  batch axes) is injective under a sufficient test: after dropping extent-one axes and sorting by
  absolute stride, every stride exceeds the span of the smaller axes. Input/output overlap cannot
  be expressed: inputs borrow `&[T]` and destinations `&mut [T]`.
* **No driver allocation, within the inline bound.** Normalised batch axes are stored inline (up to
  8 axes; heap only beyond, independently of the numerical workspace). Lane views are derived from
  the lane range on demand, never collected into a list, and a multi-lane run records the lowest
  failing lane in a mutex instead of a per-lane result vector. On one lane the driver allocates
  nothing; `tests/alloc_counts.rs` pins the remaining per-call allocations of compact Householder,
  Cholesky, QR, rank-revealing QR, LU solve, full-pivot LU solve, eigh, SVD, packed LU and
  triangular solve (their faer lane scratch), which do not grow with the batch. The families the
  table does not name are not pinned yet. Worker allocations are observed by
  `tlinalg_testkit::alloc::allocations_all_threads`, which the thread-local counter cannot see.
* **Output assembly.** Library-created vectors are cleared, reserved, and their spare capacity is
  split into disjoint per-lane `MaybeUninit` chunks written sequentially. `set_len` runs only after
  every lane returned `Ok` with its chunk full; otherwise the vectors stay empty.
* **Scratch.** faer work matrices and `MemBuffer`s are built once per lane and reused; buffers
  whose prior contents faer could observe (decomposition outputs, QR block coefficients, the thin
  `Q` seed) are reset per item to the state the pre-batching code allocated them in.
* **Empty `full` SVD.** For a matrix with a zero dimension in `full` mode the two providers differ,
  as they did before the extraction: faer emits the identity `U` and `Vᴴ` blocks while `tlinalg-blas`
  returns empty factors and leaves those blocks to the host.
