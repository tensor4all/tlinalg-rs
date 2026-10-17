# Borrowed workspace contract for the faer lane (design, 0.3)

Status: **contracted, not implemented** in this change. `tlinalg-rs#57`/`#58` require every family
to grow a `*_workspace(...) -> WorkspaceReq` query and a `*_into(..., &mut Workspace)` entry whose
live scratch comes from caller-owned storage. The independent pre-review flagged that the
borrowing contract must be defined *before* that implementation (alignment, disjoint lane regions,
execution-dependent sizing, insufficient-space behaviour); this note is that definition. The
existing `Vec`-output wrappers stay as they are until it lands.

## The requirement

```text
pub struct WorkspaceReq {
    pub per_lane_bytes: usize,   // one lane's live scratch, alignment-padded
    pub align: usize,            // the strictest alignment any family buffer needs
    pub lanes: usize,            // lanes the decided route may run
}
```

* `per_lane_bytes` is a **conservative bound** over the routes a family can take on this shape and
  `Exec`, not the execution's exact size: it is the maximum live scratch of one lane, including the
  family's complete inventory (see below) and the maximum of a default/fallback pair where a family
  may run twice (SVD's divide-and-conquer check and its QR repeat share one buffer sized to the
  larger request). It is component-wise per family, not a scalar `len`: different families carve
  different buffers.
* `lanes` is the lane count the route may use, `min(exec.budget(), batch)`; a query therefore takes
  the `Exec` as well as the shape. `Auto`'s lane count is already resolved by `batch_decision`, so
  the query calls the same function; it never returns a route-dependent under-estimate.
* A family whose scratch does not depend on the decided item width reports the same requirement for
  every route; a family that sizes faer scratch per item thread (`faer::Par`) uses the item width the
  route can grant (`Split`/`Inner`), because that is the widest one.

## The storage

`Workspace<'w>` is a caller-owned, **page-aligned** (4096-byte) byte region:

* base alignment `max(WorkspaceReq::align, 4096)`, so every lane region and every typed sub-buffer
  starts aligned for its element (`align_of::<T>()`) and for faer's internal `Align`;
* lane `l` of `lanes` owns exactly `[l * per_lane_bytes, (l + 1) * per_lane_bytes)`, and no two lanes
  share a byte. A lane's sub-slices are carved in a fixed order, so a re-run of the same shape maps
  the same offsets. Lanes are joined before the storage is reused or read by the caller.
* the region is **disjoint from every input and output**: inputs borrow `&[T]`, outputs are either
  library-owned `Vec`s or caller-owned `RawStridedMut`s, and the `Workspace` is a third, distinct
  borrow. No family may alias workspace memory with an operand.

## Checked before any write

`*_into` checks `have.covered_by(&req)` **component-wise** (`per_lane_bytes <= have`, `align >=`,
`lanes >=`) after clearing/reserving library-owned outputs and after resolving the route, but
*before* writing any item. A shortage is a typed refusal:

```text
Error::InsufficientWorkspace { op, needed: usize, have: usize }
```

and it leaves every library-created output empty and every caller-owned output unchanged — the same
"no partial writes on refusal" rule the driver already has for a forced unavailable strategy. A
larger workspace never stands in for a missing component.

`have` is the caller's actual capacity, not the requirement it was sized to: a call with a bigger
workspace than it asked for passes, and a call with the matching requirement but a different route
that needs more refuses.

## Initialization and errors

* Each sub-buffer is initialised exactly as the `Vec` wrapper initialises its counterpart today:
  zeroed where faer reads before writing, full-overwrite where it does not, capacity-only where the
  family populates with `push`. The reset-per-item policy (decomposition outputs, QR block
  coefficients, the thin-`Q` seed) is preserved.
* On success the storage stays the caller's; nothing is freed. On an error, a lane's scratch is
  **abandoned**, not returned or freed: the host owns in-flight accounting and unwind
  replenishment, matching `tcontract_exec`'s `WorkspaceProvider` abandonment contract. A provider
  must not release on a path where the `Vec` wrapper would drop.
* No `Sync` requirement: lanes own disjoint regions, and scratch is not shared across lanes.

## Family inventory

The per-lane live scratch that must move into the borrowed region, family by family:

| family | live scratch |
|---|---|
| cholesky | `MemBuffer` |
| lu (factor / solve) | `Mat`, four `Vec<usize>` permutations, `MemBuffer` |
| full_piv_lu | `Mat`, four `Vec<usize>` permutations, `MemBuffer` (factor and solve) |
| qr / rank_revealing_qr | `Mat` work, `Mat` coefficients, permutation vectors, `MemBuffer`s; rank-revealing adds `jpvt` and the per-item check |
| householder compact/apply | `Mat` compact-WY factor, `MemBuffer` |
| eigh | `Diag` values, `MemBuffer` |
| eig | real: `Mat`, two `Diag`, `MemBuffer`; complex: `Mat`, `Diag`, `MemBuffer` |
| svd | `Mat` V, `Diag` S, `MemBuffer` (the max of the default/fallback scratch), the `Check` pair when vectors are computed |
| packed_lu | four `Vec<usize>` permutations, `MemBuffer` |

The SVD row is the one where "execution-dependent sizing" matters: the buffer must be sized to the
*larger* of the default and QR-algorithm requests, because a checked decomposition can run both in
the same storage.
