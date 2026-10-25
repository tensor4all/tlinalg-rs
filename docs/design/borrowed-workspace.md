# Borrowed workspace contract for the faer lane (0.3.x)

Status: **implemented, additive (0.3.1)**. `tlinalg-rs#57` requires every family to grow a
`*_workspace(...)` query and a `*_into(...)` entry whose live scratch comes from caller-owned
storage; #57 classifies these additions as non-breaking. The `Vec` entry points stay exactly as
they were — they become wrappers that reserve their outputs and allocate one block of the queried
size per call. The independent pre-review flagged that the borrowing contract must be defined
*before* the implementation (alignment, disjoint lane regions, execution-dependent sizing,
insufficient-space behaviour); this note is that definition, in the form that landed.

## The requirement

```text
pub fn <family>_workspace(<shape args>, batch, exec: Exec<'_>) -> Result<usize>
```

* The value is the number of **bytes** one call needs, worst-case alignment padding included. It
  is `lanes * unaligned_bytes_required(per_lane)`, where `per_lane` is the family's complete
  per-lane inventory (see below) and `lanes` is the lane count the resolved route may use. The
  count does not depend on where the caller's buffer happens to start, so a caller can size once.
* It takes the `&Exec` because the requirement is execution-dependent: faer sizes part of its
  scratch per thread, so `per_lane` is computed at the widest lane share the batch route grants
  (`Route::item_width`) — the *same* `batch::decide` call the execution makes, with the same work
  estimate. A query and its matching call therefore resolve the same lanes, and every lane's actual
  width is at most the one the chunk was sized for.
* A family whose per-lane scratch does not depend on the item width (cholesky has only a faer
  `MemStack`) has a route-independent *per-lane* requirement; the reported total still scales with
  the lane count the route resolves (`Outer` and `Inner` therefore differ). Triangular solve and the
  packed-LU prepared solve need no scratch at all and report `0`.
* The order is fixed: resolve the route first (arithmetic from the shared decision; it never
  fails), then return `0` for `batch == 0` **before** building any
  requirement (a shape-only huge extent must not become a huge or overflowing byte count), then
  check every shape product the requirement builders multiply — including the products inside
  faer's block-size recommendation helpers — and only then compose the requirement. `workspace::lane_requirement`
  is that one implementation, used by every family's query and `_into` route.
* A query reports its family's canonical `Op` in a refusal; a `Vec` wrapper passes the caller's `op`
  through a crate-private variant of the same query, so the error a caller sees is unchanged from
  0.3.0. Wrappers also validate their operands in the published order *before* sizing a workspace.

Why bytes and not a layout type: `tcontract_exec::WorkspaceReq` describes GEMM arena components
and is deliberately checked component-wise, so it is the wrong contract for a generic borrowed
byte region. The requirement is one lane's complete inventory at a route-specific width — not a
provider-independent shape (see the pre-review disposition in the campaign proxy log).

## The storage

```text
pub fn <family>_into(<inputs>, <outputs as caller-owned mut>, workspace: &mut [MaybeUninit<u8>],
                     exec: Exec<'_>) -> Result<()>
```

* `workspace` is a caller-owned byte region of **at least** the queried length. Lane `l` owns
  exactly `[l * per_lane, (l + 1) * per_lane)`, and no two lanes share a byte. Each lane chunk
  carries its own worst-case alignment padding, and every sub-buffer is carved from it with faer's
  `MemStack`, so each typed buffer and each faer scratch region is correctly aligned whatever the
  caller buffer's own alignment is.
* The sub-buffers are carved in a fixed order, so a re-run of the same shape maps the same
  offsets. Lanes are joined before the storage is reused by the caller.
* The region is **disjoint from every input and output**: inputs borrow `&[T]`, outputs are either
  library-owned vectors or caller-owned buffers, and the `Workspace` is a third, distinct borrow.
* A family that needs no scratch accepts an empty slice.

### Outputs

* Fresh-output families take `&mut [MaybeUninit<T>]` (or the real/complex element type of the
  output) holding exactly `item_len * batch` elements, compact column-major in batch order. The
  kernel initialises every element; an item that writes fewer than its `item_len` is a detected
  `Error::Inconsistent` rather than uninitialised memory handed back.
* In-place families keep their existing caller-owned buffers (`&mut [T]`, a `RawStridedMut`
  destination, the packed LU factor/RHS arrays).
* The `Vec` wrappers hand the vector's reserved spare capacity to the same `*_into` route, so
  `set_len` runs only when every item is complete; a failure leaves the vector empty.

## Checked before any write

`_into` resolves the route and checks `workspace.len() >= lanes * per_lane` **before** any output
is touched. A shortage is a typed refusal:

```text
Error::InsufficientWorkspace { op, needed, have }
```

It leaves every library-created vector empty and every caller-owned output unchanged. A larger
workspace than queried passes; a smaller one refuses, even if it matches a different (narrower)
route's requirement. After an error *during* execution, a caller-owned output may hold the items
already completed (the same "in-place buffers are not rolled back" rule the packed LU and solve
families already document).

## Initialization and errors

* Each sub-buffer is initialised exactly as the `Vec` wrapper's counterpart used to be built:
  zeroed where faer reads before writing, full-overwrite where it does not. The reset-per-item
  policy (decomposition outputs, QR block coefficients, the thin-`Q` seed) is preserved.
* On success the storage stays the caller's; nothing is freed. On an error a lane's scratch is
  **abandoned**, not returned or freed: the host owns in-flight accounting and unwind
  replenishment.
* No `Sync` requirement: lanes own disjoint regions, and scratch is not shared across lanes.

## Family inventory

The per-lane live scratch that moved into the borrowed region, family by family:

| family | live scratch |
|---|---|
| cholesky | faer `MemStack` |
| lu (factor) | `Mat` work matrix, two `usize` permutations, faer `MemStack` |
| lu (solve) | the factor scratch, plus the solve `MemStack` |
| full_piv_lu | `Mat` work matrix, four `usize` permutations, faer `MemStack` (factor and solve) |
| qr / rank_revealing_qr | `Mat` work, `Mat` block coefficients, permutation vectors, factor/apply `MemStack`s |
| householder compact / apply | `Mat` compact-WY factor, apply `MemStack` |
| eigh | `Diag` values, faer `MemStack` |
| eig | real: `Mat`, two `Diag`, `MemStack`; complex: `Mat`, `Diag`, `MemStack` |
| svd / svd_values | `Mat` V, `Diag` S, `MemStack` (the max of the default/fallback requests), the check pair when vectors are computed |
| packed_lu | four `usize` permutations, faer `MemStack` (factor and fused factor+solve) |
| triangular_solve | none (the output chunk is the work matrix; faer's triangular solve takes no workspace) |

The SVD row is the one where "execution-dependent sizing" matters: the buffer is sized to the
*larger* of the default and QR-algorithm requests, because a checked decomposition can run both in
the same storage.

## Wrappers

Each `Vec` entry point reserves its outputs and allocates exactly one block of `*_workspace`
bytes per call (zero for triangular solve). With reserved outputs that is the only allocation it
makes; cold outputs allocate for their own capacity. `tests/alloc_counts.rs` pins both the
wrapper's one-allocation steady state and the `*_into` zero-allocation gate, and
`tlinalg-bench/benches/overhead_smoke.rs` (L1) gates the wrapper/`_into` timing ratio.

## Compatibility with 0.3.0

The storage change — padded, over-aligned faer `Mat`/`Diag` lane storage replaced by compact typed
slices carved from borrowed bytes — is inside the same algorithms, so it has to be checked rather
than assumed. `tests/alloc_counts.rs` compares the new wrappers against the **published**
`tlinalg 0.3.0` release (a dev-dependency) on the whole battery: four scalar types, sizes {1, 2, 8},
batches {1, 64}, both input layouts, every family. The measured result on x86_64 is **0 differing
outputs** (bitwise). The `_into` routes are asserted bitwise equal to the wrappers on the same
shapes, so that comparison covers them too; a family that differed would be reported in that test
rather than folded into a tolerance.
