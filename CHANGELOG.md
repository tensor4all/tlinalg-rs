# Changelog

Changes to the published `tlinalg` crate. `tlinalg-blas` is versioned separately; a release here
names it only when its version moves.

## Unreleased

### Added: rank-truncated full-pivot LU (`full_piv_lu_truncated[_into]`)

`full_piv_lu_truncated` and `full_piv_lu_truncated_into` take a `FullPivLuTruncation`
(`max_rank`, `abs_tol`, `rel_tol`) and add a per-item `rank` output (tensor4all/tenferro-rs#2488
step 2). The kernel is faer's: `faer::linalg::lu::full_pivoting::factor::lu_in_place` with the
truncation fields of `FullPivLuParams` and `FullPivLuInfo::rank` (tensor4all/faer#294, `colmat`
0.34.0); the stopping rule is forwarded unchanged (`|pivot_j| > max(abs_tol, rel_tol * |pivot_0|)`,
`rel_tol` relative to the first pivot; faer drops a threshold that is not finite, `abs_tol`
included, which for validated tolerances needs `rel_tol * |pivot_0|` to overflow, so `rel_tol > 1`). With the capacity `c = min(m, n, max_rank)` the outputs are
`P (m x m)`, `L (m x c)`, `U (c x n)`, `Q (n x n)`, parity and rank; an item that stops at `r < c`
is zero in `L[:, r..]` and `U[r.., :]` (the Schur complement faer leaves in the work matrix is never
copied out), and its permutations stay full. Items of one batch may stop at different ranks. The
workspace is `full_piv_lu_workspace`'s. A negative tolerance, or one that does not stay finite in the scalar's real type (`1e300` for `f32`), is
`InvalidArgument`.

This is additive: `full_piv_lu`, `full_piv_lu_into` and the C API are unchanged, and with
`FullPivLuTruncation::default()` the new entry points return bitwise the factors of `full_piv_lu`
with `rank = min(m, n)`. The C API does not expose truncation yet.

## tlinalg 0.12.0

This release is **breaking**. It has two independent breaking changes.

### Breaking: `full_piv_lu` takes `m x n`, and its work query takes `(m, n)`

`full_piv_lu`, `full_piv_lu_into` and `full_piv_lu_workspace` accept a rectangular `m x n` core
with a batch. Per item and with `k = min(m, n)`, the outputs are the `m x m` row permutation `P`,
the unit-lower `m x k` factor `L`, the upper `k x n` factor `U`, the `n x n` column permutation
`Q` and the parity, with `P A Qᵀ = L U` as before. `estimate_work` and `estimate_payback_work`
take `(m, n)`; the estimate is the same `m n k - (m+n) k²/2 + k³/3` the partial-pivot `lu`
reports, which is the previous `n³/3` at `m = n`. Square input produces the same factors
bitwise. The C API's `tlinalg_full_piv_lu_sizes` reports one element count per factor, so the
struct does not change. `full_piv_lu_solve` stays square-only. Rank truncation stays unexposed.

### Breaking: the dense-matrix dependency is `colmat` 0.34, on threadteam 0.6 and tcontract-exec 0.13

The dependency moved from `t4a-faer` 0.33.0 to the renamed crate `colmat` 0.34.0
(tensor4all/faer → tensor4all/colmat-rs). The dependency key stays `faer` and the source keeps
`use faer::...`; the upstream library target is `colmat`, which the key alias hides. Every tlinalg
signature is unchanged for this part of the release. What changes is the identity: `colmat` and
`t4a-faer` are two different crates providing the same library, so a graph that still has
`t4a-faer` 0.33 alongside has two of every type they exchange.

Only a caller that names those types has to move. Anyone handing this crate arrays, `Vec`s,
`Op`/`Error`/`Exec` and the workspace queries (`svd_workspace`, `cholesky_workspace` and the rest
take shape arguments and an `Exec`, no view) is unaffected and gains no dependency. A generic
caller bounded by `tlinalg::FaerScalar` reaches faer's own items through its supertraits
(`ScalarEntity`, `EigScalar`, as in 0.11.0), and every entry point takes a
`strided_view::RawStridedRef`, which moved 0.7 → 0.8. Such a caller moves to `colmat` 0.34,
`strided-view` 0.8 and `tcontract-exec` 0.13 together with this release; moving tlinalg alone does
not compile.

`Exec` still comes from `tcontract_exec`, now 0.13.0, whose `threadteam` token is 0.6.0 and which
is itself built on `colmat` 0.34.0. A host builds its pool, token and `Exec` from threadteam 0.6.0
and tcontract-exec 0.13.0; with one version of each crate in the graph the token passes through to
tlinalg and to faer's `Par::Team` as before.

The C API's pool entry points follow tcontract-rs#149 / #185 and are now `tcontract_pool_create`
and `tcontract_tapp_executor_create_pool`; they were `*_create_rayon`, their options type is now
`tcontract_pool_opts` (was `tcontract_rayon_opts`), and tcontract-capi's own ABI version moved to
101. The `tlinalg_*` prototypes are unchanged, so a C caller of these two functions renames the
call and nothing else. That is about names, not about mixing builds: a nonzero `exec` handle is a
Rust `TAPP_executor` of the tcontract-capi build this library was linked against, not an opaque
handle any build can read, so the host uses one tcontract-capi version on both sides.

`strided-view` moves 0.7.1 → 0.8.1, the view crate of the strided 0.15 line.

### Behavior: the SVD accuracy check's size gate moved from 128 to 40

`svd_into` checks a decomposition with vectors against its input and repeats it with the QR
algorithm when the result is inaccurate. It only does so where faer's *default* parameters can
take the divide-and-conquer path, which it reads from `SvdParams::recursion_threshold` and applies
to `k = min(m, n)`. `colmat` 0.34 lowered that default from 128 to 40, so decompositions with a
smaller dimension of at least 40 run with the default parameters first instead of straight to QR.
The retry policy is unchanged: a first attempt that returns an error, or whose factors do not
reproduce the input to `24 √max(m, n) ε`, is repeated with QR, and that repeat is not checked
again (it is allowed to fail with `NonConvergence`). What the move costs is the reconstruction
product for an item in the 40 to 127 band whose first attempt does not error, and the check
buffers in `svd_workspace` from 40 on, so a caller that kept a byte count from 0.11.0 has to query
again. The values-only route is unchanged: it still passes the QR parameters, and faer's own dqds
first pass for values is the same as in 0.33. `40 x 40` and `96 x 64` (`k = 40` and `k = 64`) are
examples of the newly covered band in `accurate_decompositions_are_not_repeated`, and
`only_decompositions_that_can_divide_are_checked` pins the gate at 40.

### Dependencies

| Crate | 0.11.0 | 0.12.0 |
| --- | --- | --- |
| `colmat` (`faer`; `t4a-faer` up to 0.33) | 0.33.0 | 0.34.0 |
| `tcontract-exec` | 0.12.0 | 0.13.0 |
| `threadteam` | 0.5.1 | 0.6.0 |
| `strided-view` | 0.7.1 | 0.8.1 |

The workspace's own development pins moved with them: the `tcontract` / `tcontract-exec`
patch to 0.13.0, and the private `tcontract-capi` git revision to the tcontract-rs commit
`ca44b476`, which is the commit tagged `v0.13.0` and the one the published tcontract and
tcontract-exec 0.13.0 name in their `.cargo_vcs_info.json`. `tlinalg-blas` moves 0.5.1 → 0.6.0:
its entry points take `strided_view::RawStridedRef` and `RawStridedMut`, so the strided-view 0.8
requirement is a public-dependency change, as strided 0.7 was for 0.5.0. Nothing in its own
signatures changed.

## tlinalg 0.11.0

This release is **breaking** only through a dependency identity. No `.rs` file changed and no
tlinalg signature changed.

### Breaking: `Exec` comes from tcontract-exec 0.12.0, built on t4a-faer 0.33

Every entry point takes `&mut tcontract_exec::Exec<'_>`. That type now comes from
tcontract-exec 0.12.0, whose source is unchanged from 0.11.0 but which is built against
t4a-faer 0.33.0, where the 0.11.0 one is built against 0.32.0. The `tcontract_exec::Exec` API
itself did not change, but the type is distinct from the 0.11.0 one, so a host has to build its
pool, token and `Exec` from threadteam 0.5.1 and tcontract-exec 0.12.0. With one version of each
crate in the dependency graph the token passes through to tlinalg and to faer's `Par::Team` as
before. The two `t4a-faer` copies are not interchangeable:
`tcontract_exec::with_faer_par` hands its closure a `Par` from the copy its own crate was built
against, so tlinalg on 0.33 against tcontract-exec 0.11 does not compile.

### t4a-faer 0.33's LU rank truncation is not exposed here

t4a-faer 0.33.0 adds `max_rank`, `abs_tol` and `rel_tol` to both LU parameter types and `rank`
to their info types (tensor4all/faer#294). Tlinalg constructs none of those types, so no tlinalg
entry point gains an option or an output, and at their defaults the factorizations keep the
stopping rule they had in 0.32: full-pivot LU already stopped early on a zero or subnormal Schur
complement, so neither runs to `min(m, n)` in every case today. Exposing rank truncation is
separate work.

### Breaking for generic callers: the inherited faer-facing surface moves to 0.33

Tlinalg does not re-export `faer`, but it does publish `FaerScalar` (`src/lib.rs`), and that
trait's supertraits are faer-facing. `ScalarEntity` ties `Entity` and `Real` to
`faer::traits::{ComplexField, RealField}`; `EigScalar` has `eig_req`, `eig_scratch` and
`eig_into`, which take `faer::ParShape`, `faer::dyn_stack::{StackReq, MemStack}`,
`faer::MatRef` and `faer::Par`. The supertraits are defined in a private module, so their paths
cannot be named, but a generic caller bounded by `tlinalg::FaerScalar` still reaches their items.
Such a caller has to move its own `t4a-faer` to 0.33 too: 0.32 and 0.33 are distinct types even
where the definition is unchanged.

### Dependencies

| Crate | 0.10.0 | 0.11.0 |
| --- | --- | --- |
| `t4a-faer` (`faer`) | 0.32.0 | 0.33.0 |
| `tcontract-exec` | 0.11.0 | 0.12.0 |
| `threadteam` | 0.5.1 | 0.5.1 |
| `strided-view` | 0.7.1 | 0.7.1 |

The workspace's own development pins moved with them: the `tcontract` / `tcontract-exec`
patch to 0.12.0, and the private `tcontract-capi` git revision to the tcontract-rs commit
`abf8c723` that 0.12.0 was published from (#184). `tlinalg-blas` stays at 0.5.1: none of its
requirements changed.

## tlinalg 0.10.0

This release is **breaking**. It has two independent breaking changes.

### Breaking: the SVD and Hermitian-eigen spectra are written as the real type (#101)

The singular values of `svd_into`/`svd_into_init` and the eigenvalues of
`eigh_into`/`eigh_into_init` were written through `T`. A complex input got its spectrum back as
complex numbers with a zero imaginary part. These spectra are real by definition, so the output
element is now `<T as ScalarEntity>::Real`, the same type `svd_values_into` already used.

| Function | 0.9.0 spectrum argument | 0.10.0 spectrum argument |
| --- | --- | --- |
| `svd_into` | `s: &mut [MaybeUninit<T>]` | `s: &mut [MaybeUninit<<T as ScalarEntity>::Real>]` |
| `svd_into_init` | `s: &mut [T]` | `s: &mut [<T as ScalarEntity>::Real]` |
| `eigh_into` | `values: &mut [MaybeUninit<T>]` | `values: &mut [MaybeUninit<<T as ScalarEntity>::Real>]` |
| `eigh_into_init` | `values: &mut [T]` | `values: &mut [<T as ScalarEntity>::Real]` |

The other arguments are unchanged. For real `T` (`f32`, `f64`) the real type is `T` itself, so
real callers compile unchanged. Complex callers pass a real slice of the same length.

### Breaking: `Exec` comes from tcontract-exec 0.11 and carries a threadteam 0.5 token

Every entry point takes `&mut tcontract_exec::Exec<'_>`. That type now comes from
tcontract-exec 0.11.0, and the `threadteam::Exec` token inside it comes from threadteam 0.5.
The `tcontract_exec::Exec` API itself did not change between 0.10 and 0.11, but the type is
distinct from the 0.10 / threadteam 0.4 one. A host has
to build its pool, token and `Exec` from threadteam 0.5.1 and tcontract-exec 0.11.0. With only
one version of each crate in the dependency graph, the token passes through to tlinalg and to
faer's `Par::Team` as before.

### Dependencies

| Crate | 0.9.0 | 0.10.0 |
| --- | --- | --- |
| `threadteam` | 0.4.0 | 0.5.1 |
| `tcontract-exec` | 0.10.0 | 0.11.0 |
| `t4a-faer` (`faer`) | 0.30.0 | 0.32.0 |
| `strided-view` | 0.7.0 | 0.7.1 |

The workspace's own development pins moved with them: the `tcontract` / `tcontract-exec`
patch to 0.11.0, and the private `tcontract-capi` git revision to tcontract-rs tag `v0.11.0`
(`c1ad85f6`). `tlinalg-blas` stays at 0.5.1: none of its requirements changed.
