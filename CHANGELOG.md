# Changelog

Changes to the published `tlinalg` crate. `tlinalg-blas` is versioned separately; a release here
names it only when its version moves.

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
