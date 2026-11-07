# Changelog

Changes to the published `tlinalg` crate. `tlinalg-blas` is versioned separately; a release here
names it only when its version moves.

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
