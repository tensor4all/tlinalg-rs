# The complex values-only `?gesdd` real workspace on Apple Accelerate

## Purpose

`crates/tlinalg-blas/src/svd.rs::complex_rwork_len` sized the real workspace of a complex
values-only `?gesdd` at `5 * min(m, n)`, the length LAPACK 3.7 and later document for `JOBZ = 'N'`.
The same expression came from tenferro-linalg's LAPACK provider, whose sizing the kernel moved here
with. Apple Accelerate's `?gesdd` is the 3.2.1 interface, and `SRC/zgesdd.f` says of exactly that
case that it "needs `7*mn`". Accelerate therefore writes past a `5*mn` buffer, and on macOS the
overrun lands in the caller's heap, where it surfaces as unrelated corruption rather than as an
error. That is the failure a downstream crate reported as tenferro-rs#2052 (heap corruption,
`malloc` aborts, and a hang, all without an SVD error).

## What changed

`complex_rwork_len` now asks for `7 * max(min(m, n), 1)` in that branch, and a unit test asserts the
contract. Both `?gesdd` branches that compute with the factors already follow the documented
formula, and the `provider-inject` variant sizes `?gesvd`, whose real workspace `zgesvd.f` fixes at
`5 * min(M, N)` for every `jobu`/`jobvt` combination. Those are unchanged.

## Verification

A standalone C program drives Accelerate's `zgesdd_("N", ...)` with the `rwork` buffer placed so
that its last element ends exactly at a page boundary followed by a `PROT_NONE` page, so any write
past the length faults instead of corrupting silently. On this machine (macOS, aarch64,
`Apple M3 Pro`):

| shape | `5*mn` | `6*mn` | `7*mn` |
| --- | --- | --- | --- |
| 3x4, 4x5, 5x6, 6x7, 7x8, 8x9, 9x10, 7x10 | fault | fault | ok |
| 9x4, 8x8, 10x20, 24x24 | ok | ok | ok |

Wide shapes from just wider than square up to about `1.6*mn` fault at `5*mn` and `6*mn`, which is
the band the issue reports and the reason square and tall shapes never showed it. `cargo test -p
tlinalg-blas --lib` fails on the values-only assertion with the previous `5` and passes with `7`,
under both the default and the `provider-inject` feature sets.

The crate's own LAPACK suite then repeats both sides of the change end to end against Accelerate,
with `(3, 4)` and `(7, 10)` added to `every_mode_and_shape` so that the complex values-only cases
land in the failing band. The vendor source features were repointed at Accelerate for the run
(`blas-src/accelerate`, `lapack-src/accelerate` in place of `openblas`), since the from-source
OpenBLAS in the pinned `openblas-src` does not build with the toolchain here: its vendored LAPACK
test program `zblat3.f` fails to compile. Under Guard Malloc
(`DYLD_INSERT_LIBRARIES=/usr/lib/libgmalloc.dylib`, `--test-threads=1`), the previous `5` dies with
`SIGSEGV` (exit 139) inside the SVD, and `7` passes all five tests (exit 0).

## Alternatives considered

* **`7*mn` only when the linked LAPACK is old.** The provider is the host's choice and the interface
  here carries no library identity, so this would mean widening the vocabulary or a
  version-sniffing shim, which the repository rules reject. The cost of not knowing is two extra
  reals per singular value.
* **`5*mn` with a note.** LAPACK 3.7 and later accept it, but the buffer the routine is handed has
  to cover the oldest interface the host may link, and Accelerate is reachable from tenferro's
  `blas-accelerate` feature.

## Residual risk

Accelerate beyond the shapes above is untested here. The sizing is now the documented maximum of
the two interfaces rather than a guess, so the remaining risk is a provider that needs more than
its documentation states, which no sizing rule can cover.
