# AGENTS.md

Guidance for agents working in this repository.

Before acting, read the latest shared tensor4all agent rules:
`https://github.com/tensor4all/tensor4all-agent-rules/blob/main/rules/index.md`.
If the network is unavailable, use the sibling checkout `../tensor4all-agent-rules/rules/index.md`.
Load only the rule files the task needs. If neither source is available, continue and state that
the shared rules were unavailable.

## Repository-specific rules

* `tlinalg` is a provider: tensor-free kernels plus the small vocabulary its own entry points take
  (`tcontract_exec::Exec`, `Scalar`, `Error`, `Op`; the batch route comes from the shared
  `tcontract_exec::BatchPolicy`). It contains no tensor types.
* Every public entry point is batched (`docs/design/batched-api.md`); per-matrix kernels stay
  crate-private, and `crates/tlinalg/src/batch.rs` is the only batch loop.
* The interface a host requires of its linear-algebra providers belongs to the host (tenferro
  defines it and adapts each provider to it). Do not reintroduce a shared trait crate here, and do
  not shape these types for another provider: the sibling crate `tlinalg-blas` never depends on
  `tlinalg`, nor `tlinalg` on it. Shared test helpers go in the dev-only, unpublished
  `tlinalg-testkit` (only ever a `[dev-dependencies]` entry of a provider), never in either
  provider; cross-provider checks go in `tlinalg-parity`, kernel benchmarks in `tlinalg-bench`. A
  parity failure is a bug in a provider or in a documented convention, not a tolerance to loosen.
* Only `threadteam` starts tasks: parallelism reaches this crate as the `threadteam::Exec` token inside
  `tcontract_exec::Exec` (`fork` for lanes; `external` for a width handed to faer, or a team `Par::Team` when the token carries a team). There is no direct
  `rayon` dependency (`cargo deny check bans`), and this crate neither depends on spindle nor opens a
  spindle region (`with_lock`); that belongs to t4a-faer (tensor4all/faer#59).
* The public types are published behaviour: changing `Error`, the `tcontract_exec::Exec` token, or
  the batched contract in `docs/design/batched-api.md` is a contract change and needs the design
  reviewed first.
* `tlinalg-blas` owns its vocabulary (`Error`, `Op`, `Workspace`, `IndexWorkspace`, `Scalar`).
  Its entry points take no parallelism token (LAPACK and BLAS own their threading), loop serially
  over the batch, and query and acquire workspace once per call.
* Keep SIMD and reusable strided kernels in `strided-rs`, not here.
* Do not add a provider registry, autotuner, or automatic provider switching. Implementations are
  selected by the host.
* Do not widen the interface to make a host-side detail easier. The host adapts.
