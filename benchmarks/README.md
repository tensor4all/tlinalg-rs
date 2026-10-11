# benchmarks/

The measurement side of `tlinalg-rs`: the `tlbench` harness, the protocol scripts that make a
timing trustworthy, and the record of what the provider comparison is allowed to claim.

This directory is the *library-level* harness. The **campaign** — which commits were measured on
which machines, the run manifests, the published reports and their staleness — lives in
[tensor4all/tlinalg-benchmark](https://github.com/tensor4all/tlinalg-benchmark), which builds
`tlbench` out of a pinned checkout of this repository. The harness stays here because it must be
built out of the same checkout as the library it measures, so that a result's recorded commit
describes both. This split is the one `tcontract-rs`/`tcontract-benchmark` already use.

| | |
|---|---|
| `tlbench` | `crates/tlinalg-bench`: `info`, `verify`, `run` over the batched kernel families |
| Criterion rows | `crates/tlinalg-bench/benches/kernels.rs` — per-PR rows, not campaign evidence |
| `scripts/pinned.sh` | runs one measurement pinned to a CPU set, valid only if those cores were idle before and after |
| `scripts/idle_cpus.py` | picks or checks idle CPUs of one L3 domain from `/proc/stat` |

## Building and running

```bash
CARGO_BUILD_JOBS=12 cargo build --release -p tlinalg-bench --features link-openblas-static --bin tlbench
./target/release/tlbench info                       # what was linked, and to which kernel
cpus=$(python3 benchmarks/scripts/idle_cpus.py pick 8)
benchmarks/scripts/pinned.sh "$cpus" -- ./target/release/tlbench verify \
    --threads 8 --n 4,16,64 --batch 1,64 --dtype f64
benchmarks/scripts/pinned.sh "$cpus" -- ./target/release/tlbench run \
    --threads 8 --n 4,16,64 --batch 1,64 --dtype f64 --csv /tmp/small-8t.csv
```

`link-openblas-static` links the vendor statically, so the recorder can execute the binary
directly with no loader search path to arrange. `link-openblas` (shared) is what the crate's own
test configurations use. Tenferro selects the vendor and the injected-symbol path itself; these
features exist for measurement, not as part of the provider's contract.

## What a number from here may and may not claim

`tlbench` measures the batched provider entry points with compact inputs, reused output vectors and
a recycling LAPACK workspace, so steady-state allocation is not what is timed. It says nothing
about tensor construction, dtype dispatch, session entry or pool checkout — those belong to
tenferro, and route-level performance belongs in
[tenferro-benchmark](https://github.com/tensor4all/tenferro-benchmark).

A timing is only evidence with the four things the protocol scripts supply: the measured commit,
the core set, an idle window before and after, and a correctness pass over the same grid. A row
labelled `1T` also requires the vendor library to have been set to one thread; `tlbench` sets that
budget and reads it back, and `info` names the vendor and the kernel the runtime dispatched to.
