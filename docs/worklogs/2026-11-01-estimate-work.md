# Public work estimates

Date: 2026-11-01

## Decision

Expose shape taking free functions in each numerical module, rather than introduce plans or
new descriptors into an API that currently executes through free functions. Each query describes
one batch item, independent of the batch count, thread width, machine and execution strategy.
Distinct models within a module have explicit solve or reflector application names.

The public values come directly from the helpers the planner already consumes. The packed LU
fused and prepared solve expressions were moved unchanged into private helpers so queries and
execution use the same source. No arithmetic, payback shares, strategy selection or numerical
kernel changed. Every query takes constant work and allocates no storage.

The unit follows tensor4all-meta `0a5f77e`, section "Work units and rates": one real scalar
multiply add, with a weight of four for complex scalars. Unlike an exact integer model, these
existing formulas already evaluate in `f64`. Keep that behavior and document that not every
integer above `2^53` is exactly representable. A caller measures the rate on its machine and
divides; rate ownership belongs in the caller or a crate above the providers. No duration API
or shared rate model was added.

## Verification

The regression observes every actual `Work` input to the planner during a query window, rather
than reconstructing formulas or checking only the last observation. It compares totals and
payback work bit for bit against the public queries at widths one and eight under forced inner,
sequential and automatic strategies. Forced inner explicitly grants the two different widths.
Workspace queries cover each model and the values only and rank revealing variants; routes
without scratch are checked through execution instead. This also fixes the meaning as work per
item rather than work summed over the two item workspace query batches.

Public tests cover the complex weight, each payback share including reflector application with
share one, empty and rectangular shapes, prepared solve accounting and both triangular solve
sides. The existing counting allocator covers every estimate for all four scalar types. Every
new public function has an executable documentation example.

Native tests and documentation examples passed, as did workspace lint, formatting and API
documentation checks with warnings denied. The existing ignored GEMM allocation test remains
ignored because of its documented dependency limitation; the new estimate allocation test passed.

## Scope

Only the native tlinalg API, its tests and its documentation change. No other crate, dependency,
C binding or Duration API changes. No measurements were performed and no performance comparison
is claimed. No code or tests were imported from another project.
