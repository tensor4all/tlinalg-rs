# The payback share: when a team inside one item pays

A batched entry point resolves one route over the caller's token: how many lanes the batch fans out
into, and how wide each lane runs its items (`library-owned-lanes.md`). The second question —
whether one *item* should run on a team of `w` threads — is answered today by the shared width rule
of `threadteam`: the item runs in parallel when its serial estimate reaches `PAYS` (3) times the
dispatch cost of that team, and the dispatch cost is the measured row of the host class.

That rule prices work as work. It is right for a GEMM-shaped family, whose parallel part is
essentially all of it. It is wrong for a factorization whose wall clock is a serial phase: the rule
sees a large estimate, grants a team, and the team then spends its dispatch cost plus barriers on
the part of the work that does not split. The measured curves (`tensor4all/tlinalg-rs#88`, one
exclusive 96-core `x86_64-avx512` node, `tlinalg` 0.8.0, batch of one, team tokens) say exactly
that: at width 8, `svd` `s(8) = t(1)/t(8)` is 0.60 to 0.85 for every shape up to `256 x 256`, `qr`
is 0.58 to 0.74 up to `128 x 128`, `eigh` is 0.73 to 0.92 up to `256 x 256`; the same families gain
from a team only at the largest shapes (`svd` 1.36 and `eigh` 1.17 at `512 x 512`, `qr` 2.16).

## The model

`batch::Work` carries two numbers per item:

* `total` — the whole estimate in multiply-add equivalents, scaled by the scalar's weight. Lanes
  split it, so the fan-out decision and the shared `BatchQuery` keep using it unchanged.
* `payback_share` — the share of `total` a team inside one item can pay back. The item width is
  `min(decision.item_threads, width_for(total * payback_share))`, so an item never receives a team
  the convertible part of its work cannot repay. It is a cap: it narrows a route, never widens one,
  and at a share of one it is inert.

`payback_share` is **not** a parallel fraction, and it is not fitted to predict a speedup. It is the
share of the estimate that must repay the dispatch cost of a team before entering that team is worth
it, chosen so the grant boundary lands where the measured speedup crosses one. Because the
convertible part is compared against the token's own dispatch cost, the model carries over to
another host class as "the item has to be this much bigger before a team pays there", which is the
behaviour the measured dispatch rows describe.

| family | `payback_share` | measured boundary (f64, `x86_64-avx512`) |
|---|---|---|
| `qr`, `rank_revealing_qr` | 1e-2 | refuses to 128, grants from 192 |
| `eigh` | 6e-4 | refuses to 256, grants from 512 |
| `svd`, `svd_values` | 6e-4 | refuses to 256, grants from 512 |
| every other family | 1 | unchanged |

## What the cap does not do

* It does not touch the fan-out of a batch: a batch of 16 small `svd`s still splits into the lanes
  the decision gives (measured 6.7x to 7.2x on 8 slots), because the lanes split the whole estimate.
  What it changes is the width a lane's item may use when the lanes do not fill the budget.
* It does not touch a forced strategy. `Inner` and `Split` are the caller's own instruction; the
  decision is clamped to the items and the budget and is otherwise honoured.
* It does not change any result: only the width faer is handed, never the arithmetic.

## Evidence

`crates/tlinalg/src/batch/tests/payback.rs` is the acceptance record: every cell of the measured
curves above with its `s(8)`, and the assertion that the model refuses a team exactly where the team
lost time and grants it where the team gained, on the measured `x86_64-avx512` team dispatch row.
Cells within 10% of break-even are excluded — they say nothing about the boundary.

## Open

The public `*_estimate(..) -> Duration` API asked for in the same issue is not part of this change.
It needs one shared rate model across `tlinalg` and `tcontract` (`tcontract-rs#163`), which is a
cross-crate decision; the crate-internal model above is where that API would read the per-operation
numbers from.
