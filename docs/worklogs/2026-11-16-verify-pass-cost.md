# What a verify pass costs, and why it is now the products

Date: 2026-11-16

## Decision

`tensor4all/tlinalg-benchmark#37`: one `tlbench verify` pass over the batched grid at n=64 ran past
45 minutes, which is why that suite's `verify:` block is narrowed to n=4/8/16. The cost was not the
providers — it was three things in `crates/tlinalg-bench/src/harness/verify.rs`:

1. `eigvals` called `check_eigen` **inside** its per-batch loop while `check_eigen` loops over every
   batch itself: `O(batch²·n³)`. Measured, `--threads 8 --batch 1024 --dtype f64`, one family:
   n=16 52.09 s, n=32 342.80 s, n=64 1328.42 s (first session; see the host-state caveat under
   "What it bought" — the same figures on the second session were 2× lower at n=32 and within 6 %
   at n=64). Every other family took 0.1–1.5 s at n=16 and 0.2–5.2 s at n=64. That single line was
   95 % of an n=64 pass.
2. The reference matrix was generated four times per case: each arm's `output_call_checked` built
   its own `a`/`b`/`op`, and `check_one` rebuilt the same matrix twice more. For
   `cholesky`/`eigh`/`eigvalsh` each generation is `batch` × `hpd_seeded`, a batched `n³` product.
3. Every residual product was a naive, allocating, `num_complex` triple loop (`matmul`, `orth`,
   the cholesky and SVD reconstructions, the packed-LU loop), one `n×n` buffer per matrix.

The pass now generates the reference once per case and hands the same batches to both arms and to
the checks (they cannot drift apart, and they cannot disagree about which matrix was verified), and
the products are one allocation-free complex GEMM (`gemm`) plus a conjugate dot product for `QᴴQ`.
The product is the harness's own, not faer and not the vendor BLAS, so neither arm is checked with
the arithmetic it is being checked against; `tlbench verify` names that backend on stderr once.

The loop order and the real/imag split were chosen to round exactly as the loops they replaced did,
and they do: every `OK` line at n=16 and n=32 (f64) and at the c64 and rectangular spot checks is
byte-identical to the pre-change binary, not merely within tolerance.

## What it bought

One pass over the declared grid, per-family sum of `verify` wall time (`--threads 8 --batch 1024`,
all 20 families, host `EVO-X2`, OpenBLAS 0.3.32 Cooperlake). Both binaries were measured back to
back in one session: **before** = `main` at `3ed7017` (whose `tlinalg-bench` is identical to the
branch point), **after** = this change.

| grid | before | after | factor |
| --- | --- | --- | --- |
| n=32 f64 | 176.59 s | 5.74 s | 31× |
| n=32 c64 | 178.82 s | 7.54 s | 24× |
| n=64 f64 | 1299.43 s (21.7 min) | 37.67 s | 35× |
| n=64 c64 | 1377.51 s (23.0 min) | 48.26 s | 29× |

A whole n=64 pass over both dtypes is 2677 s (44.6 min) before and 86 s after — the recorded
">45 minutes" is the 44.6 minutes, and `eigvals` is ~2570 s of it. After the change `eigvals` at
n=64 is 1.7 s (f64) and 2.1 s (c64).

Absolute verify times move with host state, and the first version of this table was measured across
sessions: the same unmodified `tlinalg-bench` read 2× slower at n=32 in the earlier session
(354.29 s / 342.80 s for `eigvals`) while agreeing at n=64 within 6 %. The factors above — not the
absolute seconds — are the transferable number, and the dominant family was re-run interleaved
(`eigvals`, n=32 f64: before 171.9 s → after 0.38 s → before 171.3 s) so the factor is not an
ordering artefact.

The remaining cost is the providers' own work plus the one residual product each check needs; there
is no longer a superlinear term in the pass. Sampling the residuals would therefore buy little and
would weaken the gate, so the full residual stays on every case.

## Not changed

Tolerances, `gate`, the printed lines, the CSV census and the timed rows. The benchmark suite's
`verify:` grid and result pages are untouched: widening the grid is a campaign decision with its own
re-record, and this change only removes the reason it was narrowed.

## Review

Independent plan review by `reviewer-glm` on `ollama-cloud/glm-5.3-flash` (thinking `high`):
All eight findings were accepted and folded into the plan before any code was written — the SVD
row of the plan would have conjugated `vt` (the code consumes it unconjugated), the equivalence
commands were square-only, and the hpd families' input batch was named ambiguously.

The finished diff then got the repository's independent post-review (`openai/gpt-6.1-sol`):
`CHANGES REQUESTED`, all findings fixed — one private-intra-doc-link (rustdoc `-D warnings`), four
needless borrows (clippy `-D warnings`), and a demand to harden the before/after equivalence
method, which now requires each binary to exit 0 with exactly one result line before its output is
compared (80 case comparisons, all byte-identical).
