# What a verify pass costs, and why it is now the products

Date: 2026-11-16

## Decision

`tensor4all/tlinalg-benchmark#37`: one `tlbench verify` pass over the batched grid at n=64 ran past
45 minutes, which is why that suite's `verify:` block is narrowed to n=4/8/16. The cost was not the
providers — it was three things in `crates/tlinalg-bench/src/harness/verify.rs`:

1. `eigvals` called `check_eigen` **inside** its per-batch loop while `check_eigen` loops over every
   batch itself: `O(batch²·n³)`. Measured, `--threads 8 --batch 1024 --dtype f64`, one family:
   n=16 52.09 s, n=32 342.80 s, n=64 1328.42 s. Every other family took 0.1–1.5 s at n=16 and
   0.2–5.2 s at n=64. That single line was 95 % of an n=64 pass (2687 s of 2825 s over both
   dtypes).
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
all 20 families, host `EVO-X2`, OpenBLAS 0.3.32 Cooperlake):

| grid | before | after | factor |
| --- | --- | --- | --- |
| n=32 f64 | 354.29 s | 5.63 s | 63× |
| n=32 c64 | 310.36 s | 7.57 s | 41× |
| n=64 f64 | 1380.32 s (23.0 min) | 37.72 s | 37× |
| n=64 c64 | 1444.81 s (24.1 min) | 48.90 s | 30× |

A whole n=64 pass over both dtypes is 2825 s (47.1 min) before and 86.6 s after — the recorded
">45 minutes" is the 47.1 minutes, and `eigvals` is 2686.84 s of it. After the change `eigvals` at
n=64 is 1.69 s (f64) and 2.12 s (c64).

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
