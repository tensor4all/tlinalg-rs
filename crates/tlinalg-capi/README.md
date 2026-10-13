# tlinalg-capi (libtlinalg_capi)

The **caller-owned-output C ABI** over [`tlinalg`](../tlinalg), the faer-backed tensor-free
linear-algebra kernels. Built as one shared library, one static library and one `rlib`, so the same
crate serves a Rust dependent, a `dlopen`-ing host and a statically linked C program.

The provider allocates its outputs through Rust `Vec`s, which cannot cross an FFI boundary: the C
side must not `free()` Rust-allocated memory, and a host that already owns buffers wants the kernels
to write into them. The ABI is therefore two-phase:

1. `tlinalg_<op>_sizes` reports, per batch item, the element count of every output and the
   workspace bytes (always 0 in v1).
2. `tlinalg_<op>_into` writes compact column-major results into caller-owned buffers, each passed
   with its element capacity. A buffer shorter than the query reports is rejected before anything
   is written.

Inputs keep the provider's convention: a borrowed, dtype-erased `tlinalg_strided` descriptor of
rank `2 + B` with extents and strides in elements, `[rows, cols, b_1, ..., b_B]`, the first batch
axis fastest, plus an offset. The element pointer, its element count and the dtype tag are separate
parameters. v1 is sequential; borrowing a host executor is out of scope (tensor4all/tcontract-rs#112)
and this crate does not depend on tcontract-rs.

## Coverage (v1)

| Operation | Entry pair | Notes |
|---|---|---|
| SVD with vectors | `tlinalg_svd_sizes/into` | `full` flag; returns `U`, `S`, `Vᴴ` |
| Singular values | `tlinalg_svd_values_sizes/into` | real values |
| Hermitian eigendecomposition | `tlinalg_eigh_sizes/into` | lower triangle; non-decreasing real values |
| Hermitian eigenvalues | `tlinalg_eigh_values_sizes/into` | real values |
| Cholesky | `tlinalg_cholesky_sizes/into` | lower triangle; `A = L Lᴴ` |
| Partial-pivot LU | `tlinalg_lu_sizes/into` | `P A = L U` + parity |
| Full-pivot LU | `tlinalg_full_piv_lu_sizes/into` | `P A Qᵀ = L U` + parity |

Dtypes `TLINALG_F32` / `F64` / `C32` / `C64`; `F16`/`BF16` are reserved and rejected with
`TLINALG_ERR_UNSUPPORTED`.

## From C

```bash
cargo build --release -p tlinalg-capi
./install.sh --prefix=/opt/tlinalg        # header, libraries, pkg-config

cc myprog.c $(PKG_CONFIG_PATH=/opt/tlinalg/lib/pkgconfig \
              pkg-config --cflags --libs tlinalg_capi) -lm
```

## Verification

The numeric behaviour is checked against `tlinalg` itself per operation, including `full` versus
economy SVD shapes and a batch larger than one; the error paths (too-short buffer, singular
Cholesky, shape mismatch, null pointer) are checked too. A re-declaration of the entire header in an
`extern "C"` block makes a renamed symbol or a changed prototype a link failure. A small C program
is compiled against the shipped header and linked to the built library when a C compiler is
available.

## Status

Prerelease and unpublished. The API may change.

## License

MIT OR Apache-2.0.
