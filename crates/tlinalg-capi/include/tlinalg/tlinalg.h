/* tlinalg: the C ABI of the tensor-free linear-algebra kernels (tlinalg-rs).
 *
 * The provider (`tlinalg`) allocates its outputs through Rust `Vec`s, which cannot
 * cross an FFI boundary. This ABI is therefore two-phase and caller-owned:
 *
 *   1. tlinalg_<op>_sizes(&input, ..., &sizes) reports, per batch item, the
 *      element count of every output, the batch count and the workspace bytes.
 *   2. tlinalg_<op>_into(&input, ..., void *out, size_t out_len, ...) writes
 *      compact column-major results into buffers the caller owns, passing each
 *      buffer's element capacity. A buffer shorter than the query reports is
 *      rejected before anything is written.
 *
 * Threads: v1 is sequential (`tlinalg::Parallel::Sequential`). Borrowing a host
 * executor is an explicit non-goal of this version and follows
 * tensor4all/tcontract-rs#112; there is no pool vocabulary in this header.
 *
 * Conventions the operations carry:
 *   - SVD returns U, non-increasing singular values and V^H (not V); the `full`
 *     flag selects the square unitary factors.
 *   - eigh and cholesky read the LOWER triangle; eigh returns non-decreasing
 *     REAL eigenvalues.
 *   - svd and eigh write their values in the scalar type (a complex scalar with
 *     a zero imaginary part for the complex dtypes); svd_values and eigh_values
 *     write the real scalar.
 *   - lu and full_piv_lu do not fail on an exactly singular input; cholesky
 *     reports a non-positive-definite item as TLINALG_ERR_NONCONVERGENCE.
 *   - every output is compact column-major per item, items contiguous in batch
 *     order (the first batch axis is fastest).
 */
#ifndef TLINALG_TLINALG_H
#define TLINALG_TLINALG_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ABI version: major * 10000 + minor * 100 + patch. Compare with
   tlinalg_abi_version() to detect a header and a library from different versions. */
#define TLINALG_ABI_VERSION 200 /* 0.2.0 */

/* Status codes. These mirror the provider's typed errors; zero is the only
   success. Text for any code is in tlinalg_explain_error / tlinalg_last_error. */
typedef int32_t tlinalg_status;
#define TLINALG_OK 0                     /* outputs written */
#define TLINALG_ERR_INVALID_ARGUMENT 1   /* null pointer or invalid argument */
#define TLINALG_ERR_SHAPE 2              /* inconsistent/overflowing shape, stride or buffer size */
#define TLINALG_ERR_SINGULAR 3           /* singular factor or input */
#define TLINALG_ERR_UNSUPPORTED 4        /* well-formed request this library declines */
#define TLINALG_ERR_INTERNAL 5           /* bug or allocation failure */
#define TLINALG_ERR_NONCONVERGENCE 6     /* the algorithm did not converge */
#define TLINALG_ERR_NONFINITE 7          /* non-finite input or computed value */

/* Scalar storage types. F16/BF16 are reserved: they are rejected with
   TLINALG_ERR_UNSUPPORTED. Complex scalars are interleaved and layout-compatible
   with C99 `float _Complex` / `double _Complex`. */
typedef int32_t tlinalg_dtype;
#define TLINALG_F32 0
#define TLINALG_F64 1
#define TLINALG_C32 2
#define TLINALG_C64 3
#define TLINALG_F16 4 /* reserved */
#define TLINALG_BF16 5 /* reserved */

/* A borrowed, dtype-erased input layout. `extents` and `strides` each hold
   `rank` int64_t entries: extents in elements `[rows, cols, b_1, ..., b_B]` with
   `rank = 2 + B >= 2`, strides in elements with the first batch axis fastest.
   `offset` is the element offset of the first matrix element from the buffer.
   The buffer pointer, its element count and its dtype are separate arguments of
   each entry point.
 *
 * The descriptor must not reach outside the caller's buffer; a stride or offset
 * that would (or any overflow) is rejected with TLINALG_ERR_SHAPE before the
 * provider is called. The arrays must stay alive for the duration of the call. */
typedef struct {
    int32_t rank;           /* 2 + B, at least 2 */
    const int64_t *extents; /* rank extents in elements */
    const int64_t *strides; /* rank strides in elements */
    int64_t offset;         /* element offset from `data` */
} tlinalg_strided;

/* Size queries. Every count is per batch item; total buffer elements are
   item * batch. `workspace` is 0 in v1. */
typedef struct {
    size_t batch;
    size_t u;  /* rows * u_cols (u_cols = full ? rows : min(rows, cols)) */
    size_t s;  /* min(rows, cols) */
    size_t vt; /* v_cols * cols (v_cols = full ? cols : min(rows, cols)) */
    size_t workspace;
} tlinalg_svd_out;

typedef struct {
    size_t batch;
    size_t s; /* min(rows, cols), in the real scalar */
    size_t workspace;
} tlinalg_svd_values_out;

typedef struct {
    size_t batch;
    size_t values;  /* n, in the scalar type */
    size_t vectors; /* n * n, column-major eigenvectors */
    size_t workspace;
} tlinalg_eigh_out;

typedef struct {
    size_t batch;
    size_t values; /* n, in the real scalar */
    size_t workspace;
} tlinalg_eigh_values_out;

typedef struct {
    size_t batch;
    size_t l; /* n * n */
    size_t workspace;
} tlinalg_cholesky_out;

typedef struct {
    size_t batch;
    size_t p;      /* rows * rows */
    size_t l;      /* rows * min(rows, cols) */
    size_t u;      /* min(rows, cols) * cols */
    size_t parity; /* always 1 */
    size_t workspace;
} tlinalg_lu_out;

typedef struct {
    size_t batch;
    size_t p;      /* n * n, the row permutation */
    size_t l;      /* n * n */
    size_t u;      /* n * n */
    size_t q;      /* n * n, the column permutation */
    size_t parity; /* always 1 */
    size_t workspace;
} tlinalg_full_piv_lu_out;

/* Library identity. `tlinalg_last_error` is never NULL and must not be freed;
   it is valid until the next tlinalg call on the same thread. */
uint32_t tlinalg_abi_version(void);
const char *tlinalg_last_error(void);

/* Copy a description of `status` into `message` (NUL-terminated, at most
   `maxlen - 1` characters) and return the untruncated length without the NUL,
   as snprintf does. `maxlen == 0` writes nothing. */
size_t tlinalg_explain_error(tlinalg_status status, size_t maxlen, char *message);

/* --- svd: U, S, V^H ------------------------------------------------------ */

/* Query the per-item element counts for an SVD. `full` is 0 (economy) or 1. */
tlinalg_status tlinalg_svd_sizes(const tlinalg_strided *input, int32_t full,
                                 tlinalg_svd_out *out);

/* Write U, S, V^H into caller-owned buffers. `s` is in the SCALAR type of
   `dtype` (complex values carry a zero imaginary part). `data_len` is the number
   of elements in `data`. */
tlinalg_status tlinalg_svd_into(const tlinalg_strided *input, tlinalg_dtype dtype,
                                const void *data, size_t data_len, int32_t full,
                                void *u, size_t u_len, void *s, size_t s_len,
                                void *vt, size_t vt_len,
                                void *workspace, size_t workspace_len);

/* --- svd_values: S only -------------------------------------------------- */

tlinalg_status tlinalg_svd_values_sizes(const tlinalg_strided *input,
                                        tlinalg_svd_values_out *out);

/* Write the singular values; `s` is in the REAL scalar of `dtype`. */
tlinalg_status tlinalg_svd_values_into(const tlinalg_strided *input, tlinalg_dtype dtype,
                                       const void *data, size_t data_len,
                                       void *s, size_t s_len,
                                       void *workspace, size_t workspace_len);

/* --- eigh: Hermitian, lower triangle, non-decreasing real values --------- */

tlinalg_status tlinalg_eigh_sizes(const tlinalg_strided *input, tlinalg_eigh_out *out);

/* Write the eigenvalues (scalar type) and column-major eigenvectors. */
tlinalg_status tlinalg_eigh_into(const tlinalg_strided *input, tlinalg_dtype dtype,
                                 const void *data, size_t data_len,
                                 void *values, size_t values_len,
                                 void *vectors, size_t vectors_len,
                                 void *workspace, size_t workspace_len);

tlinalg_status tlinalg_eigh_values_sizes(const tlinalg_strided *input,
                                         tlinalg_eigh_values_out *out);

/* Write the real eigenvalues only. */
tlinalg_status tlinalg_eigh_values_into(const tlinalg_strided *input, tlinalg_dtype dtype,
                                        const void *data, size_t data_len,
                                        void *values, size_t values_len,
                                        void *workspace, size_t workspace_len);

/* --- cholesky: lower triangle, A = L L^H --------------------------------- */

tlinalg_status tlinalg_cholesky_sizes(const tlinalg_strided *input,
                                      tlinalg_cholesky_out *out);

tlinalg_status tlinalg_cholesky_into(const tlinalg_strided *input, tlinalg_dtype dtype,
                                     const void *data, size_t data_len,
                                     void *l, size_t l_len,
                                     void *workspace, size_t workspace_len);

/* --- lu: partial pivot, P A = L U ---------------------------------------- */

tlinalg_status tlinalg_lu_sizes(const tlinalg_strided *input, tlinalg_lu_out *out);

tlinalg_status tlinalg_lu_into(const tlinalg_strided *input, tlinalg_dtype dtype,
                               const void *data, size_t data_len,
                               void *p, size_t p_len, void *l, size_t l_len,
                               void *u, size_t u_len, void *parity, size_t parity_len,
                               void *workspace, size_t workspace_len);

/* --- full_piv_lu: full pivot, P A Q^T = L U ------------------------------ */

tlinalg_status tlinalg_full_piv_lu_sizes(const tlinalg_strided *input,
                                         tlinalg_full_piv_lu_out *out);

tlinalg_status tlinalg_full_piv_lu_into(const tlinalg_strided *input, tlinalg_dtype dtype,
                                        const void *data, size_t data_len,
                                        void *p, size_t p_len, void *l, size_t l_len,
                                        void *u, size_t u_len, void *q, size_t q_len,
                                        void *parity, size_t parity_len,
                                        void *workspace, size_t workspace_len);

#ifdef __cplusplus
}
#endif
#endif /* TLINALG_TLINALG_H */
