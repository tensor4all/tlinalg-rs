/* End-to-end test of libtlinalg_capi from C: the two-phase SVD, the conventions,
   and the error statuses. Exit status 0 on success. */
#include <math.h>
#include <stdio.h>

#include "tlinalg/tlinalg.h"

#define CHECK(cond, msg)                                                              \
    do {                                                                              \
        if (!(cond)) {                                                                \
            fprintf(stderr, "FAIL %s:%d: %s (%s)\n", __FILE__, __LINE__, msg,         \
                    tlinalg_last_error());                                            \
            return 1;                                                                 \
        }                                                                             \
    } while (0)

int main(void) {
    CHECK(tlinalg_abi_version() == TLINALG_ABI_VERSION, "abi version");

    int64_t extents[2] = {2, 2};
    int64_t strides[2] = {1, 2};
    tlinalg_strided in = {2, extents, strides, 0};
    double a[4] = {3.0, 0.0, 0.0, 1.0};

    /* Two-phase SVD: query, then write into caller-owned buffers. */
    tlinalg_svd_out sizes;
    double u[4], s[2], vt[4];
    CHECK(tlinalg_svd_sizes(&in, 0, &sizes) == TLINALG_OK, "svd sizes");
    CHECK(sizes.batch == 1 && sizes.u == 4 && sizes.s == 2 && sizes.vt == 4,
          "svd size values");
    CHECK(tlinalg_svd_into(&in, TLINALG_F64, a, 4, 0, u, 4, s, 2, vt, 4, NULL, 0) ==
              TLINALG_OK,
          "svd into");
    CHECK(fabs(s[0] - 3.0) < 1e-12 && fabs(s[1] - 1.0) < 1e-12, "svd values");

    /* A buffer one element short is rejected before any write. */
    double guard[4] = {7.0, 7.0, 7.0, 7.0};
    CHECK(tlinalg_svd_into(&in, TLINALG_F64, a, 4, 0, guard, 3, s, 2, vt, 4, NULL, 0) ==
              TLINALG_ERR_SHAPE,
          "too-short u rejected");
    CHECK(guard[0] == 7.0 && guard[1] == 7.0 && guard[2] == 7.0 && guard[3] == 7.0,
          "too-short u wrote");

    /* A singular (not positive-definite) Cholesky is rejected. */
    double singular[4] = {1.0, 1.0, 1.0, 1.0};
    double l[4];
    CHECK(tlinalg_cholesky_into(&in, TLINALG_F64, singular, 4, l, 4, NULL, 0) ==
              TLINALG_ERR_NONCONVERGENCE,
          "singular cholesky");

    /* Reserved and unknown dtype tags. */
    CHECK(tlinalg_svd_into(&in, TLINALG_F16, a, 4, 0, u, 4, s, 2, vt, 4, NULL, 0) ==
              TLINALG_ERR_UNSUPPORTED,
          "f16 unsupported");
    CHECK(tlinalg_svd_into(&in, 99, a, 4, 0, u, 4, s, 2, vt, 4, NULL, 0) ==
              TLINALG_ERR_INVALID_ARGUMENT,
          "unknown dtype");

    /* Null descriptor. */
    CHECK(tlinalg_svd_sizes(NULL, 0, &sizes) == TLINALG_ERR_INVALID_ARGUMENT, "null desc");

    char msg[32];
    size_t n = tlinalg_explain_error(TLINALG_OK, sizeof msg, msg);
    CHECK(n == 7 && msg[0] == 's', "explain");

    printf("abi_test ok\n");
    return 0;
}
