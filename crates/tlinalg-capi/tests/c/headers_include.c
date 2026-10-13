/* A translation unit that includes the shipped header alone, compiled as C and as
   C++ with -Wall -Werror. */
#include "tlinalg/tlinalg.h"

int main(void) {
    return (int)(tlinalg_abi_version() == TLINALG_ABI_VERSION) - 1;
}
