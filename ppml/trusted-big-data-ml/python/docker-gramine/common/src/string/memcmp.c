/* SPDX-License-Identifier: LGPL-3.0-or-later */

#include "api.h"

#ifndef ASAN
__attribute__((alias("_real_memcmp")))
int memcmp(const void*, const void*, size_t);
#endif

__attribute_no_sanitize_address
int _real_memcmp(const void* lhs, const void* rhs, size_t count) {
    const unsigned char* l = lhs;
    const unsigned char* r = rhs;
    while (count && *l == *r) {
        count--;
        l++;
        r++;
    }
    return count ? *l - *r : 0;
}
