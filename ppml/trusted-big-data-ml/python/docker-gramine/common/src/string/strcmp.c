/* SPDX-License-Identifier: LGPL-3.0-or-later */

#include "api.h"

int strcmp(const char* lhs, const char* rhs) {
    while (*lhs == *rhs && *lhs) {
        lhs++;
        rhs++;
    }
    return *(unsigned char*)lhs - *(unsigned char*)rhs;
}

int strncmp(const char* lhs, const char* rhs, size_t maxlen) {
    if (!maxlen)
        return 0;

    maxlen--;
    while (*lhs == *rhs && *lhs && maxlen) {
        lhs++;
        rhs++;
        maxlen--;
    }
    return *(unsigned char*)lhs - *(unsigned char*)rhs;
}
