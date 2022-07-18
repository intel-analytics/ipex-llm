/* SPDX-License-Identifier: MIT */
/* Adapted from musl v1.2.1. Copyright © 2005-2020 Rich Felker, et al. */

#include "api.h"

int islower(int c) {
    return (unsigned)c - 'a' < 26;
}

int isupper(int c) {
    return (unsigned)c - 'A' < 26;
}

int tolower(int c) {
    if (isupper(c))
        return c | 32;
    return c;
}

int toupper(int c) {
    if (islower(c))
        return c & 0x5f;
    return c;
}

int isalpha(int c) {
    return ((unsigned)c | 32) - 'a' < 26;
}

int isdigit(int c) {
    return (unsigned)c - '0' < 10;
}

int isalnum(int c) {
    return isalpha(c) || isdigit(c);
}
