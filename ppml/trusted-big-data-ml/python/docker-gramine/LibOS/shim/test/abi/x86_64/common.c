/* SPDX-License-Identifier: LGPL-3.0-or-later */
/* Copyright (C) 2022 Intel Corporation
 *                    Mariusz Zaborski <oshogbo@invisiblethingslab.com>
 */

/*
 * We have to add a function declaration to avoid warnings.
 * This function is used with NASM, so creating a header is pointless.
 */
int test_str_neq(const char* orig, const char* new);

int test_str_neq(const char* orig, const char* new) {
    if (orig == new)
        return 0;

    while (*orig && *orig == *new) {
        orig++;
        new++;
    }

    return *orig != *new;
}
