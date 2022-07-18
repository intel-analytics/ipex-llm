/* SPDX-License-Identifier: LGPL-3.0-or-later */
/* Copyright (C) 2021 Intel Corporation
 *                    Paweł Marczewski <pawel@invisiblethingslab.com>
 */

/* Wrapper for invoking testing code from inside Gramine (`shim_run_test`). */

#include <stdio.h>

#include "gramine_entry_api.h"

int main(int argc, char** argv) {
    if (argc != 2) {
        fprintf(stderr, "usage: %s TEST_NAME\n", argv[0]);
        return 1;
    }

    const char* test_name = argv[1];
    int ret = gramine_run_test(test_name);
    printf("gramine_run_test(\"%s\") = %d\n", test_name, ret);
    return ret == 0 ? 0 : 1;
}
