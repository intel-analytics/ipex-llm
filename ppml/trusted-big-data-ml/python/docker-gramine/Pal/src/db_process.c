/* SPDX-License-Identifier: LGPL-3.0-or-later */
/* Copyright (C) 2014 Stony Brook University */

/*
 * This source file contains functions to create a child process and terminate the running process.
 * Child does not inherit any objects or memory from its parent process. A parent process may not
 * modify the execution of its children. It can wait for a child to exit using its handle. Also,
 * parent and child may communicate through I/O streams provided by the parent to the child at
 * creation.
 */

#include "pal.h"
#include "pal_internal.h"

int DkProcessCreate(const char** args, PAL_HANDLE* handle) {
    *handle = NULL;
    return _DkProcessCreate(handle, args);
}

noreturn void DkProcessExit(PAL_NUM exitcode) {
    _DkProcessExit(exitcode);
    die_or_inf_loop();
}
