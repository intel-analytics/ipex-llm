/* SPDX-License-Identifier: LGPL-3.0-or-later */
/* Copyright (C) 2014 Stony Brook University */

/*
 * This source file contains functions to create a child process and terminate the running process.
 * Child does not inherit any objects or memory from its parent process. A parent process may not
 * modify the execution of its children. It can wait for a child to exit using its handle. Also,
 * parent and child may communicate through I/O streams provided by the parent to the child at
 * creation.
 */

#include "api.h"
#include "pal.h"
#include "pal_error.h"
#include "pal_internal.h"

int _DkProcessCreate(PAL_HANDLE* handle, const char** args) {
    return -PAL_ERROR_NOTIMPLEMENTED;
}

noreturn void _DkProcessExit(int exitcode) {
    die_or_inf_loop();
}

static int64_t proc_read(PAL_HANDLE handle, uint64_t offset, uint64_t count, void* buffer) {
    return -PAL_ERROR_NOTIMPLEMENTED;
}

static int64_t proc_write(PAL_HANDLE handle, uint64_t offset, uint64_t count, const void* buffer) {
    return -PAL_ERROR_NOTIMPLEMENTED;
}

static int proc_close(PAL_HANDLE handle) {
    return -PAL_ERROR_NOTIMPLEMENTED;
}

struct handle_ops g_proc_ops = {
    .read  = &proc_read,
    .write = &proc_write,
    .close = &proc_close,
};
