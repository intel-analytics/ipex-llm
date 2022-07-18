/* SPDX-License-Identifier: LGPL-3.0-or-later */
/* Copyright (C) 2014 Stony Brook University */

/*
 * This file contains APIs to create, exit and yield a thread.
 */

#include "pal.h"
#include "pal_error.h"
#include "pal_internal.h"

/* PAL call DkThreadCreate: create a thread inside the current process */
int DkThreadCreate(int (*callback)(void*), void* param, PAL_HANDLE* handle) {
    *handle = NULL;
    return _DkThreadCreate(handle, callback, param);
}

/* PAL call DkThreadYieldExecution. Yield the execution of the current thread. */
void DkThreadYieldExecution(void) {
    _DkThreadYieldExecution();
}

/* PAL call DkThreadExit: simply exit the current thread no matter what */
noreturn void DkThreadExit(int* clear_child_tid) {
    _DkThreadExit(clear_child_tid);
    /* UNREACHABLE */
}

/* PAL call DkThreadResume: resume the execution of a thread which is delayed before */
int DkThreadResume(PAL_HANDLE thread_handle) {
    if (!thread_handle || HANDLE_HDR(thread_handle)->type != PAL_TYPE_THREAD) {
        return -PAL_ERROR_INVAL;
    }

    return _DkThreadResume(thread_handle);
}

int DkThreadSetCpuAffinity(PAL_HANDLE thread, PAL_NUM cpumask_size, unsigned long* cpu_mask) {
    return _DkThreadSetCpuAffinity(thread, cpumask_size, cpu_mask);
}

int DkThreadGetCpuAffinity(PAL_HANDLE thread, PAL_NUM cpumask_size, unsigned long* cpu_mask) {
    return _DkThreadGetCpuAffinity(thread, cpumask_size, cpu_mask);
}
