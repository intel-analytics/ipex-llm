/* SPDX-License-Identifier: LGPL-3.0-or-later */
/* Copyright (C) 2014 Stony Brook University */

/*
 * This file contains APIs to open, read, write and get attributes of streams.
 */

#include "api.h"
#include "pal.h"
#include "pal_error.h"
#include "pal_internal.h"

int _DkStreamUnmap(void* addr, uint64_t size) {
    return -PAL_ERROR_NOTIMPLEMENTED;
}

int _DkSendHandle(PAL_HANDLE target_process, PAL_HANDLE cargo) {
    return -PAL_ERROR_NOTIMPLEMENTED;
}

int _DkReceiveHandle(PAL_HANDLE source_process, PAL_HANDLE* out_cargo) {
    return -PAL_ERROR_NOTIMPLEMENTED;
}

int _DkInitDebugStream(const char* path) {
    return -PAL_ERROR_NOTIMPLEMENTED;
}

int _DkDebugLog(const void* buf, size_t size) {
    return -PAL_ERROR_NOTIMPLEMENTED;
}
