/* SPDX-License-Identifier: LGPL-3.0-or-later */
/* Copyright (C) 2014 Stony Brook University */

/*
 * Operations to handle devices (currently only "dev:tty" which is stdin/stdout).
 */

#include "api.h"
#include "pal.h"
#include "pal_error.h"
#include "pal_internal.h"

static int dev_open(PAL_HANDLE* handle, const char* type, const char* uri, enum pal_access access,
                    pal_share_flags_t share, enum pal_create_mode create,
                    pal_stream_options_t options) {
    return -PAL_ERROR_NOTIMPLEMENTED;
}

static int64_t dev_read(PAL_HANDLE handle, uint64_t offset, uint64_t size, void* buffer) {
    return -PAL_ERROR_NOTIMPLEMENTED;
}

static int64_t dev_write(PAL_HANDLE handle, uint64_t offset, uint64_t size, const void* buffer) {
    return -PAL_ERROR_NOTIMPLEMENTED;
}

static int dev_close(PAL_HANDLE handle) {
    return -PAL_ERROR_NOTIMPLEMENTED;
}

static int dev_flush(PAL_HANDLE handle) {
    return -PAL_ERROR_NOTIMPLEMENTED;
}

static int dev_attrquery(const char* type, const char* uri, PAL_STREAM_ATTR* attr) {
    return -PAL_ERROR_NOTIMPLEMENTED;
}

static int dev_attrquerybyhdl(PAL_HANDLE handle, PAL_STREAM_ATTR* attr) {
    return -PAL_ERROR_NOTIMPLEMENTED;
}

/* this dummy function is implemented to support opening TTY devices with O_TRUNC flag */
static int64_t dev_setlength(PAL_HANDLE handle, uint64_t length) {
    return -PAL_ERROR_NOTIMPLEMENTED;
}

struct handle_ops g_dev_ops = {
    .open           = &dev_open,
    .read           = &dev_read,
    .write          = &dev_write,
    .close          = &dev_close,
    .setlength      = &dev_setlength,
    .flush          = &dev_flush,
    .attrquery      = &dev_attrquery,
    .attrquerybyhdl = &dev_attrquerybyhdl,
};

int _DkDeviceIoControl(PAL_HANDLE handle, unsigned int cmd, uint64_t arg) {
    __UNUSED(handle);
    __UNUSED(cmd);
    __UNUSED(arg);
    return -PAL_ERROR_NOTIMPLEMENTED;
}
