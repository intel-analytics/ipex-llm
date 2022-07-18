/* SPDX-License-Identifier: LGPL-3.0-or-later */
/* Copyright (C) 2019 Intel Corporation */

/*
 * This file contains operations to handle streams with URIs that have "eventfd:".
 */

#include "api.h"
#include "pal.h"
#include "pal_error.h"
#include "pal_internal.h"

/* `type` must be eventfd, `uri` & `access` & `share` are unused, `create` holds eventfd's initval,
 * `options` holds eventfd's flags */
static int eventfd_pal_open(PAL_HANDLE* handle, const char* type, const char* uri,
                            enum pal_access access, pal_share_flags_t share,
                            enum pal_create_mode create, pal_stream_options_t options) {
    return -PAL_ERROR_NOTIMPLEMENTED;
}

/* offset does not apply here. */
static int64_t eventfd_pal_read(PAL_HANDLE handle, uint64_t offset, uint64_t len, void* buffer) {
    return -PAL_ERROR_NOTIMPLEMENTED;
}

/* offset does not apply here. */
static int64_t eventfd_pal_write(PAL_HANDLE handle, uint64_t offset, uint64_t len,
                                 const void* buffer) {
    return -PAL_ERROR_NOTIMPLEMENTED;
}

/* gets used for polling(query) on eventfd from LibOS. */
static int eventfd_pal_attrquerybyhdl(PAL_HANDLE handle, PAL_STREAM_ATTR* attr) {
    return -PAL_ERROR_NOTIMPLEMENTED;
}

static int eventfd_pal_close(PAL_HANDLE handle) {
    return -PAL_ERROR_NOTIMPLEMENTED;
}

struct handle_ops g_eventfd_ops = {
    .open           = &eventfd_pal_open,
    .read           = &eventfd_pal_read,
    .write          = &eventfd_pal_write,
    .close          = &eventfd_pal_close,
    .attrquerybyhdl = &eventfd_pal_attrquerybyhdl,
};
