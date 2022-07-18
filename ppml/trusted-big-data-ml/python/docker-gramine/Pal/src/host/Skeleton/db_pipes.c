/* SPDX-License-Identifier: LGPL-3.0-or-later */
/* Copyright (C) 2014 Stony Brook University */

/*
 * This file contains operands to handle streams with URIs that start with "pipe:" or "pipe.srv:".
 */

#include "api.h"
#include "pal.h"
#include "pal_error.h"
#include "pal_internal.h"

static int pipe_listen(PAL_HANDLE* handle, const char* name, pal_stream_options_t options) {
    return -PAL_ERROR_NOTIMPLEMENTED;
}

static int pipe_waitforclient(PAL_HANDLE handle, PAL_HANDLE* client, pal_stream_options_t options) {
    return -PAL_ERROR_NOTIMPLEMENTED;
}

static int pipe_connect(PAL_HANDLE* handle, const char* name, pal_stream_options_t options) {
    return -PAL_ERROR_NOTIMPLEMENTED;
}

static int pipe_open(PAL_HANDLE* handle, const char* type, const char* uri, enum pal_access access,
                     pal_share_flags_t share, enum pal_create_mode create,
                     pal_stream_options_t options) {
    if (strlen(uri) + 1 > PIPE_NAME_MAX)
        return -PAL_ERROR_INVAL;

    if (!strcmp(type, URI_TYPE_PIPE_SRV))
        return pipe_listen(handle, uri, options);

    if (!strcmp(type, URI_TYPE_PIPE))
        return pipe_connect(handle, uri, options);

    return -PAL_ERROR_INVAL;
}

/* 'read' operation of pipe stream. offset does not apply here. */
static int64_t pipe_read(PAL_HANDLE handle, uint64_t offset, uint64_t len, void* buffer) {
    return -PAL_ERROR_NOTIMPLEMENTED;
}

/* 'write' operation of pipe stream. offset does not apply here. */
static int64_t pipe_write(PAL_HANDLE handle, uint64_t offset, uint64_t len, const void* buffer) {
    return -PAL_ERROR_NOTIMPLEMENTED;
}

/* 'close' operation of pipe stream. */
static int pipe_close(PAL_HANDLE handle) {
    return -PAL_ERROR_NOTIMPLEMENTED;
}

/* 'delete' operation of pipe stream. */
static int pipe_delete(PAL_HANDLE handle, enum pal_delete_mode delete_mode) {
    return -PAL_ERROR_NOTIMPLEMENTED;
}

static int pipe_getname(PAL_HANDLE handle, char* buffer, size_t count) {
    return -PAL_ERROR_NOTIMPLEMENTED;
}

struct handle_ops g_pipe_ops = {
    .getname       = &pipe_getname,
    .open          = &pipe_open,
    .waitforclient = &pipe_waitforclient,
    .read          = &pipe_read,
    .write         = &pipe_write,
    .close         = &pipe_close,
    .delete        = &pipe_delete,
};
