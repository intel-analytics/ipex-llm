/* SPDX-License-Identifier: LGPL-3.0-or-later */
/* Copyright (C) 2019 Intel Corporation */

/*
 * This file contains operations to handle streams with URIs that have "eventfd:".
 */

#include <asm/poll.h>
#include <sys/eventfd.h>

#include "api.h"
#include "pal.h"
#include "pal_error.h"
#include "pal_internal.h"
#include "pal_linux.h"
#include "pal_linux_error.h"

static inline int eventfd_type(int options) {
    int type = 0;
    if (options & PAL_OPTION_NONBLOCK)
        type |= EFD_NONBLOCK;

    if (options & PAL_OPTION_CLOEXEC)
        type |= EFD_CLOEXEC;

    if (options & PAL_OPTION_EFD_SEMAPHORE)
        type |= EFD_SEMAPHORE;

    return type;
}

/* `type` must be eventfd, `uri` & `access` & `share` are unused, `create` holds eventfd's initval,
 * `options` holds eventfd's flags */
static int eventfd_pal_open(PAL_HANDLE* handle, const char* type, const char* uri,
                            enum pal_access access, pal_share_flags_t share,
                            enum pal_create_mode create, pal_stream_options_t options) {
    __UNUSED(access);
    __UNUSED(share);
    __UNUSED(create);
    assert(create == PAL_CREATE_IGNORED);

    if (strcmp(type, URI_TYPE_EVENTFD) != 0 || *uri != '\0') {
        return -PAL_ERROR_INVAL;
    }

    int fd = ocall_eventfd(eventfd_type(options));
    if (fd < 0)
        return unix_to_pal_error(fd);

    PAL_HANDLE hdl = calloc(1, HANDLE_SIZE(eventfd));
    if (!hdl) {
        ocall_close(fd);
        return -PAL_ERROR_NOMEM;
    }
    init_handle_hdr(hdl, PAL_TYPE_EVENTFD);

    /* Note: using index 0, given that there is only 1 eventfd FD per pal-handle. */
    hdl->flags = PAL_HANDLE_FD_READABLE | PAL_HANDLE_FD_WRITABLE;

    hdl->eventfd.fd          = fd;
    hdl->eventfd.nonblocking = !!(options & PAL_OPTION_NONBLOCK);
    *handle = hdl;

    return 0;
}

static int64_t eventfd_pal_read(PAL_HANDLE handle, uint64_t offset, uint64_t len, void* buffer) {
    if (offset)
        return -PAL_ERROR_INVAL;

    if (HANDLE_HDR(handle)->type != PAL_TYPE_EVENTFD)
        return -PAL_ERROR_NOTCONNECTION;

    if (len < sizeof(uint64_t))
        return -PAL_ERROR_INVAL;

    /* TODO: verify that the value returned in buffer is somehow meaningful (to prevent Iago
     * attacks) */
    ssize_t bytes = ocall_read(handle->eventfd.fd, buffer, len);

    if (bytes < 0)
        return unix_to_pal_error(bytes);

    return bytes;
}

static int64_t eventfd_pal_write(PAL_HANDLE handle, uint64_t offset, uint64_t len,
                                 const void* buffer) {
    if (offset)
        return -PAL_ERROR_INVAL;

    if (HANDLE_HDR(handle)->type != PAL_TYPE_EVENTFD)
        return -PAL_ERROR_NOTCONNECTION;

    if (len < sizeof(uint64_t))
        return -PAL_ERROR_INVAL;

    ssize_t bytes = ocall_write(handle->eventfd.fd, buffer, len);
    if (bytes < 0)
        return unix_to_pal_error(bytes);

    return bytes;
}

/* invoked during poll operation on eventfd from LibOS. */
static int eventfd_pal_attrquerybyhdl(PAL_HANDLE handle, PAL_STREAM_ATTR* attr) {
    int ret;

    if (handle->eventfd.fd == PAL_IDX_POISON)
        return -PAL_ERROR_BADHANDLE;

    attr->handle_type  = HANDLE_HDR(handle)->type;
    attr->nonblocking  = handle->eventfd.nonblocking;

    /* get number of bytes available for reading */
    ret = ocall_fionread(handle->eventfd.fd);
    if (ret < 0)
        return unix_to_pal_error(ret);

    attr->pending_size = ret;

    return 0;
}

static int eventfd_pal_close(PAL_HANDLE handle) {
    if (HANDLE_HDR(handle)->type == PAL_TYPE_EVENTFD) {
        if (handle->eventfd.fd != PAL_IDX_POISON) {
            ocall_close(handle->eventfd.fd);
            handle->eventfd.fd = PAL_IDX_POISON;
        }
        return 0;
    }

    return 0;
}

struct handle_ops g_eventfd_ops = {
    .open           = &eventfd_pal_open,
    .read           = &eventfd_pal_read,
    .write          = &eventfd_pal_write,
    .close          = &eventfd_pal_close,
    .attrquerybyhdl = &eventfd_pal_attrquerybyhdl,
};
