/* SPDX-License-Identifier: LGPL-3.0-or-later */
/* Copyright (C) 2021 Intel Corporation
 *                    Borys Popławski <borysp@invisiblethingslab.com>
 */

#include "api.h"
#include "log.h"
#include "pal.h"
#include "shim_internal.h"
#include "shim_pollable_event.h"
#include "shim_utils.h"

int create_pollable_event(struct shim_pollable_event* event) {
    char uri[PIPE_URI_SIZE];
    PAL_HANDLE srv_handle;
    int ret = create_pipe(/*name=*/NULL, uri, sizeof(uri), &srv_handle,
                          /*use_vmid_for_name=*/false);
    if (ret < 0) {
        log_error("%s: create_pipe failed: %d", __func__, ret);
        return ret;
    }

    PAL_HANDLE write_handle;
    do {
        ret = DkStreamOpen(uri, PAL_ACCESS_RDWR, /*share_flags=*/0, PAL_CREATE_IGNORED,
                           PAL_OPTION_CLOEXEC, &write_handle);
    } while (ret == -PAL_ERROR_INTERRUPTED);
    if (ret < 0) {
        log_error("%s: DkStreamOpen failed: %d", __func__, ret);
        ret = pal_to_unix_errno(ret);
        goto out;
    }

    PAL_HANDLE read_handle;
    do {
        ret = DkStreamWaitForClient(srv_handle, &read_handle, PAL_OPTION_NONBLOCK);
    } while (ret == -PAL_ERROR_INTERRUPTED);
    if (ret < 0) {
        log_error("%s: DkStreamWaitForClient failed: %d", __func__, ret);
        DkObjectClose(write_handle);
        ret = pal_to_unix_errno(ret);
        goto out;
    }

    event->read_handle = read_handle;
    event->write_handle = write_handle;
    spinlock_init(&event->read_lock);
    spinlock_init(&event->write_lock);
    ret = 0;

out:;
    int tmp_ret = pal_to_unix_errno(DkStreamDelete(srv_handle, PAL_DELETE_ALL));
    DkObjectClose(srv_handle);
    if (!ret && tmp_ret) {
        DkObjectClose(read_handle);
        DkObjectClose(write_handle);
        /* Clearing just for sanity. */
        event->read_handle = NULL;
        event->write_handle = NULL;
    }
    return ret ?: tmp_ret;
}

void destroy_pollable_event(struct shim_pollable_event* event) {
    DkObjectClose(event->read_handle);
    DkObjectClose(event->write_handle);
}

int set_pollable_event(struct shim_pollable_event* event, size_t n) {
    int ret = 0;

    spinlock_lock(&event->write_lock);

    while (n > 0) {
        char buf[0x20] = { 0 };
        size_t size = MIN(sizeof(buf), n);
        int ret = DkStreamWrite(event->write_handle, /*offset=*/0, &size, buf, /*dest=*/NULL);
        if (ret < 0) {
            if (ret == -PAL_ERROR_INTERRUPTED) {
                continue;
            }
            ret = pal_to_unix_errno(ret);
            goto out;
        }
        if (size == 0) {
            ret = -EINVAL;
            goto out;
        }
        n -= size;
    }
    ret = 0;

out:
    spinlock_unlock(&event->write_lock);
    return ret;
}

int wait_pollable_event(struct shim_pollable_event* event) {
    int ret = 0;

    spinlock_lock(&event->read_lock);

    do {
        char c;
        size_t size = sizeof(c);
        ret = DkStreamRead(event->read_handle, /*offset=*/0, &size, &c, NULL, 0);
        if (ret < 0) {
            ret = pal_to_unix_errno(ret);
        } else if (size == 0) {
            ret = -EINVAL;
        }
    } while (ret == -EINTR || ret == -EAGAIN);

    spinlock_unlock(&event->read_lock);
    return ret;
}

int clear_pollable_event(struct shim_pollable_event* event) {
    int ret = 0;

    spinlock_lock(&event->read_lock);

    while (1) {
        char buf[0x100];
        size_t size = sizeof(buf);
        int ret = DkStreamRead(event->read_handle, /*offset=*/0, &size, buf, NULL, 0);
        if (ret < 0) {
            if (ret == -PAL_ERROR_INTERRUPTED) {
                continue;
            } else if (ret == -PAL_ERROR_TRYAGAIN) {
                /* Event not set. */
                break;
            }
            ret = pal_to_unix_errno(ret);
            goto out;
        }
        if (size == 0) {
            ret = -EINVAL;
            goto out;
        }
    }
    ret = 0;

out:
    spinlock_unlock(&event->read_lock);
    return ret;
}
