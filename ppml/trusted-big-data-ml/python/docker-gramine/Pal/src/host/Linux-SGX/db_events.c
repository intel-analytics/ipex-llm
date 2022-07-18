/* SPDX-License-Identifier: LGPL-3.0-or-later */
/* Copyright (C) 2021 Intel Corporation
 *                    Borys Popławski <borysp@invisiblethingslab.com>
 */

#include <asm/errno.h>
#include <limits.h>
#include <linux/futex.h>
#include <stdbool.h>

#include "assert.h"
#include "enclave_ocalls.h"
#include "pal.h"
#include "pal_internal.h"
#include "pal_linux_error.h"
#include "spinlock.h"

int _DkEventCreate(PAL_HANDLE* handle_ptr, bool init_signaled, bool auto_clear) {
    PAL_HANDLE handle = calloc(1, HANDLE_SIZE(event));
    if (!handle) {
        return -PAL_ERROR_NOMEM;
    }

    init_handle_hdr(handle, PAL_TYPE_EVENT);
    handle->event.signaled_untrusted = malloc_untrusted(sizeof(*handle->event.signaled_untrusted));
    if (!handle->event.signaled_untrusted) {
        free(handle);
        return -PAL_ERROR_NOMEM;
    }
    spinlock_init(&handle->event.lock);
    handle->event.waiters_cnt = 0;
    handle->event.signaled = init_signaled;
    handle->event.auto_clear = auto_clear;
    __atomic_store_n(handle->event.signaled_untrusted, init_signaled ? 1 : 0, __ATOMIC_RELEASE);

    *handle_ptr = handle;
    return 0;
}

void _DkEventSet(PAL_HANDLE handle) {
    spinlock_lock(&handle->event.lock);
    handle->event.signaled = true;
    __atomic_store_n(handle->event.signaled_untrusted, 1, __ATOMIC_RELEASE);
    bool need_wake = handle->event.waiters_cnt > 0;
    spinlock_unlock(&handle->event.lock);

    if (need_wake) {
        int ret = 0;
        do {
            ret = ocall_futex(handle->event.signaled_untrusted, FUTEX_WAKE,
                              handle->event.auto_clear ? 1 : INT_MAX, /*timeout=*/NULL);
        } while (ret == -EINTR);
        /* This `FUTEX_WAKE` cannot really fail. Negative return value would mean malicious host,
         * but it could also report `0` here and not perform the wakeup, so the worst case scenario
         * is just a DoS, which we don't really care about. */
        assert(ret >= 0);
    }
}

void _DkEventClear(PAL_HANDLE handle) {
    spinlock_lock(&handle->event.lock);
    handle->event.signaled = false;
    __atomic_store_n(handle->event.signaled_untrusted, 0, __ATOMIC_RELEASE);
    spinlock_unlock(&handle->event.lock);
}

/* We use `handle->event.signaled` as the source of truth whether the event was signaled.
 * `handle->event.signaled_untrusted` acts only as a futex sleeping word. */
int _DkEventWait(PAL_HANDLE handle, uint64_t* timeout_us) {
    bool added_to_count = false;
    while (1) {
        spinlock_lock(&handle->event.lock);
        if (handle->event.signaled) {
            if (handle->event.auto_clear) {
                handle->event.signaled = false;
                __atomic_store_n(handle->event.signaled_untrusted, 0, __ATOMIC_RELEASE);
            }
            if (added_to_count) {
                handle->event.waiters_cnt--;
            }
            spinlock_unlock(&handle->event.lock);
            return 0;
        }

        if (!added_to_count) {
            handle->event.waiters_cnt++;
            added_to_count = true;
        }
        spinlock_unlock(&handle->event.lock);

        int ret = ocall_futex(handle->event.signaled_untrusted, FUTEX_WAIT, 0, timeout_us);
        if (ret < 0 && ret != -EAGAIN) {
            if (added_to_count) {
                spinlock_lock(&handle->event.lock);
                handle->event.waiters_cnt--;
                spinlock_unlock(&handle->event.lock);
            }
            return unix_to_pal_error(ret);
        }
    }
}

static int event_close(PAL_HANDLE handle) {
    free_untrusted(handle->event.signaled_untrusted);
    return 0;
}

struct handle_ops g_event_ops = {
    .close = event_close,
};
