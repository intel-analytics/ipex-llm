/* SPDX-License-Identifier: LGPL-3.0-or-later */
/* Copyright (C) 2021 Intel Corporation
 *                    Borys Popławski <borysp@invisiblethingslab.com>
 */
#ifndef SHIM_POLLABLE_EVENT_H
#define SHIM_POLLABLE_EVENT_H

#include "pal.h"
#include "spinlock.h"

/* TODO: once epoll is rewritten, change these to normal events with two states (set and not set),
 * remove `wait_pollable_event` and make all handles nonblocking. */
/*
 * These events have counting semaphore semantics:
 * - `set_pollable_event(e, n)` increases value of the semaphore by `n`,
 * - `wait_pollable_event(e)` decreases value by 1 (blocking if it's 0),
 * - `clear_pollable_event(e)` decreases value to 0 without blocking - this operation is not atomic.
 * Additionally `e->read_handle` can be passed to `DkStreamsWaitEvents` (which is actually the only
 * purpose these events exist for).
 */

struct shim_pollable_event {
    PAL_HANDLE read_handle;
    PAL_HANDLE write_handle;
    spinlock_t read_lock;
    spinlock_t write_lock;
};

int create_pollable_event(struct shim_pollable_event* event);
void destroy_pollable_event(struct shim_pollable_event* event);
int set_pollable_event(struct shim_pollable_event* event, size_t n);
int wait_pollable_event(struct shim_pollable_event* event);
int clear_pollable_event(struct shim_pollable_event* event);

#endif // SHIM_POLLABLE_EVENT_H
