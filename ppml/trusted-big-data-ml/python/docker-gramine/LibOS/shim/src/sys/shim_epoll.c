/* SPDX-License-Identifier: LGPL-3.0-or-later */
/* Copyright (C) 2014 Stony Brook University
 * Copyright (C) 2022 Intel Corporation
 *                    Borys Popławski <borysp@invisiblethingslab.com>
 */

/*
 * epoll family of syscalls implementation.
 * Current limitations:
 * - sharing an epoll instance between processes - updates in one process (e.g. adding an fd to be
 *   monitored) won't be visible in the other process; state is only migrated at the moment of
 *   `fork()` call,
 * - `EPOLLEXCLUSIVE` is a no-op - this is correct semantically, but may reduce performance of apps
 *   using this flag,
 * - adding an epoll to another epoll instance is not supported, but should be implementable without
 *   design changes if need be,
 * - `EPOLLRDHUP` is not reported and `EPOLLHUP` is always reported together with `EPOLLERR` - this
 *   is current limitation of PAL API, which does not distinguish these conditions.
 */

#include <stdint.h>

#include "api.h"
#include "list.h"
#include "shim_checkpoint.h"
#include "shim_fs.h"
#include "shim_handle.h"
#include "shim_internal.h"
#include "shim_pollable_event.h"
#include "shim_signal.h"
#include "shim_table.h"
#include "shim_thread.h"
#include "shim_types.h"

/* This bit is currently unoccupied in epoll events mask. */
#define EPOLL_NEEDS_REARM ((uint32_t)(1u << 24))

/*
 * The following diagram could help you understand relationships between different structs used in
 * this code.

                                     +-----------------------+
                                     |                       |
                                     |  shim_epoll_item      |         item in the interest
                                     |                       |        list of epoll instance
          item in the list           |    epoll_list   <-------------------------------------------+
         of epoll instances          |                       |     guarded by handle::epoll::lock  |
+---------------------------------------> handle_list        |                                     |
|     guarded by handle::lock        |                       |                                     |
|                                    |    epoll_handle +-------+                                   |
|                                    |                       | |                                   |
|                                  +----+ handle             | +->+----------------------------+   |
|                                  | |                       |    |                            |   |
|  +----------------------------+<-+ |    fd                 |    |  shim_handle (epoll)       |   |
|  |                            |    |                       |    |                            |   |
|  | shim_handle (pipe,sock,..) |    +-----------------------+    |    epoll_items list        |   |
|  |                            |                                 |                            |   |
+-----+ epoll_items list        |       +--------------------+    |    shim_epoll_handle epoll |   |
   |                            |       |                    |    |                            |   |
   |    lock                    |       |  shim_epoll_waiter | +--------+ waiters list         |   |
   |                            |       |                    | |  |                            |   |
   |                            |       |    list <------------+  |       items   list +-----------+
   |                            |       |                    |    |                            |
   +----------------------------+       |    event           |    |       lock                 |
                                        |                    |    |                            |
                                        +--------------------+    +----------------------------+
*/

DEFINE_LIST(shim_epoll_item);
struct shim_epoll_item {
    /* Guarded by `epoll_handle->info.epoll.lock`. */
    LIST_TYPE(shim_epoll_item) epoll_list; // epoll_handle->items
    /* Guarded by `handle->lock`. */
    LIST_TYPE(shim_epoll_item) handle_list; // handle->epoll_items
    /* `epoll_handle`, `handle` and `fd` are constant and thus require no locking. */
    struct shim_handle* epoll_handle;
    struct shim_handle* handle;
    int fd;
    /* `events` and `data` are guarded by `epoll_handle->info.epoll.lock`. */
    uint32_t events;
    uint64_t data;
    REFTYPE ref_count;
};

DEFINE_LIST(shim_epoll_waiter);
struct shim_epoll_waiter {
    /* Guarded by `epoll_handle->info.epoll.lock`, where `epoll_handle` is handle this waiter called
     * `epoll_wait` on. */
    LIST_TYPE(shim_epoll_waiter) list; // shim_epoll_handle::waiters
    struct shim_pollable_event* event;
};

static void get_epoll_item(struct shim_epoll_item* item) {
    REF_INC(item->ref_count);
}

static void put_epoll_item(struct shim_epoll_item* item) {
    int64_t ref_count = REF_DEC(item->ref_count);

    if (!ref_count) {
        put_handle(item->epoll_handle);
        put_handle(item->handle);
        free(item);
    }
}

static void put_epoll_items_array(struct shim_epoll_item** items, size_t items_count) {
    for (size_t i = 0; i < items_count; i++) {
        put_epoll_item(items[i]);
    }
}

static void _interrupt_epoll_waiters(struct shim_epoll_handle* epoll) {
    assert(locked(&epoll->lock));

    struct shim_epoll_waiter* waiter;
    struct shim_epoll_waiter* tmp;
    LISTP_FOR_EACH_ENTRY_SAFE(waiter, tmp, &epoll->waiters, list) {
        set_pollable_event(waiter->event, 1);
        LISTP_DEL_INIT(waiter, &epoll->waiters, list);
    }
    assert(LISTP_EMPTY(&epoll->waiters));
}

void interrupt_epolls(struct shim_handle* handle) {
    lock(&handle->lock);
    struct shim_epoll_item** items = NULL;
    /* 4 is an arbitrary number. We don't expect more than 1-2 epoll items per handle. */
    struct shim_epoll_item* items_inline[4] = { 0 };
    size_t items_count = handle->epoll_items_count;

    if (items_count <= ARRAY_SIZE(items_inline)) {
        /* Optimize common case of small number of items per handle. */
        items = items_inline;
    } else {
        items = malloc(items_count * sizeof(*items));
        if (!items) {
            unlock(&handle->lock);
            log_error("%s: failed to allocate memory for the epoll items array", __func__);
            /* No way to handle this cleanly. */
            DkProcessExit(1);
        }
    }

    struct shim_epoll_item* item;
    size_t i = 0;
    LISTP_FOR_EACH_ENTRY(item, &handle->epoll_items, handle_list) {
        items[i++] = item;
        get_epoll_item(item);
    }
    assert(i == items_count);
    unlock(&handle->lock);

    for (size_t i = 0; i < items_count; i++) {
        struct shim_epoll_handle* epoll = &items[i]->epoll_handle->info.epoll;
        lock(&epoll->lock);
        _interrupt_epoll_waiters(epoll);
        unlock(&epoll->lock);
    }

    put_epoll_items_array(items, items_count);
    if (items != items_inline) {
        free(items);
    }
}

void maybe_epoll_et_trigger(struct shim_handle* handle, int ret, bool in, bool was_partial) {
    bool needs_et = false;
    switch (handle->type) {
        case TYPE_SOCK:
        case TYPE_PIPE:
            needs_et = ret == -EAGAIN || was_partial;
            break;
        case TYPE_EVENTFD:
            needs_et = handle->info.eventfd.is_semaphore ? ret == -EAGAIN : true;
            if (!in) {
                /*
                 * Some workloads (e.g. rust's tokio crate) use eventfd with EPOLLET in a peculiar
                 * way: each write to that eventfd increases counter by 1 and thanks to EPOLLET is
                 * reported by epoll only once, even if there is no read from eventfd.
                 * To handle such usage pattern, we mark eventfd as read-epollet-pollable on each
                 * write - we assume that eventfd is not shared between processes.
                 * Hopefully no app tries to increase the eventfd counter by 0...
                 */
                __atomic_store_n(&handle->needs_et_poll_in, true, __ATOMIC_RELEASE);
                needs_et = true;
            }
            break;
        default:
            /* Type unsupported with EPOLLET. */
            break;
    }

    if (needs_et) {
        if (in) {
            __atomic_store_n(&handle->needs_et_poll_in, true, __ATOMIC_RELEASE);
        } else {
            __atomic_store_n(&handle->needs_et_poll_out, true, __ATOMIC_RELEASE);
        }

        interrupt_epolls(handle);
    }
}

/* Make sure we have an owned reference to `item` when calling this function, as it unlinks `item`
 * from all lists (hence dropping these references). This means we cannot rely on borrowing
 * a reference when traversing one of item's lists. */
static void _unlink_epoll_item(struct shim_epoll_item* item) {
    struct shim_handle* handle = item->handle;
    struct shim_epoll_handle* epoll = &item->epoll_handle->info.epoll;
    assert(locked(&epoll->lock));

    lock(&handle->lock);
    if (!LIST_EMPTY(item, handle_list)) {
        LISTP_DEL_INIT(item, &handle->epoll_items, handle_list);
        handle->epoll_items_count--;
        put_epoll_item(item);
    }
    unlock(&handle->lock);

    if (!LIST_EMPTY(item, epoll_list)) {
        LISTP_DEL_INIT(item, &epoll->items, epoll_list);
        epoll->items_count--;
        put_epoll_item(item);
    }
}

void delete_epoll_items_for_fd(int fd, struct shim_handle* handle) {
    /* This looks scarry, but in practice shouldn't be that bad - `fd` is rarely registered on
     * multiple epolls and even if it is, there shouldn't be many of them. */
    while (1) {
        struct shim_epoll_item* to_unlink = NULL;
        lock(&handle->lock);
        struct shim_epoll_item* item;
        LISTP_FOR_EACH_ENTRY(item, &handle->epoll_items, handle_list) {
            assert(item->handle == handle);
            if (item->fd == fd) {
                to_unlink = item;
                get_epoll_item(item);
                break;
            }
        }
        unlock(&handle->lock);

        if (to_unlink) {
            struct shim_epoll_handle* epoll = &to_unlink->epoll_handle->info.epoll;
            lock(&epoll->lock);
            _unlink_epoll_item(to_unlink);
            _interrupt_epoll_waiters(epoll);
            unlock(&epoll->lock);
            put_epoll_item(to_unlink);
        } else {
            break;
        }
    }
}

long shim_do_epoll_create1(int flags) {
    if (!WITHIN_MASK(flags, EPOLL_CLOEXEC)) {
        return -EINVAL;
    }

    struct shim_handle* handle = get_new_handle();
    if (!handle) {
        return -ENOMEM;
    }

    handle->type = TYPE_EPOLL;
    handle->fs = &epoll_builtin_fs;

    struct shim_epoll_handle* epoll = &handle->info.epoll;
    INIT_LISTP(&epoll->waiters);
    INIT_LISTP(&epoll->items);
    epoll->items_count = 0;
    epoll->last_returned_index = -1;
    if (!create_lock(&epoll->lock)) {
        put_handle(handle);
        return -ENOMEM;
    }

    int ret = set_new_fd_handle(handle, (flags & EPOLL_CLOEXEC) ? FD_CLOEXEC : 0,
                                /*handle_map=*/NULL);
    put_handle(handle);
    return ret;
}

long shim_do_epoll_create(int size) {
    if (size <= 0) {
        return -EINVAL;
    }

    /* `size` argument is obsolete and unused. */
    return shim_do_epoll_create1(/*flags=*/0);
}

static int do_epoll_add(struct shim_handle* epoll_handle, struct shim_handle* handle, int fd,
                        struct epoll_event* event) {
    if (event->events & EPOLLEXCLUSIVE) {
        if (!WITHIN_MASK(event->events, EPOLLIN | EPOLLOUT | EPOLLHUP | EPOLLERR | EPOLLWAKEUP
                                        | EPOLLET | EPOLLEXCLUSIVE)) {
            return -EINVAL;
        }
        /* We do not support `EPOLLEXCLUSIVE`, but a no-op implementation is correct (although not
         * the most performant), so we allow for it. */
    }

    static_assert(!WITHIN_MASK(EPOLL_NEEDS_REARM, EPOLLIN | EPOLLPRI | EPOLLOUT | EPOLLERR
                                                  | EPOLLHUP | EPOLLNVAL | EPOLLRDNORM | EPOLLRDBAND
                                                  | EPOLLWRNORM | EPOLLWRBAND | EPOLLMSG
                                                  | EPOLLRDHUP | EPOLLEXCLUSIVE | EPOLLWAKEUP
                                                  | EPOLLONESHOT | EPOLLET),
                  "EPOLL_NEEDS_REARM bit occupied by another epoll event");

    int ret;
    struct shim_epoll_item* new_item = malloc(sizeof(*new_item));
    if (!new_item) {
        return -ENOMEM;
    }

    new_item->handle = handle;
    get_handle(handle);
    new_item->fd = fd;
    new_item->epoll_handle = epoll_handle;
    get_handle(epoll_handle);
    new_item->data = event->data;
    new_item->events = event->events & ~EPOLL_NEEDS_REARM;
    if (!(handle->acc_mode & MAY_READ)) {
        new_item->events &= ~(EPOLLIN | EPOLLRDNORM);
    }
    if (!(handle->acc_mode & MAY_WRITE)) {
        new_item->events &= ~(EPOLLOUT | EPOLLWRNORM);
    }
    REF_SET(new_item->ref_count, 1);

    struct shim_epoll_handle* epoll = &epoll_handle->info.epoll;

    lock(&epoll->lock);

    struct shim_epoll_item* item;
    LISTP_FOR_EACH_ENTRY(item, &epoll->items, epoll_list) {
        if (item->fd == fd && item->handle == handle) {
            ret = -EEXIST;
            goto out_unlock;
        }
    }

    LISTP_ADD_TAIL(new_item, &epoll->items, epoll_list);
    get_epoll_item(new_item);
    epoll->items_count++;

    lock(&handle->lock);
    LISTP_ADD_TAIL(new_item, &handle->epoll_items, handle_list);
    get_epoll_item(new_item);
    handle->epoll_items_count++;
    unlock(&handle->lock);

    if (new_item->events & EPOLLET) {
        __atomic_store_n(&handle->needs_et_poll_in, true, __ATOMIC_RELEASE);
        __atomic_store_n(&handle->needs_et_poll_out, true, __ATOMIC_RELEASE);
    }

    _interrupt_epoll_waiters(epoll);

    log_debug("epoll: added %d (%p) to epoll handle %p", fd, handle, epoll_handle);
    ret = 0;

out_unlock:
    unlock(&epoll->lock);
    put_epoll_item(new_item);
    return ret;
}

static int do_epoll_mod(struct shim_handle* epoll_handle, struct shim_handle* handle, int fd,
                        struct epoll_event* event) {
    if (event->events & EPOLLEXCLUSIVE) {
        return -EINVAL;
    }

    struct shim_epoll_handle* epoll = &epoll_handle->info.epoll;
    int ret = -ENOENT;

    lock(&epoll->lock);

    struct shim_epoll_item* item;
    LISTP_FOR_EACH_ENTRY(item, &epoll->items, epoll_list) {
        if (item->fd == fd && item->handle == handle) {
            if (item->events & EPOLLEXCLUSIVE) {
                ret = -EINVAL;
                goto out_unlock;
            }

            item->events = event->events & ~EPOLL_NEEDS_REARM;
            item->data = event->data;

            if (item->events & EPOLLET) {
                __atomic_store_n(&handle->needs_et_poll_in, true, __ATOMIC_RELEASE);
                __atomic_store_n(&handle->needs_et_poll_out, true, __ATOMIC_RELEASE);
            }

            _interrupt_epoll_waiters(epoll);

            log_debug("epoll: modified %d (%p) on epoll handle %p", fd, handle, epoll_handle);
            ret = 0;
            break;
        }
    }

out_unlock:
    unlock(&epoll->lock);
    return ret;
}

static int do_epoll_del(struct shim_handle* epoll_handle, struct shim_handle* handle, int fd) {
    struct shim_epoll_handle* epoll = &epoll_handle->info.epoll;
    int ret = -ENOENT;

    lock(&epoll->lock);

    struct shim_epoll_item* item;
    LISTP_FOR_EACH_ENTRY(item, &epoll->items, epoll_list) {
        if (item->fd == fd && item->handle == handle) {
            get_epoll_item(item);
            _unlink_epoll_item(item);

            _interrupt_epoll_waiters(epoll);

            put_epoll_item(item);

            log_debug("epoll: deleted %d (%p) from epoll handle %p", fd, handle, epoll_handle);
            ret = 0;
            break;
        }
    }

    unlock(&epoll->lock);
    return ret;
}

long shim_do_epoll_ctl(int epfd, int op, int fd, struct epoll_event* event) {
    int ret;
    struct shim_handle* epoll_handle = get_fd_handle(epfd, /*fd_flags=*/NULL, /*map=*/NULL);
    if (!epoll_handle) {
        return -EBADF;
    }
    struct shim_handle* handle = get_fd_handle(fd, /*fd_flags=*/NULL, /*map=*/NULL);
    if (!handle) {
        put_handle(epoll_handle);
        return -EBADF;
    }

    if (epfd == fd || epoll_handle->type != TYPE_EPOLL) {
        ret = -EINVAL;
        goto out;
    }

    switch (handle->type) {
        case TYPE_PIPE:
        case TYPE_SOCK:
        case TYPE_EVENTFD:
            break;
        default:
            /* epoll not supported by this type of handle */
            ret = -EPERM;
            goto out;
    }

    if (op == EPOLL_CTL_ADD || op == EPOLL_CTL_MOD) {
        if (!is_user_memory_readable(event, sizeof(*event))) {
            ret = -EFAULT;
            goto out;
        }
    }

    switch (op) {
        case EPOLL_CTL_ADD:
            ret = do_epoll_add(epoll_handle, handle, fd, event);
            break;
        case EPOLL_CTL_MOD:
            ret = do_epoll_mod(epoll_handle, handle, fd, event);
            break;
        case EPOLL_CTL_DEL:
            ret = do_epoll_del(epoll_handle, handle, fd);
            break;
        default:
            ret = -EINVAL;
            goto out;
    }

out:
    put_handle(epoll_handle);
    put_handle(handle);
    return ret;
}

static int do_epoll_wait(int epfd, struct epoll_event* events, int maxevents, int timeout_ms) {
    if (maxevents <= 0) {
        return -EINVAL;
    }

    if (!is_user_memory_writable(events, sizeof(*events) * maxevents)) {
        return -EFAULT;
    }

    struct shim_handle* epoll_handle = get_fd_handle(epfd, /*fd_flags=*/NULL, /*map=*/NULL);
    if (!epoll_handle) {
        return -EBADF;
    }
    if (epoll_handle->type != TYPE_EPOLL) {
        put_handle(epoll_handle);
        return -EINVAL;
    }

    uint64_t timeout_us = (unsigned int)timeout_ms * TIME_US_IN_MS;
    struct shim_epoll_waiter waiter = {
        .event = &get_cur_thread()->pollable_event,
    };

    int ret;
    struct shim_epoll_handle* epoll = &epoll_handle->info.epoll;
    struct shim_epoll_item** items = NULL;
    PAL_HANDLE* pal_handles = NULL;
    pal_wait_flags_t* pal_events = NULL;
    size_t arrays_len = 0;

    lock(&epoll->lock);

    while (1) {
        if (arrays_len < epoll->items_count) {
            free(items);
            free(pal_handles);
            free(pal_events);

            arrays_len = epoll->items_count;
            items = malloc(arrays_len * sizeof(*items));
            /* Reserve one slot for the waiter's wakeup handle. */
            pal_handles = malloc((arrays_len + 1) * sizeof(*pal_handles));
            /* Double the amount of PAL events - one part are input events, the other - output. */
            pal_events = malloc(2 * (arrays_len + 1) * sizeof(*pal_events));
            if (!items || !pal_handles || !pal_events) {
                ret = -ENOMEM;
                goto out_unlock;
            }
        }

        pal_wait_flags_t* pal_ret_events = pal_events + epoll->items_count + 1;

        struct shim_epoll_item* item;
        size_t items_count = 0;
        LISTP_FOR_EACH_ENTRY(item, &epoll->items, epoll_list) {
            /* XXX: this is not correct if `pal_handle` can change (we hold no lock)
             * see: https://github.com/gramineproject/gramine/issues/322 */
            if (!item->handle->pal_handle) {
                /* Sockets that are still not connected have no `pal_handle`. */
                continue;
            }

            if (item->events & EPOLL_NEEDS_REARM) {
                assert(item->events & EPOLLONESHOT);
                continue;
            }

            items[items_count] = item;
            get_epoll_item(item);

            /* Since we have a reference to `item` (saved above), we can safely copy and use this
             * PAL handle, even after releasing `epoll->lock`. */
            pal_handles[items_count] = item->handle->pal_handle;

            pal_events[items_count] = 0;
            if (item->events & (EPOLLIN | EPOLLRDNORM)) {
                pal_events[items_count] |= PAL_WAIT_READ;
            }
            if (item->events & (EPOLLOUT | EPOLLWRNORM)) {
                pal_events[items_count] |= PAL_WAIT_WRITE;
            }
            if (item->events & EPOLLET) {
                if (!__atomic_load_n(&item->handle->needs_et_poll_in, __ATOMIC_ACQUIRE)) {
                    pal_events[items_count] &= ~PAL_WAIT_READ;
                }
                if (!__atomic_load_n(&item->handle->needs_et_poll_out, __ATOMIC_ACQUIRE)) {
                    pal_events[items_count] &= ~PAL_WAIT_WRITE;
                }
            }
            pal_ret_events[items_count] = 0;

            items_count++;
        }
        assert(items_count <= epoll->items_count);

        pal_handles[items_count] = waiter.event->read_handle;
        pal_events[items_count] = PAL_WAIT_READ;
        pal_ret_events[items_count] = 0;

        LISTP_ADD_TAIL(&waiter, &epoll->waiters, list);

        unlock(&epoll->lock);

        if (!have_pending_signals()) {
            ret = DkStreamsWaitEvents(items_count + 1, pal_handles, pal_events, pal_ret_events,
                                      timeout_ms == -1 ? NULL : &timeout_us);
            ret = pal_to_unix_errno(ret);
        } else {
            ret = -EINTR;
        }

        lock(&epoll->lock);
        if (!LIST_EMPTY(&waiter, list)) {
            LISTP_DEL(&waiter, &epoll->waiters, list);
        }

        if (ret < 0) {
            if (ret == -EAGAIN) {
                /* Timed out. */
                assert(timeout_us == 0);
                ret = 0;
            } else if (ret == -EINTR) {
                /* `epoll_wait` and `epoll_pwait` are not restarted after being interrupted by
                 * a signal handler. */
                ret = -ERESTARTNOHAND;
            }
            put_epoll_items_array(items, items_count);
            goto out_unlock;
        }

        if (pal_ret_events[items_count]) {
            clear_pollable_event(waiter.event);
        }

        /* Round robin returned events to help avoid starvation scenarios. If there was
         * an asynchronous update on the list of items, it isn't real round robin, but that's fine
         * - no user app should depend on it anyway. */
        size_t start_index = items_count ? (epoll->last_returned_index + 1) % items_count : 0;
        size_t counter = 0;
        size_t ret_events_count = 0;
        for (; counter < items_count; counter++) {
            size_t i = (start_index + counter) % items_count;
            if (!pal_ret_events[i]) {
                continue;
            }

            if (items[i]->events & EPOLL_NEEDS_REARM) {
                /* Another waiter reported events for this EPOLLONESHOT item asynchronously. */
                continue;
            }

            uint32_t this_item_events = 0;
            if (pal_ret_events[i] & PAL_WAIT_ERROR) {
                /* XXX: unfortunately there is no way to distinguish these two. */
                this_item_events |= EPOLLERR | EPOLLHUP;
            }
            if (pal_ret_events[i] & PAL_WAIT_READ) {
                this_item_events |= items[i]->events & (EPOLLIN | EPOLLRDNORM);
            }
            if (pal_ret_events[i] & PAL_WAIT_WRITE) {
                this_item_events |= items[i]->events & (EPOLLOUT | EPOLLWRNORM);
            }

            if (!this_item_events) {
                /* This handle is not interested in events that were detected - epoll item was
                 * probably updated asynchronously. */
                continue;
            }

            events[ret_events_count].events = this_item_events;
            events[ret_events_count].data = items[i]->data;

            if (items[i]->events & EPOLLET) {
                if (this_item_events & (EPOLLIN | EPOLLRDNORM)) {
                    __atomic_store_n(&items[i]->handle->needs_et_poll_in, false, __ATOMIC_RELEASE);
                }
                if (this_item_events & (EPOLLOUT | EPOLLWRNORM)) {
                    __atomic_store_n(&items[i]->handle->needs_et_poll_out, false, __ATOMIC_RELEASE);
                }
            }

            if (items[i]->events & EPOLLONESHOT) {
                items[i]->events |= EPOLL_NEEDS_REARM;
            }

            ret_events_count++;
            if (ret_events_count == (size_t)maxevents) {
                break;
            }
        }

        put_epoll_items_array(items, items_count);

        if (ret_events_count) {
            if (counter == items_count) {
                /* All items were returned to user app. */
                epoll->last_returned_index = -1;
            } else {
                epoll->last_returned_index = (start_index + counter) % items_count;
            }
            ret = ret_events_count;
            break;
        }
        /* There was an update on polled items, gather items once again. */
    }

out_unlock:
    unlock(&epoll->lock);

    free(items);
    free(pal_handles);
    free(pal_events);
    put_handle(epoll_handle);
    return ret;
}

long shim_do_epoll_wait(int epfd, struct epoll_event* events, int maxevents, int timeout_ms) {
    return do_epoll_wait(epfd, events, maxevents, timeout_ms);
}

long shim_do_epoll_pwait(int epfd, struct epoll_event* events, int maxevents, int timeout_ms,
                         const __sigset_t* sigmask, size_t sigsetsize) {
    int ret = set_user_sigmask(sigmask, sigsetsize);
    if (ret < 0) {
        return ret;
    }

    return do_epoll_wait(epfd, events, maxevents, timeout_ms);
}

static int epoll_close(struct shim_handle* epoll_handle) {
    assert(epoll_handle->type == TYPE_EPOLL);
    struct shim_epoll_handle* epoll = &epoll_handle->info.epoll;

    /*
     * This function is called only once the last reference to this epoll handle was put. This means
     * that either:
     * - all items were removed from this epoll prior to closing the last fd refering to it,
     * - all fds which were registered on this epoll were closed.
     * Otherwise some epoll item would hold a reference to this epoll handle and prevent it from
     * going down to `0`.
     */
    assert(LISTP_EMPTY(&epoll->waiters));
    assert(LISTP_EMPTY(&epoll->items));
    assert(epoll->items_count == 0);

    destroy_lock(&epoll->lock);
    return 0;
}

struct shim_fs_ops epoll_fs_ops = {
    .close = &epoll_close,
};

struct shim_fs epoll_builtin_fs = {
    .name   = "epoll",
    .fs_ops = &epoll_fs_ops,
};

/* Checkpoint list of `struct shim_epoll_item` from an epoll handle. Each checkpointed item is also
 * linked into an appropriate handle (which it refers to). */
BEGIN_CP_FUNC(epoll_items_list) {
    __UNUSED(size);
    assert(size == sizeof(struct shim_handle));

    struct shim_handle* old_handle = (struct shim_handle*)obj;
    struct shim_handle* new_handle = (struct shim_handle*)objp;
    assert(old_handle->type == TYPE_EPOLL && new_handle->type == TYPE_EPOLL);

    lock(&old_handle->info.epoll.lock);
    struct shim_epoll_item* item;
    LISTP_FOR_EACH_ENTRY(item, &old_handle->info.epoll.items, epoll_list) {
        size_t off = ADD_CP_OFFSET(sizeof(struct shim_epoll_item));
        struct shim_epoll_item* new_item = (struct shim_epoll_item*)(base + off);

        new_item->epoll_handle = new_handle;
        new_item->fd = item->fd;
        new_item->events = item->events;
        new_item->data = item->data;
        REF_SET(new_item->ref_count, 0);

        LISTP_ADD(new_item, &new_handle->info.epoll.items, epoll_list);
        new_handle->info.epoll.items_count++;

        DO_CP(handle, item->handle, &new_item->handle);

        LISTP_ADD(new_item, &new_item->handle->epoll_items, handle_list);
        new_item->handle->epoll_items_count++;
    }
    unlock(&old_handle->info.epoll.lock);

    ADD_CP_FUNC_ENTRY((uintptr_t)objp - base);
}
END_CP_FUNC(epoll_items_list)

BEGIN_RS_FUNC(epoll_items_list) {
    __UNUSED(offset);
    struct shim_handle* new_handle = (void*)(base + GET_CP_FUNC_ENTRY());
    assert(new_handle->type == TYPE_EPOLL);

    CP_REBASE(new_handle->info.epoll.items);

    struct shim_epoll_item* item;
    LISTP_FOR_EACH_ENTRY(item, &new_handle->info.epoll.items, epoll_list) {
        CP_REBASE(item->epoll_handle);
        get_handle(item->epoll_handle);
        assert(item->epoll_handle == new_handle);

        CP_REBASE(item->handle);
        get_handle(item->handle);

        CP_REBASE(item->epoll_list);
        get_epoll_item(item);

        CP_REBASE(item->handle_list);
        if (!LIST_EMPTY(item, handle_list)) {
            get_epoll_item(item);
        }
    }
}
END_RS_FUNC(epoll_items_list)
