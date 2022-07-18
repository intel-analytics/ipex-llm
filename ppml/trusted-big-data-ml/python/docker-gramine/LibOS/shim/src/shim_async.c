/* SPDX-License-Identifier: LGPL-3.0-or-later */
/* Copyright (C) 2014 Stony Brook University */

/*
 * This file contains functions to add asyncronous events triggered by timer.
 */

#include "list.h"
#include "pal.h"
#include "shim_internal.h"
#include "shim_lock.h"
#include "shim_pollable_event.h"
#include "shim_thread.h"
#include "shim_utils.h"

#define IDLE_SLEEP_TIME 1000000
#define MAX_IDLE_CYCLES 10000

DEFINE_LIST(async_event);
struct async_event {
    IDTYPE caller; /* thread installing this event */
    LIST_TYPE(async_event) list;
    LIST_TYPE(async_event) triggered_list;
    void (*callback)(IDTYPE caller, void* arg);
    void* arg;
    PAL_HANDLE object;    /* handle (async IO) to wait on */
    uint64_t expire_time; /* alarm/timer to wait on */
};
DEFINE_LISTP(async_event);
static LISTP_TYPE(async_event) async_list;

/* Should be accessed with async_worker_lock held. */
static enum { WORKER_NOTALIVE, WORKER_ALIVE } async_worker_state;

static struct shim_thread* async_worker_thread;
static struct shim_lock async_worker_lock;

/* TODO: use async_worker_thread->pollable_event instead */
static struct shim_pollable_event install_new_event;

static int create_async_worker(void);

/* Threads register async events like alarm(), setitimer(), ioctl(FIOASYNC)
 * using this function. These events are enqueued in async_list and delivered
 * to async worker thread by triggering install_new_event. When event is
 * triggered in async worker thread, the corresponding event's callback with
 * arguments `arg` is called. This callback typically sends a signal to the
 * thread which registered the event (saved in `event->caller`).
 *
 * We distinguish between alarm/timer events and async IO events:
 *   - alarm/timer events set object = NULL and time = seconds
 *     (time = 0 cancels all pending alarms/timers).
 *   - async IO events set object = handle and time = 0.
 *
 * Function returns remaining usecs for alarm/timer events (same as alarm())
 * or 0 for async IO events. On error, it returns a negated error code.
 */
int64_t install_async_event(PAL_HANDLE object, uint64_t time,
                            void (*callback)(IDTYPE caller, void* arg), void* arg) {
    /* if event happens on object, time must be zero */
    assert(!object || (object && !time));

    uint64_t now = 0;
    int ret = DkSystemTimeQuery(&now);
    if (ret < 0) {
        return pal_to_unix_errno(ret);
    }

    uint64_t max_prev_expire_time = now;

    struct async_event* event = malloc(sizeof(struct async_event));
    if (!event) {
        return -ENOMEM;
    }

    event->callback    = callback;
    event->arg         = arg;
    event->caller      = get_cur_tid();
    event->object      = object;
    event->expire_time = time ? now + time : 0;

    lock(&async_worker_lock);

    if (callback != &cleanup_thread && !object) {
        /* This is alarm() or setitimer() emulation, treat both according to
         * alarm() syscall semantics: cancel any pending alarm/timer. */
        struct async_event* tmp;
        struct async_event* n;
        LISTP_FOR_EACH_ENTRY_SAFE(tmp, n, &async_list, list) {
            if (tmp->expire_time) {
                /* this is a pending alarm/timer, cancel it and save its expiration time */
                if (max_prev_expire_time < tmp->expire_time)
                    max_prev_expire_time = tmp->expire_time;

                LISTP_DEL(tmp, &async_list, list);
                free(tmp);
            }
        }

        if (!time) {
            /* This is alarm(0), we cancelled all pending alarms/timers
             * and user doesn't want to set a new alarm: we are done. */
            free(event);
            unlock(&async_worker_lock);
            return max_prev_expire_time - now;
        }
    }

    INIT_LIST_HEAD(event, list);
    LISTP_ADD_TAIL(event, &async_list, list);

    if (async_worker_state == WORKER_NOTALIVE) {
        int ret = create_async_worker();
        if (ret < 0) {
            unlock(&async_worker_lock);
            return ret;
        }
    }

    unlock(&async_worker_lock);

    log_debug("Installed async event at %lu", now);
    set_pollable_event(&install_new_event, 1);
    return max_prev_expire_time - now;
}

int init_async_worker(void) {
    /* early enough in init, can write global vars without the lock */
    async_worker_state = WORKER_NOTALIVE;
    if (!create_lock(&async_worker_lock)) {
        return -ENOMEM;
    }
    int ret = create_pollable_event(&install_new_event);
    if (ret < 0) {
        return ret;
    }

    /* enable locking mechanisms since we are going in multi-threaded mode */
    enable_locking();

    return 0;
}

static int shim_async_worker(void* arg) {
    struct shim_thread* self = (struct shim_thread*)arg;
    if (!arg)
        return -1;

    shim_tcb_init();
    set_cur_thread(self);

    log_setprefix(shim_get_tcb());

    lock(&async_worker_lock);
    bool notme = (self != async_worker_thread);
    unlock(&async_worker_lock);

    if (notme) {
        put_thread(self);
        DkThreadExit(/*clear_child_tid=*/NULL);
        /* UNREACHABLE */
    }

    /* Assume async worker thread will not drain the stack that PAL provides,
     * so for efficiency we don't swap the stack. */
    log_debug("Async worker thread started");

    /* Simple heuristic to not burn cycles when no async events are installed:
     * async worker thread sleeps IDLE_SLEEP_TIME for MAX_IDLE_CYCLES and
     * if nothing happens, dies. It will be re-spawned if some thread wants
     * to install a new event. */
    uint64_t idle_cycles = 0;

    /* init `pals` so that it always contains at least install_new_event */
    size_t pals_max_cnt = 32;
    PAL_HANDLE* pals = malloc(sizeof(*pals) * (1 + pals_max_cnt));
    if (!pals) {
        log_error("Allocation of pals failed");
        goto out_err;
    }

    /* allocate one memory region to hold two pal_wait_flags_t arrays: events and revents */
    pal_wait_flags_t* pal_events = malloc(sizeof(*pal_events) * (1 + pals_max_cnt) * 2);
    if (!pal_events) {
        log_error("Allocation of pal_events failed");
        goto out_err;
    }
    pal_wait_flags_t* ret_events = pal_events + 1 + pals_max_cnt;

    PAL_HANDLE install_new_event_pal = install_new_event.read_handle;
    pals[0] = install_new_event_pal;
    pal_events[0] = PAL_WAIT_READ;
    ret_events[0] = 0;

    while (true) {
        uint64_t now = 0;
        int ret = DkSystemTimeQuery(&now);
        if (ret < 0) {
            ret = pal_to_unix_errno(ret);
            log_error("DkSystemTimeQuery failed with: %d", ret);
            goto out_err;
        }

        lock(&async_worker_lock);
        if (async_worker_state != WORKER_ALIVE) {
            async_worker_thread = NULL;
            unlock(&async_worker_lock);
            break;
        }

        uint64_t next_expire_time = 0;
        size_t pals_cnt = 0;

        struct async_event* tmp;
        struct async_event* n;
        bool other_event = false;
        LISTP_FOR_EACH_ENTRY_SAFE(tmp, n, &async_list, list) {
            /* repopulate `pals` with IO events and find the next expiring alarm/timer */
            if (tmp->object) {
                if (pals_cnt == pals_max_cnt) {
                    /* grow `pals` to accommodate more objects */
                    PAL_HANDLE* tmp_pals = malloc(sizeof(*tmp_pals) * (1 + pals_max_cnt * 2));
                    if (!tmp_pals) {
                        log_error("tmp_pals allocation failed");
                        goto out_err_unlock;
                    }
                    pal_wait_flags_t* tmp_pal_events =
                        malloc(sizeof(*tmp_pal_events) * (2 + pals_max_cnt * 4));
                    if (!tmp_pal_events) {
                        log_error("tmp_pal_events allocation failed");
                        goto out_err_unlock;
                    }
                    pal_wait_flags_t* tmp_ret_events = tmp_pal_events + 1 + pals_max_cnt * 2;

                    memcpy(tmp_pals, pals, sizeof(*tmp_pals) * (1 + pals_max_cnt));
                    memcpy(tmp_pal_events, pal_events,
                           sizeof(*tmp_pal_events) * (1 + pals_max_cnt));
                    memcpy(tmp_ret_events, ret_events,
                           sizeof(*tmp_ret_events) * (1 + pals_max_cnt));

                    pals_max_cnt *= 2;

                    free(pals);
                    free(pal_events);

                    pals = tmp_pals;
                    pal_events = tmp_pal_events;
                    ret_events = tmp_ret_events;
                }

                pals[pals_cnt + 1]       = tmp->object;
                pal_events[pals_cnt + 1] = PAL_WAIT_READ;
                ret_events[pals_cnt + 1] = 0;
                pals_cnt++;
            } else if (tmp->expire_time && tmp->expire_time > now) {
                if (!next_expire_time || next_expire_time > tmp->expire_time) {
                    /* use time of the next expiring alarm/timer */
                    next_expire_time = tmp->expire_time;
                }
            } else {
                /* cleanup events do not have an object nor a timeout */
                other_event = true;
            }
        }

        uint64_t sleep_time;
        if (next_expire_time) {
            sleep_time  = next_expire_time - now;
            idle_cycles = 0;
        } else if (pals_cnt || other_event) {
            sleep_time = NO_TIMEOUT;
            idle_cycles = 0;
        } else {
            /* no async IO events and no timers/alarms: thread is idling */
            sleep_time = IDLE_SLEEP_TIME;
            idle_cycles++;
        }

        if (idle_cycles == MAX_IDLE_CYCLES) {
            async_worker_state  = WORKER_NOTALIVE;
            async_worker_thread = NULL;
            unlock(&async_worker_lock);
            log_debug("Async worker thread has been idle for some time; stopping it");
            break;
        }
        unlock(&async_worker_lock);

        /* wait on async IO events + install_new_event + next expiring alarm/timer */
        ret = DkStreamsWaitEvents(pals_cnt + 1, pals, pal_events, ret_events, &sleep_time);
        if (ret < 0 && ret != -PAL_ERROR_INTERRUPTED && ret != -PAL_ERROR_TRYAGAIN) {
            ret = pal_to_unix_errno(ret);
            log_error("DkStreamsWaitEvents failed with: %d", ret);
            goto out_err;
        }
        bool polled = ret == 0;

        ret = DkSystemTimeQuery(&now);
        if (ret < 0) {
            ret = pal_to_unix_errno(ret);
            log_error("DkSystemTimeQuery failed with: %d", ret);
            goto out_err;
        }

        LISTP_TYPE(async_event) triggered;
        INIT_LISTP(&triggered);

        /* acquire lock because we read/modify async_list below */
        lock(&async_worker_lock);

        for (size_t i = 0; polled && i < pals_cnt + 1; i++) {
            if (ret_events[i]) {
                if (pals[i] == install_new_event_pal) {
                    /* some thread wants to install new event; this event is found in async_list,
                     * so just re-init install_new_event */
                    clear_pollable_event(&install_new_event);
                    continue;
                }

                /* check if this event is an IO event found in async_list */
                LISTP_FOR_EACH_ENTRY_SAFE(tmp, n, &async_list, list) {
                    if (tmp->object == pals[i]) {
                        log_debug("Async IO event triggered at %lu", now);
                        LISTP_ADD_TAIL(tmp, &triggered, triggered_list);
                        break;
                    }
                }
            }
        }

        /* check if exit-child or alarm/timer events were triggered */
        LISTP_FOR_EACH_ENTRY_SAFE(tmp, n, &async_list, list) {
            if (tmp->callback == &cleanup_thread) {
                log_debug("Thread exited, cleaning up");
                LISTP_DEL(tmp, &async_list, list);
                LISTP_ADD_TAIL(tmp, &triggered, triggered_list);
            } else if (tmp->expire_time && tmp->expire_time <= now) {
                log_debug("Alarm/timer triggered at %lu (expired at %lu)", now, tmp->expire_time);
                LISTP_DEL(tmp, &async_list, list);
                LISTP_ADD_TAIL(tmp, &triggered, triggered_list);
            }
        }

        unlock(&async_worker_lock);

        /* call callbacks for all triggered events */
        if (!LISTP_EMPTY(&triggered)) {
            LISTP_FOR_EACH_ENTRY_SAFE(tmp, n, &triggered, triggered_list) {
                LISTP_DEL(tmp, &triggered, triggered_list);
                tmp->callback(tmp->caller, tmp->arg);
                if (!tmp->object) {
                    /* this is a one-off exit-child or alarm/timer event */
                    free(tmp);
                }
            }
        }
    }

    put_thread(self);
    log_debug("Async worker thread terminated");

    free(pals);
    free(pal_events);

    DkThreadExit(/*clear_child_tid=*/NULL);
    /* UNREACHABLE */

out_err_unlock:
    unlock(&async_worker_lock);
out_err:
    log_error("Terminating the process due to a fatal error in async worker");
    put_thread(self);
    DkProcessExit(1);
}

/* this should be called with the async_worker_lock held */
static int create_async_worker(void) {
    assert(locked(&async_worker_lock));

    if (async_worker_state == WORKER_ALIVE)
        return 0;

    struct shim_thread* new = get_new_internal_thread();
    if (!new)
        return -ENOMEM;

    async_worker_thread = new;
    async_worker_state  = WORKER_ALIVE;

    PAL_HANDLE handle = NULL;
    int ret = DkThreadCreate(shim_async_worker, new, &handle);

    if (ret < 0) {
        async_worker_thread = NULL;
        async_worker_state  = WORKER_NOTALIVE;
        put_thread(new);
        return pal_to_unix_errno(ret);
    }

    new->pal_handle = handle;
    return 0;
}

/* On success, the reference to async worker thread is returned with refcount
 * incremented. It is the responsibility of caller to wait for async worker's
 * exit and then release the final reference to free related resources (it is
 * problematic for the thread itself to release its own resources e.g. stack).
 */
struct shim_thread* terminate_async_worker(void) {
    lock(&async_worker_lock);

    if (async_worker_state != WORKER_ALIVE) {
        unlock(&async_worker_lock);
        return NULL;
    }

    struct shim_thread* ret = async_worker_thread;
    if (ret)
        get_thread(ret);
    async_worker_state = WORKER_NOTALIVE;
    unlock(&async_worker_lock);

    /* force wake up of async worker thread so that it exits */
    set_pollable_event(&install_new_event, 1);
    return ret;
}
