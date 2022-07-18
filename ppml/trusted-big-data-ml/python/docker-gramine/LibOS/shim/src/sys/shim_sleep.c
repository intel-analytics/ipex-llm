/* SPDX-License-Identifier: LGPL-3.0-or-later */
/* Copyright (C) 2021 Intel Corporation
 *                    Borys Popławski <borysp@invisiblethingslab.com>
 */

#include "api.h"
#include "cpu.h"
#include "pal.h"
#include "shim_internal.h"
#include "shim_signal.h"
#include "shim_table.h"
#include "shim_thread.h"
#include "shim_utils.h"

long shim_do_pause(void) {
    thread_prepare_wait();
    while (!have_pending_signals()) {
        int ret = thread_wait(/*timeout_us=*/NULL, /*ignore_pending_signals=*/false);
        __UNUSED(ret);
        assert(ret == 0 || ret == -EINTR);
    }
    return -ERESTARTNOHAND;
}

int do_nanosleep(uint64_t timeout_us, struct __kernel_timespec* rem) {
    int ret = -EINTR;
    thread_prepare_wait();
    while (!have_pending_signals()) {
        ret = thread_wait(&timeout_us, /*ignore_pending_signals=*/false);
        if (ret == -ETIMEDOUT) {
            ret = 0;
            break;
        }
        ret = -EINTR;
    }
    /*
     * If `have_pending_signals` spotted a signal, we just pray it was targeted directly at this
     * thread or no other thread handles it first.
     * Ideally, we could have something like the Linux kernel: restart block, which holds a pointer
     * to a function to be called instead of restarting the syscall.
     */

    if (rem) {
        rem->tv_sec = timeout_us / TIME_US_IN_S;
        rem->tv_nsec = (timeout_us % TIME_US_IN_S) * TIME_NS_IN_US;
    }

    return ret;
}

static int check_params(struct __kernel_timespec* req, struct __kernel_timespec* rem) {
    if (!is_user_memory_readable(req, sizeof(*req))) {
        return -EFAULT;
    }
    if (rem && !is_user_memory_writable(rem, sizeof(*rem))) {
        return -EFAULT;
    }

    if (req->tv_sec < 0 || req->tv_nsec < 0 || (uint64_t)req->tv_nsec >= TIME_NS_IN_S) {
        return -EINVAL;
    }

    return 0;
}

long shim_do_nanosleep(struct __kernel_timespec* req, struct __kernel_timespec* rem) {
    int ret = check_params(req, rem);
    if (ret < 0) {
        return ret;
    }
    return do_nanosleep(timespec_to_us(req), rem);;
}

long shim_do_clock_nanosleep(clockid_t clock_id, int flags, struct __kernel_timespec* req,
                             struct __kernel_timespec* rem) {
    /* In Gramine all clocks are the same. */
    if (clock_id < 0 || clock_id >= MAX_CLOCKS) {
        return -EINVAL;
    }

    /* Linux supports only CLOCK_REALTIME, CLOCK_MONOTONIC, CLOCK_PROCESS_CPUTIME_ID;
     * see https://elixir.bootlin.com/linux/v5.16/source/kernel/time/posix-timers.c#L1255 */
    if (clock_id != CLOCK_REALTIME && clock_id != CLOCK_MONOTONIC &&
            clock_id != CLOCK_PROCESS_CPUTIME_ID) {
        return -EOPNOTSUPP;
    }

    int ret = check_params(req, rem);
    if (ret < 0) {
        return ret;
    }

    if (clock_id == CLOCK_PROCESS_CPUTIME_ID) {
        if (FIRST_TIME()) {
            log_warning("Per-process CPU-time clock is not supported in clock_nanosleep(); "
                        "it is replaced with system-wide real-time clock.");
        }
    }

    uint64_t timeout_us = timespec_to_us(req);
    if (flags & TIMER_ABSTIME) {
        uint64_t current_time = 0;
        ret = DkSystemTimeQuery(&current_time);
        if (ret < 0) {
            ret = pal_to_unix_errno(ret);
            log_error("clock_nanosleep: DkSystemTimeQuery failed with: %d", ret);
            die_or_inf_loop();
        }
        if (timeout_us <= current_time) {
            /* We timed out even before reaching this point. */
            return 0;
        }
        timeout_us -= current_time;
        rem = NULL;
    }

    return do_nanosleep(timeout_us, rem);
}
