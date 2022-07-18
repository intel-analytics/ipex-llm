/* SPDX-License-Identifier: LGPL-3.0-or-later */
/* Copyright (C) 2014 Stony Brook University */

/*
 * Implementation of system calls "sched_yield", "setpriority", "getpriority", "sched_setparam",
 * "sched_getparam", "sched_setscheduler", "sched_getscheduler", "sched_get_priority_max",
 * "sched_get_priority_min", "sched_rr_get_interval", "sched_setaffinity", "sched_getaffinity",
 * "getcpu".
 */

#include <errno.h>
#include <linux/resource.h>
#include <linux/sched.h>

#include "api.h"
#include "pal.h"
#include "shim_internal.h"
#include "shim_table.h"
#include "shim_thread.h"

long shim_do_sched_yield(void) {
    DkThreadYieldExecution();
    return 0;
}

/* dummy implementation: ignore user-supplied niceval and return success */
long shim_do_setpriority(int which, int who, int niceval) {
    __UNUSED(who);

    if (which != PRIO_PROCESS && which != PRIO_PGRP && which != PRIO_USER)
        return -EINVAL;

    if (niceval < 1 || niceval > 40)
        return -EACCES;

    return 0;
}

/* dummy implementation: always return the default nice value of 20 */
long shim_do_getpriority(int which, int who) {
    __UNUSED(who);

    if (which != PRIO_PROCESS && which != PRIO_PGRP && which != PRIO_USER)
        return -EINVAL;

    return 20; /* default nice value on Linux */
}

/* dummy implementation: ignore user-supplied param and return success */
long shim_do_sched_setparam(pid_t pid, struct __kernel_sched_param* param) {
    if (pid < 0 || param == NULL)
        return -EINVAL;

    return 0;
}

/* dummy implementation: always return sched_priority of 0 (implies non-real-time sched policy) */
long shim_do_sched_getparam(pid_t pid, struct __kernel_sched_param* param) {
    if (pid < 0 || param == NULL)
        return -EINVAL;

    param->__sched_priority = 0;
    return 0;
}

/* dummy implementation: ignore user-supplied policy & param and return success */
long shim_do_sched_setscheduler(pid_t pid, int policy, struct __kernel_sched_param* param) {
    policy &= ~SCHED_RESET_ON_FORK; /* ignore reset-on-fork flag */

    if (pid < 0 || param == NULL)
        return -EINVAL;

    /* fail on unrecognized policies */
    if (policy != SCHED_NORMAL && policy != SCHED_BATCH &&
            policy != SCHED_IDLE && /* non-real-time */
            policy != SCHED_FIFO && policy != SCHED_RR /* real-time */)
        return -EINVAL;

    /* non-real-time policies must have priority of 0 */
    if ((policy == SCHED_NORMAL || policy == SCHED_BATCH || policy == SCHED_IDLE) &&
            (param->__sched_priority != 0))
        return -EINVAL;

    /* real-time policies must have priority in range [1, 99] */
    if ((policy == SCHED_FIFO || policy == SCHED_RR) &&
            (param->__sched_priority < 1 || param->__sched_priority > 99))
        return -EINVAL;

    return 0;
}

/* dummy implementation: always return SCHED_NORMAL (default round-robin time-sharing policy) */
long shim_do_sched_getscheduler(pid_t pid) {
    if (pid < 0)
        return -EINVAL;

    return SCHED_NORMAL;
}

long shim_do_sched_get_priority_max(int policy) {
    /* fail on unrecognized policies */
    if (policy != SCHED_NORMAL && policy != SCHED_BATCH &&
            policy != SCHED_IDLE && /* non-real-time */
            policy != SCHED_FIFO && policy != SCHED_RR /* real-time */)
        return -EINVAL;

    /* real-time policies have max priority of 99 */
    if (policy == SCHED_FIFO || policy == SCHED_RR)
        return 99;

    /* non-real-time policies have max priority of 0 */
    return 0;
}

long shim_do_sched_get_priority_min(int policy) {
    /* fail on unrecognized policies */
    if (policy != SCHED_NORMAL && policy != SCHED_BATCH &&
            policy != SCHED_IDLE && /* non-real-time */
            policy != SCHED_FIFO && policy != SCHED_RR /* real-time */)
        return -EINVAL;

    /* real-time policies have min priority of 1 */
    if (policy == SCHED_FIFO || policy == SCHED_RR)
        return 1;

    /* non-real-time policies have min priority of 0 */
    return 0;
}

/* dummy implementation: always return 100 ms (default in Linux) */
long shim_do_sched_rr_get_interval(pid_t pid, struct timespec* interval) {
    if (pid < 0)
        return -EINVAL;

    if (!is_user_memory_writable(interval, sizeof(*interval)))
        return -EFAULT;

    interval->tv_sec  = 0;
    interval->tv_nsec = 100000000; /* default value of 100 ms in Linux */
    return 0;
}

long shim_do_sched_setaffinity(pid_t pid, unsigned int cpumask_size, unsigned long* user_mask_ptr) {
    int ret;

    /* check if user_mask_ptr is valid */
    if (!is_user_memory_readable(user_mask_ptr, cpumask_size))
        return -EFAULT;

    struct shim_thread* thread = pid ? lookup_thread(pid) : get_cur_thread();
    if (!thread)
        return -ESRCH;

    /* lookup_thread() internally increments thread count; do the same in case of
       get_cur_thread(). */
    if (pid == 0)
        get_thread(thread);

    /* Internal Gramine threads are not affinitized; if we hit an internal thread here, this is
       some bug in user app. */
    if (is_internal(thread)) {
        put_thread(thread);
        return -ESRCH;
    }

    ret = DkThreadSetCpuAffinity(thread->pal_handle, cpumask_size, user_mask_ptr);
    if (ret < 0) {
        put_thread(thread);
        return pal_to_unix_errno(ret);
    }

    put_thread(thread);
    return 0;
}

long shim_do_sched_getaffinity(pid_t pid, unsigned int cpumask_size, unsigned long* user_mask_ptr) {
    int ret;
    size_t threads_cnt = g_pal_public_state->topo_info.threads_cnt;

    /* Check if user_mask_ptr is valid */
    if (!is_user_memory_writable(user_mask_ptr, cpumask_size))
        return -EFAULT;

    /* Linux kernel bitmap is based on long. So according to its implementation, round up the result
     * to sizeof(long) */
    size_t bitmask_size_in_bytes = BITS_TO_LONGS(threads_cnt) * sizeof(long);
    if (cpumask_size < bitmask_size_in_bytes) {
        log_warning("size of cpumask must be at least %lu but supplied cpumask is %u",
                    bitmask_size_in_bytes, cpumask_size);
        return -EINVAL;
    }

    /* Linux kernel also rejects non-natural size */
    if (cpumask_size & (sizeof(long) - 1))
        return -EINVAL;

    struct shim_thread* thread = pid ? lookup_thread(pid) : get_cur_thread();
    if (!thread)
        return -ESRCH;

    /* lookup_thread() internally increments thread count; do the same in case of
       get_cur_thread(). */
    if (pid == 0)
        get_thread(thread);

    /* Internal Gramine threads are not affinitized; if we hit an internal thread here, this is
       some bug in user app. */
    if (is_internal(thread)) {
        put_thread(thread);
        return -ESRCH;
    }

    memset(user_mask_ptr, 0, cpumask_size);
    ret = DkThreadGetCpuAffinity(thread->pal_handle, bitmask_size_in_bytes, user_mask_ptr);
    if (ret < 0) {
        put_thread(thread);
        return pal_to_unix_errno(ret);
    }

    put_thread(thread);
    /* on success, imitate Linux kernel implementation: see SYSCALL_DEFINE3(sched_getaffinity) */
    return bitmask_size_in_bytes;
}

/* dummy implementation: always return cpu0  */
long shim_do_getcpu(unsigned* cpu, unsigned* node, struct getcpu_cache* unused) {
    __UNUSED(unused);

    if (cpu) {
        if (!is_user_memory_writable(cpu, sizeof(*cpu))) {
            return -EFAULT;
        }
        *cpu = 0;
    }

    if (node) {
        if (!is_user_memory_writable(node, sizeof(*node))) {
            return -EFAULT;
        }
        *node = 0;
    }

    return 0;
}
