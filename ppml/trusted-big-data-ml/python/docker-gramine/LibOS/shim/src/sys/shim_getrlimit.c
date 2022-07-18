/* SPDX-License-Identifier: LGPL-3.0-or-later */
/* Copyright (C) 2014 Stony Brook University */

/*
 * Implementation of system calls "getrlimit", "setrlimit" and "sysinfo".
 */

#include <asm/resource.h>
#include <linux/sysinfo.h>

#include "shim_checkpoint.h"
#include "shim_internal.h"
#include "shim_lock.h"
#include "shim_process.h"
#include "shim_table.h"
#include "shim_thread.h"

/*
 * TODO: implement actual limitation on each resource.
 *
 * The current behavor(i.e. sys_stack_size, brk_max_size) may be subject
 * to be fixed.
 */

#define MAX_THREADS     (0x3fffffff / 2)
#define DEFAULT_MAX_FDS 900   /* We have to keep this lower than the standard 1024, otherwise we'll
                                 hit the limit on the host sooner than the app would reach this
                                 value (because Gramine-internal fds in the PAL also counts towards
                                 the host limit). Ideally, we should have a PAL API which tells
                                 LibOS how many PAL handles it can use simultaneously. */
#define MAX_MAX_FDS     65536
#define MLOCK_LIMIT     (64 * 1024)
#define MQ_BYTES_MAX    819200

static struct __kernel_rlimit64 __rlim[RLIM_NLIMITS] __attribute_migratable = {
    [RLIMIT_CPU]     = {RLIM_INFINITY, RLIM_INFINITY},
    [RLIMIT_FSIZE]   = {RLIM_INFINITY, RLIM_INFINITY},
    [RLIMIT_DATA]    = {RLIM_INFINITY, RLIM_INFINITY},
    [RLIMIT_STACK]   = {DEFAULT_SYS_STACK_SIZE, RLIM_INFINITY},
    [RLIMIT_CORE]    = {0, RLIM_INFINITY},
    [RLIMIT_RSS]     = {RLIM_INFINITY, RLIM_INFINITY},
    [RLIMIT_NPROC]   = {MAX_THREADS, MAX_THREADS},
    [RLIMIT_NOFILE]  = {DEFAULT_MAX_FDS, MAX_MAX_FDS},
    [RLIMIT_MEMLOCK] = {MLOCK_LIMIT, MLOCK_LIMIT},
    [RLIMIT_AS]      = {RLIM_INFINITY, RLIM_INFINITY},
    [RLIMIT_LOCKS]   = {RLIM_INFINITY, RLIM_INFINITY},
    /* [RLIMIT_SIGPENDING] = [RLIMIT_NPROC] for initial value */
    [RLIMIT_SIGPENDING] = {MAX_THREADS, MAX_THREADS},
    [RLIMIT_MSGQUEUE]   = {MQ_BYTES_MAX, MQ_BYTES_MAX},
    [RLIMIT_NICE]       = {0, 0},
    [RLIMIT_RTPRIO]     = {0, 0},
    [RLIMIT_RTTIME]     = {RLIM_INFINITY, RLIM_INFINITY},
};

static struct shim_lock rlimit_lock;

int init_rlimit(void) {
    if (!create_lock(&rlimit_lock)) {
        return -ENOMEM;
    }
    return 0;
}

uint64_t get_rlimit_cur(int resource) {
    assert(resource >= 0 && RLIM_NLIMITS > resource);
    lock(&rlimit_lock);
    uint64_t rlim = __rlim[resource].rlim_cur;
    unlock(&rlimit_lock);
    return rlim;
}

void set_rlimit_cur(int resource, uint64_t rlim) {
    assert(resource >= 0 && RLIM_NLIMITS > resource);
    lock(&rlimit_lock);
    __rlim[resource].rlim_cur = rlim;
    unlock(&rlimit_lock);
}

long shim_do_getrlimit(int resource, struct __kernel_rlimit* rlim) {
    if (resource < 0 || RLIM_NLIMITS <= resource)
        return -EINVAL;
    if (!is_user_memory_writable(rlim, sizeof(*rlim)))
        return -EFAULT;

    lock(&rlimit_lock);
    rlim->rlim_cur = __rlim[resource].rlim_cur;
    rlim->rlim_max = __rlim[resource].rlim_max;
    unlock(&rlimit_lock);
    return 0;
}

long shim_do_setrlimit(int resource, struct __kernel_rlimit* rlim) {
    struct shim_thread* cur_thread = get_cur_thread();
    assert(cur_thread);

    if (resource < 0 || RLIM_NLIMITS <= resource)
        return -EINVAL;
    if (!is_user_memory_readable(rlim, sizeof(*rlim)))
        return -EFAULT;
    if (rlim->rlim_cur > rlim->rlim_max)
        return -EINVAL;

    lock(&rlimit_lock);
    if (rlim->rlim_max > __rlim[resource].rlim_max && cur_thread->euid) {
        unlock(&rlimit_lock);
        return -EPERM;
    }

    __rlim[resource].rlim_cur = rlim->rlim_cur;
    __rlim[resource].rlim_max = rlim->rlim_max;
    unlock(&rlimit_lock);
    return 0;
}

long shim_do_prlimit64(pid_t pid, int resource, const struct __kernel_rlimit64* new_rlim,
                       struct __kernel_rlimit64* old_rlim) {
    struct shim_thread* cur_thread = get_cur_thread();
    assert(cur_thread);
    int ret = 0;

    // XXX: Do not support setting/getting the rlimit of other processes yet.
    if (pid && (IDTYPE)pid != g_process.pid)
        return -ENOSYS;

    if (resource < 0 || RLIM_NLIMITS <= resource)
        return -EINVAL;

    if (old_rlim) {
        if (!is_user_memory_writable(old_rlim, sizeof(*old_rlim)))
            return -EFAULT;
    }

    if (new_rlim) {
        if (!is_user_memory_readable((void*)new_rlim, sizeof(*new_rlim))) {
            ret = -EFAULT;
            goto out;
        }
        if (new_rlim->rlim_cur > new_rlim->rlim_max) {
            ret = -EINVAL;
            goto out;
        }
    }

    lock(&rlimit_lock);

    if (new_rlim) {
        if (new_rlim->rlim_max > __rlim[resource].rlim_max && cur_thread->euid) {
            ret = -EPERM;
            goto out;
        }
    }

    if (old_rlim)
        *old_rlim = __rlim[resource];
    if (new_rlim)
        __rlim[resource] = *new_rlim;

out:
    unlock(&rlimit_lock);
    return ret;
}

/* minimal implementation; tested apps only care about total/free RAM */
long shim_do_sysinfo(struct sysinfo* info) {
    if (!is_user_memory_writable(info, sizeof(*info)))
        return -EFAULT;

    memset(info, 0, sizeof(*info));
    info->totalram  = g_pal_public_state->mem_total;
    info->totalhigh = g_pal_public_state->mem_total;
    info->freeram   = DkMemoryAvailableQuota();
    info->freehigh  = DkMemoryAvailableQuota();
    info->mem_unit  = 1;
    info->procs     = 1; /* report only this Gramine process */
    return 0;
}
