/* SPDX-License-Identifier: LGPL-3.0-or-later */
/* Copyright (C) 2014 Stony Brook University */

/*
 * This file contains APIs to create, exit and yield a thread.
 */

#include <stddef.h> /* needed by <linux/signal.h> for size_t */
#include <errno.h>
#include <linux/sched.h>
#include <linux/signal.h>
#include <stdnoreturn.h>

#include "api.h"
#include "asan.h"
#include "pal.h"
#include "pal_error.h"
#include "pal_internal.h"
#include "pal_linux.h"
#include "pal_linux_defs.h"
#include "spinlock.h"

/* Linux PAL cannot use mmap/unmap to manage thread stacks because this may overlap with
 * g_pal_public_state.user_address_{start,end}. Linux PAL also cannot just use malloc/free because
 * DkThreadExit needs to use raw system calls and inline asm. Thus, we resort to recycling thread
 * stacks allocated by previous threads and not used anymore. This still leaks memory but at least
 * it is bounded by the maximum number of simultaneously executing threads. Note that main thread
 * is not a part of this mechanism (it only allocates a tiny altstack). */
struct thread_stack_map_t {
    void* stack;
    bool used;
};

static struct thread_stack_map_t* g_thread_stack_map = NULL;
static size_t g_thread_stack_num  = 0;
static size_t g_thread_stack_size = 0;
static spinlock_t g_thread_stack_lock = INIT_SPINLOCK_UNLOCKED;

static void* get_thread_stack(void) {
    void* ret = NULL;
    spinlock_lock(&g_thread_stack_lock);
    for (size_t i = 0; i < g_thread_stack_num; i++) {
        if (!g_thread_stack_map[i].used) {
            /* found allocated and unused stack -- use it */
            g_thread_stack_map[i].used = true;
            ret = g_thread_stack_map[i].stack;
            goto out;
        }
    }

    if (g_thread_stack_num == g_thread_stack_size) {
        /* realloc g_thread_stack_map to accommodate more objects (includes the very first time) */
        g_thread_stack_size += 8;
        struct thread_stack_map_t* tmp = malloc(g_thread_stack_size * sizeof(*tmp));
        if (!tmp)
            goto out;

        memcpy(tmp, g_thread_stack_map, g_thread_stack_num * sizeof(*tmp));
        free(g_thread_stack_map);
        g_thread_stack_map = tmp;
    }

    ret = malloc(THREAD_STACK_SIZE + ALT_STACK_SIZE);
    if (!ret)
        goto out;

    g_thread_stack_map[g_thread_stack_num].stack = ret;
    g_thread_stack_map[g_thread_stack_num].used  = true;
    g_thread_stack_num++;
out:
    spinlock_unlock(&g_thread_stack_lock);
    return ret;
}

/* Initialization wrapper of a newly-created thread (including the first thread). This function
 * accepts a TCB pointer to be set to the GS register (on x86-64) of the thread. The rest of the TCB
 * is used as the alternate stack for signal handling. Since Gramine uses GCC's stack protector,
 * and this function modifies the stack protector's GS register, we disable stack protector here. */
__attribute_no_stack_protector
int pal_thread_init(void* tcbptr) {
    PAL_TCB_LINUX* tcb = tcbptr;
    int ret;

    /* we inherited the parent's GS register which we shouldn't use in the child thread, but GCC's
     * stack protector will look for a canary at gs:[0x8] in functions called below (e.g.,
     * _DkRandomBitsRead), so let's install a default canary in the child's TCB */
    pal_tcb_set_stack_canary(&tcb->common, STACK_PROTECTOR_CANARY_DEFAULT);
    ret = pal_set_tcb(&tcb->common);
    if (ret < 0)
        return ret;

    /* each newly-created thread (including the first thread) has its own random stack canary */
    uint64_t stack_protector_canary;
    ret = _DkRandomBitsRead(&stack_protector_canary, sizeof(stack_protector_canary));
    if (ret < 0)
        return -EPERM;

    pal_tcb_set_stack_canary(&tcb->common, stack_protector_canary);

    if (tcb->alt_stack) {
        stack_t ss = {
            .ss_sp    = tcb->alt_stack,
            .ss_flags = 0,
            .ss_size  = ALT_STACK_SIZE - sizeof(*tcb),
        };

        ret = DO_SYSCALL(sigaltstack, &ss, NULL);
        if (ret < 0)
            return ret;
    }

    if (tcb->callback)
        return (*tcb->callback)(tcb->param);

    return 0;
}

static noreturn void pal_thread_exit_wrapper(int ret_val) {
    __UNUSED(ret_val);
    _DkThreadExit(/*clear_child_tid=*/NULL);
}

int _DkThreadCreate(PAL_HANDLE* handle, int (*callback)(void*), void* param) {
    int ret = 0;
    PAL_HANDLE hdl = NULL;
    void* stack = get_thread_stack();
    if (!stack) {
        ret = -PAL_ERROR_NOMEM;
        goto err;
    }

    /* Stack layout for the new thread looks like this (recall that stacks grow towards lower
     * addresses on Linux on x86-64):
     *
     *       stack +--> +-------------------+
     *                  |  child stack      | THREAD_STACK_SIZE
     * child_stack +--> +-------------------+
     *                  |  alternate stack  | ALT_STACK_SIZE - sizeof(PAL_TCB_LINUX)
     *         tcb +--> +-------------------+
     *                  |  PAL TCB          | sizeof(PAL_TCB_LINUX)
     *                  +-------------------+
     *
     * We zero out only the first page of the main stack (to comply with the requirement of
     * gcc ABI, in particular that the initial stack frame's return address must be NULL).
     * We zero out the whole altstack (since it is small anyway) and also the PAL TCB. */
    memset(stack + THREAD_STACK_SIZE - PRESET_PAGESIZE, 0, PRESET_PAGESIZE);
    memset(stack + THREAD_STACK_SIZE, 0, ALT_STACK_SIZE);

    void* child_stack = stack + THREAD_STACK_SIZE;

    hdl = calloc(1, HANDLE_SIZE(thread));
    if (!hdl) {
        ret = -PAL_ERROR_NOMEM;
        goto err;
    }
    init_handle_hdr(hdl, PAL_TYPE_THREAD);

    // Initialize TCB at the top of the alternative stack.
    PAL_TCB_LINUX* tcb = child_stack + ALT_STACK_SIZE - sizeof(PAL_TCB_LINUX);
    pal_tcb_linux_init(tcb, hdl, child_stack, callback, param);

    /* align child_stack to 16 */
    child_stack = ALIGN_DOWN_PTR(child_stack, 16);

    // TODO: pal_thread_init() may fail during initialization, we should check its result (but this
    // happens asynchronously, so it's not trivial to do).
    ret = clone(pal_thread_init, child_stack,
                CLONE_VM | CLONE_FS | CLONE_FILES | CLONE_SYSVSEM | CLONE_THREAD | CLONE_SIGHAND
                | CLONE_PARENT_SETTID,
                tcb, &hdl->thread.tid, /*tls=*/NULL, /*child_tid=*/NULL, pal_thread_exit_wrapper);

    if (ret < 0) {
        ret = -PAL_ERROR_DENIED;
        goto err;
    }

    hdl->thread.stack = stack;
    *handle = hdl;
    return 0;
err:
    free(stack);
    free(hdl);
    return ret;
}

/* PAL call DkThreadYieldExecution. Yield the execution
   of the current thread. */
void _DkThreadYieldExecution(void) {
    DO_SYSCALL(sched_yield);
}

/* _DkThreadExit for internal use: Thread exiting */
__attribute_no_sanitize_address
noreturn void _DkThreadExit(int* clear_child_tid) {
    PAL_TCB_LINUX* tcb = get_tcb_linux();
    PAL_HANDLE handle = tcb->handle;
    assert(handle);

    block_async_signals(true);
    if (tcb->alt_stack) {
        stack_t ss;
        ss.ss_sp    = NULL;
        ss.ss_flags = SS_DISABLE;
        ss.ss_size  = 0;

        /* take precautions to unset alternate stack; note that we cannot unset the TCB because
         * GCC's stack protector still uses the GS register until the end of this function */
        DO_SYSCALL(sigaltstack, &ss, NULL);
    }

    /* we do not free thread stack but instead mark it as recycled, see get_thread_stack() */
    spinlock_lock(&g_thread_stack_lock);
    for (size_t i = 0; i < g_thread_stack_num; i++) {
        if (g_thread_stack_map[i].stack == handle->thread.stack) {
            g_thread_stack_map[i].used = false;
            break;
        }
    }

    /* we might still be using the stack we just marked as unused until we enter the asm mode,
     * so we do not unlock now but rather in asm below */

#ifdef ASAN
    asan_unpoison_region((uintptr_t)handle->thread.stack, THREAD_STACK_SIZE + ALT_STACK_SIZE);
#endif

    /*
     * To make sure the compiler doesn't touch the stack after it was freed, we need to use asm:
     *   1. Unlock g_thread_stack_lock (so that other threads can start re-using this stack)
     *   2. Set *clear_child_tid = 0 if clear_child_tid != NULL
     *      (we thus inform LibOS, where async worker thread is waiting on this to wake up parent)
     *   3. Exit thread
     * All of these happen in `_DkThreadExit_asm_stub`.
     */
    static_assert(sizeof(g_thread_stack_lock.lock) == 4,
                  "unexpected g_thread_stack_lock.lock size");
    static_assert(offsetof(__typeof__(g_thread_stack_lock), lock) % 4 == 0,
                  "lock is not naturally aligned in g_thread_stack_lock");
    static_assert(sizeof(*clear_child_tid) == 4, "unexpected clear_child_tid size");

    _DkThreadExit_asm_stub(&g_thread_stack_lock.lock, clear_child_tid);
}

int _DkThreadResume(PAL_HANDLE thread_handle) {
    int ret = DO_SYSCALL(tgkill, g_pal_linux_state.host_pid, thread_handle->thread.tid, SIGCONT);

    if (ret < 0)
        return -PAL_ERROR_DENIED;

    return 0;
}

int _DkThreadSetCpuAffinity(PAL_HANDLE thread, PAL_NUM cpumask_size, unsigned long* cpu_mask) {
    int ret = DO_SYSCALL(sched_setaffinity, thread->thread.tid, cpumask_size, cpu_mask);

    return ret < 0 ? unix_to_pal_error(ret) : ret;
}

int _DkThreadGetCpuAffinity(PAL_HANDLE thread, PAL_NUM cpumask_size, unsigned long* cpu_mask) {
    int ret = DO_SYSCALL(sched_getaffinity, thread->thread.tid, cpumask_size, cpu_mask);

    return ret < 0 ? unix_to_pal_error(ret) : ret;
}

struct handle_ops g_thread_ops = {
    /* nothing */
};
