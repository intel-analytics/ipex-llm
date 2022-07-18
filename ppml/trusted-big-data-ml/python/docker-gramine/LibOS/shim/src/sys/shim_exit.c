/* SPDX-License-Identifier: LGPL-3.0-or-later */
/* Copyright (C) 2014 Stony Brook University
 * Copyright (C) 2020 Invisible Things Lab
 *                    Borys Popławski <borysp@invisiblethingslab.com>
 * Copyright (C) 2020 Intel Corporation
 *                    Borys Popławski <borysp@invisiblethingslab.com>
 */

#include "pal.h"
#include "shim_fs_lock.h"
#include "shim_ipc.h"
#include "shim_lock.h"
#include "shim_process.h"
#include "shim_signal.h"
#include "shim_table.h"
#include "shim_thread.h"
#include "shim_utils.h"

static noreturn void libos_clean_and_exit(int exit_code) {
    /*
     * TODO: if we are the IPC leader, we need to either:
     * 1) kill all other Gramine processes
     * 2) wait for them to exit here, before we terminate the IPC helper
     */

    shutdown_sync_client();

    struct shim_thread* async_thread = terminate_async_worker();
    if (async_thread) {
        /* TODO: wait for the thread to finish its tasks and exit in the host OS.
         * This is tracked by the following issue:
         * https://github.com/gramineproject/graphene/issues/440
         */
        put_thread(async_thread);
    }

    /*
     * At this point there should be only 2 threads running: this + IPC worker.
     * XXX: We release current thread's ID, yet we are still running. We never put the (possibly)
     * last reference to the current thread (from TCB) and there should be no other references to it
     * lying around, so nothing bad should happen™. Hopefully...
     */
    /*
     * We might still be a zombie in the parent process. In an unlikely case that the parent does
     * not wait for us for a long time and pids overflow (currently we can have 2**32 pids), IPC
     * leader could give this ID to somebody else. This could be a nasty conflict.
     * The problem is that solving this is hard: we would need to make the parent own (or at least
     * release) our pid, but that would require "reparenting" in case the parent dies before us.
     * Such solution would also have some nasty consequences: Gramine pid 1 (which I guess would
     * be the new parent) might not be expecting to have more children than it spawned (normal apps
     * do not expect that, init process is pretty special).
     */
    release_id(get_cur_thread()->tid);

    terminate_ipc_worker();

    log_debug("process %u exited with status %d", g_process_ipc_ids.self_vmid, exit_code);

    /* TODO: We exit whole libos, but there are some objects that might need cleanup - we should do
     * a proper cleanup of everything. */
    DkProcessExit(exit_code);
}

noreturn void thread_exit(int error_code, int term_signal) {
    /* Remove current thread from the threads list. */
    if (!check_last_thread(/*mark_self_dead=*/true)) {
        struct shim_thread* cur_thread = get_cur_thread();

        /* ask async worker thread to cleanup this thread */
        cur_thread->clear_child_tid_pal = 1; /* any non-zero value suffices */
        /* We pass this ownership to `cleanup_thread`. */
        get_thread(cur_thread);
        int64_t ret = install_async_event(NULL, 0, &cleanup_thread, cur_thread);

        /* Take the reference to the current thread from the tcb. */
        lock(&cur_thread->lock);
        assert(cur_thread->shim_tcb->tp == cur_thread);
        cur_thread->shim_tcb->tp = NULL;
        unlock(&cur_thread->lock);
        put_thread(cur_thread);

        if (ret < 0) {
            log_error("failed to set up async cleanup_thread (exiting without clear child tid),"
                      " return code: %ld", ret);
            /* `cleanup_thread` did not get this reference, clean it. We have to be careful, as
             * this is most likely the last reference and will free this `cur_thread`. */
            put_thread(cur_thread);
            DkThreadExit(NULL);
            /* UNREACHABLE */
        }

        DkThreadExit(&cur_thread->clear_child_tid_pal);
        /* UNREACHABLE */
    }

    /* Clear POSIX locks before we notify parent: after a successful `wait()` by parent, our locks
     * should already be gone. */
    int ret = posix_lock_clear_pid(g_process.pid);
    if (ret < 0)
        log_warning("error clearing POSIX locks: %d", ret);

    /* This is the last thread of the process. Let parent know we exited. */
    ret = ipc_cld_exit_send(error_code, term_signal);
    if (ret < 0) {
        log_error("Sending IPC process-exit notification failed: %d", ret);
    }

    /* At this point other threads might be still in the middle of an exit routine, but we don't
     * care since the below will call `exit_group` eventually. */
    libos_clean_and_exit(term_signal ? 128 + (term_signal & ~__WCOREDUMP_BIT) : error_code);
}

static int mark_thread_to_die(struct shim_thread* thread, void* arg) {
    if (thread == (struct shim_thread*)arg) {
        return 0;
    }

    bool need_wakeup = !__atomic_exchange_n(&thread->time_to_die, true, __ATOMIC_ACQ_REL);

    /* Now let's kick `thread`, so that it notices (in `handle_signal`) the flag `time_to_die`
     * set above (but only if we really set that flag). */
    if (need_wakeup) {
        thread_wakeup(thread);
        (void)DkThreadResume(thread->pal_handle); // There is nothing we can do on errors.
    }
    return 1;
}

bool kill_other_threads(void) {
    bool killed = false;
    /* Tell other threads to exit. Since `mark_thread_to_die` never returns an error, this call
     * cannot fail. */
    if (walk_thread_list(mark_thread_to_die, get_cur_thread(), /*one_shot=*/false) != -ESRCH) {
        killed = true;
    }
    DkThreadYieldExecution();

    /* Wait for all other threads to exit. */
    while (!check_last_thread(/*mark_self_dead=*/false)) {
        /* Tell other threads to exit again - the previous announcement could have been missed by
         * threads that were just being created. */
        if (walk_thread_list(mark_thread_to_die, get_cur_thread(), /*one_shot=*/false) != -ESRCH) {
            killed = true;
        }
        DkThreadYieldExecution();
    }

    return killed;
}

noreturn void process_exit(int error_code, int term_signal) {
    assert(!is_internal(get_cur_thread()));

    /* If process_exit is invoked multiple times, only a single invocation proceeds past this
     * point. */
    if (!FIRST_TIME()) {
        /* Just exit current thread. */
        thread_exit(error_code, term_signal);
    }

    (void)kill_other_threads();

    /* Now quit our thread. Since we are the last one, this will exit the whole LibOS. */
    thread_exit(error_code, term_signal);
}

long shim_do_exit_group(int error_code) {
    assert(!is_internal(get_cur_thread()));

    error_code &= 0xFF;

    log_debug("---- shim_exit_group (returning %d)", error_code);

    process_exit(error_code, 0);
}

long shim_do_exit(int error_code) {
    assert(!is_internal(get_cur_thread()));

    error_code &= 0xFF;

    log_debug("---- shim_exit (returning %d)", error_code);

    thread_exit(error_code, 0);
}
