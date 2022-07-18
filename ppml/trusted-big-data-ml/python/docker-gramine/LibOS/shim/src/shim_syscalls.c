/* SPDX-License-Identifier: LGPL-3.0-or-later */
/* Copyright (C) 2020 Intel Corporation
 *                    Borys Popławski <borysp@invisiblethingslab.com>
 */

#include "asan.h"
#include "shim_defs.h"
#include "shim_internal.h"
#include "shim_lock.h"
#include "shim_signal.h"
#include "shim_table.h"
#include "shim_tcb.h"
#include "shim_thread.h"

typedef arch_syscall_arg_t (*six_args_syscall_t)(arch_syscall_arg_t, arch_syscall_arg_t,
                                                 arch_syscall_arg_t, arch_syscall_arg_t,
                                                 arch_syscall_arg_t, arch_syscall_arg_t);

/*
 * `context` is expected to be placed at the bottom of Gramine-internal stack.
 * If you change this function please also look at `shim_do_rt_sigsuspend`!
 */
noreturn void shim_emulate_syscall(PAL_CONTEXT* context) {
    SHIM_TCB_SET(context.regs, context);

    unsigned long sysnr = pal_context_get_syscall(context);
    arch_syscall_arg_t ret = 0;

    if (sysnr == GRAMINE_CUSTOM_SYSCALL_NR) {
        unsigned long args[] = { ALL_SYSCALL_ARGS(context) };
        ret = handle_libos_call(args[0], args[1], args[2]);
    } else {
        if (sysnr >= LIBOS_SYSCALL_BOUND || !shim_table[sysnr]) {
            warn_unsupported_syscall(sysnr);
            ret = -ENOSYS;
            goto out;
        }

        SHIM_TCB_SET(context.syscall_nr, sysnr);
        six_args_syscall_t syscall_func = (six_args_syscall_t)shim_table[sysnr];

        debug_print_syscall_before(sysnr, ALL_SYSCALL_ARGS(context));
        ret = syscall_func(ALL_SYSCALL_ARGS(context));
        debug_print_syscall_after(sysnr, ret, ALL_SYSCALL_ARGS(context));
    }
out:
    pal_context_set_retval(context, ret);

    /* Some syscalls e.g. `sigreturn` could have changed context and in reality we might be not
     * returning from a syscall. */
    if (!handle_signal(context) && SHIM_TCB_GET(context.syscall_nr) >= 0) {
        switch (ret) {
            case -ERESTARTNOHAND:
            case -ERESTARTSYS:
            case -ERESTARTNOINTR:
                restart_syscall(context, sysnr);
                break;
            default:
                break;
        }
    }

    struct shim_thread* current = get_cur_thread();
    if (current->has_saved_sigmask) {
        lock(&current->lock);
        set_sig_mask(current, &current->saved_sigmask);
        unlock(&current->lock);
        current->has_saved_sigmask = false;
    }

    SHIM_TCB_SET(context.syscall_nr, -1);
    SHIM_TCB_SET(context.regs, NULL);

    return_from_syscall(context);
}

__attribute_no_sanitize_address
noreturn void return_from_syscall(PAL_CONTEXT* context) {
#ifdef ASAN
    uintptr_t libos_stack_bottom = (uintptr_t)SHIM_TCB_GET(libos_stack_bottom);
    asan_unpoison_region(libos_stack_bottom - SHIM_THREAD_LIBOS_STACK_SIZE,
                         SHIM_THREAD_LIBOS_STACK_SIZE);
#endif
    _return_from_syscall(context);
}
