/* SPDX-License-Identifier: LGPL-3.0-or-later */
/* Copyright (C) 2014 Stony Brook University
 * Copyright (C) 2020 Intel Labs
 * Copyright (C) 2021 Intel Corporation
 *                    Borys Popławski <borysp@invisiblethingslab.com>
 */

/*
 * This file contains APIs to set up signal handlers and seccomp.
 */

#include <stddef.h> /* needed by <linux/signal.h> for size_t */
#include <linux/filter.h>
#include <linux/prctl.h>
#include <linux/seccomp.h>
#include <linux/signal.h>

#include "api.h"
#include "pal.h"
#include "pal_error.h"
#include "pal_internal.h"
#include "pal_linux.h"
#include "ucontext.h"

static const int ASYNC_SIGNALS[] = {SIGTERM, SIGCONT};

static int block_signal(int sig, bool block) {
    int how = block ? SIG_BLOCK : SIG_UNBLOCK;
    int ret = arch_do_rt_sigprocmask(sig, how);

    return ret < 0 ? unix_to_pal_error(ret) : 0;
}

static int set_signal_handler(int sig, void* handler) {
    int ret = arch_do_rt_sigaction(sig, handler, ASYNC_SIGNALS, ARRAY_SIZE(ASYNC_SIGNALS));
    if (ret < 0)
        return unix_to_pal_error(ret);

    return block_signal(sig, /*block=*/false);
}

int block_async_signals(bool block) {
    for (size_t i = 0; i < ARRAY_SIZE(ASYNC_SIGNALS); i++) {
        int ret = block_signal(ASYNC_SIGNALS[i], block);
        if (ret < 0)
            return unix_to_pal_error(ret);
    }
    return 0;
}

static enum pal_event signal_to_pal_event(int sig) {
    switch (sig) {
        case SIGFPE:
            return PAL_EVENT_ARITHMETIC_ERROR;
        case SIGSEGV:
        case SIGBUS:
            return PAL_EVENT_MEMFAULT;
        case SIGILL:
        case SIGSYS:
            return PAL_EVENT_ILLEGAL;
        case SIGTERM:
            return PAL_EVENT_QUIT;
        case SIGCONT:
            return PAL_EVENT_INTERRUPTED;
        default:
            BUG();
    }
}

/*
 * This function must be reentrant and thread-safe - this includes `upcall` too! Technically,
 * only for cases when the exception arrived while in Gramine code; if signal arrived while in
 * the user app, this function doesn't need to be reentrant and thread-safe.
 */
static void perform_signal_handling(enum pal_event event, bool is_in_pal, PAL_NUM addr,
                                    ucontext_t* uc) {
    pal_event_handler_t upcall = _DkGetExceptionHandler(event);
    if (!upcall)
        return;

    PAL_CONTEXT context;
    ucontext_to_pal_context(&context, uc);
    (*upcall)(is_in_pal, addr, &context);
    pal_context_to_ucontext(uc, &context);
}

static void handle_sync_signal(int signum, siginfo_t* info, struct ucontext* uc) {
    if (info->si_signo == SIGSYS && info->si_code == SYS_SECCOMP) {
        ucontext_revert_syscall(uc, info->si_arch, info->si_syscall, info->si_call_addr);
        if (FIRST_TIME()) {
            log_always("Emulating a raw system/supervisor call. This degrades performance, consider"
                       " patching your application to use Gramine syscall API.");
        }
    }

    enum pal_event event = signal_to_pal_event(signum);

    uintptr_t rip = ucontext_get_ip(uc);
    if (!ADDR_IN_PAL_OR_VDSO(rip)) {
        /* exception happened in application or LibOS code, normal benign case */
        perform_signal_handling(event, /*is_in_pal=*/false, (PAL_NUM)info->si_addr, uc);
        return;
    }

    /* exception happened in PAL code: this is fatal in Gramine */

    char buf[LOCATION_BUF_SIZE];
    pal_describe_location(rip, buf, sizeof(buf));

    const char* event_name = pal_event_name(event);
    bool in_vdso = is_in_vdso(rip);
    log_error("Unexpected %s occurred inside %s (%s, PID = %u, TID = %ld)", event_name,
              in_vdso ? "VDSO" : "PAL", buf, g_pal_linux_state.host_pid, DO_SYSCALL(gettid));

    _DkProcessExit(1);
}

static void handle_async_signal(int signum, siginfo_t* info, struct ucontext* uc) {
    __UNUSED(info);

    enum pal_event event = signal_to_pal_event(signum);

    uintptr_t rip = ucontext_get_ip(uc);
    perform_signal_handling(event, ADDR_IN_PAL_OR_VDSO(rip), /*addr=*/0, uc);
}

#define BPF_GENERATE_ALLOWED_RANGE(range_begin_low, range_begin_high,                           \
                                   range_end_low, range_end_high)                               \
    /* 0: A = ip >> 32 */                                                                       \
    BPF_STMT(BPF_LD | BPF_W | BPF_ABS, offsetof(struct seccomp_data, instruction_pointer) + 4), \
    /* 1: A >= range_begin_high ? 0 : NEXT */                                                   \
    BPF_JUMP(BPF_JMP | BPF_JGE | BPF_K, (range_begin_high), 0, /*NEXT*/9),                      \
    /* 2: A == range_begin_high ? 0 : CMP_END */                                                \
    BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, (range_begin_high), 0, /*CMP_END*/2),                   \
    /* 3: A = ip & (2**32 - 1) */                                                               \
    BPF_STMT(BPF_LD | BPF_W | BPF_ABS, offsetof(struct seccomp_data, instruction_pointer)),     \
    /* 4: A >= range_begin_low ? CMP_END : NEXT */                                              \
    BPF_JUMP(BPF_JMP | BPF_JGE | BPF_K, (range_begin_low), /*CMP_END*/0, /*NEXT*/6),            \
    /* 5: CMP_END: A = ip >> 32 */                                                              \
    BPF_STMT(BPF_LD | BPF_W | BPF_ABS, offsetof(struct seccomp_data, instruction_pointer) + 4), \
    /* 6: A > range_end_high ? NEXT : 0 */                                                      \
    BPF_JUMP(BPF_JMP | BPF_JGT | BPF_K, (range_end_high), /*NEXT*/4, 0),                        \
    /* 7: A == range_end_high ? 0 : ALLOW */                                                    \
    BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, (range_end_high), 0, /*ALLOW*/2),                       \
    /* 8: A = ip & (2**32 - 1) */                                                               \
    BPF_STMT(BPF_LD | BPF_W | BPF_ABS, offsetof(struct seccomp_data, instruction_pointer)),     \
    /* 9: A >= range_end_low ? NEXT : ALLOW */                                                  \
    BPF_JUMP(BPF_JMP | BPF_JGE | BPF_K, (range_end_low), /*NEXT*/1, 0),                         \
                                                                                                \
    BPF_STMT(BPF_RET | BPF_K, SECCOMP_RET_ALLOW)

static int setup_seccomp(uintptr_t vdso_start, uintptr_t vdso_end) {
    int ret = DO_SYSCALL(prctl, PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0);
    if (ret < 0) {
        log_error("prctl(PR_SET_NO_NEW_PRIVS, 1) failed: %d", ret);
        return -1;
    }

    uint32_t syscalls_code_begin_low = (uintptr_t)gramine_raw_syscalls_code_begin;
    uint32_t syscalls_code_begin_high = (uintptr_t)gramine_raw_syscalls_code_begin >> 32;
    uint32_t syscalls_code_end_low = (uintptr_t)gramine_raw_syscalls_code_end;
    uint32_t syscalls_code_end_high = (uintptr_t)gramine_raw_syscalls_code_end >> 32;
    uint32_t vdso_begin_low = vdso_start;
    uint32_t vdso_begin_high = vdso_start >> 32;
    uint32_t vdso_end_low = vdso_end;
    uint32_t vdso_end_high = vdso_end >> 32;

    struct sock_filter filter[] = {
        BPF_GENERATE_ALLOWED_RANGE(syscalls_code_begin_low, syscalls_code_begin_high,
                                   syscalls_code_end_low, syscalls_code_end_high),
        BPF_GENERATE_ALLOWED_RANGE(vdso_begin_low, vdso_begin_high, vdso_end_low, vdso_end_high),
        BPF_STMT(BPF_RET | BPF_K, SECCOMP_RET_TRAP),
    };

    struct sock_fprog seccomp_filter = {
        .len = ARRAY_SIZE(filter),
        .filter = filter,
    };
    ret = DO_SYSCALL(prctl, PR_SET_SECCOMP, SECCOMP_MODE_FILTER, &seccomp_filter);
    if (ret < 0) {
        log_error("Setting seccomp filter failed: %d", ret);
        return -1;
    }
    return 0;
}

void signal_setup(bool is_first_process, uintptr_t vdso_start, uintptr_t vdso_end) {
    int ret;

    /* SIGPIPE and SIGCHLD are emulated completely inside LibOS */
    ret = set_signal_handler(SIGPIPE, SIG_IGN);
    if (ret < 0)
        goto err;

    ret = set_signal_handler(SIGCHLD, SIG_IGN);
    if (ret < 0)
        goto err;

    /* register synchronous signals (exceptions) in host Linux */
    ret = set_signal_handler(SIGFPE, handle_sync_signal);
    if (ret < 0)
        goto err;

    ret = set_signal_handler(SIGSEGV, handle_sync_signal);
    if (ret < 0)
        goto err;

    ret = set_signal_handler(SIGBUS, handle_sync_signal);
    if (ret < 0)
        goto err;

    ret = set_signal_handler(SIGILL, handle_sync_signal);
    if (ret < 0)
        goto err;

    ret = set_signal_handler(SIGSYS, handle_sync_signal);
    if (ret < 0)
        goto err;

    /* register asynchronous signals in host Linux */
    for (size_t i = 0; i < ARRAY_SIZE(ASYNC_SIGNALS); i++) {
        ret = set_signal_handler(ASYNC_SIGNALS[i], handle_async_signal);
        if (ret < 0)
            goto err;
    }

    if (is_first_process) {
        ret = setup_seccomp(vdso_start, vdso_end);
        if (ret < 0) {
            INIT_FAIL(PAL_ERROR_DENIED, "Setting up seccomp for inline syscall handling failed");
        }
    }

    return;
err:
    INIT_FAIL(-ret, "Cannot setup signal handlers!");
}
