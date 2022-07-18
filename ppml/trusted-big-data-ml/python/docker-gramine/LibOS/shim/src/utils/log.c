/* SPDX-License-Identifier: LGPL-3.0-or-later */
/* Copyright (C) 2014 Stony Brook University */
/* Copyright (C) 2021 Intel Corporation
 *                    Michał Kowalczyk <mkow@invisiblethingslab.com>
 */

#include <stdarg.h>
#include <stdint.h>

#include "api.h"
#include "pal.h"
#include "shim_internal.h"
#include "shim_ipc.h"
#include "shim_lock.h"
#include "shim_process.h"

int g_log_level = LOG_LEVEL_NONE;

/* NOTE: We could add "libos" prefix to the below strings for more fine-grained log info */
static const char* log_level_to_prefix[] = {
    [LOG_LEVEL_NONE]    = "",
    [LOG_LEVEL_ERROR]   = "error: ",
    [LOG_LEVEL_WARNING] = "warning: ",
    [LOG_LEVEL_DEBUG]   = "debug: ",
    [LOG_LEVEL_TRACE]   = "trace: ",
    [LOG_LEVEL_ALL]     = "", // not a valid entry actually (no public wrapper uses this log level)
};

void log_setprefix(shim_tcb_t* tcb) {
    if (g_log_level <= LOG_LEVEL_NONE)
        return;

    lock(&g_process.fs_lock);

    const char* exec_name;
    if (g_process.exec) {
        if (g_process.exec->dentry) {
            exec_name = g_process.exec->dentry->name;
        } else {
            /* Unknown executable name */
            exec_name = "?";
        }
    } else {
        /* `g_process.exec` not available yet, happens on process init */
        exec_name = "";
    }

    uint32_t vmid = g_process_ipc_ids.self_vmid;
    size_t total_len;
    if (tcb->tp) {
        if (!is_internal(tcb->tp)) {
            /* normal app thread: show Process ID, Thread ID, and exec name */
            total_len = snprintf(tcb->log_prefix, ARRAY_SIZE(tcb->log_prefix), "[P%u:T%u:%s] ",
                                 vmid, tcb->tp->tid, exec_name);
        } else {
            /* internal LibOS thread: show Process ID and Internal-thread marker */
            total_len = snprintf(tcb->log_prefix, ARRAY_SIZE(tcb->log_prefix), "[P%u:shim] ", vmid);
        }
    } else if (vmid) {
        /* unknown thread (happens on process init): show just Process ID and exec name */
        total_len = snprintf(tcb->log_prefix, ARRAY_SIZE(tcb->log_prefix), "[P%u::%s] ", vmid,
                             exec_name);
    } else {
        /* unknown process (happens on process init): show exec name */
        total_len = snprintf(tcb->log_prefix, ARRAY_SIZE(tcb->log_prefix), "[::%s] ", exec_name);
    }
    if (total_len > ARRAY_SIZE(tcb->log_prefix) - 1) {
        /* exec name too long, snip it */
        const char* snip = "...] ";
        size_t snip_size = strlen(snip) + 1;
        memcpy(tcb->log_prefix + ARRAY_SIZE(tcb->log_prefix) - snip_size, snip, snip_size);
    }

    unlock(&g_process.fs_lock);
}

static int buf_write_all(const char* str, size_t size, void* arg) {
    __UNUSED(arg);
    DkDebugLog(str, size);
    return 0;
}

void shim_log(int level, const char* fmt, ...) {
    if (level <= g_log_level) {
        struct print_buf buf = INIT_PRINT_BUF(buf_write_all);

        buf_puts(&buf, shim_get_tcb()->log_prefix);
        buf_puts(&buf, log_level_to_prefix[level]);

        va_list ap;
        va_start(ap, fmt);
        buf_vprintf(&buf, fmt, ap);
        va_end(ap);
        buf_printf(&buf, "\n");

        buf_flush(&buf);
    }
}
