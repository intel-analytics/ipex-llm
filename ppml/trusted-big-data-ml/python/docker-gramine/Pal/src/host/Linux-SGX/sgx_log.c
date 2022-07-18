/* SPDX-License-Identifier: LGPL-3.0-or-later */
/* Copyright (C) 2021 Intel Corporation
 *                    Paweł Marczewski <pawel@invisiblethingslab.com>
 *                    Michał Kowalczyk <mkow@invisiblethingslab.com>
 */

#include "assert.h"
#include "linux_utils.h"
#include "pal_internal.h"
#include "pal_linux.h"
#include "perm.h"
#include "sgx_log.h"

int g_urts_log_level = PAL_LOG_DEFAULT_LEVEL;
int g_urts_log_fd = PAL_LOG_DEFAULT_FD;

int urts_log_init(const char* path) {
    int ret;

    if (g_urts_log_fd != PAL_LOG_DEFAULT_FD) {
        ret = DO_SYSCALL(close, g_urts_log_fd);
        g_urts_log_fd = PAL_LOG_DEFAULT_FD;
        if (ret < 0)
            return ret;
    }

    ret = DO_SYSCALL(open, path, O_WRONLY | O_APPEND | O_CREAT, PERM_rw_______);
    if (ret < 0)
        return ret;
    g_urts_log_fd = ret;
    return 0;
}

static int buf_write_all(const char* str, size_t size, void* arg) {
    int fd = *(int*)arg;
    return write_all(fd, str, size);
}

static void print_to_fd(int fd, const char* prefix, const char* fmt, va_list ap) {
    struct print_buf buf = INIT_PRINT_BUF_ARG(buf_write_all, &fd);

    if (prefix)
        buf_puts(&buf, prefix);
    buf_vprintf(&buf, fmt, ap);
    buf_printf(&buf, "\n");
    buf_flush(&buf);
    // No error handling, as `pal_log` doesn't return errors anyways.
}

void pal_log(int level, const char* fmt, ...) {
    if (level <= g_urts_log_level) {
        va_list ap;
        va_start(ap, fmt);
        assert(0 <= level && (size_t)level < LOG_LEVEL_ALL);

        /* NOTE: We could add "untrusted-pal" prefix to the below strings for more fine-grained log
         *       info */
        const char* prefix = NULL;
        switch (level) {
            case LOG_LEVEL_NONE:    prefix = ""; break;
            case LOG_LEVEL_ERROR:   prefix = "error: "; break;
            case LOG_LEVEL_WARNING: prefix = "warning: "; break;
            case LOG_LEVEL_DEBUG:   prefix = "debug: "; break;
            case LOG_LEVEL_TRACE:   prefix = "trace: "; break;
        }

        print_to_fd(g_urts_log_fd, prefix, fmt, ap);
        va_end(ap);
    }
}
