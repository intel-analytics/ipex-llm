/* SPDX-License-Identifier: LGPL-3.0-or-later */
/* Copyright (C) 2014 Stony Brook University */

#include "api.h"
#include "assert.h"
#include "pal.h"
#include "pal_internal.h"

static int buf_write_all(const char* str, size_t size, void* arg) {
    __UNUSED(arg);
    _DkDebugLog(str, size);
    return 0;
}

static void log_vprintf(const char* prefix, const char* fmt, va_list ap) {
    struct print_buf buf = INIT_PRINT_BUF(buf_write_all);

    if (prefix)
        buf_puts(&buf, prefix);
    buf_vprintf(&buf, fmt, ap);
    buf_printf(&buf, "\n");
    buf_flush(&buf);
}

void pal_log(int level, const char* fmt, ...) {
    if (level <= g_pal_public_state.log_level) {
        va_list ap;
        va_start(ap, fmt);
        assert(0 <= level && (size_t)level < LOG_LEVEL_ALL);

        /* NOTE: We could add "pal" prefix to the below strings for more fine-grained log info */
        const char* prefix = NULL;

        /* NOTE: We use the switch instead of an array because we may log messages before/during
         *       PAL symbol relocation, and the array would be a to-be-relocated symbol and thus
         *       accessing it would segfault (asm will use the offset instead of real address) */
        switch (level) {
            case LOG_LEVEL_NONE:    prefix = ""; break;
            case LOG_LEVEL_ERROR:   prefix = "error: "; break;
            case LOG_LEVEL_WARNING: prefix = "warning: "; break;
            case LOG_LEVEL_DEBUG:   prefix = "debug: "; break;
            case LOG_LEVEL_TRACE:   prefix = "trace: "; break;
        }

        log_vprintf(prefix, fmt, ap);
        va_end(ap);
    }
}
