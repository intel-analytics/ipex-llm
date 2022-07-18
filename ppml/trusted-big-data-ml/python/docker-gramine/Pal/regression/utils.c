#include "api.h"
#include "pal.h"
#include "pal_regression.h"

// pal_printf() is required by PAL regression tests.
static int buf_write_all(const char* str, size_t size, void* arg) {
    __UNUSED(arg);
    DkDebugLog(str, size);
    return 0;
}

static void log_vprintf(const char* fmt, va_list ap, bool append_newline) {
    struct print_buf buf = INIT_PRINT_BUF(buf_write_all);
    buf_vprintf(&buf, fmt, ap);
    if (append_newline)
        buf_printf(&buf, "\n");
    buf_flush(&buf);
}

void pal_printf(const char* fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    log_vprintf(fmt, ap, /*append_newline=*/false);
    va_end(ap);
}

/* The below two functions are used by stack protector's __stack_chk_fail(), _FORTIFY_SOURCE's
 * *_chk() functions and by assert.h's assert() defined in the common library. Thus they might be
 * called by any execution context, including these PAL tests. */
void _log(int level, const char* fmt, ...) {
    (void)level; /* PAL regression always prints log messages */
    va_list ap;
    va_start(ap, fmt);
    log_vprintf(fmt, ap, /*append_newline=*/true);
    va_end(ap);
}

noreturn void abort(void) {
    DkProcessExit(131); /* ENOTRECOVERABLE = 131 */
}

/* We just need these symbols for `printf`, but they won't be used in runtime. */
void* malloc(size_t size) {
    abort();
}

void free(void* ptr) {
    if (!ptr) {
        return;
    }
    abort();
}
