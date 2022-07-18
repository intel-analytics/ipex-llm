/* SPDX-License-Identifier: LGPL-3.0-or-later */
/* Copyright (C) 2021 Intel Corporation
 *                    Borys Popławski <borysp@invisiblethingslab.com>
 */

#include <asm/errno.h>
#include <limits.h>
#include <stdbool.h>
#include <stdint.h>

#include "api.h"
#include "assert.h"
#include "log.h"

/* "api_fortified.h" might be included and it defines these. */
#undef vsnprintf
#undef snprintf

enum length_modifier {
    None,
    HH,
    H,
    L,
    LL,
    Z,
};

static char to_digit(unsigned val, unsigned base) {
    __UNUSED(base);
    assert(val < base);
    assert(base <= 16);

    if (val < 10) {
        return '0' + val;
    }
    return 'a' + val - 10;
}

static int printf_padding(int (*write_callback)(const char* buf, size_t size, void* arg), void* arg,
                          char pad_char, size_t size) {
    char buf[0x100];
    memset(buf, pad_char, sizeof(buf));

    while (size) {
        size_t this_size = MIN(size, sizeof(buf));
        int ret = write_callback(buf, this_size, arg);
        if (ret < 0) {
            return ret;
        }
        size -= this_size;
    }

    return 0;
}

/*!
 * \brief Core printf implementation.
 *
 * \param      write_callback  Function called on each generated chunk of data.
 * \param      arg             Passed to \p write_callback.
 * \param      fmt             Format string.
 * \param      ap              List of optional variadic arguments.
 * \param[out] out_size        Total size of written data (sum of sizes passed to all
 *                             \p write_callback invocations).
 *
 * \returns 0 on success, negative error code on failure.
 *
 * Note that this function does not append a trailing null byte i.e. \p write_callback gets only
 * actual data.
 */
static int vprintf_core(int (*write_callback)(const char* buf, size_t size, void* arg), void* arg,
                        const char* fmt, va_list ap, size_t* out_size) {
    int ret;
    size_t total_print_len = 0;

    while (1) {
        const char* percent_ptr = strchr(fmt, '%');
        if (!percent_ptr) {
            break;
        }
        ret = write_callback(fmt, percent_ptr - fmt, arg);
        if (ret < 0) {
            return ret;
        }
        total_print_len += percent_ptr - fmt;
        fmt = percent_ptr + 1;

        bool use_alternative_form = false;
        bool zero_pad = false;
        bool pad_right = false;
        bool signed_positive_blank = false;
        bool force_sign = false;

        /* Parse flags. */
        bool end_of_flags = false;
        while (!end_of_flags) {
            switch (*fmt++) {
                case '#':
                    use_alternative_form = true;
                    break;
                case '0':
                    zero_pad = true;
                    break;
                case '-':
                    pad_right = true;
                    break;
                case ' ':
                    signed_positive_blank = true;
                    break;
                case '+':
                    force_sign = true;
                    break;
                default:
                    end_of_flags = true;
                    fmt--; /* Unget this character. */
                    break;
            }
        }

        size_t width = 0;
        if (*fmt == '*') {
            int x = va_arg(ap, int);
            if (x >= 0) {
                width = x;
            }
            fmt++;
        } else if ('0' <= *fmt && *fmt <= '9') {
            /* This differs from normal `printf` because it saturates instead of overflowing, but
             * such big width would be an undefined or unspecified behavior anyway (signed
             * overflow). */
            width = strtol(fmt, (char**)&fmt, 10);
        }

        size_t precision = 1;
        bool precision_seen = false;
        if (*fmt == '.') {
            fmt++;
            if (*fmt == '*') {
                int x = va_arg(ap, int);
                if (x >= 0) {
                    precision = x;
                    precision_seen = true;
                }
                fmt++;
            } else if ('0' <= *fmt && *fmt <= '9') {
                precision = strtol(fmt, (char**)&fmt, 10);
                precision_seen = true;
            }
        }

        enum length_modifier len_modifier = None;
        switch (*fmt++) {
            case 'l':
                len_modifier = L;
                if (*fmt == 'l') {
                    len_modifier = LL;
                    fmt++;
                }
                break;
            case 'h':
                len_modifier = H;
                if (*fmt == 'h') {
                    len_modifier = HH;
                    fmt++;
                }
                break;
            case 'z':
                len_modifier = Z;
                break;
            default:
                fmt--; /* Unget this character. */
                break;
        }

        char conversion_specifier = *fmt++;
        bool integer_conversion = true;
        uintmax_t val = 0;
        bool negative = false;

        switch (conversion_specifier) {
            case 'p':
                val = (uintptr_t)va_arg(ap, void*);
                conversion_specifier = 'x';
                use_alternative_form = true;
                break;
            case 'x':
            case 'o':
            case 'u':
                switch (len_modifier) {
                    case HH:
                        val = (unsigned char)va_arg(ap, int);
                        break;
                    case H:
                        val = (unsigned short)va_arg(ap, int);
                        break;
                    case L:
                        val = va_arg(ap, unsigned long);
                        break;
                    case LL:
                        val = va_arg(ap, unsigned long long);
                        break;
                    case Z:
                        val = va_arg(ap, size_t);
                        break;
                    case None:
                        val = va_arg(ap, unsigned int);
                        break;
                    default:
                        BUG();
                }
                break;
            case 'i':
            case 'd':;
                intmax_t x;
                switch (len_modifier) {
                    case HH:
                        x = (signed char)va_arg(ap, int);
                        break;
                    case H:
                        x = (short)va_arg(ap, int);
                        break;
                    case L:
                        x = va_arg(ap, long);
                        break;
                    case LL:
                        x = va_arg(ap, long long);
                        break;
                    case Z:
                        x = va_arg(ap, ssize_t);
                        break;
                    case None:
                        x = va_arg(ap, int);
                        break;
                    default:
                        BUG();
                }
                if (x >= 0) {
                    val = x;
                } else {
                    negative = true;
                    (void)__builtin_sub_overflow(0, x, &val);
                }
                break;
            default:
                integer_conversion = false;
                break;
        }

        const char* val_ptr = NULL;
        size_t val_len = 0;
        void* allocated_mem = NULL;
        char inline_buf[0x80];

        if (integer_conversion) {
            /* 0x20 should be enough to store all possible values.
             * log2(8) = 3 (octal produces most digits). */
            static_assert(0x20 > sizeof(UINTMAX_MAX) * 8 / 3, "oops");
            /* +1 for sign, +2 for optional prefix (e.g. "0x"). */
            size_t backing_store_size = MAX(0x20u, precision) + 3;
            char* backing_store = inline_buf;
            if (backing_store_size > sizeof(inline_buf)) {
                allocated_mem = malloc(backing_store_size);
                if (!allocated_mem) {
                    return -ENOMEM;
                }
                backing_store = allocated_mem;
            }
            memset(backing_store, '0', backing_store_size);

            size_t idx = backing_store_size;
            unsigned base = 10;
            if (conversion_specifier == 'x') {
                base = 16;
            } else if (conversion_specifier == 'o') {
                base = 8;
            }

            bool was_zero = val == 0;

            while (val) {
                assert(idx > 0);
                backing_store[--idx] = to_digit(val % base, base);
                val /= base;
            }

            if (backing_store_size - idx < precision) {
                idx = backing_store_size - precision;
            }

            if (use_alternative_form) {
                if (conversion_specifier == 'x' && !was_zero) {
                    backing_store[--idx] = 'x';
                    backing_store[--idx] = '0';
                } else if (conversion_specifier == 'o') {
                    if (idx == backing_store_size || backing_store[idx] != '0') {
                        backing_store[--idx] = '0';
                    }
                }
            }

            if (negative) {
                backing_store[--idx] = '-';
            } else if (conversion_specifier == 'd' || conversion_specifier == 'i') {
                if (force_sign) {
                    backing_store[--idx] = '+';
                } else if (signed_positive_blank) {
                    backing_store[--idx] = ' ';
                }
            }

            val_ptr = &backing_store[idx];
            val_len = backing_store_size - idx;
        } else {
            switch (conversion_specifier) {
                case 's':
                    val_ptr = va_arg(ap, const char*);
                    if (!val_ptr)
                        val_ptr = "(null)";
                    val_len = strnlen(val_ptr, precision_seen ? precision : SIZE_MAX);
                    break;
                case 'c':;
                    unsigned char x = va_arg(ap, int);
                    inline_buf[0] = x;
                    val_ptr = inline_buf;
                    val_len = 1;
                    break;
                case '%':
                    if (percent_ptr == &fmt[-2]) {
                        percent_ptr++;
                    }
                    /* Fallthrough. */
                default:
                    /* This format is either invalid or not supported, just print it. */
                    val_ptr = percent_ptr;
                    val_len = fmt - percent_ptr;
                    pad_right = false;
                    break;
            }
        }

        size_t padding_size = width > val_len ? width - val_len : 0;
        char pad_char = zero_pad && integer_conversion && !precision_seen ? '0' : ' ';

        if (!pad_right && padding_size) {
            ret = printf_padding(write_callback, arg, pad_char, padding_size);
            if (ret < 0) {
                free(allocated_mem);
                return ret;
            }
            total_print_len += padding_size;
        }
        ret = write_callback(val_ptr, val_len, arg);
        if (ret < 0) {
            free(allocated_mem);
            return ret;
        }
        total_print_len += val_len;
        if (pad_right && padding_size) {
            ret = printf_padding(write_callback, arg, pad_char, padding_size);
            if (ret < 0) {
                free(allocated_mem);
                return ret;
            }
            total_print_len += padding_size;
        }

        free(allocated_mem);
    }

    size_t len = strlen(fmt);
    ret = write_callback(fmt, len, arg);
    if (ret < 0) {
        return ret;
    }
    total_print_len += len;

    *out_size = total_print_len;
    return 0;
}

struct snprintf_arg {
    char* buf;
    size_t size;
    size_t pos;
};

static int snprintf_callback(const char* buf, size_t size, void* _arg) {
    struct snprintf_arg* arg = _arg;
    if (arg->pos < arg->size) {
        size_t copy_size = MIN(size, arg->size - arg->pos);
        memcpy(&arg->buf[arg->pos], buf, copy_size);
        arg->pos += copy_size;
    }
    return 0;
}

int vsnprintf(char* buf, size_t buf_size, const char* fmt, va_list ap) {
    struct snprintf_arg arg = {
        .buf = buf,
        .size = buf_size ? buf_size - 1 : 0,
        .pos = 0,
    };
    size_t len = 0;
    int ret = vprintf_core(snprintf_callback, &arg, fmt, ap, &len);
    if (ret < 0) {
        return ret;
    }
    if (buf_size) {
        buf[arg.pos] = 0;
    }
    if (len > INT_MAX) {
        return -EOVERFLOW;
    }
    return (int)len;
}

int snprintf(char* buf, size_t buf_size, const char* fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    int ret = vsnprintf(buf, buf_size, fmt, ap);
    va_end(ap);
    return ret;
}

int __vsnprintf_chk(char* buf, size_t buf_size, int flag, size_t real_size, const char* fmt,
                    va_list ap) {
    __UNUSED(flag);
    if (buf_size > real_size) {
        log_always("vsnprintf() check failed");
        abort();
    }
    return vsnprintf(buf, buf_size, fmt, ap);
}

int __snprintf_chk(char* buf, size_t buf_size, int flag, size_t real_size, const char* fmt, ...) {
    __UNUSED(flag);
    if (buf_size > real_size) {
        log_always("vsnprintf() check failed");
        abort();
    }

    va_list ap;
    va_start(ap, fmt);
    int ret = vsnprintf(buf, buf_size, fmt, ap);
    va_end(ap);
    return ret;
}

static int buf_write(const char* data, size_t size, void* arg) {
    struct print_buf* print_buf = arg;
    while (size) {
        if (print_buf->pos == ARRAY_SIZE(print_buf->data)) {
            int ret = buf_flush(print_buf);
            if (ret < 0) {
                return ret;
            }
        }

        size_t this_size = MIN(size, ARRAY_SIZE(print_buf->data) - print_buf->pos);
        memcpy(&print_buf->data[print_buf->pos], data, this_size);
        print_buf->pos += this_size;
        data += this_size;
        size -= this_size;
    }
    return 0;
}

int buf_putc(struct print_buf* buf, char c) {
    return buf_write(&c, 1, buf);
}

int buf_puts(struct print_buf* buf, const char* str) {
    return buf_write(str, strlen(str), buf);
}

int buf_flush(struct print_buf* buf) {
    int ret;
    if (buf->pos > 0) {
        if ((ret = buf->buf_write_all(&buf->data[0], buf->pos, buf->arg)) < 0)
            return ret;
        buf->pos = 0;
    }
    return 0;
}

int buf_vprintf(struct print_buf* buf, const char* fmt, va_list ap) {
    size_t size = 0;
    return vprintf_core(buf_write, buf, fmt, ap, &size);
}

int buf_printf(struct print_buf* buf, const char* fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    int ret = buf_vprintf(buf, fmt, ap);
    va_end(ap);
    return ret;
}
