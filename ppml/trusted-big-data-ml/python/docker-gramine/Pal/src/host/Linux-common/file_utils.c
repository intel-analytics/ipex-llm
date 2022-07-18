/* SPDX-License-Identifier: LGPL-3.0-or-later */
/* Copyright (C) 2020 Intel Corporation
 *                    Michał Kowalczyk <mkow@invisiblethingslab.com>
 */

#include <asm/errno.h>
#include <asm/fcntl.h>
#include <linux/fs.h>

#include "api.h"
#include "linux_utils.h"
#include "syscall.h"

int read_all(int fd, void* buf, size_t size) {
    size_t bytes_read = 0;
    while (bytes_read < size) {
        long ret = DO_SYSCALL(read, fd, buf + bytes_read, size - bytes_read);
        if (ret <= 0) {
            if (ret == -EINTR)
                continue;
            if (ret == 0)
                ret = -EINVAL; // unexpected EOF
            return ret;
        }
        bytes_read += (size_t)ret;
    }
    return 0;
}

int write_all(int fd, const void* buf, size_t size) {
    size_t bytes_written = 0;
    while (bytes_written < size) {
        long ret = DO_SYSCALL(write, fd, buf + bytes_written, size - bytes_written);
        if (ret <= 0) {
            if (ret == -EINTR)
                continue;
            if (ret == 0) {
                /* This case should be impossible. */
                ret = -EINVAL;
            }
            return ret;
        }
        bytes_written += (size_t)ret;
    }
    return 0;
}

int read_text_file_to_cstr(const char* path, char** out) {
    long ret;
    char* buf = NULL;
    long fd = DO_SYSCALL(open, path, O_RDONLY, 0);
    if (fd < 0) {
        ret = fd;
        goto out;
    }

    ret = DO_SYSCALL(lseek, fd, 0, SEEK_END);
    if (ret < 0) {
        goto out;
    }
    size_t size = ret;

    ret = DO_SYSCALL(lseek, fd, 0, SEEK_SET);
    if (ret < 0) {
        goto out;
    }

    if (size + 1 < size) {
        ret = -E2BIG; // int overflow
        goto out;
    }
    buf = malloc(size + 1);
    if (!buf) {
        ret = -ENOMEM;
        goto out;
    }

    size_t bytes_read = 0;
    while (bytes_read < size) {
        ret = DO_SYSCALL(read, fd, buf + bytes_read, size - bytes_read);
        if (ret <= 0) {
            if (ret == -EINTR)
                continue;
            if (ret == 0)
                ret = -EINVAL; // unexpected EOF
            goto out;
        }
        bytes_read += ret;
    }
    buf[size] = '\0';
    *out = buf;
    buf = NULL;
    ret = 0;
out:
    if (fd >= 0) {
        long close_ret = DO_SYSCALL(close, fd);
        if (ret == 0)
            ret = close_ret;
    }
    free(buf);
    return (int)ret;
}

int read_text_file_iter_lines(const char* path, int (*callback)(const char* line, void* arg,
                                                                bool* out_stop),
                              void* arg) {
    int ret;

    int fd = DO_SYSCALL(open, path, O_RDONLY, 0);
    if (fd < 0)
        return fd;

    size_t buf_size = 256;
    char* buf = malloc(buf_size);
    if (!buf) {
        ret = -ENOMEM;
        goto out;
    }

    bool stop = false;
    size_t len = 0;
    while (true) {
        ssize_t n = DO_SYSCALL(read, fd, &buf[len], buf_size - 1 - len);
        if (n == -EINTR) {
            continue;
        } else if (n < 0) {
            ret = n;
            goto out;
        } else if (n == 0) {
            /* EOF; we will process the remainder after the loop */
            break;
        }
        len += n;
        buf[len] = '\0';

        /* Process all finished lines that are in the buffer */
        char* line_end;
        while ((line_end = strchr(buf, '\n')) != NULL) {
            *line_end = '\0';
            ret = callback(buf, arg, &stop);
            if (ret < 0)
                goto out;
            if (stop) {
                ret = 0;
                goto out;
            }

            /* Move remaining part of buffer to beginning (including the final null terminator) */
            len -= line_end + 1 - buf;
            memmove(buf, line_end + 1, len + 1);
        }

        if (len == buf_size - 1) {
            /* The current line might be longer than buffer. Reallocate. */
            size_t new_buf_size = buf_size * 2;
            char* new_buf = malloc(new_buf_size);
            if (!new_buf) {
                ret = -ENOMEM;
                goto out;
            }
            memcpy(new_buf, buf, buf_size);
            free(buf);
            buf_size = new_buf_size;
            buf = new_buf;
        }
    }
    /* Process the rest of buffer; it should not contain any newlines. */
    if (len > 0) {
        ret = callback(buf, arg, &stop);
        if (ret < 0)
            goto out;
        /* ignore `stop`, we've finished either way */
    }
    ret = 0;
out:
    free(buf);
    int close_ret = DO_SYSCALL(close, fd);
    if (close_ret < 0)
        return close_ret;

    return ret;
}
